// CodeDouble Capture — passively records your interactions with AI changes
// (accept / edit-over / revert / view) into the GLOBAL ~/.codedouble store, and
// shows what your *double* did for you: which of Claude's actions it handled
// silently vs. paused to check with you, and what it inferred you'd want.
// Nothing here asks you to label anything — it learns from what you do.
//   analysis/visualization:  codedouble report

import * as fs from "fs";
import * as os from "os";
import * as path from "path";
import * as vscode from "vscode";

interface Proposal {
  startLine: number;
  endLine: number;
  text: string;
  tsMs: number;
  viewed: boolean;
  timer?: NodeJS.Timeout;
}

const pending = new Map<string, Proposal[]>();
let sessionCount = 0;
let statusBar: vscode.StatusBarItem;
let view: CodeDoubleView | undefined;
let flashDeco: vscode.TextEditorDecorationType;

function cfg<T>(key: string, dflt: T): T {
  return vscode.workspace.getConfiguration("codedouble").get<T>(key, dflt);
}

// GLOBAL, machine-wide store by default (matches the Python CLI default,
// ~/.codedouble). Override with CODEDOUBLE_LOG (file) or CODEDOUBLE_HOME (dir).
function logPath(): string {
  const envLog = process.env.CODEDOUBLE_LOG;
  if (envLog) {
    try { fs.mkdirSync(path.dirname(envLog), { recursive: true }); } catch { /* ignore */ }
    return envLog;
  }
  const home = process.env.CODEDOUBLE_HOME || path.join(os.homedir(), ".codedouble");
  try { fs.mkdirSync(home, { recursive: true }); } catch { /* ignore */ }
  return path.join(home, "interactions.jsonl");
}

const LANG: Record<string, string> = {
  python: "python", typescript: "ts", javascript: "ts", typescriptreact: "ts",
  javascriptreact: "ts", java: "java", go: "go", rust: "rust", c: "c",
  cpp: "cpp", ruby: "ruby", php: "php", csharp: "csharp",
};

function firstLine(text: string): string {
  for (const ln of text.split("\n")) { const t = ln.trim(); if (t) return t.slice(0, 80); }
  return text.trim().slice(0, 80);
}

function relTime(tsSec: number): string {
  if (!tsSec) return "";
  const s = Math.max(0, Date.now() / 1000 - tsSec);
  if (s < 60) return `${Math.floor(s)}s`;
  if (s < 3600) return `${Math.floor(s / 60)}m`;
  if (s < 86400) return `${Math.floor(s / 3600)}h`;
  return `${Math.floor(s / 86400)}d`;
}

function reversibilityOf(doc: vscode.TextDocument, actionKind: string): string {
  const p = doc.uri.fsPath.toLowerCase();
  if (actionKind === "delete" || actionKind === "rename" || p.includes("migrat") || p.includes("schema"))
    return "high";
  return "low";
}

function flash(doc: vscode.TextDocument, startLine: number, endLine: number): void {
  if (!cfg<boolean>("highlight", true)) return;
  const ed = vscode.window.visibleTextEditors.find((e) => e.document.uri.toString() === doc.uri.toString());
  if (!ed) return;
  const last = Math.max(0, Math.min(endLine, doc.lineCount - 1));
  const range = new vscode.Range(Math.max(0, startLine), 0, last, doc.lineAt(last).text.length);
  ed.setDecorations(flashDeco, [range]);
  setTimeout(() => ed.setDecorations(flashDeco, []), 1400);
}

function record(
  doc: vscode.TextDocument, outcome: string, resolution: string,
  correctedFrom: string | null, diff: string, actionKind = "edit",
  flashRange?: { start: number; end: number }
): void {
  const lp = logPath();
  if (!lp) return;
  const rel = vscode.workspace.asRelativePath(doc.uri);
  const wsf = vscode.workspace.getWorkspaceFolder(doc.uri);
  const repo = wsf ? wsf.uri.fsPath : "";
  const rec = {
    ts: Date.now() / 1000, source: "editor", request: `${actionKind} ${rel}`,
    diff: diff.slice(0, 2000), error: "", lang: LANG[doc.languageId] ?? doc.languageId,
    repo, files: [rel], action_kind: actionKind, reversibility: reversibilityOf(doc, actionKind),
    outcome, resolution, corrected_from: correctedFrom, sha: null,
  };
  try {
    fs.appendFileSync(lp, JSON.stringify(rec) + "\n");
  } catch (e) {
    console.error("codedouble: write failed", e);
    return;
  }
  sessionCount++;
  updateStatus();
  view?.refresh();
  if (flashRange) flash(doc, flashRange.start, flashRange.end);
}

// ---- the double's behavior (decisions) + what it's passively watching -------
function decisionsPath(): string {
  return path.join(path.dirname(logPath()), "decisions.jsonl");
}

interface Panel {
  watching: number;   // interactions captured passively (all repos)
  repos: number;
  reactions: Array<{ k: string; v: number }>;                 // your reactions, by kind
  overrides: Array<{ file: string; from: string; to: string; rel: string }>;  // override detail
  seen: number;       // Claude actions the double weighed in on
  handled: number;    // let through without interrupting you
  asked: number;      // paused to check with you
  interceptRate: number;
  recent: Array<{ intent: string; resolution: string; asked: boolean; conf: number; n: number; rel: string }>;
}

function shortIntent(s: string): string {
  s = (s || "").replace(/\s+/g, " ").trim();
  return s.length > 72 ? s.slice(0, 71) + "…" : s;
}

function readPanel(): Panel {
  // passive signal: your reactions to AI changes (interactions.jsonl, all repos)
  let watching = 0; const repos = new Set<string>();
  const rcounts: Record<string, number> = {};
  const overrides: Panel["overrides"] = [];
  try {
    const recs: Array<Record<string, unknown>> = [];
    for (const line of fs.readFileSync(logPath(), "utf8").split("\n")) {
      if (!line.trim()) continue; watching++;
      try {
        const r = JSON.parse(line) as Record<string, unknown>;
        recs.push(r);
        const oc = String(r.outcome || "?");
        rcounts[oc] = (rcounts[oc] || 0) + 1;
        if (r.repo) repos.add(String(r.repo));
      } catch { /* skip */ }
    }
    for (const r of recs.filter((x) => x.outcome === "override").slice(-8).reverse()) {
      const files = (r.files as string[]) || [];
      overrides.push({
        file: String(files[0] || "").split("/").pop() || "",
        from: shortIntent(String(r.corrected_from || "")),
        to: shortIntent(String(r.resolution || "")),
        rel: relTime(Number(r.ts) || 0),
      });
    }
  } catch { /* none yet */ }
  const reactions = Object.entries(rcounts).sort((a, b) => b[1] - a[1]).map(([k, v]) => ({ k, v }));
  // the double's act-vs-ask decisions on your real Claude usage (decisions.jsonl)
  let seen = 0, handled = 0, asked = 0;
  const recent: Panel["recent"] = [];
  try {
    const lines = fs.readFileSync(decisionsPath(), "utf8").split("\n").filter((l) => l.trim());
    seen = lines.length;
    for (const l of lines) { try { const r = JSON.parse(l); if (r.ask) asked++; else handled++; } catch { /* skip */ } }
    for (const l of lines.slice(-14).reverse()) {
      try {
        const r = JSON.parse(l);
        recent.push({
          intent: shortIntent(r.intent || r.tool || r.event || "decision"),
          resolution: shortIntent(r.resolution || ""),
          asked: !!r.ask, conf: r.confidence || 0, n: r.n || 0, rel: relTime(r.ts || 0),
        });
      } catch { /* skip */ }
    }
  } catch { /* gateway not active yet */ }
  return {
    watching, repos: repos.size, reactions, overrides, seen, handled, asked,
    interceptRate: seen ? Math.round((100 * asked) / seen) : 0, recent,
  };
}

function updateStatus(): void {
  if (!statusBar) return;
  const on = cfg<boolean>("enabled", true);
  statusBar.text = `$(eye) CodeDouble ${on ? sessionCount : "off"}`;
  statusBar.tooltip = "CodeDouble — interactions watched this session. Click to open the panel.";
  statusBar.show();
}

function finalizeAccepted(key: string, p: Proposal, doc: vscode.TextDocument): void {
  if (p.timer) clearTimeout(p.timer);
  const arr = pending.get(key);
  if (arr) pending.set(key, arr.filter((x) => x !== p));
  record(doc, p.viewed ? "accepted_silent" : "never_viewed", firstLine(p.text), null,
    p.text, "edit", { start: p.startLine, end: p.endLine });
}

function onChange(e: vscode.TextDocumentChangeEvent): void {
  if (!cfg<boolean>("enabled", true)) return;
  const doc = e.document;
  if (doc.uri.scheme !== "file") return;
  const key = doc.uri.toString();
  const minLines = cfg<number>("minInsertLines", 3);
  const minChars = cfg<number>("minInsertChars", 80);
  const overrideMs = cfg<number>("overrideWindowMs", 120000);
  const settleMs = cfg<number>("settleMs", 90000);

  for (const ch of e.contentChanges) {
    const newlines = (ch.text.match(/\n/g) || []).length;
    const isProposal = ch.text.trim().length > 0 && (newlines + 1 >= minLines || ch.text.length >= minChars);
    if (isProposal) {
      const startLine = ch.range.start.line;
      const proposal: Proposal = { startLine, endLine: startLine + newlines, text: ch.text, tsMs: Date.now(), viewed: false };
      proposal.timer = setTimeout(() => finalizeAccepted(key, proposal, doc), settleMs);
      const arr = pending.get(key) ?? [];
      arr.push(proposal);
      pending.set(key, arr);
    } else {
      const arr = pending.get(key);
      if (!arr || !arr.length) continue;
      const line = ch.range.start.line;
      const now = Date.now();
      for (const p of [...arr]) {
        if (now - p.tsMs > overrideMs) continue;
        if (line >= p.startLine - 2 && line <= p.endLine + 5) {
          if (p.timer) clearTimeout(p.timer);
          pending.set(key, arr.filter((x) => x !== p));
          const corrected = doc.lineAt(Math.min(line, doc.lineCount - 1)).text.trim().slice(0, 80);
          record(doc, "override", corrected || firstLine(ch.text), firstLine(p.text), ch.text, "edit", { start: line, end: line });
          break;
        }
      }
    }
  }
}

function onVisible(e: vscode.TextEditorVisibleRangesChangeEvent): void {
  const arr = pending.get(e.textEditor.document.uri.toString());
  if (!arr) return;
  for (const p of arr)
    for (const vr of e.visibleRanges)
      if (vr.start.line <= p.endLine && vr.end.line >= p.startLine) p.viewed = true;
}

function selectionText(): { doc: vscode.TextDocument; text: string; range: { start: number; end: number } } | undefined {
  const ed = vscode.window.activeTextEditor;
  if (!ed) return undefined;
  const sel = ed.selection;
  const text = sel.isEmpty ? ed.document.lineAt(sel.active.line).text : ed.document.getText(sel);
  return { doc: ed.document, text, range: { start: sel.start.line, end: sel.end.line } };
}

// ----------------------------- panel ---------------------------------------
class CodeDoubleView implements vscode.WebviewViewProvider {
  private wv?: vscode.WebviewView;
  private timer?: NodeJS.Timeout;

  resolveWebviewView(v: vscode.WebviewView): void {
    this.wv = v;
    v.webview.options = { enableScripts: true };
    v.webview.html = this.html();
    v.webview.onDidReceiveMessage((m) => {
      if (m?.cmd === "openReport") vscode.commands.executeCommand("codedouble.openReport");
      if (m?.cmd === "toggle") vscode.commands.executeCommand("codedouble.toggle");
    });
    v.onDidChangeVisibility(() => { if (v.visible) this.refresh(); });
    this.refresh();
    if (this.timer) clearInterval(this.timer);
    this.timer = setInterval(() => { if (this.wv?.visible) this.refresh(); }, 15000);
  }

  refresh(): void {
    if (!this.wv) return;
    this.wv.webview.postMessage({
      enabled: cfg<boolean>("enabled", true),
      session: sessionCount,
      store: logPath(),
      ...readPanel(),
    });
  }

  dispose(): void { if (this.timer) clearInterval(this.timer); }

  private html(): string {
    return `<!doctype html><html><head><meta charset="utf-8"><style>
      body{font-family:var(--vscode-font-family);color:var(--vscode-foreground);padding:10px 12px;font-size:12px}
      .row{display:flex;align-items:center;gap:6px}
      .dot{width:9px;height:9px;border-radius:50%;cursor:pointer}.on{background:#3fb950}.off{background:#8b949e}
      .grow{flex:1}
      .sub{opacity:.62;font-size:11px;line-height:1.45;margin-top:5px}
      .lead{font-weight:600;margin-top:12px}
      .cards{display:flex;gap:8px;margin-top:7px}
      .card{flex:1;background:var(--vscode-input-background,#80808022);border-radius:7px;padding:9px 6px;text-align:center}
      .num{font-size:25px;font-weight:700;line-height:1}.num.pos{color:#3fb950}.num.amber{color:#d29922}
      .clbl{font-size:10px;opacity:.72;margin-top:5px;line-height:1.25}
      h4{margin:15px 0 4px;font-size:10px;text-transform:uppercase;letter-spacing:.06em;opacity:.55}
      .metric{font-size:23px;font-weight:700}
      ul{list-style:none;padding:0;margin:0}
      li{padding:6px 0;border-top:1px solid var(--vscode-input-background,#80808026)}
      li:first-child{border-top:none}
      .it{display:flex;align-items:center;gap:6px}
      .iv{flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
      .inf{opacity:.62;font-size:11px;margin-top:2px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
      .tag{font-size:9px;text-transform:uppercase;letter-spacing:.03em;padding:1px 6px;border-radius:9px;color:#fff;flex:none}
      .tag.green{background:#3fb950}.tag.amber{background:#d29922}
      .t{opacity:.7}
      .rx{margin:7px 0}.rxh{display:flex;align-items:center;gap:6px}
      .rdot{width:8px;height:8px;border-radius:50%;flex:none}
      .rk{flex:1}.rn{font-weight:600}
      .rd{opacity:.6;font-size:10.5px;margin:1px 0 0 14px;line-height:1.35}
      .hint{opacity:.55;font-weight:400;text-transform:none;letter-spacing:0}
      button{width:100%;padding:6px;cursor:pointer;border:none;border-radius:4px;background:var(--vscode-button-background);color:var(--vscode-button-foreground);margin-top:14px}
      .muted{opacity:.5;font-size:10px;margin-top:10px;word-break:break-all}
      .empty{opacity:.72;font-size:11px;margin-top:8px;line-height:1.55}
    </style></head><body>
      <div class="row"><span id="dot" class="dot off" title="pause / resume watching"></span>
        <b id="state">…</b><span class="grow"></span><span class="sub" id="sess"></span></div>

      <div class="lead">What your double did for you</div>
      <div class="cards">
        <div class="card"><div class="num pos" id="handled">0</div><div class="clbl">handled<br>without asking</div></div>
        <div class="card"><div class="num amber" id="asked">0</div><div class="clbl">checked<br>with you</div></div>
      </div>
      <div class="sub">When Claude changes code, your double decides whether to <b>let it through</b> or
        <b>pause and ask you</b> — learned from how you've accepted, edited, or reverted changes before.
        It infers from what you do; you never label anything.</div>

      <div id="react">
        <h4>Your reactions to AI changes</h4>
        <div id="reactions"></div>
        <h4>Recent overrides <span class="hint">— where you corrected the AI</span></h4>
        <ul id="overrides"></ul>
      </div>

      <div id="has">
        <h4>How often it interrupts you</h4>
        <div class="metric" id="rate">0%</div>
        <div class="sub">share of Claude's actions it paused on. This should fall as it learns your taste.</div>

        <h4>Recently inferred</h4>
        <ul id="recent"></ul>
      </div>

      <div id="none" class="empty">
        Your double is watching, but hasn't weighed in on a Claude action yet. As you use Claude
        (with the gateway on), every edit/command it sees shows up here as <b>handled</b> or
        <b>checked</b>, along with what it inferred you'd want.
      </div>

      <button id="report">Open report ▸</button>
      <div class="muted" id="foot"></div>

      <script>
        const vscode=acquireVsCodeApi();
        const $=id=>document.getElementById(id);
        $('report').onclick=()=>vscode.postMessage({cmd:'openReport'});
        $('dot').onclick=()=>vscode.postMessage({cmd:'toggle'});
        addEventListener('message',e=>{const s=e.data;
          $('dot').className='dot '+(s.enabled?'on':'off');
          $('state').textContent=s.enabled?'watching':'paused';
          $('sess').textContent=(s.session||0)+' captured this session';
          $('handled').textContent=s.handled||0;
          $('asked').textContent=s.asked||0;
          const seen=s.seen||0;
          $('has').style.display=seen?'block':'none';
          $('none').style.display=seen?'none':'block';
          $('rate').textContent=(s.interceptRate||0)+'%';
          $('rate').className='metric '+((s.interceptRate||0)>30?'amber':'pos');
          $('recent').innerHTML=(s.recent||[]).map(r=>{
            const tag=r.asked?'<span class="tag amber">checked</span>':'<span class="tag green">handled</span>';
            const inf=r.resolution?('you usually \\u2192 '+r.resolution):(r.n?'weak precedent':'no precedent yet');
            const meta=(r.rel?(' \\u00b7 '+r.rel):'')+(r.conf?(' \\u00b7 '+Math.round(r.conf*100)+'%'):'');
            return '<li><div class="it">'+tag+'<span class="iv">'+r.intent+'</span></div>'+
                   '<div class="inf">'+inf+'<span class="t">'+meta+'</span></div></li>';
          }).join('')||'<li class="inf">—</li>';
          const RC={override:'#f85149',revert:'#f85149',interrupt:'#f85149',confirmed_good:'#3fb950',answered:'#3fb950',accepted_silent:'#8b949e',never_viewed:'#6e7681',pending:'#d29922'};
          const RD={override:"you edited the AI's change — your strongest preference signal",
            revert:"you undid the AI's change entirely",interrupt:"you stopped / rejected the change",
            accepted_silent:"you kept it unchanged after seeing it — tacit approval",
            never_viewed:"it settled before you scrolled to it — a weak signal",
            confirmed_good:"you explicitly marked it good",answered:"the double asked and you answered",
            pending:"awaiting your reaction"};
          $('react').style.display=(s.watching?'block':'none');
          $('reactions').innerHTML=(s.reactions||[]).map(o=>
            '<div class="rx"><div class="rxh"><span class="rdot" style="background:'+(RC[o.k]||'#888')+'"></span>'+
            '<span class="rk">'+o.k+'</span><span class="rn">'+o.v+'</span></div>'+
            '<div class="rd">'+(RD[o.k]||'')+'</div></div>').join('');
          $('overrides').innerHTML=(s.overrides||[]).map(o=>
            '<li><div class="it"><span class="iv">'+(o.file||'?')+'</span><span class="t">'+(o.rel||'')+'</span></div>'+
            '<div class="inf">'+(o.from?('AI: '+o.from+' \\u2192 you: '+(o.to||'?')):('you wrote: '+(o.to||'?')))+'</div></li>').join('')
            ||'<li class="inf">no overrides yet — when you edit an AI change, the before\\u2192after appears here</li>';
          $('foot').textContent='Watching '+(s.watching||0)+' interactions'+
            (s.repos>1?(' across '+s.repos+' repos'):'')+' — passively, no buttons. '+(s.store||'~/.codedouble');
        });
      </script></body></html>`;
  }
}

export function activate(context: vscode.ExtensionContext): void {
  flashDeco = vscode.window.createTextEditorDecorationType({
    isWholeLine: true,
    backgroundColor: new vscode.ThemeColor("editor.findMatchHighlightBackground"),
    overviewRulerColor: new vscode.ThemeColor("editor.findMatchHighlightBackground"),
    overviewRulerLane: vscode.OverviewRulerLane.Center,
  });
  statusBar = vscode.window.createStatusBarItem(vscode.StatusBarAlignment.Right, 100);
  statusBar.command = "workbench.view.explorer";
  updateStatus();
  view = new CodeDoubleView();

  context.subscriptions.push(
    flashDeco, statusBar, { dispose: () => view?.dispose() },
    vscode.window.registerWebviewViewProvider("cdCaptureView", view),
    vscode.workspace.onDidChangeTextDocument(onChange),
    vscode.window.onDidChangeTextEditorVisibleRanges(onVisible),

    // Power-user overrides (palette only) — capture is otherwise fully passive.
    vscode.commands.registerCommand("codedouble.acceptReviewed", () => {
      const s = selectionText(); if (!s) return;
      record(s.doc, "confirmed_good", firstLine(s.text), null, s.text, "edit", s.range);
      vscode.window.setStatusBarMessage("CodeDouble: marked accepted", 2500);
    }),
    vscode.commands.registerCommand("codedouble.markOverride", async () => {
      const s = selectionText(); if (!s) return;
      const from = await vscode.window.showInputBox({ prompt: "What did the AI propose that you changed? (the original)", placeHolder: "blank if unknown" });
      record(s.doc, "override", firstLine(s.text), from ? firstLine(from) : null, s.text, "edit", s.range);
      vscode.window.setStatusBarMessage("CodeDouble: marked override", 2500);
    }),
    vscode.commands.registerCommand("codedouble.reject", () => {
      const s = selectionText(); if (!s) return;
      record(s.doc, "interrupt", firstLine(s.text), null, s.text, "edit", s.range);
      vscode.window.setStatusBarMessage("CodeDouble: marked rejected", 2500);
    }),
    vscode.commands.registerCommand("codedouble.openReport", () => {
      const term = vscode.window.createTerminal("codedouble");
      term.show();
      term.sendText("python3 -m codedouble.cli report");
    }),
    vscode.commands.registerCommand("codedouble.toggle", async () => {
      const conf = vscode.workspace.getConfiguration("codedouble");
      const next = !conf.get<boolean>("enabled", true);
      await conf.update("enabled", next, vscode.ConfigurationTarget.Global);
      updateStatus(); view?.refresh();
      vscode.window.setStatusBarMessage(`CodeDouble ${next ? "watching" : "paused"}`, 2500);
    })
  );
}

export function deactivate(): void {
  view?.dispose();
  for (const arr of pending.values()) for (const p of arr) if (p.timer) clearTimeout(p.timer);
}
