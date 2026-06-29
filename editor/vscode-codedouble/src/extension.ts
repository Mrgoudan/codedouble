// CodeDouble Capture — records editor interactions (accept / override / reject /
// viewed) into the GLOBAL ~/.codedouble/interactions.jsonl (same schema + path the
// Python CLI defaults to), so the double learns across ALL VS Code windows / repos.
// The "CodeDouble" panel (in the Explorer) shows live, all-repo stats so you can
// SEE it work; analysis/visualization is in Python:  codedouble report
//
// Capture in the editor; analysis/visualization in Python:
//   python3 -m codedouble.cli report   (or: codedouble report)

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

// GLOBAL, machine-wide store by default so the double learns across ALL VS Code
// windows / repos (the original proposal). Matches the Python CLI default
// (~/.codedouble). Override with CODEDOUBLE_LOG (file) or CODEDOUBLE_HOME (dir).
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

const OUTCOME_COLOR: Record<string, string> = {
  override: "#f85149", revert: "#f85149", interrupt: "#f85149",
  confirmed_good: "#3fb950", answered: "#3fb950",
  accepted_silent: "#8b949e", never_viewed: "#6e7681", pending: "#d29922",
};
const NEG = new Set(["override", "revert", "interrupt"]);

function firstLine(text: string): string {
  for (const ln of text.split("\n")) { const t = ln.trim(); if (t) return t.slice(0, 80); }
  return text.trim().slice(0, 80);
}

function relTime(tsSec: number): string {
  const s = Math.max(0, Date.now() / 1000 - tsSec);
  if (!tsSec) return "";
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

// ---- read all-repo stats from the global log (the panel's data) -------------
interface Stats {
  total: number;
  repos: number;
  overrideRate: number;             // % of decisions you overrode/reverted/interrupted
  outcomes: Array<{ k: string; v: number; color: string; pct: number }>;
  recent: Array<{ outcome: string; file: string; rel: string; neg: boolean; pos: boolean }>;
}

function readStats(): Stats {
  const lp = logPath();
  const counts: Record<string, number> = {};
  const repos = new Set<string>();
  const recent: Stats["recent"] = [];
  let total = 0;
  try {
    const lines = fs.readFileSync(lp, "utf8").split("\n");
    for (const line of lines) {
      if (!line.trim()) continue;
      total++;
      try {
        const r = JSON.parse(line);
        counts[r.outcome] = (counts[r.outcome] || 0) + 1;
        if (r.repo) repos.add(r.repo);
      } catch { /* skip malformed */ }
    }
    const tail = lines.filter((l) => l.trim()).slice(-12).reverse();
    for (const line of tail) {
      try {
        const r = JSON.parse(line);
        const f = (r.files && r.files[0]) || r.request || "";
        recent.push({
          outcome: r.outcome || "?", file: String(f).split("/").pop() || String(f),
          rel: relTime(r.ts || 0), neg: NEG.has(r.outcome),
          pos: r.outcome === "confirmed_good" || r.outcome === "answered",
        });
      } catch { /* skip */ }
    }
  } catch { /* no log yet */ }
  const neg = Object.entries(counts).filter(([k]) => NEG.has(k)).reduce((a, [, v]) => a + v, 0);
  const max = Math.max(1, ...Object.values(counts));
  const outcomes = Object.entries(counts)
    .sort((a, b) => b[1] - a[1])
    .map(([k, v]) => ({ k, v, color: OUTCOME_COLOR[k] || "#8b949e", pct: Math.round((100 * v) / max) }));
  return {
    total, repos: repos.size,
    overrideRate: total ? Math.round((100 * neg) / total) : 0,
    outcomes, recent,
  };
}

function updateStatus(): void {
  if (!statusBar) return;
  const on = cfg<boolean>("enabled", true);
  statusBar.text = `$(eye) CodeDouble ${on ? sessionCount : "off"}`;
  statusBar.tooltip = "CodeDouble — interactions captured this session. Click to open the panel.";
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
      const map: Record<string, string> = {
        openReport: "codedouble.openReport", toggle: "codedouble.toggle",
        accept: "codedouble.acceptReviewed", override: "codedouble.markOverride",
        reject: "codedouble.reject",
      };
      if (m?.cmd && map[m.cmd]) vscode.commands.executeCommand(map[m.cmd]);
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
      ...readStats(),
    });
  }

  dispose(): void { if (this.timer) clearInterval(this.timer); }

  private html(): string {
    return `<!doctype html><html><head><meta charset="utf-8"><style>
      :root{color-scheme:light dark}
      body{font-family:var(--vscode-font-family);color:var(--vscode-foreground);padding:10px 12px;font-size:12px}
      .row{display:flex;align-items:center;gap:6px}
      .dot{width:9px;height:9px;border-radius:50%;cursor:pointer}
      .on{background:#3fb950}.off{background:#8b949e}
      .grow{flex:1}
      .sub{opacity:.6;font-size:11px}
      .big{font-size:28px;font-weight:600;line-height:1.1;margin-top:6px}
      h4{margin:14px 0 5px;font-size:10px;text-transform:uppercase;letter-spacing:.06em;opacity:.55}
      .metric{font-size:22px;font-weight:600}
      .bar{height:6px;border-radius:4px;background:var(--vscode-input-background,#2228);overflow:hidden;margin:2px 0 6px}
      .bar>i{display:block;height:100%}
      .obrow{display:flex;align-items:center;gap:8px;font-size:11px;margin-top:6px}
      .obrow .lbl{width:104px}.obrow .n{width:28px;text-align:right;opacity:.8}
      .obrow .track{flex:1}
      ul{list-style:none;padding:0;margin:0}
      li{display:flex;gap:8px;padding:2px 0;font-size:11px;white-space:nowrap;overflow:hidden}
      li .f{flex:1;overflow:hidden;text-overflow:ellipsis}
      li .t{opacity:.5}
      .neg{color:#f85149}.pos{color:#3fb950}.weak{opacity:.6}
      button{width:100%;padding:6px;cursor:pointer;border:none;border-radius:4px;
        background:var(--vscode-button-background);color:var(--vscode-button-foreground);margin-top:10px;font-size:12px}
      .acts{display:flex;gap:6px;margin-top:8px}
      .acts button{margin-top:0;padding:5px 0;font-size:11px;background:var(--vscode-button-secondaryBackground,#3a3d41);color:var(--vscode-button-secondaryForeground,#fff)}
      .muted{opacity:.5;font-size:10px;margin-top:12px;word-break:break-all}
      .empty{opacity:.6;font-size:11px;margin-top:6px;line-height:1.5}
    </style></head><body>
      <div class="row"><span id="dot" class="dot off" title="toggle capture"></span>
        <b id="state">…</b><span class="grow"></span><span class="sub" id="sess"></span></div>

      <div class="big" id="total">0</div>
      <div class="sub" id="totalsub">interactions captured</div>

      <div id="content" style="display:none">
        <h4>Override rate</h4>
        <div class="metric" id="orate">0%</div>
        <div class="sub">overrides + reverts ÷ total — the number the double should drive down as it learns you</div>

        <h4>Outcomes</h4><div id="outcomes"></div>

        <h4>Recent</h4><ul id="recent"></ul>
      </div>

      <div id="empty" class="empty">
        No interactions yet.<br>Paste or accept a 3+ line block in any file — it'll show up here,
        and the affected lines flash briefly. Or run <code>codedouble on</code> in a repo to mine its git history.
      </div>

      <button id="report">Open report ▸</button>
      <div class="acts">
        <button id="accept" title="Mark selection as reviewed & accepted">✓ accept</button>
        <button id="override" title="Mark selection as an override">✎ override</button>
        <button id="reject" title="Mark selection as rejected">✗ reject</button>
      </div>
      <div class="muted" id="store"></div>

      <script>
        const vscode=acquireVsCodeApi();
        const $=id=>document.getElementById(id);
        const send=cmd=>vscode.postMessage({cmd});
        $('report').onclick=()=>send('openReport');
        $('dot').onclick=()=>send('toggle');
        $('accept').onclick=()=>send('accept');
        $('override').onclick=()=>send('override');
        $('reject').onclick=()=>send('reject');
        addEventListener('message',e=>{const s=e.data;
          $('dot').className='dot '+(s.enabled?'on':'off');
          $('state').textContent=s.enabled?'capturing':'paused';
          $('sess').textContent=(s.session||0)+' this session';
          $('total').textContent=s.total||0;
          $('totalsub').textContent='interactions captured'+(s.repos>1?(' · '+s.repos+' repos'):'');
          const has=(s.total||0)>0;
          $('content').style.display=has?'block':'none';
          $('empty').style.display=has?'none':'block';
          $('orate').textContent=(s.overrideRate||0)+'%';
          $('orate').className='metric '+((s.overrideRate||0)>25?'neg':(s.overrideRate<=10?'pos':''));
          $('outcomes').innerHTML=(s.outcomes||[]).map(o=>
            '<div class="obrow"><span class="lbl">'+o.k+'</span>'+
            '<span class="track"><span class="bar"><i style="width:'+o.pct+'%;background:'+o.color+'"></i></span></span>'+
            '<span class="n">'+o.v+'</span></div>').join('');
          $('recent').innerHTML=(s.recent||[]).map(r=>
            '<li class="'+(r.neg?'neg':(r.pos?'pos':'weak'))+'">'+
            '<span class="f">'+r.outcome+' · '+(r.file||'')+'</span><span class="t">'+(r.rel||'')+'</span></li>').join('')
            || '<li class="weak">—</li>';
          $('store').textContent=s.store||'';
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

    vscode.commands.registerCommand("codedouble.acceptReviewed", () => {
      const s = selectionText(); if (!s) return;
      record(s.doc, "confirmed_good", firstLine(s.text), null, s.text, "edit", s.range);
      vscode.window.setStatusBarMessage("CodeDouble: confirmed-good (reviewed)", 2500);
    }),
    vscode.commands.registerCommand("codedouble.markOverride", async () => {
      const s = selectionText(); if (!s) return;
      const from = await vscode.window.showInputBox({ prompt: "What did the AI propose that you changed? (the original X)", placeHolder: "blank if unknown" });
      record(s.doc, "override", firstLine(s.text), from ? firstLine(from) : null, s.text, "edit", s.range);
      vscode.window.setStatusBarMessage("CodeDouble: override (X→Y)", 2500);
    }),
    vscode.commands.registerCommand("codedouble.reject", () => {
      const s = selectionText(); if (!s) return;
      record(s.doc, "interrupt", firstLine(s.text), null, s.text, "edit", s.range);
      vscode.window.setStatusBarMessage("CodeDouble: reject/interrupt", 2500);
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
      vscode.window.setStatusBarMessage(`CodeDouble capture ${next ? "ON" : "OFF"}`, 2500);
    })
  );
}

export function deactivate(): void {
  view?.dispose();
  for (const arr of pending.values()) for (const p of arr) if (p.timer) clearTimeout(p.timer);
}
