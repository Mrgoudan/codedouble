// CodeDouble Capture — passively records your interactions with AI changes
// (accept / edit-over / revert / view) into the GLOBAL ~/.codedouble store, and
// shows what your *double* did for you: which of Claude's actions it handled
// silently vs. paused to check with you, and what it inferred you'd want.
// Nothing here asks you to label anything — it learns from what you do.
//   analysis/visualization:  codedouble report

import * as cp from "child_process";
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
let lastActivity = Date.now();
let lastReflect = 0;
const MATTERS = new Set(["override", "revert", "interrupt"]);   // high-signal moments
let sinceReflect = 0;

// Reflect/summarize when something that MATTERS happens, or every N interactions —
// event-driven, not on an idle clock. Coalesced so bursts don't thrash.
function maybeReflect(): void {
  const now = Date.now();
  if (now - lastReflect < 15000) return;
  lastReflect = now; sinceReflect = 0;
  cp.exec("python3 -m codedouble.cli reflect --quiet; python3 -m codedouble.cli summarize --quiet",
    { timeout: 90000 }, () => view?.refresh());
}
let statusBar: vscode.StatusBarItem;
let view: CodeDoubleView | undefined;
let flashDeco: vscode.TextEditorDecorationType;

function cfg<T>(key: string, dflt: T): T {
  return vscode.workspace.getConfiguration("codedouble").get<T>(key, dflt);
}

// GLOBAL, machine-wide store by default (matches the Python CLI default,
// ~/.codedouble). Override with CODEDOUBLE_LOG (file) or CODEDOUBLE_HOME (dir).
function repoRoot(start: string): string {
  let cur = path.resolve(start);
  for (;;) {
    if (fs.existsSync(path.join(cur, ".git"))) return cur;
    const parent = path.dirname(cur);
    if (parent === cur) return path.resolve(start);
    cur = parent;
  }
}

function scopeState(): { root: string; off: boolean; byMarker: boolean } {
  const ws = vscode.workspace.workspaceFolders?.[0]?.uri.fsPath;
  if (!ws) return { root: "", off: false, byMarker: false };
  const root = repoRoot(ws);
  if (fs.existsSync(path.join(root, ".codedouble.off"))) return { root, off: true, byMarker: true };
  try {
    const cfg = JSON.parse(fs.readFileSync(path.join(path.dirname(logPath()), "scope.json"), "utf8"));
    for (const o of (cfg.off || [])) {
      const p = String(o).replace(/\/+$/, "");
      if (root === p || root.startsWith(p + "/")) return { root, off: true, byMarker: false };
    }
  } catch { /* no config -> on */ }
  return { root, off: false, byMarker: false };
}

function scopeToggle(): void {
  const st = scopeState();
  if (!st.root || st.byMarker) return;          // marker-controlled: managed in the repo, not here
  const p = path.join(path.dirname(logPath()), "scope.json");
  let cfg: { off?: string[] } = {};
  try { cfg = JSON.parse(fs.readFileSync(p, "utf8")); } catch { /* fresh */ }
  const off = (cfg.off || []).map((x) => String(x).replace(/\/+$/, ""));
  cfg.off = st.off ? off.filter((x) => !(st.root === x || st.root.startsWith(x + "/")))
                   : off.concat([st.root]);
  fs.writeFileSync(p, JSON.stringify(cfg, null, 2));
}

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

// Don't capture machine-written files (logs/data) or build/vendor dirs — external
// processes (e.g. an AI loop appending to loop.log) are not you editing AI code.
const IGNORE_EXT = new Set([
  ".log", ".jsonl", ".tmp", ".temp", ".lock", ".map", ".csv", ".tsv",
  ".out", ".pyc", ".bin", ".cache", ".pid", ".sqlite", ".db",
]);
const IGNORE_DIR = [
  "/node_modules/", "/.git/", "/dist/", "/build/", "/out/", "/.codedouble/",
  "/__pycache__/", "/.venv/", "/venv/", "/.next/", "/target/", "/.cache/", "/logs/",
];

function isCapturable(doc: vscode.TextDocument): boolean {
  if (doc.uri.scheme !== "file") return false;
  const p = doc.uri.fsPath.toLowerCase();
  if (IGNORE_DIR.some((d) => p.includes(d))) return false;
  if (IGNORE_EXT.has(path.extname(p))) return false;
  if (p.endsWith(".log") || /\.log\.\d+$/.test(p)) return false;   // foo.log, foo.log.1
  return true;
}

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

function scopedOff(doc: vscode.TextDocument): boolean {
  // "off" must mean fully dark: the editor-capture path honors scope like the hooks do
  const ws = vscode.workspace.getWorkspaceFolder(doc.uri)?.uri.fsPath;
  if (!ws) return false;
  const root = repoRoot(ws);
  if (fs.existsSync(path.join(root, ".codedouble.off"))) return true;
  try {
    const cfg = JSON.parse(fs.readFileSync(path.join(path.dirname(logPath()), "scope.json"), "utf8"));
    return (cfg.off || []).some((o: string) => {
      const p = String(o).replace(/\/+$/, "");
      return root === p || root.startsWith(p + "/");
    });
  } catch { return false; }
}

function record(
  doc: vscode.TextDocument, outcome: string, resolution: string,
  correctedFrom: string | null, diff: string, actionKind = "edit",
  flashRange?: { start: number; end: number }
): void {
  const lp = logPath();
  if (!lp) return;
  if (scopedOff(doc)) return;                    // folder opted out -> capture nothing
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
  sinceReflect++;
  lastActivity = Date.now();
  if (MATTERS.has(outcome) || sinceReflect >= 15) maybeReflect();   // when it matters / every N
  updateStatus();
  view?.refresh();
  if (flashRange) flash(doc, flashRange.start, flashRange.end);
}

// ---- the double's behavior (decisions) + what it's passively watching -------
function decisionsPath(): string {
  return path.join(path.dirname(logPath()), "decisions.jsonl");
}

interface Panel {
  goal: string;         // this session's overall goal (from its anchors)
  constraints: string[]; // the session's steering anchors (same block injected to the AI)
  decisions: string[];
  todos: string[];
  handled: number;      // let through, on your behalf
  rejected: number;     // sent back to the AI to redo
  sentBack: Array<{ intent: string; before: string; why: string; rel: string }>;  // AI tried → why sent back
}

function cleanPrompt(p: unknown): string {
  // strip injected IDE/system context so it never becomes the session goal
  return String(p || "")
    .replace(/<(ide_selection|ide_opened_file|system-reminder|system)\b[^>]*>[\s\S]*?(<\/\1>|$)/gi, " ")
    .replace(/\s+/g, " ").trim();
}

function shortIntent(s: string, n = 72): string {
  s = (s || "").replace(/\s+/g, " ").trim();
  return s.length > n ? s.slice(0, n - 1) + "…" : s;
}

function readPanel(): Panel {
  // the double's verdicts on your real Claude usage (decisions.jsonl)
  // verdict: inject/allow = handled, ask = checked, deny = rejected (shadow falls back to would-ask)
  // the double acts toward the AI: "sent back" (deny / legacy ask) vs "handled" (let through)
  const tagOf = (r: Record<string, unknown>): string => {
    const v = r.verdict as string | undefined;
    if (v === "deny" || v === "ask") return "sent back";
    if (v === undefined || v === "shadow") return r.ask ? "sent back" : "handled";
    return "handled"; // inject | allow | watch | answered
  };
  let handled = 0, rejected = 0, goal = "";
  const constraints: string[] = [], decisions: string[] = [], todos: string[] = [];
  const sentBack: Panel["sentBack"] = [];
  try {
    const recs = fs.readFileSync(decisionsPath(), "utf8").split("\n")
      .filter((l) => l.trim())
      .map((l) => { try { return JSON.parse(l) as Record<string, unknown>; } catch { return null; } })
      .filter((r): r is Record<string, unknown> => !!r);
    // PAIR the panel with the AI session running in THIS window's folder: only
    // consider decisions whose cwd is inside an open workspace folder (the global
    // decisions log mixes every project's sessions). Fall back to all if none match.
    const folders = (vscode.workspace.workspaceFolders || []).map((f) => f.uri.fsPath);
    const here = (r: Record<string, unknown>): boolean => {
      const c = String(r.cwd || "");
      return !c ? false : (!folders.length || folders.some((f) => c === f || c.startsWith(f + "/")));
    };
    const acts = recs.filter((r) => r.verdict !== "bypassed");   // meta-records: not actions
    const local = acts.filter(here);
    // only fall back to the global log when NO record carries a cwd yet (old data);
    // if this folder simply has no session, show nothing rather than leak another's.
    const anyCwd = acts.some((r) => !!r.cwd);
    const pool = (folders.length && anyCwd) ? local : acts;
    // "this session" = the most recent session in this window's folder
    let sid = "";
    for (let i = pool.length - 1; i >= 0; i--) {
      const v = String(pool[i].session_id || "");
      if (v) { sid = v; break; }
    }
    const scoped = sid ? pool.filter((r) => String(r.session_id || "") === sid) : pool;
    // GOAL persists per session: its consolidated goal, else a heuristic from its OWN
    // first prompt (always stored in its notes) — so opening ANY of the folder's
    // sessions shows that session's goal, never another's, even before consolidation.
    const sdir = path.join(path.dirname(logPath()), "sessions");
    const strs = (v: unknown): string[] =>
      Array.isArray(v) ? v.map((x) => String(x)).filter(Boolean).slice(0, 8) : [];
    try {
      if (sid) {
        const a = JSON.parse(fs.readFileSync(path.join(sdir, sid + ".anchors.json"), "utf8")) as Record<string, unknown>;
        goal = String(a.goal || "");
        constraints.push(...strs(a.constraints)); decisions.push(...strs(a.decisions)); todos.push(...strs(a.todos));
      }
    } catch { /* not consolidated yet */ }
    if (!goal && sid) {
      try {
        for (const l of fs.readFileSync(path.join(sdir, sid + ".jsonl"), "utf8").split("\n")) {
          if (!l.trim()) continue;
          const p = cleanPrompt((JSON.parse(l) as Record<string, unknown>).prompt);
          if (p.length >= 12) {
            goal = p.split(/(?<=[.!?])\s/)[0].split(/\s+at\s+[~/]/)[0].replace(/\s*\([^)]*\)/g, "").trim().slice(0, 140);
            break;
          }
        }
      } catch { /* no notes */ }
    }
    for (const r of scoped) { if (tagOf(r) === "sent back") rejected++; else handled++; }
    // every send-back THIS session, newest first — the AI's try → why it was sent back
    for (let i = scoped.length - 1; i >= 0 && sentBack.length < 25; i--) {
      const r = scoped[i];
      if (r.verdict === "deny" && (r.before || r.after || r.reason)) {
        sentBack.push({
          intent: shortIntent(String(r.target || r.intent || r.tool || "")),
          before: shortIntent(String(r.before || ""), 300),
          why: shortIntent(String(r.reason || r.after || "redo"), 2000),  // the WHY is the point — show it whole
          rel: relTime(Number(r.ts) || 0),
        });
      }
    }
  } catch { /* gateway not active yet */ }
  return { goal, constraints, decisions, todos, handled, rejected, sentBack };
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
  if (p.viewed) {
    record(doc, "accepted_silent", firstLine(p.text), null, p.text, "edit",
      { start: p.startLine, end: p.endLine });
    return;
  }
  // Not viewed: only record if it's a clearly large block (AI-like). Small unviewed
  // inserts are usually your own typing/paste — skipping them cuts never_viewed noise.
  const lines = p.endLine - p.startLine + 1;
  if (lines >= Math.max(8, cfg<number>("minInsertLines", 3) * 3)) {
    record(doc, "never_viewed", firstLine(p.text), null, p.text, "edit",
      { start: p.startLine, end: p.endLine });
  }
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
      if (m?.cmd === "scopeToggle") { scopeToggle(); this.refresh(); }
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
      scope: scopeState(),
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
      .num{font-size:23px;font-weight:700;line-height:1}.num.pos{color:#3fb950}.num.amber{color:#d29922}.num.red{color:#f85149}
      .clbl{font-size:10px;opacity:.72;margin-top:5px;line-height:1.25}
      h4{margin:15px 0 4px;font-size:10px;text-transform:uppercase;letter-spacing:.06em;opacity:.55}
      ul{list-style:none;padding:0;margin:0}
      li{padding:6px 0;border-top:1px solid var(--vscode-input-background,#80808026)}
      li:first-child{border-top:none}
      .it{display:flex;align-items:center;gap:6px}
      .iv{flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
      .inf{opacity:.62;font-size:11px;margin-top:2px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
      .tag{font-size:9px;text-transform:uppercase;letter-spacing:.03em;padding:1px 6px;border-radius:9px;color:#fff;flex:none}
      .tag.green{background:#3fb950}.tag.amber{background:#d29922}.tag.red{background:#f85149}
      .t{opacity:.7}
      #anchors h4{margin:11px 0 3px}
      #anchors li{padding:3px 0;border-top:none;opacity:.85;font-size:11px;line-height:1.4;white-space:normal;word-break:break-word}
      .hint{opacity:.55;font-weight:400;text-transform:none;letter-spacing:0}
      .ba{margin:3px 0 2px 2px;font-size:11px;line-height:1.45;white-space:pre-wrap;word-break:break-word}
      .bf{color:#f85149}.af{color:#3fb950}
      button{width:100%;padding:6px;cursor:pointer;border:none;border-radius:4px;background:var(--vscode-button-background);color:var(--vscode-button-foreground);margin-top:14px}
      .muted{opacity:.5;font-size:10px;margin-top:10px;word-break:break-all}
      .empty{opacity:.72;font-size:11px;margin-top:8px;line-height:1.55}
      .goal{margin-top:12px;padding:8px 10px;border-radius:7px;background:var(--vscode-input-background,#80808022);font-size:12px;line-height:1.4}
      .scope{display:flex;align-items:center;gap:8px;margin-top:8px;padding:6px 10px;border-radius:7px;background:var(--vscode-input-background,#80808022)}
      .scopepill{font-size:10px;font-weight:700;letter-spacing:.04em;padding:2px 8px;border-radius:9px;color:#fff}
      .scopepill.on{background:#3fb950}.scopepill.off{background:#8b949e}
      .scopefolder{flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;font-size:11.5px}
      .scopebtn{width:auto;margin:0;padding:3px 10px;font-size:11px}
      .dimmed{opacity:.35;pointer-events:none}
      .offnote{margin-top:10px;font-size:11.5px;line-height:1.5;opacity:.8}
      .glbl{font-size:9px;text-transform:uppercase;letter-spacing:.06em;opacity:.55;margin-right:6px}
    </style></head><body>
      <div class="row"><span id="dot" class="dot off" title="pause / resume watching"></span>
        <b id="state">…</b><span class="grow"></span><span class="sub" id="sess"></span></div>

      <div class="scope"><span id="scopepill" class="scopepill on">ON</span>
        <span class="scopefolder" id="scopefolder"></span>
        <button class="scopebtn" id="scopebtn">Turn off here</button></div>
      <div id="offnote" class="offnote" style="display:none">The double is <b>off for this folder</b> —
        nothing is captured, remembered, injected, or gated here. Everything below is history from
        before it was turned off.</div>

      <div id="content">
      <div id="goalbox" class="goal"><span class="glbl">this session’s goal</span><span id="goal"></span></div>
      <div id="anchors"></div>

      <div class="lead">Sent back to redo <span class="hint">— this session</span></div>
      <div class="cards">
        <div class="card"><div class="num red" id="rejected">0</div><div class="clbl">sent back<br>to the AI</div></div>
        <div class="card"><div class="num pos" id="handled">0</div><div class="clbl">let<br>through</div></div>
      </div>

      <div id="sb">
        <h4>What the AI tried <span class="hint">→ why codedouble sent it back</span></h4>
        <ul id="sentback"></ul>
      </div>

      <div id="none" class="empty">
        No send-backs in this session yet. When codedouble sends an AI change back to redo,
        the AI's version and the reason it was rejected show up here. <i>(Requires enforce mode.)</i>
      </div>

      </div>

      <button id="report">Open report ▸</button>
      <div class="muted" id="foot"></div>

      <script>
        const vscode=acquireVsCodeApi();
        const $=id=>document.getElementById(id);
        $('report').onclick=()=>vscode.postMessage({cmd:'openReport'});
        $('dot').onclick=()=>vscode.postMessage({cmd:'toggle'});
        $('scopebtn').onclick=()=>vscode.postMessage({cmd:'scopeToggle'});
        const esc=t=>String(t==null?'':t).replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
        addEventListener('message',e=>{const s=e.data;
          $('dot').className='dot '+(s.enabled?'on':'off');
          $('state').textContent=s.enabled?'watching':'paused';
          $('sess').textContent='this session';
          const sc=s.scope||{};
          const base=(sc.root||'').split('/').pop()||'(no folder)';
          $('scopefolder').textContent=base;
          $('scopepill').textContent=sc.off?'OFF':'ON';
          $('scopepill').className='scopepill '+(sc.off?'off':'on');
          $('scopebtn').textContent=sc.off?'Turn on here':'Turn off here';
          $('scopebtn').style.display=(sc.root&&!sc.byMarker)?'inline-block':'none';
          if(sc.byMarker){$('scopefolder').textContent=base+'  (.codedouble.off in repo)';}
          $('offnote').style.display=sc.off?'block':'none';
          $('content').className=sc.off?'dimmed':'';
          $('goal').textContent=s.goal||'';
          $('goalbox').style.display=s.goal?'block':'none';
          const sec=(title,items)=>(items&&items.length)?('<h4>'+title+'</h4><ul>'+items.map(x=>'<li>'+esc(x)+'</li>').join('')+'</ul>'):'';
          $('anchors').innerHTML=sec('Constraints',s.constraints)+sec('Decisions made',s.decisions)+sec('To do',s.todos);
          $('rejected').textContent=s.rejected||0;
          $('handled').textContent=s.handled||0;
          const sb=s.sentBack||[];
          $('sb').style.display=sb.length?'block':'none';
          $('none').style.display=sb.length?'none':'block';
          $('sentback').innerHTML=sb.map(o=>
            '<li><div class="it"><span class="tag red">sent back</span><span class="iv">'+esc(o.intent||'?')+'</span><span class="t">'+esc(o.rel||'')+'</span></div>'+
            '<div class="ba"><span class="bf">AI: '+esc(o.before||'?')+'</span><br><span class="af">→ '+esc(o.why||'redo')+'</span></div></li>').join('');
          $('foot').textContent=(s.store||'~/.codedouble');
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
