// CodeDouble Capture — records editor interactions (accept / override / reject /
// viewed) into the GLOBAL ~/.codedouble/interactions.jsonl (same schema + path the
// Python CLI defaults to), so the double learns across ALL VS Code windows / repos,
// with a live panel and a brief highlight so you can SEE it work.
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
const recent: Array<{ outcome: string; file: string }> = [];
const tally: Record<string, number> = {};
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

function firstLine(text: string): string {
  for (const ln of text.split("\n")) { const t = ln.trim(); if (t) return t.slice(0, 80); }
  return text.trim().slice(0, 80);
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
  tally[outcome] = (tally[outcome] ?? 0) + 1;
  recent.unshift({ outcome, file: rel });
  if (recent.length > 15) recent.pop();
  updateStatus();
  view?.refresh();
  if (flashRange) flash(doc, flashRange.start, flashRange.end);
}

function updateStatus(): void {
  if (!statusBar) return;
  const on = cfg<boolean>("enabled", true);
  statusBar.text = `$(eye) CodeDouble ${on ? sessionCount : "off"}`;
  statusBar.tooltip = "Interactions captured this session — click to open the panel";
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

// ----------------------------- sidebar -------------------------------------
class CodeDoubleView implements vscode.WebviewViewProvider {
  private wv?: vscode.WebviewView;
  resolveWebviewView(v: vscode.WebviewView): void {
    this.wv = v;
    v.webview.options = { enableScripts: true };
    v.webview.html = this.html();
    v.webview.onDidReceiveMessage((m) => {
      if (m?.cmd === "openReport") vscode.commands.executeCommand("codedouble.openReport");
      if (m?.cmd === "toggle") vscode.commands.executeCommand("codedouble.toggle");
    });
    this.refresh();
  }
  refresh(): void {
    this.wv?.webview.postMessage({
      enabled: cfg<boolean>("enabled", true), count: sessionCount, tally, recent,
    });
  }
  private html(): string {
    return `<!doctype html><html><head><meta charset="utf-8"><style>
      body{font-family:var(--vscode-font-family);color:var(--vscode-foreground);padding:8px 10px}
      .dot{display:inline-block;width:8px;height:8px;border-radius:50%;margin-right:6px}
      .on{background:#5cb85c}.off{background:#888}
      h4{margin:10px 0 4px;font-size:11px;text-transform:uppercase;opacity:.7;letter-spacing:.04em}
      .chip{display:inline-block;padding:1px 7px;border-radius:9px;font-size:11px;margin:2px 4px 2px 0;color:#fff}
      ul{list-style:none;padding:0;margin:0;font-size:12px}
      li{padding:2px 0;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
      .neg{color:#d9534f}.pos{color:#5cb85c}.weak{opacity:.6}
      button{margin-top:10px;width:100%;padding:5px;cursor:pointer;
        background:var(--vscode-button-background);color:var(--vscode-button-foreground);border:none;border-radius:3px}
      .muted{opacity:.6;font-size:11px;margin-top:6px}
    </style></head><body>
      <div><span id="dot" class="dot off"></span><b id="state">…</b></div>
      <div id="count" style="font-size:22px;margin:6px 0 2px">0</div>
      <div class="muted">interactions captured this session</div>
      <h4>Outcomes</h4><div id="tally"></div>
      <h4>Recent</h4><ul id="recent"></ul>
      <button id="report">Open report ▸</button>
      <div class="muted">Global store: ~/.codedouble/interactions.jsonl (all windows)</div>
      <script>
        const COLORS={override:'#d9534f',revert:'#d9534f',interrupt:'#d9534f',
          confirmed_good:'#5cb85c',answered:'#5cb85c',accepted_silent:'#888',never_viewed:'#bbb'};
        const NEG=new Set(['override','revert','interrupt']);
        const vscode=acquireVsCodeApi();
        document.getElementById('report').onclick=()=>vscode.postMessage({cmd:'openReport'});
        document.getElementById('dot').onclick=()=>vscode.postMessage({cmd:'toggle'});
        addEventListener('message',e=>{const s=e.data;
          document.getElementById('dot').className='dot '+(s.enabled?'on':'off');
          document.getElementById('state').textContent=s.enabled?'capturing':'paused';
          document.getElementById('count').textContent=s.count;
          document.getElementById('tally').innerHTML=Object.keys(s.tally).length?
            Object.entries(s.tally).map(([k,v])=>'<span class="chip" style="background:'+(COLORS[k]||'#888')+'">'+k+' '+v+'</span>').join(''):'<span class="muted">none yet — paste 3+ lines to test</span>';
          document.getElementById('recent').innerHTML=s.recent.map(r=>'<li class="'+(NEG.has(r.outcome)?'neg':(r.outcome.startsWith('confirmed')||r.outcome==='answered'?'pos':'weak'))+'">'+r.outcome+' · '+r.file+'</li>').join('');
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
  statusBar.command = "codedouble.openReport";
  updateStatus();
  view = new CodeDoubleView();

  context.subscriptions.push(
    flashDeco, statusBar,
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
      await conf.update("enabled", next, vscode.ConfigurationTarget.Workspace);
      updateStatus(); view?.refresh();
      vscode.window.showInformationMessage(`CodeDouble capture ${next ? "ON" : "OFF"}`);
    })
  );
}

export function deactivate(): void {
  for (const arr of pending.values()) for (const p of arr) if (p.timer) clearTimeout(p.timer);
}
