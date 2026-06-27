// CodeDouble Capture — a VS Code extension that faithfully records editor
// interactions into .codedouble/interactions.jsonl (same schema the Python
// `codedouble` CLI reads). This is the editor-level §10 logger: it sees the
// accept / override / reject / viewed signals a git hook cannot.
//
// Two capture modes:
//   - automatic heuristic: a large single insert (AI output or paste) becomes a
//     "proposal"; if its region is edited soon after -> OVERRIDE; if it survives
//     and was on screen -> weak ACCEPTED_SILENT; never on screen -> NEVER_VIEWED
//     (zero signal, per README §6).
//   - explicit commands: "Accept (reviewed)" -> CONFIRMED_GOOD (the strong
//     positive), "Mark override", "Reject".
//
// Analysis/visualization stays in Python: `python3 -m codedouble.cli report`.

import * as fs from "fs";
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

const pending = new Map<string, Proposal[]>(); // docUri -> proposals
let sessionCount = 0;
let statusBar: vscode.StatusBarItem;

function cfg<T>(key: string, dflt: T): T {
  return vscode.workspace.getConfiguration("codedouble").get<T>(key, dflt);
}

function workspaceRoot(): string | undefined {
  const ws = vscode.workspace.workspaceFolders;
  return ws && ws.length ? ws[0].uri.fsPath : undefined;
}

function logPath(): string | undefined {
  const root = workspaceRoot();
  if (!root) return undefined;
  const dir = path.join(root, ".codedouble");
  try {
    fs.mkdirSync(dir, { recursive: true });
  } catch {
    /* ignore */
  }
  return path.join(dir, "interactions.jsonl");
}

const LANG: Record<string, string> = {
  python: "python", typescript: "ts", javascript: "ts", typescriptreact: "ts",
  javascriptreact: "ts", java: "java", go: "go", rust: "rust", c: "c",
  cpp: "cpp", ruby: "ruby", php: "php", csharp: "csharp",
};

function firstLine(text: string): string {
  for (const ln of text.split("\n")) {
    const t = ln.trim();
    if (t) return t.slice(0, 80);
  }
  return text.trim().slice(0, 80);
}

function reversibilityOf(doc: vscode.TextDocument, actionKind: string): string {
  const p = doc.uri.fsPath.toLowerCase();
  if (actionKind === "delete" || actionKind === "rename" || p.includes("migrat") || p.includes("schema"))
    return "high";
  return "low";
}

function record(
  doc: vscode.TextDocument,
  outcome: string,
  resolution: string,
  correctedFrom: string | null,
  diff: string,
  actionKind = "edit"
): void {
  const lp = logPath();
  if (!lp) return;
  const rel = vscode.workspace.asRelativePath(doc.uri);
  const rec = {
    ts: Date.now() / 1000,
    source: "editor",
    request: `${actionKind} ${rel}`,
    diff: diff.slice(0, 2000),
    error: "",
    lang: LANG[doc.languageId] ?? doc.languageId,
    files: [rel],
    action_kind: actionKind,
    reversibility: reversibilityOf(doc, actionKind),
    outcome,
    resolution,
    corrected_from: correctedFrom,
    sha: null,
  };
  try {
    fs.appendFileSync(lp, JSON.stringify(rec) + "\n");
    sessionCount++;
    updateStatus();
  } catch (e) {
    console.error("codedouble: write failed", e);
  }
}

function updateStatus(): void {
  if (!statusBar) return;
  const on = cfg<boolean>("enabled", true);
  statusBar.text = `$(eye) CodeDouble ${on ? sessionCount : "off"}`;
  statusBar.tooltip = "Interactions captured this session — click to open report";
  statusBar.show();
}

function finalizeAccepted(docUri: string, p: Proposal, doc?: vscode.TextDocument): void {
  if (p.timer) clearTimeout(p.timer);
  const arr = pending.get(docUri);
  if (arr) pending.set(docUri, arr.filter((x) => x !== p));
  if (!doc) return;
  // survived without an override: weak positive only if it was actually viewed
  if (p.viewed) {
    record(doc, "accepted_silent", firstLine(p.text), null, p.text);
  } else {
    record(doc, "never_viewed", firstLine(p.text), null, p.text);
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
    const isProposal =
      ch.text.length > 0 &&
      ch.text.trim().length > 0 &&
      (newlines + 1 >= minLines || ch.text.length >= minChars);

    if (isProposal) {
      const startLine = ch.range.start.line;
      const proposal: Proposal = {
        startLine,
        endLine: startLine + newlines,
        text: ch.text,
        tsMs: Date.now(),
        viewed: false,
      };
      proposal.timer = setTimeout(() => finalizeAccepted(key, proposal, doc), settleMs);
      const arr = pending.get(key) ?? [];
      arr.push(proposal);
      pending.set(key, arr);
    } else {
      // small edit: does it land inside a recent proposal? -> OVERRIDE
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
          record(doc, "override", corrected || firstLine(ch.text), firstLine(p.text), ch.text);
          break;
        }
      }
    }
  }
}

function onVisible(e: vscode.TextEditorVisibleRangesChangeEvent): void {
  const arr = pending.get(e.textEditor.document.uri.toString());
  if (!arr) return;
  for (const p of arr) {
    for (const vr of e.visibleRanges) {
      if (vr.start.line <= p.endLine && vr.end.line >= p.startLine) {
        p.viewed = true;
      }
    }
  }
}

function selectionText(): { doc: vscode.TextDocument; text: string } | undefined {
  const ed = vscode.window.activeTextEditor;
  if (!ed) return undefined;
  const sel = ed.selection;
  const text = sel.isEmpty ? ed.document.lineAt(sel.active.line).text : ed.document.getText(sel);
  return { doc: ed.document, text };
}

export function activate(context: vscode.ExtensionContext): void {
  statusBar = vscode.window.createStatusBarItem(vscode.StatusBarAlignment.Right, 100);
  statusBar.command = "codedouble.openReport";
  updateStatus();

  context.subscriptions.push(
    statusBar,
    vscode.workspace.onDidChangeTextDocument(onChange),
    vscode.window.onDidChangeTextEditorVisibleRanges(onVisible),

    vscode.commands.registerCommand("codedouble.acceptReviewed", () => {
      const s = selectionText();
      if (!s) return;
      record(s.doc, "confirmed_good", firstLine(s.text), null, s.text);
      vscode.window.setStatusBarMessage("CodeDouble: recorded confirmed-good (reviewed)", 2500);
    }),

    vscode.commands.registerCommand("codedouble.markOverride", async () => {
      const s = selectionText();
      if (!s) return;
      const from = await vscode.window.showInputBox({
        prompt: "What did the AI propose that you changed? (the original, X)",
        placeHolder: "leave blank if unknown",
      });
      record(s.doc, "override", firstLine(s.text), from ? firstLine(from) : null, s.text);
      vscode.window.setStatusBarMessage("CodeDouble: recorded override (X→Y)", 2500);
    }),

    vscode.commands.registerCommand("codedouble.reject", () => {
      const s = selectionText();
      if (!s) return;
      record(s.doc, "interrupt", firstLine(s.text), null, s.text);
      vscode.window.setStatusBarMessage("CodeDouble: recorded reject/interrupt", 2500);
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
      updateStatus();
      vscode.window.showInformationMessage(`CodeDouble capture ${next ? "ON" : "OFF"}`);
    })
  );
}

export function deactivate(): void {
  for (const arr of pending.values()) {
    for (const p of arr) if (p.timer) clearTimeout(p.timer);
  }
}
