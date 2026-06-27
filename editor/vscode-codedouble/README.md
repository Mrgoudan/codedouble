# CodeDouble Capture — VS Code extension

The editor-level capture surface for the Self-Learning Code Double. It records
the signals a git hook can't see — **accept / override / reject / viewed** —
into the **same** `.codedouble/interactions.jsonl` the Python CLI analyzes.
Capture happens in the editor; analysis + visualization stay in Python
(`python3 -m codedouble.cli report`). Two-phase by design (README §6).

## What it captures

**Automatically (heuristic):**
- A single large insert (≥ `minInsertLines` lines or ≥ `minInsertChars` chars) =
  a **proposal** (AI output or paste).
- If that region is edited within `overrideWindowMs` → **override** (records the
  before/after).
- If it survives `settleMs` unedited → **accepted_silent** *only if it was on
  screen*; if never on screen → **never_viewed** (zero signal, README §6).

**Explicitly (commands / bind keys to these):**
- `CodeDouble: Accept selection (reviewed, wholesale)` → **confirmed_good** (the strong positive)
- `CodeDouble: Mark selection as an override` → **override** (prompts for the original X)
- `CodeDouble: Reject / interrupt` → **interrupt**
- `CodeDouble: Open report` → runs `python3 -m codedouble.cli report`
- `CodeDouble: Toggle capture on/off`

A status-bar item (`$(eye) CodeDouble N`) shows the session count; click it for the report.

## Build & run

```bash
cd editor/vscode-codedouble
npm install
npm run compile           # -> out/extension.js
```
Then in VS Code: open this folder and press **F5** (launches an Extension
Development Host with the extension loaded). Or package and install:
```bash
npx @vscode/vsce package  # -> vscode-codedouble-0.1.0.vsix
code --install-extension vscode-codedouble-0.1.0.vsix
```

Open your project, code as usual, then `python3 -m codedouble.cli report`.

## Settings

`codedouble.enabled`, `minInsertLines` (3), `minInsertChars` (80),
`overrideWindowMs` (120000), `settleMs` (90000).

## Honest limits

This is heuristic capture — it infers "the AI proposed this" from *large
inserts*, not from any specific agent's API. That conflates AI output with large
pastes (acceptable: both are "code that appeared, kept or changed"), and range
tracking ignores line drift. For precise per-suggestion accept/reject you'd
integrate with your AI agent's own API and call the same JSONL writer. The
explicit commands give clean, unambiguous labels in the meantime — and the
strong `confirmed_good` signal only comes from an explicit reviewed-accept,
exactly as the design intends.
