#!/usr/bin/env bash
# codedouble — frictionless, idempotent setup. Safe to re-run.
#
#   ./setup.sh            # DEFAULT: enforce + QC (acts on your behalf + LLM drift-check)
#   ./setup.sh --no-qc    # act (send-back), but skip the LLM drift-check
#   ./setup.sh --shadow   # observe only — nothing blocked
#
# Installs the package, the local model (cheap tier), the VS Code panel, and wires
# the Claude Code hooks. Each step is optional and skipped cleanly if its tool is absent.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "== codedouble setup =="

# 1) Python package (editable install)
( python3 -m pip install -e "$HERE" >/dev/null 2>&1 || pip install -e "$HERE" >/dev/null 2>&1 ) \
  && echo "[1/4] python package installed" \
  || echo "[1/4] pip install skipped (already installed, or old pip)"

# 2) Local model for the cheap/frequent tier (optional)
if command -v ollama >/dev/null 2>&1; then
  if ollama list 2>/dev/null | grep -qi "qwen2.5-coder:7b"; then
    echo "[2/4] local model present (qwen2.5-coder:7b)"
  else
    echo "[2/4] pulling local model qwen2.5-coder:7b (~4.7GB) …"
    ollama pull qwen2.5-coder:7b || echo "      (pull skipped/failed — the remote LLM port still works)"
  fi
else
  echo "[2/4] ollama not found — skipping local tier (remote LLM port still works)"
fi

# 3) VS Code capture panel (optional)
if command -v code >/dev/null 2>&1 && [ -d "$HERE/editor/vscode-codedouble" ]; then
  ( cd "$HERE/editor/vscode-codedouble" \
      && npm run compile >/dev/null 2>&1 \
      && npx --yes @vscode/vsce package >/dev/null 2>&1 \
      && code --install-extension vscode-codedouble-*.vsix >/dev/null 2>&1 ) \
    && echo "[3/4] VS Code capture panel installed" \
    || echo "[3/4] VS Code panel skipped (build/install failed)"
else
  echo "[3/4] 'code' not found — skipping VS Code panel"
fi

# 4) Wire the Claude Code hooks + store + LLM-port template (idempotent, backs up)
echo "[4/4] wiring Claude Code hooks …"
python3 -m codedouble.cli setup "$@"

echo
echo "== done =="
echo "Remote LLM for hard, session-wide tasks (optional): edit ~/.codedouble/llm.json, e.g."
echo '  {"base_url":"http://HOST:PORT/v1","model":"GLM-5.2","api_key_file":"/path/.env","api_key_field":["KEY1","KEY2"]}'
echo "Start a new Claude Code session to activate."
