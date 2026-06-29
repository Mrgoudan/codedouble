"""codedouble CLI — turn it on and watch the effect.

  python3 -m codedouble.cli on            # capture git history + install hook (monitor going forward)
  python3 -m codedouble.cli capture-git   # (re)mine commits into the log
  python3 -m codedouble.cli log ...        # record one real interaction by hand
  python3 -m codedouble.cli report          # replay your real log -> ASCII + report.html
  python3 -m codedouble.cli report --sim    # same visual on the simulated user (guaranteed rich)
  python3 -m codedouble.cli status          # what's captured / is the hook on

Real mode (a real embedder) is the default for `report`.
"""

from __future__ import annotations

import argparse
import os
import stat
import sys

import json
import os
import sys
import time

from .embedder import HashingEmbedder
from .logger import (
    DEFAULT_LOG,
    EventLog,
    build_index,
    capture_git,
    codedouble_home,
    moment_of,
    record_interaction,
    replay,
)
from .signature import RuleBasedExtractor
from .viz import ascii_report, render_html


def resolve_ollama_model(model: str = None) -> str:
    """Which local model to use: --model arg > CODEDOUBLE_OLLAMA_MODEL > first
    installed model > 'mistral'. So it 'just works' with whatever you've pulled."""
    if model:
        return model
    env = os.environ.get("CODEDOUBLE_OLLAMA_MODEL")
    if env:
        return env
    from .backends import list_ollama_models
    installed = list_ollama_models()
    return installed[0] if installed else "mistral"


def build_extractor(backend: str, model: str = None):
    backend = (backend or "real").lower()
    if backend == "mistral":
        from .backends import mistral_extractor
        print("[backend] mistral (mistral-embed + LLM extractor)")
        return mistral_extractor()
    if backend in ("ollama", "local"):
        # reasoning on a LOCAL Ollama model + local embedder (no key, no cloud).
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        try:
            from .backends import STEmbedder
            emb = STEmbedder()
            emb_name = "sentence-transformers"
        except Exception:
            emb = HashingEmbedder(256)
            emb_name = "hashing"
        from .backends import ollama_extractor
        m = resolve_ollama_model(model)
        print(f"[backend] ollama (local LLM reasoning: {m}) + {emb_name} retrieval")
        return ollama_extractor(emb, model=m)
    if backend in ("real", "st", "auto"):
        # offline-first: use the cached model if present, NEVER hit the network
        # (avoids the HF HEAD-check + 5x retry hang on an offline machine).
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        try:
            from .backends import STEmbedder
            emb = STEmbedder()
            print("[backend] real: sentence-transformers (CPU, offline/cached)")
            return RuleBasedExtractor(emb)
        except Exception as e:
            print(f"[backend] real embedder not available offline ({type(e).__name__}); "
                  f"using the no-model hashing embedder instead")
            return RuleBasedExtractor(HashingEmbedder(256))
    print("[backend] default: hashing (no model)")
    return RuleBasedExtractor(HashingEmbedder(256))


HOOK = """#!/bin/sh
# codedouble: faithfully record every commit (cheap; analysis happens at report time)
python3 -m codedouble.cli capture-git --quiet >/dev/null 2>&1 || true
"""


def install_hook(repo: str = ".") -> str:
    hook_dir = os.path.join(repo, ".git", "hooks")
    if not os.path.isdir(hook_dir):
        print("not a git repo (no .git/hooks) — skipping hook")
        return ""
    path = os.path.join(hook_dir, "post-commit")
    with open(path, "w") as f:
        f.write(HOOK)
    os.chmod(path, os.stat(path).st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    print(f"installed post-commit hook -> {path}  (monitoring is ON)")
    return path


def cmd_report(args):
    if args.sim:
        from . import demo
        history, index, _ = demo.run(seed=args.seed)
        rules = len(index.semantic.rules)
        sub = "SIMULATED user — validates mechanism, not the thesis. "
    else:
        raw = EventLog(args.log).read()
        if not raw:
            print(f"no interactions captured yet at {args.log}.")
            print("run:  python3 -m codedouble.cli on        (mine git + monitor going forward)")
            print("  or: python3 -m codedouble.cli report --sim   (see it on the simulated user)")
            return
        history, double = replay(raw, build_extractor(args.backend, getattr(args, "model", None)),
                                 conf_threshold=args.conf)
        rules = len(double.index.semantic.rules)
        sub = "REAL captured log. "
    print()
    ascii_report(history, window=args.window, conf_threshold=args.conf)
    out = render_html(history, path=args.out, window=args.window, conf_threshold=args.conf, subtitle=sub)
    print(f"\nrules distilled: {rules}   visual: {os.path.abspath(out)}")


def cmd_capture_git(args):
    n = capture_git(EventLog(args.log), repo=args.repo, quiet=args.quiet)
    if not args.quiet and n:
        print("now run:  python3 -m codedouble.cli report")


def cmd_log(args):
    rec = record_interaction(
        EventLog(args.log), request=args.request, resolution=args.resolution,
        outcome=args.outcome, corrected_from=args.corrected_from, lang=args.lang,
    )
    print("logged:", rec["request"], "->", rec["resolution"], f"({rec['outcome']})")


def cmd_on(args):
    n = capture_git(EventLog(args.log), repo=args.repo)
    install_hook(args.repo)
    print("\ncodedouble is ON. Every commit is now recorded. See the effect with:")
    print("  python3 -m codedouble.cli report")


def _decide(log_path, payload, conf):
    """Run one gateway decision (fast: hashing embedder, no model load)."""
    from .double import Double
    from .types import Reversibility
    raw = EventLog(log_path).read()
    ext = RuleBasedExtractor(HashingEmbedder(256))
    double = Double(ext, build_index(raw, ext), conf_threshold=conf)
    double.now = float(len(raw) + 1)
    rev = Reversibility.HIGH if payload.get("reversibility") == "high" else Reversibility.LOW
    return double.resolve(moment_of(payload), rev)


def _log_decision(log_path, event, tool, dec, enforce, intent="", verdict=""):
    d = os.path.dirname(log_path) or "."
    try:
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "decisions.jsonl"), "a") as f:
            f.write(json.dumps({
                "ts": time.time(), "event": event, "tool": tool,
                "intent": (intent or "").strip()[:140],
                "resolution": (dec.resolution or "").strip()[:140],
                "verdict": verdict,            # inject | allow | ask | deny | shadow | watch
                "action": dec.action.value, "ask": dec.action.value == "ask",
                "confidence": round(dec.confidence, 3), "coverage": round(dec.coverage, 3),
                "n": len(dec.retrieved), "enforce": enforce,
            }) + "\n")
    except Exception:
        pass


def cmd_gate(args):
    """The gateway decision: read an intent (JSON on stdin), emit a decision JSON.
    Used for BOTH directions: intake -> use `inject`; outtake -> use `allow`/`ask`."""
    from .types import Action
    payload = {}
    try:
        if not sys.stdin.isatty():
            data = sys.stdin.read().strip()
            if data:
                payload = json.loads(data)
    except Exception:
        payload = {}
    if args.request:
        payload.setdefault("request", args.request)
    dec = _decide(args.log, payload, args.conf)
    inject = ""
    if dec.retrieved:
        inject = (f"[codedouble] precedent for this kind of change: '{dec.resolution}' "
                  f"(confidence {dec.confidence:.2f}, {len(dec.retrieved)} similar past decisions).")
    print(json.dumps({
        "action": dec.action.value,
        "allow": (True if args.shadow else dec.action is not Action.ASK),
        "ask": dec.action is Action.ASK and not args.shadow,
        "resolution": dec.resolution, "confidence": round(dec.confidence, 3),
        "coverage": round(dec.coverage, 3), "shadow": bool(args.shadow),
        "inject": inject, "rationale": dec.rationale,
    }))


def cmd_hook(args):
    """Claude Code hook adapter: read the hook event JSON on stdin, run the gate,
    emit Claude Code's expected output. FAIL-OPEN (never breaks your session) and
    SHADOW by default (no blocking) unless CODEDOUBLE_ENFORCE=1.

    intake (UserPromptSubmit) -> inject precedent as additionalContext (always)
    outtake (PreToolUse)      -> shadow: log only; enforce: allow / ask via 2x2
    """
    from .types import Action
    try:
        data = sys.stdin.read()
        ev = json.loads(data) if data.strip() else {}
    except Exception:
        return  # fail-open
    try:
        name = ev.get("hook_event_name", "")
        enforce = os.environ.get("CODEDOUBLE_ENFORCE") == "1"

        if name == "UserPromptSubmit":
            prompt = ev.get("prompt", "")
            dec = _decide(args.log, {"request": prompt, "reversibility": "low"}, args.conf)
            injected = bool(dec.retrieved and dec.confidence >= 0.4 and dec.resolution)
            _log_decision(args.log, name, None, dec, enforce, intent=prompt,
                          verdict=("inject" if injected else "watch"))
            if injected:
                # steer the input toward what the developer has consistently preferred
                ctx = (f"[codedouble] The developer has consistently preferred '{dec.resolution}' "
                       f"for this kind of request ({len(dec.retrieved)} precedents, "
                       f"confidence {dec.confidence:.2f}). Apply that approach unless this case clearly differs.")
                print(json.dumps({"hookSpecificOutput": {
                    "hookEventName": "UserPromptSubmit", "additionalContext": ctx}}))
            return

        if name == "PreToolUse":
            tool = ev.get("tool_name", "")
            ti = ev.get("tool_input", {}) or {}
            blob = (str(ti.get("command", "")) + " " + str(ti.get("file_path", ""))).lower()
            rev = "high" if any(k in blob for k in (
                "rm ", "rm -", "drop ", "delete", "truncate", "migrat", "schema",
                "force", "reset --hard")) else "low"
            payload = {
                "request": f"{tool} {ti.get('file_path','')} {ti.get('command','')}".strip(),
                "diff": str(ti.get("new_string", "") or ti.get("command", "")),
                "files": [ti["file_path"]] if ti.get("file_path") else [],
                "reversibility": rev,
            }
            dec = _decide(args.log, payload, args.conf)
            verdict, reason = "shadow", f"[codedouble] {dec.rationale}"
            if enforce:
                if dec.action is Action.ASK and rev == "high":
                    verdict = "deny"          # reject & re-ask: hard to undo + under-determined
                    reason = ("[codedouble] This is hard to undo and I have no clear precedent that "
                              "you wanted it. Redo it more safely, or rerun if it's intended.")
                elif dec.action is Action.ASK:
                    verdict = "ask"           # check with you
                    reason = (f"[codedouble] Under-determined; you've sometimes preferred "
                              f"'{dec.resolution}' here." if dec.resolution
                              else "[codedouble] Under-determined — worth a quick check.")
                else:
                    verdict = "allow"         # handled for you
            _log_decision(args.log, name, tool, dec, enforce, intent=payload["request"], verdict=verdict)
            if enforce:
                print(json.dumps({"hookSpecificOutput": {
                    "hookEventName": "PreToolUse",
                    "permissionDecision": ("deny" if verdict == "deny" else verdict),
                    "permissionDecisionReason": reason}}))
            return  # shadow: emit nothing -> normal flow
    except Exception:
        return  # fail-open: never block the user's session


def cmd_models(args):
    """List local Ollama models the user can pick for the reasoning slot."""
    from .backends import list_ollama_models
    ms = list_ollama_models()
    if not ms:
        print("no local ollama models found.")
        print("  is `ollama serve` running?  then e.g.:  ollama pull qwen2.5-coder:7b")
        return
    default = resolve_ollama_model(None)
    print("local ollama models  (use:  codedouble report --backend ollama --model NAME)")
    for m in ms:
        print(f"  {'* ' if m == default else '  '}{m}")
    print(f"\ndefault when no --model/CODEDOUBLE_OLLAMA_MODEL is set:  {default}")


def cmd_status(args):
    raw = EventLog(args.log).read()
    hook = os.path.join(args.repo, ".git", "hooks", "post-commit")
    hooked = os.path.exists(hook) and "codedouble" in open(hook).read()
    print(f"log:    {args.log}  ({len(raw)} interactions)")
    print(f"hook:   {'ON' if hooked else 'off'}  ({hook})")
    if raw:
        from collections import Counter
        c = Counter(r.get("outcome") for r in raw)
        print("outcomes:", dict(c))


def main(argv=None):
    ap = argparse.ArgumentParser(prog="codedouble")
    ap.add_argument("--log", default=DEFAULT_LOG)
    sub = ap.add_subparsers(dest="cmd")

    p = sub.add_parser("on", help="mine git + install hook (turn monitoring ON)")
    p.add_argument("--repo", default="."); p.set_defaults(func=cmd_on)

    p = sub.add_parser("capture-git", help="mine commits into the log")
    p.add_argument("--repo", default="."); p.add_argument("--quiet", action="store_true")
    p.set_defaults(func=cmd_capture_git)

    p = sub.add_parser("log", help="record one interaction by hand")
    p.add_argument("request"); p.add_argument("resolution")
    p.add_argument("--outcome", default="confirmed_good")
    p.add_argument("--corrected-from", dest="corrected_from", default=None)
    p.add_argument("--lang", default=""); p.set_defaults(func=cmd_log)

    p = sub.add_parser("report", help="replay the log -> ASCII + report.html")
    p.add_argument("--backend", default="real",
                   help="real (local ST embedder) | ollama (local LLM reasoning) | mistral | default")
    p.add_argument("--model", default=None,
                   help="ollama model for --backend ollama (overrides env; see `codedouble models`)")
    p.add_argument("--window", type=int, default=40)
    p.add_argument("--conf", type=float, default=0.6)
    p.add_argument("--out", default=os.path.join(codedouble_home(), "report.html"),
                   help="where to write the HTML (default: global ~/.codedouble/report.html)")
    p.add_argument("--sim", action="store_true", help="use the simulated user")
    p.add_argument("--seed", type=int, default=7); p.set_defaults(func=cmd_report)

    p = sub.add_parser("gate", help="gateway decision for one intent (stdin JSON) -> decision JSON")
    p.add_argument("--request", default="", help="intent text (or pipe JSON on stdin)")
    p.add_argument("--conf", type=float, default=0.6)
    p.add_argument("--shadow", action="store_true", help="never block; just log/emit the decision")
    p.set_defaults(func=cmd_gate)

    p = sub.add_parser("hook", help="Claude Code hook adapter (reads hook JSON on stdin)")
    p.add_argument("--conf", type=float, default=0.6); p.set_defaults(func=cmd_hook)

    p = sub.add_parser("install-hook"); p.add_argument("--repo", default=".")
    p.set_defaults(func=lambda a: install_hook(a.repo))

    p = sub.add_parser("models", help="list local ollama models you can pick for reasoning")
    p.set_defaults(func=cmd_models)

    p = sub.add_parser("status"); p.add_argument("--repo", default=".")
    p.set_defaults(func=cmd_status)

    args = ap.parse_args(argv)
    if not getattr(args, "func", None):
        ap.print_help(); return
    args.func(args)


if __name__ == "__main__":
    main()
