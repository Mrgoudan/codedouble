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

from .embedder import HashingEmbedder
from .logger import DEFAULT_LOG, EventLog, capture_git, record_interaction, replay
from .signature import RuleBasedExtractor
from .viz import ascii_report, render_html


def build_extractor(backend: str):
    backend = (backend or "real").lower()
    if backend == "mistral":
        from .backends import mistral_extractor
        print("[backend] mistral (mistral-embed + LLM extractor)")
        return mistral_extractor()
    if backend in ("real", "st"):
        try:
            from .backends import STEmbedder
            print("[backend] real: sentence-transformers (CPU)")
            return RuleBasedExtractor(STEmbedder())
        except Exception as e:
            print(f"[backend] real embedder unavailable ({e}); using hashing")
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
        history, double = replay(raw, build_extractor(args.backend), conf_threshold=args.conf)
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
    p.add_argument("--backend", default="real", help="real | mistral | default")
    p.add_argument("--window", type=int, default=40)
    p.add_argument("--conf", type=float, default=0.6)
    p.add_argument("--out", default="report.html")
    p.add_argument("--sim", action="store_true", help="use the simulated user")
    p.add_argument("--seed", type=int, default=7); p.set_defaults(func=cmd_report)

    p = sub.add_parser("install-hook"); p.add_argument("--repo", default=".")
    p.set_defaults(func=lambda a: install_hook(a.repo))

    p = sub.add_parser("status"); p.add_argument("--repo", default=".")
    p.set_defaults(func=cmd_status)

    args = ap.parse_args(argv)
    if not getattr(args, "func", None):
        ap.print_help(); return
    args.func(args)


if __name__ == "__main__":
    main()
