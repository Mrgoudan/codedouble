"""Gate-calibration suite — measures QC + drift send-back quality on FROZEN fixtures.

  python3 tests/calibration.py                 # report FP/FN rates per gate
  python3 tests/calibration.py --assert-fp 0.2 # nonzero exit if FP rate exceeds

Fixtures (tests/fixtures/calibration.jsonl) are frozen — mutated from REAL failures
observed in live use plus an adversarial set authored once — never regenerated per
run (a regenerated suite is a moving target and a flaky test). This validates the
gates' MECHANISM; the product's real numbers (anchor fidelity, send-back precision)
stay grounded in live use — simulation validates mechanism, not thesis (README).

FP = expected-allow but denied (a fought send-back). FN = expected-deny but allowed.
The QC gate needs the local LLM; those cases are skipped (not failed) when it is
unreachable. Drift's deterministic layer always runs.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from codedouble import cli  # noqa: E402

FIXTURES = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "fixtures", "calibration.jsonl")


def llm_reachable() -> bool:
    try:
        from codedouble.backends import llm_complete
        return "ok" in llm_complete("reply with the single word ok",
                                    timeout=25, prefer="local").lower()
    except Exception:
        return False


def run_case(c):
    """-> 'allow' | 'deny' as the gate actually decided."""
    if c["gate"] == "drift":
        return "deny" if cli._drift_check(c["anchors"], c["input"], "")[0] else "allow"
    ok, violated, _ = cli._quality_check(c["anchors"], c["input"])
    return "allow" if ok else "deny"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--assert-fp", type=float, default=None,
                    help="exit 1 if any gate's FP rate exceeds this")
    ap.add_argument("--assert-fn", type=float, default=None)
    ap.add_argument("--gate", choices=["qc", "drift"], default=None)
    args = ap.parse_args()

    cases = [json.loads(l) for l in open(FIXTURES) if l.strip()]
    if args.gate:
        cases = [c for c in cases if c["gate"] == args.gate]
    have_llm = llm_reachable()
    if not have_llm:
        print("NOTE: local LLM unreachable — QC cases skipped (fail-open would "
              "trivially pass allow-cases and fail deny-cases).")

    stats = {}   # gate -> dict(fp,fn,ok,n)
    failures = []
    for c in cases:
        if c["gate"] == "qc" and not have_llm:
            continue
        t = time.time()
        got = run_case(c)
        st = stats.setdefault(c["gate"], {"fp": 0, "fn": 0, "ok": 0, "n": 0})
        st["n"] += 1
        if got == c["expect"]:
            st["ok"] += 1
        elif got == "deny":                      # expected allow, got deny
            st["fp"] += 1
            failures.append((c, got))
        else:                                    # expected deny, got allow
            st["fn"] += 1
            failures.append((c, got))
        print(f"  {c['id']:34s} expect={c['expect']:5s} got={got:5s} "
              f"{'OK ' if got == c['expect'] else 'MISS'}  ({time.time()-t:.1f}s)")

    print("\n=== gate calibration (frozen fixtures; results depend on the local model) ===")
    bad = False
    for gate, st in sorted(stats.items()):
        fp = st["fp"] / st["n"] if st["n"] else 0.0
        fn = st["fn"] / st["n"] if st["n"] else 0.0
        print(f"  {gate:5s}: n={st['n']:2d}  correct={st['ok']:2d}  "
              f"FP(fought send-back)={fp:.0%}  FN(missed violation)={fn:.0%}")
        if args.assert_fp is not None and fp > args.assert_fp:
            bad = True
        if args.assert_fn is not None and fn > args.assert_fn:
            bad = True
    for c, got in failures:
        print(f"  MISS {c['id']}: expected {c['expect']}, got {got} — {c['why']}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
