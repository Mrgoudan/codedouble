"""Session-end reflection (README §6: faithful record -> reflect).

Two jobs, run once per session, off the hot path:
  1. credit assignment  — turn the session arc into per-decision labels
                          (explicit interventions are precise; the session
                          outcome labels the rest).
  2. distillation       — promote repeated, consistent resolutions into stable
                          PreferenceRules; fit the calibrator from outcomes.

"Calibrated reflection" (README §6): a resolution becomes a *rule* only with
enough consistent support — we don't confidently learn from one event.
"""

from __future__ import annotations

from collections import defaultdict
from typing import List, Optional, Tuple

from .index import Calibrator, ResolutionIndex
from .types import Action, Decision, Outcome, is_positive


def reflect_log(raw, now, idle_fast, idle_good, extractor, min_promote=3):
    """Survival-time *layered* reflection on the real interaction log — no commits
    needed. Un-reacted changes are tiered by how long they've lived untouched:

        age < idle_fast            -> pending           (too soon to trust)
        idle_fast <= age < idle_good -> accepted_silent (mostly good)
        age >= idle_good           -> confirmed_good    (survived long -> better)

    A later override/revert on the same file supersedes it (stays pending). Then
    distil stable (coarse-key -> resolution) patterns into general rules, and
    coarse-key -> avoid for repeatedly-corrected patterns. Idempotent (recomputed
    from timestamps), so it can run every idle tick and progressively generalise.
    Returns (updated_records, rules, tier_counts)."""
    from .logger import moment_of
    NEG = {"override", "revert", "interrupt"}
    neg_ts = defaultdict(list)                      # cheap reaction attribution by file
    for r in raw:
        if r.get("outcome") in NEG:
            for f in (r.get("files") or []):
                neg_ts[f].append(r.get("ts", 0))

    tiers: dict = defaultdict(int)
    updated = []
    for r in raw:
        oc = r.get("outcome")
        if oc in NEG or oc == "answered":          # explicit signals stay as-is
            tiers[oc] += 1; updated.append(r); continue
        ts = r.get("ts", 0); age = now - ts
        superseded = any(any(t > ts for t in neg_ts.get(f, [])) for f in (r.get("files") or []))
        if superseded or age < idle_fast:
            new = "pending"
        elif age < idle_good:
            new = "accepted_silent"
        else:
            new = "confirmed_good"
        if new != oc:
            r = dict(r); r["outcome"] = new
        tiers[new] += 1
        updated.append(r)

    # distillation: compress + generalise to coarse-key rules
    pos = defaultdict(lambda: defaultdict(float))
    neg = defaultdict(float)
    for r in updated:
        oc = r.get("outcome")
        key = extractor.extract(moment_of(r)).coarse_key()
        if oc in ("confirmed_good", "answered"):
            pos[key][r.get("resolution", "")] += 2.0
        elif oc in ("accepted_silent", "override"):
            pos[key][r.get("resolution", "")] += 1.0
        if oc in NEG:
            neg[key] += 1.0
    rules = []
    for key, resw in pos.items():
        res, support = max(resw.items(), key=lambda kv: kv[1])
        if support >= min_promote and res:
            rules.append({"key": list(key), "prefer": res, "support": round(support, 1)})
    for key, w in neg.items():
        if w >= min_promote:
            rules.append({"key": list(key), "avoid": True, "support": round(w, 1)})
    return updated, rules, dict(tiers)


def credit_assign(
    records: List[Tuple[Decision, Outcome, Optional[bool]]],
    session_outcome: str,
) -> List[Tuple[Decision, Outcome, bool]]:
    """Fill PENDING outcomes from the session result. Explicit interventions
    keep their precise (negative) label; un-intervened decisions inherit the
    session outcome: merged-clean -> confirmed-good; reverted/abandoned -> revert.
    """
    out = []
    merged = session_outcome == "merged"
    for decision, outcome, correct in records:
        if outcome is Outcome.PENDING:
            if decision.action is Action.ASK:
                outcome = Outcome.ANSWERED  # an ask that got answered
                correct = True if correct is None else correct
            elif merged:
                outcome = Outcome.CONFIRMED_GOOD
                correct = True if correct is None else correct
            else:
                outcome = Outcome.REVERT
                correct = False if correct is None else correct
        out.append((decision, outcome, bool(correct)))
    return out


def reflect_session(
    index: ResolutionIndex,
    calibrator: Calibrator,
    records: List[Tuple[Decision, Outcome, bool]],
    now: float,
    min_promote: int = 3,
) -> dict:
    """Fit calibration on this session's labels, then distill rules from the
    whole index. Returns a small summary."""

    # 1. calibration — only silent decisions (the ones §8 grades)
    observed = 0
    for decision, outcome, correct in records:
        if decision.action.is_silent:
            calibrator.observe(decision.confidence, correct)
            observed += 1

    # 2. distillation — group endorsed resolutions by coarse key (the altitude)
    groups = defaultdict(lambda: defaultdict(float))
    for e in index.events:
        if is_positive(e.outcome) or e.outcome is Outcome.OVERRIDE:
            groups[e.signature.coarse_key()][e.resolution] += 1.0

    promoted = 0
    for key, resolutions in groups.items():
        res, support = max(resolutions.items(), key=lambda kv: kv[1])
        if support >= min_promote:
            index.semantic.upsert(key, res, int(support), now)
            promoted += 1

    return {
        "calibration_observations": observed,
        "rules_total": len(index.semantic.rules),
        "rules_promoted_or_updated": promoted,
        "ece": calibrator.ece(),
    }
