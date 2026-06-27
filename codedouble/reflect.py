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
