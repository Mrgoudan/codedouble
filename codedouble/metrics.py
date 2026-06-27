"""The metric that judges everything (README §8).

The headline number: of the times the double stayed silent at high confidence,
how often did the user override/revert — and is that rate falling over time?

That is a calibration check on the "stay silent" decisions. We also expose the
broader health number (total intervention rate) and overall accuracy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

from .types import Action, Decision, Outcome, is_negative


@dataclass
class Record:
    decision: Decision
    outcome: Outcome
    correct: bool


def section8_rate(records: List[Record], conf_threshold: float = 0.6) -> Tuple[Optional[float], int]:
    """Override/revert rate among *confident, silent* decisions.
    Returns (rate, n); rate is None when there were no such decisions."""
    silent_confident = [
        r for r in records
        if r.decision.action.is_silent and r.decision.confidence >= conf_threshold
    ]
    if not silent_confident:
        return None, 0
    bad = sum(1 for r in silent_confident if is_negative(r.outcome))
    return bad / len(silent_confident), len(silent_confident)


def intervention_rate(records: List[Record]) -> float:
    """Total intervention rate (the umbrella health number, README §8)."""
    if not records:
        return 0.0
    return sum(1 for r in records if is_negative(r.outcome)) / len(records)


def accuracy(records: List[Record]) -> float:
    if not records:
        return 0.0
    return sum(1 for r in records if r.correct) / len(records)


def ask_rate(records: List[Record]) -> float:
    if not records:
        return 0.0
    return sum(1 for r in records if r.decision.action is Action.ASK) / len(records)


def windowed_section8(
    records: List[Record], window: int = 40, conf_threshold: float = 0.6
) -> List[Tuple[int, Optional[float], int]]:
    """The curve: §8 rate per consecutive window. Returns [(window_idx, rate, n)]."""
    out = []
    for i in range(0, len(records), window):
        chunk = records[i : i + window]
        rate, n = section8_rate(chunk, conf_threshold)
        out.append((i // window, rate, n))
    return out


def window_stats(
    records: List[Record], window: int = 40, conf_threshold: float = 0.6
) -> List[dict]:
    """Per-window rollup for visualization: §8 rate, ask-rate, accuracy, n."""
    rows = []
    for i in range(0, len(records), window):
        chunk = records[i : i + window]
        s8, n = section8_rate(chunk, conf_threshold)
        rows.append(
            {
                "i": i // window,
                "s8": s8,
                "ask": ask_rate(chunk),
                "acc": accuracy(chunk),
                "n": n,
            }
        )
    return rows
