"""Core data structures (README §6).

A `Signature` is the composite "what made this ambiguous" key:
  - structured/categorical fields  -> exact-match FILTERS
  - dense vectors (code + intent)  -> SIMILARITY
  - raw text                       -> kept for re-reflection (event-sourcing)

A `ResolutionEvent` is one logged decision + how the user reacted (the label).
The index is an append-only log of these — the source of truth.
"""

from __future__ import annotations

import enum
import itertools
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np


class Outcome(enum.Enum):
    """How the user reacted (README §6). Negatives raise the threshold,
    positives lower it; `viewed` gates the silent positive."""

    # negatives (intervention = failure signal)
    INTERRUPT = "interrupt"          # halted mid-execution; earliest, strongest
    OVERRIDE = "override"            # corrected X -> Y in conversation
    REVERT = "revert"               # undid a committed change (git revert)
    # positives
    CONFIRMED_GOOD = "confirmed_good"  # reviewed + accepted wholesale
    ANSWERED = "answered"              # answered a clarifying question (gold)
    ACCEPTED_SILENT = "accepted_silent"  # weak; ONLY if viewed
    # bookkeeping
    PENDING = "pending"             # not yet resolved
    NEVER_VIEWED = "never_viewed"   # zero signal — must NOT count as positive


_NEGATIVE = {Outcome.INTERRUPT, Outcome.OVERRIDE, Outcome.REVERT}
_POSITIVE = {Outcome.CONFIRMED_GOOD, Outcome.ANSWERED, Outcome.ACCEPTED_SILENT}


def is_negative(o: Outcome) -> bool:
    return o in _NEGATIVE


def is_positive(o: Outcome) -> bool:
    return o in _POSITIVE


class Action(enum.Enum):
    """The 2x2 gate output (README §4)."""

    ACT = "act"               # high conf, low revert-cost  -> bypass, log
    ACT_FLAG = "act_flag"     # high conf, high revert-cost -> act, flag for review
    ACT_LOUD = "act_loud"     # low conf, low revert-cost   -> act loudly, easy revert
    ASK = "ask"               # low conf, high revert-cost  -> the only real ask quadrant

    @property
    def is_silent(self) -> bool:
        return self is not Action.ASK


class Source(enum.Enum):
    """Which rung of the escalation ladder resolved it (README §3)."""

    SITUATION = "situation"
    INDEX = "index"        # this user's precedent
    COHORT = "cohort"      # cross-user prior
    FRONTIER = "frontier"  # escalated compute
    HUMAN = "human"


class Reversibility(enum.Enum):
    LOW = "low"    # cheap to undo (leaf, git-undoable)
    HIGH = "high"  # expensive to undo (foundational, entangled)


# weight a stored event contributes when computing confidence — by how strong
# the signal that produced its (endorsed) resolution was.
LABEL_QUALITY = {
    Outcome.ANSWERED: 3.0,
    Outcome.CONFIRMED_GOOD: 2.0,    # survived long without a negative (reflect layer 2)
    Outcome.OVERRIDE: 1.5,          # the correction reveals the true Y — solid
    Outcome.REVERT: 1.0,
    Outcome.INTERRUPT: 1.0,
    Outcome.ACCEPTED_SILENT: 0.5,   # survived short idle (reflect layer 1) — weak positive
    Outcome.PENDING: 0.0,           # awaiting judgement — contributes nothing yet
    Outcome.NEVER_VIEWED: 0.0,      # zero signal
}


@dataclass
class Signature:
    # ---- structured (filters) ----
    lang: str = ""
    repo: str = ""
    error_type: Optional[str] = None
    action_kind: str = ""        # edit | delete | rename | refactor | add
    phrasing_class: str = ""     # clean-up | fix-it | make-like-X | add-feature | other
    files: Tuple[str, ...] = ()
    symbols: Tuple[str, ...] = ()
    # ---- embedded (similarity) ----
    code_vec: Optional[np.ndarray] = None
    intent_vec: Optional[np.ndarray] = None
    # ---- raw (re-reflection / audit) ----
    raw_request: str = ""
    raw_diff: str = ""
    interpretation_space: List[str] = field(default_factory=list)

    def coarse_key(self) -> Tuple[str, str, str, Optional[str]]:
        """The altitude reflection distills rules at (README §6 generalization)."""
        return (self.lang, self.phrasing_class, self.action_kind, self.error_type)

    def metadata(self) -> dict:
        return {
            "lang": self.lang,
            "repo": self.repo,
            "error_type": self.error_type,
            "action_kind": self.action_kind,
            "phrasing_class": self.phrasing_class,
        }


@dataclass
class ResolutionEvent:
    id: int
    ts: float                     # logical time (round index) — recency weighting
    signature: Signature
    resolution: str               # the endorsed reading (what the user ultimately wanted)
    outcome: Outcome              # how we learned it
    source: Source
    confidence_at_decision: float
    corrected_from: Optional[str] = None  # X, when outcome == OVERRIDE
    session_id: Optional[int] = None

    @property
    def weight_base(self) -> float:
        return LABEL_QUALITY.get(self.outcome, 1.0)


@dataclass
class Decision:
    action: Action
    resolution: Optional[str]     # None when ASK with no usable precedent
    confidence: float
    agreement: float
    coverage: float
    source: Source
    rationale: str
    signature: Signature
    # the precedent events that produced this (for transparency / "I assumed X because…")
    retrieved: List[ResolutionEvent] = field(default_factory=list)
    # known-bad signal: weighted fraction of NEGATIVE precedent (you overrode/reverted
    # similar before) and how much precedent backs it. Drives "send back", NOT ignorance.
    risk: float = 0.0
    risk_coverage: float = 0.0


@dataclass
class PreferenceRule:
    """A distilled, stable preference (README §3 semantic memory)."""

    coarse_key: Tuple[str, str, str, Optional[str]]
    resolution: str
    support: int                  # how many consistent events backed it
    last_ts: float


_id_counter = itertools.count(1)


def next_event_id() -> int:
    return next(_id_counter)
