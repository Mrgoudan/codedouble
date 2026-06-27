"""The Double: external monitor wiring extractor + index + the 2x2 gate.

resolve()  -> look at one intent, retrieve precedent, compute calibrated
              confidence, pick a quadrant of the ask/act gate (README §4).
record()   -> faithfully log how the user reacted (the label) into the index.

The executor is NOT here — the Double watches it. Per README §2 the monitor
must be external; this class never asks the executor to judge its own confidence.
"""

from __future__ import annotations

from typing import Callable, Optional

from .index import ResolutionIndex, Calibrator
from .signature import SignatureExtractor
from .types import (
    Action,
    Decision,
    Outcome,
    ResolutionEvent,
    Reversibility,
    Signature,
    Source,
    next_event_id,
)


def gate(confidence: float, reversibility: Reversibility, conf_threshold: float) -> Action:
    """The 2x2 (README §4): ask ONLY when under-determined AND hard to undo."""
    conf_high = confidence >= conf_threshold
    rev_high = reversibility is Reversibility.HIGH
    if conf_high and not rev_high:
        return Action.ACT          # bypass, log the assumption
    if conf_high and rev_high:
        return Action.ACT_FLAG     # act, but flag for review
    if not conf_high and not rev_high:
        return Action.ACT_LOUD     # act loudly, trivially revertable
    return Action.ASK              # the only real ask quadrant


class Double:
    def __init__(
        self,
        extractor: SignatureExtractor,
        index: Optional[ResolutionIndex] = None,
        calibrator: Optional[Calibrator] = None,
        conf_threshold: float = 0.6,
        escalate: Optional[Callable[[Signature], Optional[str]]] = None,
    ):
        self.extractor = extractor
        self.index = index or ResolutionIndex()
        self.calibrator = calibrator or Calibrator()
        self.conf_threshold = conf_threshold
        self.escalate = escalate  # frontier-LLM rung stub (README §3 [2c])
        self.now: float = 0.0

    def resolve(self, moment: dict, reversibility: Reversibility) -> Decision:
        sig = self.extractor.extract(moment)
        retrieved = self.index.retrieve(sig, now=self.now)
        raw_conf, winner, agreement, coverage = self.index.confidence(retrieved, self.now)
        conf = self.calibrator.transform(raw_conf)
        source = Source.INDEX if retrieved else Source.SITUATION

        action = gate(conf, reversibility, self.conf_threshold)

        # nothing to act on (no precedent guess) -> must ask, regardless of 2x2
        if winner is None:
            action = Action.ASK

        # escalation rung [2c]: before asking the human, try frontier compute —
        # but only for *preventable* hardness (we have low coverage, not zero
        # signal). Stubbed: a real impl calls a stronger frozen model here.
        if action is Action.ASK and self.escalate is not None and coverage > 0:
            guess = self.escalate(sig)
            if guess is not None:
                winner = guess
                source = Source.FRONTIER
                action = Action.ACT_LOUD  # act, but stay revertable

        rationale = (
            f"conf={conf:.2f} (agree={agreement:.2f}, cover={coverage:.2f}), "
            f"rev={reversibility.value}, n={len(retrieved)} -> {action.value}"
        )
        return Decision(
            action=action,
            resolution=winner,
            confidence=conf,
            agreement=agreement,
            coverage=coverage,
            source=source,
            rationale=rationale,
            signature=sig,
            retrieved=[e for e, _ in retrieved],
        )

    def record(
        self,
        decision: Decision,
        outcome: Outcome,
        corrected_to: Optional[str] = None,
        session_id: Optional[int] = None,
    ) -> ResolutionEvent:
        """Log the endorsed resolution (event-sourced). For an OVERRIDE/ANSWERED,
        `corrected_to` is the resolution the user actually wanted."""
        resolution = corrected_to if corrected_to is not None else decision.resolution
        if resolution is None:
            # nothing usable to store (e.g. asked but no answer captured)
            resolution = "<unresolved>"
        event = ResolutionEvent(
            id=next_event_id(),
            ts=self.now,
            signature=decision.signature,
            resolution=resolution,
            outcome=outcome,
            source=decision.source,
            confidence_at_decision=decision.confidence,
            corrected_from=(decision.resolution if outcome is Outcome.OVERRIDE else None),
            session_id=session_id,
        )
        self.index.add(event)
        return event

    def tick(self, dt: float = 1.0) -> None:
        self.now += dt
