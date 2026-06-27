"""The index: episodic log + vector retrieval + semantic store + calibration.

This is the heart of "learning lives in retrieval, models stay frozen"
(README §1, §7). Confidence is computed from EVIDENCE — coverage and agreement
of retrieved precedent — never from a model's self-report (README §8).

Three stores, as in docs/SPEC-models.md:
  - episodic:  append-only ResolutionEvents (source of truth, re-reflectable)
  - vector:    the events' embeddings, searched by filter-then-similarity
  - semantic:  distilled PreferenceRules (written by reflection)
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np

from .embedder import cosine
from .types import (
    Outcome,
    PreferenceRule,
    ResolutionEvent,
    Signature,
)

# tuning knobs (prototype defaults)
RECENCY_HALFLIFE = 50.0   # ts units after which an event's weight halves
COVERAGE_TAU = 3.0        # weight at which coverage-factor ~ 0.63 (saturates)
SIM_THRESHOLD = 0.45      # min combined cosine to count as "similar"
CODE_W, INTENT_W = 0.5, 0.5


class SemanticStore:
    """Distilled, stable preferences (the 'rules' reflection produces)."""

    def __init__(self) -> None:
        self.rules: Dict[Tuple, PreferenceRule] = {}

    def upsert(self, key: Tuple, resolution: str, support: int, ts: float) -> None:
        existing = self.rules.get(key)
        if existing is None or support >= existing.support:
            self.rules[key] = PreferenceRule(key, resolution, support, ts)

    def get(self, key: Tuple) -> Optional[PreferenceRule]:
        return self.rules.get(key)


class Calibrator:
    """Histogram calibrator (README §8): map raw confidence -> measured
    probability-correct, fit from accumulated (confidence, was_correct) pairs."""

    def __init__(self, n_bins: int = 10, min_count: int = 8):
        self.n_bins = n_bins
        self.min_count = min_count
        self.n = [0] * n_bins
        self.correct = [0] * n_bins

    def _bin(self, c: float) -> int:
        return min(self.n_bins - 1, max(0, int(c * self.n_bins)))

    def observe(self, confidence: float, correct: bool) -> None:
        b = self._bin(confidence)
        self.n[b] += 1
        self.correct[b] += 1 if correct else 0

    def transform(self, confidence: float) -> float:
        b = self._bin(confidence)
        if self.n[b] >= self.min_count:
            return self.correct[b] / self.n[b]
        return confidence  # identity until we have enough data

    def ece(self) -> float:
        total = sum(self.n)
        if total == 0:
            return 0.0
        err = 0.0
        for b in range(self.n_bins):
            if self.n[b] == 0:
                continue
            conf_mid = (b + 0.5) / self.n_bins
            acc = self.correct[b] / self.n[b]
            err += (self.n[b] / total) * abs(conf_mid - acc)
        return err


class ResolutionIndex:
    def __init__(self, semantic: Optional[SemanticStore] = None):
        self.events: List[ResolutionEvent] = []
        self.semantic = semantic or SemanticStore()

    # ---- write ----
    def add(self, event: ResolutionEvent) -> None:
        self.events.append(event)

    # ---- retrieve (filter -> similarity, with backoff) ----
    def _filtered(self, sig: Signature, level: int) -> List[ResolutionEvent]:
        """Hierarchical metadata filter; higher level = looser (README §6
        multi-granularity backoff)."""
        out = []
        for e in self.events:
            s = e.signature
            if level == 0:
                ok = (
                    s.lang == sig.lang
                    and s.repo == sig.repo
                    and s.error_type == sig.error_type
                    and s.action_kind == sig.action_kind
                )
            elif level == 1:
                ok = s.lang == sig.lang and s.phrasing_class == sig.phrasing_class
            else:
                ok = s.lang == sig.lang
            if ok:
                out.append(e)
        return out

    def retrieve(
        self,
        sig: Signature,
        now: float,
        k: int = 12,
        sim_threshold: float = SIM_THRESHOLD,
        min_coverage_events: int = 2,
    ) -> List[Tuple[ResolutionEvent, float]]:
        """Return [(event, similarity)] for similar in-scope precedent."""
        scored: List[Tuple[ResolutionEvent, float]] = []
        for level in (0, 1, 2):
            candidates = self._filtered(sig, level)
            scored = []
            for e in candidates:
                sim = CODE_W * cosine(sig.code_vec, e.signature.code_vec) + INTENT_W * cosine(
                    sig.intent_vec, e.signature.intent_vec
                )
                if sim >= sim_threshold:
                    scored.append((e, sim))
            if len(scored) >= min_coverage_events:
                break  # enough at this altitude; stop backing off
        scored.sort(key=lambda t: t[1], reverse=True)
        return scored[:k]

    # ---- confidence from evidence (README §8) ----
    def confidence(
        self, retrieved: List[Tuple[ResolutionEvent, float]], now: float
    ) -> Tuple[float, Optional[str], float, float]:
        """Returns (raw_confidence, winning_resolution, agreement, coverage)."""
        if not retrieved:
            return 0.0, None, 0.0, 0.0
        weights: Dict[str, float] = defaultdict(float)
        total = 0.0
        for e, sim in retrieved:
            recency = 0.5 ** ((now - e.ts) / RECENCY_HALFLIFE)
            w = sim * recency * e.weight_base
            weights[e.resolution] += w
            total += w
        if total <= 0:
            return 0.0, None, 0.0, 0.0
        winner = max(weights.items(), key=lambda kv: kv[1])
        agreement = winner[1] / total                       # do precedents concur?
        coverage = 1.0 - math.exp(-total / COVERAGE_TAU)    # is there enough?
        raw_conf = agreement * coverage
        return raw_conf, winner[0], agreement, coverage

    # ---- persistence (event-sourced; vectors stored as lists) ----
    def save(self, path: str) -> None:
        with open(path, "w") as f:
            for e in self.events:
                s = e.signature
                f.write(
                    json.dumps(
                        {
                            "id": e.id,
                            "ts": e.ts,
                            "resolution": e.resolution,
                            "outcome": e.outcome.value,
                            "source": e.source.value,
                            "confidence_at_decision": e.confidence_at_decision,
                            "corrected_from": e.corrected_from,
                            "session_id": e.session_id,
                            "sig": {
                                "lang": s.lang,
                                "repo": s.repo,
                                "error_type": s.error_type,
                                "action_kind": s.action_kind,
                                "phrasing_class": s.phrasing_class,
                                "files": list(s.files),
                                "symbols": list(s.symbols),
                                "code_vec": s.code_vec.tolist() if s.code_vec is not None else None,
                                "intent_vec": s.intent_vec.tolist() if s.intent_vec is not None else None,
                                "raw_request": s.raw_request,
                                "raw_diff": s.raw_diff,
                            },
                        }
                    )
                    + "\n"
                )
