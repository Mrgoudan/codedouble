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
import re
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np

from .embedder import cosine
from .types import (
    Outcome,
    PreferenceRule,
    ResolutionEvent,
    Signature,
    Source,
    is_negative,
)

# tuning knobs (prototype defaults)
RECENCY_HALFLIFE = 50.0   # ts units after which an event's weight halves
COVERAGE_TAU = 3.0        # weight at which coverage-factor ~ 0.63 (saturates)
SIM_THRESHOLD = 0.45      # min combined score to count as "similar"
CODE_W, INTENT_W = 0.5, 0.5
SEM_W, LEX_W = 0.5, 0.5    # hybrid: semantic (vectors) + lexical (identifier overlap)
K_MIN, K_MAX = 5, 20       # determinacy-adaptive retrieval scope


def _sig_tokens(sig: Signature) -> set:
    """Lexical view of a signature: identifiers/keywords from request, symbols,
    file names, error/phrasing, and a slice of the diff. snake_case and camelCase
    are split so 'flag_skip_reason' / 'flagSkipReason' match 'skip'."""
    parts = [
        sig.raw_request or "",
        " ".join(sig.symbols or ()),
        " ".join(f.rsplit("/", 1)[-1] for f in (sig.files or ())),
        sig.error_type or "", sig.phrasing_class or "", sig.action_kind or "",
        (sig.raw_diff or "")[:600],
    ]
    text = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", " ".join(parts))   # split camelCase
    return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if len(w) >= 2}


def _determinacy(sig: Signature) -> float:
    """How *determined* the intent is (0=vague, 1=precise). Drives scope:
    vague -> gather more & loosen; precise -> stay tight."""
    d = 0.25
    if sig.error_type:
        d += 0.30
    if sig.symbols:
        d += 0.20
    if sig.files:
        d += 0.10
    if sig.phrasing_class in ("fix-it", "make-like-X", "remove", "add-feature"):
        d += 0.15
    if len(_sig_tokens(sig)) >= 6:
        d += 0.10
    return max(0.0, min(1.0, d))


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
            elif level == 2:
                ok = s.lang == sig.lang
            else:
                ok = True  # last resort: ignore metadata, rely on similarity
            if ok:
                out.append(e)
        return out

    def _event_tokens(self, e: ResolutionEvent) -> set:
        cache = getattr(e, "_lextok", None)
        if cache is None:
            cache = _sig_tokens(e.signature)
            try:
                e._lextok = cache       # memoise on the event (no __slots__)
            except Exception:
                pass
        return cache

    def _idf(self) -> Dict[str, float]:
        """Inverse doc frequency over events — rare identifiers weigh more."""
        n = max(1, len(self.events))
        df: Dict[str, int] = defaultdict(int)
        for e in self.events:
            for t in self._event_tokens(e):
                df[t] += 1
        return {t: math.log(1.0 + n / c) for t, c in df.items()}

    def retrieve(
        self,
        sig: Signature,
        now: float,
        k: Optional[int] = None,
        sim_threshold: float = SIM_THRESHOLD,
        min_coverage_events: int = 2,
    ) -> List[Tuple[ResolutionEvent, float]]:
        """Intent-aware HYBRID retrieval (README §6): blend semantic vectors with
        lexical identifier overlap (idf-weighted), and size the scope to how
        *determined* the intent is — vague intents gather more and loosen the bar,
        precise intents stay tight."""
        d = _determinacy(sig)
        k_dyn = k if k is not None else int(round(K_MIN + (1.0 - d) * (K_MAX - K_MIN)))
        thr = max(0.15, sim_threshold - (1.0 - d) * 0.12)     # loosen for vague intents
        qtok = _sig_tokens(sig)
        idf = self._idf()
        qsum = sum(idf.get(t, 0.0) for t in qtok) or 1.0
        qc = sig.code_vec
        q_has_code = qc is not None and float(np.dot(qc, qc)) > 1e-9
        scored: List[Tuple[ResolutionEvent, float]] = []
        for level in (0, 1, 2, 3):
            scored = []
            for e in self._filtered(sig, level):
                ci = cosine(sig.intent_vec, e.signature.intent_vec)
                ec = e.signature.code_vec
                if q_has_code and ec is not None and float(np.dot(ec, ec)) > 1e-9:
                    sem = CODE_W * cosine(qc, ec) + INTENT_W * ci
                else:
                    sem = ci
                sem = max(0.0, min(1.0, sem))
                etok = self._event_tokens(e)
                lex = (sum(idf.get(t, 0.0) for t in (qtok & etok)) / qsum) if qtok else 0.0
                score = SEM_W * sem + LEX_W * lex            # hybrid
                if score >= thr:
                    scored.append((e, score))
            if len(scored) >= min_coverage_events:
                break       # enough at this altitude; stop backing off
        scored.sort(key=lambda t: t[1], reverse=True)
        return scored[:k_dyn]

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

    # ---- known-bad signal: did the user REJECT this pattern before? -----------
    def polarity(
        self, retrieved: List[Tuple[ResolutionEvent, float]], now: float
    ) -> Tuple[float, float]:
        """Weighted fraction of NEGATIVE precedent (override/revert/interrupt) and
        its coverage. High + covered => a *known-bad* pattern (send back). NOT
        about ignorance — no precedent returns (0, 0), i.e. allow & observe."""
        if not retrieved:
            return 0.0, 0.0
        bad = total = 0.0
        for e, sim in retrieved:
            recency = 0.5 ** ((now - e.ts) / RECENCY_HALFLIFE)
            w = sim * recency * e.weight_base
            total += w
            if is_negative(e.outcome):
                bad += w
        if total <= 0:
            return 0.0, 0.0
        return bad / total, 1.0 - math.exp(-total / COVERAGE_TAU)

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

    @classmethod
    def load(cls, path: str) -> "ResolutionIndex":
        """Reconstruct an index (with embeddings) from a save() file — so the hot
        path can reuse cached vectors instead of re-embedding the whole log."""
        idx = cls()
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                s = d["sig"]
                cv = s.get("code_vec")
                iv = s.get("intent_vec")
                sig = Signature(
                    lang=s["lang"], repo=s["repo"], error_type=s["error_type"],
                    action_kind=s["action_kind"], phrasing_class=s["phrasing_class"],
                    files=tuple(s.get("files", ())), symbols=tuple(s.get("symbols", ())),
                    code_vec=np.asarray(cv, dtype=np.float32) if cv else None,
                    intent_vec=np.asarray(iv, dtype=np.float32) if iv else None,
                    raw_request=s.get("raw_request", ""), raw_diff=s.get("raw_diff", ""),
                )
                idx.add(ResolutionEvent(
                    id=d["id"], ts=d["ts"], signature=sig, resolution=d["resolution"],
                    outcome=Outcome(d["outcome"]), source=Source(d["source"]),
                    confidence_at_decision=d["confidence_at_decision"],
                    corrected_from=d.get("corrected_from"), session_id=d.get("session_id"),
                ))
        return idx
