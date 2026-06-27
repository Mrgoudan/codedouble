"""The extractor (Mistral slot, README §6).

Turns a raw "moment" (the request, diff, error, files…) into a `Signature`:
infers the structured fields and embeds the fuzzy parts. This is the
inference step the design flags as lossy/fallible — the make-or-break is
whether these signatures put "the same kind of situation" near each other.

The default `RuleBasedExtractor` uses cheap deterministic rules so it runs
with no LLM. A real `LLMExtractor` would call Mistral/Claude to infer
`phrasing_class`, `error_type`, `action_kind`, and the interpretation space —
same interface, better extraction.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from typing import Optional

from .embedder import Embedder
from .types import Signature

# request phrasing -> class (README §6: "fix it" / "clean this up" / "make it like X")
_PHRASING = [
    (re.compile(r"\b(clean|tidy|refactor|simplify|lint)\b"), "clean-up"),
    (re.compile(r"\b(fix|repair|bug|broken|fails?|error)\b"), "fix-it"),
    (re.compile(r"\b(like|same as|match|mirror|consistent with)\b"), "make-like-X"),
    (re.compile(r"\b(add|implement|create|new|support)\b"), "add-feature"),
    (re.compile(r"\b(remove|delete|drop|strip)\b"), "remove"),
]

_ERROR = [
    (re.compile(r"null|none\b|nonetype|nullpointer|nil"), "null-deref"),
    (re.compile(r"type ?error|expected .* got|not assignable|mismatch"), "type-mismatch"),
    (re.compile(r"index|out of range|bounds"), "index-error"),
    (re.compile(r"timeout|timed out|deadline"), "timeout"),
    (re.compile(r"assert|test .*fail"), "test-failure"),
]


class SignatureExtractor(ABC):
    @abstractmethod
    def extract(self, moment: dict) -> Signature:
        ...


class RuleBasedExtractor(SignatureExtractor):
    def __init__(self, embedder: Embedder):
        self.embedder = embedder

    def _phrasing_class(self, request: str) -> str:
        r = (request or "").lower()
        for rx, label in _PHRASING:
            if rx.search(r):
                return label
        return "other"

    def _error_type(self, error: str) -> Optional[str]:
        e = (error or "").lower()
        if not e:
            return None
        for rx, label in _ERROR:
            if rx.search(e):
                return label
        return "other-error"

    def _action_kind(self, moment: dict) -> str:
        if moment.get("action_kind"):
            return moment["action_kind"]
        diff = (moment.get("diff") or "").lower()
        req = (moment.get("request") or "").lower()
        blob = diff + " " + req
        for kw, kind in (
            ("rename", "rename"),
            ("delete", "delete"),
            ("remove", "delete"),
            ("refactor", "refactor"),
            ("add", "add"),
        ):
            if kw in blob:
                return kind
        return "edit"

    def extract(self, moment: dict) -> Signature:
        request = moment.get("request", "")
        diff = moment.get("diff", "")
        error = moment.get("error", "")
        files = tuple(moment.get("files", ()) or ())
        symbols = tuple(moment.get("symbols", ()) or ())

        code_text = " ".join(symbols) + " " + diff + " " + " ".join(files)
        intent_text = request + " " + (error or "")

        return Signature(
            lang=moment.get("lang", ""),
            repo=moment.get("repo", ""),
            error_type=self._error_type(error),
            action_kind=self._action_kind(moment),
            phrasing_class=self._phrasing_class(request),
            files=files,
            symbols=symbols,
            code_vec=self.embedder.embed(code_text),
            intent_vec=self.embedder.embed(intent_text),
            raw_request=request,
            raw_diff=diff,
            interpretation_space=list(moment.get("interpretation_space", []) or []),
        )
