"""The matcher (BERT slot, docs/SPEC-models.md).

`Embedder` is the pluggable interface. The default `HashingEmbedder` needs no
model and no network — it hashes token + bigram features into a fixed-dim,
L2-normalized vector, so identical text maps to identical vectors and similar
text maps to nearby ones. Deterministic, fast, runs anywhere.

To use a real code-aware embedder (CodeBERT / UniXcoder / a sentence-transformer),
implement `Embedder.embed` over that model and pass it in — nothing else changes.
"""

from __future__ import annotations

import hashlib
import re
from abc import ABC, abstractmethod
from typing import List

import numpy as np

_TOKEN = re.compile(r"[a-z0-9_]+")


class Embedder(ABC):
    dim: int

    @abstractmethod
    def embed(self, text: str) -> np.ndarray:
        ...


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    if a is None or b is None:
        return 0.0
    return float(np.dot(a, b))  # vectors are L2-normalized on creation


class HashingEmbedder(Embedder):
    """Dependency-free feature-hashing embedder (the no-model default)."""

    def __init__(self, dim: int = 256):
        self.dim = dim

    def _tokens(self, text: str) -> List[str]:
        toks = _TOKEN.findall((text or "").lower())
        bigrams = [a + "_" + b for a, b in zip(toks, toks[1:])]
        return toks + bigrams

    def embed(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        for tok in self._tokens(text):
            h = int.from_bytes(hashlib.md5(tok.encode()).digest()[:8], "little")
            idx = h % self.dim
            sign = 1.0 if (h >> 63) & 1 else -1.0  # signed hashing reduces collisions
            v[idx] += sign
        n = float(np.linalg.norm(v))
        if n > 0:
            v /= n
        return v
