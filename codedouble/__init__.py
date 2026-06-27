"""codedouble — a runnable prototype of the Self-Learning Code Double.

This is the §10 "first artifact" plus the signature/index/gate/reflection core
described in README.md and docs/SPEC-models.md. It deliberately uses NO heavy
dependencies (numpy only) so it runs anywhere — the embedder and signature
extractor are pluggable, with dependency-free defaults, and real
BERT/Mistral-style backends can be dropped in behind the same interfaces.

The pieces map 1:1 to the design:
  - Signature / ResolutionEvent  (README §6)            -> types.py
  - BERT (the "matcher")                                 -> embedder.py
  - Mistral (the "extractor")                            -> signature.py
  - episodic + vector + semantic stores, confidence      -> index.py
  - the 2x2 ask/act gate + escalation                    -> double.py
  - faithful record -> session-end reflection            -> reflect.py
  - the §8 metric (override-rate on confident-silent)    -> metrics.py
"""

from .types import (
    Action,
    Decision,
    Outcome,
    PreferenceRule,
    ResolutionEvent,
    Reversibility,
    Signature,
    Source,
    is_negative,
    is_positive,
)
from .embedder import Embedder, HashingEmbedder
from .signature import RuleBasedExtractor, SignatureExtractor
from .index import Calibrator, ResolutionIndex, SemanticStore
from .double import Double
from .reflect import reflect_session
from .backends import (
    FakeLLM,
    LLMExtractor,
    MistralClient,
    MistralEmbedder,
    STEmbedder,
    mistral_extractor,
)
from . import metrics

__all__ = [
    "Action",
    "Decision",
    "Outcome",
    "PreferenceRule",
    "ResolutionEvent",
    "Reversibility",
    "Signature",
    "Source",
    "is_negative",
    "is_positive",
    "Embedder",
    "HashingEmbedder",
    "RuleBasedExtractor",
    "SignatureExtractor",
    "Calibrator",
    "ResolutionIndex",
    "SemanticStore",
    "Double",
    "reflect_session",
    "FakeLLM",
    "LLMExtractor",
    "MistralClient",
    "MistralEmbedder",
    "STEmbedder",
    "mistral_extractor",
    "metrics",
]
