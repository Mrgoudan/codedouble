"""Real model backends — the BERT (embedder) and Mistral (extractor) slots.

These plug in behind the same `Embedder` / `SignatureExtractor` interfaces the
no-model defaults use, so swapping them changes nothing else (README §11:
models are swappable; the index is the moat).

Nothing here is imported at package load unless you ask for it — the defaults
stay dependency-free. Backends degrade gracefully: a missing key / package / net
raises a clear error, and `LLMExtractor` falls back to rule-based per field.

  - MistralClient   thin stdlib-urllib wrapper (chat + embeddings), no `requests`
  - MistralEmbedder embeddings via `mistral-embed`            (the matcher slot)
  - STEmbedder      local sentence-transformers (no API/GPU)  (the matcher slot)
  - LLMExtractor    LLM-inferred signature fields             (the extractor slot)
  - FakeLLM         canned-JSON completer for offline tests
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from typing import Callable, List, Optional

import numpy as np

from .embedder import Embedder
from .signature import RuleBasedExtractor, SignatureExtractor
from .types import Signature


def _l2(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v


# --------------------------------------------------------------------------- #
# Mistral HTTP client (stdlib only)
# --------------------------------------------------------------------------- #
class MistralClient:
    BASE = "https://api.mistral.ai/v1"

    def __init__(self, api_key: Optional[str] = None, timeout: float = 60.0):
        self.api_key = api_key or os.environ.get("MISTRAL_API_KEY")
        self.timeout = timeout
        if not self.api_key:
            raise RuntimeError(
                "MISTRAL_API_KEY not set. Export it, or use the no-model defaults "
                "(HashingEmbedder + RuleBasedExtractor)."
            )

    def _post(self, path: str, payload: dict) -> dict:
        req = urllib.request.Request(
            f"{self.BASE}/{path}",
            data=json.dumps(payload).encode(),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            return json.loads(resp.read().decode())

    def chat(self, prompt: str, model: str = "mistral-small-latest",
             system: Optional[str] = None, json_mode: bool = True) -> str:
        msgs = []
        if system:
            msgs.append({"role": "system", "content": system})
        msgs.append({"role": "user", "content": prompt})
        payload = {"model": model, "messages": msgs, "temperature": 0}
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        out = self._post("chat/completions", payload)
        return out["choices"][0]["message"]["content"]

    def embed(self, texts: List[str], model: str = "mistral-embed") -> List[List[float]]:
        out = self._post("embeddings", {"model": model, "input": texts})
        return [d["embedding"] for d in out["data"]]


# --------------------------------------------------------------------------- #
# Embedder backends (the matcher / BERT slot)
# --------------------------------------------------------------------------- #
class MistralEmbedder(Embedder):
    """Embeddings via Mistral's `mistral-embed` (1024-dim). API key + net required."""

    def __init__(self, client: Optional[MistralClient] = None, model: str = "mistral-embed"):
        self.client = client or MistralClient()
        self.model = model
        self.dim = 1024

    def embed(self, text: str) -> np.ndarray:
        vec = self.client.embed([text or " "], model=self.model)[0]
        return _l2(np.asarray(vec, dtype=np.float32))


class STEmbedder(Embedder):
    """Local sentence-transformers — no API, no GPU needed (CPU is fine).

    Defaults to a small general model; pass a code-aware model for code
    (e.g. 'flax-sentence-embeddings/st-codesearch-distilroberta-base').
    Requires `pip install sentence-transformers`.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        try:
            from sentence_transformers import SentenceTransformer  # lazy
        except ImportError as e:  # pragma: no cover
            raise RuntimeError(
                "sentence-transformers not installed. `pip install sentence-transformers` "
                "or use HashingEmbedder / MistralEmbedder."
            ) from e
        self._model = SentenceTransformer(model_name)
        self.dim = int(self._model.get_sentence_embedding_dimension())

    def embed(self, text: str) -> np.ndarray:
        v = self._model.encode(text or " ", normalize_embeddings=True)
        return np.asarray(v, dtype=np.float32)


# --------------------------------------------------------------------------- #
# LLM extractor (the Mistral / reasoner slot for signatures)
# --------------------------------------------------------------------------- #
Completer = Callable[[str], str]  # prompt -> JSON string

_EXTRACT_SYSTEM = (
    "You label a software-engineering 'moment' so similar situations can be "
    "retrieved later. Return ONLY a JSON object with keys: "
    "phrasing_class (one of: clean-up, fix-it, make-like-X, add-feature, remove, other), "
    "error_type (a short slug or null), "
    "action_kind (one of: edit, delete, rename, refactor, add), "
    "interpretation_space (array of 1-3 short plausible readings of the request)."
)


def _moment_prompt(moment: dict) -> str:
    return (
        "Label this moment:\n"
        f"request: {moment.get('request','')}\n"
        f"error: {moment.get('error','')}\n"
        f"diff: {moment.get('diff','')[:800]}\n"
        f"files: {list(moment.get('files',()) )}\n"
        f"lang: {moment.get('lang','')}\n"
    )


def _parse_json(s: str) -> dict:
    try:
        return json.loads(s)
    except Exception:
        m = re.search(r"\{.*\}", s, re.DOTALL)  # tolerate prose around the JSON
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                return {}
        return {}


class LLMExtractor(SignatureExtractor):
    """Infer the structured signature fields with an LLM; embed the fuzzy parts.

    `complete` maps a prompt -> JSON string (e.g. `MistralClient().chat`). Any
    field the LLM omits or malforms falls back to the rule-based extractor, so a
    flaky model degrades rather than crashes.
    """

    def __init__(self, complete: Completer, embedder: Embedder,
                 fallback: Optional[SignatureExtractor] = None):
        self.complete = complete
        self.embedder = embedder
        self.fallback = fallback or RuleBasedExtractor(embedder)

    def extract(self, moment: dict) -> Signature:
        base = self.fallback.extract(moment)  # fields + embeddings as a floor
        try:
            data = _parse_json(self.complete(_moment_prompt(moment)))
        except (urllib.error.URLError, RuntimeError, KeyError):
            return base  # network/key failure -> rule-based result

        base.phrasing_class = data.get("phrasing_class") or base.phrasing_class
        if "error_type" in data:
            base.error_type = data.get("error_type") or base.error_type
        base.action_kind = data.get("action_kind") or base.action_kind
        isp = data.get("interpretation_space")
        if isinstance(isp, list) and isp:
            base.interpretation_space = [str(x) for x in isp][:3]
        return base


class FakeLLM:
    """Offline completer for tests — returns canned JSON, no network."""

    def __init__(self, payload: dict):
        self.payload = payload
        self.calls = 0

    def __call__(self, prompt: str) -> str:
        self.calls += 1
        return json.dumps(self.payload)


# --------------------------------------------------------------------------- #
# convenience factory: the full Mistral extractor (embed + chat)
# --------------------------------------------------------------------------- #
class OllamaClient:
    """Local LLM via Ollama (no API key, no cloud). Talks to a running
    `ollama serve` at OLLAMA_HOST (default http://localhost:11434)."""

    def __init__(self, model: Optional[str] = None, host: Optional[str] = None, timeout: float = 120.0):
        self.host = (host or os.environ.get("OLLAMA_HOST", "http://localhost:11434")).rstrip("/")
        # explicit arg wins over env, env over the built-in default
        self.model = model or os.environ.get("CODEDOUBLE_OLLAMA_MODEL") or "mistral"
        self.timeout = timeout

    def chat(self, prompt: str, system: Optional[str] = None, json_mode: bool = True) -> str:
        msgs = []
        if system:
            msgs.append({"role": "system", "content": system})
        msgs.append({"role": "user", "content": prompt})
        payload = {"model": self.model, "messages": msgs, "stream": False,
                   "options": {"temperature": 0}}
        if json_mode:
            payload["format"] = "json"
        req = urllib.request.Request(
            self.host + "/api/chat", data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            return json.loads(resp.read().decode())["message"]["content"]


def list_ollama_models(host: Optional[str] = None) -> List[str]:
    """Names of models a local `ollama serve` has pulled (empty if unreachable)."""
    host = (host or os.environ.get("OLLAMA_HOST", "http://localhost:11434")).rstrip("/")
    try:
        with urllib.request.urlopen(host + "/api/tags", timeout=5) as resp:
            data = json.loads(resp.read().decode())
        return [m["name"] for m in data.get("models", [])]
    except Exception:
        return []


def _read_env_field(path, field):
    """Read FIELD=value from a .env-style file (so an API key need not be duplicated)."""
    if not path or not field:
        return ""
    try:
        for line in open(os.path.expanduser(path)):
            line = line.strip()
            if line.startswith(field + "="):
                return line.split("=", 1)[1].strip().strip("\"'")
    except Exception:
        pass
    return ""


def _llm_config():
    """Resolve the LLM 'port' (base_url, [keys], model) from env then
    ~/.codedouble/llm.json. api_key_field may be a LIST -> key ROTATION (try the next
    on 429/auth, like glm_run.sh). The key stays referenced in its .env, not copied."""
    base = os.environ.get("CODEDOUBLE_LLM_BASE_URL", "")
    model = os.environ.get("CODEDOUBLE_LLM_MODEL", "")
    env_key = os.environ.get("CODEDOUBLE_LLM_API_KEY", "")
    keys = [env_key] if env_key else []
    if not base or not keys:
        home = os.environ.get("CODEDOUBLE_HOME") or os.path.join(os.path.expanduser("~"), ".codedouble")
        try:
            d = json.load(open(os.path.join(home, "llm.json")))
            base = base or d.get("base_url", "")
            model = model or d.get("model", "")
            if not keys:
                if d.get("api_key"):
                    keys = [d["api_key"]]
                else:
                    fields = d.get("api_key_field", [])
                    if isinstance(fields, str):
                        fields = [fields]
                    path = d.get("api_key_file", "")
                    keys = [k for k in (_read_env_field(path, f) for f in fields) if k]
        except Exception:
            pass
    return base.rstrip("/"), keys, (model or "gpt-4o-mini")


class OpenAICompatLLM:
    """Any OpenAI-compatible chat endpoint (OpenAI, vLLM, Ollama's /v1, the GLM
    gateway). Filled via CODEDOUBLE_LLM_* env or ~/.codedouble/llm.json. Rotates
    across keys on 429/auth, like glm_run.sh."""

    def __init__(self, base_url=None, keys=None, model=None):
        b, k, m = _llm_config()
        self.base = (base_url or b).rstrip("/")
        self.keys = keys if keys is not None else k
        if isinstance(self.keys, str):
            self.keys = [self.keys]
        self.model = model or m

    def chat(self, prompt: str, system: Optional[str] = None,
             json_mode: bool = False, timeout: float = 90.0) -> str:
        msgs = []
        if system:
            msgs.append({"role": "system", "content": system})
        msgs.append({"role": "user", "content": prompt})
        payload = {"model": self.model, "messages": msgs, "temperature": 0}
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        data = json.dumps(payload).encode()
        last = None
        for key in (self.keys or [""]):
            headers = {"Content-Type": "application/json"}
            if key:
                headers["Authorization"] = f"Bearer {key}"
            req = urllib.request.Request(self.base + "/chat/completions",
                                         data=data, headers=headers, method="POST")
            try:
                with urllib.request.urlopen(req, timeout=timeout) as resp:
                    return json.loads(resp.read().decode())["choices"][0]["message"]["content"]
            except urllib.error.HTTPError as e:
                last = e
                if e.code in (401, 403, 429):     # key exhausted / rate-limited -> rotate
                    continue
                raise
        raise last or RuntimeError("LLM port: all keys failed")


def llm_complete(prompt, system=None, json_mode=False, timeout=90.0) -> str:
    """LLM-FIRST, local-fallback dispatcher for the semantic 'hunting' tasks. Tries
    the configured LLM port (with key rotation); else a local Ollama model; else
    raises (callers then fall back to their cheap heuristic)."""
    base, keys, model = _llm_config()
    if base:
        return OpenAICompatLLM(base, keys, model).chat(prompt, system, json_mode, timeout)
    models = list_ollama_models()
    if models:
        m = os.environ.get("CODEDOUBLE_OLLAMA_MODEL") or models[0]
        return OllamaClient(model=m, timeout=timeout).chat(prompt, system, json_mode)
    raise RuntimeError("no LLM endpoint available (set CODEDOUBLE_LLM_BASE_URL / ~/.codedouble/llm.json, or run ollama)")


def ollama_extractor(embedder: Embedder, model: Optional[str] = None) -> LLMExtractor:
    """LLMExtractor whose reasoning runs on a LOCAL Ollama model; embeddings come
    from the given (local) embedder. Fully offline once the model is pulled."""
    client = OllamaClient(model=model)

    def complete(prompt: str) -> str:
        return client.chat(prompt, system=_EXTRACT_SYSTEM, json_mode=True)

    return LLMExtractor(complete, embedder)


def mistral_extractor(
    client: Optional[MistralClient] = None,
    chat_model: str = "mistral-small-latest",
    embed_model: str = "mistral-embed",
) -> LLMExtractor:
    """Wire LLMExtractor with Mistral chat (field inference) + mistral-embed
    (vectors). Needs MISTRAL_API_KEY + network."""
    client = client or MistralClient()
    emb = MistralEmbedder(client, model=embed_model)

    def complete(prompt: str) -> str:
        return client.chat(prompt, model=chat_model, system=_EXTRACT_SYSTEM, json_mode=True)

    return LLMExtractor(complete, emb)
