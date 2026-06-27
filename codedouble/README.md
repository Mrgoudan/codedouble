# codedouble — runnable prototype

A working, dependency-light implementation of the Self-Learning Code Double
(see [../README.md](../README.md)). It realizes the signature → index → 2×2 gate
→ reflection loop we speced, plus the §10 "first artifact" (a logger + the §8
metric) — and it runs on plain Python 3.8 + numpy, no GPU, no network, no LLM.

## Run

```bash
python3 -m codedouble.demo        # end-to-end demo on a simulated user
python3 tests/test_codedouble.py  # test suite (stdlib unittest)
```

## What the demo shows

A simulated developer with stable preferences. As the index fills, the double
**asks less, acts more, and stays right** — the §8 override-rate on its
*confident, silent* decisions falls to the irreducible floor (the rare
"changed my mind" events), while accuracy holds and calibration error (ECE)
drops:

```
§8 rate  early=0.06  ->  late=0.03      (override rate on confident-silent)
ask-rate early=0.10  ->  late=0.00      (asks less as it learns)
accuracy early=0.95  ->  late=0.97
```

## How it maps to the design

| File | Design (README §) |
|---|---|
| `types.py` | `Signature`, `ResolutionEvent`, labels (§6); 2×2 actions (§4) |
| `embedder.py` | the matcher / BERT slot (SPEC-models) — default = no-model hashing embedder |
| `signature.py` | the extractor / Mistral slot (§6) — default = rule-based, LLM-pluggable |
| `index.py` | episodic + vector + semantic stores; **confidence from coverage × agreement** (§8); histogram calibrator |
| `double.py` | the external monitor + the ask/act gate + escalation stub (§2, §3, §4) |
| `reflect.py` | faithful-record → session-end reflection: credit assignment + distillation (§6) |
| `metrics.py` | the §8 metric (override-rate on confident-silent) + ECE + the curve |
| `demo.py` | the simulated-user driver |

Key design properties it demonstrates:
- **Frozen models, learning in retrieval** — nothing is trained; the index grows.
- **Confidence from evidence, not the model's gut** — `coverage × agreement`.
- **Transparent** — every decision carries its `retrieved` precedent and a rationale.
- **The 2×2 gate** — asks only when under-determined *and* hard to undo.
- **Calibrated reflection** — a resolution becomes a rule only with enough support.

## The honest caveat (read this)

This validates the **mechanism**, not the thesis. The simulated user has a
*knowable, stable* preference function — exactly what a real developer does
**not** hand you. A falling curve here proves the plumbing works; it says
**nothing** about whether real behavioral `situation_signature`s actually
cluster into consistent resolutions. That clustering is the make-or-break
(README §10), and only real developers can answer it.

## Real models (the model is in)

Real backends live in [backends.py](backends.py), behind the same interfaces.
Switch with `CODEDOUBLE_BACKEND`:

```bash
python3 -m codedouble.demo                          # default — no model (hashing + rules)
CODEDOUBLE_BACKEND=st      python3 -m codedouble.demo   # local sentence-transformers (CPU, no key)
CODEDOUBLE_BACKEND=mistral python3 -m codedouble.demo   # mistral-embed + LLM-inferred fields (needs MISTRAL_API_KEY)
```

| Backend | Embedder | Extractor fields | Needs |
|---|---|---|---|
| `default` | `HashingEmbedder` (no model) | `RuleBasedExtractor` | nothing |
| `st` | `STEmbedder` (sentence-transformers, CPU) | rule-based | `pip install sentence-transformers` |
| `mistral` | `MistralEmbedder` (`mistral-embed`) | `LLMExtractor` (Mistral chat) | `MISTRAL_API_KEY` + network |

`MistralClient` is stdlib-`urllib` only (no `requests`). `LLMExtractor` degrades
to rule-based per field if the model is unreachable or returns garbage, and is
unit-tested offline via `FakeLLM`.

Verified here: the **`st` backend runs end-to-end on CPU** (MiniLM, 384-dim;
similar/dissimilar cosine 0.81 vs 0.08; same falling §8 curve, ECE 0.020). The
`mistral` path is wired and offline-tested; it needs a key + network to run live.

## The make-or-break is still open

Swapping in real models does **not** close the real question — the simulated
user does. The honest §10 step is: replace the simulated user with a **logger
riding a real coding agent**, collect real overrides/reverts, and re-measure the
§8 curve. If it bends on real behavior, the idea has legs; if it doesn't, no
embedder or LLM saves it.
