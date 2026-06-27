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

To pressure-test toward reality, swap the defaults behind the same interfaces:
- `HashingEmbedder` → a real code embedder (CodeBERT / a sentence-transformer)
- `RuleBasedExtractor` → an `LLMExtractor` calling Mistral/Claude
- the simulated user → a logger riding a real coding agent (the actual §10 step)

Then re-measure the §8 curve on real overrides/reverts. If it bends, the idea
has legs; if it doesn't, no model quality saves it.
