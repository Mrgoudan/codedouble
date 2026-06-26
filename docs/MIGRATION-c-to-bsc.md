# C → BSC Migration — Auto-Transformation at Scale

Separate from the code-double design ([README.md](../README.md)). Captures the side discussion on
bulk-porting a large C codebase to **BiSheng C (BSC)** under complex house rules.

> **This is the *inverse* of the double: here you TRAIN; the double RETRIEVES.** Same person, opposite
> strategy — because the problem shapes are opposite (below).

## The task

- Auto-transform a large C codebase → BSC, applying **complex house rules** (ownership/borrow
  annotations, safe-zone idioms, team conventions).
- **Asset:** ~1M lines already in good, house-rule-compliant BSC — but **monolingual** (no C↔BSC pairs).
- **Goal:** port new C → BSC that is fluent, correct, *and* house-rule-compliant, at scale.

## Why it's the inverse of the double

| | Migration (C→BSC) | The double |
|---|---|---|
| Task | narrow, **stable** transpilation | broad, **drifting** preferences |
| Data | large corpus (good BSC) | sparse, per-user events |
| Oracle | **present** — compiler + tests (+ linter) | absent (what the user *meant*) |
| Scope | one shared model | per-user |
| → | **TRAIN** (DAPT → SFT → RLVR) | **RETRIEVE**, don't train |

## Two ground-truth checkers — and the gap that forces training

- **Compiler + tests** catch **correctness** → present & queryable → run a **closed loop**
  (generate → compile → on fail, retry with the error).
- **But the compiler is BLIND to house rules** — wrong-style-but-valid BSC compiles and passes tests.
  So correctness has a free checker; **conventions do not**. → you need a **house-rule linter** to make
  compliance verifiable.

## Key conclusion: house rules *require* training

Two independent reasons in-context (RAG/prompt) can't reliably enforce house rules:
1. **In-context rule-following is unreliable** — PrefEval (README §12): preference-following collapses
   **<10% by ~10 turns**. At 1M lines, even 90% adherence = **100k scattered violations**.
2. **The compiler can't catch the violations** (they compile) — so the compile-loop backstop does
   nothing for conventions.

→ **DAPT/SFT on the compliant corpus internalizes the rules as the model's *default output*** (RAG
only *shows* them; training *bakes them in*). Pair with a **house-rule linter-loop** to enforce the
residual. Mirror of correctness: *correctness = SFT + compiler-loop; house rules = SFT + linter-loop.*

## Pattern-matching has two halves — only one is free

`recognize C pattern` → `emit BSC pattern`. **Source half is free** (C is everywhere in pretraining).
**Target half is the niche gap** (BSC is not in pretraining) — and it's exactly what DAPT fills.
RAG + compile-loop close the *fluency/correctness* part of the gap cheaply; they do **not** close the
*house-rule* part (see above).

## Training recipe (monolingual BSC, no pairs)

1. **DAPT** on the BSC corpus → BSC fluency + house-rule style. No pairs. **Biggest, most reliable gain.**
2. **Back-translation** → manufacture pairs: lower real BSC → C (the easy direction). Possibly
   near-free via the BSC toolchain's **source-to-source rewrite to C**. Caveat: toolchain-lowered C is
   *machine-desugared*, not human-style → also do **LLM** back-translation for distribution match.
3. **SFT** on the synthetic (C→BSC) pairs → the mapping.
4. **RLVR** (RL, verifiable reward) against **compiler + tests + lint** → correctness + house-rule
   compliance, no human labels. The expensive, optional, highest-effort step.
5. **+ RAG** over the corpus for long-tail idioms.
6. **+ compile-loop + lint-loop + escalate** the hard residue.

All via **LoRA / QLoRA** (cheap). Filter the corpus to known-good (compiles + lint-clean) first.

## Models

- **Train:** **Codestral** (22B, 256K ctx, code-specialized, open, self-hostable) via LoRA — bulk path.
- **Escalate hard residue:** **GLM-5.1** (754B MoE, #1 SWE-bench Pro, open) or Claude — **untrained**.
- **Retrieval:** code-aware embedder (CodeBERT / UniXcoder, or modern) + optional cross-encoder rerank.
- **Don't train the 754B** — too expensive; use it as the escalation tier.

## Hardware (rent, don't buy)

| Phase | GPU | Cost |
|---|---|---|
| DAPT / SFT — **QLoRA** | 1× 24–48 GB (4090 / A6000 / L40S) | ~$50, hours |
| DAPT / SFT — LoRA bf16 | 1× 80 GB (A100/H100) | tens of $ |
| RLVR | 1 node (4–8 GPUs) | hundreds–low-thousands |
| Inference (4-bit) | 1× 24 GB | cheap |

Rent a **private** GPU (open-weight Codestral) — proprietary C never leaves; don't use a hosted
fine-tune API.

## Discipline

- **Measure the no-train baseline first** (strong model + RAG + compile-loop) on held-out C → compile
  + test + lint → pass-rate. For *fluency/correctness* it may suffice.
- **But house-rule adherence specifically needs training + a linter** — the baseline cannot reliably
  do it (above). Measure before/after with the same eval.

## Glossary

DAPT = Domain-Adaptive Pretraining · SFT = Supervised Fine-Tuning · RLVR = RL from Verifiable Rewards ·
RLHF = RL from Human Feedback · LoRA/QLoRA = (Quantized) Low-Rank Adaptation · RAG = Retrieval-Augmented
Generation · ANN = Approximate Nearest Neighbor · FIM = Fill-In-the-Middle · MoE = Mixture of Experts ·
BERT = Bidirectional Encoder Representations from Transformers · back-translation = synthesize source
from target · BSC = BiSheng C.

## BSC tooling available (this environment)

`c-to-bsc` (translation), `bsc-errors` (compiler diagnostics for the fix-loop), `bsc-compile`
(source-to-source rewrite — possible near-free back-translation), `bsc-ownership` / `bsc-borrowing`
(house-rule semantics).
