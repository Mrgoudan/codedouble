# Model Spec — BERT + Mistral in the Code Double

Companion to [README.md](../README.md) §11. Models are **swappable plumbing**; the index is the moat
(§5). Specs current as of 2026-06 — verify before quoting.

## Slots (which model = which organ)

| Slot | Model | Role |
|---|---|---|
| **Retrieval / `situation_signature`** | BERT-family code embedder | *recall* — what past decisions are relevant |
| **Monitor reasoning** | Mistral (Codestral / Mixtral / Large) — **frozen** | *judgment* — resolve / estimate VOI / classify |
| **Executor** | any model (external to the double) | does the coding |

Both double-side models are **frozen**; all learning lives in the index/persona, never in weights (§7).

## BERT — the recall organ

- Encoder → embeddings, not generation. Use a **code-aware** variant (CodeBERT / GraphCodeBERT /
  UniXcoder) or a modern embedder (BGE / E5 / GTE / jina-code / Voyage-code).
- Two roles: **bi-encoder** (fast ANN recall) + optional **cross-encoder reranker** (precise top-k).
- The one OK fine-tune: **domain-adapt the embedder once** on "are these two coding situations
  similar?" — shared, non-personal, retrieval-only (distinct from the no-fine-tune rule for the
  reasoner/preferences).
- Caps everything: if retrieval can't tell two situations apart, the reasoner never sees the right
  precedent. Validate clustering first (§10).

## Mistral — the judgment organ (current lineup)

- **Codestral** — 22B code model, **256K context**, 80+ languages, **86.6% HumanEval**, SOTA for
  fill-in-the-middle. The default reasoner for code; cheap to self-host.
- **Devstral 2** — agentic-coding model (multi-step tool use).
- **Mixtral (MoE)** — only a few experts fire per token → cheap inference for the always-on monitor.
- **Mistral Large / Medium 3.5** — strongest general; for the hard calibration call.
- **Why Mistral:** open weights → **self-host → data/IP never leaves** (the B2B lever, §9);
  function calling / JSON mode → emit structured `resolution_event`s and VOI estimates.
- **Caveat:** open-weight models lag frontier on subtle ask-vs-act calibration → tier it.

## Interaction — the RAG loop, both phases

```
ADVISE (hot path, per decision):
  signature ─BERT embed─► ANN top-k ─(BERT cross-encoder rerank)─► MISTRAL reason ─► resolution / VOI
LEARN (reflection, session end):
  raw log ─MISTRAL distill + classify─► preference statements ─BERT embed─► back into the index
```

BERT = "what's relevant"; Mistral = "what to do about it." At advise time BERT feeds Mistral; at
reflect time Mistral feeds BERT.

## Reasoner tiering

| Task | Model |
|---|---|
| signature prep, distillation, correction-vs-iteration classification, rerank | Codestral / Mixtral (cheap) |
| the ask-vs-act calibration verdict + VOI | Mistral Large *or* a frontier model |

## Decision points

1. Models are swappable; **the index is the asset** — don't over-invest in model choice.
2. The one strategic lever is **self-hosting** (Mistral + self-hosted vector store → data stays
   in-house → serves B2B/IP, §9).
3. Frontier-vs-open for the calibration verdict is an **empirical** question — bake off on the §8
   metric, don't decide on vibes.

Sources: [Codestral (Mistral)](https://mistral.ai/news/codestral/) ·
[Mistral models overview](https://docs.mistral.ai/models/overview)
