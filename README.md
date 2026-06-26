# The Self-Learning Code Double

A design for an **external double of a developer's judgment** — a *self-learning, reusable,
forceful* agent that learns **when you'd want to be asked versus when to just act**, from what you
actually do, and carries that judgment across every tool and repo you own.

> **The future is a market for doubles.** A double is trained on your behavior and lives *outside*
> any one tool — so it is an asset **you own**, portable across executors, repos, and time. The
> endgame: doubles are **sold** — a top engineer's coding judgment, packaged, bought, and plugged
> into anyone's agent.
>
> **But you must prove it works first.** A marketplace is worthless if the underlying loop doesn't
> actually learn. The gate before everything else is a single empirical number that *nobody has*:
> **does a real human's override-rate fall as the double learns their way?** Earn that number
> first. The marketplace is what it *unlocks* — not what we build first.

This document is the design discussion, not a finished spec. Open questions are marked **[OPEN]**.

---

## TL;DR — the whole design in one screen

- **§0 What a double is** — an external *self-learning · reusable · forceful* double of your
  judgment; must clear two bars: no agent can do it **and** it's a real pain.
- **§1 Core principle** — detect where the agent is confident-but-wrong, route to the cheapest fix
  (repo → history → human); *index, not intelligence*. Two of three sources are commoditizing — the
  double lives in the third.
- **§2 External supervisor** — a *separate* agent watches intent; it cannot be a prompt to the
  executor (61.2% self-reflect vs 69.4% external; RLHF killed self-calibration).
- **§3 Ask · act · preference** — preference *resolves*, ask/act *gates*; every answered ask becomes
  a preference that lets it act next time. That loop is the learning. Escalation: your index → cohort
  → frontier LLM → human.
- **§4 Reversibility** — `git` makes undo ~free, so act-first/revert-wrong beats ask-on-everything;
  be *forceful only* when under-determined **and** irreversible. Correctness *is* convenience once
  you count cleanup.
- **§5 Defensibility** — the moat is the *combination* A+B+C+D (external · ask/act+preference ·
  passive behavior · transparent+measured); nobody has all four. Plus three structural moats:
  git-as-label, per-user flywheel, cross-user network effect.
- **§6 Data layer** — log `resolution_event`s; `situation_signature` is make-or-break; *every user
  intervention (interrupt / correct / reject / revert) is a failure signal* — drive **preventable**
  to zero, make **irreducible** ("I'll know it when I see it") cheap (§4); `git revert` cleanest,
  not primary; never count never-viewed. **Positives matter too** — *confirmed-good* (reviewed
  wholesale accept) is the signal that lowers the threshold. Record faithfully per-decision; reflect
(label + distill) at session end. **Calibrated reflection:** distill a rule only from repeated /
consistent signal — confidently learning the wrong lesson = an automated mask *inside* the loop.
- **§7 Cold start** — never *fully* cold (the repo **and its git history** are a prior); retrieval
  not training → useful from the first event; mine commit history to pre-warm; **configurable cohort
  prior** (clean start / average / bought-expert).
- **§8 Proof gate** — the one number that gates everything: *does a real human's override-rate
  fall?* Nobody has it. Prove it before the marketplace; it later becomes the trust certificate.
- **§9 Path to a market** — prove on top coders → sell B2B (institutional memory) → consumer
  marketplace **last** (a sold double = copyable preset; IP risk).
- **§10 First artifact** — a logger that resolves nothing; check whether signatures cluster before
  building anything.
- **§11 How it fits** — three processes (executor · monitor · capture), one shared index; the index
  *is* the double.
- **§12 Literature** — the field converged on this architecture in 2025–26; the real-human curve is
  the open gap.

---

## 0. What a double is — and the bar it must clear

Four words, each load-bearing — and each a thing no shipped code agent can claim *together*:

- **Self-learning** — learns, from your behavior (overrides, `git revert`), both *what* you'd want
  (**preference**) and *whether to bother you* (**ask vs. act** calibration) — not from hand-written
  rules. The commoditized slice is *static* content prefs (tabs, async); the unclaimed slice is
  *situation-keyed* resolution preference **+** the calibration gate (§3).
- **Reusable** — it is *external* to the executor, so it is portable across models, tools, and
  repos. An asset **you own**, not a feature locked to one vendor.
- **Forceful** — a supervisor with teeth: it can *halt* the executor and *inject* what the executor
  missed — but only where intervention is genuinely worth it.
- **Double** — of *your judgment*, kept **transparent**: every assumption it makes is visible and
  vetoable. Not an opaque twin.

**Positioning:** *not another assistant that writes code — a double that learns your judgment about
when to ask, and that you keep.*

**The bar (two gating criteria).** Every design choice must clear both:
1. **No existing code agent can resolve it** — a genuine unsolved gap, not something Cursor /
   Claude Code / Copilot / Devin already do.
2. **It is a real pain point** — actually felt, not hypothetical.

The intersection is narrow on purpose: the famous pains attract everyone, so the target is
*real-but-under-attacked*. (See §5 for why this design sits in that intersection — and §9 for the
one move that would push it back out.)

---

## 1. The core principle

> **Detect where the agent is confident but shouldn't be, and either resolve it from an index or
> escalate to the human — exhausting cheap provable resolution before spending the human's
> attention.**

The agent under-uses three sources of truth because it doesn't recognize its own ignorance:

1. **The codebase** — under-used because it won't search. → a monitor *injects* from the index.
2. **The user's prior decisions** — under-used because it doesn't recall. → *resolve silently* from history.
3. **The user themselves** — under-used because it doesn't know what to ask. → a monitor *queries* the human.

One meta-pattern governs all three: **an external monitor that detects the agent's unrecognized
ignorance and routes it to the cheapest source that can fix it.** The human is the last resort — the
most expensive index — but for irreducible ambiguity, the only one with the answer.

**Index, not intelligence.** The double is *retrieval over a logged index*, not a trained model of
the person. This is what keeps it transparent ("I assumed X because last week you…"), dodges the
data-volume ceiling, and avoids "precedent wearing a mask."

**Two of these three are already commoditizing — the double lives in the third.** Modern agents
*do* grep the repo (source 1) and *do* store content preferences via rules/memory files (source 3,
weakly). What survives criterion 1 is **source 2 in its hard form**: learning *your ask/act
calibration* from *passive behavior*. Everything below is built around that.

---

## 2. The double is an external supervisor

It is not the executor in critic mode. It is a **separate agent** watching the executor's declared
intent before it acts. Two monitors, both detecting *false confidence*:

| | Code monitor | User monitor |
|---|---|---|
| **The gap** | agent doesn't know a helper exists | agent doesn't know what to ask the user |
| **Why search fails** | won't grep for what it doesn't suspect | won't ask about intent it doesn't suspect is ambiguous |
| **Fires when** | agent intent diverges from what artifacts prove (too sure, and *wrong*) | agent intent is under-determined by what the user said (too sure, and *can't know yet*) |
| **Ground truth** | present & queryable (grep the repo) | **absent** — what the user *meant* exists nowhere |
| **Resolution** | inject from code | ask the human |

**Why the symmetry leaks (and how the double fixes it):** the code monitor works because its ground
truth is present and queryable. The user's intent is absent — that's what makes it ambiguous. The
bare user monitor can therefore only detect *"interpretation space is visibly plural,"* not
*"wrong-for-this-user."* **The double's whole job is to make the absent oracle present** — turn
"what the user meant" into a queryable index — the same move that made the code monitor work,
applied to the missing oracle.

**The supervisor MUST be external — not a prompt to the executor.** This is non-negotiable, and
empirical: a *separate* supervising process recovers **69.4%** vs. only **61.2%** for telling the
executor to "be careful" (§12, arXiv 2603.26233). The reason is structural — RLHF destroyed the
executor's self-calibration; it is confidently wrong and *cannot feel it* (§12, arXiv 2505.22655).
An agent cannot supervise its own confidence. That single fact is why this is a separate product
and can never be a system prompt.

---

## 3. The escalation ladder — ask, act, preference

The double runs on three primitives: **ask**, **act**, and **preference**. **Preference** is the
*resolver* (what you'd want); **ask** and **act** are the *gate* (whether to bother you). They are
layered, not competing: the double tries to resolve from preference, and its *confidence in that
preference-match* picks act vs. ask. The three form a **loop** — every **ask** that gets answered
becomes a new **preference**, which lets the double **act** silently next time. That loop *is* the
self-learning.

**Two kinds of preference — only one is yours to defend.** *Static content* prefs (tabs, async,
naming) are stateless and commoditized (rules/memory files). *Resolution* preference — how **you**
resolve **this class** of ambiguity, keyed to the `situation_signature` and learned from
overrides/reverts — is the behavioral, uncopyable kind. The double needs both; the moat is the
second, plus the gate.

```
agent intent under-determined (the monitor's trigger)
   │
 [1] situation resolves it?   (failing test, open file, diff, repo structure)   → act
   │ no
 [2] YOUR index resolves it WITH HIGH calibrated confidence?                     → act, LOG assumption visibly
   │   (this user's recent decisions, distilled prefs, how THIS repo does it)
   │ no / low confidence
 [2b] cohort resolves it?  (cross-user prior — in-scope, abstracted, low trust)  → act, LOG ("borrowed")
   │   no / low confidence                                          (configurable — §7)
 [2c] frontier LLM reasons it out WITH HIGH confidence?                          → act, LOG
   │   (escalate COMPUTE before the human — tokens are cheaper than attention)
   │   no / still under-determined / IRREDUCIBLE
 [3] ASK the human  →  AND write the answer back into the index
        └─ the back-arrow is the whole game: every escalation is a training example;
           the threshold drops FOR THIS SIGNATURE next time.
```

**Rungs [2b] and [2c] are both cheaper than the human — spend them first.** [2c] escalates
*compute*: a hard case a frontier model can derive shouldn't cost an interruption — but only for
**preventable** hardness (the answer exists, just hard to reach); **never** burn compute on
**irreducible** ambiguity (§8), where no reasoning manufactures absent intent. [2b] borrows from the
**cohort** (cross-user prior, §7): a fallback used only at low personal coverage, kept low-confidence
and logged as *borrowed*, yielding instantly to your own signal, and **in-scope only** (free within
an org/team; cross-org = abstracted patterns + consent, §9).

**The trap to engineer against (rung 2):** "resolve silently from history" is the code-monitor's
strength turned into the user-monitor's trap. Precedent ("how they fixed it before") is not current
intent ("what they want now"). A double is *precedent compiled into a model* — more persuasive
precedent, wearing the user's voice. Mitigations:
- keep the double as **transparent retrieval**, not an opaque twin → you can show & log the assumption;
- **calibrated confidence gates rung 2** — only high-confidence reads resolve silently;
- **log the assumption visibly** so the user can catch a bad silent resolution.

---

## 4. Reversibility (act-first, revert-later) — how the double escapes the tradeoff

The whole thing lives on the **convenience ↔ correctness** axis. The bet is to *escape* the
tradeoff, not balance it. A second instrument, with inverted economics:

- **Ask-before-act:** pay `C_ask` (user attention) on *every* ambiguous action.
- **Bypass-then-revert:** pay `p · C_revert` — only on the wrong fraction.
- **Bypass wins when `p · C_revert < C_ask`** — actions usually right *and* cheaply reversible.

**Coding has the lowest `C_revert` of any domain** because `git` is a universal, free, perfect
undo. This is why act-first is *especially* viable here — the reversal infrastructure already
exists. (It is also why the domain is *locked*: in coding, `git revert` is **both** the undo **and**
the training label — see §5.)

The two instruments combine into a **2×2** the monitor selects within — and this is exactly how the
double rations its **forcefulness**:

```
                  LOW reversibility cost          HIGH reversibility cost
                  (leaf, git-undoable)            (foundational, entangled)
HIGH confidence │ BYPASS, log assumption          │ ACT, but flag for review
(index dense)   │                                 │
LOW confidence  │ BYPASS, loud + trivially revert │ ASK FIRST  ← the only real ask quadrant
(under-det.)    │                                 │
```

**Forceful only in the bottom-right** — under-determined *and* hard to undo. Everywhere else the
double is passive (act-first, log, revert-if-wrong). Be forceful everywhere and you've rebuilt the
"always ask" strawman the whole design exists to beat. Teeth that bite only where it matters.

**Reversibility is mechanically estimable** (clean git boundary? downstream deps? leaf edit vs.
foundational type change?) — a *far more tractable* signal than calibrated intent. This is the good
news: it lowers the asking rate where safe without lowering it where dangerous.

**Failure modes of naive bypass-everything (must engineer against):**
1. **Detection latency** — `C_revert` grows with time-to-notice; a bug caught in 10s is free, three commits deep is expensive.
2. **Compounding/entanglement** — action B depends on wrong action A; reverting becomes untangling a dependency graph, not one `git revert`.
3. **Silent-accept poison** — the system mistakes *unnoticed* for *endorsed*, grows confident, and trains its own blind spot deeper. **"Never viewed" must be NO signal, not a positive.**

**Authorization caveat:** "act-first on reversible edits in a branch/worktree/sandbox" is
defensible (that's what branches are *for*). A forceful double must **not** bypass permission gates
that guard irreversible or outward-facing actions. The 2×2 keeps genuinely irreversible actions in
the ask-first quadrant.

**The kicker:** past a point, convenience and correctness *align* — a confident-wrong agent is the
*most* inconvenient outcome (debugging its mess three commits deep). **Correctness IS convenience,
once you count the cleanup.**

---

## 5. Why no one can copy this (defensibility)

Don't argue feature-by-feature — you lose, because each feature exists *somewhere*. Argue the
**combination**: it takes four pieces, they only work *together*, and **every existing product has
at most one.**

| Piece | What it means | Who has it | Who doesn't |
|---|---|---|---|
| **A — External** | a *separate* agent judges confidence, not the executor self-reflecting | reviewer/critic agents | Cursor, Copilot, "be careful" prompts |
| **B — Right object** | learns the **ask/act decision + situation-keyed preference**, not static content prefs | arXiv 2603.26233 | Cursor Memories, all rules files |
| **C — Right signal** | learns from **passive behavior** (overrides, `git revert`), not chat / manual rules | ~nobody | everyone (chat / manual) |
| **D — Provable** | **transparent + measured** — visible/vetoable + a falling override-rate curve | ~nobody | everyone |

- Cursor/Copilot memory = content prefs, from chat/manual → **B and C missing.**
- Code reviewers = external, but stateless, review *output* not *intent* → **B and learning missing.**
- Closest research (arXiv 2603.26233) = A + B, but **no cross-task persistence** → **C and D missing.**

**Nobody has A+B+C+D.** The rebuttal in one line: *"yes, each piece exists — and a car exists as a
pile of parts. The product is the assembly, and no one has assembled it on a real user."*

**The hardest objection, with the hardest rebuttal.** *"Just prompt the agent to be careful."*
Answer with the number: self-reflection **61.2%** vs. external supervisor **69.4%** (§12). Not a
tuning gap — structural.

**Three moats that survive even if a competitor copies the architecture tomorrow:**
1. **Domain-locked.** It works because `git revert` is *both* the free undo *and* the training
   label — the same action that fixes the mistake teaches the double. No other domain hands you
   that. Durable *because* it's coding-specific.
2. **Per-user data flywheel.** Even a perfect clone of your code ships cold. Your double has *your*
   months of resolution history — uncopyable, and the switching cost compounds with use. "The more
   you use it, the better" is also "the more you use it, the more locked-in you are."
3. **Cross-user network effect.** With many users, a new user's cold start is warmed by a
   *data-driven cohort prior* (§7), and each resolution can improve similar users' doubles — so the
   product gets better with *total* users, not just your own history. A stronger flywheel than #2
   (Waze, not a private notebook). Bounded by the IP line: free within an org/team, abstracted
   patterns only across orgs (§9).

---

## 6. Data layer — what to log

The atomic unit is a **resolution event**: a moment where intent was under-determined and got
resolved. You generate this data for free — capture decisions already happening, no labeling effort
required. *(This "for free, from behavior" capture is piece C — the right signal.)*

**Git history is the same schema, retrospective.** Commits — especially reverts and fix-ups — are
resolution events you can mine *before* the user ever runs the double live; the single strongest
cold-start accelerant (§7). Caveat: git gives you the `resolution` and (for reverts) the `outcome`,
but the `situation_signature` and `interpretation_space` must be **inferred** from the diff, not
read off — a wrong inference pollutes the index.

```
resolution_event {
  situation_signature:  what made this ambiguous   (THE RETRIEVAL KEY)
  interpretation_space: the plausible readings
  resolution:           which reading won
  source:               situation | double | human         (which rung)
  reversibility_est:    low | high                          (the 2×2 axis)
  outcome:              accepted | overridden | reverted | answered   (THE LABEL)
}
```

**The `situation_signature` is the make-or-break decision.** It is the retrieval key. For coding it
is unusually rich & cheap to extract: failing test/error type, files & symbols in play, repo
conventions touched (**reuse the code monitor's index**), phrasing class of the request ("fix it" /
"clean this up" / "make it like X"), prior art in this repo.
- Too coarse → confidently misapplies last week's resolution (automated mask).
- Too fine → coverage always sparse, never exits cold start, asks forever.

This same coarse/fine knife-edge governs **generalization**: turning one event ("remove this
comment") into a principle ("stop writing obvious comments") = choosing a coarser signature. The LLM
*proposes* a generalization; the user's later behavior *corrects the altitude* (an over-broad
generalization shows up as an override spike, §8). Generalize aggressively for coverage; let
behavior pull it back.

**Labels come from behavior, not surveys — and the unit is *intervention*.** A perfectly calibrated
double would need **zero** user intervention; so *every* intervention — an interrupt, a "no, do it
this way," a re-prompt, a rejected diff, a hand-edit, a `git revert` — is a deviation from perfect:
a **failure signal**. The cheapest arrive *seconds* after the mistake, before commit, before
compounding (§4); `git revert` is the *cleanest* but the *latest*. **Treat intervention as the unit
you count and drive down.**

*Negatives (raise the threshold — "be more careful here"):*
- **interrupt** — user *halts* the agent mid-execution. The **earliest and among the strongest**
  negatives — they couldn't even let it finish.
- **override** — agent silently resolved an under-determined point as X; user corrected to Y (X→Y).
  The **primary, highest-volume** label — *any* in-conversation change request counts. The
  "confident-and-wrong" event.
- **revert** — user undid a *committed* change (`git revert`/`checkout`). The *cleanest* unambiguous
  negative (timestamped, free from the medium — the moat, §5), but the **latest**: it only fires
  after the user shipped and lived with the mistake.

*Positives (lower the threshold — "you got this right, act silently next time"):*
- **confirmed-good** — the double reasoned it out, **laid the breakdown out legibly**, and the user
  **reviewed and accepted it wholesale** (accept-all on a diff/plan, merged unchanged, "yes, all of
  that"). The **confident-and-RIGHT** event — the mirror of `override`, and the strongest
  *calibration* positive: acting silently was vindicated → the threshold drops for this signature.
- **answered** — human answered a clarifying question. The strongest *content* positive (gives the
  right reading directly) and confirms the ask was warranted; rarest, worth ~100× a silent-accept.
- **accepted (silent)** — weak positive **only if `viewed`**; **down-weight hard, time-decay, and
  never count never-viewed as positive** (that's silent-accept poison, §4, not endorsement).

**Why positives are not optional.** Negatives alone only ever say "be more careful" → the threshold
drifts *up* → the double regresses to the "always ask" tool. **Positives are what make the threshold
come *down*** — the §8 "rate falling" claim is impossible without them. A positive's strength scales
with **evidence of review**: reviewed-wholesale-accept is gold, never-viewed is *zero*. And
**legibility earns it** — the double's clear breakdown is what turns a wholesale accept into a real
endorsement rather than a rubber stamp (the transparency of §3 paying off as signal).

**Granularity: capture per *decision*, label and learn per *session*.** The raw `resolution_event`
is logged **per decision** — it must be, because the index is keyed by `situation_signature` and
retrieval matches a new situation to past *decisions*, not whole sessions. But **outcomes and the
reflection/distillation pass run per session** (best defined git-aligned: a branch / PR / task),
because that's how users actually react — they don't grade each answer, they merge the PR, ship it,
or redo it. **Credit assignment:** explicit interventions label *their* decision (override at turn 5,
revert of file X = precise negatives); the **session outcome labels the rest** (merged-clean →
`confirmed-good` for the un-intervened decisions; reverted / abandoned → negative). Session is also
the natural **measurement window** for §8. Trade-off: session labels are coarser (they can't pin
*which* decision mattered) — which is exactly why the precise per-decision interventions carry the
fine signal and the session outcome only fills the gaps.

**Two phases: faithful record (hot path) → reflect (session end).** During the session, *record
faithfully and judge nothing* — log the raw event stream (intents, actions, interventions, `viewed`,
diffs, SHAs, timestamps) losslessly and unopinionated. All interpretation waits for a single
**reflection pass at session end**: credit assignment, correction-vs-iteration classification,
distillation of raw events → stable preference statements, signature extraction, index/persona
update, and (if opted in) abstract-then-contribute to the cohort (§7 — reflection is the single
chokepoint where the IP boundary is enforced). Two payoffs: the hot path stays cheap and never slows
the executor; and because the raw log is **event-sourced ground truth**, a wrong signature or
labeling rule is fixed by **re-reflecting over the faithful log, not re-collecting** — which de-risks
the make-or-break signature design (§10). Durable learning lands only at reflection; *within* a
session the double still won't repeat a just-corrected mistake, because the recent raw events sit in
its live context — that's ephemeral, not an index commit.

**Calibrated reflection — the double must not learn the *wrong* thing.** Reflection faces two
*under-determined* inferences — the §2 problem turned inward:
- **Credit assignment** — many decisions, one fuzzy outcome: which decision earned the merge, which
  caused the revert three commits later?
- **Distillation** — "user changed X→Y" could mean *prefers Y* (a rule), *this file needed Y*
  (narrow), or *changed their mind once* (nothing). Same behavior, multiple lessons.

**The danger:** if reflection **confidently extracts the wrong lesson**, it bakes an *automated mask
of the user* into the index (§1, §3) — the double becomes confidently-wrong *about you*, the exact
failure it exists to prevent, now living **inside the learning loop**. So the make-or-break is not
"can it learn" but "**can it avoid learning the wrong thing.**"

**The fix is self-similar — point the double's own discipline back at reflection:**
- **Calibrated lesson-extraction** — one X→Y stays a *raw event* (weak, retrievable); only
  **repeated, consistent** signals distill into a **preference rule**. (The §3 "high confidence gates
  silent action" rule, applied to *what becomes a rule*.)
- **Generalization is a hypothesis, not a fact** — reflection *proposes* the altitude; later behavior
  confirms or narrows it (§6 above). Never commit an altitude from one event.
- **Transparency + veto** — every distilled rule is visible ("the double now thinks you prefer Y")
  and deletable. A wrong lesson is correctable precisely because it's a *record, not a weight* —
  which is the whole reason the model stays frozen and learning lives in the index.
- **Escalate when genuinely ambiguous** — keep it raw, or (rarely) ask: "a preference, or just this
  once?" Same escalation ladder, pointed at the learning step; the answer is a gold label.

**The payoff:** one discipline runs at *both* layers — **advising** the user (don't act confidently
on under-determined intent) and **learning** from the user (don't learn confidently from
under-determined behavior). The thing that makes the double safe is the same thing that makes its
learning safe.

**"All intervention is failure" is the right lens — but failure splits two ways.** Default-assume
every intervention was preventable; that keeps the double honest (the opposite of silent-accept
complacency). Then classify:
- **Preventable** — the double should have *asked* or *known* (the right reading was recoverable from
  repo / history / preference, or the point was under-determined and it should have flagged it).
  These are the failures its job is to drive to zero. **Gold negative — learn (X→Y).**
- **Irreducible** — intent that only crystallizes once the user *sees* a concrete attempt ("I'll
  know it when I see it"). No amount of asking prevents it; asking upfront would just annoy. The
  answer here is **not** "ask more" — it's §4's bottom-left quadrant: act fast, make the iteration
  *trivially cheap to revert*. Don't penalize the double; this floor is structural.

The trap is collapsing the two — count irreducible iteration as the double's fault and it concludes
it fails constantly → confidence collapses → it asks about everything → it becomes the "always ask"
tool you were beating. **Drive preventable intervention to zero; make irreducible intervention cheap.**

---

## 7. Cold start — four layers, four costs

It's not one problem.

- **Layer 0 — the situation index is never cold.** On a brand-new user, the double still has the
  **repo**. "How does this codebase resolve this kind of ambiguity?" is answerable from the code —
  the **same index the code monitor already built.** Rung [1] works day zero. *This is why coding is
  the right scope.*
- **Layer 1 — population prior, *configurable*.** For repo-unresolvable cases, fall back to a prior
  — and let the user pick its strength (this is the §4 convenience↔correctness tradeoff applied to
  cold start):
    - **Clean start** — no cohort borrowing; pure personal + repo. Max privacy, purest *you*, slow ramp.
    - **Average good** — start from a **data-driven cohort prior** ("developers like you"). Fast ramp,
      decent defaults, but crowd-flavored early.
    - **Bought expert** — seed from a *specific* expert's double (the marketplace premium tier, §9).

  The knob sets the **ramp and the long tail, not the destination**: for any situation you personally
  cover, your signal overrides the cohort, so clean and average *converge* on the head and differ only
  on the rarely-hit tail. Two **independent consent toggles** — *receive* a cohort prior / *contribute*
  your resolutions (cross-org contribute = abstracted patterns only, §9). Defaults: within-team →
  cohort on (institutional memory); individual / cross-org → clean by default, cohort opt-in.
  *(Advanced: set it per situation-class — borrow the crowd for common/low-stakes, stay clean for your
  distinctive, high-stakes judgment.)*
- **Layer 2 — fast personalization, not training.** The double is **in-context/retrieval, not
  fine-tuned** → a *single* resolution event is usable immediately. Cold start ends not after N
  examples but after the *first* resolution **per situation-class**. *Verified backing (§12):*
  in-context preference-following **collapses below 10% by ~10 turns** (PrefEval, ICLR'25) and
  retrieval is the best non-fine-tuning fix — so the double must **retrieve**, never ride the
  executor's context window.
- **Layer 3 — explicit bootstrap (optional accelerant).** Show 5–10 real ambiguous coding
  situations, let the user resolve them, seed high-confidence events. Optional.

**The biggest accelerant: mine the git history (user-provided).** A repo's commit log is a
*retrospective stream of resolution events* — real decisions the author already made, for free,
before the double is ever used live:
- **revert commits** → historical *revert* labels (the cleanest negative, §6);
- **fix-up / follow-up commits** ("use the existing helper instead") → historical *override* (X→Y);
- **review-driven changes** → what this team treats as wrong → resolution preference;
- **recurring diff patterns + `git blame`** → per-author preference (the portable layer) and team
  consensus (the repo-scoped layer).

Ingesting history upgrades "never cold" (Layer 0) from *the repo as it stands* to *the repo's whole
history of decisions* — a far denser prior. **Caveats (same discipline as live capture):**
- git shows the **resolution, not the interpretation_space** — you must *infer* the
  `situation_signature` each commit resolved; a wrong inference pollutes the index;
- **committed ≠ endorsed** — shipped code can be a deadline hack or a reviewer's demand; treating
  every committed line as a positive repeats silent-accept poison (§4) at historical scale.
  Up-weight explicit reverts/fix-ups; down-weight "merely committed";
- **time-decay** — a 3-year-old preference may be stale; the codebase and author both moved on;
- it **warms the prior, doesn't end cold start** (still per-situation-class) — the live
  override/revert/answered loop stays the gold label.

*Literature status (2026 verified sweep, §12):* mining git/PR history to seed a **per-developer**
double *before* live interaction is essentially **unstudied** — a genuine opening, not a solved
technique. The closest work, **CodeFavor** (arXiv 2410.03837, "Commit-Instruct"), extracts
*population-level* code preferences from commit pairs (not a personal double); **CIPHER** learns
latent preference from a user's *edits* but on text, not code. So the per-developer git-seeding lever
is ours to invent.

**Reusability half-solves cold start in a new repo.** Because the double is portable and owned, a
*new* project is not fully cold: the user's **personal layer** (e.g. "I hate obvious comments")
transfers; only the **repo-specific layer** starts from zero. The discipline is keeping these two
tiers separate — a repo-specific resolution must **not** leak into the portable layer, or it
misfires (confident-wrong) in the next repo.

**Policy: ask more when you have less.** The asking threshold is a function of how much
resolution-history covers the *current signature*. Cold start isn't a phase you exit — it's a
per-situation coverage number. A *new kind* of ambiguity correctly re-triggers asking even for an
experienced user.

---

## 8. The proof gate — the one number that must come first

> **Of the times the index said "high confidence, stay silent," how often did the user
> override/revert — and is that rate falling over time?**

This is the sharpest *slice* of a broader health number: the **total intervention rate** (§6) —
every interrupt / correction / rejection / revert per unit of work — trending toward its
*irreducible floor* (the "I'll know it when I see it" residue no calibration can remove). The double
is working when **preventable** intervention falls and only the structural floor remains. Measure it
over **reviewed** decisions only — a never-viewed accept is neither success nor failure (§6); the
positive complement is **confirmed-good** (reviewed wholesale acceptance), and a healthy curve shows
that *rising* as override/revert *falls*.

**This is the gate before the marketplace.** Everything in §9 is worthless if this curve is flat.
Every published paper measures this against a *simulated* user — and a 2026 real-human study (**Lost
in Simulation**, arXiv 2601.17087) shows simulated users are *unreliable proxies* for real ones on
agentic tasks, so the gap **can't be closed with a better simulator** (§12). **Nobody has the
real-human longitudinal number.** Building this on a real developer's coding history would
contribute exactly what the literature can't — *and* it is the precise thing that makes the double
sellable later.

**Later, the same curve becomes the marketplace's trust certificate.** You don't sell "trust me,
it's an expert's double" — you sell "this double's override-rate on a held-out suite is X%." The
metric that *proves it works* (now) is the metric that *makes it tradeable* (later). Build it once.

---

## 9. The path to a market for doubles

The vision is double-selling; the discipline is sequence. Do **not** lead with the marketplace — it
is the most exciting slide and the most copyable product.

**Step 1 — Prove the loop on top coders (the wedge experiment).** Put elite engineers on it for
weeks. Triple win: (a) it produces the §8 real-human curve nobody has; (b) it seeds the first
sellable double; (c) it proves the thesis — the curve bends or it doesn't, and no model quality
saves a flat one (§10). Top coders are the *right* subjects because they **override** — overrides/
reverts are the highest-quality labels (§6); novices silent-accept garbage and poison the index.
**Pre-warm each coder's double from their commit history before the week starts (§7)**, so the curve
measures *learning on top of a real prior*, not cold flailing.

**Step 2 — Monetize B2B first.** A team's double = **institutional memory**: a new hire inherits
the team's judgment on day one. Clear buyer, real budget, retention via the flywheel — and the
double **stays in-house**, which sidesteps the marketplace's worst problem (IP, below).

**Step 3 — The consumer marketplace (the future, last).** People sell their doubles online: a top
engineer's judgment, packaged and bought. Compelling — and the natural endpoint of *reusable +
owned*. It is also the **premium tier of the cross-user spectrum** (§7): *clean start* → free
*average cohort* prior → *bought specific-expert* double — the marketplace just sells a **named**
prior instead of the anonymous average (same axis, higher price, same IP problems in sharper form).
But three hard problems make it phase-3, not the wedge:

1. **Personal doesn't transfer — a *sold* double is a weaker, more copyable product.** The moat was
   "learns *your* judgment from *your* behavior." A *bought* double learns from someone else's, so
   it degrades into an **expert preset** — which is far more copyable (a competitor ships "Senior
   Staff Eng rule-pack" next quarter) and **re-weakens criterion 1**. Worse, it imposes the
   *seller's* ask/act anxieties on the buyer — "precedent wearing a mask" (§3), but someone else's,
   that you paid for.
2. **IP / privacy minefield.** A double trained over weeks at a day job encodes employer
   conventions, proprietary patterns, secrets-by-inference. "Sell your double" can mean "sell your
   employer's code judgment." (Second reason B2B-first is safer — the double stays inside the org.)
3. **Provenance** — solved by §8's curve-as-certificate, but only if the metric exists first.

**The discipline, restated:** lead with the personal learning loop (uncopyable — it's *your*
behavior over time), not the marketplace (a sold double is a copyable preset). The marketplace is
what the proven loop **unlocks**, not what we build first.

---

## 10. First artifact (buildable this week)

**A logger that captures resolution events — and resolves nothing yet.** Rides on top of the
existing code agent. Run a week. Then check: *did similar signatures actually get consistent
resolutions?*
- If yes → the signature design works, the double is viable, build the monitor.
- If no → the signature is wrong, and no model quality saves it. Fix the signature first.

**Measure before building the resolver.** The signature function is the single make-or-break
engineering decision; validate it empirically before investing in the monitor or the resolver. This
logger is also Step 1 of §9 — it is what you run on the top coders.

---

## 11. How the pieces fit

Three processes, one shared index. The double is **external** to the executor — never in its head
(post-training corrupts the executor's own confidence; calibration must be externalized). Being
external is also what makes it **reusable**: a layer that wraps any executor, owned by the user.

```
┌─────────────────────────────────────────────────────────────────┐
│  WORKING AGENT (executor) — does the coding. Never judges its own │
│  intent-confidence. Emits intent before acting.                   │
└───────────────┬───────────────────────────────────┬──────────────┘
                │ "about to do X because..."         │ acts / user reacts
                ▼                                     ▼
┌──────────────────────────────┐         ┌───────────────────────────┐
│  MONITOR (the supervisor)    │         │  CAPTURE                  │
│  external. per intent:       │         │  watches outcome:         │
│  signature → query index     │         │  override / revert /      │
│  → coverage + confidence     │         │  accept / answer          │
│  + reversibility estimate    │         │  → writes resolution event│
│  → select 2×2 quadrant       │         └─────────────┬─────────────┘
└──────────────┬───────────────┘                       │
               │ human answer (gold label)              │
               ▼                                        ▼
        ┌────────────────────────────────────────────────────┐
        │  THE INDEX ("the double")                           │
        │  resolution events keyed by situation_signature.    │
        │  SHARES the code monitor's repo index as its prior. │
        │  retrieval, not a trained twin → transparent.       │
        │  portable + user-owned → reusable, sellable later.  │
        └────────────────────────────────────────────────────┘
```

**Same architecture as the code monitor, not a new one:** both are an external process that signs a
situation, queries a shared index, acts on divergence. **One index infrastructure, two query
patterns, two triggers.**

---

## 12. Relevant literature (2025–2026)

The field converged on this architecture this year — which both validates the design and leaves the
real-human longitudinal metric (the proof gate, §8) open.

- **Ask or Assume? Uncertainty-Aware Clarification-Seeking in Coding Agents** (arXiv 2603.26233) —
  the closest built version. Separate "Intent Agent" watches execution history and halts on
  under-determination. Monitored multi-agent recovers **69.4%** resolve vs **70.8%**
  fully-specified; single-agent "be careful" only **61.2%** → **the monitor must be a separate
  process, not a prompt to the executor.** Calibration is real & coarse (queries 62% of easy tasks
  → 100% of hardest). Caveats: GPT-5.1 *simulator* not human; needs frontier model;
  prompt-scaffolded not native; cost ~2× (~$3.50 vs $1.63/task). **No cross-task persistence — they
  sidestep cold start entirely** (i.e. they have A+B, not C+D — §5).
- **Training Proactive and Personalized LLM Agents** (arXiv 2511.02208) — the *trained* fork. RL on
  the consequences of asking; per-user calibrated threshold **beats both "always ask" and "never
  ask."** The retrain back-arrow made concrete.
- **T-POP: Test-Time Personalization with Online Preference Feedback** (arXiv 2509.24696) — build
  the double **at test time from recent exchanges**, not by fine-tuning. The "retrieval not
  training" mechanism; efficacy with modest feedback.
- **PersonaAgent** (NeurIPS'25, arXiv 2506.06254) — two-tier **episodic + semantic** memory, per-user
  persona prompt, **no weight updates** → the memory architecture for the double. *(Verified 3-0. The
  separate* Survey on Personalized & Pluralistic Preference Alignment *is arXiv 2504.07070.)*
- **Position: UQ Needs Reassessment for LLM Agents** (arXiv 2505.22655) — base-LLM likelihoods are
  calibrated, **but RLHF destroys it; single-turn RLHF encourages overconfidence.** → *why*
  calibration must live in an external monitor, not the executor. *(Not in the 2026 sweep's corpus —
  the "fine-tuning corrupts calibration" claim is asserted by motivation across the corpus, not
  independently re-measured here.)*
- **Learning to Ask: When LLM Agents Meet Unclear Instruction** (EMNLP 2025) — SOTA LLMs default to
  assuming intent rather than asking on ambiguous requests.

**Fast-personalization mechanisms (2026 verified sweep — 24/25 claims passed 3-vote adversarial check):**
- **PrefEval** (ICLR'25 Oral, arXiv 2502.09597) — in-context preference-following **collapses below
  10% by ~10 turns**, ~zero by 30–300; **retrieval (RAG) is the best non-fine-tuning fix** → hard
  evidence the double must *retrieve*, not ride the executor's context (§7, §11). *(3-0.)*
- **CIPHER / PRELUDE** (arXiv 2404.15269) — learns a user's latent preference from their **edits**,
  frozen model, interpretable & user-editable → closest analogue to learning from diffs/reverts. *(3-0.)*
- **SAGE-Agent** (arXiv 2511.08798) — ask-vs-act as a POMDP with a Bayesian **EVPI** objective: ask
  only when expected info value > interaction cost; 7–39% more coverage with 1.5–2.7× fewer questions.
  The formal version of the §4 ask-gate. *(3-0.)*
- **Verified ramp (ALL simulated, NONE coding):** test-time personalization is useful in **~2–10
  feedback signals** (TPO ~2 steps; T-POP ~5; response-feedback RFM/PReF/LoRe ~3–30 examples). A
  specific "~20 interactions" figure was **refuted (0-3)** — don't cite it. **No real-human,
  coding-domain ramp exists** — exactly the §8 gap.
