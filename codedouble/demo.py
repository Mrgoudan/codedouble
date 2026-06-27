"""End-to-end demo on a SIMULATED user.

⚠️  Honest caveat (README §8, and the deep-research finding "Lost in
Simulation"): this validates the *mechanism*, not the real-human claim. The
user here has a stable, knowable preference function — exactly what a real
developer does NOT hand you. A falling curve here means the plumbing works
(signatures cluster -> confidence rises -> override-rate on confident-silent
decisions falls). It says nothing about whether real behavioral signatures
cluster — that's the make-or-break, and only real developers can answer it.

Run:  python3 -m codedouble.demo
"""

from __future__ import annotations

import random
from typing import List

from .double import Double
from .embedder import HashingEmbedder
from .index import Calibrator, ResolutionIndex
from .metrics import Record, accuracy, ask_rate, section8_rate, windowed_section8
from .reflect import reflect_session
from .signature import RuleBasedExtractor
from .types import Action, Outcome, Reversibility


# ---- a simulated user: situation-classes, each with a STABLE true preference ----
class Klass:
    def __init__(self, name, vocab, lang, error_type, action_kind, phrasing, pref, rev):
        self.name = name
        self.vocab = vocab
        self.lang = lang
        self.error_type = error_type
        self.action_kind = action_kind
        self.phrasing = phrasing
        self.pref = pref            # the resolution this user always wants
        self.rev = rev


CLASSES = [
    Klass("auth-null", ["login", "session", "token", "auth", "verify"], "python",
          "null deref on user", "edit", "fix the", "guard-with-Option", Reversibility.LOW),
    Klass("api-cleanup", ["handler", "router", "endpoint", "request", "response"], "python",
          "", "refactor", "clean up the", "extract-helper", Reversibility.LOW),
    Klass("db-rename", ["table", "column", "schema", "migration", "query"], "python",
          "", "rename", "rename the", "snake_case-name", Reversibility.HIGH),
    Klass("ui-style", ["button", "color", "layout", "css", "component"], "ts",
          "", "edit", "make it like the", "house-style-tokens", Reversibility.LOW),
    Klass("perf-loop", ["loop", "cache", "batch", "latency", "query"], "python",
          "timeout in batch", "edit", "fix the", "memoize-results", Reversibility.LOW),
]


def make_moment(rng: random.Random, k: Klass) -> dict:
    toks = rng.sample(k.vocab, k=min(3, len(k.vocab)))
    noise = "".join(rng.choice("xyz") for _ in range(1))  # tiny noise token
    return {
        "request": f"{k.phrasing} {' '.join(toks)} {noise}",
        "diff": " ".join(toks),
        "error": k.error_type,
        "lang": k.lang,
        "repo": "acme/app",
        "files": (toks[0] + ".py",),
        "symbols": (toks[0],),
        "action_kind": k.action_kind,
    }


def run(rounds: int = 30, per_round: int = 20, drift_p: float = 0.04, seed: int = 7,
        extractor=None) -> List[Record]:
    rng = random.Random(seed)
    if extractor is None:
        extractor = RuleBasedExtractor(HashingEmbedder(dim=256))
    calibrator = Calibrator()
    index = ResolutionIndex()
    double = Double(extractor, index, calibrator, conf_threshold=0.6)

    history: List[Record] = []
    for r in range(rounds):
        double.now = float(r)
        session: List[Record] = []
        for _ in range(per_round):
            k = rng.choice(CLASSES)
            # rare ONE-OFF "changed my mind this time" = the irreducible floor.
            # It does NOT change the class's stable preference, so the index
            # stays mostly consistent — this is iteration noise, not drift.
            if rng.random() < drift_p:
                true_pref = k.pref + f"-oneoff{r}"
            else:
                true_pref = k.pref

            moment = make_moment(rng, k)
            decision = double.resolve(moment, k.rev)

            if decision.action is Action.ASK:
                outcome, correct, corrected = Outcome.ANSWERED, True, true_pref
            elif decision.resolution == true_pref:
                outcome, correct, corrected = Outcome.CONFIRMED_GOOD, True, None
            else:
                outcome, correct, corrected = Outcome.OVERRIDE, False, true_pref

            double.record(decision, outcome, corrected_to=corrected, session_id=r)
            rec = Record(decision, outcome, correct)
            session.append(rec)
            history.append(rec)

        reflect_session(index, calibrator, [(x.decision, x.outcome, x.correct) for x in session], now=float(r))

    return history, index, calibrator


def build_backend():
    """Pick the extractor/embedder from CODEDOUBLE_BACKEND:
       default = no-model (HashingEmbedder + RuleBasedExtractor)
       st      = local sentence-transformers embeddings + rule-based fields
       mistral = mistral-embed + LLM-inferred fields (needs MISTRAL_API_KEY)
    """
    import os
    backend = os.environ.get("CODEDOUBLE_BACKEND", "default").lower()
    if backend == "mistral":
        from .backends import mistral_extractor
        print("[backend] mistral: mistral-embed + LLMExtractor(mistral-small)")
        return mistral_extractor()
    if backend == "st":
        from .backends import STEmbedder
        print("[backend] sentence-transformers (local embeddings)")
        return RuleBasedExtractor(STEmbedder())
    print("[backend] default: HashingEmbedder + RuleBasedExtractor (no model)")
    return None


def main() -> None:
    history, index, calibrator = run(extractor=build_backend())

    print("Self-Learning Code Double — simulated-user demo")
    print("=" * 66)
    print(f"events logged: {len(index.events)}   distilled rules: {len(index.semantic.rules)}")
    print(f"final calibration error (ECE): {calibrator.ece():.3f}")
    print()
    print("§8 curve — override/revert rate among CONFIDENT, SILENT decisions")
    print("(should fall toward the irreducible floor as the index fills)")
    print()
    print(f"{'window':>7} {'§8 rate':>9} {'n_silent':>9} {'ask_rate':>9} {'accuracy':>9}")
    win = 60
    for wi, rate, n in windowed_section8(history, window=win):
        chunk = history[wi * win : wi * win + win]
        rate_s = f"{rate:.2f}" if rate is not None else "  n/a"
        print(f"{wi:>7} {rate_s:>9} {n:>9} {ask_rate(chunk):>9.2f} {accuracy(chunk):>9.2f}")

    early = history[:120]
    late = history[-120:]
    er, _ = section8_rate(early)
    lr, _ = section8_rate(late)
    print()
    print(f"§8 rate  early={er if er is not None else float('nan'):.2f}  ->  late={lr if lr is not None else float('nan'):.2f}")
    print(f"ask-rate early={ask_rate(early):.2f}  ->  late={ask_rate(late):.2f}")
    print(f"accuracy early={accuracy(early):.2f}  ->  late={accuracy(late):.2f}")
    print()
    print("Read: it starts by ASKING a lot (no precedent) and being wrong when it")
    print("guesses; as the index fills it ASKS less, acts more, and the override")
    print("rate on its confident-silent calls drops to a small floor (the drift).")


if __name__ == "__main__":
    main()
