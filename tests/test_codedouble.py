"""Tests for the codedouble prototype.  Run:  python3 tests/test_codedouble.py"""

from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from codedouble import (  # noqa: E402
    Calibrator,
    Double,
    HashingEmbedder,
    Outcome,
    ResolutionIndex,
    Reversibility,
    RuleBasedExtractor,
    reflect_session,
)
from codedouble.backends import FakeLLM, LLMExtractor  # noqa: E402
from codedouble.logger import EventLog, record_interaction, replay  # noqa: E402
from codedouble.viz import render_html  # noqa: E402
from codedouble.double import gate  # noqa: E402
from codedouble.embedder import cosine  # noqa: E402
from codedouble.metrics import Record, section8_rate, ask_rate  # noqa: E402
from codedouble.types import Action  # noqa: E402
from codedouble import demo  # noqa: E402


class TestEmbedder(unittest.TestCase):
    def test_deterministic(self):
        e = HashingEmbedder(128)
        self.assertTrue((e.embed("fix the login token") == e.embed("fix the login token")).all())

    def test_similarity_ordering(self):
        e = HashingEmbedder(256)
        a = e.embed("fix the login session token auth")
        near = e.embed("fix the login session verify auth")
        far = e.embed("rename the database table column schema")
        self.assertGreater(cosine(a, near), cosine(a, far))


class TestSignature(unittest.TestCase):
    def test_fields(self):
        ext = RuleBasedExtractor(HashingEmbedder(64))
        sig = ext.extract({
            "request": "clean up the handler",
            "error": "null deref on user",
            "lang": "python", "action_kind": "refactor",
        })
        self.assertEqual(sig.phrasing_class, "clean-up")
        self.assertEqual(sig.error_type, "null-deref")
        self.assertEqual(sig.action_kind, "refactor")
        self.assertIsNotNone(sig.code_vec)


class TestIndexConfidence(unittest.TestCase):
    def _double(self):
        return Double(RuleBasedExtractor(HashingEmbedder(256)), ResolutionIndex(), Calibrator())

    def test_confidence_rises_with_consistent_precedent(self):
        d = self._double()
        moment = {"request": "fix the login session token", "error": "null deref",
                  "lang": "python", "action_kind": "edit", "files": ("login.py",), "symbols": ("login",)}

        # cold: no precedent -> must ask
        dec0 = d.resolve(moment, Reversibility.LOW)
        self.assertEqual(dec0.action, Action.ASK)
        self.assertEqual(dec0.coverage, 0.0)

        # teach it the same resolution several times
        for _ in range(5):
            d.now += 1
            dec = d.resolve(moment, Reversibility.LOW)
            d.record(dec, Outcome.ANSWERED, corrected_to="guard-with-Option")

        d.now += 1
        dec_warm = d.resolve(moment, Reversibility.LOW)
        self.assertGreater(dec_warm.confidence, dec0.confidence)
        self.assertEqual(dec_warm.resolution, "guard-with-Option")
        self.assertGreater(dec_warm.agreement, 0.9)


class TestGate(unittest.TestCase):
    def test_quadrants(self):
        self.assertEqual(gate(0.9, Reversibility.LOW, 0.6), Action.ACT)
        self.assertEqual(gate(0.9, Reversibility.HIGH, 0.6), Action.ACT_FLAG)
        self.assertEqual(gate(0.3, Reversibility.LOW, 0.6), Action.ACT_LOUD)
        self.assertEqual(gate(0.3, Reversibility.HIGH, 0.6), Action.ASK)


class TestReflect(unittest.TestCase):
    def test_promotes_rule_and_calibrates(self):
        d = Double(RuleBasedExtractor(HashingEmbedder(128)), ResolutionIndex(), Calibrator())
        moment = {"request": "clean up the handler router endpoint", "lang": "python",
                  "action_kind": "refactor"}
        records = []
        for _ in range(4):
            d.now += 1
            dec = d.resolve(moment, Reversibility.LOW)
            d.record(dec, Outcome.CONFIRMED_GOOD if dec.resolution else Outcome.ANSWERED,
                     corrected_to=None if dec.resolution else "extract-helper")
            records.append((dec, Outcome.CONFIRMED_GOOD, True))
        summary = reflect_session(d.index, d.calibrator, records, now=d.now, min_promote=3)
        self.assertGreaterEqual(summary["rules_total"], 1)
        self.assertGreater(summary["calibration_observations"], 0)


class TestLLMExtractor(unittest.TestCase):
    def test_fake_llm_fills_fields(self):
        emb = HashingEmbedder(64)
        fake = FakeLLM({
            "phrasing_class": "fix-it", "error_type": "null-deref",
            "action_kind": "edit",
            "interpretation_space": ["guard nulls", "return Option"],
        })
        sig = LLMExtractor(fake, emb).extract(
            {"request": "do the thing", "error": "", "lang": "python"})
        self.assertEqual(sig.phrasing_class, "fix-it")
        self.assertEqual(sig.error_type, "null-deref")
        self.assertEqual(sig.interpretation_space[0], "guard nulls")
        self.assertIsNotNone(sig.code_vec)
        self.assertEqual(fake.calls, 1)

    def test_falls_back_on_garbage(self):
        emb = HashingEmbedder(64)
        ext = LLMExtractor(lambda p: "not json at all", emb)
        sig = ext.extract({"request": "clean up the handler", "lang": "python"})
        self.assertEqual(sig.phrasing_class, "clean-up")  # rule-based fallback


class TestSection8Metric(unittest.TestCase):
    def test_curve_falls_on_simulated_user(self):
        history, index, calibrator = demo.run(rounds=25, per_round=20, seed=3)
        early, late = history[:150], history[-150:]
        er, ne = section8_rate(early)
        lr, nl = section8_rate(late)
        # by the end there should be confident-silent decisions, and the
        # override rate among them should be lower than early (the curve bends)
        self.assertIsNotNone(lr)
        self.assertGreater(nl, 0)
        if er is not None:
            self.assertLessEqual(lr, er + 1e-9)
        self.assertLess(lr, 0.35)                       # falls to a small floor
        self.assertLess(ask_rate(late), ask_rate(early))  # asks less over time
        self.assertGreater(len(index.events), 100)


class TestLoggerViz(unittest.TestCase):
    def test_capture_replay_render(self):
        import tempfile
        d = tempfile.mkdtemp()
        log = EventLog(os.path.join(d, "i.jsonl"))
        for _ in range(6):
            record_interaction(log, "fix the login token", "guard-with-Option",
                               "confirmed_good", lang="python")
        raw = log.read()
        self.assertEqual(len(raw), 6)

        ext = RuleBasedExtractor(HashingEmbedder(128))
        recs, double = replay(raw, ext)
        self.assertEqual(len(recs), 6)
        # first sees no precedent -> asks; later identical moments -> confident act
        self.assertEqual(recs[0].decision.action, Action.ASK)
        self.assertTrue(any(r.decision.action is not Action.ASK for r in recs[1:]))

        out = os.path.join(d, "r.html")
        render_html(recs, path=out, window=3)
        self.assertTrue(os.path.exists(out))
        with open(out) as f:
            self.assertIn("<svg", f.read())


class TestAnchorGraduation(unittest.TestCase):
    """The local->over-time bridge: graduating session anchors into the durable
    global store, and seeding a fresh session back from it. Deterministic (the
    LLM tier is disabled) so only the heuristic paths are exercised."""

    def setUp(self):
        import tempfile, json
        from codedouble import cli
        os.environ["CODEDOUBLE_NO_LLM"] = "1"          # force heuristic dedupe (no network)
        self.cli, self.json = cli, json
        self.d = tempfile.mkdtemp()
        self.log = os.path.join(self.d, "decisions.jsonl")

    def tearDown(self):
        os.environ.pop("CODEDOUBLE_NO_LLM", None)

    def _global(self):
        gp = self.cli._scope_anchors_path(self.log, "default")
        return self.json.loads(open(gp).read())

    def test_heuristic_collapses_pileup(self):
        # the exact kind of near-duplicate pile-up seen live in anchors.global.json
        items = [
            "REAL runs only — no simulated or assumed outputs.",
            "Use ABSOLUTE paths only (cwd is empty scratch).",
            "Use ABSOLUTE paths only; cwd is empty scratch",
            "Use ABSOLUTE paths only; cwd is empty scratch.",
            "SAFE=/home/ziruichen/bsd/llvm-project-dup/libcbs/src/bishengc_safety/bishengc_safety.cb",
            "SAFE=/home/ziruichen/bsd/llvm-project-dup/libcbs/src/bishengc_safety/bishe",  # truncated
            "CL=/home/ziruichen/bsd/llvm-project-dup/build/bin/clang",
            "CL=/home/ziruuchen/bsd/llvm-project-dup/build/bin/clang",  # typo'd dup
        ]
        out = self.cli._dedupe_anchors(items)
        self.assertLess(len(out), len(items))
        # the three "ABSOLUTE paths only" phrasings collapse to one
        self.assertEqual(sum("ABSOLUTE paths only" in x for x in out), 1)
        # the truncated SAFE fragment is absorbed into the full path (does not survive)
        self.assertFalse(any(x.endswith("/bishe") for x in out))
        # the typo'd CL collapses; the correctly-spelled one is kept
        cls = [x for x in out if x.startswith("CL=")]
        self.assertEqual(len(cls), 1)
        self.assertIn("ziruichen", cls[0])
        self.assertNotIn("ziruuchen", cls[0])

    def test_heuristic_collapses_paraphrase_family(self):
        # the "real runs" paraphrase family that re-bloated the live store — must
        # collapse via the heuristic ALONE (the LLM tier is rate-limited / best-effort)
        fam = [
            "REAL runs only — no simulation or guessing",
            "Do REAL runs, no simulation",
            "Run real compilations only; never speculate",
            "Real runs only; never speculate.",
            "Do REAL runs (no simulation).",
        ]
        out = self.cli._dedupe_anchors(fam)
        self.assertLessEqual(len(out), 2)
        # but a genuinely distinct constraint is NOT swallowed
        out2 = self.cli._dedupe_anchors(fam + ["Use ABSOLUTE paths only (cwd is empty scratch)"])
        self.assertTrue(any("ABSOLUTE paths" in x for x in out2))

    def test_goal_never_enters_distill(self):
        # goal is SESSION-ONLY: it must never propagate into a project/global distill
        self.cli._merge_project_distill(self.log, {"goal": "some goal", "constraints": ["c1"]}, "default")
        g = self._global()
        self.assertNotIn("goal", g)
        self.assertIn("c1", g.get("constraints") or [])

    def test_seed_is_separated_by_project_scope(self):
        # the real fix: separation is STRUCTURAL, by project (cwd's repo root), not by
        # token overlap. Two projects on the same machine cannot contaminate each other.
        import tempfile
        proj_a = tempfile.mkdtemp()    # a BSC compiler-hunt scratch dir
        proj_b = tempfile.mkdtemp()    # the codedouble repo
        self.cli._merge_project_distill(self.log, {
            "goal": "Hunt compiler bugs in BSC",
            "constraints": ["Use INC=-I/home/x/bsc_include for includes", "REAL runs only"],
        }, self.cli._scope_key(proj_a))
        self.cli._merge_project_distill(self.log, {
            "goal": "build codedouble",
            "constraints": ["Smart dispatch: hard -> remote LLM, else ollama"],
        }, self.cli._scope_key(proj_b))

        # a fresh session whose cwd is project B inherits ONLY project B's anchors
        self.cli._session_note(self.log, "sB", "continue the work", proj_b)
        seeded = self.cli._session_anchors(self.log, "sB")
        self.assertIn("dispatch", seeded)            # B's own bucket
        self.assertNotIn("bsc_include", seeded)      # A's anchors never cross over
        self.assertNotIn("Hunt compiler", seeded)
        self.assertNotIn("Goal:", seeded)            # goal still never seeded

        # a no-cwd ('default') session shares no bucket with A or B and (no global
        # distill promoted here) seeds nothing -- a strict, non-vacuous isolation check
        self.cli._session_note(self.log, "sNone", "hello")
        self.assertEqual(self.cli._session_anchors(self.log, "sNone"), "")

    def test_decision_anchors_combine_tiers(self):
        # the decision view = session ⊕ project distill ⊕ global distill, deduped;
        # goal ONLY from the session tier; cross-session rules present automatically
        import tempfile
        proj = tempfile.mkdtemp()
        self.cli._merge_project_distill(self.log, {"constraints": ["never use relative paths in scripts"]},
                                        self.cli._scope_key(proj))
        with open(self.cli._global_distill_path(self.log), "w") as f:
            f.write(self.json.dumps({"avoid": ["speculation without real runs"]}))
        self.cli._session_note(self.log, "sM", "work on the tool", proj)
        with open(self.cli._sid_file(self.log, "sM", ".anchors.json"), "w") as f:
            f.write(self.json.dumps({"goal": "my own goal", "constraints": ["session-local rule"]}))
        m = self.cli._decision_anchors(self.log, "sM")
        self.assertEqual(m["goal"], "my own goal")
        self.assertIn("session-local rule", m["constraints"])          # session tier
        self.assertIn("never use relative paths in scripts", m["constraints"])  # project tier
        self.assertIn("speculation without real runs", m["avoid"])     # global tier
        # a session with NO own anchors is primed by the tiers but gets NO goal
        self.cli._session_note(self.log, "sM2", "another conversation", proj)
        m2 = self.cli._decision_anchors(self.log, "sM2")
        self.assertEqual(m2["goal"], "")
        self.assertIn("never use relative paths in scripts", m2["constraints"])

    def test_gate_fires_on_project_distill_rule(self):
        # the point of the combine: a rule learned in OTHER sessions of this project
        # gates THIS session's action even though its own anchors lack it
        import tempfile
        proj = tempfile.mkdtemp()
        self.cli._merge_project_distill(
            self.log, {"avoid": ["Using project ID alone as the key for separation"]},
            self.cli._scope_key(proj))
        self.cli._session_note(self.log, "sD2", "keep building", proj)
        with open(self.cli._sid_file(self.log, "sD2", ".anchors.json"), "w") as f:
            f.write(self.json.dumps({"goal": "build the tool"}))       # own anchors: goal only
        m = self.cli._decision_anchors(self.log, "sD2")
        self.assertTrue(self.cli._drift_check(
            m, "refactor to use the project id alone as the key", "")[0])

    def test_own_goal_wins_but_distill_rules_still_apply(self):
        # decision-time combine: the session's OWN goal always wins (a distill never
        # carries one), but distilled rules now MERGE IN rather than being shadowed
        # by the session's own anchors — cross-session learning applies every turn
        self.cli._merge_project_distill(self.log, {"goal": "global g", "constraints": ["global c"]})
        ap = self.cli._sid_file(self.log, "s_own", ".anchors.json")
        with open(ap, "w") as f:
            f.write(self.json.dumps({"goal": "my own goal", "constraints": ["my own constraint"]}))
        out = self.cli._session_anchors(self.log, "s_own")
        self.assertIn("my own goal", out)
        self.assertNotIn("global g", out)            # a distill goal never surfaces
        self.assertIn("my own constraint", out)      # session tier present
        self.assertIn("global c", out)               # distilled rule now applies too

    def test_global_distill_promotes_cross_project_rules(self):
        # a habit that RECURS across projects globalizes; a project-specific rule does not
        import tempfile
        a, b, c = tempfile.mkdtemp(), tempfile.mkdtemp(), tempfile.mkdtemp()
        self.cli._merge_project_distill(self.log, {"constraints": [
            "REAL runs only, never simulate", "Use INC=-I/x/bsc_include"]},
            self.cli._scope_key(a))
        self.cli._merge_project_distill(self.log, {"constraints": ["Real runs only; never simulate"]},
            self.cli._scope_key(b))
        self.cli._promote_global(self.log)
        glob = self.json.loads(open(self.cli._global_distill_path(self.log)).read())
        flat = (" | ".join(glob.get("constraints") or [])).lower()
        self.assertIn("runs only", flat)             # in A and B -> promoted to global
        self.assertNotIn("bsc_include", flat)        # A-only -> stays project-local

        # a fresh session in a NEW project C is primed by the global distill only
        self.cli._session_note(self.log, "sC", "do stuff", c)
        seeded = self.cli._session_anchors(self.log, "sC")
        self.assertIn("runs only", seeded.lower())
        self.assertNotIn("bsc_include", seeded)

    def test_drift_avoid_list_is_deterministic(self):
        # an action that matches the session's AVOID list is drift, caught without an LLM
        anchors = {"goal": "use session id as the context key",
                   "avoid": ["Using project ID alone as the key for separation"]}
        drift, reason, redirect = self.cli._drift_check(
            anchors, "refactor _scope_key to use the project id alone as the key", "")
        self.assertTrue(drift)
        self.assertIn("project", reason.lower())
        self.assertTrue(redirect)

    def test_drift_needs_an_established_goal(self):
        # no consolidated goal yet -> never flag (don't block early exploration)
        drift, _, _ = self.cli._drift_check(
            {"avoid": ["use project id alone as key"]},
            "use project id alone as the key everywhere", "")
        self.assertFalse(drift)

    def test_drift_allows_on_goal_action(self):
        # an action plainly serving the goal, with no avoid hit, is not drift (no LLM)
        anchors = {"goal": "fix the anchor dedup", "avoid": ["use project id alone as key"]}
        drift, _, _ = self.cli._drift_check(anchors, "improve _dedupe_anchors collapsing", "")
        self.assertFalse(drift)

    def test_heuristic_goal_is_clean(self):
        # offline goal fallback: first prompt, minus the path tail and parentheticals
        g = self.cli._heuristic_goal(
            ["Continue work on codedouble (the Self-Learning Code Double) at /home/x/y (gitee, main). It's an external monitor."])
        self.assertEqual(g, "Continue work on codedouble")
        self.assertEqual(self.cli._heuristic_goal([]), "")

    def test_scope_key_no_collision(self):
        # '/a/b' and '/a-b' sanitize to the same base — the hash suffix must keep them distinct
        self.assertNotEqual(self.cli._scope_key("/tmp/a/b"), self.cli._scope_key("/tmp/a-b"))
        # stable for the same path
        self.assertEqual(self.cli._scope_key("/tmp/a/b"), self.cli._scope_key("/tmp/a/b"))

    def test_drift_avoid_matching(self):
        av = {"goal": "x", "avoid": ["use global state"]}
        # a benign MENTION of a 2-token avoid is NOT drift (no false positive)
        self.assertFalse(self.cli._drift_check(av, "document the global state machine design", "")[0])
        # an ECHO of the avoid phrase IS caught (the short-avoid dead zone is fixed)
        self.assertTrue(self.cli._drift_check(av, "refactor to use global state everywhere", "")[0])
        # a >=3-token paraphrase fires
        self.assertTrue(self.cli._drift_check(
            {"goal": "x", "avoid": ["Using project ID alone as the key for separation"]},
            "refactor to use the project id alone as the key", "")[0])

    def test_clean_prompt_strips_injected_context(self):
        # a goal is NEVER an IDE selection / system notice (real reported bug)
        sel = ("<ide_selection>The user selected lines 46-60 from /x/bug.cbs: "
               "#include <stdio.h></ide_selection> fix the null deref in parse()")
        self.assertEqual(self.cli._clean_prompt("<ide_opened_file>opened x.py</ide_opened_file>"), "")
        self.assertIn("fix the null deref", self.cli._clean_prompt(sel))
        self.assertNotIn("ide_selection", self.cli._clean_prompt(sel))
        # unclosed/truncated block -> stripped to end
        self.assertEqual(self.cli._clean_prompt("<ide_selection>The user selected lines 46-60 from /x"), "")
        # the derived goal is the real intent, not the selection
        self.assertEqual(self.cli._heuristic_goal([sel]).lower()[:3], "fix")

    def test_update_anchors_incremental(self):
        # Mem0-style incremental maintenance: offline NEVER wipes; fresh gets a heuristic
        # goal; pure IDE-noise is a NOOP. (LLM disabled -> the fallback path.)
        a, ok = self.cli._update_anchors({"goal": "keep", "constraints": ["c1"]}, ["ok"])
        self.assertEqual(a["goal"], "keep")            # current preserved on LLM failure
        self.assertEqual(a["constraints"], ["c1"])
        self.assertFalse(ok)                           # LLM unreachable -> caller retries later
        a2, _ = self.cli._update_anchors({}, ["build a parser for the config format"])
        self.assertTrue(a2["goal"])                    # heuristic goal even offline
        _, ok3 = self.cli._update_anchors({"goal": "g"}, ["<ide_opened_file> opened x.py"])
        self.assertTrue(ok3)                           # only IDE-noise -> nothing to process (NOOP)

    def test_outcome_loop_records_and_feeds(self):
        # PostToolUse outcomes are recorded per session (consecutive dupes skipped)...
        self.cli._session_outcome(self.log, "sO", "Edit", "cli.py")
        self.cli._session_outcome(self.log, "sO", "Edit", "cli.py")     # dup -> skipped
        self.cli._session_outcome(self.log, "sO", "Bash", "git commit -m x")
        p = self.cli._sid_file(self.log, "sO", ".outcomes.jsonl")
        self.assertEqual(sum(1 for _ in open(p)), 2)
        # ...and an [outcomes] line is evidence, never a goal
        self.assertEqual(self.cli._heuristic_goal(["[outcomes] Edit cli.py; Bash git commit"]), "")
        # no sid / no target -> no write
        self.cli._session_outcome(self.log, "", "Edit", "x")
        self.cli._session_outcome(self.log, "sO", "Edit", "  ")
        self.assertEqual(sum(1 for _ in open(p)), 2)

    def test_qc_cascade_routing(self):
        # local = rough filter; remote confirms only denies and suspicious allows
        import codedouble.backends as B
        A = {"goal": "g", "constraints": ["Session ID is the key for separation"],
             "decisions": [], "avoid": []}
        orig = B.llm_complete
        calls = []
        def fake(responses):
            def f(prompt, system=None, json_mode=False, timeout=0, prefer="local"):
                calls.append(prefer)
                return responses[prefer]
            return f
        try:
            # comment-only -> allow, NO judge at all
            B.llm_complete = fake({})           # any call would KeyError
            calls.clear()
            self.assertTrue(self.cli._quality_check(A, "# just a comment\n// another")[0])
            self.assertEqual(calls, [])
            # local deny + remote allow -> overturned (false positive killed)
            B.llm_complete = fake({"local": '{"violates": "Session ID is the key for separation"}',
                                   "remote": '{"violates": null}'})
            calls.clear()
            self.assertTrue(self.cli._quality_check(A, "x = 1")[0])
            self.assertEqual(calls, ["local", "remote"])
            # local deny + remote deny -> deny with remote citation
            B.llm_complete = fake({"local": '{"violates": "local cite"}',
                                   "remote": '{"violates": "remote cite", "refine": "fix"}'})
            ok, v, _ = self.cli._quality_check(A, "x = 1")
            self.assertFalse(ok); self.assertEqual(v, "remote cite")
            # plain allow, NOT anchor-adjacent -> no remote call
            B.llm_complete = fake({"local": '{"violates": null}'})
            calls.clear()
            self.assertTrue(self.cli._quality_check(A, "import hashlib")[0])
            self.assertEqual(calls, ["local"])
            # local allow but anchor-adjacent (shares "session"+"separation") -> remote catches it
            B.llm_complete = fake({"local": '{"violates": null}',
                                   "remote": '{"violates": "Session ID is the key for separation"}'})
            calls.clear()
            ok, v, _ = self.cli._quality_check(A, "simplify session separation by keying off cwd")
            self.assertEqual(calls, ["local", "remote"])
            self.assertFalse(ok)
        finally:
            B.llm_complete = orig

    def test_qc_denies_only_on_nameable_violation(self):
        # contradiction-based QC: no cited anchor -> allow; cited -> deny with citation
        import codedouble.backends as B
        orig = B.llm_complete
        try:
            B.llm_complete = lambda *a, **k: '{"violates": null}'
            ok, v, r = self.cli._quality_check({"goal": "g"}, "import hashlib")
            self.assertTrue(ok)
            B.llm_complete = lambda *a, **k: ('{"violates": "Session ID is the key, not project", '
                                              '"refine": "key by session id"}')
            ok, v, r = self.cli._quality_check({"goal": "g"}, "use project id as the key")
            self.assertFalse(ok)
            self.assertIn("Session ID", v)
            # judge crashes / garbage -> fail-open
            B.llm_complete = lambda *a, **k: "not json"
            self.assertTrue(self.cli._quality_check({"goal": "g"}, "x")[0])
        finally:
            B.llm_complete = orig

    def test_bypassed_sendback_is_flagged(self):
        import json, time
        p = os.path.join(self.d, "decisions.jsonl")
        deny = {"ts": time.time(), "event": "PreToolUse", "tool": "Edit", "verdict": "deny",
                "session_id": "sB", "target": "README.md",
                "before": "scope note this section gates the taste module precedent confidence deferred"}
        open(p, "w").write(json.dumps(deny) + "\n")
        # same content lands via Bash -> flagged once, not twice
        ran = "python3 - <<PY ... scope note this section gates the taste module precedent confidence deferred ... PY"
        self.cli._flag_bypassed_sendback(self.log, "sB", "Bash", "python3 - <<PY README.md", ran)
        self.cli._flag_bypassed_sendback(self.log, "sB", "Bash", "python3 - <<PY README.md", ran)
        rows = [json.loads(l) for l in open(p) if l.strip()]
        self.assertEqual(sum(1 for r in rows if r.get("verdict") == "bypassed"), 1)
        self.assertEqual([r for r in rows if r.get("verdict") == "bypassed"][0]["ref"], deny["ts"])
        # an unrelated completed action does not flag
        self.cli._flag_bypassed_sendback(self.log, "sB", "Bash", "ls", "ls -la /tmp totally unrelated listing")
        rows = [json.loads(l) for l in open(p) if l.strip()]
        self.assertEqual(sum(1 for r in rows if r.get("verdict") == "bypassed"), 1)

    def test_norm_anchors_coerces(self):
        out = self.cli._norm_anchors({"goal": 5, "constraints": "notalist", "todos": [" x ", "", 7]},
                                     fallback={"constraints": ["fb"]})
        self.assertEqual(out["goal"], "5")
        self.assertEqual(out["constraints"], ["fb"])   # non-list -> fallback
        self.assertEqual(out["todos"], ["x", "7"])     # trimmed/stringified, empties dropped


class TestExtensionManifest(unittest.TestCase):
    """Locks the VS Code packaging invariants that caused weeks of 'the icon
    never shows up': the activity-bar container must exist, views must live
    under it, and the icon must be alpha-mask-renderable."""
    EXT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "editor", "vscode-codedouble")

    def test_manifest_contributes_activity_bar(self):
        import json
        d = json.load(open(os.path.join(self.EXT, "package.json")))
        bars = d["contributes"]["viewsContainers"]["activitybar"]
        self.assertEqual(len(bars), 1)
        cid = bars[0]["id"]
        self.assertEqual(cid, "codedoubleSidebar")     # reused id owns a visible slot; changing it orphans users' layout state
        self.assertIn(cid, d["contributes"]["views"])  # views must live under the container
        ids = [v["id"] for v in d["contributes"]["views"][cid]]
        self.assertIn("cdCaptureView", ids)            # the provider registers this id
        icon = bars[0]["icon"]
        self.assertTrue(os.path.exists(os.path.join(self.EXT, icon)))
        self.assertTrue(os.path.exists(os.path.join(self.EXT, d["icon"])))  # marketplace png

    def test_activity_icon_is_maskable(self):
        # VS Code alpha-masks activity-bar SVGs: relative (em/%) sizing renders BLANK
        # (an invisible icon in a visible slot — the original weeks-long bug)
        svg = open(os.path.join(self.EXT, "media", "eye.svg")).read()
        self.assertIn("viewBox", svg)
        self.assertNotRegex(svg, r'(width|height)="[^"]*(em|%)')
        self.assertNotIn("<style", svg)                # masked SVGs must not rely on CSS

    def test_vscodeignore_ships_media(self):
        ig = open(os.path.join(self.EXT, ".vscodeignore")).read()
        self.assertNotIn("media", ig)                  # excluding media/ ships a blank container

if __name__ == "__main__":
    unittest.main(verbosity=2)
