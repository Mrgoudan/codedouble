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


if __name__ == "__main__":
    unittest.main(verbosity=2)
