"""G4 §7/§4.4: if the leader changes under leave-one-out, class is forced ambiguous."""

import unittest

from musepipe.classify import HYPOTHESES, classify, leave_one_out_leader

THRESH = {"secure_bg_prob_max": 0.01, "probable_bg_prob_max": 0.10, "secure_min_supports": 3}


def cell(v):
    return {"verdict": v}


def only(hyp, verdict="supports", rest="neutral"):
    return {h: cell(verdict if h == hyp else rest) for h in HYPOTHESES}


class G4LeaveOneOutTests(unittest.TestCase):
    def test_stable_leader(self):
        matrix = {"T1": only("substellar_companion"), "T2": only("substellar_companion"),
                  "T7": only("substellar_companion")}
        leader, loo = leave_one_out_leader(matrix)
        self.assertEqual(leader, "substellar_companion")
        self.assertTrue(all(v == "substellar_companion" for v in loo.values()))

    def test_leader_change_forces_ambiguous(self):
        # T1 favours A, T2 favours B; dropping either flips the leader
        matrix = {"T1": only("substellar_companion"), "T2": only("m_star_background")}
        out = classify(matrix, background_prob=0.001, thresholds=THRESH)
        self.assertFalse(out["leave_one_out_stable"])
        self.assertEqual(out["robustness"], "ambiguous")

    def test_secure_requires_three_independent_supports(self):
        matrix = {"T1": only("substellar_companion"), "T2": only("substellar_companion"),
                  "T7": only("substellar_companion")}
        out = classify(matrix, background_prob=0.001, thresholds=THRESH)
        self.assertEqual(out["label"], "substellar_companion")
        self.assertEqual(out["robustness"], "secure")

    def test_two_supports_is_probable_not_secure(self):
        matrix = {"T1": only("substellar_companion"), "T2": only("substellar_companion")}
        out = classify(matrix, background_prob=0.001, thresholds=THRESH)
        self.assertEqual(out["robustness"], "probable")

    def test_high_background_downgrades(self):
        matrix = {"T1": only("m_star_background"), "T2": only("m_star_background"),
                  "T7": only("m_star_background")}
        out = classify(matrix, background_prob=0.05, thresholds=THRESH)  # 5% -> not secure
        self.assertEqual(out["robustness"], "probable")


if __name__ == "__main__":
    unittest.main()
