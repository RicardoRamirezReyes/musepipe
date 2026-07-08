"""G4 §7 V4: fabricated matrices for an obvious background M star, an obvious
artifact and a clear companion -> combination returns the right class+robustness."""

import unittest

from musepipe.classify import HYPOTHESES, classify

THRESH = {"secure_bg_prob_max": 0.01, "probable_bg_prob_max": 0.10, "secure_min_supports": 3}


def cell(v):
    return {"verdict": v}


def row(mapping):
    return {h: cell(mapping.get(h, "neutral")) for h in HYPOTHESES}


class G4SyntheticCaseTests(unittest.TestCase):
    def test_clear_companion(self):
        matrix = {
            "T1": row({"substellar_companion": "supports", "m_star_background": "disfavors"}),  # CPM bound
            "T2": row({"substellar_companion": "supports", "contaminant": "disfavors", "artifact": "excludes"}),  # point source
            "T7": row({"artifact": "excludes", "substellar_companion": "supports"}),  # not artifact
            "T3": row({"substellar_companion": "supports", "m_star_background": "disfavors"}),  # young SpT
        }
        out = classify(matrix, background_prob=0.002, thresholds=THRESH)
        self.assertEqual(out["label"], "substellar_companion")
        self.assertEqual(out["robustness"], "secure")

    def test_obvious_background_m_star(self):
        matrix = {
            "T3": row({"m_star_background": "supports", "substellar_companion": "disfavors"}),  # field M SpT
            "T6": row({"m_star_background": "supports"}),  # high A_V (behind cloud)
            "T5": row({"m_star_background": "supports", "substellar_companion": "disfavors"}),  # discrepant RV
            "T7": row({"artifact": "excludes"}),
        }
        out = classify(matrix, background_prob=0.03, thresholds=THRESH)
        self.assertEqual(out["label"], "m_star_background")
        self.assertIn(out["robustness"], ("probable", "secure"))

    def test_obvious_artifact(self):
        matrix = {
            "T2": row({"artifact": "supports", "substellar_companion": "disfavors"}),  # extended/multi-peak
            "T7": row({"artifact": "supports", "substellar_companion": "excludes"}),   # H02 flags it
        }
        out = classify(matrix, background_prob=0.5, thresholds=THRESH)
        self.assertEqual(out["label"], "artifact")

    def test_ambiguous_when_two_hypotheses_tie(self):
        matrix = {
            "T1": row({"substellar_companion": "supports"}),
            "T3": row({"m_star_background": "supports"}),
        }
        out = classify(matrix, background_prob=0.05, thresholds=THRESH)
        self.assertEqual(out["robustness"], "ambiguous")


if __name__ == "__main__":
    unittest.main()
