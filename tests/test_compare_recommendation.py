"""D1 v3 · frozen recommendation tree (spec_D1_v3 §2, WP-H3)."""

import unittest

from musepipe.stages.stage_x10_compare import (
    RECOMMENDATION_PREFERENCE,
    recommend_method_v3,
)


def predictors(value, line="halpha", in_range=True):
    return [{"line": line, "rest_A": 6562.8, "in_range": in_range, "predictor": value, "R": 0.01, "cs_over_ls": 1.0}]


class RecommendationTreeTests(unittest.TestCase):
    def test_consistent_all_validated_prefers_psffit(self):
        verdicts = {m: "validated" for m in RECOMMENDATION_PREFERENCE}
        out = recommend_method_v3("consistent", verdicts, {"companion_continuum_is_science": True})
        self.assertEqual(out["recommended_method"], "psffit")

    def test_lpm_recommended_when_psffit_not_validated(self):
        verdicts = {"lpm": "validated_with_bias", "optimal_psfsub": "validated"}
        out = recommend_method_v3("consistent", verdicts, {})
        self.assertEqual(out["recommended_method"], "lpm")

    def test_sgf_excluded_when_continuum_is_science(self):
        verdicts = {"sgf": "validated"}
        out = recommend_method_v3(
            "consistent", verdicts,
            {"companion_continuum_is_science": True, "sgf_predictors": predictors(0.01)},
        )
        self.assertIsNone(out["recommended_method"])
        self.assertIsNotNone(out["method_caveats"]["sgf"]["excluded_from_recommendation"])

    def test_sgf_allowed_for_line_only_science_with_small_predictor(self):
        verdicts = {"sgf": "validated"}
        out = recommend_method_v3(
            "consistent", verdicts,
            {"companion_continuum_is_science": False, "sgf_predictors": predictors(-0.02)},
        )
        self.assertEqual(out["recommended_method"], "sgf")

    def test_sgf_excluded_by_predictor_over_threshold(self):
        verdicts = {"sgf": "validated"}
        out = recommend_method_v3(
            "consistent", verdicts,
            {"companion_continuum_is_science": False, "sgf_predictors": predictors(-0.5)},
        )
        self.assertIsNone(out["recommended_method"])
        self.assertIn("Eq. 1", out["method_caveats"]["sgf"]["excluded_from_recommendation"])

    def test_out_of_range_predictor_does_not_exclude(self):
        verdicts = {"sgf": "validated"}
        out = recommend_method_v3(
            "consistent", verdicts,
            {"companion_continuum_is_science": False,
             "sgf_predictors": predictors(-0.5, in_range=False)},
        )
        self.assertEqual(out["recommended_method"], "sgf")

    def test_no_recommendation_when_not_consistent(self):
        verdicts = {m: "validated" for m in RECOMMENDATION_PREFERENCE}
        out = recommend_method_v3("divergent_continuum", verdicts, {})
        self.assertIsNone(out["recommended_method"])
        # Caveats and rules are still emitted for the checkpoint.
        self.assertIn("sgf", out["method_caveats"])
        self.assertEqual(out["rules"]["candidates_validated"][0], "psffit")

    def test_no_validated_candidates(self):
        out = recommend_method_v3("consistent", {}, {})
        self.assertIsNone(out["recommended_method"])
        self.assertEqual(out["rules"]["eligible"], [])


if __name__ == "__main__":
    unittest.main()
