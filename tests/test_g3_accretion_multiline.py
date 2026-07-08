"""G3 §7: multiline L_acc — combined limit = most restrictive, detections
compatibility, Mdot Monte-Carlo."""

import unittest

import numpy as np

from musepipe.models.accretion import combine_accretion, line_lacc, mdot_from_lacc, mdot_mc

REL = {"a": 1.13, "b": 1.74, "scatter_dex": 0.30, "citation": "Alcala+2017", "validity_range": [-5, -1]}


class AccretionMultilineTests(unittest.TestCase):
    def test_relation_requires_citation(self):
        with self.assertRaises(RuntimeError):
            line_lacc(1e-16, distance_pc=138.6, av=1.8, a_lambda_over_av=0.818,
                      relation={"a": 1.13, "b": 1.74, "scatter_dex": 0.30, "citation": None})

    def test_combined_upper_limit_is_most_restrictive(self):
        per_line = [
            {"name": "Ha", "status": "upper_limit", "l_acc_lsun": 2.0e-6},
            {"name": "Hb", "status": "upper_limit", "l_acc_lsun": 5.0e-6},
            {"name": "HeI", "status": "upper_limit", "l_acc_lsun": 9.0e-6},
        ]
        out = combine_accretion(per_line)
        self.assertEqual(out["kind"], "upper_limit")
        self.assertEqual(out["l_acc_lsun"], 2.0e-6)  # smallest = most restrictive
        self.assertEqual(out["from_line"], "Ha")

    def test_compatible_detections_combine(self):
        per_line = [
            {"name": "Ha", "status": "detected", "l_acc_lsun": 1.0e-4, "l_acc_err_lsun": 2e-5, "scatter_dex": 0.3},
            {"name": "Hb", "status": "detected", "l_acc_lsun": 1.2e-4, "l_acc_err_lsun": 3e-5, "scatter_dex": 0.3},
        ]
        out = combine_accretion(per_line)
        self.assertEqual(out["kind"], "detection")
        self.assertTrue(out["compatible"])
        self.assertTrue(1.0e-4 <= out["l_acc_lsun"] <= 1.2e-4)

    def test_incompatible_detections_flagged_discrepant(self):
        per_line = [
            {"name": "Ha", "status": "detected", "l_acc_lsun": 1.0e-4, "l_acc_err_lsun": 1e-6, "scatter_dex": 0.02},
            {"name": "Hb", "status": "detected", "l_acc_lsun": 1.0e-2, "l_acc_err_lsun": 1e-4, "scatter_dex": 0.02},
        ]
        out = combine_accretion(per_line)
        self.assertEqual(out["status"], "discrepant")
        self.assertIsNone(out["l_acc_lsun"])

    def test_mdot_scales_with_lacc(self):
        a = mdot_from_lacc(1e-6, 0.0167, 0.135)
        b = mdot_from_lacc(2e-6, 0.0167, 0.135)
        self.assertAlmostEqual(b / a, 2.0, places=6)

    def test_mdot_mc_returns_ordered_percentiles(self):
        out = mdot_mc(2e-6, 0.0167, 0.0014, 0.135, 0.017, 0.30, n_mc=1000, seed=0)
        self.assertLess(out["p16"], out["p50"])
        self.assertLess(out["p50"], out["p84"])


if __name__ == "__main__":
    unittest.main()
