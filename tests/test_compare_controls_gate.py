import unittest

import numpy as np

from musepipe.stages.stage_x10_compare import compare_methods
from tests.test_compare_verdicts import clean_controls, make_products, make_wave, synthetic_g1


def rough_controls(wave, n_controls=8, sigma=1.0, seed=11):
    """Noisy (non-degenerate) controls shared by all methods plus own noise."""

    rng = np.random.default_rng(seed)
    out = {}
    for method in ("aperture", "optimal_ls", "optimal_psfsub", "psffit"):
        out[method] = rng.normal(0.0, sigma, size=(n_controls, wave.size))
    return out


class CompareControlsGateTests(unittest.TestCase):
    def test_single_correlated_outlier_does_not_create_binomial_excess_per_observable(self):
        wave = make_wave()
        controls = rough_controls(wave)
        # One control sits on an artifact: a strong offset in every band makes
        # its centred t an outlier in each band -> binomial excess -> dirty.
        controls["psffit"] = controls["psffit"].copy()
        controls["psffit"][0] += 50.0
        products = make_products()

        _rows, _controls, qc = compare_methods(products, controls)

        self.assertEqual(qc["verdict"], "consistent")
        self.assertEqual(qc["pairs_degraded"], {})

    def test_uniform_additive_bias_is_absorbed_by_centring(self):
        # A constant offset common to ALL controls of a method (comparable to
        # the control scatter, like the real psffit halo level) is that
        # method's additive bias at the same-radius ring: it affects object
        # and controls alike, and spec v2 §3.4 absorbs it with the mu
        # centring, so the pair stays clean AND consistent.
        wave = make_wave()
        controls = rough_controls(wave)
        controls["psffit"] = controls["psffit"] + 1.0
        products = make_products({"psffit": [(float(wave[0]), float(wave[-1]), 1.0)]})

        _rows, _controls, qc = compare_methods(products, controls)

        self.assertEqual(qc["verdict"], "consistent")
        self.assertEqual(qc["pairs_degraded"], {})
        self.assertTrue(qc["scale_check"]["psffit_vs_optimal_psfsub"]["ok"])

    def test_secondary_pair_dirty_does_not_degrade_verdict(self):
        wave = make_wave()
        controls = rough_controls(wave)
        controls["aperture"] = controls["aperture"].copy()
        controls["aperture"][0] += 50.0
        products = make_products()

        _rows, _controls, qc = compare_methods(products, controls)

        self.assertEqual(qc["verdict"], "consistent")
        self.assertEqual(qc["pairs_degraded"], {})
        self.assertEqual(qc["verdict_by_pair"]["psffit_vs_aperture"]["role"], "secondary")

    def test_one_dirty_primary_of_three_keeps_verdict_from_clean_pairs(self):
        g1 = synthetic_g1(
            verdicts={
                "psffit": "validated_with_bias",
                "optimal_psfsub": "validated_with_bias",
                "aperture": "validated",
                "optimal_ls": "rejected",
            }
        )
        wave = make_wave()
        controls = rough_controls(wave)
        controls["aperture"] = controls["aperture"].copy()
        controls["aperture"][0] += 50.0
        products = make_products()

        _rows, _controls, qc = compare_methods(products, controls, g1_inputs=g1)

        self.assertIn("psffit_vs_aperture", qc["primary_pairs"])
        self.assertNotIn("psffit_vs_aperture", qc["pairs_degraded"])
        self.assertNotIn("optimal_psfsub_vs_aperture", qc["pairs_degraded"])
        self.assertEqual(qc["verdict"], "consistent")
        self.assertNotIn("psffit_vs_optimal_psfsub", qc["pairs_degraded"])


# Control positions of both paper runs.
REAL_N_CONTROLS = 33


class CompareControlsGateCanFireTests(unittest.TestCase):
    """2026-10-24, added with the author's approval.

    49a9dd9 (D1 v4) inverted the three tests above that showed this gate
    degrading a pair, and with their 8 controls it cannot: a control's t uses
    the sample's own scatter, so one outlier among 8 reaches at most
    |t| = 7/sqrt(8) = 2.47 (6 bad rows of 48, excess p = 0.021 > 0.01) and two
    or more hide each other. These use the 33 controls of the real runs.
    """

    def _continuum_gate(self, n_dirty):
        wave = make_wave()
        controls = rough_controls(wave, n_controls=REAL_N_CONTROLS)
        controls["psffit"] = controls["psffit"].copy()
        controls["psffit"][:n_dirty] += 50.0
        _rows, _controls, qc = compare_methods(make_products(), controls)
        return qc, qc["controls"]["by_observable"]["continuum"]["psffit_vs_optimal_psfsub"]

    def test_three_of_33_controls_on_an_artifact_degrade_the_pair(self):
        qc, gate = self._continuum_gate(3)

        self.assertFalse(gate["clean"])
        self.assertEqual(gate["reason"], "control_bad_fraction_excess")
        self.assertIn("psffit_vs_optimal_psfsub", qc["pairs_degraded"])

    def test_a_quarter_of_the_controls_on_an_artifact_still_degrade_the_pair(self):
        # Fails on 2026-10-24: from 7 of 33 dirty controls on, the dirty ones
        # inflate the scatter that judges them and no row is bad (0 of 198), so
        # the gate reads clean when the controls are worst. It fires again only
        # at ~16 of 33, through the scale-check.
        qc, gate = self._continuum_gate(8)

        self.assertFalse(gate["clean"])
        self.assertIn("psffit_vs_optimal_psfsub", qc["pairs_degraded"])


if __name__ == "__main__":
    unittest.main()
