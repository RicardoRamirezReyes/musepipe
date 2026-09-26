"""G3 §7: extinction law vs tabulated CCM values; citation mandatory."""

import unittest

import numpy as np

from musepipe.models.extinction import CCMExtinction


class ExtinctionTests(unittest.TestCase):
    def test_citation_is_mandatory(self):
        with self.assertRaises(RuntimeError):
            CCMExtinction(rv=3.1, citation=None)

    def test_ccm_values_match_tabulated(self):
        law = CCMExtinction(rv=3.1, citation="Cardelli+1989")
        # A(Halpha 6563)/A_V ~ 0.81-0.82 for R_V=3.1 (standard, matches E3 config 0.818)
        a_ha = float(law.a_lambda_over_av(6562.8))
        self.assertAlmostEqual(a_ha, 0.818, delta=0.02)
        # A(lambda) decreases toward the red
        a_red = float(law.a_lambda_over_av(8446.0))
        self.assertLess(a_red, a_ha)

    def test_deredden_factor_increases_flux(self):
        law = CCMExtinction(rv=3.1, citation="Cardelli+1989")
        factor = law.deredden_factor(6562.8, av=1.8)
        self.assertGreater(factor, 1.0)  # dereddening brightens

    def test_out_of_range_is_nan(self):
        law = CCMExtinction(rv=3.1, citation="Cardelli+1989")
        # 4.0 um (x = 0.25 um^-1) is outside CCM89 altogether (IR branch: x >= 0.3)
        self.assertTrue(np.isnan(float(law.a_lambda_over_av(40000.0))))

    def test_ir_branch_covers_the_g3_range(self):
        """La rama IR de CCM89 (x < 1.1): sin ella, λ > 9091 Å era NaN y los
        canales 9091–9349.5 Å se caían en silencio de todo modelo con A_V > 0."""
        law = CCMExtinction(rv=3.1, citation="Cardelli+1989")
        a = law.a_lambda_over_av(np.array([9200.0, 9349.5, 22000.0]))
        self.assertTrue(np.all(np.isfinite(a)))
        # K band (2.2 um): CCM89 A_K/A_V ~ 0.11-0.12 for R_V=3.1
        self.assertAlmostEqual(float(a[2]), 0.117, delta=0.01)
        # continuous across x = 1.1 (9090.9 A) within 1 %
        lo = float(law.a_lambda_over_av(9090.0))
        hi = float(law.a_lambda_over_av(9092.0))
        self.assertLess(abs(lo - hi) / lo, 0.01)


if __name__ == "__main__":
    unittest.main()
