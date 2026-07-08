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
        # 3.0 um (30000 A) is outside the optical CCM range
        self.assertTrue(np.isnan(float(law.a_lambda_over_av(30000.0))))


if __name__ == "__main__":
    unittest.main()
