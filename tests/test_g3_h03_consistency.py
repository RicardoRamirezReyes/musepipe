"""G3 §7 V5: the Halpha L_acc path in G3 reproduces H03 exactly (same inputs)."""

import unittest

from musepipe.models.accretion import L_SUN_ERG_S, line_lacc
from musepipe.stages.stage_h03_limits import lacc_lsun_from_lha, luminosity_erg_s

REL = {"a": 1.13, "b": 1.74, "scatter_dex": 0.30, "citation": "Alcala+2017"}


class G3H03ConsistencyTests(unittest.TestCase):
    def test_halpha_lacc_matches_h03_chain(self):
        f_obs = 1.136e-16          # observed line flux in erg/s/cm2 (E3 f_lim_observed x 1e-20)
        distance_pc = 138.6
        av = 1.8
        a_ha = 0.818               # A(Halpha)/A_V (E3 value)
        # G3 path
        g3 = line_lacc(f_obs, distance_pc=distance_pc, av=av, a_lambda_over_av=a_ha,
                       relation=REL, flux_unit_cgs=1.0)
        # H03 path recomputed independently
        f_dered = f_obs * (10.0 ** (0.4 * a_ha * av))
        l_line_lsun = luminosity_erg_s(f_dered, distance_pc) / L_SUN_ERG_S
        l_acc_h03 = lacc_lsun_from_lha(l_line_lsun, REL["a"], REL["b"])
        self.assertAlmostEqual(g3["l_line_lsun"], l_line_lsun, places=6)
        self.assertAlmostEqual(g3["l_acc_lsun"] / l_acc_h03, 1.0, places=10)

    def test_flux_unit_scales_luminosity(self):
        g3_native = line_lacc(1.136e4, distance_pc=138.6, av=1.8, a_lambda_over_av=0.818,
                              relation=REL, flux_unit_cgs=1e-20)
        g3_cgs = line_lacc(1.136e-16, distance_pc=138.6, av=1.8, a_lambda_over_av=0.818,
                           relation=REL, flux_unit_cgs=1.0)
        self.assertAlmostEqual(g3_native["l_line_lsun"], g3_cgs["l_line_lsun"], places=6)


if __name__ == "__main__":
    unittest.main()
