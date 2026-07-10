import unittest

from musepipe.stages.stage_h01_detect import HALPHA_REST_A, analyze_halpha_method
from tests.test_h01_helpers import h01_controls, h01_product


class H01TemplateDimensionTests(unittest.TestCase):
    def test_three_template_global_fap_counts_more_trials_than_one_template(self):
        product = h01_product("psffit", signal_flux=1.5, rv_sys_kms=0.0, lsf_fwhm_A=2.5)
        controls = h01_controls(noise=0.5, seed=9)

        one = analyze_halpha_method(
            "psffit",
            product,
            controls,
            rv_sys_kms=0.0,
            lsf_fwhm_A=2.5,
            rest_A=HALPHA_REST_A,
            width_factors=(1.0,),
        )
        three = analyze_halpha_method(
            "psffit",
            product,
            controls,
            rv_sys_kms=0.0,
            lsf_fwhm_A=2.5,
            rest_A=HALPHA_REST_A,
            width_factors=(1.0, 2.0, 4.0),
        )

        self.assertEqual(one.row["template_factor"], 1.0)
        self.assertEqual(three.row["template_factor"], 1.0)
        self.assertGreaterEqual(three.row["global_empirical_fap"], one.row["global_empirical_fap"])


if __name__ == "__main__":
    unittest.main()
