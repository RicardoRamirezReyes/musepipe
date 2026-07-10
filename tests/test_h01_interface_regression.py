import unittest

from musepipe.stages.stage_h01_detect import HALPHA_REST_A, analyze_halpha_method
from tests.test_h01_helpers import h01_controls, h01_product


class H01InterfaceRegressionTests(unittest.TestCase):
    def test_spectrum_product_interface_recovers_injected_matched_flux(self):
        product = h01_product("psffit", signal_flux=8.0, rv_sys_kms=0.0, lsf_fwhm_A=2.5)
        controls = h01_controls(noise=0.0)

        result = analyze_halpha_method(
            "psffit",
            product,
            controls,
            rv_sys_kms=0.0,
            lsf_fwhm_A=2.5,
            rest_A=HALPHA_REST_A,
            width_factors=(1.0,),
        )

        self.assertLess(abs(result.row["matched_flux"] - 8.0) / 8.0, 0.02)
        self.assertLess(result.row["global_empirical_fap"], 0.01)
        self.assertTrue(result.row["rv_consistent"])


if __name__ == "__main__":
    unittest.main()
