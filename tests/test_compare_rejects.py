import unittest

import numpy as np

from musepipe.stages.stage_x10_compare import validate_product_set
from tests.test_compare_verdicts import make_product, make_products, make_wave


class CompareRejectsTests(unittest.TestCase):
    def test_rejects_wavelength_grid_mismatch_without_resampling(self):
        products = make_products()
        wave = make_wave()
        flux = np.full(wave.size, 100.0, dtype=np.float64)
        wave = wave.copy()
        wave[10] += 0.25
        products["optimal_ls"] = make_product("optimal_ls", flux, wave=wave)

        with self.assertRaisesRegex(ValueError, "wavelength grid"):
            validate_product_set(products)

    def test_rejects_normrad_mismatch(self):
        products = make_products()
        wave = make_wave()
        flux = np.full(wave.size, 100.0, dtype=np.float64)
        products["psffit"] = make_product("psffit", flux, wave=wave, normrad=30.0)

        with self.assertRaisesRegex(ValueError, "NORMRAD"):
            validate_product_set(products)

    def test_rejects_incubesh_mismatch(self):
        products = make_products()
        wave = make_wave()
        flux = np.full(wave.size, 100.0, dtype=np.float64)
        products["optimal_psfsub"] = make_product("optimal_psfsub", flux, wave=wave, incubesh="othercube")

        with self.assertRaisesRegex(ValueError, "INCUBESH"):
            validate_product_set(products)

    def test_rejects_non_box3_aperture_product(self):
        products = make_products()
        products["aperture"].header["APERTURE"] = "box5"

        with self.assertRaisesRegex(ValueError, "APERTURE=box3"):
            validate_product_set(products)

    def test_missing_bkgmode_headers_warn_but_do_not_reject(self):
        products = make_products()  # fixture products predate BKGMODE/SCALEREF

        warnings_list = validate_product_set(products)

        self.assertTrue(warnings_list)
        self.assertTrue(all("BKGMODE" in msg for msg in warnings_list))

    def test_rejects_inconsistent_scaleref(self):
        products = make_products()
        products["psffit"].header["SCALEREF"] = "normrad_total_flux"
        products["aperture"].header["SCALEREF"] = "raw_sum"

        with self.assertRaisesRegex(ValueError, "SCALEREF"):
            validate_product_set(products)


if __name__ == "__main__":
    unittest.main()
