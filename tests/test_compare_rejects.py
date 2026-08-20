import unittest

import numpy as np

from musepipe.stages.stage_x10_compare import DERIVED_CUBE_SOURCES, validate_product_set
from musepipe.stages.stage_x02_optimal import PEROBS_CUBE_SOURCE
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

    def _psfsub_from_another_cube(self, cubesrc):
        """psfsub sacado de OTRO cubo, declarando (o no) de cual."""
        products = make_products()
        wave = make_wave()
        flux = np.full(wave.size, 100.0, dtype=np.float64)
        product = make_product("optimal_psfsub", flux, wave=wave, incubesh="othercube")
        if cubesrc is not None:
            product.header["CUBESRC"] = cubesrc
        products["optimal_psfsub"] = product
        return products

    def test_accepts_incubesh_mismatch_when_the_product_declares_a_known_source(self):
        # C1b resta la primaria en cada exposicion: el psfsub sale de SU cubo y
        # su hash no puede coincidir. Se acepta porque el producto lo declara.
        products = self._psfsub_from_another_cube(PEROBS_CUBE_SOURCE)

        warnings_list = validate_product_set(products)

        self.assertTrue(any(PEROBS_CUBE_SOURCE in msg and "optimal_psfsub" in msg
                            for msg in warnings_list),
                        f"la procedencia tiene que quedar dicha, no en silencio: {warnings_list}")

    def test_rejects_incubesh_mismatch_when_the_declared_source_is_unknown(self):
        products = self._psfsub_from_another_cube("un_cubo_cualquiera")

        with self.assertRaisesRegex(ValueError, "CUBESRC"):
            validate_product_set(products)

    def test_rejects_incubesh_mismatch_declared_on_the_wrong_method(self):
        # La excepcion es de la procedencia, no del metodo: si quien cambia de
        # cubo es otro, tiene que declararlo igual.
        products = make_products()
        wave = make_wave()
        flux = np.full(wave.size, 100.0, dtype=np.float64)
        products["psffit"] = make_product("psffit", flux, wave=wave, incubesh="othercube")

        with self.assertRaisesRegex(ValueError, "CUBESRC"):
            validate_product_set(products)

    def test_the_source_c3_stamps_is_one_d1_recognizes(self):
        # La clase de fallo: dos literales sueltos que se separan sin que nadie
        # lo vea. Ambos lados salen del slug de C1b en `stage_registry`.
        self.assertIn(PEROBS_CUBE_SOURCE, DERIVED_CUBE_SOURCES)

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
