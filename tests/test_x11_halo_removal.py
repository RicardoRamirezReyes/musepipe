"""La apertura SIN sustraer como referencia comun de los 6 metodos.

La comparacion inter-metodo del gate v3 usa DOS metodos y solo dice si se
parecen entre si. Para responder "¿cuanto quito cada uno?" hace falta una medida
de lo que habia antes de restar nada, y esa medida tiene que ser la MISMA
operacion que hace C2 (misma caja, misma posicion de B3, misma correccion de
apertura) sobre el cubo que entra a 04b: lo unico que puede cambiar es el cubo.
"""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from astropy.io import fits

from musepipe.extraction.product import SpectrumProduct
from musepipe.stages.stage_x11_calibrate import (
    UNSUBTRACTED_CUBE_NAME,
    halo_removal_figure,
    unsubtracted_aperture_reference,
)
from tests.test_calibrate_no_double import calibration_product

METHODS = ("aperture", "optimal_ls", "optimal_psfsub", "psffit", "sgf", "lpm")
COMPANION_YX = (12.0, 16.0)


def _write_cube(path, wave, *, pedestal=100.0, companion=5.0):
    """Cubo sintetico: pedestal plano (el «halo») + fuente en la posicion de B3."""
    cube = np.full((wave.size, 25, 30), float(pedestal), dtype=np.float32)
    cube[:, int(COMPANION_YX[0]), int(COMPANION_YX[1])] += float(companion)
    fits.PrimaryHDU(cube).writeto(path, overwrite=True)


class UnsubtractedReferenceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.stage_dir = Path(self.tmp.name)
        self.wave = np.linspace(7000.0, 7300.0, 121)
        (self.stage_dir / "stage01c_qc.json").write_text(
            json.dumps({"companion": {"pos_yx": list(COMPANION_YX)}}), encoding="utf-8"
        )
        for method in METHODS:
            product = calibration_product(
                method=method, wave=self.wave, flux=np.full(self.wave.size, 10.0)
            )
            product.write(self.stage_dir / f"spec_calibrated_{method}_object.fits")
        _write_cube(self.stage_dir / UNSUBTRACTED_CUBE_NAME, self.wave)

    def tearDown(self):
        self.tmp.cleanup()

    def test_it_is_the_same_box_aperture_on_the_unsubtracted_cube(self):
        ref, why = unsubtracted_aperture_reference(self.stage_dir)
        self.assertIsNone(why)
        # box3 centrada en el compañero = 9 pixeles de pedestal + la fuente.
        np.testing.assert_allclose(ref["flux_box"], 9 * 100.0 + 5.0, rtol=1e-6)
        self.assertEqual(ref["aperture"], "box3")
        self.assertEqual(ref["position_yx"], list(COMPANION_YX))

    def test_without_a_psf_model_the_aperture_correction_is_neutral(self):
        # Sin modelo de PSF no se inventa una correccion: apcorr = 1 y el flujo
        # total coincide con el de la caja (asi la ausencia se ve, no se disimula).
        ref, _ = unsubtracted_aperture_reference(self.stage_dir)
        np.testing.assert_allclose(ref["apcorr"], 1.0)
        np.testing.assert_allclose(ref["flux"], ref["flux_box"])
        self.assertEqual(ref["apcorr_mode"], "none")

    def test_the_continuum_uses_the_same_running_median_as_the_products(self):
        ref, _ = unsubtracted_aperture_reference(self.stage_dir)
        finite = np.isfinite(ref["continuum"])
        self.assertTrue(finite.any())
        np.testing.assert_allclose(ref["continuum"][finite], 905.0, rtol=1e-6)

    def test_a_grid_that_does_not_match_the_products_is_refused(self):
        # Un cubo con otro numero de canales no es comparable con el producto:
        # mejor decirlo que alinear a ojo dos rejillas distintas.
        _write_cube(self.stage_dir / UNSUBTRACTED_CUBE_NAME, np.linspace(7000.0, 7300.0, 99))
        ref, why = unsubtracted_aperture_reference(self.stage_dir)
        self.assertIsNone(ref)
        self.assertIn("no son comparables", why)

    def test_missing_pieces_degrade_with_a_reason(self):
        (self.stage_dir / UNSUBTRACTED_CUBE_NAME).unlink()
        ref, why = unsubtracted_aperture_reference(self.stage_dir)
        self.assertIsNone(ref)
        self.assertIn(UNSUBTRACTED_CUBE_NAME, why)


class HaloRemovalFigureTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.stage_dir = Path(self.tmp.name)
        self.wave = np.linspace(7000.0, 7300.0, 121)
        (self.stage_dir / "stage01c_qc.json").write_text(
            json.dumps({"companion": {"pos_yx": list(COMPANION_YX)}}), encoding="utf-8"
        )
        for method in METHODS:
            product = calibration_product(
                method=method, wave=self.wave, flux=np.full(self.wave.size, 10.0)
            )
            if method == "psffit":
                product.header["CANON"] = True
            product.write(self.stage_dir / f"spec_calibrated_{method}_object.fits")
        _write_cube(self.stage_dir / UNSUBTRACTED_CUBE_NAME, self.wave)

    def tearDown(self):
        self.tmp.cleanup()

    def test_one_row_per_method_canonical_first_and_two_columns(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = halo_removal_figure(self.stage_dir, plt=plt)
        self.assertIsNotNone(fig)
        self.assertEqual(axes.shape, (len(METHODS), 2))
        self.assertIn("psffit", axes[0, 0].get_ylabel())
        plt.close(fig)

    def test_it_degrades_with_a_reason_when_there_is_nothing_to_compare_against(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        (self.stage_dir / UNSUBTRACTED_CUBE_NAME).unlink()
        fig, why = halo_removal_figure(self.stage_dir, plt=plt)
        self.assertIsNone(fig)
        self.assertIn(UNSUBTRACTED_CUBE_NAME, why)


if __name__ == "__main__":
    unittest.main()
