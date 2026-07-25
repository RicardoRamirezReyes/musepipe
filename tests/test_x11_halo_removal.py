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
    CONTROL_DELTA_PA_DEG,
    UNSUBTRACTED_CUBE_NAME,
    halo_remaining_figure,
    halo_removal_figure,
    photometry_map_figure,
    unsubtracted_aperture_reference,
)
from tests.test_calibrate_no_double import calibration_product

METHODS = ("aperture", "optimal_ls", "optimal_psfsub", "psffit", "sgf", "lpm")
PRIMARY_YX = (20.0, 20.0)
COMPANION_YX = (20.0, 28.0)
NY = NX = 41


def _write_cube(path, wave, *, pedestal=100.0, companion=5.0, shape=(NY, NX)):
    """Cubo sintetico: pedestal plano (el «halo») + fuente en la posicion de B3."""
    cube = np.full((wave.size, *shape), float(pedestal), dtype=np.float32)
    cube[:, int(COMPANION_YX[0]), int(COMPANION_YX[1])] += float(companion)
    fits.PrimaryHDU(cube).writeto(path, overwrite=True)


def _write_b3_qc(stage_dir, *, primary=PRIMARY_YX, companion=COMPANION_YX):
    """QC de B3 con lo que la referencia necesita para colocar los controles."""
    dy = companion[0] - primary[0]
    dx = companion[1] - primary[1]
    sep = float(np.hypot(dy, dx))
    pa = float(np.degrees(np.arctan2(dx, -dy)) % 360.0)
    (Path(stage_dir) / "stage01c_qc.json").write_text(json.dumps({
        "primary": {"pos_yx": list(primary)},
        "companion": {"pos_yx": list(companion)},
        "pixel_scale_arcsec": 1.0,
        "wcs_orientation": {"north_angle_deg": 0.0},
        "astrometry": {"sep_arcsec": sep, "pa_deg": pa},
    }), encoding="utf-8")
    return sep, pa


class UnsubtractedReferenceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.stage_dir = Path(self.tmp.name)
        self.wave = np.linspace(7000.0, 7300.0, 121)
        self.sep, self.pa = _write_b3_qc(self.stage_dir)
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

    def test_the_controls_sit_at_the_same_separation_and_at_the_asked_angles(self):
        ref, _ = unsubtracted_aperture_reference(self.stage_dir)
        self.assertEqual([c["name"] for c in ref["controls"]],
                         [name for name, _ in CONTROL_DELTA_PA_DEG])
        primary = np.asarray(PRIMARY_YX, dtype=np.float64)
        for control, (_, delta) in zip(ref["controls"], CONTROL_DELTA_PA_DEG):
            pos = np.asarray(control["position_yx"], dtype=np.float64)
            # misma separacion de la primaria: son el MISMO halo, otro angulo
            self.assertAlmostEqual(float(np.hypot(*(pos - primary))), self.sep, places=6)
            self.assertAlmostEqual(control["pa_deg"], (self.pa + delta) % 360.0, places=6)

    def test_the_controls_are_pure_halo_and_the_companion_is_the_difference(self):
        # El cubo sintetico tiene pedestal plano + una fuente solo en el
        # compañero: los tres controles deben dar exactamente el pedestal.
        ref, _ = unsubtracted_aperture_reference(self.stage_dir)
        for control in ref["controls"]:
            np.testing.assert_allclose(control["flux"], 9 * 100.0, rtol=1e-6)
        np.testing.assert_allclose(ref["flux"] - ref["controls"][0]["flux"], 5.0, rtol=1e-6)

    def test_the_same_aperture_correction_is_used_for_the_four_positions(self):
        # Es la convencion del pipeline (aperture.py: control_spectra_cal): con
        # una apcorr por posicion, su fase subpixel entraria en la comparacion.
        ref, _ = unsubtracted_aperture_reference(self.stage_dir)
        for control in ref["controls"]:
            np.testing.assert_allclose(
                np.asarray(control["flux"]) / (9 * 100.0), ref["apcorr"], rtol=1e-9
            )

    def test_a_control_outside_the_cube_is_skipped_with_a_reason(self):
        # Compañero pegado al borde: uno de los perpendiculares se sale.
        _write_b3_qc(self.stage_dir, primary=(8.0, 20.0), companion=(8.0, 28.0))
        ref, why = unsubtracted_aperture_reference(self.stage_dir)
        self.assertIsNone(why)
        self.assertEqual(len(ref["controls"]), 2)
        self.assertEqual(len(ref["controls_skipped"]), 1)
        self.assertIn("fuera del cubo", ref["controls_skipped"][0]["reason"])

    def test_controls_can_be_turned_off(self):
        ref, _ = unsubtracted_aperture_reference(self.stage_dir, with_controls=False)
        self.assertEqual(ref["controls"], [])


class HaloRemovalFigureTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.stage_dir = Path(self.tmp.name)
        self.wave = np.linspace(7000.0, 7300.0, 121)
        self.sep, self.pa = _write_b3_qc(self.stage_dir)
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

    def test_one_row_per_method_canonical_first_in_both_figures(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # halo_removal lleva dos columnas (lo que deja el metodo | la misma
        # resta en los controles); halo_remaining es de una sola.
        for figure, shape in ((halo_removal_figure, (len(METHODS), 2)),
                              (halo_remaining_figure, (len(METHODS),))):
            fig, axes = figure(self.stage_dir, plt=plt)
            self.assertIsNotNone(fig, figure.__name__)
            self.assertEqual(axes.shape, shape, figure.__name__)
            first = axes[0, 0] if axes.ndim == 2 else axes[0]
            self.assertIn("psffit", first.get_ylabel())
            plt.close(fig)

    def test_the_control_panel_shows_each_control_before_and_after_the_subtraction(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        reference, _ = unsubtracted_aperture_reference(self.stage_dir)
        fig, axes = halo_removal_figure(self.stage_dir, plt=plt, reference=reference)
        n_controls = len(reference["controls"])
        # Por fila: los 3 controles tal cual + los 3 tras la resta del metodo.
        self.assertEqual(len(axes[0, 1].lines), 2 * n_controls + 1)  # +1 = la linea del cero
        plt.close(fig)

    def test_the_map_marks_the_four_apertures_and_the_primary(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = photometry_map_figure(self.stage_dir, plt=plt, channel_step=1)
        self.assertIsNotNone(fig)
        labels = {t.get_text() for t in ax.texts}
        self.assertIn("compañero", labels)
        self.assertIn("primaria", labels)
        self.assertEqual(sum(1 for t in ax.texts if "perpendicular" in t.get_text()), 2)
        plt.close(fig)

    def test_a_precomputed_reference_is_reused_instead_of_measured_again(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        reference, _ = unsubtracted_aperture_reference(self.stage_dir)
        # Si la figura volviera a medir, el cubo borrado la haria fallar.
        (self.stage_dir / UNSUBTRACTED_CUBE_NAME).unlink()
        fig, axes = halo_remaining_figure(self.stage_dir, plt=plt, reference=reference)
        self.assertIsNotNone(fig)
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
