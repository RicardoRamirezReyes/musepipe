"""D2 calibra tambien la PRIMARIA, con los dos terminos de error separados.

Antes, `spec_psffit_star.fits` (que escribe C4) no pasaba por D2: se quedaba en
7 columnas, sin correccion en lambda, sin escala de flujo y sin presupuesto de
error, frente a las 18 del compañero. Opcion (iii) del diseño: el empirico de
anillo y el presupuesto de sistematicos viven en columnas DISTINTAS, porque
miden cosas distintas (estabilidad del ajuste vs incertidumbre de calibracion).
"""
import tempfile
import unittest
from pathlib import Path

import numpy as np

from musepipe.extraction.product import SpectrumProduct
from musepipe.stages.stage_x11_calibrate import (
    CalibrationCorrections,
    calibrate_star_product,
    stage_x11_paths,
)


def _star_product(n=64):
    wave = np.linspace(5000.0, 9000.0, n)
    flux = 1000.0 + 10.0 * np.sin(wave / 400.0)
    return SpectrumProduct(
        wave_A=wave,
        flux=flux,
        flux_err=np.full(n, 5.0),
        flux_err_emp=np.full(n, 7.0),
        apcorr=np.ones(n),
        npix_eff=np.full(n, 30.0),
        flags=np.zeros(n, dtype=np.int32),
        # Mismas claves que deja C4 en el producto real de la primaria: D2
        # rechaza una cabecera ambigua (sin WFRAME/APCMODE/ERRMODE).
        header={"FORMATV": 1, "METHOD": "psffit", "RUNID": "T", "APERTURE": "psffit_star",
                "SRCPOS_Y": 10.0, "SRCPOS_X": 10.0, "WFRAME": "barycentric",
                "INCUBE": "cube.fits", "INCUBESH": "0" * 64, "NORMRAD": 25.0,
                "APCMODE": "psf_model_norm_radius", "ERRMODE": "empirical"},
    )


def _corrections():
    return CalibrationCorrections(
        wavelength_status="green", wavelength_offset_A=0.07, wavelength_apply=True,
        frame_final="barycentric", flux_scale=1.0, flux_scale_err_frac=0.03,
        psf_frac=0.02,
    )


class StarCalibrationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.paths = stage_x11_paths("T", root)
        self.paths["paths"].ensure_base_dirs()
        self.cfg = {"run_id": "T", "project_root": str(root)}

    def tearDown(self):
        self.tmp.cleanup()

    def _calibrate(self):
        _star_product().write(self.paths["spec_psffit_star"], overwrite=True)
        return calibrate_star_product(self.cfg, self.paths, _corrections())

    def test_missing_star_product_degrades_instead_of_raising(self):
        cal, summary = calibrate_star_product(self.cfg, self.paths, _corrections())
        self.assertIsNone(cal)
        self.assertFalse(summary["available"])
        self.assertIn("spec_psffit_star.fits", summary["reason"])

    def test_star_gets_the_same_budget_columns_as_the_companion(self):
        cal, _ = self._calibrate()
        cols = set((cal.product.extra_columns or {}))
        for name in ("flux_err_stat", "flux_err_total", "sys_fluxcal", "sys_psf",
                     "sys_sky", "sys_telluric", "sys_continuum"):
            self.assertIn(name, cols)

    def test_ring_empirical_stays_in_its_own_column(self):
        # El empirico NO se funde con el presupuesto: sigue accesible aparte y
        # es distinto del total, que ademas lo supera al incluir sistematicos.
        cal, _ = self._calibrate()
        emp = np.asarray(cal.product.flux_err_emp, dtype=np.float64)
        total = np.asarray(cal.product.extra_columns["flux_err_total"], dtype=np.float64)
        self.assertTrue(np.all(emp > 0))
        self.assertFalse(np.allclose(emp, total))
        self.assertTrue(np.all(total >= emp))

    def test_budget_dominates_over_the_statistical_term_for_a_bright_star(self):
        cal, _ = self._calibrate()
        extra = cal.product.extra_columns
        stat = np.median(np.asarray(extra["flux_err_stat"], dtype=np.float64))
        fluxcal = np.median(np.asarray(extra["sys_fluxcal"], dtype=np.float64))
        self.assertGreater(fluxcal, stat)

    def test_wavelength_and_flux_calibration_are_applied(self):
        cal, _ = self._calibrate()
        header = cal.product.header
        self.assertTrue(header["WLCORR"])
        self.assertEqual(header["CALSTAGE"], "x11")
        self.assertEqual(header["SOURCE"], "primary")
        self.assertIn("ring", header["EMPSRC"])
        np.testing.assert_allclose(cal.product.wave_A, _star_product().wave_A - 0.07)

    def test_summary_reports_both_error_blocks(self):
        _, summary = self._calibrate()
        self.assertTrue(summary["available"])
        self.assertEqual(summary["error_terms"]["empirical_ring"], "flux_err_emp")
        self.assertIn("sys_fluxcal", summary["error_terms"]["budget"])
        self.assertGreater(summary["median_snr_emp_only"], summary["median_snr_total"])


if __name__ == "__main__":
    unittest.main()
