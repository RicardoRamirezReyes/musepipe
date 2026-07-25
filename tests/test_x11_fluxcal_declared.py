"""El ~3% de calibracion absoluta vs Gaia: declarado, no plegado.

A4/M3 mide `flux_factor` (0.973 en ROXs 12 b) pero no publica barra de error, asi
que `flux_scale_err_frac` sale 0 y `sys_fluxcal` era identicamente cero: el
desvio que M3 acababa de medir no aparecia en ningun sitio del presupuesto.
Decision (2026-07-25): se declara como |1 - flux_factor| en su propia columna
`sys_fluxcal_declared` y **no** se suma a `flux_err_total`, porque el error del
compañero sostiene decisiones congeladas (E1/E3/G3).
"""
import unittest

import numpy as np

from musepipe.stages.stage_x11_calibrate import (
    calibrate_spectrum_product,
    calibration_corrections_from_qc,
)
from tests.test_calibrate_no_double import calibration_product


def _qc00(m3):
    return {
        "cube": {"wavelength_frame": "barycentric"},
        "m1_wavelength": {"status": "green", "offset_median_A": 0.0},
        "m3_flux": m3,
    }


def _calibrate(m3):
    corrections = calibration_corrections_from_qc(_qc00(m3))
    product = calibration_product(flux=np.full(31, 100.0))
    return corrections, calibrate_spectrum_product(product, corrections, method="psffit", canonical=True)


class DeclaredFluxcalTests(unittest.TestCase):
    def test_the_measured_departure_from_unity_becomes_the_declared_term(self):
        corrections, cal = _calibrate({"status": "green", "flux_factor": 0.973, "band": "RP"})
        self.assertAlmostEqual(corrections.flux_declared_err_frac, 0.027, places=6)
        self.assertIn("flux_factor", corrections.flux_declared_source)
        declared = np.asarray(cal.product.extra_columns["sys_fluxcal_declared"], dtype=np.float64)
        np.testing.assert_allclose(declared, 100.0 * 0.027, rtol=1e-9)

    def test_it_is_not_folded_into_the_total(self):
        # El total debe salir igual que con el termino declarado a cero: es la
        # condicion que protege la ciencia congelada.
        _, cal = _calibrate({"status": "green", "flux_factor": 0.973, "band": "RP"})
        _, flat = _calibrate({"status": "green"})
        extra = cal.product.extra_columns
        applied = np.sqrt(
            np.asarray(extra["flux_err_stat"], dtype=np.float64) ** 2
            + np.asarray(extra["sys_fluxcal"], dtype=np.float64) ** 2
            + np.asarray(extra["sys_psf"], dtype=np.float64) ** 2
            + np.asarray(extra["sys_sky"], dtype=np.float64) ** 2
            + np.asarray(extra["sys_telluric"], dtype=np.float64) ** 2
        )
        np.testing.assert_allclose(extra["flux_err_total"], applied, rtol=1e-12)
        np.testing.assert_allclose(extra["flux_err_total"],
                                   flat.product.extra_columns["flux_err_total"], rtol=1e-12)
        self.assertTrue(np.all(np.asarray(extra["sys_fluxcal_declared"]) > 0))

    def test_the_budget_says_it_is_declared_and_not_applied(self):
        _, cal = _calibrate({"status": "green", "flux_factor": 0.973, "band": "RP"})
        row = next(r for r in cal.error_budget if r["term"] == "fluxcal_declared")
        self.assertEqual(row["type"], "declared_not_applied")
        self.assertAlmostEqual(row["value"], 0.027, places=6)
        self.assertIn("NO entra en `flux_err_total`", row["note"])
        self.assertIn("g3_sys_fluxcal_frac", row["note"])

    def test_the_header_carries_the_declared_fraction_and_its_warning(self):
        _, cal = _calibrate({"status": "green", "flux_factor": 0.973, "band": "RP"})
        header = cal.product.header
        self.assertAlmostEqual(header["SYSFLXD"], 0.027, places=6)
        self.assertIn("NOT in flux_err_total", header["SYSFLXDN"])

    def test_a_real_error_bar_from_m3_takes_over_and_nothing_is_declared(self):
        # Si M3 llega a publicar su incertidumbre, esa se APLICA y el termino
        # declarado desaparece: no se cuenta el mismo sistematico dos veces.
        corrections, cal = _calibrate(
            {"status": "green", "flux_factor": 0.973, "scale_err_frac": 0.05, "band": "RP"}
        )
        self.assertEqual(corrections.flux_declared_err_frac, 0.0)
        self.assertAlmostEqual(corrections.flux_scale_err_frac, 0.05)
        np.testing.assert_allclose(cal.product.extra_columns["sys_fluxcal"], 100.0 * 0.05)
        np.testing.assert_allclose(cal.product.extra_columns["sys_fluxcal_declared"], 0.0)
        self.assertFalse(any(r["term"] == "fluxcal_declared" for r in cal.error_budget))

    def test_no_m3_factor_leaves_the_declared_term_at_zero(self):
        corrections, cal = _calibrate({"status": "green"})
        self.assertEqual(corrections.flux_declared_err_frac, 0.0)
        np.testing.assert_allclose(cal.product.extra_columns["sys_fluxcal_declared"], 0.0)


if __name__ == "__main__":
    unittest.main()
