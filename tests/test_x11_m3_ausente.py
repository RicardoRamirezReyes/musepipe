"""D2 sin el M3 de A4: la salida de emergencia tiene que poder tomarse.

`_flux_scale_from_m3` tiene dos returns y desde que se añadio el fluxcal
declarado no devolvian lo mismo: siete valores la salida normal, cinco la de
`m3` ausente. El llamador desempaqueta siete, asi que la rama escrita para
seguir con `scale=1` y dejar el aviso en `open_issues` hacia justo lo contrario
-un `ValueError: not enough values to unpack` que no nombra ni A4, ni M3, ni el
flujo-. No salto nunca porque los runs en uso tienen `m3_flux`; `ROXs12b_raw`,
que no lo tiene, no llega a D2.
"""
import unittest

from musepipe.stages.stage_x11_calibrate import (
    _flux_scale_from_m3,
    calibration_corrections_from_qc,
)

M3_REAL = {"flux_factor": 0.957, "band": "RP", "status": "yellow"}


class M3AusenteTests(unittest.TestCase):
    def test_las_dos_salidas_tienen_la_misma_aridad(self):
        """El cable trampa: ampliar una salida y no la otra es lo que rompio esto."""
        self.assertEqual(len(_flux_scale_from_m3({})), len(_flux_scale_from_m3(M3_REAL)))

    def test_sin_m3_cae_a_escala_1_y_lo_dice(self):
        scale, err, _var, source, issues, declarado, _ds = _flux_scale_from_m3({})
        self.assertEqual(scale, 1.0)
        self.assertEqual(err, 0.0)
        self.assertEqual(declarado, 0.0)
        self.assertIn("unavailable", source)
        self.assertTrue(any("scale=1" in i for i in issues))

    def test_el_qc_de_a4_sin_m3_no_revienta_a_d2(self):
        """El caso real: A4 sin `m3-flux` corrido, o un QC de otra cosecha."""
        corr = calibration_corrections_from_qc({"m1_wavelength": {}}, config={})
        self.assertEqual(corr.flux_scale, 1.0)
        self.assertEqual(corr.flux_declared_err_frac, 0.0)
        self.assertTrue(any("A4/M3" in i for i in corr.open_issues))

    def test_sin_qc_de_a4_tampoco(self):
        corr = calibration_corrections_from_qc({}, config={})
        self.assertEqual(corr.flux_scale, 1.0)
        self.assertTrue(any("A4/M3" in i for i in corr.open_issues))

    def test_con_m3_de_verdad_no_se_toca_nada(self):
        """El desvio declarado sigue saliendo de |1 - flux_factor|."""
        scale, _err, _var, source, _issues, declarado, ds = _flux_scale_from_m3(M3_REAL)
        self.assertEqual(scale, 1.0)
        self.assertAlmostEqual(declarado, abs(1.0 - 0.957))
        self.assertIn("Gaia", ds)
        self.assertIn("validated vs Gaia DR3", source)


if __name__ == "__main__":
    unittest.main()
