"""D2 sin el QC de A2 o de A3: el termino vale cero, pero lo dice.

`_sky_frac_from_qc` y `_telluric_fracs_from_qc` devolvian cero sin un solo
aviso cuando su QC no existia, al contrario que `_psf_frac_from_qc`. Asi el D2
de `ROXs12b_invvar` (el cubo de publicacion) del 2026-09-22 salio sin
`sys_telluric` y con `open_issues` limpio: A3 no se habia corrido sobre ese
cubo, y nada lo senalaba.
"""
import unittest

from musepipe.stages.stage_x11_calibrate import (
    _sky_frac_from_qc,
    _telluric_fracs_from_qc,
    calibration_corrections_from_qc,
)

A3_REAL = {"telluric_systematic_frac_by_band": {"O2_A": 0.0148, "H2O_8200": 0.0265}}


class QcAusenteA2A3Tests(unittest.TestCase):
    def test_sin_qc_de_a3_el_termino_queda_vacio_y_lo_dice(self):
        fracs, issues = _telluric_fracs_from_qc(None)
        self.assertEqual(fracs, {})
        self.assertTrue(any("A3" in i and "unavailable" in i for i in issues))

    def test_sin_qc_de_a2_el_termino_vale_cero_y_lo_dice(self):
        frac, issues = _sky_frac_from_qc(None)
        self.assertEqual(frac, 0.0)
        self.assertTrue(any("A2" in i and "unavailable" in i for i in issues))

    def test_llega_a_los_open_issues_de_d2(self):
        corr = calibration_corrections_from_qc({}, qc_sky=None, qc_telluric=None, config={})
        self.assertEqual(corr.telluric_frac_by_band, {})
        self.assertEqual(corr.sky_frac, 0.0)
        self.assertTrue(any("A3 telluric QC unavailable" in i for i in corr.open_issues))
        self.assertTrue(any("A2 sky QC unavailable" in i for i in corr.open_issues))

    def test_con_qc_de_a3_no_hay_aviso(self):
        fracs, issues = _telluric_fracs_from_qc(A3_REAL)
        self.assertEqual(fracs, {"O2_A": 0.0148, "H2O_8200": 0.0265})
        self.assertEqual(issues, [])


if __name__ == "__main__":
    unittest.main()
