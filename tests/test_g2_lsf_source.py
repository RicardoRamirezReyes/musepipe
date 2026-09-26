"""G2 y la LSF: un solo resolutor, y el aviso cuando el declarado no es el medido.

G2 tenia su propio buscador -el tercero del proyecto- que exigia
`m2_lsf["status"] == "ok"` y la clave `m2_lsf["fwhm_A"]`. A4 escribe
`lsf_fwhm_at_halpha_A` con `status` "yellow", asi que esa rama no podia
dispararse en ningun run y G2 siempre caia al config diciendo «A4 M2
unavailable». Estos tests fijan las dos mitades del arreglo: que la medida se ve,
y que un declarado de otra cosecha deja de entrar en silencio.
"""
import unittest

from musepipe.stages.stage_g2_measure_lines import _resolve_lsf


def _qc_a4(fwhm=2.2849, status="yellow"):
    """Un `stage00q_qc.json` con la forma REAL que escribe A4/M2."""
    return {"m2_lsf": {"status": status, "lsf_fwhm_at_halpha_A": fwhm,
                       "n_measurements": 185, "n_exposures": 29}}


class G2LsfSourceTests(unittest.TestCase):
    def test_la_medida_de_a4_se_ve_aunque_el_status_sea_yellow(self):
        valor, procedencia, issue = _resolve_lsf({}, _qc_a4())
        self.assertAlmostEqual(valor, 2.2849)
        self.assertEqual(procedencia, "stage00q_qc.m2_lsf.lsf_fwhm_at_halpha_A")
        self.assertIsNone(issue)

    def test_el_declarado_gana_sobre_el_medido(self):
        # La convencion del proyecto, la misma de `resolve_flux_unit`: un run
        # puede congelar el numero que ya publico.
        valor, procedencia, _ = _resolve_lsf({"h01_lsf_fwhm_A": 2.383}, _qc_a4())
        self.assertAlmostEqual(valor, 2.383)
        self.assertEqual(procedencia, "config.h01_lsf_fwhm_A")

    def test_el_knob_de_g2_gana_al_general(self):
        valor, procedencia, _ = _resolve_lsf(
            {"g2_lsf_fwhm_A": 2.30, "h01_lsf_fwhm_A": 2.383}, _qc_a4())
        self.assertAlmostEqual(valor, 2.30)
        self.assertEqual(procedencia, "config.g2_lsf_fwhm_A")

    def test_declarado_lejos_del_medido_levanta_issue(self):
        # El caso real del canonico de ROXs 12 b: 2.383 (julio) contra 2.2849.
        _, _, issue = _resolve_lsf({"h01_lsf_fwhm_A": 2.383}, _qc_a4())
        self.assertIsNotNone(issue)
        self.assertEqual(issue["priority"], "major")
        self.assertIn("2.383", issue["issue"])
        self.assertIn("2.2849", issue["issue"])

    def test_declarado_igual_al_medido_no_levanta_nada(self):
        _, _, issue = _resolve_lsf({"h01_lsf_fwhm_A": 2.2849}, _qc_a4(2.2849))
        self.assertIsNone(issue)

    def test_sin_a4_lo_dice_pero_no_para(self):
        valor, procedencia, issue = _resolve_lsf({"h01_lsf_fwhm_A": 2.383}, {})
        self.assertAlmostEqual(valor, 2.383)
        self.assertEqual(procedencia, "config.h01_lsf_fwhm_A")
        self.assertIn("not measured", issue["issue"])

    def test_sin_a4_y_sin_config_es_error_no_default(self):
        with self.assertRaises(RuntimeError):
            _resolve_lsf({}, {})


if __name__ == "__main__":
    unittest.main()
