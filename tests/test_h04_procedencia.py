"""E4 tiene que decir CON QUE VARA midio, no solo el resultado.

Dos huecos, los dos del 2026-09-18:

1. `input_snr` no es una S/N por spaxel ni del array de varianza: es el flujo
   total de linea en unidades de `sigma_flux`, calibrada en la POSICION DEL
   COMPANERO. Esa calibracion se calculaba, se guardaba en `cfg` y no llegaba al
   QC, asi que el numero solo se podia interpretar leyendo el codigo -y el
   paper lo cita-.
2. La LSF que E4 inyecta tampoco se anotaba, y su buscador privado era el cuarto
   del proyecto, con el mismo agujero que los otros tres: su lista de claves de
   A4/M2 no incluia `lsf_fwhm_at_halpha_A`, que es la que A4 escribe.
"""
import unittest

from musepipe.stages import stage_h04_injection as h04


class EscalaDeInputSnrTests(unittest.TestCase):
    def _cfg(self, **extra):
        cfg = {"run_id": "R", "h01_lsf_fwhm_A": 2.3, "h04_line_center_A": 6562.8}
        cfg.update(extra)
        return cfg

    def test_el_qc_declara_la_vara_y_la_calibracion(self):
        calib = {"reference_method": "psffit", "throughput_ref": 0.9,
                 "f_cal": 1000.0, "recovered_cal": 900.0, "sigma_cal": 10.0}
        d = h04._injection_scale_qc(self._cfg(h04_injection_flux_sigma=878.7,
                                              h04_sigma_calibration=calib))
        self.assertEqual(d["sigma_flux"], 878.7)
        self.assertEqual(d["sigma_flux_calibration"], calib)
        self.assertEqual(d["sigma_flux_source"], "calibration_injection_at_companion_position")
        self.assertEqual(d["input_snr_unit"], "total_line_flux / sigma_flux")
        self.assertEqual(d["completeness_measured_at"], "control_positions")
        self.assertEqual(d["lsf_fwhm_A"], 2.3)
        self.assertIn("COMPANERO", d["note"])

    def test_distingue_la_sigma_declarada_de_la_calibrada(self):
        d = h04._injection_scale_qc(self._cfg(h04_injection_flux_sigma=100.0))
        self.assertEqual(d["sigma_flux_source"], "declared:h04_injection_flux_sigma")
        self.assertIsNone(d["sigma_flux_calibration"])

    def test_declara_el_hueco_en_vez_de_reventar_el_qc(self):
        # Sin LSF por ningun lado: el QC se escribe igual y lo dice. Reventar
        # aqui tiraria una corrida de 2 h 30 ya terminada.
        d = h04._injection_scale_qc({"run_id": "R"})
        self.assertIsNone(d["lsf_fwhm_A"])
        self.assertIn("no resoluble", d["lsf_fwhm_A_source"])


class LsfPorElResolutorUnicoTests(unittest.TestCase):
    def test_encuentra_la_clave_que_A4_escribe_de_verdad(self):
        # El buscador privado miraba `fwhm_at_halpha_A`, `halpha_fwhm_A` y
        # `lsf_fwhm_A`, nunca `lsf_fwhm_at_halpha_A`: con A4/M2 medida en disco y
        # sin knob en el config, reventaba.
        class _Paths(dict):
            pass
        import json
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as d:
            qc = Path(d) / "stage00q_qc.json"
            qc.write_text(json.dumps({"m2_lsf": {"lsf_fwhm_at_halpha_A": 2.31, "status": "yellow"}}))
            valor, fuente = h04._lsf_fwhm_from_config_or_qc({"run_id": "R"},
                                                            _Paths(stage00q_qc_json=qc))
        self.assertAlmostEqual(valor, 2.31)
        self.assertEqual(fuente, "stage00q_qc.m2_lsf.lsf_fwhm_at_halpha_A")

    def test_el_knob_de_la_etapa_sigue_ganando(self):
        valor, fuente = h04._lsf_fwhm_from_config_or_qc(
            {"h04_lsf_fwhm_A": 2.1, "h01_lsf_fwhm_A": 2.9})
        self.assertEqual((valor, fuente), (2.1, "config.h04_lsf_fwhm_A"))

    def test_sin_knob_usa_el_del_run(self):
        valor, fuente = h04._lsf_fwhm_from_config_or_qc({"h01_lsf_fwhm_A": 2.9})
        self.assertEqual((valor, fuente), (2.9, "config.h01_lsf_fwhm_A"))

    def test_sin_nada_revienta_y_no_inventa_default(self):
        with self.assertRaises(RuntimeError):
            h04._lsf_fwhm_from_config_or_qc({"run_id": "R"})


if __name__ == "__main__":
    unittest.main()
