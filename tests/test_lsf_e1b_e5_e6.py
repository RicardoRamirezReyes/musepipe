"""E1b, E5 y E6 resuelven la LSF con el resolutor compartido, sin default.

Hasta el 2026-10-21 tenian su propia cascada (`lsf.fwhm_A`, `cube.lsf_fwhm_A`,
`lsf_fwhm_A`) con un default de 2.6 A, y en los dos runs del paper ninguna de esas
claves existia: los mapas ciegos, las curvas de contraste y las ROC corrian con
2.6 A mientras E1 usaba 2.285 / 2.297 A. Es el mismo fallo que se arreglo en C5/C6
el 2026-09-10; lo que se vigila aqui es que no vuelva.
"""
import inspect
import json
import tempfile
import unittest
from pathlib import Path

from musepipe import spectral
from musepipe.stages import stage_h01b_fovmap as h01b
from musepipe.stages import stage_h05_contrast as h05
from musepipe.stages import stage_h06_roc as h06


def _paths(tmp, qc00=None):
    path = Path(tmp) / "stage00q_qc.json"
    if qc00 is not None:
        path.write_text(json.dumps(qc00))
    return {"stage00q_qc_json": path}


class LsfCompartidaTests(unittest.TestCase):
    def test_las_tres_etapas_usan_la_misma_funcion_y_el_resolutor_compartido(self):
        self.assertIs(h05._lsf_fwhm, h01b._lsf_fwhm)
        self.assertIs(h06._lsf_fwhm, h01b._lsf_fwhm)
        self.assertIs(h01b.resolve_lsf_fwhm_A, spectral.resolve_lsf_fwhm_A)

    def test_no_queda_un_default(self):
        self.assertNotIn("default", inspect.signature(h01b._lsf_fwhm).parameters)
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(RuntimeError):
                h01b._lsf_fwhm(_paths(tmp), {}, stage_key="h01b_lsf_fwhm_A")

    def test_el_knob_de_e1_y_la_medida_de_a4_con_su_procedencia(self):
        with tempfile.TemporaryDirectory() as tmp:
            valor, fuente = h01b._lsf_fwhm(_paths(tmp), {"h01_lsf_fwhm_A": 2.297}, stage_key="h05_lsf_fwhm_A")
            self.assertEqual((valor, fuente), (2.297, "config.h01_lsf_fwhm_A"))
            qc00 = {"m2_lsf": {"lsf_fwhm_at_halpha_A": 2.285}}
            valor, fuente = h01b._lsf_fwhm(_paths(tmp, qc00), {}, stage_key="h06_lsf_fwhm_A")
            self.assertEqual((valor, fuente), (2.285, "stage00q_qc.m2_lsf.lsf_fwhm_at_halpha_A"))

    def test_el_knob_de_la_etapa_gana(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = {"h01_lsf_fwhm_A": 2.297, "h01b_lsf_fwhm_A": 2.6}
            self.assertEqual(h01b._lsf_fwhm(_paths(tmp), cfg, stage_key="h01b_lsf_fwhm_A"),
                             (2.6, "config.h01b_lsf_fwhm_A"))

    def test_las_claves_viejas_ya_no_cuentan(self):
        # Eran las que miraba la cascada antigua: A4 no las escribe.
        with tempfile.TemporaryDirectory() as tmp:
            qc00 = {"lsf": {"fwhm_A": 9.9}, "cube": {"lsf_fwhm_A": 9.9}}
            with self.assertRaises(RuntimeError):
                h01b._lsf_fwhm(_paths(tmp, qc00), {}, stage_key="h01b_lsf_fwhm_A")


if __name__ == "__main__":
    unittest.main()
