"""El cociente de ruido «7 and 3» del Apéndice A (`scripts/e4_ruido_en_el_companero.py`).

`PiezasTests` no toca `runs/`. `CifrasDelPaperTests` lee el QC y la tabla de E4 de los dos runs del paper
y de sus copias de antes de c0 (solo lectura, < 1 s): 7.18 y 3.36 con C4 v2, 8.51 y 3.10 antes. Busca
`runs/` en este clon o en `MUSE_PROJECT_ROOT`; si falta un run o una copia, su caso se salta.
"""
import importlib.util
import os
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _carga(nombre):
    spec = importlib.util.spec_from_file_location(nombre, ROOT / "scripts" / f"{nombre}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[nombre] = mod
    spec.loader.exec_module(mod)
    return mod


def _raiz_con_runs(*runs):
    for raiz in (ROOT, os.environ.get("MUSE_PROJECT_ROOT")):
        if raiz and all((Path(raiz) / "runs" / r / "config" / "config.json").exists() for r in runs):
            return Path(raiz)
    return None


class PiezasTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.m = _carga("e4_ruido_en_el_companero")

    def test_el_cociente_es_sigma_en_el_companero_sobre_sigma_de_las_nulas(self):
        qc = {"injection_scale": {"sigma_flux": 846.8},
              "snr_standardization": {"scale_by_stratum": {"psffit|1|none": {"sigma": 118.0, "n": 33},
                                                           "aperture|1|none": {"sigma": 1.0, "n": 33}}}}
        s_comp, s_nulas, c = self.m.cociente_de_ruido(qc)
        self.assertEqual((s_comp, s_nulas), (846.8, 118.0))
        self.assertAlmostEqual(c, 846.8 / 118.0, places=12)
        self.assertEqual(self.m.cociente_de_ruido(qc, "aperture|1|none")[2], 846.8)

    def test_posiciones_y_completitud_cuentan_solo_las_nominales_fuera_de_la_real(self):
        filas = []
        for pos in ("real", "c01", "c02", "c03"):
            for snr, completa in (("0.0", "False"), ("5.0", "True"), ("3.0", "False")):
                filas.append({"variant": "nominal", "position_label": pos, "method": "psffit",
                              "input_snr": snr, "complete": completa if pos != "real" else "False"})
        filas.append({"variant": "psf_wide", "position_label": "c09", "method": "psffit",
                      "input_snr": "5.0", "complete": "False"})
        filas.append({"variant": "nominal", "position_label": "c04", "method": "aperture",
                      "input_snr": "5", "complete": "False"})
        n, comp = self.m.posiciones_y_completitud(filas)
        self.assertEqual(n, 4)  # c01-c04: ni la real ni la de otra variante
        self.assertEqual(comp, {"aperture": 0.0, "psffit": 1.0})


RAIZ = _raiz_con_runs("ROXs42Bb_realigned", "ROXs12b_invvar")
COPIAS = {"ROXs12b_invvar": "_antes_c4_c0_20261024T200833", "ROXs42Bb_realigned": "_antes_c4_c0_20261024T225459"}


@unittest.skipUnless(RAIZ is not None, "sin los runs del paper (ni aquí ni en MUSE_PROJECT_ROOT)")
class CifrasDelPaperTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.m = _carga("e4_ruido_en_el_companero")

    def test_con_c4_v2_son_7_y_3(self):
        a = self.m.lee("ROXs12b_invvar", RAIZ)
        b = self.m.lee("ROXs42Bb_realigned", RAIZ)
        self.assertEqual((round(a["cociente"], 2), round(b["cociente"], 2)), (7.18, 3.36))
        self.assertEqual((round(a["cociente"]), round(b["cociente"])), (7, 3))  # lo que imprime el paper
        self.assertEqual((a["posiciones_de_inyeccion"], b["posiciones_de_inyeccion"]), (11, 35))
        for r in (a, b):
            self.assertEqual(set(r["completitud_a_5"].values()), {1.0})
            self.assertEqual(len(r["completitud_a_5"]), 6)

    def test_las_copias_de_antes_de_c0(self):
        esperado = {"ROXs12b_invvar": 8.51, "ROXs42Bb_realigned": 3.10}
        for run, copia in COPIAS.items():
            d = RAIZ / "runs" / run / "stages" / copia
            with self.subTest(run=run):
                if not (d / "stages" / "stage_h04_qc.json").exists():
                    self.skipTest(f"sin la copia {d}")
                self.assertEqual(round(self.m.lee(run, RAIZ, d)["cociente"], 2), esperado[run])


if __name__ == "__main__":
    unittest.main()
