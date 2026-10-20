"""Test cromosférico de Manara et al. 2013 (`scripts/suelo_cromosferico.py`).

Da las cifras del paper sobre el nivel cromosférico: ROXs 12 b en log(L_acc/L_bol) = −3.62, en el suelo a
2500 K y 0.6 dex por debajo a 3100 K; la cota de ROXs 42B b 0.6 dex por debajo, extrapolada
(docs/2026-10-13_correcciones_literatura.md). Desde el 2026-10-14 el paper ya no compara con las T_eff publicadas
sino con la de su propio ajuste, muestra a muestra (`scripts/teff_bayes.py`, tests/test_teff_bayes.py); de aquí
quedan el −3.62, el −4.40 (−4.45 hasta el 2026-10-20, con la cola de E3 por momentos) y, para 42B b con la L_bol
publicada, el 0.5–0.7 dex a 2200–2350 K (−0.57 a 2270 K).
Las primeras clases no leen runs: la relación, el margen, la
marca de extrapolación y la clave `fisica` de `targets/`. `CifrasDelPaperTests` lee los QC de los runs del
paper; si faltan se salta, no falla.
"""
import importlib.util
import json
import math
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _carga(nombre):
    spec = importlib.util.spec_from_file_location(nombre, ROOT / "scripts" / f"{nombre}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[nombre] = mod
    spec.loader.exec_module(mod)
    return mod


class RelacionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.m = _carga("suelo_cromosferico")

    def test_coeficientes_de_la_ecuacion_2(self):
        self.assertEqual((self.m.MANARA13["A"], self.m.MANARA13["B"]), (6.17, -24.54))
        self.assertEqual(self.m.MANARA13["teff_aplicable_k"], (2500.0, 4000.0))

    def test_suelo_es_la_recta_en_log_teff(self):
        for t in (2270.0, 2500.0, 3100.0, 4000.0):
            self.assertAlmostEqual(self.m.suelo(t), 6.17 * math.log10(t) - 24.54, places=12)
        self.assertAlmostEqual(self.m.suelo(2500.0), -3.575, places=3)
        self.assertAlmostEqual(self.m.suelo(3100.0), -2.998, places=3)

    def test_el_suelo_sube_con_la_temperatura(self):
        self.assertLess(self.m.suelo(2500.0), self.m.suelo(3000.0))

    def test_margen_positivo_por_encima_y_negativo_por_debajo(self):
        filas = self.m.evalua(-3.0, {"a": 2500.0, "b": 3500.0})
        self.assertGreater(filas[0]["margen_dex"], 0)
        self.assertLess(filas[1]["margen_dex"], 0)
        self.assertAlmostEqual(filas[0]["margen_dex"], round(-3.0 - self.m.suelo(2500.0), 3))

    def test_extrapolada_fuera_de_2500_4000(self):
        filas = self.m.evalua(-4.0, {"bajo": 2499.0, "borde_bajo": 2500.0, "borde_alto": 4000.0, "alto": 4001.0})
        self.assertEqual([f["extrapolada"] for f in filas], [True, False, False, True])

    def test_teffs_juntan_principal_alternativas_y_extra(self):
        fis = dict(teff_k=2500, teff_fuente="principal", teff_alternativas_k={"otra": 3100})
        self.assertEqual(self.m.teffs_de(fis, [2800.0]), {"principal": 2500.0, "otra": 3100.0, "--teff 2800": 2800.0})
        self.assertEqual(self.m.teffs_de(dict(teff_k=2270, teff_fuente="x")), {"x": 2270.0})


class PerfilesTests(unittest.TestCase):
    def test_los_dos_perfiles_declaran_fisica_con_fuente(self):
        for slug in ("ROXs12b", "ROXs42Bb"):
            fis = json.loads((ROOT / "targets" / f"{slug}.json").read_text())["fisica"]
            for clave in ("log_lbol", "teff_k"):
                self.assertIsInstance(fis[clave], (int, float), (slug, clave))
            for clave in ("log_lbol_fuente", "teff_fuente"):
                self.assertTrue(str(fis[clave]).strip(), (slug, clave))

    def test_main_con_lacc_dada_no_lee_runs(self):
        m = _carga("suelo_cromosferico")
        with tempfile.TemporaryDirectory() as d:
            salida = Path(d) / "s.json"
            with redirect_stdout(StringIO()):
                self.assertEqual(m.main(["ROXs12b", "--lacc", "3.734e-7", "--teff", "2800", "--json", str(salida)]), 0)
            out = json.loads(salida.read_text())
        self.assertEqual(out["lacc_tipo"], "dada")
        self.assertAlmostEqual(out["log_lacc_lbol"], round(math.log10(3.734e-7) + 2.81, 3))
        self.assertEqual([f["teff_k"] for f in out["filas"]], [2500.0, 3100.0, 2800.0])


class CifrasDelPaperTests(unittest.TestCase):
    """Las cifras del paper, desde los QC de los runs del paper (G3 para 12 b, E3 para 42B b)."""

    @classmethod
    def setUpClass(cls):
        cls.m = _carga("suelo_cromosferico")

    def _corre(self, slug):
        perfil = json.loads((ROOT / "targets" / f"{slug}.json").read_text())
        rd = ROOT / "runs" / perfil["paper"]["run"] / "stages"
        for f in ("stage_g3_qc.json", "stage_h03_qc.json", "stage_x11_qc.json"):
            if not (rd / f).exists():
                self.skipTest(f"falta {rd / f}")
        with tempfile.TemporaryDirectory() as d:
            salida = Path(d) / "s.json"
            with redirect_stdout(StringIO()):
                self.m.main([slug, "--json", str(salida)])
            return json.loads(salida.read_text())

    def test_12b_en_el_suelo_a_2500_y_por_debajo_a_3100(self):
        out = self._corre("ROXs12b")
        self.assertEqual(out["lacc_tipo"], "detection")
        self.assertAlmostEqual(out["log_lacc_lbol"], -3.62, delta=0.005)
        margen = {f["teff_k"]: f["margen_dex"] for f in out["filas"]}
        self.assertAlmostEqual(margen[2500.0], -0.04, delta=0.01)
        self.assertAlmostEqual(margen[3100.0], -0.62, delta=0.01)

    def test_42bb_cota_06_dex_por_debajo_y_extrapolada(self):
        out = self._corre("ROXs42Bb")
        self.assertEqual(out["lacc_tipo"], "upper_limit")
        # Decision B (2026-10-14): con la distancia de Gaia DR3 (146.45 pc en vez de 143.6) la L_acc
        # limite sube x1.045 (d^2.26) y la cota pasa de -4.46 a -4.45; sigue 0.6 dex bajo el suelo.
        # 2026-10-20: E3 ajusta la cola como E1 (MLE, docs/2026-10-20_e3_misma_cola_que_e1.md); el
        # umbral de psffit sube de z 2.733 a 3.003, la L_acc limite x1.11 (+0.046 dex) y la cota pasa
        # a -4.40, 0.57 dex bajo el suelo a 2270 K.
        self.assertAlmostEqual(out["log_lacc_lbol"], -4.40, delta=0.01)
        (fila,) = out["filas"]
        self.assertAlmostEqual(fila["margen_dex"], -0.57, delta=0.01)
        self.assertTrue(fila["extrapolada"])


if __name__ == "__main__":
    unittest.main()
