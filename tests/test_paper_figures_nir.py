"""La figura de los espectros J/H/K publicados (`nir_spectra` en `scripts/paper_figures.py`).

La figura normaliza cada espectro a la ventana del denominador del índice de agua de su banda, para que
la media en la ventana del numerador sea el índice que entra en el tipo IR (`scripts/tipo_ir_indices.py`).
Las dos primeras clases no leen nada de disco. La última dibuja la figura si el catálogo CDS y los runs
del paper están en disco, y se salta si no (como `tests/test_tipo_ir_indices.py`).
"""
import importlib.util
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _carga(nombre):
    spec = importlib.util.spec_from_file_location(nombre, ROOT / "scripts" / f"{nombre}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[nombre] = mod
    spec.loader.exec_module(mod)
    return mod


def _rejilla(lo, hi, R=500):
    """Rejilla con λ/Δλ = R por píxel, como los espectros del catálogo."""
    n = int(math.log(hi / lo) * R) + 1
    return lo * np.exp(np.arange(n) / R)


class NormalizacionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pf = _carga("paper_figures")
        cls.tir = _carga("tipo_ir_indices")

    def test_la_ventana_del_numerador_da_el_indice(self):
        for banda, (lo, hi) in self.pf.BANDAS_NIR.items():
            indice = self.tir.BAND_INDEX[banda]
            w = _rejilla(lo, hi)
            f = 3.0e-16 * (w / w[0]) ** -2.5 * (1 + 0.1 * np.sin(40 * w))
            esperado = self.tir.ratio_index(w, f, np.full_like(f, 1e-18), indice)["index"]
            escala = self.pf.normaliza_nir(w, f, indice)
            d = self.tir.INDICES[indice]
            self.assertAlmostEqual(self.tir.window_mean(w, f * escala, *d["den"]), 1.0, places=12, msg=banda)
            self.assertAlmostEqual(self.tir.window_mean(w, f * escala, *d["num"]), esperado, places=12, msg=banda)

    def test_la_escala_no_depende_de_la_unidad(self):
        w = _rejilla(*self.pf.BANDAS_NIR["K"])
        f = (w / w[0]) ** -1.5
        self.assertAlmostEqual(self.pf.normaliza_nir(w, f, "H2O-2") / self.pf.normaliza_nir(w, 1e3 * f, "H2O-2"),
                               1e3, places=9)


class RegistroTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pf = _carga("paper_figures")
        cls.tir = _carga("tipo_ir_indices")

    def test_un_panel_por_banda_con_indice(self):
        self.assertEqual(list(self.pf.BANDAS_NIR), list(self.tir.BAND_INDEX))

    def test_cada_panel_contiene_las_dos_ventanas_de_su_indice(self):
        for banda, (lo, hi) in self.pf.BANDAS_NIR.items():
            d = self.tir.INDICES[self.tir.BAND_INDEX[banda]]
            for ventana in (d["num"], d["den"]):
                self.assertTrue(lo < ventana[0] < ventana[1] < hi, (banda, ventana))

    def test_figura_registrada(self):
        self.assertIs(self.pf.FIGURAS["nir_spectra"], self.pf.nir_spectra)


def _hay_datos():
    tir = _carga("tipo_ir_indices")
    if not tir.DEFAULT_CDS.is_dir():
        return False
    for ficha in (ROOT / "targets").glob("*.json"):
        paper = json.loads(ficha.read_text()).get("paper")
        if paper and not (ROOT / "runs" / paper["run"] / "config" / "config.json").exists():
            return False
    return True


@unittest.skipUnless(_hay_datos(), "sin catálogo CDS o sin los runs del paper")
class FiguraTests(unittest.TestCase):
    def test_dibuja_el_pdf(self):
        pf = _carga("paper_figures")
        pf.estilo_aa()
        with tempfile.TemporaryDirectory() as tmp:
            destino = Path(tmp) / "nir_spectra.pdf"
            pf.nir_spectra(pf.objetos_del_paper(), destino)
            self.assertGreater(destino.stat().st_size, 10_000)


if __name__ == "__main__":
    unittest.main()
