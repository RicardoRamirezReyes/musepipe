"""El umbral calibrado por FPR: lo que hace la funcion y lo que NO puede hacer.

El riesgo de esta herramienta no es equivocarse en la cuenta, es prometer una
tasa que la muestra no puede resolver: con 15 nulos, pedir un 1 % devuelve el
umbral que fija el maximo y una FPR conseguida de 0. Eso no es cumplir el 1 %.
Por eso se prueba que `fpr_conseguida` y `n_por_encima` delatan el caso, que es
lo unico que separa un umbral medido de uno inventado.
"""

import math
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from detection_threshold_by_method import (  # noqa: E402
    resumen_nulo,
    tasa,
    umbral_para_fpr,
)


class TestUmbralParaFpr(unittest.TestCase):
    def test_deja_exactamente_k_nulos_por_encima(self):
        nulos = list(range(100))  # 0..99
        cal = umbral_para_fpr(nulos, 0.05)
        self.assertEqual(cal["n_por_encima"], 5)
        self.assertAlmostEqual(cal["fpr_conseguida"], 0.05)
        self.assertEqual(tasa(nulos, cal["umbral"]) * len(nulos), 5)

    def test_el_umbral_es_el_mas_bajo_que_cumple(self):
        nulos = list(range(100))
        cal = umbral_para_fpr(nulos, 0.05)
        # Bajarlo hasta el siguiente nulo mete un falso positivo mas.
        self.assertEqual(tasa(nulos, cal["umbral"] - 1.0) * len(nulos), 6)

    def test_sin_resolucion_lo_declara_en_vez_de_fingirlo(self):
        nulos = [float(v) for v in range(15)]
        cal = umbral_para_fpr(nulos, 0.01)  # floor(0.15) = 0 eventos
        self.assertEqual(cal["n_por_encima"], 0)
        self.assertEqual(cal["fpr_conseguida"], 0.0)
        self.assertGreater(cal["umbral"], max(nulos))
        self.assertNotAlmostEqual(cal["fpr_conseguida"], 0.01)

    def test_resolucion_es_uno_partido_n(self):
        # La rejilla de FPR alcanzables es k/n: 0.02 y 0.05 caen en el mismo k.
        nulos = list(range(15))
        self.assertEqual(umbral_para_fpr(nulos, 0.02)["umbral"],
                         umbral_para_fpr(nulos, 0.05)["umbral"])

    def test_fpr_de_uno_no_deja_ningun_umbral(self):
        cal = umbral_para_fpr([1.0, 2.0], 1.0)
        self.assertEqual(cal["fpr_conseguida"], 1.0)
        self.assertEqual(cal["umbral"], float("-inf"))

    def test_ignora_no_finitos_y_los_descuenta_del_n(self):
        cal = umbral_para_fpr([1.0, 2.0, float("nan"), float("inf")], 0.5)
        self.assertEqual(cal["n_nulos"], 2)

    def test_muestra_vacia(self):
        cal = umbral_para_fpr([], 0.05)
        self.assertEqual(cal["n_nulos"], 0)
        self.assertTrue(math.isnan(cal["umbral"]))


class TestResumenNulo(unittest.TestCase):
    def test_un_estimador_calibrado_da_media_cero_sigma_uno(self):
        # La referencia contra la que se leen las medias y sigmas medidas.
        nulos = [-1.0, 1.0, -1.0, 1.0]
        r = resumen_nulo(nulos)
        self.assertAlmostEqual(r["media"], 0.0)
        self.assertAlmostEqual(r["sigma"], 1.0)
        self.assertEqual(r["n"], 4)

    def test_pedestal_y_cola_se_distinguen(self):
        pedestal = resumen_nulo([4.0, 6.0, 4.0, 6.0])
        cola = resumen_nulo([-5.0, 5.0, -5.0, 5.0])
        self.assertAlmostEqual(pedestal["media"], 5.0)
        self.assertAlmostEqual(pedestal["sigma"], 1.0)
        self.assertAlmostEqual(cola["media"], 0.0)
        self.assertAlmostEqual(cola["sigma"], 5.0)


class TestTasa(unittest.TestCase):
    def test_es_inclusiva_en_el_umbral(self):
        # E4 detecta con `snr >= umbral`; la tasa tiene que usar el mismo signo.
        self.assertAlmostEqual(tasa([5.0], 5.0), 1.0)

    def test_sin_medidas_finitas(self):
        self.assertTrue(math.isnan(tasa([float("nan")], 5.0)))


if __name__ == "__main__":
    unittest.main()
