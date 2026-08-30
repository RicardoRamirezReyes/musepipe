"""El barrido de radios: lo que decide, y lo que NO puede decidir.

El riesgo de esta sonda no es la aritmetica, es publicar un `f50` que la rejilla
no ha medido. Si la completitud ya satura en el nivel mas bajo, el cruce del 50 %
esta por debajo de lo medido: devolver ese nivel lo haria pasar por una medida y
empataria configuraciones que nunca se compararon. Eso —y que la particion A/B
sea de verdad disjunta y estable— es lo unico que separa este criterio del que
se sobreajusto el 2026-08-29.
"""

import math
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from psffit_radius_sweep import _f50, _particion, _valor_snr  # noqa: E402


class TestF50(unittest.TestCase):
    def test_interpola_el_cruce_entre_los_dos_niveles_que_lo_bracketean(self):
        # 0.3 -> 30 %, 0.5 -> 70 %: el 50 % cae justo en medio.
        self.assertAlmostEqual(_f50([0.2, 0.3, 0.5, 1.0], [0.1, 0.3, 0.7, 1.0]), 0.4)

    def test_satura_en_el_nivel_mas_bajo_es_sin_resolver_no_ese_nivel(self):
        # El fallo que este test existe para impedir: devolver 0.2 haria que dos
        # configuraciones que saturan a niveles distintos empataran en 0.2.
        self.assertTrue(math.isnan(_f50([0.2, 0.5, 1.0], [1.0, 1.0, 1.0])))

    def test_no_llega_al_cincuenta_por_ciento_tampoco_es_un_numero(self):
        self.assertTrue(math.isnan(_f50([0.2, 0.5, 1.0], [0.0, 0.1, 0.2])))

    def test_el_nivel_cero_no_entra(self):
        # La nula no es un punto de la curva de completitud: es de donde sale el
        # umbral. Si entrara, una nula "detectada" desplazaria el cruce.
        con_cero = _f50([0.0, 0.3, 0.5], [1.0, 0.3, 0.7])
        sin_cero = _f50([0.3, 0.5], [0.3, 0.7])
        self.assertAlmostEqual(con_cero, sin_cero)

    def test_ignora_niveles_no_medidos(self):
        self.assertAlmostEqual(
            _f50([0.2, 0.3, 0.5], [float("nan"), 0.3, 0.7]), 0.4
        )

    def test_es_monotono_en_la_dificultad(self):
        # Un estimador que detecta menos a cada nivel no puede tener f50 menor.
        bueno = _f50([0.2, 0.3, 0.5, 1.0], [0.2, 0.6, 0.9, 1.0])
        malo = _f50([0.2, 0.3, 0.5, 1.0], [0.0, 0.1, 0.4, 0.8])
        self.assertLess(bueno, malo)


class TestParticion(unittest.TestCase):
    def test_las_mitades_son_disjuntas_y_cubren_todo(self):
        etiquetas = [f"control_{i:02d}" for i in range(9)]
        mitades = _particion(etiquetas)
        self.assertEqual(mitades["A"] & mitades["B"], set())
        self.assertEqual(mitades["A"] | mitades["B"], set(etiquetas))

    def test_no_depende_del_orden_de_llegada(self):
        # Las filas de E4 vuelven en el orden en que terminan los workers. Si la
        # particion dependiera de eso, dos ejecuciones del barrido no serian
        # comparables entre si.
        etiquetas = [f"control_{i:02d}" for i in range(9)]
        self.assertEqual(_particion(etiquetas), _particion(list(reversed(etiquetas))))

    def test_tolera_repeticiones(self):
        # Cada posicion aparece en muchas filas (una por nivel de SNR).
        etiquetas = [f"control_{i:02d}" for i in range(4)] * 7
        mitades = _particion(etiquetas)
        self.assertEqual(len(mitades["A"]) + len(mitades["B"]), 4)


class TestValorSnr(unittest.TestCase):
    def test_usa_la_estandarizada(self):
        self.assertEqual(_valor_snr({"recovered_snr_std": 2.5, "recovered_snr": 9.9}), 2.5)

    def test_sin_estandarizar_es_nan_y_NO_la_snr_formal(self):
        # Son dos escalas: una lleva restada la mediana de las nulas de su
        # estrato y la otra no. Sustituir una por otra mezclaria poblaciones
        # dentro de la misma curva de completitud sin que nada lo dijera.
        for fila in ({"recovered_snr_std": None, "recovered_snr": 9.9},
                     {"recovered_snr_std": float("nan"), "recovered_snr": 9.9}):
            with self.subTest(fila=fila):
                self.assertTrue(math.isnan(_valor_snr(fila)))


if __name__ == "__main__":
    unittest.main()
