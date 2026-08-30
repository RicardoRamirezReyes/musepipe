"""El remuestreo del barrido: que distingue una ventaja real de una del azar.

La sonda existe porque `20x8` gano en ROXs 42B b con UNA particion, y una
particion no dice si ganaria con otra. Lo que se prueba aqui es que la
herramienta separa los dos casos: cuando una configuracion es de verdad mejor
`gana` se va al 100 %, y cuando los dos estimadores son el mismo no fabrica un
ganador.

**Las configuraciones comparten el cubo**, asi que en los datos reales estan
emparejadas en ruido: lo unico que las separa es el estimador. Los sinteticos de
aqui reproducen eso compartiendo la semilla. El caso de ruido independiente NO
es el escenario real, y se prueba aparte precisamente para dejar documentado
cuanto sesgaria si alguien lo confundiera.
"""

import math
import random
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from psffit_radius_sweep_analiza import (  # noqa: E402
    NO_LLEGA,
    RESUELTO,
    SATURA,
    analiza,
    clave_orden,
    f50_de,
)

NIVELES = (0.0, 0.2, 0.3, 0.5, 0.7, 1.0)
REFERENCIA = (20.0, 12.0)


def filas_sinteticas(config, *, ganancia, n_pos=20, ruido=1.0, semilla=0):
    """Filas con la forma que vuelca el barrido, con una sensibilidad conocida.

    `ganancia` es cuanto flujo recupera por unidad de `input_snr`: subirla es un
    estimador que detecta con menos senal, es decir un `f50` menor.
    """
    rng = random.Random(semilla)
    filas = []
    for i in range(n_pos):
        for nivel in NIVELES:
            filas.append({
                "run": "SINTETICO", "r_star_px": config[0], "r_comp_px": config[1],
                "injection_id": f"{config[0]:g}_{config[1]:g}_{i:02d}_{nivel:g}",
                "variant": "nominal", "method": "psffit",
                "position_label": f"control_{i:02d}",
                "template_factor": 1.0, "continuum_mode": "none",
                "input_snr": nivel,
                "recovered_flux": nivel * ganancia + rng.gauss(0.0, ruido),
            })
    return filas


class TestF50De(unittest.TestCase):
    def test_mas_sensible_da_f50_menor(self):
        # Ganancias elegidas para que las DOS crucen el 50 % dentro de la
        # rejilla: con ganancia alta la completitud satura en el nivel mas bajo
        # y `f50` sale `nan` con razon, que es otro comportamiento y se prueba
        # aparte.
        pos = {f"control_{i:02d}" for i in range(20)}
        sensible = f50_de(filas_sinteticas(REFERENCIA, ganancia=5.0, semilla=1), pos, fpr=0.05)
        sordo = f50_de(filas_sinteticas(REFERENCIA, ganancia=2.5, semilla=1), pos, fpr=0.05)
        self.assertEqual(sensible["estado"], RESUELTO)
        self.assertEqual(sordo["estado"], RESUELTO)
        self.assertLess(sensible["f50"], sordo["f50"])

    def test_saturar_es_censura_por_abajo_no_un_dato_ausente(self):
        # El fallo que este test existe para impedir: si saturar fuera `nan` y se
        # descartara, la comparacion emparejada perderia justo los sorteos en los
        # que la configuracion es mejor, y una ventaja saldria como empate.
        pos = {f"control_{i:02d}" for i in range(20)}
        buenisima = f50_de(filas_sinteticas(REFERENCIA, ganancia=20.0, semilla=1), pos, fpr=0.05)
        normal = f50_de(filas_sinteticas(REFERENCIA, ganancia=2.5, semilla=1), pos, fpr=0.05)
        self.assertEqual(buenisima["estado"], SATURA)
        self.assertLess(clave_orden(buenisima), clave_orden(normal))

    def test_no_llegar_es_censura_por_arriba(self):
        pos = {f"control_{i:02d}" for i in range(20)}
        sorda = f50_de(filas_sinteticas(REFERENCIA, ganancia=0.2, semilla=1), pos, fpr=0.05)
        normal = f50_de(filas_sinteticas(REFERENCIA, ganancia=2.5, semilla=1), pos, fpr=0.05)
        self.assertEqual(sorda["estado"], NO_LLEGA)
        self.assertGreater(clave_orden(sorda), clave_orden(normal))

    def test_sin_nulas_suficientes_no_inventa_un_numero(self):
        # Una sola posicion no da referencia por estrato: se declara, no se estima.
        filas = filas_sinteticas(REFERENCIA, ganancia=5.0, n_pos=1, semilla=1)
        self.assertIsNone(f50_de(filas, {"control_00"}, fpr=0.05)["estado"])

    def test_conjunto_vacio(self):
        self.assertIsNone(
            f50_de(filas_sinteticas(REFERENCIA, ganancia=5.0, semilla=1), set(), fpr=0.05)["estado"]
        )


class TestAnaliza(unittest.TestCase):
    def _corre(self, ganancia_rival, sorteos=40, n_pos=30, semilla_rival=7,
               modo_umbral="fpr"):
        """Dos configuraciones sobre las mismas posiciones y el mismo ruido.

        Compartir la semilla es lo que reproduce la situacion real: los dos
        extractores corren sobre el MISMO cubo, asi que lo unico que los separa
        es el estimador. Pasar `semilla_rival` distinta modela ruido
        independiente, que no es el caso real y solo se usa para medir el sesgo
        que introduciria.
        """
        filas = (filas_sinteticas(REFERENCIA, ganancia=3.0, n_pos=n_pos, semilla=7)
                 + filas_sinteticas((20.0, 8.0), ganancia=ganancia_rival, n_pos=n_pos,
                                    semilla=semilla_rival))
        return analiza(filas, fpr=0.05, n_sorteos=sorteos, semilla=1,
                       referencia=REFERENCIA, modo_umbral=modo_umbral)

    def _duelo(self, res):
        return res["duelo"][(20.0, 8.0)]

    def _gana(self, res):
        d = self._duelo(res)
        decididos = d["gana"] + d["pierde"] + d["empata"]
        self.assertGreater(decididos, 0)
        return d["gana"] / decididos

    def test_una_ventaja_real_se_ve_en_casi_todas_las_particiones(self):
        self.assertGreater(self._gana(self._corre(ganancia_rival=6.0)), 0.8)

    def test_una_ventaja_que_satura_sigue_contando_como_ventaja(self):
        # Rival tan sensible que su cruce cae por debajo de la rejilla. Sin
        # tratar la saturacion como censura, estos sorteos se caian de la cuenta
        # y una ventaja enorme se leia como empate.
        res = self._corre(ganancia_rival=20.0)
        self.assertGreater(self._gana(res), 0.8)
        self.assertEqual(self._duelo(res)["pierde"], 0)

    def test_dos_estimadores_identicos_no_producen_ganador(self):
        # Mismo estimador y mismo cubo: no hay nada que elegir, y la sonda no
        # puede inventarse una ventaja. Es el caso nulo de verdad.
        d = self._duelo(self._corre(ganancia_rival=3.0))
        self.assertEqual(d["gana"], 0)
        self.assertEqual(d["pierde"], 0)

    def test_el_veredicto_no_cambia_de_signo_con_el_umbral(self):
        # Un veredicto que solo aparece con una forma de fijar el umbral seria
        # del umbral, no del estimador. Por eso el analisis publica los dos.
        for modo in ("fpr", "robusto"):
            with self.subTest(modo=modo):
                self.assertGreater(
                    self._gana(self._corre(ganancia_rival=6.0, modo_umbral=modo)), 0.8
                )

    def test_con_ruido_independiente_el_veredicto_se_sesga(self):
        # Limite documentado, NO el caso real. Dos estimadores de sensibilidad
        # identica con realizaciones de ruido distintas salen 37-3: la
        # realizacion es una por configuracion y re-particionar posiciones no la
        # vuelve a sortear. Vale como aviso de que el remuestreo mide
        # incertidumbre DE POSICION y de ninguna otra cosa.
        gana = self._gana(self._corre(ganancia_rival=3.0, semilla_rival=8))
        self.assertLess(gana, 0.5)

    def test_es_determinista_con_la_misma_semilla(self):
        # Un remuestreo que no se puede reproducir no es una barra de error: dos
        # lecturas del mismo CSV tienen que dar el mismo veredicto.
        uno = self._corre(ganancia_rival=6.0, sorteos=25)
        otro = self._corre(ganancia_rival=6.0, sorteos=25)
        self.assertEqual(uno["elegidas"], otro["elegidas"])
        self.assertEqual(uno["delta"][(20.0, 8.0)], otro["delta"][(20.0, 8.0)])

    def test_no_cuenta_mas_elecciones_que_sorteos(self):
        res = self._corre(ganancia_rival=6.0, sorteos=25)
        self.assertGreater(sum(res["elegidas"].values()), 0)
        self.assertLessEqual(sum(res["elegidas"].values()), 25)


if __name__ == "__main__":
    unittest.main()
