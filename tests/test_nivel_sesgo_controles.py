"""El estimador del nivel del sesgo de controles de psffit (`scripts/nivel_sesgo_controles.py`).

Es lo que sostiene el r₀ de los dos objetos (docs/2026-09-27_r0_entregado_42bb.md) y vivió fuera del
repo hasta el 2026-09-28. Las tres primeras clases no tocan `runs/`: fijan las propiedades de las que
dependen las cifras -la geometría de los controles de C4, el emparejado y la lectura del cruce en su
tramo-. `CifrasPublicadasTests` sí lee `runs/` (solo lectura, < 1 s) y comprueba las cifras de los
docs del 09-21 y del 09-27 a media unidad del último dígito publicado. Son copias de trabajo que se
pueden borrar para liberar disco: si falta alguna, su caso se salta, no falla.
"""
import importlib.util
import math
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def _carga(nombre):
    spec = importlib.util.spec_from_file_location(nombre, ROOT / "scripts" / f"{nombre}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[nombre] = mod
    spec.loader.exec_module(mod)
    return mod


def _phi(n=33):
    return np.linspace(0.3, 2 * np.pi - 0.3, n)


class GeometriaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.m = _carga("nivel_sesgo_controles")

    def test_la_geometria_de_42bb_da_sus_33_controles(self):
        # primaria y compañero de B3 en el cubo de 170x170 de C4; 38 controles, ±25° excluidos
        phi = self.m.azimuts((85.27, 131.68), (84.99, 85.03), 170, 170, n_controls=38,
                             exclude_angle_deg=25.0, comp_radius_px=12.0)
        self.assertEqual(phi.size, 33)
        lejos = np.abs(np.angle(np.exp(1j * phi)))
        self.assertGreater(lejos.min(), math.radians(25.0) - math.radians(1.0))

    def test_el_margen_es_el_de_psffit(self):
        # tres controles a 12 px del borde: caben con comp_radius 11 (margen ceil(11)+1 = 12) y
        # ninguno con 12 (margen 13), que es el de C4
        kw = dict(n_controls=4, exclude_angle_deg=25.0)
        self.assertEqual(self.m.azimuts((52.0, 92.0), (52.0, 52.0), 105, 105, comp_radius_px=11.0, **kw).size, 3)
        self.assertEqual(self.m.azimuts((52.0, 92.0), (52.0, 52.0), 105, 105, comp_radius_px=12.0, **kw).size, 0)


class AjusteTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.m = _carga("nivel_sesgo_controles")

    def test_recupera_nivel_y_dipolo_sin_ruido(self):
        phi = _phi()
        b = 2.5 + 3.0 * np.cos(phi) - 4.0 * np.sin(phi)
        A, eA, dip = self.m.ajuste(b, phi)
        self.assertAlmostEqual(A, 2.5, places=10)
        self.assertAlmostEqual(dip, 5.0, places=10)
        self.assertLess(eA, 1e-10)

    def test_los_canales_no_finitos_del_objeto_no_cuentan(self):
        obj = np.array([10.0, np.nan, 10.0, 10.0])
        ctl = np.array([[1.0, 1e9, 1.0, 1.0], [-2.0, -1e9, -2.0, -2.0]])
        np.testing.assert_allclose(self.m.sesgos_de(obj, ctl), [10.0, -20.0])

    def test_el_ajuste_es_lineal_en_b(self):
        # A(b2 - b1) = A2 - A1 exactamente: r0 no depende del extremo del tramo
        rng = np.random.default_rng(3)
        phi = _phi()
        b1, b2 = rng.normal(0, 5, phi.size), rng.normal(3, 5, phi.size)
        d = self.m.emparejado(b1, b2, phi, 45.0, 55.0)
        self.assertAlmostEqual(d["dA"], self.m.ajuste(b2, phi)[0] - self.m.ajuste(b1, phi)[0], places=10)

    def test_emparejar_quita_el_error_comun(self):
        # el error de cada nivel es comun a los dos radios (correlacion 0.999 medida): la diferencia no lo ve
        rng = np.random.default_rng(5)
        phi = _phi()
        comun = rng.normal(0, 6.0, phi.size)
        b55 = 1.5 + comun + rng.normal(0, 0.2, phi.size)
        b78 = 14.0 + comun + rng.normal(0, 0.2, phi.size)
        d = self.m.emparejado(b55, b78, phi, 55.0, 78.0)
        self.assertLess(d["e_dA"] * 4, self.m.ajuste(b55, phi)[1])
        self.assertAlmostEqual(d["dA"], 12.5, delta=4 * d["e_dA"])


class CruceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.m = _carga("nivel_sesgo_controles")

    def _radios(self, niveles_por_radio, ruido=0.0, semilla=1):
        rng = np.random.default_rng(semilla)
        phi = _phi()
        comun = rng.normal(0, ruido, phi.size)
        X = np.c_[np.ones_like(phi), np.cos(phi), np.sin(phi)]
        comun -= X @ np.linalg.lstsq(X, comun, rcond=None)[0]   # los niveles salen exactos
        datos = [(A + 2.0 * np.cos(phi) + comun, r) for r, A in sorted(niveles_por_radio.items())]
        niveles = [self.m.resumen(f"r{r:.0f}", b, phi, r) for b, r in datos]
        tramos = [self.m.emparejado(datos[i][0], datos[i + 1][0], phi, datos[i][1], datos[i + 1][1])
                  for i in range(len(datos) - 1)]
        return phi, datos, niveles, tramos

    def test_el_cruce_se_lee_en_el_tramo_que_lo_encierra(self):
        # curvo: pendiente 1.2 %/px por debajo de 55 y 0.5 por encima; el cero esta en 53
        _, _, niveles, tramos = self._radios({45.0: -9.6, 55.0: 2.4, 78.0: 13.9}, ruido=6.0)
        c = self.m.cruce(niveles, tramos)
        self.assertTrue(c["encierra_el_cruce"])
        self.assertEqual(c["tramo_px"], [45.0, 55.0])
        self.assertEqual(c["referencia_px"], 55.0)
        self.assertAlmostEqual(c["r0_px"], 53.0, places=6)

    def test_sin_tramo_que_lo_encierre_extrapola_y_lo_dice(self):
        _, _, niveles, tramos = self._radios({55.0: 1.57, 78.0: 14.19}, ruido=6.0)
        c = self.m.cruce(niveles, tramos)
        self.assertFalse(c["encierra_el_cruce"])
        self.assertIn("EXTRAPOLADO", c["nota"])
        self.assertAlmostEqual(c["r0_px"], 55.0 - 1.57 / ((14.19 - 1.57) / 23.0), places=6)

    def test_el_error_de_r0_es_sigma_del_nivel_sobre_la_pendiente(self):
        # con ruido comun, mas radios no lo bajan: lo fija sigma(A_ref)
        _, _, niveles, tramos = self._radios({45.0: -9.6, 55.0: 2.4, 78.0: 13.9}, ruido=6.0)
        c = self.m.cruce(niveles, tramos)
        eA55 = niveles[1]["eA"]
        self.assertAlmostEqual(c["e_r0_px"], eA55 / tramos[0]["pendiente"], delta=0.05 * c["e_r0_px"])

    def test_la_curvatura_emparejada(self):
        phi, datos, _, _ = self._radios({45.0: -9.6, 55.0: 2.4, 78.0: 13.9}, ruido=6.0)
        (b1, r1), (b2, r2), (b3, r3) = datos
        c = self.m.curvatura(b1, b2, b3, phi, r1, r2, r3)
        self.assertAlmostEqual(c["pendiente_alta_menos_baja"], 11.5 / 23.0 - 12.0 / 10.0, places=10)


#: (run, cifras del doc del 09-21) — mediana, media, nivel A ± σ, dipolo.
CIFRAS_0921 = [
    ("ROXs12b_comb78_20260921", {"mediana": 3.70, "media": 3.33, "A": 3.11, "eA": 0.48, "dipolo": 6.8}),
    ("ROXs12b_comb55_20260921", {"mediana": -0.52}),
    ("ROXs12b_comb40_20260921", {"mediana": -4.17}),
    ("ROXs12b_r64_20260921", {"A": 1.21, "eA": 0.53}),
    ("ROXs12b_r55_20260921", {"A": 0.00, "eA": 0.55, "dipolo": 8.6}),
    ("ROXs42Bb_comb78_20260921", {"A": 10.77, "eA": 4.74, "dipolo": 22.2}),
    ("ROXs42Bb_comb60_20260921", {"A": 1.12, "eA": 5.80, "dipolo": 28.4}),
    ("ROXs42Bb_comb45_20260921", {"A": -17.87, "eA": 7.44, "dipolo": 39.1}),
]
#: Niveles y r₀ entregados del doc del 09-27 (§3).
CIFRAS_0927 = [
    ("ROXs42Bb_realigned", {"A": 1.57, "eA": 6.14}),
    ("ROXs42Bb_r78_20260927", {"A": 14.19, "eA": 4.78}),
]
#: §5.2 del 09-27: alcance combinado, pendientes por tramo y r₀ en el tramo que encierra el cruce.
TRAMOS_0927 = [
    (("ROXs12b_comb40_20260921", "ROXs12b_comb55_20260921", "ROXs12b_comb78_20260921"),
     {"pendientes": [(0.3171, 0.0066), (0.1965, 0.0047)], "tol": 0.00005, "r0_px": 62.2, "e_r0_px": 2.8, "tramo_px": [55.0, 78.0]}),
    (("ROXs42Bb_comb45_20260921", "ROXs42Bb_comb60_20260921", "ROXs42Bb_comb78_20260921"),
     {"pendientes": [(1.266, 0.110), (0.536, 0.059)], "tol": 0.0005, "r0_px": 59.1, "e_r0_px": 4.6, "tramo_px": [45.0, 60.0]}),
]
#: docs/2026-09-28_r0_entregado_42bb_tres_radios.md §2: el entregado de 42B b con el cruce encerrado.
TRES_RADIOS_0928 = (("ROXs42Bb_r45_20260928", "ROXs42Bb_realigned", "ROXs42Bb_r78_20260927"),
                    {"A45": (-11.52, 7.21), "pendiente_baja": (1.310, 0.108), "curvatura": (-0.761, 0.049),
                     "r0_px": 53.8, "e_r0_px": 4.7})
R0_0927 = [
    (("ROXs42Bb_realigned", "ROXs42Bb_r78_20260927"), {"r0_px": 52.1, "e_r0_px": 11.2}),
    (("ROXs12b_r55_20260921", "ROXs12b_r64_20260921"), {"r0_px": 55.0, "e_r0_px": 4.1}),
]


def _hay(*runs):
    return all((ROOT / "runs" / r / "stages" / "spec_calibrated_psffit_controls.npz").exists() for r in runs)


class CifrasPublicadasTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.m = _carga("nivel_sesgo_controles")

    def test_niveles_de_los_docs_del_0921_y_0927(self):
        for run, doc in CIFRAS_0921 + CIFRAS_0927:
            with self.subTest(run=run):
                if not _hay(run):
                    self.skipTest(f"{run} no está en runs/")
                res = self.m.resumen(run, *self.m.lee_run(run))
                for k, v in doc.items():
                    self.assertAlmostEqual(res[k], v, delta=(0.05 if k == "dipolo" else 0.005) + 1e-9, msg=k)

    def test_r0_del_0927(self):
        for pareja, doc in R0_0927:
            with self.subTest(runs=pareja):
                if not _hay(*pareja):
                    self.skipTest(f"{pareja} no está en runs/")
                c = self.m.radios(pareja)["cruce"]
                for k, v in doc.items():
                    self.assertAlmostEqual(c[k], v, delta=0.05 + 1e-9, msg=k)

    def test_tramos_y_cruce_combinados_del_0927(self):
        for trio, doc in TRAMOS_0927:
            with self.subTest(runs=trio):
                if not _hay(*trio):
                    self.skipTest(f"{trio} no está en runs/")
                r = self.m.radios(trio)
                for t, (s_doc, es_doc) in zip(r["tramos"], doc["pendientes"]):
                    # «tol»: media unidad del último dígito publicado
                    self.assertAlmostEqual(t["pendiente"], s_doc, delta=doc["tol"] + 1e-12)
                    self.assertAlmostEqual(t["e_pendiente"], es_doc, delta=doc["tol"] + 1e-12)
                self.assertTrue(r["cruce"]["encierra_el_cruce"])
                self.assertEqual(r["cruce"]["tramo_px"], doc["tramo_px"])
                self.assertAlmostEqual(r["cruce"]["r0_px"], doc["r0_px"], delta=0.05 + 1e-9)
                self.assertAlmostEqual(r["cruce"]["e_r0_px"], doc["e_r0_px"], delta=0.05 + 1e-9)


    def test_tres_radios_entregados_del_0928(self):
        trio, doc = TRES_RADIOS_0928
        if not _hay(*trio):
            self.skipTest(f"{trio} no está en runs/")
        r = self.m.radios(trio)
        n45 = r["niveles"][0]
        self.assertAlmostEqual(n45["A"], doc["A45"][0], delta=0.005 + 1e-9)
        self.assertAlmostEqual(n45["eA"], doc["A45"][1], delta=0.005 + 1e-9)
        self.assertAlmostEqual(r["tramos"][0]["pendiente"], doc["pendiente_baja"][0], delta=0.0005 + 1e-12)
        self.assertAlmostEqual(r["tramos"][0]["e_pendiente"], doc["pendiente_baja"][1], delta=0.0005 + 1e-12)
        c = r["curvatura"][0]
        self.assertAlmostEqual(c["pendiente_alta_menos_baja"], doc["curvatura"][0], delta=0.0005 + 1e-12)
        self.assertAlmostEqual(c["e"], doc["curvatura"][1], delta=0.0005 + 1e-12)
        x = r["cruce"]
        self.assertTrue(x["encierra_el_cruce"])
        self.assertEqual(x["tramo_px"], [45.0, 55.0])
        self.assertAlmostEqual(x["r0_px"], doc["r0_px"], delta=0.05 + 1e-9)
        self.assertAlmostEqual(x["e_r0_px"], doc["e_r0_px"], delta=0.05 + 1e-9)


if __name__ == "__main__":
    unittest.main()
