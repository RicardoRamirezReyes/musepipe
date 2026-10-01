"""El estimador del nivel del sesgo de controles de psffit (`scripts/nivel_sesgo_controles.py`).

Es lo que sostiene el r₀ de los dos objetos (docs/2026-09-27_r0_entregado_42bb.md) y vivió fuera del
repo hasta el 2026-09-28. Las cuatro primeras clases no tocan `runs/`: fijan las propiedades de las que
dependen las cifras -la geometría de los controles de C4, el emparejado, la lectura del cruce en su
tramo y el bootstrap-. `CifrasPublicadasTests` sí lee `runs/` (solo lectura, ~5 s por el bootstrap) y
comprueba las cifras de los docs del 09-21, 09-27 y 09-28 a media unidad del último dígito publicado.
Son copias de trabajo que se pueden borrar para liberar disco: si falta alguna, su caso se salta, no
falla (como los demás tests que leen `runs/`; `external_data` es para las bibliotecas de G3).
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
        np.testing.assert_allclose(self.m.sesgos_de(obj, ctl, absoluto=True), [1.0, -2.0])

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

    def test_el_error_de_r0_es_sigma_del_nivel_de_referencia_sobre_la_pendiente(self):
        # Cada radio con su propio ruido (proyectado fuera de 1, cos, sin: los niveles salen exactos), así
        # que σ(A) es distinta en cada radio: el error tiene que salir de la del extremo más cercano al
        # cruce (55), no de la del otro extremo del tramo (45). Hasta el 2026-10-01 el ruido era el mismo
        # en los tres radios y el test pasaba con cualquiera de las dos.
        rng = np.random.default_rng(7)
        phi = _phi()
        X = np.c_[np.ones_like(phi), np.cos(phi), np.sin(phi)]
        fuera = lambda v: v - X @ np.linalg.lstsq(X, v, rcond=None)[0]  # noqa: E731
        comun = fuera(rng.normal(0, 6.0, phi.size))
        ruido = {45.0: 9.0, 55.0: 1.0, 78.0: 3.0}
        datos = [(A + comun + fuera(rng.normal(0, ruido[r], phi.size)), r)
                 for r, A in sorted({45.0: -9.6, 55.0: 2.4, 78.0: 13.9}.items())]
        niveles = [self.m.resumen("", b, phi, r) for b, r in datos]
        tramos = [self.m.emparejado(datos[i][0], datos[i + 1][0], phi, datos[i][1], datos[i + 1][1])
                  for i in range(2)]
        c = self.m.cruce(niveles, tramos)
        s, es = tramos[0]["pendiente"], tramos[0]["e_pendiente"]
        self.assertEqual(c["referencia_px"], 55.0)
        self.assertAlmostEqual(c["e_r0_px"], math.hypot(niveles[1]["eA"] / s, niveles[1]["A"] * es / s ** 2),
                               places=9)
        # el test distingue: con la σ(A) del otro extremo saldría otra cosa
        self.assertGreater(abs(niveles[0]["eA"] / s - c["e_r0_px"]), 0.3 * c["e_r0_px"])

    def test_la_curvatura_emparejada(self):
        phi, datos, _, _ = self._radios({45.0: -9.6, 55.0: 2.4, 78.0: 13.9}, ruido=6.0)
        (b1, r1), (b2, r2), (b3, r3) = datos
        c = self.m.curvatura(b1, b2, b3, phi, r1, r2, r3)
        self.assertAlmostEqual(c["pendiente_alta_menos_baja"], 11.5 / 23.0 - 12.0 / 10.0, places=10)

    def test_el_error_de_la_curvatura_esta_calibrado(self):
        # Ruido común grande (lo quita el emparejado) más uno propio de cada radio: la dispersión de la
        # curvatura entre realizaciones tiene que ser la que dice su error, ni mayor ni menor.
        rng = np.random.default_rng(11)
        phi = _phi()
        cs, es = [], []
        for _ in range(600):
            comun = rng.normal(0, 6.0, phi.size)
            b = [A + comun + rng.normal(0, 1.0, phi.size) for A in (-9.6, 2.4, 13.9)]
            c = self.m.curvatura(*b, phi, 45.0, 55.0, 78.0)
            cs.append(c["pendiente_alta_menos_baja"])
            es.append(c["e"])
        self.assertAlmostEqual(np.std(cs, ddof=1) / np.mean(es), 1.0, delta=0.1)


class BootstrapTests(unittest.TestCase):
    """El error de r₀ cuando el cruce puede caer a cualquier lado del radio de referencia (2026-10-01)."""

    @classmethod
    def setUpClass(cls):
        cls.m = _carga("nivel_sesgo_controles")

    def _semi_anchos(self, niveles_por_radio):
        rng = np.random.default_rng(4)
        phi = _phi()
        rs = sorted(niveles_por_radio)
        X = np.c_[np.ones_like(phi), np.cos(phi), np.sin(phi)]
        comun = rng.normal(0, 6.0, phi.size)
        comun -= X @ np.linalg.lstsq(X, comun, rcond=None)[0]   # A55 sale ≈ 0, no desplazado por el ruido
        bs = [niveles_por_radio[r] + comun + rng.normal(0, 0.5, phi.size) for r in rs]
        x = self.m.bootstrap_r0(bs, phi, rs, n=1500, semilla=1)
        r0 = self.m.analiza(bs, phi, rs)["cruce"]["r0_px"]
        res = self.m.resumen_bootstrap(x, r0)
        return res["menos_68"], res["mas_68"]

    def test_con_curvatura_y_a_ref_cerca_de_cero_el_error_es_asimetrico(self):
        # pendiente 1.2 %/px por debajo de 55 y 0.5 por encima, A55 ≈ 0: las muestras con A55 < 0
        # cruzan en el tramo de arriba, 2.4 veces más tendido (medido: ~2.1 veces más ancho por arriba)
        menos, mas = self._semi_anchos({45.0: -12.0, 55.0: 0.0, 78.0: 11.5})
        self.assertGreater(mas, 1.5 * menos)

    def test_sin_curvatura_es_simetrico(self):
        menos, mas = self._semi_anchos({45.0: -10.5, 55.0: 0.0, 78.0: 24.15})
        self.assertAlmostEqual(mas / menos, 1.0, delta=0.25)

    def test_la_cola_va_del_lado_de_la_hipotesis(self):
        x = np.random.default_rng(2).normal(50.0, 5.0, 200_000)
        res = self.m.resumen_bootstrap(x, 50.0, hipotesis=(40.0, 55.0))
        self.assertAlmostEqual(res["hipotesis"]["40"]["fraccion"], 0.02275, delta=0.002)
        self.assertAlmostEqual(res["hipotesis"]["40"]["sigma_una_cola"], 2.0, delta=0.03)
        self.assertAlmostEqual(res["hipotesis"]["55"]["fraccion"], 0.1587, delta=0.003)
        self.assertAlmostEqual(res["menos_68"], 5.0, delta=0.1)
        self.assertAlmostEqual(res["mas_68"], 5.0, delta=0.1)


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
     {"pendientes": [(0.3171, 0.0066), (0.1965, 0.0047)], "tol": 0.00005, "r0_px": 62.2, "e_r0_px": 2.8,
      "tramo_px": [55.0, 78.0], "curvatura": (-0.1206, 0.0029)}),
    (("ROXs42Bb_comb45_20260921", "ROXs42Bb_comb60_20260921", "ROXs42Bb_comb78_20260921"),
     {"pendientes": [(1.266, 0.110), (0.536, 0.059)], "tol": 0.0005, "r0_px": 59.1, "e_r0_px": 4.6,
      "tramo_px": [45.0, 60.0], "curvatura": (-0.730, 0.051)}),
]
#: docs/2026-09-28_r0_entregado_42bb_tres_radios.md §2: el entregado de 42B b con el cruce encerrado.
TRES_RADIOS_0928 = (("ROXs42Bb_r45_20260928", "ROXs42Bb_realigned", "ROXs42Bb_r78_20260927"),
                    {"A45": (-11.52, 7.21), "dipolo45": 37.0, "mediana45": -15.91,
                     "tramos": [(13.10, 1.08, 1.310, 0.108), (12.61, 1.37, 0.548, 0.059)],
                     "curvatura": (-0.761, 0.049), "r0_px": 53.8, "e_r0_px": 4.7,
                     # §2: en flujo absoluto, sin dividir por la mediana del objeto
                     "mediana_objeto": (100.8, 119.6, 156.3), "curvatura_abs": (-0.467, 0.0075), "r0_abs_px": 53.6,
                     # §2-§3: bootstrap de 10 000 con semilla 0; percentiles 15.9 y 84.1, y por hipótesis la
                     # fracción de muestras más allá
                     "bootstrap_68": (48.4, 62.7), "bootstrap_95": (41.2, 70.3),
                     "hipotesis": {55.0: 0.4217, 53.1: 0.4210, 48.0: 0.1449, 46.6: 0.1000, 39.0: 0.0126, 36.0: 0.0060}})
#: §3 del 09-28: distancia gaussiana (r₀ − predicho)/σ(r₀) con cada predicción tomada como exacta.
DISTANCIAS_0928 = {55.0: -0.26, 53.1: +0.15, 48.0: +1.24, 46.6: +1.54, 39.0: +3.16, 36.0: +3.80}
R0_0927 = [
    (("ROXs42Bb_realigned", "ROXs42Bb_r78_20260927"), {"r0_px": 52.1, "e_r0_px": 11.2}, False),
    # 12 b: A55 = +0.004 ± 0.55, el cruce está en 55 pero ningún tramo lo encierra (extrapolado desde 55 → 64)
    (("ROXs12b_r55_20260921", "ROXs12b_r64_20260921"), {"r0_px": 55.0, "e_r0_px": 4.1}, False),
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
        for pareja, doc, encierra in R0_0927:
            with self.subTest(runs=pareja):
                if not _hay(*pareja):
                    self.skipTest(f"{pareja} no está en runs/")
                c = self.m.radios(pareja)["cruce"]
                self.assertIs(c["encierra_el_cruce"], encierra)
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
                # la nota de §5.2 (añadida el 09-28): curvatura emparejada, control a control
                c = r["curvatura"][0]
                self.assertAlmostEqual(c["pendiente_alta_menos_baja"], doc["curvatura"][0], delta=doc["tol"] + 1e-12)
                self.assertAlmostEqual(c["e"], doc["curvatura"][1], delta=doc["tol"] + 1e-12)

    def test_tres_radios_entregados_del_0928(self):
        trio, doc = TRES_RADIOS_0928
        if not _hay(*trio):
            self.skipTest(f"{trio} no está en runs/")
        bs, phi, rs, nombres = self.m.sesgos_por_radio(trio)
        r = self.m.analiza(bs, phi, rs, nombres)
        n45 = r["niveles"][0]
        self.assertAlmostEqual(n45["A"], doc["A45"][0], delta=0.005 + 1e-9)
        self.assertAlmostEqual(n45["eA"], doc["A45"][1], delta=0.005 + 1e-9)
        self.assertAlmostEqual(n45["dipolo"], doc["dipolo45"], delta=0.05 + 1e-9)
        self.assertAlmostEqual(n45["mediana"], doc["mediana45"], delta=0.005 + 1e-9)
        for t, (dA, edA, s, es) in zip(r["tramos"], doc["tramos"]):
            self.assertAlmostEqual(t["dA"], dA, delta=0.005 + 1e-9)
            self.assertAlmostEqual(t["e_dA"], edA, delta=0.005 + 1e-9)
            self.assertAlmostEqual(t["pendiente"], s, delta=0.0005 + 1e-12)
            self.assertAlmostEqual(t["e_pendiente"], es, delta=0.0005 + 1e-12)
        c = r["curvatura"][0]
        self.assertAlmostEqual(c["pendiente_alta_menos_baja"], doc["curvatura"][0], delta=0.0005 + 1e-12)
        self.assertAlmostEqual(c["e"], doc["curvatura"][1], delta=0.0005 + 1e-12)
        x = r["cruce"]
        self.assertTrue(x["encierra_el_cruce"])
        self.assertEqual(x["tramo_px"], [45.0, 55.0])
        self.assertAlmostEqual(x["r0_px"], doc["r0_px"], delta=0.05 + 1e-9)
        self.assertAlmostEqual(x["e_r0_px"], doc["e_r0_px"], delta=0.05 + 1e-9)
        for h, d in DISTANCIAS_0928.items():
            self.assertAlmostEqual((x["r0_px"] - h) / x["e_r0_px"], d, delta=0.005 + 1e-9, msg=f"distancia a {h}")
        # el geométrico en el marco entregado, con el error de su predicción: 46.6 · (55.0 ± 4.1) / 71.2
        self.assertAlmostEqual((x["r0_px"] - 36.0) / math.hypot(x["e_r0_px"], 4.1 * 46.6 / 71.2), 3.30,
                               delta=0.005 + 1e-9)

        ra = self.m.radios(trio, absoluto=True)
        # la mediana del objeto es el cociente entre los dos modos, la misma para todos los controles
        for b, babs, med in zip(bs, self.m.sesgos_por_radio(trio, absoluto=True)[0], doc["mediana_objeto"]):
            self.assertAlmostEqual(float(np.median(100.0 * babs / b)), med, delta=0.05 + 1e-9)
        ca = ra["curvatura"][0]
        self.assertAlmostEqual(ca["pendiente_alta_menos_baja"], doc["curvatura_abs"][0], delta=0.0005 + 1e-12)
        self.assertAlmostEqual(ca["e"], doc["curvatura_abs"][1], delta=0.00005 + 1e-12)
        self.assertAlmostEqual(ra["cruce"]["r0_px"], doc["r0_abs_px"], delta=0.05 + 1e-9)

        bt = self.m.resumen_bootstrap(self.m.bootstrap_r0(bs, phi, rs, n=10000, semilla=0), x["r0_px"],
                                      hipotesis=tuple(doc["hipotesis"]))
        p = bt["percentiles"]
        self.assertAlmostEqual(p["p15.9"], doc["bootstrap_68"][0], delta=0.05 + 1e-9)
        self.assertAlmostEqual(p["p84.1"], doc["bootstrap_68"][1], delta=0.05 + 1e-9)
        self.assertAlmostEqual(p["p2.3"], doc["bootstrap_95"][0], delta=0.05 + 1e-9)
        self.assertAlmostEqual(p["p97.7"], doc["bootstrap_95"][1], delta=0.05 + 1e-9)
        for h, f in doc["hipotesis"].items():
            self.assertAlmostEqual(bt["hipotesis"][f"{h:g}"]["fraccion"], f, delta=1e-9, msg=f"hipótesis {h}")


if __name__ == "__main__":
    unittest.main()
