"""`scripts/teff_bayes.py --cov-banda` (2026-11-02): la verosimilitud de cada banda es una gaussiana con covarianza,
ruido + f_b² m mᵀ ∘ k(ℓ_b) (Matérn 3/2 sobre todos los espectros de la banda), con ℓ_b libre; el ruido de MUSE lleva
su correlación entre bins solo con los datos reales.

Lo que se prueba son invariantes, no cifras del run: sin la opción todo es lo de siempre (la etiqueta, el json, el
`Modelo`); las combinaciones sin sentido fallan antes de leer nada; `modelo_de_cadena` lee la opción del json. Las
piezas: el núcleo, la gaussiana (contra scipy) y la correlación del ruido entre bins (contra la suma directa). Con los
datos: los parámetros de siempre más un `lnl_` por banda; cada banda junta sus espectros; la verosimilitud es la
gaussiana multivariante; con datos de prueba el ruido es diagonal y una ℓ diminuta da la verosimilitud de siempre; con
los datos reales, el ruido de MUSE está correlacionado entre bins vecinos y el del IR no.
"""
import argparse
import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pytest
from scipy import stats

from tests._datos import carga_script, modelo_teff, requiere, runs_del_paper

#: un json de cadena del paper, con las claves que llevan (las de `tests/test_teff_bayes_variantes.py`)
PAPER = dict(etiqueta="ROXs12b_libre_avprimaria_s1", slug="ROXs12b", run="ROXs12b_invvar", modo="libre",
             av="primaria", av_fija=None, rejilla="btsettl", datos="todo", r_analisis=500, nodos_interpolados=[],
             ejes={}, tracks=None, nlive=500, semilla=1, ndim=12, nombres=[], segundos=1.0, ncall=1, logz=-1.0,
             logz_err=0.1, logl_max=-1.0, chi2nu_mejor={}, chi2nu_mejor_inflado={}, mejor={}, verdad=None,
             posterior={}, n_puntos={})
#: los compañeros con una banda de dos espectros (la que la covarianza junta)
DOBLE = {"ROXs12b": "J", "ROXs42Bb": "H"}


class ValidaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tb = carga_script("teff_bayes")

    def test_sin_la_opcion_no_falla_nada(self):
        self.tb.valida_cov_banda(False)
        self.tb.valida_cov_banda(False, muse_nativo=True, r250=True, banda_una_vez=True)
        self.tb.valida_cov_banda(True)

    def test_las_combinaciones_sin_sentido_fallan(self):
        for kw in (dict(banda_una_vez=True), dict(muse_nativo=True), dict(r250=True)):
            with self.assertRaises(ValueError, msg=str(kw)):
                self.tb.valida_cov_banda(True, **kw)

    def test_modelo_falla_antes_de_leer_nada(self):
        # un slug que no existe: si validara después de leer, el error sería otro
        with self.assertRaises(ValueError):
            self.tb.Modelo("no_existe", cov_banda=True, banda_una_vez=True)
        with self.assertRaises(ValueError):
            self.tb.Modelo("no_existe", cov_banda=True, rejilla="sphinx")   # R = 250 por la rejilla

    def test_un_modelo_de_new_no_lleva_covarianza(self):
        self.assertFalse(self.tb.Modelo.__new__(self.tb.Modelo).cov_banda)


class PiezasTests(unittest.TestCase):
    """El núcleo, la gaussiana y la correlación del ruido entre bins, sin datos."""

    @classmethod
    def setUpClass(cls):
        cls.tb = carga_script("teff_bayes")

    def test_nucleo(self):
        d3 = math.sqrt(3.0) * np.array([0.0, 10.0, 50.0, 200.0])
        k = self.tb.matern32(d3, 50.0)
        self.assertEqual(k[0], 1.0)
        self.assertTrue(np.all(np.diff(k) < 0))
        self.assertAlmostEqual(k[2], (1 + math.sqrt(3)) * math.exp(-math.sqrt(3)), places=14)

    def test_gaussiana_diagonal_y_completa(self):
        rng = np.random.default_rng(0)
        n = 40
        r, v = rng.normal(size=n), rng.uniform(0.5, 2.0, n)
        diag = -0.5 * float(np.sum(r * r / v + np.log(v)))
        self.assertAlmostEqual(self.tb.loglike_gauss(r, np.diag(v)), diag, delta=1e-12 * abs(diag))
        A = rng.normal(size=(n, n))
        C = A @ A.T + n * np.eye(n)
        ref = stats.multivariate_normal(mean=np.zeros(n), cov=C).logpdf(r) + 0.5 * n * math.log(2 * math.pi)
        self.assertAlmostEqual(self.tb.loglike_gauss(r, C), ref, delta=1e-10 * abs(ref))
        self.assertIsNone(self.tb.loglike_gauss(r, -np.eye(n)))

    def test_correlacion_entre_bins(self):
        n, L = 240, 14.0                 # L = n/n_eff de un canal degradado (~ el de MUSE a R = 500)
        e = np.ones(n)
        grupos = [np.arange(a, a + 12) for a in range(60, 180, 12)]
        rho = self.tb.correlacion_bins(e, np.full(n, L), grupos)
        np.testing.assert_allclose(np.diag(rho), 1.0, rtol=0, atol=1e-14)
        np.testing.assert_allclose(rho, rho.T, rtol=0, atol=1e-14)
        self.assertGreater(np.linalg.eigvalsh(rho).min(), -1e-12)
        # contra la suma directa con ρ(Δ) = exp(−Δ²/4s²), s = L/(2√π)
        s = L / (2 * math.sqrt(math.pi))

        def c(A, B):
            return sum(math.exp(-(i - j) ** 2 / (4 * s * s)) for i in A for j in B)
        directa = c(grupos[0], grupos[1]) / math.sqrt(c(grupos[0], grupos[0]) * c(grupos[1], grupos[1]))
        self.assertAlmostEqual(rho[0, 1], directa, delta=1e-3)
        self.assertGreater(rho[0, 1], 0.2)
        self.assertLess(abs(rho[0, 4]), 1e-6)
        # σ distinta por canal no cambia la correlación si es la misma en todos; un ruido casi blanco no correlaciona
        np.testing.assert_allclose(self.tb.correlacion_bins(3.0 * e, np.full(n, L), grupos), rho, atol=1e-14)
        self.assertLess(self.tb.correlacion_bins(e, np.full(n, 1.0), grupos)[0, 1], 1e-3)


class EtiquetaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tb = carga_script("teff_bayes")

    def _a(self, **kw):
        base = dict(etiqueta=None, slug="ROXs42Bb", rejilla="atmo_ceq1500", modo="libre", tracks="ATMO2020",
                    av="primaria", plantilla=None, av_fija=1.8, inyecta_bt=None, datos="ir", r250=False,
                    semilla=1)
        base.update(kw)
        return argparse.Namespace(**base)

    def test_sin_la_opcion_la_de_siempre(self):
        for a in (self._a(), self._a(cov_banda=False)):
            self.assertEqual(self.tb.etiqueta_de(a), "ROXs42Bb_atmo_ceq1500_libre_avprimaria_avfija1.8_soloir_s1")

    def test_la_pieza_va_detras_de_la_inclinacion(self):
        self.assertEqual(self.tb.etiqueta_de(self._a(cov_banda=True)),
                         "ROXs42Bb_atmo_ceq1500_libre_avprimaria_avfija1.8_covbanda_soloir_s1")
        self.assertEqual(self.tb.etiqueta_de(self._a(cov_banda=True, pendiente_ir=0.3)),
                         "ROXs42Bb_atmo_ceq1500_libre_avprimaria_avfija1.8_pendir0.3_covbanda_soloir_s1")


class VarianteTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tb = carga_script("teff_bayes")

    def test_es_variante_y_no_entra_en_las_tablas(self):
        d = {**PAPER, "cov_banda": True}
        self.assertTrue(self.tb._es_variante(d))
        self.assertFalse(self.tb._de_la_tabla(d))
        self.assertFalse(self.tb._de_rejillas(d))
        self.assertFalse(self.tb._es_variante({**PAPER, "cov_banda": False}))


class ReconstruyeTests(unittest.TestCase):
    """`modelo_de_cadena` le pasa al `Modelo` lo que dice el json (con un `Modelo` de mentira que lo apunta)."""

    @classmethod
    def setUpClass(cls):
        cls.tb = carga_script("teff_bayes")

    def test_lee_cov_banda(self):
        visto = {}

        class Falso:
            nombres = ["teff"]

            def __init__(self, *args, **kw):
                visto.update(kw)

        with mock.patch.object(self.tb, "Modelo", Falso):
            self.tb.modelo_de_cadena({**PAPER, "mejor": {"teff": 2600.0}})
            self.assertIs(visto["cov_banda"], False)
            self.tb.modelo_de_cadena({**PAPER, "mejor": {"teff": 2600.0}, "cov_banda": True})
            self.assertIs(visto["cov_banda"], True)


class _ModeloFalso:
    """Lo que `ajusta` lee de un `Modelo`, sin datos; apunta los argumentos por nombre."""
    visto = {}

    def __init__(self, *args, **kw):
        type(self).visto = dict(kw)
        self.run, self.r_analisis, self.rellenados, self.degrada = "RUN", 500, [], None
        self.rk_fijo, self.td_rango = None, None
        self.T0, self.dT, self.nT, self.G0, self.dG, self.nG = 2000.0, 100.0, 13, 2.5, 0.5, 6
        self.nombres, self.ndim = ["teff", "logg", "av"], 3
        self.dir_rejilla = Path("/rejilla")
        self.segs = [dict(nombre="OSIRIS_K", d=np.zeros(84))]

    def chi2(self, th, inflado=False):
        return {"OSIRIS_K": 1.0}


def _corre_falso(mod, nlive, semilla):
    m = np.ones((10, mod.ndim))
    return dict(segundos=0.0, ncall=1, logz=-1.0, logz_err=0.1, muestras=m, derivadas=np.ones((10, 5)),
                mejor=m[0], logl_max=-1.0)


class AjustaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tb = carga_script("teff_bayes")

    def _json(self, *opciones):
        with tempfile.TemporaryDirectory() as t, mock.patch.object(self.tb, "Modelo", _ModeloFalso), \
                mock.patch.object(self.tb, "corre", _corre_falso):
            self.tb.main(["ajusta", "ROXs42Bb", "--datos", "ir", "--salida", t, *opciones])
            (f,) = Path(t).glob("*.json")
            return json.loads(f.read_text())

    def test_sin_la_opcion_ni_clave_ni_argumento(self):
        d = self._json()
        self.assertNotIn("cov_banda", d)
        self.assertNotIn("cov_banda", _ModeloFalso.visto)

    def test_con_la_opcion_clave_etiqueta_y_argumento(self):
        d = self._json("--cov-banda")
        self.assertIs(d["cov_banda"], True)
        self.assertEqual(d["etiqueta"], "ROXs42Bb_libre_avprimaria_covbanda_soloir_s0")
        self.assertIs(_ModeloFalso.visto["cov_banda"], True)
        self.assertLessEqual(set(d), self.tb.CLAVES_JSON | set(self.tb.VARIANTES_JSON))


# ------------------------------------------------------------------ datos reales
def _a_mano(mod, th):
    """La verosimilitud con covarianza, con scipy: por banda, la gaussiana multivariante de media el modelo."""
    M = mod.modelos(th)
    tot = 0.0
    for c in mod._cov:
        m = np.concatenate([M[k] for k in c["ks"]])
        fb, ell = math.exp(th[mod.pos[f"lnf_{c['banda']}"]]), math.exp(th[mod.pos[f"lnl_{c['banda']}"]])
        x = c["d3"] / ell
        C = c["ruido"] + fb ** 2 * np.outer(m, m) * (1 + x) * np.exp(-x)
        tot += stats.multivariate_normal(mean=m, cov=C).logpdf(c["d"]) + 0.5 * m.size * math.log(2 * math.pi)
    return tot


@pytest.mark.external_data
@pytest.mark.slow
@requiere("runs", "ir", "bibliotecas")
class ModeloDatosTests(unittest.TestCase):
    """Con BT-Settl de /mnt/2TB: una plantilla con MUSE y el IR, sin y con la covarianza, y los datos con ella."""

    @classmethod
    def setUpClass(cls):
        cls.tb = carga_script("teff_bayes")
        slugs = [s for s in runs_del_paper() if s in DOBLE]
        if not slugs:
            raise unittest.SkipTest("hace falta un compañero con dos espectros de una banda")
        cls.a = slugs[0]
        try:
            cls.sin = modelo_teff(cls.a, plantilla="TWA26", semilla=1)
            cls.con = modelo_teff(cls.a, plantilla="TWA26", semilla=1, cov_banda=True)
            cls.real = modelo_teff(cls.a, cov_banda=True)
        except (OSError, SystemExit) as exc:   # sin las rejillas de /mnt/2TB
            raise unittest.SkipTest(f"no se puede construir el modelo: {exc}")

    def _puntos(self, mod, n, semilla):
        rng = np.random.default_rng(semilla)
        return [mod.prior_transform(rng.uniform(0.05, 0.95, mod.ndim)) for _ in range(n)]

    def test_los_de_siempre_mas_un_lnl_por_banda(self):
        self.assertEqual(self.con.nombres[:self.sin.ndim], self.sin.nombres)
        self.assertEqual(self.con.nombres[self.sin.ndim:], [f"lnl_{b}" for b in self.con.bandas])
        for s, c in zip(self.sin.segs, self.con.segs):
            for k in ("w", "d", "e2", "lg"):
                np.testing.assert_array_equal(c[k], s[k], err_msg=f"{s['nombre']} {k}")
        th = self._puntos(self.con, 1, 0)[0]
        for mc, ms in zip(self.con.modelos(th), self.sin.modelos(th[:self.sin.ndim])):
            np.testing.assert_array_equal(mc, ms)

    def test_cada_banda_junta_sus_espectros(self):
        self.assertEqual([c["banda"] for c in self.con._cov], self.con.bandas)
        for c in self.con._cov:
            self.assertEqual(c["ks"], [k for k, s in enumerate(self.con.segs) if s["banda"] == c["banda"]])
            np.testing.assert_array_equal(c["d"], np.concatenate([self.con.segs[k]["d"] for k in c["ks"]]))
            np.testing.assert_array_equal(c["d3"], c["d3"].T)
            self.assertTrue(np.all(np.diag(c["d3"]) == 0))
        doble = next(c for c in self.con._cov if c["banda"] == DOBLE[self.a])
        self.assertEqual(len(doble["ks"]), 2)

    def test_la_verosimilitud_es_la_gaussiana_multivariante(self):
        for mod in (self.con, self.real):
            for th in self._puntos(mod, 3, 1):
                ref = _a_mano(mod, th)
                self.assertAlmostEqual(mod.loglike(th), ref, delta=1e-9 * abs(ref))

    def test_con_datos_de_prueba_ruido_diagonal_y_l_diminuta_es_lo_de_siempre(self):
        for c in self.con._cov:
            self.assertFalse(c["ruido_correlado"])
            np.testing.assert_array_equal(c["ruido"], np.diag(np.diag(c["ruido"])))
        for th in self._puntos(self.con, 3, 2):
            th = np.array(th)
            for b in self.con.bandas:
                th[self.con.pos[f"lnl_{b}"]] = math.log(1e-3)
            ref = self.sin.loglike(th[:self.sin.ndim])
            self.assertAlmostEqual(self.con.loglike(th), ref, delta=1e-9 * abs(ref))

    def test_con_los_datos_reales_muse_lleva_la_correlacion_del_ruido(self):
        for c in self.real._cov:
            self.assertEqual(c["ruido_correlado"], c["banda"] == "vis", c["banda"])
            if c["banda"] != "vis":
                np.testing.assert_array_equal(c["ruido"], np.diag(np.diag(c["ruido"])))
                continue
            sd = np.sqrt(np.diag(c["ruido"]))
            rho = c["ruido"] / np.outer(sd, sd)
            np.testing.assert_allclose(np.diag(rho), 1.0, rtol=0, atol=1e-12)
            self.assertGreater(np.linalg.eigvalsh(rho).min(), -1e-10)
            vecinos = np.diag(rho, 1)
            self.assertTrue(0.2 < np.median(vecinos) < 0.6, np.median(vecinos))
            self.assertLess(np.abs(np.diag(rho, 3)).max(), 0.05)
            # la σ de cada bin es la del segmento
            np.testing.assert_array_equal(np.diag(c["ruido"]), self.real.segs[c["ks"][0]]["e2"])


if __name__ == "__main__":
    unittest.main()
