"""El prior rápido de `scripts/teff_bayes.py` da los mismos números BIT A BIT que los `ppf` públicos de scipy.

Hasta el 2026-10-29 el prior usaba `stats.norm.ppf` y `stats.truncnorm.ppf`, que costaban 6–10 veces la
verosimilitud por las comprobaciones de argumentos de `rv_continuous.ppf`. `_ppf_normal_rapido` (`ndtri`) y
`_ppf_trunc_normal_rapido` (`truncnorm._ppf`, API privada de scipy) hacen la misma aritmética. Si una versión nueva
de scipy la cambia, estos tests fallan: entonces las cadenas nuevas ya no serían reproducibles bit a bit con las del
paper, y hay que volver al camino público o aceptarlo a sabiendas.
"""
import math
import unittest

import numpy as np
import pytest
from scipy import stats

from tests._datos import carga_script, modelo_teff, requiere, runs_del_paper

#: u de prueba: una malla, los extremos que dynesty puede dar y números al azar
U = np.concatenate([np.linspace(1e-12, 1 - 1e-12, 1001), [1e-300, 1e-15, 0.5, 1 - 1e-15],
                    np.random.default_rng(20261029).random(3000)])
#: (mu, sigma, lo) de los priores del modelo: A_V de las primarias (con su error; el de 42B b y el de 12 b, y otros),
#: las escalas de MUSE y de las bandas ancladas (N(1, 0.20) y N(1, 0.05) truncadas en 0.05)
TRUNCADAS = [(1.8, 0.5, 0.0), (1.8, 0.3, 0.0), (2.0, 0.4, 0.0), (0.5, 1.0, 0.0), (1.0, 0.20, 0.05), (1.0, 0.05, 0.05)]
NORMALES = [(144.0, 6.0), (135.0, 1.5), (math.log10(6.0), 0.2), (0.0, 1.0)]


class PpfTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tb = carga_script("teff_bayes")

    def test_normal_bit_a_bit(self):
        for mu, sig in NORMALES:
            f = self.tb._ppf_normal_rapido(mu, sig)
            rapido = np.array([f(u) for u in U])
            publico = np.array([float(stats.norm.ppf(u, mu, sig)) for u in U])
            self.assertTrue(np.array_equal(rapido, publico), (mu, sig))

    def test_normal_truncada_bit_a_bit(self):
        for mu, sig, lo in TRUNCADAS:
            f = self.tb._ppf_trunc_normal_rapido(mu, sig, lo=lo)
            rapido = np.array([f(u) for u in U[::4]])
            publico = np.array([self.tb._ppf_trunc_normal(u, mu, sig, lo=lo) for u in U[::4]])
            self.assertTrue(np.array_equal(rapido, publico), (mu, sig, lo))
            self.assertTrue(np.all(rapido >= lo))


def _sintetico(tb):
    """Un `Modelo` sin datos, con todos los tipos de prior: MUSE (escala N(1, 0.20)), K anclada (N(1, 0.05)) y H
    libre (log-uniforme)."""
    m = tb.Modelo.__new__(tb.Modelo)
    m.modo, m.T0, m.dT, m.nT, m.G0, m.dG, m.nG = "libre", 2000.0, 100.0, 13, 2.5, 0.5, 6
    m.d_pc, m.d_err, m.av_fija, m.av_modo, m.av0, m.av_err = 144.0, 6.0, None, "primaria", 1.8, 0.5
    m.segs = [dict(nombre="MUSE", banda="vis", sigma=0.20), dict(nombre="K", banda="K", sigma=0.05),
              dict(nombre="H", banda="H", sigma=None)]
    m.bandas = ["H", "K", "vis"]
    m._prepara_priores()
    return m


def _prior_publico(m):
    """El prior de `teff_bayes` hasta el 2026-10-29, con los `ppf` públicos (en modo `libre`, A_V con prior)."""
    t = []
    for k in m.nombres:
        if k == "teff":
            t.append(lambda u: m.T0 + u * (m.nT - 1) * m.dT)
        elif k == "logg":
            t.append(lambda u: m.G0 + u * (m.nG - 1) * m.dG)
        elif k == "r_rj":
            t.append(lambda u: 10 ** (math.log10(0.5) + u * (math.log10(6.0) - math.log10(0.5))))
        elif k == "d_pc":
            t.append(lambda u: float(stats.norm.ppf(u, m.d_pc, m.d_err)))
        elif k == "av":
            t.append(lambda u: float(stats.truncnorm.ppf(u, (0.0 - m.av0) / m.av_err, np.inf, loc=m.av0,
                                                         scale=m.av_err)))
        elif k == "rk":
            t.append(lambda u: 1.5 * u)
        elif k.startswith("s_"):
            sg = next(s["sigma"] for s in m.segs if f"s_{s['nombre']}" == k)
            if sg is None:
                t.append(lambda u: 10 ** (math.log10(0.2) + u * 2 * math.log10(5.0)))
            else:
                t.append((lambda sg: lambda u: float(stats.truncnorm.ppf(u, (0.05 - 1.0) / sg, np.inf, loc=1.0,
                                                                          scale=sg)))(sg))
        else:
            t.append(lambda u: math.log(1e-3) + u * (math.log(1.0) - math.log(1e-3)))
    return lambda u: np.array([f(x) for f, x in zip(t, u)])


class PriorDelModeloTests(unittest.TestCase):
    def test_prior_transform_igual_al_publico(self):
        tb = carga_script("teff_bayes")
        m = _sintetico(tb)
        self.assertEqual(m.nombres, ["teff", "logg", "r_rj", "d_pc", "av", "rk", "s_MUSE", "s_K", "s_H", "lnf_H",
                                     "lnf_K", "lnf_vis"])
        publico = _prior_publico(m)
        for u in np.random.default_rng(7).random((3000, m.ndim)):
            self.assertTrue(np.array_equal(m.prior_transform(u), publico(u)))


@pytest.mark.external_data
@pytest.mark.slow
@requiere("runs", "ir", "bibliotecas")
class DynestyTests(unittest.TestCase):
    def test_una_corrida_corta_da_las_mismas_muestras(self):
        """Misma semilla, prior rápido y prior público: las mismas muestras, logL y ln Z bit a bit (~1 min)."""
        import dynesty
        tb = carga_script("teff_bayes")
        slug = next(iter(runs_del_paper()))
        try:
            m = modelo_teff(slug)
        except (OSError, SystemExit) as exc:
            self.skipTest(f"no se puede construir el modelo: {exc}")
        publico = _prior_publico(m)

        def corre(pt):
            s = dynesty.NestedSampler(m.loglike, pt, m.ndim, nlive=60, sample=tb.MUESTREO["sample"],
                                      bound=tb.MUESTREO["bound"], rstate=np.random.default_rng(7))
            s.run_nested(maxcall=15000, print_progress=False)
            return s.results
        r1, r2 = corre(m.prior_transform), corre(publico)
        self.assertTrue(np.array_equal(r1.samples, r2.samples))
        self.assertTrue(np.array_equal(r1.logl, r2.logl))
        self.assertEqual(r1.logz[-1], r2.logz[-1])


if __name__ == "__main__":
    unittest.main()
