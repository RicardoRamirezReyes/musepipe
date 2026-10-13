"""Ajuste conjunto MUSE + J/H/K (`scripts/ajuste_conjunto.py`).

Da las cifras de la Tabla B.4 y de §5.3 del paper (docs/2026-10-13_ajuste_conjunto.md). Las primeras clases no
leen nada de disco: con plantillas y segmentos sintéticos, el ajuste recupera la plantilla, la A_V y el r_K con
que se fabricó el dato, y el prior de escala cuesta (Δ/σ)². `CifrasDelPaperTests` lee los runs, las bibliotecas
de G3 y el IR de /mnt/2TB (marca `slow`); si falta algo se salta, no falla.
"""
import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def _carga(nombre):
    spec = importlib.util.spec_from_file_location(nombre, ROOT / "scripts" / f"{nombre}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[nombre] = mod
    spec.loader.exec_module(mod)
    return mod


def _escenario(aj, i_verdad=1, av=2.0, rv=3.1, rk=0.3, td=1500.0, desvio_k=1.0, sigma_k=0.05):
    from musepipe.models.extinction import CCMExtinction
    tp = _carga("tipo_ir_plantillas")
    ws = {"V": np.linspace(6400.0, 9300.0, 120), "H": np.linspace(15000.0, 17800.0, 80),
          "K": np.linspace(20800.0, 23500.0, 80)}
    brazo = {"V": "vis", "H": "nir", "K": "nir"}

    def plantilla(a, c):
        return {k: (w / 1e4) ** -a * (1 - 0.2 * np.exp(-0.5 * ((w - c * w.mean()) / (0.02 * w.mean())) ** 2))
                for k, w in ws.items()}
    base_d = {0: plantilla(1.0, 0.98), 1: plantilla(2.0, 1.00), 2: plantilla(3.0, 1.02)}
    bases = {i: [d[k] for k in ("V", "H", "K")] for i, d in base_d.items()}
    m = bases[i_verdad]
    mref = float(np.interp(aj.LAM_K, ws["K"], m[2]))
    segs = []
    for j, k in enumerate(("V", "H", "K")):
        w = ws[k]
        e = 10 ** (-0.4 * CCMExtinction(rv, citation="x").a_lambda_over_av(w) * av)
        bb = aj._planck(w, td) / aj._planck(aj.LAM_K, td) if brazo[k] == "nir" else 0.0
        f = 3.0 * (m[j] + rk * mref * bb) * e * (desvio_k if k == "K" else 1.0)
        b = tp.bordes(w)
        segs.append(dict(nombre=k, brazo=brazo[k], w=w, lo=b[:-1], hi=b[1:], f=f, e=0.002 * f,
                         bueno=np.ones(w.size, bool), sigma_escala=(sigma_k if brazo[k] == "nir" else 0.2)))
    return segs, bases


class ConjuntoTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.aj = _carga("ajuste_conjunto")

    def test_recupera_plantilla_av_y_rk(self):
        segs, bases = _escenario(self.aj)
        conj = self.aj.Conjunto(segs, bases, orden_ir=0)
        rej = dict(av=np.round(np.arange(0, 4.01, 0.2), 2), rv=(3.1,), rk=np.round(np.arange(0, 0.81, 0.1), 2), td=(1500.0,))
        cubos = self.aj.explora(conj, bases, list(bases), rej)
        i = min(cubos, key=lambda q: float(np.min(cubos[q])))
        a, _, c, _ = np.unravel_index(int(np.argmin(cubos[i])), cubos[i].shape)
        self.assertEqual(i, 1)
        self.assertAlmostEqual(rej["av"][a], 2.0, places=6)
        self.assertAlmostEqual(rej["rk"][c], 0.3, places=6)
        self.assertLess(float(np.min(cubos[i])), 1e-6)

    def test_el_prior_de_escala_cuesta_su_cuadrado(self):
        segs, bases = _escenario(self.aj, desvio_k=1.10, sigma_k=0.05)
        conj = self.aj.Conjunto(segs, bases, orden_ir=0)
        chi, g, _ = conj.chi2(bases[1], 2.0, 3.1, 0.3, 1500.0)
        # K un 10 % alto con σ 5 %: el prior reparte el desvío entre K, H y MUSE; el coste queda entre 0 y 4
        self.assertGreater(chi, 0.5)
        self.assertLess(chi, 4.0 + 1e-6)


class ModelosBtSettlTests(unittest.TestCase):
    def test_filtra_por_teff_y_logg_y_conserva_el_flujo(self):
        import tempfile
        from musepipe.models.cache import write_spectrum_npz
        aj = _carga("ajuste_conjunto")
        w = np.arange(4000.0, 25500.0, 0.25)
        with tempfile.TemporaryDirectory() as tmp:
            for teff, logg in ((2600, 3.5), (2600, 5.0), (3500, 3.5)):
                write_spectrum_npz(Path(tmp) / f"teff{teff}_logg{logg:.1f}.npz", w, 2.0 + np.sin(w / 300.0),
                                   {"teff_k": float(teff), "logg": logg}, wave_lo=4000.0, wave_hi=25500.0)
            ms = aj.modelos_btsettl(tmp)
        self.assertEqual([(m["teff"], m["logg"]) for m in ms], [(2600.0, 3.5)])
        g, f = ms[0]["vis"]
        np.testing.assert_allclose(f, 2.0 + np.sin(g / 300.0), atol=2e-6)
        self.assertLess(g.max(), 10500.0)
        self.assertGreater(ms[0]["nir"][0].min(), 9500.0)


def _hay_datos():
    return (_carga("tipo_ir_indices").DEFAULT_CDS.is_dir() and _carga("tipo_ir_plantillas").DEFAULT_NIR.is_dir()
            and (ROOT / "runs" / "ROXs12b_invvar" / "stages").is_dir()
            and (ROOT / "runs" / "ROXs42Bb_realigned" / "stages").is_dir())


@pytest.mark.slow
@unittest.skipUnless(_hay_datos(), "sin catálogo CDS, sin el NIR de las plantillas o sin los runs del paper")
class CifrasDelPaperTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.aj = _carga("ajuste_conjunto")

    def test_mejor_ajuste(self):
        r12, r42 = self.aj.ajusta("ROXs12b"), self.aj.ajusta("ROXs42Bb")
        self.assertEqual((r12["mejor"]["spt"], r12["mejor"]["av"], r12["mejor"]["rk"]), ("M9", 2.2, 0.3))
        self.assertEqual((r42["mejor"]["spt"], r42["mejor"]["av"], r42["mejor"]["rk"]), ("M9.5", 3.0, 0.2))

    def test_error_dejando_una_fuera(self):
        c = self.aj.calibra("ROXs12b")
        self.assertAlmostEqual(c["rms_av"], 0.88, delta=0.01)
        self.assertAlmostEqual(c["rms_tipo"], 1.18, delta=0.01)
