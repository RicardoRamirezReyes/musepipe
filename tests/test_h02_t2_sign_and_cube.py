"""T2: el signo de la amplitud, y el cubo en el que busca.

Los dos fallos que este test fija se midieron el 2026-09-02 sobre ROXs 12 b:

1. T2 leia `cube_psffit_residual`, el residuo de C4. Pero C4 ajusta
   `psf_pair_design` -primaria Y companera- y resta las dos, asi que T2 buscaba
   una fuente puntual en un cubo del que ya habia sido eliminada. En el mismo
   canal, el residuo de C4 daba -7.2 sigma y el cubo de 04b +10.4 sigma.
2. Sin exigir amplitud positiva, una SOBRE-sustraccion mejora el chi2 igual que
   una emision: el combinado pasaba T2 con una PSF de -7.2 sigma.
"""
from __future__ import annotations

import unittest

import numpy as np

from musepipe.stages.stage_h02_artifacts import _load_signal_cube_and_wave, t2_spatial_coherence


def _psf(shape=(11, 11), fwhm=2.5):
    yy, xx = np.indices(shape, dtype=float)
    cy, cx = (shape[0] - 1) / 2, (shape[1] - 1) / 2
    sig = fwhm / 2.3548
    return np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * sig ** 2))


class T2SignTests(unittest.TestCase):
    def test_fuente_positiva_pasa(self):
        psf = _psf()
        r = t2_spatial_coherence(10.0 * psf, psf)
        self.assertEqual(r["status"], "pass")
        self.assertGreater(r["psf_amplitude"], 0)

    def test_sobre_sustraccion_falla_aunque_ajuste_bien(self):
        """Una PSF NEGATIVA ajusta el chi2 igual de bien: no puede pasar."""
        psf = _psf()
        r = t2_spatial_coherence(-10.0 * psf, psf)
        self.assertLess(r["chi2_ratio_psf_vs_plane"], 0.8,
                        "el ajuste mejora el chi2 igual que con emision")
        self.assertLess(r["psf_amplitude"], 0)
        self.assertEqual(r["status"], "fail", "una absorcion no es una deteccion")

    def test_el_signo_se_puede_desactivar_declarandolo(self):
        psf = _psf()
        r = t2_spatial_coherence(-10.0 * psf, psf, require_positive_amplitude=False)
        self.assertEqual(r["status"], "pass")

    def test_publica_amplitud_y_su_snr(self):
        psf = _psf()
        rng = np.random.default_rng(3)
        r = t2_spatial_coherence(10.0 * psf + rng.normal(0, 0.2, psf.shape), psf)
        self.assertIn("psf_amplitude", r)
        self.assertIn("psf_amplitude_snr", r)
        self.assertGreater(r["psf_amplitude_snr"], 3.0)


class T2CubeTests(unittest.TestCase):
    def test_prefiere_el_cubo_con_la_companera_dentro(self):
        """El orden de candidatos es el arreglo: 04b antes que el residuo de C4."""
        import inspect
        # solo la linea de candidatos: el docstring nombra los dos y en otro orden
        linea = next(l for l in inspect.getsource(_load_signal_cube_and_wave).splitlines()
                     if l.strip().startswith("candidates = ["))
        self.assertLess(linea.index("cube_residual_object"), linea.index("cube_psffit_residual"),
                        "el residuo de C4 tiene la companera restada: no puede ir primero")


if __name__ == "__main__":
    unittest.main()
