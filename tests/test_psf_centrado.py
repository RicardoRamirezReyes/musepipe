"""El modelo psfao se evalúa CENTRADO en la posición pedida.

Hasta el 2026-10-21 no lo estaba. `_psfao_image_cached` construía la imagen con
`model(x)`, y maoppy solo aplica la compensación del sobremuestreo, (k-1)/(2k) px,
cuando dx o dy no son cero. La imagen salía 1/3 px (k = 3) o 3/8 px (k = 4) fuera
en y y en x, y el evaluador la muestreaba como si su pico estuviera en
(npix//2, npix//2). Todo consumidor del modelo (la apcorr, los pesos de optimal,
las plantillas de psffit, las inyecciones de E4) ponía la PSF ~0.47 px en diagonal
lejos de donde se le pedía, y la apertura de 3x3 salía un 9-13 % alta.
`docs/2026-10-21_psf_evaluador_centrado.md`.

Lo que se fija aquí, con la simetría central y no con un centroide (el centroide en
una ventana pequeña depende del ancho de la PSF; la simetría no): la PSD de Psfao es
par, así que una PSF bien centrada cumple f(d) = f(-d) alrededor de la posición pedida.

* el evaluador la cumple a ambos lados de 6400 Å (k = 4 y k = 3), también en
  desplazamientos subpíxel;
* la apcorr de box3 que C1 usa para descartar bins sale de la PSF centrada;
* y, como guarda, el comportamiento de maoppy que causó el fallo. Si una versión
  nueva de maoppy centrara también con dx = dy = 0, ese test fallará:
  `MAOPPY_CENTRADO_PX` seguiría siendo inocuo, pero habría que actualizar la guarda.
"""

import unittest

import numpy as np

from musepipe.psf import MAOPPY_CENTRADO_PX, _psfao_image_cached, evaluate_psf_model

NORM_RADIUS = 25.0
# los de un bin de C1 cerca de Hα (ROXs12b_invvar, primera exposición, 6600 Å)
PARAMS = [0.228, 0.0103, 7.75, 1e-5, 0.828, 0.464, 1.6]
NAMES = ["r0", "C", "A", "alpha", "ratio", "theta", "beta"]
# k = 4 por debajo de ~6400 Å y k = 3 por encima, en muse_nfm
LONGITUDES_A = (5000.0, 6562.8, 9000.0)


def psfao_doc():
    """Documento psfao mínimo, con la forma que escribe C1."""

    return {
        "form": "psfao",
        "system": "muse_nfm",
        "param_names": NAMES,
        "norm_radius_px": NORM_RADIUS,
        "psfao_wave_bin_A": 100.0,
        "smoothed_poly": {n: [float(v)] for n, v in zip(NAMES, PARAMS)},
    }


def asimetria(f_mas, f_menos):
    """Mayor diferencia entre f(d) y f(-d), en unidades del máximo."""

    return float(np.max(np.abs(f_mas - f_menos)) / np.max(f_mas))


class EvaluadorCentradoTests(unittest.TestCase):
    def setUp(self):
        try:
            import maoppy  # noqa: F401
        except Exception:  # pragma: no cover - entorno sin maoppy
            self.skipTest("maoppy no disponible")
        _psfao_image_cached.cache_clear()

    def test_la_psf_es_simetrica_alrededor_de_la_posicion_pedida(self):
        # nucleo de 9x9 px, con y sin fase subpixel
        dy, dx = np.mgrid[-4:5, -4:5].astype(float)
        for lam in LONGITUDES_A:
            for fase in ((0.0, 0.0), (0.3, -0.2)):
                with self.subTest(lam=lam, fase=fase):
                    oy, ox = dy + fase[0], dx + fase[1]
                    f_mas = evaluate_psf_model(psfao_doc(), lam, oy, ox)
                    f_menos = evaluate_psf_model(psfao_doc(), lam, -oy, -ox)
                    # el fallo daba 0.41-0.52 del pico
                    self.assertLess(asimetria(f_mas, f_menos), 1e-6)

    def test_la_imagen_cacheada_es_simetrica_en_su_centro(self):
        from maoppy.instrument import muse_nfm

        for lam in LONGITUDES_A:
            with self.subTest(lam=lam):
                samp = round(float(muse_nfm.samp(lam * 1e-10)), 10)
                img, _total, c = _psfao_image_cached(tuple(PARAMS), 64, "muse_nfm", samp, NORM_RADIUS)
                nucleo = img[c - 4:c + 5, c - 4:c + 5]
                self.assertLess(asimetria(nucleo, nucleo[::-1, ::-1]), 1e-6)

    def test_la_apcorr_de_c1_sale_de_la_psf_centrada(self):
        from maoppy.instrument import muse_nfm
        from maoppy.psfmodel import Psfao

        from musepipe.stages.stage_e01_psfao import _box3_apcorr

        lam = 8000.0
        m = Psfao((52, 52), system=muse_nfm, samp=float(muse_nfm.samp(lam * 1e-10)))
        yy, xx = np.mgrid[0:52, 0:52]
        r = np.hypot(yy - 26, xx - 26)
        caja = (np.abs(yy - 26) <= 1) & (np.abs(xx - 26) <= 1)

        def apcorr(img):
            return float(np.nansum(img[r <= NORM_RADIUS]) / np.nansum(img[caja]))

        centrada = apcorr(np.asarray(m(PARAMS, dx=MAOPPY_CENTRADO_PX, dy=MAOPPY_CENTRADO_PX)))
        descentrada = apcorr(np.asarray(m(PARAMS)))
        # `_box3_apcorr` recorta los parametros a los limites de Psfao (eps 1e-6): de ahi el 1e-8
        self.assertAlmostEqual(_box3_apcorr(muse_nfm, lam, PARAMS, NORM_RADIUS), centrada, places=6)
        # descentrada, la caja de 3x3 recoge menos luz y la apcorr sobrecorrige (~3.5 % aqui)
        self.assertGreater(descentrada, centrada * 1.02)


class GuardaDeMaoppyTests(unittest.TestCase):
    """El comportamiento de maoppy que causó el fallo, para que no vuelva en silencio."""

    def setUp(self):
        try:
            import maoppy  # noqa: F401
        except Exception:  # pragma: no cover - entorno sin maoppy
            self.skipTest("maoppy no disponible")

    def test_sin_desplazamiento_maoppy_no_centra(self):
        from maoppy.instrument import muse_nfm
        from maoppy.psfmodel import Psfao

        for lam in (5000.0, 6562.8):
            with self.subTest(lam=lam):
                m = Psfao((64, 64), system=muse_nfm, samp=float(muse_nfm.samp(lam * 1e-10)))
                for img, centrada in ((np.asarray(m(PARAMS)), False),
                                      (np.asarray(m(PARAMS, dx=MAOPPY_CENTRADO_PX, dy=MAOPPY_CENTRADO_PX)), True)):
                    nucleo = img[28:37, 28:37]
                    a = asimetria(nucleo, nucleo[::-1, ::-1])
                    if centrada:
                        self.assertLess(a, 1e-6)
                    else:
                        self.assertGreater(a, 0.1, "maoppy ya centra con dx = dy = 0: actualizar esta guarda")


if __name__ == "__main__":
    unittest.main()
