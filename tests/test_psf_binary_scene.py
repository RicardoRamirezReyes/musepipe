"""La escena de dos componentes ligadas de C1, y lo que NO puede cambiar.

ROXs 42B es una binaria cercana no resuelta (rho=51 mas, PA=148 deg segun
Keck/NIRC2 2022.621 -> 2.006 px). C1 ajustaba UNA PSF a DOS estrellas y la media
mas ancha de lo que es, sesgando la `apcorr` un +6.9 % contra un +1.5 % de suelo
en ROXs 12 b, que es una estrella sola
(`docs/2026-08-30_binaria_42b_sesga_la_apcorr.md`).

Lo que estos tests protegen, por orden de importancia:

  1. **Sin el knob, nada cambia.** Es lo que impide que este cambio toque a los
     demas objetos y a los productos ya congelados.
  2. **La PSF publicada y la escena son cosas distintas.** `evaluate_moffat_fit`
     devuelve UN punto —lo que consumen la inyeccion de E4, la `apcorr` y el
     throughput— y `evaluate_moffat_scene` las dos estrellas, que es contra lo
     que se miden los residuos. Confundirlas haria que el anillo empeorase justo
     al mejorar el modelo.
  3. **El objetivo real no es medir `f`, es recuperar la PSF verdadera.** Con una
     binaria sintetica el ajuste de una componente sale ensanchado y el de dos
     vuelve al valor verdadero.
"""

import contextlib
import io
import unittest
import warnings

import numpy as np

from musepipe.psf import (
    evaluate_moffat_fit,
    evaluate_moffat_scene,
    fit_moffat_image,
    moffat_image,
)

FORMA = dict(fwhm_maj=4.2, fwhm_min=3.6, theta_deg=20.0, beta=2.6)
#: Keck/NIRC2 2022.621 llevado a pixeles: la misma geometria que usa el run.
OFFSET = (-1.701, -1.063)


#: Radio de ajuste. 9 px como la sonda `binaria_42b_probe.py`: es donde vive la
#: separacion de 2 px, asi que es donde el sesgo de ignorarla se ve.
R_FIT = 9.0


def escena(shape, centro, *, ratio=0.0, offset=OFFSET, amplitude=1000.0, fondo=5.0,
           ruido=0.0, semilla=0):
    """Una o dos Moffat, con ruido opcional.

    **El ruido no es decoracion.** Sin el, el residuo del ajuste de dos
    componentes es exactamente cero, su sigma robusta tiende a cero y el `chi2r`
    —que divide por ella— se dispara: comparar ajustes por `chi2r` sobre datos
    sin ruido no mide nada. Con ruido, la comparacion vuelve a significar algo.
    """
    img = moffat_image(shape, centro[0], centro[1], amplitude=amplitude,
                       background=fondo, **FORMA)
    if ratio:
        img = img + moffat_image(shape, centro[0] + offset[0], centro[1] + offset[1],
                                 amplitude=amplitude * ratio, background=0.0, **FORMA)
    if ruido:
        img = img + np.random.default_rng(semilla).normal(0.0, ruido, size=shape)
    return img


class SinKnobNadaCambia(unittest.TestCase):
    def test_una_sola_fuente_da_el_mismo_ajuste_que_siempre(self):
        img = escena((70, 70), (35.0, 35.0))
        viejo = fit_moffat_image(img, center_yx=(35.0, 35.0), fit_radius_px=R_FIT)
        self.assertIsNone(viejo.flux_ratio)
        self.assertIsNone(viejo.companion_offset_yx)

    def test_la_escena_sin_segunda_componente_es_la_psf(self):
        img = escena((70, 70), (35.0, 35.0))
        fit = fit_moffat_image(img, center_yx=(35.0, 35.0), fit_radius_px=R_FIT)
        np.testing.assert_array_equal(
            evaluate_moffat_fit((70, 70), fit), evaluate_moffat_scene((70, 70), fit)
        )


class RazonDeFlujosNula(unittest.TestCase):
    def test_f_tiende_a_cero_sobre_una_estrella_sola(self):
        # El control de ROXs 12 b, en sintetico: pedirle una binaria a una
        # estrella sola no puede inventarla.
        img = escena((70, 70), (35.0, 35.0))
        fit = fit_moffat_image(img, center_yx=(35.0, 35.0), fit_radius_px=R_FIT,
                               companion_offset_yx=OFFSET)
        self.assertLess(fit.flux_ratio, 0.05)

    def test_con_f_nulo_la_forma_es_la_de_una_componente(self):
        img = escena((70, 70), (35.0, 35.0))
        una = fit_moffat_image(img, center_yx=(35.0, 35.0), fit_radius_px=R_FIT)
        dos = fit_moffat_image(img, center_yx=(35.0, 35.0), fit_radius_px=R_FIT,
                               companion_offset_yx=OFFSET)
        for clave in ("fwhm_maj", "fwhm_min", "beta"):
            with self.subTest(clave=clave):
                self.assertAlmostEqual(una.params[clave], dos.params[clave], delta=0.15)


class RecuperaLaPsfVerdadera(unittest.TestCase):
    """El objetivo del cambio: la PSF, no la razon de flujos."""

    def setUp(self):
        self.shape = (70, 70)
        self.centro = (35.0, 35.0)
        self.ratio = 0.12
        self.img = escena(self.shape, self.centro, ratio=self.ratio, ruido=0.5, semilla=7)

    def test_una_componente_mide_la_psf_mas_ancha_de_lo_que_es(self):
        una = fit_moffat_image(self.img, center_yx=self.centro, fit_radius_px=R_FIT)
        self.assertGreater(una.params["fwhm_maj"], FORMA["fwhm_maj"] + 0.05)

    def test_dos_componentes_devuelven_la_forma_verdadera(self):
        dos = fit_moffat_image(self.img, center_yx=self.centro, fit_radius_px=R_FIT,
                               companion_offset_yx=OFFSET)
        self.assertAlmostEqual(dos.params["fwhm_maj"], FORMA["fwhm_maj"], delta=0.10)
        self.assertAlmostEqual(dos.params["fwhm_min"], FORMA["fwhm_min"], delta=0.10)
        self.assertAlmostEqual(dos.params["beta"], FORMA["beta"], delta=0.20)

    def test_y_de_paso_recupera_la_razon_de_flujos(self):
        dos = fit_moffat_image(self.img, center_yx=self.centro, fit_radius_px=R_FIT,
                               companion_offset_yx=OFFSET)
        self.assertAlmostEqual(dos.flux_ratio, self.ratio, delta=0.03)

    def test_la_recuperacion_no_depende_del_radio_de_ajuste(self):
        # El ajuste de UNA componente sesga de forma erratica con el radio -en
        # sintetico limpio hay radios donde acierta por casualidad-, asi que un
        # test a un solo radio podria pasar por suerte. El de dos tiene que
        # acertar en TODOS.
        img = escena(self.shape, self.centro, ratio=self.ratio)
        for radio in (6.0, 9.0, 12.0, 20.0, 30.0):
            with self.subTest(radio=radio):
                dos = fit_moffat_image(img, center_yx=self.centro, fit_radius_px=radio,
                                       companion_offset_yx=OFFSET)
                self.assertAlmostEqual(dos.params["fwhm_maj"], FORMA["fwhm_maj"], delta=0.05)
                self.assertAlmostEqual(dos.flux_ratio, self.ratio, delta=0.02)

    def test_el_ajuste_de_dos_deja_menos_residuo_sobre_una_binaria(self):
        # Con ruido, y comparando el residuo directamente: `chi2r` normaliza por
        # la sigma del propio residuo, asi que no compara dos ajustes entre si.
        yy, xx = np.indices(self.shape, dtype=float)
        dentro = np.hypot(yy - self.centro[0], xx - self.centro[1]) <= R_FIT
        una = fit_moffat_image(self.img, center_yx=self.centro, fit_radius_px=R_FIT)
        dos = fit_moffat_image(self.img, center_yx=self.centro, fit_radius_px=R_FIT,
                               companion_offset_yx=OFFSET)
        r_una = np.nansum(np.abs(self.img - evaluate_moffat_scene(self.shape, una))[dentro])
        r_dos = np.nansum(np.abs(self.img - evaluate_moffat_scene(self.shape, dos))[dentro])
        self.assertLess(float(r_dos), float(r_una))


class LaPsfYLaEscenaNoSeConfunden(unittest.TestCase):
    def setUp(self):
        self.shape = (70, 70)
        self.centro = (35.0, 35.0)
        self.img = escena(self.shape, self.centro, ratio=0.12, ruido=0.5, semilla=7)
        self.fit = fit_moffat_image(self.img, center_yx=self.centro, fit_radius_px=R_FIT,
                                    companion_offset_yx=OFFSET)

    def test_la_publicada_no_lleva_la_secundaria_y_la_escena_si(self):
        psf = evaluate_moffat_fit(self.shape, self.fit)
        esc = evaluate_moffat_scene(self.shape, self.fit)
        self.assertGreater(float(np.nansum(esc - psf)), 0.0)
        # El exceso esta DONDE dice la astrometria, no en cualquier sitio.
        dif = esc - psf
        iy, ix = np.unravel_index(int(np.nanargmax(dif)), dif.shape)
        self.assertAlmostEqual(iy - self.centro[0], OFFSET[0], delta=1.0)
        self.assertAlmostEqual(ix - self.centro[1], OFFSET[1], delta=1.0)

    def test_la_escena_deja_menos_residuo_sobre_el_dato_que_la_psf(self):
        # Es la razon de que las metricas usen la escena: el dato tiene DOS.
        psf = evaluate_moffat_fit(self.shape, self.fit)
        esc = evaluate_moffat_scene(self.shape, self.fit)
        yy, xx = np.indices(self.shape, dtype=float)
        cerca = np.hypot(yy - self.centro[0], xx - self.centro[1]) <= 12.0
        self.assertLess(float(np.nansum(np.abs(self.img - esc)[cerca])),
                        float(np.nansum(np.abs(self.img - psf)[cerca])))


class RamaPsfao(unittest.TestCase):
    """La rama que usa ROXs 42B b, y cuyo ajuste lo hace `maoppy`.

    **Las afirmaciones son sobre el COSTE y sobre `f`, no sobre los siete
    parametros de psfao ni sobre la imagen.** Ese espacio es degenerado —`r0` y
    `C` se compensan— y dos vectores muy distintos alcanzan el mismo coste; es
    una propiedad conocida de psfao, no de este cambio. Afirmar sobre los
    parametros daria un test que falla por el motivo equivocado.
    """

    @classmethod
    def setUpClass(cls):
        from maoppy.instrument import muse_nfm
        from maoppy.psfmodel import Psfao

        cls.ny = cls.nx = 64
        cls.samp = 2.0
        cls.system = muse_nfm
        cls.verdad = [0.55, 2e-2, 1.2, 2e-2, 1.1, 0.2, 1.6]
        cls.modelo = Psfao((cls.ny, cls.nx), system=muse_nfm, samp=cls.samp)
        cls.base = cls.modelo(cls.verdad, dx=0, dy=0)
        cls.comp = (58.0, 58.0)          # companero lejos: no entra en el ajuste
        cls.var = np.full((cls.ny, cls.nx), 0.16)
        rng = np.random.default_rng(3)
        cls.ruido_a = rng.normal(0, 0.4, (cls.ny, cls.nx))
        cls.ruido_b = rng.normal(0, 0.4, (cls.ny, cls.nx))
        cls.ratio = 0.12

    def _ajusta(self, img, *, offset=None, ratio0=0.1):
        from musepipe.stages.stage_e01_psfao import fit_bin, fit_bin_binary

        # maoppy imprime su progreso por stdout en cada iteracion.
        with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if offset is None:
                return fit_bin(img, self.var, self.samp, self.system, self.comp,
                               4.0, 25.0, self.verdad)
            return fit_bin_binary(img, self.var, self.samp, self.system, self.comp,
                                  4.0, 25.0, self.verdad, offset, ratio0=ratio0)

    def test_con_f_cero_alcanza_el_mismo_coste_que_maoppy(self):
        # El test de regresion que el diseno regala: con la segunda componente
        # apagada, esto ES la funcion de coste de `maoppy.psffit`.
        img = 1000.0 * self.base + 3.0 + self.ruido_a
        maoppy = self._ajusta(img)
        nuestro = self._ajusta(img, offset=OFFSET, ratio0=0.0)
        self.assertAlmostEqual(nuestro[6]["cost"], maoppy[6]["cost"],
                               delta=1e-3 * maoppy[6]["cost"])
        self.assertAlmostEqual(nuestro[8], 0.0, places=4)

    def test_recupera_la_razon_de_flujos_de_una_binaria_sintetica(self):
        img = (1000.0 * (self.base + self.ratio * self.modelo(
            self.verdad, dx=OFFSET[1], dy=OFFSET[0])) + 3.0 + self.ruido_b)
        dos = self._ajusta(img, offset=OFFSET)
        self.assertAlmostEqual(dos[8], self.ratio, delta=0.02)

    def test_dos_componentes_ajustan_mejor_una_binaria(self):
        img = (1000.0 * (self.base + self.ratio * self.modelo(
            self.verdad, dx=OFFSET[1], dy=OFFSET[0])) + 3.0 + self.ruido_b)
        una = self._ajusta(img)
        dos = self._ajusta(img, offset=OFFSET)
        self.assertLess(dos[6]["cost"], una[6]["cost"])

    def test_los_ejes_no_estan_cruzados(self):
        """La trampa silenciosa: en maoppy `dx` mueve el eje 1 y `dy` el eje 0.

        Cruzarlos pondria la secundaria en una posicion equivocada y NADA
        fallaria: el ajuste convergeria igual, solo que peor. Se comprueba
        pidiendo el mismo ajuste con el offset invertido y exigiendo que el
        correcto gane.
        """
        img = (1000.0 * (self.base + self.ratio * self.modelo(
            self.verdad, dx=OFFSET[1], dy=OFFSET[0])) + 3.0 + self.ruido_b)
        bueno = self._ajusta(img, offset=OFFSET)
        cruzado = self._ajusta(img, offset=(OFFSET[1], OFFSET[0]))
        self.assertLess(bueno[6]["cost"], cruzado[6]["cost"])

    def test_devuelve_la_escena_y_la_psf_por_separado(self):
        img = (1000.0 * (self.base + self.ratio * self.modelo(
            self.verdad, dx=OFFSET[1], dy=OFFSET[0])) + 3.0 + self.ruido_b)
        dos = self._ajusta(img, offset=OFFSET)
        escena_img, psf_img = dos[5], dos[9]
        self.assertGreater(float(np.nansum(escena_img - psf_img)), 0.0)


if __name__ == "__main__":
    unittest.main()


class LasMetricasVenLaEscena(unittest.TestCase):
    """V4 y el anillo comparan MODELO contra DATO, asi que necesitan la escena.

    El error que esto impide es sutil y va en la direccion equivocada: si la
    energia encerrada comparase la PSF sola contra un dato que tiene DOS
    estrellas, veria un deficit del tamano de la razon de flujos (~11 % en
    ROXs 42B b) y lo denunciaria como fallo del modelo — es decir, el V4
    **empeoraria justo al mejorar la PSF**, y encima por la cantidad exacta que
    el cambio acaba de arreglar.
    """

    def test_la_escena_encierra_mas_energia_que_la_psf(self):
        from musepipe.psf import encircled_energy

        shape = (70, 70)
        centro = (35.0, 35.0)
        img = escena(shape, centro, ratio=0.12, ruido=0.5, semilla=7)
        fit = fit_moffat_image(img, center_yx=centro, fit_radius_px=R_FIT,
                               companion_offset_yx=OFFSET)
        radios = np.array([3.0, 6.0, 12.0])
        ee_psf = encircled_energy(evaluate_moffat_fit(shape, fit), centro, radios,
                                  background=fit.background)
        ee_esc = encircled_energy(evaluate_moffat_scene(shape, fit), centro, radios,
                                  background=fit.background)
        # La secundaria esta a 2 px, asi que ya entra en el radio mas pequeno.
        for i, r in enumerate(radios):
            with self.subTest(radio=r):
                self.assertGreater(float(ee_esc[i]), float(ee_psf[i]))
