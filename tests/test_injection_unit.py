import unittest

import numpy as np

from musepipe.injection import (
    InjectionSource,
    inject,
    integrated_delta_flux,
    normalized_spatial_psf,
)


class InjectionUnitTests(unittest.TestCase):
    def test_inject_preserves_integrated_line_flux_in_mini_cube(self):
        wave = np.arange(6550.0, 6576.0, 1.0, dtype=np.float64)
        cube = np.zeros((wave.size, 7, 7), dtype=np.float64)
        psf = np.zeros((7, 7), dtype=np.float64)
        psf[3, 3] = 1.0
        source = InjectionSource(
            y=3.0,
            x=3.0,
            total_line_flux=12.5,
            line_center_A=6562.8,
            line_fwhm_A=2.5,
            psf_image=psf,
        )

        # `frame` explicito: esta prueba es sobre la contabilidad de flujo TOTAL
        # sobre el recorte, que es justo lo que esa convencion define. Con el
        # default (`norm_radius`) `total_line_flux` es el flujo dentro del radio
        # de normalizacion, no el del recorte, y esta igualdad no aplica.
        injected = inject(cube, [source], wavelengths_A=wave, norm_convention="frame")
        delta_flux = integrated_delta_flux(injected - cube, wave)

        self.assertAlmostEqual(delta_flux, 12.5, places=10)
        self.assertAlmostEqual(float(np.nansum(injected[:, 3, 3])), float(np.nansum(injected)))

    def test_inject_can_target_one_cube_in_stack(self):
        wave = np.arange(6558.0, 6568.0, 1.0, dtype=np.float64)
        cube = np.zeros((2, wave.size, 5, 5), dtype=np.float64)
        psf = np.zeros((5, 5), dtype=np.float64)
        psf[2, 2] = 1.0
        source = InjectionSource(
            y=2.0,
            x=2.0,
            total_line_flux=3.0,
            line_center_A=6562.8,
            line_fwhm_A=2.5,
            cube_index=1,
            psf_image=psf,
        )

        # `frame` por lo mismo que arriba: se comprueba a QUE cubo de la pila va
        # el flujo, no en que convencion se expresa.
        injected = inject(cube, [source], wavelengths_A=wave, norm_convention="frame")

        self.assertAlmostEqual(integrated_delta_flux(injected[0], wave), 0.0)
        self.assertAlmostEqual(integrated_delta_flux(injected[1], wave), 3.0)


if __name__ == "__main__":
    unittest.main()


class InjectionNormConventionTests(unittest.TestCase):
    """`total_line_flux` es flujo dentro de `norm_radius_px`, no del recorte.

    Cambiado el 2026-08-17: la extraccion entrega NORMRAD y la inyeccion
    normalizaba sobre el frame, asi que `injected_flux` y `recovered_flux` no
    eran la misma cantidad y el puente entre ellas dependia del MODELO. Medido en
    ROXs 12 b: cambiar el modelo de PSF de C1 movio el throughput de los seis
    metodos -6 % a la vez, sin relacion con la calidad de la extraccion.
    """

    def _model(self, fwhm, norm_radius):
        from tests.test_optimal_analytic import constant_model_doc

        doc = constant_model_doc(fwhm=fwhm)
        doc["norm_radius_px"] = float(norm_radius)
        return doc

    def test_the_flux_inside_the_norm_radius_is_what_was_asked_for(self):
        wave = np.arange(6558.0, 6568.0, 1.0, dtype=np.float64)
        cube = np.zeros((wave.size, 61, 61), dtype=np.float64)
        model = self._model(4.0, 12.0)
        source = InjectionSource(
            y=30.0, x=30.0, total_line_flux=100.0,
            line_center_A=6562.8, line_fwhm_A=2.5,
        )

        delta = inject(cube, [source], wavelengths_A=wave, psf_model=model) - cube
        yy, xx = np.mgrid[0:61, 0:61]
        dentro = np.hypot(yy - 30.0, xx - 30.0) <= 12.0
        widths = np.gradient(wave)
        dentro_flux = float(np.nansum(delta * widths[:, None, None] * dentro[None, :, :]))

        self.assertAlmostEqual(dentro_flux, 100.0, places=6)
        # Y sobre todo el recorte hay MAS: las alas viven fuera del radio.
        self.assertGreater(float(np.nansum(delta * widths[:, None, None])), 100.0)

    def test_two_models_inject_the_same_core_flux(self):
        """Lo que el cambio compra: la inyeccion deja de depender del modelo.

        Dos PSF muy distintas, el mismo `total_line_flux` dentro del radio. Con
        `frame` no era asi, y esa diferencia era el -6 % espurio.
        """

        wave = np.arange(6558.0, 6568.0, 1.0, dtype=np.float64)
        cube = np.zeros((wave.size, 61, 61), dtype=np.float64)
        widths = np.gradient(wave)
        yy, xx = np.mgrid[0:61, 0:61]
        dentro = np.hypot(yy - 30.0, xx - 30.0) <= 12.0
        source = InjectionSource(
            y=30.0, x=30.0, total_line_flux=100.0,
            line_center_A=6562.8, line_fwhm_A=2.5,
        )

        dentro_flux, frame_flux = [], []
        for fwhm in (3.0, 7.0):
            delta = inject(cube, [source], wavelengths_A=wave,
                           psf_model=self._model(fwhm, 12.0)) - cube
            dentro_flux.append(float(np.nansum(delta * widths[:, None, None] * dentro[None, :, :])))
            frame_flux.append(float(np.nansum(delta * widths[:, None, None])))

        self.assertAlmostEqual(dentro_flux[0], dentro_flux[1], places=6)
        # Y la diferencia que ANTES se colaba en el throughput sigue ahi, medible:
        self.assertGreater(abs(frame_flux[1] / frame_flux[0] - 1.0), 0.05)

    def test_an_unknown_convention_is_refused(self):
        with self.assertRaises(ValueError):
            normalized_spatial_psf((5, 5), 2.0, 2.0, wavelength_A=6562.8,
                                   psf_image=np.ones((5, 5)), norm_convention="lo_que_sea")

    def test_without_a_radius_it_says_so_instead_of_inventing_one(self):
        with self.assertRaises(ValueError) as ctx:
            normalized_spatial_psf((5, 5), 2.0, 2.0, wavelength_A=6562.8,
                                   psf_image=np.ones((5, 5)))
        self.assertIn("norm_radius_px", str(ctx.exception))
