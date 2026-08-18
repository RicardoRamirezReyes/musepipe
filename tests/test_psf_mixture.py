"""La forma `mixture`: la PSF del combinado como media pesada de las de cada exposición.

La razón de que exista está medida: el combinado suma exposiciones con seeing
y calidad de AO distintas, y **la mezcla de N PSF no es una PSF**, así que
ajustarle una Psfao o una Moffat deja la razón núcleo/halo —y con ella la
corrección de apertura— sistemáticamente mal.

Lo que se fija aquí:

* el **contrato** es el de las otras dos formas: normalizada a 1 dentro de
  `norm_radius_px`, para que apcorr, curva de crecimiento, optimal, psffit e
  inyección no tengan que saber que es una mezcla;
* una mezcla de **una** componente es esa componente, bit a bit;
* los pesos son `w_i · F_i(λ)` y no `w_i` sola — una exposición más brillante
  pesa más en el combinado, y eso es cromático;
* lo que rompería la mezcla en silencio (radios de normalización distintos, una
  componente sin flujo) **levanta**.
"""

import unittest

import numpy as np

from musepipe.psf import (
    MIXTURE_FORM,
    build_mixture_model_document,
    evaluate_psf_model,
    psf_roundtrip_error,
    scaled_psf_model,
)

NORM_RADIUS = 25.0
WAVES = (5000.0, 7000.0, 9000.0)


def moffat_doc(fwhm, *, norm_radius=NORM_RADIUS, beta=2.5):
    """Documento Moffat mínimo, con coeficientes constantes en λ."""

    def constant(value):
        # El formato que escribe `smooth_parameter`: coeficientes de menor a
        # mayor grado sobre `(lambda - wave_ref) / wave_scale`.
        return {
            "model": "polynomial",
            "degree": 0,
            "coefficients": [float(value)],
            "wave_ref_A": 7000.0,
            "wave_scale_A": 1000.0,
        }

    return {
        "form": "moffat",
        "norm_radius_px": float(norm_radius),
        "coefficients": {
            "y0": constant(0.0),
            "x0": constant(0.0),
            "fwhm_maj": constant(fwhm),
            "fwhm_min": constant(fwhm),
            "theta_deg": constant(0.0),
            "beta": constant(beta),
        },
    }


def component(exposure_id, model, *, weight=300.0, flux=1.0, waves=(4800.0, 9300.0)):
    return {
        "exposure_id": exposure_id,
        "weight": float(weight),
        "flux_norm": {
            "lambda_A": [float(w) for w in waves],
            "value": [float(flux)] * len(waves),
        },
        "model": model,
    }


def offsets(reach=30):
    yy, xx = np.mgrid[-reach : reach + 1, -reach : reach + 1]
    return yy.astype(float), xx.astype(float)


class MixtureContractTests(unittest.TestCase):
    def setUp(self):
        self.dy, self.dx = offsets()
        self.inside = np.hypot(self.dy, self.dx) <= NORM_RADIUS

    def _mixture(self, components, **extra):
        return build_mixture_model_document(
            components, norm_radius_px=NORM_RADIUS, system="muse_nfm", **extra
        )

    def test_it_is_normalised_inside_norm_radius(self):
        doc = self._mixture(
            [
                component("a", moffat_doc(3.0)),
                component("b", moffat_doc(6.0), weight=600.0),
            ]
        )
        for wave in WAVES:
            psf = evaluate_psf_model(doc, wave, self.dy, self.dx)
            self.assertAlmostEqual(float(np.nansum(psf[self.inside])), 1.0, places=6)

    def test_roundtrip_error_is_negligible(self):
        doc = self._mixture([component("a", moffat_doc(3.5)), component("b", moffat_doc(5.0))])
        self.assertLess(psf_roundtrip_error(doc, WAVES), 1e-6)

    def test_one_component_is_that_component(self):
        model = moffat_doc(4.0)
        doc = self._mixture([component("solo", model)])
        for wave in WAVES:
            np.testing.assert_allclose(
                evaluate_psf_model(doc, wave, self.dy, self.dx),
                evaluate_psf_model(model, wave, self.dy, self.dx),
                rtol=1e-10,
                atol=1e-12,
            )

    def test_the_mixture_is_the_weighted_mean_of_its_components(self):
        narrow, wide = moffat_doc(3.0), moffat_doc(7.0)
        doc = self._mixture(
            [component("a", narrow, weight=100.0), component("b", wide, weight=300.0)]
        )
        expected = (
            1.0 * evaluate_psf_model(narrow, 7000.0, self.dy, self.dx)
            + 3.0 * evaluate_psf_model(wide, 7000.0, self.dy, self.dx)
        ) / 4.0
        np.testing.assert_allclose(
            evaluate_psf_model(doc, 7000.0, self.dy, self.dx), expected, rtol=1e-9, atol=1e-12
        )

    def test_a_mixture_of_different_widths_is_not_any_of_them(self):
        """La razón de ser: el promedio no se parece a ninguna de las dos.

        Es lo que hace que ajustarle UNA forma analítica al combinado sesgue el
        cociente núcleo/total, que es exactamente la `apcorr`.
        """

        narrow, wide = moffat_doc(2.5), moffat_doc(8.0)
        doc = self._mixture([component("a", narrow), component("b", wide)])
        core = np.hypot(self.dy, self.dx) <= 2.0
        mixed = float(np.nansum(evaluate_psf_model(doc, 7000.0, self.dy, self.dx)[core]))
        for single in (narrow, wide):
            fraction = float(np.nansum(evaluate_psf_model(single, 7000.0, self.dy, self.dx)[core]))
            self.assertGreater(abs(mixed - fraction), 0.02)

    def test_the_weights_are_chromatic(self):
        """`w_i·F_i(λ)`: una exposición que se apaga hacia el rojo pesa menos allí."""

        narrow, wide = moffat_doc(3.0), moffat_doc(7.0)
        doc = self._mixture(
            [
                dict(component("a", narrow), flux_norm={"lambda_A": [4800.0, 9300.0], "value": [10.0, 1.0]}),
                component("b", wide, flux=1.0),
            ]
        )
        core = np.hypot(self.dy, self.dx) <= 2.0
        blue = float(np.nansum(evaluate_psf_model(doc, 4800.0, self.dy, self.dx)[core]))
        red = float(np.nansum(evaluate_psf_model(doc, 9300.0, self.dy, self.dx)[core]))
        # En el azul manda la componente estrecha, así que hay más luz en el núcleo.
        self.assertGreater(blue, red)

    def test_fwhm_scaling_dilates_the_whole_mixture(self):
        doc = self._mixture([component("a", moffat_doc(3.0)), component("b", moffat_doc(6.0))])
        scaled = scaled_psf_model(doc, 1.1)
        self.assertEqual(scaled["form"], MIXTURE_FORM)
        self.assertAlmostEqual(scaled["psf_fwhm_scale"], 1.1)
        core = np.hypot(self.dy, self.dx) <= 2.0
        base = float(np.nansum(evaluate_psf_model(doc, 7000.0, self.dy, self.dx)[core]))
        wider = float(np.nansum(evaluate_psf_model(scaled, 7000.0, self.dy, self.dx)[core]))
        self.assertLess(wider, base)
        # Y las componentes NO se tocan: la dilatación conmuta con la suma.
        self.assertEqual(scaled["components"][0]["model"], doc["components"][0]["model"])


def psfao_doc(r0, *, norm_radius=NORM_RADIUS):
    """Documento psfao mínimo, con la forma que escribe C1."""

    names = ["r0", "C", "A", "alpha", "ratio", "theta", "beta"]
    return {
        "form": "psfao",
        "system": "muse_nfm",
        "param_names": names,
        "norm_radius_px": float(norm_radius),
        "psfao_wave_bin_A": 100.0,
        "smoothed_poly": {
            "r0": [float(r0)], "C": [1e-6], "A": [10.0], "alpha": [0.5],
            "ratio": [1.0], "theta": [0.0], "beta": [1.6],
        },
    }


class MixtureOfPsfaoTests(unittest.TestCase):
    """La ruta real: las componentes son psfao, que es lo que C1 ajusta."""

    def setUp(self):
        try:
            import maoppy  # noqa: F401
        except Exception:  # pragma: no cover - entorno sin maoppy
            self.skipTest("maoppy no disponible")
        self.dy, self.dx = offsets()
        self.inside = np.hypot(self.dy, self.dx) <= NORM_RADIUS

    def test_one_psfao_component_is_that_psfao(self):
        model = psfao_doc(0.12)
        doc = build_mixture_model_document(
            [component("solo", model)],
            norm_radius_px=NORM_RADIUS,
            system="muse_nfm",
            wave_bin_A=100.0,
        )
        np.testing.assert_allclose(
            evaluate_psf_model(doc, 6562.8, self.dy, self.dx),
            evaluate_psf_model(model, 6562.8, self.dy, self.dx),
            rtol=1e-10,
            atol=1e-14,
        )

    def test_a_mixture_of_seeings_is_normalised_and_in_between(self):
        good, bad = psfao_doc(0.20), psfao_doc(0.06)
        doc = build_mixture_model_document(
            [component("buena", good), component("mala", bad)],
            norm_radius_px=NORM_RADIUS,
            system="muse_nfm",
            wave_bin_A=100.0,
        )
        psf = evaluate_psf_model(doc, 6562.8, self.dy, self.dx)
        self.assertAlmostEqual(float(np.nansum(psf[self.inside])), 1.0, places=6)
        core = np.hypot(self.dy, self.dx) <= 2.0
        mixed = float(np.nansum(psf[core]))
        fractions = [
            float(np.nansum(evaluate_psf_model(single, 6562.8, self.dy, self.dx)[core]))
            for single in (good, bad)
        ]
        self.assertGreater(mixed, min(fractions))
        self.assertLess(mixed, max(fractions))

    def test_the_sum_is_built_once_per_wavelength_bin(self):
        """La caché de la SUMA: sin ella, una mezcla de N costaría N evaluaciones.

        Con 29 exposiciones eso convertiría el ajuste por canal de C4 en horas,
        así que es una propiedad del diseño y no una optimización opcional. Se
        comprueba contando construcciones y no cronometrando: lo que importa es
        que 80 canales del mismo bin construyan **una** imagen.
        """

        from musepipe.psf import _mixture_image_cached, _psfao_image_cached

        components = [component(f"e{i}", psfao_doc(0.10 + 0.005 * i)) for i in range(8)]
        doc = build_mixture_model_document(
            components, norm_radius_px=NORM_RADIUS, system="muse_nfm", wave_bin_A=100.0
        )
        _mixture_image_cached.cache_clear()
        _psfao_image_cached.cache_clear()

        # Canales que caen todos en el mismo bin de 100 A (a partir de 6050 el
        # redondeo ya los manda al siguiente).
        waves = np.arange(6000.0, 6049.0, 1.25)
        for wave in waves:
            evaluate_psf_model(doc, wave, self.dy, self.dx)

        info = _mixture_image_cached.cache_info()
        self.assertEqual(info.misses, 1)
        self.assertEqual(info.hits, waves.size - 1)
        # Y las componentes NO se quedan en la caché compartida: son 8 aquí y
        # 29 en el run real, a 284x284 cada una.
        self.assertEqual(_psfao_image_cached.cache_info().currsize, 0)


class MixtureValidationTests(unittest.TestCase):
    def test_a_component_with_another_norm_radius_is_refused(self):
        with self.assertRaises(ValueError) as ctx:
            build_mixture_model_document(
                [
                    component("a", moffat_doc(3.0)),
                    component("b", moffat_doc(3.0, norm_radius=40.0)),
                ],
                norm_radius_px=NORM_RADIUS,
                system="muse_nfm",
            )
        self.assertIn("normalised within", str(ctx.exception))

    def test_a_component_without_flux_is_refused(self):
        broken = component("a", moffat_doc(3.0))
        broken.pop("flux_norm")
        with self.assertRaises(ValueError) as ctx:
            build_mixture_model_document(
                [broken], norm_radius_px=NORM_RADIUS, system="muse_nfm"
            )
        self.assertIn("flux_norm", str(ctx.exception))

    def test_an_empty_mixture_is_refused(self):
        with self.assertRaises(ValueError):
            build_mixture_model_document([], norm_radius_px=NORM_RADIUS, system="muse_nfm")

    def test_an_unknown_component_form_is_refused(self):
        doc = build_mixture_model_document(
            [component("a", moffat_doc(3.0))], norm_radius_px=NORM_RADIUS, system="muse_nfm"
        )
        doc["components"][0]["model"]["form"] = "gaussian"
        dy, dx = offsets(5)
        with self.assertRaises(ValueError):
            evaluate_psf_model(doc, 7000.0, dy, dx)


if __name__ == "__main__":
    unittest.main()
