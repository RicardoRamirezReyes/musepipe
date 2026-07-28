"""Curva de crecimiento empirica: que mida, que no se aplique sola, y que
declare la convencion de flujo que usa.

El fallo que motiva estas pruebas no era un crash: `apcorr` normalizaba la PSF a
1 dentro de `norm_radius_px` y llamaba a eso "flujo total", cuando fuera de ese
radio queda ~la mitad de la luz y el factor que falta es cromatico (~2.5 azul,
~1.9 rojo). Nada en el codigo lo decia y nada lo medía.
"""

import unittest
from pathlib import Path

import numpy as np

from musepipe.extraction.aperture import aperture_correction_from_psf
from musepipe.growth_curve import (
    DEFAULT_NORM_RADIUS_PX,
    RUN_PRODUCT_NAME,
    resolve_flux_convention,
    factor_at_wavelengths,
    integrate_total,
    locate_brightest,
    measure_growth_curve,
    radial_profile,
)


def _moffat_doc(fwhm=4.0, norm_radius=8.0):
    def poly(value):
        return {"model": "polynomial", "degree": 0, "coefficients": [float(value)],
                "wave_ref_A": 7000.0, "wave_scale_A": 1000.0}

    return {
        "form": "moffat",
        "norm_radius_px": float(norm_radius),
        "hybrid": False,
        "coefficients": {
            "fwhm_maj": poly(fwhm), "fwhm_min": poly(fwhm), "theta_deg": poly(0.0),
            "beta": poly(2.5), "y0": poly(0.0), "x0": poly(0.0),
        },
    }


def _synthetic_cube(nz=24, n=121, power=3.2, sky=-0.4, amp=4.0e4):
    """Cubo con un halo en ley de potencias conocida sobre un suelo NEGATIVO.

    El suelo negativo es el caso real: el DRS sobre-resta cielo, y suponerlo
    cero infla el flujo total porque multiplica por decenas de miles de spaxels.
    """

    yy, xx = np.indices((n, n), dtype=float)
    r = np.hypot(yy - n // 2, xx - n // 2)
    r = np.where(r < 1.0, 1.0, r)
    image = amp * r ** (-power) + sky
    wave = np.linspace(5000.0, 9000.0, nz)
    return np.repeat(image[None, :, :], nz, axis=0), wave, r


class GrowthCurveTests(unittest.TestCase):
    def test_it_recovers_the_injected_power_law_and_negative_sky(self):
        cube, wave, _ = _synthetic_cube(power=3.2, sky=-0.4)
        out = measure_growth_curve(cube, wave, n_bands=4, norm_radius_px=8.0)
        for band in out["bands"]:
            self.assertAlmostEqual(band["halo_power"], 3.2, delta=0.15)
            self.assertAlmostEqual(band["sky_floor"], -0.4, delta=0.05)
            # Fuera del radio de normalizacion queda luz: el factor > 1.
            self.assertGreater(band["ratio_total_over_normrad"], 1.0)

    def test_it_reports_the_fit_range_spread_as_the_systematic(self):
        cube, wave, _ = _synthetic_cube()
        out = measure_growth_curve(cube, wave, n_bands=3, norm_radius_px=8.0)
        self.assertIn("systematic_spread_pct_max", out)
        for band in out["bands"]:
            self.assertLessEqual(band["ratio_min"], band["ratio_total_over_normrad"] + 1e-9)
            self.assertGreaterEqual(band["ratio_max"], band["ratio_total_over_normrad"] - 1e-9)

    def test_it_refuses_to_extrapolate_a_divergent_halo(self):
        """p <= 2 hace divergente la integral: hay que fallar, no inventar cola."""

        with self.assertRaises(ValueError):
            integrate_total(np.ones((5, 5)), np.ones((5, 5)), 1.0, 1.8, 0.0, 2.0)

    def test_locate_brightest_finds_the_peak(self):
        img = np.zeros((41, 41))
        img[25, 17] = 10.0
        cy, cx = locate_brightest(img)
        self.assertAlmostEqual(cy, 25.0, delta=0.5)
        self.assertAlmostEqual(cx, 17.0, delta=0.5)

    def test_radial_profile_is_a_median_so_a_companion_does_not_move_it(self):
        n = 81
        yy, xx = np.indices((n, n), dtype=float)
        r = np.hypot(yy - 40, xx - 40)
        img = np.ones((n, n))
        clean = radial_profile(img, r, np.arange(0, 41, 1.0))
        img[40 + 20, 40] = 1e6  # una fuente puntual sobre el anillo r=20
        dirty = radial_profile(img, r, np.arange(0, 41, 1.0))
        np.testing.assert_allclose(clean, dirty, equal_nan=True)

    def test_factor_interpolation_is_smooth_and_bracketed(self):
        qc = {"bands": [{"wave_A": w, "ratio_total_over_normrad": v}
                        for w, v in zip([5000, 6000, 7000, 8000, 9000],
                                        [2.5, 2.3, 2.1, 2.0, 1.9])]}
        got = factor_at_wavelengths(qc, np.array([5000.0, 7000.0, 9000.0]))
        self.assertTrue(np.all(np.diff(got) < 0))       # decreciente con lambda
        self.assertTrue(np.all((got > 1.5) & (got < 3.0)))


class ApcorrConventionTests(unittest.TestCase):
    """A2 mide; C2 decide. Sin `growth_curve` nada se mueve."""

    def setUp(self):
        self.wave = np.array([5000.0, 7000.0, 9000.0])
        self.aperture = {"kind": "box", "size": 3}
        self.model = _moffat_doc()
        self.qc = {"bands": [{"wave_A": w, "ratio_total_over_normrad": v}
                             for w, v in zip([5000, 6000, 7000, 8000, 9000],
                                             [2.5, 2.3, 2.1, 2.0, 1.9])]}

    def test_default_is_unchanged(self):
        a, mode, _ = aperture_correction_from_psf(self.wave, self.aperture, self.model)
        self.assertEqual(mode, "psf_growth_curve")
        self.assertTrue(np.all(np.isfinite(a)))

    def test_growth_curve_multiplies_and_renames_the_mode(self):
        base, _, _ = aperture_correction_from_psf(self.wave, self.aperture, self.model)
        got, mode, _ = aperture_correction_from_psf(
            self.wave, self.aperture, self.model, growth_curve=self.qc
        )
        self.assertEqual(mode, "psf_growth_curve+empirical_total")
        np.testing.assert_allclose(got / base, factor_at_wavelengths(self.qc, self.wave), rtol=1e-12)
        # Cromatico: el factor NO se cancela entre azul y rojo.
        self.assertGreater((got / base)[0], (got / base)[-1])

    def test_a_degenerate_growth_curve_raises_instead_of_silently_scaling(self):
        bad = {"bands": [{"wave_A": w, "ratio_total_over_normrad": v}
                         for w, v in zip([5000, 6000, 7000], [0.0, -1.0, -2.0])]}
        with self.assertRaises(RuntimeError):
            aperture_correction_from_psf(self.wave, self.aperture, self.model, growth_curve=bad)

    def test_norm_radius_default_matches_the_frozen_convention(self):
        self.assertEqual(DEFAULT_NORM_RADIUS_PX, 25.0)


class FluxConventionResolutionTests(unittest.TestCase):
    """El interruptor: por defecto no se mueve nada, y pedir la convencion
    nueva sin medida FALLA en vez de caer en silencio a la vieja."""

    def setUp(self):
        import json
        import tempfile
        self._tmp = tempfile.TemporaryDirectory()
        self.stage_dir = Path(self._tmp.name)
        (self.stage_dir / RUN_PRODUCT_NAME).write_text(json.dumps({
            "bands": [{"wave_A": w, "ratio_total_over_normrad": v}
                      for w, v in zip([5000, 6000, 7000, 8000, 9000],
                                      [2.5, 2.3, 2.1, 2.0, 1.9])]
        }), encoding="utf-8")

    def tearDown(self):
        self._tmp.cleanup()

    def test_default_is_the_historical_convention(self):
        doc, convention = resolve_flux_convention({}, self.stage_dir)
        self.assertIsNone(doc)
        self.assertEqual(convention, "normrad")

    def test_total_loads_the_run_product(self):
        doc, convention = resolve_flux_convention({"flux_convention": "total"}, self.stage_dir)
        self.assertEqual(convention, "total")
        self.assertEqual(len(doc["bands"]), 5)

    def test_a_per_stage_knob_overrides_the_generic_one(self):
        cfg = {"flux_convention": "normrad", "x01_flux_convention": "total"}
        _, convention = resolve_flux_convention(cfg, self.stage_dir, knob="x01_flux_convention")
        self.assertEqual(convention, "total")

    def test_total_without_a_measurement_raises(self):
        empty = self.stage_dir / "vacio"
        empty.mkdir()
        with self.assertRaises(RuntimeError) as ctx:
            resolve_flux_convention({"flux_convention": "total"}, empty)
        self.assertIn("measure_growth_curve", str(ctx.exception))

    def test_an_unknown_convention_raises(self):
        with self.assertRaises(ValueError):
            resolve_flux_convention({"flux_convention": "raro"}, self.stage_dir)


if __name__ == "__main__":
    unittest.main()
