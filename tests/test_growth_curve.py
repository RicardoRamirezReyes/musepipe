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


class CollapsedFitGuardTests(unittest.TestCase):
    """Un ajuste colapsado tiene que rechazarse, no adoptarse en silencio.

    Sobre el cubo combinado de 200 px el ajuste halo+cielo se degeneraba: `p`
    bajaba a 2.2-2.6, el suelo huia a -2.6 y hasta el 45% del "total" era cola
    extrapolada. El factor resultante NO era monotono en lambda (maximo interior
    hacia 7400 A) y deformaba la pendiente del continuo de todo el bloque C un
    ~47%. Nada avisaba: `MIN_HALO_POWER` valia 2.05 y no habia limite ni para la
    cola ni para la dispersion entre rangos de ajuste.
    """

    #: Las 8 bandas REALES que midio `ROXs12b_realigned` sobre el cubo de 200 px.
    BANDAS_ROTAS = [2.178, 1.894, 1.823, 2.031, 2.416, 2.473, 2.283, 1.694]
    LAMBDAS = [5037, 5612, 6187, 6762, 7337, 7912, 8487, 9062]

    def test_the_real_broken_bands_are_refused_by_the_monotonic_check(self):
        qc = {"bands": [{"wave_A": w, "ratio_total_over_normrad": v}
                        for w, v in zip(self.LAMBDAS, self.BANDAS_ROTAS)]}
        with self.assertRaises(RuntimeError) as ctx:
            factor_at_wavelengths(qc, np.linspace(4800.0, 9300.0, 200))
        mensaje = str(ctx.exception)
        self.assertIn("not monotonic", mensaje)
        self.assertIn("measure_growth_curve", mensaje)   # dice cómo arreglarlo

    def test_a_physical_decreasing_curve_still_passes(self):
        qc = {"bands": [{"wave_A": w, "ratio_total_over_normrad": v}
                        for w, v in zip(self.LAMBDAS,
                                        [2.6, 2.45, 2.32, 2.21, 2.12, 2.05, 2.00, 1.97])]}
        got = factor_at_wavelengths(qc, np.linspace(4800.0, 9300.0, 200))
        self.assertTrue(np.all(np.diff(got) < 0))

    def test_the_check_can_be_switched_off_for_diagnostics(self):
        """El notebook de análisis necesita PODER dibujar la curva rota."""
        qc = {"bands": [{"wave_A": w, "ratio_total_over_normrad": v}
                        for w, v in zip(self.LAMBDAS, self.BANDAS_ROTAS)]}
        got = factor_at_wavelengths(qc, np.linspace(4800.0, 9300.0, 50),
                                    require_monotonic=False)
        self.assertEqual(got.size, 50)

    def test_the_power_floor_is_physical(self):
        """2.05 no rechazaba nada: lo medido en el colapso llegaba a 2.20."""
        from musepipe.growth_curve import MIN_HALO_POWER

        self.assertGreaterEqual(MIN_HALO_POWER, 2.5)
        self.assertLess(MIN_HALO_POWER, 3.1)   # por debajo de lo medido en campo grande

    def test_an_extrapolated_tail_is_capped(self):
        from musepipe.growth_curve import MAX_TAIL_FRACTION

        self.assertLessEqual(MAX_TAIL_FRACTION, 0.25)
        # El caso real que hay que cortar.
        self.assertGreater(0.455, MAX_TAIL_FRACTION)

    def test_a_small_field_is_refused_instead_of_returning_a_factor(self):
        """El campo recortado es la causa raíz: sin brazo, no hay medida."""
        cube, wave, _ = _synthetic_cube(n=121, power=3.2, sky=-0.4)
        completo = measure_growth_curve(cube, wave, n_bands=3)
        self.assertTrue(completo["bands"])
        recortado = cube[:, 30:91, 30:91]      # la mitad de campo
        with self.assertRaises(RuntimeError) as ctx:
            measure_growth_curve(recortado, wave, n_bands=3)
        self.assertIn("field is too small", str(ctx.exception))


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


class InteriorDipTests(unittest.TestCase):
    """El repunte rojo que `interior_bump` no ve.

    `interior_bump` mide el maximo interior, asi que una curva que baja, toca
    fondo dentro del rango y vuelve a subir le da 0.0 — y el diagnostico de D2
    la declaraba «monotona». La `apcorr` de `ROXs12b_realigned` es exactamente
    ese caso: minimo en 7978 A y repunte del 3.15% hasta 9350 A.
    """

    def test_a_monotone_falling_curve_has_no_dip(self):
        from musepipe.growth_curve import interior_dip

        self.assertEqual(interior_dip([2.0, 1.8, 1.6, 1.5]), 0.0)

    def test_a_curve_that_bottoms_out_and_climbs_back_is_measured(self):
        from musepipe.growth_curve import interior_dip

        self.assertAlmostEqual(interior_dip([2.0, 1.5, 1.0, 1.1, 1.2]), 0.2, places=9)

    def test_interior_bump_is_blind_to_it(self):
        # La razon exacta por la que hace falta la funcion nueva.
        from musepipe.growth_curve import interior_bump, interior_dip

        curva = [2.0, 1.5, 1.0, 1.1, 1.2]
        self.assertEqual(interior_bump(curva), 0.0)
        self.assertGreater(interior_dip(curva), 0.1)

    def test_only_the_climb_after_the_minimum_counts(self):
        # La bajada previa es lo que la curva DEBE hacer: no es el defecto.
        from musepipe.growth_curve import interior_dip

        self.assertAlmostEqual(interior_dip([5.0, 1.0, 1.5]), 0.5, places=9)

    def test_too_few_points_is_not_an_error(self):
        from musepipe.growth_curve import interior_dip

        self.assertEqual(interior_dip([1.0, 2.0]), 0.0)


class PolynomialMisfitTests(unittest.TestCase):
    """Cuanto se aparta la parabola de las bandas que se midieron."""

    def test_bands_on_a_parabola_are_fitted_exactly(self):
        from musepipe.growth_curve import polynomial_misfit

        w = np.linspace(5000.0, 9000.0, 8)
        y = 1.5 + 1e-9 * (w - 7000.0) ** 2
        qc = {"bands": [{"wave_A": a, "ratio_total_over_normrad": b} for a, b in zip(w, y)]}
        rms, pp = polynomial_misfit(qc)
        self.assertLess(rms, 1e-9)
        self.assertLess(pp, 1e-9)

    def test_a_curve_that_flattens_is_not_a_parabola(self):
        # La forma real: cae deprisa y se aplana. La parabola le pone vertice.
        from musepipe.growth_curve import polynomial_misfit

        w = np.linspace(5000.0, 9000.0, 8)
        y = np.array([1.684, 1.566, 1.506, 1.488, 1.472, 1.466, 1.460, 1.465])
        qc = {"bands": [{"wave_A": a, "ratio_total_over_normrad": b} for a, b in zip(w, y)]}
        rms, pp = polynomial_misfit(qc)
        self.assertGreater(rms, 0.005)
        self.assertGreater(pp, 0.02)

    def test_too_few_bands_returns_none_instead_of_a_fake_number(self):
        from musepipe.growth_curve import polynomial_misfit

        qc = {"bands": [{"wave_A": 5000.0, "ratio_total_over_normrad": 1.6},
                        {"wave_A": 7000.0, "ratio_total_over_normrad": 1.5},
                        {"wave_A": 9000.0, "ratio_total_over_normrad": 1.4}]}
        self.assertEqual(polynomial_misfit(qc), (None, None))


class MonotoneInterpolationTests(unittest.TestCase):
    """El interpolador de la apcorr, cambiado a PCHIP el 2026-08-11.

    La parabola tiene curvatura constante: no puede bajar deprisa y luego
    aplanarse, que es lo que hace esta curva. Sobre las bandas reales de
    `ROXs12b_realigned` le ponia el vertice DENTRO del rango, en 7978 A, y
    repuntaba un 3.15% hasta 9350 A. Como la apcorr multiplica, el continuo
    rojo salia sobre-corregido en los seis metodos del bloque C a la vez.
    """

    #: Las 8 bandas REALES de `ROXs12b_realigned` (cubo de 256 px, las buenas):
    #: caen 13% y se aplanan, con el ultimo punto subiendo un 0.37%.
    LAMBDAS = [5037.0, 5612.0, 6187.0, 6762.0, 7337.0, 7912.0, 8487.0, 9062.0]
    BANDAS = [1.6844, 1.5662, 1.5063, 1.4882, 1.4717, 1.4662, 1.4599, 1.4653]

    def setUp(self):
        self.qc = {"bands": [{"wave_A": w, "ratio_total_over_normrad": v}
                             for w, v in zip(self.LAMBDAS, self.BANDAS)]}
        self.wave = np.linspace(4750.0, 9350.0, 400)

    def test_pchip_passes_through_the_measured_bands_and_the_parabola_does_not(self):
        x = np.asarray(self.LAMBDAS)
        y = np.asarray(self.BANDAS)
        np.testing.assert_allclose(factor_at_wavelengths(self.qc, x), y, rtol=0, atol=0)
        poly = factor_at_wavelengths(self.qc, x, method="poly2", require_monotonic=False)
        self.assertGreater(float(np.max(np.abs(poly / y - 1.0))), 0.01)

    def test_the_parabola_invents_a_red_rebound_that_the_bands_do_not_have(self):
        from musepipe.growth_curve import interior_dip

        poly = factor_at_wavelengths(self.qc, self.wave, method="poly2",
                                     require_monotonic=False)
        pchip = factor_at_wavelengths(self.qc, self.wave)
        self.assertGreater(interior_dip(poly), 0.03)      # 3.15%: lo fabrica el ajuste
        self.assertLess(interior_dip(pchip), 0.01)        # 0.75%: lo que dicen las bandas
        # Y el repunte del dato es mucho menor que el que fabricaba la parabola.
        self.assertLess(self.BANDAS[-1] / self.BANDAS[-2] - 1.0, 0.005)

    def test_the_guard_now_refuses_the_parabola_and_says_who_manufactured_it(self):
        with self.assertRaises(RuntimeError) as ctx:
            factor_at_wavelengths(self.qc, self.wave, method="poly2")
        mensaje = str(ctx.exception)
        self.assertIn("bottoms out", mensaje)
        self.assertIn("pchip", mensaje)                   # dice como arreglarlo

    def test_pchip_on_those_same_bands_passes_the_guard(self):
        got = factor_at_wavelengths(self.qc, self.wave)   # require_monotonic por defecto
        self.assertEqual(got.size, self.wave.size)

    def test_extrapolation_outside_the_bands_is_a_straight_line_that_cannot_turn_around(self):
        """El eje llega a 4750 A y la banda mas azul esta en 5037: se extrapola."""
        bajando = {"bands": [{"wave_A": w, "ratio_total_over_normrad": v}
                             for w, v in zip(self.LAMBDAS,
                                             [2.6, 2.45, 2.32, 2.21, 2.12, 2.05, 2.00, 1.97])]}
        got = factor_at_wavelengths(bajando, self.wave)
        self.assertTrue(np.all(np.diff(got) < 0))
        azul = self.wave < self.LAMBDAS[0]
        paso = np.diff(got[azul])
        np.testing.assert_allclose(paso, paso[0], rtol=1e-9)   # recta, no cubica

    def test_poly2_still_reproduces_the_frozen_path_bit_for_bit(self):
        """Los runs congelados se reproducen pidiendo el metodo viejo."""
        x = np.asarray(self.LAMBDAS)
        y = np.asarray(self.BANDAS)
        esperado = np.polyval(np.polyfit(x, y, 2), self.wave)
        got = factor_at_wavelengths(self.qc, self.wave, method="poly2",
                                    require_monotonic=False)
        np.testing.assert_array_equal(got, esperado)

    def test_bands_out_of_order_do_not_break_the_spline(self):
        revuelto = {"bands": list(reversed(self.qc["bands"]))}
        np.testing.assert_allclose(factor_at_wavelengths(revuelto, self.wave),
                                   factor_at_wavelengths(self.qc, self.wave))

    def test_an_unknown_method_raises(self):
        with self.assertRaises(ValueError):
            factor_at_wavelengths(self.qc, self.wave, method="spline37")
