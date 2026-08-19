"""C1 por observación: el ajuste de cada exposición y la mezcla que entrega.

Lo que se fija aquí es la **geometría** y la **contabilidad**, que es donde un
error no se ve en ningún número final pero corrompe todo lo que viene detrás:

* que la exposición se lea en la MISMA rejilla en la que se combinó (ventana y
  desplazamiento del plan), con la primaria en el centro del array — que es
  donde `fit_bin` la da por hecha;
* que el compañero se sitúe por desplazamiento respecto del centro, como hace la
  cadena, y no por una posición absoluta que sólo vale en el cubo combinado;
* que el peso de cada exposición en la mezcla sea `w_i·F_i(λ)` con `F_i` medido
  sobre el dato;
* que una exposición que reviente se cuente y no tumbe a las otras 28.
"""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from musepipe.psf import evaluate_psf_model
from musepipe.reduction.stream_combine import build_stream_combine_plan
from musepipe.stages.stage_e01_perobs import (
    ObservationFit,
    PerObservationError,
    _flux_norm_table,
    _positions_in_window,
    fit_all_observations,
    load_aligned_exposure,
    mixture_from_fits,
)
from test_observations import _exposures, _write_plan, _write_run
from test_psf_mixture import moffat_doc

CROP = 20
PAD = 4


def _plan(root, centers):
    cubes = _exposures(root, centers)
    _write_run(root, "obj", {"cube_files": [str(root / "combined.fits")]})
    return cubes, _write_plan(root, "obj", cubes)


class GeometryTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.centers = [(30.2, 31.4), (28.6, 33.9)]
        self.cubes, self.plan = _plan(self.root, self.centers)

    def tearDown(self):
        self._tmp.cleanup()

    def test_the_exposure_is_read_on_the_combine_grid(self):
        cube, stat, wave = load_aligned_exposure(self.plan.exposures[0], self.plan)

        self.assertEqual(cube.shape, (int(self.plan.wavelength["n_channels"]), CROP, CROP))
        self.assertEqual(stat.shape, cube.shape)
        self.assertEqual(wave.size, cube.shape[0])
        # La estrella queda en el centro del array: es lo que `fit_bin` supone.
        plane = np.nanmedian(cube, axis=0)
        peak = np.unravel_index(int(np.nanargmax(plane)), plane.shape)
        self.assertEqual(peak, (CROP // 2, CROP // 2))

    def test_every_exposure_lands_on_the_same_pixel(self):
        """El alineado es el del combinado: dos punterías distintas, un centro."""

        peaks = []
        for exposure in self.plan.exposures:
            cube, _stat, _wave = load_aligned_exposure(exposure, self.plan)
            plane = np.nanmedian(cube, axis=0)
            peaks.append(np.unravel_index(int(np.nanargmax(plane)), plane.shape))
        self.assertEqual(len(set(peaks)), 1)

    def test_the_companion_is_placed_by_offset_from_the_centre(self):
        # Compañero 10 px arriba y 4 a la izquierda del centro del cubo de la cadena.
        frame = (170, 170)
        positions = {"companion": (85 + 10.0, 85 - 4.0), "field": None}
        center, moved = _positions_in_window(CROP, positions, frame)

        self.assertEqual(center, (CROP // 2, CROP // 2))
        self.assertAlmostEqual(moved["companion"][0], CROP // 2 + 10.0)
        self.assertAlmostEqual(moved["companion"][1], CROP // 2 - 4.0)
        self.assertIsNone(moved["field"])

    def test_the_flux_table_measures_the_star_on_the_data(self):
        cube, _stat, _wave = load_aligned_exposure(self.plan.exposures[0], self.plan)
        images = [np.nanmedian(cube, axis=0)]
        table = _flux_norm_table(images, [6200.0], (CROP // 2, CROP // 2), 6.0, None)

        self.assertEqual(table["lambda_A"], [6200.0])
        self.assertGreater(table["value"][0], 0.0)
        # Y es flujo de verdad: sube con el radio.
        wider = _flux_norm_table(images, [6200.0], (CROP // 2, CROP // 2), 9.0, None)
        self.assertGreater(wider["value"][0], table["value"][0])


class MixtureAssemblyTests(unittest.TestCase):
    """La mezcla sale de los ajustes, con sus pesos y sin re-ajustar nada."""

    def _fit(self, exposure_id, fwhm, weight, flux):
        return ObservationFit(
            exposure_id=exposure_id,
            weight=weight,
            form="moffat",
            model=moffat_doc(fwhm),
            flux_norm={"lambda_A": [4800.0, 9300.0], "value": [flux, flux]},
            summary={"exposure_id": exposure_id, "form_chosen": "moffat"},
        )

    def test_the_components_carry_weight_and_flux(self):
        fits = [self._fit("a", 3.0, 300.0, 10.0), self._fit("b", 6.0, 600.0, 5.0)]
        mixture = mixture_from_fits(fits, {"psf_norm_radius_px": 25.0, "psf_bin_A": 100.0})

        self.assertEqual(mixture["form"], "mixture")
        self.assertEqual(mixture["n_components"], 2)
        self.assertEqual([c["exposure_id"] for c in mixture["components"]], ["a", "b"])
        self.assertEqual([c["weight"] for c in mixture["components"]], [300.0, 600.0])
        self.assertEqual(mixture["forms"], {"a": "moffat", "b": "moffat"})

    def test_the_mixture_weights_are_weight_times_flux(self):
        """`w·F`: la exposición larga y brillante manda sobre la corta y débil."""

        heavy = self._fit("larga", 3.0, weight=600.0, flux=10.0)   # w*F = 6000
        light = self._fit("corta", 9.0, weight=300.0, flux=1.0)    # w*F =  300
        mixture = mixture_from_fits([heavy, light], {"psf_norm_radius_px": 25.0})

        n = 61
        yy, xx = np.indices((n, n), dtype=float)
        dy, dx = yy - n // 2, xx - n // 2
        got = evaluate_psf_model(mixture, 7000.0, dy, dx)
        expected = (
            6000.0 * evaluate_psf_model(heavy.model, 7000.0, dy, dx)
            + 300.0 * evaluate_psf_model(light.model, 7000.0, dy, dx)
        ) / 6300.0
        np.testing.assert_allclose(got, expected, rtol=1e-9, atol=1e-12)

    def test_no_fits_is_an_error(self):
        with self.assertRaises(PerObservationError):
            mixture_from_fits([], {"psf_norm_radius_px": 25.0})


class FailureAccountingTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.cubes, self.plan = _plan(self.root, [(30.2, 31.4), (28.6, 33.9), (31.0, 30.0)])

    def tearDown(self):
        self._tmp.cleanup()

    def test_a_broken_exposure_is_counted_not_fatal(self):
        from musepipe.observations import ObservationSet

        observations = ObservationSet(plan=self.plan, source="test")
        # El segundo cubo se rompe: el ajuste tiene que seguir con los otros dos.
        Path(self.plan.exposures[1].file).write_bytes(b"no soy un FITS")

        cfg = {
            "psf_bin_A": 200.0,
            "psf_min_channels_per_bin": 3,
            "psf_fit_radius_px": 8.0,
            "psf_norm_radius_px": 6.0,
            "psf_companion_mask_radius_px": 3.0,
            "psf_prelim_fwhm_px": 2.0,
            "psf_warm_start": False,
        }
        fits_ok, failures = fit_all_observations(
            observations, cfg, {"companion": (95.0, 90.0), "field": None}, (170, 170), n_jobs=1
        )

        self.assertEqual(len(fits_ok), 2)
        self.assertEqual(len(failures), 1)
        self.assertEqual(failures[0]["exposure_id"], self.plan.exposures[1].exposure_id)
        self.assertTrue(failures[0]["error"])

    def test_all_broken_is_an_error_with_the_reason(self):
        from musepipe.observations import ObservationSet

        observations = ObservationSet(plan=self.plan, source="test")
        for exposure in self.plan.exposures:
            Path(exposure.file).write_bytes(b"no soy un FITS")

        with self.assertRaises(PerObservationError) as ctx:
            fit_all_observations(
                observations, {}, {"companion": (95.0, 90.0), "field": None}, (170, 170), n_jobs=1
            )
        self.assertIn("ninguna exposición", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
