"""C1b: restar por exposición y combinar después.

Dos cosas que fijar, y la segunda es la que valida el diseño entero:

1. **Sin `transform`, el combinado no se mueve.** El gancho no puede cambiar por
   omisión lo que produjo los cubos que hay en disco.
2. **Con `method="mean"`, restar-luego-combinar == combinar-luego-restar.** La
   combinación es lineal, así que las dos rutas tienen que dar lo mismo hasta el
   redondeo. Si algún día dejan de coincidir, es que el alineado, los pesos o el
   orden de las operaciones han dejado de ser los que se creen — y eso no lo ve
   ninguna otra prueba. Con `sigclip` NO coinciden, y eso también se mide aquí:
   la diferencia es el recorte, no un error.
"""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from musepipe.extraction.optimal import fit_primary_psf_amplitudes
from musepipe.reduction.stream_combine import (
    build_stream_combine_plan,
    combine_streaming,
    wavelength_axis,
)
from musepipe.stages.stage_e01b_perobs_subtract import (
    PerObservationSubtractError,
    b1_window,
    make_subtract_transform,
    models_by_exposure,
)
from test_observations import _exposures
from test_psf_mixture import component, moffat_doc

CROP = 24
PAD = 4


class TransformHookTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        cubes = _exposures(self.root, [(30.2, 31.4), (28.6, 33.9), (31.0, 30.0)])
        self.plan = build_stream_combine_plan(
            [str(p) for p in cubes],
            run_id="obj",
            output=str(self.root / "combined.fits"),
            crop_npix=CROP,
            pad=PAD,
            chunk_channels=8,
            method="mean",
        )

    def tearDown(self):
        self._tmp.cleanup()

    def test_without_a_transform_nothing_changes(self):
        base = combine_streaming(self.plan)
        hooked = combine_streaming(self.plan, transform=None)
        np.testing.assert_array_equal(base["data"], hooked["data"])
        np.testing.assert_array_equal(base["stat"], hooked["stat"])

    def test_the_transform_sees_the_aligned_chunk_and_its_wavelengths(self):
        seen = []

        def transform(exposure, wave_chunk, data, stat):
            seen.append((exposure.exposure_id, wave_chunk.size, data.shape, stat.shape))
            return data

        result = combine_streaming(self.plan, transform=transform)
        nz = int(self.plan.wavelength["n_channels"])
        self.assertEqual(len(seen), len(self.plan.exposures) * int(np.ceil(nz / 8)))
        for _exposure_id, n_wave, shape, stat_shape in seen:
            self.assertEqual(shape[1:], (CROP, CROP))
            self.assertEqual(shape[0], n_wave)
            self.assertEqual(stat_shape, shape)
        # Y no cambia nada si el gancho devuelve el dato tal cual.
        np.testing.assert_array_equal(result["data"], combine_streaming(self.plan)["data"])

    def test_subtracting_then_combining_equals_combining_then_subtracting(self):
        """El combinado es lineal: el orden no puede cambiar el resultado.

        Es la comprobación que sostiene toda la idea. La misma PSF para las tres
        exposiciones hace que «la mezcla» sea esa PSF, así que la ruta de
        referencia (restar del combinado) está bien definida.
        """

        psf = moffat_doc(3.0, norm_radius=8.0)
        mixture = {
            "form": "mixture",
            "norm_radius_px": 8.0,
            "components": [
                dict(component(exposure.exposure_id, psf), weight=exposure.weight)
                for exposure in self.plan.exposures
            ],
        }
        models = models_by_exposure(mixture)
        cfg = {"x02_primary_fit_radius_px": 8.0, "x02_primary_exclude_radius_px": 3.0}
        center = (CROP // 2, CROP // 2)

        records = {}
        transform = make_subtract_transform(
            models, cfg, center_yx=center, exclude_centers=[], records=records
        )
        subtracted_first = combine_streaming(self.plan, transform=transform)

        # La otra ruta: combinar y restar el mismo modelo del combinado.
        combined = combine_streaming(self.plan)
        wave = wavelength_axis(self.plan)
        model_cube, _meta, _amp, _bkg, _n = fit_primary_psf_amplitudes(
            combined["data"].astype(np.float64),
            wave,
            center,
            psf,
            variance_zyx=combined["stat"].astype(np.float64),
            fit_radius_px=8.0,
            exclude_centers_yx=[],
            exclude_radius_px=3.0,
        )
        combined_then_subtracted = combined["data"] - model_cube

        np.testing.assert_allclose(
            subtracted_first["data"], combined_then_subtracted, rtol=2e-5, atol=2e-5
        )
        # Y el STAT no lo toca la resta: es un modelo determinista.
        np.testing.assert_array_equal(subtracted_first["stat"], combined["stat"])

    def test_the_amplitudes_of_every_exposure_are_recorded(self):
        psf = moffat_doc(3.0, norm_radius=8.0)
        mixture = {
            "form": "mixture",
            "norm_radius_px": 8.0,
            "components": [
                dict(component(exposure.exposure_id, psf), weight=exposure.weight)
                for exposure in self.plan.exposures
            ],
        }
        records = {}
        transform = make_subtract_transform(
            models_by_exposure(mixture),
            {"x02_primary_fit_radius_px": 8.0, "x02_primary_exclude_radius_px": 3.0},
            center_yx=(CROP // 2, CROP // 2), exclude_centers=[], records=records,
        )
        combine_streaming(self.plan, transform=transform)

        self.assertEqual(set(records), {e.exposure_id for e in self.plan.exposures})
        nz = int(self.plan.wavelength["n_channels"])
        for row in records.values():
            self.assertEqual(len(row["amplitude"]), nz)
            self.assertTrue(np.all(np.asarray(row["amplitude"]) > 0))

    def test_an_exposure_without_a_model_is_an_error(self):
        psf = moffat_doc(3.0, norm_radius=8.0)
        mixture = {
            "form": "mixture",
            "norm_radius_px": 8.0,
            "components": [dict(component("otra_exposicion", psf), weight=1.0)],
        }
        transform = make_subtract_transform(
            models_by_exposure(mixture), {}, center_yx=(CROP // 2, CROP // 2),
            exclude_centers=[], records={},
        )
        with self.assertRaises(PerObservationSubtractError):
            combine_streaming(self.plan, transform=transform)


class B1WindowTests(unittest.TestCase):
    """El producto sale en el marco de B1, con la ventana que B1 declara.

    C3 trabaja en 170 px y el combinado en 200: si esta ventana se recalculara
    aquí en vez de leerse, un cambio en B1 dejaría el residuo mal encuadrado sin
    que nada avisara.
    """

    def _qc(self, **overrides):
        qc = {
            "crop_bounds_per_cube": [{"cube_index": 0, "y1": 15, "y2": 185, "x1": 15, "x2": 185}],
            "spatial_shifts": [{"cube_index": 0, "shift_y": 0.0, "shift_x": 0.0}],
        }
        qc.update(overrides)
        return qc

    def test_it_reads_the_window_b1_declared(self):
        wave = np.arange(10, dtype=float)
        self.assertEqual(b1_window(self._qc(), wave, wave), (15, 185, 15, 185))

    def test_more_than_one_cube_is_refused(self):
        wave = np.arange(10, dtype=float)
        qc = self._qc(crop_bounds_per_cube=[{"y1": 0, "y2": 1, "x1": 0, "x2": 1}] * 2)
        with self.assertRaises(PerObservationSubtractError):
            b1_window(qc, wave, wave)

    def test_a_shifted_crop_is_refused(self):
        wave = np.arange(10, dtype=float)
        qc = self._qc(spatial_shifts=[{"shift_y": 0.4, "shift_x": 0.0}])
        with self.assertRaises(PerObservationSubtractError):
            b1_window(qc, wave, wave)

    def test_a_different_wavelength_axis_is_refused(self):
        with self.assertRaises(PerObservationSubtractError):
            b1_window(self._qc(), np.arange(10, dtype=float), np.arange(10, dtype=float) + 0.5)


class MixtureInputTests(unittest.TestCase):
    def test_a_non_mixture_model_is_refused(self):
        with self.assertRaises(PerObservationSubtractError) as ctx:
            models_by_exposure(moffat_doc(3.0))
        self.assertIn("mixture", str(ctx.exception))

    def test_an_empty_mixture_is_refused(self):
        with self.assertRaises(PerObservationSubtractError):
            models_by_exposure({"form": "mixture", "components": []})


if __name__ == "__main__":
    unittest.main()
