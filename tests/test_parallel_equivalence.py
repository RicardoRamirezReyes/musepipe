"""Channel-loop parallelization must be bit-identical to the serial path.

The per-channel loops of psffit / optimal / local-surface were wrapped in a
thread-chunked runner (``musepipe.parallel.run_channel_chunks``). Threads only
change execution order, never the per-channel arithmetic or the disjoint output
slots, so n_jobs>1 must reproduce n_jobs=1 exactly (NaN-aware).
"""

import unittest

import numpy as np

from musepipe.extraction.optimal import fit_primary_psf_model_cube, optimal_raw_spectrum
from musepipe.extraction.psffit import fit_psffit_cube, psf_pair_design
from musepipe.localfit import subtract_local_surface_cube
from musepipe.parallel import channel_chunks, resolve_n_jobs
from tests.test_optimal_analytic import constant_model_doc


def _scene(model, wave, star_yx, comp_yx, shape=(60, 60)):
    rng = np.random.default_rng(7)
    coeffs = np.column_stack([
        np.linspace(800.0, 900.0, wave.size),
        np.linspace(40.0, 60.0, wave.size),
        np.full(wave.size, 3.0),
        np.full(wave.size, 0.2),
        np.full(wave.size, -0.1),
    ])
    planes = [np.tensordot(psf_pair_design(shape, float(w), star_yx, comp_yx, model), coeffs[z], axes=([-1], [0]))
              for z, w in enumerate(wave)]
    cube = np.asarray(planes, dtype=np.float64) + rng.normal(0.0, 1.0, (wave.size, *shape))
    return cube, np.full_like(cube, 2.0)


class ParallelEquivalenceTests(unittest.TestCase):
    def setUp(self):
        self.model = constant_model_doc()
        self.wave = np.linspace(6500.0, 6580.0, 41)
        self.star = (30.0, 26.0)
        self.comp = (30.0, 40.0)
        self.cube, self.var = _scene(self.model, self.wave, self.star, self.comp)

    def test_config_n_jobs_precedence(self):
        """Knob declarado > `MUSEPIPE_N_JOBS` > cpu_count capado."""
        import os
        from unittest import mock

        from musepipe.parallel import config_n_jobs

        self.assertEqual(config_n_jobs({"x03_n_jobs": 3}, "x03_n_jobs", "n_jobs"), 3)
        self.assertEqual(config_n_jobs({"n_jobs": 2}, "x03_n_jobs", "n_jobs"), 2)
        with mock.patch.dict(os.environ, {"MUSEPIPE_N_JOBS": "5"}):
            self.assertEqual(config_n_jobs({}, "x03_n_jobs", "n_jobs"), 5)
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("MUSEPIPE_N_JOBS", None)
            self.assertGreaterEqual(config_n_jobs({}, "x03_n_jobs"), 1)

    def test_extraction_stages_actually_request_parallelism(self):
        """C3 y C4 tienen que PEDIR hilos, no solo poder pedirlos.

        Este modulo lleva desde su primer commit demostrando que hilar los
        bucles por canal es bit a bit identico, y aun asi la cadena corria en un
        nucleo: los conductores llamaban a los extractores sin `n_jobs`, que por
        defecto vale 1. La equivalencia estaba probada; el uso, no. Esto fija el
        uso.
        """
        import inspect

        from musepipe.stages import stage_x02_optimal, stage_x03_psffit

        for modulo, llamadas in ((stage_x02_optimal,
                                  ("make_optimal_product(", "fit_primary_psf_model_cube(")),
                                 (stage_x03_psffit, ("make_psffit_products(",))):
            fuente = inspect.getsource(modulo)
            self.assertIn("config_n_jobs", fuente,
                          f"{modulo.__name__} no resuelve n_jobs del run")
            for llamada in llamadas:
                for trozo in fuente.split(llamada)[1:]:
                    cierre = trozo.index("\n    )") if "\n    )" in trozo else len(trozo)
                    self.assertIn("n_jobs=", trozo[:cierre + 8],
                                  f"{modulo.__name__}: {llamada} sin n_jobs")

    def test_channel_chunks_partition_is_exact(self):
        bounds = channel_chunks(41, 6)
        self.assertEqual(bounds[0][0], 0)
        self.assertEqual(bounds[-1][1], 41)
        self.assertEqual(sum(b - a for a, b in bounds), 41)
        for (a0, a1), (b0, b1) in zip(bounds, bounds[1:]):
            self.assertEqual(a1, b0)  # contiguous, no gaps/overlaps

    def test_resolve_n_jobs_floor(self):
        self.assertEqual(resolve_n_jobs(0), 1)
        self.assertEqual(resolve_n_jobs(4), 4)

    def test_fit_psffit_cube_identical(self):
        a = fit_psffit_cube(self.cube, self.var, self.wave, self.star, self.comp, self.model, n_jobs=1)
        b = fit_psffit_cube(self.cube, self.var, self.wave, self.star, self.comp, self.model, n_jobs=4)
        np.testing.assert_array_equal(a.coeffs, b.coeffs)
        np.testing.assert_array_equal(a.chi2r, b.chi2r)
        np.testing.assert_array_equal(a.residual_cube, b.residual_cube, strict=False)

    def test_optimal_raw_spectrum_identical(self):
        a = optimal_raw_spectrum(self.cube, self.var, self.wave, self.comp, self.model, n_jobs=1)
        b = optimal_raw_spectrum(self.cube, self.var, self.wave, self.comp, self.model, n_jobs=4)
        for key in ("flux", "variance", "npix_eff", "clip_fraction", "rejection_map"):
            np.testing.assert_array_equal(a[key], b[key], err_msg=key)

    def test_fit_primary_psf_model_cube_identical(self):
        a, _ = fit_primary_psf_model_cube(self.cube, self.wave, self.star, self.model,
                                          variance_zyx=self.var, exclude_centers_yx=[self.comp], n_jobs=1)
        b, _ = fit_primary_psf_model_cube(self.cube, self.wave, self.star, self.model,
                                          variance_zyx=self.var, exclude_centers_yx=[self.comp], n_jobs=4)
        np.testing.assert_array_equal(a, b)

    def test_subtract_local_surface_cube_identical(self):
        c32 = self.cube.astype(np.float32)
        ra, ma, na = subtract_local_surface_cube(c32, target_yx=self.comp, extra_exclusion_yx=[self.star], n_jobs=1)
        rb, mb, nb = subtract_local_surface_cube(c32, target_yx=self.comp, extra_exclusion_yx=[self.star], n_jobs=4)
        np.testing.assert_array_equal(ra, rb)
        np.testing.assert_array_equal(ma, mb)
        np.testing.assert_array_equal(na, nb)


if __name__ == "__main__":
    unittest.main()
