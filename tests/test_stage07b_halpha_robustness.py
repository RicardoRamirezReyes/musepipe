import unittest

import numpy as np

from musepipe.stages.stage07b_halpha_robustness import (
    compute_stage07b_products,
    default_apertures,
    empirical_percentile,
    halpha_masks,
    leave_one_out_control_sub,
    line_metrics,
    resolve_object_yx,
    resolve_star_yx,
    stage07b_config_from_run,
)


class Stage07bPureTests(unittest.TestCase):
    def test_halpha_masks_define_line_continuum_and_plot_windows(self):
        waves = np.array([6540.0, 6548.0, 6558.0, 6562.8, 6565.0, 6570.0, 6578.0, 6585.0])
        good = np.ones(waves.size, dtype=bool)
        line, cont, plot = halpha_masks(waves, good)

        np.testing.assert_array_equal(line, [False, False, False, True, True, False, False, False])
        np.testing.assert_array_equal(cont, [False, True, False, False, False, True, True, False])
        np.testing.assert_array_equal(plot, [False, True, True, True, True, True, True, False])

    def test_line_metrics_subtracts_continuum_and_reports_peak(self):
        waves = np.array([6548.0, 6552.0, 6562.8, 6564.0, 6572.0, 6578.0])
        spec = np.array([10.0, 10.0, 13.0, 12.0, 10.0, 10.0])
        good = np.ones(waves.size, dtype=bool)

        metrics = line_metrics(waves, spec, good)
        self.assertEqual(metrics["n_line_channels"], 2)
        self.assertEqual(metrics["n_cont_channels"], 4)
        self.assertEqual(metrics["continuum_median"], 10.0)
        self.assertEqual(metrics["line_flux_native"], 5.0)
        self.assertEqual(metrics["line_peak_above_continuum"], 3.0)
        self.assertEqual(metrics["line_peak_wave_A"], 6562.8)

    def test_leave_one_out_control_subtracts_other_controls(self):
        wave_indices = np.array([1, 2])
        specs = [
            np.array([np.nan, 2.0, 4.0, np.nan]),
            np.array([np.nan, 4.0, 8.0, np.nan]),
            np.array([np.nan, 6.0, 12.0, np.nan]),
        ]
        residuals = leave_one_out_control_sub(specs, wave_indices)

        np.testing.assert_allclose(residuals[0][wave_indices], [-3.0, -6.0])
        np.testing.assert_allclose(residuals[1][wave_indices], [0.0, 0.0])
        np.testing.assert_allclose(residuals[2][wave_indices], [3.0, 6.0])

    def test_coordinate_resolution_uses_target_peak_and_crop_center(self):
        cfg = {"crop_npix": 11}
        qc = {
            "target_object": "c",
            "detected_peaks": {"c": {"y": 8, "x": 3}},
            "output_shape": [1, 6, 11, 11],
        }
        self.assertEqual(resolve_object_yx(cfg, qc), (8, 3))
        self.assertEqual(resolve_star_yx(cfg, qc), (5, 5))

    def test_stage07b_config_from_active_run_resolves_roxs12b_short(self):
        config = stage07b_config_from_run("ROXs12b_short")
        self.assertEqual(config["run_id"], "ROXs12b_short")
        self.assertEqual(config["object_xy"], (72, 152))
        self.assertEqual(config["star_yx"], (85, 85))

    def test_empirical_percentile(self):
        self.assertEqual(empirical_percentile(3.0, [1.0, 2.0, 3.0, 4.0]), 75.0)
        self.assertTrue(np.isnan(empirical_percentile(np.nan, [1.0, 2.0])))

    def test_compute_stage07b_products_on_synthetic_cube(self):
        ny, nx = 11, 11
        yy, xx = np.mgrid[:ny, :nx]
        plane = 10.0 + 0.1 * (xx - 5) - 0.2 * (yy - 5)
        waves = np.array([6548.0, 6554.0, 6561.5, 6562.8, 6564.0, 6571.0, 6578.0])
        cube = np.repeat(plane[None, :, :], waves.size, axis=0).astype(float)
        object_yx = (5, 8)
        cube[2, object_yx[0], object_yx[1]] += 4.0
        cube[3, object_yx[0], object_yx[1]] += 6.0
        cube[4, object_yx[0], object_yx[1]] += 5.0

        config = {
            "run_id": "synthetic",
            "object_xy": (object_yx[1], object_yx[0]),
            "star_yx": (5, 5),
            "apertures": [default_apertures()[0]],
            "halpha_A": 6562.8,
            "line_window_half_width_A": 18.0,
            "line_integration_half_width_A": 2.5,
            "line_continuum_inner_A": 6.0,
            "line_continuum_outer_A": 16.0,
            "local_model_kind": "plane",
            "fit_radius_px": 4.0,
            "mask_radius_px": 1.0,
            "local_fit_sigma_clip": 3.0,
            "local_fit_max_iter": 3,
            "local_fit_min_pixels": 8,
            "control_apertures": 4,
            "control_exclude_angle_deg": 10.0,
            "control_margin_px": 0,
        }
        products = compute_stage07b_products(cube, waves, np.ones(waves.size, dtype=bool), config)
        row = products["summary_rows"][0]

        self.assertEqual(row["aperture"], "pixel")
        self.assertEqual(row["n_controls"], 3)
        self.assertAlmostEqual(row["object_ctrlsub_line_flux_native"], 15.0, places=6)
        self.assertAlmostEqual(row["object_ctrlsub_line_peak_above_continuum"], 6.0, places=6)
        self.assertEqual(products["object_yx"], object_yx)


if __name__ == "__main__":
    unittest.main()
