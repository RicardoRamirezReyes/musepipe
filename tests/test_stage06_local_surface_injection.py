import tempfile
import unittest
from pathlib import Path

import numpy as np
from astropy.io import fits

from musepipe.stages.stage06_local_surface_injection import (
    compute_stage06_local_products,
    continuum_mask_from_config,
    inject_physical_template,
    local_surface_control_positions,
    make_signal_template,
    resolve_injection_geometry,
    stage06_local_config_from_run,
    stage06_local_paths,
    write_stage06_local_products,
)
from musepipe.stages.stage06_local_surface_sweep import (
    compute_stage06_local_sweep_products,
    input_snr_at_completeness,
    nearest_line_channel_targets,
    stage06_local_sweep_paths,
    write_stage06_local_sweep_products,
)


def synthetic_inputs():
    rng = np.random.default_rng(17)
    wavelengths = np.arange(6540.0, 6701.0, 1.25, dtype=np.float64)
    n_cubes, ny, nx = 2, 45, 45
    yy, xx = np.mgrid[:ny, :nx]
    plane = 8.0 + 0.02 * (xx - 22) - 0.015 * (yy - 22)
    cubes = np.repeat(plane[None, None, :, :], n_cubes, axis=0)
    cubes = np.repeat(cubes, wavelengths.size, axis=1).astype(np.float32)
    cubes += rng.normal(0.0, 0.03, cubes.shape).astype(np.float32)
    med_pix_stack = np.ones((n_cubes, ny, nx), dtype=np.float32)
    return cubes, wavelengths, med_pix_stack


def synthetic_config():
    return {
        "run_id": "synthetic",
        "target_name": "Synthetic",
        "project_root": ".",
        "input_mode": "native_stage02",
        "input_cube_fits": "synthetic_stage02.fits",
        "input_shape": [2, 129, 45, 45],
        "line_center_A": 6562.8,
        "halpha_channels_A": [6561.25, 6562.5, 6563.75],
        "cont_ha_min_A": 6570.0,
        "cont_ha_max_A": 6600.0,
        "muse_mode": "NFM",
        "pixel_scale_arcsec": 0.025,
        "spatial_psf_fwhm_arcsec": 0.070,
        "muse_R_at_halpha": 2484.0,
        "template_radius_nsigma": 5.0,
        "star_yx": (22, 22),
        "reference_object_yx": (22, 32),
        "match_reference_separation": True,
        "injection_yx": None,
        "injection_pa_offset_deg": 180.0,
        "injection_radius_arcsec": 0.25,
        "injection_pa_deg": 270.0,
        "inject_in_all_cubes": True,
        "cube_index_to_inject": 0,
        "use_stage04_selection": True,
        "n_best_cubes": None,
        "injection_snr_grid": [0.0, 2.0, 5.0],
        "nominal_snr": 2.0,
        "injection_snr_reference": "local_surface_matched_filter",
        "recovery_mode": "realistic",
        "box_size": 3,
        "n_control_angles": 16,
        "control_exclude_pa_deg": 20.0,
        "diagnostic_halfwidth_A": 30.0,
        "save_diagnostic_fits": True,
        "fig_convolve_sigma_px": 0.0,
        "local_model_kind": "plane",
        "fit_radius_px": 7.0,
        "mask_radius_px": 2.0,
        "local_fit_sigma_clip": 3.0,
        "local_fit_max_iter": 3,
        "local_fit_min_pixels": 40,
        "other_mask_radius_px": 2.0,
    }


class Stage06LocalSurfaceInjectionTests(unittest.TestCase):
    def test_template_integrates_to_unity_and_medpix_preserves_physical_flux(self):
        wavelengths = np.arange(6550.0, 6576.0, 1.25)
        dlam = float(np.median(np.diff(wavelengths)))
        zsl, ysl, xsl, template, _, _ = make_signal_template(
            wavelengths,
            31,
            31,
            15,
            15,
            0.025,
            0.070,
            6562.8,
            6562.8 / 2484.0,
            dlam,
        )
        self.assertAlmostEqual(float(np.sum(template) * dlam), 1.0, places=6)

        cube = np.zeros((wavelengths.size, 31, 31), dtype=np.float32)
        medpix = np.full((31, 31), 2.0, dtype=np.float32)
        injected = inject_physical_template(cube, medpix, zsl, ysl, xsl, template, 12.0)
        recovered_physical_flux = float(np.sum(injected * medpix[None, :, :]) * dlam)
        self.assertAlmostEqual(recovered_physical_flux, 12.0, places=4)

    def test_geometry_places_injection_opposite_reference(self):
        config = synthetic_config()
        geometry = resolve_injection_geometry(config, 45, 45)
        self.assertEqual(geometry["injection_yx"], (22, 12))
        self.assertAlmostEqual(geometry["reference_separation_px"], 10.0)
        self.assertAlmostEqual(geometry["injection_separation_px"], 10.0)

        controls = local_surface_control_positions(
            geometry["star_yx"],
            geometry["injection_yx"],
            geometry["reference_object_yx"],
            45,
            45,
            n_angles=16,
            exclude_pa_deg=20.0,
            margin_px=7,
            min_injection_distance_px=8.0,
        )
        self.assertGreaterEqual(len(controls), 4)
        self.assertNotIn((22, 12), controls)
        self.assertNotIn((22, 32), controls)

    def test_generic_line_channels_and_continuum_windows(self):
        _, wavelengths, _ = synthetic_inputs()
        targets = nearest_line_channel_targets(wavelengths, 6620.0, n_channels=3)
        self.assertEqual(len(targets), 3)
        self.assertEqual(len(set(targets)), 3)

        config = synthetic_config()
        config["continuum_windows_A"] = [[6595.0, 6605.0], [6635.0, 6645.0]]
        config["bad_wavelength_ranges_A"] = [[6600.0, 6602.0]]
        mask, windows = continuum_mask_from_config(wavelengths, config)
        self.assertEqual(windows, [(6595.0, 6605.0), (6635.0, 6645.0)])
        self.assertTrue(np.any(mask))
        self.assertFalse(np.any(mask & (wavelengths >= 6600.0) & (wavelengths <= 6602.0)))

    def test_recovery_grid_is_finite_monotonic_and_transfers_signal(self):
        cubes, wavelengths, medpix = synthetic_inputs()
        products = compute_stage06_local_products(
            cubes,
            wavelengths,
            medpix,
            synthetic_config(),
            selected_indices=[0, 1],
            selection_source="synthetic",
        )
        rows = products["rows"]
        self.assertEqual(len(rows), 3)
        self.assertAlmostEqual(products["template_integral"], 1.0, places=5)
        self.assertGreaterEqual(len(products["control_positions_yx"]), 4)
        self.assertAlmostEqual(rows[0]["delta_matched_flux_native"], 0.0, places=6)

        delta_flux = np.asarray([row["delta_matched_flux_native"] for row in rows])
        recovered_snr = np.asarray([row["local_surface_matched_snr"] for row in rows])
        self.assertTrue(np.all(np.isfinite(recovered_snr)))
        self.assertTrue(np.all(np.diff(delta_flux) > 0))
        self.assertGreater(recovered_snr[-1], recovered_snr[0])
        self.assertGreater(rows[-1]["delta_matched_signal_transfer"], 0.5)
        self.assertLess(rows[-1]["delta_matched_signal_transfer"], 1.2)

    def test_writer_preserves_local_surface_product_contract(self):
        cubes, wavelengths, medpix = synthetic_inputs()
        config = synthetic_config()
        products = compute_stage06_local_products(
            cubes,
            wavelengths,
            medpix,
            config,
            selected_indices=[0, 1],
            selection_source="synthetic",
        )
        with tempfile.TemporaryDirectory() as tmp:
            paths = stage06_local_paths("synthetic", tmp)
            written = write_stage06_local_products(products, config, paths, save_plots=False)
            self.assertTrue(written["results_csv"].exists())
            self.assertTrue(written["qc_json"].exists())
            self.assertTrue(written["truth_json"].exists())
            self.assertTrue(written["template_fits"].exists())
            self.assertTrue(written["diagnostic_fits"].exists())
            self.assertIn("local_surface", written["results_csv"].name)

            with fits.open(written["template_fits"]) as hdul:
                self.assertEqual(
                    [hdu.name for hdu in hdul],
                    ["PRIMARY", "LOCAL_TEMPLATE", "SPATIAL_PSF", "SPECTRAL_PROFILE", "WAVELENGTH"],
                )
            with fits.open(written["diagnostic_fits"]) as hdul:
                self.assertEqual(
                    [hdu.name for hdu in hdul],
                    ["PRIMARY", "BASE_TARGET_DIAG", "INJECTED_TARGET_DIAG", "DELTA_TARGET_DIAG", "WAVELENGTH"],
                )

    def test_c2_sweep_builds_completeness_and_consolidated_products(self):
        cubes, wavelengths, medpix = synthetic_inputs()
        config = synthetic_config()
        config.update(
            {
                "recovery_mode": "deterministic",
                "injection_snr_grid": [0.0, 3.0, 6.0],
                "sweep_lines": [
                    {
                        "label": "line_a",
                        "center_A": 6562.8,
                        "continuum_windows_A": [[6540.0, 6552.0], [6574.0, 6586.0]],
                    },
                    {
                        "label": "line_b",
                        "center_A": 6620.0,
                        "continuum_windows_A": [[6595.0, 6607.0], [6633.0, 6645.0]],
                    },
                ],
                "sweep_pa_offsets_deg": [90.0, 180.0],
                "sweep_n_line_channels": 3,
                "sweep_continuum_inner_A": 10.0,
                "sweep_continuum_outer_A": 30.0,
                "sweep_detection_threshold_snr": 3.0,
            }
        )
        products = compute_stage06_local_sweep_products(
            cubes,
            wavelengths,
            medpix,
            config,
            selected_indices=[0, 1],
            selection_source="synthetic",
        )
        self.assertEqual(len(products["case_rows"]), 4)
        self.assertEqual(len(products["grid_rows"]), 12)
        self.assertEqual(len(products["completeness_rows"]), 3)
        self.assertGreaterEqual(
            products["completeness_rows"][-1]["matched_completeness"],
            products["completeness_rows"][0]["matched_completeness"],
        )
        self.assertTrue(
            all(np.isfinite(row["delta_matched_transfer_median"]) for row in products["case_rows"])
        )
        self.assertTrue(
            np.isfinite(
                input_snr_at_completeness(
                    products["completeness_rows"],
                    "matched_completeness",
                    0.5,
                )
            )
        )

        config["input_cube_fits"] = "synthetic_stage02.fits"
        config["input_shape"] = list(cubes.shape)
        with tempfile.TemporaryDirectory() as tmp:
            paths = stage06_local_sweep_paths("synthetic", tmp)
            written = write_stage06_local_sweep_products(
                products,
                config,
                paths,
                save_plots=False,
            )
            self.assertTrue(written["grid_csv"].exists())
            self.assertTrue(written["case_summary_csv"].exists())
            self.assertTrue(written["completeness_csv"].exists())
            self.assertTrue(written["qc_json"].exists())

    def test_active_run_config_uses_stage04b_geometry(self):
        config = stage06_local_config_from_run("ROXs12b_short")
        self.assertEqual(config["run_id"], "ROXs12b_short")
        self.assertEqual(config["input_mode"], "native_stage02")
        self.assertEqual(config["reference_object_yx"], (152, 72))
        self.assertEqual(config["star_yx"], (85, 85))
        self.assertEqual(config["recovery_mode"], "realistic")


if __name__ == "__main__":
    unittest.main()
