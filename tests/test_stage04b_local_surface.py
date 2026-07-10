import tempfile
import unittest
from pathlib import Path

import numpy as np
from astropy.io import fits

from musepipe.stages.stage04b_local_surface import (
    build_bad_wavelength_mask,
    compute_stage04b_products,
    resolve_stage04b_positions,
    stage04b_config_from_run,
    stage04b_paths,
    write_stage04b_products,
)


def synthetic_stage04b_inputs():
    n_cubes = 2
    ny, nx = 17, 17
    yy, xx = np.mgrid[:ny, :nx]
    plane = 10.0 + 0.05 * (xx - 8) - 0.03 * (yy - 8)
    wavelengths = np.array([4805.0, 4860.96, 4900.0, 6558.0, 6560.96, 6562.21, 6563.46, 6575.0, 6600.0])
    cubes = np.repeat(plane[None, None, :, :], n_cubes, axis=0)
    cubes = np.repeat(cubes, wavelengths.size, axis=1).astype(np.float32)

    target_yx = (9, 12)
    for cube_index, amplitude in enumerate([3.0, 8.0]):
        for wave_index in [4, 5, 6]:
            cubes[cube_index, wave_index, target_yx[0], target_yx[1]] += amplitude

    med_pix_stack = np.ones((n_cubes, ny, nx), dtype=np.float32)
    return cubes, wavelengths, med_pix_stack


def synthetic_config():
    return {
        "run_id": "synthetic",
        "target_name": "Synthetic",
        "input_mode": "native_stage02",
        "input_cube_fits": "synthetic_stage02.fits",
        "input_units_label": "native_stage02_physical_like",
        "peak_selection_mode": "manual",
        "manual_peak_b_yx": (8, 8),
        "manual_peak_c_yx": (9, 12),
        "target_object": "c",
        "target_source_label": "object",
        "local_model_kind": "plane",
        "fit_radius_px": 5.0,
        "mask_radius_px": 1.0,
        "local_fit_sigma_clip": 3.0,
        "local_fit_max_iter": 3,
        "local_fit_min_pixels": 10,
        "mask_other_objects": True,
        "other_mask_radius_px": 1.0,
        "box_size": 1,
        "halpha_channels_A": [6560.96, 6562.21, 6563.46],
        "hbeta_channels_A": [4860.96],
        "cont_ha_min_A": 6570.0,
        "cont_ha_max_A": 6700.0,
        "cont_hb_min_A": 4800.0,
        "cont_hb_max_A": 4930.0,
        "fig_convolve_sigma_px": 0.0,
        "n_best_cubes": 1,
        "n_jobs": 1,
        "bad_wavelength_ranges_A": [],
    }


class Stage04bLocalSurfaceTests(unittest.TestCase):
    def test_bad_wavelength_mask_from_ranges(self):
        waves = np.array([5779.0, 5780.0, 5900.0, 6050.0, 6051.0])
        mask = build_bad_wavelength_mask(waves, [[5780.0, 6050.0]])
        np.testing.assert_array_equal(mask, [False, True, True, True, False])

    def test_resolve_positions_from_qc_and_target_object(self):
        config = synthetic_config()
        config["peak_selection_mode"] = "stage04_qc"
        qc = {"stage": "stage04_qc", "detected_peaks": {"b": {"y": 7, "x": 6}, "c": {"y": 10, "x": 13}}}
        positions = resolve_stage04b_positions(config, qc, (17, 17))

        self.assertEqual(positions["peak_b_yx"], (7, 6))
        self.assertEqual(positions["peak_c_yx"], (10, 13))
        self.assertEqual(positions["target_yx"], (10, 13))
        self.assertEqual(positions["other_yx"], (7, 6))

    def test_compute_stage04b_products_ranks_by_target_halpha_snr(self):
        cubes, wavelengths, med_pix_stack = synthetic_stage04b_inputs()
        config = synthetic_config()
        products = compute_stage04b_products(cubes, wavelengths, med_pix_stack, config)

        self.assertEqual(products["target_yx"], (9, 12))
        self.assertEqual(products["best4"], [1])
        self.assertEqual(products["worst2"], [0])
        self.assertEqual(products["cubes_res_phys"].shape, cubes.shape)
        self.assertEqual(products["cube_res_best4"].shape, cubes.shape[1:])
        self.assertGreater(products["ha_snr_maps"][1, 9, 12], products["ha_snr_maps"][0, 9, 12])
        self.assertGreater(products["cube_res_best4"][5, 9, 12], 5.0)

    def test_write_stage04b_products_preserves_file_contract(self):
        cubes, wavelengths, med_pix_stack = synthetic_stage04b_inputs()
        config = synthetic_config()
        products = compute_stage04b_products(cubes, wavelengths, med_pix_stack, config)

        with tempfile.TemporaryDirectory() as tmp:
            paths = stage04b_paths("synthetic", Path(tmp))
            written = write_stage04b_products(products, config, paths, save_plots=False)

            self.assertTrue(written["stage04b_cube_fits"].exists())
            self.assertTrue(written["summary_csv"].exists())
            self.assertTrue(written["qc_json"].exists())
            self.assertTrue(written["stage04_qc_json"].exists())
            self.assertTrue(paths["cube_residual_object"].exists())
            self.assertTrue(paths["coordinate_check_cube"].exists())

            with fits.open(written["stage04b_cube_fits"]) as hdul:
                self.assertEqual([hdu.name for hdu in hdul], ["PRIMARY", "RESIDUALS", "LOCAL_MODEL", "WAVELENGTH", "GOOD_WAVE_MASK", "BAD_WAVE_MASK"])
                self.assertEqual(hdul["RESIDUALS"].data.shape, cubes.shape)
                self.assertEqual(hdul["WAVELENGTH"].data.shape, wavelengths.shape)

            header = written["summary_csv"].read_text().splitlines()[0]
            self.assertEqual(header, "rank,cube_index,score_target,score_b,score_c")
            self.assertEqual(written["qc"]["best4"], [1])
            self.assertEqual(written["qc"]["target_yx"], [9, 12])

    def test_stage04b_config_from_active_run_resolves_defaults(self):
        config = stage04b_config_from_run("ROXs12b_short")
        self.assertEqual(config["run_id"], "ROXs12b_short")
        self.assertEqual(config["input_mode"], "native_stage02")
        self.assertEqual(config["target_object"], "c")
        self.assertEqual(config["halpha_channels_A"], [6560.96, 6562.21, 6563.46])


if __name__ == "__main__":
    unittest.main()
