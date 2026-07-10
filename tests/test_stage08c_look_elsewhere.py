import tempfile
import unittest

import numpy as np

from musepipe.stages.stage08c_look_elsewhere import (
    compute_stage08c_products,
    empirical_fap,
    empirical_threshold_rows,
    stage08c_paths,
    top_spectral_peaks,
    write_stage08c_products,
)


def synthetic_c4_inputs():
    rng = np.random.default_rng(23)
    wavelengths = 6500.0 + 0.5 * np.arange(200)
    good = np.ones(wavelengths.size, dtype=bool)
    good[20:25] = False
    controls = rng.normal(0.0, 1.0, (8, wavelengths.size))
    # Add low-frequency structure shared by each full spectrum.
    controls += 0.3 * np.sin((wavelengths - wavelengths[0])[None, :] / 8.0)
    controls[:, ~good] = np.nan
    object_spectrum = rng.normal(0.0, 1.0, wavelengths.size)
    object_spectrum += 0.3 * np.sin((wavelengths - wavelengths[0]) / 8.0)
    object_spectrum[125] += 4.0
    object_spectrum[~good] = np.nan
    star_yx = (25, 25)
    positions = [
        (25, 40),
        (36, 36),
        (40, 25),
        (36, 14),
        (25, 10),
        (14, 14),
        (10, 25),
        (14, 36),
    ]
    config = {
        "run_id": "synthetic",
        "default_aperture": "box3_sum",
        "continuum_window_A": 20.0,
        "control_reference_exclude_angle_deg": 0.0,
        "fap_levels": [0.20, 0.10, 0.01],
        "top_n_peaks": 5,
        "peak_min_separation_A": 3.0,
        "halpha_A": 6562.8,
        "halpha_half_width_A": 3.0,
        "search_wavelength_min_A": None,
        "search_wavelength_max_A": None,
    }
    return wavelengths, good, object_spectrum, controls, positions, star_yx, config


class Stage08cLookElsewhereTests(unittest.TestCase):
    def test_empirical_fap_and_resolution_are_finite_sample_corrected(self):
        null = np.array([1.0, 2.0, 3.0, 4.0])
        self.assertAlmostEqual(empirical_fap(3.5, null), 2.0 / 5.0)
        self.assertAlmostEqual(empirical_fap(5.0, null), 1.0 / 5.0)

        rows = empirical_threshold_rows(null, [0.25, 0.10])
        self.assertEqual(rows[0]["resolvable"], 1)
        self.assertIsNotNone(rows[0]["threshold_snr_like"])
        self.assertEqual(rows[1]["resolvable"], 0)
        self.assertIsNone(rows[1]["threshold_snr_like"])
        self.assertAlmostEqual(rows[1]["minimum_resolvable_fap"], 0.2)

    def test_compute_builds_null_maxima_candidates_and_masked_spectra(self):
        waves, good, obj, controls, positions, star, config = synthetic_c4_inputs()
        products = compute_stage08c_products(
            waves,
            good,
            obj,
            controls,
            positions,
            star,
            config,
        )
        self.assertEqual(products["null_snr_like"].shape, controls.shape)
        self.assertEqual(len(products["null_rows"]), len(positions))
        self.assertTrue(np.all(np.isfinite(products["null_maxima"])))
        self.assertTrue(np.all(np.isnan(products["null_snr_like"][:, ~good])))
        self.assertGreaterEqual(len(products["candidate_rows"]), 2)
        self.assertTrue(
            any(row["candidate_type"] == "Halpha_window" for row in products["candidate_rows"])
        )
        self.assertTrue(np.isfinite(products["halpha_pointwise_empirical_fap"]))
        self.assertTrue(np.isfinite(products["halpha_window_empirical_fap"]))
        self.assertGreaterEqual(products["object_global_fap"], 1.0 / (len(positions) + 1))

        peak_indices = top_spectral_peaks(waves, products["object_products"]["snr_like"], good)
        self.assertGreaterEqual(len(peak_indices), 1)

    def test_writer_preserves_c4_product_contract(self):
        waves, good, obj, controls, positions, star, config = synthetic_c4_inputs()
        products = compute_stage08c_products(
            waves,
            good,
            obj,
            controls,
            positions,
            star,
            config,
        )
        inputs = {
            "wavelengths": waves,
            "good_mask": good,
            "object_yx": (25, 39),
            "star_yx": star,
            "control_positions_yx": positions,
        }
        with tempfile.TemporaryDirectory() as tmp:
            paths = stage08c_paths("synthetic", project_root=tmp)
            qc = write_stage08c_products(products, inputs, config, paths)
            for key in (
                "null_maxima_csv",
                "candidates_csv",
                "thresholds_csv",
                "spectra_npz",
                "qc_json",
                "summary_plot",
            ):
                self.assertTrue(paths[key].exists(), key)
            self.assertEqual(qc["n_control_spectra"], len(positions))
            self.assertEqual(qc["n_search_channels"], int(np.sum(good)))
            self.assertIsNone(qc["thresholds"][-1]["threshold_snr_like"])


if __name__ == "__main__":
    unittest.main()
