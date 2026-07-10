import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from musepipe.stages.stage07_accretion_lines import (
    STAGE07_DEFAULTS,
    _clear_managed_line_plots,
    compute_stage07_products,
    filter_lines_in_range,
    line_metrics,
    stage07_config_from_run,
    wavelength_in_bad_ranges,
    write_stage07_products,
)


class Stage07AccretionLinesTests(unittest.TestCase):
    def test_clear_managed_line_plots_removes_only_stage07_windows(self):
        with tempfile.TemporaryDirectory() as tmp:
            line_dir = Path(tmp)
            stale = line_dir / "stage07_line_old_1.0A.png"
            unrelated = line_dir / "keep.png"
            stale.write_bytes(b"old")
            unrelated.write_bytes(b"keep")
            _clear_managed_line_plots(line_dir)
            self.assertFalse(stale.exists())
            self.assertTrue(unrelated.exists())

    def test_config_resolves_stage04b_geometry(self):
        config = stage07_config_from_run("ROXs12b_short")
        self.assertEqual(config["star_yx"], (85, 85))
        self.assertEqual(config["star_spectrum_yx"], (85, 85))
        self.assertEqual(config["object_yx"], (152, 72))
        self.assertEqual(config["integrated_box_size"], 3)

    def test_line_range_and_bad_range_filters(self):
        lines = [
            {"name": "blue", "wave_A": 5000.0, "family": "x", "kind": "x"},
            {"name": "middle", "wave_A": 6562.8, "family": "x", "kind": "x"},
            {"name": "red", "wave_A": 9000.0, "family": "x", "kind": "x"},
        ]
        kept = filter_lines_in_range(lines, np.linspace(6500.0, 6600.0, 101))
        self.assertEqual([line["name"] for line in kept], ["middle"])
        self.assertTrue(wavelength_in_bad_ranges(6562.8, [[6560.0, 6570.0]]))
        self.assertFalse(wavelength_in_bad_ranges(6562.8, [[5800.0, 6000.0]]))

    def test_line_metrics_recovers_synthetic_emission(self):
        waves = np.arange(6540.0, 6586.0)
        spec = 4.0 + 0.01 * (waves - waves.mean())
        spec[np.abs(waves - 6563.0) <= 2.0] += 5.0
        metrics = line_metrics(waves, spec, 6562.8)
        self.assertGreater(metrics["line_flux_native"], 20.0)
        self.assertGreater(metrics["line_peak_above_continuum"], 4.8)
        self.assertGreater(metrics["line_peak_snr"], 10.0)
        self.assertEqual(metrics["n_line_channels"], 5)

    def make_products(self):
        waves = np.arange(6535.0, 6591.0)
        nz = waves.size
        ny = nx = 45
        yy, xx = np.mgrid[:ny, :nx]
        plane = 10.0 + 0.01 * (xx - 22) - 0.02 * (yy - 22)
        control_cube = np.repeat(plane[None, :, :], nz, axis=0)
        star_cube = control_cube.copy()
        object_cube = np.zeros_like(control_cube)
        star_yx = (22, 22)
        object_yx = (22, 30)
        line = np.exp(-0.5 * ((waves - 6562.8) / 1.0) ** 2)
        star_cube[:, star_yx[0], star_yx[1]] += 20.0 * line
        object_cube[:, object_yx[0], object_yx[1]] += 8.0 * line

        config = {
            "run_id": "synthetic_stage07",
            "project_root": ".",
            "target_name": "synthetic",
            "object_name": "synthetic object",
            "object_source_label": "synthetic residual",
            "object_yx": object_yx,
            "star_yx": star_yx,
            "star_spectrum_yx": star_yx,
            "accretion_lines": [
                {"name": "Halpha", "wave_A": 6562.8, "family": "Balmer", "kind": "accretion"}
            ],
            "bad_wavelength_ranges_A": [],
            **STAGE07_DEFAULTS,
        }
        config.update(
            {
                "control_apertures": 8,
                "fit_radius_px": 7.0,
                "mask_radius_px": 2.0,
                "local_fit_min_pixels": 20,
                "full_continuum_filter_A": 21.0,
            }
        )
        products = compute_stage07_products(
            star_cube,
            waves,
            object_cube,
            waves,
            config,
            control_input_cube=control_cube,
            control_waves=waves,
        )
        return config, products

    def test_compute_uses_controls_and_finds_halpha(self):
        _, products = self.make_products()
        self.assertEqual(len(products["metrics_rows"]), 1)
        self.assertEqual(len(products["control_positions_yx"]), 7)
        row = products["metrics_rows"][0]
        self.assertGreater(row["object_box_line_flux_native"], 15.0)
        self.assertGreater(row["object_box_ctrlsub_line_flux_native"], 15.0)
        self.assertGreater(row["star_center_line_flux_native"], 40.0)

    def test_write_products_preserves_csv_and_qc_contract(self):
        config, products = self.make_products()
        with tempfile.TemporaryDirectory() as tmp:
            config = dict(config, project_root=tmp)
            outputs = write_stage07_products(config, products, save_plots=False)
            self.assertTrue(outputs["metrics_csv"].exists())
            self.assertTrue(outputs["qc_json"].exists())
            qc = json.loads(outputs["qc_json"].read_text(encoding="utf-8"))
            self.assertEqual(qc["stage"], "stage07_accretion_line_spectra")
            self.assertEqual(qc["star_spectrum_yx"], [22, 22])
            self.assertEqual(qc["object_yx"], [22, 30])
            self.assertEqual(len(qc["local_control_positions_yx"]), 7)
            header = outputs["metrics_csv"].read_text(encoding="utf-8").splitlines()[0]
            self.assertIn("object_box_ctrlsub_line_peak_snr", header)


if __name__ == "__main__":
    unittest.main()
