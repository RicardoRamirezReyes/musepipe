import tempfile
import unittest
from pathlib import Path

from musepipe.stages.stage01c_localize import compute_stage01c_products
from tests.test_stage01c_synthetic import synthetic_cube, write_stack


class Stage01cDriftTests(unittest.TestCase):
    def test_one_pixel_chromatic_drift_sets_verdict(self):
        companion_yx = (30.0, 70.0)
        cube, waves, _, _, field_yx = synthetic_cube(companion_yx=companion_yx, drift_px=1.1)
        with tempfile.TemporaryDirectory() as tmp:
            input_path = Path(tmp) / "stage02.fits"
            write_stack(input_path, cube, waves)
            cfg = {
                "run_id": "synthetic_drift",
                "project_root": tmp,
                "stage01c_input_cube_fits": str(input_path),
                "pixel_scale_arcsec": 0.05,
                "stage01c_companion_approx_yx": companion_yx,
                "stage01c_field_source_approx_yx": field_yx,
                "stage01c_psf_fwhm_px": 4.0,
                "stage01c_search_radius_px": 5.0,
                "stage01c_detection_snr_min": 5.0,
                "stage01c_chromatic_bins": 6,
                "stage01c_chromatic_threshold_px": 0.5,
                "stage01c_validate_astrometry": False,
                "stage01c_validate_legacy": False,
            }
            product = compute_stage01c_products(cfg, input_path=input_path)

        chrom = product.qc["chromatic"]
        self.assertGreater(chrom["companion_drift_px_peak_to_peak"], 0.8)
        self.assertTrue(chrom["chromatic_centroid_needed"])


if __name__ == "__main__":
    unittest.main()
