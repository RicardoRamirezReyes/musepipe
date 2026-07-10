import unittest

from musepipe.stages.stage_h02_artifacts import t5_placebo_calibration


class H02T5CalibrationTests(unittest.TestCase):
    def test_noise_placebos_do_not_cross_detection_threshold(self):
        rows = [
            {"center_A": 6200.0, "method": "psffit", "status": "ran", "global_empirical_fap": 0.52},
            {"center_A": 6400.0, "method": "psffit", "status": "ran", "global_empirical_fap": 0.31},
            {"center_A": 6700.0, "method": "aperture", "status": "ran", "global_empirical_fap": 0.24},
            {"center_A": 7100.0, "method": "aperture", "status": "ran", "global_empirical_fap": 0.88},
        ]

        result = t5_placebo_calibration(rows, detection_fap=0.01)

        self.assertEqual(result["status"], "pass")
        self.assertFalse(result["any_above_threshold"])
        self.assertEqual(result["placebo_max_fap_global"], 0.24)


if __name__ == "__main__":
    unittest.main()
