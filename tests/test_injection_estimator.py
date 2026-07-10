import unittest

import numpy as np

from musepipe.stages import stage_h01_detect
from musepipe.stages.stage_h01_detect import HALPHA_REST_A, gaussian_flux_template
from musepipe.stages.stage_h04_injection import H01_MATCHED_FILTER_POINT, measure_recovery_with_h01_estimator


class InjectionEstimatorTests(unittest.TestCase):
    def test_h04_uses_h01_matched_filter_function(self):
        self.assertIs(H01_MATCHED_FILTER_POINT, stage_h01_detect.matched_filter_point)

    def test_recovery_measurement_matches_h01_estimator(self):
        wave = np.arange(6525.0, 6601.0, 1.0, dtype=np.float64)
        fwhm = 2.5
        flux_in = 4.0
        flux = flux_in * gaussian_flux_template(wave, HALPHA_REST_A, fwhm)
        err = np.full(wave.size, 0.2, dtype=np.float64)

        measured = measure_recovery_with_h01_estimator(
            {"wave_A": wave, "flux": flux, "flux_err": err},
            line_center_A=HALPHA_REST_A,
            line_fwhm_A=fwhm,
        )

        self.assertAlmostEqual(measured["recovered_flux"], flux_in, places=10)
        self.assertGreater(measured["recovered_snr"], 0.0)


if __name__ == "__main__":
    unittest.main()
