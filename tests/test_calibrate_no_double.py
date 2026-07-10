import unittest

import numpy as np

from musepipe.extraction.product import FORMAT_VERSION, SpectrumProduct
from musepipe.stages.stage_x11_calibrate import CalibrationCorrections, calibrate_spectrum_product


def calibration_product(method="psffit", *, wave=None, flux=None, flux_err=None, flux_err_emp=None):
    wave = np.linspace(7000.0, 7100.0, 31) if wave is None else np.asarray(wave, dtype=np.float64)
    flux = np.full(wave.size, 10.0, dtype=np.float64) if flux is None else np.asarray(flux, dtype=np.float64)
    flux_err = np.full(wave.size, 1.0, dtype=np.float64) if flux_err is None else np.asarray(flux_err, dtype=np.float64)
    flux_err_emp = (
        np.full(wave.size, 1.5, dtype=np.float64)
        if flux_err_emp is None
        else np.asarray(flux_err_emp, dtype=np.float64)
    )
    fits_method = "optimal" if method.startswith("optimal") else method
    return SpectrumProduct(
        wave_A=wave,
        flux=flux,
        flux_err=flux_err,
        flux_err_emp=flux_err_emp,
        apcorr=np.full(wave.size, 2.0, dtype=np.float64),
        npix_eff=np.full(wave.size, 9.0, dtype=np.float64),
        flags=np.zeros(wave.size, dtype=np.int32),
        header={
            "FORMATV": FORMAT_VERSION,
            "METHOD": fits_method,
            "RUNID": "synthetic_x11",
            "SRCPOS_Y": 12.0,
            "SRCPOS_X": 16.0,
            "APERTURE": "psffit_object" if method == "psffit" else method,
            "WFRAME": "barycentric",
            "INCUBE": "cube.fits",
            "INCUBESH": "samecube",
            "NORMRAD": 25.0,
            "ERRMODE": "stat",
            "APCMODE": "psf_model_norm_radius",
            "STATFAC": 1.3,
            "COVFAC": 1.1,
        },
    )


class CalibrateNoDoubleTests(unittest.TestCase):
    def test_declared_apcorr_and_barycentric_frame_are_not_reapplied(self):
        product = calibration_product(flux=np.full(31, 10.0))
        corrections = CalibrationCorrections(
            wavelength_status="green",
            wavelength_apply=True,
            wavelength_offset_A=0.0,
            frame_final="barycentric",
            flux_scale=1.0,
        )

        calibrated = calibrate_spectrum_product(product, corrections, method="psffit", canonical=True)

        np.testing.assert_allclose(calibrated.product.flux, product.flux)
        np.testing.assert_allclose(calibrated.product.wave_A, product.wave_A)
        self.assertTrue(calibrated.already_applied["apcorr"])
        self.assertTrue(calibrated.already_applied["stat_factors_in_flux_err"])


if __name__ == "__main__":
    unittest.main()
