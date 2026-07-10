import unittest

import numpy as np

from musepipe.qc.cube_qc import (
    M1_CLEAN_AIRGLOW,
    Skyline,
    measure_m1_m2_from_sky_spectrum,
)


def _synthetic_sky(offset_A=0.05, fwhm_A=2.4, lines=None):
    """Sky spectrum with Gaussian airglow lines at lab+offset with a known FWHM."""

    wave = np.arange(4750.0, 9350.0, 0.3125, dtype=np.float64)
    lines = lines or [(5577.338, 100.0), (6300.304, 60.0), (6363.776, 40.0)]
    sigma = fwhm_A / 2.354820045
    flux = np.full(wave.size, 1.0, dtype=np.float64)  # faint continuum
    for lab, amp in lines:
        flux += amp * np.exp(-0.5 * ((wave - (lab + offset_A)) / sigma) ** 2)
    return wave, flux


class CubeQcM1M2Tests(unittest.TestCase):
    def test_recovers_injected_wavelength_offset(self):
        wave, flux = _synthetic_sky(offset_A=0.05, fwhm_A=2.4)
        res = measure_m1_m2_from_sky_spectrum(wave, flux, frame="topocentric", vbary_kms=0.0)
        m1 = res["m1_wavelength"]
        self.assertAlmostEqual(m1["offset_median_A"], 0.05, delta=0.02)
        self.assertGreaterEqual(m1["n_lines"], 2)

    def test_recovers_injected_lsf_fwhm_at_halpha(self):
        wave, flux = _synthetic_sky(offset_A=0.0, fwhm_A=2.4)
        lsf_lines = [
            Skyline("OI_5577", 5577.338),
            Skyline("OI_6300", 6300.304),
            Skyline("OI_6363", 6363.776),
        ]
        res = measure_m1_m2_from_sky_spectrum(
            wave, flux, lsf_skylines=lsf_lines, frame="topocentric", vbary_kms=0.0, min_snr_lsf=5.0
        )
        lsf = res["lsf_fwhm_at_halpha_A"]
        self.assertIsNotNone(lsf)
        # Moment FWHM on a clean Gaussian recovers the injected width to a few %.
        self.assertAlmostEqual(lsf, 2.4, delta=0.2)

    def test_barycentric_frame_shifts_expected_positions(self):
        # Airglow injected at topocentric rest; measuring in the barycentric
        # frame (vbary != 0) should yield a large spurious offset.
        wave, flux = _synthetic_sky(offset_A=0.0, fwhm_A=2.4)
        topo = measure_m1_m2_from_sky_spectrum(wave, flux, frame="topocentric", vbary_kms=0.0)
        bary = measure_m1_m2_from_sky_spectrum(wave, flux, frame="barycentric", vbary_kms=-29.64)
        self.assertLess(abs(topo["m1_wavelength"]["offset_median_A"]), 0.03)
        self.assertGreater(abs(bary["m1_wavelength"]["offset_median_A"]), 0.3)

    def test_default_m1_lines_are_clean_atomic(self):
        names = {s.name for s in M1_CLEAN_AIRGLOW}
        self.assertEqual(names, {"OI_5577", "OI_6300", "OI_6363"})


if __name__ == "__main__":
    unittest.main()
