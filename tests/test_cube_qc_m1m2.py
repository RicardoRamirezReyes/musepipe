import unittest

import numpy as np

from musepipe.qc.cube_qc import (
    M1_CLEAN_AIRGLOW,
    MUSE_LSF_POLY_BACON2017,
    MUSE_LSF_REFERENCE,
    Skyline,
    measure_m1_m2_from_sky_spectrum,
    nominal_muse_fwhm_A,
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


class NominalLsfReferenceTests(unittest.TestCase):
    """La referencia de M2 es la LSF publicada, no una aproximación local."""

    def test_matches_published_polynomial(self):
        # Bacon et al. 2017, A&A 608, A1, Eq. 8.
        wave = np.array([4800.0, 6563.0, 7000.0, 9300.0])
        expected = 5.866e-8 * wave**2 - 9.187e-4 * wave + 6.040
        np.testing.assert_allclose(nominal_muse_fwhm_A(wave), expected, rtol=1e-12)
        # Valor citado en el notebook A4 y en la spec para Hα.
        self.assertAlmostEqual(float(nominal_muse_fwhm_A([6563.0])[0]), 2.537, places=3)

    def test_qc_records_the_citation(self):
        # Un QC debe declarar contra qué referencia se calculó su desviación:
        # sin eso, un QC antiguo y uno nuevo son indistinguibles.
        wave, flux = _synthetic_sky(offset_A=0.0, fwhm_A=2.4)
        lsf_lines = [Skyline("OI_5577", 5577.338), Skyline("OI_6300", 6300.304)]
        m2 = measure_m1_m2_from_sky_spectrum(
            wave, flux, lsf_skylines=lsf_lines, frame="topocentric", vbary_kms=0.0,
            min_snr_lsf=5.0,
        )["m2_lsf"]
        self.assertEqual(m2["nominal_reference"], MUSE_LSF_REFERENCE)
        self.assertIn("Bacon", m2["nominal_reference"])
        self.assertEqual(m2["nominal_poly_coeffs"], list(MUSE_LSF_POLY_BACON2017))


if __name__ == "__main__":
    unittest.main()
