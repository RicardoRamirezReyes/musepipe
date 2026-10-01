import unittest

import numpy as np
from astropy.io import fits

from musepipe.qc.cube_qc import (
    C_KMS,
    Skyline,
    detect_wavelength_frame,
    expected_skyline_wave,
    fit_wavelength_offsets,
    measure_skylines,
)


def _gaussian(wave, center, sigma, amp):
    return amp * np.exp(-0.5 * ((wave - center) / sigma) ** 2)


class CubeQcFrameTests(unittest.TestCase):
    def test_barycentric_skyline_expectation_uses_vbary(self):
        lab = 6300.304
        vbary = 25.0
        expected = expected_skyline_wave(lab, frame="barycentric", vbary_kms=vbary)
        self.assertAlmostEqual(expected, lab * (1.0 + vbary / C_KMS))

    def test_con_barycorr_negativo_el_cielo_cae_al_azul(self):
        # Medido (no derivado de esta función) el 2026-09-28 en los 22 cubos por exposición de la noche
        # buena de ROXs 12 b, BARYCORR −29.5 km/s: los residuos de cielo del cubo caen al AZUL del
        # SKY_SPECTRUM topocéntrico en 21 de 22 (mediana −0.62 Å a 7700-9300 Å;
        # `scripts/cielo_en_ha_por_exposicion.py --marco 22`). El signo viejo, λ·(1 − v/c), los ponía a
        # +0.84; lo que fija el dato es el signo, no la cifra.
        esperado = expected_skyline_wave(8500.0, frame="barycentric", vbary_kms=-29.5)
        self.assertLess(esperado, 8500.0 - 0.5)

    def test_header_detection_reads_barycentric_velocity(self):
        header = fits.Header()
        header["HIERARCH ESO DRS MUSE RVCORR"] = 25.0
        frame, vbary = detect_wavelength_frame(header)
        self.assertEqual(frame, "barycentric")
        self.assertEqual(vbary, 25.0)

    def test_header_detection_reads_specsys_barycent_without_velocity(self):
        header = fits.Header()
        header["SPECSYS"] = "BARYCENT"
        frame, vbary = detect_wavelength_frame(header)
        self.assertEqual(frame, "barycentric")
        self.assertIsNone(vbary)

    def test_m1_corrects_barycentric_frame(self):
        line = Skyline("OI_6300", 6300.304)
        vbary = 25.0
        expected = expected_skyline_wave(line.wave_A, frame="barycentric", vbary_kms=vbary)
        wave = np.arange(expected - 5.0, expected + 5.0, 0.05)
        spec = 1.0 + _gaussian(wave, expected, 0.45, 100.0)
        measurements = measure_skylines(
            wave,
            spec,
            [line],
            frame="barycentric",
            vbary_kms=vbary,
            min_snr=1.0,
        )
        fit = fit_wavelength_offsets(measurements)
        self.assertAlmostEqual(fit["offset_median_A"], 0.0, places=2)


if __name__ == "__main__":
    unittest.main()
