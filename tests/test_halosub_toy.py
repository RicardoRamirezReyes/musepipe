"""WP-H0 · Analytical oracles for musepipe.halosub (Julo et al. 2025).

The paper gives closed-form predictions for the toy model of Fig. 1/Fig. 2:

* App. B.1 — the Savitzky-Golay (d=1) response at the center of a
  rectangular peak;
* Fig. 2d / Eq. 1 — post-SGF planetary line estimate, spurious negative
  continuum, and their exact ratio -(R/(1-R)) * C_S/L_S;
* Eq. B.5 — the LPM residual MSE decomposition (star underfitting +
  planet overfitting + noise overfitting).

These tests build the toy cubes and check the kernels against those
formulas, so any regression in the kernels breaks a *published* prediction,
not an ad-hoc snapshot.
"""

import unittest

import numpy as np
from scipy.signal import savgol_filter

from musepipe.halosub import (
    fill_nan_along_axis0,
    lpm_coefficient_energy_share,
    lpm_design_matrix,
    lpm_fit_mask,
    lpm_mse_terms,
    lpm_subtract,
    reference_spectrum,
    select_reference_spaxels,
    sgf_peak_response,
    sgf_self_subtraction_ratio,
    sgf_subtract,
    sgf_toy_continuum_estimate,
    sgf_toy_line_estimate,
)

# Toy geometry shared across tests (rectangular line, flat continua; Fig. 1).
NZ = 801
LINE_LO, LINE_HI = 396, 404  # 2*W_PEAK+1 = 9 channels, centered at 400
W_PEAK = 4  # (2*4+1) = 9 channels
W_MAX = 50  # SG window 2*50+1 = 101 channels
WINDOW = 2 * W_MAX + 1
R_RATIO = (2.0 * W_PEAK + 1.0) / (2.0 * W_MAX + 1.0)

C_S, L_S = 100.0, 130.0  # stellar continuum / line heights
C_P, L_P = 0.5, 20.0     # planetary continuum / line heights


def toy_spectra():
    wave = np.linspace(6300.0, 6800.0, NZ)
    star = np.full(NZ, C_S)
    star[LINE_LO : LINE_HI + 1] = L_S
    planet = np.full(NZ, C_P)
    planet[LINE_LO : LINE_HI + 1] = L_P
    return wave, star, planet


def toy_cube(star, planet, *, halo_amps, planet_at=(1, 1)):
    """Cube of stellar halo spaxels (star * amp) with one planet spaxel."""

    ny, nx = halo_amps.shape
    cube = star[:, None, None] * halo_amps[None, :, :]
    cube[:, planet_at[0], planet_at[1]] += planet
    return cube


class TestSgfPeakOracle(unittest.TestCase):
    """savgol(d=1) at a rectangular peak center == App. B.1 formula."""

    def test_peak_response_matches_savgol(self):
        h_min, h_max = 1.005, 1.2
        signal = np.full(NZ, h_min)
        signal[LINE_LO : LINE_HI + 1] = h_max
        smoothed = savgol_filter(signal, window_length=WINDOW, polyorder=1)
        center = (LINE_LO + LINE_HI) // 2
        expected = sgf_peak_response(h_min, h_max, W_PEAK, W_MAX)
        self.assertAlmostEqual(smoothed[center], expected, places=10)


class TestSgfSelfSubtraction(unittest.TestCase):
    """Noise-free toy cube reproduces Fig. 2d and Eq. 1 exactly."""

    def setUp(self):
        self.wave, self.star, self.planet = toy_spectra()
        halo = np.array([[1.0, 0.9, 0.8], [0.7, 1.1, 0.95], [1.05, 0.85, 0.75]])
        self.cube = toy_cube(self.star, self.planet, halo_amps=halo, planet_at=(1, 1))
        # Perfect reference: the known stellar spectrum (isolates the
        # filtering bias from the reference-estimation error, as in Sect. 2).
        self.res = sgf_subtract(self.cube, self.star, window=WINDOW, degree=1)

    def test_line_estimate_matches_fig2d(self):
        center = (LINE_LO + LINE_HI) // 2
        got = self.res.residual_cube[center, 1, 1]
        expected = sgf_toy_line_estimate(L_P, R_RATIO, L_S, C_S, C_P)
        self.assertAlmostEqual(got, expected, places=8)
        # And it IS self-subtracted: well below the injected line height.
        self.assertLess(got, L_P * (1.0 - R_RATIO) + 1e-9)

    def test_negative_continuum_next_to_line(self):
        # Continuum channels near the line: the centered window still covers
        # the FULL line (plateau of the smoothed box, |ch - center| <= 46),
        # so the trough depth is exactly the Fig. 2d prediction.
        ch = LINE_HI + 2
        got = self.res.residual_cube[ch, 1, 1]
        expected = sgf_toy_continuum_estimate(R_RATIO, L_P, L_S, C_S, C_P)
        self.assertLess(got, 0.0)
        self.assertAlmostEqual(got, expected, places=8)

    def test_eq1_ratio(self):
        center = (LINE_LO + LINE_HI) // 2
        line_est = self.res.residual_cube[center, 1, 1]
        cont_est = self.res.residual_cube[LINE_HI + 2, 1, 1]
        got = cont_est / line_est
        expected = sgf_self_subtraction_ratio(R_RATIO, C_S / L_S)
        self.assertAlmostEqual(got, expected, places=8)

    def test_far_continuum_unbiased(self):
        far = LINE_LO - 2 * W_MAX - 5
        self.assertAlmostEqual(self.res.residual_cube[far, 1, 1], 0.0, places=8)

    def test_halo_spaxels_fully_subtracted(self):
        resid = self.res.residual_cube[:, 0, 0]
        self.assertLess(np.max(np.abs(resid)), 1e-8)


class TestLpmRecovery(unittest.TestCase):
    """LPM with the line masked preserves the planetary signal (Sect. 3)."""

    def setUp(self):
        self.wave, self.star, self.planet = toy_spectra()
        # Stellar spaxels are smooth (quadratic) modulations of the star:
        # exactly inside the degree-4 model space.
        x = np.linspace(-1.0, 1.0, NZ)
        self.modulation = 1.0 + 0.05 * x + 0.02 * x**2
        halo = np.array([[1.0, 0.9], [0.8, 1.1]])
        cube = self.star[:, None, None] * self.modulation[:, None, None] * halo[None, :, :]
        cube[:, 1, 1] += self.planet
        self.cube = cube
        center_A = float(self.wave[(LINE_LO + LINE_HI) // 2])
        self.line_windows = ((center_A, 12.0),)
        self.res = lpm_subtract(
            self.cube, self.wave, self.star, degree=4, line_windows_A=self.line_windows
        )

    def test_stellar_spaxels_fully_subtracted(self):
        resid = self.res.residual_cube[:, 0, 0]
        self.assertLess(np.max(np.abs(resid)), 1e-6)

    def test_line_flux_preserved(self):
        center = (LINE_LO + LINE_HI) // 2
        got = self.res.residual_cube[center, 1, 1]
        # The stellar model interpolates across the masked line, so the LPM
        # line estimate is exactly the R -> 0 limit of Fig. 2d: the only loss
        # is the planetary continuum absorbed by the modulation, which costs
        # (C_P/C_S) * L_S at the line (paper Sect. 4.1 collinearity limit).
        expected = sgf_toy_line_estimate(L_P, 0.0, L_S, C_S, C_P)
        self.assertAlmostEqual(got, expected, places=5)
        self.assertGreater(got / L_P, 0.95)

    def test_no_negative_continuum_around_line(self):
        resid = self.res.residual_cube[:, 1, 1]
        near = np.r_[resid[LINE_LO - 30 : LINE_LO - 2], resid[LINE_HI + 3 : LINE_HI + 30]]
        # SGF leaves a negative trough ~ -1.6 here (TestSgfSelfSubtraction);
        # LPM must stay within the absorbed-continuum level.
        self.assertGreater(np.min(near), -C_P - 1e-6)

    def test_sgf_vs_lpm_contrast(self):
        center = (LINE_LO + LINE_HI) // 2
        sgf = sgf_subtract(self.cube, self.star * self.modulation, window=WINDOW, degree=1)
        sgf_line = sgf.residual_cube[center, 1, 1]
        lpm_line = self.res.residual_cube[center, 1, 1]
        self.assertGreater(lpm_line, sgf_line)
        self.assertLess(sgf_line / L_P, 0.93)  # SGF visibly self-subtracts
        self.assertGreater(lpm_line / L_P, 0.96)  # LPM preserves the line

    def test_coefficient_energy_share(self):
        share = lpm_coefficient_energy_share(self.res.coeffs)
        self.assertEqual(share.size, 4)
        self.assertAlmostEqual(float(np.sum(share)), 1.0, places=10)
        # The injected modulation is quadratic: degrees 1-2 dominate.
        self.assertGreater(float(share[0] + share[1]), 0.9)


class TestLpmMseOracle(unittest.TestCase):
    """Monte-Carlo residual MSE matches Eq. B.5 (paper Fig. 6)."""

    def test_mse_decomposition(self):
        rng = np.random.default_rng(7)
        wave, star, planet = toy_spectra()
        # Star slightly OUTSIDE the degree-2 model space -> underfit term.
        x = np.linspace(-1.0, 1.0, NZ)
        stellar_spaxel = star * (1.0 + 0.03 * x**3)
        sigma = 0.5
        degree = 2
        design = lpm_design_matrix(wave, star, degree=degree)
        expected = lpm_mse_terms(design, stellar_spaxel, planet, sigma)

        pinv = np.linalg.pinv(design)
        n_mc = 400
        errors = np.empty(n_mc)
        for i in range(n_mc):
            data = stellar_spaxel + planet + rng.normal(0.0, sigma, NZ)
            stellar_est = design @ (pinv @ data)
            planet_est = data - stellar_est
            errors[i] = float(np.sum((planet_est - planet) ** 2))
        mc_mse = float(np.mean(errors))
        self.assertAlmostEqual(mc_mse / expected["total"], 1.0, delta=0.05)

    def test_degree_tradeoff_direction(self):
        """Star underfit decreases and noise+planet overfit increase with d."""

        wave, star, planet = toy_spectra()
        x = np.linspace(-1.0, 1.0, NZ)
        stellar_spaxel = star * (1.0 + 0.05 * x + 0.02 * x**2 + 0.03 * x**5)
        sigma = 0.5
        under, over = [], []
        for degree in (1, 3, 5, 7):
            design = lpm_design_matrix(wave, star, degree=degree)
            terms = lpm_mse_terms(design, stellar_spaxel, planet, sigma)
            under.append(terms["star_underfit"])
            over.append(terms["planet_overfit"])
        self.assertTrue(all(a >= b - 1e-9 for a, b in zip(under, under[1:])))
        self.assertTrue(all(a <= b + 1e-9 for a, b in zip(over, over[1:])))


class TestReferenceSpectrum(unittest.TestCase):
    def test_selection_excludes_core_and_faint(self):
        wave, star, _ = toy_spectra()
        ny = nx = 21
        yy, xx = np.indices((ny, nx), dtype=np.float64)
        halo = np.exp(-np.hypot(yy - 10, xx - 10) / 3.0)  # bright core, faint edges
        cube = star[:, None, None] * halo[None, :, :]
        keep, qc = select_reference_spaxels(cube)
        self.assertFalse(keep[10, 10])  # core > 0.1 Fmax
        self.assertFalse(keep[0, 0])  # corner < 0.01 Fmax
        self.assertGreater(qc["n_spaxels_kept"], 0)
        # Kept spaxels reproduce the stellar spectrum shape.
        ref = reference_spectrum(cube, keep)
        ratio = ref / star
        self.assertLess(np.nanstd(ratio) / np.nanmean(ratio), 1e-10)

    def test_companion_exclusion(self):
        wave, star, planet = toy_spectra()
        ny = nx = 21
        yy, xx = np.indices((ny, nx), dtype=np.float64)
        halo = 0.05 * np.exp(-np.hypot(yy - 10, xx - 10) / 6.0)
        cube = star[:, None, None] * halo[None, :, :]
        cube[:, 4, 15] += planet
        keep, _ = select_reference_spaxels(cube, exclude_yx=[(4, 15)], exclude_radius_px=2.5)
        self.assertFalse(keep[4, 15])
        self.assertFalse(keep[5, 16])

    def test_nan_fill(self):
        arr = np.array([[1.0], [np.nan], [3.0], [np.nan]])
        filled, finite = fill_nan_along_axis0(arr)
        self.assertAlmostEqual(filled[1, 0], 2.0)
        self.assertAlmostEqual(filled[3, 0], 3.0)  # edge: nearest finite
        self.assertTrue(finite[0, 0] and not finite[1, 0])


class TestLpmMaskDefaults(unittest.TestCase):
    def test_fit_mask_excludes_standard_lines(self):
        wave = np.linspace(4800.0, 9300.0, 3600)
        ref = np.ones_like(wave)
        mask = lpm_fit_mask(wave, ref)
        for center in (6562.8, 4861.3, 8446.0):
            idx = int(np.argmin(np.abs(wave - center)))
            self.assertFalse(mask[idx])
        self.assertGreater(int(np.sum(mask)), 3000)


if __name__ == "__main__":
    unittest.main()
