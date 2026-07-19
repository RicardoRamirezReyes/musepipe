import unittest

import numpy as np

from musepipe.qc.stat_emp import (
    aggregate_ratio,
    check_wcs_gate,
    plan_alignment,
    pooled_ratio,
    voxel_stat_emp,
)


class TestVoxelStatEmp(unittest.TestCase):
    def test_var_of_mean_recovers_sigma2_over_n(self):
        # 5 cubes, known per-exposure sigma; var_of_mean should be ~ sigma^2/n.
        rng = np.random.default_rng(0)
        n, ny, nx = 5, 200, 200
        sigma = 3.0
        data = rng.normal(0.0, sigma, (n, ny, nx))
        res = voxel_stat_emp(data, min_n=4)
        # sample variance is unbiased in the MEAN (its median is biased low by the
        # chi-square skew), so compare the mean over voxels to sigma^2 / n.
        mean_var_mean = np.nanmean(res["var_of_mean"])
        self.assertAlmostEqual(mean_var_mean, sigma ** 2 / n, delta=0.05 * sigma ** 2 / n)
        self.assertTrue(np.all(res["n_finite"] == n))

    def test_ratio_recovers_underestimation_factor(self):
        # empirical scatter sigma=3 but DRS STAT claims sigma^2/4 -> ratio ~4.
        rng = np.random.default_rng(1)
        n, ny, nx = 7, 60, 60
        sigma = 3.0
        data = rng.normal(0.0, sigma, (n, ny, nx))
        stat = np.full((n, ny, nx), sigma ** 2 / 4.0)  # DRS underestimates by 4x
        res = voxel_stat_emp(data, stat, min_n=4)
        # pooled ratio is unbiased and recovers the 4x underestimation closely.
        pooled = pooled_ratio(res["sample_var"], res["mean_stat"])
        self.assertAlmostEqual(pooled, 4.0, delta=0.3)
        # the per-voxel median is biased LOW by the small-sample chi-square skew.
        agg = aggregate_ratio(res["ratio"])
        self.assertLess(agg["median"], pooled)

    def test_min_n_masks_low_coverage(self):
        data = np.array([[[1.0]], [[2.0]], [[np.nan]], [[np.nan]]])  # only 2 finite
        res = voxel_stat_emp(data, min_n=4)
        self.assertTrue(np.isnan(res["var_of_mean"][0, 0]))


class TestWcsGate(unittest.TestCase):
    def _hdr(self, cd11=-7e-6, cd22=7e-6, crval3=4749.5, nz=3681):
        return {"NAXIS1": 316, "NAXIS2": 306, "NAXIS3": nz,
                "CD1_1": cd11, "CD1_2": 0.0, "CD2_1": 0.0, "CD2_2": cd22,
                "CRVAL3": crval3, "CD3_3": 1.25, "CRVAL1": 246.6, "CRVAL2": -25.4,
                "CRPIX1": 158.0, "CRPIX2": 152.0}

    def test_gate_passes_translation_only(self):
        h1 = self._hdr(); h2 = dict(self._hdr(), CRPIX1=155.0, CRVAL1=246.61)  # dither
        self.assertTrue(check_wcs_gate([h1, h2])["pass"])

    def test_gate_fails_on_rotation(self):
        h1 = self._hdr()
        h2 = dict(self._hdr(), CD1_2=1e-6, CD2_1=-1e-6)  # rotated CD
        gate = check_wcs_gate([h1, h2])
        self.assertFalse(gate["pass"])
        self.assertIn("CD matrix", gate["reason"])

    def test_gate_fails_on_spectral_mismatch(self):
        h1 = self._hdr()
        h2 = self._hdr(crval3=4800.0)  # different spectral zero point
        self.assertFalse(check_wcs_gate([h1, h2])["pass"])


class TestPlanAlignment(unittest.TestCase):
    def _wcs_hdr(self, crval1, crval2, crpix1, crpix2, ny=100, nx=100):
        return {"NAXIS": 2, "NAXIS1": nx, "NAXIS2": ny, "WCSAXES": 2,
                "CTYPE1": "RA---TAN", "CTYPE2": "DEC--TAN",
                "CRVAL1": crval1, "CRVAL2": crval2, "CRPIX1": crpix1, "CRPIX2": crpix2,
                "CD1_1": -7e-6, "CD1_2": 0.0, "CD2_1": 0.0, "CD2_2": 7e-6,
                "CRVAL3": 4749.5, "CD3_3": 1.25, "NAXIS3": 10}

    def test_integer_offsets_and_overlap(self):
        # cube2 dithered by +5 px in x on sky (shift CRPIX by 5 keeps same sky center).
        h1 = self._wcs_hdr(246.6, -25.4, 50.0, 50.0)
        h2 = self._wcs_hdr(246.6, -25.4, 45.0, 50.0)  # ref sky center is 5 px away in x
        plan = plan_alignment([h1, h2])
        self.assertEqual(plan["offsets"][0], (0, 0))
        # cube2 center-sky maps 5 px off in x relative to ref
        self.assertEqual(plan["offsets"][1][0], 0)
        self.assertEqual(abs(plan["offsets"][1][1]), 5)
        self.assertGreater(plan["overlap_shape"][0], 0)
        self.assertGreater(plan["overlap_shape"][1], 0)


if __name__ == "__main__":
    unittest.main()
