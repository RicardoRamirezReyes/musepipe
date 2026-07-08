"""G1 §7: contract for the block covariance output (shapes, blocks, keys)."""

import unittest

import numpy as np

from musepipe.covariance import spectral_covariance_blocks


class CovarianceBlocksTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(11)
        self.spectra = rng.normal(size=(7, 3681))  # 7 controls, MUSE channel count

    def test_block_partition_and_keys(self):
        out = spectral_covariance_blocks(self.spectra, block_size=200, max_lag=15)
        self.assertEqual(out["n_controls"], 7)
        self.assertEqual(out["block_size"], 200)
        # ceil(3681/200) = 19 blocks (last one has 81 channels >= 3)
        self.assertEqual(len(out["blocks"]), 19)
        for b in out["blocks"]:
            self.assertEqual(set(b) >= {"z0", "z1", "rho_1", "corr_length_channels", "n_eff_over_n", "rho"}, True)
            self.assertEqual(b["rho"].shape, (16,))  # max_lag + 1
            self.assertLessEqual(b["z1"], 3681)
        self.assertTrue(np.isfinite(out["corr_length_channels_median"]))

    def test_blocks_are_contiguous_and_cover_all_channels(self):
        out = spectral_covariance_blocks(self.spectra, block_size=200, max_lag=10)
        self.assertEqual(out["blocks"][0]["z0"], 0)
        self.assertEqual(out["blocks"][-1]["z1"], 3681)
        for prev, nxt in zip(out["blocks"], out["blocks"][1:]):
            self.assertEqual(prev["z1"], nxt["z0"])

    def test_rejects_1d_input(self):
        with self.assertRaises(ValueError):
            spectral_covariance_blocks(np.zeros(3681))


if __name__ == "__main__":
    unittest.main()
