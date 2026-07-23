"""G1 covariance npz must carry the schema G3 consumes.

`musepipe.models.observed.rebin_for_fit` reads `block_bounds` (Nx2) and
`n_eff_over_n_by_block`; the legacy-only file (block_z0/block_z1/n_eff_over_n)
makes G3 fail with KeyError: 'block_bounds'. This locks the bridge in.
"""

import importlib.util
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("run_g1", ROOT / "scripts" / "run_g1.py")
assert SPEC is not None and SPEC.loader is not None
RUN_G1 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUN_G1)


class G1CovarianceSchemaTests(unittest.TestCase):
    def test_payload_has_block_bounds_and_neff_by_block(self):
        cov = {"blocks": [
            {"z0": 0, "z1": 200, "corr_length_channels": 3.1, "n_eff_over_n": 0.42, "rho": 0.18},
            {"z0": 200, "z1": 400, "corr_length_channels": 2.7, "n_eff_over_n": 0.55, "rho": 0.11},
        ]}
        payload = RUN_G1.covariance_npz_payload(cov)
        # New schema consumed by G3.
        self.assertIn("block_bounds", payload)
        self.assertIn("n_eff_over_n_by_block", payload)
        np.testing.assert_array_equal(payload["block_bounds"], np.array([[0, 200], [200, 400]]))
        np.testing.assert_allclose(payload["n_eff_over_n_by_block"], [0.42, 0.55])
        # Legacy schema still present.
        np.testing.assert_array_equal(payload["block_z0"], [0, 200])
        np.testing.assert_array_equal(payload["block_z1"], [200, 400])
        # block_bounds last upper bound equals the covered channel count (the
        # invariant _block_upper_waves / check_g3_real_inputs assert on).
        self.assertEqual(int(payload["block_bounds"][-1, 1]), 400)


if __name__ == "__main__":
    unittest.main()
