import unittest

import numpy as np

from musepipe.psf import evaluate_radial_profile, radial_hybrid_profile, source_mask
from tests.test_stage01c_synthetic import gaussian2d


class PSFHybridSafetyTests(unittest.TestCase):
    def test_radial_hybrid_does_not_absorb_masked_point_source(self):
        shape = (90, 90)
        primary_yx = (45.0, 45.0)
        source_yx = (45.0, 63.0)
        residual = np.zeros(shape, dtype=np.float64)
        residual += gaussian2d(*shape, source_yx[0], source_yx[1], 100.0, 1.2)
        mask = source_mask(shape, [source_yx], radius_px=5.0)

        radii, profile = radial_hybrid_profile(
            residual,
            primary_yx,
            mask=mask,
            smoothing_scale_px=8.0,
        )
        hybrid = evaluate_radial_profile(shape, primary_yx, radii, profile)

        aperture = source_mask(shape, [source_yx], radius_px=3.0)
        injected_flux = float(np.sum(residual[aperture]))
        absorbed_flux = float(np.sum(hybrid[aperture]))
        self.assertLess(abs(absorbed_flux) / injected_flux, 0.1)


if __name__ == "__main__":
    unittest.main()
