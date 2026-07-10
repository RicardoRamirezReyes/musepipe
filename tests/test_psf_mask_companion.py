import unittest

import numpy as np

from musepipe.psf import fit_moffat_image, moffat_image, source_mask
from tests.test_stage01c_synthetic import gaussian2d


class PSFMaskCompanionTests(unittest.TestCase):
    def test_companion_mask_prevents_parameter_absorption(self):
        shape = (90, 90)
        primary_yx = (45.0, 45.0)
        companion_yx = (45.0, 61.0)
        base = moffat_image(
            shape,
            primary_yx[0],
            primary_yx[1],
            4.0,
            3.6,
            10.0,
            2.7,
            amplitude=1000.0,
            background=2.0,
        )
        with_companion = base + gaussian2d(*shape, companion_yx[0], companion_yx[1], 250.0, 1.4)

        fit_base = fit_moffat_image(base, center_yx=primary_yx, fit_radius_px=25.0, background=2.0)
        mask = source_mask(shape, [companion_yx], radius_px=5.0)
        fit_masked = fit_moffat_image(
            with_companion,
            center_yx=primary_yx,
            fit_radius_px=25.0,
            mask=mask,
            background=2.0,
        )
        fit_unmasked = fit_moffat_image(
            with_companion,
            center_yx=primary_yx,
            fit_radius_px=25.0,
            background=2.0,
        )

        for key in ("fwhm_maj", "fwhm_min", "beta"):
            rel = abs(fit_masked.params[key] / fit_base.params[key] - 1.0)
            self.assertLess(rel, 0.005, key)

        unmasked_shift = abs(fit_unmasked.params["x0"] - fit_base.params["x0"])
        unmasked_fwhm = abs(fit_unmasked.params["fwhm_maj"] / fit_base.params["fwhm_maj"] - 1.0)
        self.assertGreater(max(unmasked_shift, unmasked_fwhm), 0.02)


if __name__ == "__main__":
    unittest.main()
