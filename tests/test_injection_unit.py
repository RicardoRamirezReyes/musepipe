import unittest

import numpy as np

from musepipe.injection import InjectionSource, inject, integrated_delta_flux


class InjectionUnitTests(unittest.TestCase):
    def test_inject_preserves_integrated_line_flux_in_mini_cube(self):
        wave = np.arange(6550.0, 6576.0, 1.0, dtype=np.float64)
        cube = np.zeros((wave.size, 7, 7), dtype=np.float64)
        psf = np.zeros((7, 7), dtype=np.float64)
        psf[3, 3] = 1.0
        source = InjectionSource(
            y=3.0,
            x=3.0,
            total_line_flux=12.5,
            line_center_A=6562.8,
            line_fwhm_A=2.5,
            psf_image=psf,
        )

        injected = inject(cube, [source], wavelengths_A=wave)
        delta_flux = integrated_delta_flux(injected - cube, wave)

        self.assertAlmostEqual(delta_flux, 12.5, places=10)
        self.assertAlmostEqual(float(np.nansum(injected[:, 3, 3])), float(np.nansum(injected)))

    def test_inject_can_target_one_cube_in_stack(self):
        wave = np.arange(6558.0, 6568.0, 1.0, dtype=np.float64)
        cube = np.zeros((2, wave.size, 5, 5), dtype=np.float64)
        psf = np.zeros((5, 5), dtype=np.float64)
        psf[2, 2] = 1.0
        source = InjectionSource(
            y=2.0,
            x=2.0,
            total_line_flux=3.0,
            line_center_A=6562.8,
            line_fwhm_A=2.5,
            cube_index=1,
            psf_image=psf,
        )

        injected = inject(cube, [source], wavelengths_A=wave)

        self.assertAlmostEqual(integrated_delta_flux(injected[0], wave), 0.0)
        self.assertAlmostEqual(integrated_delta_flux(injected[1], wave), 3.0)


if __name__ == "__main__":
    unittest.main()
