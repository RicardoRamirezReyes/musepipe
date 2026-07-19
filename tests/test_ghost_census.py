import unittest

import numpy as np

from musepipe.qc.ghost_census import (
    band_image,
    blob_fringing,
    box_spectrum,
    compute_ghost_census,
    line_ghost_strip,
)


STEP = 1.25
# blue (4800-5500) + Halpha (6540-6590) coverage in one axis.
WAVES = np.arange(4800.0, 6620.0, STEP)
NY = NX = 60
COMPANION = (40.0, 20.0)
PRIMARY = (30.0, 30.0)


def _continuum(waves):
    return 1000.0 * (1.0 + 0.2 * (waves - waves[0]) / (waves[-1] - waves[0]))


def clean_cube(seed=1, noise=0.01):
    rng = np.random.default_rng(seed)
    cont = _continuum(WAVES)
    cube = cont[:, None, None] * (1.0 + rng.normal(0.0, noise, (WAVES.size, NY, NX)))
    # a compact primary so the field is not perfectly flat
    yy, xx = np.mgrid[0:NY, 0:NX]
    psf = np.exp(-(((yy - PRIMARY[0]) ** 2 + (xx - PRIMARY[1]) ** 2) / (2 * 2.0 ** 2)))
    cube += 5000.0 * psf[None, :, :]
    return cube.astype(np.float64)


class TestBandImageAndBox(unittest.TestCase):
    def test_band_image_shape_and_full_range(self):
        cube = clean_cube()
        img = band_image(cube, WAVES, None)
        self.assertEqual(img.shape, (NY, NX))
        ha = band_image(cube, WAVES, (6540.0, 6590.0))
        self.assertEqual(ha.shape, (NY, NX))

    def test_box_spectrum_length(self):
        cube = clean_cube()
        spec = box_spectrum(cube, COMPANION)
        self.assertEqual(spec.size, WAVES.size)


class TestLineGhostStrip(unittest.TestCase):
    def test_clean_field_no_strip(self):
        img = band_image(clean_cube(), WAVES, None)
        res = line_ghost_strip(img, COMPANION, exclude=((PRIMARY[0], PRIMARY[1], 6.0),))
        self.assertFalse(res["detected"])
        self.assertLess(abs(res["max_excess_sigma"]), 3.0)

    def test_injected_row_strip_detected(self):
        cube = clean_cube()
        img = band_image(cube, WAVES, None)
        ring = line_ghost_strip(img, COMPANION)["ring_sigma"]
        icy = int(round(COMPANION[0]))
        img[icy, :] += 8.0 * ring  # bright IFU/slice strip through B's row
        res = line_ghost_strip(img, COMPANION, exclude=((PRIMARY[0], PRIMARY[1], 6.0),))
        self.assertTrue(res["detected"])
        self.assertGreater(res["row_excess_sigma"], 3.0)

    def test_injected_column_strip_detected(self):
        cube = clean_cube()
        img = band_image(cube, WAVES, None)
        ring = line_ghost_strip(img, COMPANION)["ring_sigma"]
        icx = int(round(COMPANION[1]))
        img[:, icx] += 8.0 * ring
        res = line_ghost_strip(img, COMPANION, exclude=((PRIMARY[0], PRIMARY[1], 6.0),))
        self.assertTrue(res["detected"])
        self.assertGreater(res["col_excess_sigma"], 3.0)


class TestBlobFringing(unittest.TestCase):
    def test_clean_no_fringing(self):
        # white noise gives peak/median ~8 but a tiny variance fraction -> none.
        spec = box_spectrum(clean_cube(), COMPANION)
        res = blob_fringing(WAVES, spec)
        self.assertFalse(res["detected"])
        self.assertEqual(res["verdict"], "none")
        self.assertLess(res["variance_fraction"], 0.15)

    def test_injected_fringing_detected(self):
        rng = np.random.default_rng(0)
        cont = _continuum(WAVES)
        fringe = 1.0 + 0.3 * np.sin(2 * np.pi * WAVES / 30.0)
        spec = cont * fringe * (1.0 + rng.normal(0.0, 0.01, WAVES.size))
        res = blob_fringing(WAVES, spec)
        self.assertTrue(res["detected"])
        self.assertGreater(res["peak_ratio"], 5.0)
        self.assertGreater(res["variance_fraction"], 0.15)
        self.assertAlmostEqual(res["peak_period_A"], 30.0, delta=3.0)


class TestComputeGhostCensus(unittest.TestCase):
    def test_clean_cube_reports_none(self):
        qc = compute_ghost_census(clean_cube(), WAVES, COMPANION, primary_yx=PRIMARY)
        self.assertEqual(qc["line_ghost_strip"], "none")
        self.assertEqual(qc["blob_fringing"], "none")
        self.assertIn("white", qc["strip_by_band"])
        self.assertIn("halpha", qc["strip_by_band"])
        self.assertIn("blue", qc["strip_by_band"])

    def test_injected_strip_cube_reports_detected(self):
        cube = clean_cube()
        icy = int(round(COMPANION[0]))
        img0 = band_image(cube, WAVES, None)
        ring = line_ghost_strip(img0, COMPANION)["ring_sigma"]
        cont = _continuum(WAVES)
        cube[:, icy, :] += 8.0 * ring  # strip present in every channel
        qc = compute_ghost_census(cube, WAVES, COMPANION, primary_yx=PRIMARY)
        self.assertEqual(qc["line_ghost_strip"], "detected")


if __name__ == "__main__":
    unittest.main()
