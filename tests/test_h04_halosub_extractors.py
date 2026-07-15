"""WP-H2 · E4 in-memory extractors for the sgf/lpm methods.

Checks the two properties the adapter must guarantee:

1. **Exactness of the linearity optimization**: subtracting only the injection
   delta on its spatial support (with the fixed base reference) reproduces the
   brute-force subtraction of the full injected cube.
2. **Throughput semantics**: a masked-line Gaussian injection is recovered
   ~fully by lpm and slightly less by sgf (narrow line, R << 1), with
   lpm >= sgf — the ordering G1/E3 will consume.
"""

import unittest
from types import SimpleNamespace

import numpy as np

from musepipe.halosub import (
    lpm_subtract,
    reference_spectrum,
    select_reference_spaxels,
    sgf_subtract,
)
from musepipe.injection import InjectionSource, inject
from musepipe.stages.stage_h04_extractors import build_halosub_extractors
from tests.test_optimal_analytic import constant_model_doc

NZ = 241
WAVE = np.linspace(6400.0, 6700.0, NZ)
DL = float(np.median(np.diff(WAVE)))
LINE_CENTER_A = 6563.0
LSF_FWHM_A = 2.6
PRIMARY_YX = (30.0, 30.0)
COMPANION_YX = (30.0, 44.0)
CONTROL_YX = (44.0, 30.0)  # same radius, different angle
INJECTED_FLUX = 400.0

X04_CFG = {
    "halosub_exclude_radius_px": 4.0,
    "x04_aperture_correction": "psf_growth_curve",
    "x04_bad_windows_A": [],
    "sgf_window": 101,
    "sgf_degree": 1,
}
X05_CFG = {
    "halosub_exclude_radius_px": 4.0,
    "x05_aperture_correction": "psf_growth_curve",
    "x05_bad_windows_A": [],
    "lpm_degree": 4,
    "lpm_masked_lines_A": [[LINE_CENTER_A, 15.0]],
}
BOX3 = {"name": "box3", "kind": "box", "size": 3}


def base_scene():
    yy, xx = np.indices((64, 64), dtype=np.float64)
    halo = np.exp(-np.hypot(yy - PRIMARY_YX[0], xx - PRIMARY_YX[1]) / 8.0)
    cont = 100.0 + 10.0 * (WAVE - WAVE[0]) / (WAVE[-1] - WAVE[0])
    x = 2.0 * (WAVE - WAVE[0]) / (WAVE[-1] - WAVE[0]) - 1.0
    chroma = 1.0 + 0.04 * x + 0.02 * x**2
    return (cont * chroma)[:, None, None] * halo[None, :, :]


def make_extractors(base):
    return build_halosub_extractors(
        base_cube=base,
        wave_A=WAVE,
        star_yx=PRIMARY_YX,
        companion_yx=COMPANION_YX,
        psf_model=constant_model_doc(),
        run_id="synthetic_h04",
        x04_cfg=dict(X04_CFG),
        x05_cfg=dict(X05_CFG),
        box3=BOX3,
    )


def injected_cube(base, position_yx):
    source = InjectionSource(
        y=float(position_yx[0]),
        x=float(position_yx[1]),
        total_line_flux=INJECTED_FLUX,
        line_center_A=LINE_CENTER_A,
        line_fwhm_A=LSF_FWHM_A,
        label="test",
    )
    return inject(base, [source], wavelengths_A=WAVE, psf_model=constant_model_doc(), copy=True)


def recovered_flux(product):
    wave = np.asarray(product.wave_A)
    flux = np.asarray(product.flux)
    in_line = np.abs(wave - LINE_CENTER_A) <= 4.0 * LSF_FWHM_A
    return float(np.nansum(np.where(np.isfinite(flux), flux, 0.0)[in_line]) * DL)


class HalosubExtractorTests(unittest.TestCase):
    def setUp(self):
        self.base = base_scene()
        self.extractors = make_extractors(self.base)
        self.case = SimpleNamespace(position_y=CONTROL_YX[0], position_x=CONTROL_YX[1])
        self.cube_inj = injected_cube(self.base, CONTROL_YX)

    def test_registry_and_headers(self):
        self.assertEqual(set(self.extractors), {"sgf", "lpm"})
        for method, extractor in self.extractors.items():
            product = extractor(self.cube_inj, WAVE, self.case, method, {})
            self.assertEqual(product.header["METHOD"], method)
            self.assertTrue(str(product.header["BKGMODE"]).startswith(f"{method}_residual"))

    def test_delta_path_matches_bruteforce(self):
        keep, _ = select_reference_spaxels(
            self.base, exclude_yx=[COMPANION_YX], exclude_radius_px=4.0
        )
        s_hat = reference_spectrum(self.base, keep)
        brute = {
            "sgf": sgf_subtract(self.cube_inj, s_hat, window=101, degree=1).residual_cube,
            "lpm": lpm_subtract(
                self.cube_inj, WAVE, s_hat, degree=4,
                line_windows_A=((LINE_CENTER_A, 15.0),),
            ).residual_cube,
        }
        for method, extractor in self.extractors.items():
            product = extractor(self.cube_inj, WAVE, self.case, method, {})
            # Compare through the SAME box3 aperture on the brute-force cube:
            # the extractor's residual is not exposed, but the box3 sum around
            # the injection is a complete witness of the local residual.
            y, x = int(CONTROL_YX[0]), int(CONTROL_YX[1])
            brute_box = np.nansum(brute[method][:, y - 1 : y + 2, x - 1 : x + 2], axis=(1, 2))
            apcorr = np.asarray(product.apcorr)
            raw = np.asarray(product.flux) / apcorr
            good = np.isfinite(raw) & np.isfinite(brute_box)
            self.assertGreater(int(np.sum(good)), NZ // 2)
            np.testing.assert_allclose(raw[good], brute_box[good], rtol=1e-7, atol=1e-9)

    def test_recovery_ordering(self):
        rec = {}
        for method, extractor in self.extractors.items():
            product = extractor(self.cube_inj, WAVE, self.case, method, {})
            rec[method] = recovered_flux(product) / INJECTED_FLUX
        # Narrow line (R ~ 0.014): sgf loses a little, lpm keeps ~all.
        self.assertGreater(rec["lpm"], 0.9)
        self.assertLess(rec["lpm"], 1.1)
        self.assertGreater(rec["sgf"], 0.7)
        self.assertLess(rec["sgf"], 1.05)
        self.assertGreaterEqual(rec["lpm"], rec["sgf"] - 1e-6)

    def test_no_injection_returns_base_residual(self):
        product = self.extractors["lpm"](self.base, WAVE, self.case, "lpm", {})
        flux = np.asarray(product.flux)
        # Base scene is a pure smooth halo: residual at the control ~ 0.
        self.assertLess(abs(float(np.nanmedian(flux))), 1e-6)


if __name__ == "__main__":
    unittest.main()
