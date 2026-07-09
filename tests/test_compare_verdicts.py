import unittest

import numpy as np

from musepipe.extraction.product import FORMAT_VERSION, SpectrumProduct
from musepipe.stages.stage_x10_compare import METHOD_ORDER, compare_methods


def make_wave():
    return np.arange(4800.0, 9120.0, 5.0, dtype=np.float64)


def make_product(method_label, flux, *, wave=None, flags=None, normrad=25.0, incubesh="samecube"):
    wave = make_wave() if wave is None else np.asarray(wave, dtype=np.float64)
    if flags is None:
        flags = np.zeros(wave.size, dtype=np.int32)
    fits_method = "optimal" if method_label.startswith("optimal") else method_label
    aperture_label = "box3" if method_label == "aperture" else method_label
    return SpectrumProduct(
        wave_A=wave,
        flux=np.asarray(flux, dtype=np.float64),
        flux_err=np.full(wave.size, 4.0, dtype=np.float64),
        flux_err_emp=np.full(wave.size, 4.0, dtype=np.float64),
        apcorr=np.ones(wave.size, dtype=np.float64),
        npix_eff=np.full(wave.size, 9.0, dtype=np.float64),
        flags=np.asarray(flags, dtype=np.int32),
        header={
            "FORMATV": FORMAT_VERSION,
            "METHOD": fits_method,
            "RUNID": "synthetic_x10",
            "SRCPOS_Y": 12.0,
            "SRCPOS_X": 16.0,
            "APERTURE": aperture_label,
            "WFRAME": "topocentric",
            "INCUBE": "cube.fits",
            "INCUBESH": incubesh,
            "NORMRAD": float(normrad),
        },
    )


def make_products(offsets=None):
    wave = make_wave()
    base = np.full(wave.size, 100.0, dtype=np.float64)
    offsets = offsets or {}
    products = {}
    for method in METHOD_ORDER:
        flux = base.copy()
        for lo, hi, value in offsets.get(method, []):
            flux[(wave >= lo) & (wave <= hi)] += float(value)
        products[method] = make_product(method, flux, wave=wave)
    return products


def synthetic_g1(verdicts=None, throughput=None, corr_length=1.0):
    """Synthetic G1 inputs for D1 v2 tests (spec v2 §3.2)."""

    verdicts = verdicts or {
        "psffit": "validated_with_bias",
        "optimal_psfsub": "validated_with_bias",
        "aperture": "rejected",
        "optimal_ls": "rejected",
    }
    throughput = throughput or {}
    tmap = {}
    for method in METHOD_ORDER:
        t_val = throughput.get(method)
        if t_val is None:
            tmap[method] = {"T": 1.0, "err": 0.0, "source": "unavailable_default_1"}
        else:
            tmap[method] = {"T": float(t_val), "err": 0.0, "source": "g1_bias_budget.throughput_loss"}
    return {
        "available": True,
        "method_verdicts": dict(verdicts),
        "throughput_by_method": tmap,
        "covariance": {"corr_length_channels": float(corr_length), "n_eff_over_n": None, "source": "synthetic"},
        "open_issues": [],
        "sources": {"g1_qc": "synthetic"},
    }


def clean_controls(wave=None):
    wave = make_wave() if wave is None else np.asarray(wave, dtype=np.float64)
    levels = np.array([-1.6, -1.1, -0.7, -0.3, 0.3, 0.7, 1.1, 1.6], dtype=np.float64)
    alternating = np.where(np.arange(wave.size) % 2 == 0, 1.0, -1.0)
    base = np.zeros((levels.size, wave.size), dtype=np.float64)
    return {
        "aperture": base.copy(),
        "optimal_ls": base - 0.5 * levels[:, None] * alternating[None, :],
        "optimal_psfsub": base + 0.25 * levels[:, None] * alternating[None, :],
        "psffit": base + levels[:, None] * alternating[None, :],
    }


class CompareVerdictTests(unittest.TestCase):
    def test_identical_products_with_clean_controls_are_consistent(self):
        products = make_products()
        _rows, _controls, qc = compare_methods(products, clean_controls())
        self.assertEqual(qc["verdict"], "consistent")

    def test_continuum_offset_triggers_divergent_continuum(self):
        products = make_products({"psffit": [(4900.0, 5400.0, 5.0), (6100.0, 6400.0, 5.0)]})
        _rows, _controls, qc = compare_methods(products, clean_controls())
        self.assertEqual(qc["verdict"], "divergent_continuum")

    def test_line_only_offset_triggers_divergent_lines(self):
        products = make_products({"psffit": [(6553.0, 6573.0, 20.0)]})
        _rows, _controls, qc = compare_methods(products, clean_controls())
        self.assertEqual(qc["verdict"], "divergent_lines")


if __name__ == "__main__":
    unittest.main()
