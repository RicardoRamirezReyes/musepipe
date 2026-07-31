import tempfile
import unittest
from pathlib import Path

import numpy as np

from musepipe.stages.stage_h04_injection import (
    _v2_nulls_clean,
    _v5_continuum,
    resolve_continuum_injection,
    stage_h04_paths,
)
from tests.test_compare_verdicts import make_product


def null_row(index, *, position="control1", recovered_flux=0.0, recovered_snr=0.0):
    return {
        "injection_id": f"null{index}",
        "variant": "nominal",
        "method": "aperture",
        "position_label": position,
        "template_factor": 1.0,
        "continuum_mode": "none",
        "input_snr": 0.0,
        "recovered_flux": recovered_flux,
        "recovered_snr": recovered_snr,
    }


def empirical_reference():
    return {
        "aperture": {
            "n_controls": 33,
            "by_factor": {"1": np.arange(33, dtype=np.float64)},
        }
    }


class H04V2NullPolicyTests(unittest.TestCase):
    def test_real_position_never_has_veto_power(self):
        rows = [null_row(0, recovered_flux=1000.0, recovered_snr=100.0, position="real")]
        rows.extend(null_row(i + 1, recovered_flux=10.0) for i in range(10))

        result = _v2_nulls_clean(rows, empirical_reference())

        self.assertEqual(result["status"], "pass")
        self.assertEqual(result["n_rows"], 10)
        self.assertEqual(result["science_position_diagnostics"]["n_rows"], 1)

    def test_gate_uses_empirical_flux_rank_not_formal_snr(self):
        rows = [null_row(i, recovered_flux=10.0, recovered_snr=999.0) for i in range(10)]

        result = _v2_nulls_clean(rows, empirical_reference())

        self.assertEqual(result["status"], "pass")
        self.assertEqual(result["n_extreme"], 0)

    def test_binomial_excess_fails_but_one_extreme_passes(self):
        one = [null_row(0, recovered_flux=100.0)]
        one.extend(null_row(i + 1, recovered_flux=10.0) for i in range(9))
        many = [null_row(i, recovered_flux=100.0) for i in range(10)]

        self.assertEqual(_v2_nulls_clean(one, empirical_reference())["status"], "pass")
        self.assertEqual(_v2_nulls_clean(many, empirical_reference())["status"], "fail")


class H04V2ContinuumTests(unittest.TestCase):
    def test_continuum_is_measured_in_normrad_scale_from_positive_methods(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = stage_h04_paths("synthetic", project_root=tmp)
            paths["paths"].ensure_base_dirs()
            stage_dir = paths["paths"].stage_dir
            wave = np.arange(6490.0, 6640.0, 5.0)
            values = {
                "aperture": 4.0,
                "optimal_ls": -2.0,
                "optimal_psfsub": -1.0,
                "psffit": 8.0,
            }
            names = {
                "aperture": "spec_aperture_object.fits",
                "optimal_ls": "spec_optimal_object.fits",
                "optimal_psfsub": "spec_optimal_psfsub_object.fits",
                "psffit": "spec_psffit_object.fits",
            }
            for method, value in values.items():
                product = make_product(method, np.full(wave.size, value * 2.0), wave=wave)
                product = type(product)(
                    **{**product.__dict__, "apcorr": np.full(wave.size, 2.0)}
                )
                product.write(stage_dir / names[method])

            result = resolve_continuum_injection({}, paths)

        self.assertEqual(result["source"], "median_continuum_preserving_methods")
        self.assertAlmostEqual(result["value"], 6.0)
        self.assertEqual(result["scale"], "normrad_flux_density")

    def test_v5_rejects_zero_or_identical_flat_grid(self):
        rows = []
        for mode in ("none", "flat"):
            rows.append(
                {
                    "method": "aperture",
                    "variant": "nominal",
                    "continuum_mode": mode,
                    "input_snr": 5.0,
                    "template_factor": 1.0,
                    "throughput": 1.0,
                }
            )
        result = _v5_continuum(rows, ["aperture"], {"value": 0.0})
        self.assertEqual(result["status"], "fail")
        self.assertFalse(result["continuum_positive"])
        self.assertFalse(result["flat_distinct_from_none"])


if __name__ == "__main__":
    unittest.main()
