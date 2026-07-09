import unittest

import numpy as np

from musepipe.extraction.scale_check import pair_scale_check
from musepipe.stages.stage_x10_compare import compare_methods
from tests.test_compare_verdicts import clean_controls, make_products, make_wave


class ScaleReconciliationTests(unittest.TestCase):
    def test_x20_offset_in_optimal_ls_degrades_only_its_pairs(self):
        controls = clean_controls()
        controls["optimal_ls"] = controls["optimal_ls"] + 2000.0
        products = make_products()

        _rows, _controls, qc = compare_methods(products, controls)

        for pid, check in qc["scale_check"].items():
            if "optimal_ls" in pid:
                self.assertFalse(check["ok"], pid)
            else:
                self.assertTrue(check["ok"], pid)
        self.assertEqual(qc["verdict"], "consistent")
        self.assertEqual(qc["pairs_degraded"], {})

    def test_all_primary_pairs_with_huge_level_offset_raise_extraction_bug(self):
        controls = clean_controls()
        controls["optimal_psfsub"] = controls["optimal_psfsub"] + 5000.0
        products = make_products()

        with self.assertRaisesRegex(RuntimeError, "extraction-stage convention bug"):
            compare_methods(products, controls)

    def test_pair_scale_check_detects_offset_and_passes_clean(self):
        rng = np.random.default_rng(7)
        base = rng.normal(0.0, 1.0, size=(8, 200))
        clean = pair_scale_check(base + rng.normal(0.0, 1.0, size=base.shape), base)
        self.assertTrue(clean["ok"])
        offset = pair_scale_check(base + 20.0, base)
        self.assertFalse(offset["ok"])
        self.assertGreater(offset["stat"], 100.0)


if __name__ == "__main__":
    unittest.main()
