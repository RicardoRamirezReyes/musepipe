"""G2 §7: stage output contract — table columns, QC keys, valid labels/status."""

import unittest

import numpy as np

from musepipe.lines import measure_catalog
from musepipe.stages.stage_g2_measure_lines import TABLE_FIELDS

VALID_STATUS = {"detected", "marginal", "upper_limit", "not_measurable"}
VALID_LABEL = {"direct_measurement", "upper_limit", "not_constrained"}


class StageG2ContractTests(unittest.TestCase):
    def setUp(self):
        self.wave = np.linspace(6400.0, 6720.0, 320)
        rng = np.random.default_rng(0)
        self.flux = 150.0 + rng.normal(0, 2.0, self.wave.size)
        # one strong line at 6562.8
        sig = 2.0
        self.flux += 700.0 / (sig * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((self.wave - 6562.8) / sig) ** 2)
        self.ferr = np.full_like(self.wave, 2.0)
        self.catalog = [
            {"name": "Halpha", "wave_A": 6562.8, "family": "Balmer", "kind": "accretion"},
            {"name": "He I 6678", "wave_A": 6678.15, "family": "He I", "kind": "accretion"},
        ]

    def test_measurements_have_valid_status_and_label(self):
        ms = measure_catalog(self.wave, self.flux, self.ferr, self.catalog, lsf_fwhm_A=2.5, n_mc=100)
        self.assertEqual(len(ms), 2)
        for m in ms:
            self.assertIn(m.status, VALID_STATUS)
            self.assertIn(m.label, VALID_LABEL)
        ha = next(m for m in ms if m.name == "Halpha")
        self.assertEqual(ha.status, "detected")

    def test_row_has_all_table_fields(self):
        ms = measure_catalog(self.wave, self.flux, self.ferr, self.catalog, lsf_fwhm_A=2.5, n_mc=50)
        row = ms[0].to_row()
        for field in TABLE_FIELDS:
            self.assertIn(field, row, msg=f"missing column {field}")

    def test_upper_limit_is_positive_when_line_absent(self):
        flat = 150.0 + np.random.default_rng(1).normal(0, 2.0, self.wave.size)
        ms = measure_catalog(self.wave, flat, self.ferr, self.catalog, lsf_fwhm_A=2.5, n_mc=50, throughput=0.8)
        for m in ms:
            if m.status == "upper_limit":
                self.assertGreater(m.flux_upper_limit_5sigma, 0.0)


if __name__ == "__main__":
    unittest.main()
