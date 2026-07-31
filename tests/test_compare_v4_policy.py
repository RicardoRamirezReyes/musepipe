import unittest

from musepipe.stages.stage_x10_compare import compare_methods
from tests.test_compare_verdicts import clean_controls, make_products, make_wave, synthetic_g1


def row_for(rows, pair, band):
    return next(row for row in rows if row["pair"] == pair and row["band"] == band)


class CompareV4PolicyTests(unittest.TestCase):
    def test_continuum_statistics_do_not_depend_on_halpha_throughput(self):
        products = make_products({"psffit": [(4900.0, 5400.0, 3.0)]})
        controls = clean_controls()
        first = synthetic_g1(throughput={"psffit": 0.7, "optimal_psfsub": 0.68})
        second = synthetic_g1(throughput={"psffit": 0.4, "optimal_psfsub": 0.95})

        rows_first, _controls, _qc = compare_methods(products, controls, g1_inputs=first)
        rows_second, _controls, _qc = compare_methods(products, controls, g1_inputs=second)
        a = row_for(rows_first, "psffit_vs_optimal_psfsub", "B1")
        b = row_for(rows_second, "psffit_vs_optimal_psfsub", "B1")

        self.assertEqual(a["comparison_mode"], "raw_total_continuum")
        self.assertFalse(a["throughput_applied_i"])
        self.assertFalse(a["throughput_applied_j"])
        self.assertEqual(a["flux_i_corr"], a["flux_i"])
        self.assertEqual(a["t_stat"], b["t_stat"])

    def test_line_statistics_do_depend_on_measured_throughput(self):
        products = make_products({"psffit": [(6553.0, 6573.0, 20.0)]})
        controls = clean_controls()
        first = synthetic_g1(throughput={"psffit": 0.7, "optimal_psfsub": 0.68})
        second = synthetic_g1(throughput={"psffit": 0.4, "optimal_psfsub": 0.95})

        rows_first, _controls, _qc = compare_methods(products, controls, g1_inputs=first)
        rows_second, _controls, _qc = compare_methods(products, controls, g1_inputs=second)
        a = row_for(rows_first, "psffit_vs_optimal_psfsub", "LHa")
        b = row_for(rows_second, "psffit_vs_optimal_psfsub", "LHa")

        self.assertEqual(a["comparison_mode"], "local_continuum_subtracted_throughput_line")
        self.assertTrue(a["throughput_applied_i"])
        self.assertTrue(a["throughput_applied_j"])
        self.assertNotEqual(a["flux_i_corr"], b["flux_i_corr"])

    def test_local_continuum_removal_absorbs_a_constant_pedestal_for_lines(self):
        wave = make_wave()
        products = make_products({"psffit": [(float(wave[0]), float(wave[-1]), 50.0)]})
        controls = clean_controls()

        rows, _controls, _qc = compare_methods(products, controls)
        line = row_for(rows, "psffit_vs_optimal_psfsub", "LHa")

        self.assertGreaterEqual(line["p_value"], 0.0455)

    def test_sgf_b6_divergence_cannot_drive_continuum_verdict(self):
        verdicts = {method: "validated" for method in (
            "aperture", "optimal_ls", "optimal_psfsub", "psffit", "sgf", "lpm"
        )}
        g1 = synthetic_g1(verdicts=verdicts)
        products = make_products({"sgf": [(8600.0, 9100.0, 100.0)]})

        rows, _controls, qc = compare_methods(products, clean_controls(), g1_inputs=g1)
        diagnostic = row_for(rows, "psffit_vs_sgf", "B6")

        self.assertEqual(diagnostic["observable_role"], "diagnostic_noncomparable_continuum")
        self.assertEqual(qc["verdict"], "consistent")

    def test_spatial_red_continuum_still_drives_divergent_continuum(self):
        products = make_products({"psffit": [(7600.0, 8000.0, 5.0), (8600.0, 9100.0, 5.0)]})

        _rows, _controls, qc = compare_methods(products, clean_controls())

        self.assertEqual(qc["verdict"], "divergent_continuum")
        self.assertIn("psffit_vs_optimal_psfsub", qc["primary_pairs_continuum"])


if __name__ == "__main__":
    unittest.main()
