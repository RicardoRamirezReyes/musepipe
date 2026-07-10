import unittest

from musepipe.stages.stage_h01_detect import HALPHA_REST_A, analyze_halpha_method, classify_h01_verdict
from musepipe.stages.stage_x10_compare import METHOD_ORDER
from tests.test_h01_helpers import h01_controls, h01_product


def method_rows(signal_methods):
    rows = []
    controls = h01_controls(noise=0.0)
    for method in METHOD_ORDER:
        product = h01_product(
            method,
            signal_flux=8.0 if method in signal_methods else 0.0,
            rv_sys_kms=0.0,
            lsf_fwhm_A=2.5,
        )
        rows.append(
            analyze_halpha_method(
                method,
                product,
                controls,
                rv_sys_kms=0.0,
                lsf_fwhm_A=2.5,
                rest_A=HALPHA_REST_A,
            ).row
        )
    return rows


class H01CriterionTests(unittest.TestCase):
    def test_two_admissible_methods_are_detection(self):
        verdict = classify_h01_verdict(method_rows({"psffit", "aperture"}))
        self.assertEqual(verdict["verdict"], "detection")

    def test_one_method_only_is_candidate(self):
        verdict = classify_h01_verdict(method_rows({"psffit"}))
        self.assertEqual(verdict["verdict"], "candidate")

    def test_no_method_is_non_detection(self):
        verdict = classify_h01_verdict(method_rows(set()))
        self.assertEqual(verdict["verdict"], "non_detection")


if __name__ == "__main__":
    unittest.main()
