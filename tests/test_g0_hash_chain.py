"""G0 §7: the hash-chain utility must detect a manipulated (mixed-run) QC."""

import unittest

from musepipe.g0 import verify_hash_chain


class G0HashChainTests(unittest.TestCase):
    def _chain(self):
        return [
            {"stage": "A", "input_sha": None, "output_sha": "aaa"},
            {"stage": "B", "input_sha": "aaa", "output_sha": "bbb"},
            {"stage": "C", "input_sha": "bbb", "output_sha": "ccc"},
        ]

    def test_intact_chain_passes(self):
        result = verify_hash_chain(self._chain())
        self.assertEqual(result["status"], "pass")
        self.assertTrue(all(c["status"] == "pass" for c in result["checks"]))

    def test_manipulated_link_is_detected(self):
        chain = self._chain()
        chain[2]["input_sha"] = "WRONG"  # C claims an input that B did not produce
        result = verify_hash_chain(chain)
        self.assertEqual(result["status"], "fail")
        failed = [c["stage"] for c in result["checks"] if c["status"] == "fail"]
        self.assertEqual(failed, ["C"])

    def test_missing_input_sha_is_not_flagged(self):
        chain = self._chain()
        chain[1]["input_sha"] = None  # stage does not declare its input
        result = verify_hash_chain(chain)
        self.assertEqual(result["status"], "pass")


if __name__ == "__main__":
    unittest.main()
