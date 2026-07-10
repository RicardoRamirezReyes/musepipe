import unittest

import numpy as np

from musepipe.stages.stage_h02_artifacts import t1_instrumental_coincidence


class H02T1ListsTests(unittest.TestCase):
    def test_signal_on_dirty_stripe_channel_is_detected(self):
        wave = np.arange(6500.0, 6600.0, 1.0)

        result = t1_instrumental_coincidence(
            wave,
            6550.0,
            stripe_channels=[50],
            skyline_waves_A=[6570.0],
            gap_edges_A=[6520.0, 6525.0],
        )

        self.assertEqual(result["nearest_stripe_dch"], 0)
        self.assertTrue(result["coincidence"])
        self.assertEqual(result["by_list"]["stripe"], "fail")
        self.assertEqual(result["status"], "fail")


if __name__ == "__main__":
    unittest.main()
