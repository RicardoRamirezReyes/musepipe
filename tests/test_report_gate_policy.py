import unittest

from musepipe.report import ACCEPTED_LIMITATIONS, stage_status


class GatePolicyTests(unittest.TestCase):
    def test_accepted_red_check_is_downgraded_to_yellow(self):
        qc = {"m5_stat": {"status": "red"}}
        accepted = ACCEPTED_LIMITATIONS["A4_cube_qc"]
        status, issues, applied = stage_status("A4_cube_qc", qc, accepted=accepted)
        self.assertEqual(status, "yellow")
        self.assertEqual(len(applied), 1)
        self.assertEqual(applied[0]["path"], "m5_stat.status")
        self.assertTrue(any("accepted limitation" in i for i in issues))

    def test_non_accepted_red_still_blocks(self):
        # A different red check in the same stage must stay red.
        qc = {"m5_stat": {"status": "red"}, "m4_sky": {"status": "red"}}
        accepted = ACCEPTED_LIMITATIONS["A4_cube_qc"]
        status, issues, applied = stage_status("A4_cube_qc", qc, accepted=accepted)
        self.assertEqual(status, "red")
        self.assertIn("m4_sky.status=red", issues)
        # the accepted one is still reported (annotated), not hidden
        self.assertTrue(any("m5_stat.status=red [accepted limitation" in i for i in issues))
        self.assertEqual(len(applied), 1)

    def test_e2_t2_and_e4_hierarchy_are_accepted(self):
        s2, _, a2 = stage_status("E2_artifacts", {"t2": {"status": "fail"}},
                                 accepted=ACCEPTED_LIMITATIONS["E2_artifacts"])
        self.assertEqual(s2, "yellow")
        self.assertEqual(a2[0]["path"], "t2.status")
        s4, _, a4 = stage_status("E4_injection", {"checks": {"v4_hierarchy": {"status": "fail"}}},
                                 accepted=ACCEPTED_LIMITATIONS["E4_injection"])
        self.assertEqual(s4, "yellow")
        self.assertEqual(a4[0]["path"], "checks.v4_hierarchy.status")

    def test_d2_v3_accepted_only_for_the_diagnosed_check(self):
        # D2's v3 continuum flag is accepted (diagnosed inter-method halo systematic)...
        acc = ACCEPTED_LIMITATIONS["D2_calibrate"]
        self.assertIn("checks.v3_continuum_stable.ok", acc)
        status, _, applied = stage_status(
            "D2_calibrate", {"checks": {"v3_continuum_stable": {"ok": False}}}, accepted=acc
        )
        self.assertEqual(status, "yellow")
        self.assertEqual(len(applied), 1)

    def test_d2_other_failure_still_blocks(self):
        # ...but a DIFFERENT D2 failure (e.g. a hash/flux check) must still block.
        acc = ACCEPTED_LIMITATIONS["D2_calibrate"]
        status, issues, _ = stage_status(
            "D2_calibrate",
            {"checks": {"v3_continuum_stable": {"ok": False}, "v1_skylines": {"ok": False}}},
            accepted=acc,
        )
        self.assertEqual(status, "red")
        self.assertTrue(any("v1_skylines.ok=fail" in i for i in issues))

    def test_clean_stage_stays_green(self):
        status, issues, applied = stage_status("A4_cube_qc", {"m5_stat": {"status": "green"}},
                                               accepted=ACCEPTED_LIMITATIONS["A4_cube_qc"])
        self.assertEqual(status, "green")
        self.assertEqual(applied, [])


if __name__ == "__main__":
    unittest.main()
