import importlib.util
import unittest
from pathlib import Path

from musepipe.reduction.esorex_driver import RawRecord


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("reduce_cascade", ROOT / "scripts" / "reduce_cascade.py")
assert SPEC is not None and SPEC.loader is not None
CASCADE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CASCADE)


def _record(name, tag, date_obs, *, catg="CALIB", mode="NFM-AO-N", pro_catg=""):
    return RawRecord(
        path=Path("/raw") / name,
        tag=tag,
        dpr_type=tag,
        dpr_catg=catg,
        pro_catg=pro_catg,
        date_obs=date_obs,
        ins_mode=mode,
        binning="",
        exptime=1.0,
        sha256="",
    )


class ReduceCascadeTests(unittest.TestCase):
    def test_observing_night_uses_noon_utc_boundary(self):
        before_noon = _record("science_a.fits", "OBJECT", "2022-08-29T01:00:00", catg="SCIENCE")
        after_noon = _record("science_b.fits", "OBJECT", "2022-08-29T13:00:00", catg="SCIENCE")
        self.assertEqual(CASCADE.observing_night(before_noon), "2022-08-28")
        self.assertEqual(CASCADE.observing_night(after_noon), "2022-08-29")

    def test_association_excludes_archival_standard_and_uses_archive_bias(self):
        records = [
            _record("science_1.fits", "OBJECT", "2022-08-28T23:30:00", catg="SCIENCE"),
            _record("science_2.fits", "OBJECT", "2022-08-31T00:30:00", catg="SCIENCE"),
            _record("flat_1.fits", "FLAT", "2022-08-28T12:00:00"),
            _record("flat_2.fits", "FLAT", "2022-08-30T12:00:00"),
            _record("arc_1.fits", "ARC", "2022-08-28T12:10:00"),
            _record("arc_2.fits", "ARC", "2022-08-30T12:10:00"),
            _record("illum_1.fits", "ILLUM", "2022-08-28T23:00:00"),
            _record("illum_2.fits", "ILLUM", "2022-08-30T23:00:00"),
            _record("std_raw_1.fits", "STD", "2022-08-29T02:00:00"),
            _record("std_raw_2.fits", "STD", "2022-08-30T23:45:00"),
            _record("std_archive_1.fits", "STD", "2022-08-29T02:00:00", catg=""),
            _record("master_1.fits", "MASTER_BIAS", "2022-08-29T10:00:00", mode="WFM-AO-N", pro_catg="MASTER_BIAS"),
            _record("master_2.fits", "MASTER_BIAS", "2022-08-31T10:00:00", mode="WFM-AO-N", pro_catg="MASTER_BIAS"),
            _record("lsf_nfm.fits", "LSF_PROFILE", "2018-01-01T00:00:00", pro_catg="LSF_PROFILE"),
            _record("lsf_wfm.fits", "LSF_PROFILE", "2018-01-01T00:00:00", mode="WFM-AO-N", pro_catg="LSF_PROFILE"),
        ]
        groups = CASCADE.build_night_associations(records)
        first_records, first = groups["2022-08-28"]
        second_records, second = groups["2022-08-30"]
        self.assertEqual(len([record for record in first_records if record.tag == "OBJECT"]), 1)
        self.assertEqual(first["calibrations"]["STD"], ["/raw/std_raw_1.fits"])
        self.assertEqual(first["calibrations"]["MASTER_BIAS"], ["/raw/master_1.fits"])
        self.assertEqual(first["calibrations"]["LSF_PROFILE"], ["/raw/lsf_nfm.fits"])
        self.assertEqual(second["calibrations"]["STD"], ["/raw/std_raw_2.fits"])
        self.assertEqual(second["calibrations"]["MASTER_BIAS"], ["/raw/master_2.fits"])

    def test_dynamic_gates_require_all_pixtables_and_telluric(self):
        self.assertEqual(CASCADE._gates(23, 1, "scibasic_object"), {"PIXTABLE_OBJECT": 552})
        self.assertEqual(CASCADE._gates(8, 1, "scibasic_object"), {"PIXTABLE_OBJECT": 192})
        self.assertEqual(
            CASCADE._gates(8, 1, "standard"),
            {"STD_RESPONSE": 1, "STD_TELLURIC": 1},
        )


if __name__ == "__main__":
    unittest.main()
