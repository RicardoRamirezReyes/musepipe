import json
import tempfile
import unittest
from pathlib import Path

from astropy.io import fits

from musepipe.reduction.perexp_inputs import PerExposureInputError, build_perexp_inputs


def _fits(path: Path, tag: str, **header_values: str) -> Path:
    header = fits.Header()
    header["HIERARCH ESO PRO CATG"] = tag
    for key, value in header_values.items():
        header[key] = value
    fits.PrimaryHDU(header=header).writeto(path)
    return path


class PerExposureInputTests(unittest.TestCase):
    def _manifest(self, root: Path, *, omit_one: bool = False, merged_lsf: bool = False) -> Path:
        raw_a = _fits(root / "MUSE.2022-08-28T23:00:00.000.fits", "OBJECT", **{"DATE-OBS": "2022-08-28T23:00:00.000"})
        raw_b = _fits(root / "MUSE.2022-08-28T23:10:00.000.fits", "OBJECT", **{"DATE-OBS": "2022-08-28T23:10:00.000"})
        pixtables = []
        for raw in (raw_a, raw_b):
            for ifu in range(24):
                if omit_one and raw == raw_b and ifu == 23:
                    continue
                pixtables.append(_fits(root / f"{raw.stem}_{ifu}.fits", "PIXTABLE_OBJECT", **{"HIERARCH ESO PRO REC1 RAW1 NAME": raw.name}))
        manifest = {
            "run_id": "test_run",
            "night": "2022-08-28",
            "status": "complete",
            "association": {
                "science": [str(raw_a), str(raw_b)],
                "calibrations": {
                    "static": [
                        str(_fits(root / "extinct.fits", "EXTINCT_TABLE")),
                        str(_fits(root / "filter.fits", "FILTER_LIST")),
                    ]
                },
            },
            "products": {
                "PIXTABLE_OBJECT": [str(path) for path in pixtables],
                "LSF_PROFILE": [str(_fits(root / "lsf_merged.fits", "LSF_PROFILE"))]
                if merged_lsf
                else [str(_fits(root / f"lsf_{ifu}.fits", "LSF_PROFILE")) for ifu in range(24)],
                "STD_RESPONSE": [str(_fits(root / "response.fits", "STD_RESPONSE"))],
                "STD_TELLURIC": [str(_fits(root / "telluric.fits", "STD_TELLURIC"))],
            },
        }
        path = root / "products_manifest.json"
        path.write_text(json.dumps(manifest), encoding="utf-8")
        return path

    def test_builds_explicit_provenance_mapping(self):
        with tempfile.TemporaryDirectory() as tmp:
            exposures, calibrations = build_perexp_inputs([self._manifest(Path(tmp))], run_id="test_run")
        self.assertEqual(exposures["expected_ifus"], 24)
        self.assertEqual([len(row["pixtables"]) for row in exposures["exposures"]], [24, 24])
        self.assertEqual(calibrations["night_calibrations"]["2022-08-28"]["EXTINCT_TABLE"], [str(Path(tmp) / "extinct.fits")])

    def test_rejects_incomplete_provenance_group(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(PerExposureInputError):
                build_perexp_inputs([self._manifest(Path(tmp), omit_one=True)], run_id="test_run")

    def test_accepts_one_merged_lsf_without_changing_ifu_count(self):
        with tempfile.TemporaryDirectory() as tmp:
            exposures, calibrations = build_perexp_inputs(
                [self._manifest(Path(tmp), merged_lsf=True)], run_id="test_run"
            )
        self.assertEqual(exposures["expected_ifus"], 24)
        self.assertEqual(len(calibrations["night_calibrations"]["2022-08-28"]["LSF_PROFILE"]), 1)


if __name__ == "__main__":
    unittest.main()
