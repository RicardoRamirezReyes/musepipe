import json
import tempfile
import unittest
from pathlib import Path

from astropy.io import fits

from musepipe.reduction.perexp_plan import (
    PerExposurePlanError,
    build_alignment_plan,
    build_scipost_plans,
    load_exposures,
    validate_scipost_plan,
    write_alignment_review,
    write_scipost_plan,
)


def _fits(path: Path, tag: str) -> Path:
    header = fits.Header()
    header["HIERARCH ESO PRO CATG"] = tag
    fits.PrimaryHDU(header=header).writeto(path)
    return path


class PerExposurePlanTests(unittest.TestCase):
    def _inputs(self, root: Path):
        pixtables = [_fits(root / f"pixtable_{index}.fits", "PIXTABLE_OBJECT") for index in range(2)]
        calibration = {
            "night_calibrations": {
                "2022-08-28": {
                    "STD_RESPONSE": [str(_fits(root / "response.fits", "STD_RESPONSE"))],
                    "STD_TELLURIC": [str(_fits(root / "telluric.fits", "STD_TELLURIC"))],
                    "EXTINCT_TABLE": [str(_fits(root / "extinct.fits", "EXTINCT_TABLE"))],
                    "FILTER_LIST": [str(_fits(root / "filter.fits", "FILTER_LIST"))],
                    "LSF_PROFILE": [str(_fits(root / f"lsf_{index}.fits", "LSF_PROFILE")) for index in range(2)],
                }
            }
        }
        exposure = {
            "schema_version": 1,
            "run_id": "test_run",
            "expected_ifus": 2,
            "exposures": [
                {
                    "id": "exp001",
                    "night": "2022-08-28",
                    "pixtables": [str(path) for path in pixtables],
                    "metadata": {"date_obs": "2022-08-28T23:30:00"},
                }
            ],
        }
        return exposure, calibration

    def test_build_validate_and_write_scipost_plan(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            exposure_payload, calibration = self._inputs(root)
            exposure_path = root / "exposures.json"
            exposure_path.write_text(json.dumps(exposure_payload), encoding="utf-8")
            expected_ifus, exposures = load_exposures(exposure_path, run_id="test_run")
            plan = build_scipost_plans(
                run_id="test_run",
                exposures=exposures,
                expected_ifus=expected_ifus,
                calibration_payload=calibration,
                save="cube,skymodel",
            )
            validate_scipost_plan(plan)
            manifest = write_scipost_plan(plan, root / "out")
            payload = json.loads(manifest.read_text(encoding="utf-8"))
            self.assertEqual(payload["status"], "awaiting_perexp_scipost")
            self.assertTrue((root / "out" / "sof" / "muse_scipost_exp001.sof").exists())

    def test_plan_rejects_incomplete_exposure(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            exposure_payload, calibration = self._inputs(root)
            exposure_payload["exposures"][0]["pixtables"] = exposure_payload["exposures"][0]["pixtables"][:1]
            exposure_path = root / "exposures.json"
            exposure_path.write_text(json.dumps(exposure_payload), encoding="utf-8")
            expected_ifus, exposures = load_exposures(exposure_path, run_id="test_run")
            with self.assertRaises(PerExposurePlanError):
                build_scipost_plans(
                    run_id="test_run",
                    exposures=exposures,
                    expected_ifus=expected_ifus,
                    calibration_payload=calibration,
                    save="cube",
                )

    def test_alignment_review_requires_two_images(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaises(PerExposurePlanError):
                build_alignment_plan({"exp001": root / "image_1.fits"})
            image_1 = _fits(root / "image_1.fits", "IMAGE_FOV")
            image_2 = _fits(root / "image_2.fits", "IMAGE_FOV")
            plan = build_alignment_plan({"exp001": image_1, "exp002": image_2})
            review = write_alignment_review(plan, root / "out")
            payload = json.loads(review.read_text(encoding="utf-8"))
            self.assertEqual(payload["status"], "awaiting_alignment_review")


if __name__ == "__main__":
    unittest.main()
