import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "run_perexp_scipost", ROOT / "scripts" / "run_perexp_scipost.py"
)
assert SPEC is not None and SPEC.loader is not None
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


class PerExposureScipostTests(unittest.TestCase):
    def test_skymodel_gate_requires_one_sky_spectrum(self):
        RUNNER._gate({"SKY_SPECTRUM": ["sky.fits"]}, "exp1", "skymodel")
        with self.assertRaises(RUNNER.PerExposureExecutionError):
            RUNNER._gate({}, "exp1", "skymodel")

    def test_cube_gate_remains_strict(self):
        products = {
            "DATACUBE_FINAL": ["cube.fits"],
            "IMAGE_FOV": ["image.fits"],
            "SKY_SPECTRUM": ["sky.fits"],
        }
        RUNNER._gate(products, "exp1", "cube,skymodel")
