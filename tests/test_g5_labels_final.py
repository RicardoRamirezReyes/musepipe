"""G5 §7 V3: final label re-verification over the physical-properties table."""

import csv
import tempfile
import unittest
from pathlib import Path

from musepipe.characterization import build_characterization
from musepipe.models import validate_label
from tests._g5_fixture import make_run


class G5LabelsFinalTests(unittest.TestCase):
    def test_all_property_rows_have_valid_labels(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            rid = make_run(root)
            build_characterization(rid, project_root=str(root))
            table = root / "runs" / rid / "report" / "characterization" / "final_physical_properties.csv"
            rows = list(csv.DictReader(table.open()))
            self.assertTrue(rows)
            for r in rows:
                self.assertTrue(validate_label(r["property"], r["label"]),
                                msg=f"{r['property']} -> {r['label']}")

    def test_mass_never_direct_measurement(self):
        self.assertFalse(validate_label("mass", "direct_measurement"))
        self.assertTrue(validate_label("mass", "evolutionary_model_dependent"))


if __name__ == "__main__":
    unittest.main()
