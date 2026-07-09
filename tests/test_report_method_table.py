import csv
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from musepipe.report import build_table_methods


class ReportMethodTableTests(unittest.TestCase):
    def test_build_table_methods_reads_d1_csv(self):
        with tempfile.TemporaryDirectory() as tmp:
            table_dir = Path(tmp)
            with (table_dir / "method_comparison.csv").open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["pair", "method_i", "band", "z"])
                writer.writeheader()
                writer.writerow(
                    {"pair": "psffit_vs_optimal_psfsub", "method_i": "psffit", "band": "B1", "z": "0.5"}
                )
            run_paths = SimpleNamespace(table_dir=table_dir)
            rows = build_table_methods(run_paths)

        d1_rows = [row for row in rows if row["source"] == "D1_compare"]
        self.assertEqual(len(d1_rows), 1)
        self.assertEqual(d1_rows[0]["pair"], "psffit_vs_optimal_psfsub")
        self.assertEqual(d1_rows[0]["band"], "B1")


if __name__ == "__main__":
    unittest.main()
