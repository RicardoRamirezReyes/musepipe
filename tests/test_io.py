"""Tests for musepipe.io.write_csv (O2b: fieldnames became optional)."""

import csv
import os
import tempfile
import unittest

from musepipe.io import write_csv


class WriteCsv(unittest.TestCase):
    def setUp(self):
        fd, self.path = tempfile.mkstemp(suffix=".csv")
        os.close(fd)

    def tearDown(self):
        if os.path.exists(self.path):
            os.remove(self.path)

    def _read(self):
        with open(self.path, "r", newline="", encoding="utf-8") as f:
            return list(csv.reader(f))

    def test_explicit_fieldnames(self):
        rows = [{"a": 1, "b": 2}, {"a": 3, "b": 4}]
        write_csv(self.path, rows, ["a", "b"])
        self.assertEqual(self._read(), [["a", "b"], ["1", "2"], ["3", "4"]])

    def test_fieldnames_default_derives_from_first_row(self):
        rows = [{"a": 1, "b": 2}, {"a": 3, "b": 4}]
        write_csv(self.path, rows)  # no fieldnames -> derived from rows[0]
        self.assertEqual(self._read(), [["a", "b"], ["1", "2"], ["3", "4"]])

    def test_explicit_fieldnames_control_column_order(self):
        rows = [{"a": 1, "b": 2}]
        write_csv(self.path, rows, ["b", "a"])
        self.assertEqual(self._read(), [["b", "a"], ["2", "1"]])

    def test_empty_rows_writes_only_when_fieldnames_given(self):
        write_csv(self.path, [], ["a", "b"])
        self.assertEqual(self._read(), [["a", "b"]])


if __name__ == "__main__":
    unittest.main()
