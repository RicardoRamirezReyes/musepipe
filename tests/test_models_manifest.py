"""WP-G3R-2: external-library manifest mechanics, synthetic fixtures only."""

import tempfile
import unittest
from pathlib import Path

from musepipe.models.manifest import (
    MANIFEST_NAME,
    library_root,
    parse_manifest,
    verify_manifest,
    write_manifest,
)


class ManifestTests(unittest.TestCase):
    def _family(self, tmp):
        family = Path(tmp) / "fam"
        family.mkdir()
        (family / "a.npz").write_bytes(b"alpha-bytes")
        (family / "sub").mkdir()
        (family / "sub" / "b.npz").write_bytes(b"beta-bytes")
        write_manifest(family, ["a.npz", "sub/b.npz"])
        return family

    def test_write_then_verify_ok(self):
        with tempfile.TemporaryDirectory() as tmp:
            family = self._family(tmp)
            info = verify_manifest(family)
            self.assertEqual(info["n_files"], 2)
            self.assertEqual(len(info["sha256_of_manifest"]), 64)
            self.assertIn("verified_utc", info)

    def test_corrupt_file_detected(self):
        with tempfile.TemporaryDirectory() as tmp:
            family = self._family(tmp)
            (family / "a.npz").write_bytes(b"tampered")
            with self.assertRaises(RuntimeError) as ctx:
                verify_manifest(family)
            self.assertIn("a.npz", str(ctx.exception))

    def test_missing_file_detected(self):
        with tempfile.TemporaryDirectory() as tmp:
            family = self._family(tmp)
            (family / "sub" / "b.npz").unlink()
            with self.assertRaises(RuntimeError) as ctx:
                verify_manifest(family)
            self.assertIn("sub/b.npz", str(ctx.exception))

    def test_missing_manifest_detected(self):
        with tempfile.TemporaryDirectory() as tmp:
            family = Path(tmp) / "empty"
            family.mkdir()
            with self.assertRaises(RuntimeError):
                verify_manifest(family)

    def test_parse_rejects_malformed(self):
        with self.assertRaises(RuntimeError):
            parse_manifest("not-a-sha  file.npz\n")
        with self.assertRaises(RuntimeError):
            parse_manifest("deadbeef\n")  # missing path

    def test_parse_ignores_blank_and_comments(self):
        text = "# header\n\n" + ("0" * 64) + "  x.npz\n"
        entries = parse_manifest(text)
        self.assertEqual(entries, [("0" * 64, "x.npz")])

    def test_write_manifest_refuses_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(RuntimeError):
                write_manifest(tmp, [])

    def test_write_manifest_missing_source_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(RuntimeError):
                write_manifest(tmp, ["ghost.npz"])

    def test_manifest_name_constant(self):
        self.assertEqual(MANIFEST_NAME, "MANIFEST.sha256")


class LibraryRootTests(unittest.TestCase):
    def test_missing_key_raises(self):
        with self.assertRaises(RuntimeError) as ctx:
            library_root({}, project_root=".")
        self.assertIn("g3_libraries_root", str(ctx.exception))

    def test_resolves_relative_to_project_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "libs").mkdir()
            resolved = library_root({"g3_libraries_root": "libs"}, project_root=root)
            self.assertEqual(resolved, (root / "libs").resolve())

    def test_missing_directory_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(RuntimeError) as ctx:
                library_root({"g3_libraries_root": "nope"}, project_root=tmp)
            self.assertIn("does not exist", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
