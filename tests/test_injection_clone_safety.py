import tempfile
import unittest
from pathlib import Path

from musepipe.injection import create_run_clone, tree_sha256


class InjectionCloneSafetyTests(unittest.TestCase):
    def test_clone_creation_does_not_modify_base_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = root / "runs" / "science"
            (base / "config").mkdir(parents=True)
            (base / "stages").mkdir()
            (base / "tables").mkdir()
            (base / "config" / "config.json").write_text('{"config": {"run_id": "science"}}', encoding="utf-8")
            (base / "stages" / "stage02.fits").write_bytes(b"science cube")
            before = tree_sha256(base)

            clone = root / "runs" / "science_H04Inject_inj0001"
            created = create_run_clone(base, clone)
            after = tree_sha256(base)

            self.assertEqual(created, clone)
            self.assertEqual(before, after)
            self.assertTrue((clone / "config" / "config.json").exists())
            self.assertTrue((clone / "stages" / "stage02.fits").exists())

            (clone / "stages" / "stage02.fits").write_bytes(b"mutated clone")
            self.assertEqual(before, tree_sha256(base))


if __name__ == "__main__":
    unittest.main()
