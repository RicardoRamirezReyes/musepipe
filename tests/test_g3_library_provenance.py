"""La biblioteca de atmósferas se declara desde el disco, no desde el config.

`g3_atmosphere_family` era una cadena del config y se quedó vieja: los dos runs
llevaron "BT-Settl (CIFIST) [deferred, pending data]" siete semanas después de
que la descarga terminara completa (146 nodos, partial=false, 2026-07-17), así
que todos los QC de G3 declararon provisional una biblioteca que no lo era.
"""
import json
import tempfile
import unittest
from pathlib import Path

from musepipe.models.manifest import (LIBRARY_SUBDIRS, PROVENANCE_NAME,
                                      library_is_complete, library_provenance)
from musepipe.stages.stage_g3_atmo_fit import _library_block

ROOT = Path(__file__).resolve().parents[1]


class LibraryProvenanceTests(unittest.TestCase):
    def _biblioteca(self, tmp, prov):
        fam = Path(tmp) / "libs" / LIBRARY_SUBDIRS["bt-settl"]
        fam.mkdir(parents=True)
        (fam / PROVENANCE_NAME).write_text(json.dumps(prov), encoding="utf-8")
        return {"g3_libraries_root": "libs"}, tmp

    def test_reads_the_grid_from_disk(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg, root = self._biblioteca(tmp, {"grid": "BT-Settl CIFIST2011 (SVO)",
                                               "partial": False, "n_failed": 0})
            prov = library_provenance(cfg, project_root=root)
            self.assertEqual(prov["grid"], "BT-Settl CIFIST2011 (SVO)")
            self.assertTrue(library_is_complete(prov))

    def test_absent_library_is_none_not_a_crash(self):
        self.assertIsNone(library_provenance({}))
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "libs").mkdir()
            self.assertIsNone(library_provenance({"g3_libraries_root": "libs"},
                                                 project_root=tmp))

    def test_a_partial_download_is_not_complete(self):
        for prov in ({"partial": True, "n_failed": 0},
                     {"partial": False, "n_failed": 3},
                     None, {}):
            self.assertFalse(library_is_complete(prov), prov)

    def test_the_disk_wins_over_a_stale_declared_label(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg, root = self._biblioteca(tmp, {"grid": "BT-Settl CIFIST2011 (SVO)",
                                               "partial": False, "n_failed": 0,
                                               "n_nodes": 146})
            cfg["g3_atmosphere_family"] = "BT-Settl (CIFIST) [deferred, pending data]"
            import musepipe.stages.stage_g3_atmo_fit as mod
            real = mod.library_provenance
            mod.library_provenance = lambda c, *a, **k: library_provenance(
                c, *a, project_root=root, **{k_: v for k_, v in k.items()
                                             if k_ != "project_root"})
            try:
                block = _library_block(cfg, "Allard et al. 2012")
            finally:
                mod.library_provenance = real
            self.assertEqual(block["family"], "BT-Settl CIFIST2011 (SVO)")
            self.assertEqual(block["family_declared"],
                             "BT-Settl (CIFIST) [deferred, pending data]")
            self.assertTrue(block["complete"])
            self.assertEqual(block["provenance"]["n_nodes"], 146)

    def test_no_run_config_still_calls_the_library_deferred(self):
        """El bug concreto: la cadena que decía que estaba pendiente."""
        for cfg_path in sorted(ROOT.glob("runs/*/config/config.json")):
            d = json.loads(cfg_path.read_text(encoding="utf-8"))
            if (d.get("meta") or {}).get("legacy"):
                continue
            fam = (d.get("config") or {}).get("g3_atmosphere_family")
            if fam is None:
                continue
            with self.subTest(run=cfg_path.parts[-3]):
                self.assertNotIn("deferred", fam.lower(), cfg_path)
                self.assertNotIn("pending", fam.lower(), cfg_path)


if __name__ == "__main__":
    unittest.main()
