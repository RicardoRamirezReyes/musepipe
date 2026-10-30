"""La raíz de las bibliotecas de `scripts/tipo_ir_plantillas.plantillas` (2026-10-30).

`plantillas(cfg, ..., root=ROOT)` fijaba la raíz al importarse, así que `teff_bayes.fija_raiz` (`--project-root`) no
le llegaba: desde un worktree, `g3_libraries_root` (relativo, `../Data/external_libraries`) se resolvía contra el
worktree y las cadenas con `--plantilla` fallaban. Sintético: no lee disco (`library_root` sustituido).
"""
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tests._datos import carga_script, raiz_temporal


class RaizTests(unittest.TestCase):
    def setUp(self):
        self.tp = carga_script("tipo_ir_plantillas")
        self.vistas = []

        def library_root(cfg, project_root=None):
            self.vistas.append(Path(project_root))
            return Path(project_root) / "bibliotecas"
        p = mock.patch.object(self.tp, "library_root", library_root)
        p.start()
        self.addCleanup(p.stop)

    def test_sin_root_sigue_a_fija_raiz(self):
        with tempfile.TemporaryDirectory() as d, raiz_temporal(d) as raiz:
            self.assertEqual(self.tp.plantillas({}, bibliotecas=()), [])
        self.assertEqual(self.vistas, [raiz])
        self.assertNotEqual(raiz, self.tp.ROOT)          # raiz_temporal la restaura al salir

    def test_root_explicito_gana(self):
        with tempfile.TemporaryDirectory() as d, tempfile.TemporaryDirectory() as otra, raiz_temporal(d):
            self.tp.plantillas({}, bibliotecas=(), root=Path(otra))
        self.assertEqual(self.vistas, [Path(otra)])


if __name__ == "__main__":
    unittest.main()
