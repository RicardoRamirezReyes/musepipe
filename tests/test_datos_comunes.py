"""`tests/_datos.py`: una sola precedencia para encontrar los datos y un solo cargador de scripts (2026-10-29).

Todo con raíces sintéticas en un directorio temporal: no lee runs ni cadenas reales.
"""
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tests import _datos


class RaizTests(unittest.TestCase):
    def test_sin_variable_es_este_clon(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("MUSE_PROJECT_ROOT", None)
            self.assertEqual(_datos.raiz_proyecto(), _datos.REPO)
            self.assertEqual(_datos.runs_dir(), _datos.REPO / "runs")

    def test_la_variable_gana(self):
        with tempfile.TemporaryDirectory() as t, mock.patch.dict(os.environ, {"MUSE_PROJECT_ROOT": t}):
            self.assertEqual(_datos.raiz_proyecto(), Path(t).resolve())
            self.assertEqual(_datos.runs_dir(), Path(t).resolve() / "runs")

    def test_los_runs_del_paper_salen_de_targets(self):
        runs = _datos.runs_del_paper()
        for f in (_datos.REPO / "targets").glob("*.json"):
            d = json.loads(f.read_text())
            if "paper" in d:
                self.assertEqual(runs[d["slug"]], d["paper"]["run"])
                self.assertEqual(_datos.run_del_paper(d["slug"]), d["paper"]["run"])


class CadenasTests(unittest.TestCase):
    def test_por_defecto_las_del_paper(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("MUSE_TEFF_CADENAS", None)
            self.assertEqual(_datos.cadenas_teff(), [_datos.CADENAS_DEL_PAPER.expanduser()])

    def test_una_carpeta_y_una_lista(self):
        with mock.patch.dict(os.environ, {"MUSE_TEFF_CADENAS": "/a/b"}):
            self.assertEqual(_datos.cadenas_teff(), [Path("/a/b")])
        with mock.patch.dict(os.environ, {"MUSE_TEFF_CADENAS": os.pathsep.join(("/a", "~/c", ""))}):
            self.assertEqual(_datos.cadenas_teff(), [Path("/a"), Path("~/c").expanduser()])


class BibliotecasTests(unittest.TestCase):
    def test_relativas_a_la_raiz_del_proyecto_no_al_clon(self):
        with tempfile.TemporaryDirectory() as t:
            raiz = Path(t) / "clon"
            (raiz / "runs" / "R1" / "config").mkdir(parents=True)
            (raiz / "runs" / "R1" / "config" / "config.json").write_text(
                json.dumps({"config": {"g3_libraries_root": "../Data/external_libraries"}}))
            with mock.patch.dict(os.environ, {"MUSE_PROJECT_ROOT": str(raiz)}):
                os.environ.pop("MUSE_G3_LIBRARIES", None)
                self.assertEqual(_datos.bibliotecas_g3("R1"), (Path(t) / "Data" / "external_libraries").resolve())
                with mock.patch.dict(os.environ, {"MUSE_G3_LIBRARIES": "/x/libs"}):
                    self.assertEqual(_datos.bibliotecas_g3("R1"), Path("/x/libs"))


class RequiereTests(unittest.TestCase):
    def test_salta_y_nombra_la_variable(self):
        with tempfile.TemporaryDirectory() as t, mock.patch.dict(
                os.environ, {"MUSE_PROJECT_ROOT": t, "MUSE_TEFF_CADENAS": str(Path(t) / "no"),
                             "MUSE_IR_DATA": str(Path(t) / "no")}):
            self.assertIn("MUSE_PROJECT_ROOT", _datos.falta("runs"))
            self.assertIn("MUSE_TEFF_CADENAS", _datos.falta("cadenas"))
            self.assertIn("MUSE_IR_DATA", _datos.falta("ir"))
            self.assertIn("MUSE_PROJECT_ROOT", _datos.falta("bibliotecas"))

            @_datos.requiere("runs")
            class Salta(unittest.TestCase):
                def test_x(self):
                    raise AssertionError("no debía correr")
            res = unittest.TestResult()
            unittest.defaultTestLoader.loadTestsFromTestCase(Salta).run(res)
            self.assertEqual((res.testsRun, len(res.skipped), len(res.failures)), (1, 1, 0))
            self.assertIn("MUSE_PROJECT_ROOT", res.skipped[0][1])

    def test_no_salta_si_esta(self):
        with tempfile.TemporaryDirectory() as t, mock.patch.dict(os.environ, {"MUSE_TEFF_CADENAS": t,
                                                                             "MUSE_IR_DATA": t}):
            self.assertIsNone(_datos.falta("cadenas", "ir"))

    def test_recurso_desconocido(self):
        with self.assertRaises(KeyError):
            _datos.falta("nada")


class CargaScriptTests(unittest.TestCase):
    def test_un_solo_modulo(self):
        tb = _datos.carga_script("teff_bayes")
        self.assertIs(tb, sys.modules["scripts.teff_bayes"])
        self.assertIs(_datos.carga_script("teff_bayes"), tb)
        from scripts import teff_bayes
        self.assertIs(teff_bayes, tb)
        # y sus dependencias son las mismas que importa el script
        from scripts import tipo_ir_plantillas
        self.assertIs(sys.modules["scripts.tipo_ir_plantillas"], tipo_ir_plantillas)


class RaizTemporalTests(unittest.TestCase):
    def test_restaura_tambien_con_una_excepcion(self):
        tb = _datos.carga_script("teff_bayes")
        from scripts import tipo_ir_plantillas as tp
        antes = (tb.ROOT, tp.ROOT)
        with tempfile.TemporaryDirectory() as t:
            with _datos.raiz_temporal(t) as r:
                self.assertEqual(r, Path(t).resolve())
                self.assertEqual((tb.ROOT, tp.ROOT), (r, r))
            self.assertEqual((tb.ROOT, tp.ROOT), antes)
            with self.assertRaises(RuntimeError):
                with _datos.raiz_temporal(t):
                    raise RuntimeError("x")
            self.assertEqual((tb.ROOT, tp.ROOT), antes)


if __name__ == "__main__":
    unittest.main()
