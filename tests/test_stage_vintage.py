"""Una etapa más vieja que el cubo que carga la cadena tiene que decirlo.

El 2026-07-28 el cubo de entrada de `ROXs12b_realigned` pasó de una reducción
de UN OB (7 exposiciones) a la combinación de 29 de dos OB, y D2, E, F1, G y S
nunca se re-ejecutaron. Nada lo dijo: se descubrió midiendo el Hα de la
primaria y viendo que el producto calibrado en disco daba −12 Å donde el de C4
daba −2.4 Å.

`nb.stage_vintage` fecha cada etapa contra ese cubo. Lo que se fija aquí es que
detecta el desfase, que NO inventa desfases donde no los hay, y que no marca a
las etapas del bloque A —que son las que *producen* el cubo—.

Se construyen runs sintéticos en `tmp_path`: el test no depende de `runs/`, que
no se versiona.
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NB_DIR = ROOT / "notebooks"


def _fresh_nbcommon(project_root: Path):
    """Una copia limpia de `_nbcommon` anclada a otra raíz de proyecto.

    El módulo cachea la raíz y el run activo en globals, así que se recarga por
    test en vez de compartir estado entre ellos.
    """
    spec = importlib.util.spec_from_file_location(
        f"_nbcommon_vintage_{id(project_root)}", NB_DIR / "_nbcommon.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.project_root = lambda: project_root
    return module


class _Escenario:
    """Un run sintético: cubo de entrada + QC de las etapas que se pidan."""

    def __init__(self, tmp: Path, run_id: str = "OBJ_realigned"):
        self.root = tmp
        self.run_id = run_id
        self.run = tmp / "runs" / run_id
        (self.run / "stages").mkdir(parents=True)
        (self.run / "config").mkdir(parents=True)
        (self.run / "config" / "config.json").write_text(json.dumps(
            {"meta": {}, "config": {"run_id": run_id},
             "chain": {"target": "OBJ", "default_run": run_id}}), encoding="utf-8")
        self.cube = tmp / "cube.fits"
        self.cube.write_bytes(b"")

    def cube_at(self, when: float):
        os.utime(self.cube, (when, when))

    def qc(self, relpath: str, when: float, payload: dict | None = None):
        path = self.run / relpath
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload or {}), encoding="utf-8")
        os.utime(path, (when, when))

    def b1(self, when: float):
        """B1 es quien declara el cubo: su QC es el ancla de la procedencia."""
        self.qc("stages/stage01_qc.json", when,
                {"inputs": [{"file": str(self.cube), "data_ext": 1}]})


T0 = 1_800_000_000.0          # instante arbitrario, estable
DIA = 86_400.0


class StageVintageTests(unittest.TestCase):
    def setUp(self):
        import tempfile

        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.esc = _Escenario(self.tmp)
        self.nb = _fresh_nbcommon(self.tmp)
        self.nb.resolve_run_id(self.esc.run_id)

    def tearDown(self):
        self._tmp.cleanup()

    def _vintage(self):
        return {r["id"]: r for r in self.nb.stage_vintage(self.esc.run_id)}

    def test_it_finds_the_cube_declared_by_b1(self):
        self.esc.cube_at(T0)
        self.esc.b1(T0 + DIA)
        cubo, mtime = self.nb.entry_cube(self.esc.run_id)
        self.assertEqual(cubo, self.esc.cube)
        self.assertAlmostEqual(mtime, T0, places=3)

    def test_a_stage_older_than_the_cube_is_flagged(self):
        self.esc.cube_at(T0)
        self.esc.b1(T0 + DIA)
        self.esc.qc("stages/stage_x11_qc.json", T0 - DIA)       # D2, del día ANTES
        rows = self._vintage()
        self.assertIn("D2", rows)
        self.assertIsNotNone(rows["D2"]["desfasada_por"])
        self.assertIn("cubo de entrada", rows["D2"]["desfasada_por"])

    def test_a_stage_newer_than_the_cube_is_not_flagged(self):
        self.esc.cube_at(T0)
        self.esc.b1(T0 + DIA)
        self.esc.qc("stages/stage_x11_qc.json", T0 + 2 * DIA)
        self.assertIsNone(self._vintage()["D2"]["desfasada_por"])

    def test_the_A_block_is_never_flagged_because_it_produces_the_cube(self):
        """A1-A4 escriben y califican el cubo: compararlas con él no tiene sentido."""
        self.esc.cube_at(T0)
        self.esc.b1(T0 + DIA)
        self.esc.qc("stages/stage00r_qc.json", T0 - 5 * DIA)    # A1, muy anterior
        self.esc.qc("stages/stage00q_qc.json", T0 - 5 * DIA)    # A4
        rows = self._vintage()
        self.assertIsNone(rows["A1"]["desfasada_por"])
        self.assertIsNone(rows["A4"]["desfasada_por"])

    def test_a_stage_that_never_ran_is_ignored_not_flagged(self):
        """Sin QC no hay fecha: no está desfasada, es que no existe."""
        self.esc.cube_at(T0)
        self.esc.b1(T0 + DIA)
        rows = self._vintage()
        self.assertNotIn("D2", rows)
        self.assertNotIn("G3", rows)

    def test_without_b1_there_is_no_anchor_and_nothing_is_flagged(self):
        """Un run sin B1 no declara cubo: se informa, no se inventa un desfase."""
        self.esc.qc("stages/stage_x11_qc.json", T0 - 5 * DIA)
        cubo, mtime = self.nb.entry_cube(self.esc.run_id)
        self.assertIsNone(cubo)
        self.assertIsNone(mtime)
        self.assertTrue(all(r["desfasada_por"] is None for r in self._vintage().values()))

    def test_a_declared_cube_that_is_gone_does_not_crash(self):
        """El cubo puede haberse borrado del disco: se dice, no se revienta."""
        self.esc.b1(T0)
        self.esc.cube.unlink()
        cubo, mtime = self.nb.entry_cube(self.esc.run_id)
        self.assertEqual(cubo, self.esc.cube)
        self.assertIsNone(mtime)
        self.assertTrue(all(r["desfasada_por"] is None for r in self._vintage().values()))

    def test_show_vintage_names_the_stale_stages(self):
        import contextlib
        import io

        self.esc.cube_at(T0)
        self.esc.b1(T0 + DIA)
        self.esc.qc("stages/stage_x11_qc.json", T0 - DIA)
        self.esc.qc("stages/stage_h03_qc.json", T0 + 2 * DIA)
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            self.nb.show_vintage(self.esc.run_id)
        texto = out.getvalue()
        self.assertIn("D2", texto)
        self.assertIn("MÁS VIEJAS", texto)
        self.assertIn("E3", texto.split("al día:")[-1])      # E3 está al día


class SiblingSettingTests(unittest.TestCase):
    """El valor prestado de otro run tiene que decir de qué run viene."""

    def setUp(self):
        import tempfile

        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.esc = _Escenario(self.tmp)
        # Un run hermano del MISMO objeto, anterior por orden alfabético.
        hermano = self.tmp / "runs" / "OBJ_adp" / "config"
        hermano.mkdir(parents=True)
        (hermano / "config.json").write_text(json.dumps(
            {"meta": {}, "config": {"run_id": "OBJ_adp", "cube_files": ["/data/ADP.fits"]}}),
            encoding="utf-8")
        self.nb = _fresh_nbcommon(self.tmp)
        self.nb.resolve_run_id(self.esc.run_id)

    def tearDown(self):
        self._tmp.cleanup()

    def test_it_reports_which_run_the_value_came_from(self):
        valor, origen = self.nb.sibling_setting(self.esc.run_id, "cube_files")
        self.assertEqual(valor, ["/data/ADP.fits"])
        self.assertEqual(origen, "OBJ_adp")

    def test_a_key_nobody_declares_returns_no_origin(self):
        self.assertEqual(self.nb.sibling_setting(self.esc.run_id, "no_existe"), (None, None))

    def test_the_legacy_wrapper_prints_the_origin(self):
        import contextlib
        import io

        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            valor = self.nb._sibling_setting(self.esc.run_id, "cube_files")
        self.assertEqual(valor, ["/data/ADP.fits"])
        self.assertIn("OBJ_adp", out.getvalue())


if __name__ == "__main__":
    unittest.main()
