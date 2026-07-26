"""El constructor de notebooks de análisis copia el código de la cadena.

Copiar el fuente en vez de importarlo es una decisión deliberada — el notebook
tiene que ser editable — pero solo se sostiene si la copia (a) es de verdad la
de `musepipe`, (b) llega completa (imports y constantes incluidos) y (c) el
notebook avisa cuando la cadena cambia. Eso es lo que se fija aquí.

La prueba de que la copia REPRODUCE la cadena la hace el propio notebook, en su
celda de comparación; el test de datos reales de abajo la ejecuta y exige que
diga «IDÉNTICO».
"""
import ast
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def _load_builder():
    spec = importlib.util.spec_from_file_location(
        "build_debug_notebooks", ROOT / "scripts" / "build_debug_notebooks.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class InlinedSourceTests(unittest.TestCase):
    """Se comprueban TODAS las etapas cubiertas, no solo la primera."""

    @classmethod
    def setUpClass(cls):
        cls.bdn = _load_builder()

    def _sources(self, stage_id):
        return self.bdn.extract_sources(stage_id)

    def test_the_copy_is_literally_the_source_in_musepipe(self):
        for stage_id in self.bdn.INLINE_SOURCES:
            for rel, name, src, _sha in self._sources(stage_id):
                with self.subTest(etapa=stage_id, funcion=name):
                    text = (ROOT / rel).read_text(encoding="utf-8")
                    self.assertIn(src, text, f"{rel}:{name} no coincide con el fuente")

    def test_the_recorded_sha_matches_the_copied_text(self):
        # Si el sha se calculara sobre otro texto (p.ej. sin recortar el salto
        # final), el chequeo de deriva daria falsa alarma nada mas generar.
        import hashlib
        for stage_id in self.bdn.INLINE_SOURCES:
            for rel, name, src, sha in self._sources(stage_id):
                with self.subTest(etapa=stage_id, funcion=name):
                    actual = hashlib.sha256(src.encode("utf-8")).hexdigest()[:12]
                    self.assertEqual(actual, sha, f"{rel}:{name}")

    def test_the_copy_carries_the_imports_and_constants_it_uses(self):
        for stage_id in self.bdn.INLINE_SOURCES:
            sources = self._sources(stage_id)
            imports = self.bdn.needed_imports(sources)
            constants = "\n".join(self.bdn.needed_constants(sources))
            with self.subTest(etapa=stage_id):
                # `warnings` lo usa robust_sigma_axis0 y `FLAG_BAD_WINDOW`
                # channel_flags: sin detectarlos, el notebook fallaba al ejecutarse.
                self.assertIn("import warnings", imports)
                self.assertIn("import numpy as np", imports)
                self.assertIn("FLAG_BAD_WINDOW", constants)

    def test_nothing_the_copy_calls_is_left_undefined(self):
        """Ninguna función copiada puede llamar a otra que no viaje con ella."""
        for stage_id, modules in self.bdn.INLINE_SOURCES.items():
            sources = self._sources(stage_id)
            inlined = {name for _r, name, _s, _h in sources}
            blob = "\n\n".join(src for _r, _n, src, _h in sources)
            called = {n.func.id for n in ast.walk(ast.parse(blob))
                      if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
            defined_upstream = set()
            for rel, _names in modules:
                tree = ast.parse((ROOT / rel).read_text(encoding="utf-8"))
                defined_upstream |= {n.name for n in tree.body if isinstance(n, ast.FunctionDef)}
            with self.subTest(etapa=stage_id):
                self.assertEqual(sorted((called & defined_upstream) - inlined), [])


class GeneratedNotebookTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bdn = _load_builder()
        cls.mb = cls.bdn._load_main_builder()
        cls.cells = {
            stage_id: builder(cls.mb, "ROXs12b", "ROXs12b_realigned")
            for stage_id, (_slug, builder) in cls.bdn.BUILDERS.items()
        }

    def test_every_code_cell_compiles(self):
        for stage_id, cells in self.cells.items():
            for i, cell in enumerate(cells):
                if cell["cell_type"] != "code":
                    continue
                with self.subTest(etapa=stage_id, celda=i):
                    compile("".join(cell["source"]), f"<{stage_id} celda {i}>", "exec")

    def test_it_carries_the_drift_check_and_the_comparison(self):
        productos = {"C2": ["spec_aperture_object.fits"],
                     "C3": ["spec_optimal_object.fits", "spec_optimal_psfsub_object.fits"]}
        for stage_id, cells in self.cells.items():
            text = "\n".join("".join(c["source"]) for c in cells)
            with self.subTest(etapa=stage_id):
                self.assertIn("chequeo_de_deriva", text)
                self.assertIn("IDÉNTICO", text)
                for producto in productos[stage_id]:
                    self.assertIn(producto, text)

    def test_the_knobs_come_from_the_resolved_stage_config(self):
        """No copiados como literales: la etapa rellena defaults que no están
        escritos en el run (C3 hereda el anillo de fondo de C2), y copiarlos a
        mano fue justo lo que hizo que C3 no reprodujera la cadena."""
        for stage_id, cells in self.cells.items():
            text = "\n".join("".join(c["source"]) for c in cells)
            with self.subTest(etapa=stage_id):
                self.assertIn("_config_from_run(RUN_ID)", text)

    def test_it_writes_a_valid_notebook_under_debug(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "C2_aperture_debug.ipynb"
            path.write_text(
                json.dumps(self.mb.notebook(self.cells["C2"]), indent=1, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            payload = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(payload["nbformat"], 4)
        self.assertTrue(payload["cells"])
        self.assertEqual(payload["metadata"]["kernelspec"]["name"], "python3")


@pytest.mark.slow
class ReproducesTheChainTests(unittest.TestCase):
    """Ejecuta cada notebook de análisis y exige que reproduzca su etapa.

    Es el único test que prueba de verdad que la copia sigue siendo la cadena:
    lo demás verifica que el texto coincide, esto verifica que los NÚMEROS
    coinciden. Cuesta ~1.5 min por notebook, de ahí el marcador `slow`.

    Se salta si el run no está en disco (`runs/` no se versiona), que es el caso
    de un clon limpio.
    """

    RUN = "ROXs12b_realigned"
    OBJETO = "ROXs12b"
    #: notebook -> productos de la etapa que tienen que existir para compararlo
    CASOS = {
        "C2_aperture_debug": ["spec_aperture_object.fits"],
        "C3_optimal_debug": ["spec_optimal_object.fits", "spec_optimal_psfsub_object.fits"],
    }

    def test_every_debug_notebook_reports_identical(self):
        import contextlib
        import io
        import matplotlib
        matplotlib.use("Agg")

        stage_dir = ROOT / "runs" / self.RUN / "stages"
        for slug, productos in self.CASOS.items():
            path = ROOT / "notebooks" / self.OBJETO / "debug" / f"{slug}.ipynb"
            faltan = [n for n in productos if not (stage_dir / n).exists()]
            with self.subTest(notebook=slug):
                if faltan:
                    self.skipTest(f"{self.RUN} sin productos: falta {faltan[0]}")
                if not path.exists():
                    self.skipTest(f"{slug} no generado")
                payload = json.loads(path.read_text(encoding="utf-8"))
                namespace = {"__name__": "__main__"}
                out = io.StringIO()
                with contextlib.redirect_stdout(out):
                    for i, cell in enumerate(payload["cells"]):
                        if cell["cell_type"] != "code":
                            continue
                        exec(compile("".join(cell["source"]), f"<{slug} celda {i}>", "exec"),
                             namespace)
                text = out.getvalue()
                self.assertIn("sin deriva", text, "la copia no coincide con musepipe")
                self.assertIn("IDÉNTICO: la copia reproduce la cadena.", text,
                              f"{slug} no reprodujo la cadena:\n{text[-1200:]}")


if __name__ == "__main__":
    unittest.main()
