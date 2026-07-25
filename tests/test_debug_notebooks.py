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

ROOT = Path(__file__).resolve().parent.parent


def _load_builder():
    spec = importlib.util.spec_from_file_location(
        "build_debug_notebooks", ROOT / "scripts" / "build_debug_notebooks.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class InlinedSourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bdn = _load_builder()
        cls.sources = cls.bdn.extract_sources("C2")

    def test_the_copy_is_literally_the_source_in_musepipe(self):
        for rel, name, src, _sha in self.sources:
            text = (ROOT / rel).read_text(encoding="utf-8")
            self.assertIn(src, text, f"{rel}:{name} no coincide con el fuente")

    def test_the_recorded_sha_matches_the_copied_text(self):
        # Si el sha se calculara sobre otro texto (p.ej. sin recortar el salto
        # final), el chequeo de deriva daria falsa alarma nada mas generar.
        import hashlib
        for rel, name, src, sha in self.sources:
            actual = hashlib.sha256(src.encode("utf-8")).hexdigest()[:12]
            self.assertEqual(actual, sha, f"{rel}:{name}")

    def test_the_copy_carries_the_imports_and_constants_it_uses(self):
        imports = self.bdn.needed_imports(self.sources)
        constants = "\n".join(self.bdn.needed_constants(self.sources))
        # `warnings` lo usa robust_sigma_axis0 y `FLAG_BAD_WINDOW` channel_flags:
        # sin detectarlos, el notebook fallaba al ejecutarse.
        self.assertIn("import warnings", imports)
        self.assertIn("import numpy as np", imports)
        self.assertIn("FLAG_BAD_WINDOW", constants)

    def test_nothing_the_copy_calls_is_left_undefined(self):
        """Ninguna función copiada puede llamar a otra que no viaje con ella."""
        inlined = {name for _r, name, _s, _h in self.sources}
        blob = "\n\n".join(src for _r, _n, src, _h in self.sources)
        called = {n.func.id for n in ast.walk(ast.parse(blob))
                  if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
        defined_upstream = set()
        for rel, _names in self.bdn.INLINE_SOURCES["C2"]:
            tree = ast.parse((ROOT / rel).read_text(encoding="utf-8"))
            defined_upstream |= {n.name for n in tree.body if isinstance(n, ast.FunctionDef)}
        self.assertEqual(sorted((called & defined_upstream) - inlined), [])


class GeneratedNotebookTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bdn = _load_builder()
        cls.mb = cls.bdn._load_main_builder()
        cls.cells = cls.bdn.build_c2_cells(cls.mb, "ROXs12b", "ROXs12b_realigned")

    def test_every_code_cell_compiles(self):
        for i, cell in enumerate(self.cells):
            if cell["cell_type"] != "code":
                continue
            src = "".join(cell["source"])
            with self.subTest(celda=i):
                compile(src, f"<celda {i}>", "exec")

    def test_it_carries_the_drift_check_and_the_comparison(self):
        text = "\n".join("".join(c["source"]) for c in self.cells)
        self.assertIn("chequeo_de_deriva", text)
        self.assertIn("spec_aperture_object.fits", text)
        self.assertIn("IDÉNTICO", text)

    def test_it_writes_a_valid_notebook_under_debug(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "C2_aperture_debug.ipynb"
            path.write_text(
                json.dumps(self.mb.notebook(self.cells), indent=1, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            payload = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(payload["nbformat"], 4)
        self.assertTrue(payload["cells"])
        self.assertEqual(payload["metadata"]["kernelspec"]["name"], "python3")


class ReproducesTheChainTests(unittest.TestCase):
    """Ejecuta el notebook contra un run real y exige que reproduzca la etapa.

    Se salta si el run no esta en disco (`runs/` no se versiona), que es el caso
    de un clon limpio.
    """

    RUN = "ROXs12b_realigned"

    def setUp(self):
        stage_dir = ROOT / "runs" / self.RUN / "stages"
        needed = ["spec_aperture_object.fits", "stage01c_qc.json", "stage02_xcorr_cube_stack.fits"]
        missing = [n for n in needed if not (stage_dir / n).exists()]
        if missing:
            self.skipTest(f"run {self.RUN} sin productos de C2: falta {missing[0]}")
        self.notebook_path = ROOT / "notebooks" / "ROXs12b" / "debug" / "C2_aperture_debug.ipynb"
        if not self.notebook_path.exists():
            self.skipTest("notebook de análisis no generado")

    def test_the_comparison_cell_reports_identical(self):
        import contextlib
        import io
        import matplotlib
        matplotlib.use("Agg")

        payload = json.loads(self.notebook_path.read_text(encoding="utf-8"))
        namespace = {"__name__": "__main__"}
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            for i, cell in enumerate(payload["cells"]):
                if cell["cell_type"] != "code":
                    continue
                exec(compile("".join(cell["source"]), f"<celda {i}>", "exec"), namespace)
        text = out.getvalue()
        self.assertIn("sin deriva", text, "la copia no coincide con musepipe")
        self.assertIn("IDÉNTICO: la copia reproduce la cadena.", text,
                      f"la comparación no salió idéntica:\n{text[-1200:]}")


if __name__ == "__main__":
    unittest.main()
