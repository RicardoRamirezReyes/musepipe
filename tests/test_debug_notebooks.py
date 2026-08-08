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

    #: Lo que cada etapa TIENE que arrastrar. Por etapa y no en común: lo que
    #: se fija aquí son regresiones concretas, y exigirle a A3 el
    #: `FLAG_BAD_WINDOW` del bloque C solo mediría que la lista está mal.
    ARRASTRA = {
        # `warnings` lo usa robust_sigma_axis0 y `FLAG_BAD_WINDOW` channel_flags:
        # sin detectarlos, el notebook fallaba al ejecutarse.
        "C2": (["import warnings", "import numpy as np"], ["FLAG_BAD_WINDOW"]),
        "C3": (["import warnings", "import numpy as np"], ["FLAG_BAD_WINDOW"]),
        "C4": (["import warnings", "import numpy as np"], ["FLAG_BAD_WINDOW"]),
        "C5": (["import warnings", "import numpy as np"], ["FLAG_BAD_WINDOW"]),
        "C6": (["import warnings", "import numpy as np"], ["FLAG_BAD_WINDOW"]),
        # A3: las dos ventanas protegidas solo aparecen DENTRO del valor de
        # `PROTECTED_WINDOWS`, así que fijan el cierre transitivo de
        # `needed_constants` — sin él la copia peta con NameError.
        "A3": (["import numpy as np", "from dataclasses import dataclass"],
               ["TELLURIC_BANDS", "PROTECTED_WINDOWS",
                "HALPHA_PROTECTED", "NALGS_PROTECTED"]),
        # D2: `BAD_CONTINUUM_FLAGS` está escrita en términos de dos nombres
        # IMPORTADOS, no de otras constantes. Fija que `needed_imports` mire
        # también las constantes copiadas: mirando solo los `def`, ese bloque
        # salía sin sus imports y la primera celda de código petaba.
        "D2": (["import numpy as np", "from dataclasses import dataclass",
                "from musepipe.extraction.aperture import FLAG_BAD_WINDOW",
                "from musepipe.extraction.aperture import FLAG_SKYLINE"],
               ["BAD_CONTINUUM_FLAGS"]),
    }

    def test_the_copy_carries_the_imports_and_constants_it_uses(self):
        for stage_id in self.bdn.INLINE_SOURCES:
            sources = self._sources(stage_id)
            imports = self.bdn.needed_imports(sources)
            constants = "\n".join(self.bdn.needed_constants(sources))
            with self.subTest(etapa=stage_id):
                self.assertIn(stage_id, self.ARRASTRA,
                              "etapa nueva sin expectativa declarada en ARRASTRA")
                esperados_imports, esperadas_constantes = self.ARRASTRA[stage_id]
                for imp in esperados_imports:
                    self.assertIn(imp, imports)
                for cte in esperadas_constantes:
                    self.assertIn(cte, constants)

    def test_the_constants_are_watched_by_the_drift_check_too(self):
        """Cada constante copiada tiene su sha, y el sha es el de `musepipe`.

        El guardia solo hasheaba `FunctionDef`/`ClassDef`: las constantes
        viajaban sin vigilar, así que tocar `TELLURIC_BANDS` dejaba la copia
        atrás mientras el notebook imprimía «sin deriva».
        """
        import hashlib
        for stage_id in self.bdn.INLINE_SOURCES:
            sources = self._sources(stage_id)
            shas = self.bdn.constant_shas(sources)
            seleccion = self.bdn.selected_constants(sources)
            with self.subTest(etapa=stage_id):
                self.assertTrue(seleccion, "ninguna constante copiada: ¿se rompió la detección?")
                _imports, esperadas = self.ARRASTRA[stage_id]
                for cte in esperadas:
                    self.assertTrue(any(k.endswith(f":{cte}") for k in shas),
                                    f"{stage_id}: {cte} viaja copiada pero sin sha")
            for rel, targets, src in seleccion:
                with self.subTest(etapa=stage_id, constante=targets):
                    esperado = hashlib.sha256(src.encode("utf-8")).hexdigest()[:12]
                    for name in targets:
                        self.assertEqual(shas[f"{rel}:{name}"], esperado)
                    self.assertIn(src, (ROOT / rel).read_text(encoding="utf-8"))

    def test_a_constant_never_precedes_what_it_is_written_in_terms_of(self):
        """Las constantes salen en orden de módulo, no de descubrimiento.

        `PROTECTED_WINDOWS = (HALPHA_PROTECTED, NALGS_PROTECTED)` se descubre
        antes que sus dos operandos; emitirla antes que ellos daría un notebook
        que compila y revienta al ejecutarse.

        Se ejecuta con los imports delante, que es como los emite el notebook:
        una constante puede estar escrita en términos de un nombre IMPORTADO
        (`BAD_CONTINUUM_FLAGS = FLAG_BAD_WINDOW | FLAG_SKYLINE` en D2) y sin
        ellos este test mediría otra cosa.
        """
        for stage_id in self.bdn.INLINE_SOURCES:
            sources = self._sources(stage_id)
            bloque = "\n".join(self.bdn.needed_imports(sources) + [""]
                               + self.bdn.needed_constants(sources))
            with self.subTest(etapa=stage_id):
                try:
                    exec(compile(bloque, f"<constantes {stage_id}>", "exec"), {})
                except NameError as exc:
                    self.fail(f"{stage_id}: la copia no define lo que usa ({exc})")

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
        productos = {"A3": ["stage00t_qc.json"],
                     "C2": ["spec_aperture_object.fits"],
                     "C3": ["spec_optimal_object.fits", "spec_optimal_psfsub_object.fits"],
                     "C4": ["spec_psffit_object.fits", "spec_psffit_star.fits"],
                     "C5": ["spec_sgf_object.fits"],
                     "C6": ["spec_lpm_object.fits"],
                     "D2": ["spec_calibrated_psffit_star.fits"]}
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
                # Con `project_root=ROOT`: musepipe resuelve rutas contra el cwd,
                # que en un notebook es su propia carpeta — sin eso, buscaba el
                # config bajo `notebooks/<obj>/debug/runs/...` y reventaba.
                # `_from_run(...)` y no el nombre completo: C5/C6 lo importan con
                # alias porque comparten constructor.
                self.assertIn("_from_run(RUN_ID, project_root=ROOT)", text)

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

    #: notebook -> productos de la etapa que tienen que existir para compararlo
    CASOS = {
        "A3_telluric_debug": ["stage00t_qc.json"],
        "C2_aperture_debug": ["spec_aperture_object.fits"],
        "C3_optimal_debug": ["spec_optimal_object.fits", "spec_optimal_psfsub_object.fits"],
        "C4_psffit_debug": ["spec_psffit_object.fits", "spec_psffit_star.fits"],
        "C5_sgf_debug": ["spec_sgf_object.fits"],
        "C6_lpm_debug": ["spec_lpm_object.fits"],
        "D2_primary_star_debug": ["spec_psffit_star.fits", "spec_calibrated_psffit_star.fits"],
    }
    #: (objeto, run) de cada cadena. Estaba fijado a ROXs 12 b, asi que los cinco
    #: notebooks de ROXs 42B b **no los ejecutaba nadie**: se generaban y nadie
    #: comprobaba que reprodujeran su etapa. Cada objeto se salta solo si le
    #: faltan los productos, asi que un clon limpio sigue pasando.
    OBJETOS = (("ROXs12b", "ROXs12b_realigned"), ("ROXs42Bb", "ROXs42Bb_realigned"))

    @staticmethod
    def _entrada_fuera_del_run(slug, stage_dir):
        """Ruta que falta, de las que NO viven bajo `runs/<RUN>/stages`.

        Los notebooks del bloque C se comparan contra un producto del propio
        run; A3 mide sobre los cubos de reducción, que están fuera del árbol
        del run (y en otro disco). Sin esta comprobación el notebook se
        ejecutaría y fallaría por falta de datos en vez de saltarse.
        """
        if slug != "A3_telluric_debug":
            return None
        qc = stage_dir / "stage00t_qc.json"
        if not qc.exists():
            return str(qc)
        declarado = (json.loads(qc.read_text(encoding="utf-8"))
                     .get("input", {}).get("cube", ""))
        cubo = Path(declarado) if declarado else None
        if cubo is not None and not cubo.is_absolute():
            cubo = ROOT / cubo
        return None if (cubo is not None and cubo.exists()) else str(declarado)

    #: notebook -> (producto de la etapa, entrada de la que se deriva). El
    #: notebook solo puede salir IDÉNTICO si el producto se escribió DESPUÉS de
    #: su entrada; si la etapa de arriba se re-ejecutó y la de abajo no, lo que
    #: hay en disco calibra un espectro que ya no existe y la diferencia no
    #: mide la copia, mide que el run está a medias.
    DERIVADOS = {
        "D2_primary_star_debug": ("spec_calibrated_psffit_star.fits", "spec_psffit_star.fits"),
    }

    @classmethod
    def _producto_desfasado(cls, slug, stage_dir):
        """`None`, o el motivo por el que el producto en disco no es comparable."""
        par = cls.DERIVADOS.get(slug)
        if par is None:
            return None
        salida, entrada = (stage_dir / n for n in par)
        if not (salida.exists() and entrada.exists()):
            return None
        if salida.stat().st_mtime >= entrada.stat().st_mtime:
            return None
        return (f"{salida.name} es más viejo que {entrada.name}: la etapa no se ha "
                "re-ejecutado desde su entrada")

    def test_every_debug_notebook_reports_identical(self):
        import matplotlib
        matplotlib.use("Agg")

        for objeto, run in self.OBJETOS:
            stage_dir = ROOT / "runs" / run / "stages"
            for slug, productos in self.CASOS.items():
                path = ROOT / "notebooks" / objeto / "debug" / f"{slug}.ipynb"
                faltan = [n for n in productos if not (stage_dir / n).exists()]
                with self.subTest(objeto=objeto, notebook=slug):
                    if faltan:
                        self.skipTest(f"{run} sin productos: falta {faltan[0]}")
                    fuera = self._entrada_fuera_del_run(slug, stage_dir)
                    if fuera:
                        self.skipTest(f"{slug}: entrada fuera del run sin disponer: {fuera}")
                    desfasado = self._producto_desfasado(slug, stage_dir)
                    if desfasado:
                        self.skipTest(f"{slug}: {desfasado}")
                    if not path.exists():
                        self.skipTest(f"{slug} no generado")
                    self._reproduce_la_cadena(path, slug)

    def _reproduce_la_cadena(self, path, slug):
        """Ejecuta el notebook y exige que diga «sin deriva» e «IDÉNTICO»."""
        import contextlib
        import io
        import os

        payload = json.loads(path.read_text(encoding="utf-8"))
        namespace = {"__name__": "__main__"}
        out = io.StringIO()
        # Se ejecuta DESDE LA CARPETA DEL NOTEBOOK, que es el cwd real en
        # Jupyter. Correrlo desde la raíz del repo escondía un fallo:
        # musepipe resuelve rutas contra el cwd y `stage_xNN_config_from_run`
        # buscaba el config bajo `notebooks/<obj>/debug/runs/...`.
        cwd = os.getcwd()
        try:
            os.chdir(path.parent)
            with contextlib.redirect_stdout(out):
                for i, cell in enumerate(payload["cells"]):
                    if cell["cell_type"] != "code":
                        continue
                    exec(compile("".join(cell["source"]), f"<{slug} celda {i}>", "exec"),
                         namespace)
        finally:
            os.chdir(cwd)
        text = out.getvalue()
        self.assertIn("sin deriva", text, "la copia no coincide con musepipe")
        self.assertIn("IDÉNTICO: la copia reproduce la cadena.", text,
                      f"{slug} no reprodujo la cadena:\n{text[-1200:]}")


if __name__ == "__main__":
    unittest.main()
