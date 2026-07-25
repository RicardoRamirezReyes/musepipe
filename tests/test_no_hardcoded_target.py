"""Guardián: ningún objeto observado hardcodeado en el código.

Todo lo específico de un objeto vive en `runs/<run>/config/config.json` y en
`targets/<slug>.json`. Un literal como `ROXs12b_realigned` dentro de
`musepipe/` o `scripts/` hace que un objeto nuevo herede en silencio las rutas,
los datos o la prosa del primero — que es exactamente cómo llegó a decir
"such as ROXs 12 B" el informe de ROXs 42B b.

Cobertura: `musepipe/`, `scripts/` y los notebooks generados (`notebooks/**`),
porque los tres literales más dañinos vivían en notebooks.

Contexto: `docs/plan_multiobjeto_notebooks_2026-07-24.md` (WP P6w/P6b).

Para añadir una excepción hay que escribir el motivo en `ALLOWLIST`. Si el
motivo no se puede escribir en una línea, probablemente no sea una excepción.
"""
from __future__ import annotations

import ast
import json
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

#: nombres de objeto que no deben aparecer en el código
TARGET_PATTERN = re.compile(r"ROXs?\s?_?1?2b|ROXs?\s?_?42Bb|LkCa[_ ]?15|YSES[_ ]?2b", re.IGNORECASE)

#: (ruta relativa, motivo). El motivo es obligatorio y se lee en la revisión.
ALLOWLIST: dict[str, str] = {
    # Generador de notebooks: contiene la prosa histórica y los comandos
    # literales de las etapas `kind="cli"` aún sin parametrizar (WP-E4).
    "scripts/build_review_notebooks.py": "prosa y comandos cli pendientes de WP-E4",
    # El propio guardián y el registro de objetos nombran objetos por diseño.
    "tests/test_no_hardcoded_target.py": "el guardián enumera los nombres que vigila",
    "tests/test_stage_registry.py": "comprueba la detección cross-object con runs reales",
    "tests/test_reduction_driver.py": "comprueba que run_id llega al QC en dos objetos",
}

#: extensiones que se revisan
SUFFIXES = {".py", ".ipynb"}

#: subárboles revisados
ROOTS = ("musepipe", "scripts", "notebooks")


def _iter_files():
    for sub in ROOTS:
        base = ROOT / sub
        if not base.is_dir():
            continue
        for path in sorted(base.rglob("*")):
            if path.suffix not in SUFFIXES:
                continue
            if any(part in {"__pycache__", ".ipynb_checkpoints"} for part in path.parts):
                continue
            yield path


def _docstring_nodes(tree: ast.AST) -> set[int]:
    """ids de los nodos Constant que son docstrings (módulo, clase, función)."""
    out: set[int] = set()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Module, ast.ClassDef,
                                 ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        body = getattr(node, "body", None)
        if not body:
            continue
        first = body[0]
        if (isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant)
                and isinstance(first.value.value, str)):
            out.add(id(first.value))
    return out


def _offending_literals(source: str, label: str = ""):
    """(línea, texto) de los literales de cadena que nombran un objeto.

    Se analiza el AST, no el texto plano: los comentarios y los docstrings
    documentan el historial del proyecto y ahí nombrar un objeto es correcto.
    Lo que no vale es un objeto dentro de un valor que el programa **usa**: una
    ruta, un run id, un nombre en una comparación.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return
    docstrings = _docstring_nodes(tree)
    # `RUN_ID = nb.resolve_run_id("<run>")` es el vínculo declarado entre un set
    # de notebooks y su objeto: ahí el literal es la interfaz, no un hardcode.
    bound: set[int] = set()
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and node.args
                and isinstance(node.func, ast.Attribute)
                and node.func.attr in {"resolve_run_id", "show_chain"}):
            bound.add(id(node.args[0]))
    for node in ast.walk(tree):
        if id(node) in bound:
            continue
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if id(node) in docstrings:
                continue
            if TARGET_PATTERN.search(node.value):
                yield getattr(node, "lineno", "?"), f"{label}{node.value[:90]!r}"
        elif isinstance(node, ast.Name) and TARGET_PATTERN.search(node.id):
            yield getattr(node, "lineno", "?"), f"{label}nombre {node.id}"


def _code_units(path: Path):
    """(etiqueta, fuente) de cada unidad de código analizable del fichero.

    En los notebooks solo se miran las celdas de código: el markdown es narrativa.
    """
    if path.suffix == ".py":
        try:
            yield "", path.read_text(encoding="utf-8")
        except OSError:
            return
        return
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return
    for cell_no, cell in enumerate(payload.get("cells", [])):
        if cell.get("cell_type") == "code":
            yield f"celda{cell_no} ", "".join(cell.get("source", []))


class NoHardcodedTargetTests(unittest.TestCase):
    def test_no_target_literals_in_code(self):
        offenders: list[str] = []
        for path in _iter_files():
            rel = path.relative_to(ROOT).as_posix()
            if rel in ALLOWLIST:
                continue
            for label, source in _code_units(path):
                for lineno, text in _offending_literals(source, label):
                    offenders.append(f"{rel}:{lineno}: {text}")
        self.assertEqual(
            offenders, [],
            "Nombres de objeto hardcodeados en código.\n"
            "Muévelos a runs/<run>/config/config.json o a targets/<slug>.json, "
            "o añade el fichero a ALLOWLIST con su motivo.\n  "
            + "\n  ".join(offenders),
        )

    def test_allowlist_entries_exist_and_are_justified(self):
        """Una allowlist con entradas muertas deja de ser una decisión y pasa a ser ruido."""
        for rel, reason in ALLOWLIST.items():
            self.assertTrue((ROOT / rel).exists(), f"entrada muerta en ALLOWLIST: {rel}")
            self.assertTrue(reason.strip(), f"entrada sin motivo: {rel}")

    def test_reduction_modules_have_no_module_level_run_id(self):
        """`RUN_ID = "<objeto>"` a nivel de módulo etiquetaba mal el QC de otros runs."""
        for name in ("esorex_driver", "telluric", "sky_zap"):
            path = ROOT / "musepipe" / "reduction" / f"{name}.py"
            text = path.read_text(encoding="utf-8")
            self.assertNotRegex(
                text, r"(?m)^RUN_ID\s*=",
                f"{name}.py recupera una constante RUN_ID de módulo",
            )


class TargetProfileTests(unittest.TestCase):
    def test_known_targets_have_profiles(self):
        from musepipe.targets import display_name

        self.assertEqual(display_name("ROXs12b", project_root=ROOT), "ROXs 12 B")
        self.assertEqual(display_name("ROXs42Bb", project_root=ROOT), "ROXs 42B b")

    def test_unknown_target_returns_itself_never_another_object(self):
        from musepipe.targets import display_name

        self.assertEqual(display_name("NuevoObjeto", project_root=ROOT), "NuevoObjeto")

    def test_accretion_note_names_the_run_object(self):
        from musepipe.report import accretion_relation_note

        note = accretion_relation_note("ROXs 42B b")
        self.assertIn("ROXs 42B b", note)
        self.assertNotIn("ROXs 12 B", note)


if __name__ == "__main__":
    unittest.main()
