"""Cada `load_qc` de los notebooks generados resuelve, o es una etapa pendiente.

`test_review_notebook_contract.py` valida los puntos de entrada y la sintaxis,
pero nunca comprobó que las rutas de QC RESUELVAN. Ese hueco fue el origen de
todo el plan multi-objeto: el set de ROXs 42B b fallaba en 20 llamadas y el
error engañoso decía "¿Corriste la etapa?".

Este test hace ejecutable la línea base de la Fase 0. Falla si:
  * una `load_qc` no-opcional apunta a un QC que no resuelve NI es una etapa
    legítimamente pendiente, o
  * una `load_qc` resuelve a un run de OTRO objeto (contaminación cross-object).

`skipif` cuando `runs/` no está poblado, para no atar la suite al disco.

Contexto: `docs/plan_multiobjeto_notebooks_2026-07-24.md` (WP L3).
"""
from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NB_DIR = ROOT / "notebooks"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(NB_DIR))


def _load_builder():
    spec = importlib.util.spec_from_file_location(
        "build_review_notebooks", ROOT / "scripts" / "build_review_notebooks.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _object_dirs():
    """Subcarpetas notebooks/<obj>/ con al menos un .ipynb."""
    if not NB_DIR.is_dir():
        return []
    return [d for d in sorted(NB_DIR.iterdir())
            if d.is_dir() and d.name not in {"__pycache__", ".ipynb_checkpoints"}
            and any(d.glob("*.ipynb"))]


class NotebookQcResolutionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.builder = _load_builder()
        cls.object_dirs = _object_dirs()

    def test_layout_is_symmetric(self):
        """No debe quedar ningún .ipynb suelto en la raíz de notebooks/."""
        if not NB_DIR.is_dir():
            self.skipTest("notebooks/ ausente")
        loose = list(NB_DIR.glob("*.ipynb"))
        self.assertEqual(loose, [], f"notebooks sueltos en la raíz: {[p.name for p in loose]}")

    def test_every_object_resolves_or_is_pending(self):
        if not self.object_dirs:
            self.skipTest("no hay notebooks/<obj>/ poblados")
        for obj_dir in self.object_dirs:
            run_id = self._run_for(obj_dir.name)
            if run_id is None:
                continue  # objeto sin run en disco: nada que resolver
            with self.subTest(objeto=obj_dir.name):
                problems = self.builder.check_qc_resolution(obj_dir, run_id)
                self.assertEqual(
                    problems, [],
                    f"{obj_dir.name}: rutas cross-object o irresolubles:\n  "
                    + "\n  ".join(problems),
                )

    def test_the_launch_command_points_at_the_primary_not_15px_off(self):
        """La posición de las plantillas está en el marco del cubo que reciben.

        `stage01c_qc.json` mide la primaria en el marco RECORTADO de B1 (170 px),
        y `{cube}` de las plantillas es el cubo SIN recortar (200 px). Sin sumar
        el desfase del recorte, el comando de A3 apunta 15 px fuera de la
        estrella — medía sobre vacío y nadie lo veía porque A3 se lanzó a mano.
        """
        import json
        import _nbcommon as nb

        if not self.object_dirs:
            self.skipTest("no hay notebooks/<obj>/ poblados")
        for obj_dir in self.object_dirs:
            run_id = self._run_for(obj_dir.name)
            if run_id is None:
                continue
            stages = ROOT / "runs" / run_id / "stages"
            if not (stages / "stage01c_qc.json").exists() or not (stages / "stage01_qc.json").exists():
                continue  # B1/B3 sin correr para este objeto
            with self.subTest(objeto=obj_dir.name):
                nb.resolve_run_id(run_id)
                ctx = nb._launch_context(run_id)
                centro = json.loads((stages / "stage01_qc.json").read_text(
                    encoding="utf-8"))["crop"]["center_yx"]
                self.assertIsNotNone(ctx["primary_y"], "sin posición de primaria")
                dist = ((ctx["primary_y"] - centro[0]) ** 2
                        + (ctx["primary_x"] - centro[1]) ** 2) ** 0.5
                self.assertLess(
                    dist, 1.0,
                    f"{obj_dir.name}: el comando apunta a "
                    f"({ctx['primary_y']:.2f}, {ctx['primary_x']:.2f}) y la primaria "
                    f"está en ({centro[0]:.2f}, {centro[1]:.2f}) — {dist:.1f} px fuera",
                )

    def _run_for(self, obj: str) -> str | None:
        """Run por defecto de la cadena del objeto (o None si no existe en disco)."""
        for candidate in (f"{obj}_realigned", obj):
            if (ROOT / "runs" / candidate / "config" / "config.json").exists():
                return candidate
        return None


if __name__ == "__main__":
    unittest.main()
