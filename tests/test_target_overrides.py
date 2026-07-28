"""La tercera capa de generacion (lo propio de un objeto) y sus invariantes.

El riesgo de esta capa es recrear, en grande, el problema que ya costo caro con
los notebooks editados a mano: un notebook que no coincide con lo que dice su
generador y sin nada que lo delate. Por eso se prueba que **todo notebook
modificado lo declara**, que el override **solo toca las etapas que declara**, y
que se **ancla en el texto de las secciones** y no en indices, para que fallar
sea ruidoso si el generador general reordena.
"""

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from target_overrides import (  # noqa: E402
    BANNER_PREFIX,
    apply_overrides,
    find_markdown,
    insert_after_markdown,
    load_override,
    replace_in_code,
)


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": text.splitlines(keepends=True)}


def _cells():
    return [md("# Titulo"), md("## 5 · Paso 1"), code("x = 1\n"),
            md("## 6 · Paso 2"), code("y = 2\n")]


class MachineryTests(unittest.TestCase):
    def test_insert_lands_at_the_end_of_the_named_section(self):
        out = insert_after_markdown(_cells(), "## 5 ", md("nuevo"))
        self.assertEqual("".join(out[3]["source"]), "nuevo")
        self.assertTrue("".join(out[4]["source"]).startswith("## 6 "))

    def test_a_missing_anchor_fails_loudly(self):
        """Si el generador general renombra la seccion, hay que enterarse."""

        with self.assertRaises(KeyError):
            insert_after_markdown(_cells(), "## 99 ", md("nuevo"))
        with self.assertRaises(KeyError):
            find_markdown(_cells(), "## 99 ")

    def test_replace_in_code_fails_when_the_target_moved(self):
        with self.assertRaises(KeyError):
            replace_in_code(_cells(), "no_existe", "otra_cosa")

    def test_targets_without_an_override_are_untouched(self):
        cells = _cells()
        out, applied = apply_overrides(cells, target="NoExiste", stage_id="C2",
                                       run_id="r", kind="debug", md=md, code=code)
        self.assertFalse(applied)
        self.assertIs(out, cells)


class Roxs42BbOverrideTests(unittest.TestCase):
    def setUp(self):
        self.module = load_override("ROXs42Bb")
        if self.module is None:
            self.skipTest("ROXs42Bb no tiene override")

    def test_it_declares_stages_and_a_reason(self):
        self.assertTrue(set(self.module.STAGES))
        self.assertGreater(len(self.module.REASON), 40)

    def test_it_only_touches_the_stages_it_declares(self):
        for stage in ("C4", "C5", "C6"):
            if stage in self.module.STAGES:
                continue
            cells = _cells()
            out, applied = apply_overrides(cells, target="ROXs42Bb", stage_id=stage,
                                           run_id="r", kind="debug", md=md, code=code)
            self.assertFalse(applied, f"{stage} no esta declarada y aun asi se modifico")
            self.assertIs(out, cells)

    def test_the_source_position_matches_the_published_astrometry(self):
        """ROXs 42B cc1 (Bryan+2016) extrapolado a 2022.6654 cae en r=15.1 px del
        centro; B3 lo detecto a S/N=9.4 a 1.5 px de esa prediccion. La posicion
        del override tiene que ser esa, no una inventada."""

        self.assertEqual(len(self.module.FIELD_SOURCES_YX), 1)
        y, x = self.module.FIELD_SOURCES_YX[0]
        # Marco de B3 (170x170), primaria en ~(85, 85).
        r = ((y - 85.0) ** 2 + (x - 85.0) ** 2) ** 0.5
        self.assertAlmostEqual(r, 15.1, delta=2.0)

    def test_an_overridden_notebook_declares_it(self):
        cells = _cells()
        out, applied = apply_overrides(cells, target="ROXs42Bb", stage_id="C2",
                                       run_id="r", kind="debug", md=md, code=code)
        self.assertTrue(applied)
        self.assertTrue("".join(out[0]["source"]).startswith(BANNER_PREFIX))
        self.assertIn("ROXs42Bb.py", "".join(out[0]["source"]))


class GeneratedNotebookTests(unittest.TestCase):
    """Los `.ipynb` en disco tienen que llevar el aviso si su objeto tiene override."""

    def test_generated_notebooks_carry_the_banner(self):
        for target_dir in sorted((ROOT / "notebooks").iterdir()):
            if not target_dir.is_dir() or target_dir.name.startswith("_"):
                continue
            module = load_override(target_dir.name)
            if module is None:
                continue
            for stage in sorted(module.STAGES):
                matches = sorted(target_dir.glob(f"debug/{stage}_*.ipynb"))
                for path in matches:
                    with self.subTest(notebook=str(path.relative_to(ROOT))):
                        payload = json.loads(path.read_text(encoding="utf-8"))
                        first = "".join(payload["cells"][0]["source"])
                        self.assertTrue(
                            first.startswith(BANNER_PREFIX),
                            "notebook de un objeto con override sin el aviso: regenera",
                        )


if __name__ == "__main__":
    unittest.main()
