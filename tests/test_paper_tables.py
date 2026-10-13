"""La columna `Val.` de `tab:lines` sale de G1 (`scripts/paper_tables.py`), sin datos reales."""
from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _modulo():
    spec = importlib.util.spec_from_file_location("paper_tables", ROOT / "scripts" / "paper_tables.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _Objeto:
    """Lo minimo de `Objeto` que lee `_validacion`: el nombre y el QC de G1."""

    def __init__(self, veredictos):
        self.nombre = "ROXs 42B b"
        self._g1 = {"method_verdicts": veredictos}

    def qc(self, nombre):
        assert nombre == "stage_g1_qc.json", nombre
        return self._g1


class TestColumnaDeValidacion(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pt = _modulo()

    def setUp(self):
        self.pt.AVISOS.clear()

    def test_el_asterisco_sigue_a_g1(self):
        celdas = self.pt._validacion(_Objeto({"optimal_ls": "validated", "sgf": "validated_with_bias"}))
        self.assertEqual(celdas["optimal_ls"], "ok")
        self.assertEqual(celdas["sgf"], "ok\\(^{\\ast}\\)")
        self.assertEqual(self.pt.AVISOS, [])

    def test_un_rechazo_avisa_porque_la_nota_c_dejaria_de_ser_cierta(self):
        celdas = self.pt._validacion(_Objeto({"aperture": "rejected"}))
        self.assertEqual(celdas["aperture"], "rej.")
        self.assertEqual(len(self.pt.AVISOS), 1)
        self.assertIn("RECHAZA", self.pt.AVISOS[0])

    def test_un_veredicto_desconocido_no_se_imprime(self):
        with self.assertRaises(ValueError):
            self.pt._validacion(_Objeto({"aperture": "validated_maybe"}))


if __name__ == "__main__":
    unittest.main()
