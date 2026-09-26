"""G5 dice lo que dicen los QC, no lo que se escribió en julio.

Hasta el 2026-09-26 G5 añadía siempre un bloqueante fijo («PROVISIONAL package on the ADP cube;
A-block open … G3 real fits deferred») y su README titulaba «Hα: non-detection → Ṁ ≲ 5×10⁻¹³»,
que era falso para ROXs 12 b (E1 = `detection`). Ahora el estado provisional se hereda de los
bloqueantes abiertos de verdad (F1 y G0–G4), el titular se lee de E1/E3/G3/G4, y G5 comprueba que
un Ṁ sacado de una L_acc que es cota esté etiquetado como cota.
"""

import json
import tempfile
import unittest
from pathlib import Path

from musepipe.characterization import build_characterization
from tests._g5_fixture import make_run


def _out(root, rid):
    return root / "runs" / rid / "report" / "characterization"


class G5DerivadoTests(unittest.TestCase):
    def test_sin_bloqueantes_aguas_arriba_no_es_provisional(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            rid = make_run(root)
            s = build_characterization(rid, project_root=str(root))
            self.assertFalse(s["provisional"])
            self.assertEqual(s["inherited_blocking_issues"], [])
            texto = json.dumps(s) + (_out(root, rid) / "README.md").read_text()
            for rancio in ("ADP", "deferred", "5×10⁻¹³", "paper-valid"):
                self.assertNotIn(rancio, texto)
            self.assertIn("No blocking issue is open upstream", (_out(root, rid) / "README.md").read_text())

    def test_hereda_los_bloqueantes_de_f1_y_de_las_fases(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            rid = make_run(root)
            rd = root / "runs" / rid
            (rd / "report").mkdir(parents=True, exist_ok=True)
            (rd / "report" / "run_summary.json").write_text(json.dumps({"open_issues": [
                {"issue": "anillo", "priority": "blocking", "stage": "C1_psf"},
                {"issue": "menor", "priority": "major", "stage": "C2_aperture"}]}))
            (rd / "stages" / "stage_g3_qc.json").write_text(json.dumps({"open_issues": [
                {"issue": "ajuste atmosferico", "priority": "blocking"}]}))
            s = build_characterization(rid, project_root=str(root))
            self.assertTrue(s["provisional"])
            self.assertEqual({(i["stage"], i["issue"]) for i in s["inherited_blocking_issues"]},
                             {("C1_psf", "anillo"), ("G3", "ajuste atmosferico")})
            readme = (_out(root, rid) / "README.md").read_text()
            self.assertIn("2 blocking issue(s) are open upstream", readme)
            self.assertIn("anillo", readme)
            self.assertNotIn("menor", readme)

    def test_el_titular_sale_de_e1_e3_y_g3(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            rid = make_run(root)
            st = root / "runs" / rid / "stages"
            (st / "stage_h01_qc.json").write_text(json.dumps({
                "verdict": {"verdict": "detection"},
                "parametric_fap": {"by_method": {"psffit": {"n_null": 33}}}}))
            (st / "stage_h03_qc.json").write_text(json.dumps({
                "canonical_method": "psffit",
                "limits": [{"method": "aperture", "mdot": 9.9e-14},
                           {"method": "psffit", "mdot": 3.3e-14, "throughput_err": 0.08}]}))
            s = build_characterization(rid, project_root=str(root))
            hl = s["headline"]
            self.assertEqual(hl["halpha_e1_verdict"], "detection")
            self.assertEqual(hl["e3_mdot_99_msun_yr"], 3.3e-14)
            self.assertEqual(hl["n_control_positions"], 33)
            readme = (_out(root, rid) / "README.md").read_text()
            self.assertIn("`detection`", readme)
            self.assertIn("3.30e-14", readme)
            presupuesto = (_out(root, rid) / "uncertainty_budget.csv").read_text()
            self.assertIn("33 controls", presupuesto)
            self.assertIn("throughput_psffit", presupuesto)

    def test_un_mdot_de_una_cota_etiquetado_como_medida_bloquea(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            rid = make_run(root)
            (root / "runs" / rid / "tables" / "g3_physical_properties.csv").write_text(
                "property,value,label\nl_acc_combined,1.5e-7,upper_limit\n"
                "mdot,1.9e-13,empirical_inference\n")
            s = build_characterization(rid, project_root=str(root))
            chk = next(c for c in s["consistency"] if c["check"] == "mdot_label_follows_lacc")
            self.assertFalse(chk["consistent"])
            self.assertTrue(any("upper limit" in i["issue"] and i["priority"] == "blocking"
                                for i in s["open_issues"]))


if __name__ == "__main__":
    unittest.main()
