"""WP-G3R-11: systematics-limited relabeling."""

import unittest

from musepipe.stages.stage_g3_accretion import TABLE_FIELDS
from musepipe.stages.stage_g3_assemble import _accretion_keys, _relabel_systematics_limited


def _row(**kw):
    r = {k: "" for k in TABLE_FIELDS}
    r.update(kw)
    return r


class RelabelTests(unittest.TestCase):
    def test_relabel_atmo_and_mdot(self):
        rows = [
            _row(property="teff", value="2500", label="atmospheric_model_dependent",
                 depends_on="[atmospheric_model]"),
            _row(property="mass", value="19.2", label="evolutionary_model_dependent"),
            _row(property="l_acc_combined", value="4.5e-6", label="upper_limit"),
            _row(property="mdot", value="2.2e-12", label="empirical_inference"),
            _row(property="age_used", value="6.0", label="evolutionary_model_dependent"),
        ]
        out = _relabel_systematics_limited(
            rows, reason="C3 continuum systematic", citation="cite",
            mdot_lit={"p50": 1.4e-12, "p16": 0.7e-12, "p84": 3.0e-12})
        d = {r["property"]: r for r in out}
        # atmosphere-dependent -> not_constrained, value blanked, railed value kept in note
        self.assertEqual(d["teff"]["label"], "not_constrained")
        self.assertEqual(d["teff"]["value"], "")
        self.assertIn("2500", d["teff"]["limitations"])
        self.assertEqual(d["mass"]["label"], "not_constrained")
        # accretion limit kept as-is
        self.assertEqual(d["l_acc_combined"]["value"], "4.5e-6")
        # mdot -> literature M,R
        self.assertEqual(float(d["mdot"]["value"]), 1.4e-12)
        self.assertIn("literature", d["mdot"]["assumptions"])
        # adopted prior kept
        self.assertEqual(d["age_used"]["value"], "6.0")

    def test_labels_all_valid(self):
        from musepipe.models import validate_label
        rows = [_row(property="teff", value="2500", label="atmospheric_model_dependent")]
        out = _relabel_systematics_limited(rows, reason="x", citation="c", mdot_lit=None)
        self.assertTrue(all(validate_label(r["property"], r["label"]) for r in out))



class AccretionKeysTests(unittest.TestCase):
    """El QC final de G3 conserva lo que publica la rodaja de acrecion (2026-09-26)."""

    def test_el_mdot_y_su_etiqueta_viajan_al_qc_final(self):
        acc = {"combined_accretion": {"kind": "upper_limit", "l_acc_lsun": 1.5e-7},
               "mdot_p50_msun_yr": 1.9e-13, "mdot_mr_source": {"source": "config"}}
        rows = [_row(property="mdot", value="1.9e-13", label="upper_limit")]
        k = _accretion_keys(acc, rows)
        self.assertEqual(k["combined_accretion"]["kind"], "upper_limit")
        self.assertAlmostEqual(k["mdot_p50_msun_yr"], 1.9e-13)
        self.assertEqual(k["mdot_label"], "upper_limit")
        self.assertEqual(k["mdot_mr_source"], {"source": "config"})

    def test_tras_reetiquetar_publica_el_mdot_recalculado(self):
        rows = [_row(property="l_acc_combined", value="4.5e-6", label="empirical_inference"),
                _row(property="mdot", value="2.2e-12", label="empirical_inference")]
        out = _relabel_systematics_limited(rows, reason="r", citation="c",
                                           mdot_lit={"p50": 1.4e-12, "p16": 0.7e-12, "p84": 3.0e-12})
        k = _accretion_keys({"mdot_p50_msun_yr": 2.2e-12}, out)
        self.assertAlmostEqual(k["mdot_p50_msun_yr"], 1.4e-12)

if __name__ == "__main__":
    unittest.main()
