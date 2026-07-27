"""V1 de C4: ¿describe el ajuste al dato con el error que declara?

El QC guardaba `chi2r.median` desde siempre y nadie lo miraba: en ROXs 42B b
vale 2673 —el residuo del ajuste es ~2700 veces la varianza declarada— y el
único `open_issue` de esa etapa hablaba del marco de longitud de onda.

Lo que se fija aquí es que el chequeo distinga los dos casos que la spec separa:
con STAT utilizable, χ²ᵣ tiene escala absoluta y el veredicto vale; sin él, no
se fabrica un aprobado.
"""
import unittest
from types import SimpleNamespace

import numpy as np

from musepipe.stages.stage_x03_psffit import CHI2R_RANGE, _chi2r_check


def _fit(chi2r):
    return SimpleNamespace(chi2r=np.asarray(chi2r, dtype=np.float64))


class Chi2rCheckTests(unittest.TestCase):
    def test_a_model_that_describes_the_data_passes(self):
        out = _chi2r_check(_fit(np.full(100, 1.05)), {}, stat_usable=True)
        self.assertTrue(out["ok"])
        self.assertAlmostEqual(out["median"], 1.05)
        self.assertEqual(out["fraction_in_range"], 1.0)

    def test_a_chi2r_that_blows_up_fails(self):
        # El caso de ROXs 42B b: el error formal no describe el residuo.
        out = _chi2r_check(_fit(np.full(100, 2673.0)), {}, stat_usable=True)
        self.assertFalse(out["ok"])
        self.assertEqual(out["fraction_in_range"], 0.0)

    def test_errors_overestimated_also_fails(self):
        """No solo se vigila por arriba: χ²ᵣ<<1 son barras infladas."""
        out = _chi2r_check(_fit(np.full(100, 0.05)), {}, stat_usable=True)
        self.assertFalse(out["ok"])

    def test_without_a_usable_stat_there_is_no_verdict(self):
        # La spec condiciona V1 a "STAT verde": sin varianza fiable, χ²ᵣ no
        # tiene escala absoluta. Se informa la cifra, pero no se aprueba.
        out = _chi2r_check(_fit(np.full(100, 1.0)), {}, stat_usable=False)
        self.assertIsNone(out["ok"])
        self.assertIn("escala absoluta", out["reason"])
        self.assertAlmostEqual(out["median"], 1.0)

    def test_it_lists_the_worst_channels_to_cross_check_by_hand(self):
        values = np.full(50, 1.0)
        values[7] = 900.0
        values[3] = 500.0
        out = _chi2r_check(_fit(values), {}, stat_usable=True)
        peores = [w["channel"] for w in out["worst_channels"]]
        self.assertEqual(peores[:2], [7, 3])
        self.assertEqual(out["worst_channels"][0]["chi2r"], 900.0)

    def test_the_worst_channels_survive_the_nan_ones(self):
        """Con NaN en el cubo (215 canales en los runs reales) la lista salía
        vacía: `argsort` los manda al final y, al invertir, se colaban ellos."""
        values = np.full(50, 1.0)
        values[:20] = np.nan
        values[31] = 700.0
        out = _chi2r_check(_fit(values), {}, stat_usable=True)
        self.assertEqual(len(out["worst_channels"]), 5)
        self.assertEqual(out["worst_channels"][0]["channel"], 31)
        self.assertTrue(all(np.isfinite(w["chi2r"]) for w in out["worst_channels"]))

    def test_the_range_is_configurable(self):
        fit = _fit(np.full(100, 3.0))
        self.assertFalse(_chi2r_check(fit, {}, stat_usable=True)["ok"])
        out = _chi2r_check(fit, {"x03_chi2r_range": [0.5, 5.0]}, stat_usable=True)
        self.assertTrue(out["ok"])
        self.assertEqual(out["range"], [0.5, 5.0])

    def test_non_finite_channels_are_ignored_and_do_not_fabricate_a_verdict(self):
        values = np.full(20, np.nan)
        values[:5] = 1.0
        out = _chi2r_check(_fit(values), {}, stat_usable=True)
        self.assertTrue(out["ok"])
        self.assertEqual(out["n_channels"], 5)
        self.assertIsNone(_chi2r_check(_fit(np.full(10, np.nan)), {}, stat_usable=True)["ok"])

    def test_the_default_range_is_a_factor_two(self):
        """Un factor 2 en χ²ᵣ es √2 en σ: por debajo no se distingue un modelo
        imperfecto de un STAT mal escalado."""
        self.assertEqual(tuple(CHI2R_RANGE), (0.5, 2.0))


if __name__ == "__main__":
    unittest.main()
