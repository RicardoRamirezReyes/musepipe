import unittest
from unittest import mock

import numpy as np

from musepipe.stages import stage_h03_limits as H03


def h01_row(method="psffit", factor=1.0, sigma=207.0):
    return {"method": method, "template_factor": str(factor), "matched_sigma": str(sigma)}


class MatchedSigmaSourceTests(unittest.TestCase):
    """De donde sale el sigma del filtro adaptado, y por que no puede decidirlo el azar.

    La tabla de deteccion de E1 trae UNA fila por metodo, con el ancho de
    plantilla que E1 prefirio. Cuando E3 leia de ahi, acertaba o fallaba segun
    esa preferencia: dos definiciones de la misma cantidad, elegidas por
    disponibilidad y no por criterio, discrepando un 3.3 % medido.
    """

    def setUp(self):
        self.recalculado = {"matched_sigma": 215.0, "sigma_source": "recomputed_h01_estimator"}
        patcher = mock.patch.object(H03, "_matched_sigma_from_product",
                                    return_value=dict(self.recalculado))
        self.mock_product = patcher.start()
        self.addCleanup(patcher.stop)

    def test_the_config_knob_still_wins(self):
        # Es una declaracion explicita del run: manda sobre cualquier medida.
        cfg = {"h03_matched_sigma_by_method": {"psffit": 999.0}}

        out = H03.matched_sigma_for_method_factor(None, cfg, {}, [h01_row()], "psffit", 1.0)

        self.assertEqual(out["matched_sigma"], 999.0)
        self.assertIn("config", out["sigma_source"])
        self.mock_product.assert_not_called()

    def test_the_table_no_longer_decides_the_value(self):
        # Antes: si E1 prefirio factor 1.0, E3 usaba 207.0 en vez de recalcular.
        out = H03.matched_sigma_for_method_factor(None, {}, {}, [h01_row(factor=1.0)], "psffit", 1.0)

        self.assertEqual(out["matched_sigma"], 215.0)
        self.assertEqual(out["sigma_source"], "recomputed_h01_estimator")

    def test_the_table_value_travels_with_the_result_when_comparable(self):
        out = H03.matched_sigma_for_method_factor(None, {}, {}, [h01_row(factor=1.0, sigma=207.0)],
                                                  "psffit", 1.0)

        self.assertEqual(out["sigma_h01_table"], 207.0)
        self.assertAlmostEqual(out["sigma_table_vs_recomputed_pct"],
                               (207.0 / 215.0 - 1.0) * 100.0, places=6)

    def test_a_table_at_another_factor_is_not_compared(self):
        # Un filtro mas ancho da legitimamente otro sigma: comparar seria mentir.
        out = H03.matched_sigma_for_method_factor(None, {}, {}, [h01_row(factor=2.0)], "psffit", 1.0)

        self.assertEqual(out["matched_sigma"], 215.0)
        self.assertIsNone(out["sigma_h01_table"])
        self.assertIsNone(out["sigma_table_vs_recomputed_pct"])

    def test_the_source_does_not_depend_on_which_template_e1_preferred(self):
        # La clase de fallo entera: mismo run, misma pregunta, y el `sigma_source`
        # cambiaba segun una preferencia de OTRA etapa.
        fuentes = {
            H03.matched_sigma_for_method_factor(
                None, {}, {}, [h01_row(factor=f)], "psffit", 1.0)["sigma_source"]
            for f in (1.0, 2.0)
        }

        self.assertEqual(len(fuentes), 1, f"la via depende del factor que eligio E1: {fuentes}")

    def test_a_missing_product_falls_back_to_the_table_and_says_so(self):
        # El fallback viejo tambien cubria esto: un metodo sin producto calibrado
        # (p.ej. `sgf` en un run que no lo entrega). Forzar el recalculo sin mas
        # rompia E3 entero; ahora hay respaldo, pero DECLARADO.
        self.mock_product.side_effect = RuntimeError("no calibrated product")

        out = H03.matched_sigma_for_method_factor(None, {}, {}, [h01_row(factor=1.0, sigma=207.0)],
                                                  "psffit", 1.0)

        self.assertEqual(out["matched_sigma"], 207.0)
        self.assertIn("fallback", out["sigma_source"])
        self.assertIn("h01_detection_table", out["sigma_source"])

    def test_a_missing_product_with_no_usable_table_still_raises(self):
        # Sin producto y sin tabla al mismo factor no hay respuesta: fallar es
        # lo correcto, inventarla no.
        self.mock_product.side_effect = RuntimeError("no calibrated product")

        with self.assertRaises(RuntimeError):
            H03.matched_sigma_for_method_factor(None, {}, {}, [h01_row(factor=2.0)], "psffit", 1.0)

    def test_no_table_at_all_still_resolves(self):
        out = H03.matched_sigma_for_method_factor(None, {}, {}, [], "psffit", 1.0)

        self.assertEqual(out["matched_sigma"], 215.0)
        self.assertIsNone(out["sigma_h01_table"])


if __name__ == "__main__":
    unittest.main()
