"""D1 nombraba la evidencia debil y escondia la fuerte.

En `_classify_pair_rows`, una banda con `p < p_strong` ponia `strong = True` y
**no se guardaba en ninguna lista**: solo las `marginal` (0.0027 < p < 0.0455)
se registraban. Resultado medido el 2026-09-10 sobre ROXs 12 b
(`docs/2026-09-10_apertura_y_cola_parametrica.md` §1.1):
`psffit_vs_optimal_ls` listaba como `marginal` solo B4 (|t| = 2.11) y escondia
B5 y B6, que son las que disparan el veredicto; `optimal_ls_vs_aperture`
listaba CERO bandas y su B6 era -65 sigma.

La puerta funcionaba; el que no podia leerla era quien abriera el QC.
"""
import unittest

from musepipe.stages.stage_x10_compare import (
    DEFAULT_P_DIVERGENT,
    DEFAULT_P_STRONG,
    compare_methods,
)
from tests.test_compare_verdicts import clean_controls, make_products


class StrongBandsReportedTests(unittest.TestCase):
    def setUp(self):
        products = make_products({"psffit": [(4900.0, 5400.0, 5.0), (6100.0, 6400.0, 5.0)]})
        _rows, _controls, self.qc = compare_methods(products, clean_controls())
        self.assertEqual(self.qc["verdict"], "divergent_continuum")

    def test_las_bandas_fuertes_se_listan(self):
        self.assertTrue(self.qc["strong"], "el veredicto lo dispara evidencia que nadie puede leer")
        for item in self.qc["strong"]:
            self.assertLess(item["p"], DEFAULT_P_STRONG)
            self.assertIn("band", item)
            self.assertIn("t", item)

    def test_las_dos_listas_no_se_solapan(self):
        strong = {(i["pair"], i["band"]) for i in self.qc["strong"]}
        marginal = {(i["pair"], i["band"]) for i in self.qc["marginal"]}
        self.assertFalse(strong & marginal)
        for item in self.qc["marginal"]:
            self.assertGreaterEqual(item["p"], DEFAULT_P_STRONG)
            self.assertLess(item["p"], DEFAULT_P_DIVERGENT)

    def test_ningun_par_divergente_se_queda_sin_bandas(self):
        """El caso real: un par `divergent_continuum` que no nombraba ni una banda."""
        for pid, pair in self.qc["verdict_by_pair"].items():
            # v4 anida por observable (`continuum`/`lines`) y deja `continuum`
            # a None en los pares excluidos del continuo (sgf/lpm); v2 no anida.
            block = pair.get("continuum", pair) if "continuum" in pair else pair
            if not isinstance(block, dict) or block.get("verdict") != "divergent_continuum":
                continue
            listed = len(block.get("strong", [])) + len(block.get("marginal", []))
            self.assertGreater(listed, 0, f"{pid} diverge y no declara que banda lo hace")


if __name__ == "__main__":
    unittest.main()
