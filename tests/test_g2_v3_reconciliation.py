"""V3 de G2 (spec G2 v3): qué veredictos de E1 caben con cada estado de Hα.

Hasta la v3, `marginal` solo casaba con `candidate`, y ROXs 42B b saltó como
bloqueante con la misma señal de ~3σ a los dos lados (G2 z = 3.09, E1 z = 2.97,
FAP 0.011). La puerta tiene que seguir disparando ante una contradicción de
verdad: eso es lo que fijan los dos últimos tests.
"""
import unittest

from musepipe.stages.stage_g2_measure_lines import halpha_reconciliation_v3


class V3ReconciliationTests(unittest.TestCase):
    def test_marginal_convive_con_candidate_y_con_non_detection(self):
        for e1 in ("candidate", "non_detection"):
            self.assertTrue(halpha_reconciliation_v3("marginal", e1)["consistent"], e1)

    def test_el_caso_de_42bb(self):
        v3 = halpha_reconciliation_v3("marginal", "non_detection")
        self.assertTrue(v3["consistent"])
        self.assertEqual(v3["rule"], "spec_G2_v3")

    def test_los_pares_directos_siguen_igual(self):
        self.assertTrue(halpha_reconciliation_v3("detected", "detection")["consistent"])
        self.assertTrue(halpha_reconciliation_v3("upper_limit", "non_detection")["consistent"])
        self.assertTrue(halpha_reconciliation_v3("not_measurable", "non_detection")["consistent"])

    def test_detectada_frente_a_no_deteccion_sigue_disparando(self):
        """El caso de la v1 del spec: la razón de ser de V3."""
        self.assertFalse(halpha_reconciliation_v3("detected", "non_detection")["consistent"])
        self.assertFalse(halpha_reconciliation_v3("detected", "candidate")["consistent"])

    def test_limite_frente_a_deteccion_sigue_disparando(self):
        self.assertFalse(halpha_reconciliation_v3("upper_limit", "detection")["consistent"])
        self.assertFalse(halpha_reconciliation_v3("marginal", "detection")["consistent"])


if __name__ == "__main__":
    unittest.main()
