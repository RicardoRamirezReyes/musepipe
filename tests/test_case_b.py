"""La tabla del caso B y las dos cotas de A_V que salen de ella.

Los valores de referencia son los clasicos de Storey & Hummer (1995) a
T = 10^4 K y n_e = 10^2 cm^-3: emisividad de Hbeta 1.235e-25 erg cm^3 s^-1 y
Halpha/Hbeta = 2.863. Si el lector se equivoca de indice, salen otros.
"""
import math
import unittest

from musepipe.models import case_b


class TablaTests(unittest.TestCase):
    def test_reproduce_los_valores_clasicos(self):
        tab = case_b.load_case_b_table()
        it = tab["temperatures_K"].index(10000.0)
        idens = tab["densities_cm3"].index(100.0)
        self.assertAlmostEqual(tab["emissivity"]["Hbeta"][it][idens], 1.235e-25, places=30)
        self.assertAlmostEqual(case_b.case_b_ratio("Halpha", "Hbeta"), 2.863, places=3)

    def test_declara_su_procedencia(self):
        tab = case_b.load_case_b_table()
        self.assertIn("Storey", tab["citation"])
        self.assertEqual(len(tab["source_sha256"]), 64)

    def test_paschen_mas_debil_cuanto_mas_alta(self):
        r = [case_b.case_b_ratio(f"Pa {n}", "Hbeta") for n in range(9, 19)]
        self.assertTrue(all(a > b for a, b in zip(r, r[1:])))

    def test_fuera_de_nodo_para_en_vez_de_interpolar(self):
        with self.assertRaisesRegex(ValueError, "not a node"):
            case_b.case_b_ratio("Halpha", "Hbeta", temperature_K=9000.0)


class CotasTests(unittest.TestCase):
    def test_cota_superior_se_satura_justo_en_el_limite(self):
        # una Paschen que a A_V = 3 saldria exactamente en su limite
        r, k_det, k_lim, av = 100.0, 0.82, 0.45, 3.0
        f_det = 1.0
        f_lim = f_det * 10 ** (0.4 * av * (k_det - k_lim)) / r
        self.assertAlmostEqual(case_b.av_upper_limit_undetected_redder(f_det, f_lim, r, k_det, k_lim), av)

    def test_cota_inferior_se_satura_justo_en_el_limite(self):
        r, k_det, k_lim, av = 2.86, 0.82, 1.16, -2.0
        f_det = 1.0
        f_lim = f_det * 10 ** (-0.4 * av * (k_lim - k_det)) / r
        self.assertAlmostEqual(case_b.av_lower_limit_undetected_bluer(f_det, f_lim, r, k_det, k_lim), av)

    def test_el_sentido_de_las_lineas_se_comprueba(self):
        with self.assertRaisesRegex(ValueError, "bluer"):
            case_b.av_upper_limit_undetected_redder(1.0, 1.0, 10.0, 0.4, 0.8)
        with self.assertRaisesRegex(ValueError, "redder"):
            case_b.av_lower_limit_undetected_bluer(1.0, 1.0, 2.86, 1.2, 0.8)

    def test_un_limite_no_positivo_no_es_una_cota(self):
        with self.assertRaises(ValueError):
            case_b.av_upper_limit_undetected_redder(1.0, -1.0, 10.0, 0.8, 0.4)
        self.assertTrue(math.isfinite(case_b.av_upper_limit_undetected_redder(1.0, 0.5, 10.0, 0.8, 0.4)))


if __name__ == "__main__":
    unittest.main()
