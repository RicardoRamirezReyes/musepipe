"""El sesgo del fondo local viaja con los datos, medido o no.

El anillo de fondo se come parte de las alas de la compañera: -3.5 % en
ROXs 12 b y -1.6 % en ROXs 42B b al radio en produccion, medido por
inyeccion-recuperacion el 2026-09-12. La apcorr no lo corrige -corrige perdidas
de APERTURA, no lo que el fondo resta- y hasta ahora no estaba en ningun sitio
salvo en el manuscrito: quien reutilizara estos espectros sin leerlo no lo veia.

La LINEA ya lo lleva absorbido, porque el throughput de E4 se mide con el mismo
tratamiento de fondo. El CONTINUO no, y de ahi salen las atmosferas.
"""
import unittest

from musepipe.stages.stage_x11_calibrate import CalibrationCorrections, _error_budget_rows

import numpy as np


def _filas(**kw):
    corr = CalibrationCorrections(**kw)
    ceros = np.zeros(4)
    return {r["term"]: r for r in _error_budget_rows(
        ceros, ceros, ceros, ceros, ceros, ceros, corr)}


class SesgoDelFondoTests(unittest.TestCase):
    def test_la_fila_existe_aunque_no_se_haya_medido(self):
        """Un termino ausente se lee como inexistente: por eso se declara."""
        fila = _filas()["background_selfsub"]
        self.assertIsNone(fila["value"])
        self.assertIn("NO MEDIDO", fila["source"])
        self.assertIn("m_elige_r_out", fila["note"])

    def test_cuando_se_declara_lleva_su_valor_y_su_procedencia(self):
        fila = _filas(bkg_selfsub_frac=-0.035,
                      bkg_selfsub_source="i_inyeccion_continuo 2026-09-12")[
            "background_selfsub"]
        self.assertAlmostEqual(fila["value"], -0.035)
        self.assertIn("2026-09-12", fila["source"])

    def test_es_un_sesgo_con_signo_y_no_se_pliega(self):
        """No es un error simetrico y no entra en `flux_err_total`: plegarlo
        moveria decisiones congeladas, y sumarlo en cuadratura perderia el signo."""
        fila = _filas(bkg_selfsub_frac=-0.016)["background_selfsub"]
        self.assertEqual(fila["type"], "bias_declared_not_applied")
        self.assertIsNone(fila["median"])
        self.assertIn("SESGO", fila["note"])
        self.assertIn("flux_err_total", fila["note"])

    def test_la_nota_dice_que_la_linea_ya_lo_lleva_y_el_continuo_no(self):
        """Es la asimetria que decide a quien afecta: E4 mide su throughput con
        el mismo fondo, asi que la linea esta cubierta; el continuo llega crudo."""
        nota = _filas(bkg_selfsub_frac=-0.035)["background_selfsub"]["note"]
        self.assertIn("LINEA", nota)
        self.assertIn("E4", nota)
        self.assertIn("CONTINUO", nota)

    def test_sin_declarar_no_se_inventa_un_cero(self):
        """Un cero silencioso diria que el efecto no existe, que es justo lo que
        no se sabe si el run no lo ha medido."""
        self.assertIsNone(CalibrationCorrections().bkg_selfsub_frac)


if __name__ == "__main__":
    unittest.main()
