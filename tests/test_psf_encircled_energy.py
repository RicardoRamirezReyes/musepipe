"""V4 de la spec C1: la energia encapsulada del modelo contra la del dato.

El test que importa es el segundo: fabrica el fallo que la metrica del anillo NO
puede ver —un modelo que clava el halo en el radio del compañero y tiene el
nucleo del todo equivocado— y demuestra que V4 sí lo ve. Es el caso real de la
rama Moffat en ROXs 42B b, donde el anillo salia al 6.9 % y la razon
`F(<=25)/F(box3)` estaba un ~375 % por encima del dato, que es justo el factor
con el que C2/C3 multiplican el flujo.
"""
import unittest

import numpy as np

from musepipe.psf import (
    companion_ring_metric,
    core_to_norm_ratio,
    encircled_energy,
    encircled_energy_metric,
    moffat_image,
    source_mask,
)

FORMA = (91, 91)
CENTRO = (45.0, 45.0)


def escena(fwhm, beta, amplitude=1.0, background=0.0):
    return moffat_image(FORMA, CENTRO[0], CENTRO[1], fwhm, fwhm, 0.0, beta,
                        amplitude=amplitude, background=background)


class EncircledEnergyTests(unittest.TestCase):
    def test_modelo_identico_al_dato_no_tiene_error(self):
        dato = escena(4.0, 2.5, amplitude=100.0, background=3.0)
        m = encircled_energy_metric(dato, dato.copy(), CENTRO, norm_radius_px=25.0,
                                    image_background=3.0, model_background=3.0)
        self.assertAlmostEqual(m["core_ratio_error_pct"], 0.0, places=9)
        self.assertAlmostEqual(m["growth_curve_max_abs_diff_pct"], 0.0, places=9)
        self.assertGreater(m["core_ratio_data"], 1.0)

    def test_ve_el_nucleo_equivocado_que_la_metrica_del_anillo_no_ve(self):
        # Dato: nucleo estrecho. Modelo: nucleo el doble de ancho, con la
        # amplitud ajustada para que los dos den el MISMO nivel de halo en el
        # radio del compañero. El anillo queda contento; el nucleo no.
        dato = escena(4.0, 1.6, amplitude=1000.0)
        ancho = escena(8.0, 1.6, amplitude=1.0)
        companion = (45.0, 75.0)
        yy, xx = np.indices(FORMA, dtype=np.float64)
        anillo = np.abs(np.hypot(yy - CENTRO[0], xx - CENTRO[1]) - 30.0) <= 1.5
        modelo = ancho * (float(np.median(dato[anillo])) / float(np.median(ancho[anillo])))

        ring = companion_ring_metric(dato, modelo, CENTRO, companion, width_px=3.0)
        self.assertLess(ring["median_pct"], 10.0, "el anillo tenia que salir bueno")

        m = encircled_energy_metric(dato, modelo, CENTRO, norm_radius_px=25.0)
        self.assertGreater(abs(m["core_ratio_error_pct"]), 50.0,
                           "V4 tiene que ver el nucleo que el anillo no ve")
        self.assertGreater(m["growth_curve_max_abs_diff_pct"], 5.0)

    def test_la_exclusion_se_aplica_a_los_dos_lados(self):
        # Una vecina DENTRO del radio de normalizacion —el caso de ROXs 42B b,
        # con la fuente de campo a 16 px de la primaria— infla el numerador y no
        # la caja: sin enmascararla, el cociente del dato sale sesgado al alza.
        dato = escena(4.0, 2.5, amplitude=100.0)
        vecina = moffat_image(FORMA, 45.0, 57.0, 3.0, 3.0, 0.0, 2.5, amplitude=80.0)
        mascara = source_mask(FORMA, [(45.0, 57.0)], 12.0)

        limpio = core_to_norm_ratio(dato, CENTRO, norm_radius_px=25.0)
        con_vecina = core_to_norm_ratio(dato + vecina, CENTRO, norm_radius_px=25.0)
        self.assertGreater(con_vecina, limpio * 1.02, "la vecina tenia que sesgar el cociente")

        # Y con la misma mascara a los dos lados, la vecina deja de contar.
        enmascarado = core_to_norm_ratio(dato + vecina, CENTRO, norm_radius_px=25.0,
                                         exclude_mask=mascara)
        referencia = core_to_norm_ratio(dato, CENTRO, norm_radius_px=25.0, exclude_mask=mascara)
        # Lo que queda son las alas de la vecina fuera de la mascara: un radio
        # finito no las quita del todo, y eso vale para el dato y para el modelo.
        self.assertLess(abs(enmascarado - referencia) / referencia, 0.01)

    def test_la_energia_encapsulada_crece_y_respeta_el_fondo(self):
        dato = escena(4.0, 2.5, amplitude=100.0, background=7.0)
        radios = [2.0, 5.0, 10.0, 25.0]
        curva = encircled_energy(dato, CENTRO, radios, background=7.0)
        self.assertTrue(np.all(np.diff(curva) > 0))
        sin_restar = encircled_energy(dato, CENTRO, radios, background=0.0)
        self.assertGreater(sin_restar[-1], curva[-1])


if __name__ == "__main__":
    unittest.main()
