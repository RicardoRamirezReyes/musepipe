"""`h04_x03_overrides`: barrer un knob de C4 sin tocar el config del run.

Lo que protege no es el merge —son tres lineas— sino la propiedad de la que
depende cualquier sonda que compare configuraciones del extractor: **sin la
clave, la resolucion es identica a la de antes**. Si eso se rompe, el barrido
de radios y la cadena dejarian de medir sobre el mismo estimador sin que nada
lo delatara.
"""

import unittest

from musepipe.stages.stage_h04_extractors import _x03_with_overrides


class X03OverridesTest(unittest.TestCase):
    def setUp(self):
        self.x03 = {"x03_star_radius_px": 20.0, "x03_comp_radius_px": 12.0,
                    "x03_error_mode": "auto"}

    def test_sin_clave_devuelve_el_config_del_run(self):
        self.assertEqual(_x03_with_overrides(self.x03, {}), self.x03)

    def test_clave_vacia_o_nula_no_cambia_nada(self):
        for valor in (None, {}):
            with self.subTest(valor=valor):
                self.assertEqual(
                    _x03_with_overrides(self.x03, {"h04_x03_overrides": valor}), self.x03
                )

    def test_solo_pisa_las_claves_declaradas(self):
        salida = _x03_with_overrides(
            self.x03, {"h04_x03_overrides": {"x03_comp_radius_px": 8.0}}
        )
        self.assertEqual(salida["x03_comp_radius_px"], 8.0)
        self.assertEqual(salida["x03_star_radius_px"], 20.0)
        self.assertEqual(salida["x03_error_mode"], "auto")

    def test_no_muta_el_config_del_run(self):
        _x03_with_overrides(self.x03, {"h04_x03_overrides": {"x03_comp_radius_px": 8.0}})
        self.assertEqual(self.x03["x03_comp_radius_px"], 12.0)


if __name__ == "__main__":
    unittest.main()
