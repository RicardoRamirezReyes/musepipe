"""`optimal_psfsub` apila dos tratamientos de fondo, y aqui esta el knob.

`psfsub` resta el modelo de PSF de la primaria **y ademas** aplica el anillo
local alrededor de cada centro. Es el mismo patron que tenia `optimal_ls` -dos
restas de fondo sobre el mismo cubo-, que le producia el 93 % de su continuo
negativo (mediana en banda roja -2045 con el, -140 sin el) y que se corrigio el
2026-07-26 con `x02_wings_intact_ls`. A `psfsub` nunca se le dio el equivalente.

**La simetria no es literal.** LS puede soltar la PRIMERA resta y quedarse con el
anillo, porque extraer del cubo crudo es justo lo que lo hace comparable con C2.
`psfsub` no puede: restar el modelo ES la variante. Asi que lo que se suelta es
el ANILLO.

**Y va APAGADO por defecto.** A LS se le encendio con la medida delante; para
`psfsub` esa medida todavia no existe, y encenderlo cambia un metodo de
`METHOD_ORDER`, que arrastra D1, D2 y el bloque E. Se enciende con un numero
delante o no se enciende.
"""
import inspect
import unittest

from musepipe.stages import stage_x02_optimal as x02


class WingsIntactPsfsubTests(unittest.TestCase):
    def test_apagado_por_defecto(self):
        cfg = x02.stage_x02_config_from_run("ROXs12b_realigned")
        self.assertFalse(cfg["x02_wings_intact_psfsub"],
                         "encenderlo por defecto cambiaria productos sin que nadie lo pida")

    def test_se_puede_encender(self):
        cfg = x02.stage_x02_config_from_run(
            "ROXs12b_realigned", overrides={"x02_wings_intact_psfsub": True})
        self.assertTrue(cfg["x02_wings_intact_psfsub"])

    def test_lo_que_suelta_es_el_anillo_y_no_la_resta_del_modelo(self):
        src = inspect.getsource(x02.compute_stage_x02_products)
        self.assertIn("wings_intact_psfsub", src)
        self.assertIn('psfsub_common["local_bkg_annulus_px"] = None', src)
        # La resta del modelo NO es opcional: es lo que define la variante.
        self.assertIn("psfsub_cube", src)

    def test_el_qc_declara_cual_de_los_dos_se_uso(self):
        self.assertIn('"wings_intact_psfsub"', inspect.getsource(x02._qc_payload))

    def test_lo_dice_en_open_issues_cuando_esta_encendido(self):
        """Un cambio de tratamiento de fondo que no se declara es un fallback
        silencioso, que es la familia de bugs recurrente de este repo."""
        src = inspect.getsource(x02.compute_stage_x02_products)
        self.assertIn("x02_wings_intact_psfsub=false for the", src)


if __name__ == "__main__":
    unittest.main()
