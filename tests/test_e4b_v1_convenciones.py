"""V1 de E4b: las dos etapas comparten CONVENCIONES, no números.

La V1 decía «con una sola exposición y sin agrupar, la fila tiene que reproducir
lo que E4 daría sobre ese mismo cubo». Medido el 2026-09-19 eso no puede dar cero:
E4 resta la superficie local de 04b y trabaja en la escala CRUDA del extractor,
mientras E4b usa fondo de anillo y aplica `apcorr` — un 39 % de diferencia mediana
en `aperture` sobre el MISMO cubo, que es la receta y no el sustrato. La spec se
reescribió (§5, V1) y esto fija la versión nueva.

Lo que se vigila aquí es lo que sí tiene que coincidir, que es la familia de bugs
recurrente del proyecto: dos etapas que se citan juntas y dejan de compartir una
definición.
"""
import unittest
from pathlib import Path

from musepipe import spectral
from musepipe.stages import stage_h04_injection as h04
from musepipe.stages import stage_h04b_perexp_injection as h04b

ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "docs" / "spec_E4b_codex_perexp_injection.md"


class ConvencionesCompartidasTests(unittest.TestCase):
    def test_la_rejilla_el_estimador_y_las_posiciones_son_LOS_MISMOS_objetos(self):
        # Importados, no reimplementados: si alguien copia uno de estos a E4b
        # para "desacoplarla", las dos etapas empiezan a medir cosas distintas
        # con el mismo nombre y nadie se entera.
        for nombre in ("build_h04_cases", "measure_recovery_with_h01_estimator",
                       "resolve_h04_positions", "template_width_label"):
            self.assertIs(getattr(h04b, nombre), getattr(h04, nombre),
                          f"E4b dejó de compartir `{nombre}` con E4")

    def test_las_dos_resuelven_la_LSF_por_el_resolutor_unico(self):
        self.assertIs(h04b.resolve_lsf_fwhm_A, spectral.resolve_lsf_fwhm_A)
        qc = {"m2_lsf": {"lsf_fwhm_at_halpha_A": 2.31, "status": "yellow"}}
        cfg = {"h01_lsf_fwhm_A": 2.2849}
        por_e4 = h04._lsf_fwhm_from_config_or_qc(cfg)
        por_e4b = spectral.resolve_lsf_fwhm_A(qc, cfg, stage_key="x06b_lsf_fwhm_A")
        self.assertEqual(por_e4, por_e4b)

    def test_el_knob_de_cada_etapa_manda_sobre_el_del_run(self):
        # `h04_lsf_fwhm_A` y `x06b_lsf_fwhm_A` existen para poder desviarse a
        # proposito; lo que no puede pasar es que una etapa ignore el suyo.
        cfg = {"h04_lsf_fwhm_A": 2.1, "x06b_lsf_fwhm_A": 2.4, "h01_lsf_fwhm_A": 2.9}
        self.assertEqual(h04._lsf_fwhm_from_config_or_qc(cfg)[0], 2.1)
        self.assertEqual(spectral.resolve_lsf_fwhm_A({}, cfg, stage_key="x06b_lsf_fwhm_A")[0], 2.4)


class LasRecetasDivergenYEstaDeclaradoTests(unittest.TestCase):
    """Si alguien unifica las recetas, la spec deja de ser cierta: que falle aquí."""

    def test_e4b_extrae_con_fondo_de_anillo(self):
        import inspect
        fuente = inspect.getsource(h04b._extrae)
        self.assertIn("annulus_background_spectrum", fuente)
        self.assertIn("apcorr", fuente)

    def test_e4_extrae_restando_la_superficie_local(self):
        import inspect
        from musepipe.stages import stage_h04_extractors as ext
        fuente = inspect.getsource(ext.build_production_extractors)
        self.assertIn("subtract_local_surface_cube", fuente)

    def test_la_spec_declara_la_version_nueva_de_la_V1(self):
        texto = SPEC.read_text(encoding="utf-8")
        self.assertIn("V1 — Convenciones compartidas con E4", texto)
        self.assertIn("39 %", texto)
        self.assertNotIn("V1 — Ancla contra E4", texto)


if __name__ == "__main__":
    unittest.main()
