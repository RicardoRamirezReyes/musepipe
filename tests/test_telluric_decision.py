import json
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np
from astropy.io import fits

from musepipe.telluric_lines import TELLURIC_BANDS as CATALOGO_BANDS
from musepipe.reduction.telluric import (
    DEFAULT_FIT_REGIONS,
    TELLURIC_BANDS,
    TelluricError,
    check_molecfit_environment,
    decide_telluric,
    estimator_bias_by_band,
    measure_telluric_depths,
    telluric_systematic_block,
    resolve_input_cube,
    stage00t_qc_skeleton,
)


def _write_data_stat_cube(path, data):
    fits.HDUList(
        [
            fits.PrimaryHDU(),
            fits.ImageHDU(data=np.asarray(data, dtype=np.float32), name="DATA"),
            fits.ImageHDU(data=np.ones_like(data, dtype=np.float32), name="STAT"),
        ]
    ).writeto(path)


class TelluricDecisionTests(unittest.TestCase):
    def test_depth_pct_detects_synthetic_o2_band(self):
        wave = np.linspace(6800, 7000, 201)
        spec = np.ones_like(wave)
        spec[(wave >= 6864) & (wave <= 6960)] *= 0.92
        depths = measure_telluric_depths(wave, spec)
        self.assertGreater(depths["O2_B"], 7.0)

    def test_o2_a_is_measured_and_can_drive_the_verdict(self):
        # La banda mas profunda del rango de MUSE, que la etapa no medía.
        wave = np.linspace(7450, 7850, 401)
        spec = np.ones_like(wave)
        spec[(wave >= 7590) & (wave <= 7700)] *= 0.90
        depths = measure_telluric_depths(wave, spec)
        self.assertIn("O2_A", depths)
        self.assertGreater(depths["O2_A"], 9.0)
        self.assertEqual(
            decide_telluric(depths, science_needs_red_continuum=True).decision, "needed"
        )

    def test_the_new_knobs_do_not_move_the_frozen_default(self):
        # Los otros dos tests de profundidad usan un continuo PLANO, donde la recta,
        # la parabola y la cubica dan lo mismo: no prueban nada del estimador. Este
        # usa uno CURVADO, que es justo el caso en que la cuerda recta de
        # `local_continuum_linear` fabrica profundidad.
        wave = np.linspace(7450, 7850, 401)
        # Convexo: la cuerda entre las dos bandas laterales queda POR ENCIMA del
        # continuo real en el centro de la banda, que es el sentido en que el
        # estimador fabrica profundidad de mas.
        curvatura = 1.0 + 3.0e-4 * (wave - 7650.0) + 2.5e-6 * (wave - 7650.0) ** 2
        spec = curvatura.copy()
        spec[(wave >= 7590) & (wave <= 7700)] *= 0.95
        solo_o2a = {"O2_A": TELLURIC_BANDS["O2_A"]}

        por_defecto = measure_telluric_depths(wave, spec, bands=solo_o2a)
        explicito = measure_telluric_depths(
            wave, spec, bands=solo_o2a,
            continuum=None, side_width_A=40.0, gap_A=10.0, clip_negative=True,
        )
        # Bit a bit: los defaults nuevos son la llamada congelada, no una equivalente.
        self.assertEqual(por_defecto, explicito)

        # Y la banda sale mas profunda que el 5 % inyectado: la diferencia es el
        # sesgo del estimador sobre un continuo curvado, no absorcion.
        self.assertGreater(por_defecto["O2_A"], 5.0)

    def test_an_already_corrected_band_keeps_its_negative_sign(self):
        # `max(0.0, depth)` convierte en 0.0 la firma de una banda sobre-corregida.
        # Con `clip_negative=False` el signo sobrevive, que es lo unico que permite
        # distinguir "sin corregir" de "corregida de mas".
        wave = np.linspace(6800, 7000, 201)
        spec = np.ones_like(wave)
        spec[(wave >= 6864) & (wave <= 6960)] *= 1.03  # emision aparente: sobre-correccion

        self.assertEqual(measure_telluric_depths(wave, spec)["O2_B"], 0.0)
        sin_clip = measure_telluric_depths(wave, spec, clip_negative=False)["O2_B"]
        self.assertLess(sin_clip, 0.0)

    def test_a_custom_continuum_is_honoured(self):
        # El seam que hace expresable el experimento del continuo: un estimador
        # ajeno recibe (wave, spec, band) y su salida manda.
        wave = np.linspace(7450, 7850, 401)
        spec = np.full_like(wave, 2.0)
        spec[(wave >= 7590) & (wave <= 7700)] *= 0.90

        llamadas = []

        def continuo_plano(w, s, banda):
            llamadas.append(tuple(banda))
            return np.full_like(np.asarray(w, dtype=np.float64), 2.0)

        depths = measure_telluric_depths(
            wave, spec, bands={"O2_A": TELLURIC_BANDS["O2_A"]}, continuum=continuo_plano
        )
        self.assertEqual(llamadas, [TELLURIC_BANDS["O2_A"]])
        self.assertAlmostEqual(depths["O2_A"], 10.0, places=6)

    def test_o2_a_is_one_of_the_bands_the_stage_declares(self):
        self.assertEqual(TELLURIC_BANDS["O2_A"], (7590.0, 7700.0))
        # Los mismos bordes que el catálogo de `telluric_lines`, G3 y el QC de
        # ruido: una sola definición del intervalo, no cuatro.
        catalogo = next(b for b in CATALOGO_BANDS if b["species"] == "O2" and b["lo_A"] == 7590.0)
        self.assertEqual((catalogo["lo_A"], catalogo["hi_A"]), TELLURIC_BANDS["O2_A"])
        self.assertEqual(len(DEFAULT_FIT_REGIONS), len(TELLURIC_BANDS))

    def test_decision_skips_when_science_does_not_need_red_continuum(self):
        decision = decide_telluric({"O2_B": 10.0}, science_needs_red_continuum=False)
        self.assertFalse(decision.telluric_applied)
        self.assertEqual(decision.decision, "not_needed_science")

    def test_decision_needs_checkpoint_for_deep_band(self):
        decision = decide_telluric({"O2_B": 4.0}, science_needs_red_continuum=True)
        self.assertTrue(decision.telluric_applied)
        self.assertTrue(decision.checkpoint_required)

    def test_the_qc_declares_the_aperture_it_measured_with(self):
        with tempfile.TemporaryDirectory() as tmp:
            cube = Path(tmp) / "adp.fits"
            _write_data_stat_cube(cube, np.ones((4, 5, 5)))
            info = resolve_input_cube(cube, upstream="ADP", checksum=False)
            qc = stage00t_qc_skeleton(info, run_id="R", primary_yx=(100, 100),
                                      aperture_radius_px=8.0)
            self.assertEqual(qc["input"]["primary_yx"], [100.0, 100.0])
            self.assertEqual(qc["input"]["aperture_radius_px"], 8.0)
            self.assertEqual(qc["fit"]["regions_A"],
                             [list(r) for r in DEFAULT_FIT_REGIONS])

    def test_a_decided_correction_is_not_an_applied_one(self):
        # `telluric_applied` es la DECISION; `applied_to_cube` es el hecho.
        with tempfile.TemporaryDirectory() as tmp:
            cube = Path(tmp) / "adp.fits"
            _write_data_stat_cube(cube, np.ones((4, 5, 5)))
            info = resolve_input_cube(cube, upstream="ADP", checksum=False)
            qc = stage00t_qc_skeleton(info, run_id="R", primary_yx=(2, 2),
                                      aperture_radius_px=1.0)
            self.assertFalse(qc["decision"]["applied_to_cube"])
            # Con veredicto `needed` la fase de decision pone telluric_applied a
            # True sin tocar el cubo: los dos campos deben poder discrepar.
            decision = decide_telluric({"O2_A": 3.86}, science_needs_red_continuum=True)
            self.assertTrue(decision.telluric_applied)
            self.assertEqual(decision.decision, "needed")
            self.assertFalse(qc["decision"]["applied_to_cube"])
            self.assertEqual(qc["products"]["cube_telcorr"], "")

    def test_the_qc_cannot_be_built_without_the_aperture(self):
        # Sin default: el esquema multi-noche perdió estos dos campos justamente
        # porque nada obligaba a declararlos.
        with tempfile.TemporaryDirectory() as tmp:
            cube = Path(tmp) / "adp.fits"
            _write_data_stat_cube(cube, np.ones((4, 5, 5)))
            info = resolve_input_cube(cube, upstream="ADP", checksum=False)
            with self.assertRaises(TypeError):
                stage00t_qc_skeleton(info, run_id="R")

    def test_resolve_adp_input_accepts_data_stat_without_qc(self):
        with tempfile.TemporaryDirectory() as tmp:
            cube = Path(tmp) / "adp.fits"
            _write_data_stat_cube(cube, np.ones((4, 5, 5)))
            info = resolve_input_cube(cube, upstream="ADP", checksum=False)
            self.assertEqual(info.upstream, "ADP")
            self.assertTrue(info.has_stat)

    def test_resolve_a2_input_requires_qc(self):
        with tempfile.TemporaryDirectory() as tmp:
            cube = Path(tmp) / "cube.fits"
            _write_data_stat_cube(cube, np.ones((4, 5, 5)))
            with self.assertRaises(TelluricError):
                resolve_input_cube(cube, upstream="A2", checksum=False)

    def _cascade_qc(self, tmp, cube, **overrides):
        """QC de A1 en perfil cascade (`stream_combine_v1`), como el de ROXs 42B b."""
        payload = {"stage": "stream_combine", "run_id": "R", "n_exposures": 30,
                   "output": str(cube), "finite_fraction": 0.9416,
                   "rejected_fraction": 0.0314,
                   "warnings": ["CRVAL3 differs by up to 0.0031 channels"]}
        payload.update(overrides)
        qc = Path(tmp) / "cube_telcorr_qc.json"
        qc.write_text(json.dumps(payload), encoding="utf-8")
        return qc

    def test_a1_in_cascade_is_vouched_for_by_the_combine_qc(self):
        # El esquema stream_combine NO tiene gates_passed porque no tiene fases:
        # exigirselas rechazaba todo objeto reducido en cascada.
        with tempfile.TemporaryDirectory() as tmp:
            cube = Path(tmp) / "cube_telcorr.fits"
            _write_data_stat_cube(cube, np.ones((4, 5, 5)))
            info = resolve_input_cube(cube, upstream="A1",
                                      qc_path=self._cascade_qc(tmp, cube), checksum=False)
            self.assertEqual(info.upstream, "A1")
            # Los avisos del combine no son fatales, pero tampoco se pierden.
            self.assertEqual(len(info.upstream_warnings), 1)
            qc = stage00t_qc_skeleton(info, run_id="R", primary_yx=(2, 2),
                                      aperture_radius_px=1.0)
            self.assertTrue(any("CRVAL3" in w for w in qc["input"]["upstream_warnings"]))
            # NO en `open_issues`, que en este repo es el canal bloqueante.
            self.assertEqual(qc["open_issues"], [])

    def test_a1_cascade_qc_must_describe_this_very_cube(self):
        # La puerta fuerte del esquema cascade: identidad, no tramite.
        with tempfile.TemporaryDirectory() as tmp:
            cube = Path(tmp) / "cube_telcorr.fits"
            otro = Path(tmp) / "otro.fits"
            _write_data_stat_cube(cube, np.ones((4, 5, 5)))
            qc = self._cascade_qc(tmp, cube, output=str(otro))
            with self.assertRaisesRegex(TelluricError, "describes another cube"):
                resolve_input_cube(cube, upstream="A1", qc_path=qc, checksum=False)

    def test_a1_cascade_rejects_a_mostly_empty_combine(self):
        with tempfile.TemporaryDirectory() as tmp:
            cube = Path(tmp) / "cube_telcorr.fits"
            _write_data_stat_cube(cube, np.ones((4, 5, 5)))
            qc = self._cascade_qc(tmp, cube, finite_fraction=0.1)
            with self.assertRaisesRegex(TelluricError, "mostly empty"):
                resolve_input_cube(cube, upstream="A1", qc_path=qc, checksum=False)

    def test_a1_with_phases_still_needs_all_four(self):
        # El esquema monolitico (ROXs 12 b) no se ha aflojado.
        with tempfile.TemporaryDirectory() as tmp:
            cube = Path(tmp) / "cube.fits"
            _write_data_stat_cube(cube, np.ones((4, 5, 5)))
            qc = Path(tmp) / "stage00r_qc.json"
            qc.write_text(json.dumps({"stage": "00r_raw_reduction",
                                      "gates_passed": ["fase0", "fase1"]}), encoding="utf-8")
            with self.assertRaisesRegex(TelluricError, "A1 gates are incomplete"):
                resolve_input_cube(cube, upstream="A1", qc_path=qc, checksum=False)
            qc.write_text(json.dumps({"stage": "00r_raw_reduction",
                                      "gates_passed": ["fase0", "fase1", "fase2", "fase3"]}),
                          encoding="utf-8")
            info = resolve_input_cube(cube, upstream="A1", qc_path=qc, checksum=False)
            self.assertEqual(info.upstream_warnings, ())

    def test_check_molecfit_environment_requires_all_recipes(self):
        with tempfile.TemporaryDirectory() as tmp:
            exe = Path(tmp) / "esorex"
            exe.write_text("#!/bin/sh\n", encoding="utf-8")
            exe.chmod(0o755)

            def runner(args):
                return subprocess.CompletedProcess(args, 0, stdout="molecfit_model : recipe", stderr="")

            with self.assertRaises(TelluricError):
                check_molecfit_environment(esorex=str(exe), runner=runner)

    def test_check_molecfit_environment_accepts_required_recipes(self):
        with tempfile.TemporaryDirectory() as tmp:
            exe = Path(tmp) / "esorex"
            exe.write_text("#!/bin/sh\n", encoding="utf-8")
            exe.chmod(0o755)

            def runner(args):
                return subprocess.CompletedProcess(
                    args,
                    0,
                    stdout=(
                        "molecfit_model : recipe\n"
                        "molecfit_calctrans : recipe\n"
                        "molecfit_correct : recipe\n"
                    ),
                    stderr="",
                )

            env = check_molecfit_environment(esorex=str(exe), runner=runner)
            self.assertIn("molecfit_correct", env["molecfit_recipes"])


class PuertaDeDosColasTests(unittest.TestCase):
    """El umbral mira |profundidad|: una banda sobre-corregida tambien falla.

    Antes se comparaba `max(profundidades)`, asi que un residuo NEGATIVO —una
    banda corregida de mas— no podia disparar la puerta. Medido en ROXs 12 b: el
    `not_needed_shallow` de la noche del 29 escondia un -9.43 % en O2 A.
    """

    def _decide(self, depths):
        return decide_telluric(depths, science_needs_red_continuum=True, threshold_pct=3.0)

    def test_una_banda_muy_negativa_pide_correccion(self):
        d = self._decide({"O2_B": 0.483, "O2_A": -9.430, "H2O_7200": -1.356, "H2O_8200": -0.755})
        self.assertEqual(d.decision, "needed")
        self.assertTrue(d.checkpoint_required)

    def test_el_caso_positivo_no_cambia(self):
        # Lo que hoy dispara la puerta la sigue disparando igual.
        self.assertEqual(self._decide({"O2_A": 5.088, "O2_B": -0.019}).decision, "needed")

    def test_negativos_pequenos_siguen_sin_pedir_correccion(self):
        # El combinado canonico de ROXs 12 b: max|.| = 2.49 < 3.
        d = self._decide({"O2_B": -1.004, "O2_A": -0.773, "H2O_7200": -1.557, "H2O_8200": -2.488})
        self.assertEqual(d.decision, "not_needed_shallow")
        self.assertFalse(d.telluric_applied)

    def test_el_signo_no_lo_decide_el_orden_de_las_bandas(self):
        base = {"a": -4.0, "b": 1.0}
        self.assertEqual(self._decide(base).decision,
                         self._decide(dict(reversed(list(base.items())))).decision)


def _espectro_sintetico(n=3600, lo=4750.0, paso=1.25, curvatura=0.0):
    wave = lo + np.arange(n) * paso
    x = (wave - wave.mean()) / (0.5 * (wave[-1] - wave[0]))
    return wave, 100.0 * (1.0 + curvatura * x**2)


class SesgoDelEstimadorTests(unittest.TestCase):
    """`estimator_bias_by_band`: lo que el estimador marca donde no hay nada."""

    def test_sobre_un_continuo_plano_el_sesgo_es_cero(self):
        wave, spec = _espectro_sintetico()
        out = estimator_bias_by_band(wave, spec)
        medidos = [v for v in out.values() if v is not None]
        self.assertTrue(medidos, "ninguna banda pudo medirse sobre el continuo plano")
        for v in medidos:
            self.assertAlmostEqual(v["bias_pct"], 0.0, places=6)

    def test_con_curvatura_inyectada_el_metodo_responde(self):
        # Control positivo: si su cero no fuera un cero, no serviria de nada.
        wave, plano = _espectro_sintetico()
        _, curvo = _espectro_sintetico(curvatura=0.12)
        b0 = estimator_bias_by_band(wave, plano)
        b1 = estimator_bias_by_band(wave, curvo)
        banda = next(k for k, v in b0.items() if v is not None and b1.get(k) is not None)
        self.assertGreater(abs(b1[banda]["bias_pct"]), abs(b0[banda]["bias_pct"]))

    def test_una_banda_sin_ventanas_limpias_devuelve_None(self):
        # Un tramo corto no deja sitio para ventanas de control de la misma geometria.
        wave, spec = _espectro_sintetico(n=60, lo=7580.0)
        self.assertIsNone(estimator_bias_by_band(wave, spec)["O2_A"])


class SistematicoTelluricoTests(unittest.TestCase):
    """El bloque que D2 lee para dejar de declarar telurico = 0."""

    def test_la_fraccion_es_la_profundidad_descontada_y_en_tanto_por_uno(self):
        wave, spec = _espectro_sintetico()
        fracs, detalle = telluric_systematic_block(wave, spec, {"O2_A": -2.5})
        self.assertIn("O2_A", fracs)
        d = detalle["O2_A"]
        self.assertAlmostEqual(fracs["O2_A"], abs(d["discounted_pct"]) / 100.0, places=12)
        self.assertGreaterEqual(fracs["O2_A"], 0.0)

    def test_el_signo_se_pierde_a_proposito(self):
        # A D2 le hace falta la MAGNITUD: sobre-corregir contamina igual.
        wave, spec = _espectro_sintetico()
        neg, _ = telluric_systematic_block(wave, spec, {"O2_A": -2.5})
        pos, _ = telluric_systematic_block(wave, spec, {"O2_A": +2.5})
        self.assertAlmostEqual(neg["O2_A"], pos["O2_A"], places=12)

    def test_sin_ventanas_limpias_se_declara_sin_descontar(self):
        wave, spec = _espectro_sintetico(n=60, lo=7580.0)
        fracs, detalle = telluric_systematic_block(wave, spec, {"O2_A": -2.5})
        self.assertEqual(detalle["O2_A"]["source"], "no_clean_control_windows")
        self.assertIsNone(detalle["O2_A"]["bias_pct"])
        self.assertAlmostEqual(fracs["O2_A"], 0.025, places=12)


if __name__ == "__main__":
    unittest.main()
