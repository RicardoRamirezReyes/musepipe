"""C7: extraer en cada exposición y combinar las MEDIDAS.

Lo que se fija aquí son las decisiones que hacen honesta a la etapa, no el
camino feliz:

* **`sum` == `equal` salvo el factor N.** Se declaró `sum` porque se pidió para
  pruebas posteriores, y con la consecuencia medida escrita en la spec: su S/N
  es la de la media sin pesos. Si algún día `sum` empezara a "ganar", sería que
  alguien cambió el estimador de σ, que es exactamente la trampa nº 1 de
  `docs/2026-08-26_perexp_medido_y_la_noche_mala.md`.
* **`invvar` pesa por 1/σ² y baja el peso de lo ruidoso.** Es lo único que la
  vía por exposición sabe hacer y el cubo combinado no.
* **La sigma_i encogida deja la apcorr en paz.** Con 7 controles la sigma de
  cada exposicion tiene 6 grados de libertad, y 22 exposiciones de igual sigma
  verdadera dan n_eff ~15 solo por el ruido del estimador (via B, punto 2).
  `night`/`auto` encogen la parte ruidosa (los controles crudos) hacia la
  noche; la apcorr de cada exposicion —una medida del modelo, no del ruido—
  se conserva exacta. `auto` mide cuanto encoger: 1 si la dispersion es la de
  chi^2_k, ~0 si las sigma verdaderas difieren de verdad.
* **Las guardias fallan.** Un documento que no es mezcla, un `exposure_id`
  repetido —que NO es único por construcción—, una exposición sin modelo, y una
  ley o agrupación desconocidas.
* **Las rutas que la etapa lee son las que declara** (el `KeyError` con el que
  murió C1b en su primera ejecución real).
"""

import ast
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from musepipe.extraction.product import SpectrumProduct
from musepipe.observations import resolve_observation_plan
from musepipe.reduction.stream_combine import build_stream_combine_plan
from musepipe.stages import stage_x06_perexp as MOD
from musepipe.stages.stage_x06_perexp import (
    PerExpError,
    combine_measurements,
    group_exposures,
    shrink_sigma_by_night,
)
from test_observations import _exposures, _write_run
from test_psf_mixture import component, moffat_doc

CROP = 20
PAD = 4


class CombinationLawTests(unittest.TestCase):
    """Las leyes, sobre números donde la respuesta se sabe de antemano."""

    def setUp(self):
        self.valores = np.array([10.0, 12.0, 8.0, 30.0])
        self.pesos_plan = np.array([300.0, 300.0, 300.0, 720.0])
        # La cuarta es la mala: mucho mas ruidosa, y ademas la mas larga. Es el
        # caso real de ROXs 12 b, donde `exptime` le daba el 38.1 % del peso.
        self.sigma = np.array([1.0, 1.0, 1.0, 10.0])

    def test_sum_is_equal_times_n(self):
        w_eq, esc_eq = combine_measurements(self.valores, self.pesos_plan, self.sigma, "equal")
        w_sum, esc_sum = combine_measurements(self.valores, self.pesos_plan, self.sigma, "sum")
        np.testing.assert_allclose(w_eq, w_sum)
        self.assertEqual(esc_eq, 1.0)
        self.assertEqual(esc_sum, float(self.valores.size))
        media = esc_eq * float(np.sum(w_eq * self.valores))
        suma = esc_sum * float(np.sum(w_sum * self.valores))
        self.assertAlmostEqual(suma, media * self.valores.size)

    def test_invvar_downweights_the_noisy_one(self):
        w, escala = combine_measurements(self.valores, self.pesos_plan, self.sigma, "invvar")
        self.assertEqual(escala, 1.0)
        self.assertAlmostEqual(float(np.sum(w)), 1.0)
        # 1/100 contra 1/1: la ruidosa se queda por debajo del 1 %.
        self.assertLess(w[-1], 0.01)
        self.assertGreater(w[0], 0.3)

    def test_exptime_rewards_the_longest_even_if_it_is_the_worst(self):
        w, _ = combine_measurements(self.valores, self.pesos_plan, self.sigma, "exptime")
        self.assertGreater(w[-1], w[0])   # y por eso no es el valor por defecto

    def test_an_unusable_weight_set_is_refused(self):
        with self.assertRaises(PerExpError):
            combine_measurements(self.valores, np.zeros(4), self.sigma, "exptime")


class SigmaShrinkTests(unittest.TestCase):
    """La sigma_i encogida hacia su noche: que encoge, que no, y cuanto."""

    def setUp(self):
        rng = np.random.default_rng(7)
        self.n_controls = 7
        self.nights = ["a"] * 22 + ["b"] * 7
        # Sigma verdadera IGUAL en cada noche; lo que se ve es chi^2_6 / 6.
        verdadera = np.where(np.asarray(self.nights) == "a", 1.0, 4.0)
        self.sigma_raw = verdadera * np.sqrt(rng.chisquare(6, 29) / 6.0)
        # La apcorr efectiva de cada exposicion, deliberadamente dispar.
        self.apcorr = rng.uniform(3.0, 12.0, 29)
        self.sigma_peso = self.sigma_raw * self.apcorr

    def test_none_is_the_identity_and_still_publishes_the_diagnostic(self):
        s, diag = shrink_sigma_by_night(self.sigma_peso, self.sigma_raw, self.nights,
                                        self.n_controls, "none")
        np.testing.assert_allclose(s, self.sigma_peso)
        self.assertEqual(diag["dof"], 6)
        self.assertEqual(diag["by_night"]["a"]["lambda"], 0.0)
        self.assertAlmostEqual(diag["by_night"]["a"]["sd_log_var_expected"], 0.6284, places=3)
        self.assertIsNotNone(diag["by_night"]["a"]["sd_log_var_observed"])

    def test_night_equalises_the_raw_sigma_and_keeps_each_apcorr(self):
        s, diag = shrink_sigma_by_night(self.sigma_peso, self.sigma_raw, self.nights,
                                        self.n_controls, "night")
        self.assertEqual(diag["by_night"]["a"]["lambda"], 1.0)
        raw_encogida = s / self.apcorr
        for noche in ("a", "b"):
            sel = np.asarray(self.nights) == noche
            np.testing.assert_allclose(raw_encogida[sel], raw_encogida[sel][0], rtol=1e-9)
        # Entre noches la razon es la de las sigma verdaderas (4x en varianza
        # 16x), a lo que 7 y 22 muestras permiten.
        razon = raw_encogida[22] / raw_encogida[0]
        self.assertTrue(2.5 < razon < 6.5, razon)
        # Y las apcorr no se han tocado: el cociente sigma/raw es el de entrada.
        np.testing.assert_allclose(s / raw_encogida, self.apcorr, rtol=1e-9)

    def test_auto_shrinks_fully_when_the_spread_is_only_the_estimator(self):
        _, diag = shrink_sigma_by_night(self.sigma_peso, self.sigma_raw, self.nights,
                                        self.n_controls, "auto")
        # 22 muestras de chi^2_6: la dispersion observada es la esperada a
        # menos de un 40 %, asi que lambda queda cerca de 1 (o recortado a 1).
        self.assertGreater(diag["by_night"]["a"]["lambda"], 0.6)
        self.assertTrue(0.7 < diag["by_night"]["a"]["excess_ratio"] < 1.4)

    def test_auto_barely_shrinks_when_the_true_sigmas_really_differ(self):
        sigma_raw = self.sigma_raw * np.exp(np.linspace(-2.5, 2.5, 29))
        s, diag = shrink_sigma_by_night(sigma_raw * self.apcorr, sigma_raw, self.nights,
                                        self.n_controls, "auto")
        self.assertLess(diag["by_night"]["a"]["lambda"], 0.35)
        self.assertGreater(diag["by_night"]["a"]["excess_ratio"], 1.7)
        # Entre `none` y `auto` con lambda pequeno la sigma se mueve poco.
        self.assertLess(np.max(np.abs(np.log(s / (sigma_raw * self.apcorr)))), 1.0)

    def test_a_night_with_one_exposure_has_nothing_to_shrink_towards(self):
        s, diag = shrink_sigma_by_night([2.0, 3.0], [1.0, 1.5], ["a", "b"], 7, "auto")
        np.testing.assert_allclose(s, [2.0, 3.0])
        self.assertEqual(diag["by_night"]["a"]["lambda"], 0.0)
        self.assertIsNone(diag["by_night"]["a"]["sd_log_var_observed"])

    def test_an_unknown_mode_is_refused(self):
        with self.assertRaises(PerExpError):
            shrink_sigma_by_night([1.0], [1.0], ["a"], 7, "median")


class GroupingTests(unittest.TestCase):
    class _Exp:
        def __init__(self, exposure_id):
            self.exposure_id = exposure_id

    def test_none_keeps_everything_together(self):
        exps = [self._Exp("2022-08-29_MUSE.a"), self._Exp("2022-08-31_MUSE.b")]
        self.assertEqual(group_exposures(exps, "none"), {"all": [0, 1]})

    def test_night_splits_by_date(self):
        exps = [self._Exp("2022-08-29_MUSE.a"), self._Exp("2022-08-31_MUSE.b"),
                self._Exp("2022-08-29_MUSE.c")]
        self.assertEqual(group_exposures(exps, "night"), {"20220829": [0, 2], "20220831": [1]})

    def test_every_exposure_lands_in_exactly_one_group(self):
        exps = [self._Exp(f"2022-08-2{i}_MUSE.e{i}") for i in range(5)]
        grupos = group_exposures(exps, "night")
        plano = sorted(i for idx in grupos.values() for i in idx)
        self.assertEqual(plano, list(range(5)))


def _run_sintetico(root, run_id="obj", n=3):
    """Un run con `n` exposiciones, su plan cacheado y una mezcla con modelo por exposición."""

    cubes = _exposures(root, [(10.2, 10.4), (9.8, 10.1), (10.1, 9.9)][:n])
    run = _write_run(root, run_id, {"perexp_dir": str(root / "cubes"),
                                    "x01_apertures": [{"kind": "box", "size": 3}],
                                    "x06_n_controls": 6,
                                    # El cubo sintetico va de 6200 a 6229 A: la banda
                                    # de pesos tiene que caber dentro.
                                    "x06_weight_band_A": [6200.0, 6215.0]})
    plan = build_stream_combine_plan([str(p) for p in cubes], run_id=run_id,
                                     output=str(root / "combined.fits"),
                                     crop_npix=CROP, pad=PAD, chunk_channels=8)
    (run / "stages" / "stream_combine_plan.json").write_text(
        json.dumps(plan.as_dict()), encoding="utf-8")
    obs = resolve_observation_plan(run_id, project_root=root)
    (run / "stages" / "observation_plan.json").write_text(
        json.dumps(obs.to_json()), encoding="utf-8")
    mezcla = {"form": "mixture", "norm_radius_px": 6.0,
              "components": [component(e.exposure_id, moffat_doc(3.0, norm_radius=6.0))
                             for e in obs.exposures]}
    (run / "stages" / "psf_model_mixture.json").write_text(json.dumps(mezcla), encoding="utf-8")
    centro = CROP / 2.0
    (run / "stages" / "stage01c_qc.json").write_text(json.dumps({
        "primary": {"pos_yx": [centro, centro]},
        "companion": {"pos_yx": [centro + 5.0, centro]},
        "frame_shape": [CROP, CROP],
    }), encoding="utf-8")
    return run, obs, mezcla


class GuardTests(unittest.TestCase):
    """Cada negativa de la etapa, comprobada fallando."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.run, self.obs, self.mezcla = _run_sintetico(self.root)

    def tearDown(self):
        self._tmp.cleanup()

    def _cfg(self, **extra):
        cfg = MOD.stage_x06_config_from_run("obj", project_root=self.root)
        cfg.update(extra)
        return cfg

    def test_a_document_that_is_not_a_mixture_is_refused(self):
        (self.run / "stages" / "psf_model_mixture.json").write_text(
            json.dumps(moffat_doc(3.0)), encoding="utf-8")
        with self.assertRaises(PerExpError) as ctx:
            MOD.compute_stage_x06_products(self._cfg())
        self.assertIn("mixture", str(ctx.exception).lower())

    def test_an_exposure_without_a_model_is_refused_with_the_list(self):
        recortada = dict(self.mezcla)
        recortada["components"] = self.mezcla["components"][:-1]
        (self.run / "stages" / "psf_model_mixture.json").write_text(
            json.dumps(recortada), encoding="utf-8")
        with self.assertRaises(PerExpError) as ctx:
            MOD.compute_stage_x06_products(self._cfg())
        self.assertIn("modelo", str(ctx.exception))

    def test_an_unknown_combination_law_is_refused(self):
        with self.assertRaises(PerExpError):
            MOD.stage_x06_config_from_run("obj", project_root=self.root,
                                          overrides={"x06_combine": "mediana"})

    def test_a_weight_band_outside_the_cube_is_refused_by_name(self):
        with self.assertRaises(PerExpError) as ctx:
            MOD.compute_stage_x06_products(self._cfg(x06_weight_band_A=[8600.0, 9000.0]))
        self.assertIn("x06_weight_band_A", str(ctx.exception))

    def test_an_unknown_sigma_shrink_is_refused(self):
        with self.assertRaisesRegex(PerExpError, "x06_sigma_shrink"):
            MOD.stage_x06_config_from_run("obj", project_root=self.root,
                                          overrides={"x06_sigma_shrink": "median"})

    def test_an_unknown_grouping_is_refused(self):
        with self.assertRaises(PerExpError):
            MOD.stage_x06_config_from_run("obj", project_root=self.root,
                                          overrides={"x06_group_by": "ob"})


class EndToEndTests(unittest.TestCase):
    """La etapa entera contra un run sintético: escribe, se relee y dice de dónde sale."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.run, self.obs, _ = _run_sintetico(self.root)

    def tearDown(self):
        self._tmp.cleanup()

    def test_it_writes_a_readable_product_and_says_what_it_did(self):
        out = MOD.run_stage_x06("obj", project_root=self.root)
        qc = out["qc"]
        self.assertEqual(qc["stage"], "x06_perexp")
        self.assertEqual(qc["input"]["n_exposures"], len(self.obs.exposures))
        self.assertEqual(qc["convention"]["combine"], "invvar")
        self.assertEqual(qc["convention"]["group_by"], "none")
        # La sigma NO puede venir del STAT: la etapa declara de donde sale.
        self.assertIn("controles", qc["convention"]["sigma"])

        destino = MOD.product_path(MOD.stage_x06_paths("obj", project_root=self.root),
                                   "all", "box3")
        spec = SpectrumProduct.read(destino)
        spec.validate()
        self.assertEqual(spec.header["COMBMODE"], "perexp_invvar")
        self.assertEqual(int(spec.header["NEXP"]), len(self.obs.exposures))
        self.assertEqual(spec.header["GROUP"], "all")
        self.assertTrue(np.all(np.asarray(spec.apcorr) > 0))

    def test_the_qc_says_how_sigma_was_estimated_and_what_each_mode_would_give(self):
        out = MOD.run_stage_x06("obj", project_root=self.root)
        qc = out["qc"]
        self.assertEqual(qc["convention"]["sigma_shrink"], "none")
        box = qc["groups"]["all"]["apertures"]["box3"]
        shrink = box["sigma_shrink"]
        self.assertEqual(shrink["mode"], "none")
        self.assertEqual(sorted(shrink["n_eff_by_mode"]), ["auto", "night", "none"])
        # Con `none` el n_eff del producto ES el de ese modo.
        self.assertAlmostEqual(shrink["n_eff_by_mode"]["none"], box["n_eff"])
        # `night` iguala la sigma cruda dentro de la (unica) noche: n_eff sube o
        # queda igual, nunca baja.
        self.assertGreaterEqual(shrink["n_eff_by_mode"]["night"], box["n_eff"] - 1e-9)
        self.assertEqual(set(box["sigma_weight_by_exposure"]), set(box["weights_normalised"]))
        self.assertEqual(set(box["sigma_raw_weight_by_exposure"]), set(box["weights_normalised"]))
        # El reparto por noche suma 1 y ningun n_eff dentro de una noche supera
        # el numero de exposiciones que tiene.
        filas = shrink["by_night_by_mode"]["none"]
        self.assertAlmostEqual(sum(f["share"] for f in filas.values()), 1.0)
        por_noche = group_exposures(self.obs.exposures, "night")
        self.assertEqual(sorted(filas), sorted(por_noche))
        for noche, fila in filas.items():
            self.assertLessEqual(fila["n_eff_within"], len(por_noche[noche]) + 1e-9)
        destino = MOD.product_path(MOD.stage_x06_paths("obj", project_root=self.root),
                                   "all", "box3")
        self.assertEqual(SpectrumProduct.read(destino).header["WSHRINK"], "none")

    def test_night_mode_weights_are_the_night_mode_column_of_the_diagnostic(self):
        out = MOD.run_stage_x06("obj", project_root=self.root,
                                overrides={"x06_sigma_shrink": "night"})
        box = out["qc"]["groups"]["all"]["apertures"]["box3"]
        self.assertEqual(out["qc"]["convention"]["sigma_shrink"], "night")
        self.assertAlmostEqual(box["sigma_shrink"]["n_eff_by_mode"]["night"], box["n_eff"])
        # La tabla de pesos que lee el combinado declara como se estimo la sigma.
        from musepipe.reduction.stream_combine import load_weight_table
        _, fuente = load_weight_table(out["written"]["qc_json"])
        self.assertEqual(fuente["sigma_shrink"], "night")

    def test_grouping_by_night_covers_every_exposure_once(self):
        out = MOD.run_stage_x06("obj", project_root=self.root,
                                overrides={"x06_group_by": "night"})
        grupos = out["qc"]["groups"]
        vistas = [e for g in grupos.values() for e in g["exposures"]]
        self.assertEqual(sorted(vistas), sorted(e.exposure_id for e in self.obs.exposures))
        self.assertEqual(len(vistas), len(set(vistas)))

    def test_sum_and_equal_differ_only_by_the_scale(self):
        paths = MOD.stage_x06_paths("obj", project_root=self.root)
        MOD.run_stage_x06("obj", project_root=self.root, overrides={"x06_combine": "equal"})
        media = SpectrumProduct.read(MOD.product_path(paths, "all", "box3"))
        MOD.run_stage_x06("obj", project_root=self.root, overrides={"x06_combine": "sum"})
        suma = SpectrumProduct.read(MOD.product_path(paths, "all", "box3"))
        n = len(self.obs.exposures)
        fin = np.isfinite(media.flux) & np.isfinite(suma.flux)
        np.testing.assert_allclose(suma.flux[fin], media.flux[fin] * n, rtol=1e-9)
        # Y la sigma escala igual, que es POR QUE la S/N no cambia.
        np.testing.assert_allclose(suma.flux_err[fin], media.flux_err[fin] * n, rtol=1e-9)


class PathsContractTests(unittest.TestCase):
    """Las claves que la etapa LEE tienen que ser las que su `paths` DECLARA.

    C1b se quedó sin `observation_plan_json` en su `paths` mientras la etapa la
    consultaba, y murió con `KeyError` en la primera ejecución real: ningún test
    lo vio porque todos entraban por las funciones internas.
    """

    MODULO = Path(__file__).resolve().parents[1] / "musepipe" / "stages" / "stage_x06_perexp.py"

    def test_every_key_the_stage_reads_is_declared(self):
        arbol = ast.parse(self.MODULO.read_text(encoding="utf-8"))
        leidas = {
            nodo.slice.value
            for nodo in ast.walk(arbol)
            if isinstance(nodo, ast.Subscript)
            and isinstance(nodo.value, ast.Name) and nodo.value.id == "paths"
            and isinstance(nodo.slice, ast.Constant) and isinstance(nodo.slice.value, str)
        }
        with tempfile.TemporaryDirectory() as tmp:
            declaradas = set(MOD.stage_x06_paths("RUN_X", project_root=tmp))
        self.assertTrue(leidas, "no se ha encontrado ningun `paths[...]`: revisa el test")
        self.assertEqual(leidas - declaradas, set(),
                         "claves leidas y no declaradas en stage_x06_paths")


if __name__ == "__main__":
    unittest.main()
