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
