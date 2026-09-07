"""Material de revisión de A1 (`musepipe.reduction.a1_review`).

Los tests montan runs sintéticos completos —config con `chain`, work-dir con
manifiestos de noche, plan de combinado y cubos FITS— para poder comprobar el
comportamiento que motivó el módulo: que los números salgan del objeto que se
audita y no del primero que se redujo.
"""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from astropy.io import fits

from musepipe.reduction import a1_review as ar

ROOT = Path(__file__).resolve().parents[1]


# --------------------------------------------------------------------------
# Andamiaje
# --------------------------------------------------------------------------
def _write_run(root, run_id, config, chain=None, meta=None):
    run = root / "runs" / run_id
    (run / "config").mkdir(parents=True, exist_ok=True)
    (run / "stages").mkdir(parents=True, exist_ok=True)
    payload = {"meta": meta or {}, "config": dict(config)}
    if chain is not None:
        payload["chain"] = chain
    (run / "config" / "config.json").write_text(json.dumps(payload), encoding="utf-8")
    return run


def _write_night(work, night, recipes, checkpoints, n_science):
    directory = work / f"night_{night}"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "products_manifest.json").write_text(json.dumps({
        "schema_version": 2,
        "night": night,
        "status": "complete",
        "recipes": recipes,
        "checkpoints": checkpoints,
        "association": {"science": [f"exp{i}" for i in range(n_science)],
                        "calibrations": {"STD": ["std.fits"]}},
    }), encoding="utf-8")


def _full_checkpoints():
    return {step: {"status": "complete", "sof": f"{step}.sof", "gates": {}}
            for step in ar.NIGHT_STEP_ORDER}


def _write_cube(path, *, nz=60, ny=40, nx=40, crval3=4749.53125, cd3_3=1.25,
                exptime=300.0, peak_yx=(20, 22)):
    """Cubo mínimo con una fuente puntual y un halo, para los paneles."""

    yy, xx = np.mgrid[:ny, :nx]
    radius = np.hypot(yy - peak_yx[0], xx - peak_yx[1])
    image = 1000.0 * np.exp(-0.5 * (radius / 1.5) ** 2) + 20.0 / (1.0 + radius)
    cube = np.repeat(image[None, :, :], nz, axis=0).astype(np.float32)
    # una línea brillante en un canal conocido, para comprobar que Hα se sitúa
    cube[nz // 2] *= 3.0
    header = fits.Header()
    header["CRVAL3"] = crval3
    header["CRPIX3"] = 1.0
    header["CD3_3"] = cd3_3
    header["BUNIT"] = "10**(-20)*erg/s/cm**2/Angstrom"
    primary = fits.PrimaryHDU()
    primary.header["EXPTIME"] = exptime
    path.parent.mkdir(parents=True, exist_ok=True)
    fits.HDUList([primary, fits.ImageHDU(data=cube, header=header, name="DATA")]).writeto(path)
    return cube


# --------------------------------------------------------------------------
# El grafo de fases no puede desincronizarse de la cascada real
# --------------------------------------------------------------------------
class PhaseGraphContractTests(unittest.TestCase):
    """`a1_review` declara el orden y las puertas; `reduce_cascade` los ejecuta."""

    @staticmethod
    def _reduce_cascade():
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "_reduce_cascade_for_test", ROOT / "scripts" / "reduce_cascade.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_step_order_matches_reduce_cascade(self):
        self.assertEqual(list(ar.NIGHT_STEP_ORDER), list(self._reduce_cascade().STEP_ORDER))

    @staticmethod
    def _count_value(text):
        """`"7×24"` -> 168, `"24"` -> 24: el texto de la caja debe cuadrar."""

        total = 1
        for part in str(text).split("×"):
            total *= int(part)
        return total

    def test_declared_outputs_match_the_real_gates(self):
        cascade = self._reduce_cascade()
        n_object, n_std = 7, 3
        for step in ar.NIGHT_STEP_ORDER:
            declared = {io.tag: io.count for io in ar._outputs_for(step, n_object, n_std)}
            real = cascade._gates(n_object, n_std, step)
            self.assertEqual(set(declared), set(real), f"tags de {step}")
            for tag, count in real.items():
                self.assertEqual(self._count_value(declared[tag]), count,
                                 f"{step}/{tag}: la caja dice {declared[tag]!r}, la puerta {count}")

    def test_recipe_names_are_real_recipes(self):
        from musepipe.reduction.esorex_driver import DEFAULT_RECIPE_REQUIREMENTS

        for step, recipe in ar.STEP_RECIPE.items():
            self.assertIn(step, ar.NIGHT_STEP_ORDER)
            self.assertIn(recipe, DEFAULT_RECIPE_REQUIREMENTS, f"{step} -> {recipe}")

    def test_static_tags_match_reduce_cascade(self):
        self.assertTrue(ar.STATIC_TAGS.issubset(self._reduce_cascade().STATIC_TAGS))

    def test_inputs_come_from_the_sof_table(self):
        inputs = {io.tag for io in ar._inputs_for("scibasic_object")}
        self.assertIn("OBJECT", inputs)
        self.assertIn("MASTER_FLAT", inputs)
        # `_trim_scibasic` quita STD de la pasada de objeto
        self.assertNotIn("STD", inputs)
        self.assertNotIn("OBJECT", {io.tag for io in ar._inputs_for("scibasic_std")})


# --------------------------------------------------------------------------
# Coste
# --------------------------------------------------------------------------
class RuntimeEstimateTests(unittest.TestCase):
    def test_calibration_is_paid_per_night(self):
        one = ar.estimate_esorex_runtime(10, 1)
        two = ar.estimate_esorex_runtime(10, 2)
        self.assertGreater(two, one)
        self.assertAlmostEqual(two - one, ar.T_CAL_NIGHT_MIN + ar.T_STD_NIGHT_MIN, places=6)

    def test_reusing_the_archive_lsf_is_cheaper(self):
        self.assertLess(ar.estimate_esorex_runtime(10, 2, reuse_lsf=True),
                        ar.estimate_esorex_runtime(10, 2, reuse_lsf=False))

    def test_reusing_all_calibrations_removes_the_per_night_cost(self):
        value = ar.estimate_esorex_runtime(10, 3, reuse_calibrations=True)
        per_exp = ar.T_SCIBASIC_EXP_MIN + ar.T_SCIPOST_EXP_MIN + ar.T_COMBINE_EXP_MIN
        self.assertAlmostEqual(value, 10 * per_exp, places=6)

    def test_cores_factor_scales_the_estimate(self):
        self.assertAlmostEqual(ar.estimate_esorex_runtime(5, 1, cores_factor=2.0),
                               ar.estimate_esorex_runtime(5, 1) / 2.0, places=6)

    def test_zero_nights_is_treated_as_one(self):
        self.assertEqual(ar.estimate_esorex_runtime(4, 0), ar.estimate_esorex_runtime(4, 1))


class MeasuredRuntimeTests(unittest.TestCase):
    def test_durations_are_assigned_to_their_step_and_gaps_stay_none(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            work = root / "work"
            # flat y wavecal reanudados (sin entrada en `recipes`); lsf de archivo
            checkpoints = _full_checkpoints()
            checkpoints["lsf"] = {"status": "reused_archive_product", "source": "archivo"}
            _write_night(work, "2024-01-01", [
                {"name": "muse_scibasic", "duration_s": 600.0, "status": "ok"},
                {"name": "muse_scibasic", "duration_s": 120.0, "status": "ok"},
                {"name": "muse_standard", "duration_s": 60.0, "status": "ok"},
            ], checkpoints, n_science=5)
            _write_run(root, "obj_raw", {"work_dir": str(work)})
            _write_run(root, "obj", {}, chain={"target": "obj", "stage_runs": {"A1": "obj_raw"},
                                               "reduction_profile": "cascade"})

            runtime = ar.measured_runtime("obj", project_root=root)
            steps = {s["step"]: s for s in runtime["nights"][0]["steps"]}
            self.assertIsNone(steps["flat"]["minutes"])
            self.assertIsNone(steps["wavecal"]["minutes"])
            self.assertAlmostEqual(steps["scibasic_object"]["minutes"], 10.0)
            self.assertAlmostEqual(steps["scibasic_std"]["minutes"], 2.0)
            self.assertAlmostEqual(steps["standard"]["minutes"], 1.0)
            self.assertEqual(steps["lsf"]["status"], "reused_archive_product")
            self.assertTrue(runtime["totals"]["reuse_lsf"])
            # los dos pasos reanudados se declaran, no se cuentan como 0 min
            self.assertEqual(runtime["resumed_steps"], 2)
            self.assertAlmostEqual(runtime["totals"]["measured_minutes"], 13.0)

    def test_perexp_of_another_object_is_never_read(self):
        """El fallo real: el directorio de trabajo tiene los dos objetos dentro."""

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            shared = root / "MUSE_work"
            mine, theirs = shared / "mine_by_night", shared / "theirs_by_night"
            _write_night(mine, "2024-01-01", [], _full_checkpoints(), n_science=3)
            for directory, run_id, n_exp in ((mine, "mine_raw", 3), (theirs, "theirs_raw", 99)):
                (directory / "p2").mkdir(parents=True, exist_ok=True)
                (directory / "p2" / "perexp_scipost_execution.json").write_text(json.dumps({
                    "run_id": run_id, "save": "cube,skymodel", "status": "complete",
                    "exposures": {f"e{i}": {"duration_s": 60.0} for i in range(n_exp)},
                }), encoding="utf-8")
            _write_run(root, "mine_raw", {"work_dir": str(mine)})
            _write_run(root, "mine", {}, chain={"stage_runs": {"A1": "mine_raw"}})

            runtime = ar.measured_runtime("mine", project_root=root)
            self.assertEqual(runtime["perexp"]["n_exposures"], 3)

    def test_the_pass_that_saved_cubes_wins(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            work = root / "work_by_night"
            _write_night(work, "2024-01-01", [], _full_checkpoints(), n_science=2)
            for name, save, n_exp in (("individual", "individual", 9), ("cubes", "cube,skymodel", 4)):
                (work / name).mkdir(parents=True, exist_ok=True)
                (work / name / "perexp_scipost_execution.json").write_text(json.dumps({
                    "run_id": "raw", "save": save, "status": "complete",
                    "exposures": {f"e{i}": {"duration_s": 30.0} for i in range(n_exp)},
                }), encoding="utf-8")
            _write_run(root, "raw", {"work_dir": str(work)})
            _write_run(root, "obj", {}, chain={"stage_runs": {"A1": "raw"}})

            self.assertEqual(ar.measured_runtime("obj", project_root=root)["perexp"]["n_exposures"], 4)

    def test_missing_work_dir_degrades_without_raising(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_run(root, "obj", {}, chain={"stage_runs": {"A1": "obj"}})
            runtime = ar.measured_runtime("obj", project_root=root)
            self.assertEqual(runtime["nights"], [])
            self.assertEqual(runtime["totals"]["measured_minutes"], 0.0)


# --------------------------------------------------------------------------
# Verificaciones
# --------------------------------------------------------------------------
class VerificationStatusTests(unittest.TestCase):
    def _status(self, root, verification, **extra):
        run = _write_run(root, "obj", {}, chain={"stage_runs": {"A1": "obj"}})
        payload = {"verification": verification}
        payload.update(extra)
        (run / "stages" / "stage00r_qc.json").write_text(json.dumps(payload), encoding="utf-8")
        return {v["tag"]: v for v in ar.verification_status("obj", project_root=root)}

    def test_a_measured_number_is_not_a_verdict(self):
        """`v2_std_residual_rms=0.00122` es *verdadero* en Python y no es un `ok`."""

        with tempfile.TemporaryDirectory() as tmp:
            status = self._status(Path(tmp), {
                "v1_stat_present": True, "v1_nan_fraction_outside_edges": 2.16e-08,
                "v2_std_residual_rms": 0.00122,
                "v3_wcs_ok": False,
                "v4_adp_whitelight_corr": 0.9541,
            }, status="pass", recipes=[{"name": "muse_flat"}], gates_passed=["g"])
            self.assertEqual(status["V1"]["status"], "ok")
            self.assertEqual(status["V2"]["status"], "medido")
            self.assertIn("0.00122", status["V2"]["message"])
            self.assertEqual(status["V3"]["status"], "no")
            self.assertEqual(status["V4"]["status"], "medido")

    def test_the_empty_skeleton_is_not_a_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            status = self._status(Path(tmp), {
                "v1_stat_present": False, "v2_std_residual_rms": None,
                "v3_wcs_ok": False, "v6_sky_mask_clean": False,
            })
            for tag in ("V1", "V3", "V6"):
                self.assertEqual(status[tag]["status"], "no ejecutada", tag)

    def test_the_nested_variant_is_understood(self):
        with tempfile.TemporaryDirectory() as tmp:
            status = self._status(Path(tmp), {
                "v1_stat_present": {"ok": True, "message": "nan_fraction=0.0024"},
                "v2_std_residual_rms": {"status": "unavailable", "message": "sin curva"},
            }, status="pass", gates_passed=["g"])
            self.assertEqual(status["V1"]["status"], "ok")
            self.assertEqual(status["V2"]["status"], "unavailable")
            self.assertEqual(status["V5"]["status"], "no medido")

    def test_a_missing_qc_reports_not_measured(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_run(root, "obj", {}, chain={"stage_runs": {"A1": "obj"}})
            for check in ar.verification_status("obj", project_root=root):
                self.assertEqual(check["status"], "no medido")


# --------------------------------------------------------------------------
# Cubos y paneles
# --------------------------------------------------------------------------
class CubeDiscoveryTests(unittest.TestCase):
    def test_per_exposure_cubes_carry_the_combine_framing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cube = root / "exp1" / "DATACUBE_FINAL.fits"
            _write_cube(cube)
            run = _write_run(root, "obj", {"cube_files": [str(cube)]},
                             chain={"stage_runs": {"A1": "obj"}})
            (run / "stages" / "stream_combine_plan.json").write_text(json.dumps({
                "n_exposures": 1,
                "exposures": [{"file": str(cube), "exposure_id": "2024-01-01_A", "exptime": 300.0,
                               "window": [4, 36, 6, 38], "y_center": 20.0, "x_center": 22.0,
                               "in_bounds": True, "centroid_fallback": False}],
            }), encoding="utf-8")

            rows = ar.list_a1_cubes("obj", project_root=root)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0].kind, "exposure")
            self.assertEqual(rows[0].window, (4, 36, 6, 38))
            self.assertEqual(rows[0].center_yx, (20.0, 22.0))
            self.assertEqual(rows[0].night, "2024-01-01")

    def test_falls_back_to_the_combined_cube_when_exposures_are_gone(self):
        """El caso de ROXs 12 B: los cubos por exposición se purgaron."""

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            combined = root / "final" / "DATACUBE_FINAL.fits"
            _write_cube(combined)
            run = _write_run(root, "obj", {"cube_files": [str(combined)]},
                             chain={"stage_runs": {"A1": "obj"}})
            (run / "stages" / "stream_combine_plan.json").write_text(json.dumps({
                "n_exposures": 2,
                "reference": {"center_pixel_yx": [20, 22]},
                "exposures": [{"file": str(root / "gone" / f"e{i}.fits"), "exposure_id": f"e{i}"}
                              for i in range(2)],
            }), encoding="utf-8")

            rows = ar.list_a1_cubes("obj", project_root=root)
            self.assertEqual([r.kind for r in rows], ["combined"])
            self.assertEqual(rows[0].center_yx, (20.0, 22.0))

    def test_legacy_cubes_are_opt_in(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            combined = root / "final" / "DATACUBE_FINAL.fits"
            legacy = root / "old" / "exp1" / "DATACUBE_FINAL.fits"
            _write_cube(combined)
            _write_cube(legacy)
            _write_run(root, "obj", {"cube_files": [str(combined)], "perexp_cubes": [str(legacy)]},
                       chain={"stage_runs": {"A1": "obj"}})

            self.assertEqual([r.kind for r in ar.list_a1_cubes("obj", project_root=root)],
                             ["combined"])
            with_legacy = ar.list_a1_cubes("obj", project_root=root, include_legacy=True)
            self.assertEqual([r.kind for r in with_legacy], ["legacy"])


class ProjectRootTests(unittest.TestCase):
    """El cwd de un notebook no es la raíz del repo.

    Todos los demás tests pasan `project_root=` explícito, así que ninguno veía
    que `project_root_path(None)` devuelve el **cwd**: en Jupyter, con el cwd en
    `notebooks/<objeto>/`, las rutas salían como
    `notebooks/ROXs12b/runs/<run>/config/config.json` y el módulo reventaba.
    """

    def test_root_is_derived_from_the_module_not_the_cwd(self):
        import os

        previous = os.getcwd()
        try:
            os.chdir(ROOT / "notebooks")
            self.assertEqual(ar._root(), ROOT)
        finally:
            os.chdir(previous)

    def test_an_explicit_project_root_still_wins(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(ar._root(tmp), Path(tmp).resolve())

    def test_the_public_api_works_from_any_cwd(self):
        import os

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_run(root, "obj", {"halpha_A": 6562.8, "h01_rv_sys_kms": -7.0},
                       chain={"stage_runs": {"A1": "obj"}, "reduction_profile": "cascade"})
            previous = os.getcwd()
            try:
                os.chdir(root / "runs")  # un cwd cualquiera que no es la raíz
                self.assertEqual(ar.reduction_profile("obj", project_root=root), "cascade")
                self.assertEqual(ar.resolve_a1_run("obj", project_root=root), "obj")
                self.assertEqual(ar.measured_runtime("obj", project_root=root)["nights"], [])
            finally:
                os.chdir(previous)


class HalphaTests(unittest.TestCase):
    def test_the_systemic_rv_has_no_silent_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_run(root, "obj", {"halpha_A": 6562.8}, chain={})
            with self.assertRaises(ar.A1ReviewError):
                ar.halpha_observed_A("obj", project_root=root)

    def test_the_rest_wavelength_has_no_silent_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_run(root, "obj", {"h01_rv_sys_kms": -7.0}, chain={})
            with self.assertRaises(ar.A1ReviewError):
                ar.halpha_observed_A("obj", project_root=root)

    def test_the_line_is_shifted_by_the_systemic_rv(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_run(root, "obj", {"halpha_A": 6562.8, "h01_rv_sys_kms": -7.0}, chain={})
            observed, rest, rv = ar.halpha_observed_A("obj", project_root=root)
            self.assertEqual((rest, rv), (6562.8, -7.0))
            self.assertAlmostEqual(observed, 6562.8 * (1.0 - 7.0 / 299792.458), places=6)
            self.assertLess(observed, rest)  # desplazamiento al azul


class PanelTests(unittest.TestCase):
    def _run_with_cube(self, root, **cube_kwargs):
        cube = root / "exp1" / "DATACUBE_FINAL.fits"
        _write_cube(cube, **cube_kwargs)
        run = _write_run(root, "obj",
                         {"cube_files": [str(cube)], "halpha_A": 6562.8, "h01_rv_sys_kms": 0.0},
                         chain={"stage_runs": {"A1": "obj"}})
        (run / "stages" / "stream_combine_plan.json").write_text(json.dumps({
            "exposures": [{"file": str(cube), "exposure_id": "2024-01-01_A", "exptime": 300.0,
                           "window": [4, 36, 6, 38], "y_center": 20.0, "x_center": 22.0}],
        }), encoding="utf-8")
        return cube

    def test_five_panels_cropped_to_the_declared_window(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._run_with_cube(root)
            row = ar.list_a1_cubes("obj", project_root=root)[0]
            panels = ar.cube_panels(row, halpha_A=4749.53125 + 30 * 1.25)
            self.assertEqual(len(panels.images), len(ar.PANEL_COLUMNS))
            for image in panels.images:
                self.assertEqual(image.shape, (32, 32))
            # el centro se expresa dentro del recorte
            self.assertEqual(panels.center_yx, (16.0, 16.0))
            self.assertEqual(panels.exptime, 300.0)

    def test_the_halpha_panel_lands_on_the_requested_channel(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._run_with_cube(root)
            row = ar.list_a1_cubes("obj", project_root=root)[0]
            bright = 4749.53125 + 30 * 1.25  # el canal que _write_cube multiplica por 3
            panels = ar.cube_panels(row, halpha_A=bright, halpha_halfwidth_ch=0)
            self.assertGreater(np.nanmax(panels.images[-1]), 2.0 * np.nanmax(panels.images[0]))

    def test_bands_avoid_the_ao_laser_and_telluric_ranges(self):
        wave = np.arange(4749.53125, 9350.0, 1.25)
        for center in ar.band_centers_A(wave):
            for low, high in ar.DEFAULT_BAD_RANGES:
                self.assertFalse(low <= center <= high, f"{center} cae en ({low}, {high})")

    def test_panels_without_a_declared_window_are_centred_on_the_primary(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cube = root / "final" / "DATACUBE_FINAL.fits"
            _write_cube(cube, peak_yx=(20, 22))
            _write_run(root, "obj",
                       {"cube_files": [str(cube)], "halpha_A": 6562.8, "h01_rv_sys_kms": 0.0},
                       chain={"stage_runs": {"A1": "obj"}})
            row = ar.list_a1_cubes("obj", project_root=root)[0]
            panels = ar.cube_panels(row, halpha_A=6562.8, crop_npix=40)
            self.assertEqual(tuple(int(v) for v in panels.center_yx), (20, 22))

    def test_the_cache_round_trips(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._run_with_cube(root)
            rows = ar.list_a1_cubes("obj", project_root=root)
            first = ar.build_panels("obj", rows, project_root=root)
            self.assertTrue(ar.panel_cache_path("obj", project_root=root).exists())
            second = ar.build_panels("obj", rows, project_root=root)
            self.assertEqual(second[0].note, "desde caché")
            np.testing.assert_allclose(first[0].images[0], second[0].images[0], rtol=1e-6)

    def test_a_legacy_run_is_never_written_to(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cube = root / "exp1" / "DATACUBE_FINAL.fits"
            _write_cube(cube)
            _write_run(root, "obj",
                       {"cube_files": [str(cube)], "halpha_A": 6562.8, "h01_rv_sys_kms": 0.0},
                       chain={"stage_runs": {"A1": "obj"}}, meta={"legacy": True})
            rows = ar.list_a1_cubes("obj", project_root=root)
            ar.build_panels("obj", rows, project_root=root)
            self.assertFalse(ar.panel_cache_path("obj", project_root=root).exists())


class HaloScaleTests(unittest.TestCase):
    @staticmethod
    def _halo(ny=80, nx=80, center=(40, 40)):
        yy, xx = np.mgrid[:ny, :nx]
        radius = np.hypot(yy - center[0], xx - center[1])
        return 5e4 * np.exp(-0.5 * (radius / 1.5) ** 2) + 300.0 / (1.0 + radius)

    def test_the_scale_ignores_the_saturated_core(self):
        image = self._halo()
        vmin, vmax = ar.halo_scale([image], [(40, 40)])
        self.assertLess(vmax, np.nanmax(image) / 10.0)
        self.assertLess(vmin, vmax)

    def test_all_nan_panels_do_not_raise(self):
        blank = np.full((20, 20), np.nan)
        self.assertEqual(ar.halo_scale([blank], [(10, 10)]), (0.0, 1.0))

    def test_a_missing_centre_is_skipped(self):
        self.assertEqual(ar.halo_scale([self._halo()], [None]), (0.0, 1.0))

    def test_the_norm_is_linear_around_the_background_noise(self):
        image = self._halo()
        norm = ar.halo_norm([image], [(40, 40)])
        self.assertGreater(norm.linear_width, 0.0)
        self.assertLess(norm.vmin, 0.0)
        self.assertGreater(norm.vmax, norm.vmin)

    def test_the_norm_survives_a_flat_image(self):
        flat = np.zeros((60, 60))
        norm = ar.halo_norm([flat], [(30, 30)])
        self.assertGreater(norm.vmax, norm.vmin)


class FigureSmokeTests(unittest.TestCase):
    """Las figuras se dibujan sin datos reales; solo se comprueba que no revientan."""

    def setUp(self):
        import matplotlib

        matplotlib.use("Agg")

    def test_the_schematic_draws_every_phase(self):
        phases = [
            ar.Phase(key="a", title="Inventario", lane="P1", status="ok",
                     inputs=(ar.PhaseIO("RAW"),), outputs=(ar.PhaseIO("CSV"),)),
            ar.Phase(key="b", title="flat", lane="P1", recipe="muse_flat", status="resumed",
                     inputs=ar._inputs_for("flat"), outputs=ar._outputs_for("flat", 3, 1)),
            ar.Phase(key="c", title="combinado", lane="P3", status="rejected", duration_s=120.0),
        ]
        figure = ar.draw_phase_schematic(phases)
        self.assertTrue(figure.axes)

    def test_the_mosaic_needs_at_least_one_cube(self):
        with self.assertRaises(ar.A1ReviewError):
            ar.draw_cube_mosaic([])

    def test_the_mosaic_has_one_row_per_cube(self):
        panels = [
            ar.PanelSet(label=f"exp{i}", center_yx=(10.0, 10.0),
                        images=[np.random.default_rng(i).normal(size=(20, 20))
                                for _ in ar.PANEL_COLUMNS],
                        wavelengths_A=(float("nan"), 5000.0, 7000.0, 9000.0, 6562.8))
            for i in range(3)
        ]
        figure = ar.draw_cube_mosaic(panels)
        self.assertEqual(len(figure.axes), 3 * len(ar.PANEL_COLUMNS))


if __name__ == "__main__":
    unittest.main()
