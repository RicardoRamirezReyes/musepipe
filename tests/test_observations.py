"""La vista por observación de un run: rutas vivas y geometría utilizable.

Lo que se fija aquí son regresiones concretas, no la felicidad del camino
feliz:

* que la geometría salga del `stream_combine_plan.json` y no de una convención
  de nombres sobre un disco;
* que una ruta muerta se re-resuelva **por `exposure_id`** contra lo que el
  config declara — el caso real de ROXs 12 b, cuyo árbol de reducción se
  borró — y que al hacerlo el centroide se vuelva a medir, porque un cubo
  re-reducido no tiene por qué caer en el mismo píxel;
* que lo que no se pueda resolver **falle con la lista**, en vez de mezclar una
  PSF con menos exposiciones de las que tiene el combinado.
"""

import json
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np

from musepipe.observations import (
    ObservationInputError,
    resolve_observation_plan,
)
from musepipe.reduction.stream_combine import (
    build_stream_combine_plan,
    exposure_from_measurement,
    measure_primary_center,
)
from test_stream_combine import make_cube

CROP = 20
PAD = 4


def _write_run(root, run_id, config):
    run = root / "runs" / run_id
    (run / "config").mkdir(parents=True, exist_ok=True)
    (run / "stages").mkdir(parents=True, exist_ok=True)
    payload = {"meta": {}, "config": {"run_id": run_id, **config}}
    (run / "config" / "config.json").write_text(json.dumps(payload), encoding="utf-8")
    return run


def _exposures(root, centers, *, subdir="cubes"):
    """Un cubo sintético por exposición, en `<root>/<subdir>/<exposure_id>/`."""

    paths = []
    for index, (y, x) in enumerate(centers):
        exposure_id = f"2022-08-2{index}_MUSE.exp{index:02d}"
        path = root / subdir / exposure_id / "DATACUBE_FINAL.fits"
        path.parent.mkdir(parents=True, exist_ok=True)
        make_cube(path, y_center=y, x_center=x, mjd_obs=59819.0 + index)
        paths.append(path)
    return paths


def _write_plan(root, run_id, cubes, *, output="combined.fits"):
    plan = build_stream_combine_plan(
        [str(p) for p in cubes],
        run_id=run_id,
        output=str(root / output),
        crop_npix=CROP,
        pad=PAD,
        chunk_channels=8,
    )
    path = root / "runs" / run_id / "stages" / "stream_combine_plan.json"
    path.write_text(json.dumps(plan.as_dict()), encoding="utf-8")
    return plan


class ResolveObservationPlanTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.centers = [(30.2, 31.4), (29.6, 30.9), (30.9, 31.1)]

    def tearDown(self):
        self._tmp.cleanup()

    def test_geometry_comes_from_the_combine_plan(self):
        cubes = _exposures(self.root, self.centers)
        _write_run(self.root, "obj", {"cube_files": [str(self.root / "combined.fits")]})
        plan = _write_plan(self.root, "obj", cubes)

        obs = resolve_observation_plan("obj", project_root=self.root)

        self.assertEqual(len(obs), len(cubes))
        self.assertEqual(obs.substituted, ())
        self.assertEqual(obs.remeasured, ())
        for resolved, original in zip(obs.exposures, plan.exposures):
            self.assertEqual(resolved.file, original.file)
            self.assertEqual(resolved.window, original.window)
            self.assertAlmostEqual(resolved.shift_y, original.shift_y)
            self.assertAlmostEqual(resolved.weight, original.weight)

    def test_a_dead_path_is_re_resolved_by_exposure_id(self):
        """El caso de ROXs 12 b: el plan apunta a un árbol que ya no existe."""

        cubes = _exposures(self.root, self.centers, subdir="cubes")
        _write_run(self.root, "obj", {"cube_files": [str(self.root / "combined.fits")]})
        _write_plan(self.root, "obj", cubes)

        # Los mismos cubos, en otro sitio, y el árbol del plan borrado.
        moved = self.root / "perexp"
        shutil.copytree(self.root / "cubes", moved)
        shutil.rmtree(self.root / "cubes")
        _write_run(
            self.root,
            "obj",
            {
                "cube_files": [str(self.root / "combined.fits")],
                "perexp_dir": str(moved),
            },
        )

        obs = resolve_observation_plan("obj", project_root=self.root)

        self.assertEqual(len(obs.substituted), len(cubes))
        self.assertEqual(len(obs.remeasured), len(cubes))
        for exposure in obs.exposures:
            self.assertTrue(Path(exposure.file).exists())
            self.assertTrue(str(exposure.file).startswith(str(moved)))
        # Es el MISMO dato copiado, así que la re-medida tiene que coincidir.
        for row in obs.remeasured:
            self.assertLess(abs(row["dy_px"]), 1e-6)
            self.assertLess(abs(row["dx_px"]), 1e-6)

    def test_perexp_cubes_list_resolves_too(self):
        cubes = _exposures(self.root, self.centers)
        _write_run(self.root, "obj", {"cube_files": [str(self.root / "combined.fits")]})
        _write_plan(self.root, "obj", cubes)
        moved = self.root / "otro"
        shutil.copytree(self.root / "cubes", moved)
        shutil.rmtree(self.root / "cubes")
        _write_run(
            self.root,
            "obj",
            {
                "cube_files": [str(self.root / "combined.fits")],
                "perexp_cubes": sorted(str(p) for p in moved.glob("*/DATACUBE_FINAL.fits")),
            },
        )

        obs = resolve_observation_plan("obj", project_root=self.root)
        self.assertEqual(len(obs), len(cubes))

    def test_a_moved_star_is_measured_again_and_warned_about(self):
        """Una re-reducción puede dejar la estrella en otro píxel.

        Usar la geometría vieja sobre datos nuevos desalinearía el modelo justo
        donde más duele —el núcleo—, así que se re-mide y se avisa.
        """

        cubes = _exposures(self.root, self.centers)
        _write_run(self.root, "obj", {"cube_files": [str(self.root / "combined.fits")]})
        plan = _write_plan(self.root, "obj", cubes)

        moved = self.root / "rereduccion"
        moved.mkdir()
        for cube, (y, x) in zip(cubes, self.centers):
            destination = moved / cube.parent.name / "DATACUBE_FINAL.fits"
            destination.parent.mkdir(parents=True)
            make_cube(destination, y_center=y + 2.0, x_center=x)
        shutil.rmtree(self.root / "cubes")
        _write_run(
            self.root,
            "obj",
            {"cube_files": [str(self.root / "combined.fits")], "perexp_dir": str(moved)},
        )

        obs = resolve_observation_plan("obj", project_root=self.root)

        self.assertTrue(all(row["dy_px"] > 1.5 for row in obs.remeasured))
        self.assertTrue(any("se movió" in w for w in obs.warnings))
        # Y la ventana usada es la NUEVA, no la del plan.
        for resolved, original in zip(obs.exposures, plan.exposures):
            self.assertNotEqual(resolved.window, original.window)

    def test_an_unresolvable_exposure_is_an_error_with_the_list(self):
        cubes = _exposures(self.root, self.centers)
        _write_run(self.root, "obj", {"cube_files": [str(self.root / "combined.fits")]})
        _write_plan(self.root, "obj", cubes)
        missing_id = cubes[1].parent.name
        shutil.rmtree(cubes[1].parent)

        with self.assertRaises(ObservationInputError) as ctx:
            resolve_observation_plan("obj", project_root=self.root)
        self.assertIn(missing_id, str(ctx.exception))

    def test_without_a_plan_it_says_how_to_make_one(self):
        _exposures(self.root, self.centers)
        _write_run(self.root, "obj", {"cube_files": [str(self.root / "combined.fits")]})

        with self.assertRaises(ObservationInputError) as ctx:
            resolve_observation_plan("obj", project_root=self.root)
        self.assertIn("plan_stream_combine", str(ctx.exception))

    def test_the_resolved_view_round_trips_through_json(self):
        cubes = _exposures(self.root, self.centers)
        _write_run(self.root, "obj", {"cube_files": [str(self.root / "combined.fits")]})
        _write_plan(self.root, "obj", cubes)

        obs = resolve_observation_plan("obj", project_root=self.root)
        cache = self.root / "runs" / "obj" / "stages" / "observation_plan.json"
        cache.write_text(json.dumps(obs.to_json()), encoding="utf-8")

        again = resolve_observation_plan("obj", project_root=self.root, plan_json=cache)
        self.assertEqual([e.file for e in again.exposures], [e.file for e in obs.exposures])
        self.assertEqual([e.window for e in again.exposures], [e.window for e in obs.exposures])
        self.assertEqual(again.source, obs.source)

    def test_a_stale_cache_is_rejected_instead_of_used(self):
        cubes = _exposures(self.root, self.centers)
        _write_run(self.root, "obj", {"cube_files": [str(self.root / "combined.fits")]})
        _write_plan(self.root, "obj", cubes)
        obs = resolve_observation_plan("obj", project_root=self.root)
        cache = self.root / "runs" / "obj" / "stages" / "observation_plan.json"
        cache.write_text(json.dumps(obs.to_json()), encoding="utf-8")
        shutil.rmtree(cubes[0].parent)

        with self.assertRaises(ObservationInputError):
            resolve_observation_plan("obj", project_root=self.root, plan_json=cache)

    def test_provenance_carries_what_a_qc_needs(self):
        cubes = _exposures(self.root, self.centers)
        _write_run(self.root, "obj", {"cube_files": [str(self.root / "combined.fits")]})
        _write_plan(self.root, "obj", cubes)

        provenance = resolve_observation_plan("obj", project_root=self.root).provenance()

        self.assertEqual(provenance["n_exposures"], len(cubes))
        self.assertEqual(provenance["crop_npix"], CROP)
        self.assertTrue(provenance["source_plan"].endswith("stream_combine_plan.json"))
        self.assertEqual(len(provenance["exposures"]), len(cubes))
        self.assertIn("weight", provenance["exposures"][0])


class ExposureFromMeasurementTests(unittest.TestCase):
    """La geometría extraída de `build_stream_combine_plan` es la misma."""

    def test_it_reproduces_the_plan_geometry(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cubes = _exposures(root, [(30.2, 31.4), (29.6, 30.9)])
            plan = build_stream_combine_plan(
                [str(p) for p in cubes],
                run_id="obj",
                output=str(root / "combined.fits"),
                crop_npix=CROP,
                pad=PAD,
                chunk_channels=8,
            )
            for index, (cube, expected) in enumerate(zip(cubes, plan.exposures)):
                measurement = measure_primary_center(cube)
                rebuilt = exposure_from_measurement(
                    measurement,
                    index=index,
                    weight=expected.weight,
                    crop_npix=CROP,
                    pad=PAD,
                )
                self.assertEqual(rebuilt.window, expected.window)
                self.assertEqual(rebuilt.exposure_id, expected.exposure_id)
                np.testing.assert_allclose(
                    [rebuilt.shift_y, rebuilt.shift_x],
                    [expected.shift_y, expected.shift_x],
                    rtol=0,
                    atol=0,
                )


if __name__ == "__main__":
    unittest.main()
