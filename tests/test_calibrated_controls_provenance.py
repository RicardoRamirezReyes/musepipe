"""Los controles calibrados son un producto de D2, y traen su procedencia.

`spec_calibrated_{metodo}_controls.npz` tuvo cuatro lectores (E1, E2, G2 y el
propio D2) y **ningun escritor** hasta 2026-08-08. Los de `ROXs12b_realigned`
eran del 2026-07-09/07-15: sobrevivieron al cambio de cubo del 07-28 y a la
re-ejecucion entera del 08-07 sin que nada lo dijera, porque la unica
comprobacion era `shape[1] == n_wave` y el eje espectral no cambia entre
reducciones del mismo objeto.

Importa mas que en otros sitios porque E1 normaliza CADA control con el error
del objeto (`analyze_halpha_method`), asi que en `f_stat = z_99 x matched_sigma`
sigma se cancela y el limite de E3 se reduce a `q99(flujo de los controles)`:
unos controles de otro cubo fijan ellos solos el limite publicado. Medido sobre
el run canonico, los huerfanos tenian la MEDIANA de la distribucion nula a 8-33
sigma, y los del cubo actual a 1.7-2.9.
"""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from musepipe.io import CALIBRATED_CONTROLS_STAMP, load_calibrated_controls
from musepipe.stages.stage_x11_calibrate import (
    _RAW_CONTROL_NPZ,
    calibrate_control_spectra,
    compute_stage_x11_products,
    stage_x11_paths,
    write_stage_x11_products,
)
from tests.test_x11_spectra_table import METHOD_KEYS, _product


class LoadCalibratedControlsTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.controls = np.arange(12.0).reshape(3, 4)

    def tearDown(self):
        self.tmp.cleanup()

    def _write(self, name, **extra):
        path = self.dir / name
        np.savez(path, control_spectra=self.controls, **extra)
        return path

    def test_a_stamped_npz_loads(self):
        path = self._write("ok.npz", written_by=CALIBRATED_CONTROLS_STAMP)
        np.testing.assert_allclose(load_calibrated_controls(path), self.controls)

    def test_an_unstamped_npz_is_refused_by_name(self):
        # El caso real: el fichero de julio que nadie escribia.
        path = self._write("orphan.npz")
        with self.assertRaisesRegex(RuntimeError, "provenance stamp"):
            load_calibrated_controls(path)

    def test_controls_older_than_the_object_are_refused(self):
        path = self._write("stale.npz", written_by=CALIBRATED_CONTROLS_STAMP)
        obj = self.dir / "spec_calibrated_psffit_object.fits"
        obj.write_bytes(b"")
        import os

        old = path.stat().st_mtime - 30 * 24 * 3600
        os.utime(path, (old, old))
        with self.assertRaisesRegex(RuntimeError, "older than"):
            load_calibrated_controls(path, object_path=obj)

    def test_a_few_seconds_apart_is_not_staleness(self):
        # Escribir seis productos y sus npz no es atomico.
        path = self._write("fresh.npz", written_by=CALIBRATED_CONTROLS_STAMP)
        obj = self.dir / "obj.fits"
        obj.write_bytes(b"")
        import os

        older = path.stat().st_mtime - 20.0
        os.utime(path, (older, older))
        load_calibrated_controls(path, object_path=obj)

    def test_a_missing_file_still_raises_filenotfound(self):
        # G2 distingue "no estan" (motivo blando) de "estan y son sospechosos".
        with self.assertRaises(FileNotFoundError):
            load_calibrated_controls(self.dir / "nope.npz")


class D2WritesCalibratedControlsTests(unittest.TestCase):
    """D2 los emite, con el mismo tratamiento que le da al objeto."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.run_id = "synthetic_x11_controls"
        self.paths = stage_x11_paths(self.run_id, project_root=self.root)
        self.paths["paths"].ensure_base_dirs()
        self.paths["stage00q_qc_json"].write_text(
            json.dumps({
                "stage": "00q_cube_qc", "run_id": self.run_id,
                "cube": {"wavelength_frame": "barycentric", "vbary_kms": 4.0},
                "m1_wavelength": {"status": "green", "offset_median_A": 0.0},
                "m3_flux": {"status": "green", "scale_factor": 1.0},
            }),
            encoding="utf-8",
        )
        self.cfg = {"run_id": self.run_id, "project_root": str(self.root),
                    "x11_canonical_method": "psffit"}
        self.stage_dir = self.paths["spec_final_object"].parent
        self.raw = {}
        rng = np.random.default_rng(11)
        for method, key in METHOD_KEYS.items():
            product = _product(method)
            product.write(self.paths[key])
            controls = rng.normal(0.0, 1.0, size=(5, product.wave_A.size))
            self.raw[method] = controls
            np.savez(self.stage_dir / _RAW_CONTROL_NPZ[method], control_spectra=controls)

    def tearDown(self):
        self.tmp.cleanup()

    def _run(self):
        product = compute_stage_x11_products(self.cfg, self.paths)
        return product, write_stage_x11_products(product, self.cfg, self.paths)

    def test_every_method_gets_a_stamped_npz_that_its_consumers_accept(self):
        _product_, written = self._run()
        listed = written["qc"]["products"]["calibrated_controls_by_method"]
        self.assertEqual(set(listed), set(METHOD_KEYS))
        for method in METHOD_KEYS:
            path = self.stage_dir / f"spec_calibrated_{method}_controls.npz"
            self.assertTrue(path.exists(), method)
            # Lo que importa: pasan la guardia que rechazaba a los huerfanos.
            loaded = load_calibrated_controls(
                path, object_path=self.paths[METHOD_KEYS[method]]
            )
            self.assertEqual(loaded.shape, self.raw[method].shape)

    def test_the_controls_carry_the_same_flux_scale_as_the_object(self):
        product, _ = self._run()
        for method in METHOD_KEYS:
            scale = float(product.calibrated[method].product.header["FLXSCL"])
            written = load_calibrated_controls(
                self.stage_dir / f"spec_calibrated_{method}_controls.npz"
            )
            np.testing.assert_allclose(written, self.raw[method] * scale, rtol=1e-12)

    def test_d2_no_longer_prefers_an_orphan_file_over_the_current_controls(self):
        # La regresion exacta: antes `_method_control_bias` leia el npz
        # calibrado si existia, asi que un fichero viejo se colaba delante de
        # los controles que C acababa de escribir.
        method = "psffit"
        np.savez(
            self.stage_dir / f"spec_calibrated_{method}_controls.npz",
            control_spectra=np.full_like(self.raw[method], 1e6),
        )
        got = calibrate_control_spectra(self.stage_dir, method, flux_scale=1.0)
        np.testing.assert_allclose(got, self.raw[method], rtol=1e-12)

    def test_a_method_without_raw_controls_is_simply_absent(self):
        (self.stage_dir / _RAW_CONTROL_NPZ["lpm"]).unlink()
        _product_, written = self._run()
        listed = written["qc"]["products"]["calibrated_controls_by_method"]
        self.assertNotIn("lpm", listed)
        self.assertIn("psffit", listed)


if __name__ == "__main__":
    unittest.main()
