"""La unidad de flujo de M3, consolidada con `io.flux_unit_cgs`.

A4/M3 tenia su propio default silencioso (`m3_flux_unit_cgs`, 1e-20) — el ultimo
que quedaba tras el plan A — y a la vez anota en su QC la unidad que uso para
comparar con Gaia. Aqui se fija que M3 resuelve la unidad por la via compartida
(knob -> `BUNIT` del cubo -> error) y que ese valor anotado es la tercera fuente
para quien lea sus productos despues (E3/G3), ademas del contraste que evita que
`flux_factor` se aplique a un flujo en otra unidad sin que nadie se entere.
"""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from musepipe.io import (
    MUSE_NATIVE_BUNIT,
    flux_unit_cgs,
    flux_unit_conflict,
    flux_unit_from_m3_qc,
    read_stage00q_qc,
    resolve_flux_unit,
)
from musepipe.qc.cube_qc import compute_m3_flux
from musepipe.stages.stage_x11_calibrate import (
    compute_stage_x11_products,
    stage_x11_paths,
    write_stage_x11_products,
)
from tests.test_calibrate_no_double import calibration_product


def _qc00(m3):
    return {"stage": "00q_cube_qc", "m3_flux": m3}


class FluxUnitSourcesTests(unittest.TestCase):
    def test_m3_qc_is_read_only_when_m3_actually_measured(self):
        self.assertEqual(flux_unit_from_m3_qc(_qc00({"status": "green", "flux_unit_cgs": 1e-20})), 1e-20)
        # Un M3 que no llego a medir no respalda ninguna unidad, la anote o no.
        self.assertIsNone(flux_unit_from_m3_qc(_qc00({"status": "unavailable", "flux_unit_cgs": 1e-20})))
        self.assertIsNone(flux_unit_from_m3_qc(_qc00({"status": "green"})))
        self.assertIsNone(flux_unit_from_m3_qc(_qc00({"status": "green", "flux_unit_cgs": 0.0})))
        self.assertIsNone(flux_unit_from_m3_qc({}))
        self.assertIsNone(flux_unit_from_m3_qc(None))

    def test_the_order_is_knob_then_bunit_then_m3(self):
        qc = _qc00({"status": "green", "flux_unit_cgs": 1e-30})
        # 1. el knob manda (no mover resultados congelados)
        value, source = resolve_flux_unit({"h03_flux_unit_cgs": 1.0}, bunit=MUSE_NATIVE_BUNIT, qc_m3=qc)
        self.assertEqual(value, 1.0)
        self.assertEqual(source, "config:h03_flux_unit_cgs")
        # 2. sin knob, la unidad que viaja con el dato
        value, source = resolve_flux_unit({}, bunit=MUSE_NATIVE_BUNIT, qc_m3=qc)
        self.assertAlmostEqual(value / 1e-20, 1.0, places=9)
        self.assertIn("BUNIT", source)
        # 3. sin knob ni BUNIT (producto anterior a que B1/B2 lo propagaran)
        value, source = resolve_flux_unit({}, bunit="", qc_m3=qc)
        self.assertEqual(value, 1e-30)
        self.assertEqual(source, "stage00q_qc.m3_flux.flux_unit_cgs")

    def test_without_any_source_it_raises_instead_of_guessing(self):
        with self.assertRaises(ValueError) as ctx:
            resolve_flux_unit({}, bunit=None, qc_m3=_qc00({"status": "unavailable"}))
        self.assertIn("h03_flux_unit_cgs", str(ctx.exception))
        self.assertIn("m3", str(ctx.exception).lower())

    def test_flux_unit_cgs_accepts_the_m3_qc_too(self):
        self.assertEqual(flux_unit_cgs({}, bunit="", qc_m3=_qc00({"status": "green", "flux_unit_cgs": 1e-20})), 1e-20)

    def test_conflict_only_fires_when_both_exist_and_differ(self):
        agree = _qc00({"status": "green", "flux_unit_cgs": 1e-20})
        self.assertIsNone(flux_unit_conflict(MUSE_NATIVE_BUNIT, agree))
        self.assertIsNone(flux_unit_conflict("", agree))  # sin BUNIT no hay nada que contrastar
        self.assertIsNone(flux_unit_conflict(MUSE_NATIVE_BUNIT, _qc00({"status": "unavailable"})))
        message = flux_unit_conflict(MUSE_NATIVE_BUNIT, _qc00({"status": "green", "flux_unit_cgs": 1.0}))
        self.assertIsNotNone(message)
        self.assertIn("1e+20", message)  # el factor por el que sale mal la escala

    def test_read_stage00q_qc_is_tolerant(self):
        with tempfile.TemporaryDirectory() as tmp:
            stage_dir = Path(tmp)
            self.assertIsNone(read_stage00q_qc(None))
            self.assertIsNone(read_stage00q_qc(stage_dir))  # todavia no existe
            (stage_dir / "stage00q_qc.json").write_text(json.dumps(_qc00({"status": "green"})))
            self.assertEqual(read_stage00q_qc(stage_dir)["stage"], "00q_cube_qc")


class M3FluxUnitTests(unittest.TestCase):
    """M3 con el resolutor compartido: sin default silencioso."""

    def setUp(self):
        self.wave = np.arange(5000.0, 9300.0, 5.0, dtype=np.float64)
        self.primary = (20, 20)
        self.star_flux_native = 1000.0
        cube = np.zeros((self.wave.size, 41, 41), dtype=np.float64)
        cube[:, self.primary[0], self.primary[1]] = self.star_flux_native
        self.cube = cube
        self.tmp = tempfile.TemporaryDirectory()
        pb = Path(self.tmp.name) / "test_RP.csv"
        grid = np.arange(6100.0, 9010.0, 10.0)
        pb.write_text(
            "wavelength_A,response_photon\n"
            + "".join(f"{w:.4f},1.000000e+00\n" for w in grid),
            encoding="utf-8",
        )
        self.pb_dir = self.tmp.name

    def tearDown(self):
        self.tmp.cleanup()

    def _cfg(self, **extra):
        cfg = {
            "m3_recommended_band": "RP",
            "m3_passband_dir": str(self.pb_dir),
            "m3_passbands": {"RP": {"file": "test_RP.csv", "mag": 12.0,
                                    "ref_flambda_cgs": self.star_flux_native * 1e-20,
                                    "muse_overlap_frac": 1.0}},
            "m3_passband_source": "synthetic test",
        }
        cfg.update(extra)
        return cfg

    def _m3(self, cfg, **kwargs):
        return compute_m3_flux(self.cube, self.wave, cfg, primary_yx=self.primary,
                               aperture_radius_px=5.0, **kwargs)

    def test_the_cube_bunit_supplies_the_unit_when_the_knob_is_absent(self):
        m3 = self._m3(self._cfg(), bunit=MUSE_NATIVE_BUNIT)
        self.assertEqual(m3["status"], "green")
        self.assertAlmostEqual(m3["flux_factor"], 1.0, places=6)
        self.assertIn("BUNIT", m3["flux_unit_source"])
        self.assertEqual(m3["bunit"], MUSE_NATIVE_BUNIT)

    def test_the_declared_knob_still_wins_over_the_header(self):
        # El BUNIT de MUSE parsea como 1.0000000000000001e-20 y el knob es exacto:
        # el knob manda para no mover en el ultimo bit resultados ya congelados.
        m3 = self._m3(self._cfg(m3_flux_unit_cgs=1e-20), bunit=MUSE_NATIVE_BUNIT)
        self.assertEqual(m3["flux_unit_cgs"], 1e-20)
        self.assertEqual(m3["flux_unit_source"], "config:m3_flux_unit_cgs")

    def test_without_knob_and_without_bunit_m3_is_unavailable_not_guessed(self):
        # Antes suponia 1e-20 en silencio: un cubo en otras unidades daba un
        # flux_factor mal por 1e20 con status green.
        m3 = self._m3(self._cfg())
        self.assertEqual(m3["status"], "unavailable")
        self.assertEqual(m3["reason"], "flux_unit_unknown")
        self.assertIsNone(m3.get("flux_factor"))
        self.assertIn("m3_flux_unit_cgs", m3["detail"])

    def test_the_run_input_cube_is_the_fallback_for_the_bunit(self):
        # `resolve_bunit` cae al cubo de entrada del run: A4 puede correr sobre un
        # cubo intermedio sin cabecera sin que se pierda la unidad.
        from astropy.io import fits

        cube_path = Path(self.tmp.name) / "input_cube.fits"
        hdu = fits.PrimaryHDU(np.zeros((2, 2, 2), dtype=np.float32))
        hdu.header["BUNIT"] = MUSE_NATIVE_BUNIT
        hdu.writeto(cube_path)
        m3 = self._m3(self._cfg(cube_files=[str(cube_path)]))
        self.assertEqual(m3["status"], "green")
        self.assertEqual(m3["bunit"], MUSE_NATIVE_BUNIT)


class D2FluxUnitBlockTests(unittest.TestCase):
    """D2 deja las tres fuentes resueltas y contrastadas en un solo sitio."""

    METHOD_KEYS = {
        "aperture": "spec_aperture_object",
        "optimal_ls": "spec_optimal_object",
        "optimal_psfsub": "spec_optimal_psfsub_object",
        "psffit": "spec_psffit_object",
        "sgf": "spec_sgf_object",
        "lpm": "spec_lpm_object",
    }

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.run_id = "synthetic_x11_unit"
        self.paths = stage_x11_paths(self.run_id, project_root=self.root)
        self.paths["paths"].ensure_base_dirs()
        self.cfg = {"run_id": self.run_id, "project_root": str(self.root),
                    "x11_canonical_method": "psffit"}

    def tearDown(self):
        self.tmp.cleanup()

    def _run(self, m3, *, bunit=MUSE_NATIVE_BUNIT):
        self.paths["stage00q_qc_json"].write_text(json.dumps({
            "stage": "00q_cube_qc", "run_id": self.run_id,
            "cube": {"wavelength_frame": "barycentric", "vbary_kms": 4.0},
            "m1_wavelength": {"status": "green", "offset_median_A": 0.0},
            "m3_flux": m3,
        }), encoding="utf-8")
        wave = np.linspace(6000.0, 9000.0, 601)
        for method, key in self.METHOD_KEYS.items():
            product = calibration_product(method=method, wave=wave,
                                          flux=np.full(wave.size, 10.0))
            if bunit is not None:
                product.header["BUNIT"] = bunit
            product.write(self.paths[key])
        stage = compute_stage_x11_products(self.cfg, self.paths)
        return write_stage_x11_products(stage, self.cfg, self.paths)["qc"]

    def test_the_qc_reports_the_unit_that_e3_and_g3_will_resolve(self):
        qc = self._run({"status": "green", "scale_factor": 1.0, "flux_unit_cgs": 1e-20})
        unit = qc["flux"]["unit"]
        self.assertEqual(unit["bunit"], MUSE_NATIVE_BUNIT)
        self.assertAlmostEqual(unit["from_bunit"] / 1e-20, 1.0, places=9)
        self.assertEqual(unit["from_m3_qc"], 1e-20)
        self.assertIn("BUNIT", unit["source"])
        self.assertIsNone(unit["conflict"])

    def test_a_unit_mismatch_between_the_product_and_m3_becomes_an_open_issue(self):
        # M3 midio el factor suponiendo 1.0 y el producto viene en 1e-20: el
        # factor se aplicaria a un flujo 1e20 veces distinto.
        qc = self._run({"status": "green", "flux_factor": 0.97, "flux_unit_cgs": 1.0})
        self.assertIsNotNone(qc["flux"]["unit"]["conflict"])
        self.assertTrue(any("no coinciden" in issue for issue in qc["open_issues"]))

    def test_the_m3_qc_covers_a_product_that_lost_its_bunit(self):
        qc = self._run({"status": "green", "scale_factor": 1.0, "flux_unit_cgs": 1e-20}, bunit=None)
        unit = qc["flux"]["unit"]
        self.assertIsNone(unit["from_bunit"])
        self.assertEqual(unit["cgs"], 1e-20)
        self.assertEqual(unit["source"], "stage00q_qc.m3_flux.flux_unit_cgs")
        self.assertIsNone(unit["conflict"])

    def test_no_unit_anywhere_is_reported_without_breaking_the_stage(self):
        qc = self._run({"status": "unavailable"}, bunit=None)
        unit = qc["flux"]["unit"]
        self.assertIsNone(unit["cgs"])
        self.assertIsNone(unit["source"])
        self.assertIn("h03_flux_unit_cgs", unit["reason"])


if __name__ == "__main__":
    unittest.main()
