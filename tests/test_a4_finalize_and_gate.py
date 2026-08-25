"""A4: lo derivado del documento, y como F1 lo cuenta.

El hueco que dejo pasar todo esto: `tests/test_report_gate_policy.py` construye
QC sinteticos SIN el rollup (`status` / `status_detail`), asi que la politica de
compuerta se probaba sobre documentos que no se parecen a los de disco. Aqui los
QC de prueba tienen la forma real.
"""

import argparse
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from astropy.io import fits

from musepipe.qc.cube_qc import (
    check_cube_phase,
    derive_downstream_decision,
    derive_open_issues,
    finalize_qc,
    measure_sky_radial_profile,
    resolve_primary_yx,
    resolve_qc_wavelength_frame,
)
from musepipe.stages.stage_x01_aperture import _wavelength_frame
from musepipe.report import (
    ACCEPTED_LIMITATIONS,
    ACCEPTED_LIMITATIONS_HASH,
    aggregate_open_issues,
    stage_status,
)


def _a4_qc(*, m5="red", m4="yellow", m1="green", m2="yellow", m3="green"):
    """Un QC de A4 con la forma de los de disco: metricas + rollup."""

    qc = {
        "stage": "00q_cube_qc",
        "run_id": "TEST",
        "timestamp_utc": "2026-08-24T00:00:00Z",
        "cube": {"file": "cube.fits", "wavelength_frame": "barycentric"},
        "m1_wavelength": {"status": m1, "offset_median_A": 0.064},
        "m2_lsf": {"status": m2, "max_dev_vs_nominal_pct": 22.2},
        "m3_flux": {"status": m3, "flux_factor": 1.0},
        "m4_sky": {"status": m4, "rms_continuum": 2.848, "median_bias": 1.137, "R": 0.618},
        "m5_stat": {"status": m5, "factor_spaxel_median": 6.78, "factor_box3_median": 17.22},
        "open_issues": [],
    }
    return qc


class FinalizeTests(unittest.TestCase):
    def test_finalize_is_idempotent(self):
        qc = _a4_qc()
        finalize_qc(qc)
        once = json.dumps(qc, indent=2, sort_keys=True)
        finalize_qc(qc)
        self.assertEqual(once, json.dumps(qc, indent=2, sort_keys=True))

    def test_finalize_never_touches_the_measured_metrics(self):
        # M1-M5 los consume la cadena (stage_x01_aperture lee m5_stat para
        # elegir entre STAT y errores empiricos): finalize no puede moverlos.
        qc = _a4_qc()
        before = json.loads(json.dumps(qc))
        finalize_qc(qc)
        for key in ("m1_wavelength", "m2_lsf", "m3_flux", "m4_sky", "m5_stat"):
            self.assertEqual(qc[key], before[key], key)

    def test_rollup_is_written_and_names_its_driver(self):
        qc = _a4_qc()
        finalize_qc(qc)
        self.assertEqual(qc["status"], "red")
        self.assertEqual(qc["status_detail"]["driven_by"], "m5_stat")

    def test_downstream_decision_is_derived_from_m5(self):
        red = derive_downstream_decision(_a4_qc(m5="red"))
        self.assertEqual(red["decision"], "use_empirical_controls")
        self.assertFalse(red["native_stat_as_sigma"])
        green = derive_downstream_decision(_a4_qc(m5="green"))
        self.assertTrue(green["native_stat_as_sigma"])
        # Sin M5 medida no se declara ninguna decision: no medir no es decidir.
        self.assertIsNone(derive_downstream_decision(_a4_qc(m5="unavailable")))

    def test_decision_keeps_an_already_approved_date(self):
        qc = _a4_qc()
        qc["downstream_decision"] = {"approved_utc": "2026-07-28T23:33:17Z", "decision": "x"}
        finalize_qc(qc)
        self.assertEqual(qc["downstream_decision"]["approved_utc"], "2026-07-28T23:33:17Z")

    def test_unmeasured_metric_is_major_not_silence(self):
        # Lo que le paso a 42B b: M5 sin medir se volvia stat_factor = 1.0.
        issues = derive_open_issues(_a4_qc(m5="unavailable"))
        m5 = [i for i in issues if i["metric"] == "m5_stat"]
        self.assertEqual(len(m5), 1)
        self.assertEqual(m5[0]["priority"], "major")

    def test_m4_yellow_is_info_and_m5_red_is_accepted(self):
        issues = {i["metric"]: i["priority"] for i in derive_open_issues(_a4_qc())}
        self.assertEqual(issues["m4_sky"], "info")
        self.assertEqual(issues["m5_stat"], "accepted")

    def test_manual_notes_survive_but_literal_duplicates_do_not(self):
        qc = _a4_qc()
        derived_m5 = derive_open_issues(qc)[-1]["issue"]
        qc["open_issues"] = [derived_m5, "gaia_passbands/ has only a README."]
        finalize_qc(qc)
        texts = [i["issue"] for i in qc["open_issues"]]
        self.assertEqual(texts.count(derived_m5), 1)
        self.assertIn("gaia_passbands/ has only a README.", texts)


class RollupGateTests(unittest.TestCase):
    accepted = ACCEPTED_LIMITATIONS["A4_cube_qc"]

    def test_rollup_of_an_accepted_red_does_not_block(self):
        # El defecto original: `m5_stat.status` estaba aceptada, pero el mismo
        # rojo volvia a entrar por `status` y `status_detail.status`.
        qc = _a4_qc()
        finalize_qc(qc)
        status, issues, applied = stage_status("A4_cube_qc", qc, accepted=self.accepted)
        self.assertEqual(status, "yellow")
        paths = {a["path"] for a in applied}
        self.assertEqual(paths, {"m5_stat.status", "status", "status_detail.status"})
        # Nada se esconde: el rojo crudo sigue anotado.
        self.assertTrue(any("status=red [accepted limitation" in i for i in issues))

    def test_a_new_red_metric_still_blocks_through_the_rollup(self):
        qc = _a4_qc(m1="red")
        finalize_qc(qc)
        status, issues, _ = stage_status("A4_cube_qc", qc, accepted=self.accepted)
        self.assertEqual(status, "red")
        self.assertTrue(any("m1_wavelength.status=red" in i for i in issues))

    def test_a_rollup_red_with_no_red_metric_is_a_contradiction_and_blocks(self):
        qc = _a4_qc(m5="green", m4="green", m2="green")
        finalize_qc(qc)
        qc["status"] = "red"  # documento incoherente consigo mismo
        status, issues, _ = stage_status("A4_cube_qc", qc, accepted=self.accepted)
        self.assertEqual(status, "red")
        self.assertTrue(any("no red metric underneath" in i for i in issues))

    def test_a_qc_without_status_detail_behaves_exactly_as_before(self):
        qc = {"m5_stat": {"status": "red"}, "status": "red"}
        status, _, _ = stage_status("A4_cube_qc", qc, accepted=self.accepted)
        self.assertEqual(status, "red")

    def test_the_frozen_policy_is_untouched(self):
        # El arreglo es del mecanismo, no de la lista de limitaciones.
        self.assertEqual(ACCEPTED_LIMITATIONS_HASH, "63df641f6073")


class DeclaredPriorityTests(unittest.TestCase):
    def test_declared_priority_wins_over_the_red_stage_heuristic(self):
        rows = [{"stage": "A4_cube_qc", "status": "red", "summary": "x"}]
        payloads = {
            "A4_cube_qc": {
                "qc": {
                    "open_issues": [
                        {"issue": "retain as a systematic diagnostic", "priority": "info"},
                        {"issue": "really broken", "priority": "blocking"},
                    ]
                }
            }
        }
        out = aggregate_open_issues(rows, payloads)
        by_text = {i["issue"]: i["priority"] for i in out if not i["issue"].startswith("x")}
        self.assertEqual(by_text["retain as a systematic diagnostic"], "minor")
        self.assertEqual(by_text["really broken"], "blocking")

    def test_bare_strings_keep_the_old_heuristic(self):
        rows = [{"stage": "C2_aperture", "status": "red", "summary": "x"}]
        payloads = {"C2_aperture": {"qc": {"open_issues": ["something"]}}}
        out = aggregate_open_issues(rows, payloads)
        self.assertEqual([i["priority"] for i in out if i["issue"] == "something"], ["blocking"])

    def test_dict_issues_are_rendered_as_text_not_as_a_repr(self):
        rows = [{"stage": "A4_cube_qc", "status": "yellow", "summary": "x"}]
        payloads = {"A4_cube_qc": {"qc": {"open_issues": [{"issue": "hello", "priority": "info"}]}}}
        out = aggregate_open_issues(rows, payloads)
        self.assertIn("hello", [i["issue"] for i in out])
        self.assertFalse(any(i["issue"].startswith("{") for i in out))


class CheckCubeGuardTests(unittest.TestCase):
    """`check_cube_phase` reescribe el documento ENTERO: sin guarda, re-correrlo
    borra M1-M5 sin avisar."""

    def _cube(self, directory: Path) -> Path:
        cube = directory / "cube.fits"
        data = np.zeros((4, 3, 3), dtype=np.float32)
        header = fits.Header({"CRVAL3": 4750.0, "CD3_3": 1.25, "CRPIX3": 1.0, "SPECSYS": "BARYCENT"})
        fits.HDUList(
            [fits.PrimaryHDU(), fits.ImageHDU(data=data, header=header, name="DATA")]
        ).writeto(cube, overwrite=True)
        return cube

    def _args(self, cube: Path, qc_path: Path, *, force: bool) -> argparse.Namespace:
        return argparse.Namespace(
            cube=str(cube), provenance="raw_reduction", run_id="TEST", qc_output=str(qc_path),
            skyline_source_cube="same", skip_checksum=True, project_root=None, force=force,
        )

    def test_it_refuses_to_discard_measured_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            qc_path = tmp / "stage00q_qc.json"
            qc_path.write_text(json.dumps(_a4_qc()), encoding="utf-8")
            self.assertEqual(check_cube_phase(self._args(self._cube(tmp), qc_path, force=False)), 2)
            self.assertEqual(json.loads(qc_path.read_text())["m5_stat"]["status"], "red")

    def test_force_discards_them_on_purpose(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            qc_path = tmp / "stage00q_qc.json"
            qc_path.write_text(json.dumps(_a4_qc()), encoding="utf-8")
            self.assertEqual(check_cube_phase(self._args(self._cube(tmp), qc_path, force=True)), 0)
            self.assertEqual(json.loads(qc_path.read_text())["m5_stat"]["status"], "unavailable")

    def test_a_fresh_qc_needs_no_force(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            qc_path = tmp / "stage00q_qc.json"
            self.assertEqual(check_cube_phase(self._args(self._cube(tmp), qc_path, force=False)), 0)
            written = json.loads(qc_path.read_text())
            self.assertEqual(written["cube"]["wavelength_frame"], "barycentric")
            self.assertEqual(written["cube"]["wavelength_frame_source"], "cube header")


class WavelengthFramePrecedenceTests(unittest.TestCase):
    """El knob declarado gana; y cuando no hay ninguno, se dice, no se supone.

    El defecto que esto cierra: A4 escribia `unknown` (la cabecera del cubo de
    ROXs 42B b no trae SPECSYS ni RVCORR), C2 lo tomaba como primario, y el
    `x01_wframe: barycentric` del run se descartaba en silencio. Los espectros
    definitivos salieron estampados `WFRAME = topocentric`.
    """

    def test_declared_knob_wins_over_an_unknown_qc(self):
        issues = []
        frame = _wavelength_frame(
            {"x01_wframe": "barycentric"}, {"cube": {"wavelength_frame": "unknown"}}, issues
        )
        self.assertEqual(frame, "barycentric")
        self.assertEqual(issues, [])

    def test_qc_is_used_when_no_knob_is_declared(self):
        issues = []
        frame = _wavelength_frame({}, {"cube": {"wavelength_frame": "topocentric"}}, issues)
        self.assertEqual(frame, "topocentric")
        self.assertEqual(issues, [])

    def test_with_neither_it_says_unavailable_instead_of_guessing(self):
        issues = []
        frame = _wavelength_frame({}, {"cube": {"wavelength_frame": "unknown"}}, issues)
        self.assertEqual(frame, "unavailable")
        self.assertEqual(len(issues), 1)
        self.assertNotIn("topocentric", issues[0])

    def test_each_stage_reads_its_own_knob(self):
        # C3/C4 llaman a la misma funcion con su config, donde `x01_wframe` no
        # existe: leerla siempre a ella dejaba fuera lo que el run declaraba.
        issues = []
        cfg = {"x02_wframe": "barycentric"}
        self.assertEqual(_wavelength_frame(cfg, None, issues, knob="x02_wframe"), "barycentric")

    def test_generic_knob_serves_the_stages_without_one(self):
        # C5/C6 (halosub) no tienen knob propio.
        issues = []
        self.assertEqual(_wavelength_frame({"wavelength_frame": "barycentric"}, None, issues), "barycentric")

    def test_resolving_the_qc_frame_is_idempotent_and_only_fills_gaps(self):
        qc = {"cube": {"wavelength_frame": "unknown", "vbary_kms": None}}
        self.assertTrue(resolve_qc_wavelength_frame(qc, {"wavelength_frame": "barycentric"}))
        self.assertEqual(qc["cube"]["wavelength_frame"], "barycentric")
        self.assertFalse(resolve_qc_wavelength_frame(qc, {"wavelength_frame": "topocentric"}))
        self.assertEqual(qc["cube"]["wavelength_frame"], "barycentric")


def _halo_cube(power, sky, *, ny=81, nx=81, nz=60, center=(40, 40)):
    wave = np.linspace(5000.0, 9000.0, nz)
    yy, xx = np.ogrid[:ny, :nx]
    radius = np.hypot(yy - center[0], xx - center[1])
    radius[center[0], center[1]] = 0.5
    image = 1e4 * radius ** (-power) + sky
    cube = np.repeat(image[None, :, :], nz, axis=0).astype(np.float64)
    cube += np.random.default_rng(0).normal(0.0, 0.01, cube.shape)
    return cube, wave, radius


class RadialFloorTests(unittest.TestCase):
    """El diagnostico que atribuye el amarillo de M4: halo AO o cielo residual."""

    def test_it_recovers_a_known_power_law_and_floor(self):
        for power, sky in ((3.0, 2.0), (2.0, 0.0), (1.5, 5.0)):
            cube, wave, radius = _halo_cube(power, sky)
            out = measure_sky_radial_profile(
                cube, wave, radius < 5, np.ones(cube.shape[1:], bool), (40, 40), bin_px=2.0
            )
            self.assertAlmostEqual(out["halo_fit"]["power"], power, delta=0.05)
            self.assertAlmostEqual(out["halo_fit"]["sky"], sky, delta=0.3)

    def test_a_flat_floor_gives_no_radial_slope(self):
        # Cielo residual puro: la potencia debe salir compatible con cero.
        cube, wave, radius = _halo_cube(3.0, 2.0)
        cube[:] = 2.0 + np.random.default_rng(1).normal(0.0, 0.01, cube.shape)
        out = measure_sky_radial_profile(
            cube, wave, radius < 5, np.ones(cube.shape[1:], bool), (40, 40), bin_px=2.0
        )
        self.assertAlmostEqual(out["halo_fit"]["sky"], 2.0, delta=0.05)

    def test_annulus_centers_are_measured_not_geometric(self):
        # En un anillo hay mas pixeles fuera que dentro: usar (lo+hi)/2 sesgaba
        # la pendiente (3.0 verdadera salia 2.72).
        cube, wave, radius = _halo_cube(3.0, 2.0)
        out = measure_sky_radial_profile(
            cube, wave, radius < 5, np.ones(cube.shape[1:], bool), (40, 40), bin_px=2.0
        )
        # El contrato: el centro es la mediana MEDIDA de los radios de los
        # pixeles que contribuyen, no (lo+hi)/2. Es lo que quita el sesgo de la
        # pendiente (3.0 verdadera salia 2.72 con el centro geometrico).
        usable = np.ones(cube.shape[1:], bool) & ~(radius < 5)
        for lo in (10.0, 20.0, 30.0):
            ring = (radius >= lo) & (radius < lo + 2.0) & usable
            index = out["r_centers_px"].index(
                next(c for c in out["r_centers_px"] if lo <= c < lo + 2.0)
            )
            self.assertAlmostEqual(
                out["r_centers_px"][index], float(np.median(radius[ring])), places=9
            )

    def test_it_records_where_annuli_stop_being_complete(self):
        # No se corta ahi —la mascara de A2 tapa todo el interior y cortar
        # dejaria demasiado poco— pero cada anillo declara su cobertura.
        cube, wave, radius = _halo_cube(3.0, 2.0, center=(30, 30))
        out = measure_sky_radial_profile(
            cube, wave, radius < 5, np.ones(cube.shape[1:], bool), (30, 30), bin_px=2.0
        )
        self.assertEqual(out["r_last_complete_annulus_px"], 30.0)
        self.assertGreater(out["r_max_px"], 30.0)
        inside = [c for r, c in zip(out["r_centers_px"], out["azimuthal_coverage_frac"]) if 10 < r < 28]
        outside = [c for r, c in zip(out["r_centers_px"], out["azimuthal_coverage_frac"]) if r > 45]
        self.assertTrue(all(c > 0.9 for c in inside), inside)
        self.assertTrue(all(c < 0.9 for c in outside), outside)

    def test_it_publishes_no_verdict_and_no_semaphore(self):
        # La §4 de la spec A4 prohibe umbrales inventados: se publican numeros.
        cube, wave, radius = _halo_cube(3.0, 2.0)
        out = measure_sky_radial_profile(
            cube, wave, radius < 5, np.ones(cube.shape[1:], bool), (40, 40)
        )
        self.assertNotIn("status", out)
        self.assertNotIn("verdict", out)

    def test_a_mask_from_another_frame_is_refused(self):
        cube, wave, radius = _halo_cube(3.0, 2.0)
        with self.assertRaises(RuntimeError):
            measure_sky_radial_profile(
                cube, wave, np.zeros((10, 10), bool), np.ones(cube.shape[1:], bool), (40, 40)
            )


class PrimaryPositionTests(unittest.TestCase):
    def test_a_declared_position_from_another_frame_is_refused(self):
        # `m3_primary_yx` de ROXs 12 b vale [166,168] y es de un cubo 338x330;
        # A4 mide sobre uno de 200x200 donde la primaria esta en [100,100].
        cube, wave, _ = _halo_cube(3.0, 2.0)
        yx, source = resolve_primary_yx(cube, wave, declared=[70, 70])
        self.assertIsNone(yx)
        self.assertIn("different frame", source)

    def test_without_a_declared_position_it_uses_the_brightness_peak(self):
        cube, wave, _ = _halo_cube(3.0, 2.0)
        yx, source = resolve_primary_yx(cube, wave)
        self.assertEqual(yx, (40.0, 40.0))
        self.assertIn("brightest spaxel", source)

    def test_a_declared_position_on_the_peak_is_kept(self):
        cube, wave, _ = _halo_cube(3.0, 2.0)
        yx, source = resolve_primary_yx(cube, wave, declared=[40.5, 39.5])
        self.assertEqual(yx, (40.5, 39.5))
        self.assertIn("from this cube's brightness peak", source)


if __name__ == "__main__":
    unittest.main()
