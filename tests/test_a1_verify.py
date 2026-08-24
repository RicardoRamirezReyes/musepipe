"""El envoltorio de A1: fases por evidencia, V3 con tolerancia declarada y el
frame de la máscara de cielo, que no es el del cubo."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from astropy.io import fits

from musepipe.qc.cube_qc import aggregate_status
from musepipe.reduction.a1_verify import (
    _cube_to_mask_frame,
    _v3,
    contributing_nights,
    gates_from_evidence,
)
from musepipe.reduction.sky_zap import verification_verdict


def _cube_header(crval3=4749.533203125):
    header = fits.Header()
    header["CRVAL1"] = 247.81
    header["CRVAL2"] = -24.54
    header["CRPIX1"] = 219.4
    header["CRPIX2"] = 209.0
    header["CDELT1"] = -7.0e-06
    header["CDELT2"] = 7.0e-06
    header["CTYPE1"] = "RA---TAN"
    header["CTYPE2"] = "DEC--TAN"
    header["CRVAL3"] = crval3
    header["CRPIX3"] = 1.0
    header["CD3_3"] = 1.25
    return header


def _write_cube(path, crval3=4749.533203125, n=6):
    data = np.ones((n, 5, 5), dtype=np.float32)
    hdus = [fits.PrimaryHDU(),
            fits.ImageHDU(data=data, header=_cube_header(crval3), name="DATA"),
            fits.ImageHDU(data=data, name="STAT")]
    fits.HDUList(hdus).writeto(path)


class VerificationVerdictTests(unittest.TestCase):
    """La puerta de A2 tiene que leer un veredicto, no una verdad de Python."""

    def test_dict_form_uses_ok(self):
        self.assertTrue(verification_verdict({"ok": True, "value": 0.97}))
        self.assertFalse(verification_verdict({"ok": False, "value": 0.94}))

    def test_a_failed_check_is_not_true_just_because_the_dict_is_not_empty(self):
        # Es el fallo que tenia la puerta: `bool({...})` es True siempre.
        self.assertFalse(verification_verdict({"ok": False, "message": "psf_matched_corr=0.9483"}))

    def test_unavailable_does_not_satisfy_a_required_check(self):
        self.assertFalse(verification_verdict({"ok": None, "status": "unavailable"}))
        self.assertFalse(verification_verdict(None))

    def test_bool_form_is_respected(self):
        self.assertTrue(verification_verdict(True))
        self.assertFalse(verification_verdict(False))

    def test_legacy_scalar_is_still_accepted(self):
        # Las reducciones ya validadas publican un numero; se acepta para no
        # romperlas, y quien lo hace queda anotado en `a1_scalar_verdicts`.
        self.assertTrue(verification_verdict(0.9541))


class MaskFrameTests(unittest.TestCase):
    """La `SKY_MASK` vive en offsets de pixel, no en el frame del cubo."""

    def _mask_header(self):
        header = fits.Header()
        header["CTYPE1"] = "PIXEL"
        header["CTYPE2"] = "PIXEL"
        header["CRPIX1"] = 1.0
        header["CRPIX2"] = 1.0
        header["CRVAL1"] = -159.070297241211
        header["CRVAL2"] = -151.345260620117
        header["CD1_1"] = 1.0
        header["CD2_2"] = 1.0
        return header

    def test_reference_pixel_maps_to_the_mask_origin_offset(self):
        cube = _cube_header()
        mask = self._mask_header()
        # El pixel de referencia del cubo es el offset (0, 0) del campo, que en
        # la mascara cae en -CRVAL.
        j, i = _cube_to_mask_frame(cube, mask, (cube["CRPIX2"] - 1, cube["CRPIX1"] - 1))
        self.assertAlmostEqual(i, -mask["CRVAL1"], places=6)
        self.assertAlmostEqual(j, -mask["CRVAL2"], places=6)

    def test_the_transform_preserves_offsets(self):
        cube = _cube_header()
        mask = self._mask_header()
        base = _cube_to_mask_frame(cube, mask, (100.0, 100.0))
        moved = _cube_to_mask_frame(cube, mask, (110.0, 105.0))
        self.assertAlmostEqual(moved[0] - base[0], 10.0, places=6)
        self.assertAlmostEqual(moved[1] - base[1], 5.0, places=6)

    def test_using_cube_coordinates_directly_is_a_different_place(self):
        # Sin transformar, los centroides del cubo caen dentro del array de la
        # mascara y NO dan error: miden otra cosa en silencio.
        cube = _cube_header()
        j, i = _cube_to_mask_frame(cube, self._mask_header(), (224.15, 217.82))
        self.assertGreater(abs(j - 224.15) + abs(i - 217.82), 50.0)


class V3ToleranceTests(unittest.TestCase):
    """V3 usa la dispersion que el propio combine declara."""

    def test_delta_inside_the_declared_spread_passes(self):
        with tempfile.TemporaryDirectory() as tmp:
            cube = Path(tmp) / "cube.fits"
            adp = Path(tmp) / "adp.fits"
            _write_cube(cube, crval3=4749.533203125)
            _write_cube(adp, crval3=4749.5322265625)      # 9.8e-4 A
            combine = {"wavelength": {"crval3_spread_channels": 0.003125, "cd3_3": 1.25}}
            out = _v3(cube, {"2022-08-28": adp}, combine)
            self.assertTrue(out["ok"])
            self.assertAlmostEqual(out["v3_crval3_tolerance_A"], 0.00390625, places=8)

    def test_a_delta_beyond_the_declared_spread_still_fails(self):
        # La tolerancia se deriva, no se afloja: un desfase real sigue fallando.
        with tempfile.TemporaryDirectory() as tmp:
            cube = Path(tmp) / "cube.fits"
            adp = Path(tmp) / "adp.fits"
            _write_cube(cube, crval3=4749.533203125)
            _write_cube(adp, crval3=4749.6)               # 0.067 A
            combine = {"wavelength": {"crval3_spread_channels": 0.003125, "cd3_3": 1.25}}
            self.assertFalse(_v3(cube, {"2022-08-28": adp}, combine)["ok"])

    def test_a_combine_that_declares_no_spread_keeps_the_strict_tolerance(self):
        with tempfile.TemporaryDirectory() as tmp:
            cube = Path(tmp) / "cube.fits"
            adp = Path(tmp) / "adp.fits"
            _write_cube(cube, crval3=4749.533203125)
            _write_cube(adp, crval3=4749.5322265625)
            out = _v3(cube, {"n": adp}, {"wavelength": {}})
            self.assertFalse(out["ok"])
            self.assertEqual(out["v3_crval3_tolerance_A"], 1e-6)


class GatesFromEvidenceTests(unittest.TestCase):
    """Una fase sin su producto en disco NO se escribe."""

    def _combine(self, tmp, files):
        return {"output": str(Path(tmp) / "cube.fits"), "n_exposures": len(files),
                "exposures": [{"exposure_id": f"2022-08-28_x{i}", "file": str(f)}
                              for i, f in enumerate(files)]}

    def test_no_evidence_no_gates(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "runs" / "R" / "stages").mkdir(parents=True)
            gates, evidence = gates_from_evidence("R", root, self._combine(tmp, []), root / "nope.fits")
            self.assertEqual(gates, [])
            self.assertEqual(evidence, {})

    def test_each_gate_names_the_file_that_backs_it(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            stages = root / "runs" / "R" / "stages"
            stages.mkdir(parents=True)
            (stages / "stage00r_p0_calibration_audit.json").write_text(
                json.dumps({"groups": {"2022-08-28": {}}}), encoding="utf-8")
            exposure = root / "exp.fits"
            exposure.write_text("x", encoding="utf-8")
            cube = root / "cube.fits"
            cube.write_text("x", encoding="utf-8")
            combine = self._combine(tmp, [exposure])
            combine["output"] = str(cube)
            gates, evidence = gates_from_evidence("R", root, combine, cube)
            self.assertEqual(gates, ["fase0", "fase1", "fase2", "fase3"])
            self.assertIn("stage00r_p0_calibration_audit.json", evidence["fase0"])
            self.assertIn("1/1", evidence["fase1"])

    def test_a_missing_per_exposure_cube_drops_the_reduction_gates(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "runs" / "R" / "stages").mkdir(parents=True)
            cube = root / "cube.fits"
            cube.write_text("x", encoding="utf-8")
            combine = self._combine(tmp, [root / "ausente.fits"])
            combine["output"] = str(cube)
            gates, _ = gates_from_evidence("R", root, combine, cube)
            self.assertNotIn("fase1", gates)
            self.assertIn("fase3", gates)


class ContributingNightsTests(unittest.TestCase):
    def test_counts_exposures_per_night(self):
        payload = {"exposures": [{"exposure_id": "2022-08-28_a"}, {"exposure_id": "2022-08-28_b"},
                                 {"exposure_id": "2022-08-30_c"}]}
        self.assertEqual(contributing_nights(payload), {"2022-08-28": 2, "2022-08-30": 1})


class AggregateStatusTests(unittest.TestCase):
    """El estado global de A4, que hasta ahora nadie calculaba."""

    def _qc(self, **states):
        return {f"m{i}_{name}": {"status": states.get(f"m{i}", "green")}
                for i, name in enumerate(("wavelength", "lsf", "flux", "sky", "stat"), start=1)}

    def test_red_wins_and_says_who(self):
        out = aggregate_status(self._qc(m2="yellow", m4="yellow", m5="red"))
        self.assertEqual(out["status"], "red")
        self.assertEqual(out["driven_by"], "m5_stat")

    def test_yellow_when_nothing_is_red(self):
        self.assertEqual(aggregate_status(self._qc(m2="yellow"))["status"], "yellow")

    def test_all_green_is_green(self):
        self.assertEqual(aggregate_status(self._qc())["status"], "green")

    def test_unmeasured_is_incomplete_not_green(self):
        out = aggregate_status(self._qc(m4="unavailable", m5="unavailable"))
        self.assertEqual(out["status"], "incomplete")
        self.assertEqual(out["unavailable"], ["m4_sky", "m5_stat"])

    def test_unmeasured_does_not_hide_a_red(self):
        out = aggregate_status(self._qc(m4="unavailable", m5="red"))
        self.assertEqual(out["status"], "red")


if __name__ == "__main__":
    unittest.main()
