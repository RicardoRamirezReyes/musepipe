"""Contrato de la unidad de flujo: BUNIT viaja con el dato (plan A, 2026-07-25).

Antes, `BUNIT` se perdia en el stack de B1/B2, los productos espectrales salian
con `BUNIT=''` y la escala fisica dependia de `h03_flux_unit_cgs`, cuyo default
divergia por modulo (1.0 en H03/G3, 1e-20 en models.derived): un run sin la
clave daba L/Mdot con un factor 1e20 segun quien leyera.
"""
import tempfile
import unittest
from pathlib import Path

import numpy as np
from astropy.io import fits

from musepipe.io import (
    MUSE_NATIVE_BUNIT,
    bunit_to_cgs_scale,
    cube_bunit,
    flux_unit_cgs,
    resolve_bunit,
)


def _cube(path, bunit=None, ext=1):
    hdus = [fits.PrimaryHDU()]
    hdu = fits.ImageHDU(data=np.zeros((2, 2, 2), dtype=np.float32), name="DATA")
    if bunit is not None:
        hdu.header["BUNIT"] = bunit
    hdus.append(hdu)
    fits.HDUList(hdus).writeto(path, overwrite=True)
    return path


class BunitParsingTests(unittest.TestCase):
    def test_muse_native_bunit_is_1e_minus_20_cgs(self):
        self.assertAlmostEqual(bunit_to_cgs_scale(MUSE_NATIVE_BUNIT) / 1e-20, 1.0, places=9)

    def test_plain_cgs_is_unity(self):
        self.assertAlmostEqual(bunit_to_cgs_scale("erg/s/cm**2/Angstrom"), 1.0, places=12)

    def test_unparseable_or_wrong_dimension_is_none(self):
        for value in ("", None, "counts", "erg/s/cm**2"):
            self.assertIsNone(bunit_to_cgs_scale(value), msg=repr(value))


class CubeBunitTests(unittest.TestCase):
    def test_finds_bunit_in_any_hdu(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _cube(Path(tmp) / "c.fits", bunit=MUSE_NATIVE_BUNIT)
            self.assertEqual(cube_bunit(path), MUSE_NATIVE_BUNIT)
            self.assertEqual(cube_bunit(path, ext=1), MUSE_NATIVE_BUNIT)

    def test_absent_bunit_is_empty_not_invented(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _cube(Path(tmp) / "c.fits", bunit=None)
            self.assertEqual(cube_bunit(path), "")


class ResolveBunitTests(unittest.TestCase):
    def test_stack_wins_over_input_cube(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = {"cube_files": [str(_cube(Path(tmp) / "in.fits", bunit="erg/s/cm**2/Angstrom"))]}
            self.assertEqual(resolve_bunit(cfg, stack_bunit=MUSE_NATIVE_BUNIT), MUSE_NATIVE_BUNIT)

    def test_falls_back_to_input_cube_when_stack_lost_it(self):
        # Es el caso de los runs ya reducidos: su stack se escribio antes de que
        # B1/B2 propagaran BUNIT, y re-correrlos solo por la cabecera cuesta ~20 min.
        with tempfile.TemporaryDirectory() as tmp:
            cfg = {"cube_files": [str(_cube(Path(tmp) / "in.fits", bunit=MUSE_NATIVE_BUNIT))]}
            self.assertEqual(resolve_bunit(cfg, stack_bunit=None), MUSE_NATIVE_BUNIT)

    def test_config_override_is_last_resort(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = {"cube_files": [str(_cube(Path(tmp) / "in.fits", bunit=None))],
                   "x01_bunit": "erg/s/cm**2/Angstrom"}
            self.assertEqual(resolve_bunit(cfg, override_key="x01_bunit"), "erg/s/cm**2/Angstrom")

    def test_unknown_stays_empty(self):
        self.assertEqual(resolve_bunit({}, stack_bunit=None), "")


class FluxUnitCgsTests(unittest.TestCase):
    def test_config_knob_wins_so_frozen_results_do_not_move(self):
        value = flux_unit_cgs({"h03_flux_unit_cgs": 1e-20}, bunit="erg/s/cm**2/Angstrom")
        self.assertEqual(value, 1e-20)

    def test_falls_back_to_product_bunit(self):
        self.assertAlmostEqual(flux_unit_cgs({}, bunit=MUSE_NATIVE_BUNIT) / 1e-20, 1.0, places=9)

    def test_raises_instead_of_a_silent_default(self):
        with self.assertRaises(ValueError) as ctx:
            flux_unit_cgs({}, bunit=None)
        self.assertIn("h03_flux_unit_cgs", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
