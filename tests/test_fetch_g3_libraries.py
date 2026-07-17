"""WP-G3R-2: fetch-script parsers/converters, synthetic input only (no network)."""

import importlib.util
import tempfile
import types
import unittest
from pathlib import Path

import numpy as np

from musepipe.models.cache import load_spectrum_npz

_ROOT = Path(__file__).resolve().parents[1]


def _load_script():
    spec = importlib.util.spec_from_file_location(
        "fetch_g3_libraries", _ROOT / "scripts" / "fetch_g3_libraries.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


FETCH = _load_script()

_ISO_TEXT = """BHAC15 models
!  t (Gyr) =   0.0050
!-----------------------------------------------------------------
! M/Ms  Teff     L/Ls   g    R/Rs   Li/Li0    Mj
 0.010  2400.   -2.00  3.40  0.340    1.000   8.80
 0.020  2600.   -1.50  3.40  0.470    1.000   7.70
!-----------------------------------------------------------------

!  t (Gyr) =   0.0100
!-----------------------------------------------------------------
! M/Ms  Teff     L/Ls   g    R/Rs   Li/Li0    Mj
 0.010  2350.   -2.10  3.45  0.320    1.000   8.90
"""


class BHAC15ParserTests(unittest.TestCase):
    def test_parses_ages_and_luminosity(self):
        arrays = FETCH.parse_bhac15_iso(_ISO_TEXT)
        self.assertEqual(arrays["mass_msun"].size, 3)
        np.testing.assert_allclose(sorted(set(arrays["age_gyr"])), [0.005, 0.010])
        # l_bol is 10**log(L/Ls); first row logL=-2.00 -> 0.01 Lsun
        idx = np.argmin(np.abs(arrays["l_bol_lsun"] - 0.01))
        self.assertAlmostEqual(arrays["l_bol_lsun"][idx], 0.01, places=6)
        self.assertEqual(arrays["teff_k"][0], 2400.0)

    def test_data_before_age_header_raises(self):
        with self.assertRaises(RuntimeError):
            FETCH.parse_bhac15_iso(" 0.010 2400. -2.0 3.4 0.34\n")


class IntermediateSpectraTests(unittest.TestCase):
    def test_convert_two_column_ascii(self):
        with tempfile.TemporaryDirectory() as tmp:
            indir = Path(tmp) / "in"
            outdir = Path(tmp) / "out"
            indir.mkdir()
            outdir.mkdir()
            wave = np.linspace(5000.0, 9000.0, 100)
            flux = np.ones_like(wave)
            np.savetxt(indir / "m5.txt", np.column_stack([wave, flux]))
            (indir / "index.csv").write_text(
                "spt,file,wave_frame\nM5,m5.txt,air\n", encoding="utf-8")
            rels = FETCH.convert_spectra_from_intermediate(
                indir, outdir, key_cols=("spt",),
                provenance_base={"family": "templates_young", "citation": "X"})
            self.assertEqual(rels, ["M5.npz"])
            spec = load_spectrum_npz(outdir / "M5.npz")
            self.assertEqual(spec.meta["spt"], "M5")
            self.assertEqual(spec.meta["wave_frame"], "air")
            self.assertEqual(spec.meta["citation"], "X")

    def test_missing_index_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(RuntimeError):
                FETCH.convert_spectra_from_intermediate(
                    Path(tmp), Path(tmp), key_cols=("spt",), provenance_base={})


class InstructStopTests(unittest.TestCase):
    def test_manual_family_without_input_stops(self):
        # ATMO2020 has no direct source; without --input-dir it must STOP with
        # exit 2 and touch no network (tests never hit the wire).
        with tempfile.TemporaryDirectory() as tmp:
            args = types.SimpleNamespace(input_dir=None)
            with self.assertRaises(SystemExit) as ctx:
                FETCH.fetch_tracks_atmo2020(args, {}, Path(tmp))
            self.assertEqual(ctx.exception.code, 2)


class WavelengthUnitTests(unittest.TestCase):
    def test_nm_axis_scaled_to_angstrom(self):
        wave_A, unit = FETCH.to_angstrom(np.array([545.0, 1035.0]))
        self.assertEqual(unit, "nm")
        np.testing.assert_allclose(wave_A, [5450.0, 10350.0])

    def test_angstrom_axis_kept(self):
        wave_A, unit = FETCH.to_angstrom(np.array([3650.0, 10200.0]))
        self.assertEqual(unit, "angstrom")
        np.testing.assert_allclose(wave_A, [3650.0, 10200.0])

    def test_absurd_range_raises(self):
        with self.assertRaises(RuntimeError):
            FETCH.to_angstrom(np.array([50.0, 90.0]))

    def test_linear_wcs_axis(self):
        header = {"NAXIS1": 3, "CRVAL1": 100.0, "CDELT1": 2.0, "CRPIX1": 1.0}
        np.testing.assert_allclose(
            FETCH.wave_from_linear_wcs(header), [100.0, 102.0, 104.0])


class KesseliSelectionTests(unittest.TestCase):
    def test_selection_rules(self):
        self.assertEqual(FETCH._kesseli_wanted("M0_+0.0_Dwarf.fits"), "M0")
        self.assertEqual(FETCH._kesseli_wanted("A0.fits"), "A0")
        self.assertEqual(FETCH._kesseli_wanted("M4.5_+0.0_Dwarf.fits"), "M4.5")
        self.assertIsNone(FETCH._kesseli_wanted("A0_Giant.fits"))
        self.assertIsNone(FETCH._kesseli_wanted("M2_-1.0_Dwarf.fits"))
        self.assertIsNone(FETCH._kesseli_wanted("table2.dat"))


class FitsReaderTests(unittest.TestCase):
    def test_read_kesseli_fits(self):
        from astropy.io import fits
        with tempfile.TemporaryDirectory() as tmp:
            loglam = np.log10(np.linspace(4000.0, 9000.0, 50))
            flux = np.linspace(0.5, 1.0, 50)
            cols = fits.ColDefs([
                fits.Column(name="LogLam", format="E", array=loglam),
                fits.Column(name="Flux", format="E", array=flux),
            ])
            hdul = fits.HDUList([fits.PrimaryHDU(), fits.BinTableHDU.from_columns(cols)])
            path = Path(tmp) / "k.fits"
            hdul.writeto(path)
            wave, fl = FETCH.read_kesseli_fits(path)
            np.testing.assert_allclose(wave[0], 4000.0, rtol=1e-5)
            np.testing.assert_allclose(fl[-1], 1.0, rtol=1e-5)

    def test_read_manara_visual_fits_nm(self):
        from astropy.io import fits
        with tempfile.TemporaryDirectory() as tmp:
            flux = np.ones(100)
            hdu = fits.PrimaryHDU(data=flux.astype("float32"))
            hdu.header["CRVAL1"] = 545.0
            hdu.header["CDELT1"] = 1.0
            hdu.header["CRPIX1"] = 1.0
            path = Path(tmp) / "m.fit"
            hdu.writeto(path)
            wave_A, fl, unit = FETCH.read_manara_visual_fits(path)
            self.assertEqual(unit, "nm")
            np.testing.assert_allclose(wave_A[0], 5450.0)  # 545 nm -> 5450 A
            self.assertEqual(fl.size, 100)


class BTSettlNodeTests(unittest.TestCase):
    def test_selection_filters_metallicity_and_box(self):
        rows = [
            {"teff": 3000.0, "logg": 4.0, "meta": 0.0, "alpha": 0.0,
             "Access.Reference": "u1"},
            {"teff": 3000.0, "logg": 4.0, "meta": -0.5, "alpha": 0.0,
             "Access.Reference": "u2"},   # non-solar -> excluded
            {"teff": 1500.0, "logg": 4.0, "meta": 0.0, "alpha": 0.0,
             "Access.Reference": "u3"},   # below teff range -> excluded
            {"teff": 2500.0, "logg": 5.0, "meta": 0.0, "alpha": 0.0,
             "Access.Reference": "u4"},
        ]
        nodes = FETCH.select_btsettl_nodes(
            rows, teff_range=(2000.0, 4500.0), logg_range=(3.5, 5.5))
        self.assertEqual([n[2] for n in nodes], ["u4", "u1"])  # sorted by teff

    def test_read_svo_spectrum_votable(self):
        from astropy.table import Table
        with tempfile.TemporaryDirectory() as tmp:
            wave = np.linspace(4000.0, 10000.0, 20)
            flux = np.linspace(1.0, 2.0, 20)
            t = Table([wave, flux], names=("WAVELENGTH", "FLUX"))
            path = Path(tmp) / "svo.xml"
            t.write(path, format="votable")
            w, f = FETCH.read_svo_spectrum_votable(path)
            np.testing.assert_allclose(w, wave)
            np.testing.assert_allclose(f, flux)


if __name__ == "__main__":
    unittest.main()
