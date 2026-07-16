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
    def test_spectra_family_without_input_stops(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = types.SimpleNamespace(input_dir=None)
            with self.assertRaises(SystemExit) as ctx:
                FETCH.fetch_bt_settl(args, {}, Path(tmp))
            self.assertEqual(ctx.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
