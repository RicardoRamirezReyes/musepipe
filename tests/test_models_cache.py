"""WP-G3R-2: frozen internal cache format, synthetic fixtures only."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from musepipe.models.cache import (
    TRACK_ARRAYS,
    load_spectrum_npz,
    load_tracks_npz,
    write_spectrum_npz,
    write_tracks_npz,
)


class SpectrumCacheTests(unittest.TestCase):
    def test_roundtrip_and_trim(self):
        with tempfile.TemporaryDirectory() as tmp:
            wave = np.linspace(3000.0, 11000.0, 801)  # spans beyond the trim window
            flux = np.ones_like(wave)
            meta = {"teff": 3000.0, "logg": 4.0, "source": "unit-test"}
            out = write_spectrum_npz(Path(tmp) / "s.npz", wave, flux, meta)
            spec = load_spectrum_npz(out)
            self.assertGreaterEqual(spec.wave_A.min(), 4000.0)
            self.assertLessEqual(spec.wave_A.max(), 10000.0)
            self.assertEqual(spec.wave_A.shape, spec.flux.shape)
            self.assertEqual(spec.meta["teff"], 3000.0)

    def test_sorts_ascending(self):
        with tempfile.TemporaryDirectory() as tmp:
            wave = np.array([9000.0, 5000.0, 7000.0])
            flux = np.array([3.0, 1.0, 2.0])
            out = write_spectrum_npz(Path(tmp) / "s.npz", wave, flux, {"source": "t"})
            spec = load_spectrum_npz(out)
            self.assertTrue(np.all(np.diff(spec.wave_A) > 0))
            np.testing.assert_allclose(spec.flux, [1.0, 2.0, 3.0])

    def test_shape_mismatch_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(RuntimeError):
                write_spectrum_npz(Path(tmp) / "s.npz", [1, 2, 3], [1, 2], {})

    def test_out_of_window_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            wave = np.linspace(1000.0, 3000.0, 50)
            with self.assertRaises(RuntimeError):
                write_spectrum_npz(Path(tmp) / "s.npz", wave, np.ones_like(wave),
                                   {"source": "t"})

    def test_meta_must_be_dict(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(RuntimeError):
                write_spectrum_npz(Path(tmp) / "s.npz", [5000.0], [1.0], "notadict")


class TrackCacheTests(unittest.TestCase):
    def _arrays(self, n=5):
        return {k: np.linspace(1.0, 2.0, n) for k in TRACK_ARRAYS}

    def test_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            arrays = self._arrays()
            out = write_tracks_npz(Path(tmp) / "t.npz", arrays, {"family": "unit"})
            got = load_tracks_npz(out)
            for k in TRACK_ARRAYS:
                np.testing.assert_allclose(got[k], arrays[k])
            self.assertEqual(got["meta"]["family"], "unit")

    def test_missing_array_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            arrays = self._arrays()
            del arrays["logg"]
            with self.assertRaises(RuntimeError) as ctx:
                write_tracks_npz(Path(tmp) / "t.npz", arrays, {})
            self.assertIn("logg", str(ctx.exception))

    def test_unequal_length_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            arrays = self._arrays()
            arrays["teff_k"] = np.linspace(1.0, 2.0, 4)
            with self.assertRaises(RuntimeError):
                write_tracks_npz(Path(tmp) / "t.npz", arrays, {})


if __name__ == "__main__":
    unittest.main()
