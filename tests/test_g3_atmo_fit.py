"""WP-G3R-8: 3-D atmosphere grid fit (Teff, A_V, logg, Omega). Synthetic only."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from musepipe.models.extinction import CCMExtinction
from musepipe.models.fit import fit_grid, fit_grid_3d
from musepipe.models.observed import FitSpectrum
from musepipe.models.prep import prepare_template
from musepipe.models.synthetic import SyntheticGridLibrary

_TEFF = (2600.0, 2800.0, 3000.0)
_LOGG = (3.5, 4.0, 4.5)


def _observed(teff, av, *, scale=3.0, sigma_frac=0.02, err_frac=None, seed=0):
    lib = SyntheticGridLibrary(teff_axis=_TEFF, logg_axis=_LOGG)
    ext = CCMExtinction(rv=3.1, citation="c")
    tmpl = lib.get(teff=teff, logg=4.0)
    wave = np.linspace(6300.0, 9000.0, 120)
    model = prepare_template(tmpl, wave, lsf_fwhm_A=2.383, extinction=ext, av=av, scale=scale)
    rng = np.random.default_rng(seed)
    sigma = sigma_frac * float(np.mean(model))
    flux = model + rng.normal(0.0, sigma, size=model.size)
    err = np.full_like(model, sigma if err_frac is None else err_frac * float(np.mean(model)))
    return FitSpectrum(wave, flux, err, wave.size, float(wave.size), {}), lib, ext


class FitGrid3DTests(unittest.TestCase):
    def test_v1_recovers_teff_av_omega(self):
        fs, lib, ext = _observed(2800.0, 1.0, scale=3.0, seed=1)
        res = fit_grid_3d(fs, lib, ext, teff_axis=_TEFF, logg_axis=_LOGG,
                          av_axis=np.arange(0.0, 3.01, 0.5), lsf_fwhm_A=2.383)
        self.assertEqual(res["teff_best"], 2800.0)
        self.assertLessEqual(abs(res["av_best"] - 1.0), 0.5)
        self.assertLessEqual(abs(res["omega_best"] - 3.0) / 3.0, 0.1)
        self.assertGreater(res["omega_err_sys"], 0.0)  # 10% flux-cal systematic

    def test_logg_not_constrained(self):
        # SyntheticGridLibrary's logg only rescales flux -> absorbed by Omega
        fs, lib, ext = _observed(2800.0, 0.5, seed=2)
        res = fit_grid_3d(fs, lib, ext, teff_axis=_TEFF, logg_axis=_LOGG,
                          av_axis=np.arange(0.0, 3.01, 0.5), lsf_fwhm_A=2.383)
        self.assertEqual(res["logg_interval"], "not_constrained")

    def test_maps_shapes(self):
        fs, lib, ext = _observed(2800.0, 1.0, seed=3)
        av = np.arange(0.0, 3.01, 0.5)
        res = fit_grid_3d(fs, lib, ext, teff_axis=_TEFF, logg_axis=_LOGG,
                          av_axis=av, lsf_fwhm_A=2.383)
        self.assertEqual(res["dchi2_teff_av"].shape, (len(_TEFF), av.size))
        self.assertEqual(res["dchi2_teff_logg"].shape, (len(_TEFF), len(_LOGG)))
        self.assertEqual(res["dchi2_3d"].shape, (len(_TEFF), len(_LOGG), av.size))

    def test_edge_touch_at_boundary(self):
        fs, lib, ext = _observed(3000.0, 0.0, seed=4)  # best Teff at axis edge
        res = fit_grid_3d(fs, lib, ext, teff_axis=_TEFF, logg_axis=_LOGG,
                          av_axis=np.arange(0.0, 3.01, 0.5), lsf_fwhm_A=2.383)
        self.assertTrue(res["edge_touch"]["teff"])

    def test_inflation_applied_when_chi2red_high(self):
        # true noise sigma but errors under-reported -> chi2_red > 1.5
        fs, lib, ext = _observed(2800.0, 1.0, sigma_frac=0.02, err_frac=0.012, seed=5)
        res = fit_grid_3d(fs, lib, ext, teff_axis=_TEFF, logg_axis=_LOGG,
                          av_axis=np.arange(0.0, 3.01, 0.5), lsf_fwhm_A=2.383)
        self.assertTrue(res["inflate"]["applied"])
        self.assertGreater(res["inflate"]["factor"], 1.0)

    def test_fit_grid_2d_unchanged(self):
        # the 2-D fit_grid still returns its documented keys (behaviour intact)
        fs, lib, ext = _observed(2800.0, 1.0, seed=6)
        res = fit_grid(fs.wave_bin, fs.flux_bin, fs.err_bin, lib, ext,
                       teff_axis=_TEFF, av_axis=np.arange(0.0, 3.01, 0.5),
                       lsf_fwhm_A=2.383, logg=4.0)
        self.assertIn("teff_interval", res)
        self.assertIn("dchi2_map", res)


class StageAtmoFitTests(unittest.TestCase):
    def test_stage_assembly_and_write(self):
        from musepipe.stages.stage_g3_atmo_fit import (
            compute_stage_g3_atmo_fit, stage_g3_atmo_fit_paths,
            write_stage_g3_atmo_fit)
        with tempfile.TemporaryDirectory() as tmp:
            fs, lib, _ = _observed(2800.0, 1.0, seed=7)
            cfg = {"h01_lsf_fwhm_A": 2.383, "h03_extinction_law_citation": "c",
                   "g3_atmo_teff_axis_k": [2600.0, 3000.0, 200.0],
                   "g3_atmo_logg_axis": [3.5, 4.5, 0.5],
                   "g3_atmo_av_axis": [0.0, 3.0, 0.5],
                   "g3_veiling_variant": False, "g3_atmosphere_family": "synthetic",
                   "run_id": "syn"}
            paths = stage_g3_atmo_fit_paths("syn", project_root=tmp)
            rows, qc, primary = compute_stage_g3_atmo_fit(
                cfg, paths, fit_spec=fs, library=lib)
            props = {r["property"] for r in rows}
            self.assertEqual(props, {"teff", "a_v_spectral", "logg", "omega_scale"})
            logg_row = next(r for r in rows if r["property"] == "logg")
            self.assertEqual(logg_row["label"], "not_constrained")
            written = write_stage_g3_atmo_fit(rows, qc, primary, paths)
            self.assertTrue(Path(written["npz"]).exists())
            self.assertTrue(Path(written["plot_teff_av"]).exists())
            with np.load(written["npz"]) as z:
                self.assertIn("dchi2_teff_av", z.files)


if __name__ == "__main__":
    unittest.main()
