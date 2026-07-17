"""The single gateway of real observed data into the G3 fits (plan WP-G3R-6).

Loads the canonical spectrum, builds the frozen fit masks (D9) and rebins for
the fit (D8) with honest, correlation-inflated errors from the G1 covariance.
Everything the fitting stages consume passes through :func:`fit_spectrum`.

Anti-bias (plan §0.5.7): this module is only ever run on the real spectrum in
WP-G3R-11; here it is exercised solely on synthetic fixtures.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..constants import C_KMS

SPECTRUM_FILE = "spec_final_object.fits"
SPECTRUM_EXT = "SPECTRUM"
BAD_MASK_FILE = "stage04b_bad_wavelength_mask.npy"
G1_COV_FILE = "g1_channel_covariance.npz"
G2_TABLE = "g2_line_measurements.csv"


@dataclass(frozen=True)
class FitSpectrum:
    """Frozen fit-ready spectrum — the only real-data input to the fits."""

    wave_bin: np.ndarray
    flux_bin: np.ndarray
    err_bin: np.ndarray
    n_bins: int
    n_eff: float
    mask_provenance: dict


def load_final_spectrum(run_paths, *, err_column="flux_err_total") -> dict:
    """Read ``spec_final_object.fits`` (§0.2). RuntimeError on a missing column."""
    from astropy.io import fits

    path = Path(run_paths.stage_dir) / SPECTRUM_FILE
    if not path.exists():
        raise RuntimeError(f"missing canonical spectrum: {path}")
    with fits.open(path) as hdul:
        if SPECTRUM_EXT not in [h.name for h in hdul]:
            raise RuntimeError(f"{path} has no {SPECTRUM_EXT} extension")
        table = hdul[SPECTRUM_EXT].data
        names = list(table.columns.names)
        required = ("wave_A", "flux", err_column, "sys_fluxcal")
        missing = [c for c in required if c not in names]
        if missing:
            raise RuntimeError(f"{path} missing columns {missing} (has {names})")
        flux_unit = None
        for col in table.columns:
            if col.name == "flux":
                flux_unit = col.unit
        out = {
            "wave_A": np.asarray(table["wave_A"], dtype=np.float64),
            "flux": np.asarray(table["flux"], dtype=np.float64),
            "flux_err": np.asarray(table[err_column], dtype=np.float64),
            "sys_fluxcal": np.asarray(table["sys_fluxcal"], dtype=np.float64),
            "err_column": err_column,
            "flux_unit": flux_unit,
        }
    out["n_channels"] = out["wave_A"].size
    return out


def _g2_rest_wavelengths(table_dir) -> list[float]:
    path = Path(table_dir) / G2_TABLE
    if not path.exists():
        raise RuntimeError(f"missing G2 line table: {path}")
    with path.open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    return [float(r["rest_A"]) for r in rows]


def build_fit_masks(cfg, wave, run_paths) -> tuple[np.ndarray, dict]:
    """Frozen D9 fit mask (True = EXCLUDE): bad channels + ±line_window_kms
    around every G2 line + telluric bands. Returns ``(mask, provenance)``."""
    wave = np.asarray(wave, dtype=np.float64)
    n = wave.size
    mask = np.zeros(n, dtype=bool)
    fm = dict(cfg.get("g3_fit_masks", {}))
    prov: dict = {}

    if fm.get("use_stage04b_bad_mask", True):
        bad = np.load(Path(run_paths.stage_dir) / BAD_MASK_FILE).astype(bool)
        if bad.shape != (n,):
            raise RuntimeError(
                f"bad mask shape {bad.shape} != spectrum ({n},) — runs "
                "desalineados (plan WP-G3R-6 PARADA)")
        mask |= bad
        prov["bad_mask"] = {"file": BAD_MASK_FILE, "semantics": "True=excluded",
                            "n_excluded": int(bad.sum())}

    kms = float(fm.get("line_window_kms", 300.0))
    lines = _g2_rest_wavelengths(run_paths.table_dir)
    before = int(mask.sum())
    for rest in lines:
        half = rest * kms / C_KMS
        mask |= (wave >= rest - half) & (wave <= rest + half)
    prov["line_windows"] = {"n_lines": len(lines), "window_kms": kms,
                            "n_added": int(mask.sum()) - before}

    bands = fm.get("telluric_bands_A", [])
    before = int(mask.sum())
    for lo, hi in bands:
        mask |= (wave >= float(lo)) & (wave <= float(hi))
    prov["telluric_bands"] = {"bands_A": [list(b) for b in bands],
                              "n_added": int(mask.sum()) - before}
    prov["n_excluded_total"] = int(mask.sum())
    return mask, prov


def _block_upper_waves(wave, bad_mask, block_bounds) -> np.ndarray:
    """Upper wavelength of each covariance block. block_bounds index the
    GOOD-channel subspace (verified 2026-07-16), so map through the bad mask."""
    block_bounds = np.asarray(block_bounds)
    if bad_mask is None:  # fallback: treat bounds as full-array indices
        hi = np.minimum(block_bounds[:, 1] - 1, wave.size - 1)
        return wave[hi]
    good = ~np.asarray(bad_mask, dtype=bool)
    n_good = int(good.sum())
    if int(block_bounds[-1, 1]) != n_good:
        raise RuntimeError(
            f"covariance coverage {int(block_bounds[-1, 1])} != n_good {n_good} "
            "— espectro/bad-mask/G1 desalineados (plan WP-G3R-6 PARADA)")
    wave_good = wave[good]
    hi = np.minimum(block_bounds[:, 1], n_good) - 1
    return wave_good[hi]


def rebin_for_fit(wave, flux, err, mask, cov, *, n_channels, wave_range,
                  bad_mask=None, max_masked_frac=0.5) -> dict:
    """Rebin per D8: groups of ``n_channels`` within ``wave_range``, drop bins
    with >``max_masked_frac`` masked, variance = (Σσ²)/N² × (n/n_eff) per the
    G1 block containing the bin. Returns wave/flux/err bins + n_bins + n_eff_total."""
    wave = np.asarray(wave, float)
    flux = np.asarray(flux, float)
    err = np.asarray(err, float)
    mask = np.asarray(mask, dtype=bool)
    n = wave.size
    if not (flux.shape == err.shape == mask.shape == (n,)):
        raise RuntimeError(
            f"shape mismatch: wave{wave.shape} flux{flux.shape} err{err.shape} "
            f"mask{mask.shape} (plan WP-G3R-6 PARADA)")

    block_bounds = np.asarray(cov["block_bounds"])
    neff_over_n = np.asarray(cov["n_eff_over_n_by_block"], dtype=float)
    block_hi_wave = _block_upper_waves(wave, bad_mask, block_bounds)

    lo, hi = float(wave_range[0]), float(wave_range[1])
    in_range = np.nonzero((wave >= lo) & (wave <= hi))[0]
    if in_range.size == 0:
        raise RuntimeError(f"no channels within wave_range {wave_range}")

    wave_bin, flux_bin, err_bin, blocks_used = [], [], [], []
    n_eff_total = 0.0
    for start in range(0, in_range.size, int(n_channels)):
        chans = in_range[start:start + int(n_channels)]
        finite = np.isfinite(flux[chans]) & np.isfinite(err[chans]) & (err[chans] > 0)
        keep = (~mask[chans]) & finite
        n_tot = chans.size
        n_keep = int(keep.sum())
        if n_keep == 0 or (n_tot - n_keep) / n_tot > max_masked_frac:
            continue
        w = wave[chans][keep]
        f = flux[chans][keep]
        e = err[chans][keep]
        wb = float(w.mean())
        k = int(np.clip(np.searchsorted(block_hi_wave, wb), 0, neff_over_n.size - 1))
        factor = 1.0 / neff_over_n[k]  # n / n_eff for this block
        var = (np.sum(e ** 2) / n_keep ** 2) * factor
        wave_bin.append(wb)
        flux_bin.append(float(f.mean()))
        err_bin.append(float(np.sqrt(var)))
        blocks_used.append(k)
        n_eff_total += n_keep * neff_over_n[k]

    return {
        "wave_bin": np.asarray(wave_bin), "flux_bin": np.asarray(flux_bin),
        "err_bin": np.asarray(err_bin), "n_bins": len(wave_bin),
        "n_eff_total": float(n_eff_total),
        "blocks_used": np.asarray(blocks_used, dtype=int),
    }


def fit_spectrum(cfg, run_paths) -> FitSpectrum:
    """Assemble the fit-ready spectrum from the real run (WP-G3R-11 only)."""
    spec = load_final_spectrum(run_paths, err_column=cfg.get("g3_fit_err_column",
                                                             "flux_err_total"))
    wave, flux, err = spec["wave_A"], spec["flux"], spec["flux_err"]
    mask, prov = build_fit_masks(cfg, wave, run_paths)
    fm = dict(cfg.get("g3_fit_masks", {}))
    bad_mask = None
    if fm.get("use_stage04b_bad_mask", True):
        bad_mask = np.load(Path(run_paths.stage_dir) / BAD_MASK_FILE).astype(bool)
    cov = dict(np.load(Path(run_paths.stage_dir) / G1_COV_FILE))
    reb = rebin_for_fit(
        wave, flux, err, mask, cov,
        n_channels=int(cfg.get("g3_fit_bin_channels", 20)),
        wave_range=cfg.get("g3_fit_wave_range_A"),
        bad_mask=bad_mask,
        max_masked_frac=float(cfg.get("g3_fit_bin_max_masked_frac", 0.5)))
    prov = {**prov, "wave_range_A": list(cfg.get("g3_fit_wave_range_A")),
            "bin_channels": int(cfg.get("g3_fit_bin_channels", 20)),
            "err_column": spec["err_column"], "n_bins": reb["n_bins"],
            "variance_rule": "(sum sigma^2)/N^2 * (n/n_eff) per G1 block; "
                             "inter-bin correlation neglected (D8)"}
    return FitSpectrum(reb["wave_bin"], reb["flux_bin"], reb["err_bin"],
                       reb["n_bins"], reb["n_eff_total"], prov)


def plot_fit_spectrum(out_path, *, wave, flux, mask, fit_spec, wave_range):
    """QC figure: per-channel (gray), rebinned (points+bars), masks shaded,
    primary D8 range marked. Only the WP-G3R-11 stage runs this on real data."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    wave = np.asarray(wave, float)
    mask = np.asarray(mask, dtype=bool)
    fig = Figure(figsize=(10, 4))
    FigureCanvasAgg(fig)
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(wave, flux, color="0.6", lw=0.5, label="per channel")
    # shade masked regions
    edges = np.diff(mask.astype(int))
    starts = np.nonzero(edges == 1)[0] + 1
    stops = np.nonzero(edges == -1)[0] + 1
    if mask.size and mask[0]:
        starts = np.r_[0, starts]
    if mask.size and mask[-1]:
        stops = np.r_[stops, mask.size]
    for s, e in zip(starts, stops):
        ax.axvspan(wave[s], wave[min(e, wave.size - 1)], color="red", alpha=0.08)
    ax.errorbar(fit_spec.wave_bin, fit_spec.flux_bin, yerr=fit_spec.err_bin,
                fmt="o", ms=3, color="C0", lw=0.8, label="rebinned")
    ax.axvline(wave_range[0], color="k", ls="--", lw=0.7)
    ax.axvline(wave_range[1], color="k", ls="--", lw=0.7)
    ax.set_xlabel("wavelength [Å]")
    ax.set_ylabel("flux")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    return out_path


__all__ = [
    "FitSpectrum", "build_fit_masks", "fit_spectrum", "load_final_spectrum",
    "plot_fit_spectrum", "rebin_for_fit",
]
