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


#: Bandas telúricas enmascaradas EN EL DATO por defecto (D9): solo O2 B y O2 A.
#: Las dos de H2O (7130–7360, 8100–8400) se quitaron el 2026-09-24: el residuo
#: telúrico medido allí es compatible con cero (α = −0.01 ± 0.18 y −0.13 ± 0.16
#: en las primarias) y el sistemático que declara A3 ya va en ``flux_err_total``
#: (``docs/2026-09-24_residuo_telurico_h2o_y_mascara_d9.md``). Las máscaras
#: propias de las bibliotecas sin corrección telúrica no cambian.
DEFAULT_TELLURIC_BANDS_A = ((6860.0, 6960.0), (7590.0, 7700.0))


@dataclass(frozen=True)
class FitSpectrum:
    """Frozen fit-ready spectrum — the only real-data input to the fits.

    ``n_dof`` son los grados de libertad BASE del χ² (antes de restar los
    parámetros libres). En bins de N ≫ corr_len canales la varianza por bin
    ya es la correcta y los bins son ~independientes: ``n_dof = n_bins`` (D8).
    Canal a canal (``bin_channels = 1``, plan 2026-09-23) cada canal pesa
    σ²·(n/n_eff) y E[χ²] = Σ n_eff/n: ``n_dof = n_eff`` (efectivos). ``None``
    (llamadores antiguos) = ``n_bins``.
    """

    wave_bin: np.ndarray
    flux_bin: np.ndarray
    err_bin: np.ndarray
    n_bins: int
    n_eff: float
    mask_provenance: dict
    n_dof: float | None = None
    bin_channels: int | None = None
    resolution: dict | None = None
    dof_weight: np.ndarray | None = None

    @property
    def dof_base(self) -> float:
        return float(self.n_bins if self.n_dof is None else self.n_dof)

    def dof_used(self, used) -> float:
        """Grados de libertad base de los bins EFECTIVAMENTE usados por un modelo.

        Un modelo que no cubre algunos bins (NaN) no puede comparar su χ²
        contra los mismos dof que uno que los cubre todos. Con ``dof_weight``
        (n_eff/n por canal en el ajuste nativo; 1 por bin en el binado) se
        suma sobre los usados; sin él, la fracción usada de ``dof_base``.
        """
        used = np.asarray(used, bool)
        if self.dof_weight is not None:
            return float(np.sum(np.asarray(self.dof_weight, float)[used]))
        n = max(1, int(self.n_bins))
        return self.dof_base * float(used.sum()) / n


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

    bands = fm.get("telluric_bands_A", DEFAULT_TELLURIC_BANDS_A)
    before = int(mask.sum())
    for lo, hi in bands:
        mask |= (wave >= float(lo)) & (wave <= float(hi))
    prov["telluric_bands"] = {"bands_A": [list(b) for b in bands],
                              "n_added": int(mask.sum()) - before,
                              "source": ("config g3_fit_masks.telluric_bands_A"
                                         if "telluric_bands_A" in fm else
                                         "default D9 (O2 B + O2 A; docs/2026-09-24_"
                                         "residuo_telurico_h2o_y_mascara_d9.md)")}
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


def neff_over_n_per_channel(wave, cov, bad_mask=None) -> np.ndarray:
    """``n_eff/n`` del bloque de G1 que contiene cada canal (misma regla que D8)."""
    wave = np.asarray(wave, float)
    neff_over_n = np.asarray(cov["n_eff_over_n_by_block"], dtype=float)
    block_hi_wave = _block_upper_waves(wave, bad_mask, np.asarray(cov["block_bounds"]))
    k = np.clip(np.searchsorted(block_hi_wave, wave), 0, neff_over_n.size - 1)
    return neff_over_n[k]


def rebin_for_fit(wave, flux, err, mask, cov, *, n_channels, wave_range,
                  bad_mask=None, max_masked_frac=0.5, neff_over_n_channel=None) -> dict:
    """Rebin per D8: groups of ``n_channels`` within ``wave_range``, drop bins
    with >``max_masked_frac`` masked, variance = (Σσ²)/N² × (n/n_eff) per the
    G1 block containing the bin. Returns wave/flux/err bins + n_bins + n_eff_total.

    ``n_channels = 1`` is the native fit (plan 2026-09-23): each channel keeps
    its own σ inflated by √(n/n_eff), and ``n_eff_total = Σ n_eff/n`` are the
    effective degrees of freedom. ``neff_over_n_channel`` (per channel)
    replaces the block value — used when the DATA were degraded to a coarser
    library (:func:`degrade_observed`), which lengthens the correlation."""
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

    wave_bin, flux_bin, err_bin, blocks_used, ratio_bin = [], [], [], [], []
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
        if neff_over_n_channel is None:
            ratio = float(neff_over_n[k])
        else:
            ratio = float(np.mean(np.asarray(neff_over_n_channel, float)[chans][keep]))
        factor = 1.0 / ratio  # n / n_eff for this block
        var = (np.sum(e ** 2) / n_keep ** 2) * factor
        wave_bin.append(wb)
        flux_bin.append(float(f.mean()))
        err_bin.append(float(np.sqrt(var)))
        blocks_used.append(k)
        ratio_bin.append(ratio)
        n_eff_total += n_keep * ratio

    return {
        "wave_bin": np.asarray(wave_bin), "flux_bin": np.asarray(flux_bin),
        "err_bin": np.asarray(err_bin), "n_bins": len(wave_bin),
        "n_eff_total": float(n_eff_total),
        "blocks_used": np.asarray(blocks_used, dtype=int),
        "neff_over_n_bin": np.asarray(ratio_bin, dtype=float),
    }


def degrade_observed(wave, flux, err, mask, kernel_fwhm_A, neff_over_n):
    """Degrade the per-channel DATA by a Gaussian of FWHM ``kernel_fwhm_A`` (λ).

    Plan 2026-09-23 (Q2): a library coarser than MUSE (Kesseli, R ≈ 2000) is
    compared with the data degraded to ITS resolution, not the other way round.

    Errors, consistently. The G1 noise is correlated (``n/n_eff`` = L channels,
    corr_len ≈ 1.45). It is modelled as white noise smoothed by a Gaussian, for
    which Σρ = L grows in quadrature with the smoothing: a further Gaussian of
    σ_k channels gives ``L_out = √(L² + 4π σ_k²)`` and ``σ_out² = σ² · L/L_out``
    (exact at σ_k = 0; for white input it tends to Σk²σ² = σ²/(2√π σ_k)).
    ``n_eff/n`` becomes ``1/L_out`` and is returned per channel.

    Masked / non-finite channels are interpolated over before convolving (so
    they do not leak NaN) and the mask is GROWN by ±1 kernel FWHM, so no kept
    channel carries flux from a masked line, band or bad channel.
    Returns ``(flux_out, err_out, mask_out, neff_over_n_out)``.
    """
    from .prep import degrade_variable
    from ..constants import fwhm_to_sigma

    wave = np.asarray(wave, float)
    flux = np.asarray(flux, float)
    err = np.asarray(err, float)
    mask = np.asarray(mask, bool)
    kern = np.broadcast_to(np.asarray(kernel_fwhm_A, float), wave.shape).astype(float)
    kern = np.where(np.isfinite(kern) & (kern > 0), kern, 0.0)
    ratio_in = np.broadcast_to(np.asarray(neff_over_n, float), wave.shape)

    bad = mask | ~np.isfinite(flux) | ~np.isfinite(err) | (err <= 0)
    good = ~bad
    if good.sum() < 2:
        raise RuntimeError("degrade_observed: fewer than 2 usable channels")
    ff = np.interp(wave, wave[good], flux[good])
    vv = np.interp(wave, wave[good], err[good] ** 2)

    f_out = degrade_variable(wave, ff, kern, edge="nearest")
    v_smooth = degrade_variable(wave, vv, kern, edge="nearest")
    dl = np.gradient(wave)
    sig_k = np.array([fwhm_to_sigma(k) for k in kern]) / dl
    L_in = 1.0 / ratio_in
    L_out = np.sqrt(L_in ** 2 + 4.0 * np.pi * sig_k ** 2)
    e_out = np.sqrt(v_smooth * L_in / L_out)

    grown = bad.copy()
    if bad.any():
        idx = np.nonzero(bad)[0]
        for i in idx:
            half = kern[i] if kern[i] > 0 else 0.0
            if half <= 0:
                continue
            lo = np.searchsorted(wave, wave[i] - half)
            hi = np.searchsorted(wave, wave[i] + half, side="right")
            grown[lo:hi] = True
    return f_out, e_out, (grown | mask), 1.0 / L_out


def load_fit_inputs(cfg, run_paths) -> dict:
    """Per-channel real inputs of the fit (spectrum, D9 mask, G1 covariance)."""
    spec = load_final_spectrum(run_paths, err_column=cfg.get("g3_fit_err_column",
                                                             "flux_err_total"))
    wave = spec["wave_A"]
    mask, prov = build_fit_masks(cfg, wave, run_paths)
    fm = dict(cfg.get("g3_fit_masks", {}))
    bad_mask = None
    if fm.get("use_stage04b_bad_mask", True):
        bad_mask = np.load(Path(run_paths.stage_dir) / BAD_MASK_FILE).astype(bool)
    cov = dict(np.load(Path(run_paths.stage_dir) / G1_COV_FILE))
    return {"wave": wave, "flux": spec["flux"], "err": spec["flux_err"],
            "err_column": spec["err_column"], "mask": mask, "mask_prov": prov,
            "bad_mask": bad_mask, "cov": cov}


#: Canales por bin por defecto: nativo (plan 2026-09-23 §3; D8 decía 20). Un
#: run que declare ``g3_fit_bin_channels`` manda.
DEFAULT_BIN_CHANNELS = 1


def fit_spectrum_from_inputs(inputs, *, n_channels, wave_range, max_masked_frac=0.5,
                             data_kernel_fwhm_A=None, resolution=None) -> FitSpectrum:
    """Build the fit-ready spectrum from :func:`load_fit_inputs`.

    ``data_kernel_fwhm_A`` (scalar or callable of λ): degrade the DATA first
    (:func:`degrade_observed`) — the per-library test of a library coarser
    than MUSE. ``resolution`` is recorded as is in the FitSpectrum.
    """
    wave, flux, err, mask = (inputs["wave"], inputs["flux"], inputs["err"],
                             inputs["mask"])
    neff_ch = None
    prov_deg = None
    if data_kernel_fwhm_A is not None:
        kern = (data_kernel_fwhm_A(wave) if callable(data_kernel_fwhm_A)
                else np.full(wave.shape, float(data_kernel_fwhm_A)))
        ratio = neff_over_n_per_channel(wave, inputs["cov"], inputs["bad_mask"])
        flux, err, mask, neff_ch = degrade_observed(wave, flux, err, mask, kern, ratio)
        lo, hi = float(wave_range[0]), float(wave_range[1])
        sel = (wave >= lo) & (wave <= hi)
        prov_deg = {"kernel_fwhm_A_range": [float(np.min(kern[sel])), float(np.max(kern[sel]))]
                    if sel.any() else None,
                    "neff_over_n_range": [float(np.min(neff_ch[sel])), float(np.max(neff_ch[sel]))]
                    if sel.any() else None,
                    "rule": "L_out = sqrt(L^2 + 4 pi sigma_k^2); var *= L/L_out; "
                            "mask grown by +-1 kernel FWHM"}
    reb = rebin_for_fit(
        wave, flux, err, mask, inputs["cov"], n_channels=int(n_channels),
        wave_range=wave_range, bad_mask=inputs["bad_mask"],
        max_masked_frac=float(max_masked_frac), neff_over_n_channel=neff_ch)
    native = int(n_channels) == 1
    n_dof = reb["n_eff_total"] if native else float(reb["n_bins"])
    prov = {**inputs["mask_prov"], "wave_range_A": list(wave_range),
            "bin_channels": int(n_channels),
            "err_column": inputs["err_column"], "n_bins": reb["n_bins"],
            "n_dof": n_dof,
            "dof_rule": ("n_eff = sum n_eff/n (native, per-channel sigma^2*n/n_eff)"
                         if native else "n_bins (D8: inter-bin correlation neglected)"),
            "variance_rule": "(sum sigma^2)/N^2 * (n/n_eff) per G1 block; "
                             "inter-bin correlation neglected (D8)"}
    if prov_deg is not None:
        prov["data_degraded"] = prov_deg
    return FitSpectrum(reb["wave_bin"], reb["flux_bin"], reb["err_bin"],
                       reb["n_bins"], reb["n_eff_total"], prov, n_dof=n_dof,
                       bin_channels=int(n_channels), resolution=resolution,
                       dof_weight=(reb["neff_over_n_bin"] if native
                                   else np.ones(reb["n_bins"])))


def fit_spectrum(cfg, run_paths, *, n_channels=None, data_kernel_fwhm_A=None,
                 resolution=None, inputs=None) -> FitSpectrum:
    """Assemble the fit-ready spectrum from the real run (WP-G3R-11 only)."""
    if inputs is None:
        inputs = load_fit_inputs(cfg, run_paths)
    n = int(n_channels if n_channels is not None
            else cfg.get("g3_fit_bin_channels", DEFAULT_BIN_CHANNELS))
    return fit_spectrum_from_inputs(
        inputs, n_channels=n, wave_range=cfg.get("g3_fit_wave_range_A"),
        max_masked_frac=float(cfg.get("g3_fit_bin_max_masked_frac", 0.5)),
        data_kernel_fwhm_A=data_kernel_fwhm_A, resolution=resolution)


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
    "DEFAULT_BIN_CHANNELS", "DEFAULT_TELLURIC_BANDS_A", "FitSpectrum", "build_fit_masks", "degrade_observed",
    "fit_spectrum", "fit_spectrum_from_inputs", "load_final_spectrum", "load_fit_inputs",
    "neff_over_n_per_channel", "plot_fit_spectrum", "rebin_for_fit",
]
