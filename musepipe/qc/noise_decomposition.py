"""Noise decomposition of the post-subtraction residual (WP-S6a, wavesol plan).

Implements the FIXED model of Xie et al. 2020 (§5, their Fig. 8): at the
companion separation, measure the residual aperture-photometry noise as a
function of wavelength, and decompose it into a photon-like term that scales
with the primary flux and a flat background term,

    sigma_ap(lambda)^2 = (c * F*(lambda)^alpha)^2 + sigma_bg^2 ,

fitting c, alpha (~0.5 for photon noise), sigma_bg by least squares in log over
narrow 3-channel filters (excluding Halpha and stellar features). The paper
statement "we are X times the photon limit in Halpha at B's separation" is then

    factor = sigma_ap(Halpha, r_B) / (c * F*(Halpha)^alpha) .

Core functions are target-agnostic and I/O-free; the CLI runs one canonical
residual cube. This is a DIAGNOSTIC only — it does NOT recompute the E3 limits.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import warnings
from typing import Sequence

import numpy as np
from scipy.optimize import curve_fit

from musepipe.stages.stage_h05_contrast import ring_positions
from musepipe.stats import robust_sigma

DEFAULT_LO_A = 5100.0
DEFAULT_HI_A = 8800.0
DEFAULT_WIDTH_CH = 3
DEFAULT_HALPHA_A = 6562.8
DEFAULT_HALPHA_HALFWIDTH_A = 25.0   # excludes 6540-6590 (Xie/plan)
DEFAULT_APERTURE_RADIUS_PX = 4.0    # ~ PSF FWHM; a constant radius cancels in `factor`
DEFAULT_N_ANGLES = 36
DEFAULT_FSTAR_RADIUS_PX = 5.0

# The photon term (c*F^alpha) is only isolable where F*(lambda) spans a large
# dynamic range, so the fit uses the FULL clean continuum (Xie Fig. 8), NOT just
# the narrow 6510-6825 window (over which F* varies only ~2x and alpha is
# degenerate). These bands are dropped from the fit: AO laser notch, telluric
# O2/H2O, and the noisy red edge; Halpha is handled by its own flag.
DEFAULT_EXCLUDE_WINDOWS_A = (
    (5780.0, 6050.0),   # NaLGS AO laser notch
    (6860.0, 6960.0),   # telluric O2 B
    (7150.0, 7350.0),   # telluric H2O
    (7590.0, 7700.0),   # telluric O2 A
    (8100.0, 8400.0),   # telluric H2O
    (8800.0, 9350.0),   # noisy red edge
)


def build_lambda_filters(
    waves_A: np.ndarray,
    *,
    lo_A: float = DEFAULT_LO_A,
    hi_A: float = DEFAULT_HI_A,
    width_ch: int = DEFAULT_WIDTH_CH,
    halpha_A: float = DEFAULT_HALPHA_A,
    halpha_halfwidth_A: float = DEFAULT_HALPHA_HALFWIDTH_A,
    exclude_windows_A: Sequence[Sequence[float]] = (),
) -> list[dict]:
    """Contiguous ``width_ch``-channel filters spanning [lo, hi].

    Each filter is flagged ``is_halpha`` if its centre falls in the Halpha
    window and ``excluded`` if it overlaps an extra stellar-feature window;
    excluded/Halpha filters are kept in the table but dropped from the fit.
    """

    waves = np.asarray(waves_A, dtype=np.float64)
    idx = np.where((waves >= lo_A) & (waves <= hi_A))[0]
    if idx.size < width_ch:
        raise ValueError("wavelength range too small for any filter")
    filters: list[dict] = []
    for start in range(int(idx[0]), int(idx[-1]) + 1 - width_ch + 1, width_ch):
        sl = slice(start, start + width_ch)
        center = float(np.mean(waves[sl]))
        is_h = abs(center - halpha_A) <= halpha_halfwidth_A
        excl = any(lo <= center <= hi for lo, hi in exclude_windows_A)
        filters.append({"center_A": center, "sl": sl, "is_halpha": is_h, "excluded": excl})
    return filters


def aperture_flux(image: np.ndarray, cy: float, cx: float, radius: float,
                  *, min_finite_frac: float = 0.5) -> float:
    """Sum of finite pixels within ``radius`` of (cy, cx); NaN if too masked."""

    ny, nx = image.shape
    y0, y1 = max(0, int(cy - radius)), min(ny, int(cy + radius) + 1)
    x0, x1 = max(0, int(cx - radius)), min(nx, int(cx + radius) + 1)
    sub = image[y0:y1, x0:x1]
    yy, xx = np.mgrid[y0:y1, x0:x1]
    mask = (yy - cy) ** 2 + (xx - cx) ** 2 <= radius ** 2
    vals = sub[mask]
    finite = np.isfinite(vals)
    if finite.mean() < min_finite_frac or finite.sum() < 1:
        return np.nan
    return float(np.sum(vals[finite]))


def ring_aperture_sigma(
    image: np.ndarray,
    primary_yx: Sequence[float],
    r_px: float,
    *,
    aperture_radius_px: float = DEFAULT_APERTURE_RADIUS_PX,
    n_angles: int = DEFAULT_N_ANGLES,
    companion_yx: Sequence[float] | None = None,
    exclude_radius_px: float = 0.0,
) -> tuple[float, int]:
    """Robust scatter of aperture fluxes around a ring at radius ``r_px``."""

    positions = ring_positions(primary_yx, r_px, n_angles, image.shape)
    fluxes = []
    for (y, x, _theta) in positions:
        if companion_yx is not None and exclude_radius_px > 0:
            if np.hypot(y - companion_yx[0], x - companion_yx[1]) <= exclude_radius_px:
                continue
        f = aperture_flux(image, y, x, aperture_radius_px)
        if np.isfinite(f):
            fluxes.append(f)
    if len(fluxes) < 5:
        return np.nan, len(fluxes)
    return float(robust_sigma(np.asarray(fluxes))), len(fluxes)


def _sigma_model(F, c, alpha, sigma_bg):
    return np.sqrt((c * np.power(F, alpha)) ** 2 + sigma_bg ** 2)


def fit_noise_model(F: np.ndarray, sigma: np.ndarray) -> dict:
    """Least-squares-in-log fit of sigma^2 = (c F^alpha)^2 + sigma_bg^2."""

    F = np.asarray(F, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    ok = np.isfinite(F) & np.isfinite(sigma) & (F > 0) & (sigma > 0)
    F, sigma = F[ok], sigma[ok]
    if F.size < 4:
        raise ValueError("need >=4 finite (F, sigma) points to fit")

    c0 = float(np.median(sigma / np.sqrt(F)))
    p0 = [max(c0, 1e-12), 0.5, float(np.min(sigma))]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        popt, _ = curve_fit(
            lambda F, c, a, bg: np.log(_sigma_model(F, c, a, bg)),
            F, np.log(sigma), p0=p0,
            bounds=([0.0, 0.0, 0.0], [np.inf, 3.0, np.inf]), maxfev=10000,
        )
    c, alpha, sigma_bg = (float(v) for v in popt)
    return {"c": c, "alpha": alpha, "sigma_bg": sigma_bg, "n_fit": int(F.size)}


def compute_noise_decomposition(
    residual_cube: np.ndarray,
    waves_A: np.ndarray,
    primary_spectrum: np.ndarray,
    *,
    primary_yx: Sequence[float],
    r_B_px: float,
    companion_yx: Sequence[float] | None = None,
    aperture_radius_px: float = DEFAULT_APERTURE_RADIUS_PX,
    n_angles: int = DEFAULT_N_ANGLES,
    lo_A: float = DEFAULT_LO_A,
    hi_A: float = DEFAULT_HI_A,
    width_ch: int = DEFAULT_WIDTH_CH,
    halpha_A: float = DEFAULT_HALPHA_A,
    exclude_windows_A: Sequence[Sequence[float]] = DEFAULT_EXCLUDE_WINDOWS_A,
) -> dict:
    """Full S6a decomposition; returns the QC dict (see module docstring)."""

    cube = np.asarray(residual_cube, dtype=np.float64)
    waves = np.asarray(waves_A, dtype=np.float64)
    fstar = np.asarray(primary_spectrum, dtype=np.float64)
    if cube.shape[0] != waves.size or fstar.size != waves.size:
        raise ValueError("cube/waves/primary_spectrum spectral lengths must match")

    filters = build_lambda_filters(waves, lo_A=lo_A, hi_A=hi_A, width_ch=width_ch,
                                   halpha_A=halpha_A, exclude_windows_A=exclude_windows_A)
    exclude_r = 1.5 * aperture_radius_px if companion_yx is not None else 0.0

    table = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for f in filters:
            image = np.nanmean(cube[f["sl"]], axis=0)
            sig, n_ap = ring_aperture_sigma(
                image, primary_yx, r_B_px, aperture_radius_px=aperture_radius_px,
                n_angles=n_angles, companion_yx=companion_yx, exclude_radius_px=exclude_r)
            fbar = float(np.nanmean(fstar[f["sl"]]))
            table.append({"center_A": f["center_A"], "F_star": fbar, "sigma_ap": sig,
                          "n_ap": n_ap, "is_halpha": f["is_halpha"], "excluded": f["excluded"]})

    fit_rows = [t for t in table if not t["is_halpha"] and not t["excluded"]
                and np.isfinite(t["F_star"]) and np.isfinite(t["sigma_ap"])
                and t["F_star"] > 0 and t["sigma_ap"] > 0]
    F = np.array([t["F_star"] for t in fit_rows])
    S = np.array([t["sigma_ap"] for t in fit_rows])
    fit = fit_noise_model(F, S)
    c, alpha, sigma_bg = fit["c"], fit["alpha"], fit["sigma_bg"]
    corr = float(np.corrcoef(np.log(F), np.log(S))[0, 1]) if F.size >= 3 else float("nan")
    # The photon term is only meaningful where F* correlates with the noise and
    # alpha is non-degenerate; otherwise the residual is background/systematic-
    # limited and the factor formula (dividing by c*F^alpha) is not meaningful.
    regime = "photon_limited" if (alpha > 0.25 and corr > 0.4) else "background_or_systematic_limited"

    halpha_rows = [t for t in table if t["is_halpha"] and np.isfinite(t["sigma_ap"])
                   and np.isfinite(t["F_star"]) and t["F_star"] > 0]
    if halpha_rows:
        hrow = min(halpha_rows, key=lambda t: abs(t["center_A"] - halpha_A))
        photon = c * hrow["F_star"] ** alpha
        factor = float(hrow["sigma_ap"] / photon) if photon > 0 else float("nan")
        halpha_center = hrow["center_A"]
        halpha_sigma = hrow["sigma_ap"]
    else:
        factor = halpha_center = halpha_sigma = float("nan")

    return {
        "c": c, "alpha": alpha, "sigma_bg": sigma_bg, "n_fit": fit["n_fit"],
        "corr_logF_logS": corr, "regime": regime,
        "factor_over_photon_at_halpha_rB": factor,
        "halpha_center_A": halpha_center, "halpha_sigma_ap": halpha_sigma,
        "r_B_px": float(r_B_px), "aperture_radius_px": float(aperture_radius_px),
        "n_angles": int(n_angles),
        "per_lambda_table": [
            {"center_A": round(t["center_A"], 3),
             "F_star": None if not np.isfinite(t["F_star"]) else float(t["F_star"]),
             "sigma_ap": None if not np.isfinite(t["sigma_ap"]) else float(t["sigma_ap"]),
             "n_ap": t["n_ap"], "is_halpha": t["is_halpha"], "excluded": t["excluded"]}
            for t in table
        ],
    }


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def _load_cube(path: Path):
    from astropy.io import fits

    with fits.open(path) as hdul:
        data = None
        for hdu in hdul:
            if getattr(hdu.data, "ndim", 0) == 3:
                data = np.asarray(hdu.data, dtype=np.float64)
                break
        waves = None
        if "WAVELENGTH" in hdul:
            waves = np.asarray(hdul["WAVELENGTH"].data, dtype=np.float64).ravel()
        if data is None:
            raise ValueError(f"no 3D cube in {path}")
        if waves is None:
            hdr = hdul[0].header
            crval = float(hdr.get("CRVAL3", 0.0)); cd = float(hdr.get("CD3_3", hdr.get("CDELT3", 1.0)))
            crpix = float(hdr.get("CRPIX3", 1.0))
            waves = crval + (np.arange(data.shape[0]) - (crpix - 1.0)) * cd
    return data, waves


def _primary_spectrum(input_cube_path: Path, primary_yx, radius) -> np.ndarray:
    from musepipe.reduction.verify import extract_aperture_spectrum

    cube, _ = _load_cube(input_cube_path)
    return extract_aperture_spectrum(cube, (float(primary_yx[0]), float(primary_yx[1])), float(radius))


def _write_figure(qc: dict, out_path: Path) -> None:
    import os

    cache_dir = Path(os.environ.get("TMPDIR", "/tmp")) / "musepipe_matplotlib"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    tab = qc["per_lambda_table"]
    fit = [t for t in tab if not t["is_halpha"] and not t["excluded"] and t["F_star"] and t["sigma_ap"]]
    F = np.array([t["F_star"] for t in fit]); S = np.array([t["sigma_ap"] for t in fit])
    c, a, bg = qc["c"], qc["alpha"], qc["sigma_bg"]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(F, S, s=12, color="C0", label="filters (fit)")
    xs = np.geomspace(np.nanmin(F), np.nanmax(F), 100)
    ax.plot(xs, _sigma_model(xs, c, a, bg), "k-", lw=1.5,
            label=f"model: c={c:.3g}, alpha={a:.2f}, sigma_bg={bg:.3g}")
    ax.plot(xs, c * xs ** a, "k--", lw=1, alpha=0.6, label="photon term c*F^alpha")
    if np.isfinite(qc["halpha_sigma_ap"]):
        # F* at Halpha
        hrow = min((t for t in tab if t["is_halpha"] and t["F_star"]),
                   key=lambda t: abs(t["center_A"] - qc["halpha_center_A"]))
        ax.scatter([hrow["F_star"]], [qc["halpha_sigma_ap"]], marker="*", s=220,
                   color="tab:red", zorder=5,
                   label=f"Halpha: {qc['factor_over_photon_at_halpha_rB']:.2f}x photon limit")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("primary flux F*(lambda)"); ax.set_ylabel(f"residual aperture noise at r_B={qc['r_B_px']:.0f}px")
    ax.set_title("S6a noise decomposition (Xie Fig. 8)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def run_phase(args: argparse.Namespace) -> int:
    from musepipe.qc.cube_qc import sha256_file, utc_now_iso

    resid_path = Path(args.residual_cube).expanduser()
    input_path = Path(args.input_cube).expanduser()
    for p in (resid_path, input_path):
        if not p.exists():
            print(f"ERROR: missing {p}", file=sys.stderr)
            return 2

    cube, waves = _load_cube(resid_path)
    primary_yx = tuple(float(v) for v in args.primary_yx)
    companion_yx = tuple(float(v) for v in args.companion_yx) if args.companion_yx else None
    r_B = args.sep_px if args.sep_px is not None else (
        float(np.hypot(primary_yx[0] - companion_yx[0], primary_yx[1] - companion_yx[1]))
        if companion_yx else None)
    if r_B is None:
        print("ERROR: need --sep-px or --companion-yx", file=sys.stderr)
        return 2
    fstar = _primary_spectrum(input_path, primary_yx, args.fstar_radius)

    qc = compute_noise_decomposition(
        cube, waves, fstar, primary_yx=primary_yx, r_B_px=r_B, companion_yx=companion_yx,
        aperture_radius_px=args.aperture_radius, n_angles=args.n_angles)
    qc["stage"] = "S6_noise_decomposition"
    qc["timestamp_utc"] = utc_now_iso()
    qc["residual_cube"] = str(resid_path)
    qc["input_cube"] = str(input_path)
    qc["sha256_residual"] = "" if args.skip_checksum else sha256_file(resid_path)

    if args.plot_output:
        plot_path = Path(args.plot_output); plot_path.parent.mkdir(parents=True, exist_ok=True)
        _write_figure(qc, plot_path); qc["plot_output"] = str(plot_path)

    qc_path = Path(args.qc_output); qc_path.parent.mkdir(parents=True, exist_ok=True)
    qc_path.write_text(json.dumps(qc, indent=2) + "\n", encoding="utf-8")
    print(f"S6a: alpha={qc['alpha']:.2f} (corr logF-logS {qc['corr_logF_logS']:+.2f}, {qc['regime']}) "
          f"c={qc['c']:.3g} sigma_bg={qc['sigma_bg']:.3g} "
          f"| Halpha at r_B={r_B:.0f}px is {qc['factor_over_photon_at_halpha_rB']:.2f}x the photon limit "
          f"(n_fit={qc['n_fit']}) -> {qc_path}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="S6a residual noise decomposition (Xie Fig. 8).")
    p.add_argument("--residual-cube", required=True)
    p.add_argument("--input-cube", required=True, help="cube to extract the primary F*(lambda) from")
    p.add_argument("--qc-output", required=True)
    p.add_argument("--plot-output", default=None)
    p.add_argument("--primary-yx", nargs=2, type=float, required=True, metavar=("Y", "X"))
    p.add_argument("--companion-yx", nargs=2, type=float, default=None, metavar=("Y", "X"))
    p.add_argument("--sep-px", type=float, default=None)
    p.add_argument("--aperture-radius", type=float, default=DEFAULT_APERTURE_RADIUS_PX)
    p.add_argument("--fstar-radius", type=float, default=DEFAULT_FSTAR_RADIUS_PX)
    p.add_argument("--n-angles", type=int, default=DEFAULT_N_ANGLES)
    p.add_argument("--skip-checksum", action="store_true")
    p.set_defaults(func=run_phase)
    args = p.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
