"""Per-spaxel Halpha line-to-continuum map (WP-S1, plan wavesol/stripes).

Implements the diagnostic of Xie et al. 2020 (§4.1, their Eq. 1 and Fig. 3):
fit, for every halo spaxel of a combined cube, the primary's Halpha emission
with a Gaussian-on-continuum model

    phi(lambda) = b * (1 + a * exp(-(lambda-mu)^2 / (2 sigma^2)))

and map the line-to-continuum amplitude ``a``, the width ``sigma``, the centre
``mu`` and the integrated line power ``P = a*b*sigma*sqrt(2 pi)``. Xie's test:
if ``a`` and ``sigma`` are ANTI-correlated at roughly CONSTANT ``P``, the
variation is instrumental (a spatially varying LSF, aligned with the slicers),
not an astrophysical ghost. Slicer-aligned structure in the ``a``/``sigma``
maps is the same stripe signature that :mod:`musepipe.qc.wavesol_map` looks
for in the wavelength solution.

Core functions are target-agnostic and I/O-free (arrays in, dicts out); the CLI
at the bottom runs one cube. This module does NOT interpret the correlation
(that is paso S1b / human) and does NOT touch the E1 LSF.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Mapping, Sequence
import warnings

import numpy as np
from scipy.optimize import curve_fit

from musepipe.qc.wavesol_map import (
    bin_cube_spatial,
    brightness_selection_mask,
    spaxel_brightness_map,
    structure_metrics,
)
from musepipe.stats import robust_sigma


DEFAULT_FIT_WINDOW_A: tuple[float, float] = (6480.0, 6650.0)
DEFAULT_LINE_WINDOW_A: tuple[float, float] = (6540.0, 6590.0)
DEFAULT_BRIGHTNESS_PERCENTILE = 50.0
DEFAULT_MIN_SNR = 3.0
MIN_FIT_CHANNELS = 20

# curve_fit bounds (plan §S1a): a in [0,50], mu in the line window, sigma in
# [0.5,8] A, b > 0.
A_BOUNDS = (0.0, 50.0)
SIGMA_BOUNDS = (0.5, 8.0)
SQRT_2PI = float(np.sqrt(2.0 * np.pi))


def phi_model(lam: np.ndarray, b: float, a: float, mu: float, sigma: float) -> np.ndarray:
    """Gaussian emission line on a flat continuum (Xie Eq. 1)."""

    lam = np.asarray(lam, dtype=np.float64)
    return b * (1.0 + a * np.exp(-0.5 * ((lam - mu) / sigma) ** 2))


def _seed_parameters(
    waves: np.ndarray, spec: np.ndarray, line_window_A: Sequence[float]
) -> tuple[float, float, float, float]:
    """Moment seeds (b, a, mu, sigma) for the Halpha fit."""

    lw0, lw1 = float(line_window_A[0]), float(line_window_A[1])
    line = (waves >= lw0) & (waves <= lw1)
    cont = ~line
    b0 = float(np.median(spec[cont])) if cont.any() else float(np.median(spec))
    if not np.isfinite(b0) or b0 <= 0:
        b0 = max(float(np.median(spec)), 1e-6)

    excess = spec - b0
    if line.any() and np.nanmax(excess[line]) > 0:
        wl = waves[line]
        ex = np.clip(excess[line], 0.0, None)
        peak = float(np.nanmax(excess[line]))
        a0 = peak / b0
        total = float(ex.sum())
        if total > 0:
            mu0 = float(np.sum(wl * ex) / total)
            sig0 = float(np.sqrt(max(np.sum(ex * (wl - mu0) ** 2) / total, 0.25)))
        else:
            mu0, sig0 = 0.5 * (lw0 + lw1), 2.4
    else:
        a0, mu0, sig0 = 0.1, 0.5 * (lw0 + lw1), 2.4

    mu0 = float(np.clip(mu0, lw0, lw1))
    sig0 = float(np.clip(sig0, SIGMA_BOUNDS[0], SIGMA_BOUNDS[1]))
    a0 = float(np.clip(a0, A_BOUNDS[0], A_BOUNDS[1]))
    return b0, a0, mu0, sig0


def fit_halpha_spaxel(
    waves_w: np.ndarray,
    spec_w: np.ndarray,
    *,
    line_window_A: Sequence[float] = DEFAULT_LINE_WINDOW_A,
    min_snr: float = DEFAULT_MIN_SNR,
) -> dict[str, float]:
    """Fit one spectrum's Halpha with :func:`phi_model`.

    Returns ``{a, mu_A, sigma_A, b, P, snr, success}``. A spectrum without a
    detectable line (or a failed fit) comes back all-NaN with ``success=0.0``
    rather than raising.
    """

    fail = {"a": np.nan, "mu_A": np.nan, "sigma_A": np.nan, "b": np.nan,
            "P": np.nan, "snr": np.nan, "success": 0.0}

    waves = np.asarray(waves_w, dtype=np.float64)
    spec = np.asarray(spec_w, dtype=np.float64)
    good = np.isfinite(waves) & np.isfinite(spec)
    if int(good.sum()) < MIN_FIT_CHANNELS:
        return fail
    waves, spec = waves[good], spec[good]

    lw0, lw1 = float(line_window_A[0]), float(line_window_A[1])
    b0, a0, mu0, sig0 = _seed_parameters(waves, spec, line_window_A)

    lower = [0.0, A_BOUNDS[0], lw0, SIGMA_BOUNDS[0]]
    upper = [np.inf, A_BOUNDS[1], lw1, SIGMA_BOUNDS[1]]
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            popt, _ = curve_fit(
                phi_model, waves, spec, p0=[b0, a0, mu0, sig0],
                bounds=(lower, upper), maxfev=5000,
            )
    except (RuntimeError, ValueError):
        return fail

    b, a, mu, sigma = (float(v) for v in popt)
    if not all(np.isfinite(v) for v in (b, a, mu, sigma)) or b <= 0 or a <= 0:
        return fail

    resid = spec - phi_model(waves, b, a, mu, sigma)
    noise = robust_sigma(resid)
    peak = a * b  # line peak above continuum
    snr = peak / noise if np.isfinite(noise) and noise > 0 else np.inf
    if snr < float(min_snr):
        return fail

    P = a * b * sigma * SQRT_2PI
    return {"a": a, "mu_A": mu, "sigma_A": sigma, "b": b, "P": P,
            "snr": float(snr), "success": 1.0}


def compute_halpha_map(
    cube: np.ndarray,
    waves_A: np.ndarray,
    *,
    fit_window_A: Sequence[float] = DEFAULT_FIT_WINDOW_A,
    line_window_A: Sequence[float] = DEFAULT_LINE_WINDOW_A,
    brightness_percentile: float = DEFAULT_BRIGHTNESS_PERCENTILE,
    min_snr: float = DEFAULT_MIN_SNR,
) -> dict[str, object]:
    """Per-spaxel Halpha maps of a (nlam, ny, nx) cube.

    Returns a dict with `a_map`, `sigma_map_A`, `mu_map_A`, `P_map`, `b_map`,
    `fail_mask` (True where no usable fit), `select_mask`, `channel_step_A`,
    `windows`. Spaxels outside the brightness selection, or without a usable
    fit, are NaN in the maps.
    """

    data = np.asarray(cube, dtype=np.float64)
    if data.ndim != 3:
        raise ValueError(f"compute_halpha_map expects a 3D cube, got shape {data.shape}")
    waves = np.asarray(waves_A, dtype=np.float64)
    if waves.size != data.shape[0]:
        raise ValueError("wavelength axis length does not match cube")

    step_A = float(np.median(np.diff(waves)))
    fmask = (waves >= float(fit_window_A[0])) & (waves <= float(fit_window_A[1]))
    if int(fmask.sum()) < MIN_FIT_CHANNELS:
        raise ValueError("fit window has too few channels inside the wavelength range")
    wsub = waves[fmask]

    brightness = spaxel_brightness_map(data)
    select = brightness_selection_mask(brightness, brightness_percentile)

    ny, nx = data.shape[1:]
    a_map = np.full((ny, nx), np.nan)
    sigma_map = np.full((ny, nx), np.nan)
    mu_map = np.full((ny, nx), np.nan)
    P_map = np.full((ny, nx), np.nan)
    b_map = np.full((ny, nx), np.nan)
    fail = np.ones((ny, nx), dtype=bool)

    for iy, ix in np.argwhere(select):
        res = fit_halpha_spaxel(wsub, data[fmask, iy, ix],
                                line_window_A=line_window_A, min_snr=min_snr)
        if res["success"] >= 1.0:
            a_map[iy, ix] = res["a"]
            sigma_map[iy, ix] = res["sigma_A"]
            mu_map[iy, ix] = res["mu_A"]
            P_map[iy, ix] = res["P"]
            b_map[iy, ix] = res["b"]
            fail[iy, ix] = False

    return {
        "a_map": a_map,
        "sigma_map_A": sigma_map,
        "mu_map_A": mu_map,
        "P_map": P_map,
        "b_map": b_map,
        "fail_mask": fail,
        "select_mask": select,
        "channel_step_A": step_A,
        "windows": {"fit_window_A": [float(fit_window_A[0]), float(fit_window_A[1])],
                    "line_window_A": [float(line_window_A[0]), float(line_window_A[1])]},
    }


def halpha_structure_metrics(
    result: Mapping[str, object], orientation: str = "vertical"
) -> dict[str, object]:
    """Slicer-aligned structure of the ``a`` and ``sigma`` maps + the a-sigma
    correlation.

    Reuses :func:`musepipe.qc.wavesol_map.structure_metrics` (its
    ``structure_significance`` is a scale-free amp/noise ratio, so it applies
    to the dimensionless ``a`` map and the ``sigma`` map alike; a unit channel
    step is passed since these maps are not in channels). Xie Fig. 3: ``a`` and
    ``sigma`` anti-correlated at roughly constant ``P`` => instrumental LSF
    variation, not a ghost. This function reports the numbers; it does NOT
    interpret them (that is S1b / human).
    """

    a_map = np.asarray(result["a_map"], dtype=np.float64)
    sigma_map = np.asarray(result["sigma_map_A"], dtype=np.float64)
    P_map = np.asarray(result["P_map"], dtype=np.float64)

    a_struct = structure_metrics(a_map, 1.0, orientation)
    sig_struct = structure_metrics(sigma_map, 1.0, orientation)

    ok = np.isfinite(a_map) & np.isfinite(sigma_map)
    n_fit = int(ok.sum())
    if n_fit >= 3 and np.std(a_map[ok]) > 0 and np.std(sigma_map[ok]) > 0:
        corr_a_sigma = float(np.corrcoef(a_map[ok], sigma_map[ok])[0, 1])
    else:
        corr_a_sigma = float("nan")

    P_ok = P_map[np.isfinite(P_map)]
    P_median = float(np.median(P_ok)) if P_ok.size else float("nan")
    P_cov = (float(robust_sigma(P_ok) / P_median)
             if P_ok.size and np.isfinite(P_median) and P_median != 0 else float("nan"))

    return {
        "orientation": orientation,
        "n_fit": n_fit,
        "a_median": float(np.median(a_map[ok])) if n_fit else float("nan"),
        "a_p95": float(np.percentile(a_map[ok], 95.0)) if n_fit else float("nan"),
        "a_structure_significance": a_struct["structure_significance"],
        "a_transverse_significance": a_struct["transverse_significance"],
        "sigma_median_A": float(np.median(sigma_map[ok])) if n_fit else float("nan"),
        "sigma_p95_A": float(np.percentile(sigma_map[ok], 95.0)) if n_fit else float("nan"),
        "sigma_structure_significance": sig_struct["structure_significance"],
        "sigma_transverse_significance": sig_struct["transverse_significance"],
        "corr_a_sigma": corr_a_sigma,
        "P_median": P_median,
        "P_cov": P_cov,
    }


def stripe_halpha_correlation(
    offset_map_ch: np.ndarray,
    a_map: np.ndarray,
    sigma_map_A: np.ndarray,
    mu_map_A: np.ndarray,
    orientation: str = "vertical",
    *,
    min_count: int = 5,
) -> dict[str, object]:
    """Per-column correlation between the S0 wavesol stripe signal and the S1
    Halpha maps (paso S1b integration into E2).

    "Dirty stripe zones" are taken from the S0 offset map on the SAME grid: the
    per-column robust scatter of the offset is the local wavelength-solution
    disturbance. Xie's logic is that instrumental LSF variation (stripes) would
    make the Halpha width vary spatially IN STEP with that disturbance, so the
    key number is ``corr(stripe scatter, Halpha sigma)``. Also reports whether
    the Halpha centroid tracks the per-column offset. Reports numbers only — no
    interpretation (S1b/human).
    """

    from musepipe.qc.wavesol_map import stripe_profile

    off = stripe_profile(np.asarray(offset_map_ch, dtype=np.float64), orientation)
    a_p = stripe_profile(np.asarray(a_map, dtype=np.float64), orientation)
    s_p = stripe_profile(np.asarray(sigma_map_A, dtype=np.float64), orientation)
    m_p = stripe_profile(np.asarray(mu_map_A, dtype=np.float64), orientation)

    def _corr(x, y, count):
        ok = np.isfinite(x) & np.isfinite(y) & (count >= int(min_count))
        if int(ok.sum()) < 3 or np.std(x[ok]) == 0 or np.std(y[ok]) == 0:
            return float("nan"), int(ok.sum())
        return float(np.corrcoef(x[ok], y[ok])[0, 1]), int(ok.sum())

    corr_sigma, n_cols = _corr(off["scatter"], s_p["profile"],
                               np.minimum(off["count"], s_p["count"]))
    corr_a, _ = _corr(off["scatter"], a_p["profile"],
                      np.minimum(off["count"], a_p["count"]))
    corr_mu, _ = _corr(off["profile"], m_p["profile"],
                       np.minimum(off["count"], m_p["count"]))

    return {
        "orientation": orientation,
        "n_columns": n_cols,
        "stripe_indicator": "per-column robust scatter of the S0 wavesol offset map (channels)",
        "corr_stripe_scatter_vs_halpha_sigma": corr_sigma,
        "corr_stripe_scatter_vs_halpha_a": corr_a,
        "corr_stripe_offset_vs_halpha_mu": corr_mu,
    }


# --------------------------------------------------------------------------- #
# CLI (paso S1b runs this over the realigned cube; core stays paso S1a).
# --------------------------------------------------------------------------- #


def _write_halpha_plots(
    result: Mapping[str, object],
    metrics: Mapping[str, object],
    *,
    out_path: Path,
) -> None:
    import os

    from musepipe.stats import robust_limits

    cache_dir = Path(os.environ.get("TMPDIR", "/tmp")) / "musepipe_matplotlib"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    a_map = np.asarray(result["a_map"], dtype=np.float64)
    sigma_map = np.asarray(result["sigma_map_A"], dtype=np.float64)
    P_map = np.asarray(result["P_map"], dtype=np.float64)

    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))

    for ax, arr, title, cmap in (
        (axes[0, 0], a_map, "a (line/continuum)", "viridis"),
        (axes[0, 1], sigma_map, "sigma [A]", "magma"),
        (axes[1, 0], P_map, "P = a*b*sigma*sqrt(2pi)", "cividis"),
    ):
        vlo, vhi = robust_limits(arr, p_lo=2.0, p_hi=98.0)
        im = ax.imshow(arr, origin="lower", cmap=cmap, vmin=vlo, vmax=vhi)
        ax.set_title(title)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ok = np.isfinite(a_map) & np.isfinite(sigma_map)
    sc = axes[1, 1].scatter(a_map[ok], sigma_map[ok],
                            c=P_map[ok] if np.isfinite(P_map[ok]).any() else None,
                            s=4, cmap="cividis")
    axes[1, 1].set_xlabel("a"); axes[1, 1].set_ylabel("sigma [A]")
    axes[1, 1].set_title(f"a vs sigma (corr={metrics['corr_a_sigma']:.2f}), color=P")
    if np.isfinite(P_map[ok]).any():
        fig.colorbar(sc, ax=axes[1, 1], fraction=0.046, pad=0.04)

    fig.suptitle(
        f"S1 Halpha map — sigma median {metrics['sigma_median_A']:.3f} A, "
        f"a struct {metrics['a_structure_significance']:.1f}x, "
        f"sigma struct {metrics['sigma_structure_significance']:.1f}x, "
        f"corr(a,sigma)={metrics['corr_a_sigma']:.2f}, P cov={metrics['P_cov']:.2f}"
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _write_halpha_map_fits(result: Mapping[str, object], out_path: Path,
                           header_info: Mapping[str, object]) -> None:
    from astropy.io import fits

    primary = fits.PrimaryHDU()
    for key, value in header_info.items():
        primary.header[key] = value
    hdus = [primary]
    for name, key in (("A", "a_map"), ("SIGMA_A", "sigma_map_A"),
                      ("MU_A", "mu_map_A"), ("POWER", "P_map"), ("CONT_B", "b_map")):
        hdus.append(fits.ImageHDU(data=np.asarray(result[key], dtype=np.float32), name=name))
    hdus.append(fits.ImageHDU(data=np.asarray(result["select_mask"], dtype=np.uint8), name="SELECT"))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fits.HDUList(hdus).writeto(out_path, overwrite=True)


def run_phase(args: argparse.Namespace) -> int:
    from musepipe.qc.cube_qc import _load_cube_and_wave, sha256_file, utc_now_iso

    cube_path = Path(args.cube).expanduser()
    if not cube_path.exists():
        print(f"ERROR: cube does not exist: {cube_path}", file=sys.stderr)
        return 2

    t0 = time.time()
    cube, waves = _load_cube_and_wave(cube_path, data_ext=None)
    if args.binning and int(args.binning) > 1:
        cube = bin_cube_spatial(cube, int(args.binning))

    fit_window = tuple(float(v) for v in args.fit_window)
    line_window = tuple(float(v) for v in args.line_window)
    result = compute_halpha_map(
        cube, waves,
        fit_window_A=fit_window, line_window_A=line_window,
        brightness_percentile=args.brightness_percentile, min_snr=args.min_snr,
    )
    metrics = halpha_structure_metrics(result, args.orientation)
    runtime_s = time.time() - t0

    sha = "" if args.skip_checksum else sha256_file(cube_path)
    qc = {
        "stage": "S1_halpha_map",
        "timestamp_utc": utc_now_iso(),
        "cube": str(cube_path),
        "sha256": sha,
        "binning": int(args.binning),
        "orientation": args.orientation,
        "brightness_percentile": float(args.brightness_percentile),
        "min_snr": float(args.min_snr),
        "channel_step_A": float(result["channel_step_A"]),
        "windows": result["windows"],
        "halpha_map": metrics,
        "runtime_s": float(runtime_s),
    }

    if args.map_output:
        map_path = Path(args.map_output)
        _write_halpha_map_fits(result, map_path, {
            "CUBE": str(cube_path)[-68:],
            "BINNING": int(args.binning),
            "STEP_A": float(result["channel_step_A"]),
            "ORIENT": args.orientation,
        })
        qc["map_output"] = str(map_path)

    if args.plot_output:
        plot_path = Path(args.plot_output)
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        _write_halpha_plots(result, metrics, out_path=plot_path)
        qc["plot_output"] = str(plot_path)

    qc_path = Path(args.qc_output)
    qc_path.parent.mkdir(parents=True, exist_ok=True)
    qc_path.write_text(json.dumps(qc, indent=2) + "\n", encoding="utf-8")
    print(
        f"S1 {cube_path.name}: sigma_median={metrics['sigma_median_A']:.3f} A "
        f"(n_fit={metrics['n_fit']}), a_struct={metrics['a_structure_significance']:.1f}x "
        f"sigma_struct={metrics['sigma_structure_significance']:.1f}x "
        f"corr(a,sigma)={metrics['corr_a_sigma']:.2f} P_cov={metrics['P_cov']:.2f} "
        f"| {runtime_s:.0f} s -> {qc_path}"
    )
    return 0


def _resolve_io(args) -> None:
    """Completa cubo y salidas desde --run-id; los flags explícitos mandan.

    Antes estas rutas estaban incrustadas en el comando del notebook, así que
    ejecutarlo desde el set de otro objeto sobrescribía el QC del primero.
    """
    from musepipe.config import resolve_stage_io

    if not args.run_id:
        missing = [n for n in ("cube", "qc_output") if not getattr(args, n)]
        if missing:
            raise SystemExit(
                "Faltan " + ", ".join("--" + m.replace("_", "-") for m in missing)
                + ": pásalos explícitamente o usa --run-id."
            )
        return
    io = resolve_stage_io(args.run_id, 'stageS1', plot_subdir='s1_halpha')
    if not args.cube:
        if not io["cube"]:
            raise SystemExit(
                f"runs/{args.run_id}/config/config.json no declara 'cube_files'; "
                "pasa --cube explícitamente."
            )
        args.cube = io["cube"]
    for attr, key in (("qc_output", "qc_output"), ("map_output", "map_output"),
                      ("plot_output", "plot_output")):
        if not getattr(args, attr, None):
            setattr(args, attr, io[key])


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="S1 per-spaxel Halpha line-to-continuum map (wavesol/stripes plan)."
    )
    parser.add_argument("--run-id", default=None,
                        help="Deriva --cube y las salidas del run indicado "
                             "(config.cube_files + runs/<run>/{stages,plots}).")
    parser.add_argument("--cube", default=None)
    parser.add_argument("--qc-output", default=None)
    parser.add_argument("--map-output", default=None)
    parser.add_argument("--plot-output", default=None)
    parser.add_argument("--orientation", default="vertical", choices=["vertical", "horizontal"])
    parser.add_argument("--binning", type=int, default=1)
    parser.add_argument("--fit-window", nargs=2, type=float, default=list(DEFAULT_FIT_WINDOW_A),
                        metavar=("LO", "HI"))
    parser.add_argument("--line-window", nargs=2, type=float, default=list(DEFAULT_LINE_WINDOW_A),
                        metavar=("LO", "HI"))
    parser.add_argument("--brightness-percentile", type=float, default=DEFAULT_BRIGHTNESS_PERCENTILE)
    parser.add_argument("--min-snr", type=float, default=DEFAULT_MIN_SNR)
    parser.add_argument("--skip-checksum", action="store_true")
    parser.set_defaults(func=run_phase)

    args = parser.parse_args(argv)
    _resolve_io(args)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
