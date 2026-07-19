"""Instrumental ghost census at the companion position (WP-R3, wavesol plan).

Blindaje/shielding check for E2: verify and document that the companion position
is free of the two ghost families known for MUSE bright-source fields:

  * LINE GHOST STRIP -- an over-bright IFU/slice strip (a full bright row or
    column in the reconstructed image; Weilbacher et al. 2015). Detected here as
    a >``threshold_sigma`` excess of the median along the companion's row/column
    over the local ring background, in each of a white, an Halpha and a blue
    band image.
  * BLOB GHOST with blue spectral FRINGING (Xie et al. 2020, Appendix A) -- a
    faint patch whose blue continuum shows a periodic fringing pattern. Detected
    here as a peak in the power spectrum of the detrended 4800-5500 A continuum
    (in a 3x3 box at B) exceeding ``peak_ratio`` times the median power.

Core functions are target-agnostic and I/O-free. This is a DIAGNOSTIC only: it
does NOT correct anything. If a ghost is DETECTED the caller must STOP and
report it as a new finding (plan R3).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import warnings
from typing import Sequence

import numpy as np

from musepipe.spectral import continuum_running_median
from musepipe.stats import robust_sigma

# Default bands (A). ``white`` = None -> full spectral mean.
DEFAULT_BANDS = {
    "white": None,
    "halpha": (6540.0, 6590.0),
    "blue": (4800.0, 5500.0),
}
DEFAULT_CORE_RADIUS_PX = 3.0
DEFAULT_RING_INNER_PX = 5.0
DEFAULT_RING_OUTER_PX = 15.0
DEFAULT_STRIP_THRESHOLD_SIGMA = 3.0
DEFAULT_MIN_STRIP_PIXELS = 8
DEFAULT_BLUE_LO_A = 4800.0
DEFAULT_BLUE_HI_A = 5500.0
DEFAULT_FRINGE_PEAK_RATIO = 5.0
DEFAULT_FRINGE_FRACTION = 0.15
DEFAULT_DETREND_WINDOW_A = 80.0
DEFAULT_BOX_HALFWIDTH_PX = 1  # 3x3 box


def band_image(cube: np.ndarray, waves: np.ndarray, band_A) -> np.ndarray:
    """Mean image over the channels in ``band_A`` (``None`` -> full range)."""

    cube = np.asarray(cube, dtype=np.float64)
    if band_A is None:
        sel = np.ones(cube.shape[0], dtype=bool)
    else:
        waves = np.asarray(waves, dtype=np.float64)
        lo, hi = float(band_A[0]), float(band_A[1])
        sel = (waves >= lo) & (waves <= hi)
    if not np.any(sel):
        raise ValueError(f"no channels in band {band_A}")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(cube[sel], axis=0)


def subtract_radial_profile(image: np.ndarray, center_yx: Sequence[float],
                            bin_px: float = 2.0) -> np.ndarray:
    """Subtract the azimuthally-symmetric (radial-median) halo of a source.

    A slice/IFU ghost is a bright line at a FIXED detector direction; the primary
    AO halo is (to first order) azimuthally symmetric. Removing the radial median
    per annulus leaves ghosts and noise while cancelling the halo gradient that
    otherwise contaminates a full row/column profile with a spurious excess.
    """

    image = np.asarray(image, dtype=np.float64)
    ny, nx = image.shape
    cy, cx = float(center_yx[0]), float(center_yx[1])
    yy, xx = np.mgrid[0:ny, 0:nx]
    rr = np.hypot(yy - cy, xx - cx)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        bins = np.arange(0.0, float(np.nanmax(rr)) + bin_px, bin_px)
        idx = np.digitize(rr, bins)
        model = np.full_like(image, np.nan)
        for b in np.unique(idx):
            sel = idx == b
            vals = image[sel]
            finite = vals[np.isfinite(vals)]
            if finite.size >= 5:
                model[sel] = np.nanmedian(finite)
    return image - model


def _annulus_stats(image, cy, cx, r_in, r_out, exclude):
    ny, nx = image.shape
    yy, xx = np.mgrid[0:ny, 0:nx]
    rr = np.hypot(yy - cy, xx - cx)
    mask = (rr >= r_in) & (rr <= r_out) & np.isfinite(image)
    for (ey, ex, er) in exclude:
        mask &= np.hypot(yy - ey, xx - ex) > er
    vals = image[mask]
    if vals.size < 5:
        return np.nan, np.nan, int(vals.size)
    return float(np.nanmedian(vals)), float(robust_sigma(vals)), int(vals.size)


def line_ghost_strip(
    image: np.ndarray,
    companion_yx: Sequence[float],
    *,
    core_radius_px: float = DEFAULT_CORE_RADIUS_PX,
    ring_inner_px: float = DEFAULT_RING_INNER_PX,
    ring_outer_px: float = DEFAULT_RING_OUTER_PX,
    exclude: Sequence[Sequence[float]] = (),
    threshold_sigma: float = DEFAULT_STRIP_THRESHOLD_SIGMA,
    min_strip_pixels: int = DEFAULT_MIN_STRIP_PIXELS,
) -> dict:
    """Row/column strip excess over the local ring at the companion.

    ``exclude`` is a list of (y, x, radius) sources (e.g. the primary) masked
    from both the ring background and the strip pixels. Returns the per-axis
    excess in ring-sigma units and a ``detected`` verdict.
    """

    image = np.asarray(image, dtype=np.float64)
    ny, nx = image.shape
    cy, cx = float(companion_yx[0]), float(companion_yx[1])
    icy, icx = int(round(cy)), int(round(cx))
    ring_med, ring_sig, n_ring = _annulus_stats(
        image, cy, cx, ring_inner_px, ring_outer_px, exclude)

    def _strip_excess(values, coords, axis):
        # axis="row": pixels are (icy, coords) with coords=x; companion excluded
        # near cx. axis="col": pixels are (coords, icx) with coords=y; near cy.
        comp = cx if axis == "row" else cy
        keep = np.isfinite(values) & (np.abs(coords - comp) > core_radius_px)
        for (ey, ex, er) in exclude:
            if axis == "row":
                dist = np.hypot(icy - ey, coords - ex)
            else:
                dist = np.hypot(coords - ey, icx - ex)
            keep &= dist > er
        vals = values[keep]
        if vals.size < min_strip_pixels or not np.isfinite(ring_sig) or ring_sig <= 0:
            return np.nan, int(vals.size)
        return float((np.nanmedian(vals) - ring_med) / ring_sig), int(vals.size)

    row_excess = col_excess = np.nan
    n_row = n_col = 0
    if 0 <= icy < ny:
        row_excess, n_row = _strip_excess(image[icy, :], np.arange(nx), "row")
    if 0 <= icx < nx:
        col_excess, n_col = _strip_excess(image[:, icx], np.arange(ny), "col")

    finite = [e for e in (row_excess, col_excess) if np.isfinite(e)]
    max_excess = max(finite) if finite else np.nan
    detected = bool(np.isfinite(max_excess) and max_excess > threshold_sigma)
    return {
        "row_excess_sigma": row_excess,
        "col_excess_sigma": col_excess,
        "max_excess_sigma": max_excess,
        "ring_median": ring_med,
        "ring_sigma": ring_sig,
        "n_ring": n_ring,
        "n_row_pixels": n_row,
        "n_col_pixels": n_col,
        "threshold_sigma": float(threshold_sigma),
        "detected": detected,
    }


def box_spectrum(cube: np.ndarray, companion_yx: Sequence[float],
                 half_px: int = DEFAULT_BOX_HALFWIDTH_PX) -> np.ndarray:
    """Mean spectrum over a (2*half+1)^2 spatial box at the companion."""

    cube = np.asarray(cube, dtype=np.float64)
    _, ny, nx = cube.shape
    icy, icx = int(round(companion_yx[0])), int(round(companion_yx[1]))
    y0, y1 = max(0, icy - half_px), min(ny, icy + half_px + 1)
    x0, x1 = max(0, icx - half_px), min(nx, icx + half_px + 1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(cube[:, y0:y1, x0:x1], axis=(1, 2))


def blob_fringing(
    waves: np.ndarray,
    spectrum: np.ndarray,
    *,
    lo_A: float = DEFAULT_BLUE_LO_A,
    hi_A: float = DEFAULT_BLUE_HI_A,
    peak_ratio: float = DEFAULT_FRINGE_PEAK_RATIO,
    fraction_threshold: float = DEFAULT_FRINGE_FRACTION,
    detrend_window_A: float = DEFAULT_DETREND_WINDOW_A,
) -> dict:
    """Look for periodic blue-continuum fringing via the residual power spectrum.

    The blue continuum is detrended by a wavelength-window running median; the
    (real) FFT power spectrum of the residual is scanned (excluding the two
    lowest non-DC bins, which carry leftover broad-band curvature).

    DEVIATION from the R3 brief ("no peak > 5x the median power"): a per-channel
    FFT of white noise has an EXPECTED peak/median of ~1.4*ln(M) (~8 for our M),
    so a fixed 5x threshold false-positives on pure noise. We therefore keep
    peak/median as a NECESSARY condition but detect a fringe only when the
    dominant Fourier mode also carries a large fraction of the residual variance
    (``variance_fraction`` > ``fraction_threshold``), which is white-noise robust.
    A real fringe concentrates ~0.5 of the variance; white noise ~ln(M)/M.
    """

    waves = np.asarray(waves, dtype=np.float64)
    spectrum = np.asarray(spectrum, dtype=np.float64)
    sel = (waves >= lo_A) & (waves <= hi_A) & np.isfinite(spectrum) & np.isfinite(waves)
    w, s = waves[sel], spectrum[sel]
    if s.size < 32:
        return {"verdict": "insufficient", "peak_ratio": np.nan, "peak_period_A": np.nan,
                "n_channels": int(s.size), "detected": False}
    good = np.ones(s.size, dtype=bool)
    trend = continuum_running_median(w, s, good, window_A=detrend_window_A, min_pixels=5)
    ok = np.isfinite(trend) & (np.abs(trend) > 0)
    if int(np.sum(ok)) < 32:
        return {"verdict": "insufficient", "peak_ratio": np.nan, "peak_period_A": np.nan,
                "n_channels": int(np.sum(ok)), "detected": False}
    resid = s[ok] / trend[ok] - 1.0
    resid = resid - np.nanmean(resid)
    n = resid.size
    dwave = float(np.nanmedian(np.diff(w[ok])))
    power = np.abs(np.fft.rfft(resid)) ** 2
    freqs = np.fft.rfftfreq(n, d=dwave)
    k_min = 3  # drop DC + 2 lowest bins (broad-band residual curvature)
    if power.size <= k_min + 1:
        return {"verdict": "insufficient", "peak_ratio": np.nan, "peak_period_A": np.nan,
                "n_channels": int(n), "detected": False}
    scan_power = power[k_min:]
    scan_freqs = freqs[k_min:]
    med = float(np.median(scan_power))
    total = float(np.sum(scan_power))
    ipk = int(np.argmax(scan_power))
    peak = float(scan_power[ipk])
    ratio = float(peak / med) if med > 0 else np.inf
    variance_fraction = float(peak / total) if total > 0 else np.nan
    period = float(1.0 / scan_freqs[ipk]) if scan_freqs[ipk] > 0 else np.nan
    detected = bool(ratio > peak_ratio and np.isfinite(variance_fraction)
                    and variance_fraction > fraction_threshold)
    return {"verdict": "detected" if detected else "none", "peak_ratio": ratio,
            "variance_fraction": variance_fraction, "peak_period_A": period,
            "n_channels": int(n), "n_scan_bins": int(scan_power.size),
            "peak_ratio_threshold": float(peak_ratio),
            "fraction_threshold": float(fraction_threshold), "detected": detected}


def compute_ghost_census(
    cube: np.ndarray,
    waves: np.ndarray,
    companion_yx: Sequence[float],
    *,
    primary_yx: Sequence[float] | None = None,
    primary_core_px: float = 6.0,
    bands: dict | None = None,
    strip_kwargs: dict | None = None,
    fringe_kwargs: dict | None = None,
) -> dict:
    """Run the full census: per-band strip test + blue-continuum fringing at B."""

    cube = np.asarray(cube, dtype=np.float64)
    waves = np.asarray(waves, dtype=np.float64)
    if cube.shape[0] != waves.size:
        raise ValueError("cube spectral length must match waves")
    bands = DEFAULT_BANDS if bands is None else bands
    exclude = ()
    if primary_yx is not None:
        exclude = ((float(primary_yx[0]), float(primary_yx[1]), float(primary_core_px)),)
    skw = strip_kwargs or {}
    per_band = {}
    any_strip = False
    for name, band in bands.items():
        img = band_image(cube, waves, band)
        # Remove the primary's azimuthally-symmetric AO halo so a full row/column
        # profile is not biased by the radial gradient (the companion's column can
        # otherwise clip the bright primary halo -> spurious strip). A genuine
        # slice ghost, being a fixed-direction line, survives this subtraction.
        if primary_yx is not None:
            img = subtract_radial_profile(img, primary_yx)
        res = line_ghost_strip(img, companion_yx, exclude=exclude, **skw)
        res["band_A"] = None if band is None else [float(band[0]), float(band[1])]
        per_band[name] = res
        any_strip = any_strip or res["detected"]

    spec = box_spectrum(cube, companion_yx)
    fringe = blob_fringing(waves, spec, **(fringe_kwargs or {}))

    return {
        "line_ghost_strip": "detected" if any_strip else "none",
        "blob_fringing": fringe["verdict"],
        "companion_yx": [float(companion_yx[0]), float(companion_yx[1])],
        "primary_yx": None if primary_yx is None else [float(primary_yx[0]), float(primary_yx[1])],
        "method": ("row/column strip median excess vs local ring (Weilbacher+2015 slice ghost) "
                   "and blue-continuum FFT fringing at B (Xie+2020 App. A)"),
        "thresholds": {
            "strip_threshold_sigma": float(skw.get("threshold_sigma", DEFAULT_STRIP_THRESHOLD_SIGMA)),
            "fringe_peak_ratio": float((fringe_kwargs or {}).get("peak_ratio", DEFAULT_FRINGE_PEAK_RATIO)),
            "fringe_variance_fraction": float((fringe_kwargs or {}).get("fraction_threshold", DEFAULT_FRINGE_FRACTION)),
            "ring_inner_px": float(skw.get("ring_inner_px", DEFAULT_RING_INNER_PX)),
            "ring_outer_px": float(skw.get("ring_outer_px", DEFAULT_RING_OUTER_PX)),
        },
        "strip_by_band": per_band,
        "fringing": fringe,
    }


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def _load_cube(path: Path):
    from astropy.io import fits

    with fits.open(path) as hdul:
        data = None
        for hdu in hdul:
            arr = getattr(hdu, "data", None)
            if arr is not None and arr.ndim >= 3:
                a = np.asarray(arr, dtype=np.float64)
                while a.ndim > 3:  # collapse leading stack axis (n,z,y,x)->(z,y,x)
                    a = a[0]
                data = a
                break
        waves = None
        if "WAVELENGTH" in hdul:
            waves = np.asarray(hdul["WAVELENGTH"].data, dtype=np.float64).ravel()
        if data is None:
            raise ValueError(f"no >=3D cube in {path}")
        if waves is None:
            hdr = hdul[0].header
            crval = float(hdr.get("CRVAL3", 0.0))
            cd = float(hdr.get("CD3_3", hdr.get("CDELT3", 1.0)))
            crpix = float(hdr.get("CRPIX3", 1.0))
            waves = crval + (np.arange(data.shape[0]) - (crpix - 1.0)) * cd
    return data, waves


def _write_figure(cube, waves, qc, out_path: Path) -> None:
    import os

    cache_dir = Path(os.environ.get("TMPDIR", "/tmp")) / "musepipe_matplotlib"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    cy, cx = qc["companion_yx"]
    primary_yx = qc.get("primary_yx")
    fig, axes = plt.subplots(2, 2, figsize=(10, 9))
    for ax, name in zip(axes.flat[:3], ("white", "halpha", "blue")):
        img = band_image(cube, waves, DEFAULT_BANDS[name])
        if primary_yx is not None:
            img = subtract_radial_profile(img, primary_yx)
        vmed = np.nanmedian(img)
        vsig = robust_sigma(img[np.isfinite(img)])
        ax.imshow(img, origin="lower", cmap="magma",
                  vmin=vmed - 2 * vsig, vmax=vmed + 6 * vsig)
        ax.axhline(cy, color="cyan", lw=0.6, alpha=0.7)
        ax.axvline(cx, color="cyan", lw=0.6, alpha=0.7)
        ax.plot(cx, cy, "o", mfc="none", mec="lime", ms=12)
        r = qc["strip_by_band"][name]
        ax.set_title(f"{name}: strip max {r['max_excess_sigma']:.1f}sigma "
                     f"({'DET' if r['detected'] else 'none'})", fontsize=9)
    ax = axes.flat[3]
    spec = box_spectrum(cube, (cy, cx))
    sel = (waves >= DEFAULT_BLUE_LO_A) & (waves <= DEFAULT_BLUE_HI_A)
    ax.plot(waves[sel], spec[sel], color="C0", lw=0.7)
    ax.set_title(f"blue continuum @B: fringe ratio {qc['fringing']['peak_ratio']:.1f} "
                 f"({qc['blob_fringing']})", fontsize=9)
    ax.set_xlabel("wavelength [A]")
    fig.suptitle("R3 ghost census at companion (Xie+2020 App. A)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def run_phase(args: argparse.Namespace) -> int:
    from musepipe.qc.cube_qc import sha256_file, utc_now_iso

    cube_path = Path(args.cube).expanduser()
    if not cube_path.exists():
        print(f"ERROR: missing {cube_path}", file=sys.stderr)
        return 2
    cube, waves = _load_cube(cube_path)
    companion_yx = tuple(float(v) for v in args.companion_yx)
    primary_yx = tuple(float(v) for v in args.primary_yx) if args.primary_yx else None

    qc = compute_ghost_census(cube, waves, companion_yx, primary_yx=primary_yx)
    qc["stage"] = "R3_ghost_census"
    qc["timestamp_utc"] = utc_now_iso()
    qc["cube"] = str(cube_path)
    qc["sha256"] = "" if args.skip_checksum else sha256_file(cube_path)

    if args.plot_output:
        plot_path = Path(args.plot_output)
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        _write_figure(cube, waves, qc, plot_path)
        qc["plot_output"] = str(plot_path)

    if args.qc_output:
        qc_path = Path(args.qc_output)
        qc_path.parent.mkdir(parents=True, exist_ok=True)
        qc_path.write_text(json.dumps(qc, indent=2) + "\n", encoding="utf-8")

    # Additive E2 integration: patch stage_h02_qc.json (nothing renamed).
    if args.e2_qc:
        e2_path = Path(args.e2_qc)
        if e2_path.exists():
            e2 = json.loads(e2_path.read_text())
            e2["ghost_census"] = {k: qc[k] for k in (
                "line_ghost_strip", "blob_fringing", "method", "thresholds",
                "strip_by_band", "fringing", "companion_yx", "cube")}
            figs = e2.setdefault("figures", {})
            if args.plot_output:
                figs["r3_ghost_census"] = str(args.plot_output)
            e2_path.write_text(json.dumps(e2, indent=1) + "\n", encoding="utf-8")

    stop = qc["line_ghost_strip"] == "detected" or qc["blob_fringing"] == "detected"
    print(f"R3 ghost census: line_ghost_strip={qc['line_ghost_strip']} "
          f"blob_fringing={qc['blob_fringing']} "
          f"(strip max sigma by band: "
          + ", ".join(f"{n}={qc['strip_by_band'][n]['max_excess_sigma']:.1f}"
                      for n in qc['strip_by_band'])
          + f"; fringe ratio {qc['fringing']['peak_ratio']:.1f})")
    if stop:
        print("R3: GHOST DETECTED -> STOP and report (new finding, plan R3).", file=sys.stderr)
        return 3
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="R3 instrumental ghost census at the companion.")
    p.add_argument("--cube", required=True)
    p.add_argument("--companion-yx", nargs=2, type=float, required=True, metavar=("Y", "X"))
    p.add_argument("--primary-yx", nargs=2, type=float, default=None, metavar=("Y", "X"))
    p.add_argument("--qc-output", default=None)
    p.add_argument("--plot-output", default=None)
    p.add_argument("--e2-qc", default=None, help="stage_h02_qc.json to patch additively")
    p.add_argument("--skip-checksum", action="store_true")
    p.set_defaults(func=run_phase)
    args = p.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
