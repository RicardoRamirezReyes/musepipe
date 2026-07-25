"""Empirical inter-exposure variance STAT_EMP (WP-R4, wavesol plan).

The MUSE DRS STAT underestimates the true per-voxel variance because drizzle-
style resampling correlates neighbouring voxels (A4 M5: ~4-6x, ESTIMATED). This
measures the ONLY covariance-free variance available: the voxel-by-voxel scatter
BETWEEN the 7 regenerated per-exposure cubes.

Geometry (verified 2026-07-18): the 7 final DATACUBE_FINAL are all North-up
(CD off-diagonal = 0, identical orientation); the derotator angle (ABSROT
-16..+3 deg) was absorbed by scipost into each North-up resampling, so the cubes
differ ONLY by a dither TRANSLATION (up to ~26 px). They are therefore aligned by
an INTEGER pixel shift -- no interpolation, so no NEW resampling covariance is
introduced. The <=0.5 px sub-pixel dither residual adds a small spatial-gradient
term, and seeing/transparency vary between exposures, so STAT_EMP is an UPPER
LIMIT on the per-voxel noise (documented, per the plan note).

Because each exposure's resampling covariance is oriented along its OWN (rotated)
detector axes, the inter-exposure scatter is quasi-independent and captures the
covariance the per-exposure STAT misses. The reported ratio

    ratio = s^2_emp / mean_i(STAT_i)   (== var_of_mean / (mean STAT / n))

turns the ESTIMATED M5 factor into a MEASURED one. Diagnostic only: it does NOT
replace the canonical STAT nor re-run B->F.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
import warnings
from typing import Sequence

import numpy as np

DEFAULT_MIN_N = 4


def voxel_stat_emp(data_stack: np.ndarray, stat_stack: np.ndarray | None = None,
                   *, min_n: int = DEFAULT_MIN_N) -> dict:
    """Per-voxel empirical variance across the exposure axis (axis 0).

    ``data_stack`` shape (n_exp, ...). Returns ``var_of_mean`` = s^2/n (sample
    variance of the mean), ``sample_var`` = s^2, ``n_finite`` and (if ``stat_stack``
    given, the DRS per-exposure variances) the ``ratio`` = s^2 / mean(STAT).
    Voxels with < ``min_n`` finite exposures are NaN.
    """

    data = np.asarray(data_stack, dtype=np.float64)
    n_finite = np.sum(np.isfinite(data), axis=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        sample_var = np.nanvar(data, axis=0, ddof=1)
    valid = n_finite >= int(min_n)
    sample_var = np.where(valid, sample_var, np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        var_of_mean = np.where(valid, sample_var / n_finite, np.nan)
    out = {"var_of_mean": var_of_mean, "sample_var": sample_var,
           "n_finite": n_finite}
    if stat_stack is not None:
        stat = np.asarray(stat_stack, dtype=np.float64)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            mean_stat = np.nanmean(stat, axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            ratio = np.where(valid & np.isfinite(mean_stat) & (mean_stat > 0),
                             sample_var / mean_stat, np.nan)
        out["mean_stat"] = mean_stat
        out["ratio"] = ratio
    return out


def check_wcs_gate(headers: Sequence, *, cd_rtol: float = 1e-6,
                   spectral_atol_A: float = 0.1) -> dict:
    """Gate: all cubes must share orientation/scale (CD matrix), the spectral step
    (CD3_3) and length (NAXIS3), and the spectral zero-point (CRVAL3) to within
    ``spectral_atol_A`` (default 0.1 A ~ 0.08 channel). Pure-translation dithers
    (different CRPIX/CRVAL1/2/NAXIS1/2) and the mAngstrom per-exposure wavelength
    drift measured in S3 (~0.04 A, << 1 channel) are ALLOWED. Returns
    {'pass': bool, 'reason': str, 'max_crval3_offset_A': float}."""

    def cd(h):
        return np.array([[float(h.get("CD1_1", h.get("CDELT1", 1.0))), float(h.get("CD1_2", 0.0))],
                         [float(h.get("CD2_1", 0.0)), float(h.get("CD2_2", h.get("CDELT2", 1.0)))]])

    ref = headers[0]
    cd_ref = cd(ref)
    max_crval3 = 0.0
    for i, h in enumerate(headers):
        if not np.allclose(cd(h), cd_ref, rtol=cd_rtol, atol=0.0):
            return {"pass": False, "reason": f"cube {i} CD matrix (orientation/scale) differs from reference",
                    "max_crval3_offset_A": max_crval3}
        if int(h.get("NAXIS3", -1)) != int(ref.get("NAXIS3", -2)):
            return {"pass": False, "reason": f"cube {i} NAXIS3 differs", "max_crval3_offset_A": max_crval3}
        if not np.isclose(float(h.get("CD3_3", 0.0)), float(ref.get("CD3_3", 1.0)), rtol=1e-8, atol=0.0):
            return {"pass": False, "reason": f"cube {i} CD3_3 (spectral step) differs", "max_crval3_offset_A": max_crval3}
        d = abs(float(h.get("CRVAL3", 0.0)) - float(ref.get("CRVAL3", 0.0)))
        max_crval3 = max(max_crval3, d)
        if d > spectral_atol_A:
            return {"pass": False, "reason": f"cube {i} CRVAL3 offset {d:.4g} A exceeds {spectral_atol_A} A",
                    "max_crval3_offset_A": max_crval3}
    return {"pass": True, "reason": ("orientation + spectral step/length consistent; translation-only "
            f"dithers; max CRVAL3 offset {max_crval3:.4g} A (< {spectral_atol_A} A, S3 drift)"),
            "max_crval3_offset_A": max_crval3}


def plan_alignment(headers: Sequence) -> dict:
    """Integer-shift alignment plan for translation-dithered North-up cubes.

    Uses the celestial WCS to map the reference-cube centre sky point into each
    cube. Returns integer offsets (cube_pixel = ref_pixel + offset), the common
    ref-frame crop box, and per-cube slice boxes. Requires astropy.
    """

    from astropy.wcs import WCS

    ref = headers[0]
    ny_ref, nx_ref = int(ref["NAXIS2"]), int(ref["NAXIS1"])
    wref = WCS(ref).celestial
    yc, xc = ny_ref // 2, nx_ref // 2
    sky = wref.all_pix2world(xc, yc, 0)
    offsets, shapes, subpix = [], [], []
    for h in headers:
        w = WCS(h).celestial
        xi, yi = w.all_world2pix(sky[0], sky[1], 0)
        dy, dx = float(yi - yc), float(xi - xc)
        offsets.append((int(round(dy)), int(round(dx))))
        subpix.append((dy - round(dy), dx - round(dx)))
        shapes.append((int(h["NAXIS2"]), int(h["NAXIS1"])))
    # common ref-frame rows/cols valid in every cube
    y0 = max(-dy for (dy, _dx) in offsets)
    y1 = min(shapes[i][0] - offsets[i][0] for i in range(len(offsets)))
    x0 = max(-dx for (_dy, dx) in offsets)
    x1 = min(shapes[i][1] - offsets[i][1] for i in range(len(offsets)))
    if y1 <= y0 or x1 <= x0:
        raise RuntimeError("no common spatial overlap between the dithered cubes")
    per_cube = [(y0 + dy, y1 + dy, x0 + dx, x1 + dx) for (dy, dx) in offsets]
    return {"offsets": offsets, "subpixel_residual": subpix,
            "ref_crop": (int(y0), int(y1), int(x0), int(x1)),
            "per_cube_slices": per_cube, "overlap_shape": (int(y1 - y0), int(x1 - x0))}


def pooled_ratio(sample_var: np.ndarray, mean_stat: np.ndarray) -> float:
    """Field-pooled ratio sum(s^2) / sum(mean STAT) over valid voxels.

    Unlike the per-voxel median, this is UNBIASED for small n (the sample
    variance is unbiased in the mean; the per-voxel median is biased low by the
    chi-square skew, ~0.89 for n=7)."""

    s = np.asarray(sample_var, dtype=np.float64)
    m = np.asarray(mean_stat, dtype=np.float64)
    ok = np.isfinite(s) & np.isfinite(m) & (m > 0)
    if not np.any(ok):
        return float("nan")
    denom = float(np.sum(m[ok]))
    return float(np.sum(s[ok]) / denom) if denom > 0 else float("nan")


def median_bias_factor(n_exp: int) -> float:
    """Small-sample factor between the MEDIAN and the mean of a sample variance:
    median(chi2_{n-1}) / (n-1). The per-voxel median ratio is LOW by this factor;
    dividing by it de-biases. n<=1 -> NaN."""

    from scipy.stats import chi2

    dof = int(n_exp) - 1
    if dof < 1:
        return float("nan")
    return float(chi2.median(dof) / dof)


def _global_ratio_stats(per_channel: list, *, n_exp: int | None = None) -> dict:
    """Two headline numbers over channels: the TYPICAL (background) voxel via the
    per-channel median, and the FLUX-WEIGHTED via the pooled ratio (upper bound,
    inflated by bright-source seeing/transparency variation)."""

    med = np.array([r.get("ratio_median", np.nan) for r in per_channel], dtype=np.float64)
    pool = np.array([r.get("ratio_pooled", np.nan) for r in per_channel], dtype=np.float64)
    med = med[np.isfinite(med)]
    pool = pool[np.isfinite(pool)]
    bias = median_bias_factor(n_exp) if n_exp else float("nan")
    typ = float(np.median(med)) if med.size else float("nan")
    return {
        "ratio_typical_voxel_median": typ,
        "ratio_typical_bias_corrected": (typ / bias) if (np.isfinite(bias) and bias > 0) else float("nan"),
        "ratio_typical_p16": float(np.percentile(med, 16)) if med.size else float("nan"),
        "ratio_typical_p84": float(np.percentile(med, 84)) if med.size else float("nan"),
        "ratio_flux_weighted_pooled": float(np.median(pool)) if pool.size else float("nan"),
        "ratio_pooled_p16": float(np.percentile(pool, 16)) if pool.size else float("nan"),
        "ratio_pooled_p84": float(np.percentile(pool, 84)) if pool.size else float("nan"),
        "median_bias_factor": bias,
        "n_channels": int(med.size),
    }


def aggregate_ratio(ratio_map: np.ndarray) -> dict:
    """Global median and p16/p84 of a per-voxel ratio map (finite voxels)."""

    vals = np.asarray(ratio_map, dtype=np.float64)
    vals = vals[np.isfinite(vals) & (vals > 0)]
    if vals.size == 0:
        return {"median": float("nan"), "p16": float("nan"), "p84": float("nan"), "n": 0}
    return {"median": float(np.median(vals)), "p16": float(np.percentile(vals, 16)),
            "p84": float(np.percentile(vals, 84)), "n": int(vals.size)}


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

# Sin default por objeto: los cubos por exposición salen de `perexp_cubes` (o de
# `perexp_dir`) en el config del run, o del flag --cubes.


def _channel_wave(header, k):
    crval = float(header.get("CRVAL3", 0.0))
    cd = float(header.get("CD3_3", header.get("CDELT3", 1.0)))
    crpix = float(header.get("CRPIX3", 1.0))
    return crval + (k - (crpix - 1.0)) * cd


def run_phase(args: argparse.Namespace) -> int:
    from astropy.io import fits
    from musepipe.qc.cube_qc import utc_now_iso

    paths = [Path(p) for p in args.cubes]
    for p in paths:
        if not p.exists():
            print(f"ERROR: missing {p}", file=sys.stderr)
            return 2
    hduls = [fits.open(p, memmap=True) for p in paths]
    try:
        data_hdus = [h["DATA"] if "DATA" in h else h[1] for h in hduls]
        stat_hdus = [h["STAT"] if "STAT" in h else None for h in hduls]
        headers = [hdu.header for hdu in data_hdus]
        gate = check_wcs_gate(headers)
        if not gate["pass"]:
            print(f"R4 WCS gate FAILED: {gate['reason']}", file=sys.stderr)
            return 3
        plan = plan_alignment(headers)
        ny, nx = plan["overlap_shape"]
        nz = int(headers[0]["NAXIS3"])
        ref_hdr = headers[0]
        y0r, _y1r, x0r, _x1r = plan["ref_crop"]

        # output STAT_EMP header: ref celestial WCS shifted by the crop origin
        out_hdr = ref_hdr.copy()
        out_hdr["NAXIS1"], out_hdr["NAXIS2"] = nx, ny
        out_hdr["CRPIX1"] = float(ref_hdr.get("CRPIX1", 1.0)) - x0r
        out_hdr["CRPIX2"] = float(ref_hdr.get("CRPIX2", 1.0)) - y0r

        var_cube = np.full((nz, ny, nx), np.nan, dtype=np.float32) if args.write_fits else None
        per_channel = []
        block = int(args.block_channels)
        ratio_accum = []
        for k0 in range(0, nz, block):
            k1 = min(nz, k0 + block)
            dstack = np.empty((len(paths), k1 - k0, ny, nx), dtype=np.float64)
            sstack = np.empty_like(dstack)
            has_stat = all(s is not None for s in stat_hdus)
            for i, (dh, sh) in enumerate(zip(data_hdus, stat_hdus)):
                ya, yb, xa, xb = plan["per_cube_slices"][i]
                dstack[i] = np.asarray(dh.data[k0:k1, ya:yb, xa:xb], dtype=np.float64)
                if has_stat:
                    sstack[i] = np.asarray(sh.data[k0:k1, ya:yb, xa:xb], dtype=np.float64)
            res = voxel_stat_emp(dstack, sstack if has_stat else None, min_n=args.min_n)
            if var_cube is not None:
                var_cube[k0:k1] = res["var_of_mean"].astype(np.float32)
            if has_stat:
                for j in range(k1 - k0):
                    agg = aggregate_ratio(res["ratio"][j])
                    pooled = pooled_ratio(res["sample_var"][j], res["mean_stat"][j])
                    per_channel.append({"channel": k0 + j,
                                        "wave_A": round(_channel_wave(ref_hdr, k0 + j), 3),
                                        "ratio_pooled": pooled,
                                        "ratio_median": agg["median"], "n_voxels": agg["n"]})
                    if np.isfinite(pooled):
                        ratio_accum.append(pooled)
            print(f"  channels {k0}-{k1}: done", file=sys.stderr)

        global_stats = _global_ratio_stats(per_channel, n_exp=len(paths))

        if args.write_fits and args.output:
            outp = Path(args.output)
            outp.parent.mkdir(parents=True, exist_ok=True)
            hdu = fits.PrimaryHDU(data=var_cube, header=out_hdr)
            hdu.header["EXTNAME"] = "STAT_EMP"
            hdu.header["BUNIT"] = "(flux)**2"
            hdu.header["COMMENT"] = "Empirical inter-exposure variance of the mean (R4/STAT_EMP)"
            hdu.writeto(outp, overwrite=True)

        qc = {"stage": "R4_stat_emp", "timestamp_utc": utc_now_iso(),
              "cubes": [str(p) for p in paths], "n_exposures": len(paths),
              "gate_wcs": gate, "alignment": {"offsets": plan["offsets"],
              "subpixel_residual": [[round(a, 3), round(b, 3)] for a, b in plan["subpixel_residual"]],
              "overlap_shape": plan["overlap_shape"], "method": "integer pixel shift (no interpolation)"},
              "min_n": int(args.min_n), **global_stats,
              "note": ("ratio = s^2_emp / mean_i(STAT_i) per voxel. TWO headline numbers: "
                       "ratio_typical_voxel_median (background/typical voxel, per-channel median) "
                       "and ratio_flux_weighted_pooled (sum s^2 / sum STAT, dominated by bright "
                       "voxels whose scatter is inflated by seeing/transparency variation -> upper "
                       "bound). Both are UPPER LIMITS (sub-pixel dither residual + source variability). "
                       "Diagnostic; does NOT replace the canonical STAT."),
              "per_channel_downsampled": per_channel[::max(1, len(per_channel) // 200)]}
        if args.qc_output:
            qp = Path(args.qc_output)
            qp.parent.mkdir(parents=True, exist_ok=True)
            qp.write_text(json.dumps(qc, indent=2) + "\n", encoding="utf-8")
        if args.table_output and per_channel:
            tp = Path(args.table_output)
            tp.parent.mkdir(parents=True, exist_ok=True)
            with tp.open("w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=["channel", "wave_A", "ratio_pooled", "ratio_median", "n_voxels"])
                w.writeheader()
                w.writerows(per_channel)
        if args.plot_output and per_channel:
            _write_figure(per_channel, global_stats, Path(args.plot_output))

        print(f"R4 STAT_EMP: DRS STAT underestimates the per-voxel variance by "
              f"x{global_stats['ratio_typical_voxel_median']:.2f} at the typical (background) voxel "
              f"({global_stats['ratio_typical_p16']:.2f}-{global_stats['ratio_typical_p84']:.2f}; "
              f"bias-corrected x{global_stats['ratio_typical_bias_corrected']:.2f}), "
              f"up to x{global_stats['ratio_flux_weighted_pooled']:.2f} flux-weighted "
              f"({len(paths)} exposures, gate {gate['pass']})")
        return 0
    finally:
        for h in hduls:
            h.close()


def _write_figure(per_channel, global_stats, out_path: Path) -> None:
    import os

    cache_dir = Path(os.environ.get("TMPDIR", "/tmp")) / "musepipe_matplotlib"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    waves = [r["wave_A"] for r in per_channel]
    pooled = [r["ratio_pooled"] for r in per_channel]
    median = [r.get("ratio_median", np.nan) for r in per_channel]
    typ = global_stats["ratio_typical_voxel_median"]
    fw = global_stats["ratio_flux_weighted_pooled"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    ax1.plot(waves, pooled, color="C1", lw=0.5, alpha=0.7, label="flux-weighted (pooled)")
    ax1.plot(waves, median, color="C0", lw=0.5, label="typical voxel (median)")
    ax1.axhline(typ, color="tab:blue", ls="--", label=f"typical x{typ:.2f}")
    ax1.axhline(fw, color="tab:orange", ls="--", label=f"flux-weighted x{fw:.2f}")
    ax1.set_xlabel("wavelength [A]"); ax1.set_ylabel("s^2_emp / mean(STAT_DRS)")
    ax1.set_title("R4 STAT_EMP: DRS STAT underestimation vs wavelength")
    ax1.legend(fontsize=7)
    fin = np.array([r for r in median if np.isfinite(r)])
    ax2.hist(fin, bins=40, color="C0", alpha=0.8)
    ax2.axvline(typ, color="tab:blue", ls="--")
    ax2.set_xlabel("per-channel median ratio (typical voxel)"); ax2.set_ylabel("channels")
    ax2.set_title("distribution (background)")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(); fig.savefig(out_path, dpi=120); plt.close(fig)


def _resolve_cubes(args) -> None:
    """Completa --cubes desde el config del run (`perexp_cubes` o `perexp_dir`)."""
    from musepipe.config import run_workdir_setting
    from pathlib import Path as _P
    _root = _P(__file__).resolve().parent.parent.parent

    if args.cubes:
        return
    cubes = run_workdir_setting(args.run_id, "perexp_cubes", project_root=_root) if args.run_id else None
    if not cubes:
        perexp_dir = run_workdir_setting(args.run_id, "perexp_dir", project_root=_root) if args.run_id else None
        if perexp_dir:
            cubes = sorted(str(p) for p in Path(perexp_dir).glob("exp*/DATACUBE_FINAL.fits"))
    if not cubes:
        raise SystemExit(
            "Falta --cubes: pásalos explícitamente o declara 'perexp_cubes' (o "
            "'perexp_dir') en runs/<run>/config/config.json y pasa --run-id."
        )
    args.cubes = list(cubes)


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="R4 empirical inter-exposure variance (STAT_EMP).")
    p.add_argument("--run-id", default=None,
                   help="Run del que leer perexp_cubes/perexp_dir si no se pasa --cubes.")
    p.add_argument("--cubes", nargs="+", default=None)
    p.add_argument("--output", default=None, help="STAT_EMP.fits path")
    p.add_argument("--qc-output", default=None)
    p.add_argument("--table-output", default=None)
    p.add_argument("--plot-output", default=None)
    p.add_argument("--min-n", type=int, default=DEFAULT_MIN_N)
    p.add_argument("--block-channels", type=int, default=64)
    p.add_argument("--write-fits", action="store_true", help="write the full STAT_EMP cube")
    p.set_defaults(func=run_phase)
    args = p.parse_args(argv)
    _resolve_cubes(args)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
