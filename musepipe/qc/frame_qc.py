"""Per-exposure frame quality census (WP-R5, wavesol plan).

Hashimoto et al. (2020) discarded 2 of 6 exposures on frame quality; the referee
may ask about ours. This measures, for each of the 7 per-exposure cubes, three
simple usability metrics and applies the S4a discard criteria WITHOUT discarding
anything (a violation must be reported, not acted on):

  * (a) FWHM of the primary core in the 8000-9000 A band (flux-weighted second
    moments after ring-background subtraction -- the simplest PSF estimator, not
    Psfao);
  * (b) median background in an outer ring;
  * (c) stripe amplitude -- REUSED from the S0-per-exposure summary, not
    recomputed here.

Discard rule (S4a): FWHM > 1.5x the 7-frame median, or background > 2x the
median. Core functions are target-agnostic and I/O-free.
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

from musepipe.qc.ghost_census import band_image

DEFAULT_FWHM_BAND_A = (8000.0, 9000.0)
DEFAULT_BOX_HALF_PX = 10
DEFAULT_RING_INNER_PX = 40.0
DEFAULT_RING_OUTER_PX = 60.0
DEFAULT_FWHM_DISCARD_FACTOR = 1.5
DEFAULT_BG_DISCARD_FACTOR = 2.0
GAUSS_FWHM = 2.3548200450309493  # 2*sqrt(2 ln 2)


def locate_primary(image: np.ndarray, smooth_size: int = 3) -> tuple[int, int]:
    """Return the (y, x) of the brightest source (median-smoothed argmax)."""

    from scipy.ndimage import median_filter

    image = np.asarray(image, dtype=np.float64)
    filled = np.where(np.isfinite(image), image, -np.inf)
    if smooth_size and smooth_size > 1:
        finite = np.where(np.isfinite(image), image, np.nanmedian(image))
        sm = median_filter(finite, size=smooth_size)
        sm = np.where(np.isfinite(image), sm, -np.inf)
    else:
        sm = filled
    iy, ix = np.unravel_index(int(np.argmax(sm)), sm.shape)
    return int(iy), int(ix)


def ring_background(image: np.ndarray, cy: float, cx: float,
                    r_in: float = DEFAULT_RING_INNER_PX,
                    r_out: float = DEFAULT_RING_OUTER_PX) -> float:
    """Median of finite pixels in the annulus [r_in, r_out] around (cy, cx)."""

    image = np.asarray(image, dtype=np.float64)
    ny, nx = image.shape
    yy, xx = np.mgrid[0:ny, 0:nx]
    rr = np.hypot(yy - cy, xx - cx)
    mask = (rr >= r_in) & (rr <= r_out) & np.isfinite(image)
    vals = image[mask]
    if vals.size < 5:
        return float("nan")
    return float(np.nanmedian(vals))


def moment_fwhm(image: np.ndarray, cy: int, cx: int, *,
                box_half: int = DEFAULT_BOX_HALF_PX, background: float = 0.0) -> float:
    """Flux-weighted second-moment FWHM of the core in a box around (cy, cx).

    Background-subtracted, negative weights clipped to zero. Returns the
    geometric-mean FWHM = 2.3548 * sqrt(sigma_y * sigma_x) in pixels.
    """

    image = np.asarray(image, dtype=np.float64)
    ny, nx = image.shape
    y0, y1 = max(0, cy - box_half), min(ny, cy + box_half + 1)
    x0, x1 = max(0, cx - box_half), min(nx, cx + box_half + 1)
    sub = image[y0:y1, x0:x1] - float(background)
    yy, xx = np.mgrid[y0:y1, x0:x1]
    w = np.where(np.isfinite(sub), sub, 0.0)
    w = np.clip(w, 0.0, None)
    total = float(np.sum(w))
    if total <= 0:
        return float("nan")
    my = float(np.sum(w * yy) / total)
    mx = float(np.sum(w * xx) / total)
    sy = float(np.sqrt(max(np.sum(w * (yy - my) ** 2) / total, 0.0)))
    sx = float(np.sqrt(max(np.sum(w * (xx - mx) ** 2) / total, 0.0)))
    return GAUSS_FWHM * float(np.sqrt(sy * sx))


def frame_metrics(cube: np.ndarray, waves: np.ndarray, *,
                  band_A: Sequence[float] = DEFAULT_FWHM_BAND_A,
                  box_half: int = DEFAULT_BOX_HALF_PX,
                  ring_inner_px: float = DEFAULT_RING_INNER_PX,
                  ring_outer_px: float = DEFAULT_RING_OUTER_PX) -> dict:
    """Primary FWHM, ring background and location for one cube."""

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        img = band_image(cube, waves, band_A)
    cy, cx = locate_primary(img)
    bg = ring_background(img, cy, cx, ring_inner_px, ring_outer_px)
    fwhm = moment_fwhm(img, cy, cx, box_half=box_half,
                       background=0.0 if not np.isfinite(bg) else bg)
    return {"primary_yx": [int(cy), int(cx)], "fwhm_px": float(fwhm),
            "background_median": float(bg)}


def apply_discard_criteria(rows: list[dict], *,
                           fwhm_factor: float = DEFAULT_FWHM_DISCARD_FACTOR,
                           bg_factor: float = DEFAULT_BG_DISCARD_FACTOR) -> dict:
    """S4a rule: flag (do not discard) frames with FWHM > fwhm_factor*median or
    background > bg_factor*median. Returns a summary with per-frame flags."""

    fwhm = np.array([r.get("fwhm_px", np.nan) for r in rows], dtype=np.float64)
    bg = np.array([r.get("background_median", np.nan) for r in rows], dtype=np.float64)
    fwhm_med = float(np.nanmedian(fwhm))
    # background can be near zero on sky-subtracted cubes; compare on |bg| via the
    # median magnitude to keep the ratio meaningful.
    bg_med = float(np.nanmedian(np.abs(bg)))
    n_flagged = 0
    for r in rows:
        f = r.get("fwhm_px", np.nan)
        b = r.get("background_median", np.nan)
        fwhm_hi = bool(np.isfinite(f) and fwhm_med > 0 and f > fwhm_factor * fwhm_med)
        bg_hi = bool(np.isfinite(b) and bg_med > 0 and abs(b) > bg_factor * bg_med)
        r["fwhm_over_median"] = float(f / fwhm_med) if fwhm_med > 0 else float("nan")
        r["bg_over_median"] = float(abs(b) / bg_med) if bg_med > 0 else float("nan")
        r["flag_discardable"] = bool(fwhm_hi or bg_hi)
        n_flagged += int(r["flag_discardable"])
    return {"n_frames": len(rows), "n_flagged": int(n_flagged),
            "fwhm_median_px": fwhm_med, "bg_median": bg_med,
            "fwhm_discard_factor": float(fwhm_factor), "bg_discard_factor": float(bg_factor)}


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

DEFAULT_PEREXP_DIR = Path("/mnt/2TB/MUSE_work/ROXs12b_perexp")
DEFAULT_S0_SUMMARY = Path("runs/ROXs12b_realigned/tables/s0_perexp_summary.csv")


def _load_cube(path: Path):
    from astropy.io import fits

    with fits.open(path) as hdul:
        data = None
        for hdu in hdul:
            arr = getattr(hdu, "data", None)
            if arr is not None and arr.ndim >= 3:
                a = np.asarray(arr, dtype=np.float64)
                while a.ndim > 3:
                    a = a[0]
                data = a
                hdr = hdu.header
                break
        if data is None:
            raise ValueError(f"no >=3D cube in {path}")
        waves = None
        if "WAVELENGTH" in hdul:
            waves = np.asarray(hdul["WAVELENGTH"].data, dtype=np.float64).ravel()
        if waves is None:
            crval = float(hdr.get("CRVAL3", 0.0))
            cd = float(hdr.get("CD3_3", hdr.get("CDELT3", 1.0)))
            crpix = float(hdr.get("CRPIX3", 1.0))
            waves = crval + (np.arange(data.shape[0]) - (crpix - 1.0)) * cd
    return data, waves


def _stripe_amplitudes(summary_csv: Path) -> dict:
    out = {}
    if not summary_csv.exists():
        return out
    for row in csv.DictReader(summary_csv.open()):
        exp = row.get("exposure", "")
        try:
            out[exp] = float(row["stripe_sig_vertical"])
        except (KeyError, ValueError, TypeError):
            out[exp] = float("nan")
    return out


def run_phase(args: argparse.Namespace) -> int:
    from musepipe.qc.cube_qc import utc_now_iso

    perexp = Path(args.perexp_dir)
    stripes = _stripe_amplitudes(Path(args.s0_summary))
    rows = []
    for i in range(1, args.n_exp + 1):
        cube_path = perexp / f"exp{i}" / "DATACUBE_FINAL.fits"
        if not cube_path.exists():
            print(f"ERROR: missing {cube_path}", file=sys.stderr)
            return 2
        cube, waves = _load_cube(cube_path)
        m = frame_metrics(cube, waves)
        m["exposure"] = f"exp{i}"
        m["stripe_sig_vertical"] = stripes.get(f"exp{i}", float("nan"))
        rows.append(m)
        print(f"exp{i}: FWHM={m['fwhm_px']:.2f}px bg={m['background_median']:.3g} "
              f"stripe_sig={m['stripe_sig_vertical']:.2f} primary@{m['primary_yx']}")

    summary = apply_discard_criteria(rows)

    if args.table_output:
        tp = Path(args.table_output)
        tp.parent.mkdir(parents=True, exist_ok=True)
        fields = ["exposure", "fwhm_px", "background_median", "stripe_sig_vertical",
                  "fwhm_over_median", "bg_over_median", "flag_discardable", "primary_yx"]
        with tp.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)

    qc = {"stage": "R5_frame_qc", "timestamp_utc": utc_now_iso(),
          "perexp_dir": str(perexp), "s0_summary": str(args.s0_summary),
          "summary": summary, "per_frame": rows,
          "conclusion": (f"{summary['n_frames'] - summary['n_flagged']}/{summary['n_frames']} "
                         "exposures usable; "
                         f"{summary['n_flagged']} flagged by S4a criteria")}
    if args.qc_output:
        qp = Path(args.qc_output)
        qp.parent.mkdir(parents=True, exist_ok=True)
        qp.write_text(json.dumps(qc, indent=2) + "\n", encoding="utf-8")

    print(f"R5: {qc['conclusion']} "
          f"(FWHM median {summary['fwhm_median_px']:.2f}px, "
          f"|bg| median {summary['bg_median']:.3g})")
    if summary["n_flagged"] > 0:
        print("R5: a frame violates the S4a criterion -> STOP and report "
              "(input to reopen S4, human decision).", file=sys.stderr)
        return 3
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="R5 per-exposure frame-QC census.")
    p.add_argument("--perexp-dir", default=str(DEFAULT_PEREXP_DIR))
    p.add_argument("--s0-summary", default=str(DEFAULT_S0_SUMMARY))
    p.add_argument("--n-exp", type=int, default=7)
    p.add_argument("--table-output", default=None)
    p.add_argument("--qc-output", default=None)
    p.set_defaults(func=run_phase)
    args = p.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
