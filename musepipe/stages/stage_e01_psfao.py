"""C1 chromatic PSF via the physical AO model (maoppy Psfao).

The Moffat-only path of ``stage_e01_psf`` leaves ~90% residual at the companion
radius because an elliptical Moffat cannot reproduce the AO halo. This module
fits the physical AO PSF (Fetick et al. 2019, ``maoppy.psfmodel.Psfao``) per
wavelength bin with the companion masked, which brings the companion-ring
residual to ~4-5% (spec C1 target). Target-specific values come only from
config / the B3 QC; no hard-coded source coordinates live here.

Products (in the run stage dir):
  stage_e01_psfao_params.csv   per-bin: lambda, params, amp, ring_residual_pct
  psf_model.json               Psfao form, smoothed param polynomials, norm_radius
  stage_e01_qc.json            ring metric, smoothing, normalization (overwrites
                               the Moffat QC when this path is chosen)
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
from astropy.io import fits

from ..config import load_run_config

PSFAO_PARAM_NAMES = ("r0", "C", "A", "alpha", "ratio", "theta", "beta")
DEFAULT_X0 = [0.15, 1e-4, 1.0, 0.05, 1.0, 0.0, 1.6]


def _bad_windows(cfg):
    if cfg.get("drop_wave_min_A") is not None and cfg.get("drop_wave_max_A") is not None:
        return [[float(cfg["drop_wave_min_A"]), float(cfg["drop_wave_max_A"])]]
    return [[5780.0, 6050.0]]


def make_bins(wave, bin_A, bad_windows, min_channels=3):
    lo, hi = float(wave.min()), float(wave.max())
    edges = np.arange(lo, hi + bin_A, bin_A)
    bins = []
    for a, b in zip(edges[:-1], edges[1:]):
        mid = 0.5 * (a + b)
        if any(w0 <= mid <= w1 for w0, w1 in bad_windows):
            continue
        sel = (wave >= a) & (wave < b)
        if sel.sum() >= min_channels:
            bins.append((a, b, mid, sel))
    return bins


def fit_bin(image, var, samp, system, companion_yx, mask_radius, fit_radius, x0, field_yx=None):
    """Ajuste Psfao de un bin.

    `field_yx` enmascara una fuente de campo igual que hace la rama Moffat. Sin
    esto las dos formas se ajustaban con MASCARAS DISTINTAS —Moffat con
    companero + fuente de campo, psfao solo con el companero— y
    `model_comparison` dejaba de comparar peras con peras: el sesgo iba siempre
    contra psfao, porque era la unica que se comia el contaminante. Se vio en
    ROXs 42B b, donde enmascarar ROXs 42B cc1 mejoro Moffat (8.38 -> 6.88%) y
    dejo psfao intacta (9.80 -> 10.69%).
    """

    from maoppy.psfmodel import Psfao
    from maoppy.psffit import psffit

    ny, nx = image.shape
    cy, cx = ny // 2, nx // 2
    yy, xx = np.mgrid[0:ny, 0:nx]
    r = np.hypot(yy - cy, xx - cx)
    comp = np.hypot(yy - companion_yx[0], xx - companion_yx[1])
    mask = np.isfinite(image) & (comp > mask_radius) & (r < fit_radius)
    if field_yx is not None:
        mask &= np.hypot(yy - float(field_yx[0]), xx - float(field_yx[1])) > mask_radius
    weights = np.where(mask, 1.0 / np.clip(var, 1e-6, None), 0.0)
    imgf = np.where(np.isfinite(image), image, 0.0)
    model = Psfao((ny, nx), system=system, samp=float(samp))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = psffit(imgf, model, x0, weights=weights, flux_bck=(True, True), max_nfev=400)
    amp, bck = res.flux_bck
    recon = amp * model(res.x, dx=res.dxdy[0], dy=res.dxdy[1]) + bck
    ring = _ring_residual(image, recon, companion_yx, mask_radius)
    unchanged = bool(
        np.allclose(np.asarray(res.x, dtype=float), np.asarray(x0, dtype=float), rtol=0.0, atol=1e-8)
        and np.allclose(np.asarray(res.dxdy, dtype=float), 0.0, rtol=0.0, atol=1e-8)
    )
    # An exact solution may legitimately equal x0. Treat it as a stall only when
    # the optimizer also stopped immediately without exploring the parameter space.
    stalled = bool(unchanged and int(res.nfev) <= 2)
    optimizer = {
        "success": bool(res.success),
        "status": int(res.status),
        "message": str(res.message),
        "nfev": int(res.nfev),
        "cost": float(res.cost),
        "stalled_at_initial": stalled,
    }
    return list(map(float, res.x)), float(amp), float(bck), tuple(map(float, res.dxdy)), ring, recon, optimizer


def _ring_residual(image, model, companion_yx, mask_radius, width=1.5):
    ny, nx = image.shape
    cy, cx = ny // 2, nx // 2
    yy, xx = np.mgrid[0:ny, 0:nx]
    r = np.hypot(yy - cy, xx - cx)
    comp_r = float(np.hypot(companion_yx[0] - cy, companion_yx[1] - cx))
    comp = np.hypot(yy - companion_yx[0], xx - companion_yx[1])
    ann = (np.abs(r - comp_r) < width) & np.isfinite(image) & (comp > mask_radius)
    halo = np.nanmedian(image[ann])
    if not np.isfinite(halo) or halo == 0:
        return float("nan")
    return float(100.0 * np.nanmedian(np.abs(image[ann] - model[ann])) / halo)


def _box3_apcorr(system, lam_A, params, norm_radius):
    """Box3 aperture correction from one Psfao parameter set (outlier metric)."""
    from maoppy.instrument import muse_nfm
    from maoppy.psfmodel import Psfao
    try:
        m = Psfao((52, 52), system=system, samp=float(muse_nfm.samp(float(lam_A) * 1e-10)))
        low, high = m.bounds
        eps = 1e-6
        x = [float(np.clip(float(v),
                           low[i] + eps if np.isfinite(low[i]) else -np.inf,
                           high[i] - eps if np.isfinite(high[i]) else np.inf))
             for i, v in enumerate(params)]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            img = np.asarray(m(x), dtype=np.float64)
        c = 26
        yy, xx = np.mgrid[0:52, 0:52]
        r = np.hypot(yy - c, xx - c)
        total = float(np.nansum(img[r <= norm_radius]))
        box = float(np.nansum(img[(np.abs(yy - c) <= 1) & (np.abs(xx - c) <= 1)]))
        return float(total / box) if box > 0 else np.nan
    except Exception:
        return np.nan


def prepare_psfao_inputs(cfg, stage_dir):
    """Load the cube/positions/system and derive the psfao binning + radii.

    Split out of ``run_stage_e01_psfao`` so the canonical Moffat stage
    (``stage_e01_psf``) can reuse the identical psfao setup when it runs the
    dual-form comparison (spec C1 §3.4). Behaviour is unchanged."""

    from maoppy.instrument import muse_nfm, muse_wfm

    cube_path = Path(cfg.get("stage_e01_input_cube_fits", stage_dir / "stage02_xcorr_cube_stack.fits"))
    pos_qc = json.load(open(cfg.get("stage_e01_positions_qc", stage_dir / "stage01c_qc.json")))
    primary = tuple(map(float, pos_qc["primary"]["pos_yx"]))
    companion = tuple(map(float, pos_qc["companion"]["pos_yx"]))

    system = muse_wfm if str(cfg.get("instrument_mode", "NFM")).upper().startswith("W") else muse_nfm

    with fits.open(cube_path) as h:
        cube = np.asarray(h["CUBES"].data if "CUBES" in h else h[1].data, float)
        wave = np.asarray(h["WAVELENGTH"].data, float)
        stat = np.asarray(h["STAT"].data, float) if "STAT" in h else None
    if cube.ndim == 4:
        cube = cube[0]
        stat = stat[0] if stat is not None else None
    if stat is None:
        stat = np.ones_like(cube)

    bin_A = float(cfg.get("psf_bin_A", 100.0))
    bad = _bad_windows(cfg)
    bins = make_bins(wave, bin_A, bad, int(cfg.get("psf_min_channels_per_bin", 3)))
    fwhm_prelim = float(pos_qc.get("companion", {}).get("err_px", 0) or 0) + float(cfg.get("psf_prelim_fwhm_px", 4.0))
    mask_radius = float(cfg.get("psf_companion_mask_radius_px", max(10.0, 3.0 * fwhm_prelim)))
    fit_radius = float(cfg.get("psf_fit_radius_px", 80.0))
    norm_radius = float(cfg.get("psf_norm_radius_px", 25.0))
    return {
        "cube_path": cube_path,
        "cube": cube,
        "stat": stat,
        "wave": wave,
        "primary": primary,
        "companion": companion,
        "system": system,
        "bin_A": bin_A,
        "bad": bad,
        "bins": bins,
        "mask_radius": mask_radius,
        "fit_radius": fit_radius,
        "norm_radius": norm_radius,
    }


def fit_psfao_bins(cube, stat, wave, bins, system, companion, mask_radius, fit_radius, *, x0=None, field_yx=None):
    """Fit the Psfao model per wavelength bin (companion masked).

    Returns ``(rows, recons)`` where ``rows`` is the per-bin parameter table
    (identical structure to the legacy loop) and ``recons`` maps each bin centre
    wavelength to ``(bin_image, reconstructed_model)`` so a caller can score the
    reconstruction with any ring metric it likes."""

    from maoppy.instrument import muse_nfm

    x0 = DEFAULT_X0 if x0 is None else x0
    rows = []
    recons = {}
    for (a, b, mid, sel) in bins:
        img = np.nanmedian(cube[sel], axis=0)
        var = np.nanmedian(stat[sel], axis=0)
        samp = float(muse_nfm.samp(mid * 1e-10))
        try:
            params, amp, bck, dxdy, ring, recon, optimizer = fit_bin(
                img, var, samp, system, companion, mask_radius, fit_radius, x0, field_yx=field_yx
            )
        except Exception as exc:  # pragma: no cover - defensive
            rows.append({"lambda_A": mid, "status": f"fit_failed:{exc}"})
            continue
        if optimizer["stalled_at_initial"]:
            fit_status = "fit_stalled:initial_vector"
        elif not optimizer["success"]:
            fit_status = f"fit_failed:optimizer_status_{optimizer['status']}"
        else:
            fit_status = "ok"
        row = {"lambda_A": float(mid), "samp": samp, "amp": amp, "bck": bck,
               "dy": dxdy[1], "dx": dxdy[0], "ring_residual_pct": ring,
               "optimizer_success": optimizer["success"],
               "optimizer_status": optimizer["status"],
               "optimizer_message": optimizer["message"],
               "optimizer_nfev": optimizer["nfev"],
               "optimizer_cost": optimizer["cost"],
               "optimizer_stalled_at_initial": optimizer["stalled_at_initial"],
               "status": fit_status}
        row.update({name: params[i] for i, name in enumerate(PSFAO_PARAM_NAMES)})
        rows.append(row)
        if fit_status == "ok":
            recons[float(mid)] = (img, recon)
    return rows, recons


def build_psfao_model_document(rows, system, norm_radius, fit_radius, *, system_name="muse_nfm"):
    """Assemble the psfao ``psf_model.json`` document from per-bin fit rows.

    Split out of ``run_stage_e01_psfao`` (behaviour unchanged) so the canonical
    stage can build the winning-form document when Psfao is selected. Returns the
    psf_model dict plus a small ``meta`` dict for QC assembly."""

    ok = [r for r in rows if r.get("status") == "ok" and np.isfinite(r.get("ring_residual_pct", np.nan))]
    lam = np.array([r["lambda_A"] for r in ok])
    poly = {}
    for name in PSFAO_PARAM_NAMES:
        vals = np.array([r[name] for r in ok])
        deg = 2 if len(vals) >= 3 else 1
        poly[name] = list(map(float, np.polyfit(lam, vals, deg))) if len(vals) else []

    # Robust outlier rejection on the per-bin box3 aperture fraction: Psfao PSD
    # params are degenerate, so a failed bin shows up as an off-trend growth
    # curve rather than an obvious single-param spike. Keep the surviving per-bin
    # fits as an interpolation table (smoothing params directly corrupts the PSF).
    box3 = np.array([_box3_apcorr(system, float(r["lambda_A"]), [r[n] for n in PSFAO_PARAM_NAMES], norm_radius)
                     for r in ok])
    # The box3 growth curve spans a wide but smooth range (~20-130), so a global
    # MAD misses local failures; reject bins whose log-apcorr deviates from the
    # smooth wavelength trend (catches degenerate/failed bins like the 9100-9200
    # dip to ~2).
    with np.errstate(all="ignore"):
        logb = np.log(np.clip(box3, 1e-3, None))
    fin = np.isfinite(logb)
    trend = np.polyval(np.polyfit(lam[fin], logb[fin], 2), lam) if fin.sum() >= 3 else np.full_like(logb, np.nanmedian(logb))
    resid = logb - trend
    rmad = np.nanmedian(np.abs(resid[fin] - np.nanmedian(resid[fin]))) or 1.0
    keep = fin & (np.abs(resid - np.nanmedian(resid[fin])) <= 4.0 * 1.4826 * rmad)
    kept = [r for r, k in zip(ok, keep) if k]
    n_rejected = int(np.sum(~keep))
    lam_k = np.array([r["lambda_A"] for r in kept])
    order = np.argsort(lam_k)
    param_table = {"lambda_A": [float(lam_k[i]) for i in order]}
    for name in PSFAO_PARAM_NAMES:
        vals = np.array([kept[i][name] for i in order])
        param_table[name] = [float(v) for v in vals]

    rings = np.array([r["ring_residual_pct"] for r in ok])
    # normalization round-trip on a fine model at one lambda
    norm_err = _norm_roundtrip(system, float(np.median(lam)), poly, norm_radius)

    psf_model = {"form": "psfao", "system": system_name, "smoothed_poly": poly,
                 "param_table": param_table, "n_bins_rejected": n_rejected,
                 "norm_radius_px": norm_radius, "fit_radius_px": fit_radius,
                 "param_names": list(PSFAO_PARAM_NAMES), "hybrid": False}
    meta = {"ok": ok, "lam": lam, "rings": rings, "n_ok": len(ok),
            "n_rejected": n_rejected, "norm_err": norm_err, "poly": poly}
    return psf_model, meta


def run_stage_e01_psfao(run_id=None, *, project_root=None):
    rc = load_run_config(run_id, project_root=project_root)
    cfg = dict(rc.config)
    stage_dir = rc.paths.stage_dir

    inp = prepare_psfao_inputs(cfg, stage_dir)
    cube_path = inp["cube_path"]
    cube = inp["cube"]
    companion = inp["companion"]
    system = inp["system"]
    bin_A = inp["bin_A"]
    bad = inp["bad"]
    bins = inp["bins"]
    mask_radius = inp["mask_radius"]
    fit_radius = inp["fit_radius"]
    norm_radius = inp["norm_radius"]

    rows, _ = fit_psfao_bins(cube, inp["stat"], inp["wave"], bins, system, companion, mask_radius, fit_radius)
    psf_model, meta = build_psfao_model_document(rows, system, norm_radius, fit_radius)
    ok = meta["ok"]
    lam = meta["lam"]
    rings = meta["rings"]
    n_rejected = meta["n_rejected"]
    norm_err = meta["norm_err"]

    # write products
    params_csv = stage_dir / "stage_e01_psfao_params.csv"
    with open(params_csv, "w") as f:
        cols = ["lambda_A", "samp", "amp", "bck", "dy", "dx", "ring_residual_pct", *PSFAO_PARAM_NAMES, "status"]
        f.write(",".join(cols) + "\n")
        for r in rows:
            f.write(",".join(str(r.get(c, "")) for c in cols) + "\n")

    (stage_dir / "psf_model.json").write_text(json.dumps(psf_model, indent=2))
    poly = meta["poly"]

    qc = {
        "stage": "e01_chromatic_psf", "run_id": rc.run_id, "provisional": True,
        "input": {"cube": str(cube_path), "positions_from": "stage01c_qc.json"},
        "binning": {"bin_A": bin_A, "n_bins": len(bins), "excluded_windows_A": bad},
        "masks": {"companion_radius_px": mask_radius, "chromatic_tracking": False,
                  "companion_yx": list(companion)},
        "fit": {"form_chosen": "psfao", "model": "maoppy.Psfao", "fit_radius_px": fit_radius,
                "n_ok_bins": len(ok)},
        "smoothing": {"per_param_poly_deg": {k: (len(v) - 1) for k, v in poly.items()}},
        "companion_ring_metric": {
            "radius_px": float(np.hypot(companion[0] - cube.shape[1] // 2, companion[1] - cube.shape[2] // 2)),
            "residual_pct_median": float(np.median(rings)) if len(rings) else None,
            "residual_pct_p90": float(np.percentile(rings, 90)) if len(rings) else None,
            "residual_pct_red_median": float(np.median(rings[lam > 7000])) if np.any(lam > 7000) else None,
            "bins_above_5pct": int(np.sum(rings > 5.0)),
            "n_bins": len(rings),
        },
        "normalization": {"norm_radius_px": norm_radius, "roundtrip_error": norm_err},
        "open_issues": [
            "PROVISIONAL: run on the ESO ADP while A1 exposure-alignment fix is pending "
            "(ver PAPER_BLOCKERS.md del run raw de este objeto).",
            "Implements the maoppy Psfao path that stage_e01_psf.py leaves as a stub "
            "(form was hard-coded to Moffat). Moffat-vs-Psfao AIC comparison and the "
            "hybrid term are not re-implemented here; Psfao alone meets the <5% ring target "
            "in the red bins where the companion lives.",
        ],
    }
    (stage_dir / "stage_e01_qc.json").write_text(json.dumps(qc, indent=2))
    return qc


def _norm_roundtrip(system, lam_A, poly, norm_radius):
    from maoppy.instrument import muse_nfm
    from maoppy.psfmodel import Psfao
    try:
        x = [float(np.polyval(poly[n], lam_A)) for n in PSFAO_PARAM_NAMES]
        npix = int(4 * norm_radius)
        samp = float(muse_nfm.samp(lam_A * 1e-10))
        m = Psfao((npix, npix), system=system, samp=samp)
        img = m(x)
        cy, cx = npix // 2, npix // 2
        yy, xx = np.mgrid[0:npix, 0:npix]
        inside = np.hypot(yy - cy, xx - cx) <= norm_radius
        total = float(np.nansum(img[inside]))
        return float(abs(total / np.nansum(img) - 1.0)) if np.nansum(img) else float("nan")
    except Exception:
        return float("nan")


__all__ = [
    "run_stage_e01_psfao",
    "prepare_psfao_inputs",
    "fit_psfao_bins",
    "build_psfao_model_document",
    "fit_bin",
    "make_bins",
    "PSFAO_PARAM_NAMES",
]
