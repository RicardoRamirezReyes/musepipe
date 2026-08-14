"""Stage E01/C1: chromatic Moffat PSF model for extraction stages."""

from __future__ import annotations

import argparse
import csv
import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from astropy.io import fits

from ..config import load_run_config
from .stage_e01_psfao import PSFAO_DEFAULT_WEIGHT_CAP, PSFAO_DEFAULT_WEIGHTING
from ..io import read_json, write_json
from ..paths import RunPaths
from ..psf import (
    build_psf_model_document,
    companion_ring_metric,
    corner_background,
    evaluate_moffat_fit,
    evaluate_radial_profile,
    fit_moffat_image,
    psf_roundtrip_error,
    radial_hybrid_profile,
    source_mask,
)


@dataclass(frozen=True)
class StageE01Product:
    fit_rows: list[dict]
    psf_model: dict
    qc: dict
    hybrid_profiles: np.ndarray | None
    hybrid_radii: np.ndarray | None
    psf_form: str = "moffat"
    psfao_rows: list[dict] | None = None


def stage_e01_paths(run_id, project_root=None):
    root = Path(project_root or Path.cwd()).resolve()
    paths = RunPaths.from_project_root(run_id, root)
    plot_dir = paths.plot_stage_dir("stage_e01")
    return {
        "paths": paths,
        "stage02_cube_fits": paths.stage_dir / "stage02_xcorr_cube_stack.fits",
        "stage01c_qc_json": paths.stage_dir / "stage01c_qc.json",
        "stage_e01_params_csv": paths.stage_dir / "stage_e01_psf_params.csv",
        "psf_model_json": paths.stage_dir / "psf_model.json",
        "stage_e01_qc_json": paths.stage_dir / "stage_e01_qc.json",
        "psf_hybrid_residual_fits": paths.stage_dir / "psf_hybrid_residual.fits",
        "plot_dir": plot_dir,
        "summary_plot": plot_dir / "stage_e01_psf_summary.png",
    }


def stage_e01_config_from_run(
    run_id=None,
    *,
    project_root=None,
    overrides=None,
    allow_run_id_mismatch=False,
):
    run_config = load_run_config(
        run_id,
        project_root=project_root,
        allow_run_id_mismatch=allow_run_id_mismatch,
    )
    cfg = dict(run_config.config)
    if overrides:
        cfg.update(overrides)
    cfg["run_id"] = run_config.run_id
    cfg["project_root"] = str(run_config.paths.project_root)
    cfg.setdefault("stage_e01_input_cube_fits", str(run_config.paths.stage_dir / "stage02_xcorr_cube_stack.fits"))
    cfg.setdefault("stage_e01_positions_qc", str(run_config.paths.stage_dir / "stage01c_qc.json"))
    cfg.setdefault("e01_psf_form", "auto")
    cfg.setdefault("psf_bin_A", 100.0)
    cfg.setdefault("psf_min_channels_per_bin", 3)
    cfg.setdefault("psf_fit_radius_px", 28.0)
    cfg.setdefault("psf_norm_radius_px", 25.0)
    cfg.setdefault("psf_companion_ring_width_px", 3.0)
    cfg.setdefault("psf_mask_radius_factor", 3.0)
    cfg.setdefault("psf_sigma_clip", 3.0)
    cfg.setdefault("psf_max_clip_iter", 3)
    cfg.setdefault("psf_hybrid_threshold_pct", 5.0)
    cfg.setdefault("psf_hybrid_bin_fraction", 0.2)
    cfg.setdefault("psf_hybrid_smoothing_scale_factor", 2.0)
    cfg.setdefault("psf_save_plots", bool(cfg.get("save_intermediate_plots", False)))
    cfg.setdefault("stage_e01_bad_windows_A", _bad_windows_from_config(cfg))
    return cfg


def _bad_windows_from_config(cfg):
    windows = cfg.get("stage_e01_bad_windows_A") or cfg.get("bad_wavelength_ranges_A")
    if windows is not None:
        return windows
    if cfg.get("drop_wave_min_A") is not None and cfg.get("drop_wave_max_A") is not None:
        return [[float(cfg["drop_wave_min_A"]), float(cfg["drop_wave_max_A"])]]
    return []


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _maoppy_status(form_chosen=None):
    """Estado de maoppy EN ESTA EJECUCION, no solo si se puede importar.

    Antes devolvia siempre ``"available_not_used"`` con que el import
    funcionara, sin mirar si la rama psfao se habia elegido. Era una etiqueta de
    cuando maoppy era opcional y no se usaba, y quedaba contradiciendo a
    ``fit.model = "maoppy.Psfao"`` y ``model_comparison.form_chosen = "psfao"``
    en el mismo QC: quien auditara el fichero leia justo lo contrario de la
    verdad.
    """

    try:
        import maoppy  # noqa: F401
    except Exception as exc:
        return f"unavailable:{exc.__class__.__name__}"
    if form_chosen is None:
        return "available"
    return "used_selected" if str(form_chosen).lower() == "psfao" else "available_not_selected"


def _load_cube(path):
    with fits.open(path, memmap=True) as hdul:
        cubes = hdul["CUBES"].data.astype(np.float32)
        wavelengths = hdul["WAVELENGTH"].data.astype(np.float64)
        stat = hdul["STAT"].data.astype(np.float32) if "STAT" in hdul else None
    if cubes.ndim != 4:
        raise RuntimeError(f"Expected CUBES shape (N,nz,ny,nx), got {cubes.shape}.")
    return cubes, wavelengths, stat


def _good_wave_mask(wavelengths, bad_windows_A):
    wave = np.asarray(wavelengths, dtype=np.float64)
    good = np.isfinite(wave)
    for lo, hi in bad_windows_A or ():
        good &= ~((wave >= float(lo)) & (wave <= float(hi)))
    return good


def make_psf_bins(wavelengths, *, bin_A=100.0, bad_windows_A=(), min_channels=3):
    wave = np.asarray(wavelengths, dtype=np.float64)
    good = _good_wave_mask(wave, bad_windows_A)
    if np.count_nonzero(good) < int(min_channels):
        raise RuntimeError("Too few good wavelength channels for PSF bins.")
    wmin = float(np.nanmin(wave[good]))
    wmax = float(np.nanmax(wave[good]))
    edges = np.arange(wmin, wmax + float(bin_A), float(bin_A))
    bins = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = good & (wave >= lo) & (wave < hi)
        if np.count_nonzero(mask) >= int(min_channels):
            bins.append(
                {
                    "wave_min_A": float(lo),
                    "wave_max_A": float(hi),
                    "wave_center_A": float(np.nanmedian(wave[mask])),
                    "indices": np.where(mask)[0],
                }
            )
    if not bins:
        raise RuntimeError("No valid PSF bins after wavelength masking.")
    return bins


def _median_image(cubes, indices):
    with np.errstate(all="ignore"):
        return np.nanmedian(cubes[:, indices, :, :], axis=(0, 1)).astype(np.float64)


def _positions_from_qc(qc):
    primary = tuple(map(float, qc["primary"]["pos_yx"]))
    companion = tuple(map(float, qc["companion"]["pos_yx"]))
    field = None
    if qc.get("field_source") is not None:
        field = tuple(map(float, qc["field_source"]["pos_yx"]))
    return primary, companion, field


def _b3_chromatic_track(path):
    path = Path(path)
    if not path.exists():
        return None, "unavailable:missing_track"
    rows = []
    with open(path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            try:
                rows.append((
                    0.5 * (float(row["wave_min_A"]) + float(row["wave_max_A"])),
                    float(row["primary_y"]),
                    float(row["primary_x"]),
                ))
            except (KeyError, TypeError, ValueError):
                continue
    if not rows:
        return None, "unavailable:empty_or_invalid_track"
    rows.sort(key=lambda value: value[0])
    return np.asarray(rows, dtype=np.float64), "used"


def _centroid_vs_b3(rows, form, track, image_shape=None):
    if track is None:
        return None
    wave_ref, y_ref, x_ref = track.T
    if form == "psfao":
        if image_shape is None:
            raise ValueError("image_shape is required for Psfao absolute centroids.")
        center_y, center_x = image_shape[0] // 2, image_shape[1] // 2
        values = [
            (float(row["lambda_A"]), center_y + float(row["dy"]), center_x + float(row["dx"]))
            for row in rows if row.get("status") == "ok"
        ]
    else:
        values = [
            (float(row["wave_center_A"]), float(row["y0"]), float(row["x0"]))
            for row in rows if row.get("success", True)
        ]
    if not values:
        return None
    values = np.asarray(values, dtype=np.float64)
    expected_y = np.interp(values[:, 0], wave_ref, y_ref)
    expected_x = np.interp(values[:, 0], wave_ref, x_ref)
    return float(np.nanmax(np.hypot(values[:, 1] - expected_y, values[:, 2] - expected_x)))


def _ring_qc_summary(ring_after, p90_values, cfg):
    ring_after = np.asarray(ring_after, dtype=np.float64)
    p90_values = np.asarray(p90_values, dtype=np.float64)
    threshold = float(cfg.get("psf_hybrid_threshold_pct", 5.0))
    bins_above = int(np.count_nonzero(ring_after > threshold))
    fail_fraction = float(bins_above / max(ring_after.size, 1))
    return {
        "median": float(np.nanmedian(ring_after)) if ring_after.size else float("nan"),
        "p90": float(np.nanpercentile(p90_values, 90)) if p90_values.size else float("nan"),
        "bins_above": bins_above,
        "issue": fail_fraction > float(cfg.get("psf_hybrid_bin_fraction", 0.2)),
    }


def _fit_source_mask(shape, primary_yx, companion_yx, field_yx, mask_radius_px):
    centers = [companion_yx]
    if field_yx is not None:
        centers.append(field_yx)
    return source_mask(shape, centers, float(mask_radius_px))


def _detect_core_mask(image, primary_yx, cfg):
    threshold = cfg.get("psf_saturation_threshold")
    if threshold is None:
        return 0.0, False
    y, x = map(int, map(round, primary_yx))
    peak = float(image[y, x])
    if peak > float(threshold):
        return float(cfg.get("psf_core_mask_px", 2.0)), True
    return 0.0, False


def _row_from_fit(bin_index, bin_info, fit, metric, metric_hybrid=None):
    p = fit.params
    row = {
        "bin_index": int(bin_index),
        "wave_min_A": float(bin_info["wave_min_A"]),
        "wave_max_A": float(bin_info["wave_max_A"]),
        "wave_center_A": float(bin_info["wave_center_A"]),
        "n_channels": int(len(bin_info["indices"])),
        "success": bool(fit.success),
        "background": float(fit.background),
        "amplitude": float(p["amplitude"]),
        "y0": float(p["y0"]),
        "x0": float(p["x0"]),
        "fwhm_maj": float(p["fwhm_maj"]),
        "fwhm_min": float(p["fwhm_min"]),
        "theta_deg": float(p["theta_deg"]),
        "beta": float(p["beta"]),
        "chi2r": float(fit.chi2r),
        "clip_frac": float(fit.clip_frac),
        "n_fit": int(fit.n_fit),
        "ring_residual_pct": float(metric["median_pct"]),
        "ring_residual_p90_pct": float(metric["p90_pct"]),
    }
    for key, value in fit.errors.items():
        row[f"{key}_err"] = None if not np.isfinite(value) else float(value)
    if metric_hybrid is not None:
        row["ring_residual_pct_after_hybrid"] = float(metric_hybrid["median_pct"])
        row["ring_residual_p90_pct_after_hybrid"] = float(metric_hybrid["p90_pct"])
    return row


def _moffat_fit_rows(cubes, wavelengths, bins, positions_qc, cfg):
    primary_yx, companion_yx, field_yx = _positions_from_qc(positions_qc)
    fwhm_prelim = float(positions_qc.get("psf", {}).get("fwhm_px", cfg.get("psf_prelim_fwhm_px", 4.0)))
    mask_radius = float(cfg.get("psf_companion_mask_radius_px", cfg.get("psf_mask_radius_factor", 3.0) * fwhm_prelim))
    rows = []
    images = []
    models = []
    masks = []
    core_masks = []
    for i, bin_info in enumerate(bins):
        image = _median_image(cubes, bin_info["indices"])
        mask = _fit_source_mask(image.shape, primary_yx, companion_yx, field_yx, mask_radius)
        core_mask_px, saturation = _detect_core_mask(image, primary_yx, cfg)
        background = corner_background(image)
        fit = fit_moffat_image(
            image,
            center_yx=primary_yx,
            fit_radius_px=float(cfg.get("psf_fit_radius_px", 28.0)),
            mask=mask,
            background=background,
            core_mask_px=core_mask_px,
            sigma_clip=cfg.get("psf_sigma_clip", 3.0),
            max_iter=int(cfg.get("psf_max_clip_iter", 3)),
        )
        model = evaluate_moffat_fit(image.shape, fit)
        metric = companion_ring_metric(
            image,
            model,
            primary_yx,
            companion_yx,
            width_px=float(cfg.get("psf_companion_ring_width_px", 3.0)),
            source_exclusion_radius_px=mask_radius,
        )
        row = _row_from_fit(i, bin_info, fit, metric)
        row["core_mask_px"] = float(core_mask_px)
        row["saturation_detected"] = bool(saturation)
        rows.append(row)
        images.append(image)
        models.append(model)
        masks.append(mask)
        core_masks.append(core_mask_px)
    return rows, images, models, masks, {
        "mask_radius_px": mask_radius,
        "core_mask_px_max": float(np.nanmax(core_masks)) if core_masks else 0.0,
        "saturation_detected": bool(any(row["saturation_detected"] for row in rows)),
    }


def _apply_hybrid(ring_pcts, images, models, masks, primary_yx, companion_yx, fwhm_med, cfg):
    """Add the azimuthal-median residual (AO ring) hybrid term to the chosen
    form's per-bin models when the ring metric fails on >20% of bins (spec §3.5).

    Form-agnostic: ``models`` are per-bin model images (Moffat evaluations or
    Psfao reconstructions). The smoothing scale (>=2*FWHM) and the azimuthal
    symmetry of ``radial_hybrid_profile`` guarantee it cannot absorb the
    (masked) companion. Returns
    ``(models_hybrid, profiles, radii_ref, applied, after_pcts)``."""

    threshold = float(cfg.get("psf_hybrid_threshold_pct", 5.0))
    frac_limit = float(cfg.get("psf_hybrid_bin_fraction", 0.2))
    values = np.asarray(ring_pcts, dtype=np.float64)
    fail_frac = float(np.count_nonzero(values > threshold) / max(values.size, 1))
    if fail_frac <= frac_limit:
        return list(models), None, None, False, list(map(float, ring_pcts)), None

    fwhm_med = float(fwhm_med)
    smooth = max(2.0 * fwhm_med, float(cfg.get("psf_hybrid_smoothing_scale_factor", 2.0)) * fwhm_med)
    width = float(cfg.get("psf_companion_ring_width_px", 3.0))
    excl = float(cfg.get("psf_companion_mask_radius_px", cfg.get("psf_mask_radius_factor", 3.0) * fwhm_med))
    profiles = []
    radii_ref = None
    models_hybrid = []
    after_pcts = []
    after_p90s = []
    for image, model, mask in zip(images, models, masks):
        radii, profile = radial_hybrid_profile(
            image - model,
            primary_yx,
            mask=mask,
            smoothing_scale_px=smooth,
        )
        hybrid = evaluate_radial_profile(image.shape, primary_yx, radii, profile)
        model_h = model + hybrid
        metric_h = companion_ring_metric(
            image,
            model_h,
            primary_yx,
            companion_yx,
            width_px=width,
            source_exclusion_radius_px=excl,
        )
        after_pcts.append(float(metric_h["median_pct"]))
        after_p90s.append(float(metric_h["p90_pct"]))
        if radii_ref is None:
            radii_ref = radii
        if radii.size != radii_ref.size:
            profile = np.interp(radii_ref, radii, profile)
        profiles.append(profile)
        models_hybrid.append(model_h)
    # Safety guard: the hybrid term must reduce the ring residual. When the base
    # form already models the AO halo (Psfao), the azimuthal-residual term is
    # degenerate and — scaled by the inflated Moffat FWHM — can make the metric
    # WORSE. In that case discard it rather than corrupt a good model.
    before_med = float(np.nanmedian(values))
    after_med = float(np.nanmedian(after_pcts))
    if not (np.isfinite(after_med) and after_med < before_med):
        return list(models), None, None, False, list(map(float, ring_pcts)), None
    return models_hybrid, np.asarray(profiles, dtype=np.float32), radii_ref.astype(np.float32), True, after_pcts, after_p90s


def _run_psfao_branch(cfg, stage_dir, primary_yx, companion_yx, field_yx=None):
    """Fit the physical AO (Psfao) model per bin and score each reconstruction
    with the SAME canonical companion-ring metric used for Moffat, so §3.4 is an
    apples-to-apples comparison. Returns a dict with status and, on success, the
    per-bin rows, reconstructions, per-bin ring pcts and the psfao inputs.
    Never raises for a missing maoppy: returns ``status='unavailable:...'``."""

    try:
        import maoppy  # noqa: F401
        from .stage_e01_psfao import (
            build_psfao_model_document,
            fit_psfao_bins,
            prepare_psfao_inputs,
        )
    except Exception as exc:  # pragma: no cover - environment dependent
        return {"status": f"unavailable:{exc.__class__.__name__}"}

    inp = prepare_psfao_inputs(cfg, stage_dir)
    rows, recons = fit_psfao_bins(
        inp["cube"], inp["stat"], inp["wave"], inp["bins"],
        inp["system"], inp["companion"], inp["mask_radius"], inp["fit_radius"],
        # MISMA mascara que la rama Moffat: sin esto `model_comparison` compara
        # dos ajustes con contaminantes distintos.
        field_yx=field_yx,
        # Rescate de los bins que se estancan en el vector de arranque comun.
        # Cada hueco que deja un bin caido lo cruza `_evaluate_psfao` con una
        # recta, y ahi nacen las mesetas del modelo cromatico.
        warm_start=bool(cfg.get("psf_warm_start", True)),
        # QUE PARTE DE LA IMAGEN manda en el ajuste. El default (`stat`) es el
        # historico y no cambia ningun run que no lo declare.
        weighting=str(cfg.get("psf_fit_weighting", PSFAO_DEFAULT_WEIGHTING)),
        weight_cap=cfg.get("psf_fit_weight_cap", PSFAO_DEFAULT_WEIGHT_CAP),
    )
    if not recons:
        return {"status": "unavailable:no_valid_fits", "rows": rows}
    width = float(cfg.get("psf_companion_ring_width_px", 3.0))
    excl = float(inp["mask_radius"])
    ring_rows = []
    for mid in sorted(recons):
        image, recon = recons[mid]
        metric = companion_ring_metric(
            image, recon, primary_yx, companion_yx,
            width_px=width, source_exclusion_radius_px=excl,
        )
        ring_rows.append({
            "wave_center_A": float(mid),
            "ring_residual_pct": float(metric["median_pct"]),
            "ring_residual_p90_pct": float(metric["p90_pct"]),
        })
    rows_by_wave = {float(row["lambda_A"]): row for row in rows}
    for ring_row in ring_rows:
        row = rows_by_wave[ring_row["wave_center_A"]]
        row["ring_residual_pct_canonical"] = ring_row["ring_residual_pct"]
        row["ring_residual_p90_pct_canonical"] = ring_row["ring_residual_p90_pct"]
    return {
        "status": "ok",
        "rows": rows,
        "recons": recons,
        "ring_rows": ring_rows,
        "inp": inp,
        "build_doc": build_psfao_model_document,
    }


def compute_stage_e01_products(config) -> StageE01Product:
    cfg = dict(config)
    paths = stage_e01_paths(cfg["run_id"], project_root=cfg.get("project_root"))
    stage_dir = paths["paths"].stage_dir
    cube_path = Path(cfg.get("stage_e01_input_cube_fits", paths["stage02_cube_fits"]))
    positions_path = Path(cfg.get("stage_e01_positions_qc", paths["stage01c_qc_json"]))
    if not positions_path.exists():
        raise FileNotFoundError(f"Stage01c QC is required for C1: {positions_path}")
    positions_qc = read_json(positions_path)
    cubes, wavelengths, _ = _load_cube(cube_path)
    bins = make_psf_bins(
        wavelengths,
        bin_A=float(cfg.get("psf_bin_A", 100.0)),
        bad_windows_A=cfg.get("stage_e01_bad_windows_A", []),
        min_channels=int(cfg.get("psf_min_channels_per_bin", 3)),
    )
    primary_yx, companion_yx, field_yx = _positions_from_qc(positions_qc)
    track_path = positions_path.parent.parent / "tables" / "stage01c_chromatic_centroids.csv"
    b3_track, b3_track_status = _b3_chromatic_track(track_path)
    sep_px = float(np.hypot(companion_yx[0] - primary_yx[0], companion_yx[1] - primary_yx[1]))

    # --- Moffat fit (always run: it is the tie-break form and the FWHM source
    # for the hybrid smoothing scale). -----------------------------------------
    rows, images, models, masks, mask_meta = _moffat_fit_rows(cubes, wavelengths, bins, positions_qc, cfg)
    fwhm_med = float(np.nanmedian([row["fwhm_maj"] for row in rows]))
    moffat_ring = np.asarray([row["ring_residual_pct"] for row in rows], dtype=np.float64)
    moffat_p90 = np.asarray([row["ring_residual_p90_pct"] for row in rows], dtype=np.float64)
    moffat_median = float(np.nanmedian(moffat_ring)) if moffat_ring.size else float("nan")

    # --- Psfao fit + selection (spec §3.4). ------------------------------------
    form_cfg = str(cfg.get("e01_psf_form", "auto")).lower()
    if form_cfg not in ("auto", "moffat", "psfao"):
        raise ValueError(f"e01_psf_form must be auto|moffat|psfao, got {form_cfg!r}.")
    psfao = {"status": "skipped"}
    if form_cfg in ("auto", "psfao"):
        psfao = _run_psfao_branch(cfg, stage_dir, primary_yx, companion_yx, field_yx)
    psfao_ok = psfao.get("status") == "ok"
    psfao_ring = (
        np.asarray([r["ring_residual_pct"] for r in psfao["ring_rows"]], dtype=np.float64)
        if psfao_ok else np.asarray([], dtype=np.float64)
    )
    psfao_median = float(np.nanmedian(psfao_ring)) if psfao_ring.size else float("nan")

    if form_cfg == "moffat":
        chosen, reason = "moffat", "forced by config (e01_psf_form=moffat)"
    elif form_cfg == "psfao":
        if not psfao_ok:
            raise RuntimeError(f"e01_psf_form=psfao but the Psfao fit is unavailable: {psfao.get('status')}")
        chosen, reason = "psfao", "forced by config (e01_psf_form=psfao)"
    else:  # auto: lower median ring residual wins; tie -> moffat
        if psfao_ok and np.isfinite(psfao_median) and psfao_median < moffat_median:
            chosen = "psfao"
            reason = f"auto: psfao median ring residual {psfao_median:.2f}% < moffat {moffat_median:.2f}%"
        else:
            chosen = "moffat"
            if psfao_ok:
                reason = f"auto: moffat median ring residual {moffat_median:.2f}% <= psfao {psfao_median:.2f}% (tie->moffat)"
            else:
                reason = f"auto: psfao unavailable ({psfao.get('status')}); moffat by default"

    open_issues = []
    if not psfao_ok and form_cfg in ("auto", "psfao"):
        open_issues.append(f"Psfao comparison fit not run ({psfao.get('status')}).")

    norm_radius = float(cfg.get("psf_norm_radius_px", 25.0))
    chromatic = bool(positions_qc.get("chromatic", {}).get("chromatic_centroid_needed", False))
    excluded = cfg.get("stage_e01_bad_windows_A", [])

    if chosen == "moffat":
        chosen_models, hybrid_profiles, hybrid_radii, hybrid_applied, after_pcts, after_p90s = _apply_hybrid(
            [row["ring_residual_pct"] for row in rows], images, models, masks,
            primary_yx, companion_yx, fwhm_med, cfg,
        )
        if hybrid_applied:
            for row, after, after_p90 in zip(rows, after_pcts, after_p90s):
                row["ring_residual_pct_after_hybrid"] = float(after)
                row["ring_residual_p90_pct_after_hybrid"] = float(after_p90)
        model_doc = build_psf_model_document(
            [row["wave_center_A"] for row in rows], rows,
            form="moffat", norm_radius_px=norm_radius, hybrid=hybrid_applied,
        )
        waves_rt = np.linspace(rows[0]["wave_center_A"], rows[-1]["wave_center_A"], min(10, len(rows)))
        ring_after = np.asarray(after_pcts, dtype=np.float64) if hybrid_applied else moffat_ring
        p90_values = np.asarray(after_p90s, dtype=np.float64) if hybrid_applied else moffat_p90
        centroid_diff = _centroid_vs_b3(rows, "moffat", b3_track)
        bins_interpolated = []
        for lo, hi in excluded:
            if float(hi) >= rows[0]["wave_center_A"] and float(lo) <= rows[-1]["wave_center_A"]:
                bins_interpolated.append([float(lo), float(hi)])
        binning = {
            "bin_A": float(cfg.get("psf_bin_A", 100.0)),
            "n_bins": int(len(rows)),
            "excluded_windows_A": excluded,
            "bins_interpolated": bins_interpolated,
        }
        masks_qc = {
            "companion_radius_px": float(mask_meta["mask_radius_px"]),
            "chromatic_tracking": chromatic,
            "core_mask_px": float(mask_meta["core_mask_px_max"]),
            "saturation_detected": bool(mask_meta["saturation_detected"]),
        }
        fit_qc = {
            "form_chosen": "moffat",
            "maoppy_status": _maoppy_status("moffat"),
            "background_mode": "fixed_external",
            "clip_frac_max": float(np.nanmax([row["clip_frac"] for row in rows])),
            "chi2r_median": float(np.nanmedian([row["chi2r"] for row in rows])),
        }
        smoothing_qc = {
            "per_param_model": {key: model_doc["coefficients"][key]["model"] for key in model_doc["coefficients"]},
            "outlier_bins": [],
            "centroid_vs_b3_max_diff_px": centroid_diff,
            "centroid_vs_b3_status": b3_track_status,
        }
        psfao_rows = psfao["rows"] if psfao_ok else None
    else:  # psfao
        inp = psfao["inp"]
        psfao_rows = psfao["rows"]
        ring_rows = psfao["ring_rows"]
        recons = psfao["recons"]
        model_doc, meta = psfao["build_doc"](
            psfao_rows, inp["system"], inp["norm_radius"], inp["fit_radius"],
            # La rejilla de evaluacion viaja en el documento, que es donde la lee
            # `_evaluate_psfao`. Por defecto, el ancho de bin que se acaba de
            # ajustar; el config manda si lo declara.
            wave_bin_A=float(cfg.get("psfao_wave_bin_A", inp["bin_A"])),
            weighting=str(cfg.get("psf_fit_weighting", PSFAO_DEFAULT_WEIGHTING)),
            weight_cap=cfg.get("psf_fit_weight_cap", PSFAO_DEFAULT_WEIGHT_CAP),
        )
        mids = sorted(recons)
        p_images = [recons[m][0] for m in mids]
        p_models = [recons[m][1] for m in mids]
        p_masks = [source_mask(p_images[0].shape, [companion_yx], inp["mask_radius"]) for _ in mids]
        chosen_models, hybrid_profiles, hybrid_radii, hybrid_applied, after_pcts, after_p90s = _apply_hybrid(
            [r["ring_residual_pct"] for r in ring_rows], p_images, p_models, p_masks,
            primary_yx, companion_yx, fwhm_med, cfg,
        )
        if hybrid_applied:
            for r, after in zip(ring_rows, after_pcts):
                r["ring_residual_pct_after_hybrid"] = float(after)
            for r, after_p90 in zip(ring_rows, after_p90s):
                r["ring_residual_p90_pct_after_hybrid"] = float(after_p90)
            rows_by_wave = {float(row["lambda_A"]): row for row in psfao_rows}
            for ring_row in ring_rows:
                row = rows_by_wave[ring_row["wave_center_A"]]
                row["ring_residual_pct_after_hybrid_canonical"] = ring_row["ring_residual_pct_after_hybrid"]
                row["ring_residual_p90_pct_after_hybrid_canonical"] = ring_row["ring_residual_p90_pct_after_hybrid"]
            model_doc["hybrid"] = True
        waves = [r["wave_center_A"] for r in ring_rows]
        waves_rt = np.linspace(min(waves), max(waves), min(10, len(waves)))
        ring_after = np.asarray(after_pcts, dtype=np.float64) if hybrid_applied else psfao_ring
        p90_values = np.asarray(
            after_p90s if hybrid_applied else [r["ring_residual_p90_pct"] for r in ring_rows],
            dtype=np.float64,
        )
        ok_rows = [r for r in psfao_rows if r.get("status") == "ok"]
        centroid_diff = _centroid_vs_b3(ok_rows, "psfao", b3_track, image_shape=p_images[0].shape)
        bins_interpolated = [[float(lo), float(hi)] for lo, hi in inp["bad"]]
        binning = {
            "bin_A": float(inp["bin_A"]),
            "n_bins": int(len(inp["bins"])),
            "excluded_windows_A": inp["bad"],
            "bins_interpolated": bins_interpolated,
        }
        masks_qc = {
            "companion_radius_px": float(inp["mask_radius"]),
            "chromatic_tracking": chromatic,
            "core_mask_px": 0.0,
            "saturation_detected": False,
        }
        fit_qc = {
            "form_chosen": "psfao",
            "model": "maoppy.Psfao",
            "maoppy_status": _maoppy_status("psfao"),
            "background_mode": "psffit_flux_bck",
            "fit_radius_px": float(inp["fit_radius"]),
            "n_ok_bins": int(meta["n_ok"]),
            "n_bins_rejected": int(meta["n_rejected"]),
            "n_fit_failed": int(len(psfao_rows) - meta["n_ok"]),
            "clip_frac_max": 0.0,
            "chi2r_median": None,
            # Que parte de la imagen decidio el ajuste. Va aqui ademas de en
            # `psf_model.json` porque este es el camino canonico (el de la
            # comparacion de formas) y su QC es lo que lee F1.
            "weighting": str(cfg.get("psf_fit_weighting", PSFAO_DEFAULT_WEIGHTING)),
            "weight_cap": (None if cfg.get("psf_fit_weight_cap",
                                           PSFAO_DEFAULT_WEIGHT_CAP) is None
                           else float(cfg.get("psf_fit_weight_cap",
                                              PSFAO_DEFAULT_WEIGHT_CAP))),
        }
        smoothing_qc = {
            # `smoothed_poly` NO es lo que se evalua mientras exista `param_table`:
            # `_evaluate_psfao` interpola la tabla por bin y el polinomio queda
            # inerte. Se declara asi para que el QC no siga sugiriendo que el
            # modelo por canal es una parabola. Medido en el notebook de C1.
            "per_param_model": {name: f"polynomial_deg{max(len(v) - 1, 0)}" for name, v in model_doc["smoothed_poly"].items()},
            "per_param_model_is_evaluated": not bool(model_doc.get("param_table")),
            "evaluated_as": ("param_table_linear_interp" if model_doc.get("param_table")
                             else "smoothed_poly"),
            "outlier_bins": [],
            "centroid_vs_b3_max_diff_px": centroid_diff,
            "centroid_vs_b3_status": b3_track_status,
            "warm_start": bool(cfg.get("psf_warm_start", True)),
            # Las dos direcciones cuentan: un bin que solo tiene vecino bueno al
            # rojo se recupera en la pasada hacia atras, y dejarlo fuera de la
            # cuenta hacia parecer que el arranque caliente no habia hecho nada.
            "n_bins_rescued_by_warm_start": int(sum(
                1 for r in psfao_rows
                if r.get("start_vector") in ("warm_start", "warm_start_back")
                and r.get("status") == "ok")),
        }

    roundtrip = psf_roundtrip_error(model_doc, waves_rt)
    ring_summary = _ring_qc_summary(ring_after, p90_values, cfg)
    ring_median = ring_summary["median"]
    ring_p90 = ring_summary["p90"]

    if ring_summary["issue"]:
        open_issues.append("Companion-ring residual remains above 5 pct after the fixed C1 sequence.")
    if centroid_diff is not None and centroid_diff > 0.3 and chromatic:
        open_issues.append("PSF centroid differs from B3 chromatic centroid story by >0.3 px.")

    model_comparison = {
        "metric": "companion_ring_metric.median_pct (canonical, applied to both forms)",
        "selection_mode": form_cfg,
        "form_chosen": chosen,
        "reason": reason,
        "moffat": {
            "ring_residual_pct_median": moffat_median,
            "ring_residual_pct_p90": float(np.nanpercentile(moffat_p90, 90)) if moffat_p90.size else None,
            "n_bins": int(len(rows)),
        },
        "psfao": {
            "status": psfao.get("status"),
            "ring_residual_pct_median": psfao_median if psfao_ok else None,
            "ring_residual_pct_p90": float(np.nanpercentile(
                [r["ring_residual_p90_pct"] for r in psfao["ring_rows"]], 90)) if psfao_ok else None,
            "n_bins": int(len(psfao["ring_rows"])) if psfao_ok else 0,
        },
    }

    qc = {
        "stage": "e01_chromatic_psf",
        "run_id": cfg["run_id"],
        "input": {"cube": str(cube_path), "sha256": _sha256(cube_path), "positions_from": str(positions_path)},
        "binning": binning,
        "masks": masks_qc,
        "fit": fit_qc,
        "smoothing": smoothing_qc,
        "companion_ring_metric": {
            "radius_px": sep_px,
            "width_px": float(cfg.get("psf_companion_ring_width_px", 3.0)),
            "residual_pct_median": ring_median,
            "residual_pct_p90": ring_p90,
            "bins_above_5pct": ring_summary["bins_above"],
            "after_hybrid": bool(hybrid_applied),
        },
        "hybrid": {
            "applied": bool(hybrid_applied),
            "smoothing_scale_px": None if not hybrid_applied else float(max(2.0 * fwhm_med, 0.0)),
        },
        "normalization": {
            "norm_radius_px": norm_radius,
            "roundtrip_error": float(roundtrip),
        },
        "model_comparison": model_comparison,
        "open_issues": open_issues,
    }
    return StageE01Product(
        fit_rows=rows,
        psf_model=model_doc,
        qc=qc,
        hybrid_profiles=hybrid_profiles,
        hybrid_radii=hybrid_radii,
        psf_form=chosen,
        psfao_rows=psfao_rows,
    )


def write_stage_e01_products(product: StageE01Product, config, paths):
    paths["paths"].ensure_base_dirs()
    paths["plot_dir"].mkdir(parents=True, exist_ok=True)
    _write_csv(paths["stage_e01_params_csv"], product.fit_rows)
    if product.psf_form == "psfao" and product.psfao_rows:
        _write_psfao_csv(paths["paths"].stage_dir / "stage_e01_psfao_params.csv", product.psfao_rows)
    write_json(paths["psf_model_json"], product.psf_model)
    if product.hybrid_profiles is not None:
        fits.HDUList(
            [
                fits.PrimaryHDU(),
                fits.ImageHDU(product.hybrid_profiles.astype(np.float32), name="PROFILE"),
                fits.ImageHDU(product.hybrid_radii.astype(np.float32), name="RADIUS_PX"),
            ]
        ).writeto(paths["psf_hybrid_residual_fits"], overwrite=True)
    if bool(config.get("psf_save_plots", config.get("save_intermediate_plots", False))):
        _write_summary_plot(product, paths)
        product.qc["figures"] = {"summary": str(Path("plots") / "stage_e01" / paths["summary_plot"].name)}
    write_json(paths["stage_e01_qc_json"], product.qc)
    return {"params_csv": paths["stage_e01_params_csv"], "model_json": paths["psf_model_json"], "qc_json": paths["stage_e01_qc_json"], "qc": product.qc}


def _write_csv(path, rows):
    if not rows:
        Path(path).write_text("", encoding="utf-8")
        return
    fieldnames = list(dict.fromkeys(key for row in rows for key in row.keys()))
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_psfao_csv(path, rows):
    from .stage_e01_psfao import PSFAO_PARAM_NAMES

    # Las `*_err` son las incertidumbres formales que `psffit` ya calculaba y que
    # `fit_bin` tiraba (ver `_psfao_param_errors`). Van detras de sus parametros
    # y son aditivas: nada aguas abajo las lee, y los CSV escritos antes de que
    # existieran siguen siendo legibles (el lector va por nombre de columna).
    cols = ["lambda_A", "samp", "amp", "bck", "dy", "dx", "ring_residual_pct",
            "ring_residual_pct_canonical", "ring_residual_p90_pct_canonical",
            *PSFAO_PARAM_NAMES, *(f"{name}_err" for name in PSFAO_PARAM_NAMES),
            "dy_err", "dx_err", "ring_residual_pct_after_hybrid_canonical",
            "ring_residual_p90_pct_after_hybrid_canonical", "optimizer_success",
            "optimizer_status", "optimizer_message", "optimizer_nfev", "optimizer_cost",
            "optimizer_stalled_at_initial", "start_vector", "status"]
    with open(path, "w", encoding="utf-8") as f:
        f.write(",".join(cols) + "\n")
        for r in rows:
            f.write(",".join(str(r.get(c, "")) for c in cols) + "\n")


def _write_summary_plot(product, paths):
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    rows = product.psfao_rows if product.psf_form == "psfao" else product.fit_rows
    rows = [row for row in rows if row.get("status", "ok") == "ok"]
    wave_key = "lambda_A" if product.psf_form == "psfao" else "wave_center_A"
    wave = np.asarray([row[wave_key] for row in rows], dtype=float)
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    if product.psf_form == "psfao":
        axes[0, 0].plot(wave, [row["r0"] for row in rows])
        axes[0, 0].set_ylabel("Psfao r0")
        axes[0, 1].plot(wave, [row["beta"] for row in rows])
        axes[0, 1].set_ylabel("Psfao beta")
        before_key = "ring_residual_pct_canonical"
        after_key = "ring_residual_pct_after_hybrid_canonical"
        axes[1, 1].plot(wave, [row["dy"] for row in rows], label="dy")
        axes[1, 1].plot(wave, [row["dx"] for row in rows], label="dx")
        axes[1, 1].set_ylabel("centroid offset [px]")
        axes[1, 1].legend()
    else:
        axes[0, 0].plot(wave, [row["fwhm_maj"] for row in rows], label="maj")
        axes[0, 0].plot(wave, [row["fwhm_min"] for row in rows], label="min")
        axes[0, 0].set_ylabel("FWHM [px]")
        axes[0, 0].legend()
        axes[0, 1].plot(wave, [row["beta"] for row in rows])
        axes[0, 1].set_ylabel("beta")
        before_key = "ring_residual_pct"
        after_key = "ring_residual_pct_after_hybrid"
        axes[1, 1].plot(wave, [row["chi2r"] for row in rows])
        axes[1, 1].set_ylabel("chi2r")
    axes[1, 0].plot(wave, [row[before_key] for row in rows], label=product.psf_form)
    if product.qc["hybrid"]["applied"] and all(after_key in row for row in rows):
        axes[1, 0].plot(wave, [row[after_key] for row in rows], label="hybrid")
    axes[1, 0].axhline(5.0, color="0.4", ls="--")
    axes[1, 0].set_ylabel("Ring residual [%]")
    axes[1, 0].legend()
    for ax in axes.ravel():
        ax.set_xlabel("Wavelength [A]")
    fig.savefig(paths["summary_plot"], dpi=150)
    plt.close(fig)


def run_stage_e01(run_id=None, *, project_root=None, overrides=None, allow_run_id_mismatch=False):
    cfg = stage_e01_config_from_run(
        run_id,
        project_root=project_root,
        overrides=overrides,
        allow_run_id_mismatch=allow_run_id_mismatch,
    )
    paths = stage_e01_paths(cfg["run_id"], project_root=cfg.get("project_root"))
    product = compute_stage_e01_products(cfg)
    written = write_stage_e01_products(product, cfg, paths)
    return {"config": cfg, "paths": paths, "qc": product.qc, "written": written}


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run Stage E01/C1 chromatic PSF model.")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--project-root", default=None)
    parser.add_argument("--allow-run-id-mismatch", action="store_true")
    parser.add_argument("--save-plots", action="store_true")
    args = parser.parse_args(argv)
    overrides = {"psf_save_plots": True} if args.save_plots else None
    result = run_stage_e01(
        args.run_id,
        project_root=args.project_root,
        overrides=overrides,
        allow_run_id_mismatch=args.allow_run_id_mismatch,
    )
    print(result["paths"]["stage_e01_qc_json"])


__all__ = [
    "StageE01Product",
    "compute_stage_e01_products",
    "make_psf_bins",
    "run_stage_e01",
    "stage_e01_config_from_run",
    "stage_e01_paths",
    "write_stage_e01_products",
]


if __name__ == "__main__":
    main()
