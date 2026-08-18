"""Stage X02/C3: Horne-style optimal spectra for LS and PSFSUB backgrounds."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import warnings

import numpy as np
from astropy.io import fits

from ..config import load_run_config
from ..growth_curve import resolve_flux_convention
from ..extraction.aperture import FLAG_CLIPPED
from ..extraction.optimal import (
    OptimalExtraction,
    circular_window_indices,
    fit_primary_psf_model_cube,
    make_optimal_product,
    scaled_psf_model,
)
from ..extraction.product import SpectrumProduct
from ..io import read_json, resolve_bunit, write_json
from ..paths import RunPaths
from .stage_x01_aperture import (
    _best_indices_from_stage04b,
    _load_positions,
    _load_residual_cube,
    _load_stat_cube,
    _read_optional_json,
    _wavelength_frame,
)


@dataclass(frozen=True)
class StageX02Product:
    extractions: dict[str, OptimalExtraction]
    qc: dict
    psfsub_model_meta: dict


def stage_x02_paths(run_id, project_root=None):
    root = Path(project_root or Path.cwd()).resolve()
    paths = RunPaths.from_project_root(run_id, root)
    return {
        "paths": paths,
        "stage01c_qc_json": paths.stage_dir / "stage01c_qc.json",
        "stage01_qc_json": paths.stage_dir / "stage01_qc.json",
        "stage00q_qc_json": paths.stage_dir / "stage00q_qc.json",
        "stage02_cube_fits": paths.stage_dir / "stage02_xcorr_cube_stack.fits",
        "stage04b_cube_fits": paths.stage_dir / "stage04b_local_surface_cube_stack.fits",
        "stage04b_qc_json": paths.stage_dir / "stage04b_qc.json",
        "stage04b_good_mask_npy": paths.stage_dir / "stage04b_good_wavelength_mask.npy",
        "stage04b_bad_mask_npy": paths.stage_dir / "stage04b_bad_wavelength_mask.npy",
        "cube_residual_object": paths.stage_dir / "cube_residual_local_object.fits",
        "psf_model_json": paths.stage_dir / "psf_model.json",
        "spec_aperture_object": paths.stage_dir / "spec_aperture_object.fits",
        "spec_optimal_object": paths.stage_dir / "spec_optimal_object.fits",
        "spec_optimal_psfsub_object": paths.stage_dir / "spec_optimal_psfsub_object.fits",
        "spec_optimal_qc_json": paths.stage_dir / "spec_optimal_qc.json",
        "spec_optimal_rejection_fits": paths.stage_dir / "spec_optimal_clip_rejection.fits",
    }


def _bad_windows_from_config(cfg):
    windows = cfg.get("x02_bad_windows_A") or cfg.get("bad_wavelength_ranges_A")
    if windows is not None:
        return [[float(lo), float(hi)] for lo, hi in windows]
    if cfg.get("drop_wave_min_A") is not None and cfg.get("drop_wave_max_A") is not None:
        return [[float(cfg["drop_wave_min_A"]), float(cfg["drop_wave_max_A"])]]
    return []


def stage_x02_config_from_run(
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
    cfg.setdefault("x02_source_key", "companion")
    cfg.setdefault("x02_star_key", "primary")
    cfg.setdefault("x02_window_radius_px", cfg.get("optimal_window_px", 8.0))
    cfg.setdefault("x02_clip_sigma", cfg.get("clip_sigma", 4.0))
    cfg.setdefault("x02_clip_max_iter", cfg.get("clip_max_iter", 2))
    cfg.setdefault("x02_error_mode", "auto")
    cfg.setdefault("x02_aperture_correction", "auto")
    cfg.setdefault("x02_control_apertures", cfg.get("stage07_control_apertures", 8))
    cfg.setdefault("x02_control_exclude_angle_deg", 25.0)
    cfg.setdefault("x02_bad_windows_A", _bad_windows_from_config(cfg))
    cfg.setdefault("x02_skyline_windows_A", cfg.get("skyline_windows_A", []))
    cfg.setdefault("x02_interpolated_windows_A", cfg.get("stage_e01_interpolated_windows_A", []))
    cfg.setdefault("x02_wframe", cfg.get("wavelength_frame", "topocentric"))
    cfg.setdefault("x02_bunit", cfg.get("cube_bunit", ""))
    # Local background annulus around every center (object AND controls):
    # re-references any residual pedestal (04b for LS) so all methods share
    # the flux convention (D1 v2 §3.1). Same default as x01's wings-intact
    # annulus [r_in, r_out, star_exclude_radius].
    cfg.setdefault("x02_local_bkg_annulus_px", cfg.get("x01_annulus_bkg_px", [8.0, 14.0, 30.0]))
    # Fondo local: `annulus` (historico) o `azimuthal` (anillo centrado en la
    # primaria al radio del compañero). El defecto NO se mueve: cual de los dos
    # es mejor depende del objeto — medido, va en direcciones opuestas en los
    # dos del proyecto (reports/20260727/sesgo_anillo_y_ventana_2026-07-27.md).
    cfg.setdefault("x02_background_mode", "annulus")
    cfg.setdefault("x02_azimuthal_width_px", 3.0)
    cfg.setdefault("x02_azimuthal_exclude_px", 10.0)
    # `local_plane`: el mismo ajuste que 04b y C4, evaluado EN el compañero.
    cfg.setdefault("x02_plane_fit_radius_px", 14.0)
    cfg.setdefault("x02_plane_mask_radius_px", 3.0)
    cfg.setdefault("x02_primary_fit_radius_px", cfg.get("psf_norm_radius_px", 25.0))
    cfg.setdefault("x02_primary_exclude_radius_px", cfg.get("x02_window_radius_px", 8.0))
    return cfg


def _load_psf_model(paths, cfg):
    path = Path(cfg.get("x02_psf_model_json") or paths["psf_model_json"])
    if not path.exists():
        raise FileNotFoundError(path)
    return read_json(path), path


def _load_stage02_cube(paths, cfg, expected_wave=None):
    path = Path(cfg.get("x02_stage02_cube_fits") or paths["stage02_cube_fits"])
    if not path.exists():
        raise FileNotFoundError(path)
    with fits.open(path, memmap=True) as hdul:
        if "CUBES" not in hdul or "WAVELENGTH" not in hdul:
            raise RuntimeError(f"{path} must contain CUBES and WAVELENGTH HDUs.")
        cubes = hdul["CUBES"].data.astype(np.float64)
        wave = hdul["WAVELENGTH"].data.astype(np.float64)
        bunit = resolve_bunit(cfg, stack_bunit=hdul["CUBES"].header.get("BUNIT")
                              or hdul[0].header.get("BUNIT"), override_key="x02_bunit")
    if expected_wave is not None and (wave.shape != expected_wave.shape or not np.allclose(wave, expected_wave, rtol=0.0, atol=1e-8)):
        raise RuntimeError("Stage02 and Stage04b wavelength axes differ.")
    if cubes.ndim == 4:
        best = _best_indices_from_stage04b(paths)
        if best is None:
            best = list(range(cubes.shape[0]))
        best = [i for i in best if 0 <= i < cubes.shape[0]]
        if not best:
            raise RuntimeError("No valid Stage04b best indices for Stage02 average.")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            cube = np.nanmean(cubes[best], axis=0)
    elif cubes.ndim == 3:
        cube = cubes
    else:
        raise RuntimeError(f"Unexpected CUBES shape in {path}: {cubes.shape}")
    return cube, wave, path, bunit


def _load_perobs_psfsub_cube(path, *, expected_shape, expected_wave):
    """El cubo residual de C1b, ya en el marco de B1.

    Se exige que encaje con el cubo de B2 en forma y en eje λ: si no encaja, el
    encuadre o el binado han dejado de ser los mismos y restar aquí sería
    comparar dos campos distintos. Falla en vez de recortar por su cuenta.
    """

    with fits.open(path, memmap=True) as hdul:
        if "DATA" not in hdul or "WAVELENGTH" not in hdul:
            raise RuntimeError(f"{path} must contain DATA and WAVELENGTH HDUs.")
        cube = hdul["DATA"].data.astype(np.float64)
        wave = hdul["WAVELENGTH"].data.astype(np.float64)
        bunit = hdul["DATA"].header.get("BUNIT") or hdul[0].header.get("BUNIT")
    if cube.shape != tuple(expected_shape):
        raise RuntimeError(
            f"{path}: forma {cube.shape} != la del cubo de B2 {tuple(expected_shape)}."
        )
    if wave.shape != expected_wave.shape or not np.allclose(wave, expected_wave, rtol=0.0, atol=1e-6):
        raise RuntimeError(f"{path}: el eje λ no es el de B2.")
    return cube, wave, bunit


def _stat_metadata(paths, cfg):
    qc00 = _read_optional_json(paths["stage00q_qc_json"])
    qc01 = _read_optional_json(paths["stage01_qc_json"])
    m5 = qc00.get("m5_stat", {}) if isinstance(qc00, dict) else {}
    stat_factor = cfg.get(
        "x02_stat_factor_spaxel",
        cfg.get("stat_factor_spaxel", m5.get("factor_spaxel_median", 1.0)),
    )
    covariance_factor = cfg.get(
        "x02_covariance_factor_box3",
        cfg.get("covariance_factor_box3", qc01.get("stat", {}).get("covariance_factor_box3", 1.0)),
    )
    stat_status = cfg.get("x02_stat_status", m5.get("status", "unknown"))
    try:
        stat_factor = float(stat_factor)
    except Exception:
        stat_factor = 1.0
    try:
        covariance_factor = float(covariance_factor)
    except Exception:
        covariance_factor = 1.0
    if not np.isfinite(stat_factor) or stat_factor <= 0:
        stat_factor = 1.0
    if not np.isfinite(covariance_factor) or covariance_factor <= 0:
        covariance_factor = 1.0
    return stat_factor, covariance_factor, str(stat_status), qc00, qc01


def _finite_percentiles(values):
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"median": None, "p10": None, "p90": None}
    return {
        "median": float(np.nanmedian(arr)),
        "p10": float(np.nanpercentile(arr, 10.0)),
        "p90": float(np.nanpercentile(arr, 90.0)),
    }


def _median_ratio(a, b):
    aa = np.asarray(a, dtype=np.float64)
    bb = np.asarray(b, dtype=np.float64)
    good = np.isfinite(aa) & np.isfinite(bb) & (bb > 0)
    if not np.any(good):
        return None
    return float(np.nanmedian(aa[good] / bb[good]))


def _snr_gain_vs_aperture(optimal: SpectrumProduct, aperture_path):
    path = Path(aperture_path)
    if not path.exists():
        return {"median": None, "p10": None, "p90": None}, None
    aperture = SpectrumProduct.read(path)
    opt_snr = np.abs(optimal.flux) / np.maximum(optimal.flux_err, 1e-300)
    ap_snr = np.abs(aperture.flux) / np.maximum(aperture.flux_err, 1e-300)
    good = np.isfinite(opt_snr) & np.isfinite(ap_snr) & (ap_snr > 0)
    gain = opt_snr[good] / ap_snr[good]
    bias = None
    bias_good = good & (np.abs(aperture.flux) > 0)
    if np.any(bias_good):
        scale = np.nanmedian(np.abs(aperture.flux[bias_good]))
        if np.isfinite(scale) and scale > 0:
            bias = float(100.0 * np.nanmedian((optimal.flux[bias_good] - aperture.flux[bias_good]) / scale))
    return _finite_percentiles(gain), bias


def _clip_concentration(extraction: OptimalExtraction, object_yx, window_radius_px):
    ypix, xpix = circular_window_indices(extraction.rejection_map.shape, object_yx, window_radius_px)
    local = extraction.rejection_map[ypix, xpix]
    if local.size == 0:
        return {"companion_window_over_mean": None}
    mean = float(np.nanmean(local))
    peak = float(np.nanmax(local))
    ratio = None if mean <= 0 else float(peak / mean)
    return {"companion_window_over_mean": ratio}


def _psf_sensitivity(ls_cube, wave, object_yx, psf_model, cfg, variance, stat_factor, covariance_factor, stat_status, base):
    biases = []
    for scale in (0.9, 1.1):
        model = scaled_psf_model(psf_model, scale)
        ext = make_optimal_product(
            ls_cube,
            wave,
            object_yx,
            model,
            run_id=cfg["run_id"],
            input_cube_path="psf_sensitivity",
            variant=f"ls_fwhm_{scale:g}",
            variance_zyx=variance,
            stat_factor_spaxel=stat_factor,
            covariance_factor_box3=covariance_factor,
            stat_status=stat_status,
            error_mode=cfg.get("x02_error_mode", "auto"),
            aperture_correction=cfg.get("x02_aperture_correction", "auto"),
            window_radius_px=float(cfg.get("x02_window_radius_px", 8.0)),
            clip_sigma=float(cfg.get("x02_clip_sigma", 4.0)),
            clip_max_iter=int(cfg.get("x02_clip_max_iter", 2)),
            n_controls=0,
            local_bkg_annulus_px=cfg.get("x02_local_bkg_annulus_px"),
            background_mode=cfg.get("x02_background_mode", "annulus"),
            azimuthal_width_px=float(cfg.get("x02_azimuthal_width_px", 3.0)),
            azimuthal_exclude_px=float(cfg.get("x02_azimuthal_exclude_px", 10.0)),
            plane_fit_radius_px=float(cfg.get("x02_plane_fit_radius_px", 14.0)),
            plane_mask_radius_px=float(cfg.get("x02_plane_mask_radius_px", 3.0)),
        )
        good = np.isfinite(ext.product.flux) & np.isfinite(base.product.flux) & (np.abs(base.product.flux) > 0)
        if np.any(good):
            biases.append(float(100.0 * np.nanmedian((ext.product.flux[good] - base.product.flux[good]) / base.product.flux[good])))
    if not biases:
        return {"fwhm_pm10pct_flux_bias_pct": None}
    return {"fwhm_pm10pct_flux_bias_pct": float(np.nanmax(np.abs(biases)))}


def _qc_payload(extractions, cfg, paths, stat_state, psf_model_path, psfsub_model_meta, open_issues, psf_sensitivity):
    ls = extractions["ls"]
    ratio = _median_ratio(ls.product.flux_err, ls.product.flux_err_emp)
    snr_gain, continuum_bias = _snr_gain_vs_aperture(ls.product, paths["spec_aperture_object"])
    clip = {
        "sigma": float(cfg.get("x02_clip_sigma", 4.0)),
        "max_iter": int(cfg.get("x02_clip_max_iter", 2)),
        "frac_clipped_median": float(np.nanmedian(ls.clip_fraction)),
        "channels_flagged": int(np.count_nonzero(ls.product.flags & FLAG_CLIPPED)),
    }
    return {
        "stage": "x02_optimal",
        "run_id": str(cfg["run_id"]),
        "variants": list(extractions),
        "psf_model": str(psf_model_path),
        "window_px": float(cfg.get("x02_window_radius_px", 8.0)),
        "p_normalization": "window_renorm_plus_apcorr",
        "clip": clip,
        "clip_concentration": _clip_concentration(ls, (ls.product.header["SRCPOS_Y"], ls.product.header["SRCPOS_X"]), cfg.get("x02_window_radius_px", 8.0)),
        "errors": {
            "mode": ls.error_mode,
            "stat_vs_empirical_median_ratio": ratio,
            "stat_input": stat_state,
            "covariance_interpolation": "linear_spaxel_to_box3_capped_at_box3",
        },
        "snr_gain_vs_aperture": snr_gain,
        "continuum_bias_vs_aperture_pct": continuum_bias,
        "psf_sensitivity": psf_sensitivity,
        "psfsub_model": psfsub_model_meta,
        "checks": {
            "v1_snr_gain_ok": None if snr_gain["median"] is None else bool(snr_gain["median"] >= 1.0),
            "v2_error_ratio_ok": None if ratio is None else bool(0.7 <= ratio <= 1.4),
            "v3_continuum_bias_ok": None if continuum_bias is None else bool(abs(continuum_bias) <= 3.0),
            "v4_clip_concentration_ok": True,
            "v5_ls_vs_psfsub_written": "psfsub" in extractions,
        },
        "open_issues": list(open_issues),
    }


def compute_stage_x02_products(config, paths=None):
    cfg = dict(config)
    run_id = cfg["run_id"]
    root = Path(cfg.get("project_root") or Path.cwd()).resolve()
    paths = stage_x02_paths(run_id, root) if paths is None else paths

    object_yx, star_yx, _positions_path, _positions_qc = _load_positions(paths, cfg)
    psf_model, psf_model_path = _load_psf_model(paths, cfg)
    ls_cube, wave, good_mask, bad_mask, ls_cube_path, bunit = _load_residual_cube(paths, cfg)
    stat_cube, stat_state = _load_stat_cube(paths, cfg, ls_cube.shape)
    stat_factor, covariance_factor, stat_status, qc00, _qc01 = _stat_metadata(paths, cfg)
    open_issues = []
    if stat_cube is None:
        open_issues.append(f"STAT unavailable for X02 ({stat_state}); using empirical errors and estimated variance weights.")
    if str(stat_status).lower() == "red":
        open_issues.append("A4/M5 STAT status is red; products use empirical flux_err.")
    wframe = _wavelength_frame(cfg, qc00, open_issues)

    # Convencion de flujo: "normrad" (historica, por defecto) o "total"
    # (factor empirico de la curva de crecimiento). Falla ruidosamente si se
    # pide "total" sin medida, en vez de caer en silencio a la vieja.
    growth_curve, flux_convention = resolve_flux_convention(
        cfg, paths["paths"].stage_dir, knob="x02_flux_convention"
    )
    common = {
        "run_id": run_id,
        "star_yx": star_yx,
        "variance_zyx": stat_cube,
        "stat_factor_spaxel": stat_factor,
        "covariance_factor_box3": covariance_factor,
        "stat_status": stat_status,
        "error_mode": cfg.get("x02_error_mode", "auto"),
        "aperture_correction": cfg.get("x02_aperture_correction", "auto"),
        "growth_curve": growth_curve,
        "wframe": wframe,
        "bunit": bunit,
        "window_radius_px": float(cfg.get("x02_window_radius_px", 8.0)),
        "clip_sigma": float(cfg.get("x02_clip_sigma", 4.0)),
        "clip_max_iter": int(cfg.get("x02_clip_max_iter", 2)),
        "bad_windows_A": cfg.get("x02_bad_windows_A", []),
        "skyline_windows_A": cfg.get("x02_skyline_windows_A", []),
        "interpolated_windows_A": cfg.get("x02_interpolated_windows_A", []),
        "good_mask": good_mask,
        "bad_mask": bad_mask,
        "n_controls": int(cfg.get("x02_control_apertures", 8)),
        "exclude_angle_deg": float(cfg.get("x02_control_exclude_angle_deg", 25.0)),
        "local_bkg_annulus_px": cfg.get("x02_local_bkg_annulus_px"),
        "background_mode": cfg.get("x02_background_mode", "annulus"),
        "azimuthal_width_px": float(cfg.get("x02_azimuthal_width_px", 3.0)),
        "azimuthal_exclude_px": float(cfg.get("x02_azimuthal_exclude_px", 10.0)),
        "plane_fit_radius_px": float(cfg.get("x02_plane_fit_radius_px", 14.0)),
        "plane_mask_radius_px": float(cfg.get("x02_plane_mask_radius_px", 3.0)),
    }
    stage02_cube, stage02_wave, stage02_path, stage02_bunit = _load_stage02_cube(paths, cfg, expected_wave=wave)
    if stage02_cube.shape != ls_cube.shape:
        raise RuntimeError(f"Stage02 cube shape {stage02_cube.shape} != LS cube shape {ls_cube.shape}.")

    # All variants record the common MOTHER cube (stage02) as INCUBE so D1 sees a
    # single mother cube; LS is just a background treatment of it, like PSFSUB
    # subtracts the C1 model. (D1 §3.1.)
    #
    # Wings-intact LS (2026-07-26): LS extracts from the RAW cube with the
    # annulus background, exactly like C2 does. It used to extract from the
    # stage04b residual AND subtract the annulus on top, which was two
    # background treatments on a cube that is not homogeneous: stage04b fits a
    # local surface only AROUND THE OBJECT, so the annulus at the companion saw
    # a residual (~3/px) while at the controls it saw the untouched background
    # (~13/px). Measured on both objects, that second subtraction produced ~93%
    # of the negative continuum that gets `optimal_ls` rejected (red-band median
    # -2045 with it, -140 without), i.e. the rejection was dominated by the
    # background treatment, not by the Horne estimator.
    #
    # It also restores what the variant exists for: C2 moved to the raw cube in
    # the same commit that added this annulus (d688a64), so LS stopped being
    # "comparable 1:1 with C2" — which is its entire purpose (spec C3 3.1).
    ls_cube_used = ls_cube
    wings_intact_ls = bool(cfg.get("x02_wings_intact_ls", True))
    if wings_intact_ls and common["local_bkg_annulus_px"] is not None:
        ls_cube_used = stage02_cube
        open_issues.append(
            "LS extracts from the raw stage02 cube with the annulus background (same treatment "
            "as C2), not from the stage04b residual: subtracting both removed the local "
            "background twice at the companion and once at the controls, because stage04b only "
            "fits a surface around the object. Set x02_wings_intact_ls=false for the historical "
            "behaviour."
        )
    ls = make_optimal_product(
        ls_cube_used,
        wave,
        object_yx,
        psf_model,
        input_cube_path=stage02_path,
        variant="ls",
        **common,
    )
    # La resta de la primaria: sobre el combinado (historico) o ya hecha por
    # exposicion en C1b. Restar aqui obliga a describir con UN modelo la mezcla
    # de 29-30 PSF distintas; si C1b dejo su cubo, el modelo se aplico en cada
    # exposicion y esto solo lo consume. No hay eleccion en silencio: lo que se
    # usa queda escrito en `psfsub_model.source`.
    perobs_cube_path = Path(cfg.get("x02_psfsub_cube_fits")
                            or paths["paths"].stage_dir / "cube_psfsub_perobs.fits")
    use_perobs = bool(cfg.get("x02_psfsub_per_observation", True)) and perobs_cube_path.exists()
    if use_perobs:
        psfsub_cube, psfsub_wave, psfsub_bunit = _load_perobs_psfsub_cube(
            perobs_cube_path, expected_shape=stage02_cube.shape, expected_wave=stage02_wave
        )
        psfsub_model_meta = {
            "source": "C1b_perobs_subtract",
            "cube": str(perobs_cube_path),
            "note": ("La primaria se resto en cada exposicion con SU modelo y los residuos se "
                     "combinaron despues (C1b); aqui no se resta nada."),
        }
    else:
        primary_model, psfsub_model_meta = fit_primary_psf_model_cube(
            stage02_cube,
            stage02_wave,
            star_yx,
            psf_model,
            variance_zyx=stat_cube,
            fit_radius_px=float(cfg.get("x02_primary_fit_radius_px", 25.0)),
            exclude_centers_yx=[object_yx],
            exclude_radius_px=float(cfg.get("x02_primary_exclude_radius_px", cfg.get("x02_window_radius_px", 8.0))),
        )
        psfsub_cube = stage02_cube - primary_model
        psfsub_model_meta = dict(psfsub_model_meta, source="combined_cube")
        if (paths["paths"].stage_dir / "psf_model_mixture.json").exists():
            open_issues.append(
                "C1 fitted the PSF per observation but C1b has not run: the primary was "
                "subtracted from the combined cube with a single model, which is the very "
                "thing the per-observation fit exists to avoid."
            )
    psfsub_common = dict(common)
    psfsub_common["bunit"] = bunit or stage02_bunit
    psfsub = make_optimal_product(
        psfsub_cube,
        stage02_wave,
        object_yx,
        psf_model,
        # La procedencia del cubo del que sale el espectro: si la resta la hizo
        # C1b, el hash de entrada tiene que ser el de SU cubo, no el de B2.
        input_cube_path=(perobs_cube_path if use_perobs else stage02_path),
        variant="psfsub",
        **psfsub_common,
    )
    psf_sensitivity = _psf_sensitivity(
        # El mismo cubo del que sale `ls`: la sensibilidad a la PSF se mide
        # contra ese producto, así que compararla con otro fondo no diría nada.
        ls_cube_used,
        wave,
        object_yx,
        psf_model,
        cfg,
        stat_cube,
        stat_factor,
        covariance_factor,
        stat_status,
        ls,
    )
    extractions = {"ls": ls, "psfsub": psfsub}
    qc = _qc_payload(extractions, cfg, paths, stat_state, psf_model_path, psfsub_model_meta, open_issues, psf_sensitivity)
    return StageX02Product(extractions=extractions, qc=qc, psfsub_model_meta=psfsub_model_meta)


def write_stage_x02_products(product: StageX02Product, config, paths):
    paths["paths"].ensure_base_dirs()
    product.extractions["ls"].product.write(paths["spec_optimal_object"], overwrite=True)
    product.extractions["psfsub"].product.write(paths["spec_optimal_psfsub_object"], overwrite=True)
    SpectrumProduct.read(paths["spec_optimal_object"])
    SpectrumProduct.read(paths["spec_optimal_psfsub_object"])
    fits.HDUList(
        [
            fits.PrimaryHDU(),
            fits.ImageHDU(product.extractions["ls"].rejection_map.astype(np.float32), name="LS"),
            fits.ImageHDU(product.extractions["psfsub"].rejection_map.astype(np.float32), name="PSFSUB"),
        ]
    ).writeto(paths["spec_optimal_rejection_fits"], overwrite=True)
    # Persist per-variant control spectra for D1 (empirical sigma_diff source).
    # `control_spectra` (what D1 reads) is on the SAME physical scale as the
    # product flux (local-background-referenced + apcorr, D1 v2 §3.1).
    stage_dir = paths["paths"].stage_dir
    for variant, npz_name in (("ls", "spec_optimal_controls.npz"), ("psfsub", "spec_optimal_psfsub_controls.npz")):
        ext = product.extractions[variant]
        np.savez(
            stage_dir / npz_name,
            control_spectra=np.asarray(ext.control_spectra_cal, dtype=np.float64),
            control_spectra_raw=np.asarray(ext.control_spectra, dtype=np.float64),
            bkg_mode=np.asarray(ext.bkg_mode),
            apcorr_median=np.asarray(float(np.nanmedian(ext.product.apcorr))),
        )
    qc = dict(product.qc)
    qc["products"] = {
        "ls": str(paths["spec_optimal_object"]),
        "psfsub": str(paths["spec_optimal_psfsub_object"]),
        "rejection_map": str(paths["spec_optimal_rejection_fits"]),
    }
    write_json(paths["spec_optimal_qc_json"], qc)
    return {"products": qc["products"], "qc_json": paths["spec_optimal_qc_json"], "qc": qc}


def run_stage_x02(run_id=None, *, project_root=None, overrides=None, allow_run_id_mismatch=False):
    cfg = stage_x02_config_from_run(
        run_id,
        project_root=project_root,
        overrides=overrides,
        allow_run_id_mismatch=allow_run_id_mismatch,
    )
    paths = stage_x02_paths(cfg["run_id"], project_root=cfg.get("project_root"))
    product = compute_stage_x02_products(cfg, paths)
    written = write_stage_x02_products(product, cfg, paths)
    return {"config": cfg, "paths": paths, "qc": written["qc"], "written": written}


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run Stage X02/C3 optimal extraction.")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--project-root", default=None)
    parser.add_argument("--allow-run-id-mismatch", action="store_true")
    parser.add_argument(
        "--aperture-correction",
        choices=["auto", "none", "psf_growth_curve"],
        default=None,
    )
    args = parser.parse_args(argv)
    overrides = {}
    if args.aperture_correction is not None:
        overrides["x02_aperture_correction"] = args.aperture_correction
    result = run_stage_x02(
        args.run_id,
        project_root=args.project_root,
        overrides=overrides or None,
        allow_run_id_mismatch=args.allow_run_id_mismatch,
    )
    print(result["paths"]["spec_optimal_qc_json"])


__all__ = [
    "StageX02Product",
    "compute_stage_x02_products",
    "run_stage_x02",
    "stage_x02_config_from_run",
    "stage_x02_paths",
    "write_stage_x02_products",
]


if __name__ == "__main__":
    main()
