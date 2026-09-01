"""Production extractor wrappers for Stage H04 / E4 (in-memory injection recovery).

These adapt the real C2/C3/C4 extraction chain into the callable contract that
``compute_stage_h04_products`` expects (``extractor(cube, wave_A, case, method,
config) -> SpectrumProduct``). The wrappers reuse the SAME high-level builders
and per-stage config resolution the production stages use, so the recovered
spectra follow the production extraction to the extent noted below.

Fidelity caveats (documented; this is the in-memory path chosen for the
PROVISIONAL ROXs12b_B_adp run, spec E4 option B):

* Each method feeds its production input cube: aperture and optimal-ls use the
  stage04b local-surface residual of the injected cube; optimal-psfsub uses
  ``stage02 - fit_primary_psf_model_cube``; psffit fits star+companion on the
  injected stage02 cube. This mirrors ``compute_stage_x0N_products``.
* The stage04b knobs, aperture list, window radii, fit radii and STAT handling
  are pulled from ``stage_xNN_config_from_run`` / ``stage04b_config_from_run`` so
  they equal production.
* The x01 "wings-intact" aperture-correction refinement (which reloads the raw
  cube from disk) is NOT reproduced; apcorr uses the PSF growth curve directly.
* The builders' internal control apertures are disabled (``n_controls=0``); E4
  characterises positions through its own grid (real + 3 controls), not through
  the per-extraction control ring.
* sgf/lpm (C5/C6) exploit that both subtractions are LINEAR in the data for a
  fixed reference spectrum: ``residual(base + injection) = residual(base) +
  (I - L)(injection)``. The base residual and reference are computed ONCE from
  the un-injected cube; each case applies the kernel only to the injection
  delta on its spatial support (WP-H2 optimization, exact given the fixed
  reference). Fidelity note: production re-estimates the reference per
  exposure from the data; a faint injected fake would perturb that median over
  >=50 spaxels negligibly, so the fixed base reference is the honest
  approximation (documented here like the other caveats).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from astropy.io import fits

from ..extraction.aperture import aperture_label, default_apertures, make_aperture_product
from ..extraction.optimal import fit_primary_psf_model_cube, make_optimal_product
from ..extraction.psffit import make_psffit_products
from ..halosub import lpm_subtract, reference_spectrum, select_reference_spaxels, sgf_subtract
from ..localfit import subtract_local_surface_cube
from .halosub_stage import wave_range_mask
from .stage04b_local_surface import stage04b_config_from_run
from .stage_x01_aperture import (
    _load_positions as _x01_load_positions,
    _stat_metadata as _x01_stat_metadata,
    stage_x01_config_from_run,
    stage_x01_paths,
)
from .stage_x02_optimal import stage_x02_config_from_run
from .stage_x03_psffit import stage_x03_config_from_run
from .stage_x04_sgf import stage_x04_config_from_run
from .stage_x05_lpm import masked_line_windows, stage_x05_config_from_run


def _load_stat_cube_direct(cfg, paths):
    """Load a 3D STAT cube from the stage02 file's STAT HDU, or None."""
    path = Path(cfg.get("x01_stat_cube_fits") or paths["stage02_cube_fits"])
    if not path.exists():
        return None
    with fits.open(path, memmap=True) as hdul:
        if "STAT" not in hdul:
            return None
        stat = np.asarray(hdul["STAT"].data, dtype=np.float64)
    if stat.ndim == 4:
        stat = np.nansum(stat, axis=0) / float(stat.shape[0] ** 2)
    return stat if stat.ndim == 3 else None


def _box3_aperture(cfg):
    for ap in cfg.get("x01_apertures", default_apertures()):
        if aperture_label(ap) == "box3":
            return ap
    return {"name": "box3", "kind": "box", "size": 3}


def _s04b_kwargs(run_id, project_root):
    cfg = stage04b_config_from_run(run_id, project_root=project_root)
    return {
        "fit_radius_px": float(cfg["fit_radius_px"]),
        "mask_radius_px": float(cfg["mask_radius_px"]),
        "model_kind": cfg["local_model_kind"],
        "extra_exclusion_radius_px": float(cfg["other_mask_radius_px"]),
        "sigma_clip": float(cfg["local_fit_sigma_clip"]),
        "max_iter": int(cfg["local_fit_max_iter"]),
        "min_fit_pixels": int(cfg["local_fit_min_pixels"]),
        "mask_other_objects": bool(cfg.get("mask_other_objects", True)),
    }


def build_halosub_extractors(
    *,
    base_cube,
    wave_A,
    star_yx,
    companion_yx,
    psf_model,
    run_id,
    x04_cfg,
    x05_cfg,
    box3,
    stat_for=lambda cube: None,
    stat_factor=1.0,
    covariance_factor=1.0,
    stat_status="unknown",
):
    """{``sgf``/``lpm``: extractor} adapting C5/C6 to the H04 contract.

    Base residual + reference are computed lazily ONCE per method from
    ``base_cube``; each case subtracts only the injection delta on its spatial
    support (linearity of both kernels — see module docstring).
    """

    wave = np.asarray(wave_A, dtype=np.float64)
    base64 = np.asarray(base_cube, dtype=np.float64)
    if base64.ndim == 4:
        if base64.shape[0] != 1:
            raise RuntimeError("halosub extractors expect a single combined cube.")
        base64 = base64[0]
    state_cache: dict[str, dict] = {}

    def _method_cfg(method):
        return x04_cfg if method == "sgf" else x05_cfg

    def _base_state(method):
        state = state_cache.get(method)
        if state is not None:
            return state
        cfg_m = _method_cfg(method)
        wmask = wave_range_mask(wave, cfg_m.get("halosub_wave_range_A"))
        keep, _keep_qc = select_reference_spaxels(
            base64,
            flux_lo_frac=float(cfg_m.get("halosub_flux_mask_lo", 0.01)),
            flux_hi_frac=float(cfg_m.get("halosub_flux_mask_hi", 0.1)),
            wave_mask=wmask,
            exclude_yx=[companion_yx],
            exclude_radius_px=float(cfg_m.get("halosub_exclude_radius_px", 3.0)),
        )
        s_hat = reference_spectrum(base64, keep)
        if method == "sgf":
            res = sgf_subtract(
                base64, s_hat,
                window=int(cfg_m.get("sgf_window", 101)),
                degree=int(cfg_m.get("sgf_degree", 1)),
            )
        else:
            res = lpm_subtract(
                base64, wave, s_hat,
                degree=int(cfg_m.get("lpm_degree", 4)),
                line_windows_A=masked_line_windows(cfg_m),
            )
        residual = res.residual_cube.copy()
        residual[~wmask] = np.nan
        state = {"s_hat": s_hat, "residual": residual, "wmask": wmask}
        state_cache[method] = state
        return state

    def _delta_residual(method, delta_sup):
        cfg_m = _method_cfg(method)
        s_hat = state_cache[method]["s_hat"]
        if method == "sgf":
            res = sgf_subtract(
                delta_sup, s_hat,
                window=int(cfg_m.get("sgf_window", 101)),
                degree=int(cfg_m.get("sgf_degree", 1)),
            )
        else:
            res = lpm_subtract(
                delta_sup, wave, s_hat,
                degree=int(cfg_m.get("lpm_degree", 4)),
                line_windows_A=masked_line_windows(cfg_m),
            )
        return res.residual_cube

    # The box3 product only reads pixels within the aperture + background
    # annulus (r_out 14 px) of the case position; the injected Moffat wings
    # beyond that are <~1e-4 of the peak and the subtraction is per-spaxel
    # independent, so clipping the delta support to this box is exact for the
    # measurement while keeping the per-case cost bounded (same argument as
    # the E5 kernel-window clip).
    support_halfsize_px = 20.0

    def _extract(cube, case, method):
        state = _base_state(method)
        cfg_m = _method_cfg(method)
        prefix = "x04" if method == "sgf" else "x05"
        cube64 = np.asarray(cube, dtype=np.float64)
        delta = cube64 - base64
        with np.errstate(all="ignore"):
            col_amp = np.nanmax(np.abs(delta), axis=0)
        col_amp = np.nan_to_num(col_amp, nan=0.0)
        amax = float(np.max(col_amp)) if col_amp.size else 0.0
        residual = state["residual"].copy()
        if amax > 0.0:
            support = col_amp > 1e-9 * amax
            yy, xx = np.indices(support.shape, dtype=np.float64)
            support &= (np.abs(yy - float(case.position_y)) <= support_halfsize_px) & (
                np.abs(xx - float(case.position_x)) <= support_halfsize_px
            )
            delta_sup = delta[:, support][:, :, None]
            resid_delta = _delta_residual(method, delta_sup)[:, :, 0]
            residual[:, support] = residual[:, support] + resid_delta
        comp_yx = (float(case.position_y), float(case.position_x))
        ext = make_aperture_product(
            residual, wave, comp_yx, dict(box3),
            run_id=run_id, input_cube_path="h04:inmemory", star_yx=star_yx,
            stat_zyx=stat_for(residual), stat_factor=stat_factor,
            covariance_factor=covariance_factor, stat_status=stat_status,
            error_mode=cfg_m.get(f"{prefix}_error_mode", "auto"),
            psf_model=psf_model,
            aperture_correction=cfg_m.get(f"{prefix}_aperture_correction", "auto"),
            bad_windows_A=cfg_m.get(f"{prefix}_bad_windows_A", []),
            skyline_windows_A=cfg_m.get(f"{prefix}_skyline_windows_A", []),
            interpolated_windows_A=cfg_m.get(f"{prefix}_interpolated_windows_A", []),
            n_controls=0,
            annulus_bkg_px=cfg_m.get(f"{prefix}_annulus_bkg_px"),
        )
        ext.product.header["METHOD"] = method
        ext.product.header["BKGMODE"] = f"{method}_residual+{ext.bkg_mode}"
        return ext.product

    def sgf_extractor(cube, wave_A, case, method, config):  # noqa: ARG001
        return _extract(cube, case, "sgf")

    def lpm_extractor(cube, wave_A, case, method, config):  # noqa: ARG001
        return _extract(cube, case, "lpm")

    return {"sgf": sgf_extractor, "lpm": lpm_extractor}


def _x03_with_overrides(x03_cfg, config):
    """Los knobs de C4 que una sonda puede barrer sin tocar el config del run.

    Existe para medir configuraciones del extractor —los radios del ajuste de
    `psffit`— sin escribir en `runs/<run>/config/config.json`: la unica forma
    anterior de cambiarlos era editar ese fichero, que es justo lo que no puede
    hacer una medida que no debe mover los productos publicados. Vacio por
    defecto, asi que la cadena resuelve exactamente igual que antes.
    """

    overrides = config.get("h04_x03_overrides") or {}
    if not overrides:
        return x03_cfg
    return {**x03_cfg, **dict(overrides)}


def build_production_extractors(config, paths, *, wave_A, psf_model, base_cube=None):
    """Return {method: callable} adapting C2/C3/C4/C5/C6 to the H04 extractor contract."""

    run_id = config["run_id"]
    root = Path(config.get("project_root") or Path.cwd()).resolve()

    x01_paths = stage_x01_paths(run_id, root)
    x01_cfg = stage_x01_config_from_run(run_id, project_root=root)
    x02_cfg = stage_x02_config_from_run(run_id, project_root=root)
    x03_cfg = stage_x03_config_from_run(run_id, project_root=root)
    x04_cfg = stage_x04_config_from_run(run_id, project_root=root)
    x05_cfg = stage_x05_config_from_run(run_id, project_root=root)
    x03_cfg = _x03_with_overrides(x03_cfg, config)

    # Star position (companion position comes from each case; the REAL
    # companion position is excluded from the halosub reference spectrum).
    object_yx, star_yx, _pos_path, _pos_qc = _x01_load_positions(x01_paths, x01_cfg)

    # STAT weights + metadata (shared; injection is additive so base STAT holds).
    stat_cube = _load_stat_cube_direct(x01_cfg, x01_paths)
    stat_factor, covariance_factor, stat_status, _qc00, _qc01 = _x01_stat_metadata(x01_paths, x01_cfg)

    s04b = _s04b_kwargs(run_id, root)
    box3 = _box3_aperture(x01_cfg)
    wave = np.asarray(wave_A, dtype=np.float64)

    def _stat_for(cube):
        if stat_cube is not None and np.shape(stat_cube) == np.shape(cube):
            return stat_cube
        return None

    def _residual(cube, comp_yx):
        extra = [tuple(star_yx)] if s04b["mask_other_objects"] else None
        residual, _model, _nfit = subtract_local_surface_cube(
            np.asarray(cube, dtype=np.float32),
            target_yx=(float(comp_yx[0]), float(comp_yx[1])),
            fit_radius_px=s04b["fit_radius_px"],
            mask_radius_px=s04b["mask_radius_px"],
            model_kind=s04b["model_kind"],
            extra_exclusion_yx=extra,
            extra_exclusion_radius_px=s04b["extra_exclusion_radius_px"],
            sigma_clip=s04b["sigma_clip"],
            max_iter=s04b["max_iter"],
            min_fit_pixels=s04b["min_fit_pixels"],
        )
        return np.asarray(residual, dtype=np.float64)

    def aperture_extractor(cube, wave_A, case, method, config):  # noqa: ARG001
        comp_yx = (float(case.position_y), float(case.position_x))
        resid = _residual(cube, comp_yx)
        ext = make_aperture_product(
            resid, wave, comp_yx, box3,
            run_id=run_id, input_cube_path="h04:inmemory", star_yx=star_yx,
            stat_zyx=_stat_for(resid), stat_factor=stat_factor,
            covariance_factor=covariance_factor, stat_status=stat_status,
            error_mode=x01_cfg.get("x01_error_mode", "auto"),
            psf_model=psf_model,
            aperture_correction=x01_cfg.get("x01_aperture_correction", "auto"),
            bad_windows_A=x01_cfg.get("x01_bad_windows_A", []),
            skyline_windows_A=x01_cfg.get("x01_skyline_windows_A", []),
            interpolated_windows_A=x01_cfg.get("x01_interpolated_windows_A", []),
            n_controls=0,
        )
        return ext.product

    def optimal_ls_extractor(cube, wave_A, case, method, config):  # noqa: ARG001
        comp_yx = (float(case.position_y), float(case.position_x))
        resid = _residual(cube, comp_yx)
        ext = make_optimal_product(
            resid, wave, comp_yx, psf_model,
            run_id=run_id, input_cube_path="h04:inmemory", variant="ls",
            star_yx=star_yx, variance_zyx=_stat_for(resid),
            stat_factor_spaxel=stat_factor, covariance_factor_box3=covariance_factor,
            stat_status=stat_status, error_mode=x02_cfg.get("x02_error_mode", "auto"),
            window_radius_px=float(x02_cfg.get("x02_window_radius_px", 8.0)),
            clip_sigma=float(x02_cfg.get("x02_clip_sigma", 4.0)),
            clip_max_iter=int(x02_cfg.get("x02_clip_max_iter", 2)),
            bad_windows_A=x02_cfg.get("x02_bad_windows_A", []),
            skyline_windows_A=x02_cfg.get("x02_skyline_windows_A", []),
            interpolated_windows_A=x02_cfg.get("x02_interpolated_windows_A", []),
            n_controls=0,
            local_bkg_annulus_px=x02_cfg.get("x02_local_bkg_annulus_px"),
        )
        return ext.product

    def optimal_psfsub_extractor(cube, wave_A, case, method, config):  # noqa: ARG001
        comp_yx = (float(case.position_y), float(case.position_x))
        cube64 = np.asarray(cube, dtype=np.float64)
        primary_model, _meta = fit_primary_psf_model_cube(
            cube64, wave, star_yx, psf_model, variance_zyx=_stat_for(cube64),
            fit_radius_px=float(x02_cfg.get("x02_primary_fit_radius_px", 25.0)),
            exclude_centers_yx=[comp_yx],
            exclude_radius_px=float(x02_cfg.get("x02_primary_exclude_radius_px", x02_cfg.get("x02_window_radius_px", 8.0))),
        )
        psfsub_cube = cube64 - primary_model
        ext = make_optimal_product(
            psfsub_cube, wave, comp_yx, psf_model,
            run_id=run_id, input_cube_path="h04:inmemory", variant="psfsub",
            star_yx=star_yx, variance_zyx=_stat_for(psfsub_cube),
            stat_factor_spaxel=stat_factor, covariance_factor_box3=covariance_factor,
            stat_status=stat_status, error_mode=x02_cfg.get("x02_error_mode", "auto"),
            window_radius_px=float(x02_cfg.get("x02_window_radius_px", 8.0)),
            clip_sigma=float(x02_cfg.get("x02_clip_sigma", 4.0)),
            clip_max_iter=int(x02_cfg.get("x02_clip_max_iter", 2)),
            bad_windows_A=x02_cfg.get("x02_bad_windows_A", []),
            skyline_windows_A=x02_cfg.get("x02_skyline_windows_A", []),
            interpolated_windows_A=x02_cfg.get("x02_interpolated_windows_A", []),
            n_controls=0,
            local_bkg_annulus_px=x02_cfg.get("x02_local_bkg_annulus_px"),
        )
        return ext.product

    def psffit_extractor(cube, wave_A, case, method, config):  # noqa: ARG001
        comp_yx = (float(case.position_y), float(case.position_x))
        cube64 = np.asarray(cube, dtype=np.float64)
        prods = make_psffit_products(
            cube64, wave, star_yx, comp_yx, psf_model,
            run_id=run_id, input_cube_path="h04:inmemory",
            variance_zyx=_stat_for(cube64), stat_factor_spaxel=stat_factor,
            covariance_factor_box3=covariance_factor, stat_status=stat_status,
            error_mode=x03_cfg.get("x03_error_mode", "auto"),
            star_radius_px=float(x03_cfg.get("x03_star_radius_px", 20.0)),
            comp_radius_px=float(x03_cfg.get("x03_comp_radius_px", 12.0)),
            bad_windows_A=x03_cfg.get("x03_bad_windows_A", []),
            skyline_windows_A=x03_cfg.get("x03_skyline_windows_A", []),
            interpolated_windows_A=x03_cfg.get("x03_interpolated_windows_A", []),
            n_controls=0,
        )
        return prods.companion

    def _load_base_cube():
        path = Path(config.get("h04_stage02_cube_fits", paths["stage02_cube_fits"]))
        with fits.open(path, memmap=False) as hdul:
            cube = np.asarray(hdul["CUBES"].data if "CUBES" in hdul else hdul[0].data, dtype=np.float64)
        if cube.ndim == 4:
            if cube.shape[0] != 1:
                raise RuntimeError("halosub extractors expect a single combined cube.")
            cube = cube[0]
        return cube

    halosub = build_halosub_extractors(
        base_cube=base_cube if base_cube is not None else _load_base_cube(),
        wave_A=wave,
        star_yx=star_yx,
        companion_yx=object_yx,
        psf_model=psf_model,
        run_id=run_id,
        x04_cfg=x04_cfg,
        x05_cfg=x05_cfg,
        box3=box3,
        stat_for=_stat_for,
        stat_factor=stat_factor,
        covariance_factor=covariance_factor,
        stat_status=stat_status,
    )

    return {
        "aperture": aperture_extractor,
        "optimal_ls": optimal_ls_extractor,
        "optimal_psfsub": optimal_psfsub_extractor,
        "psffit": psffit_extractor,
        **halosub,
    }


__all__ = ["build_halosub_extractors", "build_production_extractors"]
