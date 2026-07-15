"""Stage H01b/E1b: blind spatio-spectral matched-filter detection maps.

Spec: docs/spec_E1b_codex_fov_detection.md (Julo et al. 2025 §3.3.1 + App. G).
Consumes the residual cubes of the halo-subtraction methods (sgf/lpm from
C5/C6; optimal_psfsub rebuilt with the production x02 routine), correlates
them with a Gaussian line template x chromatic PSF kernel, normalizes by
Andres-ring robust noise, and reports candidates above the frozen threshold.
Candidate promotion to ``companion`` (B3) is always a user checkpoint.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from astropy.io import fits
from scipy.ndimage import correlate

from ..config import load_run_config
from ..io import read_json, write_json
from ..paths import RunPaths
from ..psf import evaluate_psf_model
from ..stats import robust_sigma
from .stage_h01_detect import expected_line_center_A, gaussian_flux_template

SPEC_VERSION = "E1b_v1"
DEFAULT_THRESHOLD_SIGMA = 5.0
DEFAULT_R_MIN_PX = 3.0
DEFAULT_KERNEL_HALFSIZE_PX = 7
DEFAULT_RING_WIDTH_PX = 2


# ---------------------------------------------------------------------------
# Core (pure functions, unit-tested without run directories)
# ---------------------------------------------------------------------------

def spectral_template(wave_A, center_A, fwhm_A, good_mask=None):
    """Unit-norm Gaussian line template over the good channels (spec §2.1.1)."""

    wave = np.asarray(wave_A, dtype=np.float64)
    template = gaussian_flux_template(wave, float(center_A), float(fwhm_A))
    good = np.isfinite(template)
    if good_mask is not None:
        good &= np.asarray(good_mask, dtype=bool)
    template = np.where(good, template, 0.0)
    norm = float(np.sqrt(np.sum(template**2)))
    if norm <= 0:
        raise RuntimeError("Spectral template has zero norm; line outside coverage?")
    return template / norm


def spectral_map(residual_cube, template, *, min_cover=0.5):
    """M(y,x) = sum_l f(l) * residual(l,y,x), NaN-safe (spec §2.1.2)."""

    cube = np.asarray(residual_cube, dtype=np.float64)
    f = np.asarray(template, dtype=np.float64)
    support = f != 0.0
    finite = np.isfinite(cube)
    values = np.where(finite, cube, 0.0)
    m = np.tensordot(f, values, axes=(0, 0))
    cover = finite[support].sum(axis=0) / max(int(support.sum()), 1)
    m[cover < float(min_cover)] = np.nan
    return m


def psf_kernel_from_model(psf_model, wavelength_A, halfsize_px=DEFAULT_KERNEL_HALFSIZE_PX):
    """Unit-norm spatial kernel from the C1 chromatic PSF (spec §2.1.3)."""

    half = int(halfsize_px)
    yy, xx = np.mgrid[-half : half + 1, -half : half + 1].astype(np.float64)
    kernel = evaluate_psf_model(psf_model, float(wavelength_A), yy, xx)
    kernel = np.asarray(kernel, dtype=np.float64)
    norm = float(np.sqrt(np.nansum(kernel**2)))
    if not np.isfinite(norm) or norm <= 0:
        raise RuntimeError("PSF kernel has zero norm.")
    return kernel / norm


def empirical_kernel(pre_cube, template, star_yx, halfsize_px=DEFAULT_KERNEL_HALFSIZE_PX):
    """Fallback kernel: cutout of the pre-subtraction line-band image at the star."""

    star_map = spectral_map(pre_cube, template, min_cover=0.0)
    half = int(halfsize_px)
    y0, x0 = int(round(star_yx[0])), int(round(star_yx[1]))
    cut = star_map[y0 - half : y0 + half + 1, x0 - half : x0 + half + 1]
    if cut.shape != (2 * half + 1, 2 * half + 1) or not np.any(np.isfinite(cut)):
        raise RuntimeError("Cannot build empirical kernel: star cutout out of bounds.")
    cut = np.nan_to_num(cut, nan=0.0)
    cut -= float(np.median(cut))
    cut[cut < 0] = 0.0
    norm = float(np.sqrt(np.sum(cut**2)))
    if norm <= 0:
        raise RuntimeError("Empirical kernel has zero norm.")
    return cut / norm


def crosscorr_map(spec_map, kernel):
    """NaN-safe normalized cross-correlation (spec §2.1.4)."""

    m = np.asarray(spec_map, dtype=np.float64)
    k = np.asarray(kernel, dtype=np.float64)
    valid = np.isfinite(m).astype(np.float64)
    num = correlate(np.nan_to_num(m, nan=0.0), k, mode="constant", cval=0.0)
    denom = np.sqrt(np.maximum(correlate(valid, k**2, mode="constant", cval=0.0), 0.0))
    out = np.full(m.shape, np.nan)
    ok = denom > 1e-12
    out[ok] = num[ok] / denom[ok]
    out[valid == 0.0] = np.nan
    return out


def ring_noise_stats(corr_map, star_yx, *, ring_width_px=DEFAULT_RING_WIDTH_PX,
                     exclude_yx=None, exclude_radius_px=0.0):
    """Robust per-ring noise (Andres rings) + z map (spec §2.2, App. G)."""

    c = np.asarray(corr_map, dtype=np.float64)
    ny, nx = c.shape
    yy, xx = np.indices((ny, nx), dtype=np.float64)
    rr = np.hypot(yy - float(star_yx[0]), xx - float(star_yx[1]))
    use = np.isfinite(c)
    if exclude_yx is not None and float(exclude_radius_px) > 0:
        r2 = np.hypot(yy - float(exclude_yx[0]), xx - float(exclude_yx[1]))
        use &= r2 > float(exclude_radius_px)

    width = max(int(ring_width_px), 1)
    r_max = float(np.max(rr))
    zmap = np.full(c.shape, np.nan)
    rings = []
    lo = 0.0
    while lo < r_max:
        hi = lo + width
        ring = (rr >= lo) & (rr < hi)
        values = c[ring & use]
        if values.size >= 8:
            mu = float(np.median(values))
            sigma = float(robust_sigma(values))
            if np.isfinite(sigma) and sigma > 0:
                ring_finite = ring & np.isfinite(c)
                zmap[ring_finite] = (c[ring_finite] - mu) / sigma
                centred = (values - mu) / sigma
                m2 = float(np.mean(centred**2))
                m3 = float(np.mean(centred**3))
                m4 = float(np.mean(centred**4))
                rings.append(
                    {
                        "r_lo": float(lo),
                        "r_hi": float(hi),
                        "n_pixels": int(values.size),
                        "mu": mu,
                        "sigma": sigma,
                        "skew": m3 / m2**1.5 if m2 > 0 else None,
                        "kurtosis_excess": m4 / m2**2 - 3.0 if m2 > 0 else None,
                        # Fraction over NOISE pixels (companion excluded).
                        "frac_abs_z_gt3": float(np.mean(np.abs(centred) > 3.0)),
                    }
                )
        lo = hi
    return zmap, rings


def find_candidates(zmap, star_yx, *, threshold_sigma=DEFAULT_THRESHOLD_SIGMA,
                    r_min_px=DEFAULT_R_MIN_PX, min_separation_px=4.0,
                    exclude_yx=None, exclude_radius_px=0.0):
    """Local maxima of z above the frozen threshold, outside the AO core."""

    z = np.asarray(zmap, dtype=np.float64)
    ny, nx = z.shape
    yy, xx = np.indices((ny, nx), dtype=np.float64)
    rr = np.hypot(yy - float(star_yx[0]), xx - float(star_yx[1]))
    eligible = np.isfinite(z) & (z >= float(threshold_sigma)) & (rr >= float(r_min_px))
    if exclude_yx is not None and float(exclude_radius_px) > 0:
        r2 = np.hypot(yy - float(exclude_yx[0]), xx - float(exclude_yx[1]))
        eligible &= r2 > float(exclude_radius_px)

    order = np.argsort(z[eligible])[::-1]
    ys, xs = np.where(eligible)
    picked = []
    for idx in order:
        y, x = int(ys[idx]), int(xs[idx])
        window = z[max(0, y - 1) : y + 2, max(0, x - 1) : x + 2]
        if not np.isclose(z[y, x], np.nanmax(window)):
            continue
        if any(np.hypot(y - p["y"], x - p["x"]) < float(min_separation_px) for p in picked):
            continue
        picked.append(
            {
                "y": y,
                "x": x,
                "r_px": float(rr[y, x]),
                "z": float(z[y, x]),
            }
        )
    return picked


# ---------------------------------------------------------------------------
# Stage wiring
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class StageH01bProduct:
    maps: dict
    qc: dict


def stage_h01b_paths(run_id, project_root=None):
    root = Path(project_root or Path.cwd()).resolve()
    paths = RunPaths.from_project_root(run_id, root)
    plot_dir = paths.plot_stage_dir("stage_h01b")
    return {
        "paths": paths,
        "stage00q_qc_json": paths.stage_dir / "stage00q_qc.json",
        "stage01c_qc_json": paths.stage_dir / "stage01c_qc.json",
        "stage02_cube_fits": paths.stage_dir / "stage02_xcorr_cube_stack.fits",
        "psf_model_json": paths.stage_dir / "psf_model.json",
        "sgf_residual_cube": paths.stage_dir / "stage_x04_sgf_residual_cube.fits",
        "lpm_residual_cube": paths.stage_dir / "stage_x05_lpm_residual_cube.fits",
        "map_fits_template": paths.stage_dir / "stage_h01b_fovmap_{method}.fits",
        "stage_h01b_qc_json": paths.stage_dir / "stage_h01b_qc.json",
        "plot_dir": plot_dir,
        "summary_plot": plot_dir / "stage_h01b_maps.png",
    }


def stage_h01b_config_from_run(run_id=None, *, project_root=None, overrides=None, allow_run_id_mismatch=False):
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
    cfg.setdefault("h01b_line_rest_A", 6562.8)
    cfg.setdefault("h01b_threshold_sigma", DEFAULT_THRESHOLD_SIGMA)
    cfg.setdefault("h01b_r_min_px", DEFAULT_R_MIN_PX)
    cfg.setdefault("h01b_kernel_halfsize_px", DEFAULT_KERNEL_HALFSIZE_PX)
    cfg.setdefault("h01b_ring_width_px", DEFAULT_RING_WIDTH_PX)
    cfg.setdefault("h01b_include_psfsub", True)
    return cfg


def _read_optional_json(path):
    path = Path(path)
    if not path.exists():
        return {}
    return read_json(path)


def _positions(paths, cfg):
    qc = _read_optional_json(paths["stage01c_qc_json"])
    star = None
    companion = None
    for key in ("primary", "star"):
        if isinstance(qc.get(key), dict) and qc[key].get("pos_yx"):
            star = tuple(float(v) for v in qc[key]["pos_yx"])
            break
    for key in ("companion", "target", "object"):
        if isinstance(qc.get(key), dict) and qc[key].get("pos_yx"):
            companion = tuple(float(v) for v in qc[key]["pos_yx"])
            break
    if star is None and cfg.get("h01b_star_yx") is not None:
        star = tuple(float(v) for v in cfg["h01b_star_yx"])
    if star is None:
        raise RuntimeError("E1b needs the star position (stage01c QC or h01b_star_yx).")
    return star, companion


def _rv_sys_kms(cfg):
    """Systemic RV with the run-config key precedence h01b > h01 > generic."""

    for key in ("h01b_rv_sys_kms", "h01_rv_sys_kms", "rv_sys_kms"):
        value = cfg.get(key)
        if value is not None:
            try:
                return float(value)
            except Exception:
                continue
    return 0.0


def _lsf_fwhm(paths, cfg, default=2.6):
    qc00 = _read_optional_json(paths["stage00q_qc_json"])
    value = qc00.get("lsf", {}).get("fwhm_A") or qc00.get("cube", {}).get("lsf_fwhm_A") or cfg.get("lsf_fwhm_A", default)
    try:
        value = float(value)
    except Exception:
        value = float(default)
    return value if np.isfinite(value) and value > 0 else float(default)


def _load_residual(path):
    with fits.open(path, memmap=False) as hdul:
        cube = np.asarray(hdul["RESIDUAL"].data if "RESIDUAL" in hdul else hdul[0].data, dtype=np.float64)
        wave = np.asarray(hdul["WAVELENGTH"].data, dtype=np.float64)
    return cube, wave


def _load_stage02_mean(paths):
    with fits.open(paths["stage02_cube_fits"], memmap=False) as hdul:
        cube = np.asarray(hdul["CUBES"].data, dtype=np.float64)
        wave = np.asarray(hdul["WAVELENGTH"].data, dtype=np.float64)
    if cube.ndim == 4:
        with np.errstate(all="ignore"):
            cube = np.nanmean(cube, axis=0)
    return cube, wave


def _residual_cubes(paths, cfg, star_yx, companion_yx, open_issues):
    cubes = {}
    wave_ref = None
    for method, key in (("sgf", "sgf_residual_cube"), ("lpm", "lpm_residual_cube")):
        path = Path(paths[key])
        if path.exists():
            cube, wave = _load_residual(path)
            cubes[method] = (cube, wave)
            wave_ref = wave if wave_ref is None else wave_ref
        else:
            open_issues.append(f"Residual cube for {method} not found ({path.name}); method skipped.")
    if bool(cfg.get("h01b_include_psfsub", True)) and Path(paths["stage02_cube_fits"]).exists():
        try:
            from ..extraction.optimal import fit_primary_psf_model_cube

            pre_cube, wave = _load_stage02_mean(paths)
            psf_model = _read_optional_json(paths["psf_model_json"])
            if psf_model:
                exclude = [companion_yx] if companion_yx is not None else None
                primary_model, _meta = fit_primary_psf_model_cube(
                    pre_cube, wave, star_yx, psf_model,
                    fit_radius_px=float(cfg.get("x02_primary_fit_radius_px", 25.0)),
                    exclude_centers_yx=exclude,
                    exclude_radius_px=float(cfg.get("x02_primary_exclude_radius_px", 8.0)),
                )
                cubes["optimal_psfsub"] = (pre_cube - primary_model, wave)
            else:
                open_issues.append("psf_model.json missing; optimal_psfsub map skipped.")
        except Exception as exc:  # pragma: no cover - defensive path
            open_issues.append(f"optimal_psfsub residual rebuild failed: {type(exc).__name__}: {exc}.")
    if not cubes:
        raise RuntimeError("E1b found no residual cubes; run C5/C6 (or enable psfsub) first.")
    return cubes


def compute_stage_h01b_products(config, paths=None):
    cfg = dict(config)
    root = Path(cfg.get("project_root") or Path.cwd()).resolve()
    paths = stage_h01b_paths(cfg["run_id"], root) if paths is None else paths
    open_issues = []

    star_yx, companion_yx = _positions(paths, cfg)
    lsf_fwhm_A = _lsf_fwhm(paths, cfg)
    rv_sys = _rv_sys_kms(cfg)
    line_rest = float(cfg.get("h01b_line_rest_A", 6562.8))
    line_center = expected_line_center_A(line_rest, rv_sys)
    threshold = float(cfg.get("h01b_threshold_sigma", DEFAULT_THRESHOLD_SIGMA))
    r_min = float(cfg.get("h01b_r_min_px", DEFAULT_R_MIN_PX))
    halfsize = int(cfg.get("h01b_kernel_halfsize_px", DEFAULT_KERNEL_HALFSIZE_PX))
    ring_width = int(cfg.get("h01b_ring_width_px", DEFAULT_RING_WIDTH_PX))

    cubes = _residual_cubes(paths, cfg, star_yx, companion_yx, open_issues)

    psf_model = _read_optional_json(paths["psf_model_json"])
    kernel_source = "c1_psf_model" if psf_model else "empirical_star_cutout"

    maps = {}
    methods_qc = {}
    exclude_radius = 1.5 * max(lsf_fwhm_A, 4.0)  # px; guards mu/sigma of rings
    for method, (cube, wave) in cubes.items():
        template = spectral_template(wave, line_center, lsf_fwhm_A)
        m = spectral_map(cube, template)
        if psf_model:
            kernel = psf_kernel_from_model(psf_model, line_center, halfsize)
        else:
            pre_cube, pre_wave = _load_stage02_mean(paths)
            kernel = empirical_kernel(pre_cube, spectral_template(pre_wave, line_center, lsf_fwhm_A), star_yx, halfsize)
        corr = crosscorr_map(m, kernel)
        zmap, rings = ring_noise_stats(
            corr, star_yx, ring_width_px=ring_width,
            exclude_yx=companion_yx, exclude_radius_px=exclude_radius if companion_yx else 0.0,
        )
        candidates = find_candidates(
            zmap, star_yx, threshold_sigma=threshold, r_min_px=r_min,
            exclude_yx=companion_yx, exclude_radius_px=exclude_radius if companion_yx else 0.0,
        )
        known = None
        if companion_yx is not None:
            cy, cx = int(round(companion_yx[0])), int(round(companion_yx[1]))
            zval = zmap[cy, cx] if 0 <= cy < zmap.shape[0] and 0 <= cx < zmap.shape[1] else np.nan
            known = {"y": float(companion_yx[0]), "x": float(companion_yx[1]),
                     "z": None if not np.isfinite(zval) else float(zval)}
        maps[method] = {"spectral_map": m, "corr_map": corr, "z_map": zmap}
        methods_qc[method] = {
            "map_fits": str(paths["map_fits_template"]).format(method=method),
            "candidates": candidates,
            "known_source": known,
            "ring_noise": rings,
            "n_pixels_valid": int(np.sum(np.isfinite(zmap))),
        }

    checks = {
        "v1_maps_written": bool(maps),
        "v2_known_source_reported": (
            None if companion_yx is None
            else bool(all(mq["known_source"] is not None for mq in methods_qc.values()))
        ),
        "v3_ring_noise_written": bool(all(mq["ring_noise"] for mq in methods_qc.values())),
        "v4_no_candidate_at_star_core": bool(
            all(c["r_px"] >= r_min for mq in methods_qc.values() for c in mq["candidates"])
        ),
    }
    qc = {
        "stage": "h01b_fovmap",
        "spec_version": SPEC_VERSION,
        "run_id": str(cfg["run_id"]),
        "params": {
            "line_rest_A": line_rest,
            "line_center_A": float(line_center),
            "lsf_fwhm_A": float(lsf_fwhm_A),
            "threshold_sigma": threshold,
            "r_min_px": r_min,
            "kernel_halfsize_px": halfsize,
            "ring_width_px": ring_width,
            "kernel_source": kernel_source,
        },
        "star_yx": [float(star_yx[0]), float(star_yx[1])],
        "companion_yx": None if companion_yx is None else [float(companion_yx[0]), float(companion_yx[1])],
        "methods": methods_qc,
        "checks": checks,
        "open_issues": open_issues,
    }
    return StageH01bProduct(maps=maps, qc=qc)


def write_stage_h01b_products(product: StageH01bProduct, config, paths):
    paths["paths"].ensure_base_dirs()
    paths["plot_dir"].mkdir(parents=True, exist_ok=True)
    for method, data in product.maps.items():
        out = Path(str(paths["map_fits_template"]).format(method=method))
        fits.HDUList(
            [
                fits.PrimaryHDU(),
                fits.ImageHDU(np.asarray(data["spectral_map"], dtype=np.float32), name="SPECMAP"),
                fits.ImageHDU(np.asarray(data["corr_map"], dtype=np.float32), name="CORR"),
                fits.ImageHDU(np.asarray(data["z_map"], dtype=np.float32), name="Z"),
            ]
        ).writeto(out, overwrite=True)
    _write_stage_h01b_plot(product, paths)
    write_json(paths["stage_h01b_qc_json"], product.qc)
    return {"qc_json": paths["stage_h01b_qc_json"], "qc": product.qc}


def _write_stage_h01b_plot(product: StageH01bProduct, paths):
    import os

    cache_dir = Path(os.environ.get("TMPDIR", "/tmp")) / "musepipe_matplotlib"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    methods = list(product.maps)
    fig, axes = plt.subplots(1, max(len(methods), 1), figsize=(5.2 * max(len(methods), 1), 4.6))
    axes = np.atleast_1d(axes)
    for ax, method in zip(axes, methods):
        z = product.maps[method]["z_map"]
        im = ax.imshow(z, origin="lower", cmap="viridis", vmin=-3, vmax=8)
        sy, sx = product.qc["star_yx"]
        ax.plot(sx, sy, "*", color="red", ms=10)
        if product.qc.get("companion_yx"):
            cy, cx = product.qc["companion_yx"]
            ax.plot(cx, cy, "o", mfc="none", mec="lime", ms=12)
        for cand in product.qc["methods"][method]["candidates"]:
            ax.plot(cand["x"], cand["y"], "s", mfc="none", mec="orange", ms=10)
        ax.set_title(f"{method} · z map")
        fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(paths["summary_plot"], dpi=120)
    plt.close(fig)


def run_stage_h01b(run_id=None, *, project_root=None, overrides=None, allow_run_id_mismatch=False):
    cfg = stage_h01b_config_from_run(
        run_id,
        project_root=project_root,
        overrides=overrides,
        allow_run_id_mismatch=allow_run_id_mismatch,
    )
    paths = stage_h01b_paths(cfg["run_id"], project_root=cfg.get("project_root"))
    product = compute_stage_h01b_products(cfg, paths)
    written = write_stage_h01b_products(product, cfg, paths)
    return {"config": cfg, "paths": paths, "qc": written["qc"], "written": written}


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run Stage H01b/E1b FoV matched-filter maps.")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--project-root", default=None)
    parser.add_argument("--allow-run-id-mismatch", action="store_true")
    args = parser.parse_args(argv)
    result = run_stage_h01b(
        args.run_id,
        project_root=args.project_root,
        allow_run_id_mismatch=args.allow_run_id_mismatch,
    )
    print(result["paths"]["stage_h01b_qc_json"])


__all__ = [
    "SPEC_VERSION",
    "StageH01bProduct",
    "compute_stage_h01b_products",
    "crosscorr_map",
    "empirical_kernel",
    "find_candidates",
    "psf_kernel_from_model",
    "ring_noise_stats",
    "run_stage_h01b",
    "spectral_map",
    "spectral_template",
    "stage_h01b_config_from_run",
    "stage_h01b_paths",
    "write_stage_h01b_products",
]


if __name__ == "__main__":
    main()
