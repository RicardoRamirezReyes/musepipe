"""Stage H05/E5: contrast curves from ring-grid fake injections.

Spec: docs/spec_E5_codex_contrast_curves.md (Julo et al. 2025 §3.3.2).
Reuses the E1b detection chain (spectral template x C1 kernel x ring-noise
normalization). For a fixed reference spectrum the sgf/lpm subtraction is
linear, so the base residual map and its ring statistics are computed once
per method and each injection only contributes its delta at the injected
position — the full grid costs seconds, and the decision uses the REAL noise
realization at each position (as in the paper).
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from astropy.io import fits

from ..config import load_run_config

from ..halosub import lpm_subtract, reference_spectrum, select_reference_spaxels, sgf_subtract
from ..io import write_json
from ..paths import RunPaths
from ..psf import evaluate_psf_model
from .halosub_stage import wave_range_mask
from .stage_h01_detect import expected_line_center_A
from .stage_h01b_fovmap import (
    crosscorr_map,
    psf_kernel_from_model,
    ring_noise_stats,
    spectral_map,
    spectral_template,
    stage_h01b_paths,
)
from .stage_h01b_fovmap import _load_residual, _load_stage02_mean, _lsf_fwhm, _positions, _read_optional_json, _rv_sys_kms
from .stage_x04_sgf import stage_x04_config_from_run
from .stage_x05_lpm import masked_line_windows, stage_x05_config_from_run

SPEC_VERSION = "E5_v1"
DEFAULT_R_MIN_PX = 4.0
DEFAULT_R_STEP_PX = 4.0
DEFAULT_N_ANGLES = 8
DEFAULT_THRESHOLD_SIGMA = 5.0
DEFAULT_CONTRASTS = tuple(float(c) for c in np.logspace(-5, -2, 9))


@dataclass(frozen=True)
class StageH05Product:
    rows: list
    curves: dict
    qc: dict


def stage_h05_paths(run_id, project_root=None):
    root = Path(project_root or Path.cwd()).resolve()
    paths = RunPaths.from_project_root(run_id, root)
    plot_dir = paths.plot_stage_dir("stage_h05")
    base = stage_h01b_paths(run_id, root)
    base.update(
        {
            "contrast_curve_csv": paths.table_dir / "contrast_curve_by_method.csv",
            "contrast_injections_csv": paths.table_dir / "contrast_injections.csv",
            "stage_h05_qc_json": paths.stage_dir / "stage_h05_qc.json",
            "plot_dir": plot_dir,
            "summary_plot": plot_dir / "stage_h05_contrast_curves.png",
        }
    )
    return base


def stage_h05_config_from_run(run_id=None, *, project_root=None, overrides=None, allow_run_id_mismatch=False):
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
    cfg.setdefault("h05_r_min_px", DEFAULT_R_MIN_PX)
    cfg.setdefault("h05_r_step_px", DEFAULT_R_STEP_PX)
    cfg.setdefault("h05_n_angles", DEFAULT_N_ANGLES)
    cfg.setdefault("h05_threshold_sigma", DEFAULT_THRESHOLD_SIGMA)
    cfg.setdefault("h05_contrasts", list(DEFAULT_CONTRASTS))
    cfg.setdefault("h05_include_psfsub", False)
    return cfg


def stellar_line_flux(pre_cube, wave, star_yx, *, line_center_A, lsf_fwhm_A, norm_radius_px=25.0):
    """F_star: mother-cube flux in NORMRAD x line window, x d_lambda (spec §2.1)."""

    ny, nx = pre_cube.shape[1:]
    yy, xx = np.indices((ny, nx), dtype=np.float64)
    inside = np.hypot(yy - float(star_yx[0]), xx - float(star_yx[1])) <= float(norm_radius_px)
    in_line = np.abs(wave - float(line_center_A)) <= float(lsf_fwhm_A)
    dl = float(np.nanmedian(np.diff(wave)))
    values = pre_cube[in_line][:, inside]
    flux = float(np.nansum(np.where(np.isfinite(values), values, 0.0)))
    return flux * dl


def injection_delta_spaxels(wave, line_center_A, lsf_fwhm_A, psf_model, position_yx, shape, *,
                            total_line_flux, support_halfsize_px=None):
    """Delta cube of one injection restricted to its spatial support.

    Returns ``(delta_sup, support_mask)`` with ``delta_sup`` shaped
    (nz, n_support). Uses the SAME normalization helpers as
    ``musepipe.injection.inject`` (unit-integral line profile over channel
    widths, unit-sum spatial PSF over the frame) so the delta path and a
    brute-force ``inject`` are identical by construction.

    ``support_halfsize_px`` optionally clips the support to a box around the
    position: the single-point correlation only reads the kernel window, and
    both subtractions are per-spaxel independent, so clipping to the kernel
    window is EXACT for the detection statistic (and keeps the per-injection
    cost bounded on wide Moffat wings).
    """

    from ..injection import gaussian_line_profile, normalized_spatial_psf

    wave = np.asarray(wave, dtype=np.float64)
    profile = gaussian_line_profile(wave, float(line_center_A), float(lsf_fwhm_A))
    profile = np.asarray(profile, dtype=np.float64) * float(total_line_flux)
    # `frame` EXPLICITO, que es lo que E5 ha usado siempre. El default del
    # inyector paso a `norm_radius` el 2026-08-17 para que en E4 el flujo
    # inyectado y el recuperado sean la misma cantidad; E5 tiene la misma
    # pregunta abierta —su limite de contraste se compara contra flujos medidos
    # en NORMRAD—, pero cambiarla aqui movería las curvas publicadas, y eso es
    # una decision aparte. Se fija para que no cambie en silencio.
    psf = normalized_spatial_psf(
        shape, float(position_yx[0]), float(position_yx[1]),
        wavelength_A=float(line_center_A), psf_model=psf_model,
        norm_convention="frame",
    )
    peak = float(np.nanmax(psf))
    support = psf > 1e-8 * peak if peak > 0 else np.zeros(shape, bool)
    if support_halfsize_px is not None:
        yy, xx = np.indices(shape, dtype=np.float64)
        half = float(support_halfsize_px)
        support &= (np.abs(yy - float(position_yx[0])) <= half) & (
            np.abs(xx - float(position_yx[1])) <= half
        )
    delta_sup = profile[:, None] * psf[support][None, :]
    return delta_sup, support


def ring_positions(star_yx, separation_px, n_angles, shape, *, angle_offset_deg=0.0):
    out = []
    ny, nx = shape
    for k in range(int(n_angles)):
        theta = np.deg2rad(angle_offset_deg + 360.0 * k / int(n_angles))
        y = float(star_yx[0]) + float(separation_px) * np.sin(theta)
        x = float(star_yx[1]) + float(separation_px) * np.cos(theta)
        if 1 <= y < ny - 1 and 1 <= x < nx - 1:
            out.append((y, x, float(np.rad2deg(theta) % 360.0)))
    return out


def _mu_sigma_at_radius(rings, r):
    for ring in rings:
        if ring["r_lo"] <= r < ring["r_hi"]:
            return ring["mu"], ring["sigma"]
    return None, None


class _MethodState:
    """Base residual map + ring stats + per-injection delta evaluator."""

    def __init__(self, method, cube, wave, cfg_m, *, star_yx, companion_yx, template, kernel, psf_model,
                 line_center_A, lsf_fwhm_A):
        self.method = method
        self.wave = wave
        self.cfg_m = cfg_m
        self.psf_model = psf_model
        self.line_center_A = line_center_A
        self.lsf_fwhm_A = lsf_fwhm_A
        self.template = template
        self.kernel = kernel
        self.shape = cube.shape[1:]
        m = spectral_map(cube, template)
        corr = crosscorr_map(m, kernel)
        exclude_r = 1.5 * max(lsf_fwhm_A, 4.0) if companion_yx is not None else 0.0
        self.zmap, self.rings = ring_noise_stats(
            corr, star_yx, exclude_yx=companion_yx, exclude_radius_px=exclude_r
        )
        self.corr = corr
        self.star_yx = star_yx

    def subtract_delta(self, delta_sup):
        raise NotImplementedError

    def corr_delta_at(self, position_yx, delta_sup, support):
        resid_delta = self.subtract_delta(delta_sup)
        # NaN channels (e.g. where the reference is undefined: notch) carry no
        # information; zero-fill exactly like spectral_map does for the base.
        resid_delta = np.where(np.isfinite(resid_delta), resid_delta, 0.0)
        m_delta = np.tensordot(self.template, resid_delta, axes=(0, 0))  # (n_support,)
        # Single-point correlation at the injected position.
        half = self.kernel.shape[0] // 2
        y0, x0 = int(round(position_yx[0])), int(round(position_yx[1]))
        ys, xs = np.where(support)
        dy = ys - y0
        dx = xs - x0
        inside = (np.abs(dy) <= half) & (np.abs(dx) <= half)
        return float(np.sum(self.kernel[dy[inside] + half, dx[inside] + half] * m_delta[inside]))

    def detect(self, position_yx, contrast, f_star, threshold_sigma):
        total_flux = float(contrast) * float(f_star)
        delta_sup, support = injection_delta_spaxels(
            self.wave, self.line_center_A, self.lsf_fwhm_A, self.psf_model, position_yx, self.shape,
            total_line_flux=total_flux,
            support_halfsize_px=self.kernel.shape[0] // 2,
        )
        corr_delta = self.corr_delta_at(position_yx, delta_sup, support)
        y0, x0 = int(round(position_yx[0])), int(round(position_yx[1]))
        corr_base = self.corr[y0, x0]
        r = float(np.hypot(y0 - self.star_yx[0], x0 - self.star_yx[1]))
        mu, sigma = _mu_sigma_at_radius(self.rings, r)
        if mu is None or not np.isfinite(corr_base):
            return None
        z = (float(corr_base) + corr_delta - mu) / sigma
        if not np.isfinite(z):
            return None
        return {"z": float(z), "detected": bool(z >= float(threshold_sigma))}


class _SgfState(_MethodState):
    def __init__(self, *args, s_hat=None, **kwargs):
        self.s_hat = s_hat
        super().__init__(*args, **kwargs)

    def subtract_delta(self, delta_sup):
        res = sgf_subtract(
            delta_sup[:, :, None], self.s_hat,
            window=int(self.cfg_m.get("sgf_window", 101)),
            degree=int(self.cfg_m.get("sgf_degree", 1)),
        )
        return res.residual_cube[:, :, 0]


class _LpmState(_MethodState):
    def __init__(self, *args, s_hat=None, **kwargs):
        self.s_hat = s_hat
        super().__init__(*args, **kwargs)

    def subtract_delta(self, delta_sup):
        res = lpm_subtract(
            delta_sup[:, :, None], self.wave, self.s_hat,
            degree=int(self.cfg_m.get("lpm_degree", 4)),
            line_windows_A=masked_line_windows(self.cfg_m),
        )
        return res.residual_cube[:, :, 0]


def build_method_states(paths, cfg, *, star_yx, companion_yx, line_center_A, lsf_fwhm_A, open_issues):
    psf_model = _read_optional_json(paths["psf_model_json"])
    if not psf_model:
        raise RuntimeError("E5 requires psf_model.json (C1) for the injection PSF and kernel.")
    pre_cube, pre_wave = _load_stage02_mean(paths)
    halfsize = int(cfg.get("h01b_kernel_halfsize_px", 7))

    states = {}
    specs = (
        ("sgf", "sgf_residual_cube", stage_x04_config_from_run, _SgfState),
        ("lpm", "lpm_residual_cube", stage_x05_config_from_run, _LpmState),
    )
    for method, key, cfg_fn, cls in specs:
        path = Path(paths[key])
        if not path.exists():
            open_issues.append(f"Residual cube for {method} not found; method skipped.")
            continue
        cube, wave = _load_residual(path)
        try:
            cfg_m = cfg_fn(cfg["run_id"], project_root=cfg.get("project_root"))
        except Exception:
            cfg_m = dict(cfg)
        if method == "lpm":
            # Same default the C6 stage freezes: never subtract the injection
            # delta with an unmasked line (it would absorb the line itself).
            from ..spectral import STANDARD_LINE_WINDOWS_A

            cfg_m.setdefault("lpm_masked_lines_A", [[c, h] for c, h in STANDARD_LINE_WINDOWS_A])
        wmask = wave_range_mask(pre_wave, cfg_m.get("halosub_wave_range_A"))
        keep, _qc = select_reference_spaxels(
            pre_cube,
            flux_lo_frac=float(cfg_m.get("halosub_flux_mask_lo", 0.01)),
            flux_hi_frac=float(cfg_m.get("halosub_flux_mask_hi", 0.1)),
            wave_mask=wmask,
            exclude_yx=[companion_yx] if companion_yx is not None else None,
            exclude_radius_px=float(cfg_m.get("halosub_exclude_radius_px", 3.0)),
        )
        s_hat = reference_spectrum(pre_cube, keep)
        template = spectral_template(wave, line_center_A, lsf_fwhm_A)
        kernel = psf_kernel_from_model(psf_model, line_center_A, halfsize)
        states[method] = cls(
            method, cube, wave, cfg_m,
            star_yx=star_yx, companion_yx=companion_yx, template=template, kernel=kernel,
            psf_model=psf_model, line_center_A=line_center_A, lsf_fwhm_A=lsf_fwhm_A,
            s_hat=s_hat,
        )
    if not states:
        raise RuntimeError("E5 found no residual cubes; run C5/C6 first.")
    return states, pre_cube, pre_wave, psf_model


def contrast_curve_from_rows(rows, contrasts, *, fractions=(0.25, 0.5, 0.75)):
    """Per (method, separation): minimum grid contrast reaching each detection fraction."""

    out = {}
    keyed = {}
    for row in rows:
        keyed.setdefault((row["method"], row["separation_px"], row["contrast"]), []).append(row["detected"])
    methods = sorted({row["method"] for row in rows})
    for method in methods:
        seps = sorted({row["separation_px"] for row in rows if row["method"] == method})
        curve = []
        for sep in seps:
            entry = {"separation_px": float(sep)}
            n_angles = 0
            for label, frac in zip(("contrast_25", "contrast_50", "contrast_75"), fractions):
                value = None
                for c in sorted(contrasts):
                    flags = keyed.get((method, sep, c), [])
                    if flags:
                        n_angles = max(n_angles, len(flags))
                    if flags and float(np.mean(flags)) >= frac:
                        value = float(c)
                        break
                entry[label] = value
            entry["n_angles"] = int(n_angles)
            curve.append(entry)
        out[method] = curve
    return out


def compute_stage_h05_products(config, paths=None):
    cfg = dict(config)
    root = Path(cfg.get("project_root") or Path.cwd()).resolve()
    paths = stage_h05_paths(cfg["run_id"], root) if paths is None else paths
    open_issues = []

    star_yx, companion_yx = _positions(paths, cfg)
    lsf_fwhm_A = _lsf_fwhm(paths, cfg)
    line_center = expected_line_center_A(float(cfg.get("h01b_line_rest_A", 6562.8)), _rv_sys_kms(cfg))
    threshold = float(cfg.get("h05_threshold_sigma", DEFAULT_THRESHOLD_SIGMA))
    contrasts = [float(c) for c in cfg.get("h05_contrasts", DEFAULT_CONTRASTS)]

    states, pre_cube, pre_wave, _psf_model = build_method_states(
        paths, cfg, star_yx=star_yx, companion_yx=companion_yx,
        line_center_A=line_center, lsf_fwhm_A=lsf_fwhm_A, open_issues=open_issues,
    )
    f_star = stellar_line_flux(
        pre_cube, pre_wave, star_yx,
        line_center_A=line_center, lsf_fwhm_A=lsf_fwhm_A,
        norm_radius_px=float(cfg.get("psf_norm_radius_px", 25.0)),
    )
    if not np.isfinite(f_star) or f_star <= 0:
        raise RuntimeError("E5: stellar line-band flux is not positive; check the mother cube.")

    shape = pre_cube.shape[1:]
    r_min = float(cfg.get("h05_r_min_px", DEFAULT_R_MIN_PX))
    r_step = float(cfg.get("h05_r_step_px", DEFAULT_R_STEP_PX))
    halfsize = int(cfg.get("h01b_kernel_halfsize_px", 7))
    r_max = min(shape) / 2.0 - halfsize - 1
    separations = list(np.arange(r_min, max(r_min + r_step, r_max), r_step))
    n_angles = int(cfg.get("h05_n_angles", DEFAULT_N_ANGLES))
    guard = 2.0 * max(lsf_fwhm_A, 4.0)

    rows = []
    for method, state in states.items():
        for sep in separations:
            for (y, x, angle) in ring_positions(star_yx, sep, n_angles, shape):
                if companion_yx is not None and np.hypot(y - companion_yx[0], x - companion_yx[1]) < guard:
                    continue
                for contrast in contrasts:
                    result = state.detect((y, x), contrast, f_star, threshold)
                    if result is None:
                        continue
                    rows.append(
                        {
                            "method": method,
                            "separation_px": float(sep),
                            "angle_deg": float(angle),
                            "y": float(y),
                            "x": float(x),
                            "contrast": float(contrast),
                            "injected_line_flux": float(contrast) * f_star,
                            "z": result["z"],
                            "detected": bool(result["detected"]),
                        }
                    )

    curves = contrast_curve_from_rows(rows, contrasts)

    checks = {}
    checks["v1_curves_written"] = bool(curves) and all(bool(c) for c in curves.values())
    trend_ok = []
    saturation = []
    for method, curve in curves.items():
        c50 = [e["contrast_50"] for e in curve if e["contrast_50"] is not None]
        if len(c50) >= 6:
            third = max(len(c50) // 3, 1)
            trend_ok.append(float(np.median(c50[-third:])) <= float(np.median(c50[:third])) * 1.5)
        saturated = sum(1 for e in curve if e["contrast_50"] is None or e["contrast_50"] >= max(contrasts))
        saturation.append(saturated / max(len(curve), 1))
    checks["v2_monotonic_trend"] = bool(all(trend_ok)) if trend_ok else None
    checks["v3_grid_saturation"] = {
        "max_saturated_fraction": float(max(saturation)) if saturation else None,
        "warn": bool(max(saturation) > 0.5) if saturation else None,
    }
    if checks["v3_grid_saturation"]["warn"]:
        open_issues.append("Contrast grid saturated for >50% of separations; resize h05_contrasts (spec revision).")

    qc = {
        "stage": "h05_contrast",
        "spec_version": SPEC_VERSION,
        "run_id": str(cfg["run_id"]),
        "params": {
            "separations_px": [float(s) for s in separations],
            "n_angles": n_angles,
            "contrasts": contrasts,
            "threshold_sigma": threshold,
            "f_star_line": float(f_star),
            "line_center_A": float(line_center),
            "lsf_fwhm_A": float(lsf_fwhm_A),
        },
        "methods": {
            method: {"n_injections": int(sum(1 for r in rows if r["method"] == method)), "curve": curve}
            for method, curve in curves.items()
        },
        "checks": checks,
        "open_issues": open_issues,
    }
    return StageH05Product(rows=rows, curves=curves, qc=qc)


def write_stage_h05_products(product: StageH05Product, config, paths):
    paths["paths"].ensure_base_dirs()
    paths["plot_dir"].mkdir(parents=True, exist_ok=True)
    with open(paths["contrast_injections_csv"], "w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["method", "separation_px", "angle_deg", "y", "x", "contrast",
                        "injected_line_flux", "z", "detected"],
        )
        writer.writeheader()
        writer.writerows(product.rows)
    with open(paths["contrast_curve_csv"], "w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["method", "separation_px", "contrast_25", "contrast_50", "contrast_75", "n_angles"],
        )
        writer.writeheader()
        for method, curve in product.curves.items():
            for entry in curve:
                writer.writerow({"method": method, **entry})
    _write_stage_h05_plot(product, paths)
    write_json(paths["stage_h05_qc_json"], product.qc)
    return {"qc_json": paths["stage_h05_qc_json"], "qc": product.qc}


def _write_stage_h05_plot(product: StageH05Product, paths):
    import os

    cache_dir = Path(os.environ.get("TMPDIR", "/tmp")) / "musepipe_matplotlib"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4.6))
    for method, curve in product.curves.items():
        seps = [e["separation_px"] for e in curve if e["contrast_50"] is not None]
        c50 = [e["contrast_50"] for e in curve if e["contrast_50"] is not None]
        if seps:
            ax.plot(seps, c50, marker="o", ms=3, label=f"{method} (50%)")
    ax.set_yscale("log")
    ax.set_xlabel("separation [px]")
    ax.set_ylabel("5-sigma line contrast")
    ax.legend(fontsize=8)
    ax.set_title("E5: contrast curves (Julo+25 Fig. 10)")
    fig.tight_layout()
    fig.savefig(paths["summary_plot"], dpi=120)
    plt.close(fig)


def run_stage_h05(run_id=None, *, project_root=None, overrides=None, allow_run_id_mismatch=False):
    cfg = stage_h05_config_from_run(
        run_id,
        project_root=project_root,
        overrides=overrides,
        allow_run_id_mismatch=allow_run_id_mismatch,
    )
    paths = stage_h05_paths(cfg["run_id"], project_root=cfg.get("project_root"))
    product = compute_stage_h05_products(cfg, paths)
    written = write_stage_h05_products(product, cfg, paths)
    return {"config": cfg, "paths": paths, "qc": written["qc"], "written": written}


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run Stage H05/E5 contrast curves.")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--project-root", default=None)
    parser.add_argument("--allow-run-id-mismatch", action="store_true")
    args = parser.parse_args(argv)
    result = run_stage_h05(
        args.run_id,
        project_root=args.project_root,
        allow_run_id_mismatch=args.allow_run_id_mismatch,
    )
    print(result["paths"]["stage_h05_qc_json"])


__all__ = [
    "SPEC_VERSION",
    "StageH05Product",
    "compute_stage_h05_products",
    "contrast_curve_from_rows",
    "injection_delta_spaxels",
    "ring_positions",
    "run_stage_h05",
    "stage_h05_config_from_run",
    "stage_h05_paths",
    "stellar_line_flux",
    "write_stage_h05_products",
]


if __name__ == "__main__":
    main()
