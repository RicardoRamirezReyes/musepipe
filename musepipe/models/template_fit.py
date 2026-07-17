"""Spectral-type by template fitting (spec G3 §3.2, plan WP-G3R-7).

For every template in the library, minimize chi2 over (A_V, scale) — scale
analytic (:func:`musepipe.models.fit._best_scale_chi2`), A_V on a grid. Optional
veiling (D10) adds a non-negative additive power law a·(λ/λ0)^α via NNLS. The
gravity class (young / field / non-stellar power law, D3) is decided by the
Δχ² between each class's best fit — the direct input to G4 tests T3/T4.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import nnls

from ..constants import spt_label
from .fit import _best_scale_chi2
from .prep import prepare_template

_DEFAULT_VEILING_ALPHA = (-2.0, -1.0, 0.0, 1.0, 2.0)
_LAMBDA_REF_A = 7500.0


def _model(template, wave, extinction, av, lsf_fwhm_A):
    return prepare_template(
        template, wave, lsf_fwhm_A=lsf_fwhm_A, extinction=extinction, av=av,
        scale=1.0, template_fwhm_A=template.meta.get("resolution_fwhm_A"),
        return_flag=True)


def fit_templates(fit_spec, library, extinction, *, av_axis, lsf_fwhm_A,
                  veiling=False, veiling_alpha_axis=None, lambda_ref=_LAMBDA_REF_A,
                  chi2red_inflate_threshold=1.5):
    """Rank every template by chi2 over (A_V, scale[, veiling]); summarize SpT."""
    wave = np.asarray(fit_spec.wave_bin, float)
    flux = np.asarray(fit_spec.flux_bin, float)
    err = np.asarray(fit_spec.err_bin, float)
    good = np.isfinite(flux) & np.isfinite(err) & (err > 0)
    inv_var = np.where(good, 1.0 / np.where(err > 0, err, 1.0) ** 2, 0.0)
    av_axis = np.asarray(av_axis, float)
    alpha_axis = np.asarray(veiling_alpha_axis if veiling_alpha_axis is not None
                            else _DEFAULT_VEILING_ALPHA, float)
    codes = np.asarray(library.grid()["spt"], float)
    n_free = 4 if veiling else 2
    ndof = max(1, int(fit_spec.n_bins) - n_free)

    ranking = []
    for code in codes:
        tmpl = library.get(spt=float(code))
        best = {"chi2": np.inf, "av": np.nan, "scale": np.nan,
                "veiling_a": 0.0, "veiling_alpha": np.nan, "resolution_mismatch": False}
        for av in av_axis:
            model, mismatch = _model(tmpl, wave, extinction, av, lsf_fwhm_A)
            model = np.where(good, model, np.nan)
            if not veiling:
                scale, c2 = _best_scale_chi2(flux, model, inv_var)
                if c2 < best["chi2"]:
                    best = {"chi2": c2, "av": float(av), "scale": scale,
                            "veiling_a": 0.0, "veiling_alpha": np.nan,
                            "resolution_mismatch": bool(mismatch)}
            else:
                for alpha in alpha_axis:
                    basis = (wave / lambda_ref) ** alpha
                    a_cols = np.column_stack([np.where(good, model, 0.0),
                                              np.where(good, basis, 0.0)])
                    w = np.sqrt(inv_var)
                    coef, rnorm = nnls(a_cols * w[:, None], flux * w)
                    c2 = float(rnorm ** 2)
                    if c2 < best["chi2"]:
                        best = {"chi2": c2, "av": float(av), "scale": float(coef[0]),
                                "veiling_a": float(coef[1]), "veiling_alpha": float(alpha),
                                "resolution_mismatch": bool(mismatch)}
        ranking.append({"spt": spt_label(float(code)), "spt_code": float(code),
                        "chi2": float(best["chi2"]), "chi2_red": float(best["chi2"] / ndof),
                        "av_best": best["av"], "scale_best": best["scale"],
                        "veiling_a": best["veiling_a"], "veiling_alpha": best["veiling_alpha"],
                        "resolution_mismatch": best["resolution_mismatch"]})
    ranking.sort(key=lambda r: r["chi2"])
    chi2_min = ranking[0]["chi2"]
    within = [r["spt_code"] for r in ranking if r["chi2"] - chi2_min <= 1.0]
    if len(within) == len(codes) and len(codes) > 1:
        interval = "not_constrained"
    else:
        interval = (float(min(within)), float(max(within))) if within else None
    chi2_red_min = ranking[0]["chi2_red"]
    inflate = chi2_red_min > float(chi2red_inflate_threshold)
    return {
        "ranking": ranking,
        "spt_best": ranking[0]["spt"], "spt_best_code": ranking[0]["spt_code"],
        "spt_interval": interval, "av_best": ranking[0]["av_best"],
        "scale_best": ranking[0]["scale_best"], "chi2_min": chi2_min,
        "chi2_red_min": chi2_red_min, "n_bins": int(fit_spec.n_bins),
        "n_eff": float(fit_spec.n_eff),
        "inflate": {"applied": bool(inflate),
                    "factor": float(np.sqrt(chi2_red_min)) if inflate else 1.0},
        "veiling": bool(veiling),
    }


def fit_powerlaw(fit_spec, *, alpha_axis, lambda_ref=_LAMBDA_REF_A):
    """Best F_λ ∝ (λ/λ0)^α fit (D3 non-stellar proxy). scale analytic per α."""
    wave = np.asarray(fit_spec.wave_bin, float)
    flux = np.asarray(fit_spec.flux_bin, float)
    err = np.asarray(fit_spec.err_bin, float)
    good = np.isfinite(flux) & np.isfinite(err) & (err > 0)
    inv_var = np.where(good, 1.0 / np.where(err > 0, err, 1.0) ** 2, 0.0)
    best = {"chi2": np.inf, "alpha": np.nan, "scale": np.nan}
    for alpha in np.asarray(alpha_axis, float):
        model = np.where(good, (wave / lambda_ref) ** alpha, np.nan)
        scale, c2 = _best_scale_chi2(flux, model, inv_var)
        if c2 < best["chi2"]:
            best = {"chi2": float(c2), "alpha": float(alpha), "scale": scale}
    ndof = max(1, int(fit_spec.n_bins) - 2)
    return {"chi2_min": best["chi2"], "alpha_best": best["alpha"],
            "scale_best": best["scale"], "chi2_red": best["chi2"] / ndof}


def classify_gravity(results_by_class) -> dict:
    """Δχ² between the best fit of each gravity class (young / field / nonstellar)."""
    chi2 = {cls: float(r["chi2_min"]) for cls, r in results_by_class.items()}
    best_class = min(chi2, key=chi2.get)
    best = chi2[best_class]
    out = dict(chi2)
    out["best_class"] = best_class
    out["dchi2_by_class"] = {cls: chi2[cls] - best for cls in chi2}
    return out


__all__ = ["classify_gravity", "fit_powerlaw", "fit_templates"]
