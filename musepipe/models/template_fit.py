"""Spectral-type by template fitting (spec G3 §3.2, plan WP-G3R-7).

For every template in the library, minimize chi2 over (A_V, scale) — scale
analytic (:func:`musepipe.models.fit._best_scale_chi2`), A_V on a grid. Optional
veiling (D10) adds a non-negative additive power law a·(λ/λ0)^α via NNLS. The
gravity class (young / field / non-stellar power law, D3) is decided by the
Δχ² between each class's best fit — the direct input to G4 tests T3/T4.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.optimize import nnls

from ..constants import spt_label
from .fit import _best_scale_chi2
from .prep import apply_extinction, prepare_template_base

_DEFAULT_VEILING_ALPHA = (-2.0, -1.0, 0.0, 1.0, 2.0)
_LAMBDA_REF_A = 7500.0


def declared_prep_kwargs(template, data_frame=None):
    """Resolución y marco DECLARADOS de una plantilla, para ``prepare_template``.

    Si la biblioteca trae declaración (:mod:`musepipe.models.libraries`) se
    usan su R (o FWHM) y su marco, llevados a ``data_frame``. Sin declaración
    (fixtures sintéticos) vale la regla histórica: ``resolution_fwhm_A`` de la
    meta, sin conversión de marco.
    """
    meta = template.meta
    if "declared_wave_frame" in meta:
        if data_frame is None:
            raise RuntimeError("plantilla con marco declarado y dato sin marco "
                               "(g3_data_wave_frame): no se supone ninguno")
        return {"template_R": meta.get("declared_resolution_R"),
                "template_fwhm_A": meta.get("declared_resolution_fwhm_A"),
                "template_frame": meta["declared_wave_frame"], "data_frame": data_frame}
    return {"template_fwhm_A": meta.get("resolution_fwhm_A")}


def _model(template, wave, extinction, av, lsf_fwhm_A, data_frame=None,
           target_fwhm_A=None):
    base, mismatch = prepare_template_base(
        template, wave, lsf_fwhm_A=lsf_fwhm_A, target_fwhm_A=target_fwhm_A,
        **declared_prep_kwargs(template, data_frame))
    return apply_extinction(base, wave, extinction, av), mismatch


def _dof_used(fit_spec, used):
    if hasattr(fit_spec, "dof_used"):
        return fit_spec.dof_used(used)
    n = max(1, int(fit_spec.n_bins))
    return float(fit_spec.n_bins) * float(np.sum(used)) / n


def _template_items(library, per_spectrum):
    """[(code, label, object, TemplateSpectrum)] — uno por subtipo (histórico,
    el primero) o uno por espectro (prueba por biblioteca: todas las estrellas)."""
    codes = np.asarray(library.grid()["spt"], float)
    if not per_spectrum or not hasattr(library, "entries"):
        out = []
        for code in codes:
            t = library.get(spt=float(code))
            out.append((float(code), spt_label(float(code)), t.meta.get("object"), t))
        return out
    return [(code, spt_label(code), meta.get("object", Path(path).stem),
             library.load_entry(code, path, meta))
            for code, path, meta in library.entries()]


def fit_templates(fit_spec, library, extinction, *, av_axis, lsf_fwhm_A,
                  veiling=False, veiling_alpha_axis=None, lambda_ref=_LAMBDA_REF_A,
                  chi2red_inflate_threshold=1.5, data_frame=None, target_fwhm_A=None,
                  per_spectrum=False, exclude=None, prepared=None):
    """Rank every template by chi2 over (A_V, scale[, veiling]); summarize SpT.

    The model is degraded/resampled ONCE per template and reddened per A_V
    (identical to preparing it at every A_V: extinction is applied after the
    resample). ``per_spectrum=True`` fits every spectrum of the library (not
    only the first of each subtype); the SpT curve is then the best spectrum
    per subtype. ``exclude`` (callable on the item) drops templates —
    leave-one-out of the Q1 calibration. ``prepared`` maps item index →
    precomputed base model (the calibration reuses them across draws).

    Degrees of freedom: ``fit_spec.dof_base − n_free`` (effective in the
    native fit, bins in the binned one). ``edge`` is True when the best SpT
    is the first or last subtype of the library axis.
    """
    wave = np.asarray(fit_spec.wave_bin, float)
    flux = np.asarray(fit_spec.flux_bin, float)
    err = np.asarray(fit_spec.err_bin, float)
    good = np.isfinite(flux) & np.isfinite(err) & (err > 0)
    inv_var = np.where(good, 1.0 / np.where(err > 0, err, 1.0) ** 2, 0.0)
    av_axis = np.asarray(av_axis, float)
    alpha_axis = np.asarray(veiling_alpha_axis if veiling_alpha_axis is not None
                            else _DEFAULT_VEILING_ALPHA, float)
    items = _template_items(library, per_spectrum)
    codes = np.asarray(sorted({it[0] for it in items}), float)
    n_free = 4 if veiling else 2
    ext_by_av = {}
    for av in av_axis:
        ext_by_av[float(av)] = apply_extinction(np.ones_like(wave), wave, extinction, av)

    ranking = []
    for i, (code, label, obj, tmpl) in enumerate(items):
        if exclude is not None and exclude(i, code, obj):
            continue
        if prepared is not None and i in prepared:
            base, mismatch = prepared[i]
        else:
            base, mismatch = prepare_template_base(
                tmpl, wave, lsf_fwhm_A=lsf_fwhm_A, target_fwhm_A=target_fwhm_A,
                **declared_prep_kwargs(tmpl, data_frame))
        best = {"chi2": np.inf, "av": np.nan, "scale": np.nan,
                "veiling_a": 0.0, "veiling_alpha": np.nan, "resolution_mismatch": False}
        # los χ² de todos los A_V (y de todas las plantillas) sobre los MISMOS
        # bins: los que cubre la plantilla y la ley de extinción
        common = good & np.isfinite(base)
        for f in ext_by_av.values():
            common &= np.isfinite(f)
        ndof = max(1.0, _dof_used(fit_spec, common) - n_free)
        for av in av_axis:
            model = np.where(common, base * ext_by_av[float(av)], np.nan)
            model = np.where(good, model, np.nan)
            if not veiling:
                scale, c2 = _best_scale_chi2(flux, model, inv_var)
                if c2 < best["chi2"]:
                    best = {"chi2": c2, "av": float(av), "scale": scale,
                            "veiling_a": 0.0, "veiling_alpha": np.nan,
                            "resolution_mismatch": bool(mismatch)}
            else:
                gm = good & np.isfinite(model)  # some template bins resample to NaN
                mm = np.where(gm, model, 0.0)
                wgt = np.where(gm, np.sqrt(inv_var), 0.0)
                fw = np.where(gm, flux, 0.0) * wgt
                for alpha in alpha_axis:
                    basis = np.where(gm, (wave / lambda_ref) ** alpha, 0.0)
                    coef, rnorm = nnls(np.column_stack([mm, basis]) * wgt[:, None], fw)
                    c2 = float(rnorm ** 2)
                    if c2 < best["chi2"]:
                        best = {"chi2": c2, "av": float(av), "scale": float(coef[0]),
                                "veiling_a": float(coef[1]), "veiling_alpha": float(alpha),
                                "resolution_mismatch": bool(mismatch)}
        ranking.append({"spt": label, "spt_code": float(code), "object": obj,
                        "chi2": float(best["chi2"]), "chi2_red": float(best["chi2"] / ndof),
                        "av_best": best["av"], "scale_best": best["scale"],
                        "veiling_a": best["veiling_a"], "veiling_alpha": best["veiling_alpha"],
                        "resolution_mismatch": best["resolution_mismatch"],
                        "ndof": float(ndof)})
    if not ranking:
        raise RuntimeError("fit_templates: no template left to fit")
    ranking.sort(key=lambda r: r["chi2"])
    chi2_min = ranking[0]["chi2"]
    by_code = {}
    for r in ranking:  # best spectrum per subtype
        by_code.setdefault(r["spt_code"], r)
    fitted_codes = np.asarray(sorted(by_code), float)
    within = [c for c in fitted_codes if by_code[c]["chi2"] - chi2_min <= 1.0]
    if len(within) == len(fitted_codes) and len(fitted_codes) > 1:
        interval = "not_constrained"
    else:
        interval = (float(min(within)), float(max(within))) if within else None
    chi2_red_min = ranking[0]["chi2_red"]
    inflate = chi2_red_min > float(chi2red_inflate_threshold)
    best_code = ranking[0]["spt_code"]
    edge = bool(fitted_codes.size > 1 and best_code in (fitted_codes[0], fitted_codes[-1]))
    return {
        "ranking": ranking,
        "spt_best": ranking[0]["spt"], "spt_best_code": best_code,
        "spt_best_object": ranking[0].get("object"),
        "spt_interval": interval, "av_best": ranking[0]["av_best"],
        "scale_best": ranking[0]["scale_best"], "chi2_min": chi2_min,
        "chi2_red_min": chi2_red_min, "n_bins": int(fit_spec.n_bins),
        "n_eff": float(fit_spec.n_eff), "ndof": float(ranking[0]["ndof"]),
        "chi2_by_spt": {spt_label(c): float(by_code[c]["chi2"]) for c in fitted_codes},
        "spt_axis": [float(fitted_codes[0]), float(fitted_codes[-1])],
        "edge": edge,
        "inflate": {"applied": bool(inflate),
                    "factor": float(np.sqrt(chi2_red_min)) if inflate else 1.0},
        "veiling": bool(veiling),
        "per_spectrum": bool(per_spectrum),
        "n_templates_fitted": len(ranking),
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
    ndof = max(1.0, _dof_used(fit_spec, good) - 2)
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


__all__ = ["classify_gravity", "declared_prep_kwargs", "fit_powerlaw", "fit_templates"]
