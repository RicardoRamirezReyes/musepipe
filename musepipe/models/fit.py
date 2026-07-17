"""Grid chi-square fitting with the Teff–A_V degeneracy map (spec G3 §3.3).

Scale Omega is analytic (linear least squares per template/A_V). Intervals are
read from marginalized Delta-chi2; a parameter whose Delta-chi2 stays < 1 over
its whole axis is reported ``not_constrained`` (spec §1.1), never a best point
without an interval.
"""

from __future__ import annotations

import numpy as np

from .prep import prepare_template


def _marginalized_interval(axis, marg, confidence=1.0):
    """1-sigma interval from a marginalized Delta-chi2 profile (spec §1.1).

    Returns ``(lo, hi)``; ``"not_constrained"`` when the whole axis stays within
    the confidence level; ``None`` when nothing does. Shared by fit_grid and
    fit_grid_3d (identical rule, so fit_grid's behaviour is unchanged)."""
    within = axis[marg <= float(confidence)]
    if within.size == 0:
        return None
    if within.size == axis.size and axis.size > 1:
        return "not_constrained"
    return (float(within.min()), float(within.max()))


def _local_halfstep(axis, value):
    """Half the local grid spacing around ``value`` (grid interp error, §4.3)."""
    axis = np.asarray(axis, float)
    if axis.size < 2:
        return 0.0
    i = int(np.argmin(np.abs(axis - value)))
    diffs = []
    if i > 0:
        diffs.append(axis[i] - axis[i - 1])
    if i < axis.size - 1:
        diffs.append(axis[i + 1] - axis[i])
    return 0.5 * float(min(diffs)) if diffs else 0.0


def _best_scale_chi2(data, model, inv_var):
    """Analytic positive scale that minimizes chi2 for data ~ scale*model."""
    good = np.isfinite(data) & np.isfinite(model) & np.isfinite(inv_var)
    if int(np.count_nonzero(good)) < 3:
        return np.nan, np.inf
    m, d, w = model[good], data[good], inv_var[good]
    denom = float(np.sum(w * m * m))
    if denom <= 0:
        return np.nan, np.inf
    scale = float(np.sum(w * d * m) / denom)
    resid = d - scale * m
    return scale, float(np.sum(w * resid * resid))


def fit_grid(wave, flux, flux_err, library, extinction, *, teff_axis, av_axis,
             lsf_fwhm_A, line_mask=None, logg=4.0, delta_chi2_confidence=1.0):
    """Return best (Teff, A_V, scale), the Delta-chi2(Teff,A_V) map and intervals."""
    wave = np.asarray(wave, float)
    flux = np.asarray(flux, float)
    err = np.asarray(flux_err, float)
    fit_mask = np.isfinite(flux) & np.isfinite(err) & (err > 0)
    if line_mask is not None:
        fit_mask &= ~np.asarray(line_mask, bool)  # mask emission lines (spec §3.2)
    inv_var = np.where(fit_mask, 1.0 / np.where(err > 0, err, 1.0) ** 2, 0.0)

    teff_axis = np.asarray(teff_axis, float)
    av_axis = np.asarray(av_axis, float)
    chi2 = np.full((teff_axis.size, av_axis.size), np.inf)
    scales = np.zeros_like(chi2)
    for i, teff in enumerate(teff_axis):
        template = library.get(teff=teff, logg=logg)
        for j, av in enumerate(av_axis):
            model = prepare_template(template, wave, lsf_fwhm_A=lsf_fwhm_A, extinction=extinction, av=av, scale=1.0)
            model = np.where(fit_mask, model, np.nan)
            scale, c2 = _best_scale_chi2(flux, model, inv_var)
            chi2[i, j] = c2
            scales[i, j] = scale
    i0, j0 = np.unravel_index(np.argmin(chi2), chi2.shape)
    dchi2 = chi2 - chi2[i0, j0]
    ndof = max(1, int(np.count_nonzero(fit_mask)) - 3)

    return {
        "teff_best": float(teff_axis[i0]),
        "av_best": float(av_axis[j0]),
        "scale_best": float(scales[i0, j0]),
        "chi2_min": float(chi2[i0, j0]),
        "chi2_red": float(chi2[i0, j0] / ndof),
        "dchi2_map": dchi2,
        "teff_axis": teff_axis, "av_axis": av_axis,
        "teff_interval": _marginalized_interval(teff_axis, dchi2.min(axis=1), delta_chi2_confidence),
        "av_interval": _marginalized_interval(av_axis, dchi2.min(axis=0), delta_chi2_confidence),
        "teff_av_degenerate": bool(np.count_nonzero(dchi2 <= 2.30) > 1),  # 2-param 1-sigma
    }


def _node_chi2(flux, model, inv_var, wave, *, veiling, alpha_axis, lambda_ref):
    """chi2 for one grid node: analytic scale, or NNLS (scale, veiling a>=0)."""
    if not veiling:
        scale, c2 = _best_scale_chi2(flux, model, inv_var)
        return scale, c2, 0.0, np.nan
    from scipy.optimize import nnls
    gm = np.isfinite(model) & np.isfinite(flux) & (inv_var > 0)
    w = np.where(gm, np.sqrt(inv_var), 0.0)
    m = np.where(gm, model, 0.0)
    fw = np.where(gm, flux, 0.0) * w
    best = (np.nan, np.inf, 0.0, np.nan)
    for alpha in alpha_axis:
        basis = np.where(gm, (wave / lambda_ref) ** alpha, 0.0)
        coef, rnorm = nnls(np.column_stack([m, basis]) * w[:, None], fw)
        c2 = float(rnorm ** 2)
        if c2 < best[1]:
            best = (float(coef[0]), c2, float(coef[1]), float(alpha))
    return best


def fit_grid_3d(fit_spec, library, extinction, *, teff_axis, logg_axis, av_axis,
                lsf_fwhm_A, delta_chi2_confidence=1.0, edge_sigma3=9.0,
                sys_fluxcal_frac=0.10, chi2red_inflate_threshold=1.5, n_free=4,
                veiling=False, veiling_alpha_axis=(-2.0, -1.0, 0.0, 1.0, 2.0),
                lambda_ref=7500.0):
    """3-D grid fit (Teff, logg, A_V) with analytic Omega and the V2 maps.

    2-D counterpart :func:`fit_grid` is kept for its existing callers/tests.
    Missing library nodes (``library.get`` raising) get chi2 = inf and drop out.
    """
    wave = np.asarray(fit_spec.wave_bin, float)
    flux = np.asarray(fit_spec.flux_bin, float)
    err = np.asarray(fit_spec.err_bin, float)
    good = np.isfinite(flux) & np.isfinite(err) & (err > 0)
    inv_var = np.where(good, 1.0 / np.where(err > 0, err, 1.0) ** 2, 0.0)
    T = np.asarray(teff_axis, float)
    G = np.asarray(logg_axis, float)
    A = np.asarray(av_axis, float)
    alpha_axis = np.asarray(veiling_alpha_axis, float)

    chi2 = np.full((T.size, G.size, A.size), np.inf)
    scales = np.zeros_like(chi2)
    for i, teff in enumerate(T):
        for j, logg in enumerate(G):
            try:
                tmpl = library.get(teff=float(teff), logg=float(logg))
            except RuntimeError:
                continue  # missing node -> stays inf
            for k, av in enumerate(A):
                model = prepare_template(tmpl, wave, lsf_fwhm_A=lsf_fwhm_A,
                                         extinction=extinction, av=float(av), scale=1.0)
                model = np.where(good, model, np.nan)
                scale, c2, _, _ = _node_chi2(flux, model, inv_var, wave,
                                             veiling=veiling, alpha_axis=alpha_axis,
                                             lambda_ref=lambda_ref)
                chi2[i, j, k] = c2
                scales[i, j, k] = scale

    idx = np.unravel_index(np.argmin(chi2), chi2.shape)
    i0, j0, k0 = (int(x) for x in idx)
    chi2_min = float(chi2[idx])
    dchi2 = chi2 - chi2_min
    marg_teff = dchi2.min(axis=(1, 2))
    marg_logg = dchi2.min(axis=(0, 2))
    marg_av = dchi2.min(axis=(0, 1))

    tmpl_best = library.get(teff=float(T[i0]), logg=float(G[j0]))
    model_best = prepare_template(tmpl_best, wave, lsf_fwhm_A=lsf_fwhm_A,
                                  extinction=extinction, av=float(A[k0]), scale=1.0)
    model_best = np.where(good, model_best, np.nan)
    swm2 = float(np.nansum(inv_var * model_best ** 2))
    omega_best = float(scales[idx])
    omega_err_stat = float(1.0 / np.sqrt(swm2)) if swm2 > 0 else float("nan")
    omega_err_sys = float(sys_fluxcal_frac) * abs(omega_best)
    omega_err_total = float(np.hypot(omega_err_stat, omega_err_sys))

    ndof = max(1, int(fit_spec.n_bins) - int(n_free))
    chi2_red = chi2_min / ndof
    inflate = chi2_red > float(chi2red_inflate_threshold)

    def edge(marg):
        return bool(marg[0] <= edge_sigma3 or marg[-1] <= edge_sigma3)

    return {
        "teff_best": float(T[i0]), "logg_best": float(G[j0]), "av_best": float(A[k0]),
        "omega_best": omega_best, "omega_err_stat": omega_err_stat,
        "omega_err_sys": omega_err_sys, "omega_err_total": omega_err_total,
        "chi2_min": chi2_min, "chi2_red": chi2_red, "ndof": ndof,
        "n_bins": int(fit_spec.n_bins), "n_eff": float(fit_spec.n_eff),
        "teff_axis": T, "logg_axis": G, "av_axis": A,
        "dchi2_3d": dchi2, "scales_3d": scales,
        "dchi2_teff_av": dchi2.min(axis=1),
        "dchi2_teff_logg": dchi2.min(axis=2),
        "teff_interval": _marginalized_interval(T, marg_teff, delta_chi2_confidence),
        "logg_interval": _marginalized_interval(G, marg_logg, delta_chi2_confidence),
        "av_interval": _marginalized_interval(A, marg_av, delta_chi2_confidence),
        "edge_touch": {"teff": edge(marg_teff), "logg": edge(marg_logg),
                       "av": edge(marg_av)},
        "interp_error": {"teff": _local_halfstep(T, T[i0]),
                         "logg": _local_halfstep(G, G[j0]),
                         "av": _local_halfstep(A, A[k0])},
        "inflate": {"applied": bool(inflate),
                    "factor": float(np.sqrt(chi2_red)) if inflate else 1.0},
        "veiling": bool(veiling),
    }


__all__ = ["fit_grid", "fit_grid_3d"]
