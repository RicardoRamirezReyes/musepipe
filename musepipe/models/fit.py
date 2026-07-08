"""Grid chi-square fitting with the Teff–A_V degeneracy map (spec G3 §3.3).

Scale Omega is analytic (linear least squares per template/A_V). Intervals are
read from marginalized Delta-chi2; a parameter whose Delta-chi2 stays < 1 over
its whole axis is reported ``not_constrained`` (spec §1.1), never a best point
without an interval.
"""

from __future__ import annotations

import numpy as np

from .prep import prepare_template


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

    def interval(axis, marg):
        within = axis[marg <= float(delta_chi2_confidence)]
        if within.size == 0:
            return None
        if within.size == axis.size and axis.size > 1:
            return "not_constrained"
        return (float(within.min()), float(within.max()))

    return {
        "teff_best": float(teff_axis[i0]),
        "av_best": float(av_axis[j0]),
        "scale_best": float(scales[i0, j0]),
        "chi2_min": float(chi2[i0, j0]),
        "chi2_red": float(chi2[i0, j0] / ndof),
        "dchi2_map": dchi2,
        "teff_axis": teff_axis, "av_axis": av_axis,
        "teff_interval": interval(teff_axis, dchi2.min(axis=1)),
        "av_interval": interval(av_axis, dchi2.min(axis=0)),
        "teff_av_degenerate": bool(np.count_nonzero(dchi2 <= 2.30) > 1),  # 2-param 1-sigma
    }


__all__ = ["fit_grid"]
