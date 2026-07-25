"""Derived physical quantities: L_bol, radius, mass/age via the two track
families, with the end-to-end MC (spec G3 §3.4, plan WP-G3R-9).

L_bol note: the cached BT-Settl spectra are trimmed to 4000–10000 Å and cannot
be integrated for a bolometric correction. For a model atmosphere the full-SED
integral is exact: ∫F_surface dλ ≡ σTeff⁴ (definition of Teff), so the scaled
model's bolometric luminosity is L_bol = 4πR²σTeff⁴ = 4πd²·Ω_phys·σTeff⁴ with
Ω_phys = (R/d)². That is what ``lbol_from_scaled_model`` computes — the model
provides the SED shape whose integral is σTeff⁴; nothing outside MUSE is guessed.
All outputs are ``atmospheric_model_dependent`` (L_bol, R) or
``evolutionary_model_dependent`` (mass, logg_evol).
"""

from __future__ import annotations

import numpy as np

from ..constants import LSUN_ERG_S, PC_CM, RJUP_CM, RSUN_CM, SIGMA_SB_CGS
from ..io import flux_unit_cgs

_PCTS = (16.0, 50.0, 84.0)


def radius_from_omega(omega, omega_err, distance_pc, distance_err, *,
                      flux_unit_cgs=1e-20) -> dict:
    """R from the fit scale Ω = (R/d)²·(1/flux_unit). Returns R in R☉ and R_Jup."""
    omega_phys = float(omega) * float(flux_unit_cgs)  # (R/d)^2, dimensionless
    if not np.isfinite(omega_phys) or omega_phys <= 0:
        return {"r_rsun": float("nan"), "r_rjup": float("nan"),
                "r_err_rsun": float("nan"), "r_err_rjup": float("nan"),
                "r_over_d": float("nan")}
    r_over_d = np.sqrt(omega_phys)
    r_cm = r_over_d * float(distance_pc) * PC_CM
    frac = np.hypot(0.5 * float(omega_err) / float(omega),
                    float(distance_err) / float(distance_pc))
    return {"r_rsun": r_cm / RSUN_CM, "r_rjup": r_cm / RJUP_CM,
            "r_err_rsun": r_cm / RSUN_CM * frac, "r_err_rjup": r_cm / RJUP_CM * frac,
            "r_over_d": float(r_over_d)}


def lbol_from_scaled_model(teff, omega, distance_pc, *, flux_unit_cgs=1e-20) -> dict:
    """L_bol = 4πd²·Ω_phys·σTeff⁴ (scaled model integrated over all λ)."""
    omega_phys = float(omega) * float(flux_unit_cgs)
    if not np.isfinite(omega_phys) or omega_phys <= 0 or teff <= 0:
        return {"l_bol_lsun": float("nan"), "f_bol_cgs": float("nan")}
    f_bol = omega_phys * SIGMA_SB_CGS * float(teff) ** 4          # erg/cm^2/s
    d_cm = float(distance_pc) * PC_CM
    l_erg = 4.0 * np.pi * d_cm ** 2 * f_bol
    return {"l_bol_lsun": l_erg / LSUN_ERG_S, "f_bol_cgs": float(f_bol)}


def _percentiles(arr):
    arr = np.asarray(arr, float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return [float("nan")] * 3
    return [float(x) for x in np.percentile(arr, _PCTS)]


def mass_age_from_tracks(track_grids, l_bol_samples, age_samples, teff_samples=None) -> dict:
    """Per-family mass/radius/logg from (L_bol, age) samples; between-family
    median-mass spread is the evolutionary err_sys (spec §3.4)."""
    l_bol_samples = np.asarray(l_bol_samples, float)
    age_samples = np.asarray(age_samples, float)
    per = {}
    cat_mass, cat_logg, cat_radius = [], [], []
    for family, grid in track_grids.items():
        s = grid.sample(l_bol_samples, age_samples)
        inr = np.asarray(s["in_range"], bool)
        mass_in = np.asarray(s["mass_msun"])[inr]
        logg_in = np.asarray(s["logg"])[inr]
        rad_in = np.asarray(s["radius_rsun"])[inr]
        per[family] = {
            "mass_msun": _percentiles(s["mass_msun"]),
            "radius_rsun": _percentiles(s["radius_rsun"]),
            "logg": _percentiles(s["logg"]),
            "in_range_frac": float(inr.mean()) if inr.size else 0.0,
            "mass_samples_in_range": mass_in,
        }
        cat_mass.append(mass_in[np.isfinite(mass_in)])
        cat_logg.append(logg_in[np.isfinite(logg_in)])
        cat_radius.append(rad_in[np.isfinite(rad_in)])
    medians = [per[f]["mass_msun"][1] for f in per if np.isfinite(per[f]["mass_msun"][1])]
    mass_err_sys = 0.5 * (max(medians) - min(medians)) if len(medians) > 1 else 0.0
    combined = np.concatenate(cat_mass) if cat_mass else np.array([])
    combined_logg = np.concatenate(cat_logg) if cat_logg else np.array([])
    combined_radius = np.concatenate(cat_radius) if cat_radius else np.array([])
    return {"per_family": per, "mass_err_sys": float(mass_err_sys),
            "combined_mass_samples": combined,
            "combined_mass_msun": _percentiles(combined),
            "combined_logg": _percentiles(combined_logg),
            "combined_radius_rsun": _percentiles(combined_radius)}


def _sample_age_gyr(rng, mean_gyr, err_lo_gyr, err_hi_gyr, n):
    """Two-piece (asymmetric) normal truncated > 0 (D12: 6 +4/-2 Myr)."""
    u = rng.standard_normal(n)
    age = np.where(u < 0.0, mean_gyr + u * err_lo_gyr, mean_gyr + u * err_hi_gyr)
    return np.clip(age, 1e-6, None)


def run_mc_chain(cfg, atmo_result, track_grids, rng, *, qc_m3=None) -> dict:
    """End-to-end MC (spec §3.4). Samples (Teff, A_V, Ω) from the Δχ² 3-D grid by
    weights exp(-Δχ²/2), plus distance / age / abs-cal / interpolation, then
    derives R, L_bol and per-family mass. Returns percentiles + mass posterior.

    ``qc_m3``: QC de A4 opcional, para que la escala de flujo pueda salir de la
    unidad que M3 uso si el config no declara el knob (ver ``io.flux_unit_cgs``).
    Ω sale del ajuste atmosferico sobre el espectro calibrado, asi que su unidad
    es la de ese producto, la misma que M3 fijo."""
    n = int(cfg.get("g3_n_mc", 4000))
    sysfrac = float(cfg.get("g3_sys_fluxcal_frac", 0.10))
    flux_unit = flux_unit_cgs(cfg, qc_m3=qc_m3)
    distance = float(cfg["h03_distance_pc"])
    distance_err = float(cfg.get("h03_distance_err_pc", 0.3))
    age_mean = float(cfg.get("g3_age_myr", 6.0)) / 1000.0  # Gyr
    age_err = cfg.get("g3_age_err_myr", [2.0, 4.0])
    age_lo, age_hi = float(age_err[0]) / 1000.0, float(age_err[1]) / 1000.0

    T = np.asarray(atmo_result["teff_axis"], float)
    G = np.asarray(atmo_result["logg_axis"], float)
    A = np.asarray(atmo_result["av_axis"], float)
    dchi2 = np.asarray(atmo_result["dchi2_3d"], float)
    scales = np.asarray(atmo_result["scales_3d"], float)
    interp = atmo_result.get("interp_error", {"teff": 0.0, "logg": 0.0, "av": 0.0})

    w = np.exp(-0.5 * dchi2).ravel()
    w[~np.isfinite(w)] = 0.0
    if w.sum() <= 0:
        raise RuntimeError("run_mc_chain: all Δχ² weights are zero/inf")
    w /= w.sum()
    idx = rng.choice(w.size, size=n, p=w)
    ti, gi, ai = np.unravel_index(idx, dchi2.shape)

    teff_s = T[ti] + rng.normal(0.0, interp["teff"], n)
    teff_s = np.clip(teff_s, 1.0, None)
    av_s = A[ai]
    omega_s = scales[ti, gi, ai] * (1.0 + rng.normal(0.0, sysfrac, n))  # abs-cal 10%
    omega_s = np.clip(omega_s, 1e-300, None)
    dist_s = np.clip(rng.normal(distance, distance_err, n), 1e-6, None)
    age_s = _sample_age_gyr(rng, age_mean, age_lo, age_hi, n)

    omega_phys = omega_s * flux_unit
    r_cm = np.sqrt(omega_phys) * dist_s * PC_CM
    r_rsun = r_cm / RSUN_CM
    r_rjup = r_cm / RJUP_CM
    l_bol = 4.0 * np.pi * (dist_s * PC_CM) ** 2 * omega_phys * SIGMA_SB_CGS * teff_s ** 4 / LSUN_ERG_S

    tracks = mass_age_from_tracks(track_grids, l_bol, age_s, teff_samples=teff_s)

    percentiles = {
        "teff_k": _percentiles(teff_s), "a_v": _percentiles(av_s),
        "omega": _percentiles(omega_s), "radius_rsun": _percentiles(r_rsun),
        "radius_rjup": _percentiles(r_rjup), "l_bol_lsun": _percentiles(l_bol),
        "age_gyr": _percentiles(age_s),
    }
    return {
        "n_mc": n, "seed": int(cfg.get("g3_seed", 0)),
        "percentiles": percentiles, "tracks": tracks,
        "l_bol_samples": l_bol, "age_samples": age_s, "teff_samples": teff_s,
        "mass_posterior_by_family": {f: tracks["per_family"][f]["mass_samples_in_range"]
                                     for f in tracks["per_family"]},
        "mass_posterior_combined": tracks["combined_mass_samples"],
    }


def plot_hrd(out_path, track_grids, mc, *, age_gyr, age_err_gyr):
    """V4 HRD: the source (L_bol, Teff with error ellipse) over both families'
    isochrones at the D12 age ± err (approximated by the sampled tracks)."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    fig = Figure(figsize=(5.5, 4.5))
    FigureCanvasAgg(fig)
    ax = fig.add_subplot(1, 1, 1)
    n_iso = 200
    for family, grid in track_grids.items():
        ages = np.full(n_iso, age_gyr)
        # sweep L_bol across the source posterior range to trace the isochrone
        lo, hi = np.nanpercentile(mc["l_bol_samples"], [2, 98])
        lgrid = np.linspace(max(lo, 1e-8), hi, n_iso)
        s = grid.sample(lgrid, ages)
        teff = np.asarray(s["teff_k"])
        good = np.asarray(s["in_range"]) & np.isfinite(teff)
        ax.plot(teff[good], lgrid[good], "-", lw=1.2, label=f"{family} @ {age_gyr*1e3:.0f} Myr")
    tp = mc["percentiles"]["teff_k"]
    lp = mc["percentiles"]["l_bol_lsun"]
    ax.errorbar([tp[1]], [lp[1]],
                xerr=[[tp[1] - tp[0]], [tp[2] - tp[1]]],
                yerr=[[lp[1] - lp[0]], [lp[2] - lp[1]]],
                fmt="*", ms=12, color="red", label="source")
    ax.set_xlabel("Teff [K]")
    ax.set_ylabel("L_bol [Lsun]")
    ax.set_yscale("log")
    ax.invert_xaxis()
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    return out_path


__all__ = ["lbol_from_scaled_model", "mass_age_from_tracks", "plot_hrd",
           "radius_from_omega", "run_mc_chain"]
