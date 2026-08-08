"""Generic emission/absorption line + continuum measurement (phase G2).

Reusable for any line in the MUSE range (Halpha is just one catalog row, spec
G2 §1.1). Provides local continuum, direct + profile-fit integrated flux,
equivalent width, centroid/FWHM (observed and LSF-deconvolved intrinsic),
asymmetry, radial velocity, detection status and upper limits, with Monte-Carlo
error propagation (optional G1 block covariance).

Conventions (spec G2 §1.3): wavelengths in air Angstrom; flux in the product's
BUNIT; **equivalent width is NEGATIVE for emission** (EW = integral of
(1 - F/Fc) dl); velocities in km/s. Constants from ``musepipe.constants``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .constants import C_KMS, fwhm_to_sigma, sigma_to_fwhm
from .stats import robust_sigma

try:  # SciPy is available in the MUSE env; degrade gracefully in its absence.
    from scipy.optimize import curve_fit
except Exception:  # pragma: no cover
    curve_fit = None


DETECT_Z = 5.0
MARGINAL_Z = 3.0

#: z gaussiano de una cola al 1%. Es el divisor que convierte el cuantil 99
#: EMPIRICO de los controles en un factor de inflacion: cuanto se equivoca el
#: sigma propagado respecto a lo que la medida realmente dispersa.
Z_GAUSS_99 = 2.3263478740408408


@dataclass
class LineMeasurement:
    name: str
    rest_A: float
    family: str = ""
    kind: str = ""
    status: str = "not_measurable"          # detected|marginal|upper_limit|not_measurable
    label: str = "not_constrained"           # direct_measurement|upper_limit|not_constrained
    line_window_A: tuple = (np.nan, np.nan)
    continuum_windows_A: tuple = field(default_factory=tuple)
    continuum_density: float = np.nan
    continuum_density_err: float = np.nan
    flux_direct: float = np.nan
    flux_direct_err: float = np.nan
    flux_fit: float = np.nan
    flux_fit_err: float = np.nan
    ew_A: float = np.nan
    ew_err_A: float = np.nan
    centroid_A: float = np.nan
    centroid_err_A: float = np.nan
    fwhm_obs_A: float = np.nan
    fwhm_intrinsic_A: float = np.nan
    fwhm_intrinsic_err_A: float = np.nan
    asymmetry: float = np.nan
    rv_kms: float = np.nan
    rv_err_kms: float = np.nan
    z_score: float = np.nan
    #: --- escala EMPIRICA: la misma medida repetida en los controles ---
    #: `z_score` esta en unidades del sigma PROPAGADO canal a canal, que en una
    #: fuente pegada al halo de una estrella brillante subestima lo que la medida
    #: dispersa de verdad. Estos campos la corrigen con el dato, no con un modelo.
    z_emp: float = np.nan               # z_score / sigma_inflation_applied: z en sigmas MEDIDOS
    sigma_inflation: float = np.nan     # MEDIDO: cuanto dispersan los controles vs el sigma propagado
    sigma_inflation_applied: float = np.nan  # max(medido, 1): el que entra en la etiqueta
    fap_empirical: float = np.nan       # fraccion de controles que iguala o supera al objeto
    n_controls: int = 0
    min_resolvable_fap: float = np.nan  # 1/(n+1): el suelo que dan n controles
    null_source: str = "none"           # de donde salio la distribucion nula
    flux_upper_limit_5sigma: float = np.nan
    throughput_applied: float = np.nan
    n_mc: int = 0
    seed: int = 0
    covariance_used: str = "none"
    flags: tuple = field(default_factory=tuple)
    reason: str = ""

    def to_row(self):
        d = dict(self.__dict__)
        d["line_window_A"] = ";".join(f"{v:.2f}" for v in self.line_window_A)
        d["continuum_windows_A"] = ";".join(f"{a:.1f}-{b:.1f}" for a, b in self.continuum_windows_A)
        d["flags"] = "|".join(self.flags)
        return d


def _channel_widths(wave):
    wave = np.asarray(wave, dtype=np.float64)
    edges = np.empty(wave.size + 1)
    edges[1:-1] = 0.5 * (wave[:-1] + wave[1:])
    edges[0] = wave[0] - 0.5 * (wave[1] - wave[0])
    edges[-1] = wave[-1] + 0.5 * (wave[-1] - wave[-2])
    return np.diff(edges)


def build_windows(wave, line_def, *, catalog=(), line_half_A=6.0, cont_gap_A=8.0,
                  cont_width_A=40.0, bad_ranges=(), bad_flags=None, neighbor_exclude_A=6.0):
    """Return (line_mask, blue_mask, red_mask) with neighbor/bad/flag exclusion."""
    wave = np.asarray(wave, dtype=np.float64)
    center = float(line_def["wave_A"])
    finite = np.isfinite(wave)
    excl = np.zeros(wave.shape, dtype=bool)
    for other in catalog:
        w = float(other["wave_A"])
        if abs(w - center) > 1e-6:
            excl |= np.abs(wave - w) <= float(neighbor_exclude_A)
    for lo, hi in bad_ranges or ():
        excl |= (wave >= float(lo)) & (wave <= float(hi))
    if bad_flags is not None:
        excl |= np.asarray(bad_flags, dtype=bool)

    line_mask = finite & (np.abs(wave - center) <= float(line_half_A))
    blue = finite & (~excl) & (wave >= center - cont_gap_A - cont_width_A) & (wave <= center - cont_gap_A)
    red = finite & (~excl) & (wave >= center + cont_gap_A) & (wave <= center + cont_gap_A + cont_width_A)
    return line_mask, blue, red


def _linear_continuum(wave, flux, blue, red, center):
    """Robust linear continuum from the two side windows; return (density@center, cont_array, slope)."""
    mask = (blue | red) & np.isfinite(flux)
    if int(np.count_nonzero(mask)) < 2:
        return np.nan, np.full_like(wave, np.nan), np.nan
    x = wave[mask] - center
    y = flux[mask]
    slope, intercept = np.polyfit(x, y, 1)
    cont = slope * (wave - center) + intercept
    return float(intercept), cont, float(slope)


def _gauss_lsf(wave, amp, center, sigma_int, lsf_sigma):
    sig = np.sqrt(float(sigma_int) ** 2 + float(lsf_sigma) ** 2)
    return amp * np.exp(-0.5 * ((np.asarray(wave) - center) / sig) ** 2)


def _direct_quantities(wave, resid, cont, dl, line_mask, center, rest_A, vsys):
    """Continuum-subtracted direct measurements (resid = flux - cont)."""
    w = wave[line_mask]
    r = resid[line_mask]
    c = cont[line_mask]
    dlam = dl[line_mask]
    good = np.isfinite(r) & np.isfinite(dlam)
    flux_direct = float(np.sum(r[good] * dlam[good]))
    # EW: integral of (1 - F/Fc) = -integral(resid/Fc); negative for emission
    with np.errstate(divide="ignore", invalid="ignore"):
        ew = float(np.sum(-(r[good] / c[good]) * dlam[good])) if np.all(np.isfinite(c[good])) and np.all(c[good] != 0) else np.nan
    wpos = np.clip(r, 0.0, None)
    tot = float(np.sum(wpos[good] * dlam[good]))
    if tot > 0:
        centroid = float(np.sum(w[good] * wpos[good] * dlam[good]) / tot)
        var = float(np.sum(((w[good] - centroid) ** 2) * wpos[good] * dlam[good]) / tot)
        fwhm_obs = sigma_to_fwhm(np.sqrt(var)) if var > 0 else np.nan
        m3 = float(np.sum(((w[good] - centroid) ** 3) * wpos[good] * dlam[good]) / tot)
        asym = m3 / (var ** 1.5) if var > 0 else np.nan
    else:
        centroid, fwhm_obs, asym = np.nan, np.nan, np.nan
    rv = C_KMS * (centroid - rest_A * (1.0 + vsys / C_KMS)) / rest_A if np.isfinite(centroid) else np.nan
    return flux_direct, ew, centroid, fwhm_obs, asym, rv


def measure_line(wave_A, flux, flux_err, line_def, *, lsf_fwhm_A, vsys_kms=0.0,
                 wl_cal_err_kms=0.0, throughput=1.0, cov=None, n_mc=1000, seed=0,
                 min_continuum_pixels=6, detect_z=DETECT_Z, marginal_z=MARGINAL_Z,
                 catalog=(), bad_ranges=(), bad_flags=None, windows=None,
                 null_z=None, null_source="controls"):
    """Measure one line; returns a LineMeasurement (spec G2 §3.2-§3.3)."""
    wave = np.asarray(wave_A, dtype=np.float64)
    flux = np.asarray(flux, dtype=np.float64)
    ferr = np.asarray(flux_err, dtype=np.float64)
    center0 = float(line_def["wave_A"])
    lsf_sigma = fwhm_to_sigma(float(lsf_fwhm_A))
    m = LineMeasurement(name=str(line_def.get("name", "line")), rest_A=center0,
                        family=str(line_def.get("family", "")), kind=str(line_def.get("kind", "")),
                        n_mc=int(n_mc), seed=int(seed), throughput_applied=float(throughput),
                        covariance_used="block_cov" if cov is not None else "none")
    flags = []

    finite_wave = wave[np.isfinite(wave)]
    line_half = 6.0
    if finite_wave.size == 0 or (center0 - line_half) < float(finite_wave.min()) or (center0 + line_half) > float(finite_wave.max()):
        m.status, m.reason = "not_measurable", "line_at_range_edge"
        return m

    if windows is not None:
        line_mask, blue, red = windows
    else:
        line_mask, blue, red = build_windows(wave, line_def, catalog=catalog,
                                              bad_ranges=bad_ranges, bad_flags=bad_flags)
    m.line_window_A = (float(center0 - 6.0), float(center0 + 6.0))
    m.continuum_windows_A = tuple((float(wave[msk].min()), float(wave[msk].max()))
                                  for msk in (blue, red) if np.any(msk))

    n_cont = int(np.count_nonzero(blue | red))
    if n_cont < int(min_continuum_pixels) or not np.any(line_mask):
        m.status, m.reason = "not_measurable", f"continuum_pixels={n_cont}"
        m.flags = tuple(flags)
        return m
    nan_frac = float(np.mean(~np.isfinite(flux[line_mask])))
    if nan_frac > 0.30:
        flags.append("nan_window")

    dl = _channel_widths(wave)
    c0, cont, slope = _linear_continuum(wave, flux, blue, red, center0)
    resid = flux - cont
    cont_noise = robust_sigma(resid[blue | red])
    m.continuum_density, m.continuum_density_err = c0, float(cont_noise)

    fd, ew, cen, fwhm_obs, asym, rv = _direct_quantities(wave, resid, cont, dl, line_mask, center0, center0, vsys_kms)
    m.flux_direct, m.ew_A, m.centroid_A, m.fwhm_obs_A, m.asymmetry, m.rv_kms = fd, ew, cen, fwhm_obs, asym, rv
    if np.isfinite(c0) and c0 <= 0:
        m.ew_A = np.nan
        flags.append("continuum_nonpositive")

    # integrated-flux error (analytic; MC below is the primary)
    dll = dl[line_mask]
    ferr_l = ferr[line_mask]
    m.flux_direct_err = float(np.sqrt(np.nansum((ferr_l * dll) ** 2)))

    # profile fit: gaussian (x) LSF, continuum-subtracted
    m.flux_fit, m.flux_fit_err, m.fwhm_intrinsic_A, m.fwhm_intrinsic_err_A, cen_fit, cen_fit_err = _fit_profile(
        wave[line_mask], resid[line_mask], ferr[line_mask], center0, lsf_sigma)
    if np.isnan(m.flux_fit):
        flags.append("fit_failed")

    # z-score via matched filter with LSF template on the residual
    m.z_score = _matched_z(wave, resid, ferr, center0, float(lsf_fwhm_A), (blue | red | line_mask))

    # MC error propagation (vectorized over direct/moment quantities)
    errs = _mc_errors(wave, flux, ferr, blue, red, line_mask, dl, center0, vsys_kms, n_mc, seed, cov)
    m.flux_direct_err = errs.get("flux_direct", m.flux_direct_err)
    m.ew_err_A = errs.get("ew", np.nan)
    m.centroid_err_A = errs.get("centroid", np.nan)
    m.rv_err_kms = float(np.hypot(errs.get("rv", np.nan), float(wl_cal_err_kms)))

    # --- escala empirica: los mismos umbrales, pero en sigmas MEDIDOS ---
    # Los 5/3 de la spec se conservan; lo que cambia es la unidad en que se
    # miden. Sin controles no hay escala: se etiqueta con el sigma propagado y
    # se DECLARA (`null_source`), porque un umbral sin procedencia es el mismo
    # fallback silencioso que persigue el resto de la cadena.
    escala = empirical_scale(null_z, m.z_score) if null_z is not None else None
    if escala is not None and np.isfinite(escala.get("sigma_inflation", np.nan)):
        m.sigma_inflation = escala["sigma_inflation"]
        m.n_controls = escala["n_controls"]
        m.min_resolvable_fap = escala["min_resolvable_fap"]
        m.fap_empirical = escala.get("fap_empirical", np.nan)
        m.null_source = str(null_source)
        # SUELO EN 1: los controles solo pueden DESTAPAR dispersion de mas, nunca
        # quitar la que el presupuesto ya contabiliza. Comparten con el objeto los
        # sistematicos globales (calibracion de flujo, throughput, PSF), asi que
        # esos se cancelan en la comparacion y la nula no los ve. Sin este suelo,
        # una linea cuyos controles salgan tranquilos por azar —33 muestras— se
        # divide por <1 y se fabrica significancia: paso de verdad con Ca II 8662,
        # que con z nominal 1.40 e inflacion 0.27 salia `detected`.
        m.sigma_inflation_applied = max(float(m.sigma_inflation), 1.0)
        m.z_emp = m.z_score / m.sigma_inflation_applied if np.isfinite(m.z_score) else np.nan
        if m.sigma_inflation < 1.0:
            flags.append("null_quieter_than_budget")
    else:
        m.null_source = "none"
        m.z_emp = m.z_score          # sin controles, la escala es la propagada
        m.sigma_inflation_applied = 1.0 if np.isfinite(m.z_score) else np.nan
        if null_z is not None:
            flags.append("empirical_scale_unavailable")

    # status + label + upper limit  (sobre z_emp: el z en sigmas medidos)
    z = m.z_emp
    if not np.isfinite(z):
        m.status, m.reason = "not_measurable", "z_undefined"
    elif z >= float(detect_z):
        m.status, m.label = "detected", "direct_measurement"
    elif z >= float(marginal_z):
        m.status, m.label = "marginal", "direct_measurement"
    else:
        m.status, m.label = "upper_limit", "upper_limit"
    # 5-sigma integrated upper limit with LSF template width, throughput-corrected.
    # El sigma se infla con el mismo factor medido: un limite calculado con el
    # sigma propagado seria optimista por ese factor.
    sigma_flux = m.flux_direct_err if np.isfinite(m.flux_direct_err) and m.flux_direct_err > 0 else float(cont_noise) * float(np.sqrt(np.nansum(dll ** 2)))
    if np.isfinite(m.sigma_inflation_applied) and m.sigma_inflation_applied > 0:
        sigma_flux = sigma_flux * float(m.sigma_inflation_applied)
    m.flux_upper_limit_5sigma = float(5.0 * sigma_flux / float(throughput)) if throughput else np.nan
    if np.isfinite(cen_fit):
        m.centroid_A = cen_fit if np.isfinite(cen_fit) else m.centroid_A
    m.flags = tuple(flags)
    return m


def _fit_profile(wave, resid, ferr, center0, lsf_sigma):
    if curve_fit is None or wave.size < 4:
        return (np.nan,) * 6
    good = np.isfinite(resid) & np.isfinite(ferr) & (ferr > 0)
    if int(np.count_nonzero(good)) < 4:
        return (np.nan,) * 6
    w, r, e = wave[good], resid[good], ferr[good]
    amp0 = float(r[np.argmax(np.abs(r))])
    p0 = [amp0, center0, max(0.5, lsf_sigma)]

    def model(x, amp, cen, sig_int):
        return _gauss_lsf(x, amp, cen, abs(sig_int), lsf_sigma)

    try:
        popt, pcov = curve_fit(model, w, r, p0=p0, sigma=e, absolute_sigma=True, maxfev=4000)
    except Exception:
        return (np.nan,) * 6
    amp, cen, sig_int = popt[0], popt[1], abs(popt[2])
    perr = np.sqrt(np.abs(np.diag(pcov)))
    sig_tot = np.sqrt(sig_int ** 2 + lsf_sigma ** 2)
    flux_fit = float(amp * sig_tot * np.sqrt(2.0 * np.pi))
    flux_fit_err = float(abs(flux_fit) * (perr[0] / abs(amp))) if amp != 0 else np.nan
    fwhm_int = sigma_to_fwhm(sig_int) if sig_int > 0.3 else 0.0  # ~0 => unresolved
    fwhm_int_err = sigma_to_fwhm(float(perr[2]))
    return flux_fit, flux_fit_err, float(fwhm_int), fwhm_int_err, float(cen), float(perr[1])


def _matched_z(wave, resid, ferr, center_A, fwhm_A, good_mask):
    from .stages.stage_h01_detect import matched_filter_point
    good = np.asarray(good_mask, dtype=bool) & np.isfinite(resid) & np.isfinite(ferr) & (ferr > 0)
    _flux, _sigma, z = matched_filter_point(wave, resid, ferr, center_A, fwhm_A, good)
    return z


def null_z_from_controls(wave_A, control_fluxes, flux_err, *, center_A, lsf_fwhm_A, windows):
    """`z` del MISMO estimador en cada control, donde se sabe que no hay fuente.

    Es el paso que convierte un error propagado en una escala medida. Cada
    control lleva **exactamente** el mismo camino que el objeto —las mismas
    ventanas, el mismo continuo lineal, el mismo filtro adaptado— y con el
    **mismo** `flux_err`, para que los `z` salgan en las mismas unidades y su
    dispersion se lea directamente como el factor de correccion (igual que hace
    E1 en `matched_filter_scan`).

    Devuelve `np.ndarray` de `z`, uno por control (NaN donde no se pudo medir).
    """
    wave = np.asarray(wave_A, dtype=np.float64)
    controls = np.atleast_2d(np.asarray(control_fluxes, dtype=np.float64))
    ferr = np.asarray(flux_err, dtype=np.float64)
    line_mask, blue, red = windows
    out = np.full(controls.shape[0], np.nan, dtype=np.float64)
    for i, ctrl in enumerate(controls):
        if not np.any(np.isfinite(ctrl)):
            continue
        _c0, cont, _slope = _linear_continuum(wave, ctrl, blue, red, float(center_A))
        out[i] = _matched_z(wave, ctrl - cont, ferr, float(center_A), float(lsf_fwhm_A),
                            blue | red | line_mask)
    return out


def empirical_scale(null_z, object_z=None, *, fap=0.01):
    """Escala empirica de una linea a partir de sus controles.

    `sigma_inflation` es el cuantil (1-`fap`) de los controles dividido por el z
    gaussiano equivalente: 1.0 significa que el sigma propagado ya describe la
    dispersion real, y 18 significa que se queda 18 veces corto.

    El cuantil se toma con `empirical_upper_quantile` de E3 —el mismo estimador
    que usa el limite superior— para que las dos etapas no midan lo mismo de dos
    maneras distintas.
    """
    from .stages.stage_h03_limits import empirical_upper_quantile

    vals = np.asarray(null_z, dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return None
    z_q = float(empirical_upper_quantile(vals, fap=float(fap)))
    inflation = z_q / Z_GAUSS_99 if np.isfinite(z_q) and z_q > 0 else np.nan
    out = {
        "n_controls": int(vals.size),
        "z_quantile": z_q,
        "fap_used": float(fap),
        "sigma_inflation": float(inflation),
        "min_resolvable_fap": float(1.0 / (vals.size + 1)),
        "null_median": float(np.median(vals)),
        "null_max": float(np.max(vals)),
    }
    if object_z is not None and np.isfinite(object_z):
        out["fap_empirical"] = float(np.mean(vals >= float(object_z)))
    return out


def _mc_errors(wave, flux, ferr, blue, red, line_mask, dl, center0, vsys, n_mc, seed, cov):
    if int(n_mc) <= 0:
        return {}
    rng = np.random.default_rng(int(seed))
    n = wave.size
    fd = np.empty(n_mc); ew = np.empty(n_mc); cen = np.empty(n_mc); rv = np.empty(n_mc)
    safe_err = np.where(np.isfinite(ferr) & (ferr > 0), ferr, 0.0)
    for i in range(int(n_mc)):
        pert = flux + rng.normal(0.0, 1.0, size=n) * safe_err
        c0, cont, _ = _linear_continuum(wave, pert, blue, red, center0)
        resid = pert - cont
        f, e, cc, _fw, _as, r = _direct_quantities(wave, resid, cont, dl, line_mask, center0, center0, vsys)
        fd[i], ew[i], cen[i], rv[i] = f, e, cc, r
    pct = lambda a: float(0.5 * (np.nanpercentile(a, 84) - np.nanpercentile(a, 16)))
    return {"flux_direct": pct(fd), "ew": pct(ew), "centroid": pct(cen), "rv": pct(rv)}


def measure_catalog(wave_A, flux, flux_err, catalog, *, lsf_fwhm_A,
                    control_fluxes=None, null_source="controls", **kw):
    """Measure every line in ``catalog`` on one spectrum; returns list[LineMeasurement].

    Con `control_fluxes` (N espectros de control, misma malla) cada linea saca
    ademas su distribucion nula y se etiqueta contra ella. Las **ventanas se
    construyen una sola vez por linea** y se pasan a objeto y controles: si cada
    uno eligiera las suyas, la comparacion dejaria de ser la misma medida.
    """
    wave = np.asarray(wave_A, dtype=np.float64)
    controls = None if control_fluxes is None else np.atleast_2d(
        np.asarray(control_fluxes, dtype=np.float64))
    if controls is not None and controls.shape[1] != wave.size:
        raise ValueError(
            f"control_fluxes tiene {controls.shape[1]} canales y el objeto {wave.size}: "
            "no son la misma malla, compararlos mediria canales distintos.")
    out = []
    for line_def in catalog:
        windows = kw.pop("windows", None) or build_windows(
            wave, line_def, catalog=catalog,
            bad_ranges=kw.get("bad_ranges", ()), bad_flags=kw.get("bad_flags"))
        null_z = None
        if controls is not None:
            null_z = null_z_from_controls(
                wave, controls, flux_err, center_A=float(line_def["wave_A"]),
                lsf_fwhm_A=float(lsf_fwhm_A), windows=windows)
        out.append(measure_line(wave, flux, flux_err, line_def, lsf_fwhm_A=float(lsf_fwhm_A),
                                catalog=catalog, windows=windows, null_z=null_z,
                                null_source=null_source, **kw))
    return out


__all__ = ["DETECT_Z", "MARGINAL_Z", "Z_GAUSS_99", "LineMeasurement", "build_windows",
           "empirical_scale", "measure_catalog", "measure_line", "null_z_from_controls"]
