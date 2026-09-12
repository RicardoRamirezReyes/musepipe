"""Spectral-axis and wavelength-mask helpers."""

from __future__ import annotations

import numpy as np


STANDARD_LINE_WINDOWS_A = (
    (6562.8, 15.0),
    (4861.3, 15.0),
    (8446.0, 15.0),
)


def nearest_channel_index(waves_A, target_A) -> int:
    """Return index of the finite wavelength channel closest to target_A."""

    waves = np.asarray(waves_A, dtype=np.float64)
    if not np.isfinite(target_A):
        raise RuntimeError("Cannot choose a wavelength channel from a non-finite target wavelength.")
    good = np.isfinite(waves)
    if not np.any(good):
        raise RuntimeError("Wavelength array has no finite values.")
    idx_good = np.where(good)[0]
    return int(idx_good[np.argmin(np.abs(waves[good] - float(target_A)))])


def nearest_channel_indices(waves_A, targets_A) -> np.ndarray:
    """Return nearest finite wavelength-channel indices for target wavelengths."""

    return np.array([nearest_channel_index(waves_A, target) for target in targets_A], dtype=int)


def continuum_running_median(waves, spec, good_mask, window_A=80.0, min_pixels=15) -> np.ndarray:
    """Estimate a broad continuum with a wavelength-window running median."""

    waves = np.asarray(waves, dtype=np.float64)
    spec = np.asarray(spec, dtype=np.float64)
    good = np.asarray(good_mask, dtype=bool) & np.isfinite(spec) & np.isfinite(waves)
    continuum = np.full(spec.shape, np.nan, dtype=np.float64)
    half_width = float(window_A) / 2.0
    for i, wave in enumerate(waves):
        local = good & (np.abs(waves - wave) <= half_width)
        if int(np.sum(local)) >= int(min_pixels):
            continuum[i] = np.nanmedian(spec[local])
    return continuum


def control_reference_bias(
    waves, control_spectra, good_mask=None, *, window_A=80.0, min_pixels=15
) -> np.ndarray:
    """Per-channel extraction bias estimated from same-radius control apertures.

    The controls are extracted identically to the object but contain no
    companion, so their MEAN spectrum is the extraction/background bias at the
    source radius (residual chromatic halo subtraction, pedestal). It is
    smoothed with the same running-median window used for the object continuum,
    because the bias is a smooth chromatic function with no spectral lines; the
    smoothing keeps the referenced continuum from inheriting per-channel control
    noise. Subtracting this bias from a method's continuum re-references it to a
    source-free baseline — the model-free "control = object" correction
    (D1 v2 §3.1); it does not touch emission lines (the mean control has none).
    """

    waves = np.asarray(waves, dtype=np.float64)
    controls = np.asarray(control_spectra, dtype=np.float64)
    if controls.ndim != 2 or controls.shape[1] != waves.size:
        raise ValueError("control_spectra must be (n_controls, n_channels) matching waves.")
    with np.errstate(all="ignore"):
        mean_control = np.nanmean(controls, axis=0)
    finite = np.isfinite(mean_control)
    mask = finite if good_mask is None else (np.asarray(good_mask, dtype=bool) & finite)
    return continuum_running_median(waves, mean_control, mask, window_A=window_A, min_pixels=min_pixels)


def median_filter_1d(values, width=21) -> np.ndarray:
    """Centered NaN-median filter with truncated edges."""

    arr = np.asarray(values, dtype=np.float64)
    width = int(width)
    if width <= 1 or arr.size == 0:
        return arr.copy()
    half = width // 2
    out = np.empty_like(arr)
    for i in range(arr.size):
        lo = max(0, i - half)
        hi = min(arr.size, i + half + 1)
        with np.errstate(invalid="ignore"):
            out[i] = np.nanmedian(arr[lo:hi])
    return out


def standard_line_free_mask(waves_A, base_mask=None, *, line_windows_A=STANDARD_LINE_WINDOWS_A) -> np.ndarray:
    """Mask finite wavelengths outside the standard Halpha/Hbeta/OI windows."""

    waves = np.asarray(waves_A, dtype=np.float64)
    mask = np.isfinite(waves)
    if base_mask is not None:
        mask &= np.asarray(base_mask, dtype=bool)
    for center, half_width in line_windows_A:
        mask &= np.abs(waves - float(center)) > float(half_width)
    return mask


def continuum_polyfit_loglambda(
    waves,
    spec,
    good_mask,
    *,
    degree=5,
    max_iter=4,
    lower_sigma=3.0,
    upper_sigma=5.0,
) -> np.ndarray:
    """Fit a polynomial continuum in log(lambda) with asymmetric clipping."""

    waves = np.asarray(waves, dtype=np.float64)
    spec = np.asarray(spec, dtype=np.float64)
    good = np.asarray(good_mask, dtype=bool) & np.isfinite(waves) & np.isfinite(spec) & (waves > 0)
    continuum = np.full(spec.shape, np.nan, dtype=np.float64)
    if int(np.count_nonzero(good)) < 2:
        return continuum
    x_all = np.log(waves)
    x0 = float(np.nanmedian(x_all[good]))
    xscale = float(np.nanmax(np.abs(x_all[good] - x0)))
    if not np.isfinite(xscale) or xscale <= 0:
        xscale = 1.0
    x = (x_all - x0) / xscale
    fit_mask = good.copy()
    coeff = None
    for _ in range(max(1, int(max_iter))):
        n_good = int(np.count_nonzero(fit_mask))
        if n_good < 2:
            break
        deg = min(int(degree), n_good - 1)
        coeff = np.polyfit(x[fit_mask], spec[fit_mask], deg)
        model = np.polyval(coeff, x)
        resid = spec - model
        local = resid[fit_mask]
        sigma = 1.4826 * np.nanmedian(np.abs(local - np.nanmedian(local)))
        if not np.isfinite(sigma) or sigma <= 0:
            sigma = np.nanstd(local)
        if not np.isfinite(sigma) or sigma <= 0:
            break
        new_mask = good & (resid >= -float(lower_sigma) * sigma) & (resid <= float(upper_sigma) * sigma)
        if np.array_equal(new_mask, fit_mask):
            break
        fit_mask = new_mask
    if coeff is not None:
        continuum = np.polyval(coeff, x).astype(np.float64)
    return continuum


def make_wavelength_mask(
    waves_A,
    cfg=None,
    *,
    wave_min_A=None,
    wave_max_A=None,
    exclude_drop_range=True,
    min_channels=5,
):
    """Build a finite wavelength mask with optional bounds and config drop range."""

    cfg = {} if cfg is None else cfg
    waves = np.asarray(waves_A, dtype=np.float64)
    mask = np.isfinite(waves)

    if wave_min_A is not None:
        mask &= waves >= float(wave_min_A)
    if wave_max_A is not None:
        mask &= waves <= float(wave_max_A)

    drop_min = cfg.get("drop_wave_min_A")
    drop_max = cfg.get("drop_wave_max_A")
    if exclude_drop_range and drop_min is not None and drop_max is not None:
        mask &= ~((waves >= float(drop_min)) & (waves <= float(drop_max)))

    if int(np.count_nonzero(mask)) < int(min_channels):
        raise RuntimeError("Too few wavelength channels remain after masking.")
    return mask


def spectrum_to_snr(spec_1d, noise):
    """Divide a spectrum by a scalar noise level.

    Returns an all-NaN array (shaped like the input) when ``noise`` is not
    finite or not positive; otherwise ``spec_1d / noise``.
    """

    spec_1d = np.asarray(spec_1d, dtype=np.float64)
    if not np.isfinite(noise) or noise <= 0:
        return np.full_like(spec_1d, np.nan, dtype=np.float64)
    return spec_1d / float(noise)


def integrated_line_flux(spec_1d, line_idxs, dlam_A):
    """Integrated line flux over the given channel indices.

    ``nansum(spec_1d[line_idxs]) * dlam_A``; ``line_idxs`` is cast to int.
    """

    idx = np.asarray(line_idxs, dtype=int)
    return float(np.nansum(np.asarray(spec_1d)[idx]) * dlam_A)


#: Halpha en reposo, en aire. El mismo valor que usan E1 y G2.
HALPHA_REST_A = 6562.8

#: knobs que declaran la LSF, del mas especifico al mas general. La convencion del
#: proyecto es que el DECLARADO gana sobre el medido (igual que `resolve_flux_unit` y
#: `_wavelength_frame`), para que un run pueda congelar el numero que ya publico.
LSF_CONFIG_KEYS = ("h01_lsf_fwhm_A", "lsf_fwhm_A")

#: donde A4/M2 guarda la medida, en orden de preferencia. `lsf_fwhm_at_halpha_A` es la
#: clave REAL que escribe A4; las otras son nombres que tuvo antes.
LSF_M2_KEYS = ("lsf_fwhm_at_halpha_A", "fwhm_at_halpha_A", "halpha_fwhm_A", "lsf_fwhm_A")


def resolve_lsf_fwhm_A(qc00, cfg, *, stage_key=None):
    """FWHM de la LSF en Halpha: knob declarado -> medida de A4/M2 -> error.

    Devuelve ``(valor, procedencia)``. **Nunca un default silencioso**, que es
    precisamente lo que este resolutor viene a arreglar: hasta el 2026-09-10, C5 y C6
    corrian con 2.6 A porque su buscador miraba `lsf.fwhm_A` y `cube.lsf_fwhm_A` -que
    A4 no escribe, escribe `m2_lsf.lsf_fwhm_at_halpha_A`- y luego `lsf_fwhm_A`, que el
    config declara como `h01_lsf_fwhm_A`. Los dos caminos fallaban en silencio y el
    predictor de autosustraccion que publican salia con la LSF equivocada.

    Existe UNO solo, compartido, porque el numero lo consumen E1, G2, G3, C5 y C6 y ya
    van tres veces en este proyecto que dos etapas que se citan juntas no comparten una
    definicion.
    """

    for key in ((stage_key,) if stage_key else ()) + LSF_CONFIG_KEYS:
        value = (cfg or {}).get(key)
        if value is not None:
            return float(value), f"config.{key}"

    m2 = ((qc00 or {}).get("m2_lsf") or {})
    for key in LSF_M2_KEYS:
        if m2.get(key) is not None:
            return float(m2[key]), f"stage00q_qc.m2_lsf.{key}"

    coeffs = m2.get("poly2_coeffs")
    if coeffs:
        value = float(np.polyval(np.asarray(coeffs, dtype=np.float64), HALPHA_REST_A))
        return value, "stage00q_qc.m2_lsf.poly2_coeffs"

    table = m2.get("table_A_fwhm")
    if table:
        waves = np.asarray([r[0] if isinstance(r, (list, tuple)) else r.get("wave_A")
                            for r in table], dtype=np.float64)
        fwhm = np.asarray([r[1] if isinstance(r, (list, tuple)) else r.get("fwhm_A")
                           for r in table], dtype=np.float64)
        return float(np.interp(HALPHA_REST_A, waves, fwhm)), "stage00q_qc.m2_lsf.table_A_fwhm"

    raise RuntimeError(
        "No hay FWHM de la LSF en Halpha: declara "
        + " o ".join(f"`{k}`" for k in (((stage_key,) if stage_key else ()) + LSF_CONFIG_KEYS))
        + " en el config, o corre A4/M2 para que escriba `m2_lsf` en stage00q_qc.json. "
        "No hay valor por defecto a proposito."
    )


__all__ = [
    "continuum_running_median",
    "HALPHA_REST_A",
    "LSF_CONFIG_KEYS",
    "LSF_M2_KEYS",
    "resolve_lsf_fwhm_A",
    "continuum_polyfit_loglambda",
    "integrated_line_flux",
    "make_wavelength_mask",
    "median_filter_1d",
    "nearest_channel_index",
    "nearest_channel_indices",
    "standard_line_free_mask",
    "STANDARD_LINE_WINDOWS_A",
    "spectrum_to_snr",
]
