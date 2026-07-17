"""Common model preparation: LSF degradation and flux-conserving resampling
(spec G3 §3.1). Nothing here knows the library family."""

from __future__ import annotations

import math

import numpy as np

from ..constants import fwhm_to_sigma


def _bin_edges(wave):
    wave = np.asarray(wave, dtype=np.float64)
    edges = np.empty(wave.size + 1)
    edges[1:-1] = 0.5 * (wave[:-1] + wave[1:])
    edges[0] = wave[0] - 0.5 * (wave[1] - wave[0])
    edges[-1] = wave[-1] + 0.5 * (wave[-1] - wave[-2])
    return edges


def resample_conserve_flux(wave_in, flux_in, wave_out):
    """Rebin a spectral-flux-DENSITY onto ``wave_out`` conserving integrated flux.

    Uses overlap of input/output bins (a density is integrated over each output
    bin and divided by its width), so total integral is preserved exactly.
    """
    wave_in = np.asarray(wave_in, dtype=np.float64)
    flux_in = np.asarray(flux_in, dtype=np.float64)
    ein = _bin_edges(wave_in)
    eout = _bin_edges(np.asarray(wave_out, dtype=np.float64))
    out = np.zeros(np.asarray(wave_out).size, dtype=np.float64)
    for j in range(out.size):
        lo, hi = eout[j], eout[j + 1]
        left = np.searchsorted(ein, lo) - 1
        right = np.searchsorted(ein, hi)
        acc = 0.0
        for k in range(max(left, 0), min(right, wave_in.size)):
            overlap = max(0.0, min(hi, ein[k + 1]) - max(lo, ein[k]))
            acc += flux_in[k] * overlap
        width = hi - lo
        out[j] = acc / width if width > 0 else np.nan
    return out


def degrade_to_lsf(wave, flux, lsf_fwhm_A):
    """Convolve with a Gaussian LSF of the given FWHM (assumes ~uniform sampling)."""
    wave = np.asarray(wave, dtype=np.float64)
    flux = np.asarray(flux, dtype=np.float64)
    if wave.size < 3:
        return flux.copy()
    dl = float(np.median(np.diff(wave)))
    sigma_pix = fwhm_to_sigma(float(lsf_fwhm_A)) / dl
    if sigma_pix <= 0:
        return flux.copy()
    half = max(1, int(np.ceil(4.0 * sigma_pix)))
    x = np.arange(-half, half + 1)
    kernel = np.exp(-0.5 * (x / sigma_pix) ** 2)
    kernel /= kernel.sum()
    return np.convolve(flux, kernel, mode="same")


def prepare_template(template, wave_out, *, lsf_fwhm_A, extinction=None, av=0.0,
                     scale=1.0, template_fwhm_A=None, return_flag=False):
    """Degrade to LSF → resample to wave_out → apply extinction → scale.

    ``template_fwhm_A`` (decision D2): if given and < ``lsf_fwhm_A`` the template
    is degraded by the quadrature kernel √(lsf² − tmpl²); if ≥ ``lsf_fwhm_A`` it
    is NOT degraded and ``resolution_mismatch`` is flagged. The default
    (``None``) reproduces the historical behaviour (degrade by the full LSF,
    valid for ~infinite-resolution models). With ``return_flag=True`` the return
    is ``(flux, resolution_mismatch)``; otherwise just ``flux`` (back-compatible).
    """
    resolution_mismatch = False
    if template_fwhm_A is None:
        kernel_fwhm = float(lsf_fwhm_A)
    elif float(template_fwhm_A) < float(lsf_fwhm_A):
        kernel_fwhm = math.sqrt(float(lsf_fwhm_A) ** 2 - float(template_fwhm_A) ** 2)
    else:
        kernel_fwhm = 0.0  # template coarser than the LSF → do not degrade
        resolution_mismatch = True
    if kernel_fwhm > 0:
        flux = degrade_to_lsf(template.wave_A, template.flux, kernel_fwhm)
    else:
        flux = np.asarray(template.flux, dtype=np.float64)
    flux = resample_conserve_flux(template.wave_A, flux, wave_out)
    if extinction is not None and av:
        a_lam = np.asarray(extinction.a_lambda_over_av(wave_out)) * float(av)
        flux = flux * np.power(10.0, -0.4 * a_lam)  # redden the model toward the data
    flux = float(scale) * flux
    return (flux, resolution_mismatch) if return_flag else flux


__all__ = ["degrade_to_lsf", "prepare_template", "resample_conserve_flux"]
