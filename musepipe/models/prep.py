"""Common model preparation: frame, LSF degradation and flux-conserving
resampling (spec G3 §3.1). Nothing here knows the library family.

Plan G3 2026-09-23 (decisión en ``docs/2026-09-23_decision_g3_resolucion_y_bibliotecas.md``):

* **Marco**: la plantilla se lleva al marco del dato (aire, MUSE) ANTES de
  degradarla (:mod:`musepipe.models.airvac`).
* **Resolución declarada**: la FWHM de la plantilla puede depender de λ
  (``λ/R``); el núcleo es la cuadratura ``√(FWHM_objetivo(λ)² − FWHM_plantilla(λ)²)``
  evaluada por tramos. Ninguna biblioteca es infinitamente fina salvo que lo
  declare: esa regla vive en :mod:`musepipe.models.libraries`; aquí ``None``
  conserva el comportamiento histórico para los llamadores sintéticos.
* **Resolución objetivo**: por defecto la LSF del dato; en la prueba de una
  biblioteca más gruesa que MUSE es la de la biblioteca (el DATO se degrada,
  :func:`musepipe.models.observed.degrade_observed`).
"""

from __future__ import annotations

import math

import numpy as np

from ..constants import fwhm_to_sigma
from .airvac import convert_frame, normalize_frame

#: Longitud de los tramos en que se aproxima un núcleo de anchura variable.
#: λ/R cambia < 1.5 % en 100 Å a 7000 Å: el error de tomar la anchura del
#: centro del tramo es de segundo orden frente a esa variación.
VARIABLE_KERNEL_CHUNK_A = 100.0


def _bin_edges(wave):
    wave = np.asarray(wave, dtype=np.float64)
    edges = np.empty(wave.size + 1)
    edges[1:-1] = 0.5 * (wave[:-1] + wave[1:])
    edges[0] = wave[0] - 0.5 * (wave[1] - wave[0])
    edges[-1] = wave[-1] + 0.5 * (wave[-1] - wave[-2])
    return edges


def resample_conserve_flux(wave_in, flux_in, wave_out):
    """Rebin a spectral-flux-DENSITY onto ``wave_out`` conserving integrated flux.

    Overlap of input/output bins (a density is integrated over each output bin
    and divided by its width), so the total integral is preserved exactly. The
    part of an output bin outside the input coverage contributes zero (as the
    historical loop did); an output bin overlapping a non-finite input bin is
    NaN. Vectorized with the cumulative integral (same numbers as the old
    per-bin loop to rounding).
    """
    wave_in = np.asarray(wave_in, dtype=np.float64)
    flux_in = np.asarray(flux_in, dtype=np.float64)
    ein = _bin_edges(wave_in)
    eout = _bin_edges(np.asarray(wave_out, dtype=np.float64))
    bad = ~np.isfinite(flux_in)
    f = np.where(bad, 0.0, flux_in)
    cum = np.concatenate([[0.0], np.cumsum(f * np.diff(ein))])
    nbad = np.concatenate([[0], np.cumsum(bad.astype(np.int64))])
    n = wave_in.size

    def integral(x):
        xc = np.clip(x, ein[0], ein[-1])
        k = np.clip(np.searchsorted(ein, xc, side="right") - 1, 0, n - 1)
        return cum[k] + f[k] * (xc - ein[k])

    lo, hi = eout[:-1], eout[1:]
    width = hi - lo
    acc = integral(hi) - integral(lo)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(width > 0, acc / width, np.nan)
    if bad.any():
        klo = np.clip(np.searchsorted(ein, lo, side="right") - 1, 0, n)
        khi = np.clip(np.searchsorted(ein, hi, side="left"), 0, n)
        touched = (nbad[khi] - nbad[klo]) > 0
        out = np.where(touched, np.nan, out)
    return out


def _gauss_kernel(sigma_pix):
    half = max(1, int(np.ceil(4.0 * sigma_pix)))
    x = np.arange(-half, half + 1)
    kernel = np.exp(-0.5 * (x / sigma_pix) ** 2)
    return kernel / kernel.sum(), half


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
    kernel, _ = _gauss_kernel(sigma_pix)
    return np.convolve(flux, kernel, mode="same")


def degrade_variable(wave, flux, fwhm_A, *, chunk_A=VARIABLE_KERNEL_CHUNK_A,
                     edge="nearest"):
    """Gaussian convolution whose FWHM [Å] may vary with λ (scalar or per-sample).

    Piecewise: the range is cut in tramos of ``chunk_A``; each uses the median
    FWHM and median sampling step of its tramo, convolved over a window padded
    by 4σ so the tramo interior never sees an artificial edge. ``edge`` is how
    the ARRAY ends are padded (``'nearest'`` repeats the end value; ``'zero'``
    is the historical :func:`degrade_to_lsf`). FWHM ≤ 0 leaves the tramo as is.
    """
    wave = np.asarray(wave, dtype=np.float64)
    flux = np.asarray(flux, dtype=np.float64)
    n = wave.size
    if n < 3:
        return flux.copy()
    fwhm = np.broadcast_to(np.asarray(fwhm_A, dtype=np.float64), wave.shape)
    out = flux.copy()
    edges = np.arange(wave[0], wave[-1] + chunk_A, chunk_A)
    idx = np.searchsorted(wave, edges)
    idx[-1] = n
    starts = np.unique(np.r_[idx[:-1], 0])
    bounds = [(s, e) for s, e in zip(starts, np.r_[starts[1:], n]) if e > s]
    for s, e in bounds:
        fw = float(np.median(fwhm[s:e]))
        if not np.isfinite(fw) or fw <= 0:
            continue
        seg_w = wave[max(s - 1, 0):min(e + 1, n)]
        dl = float(np.median(np.diff(seg_w))) if seg_w.size > 1 else float(wave[1] - wave[0])
        sigma_pix = fwhm_to_sigma(fw) / dl
        if sigma_pix <= 1e-6:
            continue
        kernel, half = _gauss_kernel(sigma_pix)
        lo, hi = s - half, e + half
        pad_lo, pad_hi = max(0, -lo), max(0, hi - n)
        seg = flux[max(lo, 0):min(hi, n)]
        if pad_lo or pad_hi:
            if edge == "zero":
                seg = np.pad(seg, (pad_lo, pad_hi), mode="constant")
            else:
                seg = np.pad(seg, (pad_lo, pad_hi), mode="edge")
        conv = np.convolve(seg, kernel, mode="same")
        out[s:e] = conv[half:half + (e - s)]
    return out


def _as_fwhm_function(value):
    """Scalar, callable(wave)->fwhm or None → callable or None."""
    if value is None:
        return None
    if callable(value):
        return value
    v = float(value)
    return lambda w: np.full(np.shape(w), v, dtype=np.float64)


def template_fwhm_function(*, template_fwhm_A=None, template_R=None):
    """Resolución declarada de una plantilla como función de λ, o ``None``.

    ``template_R`` (poder resolutivo) da FWHM(λ) = λ/R; ``template_fwhm_A`` una
    FWHM constante (regla D2 histórica). ``R = inf`` es «nítida» (FWHM 0).
    """
    if template_R is not None:
        r = float(template_R)
        if not r > 0:
            raise RuntimeError(f"poder resolutivo inválido: R={template_R!r}")
        if math.isinf(r):
            return lambda w: np.zeros(np.shape(w), dtype=np.float64)
        return lambda w: np.asarray(w, dtype=np.float64) / r
    return _as_fwhm_function(template_fwhm_A)


def prepare_template_base(template, wave_out, *, lsf_fwhm_A, template_fwhm_A=None,
                          template_R=None, template_frame=None, data_frame=None,
                          target_fwhm_A=None):
    """Frame → degrade → resample, WITHOUT extinction or scale.

    Returns ``(flux_on_wave_out, resolution_mismatch)``. ``target_fwhm_A``
    (scalar or callable of λ; default ``lsf_fwhm_A``) is the resolution the
    model must reach. The kernel is √(target² − tmpl²) per λ; where the
    template is already coarser than the target nothing is done there and
    ``resolution_mismatch`` is flagged (D2). With no declared template
    resolution the template is taken as sharp (historical behaviour, for
    synthetic callers only — real libraries always declare it).
    Output bins with NO overlap with the template coverage are NaN (they
    used to be 0, which a χ² would fit as real flux).
    """
    wave_t = np.asarray(template.wave_A, dtype=np.float64)
    flux_t = np.asarray(template.flux, dtype=np.float64)
    if template_frame is not None or data_frame is not None:
        if normalize_frame(template_frame) is None or normalize_frame(data_frame) is None:
            raise RuntimeError(
                f"marco incompleto: plantilla={template_frame!r}, dato={data_frame!r}; "
                "se declaran los dos o ninguno (sin marco por defecto)")
        wave_t = convert_frame(wave_t, template_frame, data_frame)

    target = _as_fwhm_function(target_fwhm_A if target_fwhm_A is not None else lsf_fwhm_A)
    tmpl = template_fwhm_function(template_fwhm_A=template_fwhm_A, template_R=template_R)
    target_t = target(wave_t)
    if tmpl is None:
        kernel = target_t
        mismatch = False
    else:
        tf = tmpl(wave_t)
        kernel = np.sqrt(np.clip(target_t ** 2 - tf ** 2, 0.0, None))
        wo = np.asarray(wave_out, dtype=np.float64)
        in_out = (wave_t >= wo.min()) & (wave_t <= wo.max()) if wo.size else np.ones_like(wave_t, bool)
        mismatch = (bool(np.any(tf[in_out] > target_t[in_out] * (1 + 1e-9)))
                    if in_out.any() else False)
    if np.all(kernel == kernel.flat[0]) and kernel.size:
        k0 = float(kernel.flat[0])
        flux = degrade_to_lsf(wave_t, flux_t, k0) if k0 > 0 else flux_t.copy()
    else:
        flux = degrade_variable(wave_t, flux_t, kernel)
    out = resample_conserve_flux(wave_t, flux, wave_out)
    wo = np.asarray(wave_out, dtype=np.float64)
    if wo.size >= 2:
        eo = _bin_edges(wo)
        et = _bin_edges(wave_t)
        uncovered = (eo[1:] <= et[0]) | (eo[:-1] >= et[-1])
        out = np.where(uncovered, np.nan, out)
    return out, mismatch


def apply_extinction(flux, wave_out, extinction, av):
    """Redden a model toward the data: × 10^(−0.4 A_λ)."""
    if extinction is None or not av:
        return np.asarray(flux, dtype=np.float64)
    a_lam = np.asarray(extinction.a_lambda_over_av(wave_out)) * float(av)
    return np.asarray(flux, dtype=np.float64) * np.power(10.0, -0.4 * a_lam)


def prepare_template(template, wave_out, *, lsf_fwhm_A, extinction=None, av=0.0,
                     scale=1.0, template_fwhm_A=None, template_R=None,
                     template_frame=None, data_frame=None, target_fwhm_A=None,
                     return_flag=False):
    """Frame → degrade to the target resolution → resample → extinction → scale.

    ``template_fwhm_A`` (decision D2): if given and < ``lsf_fwhm_A`` the template
    is degraded by the quadrature kernel √(lsf² − tmpl²); if ≥ ``lsf_fwhm_A`` it
    is NOT degraded and ``resolution_mismatch`` is flagged. ``template_R`` does
    the same with FWHM(λ) = λ/R. The default (neither) reproduces the historical
    behaviour (degrade by the full LSF, valid for ~infinite-resolution models).
    ``template_frame``/``data_frame`` ('air'/'vacuum') convert the template to
    the data frame first. With ``return_flag=True`` the return is
    ``(flux, resolution_mismatch)``; otherwise just ``flux``.
    """
    flux, mismatch = prepare_template_base(
        template, wave_out, lsf_fwhm_A=lsf_fwhm_A, template_fwhm_A=template_fwhm_A,
        template_R=template_R, template_frame=template_frame, data_frame=data_frame,
        target_fwhm_A=target_fwhm_A)
    flux = apply_extinction(flux, wave_out, extinction, av)
    flux = float(scale) * flux
    return (flux, mismatch) if return_flag else flux


__all__ = ["VARIABLE_KERNEL_CHUNK_A", "apply_extinction", "degrade_to_lsf",
           "degrade_variable", "prepare_template", "prepare_template_base",
           "resample_conserve_flux", "template_fwhm_function"]
