"""Chromatic PSF primitives for MUSE cube extraction stages."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import functools
import warnings
import math

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import least_squares

from .stats import finite_percentile, robust_sigma


PSF_SHAPE_PARAMS = ("y0", "x0", "fwhm_maj", "fwhm_min", "theta_deg", "beta")


@dataclass(frozen=True)
class MoffatFit:
    success: bool
    params: dict[str, float]
    errors: dict[str, float]
    background: float
    chi2r: float
    clip_frac: float
    n_fit: int
    message: str
    #: Razon de flujos de una segunda componente LIGADA (misma forma, posicion
    #: fija). `None` = ajuste de una sola fuente, que es el caso por defecto y el
    #: de todos los objetos menos ROXs 42B b. Ver `fit_moffat_image`.
    flux_ratio: float | None = None
    companion_offset_yx: tuple[float, float] | None = None


MOFFAT_BETA_FLOOR = 1.05  # Moffat is only a normalizable PSF for beta > 1.


def moffat_alpha_from_fwhm(fwhm, beta):
    fwhm = float(fwhm)
    beta = float(beta)
    # A Moffat has finite integral only for beta > 1; a degree-N beta(lambda)
    # polynomial from C1 can extrapolate to beta <= 0 at band edges outside its
    # fit range (seen on the LkCa 15 Moffat fit: 382/3681 channels beta<=0),
    # which sends 2**(1/beta) to an OverflowError and crashes every downstream
    # apcorr. Clamp to the physical floor: a no-op for any healthy PSF (beta>1),
    # and it keeps the aperture correction finite where the model is being
    # extrapolated into the non-normalizable regime.
    if not math.isfinite(beta) or beta < MOFFAT_BETA_FLOOR:
        beta = MOFFAT_BETA_FLOOR
    denom = 2.0 * math.sqrt(max(2.0 ** (1.0 / beta) - 1.0, 1e-12))
    return fwhm / denom


def moffat_elliptical_profile(dy, dx, fwhm_maj, fwhm_min, theta_deg, beta):
    """Unit-peak elliptical Moffat profile."""

    dy = np.asarray(dy, dtype=np.float64)
    dx = np.asarray(dx, dtype=np.float64)
    theta = np.deg2rad(float(theta_deg))
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    x_rot = dx * cos_t + dy * sin_t
    y_rot = -dx * sin_t + dy * cos_t
    alpha_maj = moffat_alpha_from_fwhm(fwhm_maj, beta)
    alpha_min = moffat_alpha_from_fwhm(fwhm_min, beta)
    rr = (x_rot / alpha_maj) ** 2 + (y_rot / alpha_min) ** 2
    return (1.0 + rr) ** (-float(beta))


def moffat_image(shape, y0, x0, fwhm_maj, fwhm_min, theta_deg, beta, amplitude=1.0, background=0.0):
    yy, xx = np.indices(shape, dtype=np.float64)
    profile = moffat_elliptical_profile(
        yy - float(y0),
        xx - float(x0),
        fwhm_maj,
        fwhm_min,
        theta_deg,
        beta,
    )
    return float(background) + float(amplitude) * profile


def fixed_radius_grid(norm_radius_px):
    radius = float(norm_radius_px)
    half = int(math.ceil(radius))
    yy, xx = np.mgrid[-half : half + 1, -half : half + 1].astype(np.float64)
    mask = (yy**2 + xx**2) <= radius**2
    return yy, xx, mask


def moffat_norm(params, norm_radius_px=25.0):
    yy, xx, mask = fixed_radius_grid(norm_radius_px)
    profile = moffat_elliptical_profile(
        yy,
        xx,
        params["fwhm_maj"],
        params["fwhm_min"],
        params.get("theta_deg", 0.0),
        params["beta"],
    )
    norm = float(np.nansum(profile[mask]))
    if not np.isfinite(norm) or norm <= 0:
        raise RuntimeError("Invalid Moffat normalization.")
    return norm


def normalized_moffat_psf(dy, dx, params, norm_radius_px=25.0):
    profile = moffat_elliptical_profile(
        dy,
        dx,
        params["fwhm_maj"],
        params["fwhm_min"],
        params.get("theta_deg", 0.0),
        params["beta"],
    )
    return profile / moffat_norm(params, norm_radius_px=norm_radius_px)


def source_mask(shape, centers_yx, radius_px):
    yy, xx = np.indices(shape, dtype=np.float64)
    mask = np.zeros(shape, dtype=bool)
    for center in centers_yx or ():
        if center is None:
            continue
        y, x = map(float, center)
        mask |= (yy - y) ** 2 + (xx - x) ** 2 <= float(radius_px) ** 2
    return mask


def corner_background(image, corner_size=12):
    img = np.asarray(image, dtype=np.float64)
    c = int(min(corner_size, max(1, img.shape[0] // 4), max(1, img.shape[1] // 4)))
    vals = np.concatenate(
        [
            img[:c, :c].ravel(),
            img[:c, -c:].ravel(),
            img[-c:, :c].ravel(),
            img[-c:, -c:].ravel(),
        ]
    )
    vals = vals[np.isfinite(vals)]
    return float(np.nanmedian(vals)) if vals.size else 0.0


def _initial_fit_params(image, center_yx, background, fit_mask):
    img = np.asarray(image, dtype=np.float64)
    y0, x0 = map(float, center_yx)
    peak_region = fit_mask & np.isfinite(img)
    peak = float(np.nanmax(img[peak_region] - float(background))) if np.any(peak_region) else 1.0
    if not np.isfinite(peak) or peak <= 0:
        peak = 1.0
    return np.array([peak, y0, x0, 4.0, 4.0, 0.0, 2.5], dtype=np.float64)


def _pack_params(values):
    amp, y0, x0, fmaj, fmin, theta, beta = values
    if fmin > fmaj:
        fmaj, fmin = fmin, fmaj
        theta += 90.0
    theta = ((float(theta) + 90.0) % 180.0) - 90.0
    return {
        "amplitude": float(amp),
        "y0": float(y0),
        "x0": float(x0),
        "fwhm_maj": float(fmaj),
        "fwhm_min": float(fmin),
        "theta_deg": theta,
        "beta": float(beta),
    }


def fit_moffat_image(
    image,
    *,
    center_yx,
    fit_radius_px=28.0,
    mask=None,
    background=None,
    core_mask_px=0.0,
    sigma_clip=3.0,
    max_iter=3,
    min_pixels=40,
    companion_offset_yx=None,
):
    """Fit a fixed-background elliptical Moffat image model.

    Con `companion_offset_yx` el modelo pasa a tener **dos componentes ligadas**:
    la misma Moffat en `(y0, x0)` y en `(y0+dy, x0+dx)`, con **los mismos**
    parametros de forma y una unica incognita nueva, la razon de flujos `f`.

    Existe porque ROXs 42B es una **binaria cercana no resuelta** (rho = 51 mas,
    PA = 148 deg segun Keck/NIRC2 2022.621, la misma epoca que estos datos; a
    25.42 mas/px son 2.006 px) y ajustar UNA PSF a DOS estrellas la mide **mas
    ancha de lo que es**: medido el 2026-08-30, eso sesga la `apcorr` un +6.9 %
    contra un +1.5 % de suelo en ROXs 12 b, que es una estrella sola.

    **No son dos fuentes libres**: comparten forma y su separacion la fija la
    astrometria publicada, asi que el ajuste gana **exactamente un** grado de
    libertad. Con el argumento a `None` el camino es identico al de siempre.

    Lo que devuelve `params` sigue describiendo **una fuente puntual** — la
    segunda componente viaja aparte, en `flux_ratio`. Esa separacion es
    deliberada: la PSF publicada la consumen la inyeccion de E4, la `apcorr` y el
    throughput, que necesitan un punto y no la imagen de la primaria.
    """

    img = np.asarray(image, dtype=np.float64)
    if img.ndim != 2:
        raise ValueError(f"Expected a 2D image, got {img.shape}.")
    ny, nx = img.shape
    cy, cx = map(float, center_yx)
    yy, xx = np.indices(img.shape, dtype=np.float64)
    rr = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    fit_mask = (rr <= float(fit_radius_px)) & np.isfinite(img)
    if mask is not None:
        fit_mask &= ~np.asarray(mask, dtype=bool)
    if float(core_mask_px) > 0:
        fit_mask &= rr > float(core_mask_px)
    if int(np.count_nonzero(fit_mask)) < int(min_pixels):
        raise RuntimeError("Too few pixels for Moffat fit.")

    background = corner_background(img) if background is None else float(background)
    y = img[fit_mask] - background
    ypix = yy[fit_mask]
    xpix = xx[fit_mask]
    good = np.ones(y.size, dtype=bool)
    x0 = _initial_fit_params(img, center_yx, background, fit_mask)
    lower = [0.0, cy - 3.0, cx - 3.0, 0.6, 0.6, -90.0, 1.05]
    upper = [
        max(float(np.nanmax(y)) * 3.0, 1.0),
        cy + 3.0,
        cx + 3.0,
        max(2.0, fit_radius_px),
        max(2.0, fit_radius_px),
        90.0,
        12.0,
    ]
    # La segunda componente añade UN parametro y ninguno mas: la forma se comparte
    # y la posicion la fija la astrometria declarada.
    offset = None if companion_offset_yx is None else tuple(map(float, companion_offset_yx))
    if offset is not None:
        x0 = np.concatenate([x0, [0.1]])
        lower = lower + [0.0]
        upper = upper + [1.0]

    def _shape_values(values):
        """Separa los 7 de forma de la razon de flujos, que va al final."""
        return (values[:7], float(values[7]) if offset is not None else None)

    # El recorte sigma existe para tirar rayos cosmicos y pixeles malos, no
    # senal. Con dos componentes la secundaria cae a ~2 px del centro y en la
    # PRIMERA iteracion su residuo es grande, asi que el clip se la lleva y las
    # iteraciones siguientes ya no la ven: medido el 2026-08-30, eso inventaba
    # f=0.073 en una estrella SOLA (contra 0.004 sin recorte) y desplazaba el
    # minimo del barrido de PA 48 grados. Los pixeles donde viven las dos
    # componentes quedan exentos del recorte; fuera de esa zona sigue igual.
    protegido = None
    if offset is not None:
        radio_protegido = 2.0 * float(np.hypot(*offset))
        protegido = np.hypot(ypix - cy, xpix - cx) <= radio_protegido

    def _scene(values, ypix_sel, xpix_sel):
        """La ESCENA: una componente, o dos ligadas. Es contra esto que se ajusta,
        porque es lo que el dato tiene."""
        head, ratio = _shape_values(values)
        params = _pack_params(head)
        model = moffat_elliptical_profile(
            ypix_sel - params["y0"], xpix_sel - params["x0"],
            params["fwhm_maj"], params["fwhm_min"], params["theta_deg"], params["beta"],
        )
        if offset is not None:
            model = model + float(ratio) * moffat_elliptical_profile(
                ypix_sel - (params["y0"] + offset[0]),
                xpix_sel - (params["x0"] + offset[1]),
                params["fwhm_maj"], params["fwhm_min"], params["theta_deg"], params["beta"],
            )
        return params["amplitude"] * model

    fit = None
    for _ in range(int(max_iter)):
        if int(np.count_nonzero(good)) < int(min_pixels):
            break

        def resid(values):
            return _scene(values, ypix[good], xpix[good]) - y[good]

        fit = least_squares(resid, x0=x0, bounds=(lower, upper), max_nfev=500)
        residual_all = _scene(fit.x, ypix, xpix) - y
        sigma = robust_sigma(residual_all[good])
        if sigma_clip is None or not np.isfinite(sigma) or sigma <= 0:
            break
        new_good = np.abs(residual_all) <= float(sigma_clip) * sigma
        if protegido is not None:
            new_good |= protegido
        if np.array_equal(new_good, good):
            break
        good = new_good
        x0 = fit.x

    if fit is None:
        raise RuntimeError("Moffat fit did not run.")
    head, flux_ratio = _shape_values(fit.x)
    params = _pack_params(head)
    resid = _scene(fit.x, ypix[good], xpix[good]) - y[good]
    sigma = robust_sigma(resid)
    dof = max(1, int(np.count_nonzero(good)) - (7 if offset is None else 8))
    chi2r = float(np.nansum((resid / sigma) ** 2) / dof) if np.isfinite(sigma) and sigma > 0 else np.nan
    errors = {key: np.nan for key in ("amplitude",) + PSF_SHAPE_PARAMS + ("flux_ratio",)}
    if fit.jac is not None and fit.jac.size and np.isfinite(sigma) and sigma > 0:
        try:
            cov = np.linalg.pinv(fit.jac.T @ fit.jac) * sigma**2
            err_values = np.sqrt(np.clip(np.diag(cov), 0.0, np.inf))
            claves = ("amplitude", "y0", "x0", "fwhm_maj", "fwhm_min", "theta_deg", "beta")
            if offset is not None:
                claves = claves + ("flux_ratio",)
            for key, err in zip(claves, err_values):
                errors[key] = float(err)
        except Exception:
            pass

    return MoffatFit(
        success=bool(fit.success),
        params=params,
        errors=errors,
        background=background,
        chi2r=chi2r,
        clip_frac=float(1.0 - np.count_nonzero(good) / y.size),
        n_fit=int(np.count_nonzero(good)),
        message=str(fit.message),
        flux_ratio=None if flux_ratio is None else float(flux_ratio),
        companion_offset_yx=offset,
    )


def evaluate_moffat_fit(shape, fit: MoffatFit):
    p = fit.params
    return moffat_image(
        shape,
        p["y0"],
        p["x0"],
        p["fwhm_maj"],
        p["fwhm_min"],
        p["theta_deg"],
        p["beta"],
        amplitude=p["amplitude"],
        background=fit.background,
    )


def evaluate_moffat_scene(shape, fit: MoffatFit):
    """La ESCENA que el dato contiene: la PSF y, si la hay, la secundaria ligada.

    **No confundir con `evaluate_moffat_fit`**, que devuelve la PSF de UNA fuente
    puntual y es lo que se publica. La distincion es la que sostiene todo el
    cambio de la binaria:

      * `evaluate_moffat_fit`  -> lo que se publica, y lo que usan la `apcorr`, la
        energia encerrada y el documento del modelo. Un punto.
      * `evaluate_moffat_scene` -> contra lo que se miden los RESIDUOS, porque el
        dato de ROXs 42B b tiene dos estrellas.

    Restar la escena de una imagen con dos estrellas deja el residuo real; restar
    la PSF dejaria la secundaria entera dentro y la metrica del anillo empeoraria
    justo al mejorar el modelo.

    Sin segunda componente las dos funciones devuelven lo mismo.
    """

    base = evaluate_moffat_fit(shape, fit)
    if fit.flux_ratio is None or fit.companion_offset_yx is None:
        return base
    p = fit.params
    dy, dx = fit.companion_offset_yx
    secundaria = moffat_image(
        shape,
        p["y0"] + float(dy),
        p["x0"] + float(dx),
        p["fwhm_maj"],
        p["fwhm_min"],
        p["theta_deg"],
        p["beta"],
        amplitude=p["amplitude"] * float(fit.flux_ratio),
        background=0.0,
    )
    return base + secundaria


def companion_ring_metric(image, model, primary_yx, companion_yx, *, width_px=3.0, source_exclusion_radius_px=0.0):
    img = np.asarray(image, dtype=np.float64)
    mod = np.asarray(model, dtype=np.float64)
    yy, xx = np.indices(img.shape, dtype=np.float64)
    py, px = map(float, primary_yx)
    cy, cx = map(float, companion_yx)
    radius = math.hypot(cy - py, cx - px)
    rr = np.sqrt((yy - py) ** 2 + (xx - px) ** 2)
    ann = np.abs(rr - radius) <= float(width_px) / 2.0
    if source_exclusion_radius_px and source_exclusion_radius_px > 0:
        ann &= (yy - cy) ** 2 + (xx - cx) ** 2 > float(source_exclusion_radius_px) ** 2
    halo = np.abs(mod)
    vals = np.abs(img - mod) / np.maximum(halo, np.nanmedian(halo[ann]) * 0.05)
    vals = vals[ann & np.isfinite(vals)]
    if vals.size == 0:
        return {"radius_px": float(radius), "median_pct": np.nan, "p90_pct": np.nan}
    return {
        "radius_px": float(radius),
        "median_pct": float(100.0 * np.nanmedian(vals)),
        "p90_pct": float(100.0 * finite_percentile(vals, 90.0)),
    }


def encircled_energy(image, center_yx, radii_px, *, background=0.0, exclude_mask=None):
    """``F(r <= radio)`` para cada radio, con el fondo restado.

    ``exclude_mask`` quita pixeles (una fuente de campo dentro del radio, p.ej.)
    y hay que pasarle la MISMA a dato y modelo, o la comparacion no es tal.
    """

    img = np.asarray(image, dtype=np.float64) - float(background)
    yy, xx = np.indices(img.shape, dtype=np.float64)
    cy, cx = map(float, center_yx)
    rr = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    ok = np.isfinite(img)
    if exclude_mask is not None:
        ok &= ~np.asarray(exclude_mask, dtype=bool)
    return np.asarray(
        [float(np.sum(np.where(ok & (rr <= float(r)), img, 0.0))) for r in np.atleast_1d(radii_px)],
        dtype=np.float64,
    )


def core_to_norm_ratio(image, center_yx, *, norm_radius_px=25.0, box_half=1,
                       background=0.0, exclude_mask=None):
    """``F(r <= norm_radius) / F(caja)`` alrededor del centro.

    Es, cifra por cifra, la correccion de apertura que aplican C2/C3: alli sale
    de ``1 / sum(PSF_normalizada * pesos_de_la_caja)`` y la PSF esta normalizada
    a 1 dentro de ``norm_radius_px``, o sea el mismo cociente evaluado sobre el
    modelo. Medirlo tambien sobre el DATO es lo que convierte la correccion de
    apertura en una cantidad verificable en vez de una consecuencia del ajuste.

    La caja se centra en el pixel redondeado, igual para dato y modelo: el
    interes es la diferencia entre los dos, no el valor absoluto al subpixel.
    """

    img = np.asarray(image, dtype=np.float64) - float(background)
    yy, xx = np.indices(img.shape, dtype=np.float64)
    cy, cx = round(float(center_yx[0])), round(float(center_yx[1]))
    ok = np.isfinite(img)
    if exclude_mask is not None:
        ok &= ~np.asarray(exclude_mask, dtype=bool)
    caja = ok & (np.abs(yy - cy) <= int(box_half)) & (np.abs(xx - cx) <= int(box_half))
    total = encircled_energy(image, center_yx, [float(norm_radius_px)],
                             background=background, exclude_mask=exclude_mask)[0]
    box = float(np.sum(np.where(caja, img, 0.0)))
    return float(total / box) if box > 0 else float("nan")


def encircled_energy_metric(image, model, center_yx, *, norm_radius_px=25.0, box_half=1,
                            image_background=0.0, model_background=0.0, exclude_mask=None,
                            radii_px=None):
    """V4 de la spec C1: energia encapsulada del MODELO contra la del DATO.

    La metrica del anillo (§3.4) mira el halo en el radio del compañero y es
    ciega al nucleo. Pero el modelo no se usa solo para el halo: C2/C3 lo usan
    para pasar de una caja de 3x3 al flujo dentro de ``norm_radius_px``, y una
    forma puede clavar el anillo con un nucleo completamente equivocado — es lo
    que hace la Moffat cuando el recorte sigma se come el nucleo del ajuste.

    Devuelve el cociente nucleo/norm de los dos, su error relativo, y la curva
    de crecimiento normalizada a ``norm_radius_px`` con su desviacion maxima.
    """

    radii = (np.asarray(radii_px, dtype=np.float64) if radii_px is not None
             else np.asarray([1.0, 2.0, 3.0, 5.0, 8.0, 12.0, 18.0, float(norm_radius_px)]))
    razon_dato = core_to_norm_ratio(image, center_yx, norm_radius_px=norm_radius_px,
                                    box_half=box_half, background=image_background,
                                    exclude_mask=exclude_mask)
    razon_modelo = core_to_norm_ratio(model, center_yx, norm_radius_px=norm_radius_px,
                                      box_half=box_half, background=model_background,
                                      exclude_mask=exclude_mask)
    ee_dato = encircled_energy(image, center_yx, radii, background=image_background,
                               exclude_mask=exclude_mask)
    ee_modelo = encircled_energy(model, center_yx, radii, background=model_background,
                                 exclude_mask=exclude_mask)
    with np.errstate(divide="ignore", invalid="ignore"):
        curva_dato = ee_dato / ee_dato[-1]
        curva_modelo = ee_modelo / ee_modelo[-1]
        error_pct = 100.0 * (razon_modelo / razon_dato - 1.0)
        curva_diff = 100.0 * np.abs(curva_modelo - curva_dato)
    return {
        "radii_px": [float(r) for r in radii],
        "core_ratio_data": float(razon_dato),
        "core_ratio_model": float(razon_modelo),
        "core_ratio_error_pct": float(error_pct),
        "growth_curve_data": [float(v) for v in curva_dato],
        "growth_curve_model": [float(v) for v in curva_modelo],
        "growth_curve_max_abs_diff_pct": float(np.nanmax(curva_diff)) if curva_diff.size else float("nan"),
    }


def smooth_parameter(wavelengths_A, values, *, max_degree=2, wave_ref_A=None, wave_scale_A=1000.0):
    wave = np.asarray(wavelengths_A, dtype=np.float64)
    vals = np.asarray(values, dtype=np.float64)
    good = np.isfinite(wave) & np.isfinite(vals)
    if int(np.count_nonzero(good)) == 0:
        raise ValueError("No finite values to smooth.")
    wave_ref = float(np.nanmedian(wave[good])) if wave_ref_A is None else float(wave_ref_A)
    x = (wave[good] - wave_ref) / float(wave_scale_A)
    y = vals[good]
    best = None
    for deg in range(0, min(int(max_degree), y.size - 1) + 1):
        coeff_high = np.polyfit(x, y, deg)
        pred = np.polyval(coeff_high, x)
        rss = float(np.nansum((y - pred) ** 2))
        k = deg + 1
        aic = y.size * math.log(max(rss / max(y.size, 1), 1e-24)) + 2 * k
        if best is None or aic < best["aic"]:
            best = {"degree": deg, "coeff_high": coeff_high, "rss": rss, "aic": aic}
    coeff_low = best["coeff_high"][::-1].astype(float).tolist()
    return {
        "degree": int(best["degree"]),
        "coefficients": coeff_low,
        "wave_ref_A": wave_ref,
        "wave_scale_A": float(wave_scale_A),
        "model": "polynomial",
    }


def eval_smoothed_parameter(spec, wavelength_A):
    x = (float(wavelength_A) - float(spec["wave_ref_A"])) / float(spec.get("wave_scale_A", 1000.0))
    coeff = np.asarray(spec["coefficients"], dtype=np.float64)
    return float(np.polynomial.polynomial.polyval(x, coeff))


def build_psf_model_document(
    wavelength_bins_A,
    fit_rows,
    *,
    form="moffat",
    norm_radius_px=25.0,
    hybrid=False,
):
    waves = np.asarray(wavelength_bins_A, dtype=np.float64)
    smoothing = {}
    for key in PSF_SHAPE_PARAMS:
        smoothing[key] = smooth_parameter(waves, [row[key] for row in fit_rows])
    return {
        "form": str(form),
        "norm_radius_px": float(norm_radius_px),
        "coefficients": smoothing,
        "hybrid": bool(hybrid),
    }


_PSFAO_PARAM_NAMES = ("r0", "C", "A", "alpha", "ratio", "theta", "beta")


@functools.lru_cache(maxsize=16384)
def _psfao_image_cached(x_key, npix, system_name, samp, norm_radius):
    """Build (and cache) the normalised Psfao image for one parameter set.

    Building the Psfao model is an FFT (~6 ms). During per-channel PSF fitting
    (C3/C4) the optimiser evaluates the SAME wavelength/params many times while
    varying only flux/position, so caching the image (keyed on the params, grid
    size, sampling and norm radius) turns hours into minutes. Returns the even
    image, its norm_radius integral, and the grid centre."""

    from maoppy.instrument import muse_nfm, muse_wfm
    from maoppy.psfmodel import Psfao

    system = muse_wfm if str(system_name).lower().endswith("wfm") else muse_nfm
    model = Psfao((npix, npix), system=system, samp=samp)
    # Clip to Psfao's physical bounds (smoothed/interpolated params can drift out
    # of range at edge/gap wavelengths).
    low, high = model.bounds
    eps = 1e-6
    x = [
        float(np.clip(
            xi,
            low[i] + eps if np.isfinite(low[i]) else -np.inf,
            high[i] - eps if np.isfinite(high[i]) else np.inf,
        ))
        for i, xi in enumerate(x_key)
    ]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        img = np.asarray(model(x), dtype=np.float64)  # peak at (npix//2, npix//2)
    c = npix // 2
    gy, gx = np.mgrid[0:npix, 0:npix]
    rr = np.hypot(gy - c, gx - c)
    total = float(np.nansum(img[rr <= float(norm_radius)]))
    if not np.isfinite(total) or total <= 0:
        raise RuntimeError("Psfao normalization within norm_radius failed.")
    return img, total, c


def _psfao_wave_bin_A(model_doc):
    """Grid ``_evaluate_psfao`` snaps lambda to, resolved FROM THE DOCUMENT.

    Order: the grid C1 declared (``psfao_wave_bin_A``) -> the spacing of the
    document's own ``param_table`` -> no snapping at all (0.0, exact lambda).
    There is deliberately no numeric default. The historic one was 50 A, half
    the width of the bins C1 actually fits, so every other channel landed
    between two bins and got its Psfao parameters by linear interpolation --
    and those parameters live in the PSD, where they are degenerate: the
    straight line between two fitted bins leaves the valley, the PSF comes out
    ~1% wrong in the halo, and the near-degenerate psffit design (PSF + PSF +
    plane) turns that into a 13% square wave in the extracted spectrum. See
    `docs/2026-08-12_handoff.md` and `apcorr_debug` sections 14-17.

    A document that carries a ``param_table`` therefore snaps to the width of
    those bins: the parameters exist there and nowhere else. Gaps (bins C1
    rejected) are whole multiples of that width, so the *smallest* spacing is
    the grid -- a median would be inflated by the gaps. A poly-only document
    has no grid to respect: the smoothed polynomial is continuous in lambda, so
    it is evaluated exactly, and if such a document ever needs the FFT cache to
    batch channels it has to declare the grid explicitly.
    """

    declared = model_doc.get("psfao_wave_bin_A")
    if declared is not None:
        wave_bin = float(declared)
        if not np.isfinite(wave_bin) or wave_bin < 0:
            raise ValueError(
                f"psfao_wave_bin_A must be finite and >= 0 (0 = no snapping), got {declared!r}.")
        return wave_bin
    table = model_doc.get("param_table") or {}
    lam = np.unique(np.asarray(table.get("lambda_A", []), dtype=np.float64))
    if lam.size >= 2:
        spacing = float(np.min(np.diff(lam)))
        if np.isfinite(spacing) and spacing > 0:
            return spacing
    return 0.0


def _psfao_grid_npix(model_doc, dy, dx, norm_radius):
    """Tamaño de la rejilla en la que se construye la PSF, INDEPENDIENTE de la posición.

    Dos regímenes, los dos dando un ``npix`` que no depende de dónde esté la
    fuente, para que la caché se reutilice entre estrella, compañero y controles
    a un mismo λ:

    * quien pide la apcorr / la curva de crecimiento manda desplazamientos
      dentro de ``norm_radius`` → rejilla pequeña y rápida;
    * ``psffit`` evalúa sobre la imagen entera; la PSF sólo hace falta sobre la
      región conjunta del ajuste (separación + radio), así que se usa un alcance
      fijo (``psfao_grid_reach_px``, 140 px). 140 px es donde converge el flujo
      del compañero en esta geometría (100 submuestrea el halo AO, ~3 % alto;
      140/180/250 coinciden al 0.2 %). Más allá de la rejilla se muestrea 0, que
      es despreciable.
    """

    max_off = 0.0
    if dy.size:
        max_off = max(float(np.nanmax(np.abs(dy))), float(np.nanmax(np.abs(dx))))
    if max_off <= norm_radius:
        reach = norm_radius
    else:
        reach = max(norm_radius, float(model_doc.get("psfao_grid_reach_px", 140.0)))
    return 2 * (int(np.ceil(reach)) + 2)  # par, la fuente en npix//2


def _psfao_params_at(model_doc, wavelength_A, names=_PSFAO_PARAM_NAMES):
    """The seven Psfao parameters of a document at one wavelength.

    Split out of ``_evaluate_psfao`` because the mixture form needs exactly the
    same resolution rule for each of its components: interpolate the per-bin
    ``param_table`` when there is one (the Psfao PSD parameters are degenerate,
    so smoothing them independently and rebuilding corrupts the PSF), and only
    fall back to ``smoothed_poly`` when the document carries no table.
    """

    table = model_doc.get("param_table")
    if table:
        lam = np.asarray(table["lambda_A"], dtype=np.float64)
        w = float(np.clip(float(wavelength_A), lam.min(), lam.max()))
        return [float(np.interp(w, lam, np.asarray(table[name], dtype=np.float64))) for name in names]
    poly = model_doc.get("smoothed_poly") or {}
    return [
        float(np.polyval(np.asarray(poly[name], dtype=np.float64), float(wavelength_A)))
        for name in names
    ]


def _evaluate_psfao(model_doc, wavelength_A, dy, dx):
    """Evaluate a physical AO PSF (maoppy Psfao) on the (dy, dx) offsets,
    normalised so it sums to 1 within ``norm_radius_px``. Mirrors the Moffat
    branch's contract so aperture-correction/growth-curve/optimal/psffit
    consumers are agnostic to the PSF form. Params come from the C1 per-bin
    ``param_table`` (interpolated) or ``smoothed_poly``. The expensive FFT build
    is cached in ``_psfao_image_cached``; here we only re-sample it."""

    from maoppy.instrument import muse_nfm
    from scipy.ndimage import map_coordinates

    dy = np.asarray(dy, dtype=np.float64)
    dx = np.asarray(dx, dtype=np.float64)
    if dy.shape != dx.shape:
        raise ValueError("psfao evaluation expects matching dy/dx offset arrays.")
    # FWHM perturbation (C3/E4 sensitivity tests). The Psfao parameters live in
    # the PSD, so there is no coefficient to multiply the way the Moffat branch
    # does: the geometric equivalent is to sample the built PSF on offsets
    # divided by the scale (a dilation by `scale`), with 1/scale**2 conserving
    # the integral. See `scaled_psf_model`.
    fwhm_scale = float(model_doc.get("psf_fwhm_scale", 1.0))
    if not np.isfinite(fwhm_scale) or fwhm_scale <= 0:
        raise ValueError(f"psf_fwhm_scale must be finite and > 0, got {fwhm_scale!r}.")
    if fwhm_scale != 1.0:
        dy = dy / fwhm_scale
        dx = dx / fwhm_scale
    names = model_doc.get("param_names", _PSFAO_PARAM_NAMES)
    # Snap the wavelength to the grid the parameters were fitted on before
    # building the PSF: consecutive channels then share ONE cached FFT build
    # (3681 builds -> ~45), and -- the reason the grid must not be finer than
    # C1's bins -- no channel gets its degenerate PSD parameters from a point
    # halfway between two fits. `_psfao_wave_bin_A` resolves it from the
    # document; 0 means evaluate at the exact wavelength. Sampling always uses
    # the exact per-call offsets, so per-channel positions/flux stay exact.
    wave_bin = _psfao_wave_bin_A(model_doc)
    w_eff = round(float(wavelength_A) / wave_bin) * wave_bin if wave_bin > 0 else float(wavelength_A)
    x = _psfao_params_at(model_doc, w_eff, names)
    system_name = "muse_wfm" if str(model_doc.get("system", "muse_nfm")).lower().endswith("wfm") else "muse_nfm"
    samp = float(muse_nfm.samp(w_eff * 1e-10))
    norm_radius = float(model_doc.get("norm_radius_px", 25.0))
    npix = _psfao_grid_npix(model_doc, dy, dx, norm_radius)
    img, total, c = _psfao_image_cached(
        tuple(round(v, 10) for v in x), npix, system_name, round(samp, 10), round(norm_radius, 6)
    )
    rows = (c + dy).ravel()
    cols = (c + dx).ravel()
    vals = map_coordinates(img, [rows, cols], order=1, mode="constant", cval=0.0).reshape(dy.shape)
    return vals / (total * fwhm_scale ** 2)


MIXTURE_FORM = "mixture"


def _mixture_component_key(component, wavelength_A):
    """La componente, reducida a algo hasheable: (forma, parámetros a ese λ).

    La clave es lo que hace cacheable la imagen de la mezcla. Es el mismo truco
    que ``_psfao_image_cached`` usa con los siete parámetros, extendido a N
    componentes: dos llamadas al mismo λ producen la misma clave y la suma
    pesada no se vuelve a construir.
    """

    model = component["model"]
    form = str(model.get("form", "moffat")).lower()
    if form == "psfao":
        names = tuple(model.get("param_names", _PSFAO_PARAM_NAMES))
        values = _psfao_params_at(model, wavelength_A, names)
        return ("psfao", tuple(round(float(v), 10) for v in values))
    if form == "moffat":
        params = {
            key: float(eval_smoothed_parameter(model["coefficients"][key], wavelength_A))
            for key in PSF_SHAPE_PARAMS
        }
        return ("moffat", tuple(round(params[key], 10) for key in PSF_SHAPE_PARAMS))
    raise ValueError(
        f"Mixture component has form={form!r}; expected 'moffat' or 'psfao'."
    )


def _mixture_component_weight(component, wavelength_A):
    """Peso de una componente a ese λ: peso del combinado × flujo de la exposición.

    El combinado es una media **pesada de brillo**, así que la PSF del combinado
    es la media de las PSF pesada por `w_i · F_i(λ)`, no por `w_i` sola: la
    transmisión y la masa de aire cambian entre exposiciones y el brillo de la
    estrella con ellas. `F_i` es el flujo dentro de `norm_radius_px`, que es la
    región en la que cada componente está normalizada, y lo mide C1 cuando
    construye el documento — aquí no se re-deriva.
    """

    weight = float(component.get("weight", 1.0))
    flux = component.get("flux_norm") or {}
    lam = np.asarray(flux.get("lambda_A", ()), dtype=np.float64)
    values = np.asarray(flux.get("value", ()), dtype=np.float64)
    if lam.size == 0 or values.size != lam.size:
        raise ValueError(
            f"Mixture component {component.get('exposure_id', '?')!r} has no usable "
            "`flux_norm`; the mixture weight is w_i*F_i(lambda) and F_i cannot be guessed."
        )
    ok = np.isfinite(lam) & np.isfinite(values) & (values > 0)
    if not ok.any():
        raise ValueError(
            f"Mixture component {component.get('exposure_id', '?')!r} has no finite "
            "positive `flux_norm` values."
        )
    lam_ok, values_ok = lam[ok], values[ok]
    w = float(np.clip(float(wavelength_A), lam_ok.min(), lam_ok.max()))
    return weight * float(np.interp(w, lam_ok, values_ok))


@functools.lru_cache(maxsize=256)
def _mixture_image_cached(keys, weights, npix, system_name, samp, norm_radius):
    """Imagen de la mezcla, normalizada a 1 dentro de ``norm_radius``.

    Se cachea la SUMA, no sólo las componentes: sin esto, cada evaluación de la
    PSF costaría N veces una evaluación normal (con 29 exposiciones, el ajuste
    por canal de C4 pasaría de minutos a horas). Con la suma cacheada, una
    mezcla cuesta lo mismo que una PSF suelta salvo la primera vez a cada λ, y
    esa primera vez son N construcciones (~0.35 s cada una en la rejilla grande
    de psffit, ~5 ms en la pequeña de la apcorr).

    Las componentes se construyen **saltándose** ``_psfao_image_cached``: al
    estar la suma cacheada sólo hacen falta una vez por λ, y dejarlas en esa
    caché guardaría 29×43 imágenes de 284² (~800 MB) que nadie volvería a
    mirar. Lo que se retiene es la mezcla, que son ~28 MB.
    """

    build_component = getattr(_psfao_image_cached, "__wrapped__", _psfao_image_cached)
    c = npix // 2
    gy, gx = np.mgrid[0:npix, 0:npix]
    dy = gy - c
    dx = gx - c
    stack = np.zeros((npix, npix), dtype=np.float64)
    for (kind, params), weight in zip(keys, weights):
        if kind == "psfao":
            img, total, _ = build_component(params, npix, system_name, samp, norm_radius)
            unit = img / total
        else:
            unit = normalized_moffat_psf(
                dy, dx, dict(zip(PSF_SHAPE_PARAMS, params)), norm_radius_px=norm_radius
            )
        stack += float(weight) * unit
    inside = np.hypot(dy, dx) <= float(norm_radius)
    total = float(np.nansum(stack[inside]))
    if not np.isfinite(total) or total <= 0:
        raise RuntimeError("Mixture normalization within norm_radius failed.")
    return stack / total, c


def _evaluate_mixture(model_doc, wavelength_A, dy, dx):
    """Evalúa una PSF de mezcla: la media pesada de las PSF por observación.

    Existe porque **la mezcla de N PSF no es una PSF**: el combinado suma
    exposiciones con seeing y calidad de AO distintas, y ninguna Psfao ni
    Moffat puede describir esa suma. Ajustar una forma analítica al combinado es
    lo que dejaba la razón núcleo/halo —y con ella la corrección de apertura—
    sistemáticamente mal. Aquí cada exposición aporta su propio modelo, ajustado
    a su propio cubo, y la suma se hace con los pesos del combinado.

    El contrato es el mismo que el de las otras dos formas: devuelve la PSF
    normalizada a 1 dentro de ``norm_radius_px``, así que quien la consuma
    (apcorr, curva de crecimiento, optimal, psffit, inyección) no necesita
    saber que es una mezcla.
    """

    from maoppy.instrument import muse_nfm
    from scipy.ndimage import map_coordinates

    dy = np.asarray(dy, dtype=np.float64)
    dx = np.asarray(dx, dtype=np.float64)
    if dy.shape != dx.shape:
        raise ValueError("mixture evaluation expects matching dy/dx offset arrays.")
    components = model_doc.get("components") or ()
    if not components:
        raise ValueError("Mixture psf_model has no components.")

    # Misma mecánica geométrica que la rama psfao (ver `scaled_psf_model`): la
    # escala de FWHM se aplica a la mezcla entera, no componente a componente,
    # porque una dilatación conmuta con la suma pesada.
    fwhm_scale = float(model_doc.get("psf_fwhm_scale", 1.0))
    if not np.isfinite(fwhm_scale) or fwhm_scale <= 0:
        raise ValueError(f"psf_fwhm_scale must be finite and > 0, got {fwhm_scale!r}.")
    if fwhm_scale != 1.0:
        dy = dy / fwhm_scale
        dx = dx / fwhm_scale

    wave_bin = _psfao_wave_bin_A(model_doc)
    w_eff = round(float(wavelength_A) / wave_bin) * wave_bin if wave_bin > 0 else float(wavelength_A)
    keys = tuple(_mixture_component_key(component, w_eff) for component in components)
    raw = np.asarray(
        [_mixture_component_weight(component, w_eff) for component in components],
        dtype=np.float64,
    )
    total_weight = float(np.nansum(raw))
    if not np.isfinite(total_weight) or total_weight <= 0:
        raise RuntimeError(f"Mixture weights are not usable at {wavelength_A} A.")
    # Normalizados antes de redondear: así la clave de la caché no depende de la
    # escala de flujo absoluta, que cambia con λ aunque la mezcla sea la misma.
    weights = tuple(round(float(v / total_weight), 9) for v in raw)

    system_name = (
        "muse_wfm"
        if str(model_doc.get("system", "muse_nfm")).lower().endswith("wfm")
        else "muse_nfm"
    )
    samp = float(muse_nfm.samp(w_eff * 1e-10))
    norm_radius = float(model_doc.get("norm_radius_px", 25.0))
    npix = _psfao_grid_npix(model_doc, dy, dx, norm_radius)
    img, c = _mixture_image_cached(
        keys, weights, npix, system_name, round(samp, 10), round(norm_radius, 6)
    )
    rows = (c + dy).ravel()
    cols = (c + dx).ravel()
    vals = map_coordinates(img, [rows, cols], order=1, mode="constant", cval=0.0).reshape(dy.shape)
    return vals / (fwhm_scale ** 2)


def build_mixture_model_document(components, *, norm_radius_px, system, wave_bin_A=None, **extra):
    """Documento `form="mixture"` a partir de los modelos por observación.

    ``components`` es una secuencia de diccionarios con ``exposure_id``,
    ``weight`` (el del plan del combinado), ``flux_norm`` (``lambda_A`` y
    ``value``: el flujo de esa exposición dentro de ``norm_radius_px``) y
    ``model`` (el documento de PSF ajustado a esa exposición).

    Se valida lo que rompería la mezcla en silencio: que todas las componentes
    compartan ``norm_radius_px`` —si no, la suma no está normalizada en la misma
    región y el cociente núcleo/total deja de significar nada— y que ninguna
    venga sin flujo.
    """

    components = [dict(component) for component in components]
    if not components:
        raise ValueError("A mixture needs at least one component.")
    for component in components:
        model = component.get("model")
        if not isinstance(model, dict):
            raise ValueError(
                f"Mixture component {component.get('exposure_id', '?')!r} carries no model document."
            )
        component_radius = float(model.get("norm_radius_px", norm_radius_px))
        if not np.isclose(component_radius, float(norm_radius_px)):
            raise ValueError(
                f"Mixture component {component.get('exposure_id', '?')!r} is normalised within "
                f"{component_radius} px but the mixture declares {norm_radius_px} px."
            )
        # Levanta si el flujo no está o no sirve: la validación vive aquí y no en
        # la evaluación, para que el documento nazca ya utilizable.
        lam = np.asarray(
            (component.get("flux_norm") or {}).get("lambda_A", ()), dtype=np.float64
        )
        _mixture_component_weight(component, float(lam[0]) if lam.size else float("nan"))
    document = {
        "form": MIXTURE_FORM,
        "system": str(system),
        "norm_radius_px": float(norm_radius_px),
        "n_components": len(components),
        "components": components,
    }
    if wave_bin_A is not None:
        document["psfao_wave_bin_A"] = float(wave_bin_A)
    document.update(extra)
    return document


def scaled_psf_model(model_doc, fwhm_scale):
    """Return a copy of the PSF model with its spatial FWHM scaled by ``fwhm_scale``.

    The single implementation behind the C3 (`psf_sensitivity`) and E4
    (`psf_perturbation_pct`) robustness tests, which perturb the PSF width by
    +-10% and measure how much the recovered flux moves.

    The forms need different mechanics:

    * **moffat** -- multiply the ``fwhm_maj``/``fwhm_min`` polynomial
      coefficients, as before (frozen behaviour).
    * **psfao** -- the maoppy parameters (``r0, C, A, alpha, ratio, theta,
      beta``) describe the *power spectral density*, not an image-plane width,
      so no coefficient corresponds to the FWHM. The scale is recorded as
      ``psf_fwhm_scale`` and applied geometrically when the PSF is evaluated
      (see ``_evaluate_psfao``).
    * **mixture** -- same geometric route as psfao, applied to the mixture as a
      whole. Scaling every component separately and summing gives the same
      thing (a dilation commutes with a weighted sum), so the flag lives on the
      mixture document and the components are left untouched.

    Raises on any other form. It used to return the document untouched when it
    found no ``fwhm_maj``/``fwhm_min`` to scale, which with a psfao model (the
    one C1 selects whenever maoppy is available) silently turned both
    robustness tests into no-ops: C3 reported a 0.0% sensitivity and E4's
    +-10% variants came out bit-for-bit identical.
    """

    scale = float(fwhm_scale)
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError(f"fwhm_scale must be finite and > 0, got {fwhm_scale!r}.")
    if np.isclose(scale, 1.0):
        return model_doc

    form = str(model_doc.get("form", "moffat")).lower()
    model = deepcopy(model_doc)
    if form in ("psfao", MIXTURE_FORM):
        model["psf_fwhm_scale"] = scale * float(model_doc.get("psf_fwhm_scale", 1.0))
        return model
    if form != "moffat":
        raise ValueError(
            f"Cannot scale the FWHM of psf_model form={form!r}; expected 'moffat', "
            "'psfao' or 'mixture'."
        )
    scaled_any = False
    for key in ("fwhm_maj", "fwhm_min"):
        if key in model.get("coefficients", {}):
            coeff = list(model["coefficients"][key].get("coefficients", []))
            model["coefficients"][key]["coefficients"] = [float(value) * scale for value in coeff]
            scaled_any = True
    if not scaled_any:
        raise ValueError(
            "Moffat psf_model has neither 'fwhm_maj' nor 'fwhm_min' coefficients to scale; "
            "refusing to return an unperturbed model."
        )
    return model


def evaluate_psf_model(model_doc, wavelength_A, dy, dx):
    form = str(model_doc.get("form", "moffat")).lower()
    if form == "psfao":
        return _evaluate_psfao(model_doc, wavelength_A, dy, dx)
    if form == MIXTURE_FORM:
        return _evaluate_mixture(model_doc, wavelength_A, dy, dx)
    if form != "moffat":
        raise ValueError(
            f"Unsupported psf_model form={form!r}; expected 'moffat', 'psfao' or 'mixture'."
        )
    params = {
        key: eval_smoothed_parameter(model_doc["coefficients"][key], wavelength_A)
        for key in PSF_SHAPE_PARAMS
    }
    return normalized_moffat_psf(
        dy,
        dx,
        params,
        norm_radius_px=float(model_doc.get("norm_radius_px", 25.0)),
    )


def psf_roundtrip_error(model_doc, wavelengths_A):
    yy, xx, mask = fixed_radius_grid(float(model_doc.get("norm_radius_px", 25.0)))
    errors = []
    for wave in wavelengths_A:
        vals = evaluate_psf_model(model_doc, float(wave), yy, xx)
        errors.append(abs(float(np.nansum(vals[mask])) - 1.0))
    return float(np.nanmax(errors)) if errors else np.nan


def radial_hybrid_profile(residual, center_yx, *, mask=None, bin_width_px=1.0, smoothing_scale_px=4.0):
    resid = np.asarray(residual, dtype=np.float64)
    yy, xx = np.indices(resid.shape, dtype=np.float64)
    rr = np.sqrt((yy - float(center_yx[0])) ** 2 + (xx - float(center_yx[1])) ** 2)
    valid = np.isfinite(resid)
    if mask is not None:
        valid &= ~np.asarray(mask, dtype=bool)
    bins = np.floor(rr / float(bin_width_px)).astype(int)
    nbin = int(np.nanmax(bins)) + 1
    profile = np.full(nbin, np.nan, dtype=np.float64)
    radii = (np.arange(nbin, dtype=np.float64) + 0.5) * float(bin_width_px)
    for b in range(nbin):
        pix = valid & (bins == b)
        if np.count_nonzero(pix) >= 3:
            profile[b] = np.nanmedian(resid[pix])
    finite = np.isfinite(profile)
    if np.count_nonzero(finite) >= 2:
        profile[~finite] = np.interp(radii[~finite], radii[finite], profile[finite])
    else:
        profile[~finite] = 0.0
    sigma_bins = max(float(smoothing_scale_px) / float(bin_width_px), 0.0)
    if sigma_bins > 0:
        profile = gaussian_filter1d(profile, sigma=sigma_bins, mode="nearest")
    return radii, profile


def evaluate_radial_profile(shape, center_yx, radii, profile):
    yy, xx = np.indices(shape, dtype=np.float64)
    rr = np.sqrt((yy - float(center_yx[0])) ** 2 + (xx - float(center_yx[1])) ** 2)
    return np.interp(rr.ravel(), np.asarray(radii), np.asarray(profile), left=profile[0], right=profile[-1]).reshape(shape)


__all__ = [
    "MIXTURE_FORM",
    "MoffatFit",
    "PSF_SHAPE_PARAMS",
    "build_mixture_model_document",
    "build_psf_model_document",
    "companion_ring_metric",
    "corner_background",
    "evaluate_moffat_fit",
    "evaluate_psf_model",
    "evaluate_radial_profile",
    "fit_moffat_image",
    "moffat_elliptical_profile",
    "moffat_image",
    "moffat_norm",
    "normalized_moffat_psf",
    "psf_roundtrip_error",
    "radial_hybrid_profile",
    "scaled_psf_model",
    "source_mask",
    "smooth_parameter",
]
