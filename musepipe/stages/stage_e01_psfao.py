"""C1 chromatic PSF via the physical AO model (maoppy Psfao).

The Moffat-only path of ``stage_e01_psf`` leaves ~90% residual at the companion
radius because an elliptical Moffat cannot reproduce the AO halo. This module
fits the physical AO PSF (Fetick et al. 2019, ``maoppy.psfmodel.Psfao``) per
wavelength bin with the companion masked, which brings the companion-ring
residual to ~4-5% (spec C1 target). Target-specific values come only from
config / the B3 QC; no hard-coded source coordinates live here.

Products (in the run stage dir):
  stage_e01_psfao_params.csv   per-bin: lambda, params, amp, ring_residual_pct
  psf_model.json               Psfao form, smoothed param polynomials, norm_radius
  stage_e01_qc.json            ring metric, smoothing, normalization (overwrites
                               the Moffat QC when this path is chosen)
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
from astropy.io import fits

from ..config import load_run_config

PSFAO_PARAM_NAMES = ("r0", "C", "A", "alpha", "ratio", "theta", "beta")
DEFAULT_X0 = [0.15, 1e-4, 1.0, 0.05, 1.0, 0.0, 1.6]

#: Como se pesan los pixeles del ajuste por bin. `stat` es el historico y sigue
#: siendo el default: NINGUN run cambia si no lo declara.
PSFAO_WEIGHTINGS = ("stat", "relative", "halo")
PSFAO_DEFAULT_WEIGHTING = "stat"
#: Radio del nucleo que `halo` deja fuera del ajuste.
PSFAO_HALO_CORE_PX = 12.0
#: Tope del termino relativo: razon maxima entre el peso mayor y el menor que
#: puede introducir la imagen. Sin tope (`None`) el nucleo deja de contar y la
#: correccion de apertura se dispara -- medido y revertido el 2026-08-14, ver el
#: barrido en la spec C1 §5.2. 5 es el codo de esa curva.
PSFAO_DEFAULT_WEIGHT_CAP = 5.0


def psfao_fit_weights(image, var, radius_px, weighting=PSFAO_DEFAULT_WEIGHTING,
                      core_px=PSFAO_HALO_CORE_PX, cap=PSFAO_DEFAULT_WEIGHT_CAP):
    """Los pesos del ajuste por bin, segun que parte de la imagen deba mandar.

    Con `stat` —1/STAT, lo que se ha usado siempre— el chi2 lo domina el NUCLEO
    por varios ordenes de magnitud, y el halo no llega a tener voz. Eso no es un
    detalle de implementacion: medido en `C1_chromatic_psf_debug` §13.f sobre
    ROXs 12 b, el modelo reproduce asi el 28 % del cromatismo del halo y deja el
    residuo de anillo en 32 %, mientras que con `relative` baja a 4.65 % —el
    objetivo de la spec C1— y sube al 45 %, y con `halo` (nucleo fuera) llega al
    101 %. El modelo siempre pudo; era el peso.

    - ``stat``     1/STAT. El historico, y el default.
    - ``relative`` 1/STAT dividido por ``clip(|imagen|, piso, techo)^2``: error
      RELATIVO **con tope**, o sea cada anillo cuenta lo mismo aunque sea mil
      veces mas debil, pero la imagen no puede introducir una razon de pesos
      mayor que ``cap``. El suelo en la mediana evita que el fondo (donde
      |imagen| es diminuto) se lleve todo el peso, y el techo evita lo
      contrario: que el NUCLEO deje de contar. Sin tope (`cap=None`) el modelo
      se ensancha y la correccion de apertura se va x2.7 -- probado en la cadena
      entera el 2026-08-14 y revertido.
    - ``halo``     1/STAT con el nucleo (``r < core_px``) a cero. Es el mas
      extremo y el que mas cromatismo recupera, pero deja el nivel del anillo
      bajo y empuja `beta` contra su cota: sirve de diagnostico, no de default.
    """

    if weighting not in PSFAO_WEIGHTINGS:
        raise ValueError(
            f"psf_fit_weighting desconocido: {weighting!r}; los validos son {PSFAO_WEIGHTINGS}.")
    w = 1.0 / np.clip(var, 1e-6, None)
    if weighting == "relative":
        img = np.abs(np.asarray(image, dtype=np.float64))
        piso = max(float(np.nanpercentile(img, 50)), 1e-6)
        if cap is None:
            techo = np.inf
        else:
            cap = float(cap)
            if not np.isfinite(cap) or cap < 1.0:
                raise ValueError(f"psf_fit_weight_cap debe ser >= 1 o None, y es {cap!r}.")
            # El peso va como 1/x^2, asi que un rango sqrt(cap) en la imagen es
            # un rango `cap` en el peso: el knob se lee como «cuantas veces mas
            # puede pesar un pixel que otro».
            techo = piso * np.sqrt(cap)
        return w / np.clip(img, piso, techo) ** 2
    if weighting == "halo":
        return np.where(np.asarray(radius_px) >= float(core_px), w, 0.0)
    return w


def _bad_windows(cfg):
    if cfg.get("drop_wave_min_A") is not None and cfg.get("drop_wave_max_A") is not None:
        return [[float(cfg["drop_wave_min_A"]), float(cfg["drop_wave_max_A"])]]
    return [[5780.0, 6050.0]]


def make_bins(wave, bin_A, bad_windows, min_channels=3):
    lo, hi = float(wave.min()), float(wave.max())
    edges = np.arange(lo, hi + bin_A, bin_A)
    bins = []
    for a, b in zip(edges[:-1], edges[1:]):
        mid = 0.5 * (a + b)
        if any(w0 <= mid <= w1 for w0, w1 in bad_windows):
            continue
        sel = (wave >= a) & (wave < b)
        if sel.sum() >= min_channels:
            bins.append((a, b, mid, sel))
    return bins


def fit_bin(image, var, samp, system, companion_yx, mask_radius, fit_radius, x0,
            field_yx=None, weighting=PSFAO_DEFAULT_WEIGHTING,
            weight_cap=PSFAO_DEFAULT_WEIGHT_CAP):
    """Ajuste Psfao de un bin.

    `field_yx` enmascara una fuente de campo igual que hace la rama Moffat. Sin
    esto las dos formas se ajustaban con MASCARAS DISTINTAS —Moffat con
    companero + fuente de campo, psfao solo con el companero— y
    `model_comparison` dejaba de comparar peras con peras: el sesgo iba siempre
    contra psfao, porque era la unica que se comia el contaminante. Se vio en
    ROXs 42B b, donde enmascarar ROXs 42B cc1 mejoro Moffat (8.38 -> 6.88%) y
    dejo psfao intacta (9.80 -> 10.69%).

    `weighting` decide QUE PARTE DE LA IMAGEN manda en el ajuste, y es lo que
    separa un modelo que reproduce el halo de uno que no. Ver
    `psfao_fit_weights`.
    """

    from maoppy.psfmodel import Psfao
    from maoppy.psffit import psffit

    ny, nx = image.shape
    cy, cx = ny // 2, nx // 2
    yy, xx = np.mgrid[0:ny, 0:nx]
    r = np.hypot(yy - cy, xx - cx)
    comp = np.hypot(yy - companion_yx[0], xx - companion_yx[1])
    mask = np.isfinite(image) & (comp > mask_radius) & (r < fit_radius)
    if field_yx is not None:
        mask &= np.hypot(yy - float(field_yx[0]), xx - float(field_yx[1])) > mask_radius
    weights = np.where(mask, psfao_fit_weights(image, var, r, weighting, cap=weight_cap), 0.0)
    imgf = np.where(np.isfinite(image), image, 0.0)
    model = Psfao((ny, nx), system=system, samp=float(samp))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = psffit(imgf, model, x0, weights=weights, flux_bck=(True, True), max_nfev=400)
    amp, bck = res.flux_bck
    recon = amp * model(res.x, dx=res.dxdy[0], dy=res.dxdy[1]) + bck
    ring = _ring_residual(image, recon, companion_yx, mask_radius)
    unchanged = bool(
        np.allclose(np.asarray(res.x, dtype=float), np.asarray(x0, dtype=float), rtol=0.0, atol=1e-8)
        and np.allclose(np.asarray(res.dxdy, dtype=float), 0.0, rtol=0.0, atol=1e-8)
    )
    # An exact solution may legitimately equal x0. Treat it as a stall only when
    # the optimizer also stopped immediately without exploring the parameter space.
    stalled = bool(unchanged and int(res.nfev) <= 2)
    optimizer = {
        "success": bool(res.success),
        "status": int(res.status),
        "message": str(res.message),
        "nfev": int(res.nfev),
        "cost": float(res.cost),
        "stalled_at_initial": stalled,
    }
    errors = _psfao_param_errors(res)
    return (list(map(float, res.x)), float(amp), float(bck), tuple(map(float, res.dxdy)),
            ring, recon, optimizer, errors)


def fit_bin_binary(image, var, samp, system, companion_yx, mask_radius, fit_radius, x0,
                   companion_offset_yx, field_yx=None,
                   weighting=PSFAO_DEFAULT_WEIGHTING,
                   weight_cap=PSFAO_DEFAULT_WEIGHT_CAP, ratio0=0.1):
    """Como `fit_bin`, pero con DOS componentes ligadas.

    Existe porque ROXs 42B es una binaria cercana no resuelta (rho = 51 mas,
    PA = 148 deg, Keck/NIRC2 2022.621 -> 2.006 px) y ajustar UNA PSF a DOS
    estrellas la mide mas ancha de lo que es, sesgando la `apcorr` un +6.9 %
    (`docs/2026-08-30_binaria_42b_sesga_la_apcorr.md`).

    **No son dos fuentes libres.** Comparten los siete parametros de la PSF
    —misma atmosfera, mismo instrumento— y su separacion la fija la astrometria
    publicada, asi que el ajuste gana **exactamente uno**: la razon de flujos.

    **Reproduce la funcion de coste de `maoppy.psffit` con una linea cambiada.**
    Su coste es `sqrt(w) * (amp*mm + bck - psf)` con `mm = model(x, dx, dy)` y
    `amp, bck` resueltos linealmente por `lsq_flux_bck`; aqui `mm` pasa a ser
    `model(...) + f*model(...desplazado...)` y se reutilizan `lsq_flux_bck`, las
    cotas de `model.bounds` y el mismo `least_squares`. No se reimplanta el
    ajuste: mismo modelo, mismos pesos, mismo solve lineal, mismas cotas.

    Con `f` forzado a 0 esto es, termino a termino, `maoppy.psffit` — y de eso
    hay test.

    **La convencion de ejes de maoppy esta invertida respecto a la del repo**:
    `dx` mueve el eje 1 (columnas, la `x` de `[y, x]`) y `dy` el eje 0 (filas,
    la `y`). `companion_offset_yx` llega como `(dy, dx)` en la convencion del
    repo, asi que se cruza al pasarlo. Equivocarlo pondria la secundaria en el
    sitio equivocado sin que nada fallara.
    """

    from maoppy.psffit import lsq_flux_bck
    from maoppy.psfmodel import Psfao
    from scipy.optimize import least_squares

    ny, nx = image.shape
    cy, cx = ny // 2, nx // 2
    yy, xx = np.mgrid[0:ny, 0:nx]
    r = np.hypot(yy - cy, xx - cx)
    comp = np.hypot(yy - companion_yx[0], xx - companion_yx[1])
    mask = np.isfinite(image) & (comp > mask_radius) & (r < fit_radius)
    if field_yx is not None:
        mask &= np.hypot(yy - float(field_yx[0]), xx - float(field_yx[1])) > mask_radius
    weights = np.where(mask, psfao_fit_weights(image, var, r, weighting, cap=weight_cap), 0.0)
    imgf = np.where(np.isfinite(image), image, 0.0)
    model = Psfao((ny, nx), system=system, samp=float(samp))

    off_y, off_x = (float(companion_offset_yx[0]), float(companion_offset_yx[1]))
    n_psd = len(x0)
    sqw = np.sqrt(weights)

    def _escena(y):
        """La imagen que el DATO contiene: las dos componentes."""
        x, dx, dy, f = y[:n_psd], float(y[n_psd]), float(y[n_psd + 1]), float(y[n_psd + 2])
        mm = model(x, dx=dx, dy=dy)
        if f > 0:
            # off_x al eje 1 y off_y al eje 0: ver el docstring.
            mm = mm + f * model(x, dx=dx + off_x, dy=dy + off_y)
        return x, (dx, dy), f, mm

    def _coste(y):
        _x, _dxdy, _f, mm = _escena(y)
        amp, bck = lsq_flux_bck(mm, imgf, weights, background=True)
        return 0.5 * np.reshape(sqw * (amp * mm + bck - imgf), imgf.size)

    b_low = np.concatenate((np.asarray(model.bounds[0], dtype=float), [-np.inf, -np.inf, 0.0]))
    b_up = np.concatenate((np.asarray(model.bounds[1], dtype=float), [np.inf, np.inf, 1.0]))
    y0 = np.concatenate((np.asarray(x0, dtype=float), [0.0, 0.0, float(ratio0)]))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = least_squares(_coste, y0, bounds=(b_low, b_up), max_nfev=400)

    x, dxdy, ratio, mm = _escena(res.x)
    amp, bck = lsq_flux_bck(mm, imgf, weights, background=True)
    # `recon` es la ESCENA -las dos estrellas-, que es contra lo que se mide el
    # residuo del anillo, porque es lo que el dato tiene. La PSF de una fuente
    # puntual -lo que se publica- se reconstruye aparte, sin la segunda.
    recon = amp * mm + bck
    recon_psf = amp * model(x, dx=dxdy[0], dy=dxdy[1]) + bck
    ring = _ring_residual(image, recon, companion_yx, mask_radius)

    unchanged = bool(
        np.allclose(np.asarray(x, dtype=float), np.asarray(x0, dtype=float), rtol=0.0, atol=1e-8)
        and np.allclose(np.asarray(dxdy, dtype=float), 0.0, rtol=0.0, atol=1e-8)
    )
    optimizer = {
        "success": bool(res.success),
        "status": int(res.status),
        "message": str(res.message),
        "nfev": int(res.nfev),
        "cost": float(res.cost),
        "stalled_at_initial": bool(unchanged and int(res.nfev) <= 2),
    }
    return (list(map(float, x)), float(amp), float(bck), tuple(map(float, dxdy)),
            ring, recon, optimizer, _errores_de_jacobiano(res, n_psd), float(ratio), recon_psf)


def _errores_de_jacobiano(res, n_psd):
    """`1/sqrt(diag(JtJ))`, la misma convencion que usa `maoppy.psffit`.

    Se recalcula aqui porque el ajuste de dos componentes no pasa por `psffit`,
    que es quien rellena `res.x_std`. Un parametro pegado a su cota tiene
    gradiente nulo y sale infinito: eso no es incertidumbre infinita, es «no
    medido», y se devuelve NaN igual que en `_psfao_param_errors`.
    """

    nombres = [f"{n}_err" for n in PSFAO_PARAM_NAMES] + ["dx_err", "dy_err", "flux_ratio_err"]
    salida = {name: float("nan") for name in nombres}
    jac = getattr(res, "jac", None)
    if jac is None or not np.size(jac):
        return salida
    try:
        diag = np.diag(np.asarray(jac, dtype=float).T @ np.asarray(jac, dtype=float))
        with np.errstate(divide="ignore", invalid="ignore"):
            std = 1.0 / np.sqrt(diag)
    except Exception:  # pragma: no cover - defensive
        return salida
    for i, name in enumerate(nombres):
        if i < std.size and np.isfinite(std[i]) and std[i] > 0:
            salida[name] = float(std[i])
    return salida


def _psfao_param_errors(res):
    """Las incertidumbres formales del ajuste, `{param: sigma}`.

    `psffit` ya las calcula —`1/sqrt(diag(JtJ))`, al final de `maoppy/psffit.py`—
    y las deja en `res.x_std` / `res.dxdy_std`, pero hasta ahora se tiraban: el
    CSV de Moffat traia sus siete columnas `*_err` y el de psfao ninguna, asi que
    no habia forma de saber si un parametro que salta entre bins vecinos esta
    medido o no.

    OJO con lo que significan. Los pesos del ajuste son `1/STAT`, y
    `docs/noise_model.md` tiene medido que STAT subestima el ruido, asi que esto
    es una **cota inferior formal**, no la incertidumbre real; ademas Moffat usa
    otra convencion (`pinv(JtJ) * sigma^2` con sigma empirica del residuo, en
    `psf.py:fit_moffat_image`). Sirven para ver la estabilidad DENTRO de cada
    forma, no para comparar tamanos ENTRE formas.

    Un parametro pegado a su limite fisico tiene gradiente nulo y su entrada de
    la diagonal sale 0 -> `1/sqrt(0)` es infinito. Eso no es una incertidumbre
    infinita, es «no medido»: se devuelve NaN, que es lo que los graficos y las
    estadisticas robustas saben ignorar.
    """

    def _limpio(value):
        v = float(value)
        return v if np.isfinite(v) and v > 0 else float("nan")

    nombres = [f"{n}_err" for n in PSFAO_PARAM_NAMES]
    try:
        x_std = np.asarray(res.x_std, dtype=float).ravel()
        dxdy_std = np.asarray(res.dxdy_std, dtype=float).ravel()
    except (AttributeError, TypeError, ValueError):  # pragma: no cover - defensive
        return {name: float("nan") for name in (*nombres, "dx_err", "dy_err")}
    errors = {name: (_limpio(x_std[i]) if i < x_std.size else float("nan"))
              for i, name in enumerate(nombres)}
    # `res.dxdy` es (dx, dy), y `res.dxdy_std` va en el mismo orden.
    errors["dx_err"] = _limpio(dxdy_std[0]) if dxdy_std.size > 0 else float("nan")
    errors["dy_err"] = _limpio(dxdy_std[1]) if dxdy_std.size > 1 else float("nan")
    return errors


def _ring_residual(image, model, companion_yx, mask_radius, width=1.5):
    ny, nx = image.shape
    cy, cx = ny // 2, nx // 2
    yy, xx = np.mgrid[0:ny, 0:nx]
    r = np.hypot(yy - cy, xx - cx)
    comp_r = float(np.hypot(companion_yx[0] - cy, companion_yx[1] - cx))
    comp = np.hypot(yy - companion_yx[0], xx - companion_yx[1])
    ann = (np.abs(r - comp_r) < width) & np.isfinite(image) & (comp > mask_radius)
    halo = np.nanmedian(image[ann])
    if not np.isfinite(halo) or halo == 0:
        return float("nan")
    return float(100.0 * np.nanmedian(np.abs(image[ann] - model[ann])) / halo)


def _box3_apcorr(system, lam_A, params, norm_radius):
    """Box3 aperture correction from one Psfao parameter set (outlier metric)."""
    from maoppy.instrument import muse_nfm
    from maoppy.psfmodel import Psfao
    try:
        m = Psfao((52, 52), system=system, samp=float(muse_nfm.samp(float(lam_A) * 1e-10)))
        low, high = m.bounds
        eps = 1e-6
        x = [float(np.clip(float(v),
                           low[i] + eps if np.isfinite(low[i]) else -np.inf,
                           high[i] - eps if np.isfinite(high[i]) else np.inf))
             for i, v in enumerate(params)]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            img = np.asarray(m(x), dtype=np.float64)
        c = 26
        yy, xx = np.mgrid[0:52, 0:52]
        r = np.hypot(yy - c, xx - c)
        total = float(np.nansum(img[r <= norm_radius]))
        box = float(np.nansum(img[(np.abs(yy - c) <= 1) & (np.abs(xx - c) <= 1)]))
        return float(total / box) if box > 0 else np.nan
    except Exception:
        return np.nan


def prepare_psfao_inputs(cfg, stage_dir):
    """Load the cube/positions/system and derive the psfao binning + radii.

    Split out of ``run_stage_e01_psfao`` so the canonical Moffat stage
    (``stage_e01_psf``) can reuse the identical psfao setup when it runs the
    dual-form comparison (spec C1 §3.4). Behaviour is unchanged."""

    from maoppy.instrument import muse_nfm, muse_wfm

    cube_path = Path(cfg.get("stage_e01_input_cube_fits", stage_dir / "stage02_xcorr_cube_stack.fits"))
    pos_qc = json.load(open(cfg.get("stage_e01_positions_qc", stage_dir / "stage01c_qc.json")))
    primary = tuple(map(float, pos_qc["primary"]["pos_yx"]))
    companion = tuple(map(float, pos_qc["companion"]["pos_yx"]))

    system = muse_wfm if str(cfg.get("instrument_mode", "NFM")).upper().startswith("W") else muse_nfm

    with fits.open(cube_path) as h:
        cube = np.asarray(h["CUBES"].data if "CUBES" in h else h[1].data, float)
        wave = np.asarray(h["WAVELENGTH"].data, float)
        stat = np.asarray(h["STAT"].data, float) if "STAT" in h else None
    if cube.ndim == 4:
        cube = cube[0]
        stat = stat[0] if stat is not None else None
    if stat is None:
        stat = np.ones_like(cube)

    bin_A = float(cfg.get("psf_bin_A", 100.0))
    bad = _bad_windows(cfg)
    bins = make_bins(wave, bin_A, bad, int(cfg.get("psf_min_channels_per_bin", 3)))
    fwhm_prelim = float(pos_qc.get("companion", {}).get("err_px", 0) or 0) + float(cfg.get("psf_prelim_fwhm_px", 4.0))
    mask_radius = float(cfg.get("psf_companion_mask_radius_px", max(10.0, 3.0 * fwhm_prelim)))
    fit_radius = float(cfg.get("psf_fit_radius_px", 80.0))
    norm_radius = float(cfg.get("psf_norm_radius_px", 25.0))
    return {
        "cube_path": cube_path,
        "cube": cube,
        "stat": stat,
        "wave": wave,
        "primary": primary,
        "companion": companion,
        "system": system,
        "bin_A": bin_A,
        "bad": bad,
        "bins": bins,
        "mask_radius": mask_radius,
        "fit_radius": fit_radius,
        "norm_radius": norm_radius,
    }


def _psfao_fit_status(optimizer):
    """El veredicto de un intento: `ok`, estancado en el arranque, o fallido."""

    if optimizer["stalled_at_initial"]:
        return "fit_stalled:initial_vector"
    if not optimizer["success"]:
        return f"fit_failed:optimizer_status_{optimizer['status']}"
    return "ok"


def fit_psfao_bins(cube, stat, wave, bins, system, companion, mask_radius, fit_radius, *,
                   x0=None, field_yx=None, warm_start=True,
                   weighting=PSFAO_DEFAULT_WEIGHTING,
                   weight_cap=PSFAO_DEFAULT_WEIGHT_CAP,
                   companion_offset_yx=None):
    """Fit the Psfao model per wavelength bin (companion masked).

    Returns ``(rows, recons)`` where ``rows`` is the per-bin parameter table
    (identical structure to the legacy loop) and ``recons`` maps each bin centre
    wavelength to ``(bin_image, reconstructed_model)`` so a caller can score the
    reconstruction with any ring metric it likes.

    ``warm_start`` fits every bin from its neighbours as well as from the shared
    ``x0``, and keeps the attempt with the LOWEST optimiser cost — the weighted
    least squares of that same bin, so the attempts are directly comparable. A
    bin only changes if another start vector fits its own data strictly better;
    where ``x0`` already wins, the parameters are unchanged bit for bit. Two
    passes, both deterministic: forward from the last accepted bin, then
    backward from the next one, so a good solution propagates in either
    direction. With ``warm_start=False`` each bin gets the single ``x0`` fit.

    This matters because every bin left out is a HOLE in the ``param_table``
    that ``_evaluate_psfao`` then spans with a straight line, and that is where
    the plateau/step structure C2-C4 inherit through the growth curve is born.
    Rescuing only the stalled bins is not enough: in ROXs 12 b the 8100-8300 A
    bins CONVERGED, from ``x0``, onto a second minimum 8x worse in cost
    (7.5e5 vs 9.4e4) whose box3 aperture correction fell 10% off the chromatic
    trend, so the outlier guard in ``build_psfao_model_document`` dropped them —
    and, being converged, they seeded the warm start that then stalled the three
    bins above them. Fitting them from either neighbour recovers the good
    minimum in all six (cost 8.7-9.4e4, ring residual better in all of them,
    box3 back on the trend), which closes the 700 A gap."""

    from maoppy.instrument import muse_nfm

    x0 = DEFAULT_X0 if x0 is None else x0

    def _intento(datos, arranque, etiqueta):
        """Un ajuste del bin, con su veredicto. Propaga lo que reviente."""

        img, var, samp, mid = datos
        if companion_offset_yx is None:
            # Camino de siempre: `fit_bin` -> `maoppy.psffit`, sin tocar.
            params, amp, bck, dxdy, ring, recon, optimizer, errors = fit_bin(
                img, var, samp, system, companion, mask_radius, fit_radius,
                list(arranque), field_yx=field_yx, weighting=weighting,
                weight_cap=weight_cap)
            ratio, recon_psf = None, recon
        else:
            (params, amp, bck, dxdy, ring, recon, optimizer, errors,
             ratio, recon_psf) = fit_bin_binary(
                img, var, samp, system, companion, mask_radius, fit_radius,
                list(arranque), companion_offset_yx, field_yx=field_yx,
                weighting=weighting, weight_cap=weight_cap)
        estado = _psfao_fit_status(optimizer)
        row = {"lambda_A": float(mid), "samp": samp, "amp": amp, "bck": bck,
               "dy": dxdy[1], "dx": dxdy[0], "ring_residual_pct": ring,
               "optimizer_success": optimizer["success"],
               "optimizer_status": optimizer["status"],
               "optimizer_message": optimizer["message"],
               "optimizer_nfev": optimizer["nfev"],
               "optimizer_cost": optimizer["cost"],
               "optimizer_stalled_at_initial": optimizer["stalled_at_initial"],
               "start_vector": etiqueta,
               "status": estado}
        if ratio is not None:
            row["flux_ratio"] = float(ratio)
        row.update({name: params[i] for i, name in enumerate(PSFAO_PARAM_NAMES)})
        # Las incertidumbres formales del intento que se queda. NaN donde el
        # parametro esta pegado a su limite fisico.
        row.update(errors)
        # `recon` es la ESCENA (lo que el dato tiene) y `recon_psf` la PSF de una
        # fuente puntual (lo que se publica). Sin binaria son la misma imagen.
        return {"row": row, "params": list(params), "recon": recon,
                "recon_psf": recon_psf,
                "ok": estado == "ok", "cost": float(optimizer["cost"])}

    def _intento_suave(datos, arranque, etiqueta):
        try:
            return _intento(datos, arranque, etiqueta)
        except Exception:  # pragma: no cover - defensive
            return None

    def _mejor(actual, nuevo):
        """El intento que se queda: `ok` gana a no-`ok`, y entre dos `ok`, el de
        menor coste. El empate lo gana el que ya estaba, para no mover un bin
        que nadie mejora."""

        if nuevo is None:
            return actual
        if actual is None:
            return nuevo
        if nuevo["ok"] != actual["ok"]:
            return nuevo if nuevo["ok"] else actual
        if nuevo["ok"] and nuevo["cost"] < actual["cost"]:
            return nuevo
        return actual

    datos = []
    for (a, b, mid, sel) in bins:
        datos.append((np.nanmedian(cube[sel], axis=0),
                      np.nanmedian(stat[sel], axis=0),
                      float(muse_nfm.samp(mid * 1e-10)),
                      float(mid)))

    aceptado = [None] * len(bins)
    reventados = {}
    x0_caliente = None   # el ultimo bin aceptado como bueno, hacia el rojo
    for i, d in enumerate(datos):
        try:
            mejor = _intento(d, x0, "initial_vector")
        except Exception as exc:  # pragma: no cover - defensive
            reventados[i] = {"lambda_A": d[3], "status": f"fit_failed:{exc}"}
            continue
        if warm_start and x0_caliente is not None:
            mejor = _mejor(mejor, _intento_suave(d, x0_caliente, "warm_start"))
        aceptado[i] = mejor
        if mejor["ok"]:
            x0_caliente = mejor["params"]

    # Segunda pasada, hacia el azul: un bin cuyo unico vecino bueno esta al rojo
    # —el caso de 8400-8600 A en ROXs 12 b— no tiene de donde partir en la
    # primera. Se propaga el ajuste ya aceptado del bin siguiente.
    if warm_start:
        for i in range(len(datos) - 2, -1, -1):
            siguiente = aceptado[i + 1]
            if aceptado[i] is None or siguiente is None or not siguiente["ok"]:
                continue
            aceptado[i] = _mejor(aceptado[i],
                                 _intento_suave(datos[i], siguiente["params"], "warm_start_back"))

    rows, recons = [], {}
    for i, d in enumerate(datos):
        if i in reventados:
            rows.append(reventados[i])
            continue
        acc = aceptado[i]
        rows.append(acc["row"])
        if acc["ok"]:
            recons[d[3]] = (d[0], acc["recon"])
    return rows, recons


def build_psfao_model_document(rows, system, norm_radius, fit_radius, *, system_name="muse_nfm",
                               wave_bin_A=None, weighting=None,
                               weight_cap=PSFAO_DEFAULT_WEIGHT_CAP):
    """Assemble the psfao ``psf_model.json`` document from per-bin fit rows.

    Split out of ``run_stage_e01_psfao`` (behaviour unchanged) so the canonical
    stage can build the winning-form document when Psfao is selected. Returns the
    psf_model dict plus a small ``meta`` dict for QC assembly.

    ``wave_bin_A`` is the grid ``_evaluate_psfao`` snaps its wavelength to. It
    travels in the document because that is where the consumer reads it, and it
    defaults to the width of the bins actually fitted here: the parameters exist
    only at those wavelengths, so anything finer interpolates the degenerate PSD
    parameters between two bins and lands outside the valley (measured: a 1%
    error in the halo becomes 13% in the psffit amplitude). ``None`` leaves the
    key out, and ``psf.py`` then falls back to its historical 50 A."""

    ok = [r for r in rows if r.get("status") == "ok" and np.isfinite(r.get("ring_residual_pct", np.nan))]
    lam = np.array([r["lambda_A"] for r in ok])
    poly = {}
    for name in PSFAO_PARAM_NAMES:
        vals = np.array([r[name] for r in ok])
        deg = 2 if len(vals) >= 3 else 1
        poly[name] = list(map(float, np.polyfit(lam, vals, deg))) if len(vals) else []

    # Robust outlier rejection on the per-bin box3 aperture fraction: Psfao PSD
    # params are degenerate, so a failed bin shows up as an off-trend growth
    # curve rather than an obvious single-param spike. Keep the surviving per-bin
    # fits as an interpolation table (smoothing params directly corrupts the PSF).
    box3 = np.array([_box3_apcorr(system, float(r["lambda_A"]), [r[n] for n in PSFAO_PARAM_NAMES], norm_radius)
                     for r in ok])
    # The box3 growth curve spans a wide but smooth range (~20-130), so a global
    # MAD misses local failures; reject bins whose log-apcorr deviates from the
    # smooth wavelength trend (catches degenerate/failed bins like the 9100-9200
    # dip to ~2).
    with np.errstate(all="ignore"):
        logb = np.log(np.clip(box3, 1e-3, None))
    fin = np.isfinite(logb)
    trend = np.polyval(np.polyfit(lam[fin], logb[fin], 2), lam) if fin.sum() >= 3 else np.full_like(logb, np.nanmedian(logb))
    resid = logb - trend
    rmad = np.nanmedian(np.abs(resid[fin] - np.nanmedian(resid[fin]))) or 1.0
    keep = fin & (np.abs(resid - np.nanmedian(resid[fin])) <= 4.0 * 1.4826 * rmad)
    kept = [r for r, k in zip(ok, keep) if k]
    n_rejected = int(np.sum(~keep))
    lam_k = np.array([r["lambda_A"] for r in kept])
    order = np.argsort(lam_k)
    param_table = {"lambda_A": [float(lam_k[i]) for i in order]}
    for name in PSFAO_PARAM_NAMES:
        vals = np.array([kept[i][name] for i in order])
        param_table[name] = [float(v) for v in vals]

    rings = np.array([r["ring_residual_pct"] for r in ok])
    # normalization round-trip on a fine model at one lambda
    norm_err = _norm_roundtrip(system, float(np.median(lam)), poly, norm_radius)

    psf_model = {"form": "psfao", "system": system_name, "smoothed_poly": poly,
                 "param_table": param_table, "n_bins_rejected": n_rejected,
                 "norm_radius_px": norm_radius, "fit_radius_px": fit_radius,
                 "param_names": list(PSFAO_PARAM_NAMES), "hybrid": False}
    if weighting is not None:
        # Viaja con el producto por el mismo motivo que la rejilla: quien lea
        # este documento tiene que poder saber QUE decidio el ajuste.
        psf_model["psfao_fit_weighting"] = str(weighting)
        if str(weighting) == "relative":
            psf_model["psfao_fit_weight_cap"] = (
                None if weight_cap is None else float(weight_cap))
    if wave_bin_A is not None:
        psf_model["psfao_wave_bin_A"] = float(wave_bin_A)
        psf_model["psfao_wave_bin_A_note"] = (
            "Rejilla a la que _evaluate_psfao redondea lambda. Es el ancho de los bins "
            "que C1 ajusta: los parametros del PSD solo existen ahi, y una rejilla mas "
            "fina interpola entre dos bins parametros degenerados.")
    meta = {"ok": ok, "lam": lam, "rings": rings, "n_ok": len(ok),
            "n_rejected": n_rejected, "norm_err": norm_err, "poly": poly}
    return psf_model, meta


def run_stage_e01_psfao(run_id=None, *, project_root=None):
    rc = load_run_config(run_id, project_root=project_root)
    cfg = dict(rc.config)
    stage_dir = rc.paths.stage_dir

    inp = prepare_psfao_inputs(cfg, stage_dir)
    cube_path = inp["cube_path"]
    cube = inp["cube"]
    companion = inp["companion"]
    system = inp["system"]
    bin_A = inp["bin_A"]
    bad = inp["bad"]
    bins = inp["bins"]
    mask_radius = inp["mask_radius"]
    fit_radius = inp["fit_radius"]
    norm_radius = inp["norm_radius"]

    weighting = str(cfg.get("psf_fit_weighting", PSFAO_DEFAULT_WEIGHTING))
    weight_cap = cfg.get("psf_fit_weight_cap", PSFAO_DEFAULT_WEIGHT_CAP)
    rows, _ = fit_psfao_bins(cube, inp["stat"], inp["wave"], bins, system, companion,
                             mask_radius, fit_radius, weighting=weighting,
                             weight_cap=weight_cap)
    psf_model, meta = build_psfao_model_document(
        rows, system, norm_radius, fit_radius,
        wave_bin_A=float(cfg.get("psfao_wave_bin_A", bin_A)), weighting=weighting,
        weight_cap=weight_cap)
    ok = meta["ok"]
    lam = meta["lam"]
    rings = meta["rings"]
    n_rejected = meta["n_rejected"]
    norm_err = meta["norm_err"]

    # write products
    params_csv = stage_dir / "stage_e01_psfao_params.csv"
    with open(params_csv, "w") as f:
        cols = ["lambda_A", "samp", "amp", "bck", "dy", "dx", "ring_residual_pct", *PSFAO_PARAM_NAMES, "status"]
        f.write(",".join(cols) + "\n")
        for r in rows:
            f.write(",".join(str(r.get(c, "")) for c in cols) + "\n")

    (stage_dir / "psf_model.json").write_text(json.dumps(psf_model, indent=2))
    poly = meta["poly"]

    qc = {
        "stage": "e01_chromatic_psf", "run_id": rc.run_id, "provisional": True,
        "input": {"cube": str(cube_path), "positions_from": "stage01c_qc.json"},
        "binning": {"bin_A": bin_A, "n_bins": len(bins), "excluded_windows_A": bad},
        "masks": {"companion_radius_px": mask_radius, "chromatic_tracking": False,
                  "companion_yx": list(companion)},
        "fit": {"form_chosen": "psfao", "model": "maoppy.Psfao", "fit_radius_px": fit_radius,
                "n_ok_bins": len(ok), "weighting": weighting,
                "weight_cap": (None if weight_cap is None else float(weight_cap))},
        "smoothing": {"per_param_poly_deg": {k: (len(v) - 1) for k, v in poly.items()}},
        "companion_ring_metric": {
            "radius_px": float(np.hypot(companion[0] - cube.shape[1] // 2, companion[1] - cube.shape[2] // 2)),
            "residual_pct_median": float(np.median(rings)) if len(rings) else None,
            "residual_pct_p90": float(np.percentile(rings, 90)) if len(rings) else None,
            "residual_pct_red_median": float(np.median(rings[lam > 7000])) if np.any(lam > 7000) else None,
            "bins_above_5pct": int(np.sum(rings > 5.0)),
            "n_bins": len(rings),
        },
        "normalization": {"norm_radius_px": norm_radius, "roundtrip_error": norm_err},
        "open_issues": [
            "PROVISIONAL: run on the ESO ADP while A1 exposure-alignment fix is pending "
            "(ver PAPER_BLOCKERS.md del run raw de este objeto).",
            "Implements the maoppy Psfao path that stage_e01_psf.py leaves as a stub "
            "(form was hard-coded to Moffat). Moffat-vs-Psfao AIC comparison and the "
            "hybrid term are not re-implemented here; Psfao alone meets the <5% ring target "
            "in the red bins where the companion lives.",
        ],
    }
    (stage_dir / "stage_e01_qc.json").write_text(json.dumps(qc, indent=2))
    return qc


def _norm_roundtrip(system, lam_A, poly, norm_radius):
    from maoppy.instrument import muse_nfm
    from maoppy.psfmodel import Psfao
    try:
        x = [float(np.polyval(poly[n], lam_A)) for n in PSFAO_PARAM_NAMES]
        npix = int(4 * norm_radius)
        samp = float(muse_nfm.samp(lam_A * 1e-10))
        m = Psfao((npix, npix), system=system, samp=samp)
        img = m(x)
        cy, cx = npix // 2, npix // 2
        yy, xx = np.mgrid[0:npix, 0:npix]
        inside = np.hypot(yy - cy, xx - cx) <= norm_radius
        total = float(np.nansum(img[inside]))
        return float(abs(total / np.nansum(img) - 1.0)) if np.nansum(img) else float("nan")
    except Exception:
        return float("nan")


__all__ = [
    "run_stage_e01_psfao",
    "prepare_psfao_inputs",
    "fit_psfao_bins",
    "build_psfao_model_document",
    "fit_bin",
    "make_bins",
    "PSFAO_PARAM_NAMES",
]
