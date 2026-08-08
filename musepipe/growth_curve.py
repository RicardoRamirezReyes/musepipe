"""Curva de crecimiento empirica de la fuente brillante, hasta el flujo total.

Nucleo numerico compartido por **A2** (`musepipe/reduction/sky_zap.py`, que ya
tiene abierto el cubo grande) y por `scripts/measure_growth_curve.py`. Una sola
implementacion a proposito: son la misma medida y divergirian.

## Que resuelve

`apcorr` (C2/C3/C4) normaliza la PSF a 1 dentro de `norm_radius_px` (25 px =
0.63" en NFM), asi que su "flujo total" es en realidad "flujo dentro de 0.63"".
Medido sobre ROXs12b_realigned, fuera de ese radio queda cerca de la mitad de la
luz: el factor que falta es **1.9-2.5 y es cromatico**, o sea que no se cancela
y deforma la pendiente del continuo.

## El metodo (anillos concentricos, cielo medido, cola extrapolada)

1. **Perfil radial** en anillos de 1 px, con **mediana azimutal**: robusta
   frente al companero y a cualquier otra fuente del campo.
2. **Ajuste `SB(r) = A*r**-p + S`** en un rango exterior. `S` es el fondo
   residual y **hay que medirlo**: el DRS sobre-resta cielo y `S` sale NEGATIVO
   (-0.2 a -0.9 en ROXs12b). Suponerlo cero infla el total, porque el suelo
   multiplica por decenas de miles de spaxels.
3. **Suma directa** de los pixeles (menos `S`) hasta el ultimo anillo COMPLETO,
   mas la **cola analitica** `2*pi*A*R**(2-p)/(p-2)`, que converge porque el
   halo AO va como r^-3.1..3.4. En ROXs12b la cola es solo ~5% del total: la
   medida esta dominada por datos, no por la extrapolacion.

## El limite honesto

Dentro del campo **no hay ningun radio donde el halo sea despreciable**: iguala
a |S| hacia r ~ 170 px y el ultimo anillo completo cae en 163. Cielo y halo se
ajustan a la vez y quedan degenerados, asi que **el rango del ajuste es la barra
de error dominante**, no el ruido: mover el borde interior de 40 a 80 px mueve
el factor ~+-12% en el azul y ~+-5% en el rojo. Por eso se recorren varios
rangos y se reporta la dispersion (`ratio_min`/`ratio_max`) junto al adoptado.

Validacion: comparando `F(<=78)/F(<=25)`, el modelo psfao de C1 reproduce estos
datos al 1.3-2.3%. El modelo esta bien; lo que estaba mal era donde se ponia el
"1".

## El campo importa, y hay que vigilarlo (2026-08-07)

Todo lo de arriba se midio sobre el cubo de **330 px**. Cuando la cadena paso al
combinado multinoche, recortado a **200 px** por `stream_combine`, esta medida
**colapso sin avisar**: el ultimo anillo completo bajo de 163 a 99 px, cielo y
halo dejaron de ser separables, `p` cayo a 2.20-2.62, el suelo huyo a -2.6 y
hasta el **45%** del "total" paso a ser cola extrapolada, con la dispersion entre
rangos de ajuste al 100%. El factor resultante **no era monotono en lambda**
—maximo interior hacia 7400 A— y deformo la pendiente del continuo de TODOS los
productos del bloque C un ~47%, en el sentido de hacerlos mas rojos.

Se descubrio dividiendo el espectro entregado por una apertura cruda sobre el
cubo: el cociente tenia que ser plano y no lo era. De ahi los guardias
(`MIN_HALO_POWER`, `MAX_TAIL_FRACTION`, `MAX_RATIO_SPREAD`, `MAX_INTERIOR_BUMP`)
y el diagnostico permanente en `D2_primary_star_debug` §10. La regla que queda:
**el factor solo vale si el halo llega al suelo de cielo DENTRO del campo**, y
si no, hay que re-medirlo en un combinado mas ancho.
"""

from __future__ import annotations

import math

import numpy as np

#: Rangos de ajuste recorridos para estimar el sistematico del suelo de cielo,
#: **en fracciones del ultimo anillo completo**, no en pixeles. Los campos
#: cambian de un objeto a otro (330 px en ROXs12b, 200 en ROXs42Bb por el
#: recorte de `stream_combine.DEFAULT_CROP_NPIX`), asi que un rango en pixeles
#: absolutos vale para un objeto y no para el siguiente. Las fracciones son las
#: que dieron 40/60/80/50/100 px sobre los 163 px de ROXs12b.
FIT_RANGE_FRACTIONS = ((0.245, 1.44), (0.368, 1.44), (0.491, 1.44), (0.307, 1.23), (0.613, 1.45))
#: Rango adoptado: donde el ajuste es estable en las tres bandas (60/163).
DEFAULT_FIT_FRACTION = (0.368, 1.44)
#: Radio enmascarado alrededor del companero, para que no entre en las medianas.
DEFAULT_COMPANION_MASK_PX = 6.0
#: Radio de referencia de la convencion actual de `apcorr`.
DEFAULT_NORM_RADIUS_PX = 25.0
#: Rango fisico del exponente del halo AO. Kolmogorov da -11/3 ~ 3.67 y lo
#: medido en ROXs12b cae en 3.1-3.7. Fuera de aqui el ajuste ha colapsado:
#: pasa cuando el campo es demasiado pequeno y cielo y halo son indistinguibles,
#: y entonces el ajuste mete todo el flujo en el suelo y dispara `p`. Medido en
#: ROXs42Bb (campo recortado a 200 px por `stream_combine.DEFAULT_CROP_NPIX`)
#: salian p = 10, 13, 16 y 24, con factores que parecian razonables (1.13-1.26)
#: y no lo eran.
#: Suelo SUBIDO de 2.05 a 2.5 el 2026-08-07. El 2.05 no rechazaba nada de lo
#: que hay que rechazar: sobre el cubo combinado de 200 px el ajuste colapsaba a
#: p = 2.20-2.62 —con el suelo de cielo huyendo a -2.6 y casi la mitad del
#: "total" en la cola extrapolada— y el factor se adoptaba en silencio. La forma
#: resultante de `apcorr` no era monotona (maximo interior hacia 7400 A) y
#: deformaba la pendiente del continuo un ~47%. Kolmogorov da 3.67 y lo medido
#: en el campo grande de ROXs12b caia en 3.1-3.4: 2.5 deja margen de sobra al
#: halo real y corta el colapso.
MIN_HALO_POWER = 2.5
MAX_HALO_POWER = 5.0
#: Fraccion maxima del "total" que puede venir de la cola EXTRAPOLADA. Por
#: encima, el numero lo domina el modelo y no el dato: en el cubo de 200 px se
#: llegaba a 0.455. En el campo grande la cola era ~5%.
MAX_TAIL_FRACTION = 0.25
#: Dispersion sistematica maxima por banda (`ratio_max/ratio_min` entre rangos
#: de ajuste). Cielo y halo son degenerados cuando el campo es pequeno y esta
#: razon se dispara: se median 1.5-2.2, o sea el factor era incierto al 100%.
MAX_RATIO_SPREAD = 1.6
#: Joroba maxima tolerada al juzgar la monotonia de una curva que deberia caer
#: con lambda. Se mide con `interior_bump`, no por signo: un test estricto marca
#: tambien un repunte del ~1% en el borde por ruido o por extrapolacion del
#: ajuste. El caso que motivo el guardia valia 11%.
MAX_INTERIOR_BUMP = 0.03
#: Brazo de palanca minimo del ajuste DENTRO de los anillos completos. Sin esto
#: el ajuste se apoya en las esquinas, que muestrean pocas direcciones.
MIN_FIT_BASELINE_PX = 40.0


def interior_bump(values):
    """Cuanto sube un maximo INTERIOR por encima del mayor de los dos extremos.

    0.0 si la curva es monotona (el maximo cae en un extremo). Es la medida que
    separa una joroba real de un repunte de borde: no mira signos de la
    derivada, mira tamano.
    """

    vals = np.asarray(values, dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    if vals.size < 3:
        return 0.0
    borde = max(float(vals[0]), float(vals[-1]))
    if borde <= 0:
        return 0.0
    return max(0.0, float(np.max(vals)) / borde - 1.0)


def interior_dip(values):
    """Cuanto sube la curva desde un minimo INTERIOR hasta su extremo final.

    El espejo de `interior_bump`, y hace falta porque `interior_bump` **no ve
    este caso**: una curva que baja, toca fondo dentro del rango y vuelve a
    subir tiene su maximo en un extremo, asi que `interior_bump` devuelve 0.0 y
    el diagnostico la declara «monotona». Sobre `ROXs12b_realigned` la parabola
    de `factor_at_wavelengths` toca fondo en 7978 A y repunta un 3.15% hasta
    9350 A, mientras la curva medida entre 8487 y 9062 A sube 0.37%: el
    repunte lo fabrica el ajuste, y `interior_bump` daba 0.00%.

    0.0 si la curva es monotona (el minimo cae en un extremo).
    """

    vals = np.asarray(values, dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    if vals.size < 3:
        return 0.0
    i = int(np.argmin(vals))
    if i in (0, vals.size - 1):
        return 0.0
    fondo = float(vals[i])
    if fondo <= 0:
        return 0.0
    # Solo cuenta lo que sube DESPUES del minimo: el tramo de bajada anterior
    # es justo lo que la curva debe hacer.
    return max(0.0, float(np.max(vals[i:])) / fondo - 1.0)


def polynomial_misfit(growth_qc, *, degree=2):
    """`(rms, pico_a_pico)` del ajuste polinomico contra las bandas medidas.

    `factor_at_wavelengths` interpola 8 bandas con un polinomio de grado bajo
    para no meter escalones en el continuo, pero nadie comprobaba cuanto se
    aparta ese polinomio de lo que se midio. En el run canonico vale 1.03% rms
    y 2.82% pico a pico, y es lo que produce el repunte rojo que mide
    `interior_dip`.
    """

    bands = (growth_qc or {}).get("bands") or []
    if len(bands) < degree + 2:
        return None, None
    x = np.asarray([b["wave_A"] for b in bands], dtype=np.float64)
    y = np.asarray([b["ratio_total_over_normrad"] for b in bands], dtype=np.float64)
    fit = np.polyval(np.polyfit(x, y, int(degree)), x)
    rel = fit / y - 1.0
    return float(np.std(rel)), float(np.max(rel) - np.min(rel))


def halo_plus_sky(r, amp, power, sky):
    """Perfil radial: halo en ley de potencias mas un suelo constante."""

    return amp * np.asarray(r, dtype=np.float64) ** (-power) + sky


def band_images(cube_zyx, wave_A, n_bands=8, chunk=150):
    """Imagenes promediadas por banda, acumulando por trozos.

    El cubo grande de A2 son ~1.5 GB en float32; se acumula en bloques de
    `chunk` canales para no traerlo entero a memoria (esta medida puede correr
    a la vez que una etapa pesada).
    """

    wave = np.asarray(wave_A, dtype=np.float64)
    nz = int(wave.size)
    ny, nx = cube_zyx.shape[1:]
    edges = np.linspace(float(wave[0]), float(wave[-1]), int(n_bands) + 1)
    images = np.full((int(n_bands), ny, nx), np.nan, dtype=np.float64)
    for band in range(int(n_bands)):
        sel = np.where((wave >= edges[band]) & (wave < edges[band + 1]))[0]
        if sel.size == 0:
            continue
        z0, z1 = int(sel[0]), int(sel[-1]) + 1
        acc = np.zeros((ny, nx), dtype=np.float64)
        cnt = np.zeros((ny, nx), dtype=np.float64)
        for c0 in range(z0, z1, int(chunk)):
            blk = np.asarray(cube_zyx[c0:min(c0 + int(chunk), z1)], dtype=np.float64)
            good = np.isfinite(blk)
            acc += np.where(good, blk, 0.0).sum(axis=0)
            cnt += good.sum(axis=0)
        images[band] = np.where(cnt > 0, acc / np.maximum(cnt, 1.0), np.nan)
    return images, 0.5 * (edges[:-1] + edges[1:])


def locate_brightest(image, box_half=4):
    """Centroide de la fuente mas brillante (la primaria)."""

    iy, ix = np.unravel_index(np.nanargmax(image), image.shape)
    y0, y1 = max(0, iy - box_half), min(image.shape[0], iy + box_half + 1)
    x0, x1 = max(0, ix - box_half), min(image.shape[1], ix + box_half + 1)
    sub = np.nan_to_num(image[y0:y1, x0:x1])
    yy, xx = np.mgrid[y0:y1, x0:x1]
    total = float(sub.sum())
    if not np.isfinite(total) or total <= 0:
        return float(iy), float(ix)
    return float((yy * sub).sum() / total), float((xx * sub).sum() / total)


def radial_profile(image, radius, bins, min_pixels=8):
    """Mediana azimutal por anillo (mediana, no media: hay fuentes en el campo)."""

    idx = np.digitize(radius.ravel(), bins) - 1
    values = np.asarray(image, dtype=np.float64).ravel()
    profile = np.full(len(bins) - 1, np.nan, dtype=np.float64)
    for k in range(len(bins) - 1):
        sel = (idx == k) & np.isfinite(values)
        if int(sel.sum()) >= int(min_pixels):
            profile[k] = float(np.median(values[sel]))
    return profile


def fit_halo_and_sky(r_centers, profile, fit_range):
    """Ajusta `A*r**-p + S`. Devuelve (amp, power, sky) o None si no hay datos."""

    from scipy.optimize import curve_fit

    lo, hi = float(fit_range[0]), float(fit_range[1])
    sel = (r_centers >= lo) & (r_centers <= hi) & np.isfinite(profile)
    if int(sel.sum()) < 10:
        return None
    popt, _ = curve_fit(
        halo_plus_sky, r_centers[sel], profile[sel], p0=[1e5, 3.0, 0.0], maxfev=20000
    )
    return tuple(float(v) for v in popt)


def integrate_total(image, radius, amp, power, sky, r_complete):
    """Suma directa hasta `r_complete` (menos el suelo) + cola analitica."""

    if power <= 2.0:
        raise ValueError(
            f"El halo ajusta con p={power:.3f} <= 2: la integral de flujo no converge "
            "y no se puede extrapolar fuera del campo."
        )
    finite = np.isfinite(image)
    net = image - sky
    inside = float(np.nansum(np.where(finite & (radius <= float(r_complete)), net, 0.0)))
    tail = 2.0 * math.pi * amp * float(r_complete) ** (2.0 - power) / (power - 2.0)
    return inside, float(tail)


def measure_growth_curve(
    cube_zyx,
    wave_A,
    *,
    n_bands=8,
    fit_fraction=DEFAULT_FIT_FRACTION,
    norm_radius_px=DEFAULT_NORM_RADIUS_PX,
    companion_yx=None,
    companion_mask_px=DEFAULT_COMPANION_MASK_PX,
    pixel_scale_arcsec=None,
):
    """Mide el factor `F_total / F(<=norm_radius_px)` por banda.

    Devuelve un diccionario listo para escribir en QC. **No aplica nada**: A2
    mide y publica, y es C2 quien decide si usa el factor (asi el cambio de
    convencion de flujo no queda enterrado en una etapa de reduccion).
    """

    images, centers = band_images(cube_zyx, wave_A, n_bands=n_bands)
    white = np.nanmean(images, axis=0)
    cy, cx = locate_brightest(white)
    yy, xx = np.indices(white.shape, dtype=np.float64)
    radius = np.hypot(yy - cy, xx - cx)
    # Ultimo anillo COMPLETO: mas alla solo contribuyen las esquinas, que
    # muestrean unas pocas direcciones y sesgan la mediana azimutal.
    r_complete = float(min(cy, white.shape[0] - 1 - cy, cx, white.shape[1] - 1 - cx))

    masked = np.zeros(white.shape, dtype=bool)
    if companion_yx is not None:
        masked = np.hypot(yy - float(companion_yx[0]), xx - float(companion_yx[1])) <= float(
            companion_mask_px
        )

    # Rangos de ajuste en pixeles, escalados al campo de ESTE objeto.
    fit_ranges = tuple(
        (round(f0 * r_complete, 1), round(f1 * r_complete, 1)) for f0, f1 in FIT_RANGE_FRACTIONS
    )
    fit_range = (round(fit_fraction[0] * r_complete, 1), round(fit_fraction[1] * r_complete, 1))

    bins = np.arange(0.0, float(np.ceil(radius.max())) + 1.0, 1.0)
    r_centers = 0.5 * (bins[:-1] + bins[1:])
    rows = []
    rejected = []
    for band in range(images.shape[0]):
        image = np.where(masked, np.nan, images[band])
        profile = radial_profile(image, radius, bins)
        ratios = {}
        for candidate in fit_ranges:
            # Brazo de palanca dentro de los anillos COMPLETOS: si el ajuste se
            # apoya casi solo en las esquinas, cielo y halo son indistinguibles.
            if min(float(candidate[1]), r_complete) - float(candidate[0]) < MIN_FIT_BASELINE_PX:
                rejected.append({"band_wave_A": float(centers[band]), "fit_range_px": list(candidate),
                                 "reason": "fit baseline inside complete annuli is too short"})
                continue
            popt = fit_halo_and_sky(r_centers, profile, candidate)
            if popt is None:
                continue
            amp, power, sky = popt
            if not (MIN_HALO_POWER <= power <= MAX_HALO_POWER):
                # El ajuste colapso: `p` fuera del rango fisico de un halo AO.
                # Se descarta en vez de devolver un factor que parece razonable.
                rejected.append({"band_wave_A": float(centers[band]), "fit_range_px": list(candidate),
                                 "halo_power": float(power), "sky_floor": float(sky),
                                 "reason": f"halo power {power:.2f} outside [{MIN_HALO_POWER}, {MAX_HALO_POWER}]"})
                continue
            try:
                inside, tail = integrate_total(image, radius, amp, power, sky, r_complete)
            except ValueError:
                continue
            ref = float(
                np.nansum(np.where(np.isfinite(image) & (radius <= float(norm_radius_px)), image - sky, 0.0))
            )
            if not np.isfinite(ref) or ref == 0.0:
                continue
            tail_frac = tail / (inside + tail) if (inside + tail) else float("nan")
            if not np.isfinite(tail_frac) or tail_frac > MAX_TAIL_FRACTION:
                # El "total" lo domina la extrapolacion: es un modelo, no una medida.
                rejected.append({"band_wave_A": float(centers[band]), "fit_range_px": list(candidate),
                                 "halo_power": float(power), "tail_fraction": float(tail_frac),
                                 "reason": f"extrapolated tail {tail_frac:.3f} > {MAX_TAIL_FRACTION}"})
                continue
            ratios[tuple(candidate)] = {
                "amp": amp, "power": power, "sky": sky,
                "ratio": (inside + tail) / ref,
                "tail_frac": tail_frac,
            }
        if not ratios:
            continue
        adopted = ratios.get(tuple(fit_range)) or next(iter(ratios.values()))
        spread = [v["ratio"] for v in ratios.values()]
        # Degeneracion cielo/halo: si mover el rango de ajuste mueve el factor
        # mas que `MAX_RATIO_SPREAD`, la banda no esta midiendo nada utilizable.
        if len(spread) > 1 and float(np.nanmax(spread)) / float(np.nanmin(spread)) > MAX_RATIO_SPREAD:
            rejected.append({"band_wave_A": float(centers[band]),
                             "ratio_min": float(np.nanmin(spread)), "ratio_max": float(np.nanmax(spread)),
                             "reason": (f"fit-range spread {np.nanmax(spread) / np.nanmin(spread):.2f}x "
                                        f"> {MAX_RATIO_SPREAD} (sky and halo are degenerate)")})
            continue
        rows.append({
            "wave_A": float(centers[band]),
            "ratio_total_over_normrad": float(adopted["ratio"]),
            "ratio_min": float(np.nanmin(spread)),
            "ratio_max": float(np.nanmax(spread)),
            "halo_power": float(adopted["power"]),
            "sky_floor": float(adopted["sky"]),
            "tail_fraction": float(adopted["tail_frac"]),
        })

    if not rows:
        raise RuntimeError(
            "No band yielded a usable growth curve. Most likely the field is too small to "
            "separate the AO halo from the sky: the halo must reach the sky floor INSIDE the "
            "last complete annulus (r={:.0f} px here). Rejections: {}".format(
                r_complete, rejected[:4]
            )
        )
    spread_pct = [100.0 * (r["ratio_max"] - r["ratio_min"]) / r["ratio_total_over_normrad"] for r in rows]
    return {
        "method": "empirical_annuli_with_fitted_sky_and_power_law_tail",
        "norm_radius_px": float(norm_radius_px),
        "pixel_scale_arcsec": None if pixel_scale_arcsec is None else float(pixel_scale_arcsec),
        "star_yx": [cy, cx],
        "r_last_complete_annulus_px": r_complete,
        "fit_range_px": [float(fit_range[0]), float(fit_range[1])],
        "fit_ranges_swept_px": [list(fr) for fr in fit_ranges],
        "companion_masked_yx": None if companion_yx is None else [float(v) for v in companion_yx],
        "bands": rows,
        "rejected_fits": rejected,
        "n_bands_requested": int(images.shape[0]),
        "systematic_spread_pct_max": float(np.nanmax(spread_pct)),
        "note": (
            "Factor to convert the apcorr `normrad_total_flux` convention into total flux. "
            "MEASURED, NOT APPLIED: C2 decides via `x01_flux_convention`. The fit-range spread "
            "(ratio_min..ratio_max) is the dominant systematic, not the noise."
        ),
    }


#: Producto de run donde vive la medida. Lo escribe A2 cuando corre, o
#: `scripts/measure_growth_curve.py --write-run-product` para los runs donde A2
#: no corre (perfil `cascade`) o corrio antes de que esto existiera.
RUN_PRODUCT_NAME = "growth_curve_qc.json"


def load_run_growth_curve(stage_dir):
    """Lee `stages/growth_curve_qc.json` del run, o None si no esta."""

    from pathlib import Path

    path = Path(stage_dir) / RUN_PRODUCT_NAME
    if not path.exists():
        return None
    import json

    doc = json.loads(path.read_text(encoding="utf-8"))
    if not doc.get("bands"):
        raise RuntimeError(f"{path} has no usable bands.")
    return doc


def resolve_flux_convention(cfg, stage_dir, *, knob="flux_convention"):
    """Devuelve `(growth_curve_o_None, convencion)` segun el config del run.

    `flux_convention`:
      * ``"normrad"`` (por defecto) -- convencion historica: el "1" es el flujo
        dentro de `norm_radius_px`. Nada se mueve.
      * ``"total"`` -- multiplica por el factor empirico medido, y el producto
        declara `SCALEREF=empirical_total_flux`.

    Falla ruidosamente si se pide ``total`` y no hay medida: la alternativa
    seria caer en silencio a la convencion vieja mientras el header dice otra
    cosa, que es justo lo que hacia que este problema no se viera.
    """

    convention = str(cfg.get(knob, cfg.get("flux_convention", "normrad"))).lower()
    if convention in {"normrad", "norm_radius", "legacy"}:
        return None, "normrad"
    if convention not in {"total", "empirical_total"}:
        raise ValueError(f"Unknown flux_convention={convention!r}; expected 'normrad' or 'total'.")
    doc = load_run_growth_curve(stage_dir)
    if doc is None:
        raise RuntimeError(
            f"flux_convention='total' requires {RUN_PRODUCT_NAME} in the run's stages/: "
            "run `python scripts/measure_growth_curve.py --run-id <RUN> --write-run-product` "
            "(or A2, which emits it) first."
        )
    return doc, "total"


def factor_at_wavelengths(growth_qc, wave_A, *, degree=2, require_monotonic=True):
    """Interpola el factor por banda a un eje de longitudes de onda.

    Polinomio de grado bajo en vez de interpolacion lineal: las bandas son 8
    puntos de una curva suave, y un polinomio no mete escalones en el continuo.

    **Se exige monotonia** (2026-08-07). El factor es `F_total / F(<=r_norm)`, o
    sea el inverso de una fraccion encerrada: con AO el Strehl empeora hacia el
    azul, mas luz se va al halo y la correccion tiene que **caer** del azul al
    rojo. Un maximo interior no es fisico. La parabola ajustada a las 8 bandas
    del cubo de 200 px tenia el maximo en ~7400 A, y eso deformaba la pendiente
    del continuo de TODOS los productos del bloque C un ~47%. Si aparece, se
    para: las bandas estan mal y hay que re-medirlas en un campo mayor, no
    interpolarlas.
    """

    bands = (growth_qc or {}).get("bands") or []
    if len(bands) < degree + 1:
        raise ValueError(f"Need at least {degree + 1} bands to fit the factor, got {len(bands)}.")
    x = np.asarray([b["wave_A"] for b in bands], dtype=np.float64)
    y = np.asarray([b["ratio_total_over_normrad"] for b in bands], dtype=np.float64)
    coeff = np.polyfit(x, y, int(degree))
    wave = np.asarray(wave_A, dtype=np.float64)
    out = np.polyval(coeff, wave)
    if require_monotonic:
        # Se juzga sobre el eje pedido, que es donde se aplica: un polinomio
        # puede tener el vertice fuera del rango de las bandas y dentro del
        # rango del espectro.
        #
        # Con TOLERANCIA, no por signo: un test estricto marca tambien un
        # repunte del 1% en el borde por ruido numerico o por extrapolacion del
        # ajuste. Lo que importa es el TAMANO de la joroba: cuanto sube el
        # maximo interior por encima del mayor de los dos extremos. En el caso
        # que motivo esto valia 11%; en una curva sana vale 0.
        if np.isfinite(out).sum() > 2:
            joroba = interior_bump(out)
            if joroba > MAX_INTERIOR_BUMP:
                pico = float(wave[np.nanargmax(out)])
                raise RuntimeError(
                    f"The growth-curve factor is not monotonic in wavelength: it bumps {100 * joroba:.1f}% "
                    f"above its endpoints, peaking at {pico:.0f} A. It is the inverse of an "
                    "enclosed fraction: under AO it must "
                    "decrease from blue to red. Non-monotonic bands mean the halo/sky fit is "
                    "degenerate — re-measure on a wider field with "
                    "`scripts/measure_growth_curve.py --cube <cubo ancho> --write-run-product`. "
                    f"bands={np.array2string(y, precision=3)}")
    return out


__all__ = [
    "DEFAULT_COMPANION_MASK_PX",
    "DEFAULT_FIT_FRACTION",
    "DEFAULT_NORM_RADIUS_PX",
    "FIT_RANGE_FRACTIONS",
    "band_images",
    "RUN_PRODUCT_NAME",
    "factor_at_wavelengths",
    "interior_bump",
    "interior_dip",
    "polynomial_misfit",
    "MAX_INTERIOR_BUMP",
    "fit_halo_and_sky",
    "growth_curve_note",
    "halo_plus_sky",
    "integrate_total",
    "load_run_growth_curve",
    "locate_brightest",
    "measure_growth_curve",
    "radial_profile",
    "resolve_flux_convention",
]

growth_curve_note = measure_growth_curve.__doc__
