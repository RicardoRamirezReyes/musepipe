#!/usr/bin/env python
"""¿Y si el centro del compañero estuviera mal? B3, la traza cromática y su efecto en el flujo y en E1, SIN escribir.

Pasa al pipeline el estudio del 2026-10-21 (`~/muse_sesion_20261021/centro/`: `INFORME.md` y `scripts/s1`, `s4`,
`s9`), que se midió ANTES de dos cambios de la cadena que mueven sus números: el evaluador de la PSF centrado
(`musepipe.psf.MAOPPY_CENTRADO_PX`, `docs/2026-10-21_psf_evaluador_centrado.md`) y C4 v2 `decoupled_c0`
(`docs/2026-10-24_c4_planos_desacoplados_decision.md`). Aquí todo sale de las funciones de la cadena, en memoria:

(a) `b3`: **precisión de B3 por inyección.** Compañeros falsos (la PSF de C1, `evaluate_psf_model`, por el continuo
    del compañero de C4) en un anillo al radio del compañero, y el algoritmo de B3 entero (`locate_companion`: banda
    roja y bandas de respaldo, mediana azimutal, `detect_restricted_source`, `centroid_2d`). Medida − verdad da
    sesgo, σ y rms; los fallos (`AmbiguousDetectionError`, `SourceNotDetectedError`) se cuentan, y también las
    medidas a más de media FWHM de la verdad, que no son un error de centroide sino OTRA fuente: si la banda roja
    es ambigua, B3 prueba las de respaldo y puede devolver otro pico sin avisar (`clasifica_inyecciones`). La
    predicción que recibe B3 lleva el mismo error que tuvo la del compañero real (predicha − medida en su QC).
(b) `traza`: **la traza cromática** de la primaria y del compañero con `chromatic_centroids` de B3. Con las bandas
    del run reproduce `tables/stage01c_chromatic_centroids.csv`; con `--bandas-traza` (12 por defecto) da la traza
    fina, y la σ de cada banda sale de las mismas inyecciones de (a). Se añade el centroide del canal de Hα que
    mide E2 (T2) respecto de la posición de B3 (`centroide_ha_e2`), la única medida de la cadena de dónde está la
    línea.
(c) `flujo`: **sensibilidad del flujo.** Los métodos de C2–C6 (`--metodos`, los seis por defecto) extraídos con la
    posición declarada del compañero movida 0.1/0.25/0.5/1 px en cuatro direcciones (radial hacia fuera/dentro,
    tangencial a +90°/−90°) y a lo largo de la traza cromática: posición(λ) = B3 + p(λ) − p(λ_B3), con p la traza de
    la PRIMARIA (B3 la escribe en su CSV; `traza_12` usa la de 12 bandas), constante dentro de cada tramo del modelo
    de PSF (el modelo de C1 es constante por tramos de `psfao_wave_bin_A`). Flujo de la línea con la plantilla del
    máximo de E1 en δ = 0 (`matched_filter_point`) y continuo en bandas.
(d) `e1`: **el z de E1 y su FAP en cada caso.** Cada espectro pasa por D2 (`calibrate_spectrum_product`) y por E1
    (`analyze_halpha_method`) con los controles RE-EXTRAÍDOS en la misma pasada —el anillo que la etapa pone
    alrededor de la posición declarada— y calibrados como en D2 (`control_spectra_cal` × `flux_scale`, que es lo que
    hace `calibrate_control_spectra`); el veredicto, con `classify_h01_verdict` y los criterios congelados del config
    (estimador, FAP < `h01_detection_fap`, pares admisibles). Con `--controles-de-produccion` se evalúa además con los
    controles y los errores del disco, que es lo que hacía `s4` de la sesión.

Qué se mueve y qué no. Solo la posición declarada del compañero en la extracción (y con ella el anillo de controles,
como lo pondría la etapa). Lo que las etapas restan antes de extraer queda como está en disco: el cubo de C1b para
`optimal_psfsub` y los cubos residuales de C5/C6 para `sgf`/`lpm` (su referencia excluye 3 px alrededor del compañero;
moverla 1 px no cambia esos cubos de forma apreciable). La primaria tampoco se mueve.

Ventanas. E1 solo mira ±500 km/s alrededor de Hα, con un continuo de 80 Å: extraer 6470–6660 Å basta. El continuo se
mide en bandas de 200 Å, un canal de cada `PASO_CONTINUO`. Que la ventana no cambie nada lo comprueba `verifica`: con
δ = 0 se reproducen el producto de C2–C6 en esos canales (flujo, errores, controles), los controles calibrados de D2 y
el z y la FAP de E1 del disco.

Nada se escribe en `runs/`. No hay ficheros temporales; las únicas salidas son `--json` y `--figura`, y antes de
escribirlas se comprueba con la guarda de `recomputa_e1_e3.py` (`comprueba_fuera`) que caen fuera de `runs/`, de cada
run (con los enlaces resueltos) y de la copia.

Reproducir la sesión del 2026-10-21. Sus entradas siguen en la copia que dejó el re-corrido del evaluador centrado
(`stages/_antes_psf_centrado_20261021T224346`: modelo de PSF, productos de C2–C4, controles, D2 y E1 de entonces); el
cubo de B2 no ha cambiado. Con `RUN@COPIA`, `--evaluador-descentrado` (`MAOPPY_CENTRADO_PX = 0`, el de entonces) y
`--fondo-c4 shared`, `--metodos aperture,optimal_ls,psffit` (los cubos de C1b, C5 y C6 de entonces no se guardaron) da
el z de E1 de la sesión en δ = 0 (12 b: 10.599 / 12.032 / 11.635; 42B b: 2.489 / 2.171 / 2.968) y, con
`--controles-de-produccion`, el 3.22 de psffit de 42B b a lo largo de la traza.

uso:
    python scripts/centro_companero.py --project-root RAIZ --sin-escribir --json SALIDA.json [--figura TRAZA.png]
    python scripts/centro_companero.py --project-root RAIZ --sin-escribir --objetivo SLUG --partes verifica,e1
    python scripts/centro_companero.py --project-root RAIZ --sin-escribir \\
        --run-id RUN@runs/RUN/stages/_antes_psf_centrado_20261021T224346 --evaluador-descentrado \\
        --fondo-c4 shared --metodos aperture,optimal_ls,psffit --partes verifica,e1 --controles-de-produccion
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import replace
from pathlib import Path

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "2")

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import numpy as np  # noqa: E402

from recomputa_e1_e3 import carpetas_prohibidas, comprueba_fuera, parsea_objetivo  # noqa: E402

#: la ventana de E1 (búsqueda ±500 km/s + continuo de 80 Å, con margen) y las bandas de continuo (Å)
VENTANA_HA_A = (6470.0, 6660.0)
BANDAS_CONTINUO_A = {
    "c7150": (7050.0, 7250.0),
    "c8000": (7900.0, 8100.0),
    "c8700": (8600.0, 8800.0),
    "c9100": (9000.0, 9200.0),
}
PASO_CONTINUO = 4
#: los desplazamientos (px) y las direcciones respecto de la primaria
DESPLAZAMIENTOS_PX = (0.1, 0.25, 0.5, 1.0)
DIRECCIONES = ("r_out", "r_in", "t_plus", "t_minus")
#: el anillo de inyección de (a): posiciones, exclusión alrededor del compañero, margen y distancia a otras fuentes
N_INYECCIONES = 40
EXCLUYE_INYECCION_DEG = 35.0
MARGEN_INYECCION_PX = 16
EVITA_FUENTES_PX = 10.0
#: semitamaño del sello de la fuente falsa: por debajo de `norm_radius_px` la PSF se construye en la rejilla pequeña
SEMI_SELLO_PX = 18
#: una banda de la traza «mide» el compañero si la σ de su centroide por inyección no pasa de esto (px)
SIGMA_MEDIDA_PX = 0.15
SEMILLA = 20261028
METODOS = ("aperture", "optimal_ls", "optimal_psfsub", "psffit", "sgf", "lpm")
PARTES = ("verifica", "b3", "traza", "flujo", "e1")


# ---------------------------------------------------------------------------------------------------------------
# Geometría (funciones puras)
# ---------------------------------------------------------------------------------------------------------------

def vectores(comp_yx, prim_yx):
    """(radial hacia fuera, tangencial a +90°) en [dy, dx], unitarios, desde la primaria hacia el compañero."""
    dy = float(comp_yx[0]) - float(prim_yx[0])
    dx = float(comp_yx[1]) - float(prim_yx[1])
    r = math.hypot(dy, dx)
    if r <= 0:
        raise ValueError("el compañero y la primaria están en el mismo sitio")
    er = np.array([dy / r, dx / r])
    return er, np.array([-er[1], er[0]])


def desplazada(comp_yx, prim_yx, direccion, delta_px):
    """La posición del compañero movida `delta_px` en `direccion` (`r_out`, `r_in`, `t_plus`, `t_minus`)."""
    er, et = vectores(comp_yx, prim_yx)
    v = {"r_out": er, "r_in": -er, "t_plus": et, "t_minus": -et}[direccion]
    return (float(comp_yx[0]) + float(delta_px) * v[0], float(comp_yx[1]) + float(delta_px) * v[1])


def traza_relativa(filas, lam_ref_A, clave="primary"):
    """La traza de `clave` (filas de `chromatic_centroids`) relativa a su valor en `lam_ref_A`.

    Devuelve (centros de banda en Å, dy, dx). Entre centros se interpola linealmente y fuera de ellos se toma el
    extremo (`np.interp`), así que en la banda de B3, que cae dentro de la última, la traza vale 0.
    """
    lam = np.array([0.5 * (float(r["wave_min_A"]) + float(r["wave_max_A"])) for r in filas])
    y = np.array([float(r[f"{clave}_y"]) for r in filas])
    x = np.array([float(r[f"{clave}_x"]) for r in filas])
    orden = np.argsort(lam)
    lam, y, x = lam[orden], y[orden], x[orden]
    return lam, y - np.interp(lam_ref_A, lam, y), x - np.interp(lam_ref_A, lam, x)


def desplazamiento_en(lam_A, traza):
    lam, dy, dx = traza
    return float(np.interp(lam_A, lam, dy)), float(np.interp(lam_A, lam, dx))


def claves_bin(wave_A, bin_A):
    """El bin del modelo de PSF de cada canal, con el mismo redondeo que `musepipe.psf` (`round` de Python)."""
    paso = float(bin_A) if bin_A and float(bin_A) > 0 else 100.0
    return np.array([round(float(w) / paso) for w in np.asarray(wave_A, dtype=np.float64)], dtype=np.int64)


def tramos(wave_A, bin_A):
    """[(i0, i1)] contiguos del conjunto de canales que comparten bin del modelo de PSF."""
    k = claves_bin(wave_A, bin_A)
    if k.size == 0:
        return []
    cortes = np.flatnonzero(np.diff(k) != 0) + 1
    ini = np.concatenate([[0], cortes])
    fin = np.concatenate([cortes, [k.size]])
    return [(int(a), int(b)) for a, b in zip(ini, fin)]


def anillo_inyeccion(prim_yx, comp_yx, *, n, excluye_deg, shape, margen_px, evita=(), evita_px=10.0, rng=None):
    """Posiciones subpíxel al radio del compañero, lejos de él y de otras fuentes (para las inyecciones de B3)."""
    sy, sx = map(float, prim_yx)
    radio = math.hypot(float(comp_yx[0]) - sy, float(comp_yx[1]) - sx)
    theta0 = math.atan2(float(comp_yx[0]) - sy, float(comp_yx[1]) - sx)
    out = []
    for k in range(int(n)):
        th = theta0 + 2.0 * math.pi * k / float(n)
        if abs(math.degrees(math.atan2(math.sin(th - theta0), math.cos(th - theta0)))) < float(excluye_deg):
            continue
        y = sy + radio * math.sin(th)
        x = sx + radio * math.cos(th)
        if rng is not None:
            y += float(rng.uniform(-0.5, 0.5))
            x += float(rng.uniform(-0.5, 0.5))
        if not (margen_px <= y < shape[0] - margen_px and margen_px <= x < shape[1] - margen_px):
            continue
        if any(math.hypot(y - float(a[0]), x - float(a[1])) < float(evita_px) for a in evita if a is not None):
            continue
        out.append((float(y), float(x)))
    return out


def estadistica_posicion(verdad, medida):
    """Sesgo (y, x), σ (y, x), σ robusta (1.4826·MAD) y rms 2D de medida − verdad, sobre las filas finitas."""
    v = np.asarray(verdad, dtype=np.float64).reshape(-1, 2)
    m = np.asarray(medida, dtype=np.float64).reshape(-1, 2)
    d = m - v
    ok = np.all(np.isfinite(d), axis=1)
    d = d[ok]
    if d.shape[0] == 0:
        return {"n": 0, "sesgo_yx": None, "sigma_yx": None, "sigma_rob_yx": None, "rms_px": None}
    sig = np.std(d, axis=0, ddof=1) if d.shape[0] > 1 else np.full(2, np.nan)
    rob = 1.4826 * np.median(np.abs(d - np.median(d, axis=0)), axis=0)
    return {"n": int(d.shape[0]), "sesgo_yx": [float(v) for v in np.mean(d, axis=0)],
            "sigma_yx": [float(v) for v in sig], "sigma_rob_yx": [float(v) for v in rob],
            "rms_px": float(np.sqrt(np.mean(np.sum(d ** 2, axis=1))))}


def clasifica_inyecciones(filas, fwhm_px, *, fraccion_fwhm=0.5):
    """Separa lo que B3 encuentra en la fuente inyectada de lo que encuentra en otro sitio.

    B3 no falla solo con una excepción: si la banda roja es ambigua prueba las de respaldo, y ahí puede quedarse con
    otro pico (un residuo del halo) y devolver una posición sin avisar. Una medida a más de `fraccion_fwhm`·FWHM de la
    verdad no es un error de centroide de la fuente: es otra fuente. La precisión (sesgo, σ, rms) se da sobre las
    identificadas, y las otras se cuentan aparte con su distancia y su banda.
    """
    ident, otras, fallos = [], [], {}
    for f in filas:
        if f.get("estado") != "ok":
            fallos[f.get("estado")] = fallos.get(f.get("estado"), 0) + 1
            continue
        d = math.hypot(f["medida_yx"][0] - f["verdad_yx"][0], f["medida_yx"][1] - f["verdad_yx"][1])
        (ident if d <= fraccion_fwhm * float(fwhm_px) else otras).append(dict(f, distancia_px=d))
    est = estadistica_posicion([f["verdad_yx"] for f in ident], [f["medida_yx"] for f in ident])
    return {"identificadas": est, "n_otra_fuente": len(otras),
            "otra_fuente": [{"k": f["k"], "distancia_px": f["distancia_px"], "banda_A": f.get("banda_A")}
                            for f in otras],
            "fallos": fallos, "umbral_px": fraccion_fwhm * float(fwhm_px),
            "bandas_usadas": sorted({tuple(f["banda_A"]) for f in filas if f.get("banda_A")})}


def centro_de_luz(imagen):
    """Centroide de primer momento (y, x) de una imagen: para comprobar dónde pone la luz el evaluador de PSF."""
    img = np.clip(np.nan_to_num(np.asarray(imagen, dtype=np.float64)), 0.0, None)
    yy, xx = np.indices(img.shape, dtype=np.float64)
    tot = float(img.sum())
    return float((yy * img).sum() / tot), float((xx * img).sum() / tot)


# ---------------------------------------------------------------------------------------------------------------
# La fuente falsa (B3) con la PSF de C1
# ---------------------------------------------------------------------------------------------------------------

def sello_psf(psf_model, lam_A, centro_yx, shape, semi=SEMI_SELLO_PX):
    """(slice y, slice x, PSF normalizada a 1 dentro de `norm_radius_px`) en un sello de ±`semi` px."""
    from musepipe.psf import evaluate_psf_model

    cy, cx = float(centro_yx[0]), float(centro_yx[1])
    iy, ix = int(round(cy)), int(round(cx))
    y0, y1 = max(0, iy - semi), min(shape[0], iy + semi + 1)
    x0, x1 = max(0, ix - semi), min(shape[1], ix + semi + 1)
    yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float64)
    p = np.asarray(evaluate_psf_model(psf_model, float(lam_A), yy - cy, xx - cx), dtype=np.float64)
    return slice(y0, y1), slice(x0, x1), np.nan_to_num(p, nan=0.0)


def inyecta(cubo, wave_A, psf_model, espectro, centro_yx, *, bin_A, semi=SEMI_SELLO_PX):
    """Copia de `cubo` (zyx) más una fuente puntual: espectro(λ) × PSF de C1(λ) en `centro_yx`."""
    out = np.array(cubo, copy=True)
    wave = np.asarray(wave_A, dtype=np.float64)
    s = np.asarray(espectro, dtype=np.float64)
    shape = out.shape[1:]
    claves = claves_bin(wave, bin_A)
    for k in np.unique(claves):
        idx = np.flatnonzero(claves == k)
        sy, sx, p = sello_psf(psf_model, float(wave[idx[0]]), centro_yx, shape, semi)
        out[idx, sy, sx] += (s[idx, None, None] * p[None]).astype(out.dtype)
    return out


# ---------------------------------------------------------------------------------------------------------------
# Rutas: el run o una copia con su disposición
# ---------------------------------------------------------------------------------------------------------------

def remapea_si_existe(paths, copia):
    """Las rutas de `stages/` y `tables/` del run que existen en la copia apuntan a la copia; el resto (los cubos,
    que las copias no guardan) siguen en el run."""
    if copia is None:
        return dict(paths)
    rp = paths["paths"]
    out = {}
    for k, v in paths.items():
        if k == "paths" or not isinstance(v, Path):
            out[k] = v
            continue
        nuevo = v
        for origen, destino in ((rp.stage_dir, copia / "stages"), (rp.table_dir, copia / "tables")):
            try:
                cand = destino / v.relative_to(origen)
            except ValueError:
                continue
            if cand.exists():
                nuevo = cand
            break
        out[k] = nuevo
    return out


def usa_evaluador_descentrado():
    """El evaluador de la PSF de antes del 2026-10-21: maoppy sin la compensación de su sobremuestreo."""
    import musepipe.psf as psf

    psf.MAOPPY_CENTRADO_PX = 0.0
    psf._psfao_image_cached.cache_clear()
    psf._mixture_image_cached.cache_clear()


def calienta_psf(psf_model, wave_A, bin_A):
    """Construye una vez por bin las dos rejillas del modelo (la de la apcorr y la de psffit), antes de bifurcar."""
    from musepipe.psf import evaluate_psf_model

    wave = np.asarray(wave_A, dtype=np.float64)
    claves = claves_bin(wave, bin_A)
    for k in np.unique(claves):
        lam = float(wave[np.flatnonzero(claves == k)[0]])
        evaluate_psf_model(psf_model, lam, np.zeros(1), np.zeros(1))
        evaluate_psf_model(psf_model, lam, np.array([0.0, 100.0]), np.array([0.0, 100.0]))


# ---------------------------------------------------------------------------------------------------------------
# (a) + (b): B3
# ---------------------------------------------------------------------------------------------------------------

_B3 = {}


def espectro_falso(stage_dir_o_producto, wave_A):
    """El continuo del compañero según C4 en la escala del cubo (amplitud NORMRAD = flujo / apcorr), suavizado."""
    from musepipe.extraction.product import SpectrumProduct
    from musepipe.spectral import continuum_running_median

    p = SpectrumProduct.read(stage_dir_o_producto)
    if p.wave_A.size != np.asarray(wave_A).size:
        raise RuntimeError("el producto de C4 no tiene el eje λ del cubo de B3")
    crudo = np.asarray(p.flux, dtype=np.float64) / np.asarray(p.apcorr, dtype=np.float64)
    cont = continuum_running_median(np.asarray(wave_A, dtype=np.float64), crudo, np.isfinite(crudo),
                                    window_A=150.0, min_pixels=15)
    return np.clip(np.nan_to_num(cont, nan=0.0), 0.0, None)


def _inyeccion_b3(k):
    from musepipe.stages.stage01c_localize import (AmbiguousDetectionError, SourceNotDetectedError,
                                                  chromatic_centroids, locate_companion)

    g = _B3
    verdad = g["anillo"][k]
    cubo = inyecta(g["cubo"], g["wave"], g["psf"], g["espectro"], verdad, bin_A=g["bin_A"])
    cfg = dict(g["cfg"], expected_sep_arcsec=None, expected_pa_deg=None,
               stage01c_companion_approx_yx=[verdad[0] + g["error_prediccion"][0],
                                             verdad[1] + g["error_prediccion"][1]])
    fila = {"k": int(k), "verdad_yx": list(verdad)}
    try:
        det, _pred = locate_companion(cubo, g["wave"], g["primaria"], cfg, g["escala"], g["norte"], g["fwhm"])
        fila.update(estado="ok", medida_yx=list(det.pos_yx), snr=float(det.snr), err_formal_px=float(det.err_px),
                    banda_A=list(det.band_used_A))
    except (AmbiguousDetectionError, SourceNotDetectedError) as exc:
        fila.update(estado=type(exc).__name__, medida_yx=[float("nan")] * 2, motivo=str(exc)[:200])
    filas, _ = chromatic_centroids(cubo, g["wave"], g["primaria"]["pos_yx"], verdad,
                                   dict(g["cfg"], stage01c_chromatic_bins=g["n_bandas"]), g["fwhm"])
    fila["bandas"] = [[r["companion_y"], r["companion_x"]] for r in filas]
    return fila


def centroide_ha_e2(cfg, paths, copia, comp_yx):
    """Dónde pone E2 (T2) la emisión del canal de Hα, respecto de la posición de B3, en px (y, x).

    T2 (`stage_h02_artifacts.t2_spatial_coherence`) da el primer momento de los píxeles positivos de un sello de
    ±`h02_stamp_half_size_px` centrado en el píxel redondeado de B3 (`_stamp`), en las coordenadas del sello; aquí
    se pasa a desplazamiento respecto de B3 con la misma cuenta que `_stamp`. Es la única medida de la posición de
    la línea que hace la cadena (un canal, sobre el residuo de 04b).
    """
    from musepipe.io import read_json

    ruta = remapea_si_existe({"paths": paths["paths"], "q": paths["paths"].stage_dir / "stage_h02_qc.json"},
                             copia)["q"]
    if not Path(ruta).exists():
        return None
    t2 = (read_json(ruta).get("t2") or {})
    if t2.get("centroid_y") is None:
        return None
    from musepipe.stages.stage_h02_artifacts import stage_h02_config_from_run

    cfg2 = stage_h02_config_from_run(cfg["run_id"], project_root=cfg.get("project_root"))
    semi = int(cfg2.get("h02_stamp_half_size_px", 5))
    y, x = float(comp_yx[0]), float(comp_yx[1])
    loc = (y - max(0, int(round(y)) - semi), x - max(0, int(round(x)) - semi))
    d = (float(t2["centroid_y"]) - loc[0], float(t2["centroid_x"]) - loc[1])
    return {"qc": str(ruta), "canal": t2.get("signal_channel"), "d_yx": list(d), "d_px": math.hypot(*d),
            "centroid_offset_px_qc": t2.get("centroid_offset_px"), "psf_amplitude_snr": t2.get("psf_amplitude_snr")}


def mide_b3(run_id, project_root, *, copia, n_inyecciones, n_bandas, n_procesos, semilla, partes):
    """(a) y (b): reproducción de B3, traza cromática real y precisión por inyección."""
    import csv
    from multiprocessing import get_context

    from musepipe.io import read_json
    from musepipe.psf import _psfao_wave_bin_A
    from musepipe.stages.stage01c_localize import (_select_input_cube_path, chromatic_centroids,
                                                  compute_stage01c_products, load_cube_stack, measure_primary,
                                                  resolve_pixel_scale_and_orientation, stage01c_config_from_run,
                                                  stage01c_paths)

    t0 = time.time()
    cfg = stage01c_config_from_run(run_id, project_root=project_root)
    paths = stage01c_paths(cfg["run_id"], project_root=project_root)
    qc_path = remapea_si_existe({"paths": paths["paths"], "qc": paths["paths"].stage_dir / "stage01c_qc.json"},
                                copia)["qc"]
    qc = read_json(qc_path)
    out = {"qc_b3": str(qc_path)}

    # Reproducción de B3 entero (compute_stage01c_products no escribe nada)
    prod = compute_stage01c_products(cfg)
    comp_qc = [float(v) for v in qc["companion"]["pos_yx"]]
    rep = [float(v) for v in prod.qc["companion"]["pos_yx"]]
    csv_path = paths["paths"].table_dir / "stage01c_chromatic_centroids.csv"
    disco = list(csv.DictReader(open(csv_path, newline="", encoding="utf-8"))) if csv_path.exists() else []
    dif_csv = None
    if disco and len(disco) == len(prod.chromatic_rows):
        dif_csv = max(abs(float(a[k]) - float(b[k])) for a, b in zip(disco, prod.chromatic_rows)
                      for k in ("primary_y", "primary_x", "companion_y", "companion_x"))
    out["reproduccion"] = {"companion_qc_yx": comp_qc, "companion_recalculado_yx": rep,
                           "max_dif_px": float(max(abs(a - b) for a, b in zip(comp_qc, rep))),
                           "csv_cromatico_max_dif_px": dif_csv, "n_bandas_csv": len(disco)}
    print(f"   B3 recalculado {rep} frente al QC {comp_qc}; CSV cromático {dif_csv}")

    cube_path, _src = _select_input_cube_path(cfg, paths)
    _cubes, cubo, wave, h0, hc = load_cube_stack(cube_path)
    del _cubes
    escala, norte, _s1, _s2 = resolve_pixel_scale_and_orientation(cfg, h0, hc)
    primaria = measure_primary(cubo, wave, cfg)
    fwhm = float(qc["psf"]["fwhm_px"])
    prim_yx = tuple(primaria["pos_yx"])
    lam_b3 = 0.5 * sum(float(v) for v in cfg["stage01c_red_band_A"])

    # (b) traza fina en el cubo real
    filas_n, _ = chromatic_centroids(cubo, wave, prim_yx, comp_qc, dict(cfg, stage01c_chromatic_bins=n_bandas), fwhm)
    out["traza"] = {"n_bandas": int(n_bandas), "lam_b3_A": lam_b3, "filas": filas_n,
                    "filas_run": prod.chromatic_rows}
    out["centroide_ha_e2"] = centroide_ha_e2(cfg, paths, copia, comp_qc)

    if "b3" in partes or "traza" in partes:
        psf_path = remapea_si_existe({"paths": paths["paths"], "p": paths["paths"].stage_dir / "psf_model.json"},
                                     copia)["p"]
        psf_model = read_json(psf_path)
        c4_path = remapea_si_existe({"paths": paths["paths"],
                                     "p": paths["paths"].stage_dir / "spec_psffit_object.fits"}, copia)["p"]
        rng = np.random.default_rng(semilla)
        evita = [qc["field_source"]["pos_yx"]] if qc.get("field_source") else []
        anillo = anillo_inyeccion(prim_yx, comp_qc, n=n_inyecciones, excluye_deg=EXCLUYE_INYECCION_DEG,
                                  shape=cubo.shape[1:], margen_px=MARGEN_INYECCION_PX, evita=evita,
                                  evita_px=EVITA_FUENTES_PX, rng=rng)
        pred = qc["companion"].get("predicted_pos_yx") or comp_qc
        bin_A = _psfao_wave_bin_A(psf_model) if str(psf_model.get("form", "")).lower() in {"psfao", "mixture"} else 0.0
        _B3.clear()
        _B3.update(cubo=cubo, wave=wave, psf=psf_model, espectro=espectro_falso(c4_path, wave), anillo=anillo,
                   cfg=cfg, primaria=primaria, escala=escala, norte=norte, fwhm=fwhm, bin_A=bin_A,
                   n_bandas=int(n_bandas),
                   error_prediccion=(float(pred[0]) - comp_qc[0], float(pred[1]) - comp_qc[1]))
        for k in np.unique(claves_bin(wave, bin_A)):  # la rejilla pequeña de cada bin, antes de bifurcar
            sello_psf(psf_model, float(wave[np.flatnonzero(claves_bin(wave, bin_A) == k)[0]]), comp_qc,
                      cubo.shape[1:])
        with get_context("fork").Pool(max(1, int(n_procesos))) as pool:
            filas = pool.map(_inyeccion_b3, range(len(anillo)), chunksize=1)
        ok = [f for f in filas if f["estado"] == "ok"]
        est = estadistica_posicion([f["verdad_yx"] for f in ok], [f["medida_yx"] for f in ok])
        clas = clasifica_inyecciones(filas, fwhm)
        out["inyeccion"] = {"n_posiciones": len(anillo), "radio_px": math.hypot(comp_qc[0] - prim_yx[0],
                                                                               comp_qc[1] - prim_yx[1]),
                            "error_prediccion_yx": list(_B3["error_prediccion"]), "fwhm_px": fwhm,
                            "estadistica_todas": est, "clasificacion": clas, "estadistica": clas["identificadas"],
                            "fallos": clas["fallos"], "err_formal_qc_px": float(qc["companion"]["err_px"]),
                            "snr_qc": float(qc["companion"]["snr_detection"]), "filas": filas}
        est = clas["identificadas"]
        fallos = dict(clas["fallos"], otra_fuente=clas["n_otra_fuente"])
        # σ de cada banda de la traza, con TODAS las inyecciones: el centroide por banda arranca en la verdad, como
        # `chromatic_centroids` arranca en la posición de B3, así que no depende de lo que haya encontrado B3
        por_banda = []
        for i, r in enumerate(filas_n):
            est_b = estadistica_posicion([f["verdad_yx"] for f in filas], [f["bandas"][i] for f in filas])
            por_banda.append({"wave_min_A": r["wave_min_A"], "wave_max_A": r["wave_max_A"], **est_b})
        out["traza"]["sigma_inyeccion"] = por_banda
        print(f"   inyección B3: {est['n']}/{len(anillo)} ok, sesgo {est['sesgo_yx']}, σ {est['sigma_yx']}, "
              f"rms {est['rms_px']:.4f} px; fallos {fallos}")
        _B3.clear()
    out["segundos"] = time.time() - t0
    return out


# ---------------------------------------------------------------------------------------------------------------
# (c) + (d): extracción con la posición movida, D2 y E1
# ---------------------------------------------------------------------------------------------------------------

def _corta(a, idx, eje=0):
    if a is None:
        return None
    return np.take(np.asarray(a), idx, axis=eje)


def _sl(a, sl):
    return None if a is None else a[sl]


def concatena(productos):
    """Un `SpectrumProduct` con los tramos uno detrás de otro (las columnas por canal; la cabecera del primero)."""
    if len(productos) == 1:
        return productos[0]
    p0 = productos[0]
    cat = lambda nombre: np.concatenate([np.asarray(getattr(p, nombre)) for p in productos])  # noqa: E731
    extra = None
    if p0.extra_columns:
        extra = {k: np.concatenate([np.asarray(p.extra_columns[k]) for p in productos]) for k in p0.extra_columns}
    return replace(p0, wave_A=cat("wave_A"), flux=cat("flux"), flux_err=cat("flux_err"),
                   flux_err_emp=cat("flux_err_emp"), apcorr=cat("apcorr"), npix_eff=cat("npix_eff"),
                   flags=cat("flags"), covariance=None, extra_columns=extra)


class Entradas:
    """Las entradas de C2–C6 de un run, con las funciones de cada etapa, recortadas a unos canales.

    C2 y C3 dan las suyas (`x01_extraction_inputs`, `x02_extraction_inputs`, las mismas que usa E4); para C4, C5 y C6
    se llaman los cargadores que usan `compute_stage_x03_products` y `extract_halosub_product`. `extrae` llama a los
    extractores de producción (`extract_aperture_products`, `make_optimal_product`, `make_psffit_products`,
    `make_aperture_product`) con la posición que se le pida.
    """

    def __init__(self, run_id, project_root, idx, *, metodos, copia=None, fondo_c4=None):
        from musepipe.extraction.aperture import aperture_label
        from musepipe.stages import halosub_stage as hs
        from musepipe.stages import stage_x02_optimal as x02
        from musepipe.stages import stage_x03_psffit as x03
        from musepipe.stages.stage_x01_aperture import (stage_x01_config_from_run, stage_x01_paths,
                                                        x01_extraction_inputs)
        from musepipe.stages.stage_x04_sgf import stage_x04_config_from_run, stage_x04_paths
        from musepipe.stages.stage_x05_lpm import stage_x05_config_from_run, stage_x05_paths

        self.idx = np.asarray(idx)
        self.metodos = tuple(metodos)
        self.c = {}
        self.psf = {}
        root = project_root
        if "aperture" in metodos:
            cfg = stage_x01_config_from_run(run_id, project_root=root)
            inp = x01_extraction_inputs(cfg, remapea_si_existe(stage_x01_paths(cfg["run_id"], root), copia))
            box3 = [ap for ap in inp.kwargs["apertures"] if aperture_label(ap) == "box3"][:1]
            if not box3:
                raise RuntimeError("C2 no declara box3")
            kw = dict(inp.kwargs, apertures=box3, input_cube_path="en_memoria",
                      stat_zyx=_corta(inp.kwargs["stat_zyx"], self.idx),
                      good_mask=_corta(inp.kwargs["good_mask"], self.idx),
                      bad_mask=_corta(inp.kwargs["bad_mask"], self.idx))
            self.c["aperture"] = (_corta(inp.cube, self.idx), kw)
            self.psf["aperture"] = kw["psf_model"]
            self.wave = _corta(inp.wave, self.idx)
            self.comp_yx, self.star_yx = tuple(inp.object_yx), tuple(inp.star_yx)
            del inp
        if {"optimal_ls", "optimal_psfsub"} & set(metodos):
            cfg = x02.stage_x02_config_from_run(run_id, project_root=root)
            p = remapea_si_existe(x02.stage_x02_paths(cfg["run_id"], root), copia)
            inp = x02.x02_extraction_inputs(cfg, p)
            common = dict(inp.common, variance_zyx=_corta(inp.common["variance_zyx"], self.idx),
                          good_mask=_corta(inp.common["good_mask"], self.idx),
                          bad_mask=_corta(inp.common["bad_mask"], self.idx))
            self.psf["optimal"] = inp.psf_model
            self.wave = _corta(inp.wave, self.idx)
            self.comp_yx, self.star_yx = tuple(inp.object_yx), tuple(inp.star_yx)
            if "optimal_ls" in metodos:
                self.c["optimal_ls"] = (_corta(inp.ls_cube, self.idx), dict(common))
            if "optimal_psfsub" in metodos:
                # lo que hace compute_stage_x02_products con el cubo de C1b
                perobs = Path(cfg.get("x02_psfsub_cube_fits") or p["paths"].stage_dir / "cube_psfsub_perobs.fits")
                if not (bool(cfg.get("x02_psfsub_per_observation", True)) and perobs.exists()):
                    raise SystemExit("optimal_psfsub sin el cubo de C1b: la etapa ajustaría la primaria en el "
                                     "combinado, y esta herramienta no lo replica")
                cubo, _w, _b = x02._load_perobs_psfsub_cube(perobs, expected_shape=inp.stage02_cube.shape,
                                                            expected_wave=inp.stage02_wave)
                psc = dict(common, bunit=inp.bunit or inp.stage02_bunit, input_cube_source=x02.PEROBS_CUBE_SOURCE)
                if bool(cfg.get("x02_wings_intact_psfsub", False)) and psc.get("local_bkg_annulus_px") is not None:
                    psc["local_bkg_annulus_px"] = None
                self.c["optimal_psfsub"] = (_corta(cubo, self.idx), psc)
                del cubo
            del inp
        if "psffit" in metodos:
            cfg = x03.stage_x03_config_from_run(run_id, project_root=root)
            if fondo_c4:
                cfg["x03_background"] = str(fondo_c4)
            p = remapea_si_existe(x03.stage_x03_paths(cfg["run_id"], root), copia)
            comp_yx, star_yx, _pp, _pq = x03._load_positions(
                {**p, "stage01c_qc_json": Path(cfg.get("x03_positions_qc", p["stage01c_qc_json"]))},
                {"x01_source_key": cfg.get("x03_source_key", "companion"),
                 "x01_star_key": cfg.get("x03_star_key", "primary")})
            psf_model, _pm = x03._load_psf_model(p, cfg)
            cubo, wave, _cp, bunit = x03._load_stage02_cube(p, cfg)
            stat, _ss = x03._load_stat_cube(p, cfg, cubo.shape)
            sf, cf, st, qc00, _q1 = x03._stat_metadata(p, cfg)
            good, bad = x03._load_masks(p, wave.size)
            wframe = x03._wavelength_frame(cfg, qc00, [], knob="x03_wframe")
            gc, _conv = x03.resolve_flux_convention(cfg, p["paths"].stage_dir, knob="x03_flux_convention")
            kw = dict(growth_curve=gc, run_id=cfg["run_id"], input_cube_path="en_memoria", input_cube_sha="",
                      variance_zyx=_corta(stat, self.idx), stat_factor_spaxel=sf, covariance_factor_box3=cf,
                      stat_status=st, error_mode=cfg.get("x03_error_mode", "auto"), wframe=wframe,
                      bunit=bunit or cfg.get("x03_bunit", ""),
                      star_radius_px=float(cfg.get("x03_star_radius_px", 20.0)),
                      comp_radius_px=float(cfg.get("x03_comp_radius_px", 12.0)),
                      bad_windows_A=cfg.get("x03_bad_windows_A", []),
                      skyline_windows_A=cfg.get("x03_skyline_windows_A", []),
                      interpolated_windows_A=cfg.get("x03_interpolated_windows_A", []),
                      good_mask=_corta(good, self.idx), bad_mask=_corta(bad, self.idx),
                      n_controls=int(cfg.get("x03_control_apertures", 8)),
                      exclude_angle_deg=float(cfg.get("x03_control_exclude_angle_deg", 25.0)),
                      background=cfg.get("x03_background", "shared"))
            self.c["psffit"] = (_corta(cubo, self.idx), psf_model, tuple(star_yx), kw)
            self.psf["psffit"] = psf_model
            self.fondo_c4 = kw["background"]
            self.wave = _corta(wave, self.idx)
            self.comp_yx, self.star_yx = tuple(comp_yx), tuple(star_yx)
            del cubo, stat
        for metodo, config_de, paths_de in (("sgf", stage_x04_config_from_run, stage_x04_paths),
                                            ("lpm", stage_x05_config_from_run, stage_x05_paths)):
            if metodo not in metodos:
                continue
            from astropy.io import fits

            pre = {"sgf": "x04", "lpm": "x05"}[metodo]
            cfg = config_de(run_id, project_root=root)
            p = remapea_si_existe(paths_de(cfg["run_id"], root), copia)
            comp_yx, star_yx, _pp, _pq = hs._load_positions(p, cfg)
            with fits.open(p["residual_cube_fits"], memmap=True) as h:
                forma = tuple(h["RESIDUAL"].shape)
                resid = np.asarray(h["RESIDUAL"].data[self.idx], dtype=np.float64)
                wave = np.asarray(h["WAVELENGTH"].data, dtype=np.float64)
            psf_model = read_json_(p["psf_model_json"]) if p["psf_model_json"].exists() else None
            stat, _ss = hs._load_stat_cube(p, cfg, forma)
            sf, cf, st, qc00, _q1 = hs._stat_metadata(p, cfg)
            wframe = hs._wavelength_frame(cfg, qc00, [])
            gc, _conv = hs.resolve_flux_convention(cfg, p["paths"].stage_dir, knob=f"{pre}_flux_convention")
            kw = dict(run_id=cfg["run_id"], input_cube_path="en_memoria", star_yx=star_yx,
                      stat_zyx=_corta(stat, self.idx), stat_factor=sf, covariance_factor=cf, stat_status=st,
                      error_mode=cfg.get(f"{pre}_error_mode", "auto"), psf_model=psf_model,
                      aperture_correction=cfg.get(f"{pre}_aperture_correction", "auto"), growth_curve=gc,
                      wframe=wframe, bunit=hs.resolve_bunit(cfg, override_key="cube_bunit"),
                      bad_windows_A=cfg.get(f"{pre}_bad_windows_A", []),
                      skyline_windows_A=cfg.get(f"{pre}_skyline_windows_A", []),
                      interpolated_windows_A=cfg.get(f"{pre}_interpolated_windows_A", []),
                      n_controls=int(cfg.get(f"{pre}_control_apertures", 8)),
                      exclude_angle_deg=float(cfg.get(f"{pre}_control_exclude_angle_deg", 25.0)),
                      annulus_bkg_px=cfg.get(f"{pre}_annulus_bkg_px"))
            self.c[metodo] = (resid, kw)
            self.psf[metodo] = psf_model
            self.wave = _corta(wave, self.idx)
            self.comp_yx, self.star_yx = tuple(comp_yx), tuple(star_yx)
            del stat

    def extrae(self, metodo, tramos_pos, *, con_controles=True):
        """(producto, controles calibrados de C o None) del método con la posición de cada tramo.

        `tramos_pos`: [(i0, i1, (y, x))] en índices de los canales recortados."""
        from musepipe.extraction.aperture import extract_aperture_products, make_aperture_product
        from musepipe.extraction.optimal import make_optimal_product
        from musepipe.extraction.psffit import make_psffit_products
        from musepipe.stages.halosub_stage import BOX3_APERTURE

        prods, ctrls = [], []
        for i0, i1, yx in tramos_pos:
            sl = slice(int(i0), int(i1))
            w = self.wave[sl]
            yx = (float(yx[0]), float(yx[1]))
            if metodo == "aperture":
                cubo, kw = self.c[metodo]
                k = dict(kw, stat_zyx=_sl(kw["stat_zyx"], sl), good_mask=_sl(kw["good_mask"], sl),
                         bad_mask=_sl(kw["bad_mask"], sl), n_controls=kw["n_controls"] if con_controles else 0)
                ext = next(iter(extract_aperture_products(cubo[sl], w, yx, **k).values()))
                prod, ctl = ext.product, ext.control_spectra_cal
            elif metodo in ("optimal_ls", "optimal_psfsub"):
                cubo, kw = self.c[metodo]
                k = dict(kw, variance_zyx=_sl(kw["variance_zyx"], sl), good_mask=_sl(kw["good_mask"], sl),
                         bad_mask=_sl(kw["bad_mask"], sl), n_controls=kw["n_controls"] if con_controles else 0)
                ext = make_optimal_product(cubo[sl], w, yx, self.psf["optimal"], input_cube_path="en_memoria",
                                           input_cube_sha="", variant="ls" if metodo == "optimal_ls" else "psfsub",
                                           **k)
                prod, ctl = ext.product, ext.control_spectra_cal
            elif metodo == "psffit":
                cubo, psf_model, star_yx, kw = self.c[metodo]
                k = dict(kw, variance_zyx=_sl(kw["variance_zyx"], sl), good_mask=_sl(kw["good_mask"], sl),
                         bad_mask=_sl(kw["bad_mask"], sl), n_controls=kw["n_controls"] if con_controles else 0)
                res = make_psffit_products(cubo[sl], w, star_yx, yx, psf_model, **k)
                prod, ctl = res.companion, res.control_comp_spectra
            else:
                cubo, kw = self.c[metodo]
                k = dict(kw, stat_zyx=_sl(kw["stat_zyx"], sl), n_controls=kw["n_controls"] if con_controles else 0)
                ext = make_aperture_product(cubo[sl], w, yx, dict(BOX3_APERTURE), **k)
                ext.product.header["METHOD"] = metodo
                ext.product.header["BKGMODE"] = f"{metodo}_residual+{ext.bkg_mode}"
                prod, ctl = ext.product, ext.control_spectra_cal
            prods.append(prod)
            ctrls.append(None if ctl is None else np.asarray(ctl, dtype=np.float64))
        controles = None
        if con_controles:
            if len({c.shape[0] for c in ctrls}) != 1:
                raise RuntimeError(f"{metodo}: el número de controles cambia entre tramos")
            controles = np.concatenate(ctrls, axis=1)
        return concatena(prods), controles


def read_json_(path):
    from musepipe.io import read_json

    return read_json(path)


class Calibracion:
    """D2 y E1 de un run (o de una copia): las correcciones, los knobs y lo que hay en disco para comparar."""

    def __init__(self, run_id, project_root, *, copia=None):
        from musepipe.io import load_calibrated_controls, read_json
        from musepipe.extraction.product import SpectrumProduct
        from musepipe.stages import stage_h01_detect as h01
        from musepipe.stages import stage_x11_calibrate as x11

        cfg11 = x11.stage_x11_config_from_run(run_id, project_root=project_root)
        p11 = remapea_si_existe(x11.stage_x11_paths(cfg11["run_id"], project_root), copia)
        qc = lambda clave: x11._read_optional_json(Path(cfg11.get(f"x11_{clave}", p11[clave])))  # noqa: E731
        qc00 = read_json(Path(cfg11.get("x11_stage00q_qc_json", p11["stage00q_qc_json"])))
        self.corr = x11.calibration_corrections_from_qc(qc00, qc_psf=qc("stage_e01_qc_json"),
                                                        qc_sky=qc("stage00s_qc_json"),
                                                        qc_telluric=qc("stage00t_qc_json"), config=cfg11)
        self.canonico = x11._resolve_canonical_method(cfg11, qc("stage_x10_qc_json"))
        self.cal_kw = dict(continuum_window_A=float(cfg11.get("x11_continuum_window_A", 80.0)),
                           continuum_poly_deg=int(cfg11.get("x11_continuum_poly_deg", 5)),
                           error_smooth_channels=int(cfg11.get("x11_error_smooth_channels", 21)))
        cfg = h01.stage_h01_config_from_run(run_id, project_root=project_root)
        ph = remapea_si_existe(h01.stage_h01_paths(cfg["run_id"], project_root), copia)
        qc00h = read_json(Path(cfg.get("h01_stage00q_qc_json", ph["stage00q_qc_json"])))
        self.lsf, self.lsf_fuente = h01._lsf_fwhm_from_qc_or_config(qc00h, cfg)
        self.rv, _ = h01._rv_from_config(cfg)
        self.rv_err = h01._rv_err_from_config(cfg)
        self.h01 = cfg
        self.criterio = {"detection_fap": float(cfg.get("h01_detection_fap", 0.01)),
                         "admissible_pairs": [tuple(p) for p in cfg.get("h01_admissible_pairs",
                                                                        h01.DEFAULT_ADMISSIBLE_PAIRS)],
                         "fap_estimator": str(cfg.get("h01_fap_estimator", "empirical"))}
        # Lo que hay en disco (o en la copia), para verificar y para `--controles-de-produccion`
        self.prod_paths = {m: Path(v) for m, v in x11._product_paths_from_config(cfg11, p11).items()}
        self.cal_paths = h01._product_paths_from_config(cfg, ph)
        self.ctl_paths = h01._control_paths_from_config(cfg, ph)
        self.stage_dir = Path(p11["spec_psffit_object"]).parent
        self.e1_csv = Path(ph["halpha_detection_csv"])
        self.e1_qc = Path(ph["stage_h01_qc_json"])
        self._SpectrumProduct = SpectrumProduct
        self._load_ctl = load_calibrated_controls

    def producto_disco(self, metodo):
        return self._SpectrumProduct.read(self.prod_paths[metodo])

    def controles_crudos_disco(self, metodo):
        from musepipe.stages.stage_x11_calibrate import _RAW_CONTROL_NPZ

        with np.load(self.stage_dir / _RAW_CONTROL_NPZ[metodo], allow_pickle=False) as d:
            return np.asarray(d["control_spectra"], dtype=np.float64)

    def controles_calibrados_disco(self, metodo):
        return self._load_ctl(self.ctl_paths[metodo], object_path=self.cal_paths[metodo])

    def calibra(self, metodo, producto):
        from musepipe.stages.stage_x11_calibrate import calibrate_spectrum_product

        return calibrate_spectrum_product(producto, self.corr, method=metodo, canonical=metodo == self.canonico,
                                          **self.cal_kw).product

    def e1(self, productos, controles, *, n_boot=0, plantilla=None):
        """Filas de E1 y veredicto. `controles` en la escala de C (se calibran aquí con `flux_scale`)."""
        from scipy import stats

        from musepipe.stages.stage_h01_detect import (DEFAULT_TEMPLATE_WIDTH_FACTORS, HALPHA_REST_A,
                                                     _good_detection_mask, _object_residual,
                                                     analyze_halpha_method, classify_h01_verdict,
                                                     matched_filter_point)

        cfg = self.h01
        filas = []
        for m, prod in productos.items():
            cal = self.calibra(m, prod)
            res = analyze_halpha_method(
                m, cal, np.asarray(controles[m], dtype=np.float64) * float(self.corr.flux_scale),
                rv_sys_kms=self.rv, rv_sys_err_kms=self.rv_err, lsf_fwhm_A=self.lsf,
                rest_A=float(cfg.get("h01_halpha_rest_A", HALPHA_REST_A)),
                search_half_width_kms=float(cfg.get("h01_search_half_width_kms", 500.0)),
                width_factors=cfg.get("h01_template_width_factors", DEFAULT_TEMPLATE_WIDTH_FACTORS),
                continuum_window_A=float(cfg.get("h01_continuum_window_A", 80.0)),
                parametric_family=str(cfg.get("h01_fap_family", "gumbel")), parametric_n_boot=int(n_boot),
                parametric_seed=cfg.get("h01_fap_seed", 20260902))
            fila = dict(res.row)
            par = fila.pop("_parametric")
            fila["gumbel_loc"], fila["gumbel_scale"] = par.get("loc"), par.get("scale")
            fila["fap_bootstrap_95"] = par.get("ci95")
            fila["z_umbral_parametrico"] = (None if par.get("loc") is None else
                                            float(stats.gumbel_r.isf(self.criterio["detection_fap"],
                                                                     par["loc"], par["scale"])))
            ext = cal.extra_columns
            fila["cont_runmed_en_pico"] = float(np.interp(fila["peak_wave_A"], cal.wave_A, ext["cont_runmed"]))
            if plantilla and m in plantilla:
                c, fw = plantilla[m]
                err = np.asarray(ext.get("flux_err_total", cal.flux_err), dtype=np.float64)
                f, s, z = matched_filter_point(cal.wave_A, _object_residual(cal), err, c, fw,
                                               _good_detection_mask(cal))
                fila["linea_plantilla_fija"] = {"centro_A": c, "fwhm_A": fw, "flujo": float(f), "sigma": float(s),
                                                "z": float(z)}
                fila["cont_runmed_plantilla"] = float(np.interp(c, cal.wave_A, ext["cont_runmed"]))
            filas.append(fila)
        veredicto = classify_h01_verdict(filas, detection_fap=self.criterio["detection_fap"],
                                         admissible_pairs=self.criterio["admissible_pairs"],
                                         fap_estimator=self.criterio["fap_estimator"])
        completo = set(productos) == set(METODOS)
        return {"filas": {f["method"]: _limpia(f) for f in filas}, "veredicto": veredicto,
                "veredicto_completo": completo}


def _limpia(d):
    out = {}
    for k, v in d.items():
        if isinstance(v, (np.floating, np.integer)):
            v = v.item()
        out[k] = v
    return out


# ---------------------------------------------------------------------------------------------------------------
# Los casos
# ---------------------------------------------------------------------------------------------------------------

_G = {}


def casos(comp_yx, prim_yx, trazas):
    """{nombre: posición} — una (y, x) acromática o ('traza', clave)."""
    out = {"cero": (float(comp_yx[0]), float(comp_yx[1]))}
    for d in DESPLAZAMIENTOS_PX:
        for direc in DIRECCIONES:
            out[f"{direc}_{d:g}"] = desplazada(comp_yx, prim_yx, direc, d)
    for clave in trazas:
        out[clave] = ("traza", clave)
    return out


def tramos_de_caso(caso, wave_sel, bin_A, comp_yx, trazas):
    """[(i0, i1, (y, x))] de un caso sobre unos canales: un solo tramo si es acromático."""
    if not (isinstance(caso, tuple) and len(caso) == 2 and caso[0] == "traza"):
        return [(0, int(np.asarray(wave_sel).size), caso)]
    out = []
    for i0, i1 in tramos(wave_sel, bin_A):
        dy, dx = desplazamiento_en(float(np.mean(wave_sel[i0:i1])), trazas[caso[1]])
        out.append((i0, i1, (float(comp_yx[0]) + dy, float(comp_yx[1]) + dx)))
    return out


def _mediana_banda(prod, lo, hi):
    w = np.asarray(prod.wave_A)
    f = np.asarray(prod.flux, dtype=np.float64)
    sel = (w >= lo) & (w <= hi) & np.isfinite(f)
    return float(np.nanmedian(f[sel])) if np.any(sel) else None


def _caso(nombre):
    g = _G
    caso = g["casos"][nombre]
    t = time.time()
    ent_ha, ent_c, cal = g["ent_ha"], g.get("ent_c"), g["cal"]
    prods, ctls = {}, {}
    pos_ha = tramos_de_caso(caso, ent_ha.wave, g["bin_A"], g["comp_yx"], g["trazas"])
    for m in g["metodos"]:
        prods[m], ctls[m] = ent_ha.extrae(m, pos_ha, con_controles=True)
    out = {"posiciones_ha": [[float(np.mean(ent_ha.wave[a:b])), list(yx)] for a, b, yx in pos_ha],
           "e1": cal.e1(prods, ctls, plantilla=g.get("plantilla"))}
    if g.get("controles_produccion"):
        prods_p = {m: replace(p, flux_err=g["disco_err"][m][0], flux_err_emp=g["disco_err"][m][1])
                   for m, p in prods.items()}
        out["e1_controles_produccion"] = cal.e1(prods_p, g["disco_ctl"], plantilla=g.get("plantilla"))
    if ent_c is not None:
        pos_c = tramos_de_caso(caso, ent_c.wave, g["bin_A"], g["comp_yx"], g["trazas"])
        cont = {}
        for m in g["metodos"]:
            p, _ = ent_c.extrae(m, pos_c, con_controles=False)
            cont[m] = {b: _mediana_banda(p, lo, hi) for b, (lo, hi) in BANDAS_CONTINUO_A.items()}
        out["continuo"] = cont
    out["segundos"] = time.time() - t
    if g.get("devuelve_productos"):
        out["_productos"] = prods
        out["_controles"] = ctls
    return nombre, out


def _max_rel(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    ok = np.isfinite(a) & np.isfinite(b)
    if not np.any(ok):
        return None
    escala = float(np.nanmedian(np.abs(b[ok]))) or 1.0
    return float(np.max(np.abs(a[ok] - b[ok])) / escala)


def verifica_cero(res0, cal, idx, metodos):
    """δ = 0 frente al disco: producto de C (en la ventana), controles, controles calibrados de D2 y E1."""
    import csv

    out = {"productos": {}, "e1": {}}
    for m in metodos:
        prod, ctl = res0["_productos"][m], res0["_controles"][m]
        disco = cal.producto_disco(m)
        d = {c: _max_rel(getattr(prod, c), np.asarray(getattr(disco, c))[idx])
             for c in ("flux", "flux_err", "flux_err_emp", "apcorr")}
        crudo = cal.controles_crudos_disco(m)
        d["controles"] = _max_rel(ctl, crudo[:, idx]) if crudo.shape[0] == ctl.shape[0] else (
            f"n distinto: {ctl.shape[0]} frente a {crudo.shape[0]}")
        calib = cal.controles_calibrados_disco(m)
        d["controles_calibrados_d2"] = (_max_rel(ctl * float(cal.corr.flux_scale), calib[:, idx])
                                        if calib.shape[0] == ctl.shape[0] else "n distinto")
        d["n_controles"] = int(ctl.shape[0])
        out["productos"][m] = d
    tabla = {}
    if cal.e1_csv.exists():
        with open(cal.e1_csv, newline="", encoding="utf-8") as fh:
            tabla = {r["method"]: r for r in csv.DictReader(fh)}
    for m, f in res0["e1"]["filas"].items():
        r = tabla.get(m)
        if r is None:
            continue
        out["e1"][m] = {k: {"disco": float(r[k]), "aqui": f[k], "dif": float(f[k]) - float(r[k])}
                        for k in ("matched_z", "global_empirical_fap", "global_parametric_fap")
                        if r.get(k) not in (None, "") and f.get(k) is not None}
    if cal.e1_qc.exists():
        out["veredicto_disco"] = json.loads(cal.e1_qc.read_text(encoding="utf-8")).get("verdict")
    out["veredicto_aqui"] = res0["e1"]["veredicto"]
    return out


def indices_continuo(wave):
    partes = [np.flatnonzero((wave >= lo) & (wave <= hi))[::PASO_CONTINUO] for lo, hi in BANDAS_CONTINUO_A.values()]
    return np.unique(np.concatenate(partes))


def mide_extraccion(run_id, project_root, *, copia, metodos, fondo_c4, trazas_b3, n_procesos, partes,
                    controles_produccion):
    """(c) y (d), y la verificación con δ = 0."""
    from multiprocessing import get_context

    from astropy.io import fits

    from musepipe.paths import RunPaths
    from musepipe.psf import _psfao_wave_bin_A

    t0 = time.time()
    rp = RunPaths.from_project_root(run_id, project_root)
    with fits.open(rp.stage_dir / "stage02_xcorr_cube_stack.fits", memmap=True) as h:
        wave = np.asarray(h["WAVELENGTH"].data, dtype=np.float64)
    idx_ha = np.flatnonzero((wave >= VENTANA_HA_A[0]) & (wave <= VENTANA_HA_A[1]))
    ent_ha = Entradas(run_id, project_root, idx_ha, metodos=metodos, copia=copia, fondo_c4=fondo_c4)
    ent_c = None
    if "flujo" in partes:
        ent_c = Entradas(run_id, project_root, indices_continuo(wave), metodos=metodos, copia=copia,
                         fondo_c4=fondo_c4)
    cal = Calibracion(run_id, project_root, copia=copia)
    psf_model = next(v for v in ent_ha.psf.values() if v is not None)
    bin_A = _psfao_wave_bin_A(psf_model) if str(psf_model.get("form", "")).lower() in {"psfao", "mixture"} else 0.0
    for pm in {id(v): v for v in ent_ha.psf.values() if v is not None}.values():
        calienta_psf(pm, ent_ha.wave, bin_A)
        if ent_c is not None:
            calienta_psf(pm, ent_c.wave, bin_A)
    comp_yx, prim_yx = ent_ha.comp_yx, ent_ha.star_yx
    lista = casos(comp_yx, prim_yx, trazas_b3)
    _G.clear()
    _G.update(casos=lista, ent_ha=ent_ha, ent_c=ent_c, cal=cal, bin_A=bin_A, comp_yx=comp_yx, trazas=trazas_b3,
              metodos=tuple(metodos), devuelve_productos=True)
    if controles_produccion:
        _G["controles_produccion"] = True
        _G["disco_err"] = {}
        _G["disco_ctl"] = {}
        for m in metodos:
            d = cal.producto_disco(m)
            _G["disco_err"][m] = (np.asarray(d.flux_err, dtype=np.float64)[idx_ha],
                                  np.asarray(d.flux_err_emp, dtype=np.float64)[idx_ha])
            _G["disco_ctl"][m] = cal.controles_calibrados_disco(m)[:, idx_ha] / float(cal.corr.flux_scale)

    _n, res0 = _caso("cero")
    out = {"run_id": run_id, "copia": None if copia is None else str(copia), "metodos": list(metodos),
           "fondo_c4": getattr(ent_ha, "fondo_c4", None), "comp_yx": list(comp_yx), "prim_yx": list(prim_yx),
           "ventana_ha_A": list(VENTANA_HA_A), "n_canales_ha": int(idx_ha.size),
           "lsf_fwhm_A": float(cal.lsf), "criterio_e1": {k: (v if not isinstance(v, list) else [list(p) for p in v])
                                                         for k, v in cal.criterio.items()},
           "flux_scale_d2": float(cal.corr.flux_scale)}
    out["verificacion"] = verifica_cero(res0, cal, idx_ha, metodos)
    imprime_verificacion(out["verificacion"])
    plantilla = {m: (f["peak_wave_A"], float(cal.lsf) * float(f["template_factor"]))
                 for m, f in res0["e1"]["filas"].items()}
    _G["plantilla"] = plantilla
    _G["devuelve_productos"] = False
    res0 = None
    resultados = {}
    if {"flujo", "e1"} & set(partes):
        nombres = list(lista)
        with get_context("fork").Pool(max(1, int(n_procesos))) as pool:
            for nombre, r in pool.imap_unordered(_caso, nombres, chunksize=1):
                resultados[nombre] = r
                v = r["e1"]["veredicto"]["verdict"]
                zs = " ".join(f"{m} {f['matched_z']:.2f}" for m, f in r["e1"]["filas"].items())
                print(f"   {nombre:12s} {zs}  -> {v} ({r['segundos']:.0f} s)", flush=True)
    out["plantilla_e1_cero"] = {m: list(v) for m, v in plantilla.items()}
    out["casos"] = {n: {"posicion": (list(lista[n]) if not isinstance(lista[n][0], str) else list(lista[n])),
                        **resultados[n]} for n in lista if n in resultados}
    out["resumen"] = resume(out)
    out["segundos"] = time.time() - t0
    _G.clear()
    return out


def resume(out):
    """Cambios relativos a δ = 0 por método: línea (plantilla fija), continuo, z de E1, FAP y veredicto."""
    casos_ = out.get("casos", {})
    if "cero" not in casos_:
        return {}
    c0 = casos_["cero"]
    res = {}
    for nombre, r in casos_.items():
        fila = {"veredicto": r["e1"]["veredicto"]["verdict"],
                "metodos_fap": r["e1"]["veredicto"].get("fap_hit_methods") or r["e1"]["veredicto"].get(
                    "significant_methods")}
        for m, f in r["e1"]["filas"].items():
            f0 = c0["e1"]["filas"][m]
            d = {"z": f["matched_z"], "fap_emp": f["global_empirical_fap"], "fap_param": f["global_parametric_fap"],
                 "z_umbral": f.get("z_umbral_parametrico")}
            lf, lf0 = f.get("linea_plantilla_fija"), f0.get("linea_plantilla_fija")
            if lf and lf0 and lf0["flujo"]:
                d["linea_rel"] = lf["flujo"] / lf0["flujo"] - 1.0
            if f.get("cont_runmed_plantilla") and f0.get("cont_runmed_plantilla"):
                d["cont_ha_rel"] = f["cont_runmed_plantilla"] / f0["cont_runmed_plantilla"] - 1.0
            if "continuo" in r and "continuo" in c0:
                for b in BANDAS_CONTINUO_A:
                    a, a0 = r["continuo"][m][b], c0["continuo"][m][b]
                    if a is not None and a0:
                        d[f"{b}_rel"] = a / a0 - 1.0
            if "e1_controles_produccion" in r:
                fp = r["e1_controles_produccion"]["filas"][m]
                d["z_ctl_prod"] = fp["matched_z"]
                d["fap_param_ctl_prod"] = fp["global_parametric_fap"]
            fila[m] = d
        if "e1_controles_produccion" in r:
            fila["veredicto_ctl_prod"] = r["e1_controles_produccion"]["veredicto"]["verdict"]
        res[nombre] = fila
    return res


def imprime_verificacion(v):
    for m, d in v["productos"].items():
        txt = "  ".join(f"{k} {d[k]:.1e}" if isinstance(d[k], float) else f"{k} {d[k]}"
                        for k in ("flux", "flux_err", "flux_err_emp", "controles", "controles_calibrados_d2"))
        print(f"   verifica {m:15s} {txt}  (n={d['n_controles']})")
    for m, d in v["e1"].items():
        print(f"   E1 {m:15s} " + "  ".join(f"{k} {x['aqui']:.6g} (disco {x['disco']:.6g})" for k, x in d.items()))
    print(f"   veredicto: aquí {v['veredicto_aqui'].get('verdict')}  disco {(v.get('veredicto_disco') or {}).get('verdict')}")


# ---------------------------------------------------------------------------------------------------------------
# Las cifras del texto, desde el resultado (o desde los JSON con --resume)
# ---------------------------------------------------------------------------------------------------------------

def _rango(vals):
    v = [float(x) for x in vals if x is not None and np.isfinite(x)]
    return [min(v), max(v)] if v else None


def cifras(r):
    """Lo que cita el texto, de un run: B3 por inyección, la traza en Hα, la sensibilidad del flujo y E1."""
    out = {}
    b3 = r.get("b3") or {}
    iny = b3.get("inyeccion")
    if iny:
        clas = clasifica_inyecciones(iny["filas"], iny["fwhm_px"]) if "fwhm_px" in iny else iny["clasificacion"]
        e = clas["identificadas"]
        out["b3"] = {"rms_px": e["rms_px"], "sigma_yx": e["sigma_yx"], "sesgo_yx": e["sesgo_yx"], "n_ok": e["n"],
                     "n_posiciones": iny["n_posiciones"], "fallos": clas["fallos"],
                     "n_otra_fuente": clas["n_otra_fuente"], "otra_fuente": clas["otra_fuente"],
                     "umbral_identificacion_px": clas["umbral_px"], "bandas_usadas": clas["bandas_usadas"],
                     "err_formal_qc_px": iny["err_formal_qc_px"], "snr_qc": iny["snr_qc"],
                     "rms_sobre_err_formal": e["rms_px"] / iny["err_formal_qc_px"] if e["rms_px"] else None}
    tr = b3.get("traza")
    ext = r.get("extraccion") or {}
    if tr:
        lam_b3 = tr["lam_b3_A"]
        comp, prim = ext.get("comp_yx"), ext.get("prim_yx")
        traza = {}
        for nombre, filas in (("traza", tr["filas_run"]), (f"traza_{tr['n_bandas']}", tr["filas"])):
            dy, dx = desplazamiento_en(6562.8, traza_relativa(filas, lam_b3, "primary"))
            d = {"d_ha_yx": [dy, dx], "d_ha_px": math.hypot(dy, dx), "n_bandas": len(filas)}
            if comp and prim:
                er, et = vectores(comp, prim)
                d["radial_px"] = float(np.dot([dy, dx], er))
                d["tangencial_px"] = float(np.dot([dy, dx], et))
            traza[nombre] = d
        # el compañero frente a la primaria, banda a banda, con la σ de la inyección
        lam, pdy, pdx = traza_relativa(tr["filas"], lam_b3, "primary")
        _l, cdy, cdx = traza_relativa(tr["filas"], lam_b3, "companion")
        sig = tr.get("sigma_inyeccion") or []
        bandas = []
        for i in range(len(lam)):
            s = sig[i] if i < len(sig) else {}
            bandas.append({"lam_A": float(lam[i]), "comp_menos_prim_yx": [float(cdy[i] - pdy[i]), float(cdx[i] - pdx[i])],
                           "prim_yx": [float(pdy[i]), float(pdx[i])],
                           "sigma_comp_yx": s.get("sigma_yx"), "sigma_rob_comp_yx": s.get("sigma_rob_yx"),
                           "sesgo_comp_yx": s.get("sesgo_yx")})
        traza["bandas"] = bandas
        # ¿sigue el compañero a la primaria donde su centroide está medido?
        medidas = [b for b in bandas if b["sigma_comp_yx"] and max(b["sigma_comp_yx"]) <= SIGMA_MEDIDA_PX]
        traza["sigue"] = None if not medidas else {
            "umbral_sigma_px": SIGMA_MEDIDA_PX, "lam_min_A": min(b["lam_A"] for b in medidas),
            "bandas_A": [b["lam_A"] for b in medidas],
            "max_comp_menos_prim_px": max(math.hypot(*b["comp_menos_prim_yx"]) for b in medidas),
            "max_sigma_px": max(max(b["sigma_comp_yx"]) for b in medidas)}
        mods = [math.hypot(*b["prim_yx"]) for b in bandas]
        imax = int(np.argmax(mods))
        traza["primaria_max"] = {"px": mods[imax], "lam_A": bandas[imax]["lam_A"], "px_banda_mas_azul": mods[0],
                                 "lam_mas_azul_A": bandas[0]["lam_A"]}
        con_ha = [b for b, f in zip(bandas, tr["filas"]) if f["wave_min_A"] <= 6562.8 <= f["wave_max_A"]]
        traza["banda_ha"] = con_ha[0] if con_ha else None
        e2 = b3.get("centroide_ha_e2")
        if e2 and comp and prim:
            er, et = vectores(comp, prim)
            e2 = dict(e2, radial_px=float(np.dot(e2["d_yx"], er)), tangencial_px=float(np.dot(e2["d_yx"], et)))
            e2["menos_traza_px"] = {k: math.hypot(e2["d_yx"][0] - traza[k]["d_ha_yx"][0],
                                                  e2["d_yx"][1] - traza[k]["d_ha_yx"][1])
                                    for k in traza if k.startswith("traza")}
        traza["centroide_ha_e2"] = e2
        out["traza"] = traza
    ver = ext.get("verificacion")
    if ver:
        out["verificacion"] = {
            "max_rel_productos": max(float(v) for d in ver["productos"].values() for k, v in d.items()
                                     if k not in ("n_controles",) and isinstance(v, float)),
            "max_rel_por_metodo": {m: max(float(v) for k, v in d.items() if k != "n_controles" and isinstance(v, float))
                                   for m, d in ver["productos"].items()},
            "max_abs_dif_e1": max(abs(float(x["dif"])) for d in ver["e1"].values() for x in d.values()),
            "veredicto": [ver["veredicto_aqui"].get("verdict"), (ver.get("veredicto_disco") or {}).get("verdict")]}
    res = ext.get("resumen") or {}
    if res:
        metodos = ext["metodos"]
        sens = {}
        for d in DESPLAZAMIENTOS_PX:
            nombres = [f"{direc}_{d:g}" for direc in DIRECCIONES if f"{direc}_{d:g}" in res]
            sens[f"{d:g}"] = {m: {k: _rango([res[n][m].get(k) for n in nombres])
                                  for k in ("linea_rel", "cont_ha_rel", "c8700_rel", "c7150_rel", "z", "fap_param")}
                              for m in metodos}
            sens[f"{d:g}"]["veredictos"] = {n: res[n]["veredicto"] for n in nombres}
        for nombre in [n for n in res if n.startswith("traza")]:
            sens[nombre] = {m: {k: res[nombre][m].get(k) for k in ("linea_rel", "cont_ha_rel", "c8700_rel", "c7150_rel",
                                                                    "z", "fap_param", "fap_emp", "z_umbral",
                                                                    "z_ctl_prod", "fap_param_ctl_prod")}
                            for m in metodos}
            sens[nombre]["veredicto"] = res[nombre]["veredicto"]
            sens[nombre]["metodos_fap"] = res[nombre]["metodos_fap"]
            sens[nombre]["veredicto_ctl_prod"] = res[nombre].get("veredicto_ctl_prod")
        sens["cero"] = {m: {k: res["cero"][m].get(k) for k in ("z", "fap_param", "fap_emp", "z_umbral")}
                        for m in metodos}
        sens["cero"]["veredicto"] = res["cero"]["veredicto"]
        todos = [n for n in res if n != "cero"]
        sens["z_minimo"] = {m: min(res[n][m]["z"] for n in todos) for m in metodos} if todos else None
        sens["z_maximo"] = {m: max(res[n][m]["z"] for n in todos) for m in metodos} if todos else None
        sens["veredictos"] = sorted({res[n]["veredicto"] for n in res})
        sens["casos_por_veredicto"] = {v: sorted(n for n in res if res[n]["veredicto"] == v)
                                       for v in sens["veredictos"]}
        out["flujo_e1"] = sens
    return out


def imprime_cifras(nombre, c):
    print(f"== cifras de {nombre}")
    if "b3" in c:
        b = c["b3"]
        print(f"   B3: rms {b['rms_px']:.4f} px, σ {np.round(b['sigma_yx'], 4).tolist()}, sesgo "
              f"{np.round(b['sesgo_yx'], 4).tolist()}, {b['n_ok']}/{b['n_posiciones']} identificadas, fallos "
              f"{b['fallos']}, otra fuente {b['n_otra_fuente']} {b['otra_fuente']}; error formal "
              f"{b['err_formal_qc_px']:.4f} px (rms/formal {b['rms_sobre_err_formal']:.2f}), S/N {b['snr_qc']:.1f}")
    if "traza" in c:
        for k, d in c["traza"].items():
            if k in ("centroide_ha_e2", "sigue", "banda_ha", "primaria_max"):
                print(f"   {k}: {json.dumps(d, default=_json_default)}")
                continue
            if k == "bandas":
                continue
            print(f"   {k}: d(Hα) = {np.round(d['d_ha_yx'], 3).tolist()} = {d['d_ha_px']:.3f} px "
                  f"(radial {d.get('radial_px', float('nan')):+.3f}, tangencial {d.get('tangencial_px', float('nan')):+.3f})")
        for b in c["traza"]["bandas"]:
            print(f"     {b['lam_A']:7.0f} Å  primaria {np.round(b['prim_yx'], 3).tolist()}  compañero−primaria "
                  f"{np.round(b['comp_menos_prim_yx'], 3).tolist()}  σ {None if b['sigma_comp_yx'] is None else np.round(b['sigma_comp_yx'], 3).tolist()}")
    if "verificacion" in c:
        print(f"   verificación δ = 0: {c['verificacion']}")
    if "flujo_e1" in c:
        s = c["flujo_e1"]
        print(f"   δ = 0: {s['cero']}")
        for k, v in s.items():
            if k in ("cero", "z_minimo", "z_maximo", "veredictos", "casos_por_veredicto"):
                continue
            print(f"   {k}: {json.dumps(v, default=_json_default)}")
        print(f"   z mín/máx: {s['z_minimo']} / {s['z_maximo']}; veredictos {s['casos_por_veredicto']}")


# ---------------------------------------------------------------------------------------------------------------
# Figura de la traza (opcional)
# ---------------------------------------------------------------------------------------------------------------

def figura_traza(resultados, ruta):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ejes = plt.subplots(len(resultados), 2, figsize=(9, 3.2 * len(resultados)), squeeze=False)
    for fila, (nombre, r) in zip(ejes, resultados.items()):
        tr = r["b3"]["traza"]
        lam_ref = tr["lam_b3_A"]
        filas = tr["filas"]
        sig = tr.get("sigma_inyeccion") or [{}] * len(filas)
        lam = np.array([0.5 * (f["wave_min_A"] + f["wave_max_A"]) for f in filas])
        for j, eje in enumerate(("y", "x")):
            ax = fila[j]
            for clave, color in (("primary", "0.2"), ("companion", "C3")):
                lam_t, dy, dx = traza_relativa(filas, lam_ref, clave)
                val = dy if eje == "y" else dx
                err = None
                if clave == "companion":
                    err = [s.get("sigma_yx", [np.nan, np.nan])[j] if s.get("sigma_yx") else np.nan for s in sig]
                ax.errorbar(lam_t, val, yerr=err, fmt="o-", ms=3, lw=1, color=color, capsize=2,
                            label={"primary": "primary", "companion": "companion"}[clave])
            ax.axvline(6562.8, color="C0", lw=0.8, ls=":")
            ax.axhline(0.0, color="0.6", lw=0.6)
            ax.set_xlabel("wavelength (Å)")
            ax.set_ylabel(f"Δ{eje} from the B3 band (px)")
            ax.set_ylim(-1.5, 1.5)
            ax.set_title(f"{nombre}: {eje}", fontsize=9)
            if j == 0:
                ax.legend(fontsize=8, frameon=False)
        del lam
    fig.tight_layout()
    fig.savefig(ruta, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------------------------------------------

def objetos_del_paper(root=ROOT):
    """Los slugs de `targets/<slug>.json` que declaran `paper`, en el orden de `paper.order`."""
    from musepipe.targets import targets_dir

    fichas = [(json.loads(f.read_text(encoding="utf-8")), f.stem) for f in sorted(targets_dir(root).glob("*.json"))]
    return tuple(s for _, s in sorted((d["paper"]["order"], s) for d, s in fichas if "paper" in d))


def run_del_objetivo(slug, root=ROOT):
    from musepipe.targets import load_target

    t = load_target(slug, project_root=root)
    if not t or "paper" not in t:
        raise SystemExit(f"{slug}: no está en targets/ o no declara `paper`")
    return t["paper"]["run"]


def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    if isinstance(o, tuple):
        return list(o)
    raise TypeError(type(o))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--project-root", default=str(ROOT), help="clon con runs/ (desde un worktree, el checkout principal)")
    ap.add_argument("--sin-escribir", action="store_true", help="obligatorio: nunca escribe en runs/")
    ap.add_argument("--objetivo", action="append", default=None, metavar="SLUG",
                    help="objeto de targets/ (repetible; por defecto, los del paper en su orden)")
    ap.add_argument("--run-id", action="append", default=None, metavar="RUN[@COPIA]",
                    help="run tal cual en vez del de targets/; RUN@CARPETA lee modelo de PSF, productos, D2 y E1 de una "
                         "copia (stages/, tables/), y los cubos del run")
    ap.add_argument("--partes", default=",".join(PARTES), help=f"subconjunto de {','.join(PARTES)}")
    ap.add_argument("--metodos", default=",".join(METODOS), help="métodos de C2–C6")
    ap.add_argument("--n-procesos", type=int, default=4)
    ap.add_argument("--n-inyecciones", type=int, default=N_INYECCIONES)
    ap.add_argument("--bandas-traza", type=int, default=12)
    ap.add_argument("--semilla", type=int, default=SEMILLA)
    ap.add_argument("--evaluador-descentrado", action="store_true",
                    help="el evaluador de PSF de antes del 2026-10-21 (para reproducir la sesión)")
    ap.add_argument("--fondo-c4", default=None, choices=("shared", "decoupled_c0"),
                    help="sustituye x03_background del config (para reproducir la sesión: shared)")
    ap.add_argument("--controles-de-produccion", action="store_true",
                    help="evalúa también E1 con los controles y errores del disco (lo que hacía s4)")
    ap.add_argument("--json", default=None, help="salida (fuera de runs/)")
    ap.add_argument("--figura", default=None, help="figura de la traza cromática (fuera de runs/)")
    ap.add_argument("--resume", nargs="+", default=None, metavar="JSON",
                    help="no mide: imprime las cifras del texto desde salidas --json anteriores (y con --figura, la figura)")
    a = ap.parse_args(argv)
    if not a.sin_escribir:
        ap.error("pasa --sin-escribir: este script solo mide en memoria")
    if a.resume:
        project_root = Path(a.project_root).resolve()
        prohibidas = carpetas_prohibidas(project_root)
        for salida in (a.json, a.figura):
            if salida:
                comprueba_fuera([Path(salida)], prohibidas)
        todos = {}
        for f in a.resume:  # por run, las claves de arriba (b3, extraccion) del último fichero que las traiga
            for k, v in json.loads(Path(f).read_text(encoding="utf-8"))["runs"].items():
                todos.setdefault(k, {}).update(v)
        salida = {}
        for nombre, r in todos.items():
            salida[nombre] = cifras(r)
            imprime_cifras(nombre, salida[nombre])
        if a.json:
            Path(a.json).write_text(json.dumps(salida, indent=1, default=_json_default, ensure_ascii=False) + "\n",
                                    encoding="utf-8")
        if a.figura and any("b3" in r for r in todos.values()):
            figura_traza({k: r for k, r in todos.items() if "b3" in r}, a.figura)
        return 0
    partes = tuple(p for p in a.partes.split(",") if p)
    metodos = tuple(m for m in a.metodos.split(",") if m)
    if set(partes) - set(PARTES) or set(metodos) - set(METODOS):
        ap.error(f"partes {PARTES}; métodos {METODOS}")
    project_root = Path(a.project_root).resolve()
    if a.run_id:
        objetivos = [parsea_objetivo(t, project_root) for t in a.run_id]
    else:
        slugs = a.objetivo or objetos_del_paper(project_root)
        objetivos = [(run_del_objetivo(s, project_root), None) for s in slugs]
    prohibidas = carpetas_prohibidas(project_root) + [c for _r, c in objetivos]
    for salida in (a.json, a.figura):
        if salida:
            comprueba_fuera([Path(salida)], prohibidas)
    if a.evaluador_descentrado:
        usa_evaluador_descentrado()
    resultados = {}
    for run, copia in objetivos:
        print(f"== {run}" + (f" @ {copia}" if copia else ""), flush=True)
        r = {"run_id": run, "copia": None if copia is None else str(copia),
             "evaluador_descentrado": bool(a.evaluador_descentrado)}
        trazas = {}
        if {"b3", "traza", "flujo", "e1"} & set(partes):
            r["b3"] = mide_b3(run, project_root, copia=copia, n_inyecciones=a.n_inyecciones,
                              n_bandas=a.bandas_traza, n_procesos=a.n_procesos, semilla=a.semilla, partes=partes)
            lam_b3 = r["b3"]["traza"]["lam_b3_A"]
            trazas["traza"] = traza_relativa(r["b3"]["traza"]["filas_run"], lam_b3, "primary")
            trazas[f"traza_{a.bandas_traza}"] = traza_relativa(r["b3"]["traza"]["filas"], lam_b3, "primary")
            r["trazas_usadas"] = {k: [list(map(float, v)) for v in t] for k, t in trazas.items()}
            for k, t in trazas.items():
                print(f"   {k}: d(6563 Å) = {desplazamiento_en(6563.0, t)} px")
        if {"verifica", "flujo", "e1"} & set(partes):
            r["extraccion"] = mide_extraccion(run, project_root, copia=copia, metodos=metodos, fondo_c4=a.fondo_c4,
                                              trazas_b3=trazas, n_procesos=a.n_procesos, partes=partes,
                                              controles_produccion=a.controles_de_produccion)
        resultados[run if copia is None else f"{run}@{copia.name}"] = r
        if a.json:  # se reescribe tras cada run: un fallo en el segundo no pierde el primero
            Path(a.json).write_text(json.dumps({"runs": resultados}, indent=1, default=_json_default,
                                               ensure_ascii=False) + "\n", encoding="utf-8")
    if a.figura and any("b3" in r for r in resultados.values()):
        figura_traza({k: r for k, r in resultados.items() if "b3" in r}, a.figura)
    return 0


if __name__ == "__main__":
    sys.exit(main())
