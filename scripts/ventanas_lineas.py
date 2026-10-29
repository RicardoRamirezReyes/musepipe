#!/usr/bin/env python
"""La ventana de las líneas en los ajustes de T_eff: alas del perfil instrumental, fuga en los bins y bins recuperados.

Pregunta del autor (2026-10-28), sobre la ventana de ±300 km/s que G3 pone alrededor de las 24 líneas del catálogo de
G2 (`g3_fit_masks.line_window_kms`, `musepipe.models.observed.build_fit_masks`) y que hereda `scripts/teff_bayes.py`:
«¿qué número parecería óptimo para extraer la mayor información no sesgada?». Tras degradar a R = 500,
`degrade_observed` agranda cada zona enmascarada ±1 FWHM del núcleo (12.4–18.6 Å), así que lo excluido por línea es
±(w + ~600 km/s). Solo lectura y fuera de la cadena; no escribe en `runs/`.

Subórdenes (cada una imprime sus cifras; con `--salida DIR` escribe además `DIR/<orden>.json`):

  alas   la fracción del flujo de una línea no resuelta que queda fuera de ±v (`V_ALAS`), medida en las líneas de cielo
         de los SKY_SPECTRUM por exposición que lista A4/M2 (como `halpha_criterios.py lsf`), a la resolución nativa:
         [O I] 6300 y 6363 (singletes) en su lado limpio hasta `D_MAX_OI` Å, y tres líneas de OH de M2 (dobletes sin
         resolver) en su lado limpio hasta `D_MAX_OH` Å (solo v ≤ 100 km/s). Contraste: la gaussiana de la misma FWHM
         y el perfil de Hα del compañero detectado (espectro canónico de D2, `spec_final_object.fits`, continuo lineal
         en las ventanas de G2), con el error de los 33 controles de D2 procesados igual.
  fuga   para cada línea del catálogo dentro del rango del ajuste, el flujo que entra en los bins de 15 Å que USA el
         ajuste, con w = `W_GRID`: se inyecta la línea (perfil de cielo medido, `perfil_fuga`) en las entradas del
         segmento de MUSE (`tipo_ir_plantillas.entradas_muse`, las mismas que el ajuste) y se pasa por
         `fit_spectrum_from_inputs` con el mismo degradado a R = 500 y el mismo crecimiento de la máscara (el código,
         no una aproximación; todo es lineal en el flujo, así que basta inyectar la línea sola). Flujo de cada línea:
         Hα detectada, `flux_direct` de G2; Hα con cota, `f_lim_observed` de E3; las demás, `flux_upper_limit_99` de
         G2. Se compara con el error de cada bin (`err_bin`, el de los datos, sin f_b: el caso conservador). Criterio:
         fuga ≤ 0.1σ (y 0.05, 0.2) en todos los bins, para centros en reposo ± `DELTAS_KMS`. Da la w mínima por línea
         y por compañero, y la de todas las líneas a la vez.
  bins   los bins de MUSE que quedan con cada w global y con una w para Hα y otra para el resto, con las máscaras
         `paper` y `sin_o2b` (`teff_bayes.MASCARAS_MUSE`; w en una copia del config, nunca en `runs/`).
  todo   las tres.

Los objetos salen de `targets/` (los que declaran `paper`), nunca literales. `--project-root` lee los runs de otro clon
(un worktree no lleva `runs/`).

uso: python scripts/ventanas_lineas.py {alas,fuga,bins,todo} [--project-root RAIZ] [--salida DIR]
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from musepipe.constants import C_KMS  # noqa: E402

#: velocidades a las que se mide la fracción fuera de ±v (km/s)
V_ALAS = (50.0, 75.0, 100.0, 150.0, 200.0, 250.0, 300.0, 400.0)
#: ventanas probadas (±km/s)
W_GRID = (50.0, 75.0, 100.0, 125.0, 150.0, 175.0, 200.0, 225.0, 250.0, 275.0, 300.0)
#: centros de cada línea: reposo × (1 + δ/c) (y el centroide de G2 si la línea está detectada)
DELTAS_KMS = (-30.0, 0.0, 30.0)
#: criterios de fuga (en σ del bin)
CRITERIOS = (0.05, 0.1, 0.2)
#: líneas de cielo: nombre -> (λ aire, lado limpio, distancia máxima limpia en Å). [O I]: singletes; el lado rojo de
#: 6300 lleva OH 6306.9 (+5…+9 Å) y el azul de 6363, OH 6356 (−6…−9 Å). OH de M2: dobletes de Λ sin resolver (el
#: compañero del doblete en el azul) con la línea vecina a +4…+10 Å: valen solo hasta `D_MAX_OH`.
D_MAX_OI = 10.0
D_MAX_OH = 3.1
LINEAS_CIELO = {"[O I] 6300": (6300.304, "azul", D_MAX_OI), "[O I] 6363": (6363.776, "rojo", D_MAX_OI),
                "OH 7913": (7913.4, "rojo", D_MAX_OH), "OH 8344": (8344.4, "rojo", D_MAX_OH),
                "OH 8827": (8826.9, "rojo", D_MAX_OH)}
#: anillo del continuo de las líneas de cielo (Å del centro)
ANILLO_CIELO = (15.0, 25.0)
#: rejilla fina del perfil (Å respecto del centroide)
PASO = 0.01
REJILLA = np.arange(-D_MAX_OI, D_MAX_OI + PASO / 2, PASO)
#: método canónico (el del espectro final) para el límite de E3
METODO = "psffit"
#: semiancho (Å) de la integral total de Hα en el espectro del compañero detectado
D_TOTAL_HA = 12.0


# ====================================================================================== utilidades
def fraccion_fuera(x, y, d_max):
    """Fracción del área de un perfil de UN lado (x ≥ 0 en Å del centro, y sin continuo) que queda más allá de cada
    x: devuelve la función f(d) = ∫_d^{d_max} y / ∫_0^{d_max} y (trapecios en la rejilla dada)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    s = (x >= 0) & (x <= d_max)
    x, y = x[s], y[s]
    cum = np.concatenate([[0.0], np.cumsum(0.5 * (y[1:] + y[:-1]) * np.diff(x))])
    tot = cum[-1]
    return lambda d: (tot - np.interp(d, x, cum)) / tot


def bordes(wave):
    """Bordes de los canales (puntos medios; extremos simétricos)."""
    from scripts.halpha_criterios import bordes as b
    return b(np.asarray(wave, float))


def linea_en_canales(wave, flujo, centro, x, p):
    """Una línea de flujo `flujo` con perfil `p(x)` (x en Å del centro, área cualquiera; se normaliza) integrada en
    los canales de `wave`: densidad media por canal (unidades de flujo / Å). Conserva el flujo dentro de la rejilla."""
    x, p = np.asarray(x, float), np.clip(np.asarray(p, float), 0, None)
    c = np.concatenate([[0.0], np.cumsum(0.5 * (p[1:] + p[:-1]) * np.diff(x))])
    c = c / c[-1]
    e = bordes(wave)
    cdf = np.interp(e - centro, x, c, left=0.0, right=1.0)
    return flujo * np.diff(cdf) / np.diff(e)


# ====================================================================================== entradas
def objetos():
    from scripts.halpha_criterios import objetos_del_paper
    return objetos_del_paper(ROOT)


def _run(slug):
    from musepipe.targets import load_target
    return load_target(slug, project_root=ROOT)["paper"]["run"]


def _rp(slug, raiz):
    from musepipe.paths import RunPaths
    return RunPaths.from_project_root(_run(slug), raiz)


def _cfg(slug, raiz):
    from musepipe.config import load_run_config
    return load_run_config(_run(slug), project_root=raiz).config


def tabla_g2(slug, raiz):
    with open(_rp(slug, raiz).table_dir / "g2_line_measurements.csv", newline="") as fh:
        return list(csv.DictReader(fh))


def escala_espectro(slug, raiz):
    """(cgs por unidad del espectro final, fuente) con el resolutor de la cadena (sin valor por defecto)."""
    from astropy.io import fits

    from musepipe.io import resolve_flux_unit
    with fits.open(_rp(slug, raiz).stage_dir / "spec_final_object.fits") as h:
        return resolve_flux_unit(_cfg(slug, raiz), bunit=h["SPECTRUM"].header.get("BUNIT"))


def flujos_maximos(slug, raiz):
    """{rest_A: dict(nombre, flujo (unidades del espectro final), fuente, centroide)} de las líneas del catálogo de G2:
    Hα detectada, `flux_direct`; Hα con cota, `f_lim_observed` de E3 (método `METODO`, en cgs, pasado a las unidades
    del espectro); las demás, `flux_upper_limit_99`. Las tablas de G2 están en las unidades del espectro que mide
    (el final)."""
    from scripts.tipo_ir_plantillas import HALPHA_A
    esc, _ = escala_espectro(slug, raiz)
    out = {}
    for r in tabla_g2(slug, raiz):
        rest = float(r["rest_A"])
        es_ha = abs(rest - HALPHA_A) < 1.0
        cen = float(r["centroid_A"]) if r.get("centroid_A") not in (None, "", "nan") else float("nan")
        if es_ha and r["status"] == "detected":
            f, fuente = float(r["flux_direct"]), "G2 flux_direct"
        elif es_ha:
            h03 = json.loads((_rp(slug, raiz).stage_dir / "stage_h03_qc.json").read_text())
            fila = next(x for x in h03["limits"] if x["method"] == METODO)
            f, fuente = float(fila["f_lim_observed"]) / esc, f"E3 f_lim_observed ({METODO})"
        else:
            f, fuente = float(r["flux_upper_limit_99"]), "G2 flux_upper_limit_99"
        out[rest] = dict(nombre=r["name"], flujo=f, flujo_cgs=f * esc, fuente=fuente, detectada=r["status"] == "detected",
                         centroide=cen, flux_direct=float(r["flux_direct"]) if r["flux_direct"] not in ("", "nan") else None)
    return out


# ====================================================================================== alas (1)
def perfiles_cielo(slug, raiz):
    """Por línea de cielo: el perfil de cada SKY_SPECTRUM de A4/M2 del run de `slug` (continuo = mediana en
    `ANILLO_CIELO`, centrado en su centroide a ±1.5 Å, normalizado a su pico), en `REJILLA`. Devuelve
    {línea: (lista de perfiles, fallos)}."""
    from musepipe.qc.cube_qc import read_sky_spectrum_fits
    qc = json.loads((_rp(slug, raiz).stage_dir / "stage00q_qc.json").read_text())["m2_lsf"]
    out = {k: ([], 0) for k in LINEAS_CIELO}
    for p in qc["sky_spectrum_files"]:
        try:
            w, f = read_sky_spectrum_fits(p)
        except Exception:  # noqa: BLE001 - un fichero ilegible no para el resto; se cuenta
            for k in out:
                out[k] = (out[k][0], out[k][1] + 1)
            continue
        for k, (lam, _, _) in LINEAS_CIELO.items():
            d = w - lam
            cont = np.nanmedian(f[(np.abs(d) >= ANILLO_CIELO[0]) & (np.abs(d) <= ANILLO_CIELO[1])])
            y = f - cont
            s = np.abs(d) <= 1.5
            yp = np.clip(y[s], 0, None)
            cen = float(np.sum(w[s] * yp) / np.sum(yp))
            g = np.interp(REJILLA, w - cen, y, left=np.nan, right=np.nan)
            out[k][0].append(g / np.nanmax(g[np.abs(REJILLA) <= 0.5]))
    return {k: (np.asarray(v[0]), v[1]) for k, v in out.items()}, qc


def lado(perfil, cual):
    """(x ≥ 0, y) del lado `cual` ('azul' o 'rojo') de un perfil en `REJILLA` (el azul, reflejado)."""
    if cual == "azul":
        s = REJILLA <= 0
        return -REJILLA[s][::-1], perfil[s][::-1]
    s = REJILLA >= 0
    return REJILLA[s], perfil[s]


def ancho_cruce(x, y, fr=0.5):
    from scripts.halpha_criterios import ancho_cruce_A
    return ancho_cruce_A(x, y, fr)


def alas_cielo(slug, raiz):
    """Fracción fuera de ±v por línea de cielo: la del perfil medio de todas las exposiciones (en su lado limpio,
    normalizado a su área hasta `d_max`) y su error (dispersión entre exposiciones / √n); la FWHM del perfil medio y
    la de la gaussiana de esa FWHM. `perfil` (rejilla, perfil medio) para `perfil_fuga`."""
    from scipy.special import erfc
    perf, qc = perfiles_cielo(slug, raiz)
    out = {}
    for k, (lam, cual, d_max) in LINEAS_CIELO.items():
        P, fallos = perf[k]
        medio = np.nanmean(P, axis=0)
        x, y = lado(medio, cual)
        fwhm = ancho_cruce(REJILLA, medio, 0.5)
        f_med = fraccion_fuera(x, y, d_max)
        por_exp = [fraccion_fuera(*lado(p, cual), d_max) for p in P]
        filas = {}
        for v in V_ALAS:
            d = lam * v / C_KMS
            if d > d_max:
                continue
            vals = np.array([g(d) for g in por_exp])
            sig = fwhm / 2.3548200450309493
            filas[f"{v:g}"] = dict(d_A=d, fraccion=float(f_med(d)), error=float(np.std(vals, ddof=1) / np.sqrt(vals.size)),
                                   gauss=float(erfc(d / (sig * np.sqrt(2)))))
        out[k] = dict(lambda_A=lam, lado=cual, d_max_A=d_max, v_max_kms=d_max / lam * C_KMS, n=int(P.shape[0]),
                      fallos=fallos, fwhm_A=fwhm, w10_A=ancho_cruce(REJILLA, medio, 0.1),
                      fwhm_m2_A=float(np.polyval(qc["poly2_coeffs"], lam)), fuera=filas,
                      perfil=(REJILLA.tolist(), medio.tolist()))
    return out


def alas_halpha(slug, raiz, perfil=None):
    """Hα del compañero detectado en el espectro final (canónico de D2): continuo lineal de G2 (`ventanas_g2` de
    `halpha_criterios`), fracción del flujo dentro de ±`D_TOTAL_HA` Å que queda fuera de ±v alrededor del centroide de
    G2, y su error: la misma cuenta (el flujo fuera de ±v) sobre los 33 controles de D2, entre el flujo de la línea.
    Con `perfil` = (x, p) (`perfil_fuga`), la misma cuenta sobre una línea sin resolver de ese perfil integrada en los
    canales del espectro (`modelo`), y sobre la gaussiana de la LSF declarada (`h01_lsf_fwhm_A`; `gauss`), sin ruido."""
    from astropy.io import fits

    from scripts.halpha_criterios import ventanas_g2
    from musepipe.lines import _linear_continuum
    rp = _rp(slug, raiz)
    with fits.open(rp.stage_dir / "spec_final_object.fits") as h:
        t = h["SPECTRUM"].data
        w, f = np.asarray(t["wave_A"], float), np.asarray(t["flux"], float)
        flags = np.asarray(t["flags"], int) if "flags" in t.columns.names else np.zeros(w.size, int)
    ctr = np.asarray(np.load(rp.stage_dir / f"spec_calibrated_{METODO}_controls.npz")["control_spectra"], float)
    fm = flujos_maximos(slug, raiz)
    ha = next(v for r, v in fm.items() if abs(r - 6562.8) < 1.0)
    cen = ha["centroide"]
    blue, red = ventanas_g2(w, cen, flags)

    def fuera(fl):
        _, c, _ = _linear_continuum(w, fl, blue, red, cen)
        y = fl - c
        y = np.where(np.abs(w - cen) <= D_TOTAL_HA + 5.0, y, 0.0)   # NaN lejos de la línea no se propaga
        e = bordes(w)
        cum = np.concatenate([[0.0], np.cumsum(y * np.diff(e))])
        F = lambda d: np.interp(cen + d, e, cum) - np.interp(cen - d, e, cum)  # noqa: E731
        return F(D_TOTAL_HA), {f"{v:g}": F(D_TOTAL_HA) - F(cen * v / C_KMS) for v in V_ALAS}

    tot, fo = fuera(f)
    ctrs = [fuera(c)[1] for c in ctr]
    modelos = {}
    if perfil is not None:
        lsf = float(_cfg(slug, raiz)["h01_lsf_fwhm_A"])
        xg = np.arange(-15.0, 15.0 + PASO / 2, PASO)
        for nombre, (x, p) in (("modelo", perfil), ("gauss", (xg, np.exp(-0.5 * (xg / (lsf / 2.3548200450309493)) ** 2)))):
            tm, fm_ = fuera(linea_en_canales(w, float(tot), cen, x, p))
            modelos[nombre] = {k: float(v / tm) for k, v in fm_.items()}
    out = dict(slug=slug, centroide_A=cen, flujo_total=float(tot), flux_direct_g2=ha["flux_direct"],
               d_total_A=D_TOTAL_HA, n_controles=len(ctrs), fuera={})
    for k, v in fo.items():
        sig = float(np.std([c[k] for c in ctrs], ddof=1))
        out["fuera"][k] = dict(fraccion=float(v / tot), error=sig / float(tot),
                               **{f"{n}_en_canales": m[k] for n, m in modelos.items()})
    return out


def perfil_fuga(alas_por_run):
    """El perfil con el que se inyecta cada línea: la ENVOLVENTE (máximo, punto a punto) de los lados limpios de
    [O I] 6300 y 6363, reflejados, de los perfiles medios de los runs del paper; negativos a 0 (ruido del continuo:
    da más alas, no menos). Simétrico, en Å del centro, hasta ±`D_MAX_OI` Å (fuera, 0). Sin escalar a la FWHM de cada
    línea: la LSF de MUSE es casi constante en Å, y este perfil es más ancho que la gaussiana declarada (el caso
    conservador)."""
    lados = []
    for alas in alas_por_run.values():
        for k in ("[O I] 6300", "[O I] 6363"):
            g, p = (np.asarray(a) for a in alas[k]["perfil"])
            x, y = lado(p, alas[k]["lado"])
            lados.append(np.interp(np.abs(REJILLA), x, y))
    env = np.clip(np.nanmax(np.asarray(lados), axis=0), 0, None)
    return REJILLA.copy(), env / (env.sum() * PASO)


# ====================================================================================== fuga (2)
def _fs(inputs, cfg, kernel):
    from musepipe.models.observed import fit_spectrum_from_inputs
    return fit_spectrum_from_inputs(inputs, n_channels=12, wave_range=cfg["g3_fit_wave_range_A"],
                                    data_kernel_fwhm_A=kernel)


def fuga_unitaria(inputs, cfg, kernel, centro, x, p, base=None):
    """Densidad que una línea de flujo 1 centrada en `centro` deja en cada bin del ajuste: la línea sola, con el
    patrón de NaN del dato, por el mismo `fit_spectrum_from_inputs` (degradado y crecimiento de la máscara). Devuelve
    (wave_bin, fuga, err_bin)."""
    base = base if base is not None else _fs(inputs, cfg, kernel)
    lin = linea_en_canales(inputs["wave"], 1.0, centro, x, p)
    iny = dict(inputs, flux=np.where(np.isfinite(inputs["flux"]), lin, np.nan))
    fs = _fs(iny, cfg, kernel)
    if not np.array_equal(fs.wave_bin, base.wave_bin):
        raise RuntimeError("los bins con la línea no son los del dato")
    return base.wave_bin, np.asarray(fs.flux_bin, float), np.asarray(base.err_bin, float)


def fuga(slug, raiz, x, p, w_grid=W_GRID, mascara="paper"):
    """Por w de `w_grid` y por línea del catálogo dentro del rango del ajuste (± 30 Å): la fuga máxima en σ del bin
    (en el peor centro de `DELTAS_KMS` y, si está detectada, en su centroide), el bin donde cae y la fracción del
    flujo de la línea que entra en los bins; más «todas»: todas las líneas a la vez, cada una en su peor centro."""
    from scripts import teff_bayes as tb
    from scripts import tipo_ir_plantillas as tp
    run = _run(slug)
    mm = tb.MASCARAS_MUSE[mascara]
    fm = flujos_maximos(slug, raiz)
    lo, hi = _cfg(slug, raiz)["g3_fit_wave_range_A"]
    lineas = {r: v for r, v in fm.items() if lo - 30 <= r <= hi + 30 and np.isfinite(v["flujo"]) and v["flujo"] > 0}
    out = {}
    for w in w_grid:
        inputs, cfg, kernel = tp.entradas_muse(run, h2o=mm["h2o"], quita_o2=mm["quita_o2"], ventana_kms=w)
        base = _fs(inputs, cfg, kernel)
        dw = float(np.median(np.diff(inputs["wave"])))
        filas, suma = {}, None
        for r, v in lineas.items():
            centros = [r * (1 + d / C_KMS) for d in DELTAS_KMS]
            if v["detectada"] and np.isfinite(v["centroide"]):
                centros.append(v["centroide"])
            peor = None
            for c in centros:
                wb, fu, eb = fuga_unitaria(inputs, cfg, kernel, c, x, p, base)
                z = v["flujo"] * fu / eb
                k = int(np.argmax(np.abs(z)))
                if peor is None or abs(z[k]) > abs(peor["z_max"]):
                    peor = dict(z_max=float(z[k]), bin_A=float(wb[k]), centro_A=float(c),
                                fraccion_en_bins=float(np.sum(fu) * 12 * dw), fuga=fu * v["flujo"])
            suma = peor["fuga"] if suma is None else suma + peor["fuga"]
            filas[f"{r:g}"] = {k: peor[k] for k in ("z_max", "bin_A", "centro_A", "fraccion_en_bins")}
            filas[f"{r:g}"].update(nombre=v["nombre"], flujo=v["flujo"], fuente=v["fuente"])
        zt = suma / np.asarray(base.err_bin, float)
        k = int(np.argmax(np.abs(zt)))
        out[f"{w:g}"] = dict(n_bins=int(base.n_bins), lineas=filas,
                             todas=dict(z_max=float(zt[k]), bin_A=float(base.wave_bin[k]),
                                        n_sobre={f"{c:g}": int(np.sum(np.abs(zt) > c)) for c in CRITERIOS}))
    return out


def w_minima(por_w, clave=None, criterio=0.1):
    """La menor w de la rejilla tal que la fuga es ≤ criterio·σ en esa w y en todas las mayores (None si ni 300)."""
    ws = sorted(por_w, key=float)
    ok = [abs((por_w[w]["todas"] if clave is None else por_w[w]["lineas"][clave])["z_max"]) <= criterio for w in ws]
    res = None
    for w, bien in zip(reversed(ws), reversed(ok)):
        if not bien:
            break
        res = float(w)
    return res


# ====================================================================================== bins (3)
def bins(slug, w_grid=W_GRID, mascaras=("paper", "sin_o2b"), pares=()):
    """{máscara: {w: n}} con w global, y {máscara: {"wha/wresto": n}} para los `pares` (w_Hα, w_resto)."""
    from scripts import teff_bayes as tb
    from scripts import tipo_ir_plantillas as tp
    run = _run(slug)
    out = {"global": {}, "pares": {}}
    for m in mascaras:
        mm = tb.MASCARAS_MUSE[m]
        n = lambda vl, vh: int(tp.segmento_muse(run, h2o=mm["h2o"], quita_o2=mm["quita_o2"], ventana_kms=vl,  # noqa: E731
                                                ventana_ha_kms=vh)["bueno"].sum())
        out["global"][m] = {f"{w:g}": n(w, None) for w in w_grid}
        out["global"][m]["run"] = n(None, None)
        out["pares"][m] = {f"{a:g}/{b:g}": n(b, a) for a, b in pares}
    return out


# ====================================================================================== órdenes
def fija_raiz(raiz):
    """Los runs de otro clon (como `teff_bayes.fija_raiz`, que cambia también la raíz de los scripts que lee)."""
    from scripts import teff_bayes as tb
    return tb.fija_raiz(raiz)


def con_cielo(raiz):
    """Los objetos cuyo QC de A4/M2 lista los SKY_SPECTRUM por exposición (hoy, solo el run de ROXs 12 b los lista;
    el de 42B b da la LSF sin la lista de ficheros)."""
    return tuple(s for s in objetos() if json.loads((_rp(s, raiz).stage_dir / "stage00q_qc.json").read_text())
                 ["m2_lsf"].get("sky_spectrum_files"))


def cmd_alas(raiz):
    out = {s: alas_cielo(s, raiz) for s in con_cielo(raiz)}
    if not out:
        raise SystemExit("ningún run del paper lista los SKY_SPECTRUM de A4/M2")
    from scripts.halpha_criterios import Contexto
    det = Contexto(raiz).detectado
    out["halpha"] = alas_halpha(det, raiz, perfil_fuga(out))
    for s in con_cielo(raiz):
        print(f"== {s} (run {_run(s)}): fracción fuera de ±v")
        for k, a in out[s].items():
            print(f"  {k}: n={a['n']} FWHM {a['fwhm_A']:.3f} Å (M2 {a['fwhm_m2_A']:.3f}), W10 {a['w10_A']:.3f}; "
                  f"lado {a['lado']} hasta {a['d_max_A']:.1f} Å ({a['v_max_kms']:.0f} km/s): " + "; ".join(
                      f"{v} {f['fraccion']:.4f}±{f['error']:.4f} (g {f['gauss']:.1e})" for v, f in a["fuera"].items()))
    h = out["halpha"]
    print(f"Hα de {det}: flujo en ±{D_TOTAL_HA:g} Å {h['flujo_total']:.1f} (G2 {h['flux_direct_g2']:.1f}); fuera: "
          + "; ".join(f"{k} {v['fraccion']:+.3f}±{v['error']:.3f} (sin resolver {v.get('modelo_en_canales', np.nan):.3f}, "
                      f"gauss {v.get('gauss_en_canales', np.nan):.4f})" for k, v in h["fuera"].items()))
    return out


def cmd_fuga(raiz, alas=None):
    alas = alas or {s: alas_cielo(s, raiz) for s in con_cielo(raiz)}
    alas = {s: alas[s] for s in con_cielo(raiz)}
    x, p = perfil_fuga({s: alas[s] for s in con_cielo(raiz)})
    out = {"perfil": dict(x=x.tolist(), p=p.tolist(), fwhm_A=ancho_cruce(x, p, 0.5), w10_A=ancho_cruce(x, p, 0.1))}
    for s in objetos():
        r = fuga(s, raiz, x, p)
        lineas = list(next(iter(r.values()))["lineas"])
        wmin = {c: {k: w_minima(r, k, c) for k in lineas} for c in CRITERIOS}
        for c in CRITERIOS:
            wmin[c]["todas"] = w_minima(r, None, c)
        out[s] = dict(por_w=r, w_minima={f"{c:g}": v for c, v in wmin.items()})
        print(f"== {s}: fuga máxima en σ del bin (criterio 0.1) y w mínima (0.05 / 0.1 / 0.2)")
        for k in lineas + ["todas"]:
            fila = (r["300"]["lineas"][k] if k != "todas" else r["300"]["todas"])
            nombre = fila.get("nombre", "todas") if k != "todas" else "todas"
            print(f"  {k:>8} {nombre:<12} " + " ".join(
                f"{w}:{(r[w]['lineas'][k] if k != 'todas' else r[w]['todas'])['z_max']:+.3f}" for w in r)
                  + f"  -> {wmin[0.05][k]} / {wmin[0.1][k]} / {wmin[0.2][k]}")
    return out


def cmd_bins(raiz, pares=()):
    out = {}
    for s in objetos():
        out[s] = bins(s, pares=pares)
        for m, v in out[s]["global"].items():
            print(f"{s} {m}: " + ", ".join(f"{k} {n}" for k, n in v.items()))
        for m, v in out[s]["pares"].items():
            if v:
                print(f"{s} {m} (w_Hα/w_resto): " + ", ".join(f"{k} {n}" for k, n in v.items()))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("orden", choices=("alas", "fuga", "bins", "todo"))
    ap.add_argument("--project-root", type=Path, default=None)
    ap.add_argument("--salida", type=Path, default=None)
    ap.add_argument("--pares", nargs="*", default=(), metavar="WHA/WRESTO",
                    help="pares w_Hα/w_resto (km/s) para `bins`, p. ej. 300/150")
    a = ap.parse_args(argv)
    warnings.simplefilter("ignore", RuntimeWarning)
    raiz = fija_raiz(a.project_root) if a.project_root else ROOT
    pares = [tuple(float(t) for t in x.split("/")) for x in a.pares]
    res = {}
    if a.orden in ("alas", "todo"):
        res["alas"] = cmd_alas(raiz)
    if a.orden in ("fuga", "todo"):
        res["fuga"] = cmd_fuga(raiz, res.get("alas"))
    if a.orden in ("bins", "todo"):
        res["bins"] = cmd_bins(raiz, pares)
    if a.salida:
        a.salida.mkdir(parents=True, exist_ok=True)
        for k, v in res.items():
            (a.salida / f"{k}.json").write_text(json.dumps(v, indent=1, ensure_ascii=False, default=float) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
