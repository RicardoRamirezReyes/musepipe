#!/usr/bin/env python
"""Las otras lineas con otras plantillas, otras extracciones y la sola dispersion de las nulas.

Es la «check outside the pipeline» del paper (Sect. 4.1 y Ap. C), pasada al repo: el metodo del
agente `lineas` (2026-10-26, `~/muse_sesion_20261026/claudio/lineas/`: `scripts/medir_lineas.py`,
`scripts/datos.py` y la parte de `scripts/tablas.py` de la luz de la primaria), portado TAL CUAL.
Con las mismas entradas reproduce sus salidas (`salidas/resumen.json`, `salidas/lineas_<obj>.csv`)
bit a bit. No es G2: es la comprobacion de cuanto dependen las otras lineas de lo que G2 fija (una
plantilla, una extraccion y el suelo del error propagado). Solo lee; escribe solo con `--json` o
`--csv`.

Para cada linea, en cada extraccion que conserva el continuo (la canonica de D2 primero y despues
las demas de `stage_x10_compare.CONTINUUM_METHODS`; SGF y LPM quitan el continuo y no admiten
plantilla) y con cada continuo (la recta; cada plantilla joven Clase III x polinomio de grado 2; una
mezcla lineal de todas, cada una x polinomio de grado 1):

- filtro adaptado gaussiano y flujo directo en ±6 A, con el continuo restado;
- la MISMA medida en los 33 controles calibrados (nula espacial) y en 22-42 longitudes de onda sin
  lineas del propio espectro (nula espectral, «placebos»);
- sigma = max(desviacion tipica de las dos nulas), z = F/sigma, limite 99 % = max(F, 0) + 2.326 sigma.

Y despues: la suma de seis Paschen con razones de caso B (Storey & Hummer 1995) y una sola amplitud,
la A_V que pediria esa amplitud frente a la Halpha detectada, si las Paschen siguen el caso B, y la
luz de la primaria (lo que restaria una fraccion f del continuo del compañero con el espectro de la
estrella).

**En que difiere de G2 v4** (`docs/spec_G2_v4_codex_spectral_measurements.md`; `musepipe.lines`):

1. **La sigma.** Aqui es la desviacion tipica (ddof=1) del flujo del mismo estimador en cada nula,
   la mayor de las dos, en flujo y **sin suelo**. En G2 v4 cada nula da una inflacion
   (cuantil 99 de sus z / 2.326), la escala aplicada es max(espacial, espectral, 1) y la sigma es la
   propagada por esa escala: nunca baja del error propagado, que en el rojo lleva el sistematico de
   la PSF y manda (las «p» de la Tabla C.1). Por eso, en las lineas del rojo donde manda, la sigma
   de G2 v4 es 0.9-2.7 veces la de aqui (medido el 2026-10-26 en los dos runs del paper) y los z de
   aqui salen mayores.
2. **El filtro.** Centrado en la sistemica, lambda_rest (1 + v_sys/c), con la FWHM de la LSF(lambda)
   del polinomio de A4/M2 en cada linea y una ventana de ±3 FWHM; O I 7773 lleva un filtro de
   triplete con pesos gf. G2 lo centra en la longitud de onda en reposo, con una sola LSF (la de
   Halpha) y el filtro de E1 sobre todo el continuo y la ventana de linea.
3. **La plantilla.** Preparada como en G3 (marco del dato, degradada y remuestreada) pero con la
   LSF(lambda); G2 la degrada a una sola LSF. Ademas de la que declara el run
   (`g2_continuum_template`) se miden las alternativas que se pasan con `--alternativas` y su
   mezcla. La mascara del ajuste es la de G2 v4 (`musepipe.lines.template_fit_mask`).
4. **La recta.** Las ventanas [-48, -8] y [+8, +48] A van centradas en la sistemica; las de G2, en
   la longitud de onda en reposo.
5. **El catalogo.** El de G2 (`stage07_accretion_lines.default_accretion_lines`) mas O I 7773 y
   [N II] 6548/6583, que entran en la exclusion de vecinas (las [N II] caen en las ventanas de Halpha).
6. **El throughput.** 1 (G2 v4 divide por el de E4 en Halpha, 1.01 y 1.00).
7. **Los placebos.** La misma malla y las mismas exclusiones que G2 v4
   (`musepipe.lines.placebo_centers`), salvo el margen del borde, 50 A tambien con plantilla (G2: 60).
8. **Lo que G2 no hace.** Varias plantillas y extracciones, la suma de Paschen, la A_V de su
   amplitud y la luz de la primaria.

De `musepipe` reutiliza, porque dan los mismos numeros bit a bit: `lines._channel_widths`,
`lines.template_fit_mask`, `lines._template_continuum`, `lines.placebo_centers`,
`stage_h03_limits.empirical_upper_quantile`, `case_b.case_b_ratio`,
`case_b.av_upper_limit_undetected_redder`, `extinction.CCMExtinction`, `telluric_lines`, los
lectores de D2 y la preparacion de plantillas de G3. Lo demas (el filtro de triplete, la recta
centrada en la sistemica, la mezcla y las nulas sin suelo) es del agente y no esta en `musepipe`.

Las cifras del paper (Sect. 4.1 y Ap. C) salen de:

    python scripts/otras_lineas_variantes.py ROXs12b --alternativas CHSM17173 TWA26
    python scripts/otras_lineas_variantes.py ROXs42Bb --alternativas TWA26
    # desde un worktree, con --project-root ../MUSE-accretion-pipeline

Las alternativas son las del agente: CHSM17173 (M8) y TWA26 (M9) para la M7.5 de ROXs 12 b y TWA26
(M9) para la M9.5 de ROXs 42B b, de la misma biblioteca `templates_young`. La fraccion de la luz
de la primaria, 6 %, es la cota de la Sect. 4.1 del paper para ROXs 12 b (1.54/26 A): es una
hipotesis que se evalua, no una medida, y se cambia con `--fraccion-primaria`.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from musepipe.lines import (  # noqa: E402
    _channel_widths, _template_continuum, placebo_centers, template_fit_mask)
from musepipe.models.case_b import av_upper_limit_undetected_redder, case_b_ratio  # noqa: E402
from musepipe.models.extinction import CCMExtinction  # noqa: E402
from musepipe.stages.stage07_accretion_lines import default_accretion_lines  # noqa: E402
from musepipe.stages.stage_h03_limits import empirical_upper_quantile  # noqa: E402
from musepipe.telluric_lines import AO_LASER_WINDOW_A, TELLURIC_BANDS, telluric_mask  # noqa: E402

C_KMS = 299792.458
Z99 = 2.3263478740408408
LINE_HALF_A = 6.0        # ventana de linea de G2
FIT_HALF_A = 60.0        # ventana del ajuste de plantilla (la de G2 v4)
MIN_FIT_PIXELS = 30      # como G2 v4
PLACEBO_EDGE_A = 50.0    # el agente: 50 A del borde tambien con plantilla (G2 v4: 60)
TELURICA_PAD_A = 6.0     # placebos a mas de 6 A de una banda fuerte (como G2 v4)
LASER_PAD_A = 50.0       # y a mas de 50 A del hueco del laser (como G2 v4)
#: canales con estas banderas no entran (FLAG_BAD_WINDOW | FLAG_SKYLINE, las de E1)
BAD_DETECTION_FLAGS = 1 | 2
#: la ley con la que se comparan las Paschen entre si y con Halpha (la del config de los runs)
CCM = CCMExtinction(rv=3.1, citation="Cardelli, Clayton & Mathis 1989 (ApJ 345, 245), R_V=3.1")

#: el catalogo de G2 mas lo que el agente anadio. O I 7771.94/7774.17/7775.39 con pesos
#: proporcionales a gf (log gf 0.369, 0.223, 0.002; NIST, no verificados por el agente: solo
#: dan la forma del filtro). Las [N II] solo se miden y se excluyen de los continuos.
EXTRA_CATALOGO = [
    {"name": "[N II] 6548", "wave_A": 6548.05, "family": "forbidden"},
    {"name": "[N II] 6583", "wave_A": 6583.45, "family": "forbidden"},
    {"name": "O I 7773", "wave_A": 7773.4, "family": "O I",
     "comps": [(7771.94, 10 ** 0.369), (7774.17, 10 ** 0.223), (7775.39, 10 ** 0.002)]},
]
CATALOGO = [{k: d[k] for k in ("name", "wave_A", "family")} for d in default_accretion_lines()] + EXTRA_CATALOGO

#: las que se MIDEN. Las Paschen fundidas con el triplete (Pa 13/15/16) y con O I (Pa 18) no se
#: miden aparte: a 2.3 A de LSF el filtro centrado en Ca II recoge la suma, y un limite sobre la
#: suma acota cada componente positiva. He I 5876 cae en el hueco del laser.
MEDIR = ["Hbeta", "He I 5016", "[O I] 6300", "[O I] 6364", "[N II] 6548", "Halpha",
         "[N II] 6583", "He I 6678", "[S II] 6716", "[S II] 6731", "He I 7065", "O I 7773",
         "O I 8446", "Pa 17", "Ca II 8498", "Ca II 8542", "Ca II 8662", "Pa 14", "Pa 12",
         "Pa 11", "Pa 10", "Pa 9"]
#: por encima de esto el continuo primario es la plantilla (como G2 v4)
TEMPLATE_MIN_A = 7000.0

#: las Paschen de la suma: las que no se funden con el triplete ni con O I
PASCHEN_STACK = ("Pa 9", "Pa 10", "Pa 11", "Pa 12", "Pa 14", "Pa 17")
#: el nodo del caso B y el barrido (T, n_e) del agente
NODO_CASO_B = (1.0e4, 1.0e2)
BARRIDO_CASO_B = ((1.0e4, 1.0e2), (5.0e3, 1.0e6), (2.0e4, 1.0e2))


# --------------------------------------------------------------------------- medida

def plantilla_filtro(wave, comps, fwhm, dl):
    """Perfil del filtro adaptado: suma de gaussianas de FWHM `fwhm`, normalizada a area 1."""
    sig = fwhm / 2.354820045
    t = np.zeros_like(wave)
    for lam, w in comps:
        t += w * np.exp(-0.5 * ((wave - lam) / sig) ** 2)
    return t / np.sum(t * dl)


def excl_vecinas(wave, catalogo, half=6.0):
    """±6 A alrededor de cada linea de `catalogo` (G2: `neighbor_exclude_A`)."""
    m = np.zeros(wave.shape, bool)
    for o in catalogo:
        m |= np.abs(wave - o["wave_A"]) <= half
    return m


def otras_lineas(catalogo, rest_A):
    """El catalogo sin la linea que se mide (la de `rest_A`)."""
    return [o for o in catalogo if abs(o["wave_A"] - rest_A) > 1e-6]


def continuo(kind, wave, y, good, centro, otras, T=None, tell=None):
    """Continuo en toda la malla, o None si no se puede ajustar.

    `lin`: recta en [-48, -8] U [+8, +48] A del centro, sin las vecinas (±6 A de `otras`).
    Plantilla(s): `T` (n_canales) o (n_plantillas, n_canales); una sola, x (c0 + c1 x + c2 x^2);
    varias, cada una x (c0 + c1 x). Se ajusta en la mascara de G2 v4 (`template_fit_mask`: ±60 A
    sin la ventana de linea, sin vecinas, sin banderas, sin `tell`) y con todas finitas.
    """
    if kind == "lin":
        excl = excl_vecinas(wave, otras)
        blue = good & ~excl & (wave >= centro - 48.0) & (wave <= centro - 8.0)
        red = good & ~excl & (wave >= centro + 8.0) & (wave <= centro + 48.0)
        m = (blue | red) & np.isfinite(y)
        if m.sum() < 6:
            return None
        p = np.polyfit(wave[m] - centro, y[m], 1)
        return np.polyval(p, wave - centro)
    T2 = np.atleast_2d(T)
    tf = np.all(np.isfinite(T2), axis=0)
    fitw = (template_fit_mask(wave, centro, T2[0], catalog=otras, bad_flags=~good, half_A=FIT_HALF_A,
                              core_A=LINE_HALF_A, neighbor_exclude_A=6.0, exclude_mask=tell)
            & tf & np.isfinite(y))
    core = np.abs(wave - centro) <= LINE_HALF_A
    if fitw.sum() < MIN_FIT_PIXELS or not np.all(tf[core]):
        return None
    if T2.shape[0] == 1:
        return _template_continuum(wave, y, fitw, T2[0], centro, FIT_HALF_A)[1]
    x = (wave - centro) / FIT_HALF_A
    A = np.vstack([c for t in T2 for c in (t, t * x)]).T
    coef, *_ = np.linalg.lstsq(A[fitw], y[fitw], rcond=None)
    return A @ coef


def medir_uno(kind, wave, y, err, good, centro, comps, fwhm, otras, dl, T=None, tell=None):
    """`{f_mf, f_dir, ew, cont, sigma_prop_mf}` de una medida, o None."""
    cont = continuo(kind, wave, y, good, centro, otras, T, tell)
    if cont is None:
        return None
    r = y - cont
    core = good & (np.abs(wave - centro) <= LINE_HALF_A) & np.isfinite(r)
    if core.sum() < 6:
        return None
    f_dir = float(np.sum(r[core] * dl[core]))
    c_core = cont[core]
    ew = float(-np.sum(r[core] / c_core * dl[core])) if np.all(c_core > 0) else np.nan
    t = plantilla_filtro(wave, comps, fwhm, dl)
    span = 3.0 * fwhm + max(abs(l - centro) for l, _ in comps)
    mm = good & (np.abs(wave - centro) <= span) & np.isfinite(r) & np.isfinite(err) & (err > 0)
    iv = 1.0 / err[mm] ** 2
    den = float(np.sum(t[mm] ** 2 * iv))
    f_mf = float(np.sum(r[mm] * t[mm] * iv) / den)
    i0 = int(np.argmin(np.abs(wave - centro)))
    return {"f_mf": f_mf, "f_dir": f_dir, "ew": ew, "cont": float(cont[i0]),
            "sigma_prop_mf": float(1.0 / math.sqrt(den))}


def placebos(wave, good, centro, catalogo, T=None):
    """Centros de la nula espectral: los de `musepipe.lines.placebo_centers` (margen 50 A) y, con
    plantilla, con todas las plantillas finitas en ±60 A."""
    fuertes = [b for b in TELLURIC_BANDS if b["severity"] == "strong"]
    excl = tuple((b["lo_A"], b["hi_A"], TELURICA_PAD_A) for b in fuertes) + (
        (AO_LASER_WINDOW_A[0], AO_LASER_WINDOW_A[1], LASER_PAD_A),)
    out = placebo_centers(wave, centro, catalogo, bad_flags=~good, edge_A=PLACEBO_EDGE_A,
                          exclude_ranges=excl, line_half_A=LINE_HALF_A)
    if T is not None:
        T2 = np.atleast_2d(T)
        out = [lp for lp in out if np.all(np.isfinite(T2[:, np.abs(wave - lp) <= FIT_HALF_A]))]
    return out


def resumen_nula(vals):
    """Desviacion tipica (ddof=1), MAD, cuantil 99 (el estimador de E3) y mediana; NaN con < 5."""
    v = np.asarray([x for x in vals if np.isfinite(x)], float)
    if v.size < 5:
        return {"n": int(v.size), "std": np.nan, "rstd": np.nan, "q99": np.nan, "median": np.nan}
    mad = 1.4826 * np.median(np.abs(v - np.median(v)))
    return {"n": int(v.size), "std": float(np.std(v, ddof=1)), "rstd": float(mad),
            "q99": float(empirical_upper_quantile(v, fap=0.01)), "median": float(np.median(v))}


def medir_linea(name, wave, obj, ctrls, good, err, vsys, lsf_fn, kind, T, catalogo, dl, tell=None):
    """La medida de una linea con un continuo, sus dos nulas y su sigma (la mayor de las dos)."""
    d = next(c for c in catalogo if c["name"] == name)
    fac = 1.0 + vsys / C_KMS
    centro = d["wave_A"] * fac
    comps = [(l * fac, w) for l, w in d.get("comps", [(d["wave_A"], 1.0)])]
    fwhm = float(lsf_fn(centro))
    otras = otras_lineas(catalogo, d["wave_A"])
    o = medir_uno(kind, wave, obj, err, good, centro, comps, fwhm, otras, dl, T, tell)
    if o is None:
        return None
    cs = [medir_uno(kind, wave, c, err, good & np.isfinite(c), centro, comps, fwhm, otras, dl, T, tell)
          for c in ctrls]
    plac_l = placebos(wave, good, centro, catalogo, T if kind != "lin" else None)
    ps = []
    for lp in plac_l:
        sh = lp - centro
        comps_p = [(l + sh, w) for l, w in comps]
        r = medir_uno(kind, wave, obj, err, good, lp, comps_p, fwhm, catalogo, dl, T, tell)
        if r is not None:
            ps.append(r)
    out = {"line": name, "rest_A": d["wave_A"], "center_A": centro, "fwhm_lsf_A": fwhm,
           "continuum": kind, **{k: o[k] for k in o}}
    for est in ("f_mf", "f_dir"):
        ns = resumen_nula([c[est] for c in cs if c is not None])
        npl = resumen_nula([p[est] for p in ps])
        F = o[est]
        sig = np.nanmax([ns["std"], npl["std"]])
        out.update({
            f"{est}_sig_sp": ns["std"], f"{est}_rsig_sp": ns["rstd"], f"{est}_q99_sp": ns["q99"],
            f"{est}_med_sp": ns["median"], f"{est}_n_sp": ns["n"],
            f"{est}_sig_spec": npl["std"], f"{est}_rsig_spec": npl["rstd"], f"{est}_q99_spec": npl["q99"],
            f"{est}_med_spec": npl["median"], f"{est}_n_spec": npl["n"],
            f"{est}_sig": float(sig),
            f"{est}_z": float(F / sig) if sig > 0 else np.nan,
            f"{est}_z_sp": float(F / ns["std"]) if ns["std"] > 0 else np.nan,
            f"{est}_z_spec": float(F / npl["std"]) if npl["std"] > 0 else np.nan,
            f"{est}_fap_sp": float(np.mean([c[est] >= F for c in cs if c is not None])),
            f"{est}_fap_spec": float(np.mean([p[est] >= F for p in ps])) if ps else np.nan,
            f"{est}_ul99": float(max(F, 0.0) + Z99 * sig),
            f"{est}_ul5s": float(5.0 * sig),
        })
    out["ew_sig_sp"] = resumen_nula([c["f_dir"] / o["cont"] for c in cs if c is not None])["std"] \
        if o["cont"] > 0 else np.nan
    z = out["f_mf_z"]
    out["status"] = ("detected" if z >= 5 else "marginal" if z >= 3 else "upper_limit")
    out["n_placebo"] = len(ps)
    out["_ctrl_f_mf"] = [c["f_mf"] if c is not None else np.nan for c in cs]
    out["_plac_f_mf"] = [p["f_mf"] for p in ps]
    out["_plac_l"] = plac_l
    return out


# --------------------------------------------------------------------------- Paschen y A_V

def a_lambda(av, wave_A):
    """A_lambda (mag) con CCM89, R_V = 3.1."""
    return float(np.atleast_1d(CCM.a_lambda_over_av(wave_A))[0]) * float(av)


def pesos_caso_b(lineas_rest, av, *, T_K=1.0e4, ne=1.0e2, referencia="Pa 9", rest_ref=9229.01):
    """Flujo OBSERVADO de cada linea relativo a `referencia` si siguen el caso B (Storey & Hummer
    1995) con la extincion diferencial a `av`. `lineas_rest`: {nombre: lambda en reposo}."""
    k9 = a_lambda(1.0, rest_ref)
    return np.array([case_b_ratio(n, referencia, temperature_K=T_K, density_cm3=ne)
                     * 10 ** (-0.4 * av * (a_lambda(1.0, lam) - k9)) for n, lam in lineas_rest.items()])


def amplitud_conjunta(F, sig, w, controles=None):
    """La amplitud A de `F_i = A w_i`, ponderada por 1/sigma_i^2, y la misma combinacion en los
    controles (columnas de `controles`, (n_lineas, n_controles))."""
    F, sig, w = (np.asarray(v, dtype=np.float64) for v in (F, sig, w))
    W = w / sig ** 2
    den = float(np.sum(W * w))
    A = float(np.sum(W * F) / den)
    Ac = None if controles is None else np.sum(W[:, None] * np.asarray(controles), axis=0) / den
    return A, W, den, Ac


def apilar_paschen(filas, met, kind, av, T_K=1.0e4, ne=1.0e2, lineas=PASCHEN_STACK):
    """Amplitud conjunta de las Paschen NO fundidas con el triplete ni con O I, con pesos de caso B
    relativos a Pa 9 y la extincion diferencial al A_V adoptado: A estima F_obs(Pa 9) si siguen el
    caso B.

    Ponderacion por la varianza EMPIRICA de cada linea (max de las dos nulas). Nula espacial: la
    misma combinacion en cada control (conserva la correlacion entre lineas de un control). Nula
    espectral: propagada desde la sigma de placebo de cada linea (supone desajustes independientes) y,
    aparte, CORRELADA: un mismo desplazamiento para todas las lineas, con el placebo de cada una a
    lambda_i + Delta, rehecha sin la que no lo tenga (al menos 4 de las 6)."""
    rows = {f["line"]: f for f in filas if f["method"] == met and f["continuum"] == kind}
    if not all(n in rows for n in lineas):
        return None
    w = pesos_caso_b({n: rows[n]["rest_A"] for n in lineas}, av, T_K=T_K, ne=ne)
    F = np.array([rows[n]["f_mf"] for n in lineas])
    sig = np.array([rows[n]["f_mf_sig"] for n in lineas])
    ssp = np.array([rows[n]["f_mf_sig_spec"] for n in lineas])
    C = np.array([rows[n]["_ctrl_f_mf"] for n in lineas])  # (nl, 33)
    A, W, den, Ac = amplitud_conjunta(F, sig, w, C)
    sig_sp = float(np.nanstd(Ac, ddof=1))
    sig_spec = float(np.sqrt(np.sum((W * ssp) ** 2)) / den)
    s_ = max(sig_sp, sig_spec)
    from musepipe.lines import PLACEBO_MAX_A, PLACEBO_MIN_A, PLACEBO_STEP_A
    shifts = np.arange(-PLACEBO_MAX_A, PLACEBO_MAX_A + 1e-6, PLACEBO_STEP_A)
    Acorr = []
    for dlt in shifts:
        if abs(dlt) < PLACEBO_MIN_A:
            continue
        num = den_c = 0.0
        n_ok = 0
        for i, n in enumerate(lineas):
            pl = np.asarray(rows[n]["_plac_l"])
            pv = np.asarray(rows[n]["_plac_f_mf"])
            j = np.where(np.abs(pl - (rows[n]["center_A"] + dlt)) < 0.01)[0]
            if j.size:
                num += W[i] * pv[j[0]]
                den_c += W[i] * w[i]
                n_ok += 1
        if n_ok >= 4:
            Acorr.append(num / den_c)
    Acorr = np.asarray(Acorr)
    sig_spec_corr = float(np.std(Acorr, ddof=1)) if Acorr.size >= 5 else np.nan
    return {"method": met, "continuum": kind, "lines": list(lineas), "T_K": T_K, "ne": ne,
            "sig_spec_corr": sig_spec_corr, "n_spec_corr": int(Acorr.size),
            "z_corr": A / max(sig_sp, sig_spec_corr) if np.isfinite(sig_spec_corr) else np.nan,
            "fap_spec_corr": float(np.mean(Acorr >= A)) if Acorr.size else np.nan,
            "weights_rel_Pa9": dict(zip(lineas, map(float, w))), "A_Pa9": A, "sig_sp": sig_sp,
            "sig_spec": sig_spec, "sig": s_, "z": A / s_, "fap_sp": float(np.mean(Ac >= A)),
            "ul99_Pa9": max(A, 0.0) + Z99 * s_, "ul5s_Pa9": 5.0 * s_}


def razones_caso_b(filas, met, kind, av, T_K=1.0e4, ne=1.0e2, lineas=PASCHEN_STACK):
    """¿Siguen las Paschen medidas el caso B? (no lo calculaba el agente; lo dice su informe).

    Ajusta la amplitud como `apilar_paschen` y da, por linea, el flujo medido, el que pide el caso B
    con esa amplitud y el residuo en sigmas (la de la linea); y el chi^2 de las seis con n-1 grados de
    libertad. Las sigmas de lineas vecinas no son independientes (comparten plantilla), asi que el
    chi^2 es orientativo. Da tambien las razones Pa n / Pa 14, medidas y de caso B.
    """
    from scipy.stats import chi2 as _chi2
    rows = {f["line"]: f for f in filas if f["method"] == met and f["continuum"] == kind}
    if not all(n in rows for n in lineas):
        return None
    w = pesos_caso_b({n: rows[n]["rest_A"] for n in lineas}, av, T_K=T_K, ne=ne)
    F = np.array([rows[n]["f_mf"] for n in lineas])
    sig = np.array([rows[n]["f_mf_sig"] for n in lineas])
    A = amplitud_conjunta(F, sig, w)[0]
    res = (F - A * w) / sig
    chi2 = float(np.sum(res ** 2))
    ref = "Pa 14" if "Pa 14" in lineas else lineas[-1]
    i14 = list(lineas).index(ref)
    return {"method": met, "continuum": kind, "T_K": T_K, "ne": ne, "A_Pa9": A,
            "por_linea": {n: {"F": float(F[i]), "caso_b": float(A * w[i]), "residuo_sigma": float(res[i]),
                              f"medido_sobre_{ref}": float(F[i] / F[i14]),
                              f"caso_b_sobre_{ref}": float(w[i] / w[i14])}
                          for i, n in enumerate(lineas)},
            "chi2": chi2, "dof": len(lineas) - 1, "p": float(_chi2.sf(chi2, len(lineas) - 1))}


def av_para_paschen(F_ha_obs, F_pa9, T_K=1.0e4, ne=1.0e2):
    """La A_V a la que el Pa 9 de caso B, escalado desde la Halpha observada, vale `F_pa9`:
    2.5 log10(R F_pa9 / F_Ha) / (k_Ha - k_Pa9), R = Halpha/Pa 9 de caso B. Con un limite es la cota
    superior de A_V; con la amplitud de la suma, la A_V que pediria esa amplitud."""
    R = case_b_ratio("Halpha", "Pa 9", temperature_K=T_K, density_cm3=ne)
    return av_upper_limit_undetected_redder(F_ha_obs, F_pa9, R, a_lambda(1.0, 6562.80),
                                            a_lambda(1.0, 9229.01)), R


def pa9_caso_b(F_ha_obs, av, T_K=1.0e4, ne=1.0e2):
    """El Pa 9 OBSERVADO que el caso B predice desde la Halpha observada a la A_V `av`."""
    R = case_b_ratio("Halpha", "Pa 9", temperature_K=T_K, density_cm3=ne)
    return F_ha_obs / R * 10 ** (-0.4 * av * (a_lambda(1.0, 9229.01) - a_lambda(1.0, 6562.80)))


# --------------------------------------------------------------------------- resumen y fuga

def primaria(lam, primaria_tmpl):
    """El continuo primario: la recta por debajo de 7000 A, la plantilla del run por encima."""
    return "lin" if lam < TEMPLATE_MIN_A else f"tmpl:{primaria_tmpl}"


def resumen(filas, unit, primaria_tmpl, canonico="psffit"):
    """Una fila por linea: la medida primaria (extraccion canonica, continuo primario), el rango de z
    entre continuos validos y extracciones, y el limite conservador (el maximo UL99 entre ellos)."""
    out = []
    for name in MEDIR:
        sel = [f for f in filas if f["line"] == name]
        if not sel:
            continue
        lam = sel[0]["rest_A"]
        kp = primaria(lam, primaria_tmpl)
        pr = next((f for f in sel if f["method"] == canonico and f["continuum"] == kp), None)
        if pr is None:
            continue
        valid = ["lin"] if lam < TEMPLATE_MIN_A else [k for k in {f["continuum"] for f in sel}
                                                       if k.startswith("tmpl:")]
        alt = [f for f in sel if f["continuum"] in valid]
        metodos = list(dict.fromkeys(f["method"] for f in filas))
        out.append({
            "line": name, "rest_A": lam, "continuum_primary": kp,
            "F_mf": pr["f_mf"] * unit, "sig": pr["f_mf_sig"] * unit, "sig_sp": pr["f_mf_sig_sp"] * unit,
            "sig_spec": pr["f_mf_sig_spec"] * unit, "z": pr["f_mf_z"], "z_sp": pr["f_mf_z_sp"],
            "fap_sp": pr["f_mf_fap_sp"], "fap_spec": pr["f_mf_fap_spec"],
            "ul99": pr["f_mf_ul99"] * unit, "ul5s": pr["f_mf_ul5s"] * unit,
            "F_dir": pr["f_dir"] * unit, "sig_dir": pr["f_dir_sig"] * unit, "ew_A": pr["ew"],
            "ew_sig_A": pr["f_dir_sig"] / pr["cont"] if pr["cont"] > 0 else np.nan,
            "cont_cgsA": pr["cont"] * unit,
            "z_range_psffit": [min(f["f_mf_z"] for f in alt if f["method"] == canonico),
                               max(f["f_mf_z"] for f in alt if f["method"] == canonico)],
            "z_range_all": [min(f["f_mf_z"] for f in alt), max(f["f_mf_z"] for f in alt)],
            "F_range_psffit": [min(f["f_mf"] for f in alt if f["method"] == canonico) * unit,
                               max(f["f_mf"] for f in alt if f["method"] == canonico) * unit],
            "ul99_cons": max(f["f_mf_ul99"] for f in alt) * unit,
            "z_by_method": {m: next((f["f_mf_z"] for f in sel if f["method"] == m and f["continuum"] == kp), None)
                            for m in metodos},
            "status": pr["status"],
        })
    return out


def ew_estrella(star, vsys, lsf_fn, catalogo=CATALOGO):
    """EW de cada linea en la primaria (recta de G2, filtro adaptado; + = absorcion)."""
    sw, sf = star["wave_A"], star["flux"]
    sgood = np.isfinite(sf) & ((star["flags"] & BAD_DETECTION_FLAGS) == 0)
    sdl = _channel_widths(sw)
    out = {}
    for name in MEDIR:
        d = next(c for c in catalogo if c["name"] == name)
        fac = 1.0 + vsys / C_KMS
        comps = [(l * fac, w) for l, w in d.get("comps", [(d["wave_A"], 1.0)])]
        rr = medir_uno("lin", sw, sf, star["flux_err_total"], sgood, d["wave_A"] * fac, comps,
                       float(lsf_fn(d["wave_A"])), otras_lineas(catalogo, d["wave_A"]), sdl)
        if rr is not None:
            out[name] = {"ew_dir_A": rr["ew"],
                         "ew_mf_A": -rr["f_mf"] / rr["cont"] if rr["cont"] > 0 else np.nan,
                         "cont": rr["cont"]}
    return out


def fuga_primaria(res, estrella, fraccion, sigma_g2=None):
    """Lo que restaria (o sumaria) a cada linea una fraccion `fraccion` del continuo del compañero
    que fuera luz de la primaria: f F_c (-EW_estrella). En flujo y en sigmas: la del agente (`sig`) y,
    si se pasa (`sigma_g2`, {linea: sigma en cgs}), la de G2 v4 (`flux_mf_sigma_emp`)."""
    out = {}
    for r in res:
        e = estrella.get(r["line"])
        if not e:
            continue
        fuga = fraccion * r["cont_cgsA"] * (-e["ew_mf_A"])
        fila = {"ew_estrella_mf_A": e["ew_mf_A"], "cont_cgsA": r["cont_cgsA"], "fuga_cgs": fuga,
                "sigma_cgs": r["sig"], "fuga_sobre_sigma": fuga / r["sig"]}
        if sigma_g2 and r["line"] in sigma_g2:
            fila["sigma_g2_cgs"] = sigma_g2[r["line"]]
            fila["fuga_sobre_sigma_g2"] = fuga / sigma_g2[r["line"]]
        out[r["line"]] = fila
    return out


# --------------------------------------------------------------------------- lectura

def lsf_de_a4(qc00):
    """FWHM(lambda) de la LSF de A4/M2 (polinomio de grado 2 sobre lineas de cielo)."""
    m2 = qc00["m2_lsf"]
    c = np.asarray(m2["poly2_coeffs"], dtype=np.float64)
    return (lambda w: np.polyval(c, np.asarray(w, dtype=np.float64))), float(m2["lsf_fwhm_at_halpha_A"])


def leer_espectro(path):
    from musepipe.extraction.product import read_spectrum_product
    p = read_spectrum_product(path)
    return {"wave_A": np.asarray(p.wave_A, dtype=np.float64), "flux": np.asarray(p.flux, dtype=np.float64),
            "flux_err_total": np.asarray(p.extra_columns["flux_err_total"], dtype=np.float64),
            "flags": np.asarray(p.flags, dtype=np.int64)}


def plantillas(cfg, proyecto, biblioteca, nombres, wave, lsf_fn):
    """Las plantillas `nombres` de `biblioteca`, preparadas como en G3 (marco del dato, degradadas a
    la LSF(lambda), remuestreadas a `wave`; sin extincion ni escala). En el orden de la biblioteca."""
    from musepipe.models.libraries import data_frame
    from musepipe.models.prep import prepare_template_base
    from musepipe.models.template_fit import declared_prep_kwargs
    from musepipe.stages.stage_g3_template_fit import _build_libraries
    c = dict(cfg)
    c["project_root"] = str(proyecto)
    libs = {lib.name: lib for lib in _build_libraries(c)}
    if biblioteca not in libs:
        raise SystemExit(f"la biblioteca {biblioteca!r} no esta en g3_template_libraries ({sorted(libs)})")
    lib = libs[biblioteca]
    out = {}
    for code, path, meta in lib.entries():
        if meta.get("object") in nombres:
            t = lib.load_entry(code, path, meta)
            kw = declared_prep_kwargs(t, data_frame=data_frame(cfg))
            flux, mismatch = prepare_template_base(t, wave, lsf_fwhm_A=float(lsf_fn(6562.80)),
                                                   target_fwhm_A=lsf_fn, **kw)
            m = {k: t.meta.get(k) for k in ("object", "spt", "declared_resolution_R",
                                             "declared_wave_frame", "citation", "source_npz")}
            m["resolution_mismatch"] = bool(mismatch)
            out[meta.get("object")] = (np.asarray(flux, dtype=np.float64), m)
    faltan = [n for n in nombres if n not in out]
    if faltan:
        raise SystemExit(f"{biblioteca} no tiene {faltan}")
    return out


def metodos_con_continuo(obj):
    """La extraccion canonica de D2 primero y despues las demas que conservan el continuo."""
    from musepipe.stages.stage_x10_compare import CONTINUUM_METHODS
    canon = obj.qc("stage_x11_qc.json")["canonical_method"]
    if canon not in CONTINUUM_METHODS:
        raise SystemExit(f"el canonico {canon!r} no conserva el continuo")
    return (canon,) + tuple(m for m in CONTINUUM_METHODS if m != canon)


def correr(obj, proyecto, alternativas, *, excluir_teluricas=True):
    """Todas las medidas de un objeto del paper (`scripts.paper_figures.Objeto`)."""
    from musepipe.io import load_calibrated_controls, resolve_flux_unit
    cfg = obj.config
    decl = cfg.get("g2_continuum_template")
    if not decl:
        raise SystemExit(f"{obj.run_id} no declara g2_continuum_template (spec G2 v4 §1)")
    biblioteca, prim = str(decl["library"]), str(decl["object"])
    vsys = float(cfg["h03_rv_sys_kms"])
    av, d_pc = float(cfg["h03_av"]), float(cfg["h03_distance_pc"])
    lsf_fn, lsf_ha = lsf_de_a4(obj.qc("stage00q_qc.json"))
    metodos = metodos_con_continuo(obj)
    nombres = [prim] + [a for a in alternativas if a != prim]
    filas, tmpl, wave_ref, unit = [], {}, None, None
    for met in metodos:
        o_path = obj.etapa(f"spec_calibrated_{met}_object.fits")
        s = leer_espectro(o_path)
        wave, flux, err = s["wave_A"], s["flux"], s["flux_err_total"]
        if unit is None:
            from astropy.io import fits
            with fits.open(o_path) as h:
                unit, _fuente = resolve_flux_unit(cfg, bunit=h["SPECTRUM"].header.get("BUNIT"))
        good = np.isfinite(wave) & np.isfinite(flux) & ((s["flags"] & BAD_DETECTION_FLAGS) == 0)
        ctrls = np.atleast_2d(load_calibrated_controls(obj.etapa(f"spec_calibrated_{met}_controls.npz"),
                                                       object_path=o_path))
        dl = _channel_widths(wave)
        if wave_ref is None:
            wave_ref = wave
            tmpl = plantillas(cfg, proyecto, biblioteca, nombres, wave, lsf_fn)
        elif not np.array_equal(wave, wave_ref):
            raise SystemExit(f"{met}: la malla no es la de {metodos[0]}; las plantillas son de esa malla")
        tell = telluric_mask(wave, severities=("strong",)) if excluir_teluricas else None
        kinds = [("lin", None)] + [(f"tmpl:{k}", v[0]) for k, v in tmpl.items()]
        if len(tmpl) > 1:
            kinds.append(("tmpl:mix", np.vstack([v[0] for v in tmpl.values()])))
        for name in MEDIR:
            for kind, T in kinds:
                r = medir_linea(name, wave, flux, ctrls, good, err, vsys, lsf_fn, kind, T, CATALOGO, dl, tell)
                if r is None:
                    continue
                r.update({"object": obj.slug, "method": met})
                filas.append(r)
    estrella = ew_estrella(leer_espectro(obj.etapa("spec_calibrated_psffit_star.fits")), vsys, lsf_fn)
    meta = dict(vsys=vsys, av=av, av_err=float(cfg.get("h03_av_err", 0)), d_pc=d_pc, unit=unit,
                lsf_ha=lsf_ha, metodos=list(metodos), biblioteca=biblioteca, primaria=prim,
                continuos_plantilla=[f"tmpl:{n}" for n in nombres] + (["tmpl:mix"] if len(tmpl) > 1 else []),
                plantillas={k: v[1] for k, v in tmpl.items()}, excluir_teluricas=bool(excluir_teluricas))
    return filas, estrella, meta


def cotas_av(res, apil, F_ha_dir, F_ha_mf, unit, primaria_tmpl, canonico="psffit"):
    """Las cotas de A_V del agente (Pa 9 sola, la suma, el decremento de Balmer) con los flujos de
    Halpha que se pasen. Solo tienen sentido con la Halpha detectada.

    `main` pasa los de G2 (`flux_direct` y `flux_mf` de Halpha). El agente usaba los del paper
    redondeados (2.38e-17 y 1.87e-17, este del filtro de E1, no del de G2): con ellos estas cotas
    salen identicas a las suyas. No son cifras del paper (las de la Sect. 5 salen de
    `paper_cifras_extincion.py` con los limites de G2 v4)."""
    cotas = {}
    p9 = next(r for r in res if r["line"] == "Pa 9")
    cotas["Pa9_ul99_primary"] = av_para_paschen(F_ha_dir, p9["ul99"])[0]
    cotas["Pa9_ul99_cons"] = av_para_paschen(F_ha_dir, p9["ul99_cons"])[0]
    cotas["Pa9_ul5s_primary"] = av_para_paschen(F_ha_dir, p9["ul5s"])[0]
    for a in apil:
        if a["method"] == canonico:
            key = f"stack_{a['continuum']}_T{int(a['T_K'])}_ne{a['ne']:.0e}"
            cotas[key] = av_para_paschen(F_ha_dir, a["ul99_Pa9"] * unit, a["T_K"], a["ne"])[0]
    prim = next(a for a in apil if a["method"] == canonico and a["continuum"] == f"tmpl:{primaria_tmpl}"
                and a["T_K"] == NODO_CASO_B[0] and a["ne"] == NODO_CASO_B[1])
    cotas["stack_primary_with_Ha_mf"] = av_para_paschen(F_ha_mf, prim["ul99_Pa9"] * unit)[0]
    hb = next(r for r in res if r["line"] == "Hbeta")
    R_ab = case_b_ratio("Halpha", "Hbeta")
    dk = a_lambda(1.0, 4861.33) - a_lambda(1.0, 6562.80)
    for tag, fha in (("Ha_dir", F_ha_dir), ("Ha_mf", F_ha_mf)):
        for ul in ("ul99", "ul5s"):
            cotas[f"balmer_AV_min_{ul}_{tag}"] = -2.5 * math.log10(R_ab * hb[ul] / fha) / dk
    cotas["caseB_Ha_Hb"] = R_ab
    return cotas


# --------------------------------------------------------------------------- salida

def _imprime(nombre, run, meta, res, apil, cifras_):
    print(f"% ---- {nombre} ({run}) ----")
    plant = ", ".join(f"{k} ({v['spt']})" for k, v in meta["plantillas"].items())
    print(f"% A_V {meta['av']}, d {meta['d_pc']} pc, v_sys {meta['vsys']} km/s; LSF(Halpha) {meta['lsf_ha']:.3f} A; "
          f"plantillas {plant}; primaria {meta['primaria']}; extracciones {', '.join(meta['metodos'])}")
    print(f"{'linea':12s} {'cont.':10s} {'F_mf':>7s} {'sigma (sp/spec)':>18s} {'z [rango]':>20s} "
          f"{'UL99 (cons.)':>13s} {'5sig':>6s}   (1e-18 erg/s/cm2)")
    for r in res:
        print(f"{r['line']:12s} {r['continuum_primary'].replace('tmpl:', ''):10s} {r['F_mf'] * 1e18:+7.2f} "
              f"{r['sig'] * 1e18:5.2f} ({r['sig_sp'] * 1e18:4.2f}/{r['sig_spec'] * 1e18:4.2f}) "
              f"{r['z']:+6.2f} [{r['z_range_all'][0]:+5.1f}, {r['z_range_all'][1]:+5.1f}] "
              f"{r['ul99'] * 1e18:5.1f} ({r['ul99_cons'] * 1e18:4.1f}) {r['ul5s'] * 1e18:6.1f}")
    for linea, tabla in cifras_["z_por_variante"].items():
        print(f"{linea}: z por extraccion y continuo")
        for met, por in tabla.items():
            print("  " + f"{met:15s} " + "  ".join(f"{k.replace('tmpl:', '')} {v:+.2f}" for k, v in por.items()))
    print(f"Suma de Paschen ({', '.join(PASCHEN_STACK)}; caso B {NODO_CASO_B[0]:g} K, "
          f"{NODO_CASO_B[1]:g} cm-3), A = F(Pa 9) en 1e-18:")
    for a in apil:
        if (a["T_K"], a["ne"]) != NODO_CASO_B:
            continue
        print(f"  {a['method']:15s} {a['continuum'][5:]:10s} A {a['A_Pa9'] * meta['unit'] * 1e18:+6.2f}  "
              f"z {a['z']:+5.2f}  z_corr {a['z_corr']:+5.2f}  UL99 {a['ul99_Pa9'] * meta['unit'] * 1e18:5.1f}")
    for grupo, v in cifras_["paschen_z_corr"].items():
        print(f"  z_corr con {grupo}: {v['min']:+.2f} a {v['max']:+.2f} (max en {v['en_max']})")
    if "paschen_z_corr_max" in cifras_:
        v = cifras_["paschen_z_corr_max"]
        print(f"  z_corr maximo: {v['z_corr']:+.2f} ({v['method']}, {v['continuum'][5:]})")
    if "paschen_av" in cifras_:
        p = cifras_["paschen_av"]
        print(f"  la amplitud ({p['variante']}) es {p['A_cgs']:.2e}: {p['veces_caso_b']:.1f} veces el Pa 9 de caso B "
              f"desde la Halpha a A_V = {meta['av']} ({p['pa9_caso_b_cgs']:.2e}); pediria A_V = {p['av']:.2f} mag")
    if "razones" in cifras_:
        z = cifras_["razones"]
        print(f"  ¿caso B? ({z['method']}, {z['continuum'][5:]}): chi2 = {z['chi2']:.1f} con {z['dof']} gl "
              f"(p = {z['p']:.2g}; orientativo)")
        for n, v in z["por_linea"].items():
            k = [c for c in v if c.startswith("medido_sobre_")][0]
            kb = k.replace("medido", "caso_b")
            print(f"    {n:6s} F {v['F'] * meta['unit'] * 1e18:+6.2f}  caso B {v['caso_b'] * meta['unit'] * 1e18:6.2f}  "
                  f"residuo {v['residuo_sigma']:+5.2f} sigma   {k.replace('medido_sobre_', '/')}: medido "
                  f"{v[k]:+5.2f}, caso B {v[kb]:4.2f}")
    f = cifras_["fuga"]
    print(f"Luz de la primaria con f = {100 * f['fraccion']:.0f} %: la mayor es {f['linea_max']} "
          f"({f['fuga_cgs'] * 1e18:+.2f}e-18 = {f['max_sobre_sigma']:+.2f} sigma del agente"
          + (f", {f['max_sobre_sigma_g2']:+.2f} sigma de G2 v4 ({f['linea_max_g2']})" if "max_sobre_sigma_g2" in f else "")
          + ")")


def cifras(filas, res, apil, fuga, meta, F_ha_dir=None):
    """Lo que cita el paper, para imprimir: z de O I 8446 por variante, rangos de la suma de
    Paschen, la A_V de su amplitud y el maximo de la luz de la primaria."""
    out = {"z_por_variante": {}}
    for linea in ("O I 8446",):
        tabla = {}
        for f in filas:
            if f["line"] == linea and f["continuum"].startswith("tmpl:"):
                tabla.setdefault(f["method"], {})[f["continuum"]] = f["f_mf_z"]
        out["z_por_variante"][linea] = tabla
    prim = f"tmpl:{meta['primaria']}"
    nodo = [a for a in apil if (a["T_K"], a["ne"]) == NODO_CASO_B]
    # el z de la suma con la nula espectral CORRELADA (`z_corr`, el que cita el paper), por
    # plantilla y entre las cuatro extracciones
    out["paschen_z_corr"] = {}
    for k in meta["continuos_plantilla"]:
        g = [a for a in nodo if a["continuum"] == k]
        if g:
            best = max(g, key=lambda a: a["z_corr"])
            out["paschen_z_corr"][k[5:]] = {"min": min(a["z_corr"] for a in g), "max": best["z_corr"],
                                           "en_max": f"{best['method']}"}
    if nodo:
        best = max(nodo, key=lambda a: a["z_corr"])
        out["paschen_z_corr_max"] = {"z_corr": best["z_corr"], "method": best["method"],
                                     "continuum": best["continuum"]}
    canon = meta["metodos"][0]
    ap = next((a for a in nodo if a["method"] == canon and a["continuum"] == prim), None)
    if ap is not None:
        out["razones"] = razones_caso_b(filas, canon, prim, meta["av"])
    if ap is not None and F_ha_dir is not None and ap["A_Pa9"] > 0:
        A = ap["A_Pa9"] * meta["unit"]
        pred = pa9_caso_b(F_ha_dir, meta["av"])
        out["paschen_av"] = {"variante": f"{canon}, {meta['primaria']}", "A_cgs": A, "pa9_caso_b_cgs": pred,
                             "veces_caso_b": A / pred, "av": av_para_paschen(F_ha_dir, A)[0]}
    m = max(fuga, key=lambda n: abs(fuga[n]["fuga_sobre_sigma"]))
    out["fuga"] = {"fraccion": meta["fraccion_primaria"], "linea_max": m, "fuga_cgs": fuga[m]["fuga_cgs"],
                   "max_sobre_sigma": fuga[m]["fuga_sobre_sigma"]}
    con_g2 = [n for n in fuga if "fuga_sobre_sigma_g2" in fuga[n]]
    if con_g2:
        m2 = max(con_g2, key=lambda n: abs(fuga[n]["fuga_sobre_sigma_g2"]))
        out["fuga"].update({"linea_max_g2": m2, "max_sobre_sigma_g2": fuga[m2]["fuga_sobre_sigma_g2"]})
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("slug", help="objeto del paper (targets/<slug>.json, clave `paper`)")
    ap.add_argument("--alternativas", nargs="*", default=[],
                    help="otras plantillas (`object`) de la biblioteca de g2_continuum_template")
    ap.add_argument("--fraccion-primaria", type=float, default=0.06,
                    help="fraccion del continuo del compañero que fuera luz de la primaria (paper, Sect. 4.1: 6 %%)")
    ap.add_argument("--sin-excluir-teluricas", action="store_true",
                    help="no quitar las bandas teluricas fuertes del ajuste de plantilla (la variante del agente)")
    ap.add_argument("--project-root", default=None,
                    help="raiz que contiene runs/ (desde un worktree, el clon que los tiene)")
    ap.add_argument("--json", default=None, help="escribe aqui el resumen (como salidas/resumen.json del agente)")
    ap.add_argument("--csv", default=None, help="escribe aqui una fila por medida (como salidas/lineas_<obj>.csv)")
    args = ap.parse_args(argv)

    import scripts.paper_figures as pf
    if args.project_root:
        pf.PROYECTO = Path(args.project_root).resolve()
    obj = objeto_del_paper(args.slug)
    filas, todo = analiza(obj, pf.PROYECTO, args.alternativas, fraccion_primaria=args.fraccion_primaria,
                          excluir_teluricas=not args.sin_excluir_teluricas)
    _imprime(obj.nombre, obj.run_id, todo["meta"], todo["resumen"], todo["paschen_stack"], todo["cifras"])
    if args.json:
        Path(args.json).write_text(json.dumps(todo, indent=1, default=float))
    if args.csv:
        keys = sorted({k for f in filas for k in f if not k.startswith("_")})
        with open(args.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
            w.writeheader()
            for f in filas:
                w.writerow(f)
    return 0


def objeto_del_paper(slug):
    """El `scripts.paper_figures.Objeto` de `slug` (con el `PROYECTO` que tenga ese modulo)."""
    import scripts.paper_figures as pf
    obj = next((o for o in pf.objetos_del_paper() if o.slug == slug), None)
    if obj is None:
        raise SystemExit(f"{slug!r} no es un objeto del paper (targets/*.json con clave `paper`)")
    return obj


def analiza(obj, proyecto, alternativas, *, fraccion_primaria=0.06, excluir_teluricas=True):
    """Todo lo de un objeto: `(filas, resumen)`, con el resumen como el `resumen.json` del agente
    (`meta`, `estrella`, `resumen`, `paschen_stack`, `cotas_AV`) mas `fuga_primaria` y `cifras`."""
    filas, estrella, meta = correr(obj, proyecto, alternativas, excluir_teluricas=excluir_teluricas)
    meta["fraccion_primaria"] = float(fraccion_primaria)
    unit = meta["unit"]
    res = resumen(filas, unit, meta["primaria"], canonico=meta["metodos"][0])
    apil = []
    for met in meta["metodos"]:
        for kind in meta["continuos_plantilla"]:   # la del run, las alternativas, la mezcla
            for T_K, ne in BARRIDO_CASO_B:
                a = apilar_paschen(filas, met, kind, meta["av"], T_K, ne)
                if a is not None:
                    apil.append(a)
    g2 = {r["name"]: r for r in obj.filas("g2_line_measurements.csv")}
    sigma_g2 = {n: float(r["flux_mf_sigma_emp"]) * unit for n, r in g2.items()
                if r.get("flux_mf_sigma_emp") not in (None, "") and math.isfinite(float(r["flux_mf_sigma_emp"]))}
    fuga = fuga_primaria(res, estrella, meta["fraccion_primaria"], sigma_g2)
    ha = g2.get("Halpha", {})
    F_ha_dir = F_ha_mf = None
    cotas = {}
    if ha.get("status") == "detected":
        F_ha_dir, F_ha_mf = float(ha["flux_direct"]) * unit, float(ha["flux_mf"]) * unit
        cotas = cotas_av(res, apil, F_ha_dir, F_ha_mf, unit, meta["primaria"], canonico=meta["metodos"][0])
    cif = cifras(filas, res, apil, fuga, meta, F_ha_dir)
    return filas, {"meta": meta, "estrella": estrella, "resumen": res, "paschen_stack": apil, "cotas_AV": cotas,
                   "fuga_primaria": fuga, "cifras": cif,
                   "halpha_g2": {"flux_direct_cgs": F_ha_dir, "flux_mf_cgs": F_ha_mf, "status": ha.get("status")}}


if __name__ == "__main__":
    raise SystemExit(main())
