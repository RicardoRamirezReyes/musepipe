#!/usr/bin/env python
"""Paβ y Brγ en los espectros J/K publicados de los compañeros: medida, límites y L_acc (apéndice del paper).

Punto 1 de `docs/2026-10-21_lo_que_pediria_un_referi.md`. Lo midió la sesión del 2026-10-21 con scripts de una
carpeta de sesión (`~/muse_sesion_20261021/pab_brg/`: `comun.py`, `desplazamiento.py`, `medir.py`, `lacc.py`;
informe en su `INFORME.md`). Aquí está el mismo cálculo, con las entradas sacadas de los productos y de
`targets/` en vez de literales. Hermano de `otras_lineas_lacc.py` (las líneas del rango MUSE), con la misma
regla: lee lo que las etapas escribieron y no inventa entradas. Solo lectura sobre `runs/` y `/mnt/2TB`.

**Espectros.** Los del catálogo CDS J/A+A/708/A312 (Hayoz et al. 2026), en F_λ (W m⁻² µm⁻¹), remuestreados a
λ/Δλ ≈ 500 conservando flujo y sin desenrojecer; se leen por ruta (`--cds`, por defecto donde los leyó la
sesión). Cuáles y con qué R nativa y fecha lo declara el perfil del objeto (`targets/<slug>.json` →
`ir_spectra.spectra` y `ir_spectra.hydrogen_lines`). Un espectro mide una línea si la ventana y sus bandas
laterales caben en él; `hydrogen_lines.also_checked` lista los que solo se miran para decir qué cubren (ERIS).
Escala de flujo: `hydrogen_lines.flux_rescale_mag[banda]` o, en K, `colour_excess.k_rescale_mag` (la de
`ajuste_conjunto.py`); sin ellas, 1. Las EW no dependen de la escala; los flujos sí.

**Eje λ.** El catálogo está en vacío. Se mide su desplazamiento frente a BT-Settl (`hydrogen_lines.btsettl_reference`,
degradado a la R nativa y rebinado en caja a los píxeles del dato) por χ² con un polinomio multiplicativo de grado
3 en un tramo de la banda, en pasos de 0.025 píxeles en ±1 píxel; la línea se centra en λ_vacío + ese desplazamiento.

**Estimadores** (los cuatro de la sesión; EW < 0 es emisión, como en el paper; flujo > 0 es emisión):

- `perfil` (ADOPTADO, declarado antes de mirar la línea con él): filtro adaptado, f = recta + A·P/Δλ por mínimos
  cuadrados sin pesos en una ventana de 3 píxeles más 4 de banda lateral a cada lado. P es el perfil de una
  línea no resuelta: gaussiana a la R nativa ⊗ caja de 1 píxel, promediada sobre ±0.25 píxeles de posición.
- `directo`: integración en los 3 píxeles con el continuo lineal de las bandas laterales.
- `perfil_cociente` y `directo_cociente`: los mismos sobre dato / BT-Settl.

**Ruido empírico.** El MISMO estimador centrado en cada píxel sin línea de la banda útil (J 1.165–1.330 µm,
K 2.035–2.2925 µm), con la misma fase subpíxel que la línea, sin rasgos atómicos fuertes, telúrico fuerte, la
cabeza de CO ni otras líneas de H en su ventana y sin tocar la ventana de la línea con sus bandas laterales.
σ = max(std, 1.4826·MAD). Se dan N y cuántos tienen ventanas disjuntas.

**Inyección.** Una línea no resuelta inyectada en 11 fases en ±0.25 píxeles de λ0 y recuperada con el perfil
nominal (η); también con un suavizado extra a R = 500 (la resolución efectiva que el ajuste de K no excluye).

**Fotosfera.** El mismo estimador sobre BT-Settl (`hydrogen_lines.btsettl_photosphere`) y sobre las plantillas
Clase III M6.5–M9.5 de Manara et al. (2013, 2017) degradadas igual: la absorción que una emisión tendría que
rellenar. El límite le suma `hydrogen_lines.photospheric_absorption_A` (el valor que declaró la sesión).

**Límite.** F_lim = [max(F, 0) + k σ_F + a_fot·cont] / η, con k = 3 (límite) y 5 (umbral de detección, sin
max(F, 0)); EW_lim = F_lim / cont. Con la η del suavizado a R = 500, `F_lim3_suavizado500`.

**L_acc** (`musepipe.models.accretion.line_lacc`, la cadena de G3): CCM89 con la R_V del run, 4πd², y las
relaciones de Alcalá et al. (2017, Tabla B.1) de Paβ y Brγ (`ALCALA17_IR`, abajo, con su fila) y de Hα (la de
`otras_lineas_lacc.ALCALA17`, comprobada contra la que declara el run). Entradas, de los productos:

- el flujo de Hα: en la detección, la integración directa de G2 (`flux_direct`, la que usa G3 para L_acc); en el
  límite, el `f_lim_observed` al 99 % de E3 en el método canónico (`paper_figures._entrada_extincion`, la misma
  regla que la figura de la extinción y `disco_cotas.py`);
- A_V adoptada, distancia y ley de extinción del config (`h03_av`, `h03_distance_pc`, `h03_extinction_law`,
  `h03_rv_extinction`); L_bol de `targets/<slug>.json` → `fisica.log_lbol`;
- las A_V de la lectura alternativa: la adoptada, la fija intermedia `AV_INTERMEDIA` (3.0 mag, la de las cadenas
  `avfija3` y la Tabla B.5) y, con `--cadenas`, la mediana de la cadena BT-Settl con el prior de la primaria
  (`<slug>_libre_avprimaria_s1`), redondeada a 0.1 mag como la imprime la Tabla B.5.

Con la L_acc de Hα a cada A_V, la relación de cada línea predice su flujo observado (la inversa); se da la EW
predicha, cuántas veces queda el umbral a 5σ por encima (también subiendo la predicción 2σ con la dispersión de
las dos relaciones, √(σ_Hα² + σ_línea²)/a_línea) y la A_V a la que el umbral la alcanzaría (el flujo predicho
crece como 10^[(0.4 k_Hα a_Hα / a_línea − 0.4 k_línea) A_V]).

**Exceso.** Si un espectro da z ≥ `Z_EXCESO`, se dice qué implicaría si fuera una línea (L_acc, Hα frente a la
medida de MUSE, la otra línea del mismo objeto y el cociente Paβ/Brγ) y dónde cae dentro de la ventana
(centroide del exceso sobre el continuo ajustado).

Reproducir la sesión (las entradas de entonces: F(Hα) de la integración directa de entonces y el límite de E3 de
entonces, sin la A_V de la cadena, y las A_V de su hipótesis del exceso):

    python scripts/pab_brg.py --project-root RAIZ_CON_RUNS \\
        --f-halpha ROXs12b=2.54e-17 --f-halpha ROXs42Bb=5.13e-18 \\
        --av ROXs12b=1.8,3.0,3.7 --av ROXs42Bb=1.9,3.0 --av-hipotesis 1.9,3.0,5.5 --json /tmp/sesion.json

Las cifras de hoy:

    python scripts/pab_brg.py --project-root RAIZ_CON_RUNS --cadenas ~/muse_noche_20261024_c4/cadenas \\
        [--json SALIDA] [--tabla SALIDA.tex]

`tests/test_pab_brg.py` fija las piezas (filtro, conversión) y, con los datos, las cifras de la sesión.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from musepipe.models.accretion import L_SUN_ERG_S, line_lacc  # noqa: E402
from musepipe.stages.stage_h03_limits import luminosity_erg_s  # noqa: E402

#: el catálogo de Hayoz et al. (2026), donde lo leyó la sesión
DEFAULT_CDS = Path("/mnt/2TB/IR_Data/publicados_Hayoz2026_JAA708A312")
#: BT-Settl CIFIST (npz de `musepipe.models.cache`), la rejilla de `teff_bayes.py`
DEFAULT_BTSETTL = Path("/mnt/2TB/IR_Data/btsettl_cifist/npz")
#: brazo NIR de las plantillas Clase III de Manara et al. (2013, 2017), catálogos CDS
DEFAULT_MANARA = Path("/mnt/2TB/IR_Data/plantillas_xshooter_nir/manara_cds")

W_M2_UM_A_CGS = 0.1      # 1 W m⁻² µm⁻¹ = 0.1 erg s⁻¹ cm⁻² Å⁻¹

#: las dos líneas, en VACÍO (µm; en aire: Paβ 1.281808, Brγ 2.165530, 3.5 y 5.9 Å menos: 0.14 píxeles). Banda
#: útil de los controles y tramo en el que se mide el eje λ frente a BT-Settl.
LINEAS = {
    "PaB": dict(nombre="Paβ", vacio_um=1.2821578, banda_controles_um=(1.165, 1.330), tramo_lambda_um=(1.170, 1.330)),
    "BrG": dict(nombre="Brγ", vacio_um=2.1661178, banda_controles_um=(2.035, 2.2925), tramo_lambda_um=(2.080, 2.290)),
}

#: la tabla de la que salen las relaciones (la de `otras_lineas_lacc.CITA_ALCALA17`)
CITA_ALCALA17 = ("Alcala et al. 2017, A&A 600, A20, Table B.1 "
                 "(doi:10.1051/0004-6361/201629929; arXiv:1612.07054v1, p. 21)")
#: Alcalá et al. 2017, Tabla B.1 («Revisited Lacc – Lline linear fits»): log(L_acc/L☉) = a log(L_línea/L☉) + b;
#: `sigma` = σ⋆ de la tabla (dex); `recomendada` = la marca • («Suggested relations for deriving Lacc from the
#: line luminosity»), que Hα no lleva. Copiadas de arXiv:1612.07054v1 (la sesión del 2026-10-21 y otra vez el
#: 2026-10-28); `fila` es la etiqueta y la λ (nm) de la tabla.
ALCALA17_IR = {
    "PaB": dict(a=1.06, a_err=0.07, b=2.76, b_err=0.34, sigma=0.45, recomendada=True, fila="Pa5 (Paβ) 1281.8070"),
    "BrG": dict(a=1.19, a_err=0.10, b=4.02, b_err=0.51, sigma=0.45, recomendada=True, fila="Br7 (Brγ) 2166.1210"),
}

#: rasgos que no pueden caer en la ventana de un control (µm, vacío): atómicos fuertes de las M tardías
#: (Na I, K I, Al I; Na I, Ca I, Mg I) y otras líneas de H (Paγ, Brδ); bandas telúricas fuertes; cabeza de CO 2-0
RASGOS_UM = [1.1381, 1.1404, 1.1690, 1.1773, 1.2432, 1.2522, 1.3127, 1.3154,
             2.2062, 2.2090, 2.2614, 2.2631, 2.2657, 2.2814]
OTRAS_H_UM = [1.0941, 1.9451]
TELURICO_UM = [(1.10, 1.165), (1.330, 1.50), (1.80, 2.035)]
CO_UM = 2.2925

NW, NS = 3, 4            # ventana y banda lateral (píxeles)
DFASE = 0.25             # incertidumbre de la posición de la línea (píxeles): eje λ, aire/vacío, VR
N_FASES_INYECCION = 11
AMPLITUD_INYECCION = 1e-3   # flujo inyectado / (mediana del espectro × píxel): lineal, no importa el valor
R_SUAVIZADO = 500.0      # la hipótesis de un suavizado extra (resolución efectiva) que el ajuste de K no excluye
ESTIMADORES = ("perfil", "directo", "perfil_cociente", "directo_cociente")
ADOPTADO = "perfil"
VARIANTES_VENTANA = ((2, 4), (3, 3), (3, 6), (5, 4))
N_DESPLAZAMIENTOS = 81   # pasos en ±1 píxel
GRADO_DESPLAZAMIENTO = 3
K_LIMITE, K_UMBRAL = 3.0, 5.0
#: la A_V fija intermedia de las cadenas de T_eff (`avfija3`) y de la Tabla B.5
AV_INTERMEDIA = 3.0
#: z a partir del cual un espectro se trata como «exceso» y se dice qué implicaría si fuera una línea
Z_EXCESO = 3.0
#: Hα en reposo (aire), la de E1 y G2
HALPHA_A = 6562.8

#: cifras de la literatura que el párrafo compara con las de aquí, con su fuente (verificadas el 2026-10-28 sobre
#: el texto de arXiv; las entradas BibTeX están en `paper/refs.bib` o, la de Fuhrmeister, en la carpeta del
#: apéndice). No entran en ninguna cuenta salvo los cocientes que se imprimen.
LITERATURA = {
    "twa27b_L_pab_lsun": dict(valor=3.54e-9, error=0.79e-9, d_pc=65.4,
                              fuente="Marleau et al. 2024 (ApJ 964, 70; arXiv:2401.04763v4), Tabla 1, fila «Pa β1» "
                                     "(JWST/NIRSpec G140H, 4.5σ); d = 65.4 pc; sin corregir de extinción (Sect. 2.2)"),
    "fulguraciones_dpew_pab_A": dict(valor=(0.009, 0.278), n=35, n_estrellas=20,
                                     fuente="Fuhrmeister et al. 2023 (A&A 678, A1; arXiv:2308.07685v1), Tabla C.1: "
                                            "ΔpEW(Paβ) de las 35 fulguraciones detectadas automáticamente en 20 enanas "
                                            "M (subestimado, Fig. 7); en reposo, Paβ en absorción (Sect. 3.1, 4.1)"),
    "fulguraciones_ciclo_pab": dict(valor=0.0019,
                                    fuente="Fuhrmeister et al. 2023, Sect. 4.2.2: ciclo de trabajo de las fulguraciones "
                                           "en Paβ detectadas automáticamente, 0.19 % (Hα: 2.26 %)"),
}


def _trapz(y, x):
    return np.trapz(y, x) if hasattr(np, "trapz") else np.trapezoid(y, x)


# --------------------------------------------------------------------------- remuestreo y perfiles

def rebin_caja(wm, fm, w):
    """Promedio de flujo en cada píxel de `w` (bordes en los puntos medios): el remuestreo que conserva flujo."""
    bordes = np.r_[w[0] - 0.5 * (w[1] - w[0]), 0.5 * (w[1:] + w[:-1]), w[-1] + 0.5 * (w[-1] - w[-2])]
    cf = np.r_[0.0, np.cumsum(0.5 * (fm[1:] + fm[:-1]) * np.diff(wm))]
    acum = np.interp(bordes, wm, cf)
    return np.diff(acum) / np.diff(bordes)


def convoluciona_r(wm, fm, R):
    """Convoluciona a resolución R constante (FWHM = λ/R): promedio en caja sobre una rejilla log-uniforme de paso
    ≤ σ/5 (o el del modelo si es más grueso) y convolución por FFT con la gaussiana."""
    from scipy.signal import fftconvolve
    lw = np.log(wm)
    sig_log = (1.0 / R) / 2.3548
    dl = max(float(np.median(np.diff(lw))), 1e-7)
    dl = max(dl, sig_log / 5.0)
    lg = np.arange(lw[0] + dl, lw[-1] - dl, dl)
    wg = np.exp(lg)
    fg = rebin_caja(wm, fm, wg)
    sig = sig_log / dl
    n = int(np.ceil(5 * sig))
    k = np.exp(-0.5 * (np.arange(-n, n + 1) / sig) ** 2)
    k /= k.sum()
    return wg, fftconvolve(fg, k, mode="same")


def degrada(wm, fm, w, R):
    """Modelo o plantilla → resolución nativa R (gaussiana) → píxeles del dato (caja)."""
    sel = (wm > w[0] - 2000) & (wm < w[-1] + 2000)
    wg, fg = convoluciona_r(wm[sel], fm[sel], R)
    return rebin_caja(wg, fg, w)


def perfil_linea(w, lam_c, R, extra500=False, dfase=0.0):
    """Fracción del flujo de una línea no resuelta centrada en `lam_c` que cae en cada píxel (gaussiana a la R
    nativa ⊗ caja de 1 píxel; con `extra500`, además un suavizado a R = 500). Con `dfase` > 0, el promedio sobre
    centros uniformes en ±dfase píxeles. Suma 1."""
    dw = np.median(np.diff(w))
    if dfase > 0:
        ps = [perfil_linea(w, lam_c + d * dw, R, extra500) for d in np.linspace(-dfase, dfase, 9)]
        p = np.mean(ps, axis=0)
        return p / p.sum()
    wf = np.linspace(lam_c - 6 * dw, lam_c + 6 * dw, 6001)
    sig = lam_c / R / 2.3548
    if extra500:
        sig = np.hypot(sig, lam_c / R_SUAVIZADO / 2.3548)
    g = np.exp(-0.5 * ((wf - lam_c) / sig) ** 2)
    g /= _trapz(g, wf)
    wext = np.r_[w[0] - 20 * dw, w, w[-1] + 20 * dw]
    wfull = np.r_[wext[0], wf, wext[-1]]
    gfull = np.r_[0.0, g, 0.0]
    p = rebin_caja(wfull, gfull, w) * np.gradient(w)
    return p / p.sum()


# --------------------------------------------------------------------------- estimadores

def mascara(w):
    """True en los píxeles que no pueden ser ventana de control (rasgos, otras H, telúrico, CO)."""
    dw = np.median(np.diff(w))
    m = np.zeros(w.size, bool)
    for r in RASGOS_UM + OTRAS_H_UM:
        m |= np.abs(w - r * 1e4) < 0.75 * dw
    for lo, hi in TELURICO_UM:
        m |= (w > lo * 1e4) & (w < hi * 1e4)
    m |= w > CO_UM * 1e4
    return m


def mide_directo(w, f, c, nw=NW, ns=NS, gap=0, malos=None):
    """EW (Å, < 0 = emisión) y flujo (> 0 = emisión) por integración en `nw` píxeles centrados en `c`.

    Continuo: recta por mínimos cuadrados sin pesos en `ns` píxeles a cada lado (sin los de `malos`). None si
    falta continuo (< 3 píxeles por lado) o la ventana se sale del espectro.
    """
    h = nw // 2
    lo, hi = c - h, c - h + nw
    izq = np.arange(lo - gap - ns, lo - gap)
    der = np.arange(hi + gap, hi + gap + ns)
    if izq[0] < 0 or der[-1] >= w.size or lo < 0 or hi > w.size:
        return None
    if malos is not None:
        izq, der = izq[~malos[izq]], der[~malos[der]]
    if izq.size < 3 or der.size < 3:
        return None
    lados = np.r_[izq, der]
    x0 = w[c]
    p = np.polyfit(w[lados] - x0, f[lados], 1)
    ventana = np.arange(lo, hi)
    dw = np.gradient(w)[ventana]
    cont = np.polyval(p, w[ventana] - x0)
    return dict(ew=float(-np.sum((f[ventana] / cont - 1.0) * dw)), flujo=float(np.sum((f[ventana] - cont) * dw)),
                cont=float(np.polyval(p, 0.0)), ventana=(int(lo), int(hi)), p=[float(x) for x in p])


def mide_perfil(w, f, c, P, ns=NS, malos=None, nw=NW):
    """Filtro adaptado: f = recta + A·P/Δλ en ventana + bandas laterales (mínimos cuadrados sin pesos). `P` es el
    perfil (fracción por píxel) ya centrado en la posición supuesta. Devuelve el flujo A y EW = −A/continuo."""
    h = nw // 2
    lo, hi = c - h, c - h + nw
    izq = np.arange(lo - ns, lo)
    der = np.arange(hi, hi + ns)
    if izq[0] < 0 or der[-1] >= w.size:
        return None
    if malos is not None:
        izq, der = izq[~malos[izq]], der[~malos[der]]
    if izq.size < 3 or der.size < 3:
        return None
    idx = np.r_[izq, np.arange(lo, hi), der]
    dw = np.gradient(w)
    x = w[idx] - w[c]
    A = np.vstack([np.ones_like(x), x, P[idx] / dw[idx]]).T
    co, *_ = np.linalg.lstsq(A, f[idx], rcond=None)
    return dict(flujo=float(co[2]), cont=float(co[0]), pendiente=float(co[1]), ew=float(-co[2] / co[0]))


def estimador(nombre, w, f, c, lam_c, R, malos, extra500=False):
    """El estimador `nombre` centrado en el píxel `c`, con la línea supuesta en `lam_c`."""
    if nombre.startswith("perfil"):
        P = perfil_linea(w, lam_c, R, extra500=extra500, dfase=DFASE)
        return mide_perfil(w, f, c, P, NS, malos=malos, nw=NW)
    return mide_directo(w, f, c, NW, NS, malos=malos)


def sigma_robusta(x):
    x = np.asarray(x, float)
    med = np.median(x)
    return dict(std=float(np.std(x, ddof=1)), mad=float(1.4826 * np.median(np.abs(x - med))),
                media=float(np.mean(x)), mediana=float(med), n=int(x.size))


def n_disjuntos(centros, nw=NW):
    """Cuántos controles tienen ventanas disjuntas (voraz, de azul a rojo)."""
    n, ult = 0, -10 ** 9
    for c in sorted(centros):
        if c - ult >= nw:
            n += 1
            ult = c
    return n


def corre(nombre, w, f, lam0, c0, R, malos, banda_A, extra500=False):
    """El estimador en la línea y en todos sus controles (misma fase subpíxel)."""
    fase = lam0 - w[c0]
    lin = estimador(nombre, w, f, c0, lam0, R, malos, extra500)
    h = NW // 2
    zona = set(range(c0 - h - NS, c0 - h + NW + NS))
    lin_v = set(range(c0 - h, c0 - h + NW))
    ctl = []
    for c in range(w.size):
        if not (banda_A[0] <= w[c] <= banda_A[1]):
            continue
        lo, hi = c - h, c - h + NW
        if set(range(lo - NS, hi + NS)) & lin_v or set(range(lo, hi)) & zona:
            continue
        if lo < 0 or hi > w.size or malos[lo:hi].any():
            continue
        r = estimador(nombre, w, f, c, w[c] + fase, R, malos, extra500)
        if r is None:
            continue
        r["c"], r["lam_A"] = c, float(w[c])
        ctl.append(r)
    return lin, ctl


def resume(lin, ctl):
    """Medida, σ empírica de los controles y z (flujo / σ)."""
    se = sigma_robusta([x["ew"] for x in ctl])
    sf = sigma_robusta([x["flujo"] for x in ctl])
    sig_e, sig_f = max(se["std"], se["mad"]), max(sf["std"], sf["mad"])
    return dict(ew=lin["ew"], flujo=lin["flujo"], cont=lin["cont"], sigma_ew=sig_e, sigma_flujo=sig_f,
                z=lin["flujo"] / sig_f, n=se["n"], n_disjuntos=n_disjuntos([x["c"] for x in ctl]),
                media_controles_ew=se["media"], std_ew=se["std"], mad_ew=se["mad"],
                rango_controles_um=[min(x["lam_A"] for x in ctl) / 1e4, max(x["lam_A"] for x in ctl) / 1e4],
                fraccion_controles_mas_emision=float(np.mean([x["flujo"] >= lin["flujo"] for x in ctl])))


def inyeccion(nombre, w, f, lam0, c0, R, malos, cociente=None, extra500=False):
    """Fracción recuperada de una línea no resuelta inyectada (en el dato, antes del cociente) en
    `N_FASES_INYECCION` fases en ±DFASE alrededor de λ0; el estimador supone siempre el perfil nominal."""
    dw = np.median(np.diff(w))
    base = estimador(nombre, w, f if cociente is None else f / cociente, c0, lam0, R, malos)["flujo"]
    fr = []
    for ph in np.linspace(-DFASE, DFASE, N_FASES_INYECCION):
        P = perfil_linea(w, lam0 + ph * dw, R, extra500=extra500)
        amp = AMPLITUD_INYECCION * np.median(f) * dw
        g = f + amp * P / np.gradient(w)
        if cociente is not None:
            g = g / cociente
        r = estimador(nombre, w, g, c0, lam0, R, malos)["flujo"]
        fr.append((r - base) / (amp if cociente is None else amp / np.interp(lam0, w, cociente)))
    return dict(media=float(np.mean(fr)), min=float(np.min(fr)), max=float(np.max(fr)))


def controles_directo(w, f, lam0, nw, ns, banda_A):
    """El estimador directo con ventana `nw` y banda lateral `ns` en todos los centros sin línea de la banda útil
    (para las variantes de ventana: cada variante con sus propios controles)."""
    malos = mascara(w)
    c0 = int(np.argmin(np.abs(w - lam0)))
    h = nw // 2
    lin = set(range(c0 - h, c0 - h + nw))
    rango_A = float(max(abs(banda_A[0] - lam0), abs(banda_A[1] - lam0)))
    out = []
    for c in range(w.size):
        if abs(w[c] - lam0) > rango_A:
            continue
        lo, hi = c - h, c - h + nw
        if set(range(lo - ns, hi + ns)) & lin:
            continue
        if lo < 0 or hi > w.size or malos[lo:hi].any():
            continue
        r = mide_directo(w, f, c, nw, ns, malos=malos)
        if r is None:
            continue
        r["c"], r["lam_A"] = c, float(w[c])
        out.append(r)
    return [x for x in out if banda_A[0] <= x["lam_A"] <= banda_A[1]]


# --------------------------------------------------------------------------- eje λ

def desplazamiento_lambda(w, f, s, wm, fm, R, tramo_A):
    """Desplazamiento (Å) que hay que aplicar al modelo (vacío) para que cuadre con el eje λ del dato: χ² con un
    polinomio multiplicativo de grado `GRADO_DESPLAZAMIENTO`, pesos 1/σ formal; 1σ con χ² reescalado."""
    lo, hi = tramo_A
    sel = (w > lo) & (w < hi)
    dw = np.median(np.diff(w))
    x = (w[sel] - w[sel].mean()) / np.ptp(w[sel])
    ds = np.linspace(-1.0, 1.0, N_DESPLAZAMIENTOS) * dw
    c2 = []
    for d in ds:
        m = degrada(wm + d, fm, w, R)[sel]
        A = np.vstack([m * x ** k for k in range(GRADO_DESPLAZAMIENTO + 1)]).T
        co, *_ = np.linalg.lstsq(A / s[sel, None], f[sel] / s[sel], rcond=None)
        c2.append(np.sum(((f[sel] - A @ co) / s[sel]) ** 2))
    c2 = np.array(c2)
    i = int(np.argmin(c2))
    nu = int(sel.sum()) - (GRADO_DESPLAZAMIENTO + 1)
    c2n = c2 / (c2[i] / nu)
    ok = c2n <= c2n[i] + 1
    return dict(delta_A=float(ds[i]), delta_px=float(ds[i] / dw),
                int_px=[float(ds[ok].min() / dw), float(ds[ok].max() / dw)], chi2nu=float(c2[i] / nu),
                tramo_um=[lo / 1e4, hi / 1e4], n=int(sel.sum()))


# --------------------------------------------------------------------------- entradas: espectros y modelos

def perfil_objeto(slug, root=ROOT):
    from musepipe.targets import load_target
    p = load_target(slug, project_root=root)
    if not p or "ir_spectra" not in p:
        raise SystemExit(f"{slug}: sin perfil o sin 'ir_spectra' en targets/")
    if "hydrogen_lines" not in p["ir_spectra"]:
        raise SystemExit(f"targets/{slug}.json no declara ir_spectra.hydrogen_lines (R nativa, modelos, absorción)")
    return p


def factor_escala(ir: dict, banda: str) -> tuple[float, str]:
    """Factor multiplicativo sobre el flujo del catálogo y de dónde sale."""
    hl = ir.get("hydrogen_lines") or {}
    fr = (hl.get("flux_rescale_mag") or {}).get(banda)
    if fr:
        return 10 ** (-0.4 * (fr["adopted"] - fr["catalogue"])), f"hydrogen_lines.flux_rescale_mag.{banda}"
    ce = (ir.get("colour_excess") or {}).get("k_rescale_mag")
    if banda == "K" and ce:
        return 10 ** (-0.4 * (ce["adopted"] - ce["catalogue"])), "colour_excess.k_rescale_mag"
    return 1.0, "sin reescalar (catálogo)"


def carga(cds, fichero, factor=1.0):
    """λ (Å), F_λ y su error (erg s⁻¹ cm⁻² Å⁻¹) de un espectro del catálogo, por el factor de escala."""
    from scripts import tipo_ir_indices as tir
    w_um, f, e = tir.load(cds, fichero)
    return w_um * 1e4, f * W_M2_UM_A_CGS * factor, e * W_M2_UM_A_CGS * factor


def lineas_cubiertas(w):
    """Las líneas cuya ventana y bandas laterales caben en el espectro."""
    out = []
    for lin, L in LINEAS.items():
        lam0 = L["vacio_um"] * 1e4
        if not (w[0] < lam0 < w[-1]):
            continue
        c0 = int(np.argmin(np.abs(w - lam0)))
        h = NW // 2
        if c0 - h - NS >= 0 and c0 - h + NW + NS <= w.size:
            out.append(lin)
    return out


def cobertura(slug, perfil, cds):
    """Qué cubre cada espectro del objeto (los de `spectra` y los de `hydrogen_lines.also_checked`)."""
    ir = perfil["ir_spectra"]
    todos = dict(ir["spectra"])
    todos.update((ir["hydrogen_lines"].get("also_checked") or {}))
    out = {}
    for nombre, spec in todos.items():
        p = Path(cds) / spec["file"]
        if not p.exists():
            out[nombre] = dict(fichero=spec["file"], falta=True)
            continue
        w, _, _ = carga(cds, spec["file"])
        out[nombre] = dict(fichero=spec["file"], instrumento=spec.get("instrument"), ref=spec.get("ref"),
                           lam_min_um=float(w[0] / 1e4), lam_max_um=float(w[-1] / 1e4), lineas=lineas_cubiertas(w),
                           medido=nombre in ir["spectra"])
    return out


def btsettl(dir_bt, teff, logg):
    from musepipe.models.cache import load_spectrum_npz
    t = load_spectrum_npz(Path(dir_bt) / f"teff{int(teff)}_logg{float(logg):.1f}.npz")
    return np.asarray(t.wave_A, float), np.asarray(t.flux, float)


def plantillas_clase3(dir_manara, tipo_min=6.5):
    """Plantillas Clase III jóvenes M6.5–M9.5 de Manara et al. (2013, 2017), brazo NIR (λ en Å)."""
    from scripts.tipo_ir_plantillas import _fit_1d
    tipos = {}
    for sub in sorted(Path(dir_manara).iterdir()):
        sd = sub / "spectra.dat"
        if not sd.exists():
            continue
        for ln in sd.read_text().splitlines():
            t = ln.split()
            for i, tok in enumerate(t):
                if tok.endswith("N.fit"):
                    spt = t[i - 3]
                    if spt.startswith("M") and float(spt[1:]) >= tipo_min:
                        tipos[t[0]] = (spt, sub / tok)
    out = {}
    for nombre, (spt, fn) in tipos.items():
        if fn.exists():
            w, f = _fit_1d(fn)
            out[nombre] = dict(spt=spt, w=w, f=f, fichero=str(fn))
    return dict(sorted(out.items(), key=lambda kv: float(kv[1]["spt"][1:])))


# --------------------------------------------------------------------------- la medida de un espectro

def mide_espectro(w, f, s, linea, R, modelo, *, fotosfera=None, plantillas=None, estimadores=ESTIMADORES,
                  variantes=True):
    """Todo lo de un espectro: eje λ, los estimadores con sus controles, inyección y fotosfera.

    `modelo` = (λ, F) de BT-Settl de referencia (eje λ y cocientes); `fotosfera` = {(T, log g): (λ, F)};
    `plantillas` = la salida de `plantillas_clase3`.
    """
    L = LINEAS[linea]
    wm, fm = modelo
    despl = desplazamiento_lambda(w, f, s, wm, fm, R, tuple(x * 1e4 for x in L["tramo_lambda_um"]))
    dlam = despl["delta_A"]
    lam0_vac = L["vacio_um"] * 1e4
    lam0 = lam0_vac + dlam
    c0 = int(np.argmin(np.abs(w - lam0)))
    malos = mascara(w)
    banda = tuple(x * 1e4 for x in L["banda_controles_um"])
    mod = degrada(wm + dlam, fm, w, R)
    mod = mod / np.median(mod) * np.median(f)
    dw_med = float(np.median(np.diff(w)))
    res = dict(linea=linea, r_nativa=R, desplazamiento=despl, lam0_vacio_A=lam0_vac, lam0_catalogo_A=lam0,
               pixel_A=dw_med, lam_pixel_central_A=float(w[c0]), fase_px=float((lam0 - w[c0]) / dw_med),
               estimadores={}, controles={})
    for nombre in estimadores:
        coc = nombre.endswith("cociente")
        g = f / mod * np.median(f) if coc else f
        lin, ctl = corre(nombre, w, g, lam0, c0, R, malos, banda)
        r = resume(lin, ctl)
        if coc:      # flujo en unidades del dato: EW × continuo del dato
            r["flujo"] = -r["ew"] * lin["cont"] * np.interp(lam0, w, mod) / np.median(f)
            r["sigma_flujo"] = r["sigma_ew"] * lin["cont"] * np.interp(lam0, w, mod) / np.median(f)
        else:
            r["ajuste"] = {k: lin[k] for k in ("cont", "pendiente", "p", "ventana") if k in lin}
        cociente = (mod / np.median(f)) if coc else None
        r["inyeccion"] = inyeccion(nombre, w, f, lam0, c0, R, malos, cociente=cociente)
        r["inyeccion_R500_extra"] = inyeccion(nombre, w, f, lam0, c0, R, malos, cociente=cociente, extra500=True)
        if fotosfera:
            fot = []
            for (teff, logg), (wq, fq) in fotosfera.items():
                m = degrada(wq + dlam, fq, w, R)
                gm = m / mod * np.median(m) if coc else m
                fot.append(dict(teff=teff, logg=logg, ew=estimador(nombre, w, gm, c0, lam0, R, malos)["ew"]))
            r["fotosfera_btsettl"] = fot
        if plantillas:
            pls = []
            for npl, t in plantillas.items():
                ok = np.isfinite(t["f"]) & (t["f"] > 0)
                m = degrada(t["w"][ok] + dlam, t["f"][ok], w, R)
                gm = m / mod * np.median(m) if coc else m
                pls.append(dict(plantilla=npl, spt=t["spt"], ew=estimador(nombre, w, gm, c0, lam0, R, malos)["ew"]))
            r["plantillas_claseIII"] = pls
        res["estimadores"][nombre] = r
        res["controles"][nombre] = [dict(lam_A=x["lam_A"], ew=x["ew"], flujo=x["flujo"]) for x in ctl]
    if variantes:
        var = {}
        for nw, ns in VARIANTES_VENTANA:
            cc = c0 if nw % 2 else (c0 + 1 if w[c0] <= lam0 else c0)   # 2 px: los dos que rodean λ0
            rr = mide_directo(w, f, cc, nw, ns, malos=malos)
            ct = controles_directo(w, f, lam0, nw, ns, banda)
            sg = sigma_robusta([x["ew"] for x in ct])
            sgm = max(sg["std"], sg["mad"])
            var[f"directo_nw{nw}_ns{ns}"] = dict(ew=rr["ew"], sigma_ew=sgm, z=-rr["ew"] / sgm, n=sg["n"])
        res["variantes_ventana"] = var
    lo = c0 - NW // 2
    res["sigma_flujo_formal_directo"] = float(np.sqrt(np.sum((s[lo:lo + NW] * np.gradient(w)[lo:lo + NW]) ** 2)))
    # los píxeles de la ventana y el continuo del filtro adaptado en ellos (para describir un exceso)
    aj = res["estimadores"].get(ADOPTADO, {})
    P0 = perfil_linea(w, lam0, R)
    pix, frac = [], []
    for i in range(c0 - NW // 2 - 1, c0 - NW // 2 + NW + 1):
        if 0 <= i < w.size:
            cont_i = (aj.get("cont", np.nan) + aj.get("ajuste", {}).get("pendiente", np.nan) * (w[i] - w[c0]))
            pix.append(dict(lam_A=float(w[i]), f=float(f[i]), e=float(s[i]), cont=float(cont_i),
                            en_ventana=bool(c0 - NW // 2 <= i < c0 - NW // 2 + NW)))
            frac.append(float(P0[i]))
    res["pixeles"] = pix
    res["perfil_en_lam0"] = frac
    return res


def limites(r, cont, a_fot, eta, eta500):
    """Límite a 3σ, umbral a 5σ (flujo, > 0 = emisión) y límite de EW (Å, < 0 = emisión)."""
    Fm, sF = r["flujo"], r["sigma_flujo"]
    lim3 = (max(Fm, 0.0) + K_LIMITE * sF + a_fot * cont) / eta
    thr5 = (K_UMBRAL * sF + a_fot * cont) / eta
    return dict(F_lim3=lim3, F_umbral5=thr5, EW_lim3=-lim3 / cont, EW_umbral5=-thr5 / cont,
                F_lim3_suavizado500=(max(Fm, 0.0) + K_LIMITE * sF + a_fot * cont) / eta500)


def centroide_exceso(res):
    """Dónde está el exceso: centroide (Å, y su distancia a λ0 en Å y km/s) de los píxeles de la ventana ±1 cuyo
    exceso sobre el continuo del filtro adaptado supera su propio error formal, y cómo repartiría su flujo una
    línea no resuelta en λ0 (`perfil_linea` sin promediar la fase) frente al exceso medido en cada píxel."""
    v = res["pixeles"]
    exc = np.array([p["f"] - p["cont"] for p in v])
    err = np.array([p["e"] for p in v])
    lam = np.array([p["lam_A"] for p in v])
    sel = exc > err
    if not sel.any():
        return None
    cen = float(np.sum(lam[sel] * exc[sel]) / exc[sel].sum())
    d = cen - res["lam0_catalogo_A"]
    return dict(centroide_A=cen, desde_lam0_A=d, desde_lam0_kms=d / res["lam0_catalogo_A"] * 299792.458,
                pixeles_usados_A=[float(x) for x in lam[sel]],
                exceso_por_pixel=[dict(lam_A=float(l), exceso=float(x), error=float(e), fraccion_esperada=float(fr))
                                  for l, x, e, fr in zip(lam, exc, err, res.get("perfil_en_lam0", [np.nan] * lam.size))])


# --------------------------------------------------------------------------- conversión a L_acc

def relacion(linea):
    """La relación de Alcalá et al. (2017) en la forma de `line_lacc` (Hα: la de `otras_lineas_lacc`)."""
    if linea == "Halpha":
        from scripts.otras_lineas_lacc import ALCALA17
        r = ALCALA17["Halpha"]
    else:
        r = ALCALA17_IR[linea]
    return {"a": float(r["a"]), "b": float(r["b"]), "scatter_dex": float(r["sigma"]),
            "citation": f"{CITA_ALCALA17}, {r['fila']}"}


def k_lambda(ext, rest_A):
    return float(np.atleast_1d(ext.a_lambda_over_av(float(rest_A)))[0])


def reposo_A(linea):
    return HALPHA_A if linea == "Halpha" else LINEAS[linea]["vacio_um"] * 1e4


def lacc_de_flujo(flujo_cgs, linea, *, av, distancia_pc, ext):
    """`{l_line_lsun, l_acc_lsun}` de un flujo OBSERVADO (erg s⁻¹ cm⁻²): la conversión de G3."""
    return line_lacc(float(flujo_cgs), distance_pc=float(distancia_pc), av=float(av),
                     a_lambda_over_av=k_lambda(ext, reposo_A(linea)), relation=relacion(linea), flux_unit_cgs=1.0)


def linea_de_lacc(lacc_lsun, linea):
    """L_línea (L☉) que la relación de `linea` asigna a una L_acc (la inversa)."""
    r = relacion(linea)
    return 10.0 ** ((math.log10(float(lacc_lsun)) - r["b"]) / r["a"])


def flujo_de_linea(l_linea_lsun, linea, *, av, distancia_pc, ext):
    """Flujo OBSERVADO (erg s⁻¹ cm⁻²) de una L_línea (L☉), enrojecido con la misma ley y A_V."""
    f = float(l_linea_lsun) * L_SUN_ERG_S / luminosity_erg_s(1.0, distancia_pc)
    return f * 10.0 ** (-0.4 * k_lambda(ext, reposo_A(linea)) * float(av))


def flujo_predicho(lacc_lsun, linea, *, av, distancia_pc, ext):
    return flujo_de_linea(linea_de_lacc(lacc_lsun, linea), linea, av=av, distancia_pc=distancia_pc, ext=ext)


def dispersion_prediccion(linea):
    """Dispersión (dex) de L_línea predicha desde Hα: las dos relaciones, en log L_línea."""
    return math.hypot(relacion("Halpha")["scatter_dex"], relacion(linea)["scatter_dex"]) / relacion(linea)["a"]


def pendiente_av(linea, ext):
    """d log10 F_predicho / d A_V con la Hα fija: 0.4 k_Hα a_Hα / a_línea − 0.4 k_línea."""
    return (0.4 * k_lambda(ext, HALPHA_A) * relacion("Halpha")["a"] / relacion(linea)["a"]
            - 0.4 * k_lambda(ext, reposo_A(linea)))


# --------------------------------------------------------------------------- un objeto entero

def av_con_prior(slug, cadenas):
    """Mediana de A_V de la cadena BT-Settl con el prior de la primaria (`<slug>_libre_avprimaria_s1`)."""
    if not cadenas:
        return None
    from scripts.teff_cifras_texto import Cadenas, etiqueta
    K = Cadenas(cadenas)
    e = etiqueta(slug)
    if e not in K.por_etiqueta:
        return None
    d = K.por_etiqueta[e]
    return dict(etiqueta=e, av_p50=float(d["posterior"]["av"]["p50"]), fichero=d["_json"],
                av_redondeada=round(float(d["posterior"]["av"]["p50"]), 1))


def entradas_objeto(obj, *, f_halpha=None, avs=None, cadenas=None):
    """Lo que sale de los productos (o de los argumentos, para reproducir la sesión)."""
    import scripts.paper_figures as pf
    from scripts.otras_lineas_lacc import _comprueba_relacion_halpha, extincion_del_run

    cfg = obj.config
    _comprueba_relacion_halpha(cfg)
    ext = extincion_del_run(cfg)
    e = pf._entrada_extincion(obj)
    phys = e["phys"]
    if not math.isclose(k_lambda(ext, HALPHA_A), float(phys["a_halpha_over_av"]), rel_tol=1e-9):
        raise SystemExit(f"{obj.slug}: A(Hα)/A_V de CCM89 ({k_lambda(ext, HALPHA_A)}) no es la del run "
                         f"({phys['a_halpha_over_av']})")
    perfil = perfil_objeto(obj.slug)
    fis = perfil.get("fisica") or {}
    prior = av_con_prior(obj.slug, cadenas)
    av0 = float(cfg["h03_av"])
    if avs is None:
        avs = [av0, AV_INTERMEDIA] + ([prior["av_redondeada"]] if prior else [])
        avs = sorted(set(avs))
    F = float(e["f_obs"]) if f_halpha is None else float(f_halpha)
    return dict(slug=obj.slug, nombre=obj.nombre, run=obj.run_id, cfg=cfg, ext=ext, perfil=perfil,
                f_halpha=F, f_halpha_fuente=("argumento" if f_halpha is not None else
                                             ("E3 f_lim_observed (99 %, método canónico)" if e["limite"]
                                              else "G2 flux_direct (integración directa, la de G3)")),
                limite=bool(e["limite"]), av_adoptada=av0, avs=[float(a) for a in avs], av_prior=prior,
                distancia_pc=float(cfg["h03_distance_pc"]), log_lbol=fis.get("log_lbol"),
                lacc_g3=float(obj.qc("stage_g3_qc.json")["combined_accretion"]["l_acc_lsun"]))


def objeto_completo(E, cds, dir_bt, dir_manara, *, fotosfera=True, av_hipotesis=None):
    """Medida de todos los espectros del objeto y su conversión a L_acc."""
    ir = E["perfil"]["ir_spectra"]
    hl = ir["hydrogen_lines"]
    ext, d = E["ext"], E["distancia_pc"]
    ref = hl["btsettl_reference"]
    modelo = btsettl(dir_bt, ref["teff_k"], ref["logg"])
    fot = ({(m["teff_k"], m["logg"]): btsettl(dir_bt, m["teff_k"], m["logg"]) for m in hl["btsettl_photosphere"]}
           if fotosfera else None)
    pls = plantillas_clase3(dir_manara) if fotosfera else None
    # Hα → L_acc a cada A_V y lo que predice en cada línea
    halpha = {}
    for av in E["avs"]:
        c = lacc_de_flujo(E["f_halpha"], "Halpha", av=av, distancia_pc=d, ext=ext)
        pred = {}
        for lin in LINEAS:
            Ll = linea_de_lacc(c["l_acc_lsun"], lin)
            s = dispersion_prediccion(lin)
            pred[lin] = dict(L_linea_lsun=Ll, F_obs=flujo_de_linea(Ll, lin, av=av, distancia_pc=d, ext=ext),
                             sigma_dex=s, F_obs_mas2sigma=flujo_de_linea(Ll * 10 ** (2 * s), lin, av=av,
                                                                          distancia_pc=d, ext=ext))
        halpha[av] = dict(L_ha_lsun=c["l_line_lsun"], L_acc_lsun=c["l_acc_lsun"],
                          log_Lacc_Lbol=(math.log10(c["l_acc_lsun"]) - float(E["log_lbol"])
                                         if E["log_lbol"] is not None else None), prediccion=pred)
    # los espectros
    esp = {}
    for nombre, spec in ir["spectra"].items():
        fac, fuente = factor_escala(ir, spec["band"])
        w, f, s = carga(cds, spec["file"], fac)
        cubre = lineas_cubiertas(w)
        if not cubre:
            continue
        meta = (hl.get("spectra") or {}).get(nombre)
        if not meta or "r_native" not in meta:
            raise SystemExit(f"{E['slug']}: {nombre} cubre {cubre} y hydrogen_lines.spectra no da su R nativa")
        for lin in cubre:
            m = mide_espectro(w, f, s, lin, float(meta["r_native"]), modelo, fotosfera=fot, plantillas=pls)
            r = m["estimadores"][ADOPTADO]
            cont = r["cont"]
            a_fot = float(hl["photospheric_absorption_A"][lin])
            lim = limites(r, cont, a_fot, r["inyeccion"]["media"], r["inyeccion_R500_extra"]["media"])
            por_av = {}
            for av in E["avs"]:
                c = lacc_de_flujo(lim["F_lim3"], lin, av=av, distancia_pc=d, ext=ext)
                h = halpha[av]
                Fp = h["prediccion"][lin]["F_obs"]
                Fp2 = h["prediccion"][lin]["F_obs_mas2sigma"]
                por_av[av] = dict(L_linea_lim3=c["l_line_lsun"], L_acc_lim3=c["l_acc_lsun"],
                                  L_acc_halpha=h["L_acc_lsun"], cociente_Lacc=c["l_acc_lsun"] / h["L_acc_lsun"],
                                  F_pred=Fp, EW_pred=-Fp / cont, umbral5_sobre_pred=lim["F_umbral5"] / Fp,
                                  umbral5_sobre_pred_mas2sigma=lim["F_umbral5"] / Fp2)
            # A_V a la que el umbral a 5σ alcanzaría la predicción (el flujo predicho es log-lineal en A_V)
            av0 = E["av_adoptada"]
            Fp0 = flujo_predicho(halpha_lacc(E, av0), lin, av=av0, distancia_pc=d, ext=ext)
            pend = pendiente_av(lin, ext)
            av_umbral = av0 + math.log10(lim["F_umbral5"] / Fp0) / pend
            esp[f"{nombre}:{lin}"] = dict(
                espectro=nombre, linea=lin, instrumento=spec.get("instrument"), ref=spec.get("ref"),
                banda=spec["band"], fecha=meta.get("date"), factor_escala=fac, factor_escala_fuente=fuente,
                medida=m, cont=cont, a_fot_A=a_fot, eta=r["inyeccion"]["media"],
                eta_suavizado500=r["inyeccion_R500_extra"]["media"], **lim, por_av=por_av,
                pendiente_logF_av=pend, av_umbral5_igual_prediccion=av_umbral)
    out = dict(halpha=halpha, espectros=esp, cobertura=cobertura(E["slug"], E["perfil"], cds))
    out["excesos"] = excesos(E, esp, av_hipotesis or E["avs"])
    return out


def halpha_lacc(E, av):
    return lacc_de_flujo(E["f_halpha"], "Halpha", av=av, distancia_pc=E["distancia_pc"], ext=E["ext"])["l_acc_lsun"]


def excesos(E, esp, avs):
    """Para cada espectro con z ≥ Z_EXCESO: qué implicaría si fuera una línea."""
    ext, d = E["ext"], E["distancia_pc"]
    out = {}
    for clave, x in esp.items():
        r = x["medida"]["estimadores"][ADOPTADO]
        if r["z"] < Z_EXCESO:
            continue
        lin = x["linea"]
        otras = {k: y for k, y in esp.items() if y["linea"] != lin}
        hip = {}
        for av in avs:
            c = lacc_de_flujo(r["flujo"], lin, av=av, distancia_pc=d, ext=ext)
            Fha = flujo_predicho(c["l_acc_lsun"], "Halpha", av=av, distancia_pc=d, ext=ext)
            fila = dict(L_linea=c["l_line_lsun"], L_acc=c["l_acc_lsun"], F_ha_pred=Fha,
                        F_ha_pred_sobre_medida=Fha / E["f_halpha"], otras={})
            for k2, y in otras.items():
                r2 = y["medida"]["estimadores"][ADOPTADO]
                Fo = flujo_predicho(c["l_acc_lsun"], y["linea"], av=av, distancia_pc=d, ext=ext)
                ewp = -Fo / y["cont"]
                fila["otras"][k2] = dict(F_pred=Fo, EW_pred=ewp, EW_medida=r2["ew"], sigma_ew=r2["sigma_ew"],
                                         tension_sigma=(r2["ew"] - ewp) / r2["sigma_ew"],
                                         cociente_pred=Fo / r["flujo"], cociente_lim3=y["F_lim3"] / r["flujo"])
            hip[av] = fila
        est = x["medida"]["estimadores"]
        out[clave] = dict(z=r["z"], ew=r["ew"], sigma_ew=r["sigma_ew"], flujo=r["flujo"],
                          fraccion_controles_mas_emision=r["fraccion_controles_mas_emision"], n=r["n"],
                          fase_px=x["medida"]["fase_px"], centroide=centroide_exceso(x["medida"]),
                          pixeles=x["medida"]["pixeles"],
                          z_otros_estimadores={k: v["z"] for k, v in est.items()},
                          variantes_ventana=x["medida"].get("variantes_ventana"),
                          ew_btsettl_mismo_estimador=[p["ew"] for p in r.get("fotosfera_btsettl", [])],
                          ew_claseIII_mismo_estimador=[p["ew"] for p in r.get("plantillas_claseIII", [])],
                          misma_linea_otros_espectros={k: dict(ew=y["medida"]["estimadores"][ADOPTADO]["ew"],
                                                               sigma_ew=y["medida"]["estimadores"][ADOPTADO]["sigma_ew"],
                                                               z=y["medida"]["estimadores"][ADOPTADO]["z"])
                                                      for k, y in esp.items() if y["linea"] == lin and k != clave},
                          hipotesis_linea=hip)
    return out


# --------------------------------------------------------------------------- salida

def _pot(x, nd=1):
    if not np.isfinite(x) or x == 0:
        return str(x)
    e = int(math.floor(math.log10(abs(x))))
    m = x / 10 ** e
    if round(m, nd) >= 10:
        m, e = m / 10, e + 1
    return f"{m:.{nd}f}\\times10^{{{e}}}"


#: la clave de `paper/refs.bib` de cada artículo de origen de los espectros (el `ref` de `targets/`)
CLAVES_BIB = {"Bowler+2017": "ref:roxs12_characterization", "Bowler+2014": "ref:roxs42b_prior_accretion",
              "Currie+2014": "ref:roxs42b_characterization"}


def tabla_tex(salida):
    """La tabla de una columna con las seis medidas, en el estilo de las del paper (objeto como subcabecera)."""
    filas, avs, refs = [], [], []
    hay_exceso = False
    for slug, res in salida.items():
        E = res["entradas"]
        av = E["av_adoptada"]
        avs.append(f"\\({av:.1f}\\)~mag for {E['nombre'].replace(' ', '~')}")
        filas.append(f"\\multicolumn{{5}}{{l}}{{{E['nombre'].replace(' ', '~')}}} \\\\")
        for clave, x in res["espectros"].items():
            r = x["medida"]["estimadores"][ADOPTADO]
            lin = "Pa\\(\\beta\\)" if x["linea"] == "PaB" else "Br\\(\\gamma\\)"
            inst = (x["instrumento"] or "").split("/")[-1]
            ref = CLAVES_BIB.get(x["ref"])
            if ref and ref not in refs:
                refs.append(ref)
            marca = "\\tablefootmark{a}" if r["z"] >= Z_EXCESO else ""
            hay_exceso |= bool(marca)
            filas.append(f"\\quad {inst} \\({x['banda']}\\), {lin}{marca} & "
                         f"\\({r['ew']:+.2f}\\pm{r['sigma_ew']:.2f}\\) & "
                         f"\\({r['z']:+.1f}\\) & \\(<{-x['EW_lim3']:.1f}\\) & "
                         f"\\(<{_pot(x['por_av'][av]['L_acc_lim3'])}\\) \\\\")
    cab = ("\\begin{table}\n\\caption{Pa\\(\\beta\\) and Br\\(\\gamma\\) in the published near-infrared spectra of "
           "the companions.}\n\\label{tab:pab_brg}\n\\centering\n\\small\n\\setlength{\\tabcolsep}{4pt}\n"
           "\\begin{tabular}{l c c c c}\n\\hline\\hline\n"
           "Spectrum, line & EW (\\AA) & \\(z\\) & \\(W_{3\\sigma}\\) (\\AA) & \\(\\lacc\\) (\\(L_\\odot\\)) \\\\\n"
           "\\hline\n")
    pie = ("\\hline\n\\end{tabular}\n\\tablefoot{Spectra of " + ", ".join(f"\\citet{{{k}}}" for k in refs)
           + ", as compiled by \\citet{ref:hayoz2026}. EW is the matched-filter equivalent width (negative in "
           "emission) with the standard deviation of the same filter at the line-free positions of the same "
           "spectrum, and \\(z\\) is the line flux over that deviation. \\(W_{3\\sigma}\\) is the \\(3\\sigma\\) upper "
           "limit on the equivalent width in emission; it includes the measured flux, when positive, and the "
           "photospheric absorption that an emission line must first fill. \\(\\lacc\\) is the limit from it with "
           "the relations of \\citet{ref:scaling_relations} at the adopted \\(A_V\\) (" + " and ".join(avs) + ")."
           + ("\n\\tablefoottext{a}{Not a line (Appendix~\\ref{app:pab_brg}); the limit is given as for the other "
              "spectra.}" if hay_exceso else "") + "}\n\\end{table}\n")
    return cab + "\n".join(filas) + "\n" + pie


def _imprime(res):
    E = res["entradas"]
    print(f"\n==== {E['nombre']} ({E['run']}) ====")
    print(f"Hα: F = {E['f_halpha']:.4e} ({E['f_halpha_fuente']}); G3 L_acc = {E['lacc_g3']:.4e} L☉; "
          f"A_V adoptada {E['av_adoptada']}, A_V {E['avs']}; d {E['distancia_pc']} pc")
    if E["av_prior"]:
        p = E["av_prior"]
        print(f"   A_V con prior (BT-Settl, {p['etiqueta']}): p50 {p['av_p50']:.4f} → {p['av_redondeada']}")
    for nombre, c in res["cobertura"].items():
        if c.get("falta"):
            print(f"   {nombre}: falta {c['fichero']}")
            continue
        print(f"   {nombre:10s} {c['lam_min_um']:.4f}–{c['lam_max_um']:.4f} µm: cubre {c['lineas'] or 'ninguna'}"
              f"{'' if c['medido'] else ' (solo se mira)'}")
    for av, h in res["halpha"].items():
        p = h["prediccion"]
        print(f"   A_V {av}: L_acc(Hα) {'<' if E['limite'] else ''}{h['L_acc_lsun']:.4e} L☉; predicho Paβ "
              f"L {p['PaB']['L_linea_lsun']:.3e} F {p['PaB']['F_obs']:.3e}; Brγ L {p['BrG']['L_linea_lsun']:.3e} "
              f"F {p['BrG']['F_obs']:.3e}")
    for clave, x in res["espectros"].items():
        m = x["medida"]
        print(f"\n  {clave} ({x['instrumento']}, {x['ref']}, {x['fecha']}): píxel {m['pixel_A']:.1f} Å, R {m['r_nativa']:.0f},"
              f" Δλ {m['desplazamiento']['delta_A']:+.2f} Å ({m['desplazamiento']['delta_px']:+.3f} px), fase "
              f"{m['fase_px']:+.3f} px, escala ×{x['factor_escala']:.4f} ({x['factor_escala_fuente']})")
        for nombre, r in m["estimadores"].items():
            fot = [p["ew"] for p in r.get("fotosfera_btsettl", [])]
            pl = [p["ew"] for p in r.get("plantillas_claseIII", [])]
            print(f"    {nombre:17s} EW {r['ew']:+.4f} ± {r['sigma_ew']:.4f}  z {r['z']:+.3f}  F {r['flujo']:+.4e} ± "
                  f"{r['sigma_flujo']:.4e}  N {r['n']} ({r['n_disjuntos']})  η {r['inyeccion']['media']:.4f}/"
                  f"{r['inyeccion_R500_extra']['media']:.4f}  p {r['fraccion_controles_mas_emision']:.3f}"
                  + (f"  BT {min(fot):+.2f}..{max(fot):+.2f}" if fot else "")
                  + (f"  ClaseIII {np.median(pl):+.2f} [{min(pl):+.2f},{max(pl):+.2f}]" if pl else ""))
        print("    ventanas:", {k: f"{v['ew']:+.2f}±{v['sigma_ew']:.2f} (z {v['z']:+.2f}, N {v['n']})"
                                 for k, v in m.get("variantes_ventana", {}).items()})
        print(f"    3σ: F < {x['F_lim3']:.4e}, EW > {x['EW_lim3']:.4f} Å; umbral 5σ F {x['F_umbral5']:.4e} "
              f"(EW {x['EW_umbral5']:.4f}); η {x['eta']:.4f}; a_fot {x['a_fot_A']} Å; con R 500: F < "
              f"{x['F_lim3_suavizado500']:.4e}")
        for av, g in x["por_av"].items():
            print(f"      A_V {av}: L_acc < {g['L_acc_lim3']:.4e} (×{g['cociente_Lacc']:.2f} Hα)  EW pred "
                  f"{g['EW_pred']:+.5f}  umbral5/pred {g['umbral5_sobre_pred']:.2f} (+2σ {g['umbral5_sobre_pred_mas2sigma']:.2f})")
        print(f"      d log F_pred / d A_V = {x['pendiente_logF_av']:.4f}; umbral 5σ = predicción a A_V "
              f"{x['av_umbral5_igual_prediccion']:.2f}")
    for clave, h in res["excesos"].items():
        c = h["centroide"]
        print(f"\n  EXCESO {clave}: z {h['z']:+.3f} (p {h['fraccion_controles_mas_emision']:.3f} de {h['n']}), fase "
              f"{h['fase_px']:+.3f} px; centroide {c['desde_lam0_A']:+.1f} Å = {c['desde_lam0_kms']:+.0f} km/s"
              if c else f"\n  EXCESO {clave}: z {h['z']:+.3f}")
        print("    píxeles:", ", ".join(f"{p['lam_A'] / 1e4:.4f}: {p['f'] / 1e-17:.2f}±{p['e'] / 1e-17:.2f} "
                                         f"(cont {p['cont'] / 1e-17:.2f}){'*' if p['en_ventana'] else ''}"
                                         for p in h["pixeles"]))
        if c:
            print("    exceso por píxel (10⁻¹⁷) y fracción que pondría ahí una línea no resuelta en λ0:",
                  ", ".join(f"{x['lam_A'] / 1e4:.4f}: {x['exceso'] / 1e-17:+.2f}±{x['error'] / 1e-17:.2f} "
                            f"[{x['fraccion_esperada']:.2f}]" for x in c["exceso_por_pixel"]))
        print("    z por estimador:", {k: round(v, 2) for k, v in h["z_otros_estimadores"].items()})
        print("    misma línea en otros espectros:", h["misma_linea_otros_espectros"])
        for av, f in h["hipotesis_linea"].items():
            o = "; ".join(f"{k}: EW pred {v['EW_pred']:+.2f} frente a {v['EW_medida']:+.2f} ± {v['sigma_ew']:.2f} "
                          f"({v['tension_sigma']:+.1f}σ), cociente pred {v['cociente_pred']:.2f}, 3σ {v['cociente_lim3']:.2f}"
                          for k, v in f["otras"].items())
            print(f"    si fuera línea, A_V {av}: L_acc {f['L_acc']:.3e}, Hα ×{f['F_ha_pred_sobre_medida']:.0f} la medida; {o}")


def _rango(vals):
    vals = [float(v) for v in vals if v is not None and np.isfinite(v)]
    return [min(vals), max(vals)] if vals else None


def cifras_parrafo(salida):
    """Las cifras que cita el párrafo del apéndice, sin redondear, por objeto."""
    out = {}
    for slug, res in salida.items():
        E = res["entradas"]
        esp = res["espectros"]
        av0 = E["av_adoptada"]
        avs = E["avs"]
        c = dict(nombre=E["nombre"], halpha=("límite" if E["limite"] else "detección"), f_halpha=E["f_halpha"],
                 avs=avs, av_adoptada=av0,
                 lacc_halpha={a: h["L_acc_lsun"] for a, h in res["halpha"].items()},
                 cobertura={k: (v.get("lam_min_um"), v.get("lam_max_um"), v.get("lineas"))
                            for k, v in res["cobertura"].items()},
                 n_controles=_rango([x["medida"]["estimadores"][ADOPTADO]["n"] for x in esp.values()]),
                 n_disjuntos=_rango([x["medida"]["estimadores"][ADOPTADO]["n_disjuntos"] for x in esp.values()]),
                 eta=_rango([x["eta"] for x in esp.values()]),
                 eta_suavizado500=_rango([x["eta_suavizado500"] for x in esp.values()]),
                 limite_suavizado500_sobre_nominal=_rango([x["F_lim3_suavizado500"] / x["F_lim3"] for x in esp.values()]),
                 raiz_r_sobre_500=_rango([math.sqrt(x["medida"]["r_nativa"] / R_SUAVIZADO) for x in esp.values()]),
                 desplazamiento_lambda_A=_rango([x["medida"]["desplazamiento"]["delta_A"] for x in esp.values()]),
                 espectros={})
        for k, x in esp.items():
            r = x["medida"]["estimadores"][ADOPTADO]
            g = x["por_av"][av0]
            c["espectros"][k] = dict(ew=r["ew"], sigma_ew=r["sigma_ew"], z=r["z"], ew_lim3=-x["EW_lim3"],
                                     F_lim3=x["F_lim3"], lacc_lim3_av_adoptada=g["L_acc_lim3"],
                                     lacc_lim3_sobre_halpha=g["cociente_Lacc"],
                                     ew_pred=_rango([-x["por_av"][a]["EW_pred"] for a in avs]),
                                     umbral5_sobre_pred=_rango([x["por_av"][a]["umbral5_sobre_pred"] for a in avs]),
                                     umbral5_sobre_pred_mas2sigma=_rango([x["por_av"][a]["umbral5_sobre_pred_mas2sigma"]
                                                                         for a in avs]),
                                     # con ruido de fotones, a la R nativa el ruido en EW baja como √(R/500) [modelo]
                                     umbral5_sobre_pred_r_nativa=_rango(
                                         [x["por_av"][a]["umbral5_sobre_pred"]
                                          / math.sqrt(x["medida"]["r_nativa"] / R_SUAVIZADO) for a in avs]),
                                     av_umbral5=x["av_umbral5_igual_prediccion"],
                                     fotosfera_btsettl=_rango([p["ew"] for p in r.get("fotosfera_btsettl", [])]),
                                     plantillas_claseIII=_rango([p["ew"] for p in r.get("plantillas_claseIII", [])]))
        for lin in LINEAS:
            de = {k: v for k, v in c["espectros"].items() if esp[k]["linea"] == lin}
            if not de:
                continue
            c[lin] = dict(ew_lim3=_rango([v["ew_lim3"] for v in de.values()]),
                          lacc_lim3=_rango([v["lacc_lim3_av_adoptada"] for v in de.values()]),
                          lacc_lim3_sobre_halpha=_rango([v["lacc_lim3_sobre_halpha"] for v in de.values()]),
                          ew_pred=_rango([y for v in de.values() for y in v["ew_pred"]]),
                          umbral5_sobre_pred=_rango([y for v in de.values() for y in v["umbral5_sobre_pred"]]),
                          umbral5_sobre_pred_mas2sigma=_rango([y for v in de.values()
                                                               for y in v["umbral5_sobre_pred_mas2sigma"]]),
                          av_umbral5=_rango([v["av_umbral5"] for v in de.values()]),
                          pendiente_logF_av=esp[next(iter(de))]["pendiente_logF_av"],
                          fotosfera_btsettl=_rango([y for v in de.values() for y in (v["fotosfera_btsettl"] or [])]),
                          plantillas_claseIII=_rango([y for v in de.values() for y in (v["plantillas_claseIII"] or [])]),
                          L_linea_pred_av_adoptada=res["halpha"][av0]["prediccion"][lin]["L_linea_lsun"])
        todos = list(c["espectros"].values())
        c["umbral5_sobre_pred"] = _rango([y for v in todos for y in v["umbral5_sobre_pred"]])
        c["umbral5_sobre_pred_mas2sigma"] = _rango([y for v in todos for y in v["umbral5_sobre_pred_mas2sigma"]])
        c["umbral5_sobre_pred_r_nativa"] = _rango([y for v in todos for y in v["umbral5_sobre_pred_r_nativa"]])
        c["ew_pred"] = _rango([y for v in todos for y in v["ew_pred"]])
        if "PaB" in c:
            tw = LITERATURA["twa27b_L_pab_lsun"]["valor"]
            c["twa27b_L_pab_sobre_pred"] = tw / c["PaB"]["L_linea_pred_av_adoptada"]
        out[slug] = c
    return out


def _imprime_cifras(cif):
    print("\n==== cifras del párrafo ====")
    for slug, c in cif.items():
        print(f"{c['nombre']} (Hα {c['halpha']}, F {c['f_halpha']:.4e}; A_V {c['avs']}):")
        print(f"   controles N {c['n_controles']}, disjuntos {c['n_disjuntos']}; η {c['eta']}; η(R 500) "
              f"{c['eta_suavizado500']} → límite ×{c['limite_suavizado500_sobre_nominal']}; √(R/500) {c['raiz_r_sobre_500']}")
        for k, v in c["espectros"].items():
            print(f"   {k:14s} EW {v['ew']:+.3f} ± {v['sigma_ew']:.3f} z {v['z']:+.2f}; EW_3σ {v['ew_lim3']:.3f} Å; "
                  f"L_acc < {v['lacc_lim3_av_adoptada']:.3e} (×{v['lacc_lim3_sobre_halpha']:.1f} Hα); EW pred "
                  f"{v['ew_pred'][0]:.4f}–{v['ew_pred'][1]:.4f}; umbral/pred {v['umbral5_sobre_pred'][0]:.1f}–"
                  f"{v['umbral5_sobre_pred'][1]:.1f} (+2σ {v['umbral5_sobre_pred_mas2sigma'][0]:.2f}–"
                  f"{v['umbral5_sobre_pred_mas2sigma'][1]:.2f}); A_V umbral {v['av_umbral5']:.2f}")
        for lin in LINEAS:
            if lin in c:
                v = c[lin]
                print(f"   {LINEAS[lin]['nombre']}: EW_3σ {v['ew_lim3']}; L_acc< {v['lacc_lim3']}; EW pred {v['ew_pred']}; "
                      f"umbral/pred {v['umbral5_sobre_pred']}; A_V umbral {v['av_umbral5']}; pendiente "
                      f"{v['pendiente_logF_av']:.4f}; BT-Settl {v['fotosfera_btsettl']}; Clase III {v['plantillas_claseIII']}; "
                      f"L pred (A_V adoptada) {v['L_linea_pred_av_adoptada']:.3e}")
        print(f"   todas: EW pred {c['ew_pred']}; umbral/pred {c['umbral5_sobre_pred']} (+2σ {c['umbral5_sobre_pred_mas2sigma']};"
              f" a la R nativa con ruido de fotones {c['umbral5_sobre_pred_r_nativa']})")
        if "twa27b_L_pab_sobre_pred" in c:
            print(f"   TWA 27B L(Paβ) / predicha a la A_V adoptada: {c['twa27b_L_pab_sobre_pred']:.2f}")
    print("literatura (constantes del script, con su fuente):")
    for k, v in LITERATURA.items():
        print(f"   {k}: {v['valor']} — {v['fuente']}")


def _pares(lista, conv=float):
    out = {}
    for x in lista or []:
        k, v = x.split("=", 1)
        out[k] = conv(v)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--project-root", default=None, help="raíz que contiene runs/ (desde un worktree, el clon que los tiene)")
    ap.add_argument("--cds", default=str(DEFAULT_CDS), help="catálogo J/A+A/708/A312")
    ap.add_argument("--btsettl", default=str(DEFAULT_BTSETTL), help="npz de BT-Settl")
    ap.add_argument("--manara", default=str(DEFAULT_MANARA), help="catálogos CDS de las plantillas de Manara")
    ap.add_argument("--cadenas", action="append", default=[],
                    help="directorio de cadenas de teff_bayes (repetible): añade la A_V con prior")
    ap.add_argument("--f-halpha", action="append", default=[], metavar="SLUG=F",
                    help="flujo de Hα observado (cgs) en vez del de los productos (reproducir la sesión)")
    ap.add_argument("--av", action="append", default=[], metavar="SLUG=A1,A2,...", help="las A_V en vez de las de hoy")
    ap.add_argument("--av-hipotesis", default=None, help="A_V de la hipótesis del exceso (por defecto, las del objeto)")
    ap.add_argument("--sin-fotosfera", action="store_true", help="sin BT-Settl ni plantillas en la ventana (más rápido)")
    ap.add_argument("--json", default=None, help="escribe aquí todo lo calculado")
    ap.add_argument("--tabla", default=None, help="escribe aquí la tabla LaTeX")
    args = ap.parse_args(argv)

    import scripts.paper_figures as pf
    if args.project_root:
        pf.PROYECTO = Path(args.project_root).resolve()
    fh = _pares(args.f_halpha)
    avs = _pares(args.av, lambda v: [float(a) for a in v.split(",")])
    avh = [float(a) for a in args.av_hipotesis.split(",")] if args.av_hipotesis else None
    salida = {}
    for obj in pf.objetos_del_paper():
        E = entradas_objeto(obj, f_halpha=fh.get(obj.slug), avs=avs.get(obj.slug), cadenas=args.cadenas)
        res = objeto_completo(E, args.cds, args.btsettl, args.manara, fotosfera=not args.sin_fotosfera,
                              av_hipotesis=avh)
        res["entradas"] = {k: v for k, v in E.items() if k not in ("cfg", "ext", "perfil")}
        salida[obj.slug] = res
        _imprime(res)
    cif = cifras_parrafo(salida)
    _imprime_cifras(cif)
    if args.json:
        Path(args.json).write_text(json.dumps(dict(objetos=salida, cifras=cif, literatura=LITERATURA,
                                                   relaciones={k: relacion(k) for k in ("Halpha", *LINEAS)}),
                                              indent=1, ensure_ascii=False, default=float))
    if args.tabla:
        Path(args.tabla).write_text(tabla_tex(salida))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
