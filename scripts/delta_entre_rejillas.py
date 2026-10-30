#!/usr/bin/env python
"""Δ entre rejillas de atmósferas: su dispersión, un Δ que la incluye y dónde fallan los modelos en MUSE.

Solo lectura y fuera de la cadena (punto 4 de `docs/2026-10-21_lo_que_pediria_un_referi.md`, «Los modelos de
atmósfera no reproducen el espectro MUSE»; apéndice de comprobaciones del 2026-10-28). Hermano de
`teff_bayes.py rejillas` (el cuerpo de la Tabla B.7) y de `teff_cifras_texto.py` (sus frases): lee las MISMAS
cadenas de `teff_bayes.py ajusta` (json + npz) y no lanza ninguna.

Δ es, por muestra de la posterior, log(L_acc/L_bol) − nivel de ruido cromosférico (Manara et al. 2013, Ec. 2) a
la T_eff de la muestra (`teff_bayes.margen_de_corrida`: L_acc del run del paper reescalada a la A_V de la muestra,
L_bol = 4πR²σT⁴ de la muestra). El paper da Δ por rejilla (Tabla B.7, `tab:teff_grids`) con A_V = 1.8, 3.0 mag y
con el prior de la primaria; ningún Δ lleva un error que incluya la dispersión entre rejillas. Aquí, por compañero
y por caso de A_V:

1. Reproduce la Tabla B.7 desde las cadenas (`teff_bayes.filas_rejillas`, la función que escribe su cuerpo) y,
   con `--tex`, la compara fila a fila con la del paper: tiene que salir igual al redondeo impreso, o el script
   sale con 1.
2. (a) la mediana y el rango, entre rejillas, de la mediana de Δ de cada rejilla; (b) la desviación típica entre
   rejillas de esas medianas (ddof = 1; también la poblacional, y sin cada rejilla, para decir cuál la fija);
   (c) un Δ combinado: la mezcla a pesos iguales de
   las posteriores de las rejillas (muestras juntadas, cada rejilla con el mismo peso total), con su mediana, su
   16–84 % y P(Δ > 0); (d) como variante, la mezcla pesada por la evidencia (Z = e^{ln Z}, con prior 1/N por
   rejilla), diciendo si una rejilla se lleva todo el peso (> 0.99); (e) la suma en cuadratura de la anchura
   típica por rejilla (la mediana de sus semianchos 16–84 %) y la dispersión (b), para compararla con el semiancho
   de (c).
3. Con `--residuos`: para el ajuste con el prior en A_V de cada rejilla, el modelo de máxima verosimilitud
   (`mejor` de la cadena) reconstruido con `teff_bayes.Modelo` y `Modelo.modelos` (no se reescribe; se comprueba
   que su `loglike` da el `logl_max` de la cadena), y en el espectro MUSE el residuo normalizado
   (dato − modelo)/σ con la σ del ajuste, σ² = e² + (f_b·modelo)², por banda: media, rms, χ²/n, la fracción del χ²
   de MUSE y la del número de puntos, el residuo relativo (dato − modelo)/modelo (media y rms) y el χ² con solo el
   error del dato.

Los conjuntos de rejillas (cada uno con su propia (a)–(e)):

- `r500` (el cálculo principal): las de la Tabla B.7 a R = 500. Sonora Diamondback es UNA rejilla (el paper la
  cuenta así entre las siete); sus seis modelos (cinco f_sed y sin nubes) pesan lo mismo dentro de ella. Cada
  rejilla pesa lo mismo en la mezcla tenga las muestras que tenga su cadena.
- `r250`: las de R = 250 (BT-Settl y SPHINX, píxeles agrupados de a dos), aparte: su ln Z no se compara con el de
  R = 500 (otros datos).
- variantes: `r500_fsed` (cada f_sed de Diamondback como una rejilla), `siete` (las de R = 500 más SPHINX de
  R = 250: las siete rejillas del paper, una vez cada una; sin (d), porque mezcla resoluciones) y, con
  `--cadenas-atmo`, `r500_atmo` y `r250_atmo` (con ATMO 2020, que el paper da solo como prueba, «we do not use
  these values»; de su rejilla se toma la de `paper_cifras_atmo.rejilla_atmo`: la de 2000–3000 K salvo que su
  ajuste con el prior toque el borde de T_eff y esté la ampliada).

ATMO 2020 y `btsettl_bin` (el control de su muestreo) nunca entran en `r500`/`r250` aunque sus carpetas se pasen
en `--cadenas`.

Bandas de los residuos (`--bandas`, bordes en Å): el ajuste usa MUSE en 6300–9349.5 Å (`g3_fit_wave_range_A`) con
las máscaras de G3 (O₂ B y A, ±300 km/s alrededor de las líneas del catálogo, canales malos del fondo) y las de H₂O
de `tipo_ir_plantillas.H2O_MUSE_A`, en bins de 15 Å; los bordes por defecto caen dentro de los huecos (O₂ B, O₂ A,
H₂O 8100–8400 y 8900–9100 Å), así que cada banda es un tramo contiguo de datos. Por debajo de 6300 Å no hay
puntos del ajuste.

Las rejillas de ATMO 2020 se convierten a la ruta que se quiera (`convierte_atmo.py`): sus residuos necesitan
`--dir-rejilla atmo_ceq=DIR` (y `atmo_ceq1500=DIR`); sin el directorio, se omiten y el json lo dice.

Comparaciones:

- `--nivel-manara`: la dispersión de las 22 Class III de Manara et al. (2013) alrededor del nivel
  (`halpha_criterios.poblacion_manara`, con la relación L_acc–L_Hα del run del paper), que el paper ya da
  (0.25 dex) y que Δ no incluye;
- `--filas`: en las mismas bandas, la fotometría de `banda_de_filas.py` en el cubo con la primaria restada por
  exposición (el compañero, su fila sin compañero y otras filas): dice si el flujo de una banda es del compañero o
  de la banda de filas que el paper encontró en el azul de ROXs 42B b.

uso (las cadenas del paper, C4 v2 `decoupled_c0`):

    python scripts/delta_entre_rejillas.py --project-root RAIZ_CON_RUNS \\
        --cadenas ~/muse_noche_20261024_c4/cadenas \\
        --cadenas-atmo ~/muse_sesion_20261026/claudio/atmo/res \\
        --dir-rejilla atmo_ceq=~/muse_sesion_20261026/claudio/atmo/npz_ceq \\
        --dir-rejilla atmo_ceq1500=~/muse_sesion_20261026/claudio/atmo/npz_ceq1500 \\
        --tex RAIZ_CON_RUNS/paper/main.tex --residuos --nivel-manara --filas \\
        [--json SALIDA.json] [--md SALIDA.md]

Sale con 1 si la Tabla B.7 no se reproduce fila a fila (con `--tex`) o si algún modelo reconstruido no da el
`logl_max` de su cadena.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import teff_bayes as tb  # noqa: E402
from scripts import teff_cifras_texto as tc  # noqa: E402

#: las tres elecciones de A_V (como las etiquetas de las cadenas): fija 1.8, fija 3.0 y con el prior de la primaria
AVS = tc.AVS
#: las rejillas que el paper da solo como prueba (párrafo de ATMO 2020 del apéndice B): fuera del cálculo principal
PRUEBAS = ("atmo_ceq", "atmo_ceq1500", "btsettl_bin")
#: las rejillas de ATMO 2020 que pueden entrar en las variantes (la ampliada solo si la de 2000 K toca el borde)
ATMO = ("atmo_ceq", "atmo_ceq1500")
#: una rejilla «se lleva todo el peso» de la mezcla por evidencia si su peso pasa de esto
DOMINA = 0.99
#: bordes por defecto de las bandas de los residuos (Å): dentro de los huecos de las máscaras del ajuste
BANDAS_A = (6300.0, 6900.0, 7650.0, 8250.0, 9000.0, 9350.0)


# ================================================================== piezas puras
def familia(rejilla):
    """La rejilla «física» de una corrida: las seis de Diamondback (una por f_sed) son una, Diamondback, y las dos
    de ATMO 2020 (2000 y 1500 K) son una, ATMO."""
    if rejilla.startswith("diamondback"):
        return "diamondback"
    if rejilla in ATMO:
        return "atmo"
    return rejilla


def cuantiles_pesados(x, w, qs):
    """Cuantiles (en %) de una muestra con pesos, con la CDF en el punto medio de cada peso (con pesos iguales es la
    definición «hazen» de numpy, (i + ½)/n). Fuera del primer o del último punto medio, el mínimo o el máximo."""
    x = np.asarray(x, float)
    w = np.asarray(w, float)
    o = np.argsort(x, kind="stable")
    x, w = x[o], w[o]
    c = (np.cumsum(w) - 0.5 * w) / w.sum()
    return np.interp(np.asarray(qs, float) / 100.0, c, x)


def mezcla(muestras, pesos):
    """(x, w) de la mezcla de cadenas: `muestras` una lista de arrays (una cadena cada uno) y `pesos` el peso TOTAL
    de cada cadena; dentro de una cadena, todas sus muestras pesan lo mismo. sum(w) = 1."""
    if len(muestras) != len(pesos):
        raise ValueError("una lista de pesos por cadena")
    xs, ws = [], []
    for m, p in zip(muestras, pesos):
        m = np.asarray(m, float)
        if not m.size:
            raise ValueError("una cadena sin muestras")
        xs.append(m)
        ws.append(np.full(m.size, float(p) / m.size))
    x, w = np.concatenate(xs), np.concatenate(ws)
    return x, w / w.sum()


def pesos_iguales(grupos):
    """Peso de cada cadena con cada rejilla pesando lo mismo: 1/G por rejilla, repartido a partes iguales entre sus
    cadenas. `grupos`: {rejilla: [etiqueta, ...]}; devuelve {etiqueta: peso}."""
    G = len(grupos)
    return {e: 1.0 / (G * len(ets)) for ets in grupos.values() for e in ets}


def pesos_evidencia(grupos, lnz):
    """Peso de cada cadena ∝ prior × Z, con el prior de `pesos_iguales` (1/G por rejilla, repartido entre sus
    cadenas: las seis f_sed de Diamondback son seis modelos de una rejilla) y Z = exp(ln Z). `lnz`: {etiqueta: ln Z}.
    Solo tiene sentido entre cadenas ajustadas a los MISMOS datos (misma resolución)."""
    prior = pesos_iguales(grupos)
    lmax = max(lnz[e] for e in prior)
    p = {e: prior[e] * math.exp(lnz[e] - lmax) for e in prior}
    s = sum(p.values())
    return {e: v / s for e, v in p.items()}


def resume(x, w):
    """Mediana, 16–84 % (semiancho), media, desviación típica y P(Δ > 0) de una mezcla (x, w)."""
    x, w = np.asarray(x, float), np.asarray(w, float)
    w = w / w.sum()
    p16, p50, p84 = cuantiles_pesados(x, w, (16, 50, 84))
    media = float(np.sum(w * x))
    return dict(p16=float(p16), p50=float(p50), p84=float(p84), semiancho=float((p84 - p16) / 2), media=media,
                sd=float(math.sqrt(np.sum(w * (x - media) ** 2))), p_positivo=float(np.sum(w[x > 0])))


def resumen_cadena(m):
    """Lo mismo para UNA cadena, con los percentiles de `teff_bayes.percentiles` (los de la Tabla B.7)."""
    m = np.asarray(m, float)
    m = m[np.isfinite(m)]
    q = tb.percentiles(m)
    return dict(p16=q["p16"], p50=q["p50"], p84=q["p84"], semiancho=(q["p84"] - q["p16"]) / 2,
                media=float(m.mean()), sd=float(m.std()), p_positivo=float(np.mean(m > 0)))


def cuadratura(semianchos, medianas):
    """(e): la anchura típica por rejilla (mediana de sus semianchos 16–84 %) y la dispersión entre rejillas (la
    desviación típica de sus medianas, ddof = 1) sumadas en cuadratura."""
    tip = float(np.median(semianchos))
    s = float(np.std(medianas, ddof=1)) if len(medianas) > 1 else 0.0
    return dict(anchura_tipica=tip, dispersion=s, total=float(math.hypot(tip, s)))


def estadisticas(grupos, deltas, lnz=None):
    """(a)–(e) de un conjunto de rejillas. `grupos`: {rejilla: [etiqueta, ...]} (Diamondback, con sus seis f_sed);
    `deltas`: {etiqueta: muestras de Δ}; `lnz`: {etiqueta: ln Z} para (d), o None (no comparable)."""
    por = {}
    for g, ets in grupos.items():
        if len(ets) == 1:
            r = resumen_cadena(deltas[ets[0]])
        else:
            r = resume(*mezcla([deltas[e] for e in ets], [1.0 / len(ets)] * len(ets)))
        por[g] = dict(r, cadenas=list(ets))
    med = {g: v["p50"] for g, v in por.items()}
    orden = sorted(med, key=med.get)
    pi = pesos_iguales(grupos)
    ets = [e for v in grupos.values() for e in v]
    c = resume(*mezcla([deltas[e] for e in ets], [pi[e] for e in ets]))
    # (b) sin cada rejilla: la que, quitada, más la reduce es la que la fija
    sin = {g: float(np.std([m for gg, m in med.items() if gg != g], ddof=1)) for g in med} if len(med) > 2 else {}
    out = dict(
        n_rejillas=len(grupos), por_rejilla=por,
        a=dict(mediana=float(np.median(list(med.values()))), min=med[orden[0]], max=med[orden[-1]],
               rejilla_min=orden[0], rejilla_max=orden[-1]),
        b=dict(sd=float(np.std(list(med.values()), ddof=1)) if len(med) > 1 else 0.0,
               sd_poblacional=float(np.std(list(med.values()))), sd_sin=sin,
               rejilla_que_la_fija=min(sin, key=sin.get) if sin else None),
        c=c,
        e=dict(cuadratura([v["semiancho"] for v in por.values()], list(med.values())),
               cociente=(float(np.std(list(med.values()), ddof=1)) / float(np.median([v["semiancho"] for v in por.values()]))
                         if len(med) > 1 else 0.0),
               semiancho_mezcla=c["semiancho"],
               semiancho_rango=[min(v["semiancho"] for v in por.values()), max(v["semiancho"] for v in por.values())]))
    if lnz is not None:
        pe = pesos_evidencia(grupos, lnz)
        d = resume(*mezcla([deltas[e] for e in ets], [pe[e] for e in ets]))
        pg = {g: float(sum(pe[e] for e in v)) for g, v in grupos.items()}
        g_max = max(pg, key=pg.get)
        lmax = max(lnz[e] for e in ets)
        out["d"] = dict(d, peso_rejilla=pg, peso_cadena=pe, rejilla_dominante=g_max, peso_dominante=pg[g_max],
                        domina=pg[g_max] > DOMINA, dlnz={e: lnz[e] - lmax for e in ets})
    else:
        out["d"] = None
    return out


#: cada variante, frente a qué conjunto se compara
REFERENCIA_DE = {"r500_fsed": "r500", "siete": "r500", "r500_atmo": "r500", "r250_atmo": "r250"}


def diferencias_variantes(conjuntos):
    """Por variante y compañero, el mayor cambio (en valor absoluto) de la mediana y de P(Δ > 0) de la mezcla (c)
    frente a su conjunto de referencia (`REFERENCIA_DE`), entre las tres A_V."""
    out = {}
    for k, v in conjuntos.items():
        ref = REFERENCIA_DE.get(v["conjunto"])
        if ref is None:
            continue
        r = conjuntos.get(f"{v['slug']}|{ref}|{v['av']}")
        if r is None:
            continue
        o = out.setdefault(f"{v['slug']}|{v['conjunto']}", dict(referencia=ref, dp50_max=0.0, dp_positivo_max=0.0,
                                                                 por_av={}))
        dp50, dpp = v["c"]["p50"] - r["c"]["p50"], v["c"]["p_positivo"] - r["c"]["p_positivo"]
        o["por_av"][v["av"]] = dict(dp50=dp50, dp_positivo=dpp)
        o["dp50_max"] = max(o["dp50_max"], abs(dp50))
        o["dp_positivo_max"] = max(o["dp_positivo_max"], abs(dpp))
    return out


# ================================================================== la Tabla B.7
def filas_tabla_tex(tex, etiqueta="tab:teff_grids"):
    r"""Las filas del cuerpo de una tabla del paper (las que empiezan por `\quad` o `\multicolumn{N}{l}`), sin
    blancos a los lados, desde su `\label` hasta su `\end{tabular}`."""
    i = tex.index(r"\label{" + etiqueta + "}")
    j = tex.index(r"\end{tabular}", i)
    return [ln.strip() for ln in tex[i:j].splitlines()
            if ln.strip().startswith(r"\quad") or ln.strip().startswith(r"\multicolumn{8}{l}")]


def celdas_delta(fila):
    """Las tres celdas de Δ (A_V 1.8, 3.0 y prior) de una fila de la Tabla B.7, o None si es una cabecera."""
    if not fila.startswith(r"\quad"):
        return None
    c = [x.strip() for x in fila.rstrip("\\").strip().split(" & ")]
    return c[4:7]


def compara_tabla(nuestras, impresas):
    """Fila a fila: {'iguales': bool, 'n': ..., 'distintas': [(i, nuestra, impresa)], 'delta_distintas': [...]}."""
    n = max(len(nuestras), len(impresas))
    dist, ddist = [], []
    for i in range(n):
        a = nuestras[i] if i < len(nuestras) else None
        b = impresas[i] if i < len(impresas) else None
        if a != b:
            dist.append((i, a, b))
            if a is None or b is None or celdas_delta(a) != celdas_delta(b):
                ddist.append((i, a, b))
    return dict(iguales=not dist, n_nuestras=len(nuestras), n_impresas=len(impresas), distintas=dist,
                delta_distintas=ddist)


# ================================================================== las cadenas
def cadenas_b7(K, slug, excluye=PRUEBAS):
    """{(rejilla, r250): {av: cadena}} de la Tabla B.7 de un compañero (`teff_cifras_texto.rejillas_b7`: las de
    `teff_bayes.REJILLAS_TABLA` de las que hay alguna cadena), sin las rejillas de prueba. Exige las tres A_V."""
    out = {}
    for rj, r in tc.rejillas_b7(K, slug):
        if rj in excluye:
            continue
        out[(rj, r)] = dict(zip(AVS, K.exige([tc.etiqueta(slug, rj, a, r250=r) for a in AVS])))
    return out


def cadenas_atmo(KA, slug):
    """{(rejilla, r250): {av: cadena}} de ATMO 2020 de un compañero: a R = 500 la rejilla de
    `paper_cifras_atmo.rejilla_atmo` (con las cadenas de semilla 1 y el degradado por defecto), y a R = 250 la de
    2000–3000 K si está. Lo que falta, no entra."""
    from scripts import paper_cifras_atmo as pca
    previas = [KA.por_etiqueta[e] for rj in ATMO for e in [tc.etiqueta(slug, rj)] if e in KA.por_etiqueta]
    out = {}
    if previas:
        rj = pca.rejilla_atmo(pca.indice(previas), slug)
        ets = [tc.etiqueta(slug, rj, a) for a in AVS]
        if all(e in KA.por_etiqueta for e in ets):
            out[(rj, False)] = dict(zip(AVS, KA.exige(ets)))
    ets = [tc.etiqueta(slug, "atmo_ceq", a, r250=True) for a in AVS]
    if all(e in KA.por_etiqueta for e in ets):
        out[("atmo_ceq", True)] = dict(zip(AVS, KA.exige(ets)))
    return out


def conjuntos(b7, atmo=None):
    """{nombre: dict(claves=[(rejilla, r250), ...], fsed_separados, evidencia)} de un compañero (ver el docstring
    del módulo). `b7`, `atmo`: los de `cadenas_b7` y `cadenas_atmo`."""
    r500 = [k for k in b7 if not k[1]]
    r250 = [k for k in b7 if k[1]]
    out = dict(r500=dict(claves=r500, fsed_separados=False, evidencia=True),
               r250=dict(claves=r250, fsed_separados=False, evidencia=True))
    if any(k[0].startswith("diamondback") for k in r500):
        out["r500_fsed"] = dict(claves=r500, fsed_separados=True, evidencia=True)
    sph = [k for k in r250 if k[0] == "sphinx"]
    if sph:
        out["siete"] = dict(claves=r500 + sph, fsed_separados=False, evidencia=False)
    if atmo:
        a500 = [k for k in atmo if not k[1]]
        a250 = [k for k in atmo if k[1]]
        if a500:
            out["r500_atmo"] = dict(claves=r500 + a500, fsed_separados=False, evidencia=True)
        if a250:
            out["r250_atmo"] = dict(claves=r250 + a250, fsed_separados=False, evidencia=True)
    return out


def nombre_rejilla(g):
    """El nombre de una rejilla en el informe (el de la Tabla B.7 sin LaTeX)."""
    if g == "diamondback":
        return "Diamondback"
    if g == "atmo":
        return "ATMO 2020"
    n = tb.REJILLAS_TABLA.get(g, g)
    return n.replace(r"\(f_{\rm sed} = ", "f_sed = ").replace(r"\)", "")


# ================================================================== residuos por banda
def residuos_por_banda(mod, th, bandas=BANDAS_A, brazo="vis"):
    """En el segmento de `brazo` del modelo `mod` (`teff_bayes.Modelo`) y el punto `th`: el residuo normalizado con
    la σ del ajuste, σ² = e² + (f_b·m)², y su reparto por bandas (bordes `bandas` en Å, [lo, hi))."""
    k = next(i for i, s in enumerate(mod.segs) if s["banda"] == brazo)
    s = mod.segs[k]
    M = mod.modelos(th)
    m = M[k]
    fb = math.exp(th[mod.pos[f"lnf_{s['banda']}"]])
    var = s["e2"] + (fb * m) ** 2
    r = (s["d"] - m) / np.sqrt(var)
    chi2, chi2_dato = r ** 2, (s["d"] - m) ** 2 / s["e2"]
    rel = (s["d"] - m) / m
    # χ² de todos los segmentos con su σ (la parte de MUSE en el total)
    tot = 0.0
    for ss, mm in zip(mod.segs, M):
        f = math.exp(th[mod.pos[f"lnf_{ss['banda']}"]])
        tot += float(np.sum((ss["d"] - mm) ** 2 / (ss["e2"] + (f * mm) ** 2)))
    out = dict(segmento=s["nombre"], f_b=fb, n=int(r.size), chi2=float(chi2.sum()), chi2_dato=float(chi2_dato.sum()),
               chi2_todos_los_segmentos=tot, fraccion_chi2_del_total=float(chi2.sum()) / tot,
               w_min=float(s["w"].min()), w_max=float(s["w"].max()), bandas=[])
    for lo, hi in zip(bandas[:-1], bandas[1:]):
        sel = (s["w"] >= lo) & (s["w"] < hi)
        n = int(sel.sum())
        b = dict(lo=float(lo), hi=float(hi), n=n, fraccion_puntos=n / r.size)
        if n:
            b.update(w_datos=[float(s["w"][sel].min()), float(s["w"][sel].max())],
                     r_media=float(r[sel].mean()), r_rms=float(np.sqrt(np.mean(r[sel] ** 2))),
                     chi2=float(chi2[sel].sum()), chi2_n=float(chi2[sel].mean()),
                     fraccion_chi2=float(chi2[sel].sum() / chi2.sum()),
                     chi2_dato=float(chi2_dato[sel].sum()), fraccion_chi2_dato=float(chi2_dato[sel].sum() / chi2_dato.sum()),
                     rel_media=float(rel[sel].mean()), rel_rms=float(np.sqrt(np.mean(rel[sel] ** 2))))
        out["bandas"].append(b)
    return out


def modelo_de(d, dir_rejilla=None):
    """`teff_bayes.Modelo` de una cadena (`libre`, los datos, la rejilla, la resolución y la A_V con las que corrió)
    y su punto de máxima verosimilitud: `teff_bayes.modelo_de_cadena`, el único reconstructor (hasta el 2026-10-29
    había aquí una copia que no pasaba las ventanas de las líneas). `dir_rejilla`: el directorio de npz de la
    rejilla, si no es el de `teff_bayes.REJILLAS` (ATMO 2020 se convierte a la ruta que se quiera)."""
    return tb.modelo_de_cadena(d, dir_rejilla)


def residuos_cadena(d, bandas=BANDAS_A, dir_rejilla=None):
    """`residuos_por_banda` del mejor punto de una cadena, con la comprobación de que el modelo reconstruido da el
    `logl_max` de la cadena."""
    t0 = time.time()
    mod, th = modelo_de(d, dir_rejilla)
    ll = mod.loglike(th)
    r = residuos_por_banda(mod, th, bandas)
    r.update(etiqueta=d["etiqueta"], rejilla=d.get("rejilla", "btsettl"), r_analisis=d.get("r_analisis", 500),
             loglike=ll, logl_max=d["logl_max"], reproduce=bool(abs(ll - d["logl_max"]) <= 1e-6 * max(1.0, abs(ll))),
             teff=d["posterior"]["teff"]["p50"], av=d["posterior"]["av"]["p50"], segundos=round(time.time() - t0, 1))
    return r


def resumen_bandas(filas):
    """Por banda, entre las rejillas de `filas` (`residuos_cadena`): rango de la fracción de χ², del χ²/n, del
    residuo relativo medio y de su rms."""
    out = []
    for i, b in enumerate(filas[0]["bandas"]):
        bs = [f["bandas"][i] for f in filas if f["bandas"][i]["n"]]
        if not bs:
            out.append(dict(lo=b["lo"], hi=b["hi"], n=0))
            continue
        rng = lambda k: [min(x[k] for x in bs), max(x[k] for x in bs)]  # noqa: E731
        out.append(dict(lo=b["lo"], hi=b["hi"], n=b["n"], fraccion_puntos=b["fraccion_puntos"],
                        fraccion_chi2=rng("fraccion_chi2"), chi2_n=rng("chi2_n"), rel_media=rng("rel_media"),
                        rel_rms=rng("rel_rms"), r_media=rng("r_media"), fraccion_chi2_dato=rng("fraccion_chi2_dato")))
    return out


# ================================================================== el cálculo
def calcula(K, root, KA=None, tex=None, residuos=False, bandas=BANDAS_A, nivel_manara=False, dirs_rejilla=None,
            filas=False, E=None):
    """Todo, en un dict (lo que va al json). `dirs_rejilla`: {rejilla: directorio de npz} para las que no están en
    la ruta de `teff_bayes.REJILLAS` (las de ATMO 2020); una rejilla sin npz se omite en los residuos y se dice.
    `E`: {slug: `teff_bayes.entrada_margen`}; por defecto se lee de los runs del paper (las pruebas pasan uno)."""
    dirs_rejilla = dict(dirs_rejilla or {})
    root = Path(root)
    tc.fija_objetos(root)
    slugs = tc.SLUGS
    if E is None:
        E = {s: tb.entrada_margen(s, root=root) for s in slugs}
    b7 = {s: cadenas_b7(K, s) for s in slugs}
    atmo = {s: cadenas_atmo(KA, s) for s in slugs} if KA is not None else {s: {} for s in slugs}
    todas = {}
    for s in slugs:
        for fuente, cad in ((K, b7[s]), (KA, atmo[s])):
            for k, porav in cad.items():
                for a, d in porav.items():
                    todas[d["etiqueta"]] = (fuente, d)
    deltas = {e: fuente.delta(d, E) for e, (fuente, d) in todas.items()}
    lnz = {e: d["logz"] for e, (_, d) in todas.items()}
    res = dict(objetos=dict(slugs=list(slugs), nombres=dict(tc.NOMBRES)),
               entrada_margen={s: {k: v for k, v in E[s].items()} for s in slugs})

    # 1. la Tabla B.7 desde las cadenas
    corridas = [d for s in slugs for porav in b7[s].values() for d in porav.values()]
    pct = {d["etiqueta"]: tb.percentiles(deltas[d["etiqueta"]]) for d in corridas}
    filas_b7 = tb.filas_rejillas(corridas, pct, dict(tc.NOMBRES))
    semi = [(p["p84"] - p["p16"]) / 2 for p in pct.values()]
    res["tabla_b7"] = dict(filas=filas_b7, delta_por_cadena={e: dict(p16=p["p16"], p50=p["p50"], p84=p["p84"])
                                                          for e, p in pct.items()},
                           # el «±0.02–0.08 dex» de la nota de la Tabla B.7: el semiancho 16–84 % de cada fila
                           semiancho_filas=[min(semi), max(semi)] if semi else None)
    if tex is not None:
        res["tabla_b7"]["comparacion"] = compara_tabla(filas_b7, filas_tabla_tex(tex))

    # 2. (a)–(e) por compañero, A_V y conjunto de rejillas
    res["conjuntos"] = {}
    for s in slugs:
        cj = conjuntos(b7[s], atmo[s])
        porav = {**b7[s], **atmo[s]}
        for nombre, c in cj.items():
            if not c["claves"]:
                continue
            for a in AVS:
                grupos = {}
                for rj, r in c["claves"]:
                    g = rj if c["fsed_separados"] else familia(rj)
                    grupos.setdefault(g, []).append(porav[(rj, r)][a]["etiqueta"])
                st = estadisticas(grupos, deltas, lnz if c["evidencia"] else None)
                res["conjuntos"][f"{s}|{nombre}|{a}"] = dict(slug=s, conjunto=nombre, av=a, **st)

    res["variantes"] = diferencias_variantes(res["conjuntos"])

    # 3. residuos por banda del ajuste con el prior en A_V
    if residuos:
        res["residuos"] = {}
        for s in slugs:
            for (rj, r), porav in {**b7[s], **atmo[s]}.items():
                d = porav["prior"]
                dr = dirs_rejilla.get(rj)
                if not any(Path(dr or tb.REJILLAS[rj]["dir"]).glob("teff*_logg*.npz")):
                    res.setdefault("residuos_omitidos", {})[d["etiqueta"]] = (
                        f"no hay npz de {rj} en {dr or tb.REJILLAS[rj]['dir']} (pásalo con --dir-rejilla {rj}=DIR)")
                    print(f"  residuos {d['etiqueta']}: omitidos, sin la rejilla", file=sys.stderr)
                    continue
                fila = residuos_cadena(d, bandas, dr)
                res["residuos"][d["etiqueta"]] = dict(fila, slug=s, prueba=rj in PRUEBAS)
                print(f"  residuos {d['etiqueta']}: {fila['segundos']} s, logL {fila['loglike']:.3f} "
                      f"(cadena {fila['logl_max']:.3f})", file=sys.stderr)
        res["residuos_resumen"] = {}
        for s in slugs:
            for r in (500, 250):
                filas_r = [v for v in res["residuos"].values() if v["slug"] == s and v["r_analisis"] == r
                           and not v["prueba"]]
                if filas_r:
                    res["residuos_resumen"][f"{s}|{r}"] = dict(rejillas=[v["etiqueta"] for v in filas_r],
                                                               bandas=resumen_bandas(filas_r),
                                                               fraccion_chi2_del_total=[
                                                                   min(v["fraccion_chi2_del_total"] for v in filas_r),
                                                                   max(v["fraccion_chi2_del_total"] for v in filas_r)],
                                                               f_b=[min(v["f_b"] for v in filas_r),
                                                                    max(v["f_b"] for v in filas_r)])

    # comparación: la dispersión del nivel entre las Class III de Manara et al. (2013)
    if nivel_manara:
        from musepipe.config import load_run_config
        from musepipe.stages.stage_h03_limits import physical_inputs_from_config
        from scripts import halpha_criterios as hc
        from scripts import tipo_ir_indices as tir
        fis = physical_inputs_from_config(load_run_config(tir.perfil(slugs[0], root=root)["paper"]["run"],
                                                          project_root=root).config)
        pob, _ = hc.poblacion_manara(float(fis["lacc_lha_a"]), float(fis["lacc_lha_b"]), np.random.default_rng(0),
                                     n_boot=1)
        res["nivel_manara"] = dict(media=pob["ha"]["media"], sd=pob["ha"]["sd"], n=pob["ha"]["n"],
                                   coeficientes=pob["coeficientes_ha"])

    # ¿es del compañero el flujo de cada banda? La banda de filas (`banda_de_filas.py`) en las bandas de los residuos
    if filas:
        res["filas"] = banda_de_filas_en_bandas(slugs, root, bandas)
    return res


def banda_de_filas_en_bandas(slugs, root, bandas=BANDAS_A):
    """`banda_de_filas.lee_run` (fotometría de caja con anillo en el cubo con la primaria restada por exposición) en
    las bandas de los residuos, en el run del paper de cada compañero: el compañero, seis posiciones vacías de su
    fila y doce de otras filas. Si su fila sin compañero da tanto como el compañero, el flujo de esa banda no es
    (solo) suyo: el residuo del ajuste ahí no es (solo) del modelo."""
    from scripts import banda_de_filas as bf
    from scripts import tipo_ir_indices as tir
    nb = {f"{lo:.0f}-{hi:.0f}": (float(lo), float(hi)) for lo, hi in zip(bandas[:-1], bandas[1:])}
    out = {}
    for s in slugs:
        r = bf.lee_run(tir.perfil(s, root=root)["paper"]["run"], Path(root), bandas=nb)
        out[s] = dict(run=r["run_id"], companero_yx=r["companero_yx"], bandas={
            k: dict(lo=b["lo_A"], hi=b["hi_A"], companero=b["companero"],
                    misma_fila=dict(media=b["misma_fila"]["media"], sd=b["misma_fila"]["sd"], n=b["misma_fila"]["n"],
                                    media_pct=b["misma_fila"]["media_pct"], sd_pct=b["misma_fila"]["sd_pct"]),
                    otras_filas=dict(media=b["otras_filas"]["media"], sd=b["otras_filas"]["sd"], n=b["otras_filas"]["n"],
                                     media_pct=b["otras_filas"]["media_pct"], sd_pct=b["otras_filas"]["sd_pct"]))
            for k, b in r["bandas"].items()})
    return out


# ================================================================== informe
def _f(x, nd=2, signo=True):
    return "—" if x is None else (f"{x:+.{nd}f}" if signo else f"{x:.{nd}f}")


def informe_md(res):
    L = ["# Δ entre rejillas de atmósferas", ""]
    t = res["tabla_b7"]
    comp = t.get("comparacion")
    if comp is None:
        L.append("- Tabla B.7: no comparada (sin `--tex`)")
    else:
        L.append(f"- Tabla B.7 desde las cadenas: {comp['n_nuestras']} filas; impresas {comp['n_impresas']}; "
                 + ("**iguales fila a fila**" if comp["iguales"] else
                    f"**{len(comp['distintas'])} distintas** (Δ distintas: {len(comp['delta_distintas'])})"))
        for i, a, b in comp["distintas"]:
            L += [f"  - fila {i}:", f"    - nuestra: `{a}`", f"    - impresa: `{b}`"]
    L += ["", "## (a)–(e) por compañero, A_V y conjunto", "",
          "(a) mediana [mín, máx] de las medianas de Δ por rejilla; (b) desviación típica entre rejillas (ddof 1); "
          "(c) mezcla a pesos iguales: mediana [16, 84] y P(Δ>0); (d) mezcla por evidencia; (e) anchura típica ⊕ (b) "
          "frente al semiancho de (c). Para el segundo compañero Δ es la de la cota superior.", "",
          "| compañero | conjunto | A_V | N | (a) | (b) | (c) | P(Δ>0) (c) | (d) | P(Δ>0) (d) | domina (d) | (e) típica ⊕ (b) = total | semiancho (c) |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for k, v in res["conjuntos"].items():
        a, b, c, d, e = v["a"], v["b"], v["c"], v["d"], v["e"]
        dtxt = f"{_f(d['p50'])} [{_f(d['p16'])}, {_f(d['p84'])}]" if d else "—"
        dp = f"{d['p_positivo']:.3f}" if d else "—"
        dom = f"{nombre_rejilla(d['rejilla_dominante'])} {d['peso_dominante']:.3f}" if d else "—"
        L.append(f"| {v['slug']} | {v['conjunto']} | {v['av']} | {v['n_rejillas']} | {_f(a['mediana'])} "
                 f"[{_f(a['min'])}, {_f(a['max'])}] | {_f(b['sd'], 3, False)} | {_f(c['p50'])} [{_f(c['p16'])}, "
                 f"{_f(c['p84'])}] | {c['p_positivo']:.3f} | {dtxt} | {dp} | {dom} | "
                 f"{_f(e['anchura_tipica'], 3, False)} ⊕ {_f(e['dispersion'], 3, False)} = {_f(e['total'], 3, False)} | "
                 f"{_f(e['semiancho_mezcla'], 3, False)} |")
    L += ["", "### Mediana de Δ por rejilla (conjunto r500 y r250)", "",
          "| compañero | conjunto | A_V | rejilla | p50 | 16–84 % | P(Δ>0) |", "|---|---|---|---|---|---|---|"]
    for k, v in res["conjuntos"].items():
        if v["conjunto"] not in ("r500", "r250"):
            continue
        for g, p in sorted(v["por_rejilla"].items(), key=lambda x: x[1]["p50"]):
            L.append(f"| {v['slug']} | {v['conjunto']} | {v['av']} | {nombre_rejilla(g)} | {_f(p['p50'], 3)} | "
                     f"[{_f(p['p16'], 3)}, {_f(p['p84'], 3)}] | {p['p_positivo']:.3f} |")
    if "residuos" in res:
        L += ["", "## Residuos por banda en MUSE (ajuste con el prior en A_V, mejor punto)", "",
              "σ² = e² + (f_b·m)². r: (dato − modelo)/σ; rel: (dato − modelo)/modelo.", "",
              "| cadena | reproduce logL | f_b | χ² MUSE / total | banda (Å) | n | frac. χ² | χ²/n | r medio | rel medio | rel rms |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
        for e, v in res["residuos"].items():
            for i, b in enumerate(v["bandas"]):
                cab = (f"| {e} | {'sí' if v['reproduce'] else '**NO**'} | {v['f_b']:.3f} | "
                       f"{v['fraccion_chi2_del_total']:.3f} ") if i == 0 else "| | | | "
                if not b["n"]:
                    L.append(cab + f"| {b['lo']:.0f}–{b['hi']:.0f} | 0 | | | | | |")
                    continue
                L.append(cab + f"| {b['lo']:.0f}–{b['hi']:.0f} | {b['n']} | {b['fraccion_chi2']:.3f} | "
                         f"{b['chi2_n']:.2f} | {b['r_media']:+.2f} | {100 * b['rel_media']:+.1f} % | "
                         f"{100 * b['rel_rms']:.1f} % |")
        L += ["", "### Resumen por banda (rango entre rejillas, sin ATMO)", "",
              "| compañero | R | banda (Å) | n | frac. puntos | frac. χ² | χ²/n | rel medio | rel rms |",
              "|---|---|---|---|---|---|---|---|---|"]
        for k, v in res["residuos_resumen"].items():
            s, r = k.split("|")
            for b in v["bandas"]:
                if not b["n"]:
                    continue
                L.append(f"| {s} | {r} | {b['lo']:.0f}–{b['hi']:.0f} | {b['n']} | {b['fraccion_puntos']:.3f} | "
                         f"{b['fraccion_chi2'][0]:.3f}–{b['fraccion_chi2'][1]:.3f} | "
                         f"{b['chi2_n'][0]:.2f}–{b['chi2_n'][1]:.2f} | "
                         f"{100 * b['rel_media'][0]:+.1f} – {100 * b['rel_media'][1]:+.1f} % | "
                         f"{100 * b['rel_rms'][0]:.1f}–{100 * b['rel_rms'][1]:.1f} % |")
    if "filas" in res:
        L += ["", "## La banda de filas en las mismas bandas (`banda_de_filas.py`, cubo con la primaria restada)", "",
              "| compañero | banda (Å) | compañero | su fila sin compañero (% del compañero) | otras filas (%) |",
              "|---|---|---|---|---|"]
        for s, v in res["filas"].items():
            for k, b in v["bandas"].items():
                m, o = b["misma_fila"], b["otras_filas"]
                L.append(f"| {s} | {k} | {b['companero']:.1f} | {m['media']:.1f} ± {m['sd']:.1f} "
                         f"({m['media_pct']:.0f} ± {m['sd_pct']:.0f} %, n = {m['n']}) | {o['media']:.1f} ± {o['sd']:.1f} "
                         f"({o['media_pct']:.0f} ± {o['sd_pct']:.0f} %, n = {o['n']}) |")
    if "nivel_manara" in res:
        n = res["nivel_manara"]
        L += ["", f"Nivel de Manara et al. (2013): las {n['n']} Class III quedan {n['media']:+.3f} dex sobre él, con "
                  f"una dispersión de {n['sd']:.3f} dex."]
    return "\n".join(L) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cadenas", type=Path, action="append", required=True,
                    help="directorio de cadenas de `teff_bayes.py ajusta` (repetible; gana el primero)")
    ap.add_argument("--cadenas-atmo", type=Path, action="append", default=None,
                    help="directorio de las cadenas de ATMO 2020 (variantes `r500_atmo` y `r250_atmo`)")
    ap.add_argument("--project-root", type=Path, default=ROOT,
                    help="raíz que contiene runs/ y las rejillas (por defecto la del repo; necesaria desde un worktree)")
    ap.add_argument("--tex", type=Path, help="main.tex del paper: compara la Tabla B.7 fila a fila")
    ap.add_argument("--residuos", action="store_true", help="residuos por banda del espectro MUSE")
    ap.add_argument("--bandas", type=float, nargs="+", default=list(BANDAS_A), help="bordes de las bandas (Å)")
    ap.add_argument("--dir-rejilla", action="append", default=[], metavar="REJILLA=DIR",
                    help="directorio de npz de una rejilla que no está en su ruta de teff_bayes.REJILLAS (ATMO 2020)")
    ap.add_argument("--nivel-manara", action="store_true",
                    help="la dispersión del nivel entre las Class III de Manara et al. (2013), para comparar")
    ap.add_argument("--filas", action="store_true",
                    help="la banda de filas (`banda_de_filas.py`) en las bandas de los residuos, en los runs del paper")
    ap.add_argument("--json", type=Path, help="escribe todo en json")
    ap.add_argument("--md", type=Path, help="escribe el informe en markdown (si no, a la salida estándar)")
    a = ap.parse_args(argv)
    warnings.simplefilter("ignore")
    raiz = tb.fija_raiz(a.project_root)
    K = tc.Cadenas(a.cadenas)
    KA = tc.Cadenas(a.cadenas_atmo) if a.cadenas_atmo else None
    tex = a.tex.read_text() if a.tex else None
    dirs = dict(x.split("=", 1) for x in a.dir_rejilla)
    res = calcula(K, raiz, KA=KA, tex=tex, residuos=a.residuos, bandas=tuple(a.bandas), nivel_manara=a.nivel_manara,
                  dirs_rejilla={k: Path(v).expanduser() for k, v in dirs.items()}, filas=a.filas)
    md = informe_md(res)
    if a.md:
        a.md.parent.mkdir(parents=True, exist_ok=True)
        a.md.write_text(md)
        print(f"escrito {a.md}")
    else:
        print(md)
    if a.json:
        a.json.parent.mkdir(parents=True, exist_ok=True)
        a.json.write_text(json.dumps(res, indent=1, ensure_ascii=False, default=str))
        print(f"escrito {a.json}")
    falla = False
    comp = res["tabla_b7"].get("comparacion")
    if comp is not None and not comp["iguales"]:
        print(f"la Tabla B.7 no se reproduce: {len(comp['distintas'])} filas distintas", file=sys.stderr)
        falla = True
    if any(not v["reproduce"] for v in res.get("residuos", {}).values()):
        print("algún modelo reconstruido no da el logl_max de su cadena", file=sys.stderr)
        falla = True
    return 1 if falla else 0


if __name__ == "__main__":
    sys.exit(main())
