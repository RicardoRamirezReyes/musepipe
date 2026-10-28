#!/usr/bin/env python
"""Las cifras del párrafo de ATMO 2020 del apéndice B del paper y de la nota de la Tabla B.5, desde las cadenas de
`scripts/teff_bayes.py`.

Hermano de `paper_cifras_extincion.py`, con la misma regla: lee lo que escribieron las corridas de `teff_bayes.py
ajusta` (json + npz) y no inventa entradas. Lo que imprime es lo que el paper cita de la prueba con ATMO 2020
(Phillips et al. 2020; agente `atmo`, 2026-10-26, `~/muse_sesion_20261026/claudio/atmo/INFORME.md`), que hasta
ahora salía de scripts de la carpeta de sesión (`compara.py`, `tipo_equivalente.py`, `figura.py --cociente`):

- T_eff, log g (y si toca el borde de la rejilla, `teff_bayes.en_el_borde`), la masa g R²/G, ln Z menos el de
  BT-Settl con la misma A_V y Δ (el margen sobre el suelo cromosférico, `teff_bayes.margen_de_corrida`, con la L_acc
  ACTUAL de los runs del paper) de ATMO a A_V 1.8, 3.0 y con el prior, y la diferencia de T_eff con BT-Settl;
- Δ de ROXs 12 b a A_V 1.8 en el modo R = 250;
- el tipo equivalente con las plantillas ajustadas con ATMO, y cuánto cambia la misma plantilla entre el ruido de
  un compañero y el del otro (con ATMO y con BT-Settl);
- con `--dir-atmo`, el cociente ATMO / BT-Settl al mismo (T_eff, log g), los dos por el camino del ajuste
  (R = 500 en los píxeles de ROXs 42B b, sin enrojecer): cuánto más brilla ATMO en el espectro de MUSE;
- la nota de la Tabla B.5: las T_eff y Δ evolutivas (sin espectro, `--modo evo_solo`) con los tracks ATMO2020 y
  BHAC15, y la fracción de muestras que caen fuera de BHAC15.

Qué rejilla de ATMO: la de 2000–3000 K (`atmo_ceq`), salvo que su ajuste con el prior toque el borde de T_eff y
esté la ampliada (`atmo_ceq1500`): entonces la ampliada para las tres A_V del objeto (42B b; las de A_V fija dan lo
mismo con las dos, INFORME §4.2).

El tipo equivalente se da con dos reglas explícitas (el paper no tiene código para él): por la MEDIANA (el tipo
más tardío de las plantillas más calientes que el compañero y el más temprano de las más frías; la que reproduce
el «M7.5–M8» y el «≥L1» de BT-Settl en la Tabla B.5) y por el 16–84 % (lo mismo con las plantillas fuera del
intervalo del compañero). «≥L1» es la nota de la Tabla B.5: tan frío como la plantilla L1 más fría, o más.

`--cadenas` se repite: un directorio por grupo de corridas (las de ATMO con R = 500, la ampliada, R = 250, las
plantillas; las de BT-Settl c0 y sus plantillas; las `evo_solo`). Cada corrida se reconoce por su json, no por el
directorio; dos corridas con la misma configuración en los directorios dados son un error.

Ningún nombre de objeto en el código (`tests/test_no_hardcoded_target.py`): los objetos salen de las corridas, y
el del cociente se pasa (`--objeto-cociente`; el paper usa los píxeles de ROXs 42B b). La columna «redondeado como
en el paper» aplica el redondeo del texto (T_eff a 10 K, ln Z a la unidad, Δ a 0.01 dex); el texto en sí no se lee.

uso: python scripts/paper_cifras_atmo.py --cadenas DIR [--cadenas DIR ...]
                                          [--dir-atmo NPZ_ATMO_CEQ --objeto-cociente SLUG]
                                          [--json SALIDA] [--project-root RAIZ]
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

from musepipe.constants import spt_code, spt_label  # noqa: E402
from scripts import teff_bayes as tb  # noqa: E402

#: GM de Júpiter nominal (IAU 2015, Resolución B3), en cm³ s⁻²: M/M_Jup = g R² / GM_J
GM_JUP_CGS = 1.2668653e23
#: las tres corridas de cada rejilla: A_V fija en 1.8 y 3.0 mag, y con el prior de la primaria (None)
AVS = (1.8, 3.0, None)
#: nodos (T_eff, log g) del cociente ATMO / BT-Settl (los del agente `atmo`, que usó los píxeles de 42B b)
NODOS_COCIENTE = ((2200.0, 3.5), (2300.0, 4.0), (2600.0, 4.5), (2800.0, 4.5))
#: λ (µm) a la que el paper da la resolución del muestreo de ATMO
LAMBDA_K_UM = 2.2


# ------------------------------------------------------------------ las corridas
def _clave(d):
    """La configuración de una corrida: lo que la distingue de las demás."""
    v = d.get("verdad") or {}
    return (d["slug"], d["modo"], d.get("tracks"), d.get("rejilla", "btsettl"), d.get("r_analisis", 500),
            d.get("datos", "todo"), d.get("av"), d.get("av_fija"), d.get("degrada"), v.get("plantilla"),
            "inyeccion" if "teff" in v else None)


def indice(corridas):
    """{configuración: corrida}; dos corridas con la misma configuración son un error (otra semilla, otra copia)."""
    out = {}
    for d in corridas:
        k = _clave(d)
        if k in out:
            raise SystemExit(f"dos corridas con la misma configuración {k}: {out[k]['etiqueta']} y {d['etiqueta']}")
        out[k] = d
    return out


def busca(idx, slug, rejilla, av_fija=None, r=500, plantilla=None):
    """La corrida `libre` con todos los espectros y el prior de la primaria (o la A_V fija) de una rejilla, o None.
    Las rejillas muestreadas en número de onda, con el degradado por defecto (`cuadratura`)."""
    degrada = "cuadratura" if tb.REJILLAS[rejilla].get("nu_bin_cm") else None
    return idx.get((slug, "libre", None, rejilla, r, "todo", "primaria", av_fija, degrada, plantilla, None))


def busca_evo(idx, slug, tracks):
    """La corrida `evo_solo` (sin espectro) con unos tracks, o None."""
    return idx.get((slug, "evo_solo", tracks, "btsettl", 500, "todo", None, None, None, None, None))


def rejilla_atmo(idx, slug):
    """`atmo_ceq`, salvo que su ajuste con el prior toque el borde de T_eff y esté la ampliada `atmo_ceq1500`."""
    base, amp = busca(idx, slug, "atmo_ceq"), busca(idx, slug, "atmo_ceq1500")
    if amp is not None and (base is None or "teff" in tb.en_el_borde(base)):
        return "atmo_ceq1500"
    return "atmo_ceq"


def masa_mjup(d):
    """`percentiles` de la masa que implican el log g y el R de cada muestra, g R²/G, en M_Jup."""
    with np.load(d["_npz"]) as z:
        der = z["derivadas"]                       # teff, logg, r_cm, masa, lbol
    return tb.percentiles(10 ** der[:, 1] * der[:, 2] ** 2 / GM_JUP_CGS)


def resumen(d, entrada):
    """Lo que el párrafo usa de una corrida `libre` sobre los datos."""
    P = d["posterior"]
    return dict(etiqueta=d["etiqueta"], rejilla=d.get("rejilla", "btsettl"), av_fija=d.get("av_fija"),
                teff=P["teff"], logg=P["logg"], av=P["av"], logz=d["logz"], borde=tb.en_el_borde(d),
                masa_mjup=masa_mjup(d), delta=tb.percentiles(tb.margen_de_corrida(d, entrada)))


# ------------------------------------------------------------------ tipo equivalente
def plantillas_de(idx, slug, rejilla):
    """{plantilla: (tipo, T_eff p50)} de las calibraciones con plantillas de una rejilla y el ruido de un objeto."""
    out = {}
    for k, d in idx.items():
        if k[0] == slug and k[1] == "libre" and k[3] == rejilla and k[9] is not None:
            out[k[9]] = (d["verdad"]["spt"], float(d["posterior"]["teff"]["p50"]))
    return out


def tipo_equivalente(teff, plantillas):
    """El tipo equivalente de un compañero (`teff`: `percentiles` de su T_eff) con unas plantillas ({nombre: (tipo,
    T_eff)}), por dos reglas: `mediana` (frente a la mediana) y `p16_84` (frente al 16–84 %, sin las plantillas que
    caen dentro). En cada una: «≥» el tipo más tardío si el compañero es tan frío como la plantilla más fría de ese
    tipo, o más (la nota de la Tabla B.5); si no, del tipo más tardío entre las plantillas más calientes al más
    temprano entre las más frías."""
    pls = [(spt_code(t), t, T) for t, T in plantillas.values()]
    if not pls:
        return None
    tardio = max(c for c, _, _ in pls)
    t_tardio = min(T for c, _, T in pls if c == tardio)

    def regla(t_caliente, t_fria):
        if t_fria <= t_tardio:
            return "≥" + spt_label(tardio)
        calientes = [c for c, _, T in pls if T > t_caliente]
        frias = [c for c, _, T in pls if T <= t_fria]
        if not calientes:
            return "≤" + spt_label(min(c for c, _, _ in pls))
        if not frias:
            return "≥" + spt_label(tardio)
        a, b = sorted((max(calientes), min(frias)))
        return spt_label(a) if a == b else f"{spt_label(a)}–{spt_label(b)}"
    cercana = min(plantillas.items(), key=lambda kv: abs(kv[1][1] - teff["p50"]))
    return dict(mediana=regla(teff["p50"], teff["p50"]), p16_84=regla(teff["p84"], teff["p16"]),
                cercana=dict(plantilla=cercana[0], tipo=cercana[1][0], teff=cercana[1][1]))


def dispersion_plantillas(a, b):
    """|ΔT_eff| de la misma plantilla ajustada con el ruido de dos objetos ({nombre: (tipo, T_eff)}): mediana,
    máximo (y cuál) y cuántas."""
    comunes = sorted(set(a) & set(b))
    if not comunes:
        return None
    d = {p: abs(a[p][1] - b[p][1]) for p in comunes}
    pmax = max(d, key=d.get)
    return dict(n=len(comunes), mediana=float(np.median(list(d.values()))), maximo=d[pmax], plantilla_maximo=pmax,
                por_plantilla=d)


# ------------------------------------------------------------------ el cociente ATMO / BT-Settl
def cociente(dir_atmo, slug, dir_bt=None, nodos=NODOS_COCIENTE):
    """ATMO / BT-Settl al mismo (T_eff, log g), los dos por el camino del ajuste: segmentos de `slug` (sin agrupar),
    ATMO con `modelos_base_nativa` y BT-Settl con `tipo_ir_plantillas.modelos_base` (R = 500 y media en el píxel),
    sin enrojecer ni escalar. Por nodo y segmento: mediana, mínimo y máximo del cociente, y el λ de los píxeles."""
    from scripts import ajuste_conjunto as aj
    from scripts import tipo_ir_plantillas as tp
    dir_bt = Path(dir_bt) if dir_bt else tb.REJILLAS["btsettl"]["dir"]
    p = tp.preparar(slug, "combinado", av_fija=True, bibliotecas=("manara",), h2o_muse=True)
    segs = aj.segmentos_cgs(slug, p)
    out = {}
    for T, g in nodos:
        (ma,) = aj.modelos_btsettl(dir_atmo, teff=(T, T), logg=(g, g))
        (mb,) = aj.modelos_btsettl(dir_bt, teff=(T, T), logg=(g, g))
        ba = tb.modelos_base_nativa([ma], segs, tb.REJILLAS["atmo_ceq"]["nu_bin_cm"])[0]
        bb = tp.modelos_base([mb], segs)[0]
        fila = {}
        for k, s in enumerate(segs):
            c = np.isfinite(ba[k]) & np.isfinite(bb[k])
            r = ba[k][c] / bb[k][c]
            fila[s["nombre"]] = dict(mediana=float(np.median(r)), min=float(r.min()), max=float(r.max()),
                                     lambda_um=[float(s["w"][c].min() / 1e4), float(s["w"][c].max() / 1e4)])
        out[f"T{int(T)}_g{g}"] = fila
    return out


# ------------------------------------------------------------------ las cifras
def cifras(corridas, dir_atmo=None, objeto_cociente=None):
    """Todas las cifras, en un dict (el json de `--json`). El cociente, solo con `dir_atmo` y `objeto_cociente`."""
    idx = indice(corridas)
    slugs = sorted({k[0] for k in idx})
    entradas = {s: tb.entrada_margen(s) for s in slugs}
    res = dict(resolucion_atmo=dict(dnu_cm=tb.REJILLAS["atmo_ceq"]["nu_bin_cm"], lambda_um=LAMBDA_K_UM,
                                    r=1.0 / (LAMBDA_K_UM * 1e-4 * tb.REJILLAS["atmo_ceq"]["nu_bin_cm"])),
               entradas=entradas, objetos={})
    for slug in slugs:
        if busca(idx, slug, "atmo_ceq") is None and busca(idx, slug, "atmo_ceq1500") is None:
            continue
        rej = rejilla_atmo(idx, slug)
        o = dict(rejilla_atmo=rej, ajustes={})
        for av in AVS:
            a, b = busca(idx, slug, rej, av), busca(idx, slug, "btsettl", av)
            if a is None:
                continue
            fa = resumen(a, entradas[slug])
            fila = dict(atmo=fa)
            if b is not None:
                fb = resumen(b, entradas[slug])
                fila.update(btsettl=fb, dteff_bt_menos_atmo=fb["teff"]["p50"] - fa["teff"]["p50"],
                            dlnz=fa["logz"] - fb["logz"])
            o["ajustes"]["prior" if av is None else f"{av:g}"] = fila
        r250 = busca(idx, slug, "atmo_ceq", 1.8, r=250)
        if r250 is not None:
            o["r250_av1.8"] = resumen(r250, entradas[slug])
        pa, pb = plantillas_de(idx, slug, "atmo_ceq1500") or plantillas_de(idx, slug, "atmo_ceq"), \
            plantillas_de(idx, slug, "btsettl")
        prior_a, prior_b = busca(idx, slug, rej), busca(idx, slug, "btsettl")
        o["tipo_equivalente"] = dict(
            atmo=tipo_equivalente(prior_a["posterior"]["teff"], pa) if prior_a and pa else None,
            btsettl=tipo_equivalente(prior_b["posterior"]["teff"], pb) if prior_b and pb else None)
        o["plantillas"] = dict(atmo=pa, btsettl=pb)
        res["objetos"][slug] = o
    # la misma plantilla con el ruido de un objeto y del otro
    if len(res["objetos"]) == 2:
        s1, s2 = res["objetos"]
        res["dispersion_plantillas"] = {
            g: dispersion_plantillas(res["objetos"][s1]["plantillas"][g], res["objetos"][s2]["plantillas"][g])
            for g in ("atmo", "btsettl")}
    # la nota de la Tabla B.5: tracks evolutivos sin espectro
    evo = {}
    for slug in slugs:
        filas = {}
        for tracks in ("ATMO2020", "BHAC15"):
            d = busca_evo(idx, slug, tracks)
            if d is None:
                continue
            filas[tracks] = dict(teff=d["teff"], sigma=0.5 * (d["teff"]["p84"] - d["teff"]["p16"]),
                                 delta=tb.percentiles(tb.margen_de_corrida(d, entradas[slug])),
                                 fraccion_en_rejilla=d["fraccion_en_rejilla"])
        if filas:
            evo[slug] = filas
    res["evolutivos"] = evo
    if dir_atmo is not None and objeto_cociente is not None:
        res["cociente"] = cociente(dir_atmo, objeto_cociente)
        res["cociente_objeto"] = objeto_cociente
    return res


# ------------------------------------------------------------------ lo que se imprime
def _rango(xs, nd=0, signo=False, redondeo=None):
    """«lo–hi» de unos valores, redondeados como los da el paper (`redondeo`: decimales de `round`)."""
    xs = [round(x, redondeo) if redondeo is not None else x for x in xs]
    f = f"{{:{'+' if signo else ''}.{nd}f}}"
    lo, hi = min(xs), max(xs)
    return f.format(lo) if lo == hi else f"{f.format(lo)} – {f.format(hi)}"


def filas(res):
    """[(cifra, valores de este script, redondeados como en el paper)]: lo que citan el párrafo de ATMO 2020 del
    apéndice B y la nota de la Tabla B.5. Sin nombres de objeto en el código: salen de las corridas."""
    F = []
    R = res["resolucion_atmo"]
    F.append(("λ/Δλ del muestreo de ATMO a 2.2 µm", f"{R['r']:.1f} (Δν = {R['dnu_cm']:g} cm⁻¹)", f"≈ {R['r']:.0f}"))
    for slug, o in res["objetos"].items():
        aj = o["ajustes"]
        avs = [k for k in ("1.8", "3", "prior") if k in aj]
        ets = "A_V " + " / ".join(avs)
        teffs = [aj[k]["atmo"]["teff"]["p50"] for k in avs]
        F.append((f"{slug}: T_eff ATMO, {o['rejilla_atmo']} ({ets})", " / ".join(f"{x:.1f}" for x in teffs),
                  f"{_rango(teffs, redondeo=-1)} K"))
        dts = {k: aj[k]["dteff_bt_menos_atmo"] for k in avs if "dteff_bt_menos_atmo" in aj[k]}
        fijas = [dts[k] for k in ("1.8", "3") if k in dts]
        F.append((f"{slug}: T_eff BT-Settl − ATMO ({ets})", " / ".join(f"{v:.1f}" for v in dts.values()),
                  f"{_rango(list(dts.values()), redondeo=-1)} K; con A_V fija, media {np.mean(fijas):.0f} K"
                  if fijas else f"{_rango(list(dts.values()), redondeo=-1)} K"))
        F.append((f"{slug}: log g ATMO ({ets}; ᵃ en el borde, el inferior es "
                  f"{tb.REJILLAS[o['rejilla_atmo']]['logg'][0]:g})",
                  " / ".join(f"{aj[k]['atmo']['logg']['p50']:.3f}" + ("ᵃ" if "logg" in aj[k]["atmo"]["borde"] else "")
                             for k in avs), ""))
        F.append((f"{slug}: masa g R²/G ATMO ({ets}, M_Jup)",
                  " / ".join(f"{aj[k]['atmo']['masa_mjup']['p50']:.2f}" for k in avs),
                  f"{_rango([aj[k]['atmo']['masa_mjup']['p50'] for k in avs], redondeo=0)} M_Jup"))
        dl = [aj[k]["dlnz"] for k in avs if "dlnz" in aj[k]]
        F.append((f"{slug}: ΔlnZ ATMO − BT-Settl ({ets})", " / ".join(f"{x:+.2f}" for x in dl),
                  _rango(dl, signo=True, redondeo=0)))
        de = [aj[k]["atmo"]["delta"]["p50"] for k in avs]
        F.append((f"{slug}: Δ ATMO ({ets}, dex)", " / ".join(f"{x:+.3f}" for x in de),
                  f"{_rango(de, nd=2, signo=True, redondeo=2)} dex"))
        if "r250_av1.8" in o:
            x = o["r250_av1.8"]["delta"]["p50"]
            F.append((f"{slug}: Δ ATMO a A_V 1.8 y R = 250 (dex)", f"{x:+.3f}", f"{x:+.2f} dex"))
        te = o["tipo_equivalente"]
        for g in ("atmo", "btsettl"):
            if te.get(g):
                c = te[g]["cercana"]
                F.append((f"{slug}: tipo equivalente con {g} (por la mediana / por el 16–84 %)",
                          f"{te[g]['mediana']} / {te[g]['p16_84']} (más cercana: {c['plantilla']} {c['tipo']}, "
                          f"{c['teff']:.0f} K)", f"{te[g]['mediana']} / {te[g]['p16_84']}"))
    for g, d in (res.get("dispersion_plantillas") or {}).items():
        if d:
            F.append((f"la misma plantilla con el ruido de un objeto y del otro, {g}: |ΔT_eff| mediana (máx.)",
                      f"{d['mediana']:.1f} K ({d['maximo']:.0f} K, {d['plantilla_maximo']}; {d['n']} plantillas)",
                      f"{d['mediana']:.0f} K"))
    if "cociente" in res:
        mu = [(n, f["MUSE"]) for n, f in res["cociente"].items() if "MUSE" in f]
        n_max, s_max = max(mu, key=lambda x: x[1]["max"])
        F.append((f"ATMO / BT-Settl en MUSE al mismo (T_eff, log g), píxeles de {res['cociente_objeto']}: máximo "
                  "(nodo); medianas por nodo; λ de los píxeles",
                  f"{s_max['max']:.3f} ({n_max}); " + ", ".join(f"{n} {s['mediana']:.2f}" for n, s in mu)
                  + f"; {s_max['lambda_um'][0]:.3f}–{s_max['lambda_um'][1]:.3f} µm",
                  f"hasta {s_max['max']:.1f} veces, {s_max['lambda_um'][0]:.2f}–{s_max['lambda_um'][1]:.2f} µm"))
    evo = res.get("evolutivos") or {}
    dts, dds, fuera = [], [], {}
    for slug, f in evo.items():
        for tr, x in f.items():
            F.append((f"{slug}: T_eff evolutiva sin espectro, {tr} (±σ), Δ y fracción en la rejilla",
                      f"{x['teff']['p50']:.1f} ± {x['sigma']:.1f} K; Δ {x['delta']['p50']:+.4f}; "
                      f"{x['fraccion_en_rejilla']:.5f}", f"{x['teff']['p50']:.0f} ± {x['sigma']:.0f} K; "
                      f"Δ {x['delta']['p50']:+.2f}"))
        if {"ATMO2020", "BHAC15"} <= set(f):
            a, b = f["ATMO2020"], f["BHAC15"]
            dts.append(abs(a["teff"]["p50"] - b["teff"]["p50"]))
            dds.append((abs(a["delta"]["p50"] - b["delta"]["p50"]),
                        abs(round(a["delta"]["p50"], 2) - round(b["delta"]["p50"], 2))))
            fuera[slug] = 1.0 - b["fraccion_en_rejilla"]
    if dts:
        F.append(("BHAC15 frente a ATMO2020, sin espectro: máx. |ΔT_eff|; máx. |ΔΔ| (con los Δ a dos decimales, "
                  "como la Tabla B.5); muestras fuera de BHAC15",
                  f"{max(dts):.1f} K; {max(x for x, _ in dds):.4f} dex ({max(y for _, y in dds):.2f}); "
                  + ", ".join(f"{s} {100 * v:.2f} %" for s, v in fuera.items()),
                  f"{max(dts):.0f} K; {max(y for _, y in dds):.2f} dex; "
                  + ", ".join(f"{s} {100 * v:.0f} %" for s, v in fuera.items())))
    return F


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cadenas", type=Path, action="append", required=True,
                    help="directorio de corridas de `teff_bayes.py ajusta` (json + npz); se repite")
    ap.add_argument("--dir-atmo", type=Path, default=None,
                    help="npz de la rejilla ATMO 2020 de 2000–3000 K (`convierte_atmo.py atmo`), para el cociente")
    ap.add_argument("--objeto-cociente", default=None,
                    help="slug (targets/) cuyos píxeles usa el cociente; el del paper es el de 42B b")
    ap.add_argument("--json", type=Path, default=None, help="escribe aquí todas las cifras")
    ap.add_argument("--project-root", type=Path, default=None,
                    help="clon del que se leen los runs y las bibliotecas (por defecto, este; un worktree no los lleva)")
    a = ap.parse_args(argv)
    if (a.dir_atmo is None) != (a.objeto_cociente is None):
        ap.error("--dir-atmo y --objeto-cociente van juntos")
    if a.project_root is not None:
        tb.fija_raiz(a.project_root)
    for d in a.cadenas:
        if not d.is_dir():
            raise SystemExit(f"no existe {d}")
    res = cifras(tb._carga(a.cadenas), a.dir_atmo, a.objeto_cociente)
    print("| cifra | este script | redondeado como en el paper |")
    print("|---|---|---|")
    for c, v, t in filas(res):
        print(f"| {c} | {v} | {t} |")
    if a.json is not None:
        a.json.write_text(json.dumps(res, indent=1, ensure_ascii=False, default=float))
    return 0


if __name__ == "__main__":
    sys.exit(main())
