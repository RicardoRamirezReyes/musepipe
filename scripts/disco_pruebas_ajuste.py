#!/usr/bin/env python
"""Las pruebas del disco de ROXs 12 b en los ajustes en flujo absoluto: r_K = 0, T_d libre e inyecciones de polvo.

Solo lectura y fuera de la cadena (apéndice D.11 del paper, «A disk around ROXs 12 b», `app:checks:disk`;
2026-10-30). Lee las cadenas de `teff_bayes.py ajusta` (json + npz) y no lanza ninguna. El D.11 enumera tres pruebas
que no se habían hecho; este script da sus cifras y las de una cuarta que las calibra:

(i) espectros de modelo con un exceso conocido (`--inyecta-bt` + `--inyecta-polvo RK TD`), ajustados con T_d = 1500 K,
    con T_d libre y con r_K = 0: ¿se recuperan r_K, T_d y A_V?;
(ii) los datos ajustados con r_K = 0 (`--rk-fijo 0`) frente al ajuste del paper (r_K libre en 0–1.5): Δln Z;
(iii) los datos con T_d libre (`--td-libre TMIN TMAX`, uniforme), en 300–1500 K como Grasser et al. (2026) y más allá;
(iv) las plantillas empíricas de la Tabla B.5 a la S/N de los datos (`--plantilla`), con los mismos ajustes: lo que dan
    (ii) y (iii) en espectros reales cuando el exceso no es un disco inyectado sino el fallo del modelo.

Cada cadena se clasifica por su json:

- el ajuste: `c0` (r_K libre y T_d = `teff_bayes.T_D`, el modelo del paper), `rk<valor>` (`rk_fijo`) o
  `td<min>-<max>` (`td_rango`);
- lo ajustado (`objeto`): los datos (sin `verdad`), una plantilla (`verdad.plantilla`) o un espectro de modelo
  (`verdad` con T_eff, log g, A_V y radio; con `inyecta_polvo` si lleva exceso);
- la configuración (`CONFIG`: objeto, rejilla, modo, A_V, A_V fija, resolución, datos, nlive y semilla).

De `--cadenas` (las del paper) entran solo las que no son variantes (`teff_bayes._es_variante`); de `--variantes`, las
que no llevan más variantes que las del disco (ni máscaras, ni ventanas, ni `--quita-muse`). Cada ajuste se compara con
el `c0` del mismo objeto y la misma configuración: Δln Z = ln Z − ln Z(c0), con σ = √(σ² + σ²(c0)), y las medianas
(16–84 %) de T_eff, A_V, r_K y T_d. Sobre los datos, también la de Δ (`teff_bayes.margen_de_corrida`, el margen sobre
el nivel cromosférico de la Tabla B.7). Un parámetro está «en el borde» si su p84 (p16) queda a menos del 2 % del
rango de su prior del extremo superior (inferior); T_eff, si su p84 (p16) queda a menos de medio paso del borde de la
rejilla (`ejes` del json). Las plantillas van por tipo espectral (`verdad.spt`), con un resumen por grupo de tipo
(`--corte-spt`): el rango de Δln Z y de r_K de cada ajuste.

Uso (las cadenas del paper y las de las pruebas, en las carpetas de la máquina del autor):

    python scripts/disco_pruebas_ajuste.py --project-root RAIZ_CON_RUNS --cadenas ~/muse_noche_20261024_c4/cadenas \\
        --variantes ~/muse_noche_20261029_disco/lote/res --variantes ~/muse_noche_20261030/lote/res \\
        --variantes ~/muse_noche_20261030/lote_pl3000/res --json salida.json --md salida.md
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import teff_bayes as tb  # noqa: E402
from scripts import teff_cifras_texto as tc  # noqa: E402

#: lo que tiene que coincidir entre una cadena y su `c0` (con el valor por defecto del json)
CONFIG = {"slug": None, "rejilla": "btsettl", "modo": None, "av": None, "av_fija": None, "r_analisis": 500,
          "datos": "todo", "nlive": None, "semilla": None}
#: las claves de las pruebas del disco en el json de `ajusta`
DISCO = ("rk_fijo", "td_rango", "inyecta_polvo")
#: el prior de r_K de `teff_bayes.Modelo` (uniforme en 0–RK_MAX)
RK_MAX = 1.5
#: «en el borde»: a menos de esta fracción del rango del prior
BORDE = 0.02
#: orden de las clases espectrales
CLASES = "MLT"


def clave(d):
    return tuple(d.get(k, v) for k, v in CONFIG.items())


def ajuste(d):
    rk, td = d.get("rk_fijo"), d.get("td_rango")
    if rk is not None:
        return f"rk{rk:g}"
    return f"td{td[0]:g}-{td[1]:g}" if td else "c0"


def objeto(d):
    v = d.get("verdad") or {}
    if not v:
        return ("datos",)
    if "plantilla" in v:
        return ("plantilla", v["plantilla"])
    pol = d.get("inyecta_polvo")
    return ("inyeccion", v["teff"], v["logg"], v["av"], v["r_rj"], None if pol is None else tuple(pol))


def entra(d, del_paper):
    """¿Entra la cadena? Del paper: `libre` y sin variantes. De las pruebas: `libre` y sin otras variantes que las del
    disco (ni claves que este código no conozca)."""
    if d.get("modo") != "libre":
        return False
    if del_paper:
        return not tb._es_variante(d)
    if any(d.get(k, v) != v for k, v in tb.VARIANTES_JSON.items() if k not in DISCO):
        return False
    return not any(not k.startswith("_") and k not in tb.CLAVES_JSON and k not in tb.VARIANTES_JSON for k in d)


def indice(K, V):
    """{(clave, objeto, ajuste): cadena} de las del paper (`K`) y de las pruebas (`V`), `teff_cifras_texto.Cadenas`.
    Dos cadenas con la misma clave son un error: no se elige por la etiqueta."""
    out = {}
    for fuente, del_paper in ((K, True), (V, False)):
        for d in fuente.por_etiqueta.values():
            if not entra(d, del_paper):
                continue
            k = (clave(d), objeto(d), ajuste(d))
            if k in out:
                raise SystemExit(f"dos cadenas para {k}: {out[k]['_json']} y {d['_json']}")
            out[k] = d
    return out


def cuantiles(P, k):
    p = P.get(k)
    return None if p is None else dict(p16=p["p16"], p50=p["p50"], p84=p["p84"])


def borde(q, lo, hi, margen):
    """'alto', 'bajo' o None: si el p84 (p16) queda a menos de `margen` de `hi` (`lo`)."""
    if q is None:
        return None
    if q["p84"] > hi - margen:
        return "alto"
    if q["p16"] < lo + margen:
        return "bajo"
    return None


def fila(d, ref, E=None):
    """Lo que se da de una cadena frente a su `c0` (`ref`, None si no está). `E`: {slug: `entrada_margen`} para Δ."""
    P = d["posterior"]
    t0, dt, nt = d["ejes"]["teff"]
    out = dict(etiqueta=d["etiqueta"], ajuste=ajuste(d), logz=d["logz"], logz_err=d["logz_err"],
               teff=cuantiles(P, "der_teff"), av=cuantiles(P, "av"), rk=cuantiles(P, "rk"), td=cuantiles(P, "td"),
               rk_fijo=d.get("rk_fijo"), td_rango=d.get("td_rango"))
    out["borde"] = {k: b for k, b in (
        ("teff", borde(out["teff"], t0, t0 + dt * (nt - 1), dt / 2)),
        ("rk", borde(out["rk"], 0.0, RK_MAX, BORDE * RK_MAX)),
        ("td", borde(out["td"], *d["td_rango"], BORDE * (d["td_rango"][1] - d["td_rango"][0])) if d.get("td_rango") else None))
        if b}
    out["dlnz"] = out["dlnz_err"] = None
    if ref is not None and ref is not d:
        out["dlnz"] = d["logz"] - ref["logz"]
        out["dlnz_err"] = float(np.hypot(d["logz_err"], ref["logz_err"]))
    out["delta"] = None
    if E is not None and objeto(d) == ("datos",):
        m = tb.margen_de_corrida(d, E[d["slug"]])
        m = m[np.isfinite(m)]
        out["delta"] = dict(zip(("p16", "p50", "p84"), map(float, np.percentile(m, [16, 50, 84]))))
    return out


def orden_ajuste(a):
    return (a != "c0", a.startswith("td"), float(a[2:].split("-")[-1]) if a.startswith("td") else 0.0, a)


def clave_spt(spt):
    return (CLASES.index(spt[0]), float(spt[1:]))


def grupos(idx, E=None):
    """Los objetos con alguna prueba (un ajuste distinto de `c0`), cada uno con sus filas en orden: `c0`, `rk…`, `td…`
    de menor a mayor T_d máxima. `sin_referencia`: los que no tienen `c0`."""
    por = {}
    for (k, o, a), d in idx.items():
        por.setdefault((k, o), {})[a] = d
    out = []
    for (k, o), ajs in por.items():
        if set(ajs) == {"c0"}:
            continue
        ref = ajs.get("c0")
        d0 = next(iter(ajs.values()))
        g = dict(config=dict(zip(CONFIG, k)), objeto=list(o), tipo=o[0], sin_referencia=ref is None,
                 spt=(d0.get("verdad") or {}).get("spt"), lib=(d0.get("verdad") or {}).get("lib"),
                 verdad=d0.get("verdad"), inyecta_polvo=d0.get("inyecta_polvo"),
                 filas=[fila(ajs[a], ref, E) for a in sorted(ajs, key=orden_ajuste)])
        out.append(g)
    return sorted(out, key=lambda g: (g["tipo"], json.dumps(g["objeto"], default=str), json.dumps(g["config"], default=str)))


def resumen_plantillas(gs, corte="M8.5"):
    """{(rejilla, grupo de tipo, ajuste): rango de Δln Z, de r_K (mediana) y de T_d (mediana)} de las plantillas;
    grupo 'antes' = tipo < `corte`, 'desde' = tipo ≥ `corte`."""
    out = {}
    for g in gs:
        if g["tipo"] != "plantilla" or g["sin_referencia"]:
            continue
        lado = "antes" if clave_spt(g["spt"]) < clave_spt(corte) else "desde"
        for f in g["filas"]:
            r = out.setdefault(f"{g['config']['rejilla']}|{lado}|{f['ajuste']}", dict(dlnz=[], rk=[], td=[], spt=[]))
            r["spt"].append(g["spt"])
            if f["dlnz"] is not None:
                r["dlnz"].append(f["dlnz"])
            for k in ("rk", "td"):
                if f[k] is not None:
                    r[k].append(f[k]["p50"])
    return {k: {c: (sorted(v, key=clave_spt) if c == "spt" else [min(v), max(v)] if v else v) for c, v in r.items()}
            for k, r in out.items()}


def calcula(K, V, root, E=None, corte="M8.5"):
    """Todo, en un dict (lo que va al json). `E`: {slug: `teff_bayes.entrada_margen`}; por defecto, de los runs del
    paper (solo los objetos con pruebas sobre los datos)."""
    idx = indice(K, V)
    if E is None:
        slugs = sorted({k[0][0] for k in idx if k[1] == ("datos",) and k[2] != "c0"})
        E = {s: tb.entrada_margen(s, root=Path(root)) for s in slugs}
    gs = grupos(idx, E)
    return dict(grupos=gs, resumen_plantillas=resumen_plantillas(gs, corte), corte_spt=corte)


# ================================================================== informe
def _q(q, nd=0):
    if q is None:
        return "—"
    f = f"{{:.{nd}f}}"
    return f"{f.format(q['p50'])} [{f.format(q['p16'])}, {f.format(q['p84'])}]"


def _dz(f):
    return "—" if f["dlnz"] is None else f"{f['dlnz']:+.1f} ± {f['dlnz_err']:.1f}"


def _nombre_objeto(g):
    if g["tipo"] == "datos":
        return f"{g['config']['slug']} (datos)"
    if g["tipo"] == "plantilla":
        return f"plantilla {g['objeto'][1]} ({g['spt']}, {g['lib']})"
    pol = g["inyecta_polvo"]
    v = g["verdad"]
    return (f"inyección BT-Settl {v['teff']:g} K, log g {v['logg']:g}, A_V {v['av']:g}, {v['r_rj']:g} R_J, "
            + ("sin polvo" if pol is None else f"polvo r_K {pol[0]:g}, T_d {pol[1]:g} K"))


def informe_md(res):
    L = ["# Pruebas del disco en los ajustes (D.11)", ""]
    titulos = {"datos": "Los datos (ii, iii)", "inyeccion": "Espectros de modelo con un exceso conocido (i)",
               "plantilla": "Plantillas empíricas (iv)"}
    for tipo in ("datos", "inyeccion", "plantilla"):
        gs = [g for g in res["grupos"] if g["tipo"] == tipo]
        if tipo == "plantilla":
            gs = sorted(gs, key=lambda g: (g["config"]["rejilla"], clave_spt(g["spt"]), g["objeto"][1]))
        if not gs:
            continue
        L += [f"## {titulos[tipo]}", "",
              "| objeto | rejilla | A_V | ajuste | Δln Z | T_eff (K) | A_V | r_K | T_d (K) | Δ (dex) | borde |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
        for g in gs:
            c = g["config"]
            av = "prior" if c["av_fija"] is None else f"{c['av_fija']:g}"
            for f in g["filas"]:
                rk = _q(f["rk"], 2) if f["rk"] is not None else (f"{f['rk_fijo']:g} (fija)" if f["rk_fijo"] is not None else "—")
                td = _q(f["td"]) if f["td"] is not None else ("—" if f["rk_fijo"] is not None else f"{tb.T_D:g} (fija)")
                L.append(f"| {_nombre_objeto(g)} | {c['rejilla']} | {av} | {f['ajuste']} | {_dz(f)} | {_q(f['teff'])} | "
                         f"{_q(f['av'], 2)} | {rk} | {td} | {_q(f['delta'], 2)} | "
                         f"{', '.join(f'{k} {b}' for k, b in f['borde'].items()) or ''} |")
            if g["sin_referencia"]:
                L.append(f"| {_nombre_objeto(g)} | {c['rejilla']} | {av} | sin `c0`: Δln Z no se calcula | | | | | | | |")
        L.append("")
    if res["resumen_plantillas"]:
        L += [f"## Plantillas por grupo de tipo (corte {res['corte_spt']})", "",
              "| rejilla | tipos | ajuste | n | Δln Z | r_K (mediana) | T_d (mediana, K) |", "|---|---|---|---|---|---|---|"]
        for k in sorted(res["resumen_plantillas"]):
            rejilla, lado, aj = k.split("|")
            r = res["resumen_plantillas"][k]
            rango = (lambda x, fmt: "—" if not x else f"{fmt.format(x[0])} a {fmt.format(x[1])}")
            tipos = f"{'<' if lado == 'antes' else '≥'} {res['corte_spt']} ({r['spt'][0]}–{r['spt'][-1]})"
            L.append(f"| {rejilla} | {tipos} | {aj} | {len(r['spt'])} | {rango(r['dlnz'], '{:+.1f}')} | "
                     f"{rango(r['rk'], '{:.2f}')} | {rango(r['td'], '{:.0f}')} |")
        L.append("")
    return "\n".join(L)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cadenas", type=Path, action="append", required=True,
                    help="directorio de las cadenas del paper (repetible; gana el primero)")
    ap.add_argument("--variantes", type=Path, action="append", required=True,
                    help="directorio de las cadenas de las pruebas del disco (repetible; se recorre entero)")
    ap.add_argument("--project-root", type=Path, default=ROOT,
                    help="raíz que contiene runs/ (por defecto la del repo; necesaria desde un worktree)")
    ap.add_argument("--corte-spt", default="M8.5", help="tipo que separa los dos grupos de plantillas del resumen")
    ap.add_argument("--json", type=Path, help="escribe todo en json")
    ap.add_argument("--md", type=Path, help="escribe el informe en markdown (si no, a la salida estándar)")
    a = ap.parse_args(argv)
    warnings.simplefilter("ignore")
    raiz = tb.fija_raiz(a.project_root)
    res = calcula(tc.Cadenas(a.cadenas), tc.Cadenas(a.variantes), raiz, corte=a.corte_spt)
    md = informe_md(res)
    if a.md:
        a.md.parent.mkdir(parents=True, exist_ok=True)
        a.md.write_text(md)
        print(f"escrito {a.md}")
    else:
        print(md)
    if a.json:
        a.json.parent.mkdir(parents=True, exist_ok=True)
        a.json.write_text(json.dumps(res, indent=1, ensure_ascii=False, default=float))
        print(f"escrito {a.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
