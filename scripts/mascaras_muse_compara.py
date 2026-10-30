#!/usr/bin/env python
"""Las cadenas de T_eff con otras máscaras del espectro MUSE, contra las del paper: la Tabla D.2 y las cifras de D.6.

Solo lectura y fuera de la cadena (apéndice D del paper, «The masks of the MUSE spectrum», 2026-10-29). Lee las
cadenas de `teff_bayes.py ajusta` (json + npz) y no lanza ninguna. Las variantes son las de
`teff_bayes.py --mascara-muse` (`sin_o2b`, `sin_o2`, `g3`, `sin_telurico`) y `--ventana-lineas/--ventana-halpha`,
solas o juntas; el json de cada cadena las declara (`mascara_muse`, `ventana_lineas_kms`, `ventana_halpha_kms`).

Cada cadena de una variante se compara con la cadena de referencia de la misma configuración (objeto, rejilla, modo,
A_V, A_V fija, resolución, datos y semilla) con las máscaras del paper y sin ventanas. Por pareja: el cambio de la
mediana de T_eff, log g y A_V (`posterior`), el de la mediana del margen Δ sobre el nivel cromosférico
(`teff_bayes.margen_de_corrida`, el Δ de la Tabla B.7), el χ²_ν del espectro MUSE con solo el error del dato
(`chi2nu_mejor.MUSE`) y los bins de MUSE (`n_puntos.MUSE`). El ln Z NO se compara: con otras máscaras el ajuste
usa otros datos.

Por variante y objeto: el número de ajustes, los bins, el máximo |cambio| de cada cantidad (el de A_V solo en los
ajustes con el prior, porque con A_V fija no cambia) con la cadena que lo da, y la mediana de |ΔT_eff|. Aparte: los
cambios de signo de Δ, el rango de Δ con el prior, en cuántos ajustes sube el χ²_ν de MUSE y el semiancho 16–84 %
de T_eff de las cadenas de referencia usadas.

Con `--tex`, compara la Tabla D.2 del paper (`tab:masks`) fila a fila con lo calculado, al redondeo impreso (K a
entero, el resto a dos decimales); sale con 1 si no coincide. Las filas del paper van en el orden de `ORDEN`.

Uso (las cadenas del paper y las de las variantes, en las carpetas de la máquina del autor):

    python scripts/mascaras_muse_compara.py --project-root RAIZ_CON_RUNS \\
        --cadenas ~/muse_noche_20261024_c4/cadenas --cadenas ~/muse_sesion_20261026/claudio/atmo/res/atmo1500 \\
        --variantes ~/muse_sesion_20261028/mascaras_muse/lote/res --variantes ~/muse_sesion_20261028/ventanas_lineas/lote/res \\
        --tex paper/main.tex --json salida.json --md salida.md
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import teff_bayes as tb  # noqa: E402

#: las variantes de la Tabla D.2, en el orden de sus filas: (máscara, (ventana de las líneas, ventana de Hα) en km/s).
ORDEN = [("paper", (50.0, 100.0)), ("sin_o2b", None), ("sin_o2b", (50.0, 100.0)), ("g3", None), ("sin_o2", None),
         ("sin_telurico", None)]
#: las cantidades comparadas: clave de `posterior` (o "margen") y decimales de la Tabla D.2.
CANTIDADES = {"dT": ("der_teff", 0), "dg": ("der_logg", 2), "dav": ("av", 2), "dD": ("margen", 2)}
CONFIG = ("slug", "rejilla", "modo", "av", "av_fija", "r_analisis", "datos", "semilla")


def variante_de(d):
    """(máscara, ventanas o None) de una cadena; ("paper", None) es la referencia."""
    vl, vh = d.get("ventana_lineas_kms"), d.get("ventana_halpha_kms")
    ventanas = None if vl is None and vh is None else (None if vl is None else float(vl), None if vh is None else float(vh))
    return d.get("mascara_muse", "paper"), ventanas


def nombre_variante(v):
    m, w = v
    return m + ("" if w is None else f"+ventana{w[0]:g}/{w[1]:g}")


def clave(d):
    return tuple(d.get(k, "btsettl" if k == "rejilla" else None) for k in CONFIG)


def carga(dirs, sombreadas=None):
    """Las cadenas de los directorios (recorridos enteros), por etiqueta; gana la primera y las repetidas van a
    `sombreadas` (lista de (usada, sombreada)) si se pasa."""
    out = {}
    for dd in dirs:
        for raiz, _, ficheros in sorted(os.walk(Path(dd).expanduser(), followlinks=True)):
            for nombre in sorted(ficheros):
                if not nombre.endswith(".json"):
                    continue
                f = Path(raiz) / nombre
                z = f.with_suffix(".npz")
                try:
                    d = json.loads(f.read_text())
                except (OSError, ValueError):
                    continue
                if not isinstance(d, dict) or d.get("etiqueta") != f.stem or "posterior" not in d or not z.exists():
                    continue
                d["_json"], d["_npz"] = str(f), str(z)
                if d["etiqueta"] in out:
                    if sombreadas is not None:
                        sombreadas.append((out[d["etiqueta"]]["_json"], str(f)))
                    continue
                out[d["etiqueta"]] = d
    return out


def resumen_cadena(d, e):
    """Medianas, margen (mediana, 16 y 84 %), semiancho de T_eff, bins y χ²_ν de MUSE de una cadena."""
    P = d["posterior"]
    mg = tb.margen_de_corrida(d, e)
    mg = mg[np.isfinite(mg)]
    t = P["der_teff"]
    return dict(etiqueta=d["etiqueta"], der_teff=t["p50"], der_logg=P["der_logg"]["p50"], av=P["av"]["p50"],
                margen=float(np.median(mg)), margen_p16=float(np.percentile(mg, 16)),
                margen_p84=float(np.percentile(mg, 84)), semiancho_teff=(t["p84"] - t["p16"]) / 2,
                bins=(d.get("n_puntos") or {}).get("MUSE"), chi2nu_muse=(d.get("chi2nu_mejor") or {}).get("MUSE"),
                nlive=d.get("nlive"))


def calcula(referencias, variantes, entradas):
    """`referencias`, `variantes`: {etiqueta: json} (de `carga`); `entradas`: {slug: teff_bayes.entrada_margen}.
    Una referencia no es ninguna variante (`teff_bayes._es_variante`): una cadena de otra prueba (r_K fija, T_d
    libre, sin los extremos de MUSE) da ("paper", None) en `variante_de` y, por su etiqueta, podía ir primero."""
    ref = {}
    for d in referencias.values():
        if variante_de(d) == ("paper", None) and not tb._es_variante(d):
            ref.setdefault(clave(d), d)
    parejas, sin_pareja = [], []
    for d in variantes.values():
        v = variante_de(d)
        if v == ("paper", None):
            continue
        r = ref.get(clave(d))
        if r is None:
            sin_pareja.append(d["etiqueta"])
            continue
        a, b = resumen_cadena(r, entradas[d["slug"]]), resumen_cadena(d, entradas[d["slug"]])
        cambio = {q: b[c] - a[c] for q, (c, _) in CANTIDADES.items()}
        parejas.append(dict(variante=nombre_variante(v), slug=d["slug"], rejilla=clave(d)[1], av_fija=d.get("av_fija"),
                            referencia=a, cadena=b, cambio=cambio, nlive_igual=a["nlive"] == b["nlive"]))
    por = {}
    for p in parejas:
        por.setdefault(p["variante"], {}).setdefault(p["slug"], []).append(p)
    resumen = {}
    for v, objs in por.items():
        for s, ps in objs.items():
            r = dict(n=len(ps), bins=sorted({p["cadena"]["bins"] for p in ps}),
                     mediana_abs_dT=float(np.median([abs(p["cambio"]["dT"]) for p in ps])))
            for q in CANTIDADES:
                cand = [p for p in ps if q != "dav" or p["av_fija"] is None]
                if not cand:
                    r[q] = None
                    continue
                i = int(np.argmax([abs(p["cambio"][q]) for p in cand]))
                r[q] = dict(max_abs=abs(cand[i]["cambio"][q]), valor=cand[i]["cambio"][q],
                            cadena=cand[i]["cadena"]["etiqueta"])
            r["chi2_sube"] = sum(p["cadena"]["chi2nu_muse"] > p["referencia"]["chi2nu_muse"] for p in ps)
            resumen.setdefault(v, {})[s] = r
    signo = [dict(variante=p["variante"], cadena=p["cadena"]["etiqueta"], referencia=p["referencia"]["margen"],
                  variante_margen=p["cadena"]["margen"]) for p in parejas
             if (p["cadena"]["margen"] > 0) != (p["referencia"]["margen"] > 0)]
    con_prior = {}
    for p in parejas:
        if p["av_fija"] is None:
            con_prior.setdefault(f"{p['slug']}:{p['rejilla']}", []).append(p["cadena"]["margen"])
    usadas = {p["referencia"]["etiqueta"]: p["referencia"]["semiancho_teff"] for p in parejas}
    return dict(parejas=parejas, resumen=resumen, sin_pareja=sorted(sin_pareja), cambios_de_signo=signo,
                margen_con_prior={k: [min(v), max(v)] for k, v in con_prior.items()},
                semiancho_teff_referencias=[min(usadas.values()), max(usadas.values())] if usadas else None,
                nlive_distinto=sorted(p["cadena"]["etiqueta"] for p in parejas if not p["nlive_igual"]))


def filas_tabla(res, slugs):
    """Las filas de la Tabla D.2 en el orden de `ORDEN`: [bins..., ajustes..., (cambios de cada objeto)...]
    redondeados como en el paper. Una variante sin cadenas da None."""
    out = []
    for v in ORDEN:
        r = res["resumen"].get(nombre_variante(v))
        if not r or any(s not in r for s in slugs):
            out.append(None)
            continue
        bins = [r[s]["bins"][0] for s in slugs]
        fila = dict(bins=bins if len(set(bins)) > 1 else bins[:1], ajustes=[r[s]["n"] for s in slugs], cambios=[])
        for s in slugs:
            for q, (_, nd) in CANTIDADES.items():
                fila["cambios"].append(round(r[s][q]["max_abs"], nd) if r[s][q] else None)
        out.append(fila)
    return out


def _numeros(celda):
    return [float(x) for x in re.findall(r"[-+]?\d+(?:\.\d+)?", celda.replace("\\pm", " "))]


def compara_tex(tex, filas):
    """Compara las filas de datos de `tab:masks` con `filas` (de `filas_tabla`). Devuelve la lista de diferencias."""
    i = tex.find("\\label{tab:masks}")
    if i < 0:
        return ["no hay \\label{tab:masks} en el tex"]
    partes = tex[i:tex.find("\\end{tabular}", i)].split("\\hline")
    # las filas de datos van entre las dos últimas \hline (la cabecera, entre \hline\hline y la siguiente)
    datos = [l for l in partes[-2].splitlines() if "&" in l] if len(partes) >= 4 else []
    if len(datos) != len(filas):
        return [f"la tabla del tex tiene {len(datos)} filas y aquí {len(filas)}"]
    dif = []
    for k, (linea, f) in enumerate(zip(datos, filas)):
        if f is None:
            dif.append(f"fila {k + 1}: faltan cadenas de {nombre_variante(ORDEN[k])}")
            continue
        celdas = [c.strip() for c in linea.rstrip("\\ \n").split("&")]
        esperado = [f["bins"], f["ajustes"]] + [[x] for x in f["cambios"]]
        for j, (c, e) in enumerate(zip(celdas[1:], esperado)):
            if [round(x, 2) for x in _numeros(c)] != [round(x, 2) for x in e]:
                dif.append(f"fila {k + 1} ({nombre_variante(ORDEN[k])}), columna {j + 2}: tex {c!r}, calculado {e}")
    return dif


def informe_md(res, filas, slugs, dif_tex=None):
    L = ["# Máscaras del espectro MUSE: cadenas con otras máscaras contra las del paper", ""]
    L += ["## Tabla D.2 (máximo |cambio| por objeto: T_eff K, log g dex, A_V mag con prior, Δ dex)", "",
          "| variante | bins | ajustes | " + " | ".join(f"{s} {q}" for s in slugs for q in CANTIDADES) + " |",
          "|---" * (3 + 4 * len(slugs)) + "|"]
    for v, f in zip(ORDEN, filas):
        if f is None:
            L.append(f"| {nombre_variante(v)} | sin cadenas |")
            continue
        L.append(f"| {nombre_variante(v)} | {', '.join(map(str, f['bins']))} | {', '.join(map(str, f['ajustes']))} | "
                 + " | ".join("—" if x is None else f"{x:g}" for x in f["cambios"]) + " |")
    L += ["", "## Cadenas que dan cada máximo", ""]
    for v, objs in res["resumen"].items():
        for s, r in objs.items():
            L.append(f"- {v}, {s}: n = {r['n']}, bins {r['bins']}, mediana |ΔT| {r['mediana_abs_dT']:.1f} K, χ²_ν de "
                     f"MUSE sube en {r['chi2_sube']} de {r['n']}; " + "; ".join(
                         f"{q} {r[q]['valor']:+.4f} ({r[q]['cadena']})" for q in CANTIDADES if r[q]))
    L += ["", "## Pareja por pareja", "",
          "| variante | cadena | T_eff | log g | A_V | Δ | χ²_ν MUSE | bins |", "|---|---|---|---|---|---|---|---|"]
    for p in res["parejas"]:
        a, b = p["referencia"], p["cadena"]
        L.append(f"| {p['variante']} | {b['etiqueta']} | {a['der_teff']:.0f} → {b['der_teff']:.0f} | "
                 f"{a['der_logg']:.2f} → {b['der_logg']:.2f} | {a['av']:.2f} → {b['av']:.2f} | "
                 f"{a['margen']:+.3f} → {b['margen']:+.3f} | {a['chi2nu_muse']:.1f} → {b['chi2nu_muse']:.1f} | "
                 f"{a['bins']} → {b['bins']} |")
    L += ["", "## Otras cifras", ""]
    L.append("- Cambios de signo de Δ: " + ("ninguno" if not res["cambios_de_signo"] else "; ".join(
        f"{c['cadena']} {c['referencia']:+.3f} → {c['variante_margen']:+.3f}" for c in res["cambios_de_signo"])))
    L.append("- Δ con el prior, por objeto y rejilla (mín, máx): " + "; ".join(
        f"{k} [{v[0]:+.3f}, {v[1]:+.3f}]" for k, v in sorted(res["margen_con_prior"].items())))
    if res["semiancho_teff_referencias"]:
        L.append(f"- Semiancho 16–84 % de T_eff de las referencias: {res['semiancho_teff_referencias'][0]:.1f}–"
                 f"{res['semiancho_teff_referencias'][1]:.1f} K")
    if res["sin_pareja"]:
        L.append(f"- Sin cadena de referencia: {', '.join(res['sin_pareja'])}")
    if res.get("sombreadas"):
        L.append(f"- Cadenas repetidas (se usa la primera): {len(res['sombreadas'])}; " + "; ".join(
            f"{u} sobre {s}" for u, s in res["sombreadas"]))
    if res["nlive_distinto"]:
        L.append(f"- Con otro nlive que su referencia: {', '.join(res['nlive_distinto'])}")
    if dif_tex is not None:
        L.append("- Tabla D.2 del tex: " + ("IGUAL al redondeo impreso" if not dif_tex else "DISTINTA: " + "; ".join(dif_tex)))
    return "\n".join(L) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cadenas", type=Path, action="append", required=True,
                    help="directorio de las cadenas de referencia (las del paper; repetible, se recorre entero)")
    ap.add_argument("--variantes", type=Path, action="append", required=True,
                    help="directorio de las cadenas con otras máscaras (repetible, se recorre entero)")
    ap.add_argument("--project-root", type=Path, default=ROOT,
                    help="raíz que contiene runs/ (por defecto la del repo; necesaria desde un worktree)")
    ap.add_argument("--tex", type=Path, help="main.tex del paper: compara la Tabla D.2 (tab:masks) fila a fila")
    ap.add_argument("--json", type=Path, help="escribe todo en json")
    ap.add_argument("--md", type=Path, help="escribe el informe en markdown (si no, a la salida estándar)")
    a = ap.parse_args(argv)
    warnings.simplefilter("ignore")
    raiz = tb.fija_raiz(a.project_root)
    sombreadas = []
    referencias, variantes = carga(a.cadenas, sombreadas), carga(a.variantes, sombreadas)
    slugs = sorted({d["slug"] for d in variantes.values()})
    entradas = {s: tb.entrada_margen(s, root=raiz) for s in slugs}
    res = calcula(referencias, variantes, entradas)
    res["sombreadas"] = sombreadas
    filas = filas_tabla(res, slugs)
    dif = compara_tex(a.tex.read_text(), filas) if a.tex else None
    md = informe_md(res, filas, slugs, dif)
    if a.md:
        a.md.parent.mkdir(parents=True, exist_ok=True)
        a.md.write_text(md)
        print(f"escrito {a.md}")
    else:
        print(md)
    if a.json:
        a.json.parent.mkdir(parents=True, exist_ok=True)
        a.json.write_text(json.dumps(dict(res, filas_tabla=filas, slugs=slugs, tex=dif), indent=1, default=float,
                                     ensure_ascii=False))
        print(f"escrito {a.json}")
    return 1 if dif else 0


if __name__ == "__main__":
    sys.exit(main())
