#!/usr/bin/env python
"""El Δ del D.5 sin los intervalos extremos del espectro MUSE: cuánto mueven el Δ combinado entre rejillas y la T_eff.

Solo lectura y fuera de la cadena (apéndice D del paper, «Δ between grids», `app:checks:grids`; 2026-10-30). Lee las
cadenas de `teff_bayes.py ajusta` (json + npz) y no lanza ninguna. Las variantes son las de `teff_bayes.py
--quita-muse` (`extremos`, `azul`, `rojo`: `teff_bayes.INTERVALOS_MUSE`); el json de cada una lo declara
(`quita_muse`). El D.5 dice que no sabemos cuánto mueven T_eff y Δ los dos intervalos donde peor ajustan los modelos;
este script da esas cifras (todavía no están en el paper).

Por compañero y caso de A_V (1.8, 3.0 y con el prior de la primaria), con las funciones de `delta_entre_rejillas`, las
que dan el D.5: las cadenas de la Tabla B.7 a R = 500 (`cadenas_b7`; Sonora Diamondback es una rejilla, `familia`), su
Δ por muestra (`teff_bayes.margen_de_corrida`) y su mezcla a pesos iguales (`estadisticas`). Cada cadena de una
variante se empareja con la del paper por su json (`CONFIG`: objeto, rejilla, modo, A_V, A_V fija, resolución, datos,
nlive y semilla), y no puede llevar ninguna otra variante (`teff_bayes.VARIANTES_JSON`). Por variante: el Δ combinado
(mediana, 16–84 %), la fracción de la mezcla sobre el nivel, P(Δ > 0), la desviación típica entre rejillas, y por
rejilla el cambio de la mediana de Δ y el de la media de las medianas de T_eff (en Diamondback, de sus seis modelos) y
de A_V. Una variante se combina solo si están todas sus cadenas; si faltan algunas, se dice cuántas.

Con `--tex`, comprueba que las cadenas del paper dan el Δ combinado y la fracción sobre el nivel que imprime el D.5,
al redondeo impreso, y sale con 1 si no.

Uso (las cadenas del paper y las de las variantes, en las carpetas de la máquina del autor):

    python scripts/d5_intervalos_muse.py --project-root RAIZ_CON_RUNS --cadenas ~/muse_noche_20261024_c4/cadenas \\
        --variantes ~/muse_noche_20261029_disco/lote/res/d5 --variantes ~/muse_noche_20261030/lote/res \\
        --tex RAIZ_CON_RUNS/paper/main.tex --json salida.json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import delta_entre_rejillas as dr  # noqa: E402
from scripts import teff_bayes as tb  # noqa: E402
from scripts import teff_cifras_texto as tc  # noqa: E402

#: lo que tiene que coincidir entre una cadena del paper y su variante (con el valor por defecto del json)
CONFIG = {"slug": None, "rejilla": "btsettl", "modo": None, "av": None, "av_fija": None, "r_analisis": 500,
          "datos": "todo", "nlive": None, "semilla": None}


def clave(d):
    return tuple(d.get(k, v) for k, v in CONFIG.items())


def quitado(d):
    """El intervalo que quita la cadena (`extremos`, `azul`, `rojo`), o None si no es una variante de `--quita-muse`
    sola (sobre los datos, `libre`, sin ninguna otra variante)."""
    q = d.get("quita_muse")
    if q is None or d.get("modo") != "libre" or d.get("verdad"):
        return None
    otras = {k: v for k, v in tb.VARIANTES_JSON.items() if k != "quita_muse"}
    if any(d.get(k, v) != v for k, v in otras.items()):
        return None
    return q


def indice_variantes(V):
    """{(clave, intervalo): cadena} de las cadenas de `V` (`teff_cifras_texto.Cadenas`) que son variantes de
    `--quita-muse`. Dos cadenas con la misma clave y el mismo intervalo son un error: no se elige por la etiqueta."""
    out = {}
    for d in V.por_etiqueta.values():
        q = quitado(d)
        if q is None:
            continue
        k = (clave(d), q)
        if k in out:
            raise SystemExit(f"dos cadenas para la misma configuración y `{q}`: {out[k]['_json']} y {d['_json']}")
        out[k] = d
    return out


def d5_del_paper(tex, slugs):
    """{slug: {av: (mediana, fracción)}} del Δ combinado que imprime el D.5 (`app:checks:grids`): la frase «The
    combined \\(\\Delta\\) is ...» (el primer compañero) y «The combined value is ...» (el segundo), cada una con tres
    medianas (A_V 1.8, 3.0 y con el prior) seguidas de tres fracciones «above the noise level is/in ...». `slugs`, en
    el orden del paper (`teff_cifras_texto.SLUGS`)."""
    i = tex.index(r"\label{app:checks:grids}")
    fin = re.search(r"\\(?:sub)*section\{", tex[i:])
    sec = re.sub(r"\s+", " ", tex[i:i + fin.start()] if fin else tex[i:])
    frases = list(re.finditer(r"combined (?:\\\(\\Delta\\\)|value) is \\\(", sec))
    if len(frases) != len(slugs):
        raise ValueError(f"el D.5 tiene {len(frases)} frases del Δ combinado y hay {len(slugs)} compañeros")
    out = {}
    cortes = [f.start() for f in frases] + [len(sec)]
    for k, s in enumerate(slugs):
        resto = sec[cortes[k]:cortes[k + 1]]                 # hasta la frase del compañero siguiente, no más allá
        medianas = re.findall(r"\\\(([+-]\d+\.\d+)\^", resto)[:3]
        fr = re.search(r"above the noise level (?:is|in) ((?:\\\(\d\.\d+\\\)(?:,| and)? ?){3})", resto)
        if len(medianas) != 3 or fr is None:
            raise ValueError(f"no leo las tres medianas y las tres fracciones del D.5 de {s}")
        fracs = re.findall(r"\\\((\d\.\d+)\\\)", fr.group(1))
        out[s] = {av: (m, x) for av, m, x in zip(tc.AVS, medianas, fracs)}
    return out


def _media(xs):
    return float(np.mean(xs)) if xs else float("nan")


def calcula(K, V, root, E=None, tex=None):
    """Todo, en un dict (lo que va al json). `K`, `V`: `teff_cifras_texto.Cadenas` del paper y de las variantes.
    `E`: {slug: `teff_bayes.entrada_margen`}; por defecto se lee de los runs del paper."""
    root = Path(root)
    tc.fija_objetos(root)
    if E is None:
        E = {s: tb.entrada_margen(s, root=root) for s in tc.SLUGS}
    var = indice_variantes(V)
    impreso = d5_del_paper(tex, tc.SLUGS) if tex is not None else None
    res = dict(casos=[], tex=None if impreso is None else dict(iguales=True, distintos=[]))
    for s in tc.SLUGS:
        r500 = {k: v for k, v in dr.cadenas_b7(K, s).items() if not k[1]}
        for av in tc.AVS:
            grupos = {}
            for (rj, _), porav in r500.items():
                grupos.setdefault(dr.familia(rj), []).append(porav[av])
            ref_cad = [d for v in grupos.values() for d in v]
            ref = dr.estadisticas({g: [d["etiqueta"] for d in v] for g, v in grupos.items()},
                                  {d["etiqueta"]: K.delta(d, E) for d in ref_cad},
                                  {d["etiqueta"]: d["logz"] for d in ref_cad})
            caso = dict(slug=s, av=av, referencia=ref, variantes={}, incompletas={})
            if impreso is not None:
                m, x = impreso[s][av]
                propio = (f"{ref['c']['p50']:+.2f}", f"{ref['c']['p_positivo']:.2f}")
                if propio != (m, x):
                    res["tex"]["iguales"] = False
                    res["tex"]["distintos"].append(dict(slug=s, av=av, impreso=(m, x), calculado=propio))
            for q in tb.INTERVALOS_MUSE:
                gv = {g: [var.get((clave(d), q)) for d in v] for g, v in grupos.items()}
                faltan = sum(d is None for v in gv.values() for d in v)
                if faltan == len(ref_cad):
                    continue
                if faltan:
                    caso["incompletas"][q] = dict(faltan=faltan, de=len(ref_cad))
                    continue
                vc = [d for v in gv.values() for d in v]
                est = dr.estadisticas({g: [d["etiqueta"] for d in v] for g, v in gv.items()},
                                      {d["etiqueta"]: V.delta(d, E) for d in vc}, {d["etiqueta"]: d["logz"] for d in vc})
                por = {}
                for g in grupos:
                    t0, t1 = (_media([d["posterior"]["teff"]["p50"] for d in c]) for c in (grupos[g], gv[g]))
                    a0, a1 = (_media([d["posterior"]["av"]["p50"] for d in c]) for c in (grupos[g], gv[g]))
                    por[g] = dict(delta=(ref["por_rejilla"][g]["p50"], est["por_rejilla"][g]["p50"]),
                                  semiancho_ref=ref["por_rejilla"][g]["semiancho"], teff=(t0, t1), av=(a0, a1))
                caso["variantes"][q] = dict(estadisticas=est, por_rejilla=por,
                                            etiquetas=[d["etiqueta"] for d in vc])
            res["casos"].append(caso)
    return res


def informe(res):
    L = []
    for c in res["casos"]:
        r = c["referencia"]
        L.append(f"\n== {c['slug']}, A_V {c['av']}: Δ combinado {r['c']['p50']:+.2f} [{r['c']['p16']:+.2f}, "
                 f"{r['c']['p84']:+.2f}], P(Δ>0) {r['c']['p_positivo']:.2f}, sd entre rejillas {r['b']['sd']:.2f}")
        for q, v in c["variantes"].items():
            e = v["estadisticas"]["c"]
            L.append(f"   sin {q:9} Δ combinado {e['p50']:+.2f} [{e['p16']:+.2f}, {e['p84']:+.2f}] (cambio "
                     f"{e['p50'] - r['c']['p50']:+.2f}), P(Δ>0) {e['p_positivo']:.2f} (cambio "
                     f"{e['p_positivo'] - r['c']['p_positivo']:+.2f}), sd entre rejillas {v['estadisticas']['b']['sd']:.2f}")
            for g, p in v["por_rejilla"].items():
                L.append(f"      {g:12} Δ {p['delta'][0]:+.2f} → {p['delta'][1]:+.2f} ({p['delta'][1] - p['delta'][0]:+.2f};"
                         f" semiancho {p['semiancho_ref']:.2f})   T_eff {p['teff'][0]:.0f} → {p['teff'][1]:.0f} K   "
                         f"A_V {p['av'][0]:.2f} → {p['av'][1]:.2f}")
        for q, f in c["incompletas"].items():
            L.append(f"   sin {q:9} faltan {f['faltan']} de {f['de']} cadenas: no se combina")
    t = res["tex"]
    if t is None:
        L.append("\nD.5 del paper: no comparado (sin `--tex`)")
    elif t["iguales"]:
        L.append("\nD.5 del paper: las cadenas del paper dan el Δ combinado y la fracción impresos (IGUAL)")
    else:
        L.append("\nD.5 del paper: DISTINTO en " + "; ".join(
            f"{x['slug']} {x['av']}: impreso {x['impreso']}, calculado {x['calculado']}" for x in t["distintos"]))
    return "\n".join(L)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cadenas", type=Path, action="append", required=True,
                    help="directorio de las cadenas del paper (repetible; gana el primero)")
    ap.add_argument("--variantes", type=Path, action="append", required=True,
                    help="directorio de cadenas con `--quita-muse` (repetible; se recorre entero y se ignora lo demás)")
    ap.add_argument("--project-root", type=Path, default=ROOT,
                    help="raíz que contiene runs/ (por defecto la del repo; necesaria desde un worktree)")
    ap.add_argument("--tex", type=Path, help="main.tex del paper: compara el Δ combinado del D.5")
    ap.add_argument("--json", type=Path, help="escribe todo en json")
    a = ap.parse_args(argv)
    warnings.simplefilter("ignore")
    raiz = tb.fija_raiz(a.project_root)
    res = calcula(tc.Cadenas(a.cadenas), tc.Cadenas(a.variantes), raiz, tex=a.tex.read_text() if a.tex else None)
    print(informe(res))
    if a.json:
        a.json.parent.mkdir(parents=True, exist_ok=True)
        a.json.write_text(json.dumps(res, indent=1, ensure_ascii=False, default=float))
        print(f"escrito {a.json}")
    return 0 if res["tex"] is None or res["tex"]["iguales"] else 1


if __name__ == "__main__":
    sys.exit(main())
