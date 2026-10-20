#!/usr/bin/env python
"""Material suplementario del apéndice B: un corner por cadena de `teff_bayes.py ajusta`, todo en un PDF.

Pedido del autor 2026-10-18 («haz los tres puntos»): el paper muestra los corners de los ajustes con el prior en
A_V (Fig. B.4) y los de solo MUSE / solo IR (Fig. B.5); el resto de las cadenas que el apéndice cita (A_V fija,
otras semillas, otras rejillas, plantillas, inyecciones) va aquí, una página por cadena. Las cadenas no son
productos de la cadena de etapas y no viven en runs/, así que no hay directorios por defecto: se pasan por
secciones, y el PDF registra en cada página el nombre de la cadena.

Cada página: los parámetros físicos que varían en la cadena (T_eff, log g, R, A_V, r_K; en el modo `evo`, la edad
y los derivados de los tracks) y Δ, el margen sobre el suelo cromosférico (`teff_bayes.margen_de_corrida`, el de
las Tablas B.5 a B.7), salvo en las plantillas y las inyecciones, que no son el compañero. Contornos del 68 y el
95 %, títulos con la mediana y el 16–84 %, y la verdad inyectada cuando la hay. El texto del PDF va en inglés: es
material del paper.

uso: python scripts/teff_suplemento.py --salida teff_corners.pdf \\
         --seccion "Fits of Table B.6" 'DIR_BT/*.json' 'DIR_DRIFT/*.json' [--seccion TITULO PATRON ...]
Cada PATRON es un glob de json de `ajusta`, con su npz al lado; dentro de una sección, en el orden de los
patrones y, dentro de cada patrón, alfabético. `evo_solo` no es un ajuste a los espectros y no entra.
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
import textwrap
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402

from scripts import teff_bayes as tb  # noqa: E402

#: A4 vertical, en pulgadas
PAGINA_IN = (8.27, 11.69)
NOMBRE_DATOS = {"todo": "MUSE + near-IR", "muse": "MUSE only", "ir": "near-IR only"}
#: las bibliotecas de plantillas de `teff_bayes` con el nombre que les da el paper (`paper_figures.NOMBRE_BIBLIOTECA`)
NOMBRE_LIBRERIA = {"manara": "Manara+13,17 (young)", "lateM": "X-shooter young late-M",
                   "xshyne": "X-SHYNE L (10-200 Myr)"}


def nombre_rejilla(d):
    """El nombre de la rejilla de una corrida, en inglés (el de la tabla de rejillas si está en ella)."""
    rej = d.get("rejilla", "btsettl")
    nombre = tb.REJILLAS_TABLA.get(rej, tb.REJILLAS[rej]["nombre"]).replace(r"\(", "$").replace(r"\)", "$")
    nombre = nombre.replace("sin nubes", "no clouds")
    return nombre + (" ($R = 250$)" if d.get("r_analisis", 500) == 250 else "")


def descripcion(d, companero):
    """Las líneas de la cabecera de la página de la corrida `d`."""
    lineas = [f"{companero}: {nombre_rejilla(d)}, {NOMBRE_DATOS[d.get('datos', 'todo')]}"]
    if d["modo"] == "evo":
        lineas[0] += f", tied to the {tb.TRACKS[d['tracks']][1]} tracks"
    if d.get("av_fija") is not None:
        lineas.append(f"$A_V$ fixed at {d['av_fija']:.1f} mag")
    elif d.get("av") == "primaria":
        lineas.append("$A_V$ with the prior from the primary")
    else:
        lineas.append("$A_V$ free (0 to 8 mag)")
    v = d.get("verdad") or {}
    if v.get("plantilla"):
        lineas.append(f"Data replaced by the template {v['plantilla']} ({v['spt']}, "
                      f"{NOMBRE_LIBRERIA.get(v['lib'], v['lib'])}), reddened by $A_V$ = {v['av']:.1f} mag, "
                      f"with the noise of {companero}")
    elif v:
        lineas.append(f"Data replaced by a BT-Settl model with $T_{{\\rm eff}}$ = {v['teff']:.0f} K, "
                      f"$\\log g$ = {v['logg']:.2f}, $A_V$ = {v['av']:.1f} mag and $R$ = {v['r_rj']:.1f} $R_{{\\rm Jup}}$, "
                      f"with the noise of {companero}")
    lineas.append(f"{d['nlive']} live points, seed {d['semilla']}, "
                  f"$\\ln Z$ = {d['logz']:.1f} $\\pm$ {d['logz_err']:.1f}")
    return lineas


def columnas(d, entrada):
    """(etiquetas, muestras (n, k), verdades, decimales) de la página de `d`: los parámetros físicos que varían
    y Δ."""
    z = np.load(d["_npz"])
    nombres = [str(x) for x in z["nombres"]]
    m, der = z["muestras"], z["derivadas"]
    v = d.get("verdad") or {}
    cols = []
    if d["modo"] == "evo":
        cols += [(r"$\log$ age (Myr)", m[:, nombres.index("log_edad_myr")], None, 2),
                 (r"$T_{\rm eff}$ (K)", der[:, 0], None, 0), (r"$\log g$", der[:, 1], None, 2),
                 (r"$R$ ($R_{\rm Jup}$)", der[:, 2] / tb.RJUP_CM, None, 2)]
    else:
        cols += [(r"$T_{\rm eff}$ (K)", m[:, nombres.index("teff")], v.get("teff"), 0),
                 (r"$\log g$", m[:, nombres.index("logg")], v.get("logg"), 2),
                 (r"$R$ ($R_{\rm Jup}$)", m[:, nombres.index("r_rj")], v.get("r_rj"), 2)]
    # la A_V fija no es un parámetro, aunque `ajusta` la muestree en un perfil de ±0.01 mag
    if d.get("av_fija") is None:
        cols.append((r"$A_V$ (mag)", m[:, nombres.index("av")], v.get("av"), 2))
    if "rk" in nombres:
        cols.append((r"$r_K$", m[:, nombres.index("rk")], None, 2))
    if not v:
        cols.append((r"$\Delta$ (dex)", tb.margen_de_corrida(d, entrada), None, 2))
    return ([c[0] for c in cols], np.column_stack([c[1] for c in cols]), [c[2] for c in cols],
            [c[3] for c in cols])


def titulo(etiqueta, x, nd):
    """`etiqueta = mediana^{+}_{-}` con el 16–84 % y `nd` decimales (los de `corner` no admiten uno por columna)."""
    lo, m, hi = np.percentile(x, [16, 50, 84])
    return f"{etiqueta} = ${m:.{nd}f}^{{+{hi - m:.{nd}f}}}_{{-{m - lo:.{nd}f}}}$"


def pagina(pdf, d, entrada, companero, cabecera, pie):
    """Una página A4 con la cabecera, el corner de `d` y el pie."""
    import corner

    etiquetas, muestras, verdades, decimales = columnas(d, entrada)
    k = len(etiquetas)
    fig = plt.figure(figsize=PAGINA_IN)
    # corner separa la etiqueta del eje en fracciones del panel: con más paneles, más chicos, se pisaba con los
    # números de los ejes (medido: 7 columnas)
    corner.corner(muestras, fig=fig, bins=30, smooth=1.0, levels=(0.68, 0.95), color="#1f77b4",
                  plot_datapoints=False, plot_density=False, fill_contours=False, max_n_ticks=4,
                  labelpad=max(0.0, 0.05 * (k - 4)),
                  labels=etiquetas, label_kwargs=dict(fontsize=9), truths=verdades if any(x is not None for x in verdades) else None,
                  truth_color="#d55e00", hist_kwargs=dict(lw=0.9))
    for i in range(k):
        fig.axes[i * (k + 1)].set_title(titulo(etiquetas[i], muestras[:, i], decimales[i]), fontsize=7.5)
    for ax in fig.axes:
        ax.tick_params(labelsize=7)
    ancho = 0.84
    fig.subplots_adjust(left=0.11, right=0.11 + ancho, top=0.80, bottom=0.80 - ancho * PAGINA_IN[0] / PAGINA_IN[1],
                        wspace=0.06, hspace=0.06)
    fig.text(0.06, 0.965, cabecera, fontsize=12, weight="bold", va="top")
    lineas = [t for linea in descripcion(d, companero) for t in textwrap.wrap(linea, 105)]
    for i, linea in enumerate(lineas):
        fig.text(0.06, 0.94 - 0.022 * i, linea, fontsize=9.5, va="top")
    fig.text(0.06, 0.025, f"Chain: {d['etiqueta']}", fontsize=7, color="0.4")
    fig.text(0.94, 0.025, pie, fontsize=7, color="0.4", ha="right")
    pdf.savefig(fig)
    plt.close(fig)


def portada(pdf, secciones, total):
    """La primera página: qué hay en cada página y el índice de secciones."""
    fig = plt.figure(figsize=PAGINA_IN)
    y = 0.94
    fig.text(0.08, y, "Supplementary material to Appendix B", fontsize=15, weight="bold", va="top")
    y -= 0.035
    fig.text(0.08, y, "Posterior distributions of the fits of model atmospheres", fontsize=12, va="top")
    texto = (
        "Each page shows the posterior distribution of one nested-sampling fit of Appendix B: the physical\n"
        "parameters that vary in the fit and, for the fits to the companions, $\\Delta$, the difference between\n"
        "$\\log(L_{\\rm acc}/L_{\\rm bol})$ and the chromospheric noise level of Manara et al. (2013) at the\n"
        "temperature of each sample (an upper limit for ROXs 42B b). The contours enclose 68 and 95% of the\n"
        "samples; the titles give the median and the 16-84% interval. In the fits to templates and to injected\n"
        "models, the injected values are marked. Pages with a fixed $A_V$ do not show $A_V$; fits to the MUSE\n"
        "spectrum alone have no $r_K$. Each page gives the name of the chain, which identifies the run.")
    y -= 0.05
    fig.text(0.08, y, texto, fontsize=9.5, va="top", linespacing=1.5)
    y -= 0.21
    fig.text(0.08, y, "Contents", fontsize=12, weight="bold", va="top")
    y -= 0.035
    for i, (titulo, corridas, primera) in enumerate(secciones, 1):
        paginas = (f"page {primera}" if len(corridas) == 1
                   else f"pages {primera}-{primera + len(corridas) - 1}")
        # un título largo se parte, y las páginas van en su última línea, para que no se pisen
        lineas = textwrap.wrap(f"{i}. {titulo}", 68, subsequent_indent="    ")
        for k, linea in enumerate(lineas):
            fig.text(0.08, y - 0.02 * k, linea, fontsize=10, va="top")
        y -= 0.02 * (len(lineas) - 1)
        fig.text(0.92, y, f"{len(corridas)} fits, {paginas}", fontsize=10, va="top", ha="right")
        y -= 0.027
    fig.text(0.08, 0.04, f"{total} pages. Made with scripts/teff_suplemento.py of the pipeline, from the json and "
                          "npz files of scripts/teff_bayes.py.", fontsize=7.5, color="0.4")
    pdf.savefig(fig)
    plt.close(fig)


def carga_secciones(secciones):
    """[(título, [patrones])] -> [(título, [corridas])]: cada corrida, el json con `_npz`. Falla si un patrón no
    encuentra nada, si una corrida no tiene npz o si una misma cadena entra dos veces."""
    out, vistas = [], set()
    for titulo, patrones in secciones:
        corridas = []
        for patron in patrones:
            ficheros = sorted(glob.glob(str(Path(patron).expanduser())))
            if not ficheros:
                raise SystemExit(f"«{titulo}»: el patrón {patron} no encuentra ningún json")
            for f in ficheros:
                d = json.loads(Path(f).read_text())
                if d["modo"] == "evo_solo":
                    continue
                npz = Path(f).with_suffix(".npz")
                if not npz.exists():
                    raise SystemExit(f"{f} no tiene su npz")
                real = str(Path(f).resolve())
                if real in vistas:
                    raise SystemExit(f"{f} entra dos veces")
                vistas.add(real)
                d["_npz"] = str(npz)
                corridas.append(d)
        out.append((titulo, corridas))
    return out


def suplemento(secciones, salida):
    """Escribe el PDF; devuelve el número de páginas."""
    from scripts import tipo_ir_indices as tir

    cargadas = carga_secciones(secciones)
    slugs = sorted({d["slug"] for _, cs in cargadas for d in cs})
    companeros = {s: tir.perfil(s)["paper"]["display"] for s in slugs}
    entradas = {s: tb.entrada_margen(s) for s in slugs}
    total = 1 + sum(len(cs) for _, cs in cargadas)
    indice, n = [], 2
    for titulo, cs in cargadas:
        indice.append((titulo, cs, n))
        n += len(cs)
    # sin fecha de creación: el mismo PDF para las mismas cadenas
    with PdfPages(salida, metadata={"Title": "Supplementary material to Appendix B",
                                    "Creator": "scripts/teff_suplemento.py", "CreationDate": None}) as pdf:
        portada(pdf, indice, total)
        for i, (titulo, cs, primera) in enumerate(indice, 1):
            for j, d in enumerate(cs):
                pagina(pdf, d, entradas[d["slug"]], companeros[d["slug"]], f"{i}. {titulo}",
                       f"page {primera + j} of {total}")
    return total


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seccion", nargs="+", action="append", required=True, metavar=("TITULO", "PATRON"),
                    help="título de la sección y uno o más globs de json de `teff_bayes.py ajusta` (repetible)")
    ap.add_argument("--salida", type=Path, required=True)
    a = ap.parse_args(argv)
    secciones = []
    for s in a.seccion:
        if len(s) < 2:
            ap.error(f"--seccion {s[0]!r} sin ningún patrón")
        secciones.append((s[0], s[1:]))
    n = suplemento(secciones, a.salida)
    print(f"{n} páginas -> {a.salida}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
