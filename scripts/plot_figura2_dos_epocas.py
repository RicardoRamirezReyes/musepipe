#!/usr/bin/env python3
"""Figura 2 de la propuesta: completitud e inyeccion-recuperacion en DOS epocas.

    python scripts/plot_figura2_dos_epocas.py \
        --epoca1 ROXs12b_OB3445598 --epoca2 ROXs12b_OB3444577

Lo que pidio el colaborador (traspaso 2026-08-19 §6): flujo inyectado vs fraccion
recuperada, dos epocas del mismo objeto, umbral y estadistico declarados, y tasa
de falsos positivos.

**El eje X es el FLUJO inyectado, no el S/N**, y no es un detalle: con S/N en el
eje, "S/N = 5" vale 5 en las dos epocas por construccion y la diferencia entre
ellas *desaparece*. Con flujo se ve, que es justo lo que la figura afirma.

Las barras son intervalos de Wilson, no sqrt(p(1-p)/n): con n pequeno y p pegado
a 0 o 1 —que es exactamente donde cae la epoca mala— la formula normal da barras
que se salen del [0,1] o de ancho cero.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

#: Paleta categorica validada con `dataviz/scripts/validate_palette.js` (modo
#: claro): CVD delta-E 24.7 en protan, muy por encima del umbral 8. El marcador
#: cambia con la epoca ademas del color, para que la identidad no dependa solo
#: del color en impresion en gris o con daltonismo.
EPOCA_ESTILO = (
    {"color": "#2a78d6", "marker": "o", "label": None},
    {"color": "#eb6834", "marker": "s", "label": None},
)
TINTA = "#1a1a19"
TINTA_SUAVE = "#5c5b55"


def wilson(k, n, z=1.0):
    """Intervalo de Wilson para una proporcion. `z=1` ~ 68 % (1 sigma)."""

    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1.0 + z * z / n
    centro = (p + z * z / (2 * n)) / d
    medio = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, centro - medio), min(1.0, centro + medio))


def lee_inyecciones(run_id, *, solo_controles=False):
    """Filas nominales de E4, agrupadas por (metodo, flujo inyectado).

    Se fija `template_factor=1` y `continuum_mode='none'` porque es la rejilla
    que define `completeness_at_5sigma` en el QC: la figura tiene que decir lo
    mismo que el producto, no una variante suya.
    """

    ruta = ROOT / "runs" / run_id / "tables" / "injection_throughput_by_method.csv"
    if not ruta.exists():
        raise SystemExit(f"falta {ruta}: corre E4 en {run_id}")
    grupos = defaultdict(lambda: {"n": 0, "k": 0, "snr": None})
    for fila in csv.DictReader(ruta.open()):
        if fila["variant"] != "nominal" or fila["continuum_mode"] != "none":
            continue
        if abs(float(fila["template_factor"]) - 1.0) > 1e-6:
            continue
        if solo_controles and not fila["position_label"].startswith("control"):
            continue
        snr = float(fila["input_snr"])
        if snr <= 0:
            continue  # los nulos van al panel B
        clave = (fila["method"], round(float(fila["injected_flux"]), 6))
        g = grupos[clave]
        g["n"] += 1
        g["k"] += int(str(fila["complete"]).lower() in ("true", "1"))
        g["snr"] = snr
    return grupos


def lee_nulos(run_id, metodo):
    """S/N recuperado donde NO se inyecto nada, y cuantos cruzan el umbral.

    Es el panel que contesta a la vez dos de las cuatro cosas que pidio el
    colaborador: la tasa de falsos positivos y el umbral/estadistico. Un grafico
    de barras de la tasa sale vacio —es cero— y un panel vacio no dice que el
    margen sea comodo; la distribucion si.
    """

    ruta = ROOT / "runs" / run_id / "tables" / "injection_throughput_by_method.csv"
    z, k = [], 0
    for fila in csv.DictReader(ruta.open()):
        if fila["variant"] != "nominal" or float(fila["input_snr"]) > 0:
            continue
        if not fila["position_label"].startswith("control"):
            continue  # la posicion real es dato cientifico, no un nulo (E4 v2 §1.2)
        if fila["method"] != metodo:
            continue
        try:
            z.append(float(fila["recovered_snr"]))
        except (TypeError, ValueError):
            continue
        k += int(str(fila["complete"]).lower() in ("true", "1"))
    return z, k


def etiqueta_epoca(run_id):
    cfg = json.loads((ROOT / "runs" / run_id / "config" / "config.json").read_text())["config"]
    n = len(cfg.get("perexp_cubes") or [])
    return f"OB {cfg.get('obs_id','?')} · {cfg.get('night','?')} · {n} exp"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--epoca1", required=True)
    ap.add_argument("--epoca2", required=True)
    ap.add_argument("--metodo", default="psffit", help="metodo canonico (default: psffit)")
    ap.add_argument("--salida", default="reports/figura2_dos_epocas.pdf")
    ap.add_argument("--solo-controles", action="store_true",
                    help="excluir la posicion real del companero de la completitud")
    args = ap.parse_args(argv)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    runs = [args.epoca1, args.epoca2]
    #: Cuantos puntos de flujo tiene cada epoca. NO tiene por que ser el mismo
    #: numero, y el pie lo dice: la rejilla se inyecta en la escala de S/N de cada
    #: cubo, y solo la de la epoca 1 se bajo para que su transicion entrara.
    n_puntos_por_epoca = []
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(9.6, 4.0), width_ratios=(1.35, 1.0))

    # ---- Panel A: completitud vs flujo inyectado ---------------------------
    for run, estilo in zip(runs, EPOCA_ESTILO):
        grupos = lee_inyecciones(run, solo_controles=args.solo_controles)
        puntos = sorted((f, g) for (m, f), g in grupos.items() if m == args.metodo)
        if not puntos:
            raise SystemExit(f"{run}: sin filas para el metodo {args.metodo}")
        x = [f for f, _ in puntos]
        y = [g["k"] / g["n"] for _, g in puntos]
        lo, hi = zip(*(wilson(g["k"], g["n"]) for _, g in puntos))
        # Wilson NO esta centrado en p̂ —esa es justo su virtud cerca de 0 y 1—
        # asi que la barra puede quedar toda a un lado. Se recorta a >=0 para
        # dibujarla; el intervalo que se muestra sigue siendo [lo, hi].
        yerr = [[max(0.0, a - b) for a, b in zip(y, lo)],
                [max(0.0, b - a) for a, b in zip(hi, y)]]
        axA.errorbar(x, y, yerr=yerr, color=estilo["color"], marker=estilo["marker"],
                     markersize=6.5, linewidth=2.0, capsize=0, elinewidth=1.2,
                     label=etiqueta_epoca(run), zorder=3)
        n_por_punto = puntos[0][1]["n"]
        n_puntos_por_epoca.append(len(puntos))

    axA.set_xscale("log")
    axA.set_ylim(-0.04, 1.08)
    axA.set_xlabel("flujo H$\\alpha$ inyectado  [$10^{-20}$ erg s$^{-1}$ cm$^{-2}$]")
    axA.set_ylabel("fracción recuperada")
    axA.axhline(0.5, color=TINTA_SUAVE, linewidth=0.8, linestyle=":", zorder=1)
    axA.text(0.015, 0.52, "50 %", transform=axA.get_yaxis_transform(),
             fontsize=7.5, color=TINTA_SUAVE, va="bottom")
    axA.legend(frameon=False, fontsize=8.5, loc="center left", bbox_to_anchor=(0.0, 0.42))
    axA.set_title("a · el mismo flujo, dos noches", fontsize=10, color=TINTA, loc="left")

    # ---- Panel B: los nulos frente al umbral --------------------------------
    import numpy as np

    UMBRAL = 5.0
    total_fp = 0
    for i, (run, estilo) in enumerate(zip(runs, EPOCA_ESTILO)):
        z, k = lee_nulos(run, args.metodo)
        total_fp += k
        n_nulos = len(z)
        rng = np.random.default_rng(0)  # jitter reproducible, solo cosmetico
        y = i + rng.uniform(-0.16, 0.16, size=len(z))
        axB.scatter(z, y, s=34, color=estilo["color"], marker=estilo["marker"],
                    edgecolor="white", linewidth=0.8, zorder=3, alpha=0.95)

    axB.axvline(UMBRAL, color=TINTA, linewidth=1.4, zorder=4)
    axB.text(UMBRAL, 1.62, f" umbral {UMBRAL:g}$\\sigma$", fontsize=8.5, color=TINTA,
             va="top", ha="left")
    axB.set_yticks([0, 1])
    axB.set_yticklabels(["época 1", "época 2"], fontsize=9)
    axB.set_ylim(-0.5, 1.7)
    axB.set_xlabel("S/N recuperado donde no se inyectó nada")
    axB.set_title("b · los nulos, y dónde está el umbral", fontsize=10, color=TINTA, loc="left")
    axB.text(0.02, 0.06,
             f"{total_fp} falsos positivos de {n_nulos * 2} nulos",
             transform=axB.transAxes, fontsize=9, color=TINTA, ha="left", va="bottom")

    for ax in (axA, axB):
        ax.grid(True, color="#e6e5e0", linewidth=0.7, zorder=0)
        ax.set_axisbelow(True)
        for lado in ("top", "right"):
            ax.spines[lado].set_visible(False)
        for lado in ("left", "bottom"):
            ax.spines[lado].set_color(TINTA_SUAVE)
            ax.spines[lado].set_linewidth(0.8)
        ax.tick_params(colors=TINTA_SUAVE, labelsize=8.5, length=3)

    rejillas = ("misma rejilla de flujo en las dos épocas"
                if len(set(n_puntos_por_epoca)) == 1 else
                f"las rejillas de flujo NO son la misma ({' y '.join(str(n) for n in n_puntos_por_epoca)} "
                f"puntos): cada época se inyecta en su propia escala de S/N, y solo la primera se bajó "
                f"para que su transición entrara en el rango. El eje es flujo, así que las curvas siguen "
                f"siendo comparables")
    pie = (f"Método {args.metodo} (canónico). Completitud = fracción de inyecciones recuperadas "
           f"a $\\geq$5$\\sigma$ sobre {n_por_punto} posiciones por punto; barras = intervalo de "
           f"Wilson a 1$\\sigma$, que no está centrado en la fracción medida. Panel a: {rejillas}.\n"
           f"El umbral y la FAP "
           f"son empíricos, medidos sobre los espectros de control del mismo método (E4 v3). "
           f"Panel b: {n_nulos} nulos por época, solo en posiciones de control; la posición real "
           f"del compañero es dato científico y no entra.")
    fig.text(0.008, -0.10, pie, fontsize=7.6, color=TINTA_SUAVE, va="top", ha="left", wrap=True)

    salida = ROOT / args.salida
    salida.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(salida, bbox_inches="tight")
    fig.savefig(salida.with_suffix(".png"), dpi=200, bbox_inches="tight")
    print(f"figura -> {salida}")
    print(f"         {salida.with_suffix('.png')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
