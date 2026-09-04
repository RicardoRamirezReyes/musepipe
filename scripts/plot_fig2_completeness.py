#!/usr/bin/env python
"""Dibuja la Figura 2 (completitud vs SNR inyectada) a partir de los productos
de `scripts/fig2_completeness.py`.

El script de medida emite CSV + JSON de procedencia y NO dibuja; esto es el paso
de dibujo, separado a proposito para que la figura se pueda regenerar sin volver
a correr los 10000 bootstrap/permutaciones.

    python scripts/plot_fig2_completeness.py                      # productos congelados
    python scripts/plot_fig2_completeness.py --in-dir <dir>       # otro calculo
    python scripts/plot_fig2_completeness.py --exclusion sin_exclusion
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]

# Un color por objeto, asignado por orden de aparicion en los productos; el
# metodo se distingue por panel, no por color.
PALETA = ("#1f77b4", "#d62728", "#2ca02c", "#9467bd")
# El valor viaja en espanol en los productos; solo se traduce al dibujarlo.
# El valor viaja en espanol en los productos; solo se traduce al dibujarlo.
EXCL_EN = {"con_exclusion": "with exclusion", "sin_exclusion": "no exclusion"}


def nombre_a_mostrar(run_id):
    """El nombre del objeto sale del registro, nunca de un literal aqui.

    Ver `tests/test_no_hardcoded_target.py`: un nombre escrito a mano hace que
    un objeto nuevo herede en silencio la etiqueta del primero.
    """
    cfg_json = ROOT / "runs" / run_id / "config" / "config.json"
    if not cfg_json.exists():
        return run_id
    slug = (json.loads(cfg_json.read_text()).get("chain") or {}).get("target")
    if not slug:
        return run_id
    ficha = ROOT / "targets" / f"{slug}.json"
    if not ficha.exists():
        return slug
    return json.loads(ficha.read_text()).get("display_name", slug)


def leer_curvas(path: Path) -> list[dict]:
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in-dir", default=str(ROOT / "reports" / "fig2"),
                    help="directorio con fig2_curvas.csv y fig2_procedencia.json")
    ap.add_argument("--out", default=None,
                    help="ruta del PDF (por defecto <in-dir>/fig2_completitud.pdf)")
    ap.add_argument("--exclusion", default="con_exclusion",
                    choices=["con_exclusion", "sin_exclusion"],
                    help="regla de exclusion pre-registrada a dibujar")
    ap.add_argument("--metodos", default="aperture,psffit",
                    help="un panel por metodo, en este orden")
    # Para incrustar en un .docx sin que cambie la paginacion hay que fijar el
    # tamaño FISICO, no dejar que lo decida el numero de paneles.
    ap.add_argument("--figsize-cm", default=None, metavar="ANCHOxALTO",
                    help="tamaño fisico en cm, p.ej. 13.5x6.30; por defecto "
                         "4.6 in por panel x 4.2 in")
    ap.add_argument("--dpi", type=int, default=200, help="dpi del PNG")
    # A tamaño pequeño el supertitulo y el pie no caben y roban alto a los
    # paneles. Cuando la figura va dentro de un documento que ya lleva su
    # leyenda, se quitan y esa informacion la escribe el pie del documento.
    ap.add_argument("--sin-suptitulo", action="store_true")
    ap.add_argument("--sin-pie", action="store_true")
    args = ap.parse_args(argv)

    in_dir = Path(args.in_dir)
    filas = leer_curvas(in_dir / "fig2_curvas.csv")
    proc = json.loads((in_dir / "fig2_procedencia.json").read_text(encoding="utf-8"))
    comp = proc["comparacion"]
    # El p que se cita sale de las posiciones DISJUNTAS. E4 reparte las posiciones
    # uniformemente en el anillo y con `h04_n_control_positions` alto quedan mas
    # juntas que el radio de su ventana: 55 posiciones a 8.1 px con ventanas de
    # r=14 comparten casi todos sus pixeles, asi que permutar 55 unidades cuando
    # hay 13 deja el p sin calibrar (1.0e-4 nominal contra 1.05e-2 real).
    ind_json = in_dir / "fig2_independientes.json"
    ind = json.loads(ind_json.read_text(encoding="utf-8")) if ind_json.exists() else None
    metodos = [m.strip() for m in args.metodos.split(",") if m.strip()]
    # los runs salen de los propios productos, en su orden de aparicion
    runs = list(dict.fromkeys(r["run"] for r in filas))
    color = {r: PALETA[i % len(PALETA)] for i, r in enumerate(runs)}
    etiqueta = {r: nombre_a_mostrar(r) for r in runs}

    if args.figsize_cm:
        w_cm, h_cm = (float(x) for x in args.figsize_cm.lower().split("x"))
        figsize = (w_cm / 2.54, h_cm / 2.54)
        # A 13.5x6.30 cm los cuerpos por defecto no caben: se escalan con el alto.
        escala = min(1.0, (h_cm / 2.54) / 4.2)
    else:
        figsize, escala = (4.6 * len(metodos), 4.2), 1.0
    fs = lambda base: max(4.2, base * escala)
    fig, axes = plt.subplots(1, len(metodos), figsize=figsize, sharey=True)
    if len(metodos) == 1:
        axes = [axes]

    for ax, metodo in zip(axes, metodos):
        for run in runs:
            sel = [r for r in filas
                   if r["run"] == run and r["method"] == metodo
                   and r["exclusion"] == args.exclusion]
            sel.sort(key=lambda r: float(r["input_snr"]))
            x = [float(r["input_snr"]) for r in sel]
            y = [float(r["completeness"]) for r in sel]
            lo = [float(r["ic95_lo"]) for r in sel]
            hi = [float(r["ic95_hi"]) for r in sel]
            n_pos = sel[0]["positions_per_point"] if sel else "?"
            ax.fill_between(x, lo, hi, color=color[run], alpha=0.18, linewidth=0)
            # n independiente si lo hay: es el que sostiene el p del titulo.
            n_ind_run = None
            if ind:
                g = (ind.get("geometria") or {}).get(run) or {}
                n_ind_run = g.get("n_tras_exclusion") or g.get("n_disjuntas")
            etq = (f"{etiqueta[run]} ({n_ind_run} indep. positions)" if n_ind_run
                   else f"{etiqueta[run]} ({n_pos} positions)")
            ax.plot(x, y, "o-", color=color[run], markersize=3 * escala + 1,
                    linewidth=1.6 * escala + 0.3, label=etq)

        c = comp[metodo][args.exclusion]
        cita = (ind["comparacion"][metodo]["independientes"]
                if ind and metodo in ind.get("comparacion", {}) else c)
        # Las verticales salen del MISMO conjunto que el titulo: si el numero
        # citado es el de las disjuntas, la marca tiene que estar donde el numero.
        for run, key in zip(runs, ("snr50_a", "snr50_b")):
            ax.axvline(cita[key], color=color[run], linestyle=":", linewidth=1.2)
        ax.axhline(0.5, color="0.55", linestyle="--", linewidth=0.9)
        p = cita["p"]
        p_txt = "$p < 10^{-4}$" if p <= 1e-4 else f"$p = {p:.3f}$"
        n_ind = cita.get("n_posiciones")
        sufijo = (f"\nfrom {n_ind[0]}+{n_ind[1]} non-overlapping positions" if n_ind else "")
        ax.set_title(f"{metodo}\n"
                     f"SNR$_{{50}}$: {cita['snr50_a']:.2f} vs {cita['snr50_b']:.2f}  "
                     f"($\\times${cita['razon']:.2f}, {p_txt}){sufijo}", fontsize=fs(10))
        ax.set_xlabel("Injected S/N", fontsize=fs(10))
        ax.set_xlim(0, 3.1)
        ax.set_ylim(-0.03, 1.03)
        ax.grid(alpha=0.25, linewidth=0.6)
        ax.legend(loc="lower right", fontsize=fs(8), framealpha=0.9)

    axes[0].set_ylabel("Completeness (recovered fraction)", fontsize=fs(10))

    tag = proc["procedencia"].get("tag") or proc["procedencia"]["commit_corto"]
    if not args.sin_suptitulo:
        fig.suptitle("Injection\u2013recovery completeness: both companions, "
                     "identical configuration", fontsize=fs(12))
    pie = (f"{tag} · seed {proc['semilla']} · "
           f"{proc['n_boot']} bootstrap / {proc['n_perm']} permutations · "
           f"exclusion rule: {EXCL_EN[args.exclusion]}")
    if ind:
        # El p nominal se publica tambien: es trazabilidad, no el numero a citar.
        nom = {m: ind["comparacion"][m]["todas"] for m in ind["comparacion"]}
        pie += ("\ncurves use all positions; p and SNR$_{50}$ in the titles use only "
                f"positions $\\geq${ind['separacion_minima_px']:.0f} px apart "
                "(non-overlapping windows). Over all "
                f"{'+'.join(str(v['n_posiciones'][i]) for i, v in enumerate(list(nom.values())[:1] * 2))}"
                " nominal positions the same test gives "
                + ", ".join(f"{m}: p={v['p']:.4f}" for m, v in nom.items()) + ".")
    if not args.sin_pie:
        fig.text(0.99, 0.005, pie, ha="right", va="bottom", fontsize=fs(6.2), color="0.35")
    abajo = 0.0 if args.sin_pie else (0.10 if args.figsize_cm else 0.03)
    arriba = 1.0 if args.sin_suptitulo else (0.90 if args.figsize_cm else 0.94)
    fig.tight_layout(rect=(0, abajo, 1, arriba))
    # el pie sigue imprimiendose siempre, para que quien escriba la leyenda del
    # documento lo tenga a mano aunque no vaya dentro de la imagen
    print("PIE:", pie.replace("\n", " "))

    out = Path(args.out) if args.out else in_dir / "fig2_completitud.pdf"
    # Con `--figsize-cm` NO se recorta: `bbox_inches="tight"` reajusta el lienzo
    # al contenido y el fichero sale con otro tamaño fisico del pedido, que es
    # justo lo que descuadra la paginacion de un .docx.
    recorte = None if args.figsize_cm else "tight"
    fig.savefig(out, bbox_inches=recorte)
    fig.savefig(out.with_suffix(".png"), dpi=args.dpi, bbox_inches=recorte)
    print(out)
    print(out.with_suffix(".png"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
