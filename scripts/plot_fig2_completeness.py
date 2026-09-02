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
    args = ap.parse_args(argv)

    in_dir = Path(args.in_dir)
    filas = leer_curvas(in_dir / "fig2_curvas.csv")
    proc = json.loads((in_dir / "fig2_procedencia.json").read_text(encoding="utf-8"))
    comp = proc["comparacion"]
    metodos = [m.strip() for m in args.metodos.split(",") if m.strip()]
    # los runs salen de los propios productos, en su orden de aparicion
    runs = list(dict.fromkeys(r["run"] for r in filas))
    color = {r: PALETA[i % len(PALETA)] for i, r in enumerate(runs)}
    etiqueta = {r: nombre_a_mostrar(r) for r in runs}

    fig, axes = plt.subplots(1, len(metodos), figsize=(4.6 * len(metodos), 4.2),
                             sharey=True)
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
            ax.plot(x, y, "o-", color=color[run], markersize=4, linewidth=1.6,
                    label=f"{etiqueta[run]} ({n_pos} positions)")

        c = comp[metodo][args.exclusion]
        # SNR50: el estadistico de la figura, marcado en cada curva.
        for run, key in zip(runs, ("snr50_a", "snr50_b")):
            ax.axvline(c[key], color=color[run], linestyle=":", linewidth=1.2)
        ax.axhline(0.5, color="0.55", linestyle="--", linewidth=0.9)

        p = c["p"]
        p_txt = "$p < 10^{-4}$" if p <= 1e-4 else f"$p = {p:.2f}$"
        ax.set_title(f"{metodo}\n"
                     f"SNR$_{{50}}$: {c['snr50_a']:.2f} vs {c['snr50_b']:.2f}  "
                     f"($\\times${c['razon']:.2f}, {p_txt})", fontsize=10)
        ax.set_xlabel("Injected S/N")
        ax.set_xlim(0, 3.1)
        ax.set_ylim(-0.03, 1.03)
        ax.grid(alpha=0.25, linewidth=0.6)
        ax.legend(loc="lower right", fontsize=8, framealpha=0.9)

    axes[0].set_ylabel("Completeness (recovered fraction)")

    tag = proc["procedencia"].get("tag") or proc["procedencia"]["commit_corto"]
    fig.suptitle("Injection\u2013recovery completeness: both companions, "
                 "identical configuration", fontsize=12)
    fig.text(0.99, 0.01,
             f"{tag} · seed {proc['semilla']} · "
             f"{proc['n_boot']} bootstrap / {proc['n_perm']} permutations · "
             f"exclusion rule: {EXCL_EN[args.exclusion]}",
             ha="right", va="bottom", fontsize=6.5, color="0.35")
    fig.tight_layout(rect=(0, 0.03, 1, 0.94))

    out = Path(args.out) if args.out else in_dir / "fig2_completitud.pdf"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=200, bbox_inches="tight")
    print(out)
    print(out.with_suffix(".png"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
