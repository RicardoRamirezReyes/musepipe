#!/usr/bin/env python
"""Recalcula la Figura 2 usando solo posiciones que NO se solapan.

`fig2_completeness.py` trata cada posicion de E4 como una unidad independiente
-y lo dice: «las filas de una posicion comparten ruido local, residuo de PSF y
cielo»-. Pero E4 reparte las posiciones uniformemente en el anillo del radio del
companero, y con `h04_n_control_positions` alto quedan a menos distancia que el
radio de su propia ventana: en el run canonico de ROXs 12 b, 55 posiciones a
71.2 px de radio caen a 8.1 px unas de otras, con ventanas de r=12 px (ajuste) y
anillo de fondo hasta r=14. Comparten la mayor parte de sus pixeles.

Consecuencia: la muestra efectiva esta inflada y el p de permutacion no esta
calibrado. Este script mide cuanto, SIN re-correr E4 y SIN tocar los productos
congelados: filtra la tabla larga que ya esta en disco y rehace el mismo calculo
con las funciones de `fig2_completeness.py`.

    python scripts/fig2_posiciones_independientes.py
    python scripts/fig2_posiciones_independientes.py --sep-min 28 --n-perm 10000
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from fig2_completeness import (  # noqa: E402
    permutacion_snr50, posiciones_contaminadas)

#: r_out del anillo de fondo de cada inyeccion (px). Dos ventanas son disjuntas
#: si sus centros distan >= 2*R_ANILLO.
R_ANILLO_PX = 14.0


def leer_tabla_larga(path):
    filas = {}
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            filas.setdefault(r["run"], []).append({
                "run": r["run"],
                "position_id": r["position_id"],
                "position_label": r["position_label"],
                "position_y": float(r["position_y"]),
                "position_x": float(r["position_x"]),
                "injection_id": r["injection_id"],
                "method": r["method"],
                "input_snr": float(r["input_snr"]),
                "template_width": r["template_width"],
                "continuum_mode": r["continuum_mode"],
                "complete": r["complete"] == "True",
            })
    return filas


def posiciones_de(filas):
    d = {}
    for f in filas:
        d.setdefault(f["position_id"], (f["position_y"], f["position_x"], f["position_label"]))
    return d


def seleccionar_disjuntas(pos, sep_min):
    """Greedy empezando por la real: la posicion del companero nunca se descarta."""
    reales = [k for k, v in pos.items() if v[2] == "real"]
    fuera = []
    for k in reales + [k for k in pos if k not in reales]:
        y, x, _ = pos[k]
        if all(math.hypot(y - pos[j][0], x - pos[j][1]) >= sep_min for j in fuera):
            fuera.append(k)
    return fuera


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in-dir", default=str(ROOT / "reports" / "fig2"))
    ap.add_argument("--out", default=None, help="por defecto <in-dir>/fig2_independientes.json")
    ap.add_argument("--sep-min", type=float, default=2 * R_ANILLO_PX,
                    help="separacion minima entre posiciones (px)")
    ap.add_argument("--metodos", default="aperture,psffit")
    ap.add_argument("--exclusion", default="par")
    ap.add_argument("--n-perm", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=20260826)
    args = ap.parse_args(argv)

    in_dir = Path(args.in_dir)
    filas = leer_tabla_larga(in_dir / "fig2_tabla_larga.csv")
    runs = list(filas)
    if len(runs) != 2:
        raise SystemExit(f"se esperaban 2 objetos en la tabla larga, hay {len(runs)}")
    rejilla = sorted({f["input_snr"] for f in filas[runs[0]]})

    geometria, conjuntos = {}, {}
    for run in runs:
        pos = posiciones_de(filas[run])
        ind = seleccionar_disjuntas(pos, args.sep_min)
        cont = posiciones_contaminadas(filas[run], args.exclusion)
        geometria[run] = {"n_nominales": len(pos), "n_disjuntas": len(ind),
                          "n_tras_exclusion": len(set(ind) - set(cont))}
        conjuntos[run] = {"todas": sorted(set(pos) - set(cont)),
                          "independientes": sorted(set(ind) - set(cont))}
        print(f"{run}: {len(pos)} posiciones -> {len(ind)} disjuntas a >= {args.sep_min:.0f} px "
              f"-> {geometria[run]['n_tras_exclusion']} tras la exclusion '{args.exclusion}'")

    a, b = runs
    salida = {"separacion_minima_px": float(args.sep_min),
              "r_anillo_px": R_ANILLO_PX, "exclusion": args.exclusion,
              "n_perm": int(args.n_perm), "semilla": int(args.seed),
              "geometria": geometria, "comparacion": {}}
    for metodo in [m.strip() for m in args.metodos.split(",") if m.strip()]:
        salida["comparacion"][metodo] = {}
        for etiqueta in ("todas", "independientes"):
            rng = np.random.default_rng(args.seed)
            r = permutacion_snr50(filas[a], filas[b], metodo,
                                  conjuntos[a][etiqueta], conjuntos[b][etiqueta],
                                  rejilla, args.n_perm, rng)
            r["n_posiciones"] = [len(conjuntos[a][etiqueta]), len(conjuntos[b][etiqueta])]
            salida["comparacion"][metodo][etiqueta] = r
            print(f"  {metodo:<9} {etiqueta:<15} n={r['n_posiciones']}  "
                  f"SNR50 {r['snr50_a']:.3f} vs {r['snr50_b']:.3f}  "
                  f"razon {r['razon']:.3f}  p={r['p']:.5f}")

    out = Path(args.out) if args.out else in_dir / "fig2_independientes.json"
    out.write_text(json.dumps(salida, indent=1, ensure_ascii=False), encoding="utf-8")
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
