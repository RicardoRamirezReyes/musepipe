#!/usr/bin/env python3
"""Reduce la tabla de emisividades de Storey & Hummer (1995) a lo que usa el paper.

    python scripts/case_b_sh95.py <ruta>/e1b.d.gz

La entrada es `e1b.d.gz` del catalogo VizieR VI/64 (Storey & Hummer 1995,
MNRAS 272, 41): emisividades del caso B de H I, 4 pi j / (n_e n_p) en
erg cm^3 s^-1, para 10 temperaturas y 13 densidades. Su formato es el que lee
el `intrat.f` del propio catalogo: por cada (T, n_e) una cabecera
`(1x, e10.3, 5x, e10.3, 13x, i2)` con n_e, T y `ncut`, y luego
ncut (ncut - 1) / 2 emisividades en `8e10.3`, con la transicion (n_u, n_l) en
la posicion k = (ncut - n_u)(ncut + n_u - 1) / 2 + n_l (base 1).

Escribe `musepipe/models/case_b_sh95.json` con las lineas que el paper usa
(Halpha, Hbeta y Pa 9 - Pa 18) en la rejilla T = 5000-20000 K, n_e = 10^2-10^8
cm^-3, y el sha256 del fichero de entrada. El JSON es lo que se versiona; el
`.gz` no, porque es un dato publico que se vuelve a bajar con su sha.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SALIDA = ROOT / "musepipe" / "models" / "case_b_sh95.json"

#: (n_u, n_l) de cada linea, con el nombre que usa el catalogo de G2.
LINEAS = {"Halpha": (3, 2), "Hbeta": (4, 2), **{f"Pa {n}": (n, 3) for n in range(9, 19)}}
TEMPERATURAS_K = (5000.0, 7500.0, 10000.0, 12500.0, 15000.0, 20000.0)
DENSIDADES_CM3 = (1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8)


def leer_e1b(texto: str) -> dict:
    """`{(T, n_e): (ncut, [emisividades])}` del fichero secundario de SH95."""
    lineas = texto.split("\n")
    ntemp, ndens = (int(v) for v in lineas[0].split())
    i, bloques = 1, {}
    for _ in range(ntemp):
        for _ in range(ndens):
            cab = lineas[i]
            i += 1
            dens, temp, ncut = float(cab[1:11]), float(cab[16:26]), int(cab[39:41])
            n = ncut * (ncut - 1) // 2
            vals: list[float] = []
            while len(vals) < n:
                fila = lineas[i].rstrip()
                i += 1
                vals += [float(fila[k:k + 10]) for k in range(0, len(fila), 10) if fila[k:k + 10].strip()]
            bloques[(temp, dens)] = (ncut, vals[:n])
    return bloques


def emisividad(bloque, nu: int, nl: int) -> float:
    ncut, vals = bloque
    if not (nl < nu <= ncut):
        raise ValueError(f"transicion {nu}->{nl} fuera de la tabla (ncut={ncut})")
    return float(vals[((ncut - nu) * (ncut + nu - 1)) // 2 + nl - 1])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("e1b_gz", type=Path)
    a = ap.parse_args(argv)
    crudo = a.e1b_gz.read_bytes()
    bloques = leer_e1b(gzip.decompress(crudo).decode("ascii"))
    emis = {nombre: [[emisividad(bloques[(t, d)], *nl) for d in DENSIDADES_CM3] for t in TEMPERATURAS_K]
            for nombre, nl in LINEAS.items()}
    doc = {
        "citation": "Storey, P. J. & Hummer, D. G. 1995, MNRAS, 272, 41 (1995MNRAS.272...41S)",
        "source": "VizieR VI/64, e1b.d.gz (Z = 1, case B)",
        "source_sha256": hashlib.sha256(crudo).hexdigest(),
        "case": "B",
        "quantity": "4 pi j / (n_e n_p)",
        "units": "erg cm^3 s^-1",
        "temperatures_K": list(TEMPERATURAS_K),
        "densities_cm3": list(DENSIDADES_CM3),
        "transitions": {k: list(v) for k, v in LINEAS.items()},
        "emissivity": emis,
    }
    SALIDA.write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8")
    print(SALIDA)


if __name__ == "__main__":
    main()
