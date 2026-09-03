#!/usr/bin/env python
"""Construye la compilacion de literatura de la Fig. mdot_mass_plane, desde CASPAR.

CASPAR = *Comprehensive Archive of Substellar and Planetary Accretion Rates*
(Betti et al. 2023, AJ; base de datos en Zenodo, doi:10.5281/zenodo.8393054).
Se filtra a los **companeros ligados** subestelares y planetarios --- la columna
`Companion == COM` --- que es exactamente la poblacion con la que el paper se
compara, y NO las enanas marrones aisladas, que son otra cosa.

Existe como script y no como tabla escrita a mano por lo mismo que
`paper_tables.py`: una cifra copiada a mano no tiene procedencia y no se puede
volver a comprobar. Aqui cada fila sale del CSV publicado, con su referencia
original y la relacion de escala con la que se derivo.

    python scripts/build_literature_mdot.py                 # descarga y escribe
    python scripts/build_literature_mdot.py --caspar X.csv  # copia local
    python scripts/build_literature_mdot.py --out otro.csv

Ojo con lo que NO hace: no re-deriva nada ni elige un valor por objeto. Un mismo
companero aparece tantas veces como determinaciones publicadas tenga, porque esa
dispersion --- Delorme 1 (AB) b va de 2.7e-13 a 5.6e-11 segun la relacion --- es
justo el punto que el pie de la figura afirma sobre la heterogeneidad de la
literatura, y promediarla lo escondería.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

#: fichero de datos dentro del registro Zenodo de CASPAR.
CASPAR_URL = ("https://zenodo.org/api/records/8393054/files/CASPAR.csv/content")
CASPAR_CITA = ("Betti et al. 2023, AJ, 'The Comprehensive Archive of Substellar "
               "and Planetary Accretion Rates'; base de datos "
               "doi:10.5281/zenodo.8393054")

#: masa del Sol en masas de Jupiter (CASPAR da la masa en Msun).
MSUN_A_MJUP = 1047.5655

CAMPOS = ["name", "caspar_name", "mass_mjup", "mdot_msun_yr", "mdot_err_msun_yr", "is_limit",
          "relation", "diagnostic", "age_myr", "reference"]


def descarga_caspar(destino: Path) -> Path:
    destino.parent.mkdir(parents=True, exist_ok=True)
    print(f"descargando CASPAR -> {destino}", file=sys.stderr)
    with urllib.request.urlopen(CASPAR_URL, timeout=180) as resp:
        destino.write_bytes(resp.read())
    return destino


def _float(texto):
    texto = (texto or "").strip()
    return float(texto) if texto else None


def companeros(caspar: Path) -> list[dict]:
    """Las determinaciones de Mdot de los companeros ligados, tal cual estan."""
    with open(caspar, newline="", encoding="utf-8-sig") as fh:
        filas = list(csv.DictReader(fh))
    # la primera fila del CSV lleva las unidades, no un objeto
    filas = [r for r in filas if (r.get("Unique Name") or "").strip()]

    salida = []
    for r in filas:
        if (r.get("Companion") or "").strip() != "COM":
            continue
        masa = _float(r.get("Mass"))
        mdot = _float(r.get("Accretion Rate"))
        if masa is None or mdot is None or mdot <= 0:
            continue
        # el limite superior lo marca la columna `Upper Limit` del diagnostico
        # que se uso; basta con que alguna este puesta en esta fila.
        es_limite = any(k.endswith("Upper Limit") and (v or "").strip()
                        for k, v in r.items())
        salida.append({
            # el nombre de Simbad es el resoluble; el de CASPAR suele ser el
            # comun ("2MASS J0103...(AB)b" = Delorme 1 (AB) b) y se guarda
            # tambien para poder reconocer el objeto al leer el CSV.
            "name": (r.get("Simbad-Resolvable Name") or r["Unique Name"]).strip(),
            "caspar_name": r["Unique Name"].strip(),
            "mass_mjup": round(masa * MSUN_A_MJUP, 2),
            "mdot_msun_yr": f"{mdot:.4e}",
            "mdot_err_msun_yr": ("" if _float(r.get("Accretion Rate err")) is None
                                 else f"{_float(r['Accretion Rate err']):.4e}"),
            "is_limit": "1" if es_limite else "0",
            "relation": (r.get("Scaling Relation") or "").strip(),
            "diagnostic": (r.get("Accretion Diagnostic") or "").strip(),
            "age_myr": (r.get("Age") or "").strip(),
            "reference": (r.get("Original Reference") or "").strip(),
        })
    salida.sort(key=lambda d: (d["mass_mjup"], d["name"], d["reference"]))
    return salida


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--caspar", default=None,
                    help="copia local de CASPAR.csv; por defecto se descarga")
    ap.add_argument("--out", default=str(ROOT / "paper" / "literature_mdot.csv"))
    args = ap.parse_args(argv)

    caspar = Path(args.caspar) if args.caspar else ROOT / "paper" / "_caspar.csv"
    if not caspar.exists():
        descarga_caspar(caspar)

    filas = companeros(caspar)
    objetos = sorted({f["name"] for f in filas})
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as fh:
        fh.write(
            "# Compilacion de literatura para la Fig. mdot_mass_plane.\n"
            "# GENERADO por scripts/build_literature_mdot.py — no editar a mano.\n"
            f"# Fuente: {CASPAR_CITA}\n"
            f"# Descargado/procesado: {dt.date.today().isoformat()}\n"
            f"# Filtro: Companion == 'COM' (companeros LIGADOS subestelares y\n"
            "#   planetarios), con Mass y Accretion Rate. Excluye las enanas\n"
            "#   marrones aisladas, que son otra poblacion.\n"
            f"# {len(filas)} determinaciones de {len(objetos)} objetos:\n"
            + "".join(f"#   - {o}\n" for o in objetos) +
            "# Una fila por determinacion publicada, NO una por objeto: la\n"
            "#   dispersion entre relaciones de escala es el punto que el pie de\n"
            "#   la figura afirma, y promediarla lo esconderia.\n"
            "# `relation` dice con que relacion Lacc-Lline se derivo cada cifra;\n"
            "#   'Alcalà+2017' es la misma que usa este trabajo.\n")
        w = csv.DictWriter(fh, fieldnames=CAMPOS)
        w.writeheader()
        w.writerows(filas)

    print(f"{len(filas)} determinaciones de {len(objetos)} objetos -> {out}")
    for o in objetos:
        n = sum(1 for f in filas if f["name"] == o)
        print(f"   {o:<34} {n} determinacion(es)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
