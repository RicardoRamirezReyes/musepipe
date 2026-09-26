#!/usr/bin/env python
"""¿Es de la calibración el desplazamiento en λ de la primaria entre noches? El cielo, exposición a exposición.

Solo lectura. Continúa docs/2026-09-26_desplazamiento_primaria_y_ao_por_exposicion.md §1: la primaria de
ROXs 12 está +0.14 Å más al rojo en la segunda noche que en la primera, en el marco baricéntrico de los
cubos por exposición. Dos causas posibles: la solución en λ de esa noche está corrida, o la velocidad de
la estrella cambió.

A4/M1 ya mide el residuo de la solución en λ con tres líneas de airglow (O I 5577, 6300 y 6363 Å), que
están en reposo en el marco TOPOCÉNTRICO, pero junta las 29 exposiciones en una mediana. Este script hace
la misma medida exposición a exposición, con las mismas funciones y los mismos parámetros que
`cube_qc.m1m2_sky_phase` sobre los mismos `SKY_SPECTRUM` (los que lista el QC de A4), y la pone junto al
desplazamiento de la primaria y a la corrección baricéntrica (`BARYCORR`) de cada cubo.

Lectura: si el cielo de la segunda noche se mueve lo mismo que la primaria, es la calibración; si el cielo
no se mueve, la estrella sí (velocidad radial) o hay un efecto que el cielo no ve.

uso: python scripts/lambda_cielo_por_exposicion.py RUN DESPLAZAMIENTO_JSON [--json SALIDA]
     DESPLAZAMIENTO_JSON sale de scripts/desplazamiento_primaria_por_exposicion.py
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

C_KMS = 299792.458


def offsets_de_un_cielo(path):
    """Residuo de la solución en λ por línea (Å), con los parámetros de A4/M1."""
    from musepipe.qc.cube_qc import M1_CLEAN_AIRGLOW, measure_skylines, read_sky_spectrum_fits
    wave, flux = read_sky_spectrum_fits(Path(path))
    meds = measure_skylines(wave, flux, M1_CLEAN_AIRGLOW, frame="topocentric", vbary_kms=0.0,
                            min_snr=10.0, half_width_A=4.0)
    return {m.name: float(m.offset_A) for m in meds}


def exposicion_de(path):
    """El id de exposición (`<fecha>_MUSE.<fecha>T<hora>`) es el directorio padre del fichero."""
    return Path(path).parent.name


def main(argv=None):
    from astropy.io import fits
    from musepipe.config import load_run_config
    from musepipe.io import read_json

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("run_id")
    ap.add_argument("desplazamiento_json")
    ap.add_argument("--json", default=None)
    a = ap.parse_args(argv)

    run = load_run_config(a.run_id)
    qc00 = read_json(run.paths.stage_dir / "stage00q_qc.json")
    cielos = {exposicion_de(p): p for p in qc00["m1_wavelength"]["sky_spectrum_files"]}
    cubos = {exposicion_de(p): p for p in run.config.get("perexp_cubes", [])}
    desp = json.load(open(a.desplazamiento_json))
    prim = {f["id"]: f for f in desp["por_exposicion"]}

    filas = []
    for eid in sorted(prim, key=lambda k: prim[k]["mjd"]):
        if eid not in cielos:
            print(f"{eid}: sin SKY_SPECTRUM en el QC de A4", file=sys.stderr)
            continue
        por_linea = offsets_de_un_cielo(cielos[eid])
        barycorr = None
        if eid in cubos:
            barycorr = fits.getheader(cubos[eid], 0).get("BARYCORR")
        filas.append({"id": eid, "noche": prim[eid]["noche"], "mjd": prim[eid]["mjd"],
                      "cielo_por_linea_A": por_linea,
                      "cielo_mediana_A": float(np.median(list(por_linea.values()))) if por_linea else None,
                      "primaria_A": prim[eid]["media_A"], "primaria_err_A": prim[eid]["media_err_A"],
                      "barycorr_kms": barycorr})

    noches = sorted({f["noche"] for f in filas})
    res = {"run": a.run_id, "por_exposicion": filas, "por_noche": {}}
    for n in noches:
        g = [f for f in filas if f["noche"] == n]
        cielo = [f["cielo_mediana_A"] for f in g if f["cielo_mediana_A"] is not None]
        lineas = sorted({k for f in g for k in f["cielo_por_linea_A"]})
        res["por_noche"][n] = {
            "n": len(g),
            "cielo_mediana_A": float(np.median(cielo)) if cielo else None,
            "cielo_dispersion_A": float(np.std(cielo, ddof=1)) if len(cielo) > 1 else None,
            "cielo_por_linea_A": {k: float(np.median([f["cielo_por_linea_A"][k] for f in g
                                                      if k in f["cielo_por_linea_A"]])) for k in lineas},
            "primaria_media_A": float(np.mean([f["primaria_A"] for f in g])),
            "barycorr_mediano_kms": float(np.median([f["barycorr_kms"] for f in g
                                                     if f["barycorr_kms"] is not None]))
            if any(f["barycorr_kms"] is not None for f in g) else None,
        }
    if len(noches) == 2:
        n1, n2 = (res["por_noche"][n] for n in noches)
        d_cielo = n2["cielo_mediana_A"] - n1["cielo_mediana_A"]
        d_prim = n2["primaria_media_A"] - n1["primaria_media_A"]
        res["segunda_menos_primera"] = {
            "cielo_A": d_cielo, "primaria_A": d_prim, "primaria_sin_cielo_A": d_prim - d_cielo,
            "primaria_sin_cielo_kms": (d_prim - d_cielo) / 6563.0 * C_KMS,
            "cielo_por_linea_A": {k: n2["cielo_por_linea_A"][k] - n1["cielo_por_linea_A"][k]
                                  for k in n1["cielo_por_linea_A"] if k in n2["cielo_por_linea_A"]},
        }

    print(f"{'exposición':42s} {'cielo (Å)':>10s} {'primaria (Å)':>13s} {'BARYCORR':>9s}  por línea")
    for f in filas:
        pl = " ".join(f"{k}:{v:+.3f}" for k, v in sorted(f["cielo_por_linea_A"].items()))
        cm = f["cielo_mediana_A"]
        print(f"{f['id'][:42]:42s} {cm if cm is None else format(cm, '+10.3f')} "
              f"{f['primaria_A']:+13.3f} {f['barycorr_kms'] if f['barycorr_kms'] is None else format(f['barycorr_kms'], '9.3f')}  {pl}")
    for n, r in res["por_noche"].items():
        print(f"{n}: n={r['n']}, cielo {r['cielo_mediana_A']:+.3f} Å (sd {r['cielo_dispersion_A']:.3f}), "
              f"primaria {r['primaria_media_A']:+.3f} Å, BARYCORR {r['barycorr_mediano_kms']} km/s, "
              f"por línea {r['cielo_por_linea_A']}")
    if "segunda_menos_primera" in res:
        s = res["segunda_menos_primera"]
        print(f"segunda − primera: cielo {s['cielo_A']:+.3f} Å, primaria {s['primaria_A']:+.3f} Å → "
              f"primaria sin cielo {s['primaria_sin_cielo_A']:+.3f} Å ({s['primaria_sin_cielo_kms']:+.1f} km/s); "
              f"por línea {s['cielo_por_linea_A']}")
    if a.json:
        with open(a.json, "w") as fh:
            json.dump(res, fh, indent=1)


if __name__ == "__main__":
    main()
