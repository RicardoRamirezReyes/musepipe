#!/usr/bin/env python
"""¿El desnivel del 22 % entre metodos lo predice el error del modelo de PSF?

El paso 1 (`d1_descompone_continuo.py`) dejo esto: los cuatro metodos de continuo
se parten en dos bandos separados un **22 %** —{aperture, psffit} contra
{optimal_ls, optimal_psfsub}— y **cada `apcorr` es correcto respecto de su propia
base** (apertura 1.000, optimal 0.998). O sea que no es contabilidad: los cuatro
corrigen bien, con el mismo modelo, y aun asi discrepan.

Solo queda que el MODELO este mal de forma que importe distinto a cada apertura:

  * la apertura mide `box3` y corrige con 1/E_modelo(box3)
  * la optimal mide dentro de r<=8 px y corrige con 1/E_modelo(r<=8)

Si el modelo se equivoca distinto en esas dos regiones, cada metodo hereda un
sesgo distinto, y su cociente es:

    aperture/optimal = [E_dato(box3)/E_mod(box3)] / [E_dato(r<=8)/E_mod(r<=8)]

**La prediccion falsable:** eso tiene que dar ~1.22. Si da ~1.00, la hipotesis
del modelo de PSF esta muerta y hay que mirar el fondo.

Se mide sobre **la primaria**, que es donde hay señal, en el mismo cubo del que
extraen los metodos. No escribe nada.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
import sys  # noqa: E402

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from musepipe.psf import evaluate_psf_model  # noqa: E402
from musepipe.stages.stage_x01_aperture import _load_positions  # noqa: E402
from musepipe.stages.stage_x02_optimal import (  # noqa: E402
    _load_stage02_cube,
    stage_x02_config_from_run,
    stage_x02_paths,
)
from musepipe.stages.stage_x10_compare import COMPARISON_BANDS  # noqa: E402

BANDAS = [b for b in COMPARISON_BANDS if b.kind == "continuum"]


def regiones(shape, cy, cx):
    """box3, r<=8 y r<=25 alrededor de (cy, cx). box3 = 3x3 de pixeles enteros."""
    yy, xx = np.indices(shape, dtype=np.float64)
    r = np.hypot(yy - cy, xx - cx)
    iy, ix = int(round(cy)), int(round(cx))
    box3 = (np.abs(yy - iy) <= 1) & (np.abs(xx - ix) <= 1)
    return {"box3": box3, "r8": r <= 8.0, "r25": r <= 25.0}, r


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--project-root", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    cfg = stage_x02_config_from_run(args.run_id, project_root=args.project_root)
    paths = stage_x02_paths(cfg["run_id"], Path(cfg["project_root"]))
    comp_yx, star_yx, _p, _q = _load_positions(paths, cfg)
    psf = json.load(open(paths["psf_model_json"]))
    cube, wave, _cp, _bu = _load_stage02_cube(paths, cfg)

    cy, cx = float(star_yx[0]), float(star_yx[1])
    mascaras, r = regiones(cube.shape[1:], cy, cx)
    # Fondo: anillo lejano, con el compañero enmascarado.
    rc = np.hypot(*np.indices(cube.shape[1:], dtype=np.float64) - np.array([[[comp_yx[0]]], [[comp_yx[1]]]]))
    anillo = (r > 40) & (r < 60) & (rc > 12)

    filas = []
    print(f"### {cfg['run_id']} — energia encerrada: dato contra modelo, sobre la primaria\n")
    print(f"{'banda':<6}{'lambda':>7}   {'E_dato(box3)':>13}{'E_mod(box3)':>12}{'razon':>7}   "
          f"{'E_dato(r8)':>11}{'E_mod(r8)':>10}{'razon':>7}   {'prediccion':>11}")
    for b in BANDAS:
        s = (wave >= b.lo_A) & (wave < b.hi_A)
        if not np.any(s):
            continue
        img = np.nanmedian(cube[s], axis=0)
        fondo = np.nanmedian(img[anillo])
        img = img - fondo
        lam = 0.5 * (b.lo_A + b.hi_A)
        P = evaluate_psf_model(psf, lam, *(np.indices(img.shape, dtype=np.float64) - np.array([[[cy]], [[cx]]])))
        d = {k: float(np.nansum(img[m])) for k, m in mascaras.items()}
        mo = {k: float(np.nansum(P[m])) for k, m in mascaras.items()}
        # fracciones respecto de r<=25, que es donde el modelo esta normalizado
        fd = {k: d[k] / d["r25"] for k in ("box3", "r8")}
        fm = {k: mo[k] / mo["r25"] for k in ("box3", "r8")}
        rb, r8 = fd["box3"] / fm["box3"], fd["r8"] / fm["r8"]
        filas.append({"banda": b.name, "lambda_A": lam, "E_dato": fd, "E_modelo": fm,
                      "razon_box3": rb, "razon_r8": r8, "prediccion_aperture_sobre_optimal": rb / r8})
        print(f"{b.name:<6}{lam:7.0f}   {fd['box3']:13.4f}{fm['box3']:12.4f}{rb:7.3f}   "
              f"{fd['r8']:11.4f}{fm['r8']:10.4f}{r8:7.3f}   {rb/r8:11.3f}")

    pred = np.median([f["prediccion_aperture_sobre_optimal"] for f in filas])
    print(f"\n  PREDICCION mediana de aperture/optimal por el error del modelo: {pred:.3f}")
    print(f"  MEDIDO por el paso 1 (desnivel de aperture contra optimal_ls):   1.227")
    print(f"  -> {'la hipotesis del modelo de PSF EXPLICA el desnivel' if abs(pred-1.227) < 0.06 else 'NO cuadra: la hipotesis del modelo de PSF no basta'}")
    if args.out:
        Path(args.out).write_text(json.dumps({"run_id": cfg["run_id"], "prediccion_mediana": pred,
                                              "bandas": filas}, indent=2), encoding="utf-8")
    return filas


if __name__ == "__main__":
    main()


# --- Añadido tras falsar la hipotesis de la normalizacion -------------------
# La energia encerrada mide la NORMALIZACION del modelo. Pero la optimal no
# normaliza y ya esta: **pesa** cada pixel por P/sigma^2, y con un perfil
# equivocado el estimador de Horne queda sesgado por un funcional distinto,
# mucho mas sensible (C3 mide que +-10 % de FWHM le mueve el flujo un 26 %).
# Aqui se corren los DOS estimadores sobre la primaria, que es una fuente
# puntual con señal de sobra, y se comparan: si estan bien, deben coincidir.

def estimadores_sobre_la_primaria(cube, stat, wave, psf, cy, cx):
    import numpy as np
    from musepipe.psf import evaluate_psf_model
    mascaras, r = regiones(cube.shape[1:], cy, cx)
    dy, dx = np.indices(cube.shape[1:], dtype=np.float64) - np.array([[[cy]], [[cx]]])
    anillo = (r > 40) & (r < 60)
    filas = []
    for b in BANDAS:
        s = (wave >= b.lo_A) & (wave < b.hi_A)
        if not np.any(s):
            continue
        img = np.nanmedian(cube[s], axis=0)
        var = np.nanmedian(stat[s], axis=0) if stat is not None else np.abs(img) + 1.0
        img = img - np.nanmedian(img[anillo])
        lam = 0.5 * (b.lo_A + b.hi_A)
        P = evaluate_psf_model(psf, lam, dy, dx)
        e_box3 = float(np.nansum(P[mascaras["box3"]])) / float(np.nansum(P[mascaras["r25"]]))
        # apertura: suma en box3, corregida por la fraccion del modelo
        f_ap = float(np.nansum(img[mascaras["box3"]])) / e_box3
        # optimal: Horne con el perfil normalizado DENTRO de la ventana, luego apcorr
        w = mascaras["r8"]
        p = np.clip(P[w], 0, None); p = p / p.sum()
        d, v = img[w], np.clip(var[w], 1e-30, None)
        f_win = float(np.nansum(p * d / v) / np.nansum(p * p / v))
        e_r8 = float(np.nansum(P[mascaras["r8"]])) / float(np.nansum(P[mascaras["r25"]]))
        f_opt = f_win / e_r8
        filas.append({"banda": b.name, "lambda_A": lam, "f_apertura": f_ap,
                      "f_optimal": f_opt, "aperture_sobre_optimal": f_ap / f_opt})
    return filas
