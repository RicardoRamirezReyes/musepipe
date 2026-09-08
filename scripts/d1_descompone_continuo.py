#!/usr/bin/env python
"""¿La divergencia de continuo de D1 es un OFFSET DE NIVEL o una INCLINACION cromatica?

`divergent_continuum` lleva abierto desde julio y no lo cierra nada de lo que se
ha probado. Antes de buscar la causa conviene saber **de que forma** discrepan
los metodos, porque las dos formas apuntan a sitios distintos:

  * **nivel** (las seis bandas desplazadas por igual) -> la NORMALIZACION, o sea
    como cada metodo usa el modelo de PSF para convertir lo que mide en flujo:
    `apcorr` = F(<=25px)/F(box3) en la apertura, pesos de Horne en la optimal,
    la columna del ajuste en psffit.
  * **inclinacion** (la razon cambia del azul al rojo) -> algo cromatico: el
    halo de la primaria, el fondo, o el cromatismo que al modelo le falta.

Para cada par de los cuatro metodos de continuo se ajusta, sobre las seis bandas,

    razon(banda) = nivel + inclinacion * (lambda_banda - lambda_centro) / 1000 A

y se compara contra **la misma cantidad medida en las 33 posiciones de control**,
que dan la nula: si en los controles sale lo mismo, no es del objeto.

**No escribe nada**: lee los productos calibrados de D2 y sus controles.
"""
from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
import sys  # noqa: E402

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from musepipe.extraction.product import SpectrumProduct  # noqa: E402
from musepipe.stages.stage_x10_compare import (  # noqa: E402
    COMPARISON_BANDS,
    CONTINUUM_METHODS,
    _integrated_sigma,
    _integrated_sum,
    empirical_sigma_diff,
    stage_x10_config_from_run,
    stage_x10_paths,
)

BANDAS = [b for b in COMPARISON_BANDS if b.kind == "continuum"]


def bandas_de(wave, sigma, fi, fj):
    """Por banda de continuo: diferencia integrada, sigma integrada y nivel de referencia.

    Se usan **las mismas funciones que D1** (`_integrated_sum`, `_integrated_sigma`
    y la sigma empirica de los controles), para que el resultado se lea en la
    misma escala que sus t. La primera version de este script normalizaba por la
    dispersion de COCIENTES de controles, y eso no vale: un control dividido por
    otro es un cociente de dos ruidos, con dispersion patologica, y todo salia
    insignificante.
    """

    d, sig, ref, lam = [], [], [], []
    for b in BANDAS:
        m = (wave >= b.lo_A) & (wave < b.hi_A)
        di = _integrated_sum(wave, fi, m)
        dj = _integrated_sum(wave, fj, m)
        si = _integrated_sigma(wave, sigma, m)
        if not (np.isfinite(di) and np.isfinite(dj) and np.isfinite(si) and si > 0):
            continue
        d.append(di - dj); sig.append(si); ref.append(0.5 * (di + dj)); lam.append(0.5 * (b.lo_A + b.hi_A))
    return (np.array(d), np.array(sig), np.array(ref), np.array(lam))


def descompone(d, sig, ref, lam):
    """Separa la diferencia en un factor MULTIPLICATIVO y su deriva cromatica.

        d(banda) = alfa * ref(banda) + beta * ref(banda) * (lambda - lambda0)/1000

    `alfa` es el desnivel fraccionario (0.10 = un metodo da 10 % mas que el otro)
    y `beta` cuanto cambia ese desnivel por cada 1000 A. Se ajusta pesando por
    1/sigma, que es lo que hace que los numeros se lean como los t de D1.
    """

    if d.size < 3:
        return {}
    x = (lam - lam.mean()) / 1000.0
    A = np.column_stack([ref, ref * x]) / sig[:, None]
    y = d / sig
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    cov = np.linalg.pinv(A.T @ A)
    err = np.sqrt(np.diag(cov))
    resid = y - A @ coef
    return {"alfa": float(coef[0]), "alfa_sigma": float(err[0]),
            "beta": float(coef[1]), "beta_sigma": float(err[1]),
            "chi2_residual": float(np.sum(resid**2)), "n_bandas": int(d.size)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--project-root", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    cfg = stage_x10_config_from_run(args.run_id, project_root=args.project_root)
    paths = stage_x10_paths(cfg["run_id"], Path(cfg["project_root"]))
    stage = paths["paths"].stage_dir

    espectros, controles = {}, {}
    for m in CONTINUUM_METHODS:
        espectros[m] = SpectrumProduct.read(stage / f"spec_calibrated_{m}_object.fits")
        z = np.load(stage / f"spec_calibrated_{m}_controls.npz")
        clave = "flux" if "flux" in z else list(z)[0]
        controles[m] = np.asarray(z[clave], dtype=np.float64)

    wave = np.asarray(espectros[CONTINUUM_METHODS[0]].wave_A, dtype=np.float64)

    filas = []
    for mi, mj in combinations(CONTINUUM_METHODS, 2):
        sigma = empirical_sigma_diff(controles[mi], controles[mj],
                                     smooth_channels=int(cfg.get("x10_sigma_smooth_channels", 5)))
        d, sig, ref, lam = bandas_de(wave,
                                     sigma,
                                     np.asarray(espectros[mi].flux, dtype=np.float64),
                                     np.asarray(espectros[mj].flux, dtype=np.float64))
        r = descompone(d, sig, ref, lam)
        if not r:
            continue
        r["par"] = f"{mi}_vs_{mj}"
        r["t_por_banda"] = {b.name: round(float(v), 2) for b, v in zip(BANDAS, d / sig)}
        filas.append(r)

    print(f"### {cfg['run_id']} — de que forma discrepan los metodos en el continuo\n")
    print(f"{'par':<32}{'desnivel':>10}{'z':>7}   {'deriva/1000A':>13}{'z':>7}   veredicto")
    for f in filas:
        za = f["alfa"] / f["alfa_sigma"] if f["alfa_sigma"] else float("nan")
        zb = f["beta"] / f["beta_sigma"] if f["beta_sigma"] else float("nan")
        if abs(za) > 3 and abs(za) > 2 * abs(zb): v = "DESNIVEL"
        elif abs(zb) > 3 and abs(zb) > 2 * abs(za): v = "CROMATICO"
        elif abs(za) > 3 and abs(zb) > 3: v = "los dos"
        else: v = "ninguno significativo"
        print(f"{f['par']:<32}{100*f['alfa']:9.1f}%{za:7.1f}   {100*f['beta']:12.1f}%{zb:7.1f}   {v}")

    if args.out:
        Path(args.out).write_text(json.dumps({"run_id": cfg["run_id"], "pares": filas}, indent=2), encoding="utf-8")
        print(f"\nescrito {args.out}")
    return filas


if __name__ == "__main__":
    main()
