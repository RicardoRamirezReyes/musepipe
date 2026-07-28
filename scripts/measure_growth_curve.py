#!/usr/bin/env python3
"""Curva de crecimiento EMPIRICA de la primaria, para llegar al flujo total.

Mide, sobre el cubo grande (el de A2, antes del recorte de B1), cuanta luz
anade cada anillo concentrico alrededor de la estrella, resta el cielo y suma
los incrementos hasta que se hacen cero. El resultado es el factor que falta
para convertir la convencion actual de `apcorr` en flujo total.

## Por que hace falta

`apcorr` normaliza la PSF a 1 dentro de `norm_radius_px` (25 px = 0.63"), asi
que su "flujo total" es en realidad "flujo dentro de 0.63"". Medido aqui sobre
ROXs12b_realigned, fuera de ese radio queda entre el 49% (azul) y el 48% (rojo)
de la luz: el factor que falta es 1.9-2.5 y **es cromatico**, o sea que no se
cancela y deforma la pendiente del continuo.

## El metodo

1. Perfil radial en anillos de 1 px: **mediana azimutal**, robusta frente al
   companero y a cualquier otra fuente del campo.
2. Ajuste `SB(r) = A*r**-p + S` en un rango exterior. `S` es el suelo de fondo
   residual y **hay que medirlo**: el DRS sobre-resta cielo y `S` sale NEGATIVO
   (-0.2 a -0.9 en ROXs12b). Suponerlo cero infla el flujo total, porque el
   suelo multiplica por los ~83000 spaxels de r<=163 px.
3. Suma directa de los pixeles (menos `S`) hasta el ultimo anillo COMPLETO,
   mas la cola analitica del halo mas alla del campo,
   `2*pi*A*R**(2-p)/(p-2)`, que converge porque el halo AO va como r^-3.1..3.4.

## El limite honesto

Dentro del campo NO hay ningun radio donde el halo sea despreciable: iguala a
|S| hacia r ~ 170 px y el ultimo anillo completo esta en 163. Cielo y halo se
ajustan a la vez y quedan parcialmente degenerados, asi que **el rango del
ajuste es la barra de error dominante**, no el ruido: mover el borde interior
de 40 a 80 px mueve el factor un +-12% en el azul y un +-5% en el rojo. Por eso
el script recorre varios rangos y reporta la dispersion como sistematico.

Validacion: comparando manzanas con manzanas (`F(<=78)/F(<=25)`), el modelo
psfao de C1 reproduce estos datos al 1.3-2.3%. El modelo esta bien; lo que
estaba mal era donde se ponia el "1".

## Uso

    python scripts/measure_growth_curve.py --run-id ROXs12b_realigned
    python scripts/measure_growth_curve.py --run-id ROXs12b_realigned \\
        --out-dir /tmp/gc --n-bands 8 --fit-range 60 235

Es de SOLO LECTURA sobre `runs/`: escribe figura y JSON donde se le diga
(`--out-dir`, por defecto un directorio temporal), nunca dentro del run.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

# Ejecutable desde cualquier cwd: `scripts/` no esta en el path por si solo.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from astropy.io import fits

# El nucleo numerico vive en `musepipe/growth_curve.py` para que A2 y este
# script midan EXACTAMENTE lo mismo: son la misma medida y dos copias
# divergirian. Aqui solo queda el CLI y la presentacion.
from musepipe.growth_curve import (  # noqa: E402
    DEFAULT_FIT_FRACTION,
    measure_growth_curve,
)


def _resolve_cube(run_id, project_root, explicit):
    if explicit:
        return Path(explicit)
    from musepipe.config import load_run_config

    run = load_run_config(run_id, project_root=project_root)
    files = run.config.get("cube_files") or []
    if not files:
        raise SystemExit(
            "El run no declara `cube_files`; pasa el cubo grande de A2 con --cube."
        )
    return Path(files[0])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--run-id")
    parser.add_argument("--project-root")
    parser.add_argument("--cube", help="cubo grande de A2; por defecto config.cube_files[0]")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--write-run-product", action="store_true",
                        help="escribe stages/growth_curve_qc.json del run (lo que consume C2/C3/C4)")
    parser.add_argument("--n-bands", type=int, default=8)
    parser.add_argument("--fit-fraction", type=float, nargs=2, default=list(DEFAULT_FIT_FRACTION),
                        help="rango de ajuste del cielo, en fracciones del ultimo anillo completo")
    parser.add_argument("--companion-yx", type=float, nargs=2, default=None,
                        help="posicion del companero EN EL CUBO GRANDE, para enmascararlo")
    args = parser.parse_args(argv)

    cube = _resolve_cube(args.run_id, args.project_root, args.cube)
    with fits.open(cube, memmap=True) as hdul:
        header = hdul[1].header
        nz = int(header["NAXIS3"])
        cdelt = float(header.get("CD3_3") or header["CDELT3"])
        wave = float(header["CRVAL3"]) + (np.arange(nz) + 1 - float(header["CRPIX3"])) * cdelt
        pix = abs(float(header.get("CD1_1") or header["CDELT1"])) * 3600.0
        result = measure_growth_curve(
            hdul[1].data, wave, n_bands=args.n_bands, fit_fraction=tuple(args.fit_fraction),
            companion_yx=args.companion_yx, pixel_scale_arcsec=pix,
        )

    print(f"cubo   : {cube}")
    print(f"estrella en yx = {[round(v, 2) for v in result['star_yx']]}"
          f" | ultimo anillo completo r = {result['r_last_complete_annulus_px']:.0f} px")
    print(f"ajuste del cielo en r = {result['fit_range_px']} px"
          f"  (fraccion {args.fit_fraction} del ultimo anillo completo)\n")
    print(f"  {'lambda':>8s} {'factor':>8s} {'sistematico':>16s} {'p halo':>7s} {'suelo':>8s} {'cola':>6s}")
    for row in result["bands"]:
        print(f"  {row['wave_A']:8.0f} {row['ratio_total_over_normrad']:8.3f}"
              f"   {row['ratio_min']:.3f} - {row['ratio_max']:.3f}"
              f"   {row['halo_power']:6.3f} {row['sky_floor']:+8.4f}"
              f" {100 * row['tail_fraction']:5.1f}%")

    if args.write_run_product:
        # Producto de run: A2 lo emite cuando corre, pero los runs con perfil
        # `cascade` no ejecutan A2 y los antiguos corrieron antes de que esto
        # existiera. Sin esta salida no habria forma de activar la convencion
        # nueva en un run ya reducido.
        from musepipe.config import load_run_config
        from musepipe.growth_curve import RUN_PRODUCT_NAME

        run = load_run_config(args.run_id, project_root=args.project_root)
        target = run.paths.stage_dir / RUN_PRODUCT_NAME
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = dict(result)
        payload["source"] = "scripts/measure_growth_curve.py"
        payload["cube"] = str(cube)
        target.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
        print(f"\nproducto de run -> {target}")

    out_dir = Path(args.out_dir) if args.out_dir else None
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        payload = out_dir / "growth_curve.json"
        payload.write_text(json.dumps(result, indent=1), encoding="utf-8")
        print(f"\nJSON -> {payload}")
    return result


if __name__ == "__main__":
    main()
