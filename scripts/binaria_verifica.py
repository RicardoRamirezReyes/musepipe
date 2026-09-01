#!/usr/bin/env python
"""Las dos verificaciones del protocolo de parada de la binaria (spec C1 §5.4).

  * **Control**: se le pide la MISMA binaria a un objeto que es una estrella
    sola. Si el metodo tambien «encuentra» una secundaria ahi, esta inventando.
  * **Barrido de geometria**: se escanean rho y PA con la posicion FIJA en cada
    punto —no ajustada, no se gasta ningun grado de libertad— y se mira si el
    minimo cae donde dice la astrometria publicada. Eso responde si NUESTROS
    datos sostienen la binaria ahi, en vez de que se lo hayamos impuesto.

Usa la rama Moffat, que es barata; la geometria es una propiedad del cielo, no
de la forma funcional.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from astropy.io import fits

ROOT = Path(__file__).resolve().parents[1]
import sys  # noqa: E402

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from musepipe.psf import (  # noqa: E402
    evaluate_moffat_scene,
    fit_moffat_image,
    source_mask,
)


def coste(img, qc, fit, radio, mascara):
    """Suma de residuos al cuadrado sobre EL MISMO conjunto de pixeles.

    **No se usa `chi2r`**: normaliza por la sigma robusta del propio residuo, asi
    que sube cuando el ajuste mejora y no compara dos ajustes entre si. Es la
    misma trampa que ya aparecio al escribir los tests de esta rama, y colarse
    en ella aqui daba un barrido de PA con el minimo 32 grados fuera.
    """

    py, px = qc["primary"]["pos_yx"]
    yy, xx = np.indices(img.shape, dtype=np.float64)
    dentro = (np.hypot(yy - py, xx - px) <= radio) & ~mascara & np.isfinite(img)
    resid = img - evaluate_moffat_scene(img.shape, fit)
    return float(np.nansum(resid[dentro] ** 2))


def offset_px(sep_mas, pa_deg, escala_arcsec):
    """PA se mide del Norte hacia el Este; Norte es +y y Este es -x."""
    sep = float(sep_mas) / (float(escala_arcsec) * 1000.0)
    pa = math.radians(float(pa_deg))
    return (sep * math.cos(pa), -sep * math.sin(pa))


def imagen(run, banda):
    qc = json.loads((ROOT / f"runs/{run}/stages/stage01c_qc.json").read_text())
    with fits.open(ROOT / f"runs/{run}/stages/stage02_xcorr_cube_stack.fits", memmap=True) as h:
        wave = np.asarray(h["WAVELENGTH"].data, dtype=np.float64)
        cubo = h["CUBES"].data
        cubo = cubo[0] if cubo.ndim == 4 else cubo
        sel = (wave >= banda[0]) & (wave <= banda[1])
        img = np.nanmedian(np.asarray(cubo[sel], dtype=np.float64), axis=0)
    return qc, img


def _mascara(img, qc):
    centros = [tuple(qc["companion"]["pos_yx"])]
    if qc.get("field_source"):
        centros.append(tuple(qc["field_source"]["pos_yx"]))
    return source_mask(img.shape, centros, 12.0)


def ajusta(img, qc, offset, radio):
    """Ajuste de una configuracion, con el coste sobre EL MISMO conjunto de pixeles.

    `sigma_clip=None` no es una preferencia: el recorte **cambia que pixeles
    entran** en cada ajuste, asi que dos configuraciones con recortes distintos
    no se pueden comparar por su residuo. Fijando el conjunto, la unica
    diferencia entre puntos del barrido es la geometria, que es lo que se quiere
    medir.
    """

    py, px = qc["primary"]["pos_yx"]
    fit = fit_moffat_image(img, center_yx=(py, px), fit_radius_px=radio,
                           mask=_mascara(img, qc), companion_offset_yx=offset,
                           sigma_clip=None)
    return fit, coste(img, qc, fit, radio, _mascara(img, qc))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="run con la binaria declarada")
    ap.add_argument("--control", required=True, help="run de una estrella SOLA")
    ap.add_argument("--radio", type=float, default=9.0)
    ap.add_argument("--banda", type=float, nargs=2, default=[7300, 7800])
    args = ap.parse_args(argv)

    qc, img = imagen(args.run, args.banda)
    decl = json.loads((ROOT / f"runs/{args.run}/config/config.json").read_text())["config"]
    binario = decl["e01_binary_companion"]
    escala = float(qc["pixel_scale_arcsec"])
    sep0, pa0 = float(binario["sep_mas"]), float(binario["pa_deg"])

    print(f"=== {args.run} · declarado rho={sep0} mas, PA={pa0} deg ===")
    base, coste0 = ajusta(img, qc, offset_px(sep0, pa0, escala), args.radio)
    print(f"  f = {base.flux_ratio:.4f}   coste = {coste0:.6g}")

    print("\n--- barrido de PA (rho fijo en el declarado) ---")
    mejor_pa, mejor = None, np.inf
    for pa in range(0, 360, 20):
        fit, c = ajusta(img, qc, offset_px(sep0, pa, escala), args.radio)
        marca = "  <-- declarado" if abs((pa - pa0 + 180) % 360 - 180) <= 10 else ""
        print(f"  PA={pa:3d}  coste={c:12.6g}  f={fit.flux_ratio:.4f}{marca}")
        if c < mejor:
            mejor, mejor_pa = c, pa
    d = abs((mejor_pa - pa0 + 180) % 360 - 180)
    print(f"  minimo en PA={mejor_pa} deg, a {d:.0f} deg del declarado ({pa0})")

    print("\n--- barrido de rho (PA fijo en el declarado) ---")
    mejor_sep, mejor = None, np.inf
    for sep in (0.0, 20.0, 35.0, 51.0, 70.0, 90.0, 120.0):
        fit, c = (ajusta(img, qc, offset_px(sep, pa0, escala), args.radio) if sep
                  else ajusta(img, qc, None, args.radio))
        f = "-" if fit.flux_ratio is None else f"{fit.flux_ratio:.4f}"
        marca = "  <-- declarado" if sep == sep0 else ("  <-- sin binaria" if sep == 0 else "")
        print(f"  rho={sep:5.0f} mas  coste={c:12.6g}  f={f}{marca}")
        if sep and c < mejor:
            mejor, mejor_sep = c, sep
    print(f"  minimo en rho={mejor_sep:.0f} mas (declarado {sep0:.0f})")

    print(f"\n=== CONTROL: {args.control}, estrella sola, misma geometria ===")
    qc_c, img_c = imagen(args.control, args.banda)
    ctrl, _ = ajusta(img_c, qc_c, offset_px(sep0, pa0, float(qc_c["pixel_scale_arcsec"])), args.radio)
    print(f"  f = {ctrl.flux_ratio:.4f}   (suelo medido del metodo: ~0.02)")
    print(f"\n  veredicto: f del objeto / f del control = "
          f"{base.flux_ratio / max(ctrl.flux_ratio, 1e-6):.1f}x")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
