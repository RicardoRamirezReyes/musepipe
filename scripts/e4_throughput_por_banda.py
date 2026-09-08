#!/usr/bin/env python
"""¿Es cromatico el throughput de E4? Inyeccion en seis longitudes de onda.

E4 mide la recuperacion con el filtro adaptado de E1 **en Halpha**: su throughput
es un numero por metodo, medido en UNA longitud de onda. El 2026-09-08 se midio
que el sesgo de los estimadores recorre de −6 % a +9 % a lo largo de la banda
(`docs/2026-09-08_d1_tension_continuo_medida.md`), asi que un escalar no puede
corregirlo. Esto lo comprueba donde importa: **en la posicion del compañero**.

Por cada banda de continuo de D1 se inyecta una linea gaussiana de flujo
conocido en la posicion del compañero, con la misma maquinaria que usa E4
(`musepipe.injection.inject`, convencion `norm_radius`, la PSF del run evaluada
en esa lambda), y se extrae con los estimadores **antes y despues**. La
diferencia, integrada en la linea, es lo recuperado.

    throughput(banda) = recuperado / inyectado

Se trabaja sobre **rodajas** de +-60 A alrededor de cada banda: los extractores
van canal a canal, asi que el resultado es identico y cuesta 45 veces menos que
recorrer los 3681 canales.

Con `--psf-empirica` se inyecta, en vez del modelo, **la PSF medida sobre la
primaria** en esa misma banda. Esa es la version que importa: inyectar el modelo
mide la coherencia del estimador consigo mismo, no su sesgo contra la realidad —y
es lo que hace E4, asi que su throughput es **ciego al error del modelo por
construccion**—. Para poder pegar la PSF empirica sin interpolar, se inyecta en
una posicion con el **mismo desplazamiento subpixel que la estrella** y el mismo
radio que el compañero.

**No escribe en `runs/`.**
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

from musepipe.injection import inject  # noqa: E402
from musepipe.extraction.psffit import fit_psffit_cube  # noqa: E402
from musepipe.psf import evaluate_psf_model  # noqa: E402
from musepipe.stages.stage_x01_aperture import _load_positions, _load_stat_cube  # noqa: E402
from musepipe.stages.stage_x02_optimal import (  # noqa: E402
    _load_stage02_cube,
    stage_x02_config_from_run,
    stage_x02_paths,
)
from musepipe.stages.stage_x10_compare import COMPARISON_BANDS  # noqa: E402

BANDAS = [b for b in COMPARISON_BANDS if b.kind == "continuum"]


def estimadores(cube, var, wave, psf, cy, cx, sy, sx):
    """(aperture, optimal, psffit) canal a canal, en la escala `norm_radius`."""
    ny, nx = cube.shape[1:]
    yy, xx = np.indices((ny, nx), dtype=np.float64)
    r = np.hypot(yy - cy, xx - cx)
    iy, ix = int(round(cy)), int(round(cx))
    box3 = (np.abs(yy - iy) <= 1) & (np.abs(xx - ix) <= 1)
    win = r <= 8.0
    r25 = r <= 25.0
    ap, op = np.zeros(len(wave)), np.zeros(len(wave))
    cache = {}
    for z, lam in enumerate(wave):
        clave = round(float(lam) / 50.0)
        if clave not in cache:
            P = evaluate_psf_model(psf, float(lam), yy - cy, xx - cx)
            tot = float(np.nansum(P[r25]))
            cache[clave] = (P, float(np.nansum(P[box3])) / tot, float(np.nansum(P[win])) / tot)
        P, e_box3, e_r8 = cache[clave]
        ap[z] = np.nansum(cube[z][box3]) / e_box3
        p = np.clip(P[win], 0, None); p = p / p.sum()
        d, v = cube[z][win], np.clip(var[z][win], 1e-30, None)
        op[z] = float(np.nansum(p * d / v) / np.nansum(p * p / v)) / e_r8
    fit = fit_psffit_cube(cube, var, wave, (sy, sx), (cy, cx), psf,
                          star_radius_px=20.0, comp_radius_px=12.0)
    return {"aperture": ap, "optimal": op, "psffit": fit.coeffs[:, 1]}


def psf_empirica(cube, wave, banda, sy, sx, anillo, r_estrella, norm_radius):
    """La PSF de la primaria medida en esa banda, sin fondo y normalizada a 1."""
    img = np.nanmedian(cube[(wave >= banda.lo_A) & (wave < banda.hi_A)], axis=0)
    img = img - np.nanmedian(img[anillo])
    emp = np.zeros_like(img)
    dentro = r_estrella <= norm_radius
    emp[dentro] = img[dentro]
    return emp / emp.sum()


def main(argv=None):
    ap_ = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap_.add_argument("--run-id", required=True)
    ap_.add_argument("--project-root", default=None)
    ap_.add_argument("--snr", type=float, default=50.0, help="amplitud de la inyeccion, en sigma del continuo local")
    ap_.add_argument("--psf-empirica", action="store_true",
                     help="inyecta la PSF medida sobre la primaria en vez del modelo (la version que importa)")
    ap_.add_argument("--out", default=None)
    args = ap_.parse_args(argv)

    cfg = stage_x02_config_from_run(args.run_id, project_root=args.project_root)
    paths = stage_x02_paths(cfg["run_id"], Path(cfg["project_root"]))
    comp, star, _p, _q = _load_positions(paths, cfg)
    psf = json.load(open(paths["psf_model_json"]))
    cube, wave, _c, _b = _load_stage02_cube(paths, cfg)
    stat, _e = _load_stat_cube(paths, cfg, cube.shape)
    if stat is None:
        raise RuntimeError("hace falta el STAT: la optimal pesa por 1/varianza.")
    lsf = float(cfg.get("h01_lsf_fwhm_A") or cfg.get("lsf_fwhm_A") or 2.3)
    norm_radius = float(psf.get("norm_radius_px", 25.0))
    cy, cx = float(comp[0]), float(comp[1])
    sy, sx = float(star[0]), float(star[1])

    ny, nx = cube.shape[1:]
    yy, xx = np.indices((ny, nx), dtype=np.float64)
    r_estrella = np.hypot(yy - sy, xx - sx)
    anillo = (r_estrella > 40) & (r_estrella < 60)
    if args.psf_empirica:
        # Mismo radio que el compañero y mismo subpixel que la estrella: asi la
        # PSF empirica se traslada con `np.roll`, sin interpolar y sin suavizar.
        cy, cx = sy + int(round(cy - sy)), sx + int(round(cx - sx))

    filas = []
    modo = "PSF EMPIRICA de la primaria" if args.psf_empirica else "modelo (lo que hace E4)"
    print(f"### {cfg['run_id']} — throughput por banda, inyectando {modo}\n")
    print(f"{'banda':<6}{'lambda':>7}   {'aperture':>10}{'optimal':>10}{'psffit':>10}")
    for b in BANDAS:
        lam = 0.5 * (b.lo_A + b.hi_A)
        s = (wave >= lam - 60) & (wave <= lam + 60)
        w = wave[s]
        base, var = cube[s], stat[s]
        # amplitud: `snr` veces la dispersion del continuo local en una caja box3
        ruido = np.nanstd([np.nansum(base[z][int(round(cy))-1:int(round(cy))+2,
                                             int(round(cx))-1:int(round(cx))+2])
                           for z in range(len(w))])
        f_iny = float(args.snr * ruido * lsf)
        if args.psf_empirica:
            emp = psf_empirica(cube, wave, b, sy, sx, anillo, r_estrella, norm_radius)
            emp = np.roll(np.roll(emp, int(round(cy - sy)), axis=0), int(round(cx - sx)), axis=1)
            perfil = np.exp(-0.5 * ((w - lam) / (lsf / 2.355)) ** 2)
            perfil = perfil / np.sum(perfil * np.gradient(w))
            inyectado = base + f_iny * perfil[:, None, None] * emp[None, :, :]
        else:
            inyectado = inject(base, [{"y": cy, "x": cx, "total_line_flux": f_iny,
                                       "line_center_A": lam, "line_fwhm_A": lsf}],
                               wavelengths_A=w, psf_model=psf,
                               norm_convention="norm_radius", norm_radius_px=norm_radius)
        antes = estimadores(base, var, w, psf, cy, cx, sy, sx)
        despues = estimadores(inyectado, var, w, psf, cy, cx, sy, sx)
        linea = np.abs(w - lam) <= 1.5 * lsf
        dw = np.gradient(w)
        fila = {"banda": b.name, "lambda_A": lam, "flujo_inyectado": f_iny}
        for m in ("aperture", "optimal", "psffit"):
            rec = float(np.nansum(((despues[m] - antes[m]) * dw)[linea]))
            fila[m] = rec / f_iny
        filas.append(fila)
        print(f"{b.name:<6}{lam:7.0f}   {fila['aperture']:10.3f}{fila['optimal']:10.3f}{fila['psffit']:10.3f}")

    print("\n  variacion a lo largo de la banda (max-min):")
    for m in ("aperture", "optimal", "psffit"):
        v = np.array([f[m] for f in filas])
        print(f"    {m:<10} {v.min():.3f} a {v.max():.3f}   ->  {100*(v.max()-v.min()):.1f} puntos")
    if args.out:
        Path(args.out).write_text(json.dumps({"run_id": cfg["run_id"], "bandas": filas}, indent=2),
                                  encoding="utf-8")
    return filas


if __name__ == "__main__":
    main()
