#!/usr/bin/env python
"""De dónde sale el pedestal de `psffit`, y qué cuesta quitarlo.

`psffit` mide flujo positivo del compañero en posiciones de control donde no hay
nada (el «pedestal»). Esta sonda mide **la causa** y **el precio de corregirla**,
comparando tres diseños del ajuste sobre las mismas posiciones y los mismos
canales:

  * `lineal`      — el actual: PSF de la primaria, PSF del compañero, constante y
                    dos gradientes lineales (`psf_pair_design`, 5 columnas);
  * `cuadratico`  — más `y²`, `x²`, `xy` (3 grados de libertad extra);
  * `halo`        — más UNA columna: el perfil radial del residuo alrededor de la
                    primaria, que es la «corrección de halo» puesta en el diseño.

**La métrica no es el pedestal.** Medir sólo posiciones nulas hace que cualquier
término flexible parezca un éxito: quita el pedestal *y* la señal, porque el
compañero está **al mismo radio** que los controles y sobre un disco de 12 px una
fuente puntual y un fondo curvo son casi degenerados. Lo que se publica aquí es
el **exceso del compañero sobre su propia nula, dividido por el ruido entre
posiciones** — la única cantidad que dice si el estimador detecta mejor.

Medido el 2026-08-29 (`docs/2026-08-29_frente4_pedestal_y_overfit.md`): los tres
diseños alternativos **empeoran** esa SNR en los dos objetos. El pedestal ya está
tratado donde corresponde —la estandarización de E4 v4 resta la mediana de las
nulas al mismo radio, después del ajuste y sin coste en señal—.
"""

from __future__ import annotations

import argparse

import numpy as np
from astropy.io import fits

from musepipe.extraction.psffit import fit_region_mask, psf_pair_design
from musepipe.io import read_json
from musepipe.psf import evaluate_psf_model
from musepipe.stages.stage_h04_injection import (
    resolve_h04_positions, stage_h04_config_from_run, stage_h04_paths,
)

HALPHA_A = 6562.8
#: Los diseños que se comparan. `lineal` es el de produccion.
DISENOS = ("lineal", "cuadratico", "halo")


def _ajuste_ponderado(data, var, design, mask):
    A = design[mask]
    y = data[mask]
    w = 1.0 / np.sqrt(np.maximum(var[mask], 1e-12))
    ok = np.isfinite(y) & np.isfinite(w) & np.all(np.isfinite(A), axis=1)
    if int(ok.sum()) < A.shape[1] + 5:
        return np.full(A.shape[1], np.nan)
    coef, *_ = np.linalg.lstsq(A[ok] * w[ok, None], y[ok] * w[ok], rcond=None)
    return coef


def perfil_de_halo(img, star_yx, posiciones, r_star, psf_img):
    """El residuo radial alrededor de la primaria, con las fuentes enmascaradas.

    Enmascarar **no es opcional**: sin ello el perfil lleva dentro al propio
    compañero, y el término se lo come por construccion. Medido, además, que
    enmascararlo **no salva el resultado** — el término sigue comiéndose la señal,
    así que la degeneracion no venia de la contaminacion.
    """

    m0 = (r_star <= 20.0) & np.isfinite(img) & np.isfinite(psf_img)
    A = np.vstack([psf_img[m0], np.ones(int(m0.sum()))]).T
    coef, *_ = np.linalg.lstsq(A, img[m0], rcond=None)
    resid = img - (coef[0] * psf_img + coef[1])
    yy, xx = np.indices(img.shape, dtype=np.float64)
    libre = np.isfinite(resid)
    for q in posiciones:
        libre &= np.hypot(yy - q["y"], xx - q["x"]) > 9.0
    bordes = np.arange(0.0, float(r_star.max()) + 4.0, 4.0)
    perfil = np.full(bordes.size - 1, np.nan)
    for i in range(bordes.size - 1):
        m = (r_star >= bordes[i]) & (r_star < bordes[i + 1]) & libre
        if int(m.sum()) > 20:
            perfil[i] = float(np.nanmedian(resid[m]))
    centros = 0.5 * (bordes[:-1] + bordes[1:])
    bien = np.isfinite(perfil)
    imagen = np.interp(r_star, centros[bien], perfil[bien])
    return imagen / max(float(np.nanmax(np.abs(imagen))), 1e-9)


def amplitud(cube, stat, wave, star_yx, pos_yx, modelo, diseno, extra, cfg, shape):
    ny, nx = shape
    mask = fit_region_mask(
        (ny, nx), star_yx, pos_yx,
        star_radius_px=float(cfg.get("x03_star_radius_px", 20.0)),
        comp_radius_px=float(cfg.get("x03_comp_radius_px", 12.0)),
    )
    valores = []
    for z in range(cube.shape[0]):
        D = psf_pair_design((ny, nx), wave[z], star_yx, pos_yx, modelo)
        if diseno != "lineal":
            D = np.concatenate([D, extra], axis=-1)
        valores.append(_ajuste_ponderado(cube[z], stat[z], D, mask)[1])
    return float(np.nanmedian(valores))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", required=True, help="lista separada por comas")
    ap.add_argument("--media-banda-A", type=float, default=60.0,
                    help="media anchura de la ventana alrededor de Halfa")
    args = ap.parse_args(argv)

    for run in [r.strip() for r in args.runs.split(",") if r.strip()]:
        cfg = stage_h04_config_from_run(run)
        paths = stage_h04_paths(run, cfg["project_root"])
        stage_dir = paths["paths"].stage_dir
        with fits.open(paths["stage02_cube_fits"], memmap=True) as hdul:
            wave = np.asarray(hdul["WAVELENGTH"].data, dtype=np.float64)
            sel = np.abs(wave - HALPHA_A) <= args.media_banda_A
            def corte(nombre):
                d = hdul[nombre].data
                return np.asarray((d if d.ndim == 3 else d[0])[sel], dtype=np.float64)
            cube, stat = corte("CUBES"), corte("STAT")
        w = wave[sel]
        qc = read_json(stage_dir / "stage01c_qc.json")
        star = next(tuple(map(float, qc[k]["pos_yx"])) for k in ("primary", "star", "m3_primary")
                    if isinstance(qc.get(k), dict) and qc[k].get("pos_yx"))
        modelo = read_json(stage_dir / "psf_model.json")
        posiciones = resolve_h04_positions(cfg, paths)
        controles = [p for p in posiciones if p["label"].startswith("control")]
        real = next(p for p in posiciones if p["label"] == "real")

        ny, nx = cube.shape[1:]
        yy, xx = np.indices((ny, nx), dtype=np.float64)
        r_star = np.hypot(yy - star[0], xx - star[1])
        psf_img = evaluate_psf_model(modelo, HALPHA_A, yy - star[0], xx - star[1])
        halo = perfil_de_halo(np.nanmedian(cube, axis=0), star, posiciones, r_star, psf_img)

        print(f"\n=== {run}: {len(controles)} controles, {cube.shape[0]} canales ===")
        print(f"{'diseno':<12}{'pedestal':>10}{'companero':>11}{'exceso':>9}{'ruido':>9}{'SNR':>7}")
        base = None
        for diseno in DISENOS:
            if diseno == "cuadratico":
                y0, x0 = 0.5 * (star[0] + real["y"]), 0.5 * (star[1] + real["x"])
                sc = max(float(np.hypot(real["y"] - star[0], real["x"] - star[1])), 1.0)
                ys, xs = (yy - y0) / sc, (xx - x0) / sc
                extra = np.stack([ys ** 2, xs ** 2, ys * xs], axis=-1)
            else:
                extra = halo[..., None]
            nulos = np.asarray(
                [amplitud(cube, stat, w, star, (p["y"], p["x"]), modelo, diseno, extra, cfg, (ny, nx))
                 for p in controles], dtype=np.float64)
            nulos = nulos[np.isfinite(nulos)]
            senal = amplitud(cube, stat, w, star, (real["y"], real["x"]), modelo, diseno, extra, cfg, (ny, nx))
            pedestal, ruido = float(np.median(nulos)), float(nulos.std())
            exceso = senal - pedestal
            snr = exceso / ruido if ruido else np.nan
            if base is None:
                base = snr
            print(f"{diseno:<12}{pedestal:>+10.2f}{senal:>+11.2f}{exceso:>+9.2f}{ruido:>9.2f}{snr:>+7.2f}"
                  + ("" if diseno == "lineal" else f"   ({snr - base:+.2f} frente al lineal)"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
