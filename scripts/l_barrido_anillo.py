"""¿El desacuerdo entre fondos escala con el tamaño del anillo? La ley, o nada.

`docs/2026-09-12_no_es_el_instrumento_es_la_noche.md` §7.1 dejó una explicación
geométrica con **dos puntos**: el anillo de fondo tiene radio FIJO (8-14 px), así
que la fracción del radio estelocéntrico que abarca depende de la separación
—±20 % en ROXs 12 b (71.2 px), ±30 % en ROXs 42B b (46.7 px)— y el objeto con la
compañera más cerca es el que más desacuerdo tiene (1.24 contra 1.06).

Dos puntos son una tendencia, no una ley. Esto la prueba **dentro de un mismo
objeto**, barriendo `r_out` y midiendo la razón `azimuthal / annulus` en la
posición de la compañera.

**Predicción declarada antes de medir**: si la geometría es la causa, la razón
tiene que **crecer monótonamente con `r_out / separación`**, y las curvas de los
dos objetos tienen que **caer una sobre otra** al dibujarlas contra esa variable
—no contra `r_out` en píxeles—. Si la razón es plana, o si cada objeto va por su
lado, la explicación geométrica es falsa y el desacuerdo viene de otra cosa.

    python scripts/l_barrido_anillo.py --run <run> --out-json <out>.json
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
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from i_inyeccion_continuo import load_extraction_cube  # noqa: E402
from musepipe.extraction.optimal import make_optimal_product  # noqa: E402
from musepipe.stages import stage_x02_optimal as x02  # noqa: E402
from musepipe.stages.stage_x01_aperture import _load_positions  # noqa: E402


def nivel(cube, wave, centro, estrella, psf_model, cfg, mode, anillo):
    ext = make_optimal_product(
        cube, wave, centro, psf_model,
        star_yx=estrella, run_id=str(cfg["run_id"]), input_cube_path="barrido_anillo",
        variant=f"anillo_{mode}", error_mode="empirical",
        aperture_correction=cfg.get("x02_aperture_correction", "auto"),
        window_radius_px=float(cfg.get("x02_window_radius_px", 8.0)),
        clip_sigma=float(cfg.get("x02_clip_sigma", 4.0)),
        clip_max_iter=int(cfg.get("x02_clip_max_iter", 2)),
        n_controls=0, local_bkg_annulus_px=anillo, background_mode=mode,
        azimuthal_width_px=float(cfg.get("x02_azimuthal_width_px", 3.0)),
        azimuthal_exclude_px=float(cfg.get("x02_azimuthal_exclude_px", 10.0)),
        plane_fit_radius_px=float(cfg.get("x02_plane_fit_radius_px", 14.0)),
        plane_mask_radius_px=float(cfg.get("x02_plane_mask_radius_px", 3.0)),
        n_jobs=1,
    )
    return np.asarray(ext.product.flux, dtype=np.float64)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--project-root", default=None)
    ap.add_argument("--r-out", nargs="+", type=float, default=[10, 12, 14, 18, 22, 26])
    ap.add_argument("--r-in", type=float, default=8.0)
    ap.add_argument("--channel-step", type=int, default=60)
    ap.add_argument("--banda", nargs=2, type=float, default=[8600.0, 9100.0])
    ap.add_argument("--out-json", required=True)
    args = ap.parse_args(argv)

    root = Path(args.project_root or Path.cwd()).resolve()
    cfg = x02.stage_x02_config_from_run(args.run, project_root=root)
    paths = x02.stage_x02_paths(args.run, root)
    obj, star, _p, _q = _load_positions(paths, cfg)
    psf_model, _mp = x02._load_psf_model(paths, cfg)
    cube, wave, etiqueta = load_extraction_cube(paths, cfg)
    step = max(1, int(args.channel_step))
    cube, wave = cube[::step], wave[::step]
    sep = float(np.hypot(obj[0] - star[0], obj[1] - star[1]))
    lo, hi = float(args.banda[0]), float(args.banda[1])
    msk = (wave >= lo) & (wave <= hi)
    exterior = cfg.get("x02_local_bkg_annulus_px")
    borde = float(exterior[2]) if exterior and len(exterior) > 2 else 30.0
    print(f"{args.run}: separacion {sep:.2f} px | cubo {etiqueta}", flush=True)

    filas = []
    print(f"\n{'r_out':>7s} {'r_out/sep':>10s} {'annulus':>11s} {'azimuthal':>11s} {'azi/ann':>9s}")
    for r_out in args.r_out:
        anillo = [float(args.r_in), float(r_out), borde]
        n_ann = float(np.nanmedian(nivel(cube, wave, obj, star, psf_model, cfg, "annulus", anillo)[msk]))
        n_azi = float(np.nanmedian(nivel(cube, wave, obj, star, psf_model, cfg, "azimuthal", anillo)[msk]))
        razon = n_azi / n_ann if n_ann else np.nan
        filas.append({"r_in": float(args.r_in), "r_out": float(r_out), "r_out_sobre_sep": r_out / sep,
                      "nivel_annulus": n_ann, "nivel_azimuthal": n_azi, "azi_sobre_ann": razon})
        print(f"{r_out:7.1f} {r_out / sep:10.3f} {n_ann:11.2f} {n_azi:11.2f} {razon:9.4f}", flush=True)
        Path(args.out_json).write_text(json.dumps(
            {"script": "l_barrido_anillo", "run": args.run, "separacion_px": sep,
             "cubo": etiqueta, "banda_A": [lo, hi], "filas": filas}, indent=1), encoding="utf-8")

    r = np.array([f["azi_sobre_ann"] for f in filas])
    x = np.array([f["r_out_sobre_sep"] for f in filas])
    ok = np.isfinite(r)
    if ok.sum() > 2:
        pend = np.polyfit(x[ok], r[ok], 1)[0]
        monotona = bool(np.all(np.diff(r[ok]) > 0) or np.all(np.diff(r[ok]) < 0))
        print(f"\npendiente d(razon)/d(r_out/sep) = {pend:+.3f} | monotona: {monotona}")
    print("\nCrece con r_out/sep y las dos curvas coinciden -> la geometria es la causa.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
