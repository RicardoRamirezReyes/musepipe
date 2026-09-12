"""¿Cuánto continuo se come el fondo? Inyección-recuperación, con verdad conocida.

`docs/2026-09-11_el_fondo_es_la_palanca.md` §7 midió que los tres tratamientos de
fondo de C3 discrepan entre sí en el continuo de la compañera entre un 10 % y un
39 %, con fracción casi constante en λ. Pero eso es un cociente ENTRE MÉTODOS:
ninguno es la verdad, y el cociente no dice cuánto se come cada uno.

Esto lo mide con verdad conocida, que es lo que E4 hace para la LÍNEA y nadie
había hecho para el CONTINUO: se inyecta un continuo plano de amplitud conocida a
la misma separación de la primaria que la compañera -en ángulos de posición
libres- y se mide qué fracción sobrevive a cada modo de fondo.

**Diferencial a propósito.** Se extrae dos veces en cada posición, con y sin
inyección, y se mide `(con - sin) / inyectado`. Así se cancela lo que ya hubiera
allí (halo, residuo de cielo) y no hay que casar convenciones de apcorr: la
fracción recuperada es un cociente entre dos cantidades de la misma extracción.

    python scripts/i_inyeccion_continuo.py --run <run> --out-json <out>.json

Lectura: si el fondo no se comiera nada, la recuperación sería 1.0 en todo λ. Lo
que salga por debajo es lo que el tratamiento de fondo se lleva del continuo.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
import sys  # noqa: E402

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from musepipe.apertures import same_radius_control_positions  # noqa: E402
from musepipe.extraction.optimal import BACKGROUND_MODES, make_optimal_product  # noqa: E402
from musepipe.psf import evaluate_psf_model  # noqa: E402
from musepipe.stages import stage_x02_optimal as x02  # noqa: E402
from musepipe.stages.stage_x01_aperture import _load_positions, _load_residual_cube  # noqa: E402


def inject(cube, wave, psf_model, center_yx, amplitude):
    """Fuente puntual de continuo PLANO con la PSF del run, canal a canal."""
    ny, nx = cube.shape[1], cube.shape[2]
    yy, xx = np.indices((ny, nx), dtype=np.float64)
    out = np.array(cube, dtype=np.float64, copy=True)
    for k, w in enumerate(wave):
        psf = evaluate_psf_model(psf_model, float(w), yy - center_yx[0], xx - center_yx[1])
        out[k] += float(amplitude) * psf
    return out


def extract(cube, wave, center_yx, star_yx, psf_model, cfg, mode):
    ext = make_optimal_product(
        cube, wave, center_yx, psf_model,
        star_yx=star_yx, run_id=str(cfg["run_id"]), input_cube_path="inyeccion_continuo",
        variant=f"inj_{mode}",
        error_mode="empirical",
        aperture_correction=cfg.get("x02_aperture_correction", "auto"),
        window_radius_px=float(cfg.get("x02_window_radius_px", 8.0)),
        clip_sigma=float(cfg.get("x02_clip_sigma", 4.0)),
        clip_max_iter=int(cfg.get("x02_clip_max_iter", 2)),
        n_controls=0,
        local_bkg_annulus_px=cfg.get("x02_local_bkg_annulus_px"),
        background_mode=mode,
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
    ap.add_argument("--modes", nargs="+", default=list(BACKGROUND_MODES))
    ap.add_argument("--n-positions", type=int, default=4,
                    help="angulos de posicion; 1 no sirve, la del compañero se excluye")
    ap.add_argument("--channel-step", type=int, default=10,
                    help="submuestreo de canales: el continuo no necesita los 3681")
    ap.add_argument("--amplitude-scale", type=float, default=1.0,
                    help="amplitud inyectada, en unidades del continuo medido de la compañera")
    ap.add_argument("--out-json", required=True)
    args = ap.parse_args(argv)

    root = Path(args.project_root or Path.cwd()).resolve()
    cfg = x02.stage_x02_config_from_run(args.run, project_root=root)
    paths = x02.stage_x02_paths(args.run, root)
    object_yx, star_yx, _p, _q = _load_positions(paths, cfg)
    psf_model, _path = x02._load_psf_model(paths, cfg)
    cube, wave, _good, _bad, _cp, _bunit = _load_residual_cube(paths, cfg)

    step = max(1, int(args.channel_step))
    cube, wave = cube[::step], wave[::step]
    print(f"cubo {cube.shape} tras submuestrear 1 de cada {step} canales", flush=True)

    # Amplitud: el nivel de continuo de la compañera en el rojo, donde SI hay
    # senal (B5/B6 son las dos unicas bandas con continuo detectado, 09-10).
    from musepipe.extraction.product import SpectrumProduct
    apert = SpectrumProduct.read(paths["spec_aperture_object"])
    rojo = (apert.wave_A >= 7600) & (apert.wave_A <= 9100)
    nivel = float(np.nanmedian(np.asarray(apert.flux, float)[rojo]))
    amplitude = float(args.amplitude_scale) * nivel
    print(f"amplitud inyectada: {amplitude:.4g} (continuo medido de la compañera en 7600-9100 A)", flush=True)

    sep = float(np.hypot(object_yx[0] - star_yx[0], object_yx[1] - star_yx[1]))
    posiciones = same_radius_control_positions(
        object_yx, star_yx, cube.shape[1], cube.shape[2],
        n_positions=int(args.n_positions), exclude_angle_deg=25.0)
    if not posiciones:
        raise SystemExit(
            f"0 posiciones utilizables a {sep:.2f} px. `same_radius_control_positions` excluye "
            f"la del propio compañero (angulo < {25.0:g} deg), asi que --n-positions 1 nunca "
            "devuelve nada: usa 3 o mas, y comprueba que el radio cabe en el cubo."
        )
    print(f"{len(posiciones)} posiciones a {sep:.2f} px de la primaria: "
          f"{[tuple(int(v) for v in q) for q in posiciones]}", flush=True)

    report = {"script": "i_inyeccion_continuo", "run": args.run, "amplitude": amplitude,
              "separation_px": sep, "channel_step": step,
              "wave_A": [float(w) for w in wave], "modes": {}}
    for mode in args.modes:
        rec = []
        for idx, pos in enumerate(posiciones):
            t0 = time.time()
            base = extract(cube, wave, pos, star_yx, psf_model, cfg, mode)
            inyectado = inject(cube, wave, psf_model, pos, amplitude)
            conj = extract(inyectado, wave, pos, star_yx, psf_model, cfg, mode)
            rec.append((conj - base) / amplitude)
            print(f"  {mode:12s} pos {idx + 1}/{len(posiciones)} en {time.time() - t0:.0f}s", flush=True)
        rec = np.asarray(rec, dtype=np.float64)
        mediana = np.nanmedian(rec, axis=0)
        report["modes"][mode] = {
            "recovery_median": [float(v) for v in mediana],
            "recovery_scatter": [float(v) for v in (np.nanpercentile(rec, 84, axis=0) - np.nanpercentile(rec, 16, axis=0)) / 2],
            "n_positions": int(rec.shape[0]),
        }
        Path(args.out_json).write_text(json.dumps(report, indent=1), encoding="utf-8")
        azul = (wave >= 4900) & (wave <= 5400)
        rojo_w = (wave >= 8600) & (wave <= 9100)
        print(f"  -> {mode}: recuperacion mediana azul {np.nanmedian(mediana[azul]):.3f} | "
              f"rojo {np.nanmedian(mediana[rojo_w]):.3f}", flush=True)

    print(f"\n{'lambda':>9s} " + " ".join(f"{m:>13s}" for m in args.modes))
    for i in range(0, wave.size, max(1, wave.size // 20)):
        fila = " ".join(f"{report['modes'][m]['recovery_median'][i]:13.3f}" for m in args.modes)
        print(f"{wave[i]:8.0f}A {fila}")
    print(f"\nescrito {args.out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
