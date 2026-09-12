"""Elegir el radio del anillo de fondo con verdad conocida, no con el resultado.

`docs/2026-09-12_el_radio_del_anillo_mueve_42bb.md` midió que el continuo de
ROXs 42B b cambia un **40 %** con `r_out`, y que el barrido **no puede elegirlo**:
mide una dependencia, no un acierto. Esto añade la verdad que falta.

**Tres términos, y tiran en direcciones distintas:**

- **A · sesgo del fondo** donde NO hay fuente. Verdad conocida y gratis: en una
  posición vacía al mismo radio estelocéntrico la extracción **debe dar cero**.
  Anillos pequeños, dominados por el lado interior -más brillante-, sobreestiman
  el fondo y dejan el nivel corto.
- **B · auto-sustracción**: parte de las alas de la compañera cae dentro del
  anillo y el fondo se lleva señal. Verdad conocida: la recuperación de un
  continuo inyectado **debe dar 1**. Tira al revés que A.
- **C · ruido**: más píxeles en el anillo, menos ruido. Tira a favor de radios
  grandes.

**A y B son sesgos y se SUMAN coherentemente** sobre el flujo medido
(`medido = R·F + offset`, o sea `error relativo = (R-1) + offset/F`); C entra en
cuadratura. Optimizar solo B -como propuse primero- es insuficiente: el 40 % que
se midió no es multiplicativo, es el nivel de fondo restado.

**Lo que este script NO hace**: elegir por el acuerdo entre estimadores (acuerdo
no es verdad) ni por el efecto en D1 o en Mdot (eso se reporta, no se optimiza).

    python scripts/m_elige_r_out.py --run <run> --out-json <out>.json
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

from i_inyeccion_continuo import inject, load_extraction_cube  # noqa: E402
from musepipe.apertures import same_radius_control_positions  # noqa: E402
from musepipe.extraction.optimal import make_optimal_product  # noqa: E402
from musepipe.extraction.product import SpectrumProduct  # noqa: E402
from musepipe.stages import stage_x02_optimal as x02  # noqa: E402
from musepipe.stages.stage_x01_aperture import _load_positions  # noqa: E402


def extrae(cube, wave, centro, estrella, psf_model, cfg, anillo):
    ext = make_optimal_product(
        cube, wave, centro, psf_model,
        star_yx=estrella, run_id=str(cfg["run_id"]), input_cube_path="elige_r_out",
        variant="r_out", error_mode="empirical",
        aperture_correction=cfg.get("x02_aperture_correction", "auto"),
        window_radius_px=float(cfg.get("x02_window_radius_px", 8.0)),
        clip_sigma=float(cfg.get("x02_clip_sigma", 4.0)),
        clip_max_iter=int(cfg.get("x02_clip_max_iter", 2)),
        n_controls=0, local_bkg_annulus_px=anillo, background_mode="annulus",
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
    ap.add_argument("--n-positions", type=int, default=6)
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
    lo, hi = float(args.banda[0]), float(args.banda[1])
    msk = (wave >= lo) & (wave <= hi)
    exterior = cfg.get("x02_local_bkg_annulus_px")
    borde = float(exterior[2]) if exterior and len(exterior) > 2 else 30.0
    r_out_produccion = float(exterior[1]) if exterior else None

    # La escala con la que se adimensionalizan A y C: el continuo medido de la
    # compañera. Es un PROXY del flujo verdadero, que no se conoce; se declara.
    apert = SpectrumProduct.read(paths["spec_aperture_object"])
    band = (apert.wave_A >= lo) & (apert.wave_A <= hi)
    F = float(np.nanmedian(np.asarray(apert.flux, float)[band]))
    sep = float(np.hypot(obj[0] - star[0], obj[1] - star[1]))
    vacias = same_radius_control_positions(obj, star, cube.shape[1], cube.shape[2],
                                           n_positions=int(args.n_positions), exclude_angle_deg=25.0)
    print(f"{args.run}: sep {sep:.1f} px | F(proxy) {F:.1f} | {len(vacias)} posiciones vacias "
          f"| cubo {etiqueta} | r_out en produccion {r_out_produccion}", flush=True)

    filas = []
    print(f"\n{'r_out':>6s} {'A sesgo':>9s} {'B (R-1)':>9s} {'sesgo A+B':>10s} {'C ruido':>9s} {'error total':>12s}")
    for r_out in args.r_out:
        anillo = [float(args.r_in), float(r_out), borde]
        niveles = [float(np.nanmedian(extrae(cube, wave, p, star, psf_model, cfg, anillo)[msk]))
                   for p in vacias]
        A = float(np.median(niveles)) / F                       # verdad: 0
        C = float(np.std(niveles)) / F
        base = extrae(cube, wave, obj, star, psf_model, cfg, anillo)
        conj = extrae(inject(cube, wave, psf_model, obj, F), wave, obj, star, psf_model, cfg, anillo)
        B = float(np.nanmedian((conj - base)[msk])) / F - 1.0   # verdad: 0
        sesgo = A + B                                           # se suman: medido = R*F + offset
        total = float(np.hypot(sesgo, C))
        filas.append({"r_in": float(args.r_in), "r_out": float(r_out), "A_sesgo_fondo": A,
                      "B_autosustraccion": B, "sesgo_total": sesgo, "C_ruido": C, "error_total": total})
        print(f"{r_out:6.1f} {100*A:8.2f}% {100*B:8.2f}% {100*sesgo:9.2f}% {100*C:8.2f}% {100*total:11.2f}%",
              flush=True)
        Path(args.out_json).write_text(json.dumps(
            {"script": "m_elige_r_out", "run": args.run, "separacion_px": sep,
             "F_proxy": F, "cubo": etiqueta, "banda_A": [lo, hi],
             "r_out_produccion": r_out_produccion, "filas": filas}, indent=1), encoding="utf-8")

    tot = np.array([f["error_total"] for f in filas])
    mejor = filas[int(np.argmin(tot))]
    plano = bool((tot.max() - tot.min()) < 0.2 * tot.max())
    print(f"\n  minimo en r_out = {mejor['r_out']:.0f} (error total {100*mejor['error_total']:.2f} %)")
    if plano:
        print("  PERO la curva es PLANA (<20 % de variacion): no hay un optimo,")
        print("  la respuesta correcta es declarar el sistematico, no elegir un numero.")
    if r_out_produccion is not None:
        act = [f for f in filas if f["r_out"] == r_out_produccion]
        if act:
            print(f"  en produccion (r_out={r_out_produccion:.0f}): error total "
                  f"{100*act[0]['error_total']:.2f} %")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
