"""¿El residuo que falta es del INSTRUMENTO o del cielo? Exposición a exposición.

`docs/2026-09-12_el_presupuesto_no_cierra.md` dejó 8.3 puntos sin explicar: ni la
respuesta al flujo añadido, ni el offset en vacío, ni la no linealidad de la
mediana. Las cinco pruebas de ese hilo corrieron sobre el cubo **combinado**.

Y este objeto se observó **rotando el campo**: `INS DROT POSANG` toma 0, 45, 90,
135 y 180 grados, con `ADA ABSROT` barriendo 38 grados, mientras el DRS
remuestrea todas las exposiciones a una **orientación de cielo común** (`CD1_1`
idéntico, que es lo que `stream_combine` exige). Así que **cualquier estructura
fija en el marco del INSTRUMENTO queda promediada sobre 0-180 grados al
combinar**: se convierte en un anillo. Buscar estructura azimutal en el combinado
es buscarla donde la combinación la borró.

**La predicción, declarada antes de medir.** Se mide en cada exposición la
diferencia entre dos estimadores de fondo en la posición de la compañera:
`annulus` -un anillo alrededor de ELLA, sensible a la estructura azimutal local- y
`azimuthal` -un anillo a radio estelocéntrico CONSTANTE, que promedia azimut y es
ciego a ella-.

- Si el residuo es **instrumental**, la compañera está fija en el cielo pero se
  mueve en el marco del instrumento al girar el derrotador: la diferencia entre
  estimadores tiene que **variar con POSANG**.
- Si es del **cielo o del halo real**, la diferencia tiene que quedarse **quieta**
  al girar.

Es la primera prueba del hilo que separa dos familias en vez de descartar un
mecanismo más.

    python scripts/k_por_exposicion_posang.py --run <run> --out-json <out>.json

Lee los cubos por exposición declarados en el config. **No escribe en ningún run**
y procesa **un cubo cada vez**: esta máquina mata los procesos que crecen.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from astropy.io import fits

ROOT = Path(__file__).resolve().parents[1]
import sys  # noqa: E402

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from musepipe.extraction.optimal import make_optimal_product  # noqa: E402
from musepipe.reduction.stream_combine import measure_primary_center  # noqa: E402
from musepipe.stages import stage_x02_optimal as x02  # noqa: E402
from musepipe.stages.stage_x01_aperture import _load_positions  # noqa: E402

MODOS = ("annulus", "azimuthal")


def cabecera(path):
    h = fits.getheader(path, 0)
    return {
        "date_obs": str(h.get("DATE-OBS"))[:19],
        "posang": h.get("HIERARCH ESO INS DROT POSANG"),
        "absrot": h.get("HIERARCH ESO ADA ABSROT START"),
        "parang": h.get("HIERARCH ESO TEL PARANG START"),
        "exptime": h.get("EXPTIME"),
        "noche": str(h.get("DATE-OBS"))[:10],
    }


def nivel_en(cube, wave, centro, estrella, psf_model, cfg, mode):
    ext = make_optimal_product(
        cube, wave, centro, psf_model,
        star_yx=estrella, run_id=str(cfg["run_id"]), input_cube_path="perexp",
        variant=f"perexp_{mode}", error_mode="empirical",
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
    ap.add_argument("--channel-step", type=int, default=60)
    ap.add_argument("--banda", nargs=2, type=float, default=[8600.0, 9100.0])
    ap.add_argument("--limit", type=int, default=None, help="solo las N primeras exposiciones")
    ap.add_argument("--out-json", required=True)
    args = ap.parse_args(argv)

    root = Path(args.project_root or Path.cwd()).resolve()
    cfg = x02.stage_x02_config_from_run(args.run, project_root=root)
    paths = x02.stage_x02_paths(args.run, root)
    obj, star, _p, _q = _load_positions(paths, cfg)
    psf_model, _mp = x02._load_psf_model(paths, cfg)
    # El desplazamiento compañera-primaria en PIXELES vale en todas las
    # exposiciones porque el DRS las remuestrea a la MISMA orientacion de cielo
    # (CD identico); lo que cambia entre ellas es donde cae la estrella.
    dy, dx = float(obj[0] - star[0]), float(obj[1] - star[1])
    print(f"desplazamiento compañera-primaria: dy={dy:+.2f} dx={dx:+.2f} px", flush=True)

    # De donde salen los cubos, en orden y declarandolo: el config si los tiene
    # (ROXs 12 b), y si no el PLAN DEL COMBINADO, que los lista por construccion
    # -es con los que se hizo el cubo- (ROXs 42B b no declara `perexp_cubes`).
    cubos = list((cfg.get("perexp_cubes") or []))
    origen = "config.perexp_cubes"
    if not cubos:
        plan = paths["paths"].stage_dir / "stream_combine_plan.json"
        if plan.exists():
            payload = json.loads(plan.read_text(encoding="utf-8"))
            cubos = [e["file"] for e in payload.get("exposures", [])]
            origen = f"{plan.name} (metodo {payload.get('method')}, pesos {payload.get('weight_mode')})"
    if not cubos:
        raise SystemExit("ni `perexp_cubes` en el config ni `stream_combine_plan.json` en el run")
    faltan = [c for c in cubos if not Path(c).exists()]
    if faltan:
        raise SystemExit(f"{len(faltan)} cubos del plan no estan en disco, p.ej. {faltan[0]}")
    print(f"cubos: {len(cubos)} de {origen}", flush=True)
    if args.limit:
        cubos = cubos[: int(args.limit)]
    lo, hi = float(args.banda[0]), float(args.banda[1])
    step = max(1, int(args.channel_step))

    filas = []
    for i, ruta in enumerate(cubos):
        meta = cabecera(ruta)
        centro_medido = measure_primary_center(ruta, data_ext="DATA")
        sy, sx = float(centro_medido["y_center"]), float(centro_medido["x_center"])
        with fits.open(ruta, memmap=True) as hdul:
            cube = np.asarray(hdul["DATA"].data[::step], dtype=np.float64)
            hdr = hdul["DATA"].header
        nz = cube.shape[0]
        wave = (float(hdr["CRVAL3"]) + (np.arange(nz) * step - (float(hdr["CRPIX3"]) - 1))
                * float(hdr["CD3_3"]))
        msk = (wave >= lo) & (wave <= hi)
        compa = (sy + dy, sx + dx)
        niveles = {}
        for mode in MODOS:
            f = nivel_en(cube, wave, compa, (sy, sx), psf_model, cfg, mode)
            niveles[mode] = float(np.nanmedian(f[msk]))
        del cube
        razon = (niveles["azimuthal"] / niveles["annulus"]
                 if niveles["annulus"] not in (0.0,) and np.isfinite(niveles["annulus"]) else np.nan)
        filas.append({**meta, "star_yx": [sy, sx], "companion_yx": list(compa),
                      **{f"nivel_{m}": niveles[m] for m in MODOS}, "azi_sobre_ann": razon})
        print(f"  [{i + 1:2d}/{len(cubos)}] {meta['date_obs']} POSANG={str(meta['posang']):>5s} "
              f"ann={niveles['annulus']:9.2f} azi={niveles['azimuthal']:9.2f} "
              f"razon={razon:6.3f}", flush=True)
        Path(args.out_json).write_text(json.dumps({"script": "k_por_exposicion_posang",
                                                   "run": args.run, "banda_A": [lo, hi], "origen_cubos": origen,
                                                   "channel_step": step, "filas": filas}, indent=1),
                                       encoding="utf-8")

    print(f"\n{'POSANG':>8s} {'n':>3s} {'razon azi/ann':>15s} {'dispersion':>12s}")
    por_posang = {}
    for f in filas:
        por_posang.setdefault(f["posang"], []).append(f["azi_sobre_ann"])
    for pa in sorted(por_posang, key=lambda v: (v is None, v)):
        v = np.asarray([x for x in por_posang[pa] if np.isfinite(x)], dtype=float)
        if v.size:
            print(f"{str(pa):>8s} {v.size:3d} {np.median(v):15.4f} {np.std(v):12.4f}")
    print("\nVaria con POSANG -> instrumental. Constante -> cielo o halo real.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
