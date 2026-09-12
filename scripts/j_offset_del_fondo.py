"""El offset del fondo, en unidades de flujo: extraer donde NO hay nada.

`docs/2026-09-12_el_fondo_no_pierde_flujo_desplaza.md` midió que el tratamiento
de fondo recupera el 98 % de un continuo inyectado en el rojo, mientras los
métodos discrepan entre sí un 10-39 % en esa misma banda. La conclusión fue que
la divergencia es un **offset aditivo** y no una pérdida multiplicativa — pero la
inyección es diferencial y por construcción NO puede medir un offset.

Esto lo mide directamente: extraer en posiciones **sin fuente**, a la misma
separación de la primaria que la compañera. Lo que devuelva la extracción allí es
el nivel que el método pone donde no hay nada, o sea el offset.

**Predicción, declarada antes de mirar.** El barrido del 09-11 midió que en B6 el
sesgo fraccional de `optimal_ls` frente a `aperture` vale -18 % con `annulus` y
-11 % con `azimuthal`: 7 puntos de diferencia entre modos. Si el offset es la
explicación, la diferencia de nivel entre esos dos modos en una posición vacía
tiene que valer **~7 % del continuo de la compañera** — que en las unidades del
cubo son ~160, con el continuo medido en 2337.

Si sale mucho menor, el offset no explica la divergencia y hay que buscar otra
cosa. Si sale del orden, queda explicada y además **medida en unidades de flujo**,
que es lo que hace falta para corregirla o para declararla como sistemático.

    python scripts/j_offset_del_fondo.py --run <run> --out-json <out>.json
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

from i_inyeccion_continuo import extract, load_extraction_cube  # noqa: E402
from musepipe.apertures import same_radius_control_positions  # noqa: E402
from musepipe.extraction.optimal import BACKGROUND_MODES  # noqa: E402
from musepipe.extraction.product import SpectrumProduct  # noqa: E402
from musepipe.stages import stage_x02_optimal as x02  # noqa: E402
from musepipe.stages.stage_x01_aperture import _load_positions  # noqa: E402

BANDAS = (("B1", 4900, 5400), ("B3", 6100, 6400), ("B5", 7600, 8000), ("B6", 8600, 9100))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--project-root", default=None)
    ap.add_argument("--modes", nargs="+", default=list(BACKGROUND_MODES))
    ap.add_argument("--n-positions", type=int, default=6)
    ap.add_argument("--channel-step", type=int, default=20)
    ap.add_argument("--out-json", required=True)
    args = ap.parse_args(argv)

    root = Path(args.project_root or Path.cwd()).resolve()
    cfg = x02.stage_x02_config_from_run(args.run, project_root=root)
    paths = x02.stage_x02_paths(args.run, root)
    object_yx, star_yx, _p, _q = _load_positions(paths, cfg)
    psf_model, _mp = x02._load_psf_model(paths, cfg)
    cube, wave, etiqueta = load_extraction_cube(paths, cfg)
    step = max(1, int(args.channel_step))
    cube, wave = cube[::step], wave[::step]
    print(f"cubo: {etiqueta}", flush=True)

    apert = SpectrumProduct.read(paths["spec_aperture_object"])
    rojo = (apert.wave_A >= 7600) & (apert.wave_A <= 9100)
    continuo = float(np.nanmedian(np.asarray(apert.flux, float)[rojo]))

    posiciones = same_radius_control_positions(
        object_yx, star_yx, cube.shape[1], cube.shape[2],
        n_positions=int(args.n_positions), exclude_angle_deg=25.0)
    if not posiciones:
        raise SystemExit("0 posiciones utilizables: sube --n-positions")
    print(f"{len(posiciones)} posiciones VACIAS | continuo de la compañera (7600-9100 A) = {continuo:.4g}", flush=True)

    niveles = {}
    for mode in args.modes:
        filas = [extract(cube, wave, pos, star_yx, psf_model, cfg, mode) for pos in posiciones]
        niveles[mode] = np.asarray(filas, dtype=np.float64)
        print(f"  {mode}: {len(filas)} extracciones", flush=True)

    report = {"script": "j_offset_del_fondo", "run": args.run, "continuo_companera": continuo,
              "cube": etiqueta,
              "n_positions": len(posiciones), "channel_step": step, "bandas": {}}
    print(f"\n{'banda':>6s} " + " ".join(f"{m:>22s}" for m in args.modes))
    for nombre, lo, hi in BANDAS:
        msk = (wave >= lo) & (wave <= hi)
        fila, datos = [], {}
        for m in args.modes:
            por_pos = np.nanmedian(niveles[m][:, msk], axis=1)      # nivel de cada posicion
            nivel = float(np.nanmedian(por_pos))
            disp = float(np.nanpercentile(por_pos, 84) - np.nanpercentile(por_pos, 16)) / 2
            datos[m] = {"nivel": nivel, "dispersion": disp,
                        "pct_del_continuo": 100.0 * nivel / continuo}
            fila.append(f"{nivel:12.1f} ({100.0 * nivel / continuo:+5.1f}%)")
        report["bandas"][nombre] = datos
        print(f"{nombre:>6s} " + " ".join(fila))

    print(f"\nDiferencias entre modos, en % del continuo de la compañera:")
    print(f"{'banda':>6s} " + " ".join(f"{a[:3]}-{b[:3]:>8s}" for a in args.modes for b in args.modes if a < b))
    for nombre, _lo, _hi in BANDAS:
        d = report["bandas"][nombre]
        celdas = []
        for a in args.modes:
            for b in args.modes:
                if a < b:
                    celdas.append(f"{d[a]['pct_del_continuo'] - d[b]['pct_del_continuo']:+11.1f}%")
        print(f"{nombre:>6s} " + " ".join(celdas))
    Path(args.out_json).write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"\nescrito {args.out_json}")
    print("La prediccion: annulus - azimuthal en B6 ~ 7 % del continuo (~160 en unidades del cubo).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
