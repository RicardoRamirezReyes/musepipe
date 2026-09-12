"""¿Mueve el TRATAMIENTO DE FONDO la divergencia de continuo en el rojo?

Es la vía que dejó abierta `docs/2026-09-10_b6_no_es_el_cielo.md`. Allí se midió
que el residuo de cielo de B6 no es un conjunto de canales enmascarables sino un
**pedestal de banda entera** (significativo en el 100 % de los canales), y que un
pedestal así **no se puede probar enmascarando** porque no quedan canales limpios
contra los que contrastar. Lo que sí se puede variar es cómo lo trata cada
método, que es lo único medido que los separa en el rojo.

C3 declara tres modos en `BACKGROUND_MODES`: `annulus` (el que corre), `azimuthal`
y `local_plane`. Este script re-extrae C3 con cada uno sobre una COPIA del run y
vuelve a correr la comparación de D1, mirando B6 y B5.

**Dos controles vienen de regalo y hay que mirarlos:**

- `psffit_vs_aperture` NO depende de C3. Si se mueve, el experimento está mal
  montado.
- `annulus` es el modo actual: es la reproducción del estado de partida.

**Aviso de procedencia.** La copia sale del run tal cual está en disco, así que
las cifras de partida son las de ANTES de la corrección empírica de apcorr
(B6 ~ -65 en `optimal_ls_vs_aperture`), no las -37 de la copia `d1check`. Lo que
se lee aquí es el cambio RELATIVO entre modos, cada uno contra su propio
`annulus`.

    python scripts/f_barrido_fondo_b6.py --src-run <run> --dest-run <copia> \
        --out-json <out>.json

La copia se monta con **enlaces simbólicos** a las entradas y ficheros reales
solo para lo que C3 escribe, así que no puede escribir en el run de origen.
"""
from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
import sys  # noqa: E402

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from musepipe.extraction.optimal import BACKGROUND_MODES  # noqa: E402
from musepipe.stages import stage_x02_optimal as x02  # noqa: E402
from musepipe.stages import stage_x10_compare as x10  # noqa: E402

#: Lo que C3 escribe. NO se enlaza: si estuviera enlazado, C3 escribiria a traves
#: del enlace y modificaria el run de origen.
X02_OUTPUTS = (
    "spec_optimal_object.fits",
    "spec_optimal_psfsub_object.fits",
    "spec_optimal_clip_rejection.fits",
    "spec_optimal_controls.npz",
    "spec_optimal_psfsub_controls.npz",
    "spec_optimal_qc.json",
)

#: Los otros cinco metodos que D1 compara. Se COPIAN de verdad y se les reescribe
#: el RUNID, porque `validate_product_set` exige que los seis productos declaren
#: el mismo run -es la puerta anti-contaminacion cruzada, y tiene razon-: si se
#: enlazan, los de C3 dicen la copia y estos el origen, y D1 aborta.
D1_SIBLING_PRODUCTS = (
    "spec_aperture_object.fits",
    "spec_psffit_object.fits",
    "spec_sgf_object.fits",
    "spec_lpm_object.fits",
)

PAIRS = ("psffit_vs_aperture", "optimal_ls_vs_aperture", "psffit_vs_optimal_ls")


def build_copy(root: Path, src_run: str, dest_run: str, *, force=False):
    src = root / "runs" / src_run
    dest = root / "runs" / dest_run
    if dest.exists():
        if not force:
            raise SystemExit(f"{dest} ya existe (usa --force para rehacerla)")
        shutil.rmtree(dest)
    (dest / "stages").mkdir(parents=True)
    (dest / "config").mkdir(parents=True)

    cfg = json.loads((src / "config" / "config.json").read_text(encoding="utf-8"))
    meta = dict(cfg.get("meta") or {})
    if meta.get("legacy"):
        raise SystemExit(f"{src_run} esta marcado legacy: no se copia ni se re-corre.")
    meta["copied_from"] = src_run
    meta["note"] = (
        f"COPIA DE TRABAJO de {src_run}, montada por f_barrido_fondo_b6.py. Solo para barrer "
        "x02_background_mode y volver a comparar en D1. NO es un run cientifico."
    )
    cfg["meta"] = meta
    cfg["run_id"] = dest_run
    if isinstance(cfg.get("config"), dict):
        cfg["config"]["run_id"] = dest_run
    (dest / "config" / "config.json").write_text(json.dumps(cfg, indent=1), encoding="utf-8")

    n_links = 0
    for item in sorted((src / "stages").iterdir()):
        if item.name in X02_OUTPUTS or item.name in D1_SIBLING_PRODUCTS or not item.is_file():
            continue
        (dest / "stages" / item.name).symlink_to(item.resolve())
        n_links += 1
    n_copies = restamp_siblings(src, dest, dest_run)
    return dest, n_links, n_copies


def restamp_siblings(src: Path, dest: Path, dest_run: str):
    """Copia los otros cinco productos y les pone el RUNID de la copia."""

    from astropy.io import fits

    n = 0
    for name in D1_SIBLING_PRODUCTS:
        source = (src / "stages" / name).resolve()
        if not source.exists():
            continue
        target = dest / "stages" / name
        if target.is_symlink():
            target.unlink()
        shutil.copy2(source, target)
        with fits.open(target, mode="update") as hdul:
            for hdu in hdul:
                if "RUNID" in hdu.header:
                    hdu.header["RUNID"] = dest_run
                    hdu.header.add_history(f"RUNID re-stamped from {src.name} by f_barrido_fondo_b6")
        n += 1
    return n


def preserve_outputs(dest: Path, mode: str):
    """Guarda los productos del modo antes de que el siguiente los pise.

    C3 escribe siempre en las mismas rutas, asi que sin esto una corrida de tres
    modos deja UNA extraccion, la ultima. Y lo que hace falta para mirar el fondo
    en funcion de lambda son las tres.
    """
    keep = dest / "stages" / "_modos" / mode
    keep.mkdir(parents=True, exist_ok=True)
    kept = []
    for name in X02_OUTPUTS:
        src = dest / "stages" / name
        if src.exists():
            shutil.copy2(src, keep / name)
            kept.append(name)
    return keep, kept


def band_rows(rows, band):
    out = {}
    for row in rows:
        if row["band"] == band and row["pair"] in PAIRS:
            out[row["pair"]] = {
                "t": x10._finite_or_none(row.get("t_stat")),
                "p": x10._finite_or_none(row.get("p_value")),
            }
    return out


def compare(run_id, root):
    cfg = x10.stage_x10_config_from_run(run_id, project_root=root)
    paths = x10.stage_x10_paths(run_id, root)
    products = x10.load_method_products(x10._product_paths_from_config(cfg, paths))
    controls = x10.load_control_spectra(x10._control_paths_from_config(cfg, paths))
    g1 = x10.load_g1_inputs(cfg, paths)
    rows, _c, qc = x10.compare_methods(
        products, controls,
        sigma_smooth_channels=int(cfg.get("x10_sigma_smooth_channels", 5)),
        g1_inputs=g1,
        p_divergent=float(cfg.get("x10_p_divergent", x10.DEFAULT_P_DIVERGENT)),
        p_strong=float(cfg.get("x10_p_strong", x10.DEFAULT_P_STRONG)),
        scale_gate_sigma=float(cfg.get("x10_scale_gate_sigma", x10.DEFAULT_SCALE_GATE_SIGMA)),
        gate_alpha=float(cfg.get("x10_control_gate_alpha", x10.DEFAULT_CONTROL_GATE_ALPHA)),
        line_continuum_window_A=float(cfg.get("x10_line_continuum_window_A", x10.DEFAULT_LINE_CONTINUUM_WINDOW_A)),
    )
    return rows, qc


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src-run", required=True)
    ap.add_argument("--dest-run", required=True)
    ap.add_argument("--project-root", default=None)
    ap.add_argument("--modes", nargs="+", default=list(BACKGROUND_MODES))
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--reuse-copy", action="store_true",
                    help="no rehacer la copia ni repetir los modos ya escritos en --out-json")
    ap.add_argument("--out-json", required=True)
    args = ap.parse_args(argv)

    root = Path(args.project_root or Path.cwd()).resolve()
    if args.reuse_copy and (root / "runs" / args.dest_run).exists():
        dest = root / "runs" / args.dest_run
        print(f"reutilizando la copia {dest}")
    else:
        dest, n_links, n_copies = build_copy(root, args.src_run, args.dest_run, force=args.force)
        print(f"copia en {dest} ({n_links} entradas enlazadas, {n_copies} productos re-sellados, "
              f"{len(X02_OUTPUTS)} salidas reales)")

    out_path = Path(args.out_json)
    report = {}
    if args.reuse_copy and out_path.exists():
        report = json.loads(out_path.read_text(encoding="utf-8"))
        print(f"informe previo con los modos: {list((report.get('modes') or {}))}")
    report = report or {
        "script": "f_barrido_fondo_b6",
        "src_run": args.src_run,
        "dest_run": args.dest_run,
        "provenance_note": (
            "La copia sale del run en disco: las cifras son las de ANTES de la correccion "
            "empirica de apcorr. Se lee el cambio RELATIVO entre modos."
        ),
        "modes": {},
    }
    for mode in args.modes:
        if mode not in BACKGROUND_MODES:
            raise SystemExit(f"modo desconocido: {mode} (validos: {BACKGROUND_MODES})")
        if args.reuse_copy and mode in (report.get("modes") or {}):
            print(f"\n=== {mode}: ya medido en el informe previo, se salta ===", flush=True)
            continue
        print(f"\n=== C3 con background_mode={mode} ===", flush=True)
        t0 = time.time()
        written = x02.run_stage_x02(args.dest_run, project_root=root,
                                    overrides={"x02_background_mode": mode})
        dt = time.time() - t0
        qc02 = written["qc"] if isinstance(written, dict) and "qc" in written else {}
        keep, kept = preserve_outputs(root / "runs" / args.dest_run, mode)
        print(f"  C3 ok en {dt / 60:.1f} min; {len(kept)} productos guardados en {keep}", flush=True)
        rows, qc10 = compare(args.dest_run, root)
        entry = {
            "seconds_x02": dt,
            "B6": band_rows(rows, "B6"),
            "B5": band_rows(rows, "B5"),
            "verdict": qc10.get("verdict"),
            "x02_checks": (qc02 or {}).get("checks"),
            "continuum_bias_vs_aperture_pct": (qc02 or {}).get("continuum_bias_vs_aperture_pct"),
            "snr_gain_median": ((qc02 or {}).get("snr_gain_vs_aperture") or {}).get("median"),
            "clip_concentration": (qc02 or {}).get("clip_concentration"),
        }
        report["modes"][mode] = entry
        for band in ("B6", "B5"):
            line = "  ".join(f"{pid.split('_vs_')[0][:8]}/{pid.split('_vs_')[1][:8]}="
                             f"{entry[band].get(pid, {}).get('t', float('nan')):+8.2f}" for pid in PAIRS)
            print(f"  {band}: {line}", flush=True)
        Path(args.out_json).write_text(json.dumps(report, indent=1), encoding="utf-8")

    base = report["modes"].get("annulus", {}).get("B6", {})
    print(f"\n{'modo':14s} " + " ".join(f"{p:>26s}" for p in PAIRS))
    for mode, entry in report["modes"].items():
        cells = []
        for pid in PAIRS:
            t = entry["B6"].get(pid, {}).get("t")
            b = base.get(pid, {}).get("t")
            delta = "" if (t is None or b is None or mode == "annulus") else f" ({100 * (abs(t) - abs(b)) / abs(b):+.0f}%)"
            cells.append(f"{t:+10.2f}{delta:>16s}" if t is not None else f"{'-':>26s}")
        print(f"{mode:14s} " + " ".join(cells))
    print(f"\nescrito {args.out_json}")
    print("Control: psffit_vs_aperture NO depende de C3; si se mueve, el montaje esta mal.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
