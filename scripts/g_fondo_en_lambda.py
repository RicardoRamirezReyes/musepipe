"""¿Dónde deja de ser mejor cada tratamiento de fondo?

`docs/2026-09-11_el_fondo_es_la_palanca.md` midió los tres modos de
`x02_background_mode` por bandas y salió monótono: el que menos continuo se come
es el que mejor S/N da y el que menos diverge. Pero `azimuthal` -el mejor en el
rojo- está retirado **por el azul**, donde se midió que es un offset de signo
fijo que empeora el ajuste atmosférico. Las dos cosas juntas dicen que el fondo
es **cromático**, y entonces la pregunta no es qué modo gana sino **dónde**.

Esto vuelve a comparar con la maquinaria de D1 -misma sigma empírica de los
controles, mismo centrado, misma integración- pero sobre una rejilla fina de
bandas de continuo, y para los dos pares que dependen de C3. El tercero,
`psffit_vs_aperture`, viaja como control: no depende de C3 y tiene que salir
idéntico en los tres modos, banda a banda.

Lee las extracciones que `f_barrido_fondo_b6.py` guarda en
`stages/_modos/<modo>/`; los otros cuatro métodos salen del directorio normal.

    python scripts/g_fondo_en_lambda.py --run <copia> --out-json <out>.json
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

from musepipe.stages import stage_x10_compare as x10  # noqa: E402

PAIRS = ("optimal_ls_vs_aperture", "psffit_vs_optimal_ls", "psffit_vs_aperture")

#: Ventanas no solapadas. 250 A es ancho para que la sigma empirica de 8
#: controles tenga con que, y estrecho para ver la deriva dentro de B5/B6.
DEFAULT_WIDTH_A = 250.0


def fine_bands(wave, width_A):
    lo = float(np.floor(np.nanmin(wave) / width_A) * width_A)
    hi = float(np.nanmax(wave))
    bands = []
    edge = lo
    while edge < hi:
        name = f"W{int(edge)}"
        bands.append(x10.ComparisonBand(name, edge, edge + width_A, "continuum", f"{int(edge)}-{int(edge + width_A)}"))
        edge += width_A
    return tuple(bands)


def products_for_mode(root, run_id, mode):
    """Los seis productos, con `optimal_*` tomados del modo pedido."""
    paths = x10.stage_x10_paths(run_id, root)
    cfg = x10.stage_x10_config_from_run(run_id, project_root=root)
    product_paths = dict(x10._product_paths_from_config(cfg, paths))
    control_paths = dict(x10._control_paths_from_config(cfg, paths))
    mode_dir = paths["paths"].stage_dir / "_modos" / mode
    if not mode_dir.is_dir():
        raise SystemExit(f"no esta la extraccion del modo {mode!r}: {mode_dir}")
    product_paths["optimal_ls"] = mode_dir / "spec_optimal_object.fits"
    product_paths["optimal_psfsub"] = mode_dir / "spec_optimal_psfsub_object.fits"
    control_paths["optimal_ls"] = mode_dir / "spec_optimal_controls.npz"
    control_paths["optimal_psfsub"] = mode_dir / "spec_optimal_psfsub_controls.npz"
    products = x10.load_method_products(product_paths)
    controls = x10.load_control_spectra(control_paths)
    g1 = x10.load_g1_inputs(cfg, paths)
    return cfg, products, controls, g1


def rows_on_grid(cfg, products, controls, g1, bands):
    original = x10.COMPARISON_BANDS
    x10.COMPARISON_BANDS = bands
    try:
        rows, _controls, _qc = x10.compare_methods(
            products, controls,
            sigma_smooth_channels=int(cfg.get("x10_sigma_smooth_channels", 5)),
            g1_inputs=g1,
            p_divergent=float(cfg.get("x10_p_divergent", x10.DEFAULT_P_DIVERGENT)),
            p_strong=float(cfg.get("x10_p_strong", x10.DEFAULT_P_STRONG)),
            scale_gate_sigma=float(cfg.get("x10_scale_gate_sigma", x10.DEFAULT_SCALE_GATE_SIGMA)),
            gate_alpha=float(cfg.get("x10_control_gate_alpha", x10.DEFAULT_CONTROL_GATE_ALPHA)),
            line_continuum_window_A=float(cfg.get("x10_line_continuum_window_A", x10.DEFAULT_LINE_CONTINUUM_WINDOW_A)),
        )
    finally:
        x10.COMPARISON_BANDS = original
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--project-root", default=None)
    ap.add_argument("--modes", nargs="+", default=["annulus", "azimuthal", "local_plane"])
    ap.add_argument("--width-A", type=float, default=DEFAULT_WIDTH_A)
    ap.add_argument("--out-json", required=True)
    args = ap.parse_args(argv)

    root = Path(args.project_root or Path.cwd()).resolve()
    report = {"script": "g_fondo_en_lambda", "run": args.run, "width_A": args.width_A, "modes": {}}
    bands = None
    for mode in args.modes:
        cfg, products, controls, g1 = products_for_mode(root, args.run, mode)
        if bands is None:
            bands = fine_bands(np.asarray(products["aperture"].wave_A, dtype=np.float64), args.width_A)
            print(f"{len(bands)} ventanas de {args.width_A:g} A")
        rows = rows_on_grid(cfg, products, controls, g1, bands)
        per_pair = {pid: {} for pid in PAIRS}
        for row in rows:
            if row["pair"] in PAIRS:
                per_pair[row["pair"]][row["band"]] = {
                    "lo_A": row.get("wave_min_A"),
                    "hi_A": row.get("wave_max_A"),
                    "t": x10._finite_or_none(row.get("t_stat")),
                    "p": x10._finite_or_none(row.get("p_value")),
                    "n_channels": row.get("n_channels"),
                    # El SESGO FRACCIONAL, que es lo que separa "el fondo es
                    # cromatico" de "la S/N del continuo sube al rojo y destapa
                    # un sesgo constante". El t crece con las dos cosas; esto no.
                    "ratio_i_over_j": row.get("ratio_i_over_j"),
                    "flux_i": row.get("flux_i"),
                    "flux_j": row.get("flux_j"),
                    "snr_j": (None if not row.get("err_j") or not row.get("flux_j")
                              else abs(row["flux_j"]) / row["err_j"]),
                }
        report["modes"][mode] = per_pair
        print(f"  {mode}: listo")
        Path(args.out_json).write_text(json.dumps(report, indent=1), encoding="utf-8")

    # Tabla: |t| del par que mas se mueve, ventana a ventana
    print(f"\n{'ventana':>12s} " + " ".join(f"{m:>12s}" for m in args.modes) + "   mejor")
    names = [b.name for b in bands]
    centers = {b.name: 0.5 * (b.lo_A + b.hi_A) for b in bands}
    for name in names:
        cells, vals = [], {}
        for mode in args.modes:
            t = report["modes"][mode]["optimal_ls_vs_aperture"].get(name, {}).get("t")
            vals[mode] = abs(t) if t is not None else None
            cells.append(f"{t:12.2f}" if t is not None else f"{'-':>12s}")
        finite = {m: v for m, v in vals.items() if v is not None}
        best = min(finite, key=finite.get) if finite else "-"
        print(f"{centers[name]:10.0f} A " + " ".join(cells) + f"   {best}")

    # Control
    print("\nControl psffit_vs_aperture (no depende de C3; debe ser identico):")
    diffs = []
    for name in names:
        ts = [report["modes"][m]["psffit_vs_aperture"].get(name, {}).get("t") for m in args.modes]
        ts = [t for t in ts if t is not None]
        if len(ts) > 1:
            diffs.append(max(ts) - min(ts))
    print(f"  max|diferencia entre modos| sobre {len(diffs)} ventanas: {max(diffs) if diffs else float('nan'):.3e}")
    print(f"\nescrito {args.out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
