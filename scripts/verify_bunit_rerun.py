#!/usr/bin/env python3
"""Compara los productos espectrales de un run contra una linea base previa.

Criterio de aceptacion de la re-ejecucion C2-C6 + D2 que estampa `BUNIT`
(plan A) y calibra la primaria (plan B): SOLO deben cambiar cabeceras y
aparecer productos nuevos. Si se mueve un flujo o un error, hay que parar.

Uso:
    python scripts/verify_bunit_rerun.py --run-id ROXs12b_realigned
    python scripts/verify_bunit_rerun.py --run-id X --snapshot   # crear linea base

La linea base vive en `runs/<RUN>/baseline_pre_bunit_rerun.json` (dentro de
`runs/`, que no se versiona, pero persiste entre sesiones).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from astropy.io import fits

ROOT = Path(__file__).resolve().parent.parent
BASELINE_NAME = "baseline_pre_bunit_rerun.json"


def _summarize(path: Path) -> dict:
    data = fits.getdata(path, 1)
    header = fits.getheader(path, 1)
    return {
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest()[:16],
        "n": int(len(data)),
        "flux_sum": float(np.nansum(data["flux"])),
        "flux_med": float(np.nanmedian(data["flux"])),
        "err_med": float(np.nanmedian(data["flux_err"])),
        "bunit": str(header.get("BUNIT", "")),
        "ncols": len(data.columns.names),
    }


def _is_spectrum(path: Path) -> bool:
    """True si es un SpectrumProduct.

    `spec_*.fits` tambien cubre diagnosticos que no son espectros (p.ej.
    `spec_optimal_clip_rejection.fits`, un mapa de rechazo de clipping).
    """
    try:
        data = fits.getdata(path, 1)
        return "flux" in getattr(data, "columns", None).names
    except Exception:
        return False


def _products(stage_dir: Path) -> list[Path]:
    return [p for p in sorted(stage_dir.glob("spec_*.fits")) if _is_spectrum(p)]


def snapshot(run_dir: Path) -> Path:
    base = {p.name: _summarize(p) for p in _products(run_dir / "stages")}
    out = run_dir / BASELINE_NAME
    out.write_text(json.dumps(base, indent=1))
    print(f"linea base de {len(base)} productos -> {out}")
    return out


def verify(run_dir: Path) -> int:
    path = run_dir / BASELINE_NAME
    if not path.exists():
        print(f"No hay linea base en {path}; crea una con --snapshot ANTES de re-ejecutar.")
        return 2
    base = json.loads(path.read_text())
    stage_dir = run_dir / "stages"
    problems: list[str] = []

    print(f"{'producto':44s} {'BUNIT':>8s} {'flujo':>8s} {'err':>8s}  unidad")
    for name, before in sorted(base.items()):
        product = stage_dir / name
        if not product.exists():
            problems.append(f"{name}: DESAPARECIO")
            continue
        now = _summarize(product)
        flux_ok = np.isclose(now["flux_sum"], before["flux_sum"], rtol=1e-12, atol=0.0)
        err_ok = np.isclose(now["err_med"], before["err_med"], rtol=1e-12, atol=0.0)
        if not flux_ok:
            rel = abs(now["flux_sum"] - before["flux_sum"]) / max(abs(before["flux_sum"]), 1e-30)
            problems.append(
                f"{name}: flujo {before['flux_sum']:.9e} -> {now['flux_sum']:.9e} (rel {rel:.2e})"
            )
        if not err_ok:
            problems.append(f"{name}: error {before['err_med']:.9e} -> {now['err_med']:.9e}")
        print(f"  {name:42s} {'cambia' if now['bunit'] != before['bunit'] else 'igual':>8s} "
              f"{'OK' if flux_ok else 'CAMBIA':>8s} {'OK' if err_ok else 'CAMBIA':>8s}  {now['bunit']}")

    nuevos = [p.name for p in _products(stage_dir) if p.name not in base]
    if nuevos:
        print("\nProductos NUEVOS:")
        for name in nuevos:
            summary = _summarize(stage_dir / name)
            header = fits.getheader(stage_dir / name, 1)
            print(f"  {name}  cols={summary['ncols']}  BUNIT={summary['bunit']!r} "
                  f"SOURCE={header.get('SOURCE', '')!r}")

    if problems:
        print("\nPROBLEMAS (parar y revisar):")
        for line in problems:
            print("  -", line)
        return 1
    print("\nOK: ningun flujo ni error se movio; solo cabeceras y productos nuevos.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--project-root", default=None)
    parser.add_argument("--snapshot", action="store_true",
                        help="crear la linea base en vez de verificar")
    args = parser.parse_args(argv)
    run_dir = Path(args.project_root or ROOT) / "runs" / args.run_id
    if not run_dir.exists():
        print(f"No existe {run_dir}")
        return 2
    if args.snapshot:
        snapshot(run_dir)
        return 0
    return verify(run_dir)


if __name__ == "__main__":
    sys.exit(main())
