#!/usr/bin/env python3
"""Rehace un plan de combinado con OTRA ley de pesos y nada más.

Misma geometría (centroides, ventanas, desplazamientos), otros `weight`: así la
diferencia entre el cubo viejo y el nuevo es la ley de pesos y solo ella.
Re-planificar con `plan_stream_combine.py` volvería a medir los centroides —
deterministas, pero no es lo que se quiere aislar.

Uso típico (vía B de `docs/2026-09-14_plan_via_b_recombinar_12b.md`):

    python scripts/reweight_stream_combine_plan.py \\
        --plan /mnt/2TB/.../final/stream_combine_plan.json \\
        --weight invvar --weight-table runs/<RUN>/stages/spec_perexp_qc.json \\
        --run-id ROXs12b_invvar --output /mnt/2TB/.../final_invvar/DATACUBE_FINAL.fits \\
        --out-plan /mnt/2TB/.../final_invvar/stream_combine_plan.json

Imprime el reparto del peso por noche antes y después, que es el número que
hay que contrastar con el `weight_share_by_night` del QC de C7 antes de
combinar nada. No escribe cubos: eso es `run_stream_combine.py --execute`.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from musepipe.reduction.stream_combine import (  # noqa: E402
    WEIGHT_MODES,
    StreamCombineError,
    load_weight_table,
    plan_from_dict,
    reweight_plan,
)


def weight_share_by_night(plan) -> dict[str, float]:
    total = sum(exp.weight for exp in plan.exposures)
    share: dict[str, float] = {}
    for exp in plan.exposures:
        night = str(exp.exposure_id).split("_")[0].replace("-", "")
        share[night] = share.get(night, 0.0) + exp.weight / total
    return dict(sorted(share.items()))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--plan", required=True, help="Plan JSON del combinado existente")
    parser.add_argument("--out-plan", required=True, help="Dónde escribir el plan nuevo")
    parser.add_argument("--weight", required=True, choices=WEIGHT_MODES)
    parser.add_argument("--weight-table",
                        help="Con --weight invvar: QC de C7 (--group-by none --combine invvar)")
    parser.add_argument("--weight-kind", choices=("cube", "measurement"), default="cube",
                        help="que tabla del QC de C7: `cube` (1/sigma^2 de los controles crudos, sin apcorr; "
                             "la que le corresponde a un cubo) o `measurement` (la que C7 usa para combinar "
                             "medidas, con apcorr_i^2 dentro; probada el 2026-09-14 y pierde n_eff)")
    parser.add_argument("--weight-aperture", help="Apertura del QC de C7 (por defecto la primera)")
    parser.add_argument("--run-id", help="run_id del plan nuevo (por defecto el del plan viejo)")
    parser.add_argument("--output", help="Cubo de salida del plan nuevo (por defecto el del viejo, "
                                         "que run_stream_combine se negará a pisar)")
    parser.add_argument("--overwrite", action="store_true", help="Pisar --out-plan si existe")
    args = parser.parse_args(argv)

    out_plan = Path(args.out_plan)
    if out_plan.exists() and not args.overwrite:
        raise StreamCombineError(f"refusing to overwrite {out_plan} (use --overwrite)")

    plan = plan_from_dict(json.loads(Path(args.plan).read_text(encoding="utf-8")))
    weight_table, weight_source = None, None
    if args.weight == "invvar":
        if not args.weight_table:
            raise StreamCombineError("--weight invvar needs --weight-table <spec_perexp_qc.json>")
        weight_table, weight_source = load_weight_table(args.weight_table, aperture=args.weight_aperture, kind=args.weight_kind)

    before = weight_share_by_night(plan)
    new_plan = reweight_plan(plan, args.weight, weight_table=weight_table, weight_source=weight_source)
    if args.run_id:
        new_plan = replace(new_plan, run_id=str(args.run_id))
    if args.output:
        new_plan = replace(new_plan, output=str(args.output))
    after = weight_share_by_night(new_plan)

    out_plan.parent.mkdir(parents=True, exist_ok=True)
    out_plan.write_text(json.dumps(new_plan.as_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"{len(plan.exposures)} exposures, geometry unchanged")
    print(f"  weight: {plan.weight_mode} -> {new_plan.weight_mode}")
    if new_plan.weight_source:
        print(f"  weights from {new_plan.weight_source['path']} "
              f"(sha256 {new_plan.weight_source['sha256'][:12]}, band {new_plan.weight_source['weight_band_A']} A)")
    print("  weight share by night:")
    for night in sorted(set(before) | set(after)):
        print(f"    {night}: {100 * before.get(night, 0):6.2f} % -> {100 * after.get(night, 0):6.2f} %")
    print(f"  run_id: {plan.run_id} -> {new_plan.run_id}")
    print(f"  output: {new_plan.output}")
    print(f"  plan -> {out_plan}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, StreamCombineError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
