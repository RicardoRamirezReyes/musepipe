#!/usr/bin/env python3
"""S7b: Mdot upper limit vs extinction ladder (plan wavesol S7b).

Re-evaluates the E3 dereddened Halpha flux limit and the implied Mdot over a
grid of A_V WITHOUT re-running E3, reusing the exact H03 conversion chain
(`luminosity_erg_s`, `lacc_lsun_from_lha`, `mdot_msun_yr_from_lacc`). Starts
from the extinction-independent `f_lim_observed` of the canonical method and
applies the CCM A_Halpha/A_V ratio per A_V. This is the standard Hashimoto-style
caveat: a larger assumed extinction relaxes the accretion limit.

Outputs: runs/<run>/tables/mdot_limit_vs_extinction.csv + a figure, and adds an
`extinction_ladder` block to stage_h03_qc.json (additive). Diagnostic only.

    python scripts/s7b_mdot_vs_extinction.py [--run-id ROXs12b_realigned]
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from musepipe.config import load_run_config  # noqa: E402
from musepipe.stages.stage_h03_limits import (  # noqa: E402
    L_SUN_ERG_S,
    lacc_lsun_from_lha,
    luminosity_erg_s,
    mdot_msun_yr_from_lacc,
    physical_inputs_from_config,
)

DEFAULT_AV_GRID = (0.0, 1.0, 2.0, 4.0)


def _ladder_row(av, f_obs, phys):
    a_hoav = phys["a_halpha_over_av"]
    a_halpha = av * a_hoav
    ext = 10.0 ** (0.4 * av * a_hoav)
    f_dered = f_obs * ext
    lha = luminosity_erg_s(f_dered, phys["distance_pc"])
    lha_lsun = lha / L_SUN_ERG_S
    lacc = lacc_lsun_from_lha(lha_lsun, phys["lacc_lha_a"], phys["lacc_lha_b"])
    mdot = mdot_msun_yr_from_lacc(lacc, phys["companion_mass_msun"], phys["companion_radius_rsun"])
    row = {"a_v": av, "a_halpha": a_halpha, "ext_factor": ext,
           "f_lim_dereddened": f_dered, "l_halpha_lsun": lha_lsun,
           "l_acc_lsun": lacc, "mdot_msun_yr": mdot}
    # R1: parallel planetary-shock relation (Aoyama+21) on the same L_Halpha.
    aoyama = phys.get("lacc_aoyama21")
    if aoyama:
        lacc_a = lacc_lsun_from_lha(lha_lsun, aoyama["a"], aoyama["b"])
        mdot_a = mdot_msun_yr_from_lacc(lacc_a, phys["companion_mass_msun"], phys["companion_radius_rsun"])
        row["l_acc_aoyama21_lsun"] = lacc_a
        row["mdot_aoyama21_msun_yr"] = mdot_a
    return row


def _write_figure(rows, adopted_av, out_path):
    cache = Path(os.environ.get("TMPDIR", "/tmp")) / "musepipe_matplotlib"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache))
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    av = [r["a_v"] for r in rows]
    md = [r["mdot_msun_yr"] for r in rows]
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    ax.plot(av, md, "o-", color="C0")
    ax.axvline(adopted_av, color="tab:red", ls="--", label=f"adopted A_V={adopted_av:g}")
    ax.set_yscale("log")
    ax.set_xlabel("assumed A_V [mag]")
    ax.set_ylabel(r"$\dot{M}$ upper limit [$M_\odot$/yr]")
    ax.set_title("S7b: accretion limit vs assumed extinction (canonical method)")
    for r in rows:
        ax.annotate(f"A_Ha={r['a_halpha']:.1f}", (r["a_v"], r["mdot_msun_yr"]),
                    textcoords="offset points", xytext=(4, 6), fontsize=7)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="S7b Mdot vs extinction ladder.")
    ap.add_argument("--run-id", default="ROXs12b_realigned")
    ap.add_argument("--av-grid", default=None, help="comma-separated A_V values")
    args = ap.parse_args(argv)

    rc = load_run_config(args.run_id, allow_run_id_mismatch=True)
    cfg = dict(rc.config)
    phys = physical_inputs_from_config(cfg)
    run_dir = Path(rc.paths.run_dir) if hasattr(rc.paths, "run_dir") else Path("runs") / args.run_id

    h03_path = run_dir / "stages" / "stage_h03_qc.json"
    h03 = json.loads(h03_path.read_text())
    canonical = h03.get("canonical_method", "psffit")
    row = next(L for L in h03["limits"] if L["method"] == canonical)
    f_obs = float(row["f_lim_observed"])

    av_grid = ([float(x) for x in args.av_grid.split(",")] if args.av_grid
               else sorted(set(DEFAULT_AV_GRID) | {round(phys["av"], 3)}))
    rows = [_ladder_row(av, f_obs, phys) for av in av_grid]

    adopted = _ladder_row(phys["av"], f_obs, phys)
    zero = _ladder_row(0.0, f_obs, phys)
    aha2 = _ladder_row(2.0 / phys["a_halpha_over_av"], f_obs, phys)  # A_Halpha = 2
    relax_aha2 = aha2["mdot_msun_yr"] / zero["mdot_msun_yr"] if zero["mdot_msun_yr"] else float("nan")

    table_dir = run_dir / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    csv_path = table_dir / "mdot_limit_vs_extinction.csv"
    with csv_path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    plot_dir = run_dir / "plots" / "s7_extinction"
    plot_dir.mkdir(parents=True, exist_ok=True)
    fig_path = plot_dir / "mdot_vs_extinction.png"
    _write_figure(rows, phys["av"], fig_path)

    h03["extinction_ladder"] = {
        "canonical_method": canonical,
        "f_lim_observed": f_obs,
        "a_halpha_over_av": phys["a_halpha_over_av"],
        "extinction_source": phys["extinction_source"],
        "adopted_av": phys["av"],
        "adopted_mdot_msun_yr": adopted["mdot_msun_yr"],
        "av_grid": av_grid,
        "rows": rows,
        "relaxation_factor_at_a_halpha_2_vs_av0": relax_aha2,
        "note": ("Mdot upper limit vs ASSUMED A_V, canonical method, reusing the H03 "
                 "conversion chain from the extinction-independent f_lim_observed. Diagnostic "
                 "only (does not change the adopted E3 limit). Larger extinction relaxes the "
                 "limit (Hashimoto-style caveat)."),
        "table": str(csv_path), "figure": str(fig_path),
    }
    h03_path.write_text(json.dumps(h03, indent=2) + "\n", encoding="utf-8")

    print(f"S7b ({canonical}): A_V ladder {av_grid}")
    print("{:>5} {:>7} {:>8} {:>14} {:>12}".format("A_V", "A_Ha", "ext", "f_dered", "mdot"))
    for r in rows:
        print("{:>5.2f} {:>7.2f} {:>8.2f} {:>14.3e} {:>12.3e}".format(
            r["a_v"], r["a_halpha"], r["ext_factor"], r["f_lim_dereddened"], r["mdot_msun_yr"]))
    print(f"adopted A_V={phys['av']:g} -> mdot={adopted['mdot_msun_yr']:.3e}; "
          f"A_Halpha=2 relaxes vs A_V=0 by x{relax_aha2:.1f}")
    print(f"table -> {csv_path}\nfigure -> {fig_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
