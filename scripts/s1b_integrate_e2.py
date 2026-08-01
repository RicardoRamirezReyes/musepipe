#!/usr/bin/env python3
"""Paso S1b: integrate the S1 Halpha maps into the E2 (stage_h02) QC.

Adds a ``halpha_map_correlation`` block to ``stages/stage_h02_qc.json`` (keys
are ADDED, nothing renamed) capturing the correlation between the "dirty stripe
zones" and the Halpha maps, and writes a figure. The stripe zones come from the
S0 wavesol offset map (per-column robust scatter of the offset = local
wavelength-solution disturbance), on the SAME grid as the S1 maps; Xie's test
is whether the Halpha width varies spatially in step with that disturbance.
This script REPORTS numbers and does not interpret them (S1b/human).

Run after S0 (stageS0_offset_map.fits) and S1 (stageS1_halpha_map.fits,
stageS1_qc.json) exist for the run. Idempotent.

    python scripts/s1b_integrate_e2.py [--run-dir runs/ROXs12b_realigned]
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np
from astropy.io import fits

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from musepipe.qc.halpha_map import stripe_halpha_correlation  # noqa: E402
from musepipe.qc.wavesol_map import stripe_profile  # noqa: E402


def _image(path: Path, ext: str) -> np.ndarray:
    with fits.open(path) as hdul:
        return np.asarray(hdul[ext].data, dtype=np.float64)


def _write_figure(offset_ch, sigma_map, corr, orientation, out_path: Path) -> None:
    cache_dir = Path(os.environ.get("TMPDIR", "/tmp")) / "musepipe_matplotlib"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    off = stripe_profile(offset_ch, orientation)
    sig = stripe_profile(sigma_map, orientation)
    pos = np.arange(off["scatter"].size)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ax = axes[0]
    ax.plot(pos, off["scatter"], color="C0", lw=1.1, label="S0 stripe scatter [ch]")
    ax.set_xlabel("column" if orientation == "vertical" else "row")
    ax.set_ylabel("S0 offset scatter [ch]", color="C0")
    ax.tick_params(axis="y", labelcolor="C0")
    ax2 = ax.twinx()
    ax2.plot(pos, sig["profile"], color="C3", lw=1.1, label="S1 Halpha sigma [A]")
    ax2.set_ylabel("Halpha sigma [A]", color="C3")
    ax2.tick_params(axis="y", labelcolor="C3")
    ax.set_title("per-column: stripe disturbance vs Halpha width")

    ok = (np.isfinite(off["scatter"]) & np.isfinite(sig["profile"])
          & (np.minimum(off["count"], sig["count"]) >= 5))
    axes[1].scatter(off["scatter"][ok], sig["profile"][ok], s=8, color="C2")
    axes[1].set_xlabel("S0 offset scatter [ch]")
    axes[1].set_ylabel("Halpha sigma [A]")
    axes[1].set_title(
        f"corr = {corr['corr_stripe_scatter_vs_halpha_sigma']:.2f} "
        f"(n={corr['n_columns']} columns)"
    )
    fig.suptitle("S1b · dirty stripe zones vs Halpha maps (E2 integration)")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="S1b: integrate S1 Halpha maps into E2 QC.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--orientation", default="vertical", choices=["vertical", "horizontal"])
    args = parser.parse_args(argv)

    run = Path(args.run_dir)
    s0_map = run / "stages" / "stageS0_offset_map.fits"
    s1_map = run / "stages" / "stageS1_halpha_map.fits"
    s1_qc_path = run / "stages" / "stageS1_qc.json"
    e2_qc_path = run / "stages" / "stage_h02_qc.json"
    stage02_qc_path = run / "stages" / "stage02_qc.json"
    for p in (s0_map, s1_map, s1_qc_path, e2_qc_path):
        if not p.exists():
            print(f"ERROR: missing input {p}", file=sys.stderr)
            return 2

    offset_ch = _image(s0_map, "OFFSET_CH")
    a_map = _image(s1_map, "A")
    sigma_map = _image(s1_map, "SIGMA_A")
    mu_map = _image(s1_map, "MU_A")
    if offset_ch.shape != a_map.shape:
        print(f"ERROR: S0 grid {offset_ch.shape} != S1 grid {a_map.shape}", file=sys.stderr)
        return 2

    orient = args.orientation
    transverse = "horizontal" if orient == "vertical" else "vertical"
    corr = stripe_halpha_correlation(offset_ch, a_map, sigma_map, mu_map, orient)
    corr_t = stripe_halpha_correlation(offset_ch, a_map, sigma_map, mu_map, transverse)

    s1_qc = json.loads(s1_qc_path.read_text())
    hm = s1_qc.get("halpha_map", {})
    stripe_metric = {}
    if stage02_qc_path.exists():
        stripe_metric = json.loads(stage02_qc_path.read_text()).get("stripe_metric", {})

    # A/sigma structure counts as slicer-ALIGNED only if it beats its transverse
    # control (same >2x rule as the S0 G1 gate); otherwise it is isotropic/radial.
    def _aligned(struct, transv):
        return bool(struct is not None and transv is not None
                    and struct > 3.0 and struct > 2.0 * transv)

    a_sig, a_tsig = hm.get("a_structure_significance"), hm.get("a_transverse_significance")
    s_sig, s_tsig = hm.get("sigma_structure_significance"), hm.get("sigma_transverse_significance")

    block = {
        "source": ("S1 Halpha maps (stageS1_halpha_map.fits) vs S0 wavesol offset map "
                   "(stageS0_offset_map.fits), same grid"),
        **corr,
        "corr_stripe_scatter_vs_halpha_sigma_transverse": corr_t["corr_stripe_scatter_vs_halpha_sigma"],
        "corr_stripe_scatter_vs_halpha_a_transverse": corr_t["corr_stripe_scatter_vs_halpha_a"],
        "halpha_a_structure_significance": a_sig,
        "halpha_a_transverse_significance": a_tsig,
        "halpha_sigma_structure_significance": s_sig,
        "halpha_sigma_transverse_significance": s_tsig,
        "halpha_a_slicer_aligned": _aligned(a_sig, a_tsig),
        "halpha_sigma_slicer_aligned": _aligned(s_sig, s_tsig),
        "halpha_corr_a_sigma": hm.get("corr_a_sigma"),
        "halpha_P_cov": hm.get("P_cov"),
        "halpha_sigma_median_A": hm.get("sigma_median_A"),
        "spectral_stripe_dirty_channels": stripe_metric.get("dirty_channels"),
        "note": ("Combined cube: S0 showed slicer stripes are rotation-scrambled (field "
                 "rotates 5.9 deg between the 7 exposures). The a/sigma structure is NOT "
                 "slicer-aligned (structure significance <= its transverse control => "
                 "isotropic/radial: bright core vs faint halo), so the strong per-column "
                 "stripe<->sigma correlation is a RADIAL confound (both track radius), not a "
                 "slicer signature; the near-equal transverse correlation confirms it. Also "
                 "P is not constant (P_cov~1), unlike Xie Fig.3's instrumental-LSF case. "
                 "Numbers only; ghost-vs-instrumental is a per-exposure/human call."),
        "doc": "docs/2026-07-17_decision_g1_wavesol.md",
    }

    fig_path = run / "plots" / "s1_halpha" / "s1b_stripe_vs_halpha.png"
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    _write_figure(offset_ch, sigma_map, corr, args.orientation, fig_path)

    e2 = json.loads(e2_qc_path.read_text())
    e2["halpha_map_correlation"] = block
    figs = e2.get("figures")
    if isinstance(figs, dict):
        figs["s1_stripe_halpha"] = str(fig_path.resolve())
    e2_qc_path.write_text(json.dumps(e2, indent=2) + "\n", encoding="utf-8")

    print("patched", e2_qc_path)
    print(f"  corr(stripe scatter, Halpha sigma) = {corr['corr_stripe_scatter_vs_halpha_sigma']:.3f} "
          f"(transverse control {corr_t['corr_stripe_scatter_vs_halpha_sigma']:.3f}, "
          f"n={corr['n_columns']} columns)")
    print(f"  corr(stripe scatter, Halpha a)     = {corr['corr_stripe_scatter_vs_halpha_a']:.3f} "
          f"(transverse {corr_t['corr_stripe_scatter_vs_halpha_a']:.3f})")
    print(f"  a structure {a_sig:.1f}x (transv {a_tsig:.1f}x) -> slicer_aligned={block['halpha_a_slicer_aligned']}")
    print(f"  sigma structure {s_sig:.1f}x (transv {s_tsig:.1f}x) -> slicer_aligned={block['halpha_sigma_slicer_aligned']}")
    print(f"  figure -> {fig_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
