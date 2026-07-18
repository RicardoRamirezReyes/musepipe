#!/usr/bin/env python3
"""Regenerate the 7 per-exposure MUSE cubes (plan wavesol/stripes, paso S2b+S2c).

Rebuilds the pruned reduction intermediates and produces one DATACUBE_FINAL +
SKY_SPECTRUM per science exposure on the SHARED (plan-B manual OFFSET_LIST) WCS
grid, so S0 and S3 can be run per exposure to confirm the G1 closure
(docs/decision_g1_wavesol_2026-07-17.md).

This replaces the lost scratchpad `run_cascade.py`: it re-runs esorex directly
against the SOFs that survived under `runs/ROXs12b_raw/raw_reduction/sof/`
(their inputs are exact and complete — those recipes ran originally). It applies
the same fixes the old driver/orchestrator did:
  * esorex writes to the CWD (its --output-dir is not honored) -> run each recipe
    with cwd = its output dir, which is exactly the path the downstream SOFs
    reference, so products regenerate in place;
  * recipe params go AFTER the recipe name;
  * muse_scipost keeps its default skymethod, the only override is --save
    (spec_A1 §3); here --save=cube,skymodel to also emit SKY_SPECTRUM.

Per-exposure scipost SOF = the surviving `muse_scipost_exp{i}.sof` (24 PIXTABLE
of that exposure + STD_RESPONSE/EXTINCT/ASTROMETRY/SKY_LINES/LSF) plus the
manual OFFSET_LIST line (from `muse_scipost_aligned.sof`) so all 7 share the
combined cube's grid.

Usage (long; run detached):
    python scripts/regen_perexp_cubes.py --step all
    python scripts/regen_perexp_cubes.py --step scipost --exp 3   # resume one
"""
from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent.parent
RR = ROOT / "runs" / "ROXs12b_raw" / "raw_reduction"
SOF = RR / "sof"
PROD = RR / "products"
OFFSET_LIST = PROD / "muse_offset_manual" / "OFFSET_LIST.fits"
COMBINED_CUBE = PROD / "muse_scipost_aligned" / "DATACUBE_FINAL.fits"
PEREXP = Path("/mnt/2TB/MUSE_work/ROXs12b_perexp")
ESOREX = "esorex"
N_EXP = 7


def _run_esorex(recipe: str, sof: Path, outdir: Path, params: list[str]) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    log = outdir / f"{recipe}.log"
    cmd = [ESOREX, f"--log-dir={outdir}", recipe, *params, str(sof)]
    print(f"[{time.strftime('%H:%M:%S')}] {recipe}: cwd={outdir}\n    {' '.join(cmd)}", flush=True)
    t0 = time.time()
    with log.open("w") as fh:
        proc = subprocess.run(cmd, cwd=str(outdir), stdout=fh, stderr=subprocess.STDOUT)
    dt = time.time() - t0
    print(f"[{time.strftime('%H:%M:%S')}] {recipe}: exit={proc.returncode} in {dt/60:.1f} min (log {log})", flush=True)
    if proc.returncode != 0:
        raise SystemExit(f"{recipe} failed (exit {proc.returncode}); see {log}")


def _count(outdir: Path, pattern: str) -> int:
    return len(list(outdir.glob(pattern)))


def _gate(cond: bool, msg: str) -> None:
    if not cond:
        raise SystemExit(f"GATE FAILED: {msg}")
    print(f"    gate OK: {msg}", flush=True)


def step_bias() -> None:
    out = PROD / "muse_bias"
    _run_esorex("muse_bias", SOF / "muse_bias.sof", out, [])
    _gate(_count(out, "MASTER_BIAS-*.fits") == 24, f"24 MASTER_BIAS (got {_count(out, 'MASTER_BIAS-*.fits')})")


def step_flat() -> None:
    out = PROD / "muse_flat"
    _run_esorex("muse_flat", SOF / "muse_flat.sof", out, [])
    _gate(_count(out, "MASTER_FLAT-*.fits") == 24, f"24 MASTER_FLAT (got {_count(out, 'MASTER_FLAT-*.fits')})")
    _gate(_count(out, "TRACE_TABLE-*.fits") == 24, f"24 TRACE_TABLE (got {_count(out, 'TRACE_TABLE-*.fits')})")


def step_scibasic() -> None:
    out = PROD / "muse_scibasic_object"
    _run_esorex("muse_scibasic", SOF / "muse_scibasic_object.sof", out, [])
    n = _count(out, "PIXTABLE_OBJECT_*.fits")
    _gate(n == 168, f"168 PIXTABLE_OBJECT (got {n})")


def _build_scipost_sof(exp: int) -> Path:
    base = (SOF / f"muse_scipost_exp{exp}.sof").read_text().rstrip("\n")
    text = base + f"\n{OFFSET_LIST} OFFSET_LIST\n"
    dst = SOF / f"muse_scipost_exp{exp}_offset.sof"
    dst.write_text(text)
    return dst


def _wcs_ok(cube: Path) -> bool:
    """Gate the SPECTRAL grid only (CRVAL3/CD3_3/CRPIX3/NAXIS3).

    Per-exposure scipost centers each cube on its OWN exposure pointing, so the
    spatial WCS (CRVAL1/2, NAXIS1/2) legitimately differs by the dither/pointing
    (~1"); B1 aligns them for any later stacking. What must match for wavelength
    / stripe work is the spectral axis, which is identical across all exposures.
    The spatial offset vs the combined cube is reported for information.
    """

    from astropy.io import fits

    with fits.open(cube) as h, fits.open(COMBINED_CUBE) as hc:
        hd = h["DATA"].header if "DATA" in h else h[1].header
        hdc = hc["DATA"].header if "DATA" in hc else hc[1].header
        ok = True
        for k, atol in (("CRVAL3", 0.05), ("CD3_3", 1e-4), ("CRPIX3", 1e-6)):
            a, b = hd.get(k), hdc.get(k)
            if a is None or b is None or abs(float(a) - float(b)) > atol:
                print(f"    spectral WCS mismatch {k}: {a} vs {b}", flush=True)
                ok = False
        if int(hd.get("NAXIS3", 0)) != int(hdc.get("NAXIS3", 0)):
            print(f"    NAXIS3 mismatch: {hd.get('NAXIS3')} vs {hdc.get('NAXIS3')}", flush=True)
            ok = False
        dra = (float(hd.get("CRVAL1", 0)) - float(hdc.get("CRVAL1", 0))) * 3600.0
        ddec = (float(hd.get("CRVAL2", 0)) - float(hdc.get("CRVAL2", 0))) * 3600.0
        print(f"    spatial offset vs combined: dRA={dra:+.2f}\" dDEC={ddec:+.2f}\" "
              f"(size {hd.get('NAXIS1')}x{hd.get('NAXIS2')}) — expected per-exposure",
              flush=True)
    return ok


def step_scipost(exps: list[int]) -> None:
    for i in exps:
        out = PEREXP / f"exp{i}"
        cube = out / "DATACUBE_FINAL.fits"
        # resumable: skip an exposure already produced and valid
        if cube.exists() and list(out.glob("SKY_SPECTRUM*.fits")) and _wcs_ok(cube):
            print(f"    exp{i} already done (cube+SKY_SPECTRUM+spectral WCS OK) — skipping", flush=True)
            continue
        sof = _build_scipost_sof(i)
        _run_esorex("muse_scipost", sof, out, ["--save=cube,skymodel"])
        _gate(cube.exists(), f"exp{i} DATACUBE_FINAL written")
        _gate(bool(list(out.glob("SKY_SPECTRUM*.fits"))), f"exp{i} SKY_SPECTRUM written")
        _gate(_wcs_ok(cube), f"exp{i} spectral WCS matches combined cube")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Regenerate the 7 per-exposure cubes (S2b+S2c).")
    ap.add_argument("--step", default="all",
                    choices=["all", "bias", "flat", "scibasic", "scipost"])
    ap.add_argument("--exp", type=int, default=None, help="Single exposure for --step scipost (1..7).")
    args = ap.parse_args(argv)

    if not shutil.which(ESOREX):
        return int(bool(print(f"ERROR: esorex not on PATH", file=sys.stderr)) or 2)
    for p in (SOF, OFFSET_LIST, COMBINED_CUBE):
        if not p.exists():
            print(f"ERROR: missing prerequisite {p}", file=sys.stderr)
            return 2

    t0 = time.time()
    if args.step in ("all", "bias"):
        step_bias()
    if args.step in ("all", "flat"):
        step_flat()
    if args.step in ("all", "scibasic"):
        step_scibasic()
    if args.step in ("all", "scipost"):
        exps = [args.exp] if args.exp else list(range(1, N_EXP + 1))
        step_scipost(exps)
    print(f"[{time.strftime('%H:%M:%S')}] DONE step={args.step} in {(time.time()-t0)/60:.1f} min", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
