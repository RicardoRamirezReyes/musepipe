# Gaia DR3 passbands — A4/M3 absolute flux-scale reference

Gaia passband curves are external calibration data. A4/M3 code
(`musepipe/qc/cube_qc.py`: `synthetic_band_flux`, `flux_factor_from_reference`,
`status_flux`) accepts passband arrays from config or caller code; this directory
is the stable location for the project's Gaia DR3 passband CSV files.

## Files (added 2026-07-09)

`gaia_dr3_{G,BP,RP}.csv` — columns `wavelength_A,response_photon` (photon
response / quantum efficiency), header on the first 3 lines.

- Source: **SVO Filter Profile Service** `GAIA/GAIA3.{G,Gbp,Grp}` (Rodrigo &
  Solano 2020), which mirrors the official Gaia (E)DR3 passbands of
  **Riello et al. 2021 (A&A 649, A3)**.
- Coverage: G 3200–10500 Å (731 pts), BP 3250–7500 Å (426 pts),
  RP 6100–10800 Å (471 pts).

## Primary star reference photometry (ROXs 12 A)

Gaia DR3 source **6048935358761628288** (SIMBAD SpT M0.0e), retrieved via
VizieR I/355/gaiadr3 / SVO on 2026-07-09:

| Band | mag | err | SVO Vega ZP (Jy) | pivot (Å) | ref F_λ (erg/s/cm²/Å) | MUSE overlap |
|------|-----|-----|------------------|-----------|-----------------------|--------------|
| G    | 13.274388 | 0.003892 | 3228.746 | 6217.59 | 1.227e-14 | 82.9% |
| BP   | 14.594782 | 0.008928 | 3552.013 | 5109.71 | 5.924e-15 | 66.2% |
| RP   | 12.120192 | 0.007076 | 2554.948 | 7769.02 | **1.801e-14** | **93.2%** |

Parallax 7.217 mas → d = 138.56 pc (consistent with `h03_distance_pc = 138.6`).
Reference F_λ is Vega-system at the SVO pivot wavelength: `F_λ = f_ν·c/pivot²`
with `f_ν = ZP_Jy · 10^(−0.4·mag) · 1e-23`.

## How M3 consumes this (config schema, run `ROXs12b_B_adp`)

The run config carries `m3_*` keys: `m3_primary_gaia_*` (photometry),
`m3_passbands` (per-band file/mag/zeropoint/pivot/ref_flambda/overlap),
`m3_recommended_band`, `m3_flux_unit_cgs` (1e-20, MUSE native BUNIT),
`m3_caveat`, `m3_status_note`.

## Caveats (read before using M3)

1. **No Gaia band lies fully inside the MUSE range** (4749.5–9349.5 Å). Use **RP**
   (least truncated, 93.2% in-band) or apply the per-band `muse_overlap_frac`.
   G is 82.9% in-band, BP only 66.2%.
2. **M3 orchestrator is not yet implemented** — `stage00q_qc_skeleton` still
   hardcodes `m3_flux: unavailable`. The helper functions above consume these
   config keys but nothing calls them into `stage00q_qc.json` yet.
3. M3 must run on a **total-flux PRIMARY spectrum from a flux-calibrated cube**
   (the psffit "star" amplitude is not total flux — a smoke test gave factor
   ≈0.51, indicative only).
