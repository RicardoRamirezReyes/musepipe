# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

Scientific MUSE/VLT NFM pipeline: raw-cube reduction, spectral extraction of a faint
companion next to a bright primary, and accretion (Hα) analysis. `AGENTS.md` holds the
agent policy (safety, conventions, coordination) and applies here too; `README.md` has the
full stage↔module map.

## Environment & commands

The conda env is `MUSE` (`environment.yml`, pinned; python 3.10). Everything runs from the
repo root.

```bash
conda env create --file environment.yml && conda activate MUSE   # first time
conda env update --name MUSE --file environment.yml --prune      # refresh

python -m pytest tests/ -q                       # full suite (723 tests, ~4.5 min)
python -m pytest tests/ -q -m "not slow"         # same minus the notebook end-to-end (~90 s)
python -m pytest tests/test_h03_chain.py -q      # one file
python -m pytest tests/ -q -k "aperture and not injection"
python -m pytest tests/ -q -m "not external_data"  # skip tests needing g3_libraries_root
python -m compileall musepipe tests stage08_full_spectrum_for_modeling.py
```

No linter or formatter is configured. Tests are almost all `unittest.TestCase` classes, but
pytest is the runner (`pytest.ini` sets `testpaths` and the `external_data` marker; a couple
of files are pytest-only). If a dependency is missing, report it — do not install it.

## Run selection (do this explicitly, always)

Products live in `runs/<RUN_ID>/{config,stages,tables,plots,logs,report}` and are
gitignored. A run is resolved as: explicit `run_id`/`--run-id` → `MUSE_RUN_ID` env var →
first non-comment line of `active_run.txt` (a local smoke-test default, usually
`ROXs12b_short` — never rely on it).

```bash
MUSE_RUN_ID=ROXs12b python -c "from musepipe.config import load_run_config; print(load_run_config().run_id)"
```

A mismatch between the run directory and `config/config.json` is an error by design.
`--allow-run-id-mismatch` / `allow_run_id_mismatch=True` is for diagnosing a known
historical run only.

`runs/<RUN>/config/config.json` has three top-level keys: `meta` (provenance,
`legacy: true` marks read-only historical runs), `config` (stage parameters), and `chain`
(see below). Never write into a run marked legacy.

## Architecture

### Canonical chain A→G

Stages are lettered blocks, each with a frozen spec in `docs/spec_<ID>_codex_*.md` (read
the highest version, e.g. `spec_D1_v3_*`) and a JSON QC product under
`runs/<RUN>/stages/`:

- **A1–A4** reduction: esorex raw reduction → ZAP sky decision → telluric → cube QC (M1–M5)
- **B1–B3** load/align/crop → xcorr stripes → companion localization
- **C1, 04b, C2–C6** chromatic PSF → local-surface background → **five stages but six
  extraction methods**: C3 emits two variants as separate products, so `METHOD_ORDER` is
  `aperture`, `optimal_ls` (background = 04b local surface), `optimal_psfsub` (background =
  C1's primary PSF model), `psffit`, `sgf`, `lpm`
- **D1–D2** inter-method comparison → spectral calibration. D2 is where the
  **definitive spectra** are delivered: the six companion methods plus the primary
  (`spec_calibrated_psffit_star.fits`), all with `BUNIT` and an error budget, plus
  the `spectra` table and `stage_x11_spectra.png` — a result in themselves, before
  the Hα study
- **E1–E6** Hα detection → artifact battery → Ṁ upper limits → injection-recovery →
  contrast/ROC curves
- **F1** final package + gate (`report/run_summary.json`)
- **G0–G5** real-cube execution → extraction validation → line measurement → physical
  inference → source classification → synthesis
- **S0/S1** wavelength-solution and Hα maps (side diagnostics)

`musepipe/stage_registry.py` is the machine-readable source of truth for this chain: for
each stage, its QC path (plus `qc_aliases` for QC names that differ by reduction profile,
e.g. A1 emits `stages/stage00r_qc.json` in the `monolithic` profile and
`cube_telcorr_qc.json` in `cascade`), its `exec_kind`, and the launch command template. It
is **stdlib-only** because notebooks import it without the scientific stack.
`scripts/build_review_notebooks.py` and `notebooks/_nbcommon.py` both validate against it —
adding or renaming a stage means editing this registry, not scattering strings.

### Three entry-point layers

Reusable logic lives in `musepipe/`; notebooks and shell scripts are thin wrappers.

1. **Python API** — `run_stage_x01(run_id, ...)`, `stage_x01_config_from_run()` etc.,
   re-exported from `musepipe.stages`. Every stage module also has an `argparse` `main()`
   accepting `--run-id`, `--project-root`, `--allow-run-id-mismatch`, plus stage overrides,
   and prints the path of the QC it wrote.
2. **Shell scripts** — `scripts/stage_*.sh` just pick a python with astropy, `cd` to the
   project root, and `python -m musepipe.stages.<module> "$@"`. Long/heavy jobs
   (reduction, per-exposure work, streaming combine) are `scripts/*.py`.
3. **Review notebooks** — `notebooks/<Object>/A1_…ipynb … G5_…ipynb`, one set per object,
   structurally identical (no object is "the canonical one"). They audit by default and can
   launch the executable stages.
4. **Analysis ("debug") notebooks** — `notebooks/<Object>/debug/`, generated by the optional
   `scripts/build_debug_notebooks.py`. These do the opposite of the review set: they redo the
   stage **inside the notebook**, with the numeric functions **copied verbatim** from
   `musepipe` (via `ast`, with their imports and constants), so the maths can be edited
   without touching the chain. Two guards make the copy safe: a **drift cell** that flags any
   function whose source no longer matches `musepipe`, and a **comparison cell** against the
   stage's real product — with default knobs it must report identical, and
   `tests/test_debug_notebooks.py` executes each notebook end to end to enforce exactly that
   (marked `slow`, ~90 s each — that is the bulk of the suite's runtime). Knobs are read from
   the **resolved** stage config (`stage_xNN_config_from_run`), never copied as literals: the
   stage fills in defaults the run does not spell out, and hardcoding them is precisely what
   made the first C3 notebook fail to reproduce the chain. Living in `debug/` is deliberate:
   `--check` and `test_notebook_qc_resolution.py` glob `notebooks/<obj>/*.ipynb`
   non-recursively. Covered: **C2**, **C3** (its two variants) and **C4** (the canonical psffit, with a channel-subsampling knob because the per-channel fit costs ~11 min for all 3681).

### Multi-object layout

`targets/<slug>.json` declares an object (aliases, references). A run's `chain` key declares
how one object's chain is split across runs:

```json
"chain": {"target": "ROXs42Bb", "default_run": "ROXs42Bb_realigned",
          "stage_runs": {"A1": "ROXs42Bb_raw"}, "reduction_profile": "cascade"}
```

`notebooks/_nbcommon.load_qc()` maps a QC path → stage (via `stage_registry`) →
`chain.stage_runs` → run, prints where the QC came from, and flags cross-object
contamination in red. Adding an object = new run + config with `chain`, a
`targets/<slug>.json`, and `python scripts/build_review_notebooks.py --target <slug>` — no
code changes. `--check` exits non-zero on cross-object QC resolution. Never hand-edit a
generated notebook; edit the builder and regenerate only the requested stages.

### Module map (`musepipe/`)

`config.py` run selection/validation + `resolve_stage_io`; `paths.py` `RunPaths` (the only
way to build run paths); `io.py` FITS/CSV/JSON + flux-unit resolution (`resolve_bunit`,
`bunit_to_cgs_scale`, `resolve_flux_unit`, `flux_unit_conflict`); `stats.py`, `spectral.py`, `apertures.py`,
`localfit.py`, `psf.py`, `stripes.py`, `covariance.py`, `injection.py`, `halosub.py`,
`parallel.py` shared science; `extraction/` `SpectrumProduct` (the canonical spectrum
container, versioned by `FORMAT_VERSION`) and the extractors; `reduction/` esorex driver,
sky/ZAP, telluric, per-exposure planning, streaming combine; `qc/` cube/frame QC, ghost
census, STAT-empirical, wavesol/Hα maps; `models/` BT-Settl, extinction, accretion
relations, tracks, template fitting; `report.py` (F1), `characterization.py` (G5),
`classify.py` (G4), `lines.py` (G2), `g0.py`.

## Scientific invariants

- **Noise**: never use the cube's `STAT` extension directly as σ, and never assume
  neighbouring pixels or channels are independent. STAT underestimates aperture noise ~4×;
  spatial inflation is ~6.5× (3×3) / ~17× (5×5); spectral n_eff/n ≈ 0.69. σ is always
  estimated empirically from **controls processed identically to the object**. Cite
  `docs/noise_model.md` instead of re-deriving these numbers.
- **Flux units**: the unit travels with the data (`BUNIT`) and there is **no silent
  default**. Resolve it with `musepipe.io.resolve_flux_unit` / `flux_unit_cgs`, which
  reads, in order, the config knob → the product's `BUNIT` → `m3_flux.flux_unit_cgs`
  from A4's QC, and raises otherwise. The declared knob wins on purpose (astropy parses
  MUSE's `BUNIT` as `1.0000000000000001e-20`, and making it authoritative would move
  frozen results in the last bit). Never reintroduce a `1.0`/`1e-20` fallback.
- **Conventions**: cubes are `zyx`, positions are `[y, x]`, quantities carry unit suffixes
  (`_A`, `_kms`, `_px`, `_arcsec`). Preserve deterministic seeds, sha256 hash chains, spec
  versions, QC schemas, and provenance fields — F1/G0/G5 verify them.
- Frozen thresholds, the canonical extraction method, and scientific decisions
  (`docs/*_decision*.md`, `docs/g3_real_frozen_decisions.md`) change only with explicit
  approval.

## Safety

- Check `git status` before editing; preserve unrelated work.
- Do not touch raw data, `runs/`, `/mnt/2TB`, calibrations, `reports/`, or `paper/` unless
  asked. Do not commit or push unless asked.
- Never run two writing stages against the same run concurrently. Raw reduction, full
  notebook executions, downloads, and long injections need approval — they are hours-long
  and overwrite products.
- The old root-level notebooks (`00_*`–`11_*`, `Far_*`) now live in `legacy/`, which is
  gitignored — they are historical snapshots, not the current interface, and a fresh clone
  will not have them (recover with `git checkout 225a8fc -- '*.ipynb'`). Dated plans and
  reports in `docs/` and `reports/` are snapshots too, and still cite the old root paths.
