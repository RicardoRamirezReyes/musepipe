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

python -m pytest tests/ -q                       # full suite (1261 tests + 1118 subtests, ~50 min)
python -m pytest tests/ -q -m "not slow"         # same minus the notebook end-to-end (~3.5 min)
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
first non-comment line of `active_run.txt` (a local default, currently
`ROXs12b_realigned` — never rely on it).

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
the highest version — today `spec_D1_v4_*`, `spec_A3_v3_*`, `spec_E4_v3_*`) and a JSON QC product under
`runs/<RUN>/stages/`:

- **A1–A4** reduction: esorex raw reduction → ZAP sky decision → telluric → cube QC (M1–M5).
  A4 is five subcommands, not one: `check-cube` (which rewrites the WHOLE document and now
  refuses to discard measured metrics without `--force`), `m3-flux`, `m1m2-sky` and `m4m5`,
  plus `finalize` — a pure JSON→JSON pass that recomputes everything **derived**
  (`status`/`status_detail`, `open_issues` with a declared `priority`,
  `downstream_decision`, per-metric `measured_utc`). Those five fields used to be
  hand-written or vintage-dependent, which is why the six `stage00q_qc.json` on disk had
  four different shapes. `stage_registry` publishes the **whole sequence**
  (`m1m2-sky` → `m3-flux` → `m4m5` → `finalize`, in that order); it used to publish `m3-flux`
  alone, which is how one object went a month without M4/M5. `check-cube` stays out on
  purpose — it rewrites the whole document, so in the sequence it would abort every re-run.
  A test in `tests/test_stage_registry.py` introspects A4's argparse and fails if a
  subcommand is not published
- **B1–B3** load/align/crop → xcorr stripes → companion localization
- **C1, 04b, C2–C6** chromatic PSF → local-surface background → **five stages but six
  extraction methods**: C3 emits two variants as separate products, so `METHOD_ORDER` is
  `aperture`, `optimal_ls` (background = 04b local surface), `optimal_psfsub` (background =
  C1's primary PSF model), `psffit`, `sgf`, `lpm`
- **C7** (optional, off the chain) extracts the companion **in each exposure with its own
  PSF** and combines the **measurements**, instead of combining cubes and extracting once.
  It is deliberately **not** in `METHOD_ORDER` — a new method there becomes mandatory for
  every run at once and drags D1, D2, E1, E3, E4, G1 and F1 — so D2 does not calibrate it
  and it never enters block E: it is a **reference/validation** product. Measured on
  ROXs 12 b (`docs/2026-08-26_perexp_medido_y_la_noche_mala.md`): S/N **ties** with the
  combined cube (0.97–1.09×), but the per-exposure `apcorr` removes the **−8.52 %**
  chromatic drift that C1's V4 measures as +8.2 %. What it really buys is weighting by
  quality: the cube weights by `exptime` and hands **38.1 %** of the weight to the worst
  night, where inverse-variance leaves it 2.6 %.
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
each stage, its QC path (plus `qc_aliases` for the other names that QC has had: B2 wrote
`stage02_qc.json` before `stage02_xcorr_qc.json`, and A1 leaves **two different documents** —
the wrapper `stage00r_qc.json`, which declares the whole reduction with its phases and the
V1–V6 battery, and `cube_telcorr_qc.json`, the voxel-combine QC, which documents one step.
The wrapper is what the A2 gate reads; `musepipe/reduction/a1_verify.py` rebuilds it from
disk. This is **not** a `monolithic`/`cascade` split: both objects of this harvest are
`cascade` and ROXs 12 b's wrapper carries `fase0…fase3` + `V1…V6`), its `exec_kind`, and the
launch command **sequence** (a stage can need several commands — `Stage.launch` maps a
reduction profile to a tuple of them, run in order and aborted on the first failure). It is
**stdlib-only** because notebooks import it without the scientific stack. `scripts/build_review_notebooks.py`, `notebooks/_nbcommon.py` and
`musepipe/report.py` (F1) all resolve against it — adding or renaming a stage means editing
this registry, not scattering strings. F1 did keep its own copy of the filenames until
2026-08-24, and that cost it two false blocking `required QC missing` on ROXs 42B b.

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
   (marked `slow`, ~2 min each across 20 notebooks — that is the bulk of the suite's runtime).
   Knobs are read from
   the **resolved** stage config (`stage_xNN_config_from_run`), never copied as literals: the
   stage fills in defaults the run does not spell out, and hardcoding them is precisely what
   made the first C3 notebook fail to reproduce the chain. Living in `debug/` is deliberate:
   `--check` and `test_notebook_qc_resolution.py` glob `notebooks/<obj>/*.ipynb`
   non-recursively (note `--check` is **not** read-only: it regenerates the review notebooks
   — but of **one** target only, so after a change to the launch cell the other object keeps
   the stale one until you re-run with an explicit `--target`). Covered: **C1**, **A3**, **C2**, **C3** (its two variants), **C4** (the
   canonical psffit, with a channel-subsampling knob because the per-channel fit costs ~11 min
   for all 3681), **C5** and **C6** (which share one builder: same skeleton, different
   subtraction), and **D2** — the only one about the **primary star** rather than the companion:
   it redoes D2's star calibration (cheap and exact) and then measures the primary **in each
   per-exposure cube** (`perexp_cubes`/`perexp_dir` in the run config), which no stage does. Its
   comparison also guards freshness: a calibrated product older than its C4 input means D2 has
   not been re-run, and the slow test skips instead of failing. Two more live in `debug/` and
   are **not** stage notebooks: `apcorr_debug` (recomputes the aperture correction with today's
   code) and `residuos_debug`.

   **C1 is the exception to the "compare against the product" rule**, because C1 does not emit a
   spectrum: it emits `psf_model.json`, the PSF every other stage consumes. Its per-bin fit is
   expensive (~43 fits) so it is subsampled by a knob, but the model-document builder is a *pure
   function of the per-bin rows* and those rows are in the stage's CSVs written with `repr` — so
   the notebook rebuilds the model document from the CSV, exactly and without touching the cube,
   and that is what `IDÉNTICO` compares. Its long section is the **evaluation**:
   `_evaluate_psfao` is copied verbatim (inverting the usual cut, where `evaluate_psf_model` is
   imported as "C1's, not what is fitted here") because it is the function under audit — the QC
   advertises `polynomial_deg2` smoothing but while `param_table` exists the polynomial is never
   evaluated; what varies per channel is a linear interpolation of the bin table, on a λ snapped
   to 50 Å. **Both PSF forms travel in the copy**: C1 always fits Moffat and Psfao and keeps the
   lower ring residual. **Since 2026-08-14 both objects deliver `psfao`, imposed by config**
   (`e01_psf_form`) rather than selected: in ROXs 42B b the ring metric actually prefers Moffat
   (6.88 % vs 10.05 %), and what rules Moffat out is its encircled energy, +356 % off the data.
   The copy still carries both forms, so a notebook that knew only one would crash if that knob
   changed back.

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

**Provenance in time.** A run's stages can have different vintages: changing the input
cube and re-running only part of the chain leaves the rest describing a dataset that is no
longer there (that is what happened to `ROXs12b_realigned` on 2026-07-28 — see
`docs/2026-08-07_procedencia_cubo_por_ob.md`). `_nbcommon.stage_vintage()` dates every
stage against the cube B1 declares in its QC and flags the older ones; `show_chain()` calls
it, so every review notebook prints it, and `--check` reports it as a **warning, not a
failure** — it is the state of the run, not a code error. It compares against the cube and
nothing else on purpose: chaining "each stage against the previous one" using the
`stage_registry` order marks 29 of 32 stages as soon as A3 alone is re-run, and a warning
that fires when it shouldn't gets ignored. For known product pairs the fine-grained guard
lives where the pair is known (D2's debug notebook checks its calibrated star against C4's).

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
`classify.py` (G4), `lines.py` (G2), `g0.py`; `telluric_lines.py` telluric **bands** (not
lines: MUSE does not resolve them) + the transmission curve A3 actually measured for the
run; `paper_spectrum.py` the publication figure (unbinned spectrum, both error estimates,
telluric bands and accretion lines) and its ECSV export.

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
- **Wavelength frame**: same rule, same reason. `_wavelength_frame`
  (`stages/stage_x01_aperture.py`, shared by C2–C5) resolves the declared knob
  (`x0N_wframe`/`wavelength_frame`) → A4's `cube.wavelength_frame` → `unavailable`, and
  stamps `WFRAME` on the product. It used to read the QC first with the knob as a mere
  `.get` default, so an A4 that said `unknown` — the cube header carries no `SPECSYS`
  — silently discarded the run's `barycentric` and stamped `topocentric` on ROXs 42B b's
  definitive spectra (vbary = −29.64 km/s ≈ 0.65 Å at Hα). Nothing shifts λ by vbary, so
  no number moved; what failed were the gates, all disabled by that same `unknown`. Never
  reintroduce a silent `topocentric` fallback.
- **Positions cross frames silently**: `m3_primary_yx` of ROXs 12 b is declared in the
  uncropped 338×330 cube while A4 measures on a 200×200 one. Verify a declared position
  against the data (`cube_qc.resolve_primary_yx` checks it against the brightness peak)
  instead of trusting the key.
- **Conventions**: cubes are `zyx`, positions are `[y, x]`, quantities carry unit suffixes
  (`_A`, `_kms`, `_px`, `_arcsec`). Preserve deterministic seeds, sha256 hash chains, spec
  versions, QC schemas, and provenance fields — F1/G0/G5 verify them.
- Frozen thresholds, the canonical extraction method, and scientific decisions
  (`docs/*_decision*.md`, `docs/2026-07-16_g3_real_frozen_decisions.md`) change only with explicit
  approval.
- `docs/` has two naming conventions and the difference is load-bearing: **contracts and
  standing references carry no date** (`spec_<ID>_codex_*.md` — read the highest version —
  plus `noise_model.md`, `a3_telluric_justification.md`, `00_config_parameters.md`,
  `setup_ubuntu.md`), while **states and procedures are `YYYY-MM-DD_<subject>.md`**
  (handoffs, plans, execution logs, audits), dated by the work, newest wins. Start any
  session at the newest `docs/*_handoff.md`.

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
