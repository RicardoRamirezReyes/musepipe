# AGENTS.md

## Scope

Scientific MUSE/VLT NFM pipeline for cube reduction, spectral extraction, and
accretion analysis. Reusable logic lives in `musepipe/`, commands in `scripts/`,
and products in `runs/<RUN_ID>/`.

## Sources of Truth

- Use `README.md` for the general architecture.
- Read the latest `docs/spec_*.md` version for the affected stage.
- Follow `docs/noise_model.md` for noise, significance, and control handling.
- Treat root-level notebooks and dated plans or reports as historical snapshots.
- Do not change frozen thresholds, the canonical method, or scientific decisions
  without explicit approval.

## Safety

- Inspect `git status` before editing and preserve unrelated changes.
- Do not modify raw data, `runs/`, `/mnt/2TB`, external libraries,
  calibrations, reports, or `paper/` unless explicitly requested.
- Before running a stage, check active processes and
  `runs/<RUN>/config/config.json`.
- Always select a run explicitly with `--run-id` or `MUSE_RUN_ID`.
- Do not use `--allow-run-id-mismatch` except for a requested diagnosis.
- Do not run raw reduction, complete notebooks, downloads, long injections, or
  stages that overwrite products without approval.
- Do not install dependencies, change the environment, commit, or push unless
  explicitly requested.

## Tests

- Changing an existing test needs the owner's explicit authorization, given for
  that specific change. This covers anything that can turn a failing test into a
  passing one: expected values and pinned numbers, tolerances, `skip`/`xfail`/
  `skipIf`/`skipTest`, deleting or renaming a test, assertions, fixtures and
  helpers, `conftest.py`, `pytest.ini` and markers.
- A failing test is reported as a failure, with its output and why it fails,
  and stays failing until the owner decides. Never adjust a test to the code's
  new output, even when that output is expected (e.g. a pinned paper number
  after a code fix).
- To ask, name the test, the old and the new assertion, and why the failure is
  not a regression. Approval for one change does not extend to another.
- New tests may be added without asking, but they are listed in the report and
  never replace or weaken an existing one.

## Conventions

- Put reusable scientific logic in `musepipe/`, not notebooks.
- Reuse `RunPaths`, config loaders, IO helpers, and `SpectrumProduct`.
- Preserve `zyx` cube order, `[y, x]` positions, and unit-bearing suffixes such
  as `_A`, `_kms`, `_px`, and `_arcsec`.
- Preserve deterministic seeds, hashes, spec versions, QC, and provenance.
- Estimate noise from controls processed identically to the object; never use
  `STAT` directly as sigma or assume neighboring pixels are independent.
- Edit review notebooks through `scripts/build_review_notebooks.py` and
  regenerate only the requested stages.

## Verification

Use the existing `MUSE` environment. For normal changes:

```bash
python -m pytest tests/test_relevant_file.py -q
python -m pytest tests/ -q
python -m compileall musepipe tests stage08_full_spectrum_for_modeling.py
```

No linter or formatter is configured. If a dependency is missing, report it
instead of installing it silently.

## Agent Coordination

- The coordinating agent divides work into independent modules, tests, or
  documentation and owns integration.
- Exploration agents work read-only.
- Do not assign the same file to multiple agents concurrently.
- Only one agent may modify the notebook builder or generated notebooks.
- Never run two writing stages against the same run concurrently.
- Use isolated runs or separate worktrees for concurrent experiments.
- Each agent reports files changed, commands run, tests, generated products,
  assumptions, and unresolved scientific checkpoints.
- Run targeted tests per agent and the full suite once after integration.
