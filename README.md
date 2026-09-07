# MUSE accretion pipeline

A staged, reproducible pipeline for VLT/MUSE narrow-field-mode (NFM) integral-field
data: raw-cube reduction, spectral extraction of a faint companion next to a much
brighter primary star, and hydrogen-line accretion analysis.

It is built around one idea: **every stage writes a machine-readable quality-control
document**, and later stages read those documents instead of trusting assumptions.
Thresholds, scientific decisions and provenance (input cube, SHA-256 hashes, spec
version, wavelength frame, flux unit) travel with the products, and a final gate
stage aggregates them into a single run report.

## What it does

- **Reduction** — ESO `esorex` driver, sky/ZAP handling, telluric correction, and a
  cube quality-control battery.
- **Localisation** — alignment and cropping, cross-correlation stripe diagnostics,
  and chromatic companion localisation.
- **PSF and extraction** — a chromatic PSF model of the primary (analytic Moffat and
  a physical adaptive-optics model), fitted per observation or on the combined cube,
  and **six independent extraction schemes** for the companion spanning a
  spatial-model family (aperture, optimal with two background treatments, full PSF
  fitting) and a spectral-diversity family. Running six in parallel is the point:
  disagreement between them is measured, not hidden.
- **Detection and limits** — line detection calibrated against empirical nulls from
  control positions processed identically to the object, an artefact battery,
  injection–recovery throughput, and contrast/ROC curves.
- **Inference** — line measurement, accretion rates and upper limits, source
  classification, and a final packaged report.

Noise is never taken from the instrument's `STAT` extension alone: uncertainties are
estimated empirically from controls that go through the same processing as the object.

## Status

**Under active development.** This is research code, developed alongside the analysis
it supports; it is the version used in Ramírez Reyes & Cáceres (in preparation).
Interfaces and stage specifications still change between releases. Run products are
not distributed with the code.

## Installation

The environment is pinned (Python 3.10, `astropy`, `numpy`, `scipy`; the raw-reduction
stages additionally need ESO `esorex` on the `PATH`).

```bash
conda env create --file environment.yml
conda activate MUSE
```

## Running the tests

From the repository root:

```bash
python -m pytest tests/ -q                        # full suite
python -m pytest tests/ -q -m "not slow"          # skip the notebook end-to-end tests
python -m pytest tests/ -q -m "not external_data" # skip tests needing external libraries
```

The suite covers the numerical core, the quality-control schemas and the stage
registry, and executes the analysis notebooks end to end to check that they still
reproduce the products of the stages they mirror.

## Usage

Every stage is available three ways: as a Python function, as a command-line module
accepting `--run-id`, and through review notebooks that audit a run's quality-control
documents. Products live under `runs/<RUN_ID>/`, which is not versioned. The stage
map, the run-selection rules and the per-object layout are documented in
[`README.es.md`](README.es.md) (Spanish) and in [`docs/`](docs/).

## How to cite

If this software contributes to work you publish, please cite it through the metadata
in [`CITATION.cff`](CITATION.cff). A citable archived version with a DOI is being
prepared; until it is available, cite the repository and the release tag.

## License

BSD 3-Clause. See [`LICENSE`](LICENSE).
