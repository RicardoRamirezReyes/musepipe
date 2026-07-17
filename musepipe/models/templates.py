"""Empirical spectral templates behind the ``SpectralLibrary`` protocol
(spec G3 §3.2, plan WP-G3R-4). One class covers both gravity classes:

* ``young``  — X-shooter Class III templates (Manara et al. 2013/2017, D1),
* ``field``  — SDSS field-dwarf templates (Kesseli et al. 2017, D2).

Unlike the atmosphere grid, empirical templates are NOT interpolated between
spectral types (§3.2): ``get(spt=)`` returns the nearest available SpT and
records the distance in ``meta['spt_delta']``. Resolution is surfaced via
``meta['resolution_fwhm_A']`` so :func:`musepipe.models.prep.prepare_template`
can apply the D2 rule (don't degrade a template coarser than the LSF).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import TemplateSpectrum
from ..constants import spt_code, spt_label
from .cache import load_spectrum_npz
from .manifest import verify_manifest

_GRAVITY_CLASSES = ("young", "field")


class EmpiricalTemplateLibrary:
    """Empirical template library adapter (``SpectralLibrary``)."""

    def __init__(self, family_dir, *, gravity_class, citation, version,
                 resolution_fwhm_A=None):
        if gravity_class not in _GRAVITY_CLASSES:
            raise RuntimeError(
                f"gravity_class must be one of {_GRAVITY_CLASSES}, got {gravity_class!r}")
        if not citation:
            raise RuntimeError(
                "EmpiricalTemplateLibrary requires a citation (spec G3 §1.2).")
        self.family_dir = Path(family_dir)
        self.gravity_class = gravity_class
        self.citation = str(citation)
        self.version = version
        self.resolution_fwhm_A = (float(resolution_fwhm_A)
                                  if resolution_fwhm_A is not None else None)
        self.manifest_info = verify_manifest(self.family_dir)

        # {spt_code -> [(path, source_meta), ...]}; several objects may share a
        # SpT (young library) — get() returns the first deterministically.
        self._by_code: dict[float, list] = {}
        for path in sorted(self.family_dir.glob("*.npz")):
            with np.load(path, allow_pickle=False) as z:
                meta = json.loads(str(z["meta_json"]))
            code = spt_code(str(meta["spt"]))
            self._by_code.setdefault(code, []).append((path, meta))
        if not self._by_code:
            raise RuntimeError(f"no template spectra under {self.family_dir}")
        self._codes = np.array(sorted(self._by_code), dtype=np.float64)

    # -- protocol ---------------------------------------------------------- #
    def grid(self) -> dict:
        return {"spt": self._codes.copy()}

    def get(self, *, spt, **_) -> TemplateSpectrum:
        code = spt_code(spt) if isinstance(spt, str) else float(spt)
        i = int(np.argmin(np.abs(self._codes - code)))
        nearest = float(self._codes[i])
        path, src_meta = self._by_code[nearest][0]
        spec = load_spectrum_npz(path)
        fwhm = src_meta.get("resolution_fwhm_A", self.resolution_fwhm_A)
        meta = {
            **spec.meta,
            "spt": src_meta.get("spt", spt_label(nearest)),
            "spt_code": nearest,
            "spt_requested": code,
            "spt_delta": code - nearest,
            "gravity_class": self.gravity_class,
            "citation": self.citation,
            "resolution_fwhm_A": fwhm,
            "n_at_spt": len(self._by_code[nearest]),
        }
        return TemplateSpectrum(spec.wave_A, spec.flux, meta)


__all__ = ["EmpiricalTemplateLibrary"]
