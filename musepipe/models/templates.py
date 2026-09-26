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
import warnings
from pathlib import Path

import numpy as np

from . import TemplateSpectrum
from ..constants import spt_code, spt_label
from .cache import load_spectrum_npz
from .manifest import verify_manifest

#: ``intermediate``: grupos jóvenes de 10–200 Myr (X-SHYNE), ni 1–10 Myr ni campo.
_GRAVITY_CLASSES = ("young", "intermediate", "field")


class EmpiricalTemplateLibrary:
    """Empirical template library adapter (``SpectralLibrary``)."""

    def __init__(self, family_dir, *, gravity_class, citation, version,
                 resolution_fwhm_A=None, declaration=None):
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
        #: :class:`musepipe.models.libraries.LibraryDeclaration` (marco, R, cita
        #: …). Las etapas de G3 la exigen; los llamadores sintéticos pueden
        #: omitirla.
        self.declaration = declaration
        self.name = declaration.name if declaration is not None else self.family_dir.name
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
        self._resolve_per_template_R()

    def _resolve_per_template_R(self):
        """R por plantilla de las bibliotecas X-shooter (bug 2026-09-25).

        Si la meta del espectro no trae ``R``/``resolution_R`` (las de Manara no
        lo traen; las del archivo ESO sí, de SPEC_RES), se lee la rendija VIS de
        su FITS ORIGINAL (``_source/*/<source_file>``, ``ESO INS OPTI4 NAME``) y
        se traduce con la tabla de ESO. Sin cabecera, la R declarada por la
        biblioteca, con aviso. El resultado va a la meta (``resolution_R_header``
        y ``resolution_R_source``) y a :meth:`resolution_table` para el QC.
        """
        from .libraries import is_xshooter, xshooter_vis_R_from_header
        decl = self.declaration
        for code, entries in self._by_code.items():
            for k, (path, meta) in enumerate(entries):
                meta = dict(meta)
                if meta.get("R") not in (None, "") or meta.get("resolution_R") not in (None, ""):
                    meta.setdefault("resolution_R_source",
                                    meta.get("R_source") or "spectrum meta (R)")
                elif is_xshooter(decl):
                    got = None
                    src = meta.get("source_file")
                    hits = sorted(self.family_dir.glob(f"_source/*/{src}")) if src else []
                    if hits:
                        from astropy.io import fits
                        got = xshooter_vis_R_from_header(fits.getheader(hits[0]))
                    if got is not None:
                        meta["resolution_R_header"] = got[0]
                        meta["slit_vis_arcsec"] = got[1]
                        meta["resolution_R_source"] = got[2]
                    else:
                        r, _ = decl.resolution_for(meta)
                        warnings.warn(f"{self.name}/{Path(path).name}: sin cabecera X-shooter "
                                      f"legible; se usa la R declarada ({r})", RuntimeWarning)
                        meta["resolution_R_source"] = ("declared default (original header "
                                                       "missing or unreadable)")
                entries[k] = (path, meta)

    def resolution_table(self) -> dict:
        """``{npz: {object, spt, R, source}}`` — la R con la que se degrada cada
        plantilla y de dónde sale (procedencia del QC)."""
        out = {}
        for code in self._codes:
            for path, meta in self._by_code[code]:
                r = None
                if self.declaration is not None:
                    r, fw = self.declaration.resolution_for(meta)
                    if r is None:
                        r = f"FWHM {fw} A"
                out[Path(path).name] = {
                    "object": meta.get("object", Path(path).stem), "spt": meta.get("spt"),
                    "R": (None if r is None else (r if isinstance(r, str) else
                                                  ("inf" if np.isinf(r) else float(r)))),
                    "slit_vis_arcsec": meta.get("slit_vis_arcsec"),
                    "source": meta.get("resolution_R_source",
                                       "library declaration (by object / library R)")}
        return out

    # -- protocol ---------------------------------------------------------- #
    def grid(self) -> dict:
        return {"spt": self._codes.copy()}

    def n_by_spt(self) -> dict:
        """Número de espectros por subtipo (procedencia del QC)."""
        return {spt_label(c): len(self._by_code[c]) for c in self._codes}

    def entries(self) -> list:
        """Todos los espectros: ``[(spt_code, path, source_meta), ...]`` ordenados."""
        return [(float(c), path, meta) for c in self._codes
                for path, meta in self._by_code[c]]

    def load_entry(self, code, path, src_meta) -> TemplateSpectrum:
        """Un espectro concreto (no solo el primero de su subtipo)."""
        return self._spectrum(float(code), float(code), path, src_meta)

    def get(self, *, spt, **_) -> TemplateSpectrum:
        code = spt_code(spt) if isinstance(spt, str) else float(spt)
        i = int(np.argmin(np.abs(self._codes - code)))
        nearest = float(self._codes[i])
        path, src_meta = self._by_code[nearest][0]
        return self._spectrum(nearest, code, path, src_meta)

    def _spectrum(self, nearest, code, path, src_meta) -> TemplateSpectrum:
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
            "source_npz": Path(path).name,
        }
        if self.declaration is not None:
            r, fw = self.declaration.resolution_for(src_meta)
            meta["declared_resolution_R"] = r
            meta["declared_resolution_fwhm_A"] = fw
            meta["declared_wave_frame"] = self.declaration.wave_frame
        return TemplateSpectrum(spec.wave_A, spec.flux, meta)


__all__ = ["EmpiricalTemplateLibrary"]
