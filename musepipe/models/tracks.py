"""Evolutionary tracks behind the ``EvolutionaryModel`` protocol (spec G3 §3.4,
plan WP-G3R-5). One class serves both families (BHAC15, ATMO2020) from the
cached track npz (:mod:`musepipe.models.cache`).

The published grids are IRREGULAR samples in (mass, age); ``lookup`` inverts
them by scattered linear interpolation in (log age, log L_bol) — and, when a
``teff`` is supplied, in (log age, log Teff) for the consistency cross-check.
Interpolation is barycentric on a Delaunay triangulation, so points outside the
convex hull are ``in_range=False`` with NaNs (never extrapolate, spec §3.4). The
interp-error term is half the local spread of each output over the enclosing
simplex (§4.3).
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from scipy.spatial import Delaunay

from .cache import load_tracks_npz

_PRIMARY_OUTPUTS = ("mass_msun", "radius_rsun", "logg", "teff_k")
_TEFF_OUTPUTS = ("mass_msun", "radius_rsun", "logg", "l_bol_lsun")


class TrackGrid:
    """Single-family evolutionary-track adapter (``EvolutionaryModel``)."""

    def __init__(self, npz_path, *, family, citation, version):
        if not citation:
            raise RuntimeError("TrackGrid requires a citation (spec G3 §1.2).")
        self.npz_path = Path(npz_path)
        self.family = str(family)
        self.citation = str(citation)
        self.version = version
        data = load_tracks_npz(self.npz_path)
        self.meta = data.get("meta", {})
        self._cols = {k: np.asarray(data[k], dtype=np.float64) for k in
                      ("mass_msun", "age_gyr", "teff_k", "l_bol_lsun",
                       "radius_rsun", "logg")}
        # Two triangulations: primary (log age, log L) and alt (log age, log Teff).
        self._al_tri = self._delaunay(np.column_stack([
            np.log10(self._cols["age_gyr"]), np.log10(self._cols["l_bol_lsun"])]))
        self._at_tri = self._delaunay(np.column_stack([
            np.log10(self._cols["age_gyr"]), np.log10(self._cols["teff_k"])]))

    @staticmethod
    def _delaunay(pts):
        try:
            return Delaunay(pts)
        except Exception:  # noqa: BLE001 - degenerate input -> joggle
            return Delaunay(pts, qhull_options="QJ")

    # -- protocol ---------------------------------------------------------- #
    def lookup(self, l_bol, age, *, teff=None) -> dict:
        if teff is None:
            outputs, tri, second = _PRIMARY_OUTPUTS, self._al_tri, l_bol
            second_col, second_name, mode = "l_bol_lsun", "l_bol", "age_lbol"
        else:
            outputs, tri, second = _TEFF_OUTPUTS, self._at_tri, teff
            second_col, second_name, mode = "teff_k", "teff", "age_teff"

        clamped = self._clamped(age, second, second_col, second_name)
        nan = {k: float("nan") for k in outputs}
        nan.update({f"{k}_err_interp": float("nan") for k in outputs})
        if age <= 0 or second <= 0:
            return {**nan, "in_range": False, "clamped": clamped,
                    "mode": mode, "family": self.family}

        x = np.array([math.log10(age), math.log10(second)])
        ev = self._bary(tri, x)
        if ev is None:
            return {**nan, "in_range": False, "clamped": clamped,
                    "mode": mode, "family": self.family}
        weights, verts = ev
        out = {}
        for k in outputs:
            vals = self._cols[k][verts]
            out[k] = float(weights @ vals)
            out[f"{k}_err_interp"] = 0.5 * float(vals.max() - vals.min())
        out.update({"in_range": True, "clamped": clamped, "mode": mode,
                    "family": self.family})
        return out

    def sample(self, l_bol_samples, age_samples) -> dict:
        """Vectorized (log age, log L) lookup for the MC (no Python loop)."""
        lbol = np.asarray(l_bol_samples, dtype=np.float64)
        age = np.asarray(age_samples, dtype=np.float64)
        out = {k: np.full(lbol.shape, np.nan) for k in _PRIMARY_OUTPUTS}
        valid = (age > 0) & (lbol > 0)
        x = np.full((lbol.size, 2), np.nan)
        x[valid] = np.column_stack([np.log10(age[valid]), np.log10(lbol[valid])])
        simplex = np.full(lbol.size, -1, dtype=int)
        simplex[valid] = self._al_tri.find_simplex(x[valid])
        in_range = simplex >= 0
        idx = np.nonzero(in_range)[0]
        if idx.size:
            s = simplex[idx]
            verts = self._al_tri.simplices[s]                 # (n, 3)
            trans = self._al_tri.transform[s]                 # (n, 3, 2)
            d = x[idx] - trans[:, 2, :]                        # (n, 2)
            b01 = np.einsum("nij,nj->ni", trans[:, :2, :], d)  # (n, 2)
            w = np.column_stack([b01, 1.0 - b01.sum(axis=1)])  # (n, 3)
            for k in _PRIMARY_OUTPUTS:
                out[k][idx] = np.einsum("ni,ni->n", w, self._cols[k][verts])
        out["in_range"] = in_range
        return out

    # -- internals --------------------------------------------------------- #
    def _bary(self, tri, x):
        s = int(tri.find_simplex(x))
        if s < 0:
            return None
        trans = tri.transform[s]
        b = trans[:2] @ (x - trans[2])
        return np.array([b[0], b[1], 1.0 - b[0] - b[1]]), tri.simplices[s]

    def _clamped(self, age, second, second_col, second_name):
        out = []
        a = self._cols["age_gyr"]
        if age < a.min() or age > a.max():
            out.append("age")
        col = self._cols[second_col]
        if second < col.min() or second > col.max():
            out.append(second_name)
        return out


__all__ = ["TrackGrid"]
