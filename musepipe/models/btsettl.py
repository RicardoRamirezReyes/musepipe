"""BT-Settl CIFIST atmosphere grid behind the ``SpectralLibrary`` protocol
(spec G3 §3.1, plan WP-G3R-3). Nothing above this adapter knows the library
name; it just asks for ``get(teff=, logg=)`` and ``grid()``.

The cached nodes (one npz per (Teff, logg), :mod:`musepipe.models.cache`) do NOT
share a wavelength grid, so bilinear interpolation aligns the four bracketing
corners onto the low corner's grid (linear, fast) and combines them in log-flux
(BT-Settl flux is strictly positive). Exact grid nodes are returned on their
native grid with no resampling; the flux-conserving resample onto the observed
grid happens later in :func:`musepipe.models.prep.prepare_template`.
"""

from __future__ import annotations

import json
import re
from collections import OrderedDict
from pathlib import Path

import numpy as np

from . import TemplateSpectrum
from .cache import load_spectrum_npz
from .manifest import verify_manifest

_NODE_RE = re.compile(r"teff(?P<teff>\d+)_logg(?P<logg>[\d.]+)\.npz$")


def _bracket(axis: np.ndarray, value: float):
    """Return ``(lo, hi, weight)`` bracketing ``value`` on a sorted ``axis``.

    ``weight`` is the fractional position from ``lo`` to ``hi`` (0 on an exact
    node, where ``lo == hi``). ``value`` must be within ``[axis[0], axis[-1]]``.
    """
    i = int(np.searchsorted(axis, value))
    if i < len(axis) and axis[i] == value:
        return float(axis[i]), float(axis[i]), 0.0
    lo, hi = float(axis[i - 1]), float(axis[i])
    return lo, hi, (value - lo) / (hi - lo)


def _nearest_halfstep(axis: np.ndarray, value: float) -> float:
    """Half the local grid spacing around ``value`` (for the interp-error term)."""
    i = int(np.argmin(np.abs(axis - value)))
    diffs = []
    if i > 0:
        diffs.append(axis[i] - axis[i - 1])
    if i < len(axis) - 1:
        diffs.append(axis[i + 1] - axis[i])
    return 0.5 * float(min(diffs)) if diffs else 0.0


class BTSettlLibrary:
    """BT-Settl CIFIST grid adapter (``SpectralLibrary``)."""

    def __init__(self, family_dir, *, citation, version, cache_size: int = 64):
        if not citation:
            raise RuntimeError(
                "BTSettlLibrary requires a citation (spec G3 §1.2).")
        self.family_dir = Path(family_dir)
        self.citation = str(citation)
        self.version = version
        self._cache_size = int(cache_size)
        self.manifest_info = verify_manifest(self.family_dir)  # integrity gate

        # Index {(teff, logg) -> path} from each node's own metadata (robust to
        # the file name); reading meta_json does not load wave/flux.
        self._index: dict[tuple[float, float], Path] = {}
        for path in sorted(self.family_dir.glob("*.npz")):
            with np.load(path, allow_pickle=False) as z:
                meta = json.loads(str(z["meta_json"]))
            self._index[(float(meta["teff_k"]), float(meta["logg"]))] = path
        if not self._index:
            raise RuntimeError(f"BTSettlLibrary: no node spectra under {self.family_dir}")
        self._teff_axis = np.array(sorted({t for t, _ in self._index}), float)
        self._logg_axis = np.array(sorted({g for _, g in self._index}), float)
        self._lru: "OrderedDict[tuple[float, float], TemplateSpectrum]" = OrderedDict()

    # -- protocol ---------------------------------------------------------- #
    def grid(self) -> dict:
        return {"teff": self._teff_axis.copy(), "logg": self._logg_axis.copy()}

    def get(self, *, teff, logg, **_) -> TemplateSpectrum:
        teff, logg = float(teff), float(logg)
        ta, ga = self._teff_axis, self._logg_axis
        if not (ta[0] <= teff <= ta[-1]) or not (ga[0] <= logg <= ga[-1]):
            raise RuntimeError(
                f"BT-Settl: (Teff={teff}, logg={logg}) outside grid "
                f"[{ta[0]:.0f},{ta[-1]:.0f}] K x [{ga[0]},{ga[-1]}]")
        t_lo, t_hi, w_t = _bracket(ta, teff)
        g_lo, g_hi, w_g = _bracket(ga, logg)
        halfstep = {
            "teff": (0.5 * (t_hi - t_lo)) if t_hi > t_lo else _nearest_halfstep(ta, teff),
            "logg": (0.5 * (g_hi - g_lo)) if g_hi > g_lo else _nearest_halfstep(ga, logg),
        }

        if w_t == 0.0 and w_g == 0.0:  # exact node: native grid, no resampling
            spec = self._load((t_lo, g_lo))
            return TemplateSpectrum(
                spec.wave_A.copy(), spec.flux.copy(),
                meta={**spec.meta, "interp": "exact", "teff_k": teff, "logg": logg,
                      "interp_error_halfstep": halfstep, "citation": self.citation})

        corners = [(t_lo, g_lo), (t_lo, g_hi), (t_hi, g_lo), (t_hi, g_hi)]
        ref = self._load((t_lo, g_lo))
        wave = ref.wave_A
        lf, good = {}, np.ones(wave.shape, bool)
        for key in dict.fromkeys(corners):  # unique, order-preserving
            lf[key], ok = self._aligned_logflux(key, wave)
            good &= ok
        flux = np.exp((1 - w_t) * (1 - w_g) * lf[(t_lo, g_lo)]
                      + (1 - w_t) * w_g * lf[(t_lo, g_hi)]
                      + w_t * (1 - w_g) * lf[(t_hi, g_lo)]
                      + w_t * w_g * lf[(t_hi, g_hi)])
        return TemplateSpectrum(
            wave[good], flux[good],
            meta={"teff_k": teff, "logg": logg, "interp": "bilinear",
                  "nodes": [[t_lo, g_lo], [t_lo, g_hi], [t_hi, g_lo], [t_hi, g_hi]],
                  "weights": {"teff": w_t, "logg": w_g},
                  "interp_error_halfstep": halfstep, "citation": self.citation})

    # -- internals --------------------------------------------------------- #
    def _aligned_logflux(self, key, wave):
        spec = self._load(key)
        if spec.wave_A.shape == wave.shape and np.array_equal(spec.wave_A, wave):
            f = spec.flux
        else:  # align corner onto the reference grid (linear; NaN outside range)
            f = np.interp(wave, spec.wave_A, spec.flux, left=np.nan, right=np.nan)
        ok = np.isfinite(f) & (f > 0)
        return np.log(np.where(ok, f, 1.0)), ok

    def _load(self, key) -> TemplateSpectrum:
        if key not in self._index:
            raise RuntimeError(
                f"BT-Settl node (Teff={key[0]:.0f}, logg={key[1]}) is not in the "
                "grid (missing node — cannot interpolate through it).")
        if key in self._lru:
            self._lru.move_to_end(key)
            return self._lru[key]
        spec = load_spectrum_npz(self._index[key])
        self._lru[key] = spec
        if len(self._lru) > self._cache_size:
            self._lru.popitem(last=False)
        return spec


__all__ = ["BTSettlLibrary"]
