"""Extinction laws (ExtinctionLaw adapters). CCM89 reuses the H03 implementation
so H03's behaviour and tests are untouched (spec G3 §2.3)."""

from __future__ import annotations

import numpy as np

from ..stages.stage_h03_limits import ccm89_a_over_av


class CCMExtinction:
    """Cardelli, Clayton & Mathis (1989) optical law. Citation is mandatory (§1.2)."""

    def __init__(self, rv=3.1, *, citation=None):
        if not citation:
            raise RuntimeError("CCMExtinction requires an extinction_law_citation (spec G3 §1.2).")
        self.rv = float(rv)
        self.citation = str(citation)

    def a_lambda_over_av(self, wave_A):
        wave = np.atleast_1d(np.asarray(wave_A, dtype=np.float64))
        out = np.full(wave.shape, np.nan, dtype=np.float64)
        for i, w in enumerate(wave.ravel()):
            try:
                out.ravel()[i] = ccm89_a_over_av(float(w), self.rv)
            except ValueError:
                out.ravel()[i] = np.nan  # outside the optical validity range
        return out if np.ndim(wave_A) else float(out.ravel()[0])

    def deredden_factor(self, wave_A, av):
        """Multiplicative factor to convert observed → dereddened flux: 10^(0.4 A_lambda)."""
        a_lam = np.asarray(self.a_lambda_over_av(wave_A)) * float(av)
        return np.power(10.0, 0.4 * a_lam)


__all__ = ["CCMExtinction"]
