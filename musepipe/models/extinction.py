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
        """A_λ/A_V. Óptico (1.1 ≤ x ≤ 3.3 µm⁻¹) con la implementación de H03;
        infrarrojo (0.3 ≤ x < 1.1, λ > 9091 Å) con la rama IR del mismo
        Cardelli et al. (1989, ec. 2a–b): a = 0.574 x^1.61, b = −0.527 x^1.61.

        Hasta 2026-09-23 la rama IR faltaba y devolvía NaN: en el rango de G3
        (hasta 9349.5 Å) los canales > 9091 Å se caían EN SILENCIO de todo
        modelo con A_V > 0 y seguían en el de A_V = 0, así que los χ² de
        distintos A_V no se calculaban sobre los mismos canales.
        """
        wave = np.atleast_1d(np.asarray(wave_A, dtype=np.float64))
        out = np.full(wave.shape, np.nan, dtype=np.float64)
        for i, w in enumerate(wave.ravel()):
            x = 1.0e4 / float(w)
            if 0.3 <= x < 1.1:
                xp = x ** 1.61
                out.ravel()[i] = 0.574 * xp - 0.527 * xp / self.rv
                continue
            try:
                out.ravel()[i] = ccm89_a_over_av(float(w), self.rv)
            except ValueError:
                out.ravel()[i] = np.nan  # outside the CCM89 validity range
        return out if np.ndim(wave_A) else float(out.ravel()[0])

    def deredden_factor(self, wave_A, av):
        """Multiplicative factor to convert observed → dereddened flux: 10^(0.4 A_lambda)."""
        a_lam = np.asarray(self.a_lambda_over_av(wave_A)) * float(av)
        return np.power(10.0, 0.4 * a_lam)


__all__ = ["CCMExtinction"]
