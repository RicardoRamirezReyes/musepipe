"""Phase G3 model-adapter architecture (library-agnostic).

Nothing outside these adapters knows the name of a spectral library, grid or
track family (spec G3 §3.1). Every physical quantity carries a dependency
label; mass/radius/age/logg/Mdot may NEVER be ``direct_measurement`` (§1.1).
Concrete library implementations live behind the protocols; real fits require
external data (§1.3) and are gated by the §8.1 human checkpoint.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np

# Dependency labels (spec §1.1)
LABELS = (
    "direct_measurement",
    "empirical_inference",
    "atmospheric_model_dependent",
    "evolutionary_model_dependent",
    "upper_limit",
    "not_constrained",
)
# These quantities can never be a direct measurement (spec §1.1 / V6)
NEVER_DIRECT = ("mass", "radius", "age", "logg", "mdot", "mass_msun", "radius_rsun")


def validate_label(property_name, label):
    """Return True if ``label`` is valid for ``property_name`` (spec V6)."""
    if label not in LABELS:
        return False
    name = str(property_name).lower()
    if label == "direct_measurement" and any(k in name for k in NEVER_DIRECT):
        return False
    return True


@dataclass
class TemplateSpectrum:
    wave_A: np.ndarray
    flux: np.ndarray
    meta: dict = field(default_factory=dict)


@runtime_checkable
class SpectralLibrary(Protocol):
    def get(self, **params) -> TemplateSpectrum: ...
    def grid(self) -> dict: ...


@runtime_checkable
class ExtinctionLaw(Protocol):
    def a_lambda_over_av(self, wave_A) -> np.ndarray: ...


@runtime_checkable
class EvolutionaryModel(Protocol):
    def lookup(self, l_bol, age, *, teff=None) -> dict: ...


__all__ = [
    "LABELS", "NEVER_DIRECT", "EvolutionaryModel", "ExtinctionLaw",
    "SpectralLibrary", "TemplateSpectrum", "validate_label",
]
