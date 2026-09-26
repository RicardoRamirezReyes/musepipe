"""Conversión de longitudes de onda vacío ↔ aire (plan G3 2026-09-23, §3).

MUSE entrega sus longitudes de onda en **aire**; Kesseli+2017 (SDSS) y
BT-Settl (SVO) vienen en **vacío**. A 7500 Å la diferencia es ≈ 2.07 Å, casi
una LSF de MUSE: invisible en bins de 25 Å, no a resolución nativa. Toda
plantilla se lleva al marco del dato ANTES de degradarla.

Fórmula: la conversión estándar de la IAU en la forma de Morton (2000, ApJS
130, 403, ec. 8), que usa la dispersión del aire estándar de Birch & Downs
(1994) —la misma familia que Ciddor (1996)—, en unidades de Å y con
``s = 10⁴ / λ_vac``::

    n = 1 + 8.34254e-5 + 2.406147e-2 / (130 − s²) + 1.5998e-4 / (38.9 − s²)
    λ_aire = λ_vac / n

Válida para λ ≳ 2000 Å. La inversa (aire → vacío) no tiene forma cerrada con
estos coeficientes: se resuelve por punto fijo sobre la directa, hasta 1e-12
relativo, de modo que ``air_to_vacuum(vacuum_to_air(λ)) == λ`` al redondeo.
"""

from __future__ import annotations

import numpy as np

AIR = "air"
VACUUM = "vacuum"
FRAMES = (AIR, VACUUM)

MORTON2000_CITATION = ("Morton 2000 (ApJS 130, 403), ec. 8: estándar IAU "
                       "(Birch & Downs 1994; cf. Ciddor 1996)")


def refractive_index_air(wave_vac_A):
    """Índice de refracción del aire estándar a ``λ_vac`` [Å] (Morton 2000)."""
    wave = np.asarray(wave_vac_A, dtype=np.float64)
    s2 = (1.0e4 / wave) ** 2
    return 1.0 + 8.34254e-5 + 2.406147e-2 / (130.0 - s2) + 1.5998e-4 / (38.9 - s2)


def vacuum_to_air(wave_vac_A):
    """λ_vac [Å] → λ_aire [Å] (Morton 2000, ec. 8)."""
    wave = np.asarray(wave_vac_A, dtype=np.float64)
    return wave / refractive_index_air(wave)


def air_to_vacuum(wave_air_A, *, tol=1e-12, max_iter=20):
    """λ_aire [Å] → λ_vac [Å], inversa numérica exacta de :func:`vacuum_to_air`."""
    air = np.asarray(wave_air_A, dtype=np.float64)
    vac = air * refractive_index_air(air)  # primera aproximación
    for _ in range(int(max_iter)):
        new = air * refractive_index_air(vac)
        if np.all(np.abs(new - vac) <= tol * np.abs(new)):
            return new
        vac = new
    return vac


def normalize_frame(value):
    """Lee un marco declarado ('air', 'vacuum (SDSS)', 'vac', 'AIR'…) o ``None``.

    Una cadena que no dice ni aire ni vacío (``'verify air/vacuum'`` dice las
    dos cosas) devuelve ``None``: el marco no está declarado, y no se adivina.
    """
    if value is None:
        return None
    text = str(value).strip().lower()
    has_air = "air" in text
    has_vac = "vac" in text
    if has_air == has_vac:
        return None
    return AIR if has_air else VACUUM


def convert_frame(wave_A, from_frame, to_frame):
    """Convierte ``wave_A`` de ``from_frame`` a ``to_frame`` ('air'/'vacuum')."""
    src, dst = normalize_frame(from_frame), normalize_frame(to_frame)
    if src is None or dst is None:
        raise RuntimeError(
            f"marco de longitud de onda no declarado: {from_frame!r} → {to_frame!r} "
            "(no hay marco por defecto: se declara por biblioteca)")
    wave = np.asarray(wave_A, dtype=np.float64)
    if src == dst:
        return wave.copy()
    return vacuum_to_air(wave) if src == VACUUM else air_to_vacuum(wave)


__all__ = ["AIR", "FRAMES", "MORTON2000_CITATION", "VACUUM", "air_to_vacuum",
           "convert_frame", "normalize_frame", "refractive_index_air", "vacuum_to_air"]
