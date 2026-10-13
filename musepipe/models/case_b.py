"""Razones intrinsecas del caso B (Storey & Hummer 1995) y las cotas de A_V que dan.

Las dos cotas de extincion que el paper saca de lineas de recombinacion tienen
la misma forma. Una linea detectada (Halpha) y otra no detectada, con su limite
superior, acotan A_V si se supone su razon intrinseca:

- si la no detectada es MAS AZUL (Hbeta), la cota es inferior: con mucha menos
  extincion de la supuesta, Hbeta saldria por encima de su limite;
- si es MAS ROJA (Paschen), la cota es superior: con mucha mas extincion,
  Halpha se apagaria frente a la Paschen y esta saldria por encima de su limite.

La razon intrinseca sale de la tabla de emisividades de Storey & Hummer (1995),
reducida por `scripts/case_b_sh95.py` a `case_b_sh95.json` con el sha256 del
fichero de VizieR. Solo se dan valores en los nodos de la tabla: interpolar
ocultaria cuanto depende la cota de (T, n_e), y eso se declara barriendo nodos.

Hasta el 2026-10-12 la cota de Paschen del paper («15-21 mag») no salia de
ningun codigo ni producto: la razon Halpha/Paschen nunca se habia declarado
(`docs/2026-10-12_e4_extractores_de_produccion.md`).
"""

from __future__ import annotations

import json
import math
from functools import lru_cache
from pathlib import Path

TABLE_PATH = Path(__file__).with_name("case_b_sh95.json")


@lru_cache(maxsize=1)
def load_case_b_table(path: str | None = None) -> dict:
    """La tabla reducida de Storey & Hummer (1995), tal cual se versiona."""

    return json.loads(Path(path or TABLE_PATH).read_text(encoding="utf-8"))


def _node_index(values, value, what):
    for i, v in enumerate(values):
        if math.isclose(float(v), float(value), rel_tol=1e-9):
            return i
    raise ValueError(f"{what}={value!r} is not a node of the Storey & Hummer table {list(values)}")


def case_b_ratio(line, reference, *, temperature_K=1.0e4, density_cm3=1.0e2, table=None):
    """Razon intrinseca de intensidades `line / reference` en un nodo (T, n_e)."""

    tab = table or load_case_b_table()
    emis = tab["emissivity"]
    for name in (line, reference):
        if name not in emis:
            raise KeyError(f"{name!r} is not in the case-B table ({sorted(emis)})")
    it = _node_index(tab["temperatures_K"], temperature_K, "temperature_K")
    idens = _node_index(tab["densities_cm3"], density_cm3, "density_cm3")
    return float(emis[line][it][idens]) / float(emis[reference][it][idens])


def _check_positive(**values):
    for name, value in values.items():
        if not (math.isfinite(float(value)) and float(value) > 0):
            raise ValueError(f"{name} must be positive and finite, got {value!r}")


def av_upper_limit_undetected_redder(f_detected, f_limit, intrinsic_detected_over_limit,
                                     a_detected_over_av, a_limit_over_av):
    """Cota SUPERIOR de A_V con la linea detectada mas azul que la no detectada.

    Observado: F = I 10^(-0.4 A_V k). La no detectada cumple
    F_det 10^(0.4 A_V (k_det - k_lim)) / R < F_lim, con R = I_det / I_lim, asi que
    A_V < 2.5 log10(F_lim R / F_det) / (k_det - k_lim).
    """

    _check_positive(f_detected=f_detected, f_limit=f_limit,
                    intrinsic_detected_over_limit=intrinsic_detected_over_limit)
    dk = float(a_detected_over_av) - float(a_limit_over_av)
    if not dk > 0:
        raise ValueError("the detected line must be bluer (larger A_lambda/A_V) for an upper limit")
    return 2.5 * math.log10(float(f_limit) * float(intrinsic_detected_over_limit) / float(f_detected)) / dk


def av_lower_limit_undetected_bluer(f_detected, f_limit, intrinsic_detected_over_limit,
                                    a_detected_over_av, a_limit_over_av):
    """Cota INFERIOR de A_V con la linea detectada mas roja que la no detectada.

    La no detectada cumple F_det 10^(-0.4 A_V (k_lim - k_det)) / R < F_lim, asi que
    A_V > 2.5 log10(F_det / (F_lim R)) / (k_lim - k_det).
    """

    _check_positive(f_detected=f_detected, f_limit=f_limit,
                    intrinsic_detected_over_limit=intrinsic_detected_over_limit)
    dk = float(a_limit_over_av) - float(a_detected_over_av)
    if not dk > 0:
        raise ValueError("the detected line must be redder (smaller A_lambda/A_V) for a lower limit")
    return 2.5 * math.log10(float(f_detected) / (float(f_limit) * float(intrinsic_detected_over_limit))) / dk


__all__ = [
    "TABLE_PATH",
    "av_lower_limit_undetected_bluer",
    "av_upper_limit_undetected_redder",
    "case_b_ratio",
    "load_case_b_table",
]
