"""Dónde absorbe la atmósfera en el rango de MUSE, y con qué profundidad.

**Por qué un catálogo de bandas y no una lista de líneas.** A la resolución de
MUSE (R ≈ 1770 en el azul a 3590 en el rojo, FWHM ≈ 2.5 Å) las líneas
telúricas individuales de O₂ y H₂O **no se resuelven**: lo que se ve son las
bandas, mezcladas dentro de cada píxel espectral. Marcar líneas sueltas en un
espectro de MUSE daría una precisión que el dato no tiene. Por eso el catálogo
son **bandas** con sus límites, y la profundidad real se mide sobre la curva
del propio run.

**Por qué no molecfit.** `molecfit` no está disponible en este entorno, y aun
donde sí lo estuvo, para estos datos **no convergió** (el ajuste llevaba la
transmisión a 0 sin mejorar el χ²), así que A3 corrigió con la estrella
telúrica estándar (`STD_TELLURIC`). Lo que sí quedó de ahí es una **curva de
transmisión medida** para la noche, en la misma rejilla de λ del cubo: eso es
más específico que cualquier lista de laboratorio y es lo que devuelve
`measured_transmission()`. El catálogo estático de abajo es el respaldo para
los runs que no la tengan (p. ej. un objeto reducido en modo `cascade`).

Los límites de banda son los estándar del óptico; se han contrastado con la
curva medida de ROXs 12 b (ver `severity` y el docstring de cada banda).
Longitudes de onda en **aire**, como el resto de la cadena (Hα = 6562.8 Å).
"""

from __future__ import annotations

import csv
from pathlib import Path

#: Bandas de absorción telúrica en el rango de MUSE (4750–9350 Å), en aire.
#:
#: `severity` es cuánto muerden en la práctica sobre estos datos:
#:   - `strong`   : transmisión mínima medida < 0.8 — la banda inutiliza el tramo
#:   - `moderate` : 0.8–0.95 — corrige, pero deja residuo estructurado
#:   - `weak`     : > 0.95 en la curva medida de ROXs 12 b; se marca porque en
#:                  otras noches (más húmedas) sí aparece.
TELLURIC_BANDS = (
    {"name": "H₂O 5900", "species": "H2O", "lo_A": 5860.0, "hi_A": 5990.0,
     "severity": "weak"},
    {"name": "O₂ γ", "species": "O2", "lo_A": 6270.0, "hi_A": 6330.0,
     "severity": "weak"},
    {"name": "H₂O 6500", "species": "H2O", "lo_A": 6460.0, "hi_A": 6600.0,
     "severity": "weak"},
    {"name": "O₂ B", "species": "O2", "lo_A": 6860.0, "hi_A": 6960.0,
     "severity": "strong"},
    {"name": "H₂O 7200", "species": "H2O", "lo_A": 7130.0, "hi_A": 7360.0,
     "severity": "moderate"},
    {"name": "O₂ A", "species": "O2", "lo_A": 7590.0, "hi_A": 7700.0,
     "severity": "strong"},
    {"name": "H₂O 8200", "species": "H2O", "lo_A": 8100.0, "hi_A": 8400.0,
     "severity": "moderate"},
    {"name": "H₂O 8950", "species": "H2O", "lo_A": 8900.0, "hi_A": 9100.0,
     "severity": "moderate"},
    {"name": "H₂O 9300", "species": "H2O", "lo_A": 9250.0, "hi_A": 9700.0,
     "severity": "strong"},
)

#: Alpha sugerido para sombrear cada banda según lo que muerde.
SEVERITY_ALPHA = {"strong": 0.22, "moderate": 0.13, "weak": 0.06}

#: Ventana del láser de la AO en NFM: no es atmósfera, es el propio instrumento,
#: pero se marca en los mismos plots porque el hueco se ve igual.
AO_LASER_WINDOW_A = (5780.0, 6050.0)

_SKYLINES_CSV = Path(__file__).resolve().parent / "qc" / "data" / "skylines.csv"


def sky_emission_lines():
    """Líneas de **emisión** del cielo (OI, OH) del catálogo de A4.

    Estas sí se resuelven y sí dejan residuo de sustracción de cielo canal a
    canal, así que en un espectro conviene marcarlas aparte de las bandas de
    absorción.
    """
    rows = []
    with _SKYLINES_CSV.open(encoding="utf-8") as fh:
        for row in csv.DictReader(line for line in fh if not line.startswith("#")):
            rows.append({"name": row["name"], "wave_A": float(row["wave_A"])})
    return rows


def measured_transmission(run_id=None, *, project_root=None, qc=None):
    """Curva de transmisión telúrica **medida para este run**, o `None`.

    Sale del producto de A3 (`stages/stage00t_qc.json` → `products.transmission`).
    Devuelve `{"wave_A", "transmission", "source"}`; `None` si el run no pasó
    por A3 con ese producto (p. ej. el perfil `cascade`, que aplica la
    corrección dentro del combinado y no deja la curva suelta).
    """
    import numpy as np
    from astropy.io import fits

    from .paths import RunPaths

    if qc is None:
        from .io import read_json

        paths = RunPaths.from_project_root(run_id, project_root=project_root)
        for name in ("stage00t_qc.json", "stage00t_realigned_qc.json"):
            candidate = paths.stage_dir / name
            if candidate.exists():
                qc = read_json(candidate)
                break
        else:
            return None

    rel = ((qc or {}).get("products") or {}).get("transmission")
    if not rel:
        return None
    path = Path(rel)
    if not path.is_absolute():
        root = Path(project_root) if project_root else Path.cwd()
        path = root / rel
    if not path.exists():
        return None
    with fits.open(path) as hdul:
        data = hdul["TELLURIC_TRANS"].data
        wave = np.asarray(data["wave_A"], dtype=float)
        trans = np.asarray(data["transmission"], dtype=float)
    return {"wave_A": wave, "transmission": trans, "source": str(path)}


def bands_with_measured_depth(transmission=None, bands=TELLURIC_BANDS):
    """El catálogo, con la transmisión mínima **medida** en cada banda.

    Sin curva medida devuelve el catálogo tal cual (con `t_min=None`): el plot
    sigue marcando dónde están las bandas, solo que no puede decir cuánto
    absorbieron esa noche.
    """
    import numpy as np

    out = []
    wave = trans = None
    if transmission is not None:
        wave = np.asarray(transmission["wave_A"], dtype=float)
        trans = np.asarray(transmission["transmission"], dtype=float)
    for band in bands:
        row = dict(band)
        row["t_min"] = None
        if wave is not None:
            inside = (wave >= band["lo_A"]) & (wave <= band["hi_A"]) & np.isfinite(trans)
            if inside.any():
                row["t_min"] = float(np.nanmin(trans[inside]))
        out.append(row)
    return out


def telluric_mask(wave_A, *, severities=("strong", "moderate"), bands=TELLURIC_BANDS):
    """Máscara booleana de los canales que caen en bandas de esa gravedad.

    Útil para excluirlos de un ajuste de continuo sin escribir los rangos a
    mano en cada sitio.
    """
    import numpy as np

    wave = np.asarray(wave_A, dtype=float)
    mask = np.zeros(wave.shape, dtype=bool)
    for band in bands:
        if band["severity"] not in severities:
            continue
        mask |= (wave >= band["lo_A"]) & (wave <= band["hi_A"])
    return mask
