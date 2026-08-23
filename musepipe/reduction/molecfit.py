"""La vía molecfit de A3: ajuste, transmisión, y una escalera en vez de un fallback.

Este módulo existe porque durante un año el proyecto creyó que **molecfit no
convergía**, y la causa que se citó (perfil GDAS ausente) era falsa. El 2026-08-06
se midió la de verdad: `WLC_CONST = -0.05`, el **defecto de la receta**, mete el
modelo desplazado en λ; el χ² inicial sale 286× mayor, el ajuste no se mueve ni
una iteración y `rel_mol_col_O2` se queda clavado en 1.0000 (`status=4` con
χ²ᵢₙᵢ = χ²_best **exacto**). Con `--WLC_CONST=0` converge, y los 8 ajustes de la
comparación de granularidad convergieron en 35-75 s cada uno.

De ahí la regla que ordena todo lo de abajo: **una no-convergencia nunca es un
motivo para descartar molecfit**. Es un defecto de parámetros o de configuración
del dato, tiene una firma reconocible en el propio producto, y se sube un peldaño
de `LADDER`. Si la escalera se agota, esto **falla** —con los peldaños y sus
firmas dentro de la excepción— en vez de degradarse en silencio a STD_TELLURIC.
Elegir STD_TELLURIC es algo que solo puede hacer el árbitro **ganando la métrica**.

El juego de parámetros no se reescribe: es el que hizo converger a A1a y a los 8
ajustes, y viene de `scripts/molecfit_granularity.py`, que a su vez lo recuperó de
las tarjetas `ESO PRO REC1 PARAM*` del producto (no del README de A1a, que está
incompleto y **no reproduce el resultado**). Lo que cambia respecto a aquel script
es que deja de vivir en un directorio de run gitignorado.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import glob
import math
import os
from pathlib import Path
import subprocess
from typing import Any, Mapping, Sequence

import numpy as np
from astropy.io import fits

from .progress import Cronometro, params_hash
from .telluric import (
    TELLURIC_BANDS,
    TelluricError,
    enforce_protected_transmission,
    measure_telluric_depths,
    wavelength_axis_from_header,
)
from .verify import extract_aperture_spectrum


#: Humedad de respaldo, la que usó A1a. Solo se aplica si no hay crudo donde
#: leerla: el cubo no la propaga y el ajuste la necesita.
DEFAULT_HUMIDITY_PCT = 15.0

#: Ventana de la mediana móvil que normaliza el flujo. Es la de `prep.py` de A1a
#: y no se toca: cambiarla cambia el continuo de entrada, o sea el ajuste.
CONTINUUM_WINDOW_PX = 201

#: MUSE es IFU y no tiene rendija; 0.2 es el valor que usó A1a. Se mantiene
#: idéntico para que cualquier ajuste nuevo sea comparable con aquel.
SLIT_WIDTH_ARCSEC = 0.2

#: Tarjetas que la receta necesita en el `science.fits` y que hay que copiar a
#: mano desde la cabecera del cubo. `ESO TEL ALT` **no está** en esta lista a
#: propósito: no viaja al espectro 1D, y por eso el ángulo se inyecta por
#: parámetro (`TELESCOPE_ANGLE_VALUE`) calculado de la masa de aire.
SCIENCE_HEADER_KEYS = (
    "MJD-OBS", "DATE-OBS", "UTC", "EXPTIME",
    "HIERARCH ESO TEL GEOLAT", "HIERARCH ESO TEL GEOLON", "HIERARCH ESO TEL GEOELEV",
    "HIERARCH ESO TEL AMBI TEMP", "HIERARCH ESO TEL AMBI PRES START",
    "HIERARCH ESO TEL TH M1 TEMP",
)


@dataclass(frozen=True)
class MolecfitWindow:
    """Una ventana de `WAVE_INCLUDE`, con el orden de continuo que se le da."""

    lo_A: float
    hi_A: float
    cont_order: int
    fit_wlc: int
    band: str
    strong: bool

    def as_dict(self) -> dict[str, Any]:
        return {"lo_A": self.lo_A, "hi_A": self.hi_A, "cont_order": self.cont_order,
                "fit_wlc": self.fit_wlc, "band": self.band, "strong": self.strong}


#: El catálogo de ventanas posibles, una por banda de `TELLURIC_BANDS`. **Cuáles
#: se ajustan de verdad lo decide `select_fit_windows` con la profundidad medida
#: en ESTE cubo**, y no es un refinamiento: es el resultado de medirlo.
#:
#: Sobre el combinado del OB 3444577 (2026-08-22) las bandas de agua salen a
#: **0.0 % de profundidad** —el DRS ya corrigió, `muse_scipost` consume
#: `STD_TELLURIC`— y meterlas en el ajuste hace daño de dos formas distintas:
#: con el agua congelada a columna 1.0 la transmisión modelada las hunde a
#: −6.6 % y −7.4 %, y dejándola libre el ajuste pide **5.88 veces** la columna de
#: agua y las hunde a −16 % y −17 %. Lo segundo dice qué pasa: la primaria es una
#: M0 y esas dos ventanas caen sobre bandas moleculares **fotosféricas** (TiO,
#: VO), así que molecfit las lee como vapor de agua. Es exactamente el riesgo que
#: la spec §4.1 llama «punto de máxima atención de la etapa».
#:
#: `strong` marca las dos que tienen leverage telúrico limpio (O₂ A al 73 % de
#: profundidad, O₂ B junto a Hα) — las mismas dos con las que convergió A1a.
FIT_WINDOWS: tuple[MolecfitWindow, ...] = (
    MolecfitWindow(*TELLURIC_BANDS["O2_B"], cont_order=1, fit_wlc=1, band="O2_B", strong=True),
    MolecfitWindow(*TELLURIC_BANDS["O2_A"], cont_order=2, fit_wlc=1, band="O2_A", strong=True),
    MolecfitWindow(*TELLURIC_BANDS["H2O_7200"], cont_order=2, fit_wlc=0, band="H2O_7200", strong=False),
    MolecfitWindow(*TELLURIC_BANDS["H2O_8200"], cont_order=2, fit_wlc=0, band="H2O_8200", strong=False),
)

#: El juego que converge. Cada uno con su motivo, porque tres de ellos se
#: recuperaron midiendo y no están en ningún README.
MOLECFIT_BASE_PARAMS: dict[str, str] = {
    "LIST_MOLEC": "O2,H2O",
    "FIT_MOLEC": "1,0",          # el agua congelada: noche seca, y sin leverage propio
    "REL_COL": "1.0,1.0",
    "COLUMN_LAMBDA": "WAVE",
    "COLUMN_FLUX": "FLUX",
    "WLG_TO_MICRON": "0.0001",
    "WAVELENGTH_FRAME": "AIR",   # el eje de MUSE es aire; la salida sale en vacío (ver abajo)
    "FIT_CONTINUUM": "1",
    "CONTINUUM_N": "1",
    "FIT_WLC": "1",              # apagado, el modelo queda corrido -1.00 canales en O2 B
    "WLC_CONST": "0",            # el defecto (-0.05) CONGELA el ajuste. No se toca.
}


def select_fit_windows(depth_pct_by_band: Mapping[str, float], *,
                       threshold_pct: float = 3.0,
                       windows: Sequence[MolecfitWindow] = ()) -> list[MolecfitWindow]:
    """Ajusta solo las bandas que **de verdad** tienen absorción que quitar.

    La regla es la de la propia etapa: una banda por debajo del umbral de A3 no
    necesita corrección (`not_needed_shallow`), y meterla en el ajuste no es
    neutro — le da a molecfit señal que no es telúrica (fotosfera) con la que
    ajustar una columna que sí lo es. Como fuera de las ventanas T ≡ 1, no
    ajustar una banda plana es exactamente dejarla en paz.

    Devuelve la lista vacía si ninguna banda pasa el umbral: eso no es un fallo,
    es el veredicto `not_needed_shallow` visto desde aquí.
    """

    catalogo = list(windows) if windows else list(FIT_WINDOWS)
    return [w for w in catalogo
            if float(depth_pct_by_band.get(w.band, 0.0)) >= float(threshold_pct)]


@dataclass(frozen=True)
class MolecfitRung:
    """Un peldaño: qué firma atiende, qué cambia, y por qué debería arreglarlo."""

    name: str
    diagnosis: str
    attends: tuple[str, ...] = ()
    params: Mapping[str, str] = field(default_factory=dict)
    windows: str = "all"          # "all" | "strong"

    def as_dict(self) -> dict[str, Any]:
        return {"peldano": self.name, "diagnostico": self.diagnosis,
                "atiende": list(self.attends), "params": dict(self.params),
                "ventanas": self.windows}


#: La escalera. El peldaño 0 **ya lleva incorporados** los cuatro remedios que se
#: descubrieron entre 2026-07-19 y 2026-08-06 (flujo normalizado, ventana de O₂ A,
#: `WLC_CONST=0`, `FIT_WLC=1`): son el punto de partida, no un rescate. Los de
#: abajo son los que quedan cuando aun así no converge.
LADDER: tuple[MolecfitRung, ...] = (
    MolecfitRung(
        "base",
        "la receta medida que converge: flujo normalizado, O2 A dentro, WLC_CONST=0, FIT_WLC=1",
    ),
    MolecfitRung(
        "wlc_grado_1",
        "el corrimiento en lambda no es constante sino de escala; se le da grado 1",
        attends=("chi2_congelado", "residuo_alto"),
        params={"WLC_N": "1"},
    ),
    MolecfitRung(
        "continuo_grado_2",
        "el chi2 lo domina el desajuste de continuo y la molecula se queda sin leverage",
        attends=("continuo_atascado", "sin_leverage", "residuo_alto"),
        params={"CONTINUUM_N": "2"},
    ),
    MolecfitRung(
        "solo_bandas_fuertes",
        "una ventana somera o con lineas estelares arrastra el ajuste; se deja O2 A y O2 B",
        attends=("sin_leverage", "residuo_alto", "chi2_congelado"),
        windows="strong",
    ),
    MolecfitRung(
        "agua_libre",
        "el residuo lo pone el agua, que estaba congelada por la hipotesis de noche seca",
        attends=("residuo_alto",),
        params={"FIT_MOLEC": "1,1"},
    ),
    MolecfitRung(
        "nucleo_gaussiano",
        "el nucleo por defecto no describe la LSF de este cubo (medido degenerado en 2026-08-06, "
        "asi que es el ultimo peldano: es el que menos se espera que mueva nada)",
        attends=("residuo_alto",),
        params={"FIT_RES_BOX": "FALSE", "FIT_RES_GAUSS": "TRUE"},
    ),
)


class MolecfitNotConverged(TelluricError):
    """La escalera se agotó. Lleva dentro cada peldaño y su firma.

    Es `TelluricError` para que la etapa pare, y **nunca** se captura para elegir
    STD_TELLURIC: eso convertiría un defecto de configuración en una decisión
    científica, que es justo el error que este módulo existe para no repetir.
    """

    def __init__(self, message: str, *, rungs: Sequence[Mapping[str, Any]] = ()):
        super().__init__(message)
        self.rungs = [dict(r) for r in rungs]


@dataclass
class MolecfitResult:
    """Lo que sale de un ajuste, converja o no."""

    label: str
    cube: str
    out_dir: str
    converged: bool
    signature: str
    rung: str
    rungs_tried: list[dict[str, Any]]
    params: dict[str, str]
    windows: list[dict[str, Any]]
    best_fit: dict[str, tuple[float, float]]
    airmass: float
    telescope_angle_deg: float
    humidity_pct: float
    humidity_source: str
    seconds: float
    depth_pct_by_band: dict = field(default_factory=dict)
    #: El espectro sobre el que se ajusto. NO viaja en `as_dict` —son 3681
    #: numeros— pero lo necesita quien puntue esta unidad por si misma: la
    #: dispersion entre exposiciones solo significa algo si cada una se mide
    #: sobre SU espectro, no sobre el del combinado.
    wave_A: Any = None
    spectrum: Any = None
    yx: tuple[float, float] | None = None
    warnings: list[str] = field(default_factory=list)

    def value(self, name: str, default: float = float("nan")) -> float:
        pair = self.best_fit.get(name)
        return float(pair[0]) if pair else default

    def as_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "cube": self.cube,
            "out_dir": self.out_dir,
            "converged": self.converged,
            "signature": self.signature,
            "rung": self.rung,
            "rungs_tried": self.rungs_tried,
            "params": self.params,
            "windows": self.windows,
            "airmass": None if not np.isfinite(self.airmass) else round(self.airmass, 4),
            "telescope_angle_deg": round(self.telescope_angle_deg, 4),
            "humidity_pct": self.humidity_pct,
            "humidity_source": self.humidity_source,
            "seconds": round(self.seconds, 1),
            "depth_pct_by_band": self.depth_pct_by_band,
            "yx": [round(v, 2) for v in self.yx] if self.yx else None,
            "fit": {k: round(v[0], 6) for k, v in self.best_fit.items()
                    if k in ("status", "iterations", "initial_chi2", "best_chi2", "reduced_chi2",
                             "rms_rel_to_err", "rel_mol_col_O2", "ppmv_O2", "h2o_col_mm")},
            "warnings": self.warnings,
        }


# --------------------------------------------------------------------------
# Preparación de la entrada
# --------------------------------------------------------------------------

def running_median(values: Sequence[float], window: int = CONTINUUM_WINDOW_PX) -> np.ndarray:
    """Mediana móvil, la de `prep.py` de A1a, con los bordes repetidos."""

    y = np.asarray(values, dtype=np.float64)
    pad = int(window) // 2
    padded = np.pad(y, pad, mode="edge")
    from numpy.lib.stride_tricks import sliding_window_view

    return np.median(sliding_window_view(padded, int(window)), axis=-1)


def normalize_by_continuum(flux: Sequence[float], *, window: int = CONTINUUM_WINDOW_PX) -> np.ndarray:
    """Flujo a O(1). Sin esto el continuo se atasca en 1.0 y el χ² lo domina él.

    La transmisión que sale del ajuste es **independiente** de esta normalización
    —es un cociente—, así que normalizar no toca el resultado físico: toca la
    capacidad del ajuste de llegar a él.
    """

    y = np.asarray(flux, dtype=np.float64)
    cont = running_median(y, window)
    cont = np.clip(cont, np.nanmedian(cont) * 1e-3, None)
    norm = y / cont
    return np.where(np.isfinite(norm), norm, 1.0)


def write_wave_include(windows: Sequence[MolecfitWindow], path: str | Path) -> Path:
    """El `WAVE_INCLUDE` que exige la receta, en micras y mapeado al chip 1."""

    if not windows:
        raise TelluricError("molecfit needs at least one fit window.")
    lo = np.array([w.lo_A * 1e-4 for w in windows], dtype=np.float64)
    hi = np.array([w.hi_A * 1e-4 for w in windows], dtype=np.float64)
    cols = [
        fits.Column(name="LOWER_LIMIT", format="D", array=lo),
        fits.Column(name="UPPER_LIMIT", format="D", array=hi),
        fits.Column(name="MAPPED_TO_CHIP", format="J", array=np.ones(len(windows), dtype=np.int32)),
        fits.Column(name="CONT_FIT_FLAG", format="J",
                    array=np.ones(len(windows), dtype=np.int32)),
        fits.Column(name="CONT_POLY_ORDER", format="J",
                    array=np.array([w.cont_order for w in windows], dtype=np.int32)),
        fits.Column(name="WLC_FIT_FLAG", format="J",
                    array=np.array([w.fit_wlc for w in windows], dtype=np.int32)),
    ]
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fits.HDUList([fits.PrimaryHDU(), fits.BinTableHDU.from_columns(cols)]).writeto(out, overwrite=True)
    return out


#: Tarjetas que molecfit promedia sobre la exposición: son estado de la atmósfera
#: y del telescopio, así que para un cubo combinado el valor honesto es la media
#: ponderada por EXPTIME, igual que la masa de aire.
AMBIENT_KEYS = (
    "HIERARCH ESO TEL AMBI TEMP",
    "HIERARCH ESO TEL AMBI PRES START",
    "HIERARCH ESO TEL TH M1 TEMP",
)

#: Constantes del sitio. No se promedian: se copian, y si dos exposiciones no
#: coinciden es que no son del mismo telescopio y eso hay que verlo.
SITE_KEYS = (
    "HIERARCH ESO TEL GEOLAT",
    "HIERARCH ESO TEL GEOLON",
    "HIERARCH ESO TEL GEOELEV",
)


def merge_science_header(base: Mapping[str, Any],
                         exposure_headers: Sequence[Mapping[str, Any]]) -> tuple[dict, list[str]]:
    """Completa la cabecera del combinado con lo que solo traen las exposiciones.

    **Este es el fallo que mata a molecfit sobre un cubo combinado**, y es de
    configuración del dato, no del método: el combine por voxel escribe una
    cabecera propia (`RUNID`, `NEXP`, `EXPTOT`…) y deja fuera las ocho tarjetas
    que la receta exige — `MJD-OBS`, `UTC`, presión y temperatura ambiente,
    temperatura de M1 y las tres del emplazamiento. Sin ellas `molecfit_model`
    aborta con `status = 14` antes de ajustar nada, y leído desde fuera parece
    otra vez «molecfit no converge».

    Los valores salen de las exposiciones que hay dentro del cubo, con el mismo
    criterio que la masa de aire efectiva: **media ponderada por EXPTIME** para lo
    que es estado de la atmósfera, **suma** para el tiempo de integración, y copia
    para las constantes del sitio.

    `UTC` y `DATE-OBS` se derivan del MJD ya promediado en vez de promediarse
    aparte: promediar segundos-desde-medianoche se rompe al cruzar las 00:00, y
    estas observaciones cruzan la medianoche (2022-08-31 → 2022-09-01). El precio
    es ~11 s de discrepancia con la tarjeta original, que para un modelo de
    atmósfera no significa nada.
    """

    fusionada = {clave: base.get(clave) for clave in SCIENCE_HEADER_KEYS
                 if base.get(clave) is not None}
    avisos: list[str] = []
    if not exposure_headers:
        return fusionada, avisos

    pesos = np.array([float(h.get("EXPTIME") or 1.0) for h in exposure_headers], dtype=np.float64)
    if not np.isfinite(pesos).all() or pesos.sum() <= 0:
        pesos = np.ones(len(exposure_headers), dtype=np.float64)

    def _media(clave):
        valores, w = [], []
        for peso, h in zip(pesos, exposure_headers):
            valor = h.get(clave)
            if valor is None:
                continue
            valores.append(float(valor))
            w.append(peso)
        if not valores:
            return None
        return float(np.average(valores, weights=w))

    for clave in AMBIENT_KEYS:
        if fusionada.get(clave) is None:
            valor = _media(clave)
            if valor is not None:
                fusionada[clave] = valor

    for clave in SITE_KEYS:
        if fusionada.get(clave) is not None:
            continue
        valores = {float(h[clave]) for h in exposure_headers if h.get(clave) is not None}
        if not valores:
            continue
        if len(valores) > 1:
            avisos.append(f"{clave} difiere entre exposiciones ({sorted(valores)}): "
                          "no son del mismo emplazamiento")
        fusionada[clave] = sorted(valores)[0]

    if fusionada.get("MJD-OBS") is None:
        mjd = _media("MJD-OBS")
        if mjd is not None:
            fusionada["MJD-OBS"] = mjd
            fusionada["UTC"] = float((mjd % 1.0) * 86400.0)
            from datetime import datetime, timedelta
            fecha = datetime(1858, 11, 17) + timedelta(days=float(mjd))
            fusionada["DATE-OBS"] = fecha.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]
            avisos.append("MJD-OBS/UTC/DATE-OBS reconstruidos como media ponderada por EXPTIME "
                          "de las exposiciones: el combinado no los declara")
    if fusionada.get("EXPTIME") is None:
        total = float(sum(float(h.get("EXPTIME") or 0.0) for h in exposure_headers))
        if total > 0:
            fusionada["EXPTIME"] = total

    faltan = [c for c in SCIENCE_HEADER_KEYS if fusionada.get(c) is None]
    if faltan:
        avisos.append("siguen faltando tarjetas que la receta exige: " + ", ".join(faltan))
    return fusionada, avisos


def write_molecfit_science(wave_A: Sequence[float], flux: Sequence[float],
                           header: Mapping[str, Any], path: str | Path) -> Path:
    """El `science.fits` de una sola BINTABLE que come `molecfit_model`."""

    pri = fits.Header()
    for key in SCIENCE_HEADER_KEYS:
        value = header.get(key) if hasattr(header, "get") else None
        if value is not None:
            pri[key] = value
    pri["HIERARCH ESO INS SLIT1 WID"] = SLIT_WIDTH_ARCSEC
    cols = [
        fits.Column(name="WAVE", format="D", array=np.asarray(wave_A, dtype=np.float64)),
        fits.Column(name="FLUX", format="D", array=np.asarray(flux, dtype=np.float64)),
    ]
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fits.HDUList([fits.PrimaryHDU(header=pri),
                  fits.BinTableHDU.from_columns(cols)]).writeto(out, overwrite=True)
    return out


def primary_spectrum_from_cube(cube: str | Path, *, radius_px: float = 6.0,
                               yx: Sequence[float] | None = None):
    """(wave_A, flujo, (y, x), cabecera) de la primaria, localizada en ESTE cubo.

    La posición **no se hereda** de otro cubo: los cubos por exposición no
    comparten marco con el combinado, porque los offsets de alineado son
    posteriores a ellos. Pasar `yx` es para reproducir una medida, no para
    ahorrarse la búsqueda.
    """

    with fits.open(cube, memmap=True) as hdul:
        hdu = hdul["DATA"] if "DATA" in hdul else hdul[1]
        wave = wavelength_axis_from_header(hdu.header, hdu.shape[0])
        if not np.all(np.isfinite(wave)) or wave[0] == 0:
            wave = wavelength_axis_from_header(hdul[0].header, hdu.shape[0])
        if yx is None:
            blanco = np.nanmedian(np.asarray(hdu.data[::37], dtype=np.float64), axis=0)
            pico = np.unravel_index(np.nanargmax(blanco), blanco.shape)
            y0, x0 = int(pico[0]), int(pico[1])
            sy = slice(max(0, y0 - 3), y0 + 4)
            sx = slice(max(0, x0 - 3), x0 + 4)
            caja = np.nan_to_num(blanco[sy, sx])
            yy, xx = np.mgrid[sy.start:sy.start + caja.shape[0], sx.start:sx.start + caja.shape[1]]
            cy = float((yy * caja).sum() / caja.sum())
            cx = float((xx * caja).sum() / caja.sum())
        else:
            cy, cx = float(yx[0]), float(yx[1])
        half = 30
        ny, nx = hdu.shape[1:]
        a, b = max(0, int(cy) - half), min(ny, int(cy) + half + 1)
        c, d = max(0, int(cx) - half), min(nx, int(cx) + half + 1)
        ventana = np.asarray(hdu.data[:, a:b, c:d], dtype=np.float64)
        cabecera = hdul[0].header.copy()
    spec = extract_aperture_spectrum(ventana, (cy - a, cx - c), radius_px)
    return wave, spec, (cy, cx), cabecera


# --------------------------------------------------------------------------
# Condiciones de observación
# --------------------------------------------------------------------------

def exposure_conditions(header: Mapping[str, Any]) -> dict[str, Any]:
    """Masa de aire de inicio/fin, exptime y fecha, tal como los declara el cubo."""

    def _get(key):
        value = header.get(key) if hasattr(header, "get") else None
        return None if value is None else float(value)

    return {
        "airm_start": _get("HIERARCH ESO TEL AIRM START"),
        "airm_end": _get("HIERARCH ESO TEL AIRM END"),
        "exptime": _get("EXPTIME"),
        "date_obs": (header.get("DATE-OBS") if hasattr(header, "get") else None),
    }


def effective_airmass(conditions: Sequence[Mapping[str, Any]]) -> tuple[float, str]:
    """X efectiva, ponderada por EXPTIME sobre las exposiciones que hay.

    Esto es lo que arregla el sesgo estructural medido del modo combinado: la
    cabecera de un cubo combinado **hereda las condiciones de su primera
    exposición** y las presenta como si fueran las del cubo (X = 1.14 para luz que
    promedia ~1.28), y el canónico multi-noche directamente no lleva condiciones,
    con lo que caería al defecto de cenit. Con la X heredada el ajuste pide
    `rel_mol_col_O2 = 0.953` frente a 0.84-0.86 por exposición: ~11 % más columna
    de O₂ para compensar una masa de aire que no era.

    Se pondera por EXPTIME y no por número de exposiciones porque lo que promedia
    la atmósfera es el **fotón**, no la exposición.
    """

    pesos, valores = [], []
    for cond in conditions:
        x1, x2 = cond.get("airm_start"), cond.get("airm_end")
        if x1 is None or x2 is None:
            continue
        peso = float(cond.get("exptime") or 1.0)
        valores.append(0.5 * (float(x1) + float(x2)))
        pesos.append(peso)
    if not valores:
        return float("nan"), "sin AIRM en ninguna exposicion"
    x = float(np.average(valores, weights=pesos))
    if len(valores) == 1:
        return x, "mitad de exposicion"
    return x, f"media ponderada por EXPTIME de {len(valores)} exposiciones"


def telescope_angle_deg(airmass: float) -> float:
    """Altitud que se le pasa a la receta. Sin X válida, cenit — y se declara."""

    if not np.isfinite(airmass) or airmass < 1.0:
        return 90.0
    return float(math.degrees(math.asin(1.0 / float(airmass))))


def humidity_from_raw(date_obs: Any, raw_dir: str | Path | None) -> tuple[float, str]:
    """Humedad real de la exposición. El cubo no la propaga; el crudo sí."""

    if not raw_dir:
        return DEFAULT_HUMIDITY_PCT, "sin raw_dir"
    for path in sorted(glob.glob(os.path.join(str(raw_dir), "MUSE.*.fits"))):
        try:
            head = fits.getheader(path)
        except OSError:
            continue
        if str(head.get("DATE-OBS", ""))[:19] == str(date_obs)[:19]:
            return (float(head.get("HIERARCH ESO TEL AMBI RHUM", DEFAULT_HUMIDITY_PCT)),
                    os.path.basename(path))
    return DEFAULT_HUMIDITY_PCT, "no encontrado en el crudo"


# --------------------------------------------------------------------------
# Lectura del producto y firmas del fallo
# --------------------------------------------------------------------------

def read_best_fit(path: str | Path) -> dict[str, tuple[float, float]]:
    """`BEST_FIT_PARAMETERS` como {parametro: (valor, incertidumbre)}.

    Las filas de relleno (`\\x00` con valor NaN) se descartan: existen en todos
    los productos de esta build y colarlas convierte cualquier lectura en NaN.
    """

    tabla = fits.getdata(str(path), 1)
    nombres = list(tabla.columns.names)
    col_par, col_val = nombres[0], nombres[1]
    col_unc = nombres[2] if len(nombres) > 2 else None
    salida: dict[str, tuple[float, float]] = {}
    for fila in tabla:
        clave = str(fila[col_par]).strip().strip("\x00")
        if not clave:
            continue
        try:
            valor = float(fila[col_val])
        except (TypeError, ValueError):
            continue
        if not np.isfinite(valor):
            continue
        unc = float(fila[col_unc]) if col_unc is not None else float("nan")
        salida[clave] = (valor, unc)
    return salida


def classify_fit(best_fit: Mapping[str, tuple[float, float]], *, log_text: str = "",
                 returncode: int = 0) -> tuple[bool, str]:
    """(convergió, firma). La firma nombra el defecto, no el síntoma.

    Las cuatro primeras son las que este proyecto ya ha visto y documentado; el
    orden importa porque `chi2_congelado` explica a las demás cuando aparece.
    """

    if "gdas" in log_text.lower() and "failed" in log_text.lower():
        # No es fatal y no sesga la comparación: afecta igual a las dos vías.
        # Se anota como aviso, no como firma de fallo (lo hace `fit_molecfit`).
        pass
    if not best_fit:
        if "unable to find keyword" in log_text.lower():
            # La receta aborta con status=14 ANTES de ajustar: le faltan tarjetas
            # en el `science.fits`. Es configuracion del dato — el combine no las
            # escribe — y se arregla completando la cabecera, no cambiando de via.
            return False, "cabecera_incompleta"
        return False, "sin_producto" if returncode == 0 else "receta_fallo"

    status = best_fit.get("status", (float("nan"), 0.0))[0]
    chi2_ini = best_fit.get("initial_chi2", (float("nan"), 0.0))[0]
    chi2_best = best_fit.get("best_chi2", (float("nan"), 0.0))[0]
    rms_rel = best_fit.get("rms_rel_to_err", (float("nan"), 0.0))[0]
    rel_col, rel_col_unc = best_fit.get("rel_mol_col_O2", (float("nan"), float("nan")))

    if np.isfinite(chi2_ini) and np.isfinite(chi2_best) and chi2_best >= chi2_ini:
        # La firma de 2026-07-06: el ajuste no se movió ni una iteración.
        return False, "chi2_congelado"
    if np.isfinite(rel_col) and abs(rel_col - 1.0) < 1e-9 and not (rel_col_unc > 0):
        # `rel_mol_col` clavado en 1.0000 con incertidumbre 0: la molecula no
        # tenia leverage en las ventanas que se le dieron.
        return False, "sin_leverage"
    if np.isfinite(rms_rel) and rms_rel > 50.0:
        # A1a fallido medía 85.8. Es el continuo dominando el chi2.
        return False, "continuo_atascado"
    # Los codigos de MPFIT: <=0 es error, 1-4 es convergencia limpia (chi2,
    # parametros, ambos, ortogonalidad) y **5-8 tambien terminan bien** — 5 es
    # «maximo de iteraciones» y 6-8 son «ya no se puede mejorar mas». Tratar 5
    # como fallo hacia subir peldanos a un ajuste que habia bajado el chi2 45x
    # (medido en ROXs 42B b: 130876 -> 2876 con status=5 y 151 iteraciones).
    if np.isfinite(status) and int(status) <= 0:
        return False, "status_error"
    if np.isfinite(status) and int(status) > 8:
        return False, "status_desconocido"
    if not (np.isfinite(rel_col) and rel_col_unc > 0):
        return False, "sin_leverage"
    if np.isfinite(rms_rel) and rms_rel > 20.0:
        return False, "residuo_alto"
    if np.isfinite(status) and int(status) >= 5:
        # Termina bien pero no por convergencia estricta. Se declara para que no
        # pase por un ajuste limpio sin serlo.
        return True, "converge_por_iteraciones"
    return True, "converge"


# --------------------------------------------------------------------------
# El ajuste
# --------------------------------------------------------------------------

def _esorex_command(sof: Path, out_dir: Path, params: Mapping[str, str], *,
                    telescope_angle: float, humidity: float, esorex: str = "esorex") -> list[str]:
    orden = [esorex, f"--output-dir={out_dir}", "molecfit_model"]
    orden += [f"--{clave}={valor}" for clave, valor in sorted(params.items())]
    orden += [
        "--TELESCOPE_ANGLE_KEYWORD=NONE", f"--TELESCOPE_ANGLE_VALUE={telescope_angle:.4f}",
        "--RELATIVE_HUMIDITY_KEYWORD=NONE", f"--RELATIVE_HUMIDITY_VALUE={humidity}",
        str(sof),
    ]
    return orden


def fit_molecfit(cube: str | Path, *, out_root: str | Path, label: str,
                 radius_px: float = 6.0, yx: Sequence[float] | None = None,
                 raw_dir: str | Path | None = None,
                 conditions: Sequence[Mapping[str, Any]] | None = None,
                 exposure_headers: Sequence[Mapping[str, Any]] | None = None,
                 windows: Sequence[MolecfitWindow] = FIT_WINDOWS,
                 threshold_pct: float | None = None,
                 extra_params: Mapping[str, str] | None = None,
                 ladder: Sequence[MolecfitRung] = LADDER,
                 esorex: str = "esorex", reuse: bool = True,
                 runner=None) -> MolecfitResult:
    """Ajusta molecfit sobre la primaria de `cube`, subiendo peldaños si hace falta.

    `conditions` son las condiciones de las exposiciones que hay **dentro** de este
    cubo: una para un cubo por exposición, N para un combinado. De ahí sale la X
    efectiva, que es lo que evita el sesgo del modo combinado.

    Nunca devuelve un resultado no convergido: o converge, o levanta
    `MolecfitNotConverged` con la escalera entera dentro.
    """

    destino = Path(out_root) / label
    destino.mkdir(parents=True, exist_ok=True)
    ejecutar = runner if runner is not None else _run_esorex

    wave, flux, centro, cabecera = primary_spectrum_from_cube(cube, radius_px=radius_px, yx=yx)
    if conditions is None:
        conditions = [exposure_conditions(cabecera)]
    airmass, fuente_x = effective_airmass(conditions)
    angulo = telescope_angle_deg(airmass)
    humedad, fuente_h = humidity_from_raw(cabecera.get("DATE-OBS"), raw_dir)

    avisos: list[str] = []
    if not np.isfinite(airmass):
        avisos.append(f"sin masa de aire ({fuente_x}): el ajuste va a cenit, X=1")
    else:
        avisos.append(f"X = {airmass:.4f} ({fuente_x})")
    if fuente_h.startswith("sin ") or fuente_h.startswith("no encontrado"):
        avisos.append(f"humedad por defecto {DEFAULT_HUMIDITY_PCT} % ({fuente_h})")

    # La cabecera que se le da a la receta se COMPLETA con las exposiciones: un
    # cubo combinado no declara ni MJD-OBS ni las condiciones ambientales, y sin
    # ellas `molecfit_model` aborta antes de ajustar (status=14).
    cabecera_fit, avisos_cabecera = merge_science_header(cabecera, exposure_headers or [])
    avisos.extend(avisos_cabecera)

    # Cada unidad decide sobre SU propio espectro. Medido el 2026-08-22: los cubos
    # por exposicion salen de `muse_scipost` consumiendo STD_TELLURIC, o sea que ya
    # vienen corregidos, y ajustar una banda plana no es neutro — el ajuste se
    # queda sin senal telurica, `rel_mol_col_O2` cae al borde 0.0000 y la escalera
    # se pone a subir peldanos buscando un residuo que no existe. Ahi lo correcto
    # es NO ajustar: T = 1 y el veredicto de la etapa, `not_needed_shallow`.
    profundidades = measure_telluric_depths(wave, flux)
    if threshold_pct is not None:
        seleccion = select_fit_windows(profundidades, threshold_pct=threshold_pct,
                                       windows=windows)
        if not seleccion:
            return MolecfitResult(
                label=label, cube=str(cube), out_dir="", converged=True,
                signature="not_needed_shallow", rung="sin_ajuste", rungs_tried=[],
                params={}, windows=[], best_fit={}, airmass=airmass,
                telescope_angle_deg=angulo, humidity_pct=humedad, humidity_source=fuente_h,
                seconds=0.0, yx=centro, warnings=avisos + [
                    f"ninguna banda llega al {threshold_pct} %: no se ajusta nada y T = 1"],
                depth_pct_by_band=profundidades, wave_A=wave, spectrum=flux)
        windows = seleccion

    science = destino / "science.fits"
    normalizado = destino / "science_norm.fits"
    write_molecfit_science(wave, flux, cabecera_fit, science)
    write_molecfit_science(wave, normalize_by_continuum(flux), cabecera_fit, normalizado)

    intentados: list[dict[str, Any]] = []
    segundos_totales = 0.0
    for peldano in ladder:
        ventanas = [w for w in windows if w.strong] if peldano.windows == "strong" else list(windows)
        params = dict(MOLECFIT_BASE_PARAMS)
        params.update(extra_params or {})
        params.update(peldano.params)
        salida = destino / ("out" if peldano.name == "base" else f"out_{peldano.name}")
        salida.mkdir(parents=True, exist_ok=True)
        best_path = salida / "BEST_FIT_PARAMETERS.fits"

        wave_include = destino / f"wave_include_{peldano.name}.fits"
        write_wave_include(ventanas, wave_include)
        sof = destino / f"model_{peldano.name}.sof"
        sof.write_text(f"{normalizado} SCIENCE\n{wave_include} WAVE_INCLUDE\n", encoding="utf-8")
        orden = _esorex_command(sof, salida, params, telescope_angle=angulo,
                                humidity=humedad, esorex=esorex)
        (destino / f"esorex_{peldano.name}.cmd").write_text(" ".join(orden) + "\n",
                                                            encoding="utf-8")

        log_path = destino / f"run_{peldano.name}.log"
        reloj = Cronometro()
        with reloj:
            if reuse and best_path.exists():
                rc = 0
            else:
                rc = ejecutar(orden, cwd=destino, log_path=log_path)
        segundos_totales += reloj.segundos
        log_text = log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else ""
        best_fit = read_best_fit(best_path) if best_path.exists() else {}
        convergio, firma = classify_fit(best_fit, log_text=log_text, returncode=rc)

        if "gdas" in log_text.lower() and "failed" in log_text.lower():
            aviso = "sin perfil GDAS local: atmosfera estandar (afecta igual a las dos vias)"
            if aviso not in avisos:
                avisos.append(aviso)

        intentados.append({
            **peldano.as_dict(),
            "rc": rc,
            "firma": firma,
            "convergio": convergio,
            "segundos": round(reloj.segundos, 1),
            "out_dir": str(salida),
            "chi2": [best_fit.get("initial_chi2", (None,))[0], best_fit.get("best_chi2", (None,))[0]],
            "rel_mol_col_O2": best_fit.get("rel_mol_col_O2", (None,))[0],
        })
        if convergio:
            return MolecfitResult(
                label=label, cube=str(cube), out_dir=str(salida), converged=True,
                signature=firma, rung=peldano.name, rungs_tried=intentados,
                params=params, windows=[w.as_dict() for w in ventanas],
                best_fit=best_fit, airmass=airmass, telescope_angle_deg=angulo,
                humidity_pct=humedad, humidity_source=fuente_h,
                seconds=segundos_totales, yx=centro, warnings=avisos,
                depth_pct_by_band=profundidades, wave_A=wave, spectrum=flux,
            )

    detalle = "; ".join(f"{p['peldano']}→{p['firma']}" for p in intentados)
    raise MolecfitNotConverged(
        f"molecfit no convergio en {label} tras {len(intentados)} peldanos ({detalle}). "
        "Esto es un defecto de parametros o de configuracion del dato, NO un motivo para "
        "usar STD_TELLURIC: revisa las firmas y anade un peldano.",
        rungs=intentados,
    )


def _run_esorex(command: Sequence[str], *, cwd: Path, log_path: Path) -> int:
    with open(log_path, "w", encoding="utf-8") as log:
        return subprocess.run(list(command), stdout=log, stderr=subprocess.STDOUT,
                              cwd=str(cwd)).returncode


# --------------------------------------------------------------------------
# La transmisión
# --------------------------------------------------------------------------

def transmission_from_fit(out_dir: str | Path, wave_A: Sequence[float]) -> np.ndarray:
    """T(λ) sobre el eje del cubo, en **aire**, y 1.0 donde no se ajustó.

    Dos trampas, las dos medidas:

    1. `MOLECFIT_DATA.lambda` sale en **vacío** (comprobado contra Edlén sobre el
       eje de entrada, a 0.00000 Å). A 7600 Å son 2.09 Å = 1.67 canales al rojo, y
       todas las bandas de A3 están definidas en aire. Las filas de la salida van
       **1:1** con las de la entrada, así que el eje bueno es el del propio
       espectro: exacto, no aproximado. Por eso esta función pide `wave_A` y
       nunca lee la columna `lambda`.
    2. `mtrans` vale **0** fuera de las ventanas ajustadas, no 1. Aplicarla tal
       cual dividiría el cubo entero por cero. Fuera de las ventanas
       (`mrange == 0`) no hay absorción telúrica que corregir y T = 1 es lo
       correcto — que es además lo que STD_TELLURIC hace de facto entre 6590 y
       6860 Å.
    """

    datos = Path(out_dir) / "MOLECFIT_DATA.fits"
    if not datos.exists():
        raise TelluricError(f"molecfit did not write MOLECFIT_DATA: {datos}")
    tabla = fits.getdata(str(datos), 1)
    mtrans = np.asarray(tabla["mtrans"], dtype=np.float64)
    mrange = np.asarray(tabla["mrange"], dtype=np.float64)
    wave = np.asarray(wave_A, dtype=np.float64)
    if mtrans.shape != wave.shape:
        raise TelluricError(
            f"MOLECFIT_DATA has {mtrans.size} rows but the cube axis has {wave.size}: "
            "the 1:1 correspondence that makes the air axis exact is broken.")
    trans = np.where(mrange > 0, mtrans, 1.0)
    trans = np.where(np.isfinite(trans) & (trans > 0), trans, 1.0)
    return enforce_protected_transmission(wave, trans)


def fit_params_fingerprint(*, params: Mapping[str, str], windows: Sequence[MolecfitWindow],
                           radius_px: float, cube_sha: str) -> str:
    """Huella para la bitácora: si cambia, la unidad se rehace."""

    return params_hash({
        "params": dict(params),
        "windows": [w.as_dict() for w in windows],
        "radius_px": float(radius_px),
        "cube_sha": cube_sha,
    })
