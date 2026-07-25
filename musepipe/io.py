"""FITS, JSON, CSV, and NumPy IO helpers shared by pipeline stages."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from astropy.io import fits


def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path, payload) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def write_csv(path, rows, fieldnames=None) -> None:
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def get_cube_data(hdul, cube_index=0):
    """Return a 3D cube from a FITS HDUList with common pipeline layouts."""

    if hdul[0].data is not None:
        data = hdul[0].data
    elif "CUBES" in hdul:
        data = hdul["CUBES"].data
    else:
        raise RuntimeError("No cube data found in FITS file.")

    if data.ndim == 4:
        return data[int(cube_index)]
    if data.ndim == 3:
        return data
    raise RuntimeError(f"Expected a 3D or 4D cube, got {data.shape}.")


def read_wavelength_axis(hdul, data_shape=None):
    """Return a wavelength axis from an extension or standard FITS keywords."""

    if "WAVELENGTH" in hdul:
        return hdul["WAVELENGTH"].data.astype(np.float64)

    header = hdul[0].header
    if data_shape is None:
        data = get_cube_data(hdul)
        data_shape = data.shape
    nz = int(data_shape[0])

    if all(key in header for key in ("WMIN", "DW")):
        return float(header["WMIN"]) + np.arange(nz, dtype=np.float64) * float(header["DW"])
    if all(key in header for key in ("CRVAL3", "CDELT3")):
        crpix = float(header.get("CRPIX3", 1.0))
        return float(header["CRVAL3"]) + (
            np.arange(nz, dtype=np.float64) + 1.0 - crpix
        ) * float(header["CDELT3"])
    raise RuntimeError("Could not recover wavelength axis from FITS file.")


def read_wavelengths_and_masks(stack_path, good_mask_path=None, bad_mask_path=None):
    """Read wavelength, good-mask, and bad-mask arrays from a stage FITS stack."""

    with fits.open(stack_path, memmap=True) as hdul:
        waves = hdul["WAVELENGTH"].data.astype(np.float64)
        if "GOOD_WAVE_MASK" in hdul:
            good = hdul["GOOD_WAVE_MASK"].data.astype(bool)
        else:
            good = np.ones(waves.size, dtype=bool)
        if "BAD_WAVE_MASK" in hdul:
            bad = hdul["BAD_WAVE_MASK"].data.astype(bool)
        else:
            bad = ~good

    if good_mask_path is not None and Path(good_mask_path).exists():
        good = np.load(good_mask_path).astype(bool)
    if bad_mask_path is not None and Path(bad_mask_path).exists():
        bad = np.load(bad_mask_path).astype(bool)
        good = good & ~bad
    return waves, good, bad


#: Unidad nativa de los cubos MUSE flux-calibrados por scipost.
MUSE_NATIVE_BUNIT = "10**(-20)*erg/s/cm**2/Angstrom"


def cube_bunit(path, ext=None) -> str:
    """`BUNIT` de un cubo, buscando en el HDU indicado y luego en el resto.

    Devuelve "" si ningun HDU lo declara: la unidad se desconoce y quien llama
    debe decidir (no se inventa la nativa de MUSE).
    """
    with fits.open(path, memmap=True) as hdul:
        order = [ext] if ext is not None else []
        order += [i for i in range(len(hdul)) if i != ext]
        for idx in order:
            try:
                value = hdul[idx].header.get("BUNIT")
            except (IndexError, KeyError):
                continue
            if value:
                return str(value)
    return ""


def resolve_bunit(cfg, *, stack_bunit=None, override_key=None) -> str:
    """Unidad de flujo de un producto espectral, en orden de confianza.

    1. la que trae el cubo de trabajo (`stack_bunit`, si el stack la declara);
    2. la del cubo de entrada del run (`cfg["cube_files"][0]`) — los stacks
       escritos antes de que B1/B2 propagaran `BUNIT` no la llevan, y re-correr
       B1/B2 solo por la cabecera cuesta ~20 min;
    3. el override explicito de config (`override_key`), si existe;
    4. "" — desconocida; el producto lo dira en vez de fingir una unidad.

    Gana el primer candidato que sea una unidad de flujo *interpretable*. El
    cubo residual de stage04b se etiqueta `BUNIT='physical_like'`, que no es una
    unidad sino una nota ("misma escala que el cubo, ya sin fondo"): si ganara
    por ser el primero, el producto perderia la unidad real, que la sustraccion
    de fondo no cambia. Si ninguno interpreta, se devuelve el primero no vacio
    para no tirar informacion.
    """
    candidates = [stack_bunit]
    cubes = (cfg or {}).get("cube_files") or []
    if cubes:
        try:
            candidates.append(cube_bunit(cubes[0], ext=(cfg or {}).get("data_ext")))
        except (OSError, ValueError):
            pass
    if override_key:
        candidates.append((cfg or {}).get(override_key))
    present = [str(value) for value in candidates if value]
    for value in present:
        if bunit_to_cgs_scale(value) is not None:
            return value
    return present[0] if present else ""


def bunit_to_cgs_scale(bunit) -> float | None:
    """Factor que lleva `bunit` a erg/s/cm2/A, o None si no es esa dimension.

    `'10**(-20)*erg/s/cm**2/Angstrom'` -> `1e-20`. Sirve para que una etapa lea
    la escala fisica del propio producto en vez de depender de un knob de
    config (ver `h03_flux_unit_cgs`).
    """
    text = str(bunit or "").strip()
    if not text:
        return None
    import warnings

    import astropy.units as u

    try:
        with warnings.catch_warnings():
            # El BUNIT de MUSE lleva varias barras: la FITS lo desaconseja y
            # astropy avisa, pero lo interpreta bien.
            warnings.simplefilter("ignore", u.UnitsWarning)
            unit = u.Unit(text)
        return float((1.0 * unit).to(u.erg / u.s / u.cm**2 / u.AA).value)
    except (ValueError, TypeError, u.UnitConversionError):
        return None


def flux_unit_from_m3_qc(qc00) -> float | None:
    """La escala de flujo que A4/M3 uso al comparar con Gaia, o None.

    M3 integra el espectro de la primaria y lo compara con el flujo de catalogo
    en cgs, asi que para medir `flux_factor` tuvo que fijar la unidad del cubo y
    la deja anotada en su QC (`m3_flux.flux_unit_cgs`). Es una tercera fuente de
    la misma unidad: no es el dato en si (eso es `BUNIT`), pero si el testimonio
    de la etapa que la uso para un numero que despues aplica D2.

    Se ignora si M3 no llego a medir: la unidad que anota un M3 `unavailable` no
    respalda nada.
    """
    m3 = (qc00 or {}).get("m3_flux") if isinstance(qc00, dict) else None
    if not isinstance(m3, dict):
        return None
    if str(m3.get("status", "unavailable")).lower() == "unavailable":
        return None
    value = m3.get("flux_unit_cgs")
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if np.isfinite(value) and value > 0 else None


def read_stage00q_qc(stage_dir, name="stage00q_qc.json"):
    """El QC de A4 (`stage00q_qc.json`) si esta, o None. Nunca falla por no estar.

    Atajo para las etapas que quieren la unidad de M3 como tercera fuente y no
    tienen ya el QC leido (ver `flux_unit_from_m3_qc`).
    """
    if stage_dir is None:
        return None
    path = Path(stage_dir) / name
    if not path.exists():
        return None
    try:
        return read_json(path)
    except (OSError, ValueError):
        return None


def resolve_flux_unit(cfg, *, bunit=None, key="h03_flux_unit_cgs", qc_m3=None):
    """`(escala_cgs, fuente)` con las tres fuentes de la unidad en un solo sitio.

    Orden de confianza:

    1. el knob explicito de config (`key`) — manda a proposito, para no mover
       resultados ya congelados (ver `flux_unit_cgs`);
    2. el `BUNIT` del propio producto — la unidad viaja con el dato;
    3. `m3_flux.flux_unit_cgs` del QC de A4/M3 — un producto viejo puede haber
       perdido la cabecera, pero si M3 midio el factor de flujo sobre ese mismo
       cubo, su unidad es la que sostiene la escala que D2 aplica;
    4. error explicito, nunca un default silencioso.
    """
    declared = (cfg or {}).get(key)
    if declared is not None:
        return float(declared), f"config:{key}"
    scale = bunit_to_cgs_scale(bunit)
    if scale is not None:
        return scale, f"BUNIT:{bunit}"
    from_m3 = flux_unit_from_m3_qc(qc_m3)
    if from_m3 is not None:
        return from_m3, "stage00q_qc.m3_flux.flux_unit_cgs"
    raise ValueError(
        f"No se puede fijar la escala de flujo: el config no declara {key!r}, el "
        f"producto no trae un BUNIT interpretable (BUNIT={bunit!r}) y el QC de A4/M3 "
        f"no aporta `flux_unit_cgs`. Declara {key} en runs/<RUN>/config/config.json "
        f"(los cubos MUSE de scipost son {MUSE_NATIVE_BUNIT} -> 1e-20) o re-genera "
        f"el producto con BUNIT."
    )


def flux_unit_cgs(cfg, *, bunit=None, key="h03_flux_unit_cgs", qc_m3=None) -> float:
    """Escala que lleva el flujo del producto a erg/s/cm2/A.

    Envoltorio de `resolve_flux_unit` que devuelve solo el valor; usa esa si
    necesitas ademas de donde salio (provenance, o para avisar de un conflicto).

    NO hay default silencioso: los defaults divergentes que habia (`1.0` en
    H03/G3, `1e-20` en `models.derived`, `1e-20` en M3) significaban que un run
    sin el knob producia L/Mdot con un factor 1e20 de diferencia segun quien lo
    leyera.
    """
    return resolve_flux_unit(cfg, bunit=bunit, key=key, qc_m3=qc_m3)[0]


def flux_unit_conflict(bunit, qc_m3, *, rtol=1e-6):
    """Mensaje si el `BUNIT` del producto y la unidad de M3 no son la misma, o None.

    Las dos describen el mismo cubo, asi que discrepar no es un detalle de
    provenance: `flux_factor` se midio dividiendo por una y D2 lo aplica a un
    flujo expresado en la otra, de modo que la escala absoluta sale mal por
    justo ese cociente (entre cubos MUSE y un knob a 1.0, un factor 1e20).
    """
    from_bunit = bunit_to_cgs_scale(bunit)
    from_m3 = flux_unit_from_m3_qc(qc_m3)
    if from_bunit is None or from_m3 is None:
        return None
    if np.isclose(from_bunit, from_m3, rtol=rtol, atol=0.0):
        return None
    return (
        f"La unidad del producto (BUNIT={bunit!r} -> {from_bunit:g}) y la que uso A4/M3 "
        f"para medir el factor de flujo ({from_m3:g}) no coinciden: la escala absoluta "
        f"queda mal por un factor {from_m3 / from_bunit:g}. Revisa `m3_flux_unit_cgs` en "
        f"el config del run y el BUNIT del cubo que midio M3."
    )


__all__ = [
    "MUSE_NATIVE_BUNIT",
    "bunit_to_cgs_scale",
    "flux_unit_cgs",
    "flux_unit_conflict",
    "flux_unit_from_m3_qc",
    "resolve_flux_unit",
    "cube_bunit",
    "get_cube_data",
    "read_json",
    "read_stage00q_qc",
    "read_wavelength_axis",
    "read_wavelengths_and_masks",
    "resolve_bunit",
    "write_csv",
    "write_json",
]
