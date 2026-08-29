"""Stage H04/E4 v2: Halpha injection-recovery with empirical null QC."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
import datetime as _dt
import math
import os
from pathlib import Path
import time

import numpy as np
from astropy.io import fits

from ..config import load_run_config
from ..extraction.product import SpectrumProduct
from ..injection import InjectionSource, create_run_clone, inject, tree_sha256
from ..io import read_json, read_wavelength_axis, write_csv, write_json
from ..paths import RunPaths
from ..spectral import continuum_running_median as _CONTINUUM_RUNMED
from ..stats import empirical_z, finite_values, robust_sigma, robust_sigma_axis0
from .stage08c_look_elsewhere import empirical_fap
from .stage_h01_detect import BAD_DETECTION_FLAGS, HALPHA_REST_A, matched_filter_point
from .stage_x10_compare import METHOD_ORDER


C_KMS = 299792.458
H01_MATCHED_FILTER_POINT = matched_filter_point
SPEC_VERSION = "E4_v3"
DEFAULT_SNR_GRID = (0.0, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0)
DEFAULT_TEMPLATE_FACTORS = (1.0, 2.0)
# sgf/lpm added in WP-H2 (docs/2026-07-15_plan_integracion_halosub.md): their
# throughput feeds G1 validation and D1 v3 exactly like the spatial methods.
DEFAULT_METHODS = ("aperture", "optimal_ls", "optimal_psfsub", "psffit", "sgf", "lpm")
CONTINUUM_METHODS = ("aperture", "optimal_ls", "optimal_psfsub", "psffit")
DEFAULT_CONTINUUM_SIDEBANDS_A = ((6500.0, 6540.0), (6585.0, 6625.0))
DEFAULT_NULL_P = 0.0455
DEFAULT_NULL_GATE_ALPHA = 0.01

#: Como se pone en escala la SNR de una inyeccion antes de decidir `complete`.
#:
#: `none` es la SNR formal del filtro adaptado, que es lo que E4 ha usado
#: siempre. Medido el 2026-08-28
#: (`docs/2026-08-28_umbral_por_metodo_medido.md`): con senal nula esa SNR no
#: tiene media 0 ni sigma 1 en ningun metodo, y falla de dos formas distintas
#: —pedestal en el cubo combinado (`psffit` +5.55 en ROXs 12 b, `optimal_ls`
#: -3.66), cola por exposicion (`psffit` con sigma 4.86)— que un umbral unico no
#: puede corregir en ninguna de las dos direcciones.
#:
#: `injection_nulls` la estandariza contra las propias inyecciones nulas: las
#: filas con `input_snr = 0` en posiciones de control, del mismo metodo, ancho de
#: plantilla y modo de continuo.
#:
#: **La referencia tiene que ser esa y no la de la puerta V2.** V2 compara contra
#: `empirical_null_reference`, que son 33 controles de PRODUCCION repartidos por
#: el campo; las inyecciones nulas estan al radio del companero y pasan por la
#: maquinaria de inyeccion. Son dos poblaciones distintas, medido: en ROXs 12 b
#: `aperture` da 175.0 +- 20.4 en produccion y 60.3 +- 162.8 en inyeccion —un
#: factor 8 en la escala—, y estandarizar contra la de produccion deja la nula en
#: -6.55 +- 6.98 en vez de arreglarla. Contra la propia, los doce casos (seis
#: metodos, dos objetos) caen en media -0.44..+0.31 y sigma 0.95..2.21.
SNR_STANDARDIZATION_MODES = ("none", "injection_nulls")

#: Con cualquier modo distinto de `none` el QC declara E4 v4: cambia COMO se
#: decide `complete`, que es una regla y no un parametro. La version lo dice
#: para que un producto no se pueda confundir con otro calculado de la otra
#: forma.
SPEC_VERSION_STANDARDIZED_SNR = "E4_v4"

#: La SNR **inyectada** a la que se reporta la completitud. Era el mismo numero
#: que `h04_detection_threshold_snr` porque `_completeness` usaba el umbral para
#: las dos cosas: el corte de deteccion y el nivel al que se mide. Son dos
#: conceptos y ahora son dos perillas; el valor por defecto conserva el 5.0 que
#: producia el acoplamiento, asi que nada se mueve mientras no se declare.
DEFAULT_COMPLETENESS_INPUT_SNR = 5.0


TABLE_FIELDS = [
    "injection_id",
    "variant",
    "method",
    "position_label",
    "position_y",
    "position_x",
    "template_width",
    "template_factor",
    "continuum_mode",
    "psf_fwhm_scale",
    "snr",
    "input_snr",
    "injected_flux",
    "recovered_flux",
    "recovered_flux_baseline",
    "recovered_flux_net",
    "recovered_sigma",
    "recovered_snr",
    # La SNR estandarizada contra la nula empirica. Se emite SIEMPRE, tambien
    # con `h04_snr_standardization="none"`: es el diagnostico que dice si la
    # SNR formal esta en escala, y medirlo no puede depender de haber decidido
    # ya que no lo esta.
    "recovered_snr_std",
    "throughput",
    "throughput_err",
    "complete",
    "bias_flux_pct",
    "estimator",
]


@dataclass(frozen=True)
class H04Case:
    injection_id: str
    variant: str
    position_label: str
    position_y: float
    position_x: float
    template_factor: float
    input_snr: float
    continuum_mode: str
    psf_fwhm_scale: float = 1.0


@dataclass(frozen=True)
class StageH04Product:
    rows: list[dict]
    qc: dict


def _finite_or_none(value):
    if value is None:
        return None
    val = float(value)
    if not np.isfinite(val):
        return None
    return val


def _json_ready(value):
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return _finite_or_none(value)
    return value


def stage_h04_paths(run_id, project_root=None):
    root = Path(project_root or Path.cwd()).resolve()
    paths = RunPaths.from_project_root(run_id, root)
    plot_dir = paths.plot_stage_dir("stage_h04")
    return {
        "paths": paths,
        "stage02_cube_fits": paths.stage_dir / "stage02_xcorr_cube_stack.fits",
        "stage00q_qc_json": paths.stage_dir / "stage00q_qc.json",
        "stage01c_qc_json": paths.stage_dir / "stage01c_qc.json",
        "psf_model_json": paths.stage_dir / "psf_model.json",
        "stage06_truth_json": paths.table_dir / "stage06_local_surface_halpha_injection_truth_summary.json",
        "throughput_csv": paths.table_dir / "injection_throughput_by_method.csv",
        "stage_h04_qc_json": paths.stage_dir / "stage_h04_qc.json",
        "clone_manifest_json": paths.stage_dir / "stage_h04_clone_manifest.json",
        "plot_dir": plot_dir,
        "throughput_plot": plot_dir / "stage_h04_throughput.png",
        "recovery_plot": plot_dir / "stage_h04_recovered_vs_injected.png",
        "completeness_plot": plot_dir / "stage_h04_completeness.png",
    }


def stage_h04_config_from_run(
    run_id=None,
    *,
    project_root=None,
    overrides=None,
    allow_run_id_mismatch=False,
):
    run_config = load_run_config(
        run_id,
        project_root=project_root,
        allow_run_id_mismatch=allow_run_id_mismatch,
    )
    cfg = dict(run_config.config)
    if overrides:
        cfg.update(overrides)
    cfg["run_id"] = run_config.run_id
    cfg["project_root"] = str(run_config.paths.project_root)
    cfg.setdefault("h04_methods", list(DEFAULT_METHODS))
    cfg.setdefault("h04_snr_grid", list(DEFAULT_SNR_GRID))
    cfg.setdefault("h04_template_width_factors", list(DEFAULT_TEMPLATE_FACTORS))
    cfg.setdefault("h04_continuum_modes", ["none", "flat"])
    cfg.setdefault("h04_psf_perturb_snr_grid", [3.0, 5.0])
    cfg.setdefault("h04_psf_perturb_scales", [0.9, 1.1])
    cfg.setdefault("h04_max_runtime_hours", 4.0)
    cfg.setdefault("h04_expected_seconds_per_case_method", DEFAULT_SECONDS_PER_CASE_METHOD)
    cfg.setdefault("h04_allow_long_run", False)
    cfg.setdefault("h04_detection_threshold_snr", 5.0)
    cfg.setdefault("h04_snr_standardization", "none")
    cfg.setdefault("h04_completeness_input_snr", DEFAULT_COMPLETENESS_INPUT_SNR)
    cfg.setdefault("h04_continuum_sidebands_A", [list(band) for band in DEFAULT_CONTINUUM_SIDEBANDS_A])
    cfg.setdefault("h04_null_p", DEFAULT_NULL_P)
    cfg.setdefault("h04_null_gate_alpha", DEFAULT_NULL_GATE_ALPHA)
    cfg.setdefault("h04_baseline_subtract_throughput", True)
    # La convencion en la que se expresa `injected_flux`. Con `norm_radius`
    # coincide con la que devuelven los extractores (NORMRAD), y el throughput
    # deja de depender del modelo de PSF. `frame` reproduce el camino historico.
    cfg.setdefault("h04_injection_norm_convention", "norm_radius")
    # El 8.97 es un default de ESTE modulo: la medida que lo respalda no esta en
    # el repo. Se anota si el run lo declara de verdad, porque despues del
    # `setdefault` ya no hay forma de distinguirlo y el QC acabaria diciendo que
    # sale de la config cuando sale de aqui (`historic_regression_check`).
    cfg["h04_historic_expected_snr_declared"] = "h04_historic_expected_snr" in cfg
    cfg.setdefault("h04_historic_expected_snr", 8.97)
    cfg.setdefault("h04_historic_tolerance_snr", 0.25)
    cfg.setdefault("h04_require_historic_regression", True)
    cfg.setdefault("h04_injection_flux_sigma", cfg.get("h01_matched_sigma"))
    return cfg


def template_width_label(factor):
    val = float(factor)
    if np.isclose(val, 1.0):
        return "lsf"
    if val.is_integer():
        return f"{int(val)}x_lsf"
    return f"{val:g}x_lsf"


def _read_optional_json(path):
    path = Path(path)
    if not path.exists():
        return None
    return read_json(path)


def resolve_continuum_injection(config, paths):
    """Resolve the non-zero flat continuum required by E4 v2.

    Nota sobre la etiqueta `scale="normrad_flux_density"`: es un nombre heredado
    y **equivocado**. Lo que E4 usa no es la convencion `normrad` de
    `growth_curve.resolve_flux_convention`, sino la escala CRUDA del extractor:
    aqui se divide por `apcorr` (abajo) y la recuperacion se mide sobre la
    salida en memoria del extractor, que tampoco lo lleva
    (`_coerce_spectrum_product` no toca `apcorr`).

    Que la escala sea la cruda es lo que hace que E3 pueda dividir por el
    throughput sin convertir nada: `throughput = recuperado/inyectado` con los
    dos terminos en la misma escala y el pedestal restado
    (`_apply_baseline_subtraction`) es ADIMENSIONAL, asi que no le afecta
    `apcorr` ni la convencion de flujo del run. Por eso el arreglo de la curva de
    crecimiento del 2026-08-07 no obligo a re-correr E4. La etiqueta se conserva
    tal cual porque viaja en el QC y F1/G0/G5 verifican esquemas; lo que estaba
    mal era el nombre, no el numero.
    """

    configured = config.get("h04_continuum_flux_density")
    bands = [
        [float(lo), float(hi)]
        for lo, hi in config.get("h04_continuum_sidebands_A", DEFAULT_CONTINUUM_SIDEBANDS_A)
    ]
    if configured is not None:
        value = float(configured)
        if not np.isfinite(value) or value <= 0:
            raise RuntimeError("h04_continuum_flux_density must be finite and positive for E4 v2.")
        return {
            "value": value,
            "scale": "normrad_flux_density",
            "source": "config",
            "bands_A": bands,
            "by_method": {},
            "products": {},
        }

    stage_dir = paths["paths"].stage_dir
    product_paths = {
        "aperture": stage_dir / "spec_aperture_object.fits",
        "optimal_ls": stage_dir / "spec_optimal_object.fits",
        "optimal_psfsub": stage_dir / "spec_optimal_psfsub_object.fits",
        "psffit": stage_dir / "spec_psffit_object.fits",
    }
    by_method = {}
    for method in CONTINUUM_METHODS:
        path = product_paths[method]
        if not path.exists():
            raise FileNotFoundError(path)
        product = SpectrumProduct.read(path)
        wave = np.asarray(product.wave_A, dtype=np.float64)
        flux = np.asarray(product.flux, dtype=np.float64)
        apcorr = np.asarray(product.apcorr, dtype=np.float64)
        flags = np.asarray(product.flags, dtype=np.int32)
        sideband = np.zeros(wave.size, dtype=bool)
        for lo, hi in bands:
            sideband |= (wave >= lo) & (wave <= hi)
        good = (
            sideband
            & np.isfinite(flux)
            & np.isfinite(apcorr)
            & (apcorr > 0)
            & ((flags & BAD_DETECTION_FLAGS) == 0)
        )
        if not np.any(good):
            raise RuntimeError(f"No usable E4 v2 continuum channels for {method}.")
        value = float(np.nanmedian(flux[good] / apcorr[good]))
        by_method[method] = value

    usable = np.asarray([value for value in by_method.values() if np.isfinite(value) and value > 0], dtype=float)
    if usable.size < 2:
        raise RuntimeError(f"E4 v2 needs at least two finite positive continuum measurements: {by_method}.")
    return {
        "value": float(np.median(usable)),
        "scale": "normrad_flux_density",
        "source": "median_continuum_preserving_methods",
        "bands_A": bands,
        "by_method": by_method,
        "products": {method: str(path) for method, path in product_paths.items()},
    }


def build_empirical_null_reference(config, paths, methods, *, lsf_fwhm_A):
    """Matched-filter flux nulls from production controls in NORMRAD scale."""

    stage_dir = paths["paths"].stage_dir
    product_names = {
        "aperture": "spec_aperture_object.fits",
        "optimal_ls": "spec_optimal_object.fits",
        "optimal_psfsub": "spec_optimal_psfsub_object.fits",
        "psffit": "spec_psffit_object.fits",
        "sgf": "spec_sgf_object.fits",
        "lpm": "spec_lpm_object.fits",
    }
    control_names = {
        "aperture": "spec_aperture_controls.npz",
        "optimal_ls": "spec_optimal_controls.npz",
        "optimal_psfsub": "spec_optimal_psfsub_controls.npz",
        "psffit": "spec_psffit_controls.npz",
        "sgf": "spec_sgf_controls.npz",
        "lpm": "spec_lpm_controls.npz",
    }
    center_A = _line_center_from_config(config)
    continuum_window_A = float(config.get("h04_continuum_window_A", config.get("h01_continuum_window_A", 80.0)))
    width_factors = [float(value) for value in config.get("h04_template_width_factors", DEFAULT_TEMPLATE_FACTORS)]
    reference = {}
    for method in methods:
        product_path = stage_dir / product_names[method]
        control_path = stage_dir / control_names[method]
        if not product_path.exists():
            raise FileNotFoundError(product_path)
        if not control_path.exists():
            raise FileNotFoundError(control_path)
        product = SpectrumProduct.read(product_path)
        with np.load(control_path) as payload:
            if "control_spectra_raw" in payload:
                controls = np.asarray(payload["control_spectra_raw"], dtype=np.float64)
                scale_source = "control_spectra_raw"
            elif "control_spectra" in payload:
                controls = np.asarray(payload["control_spectra"], dtype=np.float64)
                controls = controls / np.asarray(product.apcorr, dtype=np.float64)[None, :]
                scale_source = "control_spectra/apcorr"
            else:
                raise RuntimeError(f"No control spectra found in {control_path}.")
        wave = np.asarray(product.wave_A, dtype=np.float64)
        if controls.ndim != 2 or controls.shape[1] != wave.size or controls.shape[0] < 2:
            raise RuntimeError(f"Invalid E4 v2 controls for {method}: {controls.shape}.")
        flags = np.asarray(product.flags, dtype=np.int32)
        good = np.isfinite(wave) & ((flags & BAD_DETECTION_FLAGS) == 0)
        residuals = np.empty_like(controls)
        min_pixels = max(3, min(15, int(np.count_nonzero(good))))
        for index, spectrum in enumerate(controls):
            finite = good & np.isfinite(spectrum)
            continuum = _CONTINUUM_RUNMED(
                wave,
                spectrum,
                finite,
                window_A=continuum_window_A,
                min_pixels=min_pixels,
            )
            residuals[index] = spectrum - continuum
        error = robust_sigma_axis0(residuals).astype(np.float64)
        error[~np.isfinite(error) | (error <= 0)] = np.nan
        by_factor = {}
        for factor in width_factors:
            fluxes = []
            for residual in residuals:
                flux, _sigma, _snr = matched_filter_point(
                    wave,
                    residual,
                    error,
                    center_A,
                    float(lsf_fwhm_A) * factor,
                    good,
                )
                fluxes.append(flux)
            values = np.asarray(fluxes, dtype=np.float64)
            if np.count_nonzero(np.isfinite(values)) < 2:
                raise RuntimeError(f"No finite E4 v2 empirical null for {method}, factor={factor:g}.")
            by_factor[f"{factor:g}"] = values
        reference[method] = {
            "n_controls": int(controls.shape[0]),
            "scale_source": scale_source,
            "product": str(product_path),
            "controls": str(control_path),
            "by_factor": by_factor,
        }
    return reference


def null_scale(reference_values):
    """El centro y la escala de la nula empirica, en las unidades del flujo.

    Es la pareja que usa `empirical_z` (mediana y sigma robusta), publicada
    aparte porque el QC tiene que poder decir **contra que** se estandarizo. Con
    33 controles la sigma robusta trae ~12 % de error relativo: el numero sirve
    para poner en escala, no para citarlo con tres cifras.
    """

    values = finite_values(reference_values)
    if values.size < 2:
        return {"center": np.nan, "sigma": np.nan, "n": int(values.size)}
    sigma = robust_sigma(values)
    return {
        "center": float(np.median(values)),
        "sigma": float(sigma) if np.isfinite(sigma) and sigma > 0 else np.nan,
        "n": int(values.size),
    }


def injection_null_reference(rows):
    """Las inyecciones nulas, agrupadas por el estrato en el que son comparables.

    Estrato = metodo x ancho de plantilla x modo de continuo. Cruzar estratos
    seria comparar cantidades de escalas distintas: el ancho cambia la escala del
    filtro adaptado y el modo de continuo cambia lo que se resta antes.

    Solo posiciones de **control**: la real puede llevar senal del companero, y
    meterla en la referencia es lo que convierte una deteccion en el cero.
    """

    reference = {}
    for row in rows:
        if row["variant"] != "nominal" or float(row["input_snr"]) != 0.0:
            continue
        if not str(row["position_label"]).startswith("control"):
            continue
        key = _standardization_key(row)
        reference.setdefault(key, []).append(
            (str(row["injection_id"]), float(row["recovered_flux"]))
        )
    return reference


def _standardization_key(row):
    return (
        str(row["method"]),
        f"{float(row['template_factor']):g}",
        str(row["continuum_mode"]),
    )


def apply_snr_standardization(rows, *, mode, threshold_snr):
    """Anade `recovered_snr_std` y, con `injection_nulls`, redecide `complete`.

    Corre como pasada posterior sobre las filas, no dentro del calculo de cada
    caso, por dos razones: la referencia son las propias filas nulas, que solo
    existen cuando estan todas, y el calculo por caso lo comparten los tres
    backends de ejecucion (serie, hilos y fork) a traves de un `ctx` de solo
    lectura, donde meter esto seria una fuente nueva de divergencia.

    Se estandariza `recovered_flux` —no `recovered_flux_net`— porque es la
    cantidad que la referencia mide: las filas nulas tambien traen su flujo sin
    restar baseline.

    **Una fila nula se estandariza dejandose fuera** (leave-one-out). Si no, cada
    nula entra en su propio centro y su propia escala: medido sobre los 15
    controles de ROXs 12 b, no hacerlo encoge el |z| de la nula un 13 % en la
    mediana (los deciles van de +2 % a -29 %), asi que la tasa de falsos
    positivos que se midiera con ellas saldria optimista por construccion.
    """

    if mode not in SNR_STANDARDIZATION_MODES:
        raise ValueError(
            f"`h04_snr_standardization` solo entiende {SNR_STANDARDIZATION_MODES}, no {mode!r}."
        )
    reference = injection_null_reference(rows)
    if mode == "injection_nulls":
        faltan = sorted(
            {
                _standardization_key(row)
                for row in rows
                if len(reference.get(_standardization_key(row), ())) < 2
            }
        )
        if faltan:
            # Sin referencia no hay escala, y heredar la SNR formal seria volver
            # justo a la escala que se ha declarado mala. Se para en vez de
            # publicar filas medidas de dos maneras distintas.
            raise ValueError(
                "`h04_snr_standardization=injection_nulls` necesita al menos dos inyecciones "
                f"nulas en posiciones de control por estrato; faltan en: {faltan}. "
                "Incluye 0.0 en `h04_snr_grid` para cada ancho y modo de continuo de la rejilla."
            )
    out = []
    for row in rows:
        pares = reference.get(_standardization_key(row), [])
        es_nula = (
            row["variant"] == "nominal"
            and float(row["input_snr"]) == 0.0
            and str(row["position_label"]).startswith("control")
        )
        valores = [
            flux
            for injection_id, flux in pares
            if not (es_nula and injection_id == str(row["injection_id"]))
        ]
        standardized = (
            empirical_z(float(row["recovered_flux"]), valores) if len(valores) >= 2 else np.nan
        )
        updated = dict(row)
        updated["recovered_snr_std"] = float(standardized)
        if mode == "injection_nulls":
            updated["complete"] = bool(
                np.isfinite(standardized) and standardized >= float(threshold_snr)
            )
        out.append(updated)
    return out


def _reference_population_check(rows, null_reference):
    """Compara la nula de inyeccion con la de produccion, metodo a metodo.

    Deberian describir lo mismo —una medida sin senal— y no lo hacen: las
    inyecciones nulas estan al radio del companero y pasan por la maquinaria de
    inyeccion, mientras los 33 controles de produccion estan repartidos por el
    campo. Medido el 2026-08-28 en ROXs 12 b, `aperture`: 175.0 +- 20.4 en
    produccion contra 60.3 +- 162.8 en inyeccion.

    No vota. Es un diagnostico, y su destinatario real es la puerta V2, que
    compara las filas de la primera contra la distribucion de la segunda.
    """

    out = {}
    injection = injection_null_reference(rows)
    for (method, factor, _continuum), pairs in sorted(injection.items()):
        production = ((null_reference or {}).get(method) or {}).get("by_factor", {}).get(factor)
        if production is None:
            continue
        mine = null_scale([flux for _id, flux in pairs])
        theirs = null_scale(production)
        offset = np.nan
        if np.isfinite(theirs["sigma"]) and theirs["sigma"] > 0:
            offset = (mine["center"] - theirs["center"]) / theirs["sigma"]
        ratio = np.nan
        if np.isfinite(theirs["sigma"]) and theirs["sigma"] > 0 and np.isfinite(mine["sigma"]):
            ratio = mine["sigma"] / theirs["sigma"]
        out[f"{method}|{factor}"] = {
            "injection": {k: _finite_or_none(v) if k != "n" else v for k, v in mine.items()},
            "production": {k: _finite_or_none(v) if k != "n" else v for k, v in theirs.items()},
            "center_offset_in_production_sigma": _finite_or_none(offset),
            "sigma_ratio_injection_over_production": _finite_or_none(ratio),
        }
    return out


def null_distribution_diagnostics(rows, methods):
    """Media y sigma de la SNR con senal nula, cruda y estandarizada.

    Si la SNR estuviera en escala, con senal nula tendria media 0 y sigma 1.
    Publicarlo en el QC es lo que convierte "el umbral no cuadra" en un numero
    que se puede mirar sin re-correr la etapa. Solo posiciones de control: la
    real puede llevar senal, y contarla como nula es lo que convierte una
    deteccion en un falso positivo.
    """

    out = {}
    for method in methods:
        nulls = [
            row
            for row in rows
            if row["method"] == method
            and row["variant"] == "nominal"
            and float(row["input_snr"]) == 0.0
            and str(row["position_label"]).startswith("control")
        ]
        if not nulls:
            continue
        entry = {"n": len(nulls)}
        for key, column in (("raw", "recovered_snr"), ("standardized", "recovered_snr_std")):
            values = finite_values([row.get(column, np.nan) for row in nulls])
            entry[key] = {
                "mean": _finite_or_none(float(np.mean(values)) if values.size else np.nan),
                "sigma": _finite_or_none(
                    float(np.std(values)) if values.size > 1 else np.nan
                ),
                "n_finite": int(values.size),
            }
        out[method] = entry
    return out


def _source_pos_yx(qc, *keys):
    for key in keys:
        item = (qc or {}).get(key)
        if isinstance(item, dict) and item.get("pos_yx") is not None:
            yx = item["pos_yx"]
            return float(yx[0]), float(yx[1])
    raise KeyError(f"Could not find any position in {keys}.")


def _same_radius_control_positions(star_yx, target_yx, *, n_controls=3):
    sy, sx = map(float, star_yx)
    ty, tx = map(float, target_yx)
    radius = math.hypot(ty - sy, tx - sx)
    start = math.atan2(ty - sy, tx - sx)
    controls = []
    for index in range(int(n_controls)):
        theta = start + 2.0 * math.pi * (index + 1) / float(n_controls + 1)
        controls.append((sy + radius * math.sin(theta), sx + radius * math.cos(theta)))
    return controls


def resolve_h04_positions(config, paths=None):
    """La rejilla de posiciones: la real del companero + N controles al mismo radio.

    `h04_n_control_positions` (por defecto 3, que es lo que habia) manda sobre las
    tres cosas a la vez, porque son la misma: el `n` de `completeness_at_5sigma`
    —que con 3 controles solo puede valer 0, 0.25, 0.5, 0.75 o 1—, la poblacion
    nula de V2 y la potencia de su puerta. La spec E4 v3 §2 ya lo dice: con 3
    posiciones la puerta solo puede fallar si 2 de 3 son extremas, y subir eso
    pide mas POSICIONES, no mas filas por posicion. Anadirlas no necesita spec
    nueva: `n_positions` sale de los datos.
    """

    n_controls = int(config.get("h04_n_control_positions", 3))
    if n_controls < 1:
        raise RuntimeError("H04 needs at least one control position.")
    raw_positions = config.get("h04_positions_yx")
    if raw_positions:
        out = []
        for index, item in enumerate(raw_positions):
            if isinstance(item, dict):
                label = str(item.get("label", f"pos{index}"))
                y = float(item["y"])
                x = float(item["x"])
            else:
                label = "real" if index == 0 else f"control{index}"
                y = float(item[0])
                x = float(item[1])
            out.append({"label": label, "y": y, "x": x})
        if len(out) != 4:
            raise RuntimeError(
                f"H04 grid requires exactly {n_controls + 1} positions: real + {n_controls} "
                f"controls (h04_n_control_positions={n_controls}); got {len(out)}.")
        return out

    real = config.get("h04_real_position_yx")
    controls = config.get("h04_control_positions_yx")
    if real is not None and controls is not None:
        out = [{"label": "real", "y": float(real[0]), "x": float(real[1])}]
        out.extend(
            {"label": f"control{index + 1}", "y": float(y), "x": float(x)}
            for index, (y, x) in enumerate(controls)
        )
        if len(out) != n_controls + 1:
            raise RuntimeError(
                f"H04 requires exactly {n_controls} control positions "
                f"(h04_n_control_positions); got {len(out) - 1}.")
        return out

    if paths is None:
        raise RuntimeError("H04 requires h04_positions_yx or stage01c_qc.json.")
    qc = _read_optional_json(paths["stage01c_qc_json"])
    target = _source_pos_yx(qc, "companion", "target", "object")
    star = _source_pos_yx(qc, "primary", "star")
    out = [{"label": "real", "y": float(target[0]), "x": float(target[1])}]
    out.extend(
        {"label": f"control{index + 1}", "y": float(y), "x": float(x)}
        for index, (y, x) in enumerate(
            _same_radius_control_positions(star, target, n_controls=n_controls))
    )
    return out


def build_h04_cases(config, positions):
    cases = []
    snr_grid = [float(value) for value in config.get("h04_snr_grid", DEFAULT_SNR_GRID)]
    factors = [float(value) for value in config.get("h04_template_width_factors", DEFAULT_TEMPLATE_FACTORS)]
    continuum_modes = [str(value) for value in config.get("h04_continuum_modes", ["none", "flat"])]
    for pos in positions:
        for factor in factors:
            for continuum in continuum_modes:
                for snr in snr_grid:
                    cases.append(
                        H04Case(
                            injection_id=f"inj{len(cases):04d}",
                            variant="nominal",
                            position_label=str(pos["label"]),
                            position_y=float(pos["y"]),
                            position_x=float(pos["x"]),
                            template_factor=factor,
                            input_snr=snr,
                            continuum_mode=continuum,
                            psf_fwhm_scale=1.0,
                        )
                    )
    real = positions[0]
    for snr in [float(value) for value in config.get("h04_psf_perturb_snr_grid", [3.0, 5.0])]:
        for scale in [float(value) for value in config.get("h04_psf_perturb_scales", [0.9, 1.1])]:
            label = "psf_minus10" if scale < 1.0 else "psf_plus10"
            cases.append(
                H04Case(
                    injection_id=f"inj{len(cases):04d}",
                    variant=label,
                    position_label=str(real["label"]),
                    position_y=float(real["y"]),
                    position_x=float(real["x"]),
                    template_factor=1.0,
                    input_snr=snr,
                    continuum_mode="none",
                    psf_fwhm_scale=scale,
                )
            )
    return cases


def _resolve_h04_n_jobs(config, n_cases):
    """Worker count for the E4 case grid (ThreadPool over independent cases)."""
    from ..parallel import resolve_n_jobs

    requested = config.get("h04_n_jobs")
    if requested is None:
        return 1  # default: serial, to keep the frozen behaviour unless opted in
    return max(1, min(resolve_n_jobs(requested), int(n_cases)))


def _resolve_h04_process_pool(config, n_cases):
    """Worker count for PROCESS-based parallelism over the E4 case grid.

    Returns 0 when disabled (the default). The ThreadPool tops out near ~1.6x
    because the per-channel psffit work builds the PSF design in Python (GIL) on
    top of the BLAS calls. A fork ProcessPool lets each worker inherit the ~3GB
    base cube and the (unpicklable) extractor closures copy-on-write, so real
    cores are used without pickling cubes; only the small H04Case goes in and the
    per-case row list comes back. ``h04_process_pool`` accepts an int worker
    count, ``True`` (cpu-based via ``resolve_n_jobs``), or ``None``/``False`` to
    stay disabled. A resolved count < 2 disables it (a 1-worker pool is pointless).
    """
    from ..parallel import resolve_n_jobs

    requested = config.get("h04_process_pool")
    if requested is None or requested is False:
        return 0
    n = resolve_n_jobs(None) if requested is True else resolve_n_jobs(requested)
    n = min(int(n), int(n_cases))
    return n if n >= 2 else 0


def _fork_supported():
    """True on POSIX platforms where multiprocessing can use the fork start method."""
    import multiprocessing as mp

    return "fork" in mp.get_all_start_methods()


#: Coste SERIE por caso-metodo, medido (no supuesto) sobre ROXs12b_realigned el
#: 2026-07-27: 112 casos x 6 metodos en 1320 s de reloj con 8 hilos, y el
#: ThreadPool rinde ~1.6x por el GIL -> ~3.1 s en serie. El default anterior
#: eran 60 s, ~19x de mas, que convertia un trabajo de 22 min en "11.2 h" y
#: encendia `requires_checkpoint` sin motivo.
DEFAULT_SECONDS_PER_CASE_METHOD = 3.1
#: Aceleracion efectiva del ThreadPool (limitada por el GIL, medida ~1.6x). Un
#: ProcessPool con fork si escala con los workers.
THREADPOOL_SPEEDUP = 1.6


def estimate_runtime_budget(config, n_cases, n_methods, *, n_workers=1, forked=False):
    """Presupuesto de reloj de E4, contando el paralelismo.

    La estimacion anterior multiplicaba casos x metodos x 60 s y **ignoraba que
    E4 corre en paralelo**, asi que sobrestimaba por ~30x. Aqui el coste serie
    se divide por la aceleracion real: lineal en los workers con ProcessPool
    (fork), y limitada por el GIL con ThreadPool.

    Sigue siendo una estimacion a priori: el coste real se mide y se escribe en
    el mismo bloque de QC (`measured_*`), para que la desviacion sea visible en
    vez de acumularse en silencio.
    """

    per_case = float(config.get("h04_expected_seconds_per_case_method", DEFAULT_SECONDS_PER_CASE_METHOD))
    serial_seconds = float(n_cases) * float(n_methods) * per_case
    workers = max(1, int(n_workers))
    speedup = float(workers) if forked else min(float(workers), THREADPOOL_SPEEDUP)
    seconds = serial_seconds / max(speedup, 1.0)
    hours = seconds / 3600.0
    max_hours = float(config.get("h04_max_runtime_hours", 4.0))
    return {
        "n_cases": int(n_cases),
        "n_methods": int(n_methods),
        "n_workers": workers,
        "backend": "process_fork" if forked else ("thread" if workers > 1 else "serial"),
        "seconds_per_case_method": per_case,
        "assumed_speedup": speedup,
        "estimated_serial_seconds": serial_seconds,
        "estimated_seconds": seconds,
        "estimated_hours": hours,
        "max_hours_without_checkpoint": max_hours,
        "requires_checkpoint": bool(hours > max_hours and not bool(config.get("h04_allow_long_run", False))),
    }


def close_runtime_budget(budget, elapsed_seconds):
    """Cierra el presupuesto con lo que costo de verdad.

    Sin esto la estimacion no se contrasta nunca con el resultado y puede
    quedarse 30x desviada indefinidamente, que es justo lo que paso.
    """

    out = dict(budget)
    elapsed = float(elapsed_seconds)
    n = max(1, int(budget.get("n_cases", 1)) * int(budget.get("n_methods", 1)))
    out["measured_seconds"] = elapsed
    out["measured_hours"] = elapsed / 3600.0
    out["measured_seconds_per_case_method_wall"] = elapsed / n
    speedup = float(budget.get("assumed_speedup", 1.0)) or 1.0
    out["measured_seconds_per_case_method_serial"] = elapsed * speedup / n
    estimated = float(budget.get("estimated_seconds", 0.0))
    out["estimate_over_measured"] = (estimated / elapsed) if elapsed > 0 else None
    return out


def _lsf_fwhm_from_config_or_qc(config, paths=None):
    for key in ("h04_lsf_fwhm_A", "h01_lsf_fwhm_A", "lsf_fwhm_A"):
        if config.get(key) is not None:
            return float(config[key]), f"config.{key}"
    qc = _read_optional_json(paths["stage00q_qc_json"]) if paths is not None else None
    m2 = (qc or {}).get("m2_lsf", {})
    for key in ("fwhm_at_halpha_A", "halpha_fwhm_A", "lsf_fwhm_A"):
        if m2.get(key) is not None:
            return float(m2[key]), f"stage00q_qc.m2_lsf.{key}"
    raise RuntimeError("H04 requires LSF FWHM at Halpha from config or stage00q QC.")


def _line_center_from_config(config):
    if config.get("h04_line_center_A") is not None:
        return float(config["h04_line_center_A"])
    rv = float(config.get("h04_rv_sys_kms", config.get("rv_sys_kms", config.get("systemic_rv_kms", 0.0))))
    return float(HALPHA_REST_A * (1.0 + rv / C_KMS))


def _injection_sigma(config):
    value = config.get("h04_injection_flux_sigma")
    if value is None:
        raise RuntimeError("H04 requires h04_injection_flux_sigma to convert input S/N into line flux.")
    val = float(value)
    if not np.isfinite(val) or val <= 0:
        raise RuntimeError("h04_injection_flux_sigma must be positive and finite.")
    return val


def _load_stage02_cube(paths, config):
    path = Path(config.get("h04_stage02_cube_fits", paths["stage02_cube_fits"]))
    if not path.exists():
        raise FileNotFoundError(path)
    with fits.open(path, memmap=False) as hdul:
        if "CUBES" in hdul:
            cube = hdul["CUBES"].data.astype(np.float64)
            wave = hdul["WAVELENGTH"].data.astype(np.float64) if "WAVELENGTH" in hdul else read_wavelength_axis(hdul)
        elif hdul[0].data is not None:
            cube = hdul[0].data.astype(np.float64)
            wave = read_wavelength_axis(hdul, data_shape=cube.shape[-3:])
        else:
            raise RuntimeError(f"No cube data found in {path}.")
    return cube, wave, path


def _load_psf_model(paths, config):
    if config.get("h04_psf_model") is not None:
        return config["h04_psf_model"], "config.h04_psf_model"
    path = Path(config.get("h04_psf_model_json", paths["psf_model_json"]))
    if not path.exists():
        raise FileNotFoundError(path)
    return read_json(path), str(path)


def _load_injection_psf_model(paths, config, extraction_model, extraction_source):
    """La PSF con la que se INYECTA, que por defecto es la misma que extrae.

    Existe porque el throughput de esta etapa es **dependiente del modelo**, y
    hasta el 2026-08-17 no habia forma de verlo: `normalized_spatial_psf`
    normaliza la fuente inyectada por la suma sobre TODO el frame, mientras la
    extraccion lee el modelo en convencion NORMRAD (1 dentro de `norm_radius`).
    Asi que un modelo con mas alas inyecta menos luz en el nucleo para el mismo
    `injected_flux` nominal, y todos los metodos recuperan menos por igual.

    MEDIDO en ROXs 12 b: la mezcla por observacion pone +9.0 % mas luz fuera de
    25 px que el ajuste analitico, lo que predice -8.3 % de throughput; medido
    salio -6.0 % en los seis metodos a la vez (-5.7 a -6.6 %), sin rastro de la
    sensibilidad a la FORMA, que habria movido la apertura 3.5x mas que psffit.

    Con el default (inyectar y extraer con el mismo modelo) la etapa mide lo que
    siempre midio: cuanto pierde cada METODO dado un modelo, que es lo que E3
    necesita para calibrar. Fijando aqui una PSF de referencia comun se mide
    otra cosa —la fidelidad de EXTRACCION de cada modelo—, que es lo unico que
    permite comparar dos modelos sin que la convencion se meta por medio.
    """

    if config.get("h04_injection_psf_model") is not None:
        return config["h04_injection_psf_model"], "config.h04_injection_psf_model", True
    declared = config.get("h04_injection_psf_model_json")
    if declared is None:
        return extraction_model, extraction_source, False
    path = Path(declared)
    if not path.exists():
        raise FileNotFoundError(path)
    return read_json(path), str(path), True


def _coerce_spectrum_product(result):
    if isinstance(result, SpectrumProduct):
        extra = result.extra_columns or {}
        return {
            "wave_A": np.asarray(result.wave_A, dtype=np.float64),
            "flux": np.asarray(result.flux, dtype=np.float64),
            "flux_err": np.asarray(extra.get("flux_err_total", result.flux_err), dtype=np.float64),
            "flags": np.asarray(result.flags, dtype=np.int32),
            "continuum": np.asarray(extra["cont_runmed"], dtype=np.float64) if "cont_runmed" in extra else None,
        }
    if isinstance(result, dict):
        wave = result.get("wave_A", result.get("wavelengths_A", result.get("wavelengths")))
        err = result.get("flux_err", result.get("error", result.get("sigma")))
        if wave is None or result.get("flux") is None or err is None:
            raise ValueError("Extractor result dict must include wave_A/wavelengths, flux, and flux_err/error.")
        flux = np.asarray(result["flux"], dtype=np.float64)
        return {
            "wave_A": np.asarray(wave, dtype=np.float64),
            "flux": flux,
            "flux_err": np.asarray(err, dtype=np.float64),
            "flags": np.asarray(result.get("flags", np.zeros(flux.size, dtype=np.int32)), dtype=np.int32),
            "continuum": None if result.get("continuum") is None else np.asarray(result["continuum"], dtype=np.float64),
        }
    if isinstance(result, tuple) and len(result) >= 3:
        wave, flux, err = result[:3]
        flux = np.asarray(flux, dtype=np.float64)
        return {
            "wave_A": np.asarray(wave, dtype=np.float64),
            "flux": flux,
            "flux_err": np.asarray(err, dtype=np.float64),
            "flags": np.zeros(flux.size, dtype=np.int32),
            "continuum": None,
        }
    raise TypeError(f"Unsupported extractor result: {type(result)!r}")


def measure_recovery_with_h01_estimator(result, *, line_center_A, line_fwhm_A, continuum_window_A=80.0):
    product = _coerce_spectrum_product(result)
    flux = np.asarray(product["flux"], dtype=np.float64)
    wave = np.asarray(product["wave_A"], dtype=np.float64)
    base_good = (
        np.isfinite(wave)
        & np.isfinite(flux)
        & np.isfinite(product["flux_err"])
        & (np.asarray(product["flux_err"]) > 0)
        # Match stage_h01's detection mask (BAD_DETECTION_FLAGS = bad-window |
        # skyline), NOT `flags == 0`. FLAG_CLIPPED/FLAG_INTERPOLATED are not
        # disqualifying in E1; the optimal extractors set FLAG_CLIPPED on most
        # channels for a messy (e.g. binary-primary) halo, so `flags == 0` would
        # wrongly reject every channel and yield null throughput.
        & ((np.asarray(product["flags"], dtype=np.int32) & BAD_DETECTION_FLAGS) == 0)
    )
    if product["continuum"] is not None:
        flux = flux - np.asarray(product["continuum"], dtype=np.float64)
    else:
        # Match E1's residual construction (stage_h01 _control_residuals): subtract a
        # broad running-median continuum. The in-memory C2/C3/C4 products carry no D2
        # cont_runmed column, so without this the matched filter sees the continuum
        # pedestal (position-dependent star-halo level) as a spurious signal.
        min_pixels = max(3, min(15, int(np.count_nonzero(base_good))))
        cont = _CONTINUUM_RUNMED(wave, flux, base_good, window_A=float(continuum_window_A), min_pixels=min_pixels)
        flux = flux - cont
    good = base_good & np.isfinite(flux)
    recovered, sigma, z = H01_MATCHED_FILTER_POINT(
        product["wave_A"],
        flux,
        product["flux_err"],
        float(line_center_A),
        float(line_fwhm_A),
        good,
    )
    return {"recovered_flux": recovered, "recovered_sigma": sigma, "recovered_snr": z}


def _call_extractor(extractor, cube, wave_A, case, method, config):
    try:
        return extractor(cube=cube, wave_A=wave_A, case=case, method=method, config=config)
    except TypeError:
        try:
            return extractor(cube, wave_A, case, method, config)
        except TypeError:
            return extractor(cube, wave_A, case)


def _row_for_method(case, method, injected_flux, measurement, threshold_snr):
    recovered = float(measurement["recovered_flux"])
    sigma = float(measurement["recovered_sigma"])
    snr = float(measurement["recovered_snr"])
    throughput = np.nan if float(injected_flux) == 0 else recovered / float(injected_flux)
    bias = np.nan if float(injected_flux) == 0 else 100.0 * (recovered - float(injected_flux)) / float(injected_flux)
    return {
        "injection_id": case.injection_id,
        "variant": case.variant,
        "method": method,
        "position_label": case.position_label,
        "position_y": float(case.position_y),
        "position_x": float(case.position_x),
        "template_width": template_width_label(case.template_factor),
        "template_factor": float(case.template_factor),
        "continuum_mode": case.continuum_mode,
        "psf_fwhm_scale": float(case.psf_fwhm_scale),
        "snr": float(case.input_snr),
        "input_snr": float(case.input_snr),
        "injected_flux": float(injected_flux),
        "recovered_flux": recovered,
        "recovered_flux_baseline": 0.0,
        "recovered_flux_net": recovered,
        "recovered_sigma": sigma,
        "recovered_snr": snr,
        # La rellena `apply_snr_standardization`, que corre despues: la
        # referencia nula se construye cuando ya estan todas las filas.
        "recovered_snr_std": np.nan,
        "throughput": throughput,
        "throughput_err": 0.0,
        "complete": bool(np.isfinite(snr) and snr >= float(threshold_snr)),
        "bias_flux_pct": bias,
        "estimator": "stage_h01_detect.matched_filter_point",
    }


def _baseline_key(row):
    # `psf_fwhm_scale` NO entra en la clave: el baseline es una extraccion SIN
    # fuente inyectada, asi que el ancho de PSF de la fuente no existe en el.
    # Incluirlo hacia que las filas psf_+-10% (scale 0.9/1.1) no encontraran
    # baseline —solo se genera a scale=1.0— y se quedaran con el 0.0 por
    # defecto: su `recovered_flux_net` conservaba el pedestal entero y
    # `psf_perturbation_pct` acababa midiendo esa resta ausente, no la PSF.
    return (
        row["method"],
        row["position_label"],
        row["continuum_mode"],
        float(row["template_factor"]),
    )


def _apply_baseline_subtraction(rows):
    """Subtract the no-injection (injected_flux==0) recovered-flux baseline before
    computing throughput/bias (differential injection-recovery).

    The matched-filter ``recovered_flux`` at a position carries a pre-existing,
    position-dependent pedestal (the star-halo/continuum systematic that E1 shows
    is present at the controls too, hence the non-detection). Dividing the raw
    recovered flux by the injected flux therefore inflates the throughput above 1
    at low injected flux and can drive it negative -- it measures the pedestal, not
    the recovery of the injected line. Subtracting the per-(method, position,
    continuum_mode, template, psf_scale) baseline measured at ``injected_flux==0``
    yields a flat, physical throughput (~the true recovery fraction).
    """

    baseline = {row_key: float(row["recovered_flux"]) for row in rows
                for row_key in (_baseline_key(row),) if float(row["injected_flux"]) == 0.0}
    missing = set()
    for row in rows:
        inj = float(row["injected_flux"])
        key = _baseline_key(row)
        if key not in baseline:
            # Antes esto era un `.get(key, 0.0)` mudo, y una fila sin baseline
            # entraba en el throughput con el pedestal sin restar como si fuera
            # senal recuperada. Si vuelve a faltar, que se vea.
            missing.add(key)
        base = baseline.get(key, 0.0)
        net = float(row["recovered_flux"]) - base
        row["recovered_flux_baseline"] = base
        row["recovered_flux_net"] = net
        if inj == 0.0:
            row["throughput"] = np.nan
            row["bias_flux_pct"] = np.nan
        else:
            row["throughput"] = net / inj
            row["bias_flux_pct"] = 100.0 * (net - inj) / inj
    if missing:
        raise RuntimeError(
            f"{len(missing)} baseline key(s) have no injected_flux==0 row: "
            f"{sorted(missing)[:3]}... Throughput would silently keep the "
            "un-subtracted pedestal for those cases."
        )
    return rows


def _finite_values(rows, key):
    arr = np.asarray([row.get(key, np.nan) for row in rows], dtype=np.float64)
    return arr[np.isfinite(arr)]


def _float_or_nan(value):
    if value is None or value == "":
        return np.nan
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def _median_or_none(values):
    vals = np.asarray(values, dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return None
    return float(np.nanmedian(vals))


def _std_or_zero(values):
    vals = np.asarray(values, dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    if vals.size < 2:
        return 0.0
    return float(np.nanstd(vals, ddof=1))


def historic_regression_check(config, paths=None):
    expected = float(config.get("h04_historic_expected_snr", 8.97))
    recovered = config.get("h04_historic_recovered_snr")
    source = "config.h04_historic_recovered_snr"
    if recovered is None and paths is not None:
        truth = _read_optional_json(paths["stage06_truth_json"])
        if truth and truth.get("nominal_local_surface_matched_snr") is not None:
            recovered = truth["nominal_local_surface_matched_snr"]
            source = str(paths["stage06_truth_json"])
    # De donde sale el valor esperado. Publicar `expected_snr` a secas junto a
    # `recovered: null` se lee como si se hubiera comparado contra algo: el 8.97
    # es un default del propio modulo y su medida no esta en el repo, asi que la
    # procedencia se declara en vez de insinuarse.
    expected_source = ("config.h04_historic_expected_snr"
                       if config.get("h04_historic_expected_snr_declared")
                       else "module_default_undocumented")
    if recovered is None:
        return {
            "expected_snr": expected,
            "expected_snr_source": expected_source,
            "recovered": None,
            "tolerance_snr": float(config.get("h04_historic_tolerance_snr", 0.25)),
            "source": "unavailable",
            "verdict": "unavailable",
            "note": ("No recovered value: neither `h04_historic_recovered_snr` in the config nor "
                     "`nominal_local_surface_matched_snr` in the Stage06 truth JSON. Nothing was "
                     "compared, so this is not a failed check."),
        }
    recovered = float(recovered)
    tol = float(config.get("h04_historic_tolerance_snr", 0.25))
    return {
        "expected_snr": expected,
        "expected_snr_source": expected_source,
        "recovered": recovered,
        "tolerance_snr": tol,
        "source": source,
        "verdict": "pass" if abs(recovered - expected) <= tol else "fail",
    }


def _throughput_at_snr5(rows, methods):
    out = {}
    for method in methods:
        subset = [
            row
            for row in rows
            if row["method"] == method
            and row["variant"] == "nominal"
            and np.isclose(float(row["input_snr"]), 5.0)
            and np.isclose(float(row["template_factor"]), 1.0)
            and row["continuum_mode"] == "none"
        ]
        vals = _finite_values(subset, "throughput")
        if vals.size:
            out[method] = {"throughput": float(np.nanmedian(vals)), "err": _std_or_zero(vals), "n": int(vals.size)}
    return out


def _psf_perturbation_pct(rows, methods):
    pcts = []
    for method in methods:
        nominal = [
            row
            for row in rows
            if row["method"] == method
            and row["variant"] == "nominal"
            and row["position_label"] == "real"
            and row["continuum_mode"] == "none"
            and np.isclose(float(row["template_factor"]), 1.0)
            and float(row["input_snr"]) in {3.0, 5.0}
        ]
        for row in rows:
            if row["method"] != method or not str(row["variant"]).startswith("psf_"):
                continue
            candidates = [nom for nom in nominal if np.isclose(float(nom["input_snr"]), float(row["input_snr"]))]
            if not candidates:
                continue
            base = _median_or_none([candidate["throughput"] for candidate in candidates])
            if base is None or base == 0 or not np.isfinite(float(row["throughput"])):
                continue
            pcts.append(100.0 * abs(float(row["throughput"]) - base) / abs(base))
    return float(np.nanmedian(pcts)) if pcts else 0.0


def _bias_at_snr5(rows, methods):
    out = {}
    for method in methods:
        vals = [
            row["bias_flux_pct"]
            for row in rows
            if row["method"] == method and np.isclose(float(row["input_snr"]), 5.0) and row["variant"] == "nominal"
        ]
        med = _median_or_none(vals)
        if med is not None:
            out[method] = med
    return out


def _centroid_bias_placeholder(rows, methods):
    return {method: None for method in methods if any(row["method"] == method for row in rows)}


def _completeness(rows, methods, input_snr):
    """Fraccion de inyecciones detectadas, al nivel de SNR **inyectada** dado.

    El nivel era `h04_detection_threshold_snr`, que ya decidia `complete`: la
    misma perilla hacia de corte de deteccion y de selector del nivel al que se
    reporta. Son dos conceptos —uno es sobre la SNR recuperada, el otro sobre la
    inyectada— y confundirlos tiene una consecuencia concreta: en cuanto el
    umbral deje de ser el mismo para todos, cada metodo reportaria su
    completitud a un nivel de senal distinto y `completeness_at_5sigma` dejaria
    de ser comparable entre metodos sin que nada lo delatara.
    """

    out = {}
    for method in methods:
        subset = [
            row
            for row in rows
            if row["method"] == method
            and row["variant"] == "nominal"
            and np.isclose(float(row["template_factor"]), 1.0)
            and row["continuum_mode"] == "none"
            and np.isclose(float(row["input_snr"]), float(input_snr))
        ]
        if subset:
            out[method] = float(np.mean([bool(row["complete"]) for row in subset]))
    return out


def _v2_nulls_clean(rows, empirical_reference, *, p_null=DEFAULT_NULL_P, gate_alpha=DEFAULT_NULL_GATE_ALPHA):
    nulls = [
        row
        for row in rows
        if row["variant"] == "nominal"
        and np.isclose(float(row["input_snr"]), 0.0)
        and str(row["position_label"]).startswith("control")
    ]
    diagnostics = []
    for row in nulls:
        factor_key = f"{float(row['template_factor']):g}"
        reference = np.asarray(
            empirical_reference[row["method"]]["by_factor"][factor_key],
            dtype=np.float64,
        )
        observed = float(row["recovered_flux"])
        fap = empirical_fap(observed, reference)
        # La misma FAP de rango, por abajo. La puerta es unilateral por diseno
        # (busca falsos positivos), asi que una fila muy NEGATIVA sale con
        # `empirical_fap = 1.0` y nunca es extrema. Eso dejaba fuera de la vista
        # la sobre-sustraccion de continuo de `optimal_ls` (docs/noise_model.md
        # y el continuo negativo de C3). Se informa, no se vota.
        finite_reference = reference[np.isfinite(reference)]
        fap_low = (
            float((int(np.sum(finite_reference <= observed)) + 1) / (finite_reference.size + 1))
            if finite_reference.size and np.isfinite(observed)
            else np.nan
        )
        diagnostics.append(
            {
                "injection_id": row["injection_id"],
                "method": row["method"],
                "position_label": row["position_label"],
                "template_factor": float(row["template_factor"]),
                "continuum_mode": row["continuum_mode"],
                "recovered_flux": observed,
                "empirical_fap": _finite_or_none(fap),
                "empirical_fap_low": _finite_or_none(fap_low),
                "n_controls": int(np.count_nonzero(np.isfinite(reference))),
            }
        )
    # La unidad de la puerta es la POSICION de control, no la fila (E4 v3 §3).
    # Las filas de una posicion son la misma zona de cielo medida seis veces
    # (los metodos), con dos anchos de filtro y dos modos de continuo: no son
    # tiradas independientes, y contarlas como tales convertia UN hecho espacial
    # en tantos rechazos como filas. El resumen por posicion es la MEDIANA de
    # sus FAP —"lo que ve el metodo tipico ahi"—, que no deja que un solo
    # extractor con un defecto conocido decida por los demas ni permite que la
    # mayoria lo tape.
    rows_pval = np.asarray(
        [np.nan if row["empirical_fap"] is None else float(row["empirical_fap"]) for row in diagnostics],
        dtype=np.float64,
    )
    n_extreme_rows = int(np.count_nonzero(np.isfinite(rows_pval) & (rows_pval < float(p_null))))

    positions = []
    for label in sorted({str(row["position_label"]) for row in diagnostics}):
        mine = [row for row in diagnostics if str(row["position_label"]) == label]
        faps = np.asarray(
            [float(row["empirical_fap"]) for row in mine if row["empirical_fap"] is not None],
            dtype=np.float64,
        )
        faps = faps[np.isfinite(faps)]
        position_fap = float(np.median(faps)) if faps.size else np.nan
        by_method = {}
        for method in sorted({str(row["method"]) for row in mine}):
            per = np.asarray(
                [float(row["empirical_fap"]) for row in mine
                 if str(row["method"]) == method and row["empirical_fap"] is not None],
                dtype=np.float64,
            )
            per = per[np.isfinite(per)]
            by_method[method] = _finite_or_none(float(np.median(per)) if per.size else np.nan)
        positions.append(
            {
                "position_label": label,
                "n_rows": len(mine),
                "position_fap": _finite_or_none(position_fap),
                "extreme": bool(np.isfinite(position_fap) and position_fap < float(p_null)),
                "by_method_fap": by_method,
            }
        )

    # Cola baja: cuenta y de quien es, sin entrar en el veredicto.
    low_pval = np.asarray(
        [np.nan if row["empirical_fap_low"] is None else float(row["empirical_fap_low"])
         for row in diagnostics],
        dtype=np.float64,
    )
    low_extreme = np.isfinite(low_pval) & (low_pval < float(p_null))
    low_methods = {}
    for row, flag in zip(diagnostics, low_extreme):
        if flag:
            low_methods[str(row["method"])] = low_methods.get(str(row["method"]), 0) + 1
    low_tail = {
        "gating": False,
        "n_extreme_rows": int(np.count_nonzero(low_extreme)),
        "by_method": dict(sorted(low_methods.items())),
        "note": ("Deficit de flujo donde no se inyecto nada: sobre-sustraccion, no falso "
                 "positivo. La puerta V2 es unilateral por diseno y no lo vota."),
    }

    n_positions = len(positions)
    n_extreme = int(sum(1 for entry in positions if entry["extreme"]))
    if n_positions:
        from scipy import stats as scipy_stats

        excess_p = float(scipy_stats.binom.sf(n_extreme - 1, n_positions, float(p_null))) if n_extreme else 1.0
    else:
        excess_p = np.nan
    passed = bool(n_positions and np.isfinite(excess_p) and excess_p >= float(gate_alpha))
    science = [
        row
        for row in rows
        if row["variant"] == "nominal"
        and np.isclose(float(row["input_snr"]), 0.0)
        and row["position_label"] == "real"
    ]
    return {
        "status": "pass" if passed else "fail",
        "population": "control_positions_only",
        # `n_extreme` cuenta lo que cuenta la puerta. Hasta E4 v2 eran filas y
        # el nombre no lo decia; ahora `unit` lo declara y `n_extreme_rows`
        # conserva el numero viejo, que sigue siendo el detalle util.
        "unit": "control_position",
        "n_positions": n_positions,
        "n_extreme": n_extreme,
        "n_rows": int(rows_pval.size),
        "n_extreme_rows": n_extreme_rows,
        "p_null": float(p_null),
        "excess_p": _finite_or_none(excess_p),
        "gate_alpha": float(gate_alpha),
        "positions": positions,
        "low_tail_diagnostics": low_tail,
        "rows": diagnostics,
        "science_position_diagnostics": {
            "n_rows": len(science),
            "max_formal_snr": _finite_or_none(
                np.nanmax([float(row["recovered_snr"]) for row in science]) if science else np.nan
            ),
            "note": "The real companion position is science data and has no veto power in V2.",
        },
    }


def _v3_monotonic(rows, methods):
    failures = []
    for method in methods:
        for factor in DEFAULT_TEMPLATE_FACTORS:
            subset = [
                row
                for row in rows
                if row["method"] == method
                and row["variant"] == "nominal"
                and row["continuum_mode"] == "none"
                and np.isclose(float(row["template_factor"]), factor)
                and float(row["input_snr"]) > 0
                and np.isfinite(float(row["throughput"]))
            ]
            by_snr = {}
            for row in subset:
                by_snr.setdefault(float(row["input_snr"]), []).append(float(row["throughput"]))
            if len(by_snr) < 2:
                continue
            snrs = sorted(by_snr)
            vals = [float(np.nanmedian(by_snr[snr])) for snr in snrs]
            if np.any(np.diff(vals) < -0.15):
                failures.append({"method": method, "template_factor": factor, "snr": snrs, "throughput": vals})
    return {"status": "pass" if not failures else "fail", "failures": failures}


def _v4_hierarchy(rows):
    med = {}
    for method in DEFAULT_METHODS:
        vals = [
            row["throughput"]
            for row in rows
            if row["method"] == method
            and row["variant"] == "nominal"
            and row["continuum_mode"] == "none"
            and np.isclose(float(row["template_factor"]), 1.0)
            and float(row["input_snr"]) >= 5.0
        ]
        if vals:
            med[method] = _median_or_none(vals)
    aperture = med.get("aperture")
    failures = []
    if aperture is not None:
        for method in ("optimal_ls", "optimal_psfsub", "psffit"):
            if med.get(method) is not None and med[method] + 0.05 < aperture:
                failures.append(method)
    return {"status": "pass" if not failures else "fail", "median_throughput": med, "failures": failures}


def _v5_continuum(rows, methods, continuum_info):
    out = {}
    failures = []
    identical = []
    for method in methods:
        no_cont = [
            row["throughput"]
            for row in rows
            if row["method"] == method
            and row["variant"] == "nominal"
            and row["continuum_mode"] == "none"
            and np.isclose(float(row["input_snr"]), 5.0)
            and np.isclose(float(row["template_factor"]), 1.0)
        ]
        flat = [
            row["throughput"]
            for row in rows
            if row["method"] == method
            and row["variant"] == "nominal"
            and row["continuum_mode"] == "flat"
            and np.isclose(float(row["input_snr"]), 5.0)
            and np.isclose(float(row["template_factor"]), 1.0)
        ]
        base = _median_or_none(no_cont)
        cont = _median_or_none(flat)
        if base is None or cont is None or base == 0:
            continue
        degradation = 100.0 * (base - cont) / abs(base)
        out[method] = degradation
        if len(no_cont) == len(flat) and np.allclose(no_cont, flat, rtol=0.0, atol=0.0, equal_nan=True):
            identical.append(method)
        if degradation > 20.0:
            failures.append(method)
    continuum_positive = bool(float((continuum_info or {}).get("value", 0.0)) > 0)
    ok = continuum_positive and not failures and not identical
    return {
        "status": "pass" if ok else "fail",
        "continuum_positive": continuum_positive,
        "flat_distinct_from_none": not identical,
        "identical_methods": identical,
        "degradation_pct": out,
        "failures": failures,
    }


def _qc_from_rows(config, paths, rows, methods, cases, budget, regression, continuum_info, null_reference):
    threshold = float(config.get("h04_detection_threshold_snr", 5.0))
    standardization = str(config.get("h04_snr_standardization", "none"))
    completeness_input_snr = float(
        config.get("h04_completeness_input_snr", DEFAULT_COMPLETENESS_INPUT_SNR)
    )
    psf_pct = _psf_perturbation_pct(rows, methods)
    checks = {
        "v1_regression": {"status": regression["verdict"], **regression},
        "v2_nulls_clean": _v2_nulls_clean(
            rows,
            null_reference,
            p_null=float(config.get("h04_null_p", DEFAULT_NULL_P)),
            gate_alpha=float(config.get("h04_null_gate_alpha", DEFAULT_NULL_GATE_ALPHA)),
        ),
        "v3_monotonic": _v3_monotonic(rows, methods),
        "v4_hierarchy": _v4_hierarchy(rows),
        "v5_continuum": _v5_continuum(rows, methods, continuum_info),
    }
    open_issues = []
    # "No se evaluo" y "se evaluo y fallo" no son lo mismo, y hasta aqui salian
    # con el mismo texto: `!= "pass"` metia las dos en el mismo saco y el QC
    # declaraba "did not pass ... not valid for E3" de una comprobacion que
    # nunca llego a correr. Decision del usuario (2026-08-20): `unavailable` se
    # ACEPTA como tal, con la limitacion declarada, y el texto lo dice.
    if regression["verdict"] == "unavailable":
        open_issues.append(
            "Historic Stage06 regression was never evaluated (no reference measurement "
            "available): its verdict is 'unavailable', not a failure. The throughput is "
            "delivered with that limitation declared; E3 records the same caveat."
        )
    elif regression["verdict"] != "pass":
        open_issues.append("Historic Stage06 regression did not pass; H04 is not valid for E3.")
    for key, check in checks.items():
        if check.get("status") == "fail":
            open_issues.append(f"{key} failed.")
    return {
        "stage": "h04_injection_recovery",
        "spec_version": (
            SPEC_VERSION_STANDARDIZED_SNR if standardization != "none" else SPEC_VERSION
        ),
        "run_id_base": str(config["run_id"]),
        "status": "complete",
        "clones_created": [],
        "regression_historic": regression,
        "grid": {
            "n_injections": int(
                len(
                    [
                        case
                        for case in cases
                        if case.variant == "nominal"
                    ]
                )
            ),
            "n_psf_perturbation": int(len([case for case in cases if case.variant != "nominal"])),
            "methods": list(methods),
        },
        "runtime_budget": budget,
        "psf_provenance": config.get("h04_psf_provenance"),
        "continuum_injection": continuum_info,
        "empirical_null_reference": {
            method: {
                "n_controls": int(reference["n_controls"]),
                "scale_source": reference.get("scale_source", "configured_test_reference"),
                "product": reference.get("product"),
                "controls": reference.get("controls"),
            }
            for method, reference in null_reference.items()
        },
        "throughput": {
            "per_method_at_snr5": _throughput_at_snr5(rows, methods),
            "psf_perturbation_pct": psf_pct,
        },
        "bias": {
            "flux_pct_at_snr5": _bias_at_snr5(rows, methods),
            "centroid_A": _centroid_bias_placeholder(rows, methods),
            "fwhm_pct": {method: None for method in methods},
        },
        # La escala en la que se decide `complete`, y la prueba de si hacia
        # falta: `null_distribution` dice, con senal nula, si la SNR de cada
        # metodo tiene media 0 y sigma 1 como supone un umbral unico.
        "snr_standardization": {
            "mode": standardization,
            "column": (
                "recovered_snr_std" if standardization == "injection_nulls" else "recovered_snr"
            ),
            "threshold_snr": threshold,
            "reference": "injection_nulls_control_positions",
            "leave_one_out": True,
            "scale_by_stratum": {
                "|".join(key): {
                    field: _finite_or_none(value) if field != "n" else value
                    for field, value in null_scale([flux for _id, flux in pairs]).items()
                }
                for key, pairs in sorted(injection_null_reference(rows).items())
            },
            # Las dos nulas que E4 tiene a mano NO son la misma poblacion, y la
            # diferencia importa mas alla de aqui: la puerta V2 compara estas
            # mismas filas nulas contra la de PRODUCCION. Publicar el cociente de
            # escalas es lo que delata el desajuste sin tener que ir a buscarlo.
            "reference_population_check": _reference_population_check(rows, null_reference),
            "null_distribution": null_distribution_diagnostics(rows, methods),
        },
        # El nombre del campo es historico: el `5sigma` era el umbral, y el nivel
        # al que se mide es `completeness_input_snr`. Se declaran los dos porque
        # hasta hoy eran el mismo numero por accidente.
        "completeness_at_5sigma": _completeness(rows, methods, completeness_input_snr),
        "completeness_input_snr": completeness_input_snr,
        "completeness_threshold_snr": threshold,
        "checks": checks,
        "open_issues": open_issues,
    }


def _blocked_product(config, paths, positions, cases, methods, budget, reason):
    regression = historic_regression_check(config, paths)
    qc = {
        "stage": "h04_injection_recovery",
        "spec_version": SPEC_VERSION,
        "run_id_base": str(config["run_id"]),
        "status": "blocked",
        "blocked_reason": reason,
        "clones_created": [],
        "regression_historic": regression,
        "grid": {
            "n_injections": int(len([case for case in cases if case.variant == "nominal"])),
            "n_psf_perturbation": int(len([case for case in cases if case.variant != "nominal"])),
            "methods": list(methods),
            "positions": positions,
        },
        "runtime_budget": budget,
        "throughput": {"per_method_at_snr5": {}, "psf_perturbation_pct": 0.0},
        "bias": {"flux_pct_at_snr5": {}, "centroid_A": {}, "fwhm_pct": {}},
        "completeness_at_5sigma": {},
        "checks": {"v1_regression": {"status": regression["verdict"], **regression}},
        "open_issues": [reason],
    }
    return StageH04Product(rows=[], qc=_json_ready(qc))


# Set on the parent immediately before forking the E4 process pool so each
# worker inherits it copy-on-write (never pickled). Cleared after the pool exits.
_FORK_CTX = None


def _compute_case_rows(case, ctx):
    """Inject one case into a private cube copy and extract every method.

    Pure function of ``(case, ctx)`` where ``ctx`` bundles the shared, read-only
    state: the case injects into its OWN copy (``inject(..., copy=True)``) and
    only reads shared read-only objects (base cube, psf model, extractor
    closures), so the returned rows are independent of execution order. The
    serial, threaded and forked backends therefore produce identical rows.
    """
    sigma_flux = ctx["sigma_flux"]
    line_center = ctx["line_center"]
    lsf_fwhm = ctx["lsf_fwhm"]
    methods = ctx["methods"]
    extractors = ctx["extractors"]
    cfg = ctx["config"]

    injected_flux = float(case.input_snr) * sigma_flux
    continuum = ctx["continuum_flux_density"] if case.continuum_mode == "flat" else 0.0
    source = InjectionSource(
        y=case.position_y,
        x=case.position_x,
        total_line_flux=injected_flux,
        line_center_A=line_center,
        line_fwhm_A=lsf_fwhm * float(case.template_factor),
        label=case.injection_id,
        continuum_flux_density=continuum,
        psf_fwhm_scale=case.psf_fwhm_scale,
    )
    cube_injected = inject(
        ctx["base_cube"], [source],
        wavelengths_A=ctx["wavelengths_A"], psf_model=ctx["injection_psf_model"], copy=True,
        norm_convention=ctx["injection_norm_convention"],
    )
    case_rows = []
    for method in methods:
        extractor = extractors[method] if isinstance(extractors, dict) else extractors
        result = _call_extractor(extractor, cube_injected, ctx["wavelengths_A"], case, method, cfg)
        measurement = measure_recovery_with_h01_estimator(
            result,
            line_center_A=line_center,
            line_fwhm_A=lsf_fwhm * float(case.template_factor),
            continuum_window_A=ctx["continuum_window_A"],
        )
        case_rows.append(_row_for_method(case, method, injected_flux, measurement, ctx["threshold"]))
    return case_rows


def _fork_worker(case):
    """Process-pool entry point: read the fork-inherited context and run one case."""
    ctx = _FORK_CTX
    if ctx is None:
        raise RuntimeError("H04 process-pool worker started without inherited context.")
    return _compute_case_rows(case, ctx)


def _map_cases_forked(cases, ctx, n_workers):
    """Map cases over a fork ProcessPool; ``pool.map`` restores input order."""
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor

    global _FORK_CTX
    fork_context = mp.get_context("fork")
    _FORK_CTX = ctx  # forked children inherit this read-only snapshot (COW)
    try:
        with ProcessPoolExecutor(max_workers=int(n_workers), mp_context=fork_context) as pool:
            return list(pool.map(_fork_worker, cases))
    finally:
        _FORK_CTX = None


def _run_case_grid(cases, ctx, config):
    """Compute all per-case rows, restoring input order across three backends.

    All three are numerically identical because each case is an independent pure
    function of ``(case, ctx)``:
      * serial (default);
      * ThreadPool (``h04_n_jobs``) — GIL-capped ~1.6x on the psffit design build;
      * fork ProcessPool (``h04_process_pool``) — the ~3GB base cube and the
        unpicklable extractor closures are inherited copy-on-write.
    ``h04_process_pool`` takes precedence when set and fork is available; if a
    process pool is requested where fork is unavailable, fall back to a
    same-width ThreadPool rather than silently dropping to serial.
    """
    process_pool = _resolve_h04_process_pool(config, len(cases))
    if process_pool >= 2 and _fork_supported():
        per_case = _map_cases_forked(cases, ctx, process_pool)
        return [row for case_rows in per_case for row in case_rows]

    n_jobs = _resolve_h04_n_jobs(config, len(cases))
    if process_pool >= 2:
        n_jobs = max(n_jobs, process_pool)
    if n_jobs == 1:
        return [row for case in cases for row in _compute_case_rows(case, ctx)]

    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=n_jobs) as pool:
        per_case = list(pool.map(lambda case: _compute_case_rows(case, ctx), cases))
    return [row for case_rows in per_case for row in case_rows]


def compute_stage_h04_products(config, paths=None, *, extractors=None, base_cube=None, wavelengths_A=None, psf_model=None):
    cfg = dict(config)
    root = Path(cfg.get("project_root") or Path.cwd()).resolve()
    paths = stage_h04_paths(cfg["run_id"], root) if paths is None else paths
    methods = [str(method) for method in cfg.get("h04_methods", DEFAULT_METHODS)]
    positions = resolve_h04_positions(cfg, paths)
    cases = build_h04_cases(cfg, positions)
    # El presupuesto se calcula con el backend que se va a usar de verdad: la
    # version anterior suponia ejecucion en serie y sobrestimaba ~30x.
    _pool = _resolve_h04_process_pool(cfg, len(cases))
    _forked = bool(_pool >= 2 and _fork_supported())
    _workers = _pool if _forked else max(_resolve_h04_n_jobs(cfg, len(cases)), _pool)
    budget = estimate_runtime_budget(
        cfg,
        len([case for case in cases if case.variant == "nominal"]),
        len(methods),
        n_workers=_workers,
        forked=_forked,
    )
    if budget["requires_checkpoint"] and extractors is None:
        return _blocked_product(
            cfg,
            paths,
            positions,
            cases,
            methods,
            budget,
            "Estimated H04 runtime exceeds the checkpoint threshold; set h04_allow_long_run=true after review.",
        )
    if extractors is None:
        return _blocked_product(
            cfg,
            paths,
            positions,
            cases,
            methods,
            budget,
            "No H04 extractor runner was supplied; base run was not modified.",
        )
    if base_cube is None or wavelengths_A is None:
        base_cube, wavelengths_A, _cube_path = _load_stage02_cube(paths, cfg)
    if psf_model is None:
        psf_model, _psf_source = _load_psf_model(paths, cfg)
    else:
        _psf_source = "caller"
    injection_psf_model, injection_psf_source, injection_psf_differs = _load_injection_psf_model(
        paths, cfg, psf_model, _psf_source
    )
    # El throughput es dependiente del modelo, asi que de que PSF salio cada
    # mitad del experimento no puede quedar implicito: viaja al QC.
    cfg["h04_psf_provenance"] = {
        "extraction_psf": str(_psf_source),
        "extraction_psf_form": str((psf_model or {}).get("form", "")),
        "injection_psf": str(injection_psf_source),
        "injection_psf_form": str((injection_psf_model or {}).get("form", "")),
        "injection_psf_differs": bool(injection_psf_differs),
        "injection_norm_convention": str(cfg.get("h04_injection_norm_convention", "norm_radius")),
        "note": (
            "`injection_norm_convention=norm_radius` pone `injected_flux` en la MISMA "
            "convencion que devuelven los extractores (NORMRAD), y con eso el throughput "
            "deja de depender del modelo de PSF por construccion. Con `frame` (el camino "
            "historico) no: la inyeccion normalizaba por la suma sobre todo el recorte, asi "
            "que un modelo con mas alas inyectaba menos nucleo para el mismo `injected_flux` "
            "nominal, y en ROXs 12 b cambiar el modelo de C1 movio los SEIS metodos -6 % a la "
            "vez. `injection_psf` distinta de `extraction_psf` aisla la fidelidad de "
            "extraccion; con la misma en los dos lados se mide lo que pierde cada METODO."
        ),
    }

    line_center = _line_center_from_config(cfg)
    lsf_fwhm, _lsf_source = _lsf_fwhm_from_config_or_qc(cfg, paths)
    sigma_flux = _injection_sigma(cfg)
    continuum_info = resolve_continuum_injection(cfg, paths)
    continuum_flux_density = float(continuum_info["value"])
    threshold = float(cfg.get("h04_detection_threshold_snr", 5.0))
    continuum_window_A = float(cfg.get("h04_continuum_window_A", cfg.get("h01_continuum_window_A", 80.0)))

    # Shared read-only state for the per-case worker. Bundling it lets serial,
    # threaded and forked backends call the SAME module-level _compute_case_rows,
    # which is what guarantees identical rows regardless of execution order.
    ctx = {
        "base_cube": base_cube,
        "wavelengths_A": wavelengths_A,
        "psf_model": psf_model,
        "injection_psf_model": injection_psf_model,
        "injection_norm_convention": str(cfg.get("h04_injection_norm_convention", "norm_radius")),
        "extractors": extractors,
        "methods": methods,
        "config": cfg,
        "line_center": line_center,
        "lsf_fwhm": lsf_fwhm,
        "sigma_flux": sigma_flux,
        "continuum_flux_density": continuum_flux_density,
        "threshold": threshold,
        "continuum_window_A": continuum_window_A,
    }
    _t0 = time.perf_counter()
    rows = _run_case_grid(cases, ctx, cfg)
    budget = close_runtime_budget(budget, time.perf_counter() - _t0)
    if bool(cfg.get("h04_baseline_subtract_throughput", True)):
        rows = _apply_baseline_subtraction(rows)

    regression = historic_regression_check(cfg, paths)
    if bool(cfg.get("h04_require_historic_regression", True)) and regression["verdict"] != "pass":
        raise RuntimeError("H04 historic regression V1 must pass before publishing throughput.")
    configured_null = cfg.get("h04_empirical_null_reference")
    if configured_null is None:
        null_reference = build_empirical_null_reference(cfg, paths, methods, lsf_fwhm_A=lsf_fwhm)
    else:
        null_reference = {
            method: {
                **reference,
                "by_factor": {
                    str(factor): np.asarray(values, dtype=np.float64)
                    for factor, values in reference["by_factor"].items()
                },
            }
            for method, reference in configured_null.items()
        }
    rows = apply_snr_standardization(
        rows,
        mode=str(cfg.get("h04_snr_standardization", "none")),
        threshold_snr=float(cfg.get("h04_detection_threshold_snr", 5.0)),
    )
    qc = _qc_from_rows(
        cfg,
        paths,
        rows,
        methods,
        cases,
        budget,
        regression,
        continuum_info,
        null_reference,
    )
    return StageH04Product(rows=_json_ready(rows), qc=_json_ready(qc))


def clone_run_for_case(config, case, *, project_root=None):
    root = Path(project_root or config.get("project_root") or Path.cwd()).resolve()
    base_paths = RunPaths.from_project_root(config["run_id"], root)
    clone_id = f"{config['run_id']}_H04Inject_{case.injection_id}"
    clone_paths = RunPaths.from_project_root(clone_id, root)
    before = tree_sha256(base_paths.run_dir)
    create_run_clone(base_paths.run_dir, clone_paths.run_dir)
    after = tree_sha256(base_paths.run_dir)
    if before != after:
        raise RuntimeError("Base run hash changed while creating H04 clone.")
    return {"clone_id": clone_id, "clone_dir": clone_paths.run_dir, "base_hash": before}


def _plot_throughput(rows, path):
    cache_dir = Path(os.environ.get("TMPDIR", "/tmp")) / "musepipe_matplotlib"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.5, 4), constrained_layout=True)
    nominal = [row for row in rows if row.get("variant") == "nominal" and row.get("continuum_mode") == "none"]
    if nominal:
        for method in sorted({row["method"] for row in nominal}):
            subset = [row for row in nominal if row["method"] == method and np.isclose(float(row["template_factor"]), 1.0)]
            by_snr = {}
            for row in subset:
                throughput = _float_or_nan(row.get("throughput"))
                if np.isfinite(throughput):
                    by_snr.setdefault(float(row["input_snr"]), []).append(throughput)
            xs = sorted(by_snr)
            ys = [float(np.nanmedian(by_snr[x])) for x in xs]
            if xs:
                ax.plot(xs, ys, marker="o", label=method)
        ax.set_xlabel("Injected matched-filter S/N")
        ax.set_ylabel("Throughput")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.25)
    else:
        ax.text(0.5, 0.5, "H04 not executed", ha="center", va="center")
        ax.set_axis_off()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def _plot_recovered(rows, path):
    cache_dir = Path(os.environ.get("TMPDIR", "/tmp")) / "musepipe_matplotlib"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5.5, 5), constrained_layout=True)
    finite = [
        row
        for row in rows
        if row.get("variant") == "nominal"
        and _float_or_nan(row.get("injected_flux")) > 0
        and np.isfinite(_float_or_nan(row.get("recovered_flux")))
    ]
    if finite:
        xs = np.asarray([_float_or_nan(row["injected_flux"]) for row in finite], dtype=np.float64)
        ys = np.asarray([_float_or_nan(row["recovered_flux"]) for row in finite], dtype=np.float64)
        ax.scatter(xs, ys, s=18, alpha=0.7)
        lo = min(float(np.nanmin(xs)), float(np.nanmin(ys)))
        hi = max(float(np.nanmax(xs)), float(np.nanmax(ys)))
        ax.plot([lo, hi], [lo, hi], color="0.3", ls="--", lw=1)
        ax.set_xlabel("Injected flux")
        ax.set_ylabel("Recovered flux")
    else:
        ax.text(0.5, 0.5, "H04 not executed", ha="center", va="center")
        ax.set_axis_off()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def _plot_completeness(rows, path):
    cache_dir = Path(os.environ.get("TMPDIR", "/tmp")) / "musepipe_matplotlib"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.5, 4), constrained_layout=True)
    nominal = [row for row in rows if row.get("variant") == "nominal" and row.get("continuum_mode") == "none"]
    if nominal:
        for method in sorted({row["method"] for row in nominal}):
            subset = [row for row in nominal if row["method"] == method and np.isclose(float(row["template_factor"]), 1.0)]
            by_snr = {}
            for row in subset:
                by_snr.setdefault(float(row["input_snr"]), []).append(bool(row["complete"]))
            xs = sorted(by_snr)
            ys = [float(np.mean(by_snr[x])) for x in xs]
            if xs:
                ax.plot(xs, ys, marker="o", label=method)
        ax.set_xlabel("Injected matched-filter S/N")
        ax.set_ylabel("Completeness")
        ax.set_ylim(-0.05, 1.05)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.25)
    else:
        ax.text(0.5, 0.5, "H04 not executed", ha="center", va="center")
        ax.set_axis_off()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def decision_column(config):
    """La columna sobre la que se decide `complete`, segun la escala declarada."""

    mode = str(config.get("h04_snr_standardization", "none"))
    if mode not in SNR_STANDARDIZATION_MODES:
        raise ValueError(
            f"`h04_snr_standardization` solo entiende {SNR_STANDARDIZATION_MODES}, no {mode!r}."
        )
    return "recovered_snr" if mode == "none" else "recovered_snr_std"


def finalize_stage_h04(run_id=None, *, project_root=None, overrides=None,
                       allow_run_id_mismatch=False):
    """Recalcula lo **derivado** de la tabla que ya esta en disco, sin re-inyectar.

    `complete` y `completeness_at_5sigma` son funciones puras de una columna de
    SNR —que la tabla ya trae— y del umbral. Cambiar el umbral no necesita
    repetir la rejilla de inyecciones, que cuesta horas: es el mismo caso que el
    `finalize` de A4, una pasada producto->producto que recomputa lo derivado y
    deja lo medido intacto.

    Reescribe **solo** la columna `complete`: las demas se copian tal cual desde
    el CSV, sin reformatear, para que un `diff` muestre exactamente lo que
    cambio y nada mas.
    """

    run_config = load_run_config(
        run_id, project_root=project_root, allow_run_id_mismatch=allow_run_id_mismatch
    )
    cfg = stage_h04_config_from_run(
        run_config.run_id,
        project_root=run_config.paths.project_root,
        overrides=overrides,
        allow_run_id_mismatch=allow_run_id_mismatch,
    )
    paths = stage_h04_paths(run_config.run_id, cfg["project_root"])
    table_path = paths["throughput_csv"]
    qc_path = paths["stage_h04_qc_json"]
    if not table_path.exists() or not qc_path.exists():
        raise FileNotFoundError(
            f"`finalize` recomputa lo derivado de productos existentes; faltan {table_path} o {qc_path}."
        )

    threshold = float(cfg.get("h04_detection_threshold_snr", 5.0))
    column = decision_column(cfg)
    with open(table_path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    if column not in fieldnames:
        # Un producto anterior a E4 v4 no trae la columna estandarizada. Parar es
        # lo correcto: decidir con la otra seria cambiar la escala en silencio.
        raise RuntimeError(
            f"La tabla no trae `{column}`, asi que es anterior a la escala declarada "
            f"(`h04_snr_standardization={cfg.get('h04_snr_standardization')}`). Re-corre E4."
        )

    changed = 0
    for row in rows:
        try:
            value = float(row[column])
        except (TypeError, ValueError):
            value = float("nan")
        decision = "True" if math.isfinite(value) and value >= threshold else "False"
        if row["complete"] != decision:
            changed += 1
        row["complete"] = decision
    with open(table_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    typed = [
        {
            **row,
            "input_snr": float(row["input_snr"]),
            "template_factor": float(row["template_factor"]),
            "complete": row["complete"] == "True",
        }
        for row in rows
    ]
    methods = [str(m) for m in cfg.get("h04_methods", DEFAULT_METHODS)]
    completeness_input_snr = float(
        cfg.get("h04_completeness_input_snr", DEFAULT_COMPLETENESS_INPUT_SNR)
    )
    qc = read_json(qc_path)
    qc["completeness_at_5sigma"] = _completeness(typed, methods, completeness_input_snr)
    qc["completeness_input_snr"] = completeness_input_snr
    qc["completeness_threshold_snr"] = threshold
    if isinstance(qc.get("snr_standardization"), dict):
        qc["snr_standardization"]["threshold_snr"] = threshold
        qc["snr_standardization"]["column"] = column
    # Procedencia: un producto que ya no sale de una sola corrida tiene que decir
    # que parte se recomputo despues y con que umbral.
    qc["derived_finalize"] = {
        "recomputed_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "threshold_snr": threshold,
        "column": column,
        "rows_changed": changed,
        "n_rows": len(rows),
        "note": ("`complete` y `completeness_at_5sigma` se recomputaron de la tabla existente; "
                 "las medidas (flujo, sigma, SNR, throughput) no se tocaron."),
    }
    write_json(qc_path, _json_ready(qc))
    return {"table": table_path, "qc_json": qc_path, "rows_changed": changed,
            "threshold_snr": threshold, "column": column}


def write_stage_h04_products(product: StageH04Product, config, paths):
    paths["paths"].ensure_base_dirs()
    paths["plot_dir"].mkdir(parents=True, exist_ok=True)
    write_csv(paths["throughput_csv"], product.rows, fieldnames=TABLE_FIELDS)
    throughput_plot = _plot_throughput(product.rows, paths["throughput_plot"])
    recovery_plot = _plot_recovered(product.rows, paths["recovery_plot"])
    completeness_plot = _plot_completeness(product.rows, paths["completeness_plot"])
    qc = dict(product.qc)
    qc["tables"] = {"throughput_by_method": str(paths["throughput_csv"])}
    qc["figures"] = {
        "throughput": str(throughput_plot),
        "recovered_vs_injected": str(recovery_plot),
        "completeness": str(completeness_plot),
    }
    write_json(paths["stage_h04_qc_json"], _json_ready(qc))
    return {"table": paths["throughput_csv"], "qc_json": paths["stage_h04_qc_json"], "qc": qc}


def _derive_injection_sigma(cfg, paths, extractors, base_cube, wave_A, psf_model,
                            injection_psf_model=None):
    """Self-consistent flux scale so input_snr == the reference method's S/N.

    The injected source is a broad AO PSF; a given aperture/optimal/psffit
    estimator only recovers a fraction of the total line flux, so the input S/N
    cannot be read off the total flux directly (spec E4 §4.1). We calibrate:
    inject a bright test source of total line flux F_cal at the real position,
    measure the reference method's recovered flux R and noise sigma, and set
    ``sigma_flux = sigma * F_cal / R``. Then ``input_snr * sigma_flux`` is the
    total flux whose reference-method matched-filter S/N equals ``input_snr``.
    Also returns the reference throughput R/F_cal for logging.
    """
    method = str(cfg.get("h04_sigma_reference_method", "psffit"))
    extractor = extractors[method] if isinstance(extractors, dict) else extractors
    line_center = _line_center_from_config(cfg)
    lsf_fwhm, _ = _lsf_fwhm_from_config_or_qc(cfg, paths)
    real = cfg["h04_real_position_yx"]

    # Baseline noise scale to size the calibration injection well above noise.
    baseline = _ZeroCase(position_y=float(real[0]), position_x=float(real[1]))
    base_meas = measure_recovery_with_h01_estimator(
        _call_extractor(extractor, base_cube, wave_A, baseline, method, cfg),
        line_center_A=line_center, line_fwhm_A=lsf_fwhm,
        continuum_window_A=float(cfg.get("h04_continuum_window_A", cfg.get("h01_continuum_window_A", 80.0))),
    )
    noise0 = float(base_meas["recovered_sigma"])
    if not np.isfinite(noise0) or noise0 <= 0:
        raise RuntimeError("Could not size the H04 calibration injection (baseline noise invalid).")

    f_cal = float(cfg.get("h04_calibration_snr", 100.0)) * noise0
    src = InjectionSource(
        y=float(real[0]), x=float(real[1]), total_line_flux=f_cal,
        line_center_A=line_center, line_fwhm_A=lsf_fwhm, label="h04_calib",
        continuum_flux_density=0.0, psf_fwhm_scale=1.0,
    )
    # La calibracion de sigma tiene que inyectar con la MISMA PSF que los casos,
    # o el `injected_flux` de referencia y el de la rejilla no son la misma escala.
    cube_cal = inject(base_cube, [src], wavelengths_A=wave_A,
                      psf_model=(psf_model if injection_psf_model is None else injection_psf_model),
                      copy=True,
                      norm_convention=str(cfg.get("h04_injection_norm_convention", "norm_radius")))
    cal_meas = measure_recovery_with_h01_estimator(
        _call_extractor(extractor, cube_cal, wave_A, baseline, method, cfg),
        line_center_A=line_center, line_fwhm_A=lsf_fwhm,
        continuum_window_A=float(cfg.get("h04_continuum_window_A", cfg.get("h01_continuum_window_A", 80.0))),
    )
    recovered = float(cal_meas["recovered_flux"])
    sigma_cal = float(cal_meas["recovered_sigma"])
    if not np.isfinite(recovered) or recovered <= 0 or not np.isfinite(sigma_cal) or sigma_cal <= 0:
        raise RuntimeError("H04 calibration injection did not recover positive flux; cannot set sigma_flux.")
    throughput_ref = recovered / f_cal
    sigma_flux = sigma_cal * f_cal / recovered
    return sigma_flux, {"reference_method": method, "throughput_ref": throughput_ref,
                        "f_cal": f_cal, "recovered_cal": recovered, "sigma_cal": sigma_cal}


class _ZeroCase:
    """Minimal case for a zero-injection baseline extraction."""

    def __init__(self, position_y, position_x):
        self.injection_id = "baseline"
        self.variant = "nominal"
        self.template_factor = 1.0
        self.psf_fwhm_scale = 1.0
        self.continuum_mode = "none"
        self.input_snr = 0.0
        self.position_y = float(position_y)
        self.position_x = float(position_x)
        self.position_label = "real"


def run_stage_h04(run_id=None, *, project_root=None, overrides=None, allow_run_id_mismatch=False, extractors=None):
    cfg = stage_h04_config_from_run(
        run_id,
        project_root=project_root,
        overrides=overrides,
        allow_run_id_mismatch=allow_run_id_mismatch,
    )
    paths = stage_h04_paths(cfg["run_id"], project_root=cfg.get("project_root"))

    base_cube = wave_A = psf_model = None
    if extractors is None and bool(cfg.get("h04_use_production_extractors", True)):
        from .stage_h04_extractors import build_production_extractors

        base_cube, wave_A, _cube_path = _load_stage02_cube(paths, cfg)
        base_cube = np.asarray(base_cube, dtype=np.float64)
        if base_cube.ndim == 4:
            if base_cube.shape[0] != 1:
                raise RuntimeError(
                    f"H04 in-memory extractors support a single cube; got {base_cube.shape[0]} "
                    "stacked cubes. Combine cubes upstream or supply extractors explicitly."
                )
            base_cube = base_cube[0]
        psf_model, _psf_source = _load_psf_model(paths, cfg)
        injection_psf_model, _inj_source, _inj_differs = _load_injection_psf_model(
            paths, cfg, psf_model, _psf_source
        )
        extractors = build_production_extractors(
            cfg, paths, wave_A=wave_A, psf_model=psf_model, base_cube=base_cube
        )
        if cfg.get("h04_injection_flux_sigma") is None:
            positions = resolve_h04_positions(cfg, paths)
            sigma_flux, calib_info = _derive_injection_sigma(
                {**cfg, "h04_real_position_yx": [positions[0]["y"], positions[0]["x"]]},
                paths, extractors, base_cube, wave_A, psf_model,
                injection_psf_model=injection_psf_model,
            )
            cfg["h04_injection_flux_sigma"] = sigma_flux
            cfg["h04_sigma_calibration"] = calib_info

    product = compute_stage_h04_products(
        cfg, paths, extractors=extractors, base_cube=base_cube,
        wavelengths_A=wave_A, psf_model=psf_model,
    )
    written = write_stage_h04_products(product, cfg, paths)
    return {"config": cfg, "paths": paths, "qc": written["qc"], "written": written}


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="stage_h04_injection.py",
        description="Run Stage H04/E4 injection-recovery throughput calibration.",
    )
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--project-root", default=None)
    parser.add_argument("--allow-run-id-mismatch", action="store_true")
    parser.add_argument("--allow-long-run", action="store_true")
    parser.add_argument(
        "--finalize", action="store_true",
        help=("No inyecta nada: recomputa lo DERIVADO (`complete` y `completeness_at_5sigma`) "
              "de la tabla que ya esta en disco, con el umbral que declare la config. Es lo que "
              "hay que correr tras cambiar `h04_detection_threshold_snr`, en vez de repetir la "
              "rejilla entera."),
    )
    parser.add_argument(
        "--injection-psf-model-json", default=None,
        help=("PSF con la que INYECTAR, distinta de la que extrae. Por defecto la misma, "
              "y entonces el throughput mide lo que pierde cada metodo. Fijando aqui una "
              "PSF de referencia comun se mide la fidelidad de EXTRACCION, que es lo unico "
              "comparable entre dos modelos."),
    )
    args = parser.parse_args(argv)
    if args.finalize:
        result = finalize_stage_h04(
            args.run_id,
            project_root=args.project_root,
            allow_run_id_mismatch=args.allow_run_id_mismatch,
        )
        print(f"{result['qc_json']}  ({result['rows_changed']} filas cambiadas, "
              f"umbral {result['threshold_snr']:g} sobre {result['column']})")
        return 0
    overrides = {}
    if args.allow_long_run:
        overrides["h04_allow_long_run"] = True
    if args.injection_psf_model_json is not None:
        overrides["h04_injection_psf_model_json"] = args.injection_psf_model_json
    overrides = overrides or None
    result = run_stage_h04(
        args.run_id,
        project_root=args.project_root,
        overrides=overrides,
        allow_run_id_mismatch=args.allow_run_id_mismatch,
    )
    print(result["paths"]["stage_h04_qc_json"])


__all__ = [
    "DEFAULT_METHODS",
    "DEFAULT_SNR_GRID",
    "SPEC_VERSION",
    "SPEC_VERSION_STANDARDIZED_SNR",
    "SNR_STANDARDIZATION_MODES",
    "H01_MATCHED_FILTER_POINT",
    "H04Case",
    "StageH04Product",
    "build_h04_cases",
    "clone_run_for_case",
    "compute_stage_h04_products",
    "apply_snr_standardization",
    "build_empirical_null_reference",
    "injection_null_reference",
    "null_distribution_diagnostics",
    "null_scale",
    "close_runtime_budget",
    "decision_column",
    "estimate_runtime_budget",
    "finalize_stage_h04",
    "historic_regression_check",
    "measure_recovery_with_h01_estimator",
    "resolve_h04_positions",
    "resolve_continuum_injection",
    "run_stage_h04",
    "stage_h04_config_from_run",
    "stage_h04_paths",
    "write_stage_h04_products",
]


if __name__ == "__main__":
    main()
