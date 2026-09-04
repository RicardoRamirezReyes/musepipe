"""Stage H03/E3: Halpha upper limits from E1 controls and E4 throughput."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import math
import os
from pathlib import Path

import numpy as np

from ..config import load_run_config
from ..extraction.product import SpectrumProduct
from ..io import cube_bunit, flux_unit_cgs as _flux_unit_cgs, read_json, write_csv, write_json
from ..paths import RunPaths
from .stage_h01_detect import (
    DEFAULT_TEMPLATE_WIDTH_FACTORS,
    HALPHA_REST_A,
    _good_detection_mask,
    expected_line_center_A,
    matched_filter_point,
)
from .stage_x10_compare import METHOD_ORDER


PC_CM = 3.0856775814913673e18
L_SUN_ERG_S = 3.828e33
M_SUN_G = 1.98847e33
R_SUN_CM = 6.957e10
G_CGS = 6.67430e-8
YR_S = 31557600.0
EULER_GAMMA = 0.5772156649015329
ONE_SIDED_5SIGMA_FAP = 2.866515718791933e-7


TABLE_FIELDS = [
    "row_kind",
    "method",
    "canonical_method",
    "template_width",
    "template_factor",
    "tail_model",
    "n_controls",
    "minimum_resolvable_fap",
    "q99_resolvable",
    "z_99_empirical",
    "z_99_gumbel",
    "tail_estimator",
    "z_threshold",
    "z_5sigma_extrap",
    "tail_extrapolated_5sigma",
    "matched_sigma",
    "sigma_source",
    "sigma_h01_table",
    "sigma_table_vs_recomputed_pct",
    "f_stat_99",
    "f_stat_5sigma_extrap",
    "throughput",
    "throughput_err",
    "snr_for_throughput",
    "throughput_source",
    "f_lim_observed",
    "f_lim_observed_err",
    "f_lim_observed_5sigma_extrap",
    "a_halpha_over_av",
    "av",
    "av_err",
    "f_lim_dereddened",
    "f_lim_dereddened_err",
    "f_lim_dereddened_5sigma_extrap",
    "distance_pc",
    "distance_err_pc",
    "l_halpha_erg_s",
    "l_halpha_err_erg_s",
    "l_halpha_lsun",
    "l_halpha_5sigma_erg_s",
    "l_halpha_5sigma_lsun",
    "l_acc_lsun",
    "l_acc_5sigma_lsun",
    "mdot_msun_yr",
    "mdot_5sigma_msun_yr",
    "mdot_err_dex",
    "l_acc_aoyama21_lsun",
    "l_acc_aoyama21_5sigma_lsun",
    "mdot_aoyama21_msun_yr",
    "mdot_aoyama21_5sigma_msun_yr",
    "mdot_aoyama21_err_dex",
    "relation_scatter_dex",
    "intermethod_scatter_pct",
]


@dataclass(frozen=True)
class StageH03Product:
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


def stage_h03_paths(run_id, project_root=None):
    root = Path(project_root or Path.cwd()).resolve()
    paths = RunPaths.from_project_root(run_id, root)
    plot_dir = paths.plot_stage_dir("stage_h03")
    return {
        "paths": paths,
        "stage_h01_qc_json": paths.stage_dir / "stage_h01_qc.json",
        "stage_h02_qc_json": paths.stage_dir / "stage_h02_qc.json",
        "stage_h04_qc_json": paths.stage_dir / "stage_h04_qc.json",
        "stage_x11_qc_json": paths.stage_dir / "stage_x11_qc.json",
        "stage00q_qc_json": paths.stage_dir / "stage00q_qc.json",
        "stage01c_qc_json": paths.stage_dir / "stage01c_qc.json",
        "halpha_detection_csv": paths.table_dir / "halpha_detection_by_method.csv",
        "h01_null_maxima_npz": paths.stage_dir / "stage_h01_null_maxima.npz",
        "injection_throughput_csv": paths.table_dir / "injection_throughput_by_method.csv",
        "spec_calibrated_aperture_object": paths.stage_dir / "spec_calibrated_aperture_object.fits",
        "spec_calibrated_optimal_ls_object": paths.stage_dir / "spec_calibrated_optimal_ls_object.fits",
        "spec_calibrated_optimal_psfsub_object": paths.stage_dir / "spec_calibrated_optimal_psfsub_object.fits",
        "spec_calibrated_psffit_object": paths.stage_dir / "spec_calibrated_psffit_object.fits",
        "halpha_upper_limits_csv": paths.table_dir / "halpha_upper_limits.csv",
        "literature_csv": root / "musepipe" / "qc" / "data" / "halpha_literature_limits.csv",
        "plot_dir": plot_dir,
        "context_plot": plot_dir / "stage_h03_limits_context.png",
        "stage_h03_qc_json": paths.stage_dir / "stage_h03_qc.json",
    }


def stage_h03_config_from_run(
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
    cfg.setdefault("h03_template_width_factors", [1.0, 2.0])
    cfg.setdefault("h03_tail_fap_99", 0.01)
    cfg.setdefault("h03_tail_fap_5sigma", ONE_SIDED_5SIGMA_FAP)
    cfg.setdefault("h03_canonical_method", cfg.get("x11_canonical_method", cfg.get("canonical_method")))
    return cfg


def _read_optional_json(path):
    if path is None:
        return None
    path = Path(path)
    if not path.exists():
        return None
    return read_json(path)


def _read_csv_rows(path):
    path = Path(path)
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _float_cell(row, keys, default=np.nan):
    for key in keys:
        if key in row and row[key] not in (None, ""):
            try:
                return float(row[key])
            except (TypeError, ValueError):
                continue
    return float(default)


def _str_cell(row, keys, default=""):
    for key in keys:
        if key in row and row[key] not in (None, ""):
            return str(row[key])
    return default


def _cfg_value(cfg, *keys, default=None):
    for key in keys:
        if key in cfg and cfg[key] is not None:
            return cfg[key]
    return default


def _required_float(cfg, keys, label):
    value = _cfg_value(cfg, *keys)
    if value is None:
        joined = ", ".join(keys)
        raise RuntimeError(f"H03 requires {label} in config ({joined}).")
    val = float(value)
    if not np.isfinite(val):
        raise RuntimeError(f"H03 config value for {label} must be finite.")
    return val


def _required_string(cfg, keys, label):
    value = _cfg_value(cfg, *keys)
    if value is None or not str(value).strip():
        joined = ", ".join(keys)
        raise RuntimeError(f"H03 requires {label} in config ({joined}).")
    return str(value)


def _e1_verdict(qc):
    verdict = (qc or {}).get("verdict")
    if isinstance(verdict, dict):
        return str(verdict.get("verdict", "unknown"))
    if verdict is None:
        return "unknown"
    return str(verdict)


def _e4_v1_status(qc):
    if not qc:
        return "missing"
    regression = qc.get("regression_historic")
    if isinstance(regression, dict) and regression.get("verdict") is not None:
        return str(regression["verdict"]).lower()
    checks = qc.get("checks")
    if isinstance(checks, dict):
        for key in ("v1_regression", "V1", "v1"):
            value = checks.get(key)
            if isinstance(value, dict):
                for status_key in ("status", "verdict"):
                    if value.get(status_key) is not None:
                        return str(value[status_key]).lower()
            elif value is not None:
                return str(value).lower()
    for key in ("e4_v1", "v1", "V1"):
        value = qc.get(key)
        if value is not None:
            return str(value).lower()
    return "unknown"


def validate_prerequisites(h01_qc, h02_qc, h04_qc, *, allow_unvalidated_throughput=False,
                           allow_detection=False):
    """Puerta de H03, por veredicto de E1.

    - `non_detection`: el caso para el que existe la etapa.
    - `candidate`: **se permite**, con aviso. Es lo que el paper reporta de un
      candidato cuyo intervalo cruza el umbral (decision congelada), y negarse
      dejaba al objeto sin limite Y sin medida, con el producto viejo como unica
      cifra citable.
    - `detection`: se niega salvo `allow_detection`. Un "limite superior al 99 %"
      sobre una linea detectada no es una cantidad publicable; con la bandera
      sale igualmente, pero el QC lo marca como SENSIBILIDAD y no como limite.
    """
    e1 = _e1_verdict(h01_qc)
    e2 = str((h02_qc or {}).get("overall", "unknown")).lower()
    e4 = _e4_v1_status(h04_qc)
    prereq = {"e1_verdict": e1, "e2_overall": e2, "e4_v1": e4,
              "product_kind": "upper_limit", "issues": []}
    if e1 == "candidate":
        prereq["issues"].append(
            "E1 calls this a candidate, not a non-detection: the limit is reported "
            "because the candidate's interval crosses the detection criterion."
        )
    elif e1 == "detection":
        if not allow_detection:
            raise RuntimeError(
                "H03 refuses to run: E1 verdict is 'detection'; a 99% upper limit on a "
                "detected line is not a publishable quantity. Pass allow_detection=True "
                "(h03_allow_detection) to get the threshold as a SENSITIVITY instead."
            )
        prereq["product_kind"] = "sensitivity"
        prereq["issues"].append(
            "E1 says 'detection': these numbers are the detection SENSITIVITY of each "
            "scheme, NOT an upper limit on the companion. Do not cite them as limits."
        )
    elif e1 != "non_detection":
        raise RuntimeError(
            f"H03 refuses to run: E1 verdict must be non_detection, candidate or "
            f"detection, got {e1!r}.")
    if e2 != "survives":
        raise RuntimeError(f"H03 refuses to run: E2 overall must be survives, got {e2!r}.")
    if e4 != "pass":
        if allow_unvalidated_throughput:
            prereq["issues"].append(
                f"E4 V1 historic regression is {e4!r}; throughput accepted as PROVISIONAL "
                "(the historical box3_sum=8.97 gate is from a different cube/flux units; the new "
                "in-memory injection machinery is instead validated by the synthetic regression test)."
            )
        else:
            raise RuntimeError(f"H03 refuses to run: valid E4 throughput is mandatory; V1 is {e4!r}.")
    return prereq


def empirical_upper_quantile(values, fap=0.01):
    vals = np.sort(np.asarray(values, dtype=np.float64))
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        raise ValueError("At least one finite null maximum is required.")
    probability = 1.0 - float(fap)
    try:
        return float(np.quantile(vals, probability, method="higher"))
    except TypeError:
        return float(np.quantile(vals, probability, interpolation="higher"))


def fit_gumbel_moments(values):
    vals = np.asarray(values, dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    if vals.size < 3:
        raise ValueError("At least three finite null maxima are required for Gumbel tail fitting.")
    std = float(np.std(vals, ddof=1))
    if not np.isfinite(std) or std <= 0:
        raise ValueError("Null maxima must have non-zero scatter for Gumbel tail fitting.")
    beta = std * math.sqrt(6.0) / math.pi
    loc = float(np.mean(vals) - EULER_GAMMA * beta)
    return loc, beta


def gumbel_isf(fap, loc, beta):
    tail = float(fap)
    if not 0.0 < tail < 1.0:
        raise ValueError("FAP must lie between zero and one.")
    cdf = 1.0 - tail
    return float(loc - float(beta) * math.log(-math.log(cdf)))


#: los dos modelos de cola con los que se puede fijar el umbral del 99 %.
TAIL_ESTIMATORS = ("empirical", "parametric")


def resolve_tail_estimator(cfg, h01_qc):
    """Con que cola se fija el umbral del 99 %, y de donde sale esa decision.

    Mismo orden que el resto de knobs con procedencia (ver `_wavelength_frame`):

    1. el knob explicito de config (`h03_tail_estimator`);
    2. el estimador con el que E1 decide (`criterion.fap_estimator` de su QC) —
       si E1 declara la deteccion con la cola parametrica, un limite fijado con
       el cuantil empirico no es el mismo umbral, y el paper acabaria citando
       una deteccion y un limite calibrados de forma distinta;
    3. error explicito, nunca un default silencioso.

    Lo que costaba no tenerlo: E3 usaba `z_99_empirical` incrustado, asi que al
    pasar E1 a la cola parametrica los limites publicados quedaron un 32 % mas
    apretados de lo que da el metodo declarado (metodo canonico de ROXs 42B b).
    """
    declarado = cfg.get("h03_tail_estimator")
    if declarado is not None:
        fuente = "config.h03_tail_estimator"
    else:
        declarado = ((h01_qc or {}).get("criterion") or {}).get("fap_estimator")
        fuente = "h01_qc.criterion.fap_estimator"
    if declarado not in TAIL_ESTIMATORS:
        raise RuntimeError(
            "H03 needs the tail estimator for the 99% threshold: declare "
            f"h03_tail_estimator ({' or '.join(TAIL_ESTIMATORS)}) or run E1 so its "
            f"QC carries criterion.fap_estimator; got {declarado!r} from {fuente}."
        )
    return str(declarado), fuente


def z_threshold_for(tail, estimator):
    """El umbral del 99 % segun la cola elegida; `parametric` = Gumbel."""
    if estimator == "parametric":
        return float(tail["z_99_gumbel"])
    return float(tail["z_99_empirical"])


def tail_limit_summary(null_values, *, fap_99=0.01, fap_5sigma=ONE_SIDED_5SIGMA_FAP):
    vals = np.asarray(null_values, dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    if vals.size < 3:
        raise ValueError("At least three finite null maxima are required.")
    loc, beta = fit_gumbel_moments(vals)
    minimum_resolvable_fap = float(1.0 / (vals.size + 1))
    return {
        "tail_model": "gumbel_moments",
        "n_controls": int(vals.size),
        "minimum_resolvable_fap": minimum_resolvable_fap,
        "q99_resolvable": bool(float(fap_99) >= minimum_resolvable_fap),
        "z_99_empirical": empirical_upper_quantile(vals, fap=fap_99),
        "z_99_gumbel": gumbel_isf(fap_99, loc, beta),
        "z_5sigma_extrap": gumbel_isf(fap_5sigma, loc, beta),
        "tail_extrapolated_5sigma": bool(float(fap_5sigma) < minimum_resolvable_fap),
        "gumbel_loc": loc,
        "gumbel_beta": beta,
    }


def template_width_label(factor):
    val = float(factor)
    if np.isclose(val, 1.0):
        return "lsf"
    if float(val).is_integer():
        return f"{int(val)}x_lsf"
    return f"{val:g}x_lsf"


def parse_template_factor(value):
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        pass
    text = str(value).strip().lower().replace("\u00d7", "x")
    text = text.replace(" ", "").replace("-", "_").replace("x", "")
    text = text.replace("*", "").replace("times", "").replace("lsf", "")
    if text in {"", "_"}:
        return 1.0
    text = text.replace("_", "")
    try:
        return float(text)
    except ValueError as exc:
        raise ValueError(f"Could not parse template width factor {value!r}.") from exc


def _psf_perturbation_frac(h04_qc):
    value = ((h04_qc or {}).get("throughput") or {}).get("psf_perturbation_pct")
    if value is None:
        value = ((h04_qc or {}).get("throughput") or {}).get("psf_perturbation_frac")
        if value is None:
            return 0.0
        return abs(float(value))
    return abs(float(value)) / 100.0


def _throughput_records(rows, *, h04_qc=None):
    records = []
    psf_frac = _psf_perturbation_frac(h04_qc)
    for row in rows:
        method = _str_cell(row, ("method", "extractor", "name"))
        if not method:
            continue
        snr = _float_cell(row, ("snr", "input_snr", "snr_in", "snr_injected", "injected_snr", "target_snr"))
        throughput = _float_cell(
            row,
            ("throughput", "throughput_median", "recovery_fraction", "flux_throughput", "recovered_over_injected"),
        )
        if not np.isfinite(snr) or not np.isfinite(throughput):
            continue
        width_value = _str_cell(
            row,
            ("template_factor", "width_factor", "template_width_factor", "line_width_factor", "template_width", "width"),
            default="",
        )
        factor = parse_template_factor(width_value) if width_value else None
        err = _float_cell(row, ("throughput_err", "throughput_sigma", "throughput_std", "throughput_unc", "throughput_error"))
        err_pct = _float_cell(row, ("throughput_err_pct", "throughput_error_pct"), default=np.nan)
        err_frac = _float_cell(row, ("throughput_err_frac", "throughput_error_frac"), default=np.nan)
        components = []
        if np.isfinite(err):
            components.append(float(err))
        if np.isfinite(err_pct):
            components.append(abs(float(throughput)) * abs(float(err_pct)) / 100.0)
        if np.isfinite(err_frac):
            components.append(abs(float(throughput)) * abs(float(err_frac)))
        if psf_frac > 0:
            components.append(abs(float(throughput)) * psf_frac)
        combined_err = float(math.sqrt(sum(value * value for value in components))) if components else np.nan
        records.append(
            {
                "method": method,
                "template_factor": factor,
                "snr": float(snr),
                "throughput": float(throughput),
                "throughput_err": combined_err,
            }
        )
    return records


def _group_throughput(records):
    grouped = {}
    for rec in records:
        key = float(rec["snr"])
        grouped.setdefault(key, []).append(rec)
    points = []
    for snr, group in grouped.items():
        vals = np.asarray([item["throughput"] for item in group], dtype=np.float64)
        errs = np.asarray([item["throughput_err"] for item in group], dtype=np.float64)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            continue
        throughput = float(np.nanmedian(vals))
        err_components = []
        finite_err = errs[np.isfinite(errs)]
        if finite_err.size:
            err_components.append(float(np.nanmedian(finite_err)))
        if vals.size > 1:
            err_components.append(float(np.nanstd(vals, ddof=1)))
        throughput_err = float(math.sqrt(sum(value * value for value in err_components))) if err_components else np.nan
        points.append({"snr": float(snr), "throughput": throughput, "throughput_err": throughput_err})
    return sorted(points, key=lambda item: item["snr"])


def interpolate_throughput(rows, method, template_factor, snr_target, *, h04_qc=None):
    records = _throughput_records(rows, h04_qc=h04_qc)
    matched = []
    for rec in records:
        if rec["method"] != method:
            continue
        factor = rec["template_factor"]
        if factor is not None and not np.isclose(float(factor), float(template_factor), rtol=0.0, atol=1e-6):
            continue
        matched.append(rec)
    if not matched:
        raise RuntimeError(
            f"H03 requires E4 throughput rows for method={method!r}, template_factor={float(template_factor):g}."
        )
    points = _group_throughput(matched)
    if not points:
        raise RuntimeError(f"H03 could not build a throughput curve for {method}.")
    xs = np.asarray([point["snr"] for point in points], dtype=np.float64)
    ys = np.asarray([point["throughput"] for point in points], dtype=np.float64)
    es = np.asarray([point["throughput_err"] for point in points], dtype=np.float64)
    target = float(snr_target)
    if xs.size == 1:
        throughput = float(ys[0])
        err = float(es[0])
        source = "single_point"
    elif target <= xs[0]:
        throughput = float(ys[0])
        err = float(es[0])
        source = "lower_edge"
    elif target >= xs[-1]:
        throughput = float(ys[-1])
        err = float(es[-1])
        source = "upper_edge"
    else:
        throughput = float(np.interp(target, xs, ys))
        err = float(np.interp(target, xs, es))
        source = "linear_interpolation"
    if not np.isfinite(throughput) or throughput <= 0:
        raise RuntimeError(f"H03 requires positive finite throughput for {method}.")
    if not np.isfinite(err):
        raise RuntimeError(f"H03 requires throughput uncertainty for {method}; E4 table/QC did not provide it.")
    return {
        "throughput": throughput,
        "throughput_err": max(0.0, err),
        "snr_for_throughput": target,
        "throughput_source": source,
    }


def ccm89_a_over_av(wavelength_A, rv):
    wave_um = float(wavelength_A) / 10000.0
    x = 1.0 / wave_um
    if not 1.1 <= x <= 3.3:
        raise ValueError("CCM89 optical law is implemented for 1.1 <= x <= 3.3 um^-1.")
    y = x - 1.82
    a = (
        1.0
        + 0.17699 * y
        - 0.50447 * y**2
        - 0.02427 * y**3
        + 0.72085 * y**4
        + 0.01979 * y**5
        - 0.77530 * y**6
        + 0.32999 * y**7
    )
    b = (
        1.41338 * y
        + 2.28305 * y**2
        + 1.07233 * y**3
        - 5.38434 * y**4
        - 0.62251 * y**5
        + 5.30260 * y**6
        - 2.09002 * y**7
    )
    return float(a + b / float(rv))


def _extinction_ratio_from_config(cfg):
    explicit = _cfg_value(cfg, "h03_a_halpha_over_av", "a_halpha_over_av")
    if explicit is not None:
        _required_string(
            cfg,
            ("h03_extinction_law_citation", "extinction_law_citation"),
            "extinction law citation",
        )
        return float(explicit), "config.h03_a_halpha_over_av"
    law = _required_string(cfg, ("h03_extinction_law", "extinction_law"), "extinction law")
    _required_string(
        cfg,
        ("h03_extinction_law_citation", "extinction_law_citation"),
        "extinction law citation",
    )
    rv = _required_float(cfg, ("h03_rv_extinction", "rv_extinction"), "extinction R_V")
    if law.strip().lower().startswith("ccm"):
        return ccm89_a_over_av(HALPHA_REST_A, rv), f"{law}(R_V={rv:g})"
    raise RuntimeError("H03 currently supports explicit h03_a_halpha_over_av or CCM extinction law.")


def _aoyama21_relation_from_config(cfg):
    """R1: optional planetary-shock L_acc-L_Halpha relation (Aoyama et al. 2021),
    read from cfg['g3_lacc_relations']['halpha_aoyama21']. Returns None if absent."""
    relations = cfg.get("g3_lacc_relations")
    rel = relations.get("halpha_aoyama21") if isinstance(relations, dict) else None
    if not isinstance(rel, dict):
        return None
    try:
        return {
            "a": float(rel["a"]),
            "b": float(rel["b"]),
            "scatter_dex": float(rel.get("scatter_dex", 0.30)),
            "citation": str(rel.get("citation", "Aoyama et al. 2021, ApJL 917, L30")),
            "validity_range": rel.get("validity_range", ""),
        }
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(f"Invalid g3_lacc_relations.halpha_aoyama21 entry: {exc}") from exc


def physical_inputs_from_config(cfg):
    distance_pc = _required_float(cfg, ("h03_distance_pc", "distance_pc"), "distance")
    distance_err_pc = _required_float(cfg, ("h03_distance_err_pc", "distance_err_pc"), "distance error")
    distance_source = _required_string(cfg, ("h03_distance_source", "distance_source"), "distance source/citation")
    av = _required_float(cfg, ("h03_av", "av"), "A_V")
    av_err = _required_float(cfg, ("h03_av_err", "av_err"), "A_V error")
    av_source = _required_string(cfg, ("h03_av_source", "av_source"), "A_V source/citation")
    rv_sys_kms = _required_float(cfg, ("h03_rv_sys_kms", "rv_sys_kms", "systemic_rv_kms"), "systemic RV")
    rv_source = _required_string(cfg, ("h03_rv_source", "rv_source", "systemic_rv_source"), "systemic RV source")
    a_halpha_over_av, extinction_source = _extinction_ratio_from_config(cfg)
    relation = _required_string(
        cfg,
        ("h03_lacc_lha_relation", "lacc_lha_relation"),
        "Lacc-LHalpha relation name",
    )
    relation_citation = _required_string(
        cfg,
        ("h03_lacc_lha_citation", "lacc_lha_citation"),
        "Lacc-LHalpha relation citation",
    )
    relation_slope = _required_float(
        cfg,
        ("h03_lacc_lha_a", "h03_lacc_lha_slope", "lacc_lha_a"),
        "Lacc-LHalpha slope",
    )
    relation_intercept = _required_float(
        cfg,
        ("h03_lacc_lha_b", "h03_lacc_lha_intercept", "lacc_lha_b"),
        "Lacc-LHalpha intercept",
    )
    relation_scatter_dex = _required_float(
        cfg,
        ("h03_relation_scatter_dex", "relation_scatter_dex"),
        "Lacc-LHalpha intrinsic scatter",
    )
    mass_msun = _required_float(
        cfg,
        ("h03_companion_mass_msun", "companion_mass_msun", "mass_msun"),
        "companion mass",
    )
    radius_rsun = _required_float(
        cfg,
        ("h03_companion_radius_rsun", "companion_radius_rsun", "radius_rsun"),
        "companion radius",
    )
    mass_source = _required_string(
        cfg,
        ("h03_mass_source", "companion_mass_source", "mass_source"),
        "companion mass source/citation",
    )
    radius_source = _required_string(
        cfg,
        ("h03_radius_source", "companion_radius_source", "radius_source"),
        "companion radius source/citation",
    )
    return {
        "distance_pc": distance_pc,
        "distance_err_pc": distance_err_pc,
        "distance_source": distance_source,
        "av": av,
        "av_err": av_err,
        "av_source": av_source,
        "rv_sys_kms": rv_sys_kms,
        "rv_source": rv_source,
        "a_halpha_over_av": a_halpha_over_av,
        "extinction_source": extinction_source,
        "lacc_lha_relation": relation,
        "lacc_lha_citation": relation_citation,
        "lacc_lha_a": relation_slope,
        "lacc_lha_b": relation_intercept,
        "relation_scatter_dex": relation_scatter_dex,
        "companion_mass_msun": mass_msun,
        "companion_radius_rsun": radius_rsun,
        "mass_source": mass_source,
        "radius_source": radius_source,
        "lacc_aoyama21": _aoyama21_relation_from_config(cfg),
    }


def luminosity_erg_s(flux, distance_pc):
    return float(4.0 * math.pi * (float(distance_pc) * PC_CM) ** 2 * float(flux))


def lacc_lsun_from_lha(lha_lsun, slope, intercept):
    if lha_lsun <= 0 or not np.isfinite(lha_lsun):
        return np.nan
    return float(10.0 ** (float(slope) * math.log10(float(lha_lsun)) + float(intercept)))


#: (1 - R/R_in)^-1 con R_in = 5R (Gullbring et al. 1998), el radio de truncamiento
#: magnetosferico que fija el spec G3 (linea 143) y que la seccion de metodos del
#: paper declara para TODA conversion a Mdot. G3 lo aplicaba (via
#: `models.accretion.mdot_from_lacc`) y E3 no, asi que la deteccion y el limite del
#: mismo paper no eran la misma formula: el limite salia 1.25x mas apretado que la
#: medida con la que se compara. Vive aqui porque `models.accretion` ya importa de
#: este modulo y al reves seria circular.
MAGNETOSPHERIC_FACTOR = 1.25


def mdot_msun_yr_from_lacc(lacc_lsun, mass_msun, radius_rsun):
    """Mdot = L_acc R / (G M), SIN el factor magnetosferico.

    Es la primitiva; quien publica un Mdot le aplica `MAGNETOSPHERIC_FACTOR`
    (`limit_conversion_chain` aqui, `models.accretion.mdot_from_lacc` en G3).
    """
    if lacc_lsun <= 0 or not np.isfinite(lacc_lsun):
        return np.nan
    lacc_cgs = float(lacc_lsun) * L_SUN_ERG_S
    radius_cm = float(radius_rsun) * R_SUN_CM
    mass_g = float(mass_msun) * M_SUN_G
    mdot_g_s = lacc_cgs * radius_cm / (G_CGS * mass_g)
    return float(mdot_g_s * YR_S / M_SUN_G)


def limit_conversion_chain(
    *,
    z_threshold,
    z_5sigma_extrap,
    matched_sigma,
    throughput,
    throughput_err,
    distance_pc,
    distance_err_pc,
    av,
    av_err,
    a_halpha_over_av,
    lacc_lha_slope,
    lacc_lha_intercept,
    relation_scatter_dex,
    mass_msun,
    radius_rsun,
    alt_lacc_slope=None,
    alt_lacc_intercept=None,
    alt_scatter_dex=None,
):
    f_stat = float(z_threshold) * float(matched_sigma)
    f_stat_5sigma = float(z_5sigma_extrap) * float(matched_sigma)
    f_obs = f_stat / float(throughput)
    f_obs_5sigma = f_stat_5sigma / float(throughput)
    throughput_frac = 0.0 if float(throughput) == 0 else float(throughput_err) / float(throughput)
    f_obs_err = abs(f_obs) * abs(throughput_frac)
    ext_factor = float(10.0 ** (0.4 * float(av) * float(a_halpha_over_av)))
    ext_frac = math.log(10.0) * 0.4 * float(a_halpha_over_av) * float(av_err)
    f_dered = f_obs * ext_factor
    f_dered_5sigma = f_obs_5sigma * ext_factor
    f_dered_err = abs(f_dered) * math.sqrt(throughput_frac**2 + ext_frac**2)
    lha = luminosity_erg_s(f_dered, distance_pc)
    lha_5sigma = luminosity_erg_s(f_dered_5sigma, distance_pc)
    distance_frac_lum = 0.0 if float(distance_pc) == 0 else 2.0 * float(distance_err_pc) / float(distance_pc)
    lha_err = abs(lha) * math.sqrt(throughput_frac**2 + ext_frac**2 + distance_frac_lum**2)
    lha_lsun = lha / L_SUN_ERG_S
    lha_5sigma_lsun = lha_5sigma / L_SUN_ERG_S
    lacc = lacc_lsun_from_lha(lha_lsun, lacc_lha_slope, lacc_lha_intercept)
    lacc_5sigma = lacc_lsun_from_lha(lha_5sigma_lsun, lacc_lha_slope, lacc_lha_intercept)
    mdot = MAGNETOSPHERIC_FACTOR * mdot_msun_yr_from_lacc(lacc, mass_msun, radius_rsun)
    mdot_5sigma = MAGNETOSPHERIC_FACTOR * mdot_msun_yr_from_lacc(lacc_5sigma, mass_msun, radius_rsun)
    lha_frac = 0.0 if lha == 0 else abs(lha_err / lha)
    measurement_dex = lha_frac / math.log(10.0)
    mdot_err_dex = math.sqrt((float(lacc_lha_slope) * measurement_dex) ** 2 + float(relation_scatter_dex) ** 2)
    # R1: parallel accretion limit under a second L_acc-L_line relation (Aoyama+21
    # planetary shock) on the SAME dereddened L_Halpha. Additive; NaN when absent.
    if alt_lacc_slope is not None and alt_lacc_intercept is not None:
        lacc_alt = lacc_lsun_from_lha(lha_lsun, alt_lacc_slope, alt_lacc_intercept)
        lacc_alt_5sigma = lacc_lsun_from_lha(lha_5sigma_lsun, alt_lacc_slope, alt_lacc_intercept)
        mdot_alt = MAGNETOSPHERIC_FACTOR * mdot_msun_yr_from_lacc(lacc_alt, mass_msun, radius_rsun)
        mdot_alt_5sigma = MAGNETOSPHERIC_FACTOR * mdot_msun_yr_from_lacc(
            lacc_alt_5sigma, mass_msun, radius_rsun)
        alt_scatter = float(relation_scatter_dex if alt_scatter_dex is None else alt_scatter_dex)
        mdot_alt_err_dex = math.sqrt((float(alt_lacc_slope) * measurement_dex) ** 2 + alt_scatter ** 2)
    else:
        lacc_alt = lacc_alt_5sigma = mdot_alt = mdot_alt_5sigma = mdot_alt_err_dex = np.nan
    return {
        "f_stat_99": f_stat,
        "f_stat_5sigma_extrap": f_stat_5sigma,
        "f_lim_observed": f_obs,
        "f_lim_observed_err": f_obs_err,
        "f_lim_observed_5sigma_extrap": f_obs_5sigma,
        "f_lim_dereddened": f_dered,
        "f_lim_dereddened_err": f_dered_err,
        "f_lim_dereddened_5sigma_extrap": f_dered_5sigma,
        "l_halpha_erg_s": lha,
        "l_halpha_err_erg_s": lha_err,
        "l_halpha_lsun": lha_lsun,
        "l_halpha_5sigma_erg_s": lha_5sigma,
        "l_halpha_5sigma_lsun": lha_5sigma_lsun,
        "l_acc_lsun": lacc,
        "l_acc_5sigma_lsun": lacc_5sigma,
        "mdot_msun_yr": mdot,
        "mdot_5sigma_msun_yr": mdot_5sigma,
        "mdot_err_dex": mdot_err_dex,
        "l_acc_aoyama21_lsun": lacc_alt,
        "l_acc_aoyama21_5sigma_lsun": lacc_alt_5sigma,
        "mdot_aoyama21_msun_yr": mdot_alt,
        "mdot_aoyama21_5sigma_msun_yr": mdot_alt_5sigma,
        "mdot_aoyama21_err_dex": mdot_alt_err_dex,
    }


def _null_maxima_by_method(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    out = {}
    with np.load(path) as data:
        for key in data.files:
            if key.endswith("_null_maxima"):
                method = key[: -len("_null_maxima")]
            else:
                method = key
            out[method] = np.asarray(data[key], dtype=np.float64)
    if not out:
        raise RuntimeError(f"No null maxima arrays found in {path}.")
    return out


def _matched_sigmas_from_config(cfg):
    raw = cfg.get("h03_matched_sigma_by_method")
    out = {}
    if not isinstance(raw, dict):
        return out
    for method, value in raw.items():
        if isinstance(value, dict):
            by_factor = {}
            for factor_key, sigma in value.items():
                factor = parse_template_factor(factor_key)
                by_factor[float(factor)] = {"matched_sigma": float(sigma), "sigma_source": "config.h03_matched_sigma_by_method"}
            out[str(method)] = by_factor
        else:
            out[str(method)] = {
                1.0: {"matched_sigma": float(value), "sigma_source": "config.h03_matched_sigma_by_method"}
            }
    return out


def _matched_sigmas_from_h01_table(rows):
    out = {}
    for row in rows:
        method = row.get("method")
        if not method:
            continue
        sigma = _float_cell(row, ("matched_sigma",))
        if not np.isfinite(sigma):
            continue
        factor = parse_template_factor(row.get("template_factor", 1.0))
        out.setdefault(method, {})[float(factor)] = {
            "matched_sigma": float(sigma),
            "sigma_source": "h01_detection_table",
        }
    return out


def _product_path_for_method(paths, method):
    key = f"spec_calibrated_{method}_object"
    return paths.get(key)



def resolve_flux_unit_cgs(cfg, paths, canonical_method):
    """Escala a erg/s/cm2/A: knob, `BUNIT` del producto canonico, o el QC de M3.

    La tercera fuente cubre los productos generados antes de que B1/B2
    propagaran `BUNIT`: si A4/M3 midio el factor de flujo sobre ese mismo cubo,
    la unidad que uso es la que sostiene la escala que D2 ya aplico.
    """
    bunit = None
    product_path = _product_path_for_method(paths, canonical_method)
    if product_path is not None and Path(product_path).exists():
        try:
            bunit = cube_bunit(product_path)
        except (OSError, ValueError):
            bunit = None
    return _flux_unit_cgs(cfg, bunit=bunit, qc_m3=_read_optional_json(paths.get("stage00q_qc_json")))

def _matched_sigma_from_product(paths, h01_qc, method, factor):
    product_path = _product_path_for_method(paths, method)
    if product_path is None or not Path(product_path).exists():
        raise RuntimeError(
            f"H03 needs matched_sigma for {method} template_factor={float(factor):g}; "
            "provide h03_matched_sigma_by_method or the calibrated H01 product."
        )
    product = SpectrumProduct.read(product_path)
    extra = product.extra_columns or {}
    error = np.asarray(extra.get("flux_err_total", product.flux_err), dtype=np.float64)
    rest_A = float(((h01_qc or {}).get("line") or {}).get("rest_A", HALPHA_REST_A))
    rv_sys = float(((h01_qc or {}).get("line") or {}).get("rv_sys_kms", 0.0))
    lsf_fwhm = ((h01_qc or {}).get("templates") or {}).get("lsf_fwhm_A")
    if lsf_fwhm is None:
        raise RuntimeError("H03 requires H01 QC templates.lsf_fwhm_A to recompute matched sigma.")
    center = expected_line_center_A(rest_A, rv_sys)
    zeros = np.zeros_like(product.wave_A, dtype=np.float64)
    _flux, sigma, _z = matched_filter_point(
        product.wave_A,
        zeros,
        error,
        center,
        float(lsf_fwhm) * float(factor),
        _good_detection_mask(product),
    )
    if not np.isfinite(sigma) or sigma <= 0:
        raise RuntimeError(f"Could not compute finite matched sigma for {method}.")
    return {"matched_sigma": float(sigma), "sigma_source": "recomputed_h01_estimator"}


def _merge_sigma_maps(*maps):
    out = {}
    for mapping in maps:
        for method, by_factor in mapping.items():
            out.setdefault(method, {}).update(by_factor)
    return out


def matched_sigma_for_method_factor(paths, cfg, h01_qc, h01_rows, method, factor):
    """El sigma del filtro adaptado, por UNA via, y con la otra declarada.

    Habia dos definiciones de la misma cantidad —el valor que E1 dejo en su
    tabla de deteccion y el recalculo sobre el producto calibrado— y la eleccion
    entre ellas no la hacia nadie: la tabla de E1 trae UNA fila por metodo, con
    el ancho de plantilla que E1 prefirio, asi que E3 acertaba o fallaba segun
    esa preferencia. Medido el 2026-08-21 sobre ROXs 12 b: la epoca de 22
    exposiciones cayo en la tabla (E1 prefirio factor 1.0) y las otras dos en el
    recalculo (E1 prefirio 2.0), y donde ambas vias apuntan al MISMO factor
    discrepan un 3.3 %. Poco para las conclusiones —esta muy dentro de los
    0.3 dex de la relacion— pero era un ~3 % arbitrario que nadie declaraba.

    Ahora: la perilla de config manda (es una declaracion explicita); si no, se
    recalcula SIEMPRE al factor pedido, que es la via que siempre esta
    disponible y siempre responde a la pregunta correcta. Si la tabla de E1
    tiene valor a ese mismo factor, se compara y la discrepancia VIAJA en el
    resultado en vez de decidirse en silencio.
    """

    by_factor = _matched_sigmas_from_config(cfg).get(method, {})
    for key, payload in by_factor.items():
        if np.isclose(float(key), float(factor), rtol=0.0, atol=1e-6):
            return dict(payload)

    tabla = _matched_sigmas_from_h01_table(h01_rows).get(method, {})
    del_tabla = None
    for key, entry in tabla.items():
        if np.isclose(float(key), float(factor), rtol=0.0, atol=1e-6):
            del_tabla = float(entry["matched_sigma"])
            break

    try:
        payload = dict(_matched_sigma_from_product(paths, h01_qc, method, factor))
    except RuntimeError:
        # No hay producto calibrado para este metodo (p.ej. `sgf` en un run que
        # no lo entrega). La tabla de E1 es el respaldo, pero solo al MISMO
        # factor —un filtro mas ancho es otra cantidad— y queda dicho en
        # `sigma_source` en vez de pasar por el camino preferente.
        if del_tabla is None or not np.isfinite(del_tabla):
            raise
        return {"matched_sigma": float(del_tabla),
                "sigma_source": "h01_detection_table (fallback: sin producto calibrado)",
                "sigma_h01_table": float(del_tabla),
                "sigma_table_vs_recomputed_pct": None}
    payload["sigma_h01_table"] = del_tabla
    if del_tabla is not None and np.isfinite(del_tabla) and payload["matched_sigma"] > 0:
        payload["sigma_table_vs_recomputed_pct"] = float(
            (del_tabla / float(payload["matched_sigma"]) - 1.0) * 100.0
        )
    else:
        # `None` distingue "no habia con que comparar" de "comparado y coincide".
        payload["sigma_table_vs_recomputed_pct"] = None
    return payload


def _canonical_method(cfg, paths):
    value = cfg.get("h03_canonical_method")
    if value:
        return str(value), "config.h03_canonical_method"
    for key in ("x11_canonical_method", "canonical_method"):
        if cfg.get(key):
            return str(cfg[key]), f"config.{key}"
    qc = _read_optional_json(paths["stage_x11_qc_json"])
    if qc and qc.get("canonical_method"):
        return str(qc["canonical_method"]), "stage_x11_qc.canonical_method"
    raise RuntimeError("H03 requires canonical method from h03_canonical_method, D1/D2 config, or stage_x11_qc.json.")


def _intermethod_scatter_pct(rows):
    vals = [
        float(row["f_lim_dereddened"])
        for row in rows
        if row.get("row_kind") == "method" and np.isclose(float(row.get("template_factor", np.nan)), 1.0)
    ]
    arr = np.asarray(vals, dtype=np.float64)
    arr = arr[np.isfinite(arr) & (arr > 0)]
    if arr.size < 2:
        return 0.0
    med = float(np.nanmedian(arr))
    if med <= 0:
        return 0.0
    return float(100.0 * np.nanstd(arr, ddof=1) / med)


def _row_from_limits(
    *,
    method,
    canonical_method,
    factor,
    tail,
    sigma_payload,
    throughput_payload,
    physical,
    tail_estimator,
    row_kind="method",
    scatter_pct=None,
):
    z_threshold = z_threshold_for(tail, tail_estimator)
    chain = limit_conversion_chain(
        z_threshold=z_threshold,
        z_5sigma_extrap=tail["z_5sigma_extrap"],
        matched_sigma=sigma_payload["matched_sigma"],
        throughput=throughput_payload["throughput"],
        throughput_err=throughput_payload["throughput_err"],
        distance_pc=physical["distance_pc"],
        distance_err_pc=physical["distance_err_pc"],
        av=physical["av"],
        av_err=physical["av_err"],
        a_halpha_over_av=physical["a_halpha_over_av"],
        lacc_lha_slope=physical["lacc_lha_a"],
        lacc_lha_intercept=physical["lacc_lha_b"],
        relation_scatter_dex=physical["relation_scatter_dex"],
        mass_msun=physical["companion_mass_msun"],
        radius_rsun=physical["companion_radius_rsun"],
        alt_lacc_slope=(physical.get("lacc_aoyama21") or {}).get("a"),
        alt_lacc_intercept=(physical.get("lacc_aoyama21") or {}).get("b"),
        alt_scatter_dex=(physical.get("lacc_aoyama21") or {}).get("scatter_dex"),
    )
    row = {
        "row_kind": row_kind,
        "method": method,
        "canonical_method": canonical_method,
        "template_width": template_width_label(factor),
        "template_factor": float(factor),
        "tail_model": tail["tail_model"],
        "n_controls": tail["n_controls"],
        "minimum_resolvable_fap": tail["minimum_resolvable_fap"],
        "q99_resolvable": tail["q99_resolvable"],
        "z_99_empirical": tail["z_99_empirical"],
        "z_99_gumbel": tail["z_99_gumbel"],
        "tail_estimator": tail_estimator,
        "z_threshold": z_threshold,
        "z_5sigma_extrap": tail["z_5sigma_extrap"],
        "tail_extrapolated_5sigma": tail["tail_extrapolated_5sigma"],
        "matched_sigma": sigma_payload["matched_sigma"],
        "sigma_source": sigma_payload["sigma_source"],
        "sigma_h01_table": sigma_payload.get("sigma_h01_table"),
        "sigma_table_vs_recomputed_pct": sigma_payload.get("sigma_table_vs_recomputed_pct"),
        "throughput": throughput_payload["throughput"],
        "throughput_err": throughput_payload["throughput_err"],
        "snr_for_throughput": throughput_payload["snr_for_throughput"],
        "throughput_source": throughput_payload["throughput_source"],
        "a_halpha_over_av": physical["a_halpha_over_av"],
        "av": physical["av"],
        "av_err": physical["av_err"],
        "distance_pc": physical["distance_pc"],
        "distance_err_pc": physical["distance_err_pc"],
        "relation_scatter_dex": physical["relation_scatter_dex"],
        "intermethod_scatter_pct": "" if scatter_pct is None else float(scatter_pct),
    }
    row.update(chain)
    return row


def compute_stage_h03_products(config, paths=None) -> StageH03Product:
    cfg = dict(config)
    root = Path(cfg.get("project_root") or Path.cwd()).resolve()
    paths = stage_h03_paths(cfg["run_id"], root) if paths is None else paths
    h01_qc = _read_optional_json(paths["stage_h01_qc_json"])
    h02_qc = _read_optional_json(paths["stage_h02_qc_json"])
    h04_qc = _read_optional_json(paths["stage_h04_qc_json"])
    prerequisites = validate_prerequisites(
        h01_qc, h02_qc, h04_qc,
        allow_unvalidated_throughput=bool(cfg.get("h03_allow_unvalidated_throughput", False)),
        allow_detection=bool(cfg.get("h03_allow_detection", False)),
    )
    physical = physical_inputs_from_config(cfg)
    canonical_method, canonical_source = _canonical_method(cfg, paths)
    null_by_method = _null_maxima_by_method(paths["h01_null_maxima_npz"])
    h01_rows = _read_csv_rows(paths["halpha_detection_csv"])
    throughput_rows = _read_csv_rows(cfg.get("h03_injection_throughput_csv", paths["injection_throughput_csv"]))
    if not throughput_rows:
        raise RuntimeError("H03 requires E4 table injection_throughput_by_method.csv.")

    factors = [float(value) for value in cfg.get("h03_template_width_factors", [1.0, 2.0])]
    fap_99 = float(cfg.get("h03_tail_fap_99", 0.01))
    fap_5sigma = float(cfg.get("h03_tail_fap_5sigma", ONE_SIDED_5SIGMA_FAP))
    tail_estimator, tail_estimator_source = resolve_tail_estimator(cfg, h01_qc)
    rows = []
    open_issues = list(prerequisites.get("issues", []))
    # La escala fisica sale del knob si esta declarado y, si no, del BUNIT del
    # propio producto canonico (ver `io.flux_unit_cgs`): el default silencioso
    # de 1.0 daba L/Mdot 1e20 veces altos sin avisar.
    flux_unit_cgs = resolve_flux_unit_cgs(cfg, paths, canonical_method)
    if flux_unit_cgs != 1.0:
        # El knob `m3_flux_factor` puede venir de otra cosecha: solo se puede
        # decir "A4/M3 lo comprobo" si M3 corrio EN ESTE cubo (status green en
        # el QC de A4). Decirlo sin mirar es afirmar una comprobacion que no se
        # hizo -- paso en la noche buena de ROXs 12 b, cuyo M3 es `unavailable`
        # mientras el config arrastraba un 0.973 del cubo combinado.
        m3_factor = cfg.get("m3_flux_factor")
        m3_qc = (_read_optional_json(paths.get("stage00q_qc_json")) or {}).get("m3_flux") or {}
        m3_medido = str(m3_qc.get("status", "")) == "green"
        if m3_factor is not None and m3_medido:
            open_issues.append(
                f"Flux unit {flux_unit_cgs:g} erg/s/cm2/A applied to the matched-filter sigma so "
                "f_lim/L/Mdot are physical (MUSE cube native unit; scipost flux-calibrated). A4/M3 "
                f"cross-checked the absolute scale against Gaia DR3 RP on this cube: factor "
                f"{float(m3_qc.get('flux_factor', m3_factor)):.3f} (consistent with 1 to ~3% after "
                "growth-curve + tail correction), so no large absolute systematic remains."
            )
        elif m3_factor is not None:
            open_issues.append(
                f"Flux unit {flux_unit_cgs:g} erg/s/cm2/A applied to the matched-filter sigma so "
                "f_lim/L/Mdot are physical. A4/M3 did NOT measure the absolute scale on this cube "
                f"(m3_flux.status={m3_qc.get('status', 'missing')!r}); config declares "
                f"m3_flux_factor={float(m3_factor):.3f}, which comes from another product and is not "
                "a measurement of this cube. The absolute-calibration term is UNMEASURED here."
            )
        else:
            open_issues.append(
                f"Flux unit {flux_unit_cgs:g} erg/s/cm2/A applied to the matched-filter sigma so f_lim/L/Mdot "
                "are physical (MUSE cube native unit; scipost flux-calibrated). NOTE: D2 flux scale=1 (M3 "
                "absolute cross-check unavailable), so an absolute-calibration systematic remains."
            )
    methods = [method for method in METHOD_ORDER if method in null_by_method]
    methods.extend(sorted(method for method in null_by_method if method not in methods))
    for method in methods:
        tail = tail_limit_summary(null_by_method[method], fap_99=fap_99, fap_5sigma=fap_5sigma)
        if not tail["q99_resolvable"]:
            if tail_estimator == "parametric":
                open_issues.append(
                    f"{method}: the 99% quantile is below the finite-control resolution "
                    f"({tail['minimum_resolvable_fap']:.4g}), so the threshold is the "
                    "fitted Gumbel tail and not a counted exceedance."
                )
            else:
                open_issues.append(
                    f"{method}: 99% empirical FAP is below the finite-control resolution "
                    f"({tail['minimum_resolvable_fap']:.4g})."
                )
        for factor in factors:
            sigma_payload = matched_sigma_for_method_factor(paths, cfg, h01_qc, h01_rows, method, factor)
            if flux_unit_cgs != 1.0:
                sigma_payload = dict(sigma_payload)
                sigma_payload["matched_sigma"] = float(sigma_payload["matched_sigma"]) * flux_unit_cgs
                sigma_payload["sigma_source"] = f"{sigma_payload.get('sigma_source', '')} x flux_unit={flux_unit_cgs:g}"
            throughput_payload = interpolate_throughput(
                throughput_rows,
                method,
                factor,
                z_threshold_for(tail, tail_estimator),
                h04_qc=h04_qc,
            )
            rows.append(
                _row_from_limits(
                    method=method,
                    canonical_method=canonical_method,
                    factor=factor,
                    tail=tail,
                    sigma_payload=sigma_payload,
                    throughput_payload=throughput_payload,
                    physical=physical,
                    tail_estimator=tail_estimator,
                )
            )
    scatter_pct = _intermethod_scatter_pct(rows)
    if scatter_pct > 30.0:
        open_issues.append(
            f"Inter-method scatter is {scatter_pct:.1f}%, above the 30% publication threshold."
        )
    canonical_lsf = [
        row
        for row in rows
        if row["method"] == canonical_method
        and row["row_kind"] == "method"
        and np.isclose(float(row["template_factor"]), 1.0)
    ]
    if not canonical_lsf:
        raise RuntimeError(f"H03 canonical method {canonical_method!r} has no LSF upper-limit row.")
    combined = dict(canonical_lsf[0])
    combined["row_kind"] = "combined_final"
    combined["method"] = "combined"
    combined["intermethod_scatter_pct"] = scatter_pct
    rows.append(combined)
    for row in rows:
        if row["row_kind"] == "method":
            row["intermethod_scatter_pct"] = scatter_pct

    qc = {
        "stage": "h03_upper_limits",
        "run_id": str(cfg["run_id"]),
        "prerequisites": prerequisites,
        "physical_inputs": {
            "distance_pc": physical["distance_pc"],
            "distance_err": physical["distance_err_pc"],
            "distance_source": physical["distance_source"],
            "av": physical["av"],
            "av_err": physical["av_err"],
            "av_source": physical["av_source"],
            "rv_sys_kms": physical["rv_sys_kms"],
            "rv_source": physical["rv_source"],
            "extinction_source": physical["extinction_source"],
            "a_halpha_over_av": physical["a_halpha_over_av"],
            "lacc_lha_relation": physical["lacc_lha_relation"],
            "lacc_lha_citation": physical["lacc_lha_citation"],
            "relation_scatter_dex": physical["relation_scatter_dex"],
            "lacc_aoyama21_relation": physical.get("lacc_aoyama21"),
            "companion_mass_msun": physical["companion_mass_msun"],
            "mass_source": physical["mass_source"],
            "companion_radius_rsun": physical["companion_radius_rsun"],
            "radius_source": physical["radius_source"],
        },
        "canonical_method": canonical_method,
        "canonical_source": canonical_source,
        "product_kind": prerequisites["product_kind"],
        "tail": {
            "estimator": tail_estimator,
            "estimator_source": tail_estimator_source,
            "fap_99": fap_99,
            "z_threshold_by_method": {
                row["method"]: row["z_threshold"]
                for row in rows if row["row_kind"] == "method"
                and np.isclose(float(row["template_factor"]), 1.0)
            },
        },
        "limits": [
            {
                "method": row["method"],
                "template_width": row["template_width"],
                "f_stat_99": row["f_stat_99"],
                "f_stat_5sigma_extrap": row["f_stat_5sigma_extrap"],
                "throughput": row["throughput"],
                "throughput_err": row["throughput_err"],
                "f_lim_observed": row["f_lim_observed"],
                "f_lim_dereddened": row["f_lim_dereddened"],
                "l_halpha": row["l_halpha_erg_s"],
                "mdot": row["mdot_msun_yr"],
                "mdot_aoyama21": row["mdot_aoyama21_msun_yr"],
                "l_acc_aoyama21_lsun": row["l_acc_aoyama21_lsun"],
            }
            for row in rows
        ],
        "intermethod_scatter_pct": scatter_pct,
        "open_issues": open_issues,
    }
    return StageH03Product(rows=_json_ready(rows), qc=_json_ready(qc))


def _separation_arcsec(cfg, paths):
    for key in ("h03_separation_arcsec", "separation_arcsec", "companion_separation_arcsec"):
        if cfg.get(key) is not None:
            value = float(cfg[key])
            if np.isfinite(value):
                return value, f"config.{key}"
    qc = _read_optional_json(paths.get("stage01c_qc_json"))
    if qc:
        for key in ("separation_arcsec", "companion_separation_arcsec"):
            if qc.get(key) is not None:
                value = float(qc[key])
                if np.isfinite(value):
                    return value, f"stage01c_qc.{key}"
    return np.nan, "unavailable"


def _read_literature_rows(path):
    rows = _read_csv_rows(path)
    out = []
    for row in rows:
        sep = _float_cell(row, ("separation_arcsec", "sep_arcsec", "separation"))
        lha = _float_cell(row, ("l_halpha_lsun", "L_Halpha_Lsun", "lha_lsun"))
        flux = _float_cell(row, ("l_halpha_erg_s", "L_Halpha_erg_s", "lha_erg_s"))
        label = _str_cell(row, ("citation", "source", "label"), default="literature")
        if np.isfinite(sep) and (np.isfinite(lha) or np.isfinite(flux)):
            out.append({"separation_arcsec": sep, "l_halpha_lsun": lha, "l_halpha_erg_s": flux, "label": label})
    return out


def _plot_context(rows, config, paths):
    cache_dir = Path(os.environ.get("TMPDIR", "/tmp")) / "musepipe_matplotlib"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    sep, sep_source = _separation_arcsec(config, paths)
    literature_path = Path(config.get("h03_literature_csv", paths["literature_csv"]))
    literature = _read_literature_rows(literature_path)
    fig, ax = plt.subplots(figsize=(6.5, 4.2), constrained_layout=True)
    if literature:
        lit_x = [row["separation_arcsec"] for row in literature]
        lit_y = [
            row["l_halpha_lsun"]
            if np.isfinite(row["l_halpha_lsun"])
            else row["l_halpha_erg_s"] / L_SUN_ERG_S
            for row in literature
        ]
        ax.scatter(lit_x, lit_y, marker="o", s=38, color="0.5", label="literature, unscaled")
    else:
        ax.text(0.02, 0.95, "literature table unavailable", transform=ax.transAxes, va="top", fontsize=8)
    combined = [row for row in rows if row.get("row_kind") == "combined_final"]
    if combined and np.isfinite(sep):
        ax.scatter(
            [sep],
            [float(combined[0]["l_halpha_lsun"])],
            marker="s",
            s=64,
            color="tab:red",
            label=f"this work ({combined[0]['canonical_method']})",
        )
    else:
        ax.text(0.02, 0.88, "own separation unavailable", transform=ax.transAxes, va="top", fontsize=8)
    ax.set_xlabel("Separation [arcsec]")
    ax.set_ylabel("Halpha luminosity limit [Lsun]")
    ax.set_yscale("log")
    ax.set_title("H03 Halpha upper-limit context")
    ax.text(0.02, 0.02, f"separation source: {sep_source}; literature not rescaled", transform=ax.transAxes, fontsize=7)
    ax.legend(loc="best", fontsize=8)
    paths["plot_dir"].mkdir(parents=True, exist_ok=True)
    fig.savefig(paths["context_plot"], dpi=160)
    plt.close(fig)
    return paths["context_plot"]


def write_stage_h03_products(product: StageH03Product, config, paths):
    paths["paths"].ensure_base_dirs()
    paths["plot_dir"].mkdir(parents=True, exist_ok=True)
    write_csv(paths["halpha_upper_limits_csv"], product.rows, fieldnames=TABLE_FIELDS)
    plot = _plot_context(product.rows, config, paths)
    qc = dict(product.qc)
    qc["tables"] = {"upper_limits": str(paths["halpha_upper_limits_csv"])}
    qc["figures"] = {"context": str(plot)}
    # `scripts/s7b_mdot_vs_extinction.py` ANADE `extinction_ladder` a este mismo
    # QC despues de que E3 corra, y de ahi sale el Mdot que se cita. Reescribir
    # el fichero entero lo borraba en silencio: un re-run de E3 dejaba el QC sin
    # el numero del paper y sin ninguna senal de que habia estado ahi. Se marca
    # como obsoleto (no se conserva: sus numeros vienen del E3 anterior) para
    # que quien lo lea sepa que hay que relanzar S7b.
    stale = _read_optional_json(paths["stage_h03_qc_json"]) or {}
    previous_ladder = stale.get("extinction_ladder")
    if previous_ladder is not None:
        qc["extinction_ladder_stale"] = {
            "reason": "E3 re-run; the previous extinction_ladder came from the earlier E3 output.",
            "previous_adopted_mdot_msun_yr": previous_ladder.get("adopted_mdot_msun_yr"),
            "rerun": "python scripts/s7b_mdot_vs_extinction.py --run-id <RUN_ID>",
        }
        qc.setdefault("open_issues", []).append(
            "extinction_ladder was dropped by this E3 re-run; re-run scripts/s7b_mdot_vs_extinction.py "
            "to restore the adopted Mdot limit."
        )
    write_json(paths["stage_h03_qc_json"], _json_ready(qc))
    return {"table": paths["halpha_upper_limits_csv"], "plot": plot, "qc_json": paths["stage_h03_qc_json"], "qc": qc}


def run_stage_h03(run_id=None, *, project_root=None, overrides=None, allow_run_id_mismatch=False):
    cfg = stage_h03_config_from_run(
        run_id,
        project_root=project_root,
        overrides=overrides,
        allow_run_id_mismatch=allow_run_id_mismatch,
    )
    paths = stage_h03_paths(cfg["run_id"], project_root=cfg.get("project_root"))
    product = compute_stage_h03_products(cfg, paths)
    written = write_stage_h03_products(product, cfg, paths)
    return {"config": cfg, "paths": paths, "qc": written["qc"], "written": written}


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="stage_h03_limits.py",
        description="Run Stage H03/E3 Halpha upper-limit conversion.",
    )
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--project-root", default=None)
    parser.add_argument("--allow-run-id-mismatch", action="store_true")
    args = parser.parse_args(argv)
    result = run_stage_h03(
        args.run_id,
        project_root=args.project_root,
        allow_run_id_mismatch=args.allow_run_id_mismatch,
    )
    print(result["paths"]["stage_h03_qc_json"])


__all__ = [
    "resolve_tail_estimator",
    "z_threshold_for",
    "ONE_SIDED_5SIGMA_FAP",
    "PC_CM",
    "L_SUN_ERG_S",
    "StageH03Product",
    "ccm89_a_over_av",
    "compute_stage_h03_products",
    "empirical_upper_quantile",
    "fit_gumbel_moments",
    "gumbel_isf",
    "interpolate_throughput",
    "limit_conversion_chain",
    "MAGNETOSPHERIC_FACTOR",
    "mdot_msun_yr_from_lacc",
    "physical_inputs_from_config",
    "run_stage_h03",
    "stage_h03_config_from_run",
    "stage_h03_paths",
    "tail_limit_summary",
    "template_width_label",
    "validate_prerequisites",
    "write_stage_h03_products",
]


if __name__ == "__main__":
    main()
