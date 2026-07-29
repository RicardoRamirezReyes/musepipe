"""Stage X10/D1 v4: compare extraction methods by physical observable.

Spec: docs/spec_D1_v4_codex_method_comparison.md. Continuum uses persisted
total-flux products without Halpha throughput and only continuum-preserving
methods have veto power. Lines use local-continuum-subtracted residuals with
the in-memory throughput correction. The primary statistic remains the frozen
control-centred Student t (df = n_controls - 1).
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import os
from pathlib import Path

import numpy as np

from ..config import load_run_config
from ..extraction.aperture import FLAG_BAD_WINDOW, FLAG_INTERPOLATED, FLAG_SKYLINE
from ..extraction.product import SpectrumProduct
from ..extraction.scale_check import pair_scale_check
from ..io import read_json, write_csv, write_json
from ..paths import RunPaths
from ..spectral import continuum_running_median
from ..stats import robust_sigma_axis0


SPEC_VERSION = "D1_v4"
BAD_COMPARISON_FLAGS = FLAG_BAD_WINDOW | FLAG_SKYLINE
# D1 v3 (spec_D1_v3 §0): the C5/C6 spectral-diversity methods join the frozen
# comparison set. Historical runs with spec_version="D1_v2" QC keep their
# 4-method interpretation.
METHOD_ORDER = ("aperture", "optimal_ls", "optimal_psfsub", "psffit", "sgf", "lpm")
CONTINUUM_METHODS = ("aperture", "optimal_ls", "optimal_psfsub", "psffit")
CONTINUUM_COMPARISON_MODE = "raw_total_continuum"
LINE_COMPARISON_MODE = "local_continuum_subtracted_throughput_line"
DEFAULT_LINE_CONTINUUM_WINDOW_A = 80.0
# v1 primary pairs, kept for reference/back-compat only: D1 v2 derives the
# primary pairs from the G1 method verdicts (spec v2 §3.2).
PRIMARY_PAIRS = (
    ("psffit", "aperture"),
    ("psffit", "optimal_ls"),
    ("optimal_ls", "aperture"),
)
DEFAULT_PAIRS = (
    ("psffit", "aperture"),
    ("psffit", "optimal_ls"),
    ("psffit", "optimal_psfsub"),
    ("psffit", "sgf"),
    ("psffit", "lpm"),
    ("optimal_ls", "aperture"),
    ("optimal_psfsub", "aperture"),
    ("optimal_psfsub", "optimal_ls"),
    ("optimal_psfsub", "sgf"),
    ("optimal_psfsub", "lpm"),
    ("lpm", "sgf"),
    ("lpm", "aperture"),
    ("lpm", "optimal_ls"),
    ("sgf", "aperture"),
    ("sgf", "optimal_ls"),
)
# Frozen fallback when G1 is unavailable (spec v2 §3.2) — NOT the v1 pairs.
FALLBACK_PRIMARY_PAIRS = (("psffit", "optimal_psfsub"),)
VALIDATED_VERDICTS = {"validated", "validated_with_bias"}
# Frozen thresholds (spec v2 §3.1, §4.2, §5; re-frozen by v3).
DEFAULT_P_DIVERGENT = 0.0455
DEFAULT_P_STRONG = 0.0027
DEFAULT_SCALE_GATE_SIGMA = 5.0
DEFAULT_CONTROL_GATE_ALPHA = 0.01
SCALE_BUG_LEVEL_RATIO = 10.0
# D1 v3 recommendation tree (spec v3 §2): frozen preference among G1-validated
# candidates; sgf guarded by the continuum-science flag and the Eq. 1
# self-subtraction predictor from the C5 QC.
RECOMMENDATION_PREFERENCE = ("psffit", "lpm", "optimal_psfsub", "sgf", "aperture", "optimal_ls")
DEFAULT_SGF_PREDICTOR_MAX = 0.10


@dataclass(frozen=True)
class ComparisonBand:
    name: str
    lo_A: float
    hi_A: float
    kind: str
    label: str


COMPARISON_BANDS = (
    ComparisonBand("B1", 4900.0, 5400.0, "continuum", "continuum_blue"),
    ComparisonBand("B2", 5450.0, 5750.0, "continuum", "continuum"),
    ComparisonBand("B3", 6100.0, 6400.0, "continuum", "continuum_pre_Halpha"),
    ComparisonBand("B4", 6600.0, 6800.0, "continuum", "continuum_post_Halpha"),
    ComparisonBand("B5", 7600.0, 8000.0, "continuum", "continuum_red"),
    ComparisonBand("B6", 8600.0, 9100.0, "continuum", "continuum_far_red"),
    ComparisonBand("LHa", 6553.0, 6573.0, "line", "Halpha"),
    ComparisonBand("LHb", 4851.0, 4871.0, "line", "Hbeta"),
    ComparisonBand("LOI", 8436.0, 8456.0, "line", "OI_8446"),
)

ACTION_BY_VERDICT = {
    "consistent": "conservative_branch_sufficient_no_C5_C6",
    "divergent_continuum": "iterate_C1_refine_PSF_before_PCA",
    "divergent_lines": "defer_to_E1_E2_line_artifact_tests",
    "uninterpretable": "audit_extraction_controls_before_method_choice",
}

OBJECT_FIELDS = [
    "kind",
    "pair",
    "method_i",
    "method_j",
    "band",
    "band_kind",
    "wave_min_A",
    "wave_max_A",
    "flux_i",
    "err_i",
    "flux_j",
    "err_j",
    "diff",
    "ratio_i_over_j",
    "sigma_diff_emp",
    "sigma_diff_naive",
    "z",
    "n_chan_used",
    "n_chan_flagged",
    "sigma_source",
    "n_controls",
    # D1 v2 additions (spec v2 §6): all v1 columns above are unchanged.
    "role",
    "flux_i_corr",
    "flux_j_corr",
    "diff_corr",
    "mu_ctrl",
    "s_ctrl",
    "t_stat",
    "p_value",
    "df",
    "z_neff",
    "throughput_i",
    "throughput_err_i",
    "throughput_source_i",
    "throughput_j",
    "throughput_err_j",
    "throughput_source_j",
    "scale_ok",
    "comparison_mode",
    "throughput_applied_i",
    "throughput_applied_j",
    "observable_role",
]

CONTROL_FIELDS = [
    "kind",
    "pair",
    "control_index",
    "method_i",
    "method_j",
    "band",
    "band_kind",
    "wave_min_A",
    "wave_max_A",
    "flux_i",
    "flux_j",
    "diff",
    "sigma_diff_emp",
    "z",
    "n_chan_used",
    "n_chan_flagged",
    # D1 v2 additions.
    "role",
    "diff_corr",
    "t_ctrl",
    "p_ctrl",
    "scale_ok",
    "comparison_mode",
    "throughput_applied_i",
    "throughput_applied_j",
    "observable_role",
]


@dataclass(frozen=True)
class StageX10Product:
    products: dict[str, SpectrumProduct]
    rows: list[dict]
    control_rows: list[dict]
    qc: dict


def pair_id(pair: tuple[str, str]) -> str:
    return f"{pair[0]}_vs_{pair[1]}"


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


def _running_nanmedian(values, width):
    arr = np.asarray(values, dtype=np.float64)
    width = int(width)
    if width <= 1 or arr.size == 0:
        return arr.copy()
    half = width // 2
    out = np.empty_like(arr)
    for i in range(arr.size):
        lo = max(0, i - half)
        hi = min(arr.size, i + half + 1)
        with np.errstate(invalid="ignore"):
            out[i] = np.nanmedian(arr[lo:hi])
    return out


def empirical_sigma_diff(control_i, control_j, *, smooth_channels=1, min_sigma=1.0e-12):
    """Return robust per-channel sigma of method differences from controls."""

    arr_i = np.asarray(control_i, dtype=np.float64)
    arr_j = np.asarray(control_j, dtype=np.float64)
    if arr_i.ndim != 2 or arr_j.ndim != 2:
        raise ValueError("Control spectra must have shape (n_controls, n_wave).")
    if arr_i.shape != arr_j.shape:
        raise ValueError(f"Control spectra shape mismatch: {arr_i.shape} != {arr_j.shape}.")
    if arr_i.shape[0] < 2:
        raise ValueError("At least two controls are required to estimate sigma_diff.")
    sigma = robust_sigma_axis0(arr_i - arr_j).astype(np.float64)
    sigma = _running_nanmedian(sigma, smooth_channels)
    bad = ~np.isfinite(sigma) | (sigma <= 0)
    if min_sigma is None:
        sigma[bad] = np.nan
    else:
        sigma[bad] = float(min_sigma)
    return sigma


def validate_product_set(products: dict[str, SpectrumProduct], *, required=METHOD_ORDER) -> None:
    missing = [name for name in required if name not in products]
    if missing:
        raise ValueError(f"Missing required SpectrumProducts: {missing}")
    ref_name = required[0]
    ref = products[ref_name]
    ref.validate()
    if str(ref.header.get("APERTURE", "")).lower() != "box3":
        raise ValueError("D1 requires the aperture product to use APERTURE=box3.")
    wave_ref = np.asarray(ref.wave_A, dtype=np.float64)
    if wave_ref.ndim != 1 or wave_ref.size == 0:
        raise ValueError("Reference wavelength grid is empty.")
    if wave_ref.size > 1 and np.any(np.diff(wave_ref) <= 0):
        raise ValueError("Wavelength grid must be strictly increasing.")
    wframe_ref = str(ref.header["WFRAME"])
    incubesh_ref = str(ref.header["INCUBESH"])
    run_id_ref = str(ref.header["RUNID"])
    normrad_ref = float(ref.header["NORMRAD"])

    for name in required[1:]:
        product = products[name]
        product.validate()
        wave = np.asarray(product.wave_A, dtype=np.float64)
        if wave.shape != wave_ref.shape or not np.allclose(wave, wave_ref, rtol=0.0, atol=1.0e-8):
            raise ValueError(f"{name} has a different wavelength grid than {ref_name}.")
        if str(product.header["RUNID"]) != run_id_ref:
            raise ValueError(f"{name} RUNID differs from {ref_name}.")
        if str(product.header["WFRAME"]) != wframe_ref:
            raise ValueError(f"{name} WFRAME differs from {ref_name}.")
        if str(product.header["INCUBESH"]) != incubesh_ref:
            raise ValueError(f"{name} INCUBESH differs from {ref_name}.")
        normrad = float(product.header["NORMRAD"])
        if not np.isclose(normrad, normrad_ref, rtol=0.0, atol=1.0e-6):
            raise ValueError(f"{name} NORMRAD differs from {ref_name}.")

    # Scale-convention headers (spec v2 §3.1): missing -> warning (pre-v2
    # products), present but inconsistent SCALEREF -> hard error.
    warnings_list = []
    scalerefs = {}
    for name in required:
        header = products[name].header
        if "BKGMODE" not in header or "SCALEREF" not in header:
            warnings_list.append(
                f"{name} product lacks BKGMODE/SCALEREF headers (pre-v2 extraction); "
                "flux-scale convention unverified from headers."
            )
        if "SCALEREF" in header:
            scalerefs[name] = str(header["SCALEREF"])
    if len(set(scalerefs.values())) > 1:
        raise ValueError(f"SCALEREF differs between products: {scalerefs}.")
    return warnings_list


def _channel_widths(wave_A):
    wave = np.asarray(wave_A, dtype=np.float64)
    if wave.size == 1:
        return np.ones(1, dtype=np.float64)
    edges = np.empty(wave.size + 1, dtype=np.float64)
    edges[1:-1] = 0.5 * (wave[:-1] + wave[1:])
    edges[0] = wave[0] - 0.5 * (wave[1] - wave[0])
    edges[-1] = wave[-1] + 0.5 * (wave[-1] - wave[-2])
    return np.diff(edges)


def _band_range_mask(wave_A, band: ComparisonBand):
    wave = np.asarray(wave_A, dtype=np.float64)
    return (wave >= float(band.lo_A)) & (wave <= float(band.hi_A))


def _product_band_mask(product: SpectrumProduct, band: ComparisonBand):
    flags = np.asarray(product.flags, dtype=np.int32)
    values = np.asarray(product.flux, dtype=np.float64)
    return _band_range_mask(product.wave_A, band) & np.isfinite(values) & ((flags & BAD_COMPARISON_FLAGS) == 0)


def local_continuum_residuals(product, spectra, *, window_A=DEFAULT_LINE_CONTINUUM_WINDOW_A):
    """Continuum-subtracted spectra using the frozen D1 v4 running median."""

    wave = np.asarray(product.wave_A, dtype=np.float64)
    values = np.asarray(spectra, dtype=np.float64)
    squeeze = values.ndim == 1
    if squeeze:
        values = values[None, :]
    if values.ndim != 2 or values.shape[1] != wave.size:
        raise ValueError(f"Expected spectra with shape (n, {wave.size}), got {values.shape}.")
    flags = np.asarray(product.flags, dtype=np.int32)
    base_good = np.isfinite(wave) & ((flags & BAD_COMPARISON_FLAGS) == 0)
    min_pixels = max(3, min(15, int(np.count_nonzero(base_good))))
    residuals = np.empty_like(values)
    for index, spectrum in enumerate(values):
        good = base_good & np.isfinite(spectrum)
        continuum = continuum_running_median(
            wave,
            spectrum,
            good,
            window_A=float(window_A),
            min_pixels=min_pixels,
        )
        residuals[index] = spectrum - continuum
    return residuals[0] if squeeze else residuals


def _pair_band_mask(product_i: SpectrumProduct, product_j: SpectrumProduct, band: ComparisonBand, sigma_diff):
    sigma = np.asarray(sigma_diff, dtype=np.float64)
    return (
        _product_band_mask(product_i, band)
        & _product_band_mask(product_j, band)
        & np.isfinite(sigma)
        & (sigma > 0)
    )


def _integrated_sum(wave_A, values, mask):
    mask = np.asarray(mask, dtype=bool)
    if not np.any(mask):
        return np.nan
    widths = _channel_widths(wave_A)
    vals = np.asarray(values, dtype=np.float64)
    good = mask & np.isfinite(vals) & np.isfinite(widths)
    if not np.any(good):
        return np.nan
    return float(np.nansum(vals[good] * widths[good]))


def _integrated_sigma(wave_A, sigma, mask):
    mask = np.asarray(mask, dtype=bool)
    if not np.any(mask):
        return np.nan
    widths = _channel_widths(wave_A)
    vals = np.asarray(sigma, dtype=np.float64)
    good = mask & np.isfinite(vals) & (vals > 0) & np.isfinite(widths)
    if not np.any(good):
        return np.nan
    return float(np.sqrt(np.nansum((vals[good] * widths[good]) ** 2)))


def _safe_ratio(numer, denom):
    if not np.isfinite(numer) or not np.isfinite(denom) or denom == 0:
        return np.nan
    return float(numer / denom)


def _safe_z(diff, sigma):
    if not np.isfinite(diff) or not np.isfinite(sigma) or sigma <= 0:
        return np.nan
    return float(diff / sigma)


def _student_t_p_two_sided(t_stat, df):
    if t_stat is None or df is None:
        return np.nan
    t_val = float(t_stat)
    df_val = float(df)
    if not np.isfinite(t_val) or df_val < 1:
        return np.nan
    from scipy import stats as scipy_stats

    return float(2.0 * scipy_stats.t.sf(abs(t_val), df_val))


def _binom_excess_p(n_bad, n_total, rate):
    """One-sided P of observing >= n_bad bad rows if each is bad with prob rate."""

    if n_total <= 0:
        return np.nan
    if n_bad <= 0:
        return 1.0
    from scipy import stats as scipy_stats

    return float(scipy_stats.binom.sf(int(n_bad) - 1, int(n_total), float(rate)))


def load_g1_inputs(cfg=None, paths=None) -> dict:
    """Load the G1 verdicts, throughput and covariance that D1 v2 consumes.

    Resolution order for throughput (spec v2 §3.2): g1_bias_budget.csv
    (term=throughput_loss, T = 1 + value_frac) -> stage_h04_qc.json
    per_method_at_snr5 -> T=1.0 with an open issue. Every method records the
    source actually used.
    """

    cfg = cfg or {}
    open_issues = []
    out = {
        "available": False,
        "method_verdicts": {},
        "throughput_by_method": {},
        "covariance": {"corr_length_channels": 1.0, "n_eff_over_n": 1.0, "source": "unavailable_default_1"},
        "open_issues": open_issues,
        "sources": {},
    }

    qc_path = cfg.get("x10_g1_qc_json")
    if qc_path is None and paths is not None:
        qc_path = paths.get("stage_g1_qc_json")
    if qc_path is not None and Path(qc_path).exists():
        qc = read_json(qc_path)
        verdicts = qc.get("method_verdicts") or {}
        out["method_verdicts"] = {str(k): str(v) for k, v in verdicts.items()}
        out["available"] = bool(out["method_verdicts"])
        out["sources"]["g1_qc"] = str(qc_path)
        cov = qc.get("covariance") or {}
        corr_length = cov.get("corr_length_channels_median")
        n_eff = cov.get("n_eff_over_n_median")
        if corr_length is not None and np.isfinite(float(corr_length)) and float(corr_length) >= 1.0:
            out["covariance"] = {
                "corr_length_channels": float(corr_length),
                "n_eff_over_n": float(n_eff) if n_eff is not None else None,
                "source": "stage_g1_qc.covariance",
            }
        else:
            open_issues.append("G1 covariance unavailable; z_neff uses corr_length=1.0.")
    else:
        open_issues.append(
            "G1 QC unavailable; primary pairs fall back to the frozen "
            f"{[pair_id(p) for p in FALLBACK_PRIMARY_PAIRS]} (spec v2 §3.2)."
        )

    budget_rows = []
    budget_path = cfg.get("x10_g1_bias_budget_csv")
    if budget_path is None and paths is not None:
        budget_path = paths.get("g1_bias_budget_csv")
    if budget_path is not None and Path(budget_path).exists():
        out["sources"]["g1_bias_budget"] = str(budget_path)
        with open(budget_path, newline="") as handle:
            budget_rows = list(csv.DictReader(handle))

    h04_throughput = {}
    h04_path = cfg.get("x10_h04_qc_json")
    if h04_path is None and paths is not None:
        h04_path = paths.get("stage_h04_qc_json")
    if h04_path is not None and Path(h04_path).exists():
        h04 = read_json(h04_path)
        h04_throughput = (h04.get("throughput") or {}).get("per_method_at_snr5") or {}
        if h04_throughput:
            out["sources"]["h04_qc"] = str(h04_path)

    for method in METHOD_ORDER:
        entry = None
        for row in budget_rows:
            if str(row.get("method")) != method or str(row.get("term")) != "throughput_loss":
                continue
            try:
                loss = float(row.get("value_frac"))
                err = abs(float(row.get("err_frac") or 0.0))
            except (TypeError, ValueError):
                continue
            if np.isfinite(loss):
                entry = {"T": 1.0 + loss, "err": err, "source": "g1_bias_budget.throughput_loss"}
            break
        if entry is None and method in h04_throughput:
            try:
                t_val = float(h04_throughput[method])
            except (TypeError, ValueError):
                t_val = np.nan
            if np.isfinite(t_val) and t_val > 0:
                entry = {"T": t_val, "err": 0.0, "source": "stage_h04_qc.per_method_at_snr5"}
        if entry is None:
            entry = {"T": 1.0, "err": 0.0, "source": "unavailable_default_1"}
        out["throughput_by_method"][method] = entry

    return out


def primary_pairs_from_verdicts(method_verdicts, *, pairs=DEFAULT_PAIRS):
    """Split DEFAULT_PAIRS into (primary, secondary) from the G1 verdicts.

    A pair is primary when BOTH methods are G1-validated (spec v2 §3.2).
    Without at least one such pair, fall back to the frozen
    FALLBACK_PRIMARY_PAIRS (reported by the caller as an open issue).
    """

    verdicts = method_verdicts or {}
    validated = {m for m in METHOD_ORDER if str(verdicts.get(m, "")) in VALIDATED_VERDICTS}
    primary = tuple(pair for pair in pairs if pair[0] in validated and pair[1] in validated)
    if not primary:
        primary = tuple(pair for pair in FALLBACK_PRIMARY_PAIRS if pair in pairs) or FALLBACK_PRIMARY_PAIRS
    secondary = tuple(pair for pair in pairs if pair not in primary)
    return primary, secondary


def throughput_for_comparison(g1_inputs):
    """Per-method throughput actually APPLIED by D1 (spec v2 §3.3).

    Only methods that G1 validated AND that have a bounded measured T are
    corrected; rejected methods keep T=1 with an explicit source tag so the
    secondary diagnostics stay on the raw scale.
    """

    verdicts = (g1_inputs or {}).get("method_verdicts") or {}
    raw = (g1_inputs or {}).get("throughput_by_method") or {}
    tmap = {}
    for method in METHOD_ORDER:
        entry = raw.get(method) or {"T": 1.0, "err": 0.0, "source": "unavailable_default_1"}
        t_val = float(entry.get("T", 1.0))
        validated = str(verdicts.get(method, "")) in VALIDATED_VERDICTS
        bounded = np.isfinite(t_val) and 0.0 < t_val <= 1.5
        if validated and bounded and entry.get("source") != "unavailable_default_1":
            tmap[method] = {"T": t_val, "err": float(entry.get("err", 0.0)), "source": str(entry.get("source"))}
        elif not validated:
            tmap[method] = {"T": 1.0, "err": 0.0, "source": "not_applied_rejected_method"}
        else:
            tmap[method] = {"T": 1.0, "err": 0.0, "source": "unavailable_default_1"}
    return tmap


def apply_throughput(products, controls_by_method, tmap):
    """In-memory throughput-corrected flux/control arrays (spec v2 §3.3).

    Never writes products: E3/G2 apply their own correction to the raw on-disk
    products downstream, so correcting here in memory cannot double count.
    """

    flux_corr = {}
    controls_corr = {}
    for method, product in products.items():
        t_val = float((tmap.get(method) or {}).get("T", 1.0))
        flux_corr[method] = np.asarray(product.flux, dtype=np.float64) / t_val
    for method, controls in (controls_by_method or {}).items():
        t_val = float((tmap.get(method) or {}).get("T", 1.0))
        controls_corr[method] = np.asarray(controls, dtype=np.float64) / t_val
    return flux_corr, controls_corr


def _pair_sigma_products(product_i, product_j, controls_by_method, pair, *, smooth_channels):
    wave = np.asarray(product_i.wave_A, dtype=np.float64)
    naive = np.sqrt(
        np.asarray(product_i.flux_err_emp, dtype=np.float64) ** 2
        + np.asarray(product_j.flux_err_emp, dtype=np.float64) ** 2
    )
    if controls_by_method is None or pair[0] not in controls_by_method or pair[1] not in controls_by_method:
        return {
            "sigma": np.full(wave.size, np.nan, dtype=np.float64),
            "naive": naive,
            "source": "missing_controls",
            "n_controls": 0,
        }
    control_i = np.asarray(controls_by_method[pair[0]], dtype=np.float64)
    control_j = np.asarray(controls_by_method[pair[1]], dtype=np.float64)
    if control_i.ndim != 2 or control_i.shape[1] != wave.size:
        raise ValueError(f"Controls for {pair[0]} must have shape (n_controls, {wave.size}).")
    if control_j.ndim != 2 or control_j.shape[1] != wave.size:
        raise ValueError(f"Controls for {pair[1]} must have shape (n_controls, {wave.size}).")
    return {
        "sigma": empirical_sigma_diff(control_i, control_j, smooth_channels=smooth_channels),
        "naive": naive,
        "source": "controls",
        "n_controls": int(control_i.shape[0]),
    }


def compare_pair_and_controls(product_i, product_j, controls_by_method, pair, pair_sigma, v2ctx) -> tuple[list[dict], list[dict]]:
    """Object and control rows for one pair: v1 columns + v2 t statistics.

    The primary statistic is the control-centred Student t on band-integrated
    throughput-corrected diffs (spec v2 §3.4): the per-control integration
    absorbs the channel-channel covariance and the mu centring absorbs common
    additive residuals (e.g. the psffit halo level at the control ring).
    """

    wave = np.asarray(product_i.wave_A, dtype=np.float64)
    sigma = pair_sigma["sigma"]
    naive = pair_sigma["naive"]
    scale_ok = v2ctx["scale_ok"]
    corr_length = float(v2ctx.get("corr_length_channels", 1.0) or 1.0)
    t_i = v2ctx["throughput_i"]
    t_j = v2ctx["throughput_j"]
    controls_raw_i = v2ctx.get("controls_raw_i")
    controls_raw_j = v2ctx.get("controls_raw_j")
    flux_line_corr_i = v2ctx["flux_line_corr_i"]
    flux_line_corr_j = v2ctx["flux_line_corr_j"]
    controls_line_corr_i = v2ctx.get("controls_line_corr_i")
    controls_line_corr_j = v2ctx.get("controls_line_corr_j")
    # Relative throughput error of the diff (spec v2 §3.3), added in quadrature.
    rel_terms = []
    for entry in (t_i, t_j):
        t_val = float(entry.get("T", 1.0))
        err_val = float(entry.get("err", 0.0))
        if np.isfinite(t_val) and t_val > 0 and np.isfinite(err_val) and err_val > 0:
            rel_terms.append((err_val / t_val) ** 2)
    rel_t_err = float(np.sqrt(np.sum(rel_terms))) if rel_terms else 0.0

    rows = []
    control_rows = []
    for band in COMPARISON_BANDS:
        if band.kind == "continuum":
            comparison_mode = CONTINUUM_COMPARISON_MODE
            comparison_flux_i = np.asarray(product_i.flux, dtype=np.float64)
            comparison_flux_j = np.asarray(product_j.flux, dtype=np.float64)
            controls_corr_i = controls_raw_i
            controls_corr_j = controls_raw_j
            role = v2ctx["continuum_role"]
            throughput_applied_i = False
            throughput_applied_j = False
            band_rel_t_err = 0.0
        else:
            comparison_mode = LINE_COMPARISON_MODE
            comparison_flux_i = flux_line_corr_i
            comparison_flux_j = flux_line_corr_j
            controls_corr_i = controls_line_corr_i
            controls_corr_j = controls_line_corr_j
            role = v2ctx["line_role"]
            throughput_applied_i = bool(v2ctx["throughput_applied_i"])
            throughput_applied_j = bool(v2ctx["throughput_applied_j"])
            band_rel_t_err = rel_t_err
        have_controls = (
            controls_corr_i is not None
            and controls_corr_j is not None
            and controls_corr_i.shape[0] >= 2
            and controls_corr_i.shape == controls_corr_j.shape
        )
        n_controls = int(controls_corr_i.shape[0]) if have_controls else 0
        all_band = _band_range_mask(wave, band)
        mask = _pair_band_mask(product_i, product_j, band, sigma)
        flux_i = _integrated_sum(wave, product_i.flux, mask)
        flux_j = _integrated_sum(wave, product_j.flux, mask)
        err_i = _integrated_sigma(wave, product_i.flux_err_emp, mask)
        err_j = _integrated_sigma(wave, product_j.flux_err_emp, mask)
        sigma_int = _integrated_sigma(wave, sigma, mask)
        naive_int = _integrated_sigma(wave, naive, mask)
        diff = flux_i - flux_j if np.isfinite(flux_i) and np.isfinite(flux_j) else np.nan

        band_flux_i_corr = _integrated_sum(wave, comparison_flux_i, mask)
        band_flux_j_corr = _integrated_sum(wave, comparison_flux_j, mask)
        diff_corr = (
            band_flux_i_corr - band_flux_j_corr
            if np.isfinite(band_flux_i_corr) and np.isfinite(band_flux_j_corr)
            else np.nan
        )

        # Control diffs integrated per control on the corrected scale.
        band_ctrl = []
        band_ctrl_raw = []
        if have_controls:
            for k in range(n_controls):
                mask_k = mask & np.isfinite(controls_corr_i[k]) & np.isfinite(controls_corr_j[k])
                ci = _integrated_sum(wave, controls_corr_i[k], mask_k)
                cj = _integrated_sum(wave, controls_corr_j[k], mask_k)
                dk = ci - cj if np.isfinite(ci) and np.isfinite(cj) else np.nan
                band_ctrl.append(dk)
                band_ctrl_raw.append((ci, cj, mask_k))
        ctrl_arr = np.asarray(band_ctrl, dtype=np.float64)
        finite_ctrl = ctrl_arr[np.isfinite(ctrl_arr)]
        if finite_ctrl.size >= 2:
            mu_ctrl = float(np.mean(finite_ctrl))
            s_ctrl = float(np.std(finite_ctrl, ddof=1))
            df = int(finite_ctrl.size - 1)
            denom = s_ctrl * float(np.sqrt(1.0 + 1.0 / finite_ctrl.size))
            if np.isfinite(diff_corr) and band_rel_t_err > 0:
                denom = float(np.sqrt(denom**2 + (abs(diff_corr) * band_rel_t_err) ** 2))
            if denom > 0:
                t_stat = _safe_z(diff_corr - mu_ctrl, denom)
                p_value = _student_t_p_two_sided(t_stat, df)
            elif np.isfinite(diff_corr):
                # Degenerate synthetic case: all control diffs identical.
                exact = bool(np.isclose(diff_corr, mu_ctrl, rtol=0.0, atol=1.0e-12))
                t_stat = 0.0 if exact else float(np.sign(diff_corr - mu_ctrl)) * np.inf
                p_value = 1.0 if exact else 0.0
            else:
                t_stat = np.nan
                p_value = np.nan
        else:
            mu_ctrl = np.nan
            s_ctrl = np.nan
            df = None
            t_stat = np.nan
            p_value = np.nan

        sigma_corr = (
            empirical_sigma_diff(controls_corr_i, controls_corr_j)
            if have_controls
            else None
        )
        sigma_corr_int = _integrated_sigma(wave, sigma_corr, mask) if sigma_corr is not None else np.nan
        z_neff = _safe_z(diff_corr, sigma_corr_int * np.sqrt(corr_length)) if np.isfinite(sigma_corr_int) else np.nan

        rows.append(
            {
                "kind": "object",
                "pair": pair_id(pair),
                "method_i": pair[0],
                "method_j": pair[1],
                "band": band.name,
                "band_kind": band.kind,
                "wave_min_A": float(band.lo_A),
                "wave_max_A": float(band.hi_A),
                "flux_i": _finite_or_none(flux_i),
                "err_i": _finite_or_none(err_i),
                "flux_j": _finite_or_none(flux_j),
                "err_j": _finite_or_none(err_j),
                "diff": _finite_or_none(diff),
                "ratio_i_over_j": _finite_or_none(_safe_ratio(flux_i, flux_j)),
                "sigma_diff_emp": _finite_or_none(sigma_int),
                "sigma_diff_naive": _finite_or_none(naive_int),
                "z": _finite_or_none(_safe_z(diff, sigma_int)),
                "n_chan_used": int(np.count_nonzero(mask)),
                "n_chan_flagged": int(np.count_nonzero(all_band) - np.count_nonzero(mask)),
                "sigma_source": pair_sigma["source"],
                "n_controls": int(pair_sigma["n_controls"]),
                "role": role,
                "flux_i_corr": _finite_or_none(band_flux_i_corr),
                "flux_j_corr": _finite_or_none(band_flux_j_corr),
                "diff_corr": _finite_or_none(diff_corr),
                "mu_ctrl": _finite_or_none(mu_ctrl),
                "s_ctrl": _finite_or_none(s_ctrl),
                "t_stat": _finite_or_none(t_stat),
                "p_value": _finite_or_none(p_value),
                "df": df,
                "z_neff": _finite_or_none(z_neff),
                "throughput_i": _finite_or_none(t_i.get("T")),
                "throughput_err_i": _finite_or_none(t_i.get("err")),
                "throughput_source_i": t_i.get("source"),
                "throughput_j": _finite_or_none(t_j.get("T")),
                "throughput_err_j": _finite_or_none(t_j.get("err")),
                "throughput_source_j": t_j.get("source"),
                "scale_ok": scale_ok,
                "comparison_mode": comparison_mode,
                "throughput_applied_i": throughput_applied_i,
                "throughput_applied_j": throughput_applied_j,
                "observable_role": role,
            }
        )

        if have_controls:
            raw_i = controls_raw_i
            raw_j = controls_raw_j
            for k in range(n_controls):
                _ci_corr, _cj_corr, mask_k = band_ctrl_raw[k]
                flux_i_k = _integrated_sum(wave, raw_i[k], mask_k)
                flux_j_k = _integrated_sum(wave, raw_j[k], mask_k)
                sigma_int_k = _integrated_sigma(wave, sigma, mask_k)
                diff_k = flux_i_k - flux_j_k if np.isfinite(flux_i_k) and np.isfinite(flux_j_k) else np.nan
                dk = ctrl_arr[k]
                if np.isfinite(dk) and np.isfinite(mu_ctrl) and np.isfinite(s_ctrl) and s_ctrl > 0:
                    t_ctrl = float((dk - mu_ctrl) / s_ctrl)
                    p_ctrl = _student_t_p_two_sided(t_ctrl, df)
                elif np.isfinite(dk) and np.isfinite(mu_ctrl) and s_ctrl == 0:
                    # Degenerate synthetic case: identical control diffs are
                    # perfect consistency, not a dirty control.
                    exact = bool(np.isclose(dk, mu_ctrl, rtol=0.0, atol=1.0e-12))
                    t_ctrl = 0.0 if exact else float(np.sign(dk - mu_ctrl)) * np.inf
                    p_ctrl = 1.0 if exact else 0.0
                else:
                    t_ctrl = np.nan
                    p_ctrl = np.nan
                control_rows.append(
                    {
                        "kind": "control",
                        "pair": pair_id(pair),
                        "control_index": int(k),
                        "method_i": pair[0],
                        "method_j": pair[1],
                        "band": band.name,
                        "band_kind": band.kind,
                        "wave_min_A": float(band.lo_A),
                        "wave_max_A": float(band.hi_A),
                        "flux_i": _finite_or_none(flux_i_k),
                        "flux_j": _finite_or_none(flux_j_k),
                        "diff": _finite_or_none(diff_k),
                        "sigma_diff_emp": _finite_or_none(sigma_int_k),
                        "z": _finite_or_none(_safe_z(diff_k, sigma_int_k)),
                        "n_chan_used": int(np.count_nonzero(mask_k)),
                        "n_chan_flagged": int(np.count_nonzero(all_band) - np.count_nonzero(mask_k)),
                        "role": role,
                        "diff_corr": _finite_or_none(dk),
                        "t_ctrl": _finite_or_none(t_ctrl),
                        "p_ctrl": _finite_or_none(p_ctrl),
                        "scale_ok": scale_ok,
                        "comparison_mode": comparison_mode,
                        "throughput_applied_i": throughput_applied_i,
                        "throughput_applied_j": throughput_applied_j,
                        "observable_role": role,
                    }
                )
    return rows, control_rows


def _rows_for_pairs(rows, pairs):
    pair_ids = {pair_id(pair) for pair in pairs}
    return [row for row in rows if row["pair"] in pair_ids]


def controls_clean_by_pair(
    control_rows,
    scale_checks,
    *,
    pairs=DEFAULT_PAIRS,
    p_divergent=DEFAULT_P_DIVERGENT,
    gate_alpha=DEFAULT_CONTROL_GATE_ALPHA,
):
    """Per-pair controls gate (spec v2 §4.2).

    A pair is dirty when its scale-check failed OR the fraction of bad control
    rows (p_ctrl < p_divergent or non-finite) is significantly above the
    expected ~5% (one-sided binomial excess test at gate_alpha).
    """

    out = {}
    for pair in pairs:
        pid = pair_id(pair)
        pair_rows = [row for row in control_rows if row["pair"] == pid]
        scale = (scale_checks or {}).get(pid)
        scale_ok = None if scale is None else scale.get("ok")
        if not pair_rows:
            out[pid] = {
                "clean": False,
                "reason": "missing_control_rows",
                "n_rows": 0,
                "bad_rows": 0,
                "excess_p": None,
                "max_abs_t": None,
                "scale_ok": scale_ok,
            }
            continue
        pvals = np.array(
            [np.nan if row["p_ctrl"] is None else float(row["p_ctrl"]) for row in pair_rows],
            dtype=np.float64,
        )
        tvals = np.array(
            [np.nan if row["t_ctrl"] is None else float(row["t_ctrl"]) for row in pair_rows],
            dtype=np.float64,
        )
        bad = (~np.isfinite(pvals)) | (pvals < float(p_divergent))
        n_bad = int(np.count_nonzero(bad))
        excess_p = _binom_excess_p(n_bad, pvals.size, p_divergent)
        excess = bool(np.isfinite(excess_p) and excess_p < float(gate_alpha))
        clean = (not excess) and scale_ok is not False
        if not clean:
            reason = "scale_check_failed" if scale_ok is False else "control_bad_fraction_excess"
        else:
            reason = "clean"
        out[pid] = {
            "clean": bool(clean),
            "reason": reason,
            "n_rows": int(pvals.size),
            "bad_rows": n_bad,
            "excess_p": _finite_or_none(excess_p),
            "max_abs_t": _finite_or_none(np.nanmax(np.abs(tvals)) if np.any(np.isfinite(tvals)) else np.nan),
            "scale_ok": scale_ok,
        }
    return out


def _classify_pair_rows(pair_rows, *, p_divergent, p_strong):
    """Verdict of a single pair from its object rows (p space, spec v2 §5)."""

    pvals = [row["p_value"] for row in pair_rows]
    if not pair_rows or any(p is None for p in pvals):
        return {"verdict": "insufficient", "marginal": [], "line_hits": []}
    marginal = []
    strong = False
    n_div_continuum = 0
    line_hits = []
    for row in pair_rows:
        p = float(row["p_value"])
        if row["band_kind"] == "continuum":
            if p < p_strong:
                strong = True
                n_div_continuum += 1
            elif p < p_divergent:
                n_div_continuum += 1
                marginal.append({"pair": row["pair"], "band": row["band"], "t": row["t_stat"], "p": p})
        elif row["band_kind"] == "line" and p < p_divergent:
            line_hits.append({"pair": row["pair"], "band": row["band"], "t": row["t_stat"], "p": p})
    if strong or n_div_continuum >= 2:
        return {"verdict": "divergent_continuum", "marginal": marginal, "line_hits": line_hits}
    if line_hits:
        return {"verdict": "divergent_lines", "marginal": marginal, "line_hits": line_hits}
    return {"verdict": "consistent", "marginal": marginal, "line_hits": []}


def classify_verdict_v2(
    rows,
    control_rows,
    *,
    primary_pairs,
    secondary_pairs,
    scale_checks,
    p_divergent=DEFAULT_P_DIVERGENT,
    p_strong=DEFAULT_P_STRONG,
    gate_alpha=DEFAULT_CONTROL_GATE_ALPHA,
):
    """Per-pair gates + verdict over the clean PRIMARY pairs (spec v2 §4/§5)."""

    all_pairs = tuple(primary_pairs) + tuple(secondary_pairs)
    gates = controls_clean_by_pair(
        control_rows, scale_checks, pairs=all_pairs, p_divergent=p_divergent, gate_alpha=gate_alpha
    )
    verdict_by_pair = {}
    for pair in all_pairs:
        pid = pair_id(pair)
        pair_rows = [row for row in rows if row["pair"] == pid]
        verdict_by_pair[pid] = _classify_pair_rows(pair_rows, p_divergent=p_divergent, p_strong=p_strong)
        verdict_by_pair[pid]["controls_clean"] = gates[pid]["clean"]
        verdict_by_pair[pid]["role"] = "primary" if pair in tuple(primary_pairs) else "secondary"

    degraded = {}
    active_ids = []
    for pair in primary_pairs:
        pid = pair_id(pair)
        if not gates[pid]["clean"]:
            degraded[pid] = gates[pid]["reason"]
        elif verdict_by_pair[pid]["verdict"] == "insufficient":
            degraded[pid] = "missing_object_band"
        else:
            active_ids.append(pid)

    controls_summary = {
        "reason": "clean" if not degraded else "primary_pairs_degraded",
        "by_pair": gates,
        "degraded_primary": degraded,
    }

    if not active_ids:
        return {
            "verdict": "uninterpretable",
            "action": ACTION_BY_VERDICT["uninterpretable"],
            "reason": "all_primary_pairs_degraded",
            "controls": controls_summary,
            "verdict_by_pair": verdict_by_pair,
            "pairs_degraded": degraded,
            "marginal": [],
        }

    marginal = []
    line_hits = []
    active_verdicts = []
    for pid in active_ids:
        active_verdicts.append(verdict_by_pair[pid]["verdict"])
        marginal.extend(verdict_by_pair[pid]["marginal"])
        line_hits.extend(verdict_by_pair[pid]["line_hits"])
    if "divergent_continuum" in active_verdicts:
        verdict = "divergent_continuum"
        reason = "continuum_threshold_crossed"
    elif "divergent_lines" in active_verdicts:
        verdict = "divergent_lines"
        reason = "line_threshold_crossed"
    else:
        verdict = "consistent"
        reason = "primary_pairs_within_thresholds"
    result = {
        "verdict": verdict,
        "action": ACTION_BY_VERDICT[verdict],
        "reason": reason,
        "controls": controls_summary,
        "verdict_by_pair": verdict_by_pair,
        "pairs_degraded": degraded,
        "marginal": marginal,
    }
    if line_hits:
        result["line_hits"] = line_hits
    return result


def classify_verdict_v4(
    rows,
    control_rows,
    *,
    primary_pairs_continuum,
    primary_pairs_lines,
    scale_checks,
    p_divergent=DEFAULT_P_DIVERGENT,
    p_strong=DEFAULT_P_STRONG,
    gate_alpha=DEFAULT_CONTROL_GATE_ALPHA,
):
    """D1 v4 verdict with separate continuum and line roles."""

    continuum_rows = [
        row for row in rows if row["band_kind"] == "continuum" and row["observable_role"] == "primary"
    ]
    line_rows = [row for row in rows if row["band_kind"] == "line" and row["observable_role"] == "primary"]
    continuum_controls = [
        row
        for row in control_rows
        if row["band_kind"] == "continuum" and row["observable_role"] == "primary"
    ]
    line_controls = [
        row for row in control_rows if row["band_kind"] == "line" and row["observable_role"] == "primary"
    ]
    gates_continuum = controls_clean_by_pair(
        continuum_controls,
        scale_checks,
        pairs=primary_pairs_continuum,
        p_divergent=p_divergent,
        gate_alpha=gate_alpha,
    )
    gates_lines = controls_clean_by_pair(
        line_controls,
        scale_checks,
        pairs=primary_pairs_lines,
        p_divergent=p_divergent,
        gate_alpha=gate_alpha,
    )

    verdict_by_pair = {}
    degraded_by_observable = {"continuum": {}, "lines": {}}
    active_continuum = []
    active_lines = []
    all_pairs = tuple(dict.fromkeys(tuple(primary_pairs_continuum) + tuple(primary_pairs_lines)))
    for pair in all_pairs:
        pid = pair_id(pair)
        continuum_result = None
        line_result = None
        if pair in tuple(primary_pairs_continuum):
            continuum_result = _classify_pair_rows(
                [row for row in continuum_rows if row["pair"] == pid],
                p_divergent=p_divergent,
                p_strong=p_strong,
            )
            if not gates_continuum[pid]["clean"]:
                degraded_by_observable["continuum"][pid] = gates_continuum[pid]["reason"]
            elif continuum_result["verdict"] == "insufficient":
                degraded_by_observable["continuum"][pid] = "missing_object_band"
            else:
                active_continuum.append((pid, continuum_result))
        if pair in tuple(primary_pairs_lines):
            line_result = _classify_pair_rows(
                [row for row in line_rows if row["pair"] == pid],
                p_divergent=p_divergent,
                p_strong=p_strong,
            )
            if not gates_lines[pid]["clean"]:
                degraded_by_observable["lines"][pid] = gates_lines[pid]["reason"]
            elif line_result["verdict"] == "insufficient":
                degraded_by_observable["lines"][pid] = "missing_object_band"
            else:
                active_lines.append((pid, line_result))
        pair_verdicts = [
            result["verdict"]
            for result in (continuum_result, line_result)
            if result is not None and result["verdict"] != "insufficient"
        ]
        if "divergent_continuum" in pair_verdicts:
            combined = "divergent_continuum"
        elif "divergent_lines" in pair_verdicts:
            combined = "divergent_lines"
        elif pair_verdicts:
            combined = "consistent"
        else:
            combined = "insufficient"
        verdict_by_pair[pid] = {
            "verdict": combined,
            "continuum": continuum_result,
            "lines": line_result,
            "controls_clean_continuum": None if pair not in tuple(primary_pairs_continuum) else gates_continuum[pid]["clean"],
            "controls_clean_lines": None if pair not in tuple(primary_pairs_lines) else gates_lines[pid]["clean"],
            "role": "primary",
        }

    degraded = dict(degraded_by_observable["continuum"])
    degraded.update(degraded_by_observable["lines"])
    merged_gates = {}
    for pair in all_pairs:
        pid = pair_id(pair)
        relevant = []
        if pid in gates_continuum:
            relevant.append(gates_continuum[pid])
        if pid in gates_lines:
            relevant.append(gates_lines[pid])
        merged_gates[pid] = {
            "clean": bool(relevant and all(gate["clean"] for gate in relevant)),
            "reason": "clean" if relevant and all(gate["clean"] for gate in relevant) else "observable_gate_failed",
            "scale_ok": (scale_checks.get(pid) or {}).get("ok"),
        }
    controls_summary = {
        "reason": "clean" if not degraded else "primary_pairs_degraded",
        "by_pair": merged_gates,
        "by_observable": {"continuum": gates_continuum, "lines": gates_lines},
        "degraded_primary": degraded,
    }
    if not active_continuum and not active_lines:
        return {
            "verdict": "uninterpretable",
            "action": ACTION_BY_VERDICT["uninterpretable"],
            "reason": "all_primary_observables_degraded",
            "controls": controls_summary,
            "verdict_by_pair": verdict_by_pair,
            "pairs_degraded": degraded,
            "pairs_degraded_by_observable": degraded_by_observable,
            "marginal": [],
        }

    marginal = []
    line_hits = []
    continuum_divergent = False
    for _pid, result in active_continuum:
        continuum_divergent |= result["verdict"] == "divergent_continuum"
        marginal.extend(result["marginal"])
    lines_divergent = False
    for _pid, result in active_lines:
        lines_divergent |= result["verdict"] == "divergent_lines"
        line_hits.extend(result["line_hits"])
    if continuum_divergent:
        verdict = "divergent_continuum"
        reason = "continuum_threshold_crossed"
    elif lines_divergent:
        verdict = "divergent_lines"
        reason = "line_threshold_crossed"
    else:
        verdict = "consistent"
        reason = "primary_observables_within_thresholds"
    result = {
        "verdict": verdict,
        "action": ACTION_BY_VERDICT[verdict],
        "reason": reason,
        "controls": controls_summary,
        "verdict_by_pair": verdict_by_pair,
        "pairs_degraded": degraded,
        "pairs_degraded_by_observable": degraded_by_observable,
        "marginal": marginal,
    }
    if line_hits:
        result["line_hits"] = line_hits
    return result


def _z_matrix(rows):
    matrix = {}
    for row in rows:
        matrix.setdefault(row["pair"], {})[row["band"]] = row["z"]
    return matrix


def _sigma_summary(rows):
    out = {}
    for pair in sorted({row["pair"] for row in rows}):
        vals = [row["sigma_diff_emp"] for row in rows if row["pair"] == pair and row["sigma_diff_emp"] is not None]
        out[pair] = _finite_or_none(np.nanmedian(vals) if vals else np.nan)
    return out


def _correlation_summary(products, pair_sigmas):
    out = {}
    for pair, pair_sigma in pair_sigmas.items():
        left = products[pair[0]]
        right = products[pair[1]]
        sigma = pair_sigma["sigma"]
        diff = np.asarray(left.flux, dtype=np.float64) - np.asarray(right.flux, dtype=np.float64)
        with np.errstate(divide="ignore", invalid="ignore"):
            z = diff / sigma
        high = np.isfinite(z) & (np.abs(z) >= 2.0)
        flags = np.asarray(left.flags, dtype=np.int32) | np.asarray(right.flags, dtype=np.int32)
        b2 = _band_range_mask(left.wave_A, COMPARISON_BANDS[1])
        out[pair_id(pair)] = {
            "n_high_abs_z_channels": int(np.count_nonzero(high)),
            "high_abs_z_in_B2": int(np.count_nonzero(high & b2)),
            "high_abs_z_bad_or_skyline_flag": int(np.count_nonzero(high & ((flags & BAD_COMPARISON_FLAGS) != 0))),
            "high_abs_z_interpolated_flag": int(np.count_nonzero(high & ((flags & FLAG_INTERPOLATED) != 0))),
        }
    return out


def recommend_method_v3(verdict, method_verdicts, context=None):
    """Frozen D1 v3 recommendation tree (spec_D1_v3 §2).

    Emits a recommendation ONLY on a ``consistent`` verdict (v2 behavior),
    chosen among G1-validated candidates by the frozen preference order, with
    ``sgf`` guarded by the continuum-science flag and the Eq. 1 predictor.
    Always returns the per-method caveats for the QC/checkpoint.
    """

    ctx = dict(context or {})
    method_verdicts = method_verdicts or {}
    continuum_science = bool(ctx.get("companion_continuum_is_science", True))
    predictor_max = float(ctx.get("sgf_predictor_max", DEFAULT_SGF_PREDICTOR_MAX))
    predictors = ctx.get("sgf_predictors")

    sgf_excluded = None
    if continuum_science:
        sgf_excluded = (
            "companion continuum is science: SGF loses continuum information "
            "irrecoverably (Julo et al. 2025 Sect. 2.1)."
        )
    exceeding = []
    if predictors:
        for row in predictors:
            pred = row.get("predictor")
            if row.get("in_range") and pred is not None and abs(float(pred)) > predictor_max:
                exceeding.append(str(row.get("line")))
        if exceeding and sgf_excluded is None:
            sgf_excluded = (
                f"Eq. 1 self-subtraction predictor exceeds {predictor_max:g} "
                f"for line(s) {exceeding}."
            )

    caveats = {
        "companion_continuum_is_science": continuum_science,
        "lpm": (
            "planetary-spectrum components collinear with the stellar reference "
            "are absorbed by the modulation (Julo et al. 2025 Sect. 4.1); optimal "
            "for line-dominated companions."
        ),
        "sgf": {
            "predictors": predictors if predictors is not None else "unavailable",
            "excluded_from_recommendation": sgf_excluded,
        },
    }
    candidates = [
        method
        for method in RECOMMENDATION_PREFERENCE
        if str(method_verdicts.get(method, "")) in VALIDATED_VERDICTS
    ]
    eligible = [m for m in candidates if not (m == "sgf" and sgf_excluded)]
    recommended = eligible[0] if (verdict == "consistent" and eligible) else None
    rules = {
        "preference_order": list(RECOMMENDATION_PREFERENCE),
        "candidates_validated": candidates,
        "eligible": eligible,
        "sgf_predictor_max": predictor_max,
    }
    return {"recommended_method": recommended, "method_caveats": caveats, "rules": rules}


def compare_methods(
    products: dict[str, SpectrumProduct],
    controls_by_method: dict[str, np.ndarray] | None = None,
    *,
    pairs=DEFAULT_PAIRS,
    sigma_smooth_channels=1,
    g1_inputs=None,
    primary_pairs=None,
    p_divergent=DEFAULT_P_DIVERGENT,
    p_strong=DEFAULT_P_STRONG,
    scale_gate_sigma=DEFAULT_SCALE_GATE_SIGMA,
    gate_alpha=DEFAULT_CONTROL_GATE_ALPHA,
    line_continuum_window_A=DEFAULT_LINE_CONTINUUM_WINDOW_A,
    recommendation_context=None,
) -> tuple[list[dict], list[dict], dict]:
    header_warnings = validate_product_set(products)
    open_issues = list(header_warnings or [])
    if g1_inputs is None:
        g1_inputs = load_g1_inputs()
    open_issues.extend(g1_inputs.get("open_issues") or [])

    if primary_pairs is None:
        primary, secondary = primary_pairs_from_verdicts(g1_inputs.get("method_verdicts"), pairs=pairs)
    else:
        primary = tuple(tuple(pair) for pair in primary_pairs)
        secondary = tuple(pair for pair in pairs if pair not in primary)
    primary_continuum = tuple(
        pair for pair in primary if pair[0] in CONTINUUM_METHODS and pair[1] in CONTINUUM_METHODS
    )
    primary_lines = tuple(primary)
    tmap = throughput_for_comparison(g1_inputs)
    for pair in primary:
        for method in pair:
            if tmap[method]["source"] == "unavailable_default_1":
                open_issues.append(f"Throughput unavailable for validated method {method}; T=1.0 applied.")
    controls_raw = {
        method: np.asarray(values, dtype=np.float64)
        for method, values in (controls_by_method or {}).items()
    }
    line_flux_corr = {}
    line_controls_corr = {}
    for method, product in products.items():
        t_val = float(tmap[method]["T"])
        line_flux_corr[method] = local_continuum_residuals(
            product,
            product.flux,
            window_A=float(line_continuum_window_A),
        ) / t_val
        if method in controls_raw:
            line_controls_corr[method] = local_continuum_residuals(
                product,
                controls_raw[method],
                window_A=float(line_continuum_window_A),
            ) / t_val
    cov = g1_inputs.get("covariance") or {}
    corr_length = float(cov.get("corr_length_channels") or 1.0)

    # D1 v4 scale-checks the persisted common flux scale, never Halpha throughput.
    scale_checks = {}
    for pair in pairs:
        if pair[0] in controls_raw and pair[1] in controls_raw:
            ci = controls_raw[pair[0]]
            cj = controls_raw[pair[1]]
            if ci.ndim == 2 and ci.shape == cj.shape and ci.shape[0] >= 2:
                scale_checks[pair_id(pair)] = pair_scale_check(ci, cj, gate_sigma=scale_gate_sigma)
    primary_scale = [scale_checks.get(pair_id(pair)) for pair in primary]
    if (
        primary_scale
        and all(sc is not None for sc in primary_scale)
        and all(sc.get("ok") is False for sc in primary_scale)
        and all((sc.get("level_ratio") or 0.0) > SCALE_BUG_LEVEL_RATIO for sc in primary_scale)
    ):
        raise RuntimeError(
            "All primary pairs fail the common-flux-scale check with control levels "
            f">x{SCALE_BUG_LEVEL_RATIO:g} apart: this is an extraction-stage convention bug "
            "(C2/C3/C4 must deliver 'control = object' on a common NORMRAD scale, spec v2 §3.1), "
            "not something D1 may absorb."
        )

    pair_sigmas = {}
    rows = []
    control_rows = []
    for pair in pairs:
        product_i = products[pair[0]]
        product_j = products[pair[1]]
        pair_sigma = _pair_sigma_products(
            product_i,
            product_j,
            controls_by_method,
            pair,
            smooth_channels=sigma_smooth_channels,
        )
        pair_sigmas[pair] = pair_sigma
        pid = pair_id(pair)
        source_i = str(tmap[pair[0]].get("source", ""))
        source_j = str(tmap[pair[1]].get("source", ""))
        v2ctx = {
            "continuum_role": (
                "primary"
                if pair in primary_continuum
                else "diagnostic_noncomparable_continuum"
            ),
            "line_role": "primary" if pair in primary_lines else "secondary",
            "scale_ok": scale_checks.get(pid, {}).get("ok") if pid in scale_checks else None,
            "corr_length_channels": corr_length,
            "throughput_i": tmap[pair[0]],
            "throughput_j": tmap[pair[1]],
            "throughput_applied_i": source_i not in {"unavailable_default_1", "not_applied_rejected_method"},
            "throughput_applied_j": source_j not in {"unavailable_default_1", "not_applied_rejected_method"},
            "flux_line_corr_i": line_flux_corr[pair[0]],
            "flux_line_corr_j": line_flux_corr[pair[1]],
            "controls_line_corr_i": line_controls_corr.get(pair[0]),
            "controls_line_corr_j": line_controls_corr.get(pair[1]),
            "controls_raw_i": controls_raw.get(pair[0]),
            "controls_raw_j": controls_raw.get(pair[1]),
        }
        pair_rows, pair_control_rows = compare_pair_and_controls(
            product_i, product_j, controls_by_method, pair, pair_sigma, v2ctx
        )
        rows.extend(pair_rows)
        control_rows.extend(pair_control_rows)

    verdict = classify_verdict_v4(
        rows,
        control_rows,
        primary_pairs_continuum=primary_continuum,
        primary_pairs_lines=primary_lines,
        scale_checks=scale_checks,
        p_divergent=p_divergent,
        p_strong=p_strong,
        gate_alpha=gate_alpha,
    )
    for pair in pairs:
        pid = pair_id(pair)
        if pid in verdict["verdict_by_pair"]:
            continue
        diagnostic = _classify_pair_rows(
            [row for row in rows if row["pair"] == pid],
            p_divergent=p_divergent,
            p_strong=p_strong,
        )
        verdict["verdict_by_pair"][pid] = {
            **diagnostic,
            "controls_clean": None,
            "role": "secondary",
            "continuum": None,
            "lines": None,
        }

    primary_ids = [pair_id(pair) for pair in primary]
    primary_continuum_ids = [pair_id(pair) for pair in primary_continuum]
    primary_line_ids = [pair_id(pair) for pair in primary_lines]
    primary_control_rows = [row for row in control_rows if row["observable_role"] == "primary"]
    n_primary_ctrl = len(primary_control_rows)
    n_primary_bad = sum(
        1 for row in primary_control_rows if row["p_ctrl"] is None or float(row["p_ctrl"]) < p_divergent
    )
    checks = {
        "v1_sigma_empirical_le_naive": _v1_sigma_check(rows),
        "v2_controls_clean": bool(not verdict["pairs_degraded"]),
        "v5_no_double_throughput": {
            "ok": True,
            "note": (
                "Throughput correction is internal to the D1 comparison (in-memory only); "
                "D1 v4 applies it only to local-continuum-subtracted line bands; "
                "on-disk products remain uncorrected and E3/G2 apply their own correction "
                "downstream (stage_h03_limits / stage_g2_measure_lines)."
            ),
        },
        "v6_scale_check_ok": bool(
            all(gate.get("scale_ok") is not False for gate in verdict["controls"]["by_pair"].values())
        ),
        "v7_t_calibration": {
            "primary_control_rows": n_primary_ctrl,
            "bad_fraction": _finite_or_none(n_primary_bad / n_primary_ctrl if n_primary_ctrl else np.nan),
            "expected_fraction": float(p_divergent),
        },
    }
    missing = sorted({row["pair"] for row in rows if row["sigma_source"] == "missing_controls"})
    if missing:
        open_issues.append(
            "Control spectra missing for pairs: "
            + ", ".join(missing)
            + ". Those pairs are degraded until same-radius controls are supplied."
        )
    if not checks["v1_sigma_empirical_le_naive"]["ok"]:
        open_issues.append("At least one empirical sigma_diff is larger than the naive independent-error scale.")
    for pid, reason in verdict["pairs_degraded"].items():
        open_issues.append(f"Primary pair {pid} degraded: {reason}.")

    n_controls = max((sigma["n_controls"] for sigma in pair_sigmas.values()), default=0)
    recommendation = recommend_method_v3(
        verdict["verdict"], g1_inputs.get("method_verdicts") or {}, recommendation_context
    )
    recommended = recommendation["recommended_method"]
    qc = {
        "stage": "x10_method_comparison",
        "spec_version": SPEC_VERSION,
        "verdict": verdict["verdict"],
        "action": verdict["action"],
        "reason": verdict["reason"],
        "bands": [
            {"name": band.name, "lo_A": band.lo_A, "hi_A": band.hi_A, "kind": band.kind, "label": band.label}
            for band in COMPARISON_BANDS
        ],
        "primary_pairs": primary_ids,
        "primary_pairs_continuum": primary_continuum_ids,
        "primary_pairs_lines": primary_line_ids,
        "secondary_pairs": [pair_id(pair) for pair in secondary],
        "pairs": [pair_id(pair) for pair in pairs],
        "z_matrix": _z_matrix(rows),
        "t_matrix": _t_matrix(rows),
        "sigma_diff_median_by_pair": _sigma_summary(rows),
        "controls": verdict["controls"],
        "verdict_by_pair": verdict["verdict_by_pair"],
        "pairs_degraded": verdict["pairs_degraded"],
        "pairs_degraded_by_observable": verdict.get("pairs_degraded_by_observable", {}),
        "g1_inputs": {
            "available": bool(g1_inputs.get("available")),
            "method_verdicts": g1_inputs.get("method_verdicts") or {},
            "covariance": cov,
            "sources": g1_inputs.get("sources") or {},
        },
        "throughput_correction": {
            "applied_in_memory": "line_bands_only",
            "by_method": tmap,
            "note": checks["v5_no_double_throughput"]["note"],
        },
        "scale_check": scale_checks,
        "statistics": {
            "kind": "t_control_centred",
            "n_controls": int(n_controls),
            "p_divergent": float(p_divergent),
            "p_strong": float(p_strong),
            "gate_alpha": float(gate_alpha),
            "scale_gate_sigma": float(scale_gate_sigma),
            "corr_length_channels": corr_length,
        },
        "comparison_policy": {
            "continuum_methods": list(CONTINUUM_METHODS),
            "continuum_mode": CONTINUUM_COMPARISON_MODE,
            "line_mode": LINE_COMPARISON_MODE,
            "line_continuum_window_A": float(line_continuum_window_A),
        },
        "excluded_from_continuum_verdict": {
            "sgf": "continuum removed by construction",
            "lpm": "collinear continuum can be absorbed",
        },
        "recommended_method": recommended,
        "method_caveats": recommendation["method_caveats"],
        "recommendation_rules": recommendation["rules"],
        "checks": checks,
        "correlations": _correlation_summary(products, pair_sigmas),
        "marginal": verdict.get("marginal", []),
        "open_issues": open_issues,
    }
    if "line_hits" in verdict:
        qc["line_hits"] = verdict["line_hits"]
    return rows, control_rows, _json_ready(qc)


def _as_control_array(value):
    if value is None:
        return None
    return np.asarray(value, dtype=np.float64)


def _t_matrix(rows):
    matrix = {}
    for row in rows:
        matrix.setdefault(row["pair"], {})[row["band"]] = row["t_stat"]
    return matrix


def _v1_sigma_check(rows):
    checked = []
    failed = []
    for row in rows:
        if row["sigma_source"] != "controls":
            continue
        emp = row["sigma_diff_emp"]
        naive = row["sigma_diff_naive"]
        if emp is None or naive is None:
            continue
        item = {"pair": row["pair"], "band": row["band"], "empirical": emp, "naive": naive}
        checked.append(item)
        if emp > naive * 1.000001:
            failed.append(item)
    return {"ok": len(failed) == 0 if checked else False, "n_checked": len(checked), "failed": failed}


def stage_x10_paths(run_id, project_root=None):
    root = Path(project_root or Path.cwd()).resolve()
    paths = RunPaths.from_project_root(run_id, root)
    return {
        "paths": paths,
        "spec_aperture_object": paths.stage_dir / "spec_aperture_object.fits",
        "spec_optimal_object": paths.stage_dir / "spec_optimal_object.fits",
        "spec_optimal_psfsub_object": paths.stage_dir / "spec_optimal_psfsub_object.fits",
        "spec_psffit_object": paths.stage_dir / "spec_psffit_object.fits",
        "spec_sgf_object": paths.stage_dir / "spec_sgf_object.fits",
        "spec_lpm_object": paths.stage_dir / "spec_lpm_object.fits",
        "controls_aperture_npz": paths.stage_dir / "spec_aperture_controls.npz",
        "controls_optimal_ls_npz": paths.stage_dir / "spec_optimal_controls.npz",
        "controls_optimal_psfsub_npz": paths.stage_dir / "spec_optimal_psfsub_controls.npz",
        "controls_psffit_npz": paths.stage_dir / "spec_psffit_controls.npz",
        "controls_sgf_npz": paths.stage_dir / "spec_sgf_controls.npz",
        "controls_lpm_npz": paths.stage_dir / "spec_lpm_controls.npz",
        "spec_sgf_qc_json": paths.stage_dir / "spec_sgf_qc.json",
        "method_comparison_csv": paths.table_dir / "method_comparison.csv",
        "method_comparison_controls_csv": paths.table_dir / "method_comparison_controls.csv",
        "stage_x10_qc_json": paths.stage_dir / "stage_x10_qc.json",
        "stage_x10_overview_png": paths.plot_dir / "stage_x10_overview.png",
        "stage_g1_qc_json": paths.stage_dir / "stage_g1_qc.json",
        "g1_bias_budget_csv": paths.table_dir / "g1_bias_budget.csv",
        "stage_h04_qc_json": paths.stage_dir / "stage_h04_qc.json",
    }


def stage_x10_config_from_run(
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
    cfg.setdefault("x10_sigma_smooth_channels", 5)
    cfg.setdefault("x10_control_spectra", {})
    # D1 v2 frozen thresholds (spec v2); overrides = spec revision, document it.
    cfg.setdefault("x10_primary_pairs", "from_g1")
    cfg.setdefault("x10_p_divergent", DEFAULT_P_DIVERGENT)
    cfg.setdefault("x10_p_strong", DEFAULT_P_STRONG)
    cfg.setdefault("x10_scale_gate_sigma", DEFAULT_SCALE_GATE_SIGMA)
    cfg.setdefault("x10_control_gate_alpha", DEFAULT_CONTROL_GATE_ALPHA)
    cfg.setdefault("x10_line_continuum_window_A", DEFAULT_LINE_CONTINUUM_WINDOW_A)
    # D1 v3 recommendation tree inputs (spec v3 §1-§2).
    cfg.setdefault("companion_continuum_is_science", True)
    cfg.setdefault("x10_sgf_predictor_max", DEFAULT_SGF_PREDICTOR_MAX)
    return cfg


def _product_paths_from_config(cfg, paths):
    return {
        "aperture": Path(cfg.get("x10_spec_aperture_object", paths["spec_aperture_object"])),
        "optimal_ls": Path(cfg.get("x10_spec_optimal_object", paths["spec_optimal_object"])),
        "optimal_psfsub": Path(cfg.get("x10_spec_optimal_psfsub_object", paths["spec_optimal_psfsub_object"])),
        "psffit": Path(cfg.get("x10_spec_psffit_object", paths["spec_psffit_object"])),
        "sgf": Path(cfg.get("x10_spec_sgf_object", paths["spec_sgf_object"])),
        "lpm": Path(cfg.get("x10_spec_lpm_object", paths["spec_lpm_object"])),
    }


def load_method_products(product_paths: dict[str, str | Path]) -> dict[str, SpectrumProduct]:
    products = {}
    for method in METHOD_ORDER:
        path = Path(product_paths[method])
        if not path.exists():
            raise FileNotFoundError(path)
        products[method] = SpectrumProduct.read(path)
    validate_product_set(products)
    return products


def _default_control_paths(paths):
    return {
        "aperture": paths["controls_aperture_npz"],
        "optimal_ls": paths["controls_optimal_ls_npz"],
        "optimal_psfsub": paths["controls_optimal_psfsub_npz"],
        "psffit": paths["controls_psffit_npz"],
        "sgf": paths["controls_sgf_npz"],
        "lpm": paths["controls_lpm_npz"],
    }


def _load_control_npz(path):
    with np.load(path) as data:
        for key in ("control_spectra", "controls", "spectra", "flux"):
            if key in data:
                return np.asarray(data[key], dtype=np.float64)
        array_keys = [key for key in data.files if np.asarray(data[key]).ndim == 2]
        if len(array_keys) == 1:
            return np.asarray(data[array_keys[0]], dtype=np.float64)
    raise ValueError(f"Could not find a 2D control spectra array in {path}.")


def load_control_spectra(control_paths: dict[str, str | Path]) -> dict[str, np.ndarray]:
    controls = {}
    for method, path in control_paths.items():
        path = Path(path)
        if path.exists():
            controls[method] = _load_control_npz(path)
    return controls


def _control_paths_from_config(cfg, paths):
    control_paths = _default_control_paths(paths)
    configured = cfg.get("x10_control_spectra") or {}
    for method, path in configured.items():
        if method in control_paths and path:
            control_paths[method] = Path(path)
    return control_paths


def compute_stage_x10_products(config, paths=None) -> StageX10Product:
    cfg = dict(config)
    root = Path(cfg.get("project_root") or Path.cwd()).resolve()
    paths = stage_x10_paths(cfg["run_id"], root) if paths is None else paths
    products = load_method_products(_product_paths_from_config(cfg, paths))
    controls = load_control_spectra(_control_paths_from_config(cfg, paths))
    g1_inputs = load_g1_inputs(cfg, paths)
    primary_override = cfg.get("x10_primary_pairs", "from_g1")
    primary_pairs = None
    if primary_override and primary_override != "from_g1":
        primary_pairs = tuple(tuple(pair) for pair in primary_override)
    sgf_qc_path = Path(cfg.get("x10_spec_sgf_qc_json", paths["spec_sgf_qc_json"]))
    sgf_predictors = None
    if sgf_qc_path.exists():
        sgf_predictors = (read_json(sgf_qc_path) or {}).get("self_subtraction_predictor")
    recommendation_context = {
        "companion_continuum_is_science": cfg.get("companion_continuum_is_science", True),
        "sgf_predictor_max": cfg.get("x10_sgf_predictor_max", DEFAULT_SGF_PREDICTOR_MAX),
        "sgf_predictors": sgf_predictors,
    }
    rows, control_rows, qc = compare_methods(
        products,
        controls,
        sigma_smooth_channels=int(cfg.get("x10_sigma_smooth_channels", 5)),
        g1_inputs=g1_inputs,
        primary_pairs=primary_pairs,
        p_divergent=float(cfg.get("x10_p_divergent", DEFAULT_P_DIVERGENT)),
        p_strong=float(cfg.get("x10_p_strong", DEFAULT_P_STRONG)),
        scale_gate_sigma=float(cfg.get("x10_scale_gate_sigma", DEFAULT_SCALE_GATE_SIGMA)),
        gate_alpha=float(cfg.get("x10_control_gate_alpha", DEFAULT_CONTROL_GATE_ALPHA)),
        line_continuum_window_A=float(
            cfg.get("x10_line_continuum_window_A", DEFAULT_LINE_CONTINUUM_WINDOW_A)
        ),
        recommendation_context=recommendation_context,
    )
    qc["run_id"] = str(cfg["run_id"])
    qc["products"] = {method: str(path) for method, path in _product_paths_from_config(cfg, paths).items()}
    qc["control_products"] = {
        method: str(path) for method, path in _control_paths_from_config(cfg, paths).items() if Path(path).exists()
    }
    return StageX10Product(products=products, rows=rows, control_rows=control_rows, qc=_json_ready(qc))


def write_stage_x10_overview(products, rows, path, *, primary_pair_ids=None):
    cache_dir = Path(os.environ.get("TMPDIR", "/tmp")) / "musepipe_matplotlib"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))

    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(3, 1, figsize=(11, 8), constrained_layout=True)
    ref = products["psffit"]
    wave = np.asarray(ref.wave_A, dtype=np.float64)
    for method in METHOD_ORDER:
        product = products[method]
        axes[0].plot(product.wave_A, product.flux, lw=1.0, label=method)
    axes[0].set_ylabel("flux")
    axes[0].legend(loc="best", fontsize=8)
    for method in METHOD_ORDER:
        if method == "psffit":
            continue
        ratio = np.asarray(products[method].flux, dtype=np.float64) / np.asarray(ref.flux, dtype=np.float64)
        axes[1].plot(wave, ratio, lw=1.0, label=f"{method}/psffit")
    axes[1].axhline(1.0, color="0.3", lw=0.8)
    axes[1].set_ylabel("ratio")
    axes[1].legend(loc="best", fontsize=8)
    if primary_pair_ids is None:
        primary_pair_ids = [pair_id(pair) for pair in FALLBACK_PRIMARY_PAIRS]
    primary_pair_ids = list(primary_pair_ids)
    heat = np.full((len(primary_pair_ids), len(COMPARISON_BANDS)), np.nan, dtype=np.float64)
    row_by_pair_band = {(row["pair"], row["band"]): row for row in rows}
    for i, pid in enumerate(primary_pair_ids):
        for j, band in enumerate(COMPARISON_BANDS):
            row = row_by_pair_band.get((pid, band.name))
            if row is None:
                continue
            value = row.get("t_stat") if row.get("t_stat") is not None else row.get("z")
            if value is not None:
                heat[i, j] = float(value)
    vmax = np.nanmax(np.abs(heat)) if np.any(np.isfinite(heat)) else 1.0
    vmax = max(2.0, float(vmax))
    im = axes[2].imshow(heat, aspect="auto", cmap="coolwarm", vmin=-vmax, vmax=vmax)
    axes[2].set_yticks(np.arange(len(primary_pair_ids)), primary_pair_ids)
    axes[2].set_xticks(np.arange(len(COMPARISON_BANDS)), [band.name for band in COMPARISON_BANDS])
    axes[2].set_ylabel("primary pair")
    axes[2].set_xlabel("band")
    fig.colorbar(im, ax=axes[2], label="t (control-centred)")
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def write_stage_x10_products(product: StageX10Product, config, paths):
    paths["paths"].ensure_base_dirs()
    write_csv(paths["method_comparison_csv"], product.rows, fieldnames=OBJECT_FIELDS)
    write_csv(paths["method_comparison_controls_csv"], product.control_rows, fieldnames=CONTROL_FIELDS)
    overview = write_stage_x10_overview(
        product.products,
        product.rows,
        paths["stage_x10_overview_png"],
        primary_pair_ids=product.qc.get("primary_pairs"),
    )
    qc = dict(product.qc)
    qc["tables"] = {
        "object": str(paths["method_comparison_csv"]),
        "controls": str(paths["method_comparison_controls_csv"]),
    }
    qc["plots"] = {"overview": str(overview)}
    write_json(paths["stage_x10_qc_json"], _json_ready(qc))
    return {
        "tables": qc["tables"],
        "plots": qc["plots"],
        "qc_json": paths["stage_x10_qc_json"],
        "qc": qc,
    }


def run_stage_x10(run_id=None, *, project_root=None, overrides=None, allow_run_id_mismatch=False):
    cfg = stage_x10_config_from_run(
        run_id,
        project_root=project_root,
        overrides=overrides,
        allow_run_id_mismatch=allow_run_id_mismatch,
    )
    paths = stage_x10_paths(cfg["run_id"], project_root=cfg.get("project_root"))
    product = compute_stage_x10_products(cfg, paths)
    written = write_stage_x10_products(product, cfg, paths)
    return {"config": cfg, "paths": paths, "qc": written["qc"], "written": written}


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="stage_x10_compare.py",
        description="Run Stage X10/D1 extraction method comparison.",
    )
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--project-root", default=None)
    parser.add_argument("--allow-run-id-mismatch", action="store_true")
    args = parser.parse_args(argv)
    result = run_stage_x10(
        args.run_id,
        project_root=args.project_root,
        allow_run_id_mismatch=args.allow_run_id_mismatch,
    )
    print(result["paths"]["stage_x10_qc_json"])


__all__ = [
    "COMPARISON_BANDS",
    "CONTINUUM_METHODS",
    "DEFAULT_PAIRS",
    "FALLBACK_PRIMARY_PAIRS",
    "METHOD_ORDER",
    "PRIMARY_PAIRS",
    "RECOMMENDATION_PREFERENCE",
    "SPEC_VERSION",
    "recommend_method_v3",
    "StageX10Product",
    "apply_throughput",
    "classify_verdict_v2",
    "classify_verdict_v4",
    "compare_methods",
    "compare_pair_and_controls",
    "compute_stage_x10_products",
    "controls_clean_by_pair",
    "empirical_sigma_diff",
    "load_control_spectra",
    "load_g1_inputs",
    "load_method_products",
    "local_continuum_residuals",
    "pair_id",
    "primary_pairs_from_verdicts",
    "run_stage_x10",
    "stage_x10_config_from_run",
    "stage_x10_paths",
    "throughput_for_comparison",
    "validate_product_set",
    "write_stage_x10_products",
]


if __name__ == "__main__":
    main()
