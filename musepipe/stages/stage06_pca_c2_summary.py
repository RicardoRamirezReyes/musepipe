"""Consolidate the 40x40 multi-line PCA C2 injection products."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from ..config import load_run_config
from ..io import read_json, write_csv, write_json
from ..paths import RunPaths


STAGE_NAME = "stage06_pca_c2_40"
DETECTION_THRESHOLD_SNR = 5.0


def stage06_pca_c2_summary_paths(run_id, project_root=None):
    """Return input and consolidated-output paths for one run."""

    run_paths = RunPaths.from_project_root(run_id, project_root=project_root)
    table_root = run_paths.table_dir / STAGE_NAME
    plot_root = run_paths.plot_dir / STAGE_NAME
    return {
        "table_root": table_root,
        "plot_root": plot_root,
        "grid_csv": table_root / f"{STAGE_NAME}_grid.csv",
        "case_summary_csv": table_root / f"{STAGE_NAME}_case_summary.csv",
        "completeness_csv": table_root / f"{STAGE_NAME}_completeness.csv",
        "qc_json": run_paths.stage_dir / f"{STAGE_NAME}_summary_qc.json",
        "summary_plot": plot_root / f"{STAGE_NAME}_summary.png",
    }


def _read_csv_rows(path):
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"PCA C2 recovery CSV is empty: {path}")
    numeric_rows = []
    for row in rows:
        converted = {}
        for key, value in row.items():
            try:
                converted[key] = float(value)
            except (TypeError, ValueError):
                converted[key] = value
        numeric_rows.append(converted)
    return numeric_rows


def load_pca_c2_cases(table_root, expected_n_cases=9):
    """Load and validate the case-level notebook products."""

    root = Path(table_root)
    qc_paths = sorted(root.glob("*/*_qc.json"))
    cases = []
    for qc_path in qc_paths:
        qc = read_json(qc_path)
        case_id = str(qc.get("case_id") or qc_path.parent.name)
        recovery_path = qc_path.parent / f"{case_id}_recovery.csv"
        if not recovery_path.exists():
            raise FileNotFoundError(f"Missing recovery CSV for {case_id}: {recovery_path}")
        rows = _read_csv_rows(recovery_path)
        csv_modes = {
            str(row["recovery_mode"]).strip().lower()
            for row in rows
            if row.get("recovery_mode") not in (None, "")
        }
        recovery_modes = csv_modes or {str(qc.get("recovery_mode", "")).strip().lower()}
        if recovery_modes != {"realistic"}:
            raise ValueError(f"Case {case_id} is not a realistic recovery run.")
        cases.append(
            {
                "case_id": case_id,
                "line_label": str(qc["line_label"]),
                "line_center_A": float(qc["line_center_A"]),
                "pa_deg": float(qc["actual_injection_pa_deg"]),
                "injection_y": int(qc["injection_yx"][0]),
                "injection_x": int(qc["injection_yx"][1]),
                "qc": qc,
                "qc_path": qc_path,
                "recovery_path": recovery_path,
                "rows": rows,
            }
        )
    if expected_n_cases is not None and len(cases) != int(expected_n_cases):
        raise ValueError(f"Expected {expected_n_cases} PCA C2 cases, found {len(cases)} in {root}.")
    identities = {(case["line_label"], case["pa_deg"]) for case in cases}
    if len(identities) != len(cases):
        raise ValueError("PCA C2 contains duplicate line/PA cases.")
    grids = {
        tuple(float(row["target_snr_input"]) for row in case["rows"])
        for case in cases
    }
    if len(grids) != 1:
        raise ValueError("PCA C2 cases do not share a common input S/N grid.")
    return cases


def _finite_percentiles(values):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return (np.nan, np.nan, np.nan)
    return tuple(float(value) for value in np.percentile(values, [16, 50, 84]))


def _input_at_level(x_values, y_values, target):
    """Interpolate the input S/N at which a monotonic envelope reaches target."""

    x = np.asarray(x_values, dtype=np.float64)
    y = np.asarray(y_values, dtype=np.float64)
    finite = np.isfinite(x) & np.isfinite(y)
    if not np.any(finite):
        return np.nan
    order = np.argsort(x[finite])
    x = x[finite][order]
    y = np.maximum.accumulate(y[finite][order])
    indices = np.where(y >= float(target))[0]
    if indices.size == 0:
        return np.nan
    index = int(indices[0])
    if index == 0 or y[index] == y[index - 1]:
        return float(x[index])
    fraction = (float(target) - y[index - 1]) / (y[index] - y[index - 1])
    return float(x[index - 1] + fraction * (x[index] - x[index - 1]))


def _case_row_at_input(case, target_snr):
    return min(
        case["rows"],
        key=lambda row: abs(float(row["target_snr_input"]) - float(target_snr)),
    )


def _completeness_rows(cases, detection_threshold_snr):
    target_grid = sorted({float(row["target_snr_input"]) for row in cases[0]["rows"]})
    groups = [("ALL", cases)]
    for line_label in sorted({case["line_label"] for case in cases}):
        groups.append((line_label, [case for case in cases if case["line_label"] == line_label]))
    output = []
    for group_label, group_cases in groups:
        for target_snr in target_grid:
            rows = [_case_row_at_input(case, target_snr) for case in group_cases]
            matched = [float(row["realistic_matched_snr"]) for row in rows]
            aperture = [float(row["realistic_aperture_snr"]) for row in rows]
            matched_p16, matched_p50, matched_p84 = _finite_percentiles(matched)
            aperture_p16, aperture_p50, aperture_p84 = _finite_percentiles(aperture)
            output.append(
                {
                    "line_group": group_label,
                    "target_snr_input": target_snr,
                    "n_trials": len(rows),
                    "detection_threshold_snr": float(detection_threshold_snr),
                    "matched_completeness": float(
                        np.mean(np.asarray(matched) >= float(detection_threshold_snr))
                    ),
                    "aperture_completeness": float(
                        np.mean(np.asarray(aperture) >= float(detection_threshold_snr))
                    ),
                    "matched_recovered_snr_p16": matched_p16,
                    "matched_recovered_snr_median": matched_p50,
                    "matched_recovered_snr_p84": matched_p84,
                    "aperture_recovered_snr_p16": aperture_p16,
                    "aperture_recovered_snr_median": aperture_p50,
                    "aperture_recovered_snr_p84": aperture_p84,
                }
            )
    return output


def summarize_pca_c2_cases(cases, detection_threshold_snr=DETECTION_THRESHOLD_SNR):
    """Build consolidated grid, case, completeness, and QC-ready summaries."""

    if not cases:
        raise ValueError("At least one PCA C2 case is required.")
    grid_rows = []
    case_rows = []
    for case in cases:
        positive = [row for row in case["rows"] if float(row["target_snr_input"]) > 0]
        matched_transfer = [float(row["delta_matched_signal_transfer"]) for row in positive]
        aperture_transfer = [float(row["delta_aperture_signal_transfer"]) for row in positive]
        mt16, mt50, mt84 = _finite_percentiles(matched_transfer)
        at16, at50, at84 = _finite_percentiles(aperture_transfer)
        baseline = _case_row_at_input(case, 0.0)
        max_input = max(float(row["target_snr_input"]) for row in case["rows"])
        high = _case_row_at_input(case, max_input)
        x_values = [float(row["target_snr_input"]) for row in case["rows"]]
        matched_values = [float(row["realistic_matched_snr"]) for row in case["rows"]]
        aperture_values = [float(row["realistic_aperture_snr"]) for row in case["rows"]]
        case_rows.append(
            {
                "case_id": case["case_id"],
                "line_label": case["line_label"],
                "line_center_A": case["line_center_A"],
                "pa_deg": case["pa_deg"],
                "injection_y": case["injection_y"],
                "injection_x": case["injection_x"],
                "baseline_matched_snr": float(baseline["realistic_matched_snr"]),
                "baseline_aperture_snr": float(baseline["realistic_aperture_snr"]),
                "max_target_snr_input": max_input,
                "max_input_matched_snr": float(high["realistic_matched_snr"]),
                "max_input_aperture_snr": float(high["realistic_aperture_snr"]),
                "input_snr_at_matched_detection": _input_at_level(
                    x_values, matched_values, detection_threshold_snr
                ),
                "input_snr_at_aperture_detection": _input_at_level(
                    x_values, aperture_values, detection_threshold_snr
                ),
                "delta_matched_transfer_p16": mt16,
                "delta_matched_transfer_median": mt50,
                "delta_matched_transfer_p84": mt84,
                "delta_aperture_transfer_p16": at16,
                "delta_aperture_transfer_median": at50,
                "delta_aperture_transfer_p84": at84,
            }
        )
        for row in case["rows"]:
            grid_rows.append(
                {
                    "case_id": case["case_id"],
                    "line_label": case["line_label"],
                    "line_center_A": case["line_center_A"],
                    "pa_deg": case["pa_deg"],
                    "injection_y": case["injection_y"],
                    "injection_x": case["injection_x"],
                    **row,
                }
            )
    completeness_rows = _completeness_rows(cases, detection_threshold_snr)
    return {
        "grid_rows": grid_rows,
        "case_rows": case_rows,
        "completeness_rows": completeness_rows,
    }


def _completeness_thresholds(completeness_rows):
    output = {}
    for group in sorted({row["line_group"] for row in completeness_rows}):
        rows = [row for row in completeness_rows if row["line_group"] == group]
        x = [row["target_snr_input"] for row in rows]
        values = {
            "matched_50pct": _input_at_level(x, [row["matched_completeness"] for row in rows], 0.5),
            "matched_90pct": _input_at_level(x, [row["matched_completeness"] for row in rows], 0.9),
            "aperture_50pct": _input_at_level(x, [row["aperture_completeness"] for row in rows], 0.5),
            "aperture_90pct": _input_at_level(x, [row["aperture_completeness"] for row in rows], 0.9),
        }
        output[group] = {
            key: (float(value) if np.isfinite(value) else None) for key, value in values.items()
        }
    return output


def save_pca_c2_summary_plot(products, path, detection_threshold_snr=DETECTION_THRESHOLD_SNR):
    """Save the consolidated recovery, completeness, and throughput figure."""

    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    colors = {"Hbeta": "tab:blue", "Halpha": "tab:red", "OI_8446": "tab:green"}
    grid_rows = products["grid_rows"]
    case_rows = products["case_rows"]
    figure, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    for case_id in sorted({row["case_id"] for row in grid_rows}):
        rows = [row for row in grid_rows if row["case_id"] == case_id]
        rows.sort(key=lambda row: float(row["target_snr_input"]))
        color = colors.get(rows[0]["line_label"], "0.4")
        x = [row["target_snr_input"] for row in rows]
        axes[0, 0].plot(x, [row["realistic_matched_snr"] for row in rows], color=color, alpha=0.45)
        axes[0, 1].plot(x, [row["realistic_aperture_snr"] for row in rows], color=color, alpha=0.45)
    for axis, title in zip(axes[0], ["Matched-filter recovery", "3x3 aperture recovery"]):
        axis.axhline(detection_threshold_snr, color="black", linestyle="--", linewidth=1)
        axis.set(xlabel="Injected S/N", ylabel="Recovered S/N", title=title)
        axis.grid(alpha=0.2)

    all_completeness = [
        row for row in products["completeness_rows"] if row["line_group"] == "ALL"
    ]
    axes[1, 0].plot(
        [row["target_snr_input"] for row in all_completeness],
        [row["matched_completeness"] for row in all_completeness],
        "o-",
        label="Matched filter",
    )
    axes[1, 0].plot(
        [row["target_snr_input"] for row in all_completeness],
        [row["aperture_completeness"] for row in all_completeness],
        "s-",
        label="3x3 aperture",
    )
    axes[1, 0].set(
        xlabel="Injected S/N",
        ylabel=f"Fraction recovered >= {detection_threshold_snr:g} sigma",
        ylim=(-0.03, 1.03),
        title="Completeness across 9 line/PA cases",
    )
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.2)

    labels = sorted({row["line_label"] for row in case_rows})
    for index, label in enumerate(labels):
        rows = sorted(
            (row for row in case_rows if row["line_label"] == label),
            key=lambda row: row["pa_deg"],
        )
        jitter = np.linspace(-0.035, 0.035, len(rows))
        axes[1, 1].scatter(
            np.full(len(rows), index) - 0.08 + jitter,
            [row["delta_matched_transfer_median"] for row in rows],
            color=colors.get(label, "0.4"),
            marker="o",
            label="Matched" if index == 0 else None,
        )
        axes[1, 1].scatter(
            np.full(len(rows), index) + 0.08 + jitter,
            [row["delta_aperture_transfer_median"] for row in rows],
            color=colors.get(label, "0.4"),
            marker="x",
            label="3x3 aperture" if index == 0 else None,
        )
    axes[1, 1].axhline(1.0, color="black", linestyle="--", linewidth=1)
    axes[1, 1].set_xticks(range(len(labels)), labels)
    axes[1, 1].set(ylabel="Delta signal transfer", title="PCA throughput by line and PA")
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.2)
    figure.suptitle("PCA C2 validation: 40x40, 3 lines x 3 position angles")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return path


def write_pca_c2_summary(products, cases, paths, detection_threshold_snr=DETECTION_THRESHOLD_SNR):
    """Write consolidated tables, QC, and plot."""

    paths["table_root"].mkdir(parents=True, exist_ok=True)
    paths["plot_root"].mkdir(parents=True, exist_ok=True)
    write_csv(paths["grid_csv"], products["grid_rows"], list(products["grid_rows"][0]))
    write_csv(
        paths["case_summary_csv"], products["case_rows"], list(products["case_rows"][0])
    )
    write_csv(
        paths["completeness_csv"],
        products["completeness_rows"],
        list(products["completeness_rows"][0]),
    )
    save_pca_c2_summary_plot(products, paths["summary_plot"], detection_threshold_snr)
    matched_transfer = [row["delta_matched_transfer_median"] for row in products["case_rows"]]
    aperture_transfer = [row["delta_aperture_transfer_median"] for row in products["case_rows"]]
    qc = {
        "run_id": str(cases[0]["qc"]["run_id"]),
        "stage": f"{STAGE_NAME}_summary",
        "analysis_crop_npix": int(cases[0]["qc"]["crop_npix"]),
        "recovery_mode": "realistic",
        "detection_threshold_snr": float(detection_threshold_snr),
        "n_cases": len(cases),
        "n_grid_rows": len(products["grid_rows"]),
        "line_labels": sorted({case["line_label"] for case in cases}),
        "position_angles_deg": sorted({float(case["pa_deg"]) for case in cases}),
        "input_snr_grid": sorted(
            {float(row["target_snr_input"]) for row in products["grid_rows"]}
        ),
        "delta_matched_transfer_percentiles": list(_finite_percentiles(matched_transfer)),
        "delta_aperture_transfer_percentiles": list(_finite_percentiles(aperture_transfer)),
        "input_snr_at_completeness": _completeness_thresholds(products["completeness_rows"]),
        "completeness": products["completeness_rows"],
        "outputs": {
            "grid_csv": str(paths["grid_csv"]),
            "case_summary_csv": str(paths["case_summary_csv"]),
            "completeness_csv": str(paths["completeness_csv"]),
            "summary_plot": str(paths["summary_plot"]),
        },
        "scope_note": (
            "Completeness is the fraction of the nine configured wavelength-position cases "
            f"above {float(detection_threshold_snr):g} sigma at each injected S/N; "
            "it is not a Monte Carlo probability."
        ),
    }
    write_json(paths["qc_json"], qc)
    return qc


def run_stage06_pca_c2_summary(
    run_id=None,
    *,
    project_root=None,
    detection_threshold_snr=DETECTION_THRESHOLD_SNR,
    expected_n_cases=9,
):
    """Consolidate completed notebook cases without rerunning PCA."""

    if run_id is None:
        run_config = load_run_config(project_root=project_root)
        run_id = run_config.run_id
        project_root = run_config.paths.project_root
    paths = stage06_pca_c2_summary_paths(run_id, project_root=project_root)
    cases = load_pca_c2_cases(paths["table_root"], expected_n_cases=expected_n_cases)
    products = summarize_pca_c2_cases(cases, detection_threshold_snr=detection_threshold_snr)
    qc = write_pca_c2_summary(products, cases, paths, detection_threshold_snr)
    return {"paths": paths, "cases": cases, "products": products, "qc": qc}


__all__ = [
    "DETECTION_THRESHOLD_SNR",
    "STAGE_NAME",
    "load_pca_c2_cases",
    "run_stage06_pca_c2_summary",
    "save_pca_c2_summary_plot",
    "stage06_pca_c2_summary_paths",
    "summarize_pca_c2_cases",
    "write_pca_c2_summary",
]
