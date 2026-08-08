"""Stage G2: measure the line catalog on the canonical X11 spectrum.

Additive (spec G2 §1.2): does NOT touch stage07 or H01–H03. Applies the generic
``musepipe.lines`` API to the standard spectrum product for the config line
catalog (default: the ~24 accretion lines of stage07) and writes a master table
+ QC. Significance for Halpha is reconciled against H01 (V3).
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from astropy.io import fits

from ..config import load_run_config
from ..io import load_calibrated_controls
from ..lines import measure_catalog
from ..paths import RunPaths
from ..stages.stage07_accretion_lines import default_accretion_lines

TABLE_FIELDS = [
    "name", "rest_A", "family", "kind", "status", "label",
    "continuum_density", "continuum_density_err",
    "flux_direct", "flux_direct_err", "flux_fit", "flux_fit_err",
    "ew_A", "ew_err_A", "centroid_A", "centroid_err_A",
    "fwhm_obs_A", "fwhm_intrinsic_A", "fwhm_intrinsic_err_A", "asymmetry",
    "rv_kms", "rv_err_kms", "z_score", "flux_upper_limit_5sigma",
    # La escala empirica viaja en la tabla, no solo en el QC: `sigma_inflation`
    # por linea y por metodo es la metrica de salud que deja ver un salto entre
    # reducciones o entre metodos que no se explique por el procedimiento.
    "z_emp", "sigma_inflation", "sigma_inflation_applied", "fap_empirical", "n_controls",
    "min_resolvable_fap", "null_source",
    "throughput_applied", "n_mc", "seed", "covariance_used",
    "line_window_A", "continuum_windows_A", "flags", "reason",
]

#: Espectros de control CALIBRADOS por metodo (los que D2 escribe junto al
#: objeto). Son las mismas 33 posiciones a la separacion del compañero que usan
#: C2-C6 y E1, procesadas igual: por eso sirven de distribucion nula.
_CALIBRATED_CONTROL_NPZ = {
    "aperture": "spec_calibrated_aperture_controls.npz",
    "optimal_ls": "spec_calibrated_optimal_ls_controls.npz",
    "optimal_psfsub": "spec_calibrated_optimal_psfsub_controls.npz",
    "psffit": "spec_calibrated_psffit_controls.npz",
    "sgf": "spec_calibrated_sgf_controls.npz",
    "lpm": "spec_calibrated_lpm_controls.npz",
}


def _load_controls(stage_dir, method, n_wave, *, object_path=None):
    """`(array (N, n_wave), procedencia)` o `(None, motivo)`. Nunca en silencio.

    Que FALTEN es un motivo blando: G2 lo dice y sigue con el error propagado.
    Que esten pero sin el sello de D2, o mas viejos que el objeto, NO lo es —
    seria volver a la escala que la v2 vino a arreglar, y ademas con la
    apariencia de haberla medido. `load_calibrated_controls` para en ese caso.
    """
    name = _CALIBRATED_CONTROL_NPZ.get(str(method))
    if name is None:
        return None, f"metodo {method!r} sin npz de controles declarado"
    path = Path(stage_dir) / name
    try:
        arr = np.atleast_2d(load_calibrated_controls(path, object_path=object_path))
    except FileNotFoundError:
        return None, f"{name} no esta en el run"
    except ValueError:
        return None, f"{name} no trae espectros de control"
    if arr.shape[1] != int(n_wave):
        return None, f"{name} tiene {arr.shape[1]} canales y el objeto {n_wave}"
    return arr, name


def stage_g2_paths(run_id, project_root=None):
    root = Path(project_root or Path.cwd()).resolve()
    p = RunPaths.from_project_root(run_id, root)
    return {
        "paths": p,
        "spectrum_fits": p.stage_dir / "spec_final_object.fits",
        "stage00q_qc_json": p.stage_dir / "stage00q_qc.json",
        "stage_x11_qc_json": p.stage_dir / "stage_x11_qc.json",
        "stage_g1_qc_json": p.stage_dir / "stage_g1_qc.json",
        "stage_h01_qc_json": p.stage_dir / "stage_h01_qc.json",
        "table_csv": p.table_dir / "g2_line_measurements.csv",
        "qc_json": p.stage_dir / "stage_g2_qc.json",
    }


def _read_optional(path):
    path = Path(path)
    return json.loads(path.read_text()) if path.exists() else None


def _load_spectrum(path):
    with fits.open(path) as h:
        sp = h["SPECTRUM"].data
        wave = np.asarray(sp["wave_A"], float)
        flux = np.asarray(sp["flux"], float)
        cols = sp.columns.names
        ferr = np.asarray(sp["flux_err_total"] if "flux_err_total" in cols else sp["flux_err"], float)
        flags = np.asarray(sp["flags"], int) if "flags" in cols else np.zeros(wave.size, int)
    return wave, flux, ferr, flags


def _resolve_lsf(cfg, qc00):
    m2 = (qc00 or {}).get("m2_lsf", {})
    if isinstance(m2, dict) and m2.get("status") == "ok" and m2.get("fwhm_A") is not None:
        return float(m2["fwhm_A"]), "stage00q_qc.m2_lsf"
    for key in ("g2_lsf_fwhm_A", "h01_lsf_fwhm_A", "lsf_fwhm_A"):
        if cfg.get(key) is not None:
            return float(cfg[key]), f"config.{key}(estimate; A4 M2 unavailable)"
    raise RuntimeError("G2 requires an LSF: A4 M2 is unavailable and no config LSF is set (spec §2.3 stop).")


def _empirical_scale_block(measurements, control_source):
    """Resumen de la escala empirica: de donde sale y cuanto se mueve.

    Se publica la MEDIANA y el rango entre lineas, no un numero suelto: lo que
    hace util a esta metrica es su estabilidad, y para verla hace falta la
    dispersion. `lines_without_scale` no puede quedarse en cero por descuido —
    una linea sin controles esta etiquetada con otra vara de medir.
    """
    con = [m for m in measurements if m.null_source not in ("", "none")
           and np.isfinite(m.sigma_inflation)]
    infl = np.asarray([m.sigma_inflation for m in con], dtype=np.float64)
    sin_escala = [m.name for m in measurements
                  if m.status != "not_measurable" and m.null_source in ("", "none")]
    block = {
        "source": str(control_source),
        "applied": bool(len(con) > 0),
        "n_lines_with_scale": int(len(con)),
        "lines_without_scale": sin_escala,
        "n_controls": int(con[0].n_controls) if con else 0,
        "min_resolvable_fap": float(con[0].min_resolvable_fap) if con else None,
        "note": (
            "sigma_inflation = cuantil 99 empirico de los controles / z gaussiano 99. "
            "Es el factor por el que el error propagado canal a canal subestima la "
            "dispersion real de ESTA medida. Las etiquetas detected/marginal/upper_limit "
            "usan los mismos umbrales 5/3 de la spec, pero sobre z_emp = z/sigma_inflation."
        ),
    }
    if infl.size:
        block.update({
            "inflation_median": float(np.median(infl)),
            "inflation_min": float(np.min(infl)),
            "inflation_max": float(np.max(infl)),
            "inflation_p16_p84": [float(np.percentile(infl, 16)), float(np.percentile(infl, 84))],
        })
    return block


def compute_stage_g2(cfg, paths):
    qc00 = _read_optional(paths["stage00q_qc_json"])
    x11 = _read_optional(paths["stage_x11_qc_json"]) or {}
    g1 = _read_optional(paths["stage_g1_qc_json"]) or {}
    h01 = _read_optional(paths["stage_h01_qc_json"]) or {}

    wave, flux, ferr, flags = _load_spectrum(paths["spectrum_fits"])
    lsf_fwhm, lsf_source = _resolve_lsf(cfg, qc00)
    catalog = cfg.get("g2_line_catalog", default_accretion_lines())
    vsys = float(cfg.get("h03_rv_sys_kms", cfg.get("h01_rv_sys_kms", 0.0)))
    wl_cal_err = float(cfg.get("g2_wl_cal_err_kms", 0.0))
    canonical = str(x11.get("canonical_method", cfg.get("x11_canonical_method", "psffit")))
    throughput, throughput_source = _resolve_throughput(cfg, paths, canonical)
    n_mc = int(cfg.get("g2_n_mc", 500))
    seed = int(cfg.get("g2_seed", 0))

    open_issues = []
    if "estimate" in lsf_source:
        open_issues.append({"issue": f"LSF from {lsf_source}; A4 M2 not measured (spec §2.3 would stop). Provisional.", "priority": "major"})
    if wl_cal_err == 0.0:
        open_issues.append({"issue": "Wavelength-calibration RV error unavailable (A4 M1); RV errors omit the λ-cal term.", "priority": "major"})
    if "default" in throughput_source:
        open_issues.append({"issue": "Canonical throughput not found (E4); upper limits use throughput=1.0 (uncorrected).", "priority": "major"})

    # La escala de deteccion sale de los MISMOS controles que usan C2-C6 y E1.
    # Sin ellos G2 etiquetaria contra el sigma propagado canal a canal, que a la
    # separacion del compañero subestima la dispersion real (E1 la mide ~18x en
    # psffit sobre este cubo) y convierte residuos de halo en "detecciones".
    controls, control_source = _load_controls(
        paths["spectrum_fits"].parent, canonical, wave.size,
        object_path=paths["spectrum_fits"],
    )
    if controls is None:
        open_issues.append({
            "issue": (f"Sin espectros de control ({control_source}): las etiquetas salen del "
                      "sigma PROPAGADO, que subestima la dispersion real de la medida. "
                      "`detected` aqui no es comparable con el veredicto de E1."),
            "priority": "blocking"})

    measurements = measure_catalog(
        wave, flux, ferr, catalog, lsf_fwhm_A=lsf_fwhm, vsys_kms=vsys,
        wl_cal_err_kms=wl_cal_err, throughput=throughput, n_mc=n_mc, seed=seed,
        bad_flags=(flags != 0),
        control_fluxes=controls,
        null_source=str(control_source),
    )

    # RV weighted over detected lines
    det = [m for m in measurements if m.status == "detected" and np.isfinite(m.rv_kms) and m.rv_err_kms > 0]
    if det:
        w = np.array([1.0 / m.rv_err_kms ** 2 for m in det])
        rvw = float(np.sum(w * np.array([m.rv_kms for m in det])) / np.sum(w))
        rvw_err = float(1.0 / np.sqrt(np.sum(w)))
    else:
        rvw, rvw_err = None, None

    counts = {s: sum(1 for m in measurements if m.status == s)
              for s in ("detected", "marginal", "upper_limit", "not_measurable")}

    # V3: Halpha category vs H01
    ha = next((m for m in measurements if abs(m.rest_A - 6562.8) < 1.0), None)
    h01_verdict = ((h01.get("verdict") or {}).get("verdict"))
    v3 = None
    if ha is not None and h01_verdict is not None:
        g2_cat = "detection" if ha.status == "detected" else ("non_detection" if ha.status in ("upper_limit", "not_measurable") else "candidate")
        v3 = {"g2_halpha_status": ha.status, "h01_verdict": h01_verdict, "consistent": bool(g2_cat == h01_verdict)}
        if not v3["consistent"]:
            open_issues.append({"issue": f"V3: G2 Halpha status {ha.status!r} vs H01 {h01_verdict!r} — two statistics on the same data disagree.", "priority": "blocking"})

    qc = {
        "stage": "g2_measure_lines", "run_id": str(cfg["run_id"]), "provisional": True,
        "input_spectrum": {"file": str(paths["spectrum_fits"].name), "method": canonical},
        "lsf_source": lsf_source, "lsf_fwhm_A": lsf_fwhm,
        "covariance_used": "none",  # per-line MC used independent errors in this provisional run
        # --- escala empirica: el diagnostico, no solo el criterio ---
        # `sigma_inflation` mide cuanto subestima el error propagado. Su valor
        # absoluto interesa menos que su ESTABILIDAD: entre lineas, entre
        # metodos y entre reducciones deberia moverse poco. Un salto que no se
        # explique por lo que se ha cambiado a proposito apunta a la reduccion,
        # no a la fisica, y por eso se publica por linea en la tabla.
        "empirical_scale": _empirical_scale_block(measurements, control_source),
        "catalog_n": len(catalog),
        "n_detected": counts["detected"], "n_marginal": counts["marginal"],
        "n_upper_limit": counts["upper_limit"], "n_not_measurable": counts["not_measurable"],
        "rv_weighted_kms": {"value": rvw, "err": rvw_err, "n_lines": len(det)},
        "mc": {"n": n_mc, "seed": seed}, "throughput_applied": throughput, "throughput_source": throughput_source,
        "halpha_reconciliation_v3": v3,
        "open_issues": open_issues,
    }
    return measurements, qc


def _resolve_throughput(cfg, paths, method):
    if cfg.get("g2_throughput") is not None:
        return float(cfg["g2_throughput"]), "config.g2_throughput"
    h04 = _read_optional(paths["paths"].stage_dir / "stage_h04_qc.json") or {}
    per = ((h04.get("throughput") or {}).get("per_method_at_snr5") or {}).get(method) or {}
    if per.get("throughput") is not None and float(per["throughput"]) > 0:
        return float(per["throughput"]), "stage_h04_qc.per_method_at_snr5"
    return 1.0, "default_1.0(E4 throughput unavailable)"


def write_stage_g2(measurements, qc, paths):
    paths["paths"].ensure_base_dirs()
    with paths["table_csv"].open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=TABLE_FIELDS, extrasaction="ignore")
        w.writeheader()
        for m in measurements:
            w.writerow(m.to_row())
    paths["qc_json"].write_text(json.dumps(qc, indent=1, default=str))
    return {"table": paths["table_csv"], "qc_json": paths["qc_json"]}


def run_stage_g2(run_id=None, *, project_root=None, overrides=None, allow_run_id_mismatch=False):
    rc = load_run_config(run_id, project_root=project_root, allow_run_id_mismatch=allow_run_id_mismatch)
    cfg = dict(rc.config)
    if overrides:
        cfg.update(overrides)
    cfg["run_id"] = rc.run_id
    cfg["project_root"] = str(rc.paths.project_root)
    paths = stage_g2_paths(cfg["run_id"], project_root=cfg["project_root"])
    measurements, qc = compute_stage_g2(cfg, paths)
    written = write_stage_g2(measurements, qc, paths)
    return {"config": cfg, "paths": paths, "qc": qc, "written": written}


__all__ = ["compute_stage_g2", "run_stage_g2", "stage_g2_paths", "TABLE_FIELDS"]
