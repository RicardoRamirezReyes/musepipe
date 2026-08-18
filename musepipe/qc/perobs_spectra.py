"""Diagnóstico por exposición: el mismo compañero, medido N veces.

La cadena estima σ empíricamente con controles procesados igual que el objeto
(`docs/noise_model.md`), porque el STAT del cubo subestima el ruido de apertura
~4×. Lo que nunca ha tenido es lo que un experimentalista pediría primero: **N
repeticiones independientes de la misma medida**. Con la PSF por observación
(`docs/2026-08-15_psf_por_observacion.md`) eso pasa a ser posible, porque cada
exposición tiene por fin su propio modelo de PSF —y por tanto su propia
corrección de apertura— en vez de heredar la del combinado.

Este módulo extrae el espectro del compañero en **cada** exposición, con su
modelo, y publica la **dispersión** entre ellas. Esa dispersión incluye todo lo
que el modelo no captura (seeing, transmisión, alineado, el halo que no se
resta bien), así que es la única vara honesta para leer si la barra de error que
publica D2 es realista o cosmética.

**Qué NO es.** No reproduce C2 ni C4: no pasa por el marco de B1, no aplica el
factor de flujo total empírico de A2 y usa un fondo de anillo idéntico en las N
exposiciones en vez del producto de 04b (que se ajusta al combinado y no existe
por exposición). Todas esas piezas son **comunes** a las N exposiciones, así que
se cancelan en la dispersión fraccional, que es lo que este diagnóstico mide;
pero el nivel absoluto de estos espectros no es el de la cadena, y compararlos
con el producto de D2 canal a canal sería un error. Por eso vive en `qc/` y no
es una etapa del `stage_registry`: nada de la cadena lo consume.

**El exceso sobre STAT no es un fallo.** El diagnóstico compara la dispersión
medida contra la propagada del STAT de cada exposición. Un cociente ~4 es lo que
`docs/noise_model.md` ya dice que pasa; lo informativo es cuánto se sale de ahí,
y sobre todo su estructura en λ. Se publica el cociente, nunca un veredicto.
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from ..config import load_run_config
from ..extraction.aperture import (
    annulus_background_spectrum,
    aperture_correction_from_psf,
    aperture_label,
    aperture_spectrum,
)
from ..extraction.psffit import fit_psffit_cube
from ..io import read_json, write_csv, write_json
from ..observations import resolve_observation_plan
from ..parallel import resolve_n_jobs
from ..paths import RunPaths
from ..reduction.stream_combine import MAX_CRVAL3_SPREAD_CHANNELS
from ..stages.stage_e01_perobs import _positions_in_window, load_aligned_exposure
from ..stages.stage_e01b_perobs_subtract import models_by_exposure

#: Cada worker sostiene un cubo alineado (3681×200×200 en float32 ≈ 0.6 G, más
#: su STAT), así que el techo lo pone la memoria y no los núcleos.
DEFAULT_MAX_WORKERS = 4

#: El anillo de fondo de C2 (`stage_x01_aperture.py`, donde vive el default real
#: porque el run no suele declararlo): (r_in, r_out, radio de exclusión de la
#: primaria) en píxeles.
ANNULUS_BKG_FALLBACK_PX = (8.0, 14.0, 30.0)

SCHEMA_VERSION = 1


class PerObsSpectraError(RuntimeError):
    """El diagnóstico por exposición no se puede hacer con lo que hay."""


def perobs_spectra_paths(run_id, project_root=None) -> dict:
    root = Path(project_root or Path.cwd()).resolve()
    paths = RunPaths.from_project_root(run_id, root)
    return {
        "run_paths": paths,
        "psf_model_mixture_json": paths.stage_dir / "psf_model_mixture.json",
        "psf_model_json": paths.stage_dir / "psf_model.json",
        "observation_plan_json": paths.stage_dir / "observation_plan.json",
        "stage01c_qc_json": paths.stage_dir / "stage01c_qc.json",
        "qc_json": paths.stage_dir / "perobs_spectra_qc.json",
        "spectra_npz": paths.stage_dir / "perobs_spectra.npz",
        "dispersion_csv": paths.table_dir / "perobs_spectra_dispersion.csv",
        "summary_csv": paths.table_dir / "perobs_spectra_exposures.csv",
        "plot": paths.plot_dir / "perobs_spectra.png",
    }


def perobs_spectra_config_from_run(run_id=None, *, project_root=None, overrides=None,
                                   allow_run_id_mismatch=False) -> dict:
    """Config resuelta. Las perillas compartidas se LEEN de quien manda en ellas.

    Las aperturas salen de C2 y los radios de ajuste de C4, y se leen de su
    **config resuelto** (`stage_xNN_config_from_run`), no del config crudo del
    run: la etapa rellena defaults que el run no deletrea —`x01_apertures` es uno
    de ellos— y copiarlos aquí como literales es exactamente lo que hizo que el
    primer notebook de C3 no reprodujera la cadena.
    """

    from ..stages.stage_x01_aperture import stage_x01_config_from_run
    from ..stages.stage_x03_psffit import stage_x03_config_from_run

    run = load_run_config(run_id, project_root=project_root,
                          allow_run_id_mismatch=allow_run_id_mismatch)
    cfg = dict(run.config)
    cfg["run_id"] = run.run_id
    cfg["project_root"] = str(run.paths.project_root)

    x01 = stage_x01_config_from_run(run.run_id, project_root=run.paths.project_root,
                                    allow_run_id_mismatch=allow_run_id_mismatch)
    x03 = stage_x03_config_from_run(run.run_id, project_root=run.paths.project_root,
                                    allow_run_id_mismatch=allow_run_id_mismatch)

    cfg.setdefault("perobs_spectra_apertures", x01.get("x01_apertures"))
    cfg.setdefault("perobs_spectra_background", "annulus")
    # `x01_annulus_bkg_px` no lo fija el config resuelto de C2: se lee en el punto
    # de uso (`stage_x01_aperture.py`), y ese es el default que se replica aquí.
    cfg.setdefault("perobs_spectra_annulus_px",
                   list(x01.get("x01_annulus_bkg_px", ANNULUS_BKG_FALLBACK_PX)))
    cfg.setdefault("perobs_spectra_star_radius_px", x03.get("x03_star_radius_px", 20.0))
    cfg.setdefault("perobs_spectra_comp_radius_px", x03.get("x03_comp_radius_px", 12.0))
    # El ajuste por canal de C4 cuesta ~11 min para los 3681 canales de UN cubo;
    # multiplicado por las N exposiciones son horas. El submuestreo es la misma
    # perilla que ya lleva el notebook debug de C4, y la dispersión es una
    # cantidad de banda ancha: no necesita todos los canales. Los que sí importan
    # a resolución completa se declaran aparte.
    cfg.setdefault("perobs_spectra_psffit_channel_step", 10)
    cfg.setdefault("perobs_spectra_psffit_full_windows_A", [[6520.0, 6610.0]])
    cfg.setdefault("perobs_spectra_max_workers", DEFAULT_MAX_WORKERS)
    cfg.setdefault("perobs_spectra_psffit", True)
    cfg.setdefault("perobs_spectra_wave_tol_channels", MAX_CRVAL3_SPREAD_CHANNELS)
    if overrides:
        cfg.update(overrides)
    if not cfg.get("perobs_spectra_apertures"):
        raise PerObsSpectraError(
            "el run no declara `x01_apertures` y no se ha pasado "
            "`perobs_spectra_apertures`: sin apertura no hay espectro que comparar."
        )
    return cfg


def _positions_from_qc(qc, source_key="companion"):
    """Compañero y fuente de campo del QC de B3, en el marco de la cadena.

    La **primaria no se lee**: en la ventana de cada exposición está en el centro
    del array por construcción del plan, que es precisamente cómo alinea el
    combinado. Es la misma convención con la que C1 llama a `_positions_in_window`
    (le pasa sólo `companion` y `field`) y con la que C1b construye sus offsets.
    """

    source = qc.get(source_key) or {}
    if "pos_yx" not in source:
        raise PerObsSpectraError(f"el QC de B3 no trae `pos_yx` para {source_key!r}.")
    field = qc.get("field_source") or {}
    shape = tuple(int(v) for v in (qc.get("cube_shape") or ())[-2:])
    if len(shape) != 2:
        raise PerObsSpectraError(
            "el QC de B3 no declara `cube_shape`: sin él no se puede llevar la posición "
            "del compañero al marco de cada exposición."
        )
    return (
        tuple(float(v) for v in source["pos_yx"]),
        tuple(float(v) for v in field["pos_yx"]) if field and "pos_yx" in field else None,
        shape,
    )


def channel_selection(wave_A, step, full_windows_A):
    """Los canales que van al psffit: uno de cada `step`, y todos en las ventanas.

    Devuelve índices ordenados y sin repetir. `step<=1` los coge todos.
    """

    wave = np.asarray(wave_A, dtype=np.float64)
    step = max(1, int(step))
    keep = np.zeros(wave.size, dtype=bool)
    keep[::step] = True
    for window in full_windows_A or ():
        lo, hi = (float(window[0]), float(window[1]))
        keep |= (wave >= lo) & (wave <= hi)
    return np.flatnonzero(keep)


def extract_one_observation(exposure, plan, cfg, model_doc, positions_yx, frame_shape):
    """El espectro del compañero en UNA exposición, con SU modelo de PSF."""

    cube, stat, wave = load_aligned_exposure(exposure, plan)
    npix = int(plan.crop_npix)
    # La primaria ES el centro de la ventana: el plan alinea cada exposición
    # sobre ella. El compañero se desplaza igual que en C1 y C1b.
    star_yx, moved = _positions_in_window(npix, positions_yx, frame_shape)
    comp_yx = moved["companion"]
    if comp_yx is None:
        raise PerObsSpectraError(f"{exposure.exposure_id}: el compañero no cae en la ventana.")

    row = {
        "exposure_id": exposure.exposure_id,
        "weight": float(exposure.weight),
        "wave_A": wave,
        "apertures": {},
    }

    background = np.zeros(wave.size, dtype=np.float64)
    mode = str(cfg.get("perobs_spectra_background", "annulus")).lower()
    if mode == "annulus":
        r_in, r_out, exclude = (list(cfg["perobs_spectra_annulus_px"]) + [30.0])[:3]
        background = annulus_background_spectrum(
            cube, comp_yx, float(r_in), float(r_out),
            exclude_yx=star_yx, exclude_radius=float(exclude),
        )
    elif mode != "none":
        raise PerObsSpectraError(
            f"`perobs_spectra_background` sólo entiende 'annulus' o 'none', no {mode!r}."
        )

    for aperture in cfg["perobs_spectra_apertures"]:
        flux, npix_eff = aperture_spectrum(cube, comp_yx, aperture)
        flux = flux - background * npix_eff
        # La apcorr SÍ es por exposición: es justo lo que la PSF por observación
        # cambia. El factor de flujo total empírico de A2 no se aplica (es común
        # a las N y se cancela en la dispersión fraccional).
        apcorr, apcorr_mode, _ = aperture_correction_from_psf(
            wave, aperture, model_doc, center_yx=comp_yx,
        )
        row["apertures"][aperture_label(aperture)] = {
            "flux": np.asarray(flux, dtype=np.float64) * np.asarray(apcorr, dtype=np.float64),
            "flux_raw": np.asarray(flux, dtype=np.float64),
            "apcorr": np.asarray(apcorr, dtype=np.float64),
            "apcorr_mode": apcorr_mode,
            "npix_eff": np.asarray(npix_eff, dtype=np.float64),
            # El STAT NO es σ (invariante del repo): entra aquí sólo para poder
            # decir cuánto se separa la dispersión medida de lo que el cubo cree.
            "stat_sigma": np.sqrt(
                np.nansum(_aperture_weights_like(cube, comp_yx, aperture) ** 2 * stat, axis=(1, 2))
            ) * np.asarray(apcorr, dtype=np.float64),
        }

    if cfg.get("perobs_spectra_psffit", True):
        sel = channel_selection(
            wave,
            cfg["perobs_spectra_psffit_channel_step"],
            cfg["perobs_spectra_psffit_full_windows_A"],
        )
        fit = fit_psffit_cube(
            cube[sel], stat[sel], wave[sel], star_yx, comp_yx, model_doc,
            star_radius_px=float(cfg["perobs_spectra_star_radius_px"]),
            comp_radius_px=float(cfg["perobs_spectra_comp_radius_px"]),
            n_jobs=1,
        )
        row["psffit"] = {
            "channels": sel,
            "wave_A": wave[sel],
            # C4 no lleva apcorr: el ajuste con P normalizada ya da flujo en la
            # convención NORMRAD (spec C4 §3.4).
            "flux": np.asarray(fit.coeffs[:, 1], dtype=np.float64),
            "star_flux": np.asarray(fit.coeffs[:, 0], dtype=np.float64),
            "stat_sigma": np.sqrt(np.abs(fit.covariance[:, 1, 1])),
            "chi2r": np.asarray(fit.chi2r, dtype=np.float64),
        }
    return row


def _aperture_weights_like(cube, center_yx, aperture):
    from ..extraction.aperture import aperture_weights

    _, ny, nx = np.asarray(cube).shape
    return aperture_weights(ny, nx, center_yx, aperture)


def extract_all_observations(observations, cfg, models, positions_yx, frame_shape, *,
                             n_jobs=None, progress=None):
    """Todas las exposiciones. Lo que falle se cuenta, no se esconde."""

    exposures = list(observations.exposures)
    workers = min(int(resolve_n_jobs(n_jobs, max_default=DEFAULT_MAX_WORKERS)),
                  max(1, len(exposures)))
    rows, failures = {}, []

    def _one(exposure):
        try:
            model_doc = models.get(exposure.exposure_id)
            if model_doc is None:
                raise PerObsSpectraError(
                    f"{exposure.exposure_id}: el plan la combina pero la mezcla de C1 no la "
                    "trae. Vuelve a correr C1 con psf_scope=per_observation."
                )
            return exposure, extract_one_observation(
                exposure, observations.plan, cfg, model_doc, positions_yx, frame_shape
            ), None
        except Exception as exc:  # una exposición no tumba las N
            return exposure, None, f"{exc.__class__.__name__}: {exc}"

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for index, (exposure, row, error) in enumerate(pool.map(_one, exposures)):
            if progress is not None:
                progress(index, len(exposures), exposure.exposure_id)
            if error is not None:
                failures.append({"exposure_id": exposure.exposure_id, "error": error})
                continue
            rows[exposure.exposure_id] = row

    if not rows:
        raise PerObsSpectraError(
            "ninguna exposición se pudo medir: "
            + "; ".join(f"{f['exposure_id']}: {f['error']}" for f in failures[:3])
        )
    ordered = [rows[e.exposure_id] for e in exposures if e.exposure_id in rows]
    return ordered, failures


def dispersion(fluxes, weights, sigmas=None):
    """Media pesada, dispersión entre exposiciones, y el exceso sobre STAT.

    `fluxes` es `(n_exposiciones, n_canales)`. La dispersión es la de una medida
    individual (no la del promedio): es lo que se compara con la σ que el cubo
    declara para UNA exposición.
    """

    flux = np.asarray(fluxes, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)[:, None]
    if flux.ndim != 2:
        raise ValueError(f"fluxes debe ser (n_exp, n_canales), no {flux.shape}.")
    n = flux.shape[0]
    good = np.isfinite(flux)
    w_eff = np.where(good, w, 0.0)
    w_sum = w_eff.sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.nansum(np.where(good, flux, 0.0) * w_eff, axis=0) / w_sum
        var = np.nansum(w_eff * (np.where(good, flux, 0.0) - mean) ** 2, axis=0) / w_sum
        # Corrección de Bessel efectiva para pesos: n/(n-1) con el n efectivo.
        n_eff = (w_sum ** 2) / np.nansum(w_eff ** 2, axis=0)
        scatter = np.sqrt(var * np.where(n_eff > 1, n_eff / (n_eff - 1.0), np.nan))
        frac = scatter / np.abs(mean)
    out = {
        "n_exposures": int(n),
        "mean": mean,
        "scatter": scatter,
        "scatter_frac": frac,
        "n_eff": n_eff,
        "n_good": good.sum(axis=0).astype(int),
    }
    if sigmas is not None:
        sigma = np.asarray(sigmas, dtype=np.float64)
        with np.errstate(invalid="ignore", divide="ignore"):
            sigma_typ = np.sqrt(np.nanmean(sigma ** 2, axis=0))
            out["stat_sigma_typical"] = sigma_typ
            out["excess_over_stat"] = scatter / sigma_typ
    return out


def _summarize(block):
    """Los tres números que se leen de un vector por canal, sin NaN de por medio."""

    def _stat(values):
        values = np.asarray(values, dtype=np.float64)
        values = values[np.isfinite(values)]
        if values.size == 0:
            return {"median": None, "p90": None, "n": 0}
        return {
            "median": float(np.median(values)),
            "p90": float(np.percentile(values, 90)),
            "n": int(values.size),
        }

    out = {"scatter_frac": _stat(block["scatter_frac"])}
    if "excess_over_stat" in block:
        out["excess_over_stat"] = _stat(block["excess_over_stat"])
    return out


def _sha256(path):
    import hashlib

    path = Path(path)
    if not path.exists():
        return ""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_perobs_spectra(run_id=None, *, project_root=None, overrides=None,
                       allow_run_id_mismatch=False, n_jobs=None, progress=None,
                       save_plot=True):
    cfg = perobs_spectra_config_from_run(
        run_id, project_root=project_root, overrides=overrides,
        allow_run_id_mismatch=allow_run_id_mismatch,
    )
    paths = perobs_spectra_paths(cfg["run_id"], project_root=cfg["project_root"])

    model_path = paths["psf_model_mixture_json"]
    if not model_path.exists():
        model_path = paths["psf_model_json"]
    if not model_path.exists():
        raise PerObsSpectraError(
            f"no hay modelo de PSF por observación en {paths['psf_model_mixture_json']}. "
            "Corre C1 con `--psf-scope per_observation` antes que este diagnóstico."
        )
    models = models_by_exposure(read_json(model_path))

    positions_path = Path(cfg.get("stage_e01_positions_qc", paths["stage01c_qc_json"]))
    if not positions_path.exists():
        raise PerObsSpectraError(f"hace falta el QC de B3 (posiciones): {positions_path}")
    positions_qc = read_json(positions_path)
    comp_yx, field_yx, frame_shape = _positions_from_qc(
        positions_qc, source_key=str(cfg.get("x01_source_key", "companion")),
    )
    positions_yx = {"companion": comp_yx, "field": field_yx}

    plan_cache = paths["observation_plan_json"]
    observations = resolve_observation_plan(
        cfg["run_id"], project_root=cfg["project_root"],
        allow_run_id_mismatch=allow_run_id_mismatch,
        plan_json=str(plan_cache) if plan_cache.exists() else None,
    )

    rows, failures = extract_all_observations(
        observations, cfg, models, positions_yx, frame_shape,
        n_jobs=n_jobs if n_jobs is not None else cfg.get("perobs_spectra_max_workers"),
        progress=progress,
    )

    wave = rows[0]["wave_A"]
    # El eje λ tiene que ser el mismo canal en las N, o la dispersión no
    # significa nada. Pero «el mismo» se mide en fracciones de canal, no en
    # angstroms absolutos: las cabeceras guardan CRVAL3 en float32, que a 4750 Å
    # cuantiza a ~0.0005 Å, y exigir igualdad exacta hace fallar un run sano.
    # Medido en ROXs 12 b (29 exposiciones): deriva máxima 0.0039 Å = 0.31 % de
    # canal, en nueve escalones de 0.0005 Å — precisión de cabecera, no deriva
    # física. La tolerancia va en canales y la deriva medida se PUBLICA.
    channel_A = float(np.median(np.diff(wave))) if wave.size > 1 else 1.25
    # La tolerancia NO se inventa aquí: es la misma con la que el plan del
    # combinado acepta o rechaza un lote (`MAX_CRVAL3_SPREAD_CHANNELS`). Un
    # desfase real no llega hasta aquí —el plan se niega a construirse—, así que
    # esto es una re-comprobación barata sobre el eje ya leído, no un segundo
    # criterio que pudiera contradecir al primero.
    tol_channels = float(cfg.get("perobs_spectra_wave_tol_channels",
                                 MAX_CRVAL3_SPREAD_CHANNELS))
    drift_A = 0.0
    for row in rows[1:]:
        if row["wave_A"].size != wave.size:
            raise PerObsSpectraError(
                f"{row['exposure_id']}: su eje λ tiene {row['wave_A'].size} canales y el de la "
                f"primera exposición {wave.size}. No son la misma medida."
            )
        drift_A = max(drift_A, float(np.max(np.abs(row["wave_A"] - wave))))
    if drift_A > tol_channels * abs(channel_A):
        raise PerObsSpectraError(
            f"el eje λ se mueve {drift_A:.4f} Å entre exposiciones "
            f"({drift_A / abs(channel_A):.2f} canales, tolerancia {tol_channels:g}): "
            "las N medidas no son del mismo canal y la dispersión no significaría nada."
        )

    weights = np.array([row["weight"] for row in rows], dtype=np.float64)
    blocks, arrays = {}, {"wave_A": wave, "weights": weights,
                          "exposure_ids": np.array([r["exposure_id"] for r in rows])}
    for label in rows[0]["apertures"]:
        flux = np.vstack([row["apertures"][label]["flux"] for row in rows])
        sigma = np.vstack([row["apertures"][label]["stat_sigma"] for row in rows])
        blocks[f"aperture_{label}"] = dispersion(flux, weights, sigmas=sigma)
        arrays[f"flux_aperture_{label}"] = flux
        arrays[f"stat_sigma_aperture_{label}"] = sigma
        # Sin la apcorr al lado no se puede separar «el dato cambió» de «la
        # corrección de apertura cambió», que es exactamente lo que la PSF por
        # observación pone en juego.
        arrays[f"flux_raw_aperture_{label}"] = np.vstack(
            [row["apertures"][label]["flux_raw"] for row in rows]
        )
        arrays[f"apcorr_aperture_{label}"] = np.vstack(
            [row["apertures"][label]["apcorr"] for row in rows]
        )
    if "psffit" in rows[0]:
        sel = rows[0]["psffit"]["channels"]
        flux = np.vstack([row["psffit"]["flux"] for row in rows])
        sigma = np.vstack([row["psffit"]["stat_sigma"] for row in rows])
        blocks["psffit"] = dispersion(flux, weights, sigmas=sigma)
        arrays["psffit_channels"] = sel
        arrays["psffit_wave_A"] = rows[0]["psffit"]["wave_A"]
        arrays["flux_psffit"] = flux
        arrays["stat_sigma_psffit"] = sigma
        arrays["chi2r_psffit"] = np.vstack([row["psffit"]["chi2r"] for row in rows])

    open_issues = []
    if failures:
        open_issues.append(
            f"{len(failures)} de {len(observations.exposures)} exposiciones no se pudieron medir."
        )
    for name, block in blocks.items():
        excess = block.get("excess_over_stat")
        if excess is None:
            continue
        median = np.nanmedian(excess)
        if np.isfinite(median) and median > 10.0:
            open_issues.append(
                f"{name}: la dispersión entre exposiciones es {median:.1f}× la σ del STAT "
                "(docs/noise_model.md documenta ~4× para apertura; por encima hay sistemático)."
            )

    qc = {
        "stage": "perobs_spectra_diagnostic",
        "schema_version": SCHEMA_VERSION,
        "run_id": cfg["run_id"],
        "input": {
            "psf_model": str(model_path),
            "psf_model_sha256": _sha256(model_path),
            "positions_from": str(positions_path),
            "observation_plan": observations.source,
            "n_exposures_planned": len(observations.exposures),
            "n_exposures_measured": len(rows),
            # No es contabilidad: es la comprobación de que las N medidas caen en
            # el mismo canal, publicada como número en vez de como un pase mudo.
            "wave_axis_drift_A": drift_A,
            "wave_axis_drift_channels": drift_A / abs(channel_A),
            "wave_axis_tol_channels": tol_channels,
        },
        "convention": {
            "frame": "ventana del combinado (crop_npix del plan), NO el marco de B1",
            "apertures": cfg["perobs_spectra_apertures"],
            "background": cfg["perobs_spectra_background"],
            "annulus_bkg_px": list(cfg["perobs_spectra_annulus_px"]),
            "aperture_correction": "por exposición, del modelo de PSF de esa exposición",
            "empirical_total_flux_factor": False,
            "psffit": {
                "enabled": bool(cfg.get("perobs_spectra_psffit", True)),
                "star_radius_px": float(cfg["perobs_spectra_star_radius_px"]),
                "comp_radius_px": float(cfg["perobs_spectra_comp_radius_px"]),
                "channel_step": int(cfg["perobs_spectra_psffit_channel_step"]),
                "full_windows_A": cfg["perobs_spectra_psffit_full_windows_A"],
                "apcorr": 1.0,
            },
            "note": (
                "Diagnóstico de DISPERSIÓN: el nivel absoluto no es el de la cadena "
                "(sin marco de B1, sin factor de flujo total empírico, fondo de anillo "
                "en vez de 04b). Lo común a las N exposiciones se cancela en la "
                "dispersión fraccional."
            ),
        },
        "exposures": [
            {
                "exposure_id": row["exposure_id"],
                "weight": row["weight"],
                "form": str((models.get(row["exposure_id"]) or {}).get("form", "")),
            }
            for row in rows
        ],
        "failures": failures,
        "dispersion": {name: _summarize(block) for name, block in blocks.items()},
        "open_issues": open_issues,
    }

    paths["run_paths"].ensure_base_dirs()
    write_json(paths["qc_json"], qc)
    np.savez_compressed(paths["spectra_npz"], **arrays)
    _write_dispersion_csv(paths["dispersion_csv"], wave, blocks, arrays)
    write_csv(
        paths["summary_csv"],
        [
            {
                "exposure_id": row["exposure_id"],
                "weight": f"{row['weight']!r}",
                "form": str((models.get(row["exposure_id"]) or {}).get("form", "")),
                "flux_median_first_aperture": repr(float(np.nanmedian(
                    next(iter(row["apertures"].values()))["flux"]
                ))),
            }
            for row in rows
        ],
    )
    if save_plot:
        _plot(paths["plot"], wave, blocks, arrays)
    return {"qc": qc, "paths": {k: str(v) for k, v in paths.items() if k != "run_paths"},
            "blocks": blocks}


def _write_dispersion_csv(path, wave, blocks, arrays):
    rows = []
    for index, lam in enumerate(wave):
        row = {"lambda_A": repr(float(lam))}
        for name, block in blocks.items():
            if name == "psffit":
                continue
            row[f"{name}_mean"] = repr(float(block["mean"][index]))
            row[f"{name}_scatter"] = repr(float(block["scatter"][index]))
            row[f"{name}_scatter_frac"] = repr(float(block["scatter_frac"][index]))
            if "excess_over_stat" in block:
                row[f"{name}_excess_over_stat"] = repr(float(block["excess_over_stat"][index]))
        rows.append(row)
    write_csv(path, rows)


def _plot(path, wave, blocks, arrays):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(blocks)
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    for name, block in blocks.items():
        lam = arrays["psffit_wave_A"] if name == "psffit" else wave
        axes[0].plot(lam, block["scatter_frac"], lw=0.7, label=name)
        if "excess_over_stat" in block:
            axes[1].plot(lam, block["excess_over_stat"], lw=0.7, label=name)
    axes[0].set_ylabel("dispersión / |media|")
    axes[0].set_yscale("log")
    axes[0].legend(fontsize=8)
    axes[0].set_title(f"Diagnóstico por exposición · {n} métodos · N repeticiones de la misma medida")
    axes[1].axhline(4.0, color="0.5", ls="--", lw=0.8,
                    label="~4× documentado en docs/noise_model.md")
    axes[1].set_ylabel("dispersión / σ(STAT)")
    axes[1].set_xlabel("λ [Å]")
    axes[1].set_yscale("log")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Diagnóstico por exposición: espectro del compañero en cada exposición y su dispersión.",
    )
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--project-root", default=None)
    parser.add_argument("--allow-run-id-mismatch", action="store_true")
    parser.add_argument("--jobs", type=int, default=None,
                        help="exposiciones medidas a la vez (cada una sostiene su cubo alineado).")
    parser.add_argument("--channel-step", type=int, default=None,
                        help="submuestreo de canales del psffit (1 = todos, caro).")
    parser.add_argument("--no-psffit", action="store_true",
                        help="sólo apertura: barato, para una primera pasada.")
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args(argv)

    overrides = {}
    if args.channel_step is not None:
        overrides["perobs_spectra_psffit_channel_step"] = int(args.channel_step)
    if args.no_psffit:
        overrides["perobs_spectra_psffit"] = False

    def _progress(index, total, exposure_id):
        print(f"[{index + 1}/{total}] {exposure_id}", flush=True)

    result = run_perobs_spectra(
        args.run_id,
        project_root=args.project_root,
        overrides=overrides or None,
        allow_run_id_mismatch=args.allow_run_id_mismatch,
        n_jobs=args.jobs,
        progress=_progress,
        save_plot=not args.no_plot,
    )
    print(json.dumps(result["qc"]["dispersion"], indent=2, ensure_ascii=False))
    print(result["paths"]["qc_json"])


__all__ = [
    "ANNULUS_BKG_FALLBACK_PX",
    "DEFAULT_MAX_WORKERS",
    "PerObsSpectraError",
    "channel_selection",
    "dispersion",
    "extract_all_observations",
    "extract_one_observation",
    "perobs_spectra_config_from_run",
    "perobs_spectra_paths",
    "run_perobs_spectra",
]


if __name__ == "__main__":  # pragma: no cover
    main()
