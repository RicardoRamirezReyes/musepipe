"""C1b: resta la PSF de CADA exposición y combina después.

Por qué existe
-------------
Restar el halo de la primaria sobre el cubo **combinado** obliga a describir con
un solo modelo la mezcla de 29–30 PSF distintas. Con la PSF ya ajustada por
observación (C1, `psf_scope=per_observation`), la resta puede hacerse donde el
modelo es válido —en su propia exposición— y combinarse después, que es el orden
correcto: primero el modelo, luego la media.

Cómo
----
Reutiliza el combinado que ya existe (`stream_combine.combine_streaming`) con su
gancho `transform`: cada trozo de cada exposición se recorta y alinea igual que
siempre, se le ajusta a **su** modelo la amplitud y el fondo por canal (la misma
función que usa C3, `fit_primary_psf_amplitudes`), se resta, y el acumulador
sigue como si nada. La memoria queda acotada por el trozo, no por el número de
exposiciones.

Lo que hay que saber para leer el producto
------------------------------------------
* El fondo ajustado **no** se resta: igual que en C3, el modelo que se quita es
  `amp·PSF`, así que el cielo residual del cubo se conserva.
* `STAT` no se toca: restar un modelo determinista no cambia la varianza.
* Con `method="mean"` esto es **exactamente** lo mismo que restar la mezcla del
  cubo combinado (la combinación es lineal). Con `sigclip` no lo es, y esa
  diferencia se mide y se publica en el QC en vez de suponerse despreciable.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from astropy.io import fits

from ..config import load_run_config
from ..extraction.optimal import fit_primary_psf_amplitudes
from ..io import read_json, write_json
from ..observations import resolve_observation_plan
from ..paths import RunPaths
from ..reduction.stream_combine import (
    build_output_header,
    combine_streaming,
    wavelength_axis,
)

STAGE = "e01b_perobs_subtract"


class PerObservationSubtractError(RuntimeError):
    """No se puede restar por observación con lo que hay en disco."""


def stage_e01b_paths(run_id, project_root=None):
    root = Path(project_root or Path.cwd()).resolve()
    paths = RunPaths.from_project_root(run_id, root)
    return {
        "paths": paths,
        "psf_model_json": paths.stage_dir / "psf_model.json",
        # La mezcla con los modelos POR EXPOSICION. Se escribe siempre en su
        # propio fichero, aunque normalmente sea tambien lo que va en
        # `psf_model.json`: asi C1b no depende de que knob eligio C1.
        "psf_model_mixture_json": paths.stage_dir / "psf_model_mixture.json",
        "stage01_qc_json": paths.stage_dir / "stage01_qc.json",
        "stage01c_qc_json": paths.stage_dir / "stage01c_qc.json",
        "stage02_cube_fits": paths.stage_dir / "stage02_xcorr_cube_stack.fits",
        "stage_e01_qc_json": paths.stage_dir / "stage_e01_qc.json",
        "cube_psfsub_fits": paths.stage_dir / "cube_psfsub_perobs.fits",
        "stage_e01b_qc_json": paths.stage_dir / "stage_e01b_qc.json",
    }


def stage_e01b_config_from_run(run_id=None, *, project_root=None, overrides=None,
                               allow_run_id_mismatch=False):
    run = load_run_config(run_id, project_root=project_root,
                          allow_run_id_mismatch=allow_run_id_mismatch)
    cfg = dict(run.config)
    if overrides:
        cfg.update(overrides)
    cfg["run_id"] = run.run_id
    cfg["project_root"] = str(run.paths.project_root)
    # El radio del ajuste de amplitud y la exclusion del compañero son los MISMOS
    # que usa C3 al restar sobre el combinado: si aqui se eligieran otros, la
    # comparacion entre los dos caminos dejaria de medir el orden de las
    # operaciones y pasaria a medir dos recetas distintas.
    cfg.setdefault("x02_primary_fit_radius_px", 25.0)
    cfg.setdefault("x02_primary_exclude_radius_px", cfg.get("x02_window_radius_px", 8.0))
    return cfg


def models_by_exposure(psf_model):
    """`exposure_id -> documento de PSF` desde la mezcla que entregó C1."""

    if str(psf_model.get("form", "")).lower() != "mixture":
        raise PerObservationSubtractError(
            "C1b necesita el documento `mixture` que escribe C1 con "
            f"psf_scope=per_observation (`psf_model_mixture.json`); éste es "
            f"{psf_model.get('form')!r}."
        )
    models = {}
    for component in psf_model.get("components") or ():
        models[str(component["exposure_id"])] = component["model"]
    if not models:
        raise PerObservationSubtractError("la mezcla no trae componentes.")
    return models


def b1_window(stage01_qc, wave_plan, wave_b1):
    """El recorte con el que B1 pasa del cubo combinado al marco de la cadena.

    C3 trabaja en el marco de B1 (170 px), no en el del combinado (200 px), así
    que el producto de C1b tiene que entregarse ya en ese marco. La ventana **no
    se recalcula**: sale del QC de B1, que la declara por cubo, y sólo vale si
    B1 recortó UN cubo sin desplazamiento — que es el caso cuando lo que entra
    en B1 es un combinado ya centrado. Con N cubos o con desplazamiento, el
    marco de la cadena no es una ventana del combinado y esto falla en vez de
    entregar un cubo mal encuadrado.
    """

    bounds = stage01_qc.get("crop_bounds_per_cube") or []
    shifts = stage01_qc.get("spatial_shifts") or []
    if len(bounds) != 1:
        raise PerObservationSubtractError(
            f"B1 recortó {len(bounds)} cubos: el marco de la cadena no es una ventana del "
            "combinado y C1b no puede entregar su producto en él."
        )
    shift = shifts[0] if shifts else {"shift_y": 0.0, "shift_x": 0.0}
    if abs(float(shift.get("shift_y", 0.0))) > 1e-9 or abs(float(shift.get("shift_x", 0.0))) > 1e-9:
        raise PerObservationSubtractError(
            "B1 aplicó un desplazamiento al recortar: el marco de la cadena no es una ventana "
            "del combinado."
        )
    if wave_b1 is not None:
        if len(wave_plan) != len(wave_b1) or not np.allclose(wave_plan, wave_b1, rtol=0, atol=1e-6):
            raise PerObservationSubtractError(
                "El eje λ del combinado no coincide con el de B1: el recorte espacial no basta "
                "para llevar el residuo al marco de la cadena."
            )
    row = bounds[0]
    return int(row["y1"]), int(row["y2"]), int(row["x1"]), int(row["x2"])


def _companion_offsets(positions_qc, frame_shape, npix):
    """Compañero y fuente de campo en la ventana del combinado, por desplazamiento."""

    center = (npix // 2, npix // 2)
    frame_center = (int(frame_shape[0]) // 2, int(frame_shape[1]) // 2)
    out = []
    for key in ("companion", "field_source"):
        entry = positions_qc.get(key)
        if not entry:
            continue
        y, x = (float(v) for v in entry["pos_yx"])
        out.append((center[0] + y - frame_center[0], center[1] + x - frame_center[1]))
    return center, out


def make_subtract_transform(models, cfg, *, center_yx, exclude_centers, records):
    """El gancho que resta a cada exposición SU modelo, trozo a trozo."""

    fit_radius = float(cfg.get("x02_primary_fit_radius_px", 25.0))
    exclude_radius = float(cfg.get("x02_primary_exclude_radius_px", 8.0))

    def transform(exposure, wave_chunk, data, stat):
        model_doc = models.get(exposure.exposure_id)
        if model_doc is None:
            raise PerObservationSubtractError(
                f"{exposure.exposure_id}: el plan la combina pero la mezcla no la trae. "
                "Vuelve a correr C1 con psf_scope=per_observation."
            )
        model, _meta, amplitudes, backgrounds, n_fit = fit_primary_psf_amplitudes(
            data,
            wave_chunk,
            center_yx,
            model_doc,
            variance_zyx=stat,
            fit_radius_px=fit_radius,
            exclude_centers_yx=exclude_centers,
            exclude_radius_px=exclude_radius,
        )
        row = records.setdefault(exposure.exposure_id, {"wave_A": [], "amplitude": [],
                                                        "background": [], "n_fit": []})
        row["wave_A"].extend(float(w) for w in wave_chunk)
        row["amplitude"].extend(float(v) for v in amplitudes)
        row["background"].extend(float(v) for v in backgrounds)
        row["n_fit"].extend(int(v) for v in n_fit)
        # El fondo ajustado NO se resta, igual que en C3: lo que se quita es el
        # halo de la primaria, no el cielo.
        return data - model

    return transform


def run_stage_e01b(run_id=None, *, project_root=None, overrides=None,
                   allow_run_id_mismatch=False, progress=None):
    cfg = stage_e01b_config_from_run(run_id, project_root=project_root, overrides=overrides,
                                     allow_run_id_mismatch=allow_run_id_mismatch)
    paths = stage_e01b_paths(cfg["run_id"], project_root=cfg["project_root"])
    # La mezcla vive en su propio fichero; sólo se mira `psf_model.json` si C1
    # entregó la mezcla ahí (`psf_mixture_as_combined_model`).
    mixture_path = (paths["psf_model_mixture_json"] if paths["psf_model_mixture_json"].exists()
                    else paths["psf_model_json"])
    psf_model = read_json(mixture_path)
    models = models_by_exposure(psf_model)

    observations = resolve_observation_plan(
        cfg["run_id"],
        project_root=cfg["project_root"],
        plan_json=(paths["observation_plan_json"]
                   if paths["observation_plan_json"].exists() else None),
    )
    plan = observations.plan
    missing = [e.exposure_id for e in plan.exposures if e.exposure_id not in models]
    if missing:
        raise PerObservationSubtractError(
            f"{len(missing)} exposiciones del plan no tienen modelo en la mezcla: "
            f"{', '.join(missing[:3])}{'…' if len(missing) > 3 else ''}."
        )

    positions_qc = read_json(paths["stage01c_qc_json"])
    with fits.open(paths["stage02_cube_fits"], memmap=True) as hdul:
        frame_shape = tuple(int(v) for v in hdul["CUBES"].data.shape[-2:])
        wave_b1 = np.asarray(hdul["WAVELENGTH"].data, dtype=np.float64)
    center, exclude_centers = _companion_offsets(positions_qc, frame_shape, int(plan.crop_npix))
    window = b1_window(read_json(paths["stage01_qc_json"]), wavelength_axis(plan), wave_b1)

    records: dict[str, dict] = {}
    transform = make_subtract_transform(
        models, cfg, center_yx=center, exclude_centers=exclude_centers, records=records
    )
    result = combine_streaming(plan, progress=progress, transform=transform)

    # Se entrega YA en el marco de B1: es donde trabaja C3, y hacer el recorte
    # aqui —con la ventana que el propio B1 declara— evita que cada consumidor
    # se invente su encuadre.
    y1, y2, x1, x2 = window
    wave = wavelength_axis(plan)
    header = build_output_header(plan)
    hdul = fits.HDUList([
        fits.PrimaryHDU(),
        fits.ImageHDU(result["data"][:, y1:y2, x1:x2], header=header, name="DATA"),
        fits.ImageHDU(result["stat"][:, y1:y2, x1:x2], header=header, name="STAT"),
        fits.ImageHDU(wave.astype(np.float64), name="WAVELENGTH"),
        fits.ImageHDU(result["count"][:, y1:y2, x1:x2].astype(np.int16), name="COUNT"),
    ])
    paths["paths"].ensure_base_dirs()
    hdul.writeto(paths["cube_psfsub_fits"], overwrite=True)

    qc = _qc_payload(cfg, plan, observations, psf_model, result, records, center, exclude_centers,
                     window=window, frame_shape=frame_shape)
    write_json(paths["stage_e01b_qc_json"], qc)
    return {"paths": paths, "qc": qc, "cube": paths["cube_psfsub_fits"]}


def _qc_payload(cfg, plan, observations, psf_model, result, records, center, exclude_centers,
                *, window, frame_shape):
    per_exposure = []
    for exposure in plan.exposures:
        row = records.get(exposure.exposure_id) or {}
        amplitudes = np.asarray(row.get("amplitude", ()), dtype=float)
        finite = amplitudes[np.isfinite(amplitudes)]
        per_exposure.append({
            "exposure_id": exposure.exposure_id,
            "weight": float(exposure.weight),
            "n_channels_fitted": int(finite.size),
            "amplitude_median": float(np.median(finite)) if finite.size else None,
            "amplitude_p10": float(np.percentile(finite, 10)) if finite.size else None,
            "amplitude_p90": float(np.percentile(finite, 90)) if finite.size else None,
            "negative_amplitude_fraction": (float(np.mean(finite < 0)) if finite.size else None),
        })
    medians = [row["amplitude_median"] for row in per_exposure if row["amplitude_median"]]
    return {
        "stage": STAGE,
        "run_id": cfg["run_id"],
        "input": {
            "psf_model": str(psf_model.get("form")),
            "n_components": int(psf_model.get("n_components", 0)),
            "source_plan": observations.source,
        },
        "geometry": {
            "crop_npix": int(plan.crop_npix),
            # El producto sale en el marco de B1: la ventana es la que B1 declara,
            # no una recalculada aqui.
            "b1_window_y1y2x1x2": [int(v) for v in window],
            "b1_frame_shape": [int(v) for v in frame_shape],
            "delivered_frame": "B1",
            "primary_yx": [float(center[0]), float(center[1])],
            "excluded_centers_yx": [[float(y), float(x)] for y, x in exclude_centers],
            "fit_radius_px": float(cfg.get("x02_primary_fit_radius_px", 25.0)),
            "exclude_radius_px": float(cfg.get("x02_primary_exclude_radius_px", 8.0)),
        },
        "combine": result["qc"],
        "subtraction": {
            "n_exposures": len(per_exposure),
            "amplitude_median_spread_pct": (
                float(100.0 * (np.max(medians) - np.min(medians)) / np.median(medians))
                if len(medians) > 1 else None
            ),
            "note": ("La amplitud por canal es el flujo de la primaria en ESA exposición: su "
                     "dispersión entre exposiciones mide lo que el combinado promedia."),
            "per_exposure": per_exposure,
        },
        "open_issues": [],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="C1b: resta la PSF por exposición y combina los residuos."
    )
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--project-root", default=None)
    parser.add_argument("--allow-run-id-mismatch", action="store_true")
    args = parser.parse_args(argv)

    def _progress(z1, z2, nz):
        print(f"  canales {z1:5d}-{z2:5d} de {nz}", flush=True)

    result = run_stage_e01b(
        args.run_id,
        project_root=args.project_root,
        allow_run_id_mismatch=args.allow_run_id_mismatch,
        progress=_progress,
    )
    print(result["paths"]["stage_e01b_qc_json"])


__all__ = [
    "PerObservationSubtractError",
    "make_subtract_transform",
    "models_by_exposure",
    "run_stage_e01b",
    "stage_e01b_config_from_run",
    "stage_e01b_paths",
]


if __name__ == "__main__":
    main()
