"""Stage E01/C1: chromatic Moffat PSF model for extraction stages."""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from astropy.io import fits

from ..config import load_run_config
from .stage_e01_psfao import PSFAO_DEFAULT_WEIGHT_CAP, PSFAO_DEFAULT_WEIGHTING
from ..io import read_json, write_json
from ..paths import RunPaths
from ..psf import (
    build_psf_model_document,
    companion_ring_metric,
    corner_background,
    encircled_energy,
    encircled_energy_metric,
    evaluate_moffat_fit,
    evaluate_moffat_scene,
    evaluate_radial_profile,
    fit_moffat_image,
    psf_roundtrip_error,
    radial_hybrid_profile,
    source_mask,
)


@dataclass(frozen=True)
class StageE01Product:
    fit_rows: list[dict]
    psf_model: dict
    qc: dict
    hybrid_profiles: np.ndarray | None
    hybrid_radii: np.ndarray | None
    psf_form: str = "moffat"
    psfao_rows: list[dict] | None = None
    #: Con `psf_scope=per_observation`: el ajuste al cubo combinado y la mezcla
    #: de los modelos por exposicion. Cual de los dos va en `psf_model.json` lo
    #: decide `psf_mixture_as_combined_model` (por defecto, el del combinado);
    #: el otro se guarda al lado, porque la comparacion entre ambos es lo que
    #: mide el emborronado que introduce combinar.
    combined_psf_model: dict | None = None
    mixture_psf_model: dict | None = None
    observation_fits: list | None = None
    observation_plan: dict | None = None


def stage_e01_paths(run_id, project_root=None):
    root = Path(project_root or Path.cwd()).resolve()
    paths = RunPaths.from_project_root(run_id, root)
    plot_dir = paths.plot_stage_dir("stage_e01")
    return {
        "paths": paths,
        "stage02_cube_fits": paths.stage_dir / "stage02_xcorr_cube_stack.fits",
        "stage01c_qc_json": paths.stage_dir / "stage01c_qc.json",
        "stage_e01_params_csv": paths.stage_dir / "stage_e01_psf_params.csv",
        "psf_model_json": paths.stage_dir / "psf_model.json",
        "stage_e01_qc_json": paths.stage_dir / "stage_e01_qc.json",
        "psf_hybrid_residual_fits": paths.stage_dir / "psf_hybrid_residual.fits",
        # Productos del alcance `per_observation`. El modelo que consumen C2-E5
        # sigue llamandose `psf_model.json` a proposito: siete etapas lo leen por
        # ruta y todas pasan por `evaluate_psf_model`, asi que ensenandole la
        # forma `mixture` la cadena entera hereda el cambio sin tocarlas.
        "psf_perobs_dir": paths.stage_dir / "psf_perobs",
        "psf_model_perobs_json": paths.stage_dir / "psf_model_perobs.json",
        "psf_model_mixture_json": paths.stage_dir / "psf_model_mixture.json",
        "psf_model_combined_json": paths.stage_dir / "psf_model_combined.json",
        "observation_plan_json": paths.stage_dir / "observation_plan.json",
        "plot_dir": plot_dir,
        "summary_plot": plot_dir / "stage_e01_psf_summary.png",
    }


def stage_e01_config_from_run(
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
    cfg.setdefault("stage_e01_input_cube_fits", str(run_config.paths.stage_dir / "stage02_xcorr_cube_stack.fits"))
    cfg.setdefault("stage_e01_positions_qc", str(run_config.paths.stage_dir / "stage01c_qc.json"))
    cfg.setdefault("e01_psf_form", "auto")
    # DONDE se ajusta la PSF. `per_observation` (el default) la ajusta en cada
    # exposicion y entrega su mezcla, porque la mezcla de N PSF no es una PSF y
    # ajustarle una forma analitica al combinado sesga la razon nucleo/halo —que
    # es la correccion de apertura de C2/C3—. `combined` reproduce el camino
    # historico bit a bit.
    cfg.setdefault("psf_scope", "per_observation")
    cfg.setdefault("psf_perobs_max_workers", None)
    # El peso del ajuste POR EXPOSICION tiene su propio knob y su propio default
    # (`stat`): el del combinado (`relative`, tope 5) se eligio sobre un halo
    # promediado sobre 29 exposiciones, y en una sola ese halo es moteado. Ver
    # `stage_e01_perobs.PEROBS_DEFAULT_WEIGHTING` para los numeros medidos.
    cfg.setdefault("psf_perobs_fit_weighting", "stat")
    cfg.setdefault("psf_perobs_fit_weight_cap", None)
    cfg.setdefault("psf_bin_A", 100.0)
    cfg.setdefault("psf_min_channels_per_bin", 3)
    cfg.setdefault("psf_fit_radius_px", 28.0)
    cfg.setdefault("psf_norm_radius_px", 25.0)
    cfg.setdefault("psf_companion_ring_width_px", 3.0)
    cfg.setdefault("psf_mask_radius_factor", 3.0)
    cfg.setdefault("psf_sigma_clip", 3.0)
    cfg.setdefault("psf_max_clip_iter", 3)
    cfg.setdefault("psf_hybrid_threshold_pct", 5.0)
    cfg.setdefault("psf_hybrid_bin_fraction", 0.2)
    cfg.setdefault("psf_hybrid_smoothing_scale_factor", 2.0)
    cfg.setdefault("psf_save_plots", bool(cfg.get("save_intermediate_plots", False)))
    cfg.setdefault("stage_e01_bad_windows_A", _bad_windows_from_config(cfg))
    return cfg


def _bad_windows_from_config(cfg):
    windows = cfg.get("stage_e01_bad_windows_A") or cfg.get("bad_wavelength_ranges_A")
    if windows is not None:
        return windows
    if cfg.get("drop_wave_min_A") is not None and cfg.get("drop_wave_max_A") is not None:
        return [[float(cfg["drop_wave_min_A"]), float(cfg["drop_wave_max_A"])]]
    return []


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _maoppy_status(form_chosen=None):
    """Estado de maoppy EN ESTA EJECUCION, no solo si se puede importar.

    Antes devolvia siempre ``"available_not_used"`` con que el import
    funcionara, sin mirar si la rama psfao se habia elegido. Era una etiqueta de
    cuando maoppy era opcional y no se usaba, y quedaba contradiciendo a
    ``fit.model = "maoppy.Psfao"`` y ``model_comparison.form_chosen = "psfao"``
    en el mismo QC: quien auditara el fichero leia justo lo contrario de la
    verdad.
    """

    try:
        import maoppy  # noqa: F401
    except Exception as exc:
        return f"unavailable:{exc.__class__.__name__}"
    if form_chosen is None:
        return "available"
    return "used_selected" if str(form_chosen).lower() == "psfao" else "available_not_selected"


def _load_cube(path):
    with fits.open(path, memmap=True) as hdul:
        cubes = hdul["CUBES"].data.astype(np.float32)
        wavelengths = hdul["WAVELENGTH"].data.astype(np.float64)
        stat = hdul["STAT"].data.astype(np.float32) if "STAT" in hdul else None
    if cubes.ndim != 4:
        raise RuntimeError(f"Expected CUBES shape (N,nz,ny,nx), got {cubes.shape}.")
    return cubes, wavelengths, stat


def _good_wave_mask(wavelengths, bad_windows_A):
    wave = np.asarray(wavelengths, dtype=np.float64)
    good = np.isfinite(wave)
    for lo, hi in bad_windows_A or ():
        good &= ~((wave >= float(lo)) & (wave <= float(hi)))
    return good


def make_psf_bins(wavelengths, *, bin_A=100.0, bad_windows_A=(), min_channels=3):
    wave = np.asarray(wavelengths, dtype=np.float64)
    good = _good_wave_mask(wave, bad_windows_A)
    if np.count_nonzero(good) < int(min_channels):
        raise RuntimeError("Too few good wavelength channels for PSF bins.")
    wmin = float(np.nanmin(wave[good]))
    wmax = float(np.nanmax(wave[good]))
    edges = np.arange(wmin, wmax + float(bin_A), float(bin_A))
    bins = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = good & (wave >= lo) & (wave < hi)
        if np.count_nonzero(mask) >= int(min_channels):
            bins.append(
                {
                    "wave_min_A": float(lo),
                    "wave_max_A": float(hi),
                    "wave_center_A": float(np.nanmedian(wave[mask])),
                    "indices": np.where(mask)[0],
                }
            )
    if not bins:
        raise RuntimeError("No valid PSF bins after wavelength masking.")
    return bins


def _median_image(cubes, indices):
    with np.errstate(all="ignore"):
        return np.nanmedian(cubes[:, indices, :, :], axis=(0, 1)).astype(np.float64)


def _positions_from_qc(qc):
    primary = tuple(map(float, qc["primary"]["pos_yx"]))
    companion = tuple(map(float, qc["companion"]["pos_yx"]))
    field = None
    if qc.get("field_source") is not None:
        field = tuple(map(float, qc["field_source"]["pos_yx"]))
    return primary, companion, field


def binary_offset_px(cfg, positions_qc):
    """La segunda componente de la primaria, en pixeles, o `None`.

    ROXs 42B es una **binaria cercana no resuelta** y C1 ajustaba UNA PSF a DOS
    estrellas, midiendola mas ancha de lo que es y sesgando la `apcorr` un +6.9 %
    de forma cromatica (`docs/2026-08-30_binaria_42b_sesga_la_apcorr.md`).

    **La geometria se declara, no se ajusta.** Viene de la astrometria publicada
    —Keck/NIRC2 2022.621, rho = 51 +- 2 mas, PA = 148 +- 3 deg, la misma epoca
    que estos datos— asi que el ajuste gana **un solo** grado de libertad: la
    razon de flujos. Con la clave ausente se devuelve `None` y todo el camino es
    el de siempre, bit a bit; por eso ningun otro run se entera.

    Convencion: `PA` se mide del Norte hacia el Este; el cubo es norte-arriba con
    el este a la IZQUIERDA (`CD1_1 < 0`), asi que Norte es +y y Este es -x. La
    misma conversion, aplicada al compañero de ROXs 42B b, devuelve PA 270.4 deg
    contra los 271 publicados, y a la fuente de campo cc1 237 contra 241.4: por
    eso se sabe que el signo es el bueno.
    """

    # Valor ya resuelto: lo deja `compute_stage_e01_products` una sola vez, con
    # el mismo patron que E4 usa para `h04_injection_flux_sigma`. Hace falta
    # porque el ajuste POR EXPOSICION arma un `positions_qc` sintetico que no
    # lleva `pixel_scale_arcsec` -es de una exposicion, no del QC de B3-, asi
    # que alli no se puede recalcular. Y el offset es el MISMO en todas: es
    # relativo, y los cubos son norte-arriba.
    ya = cfg.get("e01_binary_offset_yx_px")
    if ya is not None:
        return (float(ya[0]), float(ya[1]))

    decl = cfg.get("e01_binary_companion")
    if not decl:
        return None
    escala = positions_qc.get("pixel_scale_arcsec")
    if escala is None:
        raise RuntimeError(
            "`e01_binary_companion` necesita `pixel_scale_arcsec` en stage01c_qc.json "
            "para pasar de milisegundos de arco a pixeles; no hay valor por defecto."
        )
    sep_px = float(decl["sep_mas"]) / (float(escala) * 1000.0)
    pa = math.radians(float(decl["pa_deg"]))
    # Norte = +y, Este = -x.
    return (sep_px * math.cos(pa), -sep_px * math.sin(pa))


def _b3_chromatic_track(path):
    path = Path(path)
    if not path.exists():
        return None, "unavailable:missing_track"
    rows = []
    with open(path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            try:
                rows.append((
                    0.5 * (float(row["wave_min_A"]) + float(row["wave_max_A"])),
                    float(row["primary_y"]),
                    float(row["primary_x"]),
                ))
            except (KeyError, TypeError, ValueError):
                continue
    if not rows:
        return None, "unavailable:empty_or_invalid_track"
    rows.sort(key=lambda value: value[0])
    return np.asarray(rows, dtype=np.float64), "used"


def _fotocentro(y, x, row, binary_offset_yx):
    """Del centro AJUSTADO al fotocentro, que es lo que traquea B3.

    Con una segunda componente ligada, C1 ajusta la posicion de la **primaria**;
    el track cromatico de B3 mide el **fotocentro del par sin resolver**. Los dos
    difieren por `f/(1+f) x separacion`, hacia la secundaria, y compararlos sin
    corregir hace saltar una issue bloqueante sobre un modelo correcto.

    Medido el 2026-09-01 en ROXs 42B b comparando las dos corridas: el centro
    ajustado se desplazo **0.2187 px** con la componente perpendicular en
    **0.003 px** -o sea enteramente a lo largo del eje de la binaria- contra los
    0.2313 px que predice la formula. Y la metrica pasa de **0.3233 a 0.2836 px**
    -por debajo del limite de 0.3- al reconstruir el fotocentro.

    Sin segunda componente devuelve la posicion tal cual.
    """

    if binary_offset_yx is None:
        return y, x
    try:
        f = float(row.get("flux_ratio"))
    except (TypeError, ValueError):
        return y, x
    if not np.isfinite(f) or f <= 0:
        return y, x
    peso = f / (1.0 + f)
    return y + peso * float(binary_offset_yx[0]), x + peso * float(binary_offset_yx[1])


def _centroid_vs_b3(rows, form, track, image_shape=None, binary_offset_yx=None):
    if track is None:
        return None
    wave_ref, y_ref, x_ref = track.T
    if form == "psfao":
        if image_shape is None:
            raise ValueError("image_shape is required for Psfao absolute centroids.")
        center_y, center_x = image_shape[0] // 2, image_shape[1] // 2
        values = [
            (float(row["lambda_A"]),
             *_fotocentro(center_y + float(row["dy"]), center_x + float(row["dx"]),
                          row, binary_offset_yx))
            for row in rows if row.get("status") == "ok"
        ]
    else:
        values = [
            (float(row["wave_center_A"]),
             *_fotocentro(float(row["y0"]), float(row["x0"]), row, binary_offset_yx))
            for row in rows if row.get("success", True)
        ]
    if not values:
        return None
    values = np.asarray(values, dtype=np.float64)
    expected_y = np.interp(values[:, 0], wave_ref, y_ref)
    expected_x = np.interp(values[:, 0], wave_ref, x_ref)
    return float(np.nanmax(np.hypot(values[:, 1] - expected_y, values[:, 2] - expected_x)))


def _ring_qc_summary(ring_after, p90_values, cfg):
    ring_after = np.asarray(ring_after, dtype=np.float64)
    p90_values = np.asarray(p90_values, dtype=np.float64)
    threshold = float(cfg.get("psf_hybrid_threshold_pct", 5.0))
    bins_above = int(np.count_nonzero(ring_after > threshold))
    fail_fraction = float(bins_above / max(ring_after.size, 1))
    return {
        "median": float(np.nanmedian(ring_after)) if ring_after.size else float("nan"),
        "p90": float(np.nanpercentile(p90_values, 90)) if p90_values.size else float("nan"),
        "bins_above": bins_above,
        "issue": fail_fraction > float(cfg.get("psf_hybrid_bin_fraction", 0.2)),
    }


def _fit_source_mask(shape, primary_yx, companion_yx, field_yx, mask_radius_px):
    centers = [companion_yx]
    if field_yx is not None:
        centers.append(field_yx)
    return source_mask(shape, centers, float(mask_radius_px))


def _detect_core_mask(image, primary_yx, cfg):
    threshold = cfg.get("psf_saturation_threshold")
    if threshold is None:
        return 0.0, False
    y, x = map(int, map(round, primary_yx))
    peak = float(image[y, x])
    if peak > float(threshold):
        return float(cfg.get("psf_core_mask_px", 2.0)), True
    return 0.0, False


def _row_from_fit(bin_index, bin_info, fit, metric, metric_hybrid=None):
    p = fit.params
    row = {
        "bin_index": int(bin_index),
        "wave_min_A": float(bin_info["wave_min_A"]),
        "wave_max_A": float(bin_info["wave_max_A"]),
        "wave_center_A": float(bin_info["wave_center_A"]),
        "n_channels": int(len(bin_info["indices"])),
        "success": bool(fit.success),
        "background": float(fit.background),
        "amplitude": float(p["amplitude"]),
        "y0": float(p["y0"]),
        "x0": float(p["x0"]),
        "fwhm_maj": float(p["fwhm_maj"]),
        "fwhm_min": float(p["fwhm_min"]),
        "theta_deg": float(p["theta_deg"]),
        "beta": float(p["beta"]),
        "chi2r": float(fit.chi2r),
        "clip_frac": float(fit.clip_frac),
        "n_fit": int(fit.n_fit),
        "ring_residual_pct": float(metric["median_pct"]),
        "ring_residual_p90_pct": float(metric["p90_pct"]),
    }
    if fit.flux_ratio is not None:
        row["flux_ratio"] = float(fit.flux_ratio)
    for key, value in fit.errors.items():
        row[f"{key}_err"] = None if not np.isfinite(value) else float(value)
    if metric_hybrid is not None:
        row["ring_residual_pct_after_hybrid"] = float(metric_hybrid["median_pct"])
        row["ring_residual_p90_pct_after_hybrid"] = float(metric_hybrid["p90_pct"])
    return row


#: En que formas se ajusta la segunda componente de la primaria. **Solo psfao**,
#: y no por gusto: la rama Moffat de C1 ajusta a `psf_fit_radius_px` (78 px en
#: ROXs 42B b) con recorte sigma, que quita el nucleo dominante para que la
#: Moffat pueda describir el HALO. Sin nucleo, la razon de flujos no esta
#: constrenida: medido el 2026-08-31, `f` se pega a su cota (1.0000 a 5300 A,
#: 0.8605 a 7200) o se colapsa a 0 (8800). Forzar el nucleo dentro del ajuste
#: tampoco vale -se probo-: colapsa la Moffat al nucleo y dispara el chi2r de
#: 1.25 a 2246, y ese `fwhm` roto alimenta la escala del hibrido, asi que la
#: averia llega hasta un numero publicado.
#:
#: **Consecuencia que hay que declarar**: `model_comparison` deja de comparar
#: peras con peras en un objeto con binaria -psfao la ve y Moffat no-. Aqui no
#: elige nada, porque la forma viene forzada por `e01_psf_form`; en un run que
#: la eligiera por el anillo, habria que mirarlo antes.
BINARY_FORMS_DEFAULT = ("psfao",)


def _moffat_fit_rows(cubes, wavelengths, bins, positions_qc, cfg):
    primary_yx, companion_yx, field_yx = _positions_from_qc(positions_qc)
    formas = tuple(cfg.get("e01_binary_forms", BINARY_FORMS_DEFAULT))
    binary_offset = binary_offset_px(cfg, positions_qc) if "moffat" in formas else None
    fwhm_prelim = float(positions_qc.get("psf", {}).get("fwhm_px", cfg.get("psf_prelim_fwhm_px", 4.0)))
    mask_radius = float(cfg.get("psf_companion_mask_radius_px", cfg.get("psf_mask_radius_factor", 3.0) * fwhm_prelim))
    rows = []
    images = []
    # Se devuelven las ESCENAS, no las PSF. Sus dos unicos consumidores -la
    # energia encerrada V4 y el hibrido de la §3.5- comparan modelo contra DATO,
    # y el dato de una binaria tiene dos estrellas: darles la PSF sola haria que
    # V4 viera un deficit igual a la razon de flujos (~11 % en ROXs 42B b) y
    # denunciara como fallo del modelo lo que es la secundaria que falta. Sin
    # binaria declarada escena y PSF son la MISMA imagen.
    scenes = []
    masks = []
    core_masks = []
    for i, bin_info in enumerate(bins):
        image = _median_image(cubes, bin_info["indices"])
        mask = _fit_source_mask(image.shape, primary_yx, companion_yx, field_yx, mask_radius)
        core_mask_px, saturation = _detect_core_mask(image, primary_yx, cfg)
        background = corner_background(image)
        fit = fit_moffat_image(
            image,
            center_yx=primary_yx,
            fit_radius_px=float(cfg.get("psf_fit_radius_px", 28.0)),
            mask=mask,
            background=background,
            core_mask_px=core_mask_px,
            sigma_clip=cfg.get("psf_sigma_clip", 3.0),
            max_iter=int(cfg.get("psf_max_clip_iter", 3)),
            companion_offset_yx=binary_offset,
        )
        # Los residuos se miden contra la ESCENA, porque es lo que el dato
        # contiene. La PSF de una fuente puntual -lo que se publica- no se
        # construye aqui: sale de `build_psf_model_document` a partir de las
        # FILAS, que describen una sola fuente. Medir el residuo contra ella
        # dejaria la secundaria entera dentro y el anillo empeoraria justo al
        # mejorar el modelo.
        scene = evaluate_moffat_scene(image.shape, fit)
        metric = companion_ring_metric(
            image,
            scene,
            primary_yx,
            companion_yx,
            width_px=float(cfg.get("psf_companion_ring_width_px", 3.0)),
            source_exclusion_radius_px=mask_radius,
        )
        row = _row_from_fit(i, bin_info, fit, metric)
        row["core_mask_px"] = float(core_mask_px)
        row["saturation_detected"] = bool(saturation)
        rows.append(row)
        images.append(image)
        scenes.append(scene)
        masks.append(mask)
        core_masks.append(core_mask_px)
    return rows, images, scenes, masks, {
        "mask_radius_px": mask_radius,
        "core_mask_px_max": float(np.nanmax(core_masks)) if core_masks else 0.0,
        "saturation_detected": bool(any(row["saturation_detected"] for row in rows)),
    }


def _encircled_energy_summary(images, models, backgrounds, primary_yx, exclude_mask, cfg):
    """V4 de la spec C1, por bin y resumida: la energia encapsulada del modelo
    contra la del dato.

    Existe porque la metrica que gobierna la etapa —el residuo del anillo en el
    radio del compañero— es CIEGA AL NUCLEO, y el modelo no se usa solo para el
    halo: la correccion de apertura de C2/C3 es exactamente `F(<=norm_radius) /
    F(box3)` evaluado sobre este modelo. Una forma puede clavar el anillo con un
    nucleo del todo equivocado, y sin esto se elegiria igualmente.
    """

    norm_radius = float(cfg.get("psf_norm_radius_px", 25.0))
    filas = []
    for image, model, background in zip(images, models, backgrounds):
        filas.append(encircled_energy_metric(
            image, model, primary_yx,
            norm_radius_px=norm_radius,
            # El fondo es el mismo numero para los dos: el modelo de C1 lleva
            # sumado el fondo fijo con el que se ajusto (§3.2), y la mezcla por
            # observacion se evalua a la escala del dato con ese mismo fondo.
            image_background=float(background), model_background=float(background),
            exclude_mask=exclude_mask,
        ))
    if not filas:
        return None
    def _mediana(clave):
        return float(np.nanmedian([f[clave] for f in filas]))
    return {
        "norm_radius_px": norm_radius,
        "box_size_px": 3,
        "core_ratio_data_median": _mediana("core_ratio_data"),
        "core_ratio_model_median": _mediana("core_ratio_model"),
        "core_ratio_error_pct_median": _mediana("core_ratio_error_pct"),
        "core_ratio_error_pct_p90": float(np.nanpercentile(
            np.abs([f["core_ratio_error_pct"] for f in filas]), 90)),
        "growth_curve_max_abs_diff_pct_median": _mediana("growth_curve_max_abs_diff_pct"),
        "n_bins": int(len(filas)),
    }


def _apply_hybrid(ring_pcts, images, models, masks, primary_yx, companion_yx, fwhm_med, cfg):
    """Add the azimuthal-median residual (AO ring) hybrid term to the chosen
    form's per-bin models when the ring metric fails on >20% of bins (spec §3.5).

    Form-agnostic: ``models`` are per-bin model images (Moffat evaluations or
    Psfao reconstructions). The smoothing scale (>=2*FWHM) and the azimuthal
    symmetry of ``radial_hybrid_profile`` guarantee it cannot absorb the
    (masked) companion. Returns
    ``(models_hybrid, profiles, radii_ref, applied, after_pcts)``."""

    threshold = float(cfg.get("psf_hybrid_threshold_pct", 5.0))
    frac_limit = float(cfg.get("psf_hybrid_bin_fraction", 0.2))
    values = np.asarray(ring_pcts, dtype=np.float64)
    fail_frac = float(np.count_nonzero(values > threshold) / max(values.size, 1))
    if fail_frac <= frac_limit:
        return list(models), None, None, False, list(map(float, ring_pcts)), None

    fwhm_med = float(fwhm_med)
    smooth = max(2.0 * fwhm_med, float(cfg.get("psf_hybrid_smoothing_scale_factor", 2.0)) * fwhm_med)
    width = float(cfg.get("psf_companion_ring_width_px", 3.0))
    excl = float(cfg.get("psf_companion_mask_radius_px", cfg.get("psf_mask_radius_factor", 3.0) * fwhm_med))
    profiles = []
    radii_ref = None
    models_hybrid = []
    after_pcts = []
    after_p90s = []
    for image, model, mask in zip(images, models, masks):
        radii, profile = radial_hybrid_profile(
            image - model,
            primary_yx,
            mask=mask,
            smoothing_scale_px=smooth,
        )
        hybrid = evaluate_radial_profile(image.shape, primary_yx, radii, profile)
        model_h = model + hybrid
        metric_h = companion_ring_metric(
            image,
            model_h,
            primary_yx,
            companion_yx,
            width_px=width,
            source_exclusion_radius_px=excl,
        )
        after_pcts.append(float(metric_h["median_pct"]))
        after_p90s.append(float(metric_h["p90_pct"]))
        if radii_ref is None:
            radii_ref = radii
        if radii.size != radii_ref.size:
            profile = np.interp(radii_ref, radii, profile)
        profiles.append(profile)
        models_hybrid.append(model_h)
    # Safety guard: the hybrid term must reduce the ring residual. When the base
    # form already models the AO halo (Psfao), the azimuthal-residual term is
    # degenerate and — scaled by the inflated Moffat FWHM — can make the metric
    # WORSE. In that case discard it rather than corrupt a good model.
    before_med = float(np.nanmedian(values))
    after_med = float(np.nanmedian(after_pcts))
    if not (np.isfinite(after_med) and after_med < before_med):
        return list(models), None, None, False, list(map(float, ring_pcts)), None
    return models_hybrid, np.asarray(profiles, dtype=np.float32), radii_ref.astype(np.float32), True, after_pcts, after_p90s


def _binary_companion_qc(cfg, positions_qc, moffat_rows, psfao):
    """Que se declaro, con que procedencia, y que salio — o `None` si no aplica.

    Sin esto el cambio no seria auditable desde el QC: se sabria que la `apcorr`
    cambio y no por que. Es el mismo agujero que ya tiene la mascara de la fuente
    de campo, que se aplica y no se declara.
    """

    decl = cfg.get("e01_binary_companion")
    if not decl:
        return None
    offset = binary_offset_px(cfg, positions_qc)

    def _resumen(filas):
        vals = [float(r["flux_ratio"]) for r in (filas or [])
                if r.get("flux_ratio") is not None and np.isfinite(r.get("flux_ratio", np.nan))]
        if not vals:
            return None
        waves = [float(r.get("wave_center_A", r.get("lambda_A", np.nan))) for r in filas
                 if r.get("flux_ratio") is not None and np.isfinite(r.get("flux_ratio", np.nan))]
        azul = [v for w, v in zip(waves, vals) if np.isfinite(w) and w < 6000.0]
        rojo = [v for w, v in zip(waves, vals) if np.isfinite(w) and w > 8000.0]
        return {
            "n_bins": len(vals),
            "median": float(np.median(vals)),
            "std": float(np.std(vals)),
            "median_blue_lt6000A": float(np.median(azul)) if azul else None,
            "median_red_gt8000A": float(np.median(rojo)) if rojo else None,
        }

    return {
        "declared": {"sep_mas": float(decl["sep_mas"]), "pa_deg": float(decl["pa_deg"])},
        "fitted_in_forms": list(cfg.get("e01_binary_forms", BINARY_FORMS_DEFAULT)),
        "source": cfg.get("e01_binary_companion_source"),
        "pixel_scale_arcsec": float(positions_qc.get("pixel_scale_arcsec")),
        "offset_yx_px": list(offset) if offset else None,
        "separation_px": float(np.hypot(*offset)) if offset else None,
        "flux_ratio_moffat": _resumen(moffat_rows),
        "flux_ratio_psfao": _resumen(psfao.get("rows") if isinstance(psfao, dict) else None),
        "note": (
            "Segunda componente LIGADA de la primaria: misma forma de PSF y "
            "separacion fija por astrometria publicada, asi que el ajuste gana UN "
            "grado de libertad (la razon de flujos). `psf_model.json` sigue siendo "
            "la PSF de UNA fuente puntual -sin la secundaria dentro-, que es lo que "
            "necesitan la inyeccion de E4, la `apcorr` y el throughput. La "
            "sustraccion de la primaria en psffit/C1b NO usa esto todavia. "
            "`fitted_in_forms` dice en que formas se ajusta: en la rama Moffat, a "
            "radio 78 px y con recorte, la razon de flujos no esta constrenida "
            "-se pega a su cota o se colapsa-, asi que `model_comparison` NO "
            "compara peras con peras en un objeto con binaria."
        ),
    }


def _run_psfao_branch(cfg, stage_dir, primary_yx, companion_yx, field_yx=None,
                      binary_offset_yx=None):
    """Fit the physical AO (Psfao) model per bin and score each reconstruction
    with the SAME canonical companion-ring metric used for Moffat, so §3.4 is an
    apples-to-apples comparison. Returns a dict with status and, on success, the
    per-bin rows, reconstructions, per-bin ring pcts and the psfao inputs.
    Never raises for a missing maoppy: returns ``status='unavailable:...'``."""

    try:
        import maoppy  # noqa: F401
        from .stage_e01_psfao import (
            build_psfao_model_document,
            fit_psfao_bins,
            prepare_psfao_inputs,
        )
    except Exception as exc:  # pragma: no cover - environment dependent
        return {"status": f"unavailable:{exc.__class__.__name__}"}

    inp = prepare_psfao_inputs(cfg, stage_dir)
    rows, recons = fit_psfao_bins(
        inp["cube"], inp["stat"], inp["wave"], inp["bins"],
        inp["system"], inp["companion"], inp["mask_radius"], inp["fit_radius"],
        # MISMA mascara que la rama Moffat: sin esto `model_comparison` compara
        # dos ajustes con contaminantes distintos.
        field_yx=field_yx,
        # Rescate de los bins que se estancan en el vector de arranque comun.
        # Cada hueco que deja un bin caido lo cruza `_evaluate_psfao` con una
        # recta, y ahi nacen las mesetas del modelo cromatico.
        warm_start=bool(cfg.get("psf_warm_start", True)),
        # QUE PARTE DE LA IMAGEN manda en el ajuste. El default (`stat`) es el
        # historico y no cambia ningun run que no lo declare.
        weighting=str(cfg.get("psf_fit_weighting", PSFAO_DEFAULT_WEIGHTING)),
        weight_cap=cfg.get("psf_fit_weight_cap", PSFAO_DEFAULT_WEIGHT_CAP),
        # La segunda componente ligada de la primaria. `None` -> `fit_bin` de
        # siempre, sin tocar `maoppy.psffit`.
        companion_offset_yx=binary_offset_yx,
    )
    if not recons:
        return {"status": "unavailable:no_valid_fits", "rows": rows}
    width = float(cfg.get("psf_companion_ring_width_px", 3.0))
    excl = float(inp["mask_radius"])
    ring_rows = []
    for mid in sorted(recons):
        image, recon = recons[mid]
        metric = companion_ring_metric(
            image, recon, primary_yx, companion_yx,
            width_px=width, source_exclusion_radius_px=excl,
        )
        ring_rows.append({
            "wave_center_A": float(mid),
            "ring_residual_pct": float(metric["median_pct"]),
            "ring_residual_p90_pct": float(metric["p90_pct"]),
        })
    rows_by_wave = {float(row["lambda_A"]): row for row in rows}
    for ring_row in ring_rows:
        row = rows_by_wave[ring_row["wave_center_A"]]
        row["ring_residual_pct_canonical"] = ring_row["ring_residual_pct"]
        row["ring_residual_p90_pct_canonical"] = ring_row["ring_residual_p90_pct"]
    return {
        "status": "ok",
        "rows": rows,
        "recons": recons,
        "ring_rows": ring_rows,
        "inp": inp,
        "build_doc": build_psfao_model_document,
    }


def _mixture_bin_models(mixture, waves_A, shape, primary_yx, fluxes, backgrounds):
    """Un documento de PSF evaluado por bin, PUESTO A LA ESCALA DEL DATO.

    Sirve para cualquier forma, no sólo `mixture`: se usa también con el ajuste
    analítico al combinado, para poder compararlos con la misma vara.


    La mezcla es una PSF normalizada a 1 dentro de `norm_radius_px`, así que
    para compararla con el dato —anillo y V4— hay que devolverle la escala: el
    flujo de la estrella dentro de ese mismo radio, medido en el propio bin, más
    el fondo con el que se midió. Así el modelo entra en las dos métricas con la
    misma convención que los modelos analíticos de las dos ramas.
    """

    from ..psf import evaluate_psf_model

    yy, xx = np.indices(shape, dtype=np.float64)
    dy = yy - float(primary_yx[0])
    dx = xx - float(primary_yx[1])
    models = []
    for wave, flux, background in zip(waves_A, fluxes, backgrounds):
        psf = evaluate_psf_model(mixture, float(wave), dy, dx)
        models.append(psf * float(flux) + float(background))
    return models


def _per_observation_model(cfg, model_doc, chosen, images, rows, primary_yx, companion_yx,
                           field_yx, ee_mask, ee_chosen, open_issues):
    """Ajusta la PSF en cada exposición y devuelve la mezcla que describe al combinado.

    Devuelve `(mezcla, modelo_del_combinado, bloque_qc, ajustes, plan)`. El
    bloque de QC lleva la comparación que justifica el cambio: la V4 —el
    cociente `F(≤norm)/F(box3)`, que ES la corrección de apertura— de la forma
    analítica ajustada al combinado contra la de la mezcla, las dos medidas
    sobre el mismo dato y con la misma máscara.
    """

    from ..config import ConfigError
    from ..observations import (
        ObservationInputError,
        ObservationPlanMissing,
        resolve_observation_plan,
    )
    from .stage_e01_perobs import fit_all_observations, mixture_from_fits

    paths = stage_e01_paths(cfg["run_id"], project_root=cfg.get("project_root"))
    cache = paths["observation_plan_json"]
    try:
        observations = resolve_observation_plan(
            cfg["run_id"],
            project_root=cfg.get("project_root"),
            plan_json=cache if cache.exists() else None,
        )
    except (ObservationPlanMissing, ConfigError, FileNotFoundError) as exc:
        # El objeto no declara exposiciones (reduccion monolitica, run
        # historico, escena sintetica): aqui NO aplica ajustar por observacion.
        # Se cae al camino del combinado, pero dejando dicho por que — un
        # fallback silencioso en la eleccion de PSF es exactamente lo que no
        # puede pasar.
        open_issues.append(
            "psf_scope=per_observation requested but this run declares no exposures "
            f"({exc.__class__.__name__}); the PSF was fitted on the combined cube instead."
        )
        return model_doc, model_doc, None, {
            "status": "unavailable",
            "reason": str(exc).splitlines()[0],
            "scope_used": "combined",
        }, None, None
    except ObservationInputError:
        if not cache.exists():
            raise
        # Una vista guardada que ya no vale (cubos movidos) no puede tapar la
        # buena: se rehace desde el plan del combinado.
        observations = resolve_observation_plan(
            cfg["run_id"], project_root=cfg.get("project_root")
        )

    # La forma NO la decide cada exposición: su residuo de anillo, medido a la
    # separación del compañero, es ruido con 300 s de integración (30-310 %
    # contra el 8-20 % del combinado). Decide quien tiene S/N: el config si la
    # congela, y si no, la que ganó en el cubo combinado.
    forced = str(cfg.get("e01_psf_form", "auto")).lower()
    form_for_exposures = chosen if forced == "auto" else forced
    observation_fits, failures = fit_all_observations(
        observations,
        cfg,
        {"companion": companion_yx, "field": field_yx},
        images[0].shape,
        forced_form=form_for_exposures,
        n_jobs=cfg.get("psf_perobs_max_workers"),
    )
    system_name = model_doc.get("system", "muse_nfm")
    mixture = mixture_from_fits(
        observation_fits, cfg, system_name=system_name, provenance=observations.provenance()
    )

    waves = [float(row["wave_center_A"]) for row in rows]
    backgrounds = [float(row["background"]) for row in rows]
    fluxes = [
        float(
            encircled_energy(
                image,
                primary_yx,
                [float(cfg.get("psf_norm_radius_px", 25.0))],
                background=background,
                exclude_mask=ee_mask,
            )[0]
        )
        for image, background in zip(images, backgrounds)
    ]
    mixture_models = _mixture_bin_models(
        mixture, waves, images[0].shape, primary_yx, fluxes, backgrounds
    )
    ee_mixture = _encircled_energy_summary(
        images, mixture_models, backgrounds, primary_yx, ee_mask, cfg
    )
    # El juez del cambio tiene que medir las dos cosas con la MISMA vara.
    # `ee_chosen` sale de la rama que ganó, y cada rama trae lo suyo: Psfao mide
    # sobre sus reconstrucciones, en sus bins y con el `bck` de su ajuste,
    # mientras la mezcla se mide sobre las imágenes medianas con
    # `corner_background`. Comparar esos dos números decía en ROXs 12 b que la
    # mezcla EMPEORABA la V4 (+6.77 contra +6.04 %), cuando sobre la misma vara
    # la MEJORA (+6.77 contra +8.83 %) — medido el 2026-08-17. Así que aquí el
    # documento del combinado se evalúa igual que lo consume C2/C3, sobre los
    # mismos bins, imágenes, fondos y máscara que la mezcla.
    combined_models = _mixture_bin_models(
        model_doc, waves, images[0].shape, primary_yx, fluxes, backgrounds
    )
    ee_combined_same = _encircled_energy_summary(
        images, combined_models, backgrounds, primary_yx, ee_mask, cfg
    )
    ring_mixture = [
        float(
            companion_ring_metric(
                image, model, primary_yx, companion_yx,
                width_px=float(cfg.get("psf_companion_ring_width_px", 3.0)),
                source_exclusion_radius_px=float(cfg.get("psf_companion_mask_radius_px", 10.0)),
            )["median_pct"]
        )
        for image, model in zip(images, mixture_models)
    ]

    forms = {}
    for fit in observation_fits:
        forms[fit.form] = forms.get(fit.form, 0) + 1
    v4_per_exposure = [
        float(fit.summary["encircled_energy"]["core_ratio_error_pct_median"])
        for fit in observation_fits
        if fit.summary.get("encircled_energy")
    ]
    tolerance = float(cfg.get("psf_encircled_energy_tolerance_pct", 3.0))
    if ee_mixture is not None and abs(ee_mixture["core_ratio_error_pct_median"]) > tolerance:
        open_issues.append(
            "Encircled-energy check (C1 V4) fails for the per-observation mixture: "
            f"{ee_mixture['core_ratio_error_pct_median']:+.1f} pct off the data "
            f"(tolerance {tolerance:g} pct)."
        )
    if failures:
        open_issues.append(
            f"{len(failures)} of {len(observation_fits) + len(failures)} exposures could not be "
            "fitted; the mixture is built from the rest."
        )

    block = {
        "n_exposures_used": int(len(observation_fits)),
        "n_exposures_failed": int(len(failures)),
        "failures": failures,
        "forms_chosen": forms,
        "form_source": "config" if forced != "auto" else "combined_fit",
        "form_used": form_for_exposures,
        "ring_note": ("El residuo del anillo por exposición se mide y se publica, pero NO elige "
                      "la forma: a la separación del compañero una exposición de 300 s no tiene "
                      "señal para decidir."),
        "form_stable_across_exposures": bool(len(forms) == 1),
        "weight_mode": observations.plan.weight_mode,
        "combine_method": observations.plan.method,
        "source_plan": observations.source,
        "substituted_paths": list(observations.substituted),
        "remeasured": [dict(row) for row in observations.remeasured],
        "exposures": [fit.summary for fit in observation_fits],
        "v4_per_exposure_pct": {
            "median": float(np.nanmedian(v4_per_exposure)) if v4_per_exposure else None,
            "min": float(np.nanmin(v4_per_exposure)) if v4_per_exposure else None,
            "max": float(np.nanmax(v4_per_exposure)) if v4_per_exposure else None,
        },
        # El juez del cambio: las dos varas, sobre el MISMO dato combinado y con
        # la MISMA rejilla. Las dos cifras que se comparan (`combined_fit_v4_pct`
        # y `mixture_v4_pct`) salen ya de la misma medida.
        #
        # La mezcla NO es exacta por construccion para la V4, aunque el cociente
        # de las sumas del DATO si lo sea. MEDIDO el 2026-08-15 (ROXs 12 b, 7000
        # A): la razon nucleo/total de cada exposicion va de 3.24 a 81.88, el
        # cociente de las sumas sale 5.40 y el cubo en disco 5.48. Pero la mezcla
        # se monta con MODELOS, y cada uno llega con su propio error de V4
        # (mediana 4.2 %, hasta 9.8 % en ROXs 12 b): esos errores no se cancelan.
        # Medido el 2026-08-17 sobre la misma vara: mezcla +6.77 %, ajuste al
        # combinado +8.83 %. La mezcla gana, y las dos fallan la tolerancia.
        "delivered_vs_combined_fit": {
            "note": ("V4 = F(<=norm_radius)/F(box3) del modelo contra el del dato, que ES la "
                     "correccion de apertura de C2/C3. `combined_fit` es la forma analitica "
                     "ajustada al cubo combinado. Las dos se evaluan como las consume C2/C3, "
                     "sobre los mismos bins, imagenes, fondos y mascara: sin eso la "
                     "comparacion no significa nada, porque cada rama trae su propia rejilla."),
            "comparable": True,
            "delivered": ("mixture" if bool(cfg.get("psf_mixture_as_combined_model", True))
                          else "combined_fit"),
            "combined_fit_form": chosen,
            "combined_fit_v4_pct": (None if ee_combined_same is None
                                    else float(ee_combined_same["core_ratio_error_pct_median"])),
            "combined_fit_encircled_energy": ee_combined_same,
            # La de la rama, en SU rejilla: no es comparable con la mezcla y se
            # publica sólo para poder auditar de dónde sale la diferencia.
            "combined_fit_v4_pct_own_grid": (None if ee_chosen is None
                                             else float(ee_chosen["core_ratio_error_pct_median"])),
            "mixture_v4_pct": (None if ee_mixture is None
                               else float(ee_mixture["core_ratio_error_pct_median"])),
            "mixture_encircled_energy": ee_mixture,
            "mixture_ring_residual_pct_median": (float(np.nanmedian(ring_mixture))
                                                 if ring_mixture else None),
        },
    }
    # Qué se entrega como `psf_model.json`. Por defecto la MEZCLA, y no es una
    # preferencia: mide mejor la cantidad que C2/C3 usan. Sobre la misma vara y
    # en el mismo dato (ROXs 12 b, 44 bins, 2026-08-17): mezcla +6.77 % de V4
    # contra +8.83 % del ajuste analítico, y el reparto en λ es lo que decide —
    # la mezcla clava el azul (+0.34 % contra +17.32 % por debajo de 6000 Å, que
    # es el problema abierto de C1) y paga en el rojo (+8.55 % contra +5.90 %).
    #
    # Lo que NO es cierto, aunque se escribió el 2026-08-15: que la mezcla sea
    # exacta por construcción. El cociente de las sumas del DATO sí lo es (5.40
    # contra 5.48 del cubo en disco), pero la mezcla se monta con MODELOS y cada
    # uno trae su propio error de V4, que no se cancela. Las dos siguen fallando
    # la tolerancia del 3 %.
    #
    # Y el rechazo de exposiciones no arregla esto por re-pesado: quitar de la
    # mezcla las 5 peores la lleva a −12.55 % y las 10 peores a −20.64 %, porque
    # el cubo combinado SIGUE conteniéndolas. Rechazar exige re-combinar.
    if bool(cfg.get("psf_mixture_as_combined_model", True)):
        delivered, kept = mixture, model_doc
    else:
        delivered, kept = model_doc, model_doc
    return delivered, kept, mixture, block, observation_fits, observations.to_json()


def compute_stage_e01_products(config) -> StageE01Product:
    cfg = dict(config)
    paths = stage_e01_paths(cfg["run_id"], project_root=cfg.get("project_root"))
    stage_dir = paths["paths"].stage_dir
    cube_path = Path(cfg.get("stage_e01_input_cube_fits", paths["stage02_cube_fits"]))
    positions_path = Path(cfg.get("stage_e01_positions_qc", paths["stage01c_qc_json"]))
    if not positions_path.exists():
        raise FileNotFoundError(f"Stage01c QC is required for C1: {positions_path}")
    positions_qc = read_json(positions_path)
    cubes, wavelengths, _ = _load_cube(cube_path)
    bins = make_psf_bins(
        wavelengths,
        bin_A=float(cfg.get("psf_bin_A", 100.0)),
        bad_windows_A=cfg.get("stage_e01_bad_windows_A", []),
        min_channels=int(cfg.get("psf_min_channels_per_bin", 3)),
    )
    primary_yx, companion_yx, field_yx = _positions_from_qc(positions_qc)
    # Se resuelve UNA vez y viaja en el config: el ajuste por exposicion lo
    # necesita y alli no hay `pixel_scale_arcsec` con el que recalcularlo.
    _off = binary_offset_px(cfg, positions_qc)
    if _off is not None:
        cfg["e01_binary_offset_yx_px"] = [float(_off[0]), float(_off[1])]
    track_path = positions_path.parent.parent / "tables" / "stage01c_chromatic_centroids.csv"
    b3_track, b3_track_status = _b3_chromatic_track(track_path)
    sep_px = float(np.hypot(companion_yx[0] - primary_yx[0], companion_yx[1] - primary_yx[1]))

    # --- Moffat fit (always run: it is the tie-break form and the FWHM source
    # for the hybrid smoothing scale). -----------------------------------------
    rows, images, scenes, masks, mask_meta = _moffat_fit_rows(cubes, wavelengths, bins, positions_qc, cfg)
    fwhm_med = float(np.nanmedian([row["fwhm_maj"] for row in rows]))
    moffat_ring = np.asarray([row["ring_residual_pct"] for row in rows], dtype=np.float64)
    moffat_p90 = np.asarray([row["ring_residual_p90_pct"] for row in rows], dtype=np.float64)
    moffat_median = float(np.nanmedian(moffat_ring)) if moffat_ring.size else float("nan")

    # --- Psfao fit + selection (spec §3.4). ------------------------------------
    form_cfg = str(cfg.get("e01_psf_form", "auto")).lower()
    if form_cfg not in ("auto", "moffat", "psfao"):
        raise ValueError(f"e01_psf_form must be auto|moffat|psfao, got {form_cfg!r}.")
    # Forzar una forma decide el GANADOR, no silencia la medida: la §3.4 de la
    # spec pide que las dos formas se ajusten y se comparen en el QC. Antes,
    # `e01_psf_form=moffat` se saltaba psfao entero y el `model_comparison` se
    # quedaba sin la mitad -- justo cuando congelar la forma es lo que hace falta
    # para que un knob de ajuste no pueda voltearla sin que nadie mire.
    comparar = bool(cfg.get("e01_psf_compare_forms", True))
    psfao = {"status": "skipped"}
    if form_cfg in ("auto", "psfao") or comparar:
        psfao = _run_psfao_branch(cfg, stage_dir, primary_yx, companion_yx, field_yx,
                                  binary_offset_yx=binary_offset_px(cfg, positions_qc))
    psfao_ok = psfao.get("status") == "ok"
    psfao_ring = (
        np.asarray([r["ring_residual_pct"] for r in psfao["ring_rows"]], dtype=np.float64)
        if psfao_ok else np.asarray([], dtype=np.float64)
    )
    psfao_median = float(np.nanmedian(psfao_ring)) if psfao_ring.size else float("nan")

    # --- V4 (spec §7): energia encapsulada de CADA forma contra el dato. --------
    # Se mide antes del hibrido y con la MISMA exclusion de fuentes para las dos,
    # o una forma cobraria por pixeles que la otra enmascara.
    excl_radius = float(mask_meta["mask_radius_px"])
    if psfao_ok:
        excl_radius = max(excl_radius, float(psfao["inp"]["mask_radius"]))
    ee_mask = source_mask(images[0].shape, [companion_yx, field_yx], excl_radius)
    ee_moffat = _encircled_energy_summary(
        images, scenes, [row["background"] for row in rows], primary_yx, ee_mask, cfg)
    ee_psfao = None
    if psfao_ok:
        mids_ee = sorted(psfao["recons"])
        rows_by_wave = {float(r["lambda_A"]): r for r in psfao["rows"]}
        ee_psfao = _encircled_energy_summary(
            [psfao["recons"][m][0] for m in mids_ee],
            [psfao["recons"][m][1] for m in mids_ee],
            [float(rows_by_wave[m].get("bck", 0.0)) for m in mids_ee],
            primary_yx, ee_mask, cfg)

    if form_cfg == "moffat":
        chosen, reason = "moffat", "forced by config (e01_psf_form=moffat)"
    elif form_cfg == "psfao":
        if not psfao_ok:
            raise RuntimeError(f"e01_psf_form=psfao but the Psfao fit is unavailable: {psfao.get('status')}")
        chosen, reason = "psfao", "forced by config (e01_psf_form=psfao)"
    else:  # auto: lower median ring residual wins; tie -> moffat
        if psfao_ok and np.isfinite(psfao_median) and psfao_median < moffat_median:
            chosen = "psfao"
            reason = f"auto: psfao median ring residual {psfao_median:.2f}% < moffat {moffat_median:.2f}%"
        else:
            chosen = "moffat"
            if psfao_ok:
                reason = f"auto: moffat median ring residual {moffat_median:.2f}% <= psfao {psfao_median:.2f}% (tie->moffat)"
            else:
                reason = f"auto: psfao unavailable ({psfao.get('status')}); moffat by default"

    open_issues = []
    if not psfao_ok and form_cfg in ("auto", "psfao"):
        open_issues.append(f"Psfao comparison fit not run ({psfao.get('status')}).")

    norm_radius = float(cfg.get("psf_norm_radius_px", 25.0))
    chromatic = bool(positions_qc.get("chromatic", {}).get("chromatic_centroid_needed", False))
    excluded = cfg.get("stage_e01_bad_windows_A", [])

    if chosen == "moffat":
        chosen_models, hybrid_profiles, hybrid_radii, hybrid_applied, after_pcts, after_p90s = _apply_hybrid(
            [row["ring_residual_pct"] for row in rows], images, scenes, masks,
            primary_yx, companion_yx, fwhm_med, cfg,
        )
        if hybrid_applied:
            for row, after, after_p90 in zip(rows, after_pcts, after_p90s):
                row["ring_residual_pct_after_hybrid"] = float(after)
                row["ring_residual_p90_pct_after_hybrid"] = float(after_p90)
        model_doc = build_psf_model_document(
            [row["wave_center_A"] for row in rows], rows,
            form="moffat", norm_radius_px=norm_radius, hybrid=hybrid_applied,
        )
        waves_rt = np.linspace(rows[0]["wave_center_A"], rows[-1]["wave_center_A"], min(10, len(rows)))
        ring_after = np.asarray(after_pcts, dtype=np.float64) if hybrid_applied else moffat_ring
        p90_values = np.asarray(after_p90s, dtype=np.float64) if hybrid_applied else moffat_p90
        centroid_diff = _centroid_vs_b3(rows, "moffat", b3_track,
                                        binary_offset_yx=cfg.get("e01_binary_offset_yx_px"))
        bins_interpolated = []
        for lo, hi in excluded:
            if float(hi) >= rows[0]["wave_center_A"] and float(lo) <= rows[-1]["wave_center_A"]:
                bins_interpolated.append([float(lo), float(hi)])
        binning = {
            "bin_A": float(cfg.get("psf_bin_A", 100.0)),
            "n_bins": int(len(rows)),
            "excluded_windows_A": excluded,
            "bins_interpolated": bins_interpolated,
        }
        masks_qc = {
            "companion_radius_px": float(mask_meta["mask_radius_px"]),
            "chromatic_tracking": chromatic,
            "core_mask_px": float(mask_meta["core_mask_px_max"]),
            "saturation_detected": bool(mask_meta["saturation_detected"]),
        }
        fit_qc = {
            "form_chosen": "moffat",
            "maoppy_status": _maoppy_status("moffat"),
            "background_mode": "fixed_external",
            "clip_frac_max": float(np.nanmax([row["clip_frac"] for row in rows])),
            "chi2r_median": float(np.nanmedian([row["chi2r"] for row in rows])),
        }
        smoothing_qc = {
            "per_param_model": {key: model_doc["coefficients"][key]["model"] for key in model_doc["coefficients"]},
            "outlier_bins": [],
            "centroid_vs_b3_max_diff_px": centroid_diff,
            "centroid_vs_b3_status": b3_track_status,
        }
        psfao_rows = psfao["rows"] if psfao_ok else None
    else:  # psfao
        inp = psfao["inp"]
        psfao_rows = psfao["rows"]
        ring_rows = psfao["ring_rows"]
        recons = psfao["recons"]
        model_doc, meta = psfao["build_doc"](
            psfao_rows, inp["system"], inp["norm_radius"], inp["fit_radius"],
            # La rejilla de evaluacion viaja en el documento, que es donde la lee
            # `_evaluate_psfao`. Por defecto, el ancho de bin que se acaba de
            # ajustar; el config manda si lo declara.
            wave_bin_A=float(cfg.get("psfao_wave_bin_A", inp["bin_A"])),
            weighting=str(cfg.get("psf_fit_weighting", PSFAO_DEFAULT_WEIGHTING)),
            weight_cap=cfg.get("psf_fit_weight_cap", PSFAO_DEFAULT_WEIGHT_CAP),
        )
        mids = sorted(recons)
        p_images = [recons[m][0] for m in mids]
        p_models = [recons[m][1] for m in mids]
        p_masks = [source_mask(p_images[0].shape, [companion_yx], inp["mask_radius"]) for _ in mids]
        chosen_models, hybrid_profiles, hybrid_radii, hybrid_applied, after_pcts, after_p90s = _apply_hybrid(
            [r["ring_residual_pct"] for r in ring_rows], p_images, p_models, p_masks,
            primary_yx, companion_yx, fwhm_med, cfg,
        )
        if hybrid_applied:
            for r, after in zip(ring_rows, after_pcts):
                r["ring_residual_pct_after_hybrid"] = float(after)
            for r, after_p90 in zip(ring_rows, after_p90s):
                r["ring_residual_p90_pct_after_hybrid"] = float(after_p90)
            rows_by_wave = {float(row["lambda_A"]): row for row in psfao_rows}
            for ring_row in ring_rows:
                row = rows_by_wave[ring_row["wave_center_A"]]
                row["ring_residual_pct_after_hybrid_canonical"] = ring_row["ring_residual_pct_after_hybrid"]
                row["ring_residual_p90_pct_after_hybrid_canonical"] = ring_row["ring_residual_p90_pct_after_hybrid"]
            model_doc["hybrid"] = True
        waves = [r["wave_center_A"] for r in ring_rows]
        waves_rt = np.linspace(min(waves), max(waves), min(10, len(waves)))
        ring_after = np.asarray(after_pcts, dtype=np.float64) if hybrid_applied else psfao_ring
        p90_values = np.asarray(
            after_p90s if hybrid_applied else [r["ring_residual_p90_pct"] for r in ring_rows],
            dtype=np.float64,
        )
        ok_rows = [r for r in psfao_rows if r.get("status") == "ok"]
        centroid_diff = _centroid_vs_b3(ok_rows, "psfao", b3_track, image_shape=p_images[0].shape,
                                        binary_offset_yx=cfg.get("e01_binary_offset_yx_px"))
        bins_interpolated = [[float(lo), float(hi)] for lo, hi in inp["bad"]]
        binning = {
            "bin_A": float(inp["bin_A"]),
            "n_bins": int(len(inp["bins"])),
            "excluded_windows_A": inp["bad"],
            "bins_interpolated": bins_interpolated,
        }
        masks_qc = {
            "companion_radius_px": float(inp["mask_radius"]),
            "chromatic_tracking": chromatic,
            "core_mask_px": 0.0,
            "saturation_detected": False,
        }
        fit_qc = {
            "form_chosen": "psfao",
            "model": "maoppy.Psfao",
            "maoppy_status": _maoppy_status("psfao"),
            "background_mode": "psffit_flux_bck",
            "fit_radius_px": float(inp["fit_radius"]),
            "n_ok_bins": int(meta["n_ok"]),
            "n_bins_rejected": int(meta["n_rejected"]),
            "n_fit_failed": int(len(psfao_rows) - meta["n_ok"]),
            "clip_frac_max": 0.0,
            "chi2r_median": None,
            # Que parte de la imagen decidio el ajuste. Va aqui ademas de en
            # `psf_model.json` porque este es el camino canonico (el de la
            # comparacion de formas) y su QC es lo que lee F1.
            "weighting": str(cfg.get("psf_fit_weighting", PSFAO_DEFAULT_WEIGHTING)),
            "weight_cap": (None if cfg.get("psf_fit_weight_cap",
                                           PSFAO_DEFAULT_WEIGHT_CAP) is None
                           else float(cfg.get("psf_fit_weight_cap",
                                              PSFAO_DEFAULT_WEIGHT_CAP))),
        }
        smoothing_qc = {
            # `smoothed_poly` NO es lo que se evalua mientras exista `param_table`:
            # `_evaluate_psfao` interpola la tabla por bin y el polinomio queda
            # inerte. Se declara asi para que el QC no siga sugiriendo que el
            # modelo por canal es una parabola. Medido en el notebook de C1.
            "per_param_model": {name: f"polynomial_deg{max(len(v) - 1, 0)}" for name, v in model_doc["smoothed_poly"].items()},
            "per_param_model_is_evaluated": not bool(model_doc.get("param_table")),
            "evaluated_as": ("param_table_linear_interp" if model_doc.get("param_table")
                             else "smoothed_poly"),
            "outlier_bins": [],
            "centroid_vs_b3_max_diff_px": centroid_diff,
            "centroid_vs_b3_status": b3_track_status,
            "warm_start": bool(cfg.get("psf_warm_start", True)),
            # Las dos direcciones cuentan: un bin que solo tiene vecino bueno al
            # rojo se recupera en la pasada hacia atras, y dejarlo fuera de la
            # cuenta hacia parecer que el arranque caliente no habia hecho nada.
            "n_bins_rescued_by_warm_start": int(sum(
                1 for r in psfao_rows
                if r.get("start_vector") in ("warm_start", "warm_start_back")
                and r.get("status") == "ok")),
        }

    roundtrip = psf_roundtrip_error(model_doc, waves_rt)
    ring_summary = _ring_qc_summary(ring_after, p90_values, cfg)
    ring_median = ring_summary["median"]
    ring_p90 = ring_summary["p90"]

    if ring_summary["issue"]:
        open_issues.append("Companion-ring residual remains above 5 pct after the fixed C1 sequence.")
    if centroid_diff is not None and centroid_diff > 0.3 and chromatic:
        open_issues.append("PSF centroid differs from B3 chromatic centroid story by >0.3 px.")

    # V4 de la spec (§7): «curva de crecimiento modelo vs dato; acuerdo < 3%
    # hasta norm_radius_px». Estaba escrita y no estaba implementada, y es la
    # unica verificacion de C1 que mira el nucleo.
    ee_chosen = ee_moffat if chosen == "moffat" else ee_psfao
    ee_tolerance = float(cfg.get("psf_encircled_energy_tolerance_pct", 3.0))
    ee_ok = None
    if ee_chosen is not None:
        ee_ok = bool(abs(ee_chosen["core_ratio_error_pct_median"]) <= ee_tolerance)
        if not ee_ok:
            open_issues.append(
                "Encircled-energy check (C1 V4) fails for the chosen form: the model "
                f"F(<={ee_chosen['norm_radius_px']:g})/F(box3) is "
                f"{ee_chosen['core_ratio_error_pct_median']:+.1f} pct off the data "
                f"(tolerance {ee_tolerance:g} pct). That ratio IS the C2/C3 aperture correction."
            )
    # §3.2: «los anillos AO no deben cliparse: verificar que la fraccion clipeada
    # sea < 5% y anotarla». Se anotaba y no se verificaba.
    clip_limit = float(cfg.get("psf_clip_frac_max", 0.05))
    clip_max = float(fit_qc.get("clip_frac_max") or 0.0)
    if chosen == "moffat" and clip_max > clip_limit:
        open_issues.append(
            f"Moffat sigma-clipping removed {100 * clip_max:.1f} pct of the fit pixels "
            f"(spec C1 §3.2 limit: {100 * clip_limit:g} pct); the clipped pixels are the core."
        )

    # --- Alcance por observacion (spec C1 v2). --------------------------------
    # Hasta aqui todo ha ajustado el cubo COMBINADO, que es una media pesada de
    # 29-30 exposiciones con seeing distinto. La mezcla de N PSF no es una PSF:
    # ninguna forma analitica puede tener a la vez el nucleo de la mejor noche y
    # el halo de la peor, y ese sesgo cae entero sobre la razon nucleo/halo, que
    # es la correccion de apertura de C2/C3. Con `psf_scope=per_observation` se
    # ajusta cada exposicion por separado y se entrega su MEZCLA; el ajuste al
    # combinado se conserva al lado, porque su comparacion es el juez.
    scope = str(cfg.get("psf_scope", "per_observation")).lower()
    if scope not in ("combined", "per_observation"):
        raise ValueError(f"psf_scope must be combined|per_observation, got {scope!r}.")
    per_observation = None
    combined_model_doc = None
    observation_fits = None
    observation_plan = None
    scope_requested = scope
    mixture_model_doc = None
    if scope == "per_observation":
        (model_doc, combined_model_doc, mixture_model_doc, per_observation, observation_fits,
         observation_plan) = _per_observation_model(
            cfg, model_doc, chosen, images, rows, primary_yx, companion_yx, field_yx,
            ee_mask, ee_chosen, open_issues,
        )
        # Un objeto sin exposiciones declaradas se queda en el camino del
        # combinado, y el QC lo dice con nombre y motivo.
        if per_observation.get("status") == "unavailable":
            scope = "combined"

    model_comparison = {
        "metric": "companion_ring_metric.median_pct (canonical, applied to both forms)",
        "selection_mode": form_cfg,
        "form_chosen": chosen,
        "reason": reason,
        # El anillo elige, pero no es lo unico que hay que saber de una forma: se
        # publica al lado el error de la energia encapsulada (V4), que es el que
        # se propaga a la correccion de apertura de C2/C3.
        "secondary_metric": "encircled_energy.core_ratio_error_pct_median (C1 V4, informativa)",
        "moffat": {
            "ring_residual_pct_median": moffat_median,
            "ring_residual_pct_p90": float(np.nanpercentile(moffat_p90, 90)) if moffat_p90.size else None,
            "n_bins": int(len(rows)),
            "encircled_energy": ee_moffat,
        },
        "psfao": {
            "status": psfao.get("status"),
            "ring_residual_pct_median": psfao_median if psfao_ok else None,
            "ring_residual_pct_p90": float(np.nanpercentile(
                [r["ring_residual_p90_pct"] for r in psfao["ring_rows"]], 90)) if psfao_ok else None,
            "n_bins": int(len(psfao["ring_rows"])) if psfao_ok else 0,
            "encircled_energy": ee_psfao,
        },
    }

    qc = {
        "stage": "e01_chromatic_psf",
        "run_id": cfg["run_id"],
        "input": {"cube": str(cube_path), "sha256": _sha256(cube_path), "positions_from": str(positions_path)},
        "binning": binning,
        "binary_companion": _binary_companion_qc(cfg, positions_qc, rows, psfao),
        "masks": masks_qc,
        "fit": fit_qc,
        "smoothing": smoothing_qc,
        "companion_ring_metric": {
            "radius_px": sep_px,
            "width_px": float(cfg.get("psf_companion_ring_width_px", 3.0)),
            "residual_pct_median": ring_median,
            "residual_pct_p90": ring_p90,
            "bins_above_5pct": ring_summary["bins_above"],
            "after_hybrid": bool(hybrid_applied),
        },
        "hybrid": {
            "applied": bool(hybrid_applied),
            "smoothing_scale_px": None if not hybrid_applied else float(max(2.0 * fwhm_med, 0.0)),
        },
        "normalization": {
            "norm_radius_px": norm_radius,
            "roundtrip_error": float(roundtrip),
        },
        "encircled_energy": None if ee_chosen is None else {
            **ee_chosen,
            "spec_check": "V4",
            "tolerance_pct": ee_tolerance,
            "ok": ee_ok,
            "measured_on": "per-bin model before the hybrid term",
            "note": ("core_ratio = F(<=norm_radius)/F(box3): the same quantity C2/C3 "
                     "invert as the aperture correction."),
        },
        "model_comparison": model_comparison,
        "psf_scope": scope,
        "psf_scope_requested": scope_requested,
        "per_observation": per_observation,
        "open_issues": open_issues,
    }
    return StageE01Product(
        fit_rows=rows,
        psf_model=model_doc,
        qc=qc,
        hybrid_profiles=hybrid_profiles,
        hybrid_radii=hybrid_radii,
        # La forma sigue siendo la que gano en el COMBINADO: es la que manda
        # sobre los CSV por bin, el termino hibrido y la comparacion de la §3.4.
        # Que el documento entregado sea una mezcla se dice en `psf_scope`.
        psf_form=chosen,
        psfao_rows=psfao_rows,
        combined_psf_model=combined_model_doc,
        mixture_psf_model=mixture_model_doc,
        observation_fits=observation_fits,
        observation_plan=observation_plan,
    )


def write_stage_e01_products(product: StageE01Product, config, paths):
    paths["paths"].ensure_base_dirs()
    paths["plot_dir"].mkdir(parents=True, exist_ok=True)
    _write_csv(paths["stage_e01_params_csv"], product.fit_rows)
    if product.psf_form == "psfao" and product.psfao_rows:
        _write_psfao_csv(paths["paths"].stage_dir / "stage_e01_psfao_params.csv", product.psfao_rows)
    write_json(paths["psf_model_json"], product.psf_model)
    if product.combined_psf_model is not None:
        # El ajuste analitico al combinado, siempre al lado: es lo que
        # `psf_scope=combined` reproduce y la referencia de la comparacion.
        write_json(paths["psf_model_combined_json"], product.combined_psf_model)
    if product.mixture_psf_model is not None:
        # La mezcla, con los modelos por exposicion dentro. La lee C1b para
        # restar, aunque no sea lo que se entrega como modelo del combinado.
        write_json(paths["psf_model_mixture_json"], product.mixture_psf_model)
    if product.observation_plan is not None:
        write_json(paths["observation_plan_json"], product.observation_plan)
    if product.observation_fits:
        _write_per_observation_products(product, paths)
    if product.hybrid_profiles is not None:
        fits.HDUList(
            [
                fits.PrimaryHDU(),
                fits.ImageHDU(product.hybrid_profiles.astype(np.float32), name="PROFILE"),
                fits.ImageHDU(product.hybrid_radii.astype(np.float32), name="RADIUS_PX"),
            ]
        ).writeto(paths["psf_hybrid_residual_fits"], overwrite=True)
    else:
        # Si el hibrido NO se aplica, el residuo de una corrida ANTERIOR no puede
        # quedarse: describe un modelo que ya no existe y quien lo lea estara
        # mirando otra cosecha. Paso el 2026-08-31 en ROXs 42B b -el fichero
        # quedo fechado a las 04:41, de una corrida descartada, mientras el resto
        # del run era de las 20:35- y tumbo dos notebooks debug que se anclan
        # contra el. Es el agujero que `stage_vintage` vigila ENTRE etapas,
        # dentro de una sola.
        paths["psf_hybrid_residual_fits"].unlink(missing_ok=True)
    if bool(config.get("psf_save_plots", config.get("save_intermediate_plots", False))):
        _write_summary_plot(product, paths)
        product.qc["figures"] = {"summary": str(Path("plots") / "stage_e01" / paths["summary_plot"].name)}
    write_json(paths["stage_e01_qc_json"], product.qc)
    return {"params_csv": paths["stage_e01_params_csv"], "model_json": paths["psf_model_json"], "qc_json": paths["stage_e01_qc_json"], "qc": product.qc}


def _write_per_observation_products(product: StageE01Product, paths):
    """Los ajustes por exposición: el índice y las filas por bin de cada una.

    Los documentos de PSF de cada exposición NO se duplican aquí: viajan dentro
    de `psf_model.json`, que es la mezcla, y tenerlos dos veces invitaría a que
    alguien editara la copia que no se evalúa. Lo que sí se escribe aparte son
    las filas por bin —lo que un notebook querría dibujar— y el resumen.
    """

    directory = paths["psf_perobs_dir"]
    directory.mkdir(parents=True, exist_ok=True)
    index = []
    for fit in product.observation_fits:
        safe = str(fit.exposure_id).replace("/", "_")
        if fit.psfao_rows:
            _write_psfao_csv(directory / f"params_{safe}_psfao.csv", fit.psfao_rows)
        if fit.moffat_rows:
            _write_csv(directory / f"params_{safe}_moffat.csv", fit.moffat_rows)
        index.append(dict(fit.summary))
    write_json(
        paths["psf_model_perobs_json"],
        {
            "schema_version": 1,
            "run_id": product.qc.get("run_id"),
            "psf_scope": product.qc.get("psf_scope"),
            "n_exposures": len(index),
            "delivered_model": str(paths["psf_model_json"].name),
            "combined_fit_model": str(paths["psf_model_combined_json"].name),
            "exposures": index,
            "per_observation": product.qc.get("per_observation"),
        },
    )


def _write_csv(path, rows):
    if not rows:
        Path(path).write_text("", encoding="utf-8")
        return
    fieldnames = list(dict.fromkeys(key for row in rows for key in row.keys()))
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_psfao_csv(path, rows):
    from .stage_e01_psfao import PSFAO_PARAM_NAMES

    # Las `*_err` son las incertidumbres formales que `psffit` ya calculaba y que
    # `fit_bin` tiraba (ver `_psfao_param_errors`). Van detras de sus parametros
    # y son aditivas: nada aguas abajo las lee, y los CSV escritos antes de que
    # existieran siguen siendo legibles (el lector va por nombre de columna).
    cols = ["lambda_A", "samp", "amp", "bck", "dy", "dx", "ring_residual_pct",
            "ring_residual_pct_canonical", "ring_residual_p90_pct_canonical",
            *PSFAO_PARAM_NAMES, *(f"{name}_err" for name in PSFAO_PARAM_NAMES),
            "dy_err", "dx_err",
            # La razon de flujos de la segunda componente ligada. Se calculaba,
            # viajaba en las filas y **se tiraba al escribir**: solo sobrevivia
            # el resumen del QC. La cantidad de la que depende todo el cambio de
            # la binaria no puede ser inauditable por bin -y `_centroid_vs_b3` la
            # necesita para reconstruir el fotocentro-. Vacia donde no aplique,
            # igual que las `*_err`.
            "flux_ratio", "flux_ratio_err",
            "ring_residual_pct_after_hybrid_canonical",
            "ring_residual_p90_pct_after_hybrid_canonical", "optimizer_success",
            "optimizer_status", "optimizer_message", "optimizer_nfev", "optimizer_cost",
            "optimizer_stalled_at_initial", "start_vector", "status"]
    with open(path, "w", encoding="utf-8") as f:
        f.write(",".join(cols) + "\n")
        for r in rows:
            f.write(",".join(str(r.get(c, "")) for c in cols) + "\n")


def _write_summary_plot(product, paths):
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    rows = product.psfao_rows if product.psf_form == "psfao" else product.fit_rows
    rows = [row for row in rows if row.get("status", "ok") == "ok"]
    wave_key = "lambda_A" if product.psf_form == "psfao" else "wave_center_A"
    wave = np.asarray([row[wave_key] for row in rows], dtype=float)
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    if product.psf_form == "psfao":
        axes[0, 0].plot(wave, [row["r0"] for row in rows])
        axes[0, 0].set_ylabel("Psfao r0")
        axes[0, 1].plot(wave, [row["beta"] for row in rows])
        axes[0, 1].set_ylabel("Psfao beta")
        before_key = "ring_residual_pct_canonical"
        after_key = "ring_residual_pct_after_hybrid_canonical"
        axes[1, 1].plot(wave, [row["dy"] for row in rows], label="dy")
        axes[1, 1].plot(wave, [row["dx"] for row in rows], label="dx")
        axes[1, 1].set_ylabel("centroid offset [px]")
        axes[1, 1].legend()
    else:
        axes[0, 0].plot(wave, [row["fwhm_maj"] for row in rows], label="maj")
        axes[0, 0].plot(wave, [row["fwhm_min"] for row in rows], label="min")
        axes[0, 0].set_ylabel("FWHM [px]")
        axes[0, 0].legend()
        axes[0, 1].plot(wave, [row["beta"] for row in rows])
        axes[0, 1].set_ylabel("beta")
        before_key = "ring_residual_pct"
        after_key = "ring_residual_pct_after_hybrid"
        axes[1, 1].plot(wave, [row["chi2r"] for row in rows])
        axes[1, 1].set_ylabel("chi2r")
    axes[1, 0].plot(wave, [row[before_key] for row in rows], label=product.psf_form)
    if product.qc["hybrid"]["applied"] and all(after_key in row for row in rows):
        axes[1, 0].plot(wave, [row[after_key] for row in rows], label="hybrid")
    axes[1, 0].axhline(5.0, color="0.4", ls="--")
    axes[1, 0].set_ylabel("Ring residual [%]")
    axes[1, 0].legend()
    for ax in axes.ravel():
        ax.set_xlabel("Wavelength [A]")
    fig.savefig(paths["summary_plot"], dpi=150)
    plt.close(fig)


def run_stage_e01(run_id=None, *, project_root=None, overrides=None, allow_run_id_mismatch=False):
    cfg = stage_e01_config_from_run(
        run_id,
        project_root=project_root,
        overrides=overrides,
        allow_run_id_mismatch=allow_run_id_mismatch,
    )
    paths = stage_e01_paths(cfg["run_id"], project_root=cfg.get("project_root"))
    product = compute_stage_e01_products(cfg)
    written = write_stage_e01_products(product, cfg, paths)
    return {"config": cfg, "paths": paths, "qc": product.qc, "written": written}


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run Stage E01/C1 chromatic PSF model.")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--project-root", default=None)
    parser.add_argument("--allow-run-id-mismatch", action="store_true")
    parser.add_argument("--save-plots", action="store_true")
    parser.add_argument(
        "--psf-scope", choices=("per_observation", "combined"), default=None,
        help=("donde se ajusta la PSF: en cada exposicion y se entrega su mezcla "
              "(por defecto), o en el cubo combinado (el camino historico)."),
    )
    parser.add_argument(
        "--psf-perobs-max-workers", type=int, default=None,
        help="exposiciones ajustadas a la vez (cada una sostiene su cubo alineado).",
    )
    args = parser.parse_args(argv)
    overrides = {}
    if args.save_plots:
        overrides["psf_save_plots"] = True
    if args.psf_scope is not None:
        overrides["psf_scope"] = args.psf_scope
    if args.psf_perobs_max_workers is not None:
        overrides["psf_perobs_max_workers"] = int(args.psf_perobs_max_workers)
    overrides = overrides or None
    result = run_stage_e01(
        args.run_id,
        project_root=args.project_root,
        overrides=overrides,
        allow_run_id_mismatch=args.allow_run_id_mismatch,
    )
    print(result["paths"]["stage_e01_qc_json"])


__all__ = [
    "StageE01Product",
    "compute_stage_e01_products",
    "make_psf_bins",
    "run_stage_e01",
    "stage_e01_config_from_run",
    "stage_e01_paths",
    "write_stage_e01_products",
]


if __name__ == "__main__":
    main()
