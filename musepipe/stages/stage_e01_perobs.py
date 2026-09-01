"""C1 por observación: la PSF de CADA exposición, y la mezcla que describe al combinado.

Por qué existe
--------------
La cadena ajustaba una PSF analítica al cubo **combinado**. Pero el combinado
es una media pesada de 29–30 exposiciones con seeing y calidad de AO distintas,
y **la mezcla de N PSF no es una PSF**: ninguna Psfao ni Moffat puede tener a la
vez el núcleo de la mejor noche y el halo de la peor. Ajustarle una sola forma
sesga la razón núcleo/halo, que es exactamente la corrección de apertura que
aplican C2/C3 — el mismo mecanismo que en ROXs 42B b dejó la `apcorr` 4-5× de
más (traspaso del 2026-08-15).

Medido sobre una exposición suelta de ROXs 12 b (`residuos_debug` §7.d): con
dato y modelo igualados en energía dentro de 25 px, el dato tiene +27…+39 % más
luz dentro de 5 px que el modelo del combinado, y al modelo le sobra ~2× en el
anillo 25–32 px.

Qué hace
--------
1. Ajusta **las dos formas** (Moffat y Psfao) en cada exposición, con la
   geometría que el propio combinado usó — la del `stream_combine_plan.json`, vía
   `musepipe.observations` —, de modo que la primaria cae en el centro del array,
   que es donde `fit_bin` la da por hecha.
2. Mide, por exposición, el flujo de la estrella dentro de `norm_radius_px`
   **sobre el dato**: es el peso con el que esa exposición entra en el combinado
   y por tanto el peso de su PSF en la mezcla.
3. Devuelve las piezas para construir el documento `form="mixture"` de
   `musepipe.psf`, que es lo que C2–E5 evalúan sin enterarse de nada.

Lo que NO hace: no combina cubos (eso es C1b) ni decide la forma global (la
elección de forma del combinado sigue donde estaba, en la §3.4 de la spec C1).
"""

from __future__ import annotations

import io
import warnings
from concurrent.futures import ThreadPoolExecutor
from contextlib import redirect_stdout
from dataclasses import dataclass, field

import numpy as np
from astropy.io import fits

from ..parallel import resolve_n_jobs
from ..psf import (
    build_mixture_model_document,
    build_psf_model_document,
    companion_ring_metric,
    corner_background,
    encircled_energy,
    source_mask,
)
from ..reduction.stream_combine import _aligned_chunk
from ..reduction.telluric import wavelength_axis_from_header
from .stage_e01_psf import _encircled_energy_summary, _moffat_fit_rows, make_psf_bins
from .stage_e01_psfao import (
    PSFAO_DEFAULT_WEIGHT_CAP,
    PSFAO_DEFAULT_WEIGHTING,
    _bad_windows,
    build_psfao_model_document,
    make_bins,
)

#: Cuántas exposiciones se ajustan a la vez. Cada una sostiene su cubo alineado
#: (~1.2 GB en float32 entre DATA y STAT), así que el techo no es la CPU.
DEFAULT_MAX_WORKERS = 4

#: Peso del ajuste EN UNA EXPOSICIÓN, que no es el del combinado.
#:
#: El peso `relative` (con tope 5) se eligió midiendo sobre el cubo combinado
#: (traspaso del 2026-08-14), donde el halo está promediado sobre 29
#: exposiciones y es suave. En una exposición suelta ese mismo halo está
#: dominado por el moteado de la AO residual, y correr la atención hacia el
#: radio grande empeora el ajuste. Medido en ROXs 12 b (una exposición, 6 bins,
#: residuo del anillo del compañero):
#:
#:     λ [Å]     relative cap 5   relative cap 2   stat
#:      4800          310 %            199 %       133 %
#:      5600          139 %             89 %        72 %
#:      6700           50 %             42 %        42 %
#:      9100           42 %             34 %        32 %
#:
#: `stat` gana en toda la banda y por mucho en el azul, así que es el default
#: aquí aunque el run declare otro para el combinado. La V4 —el cociente
#: núcleo/total, que ES la corrección de apertura— sale bien con los tres
#: (−4 % a +1 %): lo que el peso decide es el halo, no el núcleo.
PEROBS_DEFAULT_WEIGHTING = "stat"
PEROBS_DEFAULT_WEIGHT_CAP = None


def perobs_weighting(cfg):
    """El peso del ajuste por exposición: knob propio, no el del combinado."""

    return str(cfg.get("psf_perobs_fit_weighting", PEROBS_DEFAULT_WEIGHTING))


def perobs_weight_cap(cfg):
    return cfg.get("psf_perobs_fit_weight_cap", PEROBS_DEFAULT_WEIGHT_CAP)


class PerObservationError(RuntimeError):
    """El ajuste por observación no se puede hacer con lo que hay."""


@dataclass
class ObservationFit:
    """Lo que sale de ajustar UNA exposición."""

    exposure_id: str
    weight: float
    form: str
    model: dict
    flux_norm: dict
    summary: dict
    psfao_rows: list = field(default_factory=list)
    moffat_rows: list = field(default_factory=list)

    def as_component(self) -> dict:
        """La componente de la mezcla: peso del combinado × flujo por λ."""

        return {
            "exposure_id": self.exposure_id,
            "weight": float(self.weight),
            "flux_norm": self.flux_norm,
            "model": self.model,
        }


def load_aligned_exposure(exposure, plan, *, dtype=np.float32):
    """El cubo de una exposición en la MISMA rejilla en la que se combinó.

    Se lee por trozos y con las mismas funciones que usa el combinado
    (`_aligned_chunk`: ventana del plan, desplazamiento subpíxel con spline para
    DATA y con el kernel bilineal al cuadrado para STAT, recorte a `crop_npix`).
    Reimplementar el alineado aquí sería tener dos convenciones con un nombre.
    """

    npix = int(plan.crop_npix)
    nz = int(plan.wavelength["n_channels"])
    chunk = max(1, int(plan.chunk_channels))
    cube = np.empty((nz, npix, npix), dtype=dtype)
    stat = np.empty((nz, npix, npix), dtype=dtype)
    with fits.open(exposure.file, memmap=True) as hdul:
        header = hdul[plan.data_ext].header
        wave = wavelength_axis_from_header(header, int(hdul[plan.data_ext].data.shape[0]))
        for z1 in range(0, nz, chunk):
            z2 = min(nz, z1 + chunk)
            data_chunk, stat_chunk, _valid = _aligned_chunk(exposure, plan, hdul, z1, z2)
            cube[z1:z2] = data_chunk.astype(dtype, copy=False)
            stat[z1:z2] = stat_chunk.astype(dtype, copy=False)
    if wave.size != nz:
        raise PerObservationError(
            f"{exposure.exposure_id}: el eje λ de la cabecera tiene {wave.size} canales y el "
            f"plan declara {nz}."
        )
    return cube, stat, wave


def _positions_in_window(npix, positions_yx, frame_shape):
    """Compañero y fuente de campo en la ventana, por desplazamiento al CENTRO.

    El compañero se toma respecto del **centro del array** del cubo de la
    cadena, que es lo que `fit_bin` usa de verdad para enmascarar y para medir el
    anillo, y se aplica igual en cada exposición. Vale porque el DRS remuestrea
    todas a la misma orientación (el plan del combinado lo verifica: exige la
    misma matriz CD en todas).
    """

    center = (npix // 2, npix // 2)
    frame_center = (int(frame_shape[0]) // 2, int(frame_shape[1]) // 2)
    moved = {}
    for name, value in positions_yx.items():
        if value is None:
            moved[name] = None
            continue
        moved[name] = (
            center[0] + float(value[0]) - frame_center[0],
            center[1] + float(value[1]) - frame_center[1],
        )
    return center, moved


def _flux_norm_table(images, waves, center_yx, norm_radius, exclude_mask):
    """`F(≤norm_radius)` de la estrella, bin a bin y MEDIDO SOBRE EL DATO.

    Es el peso con el que esta exposición entra en el combinado —una media
    pesada de brillo—, así que es también el peso de su PSF en la mezcla. Medirlo
    sobre el dato en vez de derivarlo del ajuste lo hace independiente de la
    forma que gane en esa exposición.
    """

    values = []
    for image in images:
        background = corner_background(image)
        values.append(
            float(
                encircled_energy(
                    image,
                    center_yx,
                    [float(norm_radius)],
                    background=background,
                    exclude_mask=exclude_mask,
                )[0]
            )
        )
    return {"lambda_A": [float(w) for w in waves], "value": values}


def _psfao_branch(cube, stat, wave, cfg, center, moved, mask_radius, fit_radius, system):
    """Ajuste Psfao de una exposición, con la métrica de anillo canónica."""

    from .stage_e01_psfao import fit_psfao_bins

    bins = make_bins(
        wave,
        float(cfg.get("psf_bin_A", 100.0)),
        _bad_windows(cfg),
        int(cfg.get("psf_min_channels_per_bin", 3)),
    )
    if not bins:
        raise PerObservationError("no hay bins de λ utilizables en esta exposición.")
    rows, recons = fit_psfao_bins(
        cube,
        stat,
        wave,
        bins,
        system,
        moved["companion"],
        mask_radius,
        fit_radius,
        field_yx=moved.get("field"),
        warm_start=bool(cfg.get("psf_warm_start", True)),
        weighting=perobs_weighting(cfg),
        weight_cap=perobs_weight_cap(cfg),
        # La segunda componente ligada de la primaria, ya resuelta en pixeles por
        # C1. **Sin esto el cambio de la binaria seria inerte en este run**: con
        # `psf_scope=per_observation` el modelo que se publica es la MEZCLA de
        # los ajustes por exposicion, asi que si estos no la ven, el ajuste al
        # combinado no llega a ningun consumidor. El offset es el mismo en todas
        # las exposiciones: es relativo, y los cubos son norte-arriba.
        companion_offset_yx=cfg.get("e01_binary_offset_yx_px"),
    )
    width = float(cfg.get("psf_companion_ring_width_px", 3.0))
    ring = []
    for mid in sorted(recons):
        image, recon = recons[mid]
        metric = companion_ring_metric(
            image, recon, center, moved["companion"],
            width_px=width, source_exclusion_radius_px=mask_radius,
        )
        ring.append(float(metric["median_pct"]))
        row = next(r for r in rows if float(r["lambda_A"]) == float(mid))
        row["ring_residual_pct_canonical"] = float(metric["median_pct"])
        row["ring_residual_p90_pct_canonical"] = float(metric["p90_pct"])
    return {"rows": rows, "recons": recons, "ring": np.asarray(ring, dtype=float), "bins": bins}


def _moffat_branch(cube, wave, cfg, center, moved, fwhm_prelim):
    """Ajuste Moffat de una exposición, con las posiciones ya en la ventana."""

    bins = make_psf_bins(
        wave,
        bin_A=float(cfg.get("psf_bin_A", 100.0)),
        bad_windows_A=cfg.get("stage_e01_bad_windows_A") or (),
        min_channels=int(cfg.get("psf_min_channels_per_bin", 3)),
    )
    positions_qc = {
        "primary": {"pos_yx": [float(center[0]), float(center[1])]},
        "companion": {"pos_yx": [float(v) for v in moved["companion"]]},
        "psf": {"fwhm_px": float(fwhm_prelim)},
    }
    if moved.get("field") is not None:
        positions_qc["field_source"] = {"pos_yx": [float(v) for v in moved["field"]]}
    rows, images, models, masks, meta = _moffat_fit_rows(
        cube[None, ...], wave, bins, positions_qc, cfg
    )
    ring = np.asarray([float(r["ring_residual_pct"]) for r in rows], dtype=float)
    return {
        "rows": rows,
        "images": images,
        "models": models,
        "masks": masks,
        "meta": meta,
        "bins": bins,
        "ring": ring,
    }


def fit_one_observation(exposure, plan, cfg, positions_yx, frame_shape, *, forced_form=None):
    """Ajusta las DOS formas en una exposición y entrega la que diga `forced_form`.

    Las dos siempre, como pide la §3.4 de la spec C1: el residuo del anillo de
    cada forma en cada exposición es un dato que la cadena nunca había medido, y
    viaja en el resumen.

    Pero **el anillo de una exposición suelta no decide la forma**. Se mide a la
    separación del compañero (71 px en ROXs 12 b), donde una exposición de 300 s
    apenas tiene señal: medido, 30–310 % contra el 8–20 % del combinado, y con
    esos números la comparación entre formas es una moneda al aire (en el primer
    ajuste de prueba eligió Moffat, que es la forma que el objeto tiene
    congelada como perdedora). Quien decide es quien tiene S/N para hacerlo: el
    config si congela la forma, y si no, la que ganó en el combinado. Por eso
    `forced_form` no es opcional en el camino de la etapa.
    """

    from maoppy.instrument import muse_nfm, muse_wfm

    cube, stat, wave = load_aligned_exposure(exposure, plan)
    npix = int(plan.crop_npix)
    center, moved = _positions_in_window(npix, positions_yx, frame_shape)
    system = muse_wfm if str(cfg.get("instrument_mode", "NFM")).upper().startswith("W") else muse_nfm

    fwhm_prelim = float(cfg.get("psf_prelim_fwhm_px", 4.0))
    mask_radius = float(
        cfg.get("psf_companion_mask_radius_px",
                float(cfg.get("psf_mask_radius_factor", 3.0)) * fwhm_prelim)
    )
    fit_radius = float(cfg.get("psf_fit_radius_px", 28.0))
    norm_radius = float(cfg.get("psf_norm_radius_px", 25.0))

    # `maoppy.psffit` imprime una línea por iteración: con 29 exposiciones × 43
    # bins eso son ~50 000 líneas de ruido tapando el log de la etapa.
    with warnings.catch_warnings(), redirect_stdout(io.StringIO()):
        warnings.simplefilter("ignore")
        psfao = _psfao_branch(cube, stat, wave, cfg, center, moved, mask_radius, fit_radius, system)
        moffat = _moffat_branch(cube, wave, cfg, center, moved, fwhm_prelim)
    del cube, stat

    exclude = source_mask(
        (npix, npix),
        [moved["companion"]] + ([moved["field"]] if moved.get("field") is not None else []),
        mask_radius,
    )
    psfao_median = float(np.nanmedian(psfao["ring"])) if psfao["ring"].size else float("nan")
    moffat_median = float(np.nanmedian(moffat["ring"])) if moffat["ring"].size else float("nan")

    form = str(forced_form or "").lower() or None
    if form is None:
        # Sin forma impuesta sólo queda el anillo, que en una exposición es
        # ruido: se usa, pero el resumen lo marca para que nadie lo tome por una
        # medida. La etapa siempre pasa `forced_form`.
        form = "psfao" if psfao_median <= moffat_median else "moffat"
    if form not in ("psfao", "moffat"):
        raise PerObservationError(f"forma desconocida: {form!r}")

    if form == "psfao":
        model, _meta = build_psfao_model_document(
            psfao["rows"], system, norm_radius, fit_radius,
            system_name="muse_wfm" if system is muse_wfm else "muse_nfm",
            wave_bin_A=float(cfg.get("psf_bin_A", 100.0)),
            weighting=str(cfg.get("psf_fit_weighting", PSFAO_DEFAULT_WEIGHTING)),
            weight_cap=cfg.get("psf_fit_weight_cap", PSFAO_DEFAULT_WEIGHT_CAP),
        )
        images = [psfao["recons"][mid][0] for mid in sorted(psfao["recons"])]
        models = [psfao["recons"][mid][1] for mid in sorted(psfao["recons"])]
        waves_ok = sorted(float(mid) for mid in psfao["recons"])
        backgrounds = [
            float(next(r for r in psfao["rows"] if float(r["lambda_A"]) == mid)["bck"])
            for mid in waves_ok
        ]
    else:
        model = build_psf_model_document(
            [row["wave_center_A"] for row in moffat["rows"]],
            moffat["rows"],
            form="moffat",
            norm_radius_px=norm_radius,
            hybrid=False,
        )
        images, models = moffat["images"], moffat["models"]
        waves_ok = [float(row["wave_center_A"]) for row in moffat["rows"]]
        # El fondo del AJUSTE, no el de las esquinas: `evaluate_moffat_fit`
        # devuelve el modelo con ese fondo dentro, y pasarle otro a la V4 mueve
        # el cociente nucleo/total del modelo sin que nada esté mal en el ajuste.
        backgrounds = [float(row["background"]) for row in moffat["rows"]]

    flux_norm = _flux_norm_table(images, waves_ok, center, norm_radius, exclude)
    # La V4 de la spec, exposición a exposición: la cadena sólo la medía sobre el
    # combinado, y es la única comprobación de C1 que mira el núcleo.
    v4 = _encircled_energy_summary(images, models, backgrounds, center, exclude, cfg)

    summary = {
        "exposure_id": exposure.exposure_id,
        "file": exposure.file,
        "weight": float(exposure.weight),
        "exptime": float(exposure.exptime),
        "mjd_obs": float(exposure.mjd_obs),
        "form_chosen": form,
        "form_source": "imposed" if forced_form else "ring_of_this_exposure",
        "weighting": perobs_weighting(cfg),
        "weight_cap": perobs_weight_cap(cfg),
        "ring_residual_pct_median": {"psfao": psfao_median, "moffat": moffat_median},
        "n_bins": {"psfao": int(len(psfao["rows"])), "moffat": int(len(moffat["rows"]))},
        "n_ok_bins_psfao": int(sum(1 for r in psfao["rows"] if r.get("status") == "ok")),
        "flux_norm_median": float(np.nanmedian(flux_norm["value"])) if flux_norm["value"] else None,
        "encircled_energy": v4,
        "centroid_fallback": bool(exposure.centroid_fallback),
        "in_bounds": bool(exposure.in_bounds),
    }
    return ObservationFit(
        exposure_id=exposure.exposure_id,
        weight=float(exposure.weight),
        form=form,
        model=model,
        flux_norm=flux_norm,
        summary=summary,
        psfao_rows=psfao["rows"],
        moffat_rows=moffat["rows"],
    )


def fit_all_observations(observations, cfg, positions_yx, frame_shape, *,
                         forced_form=None, n_jobs=None, progress=None):
    """Ajusta todas las exposiciones. Lo que falle se cuenta, no se esconde."""

    workers = min(
        int(resolve_n_jobs(n_jobs, max_default=DEFAULT_MAX_WORKERS)),
        max(1, len(observations.exposures)),
    )
    results: dict[str, ObservationFit] = {}
    failures: list[dict] = []

    def _one(exposure):
        try:
            return exposure, fit_one_observation(
                exposure, observations.plan, cfg, positions_yx, frame_shape,
                forced_form=forced_form,
            ), None
        except Exception as exc:  # el ajuste de una exposición no tumba las 29
            return exposure, None, f"{exc.__class__.__name__}: {exc}"

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for index, (exposure, fit, error) in enumerate(pool.map(_one, observations.exposures)):
            if progress is not None:
                progress(index, len(observations.exposures), exposure.exposure_id)
            if error is not None:
                failures.append({"exposure_id": exposure.exposure_id, "error": error})
                continue
            results[exposure.exposure_id] = fit

    if not results:
        raise PerObservationError(
            "ninguna exposición se pudo ajustar: "
            + "; ".join(f"{f['exposure_id']}: {f['error']}" for f in failures[:3])
        )
    ordered = [results[e.exposure_id] for e in observations.exposures if e.exposure_id in results]
    return ordered, failures


def mixture_from_fits(fits_by_observation, cfg, *, system_name="muse_nfm", provenance=None):
    """El documento `form="mixture"`: la PSF del combinado, sin ajustar nada a él."""

    if not fits_by_observation:
        raise PerObservationError("no hay ajustes por observación con los que mezclar.")
    norm_radius = float(cfg.get("psf_norm_radius_px", 25.0))
    document = build_mixture_model_document(
        [fit.as_component() for fit in fits_by_observation],
        norm_radius_px=norm_radius,
        system=system_name,
        wave_bin_A=float(cfg.get("psf_bin_A", 100.0)),
        psf_scope="per_observation",
        forms={fit.exposure_id: fit.form for fit in fits_by_observation},
    )
    if provenance:
        document["provenance"] = dict(provenance)
    return document


__all__ = [
    "DEFAULT_MAX_WORKERS",
    "ObservationFit",
    "PerObservationError",
    "fit_all_observations",
    "fit_one_observation",
    "load_aligned_exposure",
    "mixture_from_fits",
]
