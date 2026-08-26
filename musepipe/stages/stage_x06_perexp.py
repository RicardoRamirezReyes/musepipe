"""Etapa C7 — extraccion por exposicion y combinacion de las MEDIDAS.

La cadena combina los **cubos** y extrae una vez, lo que obliga a describir con
un solo modelo de PSF la mezcla de N exposiciones. El error de esa mezcla esta
medido: su correccion de apertura -`F(<=25)/F(box3)`, la V4 de la spec C1- falla
la tolerancia del 3 % con +6.77 %, y falla **cromaticamente** (+0.34 % en el azul,
+8.55 % en el rojo).

Esta etapa hace lo contrario: extrae en **cada exposicion con su propia PSF** y
combina las medidas. Lo medido el 2026-08-26 sobre `ROXs12b_realigned`
(`docs/2026-08-26_perexp_medido_y_la_noche_mala.md`):

* **S/N: empata** con el cubo combinado (0.97-1.09x), no lo mejora.
* **La apcorr por exposicion quita la deriva cromatica**: la razon
  `apcorr_perexp/apcorr_mezcla` deriva -8.52 % del azul al rojo, contra el
  +8.2 % que deriva la V4 de la mezcla.
* **Lo que se gana es poder pesar por calidad.** El combinado pesa por `exptime`
  y con eso le da el **38.1 %** del peso a la noche del 2022-08-31, que tiene
  sigma de controles 4.2x peor y menos de la mitad de senal; la inversa de la
  varianza le deja el **2.6 %**.

Contrato completo en `docs/spec_C7_codex_perexp_combine.md`.

**Que NO hace.** No entra en `METHOD_ORDER` (un metodo nuevo alli se vuelve
obligatorio para todos los runs al instante y arrastra D1, D2, E1, E3, E4, G1 y
F1), asi que D2 no la calibra y no entra en el bloque E: es un producto de
**referencia y validacion**. No decide que hacer con la noche mala -publica el
peso que cada ley le da y deja la decision escrita, no tomada- ni rechaza
exposiciones por umbral.
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..apertures import same_radius_control_positions
from ..config import load_run_config
from ..extraction.aperture import (
    annulus_background_spectrum,
    aperture_correction_from_psf,
    aperture_label,
    aperture_spectrum,
    channel_flags,
)
from ..extraction.product import FORMAT_VERSION, SpectrumProduct
from ..growth_curve import resolve_flux_convention
from ..io import read_json, write_json
from ..observations import resolve_observation_plan
from ..paths import RunPaths
from ..qc.cube_qc import sha256_file
from ..stats import robust_sigma_axis0
from .stage_e01_perobs import _positions_in_window, load_aligned_exposure
from .stage_e01b_perobs_subtract import models_by_exposure
from .stage_x01_aperture import _wavelength_frame

STAGE = "x06_perexp"
SPEC_VERSION = "C7"

#: Las leyes de combinacion, con lo que cada una vale MEDIDO (S/N en el continuo
#: 7000-7200 A de ROXs 12 b, box3): `invvar` 19.33, `equal` 2.26, `sum` 2.26 (la
#: misma que `equal` al decimal: difieren solo en el factor de escala N),
#: `exptime` 0.11 -- y `exptime` es justamente la del cubo combinado.
COMBINE_LAWS = ("invvar", "exptime", "equal", "sum")

#: `none` = un producto con las N; `night` = uno por noche. No es comodidad:
#: incluso pesando por calidad, anadir la noche mala de ROXs 12 b empeora las
#: tres bandas (20.60/9.24/401.36 con solo las 22 buenas contra 19.33/8.35/388.75
#: con las 29).
GROUPINGS = ("none", "night")

#: El anillo de fondo de C2, donde vive el default real (el run no suele
#: declararlo): (r_in, r_out, radio de exclusion de la primaria) en pixeles.
ANNULUS_BKG_FALLBACK_PX = (8.0, 14.0, 30.0)

#: Banda en la que se miden los pesos de `invvar`. **No puede ser una banda de
#: medida**: pesar por 1/sigma^2 con la sigma de la banda que se esta midiendo
#: infla la S/N un 40-60 % por auto-seleccion (27.94 contra 19.33 medido).
DEFAULT_WEIGHT_BAND_A = (8600.0, 9000.0)

#: Cada worker sostiene un cubo alineado (3681x200x200 en float32 ~0.6 G, mas su
#: STAT), asi que el techo lo pone la memoria y no los nucleos.
DEFAULT_MAX_WORKERS = 4


class PerExpError(RuntimeError):
    """La extraccion por exposicion no se puede hacer con lo que hay."""


@dataclass(frozen=True)
class StageX06Product:
    products: dict
    controls: dict
    qc: dict


def stage_x06_paths(run_id, project_root=None) -> dict:
    root = Path(project_root or Path.cwd()).resolve()
    paths = RunPaths.from_project_root(run_id, root)
    return {
        "paths": paths,
        # La mezcla con los modelos POR EXPOSICION. C1 la escribe siempre en su
        # propio fichero, aunque ademas sea lo que entrega como `psf_model.json`,
        # asi que esta etapa no depende de ese knob.
        "psf_model_mixture_json": paths.stage_dir / "psf_model_mixture.json",
        "observation_plan_json": paths.stage_dir / "observation_plan.json",
        "stage01c_qc_json": paths.stage_dir / "stage01c_qc.json",
        "stage00q_qc_json": paths.stage_dir / "stage00q_qc.json",
        "qc_json": paths.stage_dir / "spec_perexp_qc.json",
    }


def product_path(paths, grupo, etiqueta):
    return paths["paths"].stage_dir / f"spec_perexp_{grupo}_{etiqueta}_object.fits"


def controls_path(paths, grupo, etiqueta):
    return paths["paths"].stage_dir / f"spec_perexp_{grupo}_{etiqueta}_controls.npz"


def stage_x06_config_from_run(run_id=None, *, project_root=None, overrides=None,
                              allow_run_id_mismatch=False) -> dict:
    """Config resuelto. Las perillas compartidas se LEEN de quien manda en ellas.

    Las aperturas y el anillo salen del config **resuelto** de C2, no del crudo
    del run: la etapa rellena defaults que el run no deletrea, y copiarlos aqui
    como literales es lo que hizo que el primer notebook de C3 no reprodujera la
    cadena.
    """

    from .stage_x01_aperture import stage_x01_config_from_run

    run = load_run_config(run_id, project_root=project_root,
                          allow_run_id_mismatch=allow_run_id_mismatch)
    cfg = dict(run.config)
    if overrides:
        cfg.update(overrides)
    cfg["run_id"] = run.run_id
    cfg["project_root"] = str(run.paths.project_root)

    x01 = stage_x01_config_from_run(run.run_id, project_root=run.paths.project_root,
                                    allow_run_id_mismatch=allow_run_id_mismatch)
    cfg.setdefault("x06_apertures", x01.get("x01_apertures"))
    cfg.setdefault("x06_annulus_bkg_px", list(x01.get("x01_annulus_bkg_px",
                                                      ANNULUS_BKG_FALLBACK_PX)))
    cfg.setdefault("x06_n_controls", int(x01.get("x01_n_controls", 8)))
    cfg.setdefault("x06_norm_radius_px", float(cfg.get("psf_norm_radius_px", 25.0)))
    cfg.setdefault("x06_group_by", "none")
    cfg.setdefault("x06_combine", "invvar")
    cfg.setdefault("x06_weight_band_A", list(DEFAULT_WEIGHT_BAND_A))
    cfg.setdefault("x06_max_workers", DEFAULT_MAX_WORKERS)
    cfg.setdefault("x06_flux_convention", cfg.get("flux_convention", "normrad"))
    if not cfg.get("x06_apertures"):
        raise PerExpError(
            "el run no declara `x01_apertures` y no se ha pasado `x06_apertures`: "
            "sin apertura no hay espectro que extraer."
        )
    if str(cfg["x06_group_by"]) not in GROUPINGS:
        raise PerExpError(f"`x06_group_by` solo entiende {GROUPINGS}, no {cfg['x06_group_by']!r}.")
    if str(cfg["x06_combine"]) not in COMBINE_LAWS:
        raise PerExpError(f"`x06_combine` solo entiende {COMBINE_LAWS}, no {cfg['x06_combine']!r}.")
    return cfg


def group_exposures(exposures, mode):
    """`{grupo: [indices]}`. `night` agrupa por la fecha del `exposure_id`.

    La fecha va en el `exposure_id` (`2022-08-29_MUSE...`), que es como el plan
    nombra las carpetas; la MJD se publica ademas en el QC para poder auditarlo.
    """

    if mode == "none":
        return {"all": list(range(len(exposures)))}
    grupos: dict[str, list[int]] = {}
    for i, e in enumerate(exposures):
        noche = str(e.exposure_id).split("_")[0].replace("-", "")
        grupos.setdefault(noche, []).append(i)
    return dict(sorted(grupos.items()))


def combine_measurements(valores, pesos_plan, sigma_peso, ley):
    """Pesos normalizados y escala, para una ley.

    Devuelve `(pesos, escala)`: el combinado es `escala * sum(w_i * x_i)`. Para
    todas las leyes salvo `sum` la escala es 1 y los pesos suman 1; `sum` es
    `equal` multiplicada por N, y **da exactamente la misma S/N** (medido: 2.26
    contra 2.26 en el continuo), porque su sigma escala igual.
    """

    n = len(valores)
    if ley == "invvar":
        w = 1.0 / np.maximum(np.asarray(sigma_peso, dtype=np.float64), 1e-30) ** 2
    elif ley == "exptime":
        w = np.asarray(pesos_plan, dtype=np.float64).copy()
    else:
        w = np.ones(n, dtype=np.float64)
    total = float(np.nansum(w))
    if not np.isfinite(total) or total <= 0:
        raise PerExpError(f"los pesos de la ley {ley!r} no son utilizables.")
    w = w / total
    return w, (float(n) if ley == "sum" else 1.0)


def _positions(paths, plan):
    qc = read_json(paths["stage01c_qc_json"])
    positions_yx = {"primary": tuple(map(float, qc["primary"]["pos_yx"])),
                    "companion": tuple(map(float, qc["companion"]["pos_yx"]))}
    if qc.get("field_source") is not None:
        positions_yx["field"] = tuple(map(float, qc["field_source"]["pos_yx"]))
    frame_shape = tuple(qc.get("frame_shape") or qc.get("cube_shape_yx") or (170, 170))
    star_yx, moved = _positions_in_window(int(plan.crop_npix), positions_yx, frame_shape)
    comp_yx = moved.get("companion")
    if comp_yx is None:
        raise PerExpError("el compañero no cae en la ventana del combinado.")
    return star_yx, comp_yx, frame_shape


def extract_one_exposure(exposure, plan, cfg, model_doc, star_yx, comp_yx, controles,
                         growth_curve):
    """El compañero y sus controles en UNA exposicion, con SU modelo de PSF.

    Los controles se procesan **identicos al objeto** -mismo fondo de anillo,
    misma apertura y la misma `apcorr`-, que es el invariante de
    `docs/noise_model.md`: sin eso su dispersion no es la sigma del objeto.
    """

    cube, _stat, wave = load_aligned_exposure(exposure, plan)
    r_in, r_out, excl = (list(cfg["x06_annulus_bkg_px"]) + [30.0])[:3]
    fila = {"exposure_id": exposure.exposure_id, "weight": float(exposure.weight),
            "exptime": float(getattr(exposure, "exptime", np.nan) or np.nan),
            "mjd_obs": float(getattr(exposure, "mjd_obs", np.nan) or np.nan),
            "wave_A": wave, "apertures": {}}

    def _una(centro, apc):
        bkg = annulus_background_spectrum(cube, centro, float(r_in), float(r_out),
                                          exclude_yx=star_yx, exclude_radius=float(excl))
        flujo, npix_eff = aperture_spectrum(cube, centro, apc)
        return np.asarray(flujo - bkg * npix_eff, dtype=np.float64), np.asarray(npix_eff)

    for apertura in cfg["x06_apertures"]:
        etiqueta = aperture_label(apertura)
        apcorr, apcorr_mode, norm_radius = aperture_correction_from_psf(
            wave, apertura, model_doc, center_yx=comp_yx, growth_curve=growth_curve,
        )
        crudo, npix_eff = _una(comp_yx, apertura)
        ctrl = np.asarray([_una(c, apertura)[0] for c in controles], dtype=np.float64)
        fila["apertures"][etiqueta] = {
            "flux_raw": crudo,
            "flux": crudo * apcorr,
            "apcorr": np.asarray(apcorr, dtype=np.float64),
            "apcorr_mode": apcorr_mode,
            "norm_radius_px": float(norm_radius),
            "npix_eff": np.asarray(npix_eff, dtype=np.float64),
            # Identicos al objeto, con SU apcorr: es lo que los hace utilizables
            # como sigma (`extraction/aperture.py:608` hace lo mismo en C2).
            "controls": ctrl * apcorr[None, :],
        }
    del cube
    return fila


def _banda(wave, banda):
    lo, hi = (float(banda[0]), float(banda[1]))
    return (wave >= lo) & (wave <= hi)


def compute_stage_x06_products(config, paths=None) -> StageX06Product:
    cfg = dict(config)
    paths = paths or stage_x06_paths(cfg["run_id"], project_root=cfg.get("project_root"))
    open_issues: list[str] = []

    doc = read_json(paths["psf_model_mixture_json"])
    if str(doc.get("form", "")).lower() != "mixture":
        raise PerExpError(
            f"{paths['psf_model_mixture_json'].name} tiene form={doc.get('form')!r} y no una "
            "mezcla por observacion: extraer del combinado ya lo hacen C2-C6, y hacerlo aqui en "
            "silencio seria entregar dos veces la misma cosa con nombres distintos."
        )
    modelos = models_by_exposure(doc)

    cache = paths["observation_plan_json"]
    obs = resolve_observation_plan(cfg["run_id"], project_root=cfg.get("project_root"),
                                   plan_json=cache if cache.exists() else None)
    plan = obs.plan
    ids = [e.exposure_id for e in obs.exposures]
    repetidos = sorted({x for x in ids if ids.count(x) > 1})
    if repetidos:
        # `exposure_id` sale del directorio padre y NO es unico por construccion
        # (`reduction/stream_combine.py:110-117`). Indexar los modelos por el sin
        # comprobarlo mezclaria dos exposiciones en silencio.
        raise PerExpError(f"`exposure_id` repetido en el plan: {repetidos}.")
    faltan = [x for x in ids if x not in modelos]
    if faltan:
        raise PerExpError(
            f"{len(faltan)} exposiciones del plan no tienen modelo en la mezcla: {faltan[:3]}..."
        )

    star_yx, comp_yx, frame_shape = _positions(paths, plan)
    npix = int(plan.crop_npix)
    controles = same_radius_control_positions(
        comp_yx, star_yx, npix, npix, n_positions=int(cfg["x06_n_controls"]),
    )
    if len(controles) < 2:
        raise PerExpError(
            f"solo {len(controles)} posiciones de control caben a esa separacion: sin dos no hay "
            "sigma empirica, y `STAT` no vale como sigma (docs/noise_model.md)."
        )
    growth_curve, flux_convention = resolve_flux_convention(
        cfg, paths["paths"].stage_dir, knob="x06_flux_convention")

    n_jobs = max(1, int(cfg["x06_max_workers"]))
    with ThreadPoolExecutor(max_workers=n_jobs) as pool:
        filas = list(pool.map(
            lambda e: extract_one_exposure(e, plan, cfg, modelos[e.exposure_id],
                                           star_yx, comp_yx, controles, growth_curve),
            obs.exposures,
        ))

    wave = np.asarray(filas[0]["wave_A"], dtype=np.float64)
    qc00 = read_json(paths["stage00q_qc_json"]) if paths["stage00q_qc_json"].exists() else {}
    wframe = _wavelength_frame(cfg, qc00, open_issues, knob="x06_wframe")
    banda_peso = cfg["x06_weight_band_A"]
    ley = str(cfg["x06_combine"])
    grupos = group_exposures(obs.exposures, str(cfg["x06_group_by"]))
    plan_sha = sha256_file(obs.source) if Path(obs.source).exists() else ""
    flags = channel_flags(wave)
    if not np.any(_banda(wave, banda_peso)):
        # Sin esto el sintoma era `los pesos no son utilizables`, que no dice
        # nada: la banda de pesos fuera del rango del cubo es un error de
        # configuracion y tiene que decirse por su nombre.
        raise PerExpError(
            f"`x06_weight_band_A` = {list(banda_peso)} no selecciona ningun canal: el cubo va de "
            f"{wave[0]:.1f} a {wave[-1]:.1f} A. Los pesos de `invvar` se miden en una banda "
            "declarada y esa banda tiene que existir."
        )

    productos, controles_out, qc_grupos = {}, {}, {}
    sel_peso = _banda(wave, banda_peso)
    for grupo, idx in grupos.items():
        pesos_plan = np.asarray([filas[i]["weight"] for i in idx], dtype=np.float64)
        qc_apert = {}
        for apertura in cfg["x06_apertures"]:
            etiqueta = aperture_label(apertura)
            objetos = np.asarray([filas[i]["apertures"][etiqueta]["flux"] for i in idx])
            crudos = np.asarray([filas[i]["apertures"][etiqueta]["flux_raw"] for i in idx])
            apcorrs = np.asarray([filas[i]["apertures"][etiqueta]["apcorr"] for i in idx])
            ctrls = np.asarray([filas[i]["apertures"][etiqueta]["controls"] for i in idx])
            npix_eff = np.asarray(filas[idx[0]]["apertures"][etiqueta]["npix_eff"], dtype=np.float64)

            # Los pesos de `invvar` se miden en una banda DECLARADA, nunca en la
            # que se esta midiendo: eso ultimo infla la S/N por auto-seleccion.
            sigma_peso = np.asarray(
                [np.nanstd(np.nanmedian(c[:, sel_peso], axis=1)) for c in ctrls])
            w, escala = combine_measurements(objetos, pesos_plan, sigma_peso, ley)

            flujo = escala * np.nansum(w[:, None] * objetos, axis=0)
            crudo = escala * np.nansum(w[:, None] * crudos, axis=0)
            apcorr = np.nansum(w[:, None] * apcorrs, axis=0)
            # La sigma sale de los controles COMBINADOS igual que el objeto. No
            # de `STAT` (subestima ~4x) ni de sigma_i/sqrt(n): esta ultima trata
            # como ruido la estructura del halo, que es la misma en las N y NO
            # promedia (docs/2026-08-21_heterogeneidad_y_limite.md).
            ctrl_comb = escala * np.nansum(w[:, None, None] * ctrls, axis=0)
            sigma = robust_sigma_axis0(ctrl_comb)

            header = {
                "FORMATV": FORMAT_VERSION,
                "METHOD": "aperture",
                "RUNID": cfg["run_id"],
                "SRCPOS_Y": float(comp_yx[0]),
                "SRCPOS_X": float(comp_yx[1]),
                "APERTURE": etiqueta,
                "WFRAME": wframe,
                # La entrada NO es un cubo: son N. Se declara el plan que las
                # nombra, con su sha, y las exposiciones van en `EXPIDS`.
                "INCUBE": str(obs.source),
                "INCUBESH": plan_sha,
                "NORMRAD": float(filas[idx[0]]["apertures"][etiqueta]["norm_radius_px"]),
                "ERRMODE": "empirical",
                "APCMODE": filas[idx[0]]["apertures"][etiqueta]["apcorr_mode"],
                "SCALEREF": ("empirical_total_flux" if flux_convention == "total" else "normrad"),
                "COMBMODE": f"perexp_{ley}",
                "GROUP": grupo,
                "NEXP": int(len(idx)),
                "WBAND": f"{float(banda_peso[0]):.1f}-{float(banda_peso[1]):.1f}",
                "POSFRAME": "plan_window",
                "NCTRL": int(len(controles)),
                "EXPIDS": ",".join(filas[i]["exposure_id"] for i in idx)[:60],
            }
            producto = SpectrumProduct(
                wave_A=wave,
                flux=flujo,
                flux_err=sigma,
                flux_err_emp=sigma,
                apcorr=apcorr,
                npix_eff=npix_eff,
                flags=flags,
                header=header,
                extra_columns={"flux_raw": crudo},
            )
            producto.validate()
            productos[(grupo, etiqueta)] = producto
            controles_out[(grupo, etiqueta)] = {
                "control_spectra": ctrl_comb,
                "controls_yx": np.asarray(controles, dtype=np.float64),
                "weights": w,
                "combine": ley,
            }
            qc_apert[etiqueta] = {
                "n_exposures": int(len(idx)),
                "weights_normalised": {filas[i]["exposure_id"]: float(x) for i, x in zip(idx, w)},
                "n_eff": float(1.0 / np.nansum(w ** 2)),
                "apcorr_median": float(np.nanmedian(apcorr)),
                "sigma_median": float(np.nanmedian(sigma)),
                "flux_median": float(np.nanmedian(flujo)),
            }
        qc_grupos[grupo] = {
            "exposures": [filas[i]["exposure_id"] for i in idx],
            "mjd_range": [float(np.nanmin([filas[i]["mjd_obs"] for i in idx])),
                          float(np.nanmax([filas[i]["mjd_obs"] for i in idx]))],
            "exptime_total_s": float(np.nansum([filas[i]["exptime"] for i in idx])),
            "apertures": qc_apert,
        }

    # El reparto del peso por grupo con CADA ley, para que la decision sobre una
    # noche mala se pueda tomar mirando el QC en vez de re-midiendo. No se toma
    # aqui: se publica.
    reparto = {}
    if str(cfg["x06_group_by"]) == "none" and len(group_exposures(obs.exposures, "night")) > 1:
        etiqueta0 = aperture_label(cfg["x06_apertures"][0])
        ctrls0 = np.asarray([f["apertures"][etiqueta0]["controls"] for f in filas])
        sigma0 = np.asarray([np.nanstd(np.nanmedian(c[:, sel_peso], axis=1)) for c in ctrls0])
        pesos0 = np.asarray([f["weight"] for f in filas], dtype=np.float64)
        for otra in COMBINE_LAWS:
            w, _ = combine_measurements(filas, pesos0, sigma0, otra)
            reparto[otra] = {noche: float(np.nansum(w[idx]))
                             for noche, idx in group_exposures(obs.exposures, "night").items()}
        open_issues.append(
            "Las exposiciones se combinan sin agrupar y el run tiene mas de una noche: "
            "`weight_share_by_night` dice cuanto peso se lleva cada una con cada ley."
        )

    qc = {
        "stage": STAGE,
        "spec_version": SPEC_VERSION,
        "run_id": cfg["run_id"],
        "input": {
            "psf_model_mixture": str(paths["psf_model_mixture_json"]),
            "psf_model_mixture_sha256": sha256_file(paths["psf_model_mixture_json"]),
            "observation_plan": str(obs.source),
            "observation_plan_sha256": plan_sha,
            "positions_from": str(paths["stage01c_qc_json"]),
            "n_exposures": int(len(obs.exposures)),
        },
        "convention": {
            "frame": "ventana del combinado (crop_npix del plan)",
            "positions_yx": {"primary": [float(x) for x in star_yx],
                             "companion": [float(x) for x in comp_yx]},
            "apertures": [dict(a) for a in cfg["x06_apertures"]],
            "background": "annulus",
            "annulus_bkg_px": [float(x) for x in cfg["x06_annulus_bkg_px"]],
            "aperture_correction": "por exposicion, del modelo de PSF de esa exposicion",
            "flux_convention": flux_convention,
            "group_by": str(cfg["x06_group_by"]),
            "combine": ley,
            "weight_band_A": [float(x) for x in banda_peso],
            "n_controls": int(len(controles)),
            "sigma": "dispersion de los controles combinados con los mismos pesos",
        },
        "groups": qc_grupos,
        "weight_share_by_night": reparto,
        "wavelength_frame": wframe,
        "open_issues": open_issues,
    }
    return StageX06Product(products=productos, controls=controles_out, qc=qc)


def write_stage_x06_products(product: StageX06Product, config, paths) -> dict:
    paths["paths"].ensure_base_dirs()
    escritos = {}
    for (grupo, etiqueta), spec in product.products.items():
        destino = product_path(paths, grupo, etiqueta)
        spec.write(destino)
        # Round-trip: si el producto no se puede releer, no se ha entregado.
        SpectrumProduct.read(destino)
        escritos[f"{grupo}_{etiqueta}"] = str(destino)
        ctrl = product.controls[(grupo, etiqueta)]
        np.savez(controls_path(paths, grupo, etiqueta), **ctrl)
        escritos[f"{grupo}_{etiqueta}_controls"] = str(controls_path(paths, grupo, etiqueta))
    qc = dict(product.qc)
    qc["products"] = escritos
    write_json(paths["qc_json"], qc)
    return {"products": escritos, "qc_json": str(paths["qc_json"]), "qc": qc}


def run_stage_x06(run_id=None, *, project_root=None, overrides=None,
                  allow_run_id_mismatch=False) -> dict:
    cfg = stage_x06_config_from_run(run_id, project_root=project_root, overrides=overrides,
                                    allow_run_id_mismatch=allow_run_id_mismatch)
    paths = stage_x06_paths(cfg["run_id"], project_root=cfg["project_root"])
    product = compute_stage_x06_products(cfg, paths)
    written = write_stage_x06_products(product, cfg, paths)
    return {"config": cfg, "paths": paths, "qc": product.qc, "written": written}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--project-root", default=None)
    parser.add_argument("--allow-run-id-mismatch", action="store_true")
    parser.add_argument("--group-by", choices=list(GROUPINGS), default=None,
                        help="`night` entrega un producto por noche.")
    parser.add_argument("--combine", choices=list(COMBINE_LAWS), default=None,
                        help="ley de combinacion de las medidas.")
    parser.add_argument("--jobs", type=int, default=None)
    args = parser.parse_args(argv)
    overrides = {}
    if args.group_by:
        overrides["x06_group_by"] = args.group_by
    if args.combine:
        overrides["x06_combine"] = args.combine
    if args.jobs:
        overrides["x06_max_workers"] = int(args.jobs)
    out = run_stage_x06(run_id=args.run_id, project_root=args.project_root,
                        overrides=overrides or None,
                        allow_run_id_mismatch=args.allow_run_id_mismatch)
    print(out["written"]["qc_json"])
    return out


__all__ = [
    "COMBINE_LAWS",
    "GROUPINGS",
    "PerExpError",
    "StageX06Product",
    "combine_measurements",
    "compute_stage_x06_products",
    "extract_one_exposure",
    "group_exposures",
    "main",
    "run_stage_x06",
    "stage_x06_config_from_run",
    "stage_x06_paths",
    "write_stage_x06_products",
]


if __name__ == "__main__":
    main()
