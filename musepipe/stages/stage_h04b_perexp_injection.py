"""Etapa E4b — inyeccion-recuperacion POR EXPOSICION.

E4 no tiene modo por exposicion: no hay una sola referencia a exposiciones en
toda la etapa. Por eso **todo** test de inyeccion que existia hasta hoy -los
falsos positivos de `psffit`, el barrido de radios, la Figura 2- esta hecho
sobre el cubo combinado, y eso es una limitacion estructural, no una eleccion.

Y el sustrato importa. Medido el 2026-08-27
(`docs/2026-08-27_modelo_psf_por_exposicion.md`): con SU propio cubo y SU propio
modelo, las exposiciones de la noche buena de ROXs 12 b subestiman el halo a la
separacion del companero en +55 a +107 %, contra el +17 % del combinado. El
sustrato cambia el error del modelo por un factor de 3 a 6.

Contrato completo en `docs/spec_E4b_codex_perexp_injection.md`.

**Por que es una etapa aparte y no un knob de E4.** E3 consume la tabla de
throughput de E4, asi que un interruptor de sustrato dentro de E4 cambiaria en
silencio el limite de Mdot publicado -y el fallback silencioso es justo la
familia de bugs recurrente de este repo-. Con etapa aparte: QC propio, E3 no la
lee, la cadena no se mueve. Es producto de referencia y validacion, igual que C7.
"""
from __future__ import annotations

import argparse
import csv
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
)
from ..extraction.psffit import fit_psffit_cube
from ..growth_curve import resolve_flux_convention
from ..injection import InjectionSource, inject
from ..io import read_json, write_json
from ..observations import resolve_observation_plan
from ..paths import RunPaths
from ..qc.cube_qc import sha256_file
from ..stats import robust_sigma, robust_sigma_axis0
from .stage_e01_perobs import load_aligned_exposure
from .stage_e01b_perobs_subtract import models_by_exposure
from .stage_h04_injection import (
    build_h04_cases,
    measure_recovery_with_h01_estimator,
    resolve_h04_positions,
    template_width_label,
)
from .stage_x01_aperture import _wavelength_frame
from .stage_x06_perexp import _positions, combine_measurements, group_exposures

STAGE = "h04b_perexp_injection"
SPEC_VERSION = "E4b"
#: los dos extractores que se saben hacer por exposicion. `aperture` es el
#: control limpio y `psffit` el que tiene el pedestal medido en el combinado:
#: son justo los dos que hacen falta para poder preguntar si el pedestal es del
#: sustrato o del estimador.
METODOS = ("aperture", "psffit")


class PerExpInjectionError(RuntimeError):
    pass


@dataclass
class StageH04bProduct:
    filas: list[dict]
    combinado: list[dict]
    qc: dict


def stage_h04b_paths(run_id, project_root=None) -> dict:
    root = Path(project_root or Path.cwd()).resolve()
    paths = RunPaths.from_project_root(run_id, root)
    return {
        "paths": paths,
        "psf_model_mixture_json": paths.stage_dir / "psf_model_mixture.json",
        "observation_plan_json": paths.stage_dir / "observation_plan.json",
        "stage01c_qc_json": paths.stage_dir / "stage01c_qc.json",
        "stage00q_qc_json": paths.stage_dir / "stage00q_qc.json",
        "por_exposicion_csv": paths.table_dir / "perexp_injection_by_exposure.csv",
        "combinado_csv": paths.table_dir / "perexp_injection_combined.csv",
        "qc_json": paths.stage_dir / "stage_h04b_qc.json",
    }


def stage_h04b_config_from_run(run_id=None, *, project_root=None, overrides=None,
                               allow_run_id_mismatch=False) -> dict:
    """Config resuelto.

    Las aperturas y el anillo salen del config **resuelto** de C2, no del crudo
    del run: la etapa rellena defaults que el run no deletrea, y copiarlos aqui
    como literales es lo que hizo que el primer notebook de C3 no reprodujera la
    cadena (mismo motivo, mismo remedio que en C7).
    """

    from .stage_x01_aperture import stage_x01_config_from_run

    run = load_run_config(run_id, project_root=project_root,
                          allow_run_id_mismatch=allow_run_id_mismatch)
    cfg = dict(run.config)
    cfg.update(overrides or {})
    cfg["run_id"] = run.run_id
    cfg["project_root"] = run.paths.project_root
    x01 = stage_x01_config_from_run(run.run_id, project_root=run.paths.project_root,
                                    allow_run_id_mismatch=allow_run_id_mismatch)
    cfg.setdefault("x06b_apertures", x01.get("x01_apertures"))
    cfg.setdefault("x06b_annulus_bkg_px", list(x01.get("x01_annulus_bkg_px", [8.0, 14.0, 30.0])))
    cfg.setdefault("x06b_n_controls", int(x01.get("x01_n_controls", 8)))
    cfg.setdefault("x06b_methods", list(METODOS))
    cfg.setdefault("x06b_combine", "invvar")
    cfg.setdefault("x06b_group_by", "none")
    cfg.setdefault("x06b_weight_band_A", [8600.0, 9000.0])
    cfg.setdefault("x06b_max_workers", 4)
    cfg.setdefault("x06b_flux_convention", cfg.get("flux_convention", "normrad"))
    cfg.setdefault("x06b_psffit_star_radius_px", 20.0)
    cfg.setdefault("x06b_psffit_comp_radius_px", 12.0)
    cfg.setdefault("x06b_continuum_window_A", 80.0)
    # Ventana espectral de la medida. El filtro adaptado solo mira la linea y su
    # ventana de continuo, asi que leer los 3681 canales es tirar tiempo: medido,
    # `psffit` cuesta 24 s por ajuste sobre el cubo entero y 1.6 s sobre la banda,
    # o sea 37 h contra 2.5 h para una rejilla de 192 inyecciones x 29 exposiciones.
    # `None` lee el cubo entero, y esta declarado por si alguna vez hace falta.
    if "x06b_band_A" not in cfg:
        _linea = float(cfg.get("h04_line_center_A", cfg.get("h01_line_center_A", 6562.8)))
        _media = 1.5 * float(cfg["x06b_continuum_window_A"]) + 50.0
        cfg["x06b_band_A"] = [_linea - _media, _linea + _media]
    if not cfg.get("x06b_apertures"):
        raise PerExpInjectionError(
            "el run no declara `x01_apertures` y no se ha pasado `x06b_apertures`.")
    metodos = [str(m) for m in cfg["x06b_methods"]]
    malos = [m for m in metodos if m not in METODOS]
    if malos:
        raise PerExpInjectionError(f"`x06b_methods` solo entiende {METODOS}, no {malos}.")
    return cfg


def _extrae(metodo, cube, stat, wave, centro, star_yx, apertura, model_doc, growth_curve, cfg):
    """Un espectro y su error en UNA posicion de UNA exposicion, con SU modelo.

    Los controles se miden con esto mismo, que es el invariante de
    `docs/noise_model.md`: procesados identicos al objeto, o su dispersion no es
    la sigma del objeto.
    """
    apcorr, _modo, _norm = aperture_correction_from_psf(
        wave, apertura, model_doc, center_yx=centro, growth_curve=growth_curve)
    if metodo == "aperture":
        r_in, r_out, excl = (list(cfg["x06b_annulus_bkg_px"]) + [30.0])[:3]
        bkg = annulus_background_spectrum(cube, centro, float(r_in), float(r_out),
                                          exclude_yx=star_yx, exclude_radius=float(excl))
        crudo, npix_eff = aperture_spectrum(cube, centro, apertura)
        flujo = np.asarray(crudo - bkg * npix_eff, dtype=np.float64) * apcorr
        # El error por canal lo pone quien llama, con la dispersion entre
        # controles: `STAT` no vale como sigma (docs/noise_model.md).
        return flujo, None
    if metodo == "psffit":
        fit = fit_psffit_cube(
            cube, stat, wave, star_yx, centro, model_doc,
            star_radius_px=float(cfg["x06b_psffit_star_radius_px"]),
            comp_radius_px=float(cfg["x06b_psffit_comp_radius_px"]),
        )
        flujo = np.asarray(fit.coeffs[:, 1], dtype=np.float64) * apcorr
        err = np.sqrt(np.clip(np.asarray(fit.covariance[:, 1, 1], dtype=np.float64), 0.0, np.inf)) * apcorr
        return flujo, err
    raise PerExpInjectionError(f"metodo desconocido: {metodo!r}")


def _sigma_de_la_exposicion(controles_flux, wave, line_center, line_fwhm, err_ch, ventana):
    """La sigma con la que se convierte S/N en flujo, medida EN esa exposicion.

    Escalar `input_snr` con la sigma del combinado seria mentir: inyectar
    "S/N=1" con la sigma del combinado en una exposicion de 300 s no es S/N=1.
    Es la V3 de la spec.
    """
    medidas = []
    for ctrl in controles_flux:
        m = measure_recovery_with_h01_estimator(
            (wave, ctrl, err_ch), line_center_A=line_center,
            line_fwhm_A=line_fwhm, continuum_window_A=ventana)
        if np.isfinite(m["recovered_flux"]):
            medidas.append(float(m["recovered_flux"]))
    if len(medidas) < 2:
        raise PerExpInjectionError(
            "hacen falta al menos dos controles medibles para la sigma empirica de la exposicion.")
    return float(robust_sigma(np.asarray(medidas, dtype=np.float64))), medidas


def _una_exposicion(exposure, plan, cfg, model_doc, star_yx, controles, casos,
                    growth_curve, line_center, lsf_fwhm, apertura, etiqueta_ap):
    """Todas las inyecciones de UNA exposicion, en su propio cubo."""
    cube, stat, wave = load_aligned_exposure(exposure, plan)
    banda = cfg.get("x06b_band_A")
    if banda:
        _sel = np.where((wave >= float(banda[0])) & (wave <= float(banda[1])))[0]
        if _sel.size < 20:
            raise PerExpInjectionError(
                f"`x06b_band_A`={list(banda)} selecciona {_sel.size} canales: sin banda "
                "no hay continuo que restar ni filtro que aplicar.")
        cube = cube[_sel[0]:_sel[-1] + 1]
        stat = stat[_sel[0]:_sel[-1] + 1]
        wave = wave[_sel[0]:_sel[-1] + 1]
    ventana = float(cfg["x06b_continuum_window_A"])
    metodos = [str(m) for m in cfg["x06b_methods"]]
    filas = []

    # 1) Los controles SIN inyectar: dan el error por canal y la sigma de la
    #    exposicion. Se miden una vez, porque no dependen de la inyeccion.
    ctrl_flux = np.asarray(
        [_extrae("aperture", cube, stat, wave, c, star_yx, apertura, model_doc, growth_curve, cfg)[0]
         for c in controles], dtype=np.float64)
    # `robust_sigma_axis0` y no `robust_sigma`: la segunda devuelve un escalar y
    # aqui hace falta la dispersion CANAL A CANAL entre controles.
    err_ch = np.asarray(robust_sigma_axis0(ctrl_flux), dtype=np.float64)
    err_ch = np.where(np.isfinite(err_ch) & (err_ch > 0), err_ch, np.nan)
    sigma_exp, _ = _sigma_de_la_exposicion(ctrl_flux, wave, line_center, lsf_fwhm, err_ch, ventana)

    for caso in casos:
        centro = (float(caso.position_y), float(caso.position_x))
        # V2: senal nula es senal nula, EXACTAMENTE 0.0, o el punto S/N=0 deja
        # de medir falsos positivos.
        flujo_inyectado = float(caso.input_snr) * sigma_exp
        fuente = InjectionSource(
            y=centro[0], x=centro[1],
            total_line_flux=flujo_inyectado,
            line_center_A=line_center,
            line_fwhm_A=lsf_fwhm * float(caso.template_factor),
            label=caso.injection_id,
            continuum_flux_density=0.0,
            psf_fwhm_scale=float(caso.psf_fwhm_scale),
        )
        cube_inj = inject(cube, [fuente], wavelengths_A=wave, psf_model=model_doc, copy=True)
        for metodo in metodos:
            flujo, err = _extrae(metodo, cube_inj, stat, wave, centro, star_yx,
                                 apertura, model_doc, growth_curve, cfg)
            usa_err = err if err is not None else err_ch
            m = measure_recovery_with_h01_estimator(
                (wave, flujo, usa_err), line_center_A=line_center,
                line_fwhm_A=lsf_fwhm * float(caso.template_factor),
                continuum_window_A=ventana)
            filas.append({
                "exposure_id": exposure.exposure_id,
                "noche": str(exposure.exposure_id)[:10],
                "weight": float(exposure.weight),
                "sigma_exposicion": sigma_exp,
                "injection_id": caso.injection_id,
                "variant": caso.variant,
                "method": metodo,
                "aperture": etiqueta_ap,
                "position_label": caso.position_label,
                "position_y": centro[0],
                "position_x": centro[1],
                "template_width": template_width_label(caso.template_factor),
                "input_snr": float(caso.input_snr),
                "injected_flux": flujo_inyectado,
                "recovered_flux": float(m["recovered_flux"]),
                "recovered_sigma": float(m["recovered_sigma"]),
                "recovered_snr": float(m["recovered_snr"]),
            })
    del cube
    return filas


def compute_stage_h04b_products(config, paths=None) -> StageH04bProduct:
    cfg = dict(config)
    paths = paths or stage_h04b_paths(cfg["run_id"], project_root=cfg.get("project_root"))
    open_issues: list[str] = []

    doc = read_json(paths["psf_model_mixture_json"])
    if str(doc.get("form", "")).lower() != "mixture":
        raise PerExpInjectionError(
            f"{paths['psf_model_mixture_json'].name} tiene form={doc.get('form')!r} y no una "
            "mezcla por observacion: inyectar en el combinado ya lo hace E4.")
    modelos = models_by_exposure(doc)

    cache = paths["observation_plan_json"]
    obs = resolve_observation_plan(cfg["run_id"], project_root=cfg.get("project_root"),
                                   plan_json=cache if cache.exists() else None)
    plan = obs.plan
    faltan = [e.exposure_id for e in obs.exposures if e.exposure_id not in modelos]
    if faltan:
        raise PerExpInjectionError(
            f"{len(faltan)} exposiciones del plan no tienen modelo en la mezcla: {faltan[:3]}...")

    star_yx, comp_yx, _shape = _positions(paths, plan)
    npix = int(plan.crop_npix)
    controles = same_radius_control_positions(
        comp_yx, star_yx, npix, npix, n_positions=int(cfg["x06b_n_controls"]))
    if len(controles) < 2:
        raise PerExpInjectionError("sin dos controles no hay sigma empirica.")

    # La rejilla de casos es LA DE E4, para que la comparacion sea la misma
    # medida sobre otro sustrato y no otra medida.
    posiciones = resolve_h04_positions(cfg, paths=None) if cfg.get("h04_control_positions_yx") else [
        {"label": "real", "y": comp_yx[0], "x": comp_yx[1]},
        *({"label": f"control{i+1}", "y": c[0], "x": c[1]} for i, c in enumerate(controles)),
    ]
    casos = [c for c in build_h04_cases(cfg, posiciones) if c.variant == "nominal"]

    growth_curve, flux_convention = resolve_flux_convention(
        cfg, paths["paths"].stage_dir, knob="x06b_flux_convention")
    apertura = cfg["x06b_apertures"][0]
    etiqueta_ap = aperture_label(apertura)
    line_center = float(cfg.get("h04_line_center_A", cfg.get("h01_line_center_A", 6562.8)))
    lsf_fwhm = float(cfg.get("h01_lsf_fwhm_A", 2.383))

    # La agrupacion se resuelve ANTES del trabajo caro. `group_exposures` devuelve
    # {grupo: [INDICES]}, no objetos, y consumirla mal costo 34 min de calculo
    # tirado porque el fallo estaba DESPUES del bucle. Aqui cuesta segundos.
    grupos = {nombre: [obs.exposures[i] for i in idx]
              for nombre, idx in group_exposures(obs.exposures,
                                                 str(cfg["x06b_group_by"])).items()}
    vistos = {e.exposure_id for v in grupos.values() for e in v}
    if vistos != {e.exposure_id for e in obs.exposures}:
        raise PerExpInjectionError(
            "la agrupacion no cubre exactamente las exposiciones del plan.")

    n_jobs = max(1, int(cfg["x06b_max_workers"]))
    with ThreadPoolExecutor(max_workers=n_jobs) as pool:
        trozos = list(pool.map(
            lambda e: _una_exposicion(e, plan, cfg, modelos[e.exposure_id], star_yx,
                                      controles, casos, growth_curve, line_center,
                                      lsf_fwhm, apertura, etiqueta_ap),
            obs.exposures,
        ))
    filas = [f for t in trozos for f in t]

    ley = str(cfg["x06b_combine"])
    combinado = []
    for nombre, exps in grupos.items():
        ids = {e.exposure_id for e in exps}
        pesos_plan = {e.exposure_id: float(e.weight) for e in exps}
        clave = lambda f: (f["method"], f["position_label"], f["input_snr"], f["template_width"])
        agrupadas: dict = {}
        for f in filas:
            if f["exposure_id"] in ids:
                agrupadas.setdefault(clave(f), []).append(f)
        for k, grupo in sorted(agrupadas.items(), key=lambda kv: str(kv[0])):
            val = np.asarray([g["recovered_flux"] for g in grupo], dtype=np.float64)
            sig = np.asarray([g["recovered_sigma"] for g in grupo], dtype=np.float64)
            pes = np.asarray([pesos_plan[g["exposure_id"]] for g in grupo], dtype=np.float64)
            # `combine_measurements` devuelve (pesos, escala), no un valor: el
            # combinado es `escala * sum(w_i * x_i)`. Asumirlo en vez de leerlo
            # costo una corrida entera.
            pesos, escala = combine_measurements(val, pes, sig, ley)
            comb = float(escala * np.nansum(pesos * val))
            # La sigma se combina con LOS MISMOS pesos: si no, el cociente deja
            # de ser una S/N y pasa a mezclar dos estimadores.
            sigma_comb = float(escala * np.sqrt(np.nansum((pesos * sig) ** 2)))
            combinado.append({
                "grupo": nombre, "method": k[0], "position_label": k[1],
                "input_snr": k[2], "template_width": k[3],
                "n_exposiciones": len(grupo), "combine": ley,
                "recovered_flux": comb, "recovered_sigma": sigma_comb,
                "recovered_snr": comb / sigma_comb if sigma_comb > 0 else float("nan"),
            })

    qc00 = read_json(paths["stage00q_qc_json"]) if paths["stage00q_qc_json"].exists() else {}
    n_pos = len({f["position_label"] for f in filas})
    qc = {
        "stage": STAGE,
        "spec_version": SPEC_VERSION,
        "run_id": cfg["run_id"],
        "input": {
            "psf_model_mixture": str(paths["psf_model_mixture_json"]),
            "psf_model_mixture_sha256": sha256_file(paths["psf_model_mixture_json"]),
            "observation_plan": str(obs.source),
            "n_exposures": len(obs.exposures),
        },
        "convention": {
            "sigma": "empirica, de los controles DE CADA EXPOSICION (nunca STAT, nunca la del combinado)",
            "apertura": etiqueta_ap,
            "annulus_bkg_px": [float(x) for x in cfg["x06b_annulus_bkg_px"]],
            "flux_convention": flux_convention,
            "methods": [str(m) for m in cfg["x06b_methods"]],
            "combine": ley,
            "group_by": str(cfg["x06b_group_by"]),
            "line_center_A": line_center,
            "band_A": list(cfg["x06b_band_A"]) if cfg.get("x06b_band_A") else None,
            "lsf_fwhm_A": lsf_fwhm,
        },
        "wavelength_frame": _wavelength_frame(cfg, qc00, open_issues, knob="x06b_wframe"),
        # Tres campos distintos a proposito: confundir posiciones con filas ya
        # costo dos rondas de discusion (2026-08-27).
        "positions_per_point": n_pos,
        "injections_per_point": len({f["injection_id"] for f in filas}),
        "rows_per_point": len(filas),
        "grupos": {k: [e.exposure_id for e in v] for k, v in grupos.items()},
        "open_issues": open_issues,
    }
    return StageH04bProduct(filas=filas, combinado=combinado, qc=qc)


def _escribe_csv(path, filas):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not filas:
        return
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(filas[0]))
        w.writeheader()
        w.writerows(filas)


def write_stage_h04b_products(product: StageH04bProduct, config, paths) -> dict:
    _escribe_csv(paths["por_exposicion_csv"], product.filas)
    _escribe_csv(paths["combinado_csv"], product.combinado)
    qc = dict(product.qc)
    qc["tables"] = {
        "por_exposicion": str(paths["por_exposicion_csv"]),
        "combinado": str(paths["combinado_csv"]),
    }
    write_json(paths["qc_json"], qc)
    return {"qc": paths["qc_json"]}


def run_stage_h04b(run_id=None, *, project_root=None, overrides=None,
                   allow_run_id_mismatch=False):
    cfg = stage_h04b_config_from_run(run_id, project_root=project_root, overrides=overrides,
                                     allow_run_id_mismatch=allow_run_id_mismatch)
    paths = stage_h04b_paths(cfg["run_id"], project_root=cfg["project_root"])
    product = compute_stage_h04b_products(cfg, paths)
    written = write_stage_h04b_products(product, cfg, paths)
    return written["qc"]


def main(argv=None):
    ap = argparse.ArgumentParser(description="Etapa E4b - inyeccion-recuperacion por exposicion.")
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--project-root", default=None)
    ap.add_argument("--allow-run-id-mismatch", action="store_true")
    ap.add_argument("--group-by", choices=["none", "night"], default=None)
    ap.add_argument("--combine", choices=["invvar", "exptime", "equal", "sum"], default=None)
    ap.add_argument("--methods", default=None, help="lista separada por comas")
    ap.add_argument("--jobs", type=int, default=None)
    ap.add_argument("--band-A", nargs=2, type=float, default=None,
                    metavar=("LO", "HI"), help="ventana espectral de la medida")
    args = ap.parse_args(argv)
    overrides = {}
    if args.group_by:
        overrides["x06b_group_by"] = args.group_by
    if args.combine:
        overrides["x06b_combine"] = args.combine
    if args.methods:
        overrides["x06b_methods"] = [m.strip() for m in args.methods.split(",") if m.strip()]
    if args.jobs:
        overrides["x06b_max_workers"] = int(args.jobs)
    if args.band_A:
        overrides["x06b_band_A"] = list(args.band_A)
    print(run_stage_h04b(args.run_id, project_root=args.project_root, overrides=overrides,
                         allow_run_id_mismatch=args.allow_run_id_mismatch))
    return 0


__all__ = [
    "METODOS",
    "PerExpInjectionError",
    "StageH04bProduct",
    "compute_stage_h04b_products",
    "run_stage_h04b",
    "stage_h04b_config_from_run",
    "stage_h04b_paths",
    "write_stage_h04b_products",
]


if __name__ == "__main__":
    raise SystemExit(main())
