#!/usr/bin/env python3
"""V1 de la spec de E4b: el ancla entre los dos niveles de inyeccion.

    python v1_ancla_e4b.py ROXs12b_realigned --snr 0,1,5 --controles 8

La spec (`docs/spec_E4b_codex_perexp_injection.md` §5) dice: «con una sola
exposicion y sin agrupar, la fila tiene que reproducir lo que E4 daria sobre ese
mismo cubo». No existia ni test ni medida. Esto lo mide sobre los runs reales.

Que compara, exactamente
------------------------
Sobre UNA exposicion, en su cubo alineado y en la banda de E4b:

  * se inyecta UNA vez por caso (misma fuente, misma PSF de esa exposicion,
    misma convencion `norm_radius`);
  * se extrae por los DOS caminos: el de E4b (`_extrae`: apertura box3 con fondo
    de anillo, o psffit) y el de E4 (`build_production_extractors`, los mismos
    extractores de produccion que usa la etapa sobre el combinado);
  * se mide con EL MISMO estimador (`measure_recovery_with_h01_estimator`).

Si los dos caminos dan el mismo flujo recuperado, el ancla se sostiene y la
unica diferencia entre niveles es el SUSTRATO y la ESCALA de sigma, no la
maquinaria de medida.

Ademas informa el puente de escalas sobre ese mismo cubo: `sigma_flux` de E4
(calibrada en la posicion del companero, `_derive_injection_sigma`) contra
`sigma_exp` de E4b (empirica, de los controles de la exposicion). Ese cociente
es lo que hace comparables las dos tablas.
"""
import argparse, json, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from musepipe.apertures import same_radius_control_positions
from musepipe.growth_curve import resolve_flux_convention
from musepipe.injection import InjectionSource, inject
from musepipe.io import read_json
from musepipe.observations import resolve_observation_plan
from musepipe.stages.stage_e01_perobs import load_aligned_exposure
from musepipe.stages.stage_e01b_perobs_subtract import models_by_exposure
from musepipe.stages.stage_h04_extractors import build_production_extractors
from musepipe.stages.stage_h04_injection import (
    _call_extractor, _derive_injection_sigma, build_h04_cases,
    measure_recovery_with_h01_estimator, stage_h04_config_from_run, stage_h04_paths,
)
from musepipe.stages.stage_h04b_perexp_injection import (
    _extrae, _sigma_de_la_exposicion, stage_h04b_config_from_run, stage_h04b_paths,
)
from musepipe.stages.stage_x06_perexp import _positions

METODOS = ("aperture", "psffit")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--snr", default="0,1,5")
    ap.add_argument("--controles", type=int, default=8)
    ap.add_argument("--exposicion", type=int, default=0, help="indice en el plan")
    ap.add_argument("--salida", default=None)
    a = ap.parse_args()
    t0 = time.time()
    rejilla = [float(s) for s in a.snr.split(",")]

    cfgb = stage_h04b_config_from_run(a.run)
    pathsb = stage_h04b_paths(a.run)
    cfg4 = stage_h04_config_from_run(a.run)
    paths4 = stage_h04_paths(a.run)

    doc = read_json(pathsb["psf_model_mixture_json"])
    if str(doc.get("form", "")).lower() != "mixture":
        raise SystemExit(f"{a.run}: psf_model_mixture.json no es una mezcla por observacion")
    modelos = models_by_exposure(doc)
    cache = pathsb["observation_plan_json"]
    obs = resolve_observation_plan(cfgb["run_id"], project_root=cfgb.get("project_root"),
                                   plan_json=cache if cache.exists() else None)
    plan = obs.plan
    exposicion = obs.exposures[a.exposicion]
    model_doc = modelos[exposicion.exposure_id]

    cube, stat, wave = load_aligned_exposure(exposicion, plan)
    banda = cfgb.get("x06b_band_A")
    if banda:
        sel = np.where((wave >= float(banda[0])) & (wave <= float(banda[1])))[0]
        cube = cube[sel[0]:sel[-1] + 1]; stat = stat[sel[0]:sel[-1] + 1]; wave = wave[sel[0]:sel[-1] + 1]
    cube = np.asarray(cube, dtype=np.float64)

    star_yx, comp_yx, _shape = _positions(pathsb, plan)
    npix = int(plan.crop_npix)
    controles = same_radius_control_positions(comp_yx, star_yx, npix, npix,
                                              n_positions=int(a.controles))
    posiciones = [{"label": "real", "y": comp_yx[0], "x": comp_yx[1]},
                  *({"label": f"control{i+1}", "y": c[0], "x": c[1]} for i, c in enumerate(controles))]

    cfg4 = dict(cfg4)
    cfg4["h04_snr_grid"] = rejilla
    cfg4["h04_template_width_factors"] = [1.0]
    cfg4["h04_psf_perturb_snr_grid"] = []
    casos = [c for c in build_h04_cases(cfg4, posiciones) if c.variant == "nominal"]

    growth_curve, _conv = resolve_flux_convention(cfgb, pathsb["paths"].stage_dir,
                                                  knob="x06b_flux_convention")
    apertura = cfgb["x06b_apertures"][0]
    linea = float(cfgb.get("h04_line_center_A", cfgb.get("h01_line_center_A", 6562.8)))
    ventana = float(cfgb["x06b_continuum_window_A"])
    from musepipe.spectral import resolve_lsf_fwhm_A
    lsf, _fuente = resolve_lsf_fwhm_A(
        read_json(pathsb["stage00q_qc_json"]) if pathsb["stage00q_qc_json"].exists() else {},
        cfgb, stage_key="x06b_lsf_fwhm_A")

    # --- sigma de la exposicion (camino E4b), de sus propios controles --------
    from musepipe.stats import robust_sigma_axis0
    ctrl_flux = np.asarray(
        [_extrae("aperture", cube, stat, wave, (c[0], c[1]), star_yx, apertura, model_doc,
                 growth_curve, cfgb)[0] for c in controles], dtype=np.float64)
    err_ch = np.asarray(robust_sigma_axis0(ctrl_flux), dtype=np.float64)
    err_ch = np.where(np.isfinite(err_ch) & (err_ch > 0), err_ch, np.nan)
    sigma_exp, _m = _sigma_de_la_exposicion(ctrl_flux, wave, linea, lsf, err_ch, ventana)

    # --- extractores de produccion (camino E4) sobre ESTE cubo ---------------
    extractores = build_production_extractors(cfg4, paths4, wave_A=wave,
                                              psf_model=model_doc, base_cube=cube)
    sigma_flux_e4, calib = _derive_injection_sigma(
        {**cfg4, "h04_real_position_yx": [comp_yx[0], comp_yx[1]]},
        paths4, extractores, cube, wave, model_doc, injection_psf_model=model_doc)

    filas = []
    for caso in casos:
        centro = (float(caso.position_y), float(caso.position_x))
        flujo = float(caso.input_snr) * sigma_exp
        fuente = InjectionSource(y=centro[0], x=centro[1], total_line_flux=flujo,
                                 line_center_A=linea, line_fwhm_A=lsf * float(caso.template_factor),
                                 label=caso.injection_id, continuum_flux_density=0.0,
                                 psf_fwhm_scale=float(caso.psf_fwhm_scale))
        cube_inj = inject(cube, [fuente], wavelengths_A=wave, psf_model=model_doc, copy=True)
        for metodo in METODOS:
            f_b, e_b = _extrae(metodo, cube_inj, stat, wave, centro, star_yx, apertura,
                               model_doc, growth_curve, cfgb)
            m_b = measure_recovery_with_h01_estimator(
                (wave, f_b, e_b if e_b is not None else err_ch),
                line_center_A=linea, line_fwhm_A=lsf * float(caso.template_factor),
                continuum_window_A=ventana)
            res4 = _call_extractor(extractores[metodo], cube_inj, wave, caso, metodo, cfg4)
            m_4 = measure_recovery_with_h01_estimator(
                res4, line_center_A=linea, line_fwhm_A=lsf * float(caso.template_factor),
                continuum_window_A=ventana)
            filas.append({
                "injection_id": caso.injection_id, "position_label": caso.position_label,
                "input_snr": float(caso.input_snr), "method": metodo,
                "injected_flux": flujo,
                "e4b_recovered_flux": m_b["recovered_flux"], "e4b_recovered_snr": m_b["recovered_snr"],
                "e4_recovered_flux": m_4["recovered_flux"], "e4_recovered_snr": m_4["recovered_snr"],
            })

    resumen = {"run": a.run, "exposicion": exposicion.exposure_id,
               "n_exposiciones_plan": len(obs.exposures),
               "banda_A": banda, "n_casos": len(casos), "controles": int(a.controles),
               "rejilla_snr": rejilla, "lsf_fwhm_A": lsf,
               "sigma_exp_e4b": sigma_exp, "sigma_flux_e4": sigma_flux_e4,
               "puente_sigma_e4_sobre_e4b": sigma_flux_e4 / sigma_exp if sigma_exp else None,
               "calibracion_e4": {k: v for k, v in calib.items()},
               "minutos": round((time.time() - t0) / 60, 1), "por_metodo": {}}
    for metodo in METODOS:
        sub = [f for f in filas if f["method"] == metodo and f["input_snr"] > 0]
        if not sub:
            continue
        dif_rel = [abs(f["e4_recovered_flux"] - f["e4b_recovered_flux"]) / abs(f["e4b_recovered_flux"])
                   for f in sub if f["e4b_recovered_flux"]]
        dif_snr = [abs(f["e4_recovered_snr"] - f["e4b_recovered_snr"]) for f in sub]
        resumen["por_metodo"][metodo] = {
            "n": len(sub),
            "dif_relativa_flujo_mediana": float(np.median(dif_rel)) if dif_rel else None,
            "dif_relativa_flujo_max": float(np.max(dif_rel)) if dif_rel else None,
            "dif_snr_mediana": float(np.median(dif_snr)), "dif_snr_max": float(np.max(dif_snr)),
            "e4b_flujo_mediano": float(np.median([f["e4b_recovered_flux"] for f in sub])),
            "e4_flujo_mediano": float(np.median([f["e4_recovered_flux"] for f in sub])),
        }
    salida = Path(a.salida or f"v1_ancla_{a.run}.json")
    salida.write_text(json.dumps({"resumen": resumen, "filas": filas}, indent=1))
    print(json.dumps(resumen, indent=1))
    print("escrito", salida)


if __name__ == "__main__":
    main()
