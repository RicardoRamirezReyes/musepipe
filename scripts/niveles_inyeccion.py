#!/usr/bin/env python3
"""Inyectar por exposicion vs inyectar en el combinado, con LA MISMA receta.

    python niveles_inyeccion.py ROXs12b_realigned --k 0,0.5,1,2,5 --controles 8

Por que hace falta
------------------
E4 inyecta en el cubo combinado (`_load_stage02_cube`). E4b inyecta por
exposicion, pero su escalera esta en sigmas DE CADA EXPOSICION
(`flujo = input_snr * sigma_exp`), asi que la fuente lleva un flujo DISTINTO en
cada exposicion: mide completitud por exposicion, no una fuente fisica observada
29 veces. Ninguna de las dos responde «¿inyectar solo en el combinado
sobreestima la recuperacion?», que es lo que afirma la postulacion.

Que hace este script
--------------------
Fija un flujo de linea ABSOLUTO F y lo inyecta:

  (a) en CADA exposicion alineada, con la PSF de esa exposicion, y combina las
      MEDIDAS con inverse-variance (lo que hace C7/E4b);
  (b) en el cubo combinado, con el modelo del combinado;

y compara la S/N recuperada. Receta identica en los dos: `_extrae` de E4b
(box3 + fondo de anillo, o psffit) y el mismo estimador de E1. Lo unico que
cambia es el SUSTRATO, que es lo que se quiere medir.

La escalera F se fija en sigmas DEL COMBINADO para que los dos niveles vean el
mismo flujo fisico.
"""
import argparse, json, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from astropy.io import fits

from musepipe.apertures import same_radius_control_positions
from musepipe.growth_curve import resolve_flux_convention
from musepipe.injection import InjectionSource, inject
from musepipe.io import read_json
from musepipe.observations import resolve_observation_plan
from musepipe.spectral import resolve_lsf_fwhm_A
from musepipe.stats import robust_sigma_axis0
from musepipe.stages.stage_e01_perobs import load_aligned_exposure
from musepipe.stages.stage_e01b_perobs_subtract import models_by_exposure
from musepipe.stages.stage_h04_injection import measure_recovery_with_h01_estimator
from musepipe.stages.stage_h04b_perexp_injection import (
    _extrae, _sigma_de_la_exposicion, stage_h04b_config_from_run, stage_h04b_paths,
)
from musepipe.stages.stage_x06_perexp import _positions

METODOS = ("aperture", "psffit")



def resta_linea_base(filas, clave):
    """Convierte flujo bruto en INCREMENTO sobre la medida sin inyectar.

    `clave(fila)` identifica la serie a la que pertenece la medida (posicion,
    metodo y, por exposicion, la exposicion). La fila con `k == 0` de esa serie
    es la linea base. Hace falta porque `psffit` arrastra un pedestal grande en
    las posiciones de control -E4 estandariza sus S/N por esto mismo-, asi que
    el flujo bruto no es la senal.

    Escribe `delta_flux` y `snr_delta` en cada fila y devuelve `filas`.
    """
    base = {clave(f): f["recovered_flux"] for f in filas if f["k"] == 0}
    for f in filas:
        b = base.get(clave(f), np.nan)
        f["delta_flux"] = f["recovered_flux"] - b
        sigma = f.get("recovered_sigma")
        f["snr_delta"] = (f["delta_flux"] / sigma) if sigma else np.nan
    return filas


def combina_invvar(grupo, campo="delta_flux"):
    """(valor, sigma, n) de la combinacion inverse-variance de unas medidas.

    Devuelve `(nan, nan, n_validas)` si no hay al menos dos medidas utilizables:
    con una sola no hay combinacion que hacer y el sigma propagado no significa
    nada.

    OJO: el sigma que sale es FORMAL -supone las medidas independientes-. En
    estos datos las de una misma posicion comparten halo y cielo, asi que para
    comparar niveles hay que usar la dispersion empirica entre posiciones
    (`docs/noise_model.md`); medido el 2026-09-18, con 8 controles la respuesta
    iba de 0.4x a 17x y solo con 47 se estabiliza.
    """
    w = np.array([1.0 / max(g["recovered_sigma"], 1e-30) ** 2 for g in grupo], dtype=np.float64)
    v = np.array([g[campo] for g in grupo], dtype=np.float64)
    buena = np.isfinite(w) & np.isfinite(v) & (w > 0)
    if buena.sum() < 2:
        return float("nan"), float("nan"), int(buena.sum())
    valor = float(np.sum(w[buena] * v[buena]) / np.sum(w[buena]))
    sigma = float(np.sqrt(1.0 / np.sum(w[buena])))
    return valor, sigma, int(buena.sum())


def _banda(cube, wave, banda, stat=None):
    if not banda:
        return cube, wave, stat
    sel = np.where((wave >= float(banda[0])) & (wave <= float(banda[1])))[0]
    lo, hi = sel[0], sel[-1] + 1
    return cube[lo:hi], wave[lo:hi], (None if stat is None else stat[lo:hi])


def _mide(cube, stat, wave, centro, star_yx, apertura, modelo, growth, cfg, err_ch,
          linea, lsf, ventana, metodo):
    flujo, err = _extrae(metodo, cube, stat, wave, centro, star_yx, apertura, modelo, growth, cfg)
    return measure_recovery_with_h01_estimator(
        (wave, flujo, err if err is not None else err_ch),
        line_center_A=linea, line_fwhm_A=lsf, continuum_window_A=ventana)


def _sigma_y_err(cube, stat, wave, controles, star_yx, apertura, modelo, growth, cfg,
                 linea, lsf, ventana):
    ctrl = np.asarray([_extrae("aperture", cube, stat, wave, (c[0], c[1]), star_yx,
                               apertura, modelo, growth, cfg)[0] for c in controles],
                      dtype=np.float64)
    err_ch = np.asarray(robust_sigma_axis0(ctrl), dtype=np.float64)
    err_ch = np.where(np.isfinite(err_ch) & (err_ch > 0), err_ch, np.nan)
    sigma, _ = _sigma_de_la_exposicion(ctrl, wave, linea, lsf, err_ch, ventana)
    return sigma, err_ch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--k", default="0,0.5,1,2,5", help="escalera en sigmas del COMBINADO")
    ap.add_argument("--controles", type=int, default=8)
    ap.add_argument("--max-exposiciones", type=int, default=0, help="0 = todas")
    ap.add_argument("--salida", default=None)
    a = ap.parse_args()
    t0 = time.time()
    ks = [float(x) for x in a.k.split(",")]

    cfg = stage_h04b_config_from_run(a.run)
    paths = stage_h04b_paths(a.run)
    doc = read_json(paths["psf_model_mixture_json"])
    modelos = models_by_exposure(doc)
    cache = paths["observation_plan_json"]
    obs = resolve_observation_plan(cfg["run_id"], project_root=cfg.get("project_root"),
                                   plan_json=cache if cache.exists() else None)
    plan = obs.plan
    exposiciones = list(obs.exposures)
    if a.max_exposiciones:
        exposiciones = exposiciones[:a.max_exposiciones]

    star_yx, comp_yx, _shape = _positions(paths, plan)
    npix = int(plan.crop_npix)
    controles = same_radius_control_positions(comp_yx, star_yx, npix, npix,
                                              n_positions=int(a.controles))
    posiciones = [("real", comp_yx), *((f"control{i+1}", tuple(c)) for i, c in enumerate(controles))]

    growth, _c = resolve_flux_convention(cfg, paths["paths"].stage_dir, knob="x06b_flux_convention")
    apertura = cfg["x06b_apertures"][0]
    linea = float(cfg.get("h04_line_center_A", cfg.get("h01_line_center_A", 6562.8)))
    ventana = float(cfg["x06b_continuum_window_A"])
    lsf, _f = resolve_lsf_fwhm_A(
        read_json(paths["stage00q_qc_json"]) if paths["stage00q_qc_json"].exists() else {},
        cfg, stage_key="x06b_lsf_fwhm_A")
    banda = cfg.get("x06b_band_A")

    # ---------- nivel COMBINADO: fija la escalera -----------------------------
    from musepipe.stages.stage_h04_injection import _load_stage02_cube, stage_h04_config_from_run
    from musepipe.stages.stage_h04_injection import stage_h04_paths
    cfg4 = stage_h04_config_from_run(a.run)
    comb, wave_c, _p = _load_stage02_cube(stage_h04_paths(a.run), cfg4)
    comb = np.asarray(comb, dtype=np.float64)
    if comb.ndim == 4:
        comb = comb[0]
    from musepipe.stages.stage_h04_extractors import _load_stat_cube_direct
    from musepipe.stages.stage_x01_aperture import stage_x01_config_from_run, stage_x01_paths
    _x01c = stage_x01_config_from_run(a.run)
    _x01p = stage_x01_paths(a.run, Path(cfg.get("project_root") or ROOT))
    stat_comb_full = _load_stat_cube_direct(_x01c, _x01p)
    comb, wave_c, stat_comb = _banda(comb, np.asarray(wave_c, float), banda, stat_comb_full)
    if stat_comb is None or np.shape(stat_comb) != np.shape(comb):
        raise SystemExit("el STAT del combinado no casa con el cubo: psffit no se puede medir igual en los dos niveles")
    modelo_comb = read_json(paths["paths"].stage_dir / "psf_model.json")
    sigma_comb, err_comb = _sigma_y_err(comb, stat_comb, wave_c, controles, star_yx, apertura,
                                        modelo_comb, growth, cfg, linea, lsf, ventana)

    filas = []
    for k in ks:
        F = k * sigma_comb
        for etiqueta, centro in posiciones:
            cubo = comb if k == 0 else inject(
                comb, [InjectionSource(y=centro[0], x=centro[1], total_line_flux=F,
                                       line_center_A=linea, line_fwhm_A=lsf,
                                       label=f"{etiqueta}_k{k}", continuum_flux_density=0.0,
                                       psf_fwhm_scale=1.0)],
                wavelengths_A=wave_c, psf_model=modelo_comb, copy=True)
            for metodo in METODOS:
                m = _mide(cubo, stat_comb, wave_c, centro, star_yx, apertura, modelo_comb, growth,
                          cfg, err_comb, linea, lsf, ventana, metodo)
                filas.append({"nivel": "combinado", "k": k, "flujo_inyectado": F,
                              "posicion": etiqueta, "method": metodo,
                              "recovered_flux": m["recovered_flux"],
                              "recovered_sigma": m["recovered_sigma"],
                              "recovered_snr": m["recovered_snr"]})

    # ---------- nivel POR EXPOSICION: el MISMO flujo absoluto ----------------
    por_exp = []
    for n_exp, exposicion in enumerate(exposiciones, 1):
        modelo = modelos[exposicion.exposure_id]
        cube, stat, wave = load_aligned_exposure(exposicion, plan)
        cube = np.asarray(cube, dtype=np.float64)
        cube, wave, stat = _banda(cube, np.asarray(wave, float), banda, stat)
        sigma_e, err_e = _sigma_y_err(cube, stat, wave, controles, star_yx, apertura,
                                      modelo, growth, cfg, linea, lsf, ventana)
        for k in ks:
            F = k * sigma_comb
            for etiqueta, centro in posiciones:
                cubo = cube if k == 0 else inject(
                    cube, [InjectionSource(y=centro[0], x=centro[1], total_line_flux=F,
                                           line_center_A=linea, line_fwhm_A=lsf,
                                           label=f"{etiqueta}_k{k}", continuum_flux_density=0.0,
                                           psf_fwhm_scale=1.0)],
                    wavelengths_A=wave, psf_model=modelo, copy=True)
                for metodo in METODOS:
                    m = _mide(cubo, stat, wave, centro, star_yx, apertura, modelo, growth,
                              cfg, err_e, linea, lsf, ventana, metodo)
                    por_exp.append({"exposure_id": exposicion.exposure_id, "k": k,
                                    "flujo_inyectado": F, "posicion": etiqueta,
                                    "method": metodo, "sigma_exposicion": sigma_e,
                                    "recovered_flux": m["recovered_flux"],
                                    "recovered_sigma": m["recovered_sigma"],
                                    "recovered_snr": m["recovered_snr"]})
        print(f"  exposicion {n_exp}/{len(exposiciones)} {exposicion.exposure_id} "
              f"sigma={sigma_e:.1f} ({(time.time()-t0)/60:.1f} min)", flush=True)

    # ---------- linea base (k=0), deltas y combinacion ----------------------
    resta_linea_base([f for f in filas if f["nivel"] == "combinado"],
                     lambda f: (f["posicion"], f["method"]))
    resta_linea_base(por_exp, lambda f: (f["exposure_id"], f["posicion"], f["method"]))

    grupos = {}
    for f in por_exp:
        grupos.setdefault((f["k"], f["posicion"], f["method"]), []).append(f)
    for (k, etiqueta, metodo), grupo in grupos.items():
        delta, sigma, n = combina_invvar(grupo)
        if not np.isfinite(delta):
            continue
        filas.append({"nivel": "por_exposicion", "k": k,
                      "flujo_inyectado": grupo[0]["flujo_inyectado"],
                      "posicion": etiqueta, "method": metodo, "n_exposiciones": n,
                      "delta_flux": delta, "recovered_sigma": sigma,
                      "snr_delta": delta / sigma if sigma else np.nan})

    # ---------- resumen -------------------------------------------------------
    def _nanmed(v):
        v = np.asarray([x for x in v if x is not None], dtype=np.float64)
        return float(np.nanmedian(v)) if v.size and np.isfinite(v).any() else None

    resumen = {"run": a.run, "n_exposiciones": len(exposiciones), "banda_A": banda,
               "controles": int(a.controles), "k": ks, "lsf_fwhm_A": lsf,
               "sigma_combinado": sigma_comb,
               "nota": ("senal = incremento sobre la medida sin inyectar de la misma "
                        "posicion; S/N = incremento / sigma del nivel; solo controles"),
               "minutos": round((time.time() - t0) / 60, 1), "comparacion": {}}
    for metodo in METODOS:
        for k in ks:
            if k == 0:
                continue
            c = [f for f in filas if f["nivel"] == "combinado" and f["k"] == k
                 and f["method"] == metodo and f["posicion"] != "real"]
            p_ = [f for f in filas if f["nivel"] == "por_exposicion" and f["k"] == k
                  and f["method"] == metodo and f["posicion"] != "real"]
            if not c or not p_:
                continue
            sc = _nanmed([x["snr_delta"] for x in c])
            sp = _nanmed([x["snr_delta"] for x in p_])
            F = k * sigma_comb
            resumen["comparacion"][f"{metodo}|k={k}"] = {
                "snr_combinado": sc, "snr_por_exposicion": sp,
                "razon_perexp_sobre_combinado": (sp / sc) if (sc and sp is not None) else None,
                "recuperado_sobre_inyectado_combinado": (_nanmed([x["delta_flux"] for x in c]) / F) if F else None,
                "recuperado_sobre_inyectado_perexp": (_nanmed([x["delta_flux"] for x in p_]) / F) if F else None,
                "n_controles": len(c)}
    salida = Path(a.salida or f"niveles_{a.run}.json")
    salida.write_text(json.dumps({"resumen": resumen, "filas": filas, "por_exposicion": por_exp}, indent=1))
    print(json.dumps(resumen["comparacion"], indent=1))
    print("sigma_combinado =", sigma_comb, "| escrito", salida)


if __name__ == "__main__":
    main()
