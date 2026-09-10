"""Cuanto espectro borran SGF y LPM, y como depende de la separacion.

La pregunta. Los metodos tipo HRSDI (Haffert+19) construyen la referencia estelar
a partir de los propios datos y la restan, asi que se llevan parte del espectro del
compañero. En esta pipeline eso YA esta formalizado: `musepipe/halosub.py` implementa
SGF citando el paper, `sgf_self_subtraction_ratio` es su Eq. 1, D1 excluye `sgf` y
`lpm` del veredicto de continuo («removed by construction»), y los QC publican el
predictor sobre las LINEAS (−1.3 % a −2.1 %).

Lo que NO estaba medido es **como depende de la separacion**. Esto lo mide sobre los
cubos reales en vez de derivarlo, y lo que sale (2026-09-10) **no es lo que nadie
esperaba**:

  * **La linea de la compañera NO se borra**: inyectada SOLA, sobrevive al 0.989 (SGF)
    y al 1.000 (LPM).
  * Lo que hay es una **linea de absorcion ESPURIA** que aparece al inyectar **solo
    continuo, sin ninguna linea**. Su tamaño es proporcional al CONTINUO de la
    compañera, no a su linea.
  * El mecanismo es la **Halpha de la PRIMARIA** impresa en el halo: `d/s_hat` de un
    continuo plano tiene un hoyo estrecho donde `s_hat` tiene su emision, y el filtro
    paso-bajo (ventana 101 canales = 126 A) no lo sigue, asi que se sobre-resta ahi.
  * Por eso **no depende de la separacion** (continuo y halo escalan juntos, el
    cociente no) y **si depende del objeto** (el contraste de Halpha de cada primaria).

Por eso se inyecta cada componente POR SEPARADO. El caso combinado medido con una
metrica de amplitud NO es «fraccion de linea retenida»: la absorcion espuria y la
emision inyectada se solapan, y un maximo no es aditivo.

Como se mide, y por que asi:

- **Por diferencia**: se corre la cadena entera DOS veces por posicion, sobre el cubo
  limpio y sobre el cubo con un compañero sintetico inyectado, y se resta lo extraido.
  Las dos comparten la misma realizacion de ruido, asi que la diferencia **no tiene
  ruido de fotones**: es la respuesta no lineal del metodo, que es exactamente el
  borrado. La dispersion que quede entre angulos es estructura real del halo.
- **Tres inyecciones por posicion**: linea+continuo, SOLO linea, SOLO continuo. Es la
  descomposicion que separa «me borran la linea» de «me fabrican una linea», que son
  cosas distintas y en estos datos solo pasa la segunda.
- **Metrica integrada y LINEAL** para la linea (integral en la ventana menos continuo
  local), no un maximo: solo la lineal permite comprobar que las dos componentes suman.
  El continuo inyectado es **plano**, asi que cualquier estructura espectral en la
  diferencia es del metodo, no de la fuente.
- **Sin correccion de apertura**: se compara contra el flujo inyectado DENTRO de box3,
  calculado exacto del template. Asi el borrado no se mezcla con el error de la apcorr,
  que es un frente aparte (y hoy una puerta roja de C1/C2).
- **Control**: la misma medida sin restar halo (`none`). Sin el, «recuperado/inyectado»
  mezcla el borrado con la fotometria de box3.
- **>=3 angulos por separacion**: una sola posicion no es una medida — en el ciclo del
  09-09 una posicion sola difirio hasta 0.05 de la media de ocho.
- **Dos amplitudes** en la prueba de humo: si el cociente depende de la amplitud, la no
  linealidad es fuerte y el numero no se puede citar como «el borrado del metodo».

Supuestos declarados: el template inyecta la PSF evaluada en la lambda de la linea
para el continuo tambien (PSF gris), asi que no reproduce el cromatismo espacial; y usa
el `psf_model` publicado, no la PSF empirica.

Solo lectura sobre el run: no escribe nada dentro de `runs/`.

COSTE DE MEMORIA, medido el 2026-09-10: ~14-16 GB por proceso. `measure_one` reserva
varios cubos de (3681, 170, 170) en float64 -template, cubo inyectado, combinado- a
~850 MB cada uno, y con dos objetos a la vez se van 30 GB. Es holgado pero no gratis:
pasar los templates a float32 y soltarlos en cuanto se usan lo bajaria mucho. Sin hacer.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from musepipe.apertures import aperture_weights  # noqa: E402
from musepipe.halosub import (  # noqa: E402
    lpm_subtract,
    sgf_subtract,
    sgf_self_subtraction_ratio,
)
from musepipe.injection import InjectionSource, source_template  # noqa: E402
from musepipe.io import read_json  # noqa: E402
from musepipe.spectral import STANDARD_LINE_WINDOWS_A, resolve_lsf_fwhm_A  # noqa: E402
from musepipe.stages.halosub_stage import (  # noqa: E402
    BOX3_APERTURE,
    combine_residuals,
    load_stage02_exposures,
    subtract_exposures,
)


from musepipe.stages.stage_x04_sgf import (  # noqa: E402
    DEFAULT_SGF_DEGREE,
    DEFAULT_SGF_WINDOW,
    stage_x04_config_from_run,
    stage_x04_paths,
)
from musepipe.stages.stage_x05_lpm import (  # noqa: E402
    stage_x05_config_from_run,
    stage_x05_paths,
)

#: banda de continuo donde el compañero real SI esta detectado (S/N 13 y 65).
#: Medir el borrado del continuo en el azul seria medir un cociente de ruidos.
CONT_BAND_A = (8600.0, 9100.0)
HALPHA_REST_A = 6562.8


def resolve_lsf(paths, cfg):
    """La LSF y su procedencia, por el resolutor compartido de la pipeline.

    Esta funcion tenia aqui su propia cascada porque, cuando se midio el borrado
    (2026-09-10), `lsf_fwhm_A_from_qc_or_config` devolvia el default de 2.6 A: buscaba
    la LSF en dos claves que A4 no escribe y luego en un knob que el config no declara
    con ese nombre. Eso ya esta arreglado en `spectral.resolve_lsf_fwhm_A`, asi que
    aqui se llama a la de la pipeline y no se mantiene una copia.
    """

    qc00 = read_json(paths["stage00q_qc_json"]) if paths["stage00q_qc_json"].exists() else {}
    valor, procedencia = resolve_lsf_fwhm_A(qc00, cfg, stage_key="halosub_lsf_fwhm_A")
    medida = ((qc00.get("m2_lsf") or {}).get("lsf_fwhm_at_halpha_A")
              if isinstance(qc00, dict) else None)
    return valor, {"used_here": float(valor), "source": procedencia,
                   "measured_a4_m2": None if medida is None else float(medida)}


def _subtract_fn_for(method, cfg, wave):
    """El MISMO `subtract_fn` que monta la etapa, no una reimplementacion."""

    if method == "sgf":
        window = int(cfg.get("sgf_window", DEFAULT_SGF_WINDOW))
        degree = int(cfg.get("sgf_degree", DEFAULT_SGF_DEGREE))

        def _f(cube, wave_A, s_hat):  # noqa: ARG001
            res = sgf_subtract(cube, s_hat, window=window, degree=degree)
            return res.residual_cube, {"window": window, "degree": degree}

        return _f, {"window": window, "degree": degree}

    if method == "lpm":
        degree = int(cfg.get("lpm_degree", 4))
        windows = cfg.get("lpm_line_windows_A") or STANDARD_LINE_WINDOWS_A

        def _f(cube, wave_A, s_hat):
            res = lpm_subtract(cube, wave_A, s_hat, degree=degree, line_windows_A=windows)
            return res.residual_cube, {"degree": degree}

        return _f, {"degree": degree}

    if method == "none":
        # Control: no se resta halo. Aisla la fotometria de box3.
        def _f(cube, wave_A, s_hat):  # noqa: ARG001
            return np.asarray(cube, dtype=np.float64), {"subtraction": "none"}

        return _f, {"subtraction": "none"}

    raise ValueError(f"metodo desconocido: {method!r}")


def _box3_spectrum(cube_zyx, pos_yx):
    """Suma en box3 canal a canal, con la MISMA definicion de la pipeline."""

    cube = np.asarray(cube_zyx, dtype=np.float64)
    _nz, ny, nx = cube.shape
    w = aperture_weights(ny, nx, pos_yx, dict(BOX3_APERTURE))
    return np.nansum(cube * w[None, :, :], axis=(1, 2))


def _line_and_continuum(wave, spec, *, line_center_A, line_fwhm_A):
    """Linea (dos metricas) y nivel de continuo.

    Las DOS metricas de linea, porque dicen cosas distintas y confundirlas da un
    factor 10: `amp` es el pico sobre el continuo local, o sea cuanto de la linea
    sobrevive DONDE ESTA; `integrated` integra la ventana y por tanto se come el valle
    ancho que el filtro deja alrededor, que es un efecto aparte. Medido: -0.565 en
    amplitud contra -5.5 integrado, sobre la misma medida.
    """

    half = 2.0 * float(line_fwhm_A)
    inside = np.abs(wave - line_center_A) <= half
    near = (np.abs(wave - line_center_A) > half) & (np.abs(wave - line_center_A) <= 8.0 * half)
    cont_local = float(np.nanmedian(spec[near])) if near.any() else 0.0
    dl = np.gradient(wave)
    integrated = (float(np.nansum((spec[inside] - cont_local) * dl[inside]))
                  if inside.any() else float("nan"))
    amp = float(np.nanmax(spec[inside]) - cont_local) if inside.any() else float("nan")
    band = (wave >= CONT_BAND_A[0]) & (wave <= CONT_BAND_A[1])
    cont_band = float(np.nanmedian(spec[band])) if band.any() else float("nan")
    return {"integrated": integrated, "amp": amp}, cont_band, cont_local


def _positions(primary_yx, sep_px, pa_deg, shape, margin=14.0):
    """Posicion a `sep_px` del primario con angulo `pa_deg`, si cabe en el campo."""

    ny, nx = shape
    a = math.radians(float(pa_deg))
    y = float(primary_yx[0]) + sep_px * math.cos(a)
    x = float(primary_yx[1]) + sep_px * math.sin(a)
    if not (margin <= y <= ny - 1 - margin and margin <= x <= nx - 1 - margin):
        return None
    return (y, x)


def measure_one(exposures_raw, wave, cfg, pos_yx, method, *, psf_model, src,
                lsf_fwhm_A, line_center_A):
    """Una posicion, un metodo: cadena limpia, cadena inyectada, y la diferencia."""

    subtract_fn, info = _subtract_fn_for(method, cfg, wave)

    # 1. inyectar el MISMO template en cada exposicion
    templates = [source_template(c.shape, wave, src, psf_model=psf_model) for c in exposures_raw]
    injected_raw = [c + t for c, t in zip(exposures_raw, templates)]

    # 2. la verdad DENTRO de box3, exacta, del template combinado igual que los datos
    truth_cube = combine_residuals(
        [type("E", (), {"residual": t})() for t in templates],
        combine=cfg.get("halosub_combine", "mean"),
    )
    truth_spec = _box3_spectrum(truth_cube, pos_yx)
    truth_line, truth_cont, _ = _line_and_continuum(
        wave, truth_spec, line_center_A=line_center_A, line_fwhm_A=lsf_fwhm_A)

    out = {"method": method, "pos_yx": list(pos_yx), "info": info,
           "truth_line_box3": truth_line, "truth_cont_box3": truth_cont}

    # 3. las dos cadenas, con la MISMA exclusion de referencia en la posicion
    specs = {}
    for tag, cubes in (("clean", exposures_raw), ("injected", injected_raw)):
        exps = subtract_exposures(cubes, wave, cfg, pos_yx, subtract_fn)
        residual = combine_residuals(exps, combine=cfg.get("halosub_combine", "mean"))
        specs[tag] = _box3_spectrum(residual, pos_yx)
        out[f"n_spaxels_kept_{tag}"] = int(np.median([e.n_spaxels_kept for e in exps]))

    diff = specs["injected"] - specs["clean"]
    rec_line, rec_cont, _ = _line_and_continuum(
        wave, diff, line_center_A=line_center_A, line_fwhm_A=lsf_fwhm_A)
    out["recovered_line_box3"] = rec_line
    out["recovered_cont_box3"] = rec_cont
    out["line_retained_frac_amp"] = (
        rec_line["amp"] / truth_line["amp"] if truth_line["amp"] else None)
    out["line_retained_frac_integrated"] = (
        rec_line["integrated"] / truth_line["integrated"] if truth_line["integrated"] else None)
    out["cont_retained_frac"] = (rec_cont / truth_cont) if truth_cont else None
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--methods", default="sgf,lpm,none")
    ap.add_argument("--separations-px", default="20,35,50,71,95,120")
    ap.add_argument("--pa-deg", default="0,120,240")
    ap.add_argument("--amplitude-scale", type=float, default=1.0,
                    help="escala del compañero sintetico respecto al real")
    ap.add_argument("--smoke", action="store_true",
                    help="2 separaciones x 2 angulos y cronometra, para dimensionar")
    ap.add_argument("--checkpoint", default=None,
                    help="JSON donde ir guardando; se reanuda si existe")
    args = ap.parse_args()

    cfg = stage_x04_config_from_run(args.run)
    cfg_lpm = stage_x05_config_from_run(args.run)
    for k, v in cfg_lpm.items():
        cfg.setdefault(k, v)
    paths = stage_x04_paths(args.run, ROOT)
    _paths_lpm = stage_x05_paths(args.run, ROOT)

    exposures_raw, wave, cube_path, bunit, best = load_stage02_exposures(paths, cfg)
    pos_qc = read_json(paths["stage01c_qc_json"])
    primary_yx = tuple(float(v) for v in pos_qc["primary"]["pos_yx"])
    comp_yx = tuple(float(v) for v in pos_qc["companion"]["pos_yx"])
    real_sep = math.hypot(comp_yx[0] - primary_yx[0], comp_yx[1] - primary_yx[1])
    scale_arcsec = float(pos_qc.get("pixel_scale_arcsec", 0.0253))
    psf_model = read_json(paths["psf_model_json"])
    lsf_fwhm_A, lsf_provenance = resolve_lsf(paths, cfg)
    rv = float(cfg.get("h01_rv_sys_kms", 0.0))
    line_center_A = HALPHA_REST_A * (1.0 + rv / 299792.458)

    print(f"run {args.run}: {len(exposures_raw)} exposiciones, cubo {exposures_raw[0].shape}")
    print(f"  primaria {primary_yx}  compañera {comp_yx}")
    print(f"  separacion real {real_sep:.2f} px = {real_sep*scale_arcsec:.3f} arcsec")
    print(f"  LSF {lsf_fwhm_A:.4f} A, linea en {line_center_A:.2f} A (rv {rv:+.2f} km/s)")
    print(f"  procedencia: {lsf_provenance['source']} "
          f"(A4/M2 midio {lsf_provenance['measured_a4_m2']})")

    # amplitud del compañero sintetico, anclada al real para estar en su regimen
    from astropy.io import fits
    with fits.open(paths["paths"].stage_dir / "spec_aperture_object.fits") as h:
        t = h[1].data
        w_r = np.asarray(t["wave_A"], float)
        f_r = np.asarray(t["flux"], float)
    band = (w_r >= CONT_BAND_A[0]) & (w_r <= CONT_BAND_A[1])
    cont_real = float(np.nanmedian(f_r[band]))
    import csv as _csv
    with open(paths["paths"].table_dir / "halpha_detection_by_method.csv", newline="") as fh:
        row = [r for r in _csv.DictReader(fh) if r["method"] == "aperture"][0]
    line_real = float(row["matched_flux"])
    amp = float(args.amplitude_scale)
    print(f"  anclaje: continuo real {cont_real:.1f}, linea real {line_real:.1f}, escala x{amp}")

    seps = [float(s) for s in args.separations_px.split(",")]
    pas = [float(p) for p in args.pa_deg.split(",")]
    methods = [m.strip() for m in args.methods.split(",")]
    if args.smoke:
        seps, pas = seps[:2], pas[:2]
        print("  MODO HUMO: 2 separaciones x 2 angulos")

    done = {}
    ck = Path(args.checkpoint) if args.checkpoint else None
    if ck and ck.exists():
        done = {r["key"]: r for r in json.loads(ck.read_text()).get("rows", [])}
        print(f"  reanudando: {len(done)} medidas ya hechas")

    rows = list(done.values())
    t_all = time.time()
    for sep in seps:
        for pa in pas:
            pos = _positions(primary_yx, sep, pa, exposures_raw[0].shape[1:])
            if pos is None:
                print(f"  sep={sep:.0f} pa={pa:.0f}: fuera del campo, salto")
                continue
            modos = {"line+cont": (line_real * amp, cont_real * amp),
                     "line_only": (line_real * amp, 0.0),
                     "cont_only": (0.0, cont_real * amp)}
            for modo, (lf_i, cd_i) in modos.items():
                src = InjectionSource(
                    y=pos[0], x=pos[1],
                    total_line_flux=lf_i,
                    line_center_A=line_center_A,
                    line_fwhm_A=lsf_fwhm_A,
                    continuum_flux_density=cd_i,
                    label=f"{modo}_sep{sep:.0f}_pa{pa:.0f}",
                )
                for method in methods:
                    key = f"{modo}|{method}|{sep:.0f}|{pa:.0f}|{amp:g}"
                    if key in done:
                        continue
                    t0 = time.time()
                    r = measure_one(exposures_raw, wave, cfg, pos, method,
                                    psf_model=psf_model, src=src,
                                    lsf_fwhm_A=lsf_fwhm_A, line_center_A=line_center_A)
                    r.update({"key": key, "mode": modo, "sep_px": sep,
                              "sep_arcsec": sep * scale_arcsec, "pa_deg": pa,
                              "amplitude_scale": amp,
                              "seconds": round(time.time() - t0, 1)})
                    rows.append(r)
                    li = r["line_retained_frac_integrated"]
                    cf = r["cont_retained_frac"]
                    # En `cont_only` no hay linea inyectada, asi que la fraccion no
                    # existe: lo que importa es el flujo ESPURIO en unidades de la
                    # linea real del objeto, y eso es lo que se imprime.
                    esp = r["recovered_line_box3"]["integrated"]
                    print(f"  {modo:9s} {method:5s} sep={sep:5.0f}px "
                          f"({sep*scale_arcsec:.2f}\") pa={pa:3.0f}  "
                          + (f"linea retenida {li:8.4f}  " if li is not None
                             else f"linea espuria {esp:10.1f}  ")
                          + f"continuo {cf if cf is None else f'{cf:8.4f}'}"
                          + f"  [{r['seconds']:.0f}s]")
                    if ck:
                        ck.write_text(json.dumps({"rows": rows}, indent=1))

    # prediccion analitica en cada punto, para superponerla a la medida
    predictor = None
    try:
        # La MISMA formula que `self_subtraction_predictors`: R = (LSF/dlambda)/ventana.
        window = int(cfg.get("sgf_window", DEFAULT_SGF_WINDOW))
        dl = float(np.nanmedian(np.diff(wave)))
        r_ratio = (float(lsf_fwhm_A) / dl) / float(window)
        predictor = {"R": r_ratio,
                     "note": ("Eq.1 del paper: C~_P/L^_P = -(R/(1-R))*(C_S/L_S). R sale de "
                              "la ventana del SGF, no de la separacion; la dependencia con la "
                              "separacion entra por C_S/C_P, que es lo que esta medida aisla."),
                     "window_channels": window, "dlambda_A": dl,
                     "at_cs_over_ls_1": sgf_self_subtraction_ratio(r_ratio, 1.0)}
    except Exception as exc:  # pragma: no cover
        predictor = {"error": repr(exc)}

    payload = {
        "stage": "halosub_borrado_vs_separacion", "run": args.run,
        "cube": str(cube_path), "bunit": bunit, "n_exposures": len(exposures_raw),
        "exposure_indices": best,
        "primary_yx": list(primary_yx), "companion_yx": list(comp_yx),
        "real_sep_px": real_sep, "real_sep_arcsec": real_sep * scale_arcsec,
        "pixel_scale_arcsec": scale_arcsec,
        "lsf_fwhm_A": lsf_fwhm_A, "lsf_provenance": lsf_provenance,
        "line_center_A": line_center_A,
        "continuum_band_A": list(CONT_BAND_A),
        "anchor": {"cont_real": cont_real, "line_real": line_real,
                   "amplitude_scale": amp},
        "assumptions": [
            "el template usa la PSF evaluada en la lambda de la linea tambien para el "
            "continuo (PSF gris): no reproduce el cromatismo espacial",
            "usa el psf_model PUBLICADO, no la PSF empirica de la primaria",
            "el continuo inyectado es PLANO: cualquier estructura espectral en la "
            "diferencia es del metodo",
            "medido SIN correccion de apertura, contra el flujo inyectado dentro de "
            "box3, para no mezclarlo con el error de la apcorr",
        ],
        "controls_measured_2026_09_10": {
            "amplitude_linearity": ("respuesta LINEAL: -0.564/-0.565/-0.567 para escalas "
                                    "0.1/1.0/3.0 (rango de 30x)"),
            "reference_exclusion_radius": ("plano de 3 a 40 px en los DOS metodos: el efecto "
                                           "NO viene de que la fuente contamine su propia "
                                           "referencia -hipotesis mia, falsada con control-"),
            "additivity": ("con la metrica integrada, solo_linea + solo_continuo reproduce "
                           "el combinado al 0.02 % en los dos metodos: la descomposicion es "
                           "exacta y el efecto es lineal"),
            "decomposition": ("la linea inyectada SOLA sobrevive (0.983 SGF, 1.000 LPM); el "
                              "continuo inyectado SOLO fabrica una absorcion espuria de "
                              "-2048 (SGF) y -2211 (LPM) en 12 b, o sea 6.5x el flujo de la "
                              "linea real. NO borran la linea: la fabrican."),
        },
        "analytic_predictor": predictor,
        "rows": rows,
        "total_seconds": round(time.time() - t_all, 1),
    }
    Path(args.out_json).write_text(json.dumps(payload, indent=1, ensure_ascii=False))
    print(f"\n{len(rows)} medidas en {payload['total_seconds']:.0f}s -> {args.out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
