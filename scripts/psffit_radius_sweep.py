#!/usr/bin/env python
"""Los radios del ajuste de `psffit`, elegidos por completitud y fuera de muestra.

`x03_star_radius_px` (20) y `x03_comp_radius_px` (12) fijan sobre que pixeles se
resuelve el ajuste de 5 columnas de `psffit`. El barrido del 2026-08-29 midio que
son el lever mas grande del estimador —la SNR del companero va de +1.91 a +5.09
en ROXs 12 b— y **que el optimo no coincide entre los dos objetos**, asi que lo
que falta no son numeros sino un criterio que no se sobreajuste. Esta sonda es
ese criterio.

Tres decisiones de diseno, y las tres son la diferencia entre medir y enganarse:

1. **El criterio es `f50`, no la SNR del companero real.** `f50` es el SNR
   inyectado al que se recupera el 50 % de las inyecciones: una curva sobre
   decenas de posiciones, no una medida unica con incertidumbre ~±0.5. Menor es
   mejor. La SNR del companero real viaja como columna de referencia y **nunca**
   como criterio: elegir por ella es el sobreajuste que documento
   `docs/2026-08-29_frente4_pedestal_y_overfit.md`.

2. **Cada configuracion recalibra su propio umbral.** El 3.5 sigma de produccion
   esta calibrado sobre las nulas de los radios actuales; medir completitud de
   otra configuracion contra el seria compararla con una vara ajustada a su
   rival. Es la leccion de `docs/2026-08-29_handoff.md` §7 —«la referencia de un
   test es tan parte del test como su umbral»—, fallada dos veces en dos dias.
   El umbral sale de las nulas de la propia configuracion a FPR fija. Se publica
   ademas la columna a umbral fijo, para que se vea cuanto de la diferencia es el
   umbral y cuanto el estimador.

3. **La escala de flujo inyectado se calibra UNA vez y no se mueve.**
   `_derive_injection_sigma` convierte `input_snr` en flujo usando el ruido del
   metodo de referencia —que es `psffit`, cuyo ruido depende de los radios—. Si
   se dejara recalibrar por configuracion, `input_snr = 1.0` seria un flujo
   fisico distinto en cada una y los `f50` no serian comparables, sin que nada lo
   delatara. Se calibra con los radios de PRODUCCION y ese `sigma_flux` se
   reutiliza en todas: la escalera de flujo es comun y lo unico que cambia es el
   estimador.

**Fuera de muestra.** Las posiciones de control se parten en dos mitades
disjuntas y deterministas (A y B). Se elige la configuracion por `f50` en A y se
reporta su `f50` en B, cada mitad con su propio umbral y su propia
estandarizacion. El numero que se cita es el de B, que no participo en la
eleccion. Aviso que va en la tabla: la resolucion en FPR de una mitad es
1/n_nulos (~0.06 en ROXs 12 b, ~0.09 en ROXs 42B b), asi que el umbral de cada
mitad lo fijan pocos eventos.

**No escribe nada en `runs/`.** Usa `compute_stage_h04_products`, que calcula sin
publicar, y `h04_x03_overrides` para mover los radios sin tocar el config del
run. Los productos de E4 en disco —de los que depende el tag `fondecyt-fig2-v3`—
no se rozan. Los resultados se anaden a CSV segun terminan, asi que un fallo en
la hora 6 deja en disco lo de las cinco anteriores.

Uso:

    python -u scripts/psffit_radius_sweep.py --runs ROXs12b_realigned,ROXs42Bb_realigned \
        --salida $MUSE_WORK/psffit_radius_sweep
"""

from __future__ import annotations

import argparse
import copy
import csv
import datetime as _dt
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from detection_threshold_by_method import umbral_para_fpr, resumen_nulo  # noqa: E402

from musepipe.stages.stage_h04_extractors import build_production_extractors  # noqa: E402
from musepipe.stages.stage_h04_injection import (  # noqa: E402
    _derive_injection_sigma,
    _load_injection_psf_model,
    _load_psf_model,
    _load_stage02_cube,
    apply_snr_standardization,
    compute_stage_h04_products,
    resolve_h04_positions,
    stage_h04_config_from_run,
    stage_h04_paths,
)

#: Radios a barrer. Produccion (20, 12) va PRIMERA: si la noche se corta, lo
#: primero que hay en disco es la referencia contra la que se compara todo.
R_STAR = (20.0, 16.0, 28.0)
R_COMP = (12.0, 10.0, 8.0, 16.0)

#: Rejilla de SNR inyectada. Recortada frente a los 12 niveles de produccion
#: —`f50` solo necesita bracketear el 50 % y cada nivel cuesta lineal— pero
#: SESGADA A VALORES BAJOS: la prueba de humo del 2026-08-30 midio completitud
#: 1.0 ya en `input_snr = 1.0` con el umbral calibrado, asi que el cruce del 50 %
#: vive por debajo de 1 y una rejilla que empiece ahi no lo ve. El 0.0 no es
#: opcional: son las nulas de las que sale el umbral.
REJILLA_SNR = (0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0)

#: FPR a la que se calibra el umbral de cada configuracion.
FPR = 0.05

#: Umbral fijo de produccion, para la columna de contraste.
UMBRAL_FIJO = 3.5

CAMPOS_RESUMEN = [
    "run", "r_star_px", "r_comp_px", "mitad", "modo_umbral", "umbral",
    "fpr_pedida", "fpr_conseguida", "n_nulos", "n_posiciones", "f50",
    "nula_media", "nula_sigma", "real_snr_std", "sigma_flux", "segundos", "utc",
]
#: Filas crudas, una por caso. Se vuelcan SIEMPRE: sin ellas, re-particionar la
#: muestra o poner barras de error a `f50` obliga a recomputar el barrido entero
#: (4 h medidas el 2026-08-30). Son exactamente las columnas que necesitan
#: `apply_snr_standardization` y `injection_null_reference` para rehacer el
#: analisis desde cero, mas la SNR formal como referencia.
CAMPOS_FILAS = [
    "run", "r_star_px", "r_comp_px", "injection_id", "variant", "method",
    "position_label", "template_factor", "continuum_mode", "input_snr",
    "recovered_flux", "recovered_sigma", "recovered_snr",
]
CAMPOS_CURVA = [
    "run", "r_star_px", "r_comp_px", "mitad", "modo_umbral", "input_snr",
    "n", "n_detectadas", "completitud",
]


def _particion(etiquetas):
    """Mitades disjuntas y deterministas de las posiciones de control.

    Alternas sobre la lista ordenada, no un `random.shuffle` con semilla: asi la
    particion no depende de la version de numpy ni del orden en que E4 devolvio
    las filas, y dos ejecuciones distintas del barrido son comparables.
    """
    orden = sorted(set(etiquetas))
    return {"A": set(orden[0::2]), "B": set(orden[1::2])}


def _f50(niveles, completitudes):
    """Interpolacion lineal del cruce del 50 %.

    `nan` si la curva no llega al 50 % dentro de la rejilla: eso NO es un f50
    grande, es una configuracion cuya completitud no se ha medido donde importa,
    y confundirlos ordenaria el barrido por un valor inventado.
    """
    pares = [(float(s), float(c)) for s, c in zip(niveles, completitudes)
             if math.isfinite(s) and math.isfinite(c) and s > 0]
    pares.sort()
    previo = None
    for snr, comp in pares:
        if comp >= 0.5:
            if previo is None:
                # Ya se detecta el 50 % en el nivel MAS BAJO medido: el cruce
                # esta por debajo de la rejilla y no se ha medido. Devolver ese
                # nivel lo haria pasar por una medida y empataria configuraciones
                # que no se han comparado. Es el mismo fallo que este script
                # critica en su §1, cometido en la interpolacion.
                return float("nan")
            s0, c0 = previo
            if comp == c0:
                return snr
            return s0 + (0.5 - c0) * (snr - s0) / (comp - c0)
        previo = (snr, comp)
    return float("nan")


def _filas_de(rows, etiquetas):
    return [r for r in rows if str(r["position_label"]) in etiquetas]


def _nominales(rows):
    return [r for r in rows
            if str(r["variant"]) == "nominal"
            and np.isclose(float(r["template_factor"]), 1.0)
            and str(r["continuum_mode"]) == "none"]


def _valor_snr(row):
    """La SNR estandarizada, y SOLO esa.

    Sin fallback a `recovered_snr`: son dos escalas distintas —una lleva restada
    la mediana de las nulas de su estrato y la otra no— y sustituir una por otra
    cuando falta mezclaria poblaciones dentro de la misma curva de completitud
    sin que nada lo dijera. Una fila sin estandarizar sale como `nan`, se excluye
    del recuento y aparece en la diferencia entre `n` y el numero de casos.
    """
    v = row.get("recovered_snr_std")
    return float("nan") if v is None else float(v)


def analiza_mitad(rows, etiquetas, *, fpr, umbral_fijo):
    """Umbral, curva de completitud y `f50` de una mitad, con SUS nulas.

    La estandarizacion se aplica a las filas de esta mitad y solo a ellas: si se
    heredara la de la muestra completa, las nulas de la otra mitad entrarian en
    la referencia y la separacion dejaria de ser tal.
    """
    subset = _nominales(_filas_de(copy.deepcopy(rows), etiquetas))
    if not subset:
        return None
    subset = apply_snr_standardization(subset, mode="injection_nulls",
                                       threshold_snr=umbral_fijo)
    nulas = [_valor_snr(r) for r in subset if float(r["input_snr"]) == 0.0]
    calibrado = umbral_para_fpr(nulas, fpr)
    niveles = sorted({float(r["input_snr"]) for r in subset} - {0.0})

    salida = []
    for modo, umbral, pedida, conseguida in (
        ("fpr_calibrado", calibrado["umbral"], fpr, calibrado["fpr_conseguida"]),
        ("fijo", float(umbral_fijo), float("nan"), float("nan")),
    ):
        curva = []
        for nivel in niveles:
            vals = [_valor_snr(r) for r in subset
                    if np.isclose(float(r["input_snr"]), nivel)]
            vals = [v for v in vals if math.isfinite(v)]
            n_det = sum(1 for v in vals if v >= umbral) if math.isfinite(umbral) else 0
            curva.append({"input_snr": nivel, "n": len(vals), "n_detectadas": n_det,
                          "completitud": (n_det / len(vals)) if vals else float("nan")})
        salida.append({
            "modo_umbral": modo, "umbral": umbral,
            "fpr_pedida": pedida, "fpr_conseguida": conseguida,
            "n_nulos": len([v for v in nulas if math.isfinite(v)]),
            "nula": resumen_nulo(nulas),
            "curva": curva,
            "f50": _f50([c["input_snr"] for c in curva],
                        [c["completitud"] for c in curva]),
        })
    return salida


def snr_real(rows):
    """La SNR del companero real, estandarizada contra TODOS los controles.

    Columna de referencia, no criterio. Se calcula sobre la muestra completa
    porque no participa en la eleccion: es lo que se mira DESPUES, para ver si
    la configuracion elegida por completitud tambien mueve la medida real.
    """
    subset = _nominales(copy.deepcopy(rows))
    subset = apply_snr_standardization(subset, mode="injection_nulls",
                                       threshold_snr=UMBRAL_FIJO)
    reales = [_valor_snr(r) for r in subset
              if str(r["position_label"]) == "real" and float(r["input_snr"]) == 0.0]
    finitos = [v for v in reales if math.isfinite(v)]
    return float(np.mean(finitos)) if finitos else float("nan")


def barre_run(run, *, salida, fpr, umbral_fijo, combinaciones, rejilla, escritores):
    base = stage_h04_config_from_run(run, project_root=str(ROOT))
    paths = stage_h04_paths(base["run_id"], project_root=base.get("project_root"))

    cube, wave, _ruta = _load_stage02_cube(paths, base)
    cube = np.asarray(cube, dtype=np.float64)
    if cube.ndim == 4:
        if cube.shape[0] != 1:
            raise RuntimeError(f"{run}: se esperaba un solo cubo combinado, no {cube.shape[0]}.")
        cube = cube[0]
    psf_model, psf_source = _load_psf_model(paths, base)
    inj_model, _src, _dif = _load_injection_psf_model(paths, base, psf_model, psf_source)

    # La escala de flujo, UNA vez y con los radios de produccion (ver docstring §3).
    posiciones = resolve_h04_positions(base, paths)
    cfg_prod = {**base, "h04_methods": ["psffit"]}
    extractores_prod = build_production_extractors(
        cfg_prod, paths, wave_A=wave, psf_model=psf_model, base_cube=cube)
    sigma_flux, calib = _derive_injection_sigma(
        {**cfg_prod, "h04_real_position_yx": [posiciones[0]["y"], posiciones[0]["x"]]},
        paths, extractores_prod, cube, wave, psf_model, injection_psf_model=inj_model)
    print(f"[{run}] sigma_flux fijo = {sigma_flux:.6g} "
          f"(throughput_ref={calib['throughput_ref']:.4f}, radios de produccion)", flush=True)

    for i, (rs, rc) in enumerate(combinaciones, start=1):
        t0 = time.perf_counter()
        cfg = {
            **base,
            "h04_methods": ["psffit"],
            "h04_snr_grid": list(rejilla),
            "h04_template_width_factors": [1.0],
            "h04_continuum_modes": ["none"],
            "h04_psf_perturb_snr_grid": [],
            "h04_psf_perturb_scales": [],
            "h04_injection_flux_sigma": float(sigma_flux),
            "h04_x03_overrides": {"x03_star_radius_px": float(rs),
                                  "x03_comp_radius_px": float(rc)},
            # La estandarizacion se hace por mitad, aqui abajo. Si corriera
            # tambien dentro de E4 usaria las nulas de la muestra COMPLETA y la
            # separacion A/B seria decorativa.
            "h04_snr_standardization": "none",
            "h04_require_historic_regression": False,
            "h04_allow_long_run": True,
        }
        try:
            extractores = build_production_extractors(
                cfg, paths, wave_A=wave, psf_model=psf_model, base_cube=cube)
            producto = compute_stage_h04_products(
                cfg, paths, extractors=extractores, base_cube=cube,
                wavelengths_A=wave, psf_model=psf_model)
            rows = producto.rows
            if not rows:
                raise RuntimeError(f"E4 no devolvio filas: {producto.qc.get('status_detail', '')}")
        except Exception as exc:  # una configuracion que revienta no tumba el barrido
            print(f"[{run}] ({rs:g},{rc:g}) {i}/{len(combinaciones)} FALLO: {exc}", flush=True)
            continue

        for fila in rows:
            escritores["filas"].writerow({
                "run": run, "r_star_px": rs, "r_comp_px": rc,
                **{k: fila.get(k) for k in CAMPOS_FILAS[3:]},
            })

        controles = [str(r["position_label"]) for r in rows
                     if str(r["position_label"]).startswith("control")]
        mitades = _particion(controles)
        real = snr_real(rows)
        segundos = time.perf_counter() - t0

        for mitad, etiquetas in mitades.items():
            bloques = analiza_mitad(rows, etiquetas, fpr=fpr, umbral_fijo=umbral_fijo)
            if bloques is None:
                continue
            for bloque in bloques:
                escritores["resumen"].writerow({
                    "run": run, "r_star_px": rs, "r_comp_px": rc, "mitad": mitad,
                    "modo_umbral": bloque["modo_umbral"], "umbral": bloque["umbral"],
                    "fpr_pedida": bloque["fpr_pedida"],
                    "fpr_conseguida": bloque["fpr_conseguida"],
                    "n_nulos": bloque["n_nulos"], "n_posiciones": len(etiquetas),
                    "f50": bloque["f50"],
                    "nula_media": bloque["nula"].get("media", float("nan")),
                    "nula_sigma": bloque["nula"].get("sigma", float("nan")),
                    "real_snr_std": real, "sigma_flux": sigma_flux,
                    "segundos": round(segundos, 1),
                    "utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
                })
                for punto in bloque["curva"]:
                    escritores["curva"].writerow({
                        "run": run, "r_star_px": rs, "r_comp_px": rc, "mitad": mitad,
                        "modo_umbral": bloque["modo_umbral"], **punto,
                    })
        for fh in escritores["ficheros"]:
            fh.flush()

        f50s = {}
        for mitad, etiquetas in mitades.items():
            bloques = analiza_mitad(rows, etiquetas, fpr=fpr, umbral_fijo=umbral_fijo)
            if bloques:
                f50s[mitad] = bloques[0]["f50"]
        print(f"[{run}] ({rs:g},{rc:g}) {i}/{len(combinaciones)} "
              f"f50_A={f50s.get('A', float('nan')):.3f} f50_B={f50s.get('B', float('nan')):.3f} "
              f"real={real:+.2f} {segundos/60:.1f} min", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", required=True, help="lista separada por comas")
    ap.add_argument("--salida", required=True, help="directorio de resultados (NO dentro de runs/)")
    ap.add_argument("--fpr", type=float, default=FPR)
    ap.add_argument("--umbral-fijo", type=float, default=UMBRAL_FIJO)
    ap.add_argument("--r-star", default=",".join(f"{v:g}" for v in R_STAR))
    ap.add_argument("--r-comp", default=",".join(f"{v:g}" for v in R_COMP))
    ap.add_argument("--rejilla-snr", default=",".join(f"{v:g}" for v in REJILLA_SNR))
    ap.add_argument(
        "--pares",
        default=None,
        help=(
            "pares explicitos `r_star x r_comp`, p.ej. 20x12,20x8,16x12. Gana sobre "
            "--r-star/--r-comp. Existe porque `psffit` cuesta ~1 min por caso y una "
            "rejilla simetrica no cabe en una noche: se eligen las configuraciones "
            "que contestan la pregunta, y la de PRODUCCION va primera para que sea "
            "lo primero que hay en disco si la corrida se corta."
        ),
    )
    args = ap.parse_args(argv)

    salida = Path(args.salida).resolve()
    if "runs" in salida.parts:
        raise SystemExit(f"--salida no puede caer dentro de runs/: {salida}")
    salida.mkdir(parents=True, exist_ok=True)
    sello = _dt.datetime.now().strftime("%Y%m%dT%H%M%S")

    if args.pares:
        combinaciones = []
        for token in args.pares.split(","):
            token = token.strip()
            if not token:
                continue
            rs, _, rc = token.lower().partition("x")
            combinaciones.append((float(rs), float(rc)))
    else:
        r_star = tuple(float(v) for v in args.r_star.split(",") if v.strip())
        r_comp = tuple(float(v) for v in args.r_comp.split(",") if v.strip())
        combinaciones = [(rs, rc) for rs in r_star for rc in r_comp]
    if not combinaciones:
        raise SystemExit("no hay configuraciones que barrer.")
    rejilla = tuple(float(v) for v in args.rejilla_snr.split(",") if v.strip())
    if 0.0 not in rejilla:
        raise SystemExit("la rejilla necesita el 0.0: son las nulas de las que sale el umbral.")

    ruta_resumen = salida / f"radius_sweep_summary_{sello}.csv"
    ruta_curva = salida / f"radius_sweep_completeness_{sello}.csv"
    ruta_filas = salida / f"radius_sweep_rows_{sello}.csv"
    with ruta_resumen.open("w", newline="") as fr, \
         ruta_curva.open("w", newline="") as fc, \
         ruta_filas.open("w", newline="") as ff:
        escritores = {
            "resumen": csv.DictWriter(fr, fieldnames=CAMPOS_RESUMEN),
            "curva": csv.DictWriter(fc, fieldnames=CAMPOS_CURVA),
            "filas": csv.DictWriter(ff, fieldnames=CAMPOS_FILAS, extrasaction="ignore"),
            "ficheros": (fr, fc, ff),
        }
        for clave in ("resumen", "curva", "filas"):
            escritores[clave].writeheader()
        for fh in escritores["ficheros"]:
            fh.flush()
        for run in [r.strip() for r in args.runs.split(",") if r.strip()]:
            print(f"\n=== {run}: {len(combinaciones)} configuraciones, "
                  f"{len(rejilla)} niveles de SNR ===", flush=True)
            barre_run(run, salida=salida, fpr=args.fpr, umbral_fijo=args.umbral_fijo,
                      combinaciones=combinaciones, rejilla=rejilla, escritores=escritores)
    print(f"\nresumen: {ruta_resumen}\ncurvas:  {ruta_curva}\nfilas:   {ruta_filas}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
