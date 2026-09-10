#!/usr/bin/env python
"""Prueba la prediccion de `docs/2026-09-07_binaria_psffit_sesgo_medido.md` §6.

La prediccion: si el sesgo de la binaria entra en `psffit` con signo NEGATIVO y
en `optimal_psfsub` con signo POSITIVO, entonces **infla la dispersion
inter-metodo**, y corregirlo deberia BAJAR el `divergent_continuum` de D1.

Aqui se corre D1 dos veces sobre los MISMOS controles y la misma estadistica:

  base     los productos calibrados tal cual estan en el run
  corregido `psffit` y `optimal_psfsub` con el sesgo medido por las sondas
           quitado (variante `ligada` menos `base`), el resto sin tocar

**No escribe en `runs/`.** Reproduce primero el veredicto del QC en disco: si no
sale igual, no compara nada y avisa.

El sesgo se traslada de la sonda (sin calibrar) al producto (calibrado) de forma
ADITIVA y con la calibracion suavizada:

    corregido_cal = base_cal - delta_sonda * K(lambda),   K = mediana movil de
                                                          base_cal / base_sonda

Aditivo y no multiplicativo porque lo medido ES un pedestal, y porque el cociente
de dos espectros ruidosos inyectaria ruido donde el compañero es debil, que es
justo donde el efecto es grande.

**Lo que este test NO hace** (y por eso su resultado es indicativo, no un
veredicto): no recalcula el throughput de E4 con la extraccion cambiada, no toca
los controles, y el delta de psfsub esta medido sobre el cubo de B2 mientras el
producto del run sale del cubo de C1b.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
import sys  # noqa: E402

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from musepipe.io import read_json  # noqa: E402
from musepipe.stages.stage_x10_compare import (  # noqa: E402
    DEFAULT_CONTROL_GATE_ALPHA,
    DEFAULT_LINE_CONTINUUM_WINDOW_A,
    DEFAULT_P_DIVERGENT,
    DEFAULT_P_STRONG,
    DEFAULT_SCALE_GATE_SIGMA,
    DEFAULT_SGF_PREDICTOR_MAX,
    _control_paths_from_config,
    _product_paths_from_config,
    compare_methods,
    load_control_spectra,
    load_g1_inputs,
    load_method_products,
    stage_x10_config_from_run,
    stage_x10_paths,
)


def mediana_movil(x, ancho):
    x = np.asarray(x, dtype=np.float64)
    n = x.size
    out = np.full(n, np.nan)
    h = int(ancho) // 2
    for i in range(n):
        lo, hi = max(0, i - h), min(n, i + h + 1)
        tramo = x[lo:hi]
        tramo = tramo[np.isfinite(tramo)]
        if tramo.size:
            out[i] = np.median(tramo)
    return out


def delta_calibrado(producto, npz, clave_base, clave_var, *, ancho=201):
    """El sesgo de la sonda, llevado a la escala del producto calibrado."""
    w_p = np.asarray(producto.wave_A, dtype=np.float64)
    w_s = np.asarray(npz["wave_A"], dtype=np.float64)
    base_s = np.interp(w_p, w_s, npz[clave_base])
    var_s = np.interp(w_p, w_s, npz[clave_var])
    delta_s = var_s - base_s
    with np.errstate(divide="ignore", invalid="ignore"):
        k = np.asarray(producto.flux, dtype=np.float64) / base_s
    k = mediana_movil(k, ancho)
    k[~np.isfinite(k)] = np.nanmedian(k[np.isfinite(k)]) if np.any(np.isfinite(k)) else 1.0
    return delta_s * k, k


def corre_d1(productos, controles, cfg, paths, g1):
    sgf_qc = Path(cfg.get("x10_spec_sgf_qc_json", paths["spec_sgf_qc_json"]))
    ctx = {
        "companion_continuum_is_science": cfg.get("companion_continuum_is_science", True),
        "sgf_predictor_max": cfg.get("x10_sgf_predictor_max", DEFAULT_SGF_PREDICTOR_MAX),
        "sgf_predictors": ((read_json(sgf_qc) or {}).get("self_subtraction_predictor")
                           if sgf_qc.exists() else None),
    }
    primary = cfg.get("x10_primary_pairs", "from_g1")
    primary = None if (not primary or primary == "from_g1") else tuple(tuple(p) for p in primary)
    rows, _crows, qc = compare_methods(
        productos, controles,
        sigma_smooth_channels=int(cfg.get("x10_sigma_smooth_channels", 5)),
        g1_inputs=g1, primary_pairs=primary,
        p_divergent=float(cfg.get("x10_p_divergent", DEFAULT_P_DIVERGENT)),
        p_strong=float(cfg.get("x10_p_strong", DEFAULT_P_STRONG)),
        scale_gate_sigma=float(cfg.get("x10_scale_gate_sigma", DEFAULT_SCALE_GATE_SIGMA)),
        gate_alpha=float(cfg.get("x10_control_gate_alpha", DEFAULT_CONTROL_GATE_ALPHA)),
        line_continuum_window_A=float(cfg.get("x10_line_continuum_window_A",
                                              DEFAULT_LINE_CONTINUUM_WINDOW_A)),
        recommendation_context=ctx,
    )
    return qc, rows


def resumen_continuo(qc, rows):
    """Veredicto por par y el t de cada banda de continuo (de las filas, no del QC)."""
    out = {}
    for pid, res in (qc.get("verdict_by_pair") or {}).items():
        cont = res.get("continuum")
        if not cont:
            continue
        out[pid] = cont.get("verdict")
    tes = {}
    for row in rows or []:
        if row.get("band_kind") == "continuum" and row.get("kind", "object") != "control":
            tes.setdefault(row["pair"], {})[row["band"]] = (row.get("t_stat"), row.get("p_value"))
    return out, tes


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--project-root", default=None)
    ap.add_argument("--psffit-npz", required=True)
    ap.add_argument("--psfsub-npz", required=True)
    ap.add_argument("--variante", default="ligada", choices=("ligada", "libre"),
                    help="que punta de la horquilla se corrige (ligada = cota superior)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    cfg = stage_x10_config_from_run(args.run_id, project_root=args.project_root)
    root = Path(cfg["project_root"])
    paths = stage_x10_paths(cfg["run_id"], root)
    productos = load_method_products(_product_paths_from_config(cfg, paths))
    controles = load_control_spectra(_control_paths_from_config(cfg, paths))
    g1 = load_g1_inputs(cfg, paths)

    qc_base, filas_base = corre_d1(productos, controles, cfg, paths, g1)
    en_disco = read_json(paths["stage_x10_qc_json"]) if paths["stage_x10_qc_json"].exists() else {}
    if en_disco and en_disco.get("verdict") != qc_base.get("verdict"):
        raise SystemExit(
            f"No reproduzco el QC del run ({en_disco.get('verdict')!r} en disco contra "
            f"{qc_base.get('verdict')!r} ahora): el test no vale, mira por que antes de seguir."
        )

    sondas = {"psffit": (np.load(args.psffit_npz), "comp_base", f"comp_{args.variante}"),
              "optimal_psfsub": (np.load(args.psfsub_npz), "base", args.variante)}
    corregidos = dict(productos)
    aplicado = {}
    for metodo, (npz, kb, kv) in sondas.items():
        prod = productos[metodo]
        delta, k = delta_calibrado(prod, npz, kb, kv)
        # `SpectrumProduct` es un dataclass congelado: se reemplaza, no se muta.
        # corregido = base + (ligada - base): el espectro que el modelo de DOS
        # componentes habria entregado. El error NO se toca: el sesgo es
        # sistematico, no ruido.
        corregidos[metodo] = replace(prod, flux=np.asarray(prod.flux, dtype=np.float64) + delta)
        w = np.asarray(prod.wave_A, dtype=np.float64)
        banda = (w >= 8000) & (w < 8500)
        aplicado[metodo] = {
            "delta_rel_pct_8000_8500": float(100 * np.nansum(delta[banda])
                                             / np.nansum(np.asarray(prod.flux)[banda])),
            "k_mediana": float(np.nanmedian(k)),
        }
    qc_corr, filas_corr = corre_d1(corregidos, controles, cfg, paths, g1)

    vb, tb = resumen_continuo(qc_base, filas_base)
    vc, tc = resumen_continuo(qc_corr, filas_corr)
    res = {
        "run_id": cfg["run_id"],
        "variante": args.variante,
        "sesgo_aplicado": aplicado,
        "veredicto": {"base": qc_base.get("verdict"), "corregido": qc_corr.get("verdict")},
        "por_par": {pid: {"base": vb.get(pid), "corregido": vc.get(pid)} for pid in vb},
        "t_continuo": {pid: {b: {"base": tb[pid][b][0], "corregido": tc.get(pid, {}).get(b, (None,))[0]}
                             for b in tb.get(pid, {})} for pid in tb},
    }
    # El resumen que responde a la prediccion: |t| mediano del continuo, por par
    # y global. Si el signo opuesto inflaba la dispersion, esto tiene que BAJAR.
    def _abst(t):
        v = [abs(x) for x in t if x is not None and np.isfinite(x)]
        return float(np.median(v)) if v else None
    por_par_t = {}
    for pid in tb:
        b = _abst([tb[pid][k][0] for k in tb[pid]])
        c = _abst([tc.get(pid, {}).get(k, (None,))[0] for k in tb[pid]])
        por_par_t[pid] = {"abs_t_base": b, "abs_t_corregido": c,
                          "cambio_pct": (None if not b else float(100 * (c - b) / b))}
    res["abs_t_continuo"] = por_par_t
    todos_b = _abst([tb[pid][k][0] for pid in tb for k in tb[pid]])
    todos_c = _abst([tc.get(pid, {}).get(k, (None,))[0] for pid in tb for k in tb[pid]])
    res["abs_t_continuo_global"] = {"base": todos_b, "corregido": todos_c,
                                    "cambio_pct": (None if not todos_b
                                                   else float(100 * (todos_c - todos_b) / todos_b))}
    print(json.dumps(res, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(res, indent=2), encoding="utf-8")
    return res


if __name__ == "__main__":
    main()
