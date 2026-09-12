"""D (2026-09-10): auditar la cola paramétrica de E1, que se lee a 6.9 sigma.

Por que. La deteccion de Halpha en ROXs 12 b se declara con
`h01_fap_estimator = parametric`: se ajusta una Gumbel a los 33 maximos nulos y
se evalua su cola en el z observado. En `aperture` eso son loc=1.616,
scale=0.727, mayor nulo 4.20, z observado 10.41 -> FAP 5.6e-6, con
`extrapolation_sd = 6.87` y CI95 [5.7e-8, 6.2e-5]. La decision de adoptarla esta
tomada y documentada (docs/2026-09-02_criterio_fap_opciones.md); lo que NO se ha
medido es si la puerta que la respalda puede fallar: `ks_p = 0.994` solo dice que
la Gumbel describe el CUERPO de 33 puntos, y el numero se lee en la cola.

Es el reverso del patron de las cinco puertas: una que solo puede PASAR.

Tres preguntas, cada una con su control:

  D1 RUIDO DEL AJUSTE. Con la familia CORRECTA, ¿cuanto puede saber un ajuste de
     33 puntos a esa distancia? Monte Carlo: 33 sacadas de la Gumbel ajustada,
     re-ajustar, evaluar en z. Si el veredicto (FAP < 0.01) sobrevive en casi
     todas las repeticiones, el VEREDICTO es robusto aunque el NUMERO no lo sea.

  D2 POTENCIA DE LA PUERTA. Con n=33, ¿con que frecuencia el KS rechaza una
     familia EQUIVOCADA? Se simulan colas GEV con forma xi != 0 casadas al mismo
     cuerpo, se les ajusta una Gumbel y se mira `ks_p`. Si la tasa de rechazo se
     queda en el alfa nominal, `ks_p = 0.994` no es evidencia de nada: es una
     puerta sin potencia.

  D3 CONSECUENCIA. Bajo esas alternativas, ¿que FAP VERDADERA hay en z, y cuanto
     se equivoca la Gumbel? La pregunta que decide es si la FAP verdadera sigue
     por debajo de 0.01, no cuantas decadas se mueve.

  D4 LO QUE EL DATO SI CONSTRAINE. Ajuste GEV de 3 parametros a los 33 nulos
     reales, con xi por bootstrap, y el rango de FAP que implica.

Solo lectura: no toca ningun run.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CRITERION = 0.01
KS_ALPHA = 0.05
# xi en la convencion EVT estandar. scipy usa c = -xi: se VERIFICA abajo, no se
# supone -- equivocar el signo invertiria cola pesada y cola acotada.
XI_GRID = (-0.20, -0.10, -0.05, 0.05, 0.10, 0.20, 0.30)


def _check_scipy_shape_convention():
    """xi > 0 (Frechet, cola pesada) <=> soporte no acotado por arriba."""
    heavy = stats.genextreme(c=-0.30, loc=0.0, scale=1.0)
    light = stats.genextreme(c=+0.30, loc=0.0, scale=1.0)
    assert not np.isfinite(heavy.ppf(1.0 - 1e-12)) or heavy.ppf(1.0 - 1e-12) > 50.0, \
        "c=-xi: c<0 deberia dar cola pesada"
    assert np.isfinite(light.ppf(1.0)) and light.ppf(1.0) < 10.0, \
        "c=+xi: c>0 deberia dar soporte acotado por arriba"
    return {"scipy_c": "c = -xi", "verified": True,
            "heavy_xi_+0.30_upper": "unbounded",
            "light_xi_-0.30_upper": float(light.ppf(1.0))}


def _gev_matched(xi, mean, sd):
    """GEV con forma xi casada a la media y sd de la nula real.

    Casar el CUERPO es lo que hace la prueba honesta: si las alternativas
    tuvieran otro cuerpo, el KS las rechazaria por el cuerpo y no por la cola,
    que es lo que se quiere poner a prueba.
    """
    c = -float(xi)
    base = stats.genextreme(c=c, loc=0.0, scale=1.0)
    m0, s0 = base.mean(), base.std()
    if not (np.isfinite(m0) and np.isfinite(s0) and s0 > 0):
        return None
    scale = sd / s0
    loc = mean - m0 * scale
    return stats.genextreme(c=c, loc=loc, scale=scale)


def audit_method(name, nulls, obs, *, n_boot, rng):
    nulls = np.asarray(nulls, dtype=np.float64)
    nulls = nulls[np.isfinite(nulls)]
    n = nulls.size
    loc, scale = stats.gumbel_r.fit(nulls)
    fap = float(stats.gumbel_r.sf(obs, loc, scale))
    ks_p = float(stats.kstest(nulls, "gumbel_r", args=(loc, scale)).pvalue)
    sd = float(np.std(nulls, ddof=1))
    mean = float(np.mean(nulls))
    extrap_sd = float((obs - nulls.max()) / sd)
    out = {"method": name, "n_null": int(n), "obs_z": float(obs),
           "max_null": float(nulls.max()), "null_mean": mean, "null_sd": sd,
           "gumbel_loc": float(loc), "gumbel_scale": float(scale),
           "fap_reported": fap, "ks_p": ks_p, "extrapolation_sd": extrap_sd}

    # --- D1: ruido del ajuste con la familia correcta --------------------
    sims = rng.gumbel(loc, scale, size=(n_boot, n))
    faps = np.empty(n_boot)
    for i in range(n_boot):
        l_i, s_i = stats.gumbel_r.fit(sims[i])
        faps[i] = stats.gumbel_r.sf(obs, l_i, s_i)
    lo = np.log10(np.clip(faps, 1e-300, None))
    out["D1_fit_noise"] = {
        "n_rep": int(n_boot),
        "log10_fap_p2.5": float(np.percentile(lo, 2.5)),
        "log10_fap_p50": float(np.percentile(lo, 50)),
        "log10_fap_p97.5": float(np.percentile(lo, 97.5)),
        "decades_span_95": float(np.percentile(lo, 97.5) - np.percentile(lo, 2.5)),
        "frac_below_criterion": float(np.mean(faps < CRITERION)),
        "note": ("Con la familia correcta: cuanto mueve el numero el ruido de 33 "
                 "puntos, y si el VEREDICTO (FAP<0.01) aguanta."),
    }

    # --- D2/D3: potencia del KS y consecuencia, por xi -------------------
    per_xi = {}
    for xi in XI_GRID:
        dist = _gev_matched(xi, mean, sd)
        if dist is None:
            continue
        true_sf = float(dist.sf(obs))
        s = dist.rvs(size=(n_boot, n), random_state=rng)
        ks = np.empty(n_boot)
        gf = np.empty(n_boot)
        for i in range(n_boot):
            l_i, sc_i = stats.gumbel_r.fit(s[i])
            ks[i] = stats.kstest(s[i], "gumbel_r", args=(l_i, sc_i)).pvalue
            gf[i] = stats.gumbel_r.sf(obs, l_i, sc_i)
        # xi < 0 da soporte acotado por arriba: si z lo supera, la FAP
        # verdadera es 0 EXACTO -- esa alternativa no puede producir la
        # observacion en absoluto. Declararlo, no convertirlo en "295 decadas".
        upper = float(dist.ppf(1.0))
        beyond = bool(np.isfinite(upper) and obs > upper)
        per_xi[f"{xi:+.2f}"] = {
            "xi": float(xi),
            "true_fap_at_obs": true_sf,
            "obs_beyond_bounded_support": beyond,
            "upper_endpoint": upper if np.isfinite(upper) else None,
            "ks_rejection_rate_at_0.05": float(np.mean(ks < KS_ALPHA)),
            "ks_p_median": float(np.median(ks)),
            "gumbel_fap_median": float(np.median(gf)),
            "decades_error_median": (
                None if beyond or true_sf <= 0.0 else
                float(np.log10(max(np.median(gf), 1e-300)) - np.log10(true_sf))),
            "true_fap_below_criterion": bool(true_sf < CRITERION),
        }
    out["D2_D3_by_xi"] = per_xi
    if per_xi:
        peor = max(per_xi.values(), key=lambda v: v["true_fap_at_obs"])
        out["worst_case_over_xi"] = {
            "xi": peor["xi"], "true_fap_at_obs": peor["true_fap_at_obs"],
            "still_below_criterion": bool(peor["true_fap_at_obs"] < CRITERION)}
        out["ks_gate_max_rejection_rate"] = float(
            max(v["ks_rejection_rate_at_0.05"] for v in per_xi.values()))

    # --- D4: lo que los 33 puntos reales constrainen ---------------------
    try:
        c_hat, l_hat, s_hat = stats.genextreme.fit(nulls)
        xi_hat = -float(c_hat)
        boot_xi, boot_fap = [], []
        for _ in range(min(n_boot, 2000)):
            res = rng.choice(nulls, size=n, replace=True)
            try:
                c_b, l_b, s_b = stats.genextreme.fit(res)
            except Exception:
                continue
            boot_xi.append(-float(c_b))
            boot_fap.append(float(stats.genextreme.sf(obs, c_b, l_b, s_b)))
        out["D4_gev_on_real_nulls"] = {
            "xi_hat": xi_hat,
            "xi_ci95": [float(np.percentile(boot_xi, 2.5)), float(np.percentile(boot_xi, 97.5))]
            if boot_xi else None,
            "xi_consistent_with_gumbel": bool(
                boot_xi and np.percentile(boot_xi, 2.5) <= 0.0 <= np.percentile(boot_xi, 97.5)),
            "fap_gev_point": float(stats.genextreme.sf(obs, c_hat, l_hat, s_hat)),
            "fap_ci95": [float(np.percentile(boot_fap, 2.5)), float(np.percentile(boot_fap, 97.5))]
            if boot_fap else None,
            "frac_boot_below_criterion": float(np.mean(np.asarray(boot_fap) < CRITERION))
            if boot_fap else None,
            "note": ("La spec descarto la GEV porque con n~33 xi no esta constrenido "
                     "y su extremo finito da p=0 exacto. Aqui NO se propone cambiarla: "
                     "se mide cuanta incertidumbre de cola implica ese hecho."),
        }
    except Exception as exc:  # pragma: no cover
        out["D4_gev_on_real_nulls"] = {"error": repr(exc)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="run del que se leen la nula y el z observado")
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--n-boot", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=20260910)
    ap.add_argument("--methods", default="aperture,psffit,optimal_ls,optimal_psfsub,sgf,lpm")
    args = ap.parse_args()

    conv = _check_scipy_shape_convention()
    print("convencion de forma verificada:", conv)

    stages = ROOT / "runs" / args.run / "stages"
    tables = ROOT / "runs" / args.run / "tables"
    qc = json.loads((stages / "stage_h01_qc.json").read_text())
    nulls = np.load(stages / "stage_h01_null_maxima.npz", allow_pickle=True)
    import csv as _csv
    obs_by_method = {}
    with open(tables / "halpha_detection_by_method.csv", newline="") as fh:
        for row in _csv.DictReader(fh):
            obs_by_method[row["method"]] = float(row["matched_z"])

    rng = np.random.default_rng(args.seed)
    results = []
    for m in args.methods.split(","):
        m = m.strip()
        key = f"{m}_null_maxima"
        if key not in nulls or m not in obs_by_method:
            print(f"  (sin nula u observado para {m}, salto)")
            continue
        r = audit_method(m, nulls[key], obs_by_method[m], n_boot=args.n_boot, rng=rng)
        results.append(r)
        d1 = r["D1_fit_noise"]
        print(f"\n== {m}  z={r['obs_z']:.2f}  mayor nulo={r['max_null']:.2f}  "
              f"extrapola {r['extrapolation_sd']:.2f} sd")
        print(f"   FAP del run {r['fap_reported']:.3e}   ks_p={r['ks_p']:.3f}")
        print(f"   D1 ruido del ajuste: log10 FAP CI95 "
              f"[{d1['log10_fap_p2.5']:.2f}, {d1['log10_fap_p97.5']:.2f}] "
              f"= {d1['decades_span_95']:.1f} decadas; "
              f"veredicto aguanta en {100*d1['frac_below_criterion']:.1f} %")
        print("   D2/D3 por xi (potencia del KS y FAP verdadera):")
        for k, v in r["D2_D3_by_xi"].items():
            dec = ("  fuera del soporte" if v["obs_beyond_bounded_support"]
                   else f"  ({v['decades_error_median']:+.1f} dec)"
                   if v["decades_error_median"] is not None else "")
            print(f"     xi={k}  KS rechaza {100*v['ks_rejection_rate_at_0.05']:5.1f} %  "
                  f"FAP verdadera {v['true_fap_at_obs']:.2e}  "
                  f"Gumbel dice {v['gumbel_fap_median']:.2e}{dec}  "
                  f"<0.01: {v['true_fap_below_criterion']}")
        w = r.get("worst_case_over_xi")
        if w:
            print(f"   PEOR CASO sobre la rejilla de xi: FAP verdadera "
                  f"{w['true_fap_at_obs']:.2e} (xi={w['xi']:+.2f})  "
                  f"sigue < {CRITERION}: {w['still_below_criterion']}")
            print(f"   potencia maxima del KS contra familia equivocada: "
                  f"{100*r['ks_gate_max_rejection_rate']:.1f} % "
                  f"(nominal {100*KS_ALPHA:.0f} %)")
        g = r.get("D4_gev_on_real_nulls", {})
        if "xi_hat" in g:
            print(f"   D4 GEV al dato real: xi={g['xi_hat']:+.3f} CI95 {g['xi_ci95']}  "
                  f"compatible con Gumbel: {g['xi_consistent_with_gumbel']}")

    payload = {"stage": "D_auditoria_cola_parametrica", "run": args.run,
               "criterion_fap_lt": CRITERION, "ks_alpha": KS_ALPHA,
               "scipy_shape_convention": conv,
               "n_boot": args.n_boot, "seed": args.seed,
               "criterion_from_qc": qc["criterion"], "methods": results}
    Path(args.out_json).write_text(json.dumps(payload, indent=1, ensure_ascii=False))
    print(f"\n-> {args.out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
