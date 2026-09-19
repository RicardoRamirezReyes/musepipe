#!/usr/bin/env python3
"""SNR50 por metodo y objeto, con su p, sobre la MISMA corrida.

    python scripts/snr50_pares.py --runs RUN_A RUN_B [--etiquetas "12 b" "42B b"]

Que mide
--------
`input_snr` de E4 esta en sigmas del filtro adaptado EN LA POSICION DEL
COMPANERO (ver `_derive_injection_sigma`), y la completitud se evalua en las
posiciones de CONTROL. El SNR50 es el `input_snr` al que se recupera la mitad de
las fuentes sinteticas, `recovered_snr_std >= umbral`.

Como
----
La unidad independiente del diseno es la POSICION, no la fila: las filas de una
posicion comparten ruido local, residuo de PSF y cielo, y los puntos de la
rejilla de SNR REUTILIZAN las mismas posiciones (spec E4 v3 §2). Por eso:

  * ajuste logistico de `complete` contra log10(input_snr), SNR50 = 10^(-b0/b1);
  * incertidumbre por bootstrap DE POSICIONES (se remuestrean posiciones
    enteras, con todas sus filas), no de filas;
  * p del par = dos colas sobre la distribucion bootstrap de la diferencia
    SNR50(42B b) - SNR50(12 b), que es lo que compara a los dos objetos.

Se informa ademas el SNR50 por interpolacion lineal de la curva empirica, como
control de que el resultado no lo pone la forma logistica.
"""
import argparse, csv, json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
B = 4000
SEMILLA = 20260917


def carga(run):
    """{metodo: {posicion: [(snr, completo), ...]}} de las filas nominales de control."""
    p = ROOT / "runs" / run / "tables" / "injection_throughput_by_method.csv"
    out = defaultdict(lambda: defaultdict(list))
    for r in csv.DictReader(open(p)):
        if r["variant"] != "nominal" or r["position_label"] == "real":
            continue
        s = float(r["input_snr"])
        if s <= 0:                      # log10 indefinido; la fila snr=0 es la nula
            continue
        out[r["method"]][r["position_label"]].append((s, r["complete"] == "True"))
    return {m: dict(v) for m, v in out.items()}


def _logistica(x, y, iteraciones=60):
    """IRLS. Devuelve (b0, b1) o None si no separa."""
    X = np.column_stack([np.ones_like(x), x])
    b = np.zeros(2)
    for _ in range(iteraciones):
        eta = np.clip(X @ b, -30, 30)
        mu = 1.0 / (1.0 + np.exp(-eta))
        W = np.clip(mu * (1 - mu), 1e-9, None)
        z = eta + (y - mu) / W
        try:
            b_new = np.linalg.solve((X * W[:, None]).T @ X, (X * W[:, None]).T @ z)
        except np.linalg.LinAlgError:
            return None
        if not np.all(np.isfinite(b_new)):
            return None
        if np.max(np.abs(b_new - b)) < 1e-10:
            b = b_new
            break
        b = b_new
    return (float(b[0]), float(b[1])) if b[1] > 0 else None


def snr50_logistico(datos):
    filas = [(s, c) for pos in datos.values() for s, c in pos]
    x = np.log10([s for s, _ in filas]); y = np.array([c for _, c in filas], float)
    if y.min() == y.max():
        return None
    ab = _logistica(x, y)
    return None if ab is None else float(10 ** (-ab[0] / ab[1]))


def snr50_empirico(datos):
    """Cruce del 50 % por interpolacion lineal de la curva de completitud."""
    agg = defaultdict(lambda: [0, 0])
    for pos in datos.values():
        for s, c in pos:
            agg[s][1] += 1; agg[s][0] += c
    xs = sorted(agg); ys = [agg[s][0] / agg[s][1] for s in xs]
    for i in range(1, len(xs)):
        if ys[i - 1] < 0.5 <= ys[i]:
            return xs[i - 1] + (0.5 - ys[i - 1]) * (xs[i] - xs[i - 1]) / (ys[i] - ys[i - 1])
    return None


def bootstrap(datos, rng, b=B):
    """Remuestrea POSICIONES enteras."""
    claves = list(datos)
    salida = np.full(b, np.nan)
    for i in range(b):
        elegidas = rng.integers(0, len(claves), len(claves))
        muestra = {f"{j}": datos[claves[k]] for j, k in enumerate(elegidas)}
        v = snr50_logistico(muestra)
        if v is not None:
            salida[i] = v
    return salida


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    # Los runs se piden SIEMPRE: `tests/test_no_hardcoded_target.py` prohibe el
    # literal, y la regla del proyecto es declarar el run, no heredarlo.
    ap.add_argument("--runs", nargs="+", required=True, metavar="RUN",
                    help="los dos (o mas) runs a comparar, uno por objeto")
    ap.add_argument("--etiquetas", nargs="+", default=None,
                    help="nombre legible de cada run, en el mismo orden")
    ap.add_argument("--salida", default="snr50_pares.json")
    a = ap.parse_args(argv)
    etiquetas = a.etiquetas or list(a.runs)
    if len(etiquetas) != len(a.runs):
        ap.error("hacen falta tantas etiquetas como runs")
    RUNS = dict(zip(etiquetas, a.runs))
    rng = np.random.default_rng(SEMILLA)
    datos = {ob: carga(run) for ob, run in RUNS.items()}
    metodos = sorted(set.intersection(*[set(d) for d in datos.values()]))
    res = {"runs": RUNS, "B": B, "semilla": SEMILLA, "metodos": {}}
    a_, b_ = etiquetas[0], etiquetas[1]
    print(f"{'metodo':16s} {a_:>22s} {b_:>22s} {'dif':>16s} {'p':>8s}")
    for m in metodos:
        fila = {}
        for ob in RUNS:
            d = datos[ob][m]
            fila[ob] = {"snr50": snr50_logistico(d), "snr50_empirico": snr50_empirico(d),
                        "n_posiciones": len(d),
                        "boot": bootstrap(d, rng)}
        boot_a, boot_b = fila[a_]["boot"], fila[b_]["boot"]
        n = min(np.count_nonzero(np.isfinite(boot_a)), np.count_nonzero(np.isfinite(boot_b)))
        dif = (boot_b[:n][np.isfinite(boot_b[:n])] - boot_a[:n][np.isfinite(boot_a[:n])]
               if n else np.array([]))
        p = float(2 * min((dif <= 0).mean(), (dif >= 0).mean())) if dif.size else float("nan")
        p = min(1.0, max(p, 1.0 / max(dif.size, 1)))     # suelo de resolucion del bootstrap
        ic = lambda v: (float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5)))
        res["metodos"][m] = {
            ob: {"snr50": fila[ob]["snr50"], "snr50_empirico": fila[ob]["snr50_empirico"],
                 "ic95": ic(fila[ob]["boot"]), "n_posiciones": fila[ob]["n_posiciones"],
                 "n_boot_validos": int(np.count_nonzero(np.isfinite(fila[ob]["boot"])))}
            for ob in RUNS}
        res["metodos"][m]["diferencia"] = {"mediana": float(np.median(dif)) if dif.size else None,
                                           "ic95": ic(dif) if dif.size else None, "p": p}
        f = lambda ob: (f"{fila[ob]['snr50']:.3f} [{ic(fila[ob]['boot'])[0]:.2f},{ic(fila[ob]['boot'])[1]:.2f}]"
                        if fila[ob]["snr50"] else "n/d")
        print(f"{m:16s} {f(a_):>22s} {f(b_):>22s} "
              f"{(np.median(dif) if dif.size else float('nan')):>+16.3f} {p:>8.4f}")
    salida = Path(a.salida)
    salida.write_text(json.dumps(res, indent=1, default=lambda o: None))
    print("escrito", salida)


if __name__ == "__main__":
    main()
