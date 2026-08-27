#!/usr/bin/env python
"""Figura 2 — curvas de completitud, SNR50 y su significancia, con procedencia estampada.

La unidad independiente de este diseno es la POSICION, no la fila: las filas de una
misma posicion comparten ruido local, residuo de PSF y realizacion del cielo, y los
puntos de SNR reutilizan las MISMAS posiciones, asi que tampoco son independientes
entre si. Por eso aqui:

  * el VALOR CENTRAL de cada punto marginaliza sobre las filas (promedia sobre
    elecciones de analisis: anchura de plantilla y modo de continuo);
  * toda la INFERENCIA —intervalos y significancia— remuestrea o baraja POSICIONES,
    nunca filas. Un Fisher fila a fila sobreestima, y un CMH a lo largo de la rejilla
    sobreestimaria igual, por la misma razon.

El estadistico es el SNR al 50 % de completitud (SNR50) y no el area entre curvas,
porque el area depende del rango y del paso de la rejilla —que ya han diferido entre
runs (7 puntos contra 12)— mientras que el SNR50 es una propiedad de la curva,
invariante al muestreo mientras este acotada.
"""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
E4_TABLE = "tables/injection_throughput_by_method.csv"
#: reglas de exclusion pre-registradas. Se fija ANTES de ver los resultados.
EXCLUSIONES = ("ninguna", "por-metodo", "global", "par")
#: el par que se grafica: la exclusion `par` mantiene las dos curvas sobre
#: posiciones IDENTICAS, que es lo unico que hace de `aperture` un control valido.
PAR = ("psffit", "aperture")


def _git(*args: str) -> str:
    try:
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception:
        return ""


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def tabla_larga(run: str, project_root: Path = ROOT) -> list[dict]:
    """Una fila por inyeccion, con la posicion como identidad de primera clase."""
    path = project_root / "runs" / run / E4_TABLE
    filas = []
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            if row["variant"] != "nominal":
                continue
            filas.append({
                "run": run,
                "position_id": f"{run}:{row['position_label']}",
                "position_label": row["position_label"],
                "position_y": float(row["position_y"]),
                "position_x": float(row["position_x"]),
                "injection_id": row["injection_id"],
                "method": row["method"],
                "input_snr": float(row["input_snr"]),
                "template_width": row["template_width"],
                "continuum_mode": row["continuum_mode"],
                "complete": row["complete"] == "True",
            })
    if not filas:
        raise SystemExit(f"{path}: sin filas `nominal`.")
    return filas


def fap_por_posicion(filas: list[dict], metodo: str) -> dict[str, dict]:
    """Falsos positivos empiricos al umbral: las detecciones con SNR inyectada 0."""
    out: dict[str, dict] = {}
    for f in (x for x in filas if x["method"] == metodo and x["input_snr"] == 0.0):
        d = out.setdefault(f["position_id"], {"detectados": 0, "n": 0})
        d["n"] += 1
        d["detectados"] += int(f["complete"])
    for d in out.values():
        d["fap"] = d["detectados"] / d["n"] if d["n"] else None
    return out


def posiciones_contaminadas(filas: list[dict], regla: str) -> set[str]:
    if regla == "ninguna":
        return set()
    metodos = sorted({f["method"] for f in filas})
    if regla == "global":
        objetivo = metodos
    elif regla == "par":
        objetivo = [m for m in PAR if m in metodos]
    else:  # por-metodo se resuelve fuera, por curva
        return set()
    malas = set()
    for m in objetivo:
        for pos, d in fap_por_posicion(filas, m).items():
            if d["detectados"] > 0:
                malas.add(pos)
    return malas


def curva(filas: list[dict], metodo: str, excluidas: set[str]) -> dict:
    """Completitud por punto de SNR. Valor central sobre FILAS, n reportado en POSICIONES."""
    sub = [f for f in filas if f["method"] == metodo and f["position_id"] not in excluidas]
    puntos = []
    for s in sorted({f["input_snr"] for f in sub}):
        en_s = [f for f in sub if f["input_snr"] == s]
        pos = sorted({f["position_id"] for f in en_s})
        puntos.append({
            "input_snr": s,
            "completitud": sum(f["complete"] for f in en_s) / len(en_s),
            "positions_per_point": len(pos),
            "injections_per_point": len({f["injection_id"] for f in en_s}),
            "rows_per_point": len(en_s),
        })
    return {"metodo": metodo, "puntos": puntos,
            "posiciones": sorted({f["position_id"] for f in sub})}


def snr50(puntos: list[dict]) -> float | None:
    """Primer cruce ascendente del 50 %, por interpolacion lineal.

    Devuelve None si la curva no cruza: un SNR50 no acotado no se inventa, se declara.
    """
    xs = [p["input_snr"] for p in puntos]
    ys = [p["completitud"] for p in puntos]
    for i in range(1, len(xs)):
        if ys[i - 1] < 0.5 <= ys[i]:
            if ys[i] == ys[i - 1]:
                return xs[i]
            t = (0.5 - ys[i - 1]) / (ys[i] - ys[i - 1])
            return xs[i - 1] + t * (xs[i] - xs[i - 1])
    return None


def _curva_de_posiciones(filas, metodo, posiciones, rejilla):
    idx = {}
    for f in filas:
        if f["method"] != metodo:
            continue
        idx.setdefault((f["position_id"], f["input_snr"]), []).append(f["complete"])
    puntos = []
    for s in rejilla:
        vals = []
        for p in posiciones:
            vals.extend(idx.get((p, s), []))
        if not vals:
            return None
        puntos.append({"input_snr": s, "completitud": sum(vals) / len(vals)})
    return puntos


def bootstrap_puntos(filas, metodo, posiciones, rejilla, n_boot, rng):
    """Intervalo por punto remuestreando POSICIONES con reemplazo, no filas."""
    muestras = np.empty((n_boot, len(rejilla)))
    pos = list(posiciones)
    for b in range(n_boot):
        elegidas = [pos[i] for i in rng.integers(0, len(pos), len(pos))]
        c = _curva_de_posiciones(filas, metodo, elegidas, rejilla)
        muestras[b] = [p["completitud"] for p in c]
    lo, hi = np.percentile(muestras, [2.5, 97.5], axis=0)
    return lo.tolist(), hi.tolist()


def permutacion_snr50(filas_a, filas_b, metodo, pos_a, pos_b, rejilla, n_perm, rng):
    """Baraja la etiqueta de OBJETO entre posiciones y recalcula la curva entera."""
    todas = list(pos_a) + list(pos_b)
    juntas = filas_a + filas_b
    na = len(pos_a)
    obs_a = snr50(_curva_de_posiciones(filas_a, metodo, pos_a, rejilla))
    obs_b = snr50(_curva_de_posiciones(filas_b, metodo, pos_b, rejilla))
    if obs_a is None or obs_b is None:
        return {"snr50_a": obs_a, "snr50_b": obs_b, "delta": None, "p": None,
                "nota": "alguna curva no cruza el 50 %: el SNR50 no esta acotado."}
    obs = abs(obs_b - obs_a)
    extremos = 0
    validas = 0
    for _ in range(n_perm):
        perm = list(todas)
        rng.shuffle(perm)
        ca = _curva_de_posiciones(juntas, metodo, perm[:na], rejilla)
        cb = _curva_de_posiciones(juntas, metodo, perm[na:], rejilla)
        sa, sb = (snr50(ca) if ca else None), (snr50(cb) if cb else None)
        if sa is None or sb is None:
            continue
        validas += 1
        if abs(sb - sa) >= obs:
            extremos += 1
    return {"snr50_a": obs_a, "snr50_b": obs_b, "delta": obs_b - obs_a,
            "razon": (obs_b / obs_a) if obs_a else None,
            "p": (extremos + 1) / (validas + 1) if validas else None,
            "n_permutaciones_validas": validas}


def procedencia(runs: list[str], project_root: Path) -> dict:
    return {
        "commit": _git("rev-parse", "HEAD"),
        "commit_corto": _git("rev-parse", "--short", "HEAD"),
        "rama": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "tag": _git("describe", "--tags", "--exact-match") or None,
        "arbol_sucio": bool(_git("status", "--porcelain")),
        "generado_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "runs": {
            r: {
                "config_sha256": sha256_file(project_root / "runs" / r / "config" / "config.json"),
                "tabla_e4": E4_TABLE,
                "tabla_e4_sha256": sha256_file(project_root / "runs" / r / E4_TABLE),
                "tabla_e4_mtime_utc": _dt.datetime.fromtimestamp(
                    (project_root / "runs" / r / E4_TABLE).stat().st_mtime,
                    _dt.timezone.utc).isoformat(timespec="seconds"),
            } for r in runs
        },
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-a", required=True, help="run del objeto A")
    ap.add_argument("--run-b", required=True, help="run del objeto B")
    ap.add_argument("--metodos", default=",".join(PAR),
                    help="metodos a medir; el primero es el canonico de la figura")
    ap.add_argument("--exclusion", default="par", choices=EXCLUSIONES,
                    help="regla PRE-REGISTRADA de exclusion por falsos positivos en SNR=0")
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--n-perm", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=20260826)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--project-root", default=str(ROOT))
    args = ap.parse_args(argv)

    root = Path(args.project_root).resolve()
    rng = np.random.default_rng(args.seed)
    metodos = [m.strip() for m in args.metodos.split(",") if m.strip()]
    out = Path(args.out_dir) if args.out_dir else root / "reports" / "fig2"
    out.mkdir(parents=True, exist_ok=True)

    filas = {r: tabla_larga(r, root) for r in (args.run_a, args.run_b)}
    rejilla = sorted(set.intersection(*[{f["input_snr"] for f in v} for v in filas.values()]))

    resultado = {
        "estadistico": "SNR50 (SNR al 50 % de completitud), interpolacion lineal del primer cruce",
        "unidad_independiente": "posicion",
        "exclusion": args.exclusion,
        "rejilla_snr_comun": rejilla,
        "semilla": args.seed,
        "n_boot": args.n_boot,
        "n_perm": args.n_perm,
        "fap_por_posicion": {},
        "curvas": {},
        "comparacion": {},
        "procedencia": procedencia([args.run_a, args.run_b], root),
    }

    filas_larga = []
    for r, v in filas.items():
        filas_larga.extend(v)
        resultado["fap_por_posicion"][r] = {
            m: fap_por_posicion(v, m) for m in metodos
        }

    for metodo in metodos:
        for etiqueta, regla in (("sin_exclusion", "ninguna"), ("con_exclusion", args.exclusion)):
            if regla == "por-metodo":
                excl = {r: {p for p, d in fap_por_posicion(filas[r], metodo).items()
                            if d["detectados"] > 0} for r in filas}
            else:
                excl = {r: posiciones_contaminadas(filas[r], regla) for r in filas}
            cur = {r: curva(filas[r], metodo, excl[r]) for r in filas}
            for r in filas:
                pos = cur[r]["posiciones"]
                pts = [p for p in cur[r]["puntos"] if p["input_snr"] in rejilla]
                lo, hi = bootstrap_puntos(filas[r], metodo, pos, rejilla, args.n_boot, rng)
                for p, a, b in zip(pts, lo, hi):
                    p["ic95_lo"], p["ic95_hi"] = a, b
                cur[r]["puntos"] = pts
                cur[r]["snr50"] = snr50(pts)
                cur[r]["posiciones_excluidas"] = sorted(excl[r])
            resultado["curvas"].setdefault(metodo, {})[etiqueta] = cur
            resultado["comparacion"].setdefault(metodo, {})[etiqueta] = permutacion_snr50(
                filas[args.run_a], filas[args.run_b], metodo,
                cur[args.run_a]["posiciones"], cur[args.run_b]["posiciones"],
                rejilla, args.n_perm, rng)

    larga = out / "fig2_tabla_larga.csv"
    with open(larga, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(filas_larga[0]))
        w.writeheader()
        w.writerows(filas_larga)

    curvas_csv = out / "fig2_curvas.csv"
    with open(curvas_csv, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["run", "method", "exclusion", "input_snr", "completeness",
                    "ic95_lo", "ic95_hi", "positions_per_point",
                    "injections_per_point", "rows_per_point"])
        for metodo, porexcl in resultado["curvas"].items():
            for etiqueta, cur in porexcl.items():
                for r, c in cur.items():
                    for p in c["puntos"]:
                        w.writerow([r, metodo, etiqueta, p["input_snr"], p["completitud"],
                                    p.get("ic95_lo"), p.get("ic95_hi"),
                                    p["positions_per_point"], p["injections_per_point"],
                                    p["rows_per_point"]])

    resultado["productos"] = {"tabla_larga": str(larga), "curvas": str(curvas_csv)}
    js = out / "fig2_procedencia.json"
    js.write_text(json.dumps(resultado, indent=2, ensure_ascii=False), encoding="utf-8")
    print(js)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
