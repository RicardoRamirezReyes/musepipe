#!/usr/bin/env python
"""El umbral de deteccion, calibrado por metodo a igualdad de falsos positivos.

`h04_detection_threshold_snr` es **un solo 5 sigma para los seis metodos**. Este
script mide si ese numero unico esta justificado, y lo hace sin correr ninguna
etapa: lee las tablas que E4 y E4b ya dejaron en disco.

La comparacion correcta no es "cuantos falsos positivos da cada metodo al mismo
umbral" —un estimador mas sensible detecta mas de todo—, sino: se fija la tasa de
falsos positivos que se tolera, se busca el umbral que cada metodo necesita para
cumplirla, y se mira cuanto detecta ahi. Eso es lo que calcula `umbral_para_fpr`.

Antes de eso, el script publica la **distribucion nula** de cada metodo. Es el
diagnostico que manda: si el `recovered_snr` fuera una SNR calibrada, con senal
nula tendria media 0 y sigma 1. Cuando no lo es, un umbral unico no puede ser
correcto para todos, y la media y la sigma dicen en cual de las dos formas falla
—un pedestal (media desplazada) o una cola (sigma inflada)—, que no se arreglan
igual.

Dos advertencias sobre lo que se puede concluir:

  * **La unidad independiente es la POSICION**, no la fila (misma leccion que
    `fig2_completeness.py`). En el cubo combinado cada nulo ES una posicion. Por
    exposicion no: las ~200 medidas nulas son ~7 posiciones repetidas en ~29
    exposiciones, asi que el `n` efectivo es menor que el `n` de filas y el
    script publica los dos.
  * **La resolucion en FPR es 1/n_nulos.** Con 15 nulos no se puede calibrar un
    1 %: el umbral lo fijarian 0 eventos. El script devuelve la FPR realmente
    conseguida junto a la pedida, y son distintas justamente cuando el resultado
    no se sostiene.

Los umbrales se calibran sobre los MISMOS datos en los que se mide la
completitud; no hay muestra de validacion aparte. Por eso se piden varios
niveles de FPR: si la conclusion solo aparece en el mas apretado, es del punto de
calibracion y no del estimador.
"""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import hashlib
import json
import math
import statistics
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

#: E4, cubo combinado: los seis metodos de `METHOD_ORDER`.
TABLA_E4 = "tables/injection_throughput_by_method.csv"
#: E4b, por exposicion: solo `aperture` y `psffit` (`METODOS` de la etapa).
TABLA_E4B = "tables/perexp_injection_by_exposure.csv"

FPR_OBJETIVO = (0.05, 0.02, 0.01, 0.005)
#: el valor que `h04_detection_threshold_snr` trae por defecto en E4.
UMBRAL_ACTUAL = 5.0


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


def umbral_para_fpr(nulos, fpr):
    """El umbral mas bajo que deja como mucho `fpr` de los nulos por encima.

    Devuelve tambien la FPR **conseguida**, que es un multiplo de 1/n y solo
    coincide con la pedida por casualidad. Cuando k=0 el umbral lo fija un unico
    evento —el maximo— y la FPR conseguida es 0: eso no es "cumplir el 1 %", es
    quedarse sin resolucion, y por eso se publica.
    """
    xs = sorted((v for v in nulos if math.isfinite(v)), reverse=True)
    n = len(xs)
    if n == 0:
        return {"umbral": float("nan"), "fpr_conseguida": float("nan"),
                "n_por_encima": 0, "n_nulos": 0}
    k = int(math.floor(float(fpr) * n))
    if k >= n:
        return {"umbral": float("-inf"), "fpr_conseguida": 1.0,
                "n_por_encima": n, "n_nulos": n}
    # +eps para excluir el propio evento: por encima quedan exactamente k.
    umbral = xs[k] + 1e-9 if k > 0 else xs[0] + 1e-9
    return {"umbral": umbral, "fpr_conseguida": k / n,
            "n_por_encima": k, "n_nulos": n}


def tasa(valores, umbral):
    finitos = [v for v in valores if math.isfinite(v)]
    if not finitos:
        return float("nan")
    return sum(1 for v in finitos if v >= umbral) / len(finitos)


def resumen_nulo(valores):
    finitos = [v for v in valores if math.isfinite(v)]
    if not finitos:
        return {"n": 0}
    return {
        "n": len(finitos),
        "media": statistics.fmean(finitos),
        "sigma": statistics.pstdev(finitos) if len(finitos) > 1 else float("nan"),
        "max": max(finitos),
        "min": min(finitos),
    }


def _filas_e4(run, project_root):
    """Las filas nominales y sin continuo: la rama que define la completitud.

    Es el mismo corte que `_completeness` en `stage_h04_injection` —variante
    nominal, plantilla LSF, `continuum_mode` none—, para que el umbral se calibre
    sobre la poblacion a la que despues se le aplica.
    """
    path = project_root / "runs" / run / TABLA_E4
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            if row["variant"] != "nominal":
                continue
            if abs(float(row["template_factor"]) - 1.0) > 1e-9:
                continue
            if row["continuum_mode"] != "none":
                continue
            yield row


def _filas_e4b(run, project_root):
    path = project_root / "runs" / run / TABLA_E4B
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            if row["variant"] != "nominal":
                continue
            yield row


def _agrupa(filas, snr_ref, columna="recovered_snr"):
    """Nulos y senal por metodo, **solo en posiciones de control**.

    La posicion `real` queda fuera de las dos: puede llevar senal de verdad, y
    contarla como nulo es lo que convierte una deteccion en un falso positivo.
    """
    nulos, senal, posiciones, unidades = {}, {}, {}, {}
    for row in filas:
        if not str(row["position_label"]).startswith("control"):
            continue
        metodo = row["method"]
        try:
            snr = float(row[columna])
        except (ValueError, KeyError):
            continue
        nivel = float(row["input_snr"])
        posiciones.setdefault(metodo, set()).add(row["position_label"])
        if nivel == 0.0:
            nulos.setdefault(metodo, []).append(snr)
            unidades.setdefault(metodo, set()).add(row["position_label"])
        elif abs(nivel - float(snr_ref)) < 1e-9:
            senal.setdefault(metodo, []).append(snr)
    return nulos, senal, posiciones


def analiza(run, sustrato, project_root, snr_ref, fprs, umbral_actual, columna="recovered_snr"):
    if sustrato == "combinado":
        filas = list(_filas_e4(run, project_root))
        tabla = TABLA_E4
    else:
        filas = list(_filas_e4b(run, project_root))
        tabla = TABLA_E4B
    nulos, senal, posiciones = _agrupa(filas, snr_ref, columna)
    out = {"run": run, "sustrato": sustrato, "tabla": tabla, "columna": columna,
           "snr_inyectada_referencia": float(snr_ref), "metodos": {}}
    for metodo in sorted(nulos):
        nul = nulos[metodo]
        sen = senal.get(metodo, [])
        entrada = {
            "nulo": resumen_nulo(nul),
            "n_posiciones": len(posiciones.get(metodo, ())),
            "resolucion_fpr": 1.0 / len(nul) if nul else float("nan"),
            "umbral_actual": {
                "umbral": float(umbral_actual),
                "fpr": tasa(nul, umbral_actual),
                "completitud": tasa(sen, umbral_actual),
                "n_senal": len(sen),
            },
            "calibrado": {},
        }
        for fpr in fprs:
            cal = umbral_para_fpr(nul, fpr)
            cal["completitud"] = tasa(sen, cal["umbral"])
            entrada["calibrado"][f"{fpr:g}"] = cal
        out["metodos"][metodo] = entrada
    return out


def imprime(bloque, fprs):
    print(f"\n### {bloque['run']} — sustrato {bloque['sustrato']} — columna {bloque.get('columna')}"
          f" (completitud a SNR inyectada = {bloque['snr_inyectada_referencia']:g})")
    if not bloque["metodos"]:
        print("  (sin filas)")
        return
    n_nul = next(iter(bloque["metodos"].values()))["nulo"]["n"]
    print(f"  distribucion nula: {n_nul} medidas por metodo"
          f" -> resolucion en FPR = {100.0 / n_nul:.1f} puntos")
    cab = (f"{'metodo':<16}{'media':>7}{'sigma':>7}{'FPR@5s':>8}{'compl@5s':>10}   ")
    cab += "  ".join(f"{'u@'+format(f*100,'g')+'%':>8}{'compl':>7}" for f in fprs)
    print("  " + cab)
    for metodo, e in bloque["metodos"].items():
        linea = (f"{metodo:<16}{e['nulo']['media']:>7.2f}{e['nulo']['sigma']:>7.2f}"
                 f"{100*e['umbral_actual']['fpr']:>7.1f}%"
                 f"{100*e['umbral_actual']['completitud']:>9.1f}%   ")
        trozos = []
        for f in fprs:
            cal = e["calibrado"][f"{f:g}"]
            marca = "" if cal["n_por_encima"] > 0 else "*"
            trozos.append(f"{format(cal['umbral'],'.2f')+marca:>8}"
                          f"{100*cal['completitud']:>6.1f}%")
        print("  " + linea + "  ".join(trozos))
    print("  (*) sin resolucion: ese umbral lo fija un solo evento, la FPR conseguida es 0")


def procedencia(runs, project_root, tablas):
    ficheros = {}
    for run in runs:
        for tabla in tablas:
            path = project_root / "runs" / run / tabla
            if not path.exists():
                continue
            ficheros[f"{run}/{tabla}"] = {
                "sha256": sha256_file(path),
                "mtime_utc": _dt.datetime.fromtimestamp(
                    path.stat().st_mtime, _dt.timezone.utc).isoformat(timespec="seconds"),
            }
    return {
        "commit": _git("rev-parse", "HEAD"),
        "commit_corto": _git("rev-parse", "--short", "HEAD"),
        "rama": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "arbol_sucio": bool(_git("status", "--porcelain")),
        "generado_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "tablas": ficheros,
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", required=True, help="lista separada por comas")
    ap.add_argument("--project-root", default=str(ROOT))
    ap.add_argument("--fpr", default=",".join(format(f, "g") for f in FPR_OBJETIVO))
    ap.add_argument("--umbral-actual", type=float, default=UMBRAL_ACTUAL,
                    help="el umbral unico contra el que se compara (E4 usa 5.0)")
    ap.add_argument("--snr-combinado", type=float, default=2.0,
                    help="SNR inyectada a la que se mide la completitud en el cubo "
                         "combinado; a 5 los seis metodos saturan al 100 %% y no discrimina")
    ap.add_argument("--snr-perexp", type=float, default=3.0,
                    help="idem por exposicion")
    ap.add_argument("--columna", default="recovered_snr",
                    help="la SNR sobre la que se calibra: `recovered_snr` (la formal del filtro) "
                         "o `recovered_snr_std` (estandarizada contra la nula, E4 v4)")
    ap.add_argument("--out", default=None, help="fichero JSON de salida")
    args = ap.parse_args(argv)

    project_root = Path(args.project_root).resolve()
    runs = [r.strip() for r in args.runs.split(",") if r.strip()]
    fprs = [float(f) for f in args.fpr.split(",") if f.strip()]

    bloques = []
    for run in runs:
        for sustrato, tabla, snr_ref in (
            ("combinado", TABLA_E4, args.snr_combinado),
            ("por_exposicion", TABLA_E4B, args.snr_perexp),
        ):
            if not (project_root / "runs" / run / tabla).exists():
                print(f"(sin {tabla} en {run}: se salta)", file=sys.stderr)
                continue
            bloque = analiza(run, sustrato, project_root, snr_ref, fprs, args.umbral_actual,
                             args.columna)
            bloques.append(bloque)
            imprime(bloque, fprs)

    payload = {
        "procedencia": procedencia(runs, project_root, (TABLA_E4, TABLA_E4B)),
        "umbral_actual_snr": args.umbral_actual,
        "fpr_objetivo": fprs,
        "bloques": bloques,
    }
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
        print(f"\nJSON -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
