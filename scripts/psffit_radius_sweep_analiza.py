#!/usr/bin/env python
"""Le pone barras de error al barrido de radios, sin volver a computarlo.

El barrido (`psffit_radius_sweep.py`) parte las posiciones de control en DOS
mitades, elige en A y reporta en B. Eso es una particion, y con ella `20x8` gana
en ROXs 42B b — pero no se sabe si ganaria con otra. Esta herramienta responde a
eso releyendo `radius_sweep_rows_*.csv`: **repite la particion muchas veces** y
mira que sobrevive.

Publica dos cosas, y la segunda es la que decide:

  * **Con que frecuencia se elige cada configuracion.** Se sortea una particion,
    se elige por `f50` en A y se anota. Si una configuracion se lleva el 30 % de
    los sorteos entre cinco, no es un ganador: es lo que da el azar.
  * **La diferencia EMPAREJADA contra produccion.** En cada sorteo se calcula
    `f50` de las dos configuraciones sobre **las mismas** posiciones, y se resta.
    Emparejar quita la varianza de que a una le tocaran posiciones faciles, que
    es la que domina: las diferencias entre configuraciones (0.108 en ROXs 12 b)
    son del tamano de la dispersion entre mitades (0.050-0.107), asi que sin
    emparejar no se distingue nada.

**La unidad de remuestreo es la POSICION, no la fila** (misma leccion que
`fig2_completeness.py` y `detection_threshold_by_method.py`): las ~7 filas de una
posicion son la misma posicion a distintos niveles de senal inyectada, y tratarlas
como independientes multiplicaria el `n` efectivo por siete.

Se remuestrea **sin reemplazo** (mitades disjuntas), no con bootstrap clasico: la
estandarizacion de E4 deja fuera cada nula por su `injection_id`, y una posicion
duplicada por el reemplazo se saldria dos veces de su propia referencia. La
mitad aleatoria repetida mide lo mismo y no rompe esa maquinaria.

**Lo que estas barras de error NO cubren.** Miden incertidumbre **de posicion**, y
de ninguna otra cosa. Las configuraciones comparten el cubo, asi que estan
emparejadas en ruido —lo unico que las separa es el estimador, que es lo que se
quiere medir—, pero **sus nulas no son las mismas**: cada estimador tiene su
propia distribucion nula sobre ESTA realizacion, y de esa hay una sola. Si las
nulas de una configuracion salieron mas apretadas en este cubo, su umbral es mas
bajo en todos los sorteos y re-particionar no lo deshace. Medido con sinteticos
el 2026-08-30: dos estimadores de sensibilidad identica con realizaciones de
ruido distintas salen 37-3. Con otra observacion, no con otra particion, es como
se cerraria eso.

Uso:

    python scripts/psffit_radius_sweep_analiza.py \\
        --filas /mnt/2TB/MUSE_work/psffit_radius_sweep/manana_20260830/radius_sweep_rows_*.csv
"""

from __future__ import annotations

import argparse
import collections
import csv
import glob
import math
import random
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for extra in (ROOT, ROOT / "scripts"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from detection_threshold_by_method import umbral_para_fpr  # noqa: E402

from musepipe.stages.stage_h04_injection import apply_snr_standardization  # noqa: E402
from psffit_radius_sweep import _f50  # noqa: E402

FPR = 0.05
N_SORTEOS = 200
SEMILLA = 20260830

#: Cuantil normal del 95 %: el umbral robusto que deja el mismo 5 % nominal que
#: `--fpr 0.05`, pero calculado con la mediana y la escala de las nulas en vez de
#: con su MAXIMO.
Z_95 = 1.6448536269514722

MODOS_UMBRAL = ("fpr", "robusto")


def umbral_robusto(nulas, z=Z_95):
    """Mediana + z x escala de las nulas, en vez del maximo de la muestra.

    **Por que hace falta.** `umbral_para_fpr(nulas, 0.05)` con menos de 20 nulas
    da k=0, asi que el umbral ES el maximo — el estadistico de orden mas volatil
    que hay. Medido con filas sinteticas el 2026-08-30: dos configuraciones de
    sensibilidad IDENTICA, con realizaciones de ruido distintas, salen 37-3 a
    favor de aquella cuyas nulas quedaron mas apretadas. Y el sesgo no se va
    remuestreando posiciones, porque re-particionar **no vuelve a sortear el
    ruido**: la realizacion es una por configuracion y se arrastra a todos los
    sorteos.

    Este umbral usa dos estadisticos robustos en vez de un extremo, asi que la
    misma volatilidad pesa mucho menos. No sustituye al de FPR —ese es el que
    controla falsos positivos de verdad—: se publican **los dos**, y un veredicto
    que solo aparece en uno no es un veredicto.
    """
    finitas = [v for v in nulas if math.isfinite(v)]
    if len(finitas) < 2:
        return float("nan")
    mediana = statistics.median(finitas)
    desviaciones = [abs(v - mediana) for v in finitas]
    escala = 1.4826 * statistics.median(desviaciones)
    if not (escala > 0):
        escala = statistics.pstdev(finitas)
    if not (escala > 0):
        return float("nan")
    return mediana + z * escala


def lee_filas(patron):
    rutas = sorted(glob.glob(patron))
    if not rutas:
        raise SystemExit(f"sin ficheros que casen con {patron!r}")
    filas = []
    for ruta in rutas:
        with open(ruta, newline="") as fh:
            for fila in csv.DictReader(fh):
                fila["input_snr"] = float(fila["input_snr"])
                fila["template_factor"] = float(fila["template_factor"])
                fila["recovered_flux"] = float(fila["recovered_flux"])
                fila["variant"] = fila.get("variant") or "nominal"
                filas.append(fila)
    return filas, rutas


#: Estados de un `f50`, ordenados de MEJOR a PEOR. `satura` y `no_llega` son
#: datos **censurados**, no ausentes: si la completitud ya pasa del 50 % en el
#: nivel mas bajo de la rejilla, el cruce esta por debajo y esa configuracion es
#: mejor que cualquiera con un cruce medido — tratarla como `nan` y descartarla
#: de la comparacion emparejada eliminaria justo los sorteos en los que gana, que
#: es como se fabrica un empate a partir de una ventaja.
SATURA, RESUELTO, NO_LLEGA = "satura", "resuelto", "no_llega"


def clave_orden(medida):
    """Clave que ordena `f50` de mejor a peor respetando la censura."""
    estado, valor = medida["estado"], medida["f50"]
    if estado == SATURA:
        return (0, valor)      # cruce por DEBAJO de la rejilla
    if estado == RESUELTO:
        return (1, valor)
    return (2, float("inf"))   # nunca alcanza el 50 %


def f50_de(filas, posiciones, *, fpr, modo_umbral="fpr"):
    """`f50` de un conjunto de posiciones, con SU umbral y SU estandarizacion.

    Devuelve `{"f50", "estado"}`. El estado distingue las dos formas de no tener
    numero, que son opuestas: `satura` es mejor que cualquier valor medido y
    `no_llega` es peor.
    """
    indeterminado = {"f50": float("nan"), "estado": None}
    subset = [dict(f) for f in filas if f["position_label"] in posiciones]
    if not subset:
        return indeterminado
    try:
        subset = apply_snr_standardization(subset, mode="injection_nulls", threshold_snr=3.5)
    except ValueError:
        # Sin al menos dos nulas por estrato no hay escala. Es un sorteo que no
        # se puede evaluar —ni bueno ni malo—, distinto de la censura: se
        # descarta y no entra en ninguna cuenta.
        return indeterminado
    vals = {}
    for fila in subset:
        v = fila.get("recovered_snr_std")
        v = float("nan") if v is None else float(v)
        vals.setdefault(fila["input_snr"], []).append(v)
    nulas = [v for v in vals.get(0.0, []) if math.isfinite(v)]
    umbral = (umbral_para_fpr(nulas, fpr)["umbral"] if modo_umbral == "fpr"
              else umbral_robusto(nulas))
    if not math.isfinite(umbral):
        return indeterminado
    niveles = sorted(n for n in vals if n > 0)
    comp = []
    for nivel in niveles:
        finitos = [v for v in vals[nivel] if math.isfinite(v)]
        comp.append(sum(1 for v in finitos if v >= umbral) / len(finitos) if finitos else float("nan"))

    valor = _f50(niveles, comp)
    if math.isfinite(valor):
        return {"f50": valor, "estado": RESUELTO}
    medidos = [(n, c) for n, c in zip(niveles, comp) if math.isfinite(c)]
    if not medidos:
        return indeterminado
    # `_f50` devuelve nan en los dos extremos; cual de los dos es lo dice la
    # completitud en el nivel mas bajo medido.
    if medidos[0][1] >= 0.5:
        return {"f50": medidos[0][0], "estado": SATURA}
    return {"f50": medidos[-1][0], "estado": NO_LLEGA}


def resumen(valores):
    finitos = sorted(v for v in valores if math.isfinite(v))
    if not finitos:
        return None
    n = len(finitos)
    return {
        "n": n,
        "mediana": statistics.median(finitos),
        "p05": finitos[max(0, int(0.05 * n) - 1)],
        "p95": finitos[min(n - 1, int(0.95 * n))],
    }


def analiza(filas, *, fpr, n_sorteos, semilla, referencia, modo_umbral="fpr"):
    rng = random.Random(semilla)
    configs = sorted({(float(f["r_star_px"]), float(f["r_comp_px"])) for f in filas})
    posiciones = sorted({f["position_label"] for f in filas
                         if f["position_label"].startswith("control")})
    por_config = {c: [f for f in filas
                      if (float(f["r_star_px"]), float(f["r_comp_px"])) == c]
                  for c in configs}

    elegidas = collections.Counter()
    f50_b = {c: [] for c in configs}
    delta = {c: [] for c in configs if c != referencia}
    duelo = {c: collections.Counter() for c in configs if c != referencia}
    censura = {c: collections.Counter() for c in configs}
    sorteos_validos = 0

    for _ in range(n_sorteos):
        barajado = posiciones[:]
        rng.shuffle(barajado)
        corte = len(barajado) // 2
        mitad_a, mitad_b = set(barajado[:corte]), set(barajado[corte:])

        en_a = {c: f50_de(por_config[c], mitad_a, fpr=fpr, modo_umbral=modo_umbral)
                for c in configs}
        en_b = {c: f50_de(por_config[c], mitad_b, fpr=fpr, modo_umbral=modo_umbral)
                for c in configs}

        evaluables_a = {c: m for c, m in en_a.items() if m["estado"]}
        if evaluables_a:
            elegidas[min(evaluables_a, key=lambda c: clave_orden(evaluables_a[c]))] += 1
        for c in configs:
            censura[c][en_b[c]["estado"] or "indeterminado"] += 1
            f50_b[c].append(en_b[c]["f50"] if en_b[c]["estado"] == RESUELTO else float("nan"))

        # El duelo se resuelve en la MISMA mitad para las dos configuraciones, y
        # con la censura puesta: un rival que satura GANA a una referencia con
        # cruce medido, aunque no tenga numero que restar.
        ref = en_b.get(referencia)
        if not (ref and ref["estado"]):
            continue
        sorteos_validos += 1
        for c in delta:
            rival = en_b[c]
            if not rival["estado"]:
                duelo[c]["indeterminado"] += 1
                continue
            k_rival, k_ref = clave_orden(rival), clave_orden(ref)
            if k_rival < k_ref:
                duelo[c]["gana"] += 1
            elif k_rival > k_ref:
                duelo[c]["pierde"] += 1
            else:
                duelo[c]["empata"] += 1
            # La diferencia numerica solo existe cuando las DOS estan resueltas.
            if rival["estado"] == RESUELTO and ref["estado"] == RESUELTO:
                delta[c].append(rival["f50"] - ref["f50"])

    return {"configs": configs, "posiciones": len(posiciones), "elegidas": elegidas,
            "f50_b": f50_b, "delta": delta, "duelo": duelo, "censura": censura,
            "sorteos_validos": sorteos_validos}


def imprime(run, res, *, n_sorteos, referencia, modo_umbral):
    print(f"\n=== {run}: {res['posiciones']} posiciones de control, "
          f"{n_sorteos} particiones aleatorias · umbral={modo_umbral} ===")
    print(f"referencia = {int(referencia[0])}x{int(referencia[1])} (produccion)\n")
    print(f"{'config':>8} {'f50_B mediana':>14} {'[p05, p95]':>18} "
          f"{'elegida':>9} {'d vs prod':>11} {'gana':>7} {'censura':>16}")
    for c in res["configs"]:
        etiqueta = f"{int(c[0])}x{int(c[1])}"
        r = resumen(res["f50_b"][c])
        celda = f"{r['mediana']:.3f}" if r else "sin resolver"
        rango = f"[{r['p05']:.3f}, {r['p95']:.3f}]" if r else "-"
        frac_elegida = res["elegidas"][c] / max(1, sum(res["elegidas"].values()))
        cens = res["censura"][c]
        total_cens = max(1, sum(cens.values()))
        marca = " ".join(f"{k[:3]}={v / total_cens:.0%}" for k, v in cens.most_common(2))
        if c == referencia:
            d, gana = "-", "-"
        else:
            rd = resumen(res["delta"][c])
            d = f"{rd['mediana']:+.3f}" if rd else "-"
            duelos = res["duelo"][c]
            decididos = duelos["gana"] + duelos["pierde"] + duelos["empata"]
            gana = f"{duelos['gana'] / decididos:.0%}" if decididos else "-"
        print(f"{etiqueta:>8} {celda:>14} {rango:>18} {frac_elegida:>8.0%} "
              f"{d:>11} {gana:>7} {marca:>16}")
    print("\n  `elegida` = fraccion de particiones en las que gana en la mitad A."
          "\n  `gana`    = fraccion de duelos en la MISMA mitad B en los que es MEJOR"
          "\n              que produccion. Incluye los censurados: saturar por debajo"
          "\n              de la rejilla cuenta como ganar."
          "\n  `d vs prod` solo promedia los sorteos con las DOS resueltas, asi que"
          "\n              puede quedarse corto si el rival satura a menudo; manda `gana`."
          "\n  `censura` = como salio f50 en B (res=resuelto, sat=satura, no_=no llega).")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--filas", required=True, help="ruta o patron de radius_sweep_rows_*.csv")
    ap.add_argument("--fpr", type=float, default=FPR)
    ap.add_argument("--sorteos", type=int, default=N_SORTEOS)
    ap.add_argument("--semilla", type=int, default=SEMILLA)
    ap.add_argument("--referencia", default="20x12", help="configuracion de produccion")
    args = ap.parse_args(argv)

    rs, _, rc = args.referencia.lower().partition("x")
    referencia = (float(rs), float(rc))

    filas, rutas = lee_filas(args.filas)
    print("filas leidas de:")
    for ruta in rutas:
        print(f"  {ruta}")
    for run in sorted({f["run"] for f in filas}):
        del_run = [f for f in filas if f["run"] == run]
        # Los DOS umbrales, siempre. Un veredicto que solo aparece con uno de
        # ellos es del umbral, no del estimador.
        for modo in MODOS_UMBRAL:
            res = analiza(del_run, fpr=args.fpr, n_sorteos=args.sorteos,
                          semilla=args.semilla, referencia=referencia, modo_umbral=modo)
            imprime(run, res, n_sorteos=args.sorteos, referencia=referencia,
                    modo_umbral=modo)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
