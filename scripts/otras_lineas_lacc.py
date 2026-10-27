#!/usr/bin/env python
"""Las otras lineas en luminosidad de acrecion: los limites al 99 % de G2 v4 con Alcala et al. (2017).

Hermano de `paper_cifras_extincion.py`, con la misma regla: lee lo que las etapas escribieron y no
inventa entradas. Por cada objeto del paper (`targets/<slug>.json` -> `paper.run`) toma

- el limite al 99 % de cada linea, `flux_upper_limit_99` de `tables/g2_line_measurements.csv`
  (spec G2 v4 §4: filtro adaptado, la nula que manda y ya dividido por el throughput de E4);
- la A_V, la distancia y la ley de extincion del config del run (`h03_av`, `h03_distance_pc`,
  `h03_extinction_law`, `h03_rv_extinction`) y la unidad de flujo por `resolve_flux_unit`;
- la L_acc de Halpha de G3 (`stage_g3_qc.json -> combined_accretion`, la de la Tabla de acrecion
  del paper: deteccion en un objeto, limite en el otro);
- la L_bol de `targets/<slug>.json -> fisica.log_lbol` (Xuan et al. 2024, con su fuente);

y convierte cada limite con `musepipe.models.accretion.line_lacc`, la misma cadena (desenrojecer
con CCM89, 4 pi d^2, relacion L_acc-L_line) con la que G3 convierte Halpha. Las relaciones son las
de la Tabla B.1 de Alcala et al. (2017), copiadas abajo como constantes (`ALCALA17`), con la fila
de la tabla de la que sale cada una. La de Halpha se comprueba contra la que declara el run
(`h03_lacc_lha_a`, `h03_lacc_lha_b`): si no coinciden, el script para.

Imprime las cifras que el texto cita (paper, Sect. 4.1, 4.2, Ap. C y conclusiones):

- `L_acc(limite) / L_acc(Halpha)` por linea, y su rango sin He I 5016 («6 to 29 times» en la
  deteccion, «at least 18 times» en el limite); He I 5016 aparte («several hundred times»);
- en la deteccion, cuantas veces queda cada limite por encima del flujo que la misma relacion
  predice si la Halpha es acrecion (`flujo_predicho`, la inversa): «5-17 times below our limits»
  para Hbeta, He I 6678, Ca II 8542 y Pa 9, y «a factor of 7» para He I 6678;
- el triplete del Ca II en log(L_line/L_bol) frente a las Clase III M9 y M9.5 de Manara et al.
  (2013), TWA 26 y TWA 29 (`scripts/data/manara2013/`): «0.1-0.6 dex deeper».

Hasta el 2026-10-26 estas cifras salian de `~/muse_sesion_20261026/g2v4/conversiones_v4.py`, que
importaba las funciones del agente `lineas` (`claudio/lineas/scripts/medir_lineas.py`) y tomaba la
L_acc de Halpha redondeada del paper (3.5e-7 y 6.5e-8). Aqui se toma la de G3 sin redondear
(3.474e-7 y 6.461e-8): las cifras del texto no cambian (ver el informe del paso a la cadena).

    python scripts/otras_lineas_lacc.py
    python scripts/otras_lineas_lacc.py --project-root ../MUSE-accretion-pipeline   # desde un worktree
    python scripts/otras_lineas_lacc.py --json /tmp/otras_lineas_lacc.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from musepipe.models.accretion import L_SUN_ERG_S, line_lacc  # noqa: E402
from musepipe.models.extinction import CCMExtinction  # noqa: E402
from musepipe.stages.stage_h03_limits import luminosity_erg_s  # noqa: E402

#: la tabla de la que salen las relaciones
CITA_ALCALA17 = ("Alcala et al. 2017, A&A 600, A20, Table B.1 "
                 "(doi:10.1051/0004-6361/201629929; arXiv:1612.07054v1, p. 21)")

#: Alcala et al. 2017, Tabla B.1: log(L_acc/L_sol) = a log(L_line/L_sol) + b. `sigma` es la
#: dispersion del ajuste (dex) y `recomendada` la marca • de la tabla (relacion sugerida por los
#: autores). Verificadas por el agente `lineas` sobre el PDF de arXiv (1612.07054v1, la unica
#: version) y otra vez contra su texto al pasarlas aqui; no se compararon con el PDF de la revista.
ALCALA17 = {
    "Halpha": dict(a=1.13, a_err=0.05, b=1.74, b_err=0.19, sigma=0.41, recomendada=False, fila="H3 (Hα) 656.2800"),
    "Hbeta": dict(a=1.14, a_err=0.04, b=2.59, b_err=0.16, sigma=0.30, recomendada=True, fila="H4 (Hβ) 486.1325"),
    "He I 5016": dict(a=0.99, a_err=0.04, b=3.49, b_err=0.24, sigma=0.27, recomendada=True, fila="He i 501.5678"),
    "He I 6678": dict(a=1.25, a_err=0.06, b=4.70, b_err=0.33, sigma=0.36, recomendada=True, fila="He i 667.8151"),
    "He I 7065": dict(a=1.18, a_err=0.05, b=4.47, b_err=0.29, sigma=0.34, recomendada=True, fila="He i 706.5190"),
    "O I 7773": dict(a=1.27, a_err=0.09, b=4.66, b_err=0.49, sigma=0.45, recomendada=False,
                     fila="O i 777.3055 (triplet)"),
    "O I 8446": dict(a=1.08, a_err=0.12, b=3.46, b_err=0.62, sigma=0.60, recomendada=False, fila="O i 844.6360"),
    "Ca II 8498": dict(a=0.99, a_err=0.05, b=2.60, b_err=0.29, sigma=0.47, recomendada=False, fila="Ca ii 849.8020"),
    "Ca II 8542": dict(a=0.97, a_err=0.06, b=2.43, b_err=0.29, sigma=0.48, recomendada=False, fila="Ca ii 854.2090"),
    "Ca II 8662": dict(a=0.93, a_err=0.06, b=2.30, b_err=0.30, sigma=0.49, recomendada=False, fila="Ca ii 866.2140"),
    "Pa 9": dict(a=1.18, a_err=0.08, b=3.71, b_err=0.43, sigma=0.44, recomendada=False, fila="Pa9 922.9014"),
    "Pa 10": dict(a=1.15, a_err=0.10, b=3.60, b_err=0.52, sigma=0.53, recomendada=False, fila="Pa10 901.4909"),
}

#: las que el texto deja fuera del rango «6 to 29» / «at least 18», con el motivo que da
FUERA_DEL_RANGO = {
    "He I 5016": ("su relacion da limites cientos de veces por encima de Halpha y en los acretores "
                  "se mezcla con Fe II 5018 (paper, Ap. C)"),
}
#: las lineas cuyas predicciones cita el paper (Ap. C: «5-17 times below our limits»), una por familia
LINEAS_PREDICCION = ("Hbeta", "He I 6678", "Ca II 8542", "Pa 9")
#: la linea del «a factor of 7» (Sect. 4.1): la mas util, rara en las Clase III
LINEA_FACTOR = "He I 6678"
#: las familias de la frase de las conclusiones («the limits on He I, O I and the Ca II triplet»)
FAMILIAS_CONCLUSIONES = ("He I", "O I", "Ca II IRT")
#: el triplete que se compara con las Clase III de Manara et al. (2013)
CA_II = ("Ca II 8498", "Ca II 8542", "Ca II 8662")
#: Clase III M9 y M9.5 de Manara et al. (2013), copiadas con su README
MANARA13 = ROOT / "scripts" / "data" / "manara2013" / "clase3_m9.json"


# --------------------------------------------------------------------------- piezas puras

def relacion(nombre: str) -> dict:
    """La relacion de `ALCALA17` en la forma que pide `line_lacc` (con su cita)."""
    r = ALCALA17[nombre]
    return {"a": float(r["a"]), "b": float(r["b"]), "scatter_dex": float(r["sigma"]),
            "citation": f"{CITA_ALCALA17}, {r['fila']}"}


def extincion_del_run(cfg: dict) -> CCMExtinction:
    """La ley de extincion que declara el run (la de E3 y G3). Solo CCM89, como E3."""
    ley = str(cfg.get("h03_extinction_law", ""))
    if not ley.strip().lower().startswith("ccm"):
        raise SystemExit(f"h03_extinction_law={ley!r}: este script solo sabe aplicar CCM89, como E3")
    return CCMExtinction(float(cfg["h03_rv_extinction"]), citation=str(cfg["h03_extinction_law_citation"]))


def k_lambda(ext: CCMExtinction, rest_A: float) -> float:
    """A_lambda / A_V a la longitud de onda en reposo de la linea."""
    return float(np.atleast_1d(ext.a_lambda_over_av(float(rest_A)))[0])


def lacc_de_flujo(flujo, rest_A, *, av, distancia_pc, nombre, ext, unidad_cgs=1.0) -> dict:
    """`{l_line_lsun, l_acc_lsun, ...}` de un flujo OBSERVADO de la linea `nombre`.

    Es `musepipe.models.accretion.line_lacc`, la conversion de G3: F x unidad x 10^(0.4 k A_V),
    L = 4 pi d^2 F, y la relacion de Alcala et al. (2017) de esa linea.
    """
    return line_lacc(flujo, distance_pc=float(distancia_pc), av=float(av),
                     a_lambda_over_av=k_lambda(ext, rest_A), relation=relacion(nombre),
                     flux_unit_cgs=float(unidad_cgs))


def flujo_predicho(lacc_lsun, rest_A, *, av, distancia_pc, nombre, ext) -> float:
    """El flujo OBSERVADO (erg s^-1 cm^-2) que la relacion de `nombre` asigna a una L_acc.

    La inversa exacta de `lacc_de_flujo`: L_line = 10^((log L_acc - b)/a), F = L/(4 pi d^2),
    enrojecido con la misma ley y la misma A_V.
    """
    r = relacion(nombre)
    l_line = 10.0 ** ((math.log10(float(lacc_lsun)) - r["b"]) / r["a"])
    f_dered = l_line * L_SUN_ERG_S / luminosity_erg_s(1.0, distancia_pc)
    return f_dered * 10.0 ** (-0.4 * k_lambda(ext, rest_A) * float(av))


def log_l_sobre_lbol(flujo_cgs, distancia_pc, log_lbol) -> float:
    """log(L_line/L_bol) de un flujo ya desenrojecido (o sin extincion), en erg s^-1 cm^-2."""
    return math.log10(luminosity_erg_s(float(flujo_cgs), distancia_pc) / L_SUN_ERG_S) - float(log_lbol)


def conversiones(filas_g2, *, av, distancia_pc, unidad_cgs, ext, lacc_ha, log_lbol) -> dict:
    """Una entrada por linea de `ALCALA17` (sin Halpha) que G2 limita: el limite y su L_acc.

    `filas_g2` son las filas de `g2_line_measurements.csv` por nombre; los flujos estan en la
    unidad del producto (`unidad_cgs`). `lacc_ha` es la L_acc de Halpha de referencia (L_sol).
    """
    out = {}
    for nombre, rel in ALCALA17.items():
        fila = filas_g2.get(nombre)
        if nombre == "Halpha" or fila is None:
            continue
        ul = fila.get("flux_upper_limit_99")
        if ul in (None, "") or not math.isfinite(float(ul)):
            continue
        rest = float(fila["rest_A"])
        c = lacc_de_flujo(float(ul), rest, av=av, distancia_pc=distancia_pc, nombre=nombre, ext=ext,
                          unidad_cgs=unidad_cgs)
        ul_cgs = float(ul) * float(unidad_cgs)
        pred = flujo_predicho(lacc_ha, rest, av=av, distancia_pc=distancia_pc, nombre=nombre, ext=ext)
        out[nombre] = {
            "family": fila.get("family", ""),
            "status": fila.get("status", ""),
            "rest_A": rest,
            "z_emp": float(fila["z_emp"]),
            "ul99_cgs": ul_cgs,
            "a_lambda": k_lambda(ext, rest) * float(av),
            "l_line_lsun": c["l_line_lsun"],
            "lacc_ul": c["l_acc_lsun"],
            "lacc_over_ha": c["l_acc_lsun"] / float(lacc_ha),
            "flujo_predicho_cgs": pred,
            "ul_over_pred": ul_cgs / pred,
            "log_lacc_lbol": math.log10(c["l_acc_lsun"]) - float(log_lbol),
            "log_lline_lbol": math.log10(c["l_line_lsun"]) - float(log_lbol),
            "recomendada": bool(rel["recomendada"]),
            "sigma_dex": float(rel["sigma"]),
            "relacion": relacion(nombre)["citation"],
        }
    return out


def _rango(conv, nombres, campo):
    vals = {n: conv[n][campo] for n in nombres if n in conv}
    if not vals:
        return None
    lo, hi = min(vals, key=vals.get), max(vals, key=vals.get)
    return {"min": vals[lo], "linea_min": lo, "max": vals[hi], "linea_max": hi}


def cifras(conv: dict, *, deteccion: bool) -> dict:
    """Las cifras del texto a partir de `conversiones`."""
    dentro = [n for n in conv if n not in FUERA_DEL_RANGO]
    out = {
        "lacc_sobre_ha": _rango(conv, dentro, "lacc_over_ha"),
        "lacc_sobre_ha_conclusiones": _rango(
            conv, [n for n in dentro if conv[n]["family"] in FAMILIAS_CONCLUSIONES], "lacc_over_ha"),
        "fuera_del_rango": {n: {"lacc_over_ha": conv[n]["lacc_over_ha"], "motivo": m}
                            for n, m in FUERA_DEL_RANGO.items() if n in conv},
        "referencia_halpha": "deteccion" if deteccion else "limite",
    }
    if deteccion:
        out["limite_sobre_prediccion"] = _rango(conv, LINEAS_PREDICCION, "ul_over_pred")
        if LINEA_FACTOR in conv:
            out["factor_" + LINEA_FACTOR] = conv[LINEA_FACTOR]["ul_over_pred"]
    return out


def comparacion_manara(conv: dict, manara: dict, *, lineas=CA_II) -> dict:
    """log(L_line/L_bol) del compañero (limite, desenrojecido) menos el de cada Clase III.

    Positivo: el limite publicado de la Clase III es mas profundo que el nuestro.
    """
    filas, difs = {}, []
    for est, d in manara["objetos"].items():
        for ln in lineas:
            if ln not in conv or ln not in d["lineas"]:
                continue
            ref = d["lineas"][ln]
            theirs = log_l_sobre_lbol(ref["flujo_cgs"], d["d_pc"], d["log_lstar"])
            ours = conv[ln]["log_lline_lbol"]
            filas.setdefault(est, {})[ln] = {"clase3": theirs, "clase3_limite": bool(ref["limite"]),
                                             "companero": ours, "diferencia_dex": ours - theirs}
            difs.append((ours - theirs, est, ln))
    if not difs:
        return {"filas": filas, "rango": None}
    lo, hi = min(difs), max(difs)
    return {"filas": filas, "rango": {"min": lo[0], "en_min": f"{lo[1]} {lo[2]}",
                                      "max": hi[0], "en_max": f"{hi[1]} {hi[2]}"}}


# --------------------------------------------------------------------------- lectura y salida

def _comprueba_relacion_halpha(cfg: dict) -> None:
    a, b = float(cfg["h03_lacc_lha_a"]), float(cfg["h03_lacc_lha_b"])
    if (a, b) != (ALCALA17["Halpha"]["a"], ALCALA17["Halpha"]["b"]):
        raise SystemExit(f"el run declara la relacion de Halpha a={a}, b={b} y ALCALA17 la da "
                         f"a={ALCALA17['Halpha']['a']}, b={ALCALA17['Halpha']['b']}: no son la misma tabla")


def objeto(obj, manara: dict) -> dict:
    """Todo lo de un objeto del paper (`scripts.paper_figures.Objeto`)."""
    from musepipe.io import resolve_flux_unit
    from musepipe.targets import load_target

    cfg = obj.config
    _comprueba_relacion_halpha(cfg)
    g2_qc = obj.qc("stage_g2_qc.json")
    if g2_qc.get("rule") != "spec_G2_v4":
        raise SystemExit(f"{obj.run_id}: G2 no es la v4 (rule={g2_qc.get('rule')!r}); sin `flux_upper_limit_99`")
    filas = {r["name"]: r for r in obj.filas("g2_line_measurements.csv")}
    g3 = obj.qc("stage_g3_qc.json")["combined_accretion"]
    lacc_ha = float(g3["l_acc_lsun"])
    deteccion = g3.get("kind") == "detection"
    ficha = load_target(obj.slug, project_root=ROOT) or {}
    fisica = ficha.get("fisica") or {}
    if "log_lbol" not in fisica:
        raise SystemExit(f"targets/{obj.slug}.json no declara fisica.log_lbol")
    unidad, fuente_unidad = resolve_flux_unit(cfg)
    av, d = float(cfg["h03_av"]), float(cfg["h03_distance_pc"])
    conv = conversiones(filas, av=av, distancia_pc=d, unidad_cgs=unidad, ext=extincion_del_run(cfg),
                        lacc_ha=lacc_ha, log_lbol=float(fisica["log_lbol"]))
    return {
        "objeto": obj.nombre, "run": obj.run_id,
        "entradas": {"av": av, "av_fuente": cfg.get("h03_av_source"), "distancia_pc": d,
                     "unidad_cgs": unidad, "unidad_fuente": fuente_unidad,
                     "lacc_halpha_lsun": lacc_ha, "lacc_halpha": g3.get("kind"),
                     "lacc_halpha_fuente": "stage_g3_qc.json -> combined_accretion",
                     "log_lbol": float(fisica["log_lbol"]), "log_lbol_fuente": fisica.get("log_lbol_fuente"),
                     "g2": "tables/g2_line_measurements.csv (flux_upper_limit_99, spec G2 v4)"},
        "lineas": conv,
        "cifras": cifras(conv, deteccion=deteccion),
        "manara2013": comparacion_manara(conv, manara),
    }


def _imprime(res: dict) -> None:
    e = res["entradas"]
    print(f"% ---- {res['objeto']} ({res['run']}) ----")
    print(f"% A_V {e['av']}, d {e['distancia_pc']} pc, L_acc(Halpha) {e['lacc_halpha_lsun']:.4g} L_sol "
          f"({e['lacc_halpha']}, G3), log L_bol {e['log_lbol']}")
    print(f"{'linea':11s} {'UL99 (cgs)':>10s} {'z_emp':>6s} {'L_acc <':>9s} {'x Halpha':>8s} "
          f"{'UL/pred':>8s} {'log Lacc/Lbol':>13s} {'log L/Lbol':>10s}  •")
    for n, v in res["lineas"].items():
        print(f"{n:11s} {v['ul99_cgs']:10.3e} {v['z_emp']:+6.2f} {v['lacc_ul']:9.3e} {v['lacc_over_ha']:8.2f} "
              f"{v['ul_over_pred']:8.2f} {v['log_lacc_lbol']:13.3f} {v['log_lline_lbol']:10.3f}  "
              f"{'•' if v['recomendada'] else ''}")
    c = res["cifras"]
    r = c["lacc_sobre_ha"]
    if c["referencia_halpha"] == "deteccion":
        print(f"L_acc(limite)/L_acc(Halpha): {r['min']:.2f} ({r['linea_min']}) a {r['max']:.2f} ({r['linea_max']})"
              f"  -> «{r['min']:.0f} to {r['max']:.0f} times»")
    else:
        print(f"L_acc(limite)/L_acc(Halpha, limite): al menos {r['min']:.2f} ({r['linea_min']})"
              f"  -> «at least {math.floor(r['min'])} times»")
    rc = c["lacc_sobre_ha_conclusiones"]
    if rc:
        print(f"  solo He I, O I y Ca II (frase de las conclusiones): {rc['min']:.2f} ({rc['linea_min']}) "
              f"a {rc['max']:.2f} ({rc['linea_max']})")
    for n, v in c["fuera_del_rango"].items():
        print(f"  fuera del rango: {n} x{v['lacc_over_ha']:.0f} ({v['motivo']})")
    if "limite_sobre_prediccion" in c:
        p = c["limite_sobre_prediccion"]
        print(f"limite / flujo predicho por la acrecion ({', '.join(LINEAS_PREDICCION)}): {p['min']:.2f} "
              f"({p['linea_min']}) a {p['max']:.2f} ({p['linea_max']})  -> «{p['min']:.0f}-{p['max']:.0f} times»")
        f = c["factor_" + LINEA_FACTOR]
        print(f"  {LINEA_FACTOR}: {f:.2f}  -> «a factor of {f:.0f}»")
    m = res["manara2013"]
    if m["rango"]:
        print("Ca II en log(L_line/L_bol): compañero (limite 99 %) - Clase III de Manara et al. 2013 (limite 3 sigma)")
        for est, filas in m["filas"].items():
            print("  " + est + ": " + ", ".join(
                f"{ln} {v['companero']:.2f} vs {'<' if v['clase3_limite'] else ''}{v['clase3']:.2f} "
                f"({v['diferencia_dex']:+.2f})" for ln, v in filas.items()))
        print(f"  rango {m['rango']['min']:.2f} ({m['rango']['en_min']}) a {m['rango']['max']:.2f} "
              f"({m['rango']['en_max']})  -> «{m['rango']['min']:.1f}-{m['rango']['max']:.1f} dex deeper»")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--project-root", default=None,
                    help="raiz que contiene runs/ (desde un worktree, el clon que los tiene)")
    ap.add_argument("--json", default=None, help="escribe aqui todo lo calculado")
    args = ap.parse_args(argv)

    import scripts.paper_figures as pf
    if args.project_root:
        pf.PROYECTO = Path(args.project_root).resolve()
    manara = json.loads(MANARA13.read_text())
    salida = {}
    for obj in pf.objetos_del_paper():
        res = objeto(obj, manara)
        salida[obj.slug] = res
        _imprime(res)
    if args.json:
        Path(args.json).write_text(json.dumps(salida, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
