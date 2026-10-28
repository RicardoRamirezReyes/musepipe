#!/usr/bin/env python
"""Cuánto más bajo es el ruido en las posiciones de inyección de E4 que en el compañero («7 and 3»).

Solo lectura. Reproduce la frase del Apéndice A del paper (cabecera de procedencia de
`paper/main.tex`, decisión F del 2026-10-26), que salió de
`~/muse_sesion_20261025/paper_c0/a5/e4_ruido.py` con el estimador de la auditoría del 10-13:

    «completeness is evaluated at the 11 and 35 injection positions of the two targets, where
    the noise in these units is 7 and 3 times lower»

La escala de inyección de E4 (`input_snr`) es el flujo total de la línea en unidades de la σ del
filtro adaptado EN EL COMPAÑERO (`injection_scale.sigma_flux` del QC de E4); la completitud se
mide en las posiciones de control, cuya σ es la de las nulas de inyección del estrato
`psffit|1|none` (`snr_standardization.scale_by_stratum`). El cociente de las dos es el número
del paper: 7.18 (ROXs 12 b) y 3.36 (ROXs 42B b) con C4 v2; 8.51 y 3.10 en las copias de antes
de c0.

De la tabla `injection_throughput_by_method.csv` saca también el número de posiciones de
inyección (filas nominales, sin la posición real) y la completitud a `input_snr` = 5 por
método (la frase «whose completeness at an injected scale of 5 is 1.00 for all six schemes»).

`RUN@CARPETA` lee una copia con la disposición de un run (`CARPETA/stages`, `CARPETA/tables`),
p. ej. `runs/<RUN>/stages/_antes_c4_c0_*`.

uso:
    python scripts/e4_ruido_en_el_companero.py --project-root RAIZ [--run-id RUN[@COPIA] ...] [--json SALIDA]
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

ESTRATO = "psffit|1|none"
ESCALA_COMPLETITUD = 5.0


def cociente_de_ruido(qc_h04, estrato=ESTRATO):
    """(σ en el compañero, σ de las nulas del estrato, cociente) del QC de E4."""
    sigma_comp = float(qc_h04["injection_scale"]["sigma_flux"])
    sigma_nulas = float(qc_h04["snr_standardization"]["scale_by_stratum"][estrato]["sigma"])
    return sigma_comp, sigma_nulas, sigma_comp / sigma_nulas


def posiciones_y_completitud(filas, escala=ESCALA_COMPLETITUD):
    """(n de posiciones de inyección, completitud a `escala` por método) de las filas nominales."""
    nominal = [f for f in filas if f["variant"] == "nominal" and f["position_label"] != "real"]
    posiciones = {f["position_label"] for f in nominal}
    por_metodo = {}
    for f in nominal:
        if float(f["input_snr"]) == float(escala):
            por_metodo.setdefault(f["method"], []).append(str(f["complete"]) == "True")
    return len(posiciones), {m: sum(v) / len(v) for m, v in sorted(por_metodo.items())}


def lee(run_id, project_root, copia=None):
    from musepipe.stages.stage_h04_injection import stage_h04_paths

    p = stage_h04_paths(run_id, project_root)
    qc_path, tabla = Path(p["stage_h04_qc_json"]), Path(p["throughput_csv"])
    if copia is not None:
        qc_path = copia / "stages" / qc_path.name
        tabla = copia / "tables" / tabla.name
    qc = json.loads(qc_path.read_text(encoding="utf-8"))
    with open(tabla, newline="", encoding="utf-8") as fh:
        filas = list(csv.DictReader(fh))
    s_comp, s_nulas, cociente = cociente_de_ruido(qc)
    n_pos, completitud = posiciones_y_completitud(filas)
    return {"run_id": run_id, "copia": str(copia) if copia else None, "qc": str(qc_path),
            "sigma_flux_companero": s_comp, "sigma_nulas": s_nulas, "estrato": ESTRATO, "cociente": cociente,
            "posiciones_de_inyeccion": n_pos, f"completitud_a_{ESCALA_COMPLETITUD:g}": completitud}


def _objetivo(texto, project_root):
    run, _, copia = str(texto).partition("@")
    if not copia:
        return run, None
    c = Path(copia)
    return run, (c if c.is_absolute() else project_root / c).resolve()


def _runs_del_paper():
    fichas = []
    for f in sorted((ROOT / "targets").glob("*.json")):
        p = json.loads(f.read_text(encoding="utf-8")).get("paper")
        if p:
            fichas.append((int(p.get("order", 99)), p["run"]))
    return [r for _o, r in sorted(fichas)]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--run-id", action="append", default=None, metavar="RUN[@COPIA]",
                    help="repetible; por defecto, el run del paper de cada objeto (targets/*.json)")
    ap.add_argument("--project-root", default=str(ROOT), help="clon con runs/ (desde un worktree, el principal)")
    ap.add_argument("--json", default=None)
    a = ap.parse_args(argv)
    root = Path(a.project_root).resolve()
    res = []
    for texto in a.run_id or _runs_del_paper():
        run, copia = _objetivo(texto, root)
        res.append(lee(run, root, copia))
    for r in res:
        print(f"== {r['run_id']}" + (f" @ {r['copia']}" if r["copia"] else "")
              + f": sigma_flux {r['sigma_flux_companero']:.1f} / sigma nulas {r['estrato']} {r['sigma_nulas']:.1f}"
              f" = {r['cociente']:.2f}x; {r['posiciones_de_inyeccion']} posiciones de inyección; completitud a "
              f"{ESCALA_COMPLETITUD:g}: " + ", ".join(f"{m} {v:.2f}" for m, v in
                                                      r[f"completitud_a_{ESCALA_COMPLETITUD:g}"].items()))
    if a.json:
        Path(a.json).write_text(json.dumps(res, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
