#!/usr/bin/env python
"""Bitacora de observacion desde las cabeceras crudas, para la tabla del paper.

Lee el `observation_plan.json` de cada run (que lista las exposiciones y sus ficheros) y
abre la cabecera primaria de cada cubo por exposicion. NO escribe en `runs/`.

    python scripts/bitacora_observaciones.py --runs ROXs12b_realigned ROXs42Bb_realigned
    python scripts/bitacora_observaciones.py --runs <run> --latex > tabla.tex

Los runs se pasan siempre de forma explicita y el nombre a mostrar sale de
`chain.target` -> `targets/<slug>.json`, nunca de un literal en el codigo
(ver `tests/test_no_hardcoded_target.py`).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from astropy.io import fits

ROOT = Path(__file__).resolve().parents[1]


def nombre_a_mostrar(run_id):
    """El nombre del objeto sale del registro, no del codigo."""
    cfg = json.loads((ROOT / "runs" / run_id / "config" / "config.json").read_text())
    slug = (cfg.get("chain") or {}).get("target")
    if not slug:
        return run_id
    ficha = ROOT / "targets" / f"{slug}.json"
    if not ficha.exists():
        return slug
    return json.loads(ficha.read_text()).get("display_name", slug)

# Una clave por dato; si falta, se dice, no se inventa.
CLAVES = {
    "prog_id": "ESO OBS PROG ID",
    "ob_id": "ESO OBS ID",
    "ob_name": "ESO OBS NAME",
    "mode": "ESO INS MODE",
    "drs": "ESO PRO REC1 PIPE ID",
    "seeing_start": "ESO TEL AMBI FWHM START",
    "seeing_end": "ESO TEL AMBI FWHM END",
    "airm_start": "ESO TEL AIRM START",
    "airm_end": "ESO TEL AIRM END",
    "date_obs": "DATE-OBS",
    "exptime": "EXPTIME",
}


def leer(run_id):
    plan_json = ROOT / "runs" / run_id / "stages" / "observation_plan.json"
    if not plan_json.exists():
        return None, f"{run_id}: no hay observation_plan.json"
    plan = json.loads(plan_json.read_text())["plan"]
    filas, fallos = [], []
    for e in plan["exposures"]:
        p = Path(e["file"])
        if not p.exists():
            fallos.append(str(p))
            continue
        h = fits.getheader(p, 0)
        fila = {k: h.get(v) for k, v in CLAVES.items()}
        fila["exposure_id"] = e["exposure_id"]
        # La carpeta va por NOCHE (la tarde en que empieza); DATE-OBS es UT y puede
        # caer al dia siguiente. Se publican las dos y no se elige por el lector.
        fila["noche_label"] = str(e["exposure_id"]).split("_")[0]
        fila["mjd_obs"] = e.get("mjd_obs")
        fila["exptime_plan"] = e.get("exptime")
        filas.append(fila)
    return {"plan": plan, "filas": filas, "fallos": fallos}, None


def resumen_por_ob(filas):
    obs = {}
    for f in filas:
        obs.setdefault(f["ob_id"], []).append(f)
    out = []
    for ob, fs in sorted(obs.items(), key=lambda kv: min(x["date_obs"] or "" for x in kv[1])):
        see = [x for x in (list(_num(f, "seeing_start") for f in fs)
                           + list(_num(f, "seeing_end") for f in fs)) if x is not None]
        air = [x for x in (list(_num(f, "airm_start") for f in fs)
                           + list(_num(f, "airm_end") for f in fs)) if x is not None]
        exp = [f["exptime"] for f in fs if f["exptime"] is not None]
        fechas = sorted(f["date_obs"] for f in fs if f["date_obs"])
        out.append({
            "ob_id": ob,
            "ob_name": fs[0]["ob_name"],
            "noche": fs[0]["noche_label"],
            "ut_desde": fechas[0][:16] if fechas else None,
            "ut_hasta": fechas[-1][:16] if fechas else None,
            "n_exp": len(fs),
            "exptime_s": sorted(set(exp)),
            "t_total_s": float(np.sum(exp)) if exp else None,
            "seeing_arcsec": (float(np.min(see)), float(np.median(see)), float(np.max(see))) if see else None,
            "airmass": (float(np.min(air)), float(np.max(air))) if air else None,
            "mode": fs[0]["mode"], "prog_id": fs[0]["prog_id"], "drs": fs[0]["drs"],
        })
    return out


def _num(f, k):
    v = f.get(k)
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", required=True,
                    help="run_id de cada objeto; el nombre sale de targets/<slug>.json")
    ap.add_argument("--latex", action="store_true", help="emitir la tabla en LaTeX (A&A)")
    args = ap.parse_args(argv)

    todo = {}
    for run_id in args.runs:
        objeto = nombre_a_mostrar(run_id)
        datos, err = leer(run_id)
        if err:
            print(f"!! {err}")
            continue
        todo[objeto] = resumen_por_ob(datos["filas"])
        if datos["fallos"]:
            print(f"!! {objeto}: {len(datos['fallos'])} cubos no accesibles")

    if not args.latex:
        for objeto, obs in todo.items():
            print("=" * 74)
            print(objeto)
            for o in obs:
                see = o["seeing_arcsec"]
                air = o["airmass"]
                print(f"  OB {o['ob_id']} ({o['ob_name']}) · noche {o['noche']}"
                      f"  ·  UT {o['ut_desde']} a {o['ut_hasta']}")
                print(f"    n_exp={o['n_exp']}  exptime={o['exptime_s']} s  "
                      f"total={o['t_total_s']:.0f} s ({o['t_total_s']/3600:.2f} h)")
                print(f"    seeing DIMM: {see[0]:.2f}/{see[1]:.2f}/{see[2]:.2f}\" (min/med/max)"
                      if see else "    seeing: no disponible")
                print(f"    airmass: {air[0]:.3f}-{air[1]:.3f}" if air else "    airmass: no disponible")
                print(f"    modo={o['mode']}  prog={o['prog_id']}  DRS={o['drs']}")
        return 0

    print(r"\begin{tabular}{l c c c c c c}")
    print(r"\hline\hline")
    print(r"Target & OB & Night & $N_{\rm exp}$ & $T_{\rm exp}$ & Seeing & Airmass \\")
    print(r"       &    &       &               & (s)           & (\arcsec) &        \\")
    print(r"\hline")
    for objeto, obs in todo.items():
        for o in obs:
            see = o["seeing_arcsec"]
            air = o["airmass"]
            print(f"\\object{{{objeto}}} & {o['ob_id']} & {o['noche']} & {o['n_exp']} & "
                  f"{'/'.join(f'{v:.0f}' for v in o['exptime_s'])} & "
                  f"{see[1]:.2f} & {air[0]:.2f}--{air[1]:.2f} \\\\")
    print(r"\hline")
    print(r"\end{tabular}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
