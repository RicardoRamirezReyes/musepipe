#!/usr/bin/env python3
"""Instantanea de los numeros de un run, para poder RESTAR despues de re-correr.

    python scripts/instantanea_run.py antes.json --runs <RUN> [<RUN> ...]
    ... re-corre lo que sea ...
    python scripts/instantanea_run.py despues.json --runs ROXs12b_realigned
    python scripts/instantanea_run.py --compara antes.json despues.json

Por que existe
--------------
Re-correr una cadena entera y que NADA cambie es un resultado -dice que el
producto en disco es reproducible con el codigo de hoy-, pero solo si el estado
anterior se guardo ANTES. El 2026-09-17 se re-corrio C1->G5 en los dos objetos y
la instantanea previa no leyo los espectros: los abrio como imagen cuando son
BINTABLE, y hubo que reconstruir el "antes" desde copias de otros runs. Por eso
`_espectro` acepta las dos formas y por eso hay un test.

Guarda, por run: firma sha256 del flujo de cada espectro mas su mediana, su
continuo y su Halpha; el sha del modelo de PSF; y los escalares que mas se citan
de E4, E3, E1, D1, G2-G4 y F1.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from astropy.io import fits

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

HA = (6555.0, 6575.0)
CONT = (6500.0, 6540.0)


def _wave_de_cabecera(hdr, n):
    c1 = float(hdr.get("CRVAL1", 0.0))
    d = float(hdr.get("CDELT1", hdr.get("CD1_1", 1.0)))
    p = float(hdr.get("CRPIX1", 1.0))
    return c1 + (np.arange(n) + 1 - p) * d


def _flujo_y_lambda(hdul):
    """(flujo, lambda) de un producto espectral, sea BINTABLE o imagen.

    Los espectros de la cadena son BINTABLE con columnas `flux`/`wave_A`
    (`SpectrumProduct`), pero los productos viejos y algunos auxiliares son
    imagenes 1D con la solucion en la cabecera. Leer solo una de las dos formas
    es lo que dejo sin "antes" al re-corrido del 2026-09-17.
    """
    for h in hdul:
        cols = getattr(h, "columns", None)
        if cols is not None and h.data is not None:
            nombres = {c.lower() for c in cols.names}
            if "flux" in nombres:
                flujo = np.asarray(h.data["flux"], dtype=np.float64)
                if "wave_a" in nombres:
                    wave = np.asarray(h.data[cols.names[[c.lower() for c in cols.names].index("wave_a")]],
                                      dtype=np.float64)
                else:
                    wave = _wave_de_cabecera(h.header, flujo.size)
                return flujo, wave
    for h in hdul:
        if h.data is not None and getattr(h.data, "ndim", 0) >= 1 and getattr(h, "columns", None) is None:
            dat = np.asarray(h.data, dtype=np.float64)
            flujo = dat[0] if dat.ndim > 1 else dat
            return flujo, _wave_de_cabecera(h.header, flujo.size)
    raise ValueError("el fichero no tiene ni tabla con `flux` ni imagen 1D")


def espectro(path):
    try:
        with fits.open(path) as hdul:
            flujo, wave = _flujo_y_lambda(hdul)
            bunit = str(hdul[0].header.get("BUNIT", "") or
                        (hdul[1].header.get("BUNIT", "") if len(hdul) > 1 else ""))
            wframe = str(hdul[0].header.get("WFRAME", "") or
                         (hdul[1].header.get("WFRAME", "") if len(hdul) > 1 else ""))
        m_ha = (wave >= HA[0]) & (wave <= HA[1])
        m_c = (wave >= CONT[0]) & (wave <= CONT[1])
        return {"n": int(flujo.size),
                "sha256": hashlib.sha256(np.ascontiguousarray(flujo, "<f8").tobytes()).hexdigest()[:16],
                "mediana": float(np.nanmedian(flujo)),
                "continuo_6500_6540": float(np.nansum(flujo[m_c])) if m_c.any() else None,
                "halpha_6555_6575": float(np.nansum(flujo[m_ha])) if m_ha.any() else None,
                "bunit": bunit, "wframe": wframe,
                "mtime": Path(path).stat().st_mtime}
    except Exception as e:  # noqa: BLE001
        return {"error": f"{type(e).__name__}: {e}"}


def _jq(path):
    try:
        return json.loads(Path(path).read_text())
    except Exception:  # noqa: BLE001
        return None


def del_run(run, root=ROOT):
    R = Path(root) / "runs" / run
    S = R / "stages"
    d = {"espectros": {}, "qc": {}}
    for p in sorted(S.glob("spec_*.fits")):
        if "controls" in p.name:
            continue
        d["espectros"][p.name] = espectro(p)
    pm = S / "psf_model.json"
    if pm.exists():
        texto = pm.read_text()
        d["psf_model"] = {"sha256": hashlib.sha256(texto.encode()).hexdigest()[:16],
                          "mtime": pm.stat().st_mtime, "bytes": len(texto)}
    e4 = _jq(S / "stage_h04_qc.json")
    if e4:
        d["qc"]["E4_throughput"] = {k: v.get("throughput") for k, v in
                                    ((e4.get("throughput") or {}).get("per_method_at_snr5") or {}).items()}
        d["qc"]["E4_bias_pct"] = (e4.get("bias") or {}).get("flux_pct_at_snr5")
    e3 = _jq(S / "stage_h03_qc.json")
    if e3:
        d["qc"]["E3"] = {L["method"]: {k: L.get(k) for k in ("throughput", "f_lim_dereddened", "mdot")}
                         for L in e3.get("limits", [])}
    e1 = _jq(S / "stage_h01_qc.json")
    if e1:
        d["qc"]["E1_verdict"] = e1.get("verdict") or e1.get("status")
    d1 = _jq(S / "stage_x10_qc.json")
    if d1:
        d["qc"]["D1_status"] = d1.get("status")
    g3 = _jq(S / "stage_g3_qc.json")
    if g3:
        d["qc"]["G3"] = {k: g3.get(k) for k in ("status", "mdot", "l_acc", "spt", "teff", "mass") if k in g3}
    g4 = _jq(S / "stage_g4_classification.json")
    if g4:
        d["qc"]["G4"] = {k: g4.get(k) for k in ("status", "classification", "verdict") if k in g4}
    f1 = _jq(R / "report" / "run_summary.json")
    if f1:
        d["qc"]["F1"] = {"overall_status": f1.get("overall_status"),
                         "prioridades": dict(Counter(i.get("priority") for i in f1.get("open_issues", []))),
                         # la LISTA, no solo el recuento: el 2026-09-17 un `major`
                         # de 42B b paso de 24 a 23 y no se pudo nombrar.
                         "issues": [f"{i.get('priority')}|{i.get('stage')}|{str(i.get('issue'))[:120]}"
                                    for i in f1.get("open_issues", [])],
                         "etapas": {s.get("stage"): s.get("status") for s in f1.get("stages", [])}}
    return d


def compara(antes, despues):
    """Lineas de texto con lo que cambio entre dos instantaneas."""
    a = json.loads(Path(antes).read_text())
    b = json.loads(Path(despues).read_text())
    lineas = []
    for run in sorted(set(a) | set(b)):
        lineas.append(f"== {run}")
        ea, eb = a.get(run, {}).get("espectros", {}), b.get(run, {}).get("espectros", {})
        for n in sorted(set(ea) | set(eb)):
            A, B = ea.get(n, {}), eb.get(n, {})
            if "error" in A or "error" in B or not A or not B:
                lineas.append(f"   {n:42s} no comparable ({A.get('error') or B.get('error') or 'falta'})")
            elif A["sha256"] == B["sha256"]:
                lineas.append(f"   {n:42s} IDENTICO")
            else:
                def pct(k):
                    x, y = A.get(k), B.get(k)
                    return f"{100 * (y / x - 1):+.3f} %" if x else "n/d"
                lineas.append(f"   {n:42s} cambia  cont {pct('continuo_6500_6540')}  Ha {pct('halpha_6555_6575')}")
        qa, qb = a.get(run, {}).get("qc", {}), b.get(run, {}).get("qc", {})
        for clave in sorted(set(qa) | set(qb)):
            if qa.get(clave) != qb.get(clave):
                lineas.append(f"   QC {clave}: {json.dumps(qa.get(clave))[:120]} -> {json.dumps(qb.get(clave))[:120]}")
        ia = set((qa.get("F1") or {}).get("issues") or [])
        ib = set((qb.get("F1") or {}).get("issues") or [])
        for x in sorted(ia - ib):
            lineas.append(f"   F1 issue que DESAPARECE: {x}")
        for x in sorted(ib - ia):
            lineas.append(f"   F1 issue NUEVO: {x}")
    return lineas


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("salida", nargs="?", help="fichero JSON a escribir")
    # Sin valor por defecto a proposito: el run se declara siempre (CLAUDE.md,
    # "Run selection"), y `tests/test_no_hardcoded_target.py` prohibe el literal.
    ap.add_argument("--runs", nargs="+", metavar="RUN",
                    help="runs de los que tomar la instantanea")
    ap.add_argument("--project-root", default=str(ROOT))
    ap.add_argument("--compara", nargs=2, metavar=("ANTES", "DESPUES"))
    a = ap.parse_args(argv)
    if a.compara:
        for linea in compara(*a.compara):
            print(linea)
        return 0
    if not a.salida:
        ap.error("hace falta el fichero de salida (o --compara)")
    if not a.runs:
        ap.error("hace falta --runs con al menos un run")
    todo = {run: del_run(run, a.project_root) for run in a.runs}
    Path(a.salida).write_text(json.dumps(todo, indent=1, sort_keys=True))
    print(f"escrito {a.salida}")
    for run in a.runs:
        q = todo[run]["qc"]
        print(f"  {run}: {len(todo[run]['espectros'])} espectros | F1 "
              f"{(q.get('F1') or {}).get('overall_status')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
