#!/usr/bin/env python3
"""Re-corre la cadena C1 → G5 de un run, en serie y con bitácora.

    python scripts/rerun_chain.py --run-id ROXs12b_realigned
    python scripts/rerun_chain.py --run-id ROXs12b_realigned --desde D2 --hasta G5
    python scripts/rerun_chain.py --run-id ROXs12b_realigned --solo C1,F1 --dry-run

**En serie a propósito**: nunca dos etapas escribiendo el mismo run a la vez
(`AGENTS.md` / `CLAUDE.md`). Cada etapa se lanza como un proceso aparte, con lo
que un fallo de una no se lleva por delante a las demás ni deja el intérprete
con estado a medias.

Escribe `rerun_log.json` en el directorio del run (`runs/<RUN>/logs/`): por
etapa, `rc`, segundos y si el QC se reescribió — que es la única prueba de que
la etapa hizo algo. Para parar limpio entre etapas: crear el fichero `PARAR`
junto a la bitácora.

Sale con código ≠ 0 si alguna etapa falla, para poder encadenarlo.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from musepipe.stage_registry import by_id  # noqa: E402

PY = sys.executable

#: La cadena, en orden. `(id, comando, QC que tiene que reescribirse)`.
#: El id es el del **registro de etapas**, que es la fuente de verdad: si una
#: etapa se renombra ahí, `tests/test_rerun_chain.py` lo caza antes que una
#: corrida de madrugada.
CADENA: tuple[tuple[str, tuple[str, ...], str], ...] = (
    ("C1", (PY, "-m", "musepipe.stages.stage_e01_psf", "--run-id", "{run}"),
     "stages/stage_e01_qc.json"),
    ("04b", (PY, "-m", "musepipe.stages.stage04b_local_surface", "--run-id", "{run}"),
     "stages/stage04b_qc.json"),
    ("C2", (PY, "-m", "musepipe.stages.stage_x01_aperture", "--run-id", "{run}"),
     "stages/spec_aperture_qc.json"),
    ("C3", (PY, "-m", "musepipe.stages.stage_x02_optimal", "--run-id", "{run}"),
     "stages/spec_optimal_qc.json"),
    ("C4", (PY, "-m", "musepipe.stages.stage_x03_psffit", "--run-id", "{run}"),
     "stages/spec_psffit_qc.json"),
    ("C5", (PY, "-m", "musepipe.stages.stage_x04_sgf", "--run-id", "{run}"),
     "stages/spec_sgf_qc.json"),
    ("C6", (PY, "-m", "musepipe.stages.stage_x05_lpm", "--run-id", "{run}"),
     "stages/spec_lpm_qc.json"),
    ("D1", (PY, "-m", "musepipe.stages.stage_x10_compare", "--run-id", "{run}"),
     "stages/stage_x10_qc.json"),
    ("D2", (PY, "-m", "musepipe.stages.stage_x11_calibrate", "--run-id", "{run}"),
     "stages/stage_x11_qc.json"),
    ("E1", (PY, "-m", "musepipe.stages.stage_h01_detect", "--run-id", "{run}"),
     "stages/stage_h01_qc.json"),
    ("E1b", (PY, "-m", "musepipe.stages.stage_h01b_fovmap", "--run-id", "{run}"),
     "stages/stage_h01b_qc.json"),
    ("E2", (PY, "-m", "musepipe.stages.stage_h02_artifacts", "--run-id", "{run}"),
     "stages/stage_h02_qc.json"),
    ("E3", (PY, "-m", "musepipe.stages.stage_h03_limits", "--run-id", "{run}"),
     "stages/stage_h03_qc.json"),
    ("E4", (PY, "-m", "musepipe.stages.stage_h04_injection", "--run-id", "{run}"),
     "stages/stage_h04_qc.json"),
    ("E5", (PY, "-m", "musepipe.stages.stage_h05_contrast", "--run-id", "{run}"),
     "stages/stage_h05_qc.json"),
    ("E6", (PY, "-m", "musepipe.stages.stage_h06_roc", "--run-id", "{run}"),
     "stages/stage_h06_qc.json"),
    ("F1", (PY, "scripts/build_report.py", "--run-id", "{run}"),
     "report/run_summary.json"),
    ("G0", (PY, "scripts/run_g0.py", "--run-id", "{run}"),
     "stages/stage_g0_qc.json"),
    ("G1", (PY, "scripts/run_g1.py", "--run-id", "{run}"),
     "stages/stage_g1_qc.json"),
    ("G2", (PY, "-c", "from musepipe.stages.stage_g2_measure_lines import run_stage_g2;"
                      " print(run_stage_g2('{run}'))"),
     "stages/stage_g2_qc.json"),
    # G3: aquí va la rodaja de acreción. `run_stage_g3_all` exige la
    # configuración de «G3 real» (`g3_atmo_av_axis`, plantillas, atmósferas) y
    # revienta con KeyError antes de tocar nada en un run que no la tenga.
    ("G3", (PY, "-c", "from musepipe.stages.stage_g3_accretion import run_stage_g3_accretion;"
                      " print(run_stage_g3_accretion('{run}'))"),
     "stages/stage_g3_qc.json"),
    ("G4", (PY, "-c", "from musepipe.stages.stage_g4_classify import run_stage_g4;"
                      " print(run_stage_g4('{run}'))"),
     "stages/stage_g4_classification.json"),
    ("G5", (PY, "scripts/build_characterization.py", "--run-id", "{run}"),
     "report/characterization/characterization_summary.json"),
)

IDS = tuple(etapa for etapa, _cmd, _qc in CADENA)


def qc_del_registro(etapa: str) -> tuple[str, ...]:
    """Los QC que el registro acepta para una etapa: el suyo y sus alias."""
    spec = by_id(etapa)
    return tuple(q for q in ((spec.qc,) + tuple(spec.qc_aliases)) if q)


def selecciona(desde: str | None, hasta: str | None, solo: str | None):
    """El tramo de cadena que se va a correr."""
    if solo:
        quiere = {s.strip() for s in solo.split(",") if s.strip()}
        desconocidas = quiere - set(IDS)
        if desconocidas:
            raise SystemExit(f"etapas desconocidas: {sorted(desconocidas)}; hay {list(IDS)}")
        return [e for e in CADENA if e[0] in quiere]
    i0 = IDS.index(desde) if desde else 0
    i1 = IDS.index(hasta) + 1 if hasta else len(CADENA)
    if i1 <= i0:
        raise SystemExit(f"--desde {desde} va después de --hasta {hasta}")
    return list(CADENA[i0:i1])


def comprueba_run(run_dir: Path, run_id: str) -> None:
    """Un run legacy es de solo lectura: parar ANTES de lanzar nada."""
    cfg = run_dir / "config" / "config.json"
    if not cfg.exists():
        raise SystemExit(f"no encuentro {cfg}")
    meta = json.loads(cfg.read_text(encoding="utf-8")).get("meta", {})
    if meta.get("legacy") is True:
        raise SystemExit(
            f"{run_id} está marcado `legacy: true` en su config: es un run histórico y no se"
            " reescribe. Si de verdad hay que re-correrlo, quita la marca a mano y asume lo"
            " que significa.")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--desde", default=None, choices=IDS, metavar="ETAPA")
    ap.add_argument("--hasta", default=None, choices=IDS, metavar="ETAPA")
    ap.add_argument("--solo", default=None, help="lista separada por comas")
    ap.add_argument("--dry-run", action="store_true", help="imprime el plan y no lanza nada")
    ap.add_argument("--project-root", default=str(ROOT))
    args = ap.parse_args(argv)

    raiz = Path(args.project_root).resolve()
    run_dir = raiz / "runs" / args.run_id
    sel = selecciona(args.desde, args.hasta, args.solo)
    comprueba_run(run_dir, args.run_id)

    logs = run_dir / "logs"
    parar = logs / "PARAR"
    print(f"cadena de {args.run_id}: {len(sel)} etapas ({sel[0][0]} → {sel[-1][0]})", flush=True)
    if args.dry_run:
        for etapa, cmd, qc in sel:
            print(f"  {etapa:4s} {qc:52s} {' '.join(cmd).replace('{run}', args.run_id)[:70]}")
        print("\n--dry-run: no se ha lanzado nada.")
        return 0

    logs.mkdir(parents=True, exist_ok=True)
    bitacora, t_total, fallos = [], time.time(), 0
    for etapa, cmd, qc_rel in sel:
        if parar.exists():
            print(f"{parar.name} presente: parada limpia antes de {etapa}.", flush=True)
            break
        qc = run_dir / qc_rel
        antes = qc.stat().st_mtime if qc.exists() else 0.0
        t0 = time.time()
        proc = subprocess.run([c.replace("{run}", args.run_id) for c in cmd],
                              cwd=raiz, capture_output=True, text=True)
        dt = time.time() - t0
        reescrito = qc.exists() and qc.stat().st_mtime > antes
        fila = {"etapa": etapa, "rc": proc.returncode, "segundos": round(dt, 1),
                "qc_reescrito": bool(reescrito), "qc": qc_rel}
        if proc.returncode != 0:
            fallos += 1
            fila["stderr"] = proc.stderr[-2000:]
        bitacora.append(fila)
        marca = ("OK " if proc.returncode == 0 and reescrito
                 else "rc!=0" if proc.returncode else "SIN QC")
        print(f"  {etapa:4s} {marca:6s} {dt:7.1f}s   {qc_rel}", flush=True)
        if proc.returncode != 0 and proc.stderr.strip():
            print("      " + proc.stderr.strip().splitlines()[-1][:200], flush=True)
        (logs / "rerun_log.json").write_text(
            json.dumps(bitacora, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")

    ok = sum(1 for f in bitacora if f["rc"] == 0)
    print(f"\ntotal {time.time() - t_total:.0f}s; {ok} de {len(bitacora)} con rc=0"
          f"  ·  bitácora en {logs / 'rerun_log.json'}", flush=True)
    return 1 if fallos else 0


if __name__ == "__main__":
    raise SystemExit(main())
