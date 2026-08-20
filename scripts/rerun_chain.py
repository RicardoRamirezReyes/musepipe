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
la etapa hizo algo. La bitácora se **acumula** entre invocaciones. Para parar
limpio entre etapas: crear el fichero `PARAR` junto a la bitácora.

Ninguna etapa hace checkpoint interno: la que esté corriendo cuando se corte la
máquina pierde su trabajo y vuelve a empezar. Lo que sí sobrevive es todo lo
anterior, y `--saltar-hechos` lo aprovecha para reanudar sin repetirlo:

    python scripts/rerun_chain.py --run-id X --desde C1 --hasta E6 --saltar-hechos

Salta una etapa solo si la bitácora la da con `rc=0` y QC reescrito **y** ese
QC sigue en disco sin haber retrocedido en el tiempo. No mira si el código o
las entradas han cambiado desde entonces: es para reanudar una corrida cortada,
no para decidir qué hace falta re-correr.

Sale con código ≠ 0 si alguna etapa falla, para poder encadenarlo.
"""
from __future__ import annotations

import argparse
import datetime as _dt
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
    # E4 ANTES que E3, aunque la letra diga lo contrario: E3 divide su limite de
    # flujo por el throughput que E4 escribe en
    # `tables/injection_throughput_by_method.csv`, y E4 no lee nada de E3. Con el
    # orden por numero, E3 leia el throughput de la corrida ANTERIOR: el
    # 2026-08-20, con la extraccion nueva, eso dejo el Mdot de E3 calculado con
    # un throughput un 13-17 % mas bajo del que E4 acababa de medir. La cadena se
    # ordena por dependencia, no por nombre (`tests/test_rerun_chain.py`).
    ("E4", (PY, "-m", "musepipe.stages.stage_h04_injection", "--run-id", "{run}"),
     "stages/stage_h04_qc.json"),
    ("E3", (PY, "-m", "musepipe.stages.stage_h03_limits", "--run-id", "{run}"),
     "stages/stage_h03_qc.json"),
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


def bitacora_previa(logs: Path) -> list[dict]:
    """Las filas que dejaron invocaciones anteriores, si las hay.

    Hasta ahora la bitacora se SOBRESCRIBIA en cada invocacion, asi que
    reanudar una cadena cortada perdia el registro de lo ya hecho — justo lo
    que hace falta para saber por donde seguir. Ahora se acumula.
    """
    fichero = logs / "rerun_log.json"
    if not fichero.exists():
        return []
    try:
        datos = json.loads(fichero.read_text(encoding="utf-8"))
    except ValueError:
        return []
    return [f for f in datos if isinstance(f, dict)] if isinstance(datos, list) else []


def etapas_hechas(previas: list[dict], run_dir: Path) -> dict[str, dict]:
    """Etapa -> ultima fila que la da por terminada Y sigue respaldada por su QC.

    No basta con que la bitacora diga `rc=0`: el QC tiene que seguir en disco y
    no haber RETROCEDIDO en el tiempo. Si alguien restaura un snapshot encima
    del run, el mtime del QC vuelve atras y la etapa deja de contar como hecha,
    que es exactamente lo que se quiere — el producto que hay ya no es el que
    escribio aquella corrida.
    """
    hechas: dict[str, dict] = {}
    for fila in previas:
        if fila.get("rc") != 0 or not fila.get("qc_reescrito"):
            continue
        marca = fila.get("qc_mtime")
        qc = run_dir / str(fila.get("qc", ""))
        if marca is None or not qc.exists():
            continue
        if qc.stat().st_mtime + 1e-6 >= float(marca):
            hechas[str(fila.get("etapa"))] = fila
    return hechas


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--desde", default=None, choices=IDS, metavar="ETAPA")
    ap.add_argument("--hasta", default=None, choices=IDS, metavar="ETAPA")
    ap.add_argument("--solo", default=None, help="lista separada por comas")
    ap.add_argument("--dry-run", action="store_true", help="imprime el plan y no lanza nada")
    ap.add_argument("--saltar-hechos", action="store_true", dest="saltar_hechos",
                    help="salta las etapas que la bitacora ya da por terminadas"
                         " (rc=0, QC reescrito y ese QC todavia en disco)")
    ap.add_argument("--project-root", default=str(ROOT))
    args = ap.parse_args(argv)

    raiz = Path(args.project_root).resolve()
    run_dir = raiz / "runs" / args.run_id
    sel = selecciona(args.desde, args.hasta, args.solo)
    comprueba_run(run_dir, args.run_id)

    logs = run_dir / "logs"
    parar = logs / "PARAR"
    previas = bitacora_previa(logs)
    hechas = etapas_hechas(previas, run_dir) if args.saltar_hechos else {}
    print(f"cadena de {args.run_id}: {len(sel)} etapas ({sel[0][0]} → {sel[-1][0]})", flush=True)
    if hechas:
        ya = [e for e, _c, _q in sel if e in hechas]
        print(f"  --saltar-hechos: {len(ya)} ya en la bitácora ({', '.join(ya) or '-'})", flush=True)
    if args.dry_run:
        for etapa, cmd, qc in sel:
            marca = "SALTA " if etapa in hechas else "      "
            print(f"  {marca}{etapa:4s} {qc:52s} {' '.join(cmd).replace('{run}', args.run_id)[:70]}")
        print("\n--dry-run: no se ha lanzado nada.")
        return 0

    logs.mkdir(parents=True, exist_ok=True)
    nuevas, t_total, fallos, saltadas = [], time.time(), 0, 0

    def guarda():
        """La bitacora, con lo previo delante: se acumula, no se pisa."""
        (logs / "rerun_log.json").write_text(
            json.dumps(previas + nuevas, indent=1, ensure_ascii=False) + "\n",
            encoding="utf-8")

    for etapa, cmd, qc_rel in sel:
        if parar.exists():
            print(f"{parar.name} presente: parada limpia antes de {etapa}.", flush=True)
            break
        if etapa in hechas:
            saltadas += 1
            nuevas.append({"etapa": etapa, "rc": 0, "segundos": 0.0,
                           "qc_reescrito": False, "qc": qc_rel, "saltada": True,
                           "saltada_por": hechas[etapa].get("terminado_utc")})
            print(f"  {etapa:4s} SALTA          -   {qc_rel}", flush=True)
            guarda()
            continue
        qc = run_dir / qc_rel
        antes = qc.stat().st_mtime if qc.exists() else 0.0
        t0 = time.time()
        proc = subprocess.run([c.replace("{run}", args.run_id) for c in cmd],
                              cwd=raiz, capture_output=True, text=True)
        dt = time.time() - t0
        reescrito = qc.exists() and qc.stat().st_mtime > antes
        fila = {"etapa": etapa, "rc": proc.returncode, "segundos": round(dt, 1),
                "qc_reescrito": bool(reescrito), "qc": qc_rel,
                # Lo que `--saltar-hechos` necesita para decidir si esta fila
                # todavia describe lo que hay en disco.
                "qc_mtime": qc.stat().st_mtime if qc.exists() else None,
                "terminado_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")}
        if proc.returncode != 0:
            fallos += 1
            fila["stderr"] = proc.stderr[-2000:]
        nuevas.append(fila)
        marca = ("OK " if proc.returncode == 0 and reescrito
                 else "rc!=0" if proc.returncode else "SIN QC")
        print(f"  {etapa:4s} {marca:6s} {dt:7.1f}s   {qc_rel}", flush=True)
        if proc.returncode != 0 and proc.stderr.strip():
            print("      " + proc.stderr.strip().splitlines()[-1][:200], flush=True)
        guarda()

    corridas = [f for f in nuevas if not f.get("saltada")]
    ok = sum(1 for f in corridas if f["rc"] == 0)
    extra = f"; {saltadas} saltadas" if saltadas else ""
    print(f"\ntotal {time.time() - t_total:.0f}s; {ok} de {len(corridas)} con rc=0{extra}"
          f"  ·  bitácora en {logs / 'rerun_log.json'}", flush=True)
    return 1 if fallos else 0


if __name__ == "__main__":
    raise SystemExit(main())
