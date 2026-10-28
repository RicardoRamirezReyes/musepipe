#!/usr/bin/env python
"""Recalcula E1 (detección de Hα) y E3 (límites) de un run SIN escribir en `runs/`.

Es la herramienta de las comprobaciones del 2026-10-26 que el paper cita (cabecera de
procedencia de `paper/main.tex`, decisiones C y D), que vivían en
`~/muse_sesion_20261025/marginal_42bb/e1_sin_escribir.py` y
`~/muse_sesion_20261025/noche_e3/e3_noche_sin_escribir.py`. Sirve cuando D2 se ha re-corrido
y E1/E3 no, o para leer otro run sin tocar sus productos:

- **E1** se calcula en memoria con `compute_stage_h01_products`, sobre los espectros
  calibrados de D2 del run (o de una copia, ver abajo). No se llama a la escritura de la
  etapa sobre el run.
- **E3** se calcula con `compute_stage_h03_products` (sin `write_stage_h03_products`). E3 lee
  los productos de E1 de disco, así que el E1 recién calculado se escribe con el escritor de
  la propia etapa (`write_stage_h01_products`, para que E3 lea exactamente lo que leería en la
  cadena) en una carpeta TEMPORAL fuera del proyecto, que se borra al terminar. E2, E4, D2 y
  la config se leen del run. Con `--e1-del-disco` E3 lee el E1 que ya está en el run.

Nada se escribe en el run: antes de escribir el E1 temporal se comprueba que cada ruta de
salida, con los enlaces simbólicos resueltos, cae fuera de `runs/`, del run resuelto y de la
copia. Un run de prueba cuyos productos son enlaces a otro run (los de
`ROXs12b_fondo_barrido_20260911`, `ROXs42Bb_memtest_20260911`) no cambia nada: aquí solo se
lee de ellos.

**Copias.** `RUN@CARPETA` lee los productos de una copia con la disposición de un run
(`CARPETA/stages`, `CARPETA/tables`), p. ej. la `runs/<RUN>/stages/_antes_c4_c0_*` que deja
un re-corrido; la config sigue siendo la del run (se avisa de las claves que difieren de la
config de la copia). Así se valida la herramienta: sobre la copia de antes de c0 de la noche
buena da su QC de E3.

Cifras del paper que reproduce (2026-10-26):

- decisión C: E1 de ROXs42Bb_realigned (55 px) y ROXs42Bb_r78_20260927 (78 px), FAP
  paramétrica de psffit 0.0270 y 0.0285, z 2.86 y 2.81, intervalo bootstrap de la de 55 px;
- decisión D: E3 de la noche buena (ROXs12b_OB3445598) con E1 recalculado sobre su D2 con c0,
  4.2992e-14 (umbral de E1 z 3.68); validación 4.16151500457472e-14 sobre su copia de antes
  de c0 y 3.6434e-14 sobre ROXs12b_invvar; «15 % deeper» = 1 − 3.6434/4.2992.

Con dos runs o más imprime también `1 − Ṁ(primero)/Ṁ(cada uno)`: cuánto más hondo es el
umbral del primero.

uso:
    python scripts/recomputa_e1_e3.py --project-root RAIZ --sin-escribir --solo-e1 \\
        --run-id ROXs42Bb_realigned --run-id ROXs42Bb_r78_20260927
    python scripts/recomputa_e1_e3.py --project-root RAIZ --sin-escribir \\
        --run-id ROXs12b_invvar --run-id ROXs12b_OB3445598 \\
        --run-id ROXs12b_OB3445598@runs/ROXs12b_OB3445598/stages/_antes_c4_c0_20261025T030808
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from musepipe.paths import RunPaths  # noqa: E402
from musepipe.stages.stage_h01_detect import (  # noqa: E402
    compute_stage_h01_products,
    stage_h01_config_from_run,
    stage_h01_paths,
    write_stage_h01_products,
)
from musepipe.stages.stage_h03_limits import (  # noqa: E402
    compute_stage_h03_products,
    stage_h03_config_from_run,
    stage_h03_paths,
)
from musepipe.stages.stage_x10_compare import METHOD_ORDER  # noqa: E402

#: claves de `stage_h01_paths` que escribe `write_stage_h01_products` (más `plot_dir`)
SALIDAS_E1 = ("halpha_detection_csv", "null_maxima_npz", "stage_h01_qc_json", "summary_plot")

#: lo que E3 lee de E1: (clave en `stage_h03_paths`, clave en `stage_h01_paths`)
E1_PARA_E3 = (("stage_h01_qc_json", "stage_h01_qc_json"),
              ("h01_null_maxima_npz", "null_maxima_npz"),
              ("halpha_detection_csv", "halpha_detection_csv"))

#: entradas de E1 que el config puede redirigir: (knob de config, clave en `stage_h01_paths`).
#: `_measured_flux_from_product` de E3 resuelve el producto por estos mismos knobs.
ENTRADAS_E1 = tuple(
    [("h01_stage00q_qc_json", "stage00q_qc_json")]
    + [(f"h01_spec_calibrated_{m}_object", f"spec_calibrated_{m}_object") for m in METHOD_ORDER]
    + [(f"h01_controls_{m}_npz", f"controls_calibrated_{m}_npz") for m in METHOD_ORDER])

CAMPOS_E1 = ("matched_z", "template_factor", "global_empirical_fap", "global_parametric_fap",
             "parametric_fap_ks_p", "parametric_fap_extrapolation_sd", "n_controls")
CAMPOS_E3 = ("method", "limit_definition", "z_threshold", "f_stat_99", "throughput", "mdot_msun_yr")


def parsea_objetivo(texto, project_root):
    """`RUN` o `RUN@CARPETA` -> (run_id, copia o None); CARPETA relativa a la raíz del proyecto."""
    run, _, copia = str(texto).partition("@")
    if not run:
        raise ValueError(f"objetivo vacío: {texto!r}")
    if not copia:
        return run, None
    c = Path(copia)
    return run, (c if c.is_absolute() else Path(project_root) / c).resolve()


def remapea(ruta, run_paths: RunPaths, copia):
    """La ruta equivalente dentro de la copia (stages/ -> copia/stages, tables/ -> copia/tables)."""
    if copia is None or ruta is None:
        return ruta
    ruta = Path(ruta)
    for origen, destino in ((run_paths.stage_dir, copia / "stages"), (run_paths.table_dir, copia / "tables")):
        try:
            return destino / ruta.relative_to(origen)
        except ValueError:
            continue
    return ruta


def remapea_rutas(paths, copia):
    """Copia de un dict de rutas de etapa con todo lo de stages/ y tables/ del run en la copia."""
    if copia is None:
        return dict(paths)
    rp = paths["paths"]
    return {k: (v if k == "paths" or not isinstance(v, Path) else remapea(v, rp, copia)) for k, v in paths.items()}


def overrides_de_entradas(cfg, paths_h01, copia):
    """Knobs `h01_*` que apuntan las entradas de E1 a la copia (vacío sin copia)."""
    if copia is None:
        return {}
    rp = paths_h01["paths"]
    out = {}
    for knob, clave in ENTRADAS_E1:
        actual = Path(cfg.get(knob, paths_h01[clave]))
        nueva = remapea(actual, rp, copia)
        if nueva != actual:
            out[knob] = str(nueva)
    return out


def rutas_e1_temporales(paths_h01, run_id, tmp):
    """Las rutas de salida de E1 dentro de `tmp` (y un `RunPaths` que tampoco sale de ahí)."""
    tmp = Path(tmp)
    p = dict(paths_h01)
    p["paths"] = RunPaths.from_project_root(run_id, tmp)
    for clave in SALIDAS_E1:
        p[clave] = tmp / Path(paths_h01[clave]).name
    p["plot_dir"] = tmp
    return p


def salidas_de(paths_e1):
    """Todo lo que `write_stage_h01_products` puede crear con estas rutas."""
    rp = paths_e1["paths"]
    return [Path(paths_e1[c]) for c in SALIDAS_E1] + [Path(paths_e1["plot_dir"]), rp.config_dir, rp.stage_dir,
                                                      rp.plot_dir, rp.table_dir, rp.log_dir]


def comprueba_fuera(salidas, prohibidas):
    """Para si alguna salida (enlaces resueltos) cae dentro de una carpeta prohibida."""
    prohibidas = [Path(p).resolve() for p in prohibidas if p is not None]
    for s in salidas:
        real = Path(s).resolve()
        for p in prohibidas:
            if real == p or p in real.parents:
                raise RuntimeError(f"la salida {s} ({real}) cae dentro de {p}: este script no escribe ahí")


def carpetas_prohibidas(project_root, run_id=None, copia=None):
    """`runs/`, cada run que cuelga de él ya resuelto (los runs son enlaces a /mnt/2TB), el run y la copia."""
    runs_dir = RunPaths.from_project_root("_", project_root).runs_dir
    out = [runs_dir]
    if runs_dir.is_dir():
        out += [p.resolve() for p in sorted(runs_dir.iterdir())]
    if run_id is not None:
        out.append(RunPaths.from_project_root(run_id, project_root).run_dir)
    return out + [copia]


def _bootstrap(param):
    ci = (param or {}).get("ci95")
    return [float(v) for v in ci] if ci else None


def resumen_e1(producto):
    """Fila por método (z, FAP empírica y paramétrica, KS, intervalo bootstrap) y veredicto."""
    por_metodo = (producto.qc.get("parametric_fap") or {}).get("by_method") or {}
    filas = {}
    for r in producto.rows:
        f = {k: r.get(k) for k in CAMPOS_E1}
        f["fap_bootstrap_95"] = _bootstrap(por_metodo.get(r["method"]))
        filas[r["method"]] = f
    return {"veredicto": producto.qc.get("verdict"), "filas": filas}


def fila_combinada(producto_e3):
    for r in producto_e3.rows:
        if r.get("row_kind") == "combined_final":
            return {k: r.get(k) for k in CAMPOS_E3}
    raise RuntimeError("E3 sin fila combined_final")


def _lee_csv(path):
    path = Path(path)
    if not path.exists():
        return None
    with open(path, newline="", encoding="utf-8") as fh:
        return {r["method"]: r for r in csv.DictReader(fh)}


def _lee_json(path):
    path = Path(path)
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def _mismo(a, b):
    try:
        return float(a) == float(b)
    except (TypeError, ValueError):
        return a == b


def compara_con_disco(e1, e3, paths_h01, paths_h03):
    """Lo recalculado frente a lo que el run (o la copia) tiene escrito: igualdad exacta."""
    out = {}
    if e1 is not None:
        tabla = _lee_csv(paths_h01["halpha_detection_csv"])
        if tabla is not None:
            out["e1"] = {m: {k: {"disco": tabla[m].get(k), "igual": _mismo(tabla[m].get(k), f[k])}
                             for k in ("matched_z", "global_parametric_fap")}
                         for m, f in e1["filas"].items() if m in tabla}
    if e3 is not None:
        qc = _lee_json(paths_h03["stage_h03_qc_json"])
        if qc is not None:
            disco = next((l for l in qc.get("limits", []) if l.get("method") == "combined"), None)
            if disco is not None:
                out["e3"] = {"mdot_disco": disco.get("mdot"), "igual": _mismo(disco.get("mdot"), e3["mdot_msun_yr"])}
    return out


def diferencias_de_config(cfg_run, copia):
    """Claves de `config` que difieren entre el run y la copia (informativo)."""
    if copia is None or not (copia / "config" / "config.json").exists():
        return None
    otra = json.loads((copia / "config" / "config.json").read_text(encoding="utf-8")).get("config", {})
    propia = json.loads(Path(cfg_run).read_text(encoding="utf-8")).get("config", {})
    return sorted(k for k in set(otra) | set(propia) if otra.get(k) != propia.get(k))


def recalcula(run_id, project_root, *, copia=None, solo_e1=False, e1_del_disco=False, tmp_base=None,
              allow_run_id_mismatch=False):
    """E1 y E3 de un run sin escribir en él. Devuelve un dict con lo calculado y la comparación."""
    project_root = Path(project_root).resolve()
    cfg1 = stage_h01_config_from_run(run_id, project_root=project_root,
                                     allow_run_id_mismatch=allow_run_id_mismatch)
    paths1 = remapea_rutas(stage_h01_paths(cfg1["run_id"], project_root), copia)
    extra = overrides_de_entradas(cfg1, stage_h01_paths(cfg1["run_id"], project_root), copia)
    cfg1.update(extra)
    paths3 = remapea_rutas(stage_h03_paths(cfg1["run_id"], project_root), copia)
    prohibidas = carpetas_prohibidas(project_root, cfg1["run_id"], copia)
    res = {"run_id": cfg1["run_id"], "copia": str(copia) if copia else None,
           "config_distinta_en_la_copia": diferencias_de_config(paths1["paths"].config_json, copia),
           "e1_recalculado": not e1_del_disco}
    e1 = e3 = None
    with tempfile.TemporaryDirectory(prefix="recomputa_e1_e3_", dir=tmp_base) as tmp:
        if not e1_del_disco:
            prod1 = compute_stage_h01_products(cfg1, paths1)
            e1 = resumen_e1(prod1)
            res["e1"] = e1
        if not solo_e1:
            cfg3 = stage_h03_config_from_run(cfg1["run_id"], project_root=project_root,
                                             allow_run_id_mismatch=allow_run_id_mismatch)
            cfg3.update(extra)
            p3 = dict(paths3)
            if not e1_del_disco:
                p1tmp = rutas_e1_temporales(paths1, cfg1["run_id"], tmp)
                comprueba_fuera(salidas_de(p1tmp), prohibidas)
                write_stage_h01_products(prod1, cfg1, p1tmp)
                for k3, k1 in E1_PARA_E3:
                    p3[k3] = p1tmp[k1]
            e3 = fila_combinada(compute_stage_h03_products(cfg3, p3))
            res["e3"] = e3
    res["compara_con_disco"] = compara_con_disco(e1, e3, paths1, paths3)
    return res


def _fmt(v, fmt):
    return "—" if v is None or (isinstance(v, float) and not math.isfinite(v)) else format(v, fmt)


def imprime(res):
    etiqueta = res["run_id"] + (f" @ {res['copia']}" if res["copia"] else "")
    print(f"== {etiqueta}")
    if res.get("config_distinta_en_la_copia"):
        print(f"   (config del run, no la de la copia; difieren: {', '.join(res['config_distinta_en_la_copia'])})")
    comp = res.get("compara_con_disco", {})
    if "e1" in res:
        v = res["e1"]["veredicto"] or {}
        print(f"   E1 recalculado: {v.get('verdict')} ({v.get('reason')}; estimador {v.get('fap_estimator')})")
        for m, f in res["e1"]["filas"].items():
            ci = f["fap_bootstrap_95"]
            igual = comp.get("e1", {}).get(m, {})
            marca = "" if not igual else (" = disco" if all(x["igual"] for x in igual.values()) else " ≠ DISCO")
            print(f"     {m:15s} z {_fmt(f['matched_z'], '.3f')} (×{f['template_factor']:g})  FAP emp "
                  f"{_fmt(f['global_empirical_fap'], '.4f')}  param {_fmt(f['global_parametric_fap'], '.4g')}"
                  f"  KS p {_fmt(f['parametric_fap_ks_p'], '.3f')}"
                  f"  bootstrap 95 % {'—' if ci is None else f'[{ci[0]:.2g}, {ci[1]:.2g}]'}{marca}")
    if "e3" in res:
        e = res["e3"]
        c = comp.get("e3")
        marca = "" if c is None else (" = QC de E3 en disco" if c["igual"] else f" ≠ QC en disco ({c['mdot_disco']!r})")
        print(f"   E3 ({'E1 del disco' if not res['e1_recalculado'] else 'con el E1 recalculado'}): "
              f"{e['limit_definition']}, umbral z {e['z_threshold']:.4f}, f_stat_99 {e['f_stat_99']:.6g}, "
              f"throughput {e['throughput']:.6g}, Mdot {e['mdot_msun_yr']!r} Msun/yr{marca}")


def mas_hondo(resultados):
    """`1 − Ṁ(primero)/Ṁ(i)` para cada run con E3 después del primero."""
    con = [r for r in resultados if "e3" in r]
    if len(con) < 2:
        return []
    m0 = float(con[0]["e3"]["mdot_msun_yr"])
    return [{"referencia": con[0]["run_id"], "contra": r["run_id"], "copia": r["copia"],
             "mas_hondo": 1.0 - m0 / float(r["e3"]["mdot_msun_yr"])} for r in con[1:]]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--run-id", action="append", required=True, metavar="RUN[@COPIA]",
                    help="run (repetible); RUN@CARPETA lee los productos de una copia con stages/ y tables/")
    ap.add_argument("--project-root", default=str(ROOT),
                    help="clon con runs/ (desde un worktree, el checkout principal)")
    ap.add_argument("--sin-escribir", action="store_true",
                    help="obligatorio: este script nunca escribe en runs/ (para escribir, corre la etapa)")
    ap.add_argument("--solo-e1", action="store_true", help="no calcula E3")
    ap.add_argument("--e1-del-disco", action="store_true", help="E3 lee el E1 que ya está en el run (o la copia)")
    ap.add_argument("--tmp-dir", default=None, help="base de la carpeta temporal del E1 (por defecto, la del sistema)")
    ap.add_argument("--allow-run-id-mismatch", action="store_true")
    ap.add_argument("--json", default=None, help="escribe el resultado aquí (fuera de runs/)")
    a = ap.parse_args(argv)
    if not a.sin_escribir:
        ap.error("pasa --sin-escribir: este script solo recalcula en memoria; para escribir E1/E3 en el run, "
                 "corre la etapa (python -m musepipe.stages.stage_h01_detect / stage_h03_limits)")
    if a.solo_e1 and a.e1_del_disco:
        ap.error("--solo-e1 y --e1-del-disco juntos no calculan nada")
    project_root = Path(a.project_root).resolve()
    objetivos = [parsea_objetivo(t, project_root) for t in a.run_id]
    prohibidas = carpetas_prohibidas(project_root) + [c for _r, c in objetivos]
    for salida in (a.tmp_dir, a.json):
        if salida:
            comprueba_fuera([Path(salida)], prohibidas)
    resultados = []
    for run, copia in objetivos:
        res = recalcula(run, project_root, copia=copia, solo_e1=a.solo_e1, e1_del_disco=a.e1_del_disco,
                        tmp_base=a.tmp_dir, allow_run_id_mismatch=a.allow_run_id_mismatch)
        imprime(res)
        resultados.append(res)
    hondo = mas_hondo(resultados)
    for h in hondo:
        print(f"   umbral de {h['referencia']} {100 * h['mas_hondo']:.2f} % más hondo que el de {h['contra']}"
              + (f" @ copia" if h["copia"] else ""))
    if a.json:
        Path(a.json).write_text(json.dumps({"runs": resultados, "mas_hondo": hondo}, indent=1, default=float,
                                           ensure_ascii=False) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
