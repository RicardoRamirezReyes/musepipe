"""Calibra el umbral de bondad de ajuste de G3 en bins de 25 Å (Q1).

Reparto aprobado el 2026-09-24: el TIPO se mide en el ajuste nativo por Δχ²
(sin puerta absoluta); la BONDAD de ajuste se prueba en la variante binada, y
su umbral es el que calibra este script. Para un run, toma cada plantilla de la
biblioteca joven, la degrada a la LSF de MUSE sobre la rejilla por canal del
compañero, le suma ruido con su espectro de error, la bina exactamente como
``rebin_for_fit`` y la ajusta dejándola fuera contra el resto de su biblioteca.
Devuelve la distribución de χ²_ν en el subtipo correcto (o el más cercano
disponible) y propone su percentil 95
(:func:`musepipe.models.calibration.calibrate_gof_threshold`). La misma
distribución a resolución nativa va al JSON solo como diagnóstico
(``native_diagnostic``; se omite con ``--sin-nativo``).

**Solo lee el run** (espectro final, máscaras, covarianza de G1, config) y
**solo escribe en ``--out``**, que no puede caer dentro de ``runs/``. No toca
ningún config: el umbral lo decide el autor.

    python scripts/g3_calibrar_umbral.py --run-id ROXs42Bb_realigned \\
        --project-root <raiz> --out /ruta/umbral_42Bb.json

Un run sin las claves ``g3_*`` completas (``ROXs12b_invvar`` no declara
``g3_libraries_root``) las toma de ``--g3-config-from <config.json>`` (solo
rellena las que faltan) y de ``--set clave=<json>`` (manda), y el JSON de
salida registra qué se añadió y de dónde.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from musepipe.config import load_run_config  # noqa: E402
from musepipe.models.calibration import calibrate_gof_threshold  # noqa: E402
from musepipe.models.extinction import CCMExtinction  # noqa: E402
from musepipe.models.libraries import (data_frame, gravity_pair,  # noqa: E402
                                       resolve_declaration, template_library_entries)
from musepipe.models.manifest import library_root  # noqa: E402
from musepipe.models.observed import load_fit_inputs  # noqa: E402
from musepipe.models.templates import EmpiricalTemplateLibrary  # noqa: E402


def _axis(triple):
    lo, hi, step = (float(x) for x in triple)
    return lo + step * np.arange(int(round((hi - lo) / step)) + 1)


def build_config(rc, *, g3_config_from=None, sets=()):
    cfg = dict(rc.config)
    added = {}
    if g3_config_from:
        payload = json.loads(Path(g3_config_from).read_text())
        src = payload.get("config", payload)
        for k, v in src.items():
            if k.startswith("g3_") and k not in cfg:
                cfg[k] = v
                added[k] = f"--g3-config-from {g3_config_from}"
    for item in sets:
        key, _, raw = item.partition("=")
        cfg[key] = json.loads(raw)
        added[key] = "--set"
    cfg["run_id"] = rc.run_id
    cfg["project_root"] = str(rc.paths.project_root)
    return cfg, added


def run(run_id, *, project_root, out, library=None, n_draws=20, seed=0,
        g3_config_from=None, sets=(), native=True, rule=None, diagnostic_all=False):
    rc = load_run_config(run_id, project_root=project_root)
    out = Path(out).resolve()
    runs_dir = (Path(rc.paths.project_root) / "runs").resolve()
    if runs_dir == out or runs_dir in out.parents:
        raise SystemExit(f"--out no puede estar dentro de runs/: {out}")
    cfg, added = build_config(rc, g3_config_from=g3_config_from, sets=sets)

    lsf = float(cfg["h01_lsf_fwhm_A"])
    ext = CCMExtinction(rv=float(cfg.get("h03_rv_extinction", 3.1)),
                        citation=cfg.get("h03_extinction_law_citation", "Cardelli+1989"))
    av_axis = _axis(cfg["g3_atmo_av_axis"])
    name = library or gravity_pair(cfg)["young"]
    entry = next((e for e in template_library_entries(cfg) if e["name"] == name), None)
    if entry is None:
        raise SystemExit(f"biblioteca {name!r} no declarada en g3_template_libraries")
    root = library_root(cfg, project_root=cfg["project_root"])
    decl = resolve_declaration(entry, root / entry["subdir"])
    lib = EmpiricalTemplateLibrary(root / entry["subdir"], gravity_class=decl.gravity_class,
                                   citation=decl.citation, version=None, declaration=decl)

    inputs = load_fit_inputs(cfg, rc.paths)
    wave_range = cfg["g3_fit_wave_range_A"]
    n_gof = int(cfg.get("g3_gof_bin_channels", 20))
    kw = dict(wave_range=wave_range, lsf_fwhm_A=lsf, extinction=ext, av_axis=av_axis,
              data_frame=data_frame(cfg), n_draws=n_draws, seed=seed,
              max_masked_frac=float(cfg.get("g3_fit_bin_max_masked_frac", 0.5)))
    rule = rule or str(cfg.get("g3_gof_calibration", "same_subtype_neighbour"))
    res = calibrate_gof_threshold(lib, inputs, n_channels=n_gof, rule=rule, **kw)
    if diagnostic_all:
        alt = calibrate_gof_threshold(lib, inputs, n_channels=n_gof,
                                      rule="nearest_available", **kw)
        res["nearest_available_diagnostic"] = {k: v for k, v in alt.items() if k != "rows"}
    if native:
        nat = calibrate_gof_threshold(lib, inputs, n_channels=1, rule=rule, **kw)
        res["native_diagnostic"] = {
            "note": ("diagnostic only: at native sampling right and wrong templates all "
                     "give chi2_red ~ 1; no gate is derived from it"),
            **{k: v for k, v in nat.items() if k != "rows"}}
    res.update({
        "run_id": rc.run_id, "library": decl.to_qc(n_by_spt=lib.n_by_spt()),
        "lsf_fwhm_A": lsf, "wave_range_A": list(wave_range),
        "err_column": inputs["err_column"],
        "av_axis": [float(av_axis[0]), float(av_axis[-1]), int(av_axis.size)],
        "config_keys_added": added,
        "current_g3_chi2red_stop": cfg.get("g3_chi2red_stop"),
        "note": ("proposed binned-GOF threshold; G3 recomputes it in-run unless the config "
                 "declares g3_gof_chi2red_threshold (not written by this script)"),
        "decision": "docs/2026-09-23_decision_g3_resolucion_y_bibliotecas.md",
    })
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1, default=str))
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--project-root", default=None)
    ap.add_argument("--out", required=True, help="fichero JSON de salida (fuera de runs/)")
    ap.add_argument("--library", default=None,
                    help="nombre de la biblioteca (por defecto la joven del par de gravedad)")
    ap.add_argument("--n-draws", type=int, default=20)
    ap.add_argument("--regla", default=None, choices=("same_subtype_neighbour",
                                                      "nearest_available"),
                    help="regla de calibración (por defecto g3_gof_calibration del run, "
                         "o same_subtype_neighbour)")
    ap.add_argument("--con-todas", action="store_true",
                    help="añade la regla nearest_available como diagnóstico")
    ap.add_argument("--sin-nativo", action="store_true",
                    help="no calcular la distribución nativa de diagnóstico")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--g3-config-from", default=None)
    ap.add_argument("--set", action="append", default=[], metavar="CLAVE=JSON")
    args = ap.parse_args(argv)
    res = run(args.run_id, project_root=args.project_root, out=args.out,
              library=args.library, n_draws=args.n_draws, seed=args.seed,
              g3_config_from=args.g3_config_from, sets=args.set,
              native=not args.sin_nativo, rule=args.regla,
              diagnostic_all=args.con_todas)
    d = res["distribution"]
    print(f"{res['run_id']}: umbral binado propuesto (p{res['percentile']:.0f}, "
          f"{res['rule']}, {res['n_templates_used']}/{res['n_templates_library']} plantillas) = "
          f"{res['threshold_proposed']:.3f}; p50 = {d['p50']:.3f}, n = {d['n']}")
    print(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
