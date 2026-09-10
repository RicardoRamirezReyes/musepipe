"""Los perfiles de chi2 a lo largo de A_V que G3 calcula y no guarda.

Por que. La via 3 para medir la extincion -la forma del continuo, via el ajuste
atmosferico de G3- esta declarada cerrada, pero lo que el QC publica son solo los
best-fit y los intervalos, y eso esconde lo que de verdad pasa. En ROXs 42B b:

  * A_V sale **0.1** con intervalo **[0.1, 0.1]** -un solo nodo de una rejilla de 51-
    pese a que los errores se inflaron **x3.01** por chi2_red = 9.06.
  * La variante con **veiling** del MISMO ajuste da **A_V = 5.0**, que es el TECHO del
    eje, con chi2 **menos de la mitad** (349.7 contra 724.8).

Cinco magnitudes de horquilla son **1.85 dex en Mdot** a 0.370 dex/mag. Un intervalo de
un solo nodo sobre un ajuste que va 9x mal no es una medida, y sin el perfil no se ve.

`fit_grid_3d` YA devuelve `dchi2_3d`, `dchi2_teff_av` y los ejes; G3 se queda solo con
el resumen. Esto llama a las mismas piezas y los guarda. Ademas mira `edge_touch` en
las VARIANTES, que el QC solo publica para la primaria -y la variante que toca el borde
es justo la que mejor ajusta.

No escribe nada dentro de `runs/`.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from musepipe.models.btsettl import BTSettlLibrary  # noqa: E402
from musepipe.models.extinction import CCMExtinction  # noqa: E402
from musepipe.models.fit import fit_grid_3d  # noqa: E402
from musepipe.models.manifest import library_provenance, library_root  # noqa: E402
from musepipe.models.observed import fit_spectrum  # noqa: E402
from musepipe.config import load_run_config  # noqa: E402
from musepipe.stages.stage_g3_atmo_fit import (  # noqa: E402
    _axis, stage_g3_atmo_fit_paths)


def _cfg_and_paths(run):
    """Igual que `run_stage_g3_atmo_fit`: la etapa no tiene `config_from_run` propio."""

    rc = load_run_config(run, project_root=str(ROOT))
    cfg = dict(rc.config)
    cfg["run_id"] = rc.run_id
    cfg["project_root"] = str(rc.paths.project_root)
    paths = stage_g3_atmo_fit_paths(cfg["run_id"], project_root=cfg["project_root"])
    return cfg, paths


def _jsonable(interval):
    if interval is None:
        return None
    return [None if v is None else float(v) for v in interval]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-npz", required=True)
    args = ap.parse_args()

    cfg, paths = _cfg_and_paths(args.run)
    lsf = float(cfg["h01_lsf_fwhm_A"])
    ext = CCMExtinction(rv=float(cfg.get("h03_rv_extinction", 3.1)),
                        citation=cfg.get("h03_extinction_law_citation", "Cardelli+1989"))
    teff_axis = _axis(cfg["g3_atmo_teff_axis_k"])
    logg_axis = _axis(cfg["g3_atmo_logg_axis"])
    av_axis = _axis(cfg["g3_atmo_av_axis"])
    infl = float(cfg.get("g3_chi2red_inflate_threshold", 1.5))
    sysfrac = float(cfg.get("g3_sys_fluxcal_frac", 0.10))

    print(f"run {args.run}")
    print(f"  rejilla: Teff {teff_axis.size} x logg {logg_axis.size} x A_V {av_axis.size}"
          f" = {teff_axis.size*logg_axis.size*av_axis.size} nodos")
    print(f"  A_V de {av_axis[0]:.2f} a {av_axis[-1]:.2f} paso {av_axis[1]-av_axis[0]:.2f}")
    print(f"  LSF {lsf:.4f} A (h01_lsf_fwhm_A), R_V {ext.rv}")

    prov = library_provenance(cfg)
    root = library_root(cfg, project_root=cfg.get("project_root", str(ROOT)))
    library = BTSettlLibrary(root / "bt-settl-cifist",
                             citation=cfg.get("g3_atmosphere_citation", "Allard et al. 2012"),
                             version=cfg.get("g3_atmosphere_version"))
    fs = fit_spectrum(cfg, paths["paths"])
    print(f"  espectro: {fs.n_bins} bins, n_eff {fs.n_eff:.1f}")

    variants = {}
    arrays = {}
    for name, veiling in (("primary", False), ("veiling", True)):
        t0 = time.time()
        r = fit_grid_3d(fs, library, ext, teff_axis=teff_axis, logg_axis=logg_axis,
                        av_axis=av_axis, lsf_fwhm_A=lsf,
                        chi2red_inflate_threshold=infl, sys_fluxcal_frac=sysfrac,
                        veiling=veiling,
                        veiling_alpha_axis=cfg.get("g3_veiling_alpha_axis",
                                                   (-2.0, -1.0, 0.0, 1.0, 2.0)))
        dt = time.time() - t0
        # perfil de verosimilitud a lo largo de A_V: el minimo sobre Teff y logg
        marg_av = np.asarray(r["dchi2_3d"]).min(axis=(0, 1))
        marg_teff = np.asarray(r["dchi2_3d"]).min(axis=(1, 2))
        arrays[f"{name}_av_axis"] = np.asarray(r["av_axis"])
        arrays[f"{name}_teff_axis"] = np.asarray(r["teff_axis"])
        arrays[f"{name}_marg_av"] = marg_av
        arrays[f"{name}_marg_teff"] = marg_teff
        arrays[f"{name}_dchi2_teff_av"] = np.asarray(r["dchi2_teff_av"])
        variants[name] = {
            "av_best": float(r["av_best"]), "teff_best": float(r["teff_best"]),
            "logg_best": float(r["logg_best"]),
            "chi2_min": float(r["chi2_min"]), "chi2_red": float(r["chi2_red"]),
            "ndof": int(r["ndof"]), "n_bins": int(r["n_bins"]),
            "av_interval": _jsonable(r["av_interval"]),
            "teff_interval": _jsonable(r["teff_interval"]),
            "edge_touch": {k: bool(v) for k, v in r["edge_touch"].items()},
            "inflate": {"applied": bool(r["inflate"]["applied"]),
                        "factor": float(r["inflate"]["factor"])},
            # Lo que el intervalo NO dice: cuanto sube el chi2 en los extremos del eje.
            "dchi2_at_av_min": float(marg_av[0]),
            "dchi2_at_av_max": float(marg_av[-1]),
            "av_within_dchi2_1": [float(av_axis[i]) for i in np.where(marg_av <= 1.0)[0]],
            "av_within_dchi2_9": [float(av_axis[i]) for i in np.where(marg_av <= 9.0)[0]],
            "seconds": round(dt, 1),
        }
        v = variants[name]
        print(f"  [{name:8s}] A_V={v['av_best']:.2f} Teff={v['teff_best']:.0f} "
              f"chi2={v['chi2_min']:.1f} chi2_nu={v['chi2_red']:.2f} "
              f"intervalo A_V={v['av_interval']} borde={v['edge_touch']['av']} "
              f"[{v['seconds']:.0f}s]")
        print(f"             A_V con dchi2<=1: {len(v['av_within_dchi2_1'])} nodos de "
              f"{av_axis.size}; con dchi2<=9 (3 sigma): {len(v['av_within_dchi2_9'])}")

    np.savez_compressed(args.out_npz, **arrays)
    payload = {
        "stage": "extincion_perfiles_chi2", "run": args.run,
        "lsf_fwhm_A": lsf, "rv_extinction": float(ext.rv),
        "extinction_law": cfg.get("h03_extinction_law_citation", "Cardelli+1989"),
        "axes": {"teff": [float(teff_axis[0]), float(teff_axis[-1]), float(teff_axis[1]-teff_axis[0])],
                 "logg": [float(logg_axis[0]), float(logg_axis[-1]), float(logg_axis[1]-logg_axis[0])],
                 "av": [float(av_axis[0]), float(av_axis[-1]), float(av_axis[1]-av_axis[0])]},
        "library": {"root": str(root), "provenance": prov},
        "spectrum": {"n_bins": int(fs.n_bins), "n_eff": float(fs.n_eff)},
        "variants": variants,
        "npz": str(args.out_npz),
        "note": ("`edge_touch` que publica G3 es el de la variante PRIMARIA. Aqui va el "
                 "de las dos, porque la que toca el borde del eje es la que mejor ajusta."),
    }
    Path(args.out_json).write_text(json.dumps(payload, indent=1, ensure_ascii=False))
    print(f"\n-> {args.out_json}\n-> {args.out_npz}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
