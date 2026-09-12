"""¿Cuánto depende T_eff del rango en el que G3 ajusta?

G3 ajusta desde **6300 Å**, y allí el continuo de la compañera de ROXs 42B b tiene
**S/N 1.3** (contra 8.5–9.3 en el rojo); en ROXs 12 b la medida equivalente daba
~0 en las cuatro bandas azules. La pregunta es si el resultado depende de esa
elección — y se puede contestar **sobre el espectro final que ya existe**, sin
re-extraer nada.

**Ejes reducidos a propósito.** `A_V` va en pasos de 0.25 en vez de 0.1 y `log g`
se fija, lo que abarata el ajuste ~12×. Vale porque la comparación es
**diferencial**: todos los rangos se ajustan con los MISMOS ejes, así que lo que
se lee es el desplazamiento entre ellos, no el valor absoluto — que hay que seguir
tomando del producto de G3.

    python scripts/n_rango_de_ajuste_g3.py --run <run> --out-json <out>.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
import sys  # noqa: E402

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from musepipe.config import load_run_config  # noqa: E402
from musepipe.models.btsettl import BTSettlLibrary  # noqa: E402
from musepipe.models.extinction import CCMExtinction  # noqa: E402
from musepipe.models.fit import fit_grid_3d  # noqa: E402
from musepipe.models.observed import fit_spectrum  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--inicios", nargs="+", type=float, default=[6300.0, 7000.0, 7600.0, 8000.0])
    ap.add_argument("--av-step", type=float, default=0.25)
    ap.add_argument("--logg", type=float, default=4.0)
    ap.add_argument("--out-json", required=True)
    args = ap.parse_args(argv)

    rc = load_run_config(args.run)
    cfg, paths = dict(rc.config), rc.paths
    cfg.setdefault("project_root", str(paths.project_root))
    rango = cfg["g3_fit_wave_range_A"]
    lsf = float(cfg["h01_lsf_fwhm_A"])
    ext = CCMExtinction(rv=float(cfg.get("h03_rv_extinction", 3.1)),
                        citation=cfg.get("h03_extinction_law_citation", "Cardelli+1989"))
    t0, t1, dt = cfg["g3_atmo_teff_axis_k"]
    teff_axis = np.arange(float(t0), float(t1) + 0.5 * float(dt), float(dt))
    av_axis = np.arange(0.0, 5.0 + 0.5 * args.av_step, args.av_step)
    logg_axis = np.array([float(args.logg)])
    from musepipe.models.manifest import library_root
    lib = BTSettlLibrary(Path(library_root(cfg, project_root=cfg["project_root"])) / "bt-settl-cifist",
                         citation=cfg.get("g3_atmosphere_citation", "Allard et al. 2012"),
                         version=cfg.get("g3_atmosphere_version"))
    print(f"{args.run}: rango en produccion {rango} | LSF {lsf:.3f} A | "
          f"ejes T_eff {teff_axis.size}, A_V {av_axis.size}, log g fijo {args.logg}", flush=True)

    filas = []
    print(f"\n{'inicio':>8s} {'bins':>6s} {'Teff':>8s} {'A_V':>7s} {'chi2red':>9s} {'dTeff vs prod':>14s}")
    ref = None
    for inicio in args.inicios:
        c = dict(cfg)
        c["g3_fit_wave_range_A"] = [float(inicio), float(rango[1])]
        fs = fit_spectrum(c, paths)
        res = fit_grid_3d(fs, lib, ext, teff_axis=teff_axis, logg_axis=logg_axis,
                          av_axis=av_axis, lsf_fwhm_A=lsf,
                          chi2red_inflate_threshold=float(cfg.get("g3_chi2red_inflate_threshold", 1.5)),
                          sys_fluxcal_frac=float(cfg.get("g3_sys_fluxcal_frac", 0.10)))
        teff, av = float(res["teff_best"]), float(res["av_best"])
        chi2r = float(res.get("chi2_red", np.nan))
        if ref is None:
            ref = teff
        filas.append({"inicio_A": float(inicio), "n_bins": int(fs.n_bins), "teff_best": teff,
                      "av_best": av, "chi2_red": chi2r, "dteff_vs_produccion": teff - ref})
        print(f"{inicio:8.0f} {fs.n_bins:6d} {teff:8.0f} {av:7.2f} {chi2r:9.3f} {teff - ref:+14.0f}",
              flush=True)
        Path(args.out_json).write_text(json.dumps(
            {"script": "n_rango_de_ajuste_g3", "run": args.run, "rango_produccion": rango,
             "av_step": args.av_step, "logg_fijo": args.logg, "filas": filas}, indent=1), encoding="utf-8")
    t = np.array([f["teff_best"] for f in filas])
    print(f"\n  T_eff entre {t.min():.0f} y {t.max():.0f} K -> rango {t.max()-t.min():.0f} K")
    print("  (ejes reducidos: leer el DESPLAZAMIENTO, no el valor absoluto)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
