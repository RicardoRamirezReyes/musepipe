#!/usr/bin/env python
"""A_V propia de un compañero por exceso de color, con plantillas jóvenes como colores intrínsecos.

Solo lectura y fuera de la cadena: la cadena desenrojece con la A_V de la primaria (`h03_av`) y no usa nada
de esto. Da las cifras de la extinción de la sección de sistemáticos y del apéndice B del paper; está
contado en docs/2026-10-13_av_exceso_color.md. `tests/test_av_exceso_color.py` fija las propiedades y, si
los datos están en disco, las cifras.

**Idea.** Un color solo no separa la temperatura de la extinción. El tipo se fija con los índices de agua
(`tipo_ir_indices.lectura`), que casi no dependen de la A_V, y con el tipo fijado cada color observado da la
A_V que necesita cada plantilla joven de ese tipo (±`--ventana` subtipos) para igualarlo.

**Colores «espectrofotométricos».** Media de F_λ en las MISMAS ventanas limpias (`VENTANAS`, fuera de las
bandas telúricas) en el compañero y en las plantillas: no hace falta la curva de ningún filtro. Los espectros
IR del compañero vienen escalados a su fotometría, así que sus cocientes entre bandas son sus colores. Qué
bandas valen, y si K se reescala, lo declara `targets/<slug>.json` -> `ir_spectra.colour_excess` (en un
objeto los espectros J y H están escalados a la fotometría de otra fuente). MUSE: el espectro calibrado de
D2 en 8400–8900 Å, entre las bandas de H2O, con tres métodos de extracción para ver cuánto mueve el continuo.

**Plantillas** (`tipo_ir_plantillas.plantillas`): los colores IR usan el brazo NIR de todas; MUSE − K solo
las de Manara, cuyos brazos VIS y NIR están calibrados en flujo de forma coherente (las de ESO tienen el NIR
normalizado). Se enrojecen con CCM89 (R_V = 3.1) y la A_V se resuelve exactamente, sin longitud de onda
efectiva. Se suponen sin extinción propia: si una la tiene, la A_V que se deriva con ella sale baja por eso.

uso: python scripts/av_exceso_color.py SLUG [--tipo M9] [--ventana 1.0] [--json SALIDA]
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from musepipe.config import load_run_config  # noqa: E402
from musepipe.constants import spt_code  # noqa: E402
from musepipe.io import resolve_flux_unit  # noqa: E402
from scripts import tipo_ir_indices as tir  # noqa: E402
from scripts import tipo_ir_plantillas as tp  # noqa: E402

#: ventanas limpias (Å): fuera de `tipo_ir_plantillas.TELURICO_IR_A` y dentro de los espectros del compañero
VENTANAS = {"J": [(11700.0, 12600.0), (12750.0, 13250.0)], "H": [(15000.0, 17800.0)],
            "K": [(20800.0, 23000.0)], "MUSE": [(8400.0, 8900.0)]}
PARES_IR = (("J", "H"), ("H", "K"), ("J", "K"))
METODOS_MUSE = ("psffit", "aperture", "optimal_ls")
W_PER_M2_UM_A_CGS = 0.1      # 1 W m⁻² µm⁻¹ = 0.1 erg s⁻¹ cm⁻² Å⁻¹


def media(w, f, ventanas, av=0.0):
    """Media de F_λ, enrojecida con `av`, en las ventanas (ponderada por su anchura); NaN si falta > 20 %."""
    tot, peso = 0.0, 0.0
    for lo, hi in ventanas:
        s = (w >= lo) & (w <= hi)
        ok = s & np.isfinite(f)
        if s.sum() < 3 or ok.sum() < 0.8 * s.sum():
            return np.nan
        k = np.asarray(tp.EXT.a_lambda_over_av(w[ok]), float)
        g = f[ok] * 10 ** (-0.4 * k * av)
        tot += np.trapz(g, w[ok])
        peso += w[ok][-1] - w[ok][0]
    return tot / peso


def av_que_iguala(wa, fa, va, wb, fb, vb, r_obs, lo=-5.0, hi=20.0):
    """A_V con la que media_a/media_b de la plantilla enrojecida vale `r_obs` (NaN sin raíz en [lo, hi])."""
    from scipy.optimize import brentq
    g = lambda av: np.log(media(wa, fa, va, av) / media(wb, fb, vb, av)) - np.log(r_obs)
    a, b = g(lo), g(hi)
    if not (np.isfinite(a) and np.isfinite(b)) or np.sign(a) == np.sign(b):
        return np.nan
    return float(brentq(g, lo, hi))


def bandas_del_companero(slug):
    """{banda: [(espectro, F_λ medio en erg/s/cm²/Å)]} de las bandas que declara `colour_excess`."""
    ir = tir.perfil(slug)["ir_spectra"]
    ce = ir.get("colour_excess")
    if not ce:
        raise SystemExit(f"{slug}: targets/ sin 'ir_spectra.colour_excess' (qué bandas valen para el color)")
    k_fac = 1.0
    if ce.get("k_rescale_mag"):
        k_fac = 10 ** (-0.4 * (ce["k_rescale_mag"]["adopted"] - ce["k_rescale_mag"]["catalogue"]))
    out = {}
    for nombre, spec in ir["spectra"].items():
        if spec["band"] not in ce["bands"]:
            continue
        w_um, f, _ = tir.load(tir.DEFAULT_CDS, spec["file"])
        m = media(w_um * 1e4, f, VENTANAS[spec["band"]]) * W_PER_M2_UM_A_CGS
        if spec["band"] == "K":
            m *= k_fac
        if np.isfinite(m):
            out.setdefault(spec["band"], []).append((nombre, m))
    return out


def flujo_muse(run, metodo):
    from astropy.io import fits
    cfg = load_run_config(run, project_root=ROOT).config
    with fits.open(ROOT / "runs" / run / "stages" / f"spec_calibrated_{metodo}_object.fits") as h:
        t = h["SPECTRUM"]
        w, f, bunit = np.asarray(t.data["wave_A"], float), np.asarray(t.data["flux"], float), t.header.get("BUNIT")
    escala, _ = resolve_flux_unit(cfg, bunit=bunit)
    return media(w, f * escala, VENTANAS["MUSE"])


def _ordena(w, f):
    o = np.argsort(w)
    return w[o], f[o]


def estimacion(slug, tipo=None, ventana=1.0):
    """A_V por plantilla y color, y el resumen de las plantillas a ±`ventana` subtipos del tipo."""
    perfil = tir.perfil(slug)
    run = perfil["paper"]["run"]
    cfg = load_run_config(run, project_root=ROOT).config
    if tipo is None:
        av, av_err = tir.av_del_run(run)
        tipo_code = tir.lectura(tir.DEFAULT_CDS, perfil["ir_spectra"]["spectra"], av, av_err)["spt"]
        origen = "índices de agua"
    else:
        tipo_code, origen = spt_code(tipo), "declarado"
    bandas = bandas_del_companero(slug)
    muse = {m: flujo_muse(run, m) for m in METODOS_MUSE}
    filas = []
    for t in tp.plantillas(cfg):
        wn, fn = _ordena(*t["nir"])
        fila = dict(plantilla=f"{t['lib']}:{t['objeto']}", lib=t["lib"], spt=t["spt"], codigo=spt_code(t["spt"]),
                    av={})
        for a, b in PARES_IR:
            r = [fa / fb for (_, fa), (_, fb) in itertools.product(bandas.get(a, []), bandas.get(b, []))]
            if r:
                v = [av_que_iguala(wn, fn, VENTANAS[a], wn, fn, VENTANAS[b], x) for x in r]
                fila["av"][f"{a}-{b}"] = float(np.nanmean(v))
        if t["lib"] == "manara" and bandas.get("K"):
            wv, fv = _ordena(*t["vis"])
            for m, fm in muse.items():
                v = [av_que_iguala(wv, fv, VENTANAS["MUSE"], wn, fn, VENTANAS["K"], fm / fk) for _, fk in bandas["K"]]
                fila["av"][f"MUSE-K {m}"] = float(np.nanmean(v))
        filas.append(fila)
    cerca = [f for f in filas if abs(f["codigo"] - tipo_code) <= ventana + 1e-9]
    resumen = {}
    for c in sorted({c for f in cerca for c in f["av"]}):
        v = np.array([f["av"][c] for f in cerca if np.isfinite(f["av"].get(c, np.nan))])
        if v.size:
            resumen[c] = dict(n=int(v.size), mediana=float(np.median(v)), min=float(v.min()), max=float(v.max()))
    cota = max(r["max"] for r in resumen.values()) if resumen else np.nan
    return dict(slug=slug, run=run, av_adoptada=float(cfg["h03_av"]), tipo=tir.spt_label(tipo_code),
                tipo_codigo=float(tipo_code), tipo_origen=origen, ventana=ventana,
                plantillas_cerca=[f["plantilla"] for f in cerca], resumen=resumen, cota_superior=cota,
                muse_flujo=muse, bandas={b: [n for n, _ in v] for b, v in bandas.items()}, filas=filas)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("slug")
    ap.add_argument("--tipo", help="tipo de las plantillas (por defecto, el de los índices de agua)")
    ap.add_argument("--ventana", type=float, default=1.0, help="subtipos alrededor del tipo")
    ap.add_argument("--json", type=Path)
    a = ap.parse_args(argv)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        out = estimacion(a.slug, a.tipo, a.ventana)
    print(f"{a.slug}: tipo {out['tipo']} ({out['tipo_origen']}) ± {a.ventana}; A_V adoptada {out['av_adoptada']}; "
          f"plantillas: {', '.join(out['plantillas_cerca'])}")
    for c, r in out["resumen"].items():
        print(f"  A_V({c}): mediana {r['mediana']:.2f}, {r['min']:.2f}–{r['max']:.2f} (n {r['n']})")
    print(f"  cota superior (plantilla más azul): {out['cota_superior']:.2f}")
    if a.json:
        a.json.write_text(json.dumps(out, indent=1, ensure_ascii=False, default=float))
    return 0


if __name__ == "__main__":
    sys.exit(main())
