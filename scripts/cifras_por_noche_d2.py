#!/usr/bin/env python
"""Cifras por noche de ROXs 12 b que cita el paper, desde los espectros calibrados de D2.

Solo lectura: no escribe en runs/. Reconstruye las medidas de
docs/2026-09-02_es_residuo_de_la_primaria.md, que no dejaron script, y las de los controles de
docs/2026-09-20_invvar_pasa_a_publicacion.md §4. Procedencia y validación en
docs/2026-09-25_cifras_por_noche_a_55px.md.

- Flujo de Hα: integral en ±6 Å de 6562.8 Å con el continuo (mediana entre 10 y 40 Å del
  centro) restado; error = `flux_err_total` integrado. A 78 px reproduce exactos los errores
  del 09-02 (6.7e5, 4.1e5, 2.7e2, 5.4e3) y los flujos al 1-3 %.
- Controles del cubo analizado: el método identificado el 09-20 (continuo entre 20 y 80 Å),
  sobre `control_spectra` de `spec_calibrated_psffit_controls.npz`.

Los runs salen de `targets/<slug>.json`: `paper.previous_run` (la noche buena),
`paper.second_epoch_run` (la segunda) y `paper.run` (el cubo analizado).

uso: python scripts/cifras_por_noche_d2.py --target SLUG [--noche1 RUN] [--noche2 RUN] [--analizado RUN]
                                           [--stages-noche1 DIR --stages-noche2 DIR] [--json SALIDA]
"""
import argparse
import json
from pathlib import Path

import numpy as np
from astropy.io import fits

HA_A = 6562.8
C_KMS = 299792.458
ROOT = Path(__file__).resolve().parents[1]


def leer(path):
    with fits.open(path) as h:
        t = h[1].data
        return (np.asarray(t["wave_A"], float), np.asarray(t["flux"], float),
                np.asarray(t["flux_err_total"], float))


def integrado(w, f, *, centro=HA_A, media_A=6.0, cont_A=(10.0, 40.0)):
    """(flujo, continuo, máscara de la línea, anchos de canal)."""
    d = np.abs(w - centro)
    lin = d <= media_A
    c0 = float(np.nanmedian(f[(d >= cont_A[0]) & (d <= cont_A[1]) & np.isfinite(f)]))
    dl = np.gradient(w)
    return float(np.nansum((f[lin] - c0) * dl[lin])), c0, lin, dl


def error_integrado(e, lin, dl):
    return float(np.sqrt(np.nansum((e[lin] * dl[lin]) ** 2)))


def fwhm_kms(w, f, *, centro=HA_A, ventana_A=10.0, cont_A=(10.0, 40.0)):
    """FWHM por cruce del medio máximo (interpolado), el mismo estimador para los dos objetos."""
    d = np.abs(w - centro)
    c0 = np.nanmedian(f[(d >= cont_A[0]) & (d <= cont_A[1])])
    sel = d <= ventana_A
    x, y = w[sel], f[sel] - c0
    i = int(np.nanargmax(y))
    m = y[i] / 2
    j = i
    while j > 0 and y[j] > m:
        j -= 1
    izq = x[j] + (m - y[j]) * (x[j + 1] - x[j]) / (y[j + 1] - y[j])
    k = i
    while k < len(y) - 1 and y[k] > m:
        k += 1
    der = x[k - 1] + (m - y[k - 1]) * (x[k] - x[k - 1]) / (y[k] - y[k - 1])
    return C_KMS * float(der - izq) / float(x[i])


def noche(stages):
    ws, fs, es = leer(stages / "spec_calibrated_psffit_star.fits")
    fp, cp, lin, dl = integrado(ws, fs)
    wo, fo, eo = leer(stages / "spec_calibrated_psffit_object.fits")
    fc, _, lino, dlo = integrado(wo, fo)
    return {"F_prim": fp, "F_prim_err": error_integrado(es, lin, dl), "EW_prim_A": -fp / cp,
            "F_comp": fc, "F_comp_err": error_integrado(eo, lino, dlo),
            "fwhm_prim_kms": fwhm_kms(ws, fs), "fwhm_comp_kms": fwhm_kms(wo, fo)}


def controles_del_analizado(stages):
    w, f, _ = leer(stages / "spec_calibrated_psffit_object.fits")
    comp = integrado(w, f, cont_A=(20.0, 80.0))[0]
    cs = np.load(stages / "spec_calibrated_psffit_controls.npz")["control_spectra"]
    v = np.asarray([integrado(w, c, cont_A=(20.0, 80.0))[0] for c in cs])
    sd = float(v.std(ddof=1))
    return {"comp": comp, "media": float(v.mean()), "sd": sd, "n": int(v.size),
            "comp_sobre_sd": comp / sd, "media_en_sigmas": float(v.mean() / (sd / np.sqrt(v.size)))}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--target", required=True, help="slug de targets/<slug>.json")
    ap.add_argument("--noche1", default=None, help="por defecto, paper.previous_run del target")
    ap.add_argument("--noche2", default=None, help="por defecto, paper.second_epoch_run")
    ap.add_argument("--analizado", default=None, help="por defecto, paper.run")
    ap.add_argument("--stages-noche1", default=None, help="otra carpeta de productos (p. ej. una copia a 78 px)")
    ap.add_argument("--stages-noche2", default=None)
    ap.add_argument("--json", default=None)
    a = ap.parse_args(argv)
    paper = json.loads((ROOT / "targets" / f"{a.target}.json").read_text())["paper"]
    a.noche1 = a.noche1 or paper["previous_run"]
    a.noche2 = a.noche2 or paper["second_epoch_run"]
    a.analizado = a.analizado or paper["run"]
    s1 = Path(a.stages_noche1) if a.stages_noche1 else ROOT / "runs" / a.noche1 / "stages"
    s2 = Path(a.stages_noche2) if a.stages_noche2 else ROOT / "runs" / a.noche2 / "stages"
    n1, n2 = noche(s1), noche(s2)
    r = n2["F_prim"] / n1["F_prim"]
    r_err = r * np.hypot(n1["F_prim_err"] / n1["F_prim"], n2["F_prim_err"] / n2["F_prim"])
    pred = n1["F_comp"] / n1["F_prim"] * n2["F_prim"]
    out = {"noche1": n1, "noche2": n2, "razon_primaria": r, "razon_primaria_err": r_err,
           "fuga_prediccion_noche2": pred, "fuga_z": (pred - n2["F_comp"]) / n2["F_comp_err"],
           "controles_analizado": controles_del_analizado(ROOT / "runs" / a.analizado / "stages")}
    print(f"primaria noche2/noche1 = {r:.2f} ± {r_err:.2f}; EW {n1['EW_prim_A']:+.2f} -> {n2['EW_prim_A']:+.2f} Å")
    print(f"compañero: noche1 {n1['F_comp']:.0f} ± {n1['F_comp_err']:.0f}, noche2 {n2['F_comp']:.0f} ± {n2['F_comp_err']:.0f}")
    print(f"fuga: predicción noche2 {pred:.3g} -> {out['fuga_z']:.2f}σ")
    print(f"FWHM noche1: compañero {n1['fwhm_comp_kms']:.0f}, primaria {n1['fwhm_prim_kms']:.0f} km/s")
    c = out["controles_analizado"]
    print(f"{a.analizado}: compañero {c['comp']:.0f}; controles {c['media']:.0f} ± {c['sd']:.0f} (n={c['n']}); "
          f"compañero/sd {c['comp_sobre_sd']:.2f}; media {c['media_en_sigmas']:+.2f}σ")
    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
