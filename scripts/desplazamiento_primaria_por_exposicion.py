#!/usr/bin/env python
"""¿Está desplazada en λ alguna exposición? La primaria, exposición a exposición, y la AO de cada una.

Pregunta abierta de docs/2026-09-25_cifras_por_noche_a_55px.md §3: el segundo bloque de la noche
buena de ROXs 12 b tiene el centroide de Hα del compañero ~1.4 Å (~65 km/s) más al rojo. Si la
calibración en λ de esas exposiciones estuviera corrida, la PRIMARIA —muy brillante, con muchas
líneas fotosféricas— estaría corrida igual. Resultado y lectura en
docs/2026-09-26_desplazamiento_primaria_y_ao_por_exposicion.md.

Solo lectura: no escribe en runs/. Por cada exposición del plan de C7:

- espectro de la primaria (caja de 7x7 en el cubo alineado de C7), normalizado a su continuo;
- desplazamiento respecto a la mediana de la primera noche por mínimo χ² en tres ventanas sin Hα
  ni bandas telúricas (+ = al rojo), y su media pesada;
- centroide del Hα de la propia primaria;
- las condiciones de la cabecera primaria: seeing del DIMM, τ₀, viento, masa de aire, el Strehl
  que el DRS calcula de la telemetría del RTC de la AO, la FWHM que mide el propio DRS en el cubo,
  y el estado de los lazos de la AO.

Validado el 2026-09-25 sobre ROXs12b_invvar: precisión ~0.005 Å por exposición, y una inyección de
+1.40 Å se recupera como +1.41. Tarda ~5 min para 29 exposiciones.

uso: python scripts/desplazamiento_primaria_por_exposicion.py RUN SALIDA_PREFIJO [--n-prueba N]
     escribe SALIDA_PREFIJO.npz (espectros) y SALIDA_PREFIJO.json (medidas)
"""
import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

VENTANAS_A = {"6100-6520": (6100.0, 6520.0), "6600-6850": (6600.0, 6850.0),
              "8400-8750": (8400.0, 8750.0)}
PASOS_A = np.arange(-3.0, 3.0001, 0.01)
MITAD_CAJA_PX = 3

#: nombre corto -> clave de la cabecera primaria del cubo por exposición
CABECERA = {
    "exptime_s": "EXPTIME",
    "seeing_inicio_arcsec": "ESO TEL AMBI FWHM START",
    "seeing_fin_arcsec": "ESO TEL AMBI FWHM END",
    "tau0_s": "ESO TEL AMBI TAU0",
    "viento_ms": "ESO TEL AMBI WINDSP",
    "masa_de_aire": "ESO TEL AIRM START",
    "cielo_ir_C": "ESO TEL AMBI IRSKY TEMP",
    "strehl_rtc": "ESO DRS MUSE RTC STREHL",
    "strehl_rtc_mad": "ESO DRS MUSE RTC STREHERR",
    "fwhm_drs_x_arcsec": "ESO QC SCIPOST FWHM1 X",
    "fwhm_drs_y_arcsec": "ESO QC SCIPOST FWHM1 Y",
    "lazo_ho": "ESO AOS HO LOOP ST",
    "lazo_irlos": "ESO AOS IR LOOP ST",
    "lazo_jit": "ESO AOS JIT LOOP ST",
}


def condiciones(path):
    """Las claves de `CABECERA` de la cabecera primaria (None si falta)."""
    from astropy.io import fits
    h = fits.getheader(path, 0)
    out = {}
    for nombre, clave in CABECERA.items():
        v = h.get(clave)
        out[nombre] = v if isinstance(v, (bool, int, float, str)) or v is None else str(v)
    return out


def desplazamiento(wave, espectro, ref, lo, hi, pasos=PASOS_A):
    """Desplazamiento (Å, + = al rojo) que minimiza χ² contra `ref` en [lo, hi], con su error.

    El mínimo de la rejilla se refina con una parábola por los tres puntos que lo rodean; el error
    sale de la curvatura de χ² con la varianza de los residuos.
    """
    sel = (wave >= lo) & (wave <= hi) & np.isfinite(espectro) & np.isfinite(ref)
    x, y = wave[sel], espectro[sel]
    chi = np.asarray([np.nansum((y - np.interp(x - s, wave, ref)) ** 2) for s in pasos])
    paso = pasos[1] - pasos[0]
    k = int(np.argmin(chi))
    if 0 < k < len(pasos) - 1:
        a, b, c = chi[k - 1], chi[k], chi[k + 1]
        s0 = pasos[k] + 0.5 * (a - c) / (a - 2 * b + c) * paso
    else:
        s0 = pasos[k]
    var = np.nanvar(y - np.interp(x - s0, wave, ref))
    curv = (chi[min(k + 1, len(chi) - 1)] - 2 * chi[k] + chi[max(k - 1, 0)]) / paso ** 2
    err = float(np.sqrt(2.0 * var / curv)) if curv > 0 else float("nan")
    return float(s0), err


def centroide_ha(wave, espectro_normalizado):
    """Centroide de la emisión por encima del continuo (=1) en 6556-6570 Å."""
    sel = (wave >= 6556.0) & (wave <= 6570.0)
    y = np.clip(espectro_normalizado[sel] - 1.0, 0, None)
    return float(np.sum(wave[sel] * y) / np.sum(y)) if np.sum(y) > 0 else float("nan")


def extraer(run_id, n_prueba=None):
    from musepipe.stages import stage_x06_perexp as c7
    from musepipe.stages.stage_e01_perobs import load_aligned_exposure

    cfg = c7.stage_x06_config_from_run(run_id)
    paths = c7.stage_x06_paths(run_id)
    cache = paths["observation_plan_json"]
    obs = c7.resolve_observation_plan(run_id, plan_json=cache if cache.exists() else None)
    plan = obs.plan
    star_yx, _, _ = c7._positions(paths, plan)
    exps = list(obs.exposures)[:n_prueba] if n_prueba else list(obs.exposures)
    yy, xx = int(round(star_yx[0])), int(round(star_yx[1]))
    m = MITAD_CAJA_PX
    print(f"{run_id}: {len(exps)} exposiciones, primaria en [{yy}, {xx}]", flush=True)

    def una(e):
        cube, _stat, wave = load_aligned_exposure(e, plan)
        prim = np.nansum(cube[:, yy - m:yy + m + 1, xx - m:xx + m + 1], axis=(1, 2)).astype(np.float64)
        del cube
        return str(e.exposure_id), float(e.mjd_obs), np.asarray(wave, np.float64), prim, condiciones(e.file)

    with ThreadPoolExecutor(max_workers=int(cfg["x06_max_workers"])) as pool:
        filas = list(pool.map(una, exps))
    return {"ids": np.asarray([f[0] for f in filas]), "mjd": np.asarray([f[1] for f in filas]),
            "wave": filas[0][2], "prim": np.asarray([f[3] for f in filas]),
            "condiciones": [f[4] for f in filas], "primaria_yx": [yy, xx]}


def analizar(run_id, ex):
    from musepipe.spectral import continuum_running_median

    ids, mjd, wave, prim = ex["ids"], ex["mjd"], ex["wave"], ex["prim"]
    good = np.isfinite(wave)
    nrm = np.asarray([p / continuum_running_median(wave, p, good & np.isfinite(p), window_A=60.0,
                                                   min_pixels=15) for p in prim])
    noche = np.asarray([i.split("_")[0] for i in ids])
    noches = sorted(set(noche.tolist()))
    orden = np.argsort(mjd)
    N1 = [int(i) for i in orden if noche[i] == noches[0]]
    N2 = [int(i) for i in orden if len(noches) > 1 and noche[i] == noches[1]]
    ref = np.nanmedian(nrm[N1], axis=0)

    por_exp = []
    for i in range(len(ids)):
        fila = {"id": str(ids[i]), "mjd": float(mjd[i]), "noche": str(noche[i]),
                "centroide_ha_A": centroide_ha(wave, nrm[i])}
        for nombre, (lo, hi) in VENTANAS_A.items():
            fila[nombre] = desplazamiento(wave, nrm[i], ref, lo, hi)
        s = np.asarray([fila[n][0] for n in VENTANAS_A])
        w = 1 / np.asarray([fila[n][1] for n in VENTANAS_A]) ** 2
        fila["media_A"] = float(np.sum(w * s) / np.sum(w))
        fila["media_err_A"] = float(1 / np.sqrt(np.sum(w)))
        fila["condiciones"] = ex["condiciones"][i]
        por_exp.append(fila)

    out = {"run": run_id, "primaria_yx": ex["primaria_yx"], "ventanas_A": VENTANAS_A,
           "orden_primera_noche": N1, "orden_segunda_noche": N2, "por_exposicion": por_exp,
           "bloques": {}}
    for nb in (3, 4):
        partes = [list(map(int, p)) for p in np.array_split(np.asarray(N1), nb)] if len(N1) >= nb else []
        out["bloques"][str(nb)] = [{
            "n": len(p),
            "desplazamiento_medio_A": float(np.mean([por_exp[i]["media_A"] for i in p])),
            "dispersion_A": float(np.std([por_exp[i]["media_A"] for i in p], ddof=1)) if len(p) > 1 else None,
            "centroide_ha_medio_A": float(np.nanmean([por_exp[i]["centroide_ha_A"] for i in p])),
            "strehl_rtc_mediano": _mediana(por_exp, p, "strehl_rtc"),
            "tau0_mediano_s": _mediana(por_exp, p, "tau0_s"),
        } for p in partes]
    out["noche2"] = {"n": len(N2), "desplazamiento_medio_A": (
        float(np.mean([por_exp[i]["media_A"] for i in N2])) if N2 else None),
        "strehl_rtc_mediano": _mediana(por_exp, N2, "strehl_rtc") if N2 else None,
        "tau0_mediano_s": _mediana(por_exp, N2, "tau0_s") if N2 else None}
    return out


def _mediana(por_exp, idx, clave):
    v = [por_exp[i]["condiciones"].get(clave) for i in idx]
    v = [float(x) for x in v if isinstance(x, (int, float)) and not isinstance(x, bool)]
    return float(np.median(v)) if v else None


def imprimir(out):
    por_exp = out["por_exposicion"]
    print(f"{'exposición':42s} " + " ".join(f"{n:>14s}" for n in VENTANAS_A)
          + "   media (Å)       Hα prim  Strehl  τ₀ (ms)")
    for i in out["orden_primera_noche"] + out["orden_segunda_noche"]:
        f = por_exp[i]
        c = f["condiciones"]
        st = c.get("strehl_rtc")
        t0 = c.get("tau0_s")
        print(f"{f['id'][:42]:42s} " + " ".join(f"{f[n][0]:+7.3f}±{f[n][1]:.3f}" for n in VENTANAS_A)
              + f"   {f['media_A']:+.3f}±{f['media_err_A']:.3f}   {f['centroide_ha_A']:.2f}"
              + (f"  {st:6.2f}" if isinstance(st, (int, float)) else "     --")
              + (f"  {t0 * 1e3:5.1f}" if isinstance(t0, (int, float)) else "     --"))
    for nb, bl in out["bloques"].items():
        print(f"{nb} bloques: " + " · ".join(
            f"{b['desplazamiento_medio_A']:+.3f} Å (sd {b['dispersion_A'] or 0:.3f}; "
            f"Hα {b['centroide_ha_medio_A']:.2f}; Strehl {b['strehl_rtc_mediano']})" for b in bl))
    print(f"noche 2: {out['noche2']}")
    print("Si el bloque 2 estuviera corrido como el compañero, su media saldría cerca de +1.4 Å.")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("run_id")
    ap.add_argument("salida_prefijo", help="escribe <prefijo>.npz y <prefijo>.json")
    ap.add_argument("--n-prueba", type=int, default=None, help="solo las N primeras exposiciones")
    a = ap.parse_args(argv)
    ex = extraer(a.run_id, a.n_prueba)
    np.savez_compressed(a.salida_prefijo + ".npz", ids=ex["ids"], mjd=ex["mjd"], wave=ex["wave"],
                        prim=ex["prim"])
    out = analizar(a.run_id, ex)
    with open(a.salida_prefijo + ".json", "w") as fh:
        json.dump(out, fh, indent=1, default=float)
    imprimir(out)


if __name__ == "__main__":
    main()
