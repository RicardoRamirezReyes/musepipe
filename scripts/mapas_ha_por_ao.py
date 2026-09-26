#!/usr/bin/env python
"""¿Dónde está, en la imagen, el exceso rojo de Hα del compañero cuando la AO cae?

Solo lectura. Continúa docs/2026-09-26_desplazamiento_primaria_y_ao_por_exposicion.md §7: en la noche buena
de ROXs 12 la asimetría rojo-menos-azul de la línea del compañero sigue al Strehl del RTC y ningún control
lo hace. Si el exceso rojo de las exposiciones de AO baja es la luz del propio compañero, en un mapa
aparece puntual y centrado en él; si es estructura del halo de la primaria, aparece desplazado o extendido.

Por cada exposición de la primera noche (cubo alineado de C7): recorte de (2·mitad+1)² px centrado en el
compañero, continuo por spaxel (mediana entre 10 y 40 Å de 6562.8 Å) y dos imágenes de línea con el
continuo restado, ala azul [6556.8, 6562.8) y ala roja (6562.8, 6568.8]. Se promedian por grupo de Strehl
(umbral 12 %, como `centroide_companero_por_ao.py`). Medidas: rojo − azul en una caja 3×3 sobre el
compañero y en las mismas cajas sobre posiciones de la corona a la misma separación, y el centroide del
exceso rojo − azul de (AO baja − AO alta) alrededor del compañero.

uso: python scripts/mapas_ha_por_ao.py RUN COND_JSON PREFIJO [--umbral-strehl 12] [--mitad 20]
     COND_JSON sale de scripts/desplazamiento_primaria_por_exposicion.py; escribe PREFIJO.npz,
     PREFIJO.json y PREFIJO.png
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

CENTRO_A = 6562.8
MEDIA_A = 6.0
CONTINUO_A = (10.0, 40.0)


def mapas_de_linea(wave, cubo):
    """(azul, rojo) en flujo·Å por spaxel, con el continuo de cada spaxel restado."""
    dist = np.abs(wave - CENTRO_A)
    cont = (dist >= CONTINUO_A[0]) & (dist <= CONTINUO_A[1])
    base = np.nanmedian(cubo[cont], axis=0)
    dl = np.gradient(wave)[:, None, None]
    azul = (wave >= CENTRO_A - MEDIA_A) & (wave < CENTRO_A)
    rojo = (wave > CENTRO_A) & (wave <= CENTRO_A + MEDIA_A)
    return (np.nansum((cubo[azul] - base) * dl[azul], axis=0),
            np.nansum((cubo[rojo] - base) * dl[rojo], axis=0))


def caja3(img, y, x):
    yy, xx = int(round(y)), int(round(x))
    return float(np.nansum(img[yy - 1:yy + 2, xx - 1:xx + 2]))


def centroide_positivo(img, y0, x0, radio):
    yy, xx = np.indices(img.shape)
    sel = np.hypot(yy - y0, xx - x0) <= radio
    w = np.clip(np.where(sel, img, 0.0), 0, None)
    if w.sum() <= 0:
        return None
    return float((w * yy).sum() / w.sum()), float((w * xx).sum() / w.sum())


def strehl_o_posang(azul, rojo, strehl, posang, cy, cx, n_perm=5000, semilla=20260926):
    """¿Sigue el rojo − azul del compañero al Strehl o a la posición en el detector?

    El compañero no está en el centro de rotación: con cada POSANG cae en otras rebanadas del IFU,
    así que un efecto local de calibración seguiría al POSANG. Regresión rb ~ Strehl + cos + sin
    (POSANG), y permutación del Strehl solo entre exposiciones del mismo POSANG.
    """
    from scipy.stats import spearmanr
    rb = np.asarray([caja3(r, cy, cx) - caja3(z, cy, cx) for z, r in zip(azul, rojo)])
    rad = np.radians(posang)
    X = np.column_stack([np.ones_like(strehl), strehl, np.cos(rad), np.sin(rad)])
    coef = np.linalg.lstsq(X, rb, rcond=None)[0]
    res = rb - X @ coef
    se = np.sqrt(np.diag(res @ res / (len(rb) - X.shape[1]) * np.linalg.inv(X.T @ X)))
    rho = float(spearmanr(strehl, rb)[0])
    rng = np.random.default_rng(semilla)
    grupos = [np.where(posang == p)[0] for p in np.unique(posang)]
    n = 0
    for _ in range(n_perm):
        s = strehl.copy()
        for g in grupos:
            if len(g) > 1:
                s[g] = rng.permutation(strehl[g])
        n += spearmanr(s, rb)[0] <= rho
    return {"rojo_menos_azul": rb.tolist(), "rho_strehl": rho,
            "p_strehl": float(spearmanr(strehl, rb)[1]),
            "regresion": {k: {"coef": float(c), "err": float(e)}
                          for k, c, e in zip(("const", "strehl", "cos_posang", "sin_posang"), coef, se)},
            "p_permutando_dentro_de_cada_posang": float((n + 1) / (n_perm + 1))}


def main(argv=None):
    from musepipe.apertures import same_radius_control_positions
    from musepipe.stages import stage_x06_perexp as c7
    from musepipe.stages.stage_e01_perobs import load_aligned_exposure

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("run_id")
    ap.add_argument("cond_json")
    ap.add_argument("prefijo")
    ap.add_argument("--umbral-strehl", type=float, default=12.0)
    ap.add_argument("--mitad", type=int, default=20, help="medio lado del recorte en px")
    a = ap.parse_args(argv)

    cond = {f["id"]: f["condiciones"] for f in json.load(open(a.cond_json))["por_exposicion"]}
    cfg = c7.stage_x06_config_from_run(a.run_id)
    paths = c7.stage_x06_paths(a.run_id)
    cache = paths["observation_plan_json"]
    obs = c7.resolve_observation_plan(a.run_id, plan_json=cache if cache.exists() else None)
    plan = obs.plan
    star_yx, comp_yx, _ = c7._positions(paths, plan)
    npix = int(plan.crop_npix)
    noches = sorted({str(e.exposure_id).split("_")[0] for e in obs.exposures})
    exps = sorted([e for e in obs.exposures if str(e.exposure_id).startswith(noches[0])],
                  key=lambda e: e.mjd_obs)
    m = int(a.mitad)
    cy, cx = int(round(comp_yx[0])), int(round(comp_yx[1]))
    y0, y1, x0, x1 = cy - m, cy + m + 1, cx - m, cx + m + 1
    if y0 < 0 or x0 < 0 or y1 > npix or x1 > npix:
        raise SystemExit(f"el recorte [{y0}:{y1}, {x0}:{x1}] se sale del cubo de {npix} px")
    print(f"{a.run_id}: {len(exps)} exposiciones de {noches[0]}, compañero en [{cy}, {cx}]", flush=True)

    def una(e):
        from astropy.io import fits
        posang = float(fits.getheader(e.file, 0).get("ESO INS DROT POSANG", np.nan))
        cubo, _stat, wave = load_aligned_exposure(e, plan)
        wave = np.asarray(wave, np.float64)
        ventana = np.abs(wave - CENTRO_A) <= CONTINUO_A[1]
        rec = np.asarray(cubo[ventana, y0:y1, x0:x1], np.float64)
        del cubo
        azul, rojo = mapas_de_linea(wave[ventana], rec)
        return str(e.exposure_id), azul, rojo, posang

    with ThreadPoolExecutor(max_workers=int(cfg["x06_max_workers"])) as pool:
        filas = list(pool.map(una, exps))
    ids = [f[0] for f in filas]
    azul = np.asarray([f[1] for f in filas])
    rojo = np.asarray([f[2] for f in filas])
    posang = np.asarray([f[3] for f in filas])
    strehl = np.asarray([float(cond[i]["strehl_rtc"]) for i in ids])
    baja, alta = strehl < a.umbral_strehl, strehl >= a.umbral_strehl
    np.savez_compressed(a.prefijo + ".npz", ids=np.asarray(ids), azul=azul, rojo=rojo, strehl=strehl, posang=posang,
                        recorte=np.asarray([y0, y1, x0, x1]), comp_yx=np.asarray(comp_yx),
                        star_yx=np.asarray(star_yx))

    # posiciones de la corona que caen dentro del recorte (misma separacion, lejos del compañero)
    ctr = np.asarray(same_radius_control_positions(tuple(comp_yx), tuple(star_yx), npix, npix,
                                                   n_positions=38))
    dentro = [(p[0] - y0, p[1] - x0) for p in ctr if y0 + 2 <= p[0] < y1 - 2 and x0 + 2 <= p[1] < x1 - 2]
    cyr, cxr = comp_yx[0] - y0, comp_yx[1] - x0
    out = {"run": a.run_id, "umbral_strehl": a.umbral_strehl, "n_baja": int(baja.sum()),
           "n_alta": int(alta.sum()), "recorte": [y0, y1, x0, x1], "grupos": {}}
    medias = {}
    for nombre, sel in (("ao_baja", baja), ("ao_alta", alta)):
        az, ro = np.nanmean(azul[sel], axis=0), np.nanmean(rojo[sel], axis=0)
        medias[nombre] = (az, ro)
        out["grupos"][nombre] = {
            "companero_azul": caja3(az, cyr, cxr), "companero_rojo": caja3(ro, cyr, cxr),
            "companero_rojo_menos_azul": caja3(ro - az, cyr, cxr),
            "corona_rojo_menos_azul": [caja3(ro - az, py, px) for py, px in dentro],
        }
    dif = (medias["ao_baja"][1] - medias["ao_baja"][0]) - (medias["ao_alta"][1] - medias["ao_alta"][0])
    c = centroide_positivo(dif, cyr, cxr, 6.0)
    out["diferencia"] = {"companero": caja3(dif, cyr, cxr),
                         "corona": [caja3(dif, py, px) for py, px in dentro],
                         "centroide_exceso_px": (None if c is None else [c[0] - cyr, c[1] - cxr])}
    out["strehl_o_posang"] = strehl_o_posang(azul, rojo, strehl, posang, cyr, cxr)
    with open(a.prefijo + ".json", "w") as fh:
        json.dump(out, fh, indent=1)

    for nombre, g in out["grupos"].items():
        cor = np.asarray(g["corona_rojo_menos_azul"])
        corona = (f"corona rojo−azul {cor.mean():+.0f} ± {cor.std(ddof=1):.0f} (n={cor.size})" if cor.size > 1
                  else "sin posiciones de la corona en el recorte (ver centroide_companero_por_ao.py)")
        print(f"{nombre}: compañero azul {g['companero_azul']:+.0f} rojo {g['companero_rojo']:+.0f} "
              f"rojo−azul {g['companero_rojo_menos_azul']:+.0f} | {corona}")
    cen = out["diferencia"]["centroide_exceso_px"]
    print(f"(baja − alta) rojo−azul: compañero {out['diferencia']['companero']:+.0f}; centroide del exceso "
          + ("—" if cen is None else f"a ({cen[0]:+.2f}, {cen[1]:+.2f}) px") + " del compañero")

    sp = out["strehl_o_posang"]
    r = sp["regresion"]
    print(f"rojo−azul contra Strehl: rho = {sp['rho_strehl']:+.2f} (p = {sp['p_strehl']:.3f}); regresion "
          + ", ".join(f"{k} {v['coef'] / v['err']:+.1f}σ" for k, v in r.items() if k != "const")
          + f"; permutando dentro de cada POSANG p = {sp['p_permutando_dentro_de_cada_posang']:.3f}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 4, figsize=(12, 6), constrained_layout=True)
    for fila, nombre in enumerate(("ao_baja", "ao_alta")):
        az, ro = medias[nombre]
        for col, (img, titulo) in enumerate(((az, "blue wing"), (ro, "red wing"), (ro - az, "red − blue"))):
            lim = np.nanpercentile(np.abs(img), 99)
            ax[fila, col].imshow(img, origin="lower", cmap="RdBu_r", vmin=-lim, vmax=lim)
            ax[fila, col].plot(cxr, cyr, "k+", ms=10)
            ax[fila, col].set_title(f"{'low' if nombre == 'ao_baja' else 'high'} Strehl: {titulo}", fontsize=9)
    lim = np.nanpercentile(np.abs(dif), 99)
    ax[0, 3].imshow(dif, origin="lower", cmap="RdBu_r", vmin=-lim, vmax=lim)
    ax[0, 3].plot(cxr, cyr, "k+", ms=10)
    for py, px in dentro:
        ax[0, 3].plot(px, py, "o", mfc="none", mec="0.4", ms=6)
    ax[0, 3].set_title("(low − high) red − blue", fontsize=9)
    ax[1, 3].axis("off")
    for axx in ax.ravel():
        axx.set_xticks([]); axx.set_yticks([])
    fig.savefig(a.prefijo + ".png", dpi=110)


if __name__ == "__main__":
    main()
