#!/usr/bin/env python
"""El compañero por ajuste de PSF, exposición a exposición, en la ventana de Hα.

Solo lectura. Continúa §7-§8 de docs/2026-09-26_desplazamiento_primaria_y_ao_por_exposicion.md: con la
extracción de C7 (apertura box3 y fondo de anillo) la asimetría rojo-menos-azul de la línea del
compañero sigue al Strehl del RTC, y ningún control lo hace. Aquí la misma pregunta con otra
extracción: el ajuste de C4 (PSF de la primaria + PSF del compañero + plano de fondo, canal a canal,
dentro de sus radios), con **el modelo de PSF de cada exposición** (la mezcla de C1b) sobre su cubo
alineado. Si el efecto es del fondo de anillo, aquí desaparece; si es de la luz del compañero, sigue.

Los controles se ajustan idénticos al objeto: las mismas posiciones a la misma separación que usa C4
(`x03_control_apertures`, `x03_control_exclude_angle_deg`), la misma máscara y el mismo diseño.

Para que quepa en minutos, el diseño de C4 (`psffit.psf_pair_design`) se evalúa solo en los píxeles de
la máscara de ajuste: es la misma fórmula y el mismo ajuste por mínimos cuadrados que
`psffit._fit_one_channel` (que tampoco usa otros píxeles); `test_ajuste_en_la_mascara_es_el_de_c4`
comprueba que los coeficientes coinciden.

uso: python scripts/psffit_por_exposicion.py RUN COND_JSON PREFIJO [--ventana 40] [--umbral-strehl 12]
     COND_JSON sale de scripts/desplazamiento_primaria_por_exposicion.py; escribe PREFIJO.npz y .json
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
SEMILLA = 20260926


def ajuste_en_la_mascara(data_2d, var_2d, wave_A, star_yx, center_yx, psf_model, mask):
    """Coeficientes de C4 (primaria, compañero, cte, y, x) para un canal, evaluando solo en `mask`.

    Replica `psffit.psf_pair_design` + `psffit._fit_one_channel` sobre los píxeles de la máscara.
    """
    from musepipe.psf import evaluate_psf_model
    yy, xx = np.nonzero(mask)
    yy = yy.astype(np.float64)
    xx = xx.astype(np.float64)
    p_star = evaluate_psf_model(psf_model, float(wave_A), yy - float(star_yx[0]), xx - float(star_yx[1]))
    p_comp = evaluate_psf_model(psf_model, float(wave_A), yy - float(center_yx[0]), xx - float(center_yx[1]))
    y0 = 0.5 * (float(star_yx[0]) + float(center_yx[0]))
    x0 = 0.5 * (float(star_yx[1]) + float(center_yx[1]))
    escala = max(float(np.hypot(float(center_yx[0]) - float(star_yx[0]),
                                float(center_yx[1]) - float(star_yx[1]))), 1.0)
    a = np.stack([p_star, p_comp, np.ones_like(yy), (yy - y0) / escala, (xx - x0) / escala], axis=-1)
    y = np.asarray(data_2d, np.float64)[mask]
    var = np.asarray(var_2d, np.float64)[mask]
    ok = np.isfinite(y) & np.isfinite(var) & (var > 0) & np.all(np.isfinite(a), axis=-1)
    if np.count_nonzero(ok) <= a.shape[1]:
        return np.full(a.shape[1], np.nan)
    sw = 1.0 / np.sqrt(var[ok])
    coef, *_ = np.linalg.lstsq(a[ok] * sw[:, None], y[ok] * sw, rcond=None)
    return coef


def rojo_menos_azul(wave, espectro):
    dist = np.abs(wave - CENTRO_A)
    base = np.nanmedian(espectro[(dist >= CONTINUO_A[0]) & (dist <= CONTINUO_A[1])])
    dl = np.gradient(wave)
    rojo = (wave > CENTRO_A) & (wave <= CENTRO_A + MEDIA_A)
    azul = (wave >= CENTRO_A - MEDIA_A) & (wave < CENTRO_A)
    return float(np.nansum((espectro[rojo] - base) * dl[rojo]) - np.nansum((espectro[azul] - base) * dl[azul]))


def integrado(wave, espectro):
    dist = np.abs(wave - CENTRO_A)
    base = np.nanmedian(espectro[(dist >= CONTINUO_A[0]) & (dist <= CONTINUO_A[1])])
    sel = dist <= MEDIA_A
    return float(np.nansum((espectro[sel] - base) * np.gradient(wave)[sel]))


def centroide(wave, espectro):
    dist = np.abs(wave - CENTRO_A)
    base = np.nanmedian(espectro[(dist >= CONTINUO_A[0]) & (dist <= CONTINUO_A[1])])
    sel = dist <= MEDIA_A
    y = np.clip(espectro[sel] - base, 0, None)
    return float(np.sum(wave[sel] * y) / np.sum(y)) if np.sum(y) > 0 else float("nan")


def extraer(run_id, ventana_A):
    from astropy.io import fits
    from musepipe.apertures import same_radius_control_positions
    from musepipe.extraction.psffit import fit_region_mask
    from musepipe.io import read_json
    from musepipe.stages import stage_x03_psffit as c4
    from musepipe.stages import stage_x06_perexp as c7
    from musepipe.stages.stage_e01_perobs import load_aligned_exposure

    cfg4 = c4.stage_x03_config_from_run(run_id)
    cfg7 = c7.stage_x06_config_from_run(run_id)
    paths = c7.stage_x06_paths(run_id)
    doc = read_json(paths["psf_model_mixture_json"])
    if str(doc.get("form", "")).lower() != "mixture":
        raise SystemExit(f"{run_id}: psf_model_mixture.json no es una mezcla por exposición")
    modelos = c7.models_by_exposure(doc)
    cache = paths["observation_plan_json"]
    obs = c7.resolve_observation_plan(run_id, plan_json=cache if cache.exists() else None)
    plan = obs.plan
    star_yx, comp_yx, _ = c7._positions(paths, plan)
    npix = int(plan.crop_npix)
    r_star, r_comp = float(cfg4["x03_star_radius_px"]), float(cfg4["x03_comp_radius_px"])
    controles = same_radius_control_positions(
        comp_yx, star_yx, npix, npix, n_positions=int(cfg4["x03_control_apertures"]),
        exclude_angle_deg=float(cfg4["x03_control_exclude_angle_deg"]),
        margin_px=int(np.ceil(r_comp)) + 1)
    centros = [tuple(comp_yx)] + [tuple(c) for c in controles]
    mascaras = [fit_region_mask((npix, npix), star_yx, c, star_radius_px=r_star, comp_radius_px=r_comp)
                for c in centros]
    print(f"{run_id}: {len(obs.exposures)} exposiciones, {len(controles)} controles", flush=True)

    def una(e):
        posang = float(fits.getheader(e.file, 0).get("ESO INS DROT POSANG", np.nan))
        cubo, stat, wave = load_aligned_exposure(e, plan)
        wave = np.asarray(wave, np.float64)
        zs = np.nonzero(np.abs(wave - CENTRO_A) <= ventana_A)[0]
        modelo = modelos[e.exposure_id]
        comp = np.full((len(centros), zs.size), np.nan)
        for k, z in enumerate(zs):
            for j, (c, m) in enumerate(zip(centros, mascaras)):
                comp[j, k] = ajuste_en_la_mascara(cubo[z], stat[z], wave[z], star_yx, c, modelo, m)[1]
        del cubo, stat
        print(f"  {e.exposure_id}", flush=True)
        return str(e.exposure_id), float(e.mjd_obs), wave[zs], comp, posang

    with ThreadPoolExecutor(max_workers=int(cfg7["x06_max_workers"])) as pool:
        filas = list(pool.map(una, obs.exposures))
    return {"ids": np.asarray([f[0] for f in filas]), "mjd": np.asarray([f[1] for f in filas]),
            "wave": filas[0][2], "comp": np.asarray([f[3][0] for f in filas]),
            "controls": np.asarray([f[3][1:] for f in filas]), "posang": np.asarray([f[4] for f in filas]),
            "comp_yx": np.asarray(comp_yx), "star_yx": np.asarray(star_yx),
            "control_yx": np.asarray(controles)}


def analizar(ex, cond, umbral, n_perm=5000):
    from scipy.stats import spearmanr
    wave = ex["wave"]
    ids = [str(i) for i in ex["ids"]]
    noche = np.asarray([i.split("_")[0] for i in ids])
    N1 = [int(i) for i in np.argsort(ex["mjd"]) if noche[i] == sorted(set(noche.tolist()))[0]]
    st = np.asarray([float(cond[ids[i]]["strehl_rtc"]) for i in N1])
    pa = ex["posang"][N1]
    comp = ex["comp"][N1]
    ctrl = ex["controls"][N1]
    rb_c = np.asarray([rojo_menos_azul(wave, s) for s in comp])
    rho_c, p_c = spearmanr(st, rb_c)
    rho_ctrl = np.asarray([spearmanr(st, [rojo_menos_azul(wave, ctrl[i, j]) for i in range(len(N1))])[0]
                           for j in range(ctrl.shape[1])])
    # peso de cada exposicion: 1/sigma^2 de la linea integrada en sus controles (como C7)
    sig = np.asarray([np.nanstd([integrado(wave, c) for c in ctrl[i]], ddof=1) for i in range(len(N1))])
    w = 1.0 / sig ** 2

    def comb(idx):
        ww = w[idx] / w[idx].sum()
        return np.nansum(ww[:, None] * comp[idx], axis=0)

    baja = np.nonzero(st < umbral)[0]
    alta = np.nonzero(st >= umbral)[0]
    dif = centroide(wave, comb(baja)) - centroide(wave, comb(alta))
    rng = np.random.default_rng(SEMILLA)
    n = 0
    for _ in range(n_perm):
        p = rng.permutation(len(N1))
        d = centroide(wave, comb(p[:baja.size])) - centroide(wave, comb(p[baja.size:]))
        n += abs(d) >= abs(dif)
    rad = np.radians(pa)
    X = np.column_stack([np.ones_like(st), st, np.cos(rad), np.sin(rad)])
    coef = np.linalg.lstsq(X, rb_c, rcond=None)[0]
    res = rb_c - X @ coef
    se = np.sqrt(np.diag(res @ res / (len(rb_c) - X.shape[1]) * np.linalg.inv(X.T @ X)))
    return {
        "n_exposiciones": len(N1), "n_controles": int(ctrl.shape[1]), "umbral_strehl": umbral,
        "rojo_menos_azul": {"rho_companero": float(rho_c), "p_companero": float(p_c),
                            "rho_controles_media": float(np.mean(rho_ctrl)),
                            "rho_controles_sd": float(np.std(rho_ctrl, ddof=1)),
                            "controles_tan_extremos": int(np.sum(rho_ctrl <= rho_c) if rho_c < 0
                                                          else np.sum(rho_ctrl >= rho_c))},
        "integrado": {"rho_companero": float(spearmanr(st, [integrado(wave, s) for s in comp])[0])},
        "centroide": {"ao_baja_A": centroide(wave, comb(baja)), "ao_alta_A": centroide(wave, comb(alta)),
                      "baja_menos_alta_A": float(dif), "p_permutacion": float((n + 1) / (n_perm + 1)),
                      "n_baja": int(baja.size), "n_alta": int(alta.size)},
        "regresion_posang": {k: {"coef": float(c), "err": float(e)}
                             for k, c, e in zip(("const", "strehl", "cos_posang", "sin_posang"), coef, se)},
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("run_id")
    ap.add_argument("cond_json")
    ap.add_argument("prefijo")
    ap.add_argument("--ventana", type=float, default=40.0, help="medio ancho en Å alrededor de Hα")
    ap.add_argument("--umbral-strehl", type=float, default=12.0)
    a = ap.parse_args(argv)
    cond = {f["id"]: f["condiciones"] for f in json.load(open(a.cond_json))["por_exposicion"]}
    ex = extraer(a.run_id, a.ventana)
    np.savez_compressed(a.prefijo + ".npz", **ex)
    out = analizar(ex, cond, a.umbral_strehl)
    with open(a.prefijo + ".json", "w") as fh:
        json.dump(out, fh, indent=1)
    rb, ce, rg = out["rojo_menos_azul"], out["centroide"], out["regresion_posang"]
    print(f"rojo−azul contra Strehl: compañero rho = {rb['rho_companero']:+.2f} (p = {rb['p_companero']:.3f}); "
          f"controles {rb['rho_controles_media']:+.2f} ± {rb['rho_controles_sd']:.2f}, "
          f"{rb['controles_tan_extremos']}/{out['n_controles']} tan extremos")
    print(f"integrado contra Strehl: rho = {out['integrado']['rho_companero']:+.2f}")
    print(f"centroide: AO baja {ce['ao_baja_A']:.2f} ({ce['n_baja']}) · alta {ce['ao_alta_A']:.2f} ({ce['n_alta']}) "
          f"→ {ce['baja_menos_alta_A']:+.2f} Å, p = {ce['p_permutacion']:.4f}")
    print("regresion rojo−azul: " + ", ".join(f"{k} {v['coef'] / v['err']:+.1f}σ"
                                              for k, v in rg.items() if k != "const"))


if __name__ == "__main__":
    main()
