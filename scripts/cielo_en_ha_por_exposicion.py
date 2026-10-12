#!/usr/bin/env python
"""¿Puede un residuo de cielo en las rebanadas del compañero fabricar el rojo − azul del bloque 2?

Solo lectura. Última hipótesis abierta del §5 de docs/2026-09-26_desplazamiento_primaria_y_ao_por_exposicion.md:
con cada POSANG el compañero cae en otras rebanadas del IFU, y un residuo de la resta de cielo propio de
esas rebanadas seguiría a su posición, no a la de los controles. En la noche buena, el rojo − azul del
Hα del compañero cambia de signo con la AO (§7-§9) y ningún control lo comparte.

Hay una línea de cielo dentro de la ventana de la línea: en los `SKY_SPECTRUM` de A4 (topocéntricos,
por espaxel) aparece un pico en ~6563 Å (Hα geocoronal u OH) sobre el continuo del cielo. Un residuo
de cielo es **aditivo**, así que su efecto sobre el rojo − azul tiene una cota que no depende de la
rebanada ni del marco: la línea entera, sin restar, cayendo toda en una sola de las dos ventanas.

  I_sky  = ∫ (cielo − continuo) dλ en ±6 Å del pico de la línea, por espaxel     (mismas ventanas que el Hα)
  cota   = 9 · I_sky · apcorr(Hα)          (caja 3×3 de C7 y su corrección de apertura en Hα, por exposición)

y se compara con el cambio observado entre grupos de Strehl. Eso cubre un residuo de INTENSIDAD (la
rebanada resta de más o de menos). Un residuo de λ (la rebanada con su solución en λ corrida δ) es un
dipolo, + en una ventana y − en la otra, y llega hasta 2·cota: se calcula aparte, desplazando el cielo
mediano de la noche δ en el marco del cubo, y se busca el δ que haría falta para el cambio observado con
el caso más favorable a la hipótesis (+δ en las exposiciones de AO baja, −δ en las de AO alta). Se
compara con el residuo de la solución en λ que mide A4/M1. El marco del cubo es baricéntrico y el del
`SKY_SPECTRUM` topocéntrico (aunque su cabecera diga `SPECSYS=BARYCENT`): una línea de cielo cae en el
cubo en λ·(1 + BARYCORR/c). `--marco N` lo mide en N exposiciones, correlando las líneas de cielo de la
extensión STAT del cubo (7700-9300 Å, mediana de los espaxeles más débiles) con el `SKY_SPECTRUM`
desplazado; así se encontró el 2026-09-28 que `cube_qc.expected_skyline_wave` tenía el signo al revés.
El residuo de λ de una rebanada se compara con el de A4/M1 (cielo promediado en el campo) y, con
`--s0-qc`, con el p95 por espaxel del mapa de S0.

Entradas, las mismas de `centroide_companero_por_ao.py` (y su definición de rojo − azul, importada):

  C7_NPZ    `scripts/cifras_por_noche_c7.py extraer RUN C7_NPZ`
  COND_JSON `scripts/desplazamiento_primaria_por_exposicion.py RUN PREFIJO` (condiciones por exposición)
  RUN       para los `SKY_SPECTRUM` que lista el QC de A4 (`m1_wavelength.sky_spectrum_files`)

uso: python scripts/cielo_en_ha_por_exposicion.py RUN C7_NPZ COND_JSON [--umbral-strehl 12] [--marco N]
                                                    [--s0-qc stageS0_qc.json] [--json SALIDA]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from centroide_companero_por_ao import CENTRO_A, CONTINUO_A, MEDIA_VENTANA_A, Grupos  # noqa: E402

#: Búsqueda del pico de la línea de cielo, en el marco topocéntrico del SKY_SPECTRUM.
BUSQUEDA_A = (6559.0, 6567.0)
#: Suavizado para localizar el pico: la LSF de MUSE en Hα (~2.3-2.5 Å FWHM → σ ≈ 1 Å).
SIGMA_LSF_A = 1.0


def pico_de_cielo(wave, flux, busqueda=BUSQUEDA_A, sigma_A=SIGMA_LSF_A):
    """λ del máximo del cielo suavizado a la LSF dentro de `busqueda`."""
    from scipy.ndimage import gaussian_filter1d
    wave, flux = np.asarray(wave, float), np.asarray(flux, float)
    paso = float(np.median(np.diff(wave)))
    suave = gaussian_filter1d(np.nan_to_num(flux, nan=float(np.nanmedian(flux))), sigma_A / paso)
    sel = (wave >= busqueda[0]) & (wave <= busqueda[1])
    return float(wave[sel][np.argmax(suave[sel])])


def integral_de_linea(wave, flux, centro, media_A=MEDIA_VENTANA_A, continuo_A=CONTINUO_A):
    """∫ (flujo − continuo) dλ en ±media_A de `centro`; continuo = mediana a continuo_A de distancia."""
    wave, flux = np.asarray(wave, float), np.asarray(flux, float)
    d = np.abs(wave - centro)
    base = np.nanmedian(flux[(d >= continuo_A[0]) & (d <= continuo_A[1])])
    lin = d <= media_A
    return float(np.nansum((flux[lin] - base) * np.gradient(wave)[lin]))


def rojo_menos_azul(wave, espectro, centro=CENTRO_A, media_A=MEDIA_VENTANA_A, continuo_A=CONTINUO_A):
    """La asimetría de `centroide_companero_por_ao.Grupos.rojo_menos_azul`, en una rejilla cualquiera."""
    wave, espectro = np.asarray(wave, float), np.asarray(espectro, float)
    d = np.abs(wave - centro)
    base = np.nanmedian(espectro[(d >= continuo_A[0]) & (d <= continuo_A[1])])
    dl = np.gradient(wave)
    rojo = (wave > centro) & (wave <= centro + media_A)
    azul = (wave >= centro - media_A) & (wave < centro)
    return float(np.nansum((espectro[rojo] - base) * dl[rojo]) - np.nansum((espectro[azul] - base) * dl[azul]))


def rojo_menos_azul_de_un_desplazamiento(wave, cielo, delta_A):
    """Rojo − azul del residuo que deja una rebanada con su λ corrida `delta_A`: cielo(λ − δ) − cielo(λ)."""
    wave, cielo = np.asarray(wave, float), np.asarray(cielo, float)
    return rojo_menos_azul(wave, np.interp(wave - delta_A, wave, cielo) - cielo)


def delta_necesario(wave, cielo, objetivo, escala_baja, escala_alta, deltas=np.arange(0.01, 5.0001, 0.01)):
    """El menor δ con escala_baja·RA(+δ) − escala_alta·RA(−δ) ≥ objetivo; None si ninguno llega.

    Devuelve también el mayor vuelco alcanzable en la rejilla y su δ.
    """
    vuelco = np.array([escala_baja * rojo_menos_azul_de_un_desplazamiento(wave, cielo, d)
                       - escala_alta * rojo_menos_azul_de_un_desplazamiento(wave, cielo, -d) for d in deltas])
    if objetivo < 0:
        vuelco = -vuelco
        objetivo = -objetivo
    llega = np.flatnonzero(vuelco >= objetivo)
    k = int(np.argmax(vuelco))
    return (float(deltas[llega[0]]) if llega.size else None), float(vuelco[k]), float(deltas[k])


#: Triplete de Ca II de la estrella, que cae dentro de la banda del desfase (para la medida secundaria en DATA).
CA_II_A = (8480.0, 8680.0)


def _desfase(wave, espectro, ws, fs, banda_A, desfases_A, excluir_A=()):
    """Desfase de máxima |correlación| entre `espectro` y el cielo desplazado, ambos sin su mediana móvil."""
    from scipy.ndimage import median_filter

    banda = (wave > banda_A[0]) & (wave < banda_A[1])
    fuera = np.zeros(int(banda.sum()), bool)
    for lo, hi in excluir_A:
        fuera |= (wave[banda] > lo) & (wave[banda] < hi)
    resto = (espectro - median_filter(espectro, 25))[banda]
    cc = []
    for L in desfases_A:
        t = np.interp(wave[banda] - L, ws, fs)
        t = t - median_filter(t, 25)
        m = np.isfinite(resto) & np.isfinite(t) & ~fuera
        cc.append(np.corrcoef(resto[m], t[m])[0, 1])
    k = int(np.nanargmax(np.abs(cc)))
    return float(desfases_A[k]), float(cc[k])


def desfase_del_cielo_en_el_cubo(cubo_fits, sky_fits, banda_A=(7700.0, 9300.0), n_espaxeles=3000,
                                 desfases_A=np.arange(-2.0, 2.0001, 0.05), semilla=0):
    """Desfase (Å) de las líneas de cielo del cubo respecto al `SKY_SPECTRUM` (λ_cubo = λ_topo + desfase).

    Se mide sobre la extensión **STAT** de los espaxeles del 30 % más débil (lejos de la estrella): la
    varianza la domina el ruido de Poisson del cielo, así que sus líneas trazan dónde cayó el cielo en la
    rejilla del cubo. STAT no se usa aquí como σ de nada, solo como trazador de posición. Plantilla: el
    cielo topocéntrico desplazado; a los dos se les resta su mediana móvil.

    Hasta el 2026-10-01 se medía sobre los residuos de DATA (lo que queda tras restar el cielo): son
    débiles y se correlan mal (cc 0.08-0.30), y daban −0.62 Å de mediana contra −0.84 predicho. Se
    conservan como medida secundaria, también sin el triplete de Ca II de la estrella, para ver que lo
    corto de esa cifra no era el halo. docs/2026-09-28_cielo_en_ha_y_signo_del_marco.md §5.
    """
    from astropy.io import fits

    from musepipe.qc.cube_qc import read_sky_spectrum_fits

    with fits.open(cubo_fits, memmap=True) as h:
        hd = h["DATA"].header
        nz = h["DATA"].shape[0]
        wave = hd["CRVAL3"] + hd["CD3_3"] * (np.arange(nz) + 1 - hd["CRPIX3"])
        vbary = float(h[0].header["BARYCORR"])
        datos, stat = h["DATA"].data, h["STAT"].data
        img = np.nanmedian(datos[nz // 2 - 50:nz // 2 + 50], axis=0)
        ok = np.isfinite(img)
        yy, xx = np.nonzero(ok & (img <= np.nanpercentile(img[ok], 30)))
        sel = np.random.default_rng(semilla).choice(yy.size, size=min(n_espaxeles, yy.size), replace=False)
        espectro = np.nanmedian(np.stack([datos[:, yy[k], xx[k]] for k in sel], axis=1), axis=1)
        varianza = np.nanmedian(np.stack([stat[:, yy[k], xx[k]] for k in sel], axis=1), axis=1)
    ws, fs = read_sky_spectrum_fits(Path(sky_fits))
    lag, cc = _desfase(wave, varianza, ws, fs, banda_A, desfases_A)
    lag_d, cc_d = _desfase(wave, espectro, ws, fs, banda_A, desfases_A)
    lag_dc, cc_dc = _desfase(wave, espectro, ws, fs, banda_A, desfases_A, excluir_A=(CA_II_A,))
    return {"desfase_A": lag, "correlacion": cc, "barycorr_kms": vbary,
            "desfase_data_A": lag_d, "correlacion_data": cc_d,
            "desfase_data_sin_ca_ii_A": lag_dc, "correlacion_data_sin_ca_ii": cc_dc}


def cota_caja3(integral_por_espaxel, apcorr_ha):
    """Lo más que la línea de cielo entera puede mover el rojo − azul de la caja 3×3 de C7."""
    return 9.0 * float(integral_por_espaxel) * float(apcorr_ha)


def main(argv=None):
    from scipy.stats import spearmanr

    from musepipe.config import load_run_config
    from musepipe.io import read_json
    from musepipe.qc.cube_qc import read_sky_spectrum_fits

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("run_id")
    ap.add_argument("c7_npz")
    ap.add_argument("cond_json")
    ap.add_argument("--umbral-strehl", type=float, default=12.0,
                    help="Strehl del RTC por debajo del cual la exposición es de AO baja (defecto 12)")
    ap.add_argument("--marco", type=int, default=0,
                    help="mide el desfase del cielo en el cubo en las N primeras exposiciones (~15 s cada una)")
    ap.add_argument("--s0-qc", default=None,
                    help="stageS0_qc.json cuyo p95 de |desfase| por espaxel (metrics.p95_abs_offset_A) se usa "
                         "como δ de una rebanada, además del residuo de A4/M1 (que es del cielo promediado)")
    ap.add_argument("--json", default=None)
    a = ap.parse_args(argv)

    g = Grupos(a.c7_npz, json.load(open(a.cond_json)))
    d = np.load(a.c7_npz, allow_pickle=False)
    run = load_run_config(a.run_id)
    qc00 = read_json(run.paths.stage_dir / "stage00q_qc.json")
    cielos = {Path(p).parent.name: p for p in qc00["m1_wavelength"]["sky_spectrum_files"]}
    faltan = [g.ids[i] for i in g.N1 if g.ids[i] not in cielos]
    if faltan:
        raise SystemExit(f"{len(faltan)} exposiciones sin SKY_SPECTRUM en el QC de A4: {faltan[:3]}")
    i_ha = int(np.argmin(np.abs(d["wave"] - CENTRO_A)))

    filas = []
    for i in g.N1:
        w, f = read_sky_spectrum_fits(Path(cielos[g.ids[i]]))
        lam = pico_de_cielo(w, f)
        integral = integral_de_linea(w, f, lam)
        apc = float(d["apcorr"][i, i_ha])
        rc = [g.rojo_menos_azul(g.controles[i][j]) for j in range(g.controles.shape[1])]
        filas.append({"id": g.ids[i], "strehl": float(g.strehl[i]), "pico_cielo_A": lam,
                      "cielo_por_espaxel": integral, "apcorr_ha": apc,
                      "cota_caja3": cota_caja3(integral, apc),
                      "rojo_menos_azul": g.rojo_menos_azul(g.flux[i]),
                      "controles_rojo_menos_azul_std": float(np.std(rc, ddof=1))})

    from astropy.io import fits
    cubos = {Path(p).parent.name: p for p in run.config.get("perexp_cubes", [])}
    vbary = float(np.median([fits.getheader(cubos[g.ids[i]], 0)["BARYCORR"] for i in g.N1]))
    rejilla = np.arange(6480.0, 6650.0, 0.05)
    mediano = np.median([np.interp(rejilla, *read_sky_spectrum_fits(Path(cielos[g.ids[i]]))) for i in g.N1],
                        axis=0)
    C_KMS = 299792.458

    st = np.array([r["strehl"] for r in filas])
    R = np.array([r["rojo_menos_azul"] for r in filas])
    cielo = np.array([r["cielo_por_espaxel"] for r in filas])
    cota = np.array([r["cota_caja3"] for r in filas])
    bajo = st < a.umbral_strehl
    dR = float(R[bajo].mean() - R[~bajo].mean())
    cota_max = float(np.max(np.abs(cota)))
    out = {"run": a.run_id, "umbral_strehl": a.umbral_strehl, "n_baja": int(bajo.sum()),
           "n_alta": int((~bajo).sum()),
           "pico_cielo_mediano_A": float(np.median([r["pico_cielo_A"] for r in filas])),
           "cielo_por_espaxel_mediano": float(np.median(cielo)),
           "cota_caja3_mediana": float(np.median(cota)), "cota_caja3_maxima": cota_max,
           "rojo_menos_azul_ao_baja": float(R[bajo].mean()), "rojo_menos_azul_ao_alta": float(R[~bajo].mean()),
           "delta_rojo_menos_azul": dR,
           "fraccion_de_delta_con_la_linea_entera": cota_max / abs(dR),
           "veces_la_linea_para_invertir_el_signo": abs(dR) / (2.0 * cota_max),
           "controles_std_por_exposicion_mediana": float(np.median([r["controles_rojo_menos_azul_std"]
                                                                    for r in filas])),
           "rho_cielo_strehl": [float(x) for x in spearmanr(cielo, st)],
           "rho_rojo_menos_azul_cielo": [float(x) for x in spearmanr(R, cielo)],
           "cielo_ao_baja": float(cielo[bajo].mean()), "cielo_ao_alta": float(cielo[~bajo].mean()),
           "residuo_lambda_a4_m1_A": qc00["m1_wavelength"].get("residual_scatter_A"),
           "residuo_lambda_s0_p95_A": (float(read_json(Path(a.s0_qc))["metrics"]["p95_abs_offset_A"])
                                       if a.s0_qc else None),
           "barycorr_mediano_kms": vbary, "desplazamiento_de_rebanada": {}}
    apc = np.array([r["apcorr_ha"] for r in filas])

    def vuelco(w_cubo, delta):
        """+δ en todas las de AO baja y −δ en todas las de AO alta, el caso más favorable."""
        return (9.0 * apc[bajo].mean() * rojo_menos_azul_de_un_desplazamiento(w_cubo, mediano, delta)
                - 9.0 * apc[~bajo].mean() * rojo_menos_azul_de_un_desplazamiento(w_cubo, mediano, -delta))

    # el marco verificado es el primero; el otro se da para ver que la conclusión no depende de él
    for signo, nombre in ((1.0, "BARYCORR (1 + v/c), medido"), (-1.0, "signo contrario")):
        w_cubo = rejilla * (1.0 + signo * vbary / C_KMS)
        dnec, vmax, dmax = delta_necesario(w_cubo, mediano, dR, 9.0 * apc[bajo].mean(), 9.0 * apc[~bajo].mean())
        out["desplazamiento_de_rebanada"][nombre] = {
            "pico_de_cielo_en_el_cubo_A": pico_de_cielo(w_cubo, mediano, (6559.0, 6568.0)),
            "delta_necesario_A": dnec, "vuelco_maximo": vmax, "delta_del_vuelco_maximo_A": dmax,
            "vuelco_con_delta_igual_al_residuo_a4": vuelco(w_cubo, out["residuo_lambda_a4_m1_A"] or 0.0)}
        if out["residuo_lambda_s0_p95_A"] is not None:
            out["desplazamiento_de_rebanada"][nombre]["vuelco_con_delta_igual_al_p95_de_s0"] = vuelco(
                w_cubo, out["residuo_lambda_s0_p95_A"])
    if a.marco:
        out["marco"] = []
        for i in g.N1[:a.marco]:
            m = desfase_del_cielo_en_el_cubo(cubos[g.ids[i]], cielos[g.ids[i]])
            out["marco"].append({"id": g.ids[i], **m, "predicho_1_mas_v_c_A": 8500.0 * m["barycorr_kms"] / C_KMS})
    out["por_exposicion"] = filas

    print(f"línea de cielo: pico mediano {out['pico_cielo_mediano_A']:.2f} Å (topocéntrico), "
          f"{out['cielo_por_espaxel_mediano']:.2f} por espaxel; en la caja 3×3 con apcorr "
          f"{out['cota_caja3_mediana']:.0f} (máx. {cota_max:.0f})")
    print(f"rojo − azul del compañero: AO baja {out['rojo_menos_azul_ao_baja']:+.0f} (n={out['n_baja']}), "
          f"AO alta {out['rojo_menos_azul_ao_alta']:+.0f} (n={out['n_alta']}); Δ = {dR:+.0f}")
    print(f"la línea de cielo ENTERA en una sola ventana da como mucho el "
          f"{100 * out['fraccion_de_delta_con_la_linea_entera']:.0f} % de Δ; invertir el signo pediría "
          f"{out['veces_la_linea_para_invertir_el_signo']:.1f} veces la línea")
    print(f"cielo contra Strehl ρ = {out['rho_cielo_strehl'][0]:+.2f} (p = {out['rho_cielo_strehl'][1]:.2f}); "
          f"rojo − azul contra cielo ρ = {out['rho_rojo_menos_azul_cielo'][0]:+.2f} "
          f"(p = {out['rho_rojo_menos_azul_cielo'][1]:.2f})")
    for nombre, v in out["desplazamiento_de_rebanada"].items():
        nec = (f"haría falta δ = ±{v['delta_necesario_A']:.2f} Å" if v["delta_necesario_A"] is not None
               else f"ningún δ ≤ 5 Å llega (máx. {v['vuelco_maximo']:+.0f} con δ = {v['delta_del_vuelco_maximo_A']:.2f} Å)")
        print(f"rebanada con λ corrida, marco {nombre}: cielo en {v['pico_de_cielo_en_el_cubo_A']:.2f} Å; {nec}; "
              f"con δ = ±{out['residuo_lambda_a4_m1_A']:.3f} Å (A4/M1) el vuelco es "
              f"{v['vuelco_con_delta_igual_al_residuo_a4']:+.0f}"
              + (f"; con δ = ±{out['residuo_lambda_s0_p95_A']:.3f} Å (p95 de S0) "
                 f"{v['vuelco_con_delta_igual_al_p95_de_s0']:+.0f} ({100 * v['vuelco_con_delta_igual_al_p95_de_s0'] / dR:.1f} % de Δ)"
                 if out["residuo_lambda_s0_p95_A"] is not None else ""))
    for m in out.get("marco", []):
        print(f"marco {m['id'][-12:]}: cielo en STAT a {m['desfase_A']:+.2f} Å del SKY_SPECTRUM "
              f"(cc {m['correlacion']:+.2f}; residuos de DATA {m['desfase_data_A']:+.2f}, cc "
              f"{m['correlacion_data']:+.2f}); λ·(1 + BARYCORR/c) predice {m['predicho_1_mas_v_c_A']:+.2f}")
    if out.get("marco"):
        lags = np.array([m["desfase_A"] for m in out["marco"]])
        print(f"marco, {lags.size} exposiciones: {int((lags < 0).sum())} al azul, mediana {np.median(lags):+.3f} Å, "
              f"dispersión {np.std(lags, ddof=1) if lags.size > 1 else 0.0:.3f}")
    if a.json:
        with open(a.json, "w") as fh:
            json.dump(out, fh, indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
