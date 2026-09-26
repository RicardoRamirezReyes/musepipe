#!/usr/bin/env python
"""Particiones, bloques temporales y fuga de ROXs 12 b por exposición (vía C7), como cita el paper.

Solo lectura: no escribe en runs/. Reconstruye docs/2026-09-02_mitades_noche_buena.md y la §5 de
docs/2026-09-02_es_residuo_de_la_primaria.md, que no dejaron script. Procedencia y resultados en
docs/2026-09-25_cifras_por_noche_a_55px.md.

Dos pasos, porque el primero es caro (~20 min para 29 exposiciones):

  extraer RUN SALIDA.npz   el compañero y 33 controles en cada exposición, con la máquina de C7
                           (box3, su modelo de PSF, 38 controles pedidos como manda el 09-02 §2)
  analizar ENTRADA.npz --razon-primaria R [--json SALIDA]

El análisis combina por inversa de varianza (pesos en la banda declarada, sin encoger), saca la
σ de los controles combinados igual que el objeto, y aplica la estadística de E1. Los bloques se
miden a λ fija en los DOS canales del pico (6562.03 y 6563.28 Å) y con el flujo integrado en
±6 Å, que no depende del canal: el 2026-09-25 la «variabilidad» a λ fija resultó ser un
desplazamiento del centroide en un bloque (ver el doc).
"""
import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

CANALES_A = (6562.03, 6563.28)
SEMILLA = 20260901


# --------------------------------------------------------------------------- extraer
def extraer(run_id, salida):
    from musepipe.apertures import same_radius_control_positions
    from musepipe.io import read_json
    from musepipe.stages import stage_x06_perexp as c7

    cfg = c7.stage_x06_config_from_run(run_id, overrides={
        "x06_n_controls": 38, "x06_apertures": [{"name": "box3", "kind": "box", "size": 3}]})
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
    controles = same_radius_control_positions(comp_yx, star_yx, npix, npix,
                                              n_positions=int(cfg["x06_n_controls"]))
    growth, _ = c7.resolve_flux_convention(cfg, paths["paths"].stage_dir, knob="x06_flux_convention")
    print(f"{run_id}: {len(obs.exposures)} exposiciones, {len(controles)} controles", flush=True)
    with ThreadPoolExecutor(max_workers=int(cfg["x06_max_workers"])) as pool:
        filas = list(pool.map(lambda e: c7.extract_one_exposure(
            e, plan, cfg, modelos[e.exposure_id], star_yx, comp_yx, controles, growth), obs.exposures))
    ap = [f["apertures"]["box3"] for f in filas]
    np.savez_compressed(
        salida, run_id=run_id, wave=np.asarray(filas[0]["wave_A"], float),
        exposure_id=np.asarray([f["exposure_id"] for f in filas]),
        mjd=np.asarray([f["mjd_obs"] for f in filas], float),
        exptime=np.asarray([f["exptime"] for f in filas], float),
        weight_plan=np.asarray([f["weight"] for f in filas], float),
        flux=np.asarray([a["flux"] for a in ap], float),
        flux_raw=np.asarray([a["flux_raw"] for a in ap], float),
        apcorr=np.asarray([a["apcorr"] for a in ap], float),
        controls=np.asarray([a["controls"] for a in ap], float),
        weight_band=np.asarray(cfg["x06_weight_band_A"], float),
        comp_yx=np.asarray(comp_yx, float), star_yx=np.asarray(star_yx, float))


# --------------------------------------------------------------------------- analizar
class Analisis:
    def __init__(self, npz):
        from musepipe.extraction.aperture import channel_flags
        from musepipe.io import read_json
        from musepipe.stages import stage_h01_detect as e1

        self.e1 = e1
        self.d = np.load(npz, allow_pickle=False)
        d = self.d
        self.run_id = str(d["run_id"])
        self.wave = d["wave"]
        orden = np.argsort(d["mjd"])
        noche = np.asarray([str(x).split("_")[0] for x in d["exposure_id"]])
        noches = sorted(set(noche.tolist()))
        if len(noches) != 2:
            raise SystemExit(f"se esperaban dos noches y hay {len(noches)}: {noches}")
        # N1 es la primera noche en el tiempo; en ROXs 12 b es también la buena.
        self.N1 = [int(i) for i in orden if noche[i] == noches[0]]
        self.N2 = [int(i) for i in orden if noche[i] == noches[1]]
        self.banda = (self.wave >= d["weight_band"][0]) & (self.wave <= d["weight_band"][1])
        cfg = e1.stage_h01_config_from_run(self.run_id)
        qc00 = read_json(e1.stage_h01_paths(self.run_id)["stage00q_qc_json"])
        lsf, self.lsf_fuente = e1._lsf_fwhm_from_qc_or_config(qc00, cfg)
        self.lsf = float(lsf)
        self.rv, _ = e1._rv_from_config(cfg)
        self.rv_err = e1._rv_err_from_config(cfg)
        self.esperada = e1.expected_line_center_A(e1.HALPHA_REST_A, self.rv)
        self.flags = channel_flags(self.wave)
        dist = np.abs(self.wave - 6562.8)
        self.lin, self.cont = dist <= 6.0, (dist >= 10.0) & (dist <= 40.0)
        self.dl = np.gradient(self.wave)

    def combinar(self, idx):
        from musepipe.stats import robust_sigma_axis0
        ctrls = self.d["controls"][idx]
        s = np.asarray([np.nanstd(np.nanmedian(c[:, self.banda], axis=1)) for c in ctrls])
        w = 1.0 / s ** 2
        w /= w.sum()
        flux = np.nansum(w[:, None] * self.d["flux"][idx], axis=0)
        ctrl = np.nansum(w[:, None, None] * ctrls, axis=0)
        return flux, ctrl, robust_sigma_axis0(ctrl), w

    def producto(self, flux, sigma):
        from musepipe.extraction.product import FORMAT_VERSION, SpectrumProduct
        from musepipe.spectral import continuum_running_median
        good = np.isfinite(self.wave) & np.isfinite(flux)
        cont = continuum_running_median(self.wave, flux, good, window_A=80.0, min_pixels=15)
        header = {"FORMATV": FORMAT_VERSION, "METHOD": "aperture", "RUNID": self.run_id,
                  "WFRAME": "barycentric", "SRCPOS_Y": float(self.d["comp_yx"][0]),
                  "SRCPOS_X": float(self.d["comp_yx"][1]), "APERTURE": "box3",
                  "INCUBE": "perexp", "INCUBESH": "", "NORMRAD": 25.0}
        return SpectrumProduct(wave_A=self.wave, flux=flux, flux_err=sigma, flux_err_emp=sigma,
                               apcorr=np.ones_like(self.wave), npix_eff=np.full_like(self.wave, 9.0),
                               flags=self.flags, header=header, extra_columns={"cont_runmed": cont})

    def estadistica_e1(self, idx):
        flux, ctrl, sigma, _ = self.combinar(idx)
        r = self.e1.analyze_halpha_method("c7_box3", self.producto(flux, sigma), ctrl,
                                          rv_sys_kms=self.rv, rv_sys_err_kms=self.rv_err,
                                          lsf_fwhm_A=self.lsf).row
        return {"n": len(idx), "z": r["matched_z"], "pico_A": r["peak_wave_A"],
                "factor_plantilla": r["template_factor"], "fap_param": r["global_parametric_fap"],
                "en_linea": bool(abs(r["peak_wave_A"] - self.esperada) <= self.lsf)}

    def integrar(self, spec):
        return float(np.nansum((spec[self.lin] - np.nanmedian(spec[self.cont])) * self.dl[self.lin]))

    def medidas_de_bloque(self, idx):
        """Filtro adaptado en los dos canales del pico, flujo integrado y centroide."""
        e1 = self.e1
        flux, ctrl, sigma, _ = self.combinar(idx)
        p = self.producto(flux, sigma)
        resid, good = e1._object_residual(p), e1._good_detection_mask(p)
        out = {}
        for c in CANALES_A:
            f, s, _ = e1.matched_filter_point(self.wave, resid, sigma, c, self.lsf, good)
            out[f"filtro_{c:.2f}"] = [f, s]
        vi = np.asarray([self.integrar(c) for c in ctrl])
        out["integrado"] = [self.integrar(flux), float(np.std(vi, ddof=1))]
        centros = e1.search_centers_A(self.wave, self.esperada, 200.0, good)
        mx = e1._scan_maximum(e1.matched_filter_scan(self.wave, resid, sigma, centros, self.lsf,
                                                     (1.0,), good))
        sel = np.abs(self.wave - mx["center_A"]) <= 4.0
        y = np.clip(resid[sel], 0, None)
        out["centroide_A"] = float(np.sum(self.wave[sel] * y) / np.sum(y))
        return out


def chi2_constante(medidas):
    from scipy.stats import chi2
    f = np.asarray([m[0] for m in medidas])
    s = np.asarray([m[1] for m in medidas])
    media = np.sum(f / s ** 2) / np.sum(1 / s ** 2)
    x2 = float(np.sum(((f - media) / s) ** 2))
    return x2, float(chi2.sf(x2, len(f) - 1))


def analizar(npz, razon_primaria):
    A = Analisis(npz)
    out = {"run_id": A.run_id, "lsf_fwhm_A": A.lsf, "lsf_fuente": str(A.lsf_fuente),
           "esperada_A": A.esperada, "n_controles": int(A.d["controls"].shape[1])}
    out["noche1"], out["noche2"] = A.estadistica_e1(A.N1), A.estadistica_e1(A.N2)
    out["todas"] = A.estadistica_e1(A.N1 + A.N2)

    rng = np.random.default_rng(SEMILLA)
    mitades = [A.N1[0::2], A.N1[1::2], A.N1[:11], A.N1[11:]]
    for _ in range(12):
        p = [int(i) for i in rng.permutation(A.N1)]
        mitades += [p[:11], p[11:]]
    res = [A.estadistica_e1(sorted(m, key=lambda i: A.d["mjd"][i])) for m in mitades]
    zs = np.asarray([r["z"] for r in res])
    out["mitades"] = {"n": len(res), "en_linea": int(sum(r["en_linea"] for r in res)),
                      "z_min": float(zs.min()), "z_max": float(zs.max()),
                      "z_mediana": float(np.median(zs)),
                      "z_esperada": out["noche1"]["z"] / np.sqrt(2),
                      "fap_param_max": float(max(r["fap_param"] for r in res))}

    out["bloques"] = {}
    for nb in (2, 3, 4):
        partes = [list(map(int, p)) for p in np.array_split(np.asarray(A.N1), nb)]
        ms = [A.medidas_de_bloque(p) for p in partes]
        fila = {"duracion_min": [float((A.d["mjd"][p[-1]] - A.d["mjd"][p[0]]) * 1440
                                       + A.d["exptime"][p[-1]] / 60) for p in partes],
                "centroides_A": [m["centroide_A"] for m in ms]}
        for clave in [f"filtro_{c:.2f}" for c in CANALES_A] + ["integrado"]:
            x2, p = chi2_constante([m[clave] for m in ms])
            fila[clave] = {"medidas": [m[clave] for m in ms], "chi2": x2, "dof": nb - 1, "p": p}
        out["bloques"][str(nb)] = fila

    m1, m2 = A.medidas_de_bloque(A.N1), A.medidas_de_bloque(A.N2)
    out["fuga"] = {"razon_primaria": razon_primaria}
    for clave in [f"filtro_{c:.2f}" for c in CANALES_A] + ["integrado"]:
        (f1, _), (f2, s2) = m1[clave], m2[clave]
        out["fuga"][clave] = {"noche1": m1[clave], "noche2": m2[clave],
                              "prediccion": razon_primaria * f1, "z": (razon_primaria * f1 - f2) / s2}
    _, _, _, w = A.combinar(A.N1 + A.N2)
    out["peso_invvar_noche1_sin_encoger"] = float(w[:len(A.N1)].sum())
    return out


def imprimir(o):
    print(f"{o['run_id']}: LSF {o['lsf_fwhm_A']:.3f} Å, {o['n_controles']} controles")
    for k in ("noche1", "noche2", "todas"):
        r = o[k]
        print(f"  {k:7s} z={r['z']:.2f} pico={r['pico_A']:.2f} FAP={r['fap_param']:.2g}")
    m = o["mitades"]
    print(f"  mitades: {m['en_linea']}/{m['n']} en la línea; z {m['z_min']:.2f}-{m['z_max']:.2f}, "
          f"mediana {m['z_mediana']:.2f} (esperada {m['z_esperada']:.2f})")
    for nb, b in o["bloques"].items():
        print(f"  {nb} bloques ({', '.join(f'{x:.0f}' for x in b['duracion_min'])} min); centroides "
              + " ".join(f"{c:.2f}" for c in b["centroides_A"]))
        for clave in [k for k in b if k.startswith("filtro_") or k == "integrado"]:
            x = b[clave]
            print(f"     {clave:15s} " + " · ".join(f"{f:.0f}±{s:.0f}" for f, s in x["medidas"])
                  + f"   χ²={x['chi2']:.1f}/{x['dof']} p={x['p']:.2g}")
    for clave, x in o["fuga"].items():
        if isinstance(x, dict):
            print(f"  fuga {clave:15s} pred {x['prediccion']:.0f} frente a {x['noche2'][0]:.0f}±{x['noche2'][1]:.0f}"
                  f" -> {x['z']:.1f}σ")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("extraer")
    e.add_argument("run_id")
    e.add_argument("salida")
    a = sub.add_parser("analizar")
    a.add_argument("npz")
    a.add_argument("--razon-primaria", type=float, required=True,
                   help="Hα de la primaria noche2/noche1, de scripts/cifras_por_noche_d2.py")
    a.add_argument("--json", default=None)
    args = ap.parse_args(argv)
    if args.cmd == "extraer":
        extraer(args.run_id, args.salida)
        return
    o = analizar(args.npz, args.razon_primaria)
    imprimir(o)
    if args.json:
        Path(args.json).write_text(json.dumps(o, indent=1, default=float))


if __name__ == "__main__":
    main()
