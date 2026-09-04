#!/usr/bin/env python
"""Emite las filas de datos de las tablas del paper, desde los productos.

Hermano de `scripts/paper_figures.py` y con la misma regla: no recalcula nada,
lee el QC o la tabla que la etapa ya escribio. Existe porque la Tabla 3 se
quedo con los numeros de otro run cuando se cambio el run de publicacion, y una
tabla escrita a mano no tiene forma de enterarse (`docs/2026-09-03_figuras_del_paper.md` §4).

Que objeto sale de que run se lee de `targets/<slug>.json` -> clave `paper`,
igual que las figuras.

    python scripts/paper_tables.py                  # las dos tablas
    python scripts/paper_tables.py --table lines    # solo tab:lines

Imprime SOLO las filas `\\hline`-a-`\\hline`, para pegarlas en `main.tex`: la
cabecera, el pie y las notas son prosa y se editan a mano.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.paper_figures import NOMBRE_METODO, Objeto, objetos_del_paper  # noqa: E402

#: como se escriben los metodos en la columna `Method` de la tabla.
METODO_TEX = {
    "aperture": "aperture",
    "optimal_ls": "optimal (LS)",
    "optimal_psfsub": "optimal (PSF-sub)",
    "psffit": "PSF fit\\tablefootmark{b}",
    "sgf": "SGF",
    "lpm": "LPM",
}
#: orden de METHOD_ORDER; la tabla lo respeta para poder compararla entre objetos.
ORDEN = ("aperture", "optimal_ls", "optimal_psfsub", "psffit", "sgf", "lpm")

#: unidad nativa del cubo; el flujo de linea de E1/G2 viaja en ella.
ESCALA_CUBO = 1e-20

#: lo que no cuadra entre productos, recogido al generar y volcado a stderr.
AVISOS: list[str] = []


def tex_nombre(obj: Objeto) -> str:
    """`ROXs 12 b` -> `ROXs~12~b`: en LaTeX los espacios del nombre no se parten."""
    return obj.nombre.replace(" ", "~")


def potencia(valor: float, cifras: int = 2) -> str:
    """`2.16e-17` -> `2.16\\times10^{-17}`."""
    exp = math.floor(math.log10(abs(valor)))
    mant = valor / 10.0 ** exp
    return f"{mant:.{cifras}f}\\times10^{{{exp}}}"


def con_error(valor: float, err: float, cifras: int = 2) -> str:
    """Mantisa y error en la MISMA potencia: `(2.16\\pm0.21)\\times10^{-17}`."""
    exp = math.floor(math.log10(abs(valor)))
    return (f"({valor / 10.0 ** exp:.{cifras}f}\\pm{err / 10.0 ** exp:.{cifras}f})"
            f"\\times10^{{{exp}}}")


def _e1(obj: Objeto):
    det = {r["method"]: r for r in obj.filas("halpha_detection_by_method.csv")}
    par = obj.qc("stage_h01_qc.json")["parametric_fap"]["by_method"]
    return det, par


def _limites(obj: Objeto):
    """`(limites, avisos)` de E3: `{metodo: f_lim_observed}` y en que no cuadra.

    Un objeto **detectado** no lleva limite: la tabla pone su flujo medido. Para
    el resto se cita lo que E3 publico, pero E3 se niega a correr ante
    `detection`/`candidate`, asi que su producto puede describir un estado
    anterior de E1. Esta funcion NO decide por su cuenta cual citar -- devuelve
    los avisos para que quien edita el paper los vea, porque cambiar un limite
    publicado es una decision cientifica y no un arreglo de formato.

    Lo que se compara son las entradas reales de `f_stat_99 = z_umbral * sigma`:
    el modelo de cola que E3 uso contra el estimador que E1 usa hoy, el sigma, y
    el throughput de E4.
    """
    det, _ = _e1(obj)
    h1 = obj.qc("stage_h01_qc.json")
    h3 = obj.qc("stage_h03_qc.json")
    veredicto_hoy = h1["verdict"]["verdict"]
    avisos = []

    if veredicto_hoy == "detection":
        return {}, [f"{obj.nombre}: E1 dice `detection`, asi que no hay limite "
                    "que citar; la tabla lleva el flujo medido."]

    filas = [r for r in obj.filas("halpha_upper_limits.csv") if r["row_kind"] == "method"]
    thr_hoy = obj.qc("stage_h04_qc.json")["throughput"]["per_method_at_snr5"]

    if h3["prerequisites"].get("e1_verdict") != veredicto_hoy:
        avisos.append(
            f"{obj.nombre}: E3 corrio con e1_verdict="
            f"`{h3['prerequisites'].get('e1_verdict')}` y E1 dice hoy "
            f"`{veredicto_hoy}`.")

    salida = {}
    for r in filas:
        m = r["method"]
        salida[m] = float(r["f_lim_observed"])
        if m not in det:
            continue
        # que z uso E3: f_stat_99 = z * sigma
        z_usado = float(r["f_stat_99"]) / float(r["matched_sigma"])
        z_emp, z_gum = float(r["z_99_empirical"]), float(r["z_99_gumbel"])
        if (abs(z_usado - z_emp) < abs(z_usado - z_gum)
                and h1["criterion"]["fap_estimator"] == "parametric"):
            avisos.append(
                f"{obj.nombre}/{m}: E3 uso el cuantil EMPIRICO (z={z_emp:.3f}) y "
                f"E1 decide hoy con la cola parametrica (z_99 Gumbel={z_gum:.3f}); "
                f"el limite saldria x{z_gum / z_emp:.2f}.")
        d_thr = abs(thr_hoy[m]["throughput"] / float(r["throughput"]) - 1.0) * 100.0
        if d_thr > 0.5:
            avisos.append(f"{obj.nombre}/{m}: throughput de E4 movido {d_thr:.1f} %.")
    return salida, avisos


def _terminos_error(obj: Objeto):
    """Los tres terminos del presupuesto de error de `tab:lines`, medidos.

    Tal como los define el paper (Sect. calibration): (i) ruido de pixel con la
    correccion de covarianza, (ii) dispersion inter-metodo, (iii) incertidumbre
    de calibracion de la referencia.

    - (i) es POR METODO: la sigma del filtro adaptado medida sobre los controles,
      que es empirica y por tanto ya lleva la covarianza dentro (el STAT del cubo
      no vale, A4/M5 en rojo). En fraccion del flujo es 1/z.
    - (ii) es del conjunto, no de un metodo: `std(ddof=1)/mediana` de los seis
      flujos de linea, la MISMA definicion que `_intermethod_scatter_pct` usa en
      E3, para que las dos cifras se puedan comparar. Ojo: la de E3 es la de los
      LIMITES, dominada por el continuo; esta es la de la LINEA, y son distintas.
    - (iii) es del cubo: `sys_fluxcal_declared` de D2, que M3 mide contra Gaia
      DR3 y que D2 declara aparte a proposito. Si A4/M3 no lo midio en ESTE cubo
      no hay tercer termino, y se dice: el knob `m3_flux_factor` del config puede
      venir de otra cosecha y no es una medida de este cubo.
    """
    import numpy as np

    det, _ = _e1(obj)
    flujos = np.array([float(r["matched_flux"]) for r in det.values()])
    inter = 100.0 * np.std(flujos, ddof=1) / np.median(flujos)

    m3 = obj.qc("stage00q_qc.json").get("m3_flux") or {}
    if str(m3.get("status")) == "green":
        from astropy.io import fits
        canonico = obj.qc("stage_x11_qc.json")["canonical_method"]
        with fits.open(obj.etapa(f"spec_calibrated_{canonico}_object.fits")) as h:
            d = h["SPECTRUM"].data
            dec = np.asarray(d["sys_fluxcal_declared"], float)
            flu = np.asarray(d["flux"], float)
        ok = np.isfinite(dec) & np.isfinite(flu) & (flu != 0)
        calib = f"{100.0 * np.nanmedian(np.abs(dec[ok] / flu[ok])):.1f}"
    else:
        calib = None  # no medido en este cubo

    por_metodo = {m: 100.0 * float(r["matched_sigma"]) / float(r["matched_flux"])
                  for m, r in det.items()}
    return por_metodo, inter, calib


def tabla_lines(objetos) -> str:
    """Filas de `tab:lines`: z, las dos FAP, y flujo medido o limite."""
    out = []
    for obj in objetos:
        det, par = _e1(obj)
        lims, avisos = _limites(obj)
        AVISOS.extend(avisos)
        pixel, inter, calib = _terminos_error(obj)
        if calib is None:
            AVISOS.append(f"{obj.nombre}: A4/M3 no midio la escala absoluta en este "
                          "cubo, asi que el tercer termino del presupuesto de error "
                          "no existe; la tabla pone `n.m.`.")
        for m in ORDEN:
            if m not in det:
                continue
            r = det[m]
            z = float(r["matched_z"])
            fap_e = float(r["global_empirical_fap"])
            fap_p = par[m]["fap"]
            if m in lims:
                flujo = f"\\(<{potencia(lims[m])}\\)"
            else:
                flujo = "\\(" + con_error(float(r["matched_flux"]) * ESCALA_CUBO,
                                          float(r["matched_sigma"]) * ESCALA_CUBO) + "\\)"
            # Los limites no llevan descomposicion: el ruido ya esta dentro del
            # propio limite. Solo la lleva el objeto con flujo medido.
            err = ("\\ldots" if lims else
                   f"\\({pixel[m]:.1f}/{inter:.1f}/" +
                   (f"{calib}\\)" if calib else "\\mathrm{n.m.}\\)"))
            out.append(f"{tex_nombre(obj):<11}& \\ha & {METODO_TEX[m]:<25}"
                       f"& \\({z:.2f}\\) & \\({fap_e:.3f}\\) & \\({potencia(fap_p, 1)}\\) "
                       f"& {flujo} & {err} \\\\")
    return "\n".join(out)


def tabla_accretion(objetos) -> str:
    """Filas de `tab:accretion`: la medida de un objeto contra el limite del otro."""
    col = {}
    for obj in objetos:
        det, _ = _e1(obj)
        lims, avisos = _limites(obj)
        AVISOS.extend(avisos)
        canonico = obj.qc("stage_x11_qc.json")["canonical_method"]
        g3 = obj.qc("stage_g3_qc.json")
        g2 = {r["name"]: r for r in obj.filas("g2_line_measurements.csv")}
        ha = g2.get("Halpha", {})
        c: dict[str, str] = {}
        detectado = canonico not in lims

        if detectado:
            f = float(ha["flux_direct"]) * ESCALA_CUBO
            fe = float(ha["flux_direct_err"]) * ESCALA_CUBO
            c["kind"] = "\\emph{detection}"
            c["F"] = f"\\({con_error(f, fe)}\\)\\tablefootmark{{c}}"
            c["EW"] = f"\\({float(ha['ew_A']):.1f}\\pm{float(ha['ew_err_A']):.1f}\\)"
            # La anchura INTRINSECA: `_fit_profile` ajusta una gaussiana ya
            # convolucionada con la LSF, asi que sale deconvuelta y con error.
            # `fwhm_obs_A` es el segundo momento sobre la ventana, inflado por
            # el ruido de las alas, y no es una anchura de linea.
            kms = 299792.458 / 6562.8
            fw = float(ha["fwhm_intrinsic_A"]) * kms
            fwe = float(ha["fwhm_intrinsic_err_A"]) * kms
            c["FWHM"] = f"\\({fw:.0f}\\pm{fwe:.0f}\\)"
            c["vpeak"] = (f"\\({float(ha['rv_kms']):.0f}\\pm"
                          f"{float(ha['rv_err_kms']):.0f}\\)")
            c["Lacc"] = f"\\({potencia(g3['combined_accretion']['l_acc_lsun'], 1)}\\)"
            c["Mdot"] = f"\\({potencia(g3['mdot_p50_msun_yr'], 1)}\\)"
            c["Mdot_pl"] = "\\ldots"
        else:
            lim = next(r for r in obj.filas("halpha_upper_limits.csv")
                       if r["row_kind"] == "method" and r["method"] == canonico)
            c["kind"] = "\\emph{upper limit}"
            c["F"] = f"\\(<{potencia(float(lim['f_lim_observed']))}\\)"
            c["EW"] = c["FWHM"] = c["vpeak"] = "\\ldots"
            c["Lacc"] = f"\\(<{potencia(float(lim['l_acc_lsun']), 1)}\\)\\tablefootmark{{b}}"
            c["Mdot"] = f"\\(<{potencia(float(lim['mdot_msun_yr']), 1)}\\)"
            c["Mdot_pl"] = f"\\(<{potencia(float(lim['mdot_aoyama21_msun_yr']), 1)}\\)"
        # El error del throughput es el de E3, no el de E4: E4 publica solo el
        # estadistico de las inyecciones (0.017) y E3 le suma el sistematico de
        # la perturbacion de PSF (0.087), que es el que propaga al limite.
        fila_thr = next(r for r in obj.filas("halpha_upper_limits.csv")
                        if r["row_kind"] == "method" and r["method"] == canonico)
        c["thr"] = (f"\\({float(fila_thr['throughput']):.2f}\\pm"
                    f"{float(fila_thr['throughput_err']):.2f}\\)")
        col[obj.slug] = c

    filas = [
        ("", "kind"),
        ("\\(F_{\\mathrm{H}\\alpha}\\) obs.\\ (erg\\,s\\(^{-1}\\)\\,cm\\(^{-2}\\))", "F"),
        ("EW (\\AA)", "EW"),
        ("FWHM (km\\,s\\(^{-1}\\))", "FWHM"),
        ("\\(v_{\\rm peak}\\) (km\\,s\\(^{-1}\\))", "vpeak"),
        ("\\(\\lacc\\) stellar\\tablefootmark{a} (\\(L_\\odot\\))", "Lacc"),
        ("\\(\\macc\\) stellar\\tablefootmark{a} (\\msunyr)", "Mdot"),
        ("\\(\\macc\\) planetary\\tablefootmark{b} (\\msunyr)", "Mdot_pl"),
        ("Injection throughput (canonical)", "thr"),
    ]
    slugs = [o.slug for o in objetos]
    return "\n".join(f"{etq:<52}& {col[slugs[0]][k]:<30}& {col[slugs[1]][k]} \\\\"
                     for etq, k in filas)


TABLAS = {"lines": tabla_lines, "accretion": tabla_accretion}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--table", action="append", choices=sorted(TABLAS),
                    help="emite solo esta tabla (repetible); por defecto las dos")
    args = ap.parse_args(argv)
    objetos = objetos_del_paper()
    for nombre in (args.table or list(TABLAS)):
        cuerpo = TABLAS[nombre](objetos)
        print(f"% ---- tab:{nombre} "
              f"({', '.join(o.run_id for o in objetos)}) ----")
        print(cuerpo)
        print()
    for aviso in dict.fromkeys(AVISOS):
        print("AVISO: " + aviso, file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
