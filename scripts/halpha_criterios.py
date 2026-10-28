#!/usr/bin/env python
"""Los criterios de Hα de ROXs 12 b y ROXs 42B b que cita el paper, desde los productos de las etapas.

Pasa al repo las comprobaciones que hizo el agente `halpha` el 2026-10-26 (indicaciones 3 y 4 de Claudio;
`~/muse_sesion_20261026/claudio/halpha/INFORME.md`), y que el paper cita en la Sect. 4.1 (la EW y su continuo, la
anchura a 10 % del pico, la fracción de Class III), la Tabla 6 (EW de los dos), la Sect. 4.2 (el límite de EW de
42B b), la Sect. 5.1 (los anchos de la primaria sin la LSF del estimador) y el apéndice B (la dispersión del nivel
de ruido cromosférico de Manara et al. 2013). Hermano de `paper_cifras_extincion.py` y `cifras_por_noche_d2.py`, con
la misma regla: lee lo que las etapas escribieron y no inventa entradas. Solo lectura; no escribe en runs/.

Los objetos salen de `targets/<slug>.json` (los que declaran `paper`, en el orden de `paper.order`: run, noches) y
el papel de cada uno, de G3 (`mdot_label`): el compañero con la línea detectada (12 b) y el de la cota (42B b). Ningún
nombre de objeto va en el código (`tests/test_no_hardcoded_target.py`); las claves de las salidas que dicen
«12b»/«42bb» siguen los nombres de la sesión y significan esos dos papeles.

Lee: los espectros calibrados de D2 (`spec_calibrated_<método>_object.fits`) y sus 33 controles
(`..._controls.npz`), la tabla de G2, el QC de E3 (límites al 99 %), el de A4/M2 (la LSF y los SKY_SPECTRUM por
exposición), las bibliotecas de plantillas de G3 (`g3_libraries_root`), las cadenas de T_eff de
`scripts/teff_bayes.py ajusta` (`--cadenas`) y las tablas de VizieR de `scripts/data/manara2013/` y
`scripts/data/west2011/` (con su README). Reutiliza `musepipe.lines` (las ventanas, el continuo lineal y la medida
de G2), `musepipe.spectral` (la mediana corrida de D2), el estimador de anchos de `scripts/cifras_por_noche_d2.py`
(el de la primaria en el paper) y el margen cromosférico de `scripts/teff_bayes.py`.

Subcomandos (cada uno imprime sus cifras; con --salida DIR escribe además DIR/<subcomando>.json):

  ew          EW de Hα por método de extracción (6) y definición del continuo (4), con el error medido en los 33
              controles procesados igual que el objeto; la EW de G2 y su error MC; el límite de EW de 42B b (límite de
              flujo al 99 % de E3 entre el continuo bajo la línea).
  lsf         la forma de la LSF: el perfil medio de [O I] 6300 (y 6363) del cielo en los SKY_SPECTRUM de A4/M2, más
              cuadrado que una gaussiana. Lo usan dos de los tres modelos de LSF de `w10`.
  w10         la W10 de 12 b sobre el espectro; la de una línea sin resolver con el mismo estimador y el mismo
              muestreo (tres modelos de LSF); inyección de líneas de anchura conocida en los 33 controles (y en 10
              longitudes de onda sin líneas): intervalo de Neyman de la W10 intrínseca y fracción de líneas sin
              resolver por encima de 200 y 270 km/s.
  plantillas  plantillas Class III jóvenes medidas con los mismos estimadores a la resolución y el muestreo de MUSE
              (EW, W10); el cero fotosférico de cada definición del continuo (BT-Settl); el continuo bajo Hα que
              predicen desde el nivel rojo del compañero (8600–9100 Å) según la A_V, y la EW o el límite de EW que
              resulta. La R de cada plantilla es la de la declaración de su biblioteca (la de las cifras del paper);
              con `--r-plantillas cabecera`, la de la rendija de su FITS original, como la lee G3 (8900 y 5000 en
              vez de 8800 y 5400 para las de Manara): solo cambia la W10 de TWA 29, 230 -> 226 km/s.
  primaria    Sect. 5.1: la W10 de la primaria en las dos noches sin la W10 instrumental del estimador; Sect. 4.1: la
              cota de la fuga de la primaria (su EW de la noche buena entre la del compañero).
  cromosfera  la dispersión de las 22 Class III de Manara et al. (2013) alrededor de su Ec. 2 (Hα con la relación de
              Alcalá et al. 2017), el bootstrap de sus coeficientes y, por muestra de cada cadena de T_eff, la fracción
              de esas estrellas que alcanza la línea de 12 b o supera el límite de 42B b; la saturación de Barrado y
              Navascués & Martín (2003); las EW de las M7–M9 activas de campo de West et al. (2011).
  todo        todos los anteriores y la tabla «cifra del paper -> valor».

Aleatoriedad. `ew`: el MC de EW = −F/C con la covarianza (F, C) de los controles usa `np.random.default_rng(20261026)`
(una sola secuencia, en el orden objeto -> método -> definición). `cromosfera`: el bootstrap de Manara y el error del
flujo por muestra usan `np.random.default_rng(2013)`, en el orden bootstrap -> cadenas de 12 b -> evo_solo de 12 b ->
cadenas de 42B b -> evo_solo de 42B b (evo_solo con su propia semilla 0, la de `teff_bayes.evo_solo`). Las inyecciones
de `w10` NO usan números aleatorios: el ruido son los 33 controles reales en Hα y en 10 longitudes de onda fijas
(`DESPLAZ`), y las fases de canal de la línea sin resolver son una rejilla fija; dan lo mismo en cada ejecución.

Frente a los scripts de la sesión (`~/muse_sesion_20261026/claudio/halpha/scripts/`), `ew` y `cromosfera` salen bit a
bit con estas semillas y estos órdenes, y `w10` y `plantillas` también si se usa su normalización. Dos cambios
deliberados, que no mueven ninguna cifra impresa del paper:
- la W10 y la FWHM por cruce son `cifras_por_noche_d2.ancho_kms` (el estimador de la primaria en el paper), que pasa
  a km/s con la λ del pico; la sesión lo reimplementó dividiendo por el centro nominal: ≤ 1.3e-4 relativo (la W10 de
  12 b, 255.99 -> 256.02 km/s; la cota al 95 % con `cielo_esc`, 213.8 -> 214.1);
- los límites de flujo de 42B b se leen del QC de E3 (4.4648e-18) en vez de la Tabla 4 redondeada (4.46e-18), y el
  error del flujo de 12 b de la tabla de G2 (10.37 %) en vez de «10.4 %»: < 0.2 % y < 5e-5 en las fracciones.
`cifras_del_paper` empareja cada cifra del texto con lo que sale.

uso: python scripts/halpha_criterios.py {ew,lsf,w10,plantillas,primaria,cromosfera,todo}
                                        [--project-root RAIZ] [--cadenas DIR] [--salida DIR]
     (desde un worktree, --project-root apunta al clon que tiene runs/; `w10` tarda ~4 min, el resto segundos)
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
import sys
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import cifras_por_noche_d2 as cn  # noqa: E402

HA_A = cn.HA_A
C_KMS = cn.C_KMS
#: W10/FWHM de una gaussiana, sqrt(ln 10 / ln 2) = 1.8226
G10 = cn.ANCHO10_GAUSS
#: los papeles de los dos compañeros, que salen de los productos (G3: `mdot_label`), no de su nombre: el de la línea
#: detectada (12 b en el paper) y el de la cota (42B b). Las claves de las salidas que dicen «12b»/«42bb» siguen los
#: nombres de la sesión y significan estos papeles.
DETECCION, LIMITE = "deteccion", "limite"
#: los seis métodos de extracción, en el orden de D1 (`stage_x10_compare.METHOD_ORDER`)
METODOS = ("aperture", "optimal_ls", "optimal_psfsub", "psffit", "sgf", "lpm")
#: los que conservan el continuo: SGF y LPM lo quitan por construcción (nota c de la Tabla 4 del paper)
CON_CONTINUO = ("aperture", "optimal_ls", "optimal_psfsub", "psffit")
DATOS = ROOT / "scripts" / "data"
CADENAS_C0 = Path.home() / "muse_noche_20261024_c4" / "cadenas"

SEMILLA_EW = 20261026
SEMILLA_CROMOSFERA = 2013


# ====================================================================================== entradas
def objetos_del_paper(root=ROOT):
    """Los slugs de los objetos del paper: los `targets/<slug>.json` que declaran `paper`, en el orden de
    `paper.order` (el orden en que se recorren, y en que consumen los generadores de números aleatorios)."""
    from musepipe.targets import targets_dir
    fichas = [(json.loads(f.read_text()), f.stem) for f in sorted(targets_dir(root).glob("*.json"))]
    return tuple(s for _, s in sorted((d["paper"]["order"], s) for d, s in fichas if "paper" in d))


class Contexto:
    """Dónde están los productos: la raíz del proyecto con `runs/` (`--project-root`), los objetos del paper
    (`targets/<slug>.json` -> `paper`: run, nombre, noches) y el papel de cada uno (detección o cota, de G3).
    Cachea lo que lee."""

    def __init__(self, root=None, cadenas=None):
        self.root = Path(root).resolve() if root else ROOT
        self.cadenas = Path(cadenas) if cadenas else CADENAS_C0
        self.slugs = objetos_del_paper()
        self._esp, self._ctr, self._cfg, self._papel = {}, {}, {}, {}

    def perfil(self, slug):
        from musepipe.targets import load_target
        return load_target(slug, project_root=ROOT)

    def run(self, slug):
        """El run del paper de un objeto (o un run_id tal cual)."""
        p = self.perfil(slug) if slug in self.slugs else None
        return p["paper"]["run"] if p else slug

    def papel(self, slug):
        """`DETECCION` o `LIMITE`: lo que dice G3 del run del paper (`mdot_label`), como `suelo_cromosferico`."""
        if slug not in self._papel:
            g3 = json.loads((self.rp(slug).stage_dir / "stage_g3_qc.json").read_text())
            self._papel[slug] = LIMITE if g3.get("mdot_label") == "upper_limit" else DETECCION
        return self._papel[slug]

    def con_papel(self, papel):
        """El único objeto del paper con ese papel."""
        s = [x for x in self.slugs if self.papel(x) == papel]
        if len(s) != 1:
            raise SystemExit(f"el paper tiene un compañero con {papel} y G3 da {len(s)}: {s}")
        return s[0]

    @property
    def detectado(self):
        return self.con_papel(DETECCION)

    @property
    def limite(self):
        return self.con_papel(LIMITE)

    def rp(self, run):
        from musepipe.paths import RunPaths
        return RunPaths.from_project_root(self.run(run), self.root)

    def cfg(self, run):
        from musepipe.config import load_run_config
        r = self.run(run)
        if r not in self._cfg:
            self._cfg[r] = load_run_config(r, project_root=self.root).config
        return self._cfg[r]

    def espectro(self, slug, metodo="psffit"):
        """El espectro calibrado de D2 del compañero: wave, flux, err (`flux_err_total`), flags, cont_runmed, bunit."""
        from astropy.io import fits
        k = (slug, metodo)
        if k not in self._esp:
            p = self.rp(slug).stage_dir / f"spec_calibrated_{metodo}_object.fits"
            with fits.open(p) as h:
                t = h["SPECTRUM"].data
                self._esp[k] = dict(wave=np.asarray(t["wave_A"], float), flux=np.asarray(t["flux"], float),
                                    err=np.asarray(t["flux_err_total"], float), flags=np.asarray(t["flags"], int),
                                    cont_runmed=np.asarray(t["cont_runmed"], float),
                                    bunit=h["SPECTRUM"].header.get("BUNIT"), path=str(p))
        return self._esp[k]

    def controles(self, slug, metodo="psffit"):
        """Los 33 controles de D2, procesados igual que el objeto."""
        k = (slug, metodo)
        if k not in self._ctr:
            z = np.load(self.rp(slug).stage_dir / f"spec_calibrated_{metodo}_controls.npz")
            self._ctr[k] = np.asarray(z["control_spectra"], float)
        return self._ctr[k]

    def escala_cgs(self, slug, metodo="psffit"):
        """(factor a erg/s/cm²/Å, fuente), con el resolutor de la cadena (sin valor por defecto)."""
        from musepipe.io import resolve_flux_unit
        return resolve_flux_unit(self.cfg(slug), bunit=self.espectro(slug, metodo)["bunit"])

    def catalogo(self, slug):
        from musepipe.stages.stage_g2_measure_lines import default_accretion_lines
        return self.cfg(slug).get("g2_line_catalog", default_accretion_lines())

    def g2_halpha(self, slug):
        """La fila de Hα de la tabla de G2 (`tables/g2_line_measurements.csv`)."""
        import csv
        with open(self.rp(slug).table_dir / "g2_line_measurements.csv", newline="") as fh:
            fila = next(r for r in csv.DictReader(fh) if abs(float(r["rest_A"]) - HA_A) < 1.0)
        return {k: (float(v) if _es_numero(v) else v) for k, v in fila.items()}

    def limites_e3(self, slug):
        """{método: fila de `limits`} del QC de E3 (límite al 99 %, observado, corregido por throughput)."""
        h03 = json.loads((self.rp(slug).stage_dir / "stage_h03_qc.json").read_text())
        return {x["method"]: x for x in h03["limits"]}

    def bibliotecas(self, slug=None):
        from musepipe.models.manifest import library_root
        return library_root(self.cfg(slug or self.detectado), project_root=self.root)


def _es_numero(v):
    try:
        float(v)
        return True
    except (TypeError, ValueError):
        return False


def linea_ha(catalogo):
    return next(x for x in catalogo if abs(float(x["wave_A"]) - HA_A) < 1.0)


# ====================================================================================== EW (Sect. 4.1, Tabla 6)
#: definiciones del continuo, todas con el mismo flujo de línea «directo» (integral de flujo − continuo):
#:   G2      lineal en las ventanas de G2 (`musepipe.lines.build_windows`: 8–48 Å a cada lado, sin canales marcados
#:           ni ±6 Å de otras líneas del catálogo); línea ±6 Å. La del paper.
#:   med     plano, mediana a 10–40 Å del centro (el estimador de `cifras_por_noche_d2`); línea ±6 Å.
#:   ancha   lineal en 6470–6530 y 6600–6660 Å, línea 6540–6590 Å (el de la EW de la primaria en el paper,
#:           `cifras_por_noche_d2.LINEA_ANCHA_A`/`LATERALES_A`). Sesgada en una M8: sus ventanas caen en bandas
#:           moleculares (ver `plantillas`, BT-Settl), por eso el paper da «tres definiciones».
#:   runmed  la columna `cont_runmed` de D2 (mediana corrida de 80 Å sin las líneas); línea ±6 Å.
DEFS = ("G2", "med", "ancha", "runmed")


def ventanas(catalogo, sp, definicion):
    """(máscara de la línea, función continuo(flux) -> array, o None para `runmed`)."""
    from musepipe.lines import _linear_continuum, build_windows
    w = sp["wave"]
    if definicion == "G2":
        lin, blue, red = build_windows(w, linea_ha(catalogo), catalog=catalogo, bad_flags=(sp["flags"] != 0))
        return lin, (lambda f: _linear_continuum(w, f, blue, red, HA_A)[1])
    if definicion == "med":
        d = np.abs(w - HA_A)
        cmask = (d >= 10.0) & (d <= 40.0)
        return d <= 6.0, (lambda f: np.full(w.shape, np.nanmedian(f[cmask & np.isfinite(f)])))
    if definicion == "ancha":
        lin = (w >= cn.LINEA_ANCHA_A[0]) & (w <= cn.LINEA_ANCHA_A[1])
        cmask = np.zeros(w.shape, bool)
        for lo, hi in cn.LATERALES_A:
            cmask |= (w >= lo) & (w <= hi)

        def cont(f):
            ok = cmask & np.isfinite(f)
            return np.polyval(np.polyfit(w[ok], f[ok], 1), w)
        return lin, cont
    if definicion == "runmed":
        return np.abs(w - HA_A) <= 6.0, None
    raise ValueError(definicion)


def mide_ew(w, f, lin, cont, dl):
    """(F, C en el centro, EW) de un espectro con su continuo `cont` (array). EW < 0 en emisión."""
    r = f - cont
    ok = lin & np.isfinite(r)
    F = float(np.sum(r[ok] * dl[ok]))
    C = float(np.interp(HA_A, w, cont))
    ew = float(np.sum(-(r[ok] / cont[ok]) * dl[ok])) if np.all(cont[ok] > 0) else float("nan")
    return F, C, ew


def ew_con_controles(sp, ctr, catalogo, definicion, rng, n_mc=200000):
    """EW del objeto y su error medido en los controles. Cada estimador es lineal en el espectro, así que la
    covarianza 2×2 de (F, C) del objeto es la de los mismos estimadores sobre los 33 controles; la EW = −F/C se
    propaga por MC con esa covarianza (p16–p84). En `runmed`, el objeto lleva la columna de D2 y los controles la
    misma receta (`continuum_running_median`, 80 Å) con «sin flags y a más de 15 Å de Hα» como máscara."""
    from musepipe.lines import _channel_widths
    from musepipe.spectral import continuum_running_median
    w = sp["wave"]
    dl = _channel_widths(w)
    lin, cont_fn = ventanas(catalogo, sp, definicion)
    good_rm = (sp["flags"] == 0) & (np.abs(w - HA_A) > 15.0)

    def cont_de(f, objeto):
        if definicion != "runmed":
            return cont_fn(f)
        if objeto:
            return sp["cont_runmed"]
        return continuum_running_median(w, f, good_rm, window_A=80.0, min_pixels=15)

    F, C, ew = mide_ew(w, sp["flux"], lin, cont_de(sp["flux"], True), dl)
    Fc, Cc = np.empty(ctr.shape[0]), np.empty(ctr.shape[0])
    for k in range(ctr.shape[0]):
        Fc[k], Cc[k], _ = mide_ew(w, ctr[k], lin, cont_de(ctr[k], False), dl)
    cov = np.cov(np.vstack([Fc, Cc]))
    sF, sC = float(np.sqrt(cov[0, 0])), float(np.sqrt(cov[1, 1]))
    rho = float(cov[0, 1] / (sF * sC)) if sF > 0 and sC > 0 else float("nan")
    d = rng.multivariate_normal([F, C], cov, size=n_mc)
    ews = -d[:, 0] / d[:, 1]
    ews = ews[d[:, 1] > 0]
    q = np.percentile(ews, [2.5, 16, 50, 84, 97.5])
    return dict(definicion=definicion, F=F, C=C, EW=ew, sigma_F=sF, sigma_C=sC, rho_FC=rho,
                F_ctrl_media=float(Fc.mean()), C_ctrl_media=float(Cc.mean()),
                EW_p=dict(zip(("p2.5", "p16", "p50", "p84", "p97.5"), map(float, q))),
                EW_err_ctrl=float(0.5 * (q[3] - q[1])), n_lin=int(lin.sum()),
                frac_C_no_positivo=float(np.mean(d[:, 1] <= 0)))


def g2_reproduce(ctx, slug, metodo):
    """La fila Hα de G2 rehecha con `musepipe.lines.measure_line` sobre el espectro de un método (G2 mide solo el
    canónico; aquí se aplica igual a los seis): flujo directo, EW y su MC con `flux_err_total`."""
    from musepipe.lines import measure_line
    sp = ctx.espectro(slug, metodo)
    cfg = ctx.cfg(slug)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = measure_line(sp["wave"], sp["flux"], sp["err"], linea_ha(ctx.catalogo(slug)),
                         lsf_fwhm_A=float(cfg["h01_lsf_fwhm_A"]),
                         vsys_kms=float(cfg.get("h03_rv_sys_kms", cfg.get("h01_rv_sys_kms", 0.0))),
                         n_mc=int(cfg.get("g2_n_mc", 500)), seed=int(cfg.get("g2_seed", 0)),
                         catalog=ctx.catalogo(slug), bad_flags=(sp["flags"] != 0))
    return dict(metodo=metodo, flux_direct=m.flux_direct, flux_direct_err=m.flux_direct_err, ew_A=m.ew_A,
                ew_err_A=m.ew_err_A, continuum_density=m.continuum_density)


def cmd_ew(ctx):
    rng = np.random.default_rng(SEMILLA_EW)
    out = dict(semilla=SEMILLA_EW, unidad={}, g2={}, g2_tabla={}, filas=[], limite_ew_42bb={})
    for slug in ctx.slugs:
        esc, fuente = ctx.escala_cgs(slug)
        out["unidad"][slug] = dict(escala_cgs=esc, fuente=fuente)
        out["g2"][slug] = {m: g2_reproduce(ctx, slug, m) for m in METODOS}
        t = ctx.g2_halpha(slug)
        out["g2_tabla"][slug] = {k: t[k] for k in ("ew_A", "ew_err_A", "flux_direct", "flux_direct_err",
                                                   "continuum_density", "z_score")}
        for metodo in METODOS:
            sp, ctr = ctx.espectro(slug, metodo), ctx.controles(slug, metodo)
            for definicion in DEFS:
                fila = ew_con_controles(sp, ctr, ctx.catalogo(slug), definicion, rng)
                out["filas"].append(dict(slug=slug, metodo=metodo, **fila))
    # el de la cota (42B b): límite de EW = límite de flujo al 99 % de E3 / continuo bajo la línea (def. de G2)
    lim = ctx.limite
    esc = out["unidad"][lim]["escala_cgs"]
    lims = ctx.limites_e3(lim)
    for m in CON_CONTINUO:
        fila = _fila(out, lim, m, "G2")
        Fl = float(lims[m]["f_lim_observed"]) / esc
        C, sC = fila["C"], fila["sigma_C"]
        out["limite_ew_42bb"][m] = dict(f_lim_observed=float(lims[m]["f_lim_observed"]), C=C, sigma_C=sC,
                                        EW_lim=Fl / C, EW_lim_C_mas_menos_1sigma=[Fl / (C + sC), Fl / (C - sC)]
                                        if C > sC else [Fl / (C + sC), float("inf")])
    out["resumen"] = resumen_ew(out, ctx.detectado)
    _imprime_ew(out, ctx)
    return out


def _fila(out, slug, metodo, definicion):
    return next(x for x in out["filas"] if x["slug"] == slug and x["metodo"] == metodo
                and x["definicion"] == definicion)


def resumen_ew(out, det):
    """Las cifras de EW del texto: la de G2 y su MC, el error de los controles, el rango de PSF fit y apertura con
    tres definiciones (sin `ancha`) y el de las dos óptimas (definición de G2), del compañero detectado `det`; y el
    límite de EW del de la cota con psffit."""
    g2 = out["g2_tabla"][det]
    ps = _fila(out, det, "psffit", "G2")
    tres = [-_fila(out, det, m, d)["EW"] for m in ("psffit", "aperture") for d in ("G2", "med", "runmed")]
    opt = [-_fila(out, det, m, "G2")["EW"] for m in ("optimal_ls", "optimal_psfsub")]
    return dict(ew_g2=g2["ew_A"], ew_err_g2_mc=g2["ew_err_A"], ew_err_controles=ps["EW_err_ctrl"],
                ew_psffit_aperture_3defs=[min(tres), max(tres)], ew_optimas_g2=[min(opt), max(opt)],
                ew_lim_42bb_psffit=out["limite_ew_42bb"]["psffit"]["EW_lim"],
                reproduce_tabla_g2=bool(abs(out["g2"][det]["psffit"]["ew_A"] - g2["ew_A"]) < 1e-9
                                        and abs(out["g2"][det]["psffit"]["ew_err_A"] - g2["ew_err_A"]) < 1e-9))


def _imprime_ew(out, ctx):
    r = out["resumen"]
    print(f"G2 (tabla) Hα del compañero detectado ({ctx.detectado}): EW {r['ew_g2']:+.2f} ± {r['ew_err_g2_mc']:.2f} "
          f"Å (MC); rehecha igual: {r['reproduce_tabla_g2']}; error con los 33 controles ±{r['ew_err_controles']:.2f} Å")
    for slug in ctx.slugs:
        print(f"\n### {slug}  (unidad {out['unidad'][slug]['escala_cgs']:g}, {out['unidad'][slug]['fuente']})")
        print("| método | continuo | F | C | σ_F ctrl | σ_C ctrl | ρ | EW (Å) | EW p16 … p84 |")
        print("|---|---|---|---|---|---|---|---|---|")
        for x in out["filas"]:
            if x["slug"] == slug:
                p = x["EW_p"]
                print(f"| {x['metodo']} | {x['definicion']} | {x['F']:.0f} | {x['C']:.1f} | {x['sigma_F']:.0f} | "
                      f"{x['sigma_C']:.1f} | {x['rho_FC']:+.2f} | {x['EW']:+.1f} | {p['p16']:+.1f} … {p['p84']:+.1f} |")
    a, b = r["ew_psffit_aperture_3defs"]
    o1, o2 = r["ew_optimas_g2"]
    print(f"\nPSF fit y apertura, tres definiciones: {a:.1f}–{b:.1f} Å; las dos óptimas (G2): {o1:.1f}–{o2:.1f} Å")
    for m, x in out["limite_ew_42bb"].items():
        lo, hi = x["EW_lim_C_mas_menos_1sigma"]
        print(f"{ctx.limite} {m}: límite de E3 {x['f_lim_observed']:.3e} / C {x['C']:.1f} ± {x['sigma_C']:.1f} -> "
              f"EW < {x['EW_lim']:.2f} Å (C ± 1σ: {lo:.1f}–{hi:.1f})")


# ====================================================================================== LSF (perfil del cielo)
#: rejilla del perfil medio (Å respecto del centroide), la de la sesión
REJILLA_PERFIL = np.arange(-5.0, 5.0001, 0.05)
OI_CIELO = (("OI6300", 6300.304), ("OI6363", 6363.776))


def ancho_cruce_A(x, y, fraccion):
    """Anchura completa a `fraccion` del pico de un perfil ya sin continuo, por cruce interpolado a cada lado del
    máximo, en las unidades de `x`. Es el algoritmo de `cifras_por_noche_d2.ancho_kms` sin el continuo ni el paso a
    km/s (para perfiles de LSF en Å relativos y para el continuo lineal de G2). NaN si no baja dentro de `x`."""
    i = int(np.nanargmax(y))
    m = y[i] * fraccion
    j = i
    while j > 0 and y[j] > m:
        j -= 1
    k = i
    while k < len(y) - 1 and y[k] > m:
        k += 1
    if y[j] > m or y[k] > m:
        return float("nan")
    izq = x[j] + (m - y[j]) * (x[j + 1] - x[j]) / (y[j + 1] - y[j])
    der = x[k - 1] + (m - y[k - 1]) * (x[k] - x[k - 1]) / (y[k] - y[k - 1])
    return float(der - izq)


def _perfil_linea_cielo(path, centro, half=6.0, cont=(8.0, 20.0)):
    from musepipe.qc.cube_qc import read_sky_spectrum_fits
    w, f = read_sky_spectrum_fits(path)
    d = w - centro
    c0 = np.nanmedian(f[(np.abs(d) >= cont[0]) & (np.abs(d) <= cont[1])])
    s = np.abs(d) <= half
    x, y = w[s], f[s] - c0
    yp = np.clip(y, 0, None)
    return x, y, float(np.sum(x * yp) / np.sum(yp))


def perfil_lsf_cielo(ctx, slug=None):
    """El perfil de [O I] 6300.304 y 6363.776 Å (singletes, anchura propia despreciable) en los SKY_SPECTRUM por
    exposición que midió A4/M2 (muestreo 0.3125 Å) del run del compañero detectado (por defecto): FWHM y W10 por
    cruce en cada uno, y el perfil mediano normalizado, alineado en su centroide, en `REJILLA_PERFIL`. Devuelve
    (resumen, {línea: (rejilla, perfil)})."""
    slug = slug or ctx.detectado
    qc = json.loads((ctx.rp(slug).stage_dir / "stage00q_qc.json").read_text())["m2_lsf"]
    files, poly = qc["sky_spectrum_files"], qc["poly2_coeffs"]
    out = dict(run=ctx.run(slug), n_ficheros=len(files), poly2_coeffs=poly,
               lsf_halpha_A=qc["lsf_fwhm_at_halpha_A"], lineas={})
    perfiles = {}
    for nombre, lam in OI_CIELO:
        fw, w10, gy, fallos = [], [], [], 0
        for p in files:
            try:
                x, y, cen = _perfil_linea_cielo(p, lam)
            except Exception:  # noqa: BLE001 - un fichero ilegible no para el resto; se cuenta
                fallos += 1
                continue
            fw.append(ancho_cruce_A(x, y, 0.5))
            w10.append(ancho_cruce_A(x, y, 0.1))
            gy.append(np.interp(REJILLA_PERFIL, x - cen, y / np.nanmax(y), left=0, right=0))
        fw, w10 = np.asarray(fw), np.asarray(w10)
        prof = np.nanmedian(np.asarray(gy), axis=0)
        perfiles[nombre] = (REJILLA_PERFIL.copy(), prof)
        out["lineas"][nombre] = dict(n=int(fw.size), fallos=fallos, fwhm_A_med=float(np.nanmedian(fw)),
                                     fwhm_A_std=float(np.nanstd(fw)), w10_A_med=float(np.nanmedian(w10)),
                                     cociente_w10_fwhm=float(np.nanmedian(w10 / fw)),
                                     fwhm_poly_A=float(np.polyval(poly, lam)),
                                     fwhm_perfil_medio_A=ancho_cruce_A(REJILLA_PERFIL, prof, 0.5),
                                     w10_perfil_medio_A=ancho_cruce_A(REJILLA_PERFIL, prof, 0.1))
    return out, perfiles


#: líneas fotosféricas (aire) razonablemente aisladas en un K7–M0 a R ~ 2900
LINEAS_PRIMARIA = {"Ca I 6439.08": 6439.075, "Ca I 6462.57": 6462.567, "Ca I 6493.78": 6493.781, "Fe I 6546.24": 6546.239,
                   "Li I 6707.8": 6707.8, "Ca I 6717.68": 6717.681, "Ca I 6102.72": 6102.723, "Ca I 6122.22": 6122.217,
                   "Fe I 6750.15": 6750.152, "Ni I 7714.3": 7714.31, "Fe I 8688.6": 8688.624, "Ti I 8434.96": 8434.957}


def lineas_primaria(ctx, slug):
    """FWHM de un ajuste gaussiano con continuo lineal (±5 Å) a líneas fotosféricas estrechas de la primaria (psffit
    de D2, mismo cubo y misma extracción que el compañero). Es una cota SUPERIOR de la LSF efectiva de una fuente
    puntual (la línea tiene su propia anchura). Li I 6708 da 2.34 Å en 12 b: apoya el modelo `cielo_esc`."""
    from astropy.io import fits
    from scipy.optimize import curve_fit
    with fits.open(ctx.rp(slug).stage_dir / "spec_calibrated_psffit_star.fits") as h:
        t = h["SPECTRUM"].data
        w, f = np.asarray(t["wave_A"], float), np.asarray(t["flux"], float)
    poly = json.loads((ctx.rp(slug).stage_dir / "stage00q_qc.json").read_text())["m2_lsf"]["poly2_coeffs"]

    def gauss_lin(x, a, mu, s, c0, c1):
        return a * np.exp(-0.5 * ((x - mu) / s) ** 2) + c0 + c1 * (x - mu)

    out = {}
    for nombre, lam in LINEAS_PRIMARIA.items():
        s = np.abs(w - lam) <= 5.0
        x, y = w[s], f[s]
        if x.size < 6 or not np.all(np.isfinite(y)):
            continue
        c = np.median(np.r_[y[:2], y[-2:]])
        try:
            po, pc = curve_fit(gauss_lin, x, y, p0=[y.min() - c, lam, 1.0, c, 0.0], maxfev=20000)
            out[nombre] = dict(lam=lam, fwhm_gauss_A=float(2.3548 * abs(po[2])),
                               fwhm_err_A=float(2.3548 * np.sqrt(pc[2, 2])), centro_A=float(po[1]),
                               profundidad_rel=float(-po[0] / po[3]), lsf_m2_A=float(np.polyval(poly, lam)))
        except Exception as e:  # noqa: BLE001 - una línea que no ajusta no para las demás
            out[nombre] = dict(lam=lam, error=str(e))
    return out


def cmd_lsf(ctx, salida=None):
    out, perfiles = perfil_lsf_cielo(ctx)
    out["primaria"] = {s: lineas_primaria(ctx, s) for s in ctx.slugs}
    for k, v in out["lineas"].items():
        print(f"{k}: n={v['n']} FWHM {v['fwhm_A_med']:.3f} ± {v['fwhm_A_std']:.3f} Å (polinomio de M2 "
              f"{v['fwhm_poly_A']:.3f}); W10 {v['w10_A_med']:.3f} Å; W10/FWHM {v['cociente_w10_fwhm']:.3f} "
              f"(gaussiana {G10:.3f})")
    print(f"LSF en Hα (A4/M2) {out['lsf_halpha_A']:.3f} Å -> W10 de esa gaussiana {G10 * out['lsf_halpha_A']:.3f} Å "
          f"= {G10 * out['lsf_halpha_A'] / HA_A * C_KMS:.0f} km/s (antes de muestrear)")
    for s, d in out["primaria"].items():
        print(f"primaria {s}, FWHM gaussiana (Å) de líneas estrechas: " + "; ".join(
            f"{k} {v['fwhm_gauss_A']:.2f} ± {v['fwhm_err_A']:.2f}" for k, v in d.items() if "error" not in v))
    if salida:
        g, p = perfiles["OI6300"]
        np.savez(Path(salida) / "lsf_perfil_6300.npz", grid=g, prof=p)
    return out, perfiles


# ====================================================================================== W10 (Sect. 4.1)
FINO = 0.01
#: FWHM intrínsecas inyectadas (km/s)
V_GRID = (0.0, 25.0, 50.0, 75.0, 100.0, 125.0, 150.0, 200.0, 270.0)
#: longitudes de onda sin líneas donde los mismos controles dan más realizaciones de ruido (10 × 33 + 33 = 363)
DESPLAZ = (6440.0, 6460.0, 6480.0, 6500.0, 6520.0, 6605.0, 6625.0, 6645.0, 6695.0, 6720.0)
#: los tres modelos de LSF: `gauss` (FWHM = h01_lsf_fwhm_A, la del paper y de G2); `cielo_esc` (el perfil del
#: cielo escalado para que un ajuste gaussiano en canales de 1.25 Å dé esa FWHM, lo que da Li I 6708 de la primaria
#: en el mismo cubo); `cielo` (el perfil del cielo sin escalar)
MODELOS_LSF = ("gauss", "cielo_esc", "cielo")


def kernel(modelo, lsf_fwhm, perfil=None):
    """Kernel de la LSF en una rejilla fina (Å relativo), de área 1. `perfil` = (rejilla, perfil) del cielo."""
    x = np.arange(-15, 15 + FINO / 2, FINO)
    if modelo == "gauss":
        k = np.exp(-0.5 * (x / (lsf_fwhm / 2.3548)) ** 2)
    else:
        g, p = perfil
        esc = escala_cielo(lsf_fwhm, perfil) if modelo == "cielo_esc" else 1.0
        k = np.interp(x, g * esc, p, left=0, right=0)
    return x, k / (k.sum() * FINO)


_ESC = {}


def escala_cielo(objetivo, perfil):
    """El factor que lleva el perfil del cielo a un ajuste gaussiano de FWHM `objetivo` en canales de 1.25 Å
    (media de 9 fases de canal)."""
    from scipy.optimize import brentq, curve_fit
    g, p = perfil
    clave = (float(objetivo), _hash(p))
    if clave in _ESC:
        return _ESC[clave]

    def fw(esc):
        x = np.arange(-15, 15 + FINO / 2, FINO)
        k = np.interp(x, g * esc, p, left=0, right=0)
        vals = []
        for ph in np.linspace(-0.6, 0.6, 9):
            edges = np.arange(-10.625, 10.626, 1.25) + ph
            ys = binea(x, k, edges)
            cen = 0.5 * (edges[1:] + edges[:-1])
            po, _ = curve_fit(lambda t, a, m, s: a * np.exp(-0.5 * ((t - m) / s) ** 2), cen, ys, p0=[ys.max(), 0, 1])
            vals.append(2.3548 * abs(po[2]))
        return float(np.mean(vals)) - objetivo
    _ESC[clave] = brentq(fw, 0.6, 1.3)
    return _ESC[clave]


def binea(x, y, edges):
    """Valor medio de y(x) (rejilla fina de paso FINO) en cada canal [edges_i, edges_i+1)."""
    c = np.concatenate([[0.0], np.cumsum(y) * FINO])
    xi = np.concatenate([[x[0] - FINO / 2], x + FINO / 2])
    a = np.interp(edges, xi, c)
    return np.diff(a) / np.diff(edges)


def bordes(wave):
    """Bordes de los canales (los de `musepipe.lines._channel_widths`)."""
    e = np.empty(wave.size + 1)
    e[1:-1] = 0.5 * (wave[1:] + wave[:-1])
    e[0] = wave[0] - 0.5 * (wave[1] - wave[0])
    e[-1] = wave[-1] + 0.5 * (wave[-1] - wave[-2])
    return e


def _hash(a):
    """Clave por CONTENIDO de un array (no `id`, que se reutiliza al liberar el array)."""
    return hashlib.sha1(np.ascontiguousarray(a).tobytes()).hexdigest()


class Perfiles:
    """CDF de la línea (gaussiana intrínseca de FWHM v ⊗ kernel) tabulada en v, para evaluar rápido el modelo
    integrado en canales: modelo(F, mu, v) = F · [CDF_v(borde_sup − mu) − CDF_v(borde_inf − mu)] / ancho.
    Conserva el flujo: la integral del modelo en canales es F."""

    V_TAB = np.arange(0.0, 600.0 + 1e-9, 2.5)

    def __init__(self, kx, kk, centro=HA_A):
        from scipy.signal import fftconvolve
        cdfs = []
        for v in self.V_TAB:
            sig = v / C_KMS * centro / 2.3548
            if sig > 0:
                gi = np.exp(-0.5 * (kx / sig) ** 2)
                gi /= gi.sum()
                prof = fftconvolve(kk, gi, mode="same")
            else:
                prof = kk.copy()
            prof = np.clip(prof, 0, None)
            c = np.cumsum(prof)
            cdfs.append(c / c[-1])
        self.cdf = np.asarray(cdfs)
        self.xi = kx + FINO / 2

    def cdf_en(self, t, v):
        v = float(np.clip(v, 0.0, self.V_TAB[-1]))
        j = min(int(v // 2.5), self.V_TAB.size - 2)
        a = (v - self.V_TAB[j]) / 2.5
        c = (1 - a) * self.cdf[j] + a * self.cdf[j + 1]
        return np.interp(t, self.xi, c, left=0.0, right=1.0)

    def linea(self, edges, F, mu, v):
        return F * np.diff(self.cdf_en(edges - mu, v)) / np.diff(edges)


_PERF = {}


def _perfiles(kx, kk):
    key = _hash(kk)
    if key not in _PERF:
        _PERF[key] = Perfiles(kx, kk)
    return _PERF[key]


def linea_modelo(wave, flujo, centro, v_int_kms, kx, kk):
    """Línea (gaussiana intrínseca de FWHM `v_int_kms` ⊗ kernel) integrada en los canales de `wave`; área = flujo."""
    return _perfiles(kx, kk).linea(bordes(wave), flujo, centro, v_int_kms)


def w10_kms(w, f, centro, fraccion=0.1):
    """El estimador del paper (el de la primaria, `cifras_por_noche_d2.ancho_kms`): continuo = mediana a 10–40 Å
    del centro, cruce interpolado a `fraccion` del pico dentro de ±10 Å."""
    return cn.ancho_kms(w, f, fraccion=fraccion, centro=centro)


def ventanas_g2(w, centro, flags):
    """Las ventanas de continuo de G2 (8–48 Å a cada lado, sin canales marcados) centradas en `centro`, sin excluir
    otras líneas del catálogo: en Hα coinciden con las de G2 en los dos runs (no hay otra línea a < 54 Å), y en las
    longitudes de onda desplazadas de la inyección la línea inyectada es la única."""
    from musepipe.lines import build_windows
    _, blue, red = build_windows(w, {"wave_A": centro}, catalog=(), bad_flags=(flags != 0))
    return blue, red


def w10_lin(w, f, centro, blue, red, fraccion=0.1):
    """Como `w10_kms`, con el continuo lineal de las ventanas de G2."""
    from musepipe.lines import _linear_continuum
    _, cont, _ = _linear_continuum(w, f, blue, red, centro)
    sel = np.abs(w - centro) <= 10.0
    return ancho_cruce_A(w[sel], (f - cont)[sel], fraccion) / centro * C_KMS


def ajuste_perfil(w, f, err, centro, blue, red, kx, kk):
    """FWHM intrínseca (km/s) del modelo gaussiano ⊗ kernel integrado en canales, con el continuo lineal de G2
    restado (amplitud, centro y σ_int libres; tres arranques, gana el de menor χ²)."""
    from scipy.optimize import curve_fit
    from musepipe.lines import _linear_continuum
    _, cont, _ = _linear_continuum(w, f, blue, red, centro)
    sel = np.abs(w - centro) <= 8.0
    ws, rs, es = w[sel], (f - cont)[sel], err[sel]
    P = _perfiles(kx, kk)
    es_b = bordes(w)[np.r_[np.where(sel)[0], np.where(sel)[0][-1] + 1]]

    def mod(x, F, dmu, v):
        return P.linea(es_b, F, centro + dmu, abs(v))

    best = None
    for v0 in (10.0, 60.0, 120.0):
        try:
            po, _ = curve_fit(mod, ws, rs, p0=[max(rs.sum() * 1.25, 1.0), -0.4, v0], sigma=es, absolute_sigma=True,
                              method="trf", max_nfev=4000, diff_step=[1e-3, 1e-3, 1e-2])
            chi = float(np.sum(((rs - mod(ws, *po)) / es) ** 2))
            if best is None or chi < best[1]:
                best = (po, chi)
        except Exception:  # noqa: BLE001 - un arranque que no converge no cuenta
            continue
    if best is None:
        return dict(F=np.nan, mu=np.nan, v=np.nan, chi2=np.nan)
    po, chi = best
    return dict(F=float(po[0]), mu=float(centro + po[1]), v=float(abs(po[2])), chi2=chi)


def mide_todo(w, f, err, centro, flags, kx=None, kk=None, con_ajuste=True):
    """Los estimadores de anchura sobre un espectro: W10 y FWHM por cruce (continuo mediano, el del paper), W10 con
    el continuo de G2 y, con `con_ajuste`, la FWHM intrínseca ajustada."""
    blue, red = ventanas_g2(w, centro, flags)
    out = dict(cruce_med=w10_kms(w, f, centro), cruce_G2=w10_lin(w, f, centro, blue, red),
               fwhm_cruce_med=w10_kms(w, f, centro, fraccion=0.5))
    if con_ajuste:
        a = ajuste_perfil(w, f, err, centro, blue, red, kx, kk)
        out.update(fwhm_int_ajuste=a["v"], w10_int_ajuste=G10 * a["v"], F_ajuste=a["F"], mu_ajuste=a["mu"])
    return out


def w10_instrumental(w, err, flags, C0, kx, kk):
    """La W10 (y FWHM) que mide el estimador del paper sobre una línea SIN RESOLVER (anchura intrínseca 0) sin ruido,
    integrada en los canales de 1.25 Å: mediana sobre 11 fases de canal entre −0.625 y +0.625 Å."""
    inst = []
    for ph in np.linspace(-0.625, 0.625, 11):
        lm = linea_modelo(w, 2300.0, HA_A + ph, 0.0, kx, kk) + C0
        inst.append(mide_todo(w, lm, err, HA_A + ph, flags, con_ajuste=False))
    return float(np.median([x["cruce_med"] for x in inst])), float(np.median([x["fwhm_cruce_med"] for x in inst]))


def neyman(vs, dist, obs, nivel):
    """Intervalo de Neyman: el conjunto de v cuyo intervalo central al `nivel` (%) de la distribución del estimador
    contiene `obs`, con las bandas interpoladas linealmente en v (2001 puntos). ([lo, hi], bandas_lo, bandas_hi)."""
    lo_q, hi_q = 50 - nivel / 2, 50 + nivel / 2
    los = np.array([_pct(d, [lo_q])[0] for d in dist])
    his = np.array([_pct(d, [hi_q])[0] for d in dist])
    vf = np.linspace(vs[0], vs[-1], 2001)
    lo_f, hi_f = np.interp(vf, vs, los), np.interp(vf, vs, his)
    ok = (lo_f <= obs) & (obs <= hi_f)
    if not ok.any():
        return [float("nan"), float("nan")], los.tolist(), his.tolist()
    return [float(vf[ok].min()), float(vf[ok].max())], los.tolist(), his.tolist()


def _pct(a, q):
    a = np.asarray(a, float)
    a = a[np.isfinite(a)]
    return np.percentile(a, q) if a.size else np.full(len(q), np.nan)


def instrumentales(ctx, perfil=None):
    """{modelo: (W10, FWHM)} instrumentales del estimador en el run del compañero detectado (rápido: sin inyección)."""
    if perfil is None:
        perfil = perfil_lsf_cielo(ctx)[1]["OI6300"]
    det = ctx.detectado
    sp = ctx.espectro(det, "psffit")
    lsf = float(ctx.cfg(det)["h01_lsf_fwhm_A"])
    C0 = float(ctx.g2_halpha(det)["continuum_density"])
    out = {}
    for modelo in MODELOS_LSF:
        kx, kk = kernel(modelo, lsf, perfil)
        out[modelo] = w10_instrumental(sp["wave"], sp["err"], sp["flags"], C0, kx, kk)
    return out


def cmd_w10(ctx, perfil=None):
    """W10 del compañero detectado (12 b), la instrumental por modelo de LSF, la inyección en los controles y su
    análisis de Neyman."""
    if perfil is None:
        perfil = perfil_lsf_cielo(ctx)[1]["OI6300"]
    det = ctx.detectado
    cfg = ctx.cfg(det)
    lsf = float(cfg["h01_lsf_fwhm_A"])
    sp = ctx.espectro(det, "psffit")
    w, f, err, flags = sp["wave"], sp["flux"], sp["err"], sp["flags"]
    ctr = ctx.controles(det, "psffit")
    C0 = float(ctx.g2_halpha(det)["continuum_density"])
    out = dict(lsf_A=lsf, C0_g2=C0, w10_inst_gauss_sin_muestreo_kms=G10 * lsf / HA_A * C_KMS, metodos={},
               modelos={}, inyeccion_aleatoria=False)
    for m in METODOS:
        s = ctx.espectro(det, m)
        r = mide_todo(s["wave"], s["flux"], s["err"], HA_A, s["flags"], con_ajuste=False)
        r["cruce_med_sin_lsf_gauss"] = cn.sin_lsf_kms(r["cruce_med"], lsf)
        out["metodos"][m] = r
    for modelo in MODELOS_LSF:
        kx, kk = kernel(modelo, lsf, perfil)
        inst_w10, inst_fw = w10_instrumental(w, err, flags, C0, kx, kk)
        obj = mide_todo(w, f, err, HA_A, flags, kx, kk)
        obj["cruce_med_sin_lsf_inst"] = float(np.sqrt(max(obj["cruce_med"] ** 2 - inst_w10 ** 2, 0.0)))
        res = dict(kernel_fwhm_fino_A=ancho_cruce_A(kx, kk, 0.5), kernel_w10_fino_A=ancho_cruce_A(kx, kk, 0.1),
                   escala_cielo=(escala_cielo(lsf, perfil) if modelo == "cielo_esc" else None),
                   w10_inst_cruce_kms=inst_w10, fwhm_inst_cruce_kms=inst_fw, objeto=obj,
                   F_inyectado=obj["F_ajuste"], mu_inyectado=obj["mu_ajuste"], por_v={}, neyman={})
        dist_w10, dist_fw = [], []
        for v in V_GRID:
            filas = []
            for k in range(ctr.shape[0]):
                y = ctr[k] + linea_modelo(w, obj["F_ajuste"], obj["mu_ajuste"], v, kx, kk) + C0
                filas.append(dict(donde="Ha", **mide_todo(w, y, err, HA_A, flags, kx, kk)))
            for lam in DESPLAZ:
                d = lam - HA_A
                for k in range(ctr.shape[0]):
                    y = ctr[k] + linea_modelo(w, obj["F_ajuste"], obj["mu_ajuste"] + d, v, kx, kk) + C0
                    filas.append(dict(donde=f"{lam:.0f}", **mide_todo(w, y, err, HA_A + d, flags, kx, kk)))
            w10 = np.array([x["cruce_med"] for x in filas], float)
            fw = np.array([x["fwhm_int_ajuste"] for x in filas], float)
            en_ha = np.array([x["donde"] == "Ha" for x in filas])
            sinl = np.sqrt(np.clip(w10 ** 2 - inst_w10 ** 2, 0, None))
            fin = np.isfinite(w10)
            res["por_v"][f"{v:g}"] = dict(
                w10_obs_p16_50_84=_pct(w10, [16, 50, 84]).tolist(), w10_obs_Ha_p16_50_84=_pct(w10[en_ha], [16, 50, 84]).tolist(),
                w10_obs_otras_p50=float(_pct(w10[~en_ha], [50])[0]), fwhm_int_p16_50_84=_pct(fw, [16, 50, 84]).tolist(),
                p_w10obs_200=float(np.mean(w10[fin] > 200)), p_w10obs_270=float(np.mean(w10[fin] > 270)),
                p_w10sinlsf_200=float(np.mean(sinl[fin] > 200)), p_w10sinlsf_270=float(np.mean(sinl[fin] > 270)),
                n=int(fin.sum()), frac_nan=float(1 - fin.mean()))
            dist_w10.append(w10)
            dist_fw.append(fw)
        vs = np.asarray(V_GRID, float)
        for nombre, dist, obs in (("w10_cruce_med", dist_w10, obj["cruce_med"]),
                                  ("fwhm_int_ajuste", dist_fw, obj["fwhm_int_ajuste"])):
            for nivel in (68, 95):
                iv, los, his = neyman(vs, dist, obs, nivel)
                res["neyman"][f"{nombre}_{nivel}"] = dict(v_int_kms=iv, w10_int_kms=[G10 * x for x in iv],
                                                          bandas_lo=los, bandas_hi=his)
        out["modelos"][modelo] = res
    out["resumen"] = resumen_w10(out)
    _imprime_w10(out)
    return out


def resumen_w10(out):
    m = out["modelos"]
    return dict(w10_obs_psffit=out["metodos"]["psffit"]["cruce_med"],
                w10_inst_por_modelo={k: v["w10_inst_cruce_kms"] for k, v in m.items()},
                w10_int_68_cielo_esc=m["cielo_esc"]["neyman"]["w10_cruce_med_68"]["w10_int_kms"],
                w10_int_95_sup_por_modelo={k: v["neyman"]["w10_cruce_med_95"]["w10_int_kms"][1] for k, v in m.items()},
                p_sin_resolver_sobre_200={k: v["por_v"]["0"]["p_w10obs_200"] for k, v in m.items()},
                p_sin_resolver_sobre_270={k: v["por_v"]["0"]["p_w10obs_270"] for k, v in m.items()})


def _imprime_w10(out):
    print("| método | W10 cruce (km/s) | W10 con continuo de G2 | FWHM cruce | W10 sin LSF gaussiana |")
    print("|---|---|---|---|---|")
    for k, r in out["metodos"].items():
        print(f"| {k} | {r['cruce_med']:.0f} | {r['cruce_G2']:.0f} | {r['fwhm_cruce_med']:.0f} | "
              f"{r['cruce_med_sin_lsf_gauss']:.0f} |")
    for modelo, r in out["modelos"].items():
        o = r["objeto"]
        print(f"\n### LSF {modelo}: kernel FWHM {r['kernel_fwhm_fino_A']:.3f} Å, W10 {r['kernel_w10_fino_A']:.3f} Å; "
              f"línea sin resolver en canales: W10 {r['w10_inst_cruce_kms']:.0f}, FWHM {r['fwhm_inst_cruce_kms']:.0f} km/s")
        print(f"   objeto: W10 {o['cruce_med']:.0f}, FWHM_int ajustada {o['fwhm_int_ajuste']:.0f} km/s; W10 sin la "
              f"instrumental (cuadratura) {o['cruce_med_sin_lsf_inst']:.0f}")
        print("| FWHM_int inyectada | W10_int | W10 obs p16/p50/p84 (Hα) | FWHM_int ajustada p16/p50/p84 | "
              "P(W10 obs > 200) | P(> 270) |")
        print("|---|---|---|---|---|---|")
        for v, x in r["por_v"].items():
            a, b = x["w10_obs_Ha_p16_50_84"], x["fwhm_int_p16_50_84"]
            print(f"| {float(v):.0f} | {G10 * float(v):.0f} | {a[0]:.0f}/{a[1]:.0f}/{a[2]:.0f} | "
                  f"{b[0]:.0f}/{b[1]:.0f}/{b[2]:.0f} | {x['p_w10obs_200']:.2f} | {x['p_w10obs_270']:.2f} |")
        for k, n in r["neyman"].items():
            print(f"   Neyman {k}: FWHM_int {n['v_int_kms'][0]:.0f}–{n['v_int_kms'][1]:.0f} km/s -> "
                  f"W10_int {n['w10_int_kms'][0]:.0f}–{n['w10_int_kms'][1]:.0f} km/s")


# ====================================================================================== plantillas y continuo
#: tipos de las plantillas que entran (por el nombre del fichero)
SPT_PLANTILLAS = ("M6", "M7", "M8", "M9", "L0", "L1")
BIBLIOTECAS = ("templates_young", "templates_young_lateM_xshooter", "templates_xshyne_L_xshooter")
#: plantillas para predecir el continuo, por el papel del compañero (sus tipos ópticos en el paper: el detectado, 12 b,
#: M7.5–M8 -> M7–M8.5; el de la cota, 42B b, M9.5 o más tardío -> M9–L1). Fuera: USco J16073798 (Hα en «absorción» de
#: 27 Å a su resolución nativa: artefacto) y 2MASS0616 (continuo negativo en Hα). Las de Manara et al. (2013/2017) son
#: Class III de A_V ≲ 0.2 (ReadMe de J/A+A/551/A107); las de USco pueden estar enrojecidas (no se corrige: baja la
#: predicción, conservador para la EW).
USO_CONTINUO = {DETECCION: ("J11195652-7504529_M7", "LM601_M7.5", "CHSM17173_M8", "USco_J16104713-2239496_M8.5",
                            "VISTA_J16072381-2211021_M8.5"),
                LIMITE: ("TWA26_M9", "TWA29_M9.5", "USco_J16081842-2232252_M9", "2MASS0046_L0", "2MASS0518_L1",
                         "2MASS1207_L1", "2MASS1826_L1", "WISE0528_L1")}
MANARA = ("J11195652-7504529_M7", "LM601_M7.5", "CHSM17173_M8", "TWA26_M9", "TWA29_M9.5")
#: la comparación de W10 (Sect. 4.1: «M7–M9.5 ... 230–310 km/s») va sin las dos USco M9: J16073798 tiene Hα en
#: absorción a su resolución nativa, y en J16081842 el «pico» es más estrecho que la resolución nativa (FWHM 24 km/s
#: frente a c/R = 59 km/s), es decir, no es una línea resuelta. La sesión no dejó escrita esta selección; es la única
#: que da el rango del paper (con J16081842 sería 213–310). Ver el informe del port.
EXCLUIDAS_W10 = {"USco_J16073798-2242471_M9": "Hα en absorción a su resolución nativa (EW +27 Å): artefacto",
                 "USco_J16081842-2232252_M9": "pico más estrecho que la resolución nativa (FWHM 24 km/s, c/R 59 km/s)"}
#: las M9 y M9.5 jóvenes de Manara et al. (2013) (Sect. 4.2: «9–14 Å»), en sus dos reducciones de TWA 26
MANARA_M9 = ("TWA26_M9", "TWA29_M9.5")
#: la LSF de MUSE a la que se degradan (la de A4/M2 de 12 b; la de 42B b es 2.297 Å)
TEFF_BTSETTL = (2100, 2200, 2300, 2400, 2500, 2600, 2700, 2800)
LOGG_BTSETTL = (3.5, 4.0, 4.5)
PICO_COMPANERO_A = 6562.40  # donde está el pico de Hα de 12 b; la plantilla se lleva ahí


def r_plantilla(decl, meta, modo="declaracion", lib=None, path=None):
    """La R con la que se degrada una plantilla. `declaracion`: `LibraryDeclaration.resolution_for(meta)` sobre la
    meta del npz (la R del espectro si la trae -> la de su objeto en Manara 2013, Tabla 2 -> la de la biblioteca): lo
    que usó la sesión y da las cifras del paper. `cabecera`: además la rendija VIS de la cabecera del FITS original,
    como hace el cargador de G3 (`EmpiricalTemplateLibrary._resolve_per_template_R`): 8900 y 5000 en vez de 8800 y
    5400 para las de Manara (sensibilidad, ver el informe)."""
    if modo == "cabecera" and lib is not None:
        tabla = lib.resolution_table()
        r = tabla.get(Path(path).name, {}).get("R")
        if r not in (None, "inf"):
            return float(r), tabla[Path(path).name]["source"]
    r, fw = decl.resolution_for(meta)
    return float(r), "declaración de la biblioteca (resolution_for)"


def degrada(w, f, R, w_obj, desplaz, lsf_A):
    """Convoluciona a la LSF de MUSE (gaussiana de FWHM sqrt(lsf² − (6563/R)²)) e integra en los canales de `w_obj`,
    con la plantilla desplazada `desplaz` Å."""
    dl = np.median(np.diff(w))
    fw_k = np.sqrt(max(lsf_A ** 2 - (6563.0 / R) ** 2, 0.01))
    sig = fw_k / 2.3548 / dl
    x = np.arange(-int(6 * sig) - 1, int(6 * sig) + 2)
    k = np.exp(-0.5 * (x / sig) ** 2)
    k /= k.sum()
    fc = np.convolve(f, k, mode="same")
    e = bordes(w_obj)
    c = np.concatenate([[0.0], np.cumsum(fc * dl)])
    xi = np.concatenate([[w[0] - dl / 2], w + dl / 2]) + desplaz
    a = np.interp(e, xi, c, left=np.nan, right=np.nan)
    return np.diff(a) / np.diff(e)


def ew_definiciones(w, f, catalogo):
    """EW con las cuatro definiciones de `ventanas` sobre un espectro sin ruido (en `runmed`, la mediana corrida de
    80 Å con «finito y a más de 15 Å de Hα»); y C(Hα) con la de G2."""
    from musepipe.lines import _channel_widths
    from musepipe.spectral import continuum_running_median
    sp = dict(wave=w, flux=f, flags=np.zeros(w.size, int))
    dl = _channel_widths(w)
    out = {}
    for d in DEFS:
        lin, cont_fn = ventanas(catalogo, sp, d)
        if d == "runmed":
            good = np.isfinite(f) & (np.abs(w - HA_A) > 15.0)
            cont = continuum_running_median(w, f, good, window_A=80.0, min_pixels=15)
        else:
            cont = cont_fn(f)
        r = f - cont
        ok = lin & np.isfinite(r)
        out[d] = float(np.sum(-(r[ok] / cont[ok]) * dl[ok]))
        if d == "G2":
            out["C_G2"] = float(np.interp(HA_A, w, cont))
    return out


def nativo(w, f):
    """EW y W10 a la resolución de la plantilla (continuo lineal en 8–48 Å del pico, línea ±6 Å, cruce en ±10 Å
    tras un suavizado de 3 píxeles para que no lo marque un píxel de ruido)."""
    s = (w > 6400) & (w < 6720)
    w, f = w[s], f[s]
    sel = np.abs(w - HA_A) < 4
    pk = w[sel][np.nanargmax(f[sel])]
    d = w - pk
    cm_ = ((d >= -48) & (d <= -8)) | ((d >= 8) & (d <= 48))
    ok = cm_ & np.isfinite(f)
    cont = np.polyval(np.polyfit(d[ok], f[ok], 1), d)
    lin = np.abs(d) <= 6
    dl = np.gradient(w)
    ew = float(np.sum(-((f - cont) / cont)[lin] * dl[lin]))
    fs = np.convolve(f - cont, np.ones(3) / 3, mode="same")
    v = np.abs(d) <= 10
    return dict(pico_A=float(pk), ew_nativa=ew, w10_nativa=ancho_cruce_A(w[v], fs[v], 0.1) / pk * C_KMS,
                fwhm_nativa=ancho_cruce_A(w[v], fs[v], 0.5) / pk * C_KMS)


def rojo(w, f):
    """El nivel rojo: la mediana en 8600–9100 Å, donde los seis métodos coinciden a ~3 %."""
    s = (w >= 8600) & (w <= 9100)
    return float(np.nanmedian(f[s]))


def mide_plantillas(ctx, modo_r="declaracion"):
    """Cada plantilla M6–L1 de las bibliotecas jóvenes de G3, a su resolución nativa y degradada a MUSE e integrada en
    los canales del compañero detectado (12 b) con el pico en el suyo: EW (cuatro definiciones), W10 y FWHM
    (estimador del paper), y el cociente C(Hα, G2)/nivel rojo."""
    from musepipe.models.libraries import resolve_declaration, template_library_entries
    from musepipe.models.templates import EmpiricalTemplateLibrary
    det = ctx.detectado
    raiz = ctx.bibliotecas()
    sp = ctx.espectro(det, "psffit")
    w_obj = sp["wave"]
    lsf = float(ctx.cfg(det)["h01_lsf_fwhm_A"])
    cat = ctx.catalogo(det)
    entradas_g3 = {e["subdir"]: e for e in template_library_entries(ctx.cfg(det))}
    filas = []
    for nombre_lib in BIBLIOTECAS:
        fam = raiz / nombre_lib
        decl = resolve_declaration({"name": nombre_lib, "subdir": nombre_lib}, fam)
        lib = None
        if modo_r == "cabecera":
            # la biblioteca como la construye G3 (`stage_g3_template_fit._build_libraries`): entrada del config
            entrada = entradas_g3.get(nombre_lib, {"name": nombre_lib, "subdir": nombre_lib})
            decl_g3 = resolve_declaration(entrada, fam)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                lib = EmpiricalTemplateLibrary(fam, gravity_class=decl_g3.gravity_class, citation=decl_g3.citation,
                                               version=entrada.get("version"), declaration=decl_g3)
        for p in sorted(glob.glob(str(fam / "*.npz"))):
            nombre = Path(p).stem
            if not any(f"_{s}" in nombre for s in SPT_PLANTILLAS):
                continue
            z = np.load(p, allow_pickle=True)
            meta = json.loads(str(z["meta_json"]))
            w, f = np.asarray(z["wave_A"], float), np.asarray(z["flux"], float)
            R, Rsrc = r_plantilla(decl, meta, modo_r, lib, p)
            nat = nativo(w, f)
            fm = degrada(w, f, R, w_obj, PICO_COMPANERO_A - nat["pico_A"], lsf)
            e = ew_definiciones(w_obj, fm, cat)
            filas.append(dict(biblioteca=nombre_lib, plantilla=nombre, spt=meta.get("spt"), R=R, R_fuente=Rsrc, **nat,
                              ew_muse=e, w10_muse=w10_kms(w_obj, fm, PICO_COMPANERO_A),
                              fwhm_muse=w10_kms(w_obj, fm, PICO_COMPANERO_A, fraccion=0.5),
                              cociente_C_rojo=e["C_G2"] / rojo(w_obj, fm)))
    return filas


def fotosferas_btsettl(ctx):
    """La «EW» que cada definición del continuo mide sobre una fotosfera BT-Settl pura (sin emisión, vacío -> aire),
    degradada igual: el cero fotosférico de cada definición (> 0 = absorción)."""
    from musepipe.models.airvac import vacuum_to_air
    det = ctx.detectado
    raiz = ctx.bibliotecas()
    sp = ctx.espectro(det, "psffit")
    w_obj = sp["wave"]
    lsf = float(ctx.cfg(det)["h01_lsf_fwhm_A"])
    cat = ctx.catalogo(det)
    fot = []
    for teff in TEFF_BTSETTL:
        for logg in LOGG_BTSETTL:
            p = raiz / "bt-settl-cifist" / f"teff{teff}_logg{logg}.npz"
            if not p.exists():
                continue
            z = np.load(p, allow_pickle=True)
            w = vacuum_to_air(np.asarray(z["wave_A"], float))
            f = np.asarray(z["flux"], float)
            s = (w > 6000) & (w < 9400)
            fm = degrada(w[s], f[s], 1e6, w_obj, 0.0, lsf)
            e = ew_definiciones(w_obj, fm, cat)
            fot.append(dict(teff=teff, logg=logg, ew_fot=e, cociente_C_rojo=e["C_G2"] / rojo(w_obj, fm)))
    return fot


def delta_k_ccm():
    """A(6563)/A_V − <A(8600–9100)>/A_V con CCM R_V = 3.1 (la extinción de la cadena)."""
    from musepipe.models.extinction import CCMExtinction
    ext = CCMExtinction(3.1, citation="Cardelli, Clayton & Mathis 1989")
    k_ha = float(ext.a_lambda_over_av(np.array([6563.0]))[0])
    k_rojo = float(np.mean(ext.a_lambda_over_av(np.linspace(8600, 9100, 51))))
    return k_ha, k_rojo, k_ha - k_rojo


def av_prior_bt_settl(ctx, slug):
    """La A_V mediana del ajuste BT-Settl con el prior de la primaria (Tabla B.6), de las cadenas; None sin ellas."""
    from scripts import teff_bayes as tb
    for f in sorted(glob.glob(str(ctx.cadenas / "*" / "*" / f"{slug}_*.json"))):
        d = json.loads(Path(f).read_text())
        if (d.get("slug") == slug and tb._de_la_tabla(d) and d.get("rejilla", "btsettl") == "btsettl"
                and d.get("av_fija") is None and d.get("semilla") == 1):
            return float(d["posterior"]["av"]["p50"])
    return None


def prediccion_continuo(ctx, filas, ewj, av_nodos):
    """C_pred(A_V) = cociente(plantilla) × nivel rojo(método) × 10^(−0.4 A_V ΔK): el continuo bajo Hα que predice
    cada plantilla del tipo del objeto desde el nivel rojo del compañero; con él, la EW (12 b, flujo directo de la
    definición de G2) o el límite de EW (42B b, límite de E3) y la A_V que haría falta para dar el continuo medido."""
    k_ha, k_rojo, dk = delta_k_ccm()
    cocientes = {}
    for r in filas:
        cocientes.setdefault(r["plantilla"], []).append(r["cociente_C_rojo"])
    out = dict(k_ha=k_ha, k_rojo=k_rojo, dk=dk, objetos={})
    for slug in ctx.slugs:
        nombres = USO_CONTINUO[ctx.papel(slug)]
        q = {n: float(np.mean(cocientes[n])) for n in nombres if n in cocientes}
        qm = {n: v for n, v in q.items() if n in MANARA}
        esc = ctx.escala_cgs(slug)[0]
        lims = ctx.limites_e3(slug) if ctx.papel(slug) == LIMITE else None
        o = dict(cocientes=q, av_nodos=list(av_nodos[slug]), metodos={})
        for m in CON_CONTINUO:
            fila = next(x for x in ewj["filas"] if x["slug"] == slug and x["metodo"] == m and x["definicion"] == "G2")
            nivel = rojo(ctx.espectro(slug, m)["wave"], ctx.espectro(slug, m)["flux"])
            ratio_obs = fila["C"] / nivel
            av_nec = {n: float(-2.5 * np.log10(ratio_obs / v) / dk) for n, v in q.items()}
            pred = {}
            for av in av_nodos[slug]:
                fac = 10 ** (-0.4 * av * dk)
                cs = np.array([v * nivel * fac for v in q.values()])
                csm = np.array([v * nivel * fac for v in qm.values()]) if qm else cs
                d = dict(C_pred_min=float(cs.min()), C_pred_max=float(cs.max()), C_pred_manara_media=float(csm.mean()))
                if lims is None:
                    F = fila["F"]
                    d.update(EW_pred_rango=[float(-F / cs.min()), float(-F / cs.max())],
                             EW_pred_manara=float(-F / csm.mean()))
                else:
                    Fl = float(lims[m]["f_lim_observed"]) / esc
                    d.update(EWlim_pred_rango=[float(Fl / cs.max()), float(Fl / cs.min())],
                             EWlim_pred_manara=float(Fl / csm.mean()))
                pred[f"{av:g}"] = d
            o["metodos"][m] = dict(C_obs=fila["C"], sigma_C_ctrl=fila["sigma_C"], rojo=nivel, cociente_obs=ratio_obs,
                                   av_necesaria=av_nec, prediccion=pred)
        out["objetos"][slug] = o
    return out


def nodos_av(ctx):
    """Las A_V a las que se predice el continuo: para el compañero detectado (12 b), la adoptada (`h03_av`), 3.0 y la
    del ajuste con prior (Tabla B.6, de las cadenas: 3.74); para el de la cota (42B b), la adoptada."""
    out = {}
    for slug in ctx.slugs:
        av = [float(ctx.cfg(slug)["h03_av"])]
        if ctx.papel(slug) == DETECCION:
            av.append(3.0)
            prior = av_prior_bt_settl(ctx, slug)
            if prior is not None:
                av.append(prior)
        out[slug] = tuple(av)
    return out


def cmd_plantillas(ctx, ewj=None, modo_r="declaracion"):
    if ewj is None:
        ewj = cmd_ew(ctx)
    filas = mide_plantillas(ctx, modo_r)
    fot = fotosferas_btsettl(ctx)
    cont = prediccion_continuo(ctx, filas, ewj, nodos_av(ctx))
    out = dict(modo_r=modo_r, plantillas=filas, fotosferas=fot, continuo=cont)
    out["resumen"] = resumen_plantillas(out, ctx.detectado, ctx.limite)
    _imprime_plantillas(out)
    return out


def _code(spt):
    from musepipe.constants import spt_code
    return spt_code(str(spt))


def resumen_plantillas(out, det, lim):
    """Las cifras del texto: la W10 a la resolución de MUSE de las Class III M7–M9.5 (sin `EXCLUIDAS_W10`), la EW de
    las M9/M9.5 de Manara, la EW del compañero detectado `det` con el continuo predicho por A_V (psffit) y el límite
    de EW del de la cota `lim`."""
    sel = [r for r in out["plantillas"] if _code("M7") <= _code(r["spt"]) <= _code("M9.5")
           and r["plantilla"] not in EXCLUIDAS_W10 and np.isfinite(r["w10_muse"])]
    w10 = [r["w10_muse"] for r in sel]
    m9 = [-r["ew_muse"]["G2"] for r in out["plantillas"] if r["plantilla"] in MANARA_M9]
    c12 = out["continuo"]["objetos"][det]["metodos"]["psffit"]["prediccion"]
    c42 = out["continuo"]["objetos"][lim]["metodos"]["psffit"]["prediccion"]
    ew12 = {av: sorted(-x for x in d["EW_pred_rango"]) for av, d in c12.items()}
    lim42 = {av: d["EWlim_pred_rango"] for av, d in c42.items()}
    todos12 = [x for v in ew12.values() for x in v]
    btp = [r["ew_fot"]["ancha"] for r in out["fotosferas"] if 2400 <= r["teff"] <= 2800]
    btg = [r["ew_fot"][d] for r in out["fotosferas"] if 2400 <= r["teff"] <= 2800 for d in ("G2", "med", "runmed")]
    return dict(w10_muse_M7_M95=[min(w10), max(w10)], w10_muse_plantillas=[r["plantilla"] for r in sel],
                w10_nativa_M7_M95=[min(r["w10_nativa"] for r in sel), max(r["w10_nativa"] for r in sel)],
                ew_manara_M9_M95=[min(m9), max(m9)], ew_12b_por_av=ew12,
                ew_12b_sistematico=[min(todos12), max(todos12)], ew_lim_42bb_por_av=lim42,
                ew_fotosferica_ancha_2400_2800=[min(btp), max(btp)],
                ew_fotosferica_otras_2400_2800=[min(btg), max(btg)])


def _imprime_plantillas(out):
    print(f"R de las plantillas: {out['modo_r']}")
    print("| plantilla | SpT | R | EW nativa | W10 nativa | EW MUSE G2/med/ancha/runmed | W10 MUSE | FWHM MUSE | C(Hα)/rojo |")
    print("|---|---|---|---|---|---|---|---|---|")
    for r in out["plantillas"]:
        e = r["ew_muse"]
        print(f"| {r['plantilla']} ({r['biblioteca']}) | {r['spt']} | {r['R']:.0f} | {r['ew_nativa']:+.1f} | "
              f"{r['w10_nativa']:.0f} | {e['G2']:+.1f}/{e['med']:+.1f}/{e['ancha']:+.1f}/{e['runmed']:+.1f} | "
              f"{r['w10_muse']:.0f} | {r['fwhm_muse']:.0f} | {r['cociente_C_rojo']:.3f} |")
    print("\nBT-Settl (EW fotosférica por definición, Å; > 0 = absorción):")
    for r in out["fotosferas"]:
        e = r["ew_fot"]
        print(f"  {r['teff']} K log g {r['logg']}: G2 {e['G2']:+.2f} med {e['med']:+.2f} ancha {e['ancha']:+.2f} "
              f"runmed {e['runmed']:+.2f}")
    c = out["continuo"]
    print(f"\nCCM R_V 3.1: A(6563)/A_V {c['k_ha']:.3f}, A(8600–9100)/A_V {c['k_rojo']:.3f}, diferencia {c['dk']:.3f}")
    for slug, o in c["objetos"].items():
        print(f"\n### {slug}: " + ", ".join(f"{k} {v:.3f}" for k, v in o["cocientes"].items()))
        for m, x in o["metodos"].items():
            avs = np.array(list(x["av_necesaria"].values()))
            print(f"  {m}: C {x['C_obs']:.1f} ± {x['sigma_C_ctrl']:.1f}, rojo {x['rojo']:.0f} -> A_V necesaria "
                  f"{avs.min():.1f}–{avs.max():.1f}")
            for av, d in x["prediccion"].items():
                if "EW_pred_rango" in d:
                    print(f"     A_V {av}: C_pred {d['C_pred_min']:.0f}–{d['C_pred_max']:.0f} -> EW "
                          f"{d['EW_pred_rango'][0]:.1f} … {d['EW_pred_rango'][1]:.1f} Å")
                else:
                    print(f"     A_V {av}: C_pred {d['C_pred_min']:.0f}–{d['C_pred_max']:.0f} -> EW_lim "
                          f"{d['EWlim_pred_rango'][0]:.1f}–{d['EWlim_pred_rango'][1]:.1f} Å")
    r = out["resumen"]
    print(f"\nW10 a la resolución de MUSE, Class III M7–M9.5 ({len(r['w10_muse_plantillas'])}): "
          f"{r['w10_muse_M7_M95'][0]:.0f}–{r['w10_muse_M7_M95'][1]:.0f} km/s (nativa "
          f"{r['w10_nativa_M7_M95'][0]:.0f}–{r['w10_nativa_M7_M95'][1]:.0f}); EW de las M9/M9.5 de Manara "
          f"{r['ew_manara_M9_M95'][0]:.1f}–{r['ew_manara_M9_M95'][1]:.1f} Å")


# ====================================================================================== primaria y fuga
def cmd_primaria(ctx, inst=None, plantillas=None, ewj=None):
    """Sect. 5.1: la W10 de la primaria por noche (`cifras_por_noche_d2.noche`), sin la LSF gaussiana (lo que da ese
    script, 484/250) y sin la W10 instrumental del estimador (211–222 km/s, la de `w10`). Sect. 4.1: la fuga de la
    primaria, su EW de la noche buena (±6 Å) entre la EW del compañero (G2) o la de las plantillas a la A_V adoptada.
    Es el sistema del compañero detectado: su `targets/<slug>.json` -> `paper` declara las dos noches
    (`previous_run`, la buena, y `second_epoch_run`)."""
    det = ctx.detectado
    perfil = ctx.perfil(det)["paper"]
    if not all(k in perfil for k in ("previous_run", "second_epoch_run")):
        raise SystemExit(f"{det}: targets/ no declara las dos noches (paper.previous_run, paper.second_epoch_run)")
    if inst is None:
        inst = instrumentales(ctx)
    noches = {}
    for clave in ("previous_run", "second_epoch_run"):
        run = perfil[clave]
        n = cn.noche(ctx.rp(run).stage_dir, float(ctx.cfg(run)["h01_lsf_fwhm_A"]))
        w10 = n["ancho10_prim_kms"]
        n["ancho10_prim_sin_inst_kms"] = {m: float(math.sqrt(w10 ** 2 - v[0] ** 2)) for m, v in inst.items()}
        noches[run] = n
    if ewj is None:
        ewj = cmd_ew(ctx)
    if plantillas is None:
        plantillas = cmd_plantillas(ctx, ewj)
    n1 = noches[perfil["previous_run"]]
    ew_comp = -ewj["resumen"]["ew_g2"]
    av0 = f"{float(ctx.cfg(det)['h03_av']):g}"
    ew_plant = min(plantillas["resumen"]["ew_12b_por_av"][av0])
    fuga = dict(ew_primaria_noche1=-n1["EW_prim_A"], ew_companero_g2=ew_comp, ew_plantillas_av_adoptada=ew_plant,
                f_max=-n1["EW_prim_A"] / ew_comp, f_max_plantillas=-n1["EW_prim_A"] / ew_plant)
    out = dict(instrumental_w10_kms={m: v[0] for m, v in inst.items()}, noches=noches, fuga=fuga)
    for run, n in noches.items():
        s = n["ancho10_prim_sin_inst_kms"]
        print(f"primaria {run}: W10 {n['ancho10_prim_kms']:.0f} km/s; sin la LSF gaussiana de "
              f"{n['lsf_fwhm_A']:.3f} Å {n['ancho10_prim_sin_lsf_kms']:.0f}; sin la W10 instrumental del estimador "
              f"{min(s.values()):.0f}–{max(s.values()):.0f} km/s")
    print(f"fuga: EW de la primaria (noche buena) {fuga['ew_primaria_noche1']:.2f} Å -> f < "
          f"{fuga['ew_primaria_noche1']:.2f}/{ew_comp:.1f} = {100 * fuga['f_max']:.1f} %, o "
          f"{fuga['ew_primaria_noche1']:.2f}/{ew_plant:.1f} = {100 * fuga['f_max_plantillas']:.1f} % con el continuo "
          f"de las plantillas a A_V {av0}")
    return out


# ====================================================================================== cromosfera (ap. B)
#: Alcalá et al. 2014, Tabla 4, Hα (la que usó Manara et al. 2013 «in prep.»): solo para la variante
A14 = (1.12, 1.50)
#: Manara et al. 2013, Ec. 2 (`teff_bayes.SUELO`)
EQ2 = (6.17, -24.54)
#: estrellas que Manara et al. (2013) excluyen del ajuste de la Ec. 2 (su Fig. 13)
EXCLUIDAS_EQ2 = ("TWA9A", "Sz122")
PC_CM = 3.0857e18
LSUN_ERG_S = 3.828e33
#: saturación de la actividad cromosférica (Barrado y Navascués & Martín 2003)
SATURACION = -3.3
#: West et al. 2011: umbrales de EW (Å) a los que se da la fracción de activas
UMBRALES_WEST = (12, 20, 26.2, 40, 53)


def lee_manara2013(directorio=DATOS / "manara2013"):
    """Las 24 Class III de Manara et al. (2013) con lo que hace falta de las tablas 1, 3 y 6 de VizieR (formato byte
    a byte de su ReadMe): nombre, SpT, T_eff, log L★, ⟨log L_acc,noise⟩ (media de líneas), distancia y F(Hα)."""
    d = Path(directorio)
    t1 = {ln[0:10].strip(): ln for ln in open(d / "table1.dat")}
    t3 = {ln[0:10].strip(): ln for ln in open(d / "table3.dat")}
    t6 = {ln[0:10].strip(): ln for ln in open(d / "table6.dat")}
    return [dict(nombre=n, spt=l3[12:16].strip(), teff=int(l3[17:21]), logL=float(l3[22:27]),
                 lacc_media=float(l3[33:38]), dist_pc=float(t1[n][73:76]), f_ha=float(t6[n][11:17]) * 1e-14)
            for n, l3 in t3.items()]


def suelo(teff):
    return EQ2[0] * np.log10(teff) + EQ2[1]


def residuos_manara(estrellas, a, b):
    """Residuo de cada estrella respecto a la Ec. 2, en log(L_acc/L★) con L_acc = a·log L_Hα + b (L_Hα = 4πd²F, en
    L_sol), con los coeficientes de Alcalá 2014 y con la media de líneas de Manara."""
    filas = []
    for e in estrellas:
        lha = math.log10(4 * math.pi * (e["dist_pc"] * PC_CM) ** 2 * e["f_ha"] / LSUN_ERG_S)
        s = float(suelo(e["teff"]))
        filas.append(dict(e, d_ha=a * lha + b - e["logL"] - s, d_ha_a14=A14[0] * lha + A14[1] - e["logL"] - s,
                          d_media=e["lacc_media"] - e["logL"] - s, lha_lbol=lha - e["logL"],
                          y_media=e["lacc_media"] - e["logL"]))
    return filas


def media_y_dispersion(v):
    v = np.asarray(v, float)
    return dict(media=float(v.mean()), sd=float(v.std(ddof=1)), n=int(v.size))


def poblacion_manara(a, b, rng, estrellas=None, n_boot=4000):
    """Media y dispersión de los residuos de las 22 estrellas del ajuste (y de las tres de T_eff < 3000 K) respecto a
    la Ec. 2, y la incertidumbre de la predicción a cada T_eff por bootstrap de las 22 (MCO; el artículo no dice cómo
    ajustó). `rng` se consume en el bootstrap: `n_boot` sorteos de 22 índices."""
    filas = residuos_manara(estrellas or lee_manara2013(), a, b)
    fit = [f for f in filas if f["nombre"] not in EXCLUIDAS_EQ2]
    tarde = [f for f in fit if f["teff"] < 3000]
    pob = dict(ha=media_y_dispersion([f["d_ha"] for f in fit]), ha_a14=media_y_dispersion([f["d_ha_a14"] for f in fit]),
               media_lineas=media_y_dispersion([f["d_media"] for f in fit]),
               ha_tardias=media_y_dispersion([f["d_ha"] for f in tarde]),
               tardias=[(f["nombre"], f["spt"], f["teff"], round(f["d_ha"], 3)) for f in tarde],
               coeficientes_ha=[a, b])
    t = np.log10([f["teff"] for f in fit])
    y = np.asarray([f["y_media"] for f in fit])
    tg = np.log10(np.arange(2000, 3401, 50))
    preds = []
    for _ in range(n_boot):
        i = rng.integers(0, t.size, t.size)
        if np.unique(t[i]).size < 3:
            continue
        preds.append(np.polyval(np.polyfit(t[i], y[i], 1), tg))
    pob["sigma_coef_T"] = dict(teff=(10 ** tg).round().tolist(), sigma=np.asarray(preds).std(axis=0).tolist())
    return pob, filas


def sigma_coef(pob, teff):
    return np.interp(teff, pob["sigma_coef_T"]["teff"], pob["sigma_coef_T"]["sigma"])


def fraccion_poblacion(delta, media, sd, sigma_coeficientes=0.0):
    """P_pob = media de Φ((μ − Δ)/σ_tot) sobre las muestras: la fracción de estrellas Class III de esa T_eff cuya Hα,
    leída como acreción con la misma relación, alcanza o supera la del objeto (σ_tot² = σ_pob² + σ_coef²). Es un
    valor p bajo la hipótesis cromosférica, NO la probabilidad posterior de que la línea sea cromosférica."""
    from scipy.stats import norm
    st = np.sqrt(sd ** 2 + np.asarray(sigma_coeficientes, float) ** 2)
    return float(np.mean(norm.cdf((media - np.asarray(delta, float)) / st)))


def carga_cadenas(directorio, slug):
    """Las corridas de `teff_bayes.py ajusta` de `slug` sobre los datos (no plantillas ni inyecciones), `libre` o
    `evo`, con su npz; en orden de fichero, con su grupo (el directorio de dos niveles arriba)."""
    out = []
    for f in sorted(glob.glob(str(Path(directorio) / "*" / "*" / f"{slug}_*.json"))):
        d = json.loads(Path(f).read_text())
        if d.get("verdad") or d.get("plantilla") or d["modo"] not in ("libre", "evo"):
            continue
        z = Path(f).with_suffix(".npz")
        if not z.exists():
            continue
        d["_npz"], d["_grupo"] = str(z), Path(f).parts[-3]
        out.append(d)
    return out


def _poblaciones(pob):
    return (("pob_todas", pob["ha"]), ("pob_tardias", pob["ha_tardias"]), ("pob_media_lineas", pob["media_lineas"]))


def evalua_cadena(d, entrada, pob, rng, sig_f_dex, a, b):
    """Por muestra de una cadena: Δ (`teff_bayes.margen_de_corrida`), log(L_Hα/L_bol) a la A_V de la muestra, y las
    fracciones de población con el error del flujo de la línea (`sig_f_dex` en L_acc) sorteado por muestra."""
    from scripts import teff_bayes as tb
    delta = tb.margen_de_corrida(d, entrada)
    z = np.load(d["_npz"])
    nombres = list(z["nombres"])
    av = z["muestras"][:, nombres.index("av")] if "av" in nombres else np.full(z["derivadas"].shape[0], entrada["av0"])
    teff, lbol = z["derivadas"][:, 0], z["derivadas"][:, 4]
    rha = (math.log10(entrada["lacc0"]) - b) / a + 0.4 * entrada["ahav"] * (av - entrada["av0"]) - np.log10(lbol)
    ok = np.isfinite(delta) & np.isfinite(teff)
    delta, rha, av, teff = delta[ok], rha[ok], av[ok], teff[ok]
    dF = rng.normal(0, sig_f_dex, delta.size)
    res = dict(grupo=d["_grupo"], etiqueta=d["etiqueta"], rejilla=d.get("rejilla", "btsettl"), modo=d["modo"],
               datos=d.get("datos", "todo"), av_fija=d.get("av_fija"), av_modo=d.get("av"), r=d.get("r_analisis"),
               semilla=d.get("semilla"), tabla_b6=bool(tb._de_la_tabla(d)), tabla_b7=bool(tb._de_rejillas(d)),
               teff_p50=float(np.median(teff)), av_p50=float(np.median(av)),
               delta_p16_50_84=np.percentile(delta, [16, 50, 84]).tolist(), p_bajo_suelo=float(np.mean(delta < 0)),
               n=int(delta.size))
    for clave, p in _poblaciones(pob):
        res[f"P_{clave}"] = fraccion_poblacion(delta + dF, p["media"], p["sd"], sigma_coef(pob, teff))
    res["lha_lbol_p16_50_84"] = np.percentile(rha, [16, 50, 84]).tolist()
    res["P_saturacion"] = float(np.mean(rha > SATURACION))
    return res


def evo_solo_filas(ctx, slug, entrada, pob, rng, sig_f_dex, a, b):
    """T_eff de los tracks con la L_bol publicada (sin espectro, `teff_bayes.evo_solo`), la A_V adoptada y el error de
    L_bol (0.1 dex) sorteado por muestra."""
    from scripts import teff_bayes as tb
    cfg = ctx.cfg(slug)
    out = []
    for tracks in ("BHAC15", "ATMO2020"):
        teff = tb.evo_solo(cfg, slug, tracks, root=ctx.root)["teff"]
        dl = rng.normal(0, 0.10, teff.size)
        delta = math.log10(entrada["lacc0"]) - (math.log10(entrada["lbol_lit"]) + dl) - suelo(teff)
        dF = rng.normal(0, sig_f_dex, teff.size)
        fila = dict(etiqueta=f"evo_solo {tracks}", teff_p50=float(np.median(teff)), av_p50=entrada["av0"],
                    delta_p16_50_84=np.percentile(delta, [16, 50, 84]).tolist(), p_bajo_suelo=float(np.mean(delta < 0)),
                    tabla_b6=False, tabla_b7=False)
        for clave, p in _poblaciones(pob):
            fila[f"P_{clave}"] = fraccion_poblacion(delta + dF, p["media"], p["sd"], sigma_coef(pob, teff))
        rha = (math.log10(entrada["lacc0"]) - b) / a - (math.log10(entrada["lbol_lit"]) + dl)
        fila["lha_lbol_p16_50_84"] = np.percentile(rha, [16, 50, 84]).tolist()
        fila["P_saturacion"] = float(np.mean(rha > SATURACION))
        out.append(fila)
    return out


def lee_west2011(spt, directorio=DATOS / "west2011"):
    """(EW de Hα, actividad) de las M`spt` de West et al. (2011): EW en Å (> 0 en emisión), `MHa` '1'/'0'/''."""
    rows = [ln.rstrip("\n").split("\t") for ln in open(Path(directorio) / f"west2011_{spt}.tsv")
            if ln.strip() and not ln.startswith("#")]
    hdr, data = rows[0], rows[3:]
    i, j = hdr.index("W(Ha)"), hdr.index("MHa")
    ew, act = [], []
    for r in data:
        try:
            ew.append(float(r[i]))
            act.append(r[j].strip())
        except (ValueError, IndexError):
            continue
    return np.array(ew), np.array(act)


def fracciones_west(ew, act, umbrales=UMBRALES_WEST):
    x = ew[act == "1"]
    n_tot = int(np.isin(act, ("0", "1")).sum())
    return dict(n_activas=int(x.size), n_clasificadas=n_tot,
                frac_activas_ge={str(t): float(np.mean(x >= t)) for t in umbrales},
                frac_todas_ge={str(t): float(np.sum(x >= t) / n_tot) for t in umbrales},
                mediana_activas=float(np.median(x)))


def cmd_cromosfera(ctx):
    from scripts import teff_bayes as tb
    if not ctx.cadenas.is_dir():
        raise SystemExit(f"sin las cadenas de T_eff en {ctx.cadenas} (--cadenas DIR)")
    det = ctx.detectado
    cfg_det = ctx.cfg(det)
    a, b = float(cfg_det["h03_lacc_lha_a"]), float(cfg_det["h03_lacc_lha_b"])
    g2 = ctx.g2_halpha(det)
    # el error del flujo de la línea detectada (12 b; G2: 2381 ± 247, 10.37 %), en L_acc; el mismo ancho para los dos
    # (la sesión usó 10.4 %, redondeado: las fracciones cambian en < 5e-5)
    sig_f_dex = a * math.log10(1 + g2["flux_direct_err"] / g2["flux_direct"])
    rng = np.random.default_rng(SEMILLA_CROMOSFERA)
    pob, filas_m13 = poblacion_manara(a, b, rng)
    out = dict(semilla=SEMILLA_CROMOSFERA, coeficientes=dict(a=a, b=b), sig_f_dex=sig_f_dex, poblacion=pob,
               manara13=filas_m13, objetos={}, cadenas=str(ctx.cadenas))
    print("Población Class III (Manara et al. 2013, 22 del ajuste), residuo respecto a la Ec. 2:")
    for k in ("ha", "ha_a14", "media_lineas", "ha_tardias"):
        print(f"  {k}: media {pob[k]['media']:+.3f}, sd {pob[k]['sd']:.3f}, n {pob[k]['n']}")
    print("  tardías:", pob["tardias"])
    print("  σ de los coeficientes (bootstrap) a 2400/2600/2800 K:",
          [round(float(sigma_coef(pob, t)), 3) for t in (2400, 2600, 2800)])
    for slug in ctx.slugs:
        entrada = tb.entrada_margen(slug, root=ctx.root)
        filas = [evalua_cadena(d, entrada, pob, rng, sig_f_dex, a, b) for d in carga_cadenas(ctx.cadenas, slug)]
        filas += evo_solo_filas(ctx, slug, entrada, pob, rng, sig_f_dex, a, b)
        out["objetos"][slug] = dict(entrada=entrada, filas=filas)
        print(f"\n### {slug} ({entrada['tipo']}; log L_acc/L_bol publicada {math.log10(entrada['lacc0'] / entrada['lbol_lit']):+.2f})")
        print("| grupo | cadena | T_eff | A_V | Δ p16/p50/p84 | P(Δ<0) | P_pob (22) | P_pob (M6.5–M9.5) | log LHα/Lbol | P(>−3.3) |")
        print("|---|---|---|---|---|---|---|---|---|---|")
        for r in filas:
            dd = r["delta_p16_50_84"]
            print(f"| {r.get('grupo', '—')} | {r['etiqueta']} | {r['teff_p50']:.0f} | {r['av_p50']:.2f} | "
                  f"{dd[0]:+.2f}/{dd[1]:+.2f}/{dd[2]:+.2f} | {r['p_bajo_suelo']:.2f} | {r['P_pob_todas']:.2f} | "
                  f"{r['P_pob_tardias']:.2f} | {r['lha_lbol_p16_50_84'][1]:+.2f} | {r['P_saturacion']:.3f} |")
    out["west2011"] = {spt: fracciones_west(*lee_west2011(spt)) for spt in ("M7", "M8", "M9")}
    out["resumen"] = resumen_cromosfera(out, det, ctx.limite)
    r = out["resumen"]
    print("\nWest et al. 2011, activas con EW ≥ 26.2 Å:",
          {k: round(v["frac_activas_ge"]["26.2"], 3) for k, v in out["west2011"].items()})
    print(f"Tabla B.7 ({det}), P_pob: A_V 1.8 {r['P_12b_av18']}, 3.0 {r['P_12b_av30']}, prior {r['P_12b_prior']}; "
          f"solo M6.5–M9.5 a 1.8 {r['P_12b_tardias_av18']}")
    print(f"Tabla B.6 ({ctx.limite}, BT-Settl y DRIFT-PHOENIX, A_V ≤ 3): {r['P_42bb_av_le_3']}")
    return out


def _rango(v):
    return [float(min(v)), float(max(v))]


def resumen_cromosfera(out, det, lim):
    """Las cifras del apéndice B y de la Sect. 4.1/4.2: la media y la dispersión de la población, el bootstrap a
    2400–2800 K, las tardías, y los rangos de P_pob sobre las filas de la Tabla B.7 del compañero detectado `det`
    (12 b; semilla 1, A_V 1.8, 3.0 y con prior) y de la Tabla B.6 del de la cota `lim` (42B b; BT-Settl y
    DRIFT-PHOENIX, semilla 1, A_V ≤ 3)."""
    pob = out["poblacion"]
    f12 = [r for r in out["objetos"][det]["filas"] if r.get("tabla_b7") and r.get("semilla") == 1]
    f42 = [r for r in out["objetos"][lim]["filas"] if r.get("tabla_b6") and r.get("semilla") == 1
           and r["av_p50"] <= 3.0 + 1e-9]
    por_av = {k: [r for r in f12 if r["av_fija"] == v] for k, v in (("av18", 1.8), ("av30", 3.0), ("prior", None))}
    return dict(media_ha=pob["ha"]["media"], sd_ha=pob["ha"]["sd"], n_ha=pob["ha"]["n"],
                sigma_coef_2400_2800=_rango([sigma_coef(pob, t) for t in (2400, 2600, 2800)]),
                media_tardias=pob["ha_tardias"]["media"],
                P_12b_av18=_rango([r["P_pob_todas"] for r in por_av["av18"]]),
                P_12b_av30=_rango([r["P_pob_todas"] for r in por_av["av30"]]),
                P_12b_prior=_rango([r["P_pob_todas"] for r in por_av["prior"]]),
                P_12b_tardias_av18=_rango([r["P_pob_tardias"] for r in por_av["av18"]]),
                n_filas_tabla_b7={k: len(v) for k, v in por_av.items()},
                P_42bb_av_le_3=_rango([r["P_pob_todas"] for r in f42]), n_filas_42bb=len(f42),
                P_saturacion_12b_av_le_38=max(r["P_saturacion"] for r in out["objetos"][det]["filas"]
                                              if r["av_p50"] <= 3.8))


# ====================================================================================== las cifras del paper
def _r(x, nd=0):
    return f"{x:.{nd}f}"


def cifras_del_paper(res):
    """[(dónde, lo que dice el paper, lo que sale)] con las cifras de `res` ({subcomando: salida})."""
    filas = []
    if "ew" in res:
        r = res["ew"]["resumen"]
        filas += [("Sect. 4.1, Tabla 6", "EW 26 ± 4 Å (error de los 33 controles)",
                   f"{-r['ew_g2']:.1f} ± {r['ew_err_controles']:.2f}"),
                  ("Sect. 4.1", "G2: 26.2 ± 3.6 (MC)", f"{-r['ew_g2']:.2f} ± {r['ew_err_g2_mc']:.2f}"),
                  ("Sect. 4.1", "PSF fit y apertura, tres definiciones: 23–34 Å",
                   "–".join(_r(x, 1) for x in r["ew_psffit_aperture_3defs"])),
                  ("Sect. 4.1", "las dos óptimas: 67–89 Å", "–".join(_r(x, 1) for x in r["ew_optimas_g2"])),
                  ("Tabla 6, Sect. 4.2", "42B b EW < 6 (límite de E3 / continuo de psffit)", _r(r["ew_lim_42bb_psffit"], 2))]
    if "plantillas" in res:
        r = res["plantillas"]["resumen"]
        for av, (lo, hi) in r["ew_12b_por_av"].items():
            filas.append(("Sect. 4.1", f"plantillas M7–M8.5 a A_V {float(av):.2f}: 12–22 / 17–32 / 22–40 Å",
                          f"{lo:.1f}–{hi:.1f}"))
        filas.append(("Tabla 6 (nota f)", "sistemático del continuo −12 a −40 Å",
                      f"{r['ew_12b_sistematico'][0]:.1f}–{r['ew_12b_sistematico'][1]:.1f}"))
        for av, (lo, hi) in r["ew_lim_42bb_por_av"].items():
            filas.append(("Tabla 6, Sect. 4.2", f"42B b, plantillas M9–L1 a A_V {float(av):.2f}: EW < 13 Å",
                          f"{lo:.1f}–{hi:.1f}"))
        filas += [("Sect. 4.2", "M9 y M9.5 de Manara medidas igual: 9–14 Å",
                   "–".join(_r(x, 1) for x in r["ew_manara_M9_M95"])),
                  ("Sect. 4.1", "Class III M7–M9.5 a la resolución de MUSE: W10 230–310 km/s",
                   "–".join(_r(x, 1) for x in r["w10_muse_M7_M95"]))]
    if "w10" in res:
        r = res["w10"]["resumen"]
        inst = r["w10_inst_por_modelo"]
        filas += [("Sect. 4.1", "W10 256 km/s", _r(r["w10_obs_psffit"], 1)),
                  ("Sect. 4.1, 5.1", "línea sin resolver 211–222 km/s",
                   "–".join(_r(x, 1) for x in (min(inst.values()), max(inst.values())))),
                  ("Sect. 4.1", "intrínseca 82–175 km/s (68 %)", "–".join(_r(x, 1) for x in r["w10_int_68_cielo_esc"])),
                  ("Sect. 4.1", "< 214 km/s (95 %); 198–214 con los tres modelos",
                   ", ".join(f"{k} {v:.1f}" for k, v in r["w10_int_95_sup_por_modelo"].items())),
                  ("Sect. 4.1", "sin resolver > 200 km/s en el 89–98 % de las inyecciones",
                   ", ".join(f"{k} {100 * v:.1f} %" for k, v in r["p_sin_resolver_sobre_200"].items()))]
    if "primaria" in res:
        r = res["primaria"]
        for run, n in r["noches"].items():
            s = n["ancho10_prim_sin_inst_kms"]
            filas.append(("Sect. 5.1", f"primaria {run}: W10 521/315; sin LSF gaussiana 484/250; ~475/~230",
                          f"{n['ancho10_prim_kms']:.1f}; {n['ancho10_prim_sin_lsf_kms']:.1f}; "
                          f"{min(s.values()):.1f}–{max(s.values()):.1f}"))
        fu = r["fuga"]
        filas.append(("Sect. 4.1", "fuga: 1.54/26 = 6 %, 1.54/12 = 13 %",
                      f"{fu['ew_primaria_noche1']:.3f}/{fu['ew_companero_g2']:.2f} = {100 * fu['f_max']:.1f} %; "
                      f"{fu['ew_primaria_noche1']:.3f}/{fu['ew_plantillas_av_adoptada']:.2f} = "
                      f"{100 * fu['f_max_plantillas']:.1f} %"))
    if "cromosfera" in res:
        r = res["cromosfera"]["resumen"]
        filas += [("Ap. B", "22 Class III: media +0.03, dispersión 0.25 dex",
                   f"{r['media_ha']:+.3f}, {r['sd_ha']:.3f} (n {r['n_ha']})"),
                  ("Ap. B", "bootstrap 0.07–0.11 dex a 2400–2800 K", "–".join(_r(x, 3) for x in r["sigma_coef_2400_2800"])),
                  ("Sect. 4.1, Ap. B", "12 b: 0.55–0.73 (A_V 1.8)", "–".join(_r(x, 3) for x in r["P_12b_av18"])),
                  ("Sect. 4.1, Ap. B", "12 b: 0.15–0.44 (A_V 3.0)", "–".join(_r(x, 3) for x in r["P_12b_av30"])),
                  ("Sect. 4.1, Ap. B", "12 b: 0.06–0.40 (prior)", "–".join(_r(x, 3) for x in r["P_12b_prior"])),
                  ("Sect. 4.2, Ap. B", "42B b: 0.21–0.70", "–".join(_r(x, 3) for x in r["P_42bb_av_le_3"])),
                  ("Ap. B", "M6.5–M9.5 0.23 dex por debajo; 0.12–0.29", f"{r['media_tardias']:+.3f}; "
                   + "–".join(_r(x, 3) for x in r["P_12b_tardias_av18"]))]
    return filas


def _json(obj, path):
    Path(path).write_text(json.dumps(obj, indent=1, ensure_ascii=False, default=_a_json))


def _a_json(x):
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    return str(x)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("orden", choices=("ew", "lsf", "w10", "plantillas", "primaria", "cromosfera", "todo"))
    ap.add_argument("--project-root", default=None,
                    help="raíz con runs/ (por defecto este clon; desde un worktree, la del clon que los tiene)")
    ap.add_argument("--cadenas", default=None,
                    help=f"directorio de las cadenas de `teff_bayes.py ajusta` (por defecto {CADENAS_C0}, las c0)")
    ap.add_argument("--r-plantillas", choices=("declaracion", "cabecera"), default="declaracion",
                    help="R de las plantillas: la de la declaración de la biblioteca (la de las cifras del paper) o "
                         "la de la cabecera del FITS original, como el cargador de G3 (sensibilidad)")
    ap.add_argument("--salida", default=None, help="directorio donde escribir <subcomando>.json")
    a = ap.parse_args(argv)
    ctx = Contexto(a.project_root, a.cadenas)
    salida = Path(a.salida) if a.salida else None
    if salida:
        salida.mkdir(parents=True, exist_ok=True)
    res = {}
    perfil = None
    if a.orden in ("ew", "plantillas", "primaria", "todo"):
        res["ew"] = cmd_ew(ctx)
    if a.orden in ("lsf", "w10", "primaria", "todo"):
        print()
        res["lsf"], perfiles = cmd_lsf(ctx, salida)
        perfil = perfiles["OI6300"]
    if a.orden in ("w10", "todo"):
        print()
        res["w10"] = cmd_w10(ctx, perfil)
    if a.orden in ("plantillas", "primaria", "todo"):
        print()
        res["plantillas"] = cmd_plantillas(ctx, res["ew"], a.r_plantillas)
    if a.orden in ("primaria", "todo"):
        print()
        inst = ({m: (v["w10_inst_cruce_kms"], v["fwhm_inst_cruce_kms"]) for m, v in res["w10"]["modelos"].items()}
                if "w10" in res else instrumentales(ctx, perfil))
        res["primaria"] = cmd_primaria(ctx, inst, res["plantillas"], res["ew"])
    if a.orden in ("cromosfera", "todo"):
        print()
        res["cromosfera"] = cmd_cromosfera(ctx)
    print("\n## Cifras del paper")
    print("| dónde | el paper | sale |")
    print("|---|---|---|")
    for donde, texto, valor in cifras_del_paper(res):
        print(f"| {donde} | {texto} | {valor} |")
    if salida:
        for k, v in res.items():
            _json(v, salida / f"{k}.json")
        _json(cifras_del_paper(res), salida / f"cifras_{a.orden}.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
