#!/usr/bin/env python
"""Tipo espectral por ajuste de plantillas jóvenes: solo MUSE, solo IR (J/H/K publicados) y combinado.

Solo lectura y fuera de la cadena: G3 tipifica con MUSE y no usa nada de esto. Da las cifras del apéndice B
del paper (tipo por plantillas solo con el IR y con MUSE + IR); está contado en
docs/2026-10-13_tipo_ir_plantillas.md. Vivió en `~/muse_sesion_20261013/tipo_ir/` hasta el 2026-10-13.
`tests/test_tipo_ir_plantillas.py` fija las propiedades y, si los datos están en disco, las cifras.

**Datos.** IR: los espectros del catálogo CDS J/A+A/708/A312 que declara `targets/<slug>.json` ->
`ir_spectra` (sin desenrojecer, λ/Δλ = 500 por píxel; `tipo_ir_indices.load`). Cada espectro es un segmento
con su propia escala: los de un objeto pueden venir escalados a una fotometría ajena (la de la terciaria en
el caso declarado en su perfil), y cada instrumento tiene su calibración. MUSE: el espectro final de D2 del
run del paper con las máscaras y la σ de G3 (`load_fit_inputs`: líneas, O2 A/B, canales malos;
flux_err_total × √(n/n_eff)), degradado a R = 500 (`fit_spectrum_from_inputs`) y en bins de 12 canales
(15 Å) en el rango de G3 (`g3_fit_wave_range_A`).

**Plantillas.** Una por objeto, con los brazos VIS y NIR de X-shooter del MISMO objeto. Manara+2013/2017
(jóvenes, telúrico corregido): VIS de la biblioteca de G3 `templates_young`, NIR del mismo catálogo CDS.
M tardías jóvenes y X-SHYNE L: VIS de su biblioteca de G3, NIR de los ADP de ESO del mismo programa y noche,
sin corrección telúrica (`eso_adp_consulta.json`). El NIR está en `DEFAULT_NIR` (README y SHA256SUMS allí).
TWA 26 sale solo de Manara: su PROVENANCE avisa de que la de las M tardías es la misma observación.
En los ADP de xshoo ≥ 3.2 el bit `BIT_BENIGNO` de QUAL marca ~25 % de los píxeles NIR con flujo normal; se
quitan solo los que son atípicos (`_adp`).

**Modelo.** Plantilla degradada a R = 500, promediada en los píxeles de cada segmento (integral exacta),
enrojecida con A_V (CCM89, R_V = 3.1) y escalada por segmento (escala analítica, positiva; con
`--escala-comun-ir`, una sola para todo el IR, que conserva los colores). A_V libre en la rejilla de G3
(`g3_atmo_av_axis`) o fija a la del run (`h03_av`, la de la Tabla 1) con `--av-fija`.

**Máscaras comunes**, para que todas las plantillas vean los mismos píxeles: en MUSE, además de las de G3,
las bandas de H2O que G3 añade a las bibliotecas sin telúrico (`H2O_MUSE_A`); en el IR, `TELURICO_IR_A`.

**Equilibrio** (`--sin-equilibrio` lo apaga): la σ de cada segmento se multiplica por √(χ²_ν mínimo del
segmento ajustado solo), para que ningún segmento pese más solo por tener la σ mal estimada.

**Tipo.** χ² mínimo por subtipo. El intervalo Δχ² ≤ 1 (tras dividir el χ² total por su χ²_ν mínimo) sale
de un solo subtipo porque domina el desajuste entre plantillas, no el ruido: el error que se cita es el de
`calibra`, que deja fuera cada plantilla M5–L4, la usa como dato (enrojecida con `h03_av`, con la S/N por
píxel de los datos y ruido gaussiano de semilla fija) y la ajusta con las demás; el error es el rms de
(ajustado − verdadero).

uso: python scripts/tipo_ir_plantillas.py ajuste SLUG {muse,ir,combinado} [--av-fija] [--escala-comun-ir]
                                            [--sin-equilibrio] [--sin-telurico-ir] [--sin-h2o-muse]
                                            [--solo-binado] [--bibliotecas ...] [--json SALIDA]
     python scripts/tipo_ir_plantillas.py calibra SLUG {muse,ir,combinado} [--av-fija] [--json SALIDA]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from musepipe.config import load_run_config  # noqa: E402
from musepipe.constants import spt_code, spt_label  # noqa: E402
from musepipe.models.extinction import CCMExtinction  # noqa: E402
from musepipe.models.manifest import library_root  # noqa: E402
from musepipe.models.observed import fit_spectrum_from_inputs, load_fit_inputs  # noqa: E402
from musepipe.models.prep import degrade_variable  # noqa: E402
from musepipe.paths import RunPaths  # noqa: E402
from scripts import tipo_ir_indices as tir  # noqa: E402

DEFAULT_NIR = Path("/mnt/2TB/IR_Data/plantillas_xshooter_nir")
#: resolución común de datos y plantillas (la del muestreo de los espectros IR del catálogo)
R_OBJ = 500.0
#: bandas telúricas en el IR (Å), donde las plantillas de ESO sin corregir no valen
TELURICO_IR_A = [(11000.0, 11600.0), (12600.0, 12750.0), (13300.0, 15000.0),
                 (17800.0, 20800.0), (23600.0, 25000.0)]
#: bandas de H2O que G3 añade en MUSE a las bibliotecas sin telúrico (`models/libraries.py`)
H2O_MUSE_A = [(7130.0, 7360.0), (8100.0, 8400.0), (8900.0, 9100.0), (9250.0, 9700.0)]
#: bit de QUAL que xshoo ≥ 3.2 pone en ~25 % de los píxeles NIR con flujo normal (desvío local 0.10–0.14
#: frente a 0.08–0.12 sin marca); en xshoo 2.x es raro y sí marca atípicos
BIT_BENIGNO = 4194304
#: bibliotecas: etiqueta -> familia de G3 (la de Manara lleva el NIR del CDS; las otras, ADP de ESO)
FAMILIAS = {"manara": "templates_young", "lateM": "templates_young_lateM_xshooter",
            "xshyne": "templates_xshyne_L_xshooter"}
EXT = CCMExtinction(3.1, citation="Cardelli, Clayton & Mathis 1989 (ApJ 345, 245)")


# ------------------------------------------------------------------ utilidades

def bordes(w):
    """Bordes de píxel a partir de los centros (puntos medios; extremos simétricos)."""
    w = np.asarray(w, float)
    m = 0.5 * (w[1:] + w[:-1])
    return np.concatenate([[w[0] - (m[0] - w[0])], m, [w[-1] + (w[-1] - m[-1])]])


def media_en_intervalos(w, f, lo, hi):
    """Media de la densidad f(w) en cada intervalo [lo, hi]: integral de la interpolación lineal, exacta
    entre nodos; en los extremos fraccionarios la acumulada se interpola linealmente (error de segundo orden
    en el paso: ~1e-7 relativo con el muestreo de X-shooter). NaN si el intervalo sale de la cobertura o
    toca un tramo no finito."""
    w, f = np.asarray(w, float), np.asarray(f, float)
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    ok = np.isfinite(f)
    g = np.where(ok, f, 0.0)
    cum = np.concatenate([[0.0], np.cumsum(0.5 * (g[1:] + g[:-1]) * np.diff(w))])
    malo = np.concatenate([[0], np.cumsum(~(ok[1:] & ok[:-1]))])
    out = (np.interp(hi, w, cum) - np.interp(lo, w, cum)) / (hi - lo)
    ilo = np.clip(np.searchsorted(w, lo) - 1, 0, w.size - 1)
    ihi = np.clip(np.searchsorted(w, hi), 0, w.size - 1)
    out[(lo < w[0]) | (hi > w[-1]) | (malo[ihi] - malo[ilo] > 0)] = np.nan
    return out


def degrada(w, f):
    """Plantilla a R = 500: gaussiana de FWHM λ/500 (la R propia, ≥ 4000, se desprecia: < 1 %).
    Los huecos se interpolan para convolucionar y vuelven a NaN donde pesaban más del 20 %."""
    ok = np.isfinite(f)
    g = np.interp(w, w[ok], f[ok])
    out = degrade_variable(w, g, w / R_OBJ)
    hueco = degrade_variable(w, (~ok).astype(float), w / R_OBJ)
    out[hueco > 0.2] = np.nan
    return out


# ------------------------------------------------------------------ plantillas

def _npz(path):
    z = np.load(path, allow_pickle=False)
    return np.asarray(z["wave_A"], float), np.asarray(z["flux"], float), json.loads(str(z["meta_json"]))


def _fit_1d(path):
    """Brazo NIR de Manara (FITS 1D lineal). El WAT1 dice «angstroms» pero la λ está en nm."""
    from astropy.io import fits
    with fits.open(path) as h:
        d = np.asarray(h[0].data, float)
        hd = h[0].header
    w_nm = hd["CRVAL1"] + (np.arange(d.size) + 1 - hd.get("CRPIX1", 1.0)) * hd.get("CDELT1", hd.get("CD1_1"))
    return w_nm * 10.0, d


def limpia_qual(w, f, q, k_mad=5.0):
    """Fuera los píxeles con un bit de QUAL distinto de `BIT_BENIGNO`; los de ese bit, solo si son atípicos
    (> k_mad MAD respecto de la mediana móvil de 41 píxeles de los no marcados)."""
    from scipy.ndimage import median_filter
    q = np.asarray(q).astype(np.int64)
    b22 = (q & BIT_BENIGNO) > 0
    limpio = (q == 0) & np.isfinite(f)
    ref = median_filter(np.interp(w, w[limpio], f[limpio]), size=41, mode="nearest")
    dev = f / ref - 1.0
    mad = 1.4826 * np.nanmedian(np.abs(dev[limpio] - np.nanmedian(dev[limpio])))
    malo = ((q != 0) & ~b22) | (b22 & (np.abs(dev) > k_mad * mad)) | ~np.isfinite(f)
    return np.where(malo, np.nan, f)


def _adp(path):
    from astropy.io import fits
    with fits.open(path) as h:
        t = h[1].data
        w = np.asarray(t["WAVE"][0], float) * 10.0
        f = np.asarray(t["FLUX"][0], float)
        q = np.asarray(t["QUAL"][0])
    return w, limpia_qual(w, f, q)


def plantillas_manara(lib_root, nir=DEFAULT_NIR):
    """(VIS de la biblioteca de G3, NIR del CDS) por objeto, emparejados por el fichero VIS del catálogo."""
    nfile = {}
    for cat in sorted((Path(nir) / "manara_cds").iterdir()):
        for line in (cat / "spectra.dat").read_text().splitlines():
            v = [x for x in line.split() if x.endswith("_V.fit")]
            n = [x for x in line.split() if x.endswith("_N.fit")]
            if v and n:
                nfile[v[0]] = cat / n[0]
    out = []
    for p in sorted((Path(lib_root) / FAMILIAS["manara"]).glob("*.npz")):
        wv, fv, meta = _npz(p)
        wn, fn = _fit_1d(nfile[meta["source_file"]])
        out.append(dict(lib="manara", objeto=meta["object"], spt=meta["spt"], vis=(wv, fv), nir=(wn, fn)))
    return out


def plantillas_eso(lib_root, etiqueta, nir=DEFAULT_NIR):
    """(VIS de la biblioteca de G3, media de los ADP NIR del mismo objeto normalizados en 1.55–1.75 µm)."""
    familia = FAMILIAS[etiqueta]
    prov = json.loads((Path(lib_root) / familia / "PROVENANCE.json").read_text())
    filas = json.loads((Path(nir) / "eso_adp_consulta.json").read_text())
    out = []
    for o in prov["objects"]:
        if etiqueta == "lateM" and o["object"].upper().startswith("TWA26"):
            continue                                    # la misma observación que la de Manara
        ids = sorted({r["nir_dp_id"] for r in filas if r["family"] == familia and r["object"] == o["object"]})
        partes = []
        for i in ids:
            w, f = _adp(Path(nir) / "eso_adp" / f"{i}.fits")
            norma = np.nanmedian(f[(w > 15500) & (w < 17500)])
            if np.isfinite(norma) and norma > 0:
                partes.append(f / norma)
        if not partes:
            print(f"  sin NIR útil: {familia} {o['object']}", file=sys.stderr)
            continue
        wv, fv, _ = _npz(Path(lib_root) / familia / o["file"])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)   # columnas sin ninguna época válida
            media = np.nanmean(np.vstack(partes), axis=0)
        out.append(dict(lib=etiqueta, objeto=o["object"], spt=o["spt"], vis=(wv, fv), nir=(w, media),
                        n_epocas_nir=len(partes)))
    return out


def plantillas(cfg, bibliotecas=("manara", "lateM", "xshyne"), nir=DEFAULT_NIR):
    raiz = library_root(cfg, project_root=ROOT)
    out = []
    for b in bibliotecas:
        out += plantillas_manara(raiz, nir) if b == "manara" else plantillas_eso(raiz, b, nir)
    return out


# ------------------------------------------------------------------ datos

def segmentos_ir(slug, telurico=True):
    ir = tir.perfil(slug)["ir_spectra"]["spectra"]
    segs = []
    for nombre, spec in ir.items():
        w_um, f, e = tir.load(tir.DEFAULT_CDS, spec["file"])
        w = w_um * 1e4
        b = bordes(w)
        bueno = np.isfinite(f) & np.isfinite(e) & (e > 0)
        if telurico:
            for lo, hi in TELURICO_IR_A:
                bueno &= ~((w >= lo) & (w <= hi))
        segs.append(dict(nombre=nombre, brazo="nir", w=w, lo=b[:-1], hi=b[1:], f=f, e=e, bueno=bueno))
    return segs


def segmento_muse(run, h2o=True):
    cfg = load_run_config(run, project_root=ROOT).config
    inputs = load_fit_inputs(cfg, RunPaths.from_project_root(run, ROOT))
    if h2o:
        for lo, hi in H2O_MUSE_A:
            inputs["mask"] = inputs["mask"] | ((inputs["wave"] >= lo) & (inputs["wave"] <= hi))
    lsf = float(cfg["h01_lsf_fwhm_A"])
    fs = fit_spectrum_from_inputs(inputs, n_channels=12, wave_range=cfg["g3_fit_wave_range_A"],
                                  data_kernel_fwhm_A=lambda w: np.sqrt(np.clip((w / R_OBJ) ** 2 - lsf ** 2, 0, None)))
    w = np.asarray(fs.wave_bin, float)
    medio = 12 * float(np.median(np.diff(inputs["wave"]))) / 2
    return dict(nombre="MUSE", brazo="vis", w=w, lo=w - medio, hi=w + medio, f=np.asarray(fs.flux_bin, float),
                e=np.asarray(fs.err_bin, float), bueno=np.isfinite(fs.flux_bin) & (np.asarray(fs.err_bin) > 0))


# ------------------------------------------------------------------ ajuste

def modelos_base(pls, segs, solo_binado_ir=False):
    """{índice de plantilla: [modelo sin enrojecer por segmento]} (NaN fuera de cobertura)."""
    cache = {}
    for i, t in enumerate(pls):
        por_brazo = {}
        for brazo in ("vis", "nir"):
            w, f = t[brazo]
            orden = np.argsort(w)
            w, f = w[orden], f[orden]
            por_brazo[brazo] = (w, f if (brazo == "nir" and solo_binado_ir) else degrada(w, f))
        cache[i] = [media_en_intervalos(*por_brazo[s["brazo"]], s["lo"], s["hi"]) for s in segs]
    return cache


def grupos(segs, escala_comun_ir=False):
    """Índices de segmento que comparten escala: cada uno la suya, o todos los IR juntos."""
    if not escala_comun_ir:
        return [[k] for k in range(len(segs))]
    ir = [k for k, s in enumerate(segs) if s["brazo"] == "nir"]
    return ([ir] if ir else []) + [[k] for k, s in enumerate(segs) if s["brazo"] != "nir"]


def ajusta(segs, bases, av_axis, factores=None, escala_comun_ir=False):
    """χ² con escala positiva por grupo de segmentos y A_V en la rejilla, por plantilla. Mismos píxeles
    para todas (los que todas cubren). Devuelve ({plantilla: χ², A_V, χ² por segmento}, n por segmento)."""
    av_axis = np.asarray(av_axis, float)
    comun = [s["bueno"].copy() for s in segs]
    for m in bases.values():
        for k, mk in enumerate(m):
            comun[k] &= np.isfinite(mk)
    k_av = [np.asarray(EXT.a_lambda_over_av(s["w"]), float) for s in segs]
    fac = factores or [1.0] * len(segs)
    res = {}
    for i, m in bases.items():
        chi = np.zeros(av_axis.size)
        por_seg = np.zeros((av_axis.size, len(segs)))
        valido = np.ones(av_axis.size, bool)
        for grupo in grupos(segs, escala_comun_ir):
            dd, ivv, mms = [], [], []
            for k in grupo:
                c = comun[k]
                dd.append(segs[k]["f"][c])
                ivv.append(1.0 / (segs[k]["e"][c] * fac[k]) ** 2)
                mms.append(m[k][c][None, :] * 10 ** (-0.4 * k_av[k][c][None, :] * av_axis[:, None]))
            d, iv, mm = np.concatenate(dd), np.concatenate(ivv), np.concatenate(mms, axis=1)
            with np.errstate(invalid="ignore", divide="ignore"):
                esc = (mm * d * iv).sum(1) / (mm * mm * iv).sum(1)
            valido &= esc > 0
            r2 = (d[None, :] - esc[:, None] * mm) ** 2 * iv
            chi += r2.sum(1)
            corte = np.cumsum([0] + [x.size for x in dd])
            for j, k in enumerate(grupo):
                por_seg[:, k] = r2[:, corte[j]:corte[j + 1]].sum(1)
        chi[~valido] = np.inf
        j = int(np.argmin(chi))
        res[i] = dict(chi2=float(chi[j]), av=float(av_axis[j]), chi2_seg=por_seg[j].tolist())
    return res, [int(c.sum()) for c in comun]


def factores_equilibrio(segs, bases, av_axis):
    """√(χ²_ν mínimo de cada segmento ajustado solo); None con un solo segmento."""
    if len(segs) < 2:
        return None
    fac = []
    for k in range(len(segs)):
        r, n = ajusta([segs[k]], {i: [m[k]] for i, m in bases.items()}, av_axis)
        fac.append(math.sqrt(min(x["chi2"] for x in r.values()) / max(n[0] - 2, 1)))
    return fac


def tipo(pls, res, n_puntos, n_escalas):
    """Mejor subtipo, intervalo Δχ² ≤ 1 (χ² dividido por su χ²_ν mínimo) y curva Δχ² por subtipo."""
    ndof = sum(n_puntos) - n_escalas - 1
    chi_min = min(r["chi2"] for r in res.values())
    escala = max(chi_min / ndof, 1e-12)
    por_tipo = defaultdict(lambda: (np.inf, None))
    for i, r in res.items():
        c = spt_code(pls[i]["spt"])
        if r["chi2"] < por_tipo[c][0]:
            por_tipo[c] = (r["chi2"], i)
    codigos = sorted(por_tipo)
    mejor = min(codigos, key=lambda c: por_tipo[c][0])
    dentro = [c for c in codigos if (por_tipo[c][0] - por_tipo[mejor][0]) / escala <= 1.0]
    i_mejor = por_tipo[mejor][1]
    return dict(spt=spt_label(mejor), codigo=mejor, intervalo=[spt_label(min(dentro)), spt_label(max(dentro))],
                borde=bool(mejor in (codigos[0], codigos[-1])), chi2nu_min=chi_min / ndof, ndof=ndof,
                escala_chi2=escala,
                mejor_plantilla=f"{pls[i_mejor]['lib']}:{pls[i_mejor]['objeto']} ({pls[i_mejor]['spt']})",
                av=res[i_mejor]["av"],
                curva={spt_label(c): round((por_tipo[c][0] - por_tipo[mejor][0]) / escala, 2) for c in codigos})


def curvas_por_biblioteca(pls, res, escala):
    """Como la Fig. 5 (`companion_type`): por biblioteca, Δχ² de su mejor plantilla de cada subtipo respecto
    de SU mínimo, dividido por `escala` (el χ²_ν mínimo del ajuste conjunto); su mejor subtipo, el
    intervalo Δχ² ≤ 1 y si el mínimo cae en su primer o último subtipo (`borde`)."""
    out = {}
    for lib in dict.fromkeys(t["lib"] for t in pls):
        por_tipo = {}
        for i, r in res.items():
            if pls[i]["lib"] == lib:
                c = spt_code(pls[i]["spt"])
                por_tipo[c] = min(por_tipo.get(c, np.inf), r["chi2"])
        codigos = sorted(por_tipo)
        mejor = min(codigos, key=por_tipo.get)
        d = {c: (por_tipo[c] - por_tipo[mejor]) / escala for c in codigos}
        dentro = [c for c in codigos if d[c] <= 1.0]
        out[lib] = dict(dchi2_by_spt={spt_label(c): round(v, 3) for c, v in d.items()}, spt_best=spt_label(mejor),
                        spt_best_code=mejor, spt_interval=[min(dentro), max(dentro)],
                        edge=bool(mejor in (codigos[0], codigos[-1])))
    return out


def preparar(slug, modo, *, av_fija=False, telurico_ir=True, h2o_muse=True, solo_binado=False,
             bibliotecas=("manara", "lateM", "xshyne")):
    """Plantillas, segmentos, modelos base, rejilla de A_V y A_V de la Tabla 1 de un objeto."""
    run = tir.perfil(slug)["paper"]["run"]
    cfg = load_run_config(run, project_root=ROOT).config
    lo, hi, paso = cfg["g3_atmo_av_axis"]
    av_tabla = float(cfg["h03_av"])
    av_axis = np.array([av_tabla]) if av_fija else np.round(np.arange(lo, hi + paso / 2, paso), 3)
    pls = plantillas(cfg, bibliotecas)
    segs = []
    if modo in ("ir", "combinado"):
        segs += segmentos_ir(slug, telurico=telurico_ir)
    if modo in ("muse", "combinado"):
        segs.append(segmento_muse(run, h2o=h2o_muse))
    return dict(run=run, pls=pls, segs=segs, bases=modelos_base(pls, segs, solo_binado_ir=solo_binado),
                av_axis=av_axis, av_tabla=av_tabla)


def ajuste(slug, modo, *, equilibrio=True, escala_comun_ir=False, **kw):
    p = preparar(slug, modo, **kw)
    fac = factores_equilibrio(p["segs"], p["bases"], p["av_axis"]) if equilibrio else None
    res, n = ajusta(p["segs"], p["bases"], p["av_axis"], fac, escala_comun_ir=escala_comun_ir)
    t = tipo(p["pls"], res, n, len(grupos(p["segs"], escala_comun_ir)))
    bibliotecas = curvas_por_biblioteca(p["pls"], res, t["escala_chi2"])
    orden = sorted(res, key=lambda i: res[i]["chi2"])[:6]
    return dict(slug=slug, run=p["run"], modo=modo, equilibrio=fac is not None, escala_comun_ir=escala_comun_ir,
                opciones=kw, n_plantillas=len(p["pls"]),
                segmentos=[dict(nombre=s["nombre"], n=nk, factor_sigma=(fac[k] if fac else 1.0))
                           for k, (s, nk) in enumerate(zip(p["segs"], n))],
                tipo=t, bibliotecas=bibliotecas,
                mejores=[dict(plantilla=f"{p['pls'][i]['lib']}:{p['pls'][i]['objeto']}", spt=p["pls"][i]["spt"],
                              chi2=round(res[i]["chi2"], 1), av=res[i]["av"]) for i in orden])


def falsos(segs, base_t, av, rng):
    """Segmentos con el modelo de una plantilla enrojecido con `av`, la S/N por píxel de los datos y ruido."""
    out = []
    for s, m in zip(segs, base_t):
        f = m * 10 ** (-0.4 * np.asarray(EXT.a_lambda_over_av(s["w"]), float) * av)
        snr = np.abs(s["f"]) / s["e"]
        with np.errstate(invalid="ignore", divide="ignore"):
            e = np.abs(f) / np.where(snr > 0, snr, np.nan)
        out.append({**s, "f": f + rng.normal(0, np.nan_to_num(e)), "e": e,
                    "bueno": s["bueno"] & np.isfinite(f) & np.isfinite(e) & (e > 0)})
    return out


def calibra(slug, modo, *, semilla=0, escala_comun_ir=False, **kw):
    """Dejando una fuera: cada plantilla M5–L4 hace de dato y se ajusta con las demás."""
    p = preparar(slug, modo, **kw)
    rng = np.random.default_rng(semilla)
    filas = []
    for i, t in enumerate(p["pls"]):
        verdad = spt_code(t["spt"])
        if not 5.0 <= verdad <= 14.0:
            continue
        datos = falsos(p["segs"], p["bases"][i], p["av_tabla"], rng)
        resto = {j: m for j, m in p["bases"].items() if j != i}
        fac = factores_equilibrio(datos, resto, p["av_axis"])
        res, _ = ajusta(datos, resto, p["av_axis"], fac, escala_comun_ir=escala_comun_ir)
        j = min(res, key=lambda k: res[k]["chi2"])
        filas.append(dict(plantilla=f"{t['lib']}:{t['objeto']}", verdad=t["spt"], ajustado=p["pls"][j]["spt"],
                          delta=spt_code(p["pls"][j]["spt"]) - verdad, av=res[j]["av"]))
    d = np.array([f["delta"] for f in filas])
    return dict(slug=slug, modo=modo, opciones=kw, n=int(d.size), mediana=float(np.median(d)),
                rms=float(np.sqrt(np.mean(d ** 2))), p16_p84=[float(np.percentile(d, 16)), float(np.percentile(d, 84))],
                frac_1=float(np.mean(np.abs(d) <= 1.0)), filas=filas)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for nombre in ("ajuste", "calibra"):
        q = sub.add_parser(nombre)
        q.add_argument("slug")
        q.add_argument("modo", choices=("muse", "ir", "combinado"))
        q.add_argument("--av-fija", action="store_true", help="A_V = h03_av del run (Tabla 1) en vez de libre")
        q.add_argument("--escala-comun-ir", action="store_true",
                       help="una sola escala para todo el IR (solo si su fotometría es la del compañero)")
        q.add_argument("--json", type=Path)
        if nombre == "ajuste":
            q.add_argument("--sin-equilibrio", action="store_true")
            q.add_argument("--sin-telurico-ir", action="store_true")
            q.add_argument("--sin-h2o-muse", action="store_true")
            q.add_argument("--solo-binado", action="store_true", help="NIR de las plantillas sin convolucionar")
            q.add_argument("--bibliotecas", nargs="+", default=list(FAMILIAS), choices=list(FAMILIAS))
        else:
            q.add_argument("--semilla", type=int, default=0)
    a = ap.parse_args(argv)
    if a.cmd == "ajuste":
        out = ajuste(a.slug, a.modo, equilibrio=not a.sin_equilibrio, escala_comun_ir=a.escala_comun_ir,
                     av_fija=a.av_fija, telurico_ir=not a.sin_telurico_ir, h2o_muse=not a.sin_h2o_muse,
                     solo_binado=a.solo_binado, bibliotecas=tuple(a.bibliotecas))
        t = out["tipo"]
        print(f"{a.slug} {a.modo}: {t['spt']} [{t['intervalo'][0]}–{t['intervalo'][1]}]"
              f"{' BORDE' if t['borde'] else ''}  mejor {t['mejor_plantilla']} A_V {t['av']}  "
              f"χ²_ν {t['chi2nu_min']:.2f} (ν {t['ndof']})")
        print("  segmentos:", ", ".join(f"{s['nombre']} n={s['n']} ×{s['factor_sigma']:.2f}" for s in out["segmentos"]))
        print("  Δχ² por subtipo:", t["curva"])
    else:
        out = calibra(a.slug, a.modo, semilla=a.semilla, escala_comun_ir=a.escala_comun_ir, av_fija=a.av_fija)
        print(f"{a.slug} {a.modo}{' A_V fija' if a.av_fija else ''}: n {out['n']}, Δ mediana {out['mediana']:+.1f}, "
              f"rms {out['rms']:.2f}, p16–p84 {out['p16_p84']}, |Δ| ≤ 1 en {out['frac_1']:.0%}")
    if a.json:
        a.json.write_text(json.dumps(out, indent=1, ensure_ascii=False, default=float))
    return 0


if __name__ == "__main__":
    sys.exit(main())
