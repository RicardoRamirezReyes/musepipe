#!/usr/bin/env python
"""Ajuste conjunto MUSE + J/H/K con plantillas jóvenes: tipo, A_V, R_V, exceso en K (r_K, T_d) y escalas con prior.

Solo lectura y fuera de la cadena; explora qué combinación de parámetros describe mejor cada compañero
(pedido del autor 2026-10-13: «veamos cuáles son los parámetros disponibles ... para encontrar el mejor fit»).
Usa las plantillas, los segmentos y los modelos base de `tipo_ir_plantillas` (R = 500, máscaras comunes).

**Modelo** del segmento k para una plantilla (m) y parámetros no lineales θ = (A_V, R_V, r_K, T_d):
    F_k = P_k(λ) · e(λ; A_V, R_V) · [m_k(λ) + r_K · m(2.2 µm) · B_λ(T_d)/B_{2.2 µm}(T_d)],
con P_k = Σ_j a_kj L_j(x) (Legendre en el segmento): orden `ORDEN_MUSE` en MUSE (absorbe la pendiente y los
sistemáticos cromáticos lisos: el tipo lo dan las bandas, como en `--polinomio`) y `--orden-ir` en cada banda IR
(0 por defecto: la forma de la banda informa). Los a_kj son lineales y se resuelven exactos para cada θ.

**Escalas con prior.** El término constante a_k0 de cada segmento se ata a una escala global g (también
lineal) con un prior gaussiano de anchura relativa σ_k: `SIGMA_IR` en las bandas que `targets/<slug>.json` ->
`ir_spectra.colour_excess` declara válidas (escaladas a la fotometría del compañero; K reescalada si se
declara), `SIGMA_MUSE` en MUSE (calibración absoluta de D2: apcorr, fondo, épocas), y libre (sin prior) en las
bandas no declaradas. Así los colores entran con el peso que permite su calibración. El prior se linealiza con
la g de una primera pasada sin prior. Todos los flujos en erg s⁻¹ cm⁻² Å⁻¹.

**Rejilla** (`REJILLA`): A_V 0–8 (0.2), R_V {3.1, 4.0, 5.5}, r_K 0–1.5 (0.1), T_d {1000, 1500} K; plantillas
M5–L4. El χ² incluye el del prior. Perfiles: mínimo sobre los demás parámetros; intervalos con Δχ² ≤ 1 tras
dividir por el χ²_ν mínimo. El error que se cita es el de `--calibra` (dejando una plantilla fuera, con θ
inyectado conocido).

**Con BT-Settl** (`--btsettl DIR`): los modelos CIFIST2011 de SVO (npz de 4000–25500 Å, T_eff 2000–3200 K,
log g 2.5–4.5) en lugar de las plantillas, de modo que T_eff y log g son parámetros. `--calibra` con BT-Settl
ajusta las plantillas de tipo conocido (como datos, con la S/N del compañero y A_V = 2) con la rejilla de
modelos: da la T_eff y el log g que BT-Settl asigna a cada tipo y el sesgo en A_V.

uso: python scripts/ajuste_conjunto.py SLUG [--orden-ir 0] [--bibliotecas ...] [--sin-h2o-muse]
                                       [--btsettl DIR] [--calibra] [--json SALIDA]
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from musepipe.constants import spt_code, spt_label  # noqa: E402
from musepipe.models.extinction import CCMExtinction  # noqa: E402
from scripts import tipo_ir_indices as tir  # noqa: E402
from scripts import tipo_ir_plantillas as tp  # noqa: E402

ORDEN_MUSE = 2
SIGMA_IR = 0.05       # fotometría J/H/K del compañero (3–7 %) y escalado de los espectros del catálogo
SIGMA_MUSE = 0.20     # calibración absoluta de MUSE entre métodos y radios de fondo, y épocas distintas
REJILLA = dict(av=np.round(np.arange(0.0, 8.0001, 0.2), 2), rv=(3.1, 4.0, 5.5),
               rk=np.round(np.arange(0.0, 1.5001, 0.1), 2), td=(1000.0, 1500.0))
W_PER_M2_UM_A_CGS = 0.1
LAM_K = 22000.0


def _planck(w, T):
    return 1.0 / (w ** 5 * np.expm1(1.4388e8 / (w * T)))


def segmentos_cgs(slug, p):
    """Los segmentos de `p` en erg/s/cm²/Å, con K reescalada y con la σ del prior de su escala (None = libre)."""
    from astropy.io import fits

    from musepipe.config import load_run_config
    from musepipe.io import resolve_flux_unit
    ir = tir.perfil(slug)["ir_spectra"]
    ce = ir.get("colour_excess") or {}
    validas = set(ce.get("bands", []))
    k_fac = 1.0
    if ce.get("k_rescale_mag"):
        k_fac = 10 ** (-0.4 * (ce["k_rescale_mag"]["adopted"] - ce["k_rescale_mag"]["catalogue"]))
    cfg = load_run_config(p["run"], project_root=ROOT).config
    with fits.open(ROOT / "runs" / p["run"] / "stages" / "spec_final_object.fits") as h:
        escala_muse, _ = resolve_flux_unit(cfg, bunit=h["SPECTRUM"].header.get("BUNIT"))
    out = []
    for s in p["segs"]:
        s = dict(s)
        if s["brazo"] == "vis":
            fac, sigma = escala_muse, SIGMA_MUSE
        else:
            banda = ir["spectra"][s["nombre"]]["band"]
            fac = W_PER_M2_UM_A_CGS * (k_fac if banda == "K" else 1.0)
            sigma = SIGMA_IR if banda in validas else None
        s["f"], s["e"], s["sigma_escala"] = s["f"] * fac, s["e"] * fac, sigma
        out.append(s)
    return out


class Conjunto:
    """Precalcula lo que no depende de la plantilla ni de θ (píxeles comunes, Legendre, A_λ/A_V por R_V)."""

    def __init__(self, segs, bases, orden_ir=0, factores=None):
        self.segs = segs
        self.comun = [s["bueno"].copy() for s in segs]
        for m in bases.values():
            for k, mk in enumerate(m):
                self.comun[k] &= np.isfinite(mk)
        fac = factores or [1.0] * len(segs)
        self.ordenes = [ORDEN_MUSE if s["brazo"] == "vis" else int(orden_ir) for s in segs]
        self.w = [s["w"][c] for s, c in zip(segs, self.comun)]
        self.d = [s["f"][c] for s, c in zip(segs, self.comun)]
        self.sw = [1.0 / (s["e"][c] * fk) for s, c, fk in zip(segs, self.comun, fac)]
        self.leg = []
        for w, o in zip(self.w, self.ordenes):
            x = 2.0 * (w - w.min()) / (w.max() - w.min()) - 1.0
            self.leg.append(np.polynomial.legendre.legvander(x, o))
        self.k_av = {rv: [np.asarray(CCMExtinction(rv, citation="CCM89").a_lambda_over_av(w), float) for w in self.w]
                     for rv in REJILLA["rv"]}
        self.bb = {td: [_planck(w, td) / _planck(LAM_K, td) if s["brazo"] == "nir" else np.zeros(w.size)
                        for w, s in zip(self.w, segs)] for td in REJILLA["td"]}
        self.k_ref = next(k for k, s in enumerate(segs) if s["brazo"] == "nir" and s["w"][0] <= LAM_K <= s["w"][-1])
        self.n = sum(int(c.sum()) for c in self.comun)
        self.n_lineal = sum(o + 1 for o in self.ordenes) + 1

    def chi2(self, m, av, rv, rk, td):
        """χ² (datos + prior) y g para una plantilla (`m`: modelos base por segmento) y un θ."""
        mref = float(np.interp(LAM_K, self.segs[self.k_ref]["w"], m[self.k_ref]))
        cols, filas_d = [], []
        ncol = self.n_lineal
        bloques, inicio = [], 0
        for k in range(len(self.segs)):
            c = self.comun[k]
            t = (m[k][c] + rk * mref * self.bb[td][k]) * 10 ** (-0.4 * self.k_av[rv][k] * av)
            a = (t[:, None] * self.leg[k]) * self.sw[k][:, None]
            bloques.append((inicio, a, self.d[k] * self.sw[k]))
            inicio += a.shape[1]
        n_filas = sum(b[1].shape[0] for b in bloques)
        A = np.zeros((n_filas, ncol))
        y = np.zeros(n_filas)
        f0 = 0
        for (c0, a, yy) in bloques:
            A[f0:f0 + a.shape[0], c0:c0 + a.shape[1]] = a
            y[f0:f0 + a.shape[0]] = yy
            f0 += a.shape[0]
        sol, *_ = np.linalg.lstsq(A[:, :-1], y, rcond=None)
        c0s = [b[0] for b in bloques]
        priores = [k for k, s in enumerate(self.segs) if s["sigma_escala"] is not None]
        if not priores:
            r = y - A[:, :-1] @ sol
            return float(r @ r), np.nan, sol
        g0 = float(np.median([sol[c0s[k]] for k in priores]))
        if not np.isfinite(g0) or g0 <= 0:
            return np.inf, np.nan, sol
        P = np.zeros((len(priores), ncol))
        for j, k in enumerate(priores):
            wk = 1.0 / (self.segs[k]["sigma_escala"] * g0)
            P[j, c0s[k]] = wk
            P[j, -1] = -wk
        AA = np.vstack([A, P])
        yy = np.concatenate([y, np.zeros(len(priores))])
        sol, *_ = np.linalg.lstsq(AA, yy, rcond=None)
        if sol[-1] <= 0 or any(sol[c0s[k]] <= 0 for k in range(len(self.segs))):
            return np.inf, float(sol[-1]), sol
        r = yy - AA @ sol
        return float(r @ r), float(sol[-1]), sol


def explora(conj, bases, indices, rejilla=REJILLA):
    """{plantilla: rejilla de χ² [av, rv, rk, td]} para las plantillas de `indices`."""
    out = {}
    for i in indices:
        m = bases[i]
        cub = np.full((len(rejilla["av"]), len(rejilla["rv"]), len(rejilla["rk"]), len(rejilla["td"])), np.inf)
        for a, av in enumerate(rejilla["av"]):
            for b, rv in enumerate(rejilla["rv"]):
                for c, rk in enumerate(rejilla["rk"]):
                    for d, td in enumerate(rejilla["td"]):
                        if rk == 0 and d > 0:
                            cub[a, b, c, d] = cub[a, b, c, 0]   # sin exceso, T_d no importa
                            continue
                        cub[a, b, c, d] = conj.chi2(m, av, rv, rk, td)[0]
        out[i] = cub
    return out


def resumen(pls, cubos, ndof, rejilla=REJILLA):
    chi_min = min(float(np.min(c)) for c in cubos.values())
    esc = max(chi_min / ndof, 1e-12)
    mejor_i = min(cubos, key=lambda i: float(np.min(cubos[i])))
    idx = np.unravel_index(int(np.argmin(cubos[mejor_i])), cubos[mejor_i].shape)
    por_tipo = {}
    for i, c in cubos.items():
        k = spt_code(pls[i]["spt"])
        por_tipo[k] = min(por_tipo.get(k, np.inf), float(np.min(c)))
    curva_tipo = {spt_label(k): round((v - chi_min) / esc, 1) for k, v in sorted(por_tipo.items())}
    todos = np.stack([cubos[i] for i in cubos])               # [plantilla, av, rv, rk, td]
    perfiles = {}
    for eje, nombre in ((1, "av"), (2, "rv"), (3, "rk"), (4, "td")):
        otros = tuple(j for j in range(5) if j != eje)
        prof = (np.min(todos, axis=otros) - chi_min) / esc
        valores = np.asarray(rejilla[nombre], float)
        dentro = valores[prof <= 1.0]
        perfiles[nombre] = dict(valores=valores.tolist(), dchi2=[round(float(x), 2) for x in prof],
                                mejor=float(valores[int(np.argmin(prof))]),
                                intervalo=[float(dentro.min()), float(dentro.max())])
    t = pls[mejor_i]
    return dict(mejor=dict(plantilla=f"{t['lib']}:{t['objeto']}", spt=t["spt"],
                           av=float(rejilla["av"][idx[0]]), rv=float(rejilla["rv"][idx[1]]),
                           rk=float(rejilla["rk"][idx[2]]), td=float(rejilla["td"][idx[3]])),
                chi2nu_min=chi_min / ndof, ndof=ndof, curva_tipo=curva_tipo, perfiles=perfiles)


def modelos_btsettl(dir_npz, teff=(2000.0, 3200.0), logg=(2.5, 4.5), paso_A=1.0):
    """Los modelos BT-Settl como «plantillas» (`vis`/`nir` como tipo_ir_plantillas), rebinados a `paso_A` antes de
    degradar (a R = 500 la FWHM es ≥ 13 Å, así que 1 Å no pierde nada y abarata la convolución)."""
    out = []
    for f in sorted(Path(dir_npz).glob("teff*_logg*.npz")):
        w, fl, meta = tp._npz(f)
        if not (teff[0] <= meta["teff_k"] <= teff[1] and logg[0] <= meta["logg"] <= logg[1]):
            continue
        g = np.arange(w[0] + paso_A, w[-1] - paso_A, paso_A)
        fg = tp.media_en_intervalos(w, fl, g - paso_A / 2, g + paso_A / 2)
        out.append(dict(lib="btsettl", objeto=f"T{int(meta['teff_k'])}_g{meta['logg']:.1f}", spt="M9",
                        teff=float(meta["teff_k"]), logg=float(meta["logg"]),
                        vis=(g[g < 10500], fg[g < 10500]), nir=(g[g > 9500], fg[g > 9500])))
    if not out:
        raise SystemExit(f"no hay modelos BT-Settl en {dir_npz}")
    return out


def preparar(slug, orden_ir=0, bibliotecas=("manara", "lateM", "xshyne"), h2o_muse=True, tipos=(5.0, 14.0),
             btsettl=None):
    p = tp.preparar(slug, "combinado", av_fija=True, bibliotecas=bibliotecas, h2o_muse=h2o_muse)
    segs = segmentos_cgs(slug, p)
    if btsettl:
        p["pls"] = modelos_btsettl(btsettl)
        p["bases"] = tp.modelos_base(p["pls"], segs)
    fac = tp.factores_equilibrio(segs, p["bases"], [p["av_tabla"]])
    conj = Conjunto(segs, p["bases"], orden_ir, fac)
    indices = (list(range(len(p["pls"]))) if btsettl else
               [i for i, t in enumerate(p["pls"]) if tipos[0] <= spt_code(t["spt"]) <= tipos[1]])
    return p, conj, indices


def resumen_btsettl(pls, cubos, ndof, rejilla=REJILLA):
    """Mejor (T_eff, log g, A_V, R_V, r_K, T_d), perfiles de T_eff y log g y el mapa Δχ²(T_eff, log g)."""
    chi_min = min(float(np.min(c)) for c in cubos.values())
    esc = max(chi_min / ndof, 1e-12)
    mejor_i = min(cubos, key=lambda i: float(np.min(cubos[i])))
    idx = np.unravel_index(int(np.argmin(cubos[mejor_i])), cubos[mejor_i].shape)
    mapa = {}
    for i, c in cubos.items():
        mapa[(pls[i]["teff"], pls[i]["logg"])] = (float(np.min(c)) - chi_min) / esc
    teffs = sorted({k[0] for k in mapa}); loggs = sorted({k[1] for k in mapa})
    perf = lambda eje, vals: {v: min(d for k, d in mapa.items() if k[eje] == v) for v in vals}
    pt, pg = perf(0, teffs), perf(1, loggs)
    intervalo = lambda d: [min(v for v, x in d.items() if x <= 1.0), max(v for v, x in d.items() if x <= 1.0)]
    todos = np.stack([cubos[i] for i in cubos])
    perfiles = {}
    for eje, nombre in ((1, "av"), (2, "rv"), (3, "rk"), (4, "td")):
        prof = (np.min(todos, axis=tuple(j for j in range(5) if j != eje)) - chi_min) / esc
        valores = np.asarray(rejilla[nombre], float)
        perfiles[nombre] = dict(mejor=float(valores[int(np.argmin(prof))]),
                                intervalo=[float(valores[prof <= 1.0].min()), float(valores[prof <= 1.0].max())])
    t = pls[mejor_i]
    return dict(mejor=dict(teff=t["teff"], logg=t["logg"], av=float(rejilla["av"][idx[0]]),
                           rv=float(rejilla["rv"][idx[1]]), rk=float(rejilla["rk"][idx[2]]),
                           td=float(rejilla["td"][idx[3]])),
                chi2nu_min=chi_min / ndof, ndof=ndof,
                perfil_teff={int(k): round(v, 1) for k, v in pt.items()}, teff_intervalo=intervalo(pt),
                perfil_logg={k: round(v, 1) for k, v in pg.items()}, logg_intervalo=intervalo(pg),
                perfiles=perfiles,
                mapa=[dict(teff=k[0], logg=k[1], dchi2=round(v, 1)) for k, v in sorted(mapa.items())])


def ajusta_btsettl(slug, dir_npz, orden_ir=0, h2o_muse=True):
    p, conj, indices = preparar(slug, orden_ir, ("manara",), h2o_muse, btsettl=dir_npz)
    cubos = explora(conj, p["bases"], indices)
    r = resumen_btsettl(p["pls"], cubos, conj.n - conj.n_lineal - 6)
    r.update(slug=slug, orden_ir=orden_ir, h2o_muse=h2o_muse, n_modelos=len(indices))
    return r


def calibra_btsettl(slug, dir_npz, av_inyectada=2.0, orden_ir=0, semilla=0):
    """Plantillas de tipo conocido (Manara M6.5–M9.5 y X-SHYNE L0–L1) como datos (S/N del compañero, A_V inyectada)
    ajustadas con BT-Settl: la T_eff, el log g y la A_V que les asigna el modelo."""
    rej = dict(av=np.round(np.arange(0.0, 6.0001, 0.2), 2), rv=(3.1,), rk=(0.0,), td=(1500.0,))
    p, conj, _ = preparar(slug, orden_ir)
    bt = modelos_btsettl(dir_npz)
    bases_bt = tp.modelos_base(bt, conj.segs)
    rng = np.random.default_rng(semilla)
    filas = []
    for i, t in enumerate(p["pls"]):
        if t["lib"] not in ("manara", "xshyne") or not 6.5 <= spt_code(t["spt"]) <= 11.0:
            continue
        m = p["bases"][i]
        falsos = []
        for k, s in enumerate(conj.segs):
            ext = 10 ** (-0.4 * np.asarray(CCMExtinction(3.1, citation="CCM89").a_lambda_over_av(s["w"]), float) * av_inyectada)
            f = m[k] * ext * float(np.nanmedian(s["f"][s["bueno"]]) / np.nanmedian((m[k] * ext)[s["bueno"]]))
            snr = np.abs(s["f"]) / s["e"]
            with np.errstate(invalid="ignore", divide="ignore"):
                e = np.abs(f) / np.where(snr > 0, snr, np.nan)
            falsos.append({**s, "f": f + rng.normal(0, np.nan_to_num(e)), "e": e,
                           "bueno": s["bueno"] & np.isfinite(f) & np.isfinite(e) & (e > 0)})
        cf = Conjunto(falsos, bases_bt, orden_ir)
        cubos = explora(cf, bases_bt, list(bases_bt), rej)
        j = min(cubos, key=lambda q: float(np.min(cubos[q])))
        idx = np.unravel_index(int(np.argmin(cubos[j])), cubos[j].shape)
        filas.append(dict(plantilla=f"{t['lib']}:{t['objeto']}", spt=t["spt"], teff=bt[j]["teff"], logg=bt[j]["logg"],
                          av=float(rej["av"][idx[0]])))
    dav = np.array([f["av"] - av_inyectada for f in filas])
    return dict(av_inyectada=av_inyectada, n=len(filas), sesgo_av=float(np.median(dav)),
                rms_av=float(np.sqrt(np.mean(dav ** 2))), filas=filas)


def ajusta(slug, orden_ir=0, bibliotecas=("manara", "lateM", "xshyne"), h2o_muse=True):
    p, conj, indices = preparar(slug, orden_ir, bibliotecas, h2o_muse)
    cubos = explora(conj, p["bases"], indices)
    ndof = conj.n - conj.n_lineal - 4
    r = resumen(p["pls"], cubos, ndof)
    r.update(slug=slug, orden_ir=orden_ir, bibliotecas=list(bibliotecas), h2o_muse=h2o_muse,
             sigma_ir=SIGMA_IR, sigma_muse=SIGMA_MUSE,
             segmentos=[dict(nombre=s["nombre"], sigma_escala=s["sigma_escala"]) for s in conj.segs])
    return r


def calibra(slug, theta=(2.0, 3.1, 0.3, 1500.0), orden_ir=0, semilla=0, rejilla=None):
    """Cada plantilla M6.5–L1 hace de dato con θ = (A_V, R_V, r_K, T_d) conocido, la S/N del dato real y ruido;
    se ajusta con las demás. Devuelve los errores en tipo, A_V y r_K."""
    rej = rejilla or dict(av=np.round(np.arange(0.0, 6.0001, 0.2), 2), rv=(3.1,), rk=np.round(np.arange(0.0, 1.0001, 0.1), 2),
                          td=(1500.0,))
    p, conj, indices = preparar(slug, orden_ir)
    rng = np.random.default_rng(semilla)
    av, rv, rk, td = theta
    filas = []
    for i in indices:
        if not 6.5 <= spt_code(p["pls"][i]["spt"]) <= 11.0:
            continue
        m = p["bases"][i]
        mref = float(np.interp(LAM_K, conj.segs[conj.k_ref]["w"], m[conj.k_ref]))
        falsos = []
        for k, s in enumerate(conj.segs):
            ext = 10 ** (-0.4 * np.asarray(CCMExtinction(rv, citation="CCM89").a_lambda_over_av(s["w"]), float) * av)
            bb = _planck(s["w"], td) / _planck(LAM_K, td) if s["brazo"] == "nir" else 0.0
            f = (m[k] + rk * mref * bb) * ext * float(np.nanmedian(s["f"][s["bueno"]]) /
                                                       np.nanmedian((m[k] * ext)[s["bueno"]]))
            snr = np.abs(s["f"]) / s["e"]
            with np.errstate(invalid="ignore", divide="ignore"):
                e = np.abs(f) / np.where(snr > 0, snr, np.nan)
            falsos.append({**s, "f": f + rng.normal(0, np.nan_to_num(e)), "e": e,
                           "bueno": s["bueno"] & np.isfinite(f) & np.isfinite(e) & (e > 0)})
        resto = {j: p["bases"][j] for j in indices if j != i}
        cf = Conjunto(falsos, resto, orden_ir)
        cubos = explora(cf, resto, list(resto), rej)
        j = min(cubos, key=lambda q: float(np.min(cubos[q])))
        idx = np.unravel_index(int(np.argmin(cubos[j])), cubos[j].shape)
        filas.append(dict(dato=f"{p['pls'][i]['lib']}:{p['pls'][i]['objeto']}", verdad=p["pls"][i]["spt"],
                          ajustado=p["pls"][j]["spt"], d_tipo=spt_code(p["pls"][j]["spt"]) - spt_code(p["pls"][i]["spt"]),
                          av=float(rej["av"][idx[0]]), rk=float(rej["rk"][idx[2]])))
    dt = np.array([f["d_tipo"] for f in filas])
    dav = np.array([f["av"] - av for f in filas])
    drk = np.array([f["rk"] - rk for f in filas])
    return dict(theta=list(theta), n=len(filas), rms_tipo=float(np.sqrt(np.mean(dt ** 2))),
                rms_av=float(np.sqrt(np.mean(dav ** 2))), sesgo_av=float(np.median(dav)),
                rms_rk=float(np.sqrt(np.mean(drk ** 2))), sesgo_rk=float(np.median(drk)), filas=filas)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("slug")
    ap.add_argument("--orden-ir", type=int, default=0)
    ap.add_argument("--bibliotecas", nargs="+", default=list(tp.FAMILIAS), choices=list(tp.FAMILIAS))
    ap.add_argument("--sin-h2o-muse", action="store_true", help="sin la máscara de H2O de MUSE (entra el Na I 8183/8195)")
    ap.add_argument("--calibra", action="store_true")
    ap.add_argument("--btsettl", type=Path, help="directorio con los npz de BT-Settl (T_eff y log g como parámetros)")
    ap.add_argument("--json", type=Path)
    a = ap.parse_args(argv)
    warnings.simplefilter("ignore", RuntimeWarning)
    if a.btsettl and a.calibra:
        out = calibra_btsettl(a.slug, a.btsettl, orden_ir=a.orden_ir)
        print(f"{a.slug} plantillas con BT-Settl (A_V inyectada {out['av_inyectada']}): sesgo A_V {out['sesgo_av']:+.2f}, "
              f"rms {out['rms_av']:.2f}")
        for f in out["filas"]:
            print(f"   {f['plantilla']:<34s} {f['spt']:<5s} T_eff {f['teff']:.0f}  log g {f['logg']}  A_V {f['av']}")
    elif a.btsettl:
        out = ajusta_btsettl(a.slug, a.btsettl, a.orden_ir, not a.sin_h2o_muse)
        m = out["mejor"]
        print(f"{a.slug} BT-Settl: T_eff {m['teff']:.0f} K  log g {m['logg']}  A_V {m['av']}  R_V {m['rv']}  r_K {m['rk']}  "
              f"T_d {m['td']:.0f}  χ²_ν {out['chi2nu_min']:.2f} (ν {out['ndof']})")
        print(f"  T_eff: Δχ² ≤ 1 en {out['teff_intervalo']}; perfil {out['perfil_teff']}")
        print(f"  log g: Δχ² ≤ 1 en {out['logg_intervalo']}; perfil {out['perfil_logg']}")
        for n, pr in out["perfiles"].items():
            print(f"  {n}: mejor {pr['mejor']}, Δχ² ≤ 1 en {pr['intervalo']}")
    elif a.calibra:
        out = calibra(a.slug, orden_ir=a.orden_ir)
        print(f"{a.slug} calibración (θ {out['theta']}, n {out['n']}): rms tipo {out['rms_tipo']:.2f}, "
              f"A_V rms {out['rms_av']:.2f} (sesgo {out['sesgo_av']:+.2f}), r_K rms {out['rms_rk']:.2f} (sesgo {out['sesgo_rk']:+.2f})")
    else:
        out = ajusta(a.slug, a.orden_ir, tuple(a.bibliotecas), not a.sin_h2o_muse)
        m = out["mejor"]
        print(f"{a.slug}: {m['plantilla']} {m['spt']}  A_V {m['av']}  R_V {m['rv']}  r_K {m['rk']}  T_d {m['td']:.0f}  "
              f"χ²_ν {out['chi2nu_min']:.2f} (ν {out['ndof']})")
        print("  Δχ² por tipo:", {k: v for k, v in out["curva_tipo"].items() if k >= "M6" or k[0] == "L"})
        for n, pr in out["perfiles"].items():
            print(f"  {n}: mejor {pr['mejor']}, Δχ² ≤ 1 en {pr['intervalo']}")
    if a.json:
        a.json.write_text(json.dumps(out, indent=1, ensure_ascii=False, default=float))
    return 0


if __name__ == "__main__":
    sys.exit(main())
