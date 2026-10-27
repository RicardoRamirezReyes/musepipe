#!/usr/bin/env python
"""Veiling de los compañeros: en el óptico (MUSE) y en K (profundidad de la cabeza de banda del CO).

Solo lectura y fuera de la cadena: G3 ya ajusta una variante con veiling, pero solo guarda el cambio de tipo.
Da las cifras del veiling del apéndice B (Tabla B.3) y de la sección de sistemáticos del paper; está contado en
docs/2026-10-13_veiling.md. `tests/test_veiling.py` fija las propiedades y, con los datos en disco, las cifras.
Veiling r = flujo del exceso / flujo de la fotosfera a λ_ref.

**Óptico** (`optico`). Modelo de MUSE a R = 500 (`tipo_ir_plantillas`): s · e(A_V) · [m + r · m(7500 Å) · (λ/7500)^α],
con α en `ALFAS` (la misma forma que la variante de G3), e el enrojecimiento CCM89 (el exceso también está
detrás del polvo) y s analítica ≥ 0. El χ² de r en `R_EJE` tiene el mínimo en el borde r = 0 y lo domina el
desajuste entre plantillas, así que la cota es la de `calibra`: cada plantilla a ±1.5 subtipos del tipo hace de
dato, con la S/N del dato real, y se ajusta con las demás; el p95 del r espurio es el suelo. Con `--inyecta`,
los datos falsos llevan un veiling conocido (para ver que se recupera).

**En K, ajuste de banda completa** (`k`): el mismo modelo con polvo caliente B_λ(T_d)/B_λ(T_d, 2.2 µm). Con una
escala por espectro, r_K queda degenerado con la A_V y el tipo: corrige la PENDIENTE de cada banda, no la
profundidad de las bandas. Se guarda para documentar esa degeneración.

**En K, profundidad del CO** (`co`), insensible a la pendiente: D = 1 − ⟨F⟩(banda)/⟨F⟩(continuo), con las
ventanas de `VENTANAS_CO`. La extinción no la cambia (A_λ difiere un 0.4 % por magnitud entre banda y
continuo) y un exceso la diluye: D_obs = D_fot/(1 + r_K), así que r_K = D_fot/D_obs − 1, con D_fot de las
plantillas (degradadas a R = 500) a ±1 subtipo del tipo de los índices de agua.

**Esperado** (`esperado`): el veiling que produciría la L_acc con un continuo de cuerpo negro de T_acc sobre la
fotosfera de una plantilla: r(λ) = [L_acc πB_λ(T_acc)/(σT_acc⁴)] / [L_bol F_λ,fot(λ)/F_bol,fot], con F_bol,fot =
∫ de la plantilla (0.53–2.48 µm) / 0.85.

uso: python scripts/veiling.py optico SLUG [--av-fija] [--calibra] [--inyecta R] [--json SALIDA]
     python scripts/veiling.py k SLUG [--av-fija] [--escala-comun-ir]
     python scripts/veiling.py co SLUG [--tipo M9] [--variante nominal]
     python scripts/veiling.py esperado --lacc L --lbol L --plantilla OBJETO --slug SLUG
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

from musepipe.constants import spt_code  # noqa: E402
from scripts import tipo_ir_indices as tir  # noqa: E402
from scripts import tipo_ir_plantillas as tp  # noqa: E402

ALFAS = (-2.0, -1.0, 0.0, 1.0, 2.0)
T_POLVO = (800.0, 1000.0, 1200.0, 1500.0, 1800.0)
REF = {"vis": 7500.0, "nir": 22000.0}
R_EJE = np.round(np.concatenate([np.arange(0.0, 0.1, 0.002), np.arange(0.1, 3.0001, 0.02)]), 4)
#: (banda, continuo) en Å para la profundidad del CO; «nominal» es la que se cita
VENTANAS_CO = {"nominal": ((22950.0, 23150.0), (22600.0, 22900.0)),
               "banda estrecha": ((22940.0, 23050.0), (22600.0, 22900.0)),
               "banda ancha": ((22950.0, 23300.0), (22600.0, 22900.0)),
               "continuo estrecho": ((22950.0, 23150.0), (22750.0, 22900.0)),
               "continuo azul": ((22950.0, 23150.0), (22300.0, 22550.0))}
SIGMA_SB = 5.670e-5


def planck(w_A, T):
    """B_λ en unidades arbitrarias (solo se usa la forma)."""
    return 1.0 / (w_A ** 5 * np.expm1(1.4388e8 / (w_A * T)))


def formas(segs, brazo):
    """{parámetro: [forma por segmento]}; vale 0 en los segmentos del otro brazo y 1 en λ_ref."""
    if brazo == "vis":
        return {("alfa", a): [(s["w"] / REF["vis"]) ** a if s["brazo"] == "vis" else np.zeros(s["w"].size) for s in segs]
                for a in ALFAS}
    return {("T_d", T): [planck(s["w"], T) / planck(REF["nir"], T) if s["brazo"] == "nir" else np.zeros(s["w"].size)
                         for s in segs] for T in T_POLVO}


def _k_ref(segs, brazo):
    return next(k for k, s in enumerate(segs) if s["brazo"] == brazo and s["w"][0] <= REF[brazo] <= s["w"][-1])


def ajusta_veiling(segs, bases, av_axis, brazo, factores=None, escala_comun_ir=False):
    """Por plantilla: el mejor (χ², r, forma, A_V), el χ² con r = 0 y el perfil χ²(r) (mínimo sobre forma y A_V)."""
    comun = [s["bueno"].copy() for s in segs]
    for m in bases.values():
        for k, mk in enumerate(m):
            comun[k] &= np.isfinite(mk)
    k_av = [np.asarray(tp.EXT.a_lambda_over_av(s["w"]), float) for s in segs]
    fac = factores or [1.0] * len(segs)
    fs = formas(segs, brazo)
    kr = _k_ref(segs, brazo)
    grupos = tp.grupos(segs, escala_comun_ir)
    res = {}
    for i, m in bases.items():
        mref = float(np.interp(REF[brazo], segs[kr]["w"], m[kr]))
        mejor, perfil = dict(chi2=np.inf), np.full(R_EJE.size, np.inf)
        for par, forma in fs.items():
            for av in np.atleast_1d(av_axis):
                chi, valido = np.zeros(R_EJE.size), np.ones(R_EJE.size, bool)
                for g in grupos:
                    dd, ivv, mms = [], [], []
                    for k in g:
                        c = comun[k]
                        e = 10 ** (-0.4 * k_av[k][c] * av)
                        dd.append(segs[k]["f"][c])
                        ivv.append(1.0 / (segs[k]["e"][c] * fac[k]) ** 2)
                        mms.append(e[None, :] * (m[k][c][None, :] + R_EJE[:, None] * mref * forma[k][c][None, :]))
                    d, iv, mm = np.concatenate(dd), np.concatenate(ivv), np.concatenate(mms, axis=1)
                    with np.errstate(invalid="ignore", divide="ignore"):
                        esc = (mm * d * iv).sum(1) / (mm * mm * iv).sum(1)
                    valido &= esc > 0
                    chi += ((d[None, :] - esc[:, None] * mm) ** 2 * iv).sum(1)
                chi[~valido] = np.inf
                perfil = np.minimum(perfil, chi)
                j = int(np.argmin(chi))
                if chi[j] < mejor["chi2"]:
                    mejor = dict(chi2=float(chi[j]), r=float(R_EJE[j]), par=list(par), av=float(av),
                                 chi2_r0=float(chi[0]))
        mejor["perfil"] = perfil
        res[i] = mejor
    return res, sum(int(c.sum()) for c in comun), len(grupos)


def con_veiling(segs, base, av, brazo, r, par):
    """Copia de los segmentos con un veiling r de forma `par` añadido al flujo (para inyectar)."""
    fs = formas(segs, brazo)[par]
    mref = float(np.interp(REF[brazo], segs[_k_ref(segs, brazo)]["w"], base[_k_ref(segs, brazo)]))
    out = []
    for s, fk in zip(segs, fs):
        e = 10 ** (-0.4 * np.asarray(tp.EXT.a_lambda_over_av(s["w"]), float) * av)
        out.append({**s, "f": s["f"] + r * mref * fk * e})
    return out


def calibra(p, brazo, tipo_ref, *, r_inyectada=0.0, semilla=0, ventana=1.5, escala_comun_ir=False):
    """Cada plantilla a ±`ventana` del tipo hace de dato (S/N del real, A_V de la Tabla 1, veiling inyectado con
    forma plana en el óptico y 1200 K en K) y se ajusta con las demás. Devuelve el r ajustado de cada una."""
    rng = np.random.default_rng(semilla)
    par = ("alfa", 0.0) if brazo == "vis" else ("T_d", 1200.0)
    out = []
    for i, t in enumerate(p["pls"]):
        if abs(spt_code(t["spt"]) - tipo_ref) > ventana + 1e-9:
            continue
        falsos = tp.falsos(p["segs"], p["bases"][i], p["av_tabla"], rng)
        if r_inyectada > 0:
            falsos = con_veiling(falsos, p["bases"][i], p["av_tabla"], brazo, r_inyectada, par)
        resto = {j: m for j, m in p["bases"].items() if j != i}
        fac = tp.factores_equilibrio(falsos, resto, p["av_axis"])
        res, _, _ = ajusta_veiling(falsos, resto, p["av_axis"], brazo, fac, escala_comun_ir)
        j = min(res, key=lambda k: res[k]["chi2"])
        out.append(dict(dato=f"{t['lib']}:{t['objeto']}", spt=t["spt"], r=res[j]["r"]))
    r = np.array([o["r"] for o in out])
    return dict(inyectado=r_inyectada, par=list(par), n=int(r.size), mediana=float(np.median(r)),
                p84=float(np.percentile(r, 84)), p95=float(np.percentile(r, 95)), filas=out)


def tipo_indices(slug, root=ROOT):
    """`root`: la raíz que contiene `runs/` (el repo salvo desde un worktree, como `--project-root`)."""
    perfil = tir.perfil(slug, root=root)
    av, av_err = tir.av_del_run(perfil["paper"]["run"], root=root)
    return tir.lectura(tir.DEFAULT_CDS, perfil["ir_spectra"]["spectra"], av, av_err)["spt"]


def veiling_ajuste(slug, modo, *, av_fija=True, escala_comun_ir=False, calibrar=False, r_inyectada=0.0):
    """`modo` = 'muse' (óptico) o 'ir' (K, banda completa). Mejor plantilla con y sin veiling y, si se pide,
    la calibración dejando una fuera."""
    p = tp.preparar(slug, modo, av_fija=av_fija)
    brazo = "vis" if modo == "muse" else "nir"
    tref = tipo_indices(slug)
    fac = tp.factores_equilibrio(p["segs"], p["bases"], p["av_axis"])
    res, n, ng = ajusta_veiling(p["segs"], p["bases"], p["av_axis"], brazo, fac, escala_comun_ir)
    filas = [dict(plantilla=f"{p['pls'][i]['lib']}:{p['pls'][i]['objeto']}", spt=p["pls"][i]["spt"],
                  codigo=spt_code(p["pls"][i]["spt"]), r=r["r"], par=r["par"], av=r["av"], chi2=r["chi2"],
                  chi2_r0=r["chi2_r0"]) for i, r in res.items()]
    out = dict(slug=slug, modo=modo, brazo=brazo, tipo_indices=tref,
               mejor=min(filas, key=lambda f: f["chi2"]), mejor_sin_veiling=min(filas, key=lambda f: f["chi2_r0"]),
               cerca=sorted((f for f in filas if abs(f["codigo"] - tref) <= 1.0 + 1e-9), key=lambda f: f["codigo"]))
    if calibrar:
        out["calibracion"] = calibra(p, brazo, tref, r_inyectada=r_inyectada, escala_comun_ir=escala_comun_ir)
    return out


def profundidad_co(w, f, variante="nominal"):
    banda, cont = VENTANAS_CO[variante]
    m = lambda lo, hi: tp.media_en_intervalos(w, f, np.array([lo]), np.array([hi]))[0]
    return 1.0 - m(*banda) / m(*cont)


def r_k_co(slug, tipo=None, ventana=1.0, variante="nominal", n_mc=300, root=ROOT):
    """r_K de cada espectro K del compañero por la profundidad del CO frente a las plantillas a ±`ventana`.

    `root` es la raíz que contiene `runs/` y de la que cuelga `g3_libraries_root`: el repo, salvo desde un
    worktree (lo usa `scripts/disco_cotas.py --project-root`)."""
    from musepipe.config import load_run_config
    run = tir.perfil(slug, root=root)["paper"]["run"]
    cfg = load_run_config(run, project_root=root).config
    tref = tipo_indices(slug, root=root) if tipo is None else spt_code(tipo)
    dfot = []
    for t in tp.plantillas(cfg, root=root):
        if abs(spt_code(t["spt"]) - tref) <= ventana + 1e-9:
            w, f = t["nir"]
            o = np.argsort(w)
            d = profundidad_co(w[o], tp.degrada(w[o], f[o]), variante)
            if np.isfinite(d):
                dfot.append(d)
    dfot = np.array(dfot)
    rng = np.random.default_rng(0)
    espectros = {}
    for nombre, spec in tir.perfil(slug, root=root)["ir_spectra"]["spectra"].items():
        if spec["band"] != "K":
            continue
        wu, f, e = tir.load(tir.DEFAULT_CDS, spec["file"])
        w = wu * 1e4
        d = profundidad_co(w, f, variante)
        err = float(np.std([profundidad_co(w, f + rng.normal(0, e), variante) for _ in range(n_mc)]))
        espectros[nombre] = dict(D_obs=float(d), D_obs_err=err, r_K=float(np.median(dfot) / d - 1),
                                 r_K_min=float(dfot.min() / d - 1), r_K_max=float(dfot.max() / d - 1))
    return dict(slug=slug, tipo=tir.spt_label(tref), variante=variante, n_plantillas=int(dfot.size),
                D_fot_mediana=float(np.median(dfot)), D_fot_min=float(dfot.min()), D_fot_max=float(dfot.max()),
                espectros=espectros)


def r_esperado(lacc_lsun, lbol_lsun, plantilla, lam_A, t_acc):
    """Veiling que produce la L_acc como cuerpo negro de T_acc sobre la fotosfera de la plantilla (ver docstring)."""
    wv, fv = plantilla["vis"]
    wn, fn = plantilla["nir"]
    sv, sn = (wv > 5300) & (wv < 10000), (wn > 10000) & (wn < 24800)
    fbol = (np.trapz(np.nan_to_num(fv[sv]), wv[sv]) + np.trapz(np.nan_to_num(fn[sn]), wn[sn])) / 0.85
    w, f = (wv, fv) if lam_A < 10000 else (wn, fn)
    ffot = np.nanmedian(f[(w > lam_A - 50) & (w < lam_A + 50)]) / fbol
    x = lam_A * 1e-8
    pib = np.pi * 2 * 6.626e-27 * 2.998e10 ** 2 / x ** 5 / np.expm1(1.4388 / (x * t_acc)) * 1e-8
    return float(lacc_lsun * pib / (SIGMA_SB * t_acc ** 4) / (lbol_lsun * ffot))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for nombre in ("optico", "k"):
        q = sub.add_parser(nombre)
        q.add_argument("slug")
        q.add_argument("--av-fija", action="store_true")
        q.add_argument("--escala-comun-ir", action="store_true")
        q.add_argument("--calibra", action="store_true")
        q.add_argument("--inyecta", type=float, default=0.0)
        q.add_argument("--json", type=Path)
    q = sub.add_parser("co")
    q.add_argument("slug")
    q.add_argument("--tipo")
    q.add_argument("--variante", default="nominal", choices=list(VENTANAS_CO) + ["todas"])
    q.add_argument("--json", type=Path)
    q = sub.add_parser("esperado")
    q.add_argument("--lacc", type=float, required=True, help="L_acc en L_sol")
    q.add_argument("--lbol", type=float, required=True, help="L_bol en L_sol")
    q.add_argument("--plantilla", required=True, help="objeto de la biblioteca de Manara (p. ej. LM601)")
    q.add_argument("--slug", required=True, help="objeto cuyo run del paper da g3_libraries_root")
    a = ap.parse_args(argv)
    warnings.simplefilter("ignore", RuntimeWarning)
    if a.cmd in ("optico", "k"):
        out = veiling_ajuste(a.slug, "muse" if a.cmd == "optico" else "ir", av_fija=a.av_fija,
                             escala_comun_ir=a.escala_comun_ir, calibrar=a.calibra, r_inyectada=a.inyecta)
        m, m0 = out["mejor"], out["mejor_sin_veiling"]
        print(f"{a.slug} {a.cmd}: mejor {m['plantilla']} {m['spt']} r = {m['r']:.3f} {m['par']} A_V {m['av']}; "
              f"sin veiling {m0['plantilla']} {m0['spt']}")
        for f in out["cerca"]:
            print(f"   {f['plantilla']:<38s} {f['spt']:<5s} r = {f['r']:.3f} {f['par']}")
        if "calibracion" in out:
            c = out["calibracion"]
            print(f"   calibración (inyectado {c['inyectado']}): mediana {c['mediana']:.3f}, p84 {c['p84']:.3f}, p95 {c['p95']:.3f}")
    elif a.cmd == "co":
        variantes = list(VENTANAS_CO) if a.variante == "todas" else [a.variante]
        out = [r_k_co(a.slug, a.tipo, variante=v) for v in variantes]
        for o in out:
            print(f"{a.slug} CO ({o['variante']}), plantillas {o['tipo']} ± 1 (n {o['n_plantillas']}): "
                  f"D_fot {o['D_fot_mediana']:.3f} ({o['D_fot_min']:.3f}–{o['D_fot_max']:.3f})")
            for n, e in o["espectros"].items():
                print(f"   {n:<10s} D_obs {e['D_obs']:.3f} ± {e['D_obs_err']:.3f} → r_K {e['r_K']:+.2f} "
                      f"({e['r_K_min']:+.2f} a {e['r_K_max']:+.2f})")
    else:
        from musepipe.config import load_run_config
        cfg = load_run_config(tir.perfil(a.slug)["paper"]["run"], project_root=ROOT).config
        t = next(t for t in tp.plantillas(cfg, ("manara",)) if t["objeto"] == a.plantilla)
        out = {f"{lam:.0f}": {f"{T:.0f}": r_esperado(a.lacc, a.lbol, t, lam, T) for T in (8000.0, 10000.0)}
               for lam in (6563.0, 7500.0, 22000.0)}
        print(json.dumps(out, indent=1))
    if getattr(a, "json", None):
        a.json.write_text(json.dumps(out, indent=1, ensure_ascii=False, default=float))
    return 0


if __name__ == "__main__":
    sys.exit(main())
