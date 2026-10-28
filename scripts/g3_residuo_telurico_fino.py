#!/usr/bin/env python
"""Residuo telúrico fino: amplitud α del patrón de la transmisión que queda en el espectro corregido.

α = 0 corrección perfecta, α = +1 espectro sin corregir, α < 0 sobrecorrección. Nula: la misma curva
desplazada ±3–15 Å (18 desplazamientos). Medido primero en H2O 8100–8400 Å (2026-09-24,
`docs/2026-09-24_residuo_telurico_h2o_y_mascara_d9.md`, que redujo la máscara D9 de G3 a O2); ampliado el
2026-10-28 a O2 B (6860–6960 Å) y O2 A (7590–7700 Å), las dos bandas que G3 y los ajustes de T_eff
(`scripts/teff_bayes.py`) siguen enmascarando (pedido del autor: «las bandas telúricas de O2, si están
corregidas, deberían poder agregarse»).

Método (el de 2026-09-24, sin cambios): se quita la forma suave del espectro corregido (mediana móvil de 31
canales en ln F, `hp`) y se ajusta por mínimos cuadrados la amplitud α del mismo filtrado de ln T, con
término constante (`alpha`). T es una curva de transmisión interpolada a la rejilla del espectro: la de la
estándar de la reducción de ESO (`TELLURIC_TRANS.fits`, la que corrigió el cubo vía `muse_scipost`) o una de
las que guarda A3 en `stages/a3_molecfit/transmission_*.npy` (`--patrones-a3`). En O2 la absorción es ~20
veces la de H2O (O2 A: 25.6 % de media y 70.7 % de máximo a la resolución de MUSE con la curva de 2026-09-24),
así que la misma α es un residuo mucho mayor en flujo: por eso cada medida da también el residuo en % del
continuo, 1 − T^α (positivo = absorción que queda), su media y su máximo en la banda, su máximo en bins de 12
canales (15 Å, los de los ajustes de T_eff) y lo mismo con |α| + 2σ (la cota a 2σ).

Rasgos estelares: cada banda excluye los suyos (`BANDAS`): Na I 8183/8195 en H2O 8200, K I 7665/7699 en O2 A.
La medida «con rasgos dentro» se da al lado, como en 2026-09-24. Se da también la profundidad telúrica del
patrón en el centro de cada rasgo y el residuo α·(1 − T) allí.

Calibración (`--inyecta`): el filtro de mediana no es lineal y O2 A es profunda, así que se comprueba que α
es una medida de la amplitud: se multiplica el espectro por T^a (a = 0.1, 0.3, 1) y la ganancia
(α(a) − α(0)) / a tiene que salir ≈ 1.

Patrones de A3: molecfit corre sobre el cubo YA corregido por `muse_scipost`, así que su transmisión es un
modelo del RESIDUO (O2 solo; vale 1 en H2O, `docs/2026-09-24_...` §2), no de la atmósfera. De cada una se da
su amplitud respecto a la estándar (`alpha` con la transmisión de molecfit en el papel del espectro): la α
que molecfit midió con un modelo directo sobre la primaria. Se marcan las que no declara el QC vigente de A3
(`stage00t_qc.json` → `products.transmission_by_method`): son de una corrida anterior de A3 en el mismo run.

Entradas: los espectros del run del paper de cada objeto (`targets/<slug>.json` → `paper.run`): la primaria
(`spec_calibrated_psffit_star.fits`) y el compañero (`spec_final_object.fits`, el que ajustan G3 y
`teff_bayes.py`), columna `flux`; o los que se pasen con `--espectro NOMBRE=RUTA` (para reproducir la sesión
del 2026-09-24 con sus entradas).

uso:
    python scripts/g3_residuo_telurico_fino.py --project-root RAIZ \\
        --patron estandar=RAIZ/runs/<RUN_RAW>/raw_reduction/TELLURIC_TRANS.fits \\
        [--patrones-a3] [--bandas o2_b o2_a h2o_8200] [--slug SLUG ...] [--espectro NOMBRE=RUTA ...]
        [--inyecta] [--por-bin] [--bordes] [--json SALIDA] [--md SALIDA]

`--por-bin` y `--bordes` (2026-10-28, dos ideas de la revisión bibliográfica de la sesión): (a) en cada bin de 15 Å
del ajuste de T_eff dentro de O2 B y O2 A, el residuo que deja la corrección según la primaria del mismo cubo
(1 − T^α con su α y con |α| + 2σ, a R = 500) frente al error relativo del compañero en ese bin; (b) dónde absorbe
la estándar más de 0.5, 1 y 2 % alrededor de las máscaras (a la resolución de MUSE y a R = 500) y cuánto en las
ventanas que quedan fuera (O2 B hasta 7000 Å; el hueco 9100–9250 Å entre las máscaras de H2O).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from astropy.io import fits
from scipy.ndimage import median_filter

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

#: bandas medidas: nombre -> (lo, hi) en Å (las de `musepipe.telluric_lines.TELLURIC_BANDS`), los rasgos
#: estelares que se excluyen de la medida [(nombre, centro_A, (lo, hi))] y la especie
BANDAS = {
    "o2_b": dict(lo=6860.0, hi=6960.0, especie="O2", rasgos=()),
    "o2_a": dict(lo=7590.0, hi=7700.0, especie="O2",
                 rasgos=(("K I 7665", 7664.90, (7653.0, 7677.0)), ("K I 7699", 7698.96, (7687.0, 7711.0)))),
    "h2o_7200": dict(lo=7130.0, hi=7360.0, especie="H2O", rasgos=()),
    "h2o_8200": dict(lo=8100.0, hi=8400.0, especie="H2O",
                     rasgos=(("Na I 8183", 8183.26, (8170.0, 8210.0)), ("Na I 8195", 8194.82, (8170.0, 8210.0)))),
}
#: los desplazamientos de la nula (Å), los de 2026-09-24
DESPLAZAMIENTOS_A = np.r_[-15:-2.9:1.5, 3:15.1:1.5]
#: canales por bin de los ajustes de T_eff (`tipo_ir_plantillas.segmento_muse`, 15 Å)
CANALES_BIN = 12
#: amplitudes inyectadas en la calibración (`--inyecta`)
INYECCIONES = (0.1, 0.3, 1.0)
#: un patrón no sirve en una banda si su absorción máxima allí no llega a esto (molecfit vale 1 fuera de O2 A)
ABSORCION_MINIMA = 0.005


# ================================================================== el método de 2026-09-24 (sin cambios)
def spec(path, col='flux'):
    """(λ, flujo, error) de un espectro de la cadena (extensión SPECTRUM; error: la primera columna con 'err')."""
    with fits.open(path) as h:
        d = h['SPECTRUM'].data if 'SPECTRUM' in [x.name for x in h] else h[1].data
        return (np.asarray(d['wave_A'], float), np.asarray(d[col], float),
                np.asarray(d[[c for c in d.names if 'err' in c.lower()][0]], float))


def hp(y, k=31):
    """y menos su mediana móvil de k canales (NaN rellenados con la mediana para filtrar)."""
    y = np.where(np.isfinite(y), y, np.nan)
    base = median_filter(np.nan_to_num(y, nan=np.nanmedian(y)), size=k, mode='nearest')
    return y - base


def alpha(w, f, T, lo=8100, hi=8400, excl=((8170, 8210),), shift=0.0):
    """(α, rms del patrón filtrado) en [lo, hi] sin los tramos `excl`; `shift` desplaza T (la nula)."""
    Ts = np.interp(w, w + shift, T)
    ok = np.isfinite(f) & (f > 0) & np.isfinite(Ts) & (Ts > 0)
    lf = np.full_like(f, np.nan)
    lf[ok] = np.log(f[ok])
    lt = np.log(np.clip(Ts, 1e-3, None))
    r = hp(lf)
    t = hp(lt)
    m = (w > lo) & (w < hi) & np.isfinite(r) & np.isfinite(t)
    for a, b in excl:
        m &= ~((w > a) & (w < b))
    A = np.c_[t[m], np.ones(m.sum())]
    c, *_ = np.linalg.lstsq(A, r[m], rcond=None)
    return c[0], np.std(t[m])


def informe(nombre, w, f, T):
    """La línea de H2O 8100–8400 Å de la sesión del 2026-09-24."""
    a0, amp = alpha(w, f, T)
    nul = [alpha(w, f, T, shift=s)[0] for s in DESPLAZAMIENTOS_A]
    sd = np.std(nul)
    a_all, _ = alpha(w, f, T, excl=())
    depthNa = np.interp([8183.3, 8194.8], w, 1 - T)
    print(f"{nombre:34s} α = {a0:+.3f} ± {sd:.3f} (nula, n={len(nul)})   con Na I dentro: {a_all:+.3f}   "
          f"profundidad telúrica en Na I: {100*depthNa[0]:.1f}/{100*depthNa[1]:.1f} % → residuo ≈ {100*a0*depthNa.mean():+.2f} %")


# ================================================================== patrones
def carga_patron(ruta):
    """(λ, T) de una curva de transmisión: la tabla de la reducción (`TELLURIC_TRANS.fits`: wave_A,
    transmission), un `STD_TELLURIC` de ESO (lambda, ftelluric) o un `.npy` de A3 (con `wave_A.npy` al lado)."""
    ruta = Path(ruta)
    if ruta.suffix == ".npy":
        return np.load(ruta.parent / "wave_A.npy").astype(float), np.load(ruta).astype(float)
    with fits.open(ruta) as h:
        d = h[1].data
        nombres = [n.lower() for n in d.names]
        if "transmission" in nombres:
            return np.asarray(d["wave_A"], float), np.asarray(d["transmission"], float)
        return np.asarray(d["lambda"], float), np.asarray(d["ftelluric"], float)


def en_rejilla(w, wT, T):
    """La transmisión en la rejilla del espectro (interpolación lineal; 1 fuera de su cobertura). Es lo que
    hizo la sesión del 2026-09-24: con la curva SIN interpolar, la α de las primarias cambia en la 3.ª cifra."""
    out = np.interp(w, wT, T)
    out[(w < wT.min()) | (w > wT.max())] = 1.0
    return out


def patrones_a3(stage_dir):
    """{'a3:<método>': (λ, T, declarado)} de las transmisiones que guarda A3 en `stages/a3_molecfit/`;
    `declarado`: si el QC vigente de A3 la lista en `products.transmission_by_method`."""
    d = Path(stage_dir) / "a3_molecfit"
    if not (d / "wave_A.npy").exists():
        return {}
    qc = Path(stage_dir) / "stage00t_qc.json"
    decl = set()
    if qc.exists():
        decl = set((json.loads(qc.read_text()).get("products") or {}).get("transmission_by_method") or {})
    out = {}
    for f in sorted(d.glob("transmission_*.npy")):
        metodo = f.stem[len("transmission_"):]
        w, T = carga_patron(f)
        out[f"a3:{metodo}"] = (w, T, metodo in decl)
    return out


# ================================================================== la medida
def absorcion(w, T, banda):
    """Absorción media y máxima (%) de la curva T (en la rejilla w) en la banda."""
    b = BANDAS[banda]
    m = (w >= b["lo"]) & (w <= b["hi"])
    return float(100 * np.mean(1 - T[m])), float(100 * np.max(1 - T[m]))


def residuo_flujo(w, T, banda, a):
    """El residuo 1 − T^a en % del continuo en la banda: media, máximo |·| por canal y máximo |·| de la media
    en bins de `CANALES_BIN` canales consecutivos (los bins de 15 Å de los ajustes de T_eff)."""
    b = BANDAS[banda]
    m = (w >= b["lo"]) & (w <= b["hi"])
    r = 100 * (1 - np.clip(T[m], 1e-3, None) ** a)
    nb = r.size // CANALES_BIN
    rb = r[:nb * CANALES_BIN].reshape(nb, CANALES_BIN).mean(axis=1) if nb else r
    return dict(media=float(r.mean()), max_canal=float(r[np.argmax(np.abs(r))]),
                max_bin15=float(rb[np.argmax(np.abs(rb))]))


def mide(w, f, T, banda, nula=True):
    """α en la banda (sin sus rasgos estelares), su nula, α con los rasgos dentro, la absorción del patrón y
    el residuo en flujo (con α y con |α| + 2σ)."""
    b = BANDAS[banda]
    excl = tuple(r[2] for r in b["rasgos"])
    a0, amp = alpha(w, f, T, b["lo"], b["hi"], excl)
    out = dict(banda=banda, lo=b["lo"], hi=b["hi"], alpha=float(a0), rms_patron_filtrado=float(amp))
    if b["rasgos"]:
        out["alpha_con_rasgos"] = float(alpha(w, f, T, b["lo"], b["hi"], ())[0])
    m = (w > b["lo"]) & (w < b["hi"])
    for lo, hi in excl:
        m &= ~((w > lo) & (w < hi))
    out["n_canales"] = int(m.sum())
    out["absorcion_media_pct"], out["absorcion_max_pct"] = absorcion(w, T, banda)
    out["residuo_pct"] = residuo_flujo(w, T, banda, a0)
    if nula:
        nul = np.array([alpha(w, f, T, b["lo"], b["hi"], excl, shift=s)[0] for s in DESPLAZAMIENTOS_A])
        sd = float(np.std(nul))
        out.update(sigma_nula=sd, n_nula=int(nul.size), z=float(a0 / sd) if sd > 0 else float("nan"))
        out["residuo_2sigma_pct"] = residuo_flujo(w, T, banda, abs(a0) + 2 * sd)
    out["rasgos"] = [dict(nombre=n, centro_A=c, profundidad_telurica_pct=float(100 * np.interp(c, w, 1 - T)),
                          residuo_pct=float(100 * a0 * np.interp(c, w, 1 - T))) for n, c, _ in b["rasgos"]]
    return out


def ganancia(w, f, T, banda, amplitudes=INYECCIONES):
    """{a: (α(a) − α(0)) / a} con el espectro multiplicado por T^a: ≈ 1 si α mide la amplitud."""
    b = BANDAS[banda]
    excl = tuple(r[2] for r in b["rasgos"])
    a0 = alpha(w, f, T, b["lo"], b["hi"], excl)[0]
    return {f"{a:g}": float((alpha(w, f * np.clip(T, 1e-3, None) ** a, T, b["lo"], b["hi"], excl)[0] - a0) / a)
            for a in amplitudes}


# ================================================================== entradas
def objetos_del_paper(root=ROOT):
    """Los slugs de `targets/` que declaran `paper`, en el orden de `paper.order` (como
    `halpha_criterios.objetos_del_paper`)."""
    from musepipe.targets import targets_dir
    fichas = [(json.loads(f.read_text()), f.stem) for f in sorted(targets_dir(root).glob("*.json"))]
    return tuple(s for _, s in sorted((d["paper"]["order"], s) for d, s in fichas if "paper" in d))


def espectros_del_paper(slug, root):
    """(run, stage_dir, {'<slug> primaria': ruta, '<slug> compañero': ruta}) del run del paper. Los `targets/`
    son los de este clon; los runs, los de `root`."""
    from musepipe.paths import RunPaths
    from musepipe.targets import load_target
    run = load_target(slug, project_root=ROOT)["paper"]["run"]
    sd = Path(RunPaths.from_project_root(run, root).stage_dir)
    return run, sd, {f"{slug} primaria": sd / "spec_calibrated_psffit_star.fits",
                     f"{slug} compañero": sd / "spec_final_object.fits"}


def _pares(lista, que):
    out = {}
    for x in lista or []:
        if "=" not in x:
            raise SystemExit(f"{que} {x!r}: se espera NOMBRE=RUTA")
        k, v = x.split("=", 1)
        out[k] = Path(v).expanduser()
    return out


def calcula(espectros, patrones, bandas, patrones_por_espectro=None, inyecta=False):
    """Mide cada espectro {nombre: ruta} con cada patrón {nombre: (λ, T, declarado)} en cada banda.
    `patrones_por_espectro`: {nombre de espectro: {patrón: (λ, T, declarado)}} que solo valen para ese espectro
    (los de A3, que son de la noche de su run)."""
    filas = []
    for ne, ruta in espectros.items():
        w, f, _ = spec(ruta)
        todos = dict(patrones)
        todos.update((patrones_por_espectro or {}).get(ne, {}))
        for np_, (wT, T0, decl) in todos.items():
            T = en_rejilla(w, wT, T0)
            for banda in bandas:
                if absorcion(w, T, banda)[1] < 100 * ABSORCION_MINIMA:
                    filas.append(dict(espectro=ne, ruta=str(ruta), patron=np_, declarado_qc=decl, banda=banda,
                                      sin_patron=True))
                    continue
                r = mide(w, f, T, banda)
                r.update(espectro=ne, ruta=str(ruta), patron=np_, declarado_qc=decl)
                if inyecta:
                    r["ganancia"] = ganancia(w, f, T, banda)
                filas.append(r)
    return filas


def consistencia(patrones_a3_run, patrones, bandas):
    """Amplitud de cada patrón de A3 respecto a cada patrón externo (`alpha` con la T de A3 como «espectro»):
    la α que el modelo de A3 atribuye al residuo de la primaria de su run."""
    out = []
    for na, (wa, Ta, decl) in patrones_a3_run.items():
        for ne, (we, Te, _) in patrones.items():
            T = en_rejilla(wa, we, Te)
            for banda in bandas:
                if absorcion(wa, Ta, banda)[1] < 100 * ABSORCION_MINIMA:
                    continue
                b = BANDAS[banda]
                a, _ = alpha(wa, Ta, T, b["lo"], b["hi"], ())
                out.append(dict(patron_a3=na, declarado_qc=decl, respecto_a=ne, banda=banda, alpha=float(a),
                                absorcion_media_pct=absorcion(wa, Ta, banda)[0],
                                absorcion_max_pct=absorcion(wa, Ta, banda)[1]))
    return out


# ================================================================== bin a bin y bordes (2026-10-28, revisión bibliográfica)
#: resolución de los ajustes de T_eff (`tipo_ir_plantillas.R_OBJ`)
R_AJUSTE = 500.0
#: regiones donde se buscan los tramos con absorción (`bordes`): alrededor de O2 B, H2O 7200, O2 A, H2O 8200 y del
#: hueco 9100–9250 Å que las máscaras de H2O de los ajustes dejan entre 8900–9100 y 9250–9700
REGIONES_BORDES_A = ((6700.0, 7100.0), (7050.0, 7450.0), (7500.0, 7800.0), (8000.0, 8500.0), (8800.0, 9350.0))
#: umbrales de absorción (1 − T) de `bordes`
UMBRALES_BORDES = (0.005, 0.01, 0.02)
#: dos tramos por encima del umbral separados por menos de esto (Å) se juntan
HUECO_BORDES_A = 5.0
#: ventanas fuera de las máscaras de hoy donde se da la absorción media y máxima (Kirkpatrick et al. 1999 dan O2 B
#: hasta ~7000 Å y H2O hasta ~9300 Å: lo dijo la revisión bibliográfica del 2026-10-28)
VENTANAS_BORDES_A = {"O2 B, lado azul 6800–6860": (6800.0, 6860.0), "O2 B, cola 6960–7000": (6960.0, 7000.0),
                     "O2 A, cola 7700–7750": (7700.0, 7750.0), "H2O 7200, cola 7360–7450": (7360.0, 7450.0),
                     "H2O, hueco 9100–9250": (9100.0, 9250.0)}


def degrada_a_r(w, y, r=R_AJUSTE):
    """y degradada con una gaussiana de FWHM λ/r (la de los ajustes de T_eff; la LSF de MUSE, ~2.3 Å, se
    desprecia frente a λ/500 ≈ 14–19 Å)."""
    from musepipe.models.prep import degrade_variable
    return degrade_variable(np.asarray(w, float), np.asarray(y, float), np.asarray(w, float) / r)


def tramos(w, x, umbral, hueco=HUECO_BORDES_A):
    """[lo, hi] de los tramos donde x > umbral, juntando los separados por menos de `hueco` Å."""
    idx = np.flatnonzero(x > umbral)
    if not idx.size:
        return []
    out = [[float(w[idx[0]]), float(w[idx[0]])]]
    for i in idx[1:]:
        if w[i] - out[-1][1] <= hueco:
            out[-1][1] = float(w[i])
        else:
            out.append([float(w[i]), float(w[i])])
    return out


def bordes(w, T, regiones=REGIONES_BORDES_A, umbrales=UMBRALES_BORDES, ventanas=VENTANAS_BORDES_A):
    """(b) Los bordes de las máscaras contra la transmisión: por región, los tramos donde la absorción 1 − T pasa
    cada umbral, a la resolución de MUSE y a R = 500; la absorción media y máxima en `ventanas`; y los tramos donde
    T < 1 (`tramos_T_menor_que_1`). Ojo: la curva de `muse_standard` vale EXACTAMENTE 1 fuera de las ventanas
    telúricas en que la ajusta la reducción, así que fuera de ellas no mide la atmósfera (y la reducción no corrige
    nada allí): una absorción 0 en esas ventanas no dice que la atmósfera no absorba."""
    a_nat = 1 - np.asarray(T, float)
    a_500 = degrada_a_r(w, a_nat)
    out = dict(tramos=[], ventanas={}, tramos_T_menor_que_1=tramos(w, a_nat, 0.0),
               canales_T_igual_1=int(np.sum(np.asarray(T, float) == 1.0)), canales=int(np.size(T)))
    for lo, hi in regiones:
        m = (w >= lo) & (w <= hi)
        for u in umbrales:
            out["tramos"].append(dict(region=[lo, hi], umbral_pct=100 * u,
                                      nativa=tramos(w[m], a_nat[m], u), r500=tramos(w[m], a_500[m], u)))
    for k, (lo, hi) in ventanas.items():
        m = (w >= lo) & (w <= hi)
        out["ventanas"][k] = dict(media_pct=float(100 * a_nat[m].mean()), max_pct=float(100 * a_nat[m].max()),
                                  media_r500_pct=float(100 * a_500[m].mean()), max_r500_pct=float(100 * a_500[m].max()))
    return out


def compara_por_bin(slug, root, patron, bandas=("o2_b", "o2_a")):
    """(a) Bin a bin, en los bins de 15 Å del ajuste de T_eff que tapan hoy las máscaras de O2: el residuo que deja
    la corrección según la PRIMARIA del mismo cubo (1 − T^α, con la α de la primaria en esa banda y con |α| + 2σ,
    degradado a R = 500 y promediado en el bin) frente al error relativo σ/F del COMPAÑERO en ese bin
    (`tipo_ir_plantillas.segmento_muse` sin máscaras telúricas: el error de los datos, sin el término de modelo f_b
    de `teff_bayes`). Supone que el residuo del compañero es el de la primaria (la corrección es la misma para todo
    el cubo) y que tiene la forma T^α."""
    from scripts import teff_bayes as tb
    from scripts import tipo_ir_plantillas as tp
    tb.fija_raiz(root)
    run, _, es = espectros_del_paper(slug, root)
    wp, fp, _ = spec(es[f"{slug} primaria"])
    Tp = en_rejilla(wp, *patron)
    seg = tp.segmento_muse(run, h2o=False, quita_o2=tp.O2_MUSE)
    medio = float(np.median(seg["hi"] - seg["lo"])) / 2
    filas = []
    for banda in bandas:
        r = mide(wp, fp, Tp, banda)
        b = BANDAS[banda]
        res = {k: degrada_a_r(wp, 1 - np.clip(Tp, 1e-3, None) ** amp)
               for k, amp in (("central", r["alpha"]), ("cota_2sigma", abs(r["alpha"]) + 2 * r["sigma_nula"]))}
        for i in np.flatnonzero(seg["bueno"] & (seg["w"] >= b["lo"] - medio) & (seg["w"] <= b["hi"] + medio)):
            m = (wp >= seg["lo"][i]) & (wp < seg["hi"][i])
            err = float(100 * seg["e"][i] / seg["f"][i])
            c, s2 = (float(100 * res[k][m].mean()) for k in ("central", "cota_2sigma"))
            filas.append(dict(slug=slug, run=run, banda=banda, w_A=float(seg["w"][i]), alpha_primaria=r["alpha"],
                              sigma_primaria=r["sigma_nula"], residuo_central_pct=c, residuo_2sigma_pct=s2,
                              err_rel_companero_pct=err, cota_sobre_error=s2 / err))
    return filas


def tabla_md(res):
    """Tabla en markdown de las filas de `calcula` y de `consistencia`."""
    lin = ["| espectro | patrón | banda | α | σ nula | z | α con rasgos | absorción media / máx (%) | "
           "residuo medio (%) | máx en bin de 15 Å (%) | cota 2σ: media / máx bin (%) | ganancia |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in res["medidas"]:
        if r.get("sin_patron"):
            continue
        g = "/".join(f"{v:.2f}" for v in r["ganancia"].values()) if "ganancia" in r else "—"
        ar = f"{r['alpha_con_rasgos']:+.3f}" if "alpha_con_rasgos" in r else "—"
        p = r["patron"] + ("" if r["declarado_qc"] or not r["patron"].startswith("a3:") else " (no declarado)")
        lin.append(f"| {r['espectro']} | {p} | {r['banda']} | {r['alpha']:+.3f} | {r['sigma_nula']:.3f} | "
                   f"{r['z']:+.1f} | {ar} | {r['absorcion_media_pct']:.2f} / {r['absorcion_max_pct']:.1f} | "
                   f"{r['residuo_pct']['media']:+.2f} | {r['residuo_pct']['max_bin15']:+.2f} | "
                   f"{r['residuo_2sigma_pct']['media']:.2f} / {r['residuo_2sigma_pct']['max_bin15']:.2f} | {g} |")
    if res.get("consistencia"):
        lin += ["", "| patrón de A3 | respecto a | banda | α del patrón | absorción media / máx (%) |",
                "|---|---|---|---|---|"]
        for c in res["consistencia"]:
            p = c["patron_a3"] + ("" if c["declarado_qc"] else " (no declarado)")
            lin.append(f"| {c['run']}: {p} | {c['respecto_a']} | {c['banda']} | {c['alpha']:+.3f} | "
                       f"{c['absorcion_media_pct']:.2f} / {c['absorcion_max_pct']:.1f} |")
    if res.get("por_bin"):
        lin += ["", "| objeto | banda | bin (Å) | α primaria | residuo central (%) | cota 2σ (%) | σ/F compañero (%) | "
                "cota / σ |", "|---|---|---|---|---|---|---|---|"]
        for f in res["por_bin"]:
            lin.append(f"| {f['slug']} | {f['banda']} | {f['w_A']:.1f} | {f['alpha_primaria']:+.3f} ± "
                       f"{f['sigma_primaria']:.3f} | {f['residuo_central_pct']:+.2f} | {f['residuo_2sigma_pct']:.2f} | "
                       f"{f['err_rel_companero_pct']:.2f} | {f['cota_sobre_error']:.2f} |")
    if res.get("bordes"):
        for np_, bd in res["bordes"].items():
            fmt = lambda xs: ", ".join(f"{lo:.0f}–{hi:.0f}" for lo, hi in xs) or "—"
            lin += ["", f"Bordes contra el patrón `{np_}`: T = 1 exacto en {bd['canales_T_igual_1']} de "
                        f"{bd['canales']} canales; T < 1 en {fmt(bd['tramos_T_menor_que_1'])} Å.", "",
                    "| ventana | absorción media / máx (%) | a R = 500 |", "|---|---|---|"]
            for k, v in bd["ventanas"].items():
                lin.append(f"| {k} | {v['media_pct']:.2f} / {v['max_pct']:.2f} | {v['media_r500_pct']:.2f} / "
                           f"{v['max_r500_pct']:.2f} |")
            lin += ["", "| región (Å) | umbral (%) | tramos, resolución de MUSE | tramos, R = 500 |", "|---|---|---|---|"]
            for t in bd["tramos"]:
                lin.append(f"| {t['region'][0]:.0f}–{t['region'][1]:.0f} | {t['umbral_pct']:g} | {fmt(t['nativa'])} | "
                           f"{fmt(t['r500'])} |")
    return "\n".join(lin)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--project-root", type=Path, default=ROOT,
                    help="clon con runs/ (desde un worktree, el principal)")
    ap.add_argument("--patron", action="append", default=None, metavar="NOMBRE=RUTA",
                    help="curva de transmisión (repetible): TELLURIC_TRANS.fits, STD_TELLURIC o .npy de A3")
    ap.add_argument("--patrones-a3", action="store_true",
                    help="añade, para los espectros de cada run, las transmisiones de su A3 (stages/a3_molecfit)")
    ap.add_argument("--bandas", nargs="+", choices=tuple(BANDAS), default=["o2_b", "o2_a", "h2o_8200"])
    ap.add_argument("--slug", action="append", default=None,
                    help="objeto (repetible); por defecto, los del paper (targets/*.json con `paper`)")
    ap.add_argument("--espectro", action="append", default=None, metavar="NOMBRE=RUTA",
                    help="espectros explícitos en vez de los del run del paper (repetible)")
    ap.add_argument("--inyecta", action="store_true", help="calibración: ganancia con T^a inyectada")
    ap.add_argument("--por-bin", action="store_true",
                    help="bin a bin (15 Å, los del ajuste) en O2: el residuo según la primaria frente al error del "
                         "compañero, con el primer --patron")
    ap.add_argument("--bordes", action="store_true",
                    help="tramos de absorción alrededor de las máscaras, con cada --patron")
    ap.add_argument("--json", type=Path, default=None)
    ap.add_argument("--md", type=Path, default=None)
    a = ap.parse_args(argv)
    root = Path(a.project_root).resolve()
    patrones = {k: (*carga_patron(v), True) for k, v in _pares(a.patron, "--patron").items()}
    espectros, por_espectro, consis, runs = {}, {}, [], {}
    if a.espectro:
        espectros = _pares(a.espectro, "--espectro")
    else:
        for slug in a.slug or objetos_del_paper():
            run, sd, es = espectros_del_paper(slug, root)
            espectros.update(es)
            runs[slug] = run
            if a.patrones_a3:
                pa = patrones_a3(sd)
                for ne in es:
                    por_espectro[ne] = pa
                consis += [dict(c, run=run) for c in consistencia(pa, patrones, a.bandas)]
    if not patrones and not por_espectro:
        raise SystemExit("sin patrón: pasa --patron NOMBRE=RUTA y/o --patrones-a3")
    res = dict(runs=runs, patrones={k: str(v) for k, v in _pares(a.patron, "--patron").items()},
               bandas={b: BANDAS[b] for b in a.bandas}, desplazamientos_A=DESPLAZAMIENTOS_A.tolist(),
               medidas=calcula(espectros, patrones, a.bandas, por_espectro, a.inyecta), consistencia=consis)
    if a.por_bin:
        if not patrones or not runs:
            raise SystemExit("--por-bin necesita --patron y los runs del paper (sin --espectro)")
        primero = next(iter(patrones.values()))
        res["por_bin"] = [f for slug in runs for f in compara_por_bin(slug, root, primero[:2])]
    if a.bordes:
        res["bordes"] = {k: bordes(wT, T) for k, (wT, T, _) in patrones.items()}
    md = tabla_md(res)
    print(md)
    if a.json:
        a.json.write_text(json.dumps(res, indent=1, ensure_ascii=False, default=float) + "\n", encoding="utf-8")
    if a.md:
        a.md.write_text(md + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
