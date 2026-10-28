#!/usr/bin/env python
"""Las cifras de la PROSA del paper que salen de las cadenas de T_eff (dynesty), desde las cadenas y los runs.

Solo lectura. Hermano de `teff_bayes.py tabla|rejillas` (cuerpos de las Tablas B.6 y B.7), de
`paper_figures.py --figure teff_corner*` (Figs. B.4–B.6) y de `teff_suplemento.py`: esos reproducen las tablas
y las figuras de las cadenas; este, las frases. Hasta el 2026-10-26 las frases salían de herramientas de
carpetas de sesión (`~/muse_sesion_20261025/paper_c0/t3/herramientas/`: `cifras_t3.py`, `items.py`,
`items_c0.py`, `evo_fila.py`), que no son del pipeline. Aquí están las mismas cuentas, con la definición de cada
cifra escrita junto a la frase en la que va, y las frases que las decisiones E del autor (2026-10-26) añadieron.

Qué lee:

- las cadenas de `teff_bayes.py ajusta` (json + npz) de los directorios `--cadenas` (repetible; se recorren
  enteros, con sus subdirectorios). Una cadena es su etiqueta: si dos directorios traen la misma, gana el que se
  pasó antes y la otra sale en el informe como «sombreada». Cada parte pide sus cadenas POR ETIQUETA, así que lo
  demás que haya en los directorios no entra; si falta una, la parte no se calcula (nunca mezcla) y sus frases
  salen como «sin cadenas»;
- la L_acc de los runs del paper (`teff_bayes.entrada_margen`: G3 si hay detección, la cota de E3 si no) y su
  config (la relación L_acc–L_Hα), de `--project-root`, la raíz que contiene `runs/`;
- con `--tex`, el `main.tex` del paper: busca cada frase (sin comentarios y con los blancos colapsados) y dice en
  qué línea está, o que no está. Sin `--tex`, solo imprime las frases.

Cada cifra es un `Cifra`: la frase del paper con `«campo»` donde va cada número, cómo se calcula cada campo y,
si la frase dice algo cualitativo de las cadenas (un signo, quién es mayor, que dos intervalos se solapan), una
`guarda` que lo comprueba. Las frases son las de `paper/main.tex` del 2026-10-26 (tras las decisiones E y el
abstract de 299 palabras); si el autor cambia la redacción de una, hay que cambiarla aquí: la frase es la prueba
de que el número va donde dice. Las convenciones de redondeo son las que reproducen lo impreso (las de la pasada
T3 del 2026-10-23, `~/muse_sesion_20261023/paper_t3/validacion.md`): T_eff «at the adopted A_V» a 50 K y el resto
de rangos de T_eff a 10 K; los rangos de dispersión y de edades hacia fuera; «within N K» y «below N K» hacia
arriba; las diferencias entre conjuntos de datos con BT-Settl a 5 K; Δ con dos decimales y signo (−0.00 se
escribe 0.00). Una cifra que no sale con ninguna definición lleva `discrepancia` con el porqué: se informa, no se
ajusta.

Las cadenas del paper (C4 v2 `decoupled_c0`, 2026-10-24) son las 167 de `~/muse_noche_20261024_c4/cadenas`
más 22 que no leen MUSE y no se re-corrieron (18 solo IR y 4 `evo_solo`), que siguen en sus carpetas de octubre:

    python scripts/teff_cifras_texto.py --project-root RAIZ_CON_RUNS \\
        --cadenas ~/muse_noche_20261024_c4/cadenas \\
        --cadenas ~/muse_sesion_20261018/teff_datos_logg5/res --cadenas ~/muse_sesion_20261016/teff_datos/res_drift \\
        --cadenas ~/muse_noche_20261021/res --cadenas ~/muse_sesion_20261014/teff_mcmc/res \\
        --tex RAIZ_CON_RUNS/paper/main.tex [--json SALIDA] [--md SALIDA]

(el orden importa: las carpetas de octubre tienen también las versiones viejas de cadenas re-corridas, que
quedan sombreadas; el informe dice cuántas cadenas usadas vienen de cada carpeta: 167, 6, 6, 6 y 4). Sale con 1
si, con `--tex`, alguna frase no está en el paper (salvo las de `discrepancia` conocida), alguna guarda falla,
alguna frase cambia de sentido o falta alguna cadena.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from musepipe.constants import spt_code  # noqa: E402
from scripts import teff_bayes as tb  # noqa: E402

#: los dos objetos del paper (`targets/<slug>.json` con `paper`, en el orden de `paper.order`): O1, el compañero
#: con la línea detectada (el «12b» de las claves de las cuentas, que siguen los nombres de la sesión), y O2, el de
#: la cota (el «42b»). Los fija `fija_objetos` desde targets/: ningún nombre de objeto va en el código
#: (`tests/test_no_hardcoded_target.py`); las frases sí los llevan, porque son el texto del paper.
SLUGS: tuple[str, ...] = ()
O1 = O2 = ""
#: el nombre de cada objeto en el texto (`paper.display`, con ~)
NOMBRES: dict[str, str] = {}
#: las tres elecciones de A_V de las cadenas: fija 1.8, fija 3.0 y con el prior de la primaria
AVS = ("1.8", "3", "prior")
#: el sistemático de la T_eff que adopta el paper («we take ±150 K as the systematic error of the method»): una
#: elección del autor, no una cifra de las cadenas
SISTEMATICO_K = 150.0
#: las 12 plantillas jóvenes de tipo conocido ajustadas como datos (Tabla de bibliotecas del paper)
PLANTILLAS = ("2MASS0046", "2MASS1207", "CHSM17173", "J11195652-7504529", "LM601", "LM717", "TWA26", "TWA29",
              "USco_J16073798-2242471", "USco_J16104713-2239496", "VISTA_J16072381-2211021", "WISE0528")
#: con cuántas palabras se escribe un recuento en el texto
PALABRAS = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine"}


def fija_objetos(root=ROOT):
    """Fija `SLUGS`, `O1`, `O2` y `NOMBRES` desde `targets/` (los que declaran `paper`, por `paper.order`)."""
    global SLUGS, O1, O2
    from musepipe.targets import targets_dir
    fichas = [(json.loads(f.read_text()), f.stem) for f in sorted(targets_dir(root).glob("*.json"))]
    fichas = sorted(((d["paper"]["order"], s, d) for d, s in fichas if "paper" in d))
    if len(fichas) != 2:
        raise SystemExit(f"el paper tiene dos objetos y targets/ declara {len(fichas)} con `paper`")
    SLUGS = tuple(s for _, s, _ in fichas)
    O1, O2 = SLUGS
    NOMBRES.clear()
    NOMBRES.update({s: d["paper"]["display"].replace(" ", "~") for _, s, d in fichas})
    return SLUGS


# ================================================================== formato (las convenciones del texto)
def r_mitad(x, paso):
    """Redondeo al múltiplo de `paso` más cercano, la mitad hacia arriba."""
    return int(math.floor(x / paso + 0.5) * paso)


def dec(x, nd):
    """`x` con `nd` decimales; −0.00 se escribe 0.00."""
    s = f"{x:.{nd}f}"
    return s[1:] if s.startswith("-") and float(s) == 0 else s


def con_signo(x, nd):
    """Con signo; ±0.00 se escribe 0.00 (como el texto)."""
    s = f"{x:+.{nd}f}"
    return dec(0.0, nd) if float(s) == 0 else s


def rango(a, b, fa, fb=None):
    r"""`\(a\)--\(b\)`; si los dos extremos se escriben igual, un solo valor."""
    fb = fb or fa
    if fa(a) == fb(b):
        return rf"\({fa(a)}\)"
    return rf"\({fa(a)}\)--\({fb(b)}\)"


def hasta(a, b, fa):
    r"""`\(a\) to \(b\)`."""
    return rf"\({fa(a)}\) to \({fa(b)}\)"


def por_ciento(a, b):
    r"""`\(a\)--\(b\%\)` (enteros)."""
    return rango(a, b, i0)[:-2] + r"\%\)"


k5 = lambda x: str(r_mitad(x, 5))  # noqa: E731
k10 = lambda x: str(r_mitad(x, 10))  # noqa: E731
k50 = lambda x: str(r_mitad(x, 50))  # noqa: E731
d1 = lambda x: dec(x, 1)  # noqa: E731
d2 = lambda x: dec(x, 2)  # noqa: E731
s2 = lambda x: con_signo(x, 2)  # noqa: E731
i0 = lambda x: str(int(math.floor(x + 0.5)))  # noqa: E731


def abajo(p):
    """Hacia abajo al múltiplo de `p`."""
    return lambda x: str(int(math.floor(x / p) * p))


def arriba(p):
    """Hacia arriba al múltiplo de `p`."""
    return lambda x: str(int(math.ceil(x / p - 1e-9) * p))


def cota_mag(x):
    """«does not exceed … by more than X mag»: hacia arriba al entero si x ≤ 1, y a 0.1 si pasa de 1 (1.07 → 1.1;
    el entero, 2, sería una cota floja)."""
    return str(math.ceil(x)) if x <= 1 else d1(math.ceil(x * 10 - 1e-9) / 10)


# ================================================================== piezas puras
def se_solapan(a, b):
    """¿Se solapan los intervalos cerrados `a` = (lo, hi) y `b`? (tocarse cuenta como solaparse)."""
    return max(a[0], b[0]) <= min(a[1], b[1])


def frase_solape68(sol):
    """La frase de los intervalos del 68 % de las dos rejillas (apéndice B), de {(slug, 'teff'|'logg'): bool}.
    Solo sabe escribir los casos «ninguno» y «los dos» por compañero; otro caso es otra frase (`ValueError`)."""
    trozos = []
    for s in SLUGS:
        t, g = sol[(s, "teff")], sol[(s, "logg")]
        nombre = NOMBRES[s]
        if not t and not g:
            trozos.append(("no", f"do not overlap in temperature or gravity for {nombre}"))
        elif t and g:
            trozos.append(("si", f"overlap in both for {nombre}"))
        else:
            raise ValueError(f"{s}: el 68 % se solapa solo en {'T_eff' if t else 'log g'}: la frase cambia")
    if trozos[0][0] == trozos[1][0]:
        raise ValueError("los dos compañeros dan lo mismo: la frase cambia")
    return f"{trozos[0][1]}, and they {trozos[1][1]}"


def lista_por_debajo(negativos):
    r"""«it is below it only with BT-Settl at \(A_V = 1.8\)~mag or with the prior on \(A_V\), and with AMES-Cond at
    \(1.8\) and \(3.0\)~mag», de [(nombre de la rejilla, '1.8'|'3'|'prior'), ...] en el orden de las rejillas. La
    primera rejilla nombra la A_V («\(A_V = 1.8\)~mag»); las siguientes, solo los valores."""
    grupos: dict[str, list[str]] = {}
    for g, av in negativos:
        grupos.setdefault(g, [])
        if av not in grupos[g]:
            grupos[g].append(av)
    partes = []
    for i, (g, avs) in enumerate(grupos.items()):
        vals = [f"{float(a):.1f}" for a in AVS[:2] if a in avs]
        if not vals:
            partes.append(rf"with {g} with the prior on \(A_V\)")
            continue
        if i == 0 and len(vals) == 1:
            txt = rf"\(A_V = {vals[0]}\)~mag"
        else:
            txt = " and ".join(rf"\({v}\)" for v in vals) + "~mag"
        if "prior" in avs:
            txt += r" or with the prior on \(A_V\)"
        partes.append(f"with {g} at {txt}")
    return ", and ".join(partes)


def tipo_tabla(t):
    r"""El tipo equivalente como va en la Tabla B.5: `M7.5--M8`, `M8` o `\(\geq\)L1`."""
    if t["clase"] == "par":
        return "--".join(t["tipos"])
    if t["clase"] == "una":
        return t["tipos"][0]
    return (r"\(\geq\)" if t["clase"] == "posterior" else r"\(\leq\)") + t["tipos"][0]


def tipo_prosa(t):
    """El tipo equivalente como va en el texto: `M7.5--M8`, `M8` o `L1 or later`."""
    if t["clase"] in ("par", "una"):
        return tipo_tabla(t)
    return t["tipos"][0] + (" or later" if t["clase"] == "posterior" else " or earlier")


def normaliza(texto):
    """(el texto sin comentarios de LaTeX y con los blancos colapsados en un espacio, y la posición en el texto
    original de cada carácter del normalizado) para buscar las frases y decir su línea."""
    chars, pos, off = [], [], 0
    for linea in texto.split("\n"):
        m = re.search(r"(?<!\\)%", linea)
        util = linea if m is None else linea[:m.start()]
        for i, ch in enumerate(util):
            if ch.isspace():
                if chars and chars[-1] != " ":
                    chars.append(" ")
                    pos.append(off + i)
            else:
                chars.append(ch)
                pos.append(off + i)
        if chars and chars[-1] != " ":
            chars.append(" ")
            pos.append(off + len(linea))
        off += len(linea) + 1
    return "".join(chars), pos


def colapsa(s):
    return re.sub(r"\s+", " ", s).strip()


# ================================================================== cadenas
class Faltan(Exception):
    """Faltan cadenas de una parte (sus etiquetas en `args[0]`)."""


class Cadenas:
    """Las cadenas de uno o varios directorios, por etiqueta. Gana la primera; las repetidas, a `sombreadas`."""

    def __init__(self, dirs):
        self.dirs = list(dirs)
        self.por_etiqueta: dict[str, dict] = {}
        self.sombreadas: list[dict] = []
        self.usadas: set[str] = set()
        self._delta: dict[str, np.ndarray] = {}
        for dd in dirs:
            for raiz, _, ficheros in sorted(os.walk(Path(dd).expanduser(), followlinks=True)):
                for nombre in sorted(ficheros):
                    if not nombre.endswith(".json"):
                        continue
                    f = Path(raiz) / nombre
                    z = f.with_suffix(".npz")
                    try:
                        d = json.loads(f.read_text())
                    except (OSError, ValueError):
                        continue
                    if not isinstance(d, dict) or d.get("etiqueta") != f.stem or "slug" not in d or not z.exists():
                        continue
                    if d["etiqueta"] in self.por_etiqueta:
                        self.sombreadas.append(dict(etiqueta=d["etiqueta"], usada=self.por_etiqueta[d["etiqueta"]]["_json"],
                                                    sombreada=str(f)))
                        continue
                    d["_json"], d["_npz"] = str(f), str(z)
                    self.por_etiqueta[d["etiqueta"]] = d

    def exige(self, etiquetas):
        falta = [e for e in etiquetas if e not in self.por_etiqueta]
        if falta:
            raise Faltan(falta)
        self.usadas.update(etiquetas)
        return [self.por_etiqueta[e] for e in etiquetas]

    def procedencia(self):
        """Cuántas de las cadenas usadas vienen de cada directorio de `--cadenas`."""
        out = {str(d): 0 for d in self.dirs}
        for e in self.usadas:
            f = Path(self.por_etiqueta[e]["_json"])
            for d in self.dirs:
                if str(f).startswith(str(Path(d).expanduser()) + os.sep):
                    out[str(d)] += 1
                    break
        return out

    def fuera_de_la_primera(self):
        """Las cadenas usadas que no vienen del primer directorio, por lo que leen: 'solo IR', 'sin espectro'
        (`evo_solo`) u 'otras' (las que leen MUSE: si el primer directorio es el del último re-corrido, estas
        deberían ser cero)."""
        primero = str(Path(self.dirs[0]).expanduser()) + os.sep if self.dirs else None
        out: dict[str, int] = {}
        for e in self.usadas:
            d = self.por_etiqueta[e]
            if primero and d["_json"].startswith(primero):
                continue
            k = "sin espectro" if d.get("modo") == "evo_solo" else "solo IR" if d.get("datos") == "ir" else "otras"
            out[k] = out.get(k, 0) + 1
        return out

    def delta(self, d, entradas):
        """Δ por muestra (`teff_bayes.margen_de_corrida`), solo las finitas; en caché por etiqueta."""
        e = d["etiqueta"]
        if e not in self._delta:
            mg = tb.margen_de_corrida(d, entradas[d["slug"]])
            self._delta[e] = mg[np.isfinite(mg)]
        return self._delta[e]


def etiqueta(slug, rejilla="btsettl", av="prior", r250=False, datos="todo", semilla=1):
    """La etiqueta de `teff_bayes.py ajusta` de una corrida `libre` con el prior de la primaria."""
    return "_".join(x for x in (slug, None if rejilla == "btsettl" else rejilla, "libre", "avprimaria",
                                None if av == "prior" else f"avfija{av}", None if datos == "todo" else f"solo{datos}",
                                "r250" if r250 else None, f"s{semilla}") if x)


def clave_av(d):
    return "prior" if d.get("av_fija") is None else f"{d['av_fija']:g}"


# ================================================================== las cuentas
def rejillas_b7(K: Cadenas, slug):
    """Las (rejilla, R = 250) de la Tabla B.7 de un compañero: las de `teff_bayes.REJILLAS_TABLA`, a R = 500 y a
    R = 250, de las que hay alguna de sus tres cadenas (con A_V 1.8, 3.0 y con prior); después se exigen las
    tres. BT-Dusty no llega a la T_eff de 42B b ni Diamondback a la de 12 b, así que cada uno tiene las suyas."""
    return [(rj, r) for rj in tb.REJILLAS_TABLA for r in (False, True)
            if any(etiqueta(slug, rj, a, r250=r) in K.por_etiqueta for a in AVS)]


def unica_por_patron(K: Cadenas, patron):
    """La única cadena cuya etiqueta casa con `patron` (las inyecciones: sus parámetros van en la etiqueta)."""
    ets = sorted(e for e in K.por_etiqueta if re.fullmatch(patron, e))
    if len(ets) != 1:
        raise Faltan([f"una cadena que case con {patron} (hay {len(ets)}: {', '.join(ets)})"])
    return ets


class Calculo:
    """Las cifras de las cadenas, por partes (los nombres y las cuentas de `cifras_t3.py` de la sesión del
    2026-10-25, salvo donde se dice). `self.c[parte]` es un dict, o {'faltan': [...]} si faltan cadenas."""

    def __init__(self, cadenas: Cadenas, root: Path):
        self.K, self.root = cadenas, Path(root)
        fija_objetos(self.root)
        self.b7 = {s: rejillas_b7(self.K, s) for s in SLUGS}
        self.E = {s: tb.entrada_margen(s, root=self.root) for s in SLUGS}
        self.c: dict = {"objetos": dict(slugs=list(SLUGS), nombres=dict(NOMBRES)), "entrada_margen": self.E}
        for nombre in ("b6", "semillas", "cruce_y_avlibre", "rejillas", "datos", "ames_cond_datos", "evo",
                       "inyecciones", "plantillas", "nlive", "ejes"):
            try:
                self.c[nombre] = getattr(self, "_" + nombre)()
            except Faltan as exc:
                self.c[nombre] = {"faltan": exc.args[0]}

    # ------------------------------------------------------------------ utilidades
    def dl(self, d):
        return self.K.delta(d, self.E)

    def mediana_delta(self, d):
        return tb.percentiles(self.dl(d))["p50"]

    def b6(self):
        """(slug, rejilla, av) -> cadena de las 12 de la Tabla B.6 (semilla 1)."""
        ets = [etiqueta(s, g, a) for s in SLUGS for g in ("btsettl", "drift") for a in AVS]
        return {(d["slug"], d.get("rejilla", "btsettl"), clave_av(d)): d for d in self.K.exige(ets)}

    # ------------------------------------------------------------------ Tablas B.5 y B.6 y su texto
    def _b6(self):
        t = self.b6()
        r = {}
        T = lambda s, g, a: t[(s, g, a)]["posterior"]["teff"]["p50"]  # noqa: E731
        # Tabla B.5: T_eff y A_V (p50); log g = [mín. p16, máx. p84] de las tres A_V; Δ (p50)
        for s in SLUGS:
            for g in ("btsettl", "drift"):
                P = {a: t[(s, g, a)]["posterior"] for a in AVS}
                r[f"B5_{s}_{g}"] = dict(
                    teff={a: P[a]["teff"]["p50"] for a in AVS}, av_prior=P["prior"]["av"]["p50"],
                    logg=[min(P[a]["logg"]["p16"] for a in AVS), max(P[a]["logg"]["p84"] for a in AVS)],
                    delta={a: self.mediana_delta(t[(s, g, a)]) for a in AVS},
                    semiancho_teff_max=max(max(P[a]["teff"]["p84"] - P[a]["teff"]["p50"],
                                               P[a]["teff"]["p50"] - P[a]["teff"]["p16"]) for a in AVS))
        r["semiancho_teff_max_global"] = max(v["semiancho_teff_max"] for k, v in r.items() if k.startswith("B5_"))
        # correlaciones en las 4 cadenas con prior: T–A_V y Δ–A_V; en 12 b además R–A_V y r_K–A_V
        cor = {}
        for s in SLUGS:
            for g in ("btsettl", "drift"):
                d = t[(s, g, "prior")]
                z = np.load(d["_npz"])
                nom = [str(x) for x in z["nombres"]]
                av = z["muestras"][:, nom.index("av")]
                mg = tb.margen_de_corrida(d, self.E[s])
                cc = lambda a, b: float(np.corrcoef(a, b)[0, 1])  # noqa: E731
                cor[f"{s}_{g}"] = dict(T_av=cc(z["derivadas"][:, 0], av), delta_av=cc(mg, av),
                                       R_av=cc(z["muestras"][:, nom.index("r_rj")], av),
                                       rk_av=cc(z["muestras"][:, nom.index("rk")], av))
        r["correlaciones"] = cor
        ta = [v for c in cor.values() for kk, v in c.items() if kk in ("T_av", "delta_av")]
        r["corr_T_delta_rango"] = [min(ta), max(ta)]
        r["corr_12b_R_rk"] = [cor[f"{O1}_{g}"][k] for g in ("btsettl", "drift") for k in ("R_av", "rk_av")]
        # intervalos del 68 y del 95 % de T_eff y log g con prior (semilla 1)
        sol = {}
        for s in SLUGS:
            Pb, Pd = t[(s, "btsettl", "prior")]["posterior"], t[(s, "drift", "prior")]["posterior"]
            for p in ("teff", "logg"):
                b, dd = Pb[p], Pd[p]
                sol[f"{s}_{p}"] = dict(bt68=[b["p16"], b["p84"]], drift68=[dd["p16"], dd["p84"]],
                                       bt95=[b["p2_5"], b["p97_5"]], drift95=[dd["p2_5"], dd["p97_5"]],
                                       solapan68=se_solapan((b["p16"], b["p84"]), (dd["p16"], dd["p84"])),
                                       solapan95=se_solapan((b["p2_5"], b["p97_5"]), (dd["p2_5"], dd["p97_5"])))
        r["solapes_prior_s1"] = sol
        r["42b_T_18_30"] = [min(T(O2, g, a) for g in ("btsettl", "drift") for a in AVS),
                            max(T(O2, g, a) for g in ("btsettl", "drift") for a in AVS)]
        r["42b_T_18_30_sin_prior"] = [min(T(O2, g, a) for g in ("btsettl", "drift") for a in AVS[:2]),
                                      max(T(O2, g, a) for g in ("btsettl", "drift") for a in AVS[:2])]
        r["42b_T_todas"] = [T(O2, g, a) for g in ("btsettl", "drift") for a in AVS]
        r["42b_av_prior"] = [t[(O2, g, "prior")]["posterior"]["av"]["p50"] for g in ("btsettl", "drift")]
        r["12b_T_18"] = [T(O1, g, "1.8") for g in ("btsettl", "drift")]
        r["12b_T_30_prior"] = [T(O1, g, a) for g in ("btsettl", "drift") for a in ("3", "prior")]
        r["12b_av_prior"] = [t[(O1, g, "prior")]["posterior"]["av"]["p50"] for g in ("btsettl", "drift")]
        r["logl_max_bt_menos_drift_prior"] = {s: t[(s, "btsettl", "prior")]["logl_max"] - t[(s, "drift", "prior")]["logl_max"]
                                             for s in SLUGS}
        fb = {g: [100 * math.exp(t[(s, g, a)]["posterior"]["lnf_vis"]["p50"]) for s in SLUGS for a in AVS]
              for g in ("btsettl", "drift")}
        r["fb_muse"] = {g: [min(v), max(v)] for g, v in fb.items()}
        r["fb_muse_todas"] = fb
        r["12b_18_rk"] = [t[(O1, g, "1.8")]["posterior"]["rk"]["p50"] for g in ("btsettl", "drift")]
        r["12b_18_R"] = [t[(O1, g, "1.8")]["posterior"]["r_rj"]["p50"] for g in ("btsettl", "drift")]
        dl = {(s, g, a): self.mediana_delta(t[(s, g, a)]) for s in SLUGS for g in ("btsettl", "drift") for a in AVS}
        r["delta"] = {f"{s}_{g}_{a}": v for (s, g, a), v in dl.items()}
        r["12b_dif_rejillas_av_fija"] = [abs(dl[(O1, "btsettl", a)] - dl[(O1, "drift", a)]) for a in AVS[:2]]
        r["12b_delta_30_prior"] = [dl[(O1, g, a)] for g in ("btsettl", "drift") for a in ("3", "prior")]
        r["42b_delta_todas"] = [min(v for (s, g, a), v in dl.items() if s == O2),
                                max(v for (s, g, a), v in dl.items() if s == O2)]
        Rs = [t[(O1, g, a)]["posterior"]["r_rj"]["p50"] for g in ("btsettl", "drift") for a in AVS]
        r["12b_R_rango"] = [min(Rs), max(Rs)]
        r["dlnz_30_mayor_que_18"] = {f"{s}_{g}": t[(s, g, "3")]["logz"] > t[(s, g, "1.8")]["logz"]
                                     for s in SLUGS for g in ("btsettl", "drift")}
        # Δ con la L_bol publicada en vez de la del ajuste, a la T_eff de BT-Settl a 1.8 (12 b)
        e12 = self.E[O1]
        T18 = T(O1, "btsettl", "1.8")
        r["12b_delta_Lbol_publicada_T18"] = dict(
            teff=T18, delta=math.log10(e12["lacc0"] / e12["lbol_lit"]) - float(tb.suelo(T18)))
        # el límite de 42B b con la L_bol publicada en los extremos del rango de T_eff (A_V fija)
        e42 = self.E[O2]
        rat42 = math.log10(e42["lacc0"] / e42["lbol_lit"])
        r["42b_limite_Lbol_publicada"] = [rat42 - float(tb.suelo(x)) for x in r["42b_T_18_30_sin_prior"]]
        # log(L_Hα/L_bol) de 12 b: la L_Hα que da la L_acc de G3 con la relación de la config, invertida
        from musepipe.config import load_run_config
        from musepipe.stages.stage_h03_limits import physical_inputs_from_config
        from scripts import tipo_ir_indices as tir
        perfil = tir.perfil(O1, root=self.root)
        fis = physical_inputs_from_config(load_run_config(perfil["paper"]["run"], project_root=self.root).config)
        r["12b_log_lha_lbol_av0"] = ((math.log10(e12["lacc0"]) - float(fis["lacc_lha_b"])) / float(fis["lacc_lha_a"])
                                     - float(perfil["fisica"]["log_lbol"]))
        r["nlive"] = {f"{s}_{g}_{a}": t[(s, g, a)]["nlive"] for (s, g, a) in t}
        r["teff_post"] = {f"{s}_{g}_{a}": d["posterior"]["teff"] for (s, g, a), d in t.items()}
        return r

    # ------------------------------------------------------------------ semillas 2 y 3
    def _semillas(self):
        """Por cadena de la Tabla B.6, la diferencia máxima entre las semillas 1–3 de ln Z y de las medianas, y si
        esa diferencia es menor que el intervalo 16–84 % de cada semilla («the medians change by less than their
        intervals»: T_eff siempre; A_V con el prior)."""
        t = self.b6()
        ets = [etiqueta(s, g, a, semilla=k) for s in SLUGS for g in ("btsettl", "drift") for a in AVS for k in (2, 3)]
        grupos = {k: {1: d} for k, d in t.items()}
        for d in self.K.exige(ets):
            grupos[(d["slug"], d.get("rejilla", "btsettl"), clave_av(d))][d["semilla"]] = d
        filas = []
        for k, g in sorted(grupos.items()):
            ss = sorted(g)
            P = lambda s, x, q="p50": g[s]["posterior"][x][q]  # noqa: E731
            f = dict(clave="_".join(k), logz=[g[s]["logz"] for s in ss], teff=[P(s, "der_teff") for s in ss],
                     av=[P(s, "av") for s in ss], delta=[float(np.median(self.dl(g[s]))) for s in ss],
                     ancho_teff=[P(s, "der_teff", "p84") - P(s, "der_teff", "p16") for s in ss],
                     ancho_av=[P(s, "av", "p84") - P(s, "av", "p16") for s in ss])
            f["ptp"] = {x: float(np.ptp(f[x])) for x in ("teff", "av", "delta", "logz")}
            f["menor_que_intervalo"] = {x: bool(np.ptp(f[x]) < min(w)) for x, w in
                                        (("teff", f["ancho_teff"]), ("av", f["ancho_av"])) if x != "av" or k[2] == "prior"}
            filas.append(f)
        pr = [f for f in filas if f["clave"].endswith("prior")]
        fi = [f for f in filas if not f["clave"].endswith("prior")]
        return dict(filas=filas, max_ptp_teff=max(f["ptp"]["teff"] for f in filas),
                    max_ptp_av_prior=max(f["ptp"]["av"] for f in pr), max_ptp_delta=max(f["ptp"]["delta"] for f in filas),
                    max_ptp_logz_fija=max(f["ptp"]["logz"] for f in fi), max_ptp_logz_prior=max(f["ptp"]["logz"] for f in pr),
                    medianas_menores_que_intervalos=all(all(f["menor_que_intervalo"].values()) for f in filas),
                    av_prior_12b_todas=[a for f in pr if f["clave"].startswith(f"{O1}_") for a in f["av"]])

    # ------------------------------------------------------------------ cruce de signo de Δ y A_V libre
    def _cruce_y_avlibre(self):
        """El cruce de Δ por cero, interpolado en A_V: BT-Settl por 1.8 / 2.5 / 3.0 (la cadena a 2.5 es solo de
        BT-Settl), DRIFT-PHOENIX por 1.8 / 3.0. Y las cuatro cadenas con A_V libre."""
        t = self.b6()
        r = {}
        for d in self.K.exige([etiqueta(s, av="2.5") for s in SLUGS]):
            s = d["slug"]
            pts = [(1.8, float(np.median(self.dl(t[(s, "btsettl", "1.8")])))), (2.5, float(np.median(self.dl(d)))),
                   (3.0, float(np.median(self.dl(t[(s, "btsettl", "3")]))))]
            r[f"{s}_btsettl"] = dict(puntos=pts, cruce=cruce_lineal(pts))
        for s in SLUGS:
            pts = [(1.8, float(np.median(self.dl(t[(s, "drift", "1.8")])))), (3.0, float(np.median(self.dl(t[(s, "drift", "3")]))))]
            r[f"{s}_drift"] = dict(puntos=pts, cruce=cruce_lineal(pts))
        libres = self.K.exige([f"{s}{g}_libre_avlibre_s0" for s in SLUGS for g in ("", "_drift")])
        lnz_max = {s: max(d["logz"] for k, d in t.items() if k[0] == s) for s in SLUGS}
        r["avlibre"] = {f"{d['slug']}_{d.get('rejilla', 'btsettl')}": dict(
            av=d["posterior"]["av"]["p50"], teff=d["posterior"]["der_teff"]["p50"],
            teff_p97_5=d["posterior"]["der_teff"]["p97_5"], dlnz_vs_mejor_B6=d["logz"] - lnz_max[d["slug"]],
            borde=tb.en_el_borde(d),
            nlive=d["nlive"]) for d in libres}
        return r

    # ------------------------------------------------------------------ Tabla B.7 y su párrafo
    def _rejillas(self):
        filas_por = {}
        ets = [etiqueta(s, rj, a, r250=r) for s in SLUGS for rj, r in self.b7[s] for a in AVS]
        for d in self.K.exige(ets):
            P = d["posterior"]
            q = tb.percentiles(self.dl(d))
            filas_por.setdefault((d["slug"], clave_av(d)), []).append(dict(
                etiqueta=d["etiqueta"], rejilla=d.get("rejilla", "btsettl"), r=d.get("r_analisis", 500), av=clave_av(d),
                teff=P["teff"]["p50"],
                semiancho=(P["teff"]["p84"] - P["teff"]["p16"]) / 2, delta=q["p50"], logz=d["logz"],
                borde=tb.en_el_borde(d), delta_semiancho=(q["p84"] - q["p16"]) / 2, nlive=d["nlive"]))
        out = {}
        for (s, a), filas in filas_por.items():
            T = [f["teff"] for f in filas]
            ref = {f["r"]: f["logz"] for f in filas if f["rejilla"] == "btsettl"}
            out[f"{s}_{a}"] = dict(teff=[min(T), max(T)], dispersion=max(T) - min(T),
                                   semiancho=[min(f["semiancho"] for f in filas), max(f["semiancho"] for f in filas)],
                                   delta=[min(f["delta"] for f in filas), max(f["delta"] for f in filas)],
                                   dlnz_vs_bt={f"{f['rejilla']}@{f['r']}": f["logz"] - ref[f["r"]]
                                               for f in filas if f["rejilla"] != "btsettl"},
                                   filas=filas)
        f12 = {a: out[f"{O1}_{a}"] for a in AVS}
        f42 = {a: out[f"{O2}_{a}"] for a in AVS}
        otras12 = [v for a in AVS for v in f12[a]["dlnz_vs_bt"].values()]
        db = lambda a: [v for k, v in f42[a]["dlnz_vs_bt"].items() if k.startswith("diamondback")]  # noqa: E731
        t_db = [f["teff"] for f in f42["prior"]["filas"] if f["rejilla"].startswith("diamondback")]
        d42 = [(f["rejilla"], f["r"], a, f["delta"]) for a in AVS for f in f42[a]["filas"]]
        d42_otras = [x for x in d42 if x[0] not in ("btsettl", "drift")]
        # la dispersión de cada A_V y compañero, y la rejilla cuya T_eff la fija más (la que, quitada, más la acorta)
        disp = {}
        for k, v in out.items():
            T = {f["etiqueta"]: f["teff"] for f in v["filas"]}
            quita = {e: v["dispersion"] - (max(x for ee, x in T.items() if ee != e) - min(x for ee, x in T.items() if ee != e))
                     for e in T}
            e_max = max(quita, key=quita.get)
            f_max = next(f for f in v["filas"] if f["etiqueta"] == e_max)
            disp[k] = dict(dispersion=v["dispersion"], rejilla_extrema=f_max["rejilla"], teff_extrema=f_max["teff"])
        lnz42_r500 = {f"{f['rejilla']}": f["logz"] for f in f42["prior"]["filas"] if f["r"] == 500}
        return dict(
            por_av={k: {kk: vv for kk, vv in v.items() if kk != "filas"} for k, v in out.items()}, filas=out,
            n_rejillas={s: len({rj.split("1600_")[0] for rj, _ in self.b7[s]}) for s in SLUGS},
            n_rejillas_total=len({rj.split("1600_")[0] for s in SLUGS for rj, _ in self.b7[s]}),
            bt_mejor_12b_tres_av=all(max(f12[a]["filas"], key=lambda f: f["logz"])["rejilla"] == "btsettl" for a in AVS),
            otras_12b_dlnz=[min(otras12), max(otras12)],
            diamondback_dlnz_prior=[min(db("prior")), max(db("prior"))],
            diamondback_dlnz_fija=[min(db("1.8") + db("3")), max(db("1.8") + db("3"))],
            db_mayor_que_bt_todas=all(v > 0 for a in AVS for v in db(a)),
            diamondback_teff_prior=[min(t_db), max(t_db)],
            db_teff_mas_baja=all(min(t_db) < f["teff"] for f in f42["prior"]["filas"] if not f["rejilla"].startswith("diamondback")),
            db_mayor_evidencia_42b_prior_r500=max(lnz42_r500, key=lnz42_r500.get),
            diamondback_borde_logg_prior=all("logg" in f["borde"] for f in f42["prior"]["filas"]
                                             if f["rejilla"].startswith("diamondback")),
            dispersion=disp,
            semiancho_global=[min(v["semiancho"][0] for v in out.values()), max(v["semiancho"][1] for v in out.values())],
            delta_12b_18=f12["1.8"]["delta"], delta_12b_30=f12["3"]["delta"],
            delta_12b_18_absmax=max(abs(f["delta"]) for f in f12["1.8"]["filas"]),
            delta_42b_todas=[min(x[3] for x in d42), max(x[3] for x in d42)],
            delta_42b_negativos=[(x[0], x[2]) for x in sorted(d42, key=lambda x: (orden_rejilla(x[0]), AVS.index(x[2])))
                                 if x[3] < 0],
            delta_42b_sin_bt_drift=[min(x[3] for x in d42_otras), max(x[3] for x in d42_otras)],
            delta_42b_sin_bt_drift_min_en=min(d42_otras, key=lambda x: x[3])[:3],
            delta_intervalo_semiancho=[min(f["delta_semiancho"] for v in out.values() for f in v["filas"]),
                                       max(f["delta_semiancho"] for v in out.values() for f in v["filas"])],
            sin_borde_r500_ambos={g: not any(f["borde"] for s in SLUGS for f in out[f"{s}_prior"]["filas"]
                                             if f["rejilla"] == g and f["r"] == 500) and all(
                any(f["rejilla"] == g and f["r"] == 500 for f in out[f"{s}_prior"]["filas"]) for s in SLUGS)
                for g in ("ames_dusty", "ames_cond", "bt_dusty", "sphinx")},
            nlive_bt500_fija=sorted({f["nlive"] for v in out.values() for f in v["filas"]
                                     if f["rejilla"] == "btsettl" and f["r"] == 500 and f["av"] != "prior"}),
            nlive_resto=sorted({f["nlive"] for v in out.values() for f in v["filas"]
                                if not (f["rejilla"] == "btsettl" and f["r"] == 500 and f["av"] != "prior")}),
        )

    # ------------------------------------------------------------------ solo MUSE / solo IR (Fig. B.5)
    def _datos(self):
        t = self.b6()
        g = {(s, rj, a, "todo"): d for (s, rj, a), d in t.items()}
        ets = [etiqueta(s, rj, a, datos=x) for s in SLUGS for rj in ("btsettl", "drift") for a in AVS for x in ("muse", "ir")]
        for d in self.K.exige(ets):
            g[(d["slug"], d.get("rejilla", "btsettl"), clave_av(d), d["datos"])] = d
        P = lambda s, rj, a, x: g[(s, rj, a, x)]["posterior"]  # noqa: E731
        T = lambda s, rj, a, x: P(s, rj, a, x)["teff"]["p50"]  # noqa: E731
        dif = lambda s, rj, x: [T(s, rj, a, x) - T(s, rj, a, "todo") for a in AVS]  # noqa: E731
        ancho = lambda s, rj, a, x: P(s, rj, a, x)["logg"]["p84"] - P(s, rj, a, x)["logg"]["p16"]  # noqa: E731
        D = lambda s, rj, a, x: self.mediana_delta(g[(s, rj, a, x)])  # noqa: E731
        fb = lambda s, rj, a: 100 * math.exp(P(s, rj, a, "muse")["lnf_vis"]["p50"])  # noqa: E731
        mm = lambda xs: [min(xs), max(xs)]  # noqa: E731
        return dict(
            bt_max_abs_dif=max(abs(v) for s in SLUGS for x in ("ir", "muse") for v in dif(s, "btsettl", x)),
            bt_12b_ir=dif(O1, "btsettl", "ir"), bt_12b_muse=dif(O1, "btsettl", "muse"),
            bt_42b_ir=dif(O2, "btsettl", "ir"), bt_42b_muse=dif(O2, "btsettl", "muse"),
            bt_ancho_logg_muse=mm([ancho(s, "btsettl", a, "muse") for s in SLUGS for a in AVS]),
            bt_ancho_logg_todo=mm([ancho(s, "btsettl", a, "todo") for s in SLUGS for a in AVS]),
            drift_12b_ir_absmax=max(abs(v) for v in dif(O1, "drift", "ir")),
            drift_42b_ir_absmax=max(abs(v) for v in dif(O2, "drift", "ir")),
            drift_12b_muse=dif(O1, "drift", "muse"),
            # el error de modelo en MUSE de los ajustes solo a MUSE de ROXs 12 b (la frase habla de 12 b; con los
            # dos compañeros, 26–35 % y 14–29 %, se solapan: `autor.md` A.4 del 2026-10-25)
            fb_muse_solo_12b={rj: mm([fb(O1, rj, a) for a in AVS]) for rj in ("btsettl", "drift")},
            fb_muse_solo_bt_todos=mm([fb(s, "btsettl", a) for s in SLUGS for a in AVS]),
            av_prior={f"{s}_{rj}_{x}": P(s, rj, "prior", x)["av"]["p50"] for s in SLUGS for rj in ("btsettl", "drift")
                      for x in ("muse", "ir")},
            delta_12b_muse=mm([D(O1, rj, a, "muse") for rj in ("btsettl", "drift") for a in AVS]),
            delta_12b_ir_18=mm([D(O1, rj, "1.8", "ir") for rj in ("btsettl", "drift")]),
            delta_12b_ir_30_prior=mm([D(O1, rj, a, "ir") for rj in ("btsettl", "drift") for a in ("3", "prior")]),
            delta_42b_todo_conjunto=mm([D(O2, rj, a, x) for rj in ("btsettl", "drift") for a in AVS
                                        for x in ("muse", "ir")]),
            delta_12b_solos_18=mm([D(O1, rj, "1.8", x) for rj in ("btsettl", "drift") for x in ("muse", "ir")]),
            delta_12b_solos_30=mm([D(O1, rj, "3", x) for rj in ("btsettl", "drift") for x in ("muse", "ir")]),
        )

    # ------------------------------------------------------------------ AMES-Cond, solo MUSE / solo IR (Fig. B.6)
    def _ames_cond_datos(self):
        ets = [etiqueta(s, "ames_cond", a, datos=x) for s in SLUGS for a in AVS for x in ("muse", "ir")]
        ets += [etiqueta(s, "ames_cond") for s in SLUGS]
        g = {(d["slug"], clave_av(d), d.get("datos", "todo")): d for d in self.K.exige(ets)}
        P = lambda s, a, x, p, q="p50": g[(s, a, x)]["posterior"][p][q]  # noqa: E731
        m = [(s, a) for s in SLUGS for a in AVS]
        return dict(
            teff_prior={s: {x: P(s, "prior", x, "der_teff") for x in ("todo", "ir", "muse")} for s in SLUGS},
            logg_prior_12b={x: P(O1, "prior", x, "der_logg") for x in ("todo", "muse")},
            R_prior_42b={x: P(O2, "prior", x, "r_rj") for x in ("todo", "muse")},
            logg_p2_5_muse_max=max(P(s, a, "muse", "der_logg", "p2_5") for s, a in m),
            f_muse_solo=[min(100 * math.exp(P(s, a, "muse", "lnf_vis")) for s, a in m),
                         max(100 * math.exp(P(s, a, "muse", "lnf_vis")) for s, a in m)],
            delta_muse={f"{s}_{a}": float(np.median(self.dl(g[(s, a, "muse")]))) for s, a in m},
            ir_menos_todo_prior={s: P(s, "prior", "ir", "der_teff") - P(s, "prior", "todo", "der_teff") for s in SLUGS},
        )

    # ------------------------------------------------------------------ modo evolutivo (datos, sin espectro, plantillas)
    def _evo(self):
        r = {}
        for d in self.K.exige([f"{s}_evo_{k}_avprimaria_s0" for s in SLUGS for k in ("ATMO2020", "BHAC15")]):
            r[d["etiqueta"]] = dict(edad_myr=10 ** d["posterior"]["log_edad_myr"]["p50"])
        # sin espectro (`evo_solo`): la L_bol publicada y el prior de edad; Δ con la A_V adoptada (Tabla B.5)
        for d in self.K.exige([f"{s}_evo_solo_{k}_avprimaria_s0" for s in SLUGS for k in ("ATMO2020", "BHAC15")]):
            Tm = np.load(d["_npz"])["teff"]
            r[d["etiqueta"]] = dict(teff_p50=float(np.median(Tm)),
                                    teff_semiancho=float(np.percentile(Tm, 84) - np.percentile(Tm, 16)) / 2,
                                    delta_p50=float(np.median(self.dl(d))),
                                    fuera_de_la_rejilla=1.0 - float(d["fraccion_en_rejilla"]))
        # sobre plantillas jóvenes (las que se corrieron para cada compañero: su nombre va en la etiqueta)
        pl = {}
        for s in SLUGS:
            ets = sorted(e for e in self.K.por_etiqueta
                         if re.fullmatch(rf"{re.escape(s)}_evo_ATMO2020_avprimaria_pl-.+_s0", e))
            if not ets:
                raise Faltan([f"{s}_evo_ATMO2020_avprimaria_pl-*_s0"])
            pl[s] = [10 ** d["posterior"]["log_edad_myr"]["p50"] for d in self.K.exige(ets)]
        r["plantillas"] = {s: [min(v), max(v)] for s, v in pl.items()}
        r["n_plantillas"] = {s: len(v) for s, v in pl.items()}
        return r

    # ------------------------------------------------------------------ inyección de un modelo de la rejilla
    def _inyecciones(self):
        """Un modelo de BT-Settl inyectado con la S/N de cada compañero, ajustado con las dos rejillas: una cadena
        por compañero y rejilla (`--inyecta-bt T LOGG AV R`, que van en la etiqueta)."""
        ets = [e for s in SLUGS for pre in ("", "_drift")
               for e in unica_por_patron(self.K, rf"{re.escape(s)}{pre}_libre_avprimaria_bt-[0-9.-]+_s0")]
        r = {d["etiqueta"]: d["posterior"]["der_teff"]["p50"] - d["verdad"]["teff"] for d in self.K.exige(ets)}
        return dict(errores=r, max_abs=max(abs(v) for v in r.values()))

    # ------------------------------------------------------------------ plantillas (tipo equivalente, ±150 K, …)
    def _plantillas(self):
        t = self.b6()
        cal = {}
        for s in SLUGS:
            for rej, pre in (("btsettl", ""), ("drift", "_drift")):
                ets = [f"{s}{pre}_libre_avprimaria_pl-{p}_s0" for p in sorted(PLANTILLAS)]
                for d in self.K.exige(ets):
                    v = d["verdad"]
                    cal[(s, rej, v["plantilla"])] = dict(plantilla=v["plantilla"], spt=v["spt"], codigo=spt_code(v["spt"]),
                                                         teff=d["posterior"]["der_teff"]["p50"],
                                                         av=d["posterior"]["av"]["p50"], av_iny=v["av"])
        r = {"tipo": {}, "joven": {}}
        for s in SLUGS:
            for rej in ("btsettl", "drift"):
                T = t[(s, rej, "prior")]["posterior"]["der_teff"]["p50"]
                cc = [cal[(s, rej, p)] for p in sorted(PLANTILLAS)]
                tipo = tb.tipo_equivalente(cc, T, rejilla=rej)
                r["tipo"][f"{s}_{rej}"] = dict(teff=T, **tipo)
                # «one young M9 template gives nearly the same temperature»: de las plantillas de tipo anterior al
                # más tardío de la calibración, la de T_eff más cercana a la del compañero
                ultimo = max(c["codigo"] for c in cc)
                cerca = min((c for c in cc if c["codigo"] < ultimo), key=lambda c: abs(c["teff"] - T))
                r["joven"][f"{s}_{rej}"] = dict(plantilla=cerca["plantilla"], spt=cerca["spt"], teff=cerca["teff"],
                                                teff_companero=T,
                                                # la diferencia de las T_eff impresas (2214 − 2191, 2158 − 2145)
                                                dif_impresas=int(i0(cerca["teff"])) - int(i0(T)))
        difs = {s: [cal[(s, "drift", p)]["teff"] - cal[(s, "btsettl", p)]["teff"] for p in PLANTILLAS] for s in SLUGS}
        r["drift_menos_bt_rms"] = {s: float(np.sqrt(np.mean(np.square(v)))) for s, v in difs.items()}
        r["drift_12b_M65_M8"] = [cal[(O1, "drift", p)]["teff"] for p in PLANTILLAS
                                 if cal[(O1, "drift", p)]["codigo"] <= spt_code("M8")]
        r["av_menos_inyectada_max"] = max(v["av"] - v["av_iny"] for v in cal.values())
        r["av_42b"] = [min(cal[k]["av"] for k in cal if k[0] == O2), max(cal[k]["av"] for k in cal if k[0] == O2)]
        r["n"], r["tipos"] = len(PLANTILLAS), [min((c for c in cal.values()), key=lambda c: c["codigo"])["spt"],
                                               max((c for c in cal.values()), key=lambda c: c["codigo"])["spt"]]
        # la misma plantilla con el ruido y la cobertura de cada compañero, y plantillas del mismo tipo; el par del
        # mismo tipo más discrepante va aparte («(500 K for one pair of M9 templates)»)
        misma = [abs(cal[(O1, rej, p)]["teff"] - cal[(O2, rej, p)]["teff"])
                 for rej in ("btsettl", "drift") for p in PLANTILLAS]
        pares = {}
        ps = sorted(PLANTILLAS)
        for i, a in enumerate(ps):
            for b in ps[i + 1:]:
                if cal[(O1, "btsettl", a)]["spt"] == cal[(O1, "btsettl", b)]["spt"]:
                    pares[(a, b)] = [abs(cal[(s, rej, a)]["teff"] - cal[(s, rej, b)]["teff"])
                                     for s in SLUGS for rej in ("btsettl", "drift")]
        peor = max(pares, key=lambda k: max(pares[k]))
        resto = misma + [x for k, v in pares.items() if k != peor for x in v]
        r["dispersion_plantillas"] = dict(max_resto=max(resto), par_discrepante=list(peor),
                                          spt_par=cal[(O1, "btsettl", peor[0])]["spt"],
                                          par_discrepante_dif=[min(pares[peor]), max(pares[peor])])
        return r

    # ------------------------------------------------------------------ ejes de las rejillas con los que corrieron
    def _ejes(self):
        """Los ejes (T_eff, log g) de cada rejilla, de los `ejes` que guarda cada cadena de las Tablas B.6–B.7 (las
        de Diamondback juntas); todas las de una rejilla tienen que coincidir. Y los nodos rellenados."""
        ejes, rellenos = {}, {}
        ets = [etiqueta(s, rj, a, r250=r) for s in SLUGS for rj, r in self.b7[s] for a in AVS]
        for d in self.K.exige(ets):
            g = d.get("rejilla", "btsettl").split("1600_")[0]
            e = d.get("ejes")
            if not e:
                raise Faltan([f"{d['etiqueta']} (sin `ejes`)"])
            par = tuple(tuple(float(v) for v in e[k]) for k in ("teff", "logg"))
            if ejes.setdefault(g, par) != par:
                raise ValueError(f"{g}: cadenas con ejes distintos ({ejes[g]} / {par})")
            rellenos.setdefault(g, set()).update(tuple(x) for x in d.get("nodos_interpolados") or [])
        rango_de = lambda x0, paso, n: [x0, x0 + paso * (n - 1)]  # noqa: E731
        return {g: dict(teff=rango_de(*t), logg=rango_de(*lg), rellenados=sorted(rellenos[g]))
                for g, (t, lg) in ejes.items()}

    # ------------------------------------------------------------------ puntos vivos de cada grupo de cadenas
    def _nlive(self):
        t = self.b6()
        r = dict(b6_bt_fija=sorted({t[(s, "btsettl", a)]["nlive"] for s in SLUGS for a in AVS[:2]}),
                 b6_resto=sorted({d["nlive"] for (s, g, a), d in t.items() if not (g == "btsettl" and a != "prior")}))
        return r


def orden_rejilla(rejilla):
    return list(tb.REJILLAS_TABLA).index(rejilla)


def cruce_lineal(pts):
    """La A_V a la que Δ cruza el cero, interpolada en el primer tramo que lo encierra (de menos a más); None si
    ningún tramo lo encierra."""
    for (a0, d0), (a1, d1_) in zip(pts, pts[1:]):
        if d0 <= 0 <= d1_ and d1_ != d0:
            return a0 + (a1 - a0) * (0 - d0) / (d1_ - d0)
    return None


# ================================================================== las cifras del texto
@dataclass
class Cifra:
    r"""Una frase del paper con cifras de las cadenas. `frase`: el texto exacto (con los blancos colapsados) con
    `«campo»` donde va cada cifra; `campos`: campo -> f(c) -> texto; `guarda`: f(c) -> bool, que la frase siga
    diciendo lo que dice (un signo, un orden, un solape); `discrepancia`: si la cifra impresa no sale con ninguna
    definición, por qué (se informa y no hace fallar)."""
    id: str
    donde: str
    frase: str
    campos: dict[str, Callable] = field(default_factory=dict)
    que: str = ""
    guarda: Callable | None = None
    discrepancia: str = ""


def parte(c, nombre):
    v = c.get(nombre)
    if v is None or (isinstance(v, dict) and "faltan" in v):
        raise Faltan(v.get("faltan") if isinstance(v, dict) else [nombre])
    return v


B6 = lambda c: parte(c, "b6")  # noqa: E731
SEM = lambda c: parte(c, "semillas")  # noqa: E731
CR = lambda c: parte(c, "cruce_y_avlibre")  # noqa: E731
RJ = lambda c: parte(c, "rejillas")  # noqa: E731
DT = lambda c: parte(c, "datos")  # noqa: E731
AC = lambda c: parte(c, "ames_cond_datos")  # noqa: E731
EV = lambda c: parte(c, "evo")  # noqa: E731
INY = lambda c: parte(c, "inyecciones")  # noqa: E731
PL = lambda c: parte(c, "plantillas")  # noqa: E731
NL = lambda c: parte(c, "nlive")  # noqa: E731
EJ = lambda c: parte(c, "ejes")  # noqa: E731


def eje(c, g, p, primero=True):
    """Los extremos de un eje de la rejilla `g` (`p` = 'teff' a enteros, 'logg' a un decimal)."""
    lo, hi = EJ(c)[g][p]
    f = i0 if p == "teff" else d1
    return (rf"\(T_{{\rm eff}} = {f(lo)}\)--\({f(hi)}\)~K" if p == "teff" else rf"\(\log g = {f(lo)}\)--\({f(hi)}\)") \
        if primero else (rf"\({f(lo)}\)--\({f(hi)}\)~K" if p == "teff" else rf"\({f(lo)}\)--\({f(hi)}\)")


def unico(xs):
    xs = sorted(set(xs))
    if len(xs) != 1:
        raise ValueError(f"más de un valor: {xs}")
    return xs[0]


def cruce(c):
    a, b = CR(c)[f"{O1}_btsettl"]["cruce"], CR(c)[f"{O1}_drift"]["cruce"]
    if a is None or b is None:
        raise ValueError(f"Δ de ROXs 12 b no cruza el cero entre 1.8 y 3.0 mag (BT-Settl {a}, DRIFT-PHOENIX {b})")
    return rango(a, b, d1)


def cruce_ok(c):
    a, b = CR(c)[f"{O1}_btsettl"]["cruce"], CR(c)[f"{O1}_drift"]["cruce"]
    return a is not None and b is not None


av_alto = lambda c: d1(max(B6(c)["12b_av_prior"]))  # noqa: E731
n_rej = lambda c: PALABRAS[RJ(c)["n_rejillas"][O1]]  # noqa: E731  («the six grids»: las familias de rejillas de O1)
av12 = lambda c: rango(min(SEM(c)["av_prior_12b_todas"]), max(SEM(c)["av_prior_12b_todas"]), d1)  # noqa: E731
av42 = lambda c: rango(min(B6(c)["42b_av_prior"]), max(B6(c)["42b_av_prior"]), d1)  # noqa: E731
T12_18 = lambda c: rango(min(B6(c)["12b_T_18"]), max(B6(c)["12b_T_18"]), k50)  # noqa: E731
T12_30 = lambda c: rango(min(B6(c)["12b_T_30_prior"]), max(B6(c)["12b_T_30_prior"]), k10)  # noqa: E731
T42 = lambda c: rango(*B6(c)["42b_T_18_30_sin_prior"], k10)  # noqa: E731
DB = lambda c: rango(*RJ(c)["diamondback_teff_prior"], k10)  # noqa: E731
tipo = lambda c, s, g: PL(c)["tipo"][f"{s}_{g}"]  # noqa: E731
evo = lambda c, s, k="ATMO2020": EV(c)[f"{s}_evo_solo_{k}_avprimaria_s0"]  # noqa: E731


def tipo42(c):
    """«L1 or later» si las dos rejillas dan lo mismo (la frase dice «both grids»)."""
    a, b = (tipo_prosa(tipo(c, O2, g)) for g in ("btsettl", "drift"))
    if a != b:
        raise ValueError(f"las dos rejillas dan tipos distintos para ROXs 42B b: {a} / {b}")
    return a


def lha_lbol(c, av):
    """log(L_Hα/L_bol) de ROXs 12 b a la A_V dada: el de la A_V adoptada, más 0.4·A_Hα/A_V·(A_V − A_V0)."""
    e = c["entrada_margen"][O1]
    return B6(c)["12b_log_lha_lbol_av0"] + 0.4 * e["ahav"] * (av - e["av0"])


def dmacc(c, av):
    """Δ log Ṁ con la A_V (Eq. av_sensitivity): 0.4·a·A_Hα/A_V·(A_V − A_V0)."""
    e = c["entrada_margen"][O1]
    return 0.4 * e["a"] * e["ahav"] * (av - e["av0"])


def lim_pub(c):
    v = B6(c)["42b_limite_Lbol_publicada"]
    return rango(-max(v), -min(v), d1)


def joven(c):
    """La plantilla de tipo anterior al más tardío más cercana en T_eff, si es la misma con las dos rejillas."""
    j = [PL(c)["joven"][f"{O2}_{g}"] for g in ("btsettl", "drift")]
    if j[0]["plantilla"] != j[1]["plantilla"]:
        raise ValueError(f"no es la misma plantilla con las dos rejillas: {j[0]['plantilla']} / {j[1]['plantilla']}")
    return j


def sigma_evo(c, g):
    """(T_eff a 1.8 − T_evo) / √(150² + σ_evo² + σ_T²), σ = semiancho 16–84 % de cada una (12 b)."""
    e = evo(c, O1)
    p = B6(c)["teff_post"][f"{O1}_{g}_1.8"]
    return (p["p50"] - e["teff_p50"]) / math.sqrt(SISTEMATICO_K ** 2 + e["teff_semiancho"] ** 2 + ((p["p84"] - p["p16"]) / 2) ** 2)


def dispersion(c):
    r"""«span \(130\)--\(240\)~K (\(420\) and \(740\)~K for ROXs~42B~b with the prior on \(A_V\) and at \(A_V =
    3.0\)~mag, where AMES-Cond gives \(2890\)~K)»: máx. − mín. de T_eff de las rejillas por compañero y A_V. Las que
    pasan de dos veces el sistemático (300 K) van aparte, redondeadas a 10 K; el resto, como rango hacia fuera. De
    la más ancha se dice qué rejilla la fija (la que, quitada, más la acorta) y su T_eff."""
    disp = RJ(c)["dispersion"]
    normales = [v["dispersion"] for v in disp.values() if v["dispersion"] <= 2 * SISTEMATICO_K]
    aparte = sorted(((k, v) for k, v in disp.items() if v["dispersion"] > 2 * SISTEMATICO_K), key=lambda x: x[1]["dispersion"])
    txt = rango(min(normales), max(normales), abajo(10), arriba(10)) + "~K"
    if not aparte:
        return txt
    slugs = {k.rsplit("_", 1)[0] for k, _ in aparte}
    if len(slugs) != 1 or len(aparte) != 2:
        raise ValueError(f"dispersiones aparte: {[(k, round(v['dispersion'])) for k, v in aparte]}: la frase cambia")
    s = slugs.pop()
    etq = {"prior": r"with the prior on \(A_V\)", "1.8": r"at \(A_V = 1.8\)~mag", "3": r"at \(A_V = 3.0\)~mag"}
    (k1, v1), (k2, v2) = aparte
    return (txt + rf" (\({k10(v1['dispersion'])}\) and \({k10(v2['dispersion'])}\)~K for {NOMBRES[s]} "
            rf"{etq[k1.rsplit('_', 1)[1]]} and {etq[k2.rsplit('_', 1)[1]]}, where {tb.REJILLAS_TABLA[v2['rejilla_extrema']]} "
            rf"gives \({k10(v2['teff_extrema'])}\)~K)")


def dlnz_frase(c):
    """«higher at A_V = 3.0 than at 1.8 mag for ROXs~12~b and lower for ROXs~42B~b, with both grids»."""
    m = B6(c)["dlnz_30_mayor_que_18"]
    pal = {}
    for s in SLUGS:
        v = {m[f"{s}_{g}"] for g in ("btsettl", "drift")}
        if len(v) != 1:
            raise ValueError(f"{s}: las dos rejillas no coinciden: la frase cambia")
        pal[s] = "higher" if v.pop() else "lower"
    if pal[O1] == pal[O2]:
        raise ValueError("los dos compañeros dan lo mismo: la frase cambia")
    return pal[O1], pal[O2]


def avlibre_frase(c):
    """«In the two fits of ROXs~12~b, the evidence is higher than in all the fits of this table»."""
    sube = sorted(k for k, v in CR(c)["avlibre"].items() if v["dlnz_vs_mejor_B6"] > 0)
    if sube != [f"{O1}_btsettl", f"{O1}_drift"]:
        raise ValueError(f"con A_V libre superan a la tabla: {sube}: la frase cambia")
    return PALABRAS[len(sube)], NOMBRES[O1]


def otras_min_donde(c):
    rj, r, av = RJ(c)["delta_42b_sin_bt_drift_min_en"]
    return f"{tb.REJILLAS_TABLA[rj]} at " + (r"\(A_V = " + f"{float(av):.1f}" + r"\)~mag" if av != "prior"
                                             else r"with the prior on \(A_V\)")


def fila_T(c, av):
    return " & ".join(rf"\({i0(B6(c)[f'B5_{s}_{g}']['teff'][av])}\)" for s in SLUGS for g in ("btsettl", "drift"))


def fila_D(c, av):
    v = [B6(c)[f"B5_{s}_{g}"]["delta"][av] for s in SLUGS for g in ("btsettl", "drift")]
    return " & ".join([rf"\({s2(x)}\)" for x in v[:2]] + [rf"\(<{s2(x)}\)" for x in v[2:]])


AC_LOGG_BORDE = tb.REJILLAS["ames_cond"]["logg"][0] + tb.REJILLAS["ames_cond"]["logg"][1] / 2


def _cifras() -> list[Cifra]:  # noqa: C901 (una lista larga, no lógica)
    L: list[Cifra] = []
    A = lambda *a, **k: L.append(Cifra(*a, **k))  # noqa: E731

    # ---------------------------------------------------------------- resumen y Sect. 2
    A("resumen_cruce", "Resumen",
      r"it would trace accretion if the extinction of the companion exceeds «cruce»~mag, and its accretion rate",
      dict(cruce=cruce), "A_V a la que Δ de ROXs 12 b cruza el cero: BT-Settl por 1.8/2.5/3.0, DRIFT-PHOENIX por "
      "1.8/3.0, interpolación lineal (1 decimal)", guarda=cruce_ok)
    A("s2_R", "Sect. 2",
      r"The fits in absolute flux of Appendix~\ref{app:nir} give «R»~\(R_{\mathrm{Jup}}\) for ROXs~12~b, depending "
      r"on the extinction and the grid.",
      dict(R=lambda c: rango(*B6(c)["12b_R_rango"], d1)), "R (p50) de las seis cadenas de ROXs 12 b de la Tabla B.6")

    # ---------------------------------------------------------------- Sect. 4.1 (ROXs 12 b)
    A("s41_T", "Sect. 4.1",
      r"It gives \(T_{\rm eff} = «T18»~K at the adopted \(A_V\), and «T30»~K for \(A_V = 3.0\)--\(«av»\)~mag, with a "
      r"systematic error of about \(150\)~K (Table~\ref{tab:teff}).",
      dict(T18=lambda c: T12_18(c)[2:], T30=T12_30, av=av_alto),
      "T_eff (p50) a A_V 1.8 de las dos rejillas, a 50 K; a 3.0 y con prior, a 10 K; A_V con prior (s1, la mayor)")
    A("s41_tipo", "Sect. 4.1",
      r"the companion corresponds to «bt» with BT-Settl and to «dr» with DRIFT-PHOENIX (Table~\ref{tab:teff}).",
      dict(bt=lambda c: tipo_prosa(tipo(c, O1, "btsettl")), dr=lambda c: tipo_prosa(tipo(c, O1, "drift"))),
      "tipo equivalente (`teff_bayes.tipo_equivalente`): plantillas con la misma rejilla, T_eff con prior, s1")
    A("s41_evo", "Sect. 4.1",
      r"The evolutionary models give \(«T»\pm«e»\)~K for the published luminosity and age, and they place the "
      r"companion at the noise level (\(«d»\)~dex;",
      dict(T=lambda c: k10(evo(c, O1)["teff_p50"]), e=lambda c: k10(evo(c, O1)["teff_semiancho"]),
           d=lambda c: s2(evo(c, O1)["delta_p50"])),
      "cadena `evo_solo` ATMO2020: T_eff p50 y semiancho 16–84 % a 10 K; Δ con la L_bol publicada y la A_V adoptada",
      guarda=lambda c: abs(evo(c, O1)["delta_p50"]) < 0.05)
    A("s41_d", "Sect. 4.1",
      r"The companion is at the noise level (BT-Settl) or \(«d18»\)~dex below it (DRIFT-PHOENIX) at the adopted "
      r"\(A_V = 1.8\)~mag, and «d30»~dex above it for \(A_V = 3.0\)--\(«av»\)~mag (Table~\ref{tab:teff}).",
      dict(d18=lambda c: d1(-B6(c)["delta"][f"{O1}_drift_1.8"]),
           d30=lambda c: rango(min(B6(c)["12b_delta_30_prior"]), max(B6(c)["12b_delta_30_prior"]), d1, d2), av=av_alto),
      "Δ (p50) de la Tabla B.5; a 3.0 y con prior, el mínimo con 1 decimal y el máximo con 2 (como el texto)",
      guarda=lambda c: (abs(B6(c)["delta"][f"{O1}_btsettl_1.8"]) < 0.05 and B6(c)["delta"][f"{O1}_drift_1.8"] < 0
                        and min(B6(c)["12b_delta_30_prior"]) > 0))
    A("s41_dif", "Sect. 4.1",
      r"At fixed \(A_V\), the two grids of model atmospheres give the same difference within \(«x»\)~dex.",
      dict(x=lambda c: d1(max(B6(c)["12b_dif_rejillas_av_fija"]))), "máx. |Δ_BT − Δ_DRIFT| de 12 b con A_V fija")
    A("s41_rej", "Sect. 4.1",
      r"With the «n» grids that we fit to this companion, the difference is «d18»~dex at \(A_V = 1.8\)~mag and "
      r"«d30»~dex at \(3.0\)~mag (Table~\ref{tab:teff_grids}).",
      dict(n=n_rej, d18=lambda c: hasta(*RJ(c)["delta_12b_18"], s2), d30=lambda c: hasta(*RJ(c)["delta_12b_30"], s2)),
      "Δ de 12 b en la Tabla B.7 (todas las rejillas y resoluciones)",
      guarda=lambda c: RJ(c)["delta_12b_30"][0] > 0)
    A("s41_solos", "Sect. 4.1",
      r"spectra alone, the difference is «d18»~dex at \(A_V = 1.8\)~mag and «d30»~dex at \(3.0\)~mag "
      r"(Appendix~\ref{app:nir}).",
      dict(d18=lambda c: hasta(*DT(c)["delta_12b_solos_18"], s2), d30=lambda c: hasta(*DT(c)["delta_12b_solos_30"], s2)),
      "Δ de 12 b con solo MUSE o solo IR, las dos rejillas")
    A("s41_libre", "Sect. 4.1",
      r"with a free \(A_V\), it gives \(«T»\)~K, with the spurious extinction of Appendix~\ref{app:nir} "
      r"(\(A_V = «av»\)~mag). This is the edge of the DRIFT-PHOENIX grid, but not of the BT-Settl grid, which "
      r"reaches \(«techo»\)~K; with BT-Settl, \(97.5\%\) of the samples are below \(«p975»\)~K.",
      dict(T=lambda c: str(r_mitad(CR(c)["avlibre"][f"{O1}_btsettl"]["teff"], 100)),
           av=lambda c: d1(CR(c)["avlibre"][f"{O1}_btsettl"]["av"]),
           techo=lambda c: i0(tb.REJILLAS["btsettl"]["teff"][0] + tb.REJILLAS["btsettl"]["teff"][1] * (tb.REJILLAS["btsettl"]["teff"][2] - 1)),
           p975=lambda c: k10(CR(c)["avlibre"][f"{O1}_btsettl"]["teff_p97_5"])),
      "cadena de BT-Settl con A_V libre (T_eff a 100 K, su 97.5 % a 10 K); el techo de la rejilla",
      guarda=lambda c: ("teff" in CR(c)["avlibre"][f"{O1}_drift"]["borde"]
                        and "teff" not in CR(c)["avlibre"][f"{O1}_btsettl"]["borde"]))
    A("s41_cruce", "Sect. 4.1",
      r"the companion is above it if its extinction exceeds «cruce»~mag. With the prior on \(A_V\), the fits of its "
      r"spectra give «avs»~mag (Appendix~\ref{app:nir}). At \(A_V = 3.0\)~mag, the companion is «d30»~dex above the "
      r"noise level with the «n» grids. At \(A_V = 3.0\)--\(«av»\)~mag, \(\log(\lha/L_{\mathrm{bol}})\) is \(«l30»\) "
      r"to \(«lav»\), at the upper end",
      dict(n=n_rej, cruce=cruce, avs=av12, d30=lambda c: rango(*RJ(c)["delta_12b_30"], d2), av=av_alto,
           l30=lambda c: d1(lha_lbol(c, 3.0)), lav=lambda c: d1(lha_lbol(c, round(max(B6(c)["12b_av_prior"]), 1)))),
      "cruce de signo; A_V con prior de 12 b, tres semillas y dos rejillas; Δ a 3.0 en la Tabla B.7; "
      "log(L_Hα/L_bol): la L_Hα que da la L_acc de G3 con la relación de la config, reescalada a la A_V",
      guarda=lambda c: cruce_ok(c) and RJ(c)["delta_12b_30"][0] > 0)
    A("s41_macc", "Sect. 4.1",
      r"At \(A_V = 3.0\)--\(«av»\)~mag, both rates are «dm»~dex higher (Eq.~\ref{eq:av_sensitivity}).",
      dict(av=av_alto, dm=lambda c: rango(dmacc(c, 3.0), dmacc(c, round(max(B6(c)["12b_av_prior"]), 1)), d2)),
      "Δ log Ṁ = 0.4·a·A_Hα/A_V·(A_V − 1.8) en 3.0 y en la A_V con prior")

    # ---------------------------------------------------------------- Sect. 4.2 (ROXs 42B b)
    A("s42_lim", "Sect. 4.2",
      r"At the temperature of the companion, «T»~K (see below), this limit is «pub»~dex below the chromospheric "
      r"noise level. When the temperature, the radius and the extinction come from the same fit, as for ROXs~12~b, "
      r"the limit is between \(«dmin»\)~dex below and \(«dmax»\)~dex above the noise level for \(A_V \le 3\)~mag "
      r"with both grids of model atmospheres (Table~\ref{tab:teff}). With the other grids, it is between "
      r"\(«omin»\)~dex below («odonde») and \(«omax»\)~dex above it (Table~\ref{tab:teff_grids}). With the MUSE "
      r"spectrum or the near-infrared spectra alone, it is «solos»~dex from the noise level (Appendix~\ref{app:nir}).",
      dict(T=T42, pub=lim_pub, dmin=lambda c: d2(-B6(c)["42b_delta_todas"][0]), dmax=lambda c: d2(B6(c)["42b_delta_todas"][1]),
           omin=lambda c: d2(-RJ(c)["delta_42b_sin_bt_drift"][0]), odonde=otras_min_donde,
           omax=lambda c: d2(RJ(c)["delta_42b_sin_bt_drift"][1]),
           solos=lambda c: hasta(*DT(c)["delta_42b_todo_conjunto"], s2)),
      "T_eff con A_V fija de las dos rejillas (10 K); límite con la L_bol publicada en sus extremos; Δ de las seis "
      "cadenas de la Tabla B.6; Δ de las otras rejillas de la Tabla B.7 y dónde está su mínimo; solo MUSE / solo IR",
      guarda=lambda c: (max(B6(c)["42b_limite_Lbol_publicada"]) < 0 and B6(c)["42b_delta_todas"][0] < 0 < B6(c)["42b_delta_todas"][1]
                        and max(B6(c)["42b_av_prior"]) <= 3.0 and RJ(c)["delta_42b_sin_bt_drift"][0] < 0 < RJ(c)["delta_42b_sin_bt_drift"][1]))
    A("s42_T", "Sect. 4.2",
      r"gives \(T_{\rm eff} = «T»~K for \(A_V = 1.8\)--\(3.0\)~mag, with both grids of models",
      dict(T=lambda c: T42(c)[2:]), "T_eff (p50) de las cuatro cadenas con A_V fija de la Tabla B.6, a 10 K")
    A("s42_tipo", "Sect. 4.2",
      r"the companion corresponds to «tipo», consistent with the types above. The L templates are older "
      r"(\(10\)--\(200\)~Myr) than the companion, and one young «spt» template gives nearly the same temperature, so "
      r"the type of the companion can be earlier than L1.",
      dict(tipo=tipo42, spt=lambda c: joven(c)[0]["spt"]),
      "tipo equivalente con las dos rejillas; la plantilla de tipo anterior más cercana en T_eff (las edades de las "
      "plantillas L no son de las cadenas)",
      guarda=lambda c: all(j["teff"] > j["teff_companero"] for j in joven(c)))
    A("s42_db", "Sect. 4.2",
      r"Sonora Diamondback, which has the highest evidence for this companion, gives «db»~K with the prior on \(A_V\) "
      r"(Table~\ref{tab:teff_grids}). The temperature of BT-Settl and DRIFT-PHOENIX agrees with \(«T»\pm«e»\)~K from "
      r"evolutionary models at the published luminosity and age, and with the \(2270^{+170}_{-140}\)~K of the "
      r"\(K\)-band retrieval of \citet{ref:xuan2024}. That of Sonora Diamondback is «menos»~K lower.",
      dict(db=DB, T=lambda c: k10(evo(c, O2)["teff_p50"]), e=lambda c: k10(evo(c, O2)["teff_semiancho"]),
           menos=lambda c: rango(evo(c, O2)["teff_p50"] - RJ(c)["diamondback_teff_prior"][1],
                                 evo(c, O2)["teff_p50"] - RJ(c)["diamondback_teff_prior"][0], k10)),
      "Diamondback (6 f_sed) con prior, a 10 K; `evo_solo` ATMO2020 de 42B b; la diferencia evo − Diamondback a 10 K",
      guarda=lambda c: (RJ(c)["db_mayor_evidencia_42b_prior_r500"].startswith("diamondback")
                        and all(abs(T - evo(c, O2)["teff_p50"]) <= math.hypot(SISTEMATICO_K, evo(c, O2)["teff_semiancho"])
                                for T in B6(c)["42b_T_todas"])))

    # ---------------------------------------------------------------- Sect. 5.3
    A("s53_av", "Sect. 5.3",
      r"it moves to \(A_V = «av12»~mag for ROXs~12~b, while for templates of known type, fitted in the same way, it "
      r"does not exceed the injected value by more than \(«pl»\)~mag. For ROXs~42B~b, it gives «av42»~mag, within "
      r"the range that the templates give.",
      dict(av12=lambda c: av12(c)[2:], pl=lambda c: cota_mag(PL(c)["av_menos_inyectada_max"]), av42=av42),
      "A_V con prior de 12 b (tres semillas, dos rejillas) y de 42B b (s1); máx. A_V − A_V inyectada de las plantillas",
      guarda=lambda c: PL(c)["av_42b"][0] <= min(B6(c)["42b_av_prior"]) and max(B6(c)["42b_av_prior"]) <= PL(c)["av_42b"][1])

    # ---------------------------------------------------------------- conclusiones
    A("con_3", "Conclusiones 3",
      r"At the adopted extinction, it is within \(«x»\)~dex of the chromospheric noise level of young stars "
      r"\citep{ref:manara2013} with «n» grids of model atmospheres, and within \(«y»\)~dex with the MUSE spectrum or "
      r"the near-infrared spectra alone.",
      dict(n=n_rej, x=lambda c: d1(RJ(c)["delta_12b_18_absmax"]), y=lambda c: d1(max(abs(v) for v in DT(c)["delta_12b_solos_18"]))),
      "máx. |Δ| de 12 b a 1.8 en la Tabla B.7 y con solo MUSE o solo IR")
    A("con_4", "Conclusiones 4",
      r"the noise level of young stars if its extinction exceeds «cruce»~mag (BT-Settl and DRIFT-PHOENIX), by "
      r"«d30»~dex at \(A_V = 3.0\)~mag with the «n» grids, and the fits of its spectra prefer such an extinction.",
      dict(n=n_rej, cruce=cruce, d30=lambda c: rango(*RJ(c)["delta_12b_30"], d2)),
      "cruce de signo; Δ de 12 b a 3.0 en la Tabla B.7",
      guarda=lambda c: cruce_ok(c) and min(SEM(c)["av_prior_12b_todas"]) > 3.0)
    A("con_5", "Conclusiones 5",
      r"the companion has \(T_{\rm eff} = «T»~K with BT-Settl and DRIFT-PHOENIX («tipo»), and «db»~K with Sonora "
      r"Diamondback. At this temperature the limit is close to the chromospheric level of young stars: «pub»~dex "
      r"below it with the published luminosity, and up to \(«arriba»\)~dex above it with the luminosities of the fits "
      r"to the spectra.",
      dict(T=lambda c: T42(c)[2:], tipo=tipo42, db=DB, pub=lim_pub,
           arriba=lambda c: d1(max(RJ(c)["delta_42b_todas"][1], B6(c)["42b_delta_todas"][1], DT(c)["delta_42b_todo_conjunto"][1]))),
      "como la Sect. 4.2; «up to»: máx. Δ de 42B b en las Tablas B.6 y B.7 y con solo MUSE o solo IR",
      guarda=lambda c: max(B6(c)["42b_limite_Lbol_publicada"]) < 0 < RJ(c)["delta_42b_todas"][1])

    # ---------------------------------------------------------------- apéndice B: método y pruebas
    A("apB_ejes", "Ap. B",
      r"The first is BT-Settl, as above, but with «bt». The second is DRIFT-PHOENIX \citep{ref:helling2008, "
      r"ref:witte2009, ref:witte2011}, which models the formation of dust clouds kinetically, with «drt» and «drg» at "
      r"solar metallicity. This grid has no model at \(«nt»\)~K and \(\log g = «ng»\),",
      dict(bt=lambda c: eje(c, "btsettl", "logg"), drt=lambda c: eje(c, "drift", "teff"), drg=lambda c: eje(c, "drift", "logg"),
           nt=lambda c: i0(unico(EJ(c)["drift"]["rellenados"])[0]), ng=lambda c: d1(unico(EJ(c)["drift"]["rellenados"])[1])),
      "los `ejes` y los `nodos_interpolados` que guarda cada cadena")
    A("apB_ejes_otras", "Ap. B",
      r"AMES-Dusty covers «adt» and «adg», AMES-Cond «act» and «acg», BT-Dusty «bdt» and «bdg», Diamondback «dbt» and "
      r"«dbg», and SPHINX «spt» and «spg».",
      {k: (lambda g, p, prim: lambda c: eje(c, g, p, prim))(g, p, k.startswith("ad"))
       for k, g, p in (("adt", "ames_dusty", "teff"), ("adg", "ames_dusty", "logg"), ("act", "ames_cond", "teff"),
                       ("acg", "ames_cond", "logg"), ("bdt", "bt_dusty", "teff"), ("bdg", "bt_dusty", "logg"),
                       ("dbt", "diamondback", "teff"), ("dbg", "diamondback", "logg"), ("spt", "sphinx", "teff"),
                       ("spg", "sphinx", "logg"))},
      "los `ejes` de las cadenas de la Tabla B.7 (Diamondback: las de 1600 K)")
    A("apB_nlive", "Ap. B",
      r"For the fits of Table~\ref{tab:teff}, we use \(«n1»\) live points with BT-Settl and \(A_V\) fixed, and "
      r"\(«n2»\) in the other cases.",
      dict(n1=lambda c: str(unico(NL(c)["b6_bt_fija"])), n2=lambda c: str(unico(NL(c)["b6_resto"]))),
      "`nlive` de las 12 cadenas de la Tabla B.6")
    A("apB_iny", "Ap. B",
      r"a model of the grid, injected with the signal-to-noise ratio of each companion, is recovered within \(«x»\)~K.",
      dict(x=lambda c: str(math.ceil(INY(c)["max_abs"]))), "máx. |T_eff − T_inyectada| de las 4 inyecciones, hacia arriba")
    A("apB_plantillas", "Ap. B",
      r"we fit in the same way \(«n»\) young templates of types «t0»--«t1» (Table~\ref{tab:libraries})",
      dict(n=lambda c: str(PL(c)["n"]), t0=lambda c: PL(c)["tipos"][0], t1=lambda c: PL(c)["tipos"][1]),
      "las plantillas ajustadas como datos")
    A("apB_sistematico", "Ap. B",
      r"differ by up to \(«x»\)~K (\(«y»\)~K for one pair of «spt» templates), so we take \(\pm150\)~K as the "
      r"systematic error of the method. The fitted extinction of the templates does not exceed the injected value by "
      r"more than \(«av»\)~mag. For the same templates, the temperatures of the two grids differ by «rms»~K (root mean "
      r"square), and with the noise of ROXs~12~b, DRIFT-PHOENIX gives «dr»~K for all the templates of types M6.5--M8, "
      r"close to the edge of its grid.",
      dict(x=lambda c: arriba(10)(PL(c)["dispersion_plantillas"]["max_resto"]),
           y=lambda c: arriba(10)(PL(c)["dispersion_plantillas"]["par_discrepante_dif"][1]),
           spt=lambda c: PL(c)["dispersion_plantillas"]["spt_par"],
           av=lambda c: cota_mag(PL(c)["av_menos_inyectada_max"]),
           rms=lambda c: rango(min(PL(c)["drift_menos_bt_rms"].values()), max(PL(c)["drift_menos_bt_rms"].values()), k10),
           dr=lambda c: rango(min(PL(c)["drift_12b_M65_M8"]), max(PL(c)["drift_12b_M65_M8"]), k10)),
      "misma plantilla con el ruido de cada compañero y plantillas del mismo tipo (las dos rejillas), hacia arriba a "
      "10 K; el par del mismo tipo más discrepante, aparte; DRIFT − BT (rms por compañero); DRIFT de M6.5–M8 con el "
      "ruido de 12 b",
      guarda=lambda c: PL(c)["dispersion_plantillas"]["par_discrepante_dif"][0] > PL(c)["dispersion_plantillas"]["max_resto"])
    A("apB_corr", "Ap. B",
      r"increase with the extinction for both companions and both grids (correlation coefficients of «r»); for "
      r"ROXs~12~b, the radius also increases and the \(K\)-band excess decreases (\(|r| = «r2»).",
      dict(r=lambda c: rango(*B6(c)["corr_T_delta_rango"], d1),
           r2=lambda c: rango(min(abs(x) for x in B6(c)["corr_12b_R_rk"]), max(abs(x) for x in B6(c)["corr_12b_R_rk"]), d1)[2:]),
      "corr(T_eff, A_V) y corr(Δ, A_V) en las 4 cadenas con prior; |corr(R, A_V)| y |corr(r_K, A_V)| de 12 b",
      guarda=lambda c: (min(B6(c)["corr_T_delta_rango"]) > 0 and all(
          B6(c)["correlaciones"][f"{O1}_{g}"]["R_av"] > 0 > B6(c)["correlaciones"][f"{O1}_{g}"]["rk_av"] for g in ("btsettl", "drift"))))
    A("apB_solapes", "Ap. B",
      r"The \(68\%\) intervals of the two grids «s68». The \(95\%\) intervals meet at about \(«m95»\)~K in temperature "
      r"for ROXs~12~b, and they overlap in «s95» for ROXs~42B~b (\(\log g = «gbt» with BT-Settl and «gdr» with "
      r"DRIFT-PHOENIX).",
      dict(s68=lambda c: frase_solape68({(s, p): B6(c)["solapes_prior_s1"][f"{s}_{p}"]["solapan68"]
                                          for s in SLUGS for p in ("teff", "logg")}),
           m95=lambda c: k10((B6(c)["solapes_prior_s1"][f"{O1}_teff"]["bt95"][1]
                              + B6(c)["solapes_prior_s1"][f"{O1}_teff"]["drift95"][0]) / 2),
           s95=lambda c: {(True, True): "gravity and in temperature", (True, False): "gravity"}[
               (B6(c)["solapes_prior_s1"][f"{O2}_logg"]["solapan95"], B6(c)["solapes_prior_s1"][f"{O2}_teff"]["solapan95"])],
           gbt=lambda c: rango(*B6(c)["solapes_prior_s1"][f"{O2}_logg"]["bt95"], d2)[2:],
           gdr=lambda c: rango(*B6(c)["solapes_prior_s1"][f"{O2}_logg"]["drift95"], d2)),
      "intervalos del 68 y 95 % de T_eff y log g con prior (s1); «meet at about»: punto medio entre el 97.5 % de "
      "BT-Settl y el 2.5 % de DRIFT-PHOENIX de 12 b, a 10 K",
      guarda=lambda c: (abs(B6(c)["solapes_prior_s1"][f"{O1}_teff"]["bt95"][1] - B6(c)["solapes_prior_s1"][f"{O1}_teff"]["drift95"][0]) < 20
                        and not B6(c)["solapes_prior_s1"][f"{O1}_logg"]["solapan95"]))
    A("apB_42b", "Ap. B",
      r"For ROXs~42B~b, both grids give \(T_{\rm eff} = «T»~K for \(A_V = 1.8\)--\(3.0\)~mag and the equivalent type "
      r"«tipo». The L templates are older (\(10\)--\(200\)~Myr) than the companion, and one young «spt» template gives "
      r"nearly the same temperature («dif»~K higher), so the companion can be of an earlier type than L1. With the "
      r"prior on \(A_V\), the fit gives «av»~mag. The evolutionary models of \citet{ref:phillips2020}, with the "
      r"published luminosity and age, give \(«eT»\pm«ee»\)~K.",
      dict(T=lambda c: T42(c)[2:], tipo=tipo42, spt=lambda c: joven(c)[0]["spt"],
           dif=lambda c: rango(min(j["dif_impresas"] for j in joven(c)), max(j["dif_impresas"] for j in joven(c)), str),
           av=av42, eT=lambda c: k10(evo(c, O2)["teff_p50"]), ee=lambda c: k10(evo(c, O2)["teff_semiancho"])),
      "como la Sect. 4.2; «13–23 K higher»: la T_eff impresa de la plantilla menos la del compañero (2214 − 2191, "
      "2158 − 2145), con las dos rejillas",
      guarda=lambda c: all(j["teff"] > j["teff_companero"] for j in joven(c)))
    A("apB_12b", "Ap. B",
      r"it is «T18»~K at the adopted \(A_V\), and «T30»~K for \(A_V = 3.0\)--\(«av»\)~mag. With the prior on \(A_V\), "
      r"the fit moves to «avs»~mag with both grids, because the spectrum is redder than that of any template at "
      r"\(A_V = 1.8\)~mag. The equivalent type is «bt» with BT-Settl and «dr» with DRIFT-PHOENIX. The evolutionary "
      r"models give \(«eT»\pm«ee»\)~K, \(«sbt»\sigma\) (BT-Settl) and \(«sdr»\sigma\) (DRIFT-PHOENIX) below the value "
      r"at the adopted \(A_V\) when the systematic error of the method is included.",
      dict(T18=T12_18, T30=T12_30, av=av_alto, avs=av12,
           bt=lambda c: tipo_prosa(tipo(c, O1, "btsettl")), dr=lambda c: tipo_prosa(tipo(c, O1, "drift")),
           eT=lambda c: k10(evo(c, O1)["teff_p50"]), ee=lambda c: k10(evo(c, O1)["teff_semiancho"]),
           sbt=lambda c: d1(sigma_evo(c, "btsettl")), sdr=lambda c: d1(sigma_evo(c, "drift"))),
      "como la Sect. 4.1; σ: (T_eff a 1.8 − T_evo) / √(150² + σ_evo² + σ_T²)",
      guarda=lambda c: (min(SEM(c)["av_prior_12b_todas"]) > 1.8 + PL(c)["av_menos_inyectada_max"]
                        and min(B6(c)["12b_T_18"]) > evo(c, O1)["teff_p50"]))
    A("apB_edades", "Ap. B",
      r"(\(«e12»\)~Myr for ROXs~12~b and \(«e42»\)~Myr for ROXs~42B~b). The same fit returns «p12»~Myr for «n» "
      r"young templates of \(1\)--\(10\)~Myr with the noise of ROXs~12~b, and «p42»~Myr with that of ROXs~42B~b",
      dict(e12=lambda c: i0(EV(c)[f"{O1}_evo_ATMO2020_avprimaria_s0"]["edad_myr"]),
           e42=lambda c: i0(EV(c)[f"{O2}_evo_ATMO2020_avprimaria_s0"]["edad_myr"]),
           p12=lambda c: rango(*EV(c)["plantillas"][O1], abajo(1), arriba(1)),
           n=lambda c: PALABRAS[unico(EV(c)["n_plantillas"].values())],
           p42=lambda c: rango(*EV(c)["plantillas"][O2], abajo(1), arriba(1))),
      "modo evo (ATMO2020): mediana de la edad; sobre plantillas (las mismas en número para los dos), rango hacia "
      "fuera")
    A("apB_logg", "Ap. B",
      r"for ROXs~12~b, BT-Settl gives \(\log g = «bt» and DRIFT-PHOENIX «dr», which is high",
      dict(bt=lambda c: rango(*B6(c)[f"B5_{O1}_btsettl"]["logg"], d1)[2:], dr=lambda c: rango(*B6(c)[f"B5_{O1}_drift"]["logg"], d1)),
      "log g de la Tabla B.5: mín. p16 – máx. p84 de las tres A_V")
    A("apB_mejor", "Ap. B",
      r"BT-Settl fits both companions better. Its maximum log-likelihood is higher by \(«l12»\) and \(«l42»\), and its "
      r"model error in the MUSE spectrum is «fbt», against «fdr» for DRIFT-PHOENIX.",
      dict(l12=lambda c: i0(B6(c)["logl_max_bt_menos_drift_prior"][O1]),
           l42=lambda c: i0(B6(c)["logl_max_bt_menos_drift_prior"][O2]),
           fbt=lambda c: por_ciento(*B6(c)["fb_muse"]["btsettl"]), fdr=lambda c: por_ciento(*B6(c)["fb_muse"]["drift"])),
      "ln L máx. BT − DRIFT con prior; f_b de MUSE (p50) de las seis cadenas de cada rejilla",
      guarda=lambda c: min(B6(c)["logl_max_bt_menos_drift_prior"].values()) > 0 and B6(c)["fb_muse"]["btsettl"][1] < B6(c)["fb_muse"]["drift"][0])
    A("apB_12b_18", "Ap. B",
      r"For ROXs~12~b, the two grids give the same difference within \(«dif»\)~dex at fixed \(A_V\). At \(A_V = "
      r"1.8\)~mag, the fits of ROXs~12~b need a large \(K\)-band excess (\(r_K = «rk», at the edge of its prior with "
      r"DRIFT-PHOENIX) and a radius of «R»~\(R_{\rm Jup}\), below the adopted \(2.2\)~\(R_{\rm Jup}\). With the "
      r"published luminosity instead of the luminosity of the fit, the difference at \(«T»\)~K is \(«d»\)~dex. "
      r"DRIFT-PHOENIX gives a higher temperature, which raises the noise level, but also a smaller radius, which "
      r"lowers \(L_{\rm bol}\). The difference changes sign at \(A_V = «cbt»\)~mag with BT-Settl and \(«cdr»\)~mag "
      r"with DRIFT-PHOENIX.",
      dict(dif=lambda c: d1(max(B6(c)["12b_dif_rejillas_av_fija"])),
           rk=lambda c: rango(min(B6(c)["12b_18_rk"]), max(B6(c)["12b_18_rk"]), d1)[2:],
           R=lambda c: rango(min(B6(c)["12b_18_R"]), max(B6(c)["12b_18_R"]), d1),
           T=lambda c: i0(B6(c)["12b_delta_Lbol_publicada_T18"]["teff"]),
           d=lambda c: d2(B6(c)["12b_delta_Lbol_publicada_T18"]["delta"]),
           cbt=lambda c: d1(CR(c)[f"{O1}_btsettl"]["cruce"]), cdr=lambda c: d1(CR(c)[f"{O1}_drift"]["cruce"])),
      "r_K y R (p50) a A_V 1.8; Δ con la L_bol publicada a la T_eff de BT-Settl a 1.8; cruce de signo de cada rejilla",
      guarda=lambda c: (B6(c)["12b_18_rk"][1] > 1.45 and max(B6(c)["12b_18_R"]) < 2.2 and cruce_ok(c)
                        and B6(c)["12b_T_18"][1] > B6(c)["12b_T_18"][0] and B6(c)["12b_18_R"][1] < B6(c)["12b_18_R"][0]))

    # ---------------------------------------------------------------- apéndice B: las siete rejillas
    A("apB_rejillas", "Ap. B",
      r"For ROXs~12~b, BT-Settl has the highest evidence with the three choices of \(A_V\); with the other grids, "
      r"\(\ln Z\) is lower by «l12». For ROXs~42B~b, Diamondback has a higher evidence than BT-Settl for all values "
      r"of \(f_{\rm sed}\), by «dbp» with the prior on \(A_V\) and «dbf» with \(A_V\) fixed, but its gravity is at the "
      r"lower edge of its grid, and its temperature, «db»~K with the prior on \(A_V\), is the lowest of all the grids. "
      r"At a given extinction, the temperatures of the «n7» fits of a companion span «span», much more than the "
      r"interval of each fit (\(\pm«semi»~K), and similar to the systematic error of \(\pm150\)~K that we adopt. For "
      r"ROXs~12~b, \(\Delta\) is between \(«a»\) and \(«b»\)~dex at \(A_V = 1.8\)~mag and between \(«x»\) and "
      r"\(«y»\)~dex at \(3.0\)~mag with all the grids, so the sign of the difference is still set by the extinction. "
      r"For ROXs~42B~b, the upper limit is between \(«dmin»\)~dex below and \(«dmax»\)~dex above the noise level with "
      r"all the grids; it is below it only «neg».",
      dict(n7=lambda c: PALABRAS[RJ(c)["n_rejillas_total"]], l12=lambda c: rango(-RJ(c)["otras_12b_dlnz"][1], -RJ(c)["otras_12b_dlnz"][0], i0),
           dbp=lambda c: rango(*RJ(c)["diamondback_dlnz_prior"], i0), dbf=lambda c: rango(*RJ(c)["diamondback_dlnz_fija"], i0),
           db=DB, span=dispersion, semi=lambda c: rango(*RJ(c)["semiancho_global"], i0)[2:],
           a=lambda c: s2(RJ(c)["delta_12b_18"][0]), b=lambda c: s2(RJ(c)["delta_12b_18"][1]),
           x=lambda c: s2(RJ(c)["delta_12b_30"][0]), y=lambda c: s2(RJ(c)["delta_12b_30"][1]),
           dmin=lambda c: d2(-RJ(c)["delta_42b_todas"][0]), dmax=lambda c: d2(RJ(c)["delta_42b_todas"][1]),
           neg=lambda c: lista_por_debajo([(tb.REJILLAS_TABLA[rj], av) for rj, av in RJ(c)["delta_42b_negativos"]])),
      "Tabla B.7: ΔlnZ contra BT-Settl a la misma resolución (12 b, las tres A_V; Diamondback con prior y con A_V "
      "fija); T_eff de Diamondback con prior (10 K); dispersión entre rejillas por compañero y A_V; semiancho 16–84 % "
      "de T_eff; Δ de cada compañero y las rejillas con Δ < 0",
      guarda=lambda c: (RJ(c)["bt_mejor_12b_tres_av"] and RJ(c)["otras_12b_dlnz"][1] < 0 and RJ(c)["db_mayor_que_bt_todas"]
                        and RJ(c)["diamondback_borde_logg_prior"] and RJ(c)["db_teff_mas_baja"]
                        and RJ(c)["delta_12b_30"][0] > 0 and RJ(c)["delta_42b_todas"][0] < 0 < RJ(c)["delta_42b_todas"][1]))

    # ---------------------------------------------------------------- apéndice B: solo MUSE / solo IR (Fig. B.5)
    A("apB_datos", "Ap. B",
      r"With BT-Settl, the temperature of one set of spectra differs from that of all the spectra by up to \(«max»\)~K. "
      r"For ROXs~12~b, the near-infrared spectra give «i12»~K more and the MUSE spectrum «m12»~K less. For ROXs~42B~b, "
      r"the near-infrared spectra give «i42»~K more, and the MUSE spectrum «m42»~K less. The MUSE spectrum alone "
      r"constrains the gravity less: the \(16\)--\(84\%\) intervals of \(\log g\) are «gm» wide, against «gt» with all "
      r"the spectra. With DRIFT-PHOENIX, the near-infrared spectra alone give the temperature of all the spectra within "
      r"\(«di12»\)~K for ROXs~12~b and \(«di42»\)~K for ROXs~42B~b. The MUSE spectrum alone gives «dm12»~K less for "
      r"ROXs~12~b, and its model error in that spectrum is still «fdr», against «fbt» with BT-Settl.",
      dict(max=lambda c: arriba(10)(DT(c)["bt_max_abs_dif"]),
           i12=lambda c: rango(min(DT(c)["bt_12b_ir"]), max(DT(c)["bt_12b_ir"]), k5),
           m12=lambda c: rango(min(-x for x in DT(c)["bt_12b_muse"]), max(-x for x in DT(c)["bt_12b_muse"]), k5),
           i42=lambda c: rango(min(DT(c)["bt_42b_ir"]), max(DT(c)["bt_42b_ir"]), k5),
           m42=lambda c: rango(min(-x for x in DT(c)["bt_42b_muse"]), max(-x for x in DT(c)["bt_42b_muse"]), k5),
           gm=lambda c: rango(*DT(c)["bt_ancho_logg_muse"], d2, d1), gt=lambda c: rango(*DT(c)["bt_ancho_logg_todo"], d2, d1),
           di12=lambda c: arriba(5)(DT(c)["drift_12b_ir_absmax"]), di42=lambda c: arriba(5)(DT(c)["drift_42b_ir_absmax"]),
           dm12=lambda c: rango(min(-x for x in DT(c)["drift_12b_muse"]), max(-x for x in DT(c)["drift_12b_muse"]), k10),
           fdr=lambda c: por_ciento(*DT(c)["fb_muse_solo_12b"]["drift"]), fbt=lambda c: por_ciento(*DT(c)["fb_muse_solo_12b"]["btsettl"])),
      "T_eff (p50) de cada conjunto menos la de todos los espectros, por A_V: con BT-Settl, máx. |dif| hacia arriba a "
      "10 K y los rangos a 5 K; anchos 16–84 % de log g (mín. con 2 decimales, máx. con 1); con DRIFT-PHOENIX, máx. "
      "|dif| del IR hacia arriba a 5 K y la de MUSE de 12 b a 10 K; f_b de MUSE de los ajustes solo a MUSE de 12 b",
      guarda=lambda c: (min(DT(c)["bt_12b_ir"]) > 0 and max(DT(c)["bt_12b_muse"]) < 0 and min(DT(c)["bt_42b_ir"]) > 0
                        and max(DT(c)["bt_42b_muse"]) < 0 and max(DT(c)["drift_12b_muse"]) < 0
                        and DT(c)["fb_muse_solo_12b"]["btsettl"][1] < DT(c)["fb_muse_solo_12b"]["drift"][0]))
    A("apB_datos_av", "Ap. B",
      r"the MUSE spectrum alone gives \(A_V = «a»\)~mag (BT-Settl) and \(«b»\)~mag (DRIFT-PHOENIX) for ROXs~12~b, and "
      r"the near-infrared spectra alone give \(«c»\) and \(«d»\)~mag; all four values are above the extinction of the "
      r"primary. For ROXs~42B~b, the MUSE spectrum alone gives \(«e»\) and \(«f»\)~mag, and the near-infrared spectra "
      r"\(«g»\) and \(«h»\)~mag.",
      {k: (lambda kk: lambda c: d1(DT(c)["av_prior"][kk]))(kk) for k, kk in zip(
          "abcdefgh", [f"{s}_{rj}_{x}" for s in SLUGS for x in ("muse", "ir") for rj in ("btsettl", "drift")])},
      "A_V (p50) con prior de cada conjunto y rejilla",
      guarda=lambda c: all(DT(c)["av_prior"][f"{O1}_{rj}_{x}"] > c["entrada_margen"][O1]["av0"]
                           for rj in ("btsettl", "drift") for x in ("muse", "ir")))
    A("apB_datos_delta", "Ap. B",
      r"For ROXs~12~b, \(\Delta\) is «m12»~dex with the MUSE spectrum alone, for all the values of \(A_V\), because "
      r"this spectrum gives a lower temperature, at which the noise level is lower, and with BT-Settl also a smaller "
      r"radius. With the near-infrared spectra alone, \(\Delta\) is «i18»~dex at \(A_V = 1.8\)~mag and «i30»~dex at "
      r"\(3.0\)~mag and with the prior. For ROXs~42B~b, \(\Delta\) is between \(«a»\) and \(«b»\)~dex with both sets "
      r"of spectra and both grids.",
      dict(m12=lambda c: hasta(*DT(c)["delta_12b_muse"], s2), i18=lambda c: hasta(*DT(c)["delta_12b_ir_18"], s2),
           i30=lambda c: hasta(*DT(c)["delta_12b_ir_30_prior"], s2),
           a=lambda c: s2(DT(c)["delta_42b_todo_conjunto"][0]), b=lambda c: s2(DT(c)["delta_42b_todo_conjunto"][1])),
      "Δ (p50) de los ajustes solo a MUSE y solo IR",
      guarda=lambda c: DT(c)["delta_12b_muse"][0] > 0 and max(DT(c)["bt_12b_muse"]) < 0 and max(DT(c)["drift_12b_muse"]) < 0)

    # ---------------------------------------------------------------- apéndice B: AMES-Cond (Fig. B.6)
    A("apB_ames_cond", "Ap. B",
      r"The near-infrared spectra alone give the temperature of all the spectra within \(«i12»\)~K for ROXs~12~b and "
      r"\(«i42»\)~K for ROXs~42B~b. For ROXs~12~b, the MUSE spectrum alone gives the same temperature (\(«tm»\) against "
      r"\(«tt»\)~K), but \(\log g = «gm»\) against \(«gt»\). For ROXs~42B~b, it gives \(«dT»\)~K more (\(«t42»\)~K) and "
      r"a radius of \(«Rm»\)~\(R_{\rm J}\) against \(«Rt»\)~\(R_{\rm J}\). In all the fits to the MUSE spectrum alone, "
      r"the posterior of \(\log g\) extends to the lower edge of the grid (\(2.5\)), and the model error in that "
      r"spectrum is «fac», against «fbt» with BT-Settl. With the MUSE spectrum alone, \(\Delta\) is «d12»~dex for "
      r"ROXs~12~b and «d42»~dex for ROXs~42B~b, for all the values of \(A_V\).",
      dict(i12=lambda c: arriba(10)(abs(AC(c)["ir_menos_todo_prior"][O1])),
           i42=lambda c: arriba(10)(abs(AC(c)["ir_menos_todo_prior"][O2])),
           tm=lambda c: i0(AC(c)["teff_prior"][O1]["muse"]), tt=lambda c: i0(AC(c)["teff_prior"][O1]["todo"]),
           gm=lambda c: d1(AC(c)["logg_prior_12b"]["muse"]), gt=lambda c: d1(AC(c)["logg_prior_12b"]["todo"]),
           dT=lambda c: k10(AC(c)["teff_prior"][O2]["muse"] - AC(c)["teff_prior"][O2]["todo"]),
           t42=lambda c: i0(AC(c)["teff_prior"][O2]["muse"]),
           Rm=lambda c: d2(AC(c)["R_prior_42b"]["muse"]), Rt=lambda c: d2(AC(c)["R_prior_42b"]["todo"]),
           fac=lambda c: por_ciento(*AC(c)["f_muse_solo"]), fbt=lambda c: por_ciento(*DT(c)["fb_muse_solo_bt_todos"]),
           d12=lambda c: hasta(min(v for k, v in AC(c)["delta_muse"].items() if k.startswith(f"{O1}_")),
                               max(v for k, v in AC(c)["delta_muse"].items() if k.startswith(f"{O1}_")), s2),
           d42=lambda c: hasta(min(v for k, v in AC(c)["delta_muse"].items() if k.startswith(f"{O2}_")),
                               max(v for k, v in AC(c)["delta_muse"].items() if k.startswith(f"{O2}_")), s2)),
      "AMES-Cond con prior: |T_IR − T_todo| hacia arriba a 10 K; T_eff, log g y R de solo MUSE frente a todo; f_b de "
      "MUSE de los ajustes solo a MUSE (AMES-Cond y BT-Settl, los dos compañeros); Δ de solo MUSE",
      guarda=lambda c: (RJ(c)["sin_borde_r500_ambos"]["ames_cond"] and not any(
          v for g, v in RJ(c)["sin_borde_r500_ambos"].items() if g != "ames_cond")
          and abs(AC(c)["teff_prior"][O1]["muse"] - AC(c)["teff_prior"][O1]["todo"]) < 20
          and AC(c)["teff_prior"][O2]["muse"] > AC(c)["teff_prior"][O2]["todo"]
          and AC(c)["logg_p2_5_muse_max"] < AC_LOGG_BORDE
          and DT(c)["fb_muse_solo_bt_todos"][1] < AC(c)["f_muse_solo"][0]))

    # ---------------------------------------------------------------- Tabla B.5 (filas y nota)
    A("B5_T", "Tabla B.5",
      r"\quad \(A_V = 1.8\)~mag & «t18» \\ \quad \(A_V = 3.0\)~mag & «t30» \\ \quad \(A_V\) with prior & «tpr» \\ "
      r"\(A_V\) with prior (mag) & «av» \\ Equivalent type & «tipo» \\ \(\log g\) & «g» \\",
      dict(t18=lambda c: fila_T(c, "1.8"), t30=lambda c: fila_T(c, "3"), tpr=lambda c: fila_T(c, "prior"),
           av=lambda c: " & ".join(rf"\({d1(B6(c)[f'B5_{s}_{g}']['av_prior'])}\)" for s in SLUGS for g in ("btsettl", "drift")),
           tipo=lambda c: " & ".join(tipo_tabla(tipo(c, s, g)) for s in SLUGS for g in ("btsettl", "drift")),
           g=lambda c: " & ".join(rango(*B6(c)[f"B5_{s}_{g}"]["logg"], d1) for s in SLUGS for g in ("btsettl", "drift"))),
      "medianas de las 12 cadenas de la Tabla B.6; log g: mín. p16 – máx. p84 de las tres A_V; tipo equivalente")
    A("B5_D", "Tabla B.5",
      r"\quad \(A_V = 1.8\)~mag & «d18» \\ \quad \(A_V = 3.0\)~mag & «d30» \\ \quad \(A_V\) with prior & «dpr» \\",
      dict(d18=lambda c: fila_D(c, "1.8"), d30=lambda c: fila_D(c, "3"), dpr=lambda c: fila_D(c, "prior")),
      "Δ (p50) de las 12 cadenas; cota (<) para ROXs 42B b")
    A("B5_evo", "Tabla B.5",
      r"\quad \(T_{\rm eff}\) (K) & \multicolumn{2}{c}{\(«t12»\pm«e12»\)} & \multicolumn{2}{c}{\(«t42»\pm«e42»\)} \\ "
      r"\quad \(\Delta\) (dex) & \multicolumn{2}{c}{\(«d12»\)} & \multicolumn{2}{c}{\(<«d42»\)} \\",
      dict(t12=lambda c: i0(evo(c, O1)["teff_p50"]), e12=lambda c: k10(evo(c, O1)["teff_semiancho"]),
           t42=lambda c: i0(evo(c, O2)["teff_p50"]), e42=lambda c: k10(evo(c, O2)["teff_semiancho"]),
           d12=lambda c: s2(evo(c, O1)["delta_p50"]), d42=lambda c: s2(evo(c, O2)["delta_p50"])),
      "cadenas `evo_solo` ATMO2020 (L_bol publicada, A_V adoptada): T_eff p50 ± semiancho 16–84 % (a 10 K), Δ p50")
    A("B5_nota_40", "Tabla B.5 (nota)",
      r"The statistical uncertainties of \(T_{\rm eff}\) are below \(«x»\)~K (\(16\)--\(84\%\)), and the systematic "
      r"error of the method is about \(150\)~K.",
      dict(x=lambda c: arriba(10)(B6(c)["semiancho_teff_max_global"])),
      "mayor semiancho 16–84 % de T_eff de las 12 cadenas, hacia arriba a 10 K")
    A("B5_nota_tipo", "Tabla B.5 (nota)",
      r"\(\geq\)«t» means as cool as the coolest «t» template or cooler.",
      dict(t=lambda c: unico([tipo(c, O2, g)["tipos"][0] for g in ("btsettl", "drift")
                              if tipo(c, O2, g)["clase"] == "posterior"])),
      "el tipo más tardío de la calibración, si algún compañero queda por debajo de todas las plantillas")
    A("B5_nota_semillas", "Tabla B.5 (nota)",
      r"Two more samplings with other seeds change the medians by up to \(«t»\)~K in \(T_{\rm eff}\), \(«a»\)~mag in "
      r"\(A_V\) and \(«d»\)~dex in \(\Delta\).",
      dict(t=lambda c: i0(SEM(c)["max_ptp_teff"]), a=lambda c: d2(SEM(c)["max_ptp_av_prior"]), d=lambda c: d2(SEM(c)["max_ptp_delta"])),
      "máx. diferencia entre las semillas 1–3 de la misma cadena (T_eff y Δ: las 12; A_V: las 4 con prior)")
    A("B5_nota_bhac15", "Tabla B.5 (nota)",
      r"these give the same values within \(«t»\)~K and \(«d»\)~dex, but \(«f»\%\) of the samples of ROXs~42B~b fall "
      r"below their lowest mass.",
      dict(t=lambda c: arriba(10)(max(abs(int(i0(evo(c, s, "BHAC15")["teff_p50"])) - int(i0(evo(c, s)["teff_p50"])))
                                          for s in SLUGS)),
           d=lambda c: d2(math.ceil(100 * max(abs(evo(c, s, "BHAC15")["delta_p50"] - evo(c, s)["delta_p50"])
                                              for s in SLUGS) - 1e-9) / 100),
           f=lambda c: i0(100 * evo(c, O2, "BHAC15")["fuera_de_la_rejilla"])),
      "`evo_solo` BHAC15 frente a ATMO2020: la T_eff se compara como se imprime en la tabla (2436/2420 y 2250/2254 K; "
      "la mayor diferencia, hacia arriba a 10 K). Δ con las medianas sin redondear: 0.017 dex en ROXs 12 b, hacia "
      "arriba a 0.01 → «0.02» (autor 2026-10-28, «corrige las tablas»; antes «0.01», la diferencia de los valores "
      "impresos 0.00/+0.01 y −0.33/−0.34). Fracción de muestras de 42B b fuera de la rejilla de BHAC15")

    # ---------------------------------------------------------------- Tabla B.6 (nota)
    A("B6_nota", "Tabla B.6 (nota)",
      r"The fits use \(«n1»\) live points with BT-Settl and \(A_V\) fixed, and \(«n2»\) in the other cases. With two "
      r"more seeds, \(\ln Z\) changes by up to \(«lf»\) with a fixed \(A_V\) and \(«lp»\) with the prior, and the "
      r"medians change by less than their intervals. \(\Delta\ln Z\) is «x» at \(A_V = 3.0\) than at \(1.8\)~mag for "
      r"ROXs~12~b and «y» for ROXs~42B~b, with both grids, but it does not measure the extinction.",
      dict(n1=lambda c: str(unico(NL(c)["b6_bt_fija"])), n2=lambda c: str(unico(NL(c)["b6_resto"])),
           lf=lambda c: d1(SEM(c)["max_ptp_logz_fija"]), lp=lambda c: d1(SEM(c)["max_ptp_logz_prior"]),
           x=lambda c: dlnz_frase(c)[0], y=lambda c: dlnz_frase(c)[1]),
      "`nlive`; máx. diferencia de ln Z entre las semillas 1–3; ¿ln Z a 3.0 mayor que a 1.8?",
      guarda=lambda c: SEM(c)["medianas_menores_que_intervalos"])
    A("B6_nota_avlibre", "Tabla B.6 (nota)",
      r"With a free \(A_V\) and \(«n»\) live points, the fits give \(«a»\) (BT-Settl) and \(«b»\)~mag (DRIFT-PHOENIX) "
      r"for ROXs~12~b, and \(«c»\) and \(«d»\)~mag for ROXs~42B~b. In the «cuantos» fits of «quien», the evidence is "
      r"higher than in all the fits of this table.",
      dict(n=lambda c: str(unico(v["nlive"] for v in CR(c)["avlibre"].values())),
           a=lambda c: d1(CR(c)["avlibre"][f"{O1}_btsettl"]["av"]), b=lambda c: d1(CR(c)["avlibre"][f"{O1}_drift"]["av"]),
           c=lambda c: d1(CR(c)["avlibre"][f"{O2}_btsettl"]["av"]), d=lambda c: d1(CR(c)["avlibre"][f"{O2}_drift"]["av"]),
           cuantos=lambda c: avlibre_frase(c)[0], quien=lambda c: avlibre_frase(c)[1]),
      "las cuatro cadenas con A_V libre: A_V (p50) y ln Z frente al mejor de la Tabla B.6 del compañero")

    # ---------------------------------------------------------------- Tabla B.7 (nota)
    A("B7_nota", "Tabla B.7 (nota)",
      r"(the \(16\)--\(84\%\) intervals are \(\pm«x»~dex)",
      dict(x=lambda c: rango(*RJ(c)["delta_intervalo_semiancho"], d2)[2:]),
      "semiancho (p84 − p16)/2 de Δ en todas las cadenas de la Tabla B.7")
    A("B7_nota_nlive", "Tabla B.7 (nota)",
      r"All the fits use \(«n»\) live points, except those of BT-Settl at \(R = 500\) with \(A_V\) fixed (\(«m»\)).",
      dict(n=lambda c: str(unico(RJ(c)["nlive_resto"])), m=lambda c: str(unico(RJ(c)["nlive_bt500_fija"]))),
      "`nlive` de las cadenas de la Tabla B.7")
    return L


def cifras() -> list[Cifra]:
    """Las frases del paper con cifras de las cadenas, en el orden del texto (con los objetos de targets/)."""
    if not SLUGS:
        fija_objetos()
    return _cifras()


def evalua(c, cs: list[Cifra], tex: str | None = None):
    """Escribe cada frase y, con `tex`, la busca. Devuelve una lista de dicts (una por `Cifra`)."""
    norm, pos = normaliza(tex) if tex is not None else (None, None)
    out = []
    for it in cs:
        r = dict(id=it.id, donde=it.donde, que=it.que, campos={}, frase=None, estado=None, guarda=None, linea=None,
                 discrepancia=it.discrepancia or None)
        try:
            for k, f in it.campos.items():
                r["campos"][k] = f(c)
            frase = it.frase
            for k, v in r["campos"].items():
                frase = frase.replace(f"«{k}»", v)
            r["frase"] = frase
            if "«" in frase:
                raise ValueError(f"campo sin valor en la frase: {frase}")
            if it.guarda is not None:
                r["guarda"] = bool(it.guarda(c))
        except Faltan as exc:
            r["estado"] = "sin cadenas"
            r["error"] = f"faltan {len(exc.args[0])} cadenas: {', '.join(exc.args[0][:4])}…"
            out.append(r)
            continue
        except (ValueError, KeyError) as exc:
            r["estado"] = "la frase cambia"
            r["error"] = f"{type(exc).__name__}: {exc}"
            out.append(r)
            continue
        if norm is None:
            r["estado"] = "calculada"
        else:
            buscada = colapsa(r["frase"])
            n = norm.count(buscada)
            if n == 1:
                p = pos[norm.index(buscada)]
                r["linea"] = tex.count("\n", 0, p) + 1
                r["estado"] = "en el paper"
            else:
                r["estado"] = "NO está en el paper" if n == 0 else f"está {n} veces"
        out.append(r)
    return out


def falla(r):
    """¿Hace fallar la comprobación? Una frase que no está (salvo `discrepancia` conocida), una guarda falsa, una
    frase que cambia de sentido o cadenas que faltan."""
    if r["guarda"] is False or r["estado"] in ("sin cadenas", "la frase cambia"):
        return True
    return r["estado"] not in ("en el paper", "calculada") and not r["discrepancia"]


def informe_md(res, cadenas: Cadenas, tex_path=None):
    n_ok = sum(r["estado"] == "en el paper" for r in res)
    lineas = ["# Cifras de la prosa del paper desde las cadenas de T_eff", "",
              f"- cadenas leídas: {len(cadenas.por_etiqueta)}; sombreadas (misma etiqueta en otra carpeta, no usadas): "
              f"{len(cadenas.sombreadas)}",
              f"- cadenas usadas: {len(cadenas.usadas)} ("
              + ", ".join(f"{n} de `{d}`" for d, n in cadenas.procedencia().items()) + ")",
              "- usadas que no vienen de la primera carpeta, por lo que leen: "
              + (", ".join(f"{k} {v}" for k, v in sorted(cadenas.fuera_de_la_primera().items())) or "ninguna"),
              f"- frases: {len(res)}" + (f"; en `{tex_path}`: {n_ok}" if tex_path else ""),
              f"- fallan: {sum(falla(r) for r in res)}; con discrepancia conocida: "
              f"{sum(bool(r['discrepancia']) and r['estado'] != 'en el paper' for r in res)}", "",
              "| id | dónde | línea | estado | guarda | cifras |", "|---|---|---|---|---|---|"]
    for r in res:
        cif = "; ".join(f"{k} = `{v}`" for k, v in r["campos"].items())
        g = "—" if r["guarda"] is None else ("sí" if r["guarda"] else "**NO**")
        est = r["estado"] if not falla(r) else f"**{r['estado']}**"
        lineas.append(f"| {r['id']} | {r['donde']} | {r['linea'] or '—'} | {est} | {g} | {cif} |")
    lineas += ["", "## Frases", ""]
    for r in res:
        lineas.append(f"### {r['id']} ({r['donde']}" + (f", l. {r['linea']}" if r["linea"] else "") + f") — {r['estado']}")
        lineas.append("")
        lineas.append(f"- qué: {r['que']}")
        if r.get("error"):
            lineas.append(f"- **{r['error']}**")
        if r["discrepancia"]:
            lineas.append(f"- discrepancia conocida: {r['discrepancia']}")
        if r["guarda"] is False:
            lineas.append("- **la guarda falla: la frase ya no dice lo que dicen las cadenas**")
        if r["frase"]:
            lineas += ["", "```latex", r["frase"], "```"]
        lineas.append("")
    return "\n".join(lineas) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cadenas", type=Path, action="append", required=True,
                    help="directorio de cadenas de `teff_bayes.py ajusta` (repetible; gana el primero)")
    ap.add_argument("--project-root", type=Path, default=ROOT,
                    help="raíz que contiene runs/ (por defecto la del repo; necesaria desde un worktree)")
    ap.add_argument("--tex", type=Path, help="main.tex del paper: busca cada frase")
    ap.add_argument("--json", type=Path, help="escribe las cuentas y las frases en json")
    ap.add_argument("--md", type=Path, help="escribe el informe en markdown (si no, a la salida estándar)")
    a = ap.parse_args(argv)
    warnings.simplefilter("ignore")
    K = Cadenas(a.cadenas)
    calc = Calculo(K, a.project_root)
    tex = a.tex.read_text() if a.tex else None
    res = evalua(calc.c, cifras(), tex)
    md = informe_md(res, K, a.tex)
    if a.md:
        a.md.parent.mkdir(parents=True, exist_ok=True)
        a.md.write_text(md)
        print(f"escrito {a.md}")
    else:
        print(md)
    if a.json:
        a.json.parent.mkdir(parents=True, exist_ok=True)
        a.json.write_text(json.dumps(dict(cuentas=calc.c, frases=res, sombreadas=K.sombreadas,
                                          usadas={e: K.por_etiqueta[e]["_json"] for e in sorted(K.usadas)},
                                          procedencia=K.procedencia()),
                                     indent=1, ensure_ascii=False, default=str))
        print(f"escrito {a.json}")
    n = sum(falla(r) for r in res)
    print(f"{len(res)} frases; fallan {n}", file=sys.stderr)
    return 1 if n else 0


if __name__ == "__main__":
    sys.exit(main())
