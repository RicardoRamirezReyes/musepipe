#!/usr/bin/env python
"""Los espectros del IR que se repiten en una banda: las cifras del apéndice D.12 del paper y sus Tablas D.3 y D.4.

Solo lectura y fuera de la cadena (apéndice D.12, «Two spectra in the same band», `app:checks:bands`; 2026-11-03).
Lee las cadenas de `teff_bayes.py ajusta` (json + npz) y no lanza ninguna. La J de ROXs 12 b y la H y la K de
ROXs 42B b tienen dos espectros cada una; el D.12 prueba qué pasa al contar cada banda una vez. Las opciones son las
de `teff_bayes.py`:

- `--banda-una-vez` (json `banda_una_vez`): el término de cada espectro del IR × 1/N, con N los espectros de su banda;
- `--pendiente-ir ANCHO` (`pendiente_ir`): una inclinación libre por espectro del IR, uniforme en ±ANCHO por 0.1 µm;
- `--quita-ir SEG…` (`quita_ir`): sin esos espectros;
- `--anclas-ir` (`anclas_ir`) y `--ruido-de SLUG` (`ruido_de`): las pruebas con plantillas.

Cada cadena se clasifica por su json, nunca por la etiqueta ni por el nombre del fichero: objeto (los datos o una
plantilla, `verdad.plantilla`, con el «disfraz» `slug`), rejilla, datos (`todo`/`ir`), A_V fija o prior, semilla,
resolución, peso, inclinación, espectros quitados, anclas (sin la opción, las propias del objeto: `targets/<slug>.json`
→ `ir_spectra.colour_excess.bands`) y ruido (sin la opción, el del propio disfraz). Dos cadenas con la misma clave
son un error. De `--cadenas` entran las que no son variantes (`teff_bayes._es_variante`); de `--variantes`, las que no
llevan más variantes que las cinco de arriba.
Solo `modo` `libre` con el prior de la primaria; las plantillas, con `NLIVE_PLANTILLAS` puntos vivos. Los dos objetos
salen de `targets/` (`teff_cifras_texto.fija_objetos`), y de cada uno sus espectros del IR y sus bandas dobles
(`ir_spectra.spectra`). ATMO 2020 es, sobre cada objeto, la única rejilla de ATMO con las cuatro formas (en los lotes,
la de 2000 K para ROXs 12 b y la extendida a 1500 K para ROXs 42B b), y `ATMO_PLANTILLAS` sobre las plantillas; las
rejillas de la Tabla D.4 son las de la Tabla B.7 con las cuatro formas, menos BT-Settl, DRIFT-PHOENIX y ATMO.

Cuatro formas de tratar el IR (MUSE + IR, semilla 1): «ref» = cada espectro con peso uno y sin inclinación (las
cadenas de las Tablas B.6 y B.7), «peso», «incl» (±0.3) y «las dos». T = la mediana de T_eff (`posterior.teff`);
Δ = la mediana del margen sobre el nivel cromosférico (`teff_bayes.margen_de_corrida`, el de la Tabla B.7).

Lo que da (todo con su clave en el json y en el informe):

- la Tabla D.3 (`tab:bands`): T y Δ en las cuatro formas con BT-Settl, DRIFT-PHOENIX y ATMO 2020, y las A_V con el
  prior de su nota; y la Tabla D.4 (`tab:bands_grids`): con las demás rejillas de la Tabla B.7 (menos BT-Settl a
  R = 250), lo más que se mueven T y Δ frente a «ref», Δ con el prior y la A_V donde Δ cambia de signo;
- del texto, lo que sale de las cadenas de los datos: los cambios de las cuatro formas y la A_V donde Δ cambia de
  signo (interpolación lineal entre 1.8 y 3.0 mag) con cada rejilla, y cuánto queda la A_V del prior por encima;
  Δln Z de la inclinación sin y con el peso; las inclinaciones ajustadas entre los dos espectros de una banda (la
  mediana por muestra de p_B − p_A) frente a las medidas (`MEDIDO`); lo que cambian T y Δ con un espectro por banda,
  y dónde cae el ajuste con peso en ese rango; el ancho ±0.1 frente a ±0.3; ROXs 12 b con solo el IR e inclinación;
  los recuentos («20 of 24», «33 of 36», «71 of 72»);
- de las plantillas (ATMO 2020 y BT-Settl, 250 puntos vivos): el cambio entre los dos disfraces, Δ_pl = T(42B b) −
  T(12 b), al cambiar solo el ruido, solo los píxeles, solo las anclas, con solo el IR, con un espectro por banda y
  con el peso, sus medianas y cuántas pasan de 150 K, y el ruido entre semillas.

Con `--tex`, compara las Tablas D.3 y D.4 del paper fila a fila al redondeo impreso (sale con 1 si no coinciden), y
busca en el texto del D.12 cada cifra de `cifras_texto` con el redondeo y la frase con que se imprime: las que no
aparecen van al informe y a stderr, y con `--estricto` también hacen salir con 1.

Lo que NO da (y de dónde sale):

- la diferencia de inclinación MEDIDA entre los espectros de un par (11 %, y 7 y 9 %; −3.6, −2.1 y +4.7 % por
  0.1 µm), lo que 100 K cambian la inclinación de la H (2.6 %) y su equivalente (140 K): de los espectros, no de las
  cadenas (`~/muse_noche_20261031_quita_ir_reales/dos_h/`, `cociente_par.txt`); aquí van como constantes, solo para
  comparar;
- 140–210 K (el cambio de ATMO 2020 entre compañeros en la prueba de plantillas del apéndice B):
  `scripts/paper_cifras_atmo.py`;
- 0.05–0.09 dex (la dispersión entre rejillas, D.4): `scripts/delta_entre_rejillas.py`, y con ella el «2.6 times»
  del cierre (los 0.23 dex de aquí entre los 0.09 del D.4); 0.25 dex (la dispersión del nivel entre estrellas de
  clase III): apéndice B.

Una definición que conviene saber: «12 b con solo el IR» son las cadenas con la inclinación libre y SIN el peso (el
párrafo del D.12 es el de la inclinación); con el peso también, va aparte en el json (`12b_solo_ir_incl.t_con_peso`).

Uso (las carpetas de la máquina del autor; `--cadenas` para las de referencia, `--variantes` para las de los lotes):

    python scripts/bandas_ir_compara.py --project-root RAIZ_CON_RUNS \\
        --cadenas ~/muse_noche_20261024_c4/cadenas \\
        --cadenas ~/muse_sesion_20261026/claudio/atmo/res/atmo --cadenas ~/muse_sesion_20261026/claudio/atmo/res/atmo1500 \\
        --cadenas ~/muse_sesion_20261026/claudio/atmo/res/atmo_datos --cadenas ~/muse_sesion_20261026/claudio/atmo/res/plantillas \\
        --cadenas ~/muse_noche_20261027/atmo_s2/res/atmo_s2 --cadenas ~/muse_noche_20261027/atmo_s2/res/atmo1500_s2 \\
        --cadenas ~/muse_noche_20261027/atmo_s2/res/atmo_datos_s2 --cadenas ~/muse_noche_20261027/atmo_s2/res/plantillas_s1 \\
        --cadenas ~/muse_sesion_20261018/teff_datos_logg5/res \\
        --variantes ~/muse_noche_20261030_plantillas_atmo/res --variantes ~/muse_noche_20261030_anclas_ir/res \\
        --variantes ~/muse_noche_20261031_quita_ir/res --variantes ~/muse_noche_20261031_quita_ir_reales/res \\
        --variantes ~/muse_noche_20261101_pendiente_ir/res --variantes ~/muse_noche_20261102_banda_una_vez/res \\
        --variantes ~/muse_noche_20261102_drift_ir/res --variantes ~/muse_noche_20261103_rejillas_b7/res \\
        --tex paper/main.tex --json salida.json --md salida.md
"""
from __future__ import annotations

import argparse
import json
import re
import statistics as st
import sys
import warnings
from itertools import product
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import teff_bayes as tb  # noqa: E402
from scripts import teff_cifras_texto as tc  # noqa: E402

#: los dos objetos del paper (O1, el de la línea detectada; O2, el de la cota) y su nombre en el texto: los fija
#: `fija_objetos` desde `targets/` (ningún nombre de objeto va en el código, `tests/test_no_hardcoded_target.py`)
SLUGS: tuple = ()
O1 = O2 = ""
AVS = (1.8, 3.0, None)
FORMAS = ("ref", "peso", "incl", "las dos")
#: el ancho de la inclinación de las cuatro formas y el de la prueba de sensibilidad (por 0.1 µm)
ANCHO, ANCHO_CORTO = 0.3, 0.1
#: las claves de las opciones del D.12 en el json de `ajusta`
CLAVES_D12 = ("banda_una_vez", "pendiente_ir", "quita_ir", "anclas_ir", "ruido_de")
#: ATMO 2020 sobre las plantillas (la rejilla extendida a 1500 K)
ATMO_PLANTILLAS = "atmo_ceq1500"
#: los puntos vivos de las cadenas de plantillas
NLIVE_PLANTILLAS = 250
#: la resolución de análisis de cada rejilla (SPHINX solo a R = 250, como en la Tabla B.7)
R_ANALISIS = {"sphinx": 250}
#: las rejillas de la Tabla D.3 («atmo» es la de `ATMO_DATOS` de cada objeto)
D3 = ("btsettl", "drift", "atmo")
NOMBRE_D3 = {"btsettl": "BT-Settl", "drift": "DRIFT-PHOENIX", "atmo": "ATMO~2020"}
#: la inclinación MEDIDA del cociente B/A de los dos espectros de una banda (%/0.1 µm;
#: ~/muse_noche_20261031_quita_ir_reales/dos_h/cociente_par.txt, 2026-10-31): de los espectros, no de las cadenas;
#: solo para comparar. Sus claves orientan cada par (B, A).
MEDIDO = {("OSIRIS_J", "NIFS_J"): 4.69, ("OSIRIS_H", "NIFS_H"): -3.60, ("SINFONI_K", "OSIRIS_K"): -2.12}
#: lo propio de cada objeto, {slug: ...}; lo fijan `fija_objetos` (de `targets/`) y `fija_rejillas` (de las cadenas):
#: SEGMENTOS {espectro del IR: banda} (`ir_spectra.spectra`); PARES [(B, A)], los dos espectros de una misma banda;
#: UNA_VEZ [quitados], las configuraciones con un espectro por banda (se quita uno de cada par); MISMOS (quitados),
#: en las plantillas, los espectros que el otro objeto no tiene; PROPIAS, las anclas propias; ATMO_DATOS, la rejilla
#: de ATMO 2020 con las cuatro formas sobre los datos; D4, las rejillas de la Tabla B.7 con las cuatro formas, menos
#: BT-Settl, DRIFT-PHOENIX y ATMO, en el orden de `teff_bayes.REJILLAS_TABLA`.
SEGMENTOS: dict = {}
PARES: dict = {}
UNA_VEZ: dict = {}
MISMOS: dict = {}
PROPIAS: dict = {}
ATMO_DATOS: dict = {}
D4: dict = {}
#: las filas de las comparaciones de los datos: (A_V fija, semilla)
FILAS = [(1.8, 1), (3.0, 1), (None, 1), (None, 2)]
#: ±150 K, el sistemático del método (apéndice B)
UMBRAL = 150.0
ORDEN_SPT = ["M6.5", "M7", "M7.5", "M8", "M8.5", "M9", "M9.5", "L0", "L1"]


# ================================================================== clasificación
def fija_objetos(root=ROOT):
    """Fija `SLUGS`, `O1`, `O2` (`teff_cifras_texto.fija_objetos`, que también fija sus `NOMBRES`) y, de
    `targets/<slug>.json`, `SEGMENTOS`, `PARES`, `UNA_VEZ`, `MISMOS` y `PROPIAS`."""
    global SLUGS, O1, O2
    SLUGS = tc.fija_objetos(root)
    O1, O2 = SLUGS
    for s in SLUGS:
        esp = json.loads((Path(root) / "targets" / f"{s}.json").read_text())["ir_spectra"]["spectra"]
        SEGMENTOS[s] = {k: v["band"] for k, v in esp.items()}
        PROPIAS[s] = anclas_propias(s, root)
    for s in SLUGS:
        PARES[s] = pares_de(SEGMENTOS[s])
        UNA_VEZ[s] = [tuple(sorted(x)) for x in product(*PARES[s])]
        MISMOS[s] = tuple(sorted(set(SEGMENTOS[s]) - set(SEGMENTOS[O2 if s == O1 else O1])))
    return SLUGS


def pares_de(segmentos):
    """[(B, A)]: los pares de espectros de una misma banda (J, H, K), orientados como en `MEDIDO` si están."""
    por = {}
    for s, b in segmentos.items():
        por.setdefault(b, []).append(s)
    out = []
    for b in "JHK":
        v = sorted(por.get(b, []))
        if len(v) == 2:
            out.append(next((k for k in MEDIDO if set(k) == set(v)), tuple(v)))
        elif len(v) > 2:
            raise SystemExit(f"la banda {b} tiene {len(v)} espectros: {v}")
    return out


def cuatro_formas_completas(idx, slug, rej, propias):
    """¿Hay cadenas de los datos de `rej` en las cuatro formas y las tres A_V (MUSE + IR, semilla 1)?"""
    r = R_ANALISIS.get(rej, 500)
    return all(("datos", slug, rej, "todo", av, 1, r, *opciones(f), (), propias[slug]) in idx
               for av in AVS for f in FORMAS)


def fija_rejillas(idx):
    """Fija `ATMO_DATOS` (la única rejilla de ATMO 2020 con las cuatro formas sobre cada objeto) y `D4` (las de
    `teff_bayes.REJILLAS_TABLA` con las cuatro formas, menos BT-Settl, DRIFT-PHOENIX y ATMO)."""
    for s in SLUGS:
        rej = sorted({k[2] for k in idx if k[0] == "datos" and k[1] == s})
        atmo = [g for g in rej if g.startswith("atmo") and cuatro_formas_completas(idx, s, g, PROPIAS)]
        if len(atmo) != 1:
            raise SystemExit(f"{s}: se espera una rejilla de ATMO 2020 con las cuatro formas y hay {atmo}")
        ATMO_DATOS[s] = atmo[0]
        D4[s] = [g for g in tb.REJILLAS_TABLA if g not in ("btsettl", "drift") and not g.startswith("atmo")
                 and cuatro_formas_completas(idx, s, g, PROPIAS)]


def anclas_propias(slug, root=ROOT):
    """Las bandas que el objeto ancla a la fotometría (`targets/<slug>.json`), con el nombre de `teff_bayes.ANCLAS_IR`."""
    b = (json.loads((Path(root) / "targets" / f"{slug}.json").read_text())["ir_spectra"].get("colour_excess") or {}).get("bands", [])
    return {(): "ninguna", ("K",): "K", ("H", "J", "K"): "JHK"}[tuple(sorted(b, key="HJK".index))] if b else "ninguna"


def entra(d, de_variantes):
    """¿Entra la cadena? `libre` con el prior de la primaria, sin claves desconocidas; de `--cadenas`, sin variantes;
    de `--variantes`, sin más variantes que las del D.12."""
    if d.get("modo") != "libre" or d.get("av") != "primaria":
        return False
    if any(not k.startswith("_") and k not in tb.CLAVES_JSON and k not in tb.VARIANTES_JSON for k in d):
        return False
    if any(d.get(k, v) != v for k, v in tb.VARIANTES_JSON.items() if k not in CLAVES_D12):
        return False
    if not de_variantes and tb._es_variante(d):
        return False
    v = d.get("verdad") or {}
    if v and ("plantilla" not in v or d.get("nlive") != NLIVE_PLANTILLAS):
        return False
    return True


def clave(d, propias):
    """La clave de una cadena. Datos: ("datos", slug, rejilla, datos, A_V fija, semilla, R, peso, inclinación,
    quitados, anclas). Plantillas: ("plantilla", plantilla, disfraz, rejilla, datos, A_V fija, semilla, R, peso,
    inclinación, quitados, anclas, ruido)."""
    comun = (d.get("rejilla", "btsettl"), d.get("datos", "todo"), d.get("av_fija"), d["semilla"],
             d.get("r_analisis", 500), bool(d.get("banda_una_vez", False)), d.get("pendiente_ir"),
             tuple(sorted(d.get("quita_ir") or ())), d.get("anclas_ir") or propias[d["slug"]])
    v = d.get("verdad") or {}
    if v:
        return ("plantilla", v["plantilla"], d["slug"]) + comun + (d.get("ruido_de") or d["slug"],)
    return ("datos", d["slug"]) + comun


def indice(K, V, propias):
    """{clave: cadena} de las de referencia (`K`) y las de los lotes (`V`), `teff_cifras_texto.Cadenas`."""
    out = {}
    for fuente, de_variantes in ((K, False), (V, True)):
        for d in fuente.por_etiqueta.values():
            if not entra(d, de_variantes):
                continue
            k = clave(d, propias)
            if k in out and out[k]["_json"] != d["_json"]:
                raise SystemExit(f"dos cadenas con la clave {k}: {out[k]['_json']} y {d['_json']}")
            out[k] = d
    return out


def forma(peso, pend):
    if pend is None:
        return "peso" if peso else "ref"
    return "las dos" if peso else "incl"


def opciones(f, ancho=ANCHO):
    """(peso, inclinación) de una forma."""
    return {"ref": (False, None), "peso": (True, None), "incl": (False, ancho), "las dos": (True, ancho)}[f]


def cruce(d18, d30):
    """A_V donde Δ cambia de signo, por interpolación lineal entre 1.8 y 3.0 mag (como el paper); None si son iguales."""
    return 1.8 + 1.2 * (0 - d18) / (d30 - d18) if d30 != d18 else None


def rango(x):
    return [min(x), max(x)] if x else None


def med(x):
    return st.median(x) if x else None


# ================================================================== las cadenas de los datos
class Datos:
    """Acceso a las cadenas de los datos por configuración, con Δ en caché."""

    def __init__(self, idx, entradas, propias):
        self.idx, self.entradas, self.propias = idx, entradas, propias
        self._delta, self._pend = {}, {}

    def rejilla(self, slug, fam):
        return ATMO_DATOS[slug] if fam == "atmo" else fam

    def cadena(self, slug, fam, av, f="ref", datos="todo", semilla=1, quita=(), ancho=ANCHO):
        rej = self.rejilla(slug, fam)
        peso, pend = opciones(f, ancho)
        return self.idx.get(("datos", slug, rej, datos, av, semilla, R_ANALISIS.get(rej, 500), peso, pend,
                             tuple(sorted(quita)), self.propias[slug]))

    def t(self, d):
        return None if d is None else d["posterior"]["teff"]["p50"]

    def logg(self, d):
        return None if d is None else d["posterior"]["logg"]["p50"]

    def av(self, d):
        return None if d is None else (d["posterior"]["av"]["p50"] if d.get("av_fija") is None else d["av_fija"])

    def delta(self, d):
        if d is None:
            return None
        e = d["etiqueta"]
        if e not in self._delta:
            m = tb.margen_de_corrida(d, self.entradas[d["slug"]])
            self._delta[e] = float(np.percentile(m[np.isfinite(m)], 50))
        return self._delta[e]

    def inclinacion(self, d, b, a):
        """La mediana por muestra de p_B − p_A (%/0.1 µm) de una cadena con inclinación."""
        k = (d["etiqueta"], b, a)
        if k not in self._pend:
            z = np.load(d["_npz"])
            nom = list(z["nombres"])
            self._pend[k] = float(np.median(100.0 * (z["muestras"][:, nom.index("p_" + b)]
                                                     - z["muestras"][:, nom.index("p_" + a)])))
        return self._pend[k]


def tabla_d3(D):
    """{slug: {fam: {av: {forma: dict(t, delta, av, logg, borde, etiqueta)}}}} de la Tabla D.3."""
    out = {}
    for slug, fam, av, f in product(SLUGS, D3, AVS, FORMAS):
        d = D.cadena(slug, fam, av, f)
        if d is None:
            raise SystemExit(f"falta la cadena de la Tabla D.3: {slug} {fam} A_V {av} {f}")
        out.setdefault(slug, {}).setdefault(fam, {}).setdefault(av, {})[f] = dict(
            t=D.t(d), delta=D.delta(d), av=D.av(d), logg=D.logg(d), borde=tb.en_el_borde(d), etiqueta=d["etiqueta"])
    return out


def cuatro_formas(D, slug, rejs):
    """Lo mismo que `tabla_d3` con las rejillas `rejs` (nombres de `teff_bayes`)."""
    out = {}
    for g, av, f in product(rejs, AVS, FORMAS):
        d = D.cadena(slug, g, av, f)
        if d is None:
            raise SystemExit(f"falta la cadena: {slug} {g} A_V {av} {f}")
        out.setdefault(g, {}).setdefault(av, {})[f] = dict(t=D.t(d), delta=D.delta(d), av=D.av(d), logg=D.logg(d),
                                                           borde=tb.en_el_borde(d), etiqueta=d["etiqueta"])
    return out


def mueve(c):
    """De {av: {forma: fila}}: lo más que se mueven T y Δ frente a «ref» (las tres A_V y las tres formas), y Δ y A_V
    con el prior en las cuatro."""
    dT = [abs(c[av][f]["t"] - c[av]["ref"]["t"]) for av in AVS for f in FORMAS[1:]]
    dD = [abs(c[av][f]["delta"] - c[av]["ref"]["delta"]) for av in AVS for f in FORMAS[1:]]
    return dict(max_dT=max(dT), max_dD=max(dD), n_sobre_umbral=sum(x > UMBRAL for x in dT),
                delta_prior=[c[None][f]["delta"] for f in FORMAS], av_prior=[c[None][f]["av"] for f in FORMAS])


def cruces(c):
    """{forma: dict(cruce, dentro, av_prior_menos_cruce)} de {av: {forma: fila}}."""
    out = {}
    for f in FORMAS:
        x = cruce(c[1.8][f]["delta"], c[3.0][f]["delta"])
        out[f] = dict(cruce=x, dentro=x is not None and 1.8 <= x <= 3.0,
                      av_prior_menos_cruce=None if x is None else c[None][f]["av"] - x)
    return out


# ================================================================== las cifras de los datos
def cifras_datos(D, d3, d4):
    R = {}
    # --- las cuatro formas con BT-Settl y DRIFT-PHOENIX (12 b) y las tres rejillas (42B b)
    for slug, fam in product(SLUGS, D3):
        c = d3[slug][fam]
        R[f"{slug}|{fam}"] = dict(mueve(c), cruces=cruces(c),
                                  delta_por_av={str(av): rango([c[av][f]["delta"] for f in FORMAS]) for av in AVS},
                                  baja_con_peso=[c[av]["ref"]["delta"] - c[av]["peso"]["delta"] for av in AVS])
    # --- las demás rejillas
    for slug in SLUGS:
        for g in D4[slug]:
            R[f"{slug}|{g}"] = dict(mueve(d4[slug][g]), cruces=cruces(d4[slug][g]))
    # --- 12 b: los cruces con las seis rejillas
    seis = [("btsettl", d3[O1]["btsettl"]), ("drift", d3[O1]["drift"])] + \
           [(g, d4[O1][g]) for g in D4[O1]]
    cr = {(g, f): x for g, c in seis for f, x in cruces(c).items()}
    R["cruces_12b"] = dict(
        por_forma={f: dict(nuevas=rango([cr[(g, f)]["cruce"] for g in D4[O1]]),
                           seis=rango([cr[(g, f)]["cruce"] for g, _ in seis])) for f in FORMAS},
        todos=rango([x["cruce"] for x in cr.values()]),
        dentro=[sum(x["dentro"] for x in cr.values()), len(cr)],
        dentro_rango=rango([x["cruce"] for x in cr.values() if x["dentro"]]),
        extrapolados=[dict(rejilla=g, forma=f, cruce=x["cruce"], delta18=c[1.8][f]["delta"], delta30=c[3.0][f]["delta"])
                      for g, c in seis for f in FORMAS for x in [cr[(g, f)]] if not x["dentro"]],
        delta_prior=rango([c[None][f]["delta"] for _, c in seis for f in FORMAS]),
        av_prior=rango([c[None][f]["av"] for _, c in seis for f in FORMAS]),
        av_prior_nuevas=rango([d4[O1][g][None][f]["av"] for g in D4[O1] for f in FORMAS]),
        sobre_cruce=rango([x["av_prior_menos_cruce"] for x in cr.values()]),
        sobre_cruce_nuevas=rango([cr[(g, f)]["av_prior_menos_cruce"] for g in D4[O1] for f in FORMAS]),
        sobre_cruce_bt_drift=rango([cr[(g, f)]["av_prior_menos_cruce"] for g in ("btsettl", "drift") for f in FORMAS]),
        av_prior_bt_drift=rango([d3[O1][g][None][f]["av"] for g in ("btsettl", "drift") for f in FORMAS]))
    n = d4[O1]
    R["nuevas_12b"] = dict(max_dT=max(R[f"{O1}|{g}"]["max_dT"] for g in D4[O1]),
                           max_dD=max(R[f"{O1}|{g}"]["max_dD"] for g in D4[O1]),
                           delta_prior=rango([n[g][None][f]["delta"] for g in D4[O1] for f in FORMAS]))
    # --- 42B b: cuántas no cambian de signo; AMES con inclinación; Diamondback; SPHINX
    c42 = {(g, f): x for g in D4[O2] for f, x in cruces(d4[O2][g]).items()}
    q = d4[O2]
    ames = [(g, av, f) for g in ("ames_dusty", "ames_cond") for av in AVS for f in ("incl", "las dos")]
    dbs = [g for g in D4[O2] if g.startswith("diamondback")]
    f1 = [dict(av=av, forma=f, t=q["diamondback1600_f1"][av][f]["t"],
               dt=q["diamondback1600_f1"][av][f]["t"] - q["diamondback1600_f1"][av]["ref"]["t"])
          for av in AVS for f in FORMAS[1:]
          if abs(q["diamondback1600_f1"][av][f]["t"] - q["diamondback1600_f1"][av]["ref"]["t"]) > 100]
    todas42 = [d3[O2][fam][av][f]["delta"] for fam in D3 for av in AVS for f in FORMAS] + \
              [q[g][av][f]["delta"] for g in D4[O2] for av in AVS for f in FORMAS]
    R["nuevas_42b"] = dict(
        sin_cambio_de_signo=[sum(not x["dentro"] for x in c42.values()), len(c42)],
        ames_incl_t=rango([q[g][av][f]["t"] for g, av, f in ames]),
        ames_incl_logg=rango([q[g][av][f]["logg"] for g, av, f in ames]),
        ames_incl_sube=rango([q[g][av][f]["t"] - q[g][av]["ref"]["t"] for g, av, f in ames if av != 3.0]),
        ames_incl_delta={str(av): rango([q[g][av][f]["delta"] for g in ("ames_dusty", "ames_cond")
                                         for f in ("incl", "las dos")]) for av in AVS},
        ames_dusty_peso_30=dict(t=q["ames_dusty"][3.0]["peso"]["t"], logg=q["ames_dusty"][3.0]["peso"]["logg"],
                                delta=q["ames_dusty"][3.0]["peso"]["delta"]),
        diamondback_max_dT={g: R[f"{O2}|{g}"]["max_dT"] for g in dbs},
        diamondback_max_dT_sin_f1=max(R[f"{O2}|{g}"]["max_dT"] for g in dbs if g != "diamondback1600_f1"),
        diamondback_f1_lejos=f1,
        diamondback_logg_borde=[sum("logg" in q[g][av][f]["borde"] for g in dbs for av in AVS for f in FORMAS),
                                len(dbs) * len(AVS) * len(FORMAS)],
        diamondback_delta_prior=rango([q[g][None][f]["delta"] for g in dbs for f in FORMAS]),
        sphinx_max_dT=R[f"{O2}|sphinx"]["max_dT"],
        sphinx_delta_prior=rango([q["sphinx"][None][f]["delta"] for f in FORMAS]),
        delta_todas=rango(todas42),
        incl_prior_abajo=rango([d3[O2][fam][None][f]["delta"] for fam in ("btsettl", "drift")
                                for f in ("incl", "las dos")] + [q[g][None][f]["delta"] for g in ("ames_dusty", "ames_cond")
                                                                  for f in ("incl", "las dos")]),
        incl_prior_arriba=rango([d3[O2]["atmo"][None][f]["delta"] for f in ("incl", "las dos")]
                                + [q[g][None][f]["delta"] for g in dbs + ["sphinx"] for f in ("incl", "las dos")]))
    # --- el cierre: lo más que mueve Δ de 12 b cada familia
    R["cierre"] = dict(btsettl=R[f"{O1}|btsettl"]["max_dD"], drift=R[f"{O1}|drift"]["max_dD"],
                       otras=R["nuevas_12b"]["max_dD"])
    return R


def una_vez(D):
    """Con un espectro por banda (quitando), sin y con inclinación, y el ajuste con el peso: por objeto, rejilla y
    datos, las filas de `FILAS`."""
    out = []
    for pe, slug, fam, datos in product((None, ANCHO), SLUGS, ("btsettl", "atmo"), ("todo", "ir")):
        f_todos, f_peso = ("ref", "peso") if pe is None else ("incl", "las dos")
        for av, s in FILAS:
            todos = D.cadena(slug, fam, av, f_todos, datos, s)
            peso = D.cadena(slug, fam, av, f_peso, datos, s)
            uv = [D.cadena(slug, fam, av, f_todos, datos, s, q) for q in UNA_VEZ[slug]]
            if todos is None or any(u is None for u in uv):
                continue
            ts, ds = [D.t(u) for u in uv], [D.delta(u) for u in uv]
            r = dict(inclinacion=pe, slug=slug, rejilla=fam, datos=datos, av_fija=av, semilla=s, t_todos=D.t(todos),
                     d_todos=D.delta(todos), t_una_vez=ts, d_una_vez=ds, ancho_t=max(ts) - min(ts),
                     ancho_d=max(ds) - min(ds), logz_todos=todos["logz"])
            # por banda doble: lo más que difieren dos configuraciones que solo cambian el espectro de esa banda
            r["dif_banda_t"], r["dif_banda_d"] = {}, {}
            for b, a in PARES[slug]:
                banda = SEGMENTOS[slug][b]
                ij = [(i, j) for i, j in product(range(len(uv)), repeat=2) if i < j
                      and set(UNA_VEZ[slug][i]) ^ set(UNA_VEZ[slug][j]) == {a, b}]
                r["dif_banda_t"][banda] = max(abs(ts[i] - ts[j]) for i, j in ij)
                r["dif_banda_d"][banda] = max(abs(ds[i] - ds[j]) for i, j in ij)
            if peso is not None:
                tp = D.t(peso)
                r.update(t_peso=tp, d_peso=D.delta(peso), dentro=min(ts) <= tp <= max(ts),
                         menos_centro=tp - 0.5 * (min(ts) + max(ts)), logz_peso=peso["logz"])
            out.append(r)
    return out


def cifras_una_vez(D, uv):
    R = {}
    sin = [r for r in uv if r["inclinacion"] is None]
    con = [r for r in uv if r["inclinacion"] == ANCHO]
    # los datos con BT-Settl y todos los espectros: lo que cambia la elección del espectro (ancho del rango)
    for slug in SLUGS:
        rr = [r for r in sin if (r["slug"], r["rejilla"], r["datos"]) == (slug, "btsettl", "todo")]
        R[f"elegir_{slug}"] = dict(ancho_t=rango([r["ancho_t"] for r in rr]), ancho_d=rango([r["ancho_d"] for r in rr]))
    R["por_banda"] = {}
    for slug in SLUGS:
        rr = [r for r in sin if (r["slug"], r["rejilla"], r["datos"]) == (slug, "btsettl", "todo")]
        for banda in rr[0]["dif_banda_t"] if rr else ():
            R["por_banda"][f"{slug}|{banda}"] = dict(t=max(r["dif_banda_t"][banda] for r in rr),
                                                     d=max(r["dif_banda_d"][banda] for r in rr))

    def por_grupo(rows, k):
        g = {}
        for r in rows:
            if k in r:
                g.setdefault((r["slug"], r["rejilla"], r["datos"]), []).append(r[k])
        return g

    R["ancho_con_inclinacion_mediana_grupo"] = rango([med(v) for v in por_grupo(con, "ancho_t").values()])
    for nombre, rows in (("sin", sin), ("con", con)):
        conp = [r for r in rows if "t_peso" in r]
        R[f"peso_{nombre}"] = dict(dentro=[sum(r["dentro"] for r in conp), len(conp)],
                                   menos_centro_mediana_grupo=rango([med([abs(x) for x in v]) for v in
                                                                     por_grupo(conp, "menos_centro").values()]))
    # Δln Z de la inclinación: sin el peso (todos los espectros) y con el peso; 42B b con ATMO y solo el IR, aparte
    dz_sin, dz_con = {}, {}
    for slug, fam, datos in product(SLUGS, ("btsettl", "atmo"), ("todo", "ir")):
        for av, s in FILAS:
            a, b = D.cadena(slug, fam, av, "ref", datos, s), D.cadena(slug, fam, av, "incl", datos, s)
            if a is not None and b is not None:
                dz_sin[(slug, fam, datos, av, s)] = b["logz"] - a["logz"]
            a, b = D.cadena(slug, fam, av, "peso", datos, s), D.cadena(slug, fam, av, "las dos", datos, s)
            if a is not None and b is not None:
                dz_con[(slug, fam, datos, av, s)] = b["logz"] - a["logz"]
    for slug, av in product(SLUGS, AVS):   # DRIFT-PHOENIX, MUSE + IR, s1
        c = [D.cadena(slug, "drift", av, f) for f in FORMAS]
        dz_sin[(slug, "drift", "todo", av, 1)] = c[2]["logz"] - c[0]["logz"]
        dz_con[(slug, "drift", "todo", av, 1)] = c[3]["logz"] - c[1]["logz"]
    aparte = lambda k: k[:3] == (O2, "atmo", "ir") or (k[1] == "drift" and k[3] == 3.0 and k[0] == O2)  # noqa: E731
    R["dlnz_incl"] = dict(sin_peso=rango(list(dz_sin.values())),
                          con_peso=rango([v for k, v in dz_con.items() if not aparte(k)]),
                          con_peso_42b_atmo_ir=rango([v for k, v in dz_con.items() if k[:3] == (O2, "atmo", "ir")]),
                          sin_peso_42b_atmo_ir=rango([v for k, v in dz_sin.items() if k[:3] == (O2, "atmo", "ir")]),
                          con_peso_drift_42b_30=dz_con[(O2, "drift", "todo", 3.0, 1)],
                          n_sin=len(dz_sin), n_con=len(dz_con))
    # las inclinaciones ajustadas: todos los espectros, la inclinación libre sin el peso (el párrafo del D.12 que
    # las da es el de la inclinación, antes del peso), BT-Settl, DRIFT y ATMO, MUSE + IR y solo IR
    incl = {}
    for slug, fam, datos, (av, s), f in product(SLUGS, ("btsettl", "drift", "atmo"), ("todo", "ir"), FILAS, ("incl",)):
        d = D.cadena(slug, fam, av, f, datos, s)
        if d is None:
            continue
        for b, a in PARES[slug]:
            incl.setdefault(f"{b}-{a}", []).append(D.inclinacion(d, b, a))
    R["inclinaciones_bt_drift_atmo"] = {k: rango(v) + [len(v)] for k, v in incl.items()}
    # ±0.1 frente a ±0.3, todos los espectros
    sens = {}
    for slug, fam, datos, (av, s) in product(SLUGS, ("btsettl", "atmo"), ("todo", "ir"), FILAS):
        a = D.cadena(slug, fam, av, "incl", datos, s, ancho=ANCHO_CORTO)
        b = D.cadena(slug, fam, av, "incl", datos, s)
        if a is not None and b is not None:
            sens.setdefault((slug, datos), []).append(abs(D.t(a) - D.t(b)))
    R["ancho_01"] = {f"{s}|{d}": rango(v) for (s, d), v in sens.items()}
    R["ancho_01"][f"{O2}|todos"] = rango(sens.get((O2, "todo"), []) + sens.get((O2, "ir"), []))
    # 12 b con solo el IR y la inclinación (todos los espectros, sin peso)
    ts = [D.t(D.cadena(O1, fam, av, "incl", "ir", s)) for fam in ("btsettl", "atmo") for av, s in FILAS
          if D.cadena(O1, fam, av, "incl", "ir", s) is not None]
    tp = [D.t(D.cadena(O1, fam, av, "las dos", "ir", s)) for fam in ("btsettl", "atmo") for av, s in FILAS
          if D.cadena(O1, fam, av, "las dos", "ir", s) is not None]
    bordes = []
    for fam, (av, s) in product(("btsettl", "atmo"), FILAS):
        d = D.cadena(O1, fam, av, "incl", "ir", s)
        if d is not None:
            P = d["posterior"]["p_OSIRIS_J"]
            bordes.append(P["p97_5"] > 0.98 * ANCHO or P["p2_5"] < -0.98 * ANCHO)
    R["12b_solo_ir_incl"] = dict(t=rango(ts), t_con_peso=rango(tp), osiris_j_en_el_borde=[sum(bordes), len(bordes)])
    # ATMO, 12 b: Δ con el prior con una sola J, y lo que baja el peso
    a = [D.delta(D.cadena(O1, "atmo", None, "ref", "todo", 1, q)) for q in UNA_VEZ[O1]]   # s1, la de D.3
    R["atmo_12b_una_j_prior"] = rango(a)
    return R


# ================================================================== las plantillas
def plantillas(idx, propias):
    """{(rejilla, datos, semilla, peso, quitados, anclas, ruido, disfraz, plantilla): (T, log g, spt)} de las
    plantillas con el prior. `quitados` y `anclas` van normalizados (anclas propias = "propias")."""
    out = {}
    for k, d in idx.items():
        if k[0] != "plantilla":
            continue
        _, pl, slug, rej, datos, av, s, r, peso, pend, quita, anclas, ruido = k
        if av is not None or pend is not None or rej not in (ATMO_PLANTILLAS, "btsettl"):
            continue
        fam = "atmo" if rej == ATMO_PLANTILLAS else "btsettl"
        a = "propias" if anclas == propias[slug] else anclas
        out[(fam, datos, s, peso, quita, a, ruido, slug, pl)] = (
            d["posterior"]["teff"]["p50"], d["posterior"]["logg"]["p50"], d["verdad"]["spt"])
    return out


def resumen(ds):
    ds = [x for x in ds if x is not None]
    if not ds:
        return dict(n=0)
    return dict(n=len(ds), mediana=med(ds), mediana_abs=med([abs(x) for x in ds]), max_abs=max(abs(x) for x in ds),
                n_sobre_umbral=sum(abs(x) > UMBRAL for x in ds))


def cifras_plantillas(P):
    pls = sorted({k[-1] for k in P}, key=lambda p: (ORDEN_SPT.index(next(v[2] for k, v in P.items() if k[-1] == p)), p))

    def t(fam, datos, s, c, p, peso=False, quita=(), anclas="propias", r=None, que=0):
        x = P.get((fam, datos, s, peso, tuple(sorted(quita)), anclas, r or c, c, p))
        return None if x is None else x[que]

    def dif(a, b):
        return None if a is None or b is None else a - b

    R = {}
    # cambiar el ruido o los píxeles (MUSE + IR, ATMO)
    for s in (0, 1):
        ruido, cob, tot, filas = [], [], [], []
        for p in pls:
            t11, t12, t21, t22 = (t("atmo", "todo", s, c, p, r=r) for c, r in ((O1, O1), (O1, O2), (O2, O1),
                                                                                (O2, O2)))
            tot.append(dif(t22, t11))
            if None not in (t11, t12, t21, t22):
                ruido.append(0.5 * ((t12 - t11) + (t22 - t21)))
                cob.append(0.5 * ((t21 - t11) + (t22 - t12)))
                filas.append(dict(plantilla=p, total=t22 - t11, ruido=ruido[-1], pixeles=cob[-1]))
        R[f"atmo_todo_s{s}"] = dict(total=resumen(tot), ruido=resumen(ruido), pixeles=resumen(cob), por_plantilla=filas)
    # solo IR: ATMO s0, s1; BT-Settl s0
    for fam, s in (("atmo", 0), ("atmo", 1), ("btsettl", 0)):
        R[f"{fam}_ir_s{s}"] = dict(total=resumen([dif(t(fam, "ir", s, O2, p), t(fam, "ir", s, O1, p)) for p in pls]))
    # las anclas (ATMO, solo IR): anclar J y H además de K, con los dos juegos de espectros
    for s in (0, 1):
        da = []
        for p in pls:
            x = [t("atmo", "ir", s, c, p, anclas="propias" if a == PROPIAS[c] else a) for c in (O1, O2)
                 for a in ("K", "JHK")]
            da.append(None if None in x else 0.5 * ((x[1] - x[0]) + (x[3] - x[2])))
        R[f"atmo_ir_s{s}"]["anclas"] = resumen(da)
    # un espectro por banda, los mismos en los dos disfraces
    for fam, datos, s in (("atmo", "ir", 0), ("atmo", "ir", 1), ("btsettl", "ir", 0), ("atmo", "todo", 0)):
        R.setdefault(f"{fam}_{datos}_s{s}", {})["mismos"] = resumen(
            [dif(t(fam, datos, s, O2, p, quita=MISMOS[O2]), t(fam, datos, s, O1, p, quita=MISMOS[O1]))
             for p in pls])
    R["entre_semillas_mismos_atmo_ir"] = {
        c: med([abs(t("atmo", "ir", 0, c, p, quita=MISMOS[c]) - t("atmo", "ir", 1, c, p, quita=MISMOS[c]))
                for p in pls if None not in (t("atmo", "ir", 0, c, p, quita=MISMOS[c]),
                                              t("atmo", "ir", 1, c, p, quita=MISMOS[c]))])
        for c in SLUGS}
    # con el peso
    for fam, datos, s in (("atmo", "ir", 0), ("atmo", "ir", 1), ("btsettl", "ir", 0), ("atmo", "todo", 0),
                          ("atmo", "todo", 1)):
        R.setdefault(f"{fam}_{datos}_s{s}", {})["peso"] = resumen(
            [dif(t(fam, datos, s, O2, p, peso=True), t(fam, datos, s, O1, p, peso=True)) for p in pls])
    # el peso frente a un espectro por banda (ATMO, solo IR): |T(c, peso) − T(c − quitados)|, mediana
    R["peso_frente_a_mismos_atmo_ir"] = rango([
        med([abs(t("atmo", "ir", s, c, p, peso=True) - t("atmo", "ir", s, c, p, quita=MISMOS[c])) for p in pls
             if None not in (t("atmo", "ir", s, c, p, peso=True), t("atmo", "ir", s, c, p, quita=MISMOS[c]))])
        for s in (0, 1) for c in SLUGS])
    # entre semillas con el peso: solo IR y MUSE + IR
    for datos in ("ir", "todo"):
        e = {}
        for c in SLUGS:
            v = [(abs(t("atmo", datos, 0, c, p, peso=True) - t("atmo", datos, 1, c, p, peso=True)), p) for p in pls
                 if None not in (t("atmo", datos, 0, c, p, peso=True), t("atmo", datos, 1, c, p, peso=True))]
            e[c] = dict(mediana=med([x for x, _ in v]), max=max(v)[0], plantilla_max=max(v)[1])
        R[f"entre_semillas_peso_atmo_{datos}"] = e
    pmax = R["entre_semillas_peso_atmo_todo"]
    c = max(pmax, key=lambda k: pmax[k]["max"])
    p = pmax[c]["plantilla_max"]
    R["bimodal"] = dict(disfraz=c, plantilla=p, spt=P[("atmo", "todo", 0, True, (), "propias", c, c, p)][2],
                        t=[t("atmo", "todo", s, c, p, peso=True) for s in (0, 1)],
                        logg=[t("atmo", "todo", s, c, p, peso=True, que=1) for s in (0, 1)])
    # sin el peso, cuántas pasan de 150 K como mucho; con el peso, ninguna
    R["n_sobre_umbral_sin_peso_max"] = max(R[k]["total"]["n_sobre_umbral"] for k in R
                                           if isinstance(R[k], dict) and "total" in R[k] and R[k]["total"]["n"])
    R["n_sobre_umbral_con_peso_max"] = max(R[k]["peso"]["n_sobre_umbral"] for k in R
                                           if isinstance(R[k], dict) and "peso" in R[k] and R[k]["peso"]["n"])
    R["plantillas"] = pls
    return R


# ================================================================== todo
def calcula(K, V, root, entradas=None):
    """Todo, en un dict (lo que va al json). `entradas`: {slug: `teff_bayes.entrada_margen`}; por defecto, de los
    runs del paper."""
    fija_objetos()
    propias = dict(PROPIAS)
    idx = indice(K, V, propias)
    fija_rejillas(idx)
    if entradas is None:
        entradas = {s: tb.entrada_margen(s, root=Path(root)) for s in SLUGS}
    D = Datos(idx, entradas, propias)
    d3 = tabla_d3(D)
    d4 = {slug: cuatro_formas(D, slug, D4[slug]) for slug in SLUGS}
    uv = una_vez(D)
    res = dict(tabla_d3=d3, tabla_d4=d4, datos=cifras_datos(D, d3, d4), una_vez=cifras_una_vez(D, uv),
               plantillas=cifras_plantillas(plantillas(idx, propias)), filas_una_vez=uv,
               n_cadenas=dict(datos=sum(k[0] == "datos" for k in idx), plantillas=sum(k[0] == "plantilla" for k in idx)))
    res["filas_d3"], res["filas_d4"] = filas_d3(d3), filas_d4(d4, res["datos"])
    res["cifras_texto"] = cifras_texto(res)
    return res


# ================================================================== las tablas, al redondeo impreso
def filas_d3(d3):
    """Las filas de la Tabla D.3: [A_V o None, T×4 (enteros), Δ×4 (dos decimales)], en el orden del paper; y la nota:
    las A_V con el prior (dos decimales), por objeto y rejilla."""
    filas, nota = [], []
    for slug, fam in product(SLUGS, D3):
        for av in AVS:
            c = d3[slug][fam][av]
            filas.append([av] + [round(c[f]["t"]) for f in FORMAS] + [round(c[f]["delta"], 2) for f in FORMAS])
        nota.append([round(d3[slug][fam][None][f]["av"], 2) for f in FORMAS])
    return dict(filas=filas, nota=nota)


def filas_d4(d4, R):
    """Las filas de la Tabla D.4: [máx |ΔT| (entero), máx |ΔΔ|, Δ con el prior ×4 (dos decimales), A_V del cruce ×4
    (un decimal; (x) si es extrapolado; None en 42B b)]."""
    out = []
    for slug in SLUGS:
        for g in D4[slug]:
            m = R[f"{slug}|{g}"]
            cr = [None] * 4 if slug == O2 else [
                (round(x["cruce"], 1), not x["dentro"]) for x in (m["cruces"][f] for f in FORMAS)]
            out.append([round(m["max_dT"]), round(m["max_dD"], 2)] + [round(x, 2) for x in m["delta_prior"]] + cr)
    return out


def _num(celda):
    """Los números de una celda LaTeX (sin los de `\\tablefootmark`); el «−» de LaTeX es «-»."""
    celda = re.sub(r"\\tablefootmark\{[^}]*\}", "", celda)
    return [float(x) for x in re.findall(r"[-+]?\d+(?:\.\d+)?", celda)]


def _filas_tex(tex, etiqueta):
    """Las filas con datos (las que llevan «&») del cuerpo de una tabla, entre su `\\label` y su `\\end{tabular}`,
    sin la cabecera (lo que va antes del primer `\\hline` tras el `\\hline\\hline`)."""
    i = tex.find("\\label{" + etiqueta + "}")
    if i < 0:
        return None
    cuerpo = tex[i:tex.find("\\end{tabular}", i)]
    cuerpo = cuerpo[cuerpo.find("\\hline\\hline"):]
    cuerpo = cuerpo[cuerpo.find("\\hline", len("\\hline\\hline")) + len("\\hline"):]
    return [ln.strip() for ln in cuerpo.splitlines() if "&" in ln]


def compara_tex(tex, res):
    """dict(tablas=[diferencias entre las Tablas D.3 y D.4 del tex y las calculadas], texto=[cifras de
    `cifras_texto` que no aparecen en el D.12 tal como se calculan])."""
    dif, texto = [], []
    # Tabla D.3
    filas = _filas_tex(tex, "tab:bands")
    if filas is None:
        dif.append("no hay \\label{tab:bands} en el tex")
    else:
        nuestras = res["filas_d3"]["filas"]
        if len(filas) != len(nuestras):
            dif.append(f"Tabla D.3: el tex tiene {len(filas)} filas y aquí {len(nuestras)}")
        for k, (ln, f) in enumerate(zip(filas, nuestras)):
            celdas = [c.strip() for c in ln.rstrip("\\ ").split("&")]
            av = _num(celdas[1])
            got = (av[0] if av else None, *[_num(c)[0] for c in celdas[2:10]])
            if (got[0] != f[0]) or list(got[1:5]) != f[1:5] or [round(x, 2) for x in got[5:9]] != f[5:9]:
                dif.append(f"Tabla D.3, fila {k + 1}: tex {got}, calculado {tuple(f)}")
        # la nota: las A_V con el prior, en el orden del texto (por objeto y rejilla, cuatro cada vez)
        i = tex.find("\\label{tab:bands}")
        nota = tex[tex.find("With the prior, the fits give", i):tex.find("in the four forms", i)]
        avs = [x for x in _num(nota) if 0 < x < 10 and x not in (12.0, 42.0)]
        nuestras_av = [x for fila in res["filas_d3"]["nota"] for x in fila]
        if [round(x, 2) for x in avs] != nuestras_av:
            dif.append(f"Tabla D.3, nota (A_V con el prior): tex {avs}, calculado {nuestras_av}")
    # Tabla D.4
    filas = _filas_tex(tex, "tab:bands_grids")
    if filas is None:
        dif.append("no hay \\label{tab:bands_grids} en el tex")
    else:
        nuestras = res["filas_d4"]
        if len(filas) != len(nuestras):
            dif.append(f"Tabla D.4: el tex tiene {len(filas)} filas y aquí {len(nuestras)}")
        for k, (ln, f) in enumerate(zip(filas, nuestras)):
            celdas = [c.strip() for c in ln.rstrip("\\ ").split("&")]
            got = [_num(c)[0] for c in celdas[1:7]]
            cr = [None if c == "--" else (_num(c)[0], c.startswith("(")) for c in celdas[7:11]]
            if [round(got[0])] + [round(x, 2) for x in got[1:]] != f[:6] or cr != f[6:]:
                dif.append(f"Tabla D.4, fila {k + 1}: tex {got + cr}, calculado {f}")
    # las cifras del texto
    i = tex.find("\\label{tab:bands_grids}")
    sec = tex[tex.find("\\label{app:checks:bands}"):tex.find("\\end{table*}", i) if i >= 0 else len(tex)]
    sec = re.sub(r"\s+", " ", sec)
    for c in res["cifras_texto"]:
        if re.sub(r"\s+", " ", c["tex"]) not in sec:
            texto.append(f"texto del D.12: no aparece «{c['tex']}» ({c['que']}; calculado {c['valor']})")
    return dict(tablas=dif, texto=texto)


# ================================================================== las cifras del texto, como se imprimen
def _f(x, nd=0, signo=False):
    return f"{x:+.{nd}f}" if signo else f"{x:.{nd}f}"


def _r(a, b, nd=0, signo=False, sep="--"):
    return f"\\({_f(a, nd, signo)}\\){sep}\\({_f(b, nd, signo)}\\)"


def _rr(x, nd=0, signo=False):
    return _r(x[0], x[1], nd, signo)


def cifras_texto(res):
    """[dict(que, valor, tex)]: cada cifra del D.12 que sale de las cadenas, con el fragmento LaTeX con el que la
    imprime el paper. `compara_tex` busca cada fragmento en el texto del D.12."""
    D, U, P = res["datos"], res["una_vez"], res["plantillas"]
    N1, N2 = tc.NOMBRES[O1], tc.NOMBRES[O2]
    d3 = res["tabla_d3"]
    out = []

    def c(que, valor, tex):
        out.append(dict(que=que, valor=valor, tex=tex))

    # --- plantillas
    a0, a1 = P["atmo_todo_s0"], P["atmo_todo_s1"]
    c("plantillas: cambiar el ruido (mediana |Δ_pl|, ATMO, MUSE + IR, s0 y s1)",
      [a0["ruido"]["mediana_abs"], a1["ruido"]["mediana_abs"]],
      _r(*sorted([a0["ruido"]["mediana_abs"], a1["ruido"]["mediana_abs"]])) + "~K (median of the twelve templates, two seeds)")
    c("plantillas: cambiar los píxeles (mediana Δ_pl)", [a0["pixeles"]["mediana"], a1["pixeles"]["mediana"]],
      "it changes by " + _r(*sorted([a0["pixeles"]["mediana"], a1["pixeles"]["mediana"]])) + "~K, almost all")
    i0, i1, b0 = P["atmo_ir_s0"]["total"]["mediana"], P["atmo_ir_s1"]["total"]["mediana"], P["btsettl_ir_s0"]["total"]["mediana"]
    c("plantillas: solo el IR (mediana Δ_pl; ATMO s0, s1; BT-Settl s0)", [i0, i1, b0],
      f"difference is {_r(*sorted([i0, i1]))}~K with ATMO~2020 and \\({_f(b0)}\\)~K with BT-Settl")
    an = [P["atmo_ir_s0"]["anclas"]["mediana_abs"], P["atmo_ir_s1"]["anclas"]["mediana_abs"]]
    c("plantillas: las anclas (mediana |Δ|, ATMO solo IR)", an, "photometry changes it by " + _r(*sorted(an)) + "~K")
    m0, m1, mt = (P["atmo_ir_s0"]["mismos"]["mediana"], P["atmo_ir_s1"]["mismos"]["mediana"],
                  P["atmo_todo_s0"]["mismos"]["mediana"])
    c("plantillas: un espectro por banda (mediana Δ_pl; ATMO solo IR s0, s1; MUSE + IR s0)", [i0, i1, m0, m1, mt],
      f"from \\({_f(i0, 0, True)}\\) and \\({_f(i1, 0, True)}\\)~K to \\({_f(m0, 0, True)}\\) and \\({_f(m1, 0, True)}\\)~K"
      f" with the near-infrared spectra alone (two seeds), and from \\({_f(a0['total']['mediana'], 0, True)}\\) to "
      f"\\({_f(mt, 0, True)}\\)~K with all the spectra (one seed)")
    es = P["entre_semillas_mismos_atmo_ir"]
    c("plantillas: entre semillas con un espectro por banda (mediana, ATMO solo IR)", es,
      "between seeds of these fits, " + _r(10 * round(min(es.values()) / 10), 10 * round(max(es.values()) / 10)) + "~K")
    p0, p1 = P["atmo_ir_s0"]["peso"]["mediana"], P["atmo_ir_s1"]["peso"]["mediana"]
    pt0, pt1 = P["atmo_todo_s0"]["peso"]["mediana"], P["atmo_todo_s1"]["peso"]["mediana"]
    pb = P["btsettl_ir_s0"]["peso"]["mediana"]
    c("plantillas: con el peso (mediana Δ_pl; ATMO solo IR s0, s1; MUSE + IR s0, s1; BT-Settl solo IR)",
      [p0, p1, pt0, pt1, pb],
      f"\\({_f(p0, 0, True)}\\) and \\({_f(p1, 0, True)}\\)~K with ATMO~2020 and the near-infrared spectra alone, "
      f"\\({_f(pt0, 0, True)}\\)~K with all the spectra (in each of two seeds), and \\({_f(pb, 0, True)}\\)~K with BT-Settl"
      if round(pt0) == round(pt1) else "(las dos semillas con peso no dan la misma mediana)")
    c("plantillas: cuántas pasan de 150 K (con el peso, como mucho; sin él, como mucho)",
      [P["n_sobre_umbral_con_peso_max"], P["n_sobre_umbral_sin_peso_max"]],
      ("No template differs by more than \\(150\\)~K, against up to "
       + tc.PALABRAS.get(P["n_sobre_umbral_sin_peso_max"], str(P["n_sobre_umbral_sin_peso_max"]))
       + " of the twelve without the weight") if P["n_sobre_umbral_con_peso_max"] == 0 else "(alguna pasa de 150 K)")
    e = P["entre_semillas_peso_atmo_todo"]
    c("plantillas: entre semillas con el peso, MUSE + IR (mediana por disfraz)", [e[O1]["mediana"], e[O2]["mediana"]],
      "temperature of a template changes by " + _r(*sorted([e[O1]["mediana"], e[O2]["mediana"]])) + "~K (median)")
    b = P["bimodal"]
    c("plantillas: la de dos soluciones (tipo, log g y T de las dos semillas)", b,
      f"One template, of type {b['spt']}, has two solutions" + " with the noise and the spectra of "
      + tc.NOMBRES[b["disfraz"]] + f", at \\(\\log g = {_f(min(b['logg']), 1)}\\) and "
      f"\\({_f(max(b['logg']), 1)}\\), and the two seeds find different ones (\\({_f(b['t'][0])}\\) and \\({_f(b['t'][1])}\\)~K)")
    pm = P["peso_frente_a_mismos_atmo_ir"]
    ei = P["entre_semillas_peso_atmo_ir"]
    c("plantillas: el peso frente a un espectro por banda (mediana) y entre semillas con el peso (ATMO solo IR)",
      [pm, ei[O1]["mediana"], ei[O2]["mediana"]],
      "differ by " + _rr(pm) + "~K (median), as much as two seeds of the weighted fit (" +
      _r(*sorted([ei[O1]["mediana"], ei[O2]["mediana"]])) + "~K)")
    # --- los datos con un espectro por banda
    e42, e12 = U[f"elegir_{O2}"], U[f"elegir_{O1}"]
    c("datos: lo que cambia la elección del espectro (BT-Settl, MUSE + IR; ancho T y Δ)", [e42, e12],
      f"change the temperature of {N2} by {_rr(e42['ancho_t'])}~K and \\(\\Delta\\) by {_rr(e42['ancho_d'], 2)}~dex, "
      f"and those of {N1} by {_rr(e12['ancho_t'])}~K and {_rr(e12['ancho_d'], 2)}~dex")
    k = U["por_banda"][f"{O2}|K"]
    c("datos: las dos K de O2 (máx |dif.| con el mismo espectro en la otra banda)", k,
      f"agree within about \\({_f(10 * round(k['t'] / 10))}\\)~K and \\({_f(k['d'], 2)}\\)~dex")
    # --- la inclinación
    dz = U["dlnz_incl"]
    c("Δln Z de la inclinación sin el peso (todos los ajustes con todos los espectros)", dz["sin_peso"],
      f"\\(\\Delta\\ln Z = {_f(dz['sin_peso'][0], 0, True)}\\) to \\({_f(dz['sin_peso'][1], 0, True)}\\)")
    inc = U["inclinaciones_bt_drift_atmo"]
    h, kk, j = inc["OSIRIS_H-NIFS_H"], inc["SINFONI_K-OSIRIS_K"], inc["OSIRIS_J-NIFS_J"]
    c("inclinación ajustada de las dos H (BT-Settl, DRIFT, ATMO)", h,
      f"is \\({_f(h[1], 1)}\\) to \\({_f(h[0], 1)}\\%\\) per \\(0.1~\\mu\\)m, close to the measured \\({_f(MEDIDO[('OSIRIS_H', 'NIFS_H')], 1)}\\%\\)")
    c("inclinación ajustada de las dos K y de las dos J (BT-Settl, DRIFT, ATMO)", [kk, j],
      f"(\\({_f(kk[1], 1)}\\) to \\({_f(kk[0], 1)}\\) against \\({_f(MEDIDO[('SINFONI_K', 'OSIRIS_K')], 1)}\\%\\)) and of the "
      f"two \\(J\\) spectra of {N1} (\\({_f(j[0], 0, True)}\\) to \\({_f(j[1], 0, True)}\\) against "
      f"\\({_f(MEDIDO[('OSIRIS_J', 'NIFS_J')], 1, True)}\\%\\))")
    c("con la inclinación, ancho de la elección de espectro (mediana de cada grupo)", U["ancho_con_inclinacion_mediana_grupo"],
      "differ by " + _rr(U["ancho_con_inclinacion_mediana_grupo"]) + "~K (median of each group of fits)")
    a01 = U["ancho_01"]
    c("±0.1 frente a ±0.3: 42B b (todos los datos) y 12 b (MUSE + IR)", [a01[f"{O2}|todos"], a01[f"{O1}|todo"]],
      f"change by up to \\({_f(10 * round(a01[O2 + '|todos'][1] / 10))}\\)~K for {N2} and by "
      f"{_rr(a01[O1 + '|todo'])}~K for {N1} with all the spectra")
    s12 = U["12b_solo_ir_incl"]
    c("12 b, solo IR con inclinación: T", s12["t"],
      "temperature rises to " + _r(10 * round(s12["t"][0] / 10), 10 * round(s12["t"][1] / 10)) + "~K")
    pe, pc = U["peso_sin"], U["peso_con"]
    c("el peso dentro del rango de un espectro por banda (sin inclinación) y su distancia al centro",
      [pe["dentro"], pe["menos_centro_mediana_grupo"]],
      f"in \\({pe['dentro'][0]}\\) of \\({pe['dentro'][1]}\\) cases, " + _rr(pe["menos_centro_mediana_grupo"]) + "~K")
    c("con la inclinación, la distancia al centro", pc["menos_centro_mediana_grupo"],
      "it lies " + _rr(pc["menos_centro_mediana_grupo"]) + "~K from the centre")
    c("Δln Z de la inclinación con el peso, y las excepciones", dz,
      f"(\\(\\Delta\\ln Z = {_f(dz['con_peso'][0], 0, True)}\\) to \\({_f(dz['con_peso'][1], 0, True)}\\)), except for "
      f"{N2} with ATMO~2020 and the near-infrared spectra alone (\\({_f(dz['con_peso_42b_atmo_ir'][1], 1)}\\) to "
      f"\\({_f(dz['con_peso_42b_atmo_ir'][0], 1)}\\), against \\({_f(dz['sin_peso_42b_atmo_ir'][0], 0, True)}\\) to "
      f"\\({_f(dz['sin_peso_42b_atmo_ir'][1], 0, True)}\\) without the weight) and with DRIFT-PHOENIX at "
      f"\\(A_V = 3.0\\)~mag (\\({_f(dz['con_peso_drift_42b_30'], 1)}\\))")
    # --- BT-Settl, DRIFT, ATMO
    bt, dr = D[f"{O1}|btsettl"], D[f"{O1}|drift"]
    c("BT-Settl: lo más que mueven T las cuatro formas (12 b, 42B b)", [bt["max_dT"], D[f"{O2}|btsettl"]["max_dT"]],
      f"temperature by up to \\({_f(10 * round(bt['max_dT'] / 10))}\\)~K for {N1} and "
      f"\\({_f(10 * round(D[O2 + '|btsettl']['max_dT'] / 10))}\\)~K for {N2}")
    dpa = bt["delta_por_av"]
    c("BT-Settl, 12 b: Δ a 1.8, 3.0 y con el prior en las cuatro formas", dpa,
      f"is between \\({_f(dpa['1.8'][0], 2, True)}\\) and \\({_f(dpa['1.8'][1], 2, True)}\\)~dex at the adopted "
      f"\\(A_V = 1.8\\)~mag, between \\({_f(dpa['3.0'][0], 2, True)}\\) and \\({_f(dpa['3.0'][1], 2, True)}\\)~dex at "
      f"\\(3.0\\)~mag, and between \\({_f(dpa['None'][0], 2, True)}\\) and \\({_f(dpa['None'][1], 2, True)}\\)~dex with the prior")
    bp = bt["baja_con_peso"]
    avb = d3[O1]["btsettl"][None]
    c("BT-Settl, 12 b: lo que baja Δ el peso, y la A_V con el prior", [bp, avb["peso"]["av"], avb["ref"]["av"]],
      f"lowers \\(\\Delta\\) by {_r(min(bp), max(bp), 2)}~dex, because counting the \\(J\\) band twice raised the extinction "
      f"of the fit with the prior, from \\({_f(avb['peso']['av'], 2)}\\) to \\({_f(avb['ref']['av'], 2)}\\)~mag")
    crb = bt["cruces"]
    c("BT-Settl, 12 b: A_V del cruce en las cuatro formas", crb,
      f"sign at \\({_f(crb['ref']['cruce'], 1)}\\)~mag in the form of Table~\\ref{{tab:teff_chains}}, at "
      f"\\({_f(crb['peso']['cruce'], 1)}\\)~mag with the weight, at \\({_f(crb['incl']['cruce'], 1)}\\)~mag with the free "
      f"slopes, and at \\({_f(crb['las dos']['cruce'], 1)}\\)~mag with both")
    c("DRIFT, 12 b: lo más que mueven T las cuatro formas", dr["max_dT"],
      f"change the temperature of {N1} by up to \\({_f(dr['max_dT'])}\\)~K")
    dpd = dr["delta_por_av"]
    c("DRIFT, 12 b: Δ a 1.8, 3.0 y con el prior", dpd,
      f"is between \\({_f(dpd['1.8'][0], 2, True)}\\) and \\({_f(dpd['1.8'][1], 2, True)}\\)~dex at \\(1.8\\)~mag, "
      f"between \\({_f(dpd['3.0'][0], 2, True)}\\) and \\({_f(dpd['3.0'][1], 2, True)}\\)~dex at "
      f"\\(3.0\\)~mag, and between \\({_f(dpd['None'][0], 2, True)}\\) and \\({_f(dpd['None'][1], 2, True)}\\)~dex with the prior")
    crd = dr["cruces"]
    c("DRIFT, 12 b: A_V del cruce", crd,
      f"sign at \\({_f(crd['ref']['cruce'], 1)}\\)~mag in the form of Table~\\ref{{tab:teff_chains}}, at "
      f"\\({_f(crd['peso']['cruce'], 1)}\\)~mag with the weight, at \\({_f(crd['incl']['cruce'], 1)}\\)~mag with the free "
      f"slopes, and at \\({_f(crd['las dos']['cruce'], 1)}\\)~mag with both")
    cz = D["cruces_12b"]
    c("BT-Settl y DRIFT, 12 b: A_V con el prior y lo menos que queda por encima del cruce",
      [cz["av_prior_bt_drift"], cz["sobre_cruce_bt_drift"][0]],
      f"give \\(A_V = {_f(cz['av_prior_bt_drift'][0], 1)}\\)--\\({_f(cz['av_prior_bt_drift'][1], 1)}\\)~mag, above the "
      f"extinction at which \\(\\Delta\\) changes sign, but only by \\({_f(cz['sobre_cruce_bt_drift'][0], 2)}\\)~mag")
    at = D[f"{O1}|atmo"]["baja_con_peso"]
    c("ATMO, 12 b: lo que baja Δ el peso", at, f"lowers \\(\\Delta\\) of {N1} by {_r(min(at), max(at), 2)}~dex")
    uj = U["atmo_12b_una_j_prior"]
    c("ATMO, 12 b: Δ con el prior, todos los espectros y con una sola J", [d3[O1]["atmo"][None]["ref"]["delta"], uj],
      f"\\({_f(d3[O1]['atmo'][None]['ref']['delta'], 2, True)}\\)~dex, lies above the values with either \\(J\\) "
      f"spectrum alone (\\({_f(uj[0], 2, True)}\\) to \\({_f(uj[1], 2, True)}\\)~dex)")
    b42 = D[f"{O2}|btsettl"]["delta_por_av"]
    c("BT-Settl, 42B b: Δ de las cuatro formas", b42,
      f"give \\(\\Delta\\) within \\({_f(b42['1.8'][1] - b42['1.8'][0], 2)}\\)~dex at \\(1.8\\)~mag and within "
      f"\\({_f(b42['None'][1] - b42['None'][0], 2)}\\)~dex with the prior, and between \\({_f(b42['3.0'][0], 2, True)}\\) "
      f"and \\({_f(b42['3.0'][1], 2, True)}\\)~dex at \\(3.0\\)~mag")
    sl = {fam: rango([d3[O2][fam][None][f]["delta"] for f in ("incl", "las dos")]) for fam in D3}
    c("42B b con el prior y la inclinación: BT-Settl y ATMO", sl,
      f"\\(\\Delta\\) is \\({_f(sl['btsettl'][1], 2, True)}\\) to \\({_f(sl['btsettl'][0], 2, True)}\\)~dex with BT-Settl and "
      f"\\({_f(sl['atmo'][0], 2, True)}\\) to \\({_f(sl['atmo'][1], 2, True)}\\)~dex with ATMO~2020")
    sd = rango([d3[O2]["drift"][None][f]["delta"] for f in ("ref", "peso")])
    c("DRIFT, 42B b con el prior: sin y con inclinación", [sd, sl["drift"]],
      f"is \\({_f(sd[0], 2, True)}\\) to \\({_f(sd[1], 2, True)}\\)~dex without the free slopes and "
      f"\\({_f(sl['drift'][0], 2, True)}\\) to \\({_f(sl['drift'][1], 2, True)}\\)~dex with them")
    x = d3[O2]["drift"][3.0]
    c("DRIFT, 42B b a 3.0 con la inclinación sola: la otra solución", [x["incl"]["t"], x["incl"]["logg"], x["incl"]["t"] - x["ref"]["t"]],
      f"another solution, at \\({_f(x['incl']['t'])}\\)~K and \\(\\log g = {_f(x['incl']['logg'], 1)}\\), "
      f"\\({_f(10 * round((x['incl']['t'] - x['ref']['t']) / 10))}\\)~K above")
    # --- las demás rejillas, 12 b
    n12 = D["nuevas_12b"]
    c("otras rejillas, 12 b: lo más que se mueven T y Δ; Δ con el prior", n12,
      f"change the temperature by up to \\({_f(n12['max_dT'])}\\)~K and \\(\\Delta\\) by up to \\({_f(n12['max_dD'], 2)}\\)~dex. "
      f"With the prior, \\(\\Delta\\) is between \\({_f(n12['delta_prior'][0], 2, True)}\\) and "
      f"\\({_f(n12['delta_prior'][1], 2, True)}\\)~dex")
    pf = cz["por_forma"]
    c("otras rejillas, 12 b: A_V del cruce por forma", pf,
      f"changes sign at {_rr(pf['ref']['nuevas'], 1)}~mag in the form of Table~\\ref{{tab:teff_grids}}, at "
      f"{_rr(pf['peso']['nuevas'], 1)}~mag with the weight, at {_rr(pf['incl']['nuevas'], 1)}~mag with the free slopes, "
      f"and at {_rr(pf['las dos']['nuevas'], 1)}~mag with both")
    ext = {(x["rejilla"], x["forma"]): x for x in cz["extrapolados"]}
    sp, btd = ext.get(("sphinx", "incl")), ext.get(("bt_dusty", "incl"))
    if sp and btd:
        c("otras rejillas, 12 b: los extrapolados (SPHINX con inclinación a 1.8; BT-Dusty con inclinación a 3.0)",
          [sp["delta18"], btd["delta30"]],
          f"with SPHINX and the free slopes, \\(\\Delta\\) is already \\({_f(sp['delta18'], 2, True)}\\)~dex at \\(1.8\\)~mag, "
          f"and with BT-Dusty and the free slopes alone it is still \\({_f(btd['delta30'], 2, True)}\\)~dex at \\(3.0\\)~mag")
    c("otras rejillas, 12 b: A_V con el prior y cuánto por encima del cruce", [cz["av_prior_nuevas"], cz["sobre_cruce_nuevas"]],
      f"give \\(A_V = {_f(cz['av_prior_nuevas'][0], 1)}\\)--\\({_f(cz['av_prior_nuevas'][1], 1)}\\)~mag, "
      f"{_rr(cz['sobre_cruce_nuevas'], 1)}~mag above")
    # --- las demás rejillas, 42B b
    n42 = D["nuevas_42b"]
    c("otras rejillas, 42B b: AMES con inclinación (T, log g, cuánto sube)", n42,
      f"at {_rr(n42['ames_incl_t'])}~K and \\(\\log g = {_f(n42['ames_incl_logg'][0], 1)}\\)--\\({_f(n42['ames_incl_logg'][1], 1)}\\)")
    c("otras rejillas, 42B b: AMES con inclinación, cuánto sube a 1.8 y con el prior", n42["ames_incl_sube"],
      f"this solution is {_rr(n42['ames_incl_sube'])}~K above")
    c("otras rejillas, 42B b: AMES-Dusty con el peso solo a 3.0", n42["ames_dusty_peso_30"],
      f"AMES-Dusty also reaches it at \\(3.0\\)~mag (\\({_f(n42['ames_dusty_peso_30']['t'])}\\)~K)")
    ad = n42["ames_incl_delta"]
    c("otras rejillas, 42B b: Δ de AMES con inclinación a 1.8 y con el prior", ad,
      f"\\(\\Delta\\) is \\({_f(ad['1.8'][1], 2, True)}\\) to \\({_f(ad['1.8'][0], 2, True)}\\)~dex at \\(1.8\\)~mag and "
      f"\\({_f(ad['None'][1], 2, True)}\\) to \\({_f(ad['None'][0], 2, True)}\\)~dex with the prior")
    f1 = n42["diamondback_f1_lejos"]
    c("otras rejillas, 42B b: Diamondback (lo más que se mueve sin f1; f1 lejos)", [n42["diamondback_max_dT_sin_f1"], f1],
      f"temperature by up to \\({_f(n42['diamondback_max_dT_sin_f1'])}\\)~K, except for \\(f_{{\\rm sed}} = 1\\) with the weight "
      f"and the slopes at \\(1.8\\) and \\(3.0\\)~mag, which gives {_rr(rango([x['t'] for x in f1]))}~K, "
      f"{_rr(rango([-x['dt'] for x in f1]))}~K below")
    lb = n42["diamondback_logg_borde"]
    c("otras rejillas, 42B b: Diamondback con log g en el borde", lb,
      f"In \\({lb[0]}\\) of the \\({lb[1]}\\) fits with Diamondback, \\(\\log g\\) reaches the edge")
    c("otras rejillas, 42B b: Δ con el prior (Diamondback, SPHINX) y lo más que se mueve SPHINX",
      [n42["diamondback_delta_prior"], n42["sphinx_delta_prior"], n42["sphinx_max_dT"]],
      f"\\(\\Delta\\) is between \\({_f(abs(n42['diamondback_delta_prior'][0]), 2)}\\) and "
      f"\\({_f(n42['diamondback_delta_prior'][1], 2, True)}\\)~dex with Diamondback, and between "
      f"\\({_f(n42['sphinx_delta_prior'][0], 2, True)}\\) and \\({_f(n42['sphinx_delta_prior'][1], 2, True)}\\)~dex with "
      f"SPHINX, whose temperature changes by up to \\({_f(n42['sphinx_max_dT'])}\\)~K")
    c("42B b: cuántas no cambian de signo entre 1.8 y 3.0 (otras rejillas)", n42["sin_cambio_de_signo"],
      f"in \\({n42['sin_cambio_de_signo'][0]}\\) of the \\({n42['sin_cambio_de_signo'][1]}\\) combinations")
    # --- el cierre
    ci = res["datos"]["cierre"]
    c("cierre: lo más que cambia Δ de 12 b (BT-Settl, DRIFT, las demás)", ci,
      f"by up to \\({_f(ci['btsettl'], 2)}\\)~dex with BT-Settl, \\({_f(ci['drift'], 2)}\\)~dex with DRIFT-PHOENIX and "
      f"\\({_f(ci['otras'], 2)}\\)~dex with the other grids")
    c("cierre: el rango del cruce con las seis rejillas", cz["todos"],
      f"the extinction at which the line reaches the noise level is {_rr(cz['todos'], 1)}~mag")
    c("cierre: cuántos cruzan entre 1.8 y 3.0", cz["dentro"],
      f"In \\({cz['dentro'][0]}\\) of these \\({cz['dentro'][1]}\\) fits")
    lejos = max(max(abs(x["delta18"]) if x["cruce"] < 1.8 else abs(x["delta30"]) for x in cz["extrapolados"]
                    if x["rejilla"] not in ("btsettl", "drift")), abs(d3[O1]["btsettl"][1.8]["las dos"]["delta"]))
    c("cierre: lo lejos de cero que queda Δ en los extrapolados", lejos, f"within \\({_f(lejos, 2)}\\)~dex of zero")
    c("cierre: Δ de 42B b en todos estos ajustes", n42["delta_todas"],
      f"(\\({_f(n42['delta_todas'][0], 2, True)}\\) to \\({_f(n42['delta_todas'][1], 2, True)}\\)~dex in all these fits)")
    return out


# ================================================================== informe
def informe_md(res, dif=None):
    L = ["# Dos espectros en la misma banda (D.12): las cifras desde las cadenas", "",
         f"Cadenas leídas: {res['n_cadenas']['datos']} de los datos y {res['n_cadenas']['plantillas']} de plantillas.", ""]
    L += ["## Tabla D.3", "", "| objeto | rejilla | A_V | " + " | ".join(f"T {f}" for f in FORMAS) + " | "
          + " | ".join(f"Δ {f}" for f in FORMAS) + " |", "|" + "---|" * 11]
    for slug, fam in product(SLUGS, D3):
        for av in AVS:
            c = res["tabla_d3"][slug][fam][av]
            L.append(f"| {slug} | {fam} | {av or 'prior'} | " + " | ".join(f"{c[f]['t']:.0f}" for f in FORMAS) + " | "
                     + " | ".join(f"{c[f]['delta']:+.2f}" for f in FORMAS) + " |")
    L += ["", "A_V con el prior (nota): " + "; ".join(f"{s} {fam} " + ", ".join(f"{x:.2f}" for x in n)
                                                     for (s, fam), n in zip(product(SLUGS, D3), res["filas_d3"]["nota"])), ""]
    L += ["## Tabla D.4", "", "| objeto | rejilla | máx \\|ΔT\\| | máx \\|ΔΔ\\| | Δ prior ×4 | A_V del cruce ×4 |",
          "|---|---|---|---|---|---|"]
    for (slug, g), f in zip([(s, g) for s in SLUGS for g in D4[s]], res["filas_d4"]):
        cr = " / ".join("—" if x is None else (f"({x[0]:.1f})" if x[1] else f"{x[0]:.1f}") for x in f[6:])
        L.append(f"| {slug} | {g} | {f[0]} | {f[1]:.2f} | " + " / ".join(f"{x:+.2f}" for x in f[2:6]) + f" | {cr} |")
    L += ["", "## Las cifras del texto", ""]
    for c in res["cifras_texto"]:
        L.append(f"- {c['que']}: `{c['tex']}`")
    if dif is not None:
        L += ["", "## Contra el tex", ""]
        L += ["- Tablas D.3 y D.4: " + ("IGUALES al redondeo impreso" if not dif["tablas"] else "DISTINTAS")]
        L += [f"  - {x}" for x in dif["tablas"]]
        L += [f"- Cifras del texto: {len(res['cifras_texto']) - len(dif['texto'])} de {len(res['cifras_texto'])} "
              "aparecen tal como se calculan" + (":" if dif["texto"] else ".")]
        L += [f"  - {x}" for x in dif["texto"]]
    return "\n".join(L) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cadenas", type=Path, action="append", required=True,
                    help="directorio de cadenas de referencia, sin variantes (repetible; se recorre entero)")
    ap.add_argument("--variantes", type=Path, action="append", required=True,
                    help="directorio de las cadenas de los lotes del D.12 (repetible; se recorre entero)")
    ap.add_argument("--project-root", type=Path, default=ROOT,
                    help="raíz que contiene runs/ (por defecto la del repo; necesaria desde un worktree)")
    ap.add_argument("--tex", type=Path, help="main.tex del paper: compara las Tablas D.3 y D.4 y las cifras del D.12")
    ap.add_argument("--estricto", action="store_true",
                    help="con --tex, sale con 1 también si alguna cifra del texto no aparece (sin él, solo las tablas)")
    ap.add_argument("--json", type=Path, help="escribe todo en json")
    ap.add_argument("--md", type=Path, help="escribe el informe en markdown (si no, a la salida estándar)")
    a = ap.parse_args(argv)
    warnings.simplefilter("ignore")
    raiz = tb.fija_raiz(a.project_root)
    res = calcula(tc.Cadenas(a.cadenas), tc.Cadenas(a.variantes), raiz)
    dif = compara_tex(a.tex.read_text(), res) if a.tex else None
    md = informe_md(res, dif)
    if a.md:
        a.md.parent.mkdir(parents=True, exist_ok=True)
        a.md.write_text(md)
        print(f"escrito {a.md}")
    else:
        print(md)
    if a.json:
        a.json.parent.mkdir(parents=True, exist_ok=True)
        a.json.write_text(json.dumps(dict(res, tex=dif), indent=1, ensure_ascii=False,
                                     default=lambda x: list(x) if isinstance(x, tuple) else float(x)))
        print(f"escrito {a.json}")
    if dif and (dif["tablas"] or dif["texto"]):
        print("\n".join(["DISTINTO del tex:"] + dif["tablas"] + dif["texto"]), file=sys.stderr)
    falla = dif is not None and (dif["tablas"] or (a.estricto and dif["texto"]))
    return 1 if falla else 0


if __name__ == "__main__":
    sys.exit(main())
