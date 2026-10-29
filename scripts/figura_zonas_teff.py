#!/usr/bin/env python
"""Qué píxeles de cada espectro entran en los ajustes de T_eff del apéndice B y, si no entran, por qué.

Solo lectura y fuera de la cadena (pedido del autor 2026-10-28: «en el paper, apéndice, debe mostrar un plot
detallado de los espectros, indicando las zonas que se usaron para log Z y las zonas que no, con una indicación de
la razón»). Los ajustes son los de muestreo anidado del apéndice B (`scripts/teff_bayes.py`). Su preparación de los
datos es `tipo_ir_plantillas.preparar(slug, "combinado", av_fija=True, bibliotecas=("manara",), h2o_muse=True)`
(`segmento_muse` y `segmentos_ir`), pasada a erg s⁻¹ cm⁻² Å⁻¹ por `ajuste_conjunto.segmentos_cgs`, más la
cobertura de los modelos de la rejilla (`ajuste_conjunto.Conjunto.comun`). Aquí se rehace esa preparación paso a
paso, con las mismas funciones de `musepipe` y las mismas constantes, guardando para cada canal (MUSE) o píxel (IR)
la razón por la que no entra. Las razones son las que existen en ese código, en este orden de prioridad (un canal
con varias toma la primera):

MUSE (`spec_final_object.fits` de D2, el producto calibrado canónico, 1.25 Å por canal):

- `laser`: el hueco del láser de la AO (`telluric_lines.AO_LASER_WINDOW_A`, 5780–6050 Å): el producto no tiene
  dato (flujo no finito) y 04b marca esos canales como malos. Queda además fuera del rango.
- `rango`: fuera de `g3_fit_wave_range_A` (6300–9349.5 Å; decisión congelada D8).
- `o2`: las bandas telúricas de G3 (`g3_fit_masks.telluric_bands_A`: O2 B y O2 A).
- `h2o`: las bandas de H2O que añade `h2o_muse=True` (`tipo_ir_plantillas.H2O_MUSE_A`), para que todas las
  bibliotecas, también las de X-shooter sin corrección telúrica, vean los mismos píxeles.
- `linea`: ±`line_window_kms` (300 km/s) alrededor de cada línea del catálogo de G2, en reposo.
- `canal_malo`: los canales malos del modelo de fondo local (04b, `stage04b_bad_wavelength_mask.npy`).
- `sin_dato`: flujo o error no finitos, o error ≤ 0, fuera de lo anterior.
- `margen`: la degradación a R = 500 (`observed.degrade_observed`) hace crecer la máscara ±1 FWHM del núcleo
  alrededor de cada canal excluido (y de cada canal sin dato), para que ningún canal que se guarda lleve flujo de
  lo enmascarado. `origen` dice qué razón tiene el canal excluido más cercano que lo causa.
- `bin`: un canal libre de un bin de 12 canales (15 Å) que se descarta por tener más de la mitad de sus canales
  excluidos (`max_masked_frac` de `observed.fit_spectrum_from_inputs`). `origen`: la razón más frecuente entre
  los canales excluidos de su bin.
- `sin_modelo`: un bin que algún modelo de la rejilla no cubre (`Conjunto.comun`).
- `usado`: el canal entra, promediado en su bin de 15 Å, en el ajuste.

IR (los espectros J/H/K del catálogo CDS J/A+A/708/A312 que declara `targets/<slug>.json` -> `ir_spectra`, a
λ/Δλ = 500 por píxel): `sin_dato` (flujo o error no finitos, o error ≤ 0), `telurico_ir`
(`tipo_ir_plantillas.TELURICO_IR_A`), `sin_modelo` y `usado`.

**La prueba de que la reconstrucción es la del ajuste** (`comprueba`, siempre; si algo no cuadra, para con
`AssertionError` y no dibuja nada):
(1) la unión de las máscaras de G3 que se rehacen aquí es la de `load_fit_inputs`, y la máscara crecida la de
`degrade_observed`; (2) los bins de MUSE que salen de los canales marcados `usado` (media del flujo degradado y
error con n/n_eff, como `rebin_for_fit`) son los del segmento MUSE de `preparar` en λ, flujo y error, y los
píxeles IR marcados `usado` son su máscara `bueno`; (3) con la rejilla, los marcados `usado` son `Conjunto.comun`;
(4) con `--cadenas`, se reconstruye `teff_bayes.Modelo` de la cadena BT-Settl con el prior en A_V
(`delta_entre_rejillas.modelo_de`): sus píxeles son los marcados aquí, con los mismos flujos y errores en cgs, su
número es el `n_puntos` de la cadena, y su `loglike` en el punto de máxima verosimilitud da el `logl_max` de la
cadena (el ajuste corrió sobre estos mismos datos).

La figura (`--outdir`): por compañero, el espectro MUSE entero (4750–9350 Å, nativo en gris) con los bins que entran
en negro, y cada espectro IR en su panel (todos sus píxeles en gris, los que entran en negro). Las zonas que no
entran van sombreadas: en gris lo excluido por diseño (fuera del rango, láser) y en color lo enmascarado
(telúrico O2; telúrico H2O en MUSE y bandas telúricas en el IR, que se enmascaran por la misma razón; ventanas de
línea), con el borde de cada zona (margen de R = 500 y bins descartados) en un tono más claro. Con `--cadenas`,
el modelo de máxima verosimilitud en azul, solo sobre los píxeles que entran.

uso: python scripts/figura_zonas_teff.py --project-root RAIZ_CON_RUNS --outdir DIR
                                         [--cadenas ~/muse_noche_20261024_c4/cadenas] [--json SALIDA]
                                         [--rejilla btsettl]
"""
from __future__ import annotations

import argparse
import inspect
import json
import math
import sys
import warnings
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from musepipe.constants import C_KMS  # noqa: E402
from musepipe.models import observed as obs  # noqa: E402
from musepipe.telluric_lines import AO_LASER_WINDOW_A  # noqa: E402
from scripts import ajuste_conjunto as aj  # noqa: E402
from scripts import teff_bayes as tb  # noqa: E402
from scripts import tipo_ir_indices as tir  # noqa: E402
from scripts import tipo_ir_plantillas as tp  # noqa: E402

#: canales por bin de `tipo_ir_plantillas.segmento_muse` (12 × 1.25 Å = 15 Å); `comprueba` falla si cambia allí
CANALES_POR_BIN = 12
#: fracción máxima de canales excluidos de un bin que se conserva: el valor por defecto de
#: `fit_spectrum_from_inputs`, que `segmento_muse` no cambia
MAX_FRAC_EXCLUIDA = float(inspect.signature(obs.fit_spectrum_from_inputs).parameters["max_masked_frac"].default)

#: razones de MUSE en orden de prioridad (las que vienen de una máscara o de la falta de dato; las derivadas,
#: `margen`, `bin` y `sin_modelo`, se asignan después a los canales libres)
CAPAS_MUSE = ("laser", "rango", "o2", "h2o", "linea", "canal_malo", "sin_dato")
RAZONES_MUSE = CAPAS_MUSE + ("margen", "bin", "sin_modelo", "usado")
RAZONES_IR = ("sin_dato", "telurico_ir", "sin_modelo", "usado")
#: lo excluido por diseño (no depende del espectro); el resto está enmascarado
POR_DISENO = ("laser", "rango")

#: cómo se dice cada razón en la figura (inglés, como el paper)
TEXTO = {
    "rango": "outside the fitted range",
    "laser": "AO laser gap (no data)",
    "o2": r"telluric O$_2$ bands (MUSE)",
    "h2o": r"telluric H$_2$O (MUSE) and near-IR bands",
    "linea": r"$\pm${kms:g} km s$^{{-1}}$ around catalogue lines",     # `line_window_kms` del run
    "canal_malo": "bad channels of the local background",
    "sin_dato": "no finite data",
    "borde": r"zone edges (lighter): R = 500 kernel, bins > ½ masked",
    "sin_modelo": "not covered by the model grid",
}
#: colores de lo enmascarado: naranja, aguamarina y violeta de la paleta de referencia del skill `dataviz`
#: (validada todos-contra-todos con `validate_palette.js`: ΔE CVD ≥ 9.2, normal ≥ 27.6); lo de diseño, en grises
COLOR_ZONA = {"o2": "#eb6834", "h2o": "#1baf7a", "telurico_ir": "#1baf7a", "linea": "#6250d6",
              "canal_malo": "#7a7a7a", "sin_dato": "#7a7a7a", "sin_modelo": "#7a7a7a"}
ALFA_ZONA, ALFA_BORDE = 0.32, 0.13
GRIS_RANGO, GRIS_LASER = "0.88", "0.70"
COLOR_MODELO = "#2a78d6"
#: unidad de flujo de la figura (erg s⁻¹ cm⁻² Å⁻¹)
UNIDAD = 1e-17


# ================================================================== reconstrucción
def _en_bandas(w, bandas):
    m = np.zeros(w.size, bool)
    for lo, hi in bandas:
        m |= (w >= float(lo)) & (w <= float(hi))
    return m


def nucleo_r500(w, lsf_fwhm_A):
    """FWHM (Å) con la que `segmento_muse` degrada el dato de la LSF de MUSE a R = 500."""
    return np.sqrt(np.clip((np.asarray(w, float) / tp.R_OBJ) ** 2 - float(lsf_fwhm_A) ** 2, 0, None))


def clasifica_muse(w, f, e, componentes, rango, nucleo, f_deg, e_deg, neff_deg,
                   n_canales=CANALES_POR_BIN, max_frac=MAX_FRAC_EXCLUIDA, laser=AO_LASER_WINDOW_A):
    """La razón de cada canal del espectro MUSE y los bins que entran, sin leer nada de disco.

    `componentes`: {razón: máscara} con 'o2', 'h2o', 'linea' y 'canal_malo' (True = excluido). `nucleo`: la FWHM
    de la degradación por canal. `f_deg`, `e_deg`, `neff_deg`: lo que devuelve `degrade_observed` con la unión de
    las máscaras (los bins se forman con eso). Devuelve un dict con `razon` y `origen` por canal (arrays de str;
    `origen` vacío salvo en `margen` y `bin`), `crecida` (la máscara tras crecer, para compararla con la de
    `degrade_observed`), `bin_id` por canal (−1 fuera del rango) y los bins que entran (`w_bin`, `f_bin`,
    `e_bin`, `bin_usado` por bin del rango)."""
    w, f, e = (np.asarray(x, float) for x in (w, f, e))
    n = w.size
    sin_dato = ~np.isfinite(f) | ~np.isfinite(e) | ~(e > 0)
    en_rango = (w >= float(rango[0])) & (w <= float(rango[1]))
    capas = {"laser": (w >= laser[0]) & (w <= laser[1]), "rango": ~en_rango,
             **{k: np.asarray(componentes.get(k, np.zeros(n, bool)), bool) for k in ("o2", "h2o", "linea", "canal_malo")},
             "sin_dato": sin_dato}
    mascara = capas["o2"] | capas["h2o"] | capas["linea"] | capas["canal_malo"]
    malo = mascara | sin_dato                         # lo que `degrade_observed` interpola y hace crecer
    # razón de cada canal excluido como FUENTE del margen: sin «rango» (una ventana de línea fuera del rango
    # también hace crecer la máscara hacia dentro)
    fuente = np.full(n, "", dtype=object)
    for cod in reversed(("laser", "o2", "h2o", "linea", "canal_malo", "sin_dato")):
        fuente[capas[cod] & malo] = cod
    # el crecimiento de `degrade_observed`, guardando la fuente más cercana de cada canal crecido
    crecida = malo.copy()
    dist = np.full(n, np.inf)
    origen_margen = np.full(n, "", dtype=object)
    for i in np.nonzero(malo)[0]:
        half = float(nucleo[i]) if np.isfinite(nucleo[i]) and nucleo[i] > 0 else 0.0
        if half <= 0:
            continue
        a = np.searchsorted(w, w[i] - half)
        b = np.searchsorted(w, w[i] + half, side="right")
        crecida[a:b] = True
        dd = np.abs(w[a:b] - w[i])
        mejor = dd < dist[a:b]
        dist[a:b][mejor] = dd[mejor]
        origen_margen[a:b][mejor] = fuente[i]
    crecida |= mascara

    razon = np.full(n, "", dtype=object)
    for cod in reversed(CAPAS_MUSE):
        razon[capas[cod]] = cod
    libre = razon == ""
    margen = libre & crecida
    razon[margen] = "margen"
    origen = np.full(n, "", dtype=object)
    origen[margen] = origen_margen[margen]

    # bins de `rebin_for_fit`: grupos de `n_canales` canales del rango, desde el primero
    f_deg, e_deg, neff_deg = (np.asarray(x, float) for x in (f_deg, e_deg, neff_deg))
    fin_deg = np.isfinite(f_deg) & np.isfinite(e_deg) & (e_deg > 0)
    idx = np.nonzero(en_rango)[0]
    bin_id = np.full(n, -1, int)
    w_bin, f_bin, e_bin, usado_bin = [], [], [], []
    for j, ini in enumerate(range(0, idx.size, int(n_canales))):
        ch = idx[ini:ini + int(n_canales)]
        bin_id[ch] = j
        guarda = ~crecida[ch] & fin_deg[ch]
        n_tot, n_g = ch.size, int(guarda.sum())
        if n_g == 0 or (n_tot - n_g) / n_tot > max_frac:
            usado_bin.append(False)
            libres = ch[razon[ch] == ""]
            if libres.size:
                causas = Counter(origen[k] or razon[k] for k in ch if razon[k] not in ("", "rango"))
                razon[libres] = "bin"
                origen[libres] = causas.most_common(1)[0][0] if causas else ""
            continue
        usado_bin.append(True)
        k = ch[guarda]
        ratio = float(np.mean(neff_deg[k]))
        w_bin.append(float(w[k].mean()))
        f_bin.append(float(f_deg[k].mean()))
        e_bin.append(float(np.sqrt((np.sum(e_deg[k] ** 2) / n_g ** 2) * (1.0 / ratio))))
        razon[k] = "usado"
    # un canal libre que la degradación dejara sin dato (no ocurre: `degrade_observed` interpola)
    razon[razon == ""] = "sin_dato"
    return dict(razon=razon, origen=origen, crecida=crecida, bin_id=bin_id, en_rango=en_rango,
                w_bin=np.asarray(w_bin), f_bin=np.asarray(f_bin), e_bin=np.asarray(e_bin),
                bin_usado=np.asarray(usado_bin, bool))


def zonas_muse(run, h2o=True):
    """El espectro MUSE de `run` como lo prepara `tipo_ir_plantillas.segmento_muse`, con la razón de cada canal.

    Devuelve `seg` (el segmento, en el formato de `tipo_ir_plantillas`, con `razon` por bin) y `canales` (λ, flujo,
    error, razón y origen por canal, en las unidades del producto) más lo necesario para comprobarlo."""
    from musepipe.config import load_run_config
    from musepipe.paths import RunPaths
    raiz = tp.ROOT                      # la que `teff_bayes.fija_raiz` haya puesto
    cfg = load_run_config(run, project_root=raiz).config
    rp = RunPaths.from_project_root(run, raiz)
    inputs = obs.load_fit_inputs(cfg, rp)
    w, f, e = inputs["wave"], inputs["flux"], inputs["err"]
    fm = dict(cfg.get("g3_fit_masks", {}))
    comp = {"canal_malo": (np.asarray(inputs["bad_mask"], bool) if inputs["bad_mask"] is not None
                           else np.zeros(w.size, bool))}
    kms = float(fm.get("line_window_kms", 300.0))
    lineas = obs._g2_rest_wavelengths(rp.table_dir)
    comp["linea"] = _en_bandas(w, [(r - r * kms / C_KMS, r + r * kms / C_KMS) for r in lineas])
    bandas_o2 = [tuple(map(float, b)) for b in fm.get("telluric_bands_A", obs.DEFAULT_TELLURIC_BANDS_A)]
    comp["o2"] = _en_bandas(w, bandas_o2)
    d9 = comp["canal_malo"] | comp["linea"] | comp["o2"]
    if not np.array_equal(d9, inputs["mask"]):
        raise AssertionError(f"{run}: la unión de las máscaras rehechas no es la de load_fit_inputs "
                             f"({int((d9 != inputs['mask']).sum())} canales distintos)")
    comp["h2o"] = _en_bandas(w, tp.H2O_MUSE_A) if h2o else np.zeros(w.size, bool)
    mascara = d9 | comp["h2o"]
    lsf = float(cfg["h01_lsf_fwhm_A"])
    nucleo = nucleo_r500(w, lsf)
    ratio = obs.neff_over_n_per_channel(w, inputs["cov"], inputs["bad_mask"])
    f_deg, e_deg, masc_deg, neff_deg = obs.degrade_observed(w, f, e, mascara, nucleo, ratio)
    rango = [float(x) for x in cfg["g3_fit_wave_range_A"]]
    c = clasifica_muse(w, f, e, comp, rango, nucleo, f_deg, e_deg, neff_deg)
    if not np.array_equal(c["crecida"], masc_deg):
        raise AssertionError(f"{run}: la máscara crecida no es la de degrade_observed "
                             f"({int((c['crecida'] != masc_deg).sum())} canales distintos)")
    medio = CANALES_POR_BIN * float(np.median(np.diff(w))) / 2
    wb = c["w_bin"]
    seg = dict(nombre="MUSE", brazo="vis", w=wb, lo=wb - medio, hi=wb + medio, f=c["f_bin"], e=c["e_bin"],
               bueno=np.isfinite(c["f_bin"]) & (c["e_bin"] > 0), razon=np.full(wb.size, "usado", dtype=object))
    canales = dict(w=w, f=f, e=e, razon=c["razon"], origen=c["origen"], bin_id=c["bin_id"], en_rango=c["en_rango"],
                   bin_usado=c["bin_usado"], nucleo=nucleo, rango=rango, lsf=lsf, bandas_o2=bandas_o2,
                   lineas=[float(x) for x in lineas], line_window_kms=kms, n_bad_04b=int(comp["canal_malo"].sum()))
    return seg, canales


def clasifica_ir(w, f, e, telurico=True, bandas=None):
    """La razón de cada píxel de un espectro IR, como `tipo_ir_plantillas.segmentos_ir`."""
    w, f, e = (np.asarray(x, float) for x in (w, f, e))
    razon = np.full(w.size, "usado", dtype=object)
    if telurico:
        razon[_en_bandas(w, tp.TELURICO_IR_A if bandas is None else bandas)] = "telurico_ir"
    razon[~(np.isfinite(f) & np.isfinite(e) & (e > 0))] = "sin_dato"
    return razon


def zonas_ir(slug, telurico=True):
    """Los segmentos IR de `slug` en el formato de `tipo_ir_plantillas.segmentos_ir`, con `razon` por píxel."""
    segs = []
    for nombre, spec in tir.perfil(slug)["ir_spectra"]["spectra"].items():
        w_um, f, e = tir.load(tir.DEFAULT_CDS, spec["file"])
        w = w_um * 1e4
        b = tp.bordes(w)
        razon = clasifica_ir(w, f, e, telurico)
        segs.append(dict(nombre=nombre, brazo="nir", w=w, lo=b[:-1], hi=b[1:], f=f, e=e, bueno=razon == "usado",
                         razon=razon, banda=spec["band"], instrumento=spec["instrument"]))
    return segs


def zonas(slug, h2o_muse=True, telurico_ir=True):
    """Los segmentos de `slug` en el orden de `preparar(slug, "combinado")` (IR y luego MUSE), con sus razones;
    `canales` es el detalle por canal de MUSE."""
    run = tir.perfil(slug)["paper"]["run"]
    seg_muse, canales = zonas_muse(run, h2o=h2o_muse)
    return dict(slug=slug, run=run, segs=zonas_ir(slug, telurico_ir) + [seg_muse], canales=canales)


def aplica_cobertura(z, bases):
    """Marca `sin_modelo` lo que algún modelo de `bases` ({modelo: [base por segmento]}) no cubre, como
    `Conjunto.comun`; en MUSE, también los canales de ese bin."""
    for k, s in enumerate(z["segs"]):
        cubre = np.ones(s["w"].size, bool)
        for m in bases.values():
            cubre &= np.isfinite(m[k])
        fuera = (s["razon"] == "usado") & ~cubre
        s["razon"][fuera] = "sin_modelo"
        s["bueno_final"] = s["razon"] == "usado"
        if s["brazo"] == "vis" and fuera.any():
            c = z["canales"]
            ids = np.nonzero(c["bin_usado"])[0][fuera]
            c["razon"][np.isin(c["bin_id"], ids) & (c["razon"] == "usado")] = "sin_modelo"
    return z


# ================================================================== la prueba
def _igual(a, b, que, rtol=1e-12):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.shape != b.shape:
        raise AssertionError(f"{que}: {a.shape} contra {b.shape}")
    if not np.allclose(a, b, rtol=rtol, atol=0.0, equal_nan=True):
        d = np.nanmax(np.abs(a - b) / np.maximum(np.abs(b), 1e-300))
        raise AssertionError(f"{que}: difieren (máximo relativo {d:.3g})")
    return float(np.nanmax(np.abs(a - b) / np.maximum(np.abs(b), 1e-300))) if a.size else 0.0


def comprueba_preparar(z, segs_p):
    """(2): los segmentos de `preparar` (unidades del producto) son los reconstruidos. Devuelve el máximo
    desvío relativo por segmento."""
    if [s["nombre"] for s in segs_p] != [s["nombre"] for s in z["segs"]]:
        raise AssertionError(f"{z['slug']}: segmentos {[s['nombre'] for s in segs_p]} contra "
                             f"{[s['nombre'] for s in z['segs']]}")
    out = {}
    for sp, s in zip(segs_p, z["segs"]):
        q = f"{z['slug']} {s['nombre']}"
        if s["brazo"] == "vis":
            if not np.array_equal(sp["bueno"], np.ones(sp["w"].size, bool)):
                raise AssertionError(f"{q}: preparar trae bins de MUSE no buenos")
            out[s["nombre"]] = max(_igual(s["w"], sp["w"], f"{q} λ"), _igual(s["f"], sp["f"], f"{q} flujo"),
                                   _igual(s["e"], sp["e"], f"{q} error"), _igual(s["lo"], sp["lo"], f"{q} lo"),
                                   _igual(s["hi"], sp["hi"], f"{q} hi"))
        else:
            for clave in ("w", "f", "e", "lo", "hi"):
                if not np.array_equal(s[clave], sp[clave], equal_nan=True):
                    raise AssertionError(f"{q}: {clave} distinto del de preparar")
            if not np.array_equal(s["razon"] == "usado", sp["bueno"]):
                raise AssertionError(f"{q}: los píxeles usados no son los `bueno` de preparar "
                                     f"({int(((s['razon'] == 'usado') != sp['bueno']).sum())} distintos)")
            out[s["nombre"]] = 0.0
    return out


def comprueba_conjunto(z, conj):
    """(3): tras `aplica_cobertura`, lo marcado `usado` es `Conjunto.comun` de cada segmento."""
    for s, c in zip(z["segs"], conj.comun):
        if not np.array_equal(s["razon"] == "usado", c):
            raise AssertionError(f"{z['slug']} {s['nombre']}: lo usado no es Conjunto.comun")


def segmentos_cgs_de(z):
    """Los segmentos reconstruidos en erg s⁻¹ cm⁻² Å⁻¹ con `ajuste_conjunto.segmentos_cgs` (K reescalada) y el
    factor de cada uno."""
    cgs = aj.segmentos_cgs(z["slug"], dict(run=z["run"], segs=z["segs"]))
    fac = []
    for s, sc in zip(z["segs"], cgs):
        ok = np.isfinite(s["f"]) & (s["f"] != 0)
        fac.append(float(np.median(sc["f"][ok] / s["f"][ok])))
    return cgs, fac


def comprueba_modelo(z, mod, d=None, th=None):
    """(4): los píxeles de `teff_bayes.Modelo` son los marcados `usado`, con los mismos flujos y errores en cgs;
    con la cadena `d` y su mejor punto `th`, su número es el `n_puntos` de la cadena y `loglike(th)` su
    `logl_max`. Devuelve (máximo desvío relativo, loglike)."""
    cgs, _ = segmentos_cgs_de(z)
    por_nombre = {s["nombre"]: (s, sc) for s, sc in zip(z["segs"], cgs)}
    if (d is None or d.get("datos", "todo") == "todo") and {ms["nombre"] for ms in mod.segs} != set(por_nombre):
        raise AssertionError(f"{z['slug']}: el Modelo tiene {[ms['nombre'] for ms in mod.segs]}, las zonas "
                             f"{list(por_nombre)}")
    peor = 0.0
    for ms in mod.segs:
        s, sc = por_nombre[ms["nombre"]]
        u = s["razon"] == "usado"
        q = f"{z['slug']} {ms['nombre']} (Modelo)"
        peor = max(peor, _igual(sc["w"][u], ms["w"], f"{q} λ"), _igual(sc["f"][u], ms["d"], f"{q} flujo"),
                   _igual(sc["e"][u] ** 2, ms["e2"], f"{q} error²"))
        if d is not None and int(d["n_puntos"][ms["nombre"]]) != int(u.sum()):
            raise AssertionError(f"{q}: {int(u.sum())} píxeles contra n_puntos {d['n_puntos'][ms['nombre']]}")
    ll = None
    if d is not None and th is not None:
        ll = float(mod.loglike(th))
        if not abs(ll - float(d["logl_max"])) <= 1e-6 * max(1.0, abs(ll)):
            raise AssertionError(f"{z['slug']}: loglike {ll!r} contra logl_max {d['logl_max']!r} de la cadena")
    return peor, ll


def comprueba(slug, rejilla="btsettl", cadena=None):
    """Reconstruye las zonas de `slug` y hace las comprobaciones (1)–(4). `cadena`: el json de la cadena BT-Settl
    con el prior (con `_json`), o None para no reconstruir el modelo. Devuelve (zonas, informe, modelo, th)."""
    z = zonas(slug)                                  # (1) va dentro
    R = dict(tb.REJILLAS[rejilla])
    p, conj = tb._prepara(slug, R)                   # p["segs"]: los de `preparar`, sin pasar a cgs
    inf = dict(preparar=comprueba_preparar(z, p["segs"]))
    aplica_cobertura(z, p["bases"])
    comprueba_conjunto(z, conj)
    inf["rejilla"] = rejilla
    mod = th = None
    if cadena is not None:
        from scripts import delta_entre_rejillas as dr
        if cadena.get("rejilla", "btsettl") != rejilla:
            raise AssertionError(f"la cadena es de {cadena.get('rejilla')} y las zonas de {rejilla}")
        mod, th = dr.modelo_de(cadena)
        peor, ll = comprueba_modelo(z, mod, cadena, th)
        inf["modelo"] = dict(cadena=cadena.get("_json", cadena.get("etiqueta")), desvio_relativo_max=peor,
                             loglike=ll, logl_max=float(cadena["logl_max"]),
                             n_puntos={s["nombre"]: int(s["d"].size) for s in mod.segs})
    return z, inf, mod, th


# ================================================================== cifras
def tramos(w, sel, lo=None, hi=None):
    """[(inicio, fin)] en Å de los tramos contiguos donde `sel` es True; con `lo`/`hi` (bordes por píxel), los
    bordes de los píxeles; sin ellos, medio canal a cada lado."""
    w = np.asarray(w, float)
    sel = np.asarray(sel, bool)
    if not sel.any():
        return []
    if lo is None:
        b = tp.bordes(w)
        lo, hi = b[:-1], b[1:]
    cambio = np.diff(np.concatenate([[0], sel.astype(int), [0]]))
    ini, fin = np.nonzero(cambio == 1)[0], np.nonzero(cambio == -1)[0] - 1
    return [(float(lo[a]), float(hi[b])) for a, b in zip(ini, fin)]


def resumen(z):
    """Cifras de las zonas de un compañero (para `cifras.md` y el pie): cuentas, Å y tramos por razón."""
    c = z["canales"]
    w, r = c["w"], c["razon"]
    dw = float(np.median(np.diff(w)))
    seg = next(s for s in z["segs"] if s["brazo"] == "vis")
    u = seg["razon"] == "usado"
    muse = dict(n_canales=int(w.size), w_min=float(w[0]), w_max=float(w[-1]), paso_A=dw,
                rango=c["rango"], n_canales_rango=int(c["en_rango"].sum()),
                n_bins_rango=int(c["bin_usado"].size), n_bins_usados=int(u.sum()),
                bin_A=CANALES_POR_BIN * dw, primer_bin_A=float(seg["w"][u][0]), ultimo_bin_A=float(seg["w"][u][-1]),
                n_canales_usados=int((r == "usado").sum()),
                fraccion_canales_usados=float((r == "usado").mean()),
                fraccion_canales_rango_usados=float((r[c["en_rango"]] == "usado").mean()),
                lsf_A=c["lsf"], nucleo_A_rango=[float(c["nucleo"][c["en_rango"]].min()),
                                                float(c["nucleo"][c["en_rango"]].max())],
                bandas_o2=c["bandas_o2"], bandas_h2o=[list(b) for b in tp.H2O_MUSE_A],
                n_lineas=len(c["lineas"]), line_window_kms=c["line_window_kms"],
                lineas_en_rango=int(sum(c["rango"][0] <= x <= c["rango"][1] for x in c["lineas"])),
                n_canales_malos_04b=c["n_bad_04b"], laser_A=list(AO_LASER_WINDOW_A),
                max_frac_excluida=MAX_FRAC_EXCLUIDA, razones={})
    for cod in RAZONES_MUSE:
        sel = r == cod
        if not sel.any():
            continue
        d = dict(n_canales=int(sel.sum()), A=float(sel.sum() * dw), fraccion=float(sel.mean()),
                 n_canales_en_rango=int((sel & c["en_rango"]).sum()), tramos=tramos(w, sel))
        if cod in ("margen", "bin"):
            d["por_origen"] = {o: int(((c["origen"] == o) & sel).sum()) for o in sorted(set(c["origen"][sel]))}
        muse["razones"][cod] = d
    # S/N por canal (producto nativo) en el azul que no entra y en los canales que entran
    fin = np.isfinite(c["f"]) & np.isfinite(c["e"]) & (c["e"] > 0)
    azul = fin & (w < c["rango"][0]) & ~((w >= AO_LASER_WINDOW_A[0]) & (w <= AO_LASER_WINDOW_A[1]))
    muse["snr_mediana_azul"] = float(np.median(c["f"][azul] / c["e"][azul]))
    muse["snr_mediana_usados"] = float(np.median(c["f"][r == "usado"] / c["e"][r == "usado"]))
    muse["snr_mediana_bins_usados"] = float(np.median(seg["f"][u] / seg["e"][u]))
    # R = 250 (`teff_bayes._empareja`): los pares de bins contiguos
    par = tb._empareja({**seg, "bueno": u})
    pu = par["bueno"]
    muse["r250"] = dict(n_bins=int(pu.sum()), primer_A=float(par["w"][pu][0]), ultimo_A=float(par["w"][pu][-1]))
    ir = []
    for s in z["segs"]:
        if s["brazo"] != "nir":
            continue
        e = dict(nombre=s["nombre"], banda=s["banda"], instrumento=s["instrumento"], n=int(s["w"].size),
                 w_min=float(s["lo"][0]), w_max=float(s["hi"][-1]), n_usados=int((s["razon"] == "usado").sum()),
                 razones={})
        for cod in RAZONES_IR:
            sel = s["razon"] == cod
            if sel.any():
                e["razones"][cod] = dict(n=int(sel.sum()), tramos=tramos(s["w"], sel, s["lo"], s["hi"]))
        ir.append(e)
    n_ir = sum(e["n_usados"] for e in ir)
    return dict(slug=z["slug"], run=z["run"], muse=muse, ir=ir, n_usados_ir=n_ir,
                n_usados_total=muse["n_bins_usados"] + n_ir,
                bandas_telurico_ir=[list(b) for b in tp.TELURICO_IR_A])


# ================================================================== figura
#: geometría de la figura (pulgadas): ancho de `figure*` de A&A y alturas de cada franja
ALTO_LEYENDA, ALTO_MUSE, HUECO_MUSE, ALTO_IR, HUECO_IR = 0.74, 1.28, 0.45, 0.95, 0.53
#: lo que necesita el rótulo del eje x de la última fila (el resto de `HUECO_IR` sobra al final)
PIE_IR = 0.34
MARGEN_IZQ, MARGEN_DER, HUECO_PANELES_IR = 0.50, 0.04, 0.30


def _sombrea(ax, a, b, cod, borde=False):
    if cod == "rango":
        ax.axvspan(a, b, color=GRIS_RANGO, lw=0, zorder=0)
    elif cod == "laser":
        ax.axvspan(a, b, color=GRIS_LASER, lw=0, zorder=0)
    else:
        ax.axvspan(a, b, color=COLOR_ZONA[cod], alpha=ALFA_BORDE if borde else ALFA_ZONA, lw=0, zorder=0)


def _con_huecos(x, *ys, factor=1.6):
    """Inserta NaN donde faltan píxeles, para que una línea no cruce lo que no entra."""
    x = np.asarray(x, float)
    if x.size < 3:
        return (x, *ys)
    dx = np.diff(x)
    salto = np.nonzero(dx > factor * np.median(dx))[0]
    return (np.insert(x, salto + 1, np.nan), *[np.insert(np.asarray(y, float), salto + 1, np.nan) for y in ys])


def _panel_muse(ax, z, escala, modelo=None):
    """El espectro MUSE entero en escala logarítmica (el azul del rango y el rojo difieren en dos órdenes): nativo en
    gris, bins del ajuste en negro, zonas sombreadas por razón y rótulos de las bandas arriba."""
    c = z["canales"]
    w, r, o = c["w"], c["razon"], c["origen"]
    k = escala / UNIDAD
    vistos = set()
    for cod in ("rango", "laser", "o2", "h2o", "linea", "canal_malo", "sin_dato", "sin_modelo"):
        for a, b in tramos(w, r == cod):
            _sombrea(ax, a, b, cod)
            vistos.add(cod)
    for cod in ("margen", "bin"):
        for org in sorted(set(o[r == cod])):
            for a, b in tramos(w, (r == cod) & (o == org)):
                if not org or org in POR_DISENO:
                    _sombrea(ax, a, b, org or "sin_dato")
                else:
                    _sombrea(ax, a, b, org, borde=True)
                    vistos.add("borde")
    ax.plot(w, c["f"] * k, color="0.50", lw=0.25, zorder=1)
    seg = next(s for s in z["segs"] if s["brazo"] == "vis")
    u = seg["razon"] == "usado"
    ax.errorbar(seg["w"][u], seg["f"][u] * k, yerr=seg["e"][u] * k, fmt="o", ms=1.6, color="0.0",
                elinewidth=0.5, capsize=0, zorder=4)
    if modelo is not None:
        xm, ym = _con_huecos(seg["w"][u], modelo / UNIDAD)
        ax.plot(xm, ym, color=COLOR_MODELO, lw=0.8, zorder=5)
    import matplotlib.ticker
    ax.set_yscale("log", nonpositive="mask")
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    fb = seg["f"][u] * k
    en = c["en_rango"] & (r != "laser")
    techo = max(float(np.max(fb + seg["e"][u] * k)), float(np.nanpercentile(c["f"][en] * k, 99.8)))
    ax.set_ylim(0.25 * float(np.min(fb[fb > 0])), 2.6 * techo)
    ax.set_xlim(4750.0, 9350.0)
    # rótulos directos de las zonas grandes, arriba del panel
    rot = [("outside the fitted range", 0.5 * (4750.0 + AO_LASER_WINDOW_A[0]), "center"),
           ("AO laser", 0.5 * sum(AO_LASER_WINDOW_A), "center")]
    for (a, b), t in zip(c["bandas_o2"], (r"O$_2$ B", r"O$_2$ A")):
        rot.append((t, 0.5 * (a + b), "center"))
    for a, b in tp.H2O_MUSE_A:
        if a < 9350.0:
            rot.append((r"H$_2$O", 0.5 * (a + b), "center") if b <= 9350.0 else (r"H$_2$O", 9336.0, "right"))
    rot += [(r"H$\alpha$", 6562.8, "center"), ("Ca II, O I, Pa", 8652.0, "center")]
    for t, x, ha in rot:
        ax.annotate(t, (x, 1.0), xycoords=("data", "axes fraction"), xytext=(0, -2.5),
                    textcoords="offset points", ha=ha, va="top", fontsize=5.5, color="0.2")
    return vistos


def _panel_ir(ax, s, sc, modelo=None):
    w, r = sc["w"], s["razon"]
    k = 1.0 / UNIDAD
    vistos = set()
    for cod in ("telurico_ir", "sin_dato", "sin_modelo"):
        for a, b in tramos(w, r == cod, s["lo"], s["hi"]):
            _sombrea(ax, a / 1e4, b / 1e4, "h2o" if cod == "telurico_ir" else cod)
            vistos.add("h2o" if cod == "telurico_ir" else cod)
    ax.plot(w / 1e4, sc["f"] * k, color="0.50", lw=0.5, zorder=1)
    u = r == "usado"
    ax.errorbar(w[u] / 1e4, sc["f"][u] * k, yerr=sc["e"][u] * k, fmt="o", ms=1.6, color="0.0",
                elinewidth=0.5, capsize=0, zorder=4)
    if modelo is not None:
        xm, ym = _con_huecos(w[u] / 1e4, modelo / UNIDAD)
        ax.plot(xm, ym, color=COLOR_MODELO, lw=0.8, zorder=5)
    ax.set_xlim(s["lo"][0] / 1e4, s["hi"][-1] / 1e4)
    todo = np.concatenate([sc["f"] * k, sc["f"][u] * k + sc["e"][u] * k, sc["f"][u] * k - sc["e"][u] * k])
    lo, hi = float(np.nanmin(todo)), float(np.nanmax(todo))
    ax.set_ylim(lo - 0.10 * (hi - lo), hi + 0.10 * (hi - lo))
    return vistos


def dibuja(entradas, out_pdf, out_png=None):
    """La figura de página completa. `entradas`: [(nombre, z, modelos o None)] por compañero, con `modelos` =
    {nombre de segmento: modelo en cgs sobre los píxeles usados}."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker
    from matplotlib.legend_handler import HandlerTuple
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from scripts import paper_figures as pf

    pf.estilo_aa()
    W = pf.ANCHO_DOBLE_IN
    bloque = ALTO_MUSE + HUECO_MUSE + ALTO_IR + HUECO_IR
    H = ALTO_LEYENDA + len(entradas) * bloque - (HUECO_IR - PIE_IR)
    fig = plt.figure(figsize=(W, H))
    ancho = W - MARGEN_IZQ - MARGEN_DER

    def rect(x0, y_arriba, w_in, h_in):
        return [x0 / W, (H - y_arriba - h_in) / H, w_in / W, h_in / H]

    eje_y = r"$F_\lambda$ ($10^{%d}$ erg s$^{-1}$ cm$^{-2}$ $\mathrm{\AA}^{-1}$)" % int(round(math.log10(UNIDAD)))
    vistos = set()
    for i, (nombre, z, modelos) in enumerate(entradas):
        cgs, fac = segmentos_cgs_de(z)
        # los paneles IR por longitud de onda (J, H, K), no en el orden del ajuste
        segs_ir = sorted(((s, sc) for s, sc in zip(z["segs"], cgs) if s["brazo"] == "nir"),
                         key=lambda x: float(x[0]["lo"][0]))
        k_vis = next(f for s, f in zip(z["segs"], fac) if s["brazo"] == "vis")
        y = ALTO_LEYENDA + i * bloque
        ax = fig.add_axes(rect(MARGEN_IZQ, y, ancho, ALTO_MUSE))
        vistos |= _panel_muse(ax, z, k_vis, None if modelos is None else modelos.get("MUSE"))
        m = resumen(z)["muse"]
        ax.set_title(f"{nombre}: MUSE, {m['n_bins_usados']} bins of "
                     rf"{m['bin_A']:.0f} $\mathrm{{\AA}}$ at R = {tp.R_OBJ:.0f} in the fit "
                     f"({m['n_canales_usados']} of {m['n_canales']} channels)", loc="left", pad=3)
        ax.set_xlabel(r"Wavelength ($\mathrm{\AA}$, air)", labelpad=1)
        # un solo rótulo de eje y por compañero, centrado en sus dos filas (las mismas unidades)
        fig.text(0.02 / W, 1.0 - (y + 0.5 * (ALTO_MUSE + HUECO_MUSE + ALTO_IR)) / H, eje_y, rotation=90,
                 ha="left", va="center", fontsize=8)
        n_ir = len(segs_ir)
        a_ir = (ancho - (n_ir - 1) * HUECO_PANELES_IR) / n_ir
        y_ir = y + ALTO_MUSE + HUECO_MUSE
        ejes = []
        for j, (s, sc) in enumerate(segs_ir):
            axi = fig.add_axes(rect(MARGEN_IZQ + j * (a_ir + HUECO_PANELES_IR), y_ir, a_ir, ALTO_IR))
            vistos |= _panel_ir(axi, s, sc, None if modelos is None else modelos.get(s["nombre"]))
            nu = int((s["razon"] == "usado").sum())
            axi.set_title(f"{s['instrumento'].split('/')[-1]} {s['banda']}: {nu} of {s['w'].size}",
                          loc="left", pad=2, fontsize=6.5)
            axi.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=4, steps=[1, 2, 5, 10]))
            axi.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=4))
            axi.tick_params(labelsize=6.5)
            ejes.append(axi)
        p0, p1 = ejes[0].get_position(), ejes[-1].get_position()
        fig.text(0.5 * (p0.x0 + p1.x1), p0.y0 - 0.20 / H, r"Wavelength ($\mu$m)", ha="center", va="top", fontsize=8)

    # leyenda arriba, en tres grupos: datos y modelo, excluido por diseño, enmascarado
    datos = [(Line2D([], [], color="0.50", lw=0.6), r"all pixels (MUSE: native 1.25 $\mathrm{\AA}$ channels)"),
             (Line2D([], [], color="0.0", marker="o", ms=2.2, lw=0), r"pixels in the fit (MUSE: 15 $\mathrm{\AA}$ bins)")]
    if any(e[2] for e in entradas):
        datos.append((Line2D([], [], color=COLOR_MODELO, lw=0.9), r"BT-Settl, maximum likelihood, prior on $A_V$"))
    diseno = [(Patch(color=GRIS_RANGO, lw=0), TEXTO["rango"]), (Patch(color=GRIS_LASER, lw=0), TEXTO["laser"])]
    kms = sorted({e[1]["canales"]["line_window_kms"] for e in entradas})
    if len(kms) != 1:
        raise SystemExit(f"las ventanas de línea difieren entre compañeros ({kms} km/s): la leyenda da una sola")
    mascaras = [(Patch(color=COLOR_ZONA[cod], alpha=ALFA_ZONA, lw=0), TEXTO[cod].format(kms=kms[0]))
                for cod in ("o2", "h2o", "linea", "canal_malo", "sin_dato", "sin_modelo") if cod in vistos]
    if "borde" in vistos:
        mascaras.append((tuple(Patch(color=COLOR_ZONA[c], alpha=ALFA_BORDE, lw=0) for c in ("o2", "h2o", "linea")),
                         TEXTO["borde"]))
    y0 = 1.0 - 0.02 / H
    kw = dict(loc="upper left", fontsize=6.3, handlelength=1.6, handleheight=0.9, labelspacing=0.20,
              borderaxespad=0.0, borderpad=0.0, title_fontsize=6.5, alignment="left", frameon=False,
              handler_map={tuple: HandlerTuple(ndivide=None, pad=0.0)})
    fig.legend(*zip(*datos), bbox_to_anchor=(0.0, y0), title="Data and model", **kw)
    fig.legend(*zip(*diseno), bbox_to_anchor=(0.395, y0), title="Excluded by design", **kw)
    fig.legend(*zip(*mascaras), bbox_to_anchor=(0.605, y0), title="Masked", **kw)
    fig.savefig(out_pdf, bbox_inches=None)
    if out_png:
        fig.savefig(out_png, dpi=200, bbox_inches=None)
    plt.close(fig)


# ================================================================== main
def objetos(root=ROOT):
    """[(slug, nombre en el paper)] de `targets/` con `paper` e `ir_spectra`, por `paper.order`."""
    from musepipe.targets import targets_dir
    out = []
    for f in sorted(targets_dir(root).glob("*.json")):
        d = json.loads(f.read_text())
        if "paper" in d and "ir_spectra" in d:
            out.append((int(d["paper"].get("order", 99)), f.stem, d["paper"].get("display") or d.get("display_name", f.stem)))
    if not out:
        raise SystemExit("ningún targets/*.json declara `paper` e `ir_spectra`")
    return [(s, n) for _, s, n in sorted(out)]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--project-root", type=Path, default=None,
                    help="clon del que se leen los runs y las bibliotecas (un worktree no los lleva)")
    ap.add_argument("--outdir", type=Path, help="dónde se escriben teff_fit_zones.pdf y .png (sin él, no dibuja)")
    ap.add_argument("--cadenas", type=Path, action="append", default=None,
                    help="directorio de cadenas de `teff_bayes.py ajusta` (repetible): reconstruye el modelo de "
                         "máxima verosimilitud de BT-Settl con el prior en A_V y lo compara con la cadena")
    ap.add_argument("--rejilla", default="btsettl", choices=sorted(tb.REJILLAS),
                    help="la rejilla cuya cobertura se comprueba (y la del modelo)")
    ap.add_argument("--json", type=Path, help="escribe las cifras y las comprobaciones en json")
    a = ap.parse_args(argv)
    warnings.simplefilter("ignore", RuntimeWarning)
    if a.project_root:
        tb.fija_raiz(a.project_root)
    K = None
    if a.cadenas:
        from scripts import teff_cifras_texto as tc
        K = tc.Cadenas(a.cadenas)
    salida, entradas = {}, []
    for slug, nombre in objetos():
        cadena = None
        if K is not None:
            from scripts import teff_cifras_texto as tc
            et = tc.etiqueta(slug, a.rejilla)
            if et not in K.por_etiqueta:
                raise SystemExit(f"falta la cadena {et} en {a.cadenas}")
            cadena = K.por_etiqueta[et]
        z, inf, mod, th = comprueba(slug, a.rejilla, cadena)
        r = resumen(z)
        r["comprobaciones"] = inf
        salida[slug] = r
        modelos = None
        if mod is not None:
            modelos = {s["nombre"]: m for s, m in zip(mod.segs, mod.modelos(th))}
        entradas.append((nombre, z, modelos))
        m = r["muse"]
        print(f"{slug}: MUSE {m['n_bins_usados']} bins ({m['primer_bin_A']:.0f}–{m['ultimo_bin_A']:.0f} Å; "
              f"R = 250: {m['r250']['n_bins']}, {m['r250']['primer_A']:.0f}–{m['r250']['ultimo_A']:.0f} Å), IR "
              + ", ".join(f"{e['nombre']} {e['n_usados']}/{e['n']}" for e in r["ir"])
              + "; comprobado contra preparar" + (" y la cadena (loglike = logl_max)" if mod is not None else ""))
    if a.outdir:
        a.outdir.mkdir(parents=True, exist_ok=True)
        dibuja(entradas, a.outdir / "teff_fit_zones.pdf", a.outdir / "teff_fit_zones.png")
        print(f"escrito {a.outdir / 'teff_fit_zones.pdf'} y .png")
    if a.json:
        a.json.parent.mkdir(parents=True, exist_ok=True)
        a.json.write_text(json.dumps(salida, indent=1, ensure_ascii=False, default=float))
        print(f"escrito {a.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
