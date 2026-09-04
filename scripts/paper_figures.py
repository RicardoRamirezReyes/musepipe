#!/usr/bin/env python
"""Dibuja las figuras del paper en PDF vectorial, a partir de los productos.

Una funcion por figura, cada una nombrada como el fichero que `paper/main.tex`
espera en `figures/`. Ninguna recalcula ciencia: todas leen el QC, la tabla o el
FITS que la etapa ya escribio, de modo que la figura no puede discrepar del
numero citado en el texto.

Que objeto se publica desde que run NO se decide aqui: sale de
`targets/<slug>.json` -> clave `paper` (`run`, `order`, `note`). Un literal como
el id de un run en `scripts/` hace que un objeto nuevo herede en silencio los
datos del primero (ver `tests/test_no_hardcoded_target.py`).

    python scripts/paper_figures.py                       # las siete, a paper/figures
    python scripts/paper_figures.py --figure null_distributions
    python scripts/paper_figures.py --outdir /tmp/figs --list

`mdot_mass_plane` necesita ademas una compilacion de literatura en CSV
(`--literature`, por defecto `paper/literature_mdot.csv`): sin ella dibuja solo
las dos restricciones de este trabajo y lo avisa, porque inventar los puntos de
literatura seria inventar la comparacion que la figura existe para hacer.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

#: anchos de columna de A&A (aa.cls): `\hsize` en dos columnas y `figure*`.
ANCHO_COL_IN = 88.0 / 25.4
ANCHO_DOBLE_IN = 180.0 / 25.4

#: Halpha en reposo, en aire. El mismo valor que usa E1 (`h01` line.rest_A).
HALPHA_A = 6562.8
#: Hbeta en reposo, en aire; el mismo que el catalogo de lineas de G2.
HBETA_A = 4861.33
#: razon intrinseca Halpha/Hbeta del caso B (Osterbrock & Ferland 2006,
#: T = 10^4 K, n_e = 10^4 cm^-3). Es una SUPOSICION, no una medida: se dibuja
#: para ensenar donde caeria Hbeta sin extincion diferencial, y va rotulada.
CASO_B_HA_HB = 2.86

#: un color por objeto, asignado por el orden declarado en `targets/`.
COLOR_OBJETO = ("#1f77b4", "#d62728")
#: un color por metodo de extraccion, estable entre figuras.
COLOR_METODO = {
    "aperture": "#1f77b4",
    "optimal_ls": "#ff7f0e",
    "optimal_psfsub": "#2ca02c",
    "psffit": "#d62728",
    "sgf": "#9467bd",
    "lpm": "#8c564b",
}
#: como se escriben los metodos en el paper.
NOMBRE_METODO = {
    "aperture": "aperture",
    "optimal_ls": "optimal (LS bkg)",
    "optimal_psfsub": "optimal (PSF-sub)",
    "psffit": "PSF fitting",
    "sgf": "SGF",
    "lpm": "LPM",
}

MSUN_A_MJUP = 1047.5655


def estilo_aa():
    """Tipografia y grosores de A&A. Type 42 para que el PDF lleve el texto."""
    plt.rcParams.update({
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8,
        "legend.fontsize": 6.5,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.minor.width": 0.4,
        "ytick.minor.width": 0.4,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "lines.linewidth": 0.9,
        "legend.frameon": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    })


# --------------------------------------------------------------------------
# procedencia: que objeto, que run, como se llama en el texto
# --------------------------------------------------------------------------

class Objeto:
    """Un objeto del paper con su run de publicacion ya resuelto."""

    def __init__(self, slug: str, ficha: dict):
        bloque = ficha.get("paper")
        if not bloque or not bloque.get("run"):
            raise SystemExit(
                f"targets/{slug}.json no declara `paper.run`: sin el no se sabe "
                "sobre que run se publica este objeto."
            )
        self.slug = slug
        # `paper.display` es la tipografia del paper (companero en minuscula);
        # `display_name` es la del registro, y hay un test que la fija.
        self.nombre = bloque.get("display") or ficha.get("display_name", slug)
        self.run_id = bloque["run"]
        self.orden = int(bloque.get("order", 99))
        self.nota = bloque.get("note", "")
        # Segunda epoca, si el objeto la declara. Sale de `targets/<slug>.json`
        # y NUNCA de un literal aqui (`tests/test_no_hardcoded_target.py`).
        self.run_2a_epoca = bloque.get("second_epoch_run")
        self.spt_source_dir = bloque.get("spt_source_dir")
        self.run_dir = ROOT / "runs" / self.run_id
        if not self.run_dir.is_dir():
            raise SystemExit(f"no existe runs/{self.run_id} (objeto {slug})")

    # -- rutas ------------------------------------------------------------
    def etapa(self, nombre: str, *, run_id: str | None = None) -> Path:
        raiz = ROOT / "runs" / run_id if run_id else self.run_dir
        p = raiz / "stages" / nombre
        if not p.exists():
            raise SystemExit(f"falta {p} — {self.nombre} no tiene ese producto")
        return p

    def tabla(self, nombre: str) -> Path:
        p = self.run_dir / "tables" / nombre
        if not p.exists():
            raise SystemExit(f"falta {p} — {self.nombre} no tiene esa tabla")
        return p

    # -- lecturas ---------------------------------------------------------
    def qc(self, nombre: str) -> dict:
        return json.loads(self.etapa(nombre).read_text())

    def filas(self, nombre: str) -> list[dict]:
        with open(self.tabla(nombre), newline="") as fh:
            return list(csv.DictReader(fh))

    @property
    def config(self) -> dict:
        return json.loads((self.run_dir / "config" / "config.json").read_text())["config"]


def objetos_del_paper() -> list[Objeto]:
    fichas = sorted((ROOT / "targets").glob("*.json"))
    if not fichas:
        raise SystemExit("no hay targets/*.json: el registro de objetos esta vacio")
    salida = []
    for f in fichas:
        ficha = json.loads(f.read_text())
        if "paper" in ficha:
            salida.append(Objeto(f.stem, ficha))
    if not salida:
        raise SystemExit(
            "ningun targets/*.json declara la clave `paper`: declara ahi el run "
            "de publicacion de cada objeto antes de dibujar."
        )
    return sorted(salida, key=lambda o: o.orden)


def _color(i: int) -> str:
    return COLOR_OBJETO[i % len(COLOR_OBJETO)]


def _espectro(obj: Objeto, metodo: str, *, cual: str = "object", run_id: str | None = None):
    """`(wave_A, flux, err_total, escala_cgs)` del espectro calibrado de D2.

    `cual` elige compañero (`object`) o primaria (`star`); `run_id` permite leer
    otra epoca del MISMO objeto, declarada en `targets/<slug>.json`.

    La unidad viaja con el dato: se resuelve con `musepipe.io`, sin default.
    """
    from astropy.io import fits

    from musepipe.io import resolve_flux_unit

    with fits.open(obj.etapa(f"spec_calibrated_{metodo}_{cual}.fits", run_id=run_id)) as hdul:
        t = hdul["SPECTRUM"]
        wave = np.asarray(t.data["wave_A"], dtype=float)
        flux = np.asarray(t.data["flux"], dtype=float)
        err = np.asarray(t.data["flux_err_total"], dtype=float)
        bunit = t.header.get("BUNIT")
    cfg = obj.config if run_id is None else json.loads(
        (ROOT / "runs" / run_id / "config" / "config.json").read_text())["config"]
    escala, _ = resolve_flux_unit(cfg, bunit=bunit)
    return wave, flux, err, escala


# --------------------------------------------------------------------------
# Fig. fov_redband — imagenes de banda roja de los dos sistemas
# --------------------------------------------------------------------------

def fov_redband(objetos, out: Path):
    """Colapso en la banda roja de cada cubo, con las dos posiciones marcadas.

    La banda es la que B3 uso para localizar (`companion.band_used_A`), no una
    elegida aqui: es la que sostiene la posicion que se dibuja.
    """
    from astropy.io import fits
    from matplotlib.colors import LogNorm

    fig, axes = plt.subplots(1, len(objetos), figsize=(ANCHO_DOBLE_IN, 3.5))
    axes = np.atleast_1d(axes)

    for ax, obj in zip(axes, objetos):
        qc = obj.qc("stage01c_qc.json")
        pix = float(qc["pixel_scale_arcsec"])
        banda = qc["companion"]["band_used_A"]
        py, px = qc["primary"]["pos_yx"]
        cy, cx = qc["companion"]["pos_yx"]
        norte = float(qc["wcs_orientation"]["north_angle_deg"])

        with fits.open(obj.etapa("stage01_cropped_cube_stack.fits")) as hdul:
            wave = np.asarray(hdul["WAVELENGTH"].data, dtype=float)
            sel = (wave >= banda[0]) & (wave <= banda[1])
            cubo = np.asarray(hdul["CUBES"].data[0], dtype=np.float32)
            img = np.nanmedian(cubo[sel], axis=0)

        ny, nx = img.shape
        # ejes en segundos de arco relativos a la primaria
        ext = [(-0.5 - px) * pix, (nx - 0.5 - px) * pix,
               (-0.5 - py) * pix, (ny - 0.5 - py) * pix]

        pos = img[np.isfinite(img) & (img > 0)]
        vmax = float(np.nanpercentile(pos, 99.99))
        vmin = max(float(np.nanpercentile(pos, 40.0)), vmax * 1e-5)
        ax.imshow(img, origin="lower", extent=ext, cmap="magma",
                  norm=LogNorm(vmin=vmin, vmax=vmax), interpolation="nearest")

        dx = (cx - px) * pix
        dy = (cy - py) * pix
        ax.plot(0.0, 0.0, marker="+", color="white", ms=6, mew=1.0)
        ax.add_patch(Circle((dx, dy), 0.09, fill=False, color="#00e5ff", lw=0.8))
        ax.annotate("b", (dx, dy), textcoords="offset points", xytext=(7, 5),
                    color="#00e5ff", fontsize=7.5)
        ax.annotate("A", (0.0, 0.0), textcoords="offset points", xytext=(6, 4),
                    color="white", fontsize=7.5)

        # barra de escala de 0.5" abajo a la izquierda
        x0 = ext[0] + 0.10 * (ext[1] - ext[0])
        y0 = ext[2] + 0.08 * (ext[3] - ext[2])
        ax.plot([x0, x0 + 0.5], [y0, y0], color="white", lw=1.4, solid_capstyle="butt")
        ax.text(x0 + 0.25, y0 + 0.03 * (ext[3] - ext[2]), r'$0.5^{\prime\prime}$',
                color="white", ha="center", fontsize=7)

        # Brujula: las direcciones se piden a la MISMA funcion de B3 que coloca
        # al compañero en este panel, `pixel_offset_from_sep_pa`, en vez de
        # rehacer la trigonometria aqui. Rehecha, estaba 180 deg girada en los
        # dos objetos: la convencion de B3 pone el Norte en -y cuando
        # `north_angle_deg` es 0, y la brujula lo ponia en +y, de modo que un
        # lector que comprobase el PA publicado del compañero contra el dibujo
        # no habria encontrado 240 deg sino ~60. Con esto no pueden discrepar:
        # si B3 cambia de convencion, la brujula la sigue.
        from musepipe.stages.stage01c_localize import pixel_offset_from_sep_pa

        cxp = ext[1] - 0.16 * (ext[1] - ext[0])
        cyp = ext[3] - 0.20 * (ext[3] - ext[2])
        largo = 0.055 * (ext[1] - ext[0])
        for pa_deg, etq in ((0.0, "N"), (90.0, "E")):
            vy, vx = pixel_offset_from_sep_pa(1.0, pa_deg, pix, norte)
            norma = math.hypot(vx, vy)
            ux, uy = vx / norma * largo, vy / norma * largo
            ax.annotate("", xy=(cxp + ux, cyp + uy), xytext=(cxp, cyp),
                        arrowprops=dict(arrowstyle="-|>", color="white", lw=0.7,
                                        mutation_scale=6))
            ax.text(cxp + 1.35 * ux, cyp + 1.35 * uy, etq, color="white",
                    ha="center", va="center", fontsize=6.5)

        ax.set_title(f"{obj.nombre}   ({banda[0]:.0f}$-${banda[1]:.0f}" r"$\,\mathrm{\AA}$)")
        ax.set_xlabel("offset from primary (arcsec)")
        ax.set_ylabel("offset from primary (arcsec)")
        ax.set_aspect("equal")

    fig.savefig(out)
    plt.close(fig)


# --------------------------------------------------------------------------
# Fig. spectra_halpha — espectros del companero alrededor de Halpha
# --------------------------------------------------------------------------

def spectra_halpha(objetos, out: Path, metodos=("aperture", "optimal_ls", "psffit"),
                   media_ventana_A=45.0):
    """Espectros calibrados de D2 en la ventana de Halpha, tres metodos.

    La banda sombreada es `flux_err_total` (estadistico conservador + los
    sistematicos de C1/A3/A4 en cuadratura) del metodo canonico, no de los tres:
    solaparlas oculta las lineas.
    """
    fig, axes = plt.subplots(1, len(objetos), figsize=(ANCHO_DOBLE_IN, 2.6))
    axes = np.atleast_1d(axes)
    canonico = metodos[-1]

    for ax, obj in zip(axes, objetos):
        lo, hi = [], []
        for metodo in metodos:
            wave, flux, err, escala = _espectro(obj, metodo)
            sel = np.abs(wave - HALPHA_A) <= media_ventana_A
            w = wave[sel]
            f = flux[sel] * escala * 1e18
            e = err[sel] * escala * 1e18
            if metodo == canonico:
                ax.fill_between(w, f - e, f + e, color=COLOR_METODO[metodo],
                                alpha=0.22, lw=0)
                lo.append(np.nanmin(f - e))
            lo.append(np.nanmin(f))
            hi.append(np.nanmax(f))
            ax.plot(w, f, color=COLOR_METODO[metodo], lw=0.8,
                    label=NOMBRE_METODO[metodo])

        # El pico de un metodo salia recortado por el borde del panel: el rango
        # se fija con lo dibujado, no con el autoscale.
        y0, y1 = min(lo), max(hi)
        margen = 0.08 * (y1 - y0)
        ax.set_ylim(y0 - margen, y1 + margen)
        ax.axvline(HALPHA_A, color="0.35", lw=0.5, ls=":")
        ax.set_title(obj.nombre)
        ax.set_xlabel(r"Wavelength ($\mathrm{\AA}$, barycentric)")
        ax.set_xlim(HALPHA_A - media_ventana_A, HALPHA_A + media_ventana_A)
        ax.legend(loc="upper left", handlelength=1.4)

    axes[0].set_ylabel(r"$F_\lambda$ ($10^{-18}$ erg s$^{-1}$ cm$^{-2}$ "
                       r"$\mathrm{\AA}^{-1}$)")
    fig.savefig(out)
    plt.close(fig)


# --------------------------------------------------------------------------
# Fig. full_spectra — los dos espectros de D2, de punta a punta
# --------------------------------------------------------------------------

def full_spectra(objetos, out: Path, metodo="psffit", suavizado=41):
    """Primaria y compañero en TODO el rango, con su error y lo que lo estropea.

    Los productos definitivos de D2 son los espectros, antes que el estudio de
    Halpha, y hasta ahora el paper solo los enseñaba por ventanas de 90 A. Aqui
    van enteros y canal a canal, con:

    - la banda de \\pm1\\sigma del presupuesto total (la misma de la Fig. de Halpha);
    - las **bandas telúricas** con su profundidad medida por A3, que es lo que
      explica los huecos del rojo sin que el lector tenga que adivinarlo;
    - el **hueco del láser AO** (5780-6050 A), donde no hay dato: sombrearlo es
      obligatorio, porque ahi el flujo vale 0 y no NaN;
    - las líneas de acrecion del catalogo de G2, para situar Halpha entre ellas.

    La curva gruesa es una mediana movil; la fina, el dato canal a canal. La
    primaria y el compañero van en paneles distintos porque se llevan tres
    ordenes de magnitud.
    """
    from musepipe.paper_spectrum import AO_LASER_WINDOW_A, accretion_lines
    from musepipe.telluric_lines import TELLURIC_BANDS

    obj = objetos[0]
    fig, axes = plt.subplots(2, 1, figsize=(ANCHO_DOBLE_IN, 4.6), sharex=True)
    lineas = [l for l in accretion_lines() if l["kind"].startswith("accretion")]

    for ax, cual, etq in ((axes[0], "star", f"{obj.nombre.rsplit(' ', 1)[0]} A (primary)"),
                          (axes[1], "object", f"{obj.nombre} (companion)")):
        wave, flux, err, escala = _espectro(obj, metodo, cual=cual)
        f = flux * escala * 1e18
        e = err * escala * 1e18
        # sin dato en el hueco del laser: alli el flujo vale 0, no NaN
        hueco = (wave >= AO_LASER_WINDOW_A[0]) & (wave <= AO_LASER_WINDOW_A[1])
        f = np.where(hueco, np.nan, f); e = np.where(hueco, np.nan, e)
        for b in TELLURIC_BANDS:
            gris = {"strong": "0.82", "moderate": "0.90"}.get(b.get("severity"), "0.955")
            ax.axvspan(b["lo_A"], b["hi_A"], color=gris, lw=0, zorder=0)
        ax.axvspan(*AO_LASER_WINDOW_A, color="#ffd9d9", lw=0, zorder=0)
        ax.fill_between(wave, f - e, f + e, color=COLOR_METODO[metodo], alpha=0.20, lw=0, zorder=2)
        ax.plot(wave, f, color=COLOR_METODO[metodo], lw=0.25, alpha=0.55, zorder=3)
        ax.plot(wave, _mediana_movil(f, suavizado), color="0.10", lw=0.7, zorder=4)
        finito = np.isfinite(f)
        lo, hi = np.nanpercentile(f[finito], [0.5, 99.8])
        ax.set_ylim(lo - 0.15 * (hi - lo), hi + 0.30 * (hi - lo))
        for l in lineas:
            ax.axvline(l["wave_A"], color="0.45", lw=0.4, ls=":", zorder=1)
        ax.set_title(etq, fontsize=8)
        ax.set_ylabel(r"$F_\lambda$ ($10^{-18}$ cgs)")

    axes[-1].set_xlabel(r"Wavelength ($\mathrm{\AA}$, barycentric)")
    axes[-1].set_xlim(float(np.nanmin(wave)), float(np.nanmax(wave)))
    fig.subplots_adjust(hspace=0.22)
    fig.savefig(out)
    plt.close(fig)


# --------------------------------------------------------------------------
# Fig. primary_variability — la primaria cambia entre noches, la compañera no
# --------------------------------------------------------------------------

def _continuo_local(wave, flux, centro_A, media_ventana_A, excluir_A):
    """Mediana del continuo en la ventana, excluyendo el entorno de la linea."""
    sel = (np.abs(wave - centro_A) <= media_ventana_A) & (np.abs(wave - centro_A) > excluir_A)
    return float(np.nanmedian(flux[sel])) if np.any(sel) else np.nan


def primary_variability(objetos, out: Path, media_ventana_A=45.0):
    """Halpha de la primaria y del compañero en las dos epocas del mismo objeto.

    Es la figura del argumento de la luz dispersada: la primaria cambia entre
    noches y el compañero no la sigue. Solo se dibuja para los objetos que
    declaran `second_epoch_run` en `targets/<slug>.json`.

    La primaria va **normalizada a su propio continuo**: entre las dos noches
    cambian tambien las condiciones, asi que un F_lambda absoluto mezclaria la
    variacion de la linea con la de la transmision. Normalizado, lo que se ve es
    la anchura equivalente, que es la cantidad robusta.

    El panel del compañero lleva su banda de +-1 sigma en las dos noches, que es
    lo que deja ver que la segunda **no tiene sensibilidad** a la linea: no
    ensena que el compañero no varie, ensena que esa noche no puede decirlo.
    """
    objs = [o for o in objetos if o.run_2a_epoca]
    if not objs:
        raise SystemExit("ningun objeto declara `second_epoch_run` en targets/")
    obj = objs[0]
    fig, axes = plt.subplots(1, 2, figsize=(ANCHO_DOBLE_IN, 2.7))

    epocas = [("first night", None, "#1f77b4"), ("second night", obj.run_2a_epoca, "#d62728")]

    # --- panel a: la primaria, normalizada a su continuo
    ax = axes[0]
    for etq, run_id, color in epocas:
        wave, flux, _err, escala = _espectro(obj, "psffit", cual="star", run_id=run_id)
        sel = np.abs(wave - HALPHA_A) <= media_ventana_A
        base = _continuo_local(wave, flux, HALPHA_A, media_ventana_A, 4.0)
        ax.plot(wave[sel], flux[sel] / base, color=color, lw=0.9, label=etq)
    ax.axhline(1.0, color="0.6", lw=0.5, ls="-")
    ax.axvline(HALPHA_A, color="0.35", lw=0.5, ls=":")
    ax.set_title(f"{obj.nombre.rsplit(' ', 1)[0]} A (primary)")
    ax.set_ylabel("normalised flux")

    # --- panel b: el compañero, en flujo, con su error
    ax = axes[1]
    for etq, run_id, color in epocas:
        wave, flux, err, escala = _espectro(obj, "psffit", cual="object", run_id=run_id)
        sel = np.abs(wave - HALPHA_A) <= media_ventana_A
        w = wave[sel]; f = flux[sel] * escala * 1e18; e = err[sel] * escala * 1e18
        ax.fill_between(w, f - e, f + e, color=color, alpha=0.20, lw=0)
        ax.plot(w, f, color=color, lw=0.9, label=etq)
    ax.axhline(0.0, color="0.6", lw=0.5, ls="-")
    ax.axvline(HALPHA_A, color="0.35", lw=0.5, ls=":")
    ax.set_title(f"{obj.nombre} (companion)")
    ax.set_ylabel(r"$F_\lambda$ ($10^{-18}$ erg s$^{-1}$ cm$^{-2}$ $\mathrm{\AA}^{-1}$)")

    for ax in axes:
        ax.set_xlim(HALPHA_A - media_ventana_A, HALPHA_A + media_ventana_A)
        ax.set_xlabel(r"Wavelength ($\mathrm{\AA}$, barycentric)")
        ax.legend(loc="upper left", handlelength=1.4)
    fig.savefig(out)
    plt.close(fig)


# --------------------------------------------------------------------------
# Fig. companion_type — que clase de objeto es el compañero
# --------------------------------------------------------------------------

def _g3_template_fit(obj: Objeto):
    """El ajuste de plantillas de G3, del directorio que el objeto DECLARA.

    `paper.spt_source_dir` en `targets/<slug>.json`. Sin el se cae al mas
    reciente, pero avisando: elegir por orden alfabetico entre varios reintentos
    archivados es como se acaba publicando el de otra cosecha.
    """
    stages = obj.run_dir / "stages"
    declarado = obj.spt_source_dir
    if declarado:
        p = stages / declarado / "g3_template_fit.json"
        if not p.exists():
            raise SystemExit(f"{obj.nombre}: targets declara spt_source_dir={declarado} y no existe {p}")
        return json.loads(p.read_text()), p.parent
    cands = sorted(stages.glob("g3_real_*/g3_template_fit.json")) + sorted(stages.glob("g3_template_fit.json"))
    if not cands:
        raise SystemExit(f"{obj.nombre}: no hay g3_template_fit.json en runs/{obj.run_id}")
    print(f"  AVISO: {obj.nombre} no declara `paper.spt_source_dir`; se usa {cands[-1].parent.name}")
    return json.loads(cands[-1].read_text()), cands[-1].parent


def _binado(wave, flux, paso_A):
    bordes = np.arange(wave[0], wave[-1] + paso_A, paso_A)
    idx = np.digitize(wave, bordes) - 1
    w, f = [], []
    for k in range(len(bordes) - 1):
        m = idx == k
        if m.sum() >= 3 and np.isfinite(flux[m]).sum() >= 3:
            w.append(np.nanmean(wave[m])); f.append(np.nanmedian(flux[m]))
    return np.asarray(w), np.asarray(f)


def companion_type(objetos, out: Path, paso_A=25.0, rango_A=(6000.0, 9100.0)):
    """Espectro del compañero contra las plantillas, y el chi2 por tipo espectral.

    Lo que esta figura sostiene es la **clase**, no los parametros fisicos: el
    ajuste de atmosferas no pasa sus puertas y no se usa (Apendice A). Las dos
    plantillas se dibujan con el A_V de su propio mejor ajuste y reescaladas por
    minimos cuadrados al espectro observado, porque el `scale_best` del ajuste
    vive en las unidades de la libreria y no en las del producto.
    """
    from musepipe.models.extinction import CCMExtinction
    from musepipe.models.prep import prepare_template

    obj = objetos[0]
    tf, raiz = _g3_template_fit(obj)
    lsf = float(obj.config["h01_lsf_fwhm_A"])
    ext = CCMExtinction(3.1, citation="Cardelli et al. 1989")

    wave, flux, err, escala = _espectro(obj, "psffit")
    m = (wave >= rango_A[0]) & (wave <= rango_A[1])
    wb, fb = _binado(wave[m], flux[m] * escala * 1e18, paso_A)

    fig, axes = plt.subplots(1, 2, figsize=(ANCHO_DOBLE_IN, 2.8),
                             gridspec_kw={"width_ratios": [2.0, 1.0]})
    ax = axes[0]
    ax.plot(wb, fb, color="0.15", lw=0.9, label=f"{obj.nombre}, {paso_A:.0f} " r"$\mathrm{\AA}$ bins")

    for clase, color, etq in (("young", "#d62728", "young"), ("field", "#1f77b4", "field")):
        mejor = tf[clase]["ranking"][0]
        libro = ROOT.parent / "Data" / "external_libraries" / f"templates_{clase}"
        # las dos librerias nombran distinto: `LM601_M7.5.npz` frente a `M9.npz`
        ficheros = (sorted(libro.glob(f"*_{mejor['spt']}.npz"))
                    or sorted(libro.glob(f"{mejor['spt']}.npz")))
        if not ficheros:
            continue
        d = np.load(ficheros[0], allow_pickle=True)
        meta = json.loads(str(d["meta_json"]))
        class _T:  # el adaptador minimo que espera prepare_template
            pass
        t = _T(); t.wave_A = d["wave_A"]; t.flux = d["flux"]; t.meta = meta
        modelo = prepare_template(t, wb, lsf_fwhm_A=lsf, extinction=ext,
                                  av=float(mejor["av_best"]), scale=1.0,
                                  template_fwhm_A=meta.get("resolution_fwhm_A"))
        ok = np.isfinite(modelo) & np.isfinite(fb)
        if ok.sum() < 5:
            continue
        k = float(np.nansum(modelo[ok] * fb[ok]) / np.nansum(modelo[ok] ** 2))
        ax.plot(wb, k * modelo, color=color, lw=1.0, alpha=0.85,
                label=f"{etq} {mejor['spt']} " r"($\chi^2_\nu=$" f"{mejor['chi2_red']:.1f})")
    ax.set_xlabel(r"Wavelength ($\mathrm{\AA}$, barycentric)")
    ax.set_ylabel(r"$F_\lambda$ ($10^{-18}$ erg s$^{-1}$ cm$^{-2}$ $\mathrm{\AA}^{-1}$)")
    ax.set_xlim(*rango_A)
    ax.legend(loc="upper left", handlelength=1.4)

    ax = axes[1]
    for clase, color, marca in (("young", "#d62728", "o"), ("field", "#1f77b4", "s")):
        r = [(x["spt_code"], x["chi2_red"]) for x in tf[clase]["ranking"]]
        r.sort()
        ax.plot([x[0] for x in r], [x[1] for x in r], marker=marca, ms=3.0, lw=0.8,
                color=color, label=clase)
    ax.axhline(3.0, color="0.5", lw=0.6, ls="--")
    ax.set_yscale("log")
    # la libreria de campo llega hasta tipos tempranos (codigos negativos); el
    # panel se queda en el entorno del minimo, que es donde se decide.
    ax.set_xlim(2.0, 11.0)
    ax.set_xlabel("spectral type (M0 = 0, L0 = 10)")
    ax.set_ylabel(r"$\chi^2_\nu$")
    ax.legend(loc="upper right", handlelength=1.4)
    fig.savefig(out)
    plt.close(fig)


# --------------------------------------------------------------------------
# Fig. hbeta_limit — la NO deteccion de Hbeta, y por que no acota la extincion
# --------------------------------------------------------------------------

def _linea_g2(obj: Objeto, nombre: str) -> dict:
    """La fila de `g2_line_measurements.csv` de una linea, o `{}` si no esta."""
    for fila in obj.filas("g2_line_measurements.csv"):
        if fila.get("name") == nombre:
            return fila
    return {}


def _gaussiana(wave, centro_A, flujo, fwhm_A):
    """Perfil gaussiano de flujo INTEGRADO `flujo` y anchura `fwhm_A`."""
    sigma = float(fwhm_A) / 2.3548200450309493
    return (flujo / (sigma * np.sqrt(2.0 * np.pi))
            * np.exp(-0.5 * ((wave - centro_A) / sigma) ** 2))


def hbeta_limit(objetos, out: Path, metodo="psffit", media_ventana_A=45.0):
    """Hbeta no detectada, con el limite a 5 sigma y lo que el caso B predice.

    Existe porque la no deteccion de Hbeta es la que cierra la via del
    decremento de Balmer para medir la extincion, y un limite solo se puede
    juzgar viendo contra que ruido se midio. Tres cosas por panel:

    - el espectro calibrado de D2 y su `flux_err_total` (el mismo error que la
      figura de Halpha), en la ventana de Hbeta;
    - el limite a 5 sigma de G2 dibujado como perfil, con la LSF **medida** de
      A4/M2 que declara el run, no una anchura inventada;
    - donde caeria Hbeta si el decremento fuese el del caso B, escalando el
      Halpha MEDIDO de este mismo objeto. Solo se dibuja donde Halpha esta
      detectado, porque escalar un limite no ensena nada.

    Que ese perfil quede por debajo del ruido es el resultado: el limite de
    Hbeta no acota el decremento, y por tanto no acota A_V.
    """
    fig, axes = plt.subplots(1, len(objetos), figsize=(ANCHO_DOBLE_IN, 2.6))
    axes = np.atleast_1d(axes)

    for ax, obj in zip(axes, objetos):
        wave, flux, err, escala = _espectro(obj, metodo)
        cfg = obj.config
        lsf = float(cfg["h01_lsf_fwhm_A"])
        rv = float(cfg.get("h01_rv_sys_kms", 0.0))
        centro = HBETA_A * (1.0 + rv / 299792.458)

        sel = np.abs(wave - centro) <= media_ventana_A
        w = wave[sel]
        f = flux[sel] * escala * 1e18
        e = err[sel] * escala * 1e18
        ax.fill_between(w, f - e, f + e, color="0.6", alpha=0.30, lw=0,
                        label=r"$\pm1\sigma$")
        # el espectro va en gris oscuro y el rojo queda RESERVADO para lo que
        # el caso B predice: con los dos en el color del metodo, la prediccion
        # se confundia con el ruido del propio espectro.
        ax.plot(w, f, color="0.15", lw=0.8, label=NOMBRE_METODO[metodo])
        ax.axvline(centro, color="0.35", lw=0.5, ls=":")

        # nivel local del continuo, medido en la propia ventana y fuera de la
        # linea: dibujar los perfiles sobre cero mentiria sobre donde estan.
        fuera = np.abs(w - centro) > 3.0 * lsf
        base = float(np.nanmedian(f[fuera])) if np.any(fuera) else 0.0
        # los dos perfiles se dibujan SOLO alrededor de la linea: extendidos a
        # toda la ventana se leen como un continuo modelo, que no lo son.
        cerca = np.abs(w - centro) <= 6.0 * lsf

        hb = _linea_g2(obj, "Hbeta")
        lim = float(hb.get("flux_upper_limit_5sigma", "nan"))
        if np.isfinite(lim) and lim > 0:
            perfil = base + _gaussiana(w[cerca], centro, lim * escala * 1e18, lsf)
            ax.plot(w[cerca], perfil, color="#1f77b4", lw=0.9, ls="--",
                    label=r"$5\sigma$ upper limit")

        ha = _linea_g2(obj, "Halpha")
        if ha.get("status") == "detected" and np.isfinite(lim) and lim > 0:
            f_ha = float(ha.get("flux_fit") or ha.get("flux_direct"))
            esperado = f_ha / CASO_B_HA_HB
            perfil = base + _gaussiana(w[cerca], centro, esperado * escala * 1e18, lsf)
            ax.plot(w[cerca], perfil, color="#d62728", lw=1.0,
                    label=r"H$\alpha$/%.2f (case B)" % CASO_B_HA_HB)
            # el numero es el resultado: cuantas veces por encima de lo que
            # habria que ver esta el limite. Sin el, la figura solo ensena
            # que una curva roja es pequena.
            ax.annotate(r"limit $= %.1f\times$ the case-B line" % (lim / esperado),
                        xy=(0.97, 0.06), xycoords="axes fraction", ha="right",
                        va="bottom", fontsize=6.5, color="0.25")

        ax.set_title(obj.nombre)
        ax.set_xlim(centro - media_ventana_A, centro + media_ventana_A)
        ax.set_xlabel(r"Wavelength ($\mathrm{\AA}$, barycentric)")
        ax.legend(loc="upper left", handlelength=1.6)

    axes[0].set_ylabel(r"$F_\lambda$ ($10^{-18}$ erg s$^{-1}$ cm$^{-2}$ "
                       r"$\mathrm{\AA}^{-1}$)")
    fig.savefig(out)
    plt.close(fig)


# --------------------------------------------------------------------------
# Fig. null_distributions — el estadistico contra su nula
# --------------------------------------------------------------------------

def null_distributions(objetos, out: Path, metodo="psffit"):
    """Nula empirica de los controles, la Gumbel ajustada y el valor medido.

    La Gumbel es la que E1 usa para la FAP parametrica: se dibuja con los
    `loc`/`scale` guardados en el QC, no re-ajustados aqui.
    """
    fig, axes = plt.subplots(len(objetos), 1, figsize=(ANCHO_COL_IN, 3.6),
                             sharex=False)
    axes = np.atleast_1d(axes)

    for i, (ax, obj) in enumerate(zip(axes, objetos)):
        nulos = np.load(obj.etapa("stage_h01_null_maxima.npz"))[f"{metodo}_null_maxima"]
        qc = obj.qc("stage_h01_qc.json")
        par = qc["parametric_fap"]["by_method"][metodo]
        fila = next(r for r in obj.filas("halpha_detection_by_method.csv")
                    if r["method"] == metodo)
        z = float(fila["matched_z"])

        ax.hist(nulos, bins=12, color=_color(i), alpha=0.35,
                density=True, label=f"controls ($n={len(nulos)}$)")
        loc, scale = float(par["loc"]), float(par["scale"])
        x = np.linspace(min(nulos.min(), 0.0), max(nulos.max(), z) * 1.08, 400)
        u = (x - loc) / scale
        ax.plot(x, np.exp(-(u + np.exp(-u))) / scale, color="0.25", lw=0.9,
                label=f"Gumbel fit ($p_{{\\rm KS}}={par['ks_p']:.2f}$)")
        ax.axvline(z, color=_color(i), lw=1.3)
        # abajo y a la izquierda de la linea: arriba a la derecha esta la leyenda
        ax.annotate(f"companion\n$z={z:.1f}$", (z, ax.get_ylim()[1] * 0.16),
                    textcoords="offset points", xytext=(-4, 0), ha="right",
                    va="center", fontsize=6.5, color=_color(i))
        ax.set_title(f"{obj.nombre} — {NOMBRE_METODO[metodo]}", loc="left")
        ax.set_ylabel("density")
        ax.legend(loc="upper right", handlelength=1.3)

    axes[-1].set_xlabel(r"matched-filter statistic $z$")
    fig.tight_layout(pad=0.3)
    fig.savefig(out)
    plt.close(fig)


# --------------------------------------------------------------------------
# Fig. injection_throughput — recuperacion contra flujo inyectado
# --------------------------------------------------------------------------

def injection_throughput(objetos, out: Path):
    """Throughput de E4 contra el flujo de linea inyectado, por metodo.

    Solo `variant == nominal` y `continuum_mode == none`: las variantes de PSF
    +/-10 % son el sistematico, no la curva, y el modo `flat` no existe en los
    dos objetos, asi que mezclarlo compararia poblaciones distintas.
    """
    fig, axes = plt.subplots(len(objetos), 1, figsize=(ANCHO_COL_IN, 3.9),
                             sharex=False)
    axes = np.atleast_1d(axes)

    for ax, obj in zip(axes, objetos):
        filas = [r for r in obj.filas("injection_throughput_by_method.csv")
                 if r["variant"] == "nominal" and r["continuum_mode"] == "none"
                 and float(r["injected_flux"]) > 0]
        qc = obj.qc("stage_h04_qc.json")
        meseta = qc["throughput"]["per_method_at_snr5"]

        for metodo in COLOR_METODO:
            sub = [r for r in filas if r["method"] == metodo]
            if not sub:
                continue
            porf: dict[float, list[float]] = {}
            for r in sub:
                v = float(r["throughput"])
                if math.isfinite(v):
                    porf.setdefault(float(r["injected_flux"]), []).append(v)
            xs = sorted(porf)
            med = np.array([np.median(porf[x]) for x in xs])
            err = np.array([np.std(porf[x], ddof=1) / math.sqrt(len(porf[x]))
                            if len(porf[x]) > 1 else 0.0 for x in xs])
            ax.errorbar(xs, med, yerr=err, color=COLOR_METODO[metodo], lw=0.8,
                        marker="o", ms=2.0, capsize=1.2, elinewidth=0.6,
                        label=NOMBRE_METODO[metodo])
            if metodo in meseta:
                ax.axhline(meseta[metodo]["throughput"], color=COLOR_METODO[metodo],
                           lw=0.5, ls=":", alpha=0.7)

        ax.axhline(1.0, color="0.5", lw=0.5, ls="--")
        ax.set_xscale("log")
        ax.set_title(obj.nombre, loc="left")
        ax.set_ylabel("throughput")
        ax.set_xlabel(r"injected line flux ($10^{-20}$ erg s$^{-1}$ cm$^{-2}$)")

    axes[0].legend(ncol=2, loc="lower right", handlelength=1.3)
    fig.tight_layout(pad=0.3)
    fig.savefig(out)
    plt.close(fig)


# --------------------------------------------------------------------------
# Fig. mdot_mass_plane — las dos restricciones en el plano Mdot-masa
# --------------------------------------------------------------------------

def _mdot_de_este_trabajo(obj: Objeto) -> dict:
    """La medida o el limite de Mdot del objeto, leidos de G3/E3."""
    cfg = obj.config
    masa_mjup = float(cfg["h03_companion_mass_msun"]) * MSUN_A_MJUP
    g3 = obj.qc("stage_g3_qc.json")
    mdot = g3.get("mdot_p50_msun_yr")
    if mdot is not None:
        # El error asimetrico sale de la fila `mdot` de la tabla de G3, que es
        # el percentil del MC; el QC solo publica la mediana.
        fila = next((r for r in obj.filas("g3_physical_properties.csv")
                     if r["property"] == "mdot"), None)
        lo = hi = None
        if fila:
            lo = _o_nan(fila["err_stat_lo"])
            hi = _o_nan(fila["err_stat_hi"])
        return {"masa_mjup": masa_mjup, "mdot": float(mdot), "limite": False,
                "lo": lo, "hi": hi}

    # Sin medida: el limite superior de E3 en el metodo canonico de D2, en la
    # fila `method` (no `combined_final`, que agrega los seis). Se cita la
    # relacion de Alcala+17, la misma que G3 usa para la deteccion; la de
    # Aoyama+21 viaja en `mdot_aoyama21_msun_yr` y da ~x20 mas.
    canonico = obj.qc("stage_x11_qc.json")["canonical_method"]
    fila = next(r for r in obj.filas("halpha_upper_limits.csv")
                if r["row_kind"] == "method" and r["method"] == canonico)
    return {"masa_mjup": masa_mjup, "mdot": float(fila["mdot_msun_yr"]),
            "limite": True, "lo": None, "hi": None}


def mdot_mass_plane(objetos, out: Path, literatura: Path | None = None):
    """Las dos restricciones homogeneas, sobre la compilacion de literatura.

    La literatura NO se inventa: la genera `scripts/build_literature_mdot.py`
    desde CASPAR (Betti+23), y aqui solo se dibuja. Se separan las
    determinaciones hechas con la MISMA relacion que este trabajo (Alcala+17)
    de las demas: es la unica comparacion homogenea, y la dispersion vertical
    de un mismo objeto entre relaciones es lo que el pie afirma.
    """
    fig, ax = plt.subplots(figsize=(ANCHO_COL_IN, 2.9))

    if literatura is not None and literatura.exists():
        with open(literatura, newline="") as fh:
            # el CSV lleva la procedencia de cada cifra en cabecera comentada
            lit = list(csv.DictReader(l for l in fh if not l.lstrip().startswith("#")))

        def _es_limite(r):
            return str(r.get("is_limit", "")).strip().lower() in ("1", "true", "yes")

        def _misma_relacion(r):
            # el CSV trae el nombre con acento grave, tal cual lo escribe CASPAR
            return "alcal" in str(r.get("relation", "")).lower()

        grupos = [
            ("same relation (Alcal\u00e1+17)", _misma_relacion, "#4c72b0", 0.85),
            ("other relations", lambda r: not _misma_relacion(r), "0.62", 0.8),
        ]
        for etiqueta, prueba, color, alfa in grupos:
            med = [(float(r["mass_mjup"]), float(r["mdot_msun_yr"]))
                   for r in lit if prueba(r) and not _es_limite(r)]
            lim = [(float(r["mass_mjup"]), float(r["mdot_msun_yr"]))
                   for r in lit if prueba(r) and _es_limite(r)]
            if med:
                ax.plot(*zip(*med), ls="none", marker="o", ms=3.2, mfc=color,
                        mec="0.25", mew=0.4, alpha=alfa, label=f"lit., {etiqueta}")
            if lim:
                xs, ys = zip(*lim)
                ax.errorbar(xs, ys, yerr=[np.array(ys) * 0.5, np.zeros(len(ys))],
                            ls="none", marker="v", ms=3.2, color=color,
                            mec="0.25", mew=0.4, elinewidth=0.5, capsize=0,
                            uplims=True, alpha=alfa,
                            label=f"lit. upper lim., {etiqueta}")
    else:
        print(f"AVISO: sin compilacion de literatura ({literatura}); la figura "
              "sale solo con las dos restricciones de este trabajo.", file=sys.stderr)

    for i, obj in enumerate(objetos):
        d = _mdot_de_este_trabajo(obj)
        c = _color(i)
        if d["limite"]:
            ax.errorbar([d["masa_mjup"]], [d["mdot"]],
                        yerr=[[d["mdot"] * 0.6], [0.0]], uplims=True,
                        ls="none", marker="v", ms=5.0, color=c, mfc=c,
                        elinewidth=0.9, capsize=0,
                        label=f"{obj.nombre} (this work, upper limit)")
        else:
            # barra vertical = percentiles del MC de G3; el 0.3 dex de dispersion
            # de la relacion de Alcala+17 ya va dentro de ese MC.
            yerr = None
            if d["lo"] and d["hi"] and math.isfinite(d["lo"]) and math.isfinite(d["hi"]):
                yerr = [[d["mdot"] - d["lo"]], [d["hi"] - d["mdot"]]]
            ax.errorbar([d["masa_mjup"]], [d["mdot"]], yerr=yerr, ls="none",
                        marker="*", ms=9.0, color=c, mec="k", mew=0.4,
                        elinewidth=0.9, capsize=1.5,
                        label=f"{obj.nombre} (this work)")

    ax.set_xscale("log")
    ax.set_yscale("log")
    # Con solo dos puntos el rango automatico los pega a las esquinas; y en log
    # las etiquetas de masa salen como 1.2x10^1. Se fijan a mano.
    ax.set_xlim(4.0, 45.0)
    ax.set_xticks([5, 10, 20, 40])
    ax.set_xticklabels(["5", "10", "20", "40"])
    ax.minorticks_off()
    ax.set_xlabel(r"companion mass ($M_{\rm Jup}$)")
    ax.set_ylabel(r"$\dot{M}$ ($M_\odot$ yr$^{-1}$)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.19), ncol=2,
              handlelength=1.2, columnspacing=1.0, fontsize=5.8)
    fig.savefig(out)
    plt.close(fig)


# --------------------------------------------------------------------------
# Fig. contrast_curves — sensibilidad de linea en todo el campo
# --------------------------------------------------------------------------

def contrast_curves(objetos, out: Path):
    """Curvas de contraste de linea a 5 sigma, con la banda 25--75 %.

    Un panel por objeto y una curva por metodo de sustraccion de halo: son los
    unicos que E5 corre en rejilla de anillo.
    """
    fig, axes = plt.subplots(len(objetos), 1, figsize=(ANCHO_COL_IN, 3.9))
    axes = np.atleast_1d(axes)

    for ax, obj in zip(axes, objetos):
        qc = obj.qc("stage_h05_qc.json")
        pix = float(obj.qc("stage01c_qc.json")["pixel_scale_arcsec"])
        umbral = float(qc["params"]["threshold_sigma"])
        filas = obj.filas("contrast_curve_by_method.csv")
        sep_comp = _separacion_arcsec(obj)

        for metodo in sorted({r["method"] for r in filas}):
            sub = sorted((r for r in filas if r["method"] == metodo),
                         key=lambda r: float(r["separation_px"]))
            sep = np.array([float(r["separation_px"]) for r in sub]) * pix
            # una celda vacia es "no alcanzado con ningun contraste de la
            # rejilla" — se deja como hueco, no se rellena con el vecino.
            c50, c25, c75 = (np.array([_o_nan(r[k]) for r in sub])
                             for k in ("contrast_50", "contrast_25", "contrast_75"))
            col = COLOR_METODO.get(metodo, "0.4")
            ax.fill_between(sep, c25, c75, color=col, alpha=0.20, lw=0)
            # Con marcador: la curva es escalonada porque el contraste inyectado
            # es una rejilla de 9 valores log-espaciados, no un barrido continuo.
            # Dibujarla lisa fingiria una resolucion que la medida no tiene.
            ax.plot(sep, c50, color=col, lw=0.9, marker="o", ms=2.0,
                    label=NOMBRE_METODO.get(metodo, metodo))

        ax.axvline(sep_comp, color="0.35", lw=0.6, ls="--")
        ax.annotate("companion", (sep_comp, ax.get_ylim()[1]), rotation=90,
                    textcoords="offset points", xytext=(3, -22), fontsize=6,
                    color="0.35", va="top")
        ax.set_yscale("log")
        ax.set_title(f"{obj.nombre} — {umbral:.0f}$\\sigma$ line contrast", loc="left")
        ax.set_ylabel("line contrast")
        ax.set_xlabel("separation (arcsec)")
        ax.legend(loc="upper right", handlelength=1.3)

    fig.tight_layout(pad=0.3)
    fig.savefig(out)
    plt.close(fig)


def _o_nan(texto) -> float:
    """Celda de CSV a float; la vacia es NaN, no cero."""
    return float(texto) if str(texto).strip() else float("nan")


def _separacion_arcsec(obj: Objeto) -> float:
    qc = obj.qc("stage01c_qc.json")
    py, px = qc["primary"]["pos_yx"]
    cy, cx = qc["companion"]["pos_yx"]
    return math.hypot(cy - py, cx - px) * float(qc["pixel_scale_arcsec"])


# --------------------------------------------------------------------------
# Fig. psf_chromatic — el modelo de PSF contra la longitud de onda
# --------------------------------------------------------------------------

def _csv_float(path: Path, columnas):
    with open(path, newline="") as fh:
        filas = list(csv.DictReader(fh))
    salida = {}
    for c in columnas:
        salida[c] = np.array([float(r[c]) if r.get(c) not in ("", None) else np.nan
                              for r in filas])
    return salida, filas


def _mediana_movil(y, ventana: int):
    """Mediana movil centrada, con los bordes acortados en vez de rellenados."""
    y = np.asarray(y, dtype=float)
    media = np.empty_like(y)
    h = ventana // 2
    for i in range(y.size):
        media[i] = np.nanmedian(y[max(0, i - h):min(y.size, i + h + 1)])
    return media


def psf_chromatic(objetos, out: Path):
    """Los dos ajustes de C1 contra lambda: FWHM de Moffat y r0 de psfao.

    C1 ajusta siempre las dos formas y se queda con la de menor residuo de
    anillo; el panel de abajo dibuja ese residuo, que es el criterio. La linea
    de trazos en r0 es la ley de Fried (r0 propto lambda^{6/5}) normalizada en
    el bin central, no un ajuste.
    """
    fig, axes = plt.subplots(3, 1, figsize=(ANCHO_COL_IN, 4.7), sharex=True)

    for i, obj in enumerate(objetos):
        col = _color(i)
        moffat, _ = _csv_float(obj.etapa("stage_e01_psf_params.csv"),
                               ["wave_center_A", "fwhm_maj", "fwhm_min",
                                "ring_residual_pct"])
        psfao, _ = _csv_float(obj.etapa("stage_e01_psfao_params.csv"),
                              ["lambda_A", "r0", "ring_residual_pct"])
        pix = float(obj.qc("stage01c_qc.json")["pixel_scale_arcsec"])
        forma = obj.qc("stage_e01_qc.json").get("psf_form", "")

        # Moffat: puntos sueltos y una mediana movil. Unirlos con linea fingiria
        # una tendencia cromatica que no hay — la dispersion bin a bin es la
        # degeneracion alpha-beta del ajuste, no seeing que cambie 0.15" entre
        # bins contiguos.
        w_m = moffat["wave_center_A"]
        fwhm = 0.5 * (moffat["fwhm_maj"] + moffat["fwhm_min"]) * pix
        axes[0].plot(w_m, fwhm, ls="none", color=col, marker="o", ms=1.8,
                     alpha=0.55, label=obj.nombre)
        axes[0].plot(w_m, _mediana_movil(fwhm, 7), color=col, lw=1.1)

        w_p, r0 = psfao["lambda_A"], psfao["r0"]
        ok = np.isfinite(r0)
        axes[1].plot(w_p[ok], r0[ok], color=col, lw=0.9, marker="o", ms=1.8,
                     label=obj.nombre)
        if ok.sum() > 2:
            j = int(np.flatnonzero(ok)[ok.sum() // 2])
            fried = r0[j] * (w_p / w_p[j]) ** 1.2
            axes[1].plot(w_p, fried, color="0.15", lw=0.7, ls=(0, (4, 2)),
                         alpha=0.9, zorder=3.5)

        axes[2].plot(w_m, moffat["ring_residual_pct"], color=col, lw=0.9, ls=":",
                     label=f"{obj.nombre} — Moffat")
        axes[2].plot(w_p, psfao["ring_residual_pct"], color=col, lw=0.9,
                     label=f"{obj.nombre} — AO ({forma})" if forma
                           else f"{obj.nombre} — AO")

    axes[0].set_ylabel("Moffat FWHM (arcsec)")
    axes[1].set_ylabel(r"$r_0$ (m)")
    axes[2].set_ylabel("ring residual (%)")
    axes[2].set_xlabel(r"Wavelength ($\mathrm{\AA}$)")
    axes[0].legend(loc="upper right", handlelength=1.3)
    axes[1].legend(handles=[*axes[1].get_legend_handles_labels()[0],
                            Line2D([], [], color="0.15", lw=0.7, ls=(0, (4, 2)))],
                   labels=[*axes[1].get_legend_handles_labels()[1],
                           r"Fried, $r_0\propto\lambda^{6/5}$"],
                   loc="upper left", handlelength=1.6)
    axes[2].legend(loc="upper right", ncol=1, handlelength=1.3)
    fig.align_ylabels(axes)
    fig.tight_layout(pad=0.3)
    fig.savefig(out)
    plt.close(fig)


# --------------------------------------------------------------------------

#: nombre del PDF -> funcion, en el orden en que aparecen en el paper.
FIGURAS = {
    "fov_redband": fov_redband,
    "spectra_halpha": spectra_halpha,
    "full_spectra": full_spectra,
    "hbeta_limit": hbeta_limit,
    "primary_variability": primary_variability,
    "companion_type": companion_type,
    "null_distributions": null_distributions,
    "injection_throughput": injection_throughput,
    "mdot_mass_plane": mdot_mass_plane,
    "contrast_curves": contrast_curves,
    "psf_chromatic": psf_chromatic,
}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--outdir", default=str(ROOT / "paper" / "figures"),
                    help="donde se escriben los PDF (por defecto paper/figures)")
    ap.add_argument("--figure", action="append", choices=sorted(FIGURAS),
                    help="dibuja solo esta figura (repetible); por defecto todas")
    ap.add_argument("--literature", default=str(ROOT / "paper" / "literature_mdot.csv"),
                    help="CSV de la compilacion de literatura para mdot_mass_plane")
    ap.add_argument("--list", action="store_true",
                    help="lista las figuras y el run de cada objeto, y sale")
    args = ap.parse_args(argv)

    objetos = objetos_del_paper()
    if args.list:
        for o in objetos:
            print(f"{o.orden}. {o.nombre:<12} run={o.run_id}")
        for n in FIGURAS:
            print(f"   figures/{n}.pdf")
        return 0

    estilo_aa()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    nombres = args.figure or list(FIGURAS)

    for nombre in nombres:
        destino = outdir / f"{nombre}.pdf"
        fn = FIGURAS[nombre]
        if nombre == "mdot_mass_plane":
            fn(objetos, destino, literatura=Path(args.literature))
        else:
            fn(objetos, destino)
        print(f"{nombre:<22} -> {destino}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
