"""La figura del espectro para publicación, y su tabla de datos.

Un espectro de MUSE de 3681 canales no se lee en una figura de 11×4: el rango
4750–9350 Å mete la banda A del O₂ y el racimo Ca II + Paschen en el mismo
ancho que Hβ. Aquí se dibuja **sin binar** —cada canal, con su σ— pero
partido en tramos, con las bandas telúricas sombreadas según lo que absorben
esa noche y las líneas de acreción marcadas por familia.

La otra mitad del módulo es `write_spectrum_table`: los mismos números que se
dibujan, en columnas, en un fichero que se puede mandar por correo. El formato
por defecto es **ECSV** (el estándar de astropy para tablas portables): texto
plano legible, con las unidades y la procedencia en la cabecera YAML, y se lee
con `Table.read` sin decirle nada. Una figura sin sus datos no es un resultado
citable.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .spectral import median_filter_1d
from .telluric_lines import (
    AO_LASER_WINDOW_A,
    SEVERITY_ALPHA,
    bands_with_measured_depth,
    measured_transmission,
    sky_emission_lines,
)

#: Colores por familia de línea; los comparte `stage07_accretion_lines`.
FAMILY_COLORS = {
    "Balmer": "tab:red",
    "He I": "tab:purple",
    "forbidden": "tab:green",
    "O I": "tab:olive",
    "Ca II IRT": "tab:orange",
    "Paschen": "tab:blue",
}

#: Especies telúricas y su color de sombreado.
SPECIES_COLORS = {"O2": "tab:orange", "H2O": "tab:cyan"}


def accretion_lines():
    """El catálogo de líneas de acreción de la cadena (el de E-block/G2)."""
    from .stages.stage07_accretion_lines import default_accretion_lines

    return default_accretion_lines()


_SUPERINDICES = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")


def pretty_flux_unit(bunit):
    """El `BUNIT` del FITS escrito como se escribe en un paper.

    `10**(-20)*erg/s/cm**2/Angstrom` es válido y legible por máquina, pero en
    el eje de una figura se lee fatal. La unidad que se guarda en la tabla
    sigue siendo la original: esto es solo la etiqueta.
    """
    import re

    texto = str(bunit or "").strip()
    if not texto:
        return "sin unidad declarada"
    m = re.match(r"^10\*\*\((-?\d+)\)\s*\*\s*erg/s/cm\*\*2/Angstrom$", texto)
    if m:
        exp = m.group(1).translate(_SUPERINDICES)
        return f"10{exp} erg s⁻¹ cm⁻² Å⁻¹"
    return texto


def _panel_ranges(wave_A, n_panels):
    wave = np.asarray(wave_A, dtype=float)
    finite = wave[np.isfinite(wave)]
    lo, hi = float(finite.min()), float(finite.max())
    edges = np.linspace(lo, hi, int(n_panels) + 1)
    return [(edges[i], edges[i + 1]) for i in range(int(n_panels))]


def _robust_ylim(flux, flux_err, inside, pad=1.25):
    """Límites que enseñen el continuo, no el pico de ruido del azul."""
    f = np.asarray(flux, dtype=float)[inside]
    good = np.isfinite(f)
    if not good.any():
        return None
    lo, hi = np.nanpercentile(f[good], [2.0, 98.0])
    if flux_err is not None:
        e = np.asarray(flux_err, dtype=float)[inside]
        e = e[np.isfinite(e)]
        if e.size:
            hi = max(hi, float(np.nanpercentile(e, 90.0)))
            lo = min(lo, -float(np.nanpercentile(e, 90.0)))
    span = max(hi - lo, 1e-30)
    mid = 0.5 * (hi + lo)
    return mid - pad * span / 2, mid + pad * span / 2


def _draw_lines(ax, lines, lo_A, hi_A, *, fontsize=6.0):
    """Marca las líneas del tramo, escalonando etiquetas para que se lean."""
    visible = [ln for ln in lines if lo_A <= float(ln["wave_A"]) <= hi_A]
    visible.sort(key=lambda ln: float(ln["wave_A"]))
    for i, line in enumerate(visible):
        color = FAMILY_COLORS.get(line.get("family"), "0.3")
        w = float(line["wave_A"])
        ax.axvline(w, color=color, ls=":", lw=0.7, alpha=0.85, zorder=2)
        # Dos alturas alternas: en el racimo Ca II + Paschen (8437–8750 Å) hay
        # diez lineas en 300 A y las etiquetas se pisarian a una sola altura.
        y = 0.98 if i % 2 == 0 else 0.72
        ax.annotate(line["name"], (w, y), xycoords=("data", "axes fraction"),
                    rotation=90, ha="right", va="top", fontsize=fontsize,
                    color=color, annotation_clip=True)
    return visible


def paper_spectrum_figure(
    wave_A,
    flux,
    flux_err=None,
    *,
    flux_err_alt=None,
    bad_channels=None,
    err_label="±1σ",
    err_alt_label=None,
    title=None,
    flux_label="flujo",
    n_panels=2,
    smooth_channels=41,
    lines=None,
    sky_lines=None,
    bands=None,
    transmission=None,
    show_transmission=True,
    mask_windows_A=(AO_LASER_WINDOW_A,),
    mask_label="ventana del láser AO (sin dato)",
    ylim=None,
    width=9.5,
    panel_height=3.0,
):
    """Dibuja el espectro **canal a canal** con su error, en `n_panels` tramos.

    `flux_err` es la barra que manda (la empírica, si la hay) y se dibuja como
    banda; `flux_err_alt` permite superponer la otra estimación —típicamente la
    propagada del STAT— para que se vea la diferencia sin tener que creerse un
    número. `transmission` es la curva medida de A3 (`measured_transmission`):
    si se pasa, va en un eje gemelo, que es la única forma honesta de decir
    "aquí la atmósfera se comió el 70%".
    """
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    wave = np.asarray(wave_A, dtype=float)
    flux = np.asarray(flux, dtype=float)
    err = None if flux_err is None else np.asarray(flux_err, dtype=float)
    err_alt = None if flux_err_alt is None else np.asarray(flux_err_alt, dtype=float)
    # Donde no hay dato no hay ni flujo ni barra de error. En el hueco del
    # láser AO los canales valen 0, no NaN, así que sin este recorte la curva
    # de error bajaba a cero cruzándolo — como si allí la medida fuera
    # exquisita. `bad_channels` es la máscara de flags de la propia etapa.
    sin_dato = ~np.isfinite(flux)
    if bad_channels is not None:
        sin_dato = sin_dato | np.asarray(bad_channels, dtype=bool)
        flux = np.where(sin_dato, np.nan, flux)
    if err is not None:
        err = np.where(sin_dato, np.nan, err)
    if err_alt is not None:
        err_alt = np.where(sin_dato, np.nan, err_alt)
    if lines is None:
        lines = accretion_lines()
    if sky_lines is None:
        sky_lines = sky_emission_lines()
    band_rows = (bands_with_measured_depth(transmission) if bands is None
                 else bands_with_measured_depth(transmission, bands))
    smooth = median_filter_1d(flux, smooth_channels) if smooth_channels else None

    # Cada tramo son DOS ejes: una tira fina arriba con la transmisión y las
    # bandas etiquetadas, y debajo el espectro. Meter la transmisión en un eje
    # gemelo sobre el propio espectro cruzaba una curva plana en 1.0 por mitad
    # del dato y se leía como si fuera flujo.
    n_panels = int(n_panels)
    con_tira = transmission is not None and show_transmission
    fig = plt.figure(figsize=(width, panel_height * n_panels))
    if con_tira:
        gs = GridSpec(2 * n_panels, 1, figure=fig, height_ratios=[0.26, 1] * n_panels,
                      hspace=0.06)
    else:
        gs = GridSpec(n_panels, 1, figure=fig, hspace=0.28)
    axes, strips = [], []
    for i in range(n_panels):
        if con_tira:
            strips.append(fig.add_subplot(gs[2 * i]))
            axes.append(fig.add_subplot(gs[2 * i + 1], sharex=strips[-1]))
        else:
            strips.append(None)
            axes.append(fig.add_subplot(gs[i]))
    axes = np.array(axes, dtype=object)

    for ax, strip, (lo_A, hi_A) in zip(axes, strips, _panel_ranges(wave, n_panels)):
        inside = (wave >= lo_A) & (wave <= hi_A)

        for j, (mlo, mhi) in enumerate(mask_windows_A or ()):
            for target in (ax, strip):
                if target is not None:
                    target.axvspan(mlo, mhi, color="0.88", zorder=0,
                                   label=mask_label if (j == 0 and target is axes[0]) else None)

        for band in band_rows:
            if band["hi_A"] < lo_A or band["lo_A"] > hi_A:
                continue
            color = SPECIES_COLORS.get(band["species"], "tab:orange")
            alpha = SEVERITY_ALPHA.get(band["severity"], 0.1)
            ax.axvspan(band["lo_A"], band["hi_A"], color=color, zorder=0, alpha=alpha)
            etiqueta = band["name"]
            if band.get("t_min") is not None:
                etiqueta += f" (T={band['t_min']:.2f})"
            centro = 0.5 * (band["lo_A"] + band["hi_A"])
            if strip is not None:
                strip.axvspan(band["lo_A"], band["hi_A"], color=color, zorder=0, alpha=alpha)
            # La etiqueta va al pie del espectro, no encima de la tira: la tira
            # es demasiado baja y el texto se salia del recorte de la figura.
            ax.annotate(etiqueta, (centro, 0.015), xycoords=("data", "axes fraction"),
                        ha="center", va="bottom", fontsize=5.5, color=color,
                        annotation_clip=True)

        if strip is not None:
            strip.plot(transmission["wave_A"], transmission["transmission"],
                       lw=0.7, color="tab:green")
            strip.set_ylim(-0.05, 1.15)
            strip.set_yticks([0.0, 0.5, 1.0])
            strip.set_ylabel("T", fontsize=7, color="tab:green")
            strip.tick_params(axis="y", labelcolor="tab:green", labelsize=6)
            strip.tick_params(axis="x", labelbottom=False, length=0)
            strip.set_xlim(lo_A, hi_A)

        if err is not None:
            ax.fill_between(wave, -err, err, color="0.82", lw=0, zorder=1,
                            label=err_label if ax is axes[0] else None)
        if err_alt is not None:
            ax.plot(wave, err_alt, lw=0.6, color="tab:red", alpha=0.7, zorder=3,
                    label=err_alt_label if ax is axes[0] else None)
            ax.plot(wave, -err_alt, lw=0.6, color="tab:red", alpha=0.7, zorder=3)
        ax.plot(wave, flux, lw=0.35, color="0.45", zorder=4,
                label="flujo por canal (sin binar)" if ax is axes[0] else None)
        if smooth is not None:
            ax.plot(wave, smooth, lw=1.1, color="tab:blue", zorder=5,
                    label=f"mediana móvil {smooth_channels} ch"
                          if ax is axes[0] else None)
        ax.axhline(0.0, color="0.4", lw=0.6, zorder=1)

        for j, sky in enumerate(sky_lines or ()):
            w_sky = float(sky["wave_A"])
            if lo_A <= w_sky <= hi_A:
                ax.axvline(w_sky, color="0.55", ls="--", lw=0.5, alpha=0.7, zorder=1,
                           label=("emisión de cielo (OI/OH)"
                                  if (ax is axes[0] and j == 0) else None))

        _draw_lines(ax, lines, lo_A, hi_A)

        ax.set_xlim(lo_A, hi_A)
        limites = ylim if ylim is not None else _robust_ylim(flux, err, inside)
        if limites is not None:
            ax.set_ylim(*limites)
        ax.set_ylabel(flux_label, fontsize=8)
        ax.tick_params(labelsize=8)
        ax.minorticks_on()
    if not con_tira:
        # Que la falta de la tira se explique EN la figura: quien mire el PDF no
        # tiene el stdout del notebook, y si no se dice parece un fallo del plot.
        axes[0].annotate(
            "sin curva de transmisión medida en este run:\n"
            "se marcan las bandas del catálogo, sin su profundidad",
            (0.995, 1.02), xycoords="axes fraction", ha="right", va="bottom",
            fontsize=6, color="tab:green")
    axes[-1].set_xlabel("λ [Å]  (aire)", fontsize=9)
    # Leyenda a nivel de figura: dentro del eje tapaba justo el extremo azul,
    # que es donde el continuo se va a negativo y hay que poder verlo.
    manejadores, etiquetas = axes[0].get_legend_handles_labels()
    fig.legend(manejadores, etiquetas, fontsize=6.5, ncol=min(len(etiquetas), 3),
               loc="lower center", frameon=False, bbox_to_anchor=(0.5, 0.004),
               columnspacing=1.4, handlelength=1.8)
    if title:
        fig.suptitle(title, fontsize=10)
    # Márgenes en PULGADAS, no `tight_layout`: con la rejilla tira+espectro
    # matplotlib avisa de que no puede ajustarla, y una figura de paper tiene
    # que salir igual sea cual sea el número de tramos.
    alto = fig.get_figheight()
    fig.subplots_adjust(left=0.10, right=0.985,
                        bottom=0.95 / alto, top=1.0 - (0.42 if title else 0.12) / alto)
    return fig, axes


def write_spectrum_table(path, wave_A, flux, flux_err=None, *, meta=None,
                         extra_columns=None, units=None):
    """Escribe el espectro en columnas, en un formato portable.

    Por defecto **ECSV**: texto plano con las unidades y `meta` en la cabecera,
    que es el formato que astropy lee sin configuración y que cualquier otro
    lenguaje puede parsear como CSV con comentarios. Por el sufijo también
    acepta `.fits` (tabla binaria) y `.csv` (sin unidades — se avisa en `meta`
    pero un CSV pelado las pierde).

    Devuelve la ruta escrita.
    """
    from astropy.table import Table

    path = Path(path)
    columns = {"wave_A": np.asarray(wave_A, dtype=float),
               "flux": np.asarray(flux, dtype=float)}
    if flux_err is not None:
        columns["flux_err"] = np.asarray(flux_err, dtype=float)
    for name, values in (extra_columns or {}).items():
        columns[name] = np.asarray(values)

    table = Table(columns)
    unidades = {"wave_A": "Angstrom"}
    unidades.update(units or {})
    for name, unit in unidades.items():
        if name in table.colnames and unit:
            table[name].unit = unit
    table.meta.update(meta or {})

    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    if suffix == ".fits":
        table.write(path, overwrite=True)
    elif suffix == ".csv":
        table.write(path, format="ascii.csv", overwrite=True)
    else:
        table.write(path, format="ascii.ecsv", overwrite=True)
    return path


def spectrum_table_meta(*, run_id, target, method, product, flux_unit,
                        error_mode=None, extra=None):
    """La procedencia mínima que tiene que viajar con la tabla."""
    meta = {
        "run_id": str(run_id),
        "target": str(target),
        "method": str(method),
        "product": str(product),
        "flux_unit": str(flux_unit),
        "wavelength_frame": "air",
        "note": ("Errores empíricos de controles procesados igual que el objeto; "
                 "el STAT del cubo NO es sigma (docs/noise_model.md). Canales "
                 "correlacionados: n_eff/n = 0.43."),
    }
    if error_mode:
        meta["error_mode"] = str(error_mode)
    meta.update(extra or {})
    return meta
