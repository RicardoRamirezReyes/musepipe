"""Figuras del informe «El error de PSF».

Reune en un sitio lo que hoy esta repartido entre los CSV por bin de C1, los QC de
C1/C2/D1, la medida de la apcorr empirica del 2026-09-10 y la medida de borrado
espectral de SGF/LPM.

Cada figura se dibuja SOLO si estan sus entradas; si falta una, se anota y se sigue,
en vez de reventar a mitad del lote.

Uso:
    python scripts/informe_psf_figuras.py --runs ROXs12b_realigned,ROXs42Bb_realigned \\
        --out-dir reports/20260910/psf [--b-json ...] [--halosub-json ...]
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from musepipe.extraction.aperture import aperture_correction_from_psf  # noqa: E402
from musepipe.io import read_json  # noqa: E402
from musepipe.psf import evaluate_psf_model  # noqa: E402
from musepipe.targets import (  # noqa: E402
    load_target, run_display_name, target_of_run)

BANDS = {"B1": (4900, 5400), "B2": (5450, 5750), "B3": (6100, 6400),
         "B4": (6600, 6800), "B5": (7600, 8000), "B6": (8600, 9100)}
BOX3 = {"kind": "box", "size": 3, "name": "box3"}
BOX5 = {"kind": "box", "size": 5, "name": "box5"}
#: un color por objeto, asignado por el ORDEN en que llegan los runs, igual que
#: `paper_figures.COLOR_OBJETO`. Los nombres de objeto NUNCA van como literal aqui:
#: salen de `targets/<slug>.json` via `run_display_name` (`test_no_hardcoded_target`).
COLOR_OBJETO = ("#1f77b4", "#d62728", "#2ca02c", "#9467bd")
COLORS: dict[str, str] = {}
LABEL: dict[str, str] = {}


def registrar_runs(runs):
    """Fija color y nombre para mostrar de cada run, sin literales de objeto.

    Se usa `paper.display` y NO `display_name`: el segundo es el nombre del registro
    y en un objeto es el de la PRIMARIA («ROXs 12 B»), mientras que estas figuras van
    del COMPAÑERO («ROXs 12 b»). Misma convencion que `paper_figures.Objeto.nombre`.
    """

    for i, run in enumerate(runs):
        COLORS[run] = COLOR_OBJETO[i % len(COLOR_OBJETO)]
        slug = target_of_run(run, project_root=ROOT)
        ficha = load_target(slug, project_root=ROOT) if slug else None
        bloque = (ficha or {}).get("paper") or {}
        LABEL[run] = (bloque.get("display")
                      or run_display_name(run, project_root=ROOT))


def estilo():
    """Como `paper_figures.estilo_aa` pero a tamaño de pantalla, no de A&A."""
    plt.rcParams.update({
        "font.size": 10, "axes.labelsize": 10, "axes.titlesize": 11,
        "legend.fontsize": 8.5, "xtick.labelsize": 9, "ytick.labelsize": 9,
        "axes.linewidth": 0.8, "xtick.direction": "in", "ytick.direction": "in",
        "xtick.top": True, "ytick.right": True, "lines.linewidth": 1.3,
        "legend.frameon": False, "figure.dpi": 130,
        "savefig.bbox": "tight", "savefig.pad_inches": 0.05,
    })


def _csv_rows(path):
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def _f(row, key):
    try:
        v = float(row[key])
        return v if np.isfinite(v) else np.nan
    except (KeyError, TypeError, ValueError):
        return np.nan


def _col(rows, key):
    return np.array([_f(r, key) for r in rows])


def _band_of(lam):
    for name, (lo, hi) in BANDS.items():
        if lo <= lam <= hi:
            return name
    return None


def _shade_bands(ax, only=("B5", "B6")):
    for name in only:
        lo, hi = BANDS[name]
        ax.axvspan(lo, hi, color="0.85", zorder=0)
        ax.text(0.5 * (lo + hi), ax.get_ylim()[1], name, ha="center", va="top",
                fontsize=7, color="0.35")


class Report:
    def __init__(self, out_dir):
        self.out = Path(out_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        self.figures = []
        self.skipped = []

    def save(self, fig, name, caption):
        p = self.out / f"{name}.png"
        fig.savefig(p)
        plt.close(fig)
        self.figures.append({"name": name, "file": p.name, "caption": caption})
        print(f"  [ok]   {name}.png")

    def skip(self, name, why):
        self.skipped.append({"name": name, "why": why})
        print(f"  [skip] {name}: {why}")


# ---------------------------------------------------------------- figuras

def fig_psfao_params(rep, data):
    keys = [("r0", "$r_0$ (m)"), ("C", "C"), ("A", "A"), ("alpha", r"$\alpha$"),
            ("ratio", "ratio"), ("theta", r"$\theta$"), ("beta", r"$\beta$")]
    fig, axes = plt.subplots(4, 2, figsize=(9.5, 10.5), sharex=True)
    for ax, (k, lab) in zip(axes.ravel(), keys):
        for run, d in data.items():
            rows = d.get("psfao_rows")
            if not rows:
                continue
            lam, v = _col(rows, "lambda_A"), _col(rows, k)
            e = _col(rows, f"{k}_err")
            ax.errorbar(lam, v, yerr=np.where(np.isfinite(e), e, np.nan), fmt="o",
                        ms=2.6, lw=0.7, capsize=0, color=COLORS[run], label=LABEL[run])
        ax.set_ylabel(lab)
        ax.grid(alpha=0.25, lw=0.4)
    axes.ravel()[-1].axis("off")
    axes.ravel()[0].legend(loc="best")
    for ax in axes[-1]:
        ax.set_xlabel(r"$\lambda$ ($\AA$)")
    axes[2, 1].set_xlabel(r"$\lambda$ ($\AA$)")
    fig.suptitle("C1: los 7 parametros de Psfao por bin, con sus errores", y=0.995)
    rep.save(fig, "01_psfao_parametros",
             "Los siete parametros del ajuste Psfao en los 43 bins de 100 A, con los "
             "errores del jacobiano. Un parametro que salta entre bins vecinos es una "
             "degeneracion del ajuste, no cromatismo.")


def fig_r0_fried(rep, data):
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    any_ok = False
    for run, d in data.items():
        rows = d.get("psfao_rows")
        if not rows:
            continue
        lam, r0 = _col(rows, "lambda_A"), _col(rows, "r0")
        m = np.isfinite(lam) & np.isfinite(r0)
        if m.sum() < 5:
            continue
        any_ok = True
        ax.plot(lam[m], r0[m], "o", ms=3.2, color=COLORS[run], label=f"{LABEL[run]} medido")
        # ley de Fried: r0 propto lambda^(6/5), anclada a la mediana
        ref = np.nanmedian(r0[m])
        lam0 = np.nanmedian(lam[m])
        ax.plot(lam[m], ref * (lam[m] / lam0) ** 1.2, "--", lw=1.0,
                color=COLORS[run], alpha=0.75,
                label=r"$\lambda^{6/5}$ (Fried)" if run == list(data)[0] else None)
    if not any_ok:
        plt.close(fig)
        return rep.skip("02_r0_fried", "sin filas psfao")
    ax.set_xlabel(r"$\lambda$ ($\AA$)")
    ax.set_ylabel(r"$r_0$ (m)")
    ax.set_title(r"$r_0$ sigue la ley de Fried; el resto de la estructura en $\lambda$ es ruido")
    ax.legend()
    ax.grid(alpha=0.25, lw=0.4)
    rep.save(fig, "02_r0_fried",
             "El parametro de Fried contra la ley $r_0\\propto\\lambda^{6/5}$ anclada a la "
             "mediana de cada objeto. Es el unico de los siete con cromatismo real: la "
             "medida del 2026-08-09 que veia estructura en los demas usaba un "
             "discriminante mal planteado.")


def fig_ring(rep, data):
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    for run, d in data.items():
        for tag, key, style in (("Psfao", "psfao_rows", "-o"), ("Moffat", "moffat_rows", "--s")):
            rows = d.get(key)
            if not rows:
                continue
            lam = _col(rows, "lambda_A")
            col = ("ring_residual_pct_canonical" if any("ring_residual_pct_canonical" in r for r in rows)
                   else "ring_residual_pct")
            v = _col(rows, col)
            m = np.isfinite(lam) & np.isfinite(v)
            if m.sum() < 3:
                continue
            ax.plot(lam[m], v[m], style, ms=2.6, lw=0.9, alpha=0.9,
                    color=COLORS[run],
                    ls="-" if tag == "Psfao" else "--",
                    label=f"{LABEL[run]} {tag}")
    ax.axhline(5.0, color="k", lw=0.9, ls=":", label="umbral 5 %")
    ax.set_xlabel(r"$\lambda$ ($\AA$)")
    ax.set_ylabel("residuo de anillo (%)")
    ax.set_title("El residuo de anillo por bin: la metrica que gobierna C1, y es ciega al nucleo")
    ax.legend(ncol=2)
    ax.grid(alpha=0.25, lw=0.4)
    rep.save(fig, "03_residuo_anillo",
             "Residuo de anillo en el radio de la compañera, por bin y por forma de PSF. "
             "Es la metrica con la que C1 elige, y por construccion no mira el nucleo: de "
             "ahi que una forma pueda clavar el anillo con la energia encerrada del nucleo "
             "equivocada, que es lo que mide V4.")


def fig_v4_core_ratio(rep, data):
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(7.8, 6.4), sharex=True,
                                 gridspec_kw={"height_ratios": [2, 1.4]})
    ok = False
    for run, d in data.items():
        b = d.get("b_medida")
        if not b:
            continue
        ok = True
        per = b["per_bin"]
        lam = np.array([r["lambda_A"] for r in per])
        cd = np.array([r["core_ratio_data"] for r in per])
        cm = np.array([r["core_ratio_model"] for r in per])
        err = np.array([r["err_pct"] for r in per])
        a1.plot(lam, cd, "o-", ms=3, color=COLORS[run], label=f"{LABEL[run]} DATO")
        a1.plot(lam, cm, "s--", ms=3, color=COLORS[run], alpha=0.6,
                label=f"{LABEL[run]} MODELO")
        a2.plot(lam, err, "o-", ms=3, color=COLORS[run], label=LABEL[run])
    if not ok:
        plt.close(fig)
        return rep.skip("04_v4_core_ratio", "sin B_medida.json")
    a1.set_ylabel(r"$F(\leq 25\,\mathrm{px})\,/\,F(\mathrm{box3})$")
    a1.set_title("V4 de C1: el cociente que ES la correccion de apertura de C2/C3")
    a1.legend(ncol=2)
    a1.grid(alpha=0.25, lw=0.4)
    a2.axhline(0, color="k", lw=0.8)
    a2.axhspan(-3, 3, color="0.9", zorder=0, label="tolerancia $\\pm$3 %")
    a2.set_ylabel("error del modelo (%)")
    a2.set_xlabel(r"$\lambda$ ($\AA$)")
    a2.legend(ncol=3)
    a2.grid(alpha=0.25, lw=0.4)
    _shade_bands(a2)
    rep.save(fig, "04_v4_core_ratio",
             "Arriba, el cociente nucleo/norm medido sobre el DATO (sin modelo, en la "
             "primaria) y el del modelo que C2 invierte. Abajo, el error del modelo contra "
             "su tolerancia del 3 %: se sale, y se sale sobre todo en el rojo, que es donde "
             "vive la divergencia de D1 (B5 y B6 sombreadas).")


def fig_error_por_banda(rep, data):
    runs = [r for r in data if data[r].get("b_medida")]
    if not runs:
        return rep.skip("05_error_por_banda", "sin B_medida.json")
    names = list(BANDS)
    x = np.arange(len(names))
    w = 0.38
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    for i, run in enumerate(runs):
        by = data[run]["b_medida"]["by_band"]
        v = [by.get(n, {}).get("err_pct_median", np.nan) for n in names]
        ax.bar(x + (i - 0.5 * (len(runs) - 1)) * w, v, w, color=COLORS[run],
               label=LABEL[run], alpha=0.9)
    ax.axhspan(-3, 3, color="0.9", zorder=0, label="tolerancia $\\pm$3 %")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{n}\n{BANDS[n][0]}-{BANDS[n][1]}" for n in names], fontsize=8)
    ax.set_ylabel("error de la apcorr del modelo (%)")
    ax.set_title("El error del modelo se concentra en el rojo")
    ax.legend()
    ax.grid(alpha=0.25, lw=0.4, axis="y")
    rep.save(fig, "05_error_por_banda",
             "Mediana del error de la apcorr del modelo en cada banda de D1. En ROXs 12 b "
             "vale +2.2 % en B1 y +13.7 % en B5: no es un offset, es una pendiente, y por "
             "eso un escalar de throughput no lo puede arreglar.")


def fig_apcorr_curvas(rep, data):
    ok = False
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(7.8, 6.2), sharex=True,
                                 gridspec_kw={"height_ratios": [2, 1.2]})
    for run, d in data.items():
        b = d.get("b_medida")
        if not b:
            continue
        ok = True
        per = b["per_bin"]
        lam = np.array([r["lambda_A"] for r in per])
        cm = np.array([r["core_ratio_model"] for r in per])
        cd = np.array([r["core_ratio_data"] for r in per])
        ratio = np.array([r["ratio"] for r in per])
        a1.plot(lam, cm, "s--", ms=3, alpha=0.65, color=COLORS[run],
                label=f"{LABEL[run]} apcorr del MODELO")
        a1.plot(lam, cd, "o-", ms=3, color=COLORS[run],
                label=f"{LABEL[run]} apcorr del DATO")
        a2.plot(lam, ratio, "o-", ms=3, color=COLORS[run], label=LABEL[run])
    if not ok:
        plt.close(fig)
        return rep.skip("06_apcorr_curvas", "sin B_medida.json")
    a1.set_ylabel("apcorr (sin el factor de flujo total)")
    a1.set_title("La correccion de apertura: la del modelo contra la medida en el dato")
    a1.legend(ncol=2, fontsize=7.5)
    a1.grid(alpha=0.25, lw=0.4)
    a2.axhline(1.0, color="k", lw=0.8)
    a2.set_ylabel("dato / modelo")
    a2.set_xlabel(r"$\lambda$ ($\AA$)")
    a2.legend()
    a2.grid(alpha=0.25, lw=0.4)
    _shade_bands(a2)
    rep.save(fig, "06_apcorr_curvas",
             "La apcorr que C2 aplica sale del modelo (cuadrados); la medida directamente "
             "sobre la primaria, sin modelo, es la de circulos. El panel de abajo es el "
             "factor con el que habria que corregir cada canal.")


def fig_fase_subpixel(rep, data, waves=(5000., 6563., 8800.)):
    """El 10.4 %: apcorr de box3 contra el desplazamiento subpixel de la fuente."""
    run = next((r for r in data if data[r].get("psf_model")), None)
    if run is None:
        return rep.skip("07_fase_subpixel", "sin psf_model.json")
    psf = data[run]["psf_model"]
    offs = np.linspace(-0.5, 0.5, 21)
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for w in waves:
        vals = []
        for dy in offs:
            a, _, _ = aperture_correction_from_psf(np.array([w]), BOX3, psf,
                                                   center_yx=(dy, 0.0), growth_curve=None)
            vals.append(float(a[0]))
        vals = np.array(vals)
        ax.plot(offs, vals / vals[len(offs) // 2], "o-", ms=3,
                label=rf"$\lambda={w:.0f}\,\AA$")
    ax.axhline(1.0, color="k", lw=0.8)
    ax.set_xlabel("desplazamiento subpixel en y (px)")
    ax.set_ylabel("apcorr(box3) relativa al pixel centrado")
    ax.set_title("box3 es fragil a la fase subpixel: la mediana se mueve un 10.4 %")
    ax.legend()
    ax.grid(alpha=0.25, lw=0.4)
    rep.save(fig, "07_fase_subpixel",
             "La apcorr de box3 en funcion de donde cae la fuente dentro del pixel. Con "
             "FWHM ~3 px una caja de 3x3 captura una fraccion que depende mucho de la fase: "
             "entre la primaria y la compañera de ROXs 12 b la diferencia es del 10.4 % de "
             "mediana. No es un error -C2 usa la fase correcta- pero dice que el metodo es "
             "sensible a un error de posicion.")


def fig_box3_box5(rep, data):
    """La V4(b) de C2, antes y despues de la apcorr empirica."""
    fig, ax = plt.subplots(figsize=(7.8, 4.6))
    ok = False
    for run, d in data.items():
        for tag, key, ls in (("modelo", "spec_box3", "--"), ("dato", "spec_box3_corr", "-")):
            pair = d.get(key)
            if not pair:
                continue
            w, f3, f5 = pair
            m = np.isfinite(f3) & (np.abs(f3) > 0)
            if m.sum() < 50:
                continue
            ok = True
            # binado grueso para que se lea
            nb = 60
            idx = np.linspace(0, m.sum() - 1, nb).astype(int)
            ww = w[m][idx]
            rr = (f5[m] / f3[m])[idx]
            ax.plot(ww, rr, ls, lw=1.2, color=COLORS[run], alpha=0.9 if ls == "-" else 0.55,
                    label=f"{LABEL[run]} apcorr del {tag}")
    if not ok:
        plt.close(fig)
        return rep.skip("08_box3_box5", "faltan los productos box3/box5")
    ax.axhline(1.0, color="k", lw=0.9, label="lo que deberia ser")
    ax.axvspan(7500, 9000, color="0.88", zorder=0)
    ax.text(8250, ax.get_ylim()[1], "banda de la V4(b)", ha="center", va="top",
            fontsize=7.5, color="0.35")
    ax.set_ylim(0.4, 1.4)
    ax.set_xlabel(r"$\lambda$ ($\AA$)")
    ax.set_ylabel("box5 / box3, ya corregidos")
    ax.set_title("V4(b) de C2: dos aperturas distintas deberian dar el mismo espectro")
    ax.legend(ncol=2, fontsize=7.5)
    ax.grid(alpha=0.25, lw=0.4)
    rep.save(fig, "08_box3_box5",
             "Si la curva de crecimiento fuera correcta, el espectro corregido de box3 y el "
             "de box5 coincidirian y este cociente seria 1. Vale 0.90 en la banda donde C2 "
             "lo mide; con la apcorr medida en el dato sube a 0.93, o sea que la curva "
             "empirica de la primaria transfiere solo un cuarto de la inconsistencia. Fuera "
             "de 7500-9000 A el cociente no significa nada porque el continuo de la "
             "compañera no esta detectado.")


def fig_snr_por_banda(rep, data):
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    names = list(BANDS)
    x = np.arange(len(names))
    w = 0.38
    ok = False
    for i, (run, d) in enumerate(data.items()):
        snr = d.get("snr_by_band")
        if not snr:
            continue
        ok = True
        ax.bar(x + (i - 0.5) * w, [snr.get(n, np.nan) for n in names], w,
               color=COLORS[run], label=LABEL[run], alpha=0.9)
    if not ok:
        plt.close(fig)
        return rep.skip("09_snr_por_banda", "sin espectro de apertura")
    ax.axhline(0, color="k", lw=0.8)
    ax.axhline(3, color="k", lw=0.8, ls=":", label="S/N = 3")
    ax.set_yscale("symlog", linthresh=1)
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("S/N mediana del continuo (box3)")
    ax.set_title("El continuo de la compañera solo esta detectado en B5 y B6")
    ax.legend()
    ax.grid(alpha=0.25, lw=0.4, axis="y")
    rep.save(fig, "09_snr_por_banda",
             "S/N mediana del continuo de la compañera en cada banda de D1. En las cuatro "
             "azules es ~0, asi que su |t| pequeño no es «los metodos concuerdan» sino que "
             "no hay señal, y cualquier cociente construido alli es un cociente de ruidos. "
             "La divergencia de continuo de D1 vive en las dos unicas bandas con continuo.")


def fig_t_matrix(rep, tm_pre, tm_post, pairs):
    if not tm_pre or not tm_post:
        return rep.skip("10_t_matrix", "faltan las matrices t de D1")
    names = list(BANDS)
    pre = np.array([[tm_pre[p].get(b, np.nan) for b in names] for p in pairs])
    post = np.array([[tm_post[p].get(b, np.nan) for b in names] for p in pairs])
    vmax = np.nanmax(np.abs(np.concatenate([pre, post])))
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), sharey=True)
    for ax, M, tt in zip(axes, (pre, post), ("apcorr del MODELO", "apcorr del DATO")):
        im = ax.imshow(M, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names)
        ax.set_title(tt)
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                if np.isfinite(M[i, j]):
                    ax.text(j, i, f"{M[i, j]:.1f}", ha="center", va="center", fontsize=7,
                            color="white" if abs(M[i, j]) > 0.55 * vmax else "black")
    axes[0].set_yticks(range(len(pairs)))
    axes[0].set_yticklabels([p.replace("_vs_", " vs ") for p in pairs], fontsize=8)
    fig.colorbar(im, ax=axes, label="t de D1", fraction=0.03)
    fig.suptitle("D1 en ROXs 12 b: corregir la apertura mueve el rojo y no cierra ningun par",
                 y=1.02)
    rep.save(fig, "10_t_matrix",
             "Estadistico t por par y banda, antes y despues de sustituir la apcorr del "
             "modelo por la medida en el dato. B6 de psffit-vs-aperture pasa de -18.7 a "
             "-3.7 y el de optimal_ls-vs-aperture de -65.4 a -37.2; psffit-vs-optimal_ls no "
             "se mueve ni un digito, porque ese par no comparte la apertura. Ningun par "
             "cambia de veredicto.")


def fig_dos_modelos(rep, data):
    """El 7.5 %: descomponer si lo explica el tamaño de la rejilla."""
    run = next((r for r in data if data[r].get("psf_model")), None)
    if run is None:
        return rep.skip("11_dos_modelos", "sin psf_model.json")
    psf = data[run]["psf_model"]
    norm = float(psf.get("norm_radius_px", 25.0))
    waves = np.linspace(5000, 9000, 9)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.2, 4.3))
    for half, lab in ((int(np.ceil(norm)), "51x51 (lo que evalua C2)"),
                      (84, "169x169 (la rejilla de C1)")):
        ratios = []
        for w in waves:
            yy, xx = np.indices((2 * half + 1, 2 * half + 1), dtype=float)
            dy, dx = yy - half, xx - half
            img = evaluate_psf_model(psf, float(w), dy, dx)
            r = np.hypot(dy, dx)
            tot = float(np.nansum(img[r <= norm]))
            box = float(np.nansum(img[(np.abs(dy) <= 1) & (np.abs(dx) <= 1)]))
            ratios.append(tot / box if box > 0 else np.nan)
        a1.plot(waves, ratios, "o-", ms=3.5, label=lab)
    a1.set_xlabel(r"$\lambda$ ($\AA$)")
    a1.set_ylabel(r"$F(\leq 25)/F(\mathrm{box3})$ del modelo")
    a1.set_title("¿Explica el tamaño de la rejilla el 7.5 %?")
    a1.legend(fontsize=8)
    a1.grid(alpha=0.25, lw=0.4)
    half = int(np.ceil(norm))
    yy, xx = np.indices((2 * half + 1, 2 * half + 1), dtype=float)
    dy, dx = yy - half, xx - half
    r = np.hypot(dy, dx)
    for w, c in zip((5000., 6563., 8800.), ("#1b6ca8", "#4a7c1f", "#c1440e")):
        img = evaluate_psf_model(psf, float(w), dy, dx)
        rb = np.arange(0, norm + 1, 1.0)
        prof = [float(np.nanmean(img[(r >= a) & (r < a + 1)])) for a in rb[:-1]]
        a2.semilogy(rb[:-1] + 0.5, prof, color=c, label=rf"$\lambda={w:.0f}$")
    a2.set_xlabel("radio (px)")
    a2.set_ylabel("perfil radial del modelo")
    a2.set_title(f"{LABEL[run]}: perfil del `psf_model` publicado")
    a2.legend(fontsize=8)
    a2.grid(alpha=0.25, lw=0.4)
    rep.save(fig, "11_dos_modelos",
             "C1 mide V4 sobre la reconstruccion Psfao en la rejilla del cubo (170x170) y "
             "obtiene 5.128; C2 evalua el `psf_model` PUBLICADO en 51x51 y obtiene 5.510, "
             "un 7.5 % mas. El panel izquierdo aisla cuanto de esa diferencia es solo el "
             "tamaño de la rejilla: si las dos curvas coinciden, la diferencia esta en el "
             "modelo, no en el soporte, y entonces C1 audita un objeto distinto del que C2 "
             "consume.")


def fig_halosub(rep, halosub):
    """La descomposicion: la linea sobrevive, el continuo fabrica una absorcion."""

    if not halosub:
        return rep.skip("12_halosub_borrado", "sin la medida de borrado espectral")
    import statistics as st

    def serie(payload, method, mode, campo):
        rows = [r for r in payload["rows"]
                if r["method"] == method and r.get("mode") == mode]
        seps = sorted({r["sep_arcsec"] for r in rows})
        out = []
        for s in seps:
            sub = [r for r in rows if r["sep_arcsec"] == s]
            vals = [campo(r) for r in sub]
            vals = [v for v in vals if v is not None and np.isfinite(v)]
            out.append((s, st.mean(vals) if vals else np.nan,
                        st.pstdev(vals) if len(vals) > 1 else 0.0))
        return out

    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(14.4, 4.5))
    estilos = {"sgf": ("o", "-"), "lpm": ("s", "--"), "none": ("^", ":")}
    for run, payload in halosub.items():
        real = payload["real_sep_arcsec"]
        linea_real = payload["anchor"]["line_real"]
        for method, (mk, ls) in estilos.items():
            # panel 1: linea INYECTADA SOLA -> ¿sobrevive?
            d = serie(payload, method, "line_only",
                      lambda r: r["line_retained_frac_integrated"])
            if d:
                x, y, e = zip(*d)
                a1.errorbar(x, y, yerr=e, fmt=mk + ls, ms=4.5, lw=1.2, capsize=2,
                            color=COLORS[run], alpha=0.55 if method == "none" else 1.0,
                            label=f"{LABEL[run]} {method}")
            # panel 2: continuo SOLO -> linea espuria, en unidades del Halpha real
            d = serie(payload, method, "cont_only",
                      lambda r: r["recovered_line_box3"]["integrated"] / linea_real)
            if d and method != "none":
                x, y, e = zip(*d)
                a2.errorbar(x, y, yerr=e, fmt=mk + ls, ms=4.5, lw=1.2, capsize=2,
                            color=COLORS[run], label=f"{LABEL[run]} {method}")
            # panel 3: el continuo
            d = serie(payload, method, "line+cont", lambda r: r["cont_retained_frac"])
            if d:
                x, y, e = zip(*d)
                a3.plot(x, y, mk + ls, ms=4.5, lw=1.2, color=COLORS[run],
                        alpha=0.55 if method == "none" else 1.0,
                        label=f"{LABEL[run]} {method}")
        for ax in (a1, a2, a3):
            ax.axvline(real, color=COLORS[run], lw=0.9, ls="-.", alpha=0.6)

    a1.axhline(1.0, color="k", lw=0.9)
    a1.set_ylim(0.9, 1.05)
    a1.set_ylabel("linea retenida (inyectada SOLA)")
    a1.set_title("La linea SOBREVIVE")
    a1.legend(ncol=2, fontsize=6.8)
    a2.axhline(0.0, color="k", lw=0.9)
    a2.set_ylabel(r"linea espuria / H$\alpha$ real del objeto")
    a2.set_title("El CONTINUO fabrica una absorcion")
    a2.legend(fontsize=7)
    a3.axhline(1.0, color="k", lw=0.9)
    a3.axhline(0.0, color="k", lw=0.6, ls=":")
    a3.set_ylabel("continuo retenido")
    a3.set_title("El continuo se anula, por diseño")
    a3.legend(ncol=2, fontsize=6.8)
    for ax in (a1, a2, a3):
        ax.set_xlabel("separacion (arcsec)")
        ax.grid(alpha=0.25, lw=0.4)
    rep.save(fig, "12_halosub_borrado",
             "Que le hacen SGF y LPM al espectro de un compañero sintetico, medido por "
             "diferencia (cubo limpio contra cubo inyectado, misma realizacion de ruido, "
             "asi que la medida no tiene ruido de fotones). Inyectando la LINEA SOLA "
             "sobrevive entera (0.98 SGF, 1.00 LPM) en los dos objetos y a todas las "
             "separaciones: NO la borran. Inyectando SOLO CONTINUO, sin ninguna linea, "
             "aparece una absorcion espuria de varias veces el Halpha real del objeto: la "
             "FABRICAN. El mecanismo es la Halpha de la primaria impresa en el halo, que "
             "el filtro paso-bajo no sigue. Las verticales marcan la separacion real de "
             "cada objeto; `none` es el control sin sustraccion, que da 1.000 exacto.")


# ---------------------------------------------------------------- carga

def load_run(run, b_json_by_run, root=ROOT):
    st = root / "runs" / run / "stages"
    d = {"run": run}
    for key, name in (("psfao_rows", "stage_e01_psfao_params.csv"),
                      ("moffat_rows", "stage_e01_psf_params.csv")):
        p = st / name
        d[key] = _csv_rows(p) if p.exists() else None
    for key, name in (("e01_qc", "stage_e01_qc.json"), ("psf_model", "psf_model.json"),
                      ("growth", "growth_curve_qc.json"),
                      ("aperture_qc", "spec_aperture_qc.json"),
                      ("x10_qc", "stage_x10_qc.json")):
        p = st / name
        d[key] = read_json(p) if p.exists() else None
    bj = b_json_by_run.get(run)
    d["b_medida"] = read_json(Path(bj)) if bj and Path(bj).exists() else None

    from astropy.io import fits
    def _spec(path):
        if not path.exists():
            return None
        with fits.open(path) as h:
            t = h[1].data
            return (np.asarray(t["wave_A"], float), np.asarray(t["flux"], float),
                    np.asarray(t["flux_err_emp"], float))
    s3 = _spec(st / "spec_aperture_object.fits")
    s5 = _spec(st / "spec_aperture_object_box5.fits")
    if s3 and s5:
        d["spec_box3"] = (s3[0], s3[1], s5[1])
        d["snr_by_band"] = {n: float(np.nanmedian(s3[1][(s3[0] >= lo) & (s3[0] <= hi)] /
                                                  s3[2][(s3[0] >= lo) & (s3[0] <= hi)]))
                            for n, (lo, hi) in BANDS.items()}
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--b-json", default="",
                    help="run=ruta,run=ruta con el JSON de la apcorr empirica")
    ap.add_argument("--halosub-json", default="",
                    help="run=ruta,run=ruta con la medida de borrado")
    ap.add_argument("--t-pre", default="", help="stage_x10_qc.json ANTES de la apcorr empirica")
    ap.add_argument("--t-post", default="", help="stage_x10_qc.json DESPUES")
    args = ap.parse_args()

    def _pairs(s):
        out = {}
        for chunk in s.split(","):
            if "=" in chunk:
                k, v = chunk.split("=", 1)
                out[k.strip()] = v.strip()
        return out

    estilo()
    rep = Report(args.out_dir)
    runs = [r.strip() for r in args.runs.split(",")]
    registrar_runs(runs)
    bmap = _pairs(args.b_json)
    hmap = _pairs(args.halosub_json)
    data = {r: load_run(r, bmap) for r in runs}
    halosub = {r: read_json(Path(p)) for r, p in hmap.items() if Path(p).exists()}

    print("figuras:")
    fig_psfao_params(rep, data)
    fig_r0_fried(rep, data)
    fig_ring(rep, data)
    fig_v4_core_ratio(rep, data)
    fig_error_por_banda(rep, data)
    fig_apcorr_curvas(rep, data)
    fig_fase_subpixel(rep, data)
    fig_box3_box5(rep, data)
    fig_snr_por_banda(rep, data)
    if args.t_pre and args.t_post:
        pre = read_json(Path(args.t_pre)) if Path(args.t_pre).exists() else None
        post = read_json(Path(args.t_post)) if Path(args.t_post).exists() else None
        if pre and post:
            fig_t_matrix(rep, pre["t_matrix"], post["t_matrix"],
                         pre["primary_pairs_continuum"])
        else:
            rep.skip("10_t_matrix", "no se pudieron leer los dos QC de D1")
    else:
        rep.skip("10_t_matrix", "no se pasaron --t-pre/--t-post")
    fig_dos_modelos(rep, data)
    fig_halosub(rep, halosub)

    manifest = {"figures": rep.figures, "skipped": rep.skipped,
                "runs": runs, "out_dir": str(rep.out)}
    (rep.out / "manifest.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False))
    print(f"\n{len(rep.figures)} figuras, {len(rep.skipped)} saltadas -> {rep.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
