"""Figuras del informe «La extincion».

La extincion es la segunda palanca del Mdot y nada nuestro la mide: la sensibilidad es
0.370 dex/mag, o sea un factor 2.3 por magnitud. Las cinco vias para acotarla estan
cerradas, cada una con numeros, y este informe las pone juntas con sus figuras.

Ancla obligatoria: la escalera Mdot-vs-A_V tiene que caer sobre el numero publicado a su
A_V adoptado, y si no cae, FALLA. Ese guardian es el que destapo el factor magnetosferico
x1.25 que E3 no aplicaba y G3 si (2026-09-04), y es mas barato que auditar.

Uso:
    python scripts/informe_extincion_figuras.py --runs <run>,<run> \\
        --out-dir reports/20260910/extincion --chi2-json <run>=<ruta> --chi2-npz <run>=<ruta>
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

from musepipe.io import read_json  # noqa: E402
from musepipe.models.extinction import CCMExtinction  # noqa: E402
from musepipe.targets import load_target, run_display_name, target_of_run  # noqa: E402

from musepipe.config import load_run_config  # noqa: E402
from musepipe.stages.stage_h03_limits import (  # noqa: E402
    physical_inputs_from_config)

sys.path.insert(0, str(ROOT / "scripts"))
from s7b_mdot_vs_extinction import _ladder_row  # noqa: E402

COLOR_OBJETO = ("#1f77b4", "#d62728", "#2ca02c", "#9467bd")
COLORS: dict[str, str] = {}
LABEL: dict[str, str] = {}
#: sensibilidad del Mdot a la extincion, docs/2026-09-04_lsf_corregida_y_extincion.md §4
DLOGMDOT_DAV = 0.370


def registrar_runs(runs):
    """Color por orden y nombre del COMPAÑERO (`paper.display`), sin literales."""
    for i, run in enumerate(runs):
        COLORS[run] = COLOR_OBJETO[i % len(COLOR_OBJETO)]
        slug = target_of_run(run, project_root=ROOT)
        ficha = load_target(slug, project_root=ROOT) if slug else None
        LABEL[run] = (((ficha or {}).get("paper") or {}).get("display")
                      or run_display_name(run, project_root=ROOT))


def estilo():
    plt.rcParams.update({
        "font.size": 10, "axes.labelsize": 10, "axes.titlesize": 11,
        "legend.fontsize": 8.5, "xtick.labelsize": 9, "ytick.labelsize": 9,
        "axes.linewidth": 0.8, "xtick.direction": "in", "ytick.direction": "in",
        "xtick.top": True, "ytick.right": True, "lines.linewidth": 1.3,
        "legend.frameon": False, "figure.dpi": 130,
        "savefig.bbox": "tight", "savefig.pad_inches": 0.05,
    })


class Report:
    def __init__(self, out_dir):
        self.out = Path(out_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        self.figures, self.skipped, self.anchors = [], [], []

    def save(self, fig, name, caption):
        fig.savefig(self.out / f"{name}.png")
        plt.close(fig)
        self.figures.append({"name": name, "file": f"{name}.png", "caption": caption})
        print(f"  [ok]   {name}.png")

    def skip(self, name, why):
        self.skipped.append({"name": name, "why": why})
        print(f"  [skip] {name}: {why}")

    #: causas conocidas de que un producto no cuadre, para que el ancla no solo
    #: diga «falla» sino POR QUE. La primera se midio el 2026-09-10.
    CAUSAS = ((1.25, 0.005, "el producto no lleva el factor magnetosferico x1.25 "
                            "(entro en a9f4297, 2026-09-04): ese E3 es anterior"),)

    def anchor(self, what, expected, got, tol):
        rel = abs(got / expected - 1.0) if expected else float("inf")
        ok = rel <= tol
        causa = None
        if not ok and expected:
            for factor, ftol, texto in self.CAUSAS:
                if abs(got / expected / factor - 1.0) <= ftol:
                    causa = texto
                    break
        self.anchors.append({"what": what, "expected": expected, "got": got,
                             "rel_diff": rel, "tol": tol, "ok": bool(ok),
                             "diagnosis": causa})
        print(f"  {'[ancla ok]' if ok else '[ANCLA FALLA]'} {what}: "
              f"esperado {expected:.4g}, figura {got:.4g} ({100*rel:.2f} %)"
              + (f"\n              -> {causa}" if causa else ""))
        return ok


def ladder_from_product(run, h03, av_grid):
    """La escalera calculada DESDE el producto E3, no leida de `tables/`.

    La tabla `mdot_limit_vs_extinction.csv` de un run puede quedarse por detras de su
    propio E3 -paso el 2026-09-10: la de ROXs 12 b era del 27 de julio y su E3 del 27 de
    agosto, con el flujo limite 10.35x fuera- y una figura que la lea hereda el desfase
    en silencio. Calculandola aqui desde `f_lim_observed` con la MISMA
    `_ladder_row` que usa el script oficial, el ancla no puede fallar por rancidez, solo
    por un error real de la cadena.
    """

    # `phys` sale de `physical_inputs_from_config`, que es de donde lo saca el script
    # oficial: el bloque `physical_inputs` del QC no lleva los coeficientes de la
    # relacion L_acc-L_Halpha, solo su cita.
    rc = load_run_config(run, project_root=str(ROOT))
    phys = physical_inputs_from_config(dict(rc.config))
    canon = h03.get("canonical_method")
    lim = next((L for L in h03["limits"] if L["method"] == canon), None)
    if lim is None:
        return None, None
    f_obs = float(lim["f_lim_observed"])
    return [_ladder_row(float(av), f_obs, phys) for av in av_grid], lim


def fig_escalera(rep, data):
    """Mdot vs A_V, anclada al producto: si no cae sobre el, falla."""
    fig, ax = plt.subplots(figsize=(7.8, 5.0))
    ok_any = False
    for run, d in data.items():
        h03 = d.get("h03")
        if not h03:
            continue
        rows, lim = ladder_from_product(run, h03, np.linspace(0.0, 5.0, 51))
        if not rows:
            continue
        ok_any = True
        av = np.array([float(r["a_v"]) for r in rows])
        for key, ls, tag in (("mdot_msun_yr", "-", "Alcala+17"),
                             ("mdot_aoyama21_msun_yr", "--", "Aoyama+21")):
            y = np.array([float(r[key]) for r in rows])
            ax.semilogy(av, y, ls, color=COLORS[run], lw=1.4,
                        label=f"{LABEL[run]} {tag}")
        # ancla: el valor publicado del metodo canonico a su A_V adoptado.
        # la clave del QC es `av`, no `a_v`
        av_ad = float((h03.get("physical_inputs") or {}).get("av", np.nan))
        if lim and np.isfinite(av_ad):
            for key, pub in (("mdot_msun_yr", lim.get("mdot")),
                             ("mdot_aoyama21_msun_yr", lim.get("mdot_aoyama21"))):
                if pub is None:
                    continue
                y = np.array([float(r[key]) for r in rows])
                got = float(np.interp(av_ad, av, y))
                rep.anchor(f"{LABEL[run]} {key} a A_V={av_ad:g}", float(pub), got, 0.02)
                ax.plot([av_ad], [pub], "o", ms=7, mfc="none", mew=1.6,
                        color=COLORS[run])
            ax.axvline(av_ad, color=COLORS[run], lw=0.8, ls=":", alpha=0.7)
    if not ok_any:
        plt.close(fig)
        return rep.skip("01_escalera_mdot_av", "sin producto E3")
    ax.set_xlabel(r"$A_V$ (mag)")
    ax.set_ylabel(r"$\dot{M}$ ($M_\odot$ yr$^{-1}$)")
    ax.set_title(r"La segunda palanca: $\dot{M}$ contra la extincion supuesta")
    ax.legend(ncol=2, fontsize=8)
    ax.grid(alpha=0.25, lw=0.4, which="both")
    rep.save(fig, "01_escalera_mdot_av",
             "Mdot en funcion del A_V supuesto, para las dos relaciones de acrecion. Los "
             "circulos son el numero PUBLICADO del metodo canonico a su A_V adoptado (la "
             "vertical punteada): la figura los recalcula desde el flujo limite y falla si "
             "no caen encima. Ese anclaje es el que destapo que E3 publicaba sin el factor "
             "magnetosferico x1.25 que G3 si aplicaba.")


def fig_sensibilidad(rep):
    av = np.linspace(0, 3, 200)
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    ax.plot(av, 10 ** (DLOGMDOT_DAV * av), color="k", lw=1.6)
    for dv, c in ((0.5, "#1f77b4"), (1.0, "#d62728")):
        ax.axvline(dv, color=c, ls=":", lw=1.0)
        ax.text(dv, 10 ** (DLOGMDOT_DAV * dv), f"  x{10**(DLOGMDOT_DAV*dv):.2f}",
                color=c, va="bottom", fontsize=9)
    ax.set_xlabel(r"error en $A_V$ (mag)")
    ax.set_ylabel(r"factor en $\dot{M}$")
    ax.set_title(r"$d\log\dot{M}/dA_V = 0.370$ dex/mag: un factor 2.3 por magnitud")
    ax.grid(alpha=0.25, lw=0.4)
    rep.save(fig, "02_sensibilidad",
             "Cuanto mueve el Mdot un error en la extincion. El +-0.5 mag adoptado son "
             "0.185 dex; una magnitud entera es un factor 2.3. El A_V es el de la PRIMARIA "
             "en los dos objetos, y el sesgo tiene signo: material circumplanetario implica "
             "A_V mayor, luego el Mdot medido es una cota inferior.")


def fig_chi2_perfiles(rep, chi2):
    if not chi2:
        return rep.skip("03_chi2_av", "sin perfiles de chi2")
    fig, axes = plt.subplots(1, len(chi2), figsize=(6.0 * len(chi2), 4.6), squeeze=False)
    for ax, (run, payload) in zip(axes[0], chi2.items()):
        npz = np.load(payload["npz_path"])
        for name, c, ls in (("primary", "#1f77b4", "-"), ("veiling", "#d62728", "--")):
            key = f"{name}_marg_av"
            if key not in npz:
                continue
            av = npz[f"{name}_av_axis"]
            dchi2 = npz[key]
            v = payload["json"]["variants"][name]
            ax.plot(av, dchi2, ls, color=c, lw=1.5,
                    label=(f"{name}: $A_V$={v['av_best']:.1f}, "
                           rf"$\chi^2_\nu$={v['chi2_red']:.2f}"))
            ax.plot([v["av_best"]], [0.0], "o", ms=7, color=c)
        ax.axhline(1.0, color="0.4", lw=0.8, ls=":", label=r"$\Delta\chi^2=1$")
        ax.axhline(9.0, color="0.4", lw=0.8, ls="-.", label=r"$\Delta\chi^2=9$ (3$\sigma$)")
        ax.set_yscale("symlog", linthresh=1)
        # Delta chi2 no puede ser negativo: sin esto symlog dibuja media decada
        # de eje que no existe.
        ax.set_ylim(0, None)
        ax.set_xlabel(r"$A_V$ (mag)")
        ax.set_ylabel(r"$\Delta\chi^2$ (perfilado sobre $T_{\rm eff}$ y $\log g$)")
        ax.set_title(LABEL.get(run, run))
        ax.legend(fontsize=8)
        ax.grid(alpha=0.25, lw=0.4)
    fig.suptitle("Via 3: el ajuste atmosferico da intervalos de UN nodo y dos respuestas "
                 "que difieren 4.9 mag", y=1.02)
    rep.save(fig, "03_chi2_av",
             "Perfil de chi2 a lo largo de A_V, minimizando sobre Teff y logg. Las dos "
             "variantes del MISMO ajuste caen en nodos distintos separados 4.9 mag -1.81 dex "
             "de Mdot, un factor 65- y las dos declaran un intervalo de un solo nodo de 51, "
             "aun despues de inflar los errores x3.01 y x2.09. La que ajusta mucho mejor "
             "(veiling, chi2 349.7 contra 724.8) esta pegada al TECHO del eje, asi que su "
             "A_V preferido puede estar fuera de la rejilla. El QC de G3 publica el "
             "`edge_touch` de la variante primaria, que dice False.")


def fig_chi2_mapa(rep, chi2):
    if not chi2:
        return rep.skip("04_chi2_mapa", "sin perfiles de chi2")
    n = len(chi2) * 2
    fig, axes = plt.subplots(1, n, figsize=(5.6 * n, 4.4), squeeze=False)
    k = 0
    for run, payload in chi2.items():
        npz = np.load(payload["npz_path"])
        for name in ("primary", "veiling"):
            key = f"{name}_dchi2_teff_av"
            if key not in npz:
                continue
            ax = axes[0][k]; k += 1
            M = np.clip(npz[key], 0, 200)
            av, teff = npz[f"{name}_av_axis"], npz[f"{name}_teff_axis"]
            im = ax.pcolormesh(av, teff, M, cmap="viridis_r", shading="auto")
            v = payload["json"]["variants"][name]
            ax.plot([v["av_best"]], [v["teff_best"]], "r*", ms=13)
            ax.set_xlabel(r"$A_V$ (mag)")
            ax.set_ylabel(r"$T_{\rm eff}$ (K)")
            ax.set_title(f"{LABEL.get(run, run)} — {name}")
            fig.colorbar(im, ax=ax, label=r"$\Delta\chi^2$ (recortado a 200)")
    for j in range(k, n):
        axes[0][j].axis("off")
    rep.save(fig, "04_chi2_mapa",
             "El plano (Teff, A_V) con logg minimizado. La degeneracion entre temperatura y "
             "enrojecimiento es lo que hace que anadir veiling -un continuo extra plano- "
             "mueva la solucion de un extremo del eje al otro.")


def fig_extincion_ley(rep):
    # La cita es obligatoria por spec G3 §1.2: no hay ley de extincion anonima.
    ext = CCMExtinction(rv=3.1, citation="Cardelli et al. 1989")
    w = np.linspace(4700, 9300, 400)
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    ax.plot(w, ext.a_lambda_over_av(w), color="k", lw=1.5)
    for lam, name in ((6562.8, r"H$\alpha$"), (4861.3, r"H$\beta$"), (8446.0, "O I")):
        r = float(ext.a_lambda_over_av(np.array([lam]))[0])
        ax.plot([lam], [r], "o", ms=6)
        ax.annotate(f"{name}\n{r:.3f}", (lam, r), textcoords="offset points",
                    xytext=(6, 6), fontsize=8)
    ax.set_xlabel(r"$\lambda$ ($\AA$)")
    ax.set_ylabel(r"$A_\lambda / A_V$")
    ax.set_title("La ley de extincion (Cardelli+1989, $R_V$=3.1) en el rango de MUSE")
    ax.grid(alpha=0.25, lw=0.4)
    rep.save(fig, "05_ley_extincion",
             "El cociente A_lambda/A_V que convierte una extincion supuesta en un factor de "
             "correccion. En Halpha vale 0.818, que con a=1.13 de la relacion L_acc-L_Halpha "
             "da los 0.370 dex/mag. Las lineas de Balmer estan lo bastante juntas en este "
             "cociente como para que el decremento tenga poca palanca sobre A_V, que es la "
             "razon de fondo de que la via 1 se cierre.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--chi2-json", default="")
    ap.add_argument("--chi2-npz", default="")
    args = ap.parse_args()

    def _pairs(s):
        out = {}
        for chunk in s.split(","):
            if "=" in chunk:
                k, v = chunk.split("=", 1)
                out[k.strip()] = v.strip()
        return out

    estilo()
    runs = [r.strip() for r in args.runs.split(",")]
    registrar_runs(runs)
    rep = Report(args.out_dir)

    data = {}
    for run in runs:
        st = ROOT / "runs" / run / "stages"
        tb = ROOT / "runs" / run / "tables"
        d = {}
        p = tb / "mdot_limit_vs_extinction.csv"
        if p.exists():
            with open(p, newline="") as fh:
                d["ladder"] = list(csv.DictReader(fh))
        for key, name in (("h03", "stage_h03_qc.json"), ("g3", "stage_g3_qc.json")):
            q = st / name
            d[key] = read_json(q) if q.exists() else None
        data[run] = d

    cj, cn = _pairs(args.chi2_json), _pairs(args.chi2_npz)
    chi2 = {}
    for run in runs:
        if run in cj and Path(cj[run]).exists() and run in cn and Path(cn[run]).exists():
            chi2[run] = {"json": read_json(Path(cj[run])), "npz_path": cn[run]}

    print("figuras:")
    fig_escalera(rep, data)
    fig_sensibilidad(rep)
    fig_chi2_perfiles(rep, chi2)
    fig_chi2_mapa(rep, chi2)
    fig_extincion_ley(rep)

    bad = [a for a in rep.anchors if not a["ok"]]
    manifest = {"figures": rep.figures, "skipped": rep.skipped,
                "anchors": rep.anchors, "anchors_failed": len(bad),
                "runs": runs, "out_dir": str(rep.out)}
    (rep.out / "manifest.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False))
    print(f"\n{len(rep.figures)} figuras, {len(rep.skipped)} saltadas, "
          f"{len(rep.anchors)} anclas ({len(bad)} fallan) -> {rep.out}")
    if bad:
        print("*** ANCLAS QUE FALLAN: la figura no cae sobre el producto ***")
        for a in bad:
            print(f"    {a['what']}: esperado {a['expected']:.4g}, figura {a['got']:.4g}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
