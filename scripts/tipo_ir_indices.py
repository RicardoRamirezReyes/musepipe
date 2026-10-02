#!/usr/bin/env python
"""Tipo espectral en el infrarrojo con los índices de agua de Allers & Liu (2013) sobre los espectros
J/H/K publicados de los dos compañeros (catálogo CDS J/A+A/708/A312, Hayoz+2026).

Solo lectura y fuera de la cadena: G3 tipifica con MUSE y no usa nada de esto. Da la cifra del paper
«M9.9 ± 1.0» de ROXs 42B b (`lectura`) y el control que la sostiene (`control`); los dos están
contados en docs/2026-10-02_tipo_ir_42bb.md. Vivió fuera del repo hasta el 2026-10-02
(`~/muse_sesion_20261002/indices_ir/`). `tests/test_tipo_ir_indices.py` fija las propiedades y, si el
catálogo está en disco, las cifras publicadas.

**Datos.** Los espectros del CDS están en F_λ (W m⁻² µm⁻¹), muestreados a λ/Δλ = 500 por píxel y
**sin desenrojecer**: su flujo medio por banda coincide con la fotometría observada a 3–6 %.

**Índices** (tabla `tbl:indexfits` del LaTeX de arXiv 1305.4418, copiada tal cual):
I = ⟨F_λ⟩_num / ⟨F_λ⟩_den y SpT = c0 + c1 I + c2 I² + c3 I³, con M0 = 0 y L0 = 10. Comprobación de
la escala: el H2O-2 = 0.87 de Currie+2014b da 9.79 = L0, como dice Currie. H2OD no se puede medir:
ningún espectro K empieza antes de su ventana azul (1.951 µm). Una ventana cuenta como cubierta si
cae entera dentro del espectro sin su primer ni su último píxel.

**Ruido.** El catálogo da e_Flux por píxel, pero tras el remuestreo los píxeles vecinos están
correlacionados. El error que se usa es el conservador: correlación total dentro de cada ventana
(σ de la ventana = mediana de e_Flux en ella), con e_Flux inflado si el rms local alrededor de un
polinomio de grado 2 es mayor (el rms también lleva estructura espectral: es una cota superior). Se
propaga analíticamente, sin Monte Carlo.

**Combinación** (`combina`). Cada índice se combina primero ENTRE INSTRUMENTOS (media ponderada del
índice, factor de Birge si discrepan) y el rms del polinomio se suma UNA vez por índice: es la
dispersión del objeto respecto de la relación, común a los dos instrumentos. Entre índices, media
ponderada con σ_i² = ruido_i² + rms_i²; error = max(1, σ_formal · √max(1, χ²_ν)), el suelo de un
subtipo que adoptan Allers & Liu. El sistemático de A_V es la mitad de la diferencia del tipo entre
A_V ± σ_AV.

**Qué espectros.** Los declara el perfil del objeto (`targets/<slug>.json`, clave `ir_spectra`): fichero
del catálogo, banda e instrumento de cada espectro, y los tipos publicados con esos mismos datos
(`published_types`), que son el control. Ningún objeto se nombra aquí (`tests/test_no_hardcoded_target.py`).
Cada índice usa todos los espectros del objeto que lo cubren.

**A_V.** `lectura` usa la del run del paper (`paper.run` del perfil; `h03_av` ± `h03_av_err`, la misma de
la Tabla 1); `control` usa la de cada artículo, declarada en `published_types`. Ley CCM89.

uso: python scripts/tipo_ir_indices.py control SLUG [--cds DIR] [--json SALIDA]
     python scripts/tipo_ir_indices.py lectura SLUG [--cds DIR] [--run RUN] [--json SALIDA]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from musepipe.models.extinction import CCMExtinction  # noqa: E402

DEFAULT_CDS = Path("/mnt/2TB/IR_Data/publicados_Hayoz2026_JAA708A312")

# Allers & Liu 2013, tbl:indexfits: ventanas (µm), validez (subtipos), c0..c3, rms (subtipos).
INDICES = {
    "H2O": dict(num=(1.550, 1.560), den=(1.492, 1.502), valid=(5, 14),
                c=(-83.5437, 169.388, -104.424, 24.0476), rms=0.390),
    "H2OD": dict(num=(1.951, 1.977), den=(2.062, 2.088), valid=(10, 18),
                 c=(79.4477, -202.245, 229.884, -97.230), rms=0.757),
    "H2O-1": dict(num=(1.335, 1.345), den=(1.295, 1.305), valid=(4, 15),
                  c=(12.1927, 39.3513, -80.7404, 28.5982), rms=1.097),
    "H2O-2": dict(num=(2.035, 2.045), den=(2.145, 2.155), valid=(4, 12),
                  c=(10.8822, 55.4580, -97.8144, 37.5013), rms=0.501),
}
# tbl:indices (K I_J) y tbl:indexbounds (umbral de puntuación 1 y 2: índice ≤ umbral).
KIJ = dict(line=1.244, c1=1.220, c2=1.270, bw=0.0166)
KIJ_BOUNDS = {6: (1.042, 1.028), 7: (1.059, 1.036), 8: (1.077, 1.046), 9: (1.085, 1.053),
              10: (1.098, 1.061), 11: (1.114, 1.067), 12: (1.133, 1.073), 13: (1.135, 1.075),
              14: (1.126, 1.072), 15: (1.094, 1.061)}

BAND_INDEX = {"J": "H2O-1", "H": "H2O", "K": "H2O-2"}   # el índice de agua de cada banda
EXT = CCMExtinction(3.1, citation="Cardelli, Clayton & Mathis 1989 (ApJ 345, 245)")


def spt_label(x):
    if x is None or not np.isfinite(x):
        return "—"
    return f"M{x:.1f}" if x < 10 else f"L{x - 10:.1f}"


def spt_from(name, value):
    c = INDICES[name]["c"]
    return c[0] + c[1] * value + c[2] * value**2 + c[3] * value**3


def spt_slope(name, value):
    c = INDICES[name]["c"]
    return c[1] + 2 * c[2] * value + 3 * c[3] * value**2


def load(cds, name):
    a = np.loadtxt(Path(cds) / name, usecols=(1, 2, 3))
    return a[:, 0], a[:, 1], a[:, 2]


def deredden(w_um, f, av):
    return f if av == 0 else f * EXT.deredden_factor(w_um * 1e4, av)


def covered(w, lo, hi):
    return lo >= w[1] and hi <= w[-2]


def window_weights(w, lo, hi, n=201):
    """Pesos por píxel de la media del espectro interpolado linealmente sobre [lo, hi]."""
    g = np.linspace(lo, hi, n)
    j = np.clip(np.searchsorted(w, g) - 1, 0, w.size - 2)
    t = (g - w[j]) / (w[j + 1] - w[j])
    wt = np.zeros(w.size)
    np.add.at(wt, j, (1 - t) / n)
    np.add.at(wt, j + 1, t / n)
    return wt


def window_mean(w, f, lo, hi):
    return float(window_weights(w, lo, hi) @ f)


def noise_inflation(w, f, e, lo, hi, half=0.03):
    c = 0.5 * (lo + hi)
    m = (w >= c - half) & (w <= c + half)
    if m.sum() < 8:
        return 1.0
    p = np.polyfit(w[m] - c, f[m], 2)
    r = f[m] - np.polyval(p, w[m] - c)
    return max(1.0, float(np.std(r, ddof=3) / np.median(e[m])))


def window_sigma(w, f, e, lo, hi):
    """σ conservadora de la ventana (correlación total) e independiente (píxeles sin correlación)."""
    m = (w >= lo - 0.002) & (w <= hi + 0.002)
    cons = float(np.median(e[m])) * noise_inflation(w, f, e, lo, hi)
    indep = float(np.sqrt(window_weights(w, lo, hi) ** 2 @ e**2))
    return cons, indep


def ratio_index(w, f, e, name):
    d = INDICES[name]
    if not (covered(w, *d["num"]) and covered(w, *d["den"])):
        return None
    n, dd = window_mean(w, f, *d["num"]), window_mean(w, f, *d["den"])
    (sn, sn_i), (sd, sd_i) = window_sigma(w, f, e, *d["num"]), window_sigma(w, f, e, *d["den"])
    val = n / dd
    sig = val * math.hypot(sn / n, sd / dd)
    spt = spt_from(name, val)
    return dict(index=val, sigma=sig, sigma_indep=val * math.hypot(sn_i / n, sd_i / dd),
                spt=spt, spt_label=spt_label(spt),
                sigma_spt=math.hypot(abs(spt_slope(name, val)) * sig, d["rms"]),
                valid=bool(d["valid"][0] <= spt <= d["valid"][1]))


def kij_index(w, f):
    h = KIJ["bw"] / 2
    if not all(covered(w, KIJ[k] - h, KIJ[k] + h) for k in ("c1", "line", "c2")):
        return None
    F = {k: window_mean(w, f, KIJ[k] - h, KIJ[k] + h) for k in ("line", "c1", "c2")}
    a = (KIJ["line"] - KIJ["c1"]) / (KIJ["c2"] - KIJ["c1"])
    return ((1 - a) * F["c1"] + a * F["c2"]) / F["line"]


def kij_score(value, spt):
    b1, b2 = KIJ_BOUNDS[int(np.clip(np.round(spt), 6, 15))]
    return 2 if value <= b2 else 1 if value <= b1 else 0


def perfil(slug, root=ROOT):
    from musepipe.targets import load_target
    p = load_target(slug, project_root=root)
    if not p or "ir_spectra" not in p:
        raise SystemExit(f"{slug}: sin perfil o sin 'ir_spectra' en targets/")
    return p


def indices_of(cds, spectra, av):
    """{espectro: {índice: resultado}} de los espectros declarados, a una A_V."""
    out = {}
    for s, spec in spectra.items():
        w, f, e = load(cds, spec["file"])
        fd, ed = deredden(w, f, av), deredden(w, e, av)
        r = {"range_um": [float(w[0]), float(w[-1])]}
        for name in INDICES:
            v = ratio_index(w, fd, ed, name)
            if v is not None:
                r[name] = v
        k = kij_index(w, fd)
        if k is not None:
            r["KI_J"] = k
        out[s] = r
    return out


def grupos(rows, only=None):
    """{índice: [espectros que lo cubren]}, opcionalmente solo entre `only`."""
    out = {}
    for name in INDICES:
        got = [s for s in rows if name in rows[s] and (only is None or s in only)]
        if got:
            out[name] = got
    return out


def combina(rows, grupos):
    """Tipo de un objeto: por índice entre instrumentos (rms una vez), luego entre índices."""
    per = {}
    for name, specs in grupos.items():
        got = [rows[s][name] for s in specs if name in rows.get(s, {})]
        if not got:
            continue
        x = np.array([g["index"] for g in got]); sg = np.array([g["sigma"] for g in got])
        wt = 1 / sg**2
        mean = float(wt @ x / wt.sum()); se = float(wt.sum() ** -0.5)
        dof = len(x) - 1
        birge = math.sqrt(max(1.0, float(wt @ (x - mean) ** 2) / dof)) if dof else 1.0
        se *= birge
        spt = spt_from(name, mean)
        d = INDICES[name]
        per[name] = dict(index=mean, index_err=se, birge=birge, n=len(got), spt=spt, spt_label=spt_label(spt),
                         sigma=math.hypot(abs(spt_slope(name, mean)) * se, d["rms"]),
                         valid=bool(d["valid"][0] <= spt <= d["valid"][1]))
    use = [k for k, v in per.items() if v["valid"]]
    s = np.array([per[k]["spt"] for k in use]); sg = np.array([per[k]["sigma"] for k in use])
    wt = 1 / sg**2
    mean = float(wt @ s / wt.sum()); formal = float(wt.sum() ** -0.5)
    chi2nu = float(wt @ (s - mean) ** 2) / max(1, len(s) - 1)
    return dict(per_index=per, used=use, spt=mean, spt_label=spt_label(mean), formal=formal, chi2nu=chi2nu,
                sigma_stat=max(1.0, formal * math.sqrt(max(1.0, chi2nu))))


def lectura(cds, spectra, av, av_err):
    rows = indices_of(cds, spectra, av)
    res = combina(rows, grupos(rows))
    tipo = lambda a: (lambda r: combina(r, grupos(r))["spt"])(indices_of(cds, spectra, a))
    sys_av = abs(tipo(av + av_err) - tipo(av - av_err)) / 2
    total = math.hypot(res["sigma_stat"], sys_av)
    kij = {s: dict(value=r["KI_J"], score=kij_score(r["KI_J"], res["spt"])) for s, r in rows.items() if "KI_J" in r}
    res.update(av=av, av_err=av_err, sigma_av_sys=sys_av, sigma_total=total,
               p_below_M9_5=0.5 * (1 + math.erf((9.5 - res["spt"]) / (total * math.sqrt(2)))),
               p_above_L1=0.5 * (1 - math.erf((11.0 - res["spt"]) / (total * math.sqrt(2)))), kij=kij)
    return res


def control(cds, p):
    """Cada tipo publicado contra el mismo método sobre los mismos datos, y los pares de instrumentos."""
    ir = p["ir_spectra"]
    out = {"publicados": [], "instrumentos": []}
    for pub in ir["published_types"]:
        rows = indices_of(cds, {s: ir["spectra"][s] for s in pub["spectra"]}, pub["av"])
        if pub["method"] == "visual":
            r = combina(rows, grupos(rows))
            rnd = float(np.round(r["spt"]))
            out["publicados"].append(dict(ref=pub["ref"], method="visual", av=pub["av"], spt=r["spt"],
                                          spt_label=r["spt_label"], redondeado=spt_label(rnd),
                                          publicado=f"{spt_label(pub['spt'])} ± {pub['err']:g}",
                                          dentro=bool(abs(rnd - pub["spt"]) <= pub["err"]),
                                          por_indice={k: v["spt_label"] for k, v in r["per_index"].items()}))
        else:
            name = pub["method"]
            v = [rows[s][name] for s in pub["spectra"] if name in rows[s]]
            v0 = [r[name] for r in indices_of(cds, {s: ir["spectra"][s] for s in pub["spectra"]}, 0.0).values() if name in r]
            out["publicados"].append(dict(ref=pub["ref"], method=name, av=pub["av"], publicado=pub["index"],
                                          index=v[0]["index"], index_av0=v0[0]["index"], sigma=v[0]["sigma"],
                                          delta=v[0]["index"] - pub["index"],
                                          dentro=bool(abs(v[0]["index"] - pub["index"]) <= pub["tol"])))
    rows = indices_of(cds, ir["spectra"], 0.0)
    for band, name in BAND_INDEX.items():
        specs = [s for s, spec in ir["spectra"].items() if spec["band"] == band]
        for i, a in enumerate(specs):
            for b in specs[i + 1:]:
                A, B = rows[a].get(name), rows[b].get(name)
                if A is None or B is None:
                    out["instrumentos"].append(dict(indice=name, a=a, b=b, nota="no cubierto"))
                    continue
                out["instrumentos"].append(dict(indice=name, a=a, b=b, ia=A["index"], ib=B["index"],
                                                t=abs(A["index"] - B["index"]) / math.hypot(A["sigma"], B["sigma"])))
    return out


def av_del_run(run, root=ROOT):
    from musepipe.config import load_run_config
    c = load_run_config(run, project_root=root).config
    return float(c["h03_av"]), float(c["h03_av_err"])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, hlp in (("control", "los índices contra los tipos publicados con los mismos datos"),
                      ("lectura", "el tipo IR con todos los espectros del objeto y la A_V del run")):
        q = sub.add_parser(name, help=hlp)
        q.add_argument("slug")
        q.add_argument("--cds", type=Path, default=DEFAULT_CDS)
        q.add_argument("--json", type=Path)
        if name == "lectura":
            q.add_argument("--run", help="por defecto, paper.run del perfil")
    a = ap.parse_args(argv)
    p = perfil(a.slug)
    if a.cmd == "control":
        out = control(a.cds, p)
    else:
        run = a.run or p["paper"]["run"]
        av, av_err = av_del_run(run)
        out = lectura(a.cds, p["ir_spectra"]["spectra"], av, av_err)
        out["run"] = run
        for k, v in out["per_index"].items():
            print(f"{k:6s} {v['index']:.3f} ± {v['index_err']:.3f} (n={v['n']}, Birge {v['birge']:.2f}) "
                  f"→ {v['spt_label']} ± {v['sigma']:.2f}")
        print(f"tipo IR {out['spt_label']} ± {out['sigma_total']:.2f} (formal {out['formal']:.2f}, "
              f"χ²_ν {out['chi2nu']:.2f}, A_V {av} ± {av_err} de {run} → ±{out['sigma_av_sys']:.2f})")
    text = json.dumps(out, indent=1, default=float, ensure_ascii=False)
    if a.json:
        a.json.write_text(text)
    if a.cmd == "control":
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
