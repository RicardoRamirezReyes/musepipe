#!/usr/bin/env python
"""Nivel del sesgo de controles de psffit y el radio de ajuste de C1 en que cruza cero.

Solo lectura. Es el árbitro del radio de ajuste de C1 (docs/2026-09-21_plan_criterio_radio_ajuste_c1.md,
§R2, §D7 y §F4; docs/2026-09-27_r0_entregado_42bb.md). Vivió fuera del repo hasta el 2026-09-28
(`~/muse_noche_20260927/nivel_sesgo_controles.py`, reconstruido el 09-27).
`tests/test_nivel_sesgo_controles.py` comprueba contra `runs/` que da las cifras publicadas en los dos
docs (las nueve del 09-21, los niveles y los r₀ del 09-27); la tabla vive allí porque nombra runs.

    b_i = 100 · mediana_λ(control_i) / mediana_λ(objeto)       (% del compañero; canales finitos del objeto)
    b_i = A + B·cos φ_i + C·sin φ_i                              (mínimos cuadrados, m ≤ 1)

con los espectros **calibrados** de D2 (`spec_calibrated_psffit_{object.fits,controls.npz}`) y φ_i el
azimut de cada control respecto al compañero. C4 no guarda las posiciones de sus controles: se
regeneran con `same_radius_control_positions` y los mismos parámetros con que las genera C4
(`stage_x03_config_from_run`: número, ángulo excluido, radio del compañero → margen), sobre la forma
del cubo de C4 y las posiciones de B3. σ(A) sale de la varianza de los residuos (n − 3 g.d.l.).

**Emparejado, no independiente.** Los residuos de los 33 controles se correlacionan 0.999 entre
radios: son las mismas posiciones y el error del nivel es común. La diferencia entre dos radios se
ajusta control a control (`b_2 − b_1`) y su error es ~×4 menor que el de cada nivel. Por eso el error
de r₀ lo fija σ(A) del radio de referencia, no el número de radios.

**El cruce se lee en el tramo que lo encierra** (el nivel NO es lineal en el radio, §5.2 del doc del
09-27): r₀ = r_ref − A_ref / pendiente, con r_ref el extremo del tramo más cercano al cruce. Como el
ajuste es lineal en b, A(b_2 − b_1) = A_2 − A_1 y r₀ no depende del extremo; su error sí. Si ningún
tramo lo encierra, se extrapola desde el tramo más próximo y el resultado lo dice.

**El error de r₀ es asimétrico cuando A_ref es compatible con cero** (2026-10-01): el cruce puede
caer al otro lado de r_ref, en un tramo de otra pendiente, y σ(A_ref)/pendiente no lo ve.
`--bootstrap N` remuestrea los controles (los mismos índices en todos los radios) y da los
percentiles de r₀ y, con `--hipotesis`, la fracción de muestras más allá de cada radio predicho.

`--absoluto` deja b_i en el flujo del producto, sin dividir por la mediana del objeto, que cambia con
el radio: sirve para ver que la curvatura no la fabrica el denominador.

uso: python scripts/nivel_sesgo_controles.py nivel RUN [RUN ...] [--absoluto]
     python scripts/nivel_sesgo_controles.py radios RUN RUN [RUN ...] [--absoluto] [--json SALIDA]
                                                   [--bootstrap N [--semilla S] [--hipotesis R ...]]
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path

import numpy as np
from astropy.io import fits

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from musepipe.apertures import same_radius_control_positions  # noqa: E402

def azimuts(comp_yx, star_yx, ny, nx, *, n_controls, exclude_angle_deg, comp_radius_px):
    """φ_i de los controles de C4 respecto a la dirección del compañero (misma receta que psffit.py)."""
    ctr = same_radius_control_positions(comp_yx, star_yx, ny, nx, n_positions=int(n_controls),
                                        exclude_angle_deg=float(exclude_angle_deg),
                                        margin_px=int(np.ceil(comp_radius_px)) + 1)
    th0 = math.atan2(comp_yx[0] - star_yx[0], comp_yx[1] - star_yx[1])
    return np.array([math.atan2(y - star_yx[0], x - star_yx[1]) - th0 for y, x in ctr])


def sesgos_de(objeto, controles, absoluto=False):
    """b_i en % del objeto: mediana de cada control sobre los canales finitos del objeto.

    Con ``absoluto``, la mediana de cada control en el flujo del producto, sin dividir.
    """
    objeto = np.asarray(objeto, dtype=float)
    ok = np.isfinite(objeto)
    med = np.nanmedian(np.where(ok, np.asarray(controles, dtype=float), np.nan), axis=1)
    return med if absoluto else 100.0 * med / np.nanmedian(objeto[ok])


def ajuste(b, phi):
    """Nivel A, σ(A) y amplitud del dipolo de b = A + B cos φ + C sin φ."""
    b, phi = np.asarray(b, dtype=float), np.asarray(phi, dtype=float)
    X = np.c_[np.ones_like(phi), np.cos(phi), np.sin(phi)]
    coef, *_ = np.linalg.lstsq(X, b, rcond=None)
    r = b - X @ coef
    cov = np.linalg.inv(X.T @ X) * float(r @ r) / (b.size - 3)
    return float(coef[0]), math.sqrt(cov[0, 0]), float(math.hypot(coef[1], coef[2]))


def lee_run(run, root=ROOT, absoluto=False):
    """(b, φ, radio de C1) de un run, con los knobs RESUELTOS de C1 y C4."""
    from musepipe.stages.stage_e01_psf import stage_e01_config_from_run
    from musepipe.stages.stage_x01_aperture import _source_pos_yx
    from musepipe.stages.stage_x03_psffit import stage_x03_config_from_run, stage_x03_paths

    c1 = stage_e01_config_from_run(run, project_root=root)
    c4 = stage_x03_config_from_run(run, project_root=root)
    paths = stage_x03_paths(run, root)
    qc = json.loads(Path(paths["stage01c_qc_json"]).read_text(encoding="utf-8"))
    comp = _source_pos_yx(qc, c4.get("x03_source_key", "companion"))
    star = _source_pos_yx(qc, c4.get("x03_star_key", "primary"))
    with fits.open(paths["stage02_cube_fits"], memmap=True) as h:
        ny, nx = h["CUBES"].shape[-2:]
    phi = azimuts(comp, star, ny, nx, n_controls=c4["x03_control_apertures"],
                  exclude_angle_deg=c4["x03_control_exclude_angle_deg"],
                  comp_radius_px=c4["x03_comp_radius_px"])
    st = Path(paths["paths"].stage_dir)
    obj = fits.getdata(st / "spec_calibrated_psffit_object.fits", 1)["flux"]
    ctl = np.load(st / "spec_calibrated_psffit_controls.npz")["control_spectra"]
    if ctl.shape[0] != phi.size:
        raise ValueError(f"{run}: {ctl.shape[0]} controles en el npz y {phi.size} posiciones regeneradas")
    return sesgos_de(obj, ctl, absoluto), phi, float(c1["psf_fit_radius_px"])


def resumen(run, b, phi, radio):
    A, eA, dip = ajuste(b, phi)
    return {"run": run, "radio_px": radio, "n": int(b.size), "mediana": float(np.median(b)),
            "e_mediana": 1.2533 * float(b.std(ddof=1)) / math.sqrt(b.size),
            "media": float(b.mean()), "A": A, "eA": eA, "dipolo": dip}


def emparejado(b1, b2, phi, r1, r2):
    """ΔA = A(b2 − b1), ajustado control a control, y la pendiente del tramo r1 → r2."""
    dA, edA, _ = ajuste(np.asarray(b2) - np.asarray(b1), phi)
    return {"desde_px": r1, "hasta_px": r2, "dA": dA, "e_dA": edA,
            "significancia": abs(dA) / edA if edA else None,
            "pendiente": dA / (r2 - r1), "e_pendiente": edA / (r2 - r1)}


def curvatura(b1, b2, b3, phi, r1, r2, r3):
    """Pendiente del tramo alto menos la del bajo, control a control (comparten b2)."""
    c, ec, _ = ajuste((np.asarray(b3) - b2) / (r3 - r2) - (np.asarray(b2) - b1) / (r2 - r1), phi)
    return {"radios_px": [r1, r2, r3], "pendiente_alta_menos_baja": c, "e": ec,
            "significancia": abs(c) / ec if ec else None}


def cruce(niveles, tramos):
    """r₀ en el tramo que encierra el cero del nivel; si ninguno, extrapolado desde el más próximo.

    `niveles` ordenados por radio (salida de `resumen`), `tramos[k]` une niveles[k] y niveles[k+1].
    """
    A = [n["A"] for n in niveles]
    k = next((i for i in range(len(tramos)) if A[i] * A[i + 1] <= 0.0), None)
    encierra = k is not None
    if not encierra:
        # el tramo del extremo con el |A| más pequeño
        j = int(np.argmin(np.abs(A)))
        k = min(j, len(tramos) - 1)
    t = tramos[k]
    s, es = t["pendiente"], t["e_pendiente"]
    if not s:
        return {"encierra_el_cruce": encierra, "r0_px": float("nan"), "e_r0_px": float("nan")}
    r0 = niveles[k]["radio_px"] - A[k] / s
    ref = min((niveles[k], niveles[k + 1]), key=lambda n: abs(n["radio_px"] - r0))
    return {"encierra_el_cruce": encierra, "tramo_px": [t["desde_px"], t["hasta_px"]],
            "referencia_px": ref["radio_px"], "r0_px": float(r0),
            "e_r0_px": math.hypot(ref["eA"] / s, ref["A"] * es / s ** 2),
            "nota": ("interpolado en el tramo que lo encierra" if encierra
                     else "EXTRAPOLADO: ningún tramo encierra el cruce")
                    + "; el error es σ(A_ref)/pendiente más el de la pendiente, sin su covarianza"}


def analiza(bs, phi, rs, nombres=None):
    """Niveles, tramos emparejados, curvaturas y cruce de los sesgos `bs`, ordenados por radio `rs`."""
    nombres = nombres or [f"r{r:g}" for r in rs]
    niveles = [resumen(n, b, phi, r) for b, r, n in zip(bs, rs, nombres)]
    tramos = [emparejado(bs[i], bs[i + 1], phi, rs[i], rs[i + 1]) for i in range(len(bs) - 1)]
    curv = [curvatura(bs[i], bs[i + 1], bs[i + 2], phi, rs[i], rs[i + 1], rs[i + 2])
            for i in range(len(bs) - 2)]
    return {"niveles": niveles, "tramos": tramos, "curvatura": curv, "cruce": cruce(niveles, tramos)}


def bootstrap_r0(bs, phi, rs, n=10000, semilla=0):
    """r₀ de `n` remuestreos de los controles con reemplazo, los mismos índices en todos los radios."""
    rng = np.random.default_rng(semilla)
    bs = [np.asarray(b, dtype=float) for b in bs]
    phi = np.asarray(phi, dtype=float)
    out = np.empty(n)
    for i in range(n):
        j = rng.integers(0, phi.size, phi.size)
        niveles = [resumen("", b[j], phi[j], r) for b, r in zip(bs, rs)]
        tramos = [emparejado(bs[k][j], bs[k + 1][j], phi[j], rs[k], rs[k + 1]) for k in range(len(bs) - 1)]
        out[i] = cruce(niveles, tramos)["r0_px"]
    return out


def resumen_bootstrap(muestras, r0, hipotesis=(), semilla=0):
    """Percentiles de r₀ (±1σ y ±2σ gaussianos) y, por hipótesis, la fracción de muestras más allá.

    «Más allá» es del lado de la hipótesis: por debajo si predice menos que r₀, por encima si más. El
    equivalente en σ es el cuantil gaussiano de una cola (`statistics.NormalDist`).
    """
    x = np.asarray(muestras, dtype=float)
    x = x[np.isfinite(x)]
    p = np.percentile(x, [2.275, 15.865, 50.0, 84.135, 97.725])
    res = {"n": int(x.size), "semilla": semilla,
           "percentiles": dict(zip(["p2.3", "p15.9", "p50", "p84.1", "p97.7"], map(float, p))),
           "menos_68": float(r0 - p[1]), "mas_68": float(p[3] - r0), "hipotesis": {}}
    for h in hipotesis:
        f = float(np.mean(x <= h) if h < r0 else np.mean(x >= h))
        res["hipotesis"][f"{h:g}"] = {"fraccion": f,
                                      "sigma_una_cola": statistics.NormalDist().inv_cdf(1.0 - f) if 0 < f < 1 else None}
    return res


def sesgos_por_radio(runs, root=ROOT, absoluto=False):
    """(bs, φ, radios, runs) ordenados por radio; falla si los controles no son las mismas posiciones."""
    datos = sorted((lee_run(r, root, absoluto) + (r,) for r in runs), key=lambda d: d[2])
    phi0 = datos[0][1]
    for b, phi, _r, run in datos[1:]:
        if phi.shape != phi0.shape or not np.allclose(phi, phi0):
            raise ValueError(f"{run}: sus controles no son las mismas posiciones que los de {datos[0][3]}")
    return [d[0] for d in datos], phi0, [d[2] for d in datos], [d[3] for d in datos]


def radios(runs, root=ROOT, absoluto=False):
    return analiza(*sesgos_por_radio(runs, root, absoluto))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="orden", required=True)
    p = sub.add_parser("nivel", help="nivel A, dipolo y mediana de cada run")
    p.add_argument("runs", nargs="+")
    p.add_argument("--absoluto", action="store_true")
    p = sub.add_parser("radios", help="dos o más radios del mismo objeto: tramos emparejados, curvatura y r₀")
    p.add_argument("runs", nargs="+")
    p.add_argument("--absoluto", action="store_true")
    p.add_argument("--json", default=None)
    p.add_argument("--bootstrap", type=int, default=0, metavar="N")
    p.add_argument("--semilla", type=int, default=0)
    p.add_argument("--hipotesis", type=float, nargs="*", default=[], metavar="R")
    a = ap.parse_args(argv)
    if a.orden == "nivel":
        for run in a.runs:
            print(json.dumps(resumen(run, *lee_run(run, absoluto=a.absoluto)), ensure_ascii=False))
        return 0
    if len(a.runs) < 2:
        ap.error("radios necesita al menos dos runs")
    bs, phi, rs, nombres = sesgos_por_radio(a.runs, absoluto=a.absoluto)
    res = analiza(bs, phi, rs, nombres)
    if a.bootstrap:
        res["bootstrap"] = resumen_bootstrap(bootstrap_r0(bs, phi, rs, a.bootstrap, a.semilla),
                                             res["cruce"]["r0_px"], a.hipotesis, a.semilla)
    texto = json.dumps(res, indent=2, ensure_ascii=False)
    if a.json:
        Path(a.json).write_text(texto + "\n", encoding="utf-8")
    print(texto)
    return 0


if __name__ == "__main__":
    sys.exit(main())
