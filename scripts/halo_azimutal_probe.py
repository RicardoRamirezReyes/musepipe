#!/usr/bin/env python
"""La FASE del dipolo del halo, y en que marco vive.

El 2026-08-14 (`docs/2026-08-14_c1_bascula_moffat_y_nucleo.md` §7) se midio que
el halo de ROXs 42B b lleva un **dipolo** (m=1) que crece con λ hasta el 22.6 %,
y que **ninguna de las dos formas de PSF permitidas puede producirlo** —una
elipse es simetrica bajo 180°—. Se anoto como el suelo del residuo de anillo en
ese objeto y quedo abierto que lo causa.

Aquella medida publico la **amplitud** y no la **fase**, que es justo lo que
distingue las hipotesis. Esta sonda mide la fase, donde esta el maximo en radio,
y —lo que decide— **en que marco esta fija**.

Tres preguntas, tres salidas:

  * `fase`   — armonicos de azimut del anillo, por banda de λ. Compara la fase
               del m=1 con el azimut de la fuente de campo declarada.
  * `radio`  — la misma descomposicion a varios radios. Una FUENTE da m=1 con
               maximo a su radio; un gradiente de algo lejano crece hacia fuera.
  * `picos`  — el residuo tras quitar la mediana azimutal radio a radio, que deja
               solo lo que rompe la simetria, y sus maximos.

Todo sobre el **dato**, sin modelo de PSF: asi el resultado no depende de C1, que
es la etapa bajo sospecha.

Medido el 2026-08-30 (`docs/2026-08-30_dipolo_del_halo_es_instrumental.md`): la
fase del m=1 de ROXs 42B b es −102° a −95°, **estable** mientras la amplitud se
multiplica por 2.4, y esta a **70° de ROXs 42B cc1** (−33°), asi que la fuente de
fondo NO lo explica. Los maximos del residuo rojo caen en **±90° en LOS DOS
objetos**, cuyos cubos tienen orientaciones de cielo separadas 233°
(`north_angle_deg` 180.0 y −53.0): la estructura esta fija al **detector**, no al
cielo.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from astropy.io import fits

ROOT = Path(__file__).resolve().parents[1]

BANDAS = {"azul": (4800, 5200), "medio": (6600, 6900), "rojo": (8900, 9300)}


def azimut_deg(dy, dx):
    return np.degrees(np.arctan2(dy, dx))


def _carga(run, banda):
    qc = json.loads((ROOT / f"runs/{run}/stages/stage01c_qc.json").read_text())
    with fits.open(ROOT / f"runs/{run}/stages/stage02_xcorr_cube_stack.fits", memmap=True) as h:
        wave = np.asarray(h["WAVELENGTH"].data, dtype=np.float64)
        cubo = h["CUBES"].data
        cubo = cubo[0] if cubo.ndim == 4 else cubo
        sel = (wave >= banda[0]) & (wave <= banda[1])
        img = np.nanmedian(np.asarray(cubo[sel], dtype=np.float64), axis=0)
    return qc, img


def _armonicos(img, py, px, rc, ancho, excluir):
    """Amplitud y fase de m=1 y m=2 sobre medianas de 16 sectores."""
    ny, nx = img.shape
    yy, xx = np.indices((ny, nx), dtype=np.float64)
    r = np.hypot(yy - py, xx - px)
    m = (r >= rc - ancho) & (r <= rc + ancho)
    for ey, ex, rad in excluir:
        m &= np.hypot(yy - ey, xx - ex) > rad
    a = np.radians(azimut_deg(yy - py, xx - px)[m])
    v = img[m]
    bordes = np.linspace(-np.pi, np.pi, 17)
    idx = np.digitize(a, bordes) - 1
    sec_a, sec_v = [], []
    for k in range(16):
        mm = idx == k
        if mm.sum() >= 5:
            sec_a.append(0.5 * (bordes[k] + bordes[k + 1]))
            sec_v.append(np.nanmedian(v[mm]))
    if len(sec_a) < 8:
        return None
    sec_a, sec_v = np.array(sec_a), np.array(sec_v)
    base = np.nanmedian(sec_v)
    rel = (sec_v - base) / abs(base)
    A = np.stack([np.ones_like(sec_a), np.cos(sec_a), np.sin(sec_a),
                  np.cos(2 * sec_a), np.sin(2 * sec_a)], axis=-1)
    c, *_ = np.linalg.lstsq(A, rel, rcond=None)
    return {
        # pico a pico = 2x la amplitud del armonico, en %
        "amp1": 200.0 * float(np.hypot(c[1], c[2])),
        "fase1": float(np.degrees(np.arctan2(c[2], c[1]))),
        "amp2": 200.0 * float(np.hypot(c[3], c[4])),
        "n": int(m.sum()),
    }


def _posiciones(qc):
    py, px = qc["primary"]["pos_yx"]
    cy, cx = qc["companion"]["pos_yx"]
    campo = (qc.get("field_source") or {}).get("pos_yx")
    return (py, px), (cy, cx), (tuple(campo) if campo else None)


def cmd_fase(run, args):
    for nombre, banda in BANDAS.items():
        qc, img = _carga(run, banda)
        (py, px), (cy, cx), campo = _posiciones(qc)
        rc = float(np.hypot(cy - py, cx - px))
        if nombre == "azul":
            print(f"\n=== {run} · anillo r={rc:.1f}+-{args.ancho} px "
                  f"(north_angle_deg={qc.get('wcs_orientation', {}).get('north_angle_deg')}) ===")
            print(f"  companero az={azimut_deg(cy - py, cx - px):+7.1f}deg", end="")
            if campo:
                print(f" | fuente de campo az={azimut_deg(campo[0] - py, campo[1] - px):+7.1f}deg "
                      f"r={np.hypot(campo[0] - py, campo[1] - px):.1f}px", end="")
            print()
        h = _armonicos(img, py, px, rc, args.ancho, [(cy, cx, args.excluir)])
        if h:
            print(f"  {nombre:>6} {banda[0]}-{banda[1]} A: m=1 amp={h['amp1']:5.1f}% "
                  f"fase={h['fase1']:+7.1f}deg | m=2 amp={h['amp2']:5.1f}%")


def cmd_radio(run, args):
    qc, img = _carga(run, BANDAS["rojo"])
    (py, px), (cy, cx), _ = _posiciones(qc)
    print(f"\n=== {run} · m=1 contra radio (banda roja) ===")
    print(f"{'r [px]':>7} {'m=1 amp':>9} {'fase':>9} {'m=2 amp':>9}")
    for rc in args.radios:
        h = _armonicos(img, py, px, rc, 4.0, [(cy, cx, args.excluir)])
        if h:
            print(f"{rc:7.1f} {h['amp1']:8.1f}% {h['fase1']:+8.1f} {h['amp2']:8.1f}%")


def cmd_picos(run, args):
    for nombre, banda in BANDAS.items():
        if nombre == "medio":
            continue
        qc, img = _carga(run, banda)
        (py, px), (cy, cx), campo = _posiciones(qc)
        ny, nx = img.shape
        yy, xx = np.indices((ny, nx), dtype=np.float64)
        r = np.hypot(yy - py, xx - px)
        # Quitar la mediana azimutal radio a radio deja SOLO lo que rompe la
        # simetria: un halo perfectamente circular se va entero.
        res = img.copy()
        for rc in range(0, int(r.max()) + 1):
            m = (r >= rc) & (r < rc + 1)
            if m.sum() >= 8:
                res[m] = img[m] - np.nanmedian(img[m])
        lejos = (r > 55) & (r < 75)
        sigma = 1.4826 * np.nanmedian(np.abs(res[lejos] - np.nanmedian(res[lejos])))
        z = np.where((r > args.rmin) & (r < args.rmax), res / sigma, -np.inf)
        print(f"\n=== {run} · {nombre}: picos del residuo en "
              f"{args.rmin}<r<{args.rmax} px (sigma={sigma:.3g}) ===")
        for _ in range(args.npicos):
            k = np.unravel_index(np.nanargmax(z), z.shape)
            if not np.isfinite(z[k]):
                break
            dy, dx = k[0] - py, k[1] - px
            etiqueta = ""
            if np.hypot(k[0] - cy, k[1] - cx) < 6:
                etiqueta = "  <-- COMPANERO"
            elif campo and np.hypot(k[0] - campo[0], k[1] - campo[1]) < 6:
                etiqueta = "  <-- fuente de campo"
            print(f"  ({k[0]:3d},{k[1]:3d})  r={np.hypot(dy, dx):5.1f}px  "
                  f"az={azimut_deg(dy, dx):+7.1f}deg  {z[k]:6.1f} sigma{etiqueta}")
            z[np.hypot(yy - k[0], xx - k[1]) < 6] = -np.inf


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", required=True, help="lista separada por comas")
    ap.add_argument("--que", default="fase,radio,picos", help="fase, radio y/o picos")
    ap.add_argument("--ancho", type=float, default=4.5, help="medio ancho del anillo, px")
    ap.add_argument("--excluir", type=float, default=9.0, help="radio de exclusion del companero")
    ap.add_argument("--radios", type=float, nargs="*",
                    default=[20, 30, 40, 46.6, 55, 65, 75])
    ap.add_argument("--rmin", type=float, default=25.0)
    ap.add_argument("--rmax", type=float, default=60.0)
    ap.add_argument("--npicos", type=int, default=6)
    args = ap.parse_args(argv)

    quiere = {q.strip() for q in args.que.split(",") if q.strip()}
    for run in [r.strip() for r in args.runs.split(",") if r.strip()]:
        if "fase" in quiere:
            cmd_fase(run, args)
        if "radio" in quiere:
            cmd_radio(run, args)
        if "picos" in quiere:
            cmd_picos(run, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
