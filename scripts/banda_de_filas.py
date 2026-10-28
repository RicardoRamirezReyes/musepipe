#!/usr/bin/env python
"""El continuo azul de ROXs 42B b es una banda a lo largo de las filas del cubo (decisión A, 2026-10-25).

Solo lectura. Reproduce las cifras que el paper cita en el pie de la Fig. `full`, la Sect. 4.2 y la
Sect. 5.3 (cabecera de procedencia de `paper/main.tex`, entrada «aplica los tres cambios de texto»),
que salieron de `~/muse_sesion_20261025/azul_42bb/apertura_en_fila.py` (informe en `INFORME.md` de
esa carpeta).

Sobre el cubo con la primaria restada exposición a exposición (C1b, `cube_psfsub_perobs.fits`),
que no depende de C4:

1. colapsa cada banda (media de los canales finitos, sin el hueco del láser de la AO,
   `AO_LASER_WINDOW_A`);
2. mide la fotometría de caja con anillo: la suma de una caja de 5×5 spáxeles centrada en el
   spáxel más cercano, menos 25 veces la mediana de un anillo de 8–14 px alrededor de la posición
   (`annulus_background_spectrum` de C2, sobre la imagen colapsada);
3. la mide en el compañero (posición de B3), en seis posiciones vacías de su misma fila
   (Δx = ±10, ±14, ±18 px) y en doce de otras filas (Δy = ±7, ±10; Δx = −14, 0, +14 px),
   descartando las que quedan a menos de 16 px del borde.

Si la misma fila sin compañero da el mismo flujo que el compañero y las otras filas no, el
«exceso» es la banda y no luz del compañero. Cifras de 42B b (unidades del cubo):

- 4900–5400 Å: compañero 29.8; misma fila 29.9 ± 6.3 (100 ± 21 %); otras filas −1.4 ± 12.6
  (−5 ± 42 %);
- 8000–9300 Å: compañero 422; misma fila 3.1 ± 28.6 (1 ± 7 %).

La dispersión es la desviación típica entre posiciones (ddof = 1) y el porcentaje, la media y la
dispersión divididas por el flujo del compañero.

Diferencias con la extracción de apertura de C2, dichas aquí para que nadie las confunda: la
caja es la de 5×5 (la de verificación de C2, no la de 3×3), no hay corrección de apertura (se
comparan posiciones entre sí, en la misma imagen) y el anillo no excluye los píxeles a menos de
30 px de la primaria. `--excluye-primaria-px 30` aplica esa exclusión de C2 como prueba de
sensibilidad.

uso:
    python scripts/banda_de_filas.py --project-root RAIZ [--run-id RUN ...] [--json SALIDA]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from musepipe.extraction.aperture import annulus_background_spectrum  # noqa: E402
from musepipe.telluric_lines import AO_LASER_WINDOW_A  # noqa: E402

BANDAS_A = {"B1 4900-5400": (4900.0, 5400.0), "B2 5450-5750": (5450.0, 5750.0),
            "6100-7000": (6100.0, 7000.0), "rojo 8000-9300": (8000.0, 9300.0)}
CAJA_PX = 5
ANILLO_PX = (8.0, 14.0)
DX_MISMA_FILA_PX = (-18, -14, -10, 10, 14, 18)
DY_OTRAS_FILAS_PX = (-10, -7, 7, 10)
DX_OTRAS_FILAS_PX = (-14, 0, 14)
MARGEN_PX = 16
BLOQUE_CANALES = 100


def canales_de_banda(wave_A, lo_A, hi_A, *, hueco_A=AO_LASER_WINDOW_A):
    """Índices de los canales en [lo, hi), sin los que caen dentro del hueco del láser (abierto)."""
    w = np.asarray(wave_A, dtype=np.float64)
    return np.flatnonzero((w >= lo_A) & (w < hi_A) & ~((w > hueco_A[0]) & (w < hueco_A[1])))


def colapsa_banda(cubo, wave_A, lo_A, hi_A, *, hueco_A=AO_LASER_WINDOW_A, bloque=BLOQUE_CANALES):
    """Imagen media de la banda: por spáxel, la media de sus canales finitos (0 si no tiene ninguno).

    Lee el cubo por bloques de canales, así que sirve igual con un `memmap` de FITS.
    """
    idx = canales_de_banda(wave_A, lo_A, hi_A, hueco_A=hueco_A)
    acc = np.zeros(cubo.shape[1:])
    n = np.zeros(cubo.shape[1:])
    for i0 in range(0, idx.size, int(bloque)):
        bl = np.asarray(cubo[idx[i0:i0 + int(bloque)]], dtype=np.float64)
        ok = np.isfinite(bl)
        acc += np.where(ok, bl, 0).sum(0)
        n += ok.sum(0)
    return acc / np.maximum(n, 1)


def fotometria_caja_anillo(img, y, x, *, caja=CAJA_PX, anillo_px=ANILLO_PX, excluye_yx=None, excluye_px=0.0):
    """Suma de la caja `caja`×`caja` centrada en el spáxel más cercano a (y, x), menos su área por la
    mediana del anillo `anillo_px` alrededor de (y, x). La caja no se recorta: la posición tiene que
    caber entera en la imagen."""
    caja = int(caja)
    if caja < 1 or caja % 2 == 0:
        raise ValueError(f"la caja tiene que ser impar y positiva, no {caja}")
    h = caja // 2
    yi, xi = int(round(float(y))), int(round(float(x)))
    ny, nx = img.shape
    if not (h <= yi < ny - h and h <= xi < nx - h):
        raise ValueError(f"la caja de {caja} px en ({yi}, {xi}) se sale de la imagen {img.shape}")
    suma = img[yi - h:yi + h + 1, xi - h:xi + h + 1].sum()
    fondo = annulus_background_spectrum(np.asarray(img, dtype=np.float64)[None], (float(y), float(x)),
                                        anillo_px[0], anillo_px[1], exclude_yx=excluye_yx,
                                        exclude_radius=float(excluye_px))[0]
    return float(suma - caja * caja * fondo)


def posiciones_misma_fila(comp_yx, shape, *, dx_px=DX_MISMA_FILA_PX, margen_px=MARGEN_PX):
    yc, xc = (float(v) for v in comp_yx)
    return [(yc, xc + dx) for dx in dx_px if margen_px <= xc + dx < shape[1] - margen_px]


def posiciones_otras_filas(comp_yx, shape, *, dy_px=DY_OTRAS_FILAS_PX, dx_px=DX_OTRAS_FILAS_PX,
                           margen_px=MARGEN_PX):
    yc, xc = (float(v) for v in comp_yx)
    return [(yc + dy, xc + dx) for dy in dy_px for dx in dx_px
            if margen_px <= yc + dy < shape[0] - margen_px and margen_px <= xc + dx < shape[1] - margen_px]


def _estadistica(valores, referencia):
    v = np.asarray(valores, dtype=np.float64)
    media = float(np.mean(v))
    sd = float(np.std(v, ddof=1)) if v.size > 1 else float("nan")
    return {"n": int(v.size), "media": media, "sd": sd, "valores": [float(x) for x in v],
            "media_pct": 100.0 * media / referencia, "sd_pct": 100.0 * sd / referencia}


def mide_banda(img, comp_yx, *, excluye_yx=None, excluye_px=0.0):
    """El compañero, su fila y las otras filas en UNA imagen colapsada."""
    kw = {"excluye_yx": excluye_yx, "excluye_px": excluye_px}
    comp = fotometria_caja_anillo(img, *comp_yx, **kw)
    misma = [fotometria_caja_anillo(img, y, x, **kw) for y, x in posiciones_misma_fila(comp_yx, img.shape)]
    otras = [fotometria_caja_anillo(img, y, x, **kw) for y, x in posiciones_otras_filas(comp_yx, img.shape)]
    return {"companero": comp, "misma_fila": _estadistica(misma, comp), "otras_filas": _estadistica(otras, comp)}


def lee_run(run_id, project_root, *, bandas=BANDAS_A, excluye_px=0.0):
    """Las bandas de un run, leyendo el cubo de C1b y las posiciones de B3."""
    from astropy.io import fits

    from musepipe.io import read_json
    from musepipe.stages.stage_e01b_perobs_subtract import stage_e01b_paths
    from musepipe.stages.stage_x01_aperture import _source_pos_yx

    paths = stage_e01b_paths(run_id, project_root)
    qc = read_json(paths["stage01c_qc_json"])
    comp, star = _source_pos_yx(qc, "companion"), _source_pos_yx(qc, "primary")
    out = {"run_id": run_id, "cubo": str(paths["cube_psfsub_fits"]), "companero_yx": list(comp),
           "primaria_yx": list(star), "excluye_primaria_px": float(excluye_px), "bandas": {}}
    with fits.open(paths["cube_psfsub_fits"], memmap=True) as h:
        cubo, wave = h["DATA"].data, np.asarray(h["WAVELENGTH"].data, dtype=np.float64)
        out["forma_cubo"] = list(cubo.shape)
        for nombre, (lo, hi) in bandas.items():
            img = colapsa_banda(cubo, wave, lo, hi)
            out["bandas"][nombre] = {"lo_A": lo, "hi_A": hi, "n_canales": int(canales_de_banda(wave, lo, hi).size),
                                     **mide_banda(img, comp, excluye_yx=star, excluye_px=excluye_px)}
    return out


def imprime(res):
    print(f"== {res['run_id']}: compañero ({res['companero_yx'][0]:.1f}, {res['companero_yx'][1]:.1f}), "
          f"cubo {res['cubo']}" + (f", anillo sin r < {res['excluye_primaria_px']:g} px de la primaria"
                                   if res["excluye_primaria_px"] else ""))
    for nombre, b in res["bandas"].items():
        m, o = b["misma_fila"], b["otras_filas"]
        print(f"  {nombre:15s} compañero {b['companero']:8.1f} | misma fila sin compañero {m['media']:8.1f} ± "
              f"{m['sd']:5.1f} (n={m['n']}; {m['media_pct']:.0f} ± {m['sd_pct']:.0f} %) | otras filas "
              f"{o['media']:8.1f} ± {o['sd']:5.1f} (n={o['n']}; {o['media_pct']:.0f} ± {o['sd_pct']:.0f} %)")


def runs_del_paper():
    """`paper.run` de cada `targets/<slug>.json`, en el orden del paper."""
    fichas = []
    for f in sorted((ROOT / "targets").glob("*.json")):
        p = json.loads(f.read_text(encoding="utf-8")).get("paper")
        if p:
            fichas.append((int(p.get("order", 99)), p["run"]))
    return [r for _o, r in sorted(fichas)]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--run-id", action="append", default=None,
                    help="repetible; por defecto, el run del paper de cada objeto (targets/*.json)")
    ap.add_argument("--project-root", default=str(ROOT), help="clon con runs/ (desde un worktree, el principal)")
    ap.add_argument("--excluye-primaria-px", type=float, default=0.0,
                    help="anillo sin los píxeles a menos de R px de la primaria (C2 usa 30; la sesión, 0)")
    ap.add_argument("--json", default=None)
    a = ap.parse_args(argv)
    runs = a.run_id or runs_del_paper()
    res = [lee_run(r, Path(a.project_root).resolve(), excluye_px=a.excluye_primaria_px) for r in runs]
    for r in res:
        imprime(r)
    if a.json:
        Path(a.json).write_text(json.dumps(res, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
