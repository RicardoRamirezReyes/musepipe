#!/usr/bin/env python
"""ATMO 2020 (Phillips et al. 2020, A&A 637, A38) a npz en el formato de `musepipe.models.cache`, como rejilla de
`scripts/teff_bayes.py` (`--rejilla atmo_ceq` / `atmo_ceq1500`), y el control «BT-Settl con el muestreo de ATMO»
(`--rejilla btsettl_bin`).

Fuera de la cadena y solo lectura sobre los espectros de entrada: escribe solo en el directorio de salida que se le
da (la rejilla convertida no entra en el repo; por defecto `teff_bayes.py` la busca donde dice `REJILLAS`, y
`--dir-rejilla` la apunta a otro sitio). La prueba de ATMO 2020 del apéndice B del paper la hizo el agente `atmo`
el 2026-10-26 (indicación 1 de Claudio, `~/muse_sesion_20261026/claudio/atmo/INFORME.md`); este es su adaptador.

Validez: sus autores dan la rejilla por válida SOLO bajo ~2000 K («since we do not include some sources of opacity
(e.g. some hydrides and condensates) that form at higher temperatures», Phillips et al. 2020, Sect. 2.1); los
compañeros están en 2000–2750 K. Y piden que se les escriba (mp537@exeter.ac.uk) antes de publicar con ella.

1. `atmo DIR_TXT DIR_NPZ [--sufijo CEQ] [--tmin 2000] [--tmax 3000]`: los espectros `spec_T<T>_lg<g>_<sufijo>.txt`
   de `atmosphere_models/CEQ_spectra/` (λ en µm, F_λ en W m⁻² µm⁻¹, flujo de superficie) con tmin <= T <= tmax a
   npz `teff<T>_logg<g>.npz`: λ en Å (×1e4) y F_λ en erg s⁻¹ cm⁻² Å⁻¹ (×0.1: 1 W = 10⁷ erg s⁻¹, 1 m⁻² = 10⁻⁴
   cm⁻², 1 µm⁻¹ = 10⁻⁴ Å⁻¹), recortados a 4000–25 500 Å como las demás rejillas. Comprueba
   ∫F dλ / σT⁴ sobre el fichero completo (0.2–1980 µm) y no escribe el nodo si sale de 0.95–1.05 (en la rejilla
   real, 0.993–1.016). `--tmin 1500` da la rejilla ampliada 1500–3000 K (`atmo_ceq1500`). Deja
   `conversion.json` con el cociente y el paso medido de cada nodo.
   El muestreo de ATMO es uniforme en número de onda: 10 cm⁻¹ por punto, cada uno la media de su banda
   (correlated-k), así que su «píxel» mide Δλ = λ² · 10 cm⁻¹ (5.6 Å a 0.75 µm, 48.5 Å a 2.2 µm). Lo usa
   `teff_bayes.degrada_nativa` (`nu_bin_cm` de `REJILLAS`).
2. `btbin DIR_BT DIR_NPZ ATMO_TXT`: cada npz de BT-Settl promediado en las bandas de 10 cm⁻¹ de ATMO (la media
   exacta de la interpolación lineal en cada banda, `tipo_ir_plantillas.media_en_intervalos`) y puesto en los
   centros de banda de ATMO: BT-Settl con el muestreo de ATMO, el control que separa el efecto del muestreo del de
   la física.

uso: python scripts/convierte_atmo.py atmo /mnt/2TB/IR_Data/atmo2020/atmosphere_models/CEQ_spectra SALIDA/npz_ceq
     python scripts/convierte_atmo.py atmo .../CEQ_spectra SALIDA/npz_ceq1500 --tmin 1500
     python scripts/convierte_atmo.py btbin /mnt/2TB/IR_Data/btsettl_cifist/npz SALIDA/npz_btsettl_bin \\
            .../CEQ_spectra/spec_T2400_lg4.0_CEQ.txt
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
from scipy.integrate import trapezoid

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from musepipe.models.cache import write_spectrum_npz  # noqa: E402
from scripts.tipo_ir_plantillas import media_en_intervalos  # noqa: E402

SIGMA_SB = 5.670374419e-5          # erg s⁻¹ cm⁻² K⁻⁴
#: paso en número de onda de ATMO 2020 (cm⁻¹), medido en los ficheros públicos (Δν = 10.000 en los 112 nodos de
#: 1500–3000 K) y declarado por sus autores (5000 bandas entre 1 y 50 000 cm⁻¹, Sect. 2.4)
DNU_CM = 10.0
#: el recorte de las demás rejillas de `teff_bayes.py` (MUSE + J/H/K)
LO_A, HI_A = 4000.0, 25500.0
#: ∫F dλ / σT⁴ aceptado: el modelo es flujo de superficie
COCIENTE_OK = (0.95, 1.05)
PATRON = r"spec_T(\d+)_lg([\d.]+)_{sufijo}$"


def lee_atmo(path):
    """(λ en Å, F_λ en erg s⁻¹ cm⁻² Å⁻¹) de un espectro de ATMO 2020 (dos columnas: µm y W m⁻² µm⁻¹)."""
    d = np.loadtxt(path, comments="#")
    return d[:, 0] * 1e4, d[:, 1] * 0.1          # µm -> Å ; W m⁻² µm⁻¹ -> erg s⁻¹ cm⁻² Å⁻¹


def paso_nu_cm(w_A):
    """El paso en número de onda (cm⁻¹, mediana) de una λ en Å."""
    nu = 1e8 / np.asarray(w_A, float)
    return float(np.median(np.abs(np.diff(nu))))


def nodo_de(nombre, sufijo="CEQ"):
    """(T_eff, log g) del nombre de un fichero de ATMO (`spec_T2400_lg4.0_CEQ.txt`), o None si no es uno."""
    base = Path(nombre).name
    base = base[:-4] if base.endswith(".txt") else base      # no `stem`: «lg5.5_CEQ» parece una extensión
    m = re.match(PATRON.format(sufijo=sufijo), base)
    return (float(m.group(1)), float(m.group(2))) if m else None


def atmo(dir_txt, dir_npz, sufijo="CEQ", tmin=2000.0, tmax=3000.0):
    """Convierte los nodos con tmin <= T <= tmax; devuelve (y escribe en `conversion.json`) una fila por nodo."""
    dir_txt, dir_npz = Path(dir_txt), Path(dir_npz)
    dir_npz.mkdir(parents=True, exist_ok=True)
    filas = []
    for f in sorted(dir_txt.glob(f"spec_T*_lg*_{sufijo}.txt")):
        nodo = nodo_de(f.name, sufijo)
        if nodo is None:
            continue
        T, g = nodo
        if not tmin <= T <= tmax:
            continue
        w, fl = lee_atmo(f)
        dnu = paso_nu_cm(w)
        cociente = trapezoid(fl, w) / (SIGMA_SB * T ** 4)
        ok = bool(np.all(np.isfinite(fl)) and np.all(fl >= 0) and COCIENTE_OK[0] < cociente < COCIENTE_OK[1])
        meta = {"teff_k": T, "logg": g, "meta_fe_h": 0.0, "alpha": 0.0, "grid": f"ATMO 2020 ({sufijo})",
                "citation": "Phillips et al. 2020, A&A 637, A38", "flux_unit": "erg/cm2/s/A",
                "wave_unit": "angstrom", "wave_frame": "vacuum (synthetic; supuesto, como las demás rejillas)",
                "source_file": f.name, "flux_integral_over_sigmaT4": round(float(cociente), 4),
                "sampling": f"uniforme en número de onda, {dnu:.3f} cm^-1 por punto (media de banda)",
                "validez_autores": "Teff < 2000 K (README de ATMO 2020)"}
        if ok:
            write_spectrum_npz(dir_npz / f"teff{int(T)}_logg{g:.1f}.npz", w, fl, meta, wave_lo=LO_A, wave_hi=HI_A)
        filas.append(dict(fichero=f.name, teff=T, logg=g, n=int(w.size), dnu_cm=round(dnu, 4),
                          flujo_sobre_sigmaT4=round(float(cociente), 4), escrito=ok))
        print(f"{f.name}: ∫F/σT⁴ {cociente:.4f}, Δν {dnu:.3f} cm⁻¹, {'ok' if ok else 'RECHAZADO'}")
    if not filas:
        raise SystemExit(f"no hay espectros spec_T*_lg*_{sufijo}.txt con {tmin:g} <= T <= {tmax:g} K en {dir_txt}")
    (dir_npz / "conversion.json").write_text(json.dumps(filas, indent=1, ensure_ascii=False))
    return filas


def bandas_atmo(atmo_txt, dnu_cm=DNU_CM, lo_A=LO_A, hi_A=HI_A):
    """(centros, bordes inferior y superior en Å) de las bandas de ATMO que caen en lo_A–hi_A (con 1 Å de margen)."""
    wa, _ = lee_atmo(atmo_txt)
    nu_c = 1e8 / wa                                         # centros de banda (cm⁻¹)
    lo = 1e8 / (nu_c + dnu_cm / 2)                          # bordes en Å
    hi = 1e8 / (nu_c - dnu_cm / 2)
    sel = (lo >= lo_A + 1) & (hi <= hi_A - 1)
    return wa[sel], lo[sel], hi[sel]


def btbin(dir_bt, dir_npz, atmo_txt):
    """BT-Settl promediado en las bandas de 10 cm⁻¹ de ATMO, en sus centros."""
    dir_bt, dir_npz = Path(dir_bt), Path(dir_npz)
    dir_npz.mkdir(parents=True, exist_ok=True)
    wc, lo, hi = bandas_atmo(atmo_txt)
    n = 0
    for f in sorted(dir_bt.glob("teff*_logg*.npz")):
        with np.load(f, allow_pickle=False) as z:
            w, fl = np.asarray(z["wave_A"], float), np.asarray(z["flux"], float)
            meta = json.loads(str(z["meta_json"]))
        fb = media_en_intervalos(w, fl, lo, hi)
        meta = {**meta, "grid": meta.get("grid", "BT-Settl") + " con el muestreo de ATMO 2020 (10 cm^-1)",
                "derivado_de": str(f)}
        write_spectrum_npz(dir_npz / f.name, wc, fb, meta, wave_lo=LO_A, wave_hi=HI_A)
        n += 1
        print(f"{f.name}: {wc.size} bandas, finitas {np.isfinite(fb).mean():.3f}")
    if not n:
        raise SystemExit(f"no hay npz teff*_logg*.npz en {dir_bt}")
    return n


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="orden", required=True)
    a1 = sub.add_parser("atmo", help="espectros de ATMO 2020 (txt) a npz de `musepipe.models.cache`")
    a1.add_argument("dir_txt", type=Path, help="atmosphere_models/CEQ_spectra del tar de los autores")
    a1.add_argument("dir_npz", type=Path, help="directorio de salida (se crea)")
    a1.add_argument("--sufijo", default="CEQ", help="química: CEQ (NEQ_weak/NEQ_strong solo llegan a 1800 K)")
    a1.add_argument("--tmin", type=float, default=2000.0, help="T_eff mínima (K); 1500 para la rejilla ampliada")
    a1.add_argument("--tmax", type=float, default=3000.0, help="T_eff máxima (K)")
    a2 = sub.add_parser("btbin", help="BT-Settl promediado en las bandas de 10 cm⁻¹ de ATMO (control)")
    a2.add_argument("dir_bt", type=Path, help="npz de BT-Settl (p. ej. /mnt/2TB/IR_Data/btsettl_cifist/npz)")
    a2.add_argument("dir_npz", type=Path, help="directorio de salida (se crea)")
    a2.add_argument("atmo_txt", type=Path, help="un espectro de ATMO cualquiera: da los centros de banda")
    a = ap.parse_args(argv)
    if a.orden == "atmo":
        atmo(a.dir_txt, a.dir_npz, a.sufijo, a.tmin, a.tmax)
    else:
        btbin(a.dir_bt, a.dir_npz, a.atmo_txt)
    return 0


if __name__ == "__main__":
    sys.exit(main())
