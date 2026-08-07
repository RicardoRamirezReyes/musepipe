#!/usr/bin/env python3
"""Ajusta molecfit en las DOS granularidades y deja los productos comparables.

`docs/2026-08-06_indicaciones_reduccion.md` §5 decide que la granularidad de
molecfit —por exposición o sobre el cubo ya combinado— no se elige a priori: se
hacen las dos y se comparan. Este script produce el material de esa comparación;
quien la lee y la juzga es la §14 de `notebooks/<obj>/debug/A3_telluric_debug.ipynb`.

Para cada cubo se extrae el espectro de apertura de la primaria, se normaliza con
la misma preparación que usó A1a y se lanza `molecfit_model`. Salida:

    <out>/manifest.json     una fila por ajuste: condiciones, apertura y ajuste
    <out>/curvas.npz        wave + mtrans/mrange de cada ajuste, para cargar rápido
    <out>/<etiqueta>/       los productos de esorex, sin tocar

Nada de esto entra en la cadena: no escribe en `stages/`, no toca ningún QC y no
sustituye a A3. Es material de diagnóstico.

TRES PARÁMETROS SIN LOS CUALES ESTO NO CONVERGE, medidos el 2026-08-06
(§5.2a del documento):

* `--WLC_CONST=0`. El defecto de la receta es −0.05, que mete el modelo
  desplazado en λ: el χ² inicial sale ~286× mayor, el ajuste NO se mueve ni una
  iteración y `rel_mol_col_O2` se queda clavado en 1.0000. Es la misma firma que
  se leyó como «molecfit no converge» el 2026-07-06.
* La **altitud del telescopio**. `ESO TEL ALT` no viaja al `science.fits`, así
  que se pasa a mano. Aquí se calcula a MITAD de exposición, `asin(1/X_mid)` con
  `AIRM START`/`END`: en 720 s la masa de aire se mueve lo bastante como para
  que el valor de inicio no represente a la exposición.
* La **humedad**. El cubo no la trae; se lee del crudo de esa exposición.

Uso:

    python scripts/molecfit_granularity.py --out <dir> \
        --combined /ruta/DATACUBE_FINAL.fits \
        --exposures /ruta/exp1/DATACUBE_FINAL.fits ... \
        --prep-dir /ruta/raw_reduction/molecfit_a1a \
        --raw-dir /ruta/a/los/crudos
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
from astropy.io import fits

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from musepipe.reduction.telluric import wavelength_axis_from_header  # noqa: E402
from musepipe.reduction.verify import extract_aperture_spectrum  # noqa: E402

#: La preparación de A1a (`prep.py`, que normaliza el flujo, y el `WAVE_INCLUDE`
#: de dos ventanas) se reutiliza tal cual en vez de reescribirse: es el único
#: punto de partida de este proyecto que se sabe que converge, y rehacerlo
#: rompería la comparabilidad con aquel resultado. Vive en la carpeta
#: `raw_reduction/molecfit_a1a/` de la reducción histórica del objeto, y se pasa
#: con `--prep-dir` — no se escribe aquí, porque un nombre de run en el código es
#: exactamente lo que impide que un objeto nuevo use este script.
PREP = "prep.py"
WAVE_INCLUDE = "wave_include2.fits"

RHUM_POR_DEFECTO = 15.0


def humedad_del_crudo(date_obs, raw_dir):
    """Humedad relativa real de la exposición. El cubo no la propaga; el crudo sí."""
    if not raw_dir:
        return RHUM_POR_DEFECTO, "sin --raw-dir"
    for path in sorted(glob.glob(os.path.join(str(raw_dir), "MUSE.*.fits"))):
        try:
            head = fits.getheader(path)
        except OSError:
            continue
        if str(head.get("DATE-OBS", ""))[:19] == str(date_obs)[:19]:
            return float(head.get("HIERARCH ESO TEL AMBI RHUM", RHUM_POR_DEFECTO)), os.path.basename(path)
    return RHUM_POR_DEFECTO, "no encontrado"


def espectro_de_la_primaria(cube, radius):
    """(wave, flux, (y, x), cabecera) de la primaria, localizándola en ESTE cubo.

    La posición no se hereda: los cubos por exposición no comparten marco con el
    combinado, porque los offsets de alineado son posteriores a ellos.
    """
    with fits.open(cube, memmap=True) as hdul:
        hdu = hdul["DATA"] if "DATA" in hdul else hdul[1]
        blanco = np.nanmedian(np.asarray(hdu.data[::37], dtype=np.float64), axis=0)
        pico = np.unravel_index(np.nanargmax(blanco), blanco.shape)
        y0, x0 = int(pico[0]), int(pico[1])
        sy = slice(max(0, y0 - 3), y0 + 4)
        sx = slice(max(0, x0 - 3), x0 + 4)
        caja = np.nan_to_num(blanco[sy, sx])
        yy, xx = np.mgrid[sy.start:sy.start + caja.shape[0], sx.start:sx.start + caja.shape[1]]
        cy = float((yy * caja).sum() / caja.sum())
        cx = float((xx * caja).sum() / caja.sum())
        half = 30
        ny, nx = hdu.shape[1:]
        a, b = max(0, int(cy) - half), min(ny, int(cy) + half + 1)
        c, d = max(0, int(cx) - half), min(nx, int(cx) + half + 1)
        ventana = np.asarray(hdu.data[:, a:b, c:d], dtype=np.float64)
        wave = wavelength_axis_from_header(hdu.header, hdu.shape[0])
        if not np.all(np.isfinite(wave)) or wave[0] == 0:
            wave = wavelength_axis_from_header(hdul[0].header, hdu.shape[0])
        cabecera = hdul[0].header.copy()
    return wave, extract_aperture_spectrum(ventana, (cy - a, cx - c), radius), (cy, cx), cabecera


def escribe_science(wave, flux, cabecera, destino):
    """El `science.fits` que espera molecfit, con las tarjetas que exige."""
    pri = fits.Header()
    for clave in ("MJD-OBS", "DATE-OBS", "UTC", "EXPTIME"):
        if cabecera.get(clave) is not None:
            pri[clave] = cabecera.get(clave)
    for clave in ("HIERARCH ESO TEL GEOLAT", "HIERARCH ESO TEL GEOLON",
                  "HIERARCH ESO TEL GEOELEV", "HIERARCH ESO TEL AMBI TEMP",
                  "HIERARCH ESO TEL AMBI PRES START", "HIERARCH ESO TEL TH M1 TEMP"):
        if cabecera.get(clave) is not None:
            pri[clave] = cabecera.get(clave)
    # MUSE es IFU y no tiene rendija: 0.2 es el valor que usó A1a y se mantiene
    # idéntico para que los ajustes sean comparables entre sí y con aquel.
    pri["HIERARCH ESO INS SLIT1 WID"] = 0.2
    columnas = [fits.Column(name="WAVE", format="D", array=np.asarray(wave, dtype=np.float64)),
                fits.Column(name="FLUX", format="D", array=np.asarray(flux, dtype=np.float64))]
    fits.HDUList([fits.PrimaryHDU(header=pri),
                  fits.BinTableHDU.from_columns(columnas)]).writeto(destino, overwrite=True)


def ajusta(etiqueta, cube, out_root, radius, raw_dir, reutiliza, prep_dir):
    destino = Path(out_root) / etiqueta
    (destino / "out").mkdir(parents=True, exist_ok=True)
    inicio = time.time()

    wave, flux, yx, cabecera = espectro_de_la_primaria(cube, radius)
    x1 = cabecera.get("HIERARCH ESO TEL AIRM START")
    x2 = cabecera.get("HIERARCH ESO TEL AIRM END")
    if x1 is None or x2 is None:
        # El combinado en streaming no escribe condiciones de observación: sin
        # ellas no hay masa de aire que pasar, y eso es un RESULTADO, no un fallo
        # del script (§5.2b del documento). Se declara y se sigue a cenit.
        x_mid, altitud = float("nan"), 90.0
        aviso = "el cubo no declara AIRM: se ajusta a cenit (X=1)"
    else:
        x_mid = 0.5 * (float(x1) + float(x2))
        altitud = float(np.degrees(np.arcsin(1.0 / x_mid)))
        aviso = ""
    humedad, crudo = humedad_del_crudo(cabecera.get("DATE-OBS"), raw_dir)

    science = destino / "science.fits"
    normalizado = destino / "science_norm.fits"
    best = destino / "out" / "BEST_FIT_PARAMETERS.fits"
    if reutiliza and best.exists():
        codigo, segundos = 0, 0.0
    else:
        escribe_science(wave, flux, cabecera, science)
        subprocess.run([sys.executable, str(Path(prep_dir) / PREP), str(science),
                        str(normalizado), str(destino / "wave_include_3win.fits")],
                       check=True, capture_output=True)
        (destino / "model.sof").write_text(
            f"{normalizado} SCIENCE\n{Path(prep_dir) / WAVE_INCLUDE} WAVE_INCLUDE\n", encoding="utf-8")
        orden = [
            "esorex", f"--output-dir={destino / 'out'}", "molecfit_model",
            "--LIST_MOLEC=O2,H2O", "--FIT_MOLEC=1,0", "--REL_COL=1.0,1.0",
            "--COLUMN_LAMBDA=WAVE", "--COLUMN_FLUX=FLUX", "--WLG_TO_MICRON=0.0001",
            "--WAVELENGTH_FRAME=AIR", "--FIT_CONTINUUM=1", "--CONTINUUM_N=1",
            "--FIT_WLC=0", "--WLC_CONST=0",          # <- sin esto el ajuste se congela
            "--TELESCOPE_ANGLE_KEYWORD=NONE", f"--TELESCOPE_ANGLE_VALUE={altitud:.4f}",
            "--RELATIVE_HUMIDITY_KEYWORD=NONE", f"--RELATIVE_HUMIDITY_VALUE={humedad}",
            str(destino / "model.sof"),
        ]
        (destino / "esorex.cmd").write_text(" ".join(orden) + "\n", encoding="utf-8")
        with open(destino / "run.log", "w", encoding="utf-8") as log:
            codigo = subprocess.run(orden, stdout=log, stderr=subprocess.STDOUT, cwd=destino).returncode
        segundos = round(time.time() - inicio, 1)

    fila = {
        "etiqueta": etiqueta, "cubo": str(cube), "date_obs": str(cabecera.get("DATE-OBS")),
        "exptime": cabecera.get("EXPTIME"), "yx": [round(v, 2) for v in yx], "radio_px": radius,
        "X_start": x1, "X_end": x2, "X_mid": None if np.isnan(x_mid) else round(x_mid, 4),
        "altitud_deg": round(altitud, 4), "humedad_pct": humedad, "crudo": crudo,
        "aviso": aviso, "rc": codigo, "segundos": segundos,
    }
    if best.exists():
        tabla = fits.getdata(best, 1)
        nombre, valor = tabla.columns.names[:2]
        ajuste = {str(r[nombre]).strip(): float(r[valor]) for r in tabla}
        fila.update({k: ajuste.get(k) for k in
                     ("status", "iterations", "initial_chi2", "best_chi2", "reduced_chi2",
                      "rel_mol_col_O2", "ppmv_O2", "h2o_col_mm")})
    return fila


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", required=True, help="directorio de salida")
    parser.add_argument("--combined", required=True, help="cubo combinado (una granularidad)")
    parser.add_argument("--exposures", nargs="+", required=True, help="cubos por exposición (la otra)")
    parser.add_argument("--radius", type=float, default=6.0, help="radio de la apertura en px")
    parser.add_argument("--raw-dir", default=None, help="crudos, para leer la humedad real")
    parser.add_argument("--prep-dir", required=True,
                        help="carpeta molecfit_a1a de la reduccion historica: prep.py + wave_include2.fits")
    parser.add_argument("--reuse", action="store_true",
                        help="no re-ajustar lo que ya tenga BEST_FIT_PARAMETERS")
    args = parser.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    trabajos = [("combinado", args.combined)]
    trabajos += [(f"exp{i}", c) for i, c in enumerate(args.exposures, start=1)]

    filas, curvas = [], {}
    for etiqueta, cube in trabajos:
        fila = ajusta(etiqueta, cube, out, args.radius, args.raw_dir, args.reuse, args.prep_dir)
        filas.append(fila)
        print(json.dumps(fila, ensure_ascii=False), flush=True)
        datos = Path(out) / etiqueta / "out" / "MOLECFIT_DATA.fits"
        if datos.exists():
            tabla = fits.getdata(datos, 1)
            # OJO CON EL MARCO. `MOLECFIT_DATA.lambda` sale en VACIO: comprobado el
            # 2026-08-06 contra Edlen(1966) sobre el eje de entrada, coincide a
            # 0.00000 A. A 7600 A eso son 2.09 A = 1.67 canales de corrimiento al
            # rojo respecto al eje de MUSE, que es AIRE — y todas las bandas de A3
            # estan definidas en aire. Guardar esa lambda como `wave_A` seria
            # sembrar un error de ~1.7 canales en cualquier mascara por banda.
            #
            # Las filas de la salida estan en correspondencia 1:1 con las de la
            # entrada, asi que el eje bueno es el del propio espectro: exacto, no
            # aproximado. El de vacio se guarda aparte, por procedencia.
            science = Path(out) / etiqueta / "science.fits"
            if "wave_A" not in curvas and science.exists():
                with fits.open(science) as h:
                    curvas["wave_A"] = np.asarray(h[1].data["WAVE"], dtype=np.float64)
            curvas.setdefault("wave_vac_A", np.asarray(tabla["lambda"], dtype=np.float64) * 1e4)
            curvas[f"mtrans_{etiqueta}"] = np.asarray(tabla["mtrans"], dtype=np.float64)
            curvas[f"mrange_{etiqueta}"] = np.asarray(tabla["mrange"], dtype=np.float64)

    (out / "manifest.json").write_text(json.dumps(filas, indent=1, ensure_ascii=False) + "\n",
                                       encoding="utf-8")
    if curvas:
        np.savez_compressed(out / "curvas.npz", **curvas)
    hechos = sum(1 for f in filas if f.get("status") is not None)
    print(f"\n{hechos}/{len(filas)} ajustes con producto. manifest -> {out / 'manifest.json'}")
    return 0 if hechos == len(filas) else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
