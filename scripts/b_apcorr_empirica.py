"""B (2026-09-10): la apcorr del MODELO contra la apcorr del DATO, y su efecto en D1.

Contexto. C1 V4 falla en los dos objetos: el cociente `F(<=25px)/F(box3)` del
modelo se desvia de el del dato (+6.0 % de mediana en ROXs 12 b, p90 15.8 %), y
ESE cociente es exactamente la correccion de apertura que C2/C3 aplican. C2 lo
denuncia por su lado: su V4(b) mide box5/box3 = 0.904 tras corregir, en
7500-9000 A. Y el D1 fresco (A, hoy) pone la divergencia en B5+B6,
7600-9100 A. Los tres senalan el rojo.

Que hace. Mide `core_ratio_data(lambda)` sobre la PRIMARIA, sin modelo, en los
mismos 43 bins de C1; lo divide por el `core_ratio_model(lambda)` que C2 usa de
verdad (`1/fractions` de `aperture_correction_from_psf`, en la posicion subpixel
real); y reescala el espectro de apertura y sus 33 controles por ese cociente.
La apcorr entra MULTIPLICANDO canal a canal (`flux = raw * apcorr`), asi que el
reescalado es exacto: aisla la palanca sin re-extraer nada.

PREDICCION, declarada antes de mirar (falsable):
  1. Corregir la apertura DEBE mover `optimal_ls_vs_aperture` en B6 (hoy -65.4).
  2. NO debe arreglar `psffit_vs_optimal_ls` (hoy +14.3 en B6): ese par no
     comparte la apertura. Si tambien se arregla, la causa no es la apcorr.
  3. box5/box3 (V4(b) de C2, hoy 0.904) debe acercarse a 1 SOLO si la curva de
     crecimiento empirica de la primaria transfiere a la compañera. No es
     automatico: F(<=25) es comun, pero Fbox3 y Fbox5 son del nucleo de cada
     fuente. Es la prueba fisica de la transferencia.

Lee del run canonico (solo lectura) y escribe en la copia.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from astropy.io import fits

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from musepipe.extraction.aperture import aperture_correction_from_psf
from musepipe.growth_curve import factor_at_wavelengths
from musepipe.psf import core_to_norm_ratio, source_mask
from musepipe.stages.stage_e01_psf import _load_cube
from musepipe.stages.stage_e01_psfao import make_bins
from musepipe.stages.stage_e01_psf import stage_e01_config_from_run
from musepipe.stages.stage_x01_aperture import stage_x01_config_from_run


def _psfao_backgrounds(csv_path):
    """El fondo AJUSTADO por bin (columna `bck`), que es el que usa V4."""
    import csv as _csv
    out = {}
    with open(csv_path, newline="") as fh:
        for row in _csv.DictReader(fh):
            try:
                out[float(row["lambda_A"])] = float(row["bck"])
            except (TypeError, ValueError):
                continue
    return out


def _nearest_bck(table, lam):
    if not table:
        return 0.0
    key = min(table, key=lambda k: abs(k - lam))
    if abs(key - lam) > 1.0:
        raise RuntimeError(f"sin fondo psfao para lambda={lam} (mas cercano {key})")
    return table[key]

BANDS = {"B1": (4900, 5400), "B2": (5450, 5750), "B3": (6100, 6400),
         "B4": (6600, 6800), "B5": (7600, 8000), "B6": (8600, 9100)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-run", required=True,
                    help="run canonico del que se LEE (no se escribe en el)")
    ap.add_argument("--dest-run", required=True, help="run copy to write into")
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--apply", action="store_true",
                    help="write the rescaled products; without it only measures")
    args = ap.parse_args()

    root = Path.cwd()
    src = root / "runs" / args.src_run / "stages"
    dst = root / "runs" / args.dest_run / "stages"
    cfg = stage_x01_config_from_run(args.src_run)
    cfg_e01 = stage_e01_config_from_run(args.src_run)
    e01 = json.loads((src / "stage_e01_qc.json").read_text())
    pos = json.loads((src / "stage01c_qc.json").read_text())
    psf_model = json.loads((src / "psf_model.json").read_text())
    growth = json.loads((src / "growth_curve_qc.json").read_text())

    primary_yx = tuple(float(v) for v in pos["primary"]["pos_yx"])
    companion_yx = tuple(float(v) for v in pos["companion"]["pos_yx"])
    field = pos.get("field_source")
    field_yx = tuple(float(v) for v in field["pos_yx"]) if field else None
    norm_radius = float(psf_model.get("norm_radius_px", 25.0))

    # El cubo de C1 es una PILA (N, nz, ny, nx) en la extension CUBES, y su
    # lambda viaja con el: leerlo con el cargador de C1 y no por HDU 0.
    cubes, wave, _stat = _load_cube(Path(e01["input"]["cube"]))
    cube = cubes[0] if cubes.ndim == 4 else cubes
    print(f"cube {cube.shape}  wave {wave.shape}  primary {primary_yx}")

    # --- los MISMOS bins de C1 -------------------------------------------
    bin_A = float(e01["binning"]["bin_A"])
    bad = [tuple(w) for w in e01["binning"].get("excluded_windows_A") or []]
    # La forma elegida es psfao (`e01_psf_form`), y la V4 del QC se mide sobre
    # SUS bins y SU fondo ajustado -no el `corner_background` de la rama Moffat,
    # ni los bins de `make_psf_bins`, que dan 44 porque no descartan por el
    # punto medio. Reproducirlo es la puerta 2.
    bins = make_bins(wave, bin_A, bad, int(cfg_e01.get("psf_min_channels_per_bin", 3)))
    assert len(bins) == int(e01["binning"]["n_bins"]), (
        f"bins {len(bins)} != C1 {e01['binning']['n_bins']}")
    bck_by_lam = _psfao_backgrounds(src / "stage_e01_psfao_params.csv")

    # Mascara de exclusion como la de V4: compañera (+ campo) al radio de C1.
    excl = float(e01["masks"]["companion_radius_px"])
    centers = [companion_yx] + ([field_yx] if field_yx else [])
    ee_mask = source_mask(cubes.shape[2:], centers, excl)

    # --- lado DATO: sin modelo -------------------------------------------
    rows = []
    images = []
    for (lo_b, hi_b, mid, sel) in bins:
        img = np.nanmedian(cube[sel], axis=0)
        images.append(img)
        bkg = _nearest_bck(bck_by_lam, float(mid))
        r_data = core_to_norm_ratio(img, primary_yx, norm_radius_px=norm_radius,
                                    box_half=1, background=bkg, exclude_mask=ee_mask)
        rows.append({"lambda_A": float(mid), "core_ratio_data": float(r_data),
                     "background": float(bkg), "n_channels": int(sel.sum())})
    lam = np.array([r["lambda_A"] for r in rows])
    cr_data = np.array([r["core_ratio_data"] for r in rows])

    # --- lado MODELO: exactamente lo que C2 invierte ---------------------
    # El cociente se mide con la fase EMPAREJADA: dato y modelo, los dos en la
    # primaria, que es la unica fuente puntual brillante donde `core_ratio` se
    # puede medir sin modelo. Asi el cociente aisla el error de PERFIL del
    # modelo, no la fase subpixel.
    #
    # SUPUESTO, declarado: el error del modelo factoriza en
    # (fase subpixel) x (perfil cromatico), asi que multiplicar la apcorr de la
    # compañera -que lleva SU fase- por este cociente corrige el perfil y deja la
    # fase intacta. La prueba independiente del supuesto es la V4(b) de C2:
    # box5/box3, que tiene otra sensibilidad a la fase.
    box3 = next(a for a in cfg["x01_apertures"] if a["name"] == "box3")
    box5 = next(a for a in cfg["x01_apertures"] if a["name"] == "box5")
    apc_nofac_bins, _, _ = aperture_correction_from_psf(
        lam, box3, psf_model, center_yx=primary_yx, growth_curve=None)
    cr_model = np.asarray(apc_nofac_bins, dtype=np.float64)

    err_pct = 100.0 * (cr_model / cr_data - 1.0)
    med_err = float(np.nanmedian(err_pct))
    p90_err = float(np.nanpercentile(np.abs(err_pct), 90))

    # --- PUERTA 1: reproducir la apcorr guardada, en los 3681 canales ----
    # Es la validacion fuerte: si `1/fractions * factor` no reproduce la columna
    # `apcorr` del producto, entonces no estoy tocando la palanca que creo.
    with fits.open(src / "spec_aperture_object.fits") as hdul:
        tab = hdul[1].data
        hdr1 = hdul[1].header
        wave_prod = np.asarray(tab["wave_A"], dtype=np.float64)
        apcorr_prod = np.asarray(tab["apcorr"], dtype=np.float64)
    # C2 evalua la apcorr en la posicion subpixel de la COMPAÑERA
    # (`center_yx=object_yx` en extraction/aperture.py), no de la primaria: con
    # FWHM ~3 px, box3 es tan pequeña que la fase subpixel cambia la fraccion un
    # 11 %. Usar la primaria aqui hacia fallar esta puerta.
    apc_full, mode_full, _ = aperture_correction_from_psf(
        wave_prod, box3, psf_model, center_yx=companion_yx, growth_curve=growth)
    rel = np.abs(apc_full / apcorr_prod - 1.0)
    gate1 = {"max_rel_diff": float(np.nanmax(rel)),
             "median_rel_diff": float(np.nanmedian(rel)),
             "n_channels": int(wave_prod.size), "mode": mode_full,
             "mode_in_product": hdr1.get("APCMODE"),
             "ok": bool(np.nanmax(rel) < 1e-9)}
    print(f"PUERTA 1 reproducir apcorr del producto: max|rel|={gate1['max_rel_diff']:.3e} "
          f"ok={gate1['ok']}  ({gate1['n_channels']} canales)")

    # --- PUERTA 2: reproducir las medianas de V4 del QC de C1 ------------
    # No tiene que cuadrar al bit: C1 mide el modelo sobre la reconstruccion
    # psfao en la rejilla 170x170; C2 evalua el psf_model publicado en 51x51.
    # Se declara la diferencia, no se fuerza.
    v4 = e01["encircled_energy"]
    # El lado DATO reproduce C1 al digito. El lado MODELO no tiene por que:
    # C1 mide la reconstruccion psfao en la rejilla 170x170, C2 evalua el
    # `psf_model` PUBLICADO en 51x51. Que difieran un 7.5 % en el cociente que
    # ACABA SIENDO la apcorr es un hallazgo en si, no un fallo de esta medida.
    gate2 = {"core_ratio_data_median_here": float(np.nanmedian(cr_data)),
             "core_ratio_data_median_C1": float(v4["core_ratio_data_median"]),
             "core_ratio_model_median_here": float(np.nanmedian(cr_model)),
             "core_ratio_model_median_C1": float(v4["core_ratio_model_median"]),
             "err_pct_median_here": med_err,
             "err_pct_median_C1": float(v4["core_ratio_error_pct_median"]),
             "err_pct_p90_here": p90_err,
             "err_pct_p90_C1": float(v4["core_ratio_error_pct_p90"]),
             "informative_only": True,
             "note": ("el lado DATO debe reproducir C1 al digito; el lado MODELO "
                      "difiere por construccion (reconstruccion psfao 170x170 en C1 "
                      "contra psf_model publicado 51x51 en C2)")}
    print("PUERTA 2 contra V4 de C1 (43 bins):")
    for k in ("core_ratio_data_median", "core_ratio_model_median",
              "err_pct_median", "err_pct_p90"):
        print(f"   {k:26s} aqui={gate2[k+'_here']:9.4f}  C1={gate2[k+'_C1']:9.4f}")

    # --- el cociente, por banda de D1 ------------------------------------
    ratio_bins = cr_data / cr_model
    by_band = {}
    for name, (lo, hi) in BANDS.items():
        m = (lam >= lo) & (lam <= hi)
        if m.any():
            by_band[name] = {"ratio_median": float(np.nanmedian(ratio_bins[m])),
                             "err_pct_median": float(np.nanmedian(err_pct[m])),
                             "n_bins": int(m.sum())}
    print("\ncociente dato/modelo por banda de D1:")
    for name in BANDS:
        if name in by_band:
            b = by_band[name]
            print(f"   {name} {BANDS[name]}  ratio={b['ratio_median']:.4f}  "
                  f"error del modelo={b['err_pct_median']:+6.2f} %  (n={b['n_bins']})")

    ratio_full = np.interp(wave_prod, lam, ratio_bins)
    out = {"stage": "B_apcorr_empirica", "run_src": args.src_run, "run_dst": args.dest_run,
           "prediction": {
               "1": "corregir la apertura mueve optimal_ls_vs_aperture en B6 (hoy -65.4)",
               "2": "NO arregla psffit_vs_optimal_ls (hoy +14.3 en B6); si lo arregla, la causa no es la apcorr",
               "3": "box5/box3 -> 1 solo si la curva empirica de la primaria transfiere a la compañera"},
           "gate1_reproduce_product_apcorr": gate1,
           "gate2_vs_C1_V4": gate2,
           "n_bins": len(rows), "bin_A": bin_A,
           "primary_yx": list(primary_yx), "norm_radius_px": norm_radius,
           "companion_mask_radius_px": excl,
           "err_pct_median": med_err, "err_pct_p90": p90_err,
           "by_band": by_band,
           "ratio_full_median": float(np.nanmedian(ratio_full)),
           "ratio_full_min": float(np.nanmin(ratio_full)),
           "ratio_full_max": float(np.nanmax(ratio_full)),
           "per_bin": [{**r, "core_ratio_model": float(m), "err_pct": float(e),
                        "ratio": float(q)}
                       for r, m, e, q in zip(rows, cr_model, err_pct, ratio_bins)],
           "applied": bool(args.apply)}

    if not gate1["ok"]:
        out["ABORT"] = ("PUERTA 1 falla: `1/fractions*factor` no reproduce la columna "
                        "apcorr del producto, asi que el reescalado no aisla la apcorr.")
        print("\n*** ABORTO: " + out["ABORT"])
        Path(args.out_json).write_text(json.dumps(out, indent=1, ensure_ascii=False))
        return 1

    if args.apply:
        # box3 (el producto canonico de D1) y box5 (para la V4(b) de C2).
        for tag, src_name in (("box3", "spec_aperture_object.fits"),
                              ("box5", "spec_aperture_object_box5.fits")):
            sp = src / src_name
            if not sp.exists():
                print(f"   (sin {src_name}, salto {tag})")
                continue
            with fits.open(sp) as hdul:
                hdul = fits.HDUList([h.copy() for h in hdul])
                t = hdul[1].data
                w = np.asarray(t["wave_A"], dtype=np.float64)
                if tag == "box3":
                    r = ratio_full
                else:
                    a5, _, _ = aperture_correction_from_psf(
                        lam, box5, psf_model, center_yx=primary_yx, growth_curve=None)
                    # lado dato de box5: F(<=25)/F(box5) medido igual
                    cr5 = [core_to_norm_ratio(
                        img, primary_yx, norm_radius_px=norm_radius, box_half=2,
                        background=r["background"], exclude_mask=ee_mask)
                        for img, r in zip(images, rows)]
                    r = np.interp(w, lam, np.asarray(cr5) / np.asarray(a5))
                    out["box5_ratio_median"] = float(np.nanmedian(r))
                for col in ("flux", "flux_err", "flux_err_emp", "apcorr"):
                    if col in t.columns.names:
                        t[col] = np.asarray(t[col], dtype=np.float64) * r
                hdul[1].header["APCMODE"] = "empirical_core_ratio+empirical_total"
                hdul[1].header["HISTORY"] = "B 2026-09-10: apcorr del DATO (core_ratio empirico)"
                dest = dst / src_name
                hdul.writeto(dest, overwrite=True)
                print(f"   escrito {dest.name}  (ratio mediana {float(np.nanmedian(r)):.4f})")
        # controles: `control_spectra` ya lleva la apcorr aplicada
        z = np.load(src / "spec_aperture_controls.npz", allow_pickle=True)
        d = {k: z[k] for k in z.keys()}
        d["control_spectra"] = np.asarray(d["control_spectra"], dtype=np.float64) * ratio_full[None, :]
        np.savez(dst / "spec_aperture_controls.npz", **d)
        print(f"   escrito spec_aperture_controls.npz  {d['control_spectra'].shape}")

    Path(args.out_json).write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(f"\n-> {args.out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
