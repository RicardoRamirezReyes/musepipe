#!/usr/bin/env python
"""Cuanto sesga al espectro de C4 (`psffit`) modelar la primaria como UNA estrella.

C1 ya ajusta **dos componentes ligadas** cuando la primaria es una binaria no
resuelta, pero la extraccion no: `psf_pair_design` arma DOS columnas —estrella y
compañero—, ambas fuentes puntuales, asi que la secundaria de ROXs 42B (51 mas =
2.02 px, DENTRO del nucleo) no tiene columna propia y cae en el residuo. El paper
lo declara sin cifra ("a bias that has not been quantified"); esto la mide.

**No toca ningun producto.** Lee lo mismo que la etapa X03, repite el ajuste con
tres modelos de la columna «estrella» y compara los espectros del COMPAÑERO:

  base    5 columnas: [P(estrella), P(compañero), 1, y, x]   <- lo que corre hoy
  ligada  la columna 0 pasa a P(estrella) + f(lambda)*P(estrella+offset), con
          f interpolada de los 43 bins de C1 (`stage_e01_psfao_params.csv`).
          CERO grados de libertad nuevos: es el mismo criterio que C1.
  libre   6 columnas: la secundaria entra con amplitud propia por canal.

La geometria se declara, no se ajusta: sale de `e01_binary_companion` con la
misma conversion que C1 (`binary_offset_px`, Norte=+y / Este=-x).

El control es ROXs 12 b, que es una estrella sola: se le impone la MISMA binaria
falsa (`--fake-binary`). Si alli el espectro se mueve igual, lo medido no es la
secundaria sino el suelo del metodo.

Ejemplos:
    python scripts/binaria_psffit_probe.py --run-id ROXs42Bb_realigned
    python scripts/binaria_psffit_probe.py --run-id ROXs12b_realigned \
        --fake-binary 51,148 --out /tmp/control.json
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
import sys  # noqa: E402

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from musepipe.extraction.psffit import _fit_one_channel, fit_region_mask, psf_pair_design
from musepipe.psf import _psfao_wave_bin_A, evaluate_psf_model
from musepipe.stages.stage_e01_psf import binary_offset_px
from musepipe.stages.stage_x01_aperture import _load_positions, _load_stat_cube
from musepipe.stages.stage_x02_optimal import _load_stage02_cube, _stat_metadata
from musepipe.stages.stage_x03_psffit import (
    _load_psf_model,
    stage_x03_config_from_run,
    stage_x03_paths,
)

HALPHA_A = (6555.0, 6575.0)


def flux_ratio_table(stage_dir):
    """Los 43 `f` de C1, tal cual estan en el CSV de psfao (bin -> razon)."""
    import csv

    path = Path(stage_dir) / "stage_e01_psfao_params.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path}: hace falta el CSV de C1 para la variante ligada.")
    lam, ratio = [], []
    with open(path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            try:
                f = float(row["flux_ratio"])
            except (KeyError, TypeError, ValueError):
                continue
            if np.isfinite(f):
                lam.append(float(row["lambda_A"]))
                ratio.append(f)
    if not lam:
        raise RuntimeError(f"{path} no lleva `flux_ratio` finito en ninguna fila.")
    order = np.argsort(lam)
    return np.asarray(lam)[order], np.asarray(ratio)[order]


def core_fwhm(cube, wave, star_yx, bands=((5000, 5300), (6500, 6800), (8800, 9100))):
    """FWHM del nucleo de la primaria por bandas, del perfil radial de la mediana.

    Es la medida que sostiene la frase del paper: la binaria a 51 mas cae por
    DEBAJO de un elemento de resolucion, asi que ninguna reduccion la separa.
    Se mide sobre el mismo cubo que ajusta la etapa, no sobre otro.
    """

    out = {}
    yy, xx = np.indices(cube.shape[1:], dtype=np.float64)
    for lo, hi in bands:
        sel = (wave >= lo) & (wave < hi)
        if not np.any(sel):
            continue
        img = np.nanmedian(cube[sel], axis=0)
        cy, cx = float(star_yx[0]), float(star_yx[1])
        rr = np.hypot(yy - cy, xx - cx)
        peak = np.nanmax(img[rr <= 1.0])
        edges = np.arange(0.0, 12.0, 0.5)
        prof = np.array([np.nanmedian(img[(rr >= b) & (rr < b + 0.5)]) for b in edges])
        k = int(np.argmax(prof < 0.5 * peak))
        if k == 0:
            continue
        r_half = float(np.interp(0.5 * peak, prof[: k + 1][::-1], (edges[: k + 1] + 0.25)[::-1]))
        out[f"{int(0.5*(lo+hi))}A"] = 2.0 * r_half
    return out


def snapped_wave(psf_model, wave_A):
    """El λ con el que el modelo construye la imagen (rejilla de `wave_bin`).

    `evaluate_psf_model` redondea λ a esa rejilla antes de construir la PSF y
    cachea la imagen; la única dependencia con λ es esa. Por eso las columnas de
    PSF se pueden cachear por λ ajustado sin cambiar un solo bit: 3681 canales
    se vuelven ~46 construcciones. `--verify-cache` lo comprueba.
    """

    wave_bin = _psfao_wave_bin_A(psf_model)
    if not wave_bin or wave_bin <= 0:
        return float(wave_A)
    return round(float(wave_A) / wave_bin) * wave_bin


def psf_columns(shape, wave, star_yx, comp_yx, psf_model, offset_yx):
    """La matriz de diseño de la etapa (5 columnas) y la PSF de la secundaria."""
    base = psf_pair_design(shape, wave, star_yx, comp_yx, psf_model)
    yy, xx = np.indices(shape, dtype=np.float64)
    p2 = evaluate_psf_model(
        psf_model,
        float(wave),
        yy - (float(star_yx[0]) + float(offset_yx[0])),
        xx - (float(star_yx[1]) + float(offset_yx[1])),
    )
    return base, p2


def design_variants(base, p2, f_lambda):
    """(base, ligada, libre). `base` es la de la etapa, sin tocar."""
    ligada = base.copy()
    ligada[..., 0] = base[..., 0] + float(f_lambda) * p2
    libre = np.concatenate([base, p2[..., None]], axis=-1)
    return base, ligada, libre


def run_probe(run_id, project_root, *, fake_binary=None, step=1, n_channels=None,
              verify_cache=0, flux_ratio_from=None, fwhm_only=False):
    cfg = stage_x03_config_from_run(run_id, project_root=project_root)
    root = Path(cfg["project_root"])
    paths = stage_x03_paths(cfg["run_id"], root)
    comp_yx, star_yx, _p, positions_qc = _load_positions(
        paths, {"x01_source_key": cfg["x03_source_key"], "x01_star_key": cfg["x03_star_key"]}
    )
    psf_model, psf_path = _load_psf_model(paths, cfg)
    cube, wave, cube_path, _bunit = _load_stage02_cube(paths, cfg)
    stat_cube, stat_state = _load_stat_cube(paths, cfg, cube.shape)
    stat_factor, _cov_factor, _stat_status, _qc00, _qc01 = _stat_metadata(paths, cfg)

    probe_cfg = dict(cfg)
    if fake_binary is not None:
        sep_mas, pa_deg = fake_binary
        probe_cfg.pop("e01_binary_offset_yx_px", None)
        probe_cfg["e01_binary_companion"] = {"sep_mas": sep_mas, "pa_deg": pa_deg}
    offset = binary_offset_px(probe_cfg, positions_qc)
    if offset is None:
        raise RuntimeError(
            f"{cfg['run_id']} no declara `e01_binary_companion`; usa --fake-binary SEP,PA "
            "para imponerle una (es lo que se hace en el control)."
        )

    # La variante ligada necesita un f(lambda). El control es una estrella SOLA:
    # C1 nunca ajusto ahi una razon de flujos, asi que se le impone la del
    # objeto con binaria. Es justo lo que se quiere medir: cuanto mueve el
    # espectro imponer una secundaria que no existe.
    ratio_dir = paths["paths"].stage_dir
    if flux_ratio_from:
        ratio_dir = stage_x03_paths(flux_ratio_from, root)["paths"].stage_dir
    lam, ratio = flux_ratio_table(ratio_dir)

    variance = np.asarray(stat_cube, dtype=np.float64) if stat_cube is not None else None
    if variance is None:
        from musepipe.extraction.optimal import estimate_variance_cube

        variance = estimate_variance_cube(cube)
    variance = variance * (stat_factor if np.isfinite(stat_factor) and stat_factor > 0 else 1.0)

    nz, ny, nx = cube.shape
    mask = fit_region_mask(
        (ny, nx),
        star_yx,
        comp_yx,
        star_radius_px=float(cfg["x03_star_radius_px"]),
        comp_radius_px=float(cfg["x03_comp_radius_px"]),
    )
    escala_mas = 1000.0 * float(positions_qc.get("pixel_scale_arcsec", np.nan))
    fwhm_px = core_fwhm(cube, wave, star_yx)
    fwhm = {
        "px": {k: round(v, 3) for k, v in fwhm_px.items()},
        "mas": {k: round(v * escala_mas, 1) for k, v in fwhm_px.items()},
        "separacion_binaria_px": float(np.hypot(*offset)),
        "separacion_en_FWHM": {k: round(float(np.hypot(*offset)) / v, 3) for k, v in fwhm_px.items()},
    }
    if fwhm_only:
        return {"run_id": cfg["run_id"], "cube": str(cube_path), "core_fwhm": fwhm}, None

    channels = np.arange(0, nz, int(step))
    if n_channels:
        channels = channels[: int(n_channels)]

    out = {k: np.full(channels.size, np.nan) for k in
           ("comp_base", "comp_ligada", "comp_libre", "star_base", "star_ligada",
            "chi2_base", "chi2_ligada", "chi2_libre", "f_used", "f_libre",
            "err_base")}
    t0 = time.time()
    cache = {}
    for i, z in enumerate(channels):
        f = float(np.interp(wave[z], lam, ratio))
        key = snapped_wave(psf_model, wave[z])
        if key not in cache:
            cache[key] = psf_columns((ny, nx), wave[z], star_yx, comp_yx, psf_model, offset)
        base_cols, p2 = cache[key]
        if verify_cache and i < int(verify_cache):
            fresh_base, fresh_p2 = psf_columns((ny, nx), wave[z], star_yx, comp_yx, psf_model, offset)
            if not (np.array_equal(fresh_base, base_cols, equal_nan=True)
                    and np.array_equal(fresh_p2, p2, equal_nan=True)):
                raise RuntimeError(
                    f"La cache por λ ajustado NO es exacta en {wave[z]:.2f} A: no la uses."
                )
        d_base, d_lig, d_lib = design_variants(base_cols, p2, f)
        cb, covb, chi_b, *_ = _fit_one_channel(cube[z], variance[z], d_base, mask)
        cl, _covl, chi_l, *_ = _fit_one_channel(cube[z], variance[z], d_lig, mask)
        cf, _covf, chi_f, *_ = _fit_one_channel(cube[z], variance[z], d_lib, mask)
        out["comp_base"][i], out["comp_ligada"][i], out["comp_libre"][i] = cb[1], cl[1], cf[1]
        out["star_base"][i], out["star_ligada"][i] = cb[0], cl[0]
        out["chi2_base"][i], out["chi2_ligada"][i], out["chi2_libre"][i] = chi_b, chi_l, chi_f
        out["f_used"][i] = f
        out["f_libre"][i] = cf[5] / cf[0] if np.isfinite(cf[0]) and cf[0] != 0 else np.nan
        out["err_base"][i] = np.sqrt(covb[1, 1]) if np.isfinite(covb[1, 1]) and covb[1, 1] > 0 else np.nan
        if i and i % 200 == 0:
            done = (i + 1) / channels.size
            print(f"  {i+1}/{channels.size} canales ({100*done:.0f} %), "
                  f"{time.time()-t0:.0f} s, ETA {(time.time()-t0)*(1/done - 1):.0f} s", flush=True)

    out["wave_A"] = wave[channels]
    meta = {
        "run_id": cfg["run_id"],
        "cube": str(cube_path),
        "psf_model": str(psf_path),
        "psf_form": psf_model.get("form"),
        "star_yx": [float(star_yx[0]), float(star_yx[1])],
        "comp_yx": [float(comp_yx[0]), float(comp_yx[1])],
        "binary_offset_yx_px": [float(offset[0]), float(offset[1])],
        "binary_declared": probe_cfg.get("e01_binary_companion"),
        "fake_binary": bool(fake_binary is not None),
        "flux_ratio_from": str(flux_ratio_from) if flux_ratio_from else cfg["run_id"],
        "stat_state": stat_state,
        "stat_factor": float(stat_factor),
        "n_channels": int(channels.size),
        "channel_step": int(step),
        "psf_wave_bin_A": float(_psfao_wave_bin_A(psf_model) or 0.0),
        "n_psf_builds": int(len(cache)),
        "verify_cache_channels": int(verify_cache),
        "seconds": round(time.time() - t0, 1),
        "core_fwhm": fwhm,
    }
    return meta, out


def summarize(meta, out):
    wave = out["wave_A"]
    base = out["comp_base"]
    res = {"meta": meta, "bands": {}, "chi2": {}, "flux_ratio": {}}
    ok = np.isfinite(base) & (np.abs(base) > 0)
    bands = {
        "all": (wave.min() - 1, wave.max() + 1),
        "blue_lt6000": (wave.min() - 1, 6000.0),
        "red_gt8000": (8000.0, wave.max() + 1),
        "halpha": HALPHA_A,
    }
    for name, variant in (("ligada", out["comp_ligada"]), ("libre", out["comp_libre"])):
        entry = {}
        for band, (lo, hi) in bands.items():
            sel = ok & (wave >= lo) & (wave < hi) & np.isfinite(variant)
            if not np.any(sel):
                entry[band] = None
                continue
            rel = 100.0 * (variant[sel] - base[sel]) / base[sel]
            diff_sigma = (variant[sel] - base[sel]) / out["err_base"][sel]
            entry[band] = {
                "n": int(sel.sum()),
                "median_rel_pct": float(np.nanmedian(rel)),
                "p16_p84_rel_pct": [float(np.nanpercentile(rel, 16)), float(np.nanpercentile(rel, 84))],
                "median_shift_sigma": float(np.nanmedian(diff_sigma)),
            }
        res["bands"][name] = entry
    for name in ("base", "ligada", "libre"):
        arr = out[f"chi2_{name}"]
        res["chi2"][name] = float(np.nanmedian(arr))
    res["flux_ratio"] = {
        "c1_interpolated_median": float(np.nanmedian(out["f_used"])),
        "free_fit_median": float(np.nanmedian(out["f_libre"])),
        "free_fit_blue_lt6000": float(np.nanmedian(out["f_libre"][wave < 6000])),
        "free_fit_red_gt8000": float(np.nanmedian(out["f_libre"][wave > 8000])),
    }
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--project-root", default=None)
    ap.add_argument("--fake-binary", default=None,
                    help="SEP_MAS,PA_DEG a imponer (control sobre una estrella sola)")
    ap.add_argument("--step", type=int, default=1, help="submuestreo de canales (1 = todos)")
    ap.add_argument("--n-channels", type=int, default=None, help="corta tras N canales (prueba)")
    ap.add_argument("--flux-ratio-from", default=None,
                    help="run del que sale el f(lambda) de C1 (control: el del objeto binario)")
    ap.add_argument("--fwhm-only", action="store_true",
                    help="solo mide el nucleo de la PSF y sale (segundos, sin ajustes)")
    ap.add_argument("--verify-cache", type=int, default=3,
                    help="canales en los que se recomputa la PSF para comprobar la cache")
    ap.add_argument("--out", default=None, help="JSON de salida (por defecto, junto al npz)")
    ap.add_argument("--npz", default=None, help="npz con los espectros por canal")
    args = ap.parse_args(argv)

    fake = None
    if args.fake_binary:
        sep, pa = args.fake_binary.split(",")
        fake = (float(sep), float(pa))

    meta, out = run_probe(args.run_id, args.project_root, fake_binary=fake,
                          step=args.step, n_channels=args.n_channels,
                          verify_cache=args.verify_cache,
                          flux_ratio_from=args.flux_ratio_from,
                          fwhm_only=args.fwhm_only)
    if args.fwhm_only:
        print(json.dumps(meta, indent=2))
        if args.out:
            Path(args.out).write_text(json.dumps(meta, indent=2), encoding="utf-8")
        return meta
    res = summarize(meta, out)
    print(json.dumps(res, indent=2))
    if args.npz:
        np.savez(args.npz, **out)
    if args.out:
        Path(args.out).write_text(json.dumps(res, indent=2), encoding="utf-8")
        print(f"escrito {args.out}")
    return res


if __name__ == "__main__":
    main()
