#!/usr/bin/env python
"""Lo mismo que `binaria_psffit_probe.py`, pero para C1b y `optimal_psfsub`.

Las dos etapas restan la primaria modelandola como UNA fuente puntual:

  * `optimal_psfsub` (C3) resta `A(lambda) * P` del combinado y extrae con Horne
    (`fit_primary_psf_model_cube`, radio 25 px, compañero excluido a 8 px).
  * C1b hace esa MISMA resta en cada exposicion con SU modelo, y despues combina
    los residuos (`fit_primary_psf_amplitudes`).

Cuando la primaria es una binaria no resuelta el modelo restado esta mal en el
nucleo, la amplitud sale sesgada, y lo que sobra o falta es un pedestal del halo
justo debajo del compañero. Aqui se mide.

PARTE A -- `optimal_psfsub` sobre el combinado. Se resta con tres modelos y se
extrae con la misma Horne y las mismas perillas del run:

  base    A * P                      <- lo que corre hoy
  ligada  A' * (P + f(lambda)*P_off) <- f de los 43 bins de C1, cero grados nuevos
  libre   a*P + b*P_off              <- la secundaria con amplitud propia

PARTE B -- C1b, exposicion a exposicion (`--perexp N`). Ajusta las dos primeras
variantes en el marco PROPIO de cada exposicion, con el modelo de PSF de esa
exposicion (los `components` de la mezcla), y propaga la diferencia a la ventana
del compañero con el peso de Horne. Sin recombinar nada: la media pesada por
`exptime` de esas diferencias es lo que la resta le habria movido al combinado.

El control es el mismo de siempre: ROXs 12 b, estrella sola, con la binaria falsa
impuesta (`--fake-binary 51,148 --flux-ratio-from ROXs42Bb_realigned`).
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
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from binaria_psffit_probe import flux_ratio_table, snapped_wave  # noqa: E402

from musepipe.extraction.optimal import make_optimal_product, psf_image  # noqa: E402
from musepipe.stages.stage_e01_psf import binary_offset_px  # noqa: E402
from musepipe.stages.stage_x01_aperture import (  # noqa: E402
    _load_masks,
    _load_positions,
    _load_residual_cube,
    _load_stat_cube,
    _wavelength_frame,
)
from musepipe.stages.stage_x02_optimal import (  # noqa: E402
    _load_stage02_cube,
    _stat_metadata,
    stage_x02_config_from_run,
    stage_x02_paths,
)

HALPHA_A = (6555.0, 6575.0)


def _columns(shape, wave, star_yx, offset_yx, psf_model):
    """P y P desplazada a la secundaria, en el marco dado."""
    p1 = psf_image(shape, wave, star_yx, psf_model)
    p2 = psf_image(shape, wave, (star_yx[0] + offset_yx[0], star_yx[1] + offset_yx[1]), psf_model)
    return p1, p2


def _fit_amplitudes(data, variance, cols, mask):
    """Ajuste lineal ponderado identico al de `fit_primary_psf_amplitudes`."""
    valid = mask & np.isfinite(data)
    for c in cols:
        valid &= np.isfinite(c)
    if variance is not None:
        valid &= np.isfinite(variance) & (variance > 0)
        w = 1.0 / variance[valid]
    else:
        w = np.ones(int(np.count_nonzero(valid)), dtype=np.float64)
    n = int(np.count_nonzero(valid))
    if n < len(cols) + 2:
        return None
    a = np.column_stack([c[valid] for c in cols] + [np.ones(n)])
    sw = np.sqrt(w)
    try:
        coeff, *_ = np.linalg.lstsq(a * sw[:, None], data[valid] * sw, rcond=None)
    except np.linalg.LinAlgError:
        return None
    return coeff


def subtracted_cubes(cube, wave, variance, star_yx, comp_yx, psf_model, offset, lam, ratio,
                     *, fit_radius, exclude_radius, verify_cache=3):
    """Los tres cubos con la primaria restada. Devuelve (base, ligada, libre)."""
    nz, ny, nx = cube.shape
    yy, xx = np.indices((ny, nx), dtype=np.float64)
    mask = (yy - star_yx[0]) ** 2 + (xx - star_yx[1]) ** 2 <= fit_radius**2
    mask &= (yy - comp_yx[0]) ** 2 + (xx - comp_yx[1]) ** 2 > exclude_radius**2
    out = [np.empty_like(cube) for _ in range(3)]
    amps = np.full((3, nz), np.nan)
    cache = {}
    for z in range(nz):
        key = snapped_wave(psf_model, wave[z])
        if key not in cache:
            cache[key] = _columns((ny, nx), wave[z], star_yx, offset, psf_model)
        p1, p2 = cache[key]
        if verify_cache and z < int(verify_cache):
            f1, f2 = _columns((ny, nx), wave[z], star_yx, offset, psf_model)
            if not (np.array_equal(f1, p1, equal_nan=True) and np.array_equal(f2, p2, equal_nan=True)):
                raise RuntimeError(f"cache por lambda no exacta en {wave[z]:.2f} A")
        f = float(np.interp(wave[z], lam, ratio))
        var = None if variance is None else variance[z]
        c_base = _fit_amplitudes(cube[z], var, [p1], mask)
        c_lig = _fit_amplitudes(cube[z], var, [p1 + f * p2], mask)
        c_lib = _fit_amplitudes(cube[z], var, [p1, p2], mask)
        # El modelo restado NO lleva el fondo, igual que la etapa.
        out[0][z] = cube[z] - (c_base[0] * p1 if c_base is not None else 0.0)
        out[1][z] = cube[z] - (c_lig[0] * (p1 + f * p2) if c_lig is not None else 0.0)
        out[2][z] = cube[z] - ((c_lib[0] * p1 + c_lib[1] * p2) if c_lib is not None else 0.0)
        if c_base is not None:
            amps[0, z] = c_base[0]
        if c_lig is not None:
            amps[1, z] = c_lig[0]
        if c_lib is not None:
            amps[2, z] = c_lib[0]
    return out, amps


def parte_a(cfg, paths, star_yx, comp_yx, psf_model, offset, lam, ratio, *, verify_cache=3,
            sin_anillo=False):
    """`optimal_psfsub` con los tres modelos, con las perillas del run."""
    cube, wave, cube_path, bunit = _load_stage02_cube(paths, cfg)
    stat_cube, stat_state = _load_stat_cube(paths, cfg, cube.shape)
    stat_factor, cov_factor, stat_status, qc00, _ = _stat_metadata(paths, cfg)
    good_mask, bad_mask = _load_masks(paths, wave.size)
    issues = []
    wframe = _wavelength_frame(cfg, qc00, issues, knob="x02_wframe")

    variance = None if stat_cube is None else np.asarray(stat_cube, dtype=np.float64)
    t0 = time.time()
    cubos, amps = subtracted_cubes(
        cube, wave, variance, star_yx, comp_yx, psf_model, offset, lam, ratio,
        fit_radius=float(cfg.get("x02_primary_fit_radius_px", 25.0)),
        exclude_radius=float(cfg.get("x02_primary_exclude_radius_px", 8.0)),
        verify_cache=verify_cache,
    )
    espectros = {}
    for nombre, sub in zip(("base", "ligada", "libre"), cubos):
        ext = make_optimal_product(
            sub, wave, comp_yx, psf_model,
            star_yx=star_yx,
            run_id=cfg["run_id"], input_cube_path=str(cube_path), variant=f"psfsub_{nombre}",
            variance_zyx=stat_cube, stat_factor_spaxel=stat_factor,
            covariance_factor_box3=cov_factor, stat_status=stat_status,
            error_mode=cfg.get("x02_error_mode", "auto"),
            aperture_correction=cfg.get("x02_aperture_correction", "auto"),
            wframe=wframe, bunit=bunit,
            window_radius_px=float(cfg.get("x02_window_radius_px", 8.0)),
            clip_sigma=float(cfg.get("x02_clip_sigma", 4.0)),
            clip_max_iter=int(cfg.get("x02_clip_max_iter", 2)),
            good_mask=good_mask, bad_mask=bad_mask,
            n_controls=0,
            # `sin_anillo` apaga el fondo de anillo que psfsub aplica ENCIMA de
            # la resta del modelo. Es la unica diferencia con la etapa, y sirve
            # para ver cuanto del pedestal se lo come ese anillo.
            local_bkg_annulus_px=(None if (sin_anillo or bool(cfg.get("x02_wings_intact_psfsub", False)))
                                  else cfg.get("x02_local_bkg_annulus_px")),
            background_mode=cfg.get("x02_background_mode", "annulus"),
            azimuthal_width_px=float(cfg.get("x02_azimuthal_width_px", 3.0)),
            azimuthal_exclude_px=float(cfg.get("x02_azimuthal_exclude_px", 10.0)),
            plane_fit_radius_px=float(cfg.get("x02_plane_fit_radius_px", 14.0)),
            plane_mask_radius_px=float(cfg.get("x02_plane_mask_radius_px", 3.0)),
        )
        espectros[nombre] = np.asarray(ext.product.flux, dtype=np.float64)
        espectros[f"{nombre}_err"] = np.asarray(ext.product.flux_err, dtype=np.float64)
    return {
        "wave_A": wave,
        "amp_base": amps[0], "amp_ligada": amps[1], "amp_libre": amps[2],
        **espectros,
    }, {"seconds": round(time.time() - t0, 1), "cube": str(cube_path), "stat_state": stat_state}


def parte_b(cfg, paths, psf_model, offset, lam, ratio, comp_offset_yx, *, n_exposures, step):
    """C1b: la misma resta en el marco propio de cada exposicion."""
    from astropy.io import fits

    # Los dos objetos guardan el plan con nombres distintos: 42B b dejó el
    # `stream_combine_plan.json` en el run y 12 b lo lleva dentro de
    # `observation_plan.json` (clave `plan`). Se aceptan los dos.
    plan_path = paths["paths"].stage_dir / "stream_combine_plan.json"
    if plan_path.exists():
        plan = json.load(open(plan_path))
    else:
        alt = paths["paths"].stage_dir / "observation_plan.json"
        if not alt.exists():
            raise FileNotFoundError(f"{plan_path}: sin plan del combinado no hay parte B.")
        plan = json.load(open(alt))["plan"]
    modelos = {str(c["exposure_id"]): c["model"] for c in psf_model.get("components", [])}
    pesos = {str(c["exposure_id"]): float(c.get("weight", 1.0)) for c in psf_model.get("components", [])}
    expos = plan["exposures"]
    elegidas = expos[:: max(1, len(expos) // int(n_exposures))][: int(n_exposures)]

    fit_radius = float(cfg.get("x02_primary_fit_radius_px", 25.0))
    exclude_radius = float(cfg.get("x02_primary_exclude_radius_px", 8.0))
    ventana = float(cfg.get("x02_window_radius_px", 8.0))
    filas = []
    for e in elegidas:
        eid = str(e["exposure_id"])
        modelo = modelos.get(eid)
        if modelo is None:
            continue
        # El recorte tiene que contener el radio de ajuste de la primaria Y la
        # ventana del compañero, que esta a 46.6 px: con un recorte de +-40 px
        # el compañero cae FUERA y el estimador se alimenta de un trozo del
        # borde. Se dimensiona con la geometria, no con una constante.
        media = int(np.ceil(max(fit_radius,
                                abs(comp_offset_yx[0]) + ventana,
                                abs(comp_offset_yx[1]) + ventana) + 6))
        with fits.open(e["file"], memmap=True) as hdul:
            data = hdul[1]
            nz = data.shape[0]
            h = data.header
            cd = h.get("CD3_3", h.get("CDELT3"))
            wave_e = h["CRVAL3"] + cd * (np.arange(nz) + 1 - h["CRPIX3"])
            py, px = float(e["y_center"]), float(e["x_center"])
            y0, y1 = int(py - media), int(py + media + 1)
            x0, x1 = int(px - media), int(px + media + 1)
            canales = np.arange(0, nz, int(step))
            stamp = np.asarray(data.data[canales, y0:y1, x0:x1], dtype=np.float64)
            # C1b pesa el ajuste con el STAT de ESA exposicion; sin el, el
            # nucleo domina y la amplitud se mueve mucho mas de lo que se mueve
            # en la etapa.
            stat = None
            if "STAT" in hdul:
                stat = np.asarray(hdul["STAT"].data[canales, y0:y1, x0:x1], dtype=np.float64)
        sy, sx = py - y0, px - x0
        cy, cx = sy + comp_offset_yx[0], sx + comp_offset_yx[1]
        ny, nx = stamp.shape[1:]
        yy, xx = np.indices((ny, nx), dtype=np.float64)
        mask = ((yy - sy) ** 2 + (xx - sx) ** 2 <= fit_radius**2)
        mask &= ((yy - cy) ** 2 + (xx - cx) ** 2 > exclude_radius**2)
        win = (yy - cy) ** 2 + (xx - cx) ** 2 <= ventana**2
        dif, comp, base_amp, waves = [], [], [], []
        cache = {}
        for i, z in enumerate(canales):
            w = wave_e[z]
            key = snapped_wave(modelo, w)
            if key not in cache:
                cache[key] = (_columns((ny, nx), w, (sy, sx), offset, modelo),
                              psf_image((ny, nx), w, (cy, cx), modelo))
            (p1, p2), pcomp = cache[key]
            f = float(np.interp(w, lam, ratio))
            var = None if stat is None else stat[i]
            c_base = _fit_amplitudes(stamp[i], var, [p1], mask)
            c_lig = _fit_amplitudes(stamp[i], var, [p1 + f * p2], mask)
            if c_base is None or c_lig is None:
                continue
            # Lo que cambia el modelo restado, y el flujo del compañero en ESA
            # exposicion, con el mismo estimador de Horne y los mismos pesos:
            # asi el cociente no cruza escalas (ni apcorr).
            delta = c_base[0] * p1 - c_lig[0] * (p1 + f * p2)
            sub = stamp[i] - c_base[0] * p1
            wgt = np.ones_like(pcomp) if var is None else np.where(var > 0, 1.0 / var, 0.0)
            ok = win & np.isfinite(pcomp) & np.isfinite(sub) & np.isfinite(wgt)
            den = float(np.nansum((wgt * pcomp**2)[ok]))
            if den > 0:
                dif.append(float(np.nansum((wgt * pcomp * delta)[ok])) / den)
                comp.append(float(np.nansum((wgt * pcomp * sub)[ok])) / den)
                waves.append(float(w))
            base_amp.append(c_base[0])
        filas.append({
            "exposure_id": eid,
            "weight": pesos.get(eid, float(e.get("exptime", 1.0))),
            "n_channels": len(dif),
            "wave_A": waves,
            "delta_flux": dif,
            "comp_flux": comp,
            "delta_flux_median": float(np.nanmedian(dif)) if dif else None,
            "amplitude_median": float(np.nanmedian(base_amp)) if base_amp else None,
        })
    return filas


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--project-root", default=None)
    ap.add_argument("--fake-binary", default=None)
    ap.add_argument("--flux-ratio-from", default=None)
    ap.add_argument("--perexp", type=int, default=0, help="N exposiciones para la parte B (0 = no)")
    ap.add_argument("--perexp-step", type=int, default=40, help="submuestreo de canales en la parte B")
    ap.add_argument("--skip-a", action="store_true")
    ap.add_argument("--sin-anillo", action="store_true",
                    help="extrae SIN el fondo de anillo, para aislar cuanto absorbe")
    ap.add_argument("--verify-cache", type=int, default=3)
    ap.add_argument("--out", default=None)
    ap.add_argument("--npz", default=None)
    args = ap.parse_args(argv)

    cfg = stage_x02_config_from_run(args.run_id, project_root=args.project_root)
    root = Path(cfg["project_root"])
    paths = stage_x02_paths(cfg["run_id"], root)
    comp_yx, star_yx, _p, positions_qc = _load_positions(paths, cfg)
    psf_model = json.load(open(paths["psf_model_json"]))

    probe_cfg = dict(cfg)
    if args.fake_binary:
        sep, pa = (float(v) for v in args.fake_binary.split(","))
        probe_cfg.pop("e01_binary_offset_yx_px", None)
        probe_cfg["e01_binary_companion"] = {"sep_mas": sep, "pa_deg": pa}
    offset = binary_offset_px(probe_cfg, positions_qc)
    if offset is None:
        raise RuntimeError(f"{cfg['run_id']} no declara binaria; usa --fake-binary SEP,PA.")
    ratio_dir = paths["paths"].stage_dir
    if args.flux_ratio_from:
        ratio_dir = stage_x02_paths(args.flux_ratio_from, root)["paths"].stage_dir
    lam, ratio = flux_ratio_table(ratio_dir)

    res = {"meta": {
        "run_id": cfg["run_id"],
        "binary_offset_yx_px": [float(offset[0]), float(offset[1])],
        "fake_binary": bool(args.fake_binary),
        "flux_ratio_from": args.flux_ratio_from or cfg["run_id"],
        "star_yx": [float(star_yx[0]), float(star_yx[1])],
        "comp_yx": [float(comp_yx[0]), float(comp_yx[1])],
    }}
    datos = {}
    if not args.skip_a:
        datos, meta_a = parte_a(cfg, paths, star_yx, comp_yx, psf_model, offset, lam, ratio,
                                verify_cache=args.verify_cache, sin_anillo=args.sin_anillo)
        meta_a["sin_anillo"] = bool(args.sin_anillo)
        res["meta"]["parte_a"] = meta_a
        w = datos["wave_A"]
        base = datos["base"]
        res["parte_a"] = {}
        for nombre in ("ligada", "libre"):
            entrada = {}
            for etiqueta, (lo, hi) in {"7000-7500": (7000, 7500), "8000-8500": (8000, 8500),
                                       "8800-9300": (8800, 9300), "halpha": HALPHA_A}.items():
                s = (w >= lo) & (w < hi) & np.isfinite(base) & np.isfinite(datos[nombre])
                if not np.any(s):
                    continue
                b, v = np.nansum(base[s]), np.nansum(datos[nombre][s])
                entrada[etiqueta] = {"n": int(s.sum()), "rel_pct": float(100 * (v - b) / b)}
            res["parte_a"][nombre] = entrada
    if args.perexp:
        comp_off = (float(comp_yx[0] - star_yx[0]), float(comp_yx[1] - star_yx[1]))
        res["parte_b"] = parte_b(cfg, paths, psf_model, offset, lam, ratio, comp_off,
                                 n_exposures=args.perexp, step=args.perexp_step)
    # Parte B en las mismas unidades que la parte A: la media pesada por
    # `exptime` de lo que la resta mueve, contra el continuo del compañero.
    if args.perexp:
        filas = [f for f in res["parte_b"] if f["delta_flux"]]
        if filas:
            wref = np.asarray(filas[0]["wave_A"], dtype=np.float64)
            pesos_arr = np.asarray([f["weight"] for f in filas], dtype=np.float64)
            deltas = np.asarray([np.interp(wref, f["wave_A"], f["delta_flux"]) for f in filas])
            comps = np.asarray([np.interp(wref, f["wave_A"], f["comp_flux"]) for f in filas])
            medio = np.average(deltas, axis=0, weights=pesos_arr)
            comp = np.average(comps, axis=0, weights=pesos_arr)
            res["parte_b_resumen"] = {
                "n_exposures": len(filas),
                "bandas": {},
                "nota": ("media pesada por exptime del cambio del modelo restado, en la ventana "
                         "del compañero y con el peso de Horne, contra el flujo del compañero "
                         "medido en las MISMAS exposiciones con el mismo estimador"),
            }
            for etiqueta, (lo, hi) in {"7000-7500": (7000, 7500), "8000-8500": (8000, 8500),
                                       "8800-9300": (8800, 9300)}.items():
                s_b = (wref >= lo) & (wref < hi)
                if not np.any(s_b):
                    continue
                res["parte_b_resumen"]["bandas"][etiqueta] = {
                    "n": int(s_b.sum()),
                    "rel_pct": float(100 * np.nansum(medio[s_b]) / np.nansum(comp[s_b])),
                }
    print(json.dumps(res, indent=2))
    if args.npz and datos:
        np.savez(args.npz, **datos)
    if args.out:
        Path(args.out).write_text(json.dumps(res, indent=2), encoding="utf-8")
    return res


if __name__ == "__main__":
    main()
