#!/usr/bin/env python
"""Sesgo de la binaria de ROXs 42B sobre psffit con los dos fondos de C4: plano compartido (v1) y decoupled_c0 (v2).

Solo lectura. Reproduce las cifras del párrafo de la binaria de la Sect. `methods:psf` del paper
(cabecera de procedencia de `paper/main.tex`, decisión B del 2026-10-26), que salieron de
`~/muse_sesion_20261025/paper_c0/a2/binaria_c0.py`, `binaria_analiza.py` y `binaria_valida.py`:
con C4 v2 y C1 a 55 px, sesgo neto −0.1/+0.0/+0.0 % en 7000–7500/8000–8500/8800–9300 Å, Hα
+0.00 σ y amplitud libre −0.01 %; con el plano compartido a 55 px, −6.2/−1.5/−0.8 %.

Es `scripts/binaria_psffit_probe.py` (la sonda del 2026-09-07, que solo conoce el diseño de 5
columnas de v1) con el diseño de la etapa para cada fondo (`psf_pair_design(background=...)`).
En cada canal de las ventanas ajusta, para cada fondo, tres modelos de la columna estrella:

  base    P(estrella)                                  <- lo que corre C4
  ligada  P(estrella) + f(λ)·P(estrella + offset)       f de los 43 bins de C1, sin grados nuevos
  libre   la secundaria como columna propia por canal

La geometría de la secundaria se declara (`e01_binary_companion` del run con binaria, la misma
conversión que C1). El control es ROXs 12 b, una estrella sola, con la MISMA binaria impuesta y
la `f(λ)` del objeto: lo que se mueve allí es el suelo del método, y el sesgo NETO es objeto −
control.

Cifras (las de la sesión, mismas definiciones):

- continuo: cambio relativo integrado por banda, `100 (Σ variante − Σ base) / Σ base` sobre los
  canales de la banda (uno de cada dos), objeto y control, y neto = objeto − control;
- Hα: flujo de integración directa de `musepipe.lines.measure_line` (G2) sobre 6480–6660 Å, con
  el error del ajuste de la base, la LSF que resuelve E1 para el run y 200 realizaciones (semilla
  0); desplazamiento de la variante en unidades del error de la base;
- libre: mediana del cambio relativo canal a canal a λ > 8000 Å, y el integrado.

**Validación** (`--copia-objeto`, `--copia-control`): la variante `base` es el ajuste de la
etapa, así que multiplicada por el factor de la curva de crecimiento que C4 aplica
(`x03_flux_convention`) tiene que dar el flujo de `spec_psffit_object.fits` canal a canal; con
c0 contra el producto actual del run y con el plano compartido contra la copia de antes de c0
(`runs/<RUN>/stages/_antes_c4_c0_*`), que tiene el mismo C1. Se informa la igualdad exacta y
el cociente sonda/producto, como la sesión.

Coste: ~2 min por run (744 canales × 6 ajustes) y ~4.5 GB de memoria (el cubo y la STAT en float64);
los dos runs, ~4 min. Medido el 2026-10-26: los arrays por canal son los de la sesión bit a bit y la
base, por el factor de C4, es el producto de C4 bit a bit en los dos fondos y los dos runs.

uso:
    python scripts/binaria_psffit_c0.py --project-root RAIZ \\
        --copia-objeto runs/ROXs42Bb_realigned/stages/_antes_c4_c0_20261024T225459 \\
        --copia-control runs/ROXs12b_invvar/stages/_antes_c4_c0_20261024T200833 \\
        --npz-dir SALIDA --json SALIDA/binaria_c0.json
    python scripts/binaria_psffit_c0.py --desde-npz OBJ.npz CONTROL.npz   # solo el análisis
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

#: (lo, hi, paso): la ventana de Hα canal a canal; las bandas de continuo, uno de cada dos canales
VENTANAS_A = ((6480.0, 6660.0, 1), (7000.0, 7500.0, 2), (8000.0, 8500.0, 2), (8800.0, 9300.0, 2))
FONDOS = ("shared", "decoupled_c0")
VARIANTES = ("base", "ligada", "libre")
BANDAS_A = {"7000-7500": (7000.0, 7500.0), "8000-8500": (8000.0, 8500.0), "8800-9300": (8800.0, 9300.0)}
VENTANA_HA_A = (6480.0, 6660.0)
HALPHA = {"name": "Halpha", "wave_A": 6562.80}
HA_N_MC, HA_SEMILLA = 200, 0
LIBRE_DESDE_A = 8000.0


def _sonda_v1():
    """`scripts/binaria_psffit_probe.py` como módulo (la tabla de f y el λ ajustado de la caché)."""
    nombre = "binaria_psffit_probe"
    if nombre in sys.modules:
        return sys.modules[nombre]
    spec = importlib.util.spec_from_file_location(nombre, ROOT / "scripts" / f"{nombre}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[nombre] = mod
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------- piezas puras

def canales_de_ventanas(wave_A, ventanas=VENTANAS_A, step=1):
    """Índices ordenados y sin repetir: en cada ventana [lo, hi), uno de cada `paso·step` canales."""
    w = np.asarray(wave_A, dtype=np.float64)
    partes = [np.flatnonzero((w >= lo) & (w < hi))[:: int(paso) * int(step)] for lo, hi, paso in ventanas]
    return np.unique(np.concatenate(partes)) if partes else np.array([], dtype=int)


def cambio_por_banda(wave_A, base, variante, bandas=BANDAS_A):
    """`100 (Σ variante − Σ base) / Σ base` en cada banda [lo, hi), con los canales finitos en los dos."""
    w, b, v = (np.asarray(x, dtype=np.float64) for x in (wave_A, base, variante))
    out = {}
    for nombre, (lo, hi) in bandas.items():
        m = (w >= lo) & (w < hi) & np.isfinite(b) & np.isfinite(v)
        out[nombre] = float(100.0 * (v[m].sum() - b[m].sum()) / b[m].sum()) if m.any() else float("nan")
    return out


def cambio_neto(objeto, control):
    """Objeto − control, banda a banda (el suelo del método se resta)."""
    return {k: float(objeto[k] - control[k]) for k in objeto}


def libre_en_el_rojo(wave_A, base, libre, desde_A=LIBRE_DESDE_A):
    """(mediana del cambio relativo canal a canal, cambio integrado), en % y a λ > `desde_A`."""
    w, b, v = (np.asarray(x, dtype=np.float64) for x in (wave_A, base, libre))
    m = (w > desde_A) & np.isfinite(b) & np.isfinite(v) & (np.abs(b) > 0)
    rel = 100.0 * (v[m] - b[m]) / b[m]
    return float(np.median(rel)), float(100.0 * (v[m].sum() - b[m].sum()) / b[m].sum())


def desplazamiento_halpha(wave_A, err_base, variantes, *, lsf_fwhm_A, ventana_A=VENTANA_HA_A,
                          n_mc=HA_N_MC, seed=HA_SEMILLA):
    """Flujo de Hα de la base (y su S/N) y el desplazamiento de cada variante en σ de la base."""
    from musepipe.lines import measure_line

    w = np.asarray(wave_A, dtype=np.float64)
    m = (w >= ventana_A[0]) & (w < ventana_A[1])
    e = np.asarray(err_base, dtype=np.float64)[m]
    medidas = {}
    for nombre, flujo in variantes.items():
        lm = measure_line(w[m], np.asarray(flujo, dtype=np.float64)[m], e, HALPHA, lsf_fwhm_A=float(lsf_fwhm_A),
                          n_mc=int(n_mc), seed=int(seed))
        medidas[nombre] = (float(lm.flux_direct), float(lm.flux_direct_err))
    fb, eb = medidas["base"]
    out = {"base": fb, "base_sigma": fb / eb}
    for nombre, (f, _e) in medidas.items():
        if nombre != "base":
            out[f"{nombre}_shift_sigma"] = (f - fb) / eb
    return out


def resumen_de_sonda(out, *, lsf_fwhm_A, fondos=FONDOS):
    """Las cifras de un run (objeto o control) para cada fondo."""
    w = out["wave_A"]
    res = {}
    for fo in fondos:
        base = out[f"comp_{fo}_base"]
        res[fo] = {
            "ligada": cambio_por_banda(w, base, out[f"comp_{fo}_ligada"]),
            "libre": cambio_por_banda(w, base, out[f"comp_{fo}_libre"]),
            "libre_rojo_mediana_y_total_pct": libre_en_el_rojo(w, base, out[f"comp_{fo}_libre"]),
            "halpha": desplazamiento_halpha(w, out[f"err_{fo}_base"],
                                            {v: out[f"comp_{fo}_{v}"] for v in VARIANTES}, lsf_fwhm_A=lsf_fwhm_A),
            "chi2_mediana": {v: float(np.nanmedian(out[f"chi2_{fo}_{v}"])) for v in VARIANTES},
            "f_libre_rojo_mediana": float(np.nanmedian(out[f"flibre_{fo}"][w > LIBRE_DESDE_A])),
        }
    return res


def analiza(objeto, control, *, lsf_objeto_A, lsf_control_A, fondos=FONDOS):
    """Objeto, control y neto (= objeto − control) de la variante ligada y de la libre."""
    ro = resumen_de_sonda(objeto, lsf_fwhm_A=lsf_objeto_A, fondos=fondos)
    rc = resumen_de_sonda(control, lsf_fwhm_A=lsf_control_A, fondos=fondos)
    neto = {fo: {v: cambio_neto(ro[fo][v], rc[fo][v]) for v in ("ligada", "libre")} for fo in fondos}
    return {"objeto": ro, "control": rc, "neto": neto,
            "lsf_fwhm_A": {"objeto": float(lsf_objeto_A), "control": float(lsf_control_A)}}


# ---------------------------------------------------------------- lectura y ajuste

def lsf_de_e1(run_id, project_root):
    """La LSF con la que E1 mide el run (knob declarado -> A4/M2), con su procedencia."""
    from musepipe.io import read_json
    from musepipe.spectral import resolve_lsf_fwhm_A
    from musepipe.stages.stage_h01_detect import stage_h01_config_from_run, stage_h01_paths

    cfg = stage_h01_config_from_run(run_id, project_root=project_root)
    qc00_path = Path(cfg.get("h01_stage00q_qc_json", stage_h01_paths(cfg["run_id"], project_root)["stage00q_qc_json"]))
    return resolve_lsf_fwhm_A(read_json(qc00_path) if qc00_path.exists() else None, cfg)


def sonda(run_id, project_root, *, binaria_de=None, fondos=FONDOS, ventanas=VENTANAS_A, step=1, verbose=True):
    """Ajusta las tres variantes en cada fondo. `binaria_de` = run del que se impone la binaria y su f(λ)
    (el control); sin él, la binaria es la del propio run. Devuelve (meta, out) con los arrays por canal."""
    from musepipe.extraction.psffit import _fit_one_channel, fit_region_mask, psf_pair_design
    from musepipe.psf import evaluate_psf_model
    from musepipe.stages.stage_e01_psf import binary_offset_px
    from musepipe.stages.stage_x01_aperture import _load_positions, _load_stat_cube
    from musepipe.stages.stage_x02_optimal import _load_stage02_cube, _stat_metadata
    from musepipe.stages.stage_x03_psffit import _load_psf_model, stage_x03_config_from_run, stage_x03_paths

    v1 = _sonda_v1()
    cfg = stage_x03_config_from_run(run_id, project_root=project_root)
    paths = stage_x03_paths(cfg["run_id"], project_root)
    comp_yx, star_yx, _p, positions_qc = _load_positions(
        paths, {"x01_source_key": cfg["x03_source_key"], "x01_star_key": cfg["x03_star_key"]})
    psf_model, psf_path = _load_psf_model(paths, cfg)
    cube, wave, cube_path, _b = _load_stage02_cube(paths, cfg)
    stat_cube, stat_state = _load_stat_cube(paths, cfg, cube.shape)
    if stat_cube is None:
        raise RuntimeError(f"{run_id}: sin STAT ({stat_state}); la sonda pesa como la etapa y necesita la STAT")
    stat_factor, *_ = _stat_metadata(paths, cfg)

    pcfg = dict(cfg)
    ratio_dir = paths["paths"].stage_dir
    if binaria_de is not None:
        otra = stage_x03_config_from_run(binaria_de, project_root=project_root)
        if not otra.get("e01_binary_companion"):
            raise RuntimeError(f"{binaria_de} no declara `e01_binary_companion`: no hay binaria que imponer")
        pcfg.pop("e01_binary_offset_yx_px", None)
        pcfg["e01_binary_companion"] = dict(otra["e01_binary_companion"])
        ratio_dir = stage_x03_paths(otra["run_id"], project_root)["paths"].stage_dir
    offset = binary_offset_px(pcfg, positions_qc)
    if offset is None:
        raise RuntimeError(f"{run_id} no declara `e01_binary_companion`; para el control pasa el run con binaria")
    lam, ratio = v1.flux_ratio_table(ratio_dir)

    variance = np.asarray(stat_cube, dtype=np.float64) * (
        stat_factor if np.isfinite(stat_factor) and stat_factor > 0 else 1.0)
    nz, ny, nx = cube.shape
    rs, rc = float(cfg["x03_star_radius_px"]), float(cfg["x03_comp_radius_px"])
    mask = fit_region_mask((ny, nx), star_yx, comp_yx, star_radius_px=rs, comp_radius_px=rc)
    chans = canales_de_ventanas(wave, ventanas, step)
    yy, xx = np.indices((ny, nx), dtype=np.float64)

    out = {"wave_A": wave[chans], "f_used": np.full(chans.size, np.nan)}
    for fo in fondos:
        for v in VARIANTES:
            out[f"comp_{fo}_{v}"] = np.full(chans.size, np.nan)
            out[f"chi2_{fo}_{v}"] = np.full(chans.size, np.nan)
        out[f"err_{fo}_base"] = np.full(chans.size, np.nan)
        out[f"flibre_{fo}"] = np.full(chans.size, np.nan)
    cache = {}
    t0 = time.time()
    for i, z in enumerate(chans):
        f = float(np.interp(wave[z], lam, ratio))
        out["f_used"][i] = f
        clave = v1.snapped_wave(psf_model, wave[z])
        if clave not in cache:
            p2 = evaluate_psf_model(psf_model, float(wave[z]), yy - (float(star_yx[0]) + float(offset[0])),
                                    xx - (float(star_yx[1]) + float(offset[1])))
            disenos = {fo: psf_pair_design((ny, nx), wave[z], star_yx, comp_yx, psf_model, background=fo,
                                           star_radius_px=rs, comp_radius_px=rc) for fo in fondos}
            cache[clave] = (disenos, p2)
        disenos, p2 = cache[clave]
        for fo in fondos:
            for v, d in zip(VARIANTES, v1.design_variants(disenos[fo], p2, f)):
                c, cov, chi, *_ = _fit_one_channel(cube[z], variance[z], d, mask)
                out[f"comp_{fo}_{v}"][i] = c[1]
                out[f"chi2_{fo}_{v}"][i] = chi
                if v == "base":
                    out[f"err_{fo}_base"][i] = (np.sqrt(cov[1, 1]) if np.isfinite(cov[1, 1]) and cov[1, 1] > 0
                                                else np.nan)
                if v == "libre":
                    out[f"flibre_{fo}"][i] = c[-1] / c[0] if np.isfinite(c[0]) and c[0] != 0 else np.nan
        if verbose and i and i % 200 == 0:
            hecho = (i + 1) / chans.size
            print(f"  {run_id}: {i + 1}/{chans.size} ({100 * hecho:.0f} %), {time.time() - t0:.0f} s", flush=True)
    meta = {"run_id": cfg["run_id"], "cube": str(cube_path), "psf_model": str(psf_path),
            "psf_form": psf_model.get("form"), "star_yx": [float(v) for v in star_yx],
            "comp_yx": [float(v) for v in comp_yx], "offset_yx_px": [float(v) for v in offset],
            "binaria_de": binaria_de or cfg["run_id"], "binaria": pcfg.get("e01_binary_companion"),
            "stat_state": stat_state, "stat_factor": float(stat_factor), "radios_px": [rs, rc],
            "n_canales": int(chans.size), "x03_background_del_run": cfg.get("x03_background"),
            "segundos": round(time.time() - t0, 1), "n_psf": len(cache)}
    return meta, out


def factor_de_c4(run_id, project_root, stage_dir, wave_A):
    """El factor de flujo que C4 multiplica al coeficiente (1 en la convención normrad)."""
    from musepipe.growth_curve import factor_at_wavelengths, resolve_flux_convention
    from musepipe.stages.stage_x03_psffit import stage_x03_config_from_run

    cfg = stage_x03_config_from_run(run_id, project_root=project_root)
    curva, convencion = resolve_flux_convention(cfg, stage_dir, knob="x03_flux_convention")
    w = np.asarray(wave_A, dtype=np.float64)
    if curva is None:
        return np.ones(w.size), convencion
    return np.ones(w.size) * np.asarray(factor_at_wavelengths(curva, w), dtype=np.float64), convencion


def valida_contra_c4(meta, out, project_root, *, productos):
    """La base de cada fondo contra su producto de C4 (`productos` = {fondo: carpeta stages/})."""
    from astropy.io import fits

    w = out["wave_A"]
    res = {}
    for fo, stage_dir in productos.items():
        if stage_dir is None:
            continue
        stage_dir = Path(stage_dir)
        ruta = stage_dir / "spec_psffit_object.fits"
        with fits.open(ruta) as h:
            t, cab = h[1].data, h[1].header
            wp, fp = np.asarray(t["wave_A"], dtype=np.float64), np.asarray(t["flux"], dtype=np.float64)
        idx = np.searchsorted(wp, w)
        if not (np.all(idx < wp.size) and np.array_equal(wp[idx], w)):
            raise RuntimeError(f"{ruta}: el eje de λ no contiene los canales de la sonda")
        factor, convencion = factor_de_c4(meta["run_id"], project_root, stage_dir, wp)
        sonda_c4 = out[f"comp_{fo}_base"] * factor[idx]
        r = out[f"comp_{fo}_base"] / fp[idx]
        r = r[np.isfinite(r)]
        dif = np.abs(sonda_c4 - fp[idx]) / np.abs(fp[idx])
        res[fo] = {"producto": str(ruta), "BKGMODE": cab.get("BKGMODE"), "convencion": convencion,
                   "igual_bit_a_bit": bool(np.array_equal(sonda_c4, fp[idx], equal_nan=True)),
                   "max_dif_relativa": float(np.nanmax(dif)), "sonda_sobre_producto_mediana": float(np.median(r)),
                   "sonda_sobre_producto_p1_p99": [float(np.percentile(r, 1)), float(np.percentile(r, 99))],
                   "n": int(r.size)}
    return res


def guarda_npz(path, meta, out):
    np.savez(path, meta=json.dumps(meta), **out)


def lee_npz(path):
    with np.load(path, allow_pickle=False) as z:
        return json.loads(str(z["meta"])), {k: z[k] for k in z.files if k != "meta"}


def imprime(res):
    for fo in FONDOS:
        if fo not in res["neto"]:
            continue
        print(f"\n### fondo = {fo}")
        print("| banda | objeto ligada | control ligada | NETO ligada | objeto libre | control libre | neto libre |")
        print("|---|---:|---:|---:|---:|---:|---:|")
        for k in BANDAS_A:
            a, c = res["objeto"][fo]["ligada"][k], res["control"][fo]["ligada"][k]
            al, cl = res["objeto"][fo]["libre"][k], res["control"][fo]["libre"][k]
            print(f"| {k} | {a:+.1f} % | {c:+.1f} % | **{a - c:+.1f} %** | {al:+.2f} % | {cl:+.2f} % | {al - cl:+.2f} % |")
        for lab in ("objeto", "control"):
            r = res[lab][fo]
            h = r["halpha"]
            med, tot = r["libre_rojo_mediana_y_total_pct"]
            print(f"- {lab}: Hα base {h['base']:.1f} ({h['base_sigma']:.2f} σ); ligada {h['ligada_shift_sigma']:+.2f} σ; "
                  f"libre {h['libre_shift_sigma']:+.2f} σ; libre > 8000 Å: mediana canal {med:+.2f} %, integrado "
                  f"{tot:+.2f} %; f libre > 8000 Å {r['f_libre_rojo_mediana']:+.3f}")
    print(f"(LSF de E1: objeto {res['lsf_fwhm_A']['objeto']:.6g} Å, control {res['lsf_fwhm_A']['control']:.6g} Å)")
    for lab, val in (res.get("validacion") or {}).items():
        for fo, v in val.items():
            igual = ("= producto bit a bit" if v["igual_bit_a_bit"]
                     else f"≠ producto (máx. dif. relativa {v['max_dif_relativa']:.2e})")
            print(f"validación {lab} {fo:13s} contra {v['producto']} (BKGMODE={v['BKGMODE']}): sonda × factor de C4 "
                  f"{igual}; sonda/producto mediana {v['sonda_sobre_producto_mediana']:.6f}, n={v['n']}")


def _paper_runs():
    fichas = []
    for f in sorted((ROOT / "targets").glob("*.json")):
        p = json.loads(f.read_text(encoding="utf-8")).get("paper")
        if p:
            fichas.append(p["run"])
    return fichas


def _ruta(texto, project_root):
    if texto is None:
        return None
    p = Path(texto)
    return (p if p.is_absolute() else Path(project_root) / p).resolve()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--project-root", default=str(ROOT), help="clon con runs/ (desde un worktree, el principal)")
    ap.add_argument("--objeto", default=None, help="run con binaria (por defecto, el del paper que la declara)")
    ap.add_argument("--control", default=None, help="run de una estrella sola (por defecto, el otro run del paper)")
    ap.add_argument("--copia-objeto", default=None,
                    help="copia de antes de c0 del objeto (stages/ y tables/): valida el plano compartido")
    ap.add_argument("--copia-control", default=None, help="lo mismo para el control")
    ap.add_argument("--step", type=int, default=1, help="submuestreo extra de canales (1 = el de la sesión)")
    ap.add_argument("--npz-dir", default=None, help="guarda aquí los arrays por canal (fuera de runs/)")
    ap.add_argument("--desde-npz", nargs=2, default=None, metavar=("OBJETO", "CONTROL"),
                    help="no ajusta: analiza dos npz guardados (los de esta herramienta o los de la sesión)")
    ap.add_argument("--json", default=None)
    a = ap.parse_args(argv)
    root = Path(a.project_root).resolve()

    if a.desde_npz:
        (mo, oo), (mc, oc) = lee_npz(a.desde_npz[0]), lee_npz(a.desde_npz[1])
        validacion = None
    else:
        from musepipe.stages.stage_x03_psffit import stage_x03_config_from_run, stage_x03_paths

        if a.objeto is None or a.control is None:
            con = [r for r in _paper_runs() if stage_x03_config_from_run(r, project_root=root).get("e01_binary_companion")]
            sin = [r for r in _paper_runs() if r not in con]
            if len(con) != 1 or len(sin) != 1:
                ap.error("no se pueden deducir objeto y control de targets/*.json: pasa --objeto y --control")
            a.objeto, a.control = a.objeto or con[0], a.control or sin[0]
        mo, oo = sonda(a.objeto, root, step=a.step)
        mc, oc = sonda(a.control, root, binaria_de=a.objeto, step=a.step)
        if a.npz_dir:
            d = Path(a.npz_dir)
            d.mkdir(parents=True, exist_ok=True)
            guarda_npz(d / f"binaria_{mo['run_id']}.npz", mo, oo)
            guarda_npz(d / f"binaria_{mc['run_id']}_control.npz", mc, oc)
        validacion = {}
        for lab, meta, out, copia in (("objeto", mo, oo, a.copia_objeto), ("control", mc, oc, a.copia_control)):
            actual = stage_x03_paths(meta["run_id"], root)["paths"].stage_dir
            productos = {meta["x03_background_del_run"]: actual}
            if copia is not None:
                productos["shared"] = _ruta(copia, root) / "stages"
            validacion[lab] = valida_contra_c4(meta, out, root, productos=productos)

    res = analiza(oo, oc, lsf_objeto_A=lsf_de_e1(mo["run_id"], root)[0], lsf_control_A=lsf_de_e1(mc["run_id"], root)[0])
    res["meta"] = {"objeto": mo, "control": mc}
    res["validacion"] = validacion
    imprime(res)
    if a.json:
        Path(a.json).write_text(json.dumps(res, indent=1, ensure_ascii=False, default=float) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
