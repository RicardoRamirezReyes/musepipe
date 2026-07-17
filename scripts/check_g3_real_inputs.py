"""Gate de insumos del plan G3 real (docs/plan_g3_real_2026-07-16.md, WP-G3R-0).

Verifica que el run canonico tiene todos los insumos que las fases WP-G3R-6..12
consumen, con el esquema exacto documentado en el plan (§0.2-§0.3). Cada item
imprime ``OK`` o aborta con ``RuntimeError`` descriptivo. Pensado como gate
re-ejecutable: las fases posteriores lo corren antes de tocar datos reales.

Uso: python scripts/check_g3_real_inputs.py [--run-id ROXs12b_B_adp]
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from musepipe.config import load_run_config, project_root_path  # noqa: E402
from musepipe.models.manifest import (  # noqa: E402
    LIBRARY_SUBDIRS,
    MANIFEST_NAME,
    library_root,
    verify_manifest,
)

# Esquema verificado 2026-07-16 (plan §0.2-§0.3). Si el espectro canonico
# cambia, estos valores DEBEN revisarse junto con las decisiones congeladas.
SPECTRUM_FILE = "spec_final_object.fits"
SPECTRUM_EXT = "SPECTRUM"
SPECTRUM_ROWS = 3681
SPECTRUM_COLUMNS = (
    "wave_A", "flux", "flux_err", "flux_err_emp", "apcorr", "npix_eff",
    "flags", "flux_err_stat", "flux_err_total", "cont_runmed", "cont_poly",
    "sys_continuum", "sys_fluxcal", "sys_psf", "sys_sky", "sys_telluric",
)
G1_COV_FILE = "g1_channel_covariance.npz"
G1_COV_KEYS = (
    "rho_by_block", "block_bounds", "corr_length_by_block",
    "n_eff_over_n_by_block", "corr_length_median", "n_controls",
    "block_size", "max_lag",
)
BAD_MASK_FILE = "stage04b_bad_wavelength_mask.npy"
G2_TABLE = "g2_line_measurements.csv"
G2_N_LINES = 24
REQUIRED_CONFIG_KEYS = (
    "h01_lsf_fwhm_A", "h03_distance_pc", "h03_av", "h03_flux_unit_cgs",
    "g3_atmosphere_family", "g3_template_family", "g3_tracks_families",
)
# Mediana de flux/flux_err_total por banda, medida 2026-07-16 (plan §0.2).
SNR_REFERENCE = {
    (4800.0, 5800.0): 0.15,
    (5800.0, 6800.0): 0.03,
    (6800.0, 7800.0): 0.43,
    (7800.0, 9300.0): 3.78,
}
SNR_RTOL = 0.20


def _ok(item: str, detail: str = "") -> None:
    print(f"OK  {item}" + (f" — {detail}" if detail else ""))


def check_spectrum(stage_dir: Path):
    from astropy.io import fits

    path = stage_dir / SPECTRUM_FILE
    if not path.exists():
        raise RuntimeError(f"falta el espectro canonico: {path}")
    with fits.open(path) as hdul:
        if SPECTRUM_EXT not in [h.name for h in hdul]:
            raise RuntimeError(f"{path} sin extension {SPECTRUM_EXT}")
        table = hdul[SPECTRUM_EXT].data
        names = list(table.columns.names)
        missing = [c for c in SPECTRUM_COLUMNS if c not in names]
        if missing:
            raise RuntimeError(f"{path} sin columnas {missing} (tiene {names})")
        if len(table) != SPECTRUM_ROWS:
            raise RuntimeError(
                f"{path}: {len(table)} filas, esperadas {SPECTRUM_ROWS}")
        wave = np.asarray(table["wave_A"], float)
        flux = np.asarray(table["flux"], float)
        err = np.asarray(table["flux_err_total"], float)
    _ok(SPECTRUM_FILE, f"{len(wave)} canales, {wave.min():.1f}-{wave.max():.1f} A")
    return wave, flux, err


def check_snr(wave, flux, err):
    good = np.isfinite(flux) & np.isfinite(err) & (err > 0)
    for (lo, hi), ref in SNR_REFERENCE.items():
        band = good & (wave >= lo) & (wave <= hi)
        if not band.any():
            raise RuntimeError(f"banda {lo}-{hi} A sin canales validos")
        med = float(np.nanmedian(flux[band] / err[band]))
        if not np.isclose(med, ref, rtol=SNR_RTOL, atol=0.0):
            raise RuntimeError(
                f"S/N mediana en {lo:.0f}-{hi:.0f} A = {med:.3f}, referencia "
                f"{ref:.2f} (±{SNR_RTOL:.0%}): el espectro cambio respecto al "
                "plan congelado — PARAR y revisar decisiones (plan §0.2)")
        _ok(f"S/N {lo:.0f}-{hi:.0f} A", f"mediana {med:.3f} (ref {ref:.2f})")


def check_g1_covariance(stage_dir: Path, n_channels: int):
    path = stage_dir / G1_COV_FILE
    if not path.exists():
        raise RuntimeError(f"falta la covarianza G1: {path}")
    with np.load(path) as z:
        missing = [k for k in G1_COV_KEYS if k not in z.files]
        if missing:
            raise RuntimeError(f"{path} sin claves {missing} (tiene {z.files})")
        bounds = np.asarray(z["block_bounds"])
        if bounds.max() > n_channels:
            raise RuntimeError(
                f"{path}: block_bounds hasta {bounds.max()} > {n_channels} "
                "canales del espectro — runs desalineados (plan WP-G3R-6)")
        detail = (f"{len(z['n_eff_over_n_by_block'])} bloques, "
                  f"corr_len mediana {float(z['corr_length_median']):.2f} ch")
    _ok(G1_COV_FILE, detail)


def check_bad_mask(stage_dir: Path, n_channels: int):
    path = stage_dir / BAD_MASK_FILE
    if not path.exists():
        raise RuntimeError(f"falta la mascara de canales malos: {path}")
    mask = np.load(path)
    if mask.shape != (n_channels,):
        raise RuntimeError(
            f"{path}: shape {mask.shape}, esperada ({n_channels},)")
    _ok(BAD_MASK_FILE, f"{int(np.count_nonzero(mask.astype(bool)))} de "
        f"{n_channels} canales True (semantica bad/good se fija en WP-G3R-6)")


def check_g2_table(table_dir: Path):
    path = table_dir / G2_TABLE
    if not path.exists():
        raise RuntimeError(f"falta la tabla G2: {path}")
    with path.open() as fh:
        rows = list(csv.DictReader(fh))
    if len(rows) != G2_N_LINES:
        raise RuntimeError(
            f"{path}: {len(rows)} lineas, esperadas {G2_N_LINES}")
    for col in ("name", "rest_A", "status"):
        if col not in rows[0]:
            raise RuntimeError(f"{path} sin columna {col}")
    _ok(G2_TABLE, f"{len(rows)} lineas de catalogo")


def check_config_keys(cfg: dict):
    missing = [k for k in REQUIRED_CONFIG_KEYS if k not in cfg]
    if missing:
        raise RuntimeError(
            f"config del run sin claves requeridas: {missing} (plan §0.3)")
    _ok("config", f"{len(REQUIRED_CONFIG_KEYS)} claves requeridas presentes; "
        f"LSF={cfg['h01_lsf_fwhm_A']} A, d={cfg['h03_distance_pc']} pc")


def check_libraries(cfg: dict, project_root: Path):
    """Chequeo opcional WP-G3R-2: verifica los manifiestos de las cinco familias
    externas. Falla limpio (RuntimeError) si el root o alguna familia no estan,
    para que el gate base siga sirviendo en maquinas sin datos externos."""
    root = library_root(cfg, project_root=project_root)  # RuntimeError si no existe
    missing = []
    for sub, subdir in LIBRARY_SUBDIRS.items():
        family_dir = root / subdir
        if not (family_dir / MANIFEST_NAME).exists():
            missing.append(subdir)
            continue
        info = verify_manifest(family_dir)  # RuntimeError si corrupto/incompleto
        _ok(subdir, f"{info['n_files']} entradas verificadas")
    if missing:
        raise RuntimeError(
            f"bibliotecas externas ausentes: {missing} — ejecuta "
            "scripts/fetch_g3_libraries.py <familia> (plan WP-G3R-2)")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="ROXs12b_B_adp")
    parser.add_argument("--libraries", action="store_true",
                        help="ademas verifica los manifiestos de las 5 familias externas")
    args = parser.parse_args(argv)

    project_root = project_root_path(Path(__file__).resolve().parents[1])
    rc = load_run_config(args.run_id, project_root=project_root)
    print(f"Gate de insumos G3 real — run {rc.run_id}")
    check_config_keys(dict(rc.config))
    wave, flux, err = check_spectrum(rc.paths.stage_dir)
    check_snr(wave, flux, err)
    check_g1_covariance(rc.paths.stage_dir, len(wave))
    check_bad_mask(rc.paths.stage_dir, len(wave))
    check_g2_table(rc.paths.table_dir)
    if args.libraries:
        check_libraries(dict(rc.config), project_root)
    print("Gate completo: todos los insumos presentes y consistentes.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
