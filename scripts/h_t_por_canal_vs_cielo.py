"""¿Los métodos discrepan menos donde el cielo brilla más?

`docs/2026-09-10_b6_no_es_el_cielo.md` §6 dejó una lectura **etiquetada como
interpretación, no medida**: enmascarar los canales de cielo AUMENTA la
discrepancia entre métodos, y la explicación propuesta fue que en esos canales
todos están dominados por ruido -sus errores crecen, la contribución al
estadístico se diluye- y por eso «se parecen» más.

Su prueba es barata y no necesita re-correr D1: si es cierta, la discrepancia
POR CANAL tiene que anticorrelacionar con el brillo de cielo.

Se mide sobre dos cantidades distintas, porque dicen cosas distintas:

- `z` por canal = (f_i - f_j) / sigma_diff. Es la discrepancia en unidades de
  ruido. La hipótesis predice que BAJA donde el cielo sube.
- |f_i - f_j| por canal, sin normalizar. Si la hipótesis es la correcta, esta NO
  tiene por qué bajar: lo que cambia es el denominador, no la diferencia.

Si las dos bajan igual, la explicación no es «ruido que diluye» sino que los
métodos de verdad coinciden más allí, que es otra cosa.

    python scripts/h_t_por_canal_vs_cielo.py --run <run> --sky-run <run con A4> \
        --out-json <out>.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
import sys  # noqa: E402

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from musepipe.stages import stage_x10_compare as x10  # noqa: E402

PAIRS = (("optimal_ls", "aperture"), ("psffit", "optimal_ls"), ("psffit", "aperture"))


def spearman(a, b):
    """Rho de Spearman sin scipy: Pearson sobre los rangos."""
    def rank(v):
        order = np.argsort(v, kind="mergesort")
        r = np.empty(v.size, dtype=np.float64)
        r[order] = np.arange(v.size, dtype=np.float64)
        return r
    ra, rb = rank(np.asarray(a, float)), rank(np.asarray(b, float))
    ra -= ra.mean(); rb -= rb.mean()
    den = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / den) if den > 0 else np.nan


def permutation_p(x, y, rho, n_perm, rng):
    """p de dos colas por permutación: los canales NO son independientes
    (n_eff/n ~ 0.69 en espectral, docs/noise_model.md), así que un p analítico
    de Spearman mentiría. Permutar en BLOQUES conserva la correlación local."""
    block = 25
    n = x.size
    n_blocks = int(np.ceil(n / block))
    count = 0
    for _ in range(n_perm):
        order = rng.permutation(n_blocks)
        shuffled = np.concatenate([y[i * block:(i + 1) * block] for i in order])[:n]
        if abs(spearman(x, shuffled)) >= abs(rho):
            count += 1
    return (count + 1) / (n_perm + 1)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--sky-run", default=None)
    ap.add_argument("--project-root", default=None)
    ap.add_argument("--bands", nargs="+", default=["B5", "B6"])
    ap.add_argument("--n-perm", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20260912)
    ap.add_argument("--out-json", required=True)
    args = ap.parse_args(argv)

    root = Path(args.project_root or Path.cwd()).resolve()
    cfg = x10.stage_x10_config_from_run(args.run, project_root=root)
    paths = x10.stage_x10_paths(args.run, root)
    products = x10.load_method_products(x10._product_paths_from_config(cfg, paths))
    controls = x10.load_control_spectra(x10._control_paths_from_config(cfg, paths))
    wave = np.asarray(products["aperture"].wave_A, dtype=np.float64)

    sky_run = args.sky_run or args.run
    curves = root / "runs" / sky_run / "stages" / "stage00q_m4_m5_curves.npz"
    if not curves.exists():
        raise SystemExit(f"no esta la curva M4 de {sky_run}: {curves}")
    data = np.load(curves)
    if not np.allclose(np.asarray(data["wavelength_A"], float), wave, atol=1e-6):
        raise SystemExit("la rejilla de M4 no es la del producto")
    brillo = np.abs(np.asarray(data["m4_median"], float))   # residuo coherente de cielo

    rng = np.random.default_rng(args.seed)
    report = {"script": "h_t_por_canal_vs_cielo", "run": args.run, "sky_run": sky_run,
              "n_perm": args.n_perm, "bands": {}}
    print(f"{'banda':>6s} {'par':>26s} {'rho(z)':>9s} {'p':>8s} {'rho(|dif|)':>11s} {'p':>8s} {'n':>5s}")
    for band_name in args.bands:
        band = next(b for b in x10.COMPARISON_BANDS if b.name == band_name)
        in_band = (wave >= band.lo_A) & (wave <= band.hi_A)
        report["bands"][band_name] = {}
        for pair in PAIRS:
            pid = x10.pair_id(pair)
            pi, pj = products[pair[0]], products[pair[1]]
            sigma = x10.empirical_sigma_diff(
                controls[pair[0]], controls[pair[1]],
                smooth_channels=int(cfg.get("x10_sigma_smooth_channels", 5)))
            diff = np.asarray(pi.flux, float) - np.asarray(pj.flux, float)
            with np.errstate(invalid="ignore", divide="ignore"):
                z = diff / sigma
            ok = in_band & np.isfinite(z) & np.isfinite(brillo) & np.isfinite(diff) & (sigma > 0)
            x, zz, dd = brillo[ok], np.abs(z[ok]), np.abs(diff[ok])
            rho_z = spearman(x, zz)
            rho_d = spearman(x, dd)
            p_z = permutation_p(x, zz, rho_z, args.n_perm, rng)
            p_d = permutation_p(x, dd, rho_d, args.n_perm, rng)
            # El SIGNO, que es lo que el valor absoluto escondia. La t de D1
            # integra la diferencia CON SIGNO sobre la banda: si los canales de
            # cielo llevan signo contrario al de la banda, cancelan parte de la
            # integral, y quitarlos SUBE |t| aunque cada uno discrepe mas. Eso
            # reconciliaria el enmascarado (|t| sube al quitarlos) con la
            # correlacion positiva de arriba.
            firmado = diff[ok]
            media_banda = float(np.nanmean(firmado))
            corte = np.nanmedian(x)
            brillantes, tenues = firmado[x >= corte], firmado[x < corte]
            frac_contraria = float(np.mean(np.sign(brillantes) != np.sign(media_banda)))
            frac_contraria_tenues = float(np.mean(np.sign(tenues) != np.sign(media_banda)))
            report["bands"][band_name][pid] = {
                "rho_z": rho_z, "p_z": p_z, "rho_absdiff": rho_d, "p_absdiff": p_d,
                "n_channels": int(ok.sum()),
                "rho_signed": spearman(x, firmado),
                "band_mean_diff": media_banda,
                "mean_diff_sky_bright": float(np.nanmean(brillantes)),
                "mean_diff_sky_faint": float(np.nanmean(tenues)),
                "frac_opposite_sign_bright": frac_contraria,
                "frac_opposite_sign_faint": frac_contraria_tenues,
            }
            print(f"       signo: media banda {media_banda:+.4g} | cielo brillante "
                  f"{np.nanmean(brillantes):+.4g} | cielo tenue {np.nanmean(tenues):+.4g} | "
                  f"rho(con signo) {spearman(x, firmado):+.3f}")
            print(f"{band_name:>6s} {pid:>26s} {rho_z:9.3f} {p_z:8.4f} {rho_d:11.3f} {p_d:8.4f} {int(ok.sum()):5d}")
    Path(args.out_json).write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"\nescrito {args.out_json}")
    print("Lectura: rho(z) NEGATIVO y rho(|dif|) ~0 confirma la interpretacion;")
    print("         los dos negativos por igual la CONTRADICEN.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
