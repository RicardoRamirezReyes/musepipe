"""¿La divergencia de continuo en B6 la traen los residuos de cielo?

B6 (8600-9100 A) es, con B5, una de las dos unicas bandas donde la compañera
tiene continuo detectado, y es donde los tres metodos usables discrepan entre si
a 14-65 sigma (docs/2026-09-10_apertura_y_cola_parametrica.md). Es tambien el
bosque de OH: el unico sitio del rango donde el residuo de sustraccion de cielo
domina canal a canal. Y los tres metodos tratan el fondo de forma DISTINTA
-`aperture` no resta nada, `optimal_ls` resta la superficie local de 04b,
`psffit` lo absorbe en su pedestal-, asi que un residuo de cielo comun les entra
con pesos distintos y los separa.

La hipotesis, declarada antes de mirar: **si la divergencia de B6 vive en los
canales de linea de cielo, enmascararlos la reduce mucho mas de lo que la reduce
quitar el mismo numero de canales cualesquiera.**

La mascara NO sale de un catalogo generico -el de `telluric_lines` tiene 8 lineas
y solo 1 en B6, y `SKYLINE_WINDOWS` declara (7240, 9300) entero, que se comeria
la banda-. Sale de la **medida M4 del propio run**: el RMS canal a canal entre
32 aperturas de cielo vacio (`stage00q_m4_m5_curves.npz`). Donde hay residuo de
linea de cielo, ese RMS se dispara. Es independiente del objeto y de la
diferencia entre metodos, que es lo que se esta midiendo.

El control es la mitad del experimento: se repite enmascarando el MISMO numero de
canales elegidos al azar entre los que la mascara deja vivos. Si el efecto es de
cielo, la mascara real tiene que salirse de esa distribucion.

    python scripts/e_b6_cielo_enmascarado.py --run <run> --out-json <out>.json

Lectura, no escritura: no toca ningun producto del run.
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
from musepipe.stats import robust_sigma  # noqa: E402

#: Umbral por defecto: un canal es "de cielo" si su RMS entre aperturas vacias
#: supera este factor sobre su propia linea base local. 1.5 es el mismo orden que
#: el `R < 1.5` con el que A4/M4 declara verde el residuo de cielo.
DEFAULT_SKY_RATIO = 1.5

#: Ventana de la mediana corrida con la que se define "linea base local", en
#: canales. 101 canales ~ 125 A: mucho mas ancho que una linea de OH y mucho mas
#: estrecho que la banda.
DEFAULT_BASELINE_CHANNELS = 101


def running_median(values, width):
    out = np.full(values.size, np.nan, dtype=np.float64)
    half = max(1, int(width) // 2)
    for i in range(values.size):
        lo = max(0, i - half)
        hi = min(values.size, i + half + 1)
        chunk = values[lo:hi]
        chunk = chunk[np.isfinite(chunk)]
        if chunk.size:
            out[i] = np.median(chunk)
    return out


def sky_excess_from_m4(curves_path, wave_product, *, mode="bias",
                       baseline_channels=DEFAULT_BASELINE_CHANNELS):
    """Exceso de cielo por canal, medido en 32 aperturas de cielo VACIO.

    Dos observables distintos, y en este cubo dicen cosas distintas:

    - `rms`: la dispersion entre aperturas (`m4_rms`). Sube donde el residuo de
      cielo es RUIDOSO. En B5 tiene picos de linea (max 3.7x el suelo); en B6
      esta casi plana (1.6x), o sea que el bosque de OH no deja exceso de ruido
      canal a canal en este cubo.
    - `bias`: el sesgo comun (`m4_median`), en unidades de su propio error
      (rms/sqrt(N)). Es el residuo COHERENTE, el que sobrevive al promediar
      aperturas, y el unico que un metodo de fondo puede tratar distinto que
      otro. En B6 es significativo en el 100 % de los canales.

    En los dos casos el "suelo" local es una mediana corrida, y el exceso es el
    cociente contra ella: lo que se busca son canales que destaquen SOBRE su
    entorno, no el nivel absoluto de la banda.
    """
    data = np.load(curves_path)
    wave = np.asarray(data["wavelength_A"], dtype=np.float64)
    rms = np.asarray(data["m4_rms"], dtype=np.float64)
    median = np.asarray(data["m4_median"], dtype=np.float64)
    if wave.shape != wave_product.shape or not np.allclose(wave, wave_product, atol=1e-6):
        raise SystemExit(
            "La rejilla de M4 no es la del producto: "
            f"{wave.shape} vs {wave_product.shape}. Este script no interpola a proposito."
        )
    if mode == "rms":
        signal = rms
    elif mode == "bias":
        signal = np.abs(median)
    else:
        raise SystemExit(f"--mask-from desconocido: {mode}")
    baseline = running_median(signal, baseline_channels)
    with np.errstate(invalid="ignore", divide="ignore"):
        excess = signal / baseline
    return {"excess": excess, "signal": signal, "baseline": baseline, "rms": rms, "median": median}


def top_channels_in_band(excess, in_band, fraction):
    """Los `fraction` canales de mayor exceso DENTRO de la banda.

    Un umbral absoluto no sirve para las dos bandas a la vez -B5 tiene picos y
    B6 no-, y un experimento que a veces enmascara 0 canales y a veces 300 no se
    puede comparar con su nula. Fijando la FRACCION, el control de canales al
    azar tiene siempre el mismo tamaño que la mascara real.
    """
    idx = np.flatnonzero(in_band)
    values = np.where(np.isfinite(excess[idx]), excess[idx], -np.inf)
    n_take = max(1, int(round(float(fraction) * idx.size)))
    order = np.argsort(values)[::-1][:n_take]
    mask = np.zeros(excess.size, dtype=bool)
    mask[idx[order]] = True
    return mask


def band_stats(rows, pair_ids, band_name):
    out = {}
    for row in rows:
        if row["band"] != band_name or row["pair"] not in pair_ids:
            continue
        out[row["pair"]] = {
            "t": x10._finite_or_none(row.get("t_stat")),
            "p": x10._finite_or_none(row.get("p_value")),
            "n_channels": row.get("n_channels"),
        }
    return out


def run_compare(cfg, paths, products, controls, g1_inputs, channel_mask=None):
    """D1 tal cual, opcionalmente con canales excluidos de TODAS las bandas.

    El unico punto por el que una banda se convierte en canales es
    `_band_range_mask`, y `_product_band_mask` lo llama por nombre de modulo, asi
    que parchearlo aqui cubre los dos caminos. Se restaura siempre.
    """
    original = x10._band_range_mask
    if channel_mask is not None:
        keep = ~np.asarray(channel_mask, dtype=bool)

        def patched(wave_A, band):
            base = original(wave_A, band)
            if base.shape != keep.shape:
                raise RuntimeError("La mascara no cuadra con la rejilla del producto.")
            return base & keep

        x10._band_range_mask = patched
    try:
        return x10.compare_methods(
            products,
            controls,
            sigma_smooth_channels=int(cfg.get("x10_sigma_smooth_channels", 5)),
            g1_inputs=g1_inputs,
            p_divergent=float(cfg.get("x10_p_divergent", x10.DEFAULT_P_DIVERGENT)),
            p_strong=float(cfg.get("x10_p_strong", x10.DEFAULT_P_STRONG)),
            scale_gate_sigma=float(cfg.get("x10_scale_gate_sigma", x10.DEFAULT_SCALE_GATE_SIGMA)),
            gate_alpha=float(cfg.get("x10_control_gate_alpha", x10.DEFAULT_CONTROL_GATE_ALPHA)),
            line_continuum_window_A=float(
                cfg.get("x10_line_continuum_window_A", x10.DEFAULT_LINE_CONTINUUM_WINDOW_A)
            ),
        )
    finally:
        x10._band_range_mask = original


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="run con los productos de D1 (no se escribe en el)")
    ap.add_argument("--sky-run", default=None, help="run del que sale la curva M4 (por defecto, --run)")
    ap.add_argument("--project-root", default=None)
    ap.add_argument("--band", default="B6")
    ap.add_argument("--mask-from", choices=("bias", "rms"), default="bias",
                    help="que observable de M4 define el exceso de cielo (ver sky_excess_from_m4)")
    ap.add_argument("--top-fraction", type=float, default=0.25,
                    help="fraccion de canales de la banda que se enmascara (los de mayor exceso)")
    ap.add_argument("--baseline-channels", type=int, default=DEFAULT_BASELINE_CHANNELS)
    ap.add_argument("--n-null", type=int, default=40, help="repeticiones del control de canales al azar")
    ap.add_argument("--seed", type=int, default=20260910)
    ap.add_argument("--out-json", required=True)
    args = ap.parse_args(argv)

    root = Path(args.project_root or Path.cwd()).resolve()
    cfg = x10.stage_x10_config_from_run(args.run, project_root=root)
    paths = x10.stage_x10_paths(args.run, root)
    products = x10.load_method_products(x10._product_paths_from_config(cfg, paths))
    controls = x10.load_control_spectra(x10._control_paths_from_config(cfg, paths))
    g1_inputs = x10.load_g1_inputs(cfg, paths)

    wave = np.asarray(products["aperture"].wave_A, dtype=np.float64)
    sky_run = args.sky_run or args.run
    curves = Path(root) / "runs" / sky_run / "stages" / "stage00q_m4_m5_curves.npz"
    if not curves.exists():
        raise SystemExit(f"No esta la curva M4 de {sky_run}: {curves}")
    sky = sky_excess_from_m4(
        curves, wave, mode=args.mask_from, baseline_channels=args.baseline_channels
    )

    band = next(b for b in x10.COMPARISON_BANDS if b.name == args.band)
    in_band = (wave >= band.lo_A) & (wave <= band.hi_A)
    sky_in_band = top_channels_in_band(sky["excess"], in_band, args.top_fraction)
    n_band = int(np.count_nonzero(in_band))
    n_sky = int(np.count_nonzero(sky_in_band))
    exc = sky["excess"][sky_in_band]
    print(f"[{args.band}] {n_band} canales; mascara = los {n_sky} de mayor exceso de cielo "
          f"({100.0 * n_sky / n_band:.1f} %), modo '{args.mask_from}', "
          f"exceso {np.nanmin(exc):.2f}-{np.nanmax(exc):.2f}x sobre su entorno")
    if n_sky == 0 or n_sky >= n_band:
        raise SystemExit("La mascara deja la banda intacta o vacia: ajusta --top-fraction.")

    pairs_of_interest = ("psffit_vs_aperture", "optimal_ls_vs_aperture", "psffit_vs_optimal_ls")

    print("corriendo D1 sin mascara ...")
    rows_base, _c, qc_base = run_compare(cfg, paths, products, controls, g1_inputs)
    base = band_stats(rows_base, pairs_of_interest, args.band)
    base_b5 = band_stats(rows_base, pairs_of_interest, "B5")

    print("corriendo D1 con el cielo enmascarado ...")
    rows_sky, _c, qc_sky = run_compare(cfg, paths, products, controls, g1_inputs, channel_mask=sky_in_band)
    masked = band_stats(rows_sky, pairs_of_interest, args.band)
    masked_b5 = band_stats(rows_sky, pairs_of_interest, "B5")

    # Control: quitar el MISMO numero de canales, elegidos entre los que la
    # mascara de cielo deja vivos. Si el efecto fuera de perder banda y no de
    # cielo, la mascara real caeria dentro de esta distribucion.
    rng = np.random.default_rng(args.seed)
    candidates = np.flatnonzero(in_band & ~sky_in_band)
    nulls = {pid: [] for pid in pairs_of_interest}
    for i in range(int(args.n_null)):
        drop = rng.choice(candidates, size=n_sky, replace=False)
        null_mask = np.zeros(wave.size, dtype=bool)
        null_mask[drop] = True
        rows_null, _c, _q = run_compare(cfg, paths, products, controls, g1_inputs, channel_mask=null_mask)
        stats = band_stats(rows_null, pairs_of_interest, args.band)
        for pid in pairs_of_interest:
            if pid in stats and stats[pid]["t"] is not None:
                nulls[pid].append(float(stats[pid]["t"]))
        print(f"  nula {i + 1}/{args.n_null}", end="\r", flush=True)
    print()

    report = {
        "script": "e_b6_cielo_enmascarado",
        "run": args.run,
        "sky_run": sky_run,
        "band": args.band,
        "sky_mask": {
            "source": str(curves),
            "criterion": (f"top {args.top_fraction:.0%} del exceso '{args.mask_from}' "
                          f"sobre running_median(..., {args.baseline_channels}) dentro de la banda"),
            "mask_from": args.mask_from,
            "channels_in_band": n_band,
            "channels_masked": n_sky,
            "fraction_masked": n_sky / n_band,
        },
        "verdict_baseline": qc_base["verdict"],
        "verdict_masked": qc_sky["verdict"],
        "pairs": {},
        "control_band_B5": {"baseline": base_b5, "masked": masked_b5},
        "n_null": int(args.n_null),
        "seed": int(args.seed),
    }
    print(f"\n{'par':28s} {'t base':>9s} {'t cielo':>9s} {'nula: mediana':>14s} {'nula: sigma':>12s} {'z':>7s}")
    for pid in pairs_of_interest:
        t_b = base.get(pid, {}).get("t")
        t_m = masked.get(pid, {}).get("t")
        draws = np.asarray(nulls[pid], dtype=np.float64)
        med = float(np.median(draws)) if draws.size else float("nan")
        sig = float(robust_sigma(draws)) if draws.size > 2 else float("nan")
        z = (t_m - med) / sig if (t_m is not None and np.isfinite(sig) and sig > 0) else float("nan")
        report["pairs"][pid] = {
            "t_baseline": t_b,
            "p_baseline": base.get(pid, {}).get("p"),
            "t_sky_masked": t_m,
            "p_sky_masked": masked.get(pid, {}).get("p"),
            "null_median_t": med,
            "null_sigma_t": sig,
            "z_of_sky_mask_vs_null": None if not np.isfinite(z) else float(z),
            "null_t": [float(v) for v in draws],
        }
        print(f"{pid:28s} {t_b:9.2f} {t_m:9.2f} {med:14.2f} {sig:12.2f} {z:7.2f}")

    out = Path(args.out_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"\nescrito {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
