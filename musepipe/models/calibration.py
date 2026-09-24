"""Calibración del umbral de bondad de ajuste de G3 (Q1, en bins de 25 Å).

Plan ``docs/2026-09-23_plan_g3_bibliotecas_y_resolucion.md`` §5 (Q1, aprobada
el 2026-09-23) y reparto aprobado el 2026-09-24 (ver
``docs/2026-09-23_decision_g3_resolucion_y_bibliotecas.md``):

* el **tipo** se mide en el ajuste NATIVO por Δχ² entre subtipos; allí cada
  canal está dominado por el ruido y la plantilla buena y las malas dan todas
  χ²_ν ≈ 1, así que un umbral absoluto no discrimina;
* la **bondad de ajuste** (¿la forma del continuo casa?) se prueba en la
  variante BINADA (25 Å, Q4), y su umbral se calibra aquí.

Procedimiento (``n_channels`` = 20 para el umbral; = 1 solo como diagnóstico):

1. cada plantilla de la biblioteca se degrada a la LSF de MUSE, en el marco del
   dato, sobre la rejilla POR CANAL del compañero, y se escala a su flujo;
2. se le suma ruido con el **espectro de error del propio compañero**
   (``flux_err_total``) por canal, correlacionado con la longitud de
   correlación de G1 (``L = n/n_eff``, ruido blanco suavizado con Σρ = L);
3. se bina **exactamente como el dato real**
   (:func:`musepipe.models.observed.rebin_for_fit`: misma máscara, mismos
   bloques de n_eff/n, misma regla de varianza);
4. se ajusta **dejándola fuera** (leave-one-out) contra el resto de su
   biblioteca, con modelos preparados sobre la rejilla binada como en el
   ajuste real;
5. se anota χ²_ν en el **subtipo correcto** (el mejor espectro de ese subtipo).

Regla de calibración (decisión del autor 2026-09-24, knob
``g3_gof_calibration``):

* ``same_subtype_neighbour`` (por defecto): solo cuentan las plantillas que
  tienen **otra del mismo subtipo** en la biblioteca. Una plantilla única de su
  subtipo se compararía con el vecino más cercano, y ese desajuste de subtipo
  (no de ruido) inflaba el umbral (p95 3.38 → 2.15 en 42B b, 8.11 → 4.23 en
  12 b, medido 2026-09-24).
* ``nearest_available``: todas, con el subtipo disponible más cercano como
  «correcto» (la regla del 2026-09-23; queda como diagnóstico).

Umbral propuesto: percentil 95. El A_V inyectado es 0 (se ajusta igual).
"""

from __future__ import annotations

import numpy as np

from .fit import _best_scale_chi2
from .observed import FitSpectrum, neff_over_n_per_channel, rebin_for_fit
from .prep import prepare_template_base
from .template_fit import _template_items, declared_prep_kwargs, fit_templates


def correlated_noise(n, corr_len, rng):
    """Ruido de varianza unidad con Σρ = ``corr_len`` (blanco suavizado).

    Para un núcleo gaussiano discreto k, Σ_Δ ρ(Δ) = (Σk)²/Σk²; se busca la σ
    del núcleo que da ``corr_len`` por bisección. ``corr_len ≤ 1`` → blanco.
    """
    n = int(n)
    white = rng.standard_normal(n)
    if not corr_len or corr_len <= 1.0 + 1e-6:
        return white

    def ratio(s):
        half = max(1, int(np.ceil(5 * s)))
        x = np.arange(-half, half + 1)
        k = np.exp(-0.5 * (x / s) ** 2)
        return k.sum() ** 2 / np.sum(k ** 2), k

    lo, hi = 1e-3, 50.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        r, _ = ratio(mid)
        if r < corr_len:
            lo = mid
        else:
            hi = mid
    _, k = ratio(0.5 * (lo + hi))
    half = (k.size - 1) // 2
    padded = np.concatenate([rng.standard_normal(half), white, rng.standard_normal(half)])
    y = np.convolve(padded, k, mode="valid")
    return y / np.sqrt(np.sum(k ** 2))


def _fit_spectrum_from_rebin(reb, n_channels):
    native = int(n_channels) == 1
    n_dof = reb["n_eff_total"] if native else float(reb["n_bins"])
    return FitSpectrum(reb["wave_bin"], reb["flux_bin"], reb["err_bin"], reb["n_bins"],
                       reb["n_eff_total"], {}, n_dof=n_dof, bin_channels=int(n_channels),
                       dof_weight=(reb["neff_over_n_bin"] if native
                                   else np.ones(reb["n_bins"])))


def calibrate_gof_threshold(library, inputs, *, wave_range, n_channels, lsf_fwhm_A,
                            extinction, av_axis, data_frame, n_draws=20, seed=0,
                            percentile=95.0, max_masked_frac=0.5, late_code_min=5.0,
                            rule="same_subtype_neighbour"):
    """Distribución leave-one-out de χ²_ν en el subtipo correcto, en la
    representación de ``n_channels`` canales por bin, y umbral propuesto.

    ``inputs``: :func:`musepipe.models.observed.load_fit_inputs` del run (o un
    equivalente sintético): rejilla, flujo y error por canal del compañero,
    máscara D9, bad mask y covarianza de G1.
    """
    if rule not in CALIBRATION_RULES:
        raise RuntimeError(f"g3_gof_calibration desconocida: {rule!r} (válidas: "
                           f"{CALIBRATION_RULES})")
    wave = np.asarray(inputs["wave"], float)
    flux = np.asarray(inputs["flux"], float)
    sig = np.asarray(inputs["err"], float)
    mask = np.asarray(inputs["mask"], bool)
    cov, bad_mask = inputs["cov"], inputs["bad_mask"]
    lo, hi = float(wave_range[0]), float(wave_range[1])
    ratio = neff_over_n_per_channel(wave, cov, bad_mask)
    in_rng = (wave >= lo) & (wave <= hi)
    usable = in_rng & ~mask & np.isfinite(flux) & np.isfinite(sig) & (sig > 0)
    corr_len = float(np.median(1.0 / ratio[usable]))
    iv_native = np.where(usable, ratio / np.where(sig > 0, sig, 1.0) ** 2, 0.0)
    rng = np.random.default_rng(int(seed))

    items = _template_items(library, per_spectrum=True)
    truth_base = []
    for _c, _l, _o, tmpl in items:
        base, _ = prepare_template_base(tmpl, wave, lsf_fwhm_A=lsf_fwhm_A,
                                        **declared_prep_kwargs(tmpl, data_frame))
        truth_base.append(base)
    codes_all = np.array([it[0] for it in items], float)
    prepared_by_grid = {}

    def prepared_for(grid):
        key = (grid.size, float(grid[0]), float(grid[-1]), float(np.sum(grid)))
        if key not in prepared_by_grid:
            prepared_by_grid[key] = {
                j: prepare_template_base(t, grid, lsf_fwhm_A=lsf_fwhm_A,
                                         **declared_prep_kwargs(t, data_frame))
                for j, (_c, _l, _o, t) in enumerate(items)}
        return prepared_by_grid[key]

    rows = []
    for i, (code, label, obj, _t) in enumerate(items):
        base = truth_base[i]
        model = np.where(usable & np.isfinite(base), base, np.nan)
        scale, _ = _best_scale_chi2(flux, model, iv_native)
        if not np.isfinite(scale) or scale <= 0:
            med = np.nanmedian(model)
            scale = float(np.nanmedian(flux[usable]) / med) if med else np.nan
        truth = scale * base
        m_i = mask | ~np.isfinite(truth)
        others = np.array([j for j in range(len(items)) if j != i])
        other_codes = codes_all[others]
        if np.any(other_codes == code):
            correct, how = [float(code)], "same_subtype"
        elif rule == "same_subtype_neighbour":
            continue  # sin vecino del mismo subtipo: no entra en la calibración
        else:
            d = np.abs(other_codes - code)
            correct, how = sorted({float(c) for c in other_codes[d == d.min()]}), \
                "nearest_available"
        for draw in range(int(n_draws)):
            noise = correlated_noise(wave.size, corr_len, rng) * np.where(
                np.isfinite(sig), sig, 0.0)
            data = np.where(np.isfinite(truth), truth + noise, np.nan)
            reb = rebin_for_fit(wave, data, sig, m_i, cov, n_channels=int(n_channels),
                                wave_range=wave_range, bad_mask=bad_mask,
                                max_masked_frac=max_masked_frac)
            fs = _fit_spectrum_from_rebin(reb, n_channels)
            res = fit_templates(fs, library, extinction, av_axis=av_axis,
                                lsf_fwhm_A=lsf_fwhm_A, data_frame=data_frame,
                                per_spectrum=True, prepared=prepared_for(fs.wave_bin),
                                exclude=lambda j, _c, _o, i=i: j == i)
            at = [r for r in res["ranking"] if r["spt_code"] in correct]
            best_at = min(at, key=lambda r: r["chi2"])
            rows.append({"template_index": i, "object": obj, "spt": label,
                         "spt_code": float(code), "draw": draw,
                         "correct_subtypes": correct, "correct_how": how,
                         "chi2_red_correct": float(best_at["chi2_red"]),
                         "chi2_red_min": float(res["chi2_red_min"]),
                         "spt_best": res["spt_best"],
                         "spt_best_code": float(res["spt_best_code"]),
                         "abs_dspt_best": float(abs(res["spt_best_code"] - code)),
                         "ndof": float(res["ndof"])})

    def summary(sel):
        v = np.array([r["chi2_red_correct"] for r in sel], float)
        if v.size == 0:
            return {"n": 0}
        q = np.percentile(v, [5, 16, 50, 84, 95])
        return {"n": int(v.size),
                "n_templates": len({r["template_index"] for r in sel}),
                "p05": float(q[0]), "p16": float(q[1]), "p50": float(q[2]),
                "p84": float(q[3]), "p95": float(q[4]),
                "max": float(v.max()), "mean": float(v.mean()),
                "frac_best_within_1_subtype": float(np.mean(
                    [r["abs_dspt_best"] <= 1.0 for r in sel]))}

    late = [r for r in rows if r["spt_code"] >= float(late_code_min)]
    same = [r for r in rows if r["correct_how"] == "same_subtype"]
    used = sorted({(r["template_index"], r["object"], r["spt"]) for r in rows})
    return {
        "rule": rule,
        "n_templates_library": len(items),
        "n_templates_used": len(used),
        "templates_used": [{"object": o, "spt": t} for _i, o, t in used],
        "representation": ("native channels (diagnostic)" if int(n_channels) == 1
                           else f"{int(n_channels)}-channel bins (~25 A)"),
        "n_channels": int(n_channels),
        "method": ("leave-one-out: each template degraded to the MUSE LSF (data frame) on "
                   "the companion channel grid, scaled to the companion flux, + noise from "
                   "the companion error spectrum correlated with the G1 length, binned "
                   "exactly as rebin_for_fit; chi2_red at the correct subtype (or nearest "
                   "available)"),
        "percentile": float(percentile),
        "threshold_proposed": (float(np.percentile([r["chi2_red_correct"] for r in rows],
                                                   percentile)) if rows else None),
        "distribution": summary(rows),
        "distribution_late_M": {"code_min": float(late_code_min), **summary(late)},
        "distribution_same_subtype_only": summary(same),
        "n_draws": int(n_draws), "seed": int(seed), "corr_len_channels": corr_len,
        "library": getattr(library, "name", None),
        "injected_av": 0.0,
        "rows": rows,
    }


#: Reglas de ``g3_gof_calibration``; la primera es la de por defecto.
CALIBRATION_RULES = ("same_subtype_neighbour", "nearest_available")

__all__ = ["CALIBRATION_RULES", "calibrate_gof_threshold", "correlated_noise"]
