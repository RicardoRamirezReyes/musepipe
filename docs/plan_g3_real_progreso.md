# Bitácora · Plan G3 real (docs/plan_g3_real_2026-07-16.md)

Una entrada por fase: fecha, commit, verificación. La actualiza el agente
ejecutor al cerrar cada WP.

## WP-G3R-0 · Línea base y gate de insumos — 2026-07-16

- Rama `phase-g3-real` creada desde `stage-d1-v2`. Antes de crearla, la
  reorganización pendiente de `reports/` (ajena a este plan) se commiteó por
  separado en `stage-d1-v2` como `cc82098` con autorización explícita del
  usuario (checkpoint en sesión). El trabajo halosub WP-H0..H6 ya estaba
  commiteado previamente (`8eee51c` y posteriores).
- **Línea base pytest: 421 passed** (53 s, 225 warnings). Comando:
  `python -m pytest tests/ -q` con el env conda `MUSE`.
- `scripts/check_g3_real_inputs.py` creado y ejecutado: gate completo verde
  sobre `ROXs12b_B_adp` —
  - config: 7 claves requeridas presentes (LSF 2.383 Å, d 138.6 pc);
  - `spec_final_object.fits`: 3681 canales, 4749.5–9349.5 Å, 16 columnas;
  - S/N mediana por banda = referencia del plan §0.2 (0.155 / 0.032 / 0.435 /
    3.776 vs 0.15 / 0.03 / 0.43 / 3.78, tolerancia ±20%);
  - `g1_channel_covariance.npz`: 8 claves, 18 bloques, corr_len mediana 1.45
    canales;
  - `stage04b_bad_wavelength_mask.npy`: shape (3681,), 216 canales True
    (la semántica bad/good se fija en WP-G3R-6; existe también
    `stage04b_good_wavelength_mask.npy` en stages/ — resolver ahí cuál usar);
  - `g2_line_measurements.csv`: 24 líneas.
- Verificación de cierre: gate verde + suite 421 verde. Commit de la fase:
  (se anota tras commitear).

## WP-G3R-1 · Decisiones congeladas + checkpoint humano — 2026-07-16

(pendiente de cierre; se completa en esta misma sesión)
