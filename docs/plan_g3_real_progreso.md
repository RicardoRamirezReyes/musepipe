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
  `1597805`.

## WP-G3R-1 · Decisiones congeladas + checkpoint humano — 2026-07-16

- **Checkpoint humano cumplido**: el usuario aprobó D1–D14 tal como estaban
  propuestas en el plan («Apruebo D1–D14, ejecuta WP-G3R-0 y WP-G3R-1»,
  sesión 2026-07-16). Registro completo en
  `docs/g3_real_frozen_decisions.md`.
- 37 claves congeladas escritas en `runs/ROXs12b_B_adp/config/config.json`;
  solo `g3_template_family`/`g3_template_citation` sobrescribieron valores
  previos (corrección D1: Bonnefoy+2014 era NIR → X-shooter Class III de
  Manara+2013/2017). Como `runs/` está ignorado por git, el registro
  commiteado de la congelación es `docs/g3_real_frozen_decisions.md`.
- D12 transcrita de la fuente (abstract arXiv:1708.07611, verificado
  2026-07-16): edad ROXs 12 = 6 +4/−2 Myr → `g3_age_myr = 6.0`,
  `g3_age_err_myr = [2.0, 4.0]`.
- Verificación de cierre: `python scripts/check_g3_real_inputs.py` verde con
  la config congelada cargando vía `load_run_config` (validación incluida);
  suite completa 421 passed (51.7 s), idéntica a la línea base.
- **El commit que introduce `docs/g3_real_frozen_decisions.md` es el hash de
  congelación** que los QC de WP-G3R-11/12 deben citar como
  `frozen_decisions_commit` (localizable con
  `git log --follow --oneline docs/g3_real_frozen_decisions.md | tail -1`).
- Siguiente fase: WP-G3R-2 (infraestructura de bibliotecas externas:
  manifiestos sha256 + descarga de las cinco familias). Hay red en esta
  máquina (verificado 2026-07-16).
