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

## WP-G3R-2 · Infraestructura de bibliotecas externas — 2026-07-16

**Software (mecánica, sin ciencia) — completo y verificado:**

- `musepipe/models/manifest.py`: `verify_manifest` (→ `{n_files,
  sha256_of_manifest, verified_utc}`, detecta faltantes/corruptos),
  `write_manifest`, `parse_manifest`, `library_root(cfg, project_root=)`
  (resuelve `g3_libraries_root` D14 relativo a la raíz; RuntimeError si falta
  la clave o el directorio). Constante compartida `LIBRARY_SUBDIRS` (D14).
- `musepipe/models/cache.py`: formato de caché interno CONGELADO —
  `write_spectrum_npz`/`load_spectrum_npz` (recorte 4000–10000 Å, orden
  ascendente, `meta_json`), `write_tracks_npz`/`load_tracks_npz`
  (`TRACK_ARRAYS = mass_msun, age_gyr, teff_k, l_bol_lsun, radius_rsun,
  logg`). Módulo único para no duplicar el formato entre fetch y adaptadores
  WP-3/4/5 (§0.5.2).
- `scripts/fetch_g3_libraries.py`: subcomando por familia (`bt-settl`,
  `templates-young`, `templates-field`, `tracks-bhac15`, `tracks-atmo2020`) +
  `verify`. Descarga directa cuando la fuente lo permite; si requiere descarga
  manual/registro imprime instrucciones EXACTAS (cita congelada + layout de
  `--input-dir`) y sale con código 2 (punto de PARADA del plan). Parsers puros
  (`parse_bhac15_iso`, `convert_spectra_from_intermediate`) testeados con
  fixtures sintéticas.
- `scripts/check_g3_real_inputs.py`: chequeo opcional `--libraries` (verifica
  los 5 manifiestos; falla limpio listando ausentes; sin el flag el gate base
  sigue verde en máquinas sin datos).
- Tests nuevos: `tests/test_models_manifest.py`, `tests/test_models_cache.py`,
  `tests/test_fetch_g3_libraries.py` (25 tests). **Suite completa: 446 passed**
  (52 s), = 421 línea base + 25 nuevos. Gate base verde.

**Datos (descarga de las cinco familias) — 1/5 completada, 4/5 en PARADA:**

- ✅ **BHAC15** (`tracks_bhac15`): descargada, convertida, manifestada y
  verificada. Fuente
  `https://perso.ens-lyon.fr/isabelle.baraffe/BHAC15dir/BHAC15_iso.2mass`
  (formato verificado in situ 2026-07-16), 793 puntos, masa 0.01–1.4 M☉,
  Teff 1555–6767 K, edad 0.0005–10 Gyr, ~0.1 MB. `verify_manifest` OK
  (2 entradas). Cobertura D5 (≥0.01 M☉) satisfecha.
- ⏸ **ATMO2020** (`tracks_atmo2020`): host `opendata.erc-atmo.eu` es una SPA
  sin listado de directorio ni API resoluble; `noctis.astro.ex.ac.uk` no
  responde. Descarga manual → el subcomando imprime instrucciones y layout de
  `--input-dir`. PARADA consultiva (fuente sin descarga directa).
- ⏸ **BT-Settl CIFIST** (`bt-settl-cifist`): SVO Theory Server responde (200),
  pero es la rejilla grande (~130 nodos D4) → decisión humana de disco/tiempo
  y patrón de acceso SVO antes de tirar de ella. PARADA consultiva.
- ⏸ **X-shooter Class III jóvenes** (`templates_young`, Manara+2013/2017) y
  **SDSS campo** (`templates_field`, Kesseli+2017): VizieR/CDS en FITS/VOTable;
  requieren reducción al intermedio documentado (`index.csv` + ASCII
  `wave_A flux`) sin ADIVINAR marco/unidades (prohibido, patrón PARADA WP-5).
  Subcomando en modo instrucción hasta recibir el intermedio.

Los archivos de bibliotecas viven fuera del repo (`../Data/external_libraries`,
§0.5.4); el registro citable es este bitácora + los `PROVENANCE.json`/manifiestos.

- Siguiente: acordar con el humano la vía de adquisición de las 4 familias en
  PARADA (SVO para BT-Settl, descarga manual ATMO2020, reducción VizieR→
  intermedio para las dos plantillas). El software para ingerirlas ya está y
  testeado; solo falta el insumo. WP-G3R-3/4/5 (adaptadores) pueden avanzar en
  paralelo con fixtures sintéticas.
