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

### WP-G3R-2 (cont.) · Adquisición autónoma — 2026-07-16

El usuario eligió «adquisición autónoma donde sea factible». Descargadas y
verificadas **3 familias más** (4/5 totales) con descargadores reales añadidos
al fetch script; formatos inspeccionados in situ (sin adivinar):

- ✅ **Kesseli campo** (`templates_field`, Kesseli+2017): CDS `J/ApJS/230/16/fits/`.
  62 plantillas O5→M9 + L0–L3 (selección: enanas metalicidad solar `+0.0_Dwarf`
  + compuestas sin sufijo; se excluyen gigantes/subenanas/no-solares). Formato:
  BinTable ext1 `LogLam` (log10 Å) + `Flux` normalizado; marco **vacío (SDSS)**.
  116.6 MB, `verify_manifest` OK (63 entradas).
- ✅ **Manara jóvenes** (`templates_young`, Manara+2013 J/A+A/551/A107 + 2017
  J/A+A/605/A86): CDS `sp/*_V.fit` (brazo VIS X-shooter). 39 plantillas (22+17),
  SpT G5–M (incluye tardías del 2017). Formato: WCS lineal 1-D; **el WAT1
  etiqueta "angstroms" pero los valores son nm** (545–1035 nm) — resuelto por
  rango físico (`to_angstrom`, ×10) y registrado en meta con nota; marco
  aire/vacío a verificar en WP-6. 22.8 MB, OK (40 entradas).
- ⏸→código listo **BT-Settl CIFIST** (`bt-settl-cifist`, Allard+2012): SVO SSAP
  `model=bt-settl-cifist`. Convertidor real añadido (`read_svo_spectrum_votable`,
  `select_btsettl_nodes`), **validado end-to-end con `--limit 2` nodos reales**
  (2 npz, manifest OK; artefacto borrado). La caja D4 (meta=alpha=0, Teff
  2000–4500, logg 3.5–5.5) = **146 nodos**; cada espectro SVO es el rango
  completo 0–1000 µm (~60 MB), sin recorte servidor → **~9 GB de transferencia**
  para la rejilla completa (recortada a 4000–10000 Å quedan ~3.4 MB/nodo,
  ~0.5 GB en caché). Excede el «cientos de MB» estimado → **pendiente de
  confirmación del usuario** para lanzar la descarga completa (comando
  `python scripts/fetch_g3_libraries.py bt-settl`).
- ✅ **ATMO2020** (`tracks_atmo2020`, Phillips+2020): resuelto SIN descarga
  manual — el SPA no servía, pero los tracks están en el MISMO host ENS Lyon que
  BHAC15: `.../isabelle.baraffe/ATMO2020/ATMO_2020_models.tar.gz` (44 MB, con
  `README`). Convertidor real: lee en memoria los miembros
  `evolutionary_tracks/ATMO_CEQ/MKO_WISE_IRAC/*_ATMO_CEQ_vega.txt` del tar (76
  masas), columnas Mass/Age/Teff/Luminosity/Radius/log(g); la columna
  "Luminosity" es log10(L/Lsun) pese a la etiqueta (como BHAC15) → L linear.
  3001 puntos, masa 0.0005–0.075 M☉ (0.5–75 MJup), Teff 200–3075 K. Verificada.
  Licencia: pública, los autores piden contacto antes de publicar (registrado
  en provenance).

- ✅ **BT-Settl CIFIST** (`bt-settl-cifist`, Allard+2012): descarga completa
  terminada — **146/146 nodos, 0 fallos**, ~582 MB en caché (recortada a
  4000–10000 Å). Rejilla D4 cubierta: Teff 2000–4500 K (30 valores), logg
  {3.5,4.0,4.5,5.0,5.5}; 146 de 150 combos (4 no existen en CIFIST — normal).
  Flujo finito y positivo, unidades registradas (erg/cm²/s/Å, vacío sintético).
  `verify_manifest` OK (147 entradas).

Software añadido al fetch script: `_download_text`/`_hrefs`; lectores puros
`wave_from_linear_wcs`, `to_angstrom` (resuelve unidad por rango físico, PARADA
si absurdo), `read_kesseli_fits`, `read_manara_visual_fits`, `read_svo_spectrum_votable`,
`select_btsettl_nodes`, `_kesseli_wanted`, `parse_atmo2020_ceq`; `_download`
con reintentos; bt-settl reanudable (salta nodos cacheados, aísla fallos);
`--limit`; `parse_atmo2020_ceq`. Tests nuevos en
`tests/test_fetch_g3_libraries.py` (18 en total, todos OFFLINE con fixtures
sintéticas). **Suite completa 457 passed** (53 s).

**WP-G3R-2 CERRADO (2026-07-16)**: las 5 familias descargadas, convertidas al
formato de caché interno y verificadas (`fetch verify` y
`check_g3_real_inputs.py --libraries` verdes). Todo autónomo (ninguna descarga
manual necesaria al final). Los datos viven fuera del repo
(`../Data/external_libraries/`, ~0.7 GB en disco: bt-settl 556 MB, campo
112 MB, atmo2020 43 MB, jóvenes 22 MB, bhac15 0.1 MB); el registro citable es
esta bitácora
+ los `PROVENANCE.json`/manifiestos por familia. Resúmenes numéricos: BHAC15
793 pts; ATMO2020 CEQ 3001 pts (0.5–75 MJup); BT-Settl 146 nodos; Kesseli 62
plantillas (O5–L3); Manara 39 plantillas VIS (G5–M).

- Siguiente: WP-G3R-3 (adaptador BT-Settl), WP-G3R-4 (adaptadores plantillas),
  WP-G3R-5 (adaptadores tracks) — independientes entre sí tras la fase 2, se
  validan con fixtures sintéticas y usan los tests `external_data` locales
  contra estas familias reales; y WP-G3R-6 (preparación del espectro).

## WP-G3R-3 · Adaptador BT-Settl (`SpectralLibrary`) — 2026-07-16

- `musepipe/models/btsettl.py`: clase `BTSettlLibrary` (implementa
  `SpectralLibrary`). `__init__(family_dir, *, citation, version, cache_size=64)`
  — cita obligatoria (RuntimeError, patrón `CCMExtinction`); llama a
  `verify_manifest` una vez; construye el índice `{(teff,logg)→archivo}` leyendo
  SOLO el `meta_json` de cada npz (no carga wave/flux). `grid()` devuelve los
  ejes reales presentes. `get(teff=,logg=)`:
  - nodo exacto → espectro NATIVO sin remuestrear (camino principal del ajuste
    de rejilla WP-8);
  - interpolación BILINEAL en log-flujo entre los 4 vecinos. Como los nodos NO
    comparten grilla de longitud de onda (270k/210k/195k puntos según Teff), los
    4 vecinos se alinean por `np.interp` sobre la grilla del vértice inferior
    (rápido) antes de combinar en log; el remuestreo con conservación de flujo
    sobre la grilla observada lo hace luego `prepare_template`.
  - `meta` registra `nodes`, `weights` e `interp_error_halfstep` (medio paso
    local, spec §4.3); fuera de rejilla o nodo faltante → RuntimeError.
  - caché LRU acotada (≤64 espectros) con `OrderedDict`.
- `pytest.ini` nuevo: registra el marker `external_data` y fija
  `testpaths=tests` (colección idéntica a la actual).
- Tests `tests/test_models_btsettl.py` (8): contrato del protocolo; bilineal
  EXACTA para log-flujo lineal en (Teff,logg); nodo exacto nativo; interp 1-D en
  un eje; cita obligatoria; fuera de rejilla; nodo faltante. Test
  `@pytest.mark.external_data` que corre contra la familia real (146 nodos):
  `grid()` cubre D4 y `get(3000,4.0)` da flujo finito y positivo en 4000–10000 Å
  — **EJECUTADO y verde en esta máquina** (no skip). **Suite completa 465 passed**.
- Parada §: la rejilla real cubre D4 (Teff 2000–4500, logg 3.5–5.5) → sin parada.
- Siguiente: WP-G3R-4 (adaptadores de plantillas jóvenes/campo).

## WP-G3R-4 · Adaptador de plantillas empíricas (jóvenes + campo) — 2026-07-16

- `musepipe/constants.py`: codificación SpT numérica CONGELADA — `spt_code`/
  `spt_label` (M0=0.0…M9=9.0, L0=10.0…; clases previas negativas K5=−5.0;
  medio subtipo 0.5; secuencia OBAFGKMLTY). Con tests (roundtrip, redondeo a
  medio subtipo, etiqueta inválida → ValueError).
- `musepipe/models/templates.py`: `EmpiricalTemplateLibrary` (`SpectralLibrary`)
  con `gravity_class ∈ {young, field}`. Cita y clase obligatorias (RuntimeError);
  `verify_manifest`; índice `{spt_code → [(archivo, meta)]}` desde `meta_json`.
  `grid()` devuelve los códigos SpT presentes. `get(spt=)` NO interpola: elige el
  SpT disponible más cercano (`meta['spt_delta']`), y ante varios objetos del
  mismo SpT (biblioteca joven) devuelve el primero determinista (`n_at_spt`).
  `meta` incluye `resolution_fwhm_A` (de la fuente si la declara, si no del
  parámetro de `__init__`), `gravity_class`, `citation`.
- `musepipe/models/prep.py`: `prepare_template` gana `template_fwhm_A=None` y
  `return_flag=False` (D2): si FWHM_plantilla < LSF degrada con kernel
  √(LSF²−FWHM²); si ≥ LSF no degrada y marca `resolution_mismatch`. Default y
  firma retro-compatibles (fit.py y tests previos sin cambios).
- Cobertura real verificada: **joven G5–M9.5 (M0–M9 cubierto, sin parada)**;
  campo O5–L6 (secuencia M completa). Ambas cargan.
- Tests `tests/test_models_templates.py` (11): codificación SpT; contrato;
  regla del más cercano; múltiples por SpT; resolución surfaced (init + fuente);
  clase/cita obligatorias; regla de resolución de `prepare_template` en los dos
  sentidos + retro-compat; `external_data` contra ambas bibliotecas reales
  (EJECUTADO verde). **Suite completa 476 passed**.
- Siguiente: WP-G3R-5 (adaptadores de tracks BHAC15/ATMO2020).

## WP-G3R-5 · Adaptadores de tracks (BHAC15 y ATMO2020) — 2026-07-16

- `musepipe/models/tracks.py`: clase `TrackGrid` (`EvolutionaryModel`), una por
  familia desde el npz cacheado. Cita obligatoria (RuntimeError).
  `__init__(npz_path, *, family, citation, version)`. Las rejillas publicadas son
  IRREGULARES en (masa, edad); `lookup(l_bol, age, *, teff=None)` las invierte por
  interpolación lineal dispersa (baricéntrica sobre triangulación de Delaunay,
  `scipy.spatial`) en (log edad, log L); si se da `teff`, modo alternativo en
  (log edad, log Teff) para el chequeo de consistencia. Devuelve
  `mass_msun/radius_rsun/logg/teff_k` (o `l_bol_lsun` en modo teff) +
  `*_err_interp` (medio spread local del símplex, §4.3) + `in_range` + `clamped`
  + `mode`. Fuera del casco convexo → `in_range=False` y NaN (NUNCA extrapola).
  `sample(l_bol_samples, age_samples)` vectorizado (find_simplex + baricéntricas
  con einsum) para el MC de 4000 muestras sin bucle Python.
- Tests `tests/test_models_tracks.py` (9): con mini-tracks sintéticos AFINES en
  (log edad, log L) → recuperación EXACTA (masa y teff); `err_interp` coherente
  (>0, acotado); fuera de rango → `in_range=False`, NaN, `clamped`; modo teff;
  cita obligatoria; `sample` vectorizado ≡ lookup; dispersión entre dos familias
  sintéticas. Test `@pytest.mark.external_data` contra BHAC15 y ATMO2020 reales:
  `lookup(1e-3 Lsun, 0.006 Gyr)` (edad D12) da masa finita in-range en al menos
  una familia — EJECUTADO verde. **Suite completa 485 passed**.
- Parada §: formatos de ambas familias legibles (columnas identificadas en
  WP-G3R-2) → sin parada.
- Con esto **WP-G3R-3/4/5 (los tres adaptadores) cerrados**. Siguiente:
  WP-G3R-6 (preparación del espectro observado, `musepipe/models/observed.py`).

## WP-G3R-6 · Preparación del espectro observado — 2026-07-16

- `musepipe/models/observed.py` — ÚNICA puerta de datos reales a los ajustes:
  - `load_final_spectrum(run_paths, *, err_column="flux_err_total")`: lee la
    ext `SPECTRUM` de `spec_final_object.fits`; RuntimeError si falta columna.
  - `build_fit_masks(cfg, wave, run_paths) -> (mask, provenance)` (D9, True =
    excluir): máscara stage04b + ventanas ±`line_window_kms` (300 km/s) en torno
    a las líneas del catálogo G2 (`rest_A` de `g2_line_measurements.csv`) +
    bandas telúricas de config.
  - `rebin_for_fit(...)` (D8): bins de `n_channels`, descarta bins con
    >`max_masked_frac` enmascarado, `flux_bin`=media de canales no enmascarados,
    varianza = (Σσ²)/N² × (n/n_eff) con el `n_eff_over_n_by_block` del bloque G1
    que contiene el bin. **Hallazgo clave**: los `block_bounds` de G1 indexan el
    subespacio de canales BUENOS (contiguos, último z1 = 3465 = 3681 − 216 malos,
    verificado leyendo `spectral_covariance_blocks`), así que el mapeo bin→bloque
    se hace por longitud de onda de canal-bueno vía la bad-mask; **PARADA** si
    `n_good ≠ cobertura de block_bounds` (runs desalineados). Devuelve
    `n_eff_total` para el ranking (§4.4).
  - `fit_spectrum(cfg, run_paths) -> FitSpectrum` (dataclass congelada:
    wave_bin/flux_bin/err_bin/n_bins/n_eff/mask_provenance).
  - `plot_fit_spectrum(...)`: figura QC (canal gris + rebineado con barras +
    máscaras sombreadas + rango D8) con `Figure/FigureCanvasAgg` (sin pyplot).
- Tests `tests/test_models_observed.py` (11) con run sintético en tmpdir:
  esquema + columna faltante; ventana de línea en 6562.8 Å enmascara los canales
  correctos; PARADA por bad-mask de shape incorrecto; rebineo conserva flujo;
  inflado de varianza analítico (bloque n_eff/n=1 → factor 1; =0.5 → ×√2);
  descarte de bin >50% enmascarado; `wave_range` respeta D8; PARADA por
  cobertura de covarianza ≠ n_good; `fit_spectrum` end-to-end sintético;
  `plot_fit_spectrum` corre. **Suite completa 496 passed**.
- **Anti-sesgo respetado**: NO se ejecutó `fit_spectrum` sobre el espectro real
  (solo fixtures sintéticas); el rebineado real ocurre en WP-G3R-11.
- Siguiente: WP-G3R-7 (`stage_g3_template_fit`: SpT por plantillas + índices).

## WP-G3R-7 · `stage_g3_template_fit` — SpT plantillas + índices — 2026-07-16

- `musepipe/models/template_fit.py`:
  - `fit_templates(fit_spec, library, extinction, *, av_axis, lsf_fwhm_A,
    veiling=False, ...)`: por cada plantilla del `grid()`, χ² con (A_V, escala)
    libres (escala analítica reusando `_best_scale_chi2` de fit.py, sin tocarlo);
    con `veiling=True` añade componente aditiva no negativa a·(λ/λ0)^α por NNLS
    (`scipy.optimize.nnls`) sobre rejilla (A_V, α) (D10). Devuelve ranking
    ordenado (spt, chi2, chi2_red, av_best, scale_best, veiling) + resumen
    (spt_best, intervalo SpT por ΔΧ²≤1, n_eff, flag de inflado D11).
  - `fit_powerlaw` (proxy no estelar D3, F_λ∝(λ/λ0)^α). `classify_gravity` →
    ΔΧ² entre la mejor de cada clase `{young, field, nonstellar}` (insumo G4 T3).
- `musepipe/models/indices.py`: `measure_indices(wave, flux, err, definitions,
  ...)` — cocientes de flujo medio en ventanas SOLO desde config (D7), con
  RuntimeError si una ventana cae fuera de cobertura o >50% enmascarada; error
  por MC gaussiano sembrado. `indices_to_spt(values, calibration)` con
  polinomio SpT del paper (config); combina índices (media + dispersión).
- `musepipe/stages/stage_g3_template_fit.py` (patrón accretion): corre las dos
  vías sobre `fit_spectrum`, `classify_gravity`, variante veiling; escribe
  `stages/g3_template_fit.json` (ranking, ΔΧ² por clase, índices, veiling) y
  `stages/g3_rows_template.json` con filas SEPARADAS `spectral_type`
  (empirical_inference; err_sys = |plantillas−índices| ⊕ shift veiling),
  `spt_templates`, `spt_indices` (la discrepancia NO se promedia). `compute_`
  admite `fit_spec/per_channel/libraries` inyectados (testeable sin run real).
- Tests `tests/test_g3_template_fit.py` (8): V1 sintético (recupera SpT exacta +
  A_V a ≤0.5); ranking degrada con SpT distante; veiling inyectado recuperado
  (<40%); `classify_gravity` distingue clases; índices caso analítico +
  ventana fuera de cobertura → RuntimeError; power-law recupera α; ensamblado de
  etapa con inyección. Sin ejecución sobre datos reales. **Suite 504 passed**.
- **Pendiente D7 (checkpoint antes de WP-11)**: transcribir del paper las
  ventanas numéricas `g3_spt_indices` + `g3_spt_indices_calibration` (Riddick+2007,
  Martín+1999 PC3, Slesnick+2004). El código las consume desde config; sin ellas,
  la fila `spt_indices` sale `not_constrained` (honesto, no inventa).
- Siguiente: WP-G3R-8 (`stage_g3_atmo_fit`: Teff/A_V/logg/Ω + mapas ΔΧ²).
