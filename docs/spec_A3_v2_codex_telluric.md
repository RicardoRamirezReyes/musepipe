# Especificación A3 v2 · `00t_telluric` — brief de ejecución para Codex (ChatGPT 5.5)

> **v2, 2026-07-31.** Sustituye a `spec_A3_codex_telluric.md` (v1, 2026-07-01), que se
> conserva **intacta**: es el esquema bajo el que se emitieron los tres QC históricos de
> A3 (`ROXs12b_raw/stages/stage00t_qc.json`, `…/stage00t_realigned_qc.json` y la versión
> del canónico anterior a esta fecha). Dos cambios, ambos porque la v1 describía mal lo
> que la etapa necesita:
>
> 1. **§3 mide cuatro bandas, no tres**: entra **O₂ A (7590–7700 Å)**, la más profunda del
>    rango de MUSE. La v1 decidía sobre tres bandas que excluían justo la que más informa.
> 2. **§5 declara la apertura** (`input.primary_yx`, `input.aperture_radius_px`). Sin ellos
>    el número que publica el QC no se puede reproducir a partir del QC, y recuperarlos
>    costó un barrido a ciegas en el notebook de análisis.
>
> Ninguno de los dos mueve un veredicto: ver §3.4.

Fecha: 2026-07-01 (v1) · 2026-07-31 (v2). Etapa A3 (OPCIONAL) del plan
`docs/2026-07-10_plan_roxs12_reduccion_multimetodo.md`. Prerrequisito operativo: un cubo
de entrada con DATA+STAT y procedencia explícita (`ADP`, `A1` o `A2`). Si el
cubo viene del ADP histórico usado por la cadena anterior, A1/A2 no se
falsifican: el QC registra `upstream="ADP"` y el hash del cubo.

---

## 0. Rol, objetivo y definición de terminado

**Rol del agente**: decidir con métricas si la corrección telúrica aporta a la
ciencia de este run y, solo si aporta, ajustar molecfit sobre la primaria y
aplicar la transmisión al cubo. Como A2, es **condicional**: saltarla con
evidencia es un final exitoso.

**Contexto científico que delimita la etapa**: Hα (6563 Å) está esencialmente
libre de telúricas — **esta etapa nunca bloquea el análisis de Hα** y jamás
"corrige" la ventana 6540–6590 Å. La corrección importa solo para la fidelidad
del continuo rojo (>6800 Å) del compañero, que se usa en la comparación de
métodos (D1) y el modelado (Stage 08).

**Definición de terminado**:

1. Decisión `telluric_applied: true|false` tomada por las métricas de Fase 1,
   registrada con números en QC.
2. Si `true`: `cube_telcorr.fits` con STAT intacto y transmisión aplicada;
   `TELLURIC_TRANS.fits` (la transmisión 1D) guardada como producto propio.
3. `stage00t_qc.json` según §5; verificaciones de §6 en verde.
4. `musepipe/reduction/telluric.py` + tests versionados; reproducible con un
   comando.
5. Reporte final con el término sistemático residual declarado para D2.

---

## 1. Límites duros

1. Mismos límites de repo que A1/A2 §1 (runs históricos y `Data/` intocables;
   solo archivos nuevos de §7).
2. El cubo de entrada (A2 si hubo ZAP, A1 si no) es de solo lectura; la salida
   es un archivo nuevo.
3. **Prohibido corregir la ventana Hα** (6540–6590 Å) y la banda del láser
   (5780–6050 Å): la transmisión se fuerza a 1.0 en ambas, documentado en QC.
4. Parámetros de molecfit: partir de la configuración de referencia de esta
   spec (§4). Cambios de regiones de ajuste o de moléculas → preguntar antes.
5. Sin barridos de parámetros hasta que "pase". Una ejecución aprobada; si
   falla verificación, diagnóstico y pregunta.
6. Git: rama `stage-a3-telluric`; sin FITS commiteados.

---

## 2. Fase 0 — Entorno y entradas (gate)

1. Verificar molecfit: recipes `molecfit_model`, `molecfit_calctrans`,
   `molecfit_correct` visibles vía `esorex --recipes` (kit molecfit ≥ 4.x).
   Registrar versión exacta. Si no está instalado: proponer instalación
   (mismo canal que el pipeline MUSE de A1) y esperar confirmación.
2. Resolver el cubo de entrada desde config/CLI y registrar `upstream`:
   - `A2`: verificar QC A2 en verde (aplicada o saltada con veredicto);
   - `A1`: verificar QC A1 en verde. **A1 tiene DOS esquemas y los elige
     `chain.reduction_profile`** (v2):
     - `monolithic` → `stages/stage00r_qc.json`, y la puerta son sus cuatro
       fases (`gates_passed ⊇ {fase0..fase3}`);
     - `cascade` → `cube_telcorr_qc.json` (esquema `stream_combine_v1`), que
       **no tiene fases porque no las hay**: es reducción por exposición más
       combine por voxel. Su puerta es de **identidad** —que el QC declare como
       `output` exactamente el cubo que se va a medir— más `n_exposures ≥ 1` y
       `finite_fraction ≥ 0.5`. Atar el QC al dato es mejor garantía que una
       lista de fases. Exigirle las fases rechazaba todo objeto en cascada, y es
       lo que impedía lanzar A3 sobre ROXs 42B b.
   - `ADP`: verificar DATA+STAT y registrar que A1/A2 no aplican para esta
     entrada.
   En todos los casos registrar sha256 del cubo. Los `warnings` **no fatales**
   del QC de aguas arriba se copian a `input.upstream_warnings`, **nunca a
   `open_issues`**: ese campo es el canal bloqueante del bloque A.
3. **Punto de atención — GDAS/atmósfera**: molecfit puede requerir perfiles
   atmosféricos (GDAS) descargables; si la máquina no tiene red o el perfil de
   la fecha no está disponible, usar el perfil estándar equatorial documentado
   y anotarlo en QC como fuente de error.

## 3. Fase 1 — ¿Hace falta corrección? (gate de decisión)

1. Extraer el espectro de la primaria (apertura grande, alta S/N) del cubo de
   entrada. **La apertura usada (`primary_yx`, `aperture_radius_px`) se registra
   en el QC**: es lo único que hace reproducible el número de abajo.
2. Medir profundidad telúrica: flujo medio dentro vs fuera de las bandas O₂
   (**6864–6960 Å banda B**, **7590–7700 Å banda A**) y H₂O (7160–7340,
   8130–8350 Å), sobre continuo local interpolado. Registrar `depth_pct` por
   banda. Son las cuatro de `TELLURIC_BANDS` en
   `musepipe/reduction/telluric.py`, y `DEFAULT_FIT_REGIONS` se deriva de ellas.
3. Decisión:
   - Si el objetivo declarado del run no usa el continuo >6800 Å del
     compañero, o `depth_pct < 3%` en todas las bandas → `telluric_applied:
     false`, fin con QC y figuras.
   - Si `depth_pct ≥ 3%` en alguna banda de interés → continuar a Fase 2.
   - En duda sobre el objetivo científico → preguntar al usuario, no asumir.

### 3.4 Por qué entra O₂ A, y qué movió (v2)

Es la banda más profunda del rango de MUSE, y **todo el resto del repo ya la
trataba como tal** mientras la etapa que decide era la única pieza que la
ignoraba: `reduction/verify.py:DEFAULT_BAD_RANGES` la marca mala,
`telluric_lines.py` la cataloga `strong`, `qc/noise_decomposition.py` la excluye,
`docs/2026-07-16_g3_real_frozen_decisions.md` la enmascara en G3, y
`docs/a3_telluric_justification.md` §6 la llama «el rasgo telúrico con mayor
leverage» (73 % de profundidad máxima en el intento con molecfit).

Medido al re-emitir el QC del run canónico (2026-07-31), **ningún veredicto
cambia**:

| QC | antes | O₂ A medida | veredicto |
|---|---|---|---|
| `ROXs12b_realigned` (canónico) | máx 0.586 % (O₂ B) | **1.254 %** | `not_needed_shallow`, **igual** (1.254 < 3) |
| `ROXs12b_raw` (histórico, v1) | máx 6.76 % | no se re-emite | `needed`, intacto |
| `ROXs12b_raw` realineado (histórico, v1) | máx 7.34 % | no se re-emite | `needed`, intacto |

Lo que sí cambia y hay que saber:

- `fit.regions_A` pasa a cuatro regiones.
- **V2 cambia de soporte**: 7590–7700 Å pasa de «fuera de banda, no debe cambiar»
  a «banda corregida», porque `verify_outside_bands_unchanged` toma
  `corrected_bands=DEFAULT_FIT_REGIONS`. Solo afecta a runs que **apliquen**
  corrección; el canónico no la aplicó.
- D2 (`stage_x11_calibrate.CalibrationCorrections.telluric_bands_A`) hereda la
  banda, pero su término sistemático sale de
  `verification.v1_residual_pct_by_band`, **vacío** en el canónico → el término
  ya era cero y **sigue siendo cero**. El presupuesto de error de los espectros
  definitivos no se mueve (comprobado, no supuesto).

## 4. Fase 2 — Ajuste y aplicación (checkpoint humano)

### 4.1 Ajuste de molecfit sobre la primaria

- Entrada del ajuste: espectro 1D de la primaria (el de mayor S/N del campo).
- Moléculas: O₂ y H₂O. Regiones de ajuste iniciales: 6864–6960 Å (O₂ B-band),
  7160–7340 y 8130–8350 Å (H₂O), **evitando líneas estelares**: la primaria es
  una estrella joven K/M con fotosfera rica en líneas — punto de máxima
  atención de la etapa. Antes de ajustar, inspeccionar cada región contra una
  plantilla estelar aproximada (o el propio espectro suavizado) y recortar
  sub-regiones donde haya features estelares fuertes; las regiones efectivas
  quedan en QC.
- LSF: usar la resolución de MUSE (R~3000 en el rojo, variable con λ); dejar
  que molecfit ajuste el kernel dentro de límites físicos y registrar el valor.
- **Checkpoint humano**: figura del ajuste por región (dato, modelo, residuo)
  + χ² por región. No aplicar al cubo sin OK.

### 4.2 Aplicación al cubo

1. `molecfit_calctrans` → transmisión 1D en la malla del cubo; forzar
   transmisión = 1.0 en 6540–6590 y 5780–6050 Å.
2. Aplicar la MISMA transmisión a todos los spaxels (supuesto: telúrica
   espacialmente uniforme en un campo de arcsec — válido y documentado).
   Dividir DATA por la transmisión; **STAT se divide por transmisión²** (la
   corrección escala el error, esto sí se propaga, a diferencia de A2).
3. Header `HISTORY` con versión y regiones; log completo a `logs/`.

## 5. Esquema de `stage00t_qc.json`

```json
{
  "stage": "00t_telluric",
  "run_id": "ROXs12b_raw",
  "timestamp_utc": "...",
  "environment": {"molecfit_version": "...", "gdas_profile": "date|standard"},
  "input": {"cube": "...", "sha256": "...", "upstream": "A2|A1|ADP",
             "primary_yx": [100.0, 100.0], "aperture_radius_px": 8.0,
             "upstream_warnings": []},
  "decision": {"depth_pct_by_band": {}, "telluric_applied": false,
                "applied_to_cube": false,
                "science_needs_red_continuum": true,
                "user_checkpoint": "approved|not_needed"},
  "fit": {"molecules": ["O2", "H2O"], "regions_A": [], "excluded_subregions_A": [],
           "chi2_by_region": {}, "kernel": 0.0},
  "protected_windows_A": [[6540, 6590], [5780, 6050]],
  "products": {"cube_telcorr": "...", "transmission": "..."},
  "verification": {"v1_residual_pct_by_band": {}, "v2_outside_bands_change_pct": 0.0,
                    "v3_halpha_untouched": true, "v4_transmission_physical": true,
                    "v5_stat_scaled": true},
  "open_issues": []
}
```

**`telluric_applied` es la DECISIÓN, no el hecho** (v2). Con veredicto `needed` sale
`true` con el cubo todavía sin tocar, y los dos QC históricos de ROXs 12 b lo tienen
`true` **habiéndola aplicado**: el campo solo no distingue los dos estados. El que sí
es **`decision.applied_to_cube`**, que nace en `false` y lo pone la fase de aplicación.
`telluric_applied` se deja como está para no mover el significado de lo congelado.

**`input.primary_yx` y `input.aperture_radius_px` son obligatorios** (v2):
`stage00t_qc_skeleton` los pide sin default, así que no existe la vía silenciosa
que los perdió. `primary_yx` va en `[y, x]`, como todo el repo.

### 5.1 Dos esquemas en circulación

Los QC de A3 anteriores a la v2 vienen en **dos formas incompatibles** y quien los
lea debe distinguirlas en vez de indexar a ciegas
(`scripts/build_review_notebooks.py` lo hace, y es donde está escrito el
criterio):

| | reducciones antiguas (a mano) | cadena multi-noche (código) |
|---|---|---|
| decisión | `threshold_pct`, `method`, `max_depth_pct` | `checkpoint_required` |
| ajuste | `airmass_std`, `airmass_sci`, `scaling` | `molecules`, `regions_A` |
| verificación | `v1_o2_depth_pre_post_pct`, `v2_outside_bands_unchanged` | `v1_residual_pct_by_band`, `v2_outside_bands_change_pct` |
| apertura | **sí** | **no** (v1) · **sí** (v2) |

La v2 **no** añade `max_depth_pct` ni `threshold_pct` al esquema del código: la
presencia de `decision.threshold_pct` es justo lo que usan los lectores para
saber qué esquema tienen delante, y añadirlo rompería esa distinción.

## 6. Verificaciones finales (solo si se aplicó; si no, documentar Fase 1)

- **V1 — Residuales**: en líneas telúricas no saturadas de las bandas
  corregidas, residuo < 3–5% del continuo (espectro primaria pre/post con
  zoom por banda).
- **V2 — Fuera de bandas, nada cambia**: razón post/pre = 1 ± 0.2% fuera de
  las bandas corregidas.
- **V3 — Hα intacta**: transmisión ≡ 1.0 en 6540–6590 Å verificada sobre el
  producto, y espectro del compañero idéntico ahí.
- **V4 — Transmisión física**: 0 < T ≤ 1 en todo λ; sin picos espurios
  (dT/dλ acotada fuera de bordes de banda).
- **V5 — STAT escalado**: STAT_post = STAT_pre/T² verificado numéricamente en
  una muestra de spaxels.

## 7. Organización del código

```text
musepipe/reduction/telluric.py       # decisión, driver molecfit, aplicación al cubo
scripts/telluric.sh                  # fachada CLI reproducible para A3
tests/test_telluric_decision.py      # depth_pct sobre espectro sintético con banda O2 fabricada
tests/test_telluric_apply.py         # aplicación de T al cubo: DATA/T, STAT/T², ventanas protegidas
```

Test de potencia (mismo espíritu que A2): fabricar una sobre-corrección del 5%
fuera de banda y comprobar que V2 la detecta.

## 8. Protocolo de parada y reporte

Preguntar cuando: el objetivo científico del continuo rojo sea ambiguo; χ² de
alguna región de ajuste sea malo tras excluir features estelares; V1–V5 fallen;
falten perfiles GDAS y el estándar cambie el residuo > 1%. Reporte final: mismo
formato que A2 §9, incluyendo el residuo telúrico declarado como sistemático
para D2.
