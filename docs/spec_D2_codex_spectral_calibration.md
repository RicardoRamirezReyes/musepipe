# Especificación D2 · `X11_spectral_calibration` — brief de ejecución para Codex (ChatGPT 5.5)

Fecha: 2026-07-01. Etapa D2 (ESENCIAL, nueva) del plan
`docs/plan_roxs12_reduccion_multimetodo.md`. Prerrequisitos: D1 cerrada con
método canónico ELEGIDO por el usuario; QC de A4 (factores); C1 (sistemático
PSF); A2/A3 (sistemáticos de cielo/telúricas, si corrieron).

---

## 0. Rol, objetivo y definición de terminado

**Rol del agente**: convertir el espectro canónico en el producto científico
final: λ corregida y en marco declarado, flujo en escala validada, continuo
estimado por dos vías, y un error total con presupuesto de sistemáticos
explícito. D2 aplica factores medidos aguas arriba; **no mide nada nuevo ni
corrige nada que no venga de un QC**. Cada número aplicado debe ser trazable
a la etapa que lo midió.

**Definición de terminado**:

1. `spec_final_object.fits` (formato `SpectrumProduct` + columnas de §3.4)
   con header que declara TODAS las correcciones aplicadas y sus fuentes.
2. Presupuesto de sistemáticos tabulado en `stage_x11_qc.json` (§5).
3. Verificaciones §6 en verde (incluye el sanity check de forma espectral).
4. `musepipe/stages/stage_x11_calibrate.py` + tests; suite intacta.

---

## 1. Límites duros

1. **Cada corrección viene de un QC aguas arriba, con su clave citada**:
   Δλ de A4/M1, factor de flujo de A4/M3, factores STAT de A4/M5+B1,
   sistemático PSF de C1, residuos de A2/A3. Un factor sin procedencia = no
   se aplica.
2. **Prohibido corregir dos veces** — los tres dobles conteos clásicos que
   esta etapa debe verificar ANTES de aplicar nada (§3.1):
   corrección baricéntrica (¿el cubo ya está en marco baricéntrico?),
   corrección de apertura (¿`apcorr` ya está aplicada en el producto?),
   factor STAT (¿`flux_err` ya viene escalado por C2/C3/C4?).
3. El método canónico es el que el usuario eligió en D1; D2 se ejecuta
   TAMBIÉN sobre los demás métodos (mismos factores) para que E1 compare,
   pero solo `spec_final_object.fits` lleva ese nombre.
4. Sin estética espectral: nada de suavizados, interpolación de canales malos
   ni "limpiezas" del espectro final. Los flags viajan, los datos no se tocan.
5. Git: rama `stage-d2-calibrate`.

---

## 2. Entradas

- Producto canónico de D1 (+ los demás métodos).
- `stage00q_qc.json` (A4): Δλ y término lineal, marco de λ, factor de flujo
  por banda Gaia, factores STAT.
- `psf_model.json` / QC de C1: sistemático del anillo del compañero.
- QC de A2/A3 si corrieron: término de cielo residual, residuo telúrico.
- Corrección baricéntrica: valor y estado del marco desde A4
  (`wavelength_frame`, `vbary_kms`).
- Config: RV sistémica esperada de ROXs 12 (literatura, con error) — solo
  para la verificación V4, no se aplica al espectro.

## 3. Operaciones

### 3.1 Auditoría anti-doble-conteo (gate, antes de tocar nada)

Leer headers/QC y llenar la tabla `already_applied` de §5: marco de λ del
producto (`WFRAME`), estado de `apcorr`, modo de error (`ERRMODE`), factores
ya incluidos en `flux_err`. Cualquier ambigüedad (header que no declara algo)
→ parar y preguntar; no deducir por el valor de los datos.

### 3.2 Longitud de onda

- Aplicar Δλ (constante o término lineal, según semáforo M1 de A4) a
  `wave_A`; registrar en header `WLCORR` con la fuente.
- Dejar el espectro en el marco que YA tiene (típicamente baricéntrico, de
  scipost) y declararlo; NO transformar de marco aquí. La conversión a marco
  estelar (RV sistémica) es responsabilidad de E1/modelado, con el valor de
  config.

### 3.3 Flujo

- Factor de escala global de A4/M3 (mediana de las bandas Gaia, con su error
  como sistemático `sys_fluxcal`); anotar el caveat de variabilidad de la
  primaria tal como A4 lo reporta.
- Verificar consistencia `apcorr` (ya aplicada) contra la curva de
  crecimiento de C1 una última vez (número en QC, no re-aplicación).

### 3.4 Continuo (dos estimadores, ambos guardados como columnas)

- `cont_runmed`: `continuum_running_median` existente (`musepipe/spectral.py`),
  ventana 80 Å, máscara de líneas estándar (Hα, Hβ, O I, ±15 Å) + flags.
- `cont_poly`: polinomio de grado ≤ 5 en log-λ con la misma máscara,
  sigma-clipping asimétrico (rechaza emisión hacia arriba con más cautela que
  absorción — el objeto puede tener líneas en emisión reales).
- `sys_continuum(λ) = |cont_runmed − cont_poly|` entra al presupuesto. Si en
  la región de Hα supera el 20% del error estadístico local, señalarlo: E1 lo
  necesita saber.

### 3.5 Error total

- `flux_err_total² = flux_err_stat² + Σ sys_i²` por canal, donde `flux_err_stat`
  = max(`flux_err`, `flux_err_emp`) por canal (regla conservadora, suavizada
  con mediana móvil de 21 canales para no heredar ruido del propio estimador
  de error).
- Sistemáticos incluidos por canal donde apliquen: PSF (C1, localizado),
  telúrico (A3, por banda), cielo (A2, ventanas de skylines), fluxcal
  (global), continuo (§3.4; como columna separada, NO sumada a
  `flux_err_total` — el modelado decide si la usa). Cada término con su
  fuente en QC.

## 4. Puntos donde mantener máxima atención

1. **Los tres dobles conteos de §1.2** — la auditoría §3.1 existe porque este
   es el modo de fallo más probable de una etapa "de pegamento".
2. **No convertir marcos de λ dos veces**: skylines validan el marco en A4;
   D2 solo aplica el offset residual. Si A4 dejó el marco en `unavailable`,
   D2 se bloquea para λ (aplica solo flujo/continuo) y lo dice.
3. **La regla max() del error estadístico**: aplicada por canal SIN suavizar
   produce un error "dentado" que sesga χ² del modelado; por eso el suavizado
   de mediana móvil está en la spec. No cambiar la ventana sin preguntar.
4. **El continuo no se resta**: se entrega como columnas. Restarlo es decisión
   de E1/modelado. D2 entrega estado, no interpretación.

## 5. Esquema de `stage_x11_qc.json`

```json
{
  "stage": "x11_spectral_calibration",
  "run_id": "...",
  "canonical_method": "psffit",
  "already_applied": {"wframe": "barycentric", "apcorr": true,
                       "stat_factors_in_flux_err": true},
  "wavelength": {"dlambda_A": 0.0, "linear_term": 0.0, "source": "stage00q_qc.m1",
                  "frame_final": "barycentric"},
  "flux": {"scale_factor": 0.0, "scale_err": 0.0, "source": "stage00q_qc.m3",
            "variability_caveat": true},
  "continuum": {"runmed_window_A": 80, "poly_deg": 5,
                 "sys_at_halpha_vs_staterr": 0.0},
  "error_budget": [
    {"term": "stat", "type": "per_channel", "median": 0.0, "source": "max(stat,emp) smoothed"},
    {"term": "psf", "type": "localized", "median": 0.0, "source": "stage_e01_qc.companion_ring_metric"},
    {"term": "fluxcal", "type": "global_pct", "value": 0.0, "source": "stage00q_qc.m3"},
    {"term": "sky", "type": "windows", "value": 0.0, "source": "stage00s_qc"},
    {"term": "telluric", "type": "bands", "value": 0.0, "source": "stage00t_qc"},
    {"term": "continuum", "type": "column_only", "median": 0.0, "source": "3.4"}
  ],
  "also_calibrated": ["aperture", "optimal_ls", "optimal_psfsub"],
  "open_issues": []
}
```

## 6. Verificaciones

- **V1 — Skylines en cero**: sobre el espectro de CIELO calibrado con el mismo
  Δλ (no sobre el objeto), centroides de 5 skylines → |residuo| < 0.05 Å.
  Cierra el lazo de la corrección de λ.
- **V2 — Errores**: figura `flux_err_stat`, cada sistemático y
  `flux_err_total` vs λ; el presupuesto se ve, no solo se tabula.
- **V3 — Continuo estable**: `cont_runmed` vs `cont_poly` superpuestos al
  espectro; diferencia < 1σ estadística en el 90% de canales buenos.
- **V4 — Sanity de forma**: espectro final (continuo normalizado) contra una
  plantilla M8–L0 joven de biblioteca pública (config: ruta/cita de la
  plantilla). NO es un ajuste: es un chequeo visual+correlación de que la
  pendiente y las bandas moleculares (TiO/VO) van en la dirección esperada.
  Discrepancia grosera = algo está mal aguas arriba (fluxcal, halo) → issue.
- **V5 — Multi-método**: los demás métodos calibrados con los MISMOS factores,
  superpuestos; las conclusiones de D1 no deben cambiar tras calibrar (si
  cambian, un factor se aplicó de forma inconsistente → bug).

## 7. Tests

```text
tests/test_calibrate_no_double.py   # producto que YA declara apcorr/baricéntrico aplicados → D2 no re-aplica (compara salida)
tests/test_calibrate_budget.py      # sistemáticos sintéticos conocidos → flux_err_total exacto analíticamente
tests/test_calibrate_maxrule.py     # stat vs emp alternantes → la regla max+suavizado da lo esperado, sin dientes
tests/test_calibrate_wl.py          # espectro con offset 0.08 Å fabricado → corregido y V1 pasa
```

## 8. Protocolo de parada y reporte

Preguntar cuando: §3.1 encuentre ambigüedad de doble conteo; A4 tenga M1 o M3
en rojo (¿aplicar igual con caveat o bloquear?); V4 muestre forma incompatible
con M8–L0; V5 altere conclusiones de D1. Reporte: tabla de correcciones
aplicadas con fuentes, presupuesto de errores (figura V2), verificaciones,
checklist de límites, comando de reproducción.

## Errata v1.2 (2026-07-25) — los espectros definitivos, la unidad y la primaria

Todo lo de abajo es **aditivo**: ni un umbral, ni el método canónico, ni la
fórmula de `flux_err_total` (§3.5) cambian. Verificado re-ejecutando D2 en los
dos objetos con línea base previa (`scripts/verify_bunit_rerun.py`): ningún
flujo ni error se movió.

1. **El entregable de D2 son los espectros definitivos**, no solo
   `spec_final_object.fits`: los **6** métodos (§1.3 ya lo mandaba) más la
   **primaria**, `spec_calibrated_psffit_star.fits`. La primaria pasa por la
   misma cadena que el compañero (`calibrate_star_product` reutiliza
   `calibrate_spectrum_product`) y añade la sección `primary_star` al QC. Su
   `flux_err_emp` es el **empírico de anillo** de C4 (dispersión del coeficiente
   del psffit entre controles: mide estabilidad del ajuste, no ruido de fotones)
   y se mantiene en **columna aparte** del presupuesto `sys_*`, etiquetado en
   cabecera (`EMPSRC`/`ERRSEP`). `also_calibrated` en §5 lista hoy los cinco
   métodos no canónicos, no tres.
2. **La unidad viaja con el dato.** Los productos declaran `BUNIT` y la escala
   física se resuelve con `musepipe.io.resolve_flux_unit`: knob de config →
   `BUNIT` del producto → `m3_flux.flux_unit_cgs` del QC de A4 → error explícito.
   El QC gana `flux.unit`, que además **contrasta** el `BUNIT` del producto con
   la unidad que usó M3 para medir `flux_factor`: si no coinciden, el factor se
   midió dividiendo por una y se aplica sobre la otra → `open_issues`.
3. **Tabla y figura de los espectros.** `spectra` en el QC (una fila por método,
   canónico primero, más la primaria; medianas en **7500–9000 Å**, donde el
   compañero se detecta; cociente del continuo al canónico vía
   `cont_runmed_biasref`) y `plots/stage_x11_spectra.png` en tres paneles
   (`musepipe.stages.definitive_spectra_figure`, compartida con el notebook).
   Avisos automáticos: `sgf` filtra el continuo por construcción, y un continuo
   rojo negativo que supera el error total es sobre-sustracción del halo AO
   cromático (`docs/d2_red_continuum_diagnosis.md`), no un error de signo.
4. **`sys_fluxcal` era idénticamente cero** porque A4/M3 publica `flux_factor`
   **sin barra de error** y en una sola banda (RP), así que §3.3 no tenía de
   dónde sacar el sistemático. Decisión del usuario (2026-07-25): **declararlo
   sin plegarlo** — nueva columna `sys_fluxcal_declared` = |flujo|·|1−`flux_factor`|
   (2.7% en ROXs 12 b, 4.4% en ROXs 42B b), fila `fluxcal_declared` de tipo
   `declared_not_applied` en `error_budget`, `flux.declared_*` en el QC y
   `SYSFLXD`/`SYSFLXDN` en cabecera. **`flux_err_total` NO cambia**: plegarlo
   movería el error del compañero, que sostiene decisiones congeladas (E1/E3/G3
   y los hashes de determinismo de G5). Si M3 llega a publicar su incertidumbre,
   esa se aplica por §3.3 y el término declarado desaparece — el mismo
   sistemático no se cuenta dos veces.

## Errata v1.1 (2026-07-15, era D1 v3)

Con el set de 6 métodos, el comparador del gate v3 (acuerdo de continuo
inter-método, control-referenciado) se elige del primer par primario de D1
que contiene al canónico, EXCLUYENDO `sgf`: el SGF pierde el continuo de la
compañera por construcción (Julo et al. 2025 §2.1; caveat registrado en la
spec D1 v3 §2), de modo que un ratio de niveles canónico-vs-sgf mediría el
método, no la sistemática. Fallback congelado: `lpm`, luego
`optimal_psfsub`. Motivación observada: con el par `psffit_vs_sgf` el ratio
de banda roja daba 29.7x, dominado por la pérdida estructural de continuo del
sgf y no por el residuo cromático de halo que el gate vigila.
