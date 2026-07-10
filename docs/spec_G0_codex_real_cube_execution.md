# Especificación G0 · `real_cube_execution` — brief de ejecución para Codex (ChatGPT 5.5)

Fecha: 2026-07-06. Fase G0 (BLOQUEANTE) del plan
`docs/00_plan_caracterizacion_espectroscopica.md`. Prerrequisitos: specs
A2–A4, B3, C1–C4, D1–D2, E1–E4 implementadas (verificado en auditoría);
cubo ADP disponible. Aplica §7 del índice (reglas obligatorias).

**Propósito científico**: obtener por primera vez los productos reales de la
cadena multi-método (espectros por tres extractores, veredicto inter-método,
espectro canónico calibrado, resultado de Hα con throughput medido) sobre el
cubo ADP de ROXs 12, y dejar constancia de todo fallo de la infraestructura
ANTES de modificarla. **Esta fase es ejecutar, inspeccionar y corregir lo
mínimo; no es reescribir.**

---

## 0. Rol, objetivo y definición de terminado

**Rol del agente**: operador y diagnosticador de la cadena existente, no
autor de ciencia nueva ni de refactors.

**Definición de terminado**:

1. Run nuevo `runs/<RUN_ID>/` (proponer `ROXs12b_G0`; confirmar con el humano)
   con productos y QC de: A4 → (A2 condicional, A3 opcional según sus
   decisiones automáticas ya implementadas) → B3 → E01 → X01 → X02 → X03 →
   X10 → X11 → H04 → H01 → H02 → H03.
2. `docs/g0_execution_log.md`: bitácora etapa por etapa — comando exacto,
   duración, verificaciones de la spec original en verde/rojo, fallos con
   traza completa, y correcciones aplicadas (diff mínimo, justificación).
3. Comparación cuantitativa contra la cadena histórica local-surface (§5.3).
4. `pytest` completo verde al inicio y al final; tests que ya fallaban
   documentados con traza en la bitácora ANTES de cualquier edición.
5. Ningún criterio congelado alterado (detección H01, bandas D1, grilla E4,
   veredictos X10).

---

## 1. Límites duros

1. **Prohibido reescribir etapas.** Solo se permiten correcciones mínimas
   para que una etapa corra (imports rotos, rutas de producto mal resueltas,
   incompatibilidades de claves de config). Cada corrección: commit propio,
   test que la cubra si es lógica, y entrada en la bitácora. Si el arreglo
   necesario supera ~30 líneas o toca el núcleo numérico de una etapa →
   protocolo de parada.
2. Prohibido tocar los runs históricos (`ROXs12b`, `ROXs12b_short`, clones de
   inyección); son la referencia de comparación.
3. Ejecución SOLO vía los entrypoints oficiales (`scripts/*.sh` o
   `run_stageXX(run_id=...)` desde un shell limpio). **Prohibido ejecutar
   etapas desde celdas de notebook**: los notebooks son solo inspección
   posterior. Esto garantiza que no se hereden variables residuales de
   sesiones interactivas.
4. Config del run versionada: `runs/<RUN_ID>/config/config.json` creada al
   inicio, con hash registrado en la bitácora; cambios de config posteriores
   → nueva entrada en bitácora con diff.
5. Sin rutas absolutas en config ni código; el cubo ADP se referencia por
   ruta relativa al repo o por variable de entorno documentada.
6. Inyecciones (H04) solo en clones, según spec E4 (subregión mínima).
7. Git: rama `phase-g0-real-cube`.

---

## 2. Precondiciones e inspección obligatoria (antes de ejecutar nada)

1. Leer completa la auditoría, el índice G y las specs A4, B3, C1–C4, D1, D2,
   E1–E4 (esta fase las ejecuta; sus verificaciones internas mandan).
2. Inventario del cubo de entrada, registrado en la bitácora y en el config:
   - archivo, `sha256`, tamaño; identificación inequívoca
     (`ADP.2022-09-12T17_17_39.371.fits` según memoria del proyecto —
     verificar contra lo que exista en disco, no asumir);
   - extensiones presentes (DATA, STAT, DQ si existe) con dimensiones,
     `BUNIT`, WCS completo (CRVAL/CRPIX/CD, `CTYPE3`, unidades de λ);
   - marco de longitud de onda (topocéntrico/baricéntrico) según los headers
     que `qc/cube_qc.py` sabe leer;
   - censo de no-finitos por extensión (fracción de NaN/Inf, distribución
     espacial en imagen colapsada) y de máscaras implícitas (bordes, slices);
   - rango espectral y muestreo (Δλ), número de canales.
3. Verificar el entorno: `environment.yml` instalado, versiones en QC.
4. `pytest -q` completo: registrar el estado base (fallos preexistentes con
   traza). Este es el punto de referencia "antes".

## 3. Orden de ejecución y decisiones por etapa

```text
A4 (QC cubo: λ, LSF, flujo, STAT)
 └─ decide: ¿STAT utilizable? ¿offset λ? → consumidos por X01–X11
A2 (ZAP) solo si el QC de A4 muestra residuales de cielo sobre umbral (spec A2)
A3 (telúrica) opcional; no bloquea Hα (spec A3)
B3 (localización) → coordenadas canónicas en QC; adiós hardcodeo
E01 (PSF cromática) → psf_model para X02/X03/H04
X01, X02, X03 (tres extractores, config de producción)
X10 (comparación, veredicto congelado) → elige canónico
X11 (calibración + presupuesto de errores) → espectro canónico
H04 (inyección-recuperación multi-método, grilla congelada de E4)
H01 (detección Hα, criterio congelado) → H02 (artefactos) → H03 (límites)
```

Reglas de decisión:

- **STAT**: si A4 concluye factor fuera de [0.8, 1.5] o estructura no física,
  activar el plan B empírico previsto por las specs (no inventar factores);
  anotar la decisión en bitácora y QC.
- **B3 detecta el compañero < 5σ en continuo**: PARAR (protocolo §8); cambia
  el plan científico.
- **X10 = `divergent_*`**: no avanzar a X11 con un canónico dudoso; ejecutar
  el árbol de decisión de D1 (refinar C1 una vez) y si persiste → parar.
- **H01**: el criterio de detección NO se toca; el resultado (detección,
  candidato o no-detección) se reporta tal cual con su FAP.

## 4. Puntos de atención

1. **Primera colisión código-realidad**: las etapas nuevas solo han visto
   datos sintéticos. Espera fallos de supuestos (headers ausentes, unidades,
   NaN en bordes). El valor de G0 es el catálogo de esos fallos — documenta
   ANTES de arreglar.
2. **Encadenamiento de hashes**: cada etapa declara el sha256 de su entrada;
   verificar la cadena completa al final (patrón de F1). Ruptura = mezcla de
   productos = invalida el run.
3. **Memoria**: crop 170×170 × ~3700 canales; usar las rutas low-memory donde
   existan; monitorear pico de RAM por etapa en la bitácora.
4. **No interpretar**: cifras de Hα, flujos y límites se transcriben al
   reporte sin lectura científica; la interpretación espera a G1 (validación)
   — regla anti-sesgo del plan maestro.

## 5. Salidas y esquema de QC

### 5.1 Productos

Los propios de cada etapa según sus specs, bajo `runs/<RUN_ID>/`. Ningún
producto nuevo se inventa en G0.

### 5.2 `stage_g0_qc.json` (resumen de fase, patrón F1 reducido)

```json
{
  "stage": "g0_real_cube_execution",
  "run_id": "...",
  "input_cube": {"file": "...", "sha256": "...", "extensions": ["DATA", "STAT"],
                  "wave_frame": "...", "nan_fraction_data": 0.0},
  "stages_executed": [{"stage": "A4", "status": "green|red|skipped",
                        "duration_s": 0, "issues": []}],
  "stat_verdict": {"usable": true, "factor": 0.0, "source": "stage00q_qc"},
  "hash_chain_ok": true,
  "fixes_applied": [{"file": "...", "lines": 0, "reason": "...", "commit": "..."}],
  "pytest_before": {"passed": 0, "failed": 0, "failing": []},
  "pytest_after": {"passed": 0, "failed": 0, "failing": []},
  "frozen_criteria_untouched": true,
  "open_issues": []
}
```

### 5.3 Comparación con la cadena histórica

Tabla `tables/g0_legacy_comparison.csv`: para el espectro de X01 (apertura,
el método comparable) vs `stage05_box3x3` / `stage07` históricos —
flujo en las bandas congeladas de D1, métricas de las líneas de stage07, ruido
por canal. Discrepancias > 2σ se listan como issues (posible efecto de
coordenadas B3 vs hardcodeadas, o de calibraciones A4: identificar cuál).

## 6. Verificaciones

- **V1 — Inventario**: cubo identificado por sha256; extensiones, unidades,
  WCS y censo de no-finitos en QC; figura imagen blanca con máscara de
  no-finitos superpuesta.
- **V2 — Cadena completa**: las 13+ etapas con status en QC; ninguna `red`
  sin issue asociado; cadena de hashes intacta.
- **V3 — STAT**: veredicto de A4 documentado y propagado coherentemente (la
  misma decisión en X01–X11; grep de QCs lo confirma).
- **V4 — Legacy**: tabla §5.3 generada; cada discrepancia > 2σ con issue y
  causa candidata.
- **V5 — Higiene**: `pytest` después ≥ antes (ningún test nuevo roto);
  bitácora completa; criterios congelados intactos (diff de los módulos de
  criterios == vacío).
- **V6 — Diagnósticos por etapa**: las figuras de verificación que cada spec
  original define existen en `plots/` para el run nuevo.

## 7. Tests

G0 no añade módulos científicos; añade solo:

```text
tests/test_g0_hash_chain.py      # utilidad de verificación de cadena de hashes sobre QCs de un run (sintético)
tests/test_g0_legacy_compare.py  # la comparación §5.3 con espectros sintéticos idénticos → cero discrepancias
```

más los tests que exijan las correcciones mínimas de §1.1.

## 8. Protocolo de parada

Preguntar (con figura/traza) cuando: el compañero no se detecta en continuo
(B3 §4.3); un arreglo necesario supera los límites de §1.1; X10 persiste
`divergent_*` tras una iteración de C1; STAT y ruido empírico discrepan > 2×
de forma estructurada; la comparación legacy muestra discrepancias > 5σ; el
tiempo de H04 proyectado supera 4 h (límite de la spec E4).

## 9. Riesgos

- **Científicos**: interpretar prematuramente resultados no validados
  (mitigación: §4.4); heredar sesgo de coordenadas históricas (mitigación:
  B3 + V4).
- **Técnicos**: fallos en cascada por un supuesto de formato (mitigación:
  inventario §2.2 antes de correr); agotar memoria (rutas low-memory);
  contaminar el run con estado de notebooks (prohibición §1.3).

## 10. Reporte final de Codex

Bitácora completa (`docs/g0_execution_log.md`), QC de fase §5.2, tabla legacy
§5.3, verificaciones V1–V6 con figuras, lista de fixes con commits, tests
antes/después, issues abiertos priorizados (bloqueante/mayor/menor para G1),
y comando(s) exactos de reproducción del run completo.
