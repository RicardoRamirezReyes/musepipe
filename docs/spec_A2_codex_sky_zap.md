# Especificación A2 · `00s_sky_zap` — brief de ejecución para Codex (ChatGPT 5.5)

Fecha: 2026-07-01. Etapa A2 del plan `docs/2026-07-10_plan_roxs12_reduccion_multimetodo.md`.
Esta etapa acepta dos puntos de entrada:

1. `provenance="raw_reduction"`: A1 terminada según
   `docs/spec_A1_codex_raw_reduction.md` (existe
   `runs/ROXs12b_raw/raw_reduction/DATACUBE_FINAL.fits` con QC en verde).
2. `provenance="adp"`: cubo ADP ya reducido por ESO/esorex, usado por la cadena
   histórica (`ADP.2022-09-12T17_17_39.371.fits`). En este modo **no se
   falsifica A1**: el QC de A2 registra la procedencia `adp`, el hash del cubo y
   que A1 no aplica para esta entrada.

---

## 0. Rol, objetivo y definición de terminado

**Rol del agente**: decidir con métricas si el cubo de A1 necesita limpieza de
residuales de cielo, y si la necesita, aplicar ZAP sin distorsionar la señal
astrofísica. Esta etapa es **condicional**: "no hace falta ZAP" es un
resultado tan válido como "ZAP aplicado", y ambos terminan la etapa con éxito.

**Objetivo**: producir `cube_zap.fits` (o un veredicto documentado de que no se
necesita), con QC que demuestre dos cosas a la vez: los residuales de skylines
bajaron y el espectro de las fuentes NO cambió fuera de las ventanas de cielo.

**El peligro central de A2**, y el criterio que domina todo el diseño: ZAP es
PCA sobre spaxels de cielo; si la máscara de fuentes es pobre o el campo tiene
poco cielo, las eigencomponentes absorben señal estelar y ZAP puede **restar
flujo del compañero o fabricar/borrar una línea en Hα**. Ante cualquier duda
entre "limpiar más" y "no tocar la señal", se elige no tocar la señal.

**Definición de terminado (todas obligatorias)**:

1. Decisión `zap_applied: true|false` tomada por las métricas de la Fase 1, no
   por juicio propio, y registrada en QC con sus números.
2. Si `true`: `cube_zap.fits` existe dentro del directorio de reducción del run
   activo, con la extensión STAT copiada intacta del cubo de entrada.
3. `stage00s_qc.json` completo según §6.
4. Las 6 verificaciones de §7 ejecutadas (o V1–V2 si la etapa se salta) y en
   verde.
5. `musepipe/reduction/sky_zap.py` + tests versionados; la etapa se repite con
   un comando.
6. Reporte final (§9).

---

## 1. Límites duros

1. Mismos límites del repo que A1 §1: no tocar runs históricos ni
   `~/Documents/GitHub/Muse/Data/`; no modificar módulos existentes de
   `musepipe/`; solo crear archivos nuevos listados en §8.
2. **No sobreescribir** `DATACUBE_FINAL.fits`: ZAP escribe a un archivo nuevo.
   El cubo de entrada es de solo lectura para esta etapa.
3. **Parámetros de ZAP en default.** La única entrada obligatoria propia es la
   máscara de fuentes (§4). Cualquier otro parámetro (`cfwidthSVD`,
   `cfwidthSP`, número de eigenvalores por segmento) solo se cambia tras
   preguntar al usuario con evidencia de por qué el default falla.
4. **No iterar buscando "el mejor resultado"**: se ejecuta ZAP una vez con la
   configuración aprobada. Si las verificaciones fallan, se diagnostica y se
   pregunta; prohibido el barrido silencioso de parámetros hasta que las
   métricas pasen (eso invalida la estadística aguas abajo).
5. ZAP **no** se aplica a la región 5780–6050 Å con expectativa de arreglarla:
   esa banda (láser AO) se excluye de todas las métricas de decisión y de
   verificación. Sigue enmascarándose aguas abajo.
6. Git: rama `stage-a2-sky-zap`, commits por fase, sin FITS en el repo.
7. Presupuesto: ZAP sobre un cubo MUSE completo usa RAM del orden del tamaño
   del cubo (~5–10 GB); verificar antes de correr. Disco: +1 cubo.

---

## 2. Fase 0 — Entorno y entradas (gate)

1. Instalar/verificar el paquete ZAP **del proyecto musevlt** (`zap` en PyPI,
   repo `musevlt/zap`). Medida anti-error obligatoria: tras `import zap`,
   verificar que el módulo expone `zap.process` y registrar `zap.__version__`
   y la ruta del paquete en QC — el nombre `zap` es genérico y una colisión de
   paquete instalaría otra cosa sin error visible.
2. Confirmar compatibilidad de versiones con el entorno `MUSE` del repo
   (numpy/astropy fijados en `environment.yml`). Si ZAP exige versiones
   incompatibles: entorno virtual separado SOLO para esta etapa, documentado
   en QC — no actualizar el entorno principal.
3. Resolver el cubo de entrada desde config:
   - si `provenance="raw_reduction"`, verificar el QC de A1: `gates_passed`
     completo y V1–V6 en verde. Si A1 tiene `open_issues` que afecten al cielo
     (p. ej. V6 máscara de cielo), reportarlo antes de continuar;
   - si `provenance="adp"`, verificar que el cubo existe, tiene extensiones
     DATA y STAT, y registrar que A1 no aplica.
4. Registrar sha256 del cubo de entrada.

## 3. Fase 1 — ¿Hace falta ZAP? (gate de decisión, sin ejecutar ZAP)

Medir sobre el cubo de A1, en **aperturas vacías** (lejos de la primaria, del
compañero y del segundo objeto del campo; reutilizar la lógica de controles al
mismo radio de `musepipe/apertures.py` más aperturas de esquina):

1. `rms_sky`: rms robusto por canal en las aperturas vacías, agregado en dos
   conjuntos de ventanas: **ventanas de skylines** (5577.3, 6300.3, 6363.8 Å,
   banda O₂ 6864–6960 Å y bandas OH 7240–9300 Å) y **ventanas de continuo
   limpio** de referencia (p. ej. 5100–5500, 6600–6800 Å), siempre excluyendo
   5780–6050 Å.
2. Métrica de decisión: `R = rms(skylines) / rms(continuo)`.
   - `R ≤ 1.5` → **ZAP no necesario**: la etapa termina con `zap_applied: false`,
     QC y figuras de evidencia. Fin.
   - `R > 2.0` → ZAP necesario: continuar a Fase 2.
   - `1.5 < R ≤ 2.0` → zona gris: presentar los números y figuras al usuario y
     preguntar (checkpoint humano).
3. Medir también `sky_fraction`: fracción de spaxels utilizables como cielo
   tras la máscara de fuentes preliminar (§4.1). **Punto de máxima atención**:
   si `sky_fraction < 0.25` del campo, ZAP no tiene base estadística confiable
   en este cubo → reportar y preguntar (la alternativa es limpieza por
   spline/mediana en ventanas de skylines, fuera del alcance de A2).

Figuras de esta fase (van a `plots/` con o sin ZAP): rms por canal en aperturas
vacías con ventanas marcadas; espectro de una apertura vacía en 6250–6420 Å y
en 8600–9000 Å.

## 4. Fase 2 — Máscara de fuentes y ejecución (checkpoint humano antes de correr)

### 4.1 Máscara de fuentes (el insumo crítico de toda la etapa)

Construcción programática, en `sky_zap.py`:

1. Imagen de banda blanca del cubo → umbral en σ robusto sobre el fondo
   (default: 3σ) → dilatación morfológica (default: 2 px).
2. **Inclusión forzosa** (no negociable, independiente del umbral):
   - círculo sobre la primaria con radio suficiente para cubrir el halo AO
     visible (medirlo en la imagen blanca: radio donde el perfil cae al nivel
     del fondo + margen de 5 px);
   - círculo de radio ≥ 8 px sobre la posición del compañero (~1.75″ de la
     primaria; obtener la posición esperada del header/ADP, no a ojo);
   - ídem sobre el segundo objeto conocido del campo.
3. Guardar la máscara como FITS (`zap_source_mask.fits`) y calcular
   `sky_fraction` final.

**Checkpoint humano**: figura de imagen blanca + máscara superpuesta + posiciones
marcadas de las tres fuentes, junto con `sky_fraction` y la métrica `R`.
No ejecutar ZAP sin OK del usuario sobre esta figura.

### 4.2 Ejecución

1. `zap.process` con defaults + la máscara; entrada = cubo A1 completo (sin
   recortar: ZAP se beneficia de todo el campo).
2. Guardar también los productos de diagnóstico de ZAP (varcurve/eigenvalores
   por segmento y, si la versión lo permite, las eigenespectras) — se usan en
   V5.
3. Copiar la extensión STAT del cubo de entrada al cubo de salida sin
   modificar, y anotar en el header `HISTORY` la versión de ZAP y la máscara
   usada. ZAP no propaga varianza: el QC debe decir explícitamente que el
   error de la corrección de cielo queda como término sistemático a evaluar
   en A4/D2.
4. Log completo a `runs/ROXs12b_raw/logs/stage00s_<ts>.log`; excepción de ZAP
   → abortar y reportar, sin reintentos con otros parámetros.

## 5. Puntos donde mantener máxima atención

1. **Autoabsorción de señal**: el modo de fallo silencioso de ZAP. Se vigila
   con V2/V3/V5, y con esta regla: cualquier cambio sistemático del espectro
   del compañero fuera de ventanas de skylines es descalificante, aunque el
   cubo "se vea más limpio".
2. **La región Hα (6540–6590 Å)**: no hay skylines fuertes ahí, así que ZAP no
   debería cambiar nada. Si cambia algo por encima del ruido, es bandera roja
   inmediata (V3).
3. **Poco cielo disponible** (`sky_fraction`): campo pequeño + halo AO grande
   pueden dejar a ZAP sin spaxels de cielo. No estirar el umbral de la máscara
   para "ganar cielo": eso mete halo estelar en las eigencomponentes, que es
   exactamente el fallo del punto 1.
4. **Eigenespectras con forma estelar** (V5): si alguna componente principal
   se correlaciona con el espectro de la primaria, la máscara falló; se vuelve
   a §4.1, no se ajustan eigenvalores a mano.
5. **No usar el cubo ZAP para decidir sobre Hα**: A2 entrega un cubo más
   limpio para la cadena B–F; ninguna conclusión científica se toma aquí.

## 6. Esquema de `stage00s_qc.json`

```json
{
  "stage": "00s_sky_zap",
  "run_id": "ROXs12b_raw",
  "timestamp_utc": "...",
  "environment": {"zap_version": "...", "zap_path": "...", "separate_env": false},
  "input": {"cube": "...", "sha256": "...", "provenance": "raw_reduction|adp",
            "a1_gates_ok": true, "a1_not_applicable": false},
  "decision": {"R_skyline_over_continuum": 0.0, "threshold_used": "1.5/2.0",
                "sky_fraction": 0.0, "zap_applied": false,
                "user_checkpoint": "approved|not_needed"},
  "mask": {"file": "...", "threshold_sigma": 3, "dilation_px": 2,
            "forced_regions": ["primary", "companion", "field_source"]},
  "zap_params": {"defaults": true, "overrides": {}},
  "products": {"cube_zap": "...", "diagnostics": ["..."]},
  "verification": {"v1_rms_reduction_skylines": 0.0,
                    "v2_source_continuum_change_pct": 0.0,
                    "v3_halpha_window_change_sigma": 0.0,
                    "v4_sky_residual_symmetry": 0.0,
                    "v5_eigen_vs_star_corr_max": 0.0,
                    "v6_stat_untouched": true},
  "open_issues": []
}
```

## 7. Verificaciones finales

Implementadas en `musepipe/reduction/verify_zap.py`, cada una con figura y
número en QC. V1–V2 se ejecutan siempre; V3–V6 solo si `zap_applied: true`.

- **V1 — Reducción de residuales**: rms por canal en aperturas vacías,
  pre/post, en las ventanas de skylines de §3. Éxito: reducción ≥ 2× en la
  mediana de esas ventanas. (Si la etapa se salta: V1 documenta el `R` que lo
  justifica.)
- **V2 — Continuo de fuentes intacto**: espectros por apertura fija de
  primaria y compañero, pre/post; cambio relativo fuera de ventanas de
  skylines < 1% (primaria) y < 1σ del ruido por canal (compañero). Figura:
  razón post/pre vs λ con ventanas marcadas.
- **V3 — Hα intacta**: en 6540–6590 Å, diferencia post−pre en la apertura del
  compañero consistente con cero (|z| < 2 usando el ruido empírico local).
  Bandera roja automática si no.
- **V4 — Residuo de cielo sano**: distribución de residuales post-ZAP en
  aperturas vacías ~simétrica en torno a cero (|mediana| < 0.2×rms); la
  sobre-sustracción aparece como cola negativa.
- **V5 — Eigenespectras no estelares**: correlación máxima entre cada
  eigenespectro de ZAP y el espectro de la primaria (continuo normalizado)
  < 0.5. Si alguna supera: fallo de máscara, volver a §4.1 vía protocolo de
  parada.
- **V6 — STAT intacto**: byte-idéntico al de entrada (mismo sha256 de la
  extensión) y anotación en header presente.

## 8. Organización del código

```text
musepipe/reduction/sky_zap.py       # métricas de decisión, máscara, ejecución ZAP
musepipe/reduction/verify_zap.py    # V1–V6
scripts/sky_zap.sh                  # fachada CLI reproducible para A2
tests/test_sky_zap_decision.py      # métrica R y sky_fraction sobre mini-cubo sintético
tests/test_sky_zap_mask.py          # inclusión forzosa de fuentes en la máscara
tests/test_verify_zap.py            # V2/V3 detectan una absorción inyectada de -3% (test de potencia)
```

El test de `verify_zap` es en sí una medida anti-error: fabrica un caso donde
ZAP "roba" 3% de flujo y comprueba que V2 lo detecta. Si ese test no puede
fallar, la verificación no protege nada.

## 9. Protocolo de parada y reporte

Detenerse y preguntar (mismo formato que A1 §9) cuando: la decisión caiga en
zona gris (1.5 < R ≤ 2.0); `sky_fraction < 0.25`; cualquier V3–V6 en rojo;
ZAP lance excepción; o parezca necesario tocar parámetros de ZAP.

Reporte final: decisión y sus números; si se aplicó, tabla V1–V6 con figuras;
declaración explícita del término sistemático de cielo pendiente para A4/D2;
checklist de límites de §1; comando único de reproducción.
