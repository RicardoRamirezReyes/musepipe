# Especificación B2 · `02_xcorr_stripes` — brief de ejecución para Codex (ChatGPT 5.5)

Fecha: 2026-07-01. Etapa B2 del plan `docs/plan_roxs12_reduccion_multimetodo.md`.
Alcance deliberadamente pequeño: la maquinaria de stripes YA está centralizada
en `musepipe/stripes.py` (625 líneas, con tests) y diagnosticada en
02b/02c. B2 NO la reimplementa. Son tres añadidos acotados.

---

## 0. Rol, objetivo y definición de terminado

**Rol del agente**: completar la etapa 02 existente con lo mínimo que las
etapas D1 (comparación de métodos) y E2 (tests de artefactos) necesitan
consumir, sin alterar la corrección misma.

**Los tres añadidos**:

1. **Métrica escalar de stripes en QC**: amplitud del patrón de stripes por
   cubo y por canal, antes y después de la corrección, agregada también como
   escalar global. E2 la usará para correlacionar cualquier "señal" de Hα con
   canales de stripes residuales; D1 para comparar métodos en canales
   limpios/sucios.
2. **STAT a través de la corrección**: la corrección aplica shifts espectrales
   por stripe; verificar si el código actual los aplica también a STAT (si B1
   la propagó). Si no lo hace: aplicar a STAT el MISMO shift que a DATA
   (shift entero = reindexado, sin cambio de varianza; shift fraccional =
   kernel², como B1). Si el shift es fraccional, registrar el kernel.
3. **Migración fina del driver**: si la ejecución de la etapa aún vive en el
   notebook `02_xcorr_stripes.ipynb` (y no en `musepipe/stages/`), crear
   `musepipe/stages/stage02_xcorr.py` con `run_stage02()` por equivalencia,
   mismo protocolo que B1a (gate de equivalencia incluido). Si ya existe un
   driver equivalente, este punto se reduce a verificar el contrato.

**Definición de terminado**: los tres añadidos implementados con sus gates;
`stage02_qc.json` extendido (§3); verificaciones §4 en verde; suite del repo
intacta; notebooks 02/02b/02c siguen funcionando como capa de inspección.

---

## 1. Límites duros

1. **No cambiar la corrección**: mismos estimadores, mismos parámetros, misma
   salida numérica de DATA. La regla de equivalencia de B1 §4 aplica igual.
   Los añadidos 1 y 2 no tocan la ruta de DATA.
2. Runs históricos intocables; validación sobre copia aislada.
3. No cambiar firmas ni comportamiento de funciones existentes en
   `musepipe/stripes.py`; solo funciones nuevas.
4. Los QC históricos existentes deben seguir siendo legibles: las claves
   nuevas se AÑADEN, ninguna se renombra.
5. Git: rama `stage-b2-stripes`; añadido 3 (migración) en commits separados de
   los añadidos 1–2.

---

## 2. Especificación de la métrica de stripes (añadido 1)

- Definición: para cada canal, tras colapsar el cubo en la dirección de las
  stripes (según `xcorr_stripe_orientation`/geometría de
  `musepipe/stripes.py`), amplitud = desviación robusta (`robust_sigma` de
  `musepipe/stats.py`, no reimplementar) del perfil transversal normalizado.
- Se calcula pre y post corrección; producto:
  `tables/stage02_stripe_metric.csv` (canal, amplitud_pre, amplitud_post) y
  escalares en QC (`stripe_amp_pre/post_median`, `p95`, y lista de los canales
  con `amplitud_post > 3× mediana` — los "canales sucios" que E2 consultará).
- **Punto de atención**: excluir de la métrica las ventanas ya conocidas
  (5780–6050 Å y bordes), y calcularla sobre regiones sin fuentes (reutilizar
  la lógica de máscara existente); si la primaria entra en el perfil, la
  métrica mide halo, no stripes.

## 3. Extensión de `stage02_qc.json` (claves nuevas)

```json
{
  "stripe_metric": {"amp_pre_median": 0.0, "amp_post_median": 0.0,
                     "amp_post_p95": 0.0, "reduction_factor": 0.0,
                     "dirty_channels": [], "table": "tables/stage02_stripe_metric.csv",
                     "mask_excluded_windows_A": [[5780, 6050]]},
  "stat": {"present": true, "shift_applied": true, "interp_kernel": "integer|..."},
  "equivalence": {"verdict": "identical|allclose", "max_abs_diff": 0.0}
}
```

## 4. Verificaciones

- **V1 — Equivalencia de DATA**: producto post-corrección idéntico al baseline
  (tolerancia B1 §4) en la copia de validación.
- **V2 — Métrica sana**: `reduction_factor ≥ 1` (la corrección no empeora);
  figura amplitud vs canal pre/post con ventanas excluidas marcadas y canales
  sucios señalados.
- **V3 — STAT coherente**: los canales de STAT siguen a los de DATA tras los
  shifts (test: inyectar un canal de varianza marcada en un mini-cubo
  sintético, aplicar corrección, verificar que se movió con su canal de DATA).
- **V4 — Regresión**: suite del repo + tests existentes de stripes en verde.

## 5. Tests nuevos

```text
tests/test_stage02_stripe_metric.py  # mini-cubo con stripes sintéticas de amplitud conocida → métrica la recupera; y da ~0 sin stripes
tests/test_stage02_stat_shift.py     # el test de V3
```

## 6. Protocolo de parada

Preguntar cuando: exista discrepancia entre lo que la config dice
(`xcorr_shift_estimator`, orientación) y lo que el código hace; la métrica no
sea definible limpiamente para la geometría de stripes proyectada (ángulos no
axiales); o la equivalencia falle. No adivinar semántica de código heredado:
citar el fragmento y preguntar.

## 7. Reporte final

Métrica global pre/post y nº de canales sucios; verificaciones V1–V4;
si se hizo la migración del driver, tabla de equivalencia; checklist de
límites; comandos de reproducción.
