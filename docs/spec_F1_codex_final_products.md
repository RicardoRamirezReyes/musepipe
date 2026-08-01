# Especificación F1 · `R01_final_products` — brief de ejecución para Codex (ChatGPT 5.5)

Fecha: 2026-07-01. Etapa F1 (ESENCIAL, cierre) del plan
`docs/2026-07-10_plan_roxs12_reduccion_multimetodo.md`. Prerrequisitos: D1, D2, E1, E2 y
(E3 o flujos medidos) terminadas; E4 terminada.

---

## 0. Rol, objetivo y definición de terminado

**Rol del agente**: consolidar el run en un paquete final reproducible:
resumen de QC global, set fijo de figuras "de papel", tablas maestras y un
reporte generado — todo regenerable con un comando y determinista. F1 no
calcula ciencia nueva: **si F1 necesita computar algo que no está en un QC o
producto aguas arriba, eso es un hueco de la etapa correspondiente** y se
reporta como issue, no se parcha aquí.

**Definición de terminado**:

1. `runs/<RUN_ID>/report/` con: `run_summary.json`, las 7 figuras de §3, las
   tablas de §4 y `report.md` generado.
2. Checklist automática de §2 en verde (o issues explícitos).
3. Determinismo demostrado: dos ejecuciones seguidas → mismos archivos
   (mismo hash, salvo timestamps aislados en un único campo).
4. `musepipe/report.py` + `scripts/build_report.py` + tests.

---

## 1. Límites duros

1. Solo lectura de productos y QCs; F1 no re-ejecuta etapas.
2. El set de figuras y tablas es el de §3–§4: fijo. Figuras extra van a un
   anexo `report/extra/` claramente separado, nunca mezcladas con el set
   estándar.
3. Sin embellecimiento de datos (sin suavizados nuevos en figuras; los datos
   se muestran como los produce D2/E1, con sus flags visibles).
4. `report.md` se GENERA desde `run_summary.json` + plantilla; prohibido
   editarlo a mano (el test de determinismo lo detectaría).
5. Git: rama `stage-f1-report`.

---

## 2. Checklist automática (gate del paquete)

`run_summary.json` recolecta el QC de cada etapa y computa:

1. **Semáforo por etapa**: toda etapa ESENCIAL ejecutada y sin verificaciones
   en rojo; etapas condicionales (A2/A3) con decisión documentada; opcionales
   marcadas `not_run`.
2. **Cadena de hashes**: el `sha256` de entrada declarado por cada etapa
   coincide con el de salida de su predecesora (detecta mezcla de runs — el
   riesgo #4 del plan). Cualquier ruptura de cadena = rojo.
3. **Consistencia de convenciones**: `FORMATV`, `NORMRAD`, `WFRAME` idénticos
   en todos los productos espectrales.
4. **`open_issues` agregados** de todas las etapas, priorizados
   (bloqueante/mayor/menor) según afecten al resultado de Hα, al espectro o
   solo a documentación.

## 3. Las 7 figuras (formato de publicación, generadas desde productos)

1. Campo: imagen blanca con las tres fuentes, apertura y controles (B3/E1).
2. PSF(λ): parámetros suavizados + métrica del anillo del compañero (C1).
3. Espectros por método superpuestos con bandas de error (D1).
4. Espectro final calibrado con presupuesto de error visible (D2).
5. Zoom Hα por método con controles y plantillas (E1).
6. Throughput y completitud por método (E4).
7. Límite superior vs separación con literatura (E3) — o, si hubo detección,
   la señal con sus tests de artefactos (E2).

## 4. Tablas maestras

- `report/table_lines.csv`: por línea (Hα, Hβ, O I 8446): flujo o límite,
  método, FAP, throughput, conversiones físicas (de E1/E3).
- `report/table_methods.csv`: resumen D1 (Z por banda) + S/N por método.
- `report/table_qc.csv`: la checklist §2 en forma tabular.

## 5. `report.md` (estructura fija)

Resumen ejecutivo (5 líneas: qué se midió, veredicto Hα, límite o flujo,
método canónico, issues abiertos) → tabla QC → figuras con pies → tablas →
diferencias vs cadena histórica (el resultado local-surface previo como
contexto) → apéndice de reproducción (comandos por etapa, versiones, hashes).

## 6. Verificaciones y tests

- **V1 — Determinismo**: doble ejecución → diff vacío (salvo el campo
  timestamp único).
- **V2 — Cadena de hashes**: test con un QC sintético manipulado → la
  checklist lo detecta.
- **V3 — Completitud**: cada figura/tabla declarada existe y no está vacía;
  cada número del report.md es trazable a un QC (el generador solo puede
  insertar valores leídos, no literales).

```text
tests/test_report_determinism.py
tests/test_report_hash_chain.py
tests/test_report_traceability.py   # la plantilla no contiene números hardcodeados
```

## 7. Protocolo de parada y reporte

Preguntar cuando: la checklist §2 tenga rojos (¿publicar paquete parcial con
issues declarados o esperar?); haya productos con convenciones inconsistentes
(§2.3 — bug aguas arriba). Reporte final al usuario: el propio `report.md`
más la lista de issues priorizada y la recomendación de si el paquete está
listo para uso científico externo.
