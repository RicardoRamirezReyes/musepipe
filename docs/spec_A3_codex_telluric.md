# Especificación A3 · `00t_telluric` — brief de ejecución para Codex (ChatGPT 5.5)

Fecha: 2026-07-01. Etapa A3 (OPCIONAL) del plan
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
   - `A1`: verificar QC A1 en verde;
   - `ADP`: verificar DATA+STAT y registrar que A1/A2 no aplican para esta
     entrada.
   En todos los casos registrar sha256 del cubo.
3. **Punto de atención — GDAS/atmósfera**: molecfit puede requerir perfiles
   atmosféricos (GDAS) descargables; si la máquina no tiene red o el perfil de
   la fecha no está disponible, usar el perfil estándar equatorial documentado
   y anotarlo en QC como fuente de error.

## 3. Fase 1 — ¿Hace falta corrección? (gate de decisión)

1. Extraer el espectro de la primaria (apertura grande, alta S/N) del cubo de
   entrada.
2. Medir profundidad telúrica: flujo medio dentro vs fuera de las bandas O₂
   (6864–6960 Å) y H₂O (7160–7340, 8130–8350 Å), sobre continuo local
   interpolado. Registrar `depth_pct` por banda.
3. Decisión:
   - Si el objetivo declarado del run no usa el continuo >6800 Å del
     compañero, o `depth_pct < 3%` en todas las bandas → `telluric_applied:
     false`, fin con QC y figuras.
   - Si `depth_pct ≥ 3%` en alguna banda de interés → continuar a Fase 2.
   - En duda sobre el objetivo científico → preguntar al usuario, no asumir.

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
  "input": {"cube": "...", "sha256": "...", "upstream": "A2|A1|ADP"},
  "decision": {"depth_pct_by_band": {}, "telluric_applied": false,
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
