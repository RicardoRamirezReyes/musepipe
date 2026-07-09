# Especificación D1 v2 · `X10_method_comparison` — rediseño post-G1

Fecha: 2026-07-09. Revisión formal de `docs/spec_D1_codex_method_comparison.md`
(v1, 2026-07-01), que queda como histórico. Etapa D1 del plan
`docs/plan_roxs12_reduccion_multimetodo.md`. Rama: `stage-d1-v2`.

## 0. Por qué existe una v2 (protocolo de revisión)

La v1 se ejecutó sobre `runs/ROXs12b_B_adp` y emitió `uninterpretable`
(blocker #9). El diagnóstico posterior demostró tres cosas:

1. **Causa raíz aguas arriba**: dentro de cada extractor los controles NO se
   procesaban como el objeto (sin resta de fondo, sin apcorr; la variante
   optimal-LS además arrastra el pedestal del residual 04b, ×20). El gate de
   la v1 hizo exactamente lo que debía: detectarlo y parar.
2. **Defecto de diseño de la v1**: gate de controles GLOBAL (un par sucio
   invalida todo el veredicto) y pares primarios que ni siquiera incluían
   `optimal_psfsub` — uno de los dos únicos métodos que G1 validó.
3. **Estadística optimista**: z por banda integrando canales como
   independientes (G1 midió corr-length 1.45 canales, n_eff/n = 0.69) y σ_diff
   estimada con solo 7 controles (~40% de incertidumbre relativa) tratada como
   conocida.

**Protocolo anti cherry-picking**: esta v2 se escribe y se commitea ANTES de
re-ejecutar D1 sobre datos reales. El QC de la etapa registra
`spec_version="D1_v2"` y el commit de esta spec. Cualquier cambio posterior de
umbrales, bandas o reglas = v3 con nueva justificación escrita; está prohibido
iterar umbrales mirando los estadísticos del run. Las decisiones de alcance
(reconciliación en C2/C3/C4, primarios = validados por G1, estadística con
covarianza + throughput) fueron fijadas por el usuario el 2026-07-09.

## 1. Límites duros (heredados y nuevos)

1. Bandas (§2) y umbrales (§4, §5) congelados en esta spec. Prohibido moverlos
   mirando los espectros o los estadísticos del run.
2. D1 solo lee productos; **no re-extrae, no re-calibra y NO re-normaliza**:
   la escala común es responsabilidad de C2/C3/C4 (principio §3.1). D1 la
   verifica (scale-check) y degrada pares, nunca la corrige.
   Única excepción declarada: la corrección de throughput §3.3, que es EN
   MEMORIA, trazada en QC, y nunca se escribe a productos.
3. El veredicto lo emite el código según §5; el reporte puede matizarlo, nunca
   sustituirlo.
4. Malla de λ: idéntica entre productos (mismo cubo madre); producto con otra
   malla → rechazo, no remuestreo.
5. La elección del método canónico sigue siendo del usuario (checkpoint
   final); D1 escribe `recommended_method`, nunca `canonical_method`.

## 2. Bandas de comparación (SIN CAMBIO, re-congeladas)

Idénticas a v1 para preservar comparabilidad:

| Banda | Rango (Å) | Tipo |
|---|---|---|
| B1 | 4900–5400 | continuo azul |
| B2 | 5450–5750 | continuo |
| B3 | 6100–6400 | continuo pre-Hα |
| B4 | 6600–6800 | continuo post-Hα |
| B5 | 7600–8000 | continuo rojo |
| B6 | 8600–9100 | continuo rojo lejano |
| LHα | 6553–6573 | línea Hα |
| LHβ | 4851–4871 | línea Hβ |
| LOI | 8436–8456 | línea O I 8446 |

Canales con flags de ventana mala/skyline excluidos dentro de cada banda,
`n_chan_used` registrado.

## 3. Convención de escala, insumos G1 y estadística

### 3.1 Convención de escala común (principio "control = objeto")

Cada extractor (C2/C3/C4) procesa las posiciones de control EXACTAMENTE como
el objeto: mismo modelo de fondo, misma corrección de apertura, y persiste los
controles en la misma escala física que la columna `flux` del producto.
Convención común: flujo total de fuente puntual dentro de `NORMRAD`,
referenciado a fondo local.

- Headers nuevos obligatorios en productos re-generados: `BKGMODE` (modo de
  fondo del método, p.ej. `annulus`, `local_surface_04b`, `psffit_plane`) y
  `SCALEREF = "normrad_total_flux"`. Productos antiguos sin estos headers:
  warning + issue en QC, no excepción (compatibilidad).
- **Scale-check empírico por par** (nuevo, gate estructural): para cada par de
  métodos, sobre los diffs de control integrados en banda ancha,
  `mu = media(diff_ctrl)`, `s = std(diff_ctrl, ddof=1)`, estadístico
  `|mu| / (s/√n)`, y `level_ratio` = razón de niveles medianos absolutos de
  los controles de ambos métodos. **Falla si el offset es significativo
  (estadístico ≥ 5.0) Y ADEMÁS `level_ratio > 3.0`** (ambos congelados).
  *Revisión v2.1 (2026-07-09, ANTES de emitir ningún veredicto sobre el
  objeto)*: la regla original ("falla si ≥ 5.0" a secas) contradecía §3.4 —
  degradaba cualquier sesgo aditivo común y centrable (p.ej. el nivel de halo
  del psffit en el anillo de controles, que §3.4 absorbe por diseño con el
  centrado mu). El diagnóstico de CONTROLES del re-run reconciliado mostró
  offsets significativos pero con `level_ratio` 1.07–1.60 en todos los pares,
  mientras el caso de convención rota (×20) tiene `level_ratio` ~20: el
  discriminante estructural es el desajuste de NIVELES, no la significancia
  del offset. Un offset significativo con niveles consistentes se reporta en
  QC (mu, s, stat) y lo maneja el centrado de §3.4.
- Un scale-check fallado **degrada el PAR** (no el veredicto global). Solo si
  TODOS los pares primarios fallan Y el offset relativo supera ×10 se lanza
  excepción (señal de bug de etapa de extracción, mensaje accionable).
- Si `optimal_ls` sigue fallando el scale-check tras el fix de pedestal en C3,
  se marca formalmente **no-reconciliable** (`scale_check.ok=false` en QC) y
  permanece como diagnóstico secundario, reportado siempre, sin poder de veto.

### 3.2 Insumos G1 (jerarquía de métodos)

D1 v2 lee el QC de G1 (`stages/stage_g1_qc.json`) y el bias budget
(`tables/g1_bias_budget.csv`):

- `method_verdicts`: métodos con verdict ∈ {`validated`, `validated_with_bias`}
  son **primarios**; `rejected`/ausentes son **secundarios** (diagnóstico,
  sin veto). Con el run actual: primarios = {psffit, optimal_psfsub} →
  pares primarios = {psffit vs optimal_psfsub}. Limitación aceptada y
  declarada: con UN solo par primario el veredicto descansa en ese par; la
  matriz completa (incluidos secundarios) se reporta siempre.
- Fallback congelado si G1 no está disponible: pares primarios =
  {(psffit, optimal_psfsub)} + open_issue (NO los pares de la v1).
- `throughput_by_method`: T = 1 + value_frac del término `throughput_loss`
  (fuente primaria `g1_bias_budget.csv`; fallback
  `stage_h04_qc.json per_method_at_snr5`; fallback T=1.0 + issue). La fuente
  usada se registra por método (`source`).
- `covariance`: corr-length de canales y n_eff/n del QC G1 (fallback 1.0/1.0
  + issue).

### 3.3 Corrección de throughput (en memoria, sin doble conteo)

Antes de comparar, flux, err y controles de cada método primario con T medido
y acotado se dividen por T **en memoria**. Los productos en disco NO cambian.
Métodos rechazados o con T no acotado (aperture, optimal_ls: nan en el bias
budget) no se corrigen (`throughput_source="not_applied_rejected_method"`).

- Propagación de err_T: término `|diff_obj|·(err_T/T)` en cuadratura en el
  denominador del estadístico del par. (Con la fuente actual err_frac=0.0 →
  término nulo; el mecanismo queda implementado y documentado.)
- **V5 (no doble corrección)**: E3 (`stage_h03_limits.py`) y G2
  (`stage_g2_measure_lines._resolve_throughput`) aplican su propia corrección
  sobre los productos crudos de disco. Como D1 nunca escribe productos
  corregidos, no hay doble conteo; el QC lo declara explícitamente.
- Dominio de validez: T se midió con inyección en la posición real (E4) y se
  aplica como escalar acromático; su uso en bandas azules es extrapolación —
  limitación registrada como issue conocido, no bloqueante (en el par
  psffit vs psfsub los T son similares y el error se cancela parcialmente en
  el diff).

### 3.4 Estadístico primario: t centrada con controles integrados por banda

Sea n el número de controles (mismas posiciones para todos los métodos, mismo
radio). Para cada (par, banda):

1. `d_k` = diff integrado en la banda del control k (método i − método j,
   ambos ya corregidos por throughput si aplica), k = 1..n.
2. `mu = media(d_k)`, `s = std(d_k, ddof=1)`.
3. `d_obj` = diff integrado del objeto (corregido).
4. `t = (d_obj − mu) / (s·√(1 + 1/n))`, `df = n − 1`, p bicola de la t de
   Student.

Propiedades por las que sustituye al z de la v1:

- Los diffs **integrados por control** incorporan automáticamente la
  covarianza canal-canal (no hace falta n_eff en el estadístico primario).
- El **centrado con mu** absorbe sesgos aditivos comunes residuales (p.ej. el
  nivel de halo en los controles de psffit).
- La t con df = n−1 reconoce que s se estimó con pocos controles: con df=6 el
  t crítico al 2σ-equivalente sube a ~2.52 (honestidad automática con n
  pequeño).

Verificación cruzada (no veredicto): `z_neff` = z de la v1 con
`sigma_int · √corr_length` (≈ ×1.204 con corr-length 1.45). Se reportan ambos.
La σ_diff empírica per-canal de la v1 (`empirical_sigma_diff`, MAD entre
controles) se mantiene para el CSV per-canal y para V1.

Registrar por fila: `n_controls`, `df`, `t`, `p`, `mu_ctrl`, `s_ctrl`.

### 3.5 Sección "N pequeño"

Con 7 controles, s tiene ~40% de incertidumbre relativa (χ², 6 df) — razón
formal del cambio z→t. `n_controls` y `df` se registran SIEMPRE en CSV y QC.
Objetivo operativo: ≥31 controles (la config del run ya pide 38; el re-run de
x01–x03 debe producir ~33 tras exclusiones). Con df≈32 los t críticos
(~2.11/3.3) convergen a los z de la v1.

## 4. Comparaciones

### 4.1 Sobre el objeto

Por canal: diferencia y razón por par con banda ±1σ_diff empírica (como v1).
Por banda (§2): tabla maestra con flux por método (crudo y corregido), t/p/df
(§3.4), z_neff, y correlación de divergencias con canales sucios B2, flags de
skyline y gap interpolado de C1 (como v1).

### 4.2 Sobre controles (gate POR PAR)

Para cada par y banda, los t individuales de control
`t_k = (d_k − mu)/s` (leave-one-out no requerido; documentado como t directo
centrado). Un control es "malo" si su p (df=n−1) < 0.0455. Un par está
"sucio" si sus controles no se distribuyen como se espera O si su scale-check
§3.1 falla. Regla congelada para "no se distribuyen como se espera": test
binomial unilateral de exceso — con N filas de control del par y K malas,
el par es sucio si P(≥K | N, 0.0455) < **alpha = 0.01**. Caso degenerado
(solo sintético): si s = 0 con todos los d_k idénticos, los controles son
perfectamente consistentes (t=0, p=1), no sucios.

- Par primario sucio → se excluye del veredicto y se registra en
  `pairs_degraded` con su causa.
- Pares secundarios NUNCA degradan el veredicto: van a `secondary_summary`.

## 5. Reglas de veredicto (congeladas)

Umbrales en espacio p (equivalentes a los 2σ/3σ de la v1, calibrados a df
finito): **p_div = 0.0455**, **p_strong = 0.0027**.

Sobre los pares PRIMARIOS con controles limpios:

- **`consistent`**: p ≥ p_div en todas las bandas (continuo y línea) de todos
  los pares primarios limpios.
- **`divergent_continuum`**: p < p_div en ≥ 2 bandas de continuo, o
  p < p_strong en ≥ 1 banda de continuo, en algún par primario limpio.
  → Acción del plan: iterar C1 (refinar PSF), no saltar a PCA.
- **`divergent_lines`**: continuo consistente pero p < p_div en alguna banda
  de línea. → Acción: tratar en E1/E2.
- **`uninterpretable`**: TODOS los pares primarios sucios (controles o
  scale-check). → Acción: auditar extracción; sin veredicto sobre el objeto.
- **`marginal`** (anotación, no veredicto): p_strong ≤ p < p_div en UNA sola
  banda de continuo de un par limpio.

## 6. Salidas

```text
tables/method_comparison.csv          # todas las columnas v1 + role, flux_*_corr,
                                      #   mu_ctrl, s_ctrl, t_stat, p_value, df,
                                      #   throughput_{i,j}(+err,source), scale_ok
tables/method_comparison_controls.csv # ídem por control
stages/stage_x10_qc.json              # + spec_version, spec_commit, g1_inputs,
                                      #   pairs_primary/secondary, verdict_by_pair,
                                      #   pairs_degraded, throughput_correction,
                                      #   scale_check, statistics{kind:"t",df,n_controls},
                                      #   recommended_method (nunca canonical_method)
plots/stage_x10_overview.png          # heatmap sobre pares primarios dinámicos +
                                      #   panel de secundarios + banda mu±s en razones
```

Compatibilidad: F1 (`report/table_methods.csv`) lee `method_comparison.csv`
(se corrige el filename obsoleto en `musepipe/report.py`); todas las columnas
v1 se conservan.

## 7. Verificaciones

- **V1 — σ_diff empírica sana** (v1): empírica ≤ ingenua √(σ₁²+σ₂²) por par.
- **V2 — Controles limpios**: distribución de t de controles ~ t(df) por par
  primario (QQ-plot); gate del veredicto.
- **V3 — Panel del objeto** (v1): espectros, razones, heatmap t por
  banda×par.
- **V4 — Atribución** (v1): divergencias × canales sucios B2/flags/gap C1.
- **V5 — No doble corrección**: el QC declara la corrección de throughput
  como interna; productos en disco byte-idénticos antes/después de D1; la
  ruta de G2/E3 no consume valores corregidos por D1 (test dedicado).
- **V6 — Scale-check**: por par, mu, s, estadístico y ok; el par ×20
  sintético se detecta y solo degrada sus pares.
- **V7 — Calibración de la t**: fracción de controles con p < 0.0455 ≈ 5%
  por par primario (con n pequeño, intervalo binomial amplio — reportar el
  intervalo, no solo la fracción).

## 8. Tests

Adaptados: `tests/test_compare_verdicts.py` (primarios nuevos + g1_inputs
sintético, 3 caminos), `test_compare_controls_gate.py` (único primario sucio →
uninterpretable; secundario sucio NO degrada; de 2 primarios uno sucio →
veredicto del limpio + degradado), `test_compare_sigma_diff.py` (+ t
calibrada), `test_compare_rejects.py` (+ sin BKGMODE → warning),
`test_compare_stage.py` (columnas/QC v2, corre sin G1 con fallbacks).

Nuevos: `test_compare_scale_reconciliation.py` (offset ×20 → falla solo pares
con optimal_ls), `test_compare_throughput_roundtrip.py` (F/T·T=F; disco
intacto; guardia anti doble corrección), `test_compare_neff.py` (AR(1): z
ingenuo descalibrado, t/z_neff calibrados), `test_aperture_controls_like_object.py`
(fondo constante → controles calibrados ~0), test del filename de report.

## 9. Protocolo de parada y reporte

Preguntar cuando: excepción de escala §3.1 (todos los primarios ×10);
V2/V5/V6 fallen; el veredicto sea `uninterpretable`. Si el sesgo de halo de
psffit en controles no es centrable (depende de posición/ángulo), el par
primario queda legítimamente sucio → parar y reportar, no maquillar.
Checkpoint final SIEMPRE: veredicto, matriz t/p completa (primarios y
secundarios), recomendación razonada (default esperado: `psffit` si
`consistent`) y esperar la elección del usuario para D2. Todo resultado sobre
`ROXs12b_B_adp` sigue siendo provisional (A-block abierto).
