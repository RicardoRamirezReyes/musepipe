# Especificación D1 · `X10_method_comparison` — brief de ejecución para Codex (ChatGPT 5.5)

> **DEPRECADA (2026-07-09)**: sustituida por
> `docs/spec_D1_v2_codex_method_comparison.md` tras el veredicto
> `uninterpretable` del run real (blocker #9: escala de flujo inter-método) y
> los veredictos por método de G1. Se conserva como histórico; los umbrales y
> reglas vigentes son los de la v2.

Fecha: 2026-07-01. Etapa D1 (ESENCIAL, nueva) del plan
`docs/plan_roxs12_reduccion_multimetodo.md`. Prerrequisitos: C2, C3, C4
terminadas (productos `SpectrumProduct` válidos). Opcionalmente consume
también la rama local-surface histórica vía C2 y las variantes de C3.

D1 es el **árbitro** del plan: implementa el árbol de decisión del documento
de investigación (¿concuerdan PSF-fitting y apertura? → no hace falta PCA;
¿divergen? → refinar PSF antes de escalar) con reglas numéricas fijadas AQUÍ,
antes de mirar los datos.

---

## 0. Rol, objetivo y definición de terminado

**Rol del agente**: comparación estadística entre métodos de extracción, sobre
el objeto Y sobre los controles, con un veredicto de clasificación automática.
D1 no elige "el método verdadero": produce el veredicto y la evidencia; la
elección del método canónico es del usuario (checkpoint final).

**Definición de terminado**:

1. `musepipe/stages/stage_x10_compare.py` opera EXCLUSIVAMENTE sobre archivos
   `SpectrumProduct` (validados con `product.validate()`) — cero acceso a
   cubos. Eso lo hace genérico para cualquier target futuro.
2. `tables/method_comparison.csv` + `stage_x10_qc.json` con veredicto según §4.
3. La comparación sobre CONTROLES ejecutada (§3.4) — es la mitad del valor de
   la etapa.
4. Verificaciones §6 en verde; figura resumen generada.
5. Checkpoint final: veredicto + recomendación presentados al usuario para que
   elija el método canónico que D2 consumirá.

---

## 1. Límites duros

1. **Bandas y umbrales congelados en esta spec** (§2, §4). Prohibido añadir,
   mover o quitar bandas mirando los espectros — eso es cherry-picking y
   invalida el veredicto. Cambios → preguntar y documentar como revisión de
   la spec.
2. Solo lectura de productos; D1 no re-extrae ni re-calibra nada.
3. El veredicto lo emite el código según §4, no el agente. El reporte puede
   matizarlo, nunca sustituirlo.
4. Malla de λ: los productos DEBEN venir en la malla nativa común (mismo cubo
   madre). Si algún producto trae otra malla, es un error de su etapa —
   rechazar, no remuestrear (el remuestreo mete covarianza que rompe los χ²).
5. Git: rama `stage-d1-compare`.

---

## 2. Bandas de comparación (congeladas)

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

(Ventanas de línea = ±10 Å ≈ ±450 km/s en Hα; los canales con flags de
ventana mala/skyline se excluyen dentro de cada banda, con `n_chan_used`
registrado.)

## 3. Operaciones

### 3.1 Ingesta y validación

Cargar todos los productos disponibles (mínimo: `aperture box3`, `optimal ls`,
`optimal psfsub`, `psffit`); `validate()` sobre cada uno; verificar mallas
idénticas, mismo `WFRAME`, mismo `INCUBESH` (¡mismo cubo madre!) y la MISMA
convención de flujo total (`apcorr` aplicado / `NORMRAD` común). Cualquier
inconsistencia de convención → parar (es un bug de C2/C3/C4, no algo que D1
deba absorber).

### 3.2 El error de una DIFERENCIA entre métodos — punto de máxima atención

Los métodos comparten los mismos fotones: sus errores están fuertemente
correlacionados, y `√(σ₁²+σ₂²)` SOBREstima el error de la diferencia
(haría que todo pareciera consistente). La escala correcta se mide
empíricamente: para cada par de métodos, calcular la diferencia por canal
sobre los N controles al mismo radio → `σ_diff_empirica(λ)` (dispersión robusta
entre controles, suavizada). TODOS los χ² y z-scores de D1 usan esa σ, no la
propagada. Este es el corazón estadístico de la etapa; test dedicado en §7.

### 3.3 Comparaciones sobre el objeto

- Por canal: diferencia y razón entre cada par, con banda `±1σ_diff_empirica`.
- Por banda (§2): flujo integrado por método (con su error), z-score de cada
  par usando σ_diff empírica integrada, y tabla maestra.
- Correlación de discrepancias con: canales sucios de B2, flags de skyline,
  y el gap interpolado de C1 (si las divergencias viven ahí, la causa es
  conocida y se anota).

### 3.4 Comparaciones sobre controles

Para cada control (posiciones al mismo radio, mismas para todos los métodos):
mismas bandas, mismos pares. Criterio: sobre los controles, TODOS los pares
deben ser consistentes con cero (los métodos deben coincidir en que el ruido
es ruido). Si un par diverge en controles, la divergencia en el objeto no es
interpretable → el veredicto se degrada automáticamente (§4).

## 4. Reglas de veredicto (congeladas)

Sea Z(par, banda) el z-score con σ_diff empírica. Pares primarios:
{psffit vs aperture, psffit vs optimal-ls, optimal-ls vs aperture}.

- **`consistent`**: |Z| < 2 en TODAS las bandas de continuo y de línea, para
  los tres pares primarios, y controles §3.4 limpios.
  → Acción del plan: rama conservadora suficiente; NO se activa C5/C6.
- **`divergent_continuum`**: |Z| ≥ 2 en ≥ 2 bandas de continuo en algún par
  primario (controles limpios).
  → Acción: iterar C1 (refinar PSF) — NO saltar a PCA. Adjuntar el patrón
  espectral de la divergencia como insumo para C1.
- **`divergent_lines`**: continuo consistente pero |Z| ≥ 2 en alguna banda de
  línea. → Acción: tratar en E1/E2 (¿la divergencia es artefacto o señal
  tratada distinto por cada método?); considerar C5 SOLO si E2 lo respalda.
- **`uninterpretable`**: controles §3.4 no limpios. → Acción: auditar las
  etapas de extracción; ningún veredicto sobre el objeto.

Fronteras: 2 ≤ |Z| < 3 en UNA sola banda de continuo no dispara divergencia
(se anota como `marginal`); |Z| ≥ 3 en cualquier banda siempre cuenta.

## 5. Salidas

```text
tables/method_comparison.csv        # banda × par × {flujo_i, err_i, Z, n_chan_used}
tables/method_comparison_controls.csv
stage_x10_qc.json                   # veredicto, Z-matrix, sigma_diff mediana por par, correlaciones con B2/flags
plots/stage_x10_overview.png        # panel: espectros superpuestos + razones + Z por banda
```

## 6. Verificaciones

- **V1 — σ_diff empírica sana**: para cada par, figura σ_diff(λ) vs la
  ingenua `√(σ₁²+σ₂²)`; la empírica debe ser ≤ la ingenua (correlación
  positiva de errores). Si sale mayor, hay un problema en los controles.
- **V2 — Controles limpios**: distribución de Z de controles ~N(0,1) por par
  (QQ-plot); es el gate del veredicto.
- **V3 — Panel del objeto**: espectros con bandas de error, razones con
  ±1σ_diff, Z por banda como mapa de calor método×banda.
- **V4 — Atribución**: divergencias (si las hay) cruzadas con canales sucios
  B2/flags/gap C1; figura con las tres capas.

## 7. Tests

```text
tests/test_compare_verdicts.py     # productos sintéticos: idénticos+ruido → consistent; offset 5% en continuo → divergent_continuum; línea solo en un método → divergent_lines (los tres caminos)
tests/test_compare_sigma_diff.py   # dos "métodos" = mismo ruido base + ruido propio conocido → sigma_diff empírica recupera el valor analítico, no sqrt(s1²+s2²)
tests/test_compare_controls_gate.py# controles con un par sesgado → veredicto uninterpretable (el gate funciona)
tests/test_compare_rejects.py      # producto con malla distinta o NORMRAD distinto → excepción, no remuestreo
```

## 8. Protocolo de parada y reporte

Preguntar cuando: §3.1 detecte inconsistencia de convención entre productos;
V1/V2 fallen; el veredicto sea `uninterpretable`. Checkpoint final SIEMPRE:
presentar veredicto, matriz Z, recomendación razonada de método canónico
(default esperado: `psffit` si `consistent`), y esperar la elección del
usuario. Reporte: veredicto + acción del árbol de decisión que corresponde,
tabla maestra, verificaciones, checklist de límites, comando de reproducción.
