# Especificacion D1 v4 - comparacion por observable fisico

Fecha: 2026-07-29. Revision formal de
`docs/spec_D1_v3_codex_method_comparison.md`, que conserva la interpretacion de
los QC historicos. Esta revision se congela antes de volver a ejecutar D1 sobre
`ROXs12b_realigned`. El nuevo QC declara `spec_version="D1_v4"`.

## 0. Motivo y alcance

La auditoria del cubo multiepoca encontro que D1 v3 mezclaba tres observables
distintos en un unico estadistico:

1. Aplicaba el throughput escalar medido con inyecciones Halpha a todas las
   bandas de continuo. Ese uso es una extrapolacion cromatica no medida y puede
   crear diferencias entre continuos crudos que ya concordaban.
2. Integraba las bandas de linea sin restar continuo local. La fila `LOI` podia
   estar dominada por el nivel de continuo, no por O I 8446.
3. Dejaba que SGF y LPM, que eliminan o absorben componentes de continuo por
   construccion, gobernaran `divergent_continuum` contra extractores de flujo
   total.

D1 v4 conserva los seis productos, 15 pares, bandas, controles, umbrales,
scale-check, jerarquia G1 y arbol de recomendacion de v3. Cambia solamente la
cantidad fisica comparada en cada tipo de banda y que metodos tienen poder de
veto sobre el continuo.

## 1. Limites duros

1. `METHOD_ORDER` permanece
   `(aperture, optimal_ls, optimal_psfsub, psffit, sgf, lpm)`.
2. Las bandas B1-B6, LHalpha, LHbeta y LOI permanecen identicas a D1 v2/v3.
3. Los umbrales permanecen `p_divergent=0.0455`, `p_strong=0.0027`,
   `control_gate_alpha=0.01` y `scale_gate_sigma=5.0`.
4. Los controles se procesan como el objeto y el estadistico primario sigue
   siendo la t de Student centrada en controles, con `df=n_controls-1`.
5. D1 no re-extrae, no calibra productos, no escribe correcciones sobre los
   FITS y nunca selecciona el metodo canonico por si solo.
6. Cualquier cambio posterior de metodos, bandas, ventanas laterales o reglas
   de rol requiere D1 v5.

## 2. Familias por observable

### 2.1 Continuo

Los metodos con continuo fisico comparable son:

```text
CONTINUUM_METHODS = (aperture, optimal_ls, optimal_psfsub, psffit)
```

SGF queda excluido porque pierde el continuo de la companera. LPM queda
excluido porque la componente colineal con el espectro estelar se absorbe en la
modulacion. Ambos se siguen mostrando en todas las tablas como diagnosticos,
pero sus filas B1-B6 tienen `role="diagnostic_noncomparable_continuum"` y no
pueden activar `divergent_continuum`.

### 2.2 Lineas

Los seis metodos son comparables para lineas despues de remover el continuo
local. Un par es primario para lineas si ambos metodos tienen veredicto G1
`validated` o `validated_with_bias`. SGF y LPM conservan sus caveats de v3.

El QC publica por separado `primary_pairs_continuum` y `primary_pairs_lines`.
`primary_pairs` se conserva por compatibilidad como la union ordenada.

## 3. Transformacion por banda

### 3.1 Bandas de continuo B1-B6

Para objeto y controles:

1. Se integra el `flux` persistido, que ya esta en la `SCALEREF` comun.
2. No se divide por throughput E4/G1.
3. No se renormaliza ni se remueve una pendiente.

La fila declara `comparison_mode="raw_total_continuum"`. Las columnas
`flux_i_corr`, `flux_j_corr` y `diff_corr` conservan sus nombres por
compatibilidad, pero son iguales a los valores crudos integrados. Las columnas
de throughput se informan y declaran `applied=false` para esa fila.

### 3.2 Bandas de linea LHalpha, LHbeta y LOI

Para cada metodo, tanto en objeto como en cada control:

1. Se estima el continuo con la mediana movil canonica sobre una ventana de
   80 Angstrom, excluyendo flags de ventana mala y skyline.
2. Se resta ese continuo antes de integrar la banda congelada.
3. Se divide el residual de linea por el throughput del metodo cuando G1 lo
   valida y la medida esta acotada, exactamente una vez y solo en memoria.
4. La incertidumbre de throughput se propaga como en D1 v3.

La fila declara
`comparison_mode="local_continuum_subtracted_throughput_line"`.

LOI conserva 8436-8456 Angstrom. La posible contribucion Pa 18 se registra como
caveat; no se mueve la banda despues de mirar el resultado.

## 4. Scale-check y gate de controles

El scale-check estructural sigue usando niveles de control en la escala fisica
persistida, sin throughput Halpha. Detecta convenciones rotas, no diferencias
de linea.

El gate binomial de controles se calcula por par sobre las filas que pueden
afectar el veredicto de ese observable:

- continuo: solo pares dentro de `CONTINUUM_METHODS`;
- lineas: todos los pares G1-validados.

Las filas diagnosticas permanecen en CSV/QC pero no degradan un observable al
que no pertenecen.

## 5. Reglas de veredicto

Sobre pares limpios y roles primarios:

1. `divergent_continuum` si alguna comparacion primaria de continuo cumple las
   reglas congeladas de v3: una banda con `p<p_strong` o al menos dos con
   `p<p_divergent`.
2. Si el continuo es consistente, `divergent_lines` si alguna fila primaria de
   linea tiene `p<p_divergent`.
3. `consistent` si todos los observables primarios limpios quedan dentro de
   umbral.
4. `uninterpretable` si no queda ningun par limpio para continuo ni lineas.

El QC conserva los resultados diagnosticos excluidos y explica por que no
tuvieron poder de veto. El arbol `recommended_method` de v3 solo se ejecuta con
veredicto global `consistent`; la eleccion canonica sigue siendo humana.

## 6. QC y tablas

`method_comparison.csv` y `method_comparison_controls.csv` anaden:

```text
comparison_mode
throughput_applied_i
throughput_applied_j
observable_role
```

`stage_x10_qc.json` anade:

```json
{
  "spec_version": "D1_v4",
  "comparison_policy": {
    "continuum_methods": ["aperture", "optimal_ls", "optimal_psfsub", "psffit"],
    "continuum_mode": "raw_total_continuum",
    "line_mode": "local_continuum_subtracted_throughput_line",
    "line_continuum_window_A": 80.0
  },
  "primary_pairs_continuum": [],
  "primary_pairs_lines": [],
  "excluded_from_continuum_verdict": {
    "sgf": "continuum removed by construction",
    "lpm": "collinear continuum can be absorbed"
  }
}
```

Las matrices `t_matrix` y `z_matrix` conservan la forma par por banda, pero sus
valores siguen la transformacion v4 declarada por cada fila.

## 7. Verificaciones

- V1: sigma empirica sana, sin usar STAT como sigma.
- V2: controles limpios por observable y par.
- V3: continuo de `optimal_ls` vs `optimal_psfsub` no cambia al alterar solo
  throughputs Halpha sinteticos.
- V4: una constante aditiva de continuo no produce divergencia de linea tras la
  resta local.
- V5: throughput se aplica una vez a lineas y cero veces a B1-B6.
- V6: SGF/LPM no pueden activar `divergent_continuum`.
- V7: scale-check y calibracion t mantienen los umbrales congelados.
- V8: productos FITS y NPZ son byte-identicos antes y despues de D1.

## 8. Tests minimos

1. Cambiar T de un metodo no cambia ninguna t de B1-B6.
2. Cambiar T si cambia la comparacion de linea correspondiente.
3. Dos espectros con la misma linea y distinto pedestal dan linea consistente.
4. SGF divergente en B6 no cambia un veredicto de continuo consistente entre
   los cuatro metodos comparables.
5. Un par espacial divergente en B5/B6 conserva `divergent_continuum`.
6. Una linea divergente con continuo consistente produce `divergent_lines`.
7. Los 15 pares y todas las filas diagnosticas se escriben.
8. QCs D1 v3 historicos no se reinterpretan.

## 9. Protocolo de parada

Parar si faltan productos o controles, SCALEREF difiere, el scale-check falla
para todos los pares primarios de un observable, no se puede estimar el
continuo local, o el veredicto es `uninterpretable`. El checkpoint final siempre
presenta ambas matrices, caveats y espera la eleccion humana antes de D2.
