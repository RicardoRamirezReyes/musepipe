# Especificacion E4 v3 - la unidad del gate de nulos es la posicion

Fecha: 2026-08-20. Revision formal de
`docs/spec_E4_v2_codex_injection_recovery.md`, que sigue siendo la referencia
para todo lo demas: v3 cambia **solo la regla de agregacion del gate V2**. El
nuevo QC declara `spec_version="E4_v3"`.

La v2 §1.5 exigia v3 para cualquier cambio en las reglas de gate. Esto lo es.

## 0. Motivo

V2 fallaba en los dos objetos con `excess_p` de 1e-04 a 1e-07, y el re-run
completo C->E de `ROXs12b_realigned` del 2026-08-20 —extraccion nueva, PSF por
observacion, cubo de C1b— lo dejo **peor**: 16 filas extremas de 72 contra 12,
`excess_p` 1.2e-07 contra 9.4e-05. Lo que se midio al mirar las filas:

1. **Las 16 filas extremas son una sola posicion.** Todas `control3`, en cuatro
   de los seis metodos, por dos factores de plantilla y dos modos de continuo.
   Cero extremas en `control1` y `control2`.
2. **El binomial las contaba como 16 pruebas independientes.**
   `binom.sf(K-1, 72, p_null)` supone 72 Bernoulli independientes. Las 72 filas
   son 3 posiciones x 6 metodos x 2 anchos x 2 modos **sobre el mismo cubo**: la
   misma zona de cielo medida muchas veces. El modo de continuo mueve el numero
   un 0.8 % (mediana de las 36 parejas), o sea filas casi calcadas. La v2 §3 se
   cuidaba de "no fingir independencia gaussiana canal a canal" y luego la
   fingia entre filas.
3. **"Extremo" solo podia significar una cosa.** Con 33 controles la FAP de
   rango vale `k/34`; el unico valor por debajo de `p_null=0.0455` es
   `1/34 = 0.0294`. La prueba individual es "esta posicion es el maximo de sus
   34", no lo que el umbral sugiere.

El defecto no esta en la extraccion ni en el cubo: esta en la agregacion.

## 1. Que cambia

Solo esto:

- La **unidad** del gate pasa de la fila a la **posicion de control**. El
  binomial corre sobre `n_positions` (3 en la grilla congelada), no sobre
  `n_rows`.
- El resumen de una posicion es la **mediana** de las FAP de sus filas. Una
  posicion es extrema si esa mediana es `< p_null`.
- Se informa la **cola baja** como diagnostico, sin voto.

No cambian: la grilla de S/N, anchos, modos de continuo, posiciones, metodos,
extractores, la definicion de throughput, la FAP de rango
`(1 + N(null >= observed))/(N + 1)`, ni los dos umbrales congelados
(`p_null=0.0455`, `alpha=0.01`). La tabla de inyecciones no pierde columnas.

## 2. La regla, entera

1. Poblacion nula: filas `variant="nominal"`, `input_snr=0`, `position_label`
   que empieza por `control`. La posicion `real` sigue sin entrar (v2 §1.2).
2. Por fila, la FAP de rango unilateral contra los controles del mismo metodo y
   el mismo factor de plantilla, igual que en v2.
3. **Por posicion**, `position_fap = mediana` de las FAP de sus filas. La
   posicion es extrema si `position_fap < p_null`.
4. Con `K` posiciones extremas de `N`, el gate falla si
   `P(X >= K | N, p_null) < alpha`.

**Por que la mediana.** Es "lo que ve el metodo tipico en esa posicion". El
minimo dejaria que un solo extractor con un defecto conocido —`optimal_ls`
sobre-resta continuo, `psffit` normaliza distinto— decidiera por los otros
cinco; el maximo dejaria que la mayoria tapase un artefacto real. La mediana
exige que **mas de la mitad de las medidas de esa posicion** coincidan (con un
numero par, la mitad justa cae en el punto medio y no marca), que es la
condicion para que el exceso este en el dato y no en un extractor.

**Que puede decidir el gate con N=3.** Con `p_null=0.0455` y `alpha=0.01`:

| K de 3 | `excess_p` | veredicto |
|---|---|---|
| 0 | 1.0 | pass |
| 1 | 0.130 | pass |
| 2 | 0.0060 | **fail** |
| 3 | 9.4e-05 | **fail** |

Es decir: **con tres posiciones de control el gate solo puede fallar si dos de
las tres son extremas.** Eso es poca potencia, y hay que decirlo en vez de
disimularlo con filas: la unica forma honesta de subirla es **mas posiciones de
control**, no mas filas por posicion. Una futura grilla con mas posiciones no
necesita v4 —`n_positions` sale de los datos— pero mover `p_null`, `alpha` o la
regla de agregacion si.

**Sobre `p_null=0.0455` con 34 valores posibles.** La tasa nula alcanzable por
posicion es como mucho `1/34 = 0.0294`, menor que `p_null`. Usar 0.0455 sobre-
estima el numero esperado de extremas, o sea agranda `excess_p`: el gate queda
del lado indulgente. Se mantiene congelado a proposito —no se mueve un umbral
mirando este run (v2 §1.4)— y queda escrito aqui que la holgura existe y en que
direccion.

## 3. La cola baja, informada y sin voto

La FAP de v2 es unilateral hacia arriba porque el gate busca **falsos
positivos**. Consecuencia: una fila muy negativa sale con `empirical_fap = 1.0`
y no es extrema nunca. Eso dejaba invisible un defecto que si es real y esta
documentado —el continuo negativo de `optimal_ls`, que en este run llega a
-741 donde los demas metodos estan entre -70 y +50.

v3 calcula tambien `empirical_fap_low = (1 + N(null <= observed))/(N + 1)` por
fila y publica el recuento en `low_tail_diagnostics`, con `gating: false`. Un
deficit de flujo donde no se inyecto nada es sobre-sustraccion, no un falso
positivo: pertenece al presupuesto de sistematicos de C3/D1, no al veredicto de
V2. Informar no es votar.

## 4. QC

`checks.v2_nulls_clean` anade y renombra:

```json
{
  "status": "pass|fail",
  "population": "control_positions_only",
  "unit": "control_position",
  "n_positions": 3,
  "n_extreme": 1,
  "n_rows": 72,
  "n_extreme_rows": 16,
  "p_null": 0.0455,
  "excess_p": 0.13,
  "gate_alpha": 0.01,
  "positions": [
    {"position_label": "control3", "n_rows": 24, "position_fap": 0.0294,
     "extreme": true, "by_method_fap": {"aperture": 0.0294, "psffit": 0.088}}
  ],
  "low_tail_diagnostics": {"gating": false, "n_extreme_rows": 0, "by_method": {}},
  "rows": []
}
```

`n_extreme` cuenta **lo que cuenta la puerta**, que ahora son posiciones; `unit`
lo declara para que no se pueda leer mal, y `n_extreme_rows` conserva el numero
de v2, que sigue siendo el detalle util. `rows` no pierde nada y gana
`empirical_fap_low`.

## 5. Verificaciones

V1, V3, V4, V5 y V6 sin cambios respecto de v2. V2 es el de esta spec.

## 6. Tests minimos

A los seis de v2 se anaden:

1. Muchas filas extremas en **una** posicion no fallan el gate (era el defecto).
2. Dos posiciones extremas de tres si lo fallan.
3. La mediana no deja que un metodo solo declare extrema una posicion, ni que
   cinco tapen a uno cuando la mitad coincide.
4. `n_extreme` cuenta posiciones y `n_extreme_rows` filas, y ambos salen al QC.
5. La cola baja se cuenta y no cambia el veredicto.

## 7. Protocolo de parada

El de v2 §7, con una correccion: "V2 muestra exceso empirico" significa exceso
**de posiciones**. Un exceso de filas concentrado en una posicion es un dato
sobre esa posicion —hay que mirarlo y decir que es— pero no para la cadena.
