# Especificación G2 v3 · `g2_measure_lines` — V3 compara categorías que se pueden comparar

Fecha: 2026-09-25. Revisión de `docs/spec_G2_v2_codex_spectral_measurements.md` (v2), que a su vez
revisa la v1. Solo cambia **el chequeo V3** (v1 §7) y el bloque del QC que lo publica; el resto de
la v1 y de la v2 sigue vigente palabra por palabra. Decisión del autor, 2026-09-25, tomada como
manda la v1 §8 («preguntar cuando V3 contradiga H01»).

## 0. Por qué existe una v3

La v1 pedía que el estado de Hα en G2 «coincida en categoría» con H01. La implementación tradujo
los cuatro estados de G2 a los tres veredictos de E1 uno a uno: `detected` → `detection`,
`marginal` → `candidate`, `upper_limit`/`not_measurable` → `non_detection`. La traducción de
`marginal` no se sostiene, porque las dos categorías se definen en escalas distintas:

| | G2 `marginal` | E1 `candidate` |
|---|---|---|
| estadística | filtro adaptado a λ fija (la esperada) | máximo del filtro en ±500 km/s y 3 plantillas |
| umbral | 3 ≤ z_emp < 5 (local) | FAP global < 0.01 en algún método (con búsqueda) |

Una señal de ~3σ cae a un lado u otro de cada umbral según el ruido. Es lo que pasó en
`ROXs42Bb_realigned` al bajar el radio de C1 a 55 px (2026-09-22):

| | G2 | E1 (psffit) |
|---|---:|---:|
| z | 3.09 (inflación 0.84, suelo en 1) | 2.97 |
| categoría | `marginal` | `non_detection` (FAP paramétrica 0.0108; IC95 [0.0022, 0.0248]) |

Las dos escalas **no se han separado** —la z es la misma dentro de 0.12—, que es lo único que la v2
dejó a V3 para vigilar («si vuelve a saltar, es que las dos escalas se han separado otra vez»). V3
saltaba por la traducción, no por los datos.

## 1. §V3 revisada

El estado de Hα en G2 tiene que ser **compatible** con el veredicto de E1:

| G2 | veredictos de E1 compatibles |
|---|---|
| `detected` | `detection` |
| `marginal` | `candidate`, `non_detection` |
| `upper_limit`, `not_measurable` | `non_detection` |

Cualquier otro par es un issue `blocking`, con el mismo texto de siempre. V3 **sigue disparando**
ante la contradicción para la que existe (la de la v1: `detected` frente a `non_detection`) y ante
un límite o una marginal frente a una detección.

## 2. QC

`halpha_reconciliation_v3` conserva `g2_halpha_status`, `h01_verdict` y `consistent`, y añade
`compatible_h01_verdicts` (la fila de la tabla de §1) y `rule: "spec_G2_v3"`.

## 3. Lo que NO cambia

- Los umbrales 5/3, la escala empírica de la v2 y el suelo en 1 de la inflación.
- El veredicto de E1 y su criterio (FAP < 0.01), que es una decisión congelada.
- ROXs 12 b (`detected` / `detection`) no se mueve.

Implementación: `halpha_reconciliation_v3` y `V3_COMPATIBLE` en
`musepipe/stages/stage_g2_measure_lines.py`; tests en `tests/test_g2_v3_reconciliation.py`.
