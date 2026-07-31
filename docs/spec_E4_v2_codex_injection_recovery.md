# Especificacion E4 v2 - inyeccion y recuperacion con nulo empirico

Fecha: 2026-07-29. Revision formal de
`docs/spec_E4_codex_injection_recovery.md`, que conserva la interpretacion de
los QC historicos. Esta revision se congela antes de volver a ejecutar E4 sobre
`ROXs12b_realigned`. El nuevo QC declara `spec_version="E4_v2"`.

## 0. Motivo y alcance

La auditoria del cubo multiepoca encontro tres defectos de interpretacion en E4
v1:

1. El gate V2 llamaba nulo al caso S/N=0 en la posicion real del companero. Ese
   espectro contiene el dato cientifico y puede contener una linea real o un
   residuo; no es una posicion nula.
2. `recovered_snr` usaba el error interno del extractor con controles
   desactivados. Con STAT rojo, ese numero no cumple el modelo de ruido canonico
   de `docs/noise_model.md`.
3. El modo de continuo `flat` se ejecutaba con amplitud cero cuando faltaba una
   clave de config. Era duplicado exacto de `none`, por lo que V5 era vacuo.

No cambian la grilla de S/N, anchos, posiciones, perturbaciones de PSF,
extractores ni definicion de throughput. E4 v2 cambia solamente la amplitud del
continuo plano y el gate de nulos.

## 1. Limites duros

1. Grilla nominal congelada: S/N `{0,1,2,3,5,7,10}`, anchos `{LSF,2xLSF}`,
   continuo `{none,flat}` y posiciones `{real,control1,control2,control3}`.
2. La posicion `real` nunca entra en la poblacion nula de V2.
3. Significancia y FAP se estiman con controles del mismo metodo, al mismo radio
   y procesados como el objeto. STAT y el scatter del espectro del objeto no son
   sigma para V2.
4. No se mueve el umbral mirando este run. El nivel individual de referencia es
   `p_null=0.0455`; el exceso global usa el test binomial unilateral congelado
   con `alpha=0.01`, igual que el gate de controles D1.
5. Un cambio futuro de grilla, metodos, bandas de continuo o reglas de gate
   requiere E4 v3.

## 2. Continuo plano no nulo

Si `h04_continuum_flux_density` esta configurado, E4 usa ese valor y registra
la fuente `config`.

Si no esta configurado, E4 lo mide antes de construir la grilla:

1. Lee los productos de produccion que preservan continuo:
   `aperture`, `optimal_ls`, `optimal_psfsub` y `psffit`.
2. Usa las bandas laterales congeladas `[6500,6540]` y `[6585,6625]` Angstrom,
   excluyendo flags de ventana mala y skyline.
3. Convierte cada producto a la escala NORMRAD de la inyeccion dividiendo
   `flux/apcorr`.
4. Calcula la mediana por metodo y adopta la mediana de los valores positivos y
   finitos. Exige al menos dos metodos positivos y un valor adoptado
   estrictamente positivo; si no, E4 se detiene.

El QC registra bandas, productos, valor por metodo, valor adoptado y escala
`normrad_flux_density`. V5 falla si `flat` es cero o si sus filas son identicas
a `none`.

## 3. Nulo empirico V2

### 3.1 Poblacion evaluada

V2 considera solo filas nominales con `input_snr=0` y
`position_label in {control1,control2,control3}`. Conserva por separado metodo,
ancho y modo de continuo. Las filas de la posicion real se reportan como
`science_position_diagnostics`, sin poder de veto.

### 3.2 Referencia empirica

Para cada metodo y ancho se leen sus espectros de control de produccion. Se usa
`control_spectra_raw` cuando existe, porque la inyeccion opera en escala
NORMRAD; si falta, se revierte una unica vez la `apcorr` del producto. Sobre
cada control:

1. se resta una mediana movil de 80 Angstrom;
2. se aplica el mismo matched filter de E1 en el centro Halpha configurado;
3. se conserva el flujo recuperado como distribucion nula empirica.

Para cada fila V2 se calcula la FAP de rango unilateral
`(1 + N(null >= observed))/(N + 1)`. Una fila es extrema si
`FAP < p_null=0.0455`.

El gate global no exige cero extremos. Con `K` extremos entre `N` filas, falla
solo si `P(X>=K | N,p_null) < alpha=0.01`. Esto prueba exceso de falsos
positivos sin fingir independencia gaussiana canal a canal.

## 4. QC y salidas

`stage_h04_qc.json` anade:

```json
{
  "spec_version": "E4_v2",
  "continuum_injection": {
    "value": 0.0,
    "scale": "normrad_flux_density",
    "source": "config|median_continuum_preserving_methods",
    "bands_A": [[6500, 6540], [6585, 6625]],
    "by_method": {}
  },
  "checks": {
    "v2_nulls_clean": {
      "status": "pass|fail",
      "population": "control_positions_only",
      "n_rows": 0,
      "n_extreme": 0,
      "p_null": 0.0455,
      "excess_p": 1.0,
      "gate_alpha": 0.01,
      "rows": []
    }
  }
}
```

La tabla de inyecciones no pierde columnas. Puede anadir un CSV diagnostico de
nulos, pero el throughput nominal sigue saliendo de las mismas filas que en v1.

## 5. Verificaciones

- V1: regresion historica, con el mismo comportamiento de disponibilidad de v1.
- V2: gate empirico de esta spec; ninguna fila real se cuenta como nulo.
- V3: monotonia sin cambios.
- V4: jerarquia sin cambios.
- V5: continuo plano positivo, no identico a `none`, y degradacion reportada.
- V6: controles de referencia con la misma malla, RUNID y escala NORMRAD.

## 6. Tests minimos

1. Un hit solo en `real` no falla V2.
2. Un numero esperado de extremos en controles pasa; un exceso binomial falla.
3. La FAP se obtiene de controles, no de `recovered_snr`.
4. Sin config, el continuo se mide de cuatro productos y es positivo.
5. `flat=0` o `flat` identico a `none` falla V5.
6. Throughput y grilla nominal permanecen byte-equivalentes cuando solo cambia
   el diagnostico V2.

## 7. Protocolo de parada

Parar si faltan controles de cualquier metodo, las escalas no se pueden llevar
a NORMRAD, el continuo medido no es positivo, V2 muestra exceso empirico, o
V3/V4 fallan. No aceptar automaticamente una limitacion historica con una
poblacion nula distinta.
