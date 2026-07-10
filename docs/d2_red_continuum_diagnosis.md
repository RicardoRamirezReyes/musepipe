# Diagnóstico: el "continuo rojo inestable" de D2 (v3_continuum_stable)

Fecha: 2026-07-10. Run: `runs/ROXs12b_realigned`. Motivado por el único rojo
que quedaba en F1 tras la política de gate: `D2 checks.v3_continuum_stable=fail`.
Objetivo del usuario: **entender el problema, no esconderlo ni ignorarlo.**

## TL;DR

El "continuo rojo inestable" **no es un defecto de PSF y C1 ya usa Psfao**. Es
la superposición de tres cosas reales, que el chequeo v3 original confundía:

1. **Señal REAL del compañero frío en el rojo** (dominante). El compañero es un
   objeto subestelar frío: su flujo sube al rojo (S/N 0.7 → 8.3) con estructura
   de banda ancha (SED + bandas moleculares). Un continuo polinómico grado-5 no
   puede seguirla → `|runmed − poly5|` sale grande. Subir a grado-15 reduce los
   canales marcados de 19.2% a 8.6%: buena parte era **rigidez del modelo**, no
   inestabilidad. Esto es señal real mal etiquetada, NO un error.
2. **Un sistemático REAL dependiente del método** (la divergencia B6 de D1). Los
   dos métodos G1-validados (psffit, optimal_psfsub) coinciden en la FORMA del
   rojo (corr = **+0.957**) pero difieren en el NIVEL: razón mediana psffit/psfsub
   = **1.76×** en 8600–9100 Å. Es resta de halo cromática residual de la primaria
   brillante en la posición del compañero débil. NO es fuga espectral (corr del
   residuo del compañero con el espectro de la estrella = +0.016).
3. **C1 ya es Psfao** (`form: psfao`), residuo de anillo 4.6% (rojo 4.4%), dentro
   del objetivo <5%. No hay "refinamiento con Psfao" pendiente — ya está hecho.
   El sistemático de nivel es el piso alcanzable a esta geometría (halo brillante
   + compañero a 1.8" que es 0.04% de la estrella).

## Evidencia (todo medido, `spec_calibrated_*` del run realineado)

| Prueba | Resultado | Interpretación |
|---|---|---|
| Canales `|runmed−poly5|>staterr` por 500 Å | 0% <7250 Å, 15–60% >7250 Å | El "problema" es solo el rojo |
| S/N del compañero por banda | 0.7 (azul) → 8.3 (9250 Å) | Objeto frío, señal real que sube al rojo |
| corr(residuo compañero, espectro estrella) | +0.016 | Descarta fuga de halo espectral |
| corr(psffit, psfsub) forma del rojo | +0.957 | La forma roja es señal real compartida |
| razón nivel psffit/psfsub (rojo) | 1.76× | Sistemático de nivel dependiente de método |
| poly grado 5 / 9 / 15 → % marcado | 19.2 / 16.8 / 8.6% | Rigidez del modelo, no inestabilidad física |
| C1 form / residuo anillo rojo | psfao / 4.4% | PSF ya óptima |

## Por qué el chequeo v3 original era engañoso

`sys_continuum = |runmed(80Å) − poly(grado 5)|` mide "cuán no-polinómico es el
continuo suave". Para un espectro REAL de enana fría (bandas moleculares anchas)
eso es grande **por construcción** — marca señal real como inestabilidad. No
distingue estructura astrofísica real de un sistemático.

## Corrección honesta aplicada (no oculta el problema)

El discriminante correcto es **inter-método**: ambos métodos comparten la señal
real, así que su DIFERENCIA de continuo más allá del error combinado es el
sistemático dependiente de método (lo único que no es real). Es lo que mide D1.
`stage_x11_calibrate` ahora:

- Calcula `continuum.intermethod_systematic` (canónico vs el otro método
  G1-validado): fracción de canales donde los continua concuerdan dentro del
  error, y la razón de nivel en el rojo.
- **v3_continuum_stable pasa a gatear sobre esa métrica libre de señal**
  (`fraction_channels_methods_agree ≥ 0.90`), no sobre `|runmed−poly5|`. La
  métrica poly queda como diagnóstico secundario, etiquetada como contaminada
  por señal.
- Emite un open_issue que caracteriza el sistemático rojo (1.76×) con su causa.

**Resultado en el realineado**: `fraction_channels_methods_agree = 0.317` (rojo
0.215). v3 sigue en **fail** — pero ahora por la razón CORRECTA y documentada
(sistemático real inter-método de resta de halo), no por confundir señal real.
El umbral usado (error por canal, ~8× mayor que el ruido del continuo suavizado)
es conservador: el sistemático es robusto.

## Impacto en la ciencia (importante)

**Ninguno sobre el endpoint.** La no-detección (E1) es un test de LÍNEA DE
EMISIÓN Hα, no del continuo; el sistemático es una incertidumbre de NIVEL del
continuo en un compañero débil. El límite de Ṁ (E3) usa el flujo de la línea, no
el nivel absoluto del continuo rojo. La forma roja real (bandas moleculares) es
información de tipo espectral para G3 (caracterización), con la salvedad del
nivel 1.76× ya caracterizada.

## Conclusión

El rojo NO se puede "estabilizar" más: la forma es señal real (no se debe
aplanar) y el nivel tiene un sistemático inter-método que es el piso a esta
geometría con la PSF ya óptima (Psfao, 4.4%). Lo honesto es **caracterizarlo**
(hecho: 1.76×, causa identificada, no afecta la línea) y, dado que es un
sistemático real entendido, no removible y sin impacto en la ciencia, tratarlo
como **limitación aceptada documentada** en la política de gate de F1 — con la
caracterización completa visible, no como bandera roja sin explicar.
