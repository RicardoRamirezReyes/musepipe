# Especificación C4 · `X03_psf_fitting` — brief de ejecución para Codex (ChatGPT 5.5)

Fecha: 2026-07-01. Etapa C4 (ESENCIAL, nueva) del plan
`docs/plan_roxs12_reduccion_multimetodo.md`. Prerrequisitos: C1, C2 (formato),
C3 (deseable para comparar), B3, A4/B1.

Es el **método primario** recomendado por la literatura para un compañero a
~1.75″: ajuste simultáneo estrella+compañero por canal. Ataca la
decontaminación del halo de frente, sin sustracción agresiva.

---

## 0. Rol, objetivo y definición de terminado

**Rol del agente**: implementar un ajuste por mínimos cuadrados LINEALES por
canal:

```text
D(y,x) = a·P(y,x; pos_star) + b·P(y,x; pos_comp) + c₀ + c₁·y + c₂·x
```

con `P` = psf_model de C1, posiciones FIJAS (de B3, con seguimiento cromático
suave si aplica), y pesos 1/V. Las incógnitas por canal son solo
`(a, b, c₀, c₁, c₂)`: el problema es lineal, exacto y rápido. Toda la
no-linealidad (PSF, posiciones) quedó resuelta aguas arriba — mantenerlo así
es la decisión de diseño central de la etapa.

**Definición de terminado**:

1. `musepipe/extraction/psffit.py` produce `spec_psffit_object.fits` Y
   `spec_psffit_star.fits` (formato `SpectrumProduct`), más el cubo residual
   del ajuste.
2. Errores de `b(λ)` desde la matriz de covarianza del ajuste, escalados por
   los factores de A4/B1.
3. Diagnóstico de condicionamiento y de crosstalk (§4) ejecutado y en QC.
4. Verificaciones §6 en verde; tests §7 pasando; suite del repo intacta.
5. (Opcional, no bloqueante) `pampelmuse_adapter.py` como verificación externa
   — solo si el usuario lo aprueba al llegar ahí.

---

## 1. Límites duros

1. **El modelo lineal de arriba es el modelo.** Términos extra de fondo
   (curvatura, más grados) o posiciones libres por canal → preguntar antes;
   cada grado de libertad extra es covarianza con `b` (§4.1).
2. `psf_model` como caja negra (API de C1); posiciones de B3; formato de C2.
   Nada se re-ajusta ni se re-define aquí.
3. La región de ajuste es la de config (§2); prohibido recortarla u
   optimizarla mirando el espectro resultante.
4. Runs históricos intocables; ramas y commits como el resto del bloque
   (`stage-c4-psffit`).
5. Sin reintentos automáticos con modelos alternativos si χ² es malo: se
   diagnostica (§6) y se reporta.

---

## 2. Entradas y configuración

- Cubo de B2 + STAT escalado (A4/B1). SIN sustracción previa de fondo: el
  plano c₀+c₁y+c₂x del modelo absorbe cielo residual local.
- `psf_model.json` (con híbrido si C1 lo aplicó), posiciones B3 (+deriva
  cromática), máscaras de ventanas malas.
- Config: `fit_region` (default: unión de dos discos, radio 20 px sobre la
  primaria y 12 px sobre el compañero — cubre halo y compañero sin meter todo
  el campo), `chromatic_positions: auto` (sigue el veredicto de B3).

## 3. Operaciones

1. Construir la matriz de diseño por canal (5 columnas) sobre los píxeles
   válidos de `fit_region`; resolver por WLS (`lstsq` sobre la matriz
   blanqueada por √V); guardar `(a, b, c)`, matriz de covarianza 5×5, χ²ᵣ y
   nº de píxeles por canal.
2. Números de diagnóstico por canal: número de condición de la matriz
   normalizada y correlación normalizada `ρ(a,b)` de la covarianza.
3. Cubo residual `D − modelo` completo, guardado (lo consumen E1/E2 y V2).
4. Espectros al formato estándar: `flux=b`, `flux_err=√cov_bb×factores`;
   `apcorr=1.0` con nota en header (el ajuste con P normalizada a
   `norm_radius_px` ya devuelve flujo "total" en esa convención — heredar
   `NORMRAD` en el header, misma convención que C2/C3 para que D1 compare
   peras con peras).
5. `flux_err_emp`: repetir el ajuste completo en las posiciones de control al
   mismo radio (sustituyendo pos_comp por cada control) → dispersión de
   `b_control(λ)`. Es la validación empírica más honesta del error del método.

## 4. Puntos donde mantener máxima atención

1. **Degeneración fondo–compañero**: el término plano puede robar flujo de `b`
   si la región de ajuste es chica o el halo tiene gradiente fuerte. Vigilancia:
   `ρ(b, c)` de la covarianza por canal; si mediana |ρ| > 0.5, reportar (no
   agrandar la región en silencio).
2. **Crosstalk estrella→compañero**: errores del modelo PSF hacen que líneas
   fotosféricas de la primaria aparezcan espuriamente en `b(λ)`. Métrica
   obligatoria en QC: correlación entre `b(λ)` y `a(λ)` en el espacio de
   líneas (ambos con continuo removido por `continuum_running_median`, en
   ventanas de líneas estelares fuertes). Es LA firma del fallo de este
   método, y el motivo del test de potencia §7.3.
3. **ρ(a,b)**: a 1.75″ debe ser pequeño (~0). Si algún canal lo dispara, algo
   está mal (máscara, NaN, región). Verificarlo, no asumirlo.
4. **χ²ᵣ como diagnóstico, no como objetivo**: con STAT verde de A4, χ²ᵣ≫1
   por canal señala residuo estructurado del modelo PSF (heredado de C1) —
   se reporta hacia C1, no se esconde inflando errores. La inflación
   (`err × √χ²ᵣ`) se aplica SOLO como columna adicional documentada
   (`ERRINFL` en header), nunca sustituyendo al error base en silencio.
5. **El espectro de la primaria es un producto de control, no basura**:
   `spec_psffit_star.fits` vs apertura grande sobre la primaria (V3) valida la
   escala absoluta de todo el ajuste.
6. **Ventanas malas**: se ajustan igual (el ajuste es espacial, por canal) y
   el flag viaja en el producto; el gap del láser lleva además el flag de
   PSF interpolada de C1.

## 5. Esquema de `spec_psffit_qc.json`

```json
{
  "stage": "x03_psffit",
  "run_id": "...",
  "fit_region": {"star_radius_px": 20, "comp_radius_px": 12, "n_pixels_median": 0},
  "positions": {"from": "stage01c_qc.json", "chromatic": false},
  "conditioning": {"condition_number_median": 0.0, "rho_ab_median": 0.0,
                    "rho_bc_median": 0.0, "channels_rho_gt_0p5": 0},
  "chi2r": {"median": 0.0, "p90": 0.0, "inflation_column_written": false},
  "crosstalk": {"metric_corr_b_vs_a_lines": 0.0, "windows_used_A": []},
  "errors": {"mode": "stat|empirical", "stat_vs_empirical_median_ratio": 0.0,
              "n_controls": 0},
  "star_product_check": {"vs_large_aperture_median_ratio": 0.0},
  "open_issues": []
}
```

## 6. Verificaciones

- **V1 — χ²ᵣ(λ)**: figura con mediana ~1 (STAT verde); canales anómalos
  listados y cruzados con la métrica de stripes de B2 y skylines.
- **V2 — Residuo limpio**: mapas del cubo residual en 6 bandas; en las
  posiciones de ambas fuentes, residuo consistente con ruido (|z| medio < 1
  en stamp 5×5). Estructura remanente en el compañero = P mal centrada o mal
  ancha → cruzar con C1/V6 antes de reportar.
- **V3 — Escala absoluta**: `spec_psffit_star` vs primaria por apertura grande
  (corregida): razón mediana en [0.97, 1.03] y sin pendiente fuerte con λ.
- **V4 — Condicionamiento**: `ρ(a,b)` y número de condición vs λ, con
  umbrales marcados (|ρ|<0.3 esperado a esta separación).
- **V5 — Crosstalk**: la métrica §4.2, con figura de `b(λ)` y `a(λ)`
  (continuo removido) superpuestos en 3 ventanas de líneas estelares fuertes.
- **V6 — Contra C2/C3**: comparación en bandas de continuo y líneas — se
  ejecuta aquí como sanity check rápido; el análisis formal es D1.

## 7. Tests

```text
tests/test_psffit_synthetic.py    # escena sintética 2 fuentes + plano + ruido: a, b, c y cov recuperados analíticamente (test central)
tests/test_psffit_conditioning.py # dos fuentes acercándose: rho(a,b) crece como se espera; a 1.75"/escala real, |rho| < 0.1
tests/test_psffit_crosstalk.py    # línea inyectada SOLO en la estrella + P deliberadamente 10% ancha → la métrica de crosstalk la detecta en b(λ) (test de potencia)
tests/test_psffit_background.py   # gradiente de fondo fuerte sintético → b sin sesgo (>1%) y rho(b,c) reportado
```

## 8. Protocolo de parada y reporte

Preguntar cuando: V3 falle (escala absoluta — afecta a todo); crosstalk
detectado sobre el umbral del test de potencia en datos reales (opciones:
volver a C1, restringir fit_region, documentar como sistemático); ρ(b,c)
mediano > 0.5; χ²ᵣ ≫ 1 con STAT verde y sin causa en B2/skylines.

Reporte: tabla V1–V6; espectro del compañero con ambos errores; sistemático
heredado de C1 (de su QC) y cómo se manifiesta aquí; recomendación preliminar
para D1 (¿concuerdan los tres métodos a ojo?) sin adelantar el veredicto
formal; checklist de límites; comando de reproducción.

## Errata (2026-07-25) — V1 pasa a ser un chequeo

V1 pide χ²ᵣ mediano ~1 «(STAT verde)». El QC guardaba `chi2r.median` desde
siempre, pero **no había chequeo ni aviso**: en ROXs 42B b vale **2673** —el
residuo del ajuste es ~2700 veces la varianza declarada— y el único
`open_issue` de la etapa hablaba del marco de longitud de onda.

Implementado como `checks.v1_chi2r`, con la condición de la spec respetada:

- **Con STAT utilizable** (`errors.mode == "stat"`, que es exactamente cuando el
  producto usa el error formal) el veredicto vale: `ok` si la mediana cae en
  **[0.5, 2.0]**, configurable con `x03_chi2r_range`. Un factor 2 en χ²ᵣ es un
  factor √2 en σ; por debajo de eso no se distingue un modelo imperfecto de un
  STAT mal escalado.
- **Sin él**, `ok = None` con motivo: χ²ᵣ no tiene escala absoluta y aprobar
  sería inventar. Es el caso de ROXs 12 b, cuyo M5 está rojo (mediana 0.92,
  informada pero sin veredicto).

Publica además `p90`, `fraction_in_range` y los **5 peores canales** con su λ,
para la segunda mitad de V1: cruzarlos a mano con los canales sucios de B2 y las
skylines de A4.

Cuando falla, el `open_issue` dice que el error formal del compañero no es su
residuo y nombra a los dos sospechosos: el modelo de PSF de C1 o la escala del
STAT (A4/M5). En ROXs 42B b esto se suma a `v3_star_scale_ok = False` (el ajuste
no reproduce la primaria frente a su fotometría de apertura grande) y, en C2, a
V4(b) con box5/box3 = 0.608: **tres indicadores independientes apuntando al
mismo sitio en ese objeto**.
