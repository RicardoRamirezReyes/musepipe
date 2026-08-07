# Especificación C2 · `X01_aperture` — brief de ejecución para Codex (ChatGPT 5.5)

Fecha: 2026-07-01. Etapa C2 (ESENCIAL) del plan
`docs/2026-07-10_plan_roxs12_reduccion_multimetodo.md`. Prerrequisitos: B2, B3; C1 solo
para la corrección de apertura (puede correr en modo provisional sin ella).

C2 tiene doble papel: (a) envolver la cadena 04b ya validada como el método de
control "A: apertura + fondo local", y (b) **implementar el contrato de
producto espectral estándar** (plan §2.3) que TODOS los extractores usarán.
La parte (b) es la más importante: C3, C4, D1 y E1 se construyen sobre ella.

---

## 0. Rol, objetivo y definición de terminado

**Rol del agente**: cero ciencia nueva. La extracción es la que ya está
validada (superficie local `plane` + apertura); el trabajo es de
infraestructura: formato estándar, varianza, corrección de apertura y
trazabilidad.

**Definición de terminado**:

1. `musepipe/extraction/product.py` implementado: clase `SpectrumProduct` +
   lector/escritor FITS del formato §2 — con tests de round-trip.
2. `musepipe/extraction/aperture.py` produce `spec_aperture_object.fits` en
   ese formato, envolviendo `stage04b` + `apertures.py` SIN modificarlos.
3. Equivalencia demostrada: el flujo por canal coincide con el que la cadena
   histórica (04b→07) produce para la misma apertura (tolerancia B1 §4).
4. Columna `flux_err` poblada con STAT propagado × factores (A4/B1), y
   comparada con el error empírico histórico.
5. Corrección de apertura por λ aplicada desde la curva de crecimiento de C1
   (o `aperture_correction: "none"` explícito en modo provisional).
6. `spec_aperture_qc.json`; verificaciones §5 en verde; suite del repo intacta.

---

## 1. Límites duros

1. `stage04b`, `apertures.py`, `localfit.py`: intocables. C2 es un consumidor.
2. Runs históricos intocables; validación sobre copia aislada.
3. **El formato estándar se define UNA vez aquí y se congela**: cambios
   posteriores de esquema requieren migración versionada (`format_version` en
   el header). C3/C4 no podrán "añadir columnas a su manera".
4. La corrección de apertura NUNCA se aplica en silencio: modo y factor por
   canal quedan en el producto (columna `apcorr`) y en QC.
5. Git: rama `stage-c2-aperture`.

---

## 2. El contrato `SpectrumProduct` (congelar con C2)

FITS BinTable, HDU `SPECTRUM`:

| Columna | Tipo | Descripción |
|---|---|---|
| `wave_A` | f8 | Malla de λ (marco registrado en header: `WFRAME`) |
| `flux` | f8 | Flujo (unidades en header `BUNIT`, las del cubo) |
| `flux_err` | f8 | Error 1σ propagado (STAT×factores) |
| `flux_err_emp` | f8 | Error empírico (controles/continuo), para comparación |
| `apcorr` | f8 | Corrección de apertura aplicada (1.0 si none) |
| `npix_eff` | f8 | Píxeles efectivos usados en el canal |
| `flags` | i4 | Bitmask: 1=ventana mala, 2=skyline, 4=interpolado, 8=clipeado |

Header obligatorio: `FORMATV` (versión del formato), `METHOD`
(`aperture|optimal|psffit|hrsdi`), `RUNID`, `SRCPOS_Y`, `SRCPOS_X` (de B3),
`APERTURE` (descripción), `WFRAME` (`barycentric|topocentric`, heredado de
A4), `INCUBE` + `INCUBESH` (ruta y sha256 del cubo de entrada), `NORMRAD`
(radio de normalización PSF si aplica). HDU opcional `COVARIANCE` para D/C6.

En `product.py`: dataclass, `write(path)`, `read(path)`, y validador
`validate()` que verifica columnas, header y consistencia (usado por D1/E1 al
leer cualquier producto).

## 3. Operaciones

1. Implementar `product.py` + tests (round-trip, validador rechaza productos
   malformados).
2. `aperture.py`: ejecutar la cadena 04b sobre el run objetivo (o leer sus
   productos si ya existen — decisión por config `reuse_stage04b: true`),
   extraer con `box_spectrum_sum` (3×3 default y 5×5 como variante, ambas
   escritas), coordenadas desde `stage01c_qc.json` (B3), NUNCA hardcodeadas.
3. Varianza: STAT por la misma apertura en cuadratura × `stat_factor_box3`
   (A4/M5) × `covariance_factor_box3` (B1). Si STAT es rojo en A4:
   `flux_err = flux_err_emp` y header `ERRMODE=empirical`.
4. Error empírico: mismo procedimiento histórico (controles al mismo radio,
   `robust_sigma`), reutilizando `musepipe/stats.py`.
5. Corrección de apertura: desde la curva de crecimiento de `psf_model` (C1)
   evaluada en la apertura usada, por canal. Punto de atención: usar el
   `norm_radius_px` del `psf_model.json`, no otro.
6. Flags por canal desde las máscaras existentes (5780–6050 Å, skylines, gap
   interpolado de C1).

## 4. Esquema de `spec_aperture_qc.json`

```json
{
  "stage": "x01_aperture",
  "run_id": "...",
  "positions_from": "stage01c_qc.json",
  "reuse_stage04b": true,
  "apertures": ["box3", "box5"],
  "errors": {"mode": "stat|empirical", "stat_factor_box3": 0.0,
              "covariance_factor_box3": 0.0,
              "stat_vs_empirical_median_ratio": 0.0},
  "aperture_correction": {"mode": "psf_growth_curve|none", "median": 0.0},
  "equivalence": {"baseline": "stage07 product", "verdict": "identical|allclose",
                   "max_rel_diff": 0.0},
  "open_issues": []
}
```

## 5. Verificaciones

- **V1 — Equivalencia**: flujo por canal vs cadena histórica (misma apertura,
  mismas coordenadas): idéntico dentro de B1 §4. Si B3 midió coordenadas que
  difieren de las históricas, correr AMBAS y reportar la diferencia de
  espectro como figura (es información científica, no un fallo).
- **V2 — Errores coherentes**: razón `flux_err/flux_err_emp` vs λ, mediana en
  [0.7, 1.4]; figura con ambas bandas de error sobre el espectro.
- **V3 — Round-trip del formato**: write→read→validate sobre el producto real.
- **V4 — Corrección de apertura sana**: apcorr(λ) suave, en [1.0, ~1.6] para
  box3, y consistente entre box3 y box5 (el espectro corregido de ambas debe
  coincidir dentro de errores — esa es la prueba de que la curva de
  crecimiento de C1 funciona).

## 6. Tests

```text
tests/test_spectrum_product.py     # round-trip, validador, versionado del formato
tests/test_aperture_extractor.py   # mini-cubo sintético con fuente de flujo conocido → flux y flux_err correctos analíticamente
tests/test_aperture_apcorr.py      # PSF sintética conocida → box3/box5 corregidos coinciden con el flujo total
```

## 7. Protocolo de parada y reporte

Preguntar cuando: V1 falle sin causa identificable; V4 muestre box3 y box5
incompatibles (señal de PSF de C1 mala en el core — retroalimentar a C1);
STAT dé resultados absurdos pese a verde en A4. Reporte: tabla V1–V4,
decisión de `ERRMODE`, y confirmación de formato congelado (`FORMATV=1`).

## Errata (2026-07-25) — V4 son dos cosas, y la que importa faltaba

V4 pide (a) que `apcorr(λ)` esté en rango y (b) que **el espectro corregido de
box3 y box5 coincida dentro de errores** — «esa es la prueba de que la curva de
crecimiento de C1 funciona». El código solo implementaba (a).

**(a) no puede pasar en NFM.** El rango del código es `mediana ≥ 1.0` y
`máx ≤ 1.8`, coherente con el «[1.0, ~1.6] para box3» de arriba, que supone una
apertura que recoge casi toda la luz. Con la PSF psfao normalizada a r=25 px una
box3 recoge **~2%**, así que `apcorr` vale decenas: 44.65 en ROXs 12 b y 18.70
en ROXs 42B b. El chequeo lleva rojo desde siempre y no es accionable; la
magnitud de la corrección, por sí sola, no dice si la curva es buena.

**(b) implementada** como `checks.v4_apcorr_consistency`: fracción de canales
con |box3 − box5| ≤ √(σ3² + σ5²), medida en **7500–9000 Å** (en el azul el
compañero tiene S/N<1 y las dos aperturas «concuerdan» siempre porque las dos
miden ruido), con umbral 0.90. Knobs: `x01_apcorr_consistency_band_A` y
`x01_apcorr_consistency_threshold`. Publica además el cociente
box5/box3 mediano, que es el número interpretable.

Medido sobre los productos existentes: **ROXs 12 b 0.859× (87.9% de canales)** y
**ROXs 42B b 0.608× (15.2%)**. Los dos por debajo del umbral, el segundo de
forma grave. Según §7 el sospechoso es el modelo de PSF en el core y toca
retroalimentar a C1 — no la fotometría, que se validó contra `photutils`
(máscara idéntica bit a bit, y el redondeo del centro al píxel cuesta <1%).
