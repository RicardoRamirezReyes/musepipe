# Especificación B1 · `01_load_align_crop` — brief de ejecución para Codex (ChatGPT 5.5)

Fecha: 2026-07-01. Etapa B1 del plan `docs/2026-07-10_plan_roxs12_reduccion_multimetodo.md`.
A diferencia del bloque A (código nuevo), B1 es una **migración por
equivalencia + extensión**: la lógica ya existe y está validada en
`01_load_align_crop.ipynb` / `01_lowMemory_load_align_crop.ipynb`; el trabajo
es moverla a `musepipe/` sin cambiar resultados, y SOLO DESPUÉS extenderla.

---

## 0. Rol, objetivo y definición de terminado

**Rol del agente**: dos sub-tareas estrictamente secuenciales, cada una con su
gate. Prohibido mezclarlas en un mismo commit o fase:

- **B1a — Migración por equivalencia**: extraer la lógica del notebook a
  `musepipe/stages/stage01_align.py` reproduciendo numéricamente los productos
  existentes.
- **B1b — Extensiones**: (i) propagación de la extensión STAT por las mismas
  transformaciones, (ii) registro de shifts por exposición en QC, (iii)
  soporte de `entry_point` (cubo del bloque A o cubo histórico).

**Definición de terminado**:

1. B1a: `run_stage01()` reproduce `stage01_cube_stack.fits` del baseline sobre
   una copia de validación (tolerancia de §4), sin tocar el run original.
2. B1b: el producto incluye extensión STAT propagada; QC registra shifts,
   kernel de interpolación y factor de covarianza espacial (§5.3).
3. `stage01_qc.json` según §6; verificaciones §7 en verde.
4. Notebook `01_load_align_crop.ipynb` convertido en capa fina (config →
   `run_stage01()` → figuras), como los `Far_*` migrados.
5. Tests nuevos pasando + las 52+ pruebas existentes intactas.

---

## 1. Límites duros

1. Regla del refactor del repo (vigente): **no cambiar resultados científicos
   y estructura en la misma pasada**. B1a no altera ni un bit de la salida;
   cualquier mejora que se te ocurra durante la migración se anota en
   `open_issues`, no se implementa.
2. Runs históricos intocables. La validación de equivalencia corre sobre una
   copia aislada (`runs/ROXs12b_stage01_validation/`), patrón ya usado en
   `ROXs12b_refactor_validation`.
3. No modificar módulos existentes de `musepipe/` salvo: añadir la etapa nueva
   a `musepipe/stages/__init__.py` y, si es imprescindible, funciones NUEVAS
   en módulos comunes (nunca cambiar firmas ni comportamiento existentes).
4. El notebook original se conserva en git hasta que la equivalencia esté
   demostrada; se adelgaza solo al final de B1a.
5. Presupuesto de memoria: la variante low-memory es la ruta canónica de la
   migración (cierra parte de O4); la variante estándar puede quedar como
   opción, no como default.
6. Git: rama `stage-b1-align`; un commit B1a (migración) claramente separado
   de los commits B1b (extensiones).

---

## 2. Contexto que debes leer antes de escribir código

- `docs/2026-06-17_refactor_plan_far_objects.md`: el patrón de migración por equivalencia
  ya ejecutado para 04b/06/07/07b/08 — B1 replica ese proceso.
- Config relevante del run (`runs/ROXs12b/config/config.json`):
  `cube_files` (para ROXs12b: UN solo cubo), `data_ext=1`, `crop_npix=170`,
  `centering_method`, `stage01_profile` (`maoppy_refined` con fallback a
  centrado por pico si `maoppy` no está), `spatial_shift_mode`,
  `spectral_grid_mode`.
- Convención de coordenadas del repo: los docs usan `(x, y)` para reportar y
  `(y, x)` en índices de array. Respeta la convención existente en QC y no
  introduzcas una tercera.

## 3. B1a — Migración por equivalencia

1. Inventariar las funciones del notebook (y de la variante low-memory);
   decidir qué va a `stage01_align.py` y qué a módulos comunes como función
   nueva. Registrar el mapeo notebook→módulo en el docstring de la etapa.
2. Implementar `run_stage01(run_id=None, ...)` con el contrato estándar del
   repo (resolución de run por `MUSE_RUN_ID`/`active_run.txt`, paths por
   `musepipe/paths.py`, QC JSON, productos con nombres idénticos a los
   actuales).
3. Validación de equivalencia sobre copia aislada:
   - clonar config y entradas mínimas a `runs/ROXs12b_stage01_validation/`;
   - correr notebook original (o usar el producto baseline existente) y
     `run_stage01()`;
   - comparar FITS: mismas shapes, mismos headers relevantes, y datos
     idénticos según §4.
4. **Gate B1a**: tabla de comparación en el reporte. No empezar B1b sin esto.

**Punto de atención**: para ROXs12b hay un solo cubo, así que la rama de
alineación multi-cubo del notebook casi no se ejercita. La equivalencia sobre
ROXs12b NO valida esa rama: se cubre con el test sintético multi-cubo de §8
(no con LkCa_15, que pesa demasiado para CI).

## 4. Tolerancia de equivalencia

- Datos: idénticos bit a bit si el entorno es el mismo; si hay diferencias,
  solo se aceptan a nivel de ruido de redondeo (`np.allclose` con
  `rtol=1e-10, atol=0`) y se documenta la causa (orden de operaciones BLAS,
  etc.). Diferencias mayores = fallo de migración, no "tolerancia a negociar".
- Headers/QC: mismas claves y valores salvo timestamps y versiones.

## 5. B1b — Extensiones (solo tras gate B1a)

### 5.1 Propagación de STAT

- Cargar STAT del cubo de entrada (si existe; si no, `stat: "unavailable"` en
  QC y la etapa sigue — los cubos históricos pueden no traerla utilizable,
  ver A4/M5).
- Propagar por las MISMAS operaciones que DATA, con la matemática correcta de
  varianzas: crop = slicing directo; shift entero = slicing; shift fraccional
  por interpolación = pesos del kernel **al cuadrado** sobre STAT. Implementar
  la propagación que corresponda al `spatial_shift_mode` realmente usado; si
  el modo es fraccional, registrar el kernel exacto.
- Al apilar N cubos: suma/promedio en cuadratura consistente con cómo se
  combinan los DATA (mismos pesos).

### 5.2 Registro de shifts

- QC nuevo: shift aplicado por cubo/exposición (dy, dx, y dz si aplica), método
  de centrado usado y si actuó el fallback de `maoppy`.

### 5.3 Factor de covarianza espacial

- La interpolación correlaciona píxeles vecinos y hace que la varianza por
  apertura se subestime. Medirlo UNA vez empíricamente: sobre regiones vacías
  del producto final, razón `var(aperture 3×3) / (9 · var(spaxel))`; guardar
  `covariance_factor_box3` en QC. Es el insumo que A4/M5 y D2 cruzarán.

### 5.4 Entry points

- `entry_point: raw|eso_cube`: la entrada es el cubo del bloque A (campo
  completo); el centro del crop se define por el centroide de la primaria
  (medido, no hardcodeado) y `crop_npix` de config.
- `entry_point: cropped_cube`: comportamiento histórico intacto (regresión de
  B1a lo garantiza).

## 6. Esquema de `stage01_qc.json` (añade claves; conserva las históricas)

```json
{
  "stage": "01_load_align_crop",
  "run_id": "...",
  "entry_point": "raw|eso_cube|cropped_cube",
  "inputs": [{"file": "...", "sha256": "...", "data_ext": 1, "stat_ext": "2|null"}],
  "shifts": [{"cube": 0, "dy": 0.0, "dx": 0.0, "method": "maoppy_refined|peak", "fallback_used": false}],
  "crop": {"npix": 170, "center_yx": [0, 0], "center_source": "measured|config"},
  "stat": {"propagated": true, "interp_kernel": "...", "covariance_factor_box3": 0.0},
  "equivalence": {"baseline": "...", "max_abs_diff": 0.0, "verdict": "identical|allclose"},
  "open_issues": []
}
```

## 7. Verificaciones

- **V1 — Equivalencia B1a**: tabla de §4 en verde (es el gate, se re-adjunta).
- **V2 — STAT sin NaN nuevos**: máscara de NaN de STAT ⊆ máscara de NaN de
  DATA (más bordes); figura del mapa.
- **V3 — Consistencia STAT/empírico post-stack**: en regiones vacías del
  producto, razón var-empírica/STAT dentro de [0.5, 2.0] (el ajuste fino es de
  A4; aquí solo se detecta un error de propagación grosero, p. ej. olvidar el
  cuadrado del kernel, que típicamente da factores ~2× en un sentido).
- **V4 — Alineación**: residuo de centroide entre cubos alineados < 0.1 px
  (solo multi-cubo; en ROXs12b se verifica que el centrado del crop coincide
  con el centroide medido de la primaria a < 0.5 px).
- **V5 — Regresión del repo**: suite completa de tests verde antes y después.

## 8. Tests nuevos

```text
tests/test_stage01_equivalence.py   # contrato: shapes, claves QC, nombres de productos
tests/test_stage01_stat.py          # cubo sintético con var conocida: shift fraccional → STAT esperado analíticamente
tests/test_stage01_multicube.py     # 3 mini-cubos sintéticos desplazados → shifts recuperados y stack correcto
```

`test_stage01_stat` es el test de potencia: fabrica el caso donde olvidar el
kernel² produce un STAT erróneo en factor conocido y exige detectarlo.

## 9. Protocolo de parada

Preguntar cuando: la equivalencia B1a no dé dentro de §4 y la causa no sea
identificable; el notebook tenga ramas muertas/ambiguas cuya semántica no esté
clara (no adivinar: preguntar con el fragmento de código); STAT del cubo
histórico esté ausente o degenerada; `spatial_shift_mode` efectivo no coincida
con lo que el código hace realmente (discrepancia config-código = reportar).

## 10. Reporte final

Tabla de equivalencia B1a; verificaciones V1–V5; mapeo notebook→módulo;
factor de covarianza medido; checklist de límites; comandos de reproducción.
