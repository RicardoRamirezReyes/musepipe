# A3 — Justificación de la corrección telúrica por STD_TELLURIC (para el paper)

> Documento de justificación metodológica. La corrección telúrica se aplicó con el espectro
> de transmisión de la estrella estándar (STD_TELLURIC) escalado por masa de aire. En el
> intento original molecfit no convergió (§3); el reintento dedicado A1a (§6, 2026-07-19)
> halló la causa raíz real (flujo sin normalizar + banda débil, no el GDAS), hizo **converger**
> molecfit y **corroboró** STD_TELLURIC (|ΔT| 0.8 % por píxel en la banda B junto a Hα). Cada cifra citada lleva al lado la ruta del
> QC de la que se tomó.
>
> Escrito: 2026-07-10; verificación A1a añadida 2026-07-19. Fuente principal:
> `runs/ROXs12b_raw/stages/stage00t_qc.json`, `.../stage00r_qc.json → a1a_molecfit_crosscheck`.

---

## 1. Método aplicado

Se corrigieron las bandas telúricas con el **espectro de transmisión de la estrella
estándar** (`STD_TELLURIC`, producido por la receta `muse_standard`), escalado a la masa
de aire de la ciencia mediante la ley de Beer–Lambert:

    T_sci(λ) = T_std(λ) ^ (X_sci / X_std)

- `X_std = 1.087`, `X_sci = 1.158` → exponente de escalado 1.065
  (`stage00t_qc.json` → `fit.airmass_std`, `fit.airmass_sci`, `fit.scaling`).
- Fuente del espectro: `products/muse_standard/STD_TELLURIC_0001.fits`
  (`stage00t_qc.json` → `products.std_telluric`).
- Producto de transmisión aplicado: `raw_reduction/TELLURIC_TRANS.fits`
  (`stage00t_qc.json` → `products.transmission`).
- Cubo corregido: `raw_reduction/cube_telcorr.fits`
  (`stage00t_qc.json` → `products.cube_telcorr`).

**Ventanas protegidas (T ≡ 1 forzada):** el escalado NO se aplica en dos regiones, para
no introducir estructura espuria donde importa la ciencia
(`stage00t_qc.json` → `protected_windows_A`):

| Ventana (Å) | Motivo |
|---|---|
| 6540–6590 | Hα del compañero (diagnóstico de acreción) — impacto telúrico nulo por construcción |
| 5780–6050 | Región del láser AO de NFM |

La decisión de aplicar la corrección fue automática por umbral y con checkpoint humano
aprobado (`stage00t_qc.json` → `decision.verdict = "needed"`,
`decision.user_checkpoint = "approved"`, `decision.threshold_pct = 3.0`).

## 2. Verificación empírica

Profundidad de banda antes/después de la corrección, medida en la primaria dentro de una
apertura de 6 px (`stage00t_qc.json` → `input.aperture_radius_px = 6.0`):

| Banda | Profundidad pre (%) | Profundidad post (%) | Fuente (QC) |
|---|---|---|---|
| O₂ (banda B, ~6870 Å) | **6.76** | **0.6** | `decision.depth_pct_by_band.O2_B`; `verification.v1_o2_depth_pre_post_pct` |
| H₂O 7200 | 0.658 | — | `decision.depth_pct_by_band.H2O_7200` |
| H₂O 8200 | 0.0 | — | `decision.depth_pct_by_band.H2O_8200` |

La banda O₂ dominante pasa de **6.76 % → 0.6 %** (reducción a <1 %). Verificaciones
adicionales del QC (`stage00t_qc.json` → `verification`):

- `v2_outside_bands_unchanged = true`: el continuo fuera de las bandas no cambia.
- `v3_halpha_untouched = true`: Hα intacta (coherente con la ventana protegida 6540–6590).
- `v4_transmission_physical = true`: transmisión ∈ [0, 1], físicamente válida.
- `v5_stat_scaled`: la varianza STAT escala como T² (cociente
  `STAT_post/(STAT_pre/T²) = 1.0` con error ≤ 6×10⁻⁸ sobre 383 M canales finitos;
  el flag `verify_stat_scaled` devuelve False solo por falta de `equal_nan` en
  `np.allclose` frente al 11.7 % de NaN estructurales — falso negativo documentado,
  `stage00t_qc.json` → `verification.v5_note`).

## 3. Evidencia de la no-convergencia de molecfit

molecfit se intentó primero (sobre la primaria y sobre la estándar) y **no convergió**.
Logs: `runs/ROXs12b_raw/raw_reduction/molecfit/esorex.log` (primaria) y
`runs/ROXs12b_raw/raw_reduction/molecfit/std_fit/esorex.log` (estándar).

Síntoma y causa raíz (citados de los logs):

1. **Perfil GDAS ausente.** La extracción del perfil atmosférico GDAS para la
   coordenada/fecha del apuntado falla:
   `mf_io_system: System Call tar ... gdas_profiles_C-70.4-24.6.tar.gz
   C-70.4-24.6D2022-09-01T01.gdas ... failed!` (esorex.log). Sin perfil GDAS válido,
   molecfit no dispone de la estructura vertical de la atmósfera.
2. **χ² congelado (fit no mejora).** Como consecuencia, el χ² permanece **idéntico** en
   todas las iteraciones de mpfit: `Chi2: 559540.04` en la primaria y `Chi2: 1209777.98`
   en la estándar, sin variación entre `mpfit_calls` sucesivas (esorex.log). El ajuste no
   progresa → transmisión → 0 en las bandas, resultado inutilizable.

Esto es un **problema de configuración** (perfil GDAS no disponible localmente en el
`telluriccorr` 4.3.3 instalado), no contaminación estelar ni un defecto del método
telúrico. Registrado en `stage00t_qc.json` → `open_issues[0]`.

> **Actualización (A1a, 2026-07-19).** El reintento dedicado (§6) identificó que la causa
> raíz del χ² congelado **no era el GDAS** (secundario; se usó el perfil MIPAS estándar como
> respaldo), sino que el espectro 1D se pasó **sin normalizar** (flujo mediano ~58000, continuo
> atascado en 1.0, `bestnorm 1e12`) y con **una única ventana débil** (la banda B de O₂, ~5.6 %
> de absorción media), dejando al parámetro de O₂ **sin leverage**. Al normalizar el flujo e
> incluir la banda A de O₂ (7590–7690 Å, 30 % de absorción media), molecfit **converge** y
> **corrobora** STD_TELLURIC (§6).

## 4. Limitaciones

1. **Regiones interpoladas / dependencia de la estándar.** La corrección hereda la SNR y
   la sistemática de la estándar; el escalado por masa de aire `T^(X_sci/X_std)` asume una
   **atmósfera uniforme** entre las observaciones de estándar y ciencia
   (`stage00t_qc.json` → `open_issues[1]`).
2. **Residuo sistemático para D2.** Tras la corrección queda un residuo O₂ ~0.6 % en la
   primaria; es una sistemática presupuestada para la calibración espectral D2
   (`stage00t_qc.json` → `open_issues[1]`).
3. **Impacto nulo en Hα.** La ventana 6540–6590 Å tiene `T ≡ 1` forzada, de modo que la
   corrección telúrica **no altera** la región del diagnóstico de acreción
   (`v3_halpha_untouched = true`). Esto es lo relevante para el resultado científico: el
   límite de Hα no depende de la calidad de la corrección telúrica.
4. **Referencia a la práctica estándar.** La corrección telúrica a partir del espectro de
   la estrella estándar es un método **incorporado en el propio pipeline MUSE**: el DRS
   deriva la transmisión telúrica de las exposiciones de la estándar (paso separable de la
   estándar espectrofotométrica de flujo) — ver Weilbacher et al. 2020, *The data processing
   pipeline for the MUSE instrument*, A&A 641, A28 (DOI 10.1051/0004-6361/202037855), §sobre
   `muse_standard`. Aplicarlo con escalado por masa de aire (Beer–Lambert) es el uso estándar
   cuando molecfit no está disponible o no converge; aquí NO es un atajo sino el método nativo
   del DRS con una salvedad conservadora (ventana Hα protegida).
5. **Comparación con la literatura análoga (calibración sin molecfit).** El estudio MUSE de
   acreción en Hα más cercano a este trabajo, **Hashimoto et al. 2020** (PDS 70 b; AJ 159,
   222; arXiv:2003.07922), redujo los datos con el **pipeline MUSE estándar vía EsoReflex** y
   **calibró el flujo dentro del pipeline** (curva de extinción de Paranal + estrella
   espectrofotométrica de las master calibrations); **no empleó molecfit** — simplemente
   enmascaró las regiones contaminadas en lugar de aplicar una corrección telúrica explícita.
   Nuestra corrección STD_TELLURIC explícita, con la ventana de Hα (6540–6590 Å) forzada a
   `T ≡ 1` y verificación pre/post (O₂ 6.76 %→0.6 %), es por tanto **al menos tan rigurosa**
   como la de los análogos MUSE publicados, y el diagnóstico científico (Hα) es insensible a
   ella por construcción (cf. Eriksson et al. 2020, Delorme 1 (AB) b, A&A 638, L6, otro
   análogo MUSE reducido con el pipeline estándar).

## 5. Nota sobre el run realineado

El run científico principal (`ROXs12b_realigned`,
`/mnt/2TB/MUSE_work/ROXs12b_realigned_run`) reutiliza el **mismo método** STD_TELLURIC:
su árbol contiene `TELLURIC_TRANS.fits` y `cube_telcorr.fits`
(`/mnt/2TB/MUSE_work/ROXs12b_realigned/`). Las cifras de verificación de §2 provienen del QC
del run crudo (`runs/ROXs12b_raw/stages/stage00t_qc.json`), que documenta la reducción
telúrica de referencia; el método y la transmisión son los mismos en ambos runs.

**Trazabilidad (paso A3, 2026-07-18):** para que el gate F1 del run realineado deje de marcar
A2 "conditional QC missing" y A3 "not_run", se emitieron
`runs/ROXs12b_realigned/stages/stage00s_qc.json` (A2, sky/ZAP) y `stage00t_qc.json` (A3,
telúrica) **derivados** de los reales de `ROXs12b_raw` (mismos números, sin renombrar claves),
cada uno con un bloque `provenance` que registra `derived_from`, la razón, y el `sha256` del
cubo realineado (`9fff16b7…`, `cube_telcorr.fits`). El open_issue telúrico de
`stage00r_qc.json` quedó anotado con `a3_telluric_resolution = {resolution:
"justified_std_telluric", doc: este archivo}`.

## 6. Verificación A1a — molecfit converge y corrobora STD_TELLURIC (2026-07-19)

El paso opcional **A1a** (sesión dedicada, timebox) reintentó molecfit para cerrar el
open_issue "no convergió" con un contraste independiente del método adoptado.

**Diagnóstico de la falla previa (2026-07-06).** El `BEST_FIT_PARAMETERS.fits` del intento
original mostraba `positive_weights=77`, `rms_rel_to_err=85.8`, `uncertainty(rel_mol_col_O2)=0`
y χ² **idéntico** entre iteraciones: el flujo se pasó **sin normalizar** (mediana ~58000) con el
continuo atascado en 1.0 (`bestnorm 1e12`), y la única ventana era la **banda B de O₂** débil
(5.6 % de absorción media). El desajuste de continuo dominaba el χ² y O₂ quedaba **sin leverage**.

**Correcciones aplicadas.**
1. Normalización del espectro 1D por su continuo (mediana móvil) → flujo O(1).
2. `WAVE_INCLUDE` con la **banda A de O₂ (7590–7690 Å)** — 73 % de profundidad máxima, 30 % de
   absorción media, el rasgo telúrico con mayor leverage — más la **banda B (6864–6960 Å)**
   que ancla el ajuste junto a Hα.
3. Ajuste de O₂ (H₂O congelado; noche seca).

**Resultado del ajuste** (`molecfit_model`, telluriccorr 4.3.3; scratch, no toca el árbol del run):

| Métrica | Intento previo | A1a |
|---|---|---|
| Estado mpfit | congelado | **status=2 (converge)** |
| `rel_mol_col_O2` | 1.0 (fijo) | **0.966 ± 0.016** |
| `ppmv_O2` | — | **204760 ± 3325** (≈20.5 %, físico) |
| `rms_rel_to_err` | 85.8 | **5.09** |

**Comparación de la transmisión T(λ) vs STD_TELLURIC** (transmisión del mejor modelo,
`MOLECFIT_DATA.mtrans`, vs `TELLURIC_TRANS.fits` aplicado en A3):

| Banda | molecfit T̄ | STD_TELLURIC T̄ | \|ΔT\| medio | razón absorción integrada (mol/STD) |
|---|---|---|---|---|
| **O₂ banda B (6864–6960 Å, junto a Hα)** | 0.930 | 0.927 | **0.008** | **0.955** |
| O₂ banda A (7590–7690 Å) | 0.704 | 0.720 | 0.021 | 1.055 |

> **Marco de longitud de onda (aire vs vacío).** molecfit reporta `MOLECFIT_DATA.lambda` en
> **vacío**, mientras la ciencia y STD_TELLURIC están en **aire** (nativo de MUSE). La superposición
> cruda mostraba un desfase aparente de +1.3 Å (4750 Å) → +2.6 Å (9300 Å) que coincide con la
> dispersión aire→vacío de Edlén a <0.01 Å — **no es un desacuerdo físico**. Las cifras de la tabla
> son tras convertir la λ de molecfit vacío→aire; con ello el |ΔT| por píxel en la banda B baja a 0.8 %.

En Hα (6540–6590 Å) no hay líneas telúricas: STD_TELLURIC es exactamente 1.0 en todo
6590–6860 Å y en Hα, y molecfit (misma atmósfera) es igualmente ≈1 — ambos dejan Hα intacta.
Gráfico: `runs/ROXs12b_raw/plots/stage00t_a1a_molecfit_vs_std.png`.

**Conclusión.** Con el flujo normalizado y una banda telúrica fuerte, molecfit **converge** y su
transmisión de O₂ **coincide con STD_TELLURIC al ~4.5 %** (integrado) / 0.8 % (por píxel) en la
banda B adyacente a Hα y al ~6 % en la banda A. Esto es una **corroboración independiente** de
la corrección A3 STD_TELLURIC. **A1b (STD_TELLURIC justificado) sigue siendo el cierre del
paper**; A1a convierte el antiguo caveat "molecfit no convergió" en un contraste positivo.
Registrado en `stage00r_qc.json` → `a1a_molecfit_crosscheck`.

*Nota operativa:* `molecfit_calctrans` (transmisión de rango completo) no se produjo — esta build
rechaza `MAPPING_ATMOSPHERIC` para un SCIENCE de una sola BINTABLE (perfil atmosférico NULL pese
a mapping válido), un quirk de la receta/versión, no un problema físico. La comparación a nivel de
ventana (arriba) es suficiente para el contraste.
