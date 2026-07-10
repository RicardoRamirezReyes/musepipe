# A3 — Justificación de la corrección telúrica por STD_TELLURIC (para el paper)

> Documento de justificación metodológica. molecfit no convergió (problema de
> configuración conocido, ver §3); la corrección telúrica se aplicó con el espectro de
> transmisión de la estrella estándar (STD_TELLURIC) escalado por masa de aire. Cada
> cifra citada lleva al lado la ruta del QC de la que se tomó.
>
> Escrito: 2026-07-10. Fuente principal:
> `runs/ROXs12b_raw/stages/stage00t_qc.json`.

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
4. **Referencia a la práctica estándar.** El uso de STD_TELLURIC de la estrella estándar
   escalada por masa de aire es el enfoque de respaldo habitual en reducciones MUSE cuando
   molecfit no está disponible o no converge. [REF — completar con la cita de la práctica
   estándar MUSE, p. ej. el manual del pipeline MUSE / Weilbacher et al.]

## 5. Nota sobre el run realineado

El run científico principal (`ROXs12b_realigned`,
`/mnt/2TB/MUSE_work/ROXs12b_realigned_run`) reutiliza el **mismo método** STD_TELLURIC:
su árbol contiene `TELLURIC_TRANS.fits` y `cube_telcorr.fits`
(`/mnt/2TB/MUSE_work/ROXs12b_realigned/`), pero **no** emitió un `stage00t_qc.json`
separado. Por tanto, las cifras de verificación de §2 provienen del QC del run crudo
(`runs/ROXs12b_raw/stages/stage00t_qc.json`), que documenta la reducción telúrica de
referencia. [Pendiente: regenerar un QC telúrico para el run realineado si se quiere una
métrica pre/post propia; el método y la transmisión son los mismos.]
