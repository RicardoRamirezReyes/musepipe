# Especificación C4 v2 · el término de fondo del ajuste

Fecha: 2026-10-24. Revisión formal de `docs/spec_C4_codex_psf_fitting.md`, que sigue siendo la referencia para
todo lo demás: v2 cambia **solo el término de fondo** del modelo lineal (§0 y §3.1 de v1). Lo pidió el límite
duro 1 de v1 («términos extra de fondo → preguntar antes»); la aprobación y la evidencia están en
`docs/2026-10-24_c4_planos_desacoplados_decision.md`.

## 0. Motivo

v1 ajusta, por canal, `a·P(pos_star) + b·P(pos_comp) + c₀ + c₁·y + c₂·x` sobre la **unión** del disco de la
estrella (r 20 px) y el del compañero (r 12 px), a 47–71 px de distancia. Lo que el modelo de PSF deja en el
núcleo de la estrella, los gradientes lo recogen dentro de su disco y lo extrapolan al compañero con mucho brazo
de palanca. En los 33 controles al mismo radio eso es un dipolo `a + b cos θ + c sin θ` (R² hasta 0.75 en el
azul) y el continuo azul de psffit en el compañero es el valor de ese dipolo en su acimut. La mediana de los
controles no lo quita. La línea no la toca (es plana en 6480–6650 Å y el continuo local la quita).

## 1. El término de fondo

`x03_background` (config de C4) elige uno de dos modelos:

| valor | columnas | modelo |
|---|---|---|
| `shared` (v1, **por defecto**) | 5 | `a·P_s + b·P_c + c₀ + c₁·y + c₂·x` en toda la región |
| `decoupled_c0` | 6 | `a·P_s + b·P_c + (c₀ + c₁·y + c₂·x)·D_s + d₀·D_c` |

`D_s` y `D_c` son las indicatrices de los discos de la estrella y del compañero (los mismos radios que la región
de ajuste). En `decoupled_c0` el plano solo vive en el disco de la estrella y el compañero solo tiene su
constante: el residuo del núcleo no tiene camino hasta el compañero. `y, x` y su escala son las de v1.

- El valor por defecto es `shared` para que los runs anteriores se reproduzcan bit a bit. Los runs del paper
  declaran `decoupled_c0` en su config.
- Ningún otro valor se acepta: `ValueError`.
- Lo heredan del config resuelto de C4 todos los que ajustan psffit: E4 (`stage_h04_extractors`), E4b
  (`x06b_psffit_background`) y los espectros por observación (`perobs_spectra_psffit_background`). Inyectar con un
  fondo y extraer el objeto con otro rompería el throughput.

## 2. Productos y QC

- `BKGMODE` del header: `psffit_plane` (shared) o `psffit_plane_decoupled_c0`.
- `spec_psffit_qc.json`, `fit_region`: `background` y `n_parameters` (5 o 6). La covarianza por canal es
  `n_parameters × n_parameters`; los consumidores solo leen las columnas 0 (estrella) y 1 (compañero).
- `rho_bc` pasa a ser la máxima |ρ| del compañero con **todas** las columnas de fondo (con 5 columnas es lo
  mismo que en v1).

## 3. Lo que no cambia

Posiciones fijas (la posición cromática de la primaria se midió el 2026-10-24 y no apaga el dipolo por sí sola),
región de ajuste, pesos 1/V, errores, apcorr, controles, V1–V4. Las verificaciones de v1 se aplican igual.

## 4. Tests

`tests/test_psffit_decoupled_c0.py`: el diseño por defecto es el de v1 bit a bit; las columnas de
`decoupled_c0`; un valor desconocido se rechaza; `decoupled_c0` recupera su propia escena exacta; y el mecanismo:
con la estrella 0.3 px fuera de donde el ajuste la pone, `shared` sesga al compañero en decenas de por ciento con
signo opuesto a cada lado de la estrella y `decoupled_c0` lo deja por debajo del 1 %.
