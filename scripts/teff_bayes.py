#!/usr/bin/env python
"""T_eff de los compañeros por muestreo anidado (dynesty) con BT-Settl o DRIFT-PHOENIX sobre MUSE + J/H/K.

Solo lectura y fuera de la cadena (pedido del autor 2026-10-14: «adelante con la MCMC, instala dynesty», y la
segunda rejilla, «adelante con DRIFT-PHOENIX»). Da la T_eff, el tipo equivalente y el margen sobre el suelo
cromosférico del apéndice B del paper (`docs/2026-10-14_teff_muestreo_anidado.md`). Reutiliza los segmentos, las
escalas con prior y la preparación de modelos de `scripts/ajuste_conjunto.py` (R = 500, máscaras de telúrico y de
H2O), pero en vez de una rejilla con polinomios usa un modelo FÍSICO con flujo absoluto:

    F_k(λ) = s_k · (R/d)² · [F_mod(λ; T_eff, log g) + r_K · F_mod(2.2 µm) · B_λ(T_d)/B_2.2(T_d)] · 10^(−0.4 A_λ/A_V · A_V)

- F_mod es flujo de superficie (comprobado: BT-Settl ∫ 0.4–2.55 µm = 0.83 σT⁴ a 2500 K; DRIFT-PHOENIX sobre su
  rango completo, 0.97–1.00 σT⁴), interpolado bilineal en log F sobre la rejilla (`REJILLAS`). El nodo de
  DRIFT-PHOENIX que SVO no tiene (2500 K, log g 3.0) se rellena con la media en log F de sus vecinos en T_eff.
- s_k: escala por segmento con el prior de `ajuste_conjunto` (N(1, 0.05) en las bandas ancladas a la
  fotometría, N(1, 0.20) en MUSE, log-uniforme 0.2–5 en las no ancladas; `--anclas-ir` cambia cuáles van ancladas).
- Error: σ² = e² + (f_b · F)², con un f_b libre por banda (vis, J, H, K): el desajuste del modelo, que con
  S/N 50–170 en el IR domina sobre el ruido.

Modos de `ajusta`:
- `libre`: T_eff, log g y R libres (lo que dice el espectro solo). A_V con el prior de la primaria (`--av
  primaria`), libre en 0–8 (`--av libre`) o fija (`--av-fija`, el perfil en A_V).
- `evo`: (edad, L_bol) → tracks (BHAC15 o ATMO2020) → T_eff, R, log g, M. La edad con el prior de la
  literatura (`g3_age_myr`, log-normal); L_bol plana en −4.5…−1.5, de modo que la fija el flujo absoluto del
  espectro y no la L_bol publicada.
- `evo_solo` (sin espectro): la L_bol publicada (`targets` → `fisica`) y el prior de edad por los tracks:
  la T_eff «evolutiva», la referencia contra la que se mide lo que aporta el espectro.

Pruebas: `--inyecta-bt T LOGG AV R_RJ` sustituye los datos por un modelo de la rejilla con la S/N real;
`--plantilla NOMBRE` por una plantilla empírica de tipo conocido (normalizada con UN solo factor, así que
conserva sus colores) enrojecida con `--av-inyectada` (por defecto la A_V de la primaria): la calibración que
traduce la T_eff de cada rejilla a un tipo equivalente.

`--datos muse` o `--datos ir` ajusta solo el espectro de MUSE o solo los J/H/K (pedido del autor 2026-10-16, para
ver lo que aporta cada uno): los mismos píxeles, escalas y priores que el ajuste conjunto, sin los segmentos del
otro brazo; sin ningún segmento IR, r_K sale de los parámetros (no tendría datos). Esas corridas no entran en la
tabla del paper (`_de_la_tabla`).

`--rejilla` admite, además de las dos del paper, las del mismo pedido («probar con más modelos»): AMES-Dusty,
AMES-Cond y BT-Dusty (PHOENIX con otro trato del polvo), Sonora Diamondback por fsed (otro código, con nubes, hasta
2400 K; desde 2000 K o, `diamondback1600_<fsed>`, desde 1600 K) y SPHINX (otro código, sin nubes). SPHINX está
muestreado a λ/Δλ = 250, así que su ajuste va en el modo R = 250 (`--r250`, automático para ella): los píxeles de
los datos se agrupan de a dos y el modelo se usa a su resolución. `--r250` con una rejilla de resolución alta es
el control que separa el efecto de la resolución del de la rejilla. Estas corridas tampoco entran en la tabla de
las cadenas: van a la de rejillas (`rejillas`, Tabla B.7 del paper, pedida el 2026-10-18).

ATMO 2020 (`atmo_ceq`, 2000–3000 K; `atmo_ceq1500`, desde 1500 K; indicación 1 de Claudio, 2026-10-26) entra solo
como PRUEBA (un párrafo del apéndice B, no la Tabla B.7): sus autores la dan por válida solo bajo ~2000 K. La
rejilla se convierte con `scripts/convierte_atmo.py` a la ruta que se quiera (`--dir-rejilla`). Está muestreada en
número de onda (10 cm⁻¹ por punto, media de banda), así que el degradado a R = 500 resta en cuadratura la
resolución que el muestreo ya trae (`--degrada cuadratura`, por defecto; `ingenua` es el camino de las demás
rejillas). `btsettl_bin` (BT-Settl promediado en las bandas de ATMO) es el control del muestreo. Las cifras del
párrafo las da `scripts/paper_cifras_atmo.py`.

`--mascara-muse` (pruebas exploratorias del 2026-10-28, pedido del autor: «las bandas telúricas de O2, si están
corregidas, deberían poder agregarse») elige las máscaras telúricas del espectro MUSE (`MASCARAS_MUSE`): `paper`
(por defecto, las de las cadenas del paper: las de G3, O2 B y O2 A, más las de H2O de
`tipo_ir_plantillas.H2O_MUSE_A`), `g3` (sin H2O), `sin_o2` (con H2O, sin O2), `sin_o2b` (con H2O y O2 A, sin O2 B)
o `sin_telurico` (sin ninguna). Las
ventanas de las líneas y los canales malos se quedan siempre; el cambio de O2 es en memoria. Esas cadenas llevan
`mascara-<variante>` en la etiqueta y `mascara_muse` en el json, y no entran en `tabla` ni en `rejillas`. El residuo
telúrico en O2 lo mide `scripts/g3_residuo_telurico_fino.py`.

`--ventana-lineas KMS` y `--ventana-halpha KMS` (pregunta del autor del 2026-10-28: «¿qué número parecería óptimo
para extraer la mayor información no sesgada?») cambian, en memoria, la ventana de ±300 km/s que G3 pone alrededor
de las 24 líneas de G2 (`g3_fit_masks.line_window_kms`): la primera para todas, la segunda solo para Hα
(`tipo_ir_plantillas.ventanas_de_lineas`). Sin ellas, la de siempre. Con ellas, la etiqueta lleva `ventana-<KMS>` y
`ventanaha-<KMS>` antes de la semilla, el json `ventana_lineas_kms` / `ventana_halpha_kms`, y la cadena no entra en
`tabla` ni en `rejillas`. Qué anchura deja fuera el sesgo de las alas de una línea lo mide
`scripts/ventanas_lineas.py`.

Las pruebas del disco (apéndice D.11 del paper) y del D.5 (2026-10-29): `--rk-fijo RK` fija r_K (0 es el modelo sin
disco, cuya ln Z se compara con la de r_K libre); `--td-libre TMIN TMAX` deja libre la T del polvo (un parámetro
`td` detrás de `rk`, uniforme) en vez de 1500 K, como el retrieval de Grasser et al. 2026; `--inyecta-polvo RK TD`
suma un exceso de polvo conocido al modelo de `--inyecta-bt` (¿separa el ajuste el velo de la A_V?); y `--quita-muse
extremos` quita del espectro MUSE los bins de 6300–6900 y 9000–9350 Å (`INTERVALOS_MUSE`), los dos intervalos que
concentran el χ² de MUSE (D.5; `azul` y `rojo` quitan solo uno de los dos). Sin ellas, el modelo, la etiqueta y el json son los de siempre; con ellas, la etiqueta
lleva `rk<RK>`, `td<TMIN>-<TMAX>`, `quita-extremos` o `polvo-<RK>-<TD>` antes de la semilla, y el json `rk_fijo`,
`td_rango`, `quita_muse` o `inyecta_polvo`. Las combinaciones sin sentido fallan antes de leer nada (`valida_disco`).

`--muse-nativo` (pedido del autor del 2026-10-30, para comparar con las cadenas de solo MUSE a R = 500) ajusta el
espectro MUSE a su resolución: canal a canal (1.25 Å), con la σ de cada canal × √(n/n_eff), como el ajuste nativo de
G3, y el modelo convolucionado con la LSF de MUSE del run (`h01_lsf_fwhm_A`) y promediado en cada canal, como en G3
(`modelos_base_muse_nativo`). Sin degradar, las máscaras no crecen la FWHM del núcleo de R = 500, así que entran
algunos canales más cerca de las zonas enmascaradas. El IR sigue a R = 500. Solo vale con una rejilla de paso fino
en MUSE (`MUESTREO_MAX_NATIVO`: BT-Settl, BT-Dusty, Sonora Diamondback), sin `--r250` y sin `--plantilla`
(`valida_muse_nativo`). La etiqueta lleva `muse-nativo` y el json `muse_nativo`; no entra en las tablas.
Con `--ens-vr` (además de `--muse-nativo`; «añade ensanchamiento y velocidad libres», 2026-10-30) el modelo de MUSE
tiene dos parámetros más, uniformes: una FWHM gaussiana extra `ens_A` (0–`ENS_MAX_A` Å, en cuadratura con la LSF) y
una velocidad radial `vr_kms` (±`VR_MAX_KMS`). Se rehace en cada llamada desde el espectro fino de 0.2 Å ya a la LSF:
interpolado en log F, ensanchado por FFT, desplazado por Doppler y promediado en cada canal (`espectro_fino`,
`ensancha_y_desplaza`). La etiqueta lleva `ens-vr` y el json `ens_vr` (los dos topes).

`--ruido-de SLUG` (con `--plantilla` o `--inyecta-bt`; pedido del autor del 2026-10-30, «prepara el lote de plantillas
ATMO») separa, en la prueba de estabilidad de la calibración con plantillas, la cobertura del ruido: los datos de prueba
llevan la cobertura de este objeto (sus segmentos y píxeles, sus escalas con prior y la A_V de su primaria) y la S/N
por píxel del otro (`snr_cruzada`: la del mismo instrumento y banda o, si el otro no lo tiene, la mediana de los
suyos en esa banda). La etiqueta lleva `ruido-<SLUG>` y el json `ruido_de`; no entra en las tablas.

`--anclas-ir ninguna|K|JHK` (pedido del autor del 2026-10-30, «prepara la opción de soltar anclas y el lote») elige en
qué bandas del IR la escala del segmento va anclada a la fotometría, N(1, `SIGMA_IR`); en las demás, libre. Sin ella,
las que declara `targets/<slug>.json` (`ir_spectra.colour_excess.bands`: en ROXs 12 b solo K, porque su J y su H van
escaladas a la fotometría de la terciaria; en ROXs 42B b, J, H y K). Separa, en la misma prueba, qué segmentos lleva
cada compañero de qué escalas lleva ancladas. Solo cambia el prior de las escalas: los datos (también los de prueba,
que se escalan con el segmento anclado de siempre) son los mismos. Sobre los datos reales solo se puede soltar un
ancla; anclar una banda que el objeto no declara pide datos de prueba, cuyas escalas valen 1 por construcción
(`anclas_de`). La etiqueta lleva `anclas-<BANDAS>` y el json `anclas_ir`; no entra en las tablas.

`--quita-ir SEG [SEG ...]` (pedido del autor del 2026-10-31, «prepara el lote quitando NIFS H y SINFONI K») saca de la
verosimilitud esos segmentos del IR, por su nombre (`NIFS_H`, `SINFONI_K`…): sus píxeles, su escala y, si era el
único de su banda, el f_b de esa banda. En la misma prueba, visto que no son las anclas, dice cuál de los segmentos
que lleva un compañero y no el otro mueve la T_eff. Todo lo demás se arma con todos los segmentos (los datos de
prueba, con su escalado y su ruido, las anclas y la referencia del polvo a 2.2 µm), así que los que quedan llevan los
mismos datos que sin la opción (`quitados_de`). MUSE no se quita así (para eso, `--datos ir`), ni todo el IR
(`--datos muse`). La etiqueta lleva `quitair-<SEGS>` (en orden alfabético) y el json `quita_ir`; no entra en las
tablas.

`--pendiente-ir ANCHO` (pedido del autor del 2026-11-01, «prepara la opción (b) de pendiente libre y el lote») le da a
cada segmento del IR una inclinación libre: su modelo se multiplica por 1 + p·(λ − λ_c)/(0.1 µm), con λ_c el centro
de los píxeles del segmento y p uniforme en ±ANCHO (un parámetro `p_<SEG>` por segmento, detrás de las escalas).
Los espectros del IR están escalados a la fotometría solo en el nivel de cada banda, y dos de la misma banda difieren
en inclinación un 7–11 % (en 42B b, OSIRIS H frente a NIFS H, −3.6 %/0.1 µm, sin cambio de forma): con la opción, el
ajuste usa la forma de cada segmento y no su inclinación. MUSE no lleva pendiente. `ANCHO` tiene que dejar el factor
positivo en todo el segmento (`valida_pendiente_ir` y el `Modelo`). La etiqueta lleva `pendir<ANCHO>` y el json
`pendiente_ir`; no entra en las tablas.

`--banda-una-vez` (pedido del autor del 2026-11-02, «prepara el lote pesando cada banda una vez») cuenta cada banda del
IR una vez: el término de la verosimilitud de cada segmento del IR se multiplica por 1/N, con N el número de segmentos
de su banda que entran en el ajuste. 12 b lleva dos J (NIFS y OSIRIS) y 42B b dos H y dos K, así que sin la opción esas
bandas pesan el doble; con ella, dos espectros de la misma banda pesan lo que uno, sin elegir cuál (`--quita-ir`
elige). MUSE pesa 1. Es una verosimilitud templada: su ln Z no se compara con el de una cadena sin la opción. La
etiqueta lleva `bandaunavez` y el json `banda_una_vez`; no entra en las tablas.

Qué cadenas entran en las tablas del paper (`_de_la_tabla`, `_de_rejillas`) lo decide una lista BLANCA
(`_es_variante`): ni las variantes de `VARIANTES_JSON` ni una cadena con una clave del json que este código no
conozca (`CLAVES_JSON`). Así una opción nueva no entra por olvido en la Tabla B.6, la B.7, `disco_cotas` o el
corner. `ajusta` se niega a sobrescribir una etiqueta que ya existe y guarda en el json su `procedencia` (versiones,
commit, muestreador, rejilla resuelta, argumentos). Las cadenas de prueba van en su propia carpeta, nunca dentro de
las del paper (`halpha_criterios` sortea con un solo generador en el orden de los ficheros).

El prior usa `ndtri` y `truncnorm._ppf` en vez de `norm.ppf` y `truncnorm.ppf` (`_ppf_normal_rapido`,
`_ppf_trunc_normal_rapido`), con todas las normales truncadas en una sola llamada (`prior_transform`): los mismos
números bit a bit. El prior costaba 550–1180 µs por llamada, 4–9 veces la verosimilitud; ahora, 91 µs. Una corrida
corta de dynesty con la misma semilla da las mismas muestras en un 69 % (12 b) y un 83 % (42B b) menos de tiempo
(2026-10-29, `~/muse_sesion_20261029/fase0/mide_prior.json`).

`comprueba` reconstruye el modelo de cada cadena `libre` en su mejor punto y dice si su `loglike` da el `logl_max`
de la cadena bit a bit (sale con 1 si alguna no).

`--project-root` (todas las órdenes) lee los runs y las bibliotecas de otro clon: un worktree no lleva `runs/`.

`resume` lee los json/npz de uno o varios directorios de `ajusta` y da, por corrida, la T_eff y el margen
coherente sobre el suelo cromosférico de Manara et al. (2013, Ec. 2): por muestra, log(L_acc/L_bol) − suelo(T_eff),
con la L_acc del paper reescalada a la A_V de la muestra y la L_bol de la muestra (4πR²σT⁴; en `evo_solo`, la A_V
adoptada y la L_bol publicada). El suelo se toma sin su incertidumbre: sus autores no publican la covarianza
de los coeficientes.

uso: python scripts/teff_bayes.py ajusta SLUG [--rejilla btsettl|drift|ames_dusty|...] [--modo libre|evo|evo_solo]
                                       [--datos todo|muse|ir] [--r250]
                                       [--tracks ATMO2020] [--av libre|primaria] [--av-fija AV] [--nlive 500]
                                       [--semilla 0] [--inyecta-bt ...] [--plantilla ...] --salida DIR
                                       [--dir-rejilla DIR] [--degrada cuadratura|ingenua] [--project-root RAIZ]
                                       [--mascara-muse paper|g3|sin_o2|sin_o2b|sin_telurico]
                                       [--ventana-lineas KMS] [--ventana-halpha KMS]
                                       [--rk-fijo RK] [--td-libre TMIN TMAX] [--quita-muse extremos|azul|rojo]
                                       [--inyecta-polvo RK TD] [--muse-nativo [--ens-vr]] [--ruido-de SLUG]
                                       [--anclas-ir ninguna|K|JHK] [--quita-ir SEG [SEG ...]]
     python scripts/teff_bayes.py comprueba DIR [DIR ...] [--dir-rejilla REJILLA=DIR] [--json SALIDA]
     python scripts/teff_bayes.py resume DIR [DIR ...] [--json SALIDA]
     python scripts/teff_bayes.py tabla DIR [DIR ...]      # cuerpo LaTeX de la tabla de las cadenas (ap. B)
     python scripts/teff_bayes.py rejillas DIR [DIR ...]   # cuerpo LaTeX de la tabla de rejillas (ap. B)

El corner plot de las mismas cadenas lo dibuja `scripts/paper_figures.py --figure teff_corner --teff-chains DIR`;
el de solo MUSE / solo IR, `--figure teff_corner_data`; los de todas las cadenas, `scripts/teff_suplemento.py`.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import sys
import time
import warnings
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from musepipe.constants import C_KMS, LSUN_ERG_S, PC_CM, RJUP_CM, RSUN_CM, SIGMA_SB_CGS, spt_code  # noqa: E402
from musepipe.models.extinction import CCMExtinction  # noqa: E402

#: rejillas de atmósferas: directorio de npz (formato de `musepipe.models.cache`) y ejes (inicio, paso, n).
#: DRIFT-PHOENIX es la segunda rejilla, para el sistemático de modelo; ATMO 2020 se descartó como segunda rejilla
#: porque sus autores la dan por válida solo bajo 2000 K (`/mnt/2TB/IR_Data/atmo2020/README.md`) y entra solo como
#: prueba (abajo, `atmo_ceq`). BT-Settl llega a log g 5.0 desde
#: 2026-10-17 (decisión del autor): con 2.5–4.5, la posterior de 12 b tocaba el techo de 4.5.
REJILLAS = {"btsettl": dict(dir=Path("/mnt/2TB/IR_Data/btsettl_cifist/npz"), teff=(2000.0, 100.0, 13),
                            logg=(2.5, 0.5, 6), nombre="BT-Settl CIFIST2011"),
            "drift": dict(dir=Path("/mnt/2TB/IR_Data/drift_phoenix/npz"), teff=(2000.0, 100.0, 11),
                          logg=(3.0, 0.5, 7), nombre="DRIFT-PHOENIX")}
#: más rejillas (pedido del autor 2026-10-16, reunión con Claudio: «probar con más modelos»). Las tres de SVO son
#: PHOENIX con otro trato del polvo: AMES-Dusty (en equilibrio, sin sedimentar), AMES-Cond (sin opacidad de polvo)
#: y BT-Dusty, que a [M/H] = 0 solo tiene log g < 4.5 desde 2600 K (sirve para 12 b, no para 42B b). Sonora
#: Diamondback es otro código, con nubes (una rejilla por fsed; «nc» sin nubes), y llega solo a 2400 K (42B b).
REJILLAS.update({
    "ames_dusty": dict(dir=Path("/mnt/2TB/IR_Data/ames_dusty/npz"), teff=(2000.0, 100.0, 11),
                       logg=(3.5, 0.5, 6), nombre="AMES-Dusty"),
    "ames_cond": dict(dir=Path("/mnt/2TB/IR_Data/ames_cond/npz"), teff=(2000.0, 100.0, 13),
                      logg=(2.5, 0.5, 8), nombre="AMES-Cond"),
    "bt_dusty": dict(dir=Path("/mnt/2TB/IR_Data/bt_dusty/npz"), teff=(2600.0, 100.0, 7),
                     logg=(3.0, 0.5, 6), nombre="BT-Dusty"),
    **{f"diamondback_{f}": dict(dir=Path(f"/mnt/2TB/IR_Data/sonora_diamondback/npz/{f}"), teff=(2000.0, 100.0, 5),
                                logg=(3.5, 0.5, 5), nombre=f"Sonora Diamondback ({'sin nubes' if f == 'nc' else 'f_sed = ' + f[1:]})")
       for f in ("f1", "f2", "f3", "f4", "f8", "nc")},
    # la misma desde 1600 K: con 2000–2400 K, 42B b quedaba contra el borde de 2000 K (f1–f3, 2026-10-17). En
    # `npz_1600/` los nodos nublados de 1600–1800 K entran con ∫F/σT⁴ hasta 1.17 (el exceso crece hacia log g alto;
    # en log g 3.5–4.0, ≤ 1.10); el resto son enlaces a `npz/`, que guarda solo los de ±8 %
    **{f"diamondback1600_{f}": dict(dir=Path(f"/mnt/2TB/IR_Data/sonora_diamondback/npz_1600/{f}"), teff=(1600.0, 100.0, 9),
                                    logg=(3.5, 0.5, 5), nombre=f"Sonora Diamondback 1600–2400 K ({'sin nubes' if f == 'nc' else 'f_sed = ' + f[1:]})")
       for f in ("f1", "f2", "f3", "f4", "f8", "nc")},
    # otro código, sin nubes, log g ≥ 4.0, [M/H] = 0 y C/O = 0.5; muestreado a λ/Δλ = 250: va en el modo R = 250
    "sphinx": dict(dir=Path("/mnt/2TB/IR_Data/sphinx/npz"), teff=(2000.0, 100.0, 13), logg=(4.0, 0.25, 7),
                   nombre="SPHINX (sin nubes)", r_nativa=250.0),
})
#: ATMO 2020 CEQ (Phillips et al. 2020; indicación 1 de Claudio, 2026-10-26), convertida por
#: `scripts/convierte_atmo.py`: 2000–3000 K × log g 2.5–5.5 (77 nodos, completa). Sus autores la dan por válida SOLO
#: bajo ~2000 K, así que es una prueba, no una rejilla de las tablas. Muestreo uniforme en número de onda, 10 cm⁻¹
#: por punto (media de banda): Δλ = λ²·10 cm⁻¹ (5.6 Å a 0.75 µm, 15.7 a 1.25, 27 a 1.65, 48.5 a 2.2); `nu_bin_cm`
#: activa `degrada_nativa`. Las NEQ (weak/strong) no entran: en el tar de los autores solo llegan a 1800 K. La
#: `atmo_ceq1500` es la misma desde 1500 K (112 nodos; `convierte_atmo.py atmo ... --tmin 1500`): con 2000–3000 K,
#: 42B b con el prior en A_V tocaba el borde de 2000 K. `btsettl_bin` es el control: BT-Settl promediado en esas
#: mismas bandas (`convierte_atmo.py btbin`). Los directorios no existen hasta que se convierten; `--dir-rejilla`
#: apunta a otro.
REJILLAS.update({
    "atmo_ceq": dict(dir=Path("/mnt/2TB/IR_Data/atmo2020/npz_ceq"), teff=(2000.0, 100.0, 11), logg=(2.5, 0.5, 7),
                     nombre="ATMO 2020 (CEQ)", nu_bin_cm=10.0),
    "atmo_ceq1500": dict(dir=Path("/mnt/2TB/IR_Data/atmo2020/npz_ceq1500"), teff=(1500.0, 100.0, 16),
                         logg=(2.5, 0.5, 7), nombre="ATMO 2020 (CEQ) 1500–3000 K", nu_bin_cm=10.0),
    "btsettl_bin": dict(dir=Path("/mnt/2TB/IR_Data/atmo2020/npz_btsettl_bin"), teff=(2000.0, 100.0, 13),
                        logg=(2.5, 0.5, 6), nombre="BT-Settl con el muestreo de ATMO", nu_bin_cm=10.0),
})
#: varianza (en unidades de W², W = ancho de banda en Å) que el muestreo de ATMO ya trae, tras la interpolación
#: lineal a 1 Å de `ajuste_conjunto.modelos_btsettl`: la caja de la banda (W²/12) más la interpolación lineal
#: (W²/6). Se resta en cuadratura de la gaussiana de R = 500 (`--degrada cuadratura`); `ingenua` trata la rejilla
#: como de resolución alta (el camino de las demás). El valor lo comprobó el agente `atmo` con `btsettl_bin`: con
#: c = 1/4, BT-Settl muestreado como ATMO queda a 0.4–0.6 % (MUSE), 0.1–0.7 % (J, H) y 1.0–1.4 % (K) del nativo.
C_VAR_MUESTREO = 0.25
#: cómo se degrada a R = 500 una rejilla muestreada en número de onda (`--degrada`)
DEGRADA = ("cuadratura", "ingenua")


def degrada_nativa(w, f, nu_bin_cm, c_var=C_VAR_MUESTREO):
    """Como `tipo_ir_plantillas.degrada` (gaussiana de FWHM λ/500, huecos a NaN donde pesan > 20 %), pero con la
    resolución que el muestreo de la rejilla ya trae restada en cuadratura: σ_k² = σ_500² − c·W², W = λ²·Δν.
    Donde σ_k² ≤ 0 (K, λ ≳ 2.0 µm con ATMO) no se convoluciona: el modelo queda MÁS BASTO que R = 500
    (R ≈ 425 a 2.0 µm, 385 a 2.2 y 355 a 2.4)."""
    from musepipe.models.prep import degrade_variable
    from scripts import tipo_ir_plantillas as tp
    ok = np.isfinite(f)
    g = np.interp(w, w[ok], f[ok])
    W = w ** 2 * nu_bin_cm * 1e-8
    s500 = w / tp.R_OBJ / 2.3548200450309493
    var = s500 ** 2 - c_var * W ** 2
    fwhm = np.where(var > 0, 2.3548200450309493 * np.sqrt(np.clip(var, 0, None)), 0.0)
    out = degrade_variable(w, g, fwhm)
    hueco = degrade_variable(w, (~ok).astype(float), w / tp.R_OBJ)
    out[hueco > 0.2] = np.nan
    return out


def modelos_base_nativa(pls, segs, nu_bin_cm, c_var=C_VAR_MUESTREO):
    """`tipo_ir_plantillas.modelos_base` con `degrada_nativa` en vez de `degrada`."""
    from scripts import tipo_ir_plantillas as tp
    cache = {}
    for i, t in enumerate(pls):
        por_brazo = {}
        for brazo in ("vis", "nir"):
            w, f = t[brazo]
            orden = np.argsort(w)
            w, f = w[orden], f[orden]
            por_brazo[brazo] = (w, degrada_nativa(w, f, nu_bin_cm, c_var))
        cache[i] = [tp.media_en_intervalos(*por_brazo[s["brazo"]], s["lo"], s["hi"]) for s in segs]
    return cache


def _r250(rejilla, forzado=False):
    """¿Va el ajuste en el modo R = 250? Si se pide, o si la rejilla no llega a R = 500."""
    return bool(forzado) or REJILLAS[rejilla].get("r_nativa", np.inf) < 500


#: `--muse-nativo` (pedido del autor 2026-10-30, «sí, prepárala»: la comparación con las cadenas de solo MUSE a
#: R = 500): el paso (Å) al que se rebina el brazo visible de los modelos antes de convolucionarlo con la LSF de MUSE
#: (2.28–2.30 Å, ~11 puntos por FWHM; la caja de 0.2 Å le suma un 0.2 % a la FWHM, en cuadratura)
PASO_NATIVO_A = 0.2
#: el paso máximo de los npz en MUSE, en fracción de la LSF, para que una rejilla valga con `--muse-nativo`. Pasan
#: BT-Settl (0.02 Å), BT-Dusty (0.05) y Sonora Diamondback (0.045); DRIFT-PHOENIX, AMES-Dusty y AMES-Cond (2.0) no:
#: su muestreo ya es casi tan basto como la LSF, y convolucionarlo no le devuelve lo que no tiene
MUESTREO_MAX_NATIVO = 0.2
#: `--ens-vr` (pedido del autor 2026-10-30, «añade ensanchamiento y velocidad libres»): los topes de los priores
#: uniformes de la FWHM gaussiana extra (Å, en cuadratura con la LSF; 0–`ENS_MAX_A`) y de la velocidad radial (km/s,
#: ±`VR_MAX_KMS`). Con la LSF fija, el χ² nativo bajaba al ensanchar el modelo hasta 3.0 Å sin llegar a un mínimo, y
#: ROXs 12 b pedía mover el modelo ≥ 0.5 Å al azul (`~/muse_sesion_20261030/muse_nativo/LEEME.md`).
ENS_MAX_A = 10.0
VR_MAX_KMS = 100.0
#: margen (Å) del espectro fino a cada lado de los canales de MUSE: cubre ±`VR_MAX_KMS` (3.1 Å a 9350 Å) y las colas
#: de la gaussiana más ancha (4σ = 17 Å con `ENS_MAX_A`), de modo que los bordes de la FFT caen fuera de los canales
MARGEN_FINO_A = 50.0


def valida_muse_nativo(muse_nativo, datos="todo", rejilla="btsettl", r250=False, plantilla=None, ens_vr=None):
    """Rechaza, antes de leer nada, lo que no tiene sentido con `--muse-nativo`: sin espectro MUSE (`datos='ir'`);
    el modo R = 250, que agrupa los canales; una rejilla más basta que R = 500 (SPHINX) o muestreada en número de
    onda (ATMO 2020, 5.6 Å por punto en MUSE); y una plantilla empírica como dato, que `_sinteticos` degrada a
    R = 500. Las rejillas de 2 Å las rechaza después `comprueba_muestreo_nativo`, con el paso medido en sus npz.
    `ens_vr` = (tope de la FWHM extra en Å, tope de |v| en km/s) pide `muse_nativo` y dos topes positivos y finitos;
    la FWHM, no más que lo que cubre `MARGEN_FINO_A`."""
    if ens_vr is not None:
        if not muse_nativo:
            raise ValueError("ens_vr (ensanchamiento y velocidad libres) va sobre el espectro nativo: pide muse_nativo")
        if len(ens_vr) != 2 or not all(math.isfinite(x) and x > 0 for x in ens_vr):
            raise ValueError(f"ens_vr={ens_vr!r}: deben ser dos topes > 0 (FWHM extra en Å, |v| en km/s)")
        if 4.0 * ens_vr[0] / 2.3548200450309493 + 9350.0 * ens_vr[1] / C_KMS > MARGEN_FINO_A:
            raise ValueError(f"ens_vr={ens_vr!r}: más ancho de lo que cubre el margen del espectro fino "
                             f"({MARGEN_FINO_A:g} Å)")
    if not muse_nativo:
        return
    if datos == "ir":
        raise ValueError("con datos='ir' no hay espectro MUSE que ajustar a su resolución (muse_nativo)")
    if _r250(rejilla, r250):
        raise ValueError(f"{rejilla}: muse_nativo y el modo R = 250 se excluyen (R = 250 agrupa los canales)")
    if REJILLAS[rejilla].get("nu_bin_cm"):
        raise ValueError(f"{rejilla}: muestreada en número de onda, más basta que la LSF de MUSE (muse_nativo)")
    if plantilla:
        raise ValueError("una plantilla como dato se degrada a R = 500 (`_sinteticos`): no vale con muse_nativo")


def comprueba_muestreo_nativo(muestreo_A, lsf_fwhm_A, rejilla):
    """Con `--muse-nativo`, el paso de la rejilla en MUSE (`muestreo_A`, el mayor de sus npz) tiene que ser
    ≤ `MUESTREO_MAX_NATIVO` × la LSF de MUSE; si no, ValueError."""
    tope = MUESTREO_MAX_NATIVO * float(lsf_fwhm_A)
    if not muestreo_A <= tope:
        raise ValueError(f"{rejilla}: paso de {muestreo_A:.3g} Å en MUSE, más que {tope:.3g} Å "
                         f"({MUESTREO_MAX_NATIVO:g} × la LSF, {float(lsf_fwhm_A):.3f} Å): no vale con muse_nativo")


def degrada_lsf(w, f, lsf_fwhm_A):
    """Como `tipo_ir_plantillas.degrada` (huecos a NaN donde pesan > 20 %), pero con la LSF de MUSE: una gaussiana de
    FWHM `lsf_fwhm_A` constante en vez de λ/500."""
    from musepipe.models.prep import degrade_variable
    ok = np.isfinite(f)
    g = np.interp(w, w[ok], f[ok])
    fwhm = np.full(w.shape, float(lsf_fwhm_A))
    out = degrade_variable(w, g, fwhm)
    hueco = degrade_variable(w, (~ok).astype(float), fwhm)
    out[hueco > 0.2] = np.nan
    return out


def modelos_base_muse_nativo(pls, segs, lsf_fwhm_A):
    """`tipo_ir_plantillas.modelos_base` con el brazo visible convolucionado con la LSF de MUSE (`degrada_lsf`) en vez
    de degradado a R = 500, y promediado en cada canal, como hace G3 (`prepare_template_base`: degradar y
    `resample_conserve_flux`). Los segmentos IR, a R = 500 como siempre."""
    from scripts import tipo_ir_plantillas as tp
    cache = {}
    for i, t in enumerate(pls):
        por_brazo = {}
        for brazo in ("vis", "nir"):
            w, f = t[brazo]
            orden = np.argsort(w)
            w, f = w[orden], f[orden]
            por_brazo[brazo] = (w, degrada_lsf(w, f, lsf_fwhm_A) if brazo == "vis" else tp.degrada(w, f))
        cache[i] = [tp.media_en_intervalos(*por_brazo[s["brazo"]], s["lo"], s["hi"]) for s in segs]
    return cache


def espectro_fino(pls, idx, lsf_fwhm_A, lo_A, hi_A, paso_A=PASO_NATIVO_A):
    """Para `--ens-vr`: el brazo visible de cada nodo [T, g] de la rejilla convolucionado con la LSF (`degrada_lsf`,
    sobre todo el brazo, para que los bordes no lo toquen) y llevado a una malla común de `paso_A` en [lo_A, hi_A].
    Los nodos rellenados (sin `vis`) son la media en log F de sus vecinos en T_eff, como en `Modelo`. Devuelve
    (malla, log10 F con forma (nT, nG, n))."""
    w = np.arange(lo_A, hi_A + paso_A / 2, paso_A)
    lg = np.full(idx.shape + (w.size,), np.nan)
    for a in range(idx.shape[0]):
        for b in range(idx.shape[1]):
            t = pls[idx[a, b]]
            if "vis" in t:
                wv, fv = t["vis"]
                lg[a, b] = np.log10(np.interp(w, wv, degrada_lsf(wv, fv, lsf_fwhm_A)))
    for a in range(idx.shape[0]):
        for b in range(idx.shape[1]):
            if "vis" not in pls[idx[a, b]]:
                lg[a, b] = 0.5 * (lg[a - 1, b] + lg[a + 1, b])
    return w, lg


def ensancha_y_desplaza(w, F, lo, hi, ens_A, vr_kms, nfft, f2):
    """El espectro fino `F` (malla uniforme `w`, ya a la LSF) ensanchado por una gaussiana de FWHM `ens_A` (Å; por
    FFT, `nfft` puntos y `f2` = rfftfreq(nfft)², con ceros detrás de F para que la convolución no dé la vuelta),
    desplazado por Doppler a `vr_kms` (λ_obs = λ_rep (1 + v/c)) y promediado en los canales [lo, hi] (Å, observados)."""
    from scripts import tipo_ir_plantillas as tp
    sig = ens_A / 2.3548200450309493 / (w[1] - w[0])
    if sig > 0:
        F = np.fft.irfft(np.fft.rfft(F, nfft) * np.exp(-2.0 * (np.pi * sig) ** 2 * f2), nfft)[:F.size]
    z = 1.0 + vr_kms / C_KMS
    return tp.media_en_intervalos(w, F, lo / z, hi / z)


TRACKS = {"BHAC15": ("tracks_bhac15/bhac15_tracks.npz", "Baraffe et al. 2015"),
          "ATMO2020": ("tracks_atmo2020/atmo2020_ceq_tracks.npz", "Phillips et al. 2020")}
T_D = 1500.0
LAM_K = 22000.0
MALO = -1e300
#: suelo cromosférico de Manara et al. 2013, Ec. 2: log(L_acc,noise/L_bol) = A log T_eff + B
SUELO = (6.17, -24.54)
#: qué espectros entran en `ajusta` (`--datos`): todos, solo MUSE o solo los J/H/K
DATOS = ("todo", "muse", "ir")
#: máscaras de las bandas telúricas en el espectro MUSE (`--mascara-muse`; pruebas exploratorias del 2026-10-28,
#: `~/muse_sesion_20261028/mascaras_muse/BRIEF.md`): ¿se añaden las bandas de H2O de `tipo_ir_plantillas.H2O_MUSE_A`
#: (`h2o`)? ¿qué bandas de O2 de G3 (`g3_fit_masks.telluric_bands_A`) se destapan (`quita_o2`)? `paper` es lo de las
#: cadenas del paper; `g3`, las máscaras de G3 (sin H2O); `sin_o2`, con H2O y sin O2; `sin_o2b`, con H2O y O2 A y sin
#: O2 B (O2 A lleva el doblete de K I 7665/7699, que la ventana de las líneas no tapa: así se separa una banda de la
#: otra); `sin_telurico`, sin ninguna. Las ventanas de las líneas y los canales malos de 04b se quedan siempre. El
#: cambio de O2 se hace en una copia del config en memoria (`tipo_ir_plantillas.sin_o2`), nunca en `runs/`.
MASCARAS_MUSE = {"paper": dict(h2o=True, quita_o2=()), "g3": dict(h2o=False, quita_o2=()),
                 "sin_o2": dict(h2o=True, quita_o2=("O₂ B", "O₂ A")), "sin_o2b": dict(h2o=True, quita_o2=("O₂ B",)),
                 "sin_telurico": dict(h2o=False, quita_o2=("O₂ B", "O₂ A"))}
#: intervalos del espectro MUSE (Å, [lo, hi)) que `--quita-muse` saca del ajuste, en BINS (después de agruparlos; las
#: máscaras de `MASCARAS_MUSE` actúan antes, en canales). `extremos` (apéndice D.5 del paper, «Where the models fail in
#: the MUSE spectrum»): la primera y la última banda de los residuos (`delta_entre_rejillas.BANDAS_A`), con 25 y 6 de
#: los 84 bins del paper (centros 6324–6844 y 9128–9199 Å), las dos que concentran el χ² de MUSE. `azul` y `rojo`
#: (2026-10-30): cada una de las dos por separado, para saber cuál mueve el ajuste.
INTERVALOS_MUSE = {"extremos": ((6300.0, 6900.0), (9000.0, 9350.0)), "azul": ((6300.0, 6900.0),),
                   "rojo": ((9000.0, 9350.0),)}
#: las claves del json de `ajusta` que hacen de una corrida una VARIANTE de las del paper, con el valor que tienen (o
#: el que se supone si faltan) en las cadenas del paper. Una opción nueva que cambie el modelo pone aquí su clave.
VARIANTES_JSON = {"mascara_muse": "paper", "ventana_lineas_kms": None, "ventana_halpha_kms": None,
                  "rk_fijo": None, "td_rango": None, "quita_muse": None, "inyecta_polvo": None, "muse_nativo": False,
                  "ens_vr": None, "ruido_de": None, "anclas_ir": None, "quita_ir": None, "pendiente_ir": None,
                  "banda_una_vez": False}
#: `--anclas-ir` (2026-10-30): las bandas del IR cuya escala va anclada a la fotometría, N(1, `SIGMA_IR`); las demás,
#: libres. Sin la opción, las que declara el objeto (`ir_spectra.colour_excess.bands`).
ANCLAS_IR = {"ninguna": (), "K": ("K",), "JHK": ("J", "H", "K")}
#: las demás claves que lleva el json de `ajusta` (de configuración, que `_de_la_tabla` y `_de_rejillas` filtran por
#: su valor; de resultado; y de procedencia), y las de `evo_solo`. Son las 36 de las 1187 cadenas que había el
#: 2026-10-29, más `procedencia`. Una clave que no esté ni aquí ni en `VARIANTES_JSON` hace de la corrida una
#: variante (`_es_variante`): una opción nueva no entra en las tablas del paper por olvido.
CLAVES_JSON = frozenset({
    "etiqueta", "slug", "run", "modo", "av", "av_fija", "rejilla", "datos", "r_analisis", "degrada", "tracks",
    "nlive", "semilla", "ejes", "nodos_interpolados", "ndim", "nombres", "segundos", "ncall", "logz", "logz_err",
    "logl_max", "chi2nu_mejor", "chi2nu_mejor_inflado", "mejor", "verdad", "posterior", "n_puntos", "procedencia",
    "fraccion_en_rejilla", "teff", "logg", "r_rj", "masa_mjup"})
#: el muestreador de `corre` (la procedencia de cada corrida los guarda)
MUESTREO = dict(sample="rslice", bound="multi", dlogz=0.5)


def _entra(seg, datos):
    """¿Entra el segmento (`brazo` 'vis' o 'nir') con `datos`?"""
    return datos == "todo" or (seg["brazo"] == "vis") == (datos == "muse")


def _planck(w, T):
    return 1.0 / (w ** 5 * np.expm1(1.4388e8 / (w * T)))


def _ppf_trunc_normal(u, mu, sig, lo=0.0, hi=np.inf):
    a, b = (lo - mu) / sig, (hi - mu) / sig
    return float(stats.truncnorm.ppf(u, a, b, loc=mu, scale=sig))


def _ppf_normal_rapido(mu, sig):
    """`u -> stats.norm.ppf(u, mu, sig)`, el mismo número BIT A BIT (scipy calcula `ndtri(u) * sig + mu`), sin
    las comprobaciones de argumentos de `rv_continuous.ppf`, que costaban ~10 veces la verosimilitud. Para u en
    (0, 1), que es lo que da dynesty; `tests/test_teff_bayes_prior_rapido.py` lo vigila."""
    from scipy.special import ndtri
    return lambda u: float(ndtri(u) * sig + mu)


def _ppf_trunc_normal_rapido(mu, sig, lo=0.0, hi=np.inf):
    """`u -> _ppf_trunc_normal(u, mu, sig, lo, hi)` bit a bit, con `a` y `b` calculados una vez y llamando a
    `truncnorm._ppf`, la función que `truncnorm.ppf` usa por dentro (API privada de scipy: si cambia,
    `tests/test_teff_bayes_prior_rapido.py` lo detecta)."""
    a, b = np.array([(lo - mu) / sig]), np.array([(hi - mu) / sig])
    ppf = stats.truncnorm._ppf
    return lambda u: float(ppf(np.array([u]), a, b)[0] * sig + mu)


def quita_intervalos(s, intervalos):
    """El segmento `s` (dict con `w` y `bueno`) sin los bins cuyo centro cae en algún intervalo [lo, hi) (Å)."""
    fuera = np.zeros(s["w"].size, bool)
    for lo, hi in intervalos:
        fuera |= (s["w"] >= lo) & (s["w"] < hi)
    return {**s, "bueno": s["bueno"] & ~fuera}


def _empareja(s):
    """El segmento con sus píxeles agrupados de a dos (modo R = 250): media del flujo, σ = ½ √(σ₁² + σ₂²), bueno
    si lo son los dos. Solo se juntan píxeles contiguos (el borde de uno es el del otro); un píxel suelto se pierde."""
    i = np.arange(0, s["w"].size - 1, 2)
    i = i[np.abs(s["hi"][i] - s["lo"][i + 1]) <= 0.05 * (s["hi"][i] - s["lo"][i])]
    j = i + 1
    return {**s, "w": 0.5 * (s["w"][i] + s["w"][j]), "lo": s["lo"][i], "hi": s["hi"][j],
            "f": 0.5 * (s["f"][i] + s["f"][j]), "e": 0.5 * np.hypot(s["e"][i], s["e"][j]),
            "bueno": s["bueno"][i] & s["bueno"][j]}


def _bases_nativas(pls, segs):
    """Modelos que ya están a la resolución del análisis (SPHINX, λ/Δλ = 250 por punto): sin degradar,
    interpolados en el centro de cada píxel; NaN fuera de su cobertura."""
    out = {}
    for i, t in enumerate(pls):
        fila = []
        for s in segs:
            w, f = t[s["brazo"]]
            m = np.interp(s["w"], w, f)
            m[(s["w"] < w[0]) | (s["w"] > w[-1])] = np.nan
            fila.append(m)
        out[i] = fila
    return out


def _segmentos(slug, r250=False, mascara_muse="paper", ventana_lineas=None, ventana_halpha=None, quita_muse=None,
               muse_nativo=False):
    """Los datos de `_prepara`, sin los modelos: `tipo_ir_plantillas.preparar` y los segmentos en cgs, con
    `quita_muse` y `r250` ya aplicados (los argumentos, los de `_prepara`). `--ruido-de` lee así la S/N de otro
    compañero sin cargar la rejilla."""
    from scripts import ajuste_conjunto as aj
    from scripts import tipo_ir_plantillas as tp
    if mascara_muse not in MASCARAS_MUSE:
        raise ValueError(f"mascara_muse={mascara_muse!r}: debe ser una de {tuple(MASCARAS_MUSE)}")
    if quita_muse is not None and quita_muse not in INTERVALOS_MUSE:
        raise ValueError(f"quita_muse={quita_muse!r}: debe ser uno de {tuple(INTERVALOS_MUSE)}")
    mm = MASCARAS_MUSE[mascara_muse]
    p = tp.preparar(slug, "combinado", av_fija=True, bibliotecas=("manara",), h2o_muse=mm["h2o"],
                    quita_o2_muse=mm["quita_o2"], ventana_kms=ventana_lineas, ventana_ha_kms=ventana_halpha,
                    muse_nativo=muse_nativo)
    segs = aj.segmentos_cgs(slug, p)
    if quita_muse is not None:
        segs = [quita_intervalos(s, INTERVALOS_MUSE[quita_muse]) if s["brazo"] == "vis" else s for s in segs]
    if r250:
        segs = [_empareja(s) for s in segs]
    return p, segs


def _prepara(slug, R, r250=False, degrada="cuadratura", mascara_muse="paper", ventana_lineas=None,
             ventana_halpha=None, quita_muse=None, muse_nativo=False):
    """Como `ajuste_conjunto.preparar(..., btsettl=DIR)` pero con los rangos de la rejilla elegida (aquel fija
    2000–3200 K y log g 2.5–4.5): segmentos en cgs, modelos base por segmento y `Conjunto` (píxeles comunes).

    `r250`: los píxeles de los datos se agrupan de a dos (MUSE en bins de 30 Å, el IR a λ/Δλ ≈ 250). Lo exige una
    rejilla con `r_nativa` < 500 (SPHINX), cuyos modelos se usan tal cual (`_bases_nativas`); con una rejilla de
    resolución alta, los modelos se degradan a R = 500 y se promedian en el píxel agrupado, como los datos.

    `degrada`: solo para una rejilla muestreada en número de onda (`nu_bin_cm`, ATMO 2020): «cuadratura» degrada
    con `modelos_base_nativa` (resta la resolución del muestreo), «ingenua» por el camino de las demás. Sin
    `nu_bin_cm` no hace nada.

    `mascara_muse`: las máscaras telúricas del espectro MUSE (`MASCARAS_MUSE`; `paper` es lo de siempre).
    `ventana_lineas` / `ventana_halpha`: las ventanas (km/s) de las líneas de G2 en MUSE, todas / Hα (None = las del
    run; `tipo_ir_plantillas.ventanas_de_lineas`).
    `quita_muse`: una clave de `INTERVALOS_MUSE` (None = ninguno): esos bins del espectro MUSE salen del ajuste
    (`quita_intervalos`), antes de agrupar a R = 250.
    `muse_nativo`: el espectro MUSE canal a canal (`tipo_ir_plantillas.segmento_muse(..., nativo=True)`) y su modelo
    convolucionado con la LSF de MUSE del run (`h01_lsf_fwhm_A`, la de G3; `modelos_base_muse_nativo`); el IR, a
    R = 500. Las combinaciones sin sentido las rechaza antes `valida_muse_nativo`."""
    from scripts import ajuste_conjunto as aj
    from scripts import tipo_ir_plantillas as tp
    if degrada not in DEGRADA:
        raise ValueError(f"degrada={degrada!r}: debe ser uno de {DEGRADA}")
    p, segs = _segmentos(slug, r250, mascara_muse, ventana_lineas, ventana_halpha, quita_muse, muse_nativo)
    t0, dt, nt = R["teff"]
    g0, dg, ng = R["logg"]
    p["pls"] = aj.modelos_btsettl(R["dir"], teff=(t0, t0 + (nt - 1) * dt), logg=(g0, g0 + (ng - 1) * dg),
                                  paso_vis_A=PASO_NATIVO_A if muse_nativo else None)
    if muse_nativo:
        from musepipe.config import load_run_config
        lsf = float(load_run_config(p["run"], project_root=ROOT).config["h01_lsf_fwhm_A"])
        comprueba_muestreo_nativo(max(t["muestreo_vis_A"] for t in p["pls"]), lsf, R["nombre"])
        p["bases"] = modelos_base_muse_nativo(p["pls"], segs, lsf)
    elif R.get("r_nativa", np.inf) < 500:
        p["bases"] = _bases_nativas(p["pls"], segs)
    elif R.get("nu_bin_cm") and degrada == "cuadratura":
        p["bases"] = modelos_base_nativa(p["pls"], segs, R["nu_bin_cm"])
    else:
        p["bases"] = tp.modelos_base(p["pls"], segs)
    fac = tp.factores_equilibrio(segs, p["bases"], [p["av_tabla"]])
    return p, aj.Conjunto(segs, p["bases"], 0, fac)


def valida_disco(datos="todo", rk_fijo=None, td_rango=None, quita_muse=None, inyecta_bt=None, inyecta_polvo=None):
    """Rechaza, antes de leer nada, las combinaciones de las opciones del disco y de `--quita-muse` que no tienen
    sentido: sin r_K libre, T_d no tiene datos; sin IR no hay r_K; sin MUSE no hay nada que quitar; el polvo se
    inyecta sobre un modelo de la rejilla."""
    if rk_fijo is not None and not (math.isfinite(rk_fijo) and rk_fijo >= 0):
        raise ValueError(f"rk_fijo={rk_fijo!r}: debe ser un número ≥ 0")
    if td_rango is not None:
        if len(td_rango) != 2 or not 0 < td_rango[0] < td_rango[1]:
            raise ValueError(f"td_rango={td_rango!r}: deben ser dos temperaturas 0 < T_min < T_max")
        if rk_fijo is not None:
            raise ValueError("con r_K fija (rk_fijo), la T del polvo no tiene datos: td_rango pide r_K libre")
    if datos == "muse" and (rk_fijo is not None or td_rango is not None):
        raise ValueError("con datos='muse' no hay segmentos IR ni r_K: rk_fijo y td_rango no se aplican")
    if quita_muse is not None:
        if quita_muse not in INTERVALOS_MUSE:
            raise ValueError(f"quita_muse={quita_muse!r}: debe ser uno de {tuple(INTERVALOS_MUSE)}")
        if datos == "ir":
            raise ValueError("con datos='ir' no hay espectro MUSE del que quitar intervalos")
    if inyecta_polvo is not None:
        if inyecta_bt is None:
            raise ValueError("inyecta_polvo va sobre un modelo de la rejilla: pide también inyecta_bt")
        if len(inyecta_polvo) != 2 or inyecta_polvo[0] < 0 or inyecta_polvo[1] <= 0:
            raise ValueError(f"inyecta_polvo={inyecta_polvo!r}: debe ser (r_K ≥ 0, T_d > 0)")


def valida_ruido_de(ruido_de, slug, plantilla=None, inyecta_bt=None):
    """Rechaza, antes de leer nada, un `ruido_de` sin sentido: va sobre datos de prueba (una plantilla o un modelo
    inyectado), y el ruido del mismo objeto es la corrida de siempre."""
    if ruido_de is None:
        return
    if not (plantilla or inyecta_bt):
        raise ValueError("ruido_de cambia el ruido de los datos de prueba: pide una plantilla o inyecta_bt")
    if ruido_de == slug:
        raise ValueError(f"ruido_de={ruido_de!r} es el mismo objeto: sin la opción es la corrida de siempre")


def valida_anclas_ir(anclas_ir, datos="todo"):
    """Rechaza, antes de leer nada, un `anclas_ir` sin sentido: una clave que no está en `ANCLAS_IR`, o sin IR."""
    if anclas_ir is None:
        return
    if anclas_ir not in ANCLAS_IR:
        raise ValueError(f"anclas_ir={anclas_ir!r}: debe ser una de {tuple(ANCLAS_IR)}")
    if datos == "muse":
        raise ValueError("con datos='muse' no hay segmentos IR que anclar o soltar")


def valida_quita_ir(quita_ir, datos="todo"):
    """Rechaza, antes de leer nada, un `quita_ir` sin sentido: sin ningún segmento, o sin IR."""
    if quita_ir is None:
        return
    if not quita_ir:
        raise ValueError("quita_ir vacío: nombra al menos un segmento del IR (sin la opción es la corrida de siempre)")
    if datos == "muse":
        raise ValueError("con datos='muse' no hay segmentos IR que quitar")


def valida_pendiente_ir(pendiente_ir, datos="todo"):
    """Rechaza, antes de leer nada, un `pendiente_ir` sin sentido: un ancho que no es un número finito y positivo, o
    sin IR. Que el factor quede positivo en cada segmento se comprueba en el `Modelo`, que conoce los píxeles."""
    if pendiente_ir is None:
        return
    if not (math.isfinite(pendiente_ir) and pendiente_ir > 0):
        raise ValueError(f"pendiente_ir={pendiente_ir!r}: el ancho del prior tiene que ser finito y positivo")
    if datos == "muse":
        raise ValueError("con datos='muse' no hay segmentos IR que inclinar")


def valida_banda_una_vez(banda_una_vez, datos="todo"):
    """Rechaza, antes de leer nada, `banda_una_vez` sin IR (con datos='muse' no hay bandas del IR que pesar)."""
    if banda_una_vez and datos == "muse":
        raise ValueError("con datos='muse' no hay bandas del IR que pesar una vez")


def quitados_de(segs, quita_ir):
    """Los nombres de los segmentos del IR que `--quita-ir` (2026-10-31) saca de la verosimilitud: un conjunto vacío
    sin la opción. Falla si alguno no es un segmento del IR del objeto (MUSE sale con `--datos ir`) o si no queda
    ninguno del IR (`--datos muse`). `segs`: los de `segmentos_cgs` (con `nombre` y `brazo`)."""
    if quita_ir is None:
        return frozenset()
    ir = [s["nombre"] for s in segs if s["brazo"] == "nir"]
    raros = sorted(set(quita_ir) - set(ir))
    if raros:
        raise ValueError(f"quita_ir: {', '.join(raros)} no es un segmento del IR del objeto (tiene {', '.join(ir)})")
    if set(ir) <= set(quita_ir):
        raise ValueError("quita_ir quita todos los segmentos del IR: para eso, datos='muse'")
    return frozenset(quita_ir)


def anclas_de(segs, bandas, anclas_ir, prueba=False):
    """La σ del prior de la escala de cada segmento con `--anclas-ir` (2026-10-30): `SIGMA_IR` en los segmentos IR de
    las bandas de `ANCLAS_IR[anclas_ir]`, None (libre) en los demás del IR; MUSE, la suya (`sigma_escala`). None = las
    de siempre. Sobre los datos reales (`prueba` False) solo se puede soltar un ancla: anclar una banda que el objeto
    no declara pide datos de prueba, cuyas escalas valen 1 por construcción (`_sinteticos` escala con un solo factor).
    Si no cambia ninguna, es la corrida de siempre y falla. `bandas`: la banda de cada segmento ('vis', 'J', 'H', 'K')."""
    from scripts import ajuste_conjunto as aj
    antes = [s["sigma_escala"] for s in segs]
    if anclas_ir is None:
        return antes
    quiere = set(ANCLAS_IR[anclas_ir])
    out = [x if b == "vis" else (aj.SIGMA_IR if b in quiere else None) for x, b in zip(antes, bandas)]
    nuevas = [s["nombre"] for s, x0, x in zip(segs, antes, out) if x0 is None and x is not None]
    if nuevas and not prueba:
        raise ValueError(f"anclas_ir={anclas_ir!r} ancla {', '.join(nuevas)}, que el objeto no ancla: sobre los datos "
                         "reales solo se sueltan anclas (pide una plantilla o inyecta_bt)")
    if out == antes:
        raise ValueError(f"anclas_ir={anclas_ir!r}: son las anclas de siempre (sin la opción es la misma corrida)")
    return out


def snr_cruzada(segs, bandas, segs_otro, bandas_otro):
    """La S/N por píxel de otro compañero en los píxeles de este (`--ruido-de`, 2026-10-30): la cobertura (qué
    segmentos y qué píxeles) es la de este; el ruido, el del otro.

    Cada segmento toma la S/N del segmento del otro con el mismo nombre (instrumento y banda) y, si el otro no lo
    tiene, la mediana, en cada λ, de los segmentos del otro en esa banda que la cubren (el más cercano si ninguno la
    cubre). La del otro se interpola linealmente en λ sobre sus píxeles buenos con S/N finita y > 0 (constante más
    allá de sus extremos). Los píxeles que este no usa (S/N no finita o ≤ 0) siguen fuera, como en `_sinteticos`.
    `bandas`: la banda de cada segmento ('vis', 'J', 'H', 'K'). Devuelve una S/N por segmento de `segs`."""
    perfiles = []
    for o, b in zip(segs_otro, bandas_otro):
        with np.errstate(invalid="ignore", divide="ignore"):
            sn = np.abs(o["f"]) / o["e"]
        ok = np.isfinite(sn) & (sn > 0) & o.get("bueno", np.ones(sn.size, bool))
        if ok.any():
            perfiles.append((o["nombre"], b, o["w"][ok], sn[ok]))
    out = []
    for s, b in zip(segs, bandas):
        cand = ([q for q in perfiles if q[0] == s["nombre"]] or [q for q in perfiles if q[1] == b])
        if not cand:
            raise ValueError(f"{s['nombre']}: el otro compañero no tiene datos en la banda {b}")
        w = s["w"]
        vals = np.array([np.interp(w, wq, sq) for _, _, wq, sq in cand])
        cubre = np.array([(w >= wq[0]) & (w <= wq[-1]) for _, _, wq, _ in cand])
        dist = np.array([np.maximum(np.maximum(wq[0] - w, w - wq[-1]), 0.0) for _, _, wq, _ in cand])
        cerca = vals[np.argmin(dist, axis=0), np.arange(w.size)]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)   # columnas sin ningún segmento que cubra
            med = np.nanmedian(np.where(cubre, vals, np.nan), axis=0)
        sn = np.where(cubre.any(axis=0), med, cerca)
        with np.errstate(invalid="ignore", divide="ignore"):
            propia = np.abs(s["f"]) / s["e"]
        out.append(np.where(np.isfinite(propia) & (propia > 0), sn, np.nan))
    return out


class Modelo:
    #: r_K fija (None = libre o, sin IR, 0) y rango de la T del polvo libre (None = `T_D`). A nivel de clase para que
    #: un `Modelo` hecho con `__new__` (los tests sintéticos) tenga los valores de siempre.
    rk_fijo = None
    td_rango = None
    muse_nativo = False
    ens_vr = None
    ruido_de = None
    anclas_ir = None
    quita_ir = None
    pendiente_ir = None
    banda_una_vez = False

    def __init__(self, slug, modo="libre", av="primaria", tracks="ATMO2020", inyecta_bt=None, plantilla=None,
                 av_inyectada=None, semilla=0, av_fija=None, rejilla="btsettl", dir_rejilla=None, datos="todo",
                 r250=False, degrada="cuadratura", mascara_muse="paper", ventana_lineas=None, ventana_halpha=None,
                 rk_fijo=None, td_rango=None, quita_muse=None, inyecta_polvo=None, muse_nativo=False, ens_vr=None,
                 ruido_de=None, anclas_ir=None, quita_ir=None, pendiente_ir=None, banda_una_vez=False):
        """`rk_fijo`, `td_rango`, `quita_muse` e `inyecta_polvo` (pruebas del disco y del D.5, 2026-10-29) no cambian
        nada si se dejan en None: `rk_fijo` fija r_K (r_K = 0 es el modelo sin disco); `td_rango` = (T_min, T_max)
        deja libre la T del polvo (uniforme) en vez de `T_D`; `quita_muse` quita del espectro MUSE los bins de
        `INTERVALOS_MUSE[quita_muse]`; `inyecta_polvo` = (r_K, T_d) suma un exceso de polvo al modelo de
        `inyecta_bt`. `muse_nativo` (2026-10-30; False = lo de siempre): el espectro MUSE a su resolución, canal a
        canal, en vez de a R = 500 (`_prepara`). `ens_vr` (2026-10-30; None = lo de siempre, pide `muse_nativo`) =
        (tope de la FWHM extra en Å, tope de |v| en km/s): dos parámetros más, `ens_A` y `vr_kms`, con los que el
        modelo de MUSE se rehace en cada llamada desde el espectro fino (`espectro_fino`, `ensancha_y_desplaza`).
        `ruido_de` (2026-10-30; None = lo de siempre, pide datos de prueba) = el slug de otro compañero: los datos de
        prueba llevan la cobertura de este y la S/N por píxel de aquel (`snr_cruzada`). `anclas_ir` (2026-10-30; None =
        las anclas que declara el objeto) = una clave de `ANCLAS_IR`: las bandas del IR con la escala anclada a la
        fotometría (`anclas_de`); solo cambia el prior de las escalas, no los datos. `quita_ir` (2026-10-31; None =
        todos) = los nombres de los segmentos del IR que salen de la verosimilitud (`quitados_de`); los datos de los
        demás no cambian. `pendiente_ir` (2026-11-01; None = sin inclinación) = el ancho del prior uniforme de la
        inclinación de cada segmento del IR, en fracción por 0.1 µm: un parámetro `p_<SEG>` por segmento, y su modelo
        multiplicado por 1 + p·`xp` (`xp` = (λ − λ_c)/0.1 µm, con λ_c el centro de sus píxeles). `banda_una_vez`
        (2026-11-02; False = lo de siempre): el término de cada segmento del IR en `loglike`, por su `peso` = 1/N, con
        N los segmentos de su banda que entran (después de `quita_ir`); MUSE no lleva peso."""
        from musepipe.config import load_run_config
        from musepipe.models.tracks import TrackGrid
        from scripts import tipo_ir_indices as tir
        warnings.simplefilter("ignore", RuntimeWarning)
        if datos not in DATOS:
            raise ValueError(f"datos={datos!r}: debe ser uno de {DATOS}")
        valida_disco(datos, rk_fijo, td_rango, quita_muse, inyecta_bt, inyecta_polvo)
        valida_muse_nativo(muse_nativo, datos, rejilla, r250, plantilla, ens_vr)
        valida_ruido_de(ruido_de, slug, plantilla, inyecta_bt)
        valida_anclas_ir(anclas_ir, datos)
        valida_quita_ir(quita_ir, datos)
        valida_pendiente_ir(pendiente_ir, datos)
        valida_banda_una_vez(banda_una_vez, datos)
        self.pendiente_ir = None if pendiente_ir is None else float(pendiente_ir)
        self.banda_una_vez = bool(banda_una_vez)
        self.ruido_de = ruido_de
        self.anclas_ir = anclas_ir
        self.quita_ir = None if quita_ir is None else tuple(sorted(set(quita_ir)))
        self.muse_nativo = bool(muse_nativo)
        self.ens_vr = None if ens_vr is None else (float(ens_vr[0]), float(ens_vr[1]))
        self.slug, self.modo, self.av_modo, self.av_fija, self.datos = slug, modo, av, av_fija, datos
        self.rk_fijo = None if rk_fijo is None else float(rk_fijo)
        self.td_rango = None if td_rango is None else (float(td_rango[0]), float(td_rango[1]))
        self.quita_muse = quita_muse
        R = dict(REJILLAS[rejilla])
        if dir_rejilla is not None:
            R["dir"] = Path(dir_rejilla)
        self.dir_rejilla = Path(R["dir"]).expanduser().resolve()
        self.rejilla, self.rejilla_nombre = rejilla, R["nombre"]
        self.T0, self.dT, self.nT = R["teff"]
        self.G0, self.dG, self.nG = R["logg"]
        self.T0, self.G0 = float(self.T0), float(self.G0)
        self.r_analisis = 250 if _r250(rejilla, r250) else 500
        #: cómo se degradó la rejilla, solo si está muestreada en número de onda (ATMO 2020); None en las demás
        self.degrada = degrada if R.get("nu_bin_cm") else None
        self.mascara_muse = mascara_muse
        self.ventana_lineas, self.ventana_halpha = ventana_lineas, ventana_halpha
        p, conj = _prepara(slug, R, r250=self.r_analisis == 250, degrada=degrada, mascara_muse=mascara_muse,
                           ventana_lineas=ventana_lineas, ventana_halpha=ventana_halpha, quita_muse=quita_muse,
                           muse_nativo=self.muse_nativo)
        self.run = p["run"]
        cfg = load_run_config(self.run, project_root=ROOT).config
        self.cfg = cfg
        self.d_pc, self.d_err = float(cfg["h03_distance_pc"]), float(cfg["h03_distance_err_pc"])
        self.av0, self.av_err = float(cfg["h03_av"]), float(cfg["h03_av_err"])
        edad, (elo, ehi) = float(cfg["g3_age_myr"]), cfg["g3_age_err_myr"]
        self.edad0 = edad
        self.edad_sig_ln = 0.5 * (math.log((edad + ehi) / edad) + math.log(edad / (edad - elo)))
        perfil = tir.perfil(slug)
        self.fisica = perfil.get("fisica", {})
        ir = perfil["ir_spectra"]["spectra"]
        # la σ del prior de cada escala: la de `segmentos_cgs` o, con `anclas_ir`, la de `anclas_de` (los datos de
        # prueba se escalan antes, en `_sinteticos`, con el segmento anclado de siempre)
        sigmas = anclas_de(conj.segs, ["vis" if s["brazo"] == "vis" else ir[s["nombre"]]["band"] for s in conj.segs],
                           anclas_ir, prueba=bool(plantilla or inyecta_bt))
        # los segmentos del IR que `quita_ir` saca: solo de `self.segs`; todo lo demás se arma con todos
        quitados = quitados_de(conj.segs, quita_ir)

        # rejilla [T, g] -> índice del modelo
        idx = np.full((self.nT, self.nG), -1)
        for i, t in enumerate(p["pls"]):
            idx[int(round((t["teff"] - self.T0) / self.dT)), int(round((t["logg"] - self.G0) / self.dG))] = i
        self.rellenados = []
        for a in range(self.nT):
            for b in range(self.nG):
                if idx[a, b] >= 0:
                    continue
                if not (0 < a < self.nT - 1 and idx[a - 1, b] >= 0 and idx[a + 1, b] >= 0):
                    raise SystemExit(f"rejilla {rejilla} incompleta en ({self.T0 + a * self.dT}, {self.G0 + b * self.dG})")
                m0, m1 = p["bases"][idx[a - 1, b]], p["bases"][idx[a + 1, b]]
                nuevo = len(p["pls"])
                p["pls"].append(dict(teff=self.T0 + a * self.dT, logg=self.G0 + b * self.dG, interpolado=True))
                p["bases"][nuevo] = [np.sqrt(x0 * x1) for x0, x1 in zip(m0, m1)]   # media en log F
                idx[a, b] = nuevo
                self.rellenados.append((self.T0 + a * self.dT, self.G0 + b * self.dG))

        # log F(2.2 µm) en cada nodo: la referencia del exceso de polvo (antes de los datos de prueba, que la usan
        # para inyectar polvo; no depende de ellos)
        k_ref = next(k for k, s in enumerate(conj.segs) if s["brazo"] == "nir" and s["w"][0] <= LAM_K <= s["w"][-1])
        self.lref = np.empty((self.nT, self.nG))
        for a in range(self.nT):
            for b in range(self.nG):
                s, m = conj.segs[k_ref], p["bases"][idx[a, b]][k_ref]
                ok = np.isfinite(m)
                self.lref[a, b] = math.log10(float(np.interp(LAM_K, s["w"][ok], m[ok])))

        datos_sint = None
        if plantilla or inyecta_bt:
            datos_sint = self._sinteticos(p, conj, plantilla, inyecta_bt, av_inyectada, idx, semilla, inyecta_polvo)

        self.segs = []
        for k, s in enumerate(conj.segs):
            if not _entra(s, datos) or s["nombre"] in quitados:
                continue
            c = conj.comun[k].copy()
            f, e = s["f"], s["e"]
            if datos_sint is not None:
                f, e = datos_sint[k]
                c &= np.isfinite(f) & np.isfinite(e) & (e > 0)
            w = s["w"][c]
            lg = np.empty((self.nT, self.nG, int(c.sum())))
            for a in range(self.nT):
                for b in range(self.nG):
                    lg[a, b] = np.log10(p["bases"][idx[a, b]][k][c])
            banda = "vis" if s["brazo"] == "vis" else ir[s["nombre"]]["band"]
            self.segs.append(dict(
                nombre=s["nombre"], banda=banda, w=w, d=f[c], e2=e[c] ** 2, lg=lg,
                a=np.asarray(CCMExtinction(3.1, citation="CCM89").a_lambda_over_av(w), float),
                bb=(_planck(w, T_D) / _planck(LAM_K, T_D)) if s["brazo"] == "nir" else np.zeros(w.size),
                sigma=sigmas[k]))
            if self.pendiente_ir is not None and s["brazo"] == "nir":
                # la abscisa de la inclinación: en 0.1 µm desde el centro de los píxeles del segmento
                self.segs[-1]["xp"] = (w - 0.5 * (w.min() + w.max())) / 1000.0
            if self.ens_vr is not None and s["brazo"] == "vis":
                # el espectro fino del que se rehace el modelo de MUSE en cada llamada (`_muse_fino`)
                lo, hi = s["lo"][c], s["hi"][c]
                wf, lgf = espectro_fino(p["pls"], idx, float(cfg["h01_lsf_fwhm_A"]), float(lo.min()) - MARGEN_FINO_A,
                                        float(hi.max()) + MARGEN_FINO_A)
                nfft = 1 << int(math.ceil(math.log2(wf.size + 4.0 * self.ens_vr[0] / 2.3548200450309493
                                                    / PASO_NATIVO_A + 64)))
                self.segs[-1]["fino"] = dict(w=wf, lg=lgf, lo=lo, hi=hi, nfft=nfft, f2=np.fft.rfftfreq(nfft) ** 2)
        self.bandas = sorted({s["banda"] for s in self.segs})
        if self.banda_una_vez:
            # cada banda del IR una vez: el término de cada segmento, por 1/N (N = los segmentos de su banda que entran)
            n_banda = {b: sum(s["banda"] == b for s in self.segs) for b in self.bandas if b != "vis"}
            for s in self.segs:
                if s["banda"] != "vis":
                    s["peso"] = 1.0 / n_banda[s["banda"]]
        if self.pendiente_ir is not None:
            # el factor 1 + p·xp tiene que quedar positivo en todo segmento con |p| ≤ ancho
            mitad = max(float(np.abs(s["xp"]).max()) for s in self.segs if "xp" in s)
            if self.pendiente_ir * mitad >= 1.0:
                raise ValueError(f"pendiente_ir={self.pendiente_ir:g}: con un segmento de ±{mitad:.2f} × 0.1 µm el "
                                 f"factor de la inclinación llega a cero; el ancho tiene que ser < {1.0 / mitad:.3f}")
        self.track = None
        if modo == "evo":
            rel, cita = TRACKS[tracks]
            root = (ROOT / cfg["g3_libraries_root"]).resolve()
            self.track = TrackGrid(root / rel, family=tracks, citation=cita, version=None)
            self.tracks_nombre = tracks
        self._prepara_priores()

    # -------------------------------------------------------------- datos de prueba
    def _sinteticos(self, p, conj, plantilla, inyecta_bt, av_inj, idx, semilla, inyecta_polvo=None):
        """[(f, e)] por segmento: una plantilla o un modelo de la rejilla, con la S/N por píxel del dato real y ruido.
        `inyecta_polvo` = (r_K, T_d): al modelo de la rejilla se le suma el exceso de la Ec. B.1 en el IR,
        r_K · F(2.2 µm) · B_λ(T_d)/B_2.2(T_d), antes de la extinción (como en `modelos`)."""
        from scripts import ajuste_conjunto as aj
        from scripts import tipo_ir_plantillas as tp
        rng = np.random.default_rng(semilla)
        if plantilla:
            pls = tp.plantillas(self.cfg, ("manara", "lateM", "xshyne"))
            i = next(i for i, t in enumerate(pls) if t["objeto"] == plantilla)
            base = tp.modelos_base([pls[i]], conj.segs)[0]
            av = self.av0 if av_inj is None else float(av_inj)
            sin = [base[k] * 10 ** (-0.4 * np.asarray(CCMExtinction(3.1, citation="CCM89").a_lambda_over_av(s["w"]),
                                                       float) * av) for k, s in enumerate(conj.segs)]
            # UN solo factor (conserva los colores): el del segmento anclado a la fotometría, o MUSE
            k0 = next((k for k, s in enumerate(conj.segs) if s["sigma_escala"] == aj.SIGMA_IR),
                      next(k for k, s in enumerate(conj.segs) if s["brazo"] == "vis"))
            ok = conj.comun[k0] & np.isfinite(sin[k0])
            fac = float(np.median(conj.segs[k0]["f"][ok]) / np.median(sin[k0][ok]))
            sin = [x * fac for x in sin]
            self.verdad = dict(plantilla=plantilla, spt=pls[i]["spt"], lib=pls[i]["lib"], av=av)
        else:
            T, G, av, r_rj = inyecta_bt
            x, y = (T - self.T0) / self.dT, (G - self.G0) / self.dG
            i0, j0 = min(int(x), self.nT - 2), min(int(y), self.nG - 2)
            t, u = x - i0, y - j0
            om = (r_rj * RJUP_CM / (self.d_pc * PC_CM)) ** 2
            if inyecta_polvo is not None:
                lr = self.lref
                fref = 10 ** ((1 - t) * (1 - u) * lr[i0, j0] + t * (1 - u) * lr[i0 + 1, j0]
                              + (1 - t) * u * lr[i0, j0 + 1] + t * u * lr[i0 + 1, j0 + 1])
                rk_i, td_i = map(float, inyecta_polvo)
            sin = []
            for k, s in enumerate(conj.segs):
                L = [np.log10(p["bases"][idx[a, b]][k]) for a, b in ((i0, j0), (i0 + 1, j0), (i0, j0 + 1), (i0 + 1, j0 + 1))]
                lf = (1 - t) * (1 - u) * L[0] + t * (1 - u) * L[1] + (1 - t) * u * L[2] + t * u * L[3]
                ext = 10 ** (-0.4 * np.asarray(CCMExtinction(3.1, citation="CCM89").a_lambda_over_av(s["w"]), float) * av)
                if inyecta_polvo is not None and s["brazo"] == "nir":
                    bb = _planck(s["w"], td_i) / _planck(LAM_K, td_i)
                    sin.append(om * (10 ** lf + rk_i * fref * bb) * ext)
                else:
                    sin.append(om * 10 ** lf * ext)
            self.verdad = dict(teff=T, logg=G, av=av, r_rj=r_rj)
            if inyecta_polvo is not None:
                self.verdad.update(rk=rk_i, td=td_i)
        # con `ruido_de`, la S/N por píxel es la del otro compañero en los píxeles de este (`snr_cruzada`)
        snr_otro = None if self.ruido_de is None else self._snr_de_otro(conj.segs)
        out = []
        for k, s in enumerate(conj.segs):
            with np.errstate(invalid="ignore", divide="ignore"):
                snr = np.abs(s["f"]) / s["e"] if snr_otro is None else snr_otro[k]
                e = np.abs(sin[k]) / np.where(snr > 0, snr, np.nan)
            out.append((sin[k] + rng.normal(0.0, np.nan_to_num(e)), e))
        return out

    def _snr_de_otro(self, segs):
        """`snr_cruzada` de los segmentos de este objeto con los de `ruido_de`, leídos con las mismas opciones."""
        from scripts import tipo_ir_indices as tir
        _, otros = _segmentos(self.ruido_de, self.r_analisis == 250, self.mascara_muse, self.ventana_lineas,
                              self.ventana_halpha, self.quita_muse, self.muse_nativo)

        def bandas(slug, ss):
            ir = tir.perfil(slug)["ir_spectra"]["spectra"]
            return ["vis" if s["brazo"] == "vis" else ir[s["nombre"]]["band"] for s in ss]
        return snr_cruzada(segs, bandas(self.slug, segs), otros, bandas(self.ruido_de, otros))

    # -------------------------------------------------------------- parámetros
    def _prepara_priores(self):
        n, t, tn = [], [], []   # tn: (índice, mu, sigma, lo) de las normales truncadas

        def add(nombre, f, truncada=None):
            if truncada is not None:
                tn.append((len(n), *truncada))
            n.append(nombre)
            t.append(f)
        # los ppf de la normal y de la normal truncada van por el camino rápido (`_ppf_*_rapido` y, en
        # `prior_transform`, las truncadas juntas), bit a bit iguales a los de scipy: así se hicieron las cadenas del
        # paper, con `stats.*.ppf`, y así se reproducen
        if self.modo == "libre":
            add("teff", lambda u: self.T0 + u * (self.nT - 1) * self.dT)
            add("logg", lambda u: self.G0 + u * (self.nG - 1) * self.dG)
            add("r_rj", lambda u: 10 ** (math.log10(0.5) + u * (math.log10(6.0) - math.log10(0.5))))
        else:
            mu, sg = math.log10(self.edad0), self.edad_sig_ln / math.log(10)
            add("log_edad_myr", _ppf_normal_rapido(mu, sg))
            add("log_lbol", lambda u: -4.5 + 3.0 * u)
        add("d_pc", _ppf_normal_rapido(self.d_pc, self.d_err))
        if self.av_fija is not None:
            add("av", lambda u: self.av_fija + 0.02 * (u - 0.5))   # perfil: A_V fija (±0.01)
        elif self.av_modo == "libre":
            add("av", lambda u: 8.0 * u)
        else:
            add("av", _ppf_trunc_normal_rapido(self.av0, self.av_err), truncada=(self.av0, self.av_err, 0.0))
        # sin IR, r_K no tiene datos; con `rk_fijo`, r_K no es un parámetro
        if any(s["banda"] != "vis" for s in self.segs) and self.rk_fijo is None:
            add("rk", lambda u: 1.5 * u)
            if self.td_rango is not None:   # la T del polvo libre (uniforme), justo detrás de r_K
                t0, t1 = self.td_rango
                add("td", lambda u: t0 + u * (t1 - t0))
        if self.ens_vr is not None:         # el ensanchamiento extra y la velocidad del modelo de MUSE (uniformes)
            e_max, v_max = self.ens_vr
            add("ens_A", lambda u: e_max * u)
            add("vr_kms", lambda u: v_max * (2.0 * u - 1.0))
        for s in self.segs:
            if s["sigma"] is None:
                add(f"s_{s['nombre']}", lambda u: 10 ** (math.log10(0.2) + u * 2 * math.log10(5.0)))
            else:
                add(f"s_{s['nombre']}", _ppf_trunc_normal_rapido(1.0, s["sigma"], lo=0.05),
                    truncada=(1.0, s["sigma"], 0.05))
        if self.pendiente_ir is not None:   # la inclinación de cada segmento del IR (uniforme), detrás de las escalas
            ancho = self.pendiente_ir
            for s in self.segs:
                if "xp" in s:
                    add(f"p_{s['nombre']}", lambda u: ancho * (2.0 * u - 1.0))
        for b in self.bandas:
            add(f"lnf_{b}", lambda u: math.log(1e-3) + u * (math.log(1.0) - math.log(1e-3)))
        self.nombres, self._t = n, t
        self.ndim = len(n)
        self.pos = {k: i for i, k in enumerate(n)}
        # para `prior_transform`: las truncadas, en una sola llamada a `truncnorm._ppf` con a = (lo − mu)/σ y
        # b = +∞ calculados aquí; el resto, una a una
        juntas = {i for i, *_ in tn}
        self._otros = [(i, f) for i, f in enumerate(t) if i not in juntas]
        self._truncadas = None
        if tn:
            i, mu, sg, lo = (np.array(x) for x in zip(*tn))
            self._truncadas = (i.astype(int), (lo - mu) / sg, np.full(i.size, np.inf), sg, mu)

    def prior_transform(self, u):
        """El cubo unidad -> parámetros. Las normales truncadas (A_V y las escalas) van juntas: `truncnorm._ppf`
        calcula elemento a elemento, así que da los mismos números que una a una (`_t`), y casi todo su coste es
        fijo por llamada (`tests/test_teff_bayes_prior_rapido.py` compara con los `ppf` públicos)."""
        x = np.empty(len(self._t))
        for i, f in self._otros:
            x[i] = f(u[i])
        if self._truncadas is not None:
            i, a, b, sg, mu = self._truncadas
            x[i] = stats.truncnorm._ppf(np.asarray(u)[i], a, b) * sg + mu
        return x

    # -------------------------------------------------------------- física
    def fisica_de(self, th):
        """(teff, logg, r_cm, masa_mjup, lbol_lsun) del vector de parámetros; None si cae fuera."""
        P = self.pos
        if self.modo == "libre":
            teff, logg, r_cm = th[P["teff"]], th[P["logg"]], th[P["r_rj"]] * RJUP_CM
            lbol = 4 * math.pi * r_cm ** 2 * SIGMA_SB_CGS * teff ** 4 / LSUN_ERG_S
            return teff, logg, r_cm, float("nan"), lbol
        tr = self.track.lookup(10 ** th[P["log_lbol"]], 10 ** th[P["log_edad_myr"]] / 1e3)
        if not tr["in_range"] or tr["clamped"]:
            return None
        teff, logg = tr["teff_k"], tr["logg"]
        if not (self.T0 <= teff <= self.T0 + (self.nT - 1) * self.dT and self.G0 <= logg <= self.G0 + (self.nG - 1) * self.dG):
            return None
        return teff, logg, tr["radius_rsun"] * RSUN_CM, tr["mass_msun"] * 1047.57, 10 ** th[P["log_lbol"]]

    def modelos(self, th):
        f = self.fisica_de(th)
        if f is None:
            return None
        teff, logg, r_cm, _, _ = f
        P = self.pos
        x, y = (teff - self.T0) / self.dT, (logg - self.G0) / self.dG
        i, j = min(int(x), self.nT - 2), min(int(y), self.nG - 2)
        t, u = x - i, y - j
        c00, c10, c01, c11 = (1 - t) * (1 - u), t * (1 - u), (1 - t) * u, t * u
        lr = self.lref
        fref = 10 ** (c00 * lr[i, j] + c10 * lr[i + 1, j] + c01 * lr[i, j + 1] + c11 * lr[i + 1, j + 1])
        om = (r_cm / (th[P["d_pc"]] * PC_CM)) ** 2
        av = th[P["av"]]
        rk = th[P["rk"]] if "rk" in P else (0.0 if self.rk_fijo is None else self.rk_fijo)
        td = th[P["td"]] if "td" in P else None
        out = []
        for s in self.segs:
            if "fino" in s:
                F = self._muse_fino(s["fino"], th, i, j, (c00, c10, c01, c11))
            else:
                L = s["lg"]
                F = 10 ** (c00 * L[i, j] + c10 * L[i + 1, j] + c01 * L[i, j + 1] + c11 * L[i + 1, j + 1])
            # el cuerpo negro del polvo: precalculado a `T_D` (el camino de las cadenas del paper) o a la T libre
            bb = s["bb"] if td is None or s["banda"] == "vis" else _planck(s["w"], td) / _planck(LAM_K, td)
            m = th[P[f"s_{s['nombre']}"]] * om * (F + rk * fref * bb) * 10 ** (-0.4 * s["a"] * av)
            if "xp" in s:   # `pendiente_ir`: la inclinación del segmento
                m = m * (1.0 + th[P[f"p_{s['nombre']}"]] * s["xp"])
            out.append(m)
        return out

    def _muse_fino(self, fi, th, i, j, c):
        """Con `ens_vr`: el modelo de MUSE en los canales, desde el espectro fino interpolado en log F en el nodo (i, j)
        con los pesos bilineales `c`, ensanchado y desplazado con `ens_A` y `vr_kms` (`ensancha_y_desplaza`)."""
        L = fi["lg"]
        F = 10 ** (c[0] * L[i, j] + c[1] * L[i + 1, j] + c[2] * L[i, j + 1] + c[3] * L[i + 1, j + 1])
        return ensancha_y_desplaza(fi["w"], F, fi["lo"], fi["hi"], th[self.pos["ens_A"]], th[self.pos["vr_kms"]],
                                   fi["nfft"], fi["f2"])

    def loglike(self, th):
        M = self.modelos(th)
        if M is None:
            return MALO
        tot = 0.0
        for s, m in zip(self.segs, M):
            fb = math.exp(th[self.pos[f"lnf_{s['banda']}"]])
            var = s["e2"] + (fb * m) ** 2
            r = s["d"] - m
            t = -0.5 * float(np.sum(r * r / var + np.log(var)))
            tot += t * s["peso"] if "peso" in s else t   # `banda_una_vez`: el peso del segmento
        return tot

    def chi2(self, th, inflado=False):
        M = self.modelos(th)
        out = {}
        for s, m in zip(self.segs, M):
            var = s["e2"] + ((math.exp(th[self.pos[f"lnf_{s['banda']}"]]) * m) ** 2 if inflado else 0.0)
            out[s["nombre"]] = float(np.sum((s["d"] - m) ** 2 / var)) / s["d"].size
        return out


def evo_solo(mod_cfg, slug, tracks, n=20000, semilla=0, root=None):
    """T_eff, R, M y log g de la L_bol publicada y el prior de edad, sin espectro (la referencia). `root`: la raíz
    del proyecto contra la que se resuelve `g3_libraries_root` (por defecto este clon; desde un worktree, la del
    clon con los datos)."""
    from musepipe.models.tracks import TrackGrid
    from scripts import tipo_ir_indices as tir
    rng = np.random.default_rng(semilla)
    cfg = mod_cfg
    perfil = tir.perfil(slug)
    fis = perfil["fisica"]
    edad, (elo, ehi) = float(cfg["g3_age_myr"]), cfg["g3_age_err_myr"]
    sig = 0.5 * (math.log((edad + ehi) / edad) + math.log(edad / (edad - elo)))
    rel, cita = TRACKS[tracks]
    raiz = ROOT if root is None else Path(root)
    tg = TrackGrid((raiz / cfg["g3_libraries_root"]).resolve() / rel, family=tracks, citation=cita, version=None)
    lbol = 10 ** rng.normal(fis["log_lbol"], fis["log_lbol_err"], n)
    age = np.exp(rng.normal(math.log(edad), sig, n)) / 1e3
    s = tg.sample(lbol, age)
    ok = s["in_range"]
    return dict(teff=s["teff_k"][ok], logg=s["logg"][ok], r_rj=s["radius_rsun"][ok] * RSUN_CM / RJUP_CM,
                masa_mjup=s["mass_msun"][ok] * 1047.57, fraccion_en_rejilla=float(ok.mean()))


def percentiles(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if not x.size:
        return None
    q = np.percentile(x, [2.5, 16, 50, 84, 97.5])
    return dict(p2_5=float(q[0]), p16=float(q[1]), p50=float(q[2]), p84=float(q[3]), p97_5=float(q[4]))


def corre(mod, nlive, semilla, dlogz=MUESTREO["dlogz"]):
    import dynesty
    from dynesty.utils import resample_equal
    t0 = time.time()
    rng = np.random.default_rng(semilla)
    sampler = dynesty.NestedSampler(mod.loglike, mod.prior_transform, mod.ndim, nlive=nlive,
                                    sample=MUESTREO["sample"], bound=MUESTREO["bound"], rstate=rng)
    sampler.run_nested(dlogz=dlogz, print_progress=False)
    res = sampler.results
    w = np.exp(res.logwt - res.logz[-1])
    eq = resample_equal(res.samples, w / w.sum(), rstate=rng)
    der = np.array([mod.fisica_de(th) or (np.nan,) * 5 for th in eq], float)
    imax = int(np.argmax(res.logl))
    return dict(segundos=round(time.time() - t0, 1), ncall=int(np.sum(res.ncall)), logz=float(res.logz[-1]),
                logz_err=float(res.logzerr[-1]), muestras=eq, derivadas=der, mejor=res.samples[imax],
                logl_max=float(res.logl[imax]))


def etiqueta_de(a):
    """La etiqueta de una corrida de `ajusta` (nombre de sus json/npz): la de `--etiqueta` o la que se compone de
    los argumentos. Con `--mascara-muse` distinta de `paper` lleva `mascara-<variante>` antes de la semilla, para
    que esas cadenas no se confundan con las del paper; con `--ventana-lineas` / `--ventana-halpha`, `ventana-<KMS>`
    / `ventanaha-<KMS>` después de la máscara; y después, las de las pruebas del disco y del D.5: `rk<RK>`
    (`--rk-fijo`), `td<TMIN>-<TMAX>` (`--td-libre`), `quita-<INTERVALOS>` (`--quita-muse`) y `polvo-<RK>-<TD>`
    (`--inyecta-polvo`). `muse-nativo` (`--muse-nativo`) va justo detrás de `r250` / `ingenua`, y `ens-vr` (`--ens-vr`)
    detrás de él. `ruido-<SLUG>` (`--ruido-de`) va detrás de los datos de prueba (`pl-…` / `bt-…`), y `anclas-<BANDAS>`
    (`--anclas-ir`) detrás de él, y `quitair-<SEGS>` (`--quita-ir`, en orden alfabético) detrás, y `pendir<ANCHO>`
    (`--pendiente-ir`) detrás, y `bandaunavez` (`--banda-una-vez`) detrás. Las opciones se leen con `getattr`, así
    que vale un `Namespace` sin las nuevas."""
    degrada = getattr(a, "degrada", "cuadratura")
    mascara = getattr(a, "mascara_muse", "paper")
    vl, vh = getattr(a, "ventana_lineas", None), getattr(a, "ventana_halpha", None)
    rkf, td = getattr(a, "rk_fijo", None), getattr(a, "td_libre", None)
    quita, polvo = getattr(a, "quita_muse", None), getattr(a, "inyecta_polvo", None)
    nativo, ens_vr = getattr(a, "muse_nativo", False), getattr(a, "ens_vr", False)
    ruido = getattr(a, "ruido_de", None)
    anclas = getattr(a, "anclas_ir", None)
    quita_ir = getattr(a, "quita_ir", None)
    pend = getattr(a, "pendiente_ir", None)
    una = getattr(a, "banda_una_vez", False)
    return a.etiqueta or "_".join(x for x in (a.slug, a.rejilla if a.rejilla != "btsettl" else None, a.modo, a.tracks if a.modo != "libre" else None, f"av{a.av}",
                                              f"pl-{a.plantilla}" if a.plantilla else None,
                                              f"avfija{a.av_fija:g}" if a.av_fija is not None else None,
                                              "bt-" + "-".join(f"{v:g}" for v in a.inyecta_bt) if a.inyecta_bt else None,
                                              f"ruido-{ruido}" if ruido else None,
                                              f"anclas-{anclas}" if anclas else None,
                                              "quitair-" + "-".join(sorted(set(quita_ir))) if quita_ir else None,
                                              f"pendir{pend:g}" if pend is not None else None,
                                              "bandaunavez" if una else None,
                                              f"solo{a.datos}" if a.datos != "todo" else None,
                                              "r250" if _r250(a.rejilla, a.r250) else None,
                                              "ingenua" if (degrada == "ingenua" and REJILLAS[a.rejilla].get("nu_bin_cm"))
                                              else None,
                                              "muse-nativo" if nativo else None,
                                              "ens-vr" if ens_vr else None,
                                              f"mascara-{mascara}" if mascara != "paper" else None,
                                              f"ventana-{vl:g}" if vl is not None else None,
                                              f"ventanaha-{vh:g}" if vh is not None else None,
                                              f"rk{rkf:g}" if rkf is not None else None,
                                              f"td{td[0]:g}-{td[1]:g}" if td else None,
                                              f"quita-{quita}" if quita else None,
                                              "polvo-" + "-".join(f"{v:g}" for v in polvo) if polvo else None,
                                              f"s{a.semilla}") if x)


def procedencia(mod=None, a=None):
    """Con qué se hizo una corrida, para su json: versiones de musepipe, python, numpy, scipy y dynesty; el commit
    del clon del código y si tenía cambios sin commit; el muestreador (`MUESTREO`); el directorio de la rejilla,
    resuelto; la raíz de la que se leyeron los runs; los argumentos de `ajusta`; y la fecha (UTC)."""
    import datetime
    import platform
    import subprocess

    import scipy

    import musepipe
    codigo = Path(__file__).resolve().parents[1]
    out = dict(musepipe=musepipe.__version__, python=platform.python_version(), numpy=np.__version__,
               scipy=scipy.__version__)
    try:
        import dynesty
        out["dynesty"] = dynesty.__version__
    except ImportError:
        out["dynesty"] = None
    def git(*x):
        try:
            r = subprocess.run(["git", "-C", str(codigo), *x], capture_output=True, text=True, timeout=60)
            return r.stdout.strip() if r.returncode == 0 else None
        except (OSError, subprocess.SubprocessError):
            return None
    out["commit"] = git("rev-parse", "HEAD") or None
    estado = git("status", "--porcelain", "--untracked-files=no") if out["commit"] else None
    out["cambios_sin_commit"] = None if estado is None else bool(estado)
    out["muestreo"] = dict(MUESTREO)
    if mod is not None:
        out["dir_rejilla"] = str(mod.dir_rejilla)
    out["raiz_datos"] = str(ROOT)
    if a is not None:
        out["argumentos"] = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(a).items()}
    out["fecha_utc"] = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    return out


def ajusta(a):
    from musepipe.config import load_run_config
    from scripts import tipo_ir_indices as tir
    degrada = getattr(a, "degrada", "cuadratura")
    mascara = getattr(a, "mascara_muse", "paper")
    vl, vh = getattr(a, "ventana_lineas", None), getattr(a, "ventana_halpha", None)
    rkf, td = getattr(a, "rk_fijo", None), getattr(a, "td_libre", None)
    quita, polvo = getattr(a, "quita_muse", None), getattr(a, "inyecta_polvo", None)
    nativo = getattr(a, "muse_nativo", False)
    ens_vr = (ENS_MAX_A, VR_MAX_KMS) if getattr(a, "ens_vr", False) else None
    ruido = getattr(a, "ruido_de", None)
    anclas = getattr(a, "anclas_ir", None)
    quita_ir = getattr(a, "quita_ir", None)
    pend = getattr(a, "pendiente_ir", None)
    una = bool(getattr(a, "banda_una_vez", False))
    et = etiqueta_de(a)
    # nunca sobre una corrida que ya está (una cadena tarda horas y otra con la misma etiqueta la taparía): se
    # comprueba antes de correr; para repetirla, se mueve o se borra aparte, o se usa otra --salida
    ya = [p.name for p in (a.salida / f"{et}.json", a.salida / f"{et}.npz") if p.exists()]
    if ya:
        raise SystemExit(f"{et}: ya existe en {a.salida} ({', '.join(ya)}); no se sobrescribe")
    a.salida.mkdir(parents=True, exist_ok=True)
    if a.modo == "evo_solo":
        cfg = load_run_config(tir.perfil(a.slug)["paper"]["run"], project_root=ROOT).config
        r = evo_solo(cfg, a.slug, a.tracks, semilla=a.semilla)
        out = dict(etiqueta=et, slug=a.slug, modo=a.modo, tracks=a.tracks,
                   fraccion_en_rejilla=r["fraccion_en_rejilla"],
                   **{k: percentiles(v) for k, v in r.items() if k != "fraccion_en_rejilla"},
                   procedencia=procedencia(a=a))
        np.savez(a.salida / f"{et}.npz", **{k: v for k, v in r.items() if k != "fraccion_en_rejilla"})
    else:
        mod = Modelo(a.slug, a.modo, a.av, a.tracks, a.inyecta_bt, a.plantilla, a.av_inyectada, a.semilla, a.av_fija,
                     a.rejilla, a.dir_rejilla, a.datos, a.r250, degrada, mascara_muse=mascara,
                     ventana_lineas=vl, ventana_halpha=vh, rk_fijo=rkf, td_rango=td, quita_muse=quita,
                     inyecta_polvo=polvo, **({"muse_nativo": True} if nativo else {}),
                     **({"ens_vr": ens_vr} if ens_vr is not None else {}),
                     **({"ruido_de": ruido} if ruido else {}),
                     **({"anclas_ir": anclas} if anclas else {}),
                     **({"quita_ir": quita_ir} if quita_ir else {}),
                     **({"pendiente_ir": pend} if pend is not None else {}),
                     **({"banda_una_vez": True} if una else {}))
        r = corre(mod, a.nlive, a.semilla)
        eq, der = r["muestras"], r["derivadas"]
        post = {k: percentiles(eq[:, i]) for i, k in enumerate(mod.nombres)}
        for i, k in enumerate(("teff", "logg", "r_cm", "masa_mjup", "lbol_lsun")):
            post[f"der_{k}"] = percentiles(der[:, i] / (RJUP_CM if k == "r_cm" else 1.0))
        out = dict(etiqueta=et, slug=a.slug, run=mod.run, modo=a.modo, av=a.av, av_fija=a.av_fija,
                   rejilla=a.rejilla, datos=a.datos, r_analisis=mod.r_analisis, nodos_interpolados=mod.rellenados,
                   # el degradado, solo en las rejillas muestreadas en número de onda (las demás no cambian)
                   **({"degrada": mod.degrada} if mod.degrada else {}),
                   # las máscaras telúricas de MUSE, solo si no son las del paper (las cadenas del paper no la llevan)
                   **({"mascara_muse": mascara} if mascara != "paper" else {}),
                   # las ventanas de las líneas en MUSE, solo si no son las del run
                   **({"ventana_lineas_kms": vl} if vl is not None else {}),
                   **({"ventana_halpha_kms": vh} if vh is not None else {}),
                   # las pruebas del disco y del D.5, solo si se piden (todas en `VARIANTES_JSON`)
                   **({"rk_fijo": mod.rk_fijo} if mod.rk_fijo is not None else {}),
                   **({"td_rango": list(mod.td_rango)} if mod.td_rango is not None else {}),
                   **({"quita_muse": quita} if quita is not None else {}),
                   **({"inyecta_polvo": [float(v) for v in polvo]} if polvo is not None else {}),
                   # MUSE a su resolución, solo si se pide (en `VARIANTES_JSON`)
                   **({"muse_nativo": True} if nativo else {}),
                   # el ensanchamiento y la velocidad libres: los topes de sus priores (en `VARIANTES_JSON`)
                   **({"ens_vr": list(ens_vr)} if ens_vr is not None else {}),
                   # de qué compañero es el ruido de los datos de prueba, si no del mismo (en `VARIANTES_JSON`)
                   **({"ruido_de": ruido} if ruido else {}),
                   # en qué bandas del IR va anclada la escala, si no en las que declara el objeto (en `VARIANTES_JSON`)
                   **({"anclas_ir": anclas} if anclas else {}),
                   # los segmentos del IR que salen de la verosimilitud, en orden alfabético (en `VARIANTES_JSON`)
                   **({"quita_ir": sorted(set(quita_ir))} if quita_ir else {}),
                   # el ancho del prior de la inclinación de cada segmento del IR (en `VARIANTES_JSON`)
                   **({"pendiente_ir": float(pend)} if pend is not None else {}),
                   # cada banda del IR una vez: el término de cada segmento por 1/N (en `VARIANTES_JSON`)
                   **({"banda_una_vez": True} if una else {}),
                   # los ejes con los que corrió (inicio, paso, n): una rejilla puede ampliarse después
                   ejes=dict(teff=[mod.T0, mod.dT, mod.nT], logg=[mod.G0, mod.dG, mod.nG]),
                   tracks=a.tracks if a.modo == "evo" else None, nlive=a.nlive, semilla=a.semilla,
                   ndim=mod.ndim, nombres=mod.nombres, segundos=r["segundos"], ncall=r["ncall"],
                   logz=r["logz"], logz_err=r["logz_err"], logl_max=r["logl_max"],
                   chi2nu_mejor=mod.chi2(r["mejor"]), chi2nu_mejor_inflado=mod.chi2(r["mejor"], inflado=True),
                   mejor=dict(zip(mod.nombres, map(float, r["mejor"]))),
                   verdad=getattr(mod, "verdad", None), posterior=post,
                   n_puntos={s["nombre"]: int(s["d"].size) for s in mod.segs},
                   procedencia=procedencia(mod, a))
        np.savez(a.salida / f"{et}.npz", muestras=eq, derivadas=der, nombres=np.array(mod.nombres))
    (a.salida / f"{et}.json").write_text(json.dumps(out, indent=1, ensure_ascii=False, default=float))
    t = out.get("posterior", {}).get("der_teff") or out.get("teff")
    print(f"{et}: T_eff p50 {t['p50']:.0f} [{t['p16']:.0f}, {t['p84']:.0f}]"
          + (f"  ({out['segundos']} s, {out['ncall']} llamadas, logZ {out['logz']:.1f})" if "logz" in out else ""))
    return 0


# ================================================================== resumen y margen cromosférico
def suelo(teff):
    """log(L_acc,noise/L_bol) de Manara et al. (2013, Ec. 2) a la T_eff dada."""
    return SUELO[0] * np.log10(teff) + SUELO[1]


def teff_umbral(log_lacc_lbol):
    """La T_eff a la que el suelo iguala un log(L_acc/L_bol) dado."""
    return 10 ** ((log_lacc_lbol - SUELO[1]) / SUELO[0])


def margen(teff, av, lbol_lsun, lacc0_lsun, av0, a, ahav):
    """log(L_acc/L_bol) − suelo(T_eff) por muestra. La L_acc publicada (con la A_V adoptada `av0`) se reescala a
    la A_V de la muestra: la L_Hα desenrojecida cambia 0.4·A_Hα/A_V dex por magnitud y la L_acc, `a` veces eso
    (la pendiente de la relación L_acc–L_Hα)."""
    lacc = lacc0_lsun * 10 ** (0.4 * a * ahav * (np.asarray(av, float) - av0))
    return np.log10(lacc / np.asarray(lbol_lsun, float)) - suelo(np.asarray(teff, float))


def umbral(slug, root=None):
    """(T*, log L_acc/L_bol, tipo): el umbral cromosférico con la L_acc del paper y la L_bol publicada. `root`: la
    raíz que contiene `runs/` (por defecto la del repo; desde un worktree, la del clon que los tiene)."""
    from scripts import suelo_cromosferico as sc
    from scripts import tipo_ir_indices as tir
    root = ROOT if root is None else Path(root)
    perfil = tir.perfil(slug, root=root)
    lacc, tipo = sc.lacc_hal(perfil["paper"]["run"], root=root)
    ratio = math.log10(lacc) - float(perfil["fisica"]["log_lbol"])
    return teff_umbral(ratio), ratio, tipo


def entrada_margen(slug, root=None):
    """L_acc publicada (con la A_V adoptada), A_V adoptada, pendiente de la relación y A_Hα/A_V del run del paper.
    `root`: la raíz que contiene `runs/` (por defecto la del repo; desde un worktree, la del clon que los tiene)."""
    from musepipe.config import load_run_config
    from musepipe.stages.stage_h03_limits import physical_inputs_from_config
    from scripts import suelo_cromosferico as sc
    from scripts import tipo_ir_indices as tir
    root = ROOT if root is None else Path(root)
    perfil = tir.perfil(slug, root=root)
    run = perfil["paper"]["run"]
    lacc, tipo = sc.lacc_hal(run, root=root)
    cfg = load_run_config(run, project_root=root).config
    phys = physical_inputs_from_config(cfg)
    return dict(lacc0=lacc, tipo=tipo, av0=float(cfg["h03_av"]), a=float(phys["lacc_lha_a"]),
                ahav=float(phys["a_halpha_over_av"]), lbol_lit=10 ** float(perfil["fisica"]["log_lbol"]))


def margen_de_corrida(d, e):
    """El margen por muestra de una corrida (`d`: su json con `_npz`; `e`: `entrada_margen`). En `evo_solo`, la
    A_V adoptada y la L_bol publicada."""
    z = np.load(d["_npz"])
    if d["modo"] == "evo_solo":
        teff = z["teff"]
        av = np.full(teff.size, e["av0"])
        lbol = np.full(teff.size, e["lbol_lit"])
    else:
        nombres = list(z["nombres"])
        teff, lbol = z["derivadas"][:, 0], z["derivadas"][:, 4]
        av = z["muestras"][:, nombres.index("av")]
    return margen(teff, av, lbol, e["lacc0"], e["av0"], e["a"], e["ahav"])


def _carga(dirs):
    out = []
    for dd in dirs:
        for f in sorted(glob.glob(str(Path(dd) / "*.json"))):
            d = json.loads(Path(f).read_text())
            z = Path(dd) / (Path(f).stem + ".npz")
            d["_npz"] = str(z) if z.exists() else None
            out.append(d)
    return out


def _teff_muestras(d):
    z = np.load(d["_npz"])
    return z["teff"] if d["modo"] == "evo_solo" else z["derivadas"][:, 0]


def _fmt(p, nd=0):
    if not p:
        return "—"
    f = f"{{:.{nd}f}}"
    return f"{f.format(p['p50'])} [{f.format(p['p16'])}, {f.format(p['p84'])}]"


def resume(dirs, salida=None):
    """Tablas en markdown (stdout) y, con `salida`, el json: corridas sobre los datos (T_eff, margen, log g, R,
    M, A_V, r_K, χ²_ν por banda, f), la calibración con plantillas, la sensibilidad del modo `evo` y las
    inyecciones."""
    todos = _carga(dirs)
    slugs = sorted({d["slug"] for d in todos})
    res = {"umbral": {}, "datos": [], "calibracion": [], "sensibilidad": [], "inyeccion": []}
    for slug in slugs:
        T, ratio, tipo = umbral(slug)
        res["umbral"][slug] = dict(teff_umbral=T, log_lacc_lbol=ratio, tipo=tipo)
    print("## Umbral cromosférico\n")
    for s, u in res["umbral"].items():
        print(f"- {s}: log L_acc/L_bol {'<' if u['tipo'] == 'upper_limit' else '='} {u['log_lacc_lbol']:.3f} "
              f"→ el suelo lo iguala a T* = {u['teff_umbral']:.0f} K")

    print("\n## Corridas sobre los datos\n")
    print("| objeto | modo | T_eff (K) | P(T<T*) | margen coherente (dex) | P(sobre el suelo) | log g | R (R_J) | M (M_J) | A_V | r_K | χ²_ν dato (vis/J/H/K) | f (vis/J/H/K) |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    entradas = {s: entrada_margen(s) for s in slugs}
    for d in todos:
        if d.get("plantilla") or "pl-" in d["etiqueta"] or "bt-" in d["etiqueta"]:
            continue
        t = _teff_muestras(d)
        T = res["umbral"][d["slug"]]["teff_umbral"]
        pT = float(np.mean(t < T))
        mg = margen_de_corrida(d, entradas[d["slug"]])
        mgp = np.percentile(mg[np.isfinite(mg)], [16, 50, 84])
        mtxt = f"{mgp[1]:+.2f} [{mgp[0]:+.2f}, {mgp[2]:+.2f}]"
        psobre = float(np.mean(mg > 0))
        modo = d["modo"] + (f" {d['tracks']}" if d.get("tracks") else "") + (
            (f", A_V fija {d['av_fija']:g}" if d.get("av_fija") is not None else f", A_V {d['av']}") if d["modo"] != "evo_solo" else "") + (
            f", solo {d['datos'].upper()}" if d.get("datos", "todo") != "todo" else "") + (
            f", máscara MUSE {d['mascara_muse']}" if d.get("mascara_muse", "paper") != "paper" else "") + (
            f", ventana {d['ventana_lineas_kms']:g} km/s" if d.get("ventana_lineas_kms") is not None else "") + (
            f", ventana Hα {d['ventana_halpha_kms']:g} km/s" if d.get("ventana_halpha_kms") is not None else "") + (
            f", r_K fija {d['rk_fijo']:g}" if d.get("rk_fijo") is not None else "") + (
            f", T_d libre {d['td_rango'][0]:g}–{d['td_rango'][1]:g} K" if d.get("td_rango") else "") + (
            f", MUSE sin {d['quita_muse']}" if d.get("quita_muse") else "")
        if d["modo"] == "evo_solo":
            fila = dict(slug=d["slug"], modo=modo, teff=d["teff"], p_bajo_umbral=pT, logg=d["logg"], r=d["r_rj"],
                        m=d["masa_mjup"], margen=mgp.tolist(), p_sobre_suelo=psobre)
            print(f"| {d['slug']} | {modo} (sin espectro) | {_fmt(d['teff'])} | {pT:.2f} | {mtxt} | {psobre:.2f} | {_fmt(d['logg'], 2)} | "
                  f"{_fmt(d['r_rj'], 2)} | {_fmt(d['masa_mjup'], 1)} | — | — | — | — |")
        else:
            P = d["posterior"]
            bandas = ["vis", "J", "H", "K"]
            chi = d["chi2nu_mejor"]
            # χ²_ν por banda: media de los segmentos de esa banda
            por_banda = {}
            for nom, v in chi.items():
                b = "vis" if nom == "MUSE" else nom.split("_")[-1]
                por_banda.setdefault(b, []).append(v)
            chis = "/".join(f"{np.mean(por_banda[b]):.1f}" if b in por_banda else "—" for b in bandas)
            fs = "/".join(f"{math.exp(P[f'lnf_{b}']['p50']):.3f}" if f"lnf_{b}" in P else "—" for b in bandas)
            fila = dict(slug=d["slug"], modo=modo, teff=P["der_teff"], p_bajo_umbral=pT, logg=P["der_logg"],
                        r=P["der_r_cm"], m=P.get("der_masa_mjup"), av=P["av"], rk=P.get("rk"), chi2=chi,
                        logz=d["logz"], segundos=d["segundos"], margen=mgp.tolist(), p_sobre_suelo=psobre,
                        av_fija=d.get("av_fija"), logl_max=d["logl_max"])
            print(f"| {d['slug']} | {modo} | {_fmt(P['der_teff'])} | {pT:.2f} | {mtxt} | {psobre:.2f} | {_fmt(P['der_logg'], 2)} | "
                  f"{_fmt(P['der_r_cm'], 2)} | {_fmt(P.get('der_masa_mjup'), 1)} | {_fmt(P['av'], 2)} | {_fmt(P.get('rk'), 2)} | "
                  f"{chis} | {fs} |")
        res["datos"].append(fila)

    print("\n## Calibración con plantillas (modo libre, A_V de la primaria inyectada y como prior)\n")
    print("| objeto | plantilla | tipo | T_eff ajustada (K) | log g | A_V (inyectada) | χ²_ν dato (vis/J/H/K) |")
    print("|---|---|---|---|---|---|---|")
    for d in todos:
        if d["modo"] != "libre" or not d.get("verdad") or "plantilla" not in (d.get("verdad") or {}):
            continue
        v, P = d["verdad"], d["posterior"]
        fila = dict(slug=d["slug"], plantilla=v["plantilla"], spt=v["spt"], codigo=spt_code(v["spt"]),
                    teff=P["der_teff"], logg=P["der_logg"], av=P["av"], av_inyectada=v["av"])
        res["calibracion"].append(fila)
    for f in sorted(res["calibracion"], key=lambda r: (r["slug"], r["codigo"])):
        print(f"| {f['slug']} | {f['plantilla']} | {f['spt']} | {_fmt(f['teff'])} | {_fmt(f['logg'], 2)} | "
              f"{_fmt(f['av'], 2)} ({f['av_inyectada']}) | |")

    print("\n## Sensibilidad: modo evolutivo sobre plantillas con el flujo del compañero\n")
    print("| objeto | plantilla | tipo | T_eff evo (K) | T_eff libre de la misma plantilla (K) |")
    print("|---|---|---|---|---|")
    libre = {(f["slug"], f["plantilla"]): f["teff"] for f in res["calibracion"]}
    for d in todos:
        if d["modo"] != "evo" or not d.get("verdad"):
            continue
        v, P = d["verdad"], d["posterior"]
        res["sensibilidad"].append(dict(slug=d["slug"], plantilla=v["plantilla"], spt=v["spt"], teff_evo=P["der_teff"],
                                        teff_libre=libre.get((d["slug"], v["plantilla"]))))
        print(f"| {d['slug']} | {v['plantilla']} | {v['spt']} | {_fmt(P['der_teff'])} | "
              f"{_fmt(libre.get((d['slug'], v['plantilla'])))} |")

    print("\n## Inyección de un modelo de la rejilla (maquinaria)\n")
    for d in todos:
        v = d.get("verdad") or {}
        if "teff" in v:
            P = d["posterior"]
            res["inyeccion"].append(dict(slug=d["slug"], verdad=v, teff=P["der_teff"], logg=P["der_logg"],
                                         r=P["r_rj"], av=P["av"]))
            print(f"- {d['slug']}: verdad {v} → T_eff {_fmt(P['der_teff'])}, log g {_fmt(P['der_logg'], 2)}, "
                  f"R {_fmt(P['r_rj'], 2)}, A_V {_fmt(P['av'], 2)}")
    if salida is not None:
        Path(salida).write_text(json.dumps(res, indent=1, ensure_ascii=False, default=float))
    return res


#: nombre de cada rejilla en las tablas y figuras del paper
NOMBRE_REJILLA = {"btsettl": "BT-Settl", "drift": "DRIFT-PHOENIX"}


def _pm(p, nd=0, f=None, signo=False):
    """`$m^{+a}_{-b}$` (LaTeX): la mediana y el intervalo 16–84 % de un dict de `percentiles`; `f` transforma los
    percentiles antes (monótona creciente, así que conserva el orden); `signo` escribe el + de la mediana."""
    f = f or (lambda x: x)
    m, lo, hi = (f(p[k]) for k in ("p50", "p16", "p84"))
    t = f"{{:.{nd}f}}"
    tm = f"{{:{'+' if signo else ''}.{nd}f}}"
    return f"${tm.format(m)}^{{+{t.format(hi - m)}}}_{{-{t.format(m - lo)}}}$"


def _es_variante(d):
    """¿Es la corrida (su json) una variante de las del paper? Lo es si alguna clave de `VARIANTES_JSON` no tiene el
    valor del paper, o si lleva una clave que este código no conoce (ni en `CLAVES_JSON` ni en `VARIANTES_JSON`; las
    que empiezan por `_` las añaden los cargadores). Es una lista BLANCA: una opción nueva, aunque nadie la añada
    aquí, deja su cadena fuera de las tablas del paper."""
    if any(d.get(k, v) != v for k, v in VARIANTES_JSON.items()):
        return True
    return any(not k.startswith("_") and k not in CLAVES_JSON and k not in VARIANTES_JSON for k in d)


def _de_la_tabla(d):
    """Una corrida sobre los datos (no plantilla ni inyección), `libre`, con la A_V fija o con el prior, con
    todos los espectros (no `--datos muse|ir`), con una de las rejillas del paper (`NOMBRE_REJILLA`), a R = 500 y
    sin ninguna variante (`_es_variante`: ni `--mascara-muse`, ni `--ventana-lineas` / `--ventana-halpha`, ni las
    pruebas del disco o del D.5, ni una opción que este código no conozca)."""
    return (d["modo"] == "libre" and d.get("av") == "primaria" and not d.get("verdad")
            and d.get("datos", "todo") == "todo" and d.get("rejilla", "btsettl") in NOMBRE_REJILLA
            and d.get("r_analisis", 500) == 500 and not _es_variante(d))


def filas_tabla(corridas, deltas, nombres):
    """El cuerpo LaTeX de la tabla de las cadenas del apéndice B: por compañero (en el orden de `nombres`,
    slug -> nombre en el texto), por rejilla y con A_V 1.8, 3.0 y con prior, la mediana y el 16–84 % de T_eff,
    log g, R, A_V, r_K, el error de modelo en MUSE y Δ, y ln Z relativo al mejor ajuste del compañero. `deltas`:
    etiqueta -> `percentiles` del margen. Solo las corridas `libre` con A_V fija o con prior; una A_V fija va
    sin intervalo; las plantillas e inyecciones (`verdad`) no entran."""
    filas = []
    for slug, nombre in nombres.items():
        suyas = [d for d in corridas if d["slug"] == slug and _de_la_tabla(d)]
        if not suyas:
            continue
        lnz_max = max(d["logz"] for d in suyas)
        filas.append(rf"\multicolumn{{9}}{{l}}{{{nombre}}} \\")
        orden = lambda d: (list(NOMBRE_REJILLA).index(d.get("rejilla", "btsettl")),
                           math.inf if d.get("av_fija") is None else d["av_fija"])
        for d in sorted(suyas, key=orden):
            P = d["posterior"]
            av = (f"${d['av_fija']:.1f}$" if d.get("av_fija") is not None else _pm(P["av"], 2))
            celdas = [NOMBRE_REJILLA[d.get("rejilla", "btsettl")], _pm(P["teff"]), _pm(P["logg"], 2),
                      _pm(P["r_rj"], 2), av, _pm(P["rk"], 2), _pm(P["lnf_vis"], 1, lambda x: 100 * math.exp(x)),
                      _pm(deltas[d["etiqueta"]], 2, signo=True), f"${d['logz'] - lnz_max:.1f}$"]
            filas.append("\\quad " + " & ".join(celdas) + r" \\")
    return filas


def tabla(dirs):
    """Imprime `filas_tabla` para las corridas de `dirs`, con el margen de cada muestra (como `resume`)."""
    from scripts import tipo_ir_indices as tir
    todos = [d for d in _carga(dirs) if _de_la_tabla(d)]
    slugs = sorted({d["slug"] for d in todos})
    nombres = {s: tir.perfil(s)["paper"]["display"].replace(" ", "~") for s in slugs}
    entradas = {s: entrada_margen(s) for s in slugs}
    deltas = {d["etiqueta"]: percentiles(margen_de_corrida(d, entradas[d["slug"]])) for d in todos}
    filas = filas_tabla(todos, deltas, nombres)
    print("\n".join(filas))
    return filas


#: las rejillas de la tabla de rejillas del apéndice B, en el orden de sus filas, con su nombre en LaTeX. De
#: Diamondback entra la ampliada desde 1600 K: la de 2000 K recortaba la cola de 42B b
#: (docs/2026-10-17_teff_mas_rejillas.md).
REJILLAS_TABLA = {"btsettl": "BT-Settl", "drift": "DRIFT-PHOENIX", "ames_dusty": "AMES-Dusty",
                  "ames_cond": "AMES-Cond", "bt_dusty": "BT-Dusty",
                  **{f"diamondback1600_{f}": ("Diamondback, no clouds" if f == "nc"
                                              else rf"Diamondback, \(f_{{\rm sed}} = {f[1:]}\)")
                     for f in ("f1", "f2", "f3", "f4", "f8", "nc")},
                  "sphinx": "SPHINX",
                  # la prueba de ATMO 2020 (2026-10-26): el paper la cita en un párrafo y NO en la tabla; sus filas
                  # solo salen si se pasan sus directorios (la propuesta de tabla con ATMO del agente `atmo`)
                  "atmo_ceq": "ATMO 2020", "atmo_ceq1500": "ATMO 2020 (1500--3000 K)",
                  "btsettl_bin": "BT-Settl (ATMO sampling)"}


def _de_rejillas(d):
    """Una corrida de la tabla de rejillas: como `_de_la_tabla`, pero con cualquier rejilla de `REJILLAS_TABLA` y
    a cualquier resolución (y sin ninguna variante, `_es_variante`)."""
    return (d["modo"] == "libre" and d.get("av") == "primaria" and not d.get("verdad")
            and d.get("datos", "todo") == "todo" and d.get("rejilla", "btsettl") in REJILLAS_TABLA
            and not _es_variante(d))


def en_el_borde(d):
    """Los parámetros ('teff', 'logg') cuyo 2.5 o 97.5 % cae a menos de medio paso del borde de su rejilla."""
    rej = REJILLAS[d.get("rejilla", "btsettl")]
    fuera = []
    for p in ("teff", "logg"):
        x0, paso, n = rej[p]
        q = d["posterior"][p]
        if q["p2_5"] - x0 < paso / 2 or x0 + paso * (n - 1) - q["p97_5"] < paso / 2:
            fuera.append(p)
    return fuera


def pares_que_encierran(plantillas, teff):
    """Los pares de plantillas consecutivas en tipo cuyas T_eff ajustadas encierran `teff` (la regla de
    `~/muse_noche_20261017/analiza.py`, `encierran`). `plantillas`: dicts con 'spt', 'codigo' (`spt_code`) y
    'teff', en el orden de desempate (el de sus etiquetas); se ordenan por 'codigo' de forma estable, así que dos
    plantillas del mismo tipo quedan en ese orden."""
    cc = sorted(plantillas, key=lambda c: c["codigo"])
    return [(a, b) for a, b in zip(cc, cc[1:]) if min(a["teff"], b["teff"]) <= teff <= max(a["teff"], b["teff"])]


def tipo_equivalente(plantillas, teff, rejilla=None):
    """El tipo espectral equivalente de un compañero: su T_eff (con el prior en A_V) entre las de plantillas de
    tipo conocido ajustadas con la MISMA rejilla y el ruido del compañero (Tabla B.5 y Sects. 4.1–4.2 del paper).

    - Un solo par de plantillas consecutivas la encierra (`pares_que_encierran`): el tipo es el par, `A–B`, o `A`
      si las dos son del mismo tipo. Si una plantilla del par está saturada contra un borde de la rejilla (a
      menos de medio paso, con `rejilla`), el par no ordena las T_eff: se lee la plantilla del par más cercana
      (DRIFT-PHOENIX con el ruido de ROXs 12 b, cuyas M6.5–M8 dan 2960–2990 K; `autor_9b.md` §2 del 2026-10-23).
    - Ningún par, y más fría que todas: `posterior` al tipo más tardío («≥L1»: tan fría como la plantilla más fría
      de ese tipo o más). Más caliente que todas: `anterior` al más temprano.
    - Más de un par: ambiguo (`ValueError`); el tipo no es una lectura única y hay que mirarlo.

    Devuelve dict(clase='par'|'una'|'posterior'|'anterior', tipos=(spt, ...), pares=[...], saturadas=[...])."""
    pares = pares_que_encierran(plantillas, teff)
    if len(pares) > 1:
        raise ValueError("tipo equivalente ambiguo: " + ", ".join(f"{a['spt']}–{b['spt']}" for a, b in pares))
    satura = []
    if rejilla is not None:
        x0, paso, n = REJILLAS[rejilla]["teff"]
        satura = [p for p in plantillas if p["teff"] - x0 < paso / 2 or x0 + paso * (n - 1) - p["teff"] < paso / 2]
    nombres = lambda ps: [p.get("plantilla", p["spt"]) for p in ps]  # noqa: E731
    out = dict(pares=[(a["spt"], b["spt"]) for a, b in pares], saturadas=nombres(satura))
    if pares:
        a, b = pares[0]
        if any(p is a or p is b for p in satura):
            cerca = min((a, b), key=lambda p: abs(p["teff"] - teff))
            return dict(out, clase="una", tipos=(cerca["spt"],))
        if a["spt"] == b["spt"]:
            return dict(out, clase="una", tipos=(a["spt"],))
        return dict(out, clase="par", tipos=(a["spt"], b["spt"]))
    if teff < min(p["teff"] for p in plantillas):
        ultimo = max(plantillas, key=lambda p: p["codigo"])["spt"]
        return dict(out, clase="posterior", tipos=(ultimo,))
    primero = min(plantillas, key=lambda p: p["codigo"])["spt"]
    return dict(out, clase="anterior", tipos=(primero,))


def filas_rejillas(corridas, deltas, nombres):
    """El cuerpo LaTeX de la tabla de rejillas del apéndice B: por compañero (orden de `nombres`) y por rejilla
    (orden de `REJILLAS_TABLA`, primero a R = 500 y después a R = 250), la mediana y el 16–84 % de T_eff, log g y
    A_V del ajuste con el prior en A_V, la mediana de Δ con A_V 1.8, 3.0 y con el prior, y el ln Z del ajuste con
    el prior menos el de BT-Settl a la misma resolución. Un valor con el 2.5 o el 97.5 % en el borde de su rejilla
    lleva `\\tablefootmark{a}`. `deltas`: etiqueta -> `percentiles` del margen. Lo que falta va como `\\ldots`."""
    filas = []
    for slug, nombre in nombres.items():
        suyas = {}
        for d in corridas:
            if d["slug"] != slug or not _de_rejillas(d):
                continue
            clave = (d.get("r_analisis", 500), d.get("rejilla", "btsettl"), d.get("av_fija"))
            if clave in suyas:
                raise SystemExit(f"dos corridas de {slug} con {clave}: {suyas[clave]['etiqueta']} y {d['etiqueta']}")
            suyas[clave] = d
        grupos = sorted({k[:2] for k in suyas}, key=lambda k: (-k[0], list(REJILLAS_TABLA).index(k[1])))
        if not grupos:
            continue
        filas.append(rf"\multicolumn{{8}}{{l}}{{{nombre}}} \\")
        for r, rejilla in grupos:
            prior = suyas.get((r, rejilla, None))
            ref = suyas.get((r, "btsettl", None))
            celdas = [REJILLAS_TABLA[rejilla] + (r" (\(R = 250\))" if r == 250 else "")]
            if prior is None:
                celdas += [r"\ldots"] * 3
            else:
                borde = en_el_borde(prior)
                marca = lambda p: r"\tablefootmark{a}" if p in borde else ""
                P = prior["posterior"]
                celdas += [_pm(P["teff"]) + marca("teff"), _pm(P["logg"], 2) + marca("logg"), _pm(P["av"], 2)]
            for av in (1.8, 3.0, None):
                d = suyas.get((r, rejilla, av))
                celdas.append(f"${deltas[d['etiqueta']]['p50']:+.2f}$" if d is not None else r"\ldots")
            celdas.append(f"${prior['logz'] - ref['logz']:.1f}$" if prior is not None and ref is not None
                          else r"\ldots")
            filas.append("\\quad " + " & ".join(celdas) + r" \\")
    return filas


def tabla_rejillas(dirs):
    """Imprime `filas_rejillas` para las corridas de `dirs`, con el margen de cada muestra (como `tabla`)."""
    from scripts import tipo_ir_indices as tir
    todos = [d for d in _carga(dirs) if _de_rejillas(d)]
    slugs = sorted({d["slug"] for d in todos})
    nombres = {s: tir.perfil(s)["paper"]["display"].replace(" ", "~") for s in slugs}
    entradas = {s: entrada_margen(s) for s in slugs}
    deltas = {d["etiqueta"]: percentiles(margen_de_corrida(d, entradas[d["slug"]])) for d in todos}
    filas = filas_rejillas(todos, deltas, nombres)
    print("\n".join(filas))
    return filas


def modelo_de_cadena(d, dir_rejilla=None):
    """El `Modelo` con el que corrió una cadena `libre` sobre los datos (su json: rejilla, datos, resolución,
    degradado, A_V, máscaras y ventanas de MUSE, las opciones del disco y del D.5, MUSE nativo con o sin
    ensanchamiento y velocidad, las anclas del IR, los segmentos del IR quitados, su inclinación y el peso de cada
    banda) y su punto de máxima verosimilitud
    (`mejor`). Es el único reconstructor: `delta_entre_rejillas.modelo_de` lo llama. `dir_rejilla`: el directorio de
    npz si no es el de `REJILLAS` (ATMO 2020)."""
    if d["modo"] != "libre" or d.get("verdad"):
        raise ValueError(f"{d['etiqueta']}: solo cadenas `libre` sobre los datos (sin plantilla ni inyección)")
    mod = Modelo(d["slug"], d["modo"], d["av"], "ATMO2020", None, None, None, d["semilla"], d.get("av_fija"),
                 d.get("rejilla", "btsettl"), dir_rejilla, d.get("datos", "todo"), d.get("r_analisis", 500) == 250,
                 d.get("degrada") or "cuadratura", mascara_muse=d.get("mascara_muse", "paper"),
                 ventana_lineas=d.get("ventana_lineas_kms"), ventana_halpha=d.get("ventana_halpha_kms"),
                 rk_fijo=d.get("rk_fijo"), td_rango=tuple(d["td_rango"]) if d.get("td_rango") else None,
                 quita_muse=d.get("quita_muse"), muse_nativo=bool(d.get("muse_nativo", False)),
                 ens_vr=tuple(d["ens_vr"]) if d.get("ens_vr") else None, ruido_de=d.get("ruido_de"),
                 anclas_ir=d.get("anclas_ir"), quita_ir=d.get("quita_ir"), pendiente_ir=d.get("pendiente_ir"),
                 banda_una_vez=bool(d.get("banda_una_vez", False)))
    return mod, np.array([d["mejor"][k] for k in mod.nombres])


def comprueba(dirs, dirs_rejilla=None):
    """Para cada cadena `libre` sobre los datos de `dirs`: reconstruye su modelo (`modelo_de_cadena`), evalúa
    `loglike` en su mejor punto y lo compara BIT A BIT (==) con el `logl_max` que guardó la cadena; compara
    también los puntos por segmento con `n_puntos`. Prueba que el código de hoy (y la variante de máscaras por
    defecto, `paper`) construye el mismo modelo que corrió la cadena. `dirs_rejilla`: {rejilla: directorio}."""
    filas = []
    for d in _carga(dirs):
        if d["modo"] != "libre" or d.get("verdad"):
            continue
        t0 = time.time()
        mod, th = modelo_de_cadena(d, (dirs_rejilla or {}).get(d.get("rejilla", "btsettl")))
        ll = mod.loglike(th)
        n = {s["nombre"]: int(s["d"].size) for s in mod.segs}
        filas.append(dict(etiqueta=d["etiqueta"], mascara_muse=d.get("mascara_muse", "paper"), loglike=ll,
                          logl_max=d["logl_max"], diferencia=ll - d["logl_max"], bit_a_bit=bool(ll == d["logl_max"]),
                          n_puntos=n, n_puntos_iguales=n == d["n_puntos"], segundos=round(time.time() - t0, 1)))
        f = filas[-1]
        print(f"{f['etiqueta']}: logL {ll!r} cadena {d['logl_max']!r} → {'IGUAL bit a bit' if f['bit_a_bit'] else 'DISTINTO'}"
              f" (Δ = {f['diferencia']:.3e}); puntos {'iguales' if f['n_puntos_iguales'] else 'DISTINTOS'} "
              f"(MUSE {n.get('MUSE')}); {f['segundos']} s")
    return filas


def bins_muse(slug, ventana_lineas=None, ventana_halpha=None):
    """{variante: dict(n, w_min, w_max, por_banda)} de los bins de 15 Å del espectro MUSE que entran en el ajuste con
    cada variante de `MASCARAS_MUSE` (`tipo_ir_plantillas.segmento_muse`, sin leer ninguna rejilla: con las rejillas
    de las tablas, todos los bins de MUSE caen dentro de los modelos). `por_banda`: los bins cuyo centro cae en cada
    banda telúrica de `musepipe.telluric_lines.TELLURIC_BANDS` (± medio bin). `ventana_lineas` / `ventana_halpha`:
    las ventanas de las líneas (km/s; None = las del run)."""
    from musepipe.telluric_lines import TELLURIC_BANDS
    from scripts import tipo_ir_indices as tir
    from scripts import tipo_ir_plantillas as tp
    run = tir.perfil(slug)["paper"]["run"]
    out = {}
    for m, v in MASCARAS_MUSE.items():
        s = tp.segmento_muse(run, h2o=v["h2o"], quita_o2=v["quita_o2"], ventana_kms=ventana_lineas,
                             ventana_ha_kms=ventana_halpha)
        w = s["w"][s["bueno"]]
        medio = float(np.median(s["hi"] - s["lo"])) / 2
        out[m] = dict(n=int(w.size), w_min=float(w.min()), w_max=float(w.max()),
                      por_banda={b["name"]: int(np.sum((w >= b["lo_A"] - medio) & (w <= b["hi_A"] + medio)))
                                 for b in TELLURIC_BANDS})
    return out


def fija_raiz(raiz):
    """Lee los runs y las bibliotecas de otro clon (`--project-root`), como `paper_figures.py --project-root`: un
    worktree no lleva `runs/`, y `g3_libraries_root` es relativo a la raíz. Cambia la raíz de este script y la de
    los que la usan para leer (`ajuste_conjunto`, `tipo_ir_plantillas`, `tipo_ir_indices`, `suelo_cromosferico`);
    el código y los `targets/` siguen siendo los de este clon. Devuelve la raíz resuelta."""
    global ROOT
    from scripts import ajuste_conjunto, suelo_cromosferico, tipo_ir_indices, tipo_ir_plantillas
    ROOT = Path(raiz).resolve()
    for m in (ajuste_conjunto, suelo_cromosferico, tipo_ir_indices, tipo_ir_plantillas):
        m.ROOT = ROOT
    return ROOT


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    comun = argparse.ArgumentParser(add_help=False)
    comun.add_argument("--project-root", type=Path, default=None,
                       help="clon del que se leen los runs y las bibliotecas (por defecto, este; un worktree no los lleva)")
    sub = ap.add_subparsers(dest="orden", required=True)
    aa = sub.add_parser("ajusta", help="una corrida de muestreo anidado (o `evo_solo`)", parents=[comun])
    aa.add_argument("slug")
    aa.add_argument("--modo", choices=("libre", "evo", "evo_solo"), default="libre")
    aa.add_argument("--datos", choices=DATOS, default="todo", help="todos los espectros, solo MUSE o solo J/H/K")
    aa.add_argument("--r250", action="store_true",
                    help="píxeles de a dos (λ/Δλ ≈ 250); automático con una rejilla de menos resolución (SPHINX)")
    aa.add_argument("--tracks", choices=tuple(TRACKS), default="ATMO2020")
    aa.add_argument("--av", choices=("libre", "primaria"), default="primaria")
    aa.add_argument("--nlive", type=int, default=500)
    aa.add_argument("--semilla", type=int, default=0)
    aa.add_argument("--inyecta-bt", type=float, nargs=4, metavar=("TEFF", "LOGG", "AV", "R_RJ"))
    aa.add_argument("--plantilla")
    aa.add_argument("--av-inyectada", type=float)
    aa.add_argument("--av-fija", type=float, help="perfil: A_V fija en este valor")
    aa.add_argument("--rejilla", choices=tuple(REJILLAS), default="btsettl")
    aa.add_argument("--dir-rejilla", type=Path, help="directorio de npz de la rejilla (por defecto el de REJILLAS)")
    aa.add_argument("--degrada", choices=DEGRADA, default="cuadratura",
                    help="rejillas muestreadas en número de onda (ATMO 2020): resta en cuadratura su resolución "
                         "(por defecto) o la trata como alta («ingenua»); en las demás no hace nada")
    aa.add_argument("--mascara-muse", choices=tuple(MASCARAS_MUSE), default="paper",
                    help="máscaras telúricas del espectro MUSE: paper (las de las cadenas del paper: G3 + H2O), g3 "
                         "(sin H2O), sin_o2 (con H2O, sin O2), sin_o2b (con H2O y O2 A, sin O2 B), sin_telurico "
                         "(sin ninguna); líneas y canales malos "
                         "siempre. Distinta de paper, la etiqueta lleva `mascara-<variante>`")
    aa.add_argument("--ventana-lineas", type=float, default=None, metavar="KMS",
                    help="ventana (±km/s) de TODAS las líneas de G2 en MUSE (por defecto la del run, 300); la "
                         "etiqueta lleva `ventana-<KMS>`")
    aa.add_argument("--ventana-halpha", type=float, default=None, metavar="KMS",
                    help="ventana (±km/s) de Hα en MUSE (por defecto la de las demás líneas); la etiqueta lleva "
                         "`ventanaha-<KMS>`")
    aa.add_argument("--rk-fijo", type=float, default=None, metavar="RK",
                    help="r_K fija en este valor (0 = el modelo sin disco) en vez de libre; la etiqueta lleva `rk<RK>`")
    aa.add_argument("--td-libre", type=float, nargs=2, default=None, metavar=("TMIN", "TMAX"),
                    help=f"la T del polvo libre (uniforme en TMIN–TMAX K) en vez de {T_D:g} K; la etiqueta lleva "
                         "`td<TMIN>-<TMAX>`")
    aa.add_argument("--quita-muse", choices=tuple(INTERVALOS_MUSE), default=None,
                    help="quita del espectro MUSE los bins de esos intervalos (`INTERVALOS_MUSE`; extremos = "
                         "6300–6900 y 9000–9350 Å, el D.5 del paper; azul y rojo, solo uno de los dos); la etiqueta "
                         "lleva `quita-<INTERVALOS>`")
    aa.add_argument("--inyecta-polvo", type=float, nargs=2, default=None, metavar=("RK", "TD"),
                    help="con --inyecta-bt: suma al modelo inyectado un exceso de polvo r_K a T_d (Ec. B.1); la "
                         "etiqueta lleva `polvo-<RK>-<TD>`")
    aa.add_argument("--muse-nativo", action="store_true",
                    help="el espectro MUSE canal a canal (1.25 Å) contra los modelos convolucionados con la LSF de "
                         "MUSE, en vez de los dos a R = 500 (el IR sigue a R = 500); solo con rejillas de paso fino "
                         "(BT-Settl, BT-Dusty, Diamondback); la etiqueta lleva `muse-nativo`")
    aa.add_argument("--ens-vr", action="store_true",
                    help=f"con --muse-nativo: ensanchamiento gaussiano extra del modelo de MUSE (FWHM 0–{ENS_MAX_A:g} Å, "
                         f"en cuadratura con la LSF) y velocidad radial (±{VR_MAX_KMS:g} km/s) libres; la etiqueta "
                         "lleva `ens-vr`")
    aa.add_argument("--ruido-de", default=None, metavar="SLUG",
                    help="con --plantilla o --inyecta-bt: los datos de prueba con la cobertura de este objeto y la "
                         "S/N por píxel de otro compañero (`snr_cruzada`); la etiqueta lleva `ruido-<SLUG>`")
    aa.add_argument("--anclas-ir", choices=tuple(ANCLAS_IR), default=None,
                    help="en qué bandas del IR va la escala anclada a la fotometría (las demás, libres), en vez de las "
                         "que declara el objeto; anclar una que no declara pide --plantilla o --inyecta-bt; la "
                         "etiqueta lleva `anclas-<BANDAS>`")
    aa.add_argument("--quita-ir", nargs="+", default=None, metavar="SEG",
                    help="saca de la verosimilitud esos segmentos del IR, por su nombre (NIFS_H, SINFONI_K…); los "
                         "datos de los demás no cambian; la etiqueta lleva `quitair-<SEGS>`")
    aa.add_argument("--pendiente-ir", type=float, default=None, metavar="ANCHO",
                    help="una inclinación libre por segmento del IR: su modelo × (1 + p·(λ − λ_c)/0.1 µm), con p "
                         "uniforme en ±ANCHO (p. ej. 0.1 = ±10 %% por 0.1 µm); la etiqueta lleva `pendir<ANCHO>`")
    aa.add_argument("--banda-una-vez", action="store_true",
                    help="cada banda del IR cuenta una vez: el término de cada segmento del IR × 1/N, con N los "
                         "segmentos de su banda (MUSE pesa 1); la etiqueta lleva `bandaunavez`")
    aa.add_argument("--etiqueta", default=None)
    aa.add_argument("--salida", type=Path, required=True)
    ar = sub.add_parser("resume", help="tablas y margen cromosférico de uno o varios directorios de `ajusta`",
                        parents=[comun])
    ar.add_argument("dirs", type=Path, nargs="+")
    ar.add_argument("--json", type=Path)
    at = sub.add_parser("tabla", help="cuerpo LaTeX de la tabla de las cadenas del apéndice B del paper",
                        parents=[comun])
    at.add_argument("dirs", type=Path, nargs="+")
    ag = sub.add_parser("rejillas", help="cuerpo LaTeX de la tabla de rejillas del apéndice B del paper",
                        parents=[comun])
    ag.add_argument("dirs", type=Path, nargs="+")
    ac = sub.add_parser("comprueba", parents=[comun],
                        help="reconstruye el modelo de cada cadena en su mejor punto: ¿da su logl_max bit a bit?")
    ac.add_argument("dirs", type=Path, nargs="+")
    ac.add_argument("--dir-rejilla", action="append", default=None, metavar="REJILLA=DIR",
                    help="directorio de npz de una rejilla que no está en la ruta de REJILLAS (repetible)")
    ac.add_argument("--json", type=Path)
    ab = sub.add_parser("bins", parents=[comun], help="bins de MUSE que entran en el ajuste con cada --mascara-muse")
    ab.add_argument("slugs", nargs="*", help="por defecto, los objetos del paper (targets/*.json con `paper`)")
    ab.add_argument("--ventana-lineas", type=float, default=None, metavar="KMS")
    ab.add_argument("--ventana-halpha", type=float, default=None, metavar="KMS")
    ab.add_argument("--json", type=Path)
    a = ap.parse_args(argv)
    if a.project_root is not None:
        fija_raiz(a.project_root)
    if a.orden == "ajusta":
        return ajusta(a)
    if a.orden == "comprueba":
        drj = dict(x.split("=", 1) for x in a.dir_rejilla or [])
        filas = comprueba(a.dirs, {k: Path(v).expanduser() for k, v in drj.items()})
        if a.json:
            a.json.write_text(json.dumps(filas, indent=1, ensure_ascii=False, default=float) + "\n")
        return 0 if filas and all(f["bit_a_bit"] and f["n_puntos_iguales"] for f in filas) else 1
    if a.orden == "bins":
        from scripts.halpha_criterios import objetos_del_paper
        res = {s: bins_muse(s, a.ventana_lineas, a.ventana_halpha)
               for s in (a.slugs or objetos_del_paper(Path(__file__).resolve().parents[1]))}
        for s, r in res.items():
            for m, v in r.items():
                bandas = ", ".join(f"{k} {n}" for k, n in v["por_banda"].items() if n)
                print(f"{s} {m}: {v['n']} bins en {v['w_min']:.1f}–{v['w_max']:.1f} Å" + (f" ({bandas})" if bandas else ""))
        if a.json:
            a.json.write_text(json.dumps(res, indent=1, ensure_ascii=False) + "\n")
        return 0
    if a.orden == "tabla":
        tabla(a.dirs)
        return 0
    if a.orden == "rejillas":
        tabla_rejillas(a.dirs)
        return 0
    resume(a.dirs, a.json)
    return 0


if __name__ == "__main__":
    sys.exit(main())
