# Literatura de las cotas del disco circumplanetario

Entradas publicadas que usa `scripts/disco_cotas.py` (Sect. 5.1 del paper y el párrafo del factor magnetosférico en
la sección de métodos). Las medidas de este trabajo (L_bol, Ṁ, r_K del CO, masas, radios, distancias, separaciones,
A_V y las cadenas de la Tabla B.6) **no** están aquí: el script las lee de los productos, del config del run, de
`targets/<slug>.json` y de las cadenas de `scripts/teff_bayes.py`.

`disco_literatura.csv` tiene una fila por número. Columnas: `clave`, `objeto` (slug de `targets/`, o `*`),
`valor` (admite fracciones, `1/3`), `err_mas`, `err_menos`, `unidad`, `fuente` (etiqueta de la tabla de abajo) y
`ubicacion` (tabla, ecuación o sección del artículo). Las cifras se comprobaron el 2026-10-26 contra el texto
completo de cada artículo (arXiv → `pdftotext`, en `~/muse_sesion_20261026/claudio/disco/lit/` y
`~/muse_sesion_20261026/claudio/disco_parrafo/lit/`), salvo donde se dice otra cosa. Los DOI son los de
`paper/refs.bib` o, si el paper no cita la obra, los resueltos en Crossref ese mismo día.

| fuente | referencia | DOI | filas y de dónde salen |
|---|---|---|---|
| Ayliffe2009 | Ayliffe & Bate 2009, MNRAS 397, 657 | 10.1111/j.1365-2966.2009.15002.x | `fraccion_hill_min` = 1/3: abstract y Sect. 4.2 («protoplanets of one third of the Hill radius (Quillen & Trilling 1998)»). Quillen & Trilling 1998 no se abrió. |
| MartinLubow2011 | Martin & Lubow 2011, MNRAS 413, 1447 | 10.1111/j.1365-2966.2011.18228.x | `fraccion_hill_max` = 0.4: abstract («approximately 0.4 r_H»). |
| DullemondMonnier2010 | Dullemond & Monnier 2010, ARA&A 48, 205 | 10.1146/annurev-astro-081309-130932 | `cobertura_pared` = 0.2: Sect. 3.5, la pared redondeada de AB Aur, «a covering fraction of about 20%, i.e. producing an inclination-averaged NIR luminosity of L_NIR ≃ 0.2 L∗». Es un ejemplo de estrella Herbig, no una cota general. |
| Hartmann2016 | Hartmann, Herczeg & Calvet 2016, ARA&A 54, 135 | 10.1146/annurev-astro-081915-023347 | `xi_magnetosfera` = 0.7: Sect. 2.2, Ec. 1, r_M ≈ ξ (μ⁴/4GMṀ²)^{1/7}, «Adopting values of ξ ≈ 0.7». |
| GuverOzel2009 | Güver & Özel 2009, MNRAS 400, 2050 | 10.1111/j.1365-2966.2009.15598.x (Crossref) | `nh_por_av`: Ec. 1, N_H = (2.21 ± 0.09) × 10²¹ A_V cm⁻². |
| Wu2020 | Wu et al. 2020, AJ 159, 229 | 10.3847/1538-3881/ab818c | `alma_flujo` (Tabla 3): 0.098 ± 0.071 mJy (ROXs 12 b) y −0.002 ± 0.043 mJy (ROXs 42B b). El script usa 3σ (0.213 y 0.129 mJy), la definición del texto de la Sect. 4.1 («The 3σ upper limits … span 120–210 µJy»). `alma_lambda` = 0.88 mm. |
| Kraus2014 | Kraus et al. 2014, ApJ 781, 20 | 10.1088/0004-637X/781/1/20 | `contraste_K` 5.06 ± 0.01 y `contraste_Lp` 4.12 ± 0.01: Tabla 3, ROXs 12, NIRC2, MJD 55735 (la misma noche). `mbol_primaria` 5.74 ± 0.26 y `d_mbol_primaria` 120 pc: Tabla 1, ROXs 42B. `lambda_Lp` 3.78 µm: Sect. 2.3 («an MKO L′ filter centered at 3.78 µm (3.4–4.1 µm)»). |
| Bowler2017 | Bowler et al. 2017, AJ 154, 165 | 10.3847/1538-3881/aa88bd | Tabla 2, columna ROXs 12 A: Ks 9.099 ± 0.025, W1 8.805 ± 0.023, W2 8.712 ± 0.021, W3 8.393 ± 0.042, W4 6.558 ± 0.089; masa 0.65 (+0.05/−0.09) M☉; log L_bol −0.57 ± 0.06 a 137 ± 10 pc (nota e); K−L′ = 0.0 ± 0.1 supuesto para la primaria (nota b). |
| PecautMamajek2013 | Pecaut & Mamajek 2013, ApJS 208, 9 | 10.1088/0067-0049/208/1/9 (Crossref); VizieR J/ApJS/208/9 (doi:10.26093/cds/vizier.22080009) | Tabla 6 (colores intrínsecos de estrellas de 5–30 Myr), fila M0: Ks−W1 0.11, Ks−W2 0.17, Ks−W3 0.25, Ks−W4 0.38. Descargada de VizieR el 2026-10-26 (`asu-tsv`, `-source=J/ApJS/208/9/table6`). Las dos primeras son las que cita el paper a través de Daemgen et al. 2017 (A&A 601, A65, Sect. 3.1: «expected Ks−W1 = 0.11 ± 0.01 mag and Ks−W2 = 0.17 ± 0.01 mag»). |
| Bryan2016 | Bryan et al. 2016, ApJ 827, 100 | 10.3847/0004-637X/827/2/100 | `e_max95`: abstract, «95% upper limits on the companions' eccentricities of 0.58 and 0.83» (ROXs 42B b y ROXs 12 b). |
| Inglis2026 | Inglis et al. 2026, AJ 171, 280 | 10.3847/1538-3881/ae4b34 | `masa_primaria` 1.35 ± 0.07 M☉ (Tabla 2, masa total de la binaria), `a_orbita` 222 (+134/−62) au (Sect. 3.2; la Tabla 3 da 223 +150/−69) y `e_orbita` 0.4 (+0.2/−0.3) (Tabla 3). Leído del PDF de la revista (CaltechAUTHORS) en la verificación del 2026-10-13 (`~/muse_sesion_20261013/verificacion_literatura.md`, fila 5d); no hay preprint en arXiv. |
| Xuan2024 | Xuan et al. 2024, ApJ 970, 71 | 10.3847/1538-4357/ad4796 | `vsini` de KPIC: Tabla 5, 3.6 (+1.2/−1.6) km/s (ROXs 12 b) y 4.4 (+1.6/−2.1) km/s (ROXs 42B b; el texto de la Sect. 4.4 dice 4.2 +1.8/−2.2). |
| Grasser2026 | Grasser et al. 2026, A&A 709, A141 | 10.1051/0004-6361/202659035 | `vsini` de CRIRES⁺: Tabla 3, ROXs 12B, 2.14 (+0.21/−0.16) km/s. |
| Bryan2020a | Bryan et al. 2020a, ApJ 905, 37 (así en la bibliografía de Xuan+2024) | 10.3847/1538-4357/abc0ef (de Xuan+2024) | `vsini` 8.4 (+2.1/−1.4) km/s de ROXs 12 b: Xuan+2024 Tabla 5, ref. 4. No se abrió el original; no está en `refs.bib` del paper. |
| Bryan2018 | Bryan et al. 2018, Nature Astronomy 2, 138 (así en la bibliografía de Xuan+2024) | 10.1038/s41550-017-0325-8 (de Xuan+2024) | `vsini` 9.5 (+2.1/−2.3) km/s de ROXs 42B b: Xuan+2024 Sect. 4.4 (texto). No se abrió el original. |
| Inglis2023 | Inglis, Wallack & Xuan 2023, arXiv:2312.02297 (así en la bibliografía de Xuan+2024) | — | `vsini` 10.5 ± 0.9 km/s de ROXs 42B b: Xuan+2024 Tabla 5, ref. 8. No se abrió el original. |
| Wright2010 | Wright et al. 2010, AJ 140, 1868 | 10.1088/0004-6256/140/6/1868 | Sect. 2.2: longitudes de onda isofotas 3.3526, 4.6028, 11.5608 y 22.0883 µm, y F_ν de magnitud cero 29.0448 Jy (W3) y 8.2839 Jy (W4). |
| IAU2015B2 | Resolución B2 de la IAU 2015 (Mamajek et al. 2015, arXiv:1510.06262) | — | `mbol_sol` = 4.74 mag, para pasar la M_bol de Kraus+2014 a luminosidad. |

Lo que **no** es literatura y vive como constante con nombre en el script, marcado como supuesto: las bandas de caja
(K 2.00–2.30 µm, L′ 3.42–4.12 µm), Vega como cuerpo negro de 9550 K, la ley de extinción IR de Cardelli et al.
(1989) extrapolada más allá de 3.3 µm, la órbita de ROXs 12 b sin ajuste publicado (a = ρ·U(1, 1.4), e ~ U(0, 0.5)),
los recortes del Montecarlo, el ángulo rasante del disco abocinado (0.02–0.05), la rejilla de campos, temperaturas,
masas de disco y H/r, y la semilla.
