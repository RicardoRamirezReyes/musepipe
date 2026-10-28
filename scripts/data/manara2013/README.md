# Manara et al. (2013): tablas de VizieR J/A+A/551/A107

**Cita.** Manara, C. F., Testi, L., Rigliaco, E., Alcalá, J. M., Natta, A., Stelzer, B., et al. 2013,
A&A, 551, A107, «X-shooter spectroscopy of young stellar objects. II. Impact of chromospheric emission on
accretion rate estimates». DOI [10.1051/0004-6361/201220921](https://doi.org/10.1051/0004-6361/201220921);
arXiv:1301.3058.

**De dónde salen.** Catálogo de VizieR `J/A+A/551/A107` (CDS, <https://cdsarc.cds.unistra.fr/viz-bin/cat/J/A+A/551/A107>),
descargado el 2026-10-26 por el agente `halpha` (`~/muse_sesion_20261026/claudio/halpha/literatura/vizier_manara2013/`).
Los ficheros son los de VizieR **sin tocar**; `ReadMe.txt` es el del catálogo y da el formato byte a byte. El subagente
de literatura comprobó que las tablas coinciden número a número con las del texto de arXiv
(`literatura/VERIFICACION.md` de la sesión, §1).

| fichero | qué tiene | lo usa |
|---|---|---|
| `table1.dat` | parámetros de la literatura (distancia en bytes 74–76) | `halpha_criterios.py cromosfera` |
| `table3.dat` | SpT, T_eff, log L★, M★ y ⟨log L_acc,noise⟩ (media de líneas) | idem |
| `table6.dat` | flujos y EW de las líneas de Balmer (F(Hα) en bytes 12–17, 1e-14 erg s⁻¹ cm⁻²) | idem |
| `table7.dat` | flujos y EW de He I y Ca II | (no se usa; para el criterio de otras líneas) |
| `refs.dat` | referencias de `table1.dat` | — |

sha256 (de la descarga):

```
136c81f122e46030d10b67743a419f10e888ff0cda1a6ad5ebb66859a372c007  ReadMe.txt
6946e8622e9180338066ec1d03f8d2f0726d1b4824faa1321001a0a98756b4d4  refs.dat
524abcae95b351c3dca459b97a05986b6cf8e0959d1cb5fa3fffd75a2c160a0f  table1.dat
f40ddbef8b7d02fb1d143c77a029bd143cfcac8ac701d7bdaf5ad840a59a9252  table3.dat
93dee2c1affe9308a4cee0371a4eb9c08105f04522852d74ec40e6092593d99f  table6.dat
ddc742a51e843adcb71a117f500d292ec193140ef5ee332433124c8f81df09ab  table7.dat
```

**Qué se deriva (no está en el artículo).** La dispersión de las 22 estrellas Class III del ajuste de su Ec. 2
(las 24 menos TWA 9A y Sz 122, excluidas en su Fig. 13) alrededor de esa ecuación, en la misma magnitud que compara el
paper: L_acc de la Hα sola con la relación de Alcalá et al. (2017), L_Hα = 4π d² F(Hα). Sale +0.03 dex de media y
0.25 dex de dispersión (apéndice B del paper). Lo calcula `scripts/halpha_criterios.py cromosfera`.

## `clase3_m9.json`: TWA 26 y TWA 29

**Qué hay.** `clase3_m9.json`: TWA 26 (M9) y TWA 29 (M9.5), las dos Clase III más tardías de la muestra, con lo
que hace falta para poner sus líneas en unidades de la luminosidad bolométrica, log(L_line/L*):

| dato | tabla del artículo | TWA 26 | TWA 29 |
|---|---|---|---|
| distancia D (pc) | Tabla 1 | 42 | 79 |
| log(L*/L☉), T_eff (K) | Tabla 3 | −2.70, 2400 | −2.81, 2330 |
| F(Hα) (10⁻¹⁴ erg s⁻¹ cm⁻²) | Tabla 6 | 0.052 ± 0.008 | 0.019 ± 0.003 |
| F(Ca II 849.9, 854.2, 866.2) (erg s⁻¹ cm⁻²) | Tabla 7 | < 1.10e-16, < 1.69e-16, < 8.50e-17 | < 2.59e-17, < 4.37e-17, < 2.73e-17 |

Los «<» son límites superiores a 3σ de los autores (texto de las Figs. 10–11). Los flujos son los de las tablas y
el script no los desenrojece, como no lo hacía el agente (son estrellas de la asociación TW Hya, a 42 y 79 pc).
log(L_line/L*) no depende de la distancia mientras L* se calcule con la misma.

**De dónde salen.** Copiados a mano de la versión arXiv (1301.3058v1) por el agente `lineas` el 2026-10-26
(`~/muse_sesion_20261026/claudio/lineas/scripts/tablas.py`, `MANARA13`) y comprobados otra vez contra el texto
extraído de ese PDF (`~/muse_sesion_20261026/claudio/lineas/lit/manara2013/manara2013.txt`, filas TWA26/TWA29 de
las Tablas 1, 3, 6 y 7) al pasarlos aquí. No se comparó con el PDF final de la revista.

**Quién lo usa.** `scripts/otras_lineas_lacc.py`: la comparación del triplete del Ca II de los compañeros con estas
dos estrellas en L_line/L_bol (paper, Sect. 4.1: «0.1–0.6 dex deeper»).

**Coherencia con las tablas de VizieR.** Los valores de `clase3_m9.json` son las filas TWA26 y TWA29 de
`table1.dat`, `table3.dat`, `table6.dat` y `table7.dat` de esta misma carpeta (comprobado al juntar las dos ramas,
2026-10-26). Si algún día difieren, mandan las tablas de VizieR.
