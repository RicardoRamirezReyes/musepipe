# Manara et al. (2013): las Clase III M9 y M9.5

**Cita.** Manara, C. F., Testi, L., Rigliaco, E., et al. 2013, *X-shooter observations of young stellar objects
without disks: the chromospheric noise in accretion estimates*, A&A, 551, A107.
DOI [10.1051/0004-6361/201220921](https://doi.org/10.1051/0004-6361/201220921); arXiv:1301.3058v1.

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
