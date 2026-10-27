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
