# West et al. (2011): enanas M de campo de SDSS DR7, tipos M7, M8 y M9 (VizieR J/AJ/141/97)

**Cita.** West, A. A., Morgan, D. P., Bochanski, J. J., et al. 2011, AJ, 141, 97, «The Sloan Digital Sky Survey
DR7 Spectroscopic M Dwarf Catalog. I. Data». DOI
[10.1088/0004-6256/141/3/97](https://doi.org/10.1088/0004-6256/141/3/97); arXiv:1101.1082.

**De dónde salen.** Catálogo de VizieR `J/AJ/141/97/catalog`, descargado el 2026-10-26 (04:51 UTC) por el agente
`halpha` (`~/muse_sesion_20261026/claudio/halpha/literatura/vizier_west2011/`) con una consulta ASU-TSV por tipo
espectral (`SpT=7`, `8`, `9`) y las columnas `Plate, MJD, Fiber, SpT, W(Ha), e_W(Ha), HaS/N, Ha/c, MHa, LHa/b, Z`. La
URL exacta de cada consulta está en la cabecera de su fichero (`#INFO request=...`). Los ficheros son los de VizieR
**sin tocar**.

- `W(Ha)`: EW de Hα en Å, **positiva en emisión** (Sect. 2 del artículo), integrada en una ventana de 8 Å, R ≈ 2000.
- `MHa`: 1 = activa (EW > 0.75 Å, EW > 3σ, S/N del continuo > 3 y pico > 3× el ruido), 0 = inactiva, vacío = sin
  determinar.

sha256 (de la descarga):

```
e31c48c5d7fad26b9729536f63437f744e90af09b5961c5aaa1a0b323c703ce5  west2011_M7.tsv
2138de0bcb40c5adeabeff2f43c7d8c137dbad723cf31fb8924d5f4ea126c3fb  west2011_M8.tsv
38c8ffdb7c780ed04bf2124703e518f5233567f7e2cd49f6df49eb7484d386b5  west2011_M9.tsv
```

**Qué se deriva.** La fracción de las activas con EW ≥ 12, 20, 26.2, 40 y 53 Å (p. ej., el 2.1 % de las M8 activas
alcanza la EW de ROXs 12 b). No está en el artículo ni en el paper; la da `scripts/halpha_criterios.py cromosfera`
como contexto del criterio de EW. Son espectros de una época de enanas viejas de campo: las fulguraciones van dentro.
