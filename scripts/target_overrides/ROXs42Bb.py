"""Ajustes propios de ROXs 42B b: censo de fuentes del campo y mascara opcional.

## Por que este objeto lo necesita

El campo de ROXs 42B b tiene, segun la literatura, **una fuente de fondo
brillante en un costado**, citada a veces como ROXs 42B c. Cualquier medida que
promedie azimutalmente —el fondo de anillo de C2, el perfil de C3, la curva de
crecimiento— la mete dentro sin darse cuenta.

## Donde esta, y por que importa tanto

Es **ROXs 42B cc1** de Bryan et al. (2016), un objeto de **fondo** (no ligado:
por eso su separacion cambia entre epocas). Astrometria publicada:

    2011.4767   rho = 580 +- 1 mas   PA = 224.06 +- 0.06 deg
    2014.3644   rho = 525 +- 1 mas   PA = 227.27 +- 0.06 deg

Extrapolando **linealmente en (E, N)** —que es como se mueve un objeto de fondo,
no en coordenadas polares— a la epoca de estos datos (2022.6654):

    rho = 381 mas = 0.381"    PA = 241.4 deg    ->   r = 15.1 px

Comprobado sobre el cubo: en esa posicion hay un exceso de 2.4 sigma (azul) y
3.1 sigma (rojo) sobre la mediana azimutal al mismo radio, contra 0.3 y 1.2
sigma en un punto al azar de ese mismo anillo. Es compatible, no una deteccion:
a 15 px de una estrella brillante manda el speckle.

**Y esto explica lo de Moffat.** C1 ajusta la PSF con `psf_fit_radius_px = 28`
y solo enmascara al companero (que esta a 46.6 px). cc1 cae a 15.1 px, **dentro
del radio de ajuste y sin enmascarar**, sesgando los parametros del halo: es la
explicacion mas probable de que C1 eligiera `moffat` sobre `psfao` en este
objeto (residuo de anillo 8.38% vs 9.80%, empate a favor de moffat).

La cadena ya sabia manejarlo y nadie se lo habia dicho: B3 acepta
`stage01c_field_source_approx_yx`, lo mide, y C1 lo enmascara via `field_yx`.
El QC de B3 llevaba desde siempre el aviso "No field-source approximate position
configured". Declarado ahora en el config del run.
"""

from __future__ import annotations

from . import insert_after_markdown

#: Etapas cuyo notebook se modifica. Las demas salen igual que para cualquier objeto.
STAGES = {"C2", "C3"}

REASON = (
    "el campo tiene ROXs 42B cc1 (Bryan+2016), un objeto de FONDO a 0.381\" y PA 241.4 deg "
    "en la epoca de estos datos, que cae DENTRO del radio de ajuste de C1 y contamina toda "
    "medida azimutal; se anade el censo de fuentes del campo y la mascara"
)

#: ROXs 42B cc1 en el marco de B3 (cubo recortado 170x170). Derivado de
#: Bryan+2016 extrapolado a 2022.6654; ver el docstring. El enmascarado de
#: verdad lo hace C1 via `stage01c_field_source_approx_yx` en el config del run
#: -- esto es solo para las celdas del notebook.
FIELD_SOURCES_YX: list[tuple[float, float]] = [(77.7, 98.3)]


#: Como se llama el cubo crudo en el notebook de cada etapa. NO es el mismo
#: nombre: C2 lo llama `CUBE` y C3 `LS_CUBE` (tiene dos, y este es del que
#: extrae `optimal_ls`). Esta celda se escribio mirando C2 y se aplicaba tal cual
#: a C3, donde `CUBE` no existe: el notebook de C3 de ROXs 42B b reventaba con
#: `NameError` y no lo veia nadie porque la suite solo ejecutaba los de ROXs 12 b.
CUBO_POR_ETAPA = {"C2": "CUBE", "C3": "LS_CUBE"}


def apply(stage_id, cells, ctx):
    md, code = ctx["md"], ctx["code"]
    seccion = "## 5 " if stage_id == "C2" else "## 3 "
    cubo = CUBO_POR_ETAPA[stage_id]
    return insert_after_markdown(
        cells,
        seccion,
        md(
            "## · Censo de fuentes del campo (propio de ROXs 42B b)\n\n"
            "Este objeto tiene una **fuente de fondo brillante en un costado** (la literatura la "
            "cita a veces como ROXs 42B c). Importa porque **toda medida azimutal la promedia "
            "dentro sin avisar**: el fondo de anillo de C2, el perfil radial de C3 y la curva de "
            "crecimiento de la primaria.\n\n"
            "La celda hace un pasa-altos sobre la luz blanca —que quita el halo suave y deja lo "
            "puntual— y lista lo que encuentra fuera de la primaria y del companero, con su color "
            "rojo/azul. Un color uniforme alrededor de 2-3 en fuentes pequenas y repartidas "
            "simetricamente es **speckle**, no fuente real.\n\n"
            "> **Ojo con el campo.** El cubo de este objeto esta recortado a 200 px = **5.04\"** "
            "por `stream_combine.DEFAULT_CROP_NPIX`, contra los 8.3\" de ROXs 12 b. Si la fuente "
            "de fondo esta mas lejos, **no aparece aqui**, y su ausencia en esta lista no prueba "
            "que no exista. Para enmascararla, rellena `FUENTES_CAMPO` abajo."
        ),
        code(
            "from scipy import ndimage\n\n"
            "# Posicion EN EL MARCO DEL CUBO DE LA ETAPA (170x170 de B1).\n"
            "# ROXs 42B cc1 (Bryan+2016) extrapolado a 2022.6654: 0.381\", PA 241.4 deg.\n"
            "FUENTES_CAMPO = [(77.7, 98.3)]\n"
            "RADIO_MASCARA_PX = 6.0\n\n"
            "_luz = np.nanmedian(" + cubo + "[::20], axis=0)\n"
            "_hp = _luz - ndimage.median_filter(np.nan_to_num(_luz), size=15)\n"
            "_sig = 1.4826 * np.nanmedian(np.abs(_hp - np.nanmedian(_hp)))\n"
            "_yy, _xx = np.indices(_luz.shape, dtype=float)\n"
            "_r = np.hypot(_yy - STAR_YX[0], _xx - STAR_YX[1])\n"
            "_rc = np.hypot(_yy - OBJECT_YX[0], _xx - OBJECT_YX[1])\n"
            "_lab, _n = ndimage.label((_r > 15) & (_rc > 6) & (_hp > 8 * _sig))\n"
            "_azul = np.nanmedian(" + cubo + "[WAVE < 6000][::10], axis=0)\n"
            "_rojo = np.nanmedian(" + cubo + "[WAVE > 8000][::10], axis=0)\n"
            "print(f'sigma del pasa-altos = {_sig:.4f} | fuentes a >8 sigma, sin primaria ni companero:')\n"
            "print(f\"  {'y':>7s} {'x':>7s} {'r[px]':>7s} {'pico/s':>8s} {'rojo/azul':>10s} {'n_px':>6s}\")\n"
            "_filas = []\n"
            "for _i in range(1, _n + 1):\n"
            "    _sel = _lab == _i\n"
            "    if _sel.sum() < 4:\n"
            "        continue\n"
            "    _w = _hp * _sel; _t = _w.sum()\n"
            "    _my = float((_yy * _w).sum() / _t); _mx = float((_xx * _w).sum() / _t)\n"
            "    _iy, _ix = int(round(_my)), int(round(_mx))\n"
            "    _b = float(np.nansum(_azul[_iy - 1:_iy + 2, _ix - 1:_ix + 2]))\n"
            "    _rj = float(np.nansum(_rojo[_iy - 1:_iy + 2, _ix - 1:_ix + 2]))\n"
            "    _filas.append((float(np.nanmax(_hp[_sel])) / _sig, _my, _mx,\n"
            "                   float(np.hypot(_my - STAR_YX[0], _mx - STAR_YX[1])),\n"
            "                   _rj / _b if _b > 0 else np.nan, int(_sel.sum())))\n"
            "for _pk, _my, _mx, _rr, _col, _npx in sorted(_filas, reverse=True)[:10]:\n"
            "    print(f'  {_my:7.1f} {_mx:7.1f} {_rr:7.1f} {_pk:8.1f} {_col:10.2f} {_npx:6d}')\n"
            "if not _filas:\n"
            "    print('  (ninguna)')\n"
            "print(f'\\ncampo del cubo: {" + cubo + ".shape[1]}x{" + cubo + ".shape[2]} px'\n"
            "      f' = {" + cubo + ".shape[1] * 0.0252:.2f}\" — una fuente mas lejos NO saldria aqui')\n\n"
            "if FUENTES_CAMPO:\n"
            "    _m = np.zeros(_luz.shape, dtype=bool)\n"
            "    for _fy, _fx in FUENTES_CAMPO:\n"
            "        _m |= np.hypot(_yy - _fy, _xx - _fx) <= RADIO_MASCARA_PX\n"
            "    " + cubo + " = np.where(_m[None, :, :], np.nan, " + cubo + ")\n"
            "    print(f'\\nENMASCARADAS {len(FUENTES_CAMPO)} fuentes ({int(_m.sum())} spaxels):'\n"
            "          ' el resto del notebook usa el cubo ya limpio.')\n"
            "else:\n"
            "    print('\\nsin mascara aplicada: el cubo sigue intacto.')"
        ),
    )
