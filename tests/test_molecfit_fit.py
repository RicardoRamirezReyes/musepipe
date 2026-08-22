"""La via molecfit de A3: la escalera, las firmas, y las dos trampas del marco.

La regla que estos tests protegen es una sola, y es la del encargo: **una
no-convergencia nunca es motivo para descartar molecfit**. Es un defecto de
parametros o de configuracion del dato, se diagnostica por su firma y se sube un
peldano; si la escalera se agota, la etapa PARA. No existe ninguna via por la que
un fallo de molecfit elija STD_TELLURIC.
"""

import tempfile
import unittest
from pathlib import Path

import numpy as np
from astropy.io import fits

from musepipe.reduction import molecfit as mf
from musepipe.reduction.telluric import TelluricError


def escribe_cubo(path, *, nz=600, npix=41, airm=(1.10, 1.20), exptime=300.0, banda_pct=0.0):
    """Un cubo minimo con una primaria en el centro y condiciones declaradas.

    `banda_pct` hunde la banda A de O2 (7590-7700) ese tanto por ciento, que es
    lo que decide si esa ventana entra al ajuste o no.
    """

    wave = 6800.0 + np.arange(nz) * 2.5
    espectro = np.ones(nz)
    if banda_pct:
        espectro[(wave >= 7590.0) & (wave <= 7700.0)] = 1.0 - banda_pct / 100.0
    yy, xx = np.mgrid[0:npix, 0:npix]
    psf = np.exp(-((yy - npix // 2) ** 2 + (xx - npix // 2) ** 2) / 8.0)
    cube = (espectro[:, None, None] * psf[None, :, :]).astype(np.float32)
    cabecera = fits.Header()
    cabecera["CRVAL3"] = float(wave[0])
    cabecera["CDELT3"] = 2.5
    cabecera["CRPIX3"] = 1.0
    pri = fits.Header()
    pri["DATE-OBS"] = "2022-08-29T23:22:03.446"
    pri["EXPTIME"] = exptime
    pri["MJD-OBS"] = 59820.0
    if airm is not None:
        pri["HIERARCH ESO TEL AIRM START"] = airm[0]
        pri["HIERARCH ESO TEL AIRM END"] = airm[1]
    fits.HDUList([
        fits.PrimaryHDU(header=pri),
        fits.ImageHDU(data=cube, header=cabecera, name="DATA"),
    ]).writeto(path, overwrite=True)
    return wave


def escribe_best_fit(path, **valores):
    """Un `BEST_FIT_PARAMETERS` de mentira, con las filas de relleno que trae el de verdad."""

    base = {"status": (2.0, -1.0), "initial_chi2": (4585.0, -1.0), "best_chi2": (2453.0, -1.0),
            "reduced_chi2": (15.7, -1.0), "rms_rel_to_err": (3.97, -1.0),
            "rel_mol_col_O2": (0.8619, 0.0100), "ppmv_O2": (182730.0, 2130.0)}
    for clave, valor in valores.items():
        base[clave] = valor if isinstance(valor, tuple) else (valor, -1.0)
    nombres = list(base) + ["\x00"]
    vals = [v[0] for v in base.values()] + [float("nan")]
    uncs = [v[1] for v in base.values()] + [float("nan")]
    cols = [
        fits.Column(name="parameter", format="30A", array=np.array(nombres)),
        fits.Column(name="value", format="D", array=np.array(vals, dtype=np.float64)),
        fits.Column(name="uncertainty", format="D", array=np.array(uncs, dtype=np.float64)),
    ]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fits.HDUList([fits.PrimaryHDU(), fits.BinTableHDU.from_columns(cols)]).writeto(path, overwrite=True)


class FirmaTests(unittest.TestCase):
    """Cada firma nombra un defecto que este proyecto ya ha visto y documentado."""

    def test_the_frozen_chi2_is_the_2026_07_06_signature(self):
        # chi2_best == chi2_ini exacto con status=4: el modelo entro desplazado en
        # lambda por WLC_CONST=-0.05 y no se movio ni una iteracion.
        best = {"status": (4.0, -1.0), "initial_chi2": (1267070.0, -1.0),
                "best_chi2": (1267070.0, -1.0), "rel_mol_col_O2": (1.0, 0.0)}

        convergio, firma = mf.classify_fit(best)

        self.assertFalse(convergio)
        self.assertEqual(firma, "chi2_congelado")

    def test_a_column_stuck_at_one_with_zero_uncertainty_is_no_leverage(self):
        best = {"status": (2.0, -1.0), "initial_chi2": (100.0, -1.0), "best_chi2": (99.0, -1.0),
                "rel_mol_col_O2": (1.0, 0.0)}

        self.assertEqual(mf.classify_fit(best)[1], "sin_leverage")

    def test_a_huge_rms_relative_to_the_error_is_the_continuum(self):
        # A1a fallido medía 85.8: el desajuste de continuo dominaba el chi2.
        best = {"status": (2.0, -1.0), "initial_chi2": (100.0, -1.0), "best_chi2": (99.0, -1.0),
                "rms_rel_to_err": (85.8, -1.0), "rel_mol_col_O2": (0.9, 0.01)}

        self.assertEqual(mf.classify_fit(best)[1], "continuo_atascado")

    def test_no_product_at_all_is_its_own_signature(self):
        self.assertEqual(mf.classify_fit({})[1], "sin_producto")
        self.assertEqual(mf.classify_fit({}, returncode=1)[1], "receta_fallo")

    def test_max_iterations_is_a_valid_stop_not_a_failure(self):
        # ROXs 42B b, medido 2026-08-22: status=5 (maximo de iteraciones) con el
        # chi2 cayendo 45x. Tratarlo como fallo hacia subir peldanos para nada.
        best = {"status": (5.0, -1.0), "iterations": (151.0, -1.0),
                "initial_chi2": (130875.8, -1.0), "best_chi2": (2875.8, -1.0),
                "rms_rel_to_err": (5.75, -1.0), "rel_mol_col_O2": (0.0819, 0.6678)}

        convergio, firma = mf.classify_fit(best)

        self.assertTrue(convergio)
        self.assertEqual(firma, "converge_por_iteraciones")

    def test_a_non_positive_status_is_an_error(self):
        best = {"status": (0.0, -1.0), "initial_chi2": (100.0, -1.0),
                "best_chi2": (50.0, -1.0), "rel_mol_col_O2": (0.9, 0.01)}

        self.assertEqual(mf.classify_fit(best), (False, "status_error"))

    def test_the_a1a_numbers_converge(self):
        # Los del ajuste que sí salió: status=2, chi2 baja, columna con incertidumbre.
        best = {"status": (2.0, -1.0), "initial_chi2": (4432.6, -1.0), "best_chi2": (4048.4, -1.0),
                "rms_rel_to_err": (5.09, -1.0), "rel_mol_col_O2": (0.9659, 0.016)}

        self.assertEqual(mf.classify_fit(best), (True, "converge"))


class EscaleraTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.cube = self.dir / "cube.fits"
        escribe_cubo(self.cube)

    def tearDown(self):
        self.tmp.cleanup()

    def _runner(self, guion):
        """Un esorex de mentira: `guion` dice qué escribe cada peldaño."""

        llamadas = []

        def runner(command, *, cwd, log_path):
            peldano = Path(log_path).stem.replace("run_", "")
            llamadas.append(peldano)
            salida = Path(str(command[1]).split("=", 1)[1])
            valores = guion.get(peldano)
            if valores is not None:
                escribe_best_fit(salida / "BEST_FIT_PARAMETERS.fits", **valores)
            Path(log_path).write_text("fake esorex\n")
            return 0

        return runner, llamadas

    def test_the_first_rung_already_carries_the_four_known_remedies(self):
        # No son un rescate: son el punto de partida medido.
        self.assertEqual(mf.MOLECFIT_BASE_PARAMS["WLC_CONST"], "0")
        self.assertEqual(mf.MOLECFIT_BASE_PARAMS["FIT_WLC"], "1")
        self.assertEqual(mf.LADDER[0].name, "base")
        bandas = {w.band for w in mf.FIT_WINDOWS}
        self.assertIn("O2_A", bandas, "sin la banda A el O2 no tiene leverage")

    def test_it_stops_at_the_first_rung_that_converges(self):
        runner, llamadas = self._runner({"base": {}})

        resultado = mf.fit_molecfit(self.cube, out_root=self.dir, label="uno",
                                    reuse=False, runner=runner)

        self.assertTrue(resultado.converged)
        self.assertEqual(resultado.rung, "base")
        self.assertEqual(llamadas, ["base"])

    def test_a_frozen_fit_climbs_instead_of_giving_up(self):
        congelado = {"status": (4.0, -1.0), "initial_chi2": (100.0, -1.0),
                     "best_chi2": (100.0, -1.0), "rel_mol_col_O2": (1.0, 0.0)}
        runner, llamadas = self._runner({"base": congelado, "wlc_grado_1": {}})

        resultado = mf.fit_molecfit(self.cube, out_root=self.dir, label="dos",
                                    reuse=False, runner=runner)

        self.assertTrue(resultado.converged)
        self.assertEqual(resultado.rung, "wlc_grado_1")
        self.assertEqual(llamadas, ["base", "wlc_grado_1"])
        self.assertEqual(resultado.rungs_tried[0]["firma"], "chi2_congelado")

    def test_an_exhausted_ladder_raises_and_never_falls_back(self):
        congelado = {"status": (4.0, -1.0), "initial_chi2": (100.0, -1.0),
                     "best_chi2": (100.0, -1.0), "rel_mol_col_O2": (1.0, 0.0)}
        runner, llamadas = self._runner({p.name: congelado for p in mf.LADDER})

        with self.assertRaises(mf.MolecfitNotConverged) as ctx:
            mf.fit_molecfit(self.cube, out_root=self.dir, label="tres", reuse=False, runner=runner)

        self.assertEqual(len(llamadas), len(mf.LADDER), "se prueban TODOS los peldanos")
        self.assertEqual(len(ctx.exception.rungs), len(mf.LADDER))
        # El mensaje tiene que decir que esto NO autoriza a usar STD_TELLURIC.
        self.assertIn("STD_TELLURIC", str(ctx.exception))
        self.assertIsInstance(ctx.exception, TelluricError, "la etapa tiene que parar")

    def test_the_strong_only_rung_drops_the_water_windows(self):
        peldano = next(p for p in mf.LADDER if p.windows == "strong")
        fuertes = [w.band for w in mf.FIT_WINDOWS if w.strong]

        self.assertEqual(fuertes, ["O2_B", "O2_A"])
        self.assertIn("sin_leverage", peldano.attends)


class SeleccionPorProfundidadTests(unittest.TestCase):
    """No se ajusta lo que no esta absorbido, y eso no es un fallo."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_only_the_bands_above_the_threshold_are_fitted(self):
        profundidades = {"O2_B": 0.0, "O2_A": 5.09, "H2O_7200": 0.0, "H2O_8200": 0.0}

        elegidas = mf.select_fit_windows(profundidades, threshold_pct=3.0)

        self.assertEqual([w.band for w in elegidas], ["O2_A"])

    def test_a_flat_spectrum_is_not_fitted_at_all(self):
        # Los cubos por exposicion salen de muse_scipost consumiendo STD_TELLURIC:
        # ya vienen corregidos. Ajustar ahi deja al ajuste sin senal telurica, la
        # columna cae al borde 0.0000 y la escalera se pone a subir peldanos
        # buscando un residuo que no existe.
        cube = self.dir / "plano.fits"
        escribe_cubo(cube, banda_pct=0.0)
        llamadas = []

        def runner(command, *, cwd, log_path):
            llamadas.append(command)
            return 0

        resultado = mf.fit_molecfit(cube, out_root=self.dir, label="plano",
                                    threshold_pct=3.0, reuse=False, runner=runner)

        self.assertEqual(llamadas, [], "no se lanza esorex para nada")
        self.assertTrue(resultado.converged, "no ajustar NO es un fallo")
        self.assertEqual(resultado.rung, "sin_ajuste")
        self.assertEqual(resultado.signature, "not_needed_shallow")
        self.assertEqual(resultado.out_dir, "", "sin producto: la T de esta unidad es 1")

    def test_a_deep_band_is_fitted(self):
        cube = self.dir / "honda.fits"
        escribe_cubo(cube, banda_pct=20.0)
        llamadas = []

        def runner(command, *, cwd, log_path):
            llamadas.append(command)
            salida = Path(str(command[1]).split("=", 1)[1])
            escribe_best_fit(salida / "BEST_FIT_PARAMETERS.fits")
            Path(log_path).write_text("fake\n")
            return 0

        resultado = mf.fit_molecfit(cube, out_root=self.dir, label="honda",
                                    threshold_pct=3.0, reuse=False, runner=runner)

        self.assertEqual(len(llamadas), 1)
        self.assertEqual([w["band"] for w in resultado.windows], ["O2_A"])
        self.assertGreater(resultado.depth_pct_by_band["O2_A"], 3.0)

    def test_without_a_threshold_nothing_is_gated(self):
        # `threshold_pct=None` conserva el comportamiento de pasarle ventanas a mano.
        cube = self.dir / "plano2.fits"
        escribe_cubo(cube, banda_pct=0.0)

        def runner(command, *, cwd, log_path):
            salida = Path(str(command[1]).split("=", 1)[1])
            escribe_best_fit(salida / "BEST_FIT_PARAMETERS.fits")
            Path(log_path).write_text("fake\n")
            return 0

        resultado = mf.fit_molecfit(cube, out_root=self.dir, label="plano2",
                                    reuse=False, runner=runner)

        self.assertEqual(resultado.rung, "base")


class MarcoYTransmisionTests(unittest.TestCase):
    """Las dos trampas medidas: el eje en vacio, y mtrans=0 fuera de las ventanas."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def _molecfit_data(self, wave, mtrans, mrange):
        cols = [
            fits.Column(name="lambda", format="D", array=np.asarray(wave) * 1e-4 + 2.09e-4),
            fits.Column(name="mtrans", format="D", array=np.asarray(mtrans, dtype=np.float64)),
            fits.Column(name="mrange", format="D", array=np.asarray(mrange, dtype=np.float64)),
        ]
        fits.HDUList([fits.PrimaryHDU(),
                      fits.BinTableHDU.from_columns(cols)]).writeto(self.dir / "MOLECFIT_DATA.fits",
                                                                    overwrite=True)

    def test_outside_the_fitted_windows_the_transmission_is_one_not_zero(self):
        # `mtrans` vale 0 fuera de las ventanas: aplicarla tal cual dividiria el
        # cubo entero por cero.
        wave = np.linspace(6000.0, 8000.0, 200)
        mrange = np.where((wave > 6864) & (wave < 6960), 1, 0)
        mtrans = np.where(mrange > 0, 0.9, 0.0)
        self._molecfit_data(wave, mtrans, mrange)

        trans = mf.transmission_from_fit(self.dir, wave)

        self.assertTrue(np.all(trans > 0), "ninguna T puede ser 0")
        np.testing.assert_allclose(trans[mrange == 0], 1.0)
        np.testing.assert_allclose(trans[mrange > 0], 0.9)

    def test_halpha_is_forced_to_one_even_if_the_fit_says_otherwise(self):
        wave = np.linspace(6400.0, 6700.0, 300)
        self._molecfit_data(wave, np.full(wave.size, 0.5), np.ones(wave.size))

        trans = mf.transmission_from_fit(self.dir, wave)

        halpha = (wave >= 6540.0) & (wave <= 6590.0)
        np.testing.assert_allclose(trans[halpha], 1.0)

    def test_a_row_count_mismatch_is_refused(self):
        # La correspondencia 1:1 es lo que hace exacto el eje en aire. Sin ella,
        # usar `wave_A` seria alinear a ojo.
        wave = np.linspace(6000.0, 8000.0, 200)
        self._molecfit_data(wave, np.ones(200), np.ones(200))

        with self.assertRaises(TelluricError):
            mf.transmission_from_fit(self.dir, np.linspace(6000.0, 8000.0, 199))


class MasaDeAireTests(unittest.TestCase):
    def test_it_weights_by_exptime_not_by_exposure_count(self):
        # Lo que promedia la atmosfera es el foton, no la exposicion.
        conds = [{"airm_start": 1.0, "airm_end": 1.0, "exptime": 900.0},
                 {"airm_start": 2.0, "airm_end": 2.0, "exptime": 300.0}]

        x, fuente = mf.effective_airmass(conds)

        self.assertAlmostEqual(x, 1.25, places=6)
        self.assertIn("EXPTIME", fuente)

    def test_one_exposure_is_the_middle_of_its_own_exposure(self):
        x, fuente = mf.effective_airmass([{"airm_start": 1.10, "airm_end": 1.20, "exptime": 300.0}])

        self.assertAlmostEqual(x, 1.15, places=6)
        self.assertEqual(fuente, "mitad de exposicion")

    def test_no_airmass_anywhere_is_declared_not_silently_zenith(self):
        x, fuente = mf.effective_airmass([{"airm_start": None, "airm_end": None, "exptime": 300.0}])

        self.assertTrue(np.isnan(x))
        self.assertIn("sin AIRM", fuente)
        # Y el angulo que se le pasa a la receta cae a cenit, que es lo unico que
        # se puede hacer, pero el NaN de arriba es lo que obliga a declararlo.
        self.assertEqual(mf.telescope_angle_deg(x), 90.0)

    def test_the_angle_is_the_arcsine_of_one_over_x(self):
        # A1a paso 59.72 deg para X = 1.159.
        self.assertAlmostEqual(mf.telescope_angle_deg(1.159), 59.62, places=1)


class NormalizacionTests(unittest.TestCase):
    def test_it_brings_the_flux_to_order_one(self):
        # Sin esto el continuo se atasca en 1.0 y el chi2 lo domina el.
        flux = 58000.0 * (1.0 + 0.01 * np.sin(np.linspace(0, 20, 1000)))

        norm = mf.normalize_by_continuum(flux)

        self.assertAlmostEqual(float(np.median(norm)), 1.0, places=2)

    def test_the_window_is_the_one_a1a_used(self):
        self.assertEqual(mf.CONTINUUM_WINDOW_PX, 201)


if __name__ == "__main__":
    unittest.main()
