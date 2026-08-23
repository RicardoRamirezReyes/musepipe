"""El arbitro de A3 y la aplicacion por trozos.

Dos reglas del encargo viven aqui:

- «se prueban las dos vias y se queda con la que da mejores resultados»
- «si hay empate (errores en el mismo rango), preferir molecfit»

y una tercera que sale de mirar el dato: la metrica que decide **no puede ser la
que se publico el 2026-08-06**, porque aquella es la dispersion sobre la media y
esa es ciega a una sobre-correccion constante. Aqui se emiten las dos.
"""

import tempfile
import unittest
from pathlib import Path

import numpy as np
from astropy.io import fits

from musepipe.reduction.progress import Ledger
from musepipe.reduction.telluric import (
    CLEAN_WINDOW_A,
    TelluricError,
    apply_transmission_streaming,
    choose_method,
    effective_transmission,
    score_correction,
    std_telluric_transmission,
    tie_sigma_from_scores,
    _band_residual,
)


def eje(n=3681):
    return np.linspace(4750.0, 9350.0, n)


class MetricaTests(unittest.TestCase):
    def test_the_rms_sees_an_offset_that_the_scatter_hides(self):
        # rms^2 = dispersion^2 + offset^2. Una banda 5 % por debajo del continuo,
        # sin ruido ninguno, es una correccion sesgada: el rms lo dice, la
        # dispersion vale 0.
        w = eje(2000)
        spec = np.ones_like(w)
        banda = (7590.0, 7700.0)
        dentro = (w >= banda[0]) & (w <= banda[1])
        spec[dentro] = 0.95

        rms, dispersion = _band_residual(w, spec, banda)

        self.assertAlmostEqual(rms, 0.05, places=3)
        self.assertLess(dispersion, 1e-6)

    def test_pure_noise_with_no_correction_scores_about_one(self):
        w = eje()
        rng = np.random.default_rng(0)
        spec = 1.0 + 0.01 * rng.normal(size=w.size)

        punt = score_correction(w, spec, None)

        self.assertLess(abs(punt["fom_mean"] - 1.0), 0.25)

    def test_the_fom_discounts_the_noise_the_correction_itself_amplifies(self):
        # Dividir por T multiplica el ruido por 1/T. Una correccion PERFECTA sobre
        # una banda profunda deja un residuo grande medido contra el suelo, y sin
        # descontarlo se penalizaria justo a la via que corrige mas hondo.
        w = eje()
        rng = np.random.default_rng(1)
        banda = (7590.0, 7700.0)
        dentro = (w >= banda[0]) & (w <= banda[1])
        trans = np.ones_like(w)
        trans[dentro] = 0.30
        ruido = 0.01 * rng.normal(size=w.size)
        # espectro ya corregido: el ruido queda dividido por T, la banda plana
        spec = 1.0 + ruido / trans

        punt = score_correction(w, spec, trans)
        banda_punt = punt["by_band"]["O2_A"]

        self.assertGreater(banda_punt["rms_over_floor"], 2.0, "el crudo la penaliza")
        self.assertLess(abs(banda_punt["fom"] - 1.0), 0.3, "la FoM no")

    def test_the_floor_comes_from_the_declared_clean_window(self):
        self.assertEqual(CLEAN_WINDOW_A, (7750.0, 7860.0))


class ArbitrajeTests(unittest.TestCase):
    def test_a_tie_goes_to_molecfit_even_when_std_measures_better(self):
        # La regla del usuario, literal: errores en el mismo rango -> molecfit.
        veredicto = choose_method({"molecfit": {"fom_mean": 1.90},
                                   "std_telluric": {"fom_mean": 1.85}},
                                  tie_sigma=0.20, prefer="molecfit")

        self.assertEqual(veredicto["winner"], "molecfit")
        self.assertEqual(veredicto["reason"], "tie_prefers_molecfit")
        self.assertEqual(veredicto["would_have_won"], "std_telluric")

    def test_a_real_margin_is_not_a_tie(self):
        veredicto = choose_method({"molecfit": {"fom_mean": 1.90},
                                   "std_telluric": {"fom_mean": 1.20}},
                                  tie_sigma=0.20, prefer="molecfit")

        self.assertEqual(veredicto["winner"], "std_telluric")
        self.assertEqual(veredicto["reason"], "measured")

    def test_without_a_sigma_no_tie_is_declared(self):
        # Sin dispersion medida no hay «mismo rango» que declarar: gana quien mida
        # mejor, y el margen queda escrito para que se vea lo estrecho que fue.
        veredicto = choose_method({"molecfit": {"fom_mean": 1.90},
                                   "std_telluric": {"fom_mean": 1.8999}},
                                  tie_sigma=float("nan"), prefer="molecfit")

        self.assertEqual(veredicto["winner"], "std_telluric")
        self.assertEqual(veredicto["reason"], "measured")
        self.assertIsNone(veredicto["tie_sigma"])

    def test_one_method_alone_says_so(self):
        veredicto = choose_method({"molecfit": {"fom_mean": 1.9}}, tie_sigma=0.2)

        self.assertEqual(veredicto["reason"], "single_method")

    def test_nothing_finite_is_an_error_not_a_default_winner(self):
        with self.assertRaises(TelluricError):
            choose_method({"molecfit": {"fom_mean": float("nan")}})

    def test_the_tie_sigma_is_the_dispersion_between_exposures(self):
        sigma = tie_sigma_from_scores([{"fom_mean": 2.0}, {"fom_mean": 2.2}, {"fom_mean": 1.8}])

        self.assertAlmostEqual(sigma, 0.2, places=6)

    def test_one_exposure_gives_no_sigma(self):
        self.assertTrue(np.isnan(tie_sigma_from_scores([{"fom_mean": 2.0}])))


class ArbitrajeEnDosPasosTests(unittest.TestCase):
    """El empate decide entre VIAS; contra no corregir hay que ganar por medida.

    Medido en ROXs 42B b el 2026-08-22: la sigma entre exposiciones sale 1.5035, y
    con ella el empate se tragaba un margen real de 0.52 a favor de no corregir —
    aplicando una correccion que deja O2 A en -7.40 % donde estaba en +3.86 %.
    """

    def test_between_methods_the_tie_still_goes_to_molecfit(self):
        veredicto = choose_method({"molecfit": {"fom_mean": 2.7161},
                                   "std_telluric": {"fom_mean": 2.7017}},
                                  tie_sigma=1.5035, prefer="molecfit")

        self.assertEqual(veredicto["winner"], "molecfit")
        self.assertEqual(veredicto["reason"], "tie_prefers_molecfit")

    def test_the_null_is_not_a_third_method(self):
        # Si «sin corregir» entrara en el mismo empate, con esta sigma ganaria
        # molecfit por regla aun midiendo 0.52 peor. Ese es el caso que se corrige.
        con_nulo = choose_method({"molecfit": {"fom_mean": 2.7161},
                                  "sin_corregir": {"fom_mean": 2.1946}},
                                 tie_sigma=1.5035, prefer="molecfit")

        self.assertEqual(con_nulo["winner"], "molecfit",
                         "asi se comportaba antes, y por eso el nulo se saca del empate")

    def test_the_numbers_that_made_the_rule_change(self):
        # ROXs 42B b: ninguna correccion mejora, y la diferencia es real.
        foms = {"sin_corregir": 2.1946, "molecfit_combined": 2.7017,
                "molecfit_perexp": 2.7161, "std_telluric": 7.9008}

        self.assertLess(foms["sin_corregir"], min(v for k, v in foms.items()
                                                  if k != "sin_corregir"))


class VeredictoSinNadaQueCorregirTests(unittest.TestCase):
    """`not_needed_shallow` es terminar bien, y tiene que dejar QC."""

    def test_no_band_above_the_threshold_selects_nothing(self):
        from musepipe.reduction.molecfit import select_fit_windows

        # ROXs12b_OB3445598, medido 2026-08-22: la noche del 29 no dejo residuo.
        profundidades = {"O2_B": 0.48, "O2_A": 0.0, "H2O_7200": 0.0, "H2O_8200": 0.0}

        self.assertEqual(select_fit_windows(profundidades, threshold_pct=3.0), [])

    def test_the_stage_verdict_for_that_case(self):
        from musepipe.reduction.telluric import decide_telluric

        veredicto = decide_telluric({"O2_B": 0.48, "O2_A": 0.0},
                                    science_needs_red_continuum=True, threshold_pct=3.0)

        self.assertEqual(veredicto.decision, "not_needed_shallow")
        self.assertFalse(veredicto.telluric_applied)
        self.assertFalse(veredicto.checkpoint_required, "no hay nada que aprobar")


class StdTelluricTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "STD_TELLURIC_0001.fits"
        lam = np.linspace(6272.5, 9354.5, 640)
        ftel = np.where((lam > 7590) & (lam < 7700), 0.50, 1.0)
        cols = [fits.Column(name="lambda", format="D", array=lam),
                fits.Column(name="ftelluric", format="D", array=ftel),
                fits.Column(name="ftellerr", format="D", array=np.full(640, 1e-4))]
        pri = fits.Header()
        pri["HIERARCH ESO TEL AIRM START"] = 1.086
        pri["HIERARCH ESO TEL AIRM END"] = 1.088
        fits.HDUList([fits.PrimaryHDU(header=pri),
                      fits.BinTableHDU.from_columns(cols)]).writeto(self.path, overwrite=True)

    def tearDown(self):
        self.tmp.cleanup()

    def test_beer_lambert_scales_with_the_airmass_ratio(self):
        w = eje()

        trans, meta = std_telluric_transmission(self.path, w, airmass_sci=2.174)

        self.assertAlmostEqual(meta["airmass_std"], 1.087, places=3)
        self.assertAlmostEqual(meta["exponent"], 2.0, places=2)
        banda = (w > 7600) & (w < 7690)
        np.testing.assert_allclose(trans[banda], 0.25, atol=1e-3)

    def test_outside_the_standard_range_it_is_one_not_extrapolated(self):
        w = eje()

        trans, meta = std_telluric_transmission(self.path, w, airmass_sci=1.2)

        azul = w < 6272.5
        np.testing.assert_allclose(trans[azul], 1.0)
        self.assertEqual(meta["channels_outside_std_set_to_one"], int(azul.sum()))

    def test_halpha_stays_untouched(self):
        w = eje()

        trans, _ = std_telluric_transmission(self.path, w, airmass_sci=1.2)

        halpha = (w >= 6540.0) & (w <= 6590.0)
        np.testing.assert_allclose(trans[halpha], 1.0)

    def test_a_standard_without_airmass_is_refused(self):
        sin_x = Path(self.tmp.name) / "sin_airmass.fits"
        with fits.open(self.path) as h:
            del h[0].header["HIERARCH ESO TEL AIRM START"]
            del h[0].header["HIERARCH ESO TEL AIRM END"]
            h.writeto(sin_x, overwrite=True)

        with self.assertRaises(TelluricError):
            std_telluric_transmission(sin_x, eje(), airmass_sci=1.2)


class TransmisionEfectivaTests(unittest.TestCase):
    def test_it_uses_the_same_weights_the_combine_uses(self):
        t = effective_transmission([[1.0, 1.0], [0.0, 0.0]], [900.0, 300.0])

        np.testing.assert_allclose(t, 0.75)

    def test_a_mismatched_shape_is_refused(self):
        with self.assertRaises(TelluricError):
            effective_transmission([[1.0, 1.0]], [1.0, 1.0])


class AplicacionPorTrozosTests(unittest.TestCase):
    """DATA/T y STAT/T^2 en el mismo bucle, y un corte que no cuesta el trabajo."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.cube = self.dir / "cube.fits"
        nz, npix = 40, 6
        rng = np.random.default_rng(3)
        data = rng.normal(10.0, 1.0, size=(nz, npix, npix)).astype(np.float32)
        stat = np.full((nz, npix, npix), 4.0, dtype=np.float32)
        cabecera = fits.Header()
        cabecera["CRVAL3"] = 6000.0
        cabecera["CDELT3"] = 50.0
        cabecera["CRPIX3"] = 1.0
        fits.HDUList([
            fits.PrimaryHDU(),
            fits.ImageHDU(data=data, header=cabecera, name="DATA"),
            fits.ImageHDU(data=stat, header=cabecera, name="STAT"),
        ]).writeto(self.cube, overwrite=True)
        self.wave = 6000.0 + np.arange(nz) * 50.0
        self.trans = np.where((self.wave > 6900) & (self.wave < 7100), 0.5, 1.0)

    def tearDown(self):
        self.tmp.cleanup()

    def test_stat_is_divided_by_t_squared(self):
        # La trampa: el hook `transform` del combine devuelve solo DATA y NO toca
        # STAT (correcto para restar un modelo, falso para dividir por T).
        salida = self.dir / "out.fits"

        apply_transmission_streaming(self.cube, salida, self.trans, chunk_channels=7)

        with fits.open(self.cube) as antes, fits.open(salida) as despues:
            escala = self.trans[:, None, None]
            np.testing.assert_allclose(despues["DATA"].data, antes["DATA"].data / escala, rtol=1e-5)
            np.testing.assert_allclose(despues["STAT"].data, antes["STAT"].data / escala ** 2,
                                       rtol=1e-5)

    def test_an_interrupted_run_resumes_and_lands_on_the_same_cube(self):
        completo = self.dir / "completo.fits"
        apply_transmission_streaming(self.cube, completo, self.trans, chunk_channels=7)

        parcial = self.dir / "parcial.fits"
        bitacora = Ledger(self.dir / "a3_progress.json")
        hechos = []

        def corta(z1, z2, nz):
            hechos.append(z1)
            if len(hechos) == 2:
                raise KeyboardInterrupt("se corta la maquina")

        with self.assertRaises(KeyboardInterrupt):
            apply_transmission_streaming(self.cube, parcial, self.trans, chunk_channels=7,
                                         ledger=bitacora, progress=corta)

        reanudado = Ledger(self.dir / "a3_progress.json")
        ya_hechos = sum(1 for f in reanudado.rows if ":chunk:" in f["clave"])
        self.assertEqual(ya_hechos, 2, "la bitacora guarda lo que si termino")

        rehechos = []
        apply_transmission_streaming(self.cube, parcial, self.trans, chunk_channels=7,
                                     ledger=reanudado,
                                     progress=lambda z1, z2, nz: rehechos.append(z1))

        self.assertNotIn(0, rehechos, "no repite un trozo ya hecho")
        with fits.open(completo) as a, fits.open(parcial) as b:
            # Lo importante: los trozos ya hechos NO se dividieron dos veces.
            np.testing.assert_allclose(a["DATA"].data, b["DATA"].data, rtol=1e-6)
            np.testing.assert_allclose(a["STAT"].data, b["STAT"].data, rtol=1e-6)

    def test_the_stop_file_stops_between_chunks(self):
        bitacora = Ledger(self.dir / "a3_progress.json")
        (self.dir / "PARAR").write_text("")

        with self.assertRaises(TelluricError) as ctx:
            apply_transmission_streaming(self.cube, self.dir / "out.fits", self.trans,
                                         chunk_channels=7, ledger=bitacora)

        self.assertIn("PARAR", str(ctx.exception))

    def test_a_transmission_below_the_floor_is_refused_before_writing(self):
        hundida = np.full_like(self.trans, 0.01)

        with self.assertRaises(TelluricError):
            apply_transmission_streaming(self.cube, self.dir / "out.fits", hundida,
                                         chunk_channels=7)

    def test_halpha_is_forced_to_one_on_the_way_in(self):
        w = self.wave
        agresiva = np.full_like(w, 0.5)
        salida = self.dir / "out.fits"

        _, aplicada = apply_transmission_streaming(self.cube, salida, agresiva, chunk_channels=7)

        halpha = (w >= 6540.0) & (w <= 6590.0)
        np.testing.assert_allclose(aplicada[halpha], 1.0)


if __name__ == "__main__":
    unittest.main()
