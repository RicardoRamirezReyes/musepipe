"""El fondo azimutal: anillo centrado en la PRIMARIA, al radio del compañero.

Existe porque el anillo centrado en el compañero atraviesa el gradiente radial
del halo, mientras que uno centrado en la primaria se mantiene a halo constante
por construcción.

**El sesgo NO es el que parece.** La intuición dice que el lado interior del
anillo, más cerca de la estrella y por tanto más brillante, tira la mediana
hacia arriba. Es falso, y aquí se fija: la curvatura del arco mete más área por
fuera que por dentro, así que la mediana del radio *estelar* dentro del anillo
cae 0.7 px MÁS LEJOS de la estrella y el anillo **sub**-estima el halo.

Lo que NO se fija aquí es que sea mejor: medido en los dos objetos del proyecto
el cambio va en direcciones opuestas, porque el halo de AO no es azimutalmente
simétrico. Por eso es una opción y el defecto sigue siendo `annulus`; eso último
sí se fija, porque un defecto que se mueva solo cambia resultados congelados.
"""
import unittest

import numpy as np

from musepipe.extraction.aperture import (
    annulus_background_spectrum,
    azimuthal_background_spectrum,
)
from musepipe.extraction.optimal import BACKGROUND_MODES, local_background_spectrum

STAR_YX = (60.0, 60.0)
COMP_YX = (60.0, 100.0)     # 40 px a la derecha de la primaria
SHAPE = (4, 121, 121)


def halo_cube(power=2.0, amplitude=1.0e5):
    """Halo con simetría circular perfecta que cae con el radio."""
    yy, xx = np.indices(SHAPE[1:], dtype=float)
    r = np.hypot(yy - STAR_YX[0], xx - STAR_YX[1])
    plano = amplitude / np.power(np.maximum(r, 1.0), power)
    return np.repeat(plano[None], SHAPE[0], axis=0)


class AzimuthalBackgroundTests(unittest.TestCase):
    def test_on_a_symmetric_halo_it_recovers_the_value_at_that_radius(self):
        cube = halo_cube()
        sep = float(np.hypot(COMP_YX[0] - STAR_YX[0], COMP_YX[1] - STAR_YX[1]))
        fondo = azimuthal_background_spectrum(cube, STAR_YX, sep, width_px=3.0,
                                              exclude_yx=COMP_YX, exclude_radius=10.0)
        verdadero = 1.0e5 / sep ** 2
        self.assertEqual(fondo.shape, (SHAPE[0],))
        np.testing.assert_allclose(fondo, verdadero, rtol=0.05)

    def test_the_azimuthal_is_closer_to_the_truth_on_a_symmetric_halo(self):
        """El motivo de existir, medido en un halo sin ninguna asimetría.

        Con simetría perfecta el azimutal es exacto por construcción. El anillo
        centrado en el compañero mezcla radios estelares y falla.

        **Y falla hacia ABAJO, no hacia arriba.** En un anillo de 8-14 px
        alrededor de un compañero a 71 px, la mediana del radio *estelar* es
        71.93 px, o sea 0.7 px MÁS LEJOS de la estrella: la curvatura del arco
        mete más área por fuera que por dentro. Así que el anillo muestrea halo
        algo más débil y **sub**-estima. Conviene tenerlo fijado porque la
        intuición contraria («el lado interior, más brillante, tira la mediana
        hacia arriba») es tentadora y es falsa.
        """
        cube = halo_cube()
        sep = float(np.hypot(COMP_YX[0] - STAR_YX[0], COMP_YX[1] - STAR_YX[1]))
        verdadero = 1.0e5 / sep ** 2
        azim = float(azimuthal_background_spectrum(
            cube, STAR_YX, sep, exclude_yx=COMP_YX, exclude_radius=10.0)[0])
        anillo = float(annulus_background_spectrum(
            cube, COMP_YX, 8.0, 14.0, exclude_yx=STAR_YX, exclude_radius=20.0)[0])
        self.assertLess(anillo, verdadero)       # sub-estima, no sobre-estima
        self.assertLess(abs(azim - verdadero), abs(anillo - verdadero))

    def test_the_stellar_radius_inside_the_annulus_skews_outward(self):
        """La geometría que explica el signo del test anterior, aislada."""
        yy, xx = np.indices((200, 200), dtype=float)
        estrella, compa = (60.0, 60.0), (60.0, 131.2)
        r_obj = np.hypot(yy - compa[0], xx - compa[1])
        r_star = np.hypot(yy - estrella[0], xx - estrella[1])
        dentro = (r_obj >= 8) & (r_obj <= 14)
        sep = float(np.hypot(compa[0] - estrella[0], compa[1] - estrella[1]))
        self.assertGreater(float(np.median(r_star[dentro])), sep)

    def test_it_excludes_the_source_it_is_measuring_the_background_for(self):
        # Sin exclusión, el propio compañero contamina su fondo y lo sube.
        cube = halo_cube()
        cube[:, int(COMP_YX[0]) - 1:int(COMP_YX[0]) + 2,
             int(COMP_YX[1]) - 1:int(COMP_YX[1]) + 2] += 1.0e4
        sep = float(np.hypot(COMP_YX[0] - STAR_YX[0], COMP_YX[1] - STAR_YX[1]))
        con = float(azimuthal_background_spectrum(
            cube, STAR_YX, sep, exclude_yx=COMP_YX, exclude_radius=10.0)[0])
        sin = float(azimuthal_background_spectrum(
            cube, STAR_YX, sep, exclude_yx=None)[0])
        self.assertLessEqual(con, sin)
        np.testing.assert_allclose(con, 1.0e5 / sep ** 2, rtol=0.05)

    def test_an_empty_ring_gives_zeros_instead_of_nan(self):
        cube = halo_cube()
        fondo = azimuthal_background_spectrum(cube, STAR_YX, 5000.0, width_px=1.0)
        np.testing.assert_array_equal(fondo, np.zeros(SHAPE[0]))

    def test_a_fully_masked_channel_stays_nan_and_does_not_warn(self):
        import warnings

        cube = halo_cube()
        cube[1] = np.nan
        sep = float(np.hypot(COMP_YX[0] - STAR_YX[0], COMP_YX[1] - STAR_YX[1]))
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            fondo = azimuthal_background_spectrum(cube, STAR_YX, sep)
        self.assertTrue(np.isnan(fondo[1]))
        self.assertTrue(np.isfinite(fondo[0]))


class ModeSelectionTests(unittest.TestCase):
    def test_the_default_is_still_the_annulus(self):
        """Un defecto que se mueva solo cambiaría resultados ya congelados."""
        import inspect

        from musepipe.stages import stage_x02_optimal

        fuente = inspect.getsource(stage_x02_optimal.stage_x02_config_from_run)
        self.assertIn('cfg.setdefault("x02_background_mode", "annulus")', fuente)

    def test_the_two_modes_give_the_documented_estimators(self):
        cube = halo_cube()
        anillo = local_background_spectrum(
            cube, COMP_YX, STAR_YX, mode="annulus", annulus_px=[8.0, 14.0, 20.0])
        azim = local_background_spectrum(
            cube, COMP_YX, STAR_YX, mode="azimuthal", azimuthal_exclude_px=10.0)
        sep = float(np.hypot(COMP_YX[0] - STAR_YX[0], COMP_YX[1] - STAR_YX[1]))
        np.testing.assert_allclose(anillo, annulus_background_spectrum(
            cube, COMP_YX, 8.0, 14.0, exclude_yx=STAR_YX, exclude_radius=20.0))
        np.testing.assert_allclose(azim, azimuthal_background_spectrum(
            cube, STAR_YX, sep, exclude_yx=COMP_YX, exclude_radius=10.0))

    def test_without_an_annulus_configured_the_annulus_mode_is_a_no_op(self):
        cube = halo_cube()
        self.assertIsNone(local_background_spectrum(
            cube, COMP_YX, STAR_YX, mode="annulus", annulus_px=None))

    def test_an_unknown_mode_is_an_error_and_not_a_silent_fallback(self):
        cube = halo_cube()
        with self.assertRaises(ValueError):
            local_background_spectrum(cube, COMP_YX, STAR_YX, mode="promedio")
        self.assertEqual(BACKGROUND_MODES, ("annulus", "azimuthal"))

    def test_controls_get_their_OWN_neighbourhood_excluded(self):
        """`control = objeto` exige el mismo estimador, no el mismo número.

        En modo azimutal el radio sale de cada posición y lo que se excluye es
        su propio entorno: si se excluyera siempre el del compañero, el control
        se estaría midiendo a sí mismo dentro de su fondo.
        """
        cube = halo_cube()
        control_yx = (100.0, 60.0)      # mismo radio, otro ángulo
        cube[:, 99:102, 59:62] += 1.0e4  # una fuente en el control
        sep = float(np.hypot(COMP_YX[0] - STAR_YX[0], COMP_YX[1] - STAR_YX[1]))
        propio = local_background_spectrum(
            cube, control_yx, STAR_YX, mode="azimuthal", azimuthal_exclude_px=10.0)
        np.testing.assert_allclose(propio, 1.0e5 / sep ** 2, rtol=0.05)


if __name__ == "__main__":
    unittest.main()
