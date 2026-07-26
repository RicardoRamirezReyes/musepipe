"""La figura de paper y su tabla: qué se marca, y que los datos viajen con ella.

Dos cosas que se fijan aquí y que no son cosmética:

- el catálogo telúrico son **bandas**, no líneas, porque a la resolución de MUSE
  las líneas de O₂/H₂O no se resuelven; y la profundidad que se anota sale de la
  curva **medida** del run cuando existe, no de un valor de laboratorio;
- la figura **siempre** escribe sus datos en columnas. Una figura sin su tabla
  no es un resultado citable, así que el export se prueba de ida y vuelta.
"""
import tempfile
import unittest
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")

from astropy.table import Table

from musepipe.paper_spectrum import (
    accretion_lines,
    paper_spectrum_figure,
    spectrum_table_meta,
    write_spectrum_table,
)
from musepipe.telluric_lines import (
    TELLURIC_BANDS,
    bands_with_measured_depth,
    measured_transmission,
    sky_emission_lines,
    telluric_mask,
)

MUSE_RANGE_A = (4700.0, 9400.0)


def _spectrum(n=400):
    wave = np.linspace(4750.0, 9350.0, n)
    flux = 100.0 + 0.1 * (wave - 4750.0)
    err = np.full_like(flux, 30.0)
    flux[10:20] = np.nan          # hueco: la figura no puede reventar con NaN
    return wave, flux, err


class TelluricCatalogTests(unittest.TestCase):
    def test_the_bands_are_ordered_and_inside_the_muse_range(self):
        previo = 0.0
        for band in TELLURIC_BANDS:
            self.assertLess(band["lo_A"], band["hi_A"], band["name"])
            self.assertGreaterEqual(band["lo_A"], MUSE_RANGE_A[0], band["name"])
            self.assertGreater(band["hi_A"], previo, "el catálogo va ordenado en λ")
            previo = band["lo_A"]
            self.assertIn(band["severity"], {"strong", "moderate", "weak"})
            self.assertIn(band["species"], {"O2", "H2O"})

    def test_the_two_strong_bands_are_the_oxygen_ones(self):
        # O2 A (7590-7700) y O2 B: si alguna vez desaparecen del catálogo, el
        # plot dejaria de avisar justo donde la atmosfera se come el 70%.
        fuertes = {b["name"] for b in TELLURIC_BANDS if b["severity"] == "strong"}
        self.assertIn("O₂ A", fuertes)
        self.assertIn("O₂ B", fuertes)

    def test_the_depth_is_measured_from_the_curve_when_there_is_one(self):
        wave = np.linspace(4750.0, 9350.0, 4601)
        trans = np.ones_like(wave)
        trans[(wave >= 7600) & (wave <= 7690)] = 0.30
        rows = bands_with_measured_depth({"wave_A": wave, "transmission": trans})
        por_nombre = {r["name"]: r for r in rows}
        self.assertAlmostEqual(por_nombre["O₂ A"]["t_min"], 0.30, places=6)
        self.assertAlmostEqual(por_nombre["O₂ B"]["t_min"], 1.0, places=6)

    def test_without_a_curve_the_catalog_says_it_does_not_know(self):
        rows = bands_with_measured_depth(None)
        self.assertEqual(len(rows), len(TELLURIC_BANDS))
        self.assertTrue(all(r["t_min"] is None for r in rows))

    def test_the_mask_only_takes_the_severities_asked_for(self):
        wave = np.array([5000.0, 5900.0, 6900.0, 7650.0, 8200.0])
        fuertes = telluric_mask(wave, severities=("strong",))
        np.testing.assert_array_equal(fuertes, [False, False, True, True, False])
        con_moderadas = telluric_mask(wave)
        self.assertTrue(con_moderadas[4])     # H2O 8200 entra al pedir moderate
        self.assertFalse(con_moderadas[1])    # H2O 5900 es weak: sigue fuera

    def test_a_run_without_the_a3_product_returns_none_instead_of_failing(self):
        # Es el caso del perfil `cascade`: corrige el telúrico dentro del
        # combinado y no deja la curva suelta. El plot tiene que seguir saliendo.
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIsNone(measured_transmission("no_existe", project_root=tmp))

    def test_the_sky_emission_lines_load(self):
        lineas = sky_emission_lines()
        self.assertGreaterEqual(len(lineas), 5)
        for linea in lineas:
            self.assertTrue(MUSE_RANGE_A[0] < linea["wave_A"] < MUSE_RANGE_A[1])


class AccretionLinesTests(unittest.TestCase):
    def test_halpha_and_the_ca_triplet_are_there(self):
        nombres = {ln["name"] for ln in accretion_lines()}
        self.assertIn("Halpha", nombres)
        for ca in ("Ca II 8498", "Ca II 8542", "Ca II 8662"):
            self.assertIn(ca, nombres)

    def test_every_line_carries_a_family_for_its_colour(self):
        from musepipe.paper_spectrum import FAMILY_COLORS

        for line in accretion_lines():
            self.assertIn(line["family"], FAMILY_COLORS, line["name"])


class FigureTests(unittest.TestCase):
    def test_it_builds_the_strip_only_when_there_is_a_measured_curve(self):
        wave, flux, err = _spectrum()
        fig, axes = paper_spectrum_figure(wave, flux, err, n_panels=2)
        self.assertEqual(len(axes), 2)
        self.assertEqual(len(fig.axes), 2)          # sin curva: solo espectros
        trans = {"wave_A": wave, "transmission": np.ones_like(wave)}
        fig2, axes2 = paper_spectrum_figure(wave, flux, err, n_panels=2, transmission=trans)
        self.assertEqual(len(fig2.axes), 4)         # con curva: 2 tiras + 2 espectros

    def test_it_survives_a_spectrum_with_gaps_and_no_error(self):
        wave, flux, _ = _spectrum()
        fig, axes = paper_spectrum_figure(wave, flux, None, n_panels=1)
        self.assertEqual(len(axes), 1)
        lo, hi = axes[0].get_ylim()
        self.assertLess(lo, hi)

    def test_the_panels_split_the_range_without_leaving_huecos(self):
        wave, flux, err = _spectrum()
        _fig, axes = paper_spectrum_figure(wave, flux, err, n_panels=3)
        limites = [ax.get_xlim() for ax in axes]
        self.assertAlmostEqual(limites[0][0], wave.min(), places=6)
        self.assertAlmostEqual(limites[-1][1], wave.max(), places=6)
        for izq, der in zip(limites, limites[1:]):
            self.assertAlmostEqual(izq[1], der[0], places=6)

    def test_the_flagged_channels_are_not_drawn(self):
        """El hueco del láser vale 0, no NaN: sin recortarlo, la curva de error
        bajaba a cero cruzándolo y se leía como una medida exquisita."""
        wave, flux, err = _spectrum()
        malos = (wave >= 5780) & (wave <= 6050)
        flux = np.where(malos, 0.0, flux)
        err = np.where(malos, 0.0, err)
        _fig, axes = paper_spectrum_figure(wave, flux, err, n_panels=1,
                                           bad_channels=malos, smooth_channels=0)
        curvas = [np.asarray(ln.get_ydata(), dtype=float) for ln in axes[0].lines
                  if np.size(ln.get_ydata()) == wave.size]
        self.assertTrue(curvas, "no se dibujó ninguna curva del espectro")
        for curva in curvas:
            self.assertTrue(np.all(np.isnan(curva[malos])))

    def test_the_lines_asked_for_end_up_drawn(self):
        wave, flux, err = _spectrum()
        _fig, axes = paper_spectrum_figure(
            wave, flux, err, n_panels=1, sky_lines=(),
            lines=[{"name": "Halpha", "wave_A": 6562.8, "family": "Balmer"}])
        etiquetas = [t.get_text() for t in axes[0].texts]
        self.assertIn("Halpha", etiquetas)


class FluxUnitLabelTests(unittest.TestCase):
    def test_the_muse_bunit_becomes_readable(self):
        from musepipe.paper_spectrum import pretty_flux_unit

        self.assertEqual(pretty_flux_unit("10**(-20)*erg/s/cm**2/Angstrom"),
                         "10⁻²⁰ erg s⁻¹ cm⁻² Å⁻¹")

    def test_anything_else_goes_through_untouched_and_nothing_is_invented(self):
        from musepipe.paper_spectrum import pretty_flux_unit

        self.assertEqual(pretty_flux_unit("counts"), "counts")
        # Sin BUNIT no se inventa una unidad: se dice que no la hay.
        self.assertEqual(pretty_flux_unit(None), "sin unidad declarada")
        self.assertEqual(pretty_flux_unit(""), "sin unidad declarada")


class SpectrumTableTests(unittest.TestCase):
    def test_the_ecsv_roundtrips_with_units_and_provenance(self):
        wave, flux, err = _spectrum()
        meta = spectrum_table_meta(run_id="R", target="T", method="aperture",
                                   product="spec_aperture_object.fits",
                                   flux_unit="1e-20 erg / (Angstrom s cm2)",
                                   error_mode="empirical")
        with tempfile.TemporaryDirectory() as tmp:
            path = write_spectrum_table(
                Path(tmp) / "sub" / "spec.ecsv", wave, flux, err,
                meta=meta, extra_columns={"flags": np.zeros(wave.size, dtype=int)},
                units={"flux": "1e-20 erg / (Angstrom s cm2)",
                       "flux_err": "1e-20 erg / (Angstrom s cm2)"})
            leida = Table.read(path)
        self.assertEqual(leida.colnames, ["wave_A", "flux", "flux_err", "flags"])
        self.assertEqual(str(leida["wave_A"].unit), "Angstrom")
        self.assertIsNotNone(leida["flux"].unit)
        self.assertEqual(leida.meta["method"], "aperture")
        self.assertEqual(leida.meta["wavelength_frame"], "air")
        np.testing.assert_allclose(np.asarray(leida["wave_A"]), wave)
        # Los NaN del hueco tienen que sobrevivir al texto: si se convirtieran
        # en 0 el fichero mentiria justo en los canales sin dato.
        self.assertEqual(int(np.isnan(np.asarray(leida["flux"])).sum()),
                         int(np.isnan(flux).sum()))

    def test_the_suffix_picks_the_format(self):
        wave, flux, err = _spectrum(50)
        with tempfile.TemporaryDirectory() as tmp:
            fits_path = write_spectrum_table(Path(tmp) / "s.fits", wave, flux, err)
            csv_path = write_spectrum_table(Path(tmp) / "s.csv", wave, flux, err)
            self.assertEqual(len(Table.read(fits_path)), wave.size)
            self.assertEqual(Table.read(csv_path).colnames,
                             ["wave_A", "flux", "flux_err"])

    def test_the_note_about_sigma_travels_in_the_meta(self):
        # Quien reciba el fichero suelto tiene que enterarse de que el STAT no
        # es sigma sin abrir el repo.
        meta = spectrum_table_meta(run_id="R", target="T", method="aperture",
                                   product="p.fits", flux_unit="adu")
        self.assertIn("noise_model", meta["note"])
        self.assertIn("n_eff", meta["note"])


if __name__ == "__main__":
    unittest.main()
