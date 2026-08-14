"""D2 presenta los espectros definitivos: los 6 metodos + la primaria.

Los espectros calibrados son un resultado en si mismos, no solo la entrada de
E1, pero el QC de D2 solo listaba el canonico y `also_calibrated` (nombres), y
la unica figura de los 6 era un diagnostico crudo y sin unidad. Aqui se fija el
contrato de la tabla (`qc['spectra']`) y de la figura.
"""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from musepipe.io import MUSE_NATIVE_BUNIT
from musepipe.stages.stage_x11_calibrate import (
    SPECTRA_RED_BAND_A,
    compute_stage_x11_products,
    definitive_spectra_figure,
    stage_x11_paths,
    write_stage_x11_products,
)
from tests.test_calibrate_no_double import calibration_product

METHOD_KEYS = {
    "aperture": "spec_aperture_object",
    "optimal_ls": "spec_optimal_object",
    "optimal_psfsub": "spec_optimal_psfsub_object",
    "psffit": "spec_psffit_object",
    "sgf": "spec_sgf_object",
    "lpm": "spec_lpm_object",
}


def _product(method, *, bunit=MUSE_NATIVE_BUNIT, level=10.0):
    # Rejilla que cruza la banda roja (ahi se dan las medianas de la tabla) y
    # con paso fino: la ventana de continuo son 80 A y exige >=15 canales, asi
    # que una rejilla gruesa dejaria `cont_runmed` en NaN.
    wave = np.linspace(6000.0, 9000.0, 601)
    product = calibration_product(method=method, wave=wave, flux=np.full(wave.size, level))
    if bunit is not None:
        product.header["BUNIT"] = bunit
    return product


class SpectraTableTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.run_id = "synthetic_x11"
        self.paths = stage_x11_paths(self.run_id, project_root=self.root)
        self.paths["paths"].ensure_base_dirs()
        self.paths["stage00q_qc_json"].write_text(
            json.dumps({
                "stage": "00q_cube_qc", "run_id": self.run_id,
                "cube": {"wavelength_frame": "barycentric", "vbary_kms": 4.0},
                "m1_wavelength": {"status": "green", "offset_median_A": 0.0},
                "m3_flux": {"status": "green", "scale_factor": 1.0},
            }),
            encoding="utf-8",
        )
        self.cfg = {"run_id": self.run_id, "project_root": str(self.root),
                    "x11_canonical_method": "psffit"}

    def tearDown(self):
        self.tmp.cleanup()

    def _write_inputs(self, *, star=True, bunit=MUSE_NATIVE_BUNIT, levels=None):
        levels = levels or {}
        for method, key in METHOD_KEYS.items():
            _product(method, bunit=bunit, level=levels.get(method, 10.0)).write(self.paths[key])
        if star:
            star_product = _product("psffit", bunit=bunit, level=5000.0)
            star_product.header["APERTURE"] = "psffit_star"
            star_product.write(self.paths["spec_psffit_star"])

    def _run(self, **kwargs):
        self._write_inputs(**kwargs)
        product = compute_stage_x11_products(self.cfg, self.paths)
        written = write_stage_x11_products(product, self.cfg, self.paths)
        return written["qc"], written

    def test_table_lists_the_six_methods_plus_the_primary_canonical_first(self):
        qc, _ = self._run()
        table = qc["spectra"]["table"]
        self.assertEqual(qc["spectra"]["n_companion"], 6)
        self.assertEqual(qc["spectra"]["n_primary"], 1)
        self.assertEqual(table[0]["name"], "psffit")
        self.assertTrue(table[0]["canonical"])
        self.assertEqual(table[-1]["role"], "primary")
        self.assertTrue(qc["spectra"]["companions_share_grid"])
        for row in table:
            self.assertEqual(row["red_band_A"], list(SPECTRA_RED_BAND_A))
            self.assertIsNotNone(row["snr_median"])

    def test_the_primary_is_absent_from_the_table_when_c4_left_no_star(self):
        qc, _ = self._run(star=False)
        self.assertEqual(qc["spectra"]["n_primary"], 0)
        self.assertTrue(all(row["role"] == "companion" for row in qc["spectra"]["table"]))

    def test_the_unit_travels_to_the_table(self):
        qc, _ = self._run()
        self.assertEqual(qc["spectra"]["unit"], MUSE_NATIVE_BUNIT)
        self.assertTrue(qc["spectra"]["unit_consistent"])

    def test_a_product_without_unit_is_reported_as_an_open_issue(self):
        # Es el fallo que hubo que arreglar en sgf/lpm: unos productos con
        # unidad y otros sin ella, sin que nada lo dijera.
        qc, _ = self._run(bunit=None)
        self.assertFalse(qc["spectra"]["unit_consistent"])
        self.assertTrue(any("unidad" in issue for issue in qc["open_issues"]))

    def test_ratio_to_canonical_is_relative_to_the_canonical_continuum(self):
        qc, _ = self._run(levels={"aperture": 20.0})
        rows = {row["name"]: row for row in qc["spectra"]["table"]}
        self.assertIsNone(rows["psffit"]["ratio_to_canonical_red"])   # la referencia
        self.assertAlmostEqual(rows["aperture"]["ratio_to_canonical_red"], 2.0, places=3)
        self.assertAlmostEqual(rows["lpm"]["ratio_to_canonical_red"], 1.0, places=3)

    def test_a_continuum_below_zero_is_flagged_only_when_it_beats_the_error(self):
        # El continuo negativo del rojo es la sobre-sustraccion del halo AO
        # cromatico y merece aviso; un nivel compatible con cero (los metodos
        # que filtran el continuo) no lo es.
        qc, _ = self._run(levels={"aperture": -100.0, "lpm": -0.5})
        rows = {row["name"]: row for row in qc["spectra"]["table"]}
        self.assertIn("sobre-sustraccion", rows["aperture"]["caveat"])
        self.assertIsNone(rows["lpm"]["caveat"])

    def test_sgf_carries_its_continuum_caveat(self):
        qc, _ = self._run()
        rows = {row["name"]: row for row in qc["spectra"]["table"]}
        self.assertIn("continuo", rows["sgf"]["caveat"])
        self.assertIsNone(rows["lpm"]["caveat"])

    def test_the_figure_is_written_and_declared(self):
        qc, written = self._run()
        self.assertTrue(self.paths["stage_x11_spectra_png"].exists())
        self.assertEqual(written["plots"]["spectra"], str(self.paths["stage_x11_spectra_png"]))
        self.assertEqual(qc["plots"]["spectra"], str(self.paths["stage_x11_spectra_png"]))

    def test_the_notebook_can_redraw_the_figure_from_the_products_on_disk(self):
        # El notebook D2 la dibuja con esta misma funcion en vez de duplicar el
        # codigo: lee los productos ya escritos y no re-ejecuta la etapa.
        import matplotlib
        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt

        self._run()
        stage_dir = self.paths["spec_final_object"].parent
        fig, axes = definitive_spectra_figure(stage_dir, plt=plt)
        self.assertEqual(len(axes), 3)          # primaria + compañero + continuo
        plt.close(fig)

        # Un run al que le falte un metodo (o la primaria) no rompe la figura.
        self.paths["spec_calibrated_lpm_object"].unlink()
        self.paths["spec_calibrated_psffit_star"].unlink()
        fig, axes = definitive_spectra_figure(stage_dir, plt=plt)
        self.assertEqual(len(axes), 2)
        plt.close(fig)

    def test_the_raw_data_is_drawn_behind_the_smoothed_curves(self):
        """El panel del compañero enseña el dato, no solo su mediana móvil.

        La figura existe para comparar seis métodos, y para eso hay que
        suavizar o no se lee ninguno; pero enseñar SOLO el suavizado es lo que
        hace que un espectro parezca mejor de lo que es. Van los dos: el crudo
        de los seis detrás en gris, y encima la mediana.
        """
        import matplotlib
        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt

        self._run()
        stage_dir = self.paths["spec_final_object"].parent

        fig, axes = definitive_spectra_figure(stage_dir, plt=plt, smooth_channels=15)
        companion = axes[1]
        grises = [l for l in companion.get_lines() if l.get_color() == "0.8"]
        self.assertEqual(len(grises), len(METHOD_KEYS),
                         "falta el dato por canal de algún método")
        # Detrás: por encima solo puede quedar el suavizado y el canónico.
        self.assertTrue(all(l.get_zorder() < 2 for l in grises))
        etiquetas = [t.get_text() for t in companion.get_legend().get_texts()]
        self.assertIn("por canal, sin suavizar (los 6)", etiquetas)
        self.assertIn("gris", companion.get_title())
        plt.close(fig)

        # Sin suavizado no hay dos curvas por método: el crudo YA es la curva.
        fig, axes = definitive_spectra_figure(stage_dir, plt=plt, smooth_channels=1)
        companion = axes[1]
        self.assertEqual([l for l in companion.get_lines() if l.get_color() == "0.8"], [])
        self.assertIn("sin suavizar", companion.get_title())
        plt.close(fig)

    def test_an_empty_run_says_so_instead_of_drawing_an_empty_figure(self):
        import matplotlib.pyplot as plt

        with self.assertRaises(FileNotFoundError):
            definitive_spectra_figure(self.paths["spec_final_object"].parent, plt=plt)


if __name__ == "__main__":
    unittest.main()
