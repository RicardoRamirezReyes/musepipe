"""`optimal_ls` extrae del cubo crudo, como C2, no del residual de 04b.

La variante `ls` existe para ser **comparable 1:1 con C2** y aislar así la
ganancia del ponderado óptimo (spec C3 §3.1). Pero extraía del residual de 04b
y encima le restaba la mediana de anillo, es decir dos tratamientos de fondo
sobre un cubo que NO es homogéneo: 04b ajusta una superficie local solo
alrededor del objeto, así que el anillo veía ~3/px en el compañero y ~13/px en
los controles. Medido en los dos objetos, esa segunda resta producía ~93% del
continuo negativo que hace que G1 rechace el método.

Aquí se fija el comportamiento nuevo y que la vuelta atrás sigue disponible.
"""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from astropy.io import fits

from musepipe.extraction.product import SpectrumProduct
from musepipe.stages.stage_x02_optimal import (
    compute_stage_x02_products,
    stage_x02_paths,
    write_stage_x02_products,
)
from tests.test_optimal_analytic import constant_model_doc
from tests.test_optimal_stage import psf_cube

PRIMARY_YX = (30.0, 30.0)
COMPANION_YX = (30.0, 42.0)
COMPANION_FLUX = 80.0


def _build_run(root, run_id="synthetic_ls"):
    """Run sintético con la asimetría real: 04b solo tocó al objeto.

    El residual de 04b lleva el compañero **y el fondo ya quitado**; el cubo
    crudo lleva primaria + compañero + ese mismo fondo. Es la situación en la
    que la doble resta muerde.
    """
    model = constant_model_doc()
    wave = np.linspace(6500.0, 6540.0, 5)
    companion = psf_cube(model, wave, COMPANION_YX, COMPANION_FLUX)
    primary = psf_cube(model, wave, PRIMARY_YX, 1000.0)
    fondo = np.full_like(companion, 5.0)
    paths = stage_x02_paths(run_id, project_root=root)
    paths["paths"].ensure_base_dirs()
    hdr = fits.Header()
    hdr["WMIN"] = float(wave[0])
    hdr["DW"] = float(np.nanmedian(np.diff(wave)))
    hdr["BUNIT"] = "native"
    # residual de 04b: sin fondo (04b lo quitó alrededor del objeto)
    fits.PrimaryHDU(companion, header=hdr).writeto(paths["cube_residual_object"])
    # cubo crudo: primaria + compañero + fondo
    fits.HDUList([
        fits.PrimaryHDU(),
        fits.ImageHDU((primary + companion + fondo)[None], name="CUBES"),
        fits.ImageHDU(wave, name="WAVELENGTH"),
        fits.ImageHDU(np.ones((1,) + companion.shape, dtype=np.float32), name="STAT"),
    ]).writeto(paths["stage02_cube_fits"])
    paths["stage01c_qc_json"].write_text(json.dumps({
        "stage": "01c", "run_id": run_id,
        "primary": {"pos_yx": list(PRIMARY_YX)},
        "companion": {"pos_yx": list(COMPANION_YX)},
    }), encoding="utf-8")
    paths["psf_model_json"].write_text(json.dumps(model), encoding="utf-8")
    return paths, run_id


def _config(run_id, root, **extra):
    cfg = {
        "run_id": run_id,
        "project_root": str(root),
        "x02_window_radius_px": 6.0,
        "x02_aperture_correction": "psf_growth_curve",
        "x02_primary_fit_radius_px": 18.0,
        # El anillo es lo que dispara el tratamiento doble; sin él no hay caso.
        "x02_local_bkg_annulus_px": [8.0, 12.0, 20.0],
    }
    cfg.update(extra)
    return cfg


class WingsIntactLsTests(unittest.TestCase):
    def test_ls_extracts_from_the_raw_cube_and_says_so(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths, run_id = _build_run(root)
            product = compute_stage_x02_products(_config(run_id, root), paths)
            issues = " ".join(product.qc["open_issues"])
            self.assertIn("raw stage02 cube", issues)
            self.assertIn("twice at the companion", issues)

    def test_the_historical_behaviour_is_still_reachable(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths, run_id = _build_run(root)
            cfg = _config(run_id, root, x02_wings_intact_ls=False)
            product = compute_stage_x02_products(cfg, paths)
            self.assertNotIn("raw stage02 cube", " ".join(product.qc["open_issues"]))

    def test_the_two_paths_give_different_flux(self):
        """Cambiar de cubo cambia el resultado, que es de lo que trata el cambio.

        Aquí NO se afirma cuál es mejor: en este sintético el residual de 04b es
        perfecto por construcción (se quitó la primaria entera y no quedó
        pedestal), así que la doble resta apenas cuesta y el histórico sale
        incluso más cerca del flujo verdadero. Cuál es mejor se decide con los
        datos reales, donde 04b deja ~3/px en el anillo del compañero y ~13/px
        en el de los controles: allí medido, la doble resta se lleva la mediana
        de la banda roja de -140 a -2045.
        """
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths, run_id = _build_run(root)
            nuevo = compute_stage_x02_products(_config(run_id, root), paths)
            historico = compute_stage_x02_products(
                _config(run_id, root, x02_wings_intact_ls=False), paths
            )
            f_nuevo = float(np.nanmedian(nuevo.extractions["ls"].product.flux))
            f_hist = float(np.nanmedian(historico.extractions["ls"].product.flux))
            self.assertNotAlmostEqual(f_nuevo, f_hist, places=3)
            # Los dos siguen recuperando el compañero, no una cifra cualquiera.
            for f in (f_nuevo, f_hist):
                self.assertLess(abs(f - COMPANION_FLUX) / COMPANION_FLUX, 0.10)

    def test_psfsub_is_untouched_by_the_change(self):
        # La otra variante ya salía del cubo crudo: no debe moverse ni un bit.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths, run_id = _build_run(root)
            nuevo = compute_stage_x02_products(_config(run_id, root), paths)
            historico = compute_stage_x02_products(
                _config(run_id, root, x02_wings_intact_ls=False), paths
            )
            np.testing.assert_array_equal(
                nuevo.extractions["psfsub"].product.flux,
                historico.extractions["psfsub"].product.flux,
            )

    def test_without_an_annulus_nothing_changes(self):
        # Sin anillo no hay segunda resta que evitar: `ls` sigue saliendo del
        # residual de 04b, que es el comportamiento correcto en ese caso.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths, run_id = _build_run(root)
            cfg = _config(run_id, root)
            cfg["x02_local_bkg_annulus_px"] = None
            product = compute_stage_x02_products(cfg, paths)
            self.assertNotIn("raw stage02 cube", " ".join(product.qc["open_issues"]))
            np.testing.assert_allclose(
                product.extractions["ls"].product.flux, COMPANION_FLUX, rtol=1e-6, atol=1e-6
            )


if __name__ == "__main__":
    unittest.main()
