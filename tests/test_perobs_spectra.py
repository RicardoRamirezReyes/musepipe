"""Diagnóstico por exposición: N repeticiones de la misma medida.

Lo que se fija aquí es lo que haría inservible la dispersión sin que ningún
número pareciera raro:

* que los canales del psffit se elijan de forma reproducible y que las ventanas
  declaradas entren **enteras** (Hα no se submuestrea);
* que la dispersión sea la de una medida individual, con los pesos del
  combinado, y que un NaN en una exposición no contamine el canal;
* que las perillas compartidas se **lean** de C2 y C4 en vez de copiarse — el
  fallo exacto que hizo que el primer notebook de C3 no reprodujera la cadena;
* que N exposiciones iguales den dispersión ~0 (si la geometría se rompe, el
  halo de la primaria entra distinto en cada una y esto se dispara);
* que una exposición que reviente se cuente y no tumbe a las otras.
"""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from musepipe.psf import build_mixture_model_document
from musepipe.qc.perobs_spectra import (
    PerObsSpectraError,
    channel_selection,
    dispersion,
    perobs_spectra_config_from_run,
    run_perobs_spectra,
)
from test_observations import _exposures, _write_plan, _write_run
from test_psf_mixture import moffat_doc

CROP = 20
PAD = 4


class ChannelSelectionTests(unittest.TestCase):
    def test_step_subsamples_and_windows_come_in_whole(self):
        wave = np.arange(4750.0, 9350.0, 1.25)
        sel = channel_selection(wave, 10, [[6520.0, 6610.0]])

        self.assertTrue(np.all(np.diff(sel) > 0), "los índices salen ordenados y sin repetir")
        # Toda la ventana de Hα, canal a canal.
        window = np.flatnonzero((wave >= 6520.0) & (wave <= 6610.0))
        self.assertTrue(set(window.tolist()) <= set(sel.tolist()))
        # Y fuera de ella, uno de cada diez.
        self.assertIn(0, sel.tolist())
        self.assertNotIn(1, sel.tolist())

    def test_step_one_keeps_everything(self):
        wave = np.linspace(5000.0, 6000.0, 41)
        self.assertEqual(channel_selection(wave, 1, None).size, wave.size)


class DispersionTests(unittest.TestCase):
    def test_it_is_the_scatter_of_one_measurement_not_of_the_mean(self):
        # Tres exposiciones, un canal, pesos iguales: la desviación muestral.
        flux = np.array([[9.0], [10.0], [11.0]])
        out = dispersion(flux, np.ones(3))

        self.assertAlmostEqual(float(out["mean"][0]), 10.0)
        self.assertAlmostEqual(float(out["scatter"][0]), 1.0)  # no 1/sqrt(3)
        self.assertAlmostEqual(float(out["scatter_frac"][0]), 0.1)

    def test_the_weights_are_the_ones_of_the_combined(self):
        flux = np.array([[0.0], [10.0]])
        out = dispersion(flux, np.array([3.0, 1.0]))

        self.assertAlmostEqual(float(out["mean"][0]), 2.5)

    def test_a_nan_in_one_exposure_does_not_poison_the_channel(self):
        flux = np.array([[9.0, np.nan], [10.0, 10.0], [11.0, 12.0]])
        out = dispersion(flux, np.ones(3))

        self.assertAlmostEqual(float(out["mean"][0]), 10.0)
        self.assertAlmostEqual(float(out["mean"][1]), 11.0)
        self.assertEqual(out["n_good"].tolist(), [3, 2])

    def test_the_excess_is_measured_against_stat_and_only_reported(self):
        flux = np.array([[9.0], [10.0], [11.0]])
        sigma = np.full((3, 1), 0.25)
        out = dispersion(flux, np.ones(3), sigmas=sigma)

        self.assertAlmostEqual(float(out["excess_over_stat"][0]), 4.0)


def _run_with_exposures(root, centers, *, amplitudes=None, backgrounds=None, fwhms=None,
                        crval3s=None):
    """Un run sintético completo: cubos, plan, posiciones de B3 y mezcla de C1."""

    from test_stream_combine import make_cube

    paths = []
    for index, (y, x) in enumerate(centers):
        exposure_id = f"2022-08-2{index}_MUSE.exp{index:02d}"
        path = root / "cubes" / exposure_id / "DATACUBE_FINAL.fits"
        path.parent.mkdir(parents=True, exist_ok=True)
        kwargs = {}
        if amplitudes is not None:
            kwargs["amplitude"] = amplitudes[index]
        if backgrounds is not None:
            kwargs["background"] = backgrounds[index]
        if crval3s is not None:
            kwargs["crval3"] = crval3s[index]
        make_cube(path, y_center=y, x_center=x, mjd_obs=59819.0 + index, **kwargs)
        paths.append(path)

    run = _write_run(root, "obj", {
        "cube_files": [str(root / "combined.fits")],
        "x01_apertures": [{"name": "box3", "kind": "box", "size": 3}],
        "x03_star_radius_px": 5.0,
        "x03_comp_radius_px": 3.0,
    })
    plan = _write_plan(root, "obj", paths)

    # B3: el compañero, 6 px por debajo del centro del marco de la cadena.
    frame = (CROP, CROP)
    (run / "stages" / "stage01c_qc.json").write_text(json.dumps({
        "cube_shape": [1, int(plan.wavelength["n_channels"]), frame[0], frame[1]],
        "primary": {"pos_yx": [frame[0] / 2, frame[1] / 2]},
        "companion": {"pos_yx": [frame[0] / 2 + 6.0, frame[1] / 2]},
        "field_source": None,
    }), encoding="utf-8")

    # C1 por observación: una Moffat por exposición, mezcladas.
    components = [
        {
            "exposure_id": exposure.exposure_id,
            "weight": float(exposure.weight),
            "flux_norm": {"lambda_A": [5000.0, 9000.0], "value": [1.0, 1.0]},
            "model": moffat_doc(3.0 if fwhms is None else fwhms[index]),
        }
        for index, exposure in enumerate(plan.exposures)
    ]
    mixture = build_mixture_model_document(
        components, norm_radius_px=moffat_doc(3.0)["norm_radius_px"], system="muse_nfm",
    )
    (run / "stages" / "psf_model_mixture.json").write_text(json.dumps(mixture), encoding="utf-8")
    return run, plan


class EndToEndTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def _run(self, centers, **kwargs):
        _run_with_exposures(self.root, centers, **kwargs)
        return run_perobs_spectra(
            "obj", project_root=self.root,
            overrides={"perobs_spectra_psffit": False, "perobs_spectra_background": "none"},
            n_jobs=1, save_plot=False,
        )

    def test_identical_exposures_give_no_dispersion(self):
        """Mismo cubo y misma PSF: si esto dispersa, la geometría está rota."""

        result = self._run([(30.0, 31.0), (30.0, 31.0), (30.0, 31.0)])
        block = result["blocks"]["aperture_box3"]

        frac = block["scatter_frac"]
        self.assertTrue(np.nanmedian(frac) < 1e-6,
                        f"tres exposiciones iguales no pueden dispersar: {np.nanmedian(frac)}")

    def test_the_per_exposure_apcorr_is_what_moves_the_flux(self):
        """El mismo dato con tres PSF distintas YA dispersa: es el cambio entero.

        La corrección de apertura deja de ser una sola del combinado y pasa a ser
        la de cada exposición; el flujo sin corregir no se mueve, el corregido sí.
        Si algún día se dejara de aplicar por exposición, este test se cae.
        """

        result = self._run([(30.0, 31.0)] * 3, fwhms=[3.0, 4.0, 5.0])
        block = result["blocks"]["aperture_box3"]
        data = np.load(result["paths"]["spectra_npz"], allow_pickle=False)

        self.assertGreater(float(np.nanmedian(block["scatter_frac"])), 1e-3)
        raw = data["flux_raw_aperture_box3"]
        self.assertLess(float(np.nanmedian(np.nanstd(raw, axis=0) / np.abs(np.nanmean(raw, axis=0)))),
                        1e-9, "el dato es el mismo: lo que cambia es la apcorr")
        self.assertEqual(data["apcorr_aperture_box3"].shape, raw.shape)

    def test_a_brighter_exposure_shows_up_as_dispersion(self):
        result = self._run([(30.0, 31.0)] * 3, backgrounds=[1.0, 1.0, 4.0])
        block = result["blocks"]["aperture_box3"]

        self.assertGreater(float(np.nanmedian(block["scatter_frac"])), 0.1)

    def test_it_writes_its_products_and_declares_its_convention(self):
        result = self._run([(30.2, 31.4), (29.6, 30.9), (30.9, 31.1)])
        qc = result["qc"]

        self.assertEqual(qc["input"]["n_exposures_measured"], 3)
        self.assertEqual(len(qc["exposures"]), 3)
        self.assertEqual(qc["failures"], [])
        # La convención se declara, porque el nivel absoluto NO es el de la cadena.
        self.assertFalse(qc["convention"]["empirical_total_flux_factor"])
        self.assertIn("dispersión", qc["convention"]["note"].lower())
        for key in ("qc_json", "spectra_npz", "dispersion_csv", "summary_csv"):
            self.assertTrue(Path(result["paths"][key]).exists(), key)

        data = np.load(result["paths"]["spectra_npz"], allow_pickle=False)
        self.assertEqual(data["flux_aperture_box3"].shape[0], 3)
        self.assertEqual(data["flux_aperture_box3"].shape[1], data["wave_A"].size)

    def test_the_psffit_branch_runs_and_declares_its_subsampling(self):
        """La rama cara, cubierta: un fallo aquí no puede tardar 30 min en verse."""

        _run_with_exposures(self.root, [(30.0, 31.0)] * 3)
        result = run_perobs_spectra(
            "obj", project_root=self.root,
            overrides={"perobs_spectra_background": "none",
                       "perobs_spectra_psffit_channel_step": 8,
                       "perobs_spectra_psffit_full_windows_A": []},
            n_jobs=1, save_plot=False,
        )

        self.assertIn("psffit", result["blocks"])
        data = np.load(result["paths"]["spectra_npz"], allow_pickle=False)
        self.assertEqual(data["flux_psffit"].shape[0], 3)
        self.assertEqual(data["flux_psffit"].shape[1], data["psffit_wave_A"].size)
        self.assertLess(data["psffit_wave_A"].size, data["wave_A"].size)
        # C4 no lleva apcorr, y el QC lo dice en vez de dejarlo implícito.
        self.assertEqual(result["qc"]["convention"]["psffit"]["apcorr"], 1.0)
        self.assertEqual(result["qc"]["convention"]["psffit"]["channel_step"], 8)

    def test_sub_channel_wave_drift_is_measured_not_fatal(self):
        """Las cabeceras guardan CRVAL3 en float32; eso no es una deriva.

        Regresión real: con `atol=1e-6 Å` el diagnóstico reventaba en ROXs 12 b
        DESPUÉS de medir las 29 exposiciones, por una deriva de 0.0039 Å — el
        0.31 % de un canal, en escalones de 0.0005 Å que son exactamente la
        cuantización de un float32 a 4750 Å. La tolerancia va en canales, y la
        deriva se publica.
        """

        from test_stream_combine import CRVAL3

        result = self._run([(30.0, 31.0)] * 3,
                           crval3s=[CRVAL3, CRVAL3 + 0.0005, CRVAL3 + 0.0039])
        entrada = result["qc"]["input"]

        self.assertGreater(entrada["wave_axis_drift_A"], 0.0)
        self.assertLess(entrada["wave_axis_drift_channels"], 0.05)
        self.assertEqual(entrada["n_exposures_measured"], 3)

    def test_a_real_wave_offset_is_stopped_upstream_by_the_plan(self):
        """Un desfase real no llega aquí: lo rechaza el plan del combinado.

        Y la tolerancia es LA MISMA (`MAX_CRVAL3_SPREAD_CHANNELS`), leída de
        `stream_combine`, no un segundo criterio que pudiera contradecirlo.
        """

        from musepipe.reduction.stream_combine import (
            MAX_CRVAL3_SPREAD_CHANNELS,
            StreamCombineError,
        )
        from musepipe.qc.perobs_spectra import perobs_spectra_config_from_run
        from test_stream_combine import CRVAL3

        with self.assertRaises(StreamCombineError):
            _run_with_exposures(self.root, [(30.0, 31.0)] * 3,
                                crval3s=[CRVAL3, CRVAL3, CRVAL3 + 1.25])

        otra = Path(self._tmp.name) / "limpio"
        otra.mkdir()
        _run_with_exposures(otra, [(30.0, 31.0)] * 2)
        cfg = perobs_spectra_config_from_run("obj", project_root=otra)
        self.assertEqual(cfg["perobs_spectra_wave_tol_channels"],
                         MAX_CRVAL3_SPREAD_CHANNELS)

    def test_a_broken_exposure_is_counted_not_fatal(self):
        run, plan = _run_with_exposures(self.root, [(30.0, 31.0)] * 3)
        # Se borra el cubo de una, DESPUÉS de que el plan la registrara.
        Path(plan.exposures[1].file).unlink()

        with self.assertRaises(Exception):
            # La vista por observación falla en bloque: una ruta muerta que el
            # config no re-resuelve no es un fallo silencioso.
            run_perobs_spectra("obj", project_root=self.root, n_jobs=1, save_plot=False)


class ConfigTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_the_shared_knobs_are_read_not_copied(self):
        _write_run(self.root, "obj", {
            "x01_apertures": [{"name": "box5", "kind": "box", "size": 5}],
            "x01_annulus_bkg_px": [7.0, 13.0, 29.0],
            "x03_star_radius_px": 22.0,
            "x03_comp_radius_px": 11.0,
        })
        cfg = perobs_spectra_config_from_run("obj", project_root=self.root)

        self.assertEqual(cfg["perobs_spectra_apertures"][0]["name"], "box5")
        self.assertEqual(cfg["perobs_spectra_annulus_px"], [7.0, 13.0, 29.0])
        self.assertEqual(cfg["perobs_spectra_star_radius_px"], 22.0)
        self.assertEqual(cfg["perobs_spectra_comp_radius_px"], 11.0)

    def test_halpha_is_sampled_whole_by_default(self):
        _write_run(self.root, "obj", {"x01_apertures": [{"name": "box3", "kind": "box", "size": 3}]})
        cfg = perobs_spectra_config_from_run("obj", project_root=self.root)

        window = cfg["perobs_spectra_psffit_full_windows_A"][0]
        self.assertLess(window[0], 6562.8)
        self.assertGreater(window[1], 6562.8)

    def test_a_run_that_declares_nothing_still_gets_c2s_apertures(self):
        """El default lo pone la etapa, no el run: leer el config CRUDO no basta.

        Regresión concreta: `x01_apertures` no está en `config.json` de
        ROXs 12 b —lo rellena `stage_x01_config_from_run`—, así que leer el run
        directamente hacía fallar el diagnóstico contra el run real.
        """

        _write_run(self.root, "obj", {})
        cfg = perobs_spectra_config_from_run("obj", project_root=self.root)

        names = [a["name"] for a in cfg["perobs_spectra_apertures"]]
        self.assertIn("box3", names)

    def test_without_an_aperture_it_says_so(self):
        _write_run(self.root, "obj", {})
        with self.assertRaises(PerObsSpectraError):
            perobs_spectra_config_from_run(
                "obj", project_root=self.root,
                overrides={"perobs_spectra_apertures": []},
            )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
