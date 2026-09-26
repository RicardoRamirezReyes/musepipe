import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
from astropy.io import fits

from musepipe.reduction.stream_combine import (
    StreamCombineError,
    build_stream_combine_plan,
    combine_streaming,
    load_weight_table,
    measure_primary_center,
    plan_from_dict,
    read_window,
    reweight_plan,
    shift_data_chunk,
    shift_variance_chunk,
    write_combined_cube,
)
from musepipe.stages.stage01_align import apply_spatial_alignment, bilinear_shift_variance

NZ = 24
NY = 60
NX = 62
CRVAL3 = 6200.0
CD3_3 = 1.25
PIXEL_DEG = 7.0e-06


def make_cube(
    path,
    *,
    y_center,
    x_center,
    amplitude=1000.0,
    background=1.0,
    variance=4.0,
    exptime=300.0,
    mjd_obs=59819.0,
    crval3=CRVAL3,
    cd_matrix=(-PIXEL_DEG, 0.0, 0.0, PIXEL_DEG),
    nz=NZ,
    shape=(NY, NX),
    spikes=(),
):
    """Write a synthetic per-exposure cube with one Gaussian star."""

    ny, nx = shape
    yy, xx = np.mgrid[0:ny, 0:nx].astype(np.float64)
    # sigma = 1 px keeps the star inside the 9x9 centroid box, so the flux
    # weighted centroid is not biased by truncation.
    star = amplitude * np.exp(-((yy - y_center) ** 2 + (xx - x_center) ** 2) / 2.0)
    plane = (star + background).astype(np.float32)
    data = np.repeat(plane[None, :, :], nz, axis=0)
    stat = np.full_like(data, float(variance))
    for z, y, x, value in spikes:
        data[z, y, x] = value

    primary = fits.Header()
    primary["EXPTIME"] = float(exptime)
    primary["MJD-OBS"] = float(mjd_obs)
    primary["OBJECT"] = "SYNTH"

    cube_header = fits.Header()
    cube_header["CTYPE1"] = "RA---TAN"
    cube_header["CTYPE2"] = "DEC--TAN"
    cube_header["CUNIT1"] = "deg"
    cube_header["CUNIT2"] = "deg"
    cube_header["CRPIX1"] = 1.0
    cube_header["CRPIX2"] = 1.0
    cube_header["CRVAL1"] = 247.8
    cube_header["CRVAL2"] = -24.5
    cube_header["CD1_1"] = float(cd_matrix[0])
    cube_header["CD1_2"] = float(cd_matrix[1])
    cube_header["CD2_1"] = float(cd_matrix[2])
    cube_header["CD2_2"] = float(cd_matrix[3])
    cube_header["CTYPE3"] = "AWAV"
    cube_header["CUNIT3"] = "Angstrom"
    cube_header["CRPIX3"] = 1.0
    cube_header["CRVAL3"] = float(crval3)
    cube_header["CD3_3"] = CD3_3
    cube_header["BUNIT"] = "10**(-20)*erg/s/cm**2/Angstrom"

    stat_header = cube_header.copy()
    stat_header["BUNIT"] = "(10**(-20)*erg/s/cm**2/Angstrom)**2"
    fits.HDUList(
        [
            fits.PrimaryHDU(header=primary),
            fits.ImageHDU(data=data, header=cube_header, name="DATA"),
            fits.ImageHDU(data=stat, header=stat_header, name="STAT"),
        ]
    ).writeto(path)
    return path


class ShiftKernelTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(7)
        self.cube = rng.normal(size=(NZ, 20, 21)).astype(np.float32)

    def test_chunking_does_not_change_the_shift(self):
        # A whole-cube scipy shift spline-filters the wavelength axis too, which
        # would make chunk boundaries visible. Ours must be chunk-invariant.
        whole = shift_data_chunk(self.cube, 0.37, -0.21)
        chunked = np.concatenate(
            [shift_data_chunk(self.cube[z : z + 5], 0.37, -0.21) for z in range(0, NZ, 5)],
            axis=0,
        )
        np.testing.assert_allclose(whole, chunked, rtol=0, atol=0)

    def test_matches_stage01_spatial_alignment(self):
        reference = apply_spatial_alignment(self.cube, 0.37, -0.21, "subpixel")
        np.testing.assert_allclose(shift_data_chunk(self.cube, 0.37, -0.21), reference, atol=2e-5)

    def test_zero_shift_is_the_identity(self):
        np.testing.assert_array_equal(shift_data_chunk(self.cube, 0.0, 0.0), self.cube)

    def test_variance_kernel_matches_stage01(self):
        var = np.abs(self.cube) + 1.0
        np.testing.assert_allclose(
            shift_variance_chunk(var, 0.37, -0.21),
            bilinear_shift_variance(var, 0.37, -0.21),
            rtol=1e-12,
            atol=0,
        )

    def test_variance_uses_squared_weights(self):
        var = np.ones((1, 4, 4), dtype=np.float64)
        # Four bilinear weights of 0.25 give 4 * 0.25^2 = 0.25, not 1.
        self.assertAlmostEqual(float(shift_variance_chunk(var, 0.5, 0.5)[0, 2, 2]), 0.25)

    def test_nan_survives_the_shift(self):
        cube = self.cube.copy()
        cube[:, 0, :] = np.nan
        self.assertTrue(np.isnan(shift_data_chunk(cube, 0.3, 0.0)[:, 0, :]).all())


class ReadWindowTests(unittest.TestCase):
    def test_window_inside_the_cube_is_a_plain_slice(self):
        data = np.arange(2 * 6 * 6, dtype=np.float32).reshape(2, 6, 6)
        np.testing.assert_array_equal(read_window(data, 0, 2, (1, 4, 2, 5)), data[0:2, 1:4, 2:5])

    def test_window_off_the_edge_is_nan_padded(self):
        data = np.ones((2, 6, 6), dtype=np.float32)
        window = read_window(data, 0, 2, (-2, 3, 0, 3))
        self.assertEqual(window.shape, (2, 5, 3))
        self.assertTrue(np.isnan(window[:, :2, :]).all())
        self.assertTrue(np.isfinite(window[:, 2:, :]).all())

    def test_window_entirely_outside_is_all_nan(self):
        data = np.ones((2, 6, 6), dtype=np.float32)
        self.assertTrue(np.isnan(read_window(data, 0, 2, (20, 25, 0, 3))).all())


class PlanTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

    def _plan(self, files, **kwargs):
        kwargs.setdefault("crop_npix", 20)
        kwargs.setdefault("pad", 4)
        kwargs.setdefault("centering_method", "peak")
        kwargs.setdefault("chunk_channels", 7)
        return build_stream_combine_plan(
            files, run_id="synthetic", output=str(self.tmp / "cube.fits"), **kwargs
        )

    def test_centroid_recovers_a_subpixel_position(self):
        path = make_cube(self.tmp / "a.fits", y_center=30.4, x_center=28.7)
        for method in ("peak", "maoppy"):
            measured = measure_primary_center(path, centering_method=method)
            self.assertAlmostEqual(measured["y_center"], 30.4, delta=0.02, msg=method)
            self.assertAlmostEqual(measured["x_center"], 28.7, delta=0.02, msg=method)
            self.assertEqual(measured["shape"], (NZ, NY, NX))

    def test_centroid_is_found_far_from_the_field_centre(self):
        # Dithers move the star well away from the middle of each exposure, so
        # the coarse pass has to search the whole field, not a central box.
        path = make_cube(self.tmp / "off.fits", y_center=12.3, x_center=51.6)
        measured = measure_primary_center(path, centering_method="peak")
        self.assertAlmostEqual(measured["y_center"], 12.3, delta=0.02)
        self.assertAlmostEqual(measured["x_center"], 51.6, delta=0.02)

    def test_shifts_stay_within_half_a_pixel(self):
        files = [
            make_cube(self.tmp / "a.fits", y_center=30.4, x_center=28.7),
            make_cube(self.tmp / "b.fits", y_center=26.9, x_center=33.2),
        ]
        plan = self._plan(files)
        for exposure in plan.exposures:
            self.assertLessEqual(abs(exposure.shift_y), 0.5)
            self.assertLessEqual(abs(exposure.shift_x), 0.5)
            self.assertTrue(exposure.in_bounds)

    def test_exptime_weighting_uses_the_header(self):
        files = [
            make_cube(self.tmp / "a.fits", y_center=30, x_center=30, exptime=300.0),
            make_cube(self.tmp / "b.fits", y_center=30, x_center=30, exptime=720.0),
        ]
        plan = self._plan(files, weight_mode="exptime")
        self.assertEqual([exp.weight for exp in plan.exposures], [300.0, 720.0])
        plan_none = self._plan(files, weight_mode="none")
        self.assertEqual([exp.weight for exp in plan_none.exposures], [1.0, 1.0])

    def test_rotated_cube_is_rejected(self):
        files = [
            make_cube(self.tmp / "a.fits", y_center=30, x_center=30),
            make_cube(
                self.tmp / "b.fits",
                y_center=30,
                x_center=30,
                cd_matrix=(-PIXEL_DEG * 0.7, PIXEL_DEG * 0.7, PIXEL_DEG * 0.7, PIXEL_DEG * 0.7),
            ),
        ]
        with self.assertRaises(StreamCombineError) as ctx:
            self._plan(files)
        self.assertIn("orientation", str(ctx.exception))

    def test_mismatched_channel_count_is_rejected(self):
        files = [
            make_cube(self.tmp / "a.fits", y_center=30, x_center=30),
            make_cube(self.tmp / "b.fits", y_center=30, x_center=30, nz=NZ - 2),
        ]
        with self.assertRaises(StreamCombineError):
            self._plan(files)

    def test_large_spectral_offset_is_rejected(self):
        files = [
            make_cube(self.tmp / "a.fits", y_center=30, x_center=30),
            make_cube(self.tmp / "b.fits", y_center=30, x_center=30, crval3=CRVAL3 + 0.5),
        ]
        with self.assertRaises(StreamCombineError) as ctx:
            self._plan(files)
        self.assertIn("CRVAL3", str(ctx.exception))

    def test_repeatability_is_measured_from_repeated_dither_positions(self):
        # Two exposures share a dither position and disagree by 0.10 px; the
        # third sits elsewhere and cannot contribute.
        files = [
            make_cube(self.tmp / "a.fits", y_center=30.00, x_center=30.0),
            make_cube(self.tmp / "b.fits", y_center=30.10, x_center=30.0),
            make_cube(self.tmp / "c.fits", y_center=24.00, x_center=36.0),
        ]
        repeatability = self._plan(files).reference["alignment_repeatability"]
        self.assertEqual(repeatability["n_groups"], 1)
        self.assertEqual(repeatability["n_exposures"], 2)
        # Each member deviates by half the pair difference.
        self.assertAlmostEqual(repeatability["centroid_repeatability_px"], 0.05, delta=0.01)

    def test_repeatability_ignores_the_same_dither_on_another_night(self):
        # A later night repeats the dither pattern and lands on the same rounded
        # pixel, but its sub-pixel position is unrelated and must not be scored
        # as a repeat measurement.
        files = [
            make_cube(self.tmp / "a.fits", y_center=30.05, x_center=30.0, mjd_obs=59819.0),
            make_cube(self.tmp / "b.fits", y_center=30.10, x_center=30.0, mjd_obs=59819.01),
            make_cube(self.tmp / "c.fits", y_center=29.60, x_center=30.4, mjd_obs=59822.0),
            make_cube(self.tmp / "d.fits", y_center=29.62, x_center=30.4, mjd_obs=59822.01),
        ]
        repeatability = self._plan(files).reference["alignment_repeatability"]
        self.assertEqual(repeatability["n_groups"], 2)
        self.assertEqual(repeatability["n_exposures"], 4)
        self.assertLess(repeatability["centroid_repeatability_px"], 0.05)

    def test_pointing_groups_expose_a_wcs_jump(self):
        # A cube whose WCS reference value is shifted in Dec puts the same star
        # 4 arcsec away on the sky without moving it on the detector.
        offset_deg = 4.0 / 3600.0
        files = [make_cube(self.tmp / "a.fits", y_center=30, x_center=30)]
        path = self.tmp / "b.fits"
        make_cube(path, y_center=30, x_center=30)
        with fits.open(path, mode="update") as hdul:
            for ext in (1, 2):
                hdul[ext].header["CRVAL2"] = hdul[ext].header["CRVAL2"] + offset_deg
        files.append(path)
        plan = self._plan(files)
        groups = plan.reference["astrometry_groups"]
        self.assertEqual(groups["n_groups"], 2)
        self.assertAlmostEqual(groups["max_group_offset_arcsec"], 4.0, delta=0.1)
        self.assertTrue(any("pointing groups" in w for w in plan.warnings))
        # The star is still aligned on the detector, so no shift is introduced.
        self.assertEqual(plan.exposures[0].shift_y, plan.exposures[1].shift_y)

    def test_small_spectral_offset_only_warns(self):
        files = [
            make_cube(self.tmp / "a.fits", y_center=30, x_center=30),
            make_cube(self.tmp / "b.fits", y_center=30, x_center=30, crval3=CRVAL3 + 0.004),
        ]
        plan = self._plan(files)
        self.assertTrue(any("CRVAL3" in w for w in plan.warnings))
        self.assertLess(plan.wavelength["crval3_spread_channels"], 0.05)


class CombineTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

    def _plan(self, files, **kwargs):
        kwargs.setdefault("crop_npix", 20)
        kwargs.setdefault("pad", 4)
        kwargs.setdefault("centering_method", "peak")
        kwargs.setdefault("chunk_channels", 7)
        return build_stream_combine_plan(
            files, run_id="synthetic", output=str(self.tmp / "cube.fits"), **kwargs
        )

    def _integer_placed(self, amplitudes=(1000.0, 1000.0, 1000.0), variance=4.0, exptimes=None, spikes=None):
        centers = [(30, 30), (26, 34), (33, 27)]
        exptimes = exptimes or [300.0] * len(amplitudes)
        spikes = spikes or [()] * len(amplitudes)
        return [
            make_cube(
                self.tmp / f"cube{i}.fits",
                y_center=cy,
                x_center=cx,
                amplitude=amp,
                variance=variance,
                exptime=t,
                spikes=sp,
            )
            for i, (amp, (cy, cx), t, sp) in enumerate(zip(amplitudes, centers, exptimes, spikes))
        ]

    def test_unweighted_mean_and_variance_are_exact(self):
        # Stars on exact integer pixels mean zero sub-pixel shift, so the
        # combine reduces to a plain mean and the arithmetic can be checked.
        files = self._integer_placed(amplitudes=(1000.0, 2000.0, 3000.0), variance=4.0)
        plan = self._plan(files, weight_mode="none")
        for exposure in plan.exposures:
            self.assertEqual((exposure.shift_y, exposure.shift_x), (0.0, 0.0))
        result = combine_streaming(plan)
        centre = plan.crop_npix // 2
        peak = result["data"][0, centre, centre]
        self.assertAlmostEqual(float(peak), (1000.0 + 2000.0 + 3000.0) / 3.0 + 1.0, places=3)
        # Variance of the mean of three equal variances: 3 * 4 / 9.
        self.assertAlmostEqual(float(result["stat"][0, centre, centre]), 4.0 / 3.0, places=5)
        self.assertEqual(int(result["count"].min()), 3)

    def test_exptime_weighting_matches_the_closed_form(self):
        files = self._integer_placed(amplitudes=(1000.0, 2000.0), exptimes=[300.0, 720.0], variance=4.0)
        plan = self._plan(files[:2], weight_mode="exptime")
        result = combine_streaming(plan)
        centre = plan.crop_npix // 2
        w = np.array([300.0, 720.0])
        values = np.array([1001.0, 2001.0])
        self.assertAlmostEqual(
            float(result["data"][0, centre, centre]), float((w * values).sum() / w.sum()), places=2
        )
        self.assertAlmostEqual(
            float(result["stat"][0, centre, centre]),
            float((w**2 * 4.0).sum() / w.sum() ** 2),
            places=6,
        )

    def test_star_lands_on_the_crop_centre(self):
        files = [
            make_cube(self.tmp / "a.fits", y_center=30.4, x_center=28.7),
            make_cube(self.tmp / "b.fits", y_center=26.9, x_center=33.2),
            make_cube(self.tmp / "c.fits", y_center=33.15, x_center=27.55),
        ]
        plan = self._plan(files, weight_mode="none")
        result = combine_streaming(plan)
        centre = plan.crop_npix // 2
        white = np.nanmedian(result["data"], axis=0)
        yy, xx = np.mgrid[0 : white.shape[0], 0 : white.shape[1]]
        weights = np.clip(white - np.nanmedian(white), 0, None)
        y_measured = float((yy * weights).sum() / weights.sum())
        x_measured = float((xx * weights).sum() / weights.sum())
        self.assertAlmostEqual(y_measured, centre, delta=0.05)
        self.assertAlmostEqual(x_measured, centre, delta=0.05)

    def test_sigclip_removes_a_cosmic_ray(self):
        spikes = [()] * 6
        # Voxel (2, 30, 30) is the crop centre of the first exposure's grid.
        spikes[0] = ((2, 30, 30, 5.0e6),)
        files = [
            make_cube(
                self.tmp / f"s{i}.fits",
                y_center=30,
                x_center=30,
                amplitude=1000.0,
                variance=4.0,
                spikes=sp,
            )
            for i, sp in enumerate(spikes)
        ]
        plan = self._plan(files, weight_mode="none", method="sigclip", sigclip_k=3.0)
        result = combine_streaming(plan)
        centre = plan.crop_npix // 2
        self.assertEqual(int(result["count"][2, centre, centre]), 5)
        self.assertAlmostEqual(float(result["data"][2, centre, centre]), 1001.0, places=2)
        self.assertGreater(result["qc"]["contributions_rejected"], 0)

    def test_mean_keeps_the_cosmic_ray(self):
        spikes = [()] * 6
        spikes[0] = ((2, 30, 30, 5.0e6),)
        files = [
            make_cube(self.tmp / f"m{i}.fits", y_center=30, x_center=30, spikes=sp)
            for i, sp in enumerate(spikes)
        ]
        plan = self._plan(files, weight_mode="none", method="mean")
        result = combine_streaming(plan)
        centre = plan.crop_npix // 2
        self.assertGreater(float(result["data"][2, centre, centre]), 1.0e5)

    def test_chunk_size_does_not_change_the_result(self):
        files = [
            make_cube(self.tmp / "a.fits", y_center=30.4, x_center=28.7),
            make_cube(self.tmp / "b.fits", y_center=26.9, x_center=33.2),
        ]
        small = combine_streaming(self._plan(files, chunk_channels=5))
        large = combine_streaming(self._plan(files, chunk_channels=NZ))
        np.testing.assert_allclose(small["data"], large["data"], rtol=0, atol=0, equal_nan=True)
        np.testing.assert_allclose(small["stat"], large["stat"], rtol=0, atol=0, equal_nan=True)

    def test_written_cube_matches_the_drs_layout(self):
        files = [
            make_cube(self.tmp / "a.fits", y_center=30.4, x_center=28.7),
            make_cube(self.tmp / "b.fits", y_center=26.9, x_center=33.2),
        ]
        plan = self._plan(files)
        result = combine_streaming(plan)
        output = self.tmp / "cube_telcorr.fits"
        write_combined_cube(result, plan, output, bunit="10**(-20)*erg/s/cm**2/Angstrom")
        with fits.open(output) as hdul:
            self.assertEqual(hdul[1].name, "DATA")
            self.assertEqual(hdul[2].name, "STAT")
            self.assertEqual(hdul[1].data.shape, (NZ, 20, 20))
            self.assertEqual(hdul[0].header["NEXP"], 2)
            self.assertEqual(hdul[0].header["COMBMETH"], "mean")
            # The star sits on the reference pixel of the output WCS.
            self.assertEqual(hdul[1].header["CRPIX1"], plan.crop_npix // 2 + 1)
            self.assertEqual(hdul[1].header["CRPIX2"], plan.crop_npix // 2 + 1)
            self.assertAlmostEqual(hdul[1].header["CRVAL3"], CRVAL3, places=3)
        with self.assertRaises(StreamCombineError):
            write_combined_cube(result, plan, output)


class TransmisionPorExposicionTests(unittest.TestCase):
    """La via «molecfit por exposicion» de A3, que entra ANTES de combinar.

    El punto duro es el STAT. El hook `transform` que ya existia devuelve solo
    DATA y no toca la varianza —correcto para restar un modelo determinista,
    falso para dividir por T—, asi que corregir por ahi dejaria un cubo cuyo
    error miente justo donde mas se corrigio. Por eso la T viaja en el plan.
    """

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)
        self.files = [
            make_cube(self.tmp / f"{i}.fits", y_center=30.0, x_center=30.0,
                      amplitude=1000.0, variance=4.0, exptime=300.0,
                      mjd_obs=59819.0 + 0.01 * i)
            for i in range(2)
        ]

    def _plan(self, **kwargs):
        kwargs.setdefault("crop_npix", 20)
        kwargs.setdefault("pad", 4)
        kwargs.setdefault("centering_method", "peak")
        kwargs.setdefault("chunk_channels", 7)
        kwargs.setdefault("weight_mode", "none")
        return build_stream_combine_plan(
            self.files, run_id="synthetic", output=str(self.tmp / "cube.fits"), **kwargs)

    def _transmision(self, plan, valores):
        rutas = {}
        for exposure, valor in zip(plan.exposures, valores):
            ruta = self.tmp / f"t_{exposure.index}.npy"
            np.save(ruta, np.full(NZ, float(valor)))
            rutas[exposure.file] = str(ruta)
        return rutas

    def test_data_is_divided_by_t_and_stat_by_t_squared(self):
        base = combine_streaming(self._plan())
        plan = self._plan()
        plan = replace(plan, transmission_by_exposure=self._transmision(plan, [0.5, 0.5]))

        result = combine_streaming(plan)

        centre = plan.crop_npix // 2
        np.testing.assert_allclose(result["data"][:, centre, centre],
                                   base["data"][:, centre, centre] / 0.5, rtol=1e-5)
        np.testing.assert_allclose(result["stat"][:, centre, centre],
                                   base["stat"][:, centre, centre] / 0.25, rtol=1e-5)

    def test_each_exposure_gets_its_own(self):
        # Lo que distingue esta granularidad: dos exposiciones, dos atmosferas.
        plan = self._plan()
        plan = replace(plan, transmission_by_exposure=self._transmision(plan, [0.5, 1.0]))

        result = combine_streaming(plan)

        centre = plan.crop_npix // 2
        # (1001/0.5 + 1001/1.0) / 2 = 1501.5
        self.assertAlmostEqual(float(result["data"][0, centre, centre]), 1501.5, places=1)

    def test_a_missing_exposure_is_refused_instead_of_half_corrected(self):
        plan = self._plan()
        rutas = self._transmision(plan, [0.5, 0.5])
        rutas.pop(plan.exposures[1].file)
        plan = replace(plan, transmission_by_exposure=rutas)

        with self.assertRaises(StreamCombineError):
            combine_streaming(plan)

    def test_a_wrong_length_is_caught_before_the_hours_of_work(self):
        plan = self._plan()
        ruta = self.tmp / "corta.npy"
        np.save(ruta, np.ones(NZ - 3))
        rutas = {exp.file: str(ruta) for exp in plan.exposures}
        plan = replace(plan, transmission_by_exposure=rutas)

        with self.assertRaises(StreamCombineError):
            combine_streaming(plan)

    def test_the_key_is_the_file_because_the_exposure_id_is_not_unique(self):
        # `exposure_id` sale del directorio padre, y los cubos por exposicion se
        # llaman todos DATACUBE_FINAL.fits: dos que compartan carpeta comparten
        # ID. Aqui los dos ficheros son hermanos, asi que el ID SI colisiona.
        plan = self._plan()
        self.assertEqual(plan.exposures[0].exposure_id, plan.exposures[1].exposure_id)

        plan = replace(plan, transmission_by_exposure=self._transmision(plan, [0.5, 1.0]))

        self.assertEqual(len(plan.transmission_by_exposure), 2, "una T por exposicion")

    def test_it_travels_in_the_plan_json_and_in_the_qc(self):
        plan = self._plan()
        rutas = self._transmision(plan, [0.5, 0.5])
        plan = replace(plan, transmission_by_exposure=rutas)

        vuelta = plan_from_dict(plan.as_dict())
        qc = combine_streaming(plan)["qc"]

        self.assertEqual(vuelta.transmission_by_exposure, rutas)
        self.assertEqual(set(qc["transmission_by_exposure"]), set(rutas))

    def test_without_it_the_combine_is_bit_for_bit_what_it_was(self):
        antes = combine_streaming(self._plan())
        despues = combine_streaming(replace(self._plan(), transmission_by_exposure={}))

        np.testing.assert_array_equal(antes["data"], despues["data"])
        np.testing.assert_array_equal(antes["stat"], despues["stat"])


if __name__ == "__main__":
    unittest.main()


def _c7_qc(weights, *, combine="invvar", groups=None, band=(8600.0, 9000.0), cube_weights=None):
    """Un `spec_perexp_qc.json` mínimo con la forma que C7 escribe.

    `weights` es la tabla de MEDIDA (`weights_normalised`, la que lee el
    combinado por defecto); la de CUBO (`weights_cube_normalised`, diagnostico)
    es la misma salvo que se pase otra. Un QC sin tabla de cubo (anterior al
    2026-09-14) se simula con `cube_weights={}`.
    """

    groups = groups or {"all": weights}

    def apertures(table):
        cube = table if cube_weights is None else cube_weights
        out = {"weights_normalised": dict(table), "n_eff": 1.0 / sum(v * v for v in table.values())}
        if cube:
            out["weights_cube_normalised"] = dict(cube)
            out["n_eff_cube"] = 1.0 / sum(v * v for v in cube.values())
        return {"box3": out, "box5": dict(out)}

    return {
        "stage": "x06_perexp", "spec_version": "C7", "run_id": "synthetic",
        "convention": {"combine": combine, "group_by": "none" if len(groups) == 1 else "night",
                       "weight_band_A": list(band),
                       "sigma": "dispersion de los controles combinados con los mismos pesos"},
        "groups": {name: {"apertures": apertures(table)} for name, table in groups.items()},
    }


class PesosPorVarianzaTests(unittest.TestCase):
    """`invvar`: los pesos no se deducen del cubo, llegan declarados desde C7."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)
        # Los cubos por exposición se llaman todos DATACUBE_FINAL.fits y el
        # exposure_id sale del directorio padre: se reproduce esa disposición.
        self.files = []
        for i, amp in enumerate((1000.0, 2000.0)):
            carpeta = self.tmp / f"2022-08-{29 + 2 * i}_MUSE.T{i}"
            carpeta.mkdir()
            self.files.append(make_cube(carpeta / "DATACUBE_FINAL.fits", y_center=30, x_center=30,
                                        amplitude=amp, variance=4.0, exptime=300.0))
        self.ids = [Path(f).parent.name for f in self.files]

    def _qc_path(self, payload, name="spec_perexp_qc.json"):
        path = self.tmp / name
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def _plan(self, **kwargs):
        kwargs.setdefault("crop_npix", 20)
        kwargs.setdefault("pad", 4)
        kwargs.setdefault("centering_method", "peak")
        kwargs.setdefault("chunk_channels", 7)
        return build_stream_combine_plan(
            self.files, run_id="synthetic", output=str(self.tmp / "cube.fits"), **kwargs)

    def test_the_table_sets_the_weights_and_the_closed_form_follows(self):
        table = {self.ids[0]: 0.9, self.ids[1]: 0.1}
        weights, source = load_weight_table(self._qc_path(_c7_qc(table)))
        self.assertEqual(weights, table)
        self.assertEqual(source["law"], "invvar")
        self.assertEqual(source["weight_band_A"], [8600.0, 9000.0])
        self.assertEqual(len(source["sha256"]), 64)

        plan = self._plan(weight_mode="invvar", weight_table=weights, weight_source=source)
        self.assertEqual([exp.weight for exp in plan.exposures], [0.9, 0.1])
        result = combine_streaming(plan)
        centre = plan.crop_npix // 2
        w = np.array([0.9, 0.1])
        values = np.array([1001.0, 2001.0])
        self.assertAlmostEqual(float(result["data"][0, centre, centre]),
                               float((w * values).sum() / w.sum()), places=2)
        self.assertAlmostEqual(float(result["stat"][0, centre, centre]),
                               float((w**2 * 4.0).sum() / w.sum() ** 2), places=6)
        self.assertEqual(result["qc"]["weight_mode"], "invvar")
        self.assertEqual(result["qc"]["weight_source"]["sha256"], source["sha256"])
        self.assertEqual(result["qc"]["weights_by_exposure"], table)

        # La procedencia sobrevive al JSON y llega a la cabecera del cubo.
        again = plan_from_dict(json.loads(json.dumps(plan.as_dict())))
        self.assertEqual(again.weight_source, source)
        self.assertEqual([exp.weight for exp in again.exposures], [0.9, 0.1])
        out = write_combined_cube(result, plan, self.tmp / "out.fits")
        with fits.open(out) as hdul:
            self.assertEqual(hdul[0].header["COMBWGT"], "invvar")
            self.assertEqual(hdul[0].header["COMBWSRC"], "spec_perexp_qc.json")
            self.assertEqual(hdul[0].header["COMBWSHA"], source["sha256"][:16])

    def test_invvar_without_a_table_or_without_provenance_is_an_error(self):
        with self.assertRaisesRegex(StreamCombineError, "provenance"):
            self._plan(weight_mode="invvar", weight_table={self.ids[0]: 0.5, self.ids[1]: 0.5})
        with self.assertRaisesRegex(StreamCombineError, "weight table"):
            self._plan(weight_mode="invvar", weight_source={"path": "x"})

    def test_a_table_for_another_harvest_is_refused(self):
        source = {"path": "x", "sha256": "0" * 64}
        # Falta una exposición: no se rellena con nada.
        with self.assertRaisesRegex(StreamCombineError, "1 planned exposures without weight"):
            self._plan(weight_mode="invvar", weight_table={self.ids[0]: 1.0}, weight_source=source)
        # Sobra una: la tabla describe otro conjunto (30 pesos para 29 cubos).
        with self.assertRaisesRegex(StreamCombineError, "1 weights without exposure"):
            self._plan(weight_mode="invvar", weight_source=source,
                       weight_table={self.ids[0]: 0.5, self.ids[1]: 0.4, "otra": 0.1})
        # Un peso nulo o no finito tampoco vale.
        with self.assertRaisesRegex(StreamCombineError, "finite, positive"):
            self._plan(weight_mode="invvar", weight_source=source,
                       weight_table={self.ids[0]: 1.0, self.ids[1]: 0.0})

    def test_the_qc_must_be_ungrouped_and_inverse_variance(self):
        table = {self.ids[0]: 0.9, self.ids[1]: 0.1}
        with self.assertRaisesRegex(StreamCombineError, "not 'invvar'"):
            load_weight_table(self._qc_path(_c7_qc(table, combine="exptime")))
        por_noche = _c7_qc(table, groups={"20220829": {self.ids[0]: 1.0}, "20220831": {self.ids[1]: 1.0}})
        with self.assertRaisesRegex(StreamCombineError, "ONE group"):
            load_weight_table(self._qc_path(por_noche))
        with self.assertRaisesRegex(StreamCombineError, "no aperture 'box7'"):
            load_weight_table(self._qc_path(_c7_qc(table)), aperture="box7")

    def test_the_measurement_table_is_the_default_and_the_cube_one_is_explicit(self):
        # La de medida lleva la apcorr_i dentro, y por eso es la S/N de cada
        # exposicion en el companero. La de cubo (sin apcorr) se probo el
        # 2026-09-14 y devolvio a la noche mala el 49 % del peso: queda como
        # diagnostico, y hay que pedirla.
        medida = {self.ids[0]: 0.5, self.ids[1]: 0.5}
        cubo = {self.ids[0]: 0.8, self.ids[1]: 0.2}
        qc = self._qc_path(_c7_qc(medida, cube_weights=cubo))
        weights, source = load_weight_table(qc)
        self.assertEqual(weights, medida)
        self.assertEqual((source["kind"], source["field"]), ("measurement", "weights_normalised"))
        self.assertAlmostEqual(source["n_eff"], 2.0)
        weights, source = load_weight_table(qc, kind="cube")
        self.assertEqual(weights, cubo)
        self.assertEqual(source["field"], "weights_cube_normalised")
        self.assertAlmostEqual(source["n_eff"], 1.0 / (0.64 + 0.04))
        with self.assertRaises(ValueError):
            load_weight_table(qc, kind="pixel")
        # Un QC de C7 anterior al 2026-09-14 no trae la tabla de cubo: el
        # defecto lo lee igual, y pedir la de cubo falla en vez de leer la otra.
        viejo = self._qc_path(_c7_qc(medida, cube_weights={}), name="viejo.json")
        self.assertEqual(load_weight_table(viejo)[0], medida)
        with self.assertRaisesRegex(StreamCombineError, "weights_cube_normalised.*re-run C7"):
            load_weight_table(viejo, kind="cube")

    def test_reweight_keeps_the_geometry_and_changes_only_the_weights(self):
        plan = self._plan(weight_mode="exptime")
        self.assertEqual([exp.weight for exp in plan.exposures], [300.0, 300.0])
        table = {self.ids[0]: 0.98, self.ids[1]: 0.02}
        weights, source = load_weight_table(self._qc_path(_c7_qc(table)))
        nuevo = reweight_plan(plan, "invvar", weight_table=weights, weight_source=source)
        self.assertEqual(nuevo.weight_mode, "invvar")
        self.assertEqual([exp.weight for exp in nuevo.exposures], [0.98, 0.02])
        for antes, despues in zip(plan.exposures, nuevo.exposures):
            self.assertEqual(replace(antes, weight=0.0), replace(despues, weight=0.0))
        self.assertEqual(nuevo.reference, plan.reference)
        self.assertEqual(nuevo.wavelength, plan.wavelength)
        # Y de vuelta a `exptime` la procedencia de la tabla desaparece.
        vuelta = reweight_plan(nuevo, "exptime")
        self.assertEqual([exp.weight for exp in vuelta.exposures], [300.0, 300.0])
        self.assertEqual(vuelta.weight_source, {})
        with self.assertRaises(ValueError):
            reweight_plan(plan, "quality")

    def test_the_reweight_script_reports_the_share_by_night(self):
        import subprocess, sys
        plan = self._plan(weight_mode="exptime")
        plan_path = self.tmp / "plan.json"
        plan_path.write_text(json.dumps(plan.as_dict()), encoding="utf-8")
        qc = self._qc_path(_c7_qc({self.ids[0]: 0.97, self.ids[1]: 0.03}))
        out_plan = self.tmp / "plan_invvar.json"
        r = subprocess.run(
            [sys.executable, str(Path(__file__).resolve().parent.parent / "scripts" / "reweight_stream_combine_plan.py"),
             "--plan", str(plan_path), "--out-plan", str(out_plan), "--weight", "invvar",
             "--weight-table", str(qc), "--run-id", "synthetic_invvar", "--output", str(self.tmp / "nuevo.fits")],
            capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("20220829:  50.00 % ->  97.00 %", r.stdout)
        self.assertIn("20220831:  50.00 % ->   3.00 %", r.stdout)
        nuevo = plan_from_dict(json.loads(out_plan.read_text(encoding="utf-8")))
        self.assertEqual(nuevo.run_id, "synthetic_invvar")
        self.assertEqual(nuevo.output, str(self.tmp / "nuevo.fits"))
        self.assertEqual([exp.weight for exp in nuevo.exposures], [0.97, 0.03])
        self.assertEqual(nuevo.weight_source["path"], str(qc))
        # Sin --overwrite no pisa el plan que acaba de escribir.
        r2 = subprocess.run(
            [sys.executable, str(Path(__file__).resolve().parent.parent / "scripts" / "reweight_stream_combine_plan.py"),
             "--plan", str(plan_path), "--out-plan", str(out_plan), "--weight", "none"],
            capture_output=True, text=True)
        self.assertEqual(r2.returncode, 2)
        self.assertIn("refusing to overwrite", r2.stderr)
