import csv
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from musepipe.stages.stage_e01_psf import (
    StageE01Product,
    _b3_chromatic_track,
    _centroid_vs_b3,
    _ring_qc_summary,
    _write_psfao_csv,
    _write_summary_plot,
)
from musepipe.stages.stage_e01_psfao import (
    build_psfao_model_document,
    DEFAULT_X0,
    PSFAO_PARAM_NAMES,
    _psfao_param_errors,
    fit_psfao_bins,
)

#: Los errores formales que `fit_bin` devuelve ahora como octavo elemento. Los
#: tests que solo miran convergencia no dependen de ellos, pero la tupla tiene
#: que traerlos o `fit_psfao_bins` no desempaqueta.
ERRORES_FALSOS = {f"{name}_err": 0.01 for name in PSFAO_PARAM_NAMES}
ERRORES_FALSOS.update({"dx_err": 0.02, "dy_err": 0.03})


class PsfaoConvergenceTests(unittest.TestCase):
    def test_initial_vector_stall_is_retained_but_excluded(self):
        optimizer = {
            "success": True,
            "status": 1,
            "message": "converged without moving",
            "nfev": 1,
            "cost": 12.0,
            "stalled_at_initial": True,
        }
        fit_result = (DEFAULT_X0, 1.0, 0.0, (0.0, 0.0), 8.0, np.ones((8, 8)), optimizer,
                      dict(ERRORES_FALSOS))
        bins = [(6500.0, 6600.0, 6550.0, np.ones(4, dtype=bool))]
        cube = np.ones((4, 8, 8), dtype=float)

        with mock.patch("musepipe.stages.stage_e01_psfao.fit_bin", return_value=fit_result):
            rows, recons = fit_psfao_bins(
                cube, cube, np.arange(4), bins, object(), (4.0, 6.0), 2.0, 3.0
            )

        self.assertEqual(rows[0]["status"], "fit_stalled:initial_vector")
        self.assertTrue(rows[0]["optimizer_stalled_at_initial"])
        self.assertEqual(rows[0]["optimizer_nfev"], 1)
        self.assertEqual(recons, {})

    def test_optimizer_failure_is_excluded(self):
        optimizer = {
            "success": False,
            "status": 0,
            "message": "maximum evaluations",
            "nfev": 400,
            "cost": 20.0,
            "stalled_at_initial": False,
        }
        fit_result = (DEFAULT_X0, 1.0, 0.0, (0.1, 0.0), 8.0, np.ones((8, 8)), optimizer,
                      dict(ERRORES_FALSOS))
        bins = [(6500.0, 6600.0, 6550.0, np.ones(4, dtype=bool))]
        cube = np.ones((4, 8, 8), dtype=float)

        with mock.patch("musepipe.stages.stage_e01_psfao.fit_bin", return_value=fit_result):
            rows, recons = fit_psfao_bins(
                cube, cube, np.arange(4), bins, object(), (4.0, 6.0), 2.0, 3.0
            )

        self.assertEqual(rows[0]["status"], "fit_failed:optimizer_status_0")
        self.assertFalse(rows[0]["optimizer_success"])
        self.assertEqual(recons, {})


class WarmStartRescueTests(unittest.TestCase):
    """Los bins se ajustan tambien desde sus vecinos, y gana el de menor coste.

    Cada bin que se queda fuera es un hueco en `param_table` que
    `_evaluate_psfao` cruza con una recta, y de ahi salen las mesetas del modelo
    cromatico. Rescatar solo los estancados no basta: en ROXs 12 b los bins de
    8100-8300 A CONVERGIAN desde el arranque comun a un segundo minimo 8 veces
    peor en coste, que el guardia de `build_psfao_model_document` tiraba por
    tener la apcorr fuera de tendencia, y que ademas envenenaba el arranque
    caliente de los tres bins siguientes. Asi que el criterio ya no es
    «converge / no converge» sino el COSTE del ajuste, que es comparable dentro
    de un mismo bin; un bin solo se mueve si otro arranque lo ajusta
    estrictamente mejor.
    """

    OK = {"success": True, "status": 1, "message": "ok", "nfev": 12,
          "cost": 1.0, "stalled_at_initial": False}
    ESTANCADO = {"success": True, "status": 1, "message": "sin moverse", "nfev": 2,
                 "cost": 9.0, "stalled_at_initial": True}

    @classmethod
    def _ok(cls, cost):
        """Un `optimizer` convergido con el coste que se le diga."""
        return dict(cls.OK, cost=cost)

    @staticmethod
    def _bins(n):
        return [(6000.0 + 100 * i, 6100.0 + 100 * i, 6050.0 + 100 * i,
                 np.ones(4, dtype=bool)) for i in range(n)]

    def _corre(self, n_bins, guion, **kwargs):
        """`guion`: un resultado de `fit_bin` por LLAMADA (un bin puede llevar dos).

        Cada entrada es `(params, optimizer)` o `(params, optimizer, errores)`;
        sin el tercero se usan unos errores cualesquiera, porque casi ningun
        test de este bloque los mira.
        """
        cube = np.ones((4, 8, 8), dtype=float)
        llamadas = []
        pendientes = list(guion)

        def falso(img, var, samp, system, comp, mask_r, fit_r, x0, field_yx=None):
            llamadas.append(list(x0))
            entrada = pendientes.pop(0)
            params, optimizer = entrada[0], entrada[1]
            errores = entrada[2] if len(entrada) > 2 else dict(ERRORES_FALSOS)
            return (params, 1.0, 0.0, (0.0, 0.0), 8.0, np.ones((8, 8)), optimizer, errores)

        with mock.patch("musepipe.stages.stage_e01_psfao.fit_bin", side_effect=falso):
            rows, recons = fit_psfao_bins(
                cube, cube, np.arange(4), self._bins(n_bins),
                object(), (4.0, 6.0), 2.0, 3.0, **kwargs
            )
        self.assertEqual(pendientes, [], "el guion preveia llamadas que no se hicieron")
        return rows, recons, llamadas

    def test_a_stalled_bin_is_retried_from_the_last_converged_one(self):
        bueno = [0.11, 2e-4, 2.0, 0.06, 0.9, 0.1, 1.7]
        rescatado = [0.12, 3e-4, 2.1, 0.07, 0.95, 0.2, 1.8]
        peor = [0.99, 9e-4, 9.9, 0.99, 0.99, 0.9, 9.9]
        rows, recons, llamadas = self._corre(2, [
            (bueno, self._ok(1.0)),              # bin 0: converge a la primera
            (DEFAULT_X0, dict(self.ESTANCADO)),  # bin 1: se estanca...
            (rescatado, self._ok(1.5)),          # ...y el rescate lo saca
            (peor, self._ok(5.0)),               # vuelta atras sobre el bin 0: peor
        ])
        self.assertEqual([r["status"] for r in rows], ["ok", "ok"])
        self.assertEqual([r["start_vector"] for r in rows], ["initial_vector", "warm_start"])
        # el rescate parte del bin anterior convergido, no de DEFAULT_X0
        self.assertEqual(llamadas[0], list(DEFAULT_X0))
        self.assertEqual(llamadas[1], list(DEFAULT_X0))
        self.assertEqual(llamadas[2], bueno)
        # y la pasada hacia atras prueba el bin 0 desde el que acaba de salir
        self.assertEqual(llamadas[3], rescatado)
        self.assertEqual(rows[0]["r0"], bueno[0])   # que no gano: coste 5.0 > 1.0
        self.assertEqual(sorted(recons), [6050.0, 6150.0])

    def test_a_converged_bin_is_replaced_when_a_neighbour_fits_it_better(self):
        """El caso de 8100-8300 A: converger no es ajustar bien.

        El bin 1 converge desde el arranque comun a un minimo de coste 8.0. El
        del vecino le da 1.2 sobre EL MISMO dato, asi que ese es el ajuste que
        describe el bin, y el que se queda.
        """
        bueno = [0.11, 2e-4, 2.0, 0.06, 0.9, 0.1, 1.7]
        malo = [0.30, 5e-2, 1.4, 0.05, 0.01, 2.0, 1.74]
        mejor = [0.12, 3e-4, 2.1, 0.07, 0.95, 0.2, 1.8]
        rows, _recons, llamadas = self._corre(2, [
            (bueno, self._ok(1.0)),
            (malo, self._ok(8.0)),      # bin 1: converge, pero a un minimo peor
            (mejor, self._ok(1.2)),     # desde el vecino: estrictamente mejor
            (malo, self._ok(4.0)),      # vuelta atras sobre el bin 0: peor
        ])
        self.assertEqual(rows[1]["start_vector"], "warm_start")
        self.assertEqual(rows[1]["r0"], mejor[0])
        self.assertEqual(rows[1]["optimizer_cost"], 1.2)
        self.assertEqual(llamadas[2], bueno)

    def test_a_bin_nobody_improves_keeps_its_own_fit(self):
        """Se prueban los vecinos, pero un bin solo se mueve si alguien lo mejora.

        Aqui ninguno gana —uno empata y el otro es peor— y las dos filas salen
        con los parametros y la etiqueta del arranque comun, bit a bit.
        """
        bueno = [0.11, 2e-4, 2.0, 0.06, 0.9, 0.1, 1.7]
        otro = [0.13, 4e-4, 2.2, 0.08, 0.99, 0.3, 1.9]
        rows, _recons, llamadas = self._corre(2, [
            (bueno, self._ok(1.0)),
            (bueno, self._ok(1.0)),
            (otro, self._ok(2.0)),   # desde el vecino: peor
            (otro, self._ok(1.0)),   # hacia atras: empata, y el empate no mueve
        ])
        self.assertEqual(len(llamadas), 4)
        self.assertTrue(all(r["start_vector"] == "initial_vector" for r in rows))
        self.assertTrue(all(r["r0"] == bueno[0] for r in rows))
        self.assertTrue(all(r["optimizer_cost"] == 1.0 for r in rows))

    def test_a_stalled_bin_is_rescued_from_the_red_side(self):
        """El caso de 8400-8600 A: el unico vecino bueno esta al ROJO.

        En la pasada hacia el rojo no hay de donde partir (el bin 0 es el
        primero), asi que solo la vuelta hacia el azul lo saca.
        """
        bueno = [0.11, 2e-4, 2.0, 0.06, 0.9, 0.1, 1.7]
        rescatado = [0.12, 3e-4, 2.1, 0.07, 0.95, 0.2, 1.8]
        rows, recons, llamadas = self._corre(2, [
            (DEFAULT_X0, dict(self.ESTANCADO)),  # bin 0: se estanca, sin vecino aun
            (bueno, self._ok(1.0)),              # bin 1: converge (nadie caliente antes)
            (rescatado, self._ok(1.1)),          # vuelta atras: el bin 0 sale
        ])
        self.assertEqual([r["status"] for r in rows], ["ok", "ok"])
        self.assertEqual(rows[0]["start_vector"], "warm_start_back")
        self.assertEqual(rows[0]["r0"], rescatado[0])
        self.assertEqual(llamadas, [list(DEFAULT_X0), list(DEFAULT_X0), bueno])
        self.assertEqual(sorted(recons), [6050.0, 6150.0])

    def test_a_stall_with_no_converged_predecessor_stays_stalled(self):
        """Sin bin del que partir no hay rescate: el primero no se inventa nada."""
        rows, recons, llamadas = self._corre(1, [
            (DEFAULT_X0, dict(self.ESTANCADO)),
        ])
        self.assertEqual(rows[0]["status"], "fit_stalled:initial_vector")
        self.assertEqual(len(llamadas), 1)
        self.assertEqual(recons, {})

    def test_warm_start_off_reproduces_the_old_behaviour(self):
        rows, recons, llamadas = self._corre(2, [
            ([0.11, 2e-4, 2.0, 0.06, 0.9, 0.1, 1.7], dict(self.OK)),
            (DEFAULT_X0, dict(self.ESTANCADO)),
        ], warm_start=False)
        self.assertEqual([r["status"] for r in rows], ["ok", "fit_stalled:initial_vector"])
        self.assertEqual(len(llamadas), 2)

    def test_a_failed_rescue_keeps_the_original_verdict(self):
        """Si el segundo intento tampoco sale, se conserva el diagnostico real.

        Y la fila lo dice: `start_vector` nombra al ajuste QUE SE QUEDA, que es
        el del arranque comun. Poner ahi `warm_start` describiria un intento que
        se descarto.
        """
        bueno = [0.11, 2e-4, 2.0, 0.06, 0.9, 0.1, 1.7]
        rows, recons, llamadas = self._corre(2, [
            (bueno, self._ok(1.0)),
            (DEFAULT_X0, dict(self.ESTANCADO)),
            (bueno, dict(self.ESTANCADO)),   # el rescate tambien se estanca
        ])
        self.assertEqual(rows[1]["status"], "fit_stalled:initial_vector")
        self.assertEqual(rows[1]["start_vector"], "initial_vector")
        # la vuelta hacia atras no parte de un bin que no convergio
        self.assertEqual(len(llamadas), 3)
        self.assertEqual(sorted(recons), [6050.0])

    def test_the_row_carries_the_errors_of_the_attempt_that_stays(self):
        """Si el rescate gana, los errores de la fila son los SUYOS.

        Es la trampa evidente al anadir un valor de retorno mas: dejar el
        `errors` del primer intento —el estancado— junto a los parametros del
        segundo. La fila describiria un ajuste que no existe.
        """
        bueno = [0.11, 2e-4, 2.0, 0.06, 0.9, 0.1, 1.7]
        rescatado = [0.12, 3e-4, 2.1, 0.07, 0.95, 0.2, 1.8]
        err_estancado = {f"{n}_err": 9.0 for n in PSFAO_PARAM_NAMES}
        err_estancado.update({"dx_err": 9.0, "dy_err": 9.0})
        err_rescate = {f"{n}_err": 0.5 for n in PSFAO_PARAM_NAMES}
        err_rescate.update({"dx_err": 0.5, "dy_err": 0.5})
        rows, _recons, _llamadas = self._corre(2, [
            (bueno, self._ok(1.0)),
            (DEFAULT_X0, dict(self.ESTANCADO), err_estancado),
            (rescatado, self._ok(1.5), err_rescate),
            (bueno, self._ok(5.0), err_estancado),   # vuelta atras: peor, se descarta
        ])
        self.assertEqual(rows[1]["start_vector"], "warm_start")
        self.assertEqual(rows[1]["r0"], rescatado[0])
        self.assertEqual(rows[1]["r0_err"], 0.5)
        self.assertEqual(rows[1]["dx_err"], 0.5)


class PsfaoParameterErrorTests(unittest.TestCase):
    """`_psfao_param_errors`: lo que `psffit` ya calculaba y se tiraba.

    `maoppy` deja las incertidumbres formales en `res.x_std` / `res.dxdy_std`
    (`1/sqrt(diag(JtJ))`). Lo unico con criterio aqui es que un parametro pegado
    a su limite fisico tiene gradiente nulo, la diagonal sale 0 y la division da
    infinito: eso NO es una incertidumbre enorme, es «no medido», y tiene que
    salir NaN para que las medianas robustas y los `errorbar` lo ignoren.
    """

    class _Res:
        def __init__(self, x_std, dxdy_std):
            self.x_std = x_std
            self.dxdy_std = dxdy_std

    def test_finite_positive_values_travel_one_per_parameter(self):
        x_std = [0.1 * (i + 1) for i in range(len(PSFAO_PARAM_NAMES))]
        errors = _psfao_param_errors(self._Res(x_std, [0.7, 0.8]))
        for i, name in enumerate(PSFAO_PARAM_NAMES):
            self.assertAlmostEqual(errors[f"{name}_err"], x_std[i])
        # `res.dxdy` es (dx, dy) y `res.dxdy_std` va en el mismo orden.
        self.assertAlmostEqual(errors["dx_err"], 0.7)
        self.assertAlmostEqual(errors["dy_err"], 0.8)

    def test_a_parameter_pinned_at_its_bound_is_nan_not_infinity(self):
        x_std = [np.inf, 0.0, -1.0, np.nan, 0.05, 0.05, 0.05]
        errors = _psfao_param_errors(self._Res(x_std, [np.inf, 0.0]))
        # r0 (inf), C (cero), A (negativo) y alpha (NaN) no estan medidos.
        for name in PSFAO_PARAM_NAMES[:4]:
            self.assertTrue(np.isnan(errors[f"{name}_err"]), name)
        # ratio, theta y beta si.
        for name in PSFAO_PARAM_NAMES[4:]:
            self.assertAlmostEqual(errors[f"{name}_err"], 0.05, msg=name)
        self.assertTrue(np.isnan(errors["dx_err"]))
        self.assertTrue(np.isnan(errors["dy_err"]))

    def test_a_result_without_the_attributes_degrades_to_nan(self):
        """Una version de `maoppy` que no las publique no puede tumbar C1."""
        errors = _psfao_param_errors(object())
        self.assertEqual(len(errors), len(PSFAO_PARAM_NAMES) + 2)
        self.assertTrue(all(np.isnan(v) for v in errors.values()))


class RingQCTests(unittest.TestCase):
    def test_post_hybrid_arrays_drive_all_qc_fields(self):
        summary = _ring_qc_summary(
            [1.0, 2.0, 8.0, 9.0],
            [2.0, 3.0, 10.0, 20.0],
            {"psf_hybrid_threshold_pct": 5.0, "psf_hybrid_bin_fraction": 0.6},
        )
        self.assertEqual(summary["median"], 5.0)
        self.assertAlmostEqual(summary["p90"], 17.0)
        self.assertEqual(summary["bins_above"], 2)
        self.assertFalse(summary["issue"])  # 50% is not greater than the configured 60%.

        summary = _ring_qc_summary(
            [1.0, 8.0, 9.0], [2.0, 10.0, 20.0],
            {"psf_hybrid_threshold_pct": 5.0, "psf_hybrid_bin_fraction": 0.2},
        )
        self.assertTrue(summary["issue"])


class B3CentroidTests(unittest.TestCase):
    def test_absolute_psfao_centroids_are_compared_to_interpolated_track(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "track.csv"
            path.write_text(
                "wave_min_A,wave_max_A,primary_y,primary_x\n"
                "6400,6500,10.0,10.0\n"
                "6600,6700,10.0,10.4\n",
                encoding="utf-8",
            )
            track, status = _b3_chromatic_track(path)
            rows = [
                {"lambda_A": 6450.0, "dy": 0.0, "dx": 0.0, "status": "ok"},
                {"lambda_A": 6650.0, "dy": 0.0, "dx": 0.4, "status": "ok"},
            ]

            diff = _centroid_vs_b3(rows, "psfao", track, image_shape=(20, 20))

        self.assertEqual(status, "used")
        self.assertAlmostEqual(diff, 0.0)

    def test_missing_track_is_explicit(self):
        track, status = _b3_chromatic_track(Path("/definitely/missing/track.csv"))
        self.assertIsNone(track)
        self.assertEqual(status, "unavailable:missing_track")
        self.assertIsNone(_centroid_vs_b3([], "moffat", track))


class SelectedOutputTests(unittest.TestCase):
    def _psfao_row(self):
        return {
            "lambda_A": 6500.0,
            "samp": 0.7,
            "amp": 1.0,
            "bck": 0.0,
            "dy": 0.1,
            "dx": 0.2,
            "ring_residual_pct": 99.0,
            "ring_residual_pct_canonical": 8.0,
            "ring_residual_p90_pct_canonical": 12.0,
            "ring_residual_pct_after_hybrid_canonical": 4.0,
            "ring_residual_p90_pct_after_hybrid_canonical": 6.0,
            "r0": 0.1,
            "C": 0.01,
            "A": 1.0,
            "alpha": 0.1,
            "ratio": 1.0,
            "theta": 0.0,
            "beta": 1.5,
            "optimizer_success": True,
            "optimizer_status": 1,
            "optimizer_message": "ok",
            "optimizer_nfev": 3,
            "optimizer_cost": 1.0,
            "optimizer_stalled_at_initial": False,
            "status": "ok",
        }

    def test_psfao_csv_contains_canonical_before_and_after_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "params.csv"
            _write_psfao_csv(path, [self._psfao_row()])
            row = next(csv.DictReader(path.open(encoding="utf-8")))

        self.assertEqual(float(row["ring_residual_pct_canonical"]), 8.0)
        self.assertEqual(float(row["ring_residual_pct_after_hybrid_canonical"]), 4.0)
        self.assertEqual(row["optimizer_success"], "True")

    def test_the_csv_has_one_error_column_per_parameter(self):
        fila = self._psfao_row()
        fila.update({f"{name}_err": 0.25 for name in PSFAO_PARAM_NAMES})
        fila.update({"dx_err": 0.5, "dy_err": 0.75})
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "params.csv"
            _write_psfao_csv(path, [fila])
            row = next(csv.DictReader(path.open(encoding="utf-8")))

        for name in PSFAO_PARAM_NAMES:
            self.assertEqual(float(row[f"{name}_err"]), 0.25, name)
        self.assertEqual(float(row["dx_err"]), 0.5)
        self.assertEqual(float(row["dy_err"]), 0.75)

    def test_a_row_from_before_the_error_columns_still_writes(self):
        """Las columnas nuevas son aditivas: una fila vieja sale con el hueco vacio.

        Importa porque el CSV que hay en `runs/` se escribio antes de que
        existieran, y los lectores van por nombre de columna.
        """
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "params.csv"
            _write_psfao_csv(path, [self._psfao_row()])
            row = next(csv.DictReader(path.open(encoding="utf-8")))

        self.assertEqual(row["r0_err"], "")
        self.assertEqual(float(row["r0"]), 0.1)

    def test_summary_plot_accepts_selected_psfao_rows_without_moffat_fields(self):
        product = StageE01Product(
            fit_rows=[{"wave_center_A": 6500.0}],
            psf_model={},
            qc={"hybrid": {"applied": True}},
            hybrid_profiles=None,
            hybrid_radii=None,
            psf_form="psfao",
            psfao_rows=[self._psfao_row()],
        )
        with tempfile.TemporaryDirectory() as tmp:
            summary = Path(tmp) / "summary.png"
            _write_summary_plot(product, {"summary_plot": summary})
            self.assertTrue(summary.exists())


class WaveBinInTheDocumentTests(unittest.TestCase):
    """La rejilla de evaluacion viaja en `psf_model.json`, escrita por C1.

    Antes solo vivia ahi si alguien la ponia a mano, asi que re-correr C1
    revertia en silencio la eleccion medida (100 A) al 50 A historico de
    `psf.py`. El documento tiene que declararla siempre.
    """

    FILAS = [{"lambda_A": 6000.0 + 100 * i, "status": "ok", "ring_residual_pct": 5.0,
              **{n: v for n, v in zip(PSFAO_PARAM_NAMES,
                                      [0.11, 2e-4, 2.0, 0.06, 0.9, 0.1, 1.7])}}
             for i in range(4)]

    def _doc(self, **kw):
        with mock.patch("musepipe.stages.stage_e01_psfao._box3_apcorr", return_value=7.0):
            doc, _meta = build_psfao_model_document(self.FILAS, object(), 25.0, 78.0, **kw)
        return doc

    def test_the_declared_grid_travels_with_its_reason(self):
        doc = self._doc(wave_bin_A=100.0)
        self.assertEqual(doc["psfao_wave_bin_A"], 100.0)
        self.assertIn("degenerados", doc["psfao_wave_bin_A_note"])

    def test_without_a_grid_the_key_is_absent_rather_than_invented(self):
        """Un documento viejo no adquiere una rejilla que nadie eligio."""
        self.assertNotIn("psfao_wave_bin_A", self._doc())

    def test_psf_py_reads_the_grid_from_the_document(self):
        """El consumidor y el productor hablan de la misma clave."""
        import inspect

        from musepipe import psf

        fuente = inspect.getsource(psf._psfao_wave_bin_A)
        self.assertIn('model_doc.get("psfao_wave_bin_A"', fuente)
        self.assertIn("_psfao_wave_bin_A(model_doc)", inspect.getsource(psf._evaluate_psfao))


class WaveBinResolutionTests(unittest.TestCase):
    """De donde sale la rejilla cuando el documento no la declara.

    El default historico de `psf.py` era 50 A: la mitad del ancho de los bins
    que C1 ajusta, asi que uno de cada dos canales recibia los parametros del
    PSD interpolados entre dos bins. Al ser degenerados, la recta entre dos
    ajustes se sale del valle. Ya no hay numero inventado: o lo declara C1, o
    sale de la propia tabla, o no se redondea.
    """

    def _tabla(self, lambdas):
        return {"param_table": {"lambda_A": list(lambdas)}}

    def test_the_declared_grid_wins(self):
        from musepipe.psf import _psfao_wave_bin_A

        doc = self._tabla([6000.0, 6100.0])
        doc["psfao_wave_bin_A"] = 250.0
        self.assertEqual(_psfao_wave_bin_A(doc), 250.0)

    def test_without_the_key_the_grid_is_the_width_of_the_fitted_bins(self):
        from musepipe.psf import _psfao_wave_bin_A

        # Los huecos (bins que C1 rechazo) son multiplos del ancho: la mediana
        # de las separaciones los contaria, el minimo no.
        doc = self._tabla([6000.0, 6100.0, 6200.0, 6600.0, 6700.0])
        self.assertEqual(_psfao_wave_bin_A(doc), 100.0)

    def test_no_default_of_fifty(self):
        """El 50 A historico no puede volver por ninguna via."""
        from musepipe.psf import _psfao_wave_bin_A

        self.assertNotEqual(_psfao_wave_bin_A(self._tabla([4800.0, 4900.0])), 50.0)

    def test_a_document_without_a_table_is_evaluated_at_the_exact_wavelength(self):
        """Sin `param_table` no hay rejilla que respetar: el polinomio es continuo."""
        from musepipe.psf import _psfao_wave_bin_A

        self.assertEqual(_psfao_wave_bin_A({"smoothed_poly": {"r0": [0.12]}}), 0.0)

    def test_an_invalid_declared_grid_raises(self):
        from musepipe.psf import _psfao_wave_bin_A

        for malo in (-100.0, float("nan")):
            with self.assertRaises(ValueError):
                _psfao_wave_bin_A({"psfao_wave_bin_A": malo})


if __name__ == "__main__":
    unittest.main()
