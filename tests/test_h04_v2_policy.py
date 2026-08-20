import tempfile
import unittest
from pathlib import Path

import numpy as np

from musepipe.stages.stage_h04_injection import (
    _v2_nulls_clean,
    _v5_continuum,
    resolve_continuum_injection,
    stage_h04_paths,
)
from tests.test_compare_verdicts import make_product


#: Los seis metodos de la cadena, para poder probar la mediana por posicion.
METHODS = ("aperture", "optimal_ls", "optimal_psfsub", "psffit", "sgf", "lpm")

#: Contra `np.arange(33)`: un flujo de 100 es el maximo (FAP 1/34, extremo) y
#: uno de 10 no lo es (FAP 24/34).
EXTREME_FLUX = 100.0
QUIET_FLUX = 10.0


def null_row(index, *, position="control1", recovered_flux=0.0, recovered_snr=0.0,
             method="aperture"):
    return {
        "injection_id": f"null{index}",
        "variant": "nominal",
        "method": method,
        "position_label": position,
        "template_factor": 1.0,
        "continuum_mode": "none",
        "input_snr": 0.0,
        "recovered_flux": recovered_flux,
        "recovered_snr": recovered_snr,
    }


def empirical_reference():
    return {
        method: {"n_controls": 33, "by_factor": {"1": np.arange(33, dtype=np.float64)}}
        for method in METHODS
    }


def position_rows(position, n_extreme_methods, *, start=0):
    """Una fila por metodo en `position`, `n_extreme_methods` de ellas extremas."""

    return [
        null_row(start + i, position=position, method=method,
                 recovered_flux=EXTREME_FLUX if i < n_extreme_methods else QUIET_FLUX)
        for i, method in enumerate(METHODS)
    ]


class H04V2NullPolicyTests(unittest.TestCase):
    def test_real_position_never_has_veto_power(self):
        rows = [null_row(0, recovered_flux=1000.0, recovered_snr=100.0, position="real")]
        rows.extend(null_row(i + 1, recovered_flux=10.0) for i in range(10))

        result = _v2_nulls_clean(rows, empirical_reference())

        self.assertEqual(result["status"], "pass")
        self.assertEqual(result["n_rows"], 10)
        self.assertEqual(result["science_position_diagnostics"]["n_rows"], 1)

    def test_gate_uses_empirical_flux_rank_not_formal_snr(self):
        rows = [null_row(i, recovered_flux=10.0, recovered_snr=999.0) for i in range(10)]

        result = _v2_nulls_clean(rows, empirical_reference())

        self.assertEqual(result["status"], "pass")
        self.assertEqual(result["n_extreme"], 0)

    def test_many_extreme_rows_in_a_single_position_do_not_fail_the_gate(self):
        # El defecto que motiva E4 v3: 24 filas extremas eran 24 rechazos
        # independientes cuando son UNA zona de cielo medida 24 veces.
        rows = [null_row(i, position="control1", method=method, recovered_flux=EXTREME_FLUX)
                for i, method in enumerate(METHODS) for _ in range(4)]
        rows.extend(position_rows("control2", 0, start=100))
        rows.extend(position_rows("control3", 0, start=200))

        result = _v2_nulls_clean(rows, empirical_reference())

        self.assertEqual(result["status"], "pass")
        self.assertEqual(result["n_extreme"], 1, "una posicion extrema, no veinticuatro")
        self.assertEqual(result["n_positions"], 3)

    def test_two_extreme_positions_of_three_fail_the_gate(self):
        rows = position_rows("control1", len(METHODS))
        rows.extend(position_rows("control2", len(METHODS), start=100))
        rows.extend(position_rows("control3", 0, start=200))

        result = _v2_nulls_clean(rows, empirical_reference())

        self.assertEqual(result["status"], "fail")
        self.assertEqual(result["n_extreme"], 2)

    def test_the_median_needs_more_than_half_the_methods(self):
        # Un solo metodo no declara extrema una posicion; la mitad justa tampoco
        # (cae en el punto medio); cuatro de seis si.
        for n_extreme_methods, expected in ((1, False), (3, False), (4, True)):
            with self.subTest(methods=n_extreme_methods):
                rows = position_rows("control1", n_extreme_methods)
                result = _v2_nulls_clean(rows, empirical_reference())
                self.assertEqual(result["positions"][0]["extreme"], expected)

    def test_qc_separates_positions_from_rows(self):
        rows = position_rows("control1", len(METHODS))
        rows.extend(position_rows("control2", 0, start=100))

        result = _v2_nulls_clean(rows, empirical_reference())

        self.assertEqual(result["unit"], "control_position")
        self.assertEqual(result["n_extreme"], 1)
        self.assertEqual(result["n_extreme_rows"], len(METHODS))
        self.assertEqual(result["n_rows"], 2 * len(METHODS))
        self.assertEqual({entry["position_label"] for entry in result["positions"]},
                         {"control1", "control2"})

    def test_the_low_tail_is_counted_and_does_not_vote(self):
        # Deficit de flujo donde no se inyecto nada: se informa, no vota.
        rows = [null_row(i, position=f"control{i + 1}", method="optimal_ls",
                         recovered_flux=-1000.0) for i in range(3)]

        result = _v2_nulls_clean(rows, empirical_reference())

        self.assertEqual(result["status"], "pass")
        self.assertEqual(result["n_extreme"], 0)
        self.assertFalse(result["low_tail_diagnostics"]["gating"])
        self.assertEqual(result["low_tail_diagnostics"]["n_extreme_rows"], 3)
        self.assertEqual(result["low_tail_diagnostics"]["by_method"], {"optimal_ls": 3})


class H04V2ContinuumTests(unittest.TestCase):
    def test_continuum_is_measured_in_normrad_scale_from_positive_methods(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = stage_h04_paths("synthetic", project_root=tmp)
            paths["paths"].ensure_base_dirs()
            stage_dir = paths["paths"].stage_dir
            wave = np.arange(6490.0, 6640.0, 5.0)
            values = {
                "aperture": 4.0,
                "optimal_ls": -2.0,
                "optimal_psfsub": -1.0,
                "psffit": 8.0,
            }
            names = {
                "aperture": "spec_aperture_object.fits",
                "optimal_ls": "spec_optimal_object.fits",
                "optimal_psfsub": "spec_optimal_psfsub_object.fits",
                "psffit": "spec_psffit_object.fits",
            }
            for method, value in values.items():
                product = make_product(method, np.full(wave.size, value * 2.0), wave=wave)
                product = type(product)(
                    **{**product.__dict__, "apcorr": np.full(wave.size, 2.0)}
                )
                product.write(stage_dir / names[method])

            result = resolve_continuum_injection({}, paths)

        self.assertEqual(result["source"], "median_continuum_preserving_methods")
        self.assertAlmostEqual(result["value"], 6.0)
        self.assertEqual(result["scale"], "normrad_flux_density")

    def test_v5_rejects_zero_or_identical_flat_grid(self):
        rows = []
        for mode in ("none", "flat"):
            rows.append(
                {
                    "method": "aperture",
                    "variant": "nominal",
                    "continuum_mode": mode,
                    "input_snr": 5.0,
                    "template_factor": 1.0,
                    "throughput": 1.0,
                }
            )
        result = _v5_continuum(rows, ["aperture"], {"value": 0.0})
        self.assertEqual(result["status"], "fail")
        self.assertFalse(result["continuum_positive"])
        self.assertFalse(result["flat_distinct_from_none"])


if __name__ == "__main__":
    unittest.main()
