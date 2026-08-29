"""La SNR estandarizada de E4: contra que se estandariza, y que no se hace trampa.

Lo que estas pruebas protegen no es la aritmetica del z-score, es la eleccion de
la POBLACION. Medido el 2026-08-28: estandarizar contra los 33 controles de
produccion —la referencia de la puerta V2— no arregla la escala, la empeora, y
la razon es que esa poblacion no es la de las inyecciones nulas. El estrato y el
leave-one-out son las dos formas concretas de esa eleccion, y las dos son
invisibles en el resultado si se rompen: la SNR sigue saliendo, solo que mal.
"""

import unittest

import numpy as np

from musepipe.stages.stage_h04_injection import (
    SNR_STANDARDIZATION_MODES,
    TABLE_FIELDS,
    apply_snr_standardization,
    injection_null_reference,
)


def fila(**kw):
    row = {
        "injection_id": "inj0000",
        "variant": "nominal",
        "method": "aperture",
        "position_label": "control1",
        "template_factor": 1.0,
        "continuum_mode": "none",
        "input_snr": 0.0,
        "recovered_flux": 0.0,
        "recovered_snr": 0.0,
        "complete": False,
    }
    row.update(kw)
    return row


def nulos(method="aperture", fluxes=(0.0, 1.0, -1.0, 2.0, -2.0), **kw):
    return [
        fila(injection_id=f"null{i}", method=method, position_label=f"control{i + 1}",
             input_snr=0.0, recovered_flux=float(f), **kw)
        for i, f in enumerate(fluxes)
    ]


class TestReferencia(unittest.TestCase):
    def test_el_estrato_separa_metodo_ancho_y_continuo(self):
        rows = nulos() + nulos(method="psffit") + nulos(template_factor=2.0) + nulos(continuum_mode="flat")
        ref = injection_null_reference(rows)
        self.assertEqual(
            sorted(ref),
            [("aperture", "1", "flat"), ("aperture", "1", "none"),
             ("aperture", "2", "none"), ("psffit", "1", "none")],
        )

    def test_la_posicion_real_no_entra_en_la_referencia(self):
        # Puede llevar senal del companero: meterla es poner la deteccion en el cero.
        rows = nulos() + [fila(injection_id="real0", position_label="real", recovered_flux=999.0)]
        ref = injection_null_reference(rows)
        self.assertEqual(len(ref[("aperture", "1", "none")]), 5)

    def test_solo_entran_las_nulas_nominales(self):
        rows = nulos() + [
            fila(injection_id="s1", input_snr=3.0, recovered_flux=50.0),
            fila(injection_id="p1", variant="psf_plus10", recovered_flux=60.0),
        ]
        self.assertEqual(len(injection_null_reference(rows)[("aperture", "1", "none")]), 5)


class TestLeaveOneOut(unittest.TestCase):
    def test_una_nula_no_entra_en_su_propia_referencia(self):
        rows = nulos(fluxes=(0.0, 1.0, -1.0, 2.0, -2.0))
        out = apply_snr_standardization(rows, mode="injection_nulls", threshold_snr=5.0)
        # El z de la fila extrema calculado a mano, sin ella misma en la referencia.
        resto = [0.0, 1.0, -1.0, -2.0]
        centro = float(np.median(resto))
        mad = float(np.median(np.abs(np.asarray(resto) - centro)))
        esperado = (2.0 - centro) / (1.4826 * mad)
        extrema = [r for r in out if r["injection_id"] == "null3"][0]
        self.assertAlmostEqual(extrema["recovered_snr_std"], esperado, places=9)

    def test_sin_leave_one_out_el_z_saldria_mas_pequeno(self):
        rows = nulos(fluxes=(0.0, 1.0, -1.0, 2.0, -2.0))
        out = apply_snr_standardization(rows, mode="injection_nulls", threshold_snr=5.0)
        extrema = [r for r in out if r["injection_id"] == "null3"][0]
        todos = [0.0, 1.0, -1.0, 2.0, -2.0]
        centro = float(np.median(todos))
        mad = float(np.median(np.abs(np.asarray(todos) - centro)))
        self.assertGreater(extrema["recovered_snr_std"], (2.0 - centro) / (1.4826 * mad))

    def test_una_fila_con_senal_usa_la_referencia_entera(self):
        rows = nulos() + [fila(injection_id="s1", input_snr=3.0, recovered_flux=2.0)]
        out = apply_snr_standardization(rows, mode="injection_nulls", threshold_snr=5.0)
        senal = [r for r in out if r["injection_id"] == "s1"][0]
        nula = [r for r in out if r["injection_id"] == "null3"][0]
        # Mismo flujo, distinto z: la nula se quita a si misma y la senal no.
        self.assertEqual(senal["recovered_flux"], nula["recovered_flux"])
        self.assertLess(senal["recovered_snr_std"], nula["recovered_snr_std"])


class TestDecision(unittest.TestCase):
    def test_el_pedestal_deja_de_contar_como_deteccion(self):
        # El caso de `psffit` en ROXs 12 b: la nula entera desplazada. Con la SNR
        # formal la fila esta a 5 sigma y se cuenta; estandarizada esta en el
        # centro de su propia nula y no.
        rows = nulos(fluxes=(500.0, 510.0, 490.0, 505.0, 495.0)) + [
            fila(injection_id="s1", input_snr=3.0, recovered_flux=500.0,
                 recovered_snr=9.0, complete=True)
        ]
        out = apply_snr_standardization(rows, mode="injection_nulls", threshold_snr=5.0)
        senal = [r for r in out if r["injection_id"] == "s1"][0]
        self.assertFalse(senal["complete"])
        self.assertLess(abs(senal["recovered_snr_std"]), 1.0)

    def test_una_senal_de_verdad_sigue_contando(self):
        rows = nulos(fluxes=(500.0, 510.0, 490.0, 505.0, 495.0)) + [
            fila(injection_id="s1", input_snr=3.0, recovered_flux=1000.0)
        ]
        out = apply_snr_standardization(rows, mode="injection_nulls", threshold_snr=5.0)
        self.assertTrue([r for r in out if r["injection_id"] == "s1"][0]["complete"])

    def test_modo_none_no_toca_la_decision_pero_si_publica_la_columna(self):
        rows = nulos() + [fila(injection_id="s1", input_snr=3.0, recovered_flux=100.0,
                               recovered_snr=1.0, complete=True)]
        out = apply_snr_standardization(rows, mode="none", threshold_snr=5.0)
        senal = [r for r in out if r["injection_id"] == "s1"][0]
        self.assertTrue(senal["complete"])  # la decidio la SNR formal, no se revisa
        self.assertTrue(np.isfinite(senal["recovered_snr_std"]))

    def test_no_muta_las_filas_de_entrada(self):
        rows = nulos()
        apply_snr_standardization(rows, mode="injection_nulls", threshold_snr=5.0)
        self.assertNotIn("recovered_snr_std", rows[0])


class TestContrato(unittest.TestCase):
    def test_un_estrato_sin_nulas_para_la_etapa(self):
        # Sin referencia no hay escala. Heredar la SNR formal seria volver a la
        # escala que se acaba de declarar mala, y en silencio.
        rows = nulos() + [fila(injection_id="s1", method="psffit", input_snr=3.0,
                               recovered_flux=10.0)]
        with self.assertRaises(ValueError) as ctx:
            apply_snr_standardization(rows, mode="injection_nulls", threshold_snr=5.0)
        self.assertIn("psffit", str(ctx.exception))

    def test_modo_desconocido_no_pasa_por_defecto(self):
        with self.assertRaises(ValueError):
            apply_snr_standardization(nulos(), mode="empirical_null", threshold_snr=5.0)
        self.assertEqual(SNR_STANDARDIZATION_MODES, ("none", "injection_nulls"))

    def test_la_columna_viaja_en_la_tabla(self):
        # Si no esta en TABLE_FIELDS, `csv.DictWriter` revienta al escribirla.
        self.assertIn("recovered_snr_std", TABLE_FIELDS)


if __name__ == "__main__":
    unittest.main()


class TestEtapaCompleta(unittest.TestCase):
    """La etapa entera con la perilla puesta: QC, version y columna.

    Lo que se prueba aqui y no en las unidades de arriba es el ENSAMBLADO: que el
    QC declara `E4_v4` solo cuando la escala cambia, que dice cual columna
    decidio, y que el diagnostico de poblaciones se construye sin la referencia
    de produccion real (aqui va declarada en la config).
    """

    def _corre(self, **overrides):
        import tempfile
        from pathlib import Path

        from tests.test_optimal_analytic import constant_model_doc
        from musepipe.stages.stage_h01_detect import HALPHA_REST_A, matched_filter_point
        from musepipe.stages.stage_h04_injection import (
            compute_stage_h04_products, stage_h04_paths,
        )

        wave = np.arange(6525.0, 6601.0, 1.0, dtype=np.float64)
        err = np.full(wave.size, 0.2, dtype=np.float64)
        _f, matched_sigma, _z = matched_filter_point(
            wave, np.zeros(wave.size), err, HALPHA_REST_A, 2.5, np.ones(wave.size, bool)
        )
        cube = np.zeros((wave.size, 64, 64), dtype=np.float64)
        yy, xx = np.mgrid[0:64, 0:64]
        cube += (0.001 * (yy + xx)).astype(np.float64)[None, :, :]

        def extractor(cube, wave_A, case, method, config):
            if cube.ndim == 4:
                cube = np.nanmean(cube, axis=0)
            return {"wave_A": wave_A, "flux": np.nansum(cube, axis=(1, 2)),
                    "flux_err": np.full(wave_A.size, 0.2, dtype=np.float64)}

        methods = ["aperture", "psffit"]
        null = np.linspace(-1.0, 1.0, 8).tolist()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = stage_h04_paths("synthetic_h04_std", project_root=root)
            paths["paths"].ensure_base_dirs()
            cfg = {
                "run_id": "synthetic_h04_std", "project_root": str(root),
                "h04_positions_yx": [
                    {"label": "real", "y": 30.0, "x": 30.0},
                    {"label": "control1", "y": 30.0, "x": 38.0},
                    {"label": "control2", "y": 38.0, "x": 30.0},
                    {"label": "control3", "y": 30.0, "x": 22.0},
                ],
                "h04_methods": methods, "h04_lsf_fwhm_A": 2.5,
                "h04_injection_flux_sigma": matched_sigma,
                "h04_continuum_flux_density": 1.0,
                "h04_empirical_null_reference": {
                    m: {"n_controls": len(null), "by_factor": {"1": null, "2": null}}
                    for m in methods
                },
                "h04_expected_seconds_per_case_method": 0.01,
                "h04_require_historic_regression": False,
                "h04_historic_expected_snr": 8.97,
                "h04_historic_recovered_snr": 8.80,
                "h04_historic_tolerance_snr": 0.25,
            }
            cfg.update(overrides)
            return compute_stage_h04_products(
                cfg, paths, extractors={m: extractor for m in methods},
                base_cube=cube, wavelengths_A=wave, psf_model=constant_model_doc(fwhm=4.0),
            )

    def test_apagada_declara_v3_y_no_decide(self):
        product = self._corre()
        self.assertEqual(product.qc["spec_version"], "E4_v3")
        self.assertEqual(product.qc["snr_standardization"]["mode"], "none")
        self.assertEqual(product.qc["snr_standardization"]["column"], "recovered_snr")
        self.assertIn("recovered_snr_std", product.rows[0])

    def test_encendida_declara_v4_y_dice_que_columna_decidio(self):
        product = self._corre(h04_snr_standardization="injection_nulls")
        self.assertEqual(product.qc["spec_version"], "E4_v4")
        self.assertEqual(product.qc["snr_standardization"]["column"], "recovered_snr_std")
        self.assertTrue(product.qc["snr_standardization"]["leave_one_out"])
        self.assertTrue(product.qc["snr_standardization"]["scale_by_stratum"])

    def test_el_qc_publica_la_nula_cruda_y_la_estandarizada(self):
        # Es la verificacion de la spec §5 dentro del propio producto: si la
        # estandarizada no sale ~(0,1), se ve sin re-correr nada.
        diag = self._corre().qc
        nula = diag["snr_standardization"]["null_distribution"]
        for method in ("aperture", "psffit"):
            self.assertIn("raw", nula[method])
            self.assertIn("standardized", nula[method])

    def test_el_nivel_de_completitud_ya_no_lo_fija_el_umbral(self):
        product = self._corre(h04_detection_threshold_snr=3.0)
        self.assertEqual(product.qc["completeness_threshold_snr"], 3.0)
        self.assertEqual(product.qc["completeness_input_snr"], 5.0)
