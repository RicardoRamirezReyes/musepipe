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


class TestFinalize(unittest.TestCase):
    """La pasada derivada: cambiar el umbral no puede exigir re-inyectar.

    `complete` es funcion pura de una columna que la tabla ya trae y del umbral.
    Lo que se prueba aqui es que la pasada recomputa exactamente eso y **nada
    mas**: las medidas se quedan byte a byte, y un producto sin la columna de la
    escala declarada para en vez de decidir con la otra.
    """

    def _run(self, tmp, *, threshold=5.0, mode="injection_nulls", columna_std=True):
        import csv as _csv
        import json as _json
        from pathlib import Path as _Path

        from musepipe.stages.stage_h04_injection import TABLE_FIELDS

        root = _Path(tmp)
        (root / "runs" / "r" / "config").mkdir(parents=True)
        (root / "runs" / "r" / "stages").mkdir(parents=True)
        (root / "runs" / "r" / "tables").mkdir(parents=True)
        (root / "runs" / "r" / "config" / "config.json").write_text(_json.dumps({
            "meta": {"run_id": "r"},
            "config": {"run_id": "r", "h04_detection_threshold_snr": threshold,
                       "h04_snr_standardization": mode, "h04_methods": ["aperture"]},
        }))
        campos = [f for f in TABLE_FIELDS if columna_std or f != "recovered_snr_std"]
        filas = []
        for i, (std, raw, snr) in enumerate([(6.0, 1.0, 5.0), (4.0, 9.0, 5.0), (1.0, 1.0, 0.0)]):
            row = {f: "" for f in campos}
            row.update({"injection_id": f"i{i}", "variant": "nominal", "method": "aperture",
                        "position_label": f"control{i + 1}", "template_factor": "1.0",
                        "continuum_mode": "none", "input_snr": str(snr),
                        "recovered_snr": str(raw), "complete": "False", "throughput": "0.9"})
            if columna_std:
                row["recovered_snr_std"] = str(std)
            filas.append(row)
        with open(root / "runs" / "r" / "tables" / "injection_throughput_by_method.csv",
                  "w", newline="") as fh:
            w = _csv.DictWriter(fh, fieldnames=campos)
            w.writeheader()
            w.writerows(filas)
        (root / "runs" / "r" / "stages" / "stage_h04_qc.json").write_text(_json.dumps({
            "stage": "h04_injection_recovery", "completeness_at_5sigma": {},
            "snr_standardization": {"threshold_snr": 99.0, "column": "x"},
        }))
        return root

    def test_decide_sobre_la_columna_de_la_escala_declarada(self):
        import json as _json
        import tempfile

        from musepipe.stages.stage_h04_injection import finalize_stage_h04

        with tempfile.TemporaryDirectory() as tmp:
            root = self._run(tmp, threshold=5.0)
            out = finalize_stage_h04("r", project_root=root)
            filas = list(__import__("csv").DictReader(
                open(root / "runs" / "r" / "tables" / "injection_throughput_by_method.csv")))
            # i0 tiene std 6 (>=5) y raw 1; i1 tiene std 4 y raw 9. Decide la std.
            self.assertEqual([f["complete"] for f in filas], ["True", "False", "False"])
            self.assertEqual(out["column"], "recovered_snr_std")
            qc = _json.load(open(root / "runs" / "r" / "stages" / "stage_h04_qc.json"))
            self.assertEqual(qc["snr_standardization"]["threshold_snr"], 5.0)
            self.assertEqual(qc["derived_finalize"]["rows_changed"], 1)

    def test_bajar_el_umbral_solo_cambia_complete(self):
        import csv as _csv
        import tempfile

        from musepipe.stages.stage_h04_injection import finalize_stage_h04

        with tempfile.TemporaryDirectory() as tmp:
            root = self._run(tmp, threshold=3.5)
            tabla = root / "runs" / "r" / "tables" / "injection_throughput_by_method.csv"
            antes = list(_csv.DictReader(open(tabla)))
            finalize_stage_h04("r", project_root=root)
            despues = list(_csv.DictReader(open(tabla)))
            self.assertEqual([f["complete"] for f in despues], ["True", "True", "False"])
            for a, d in zip(antes, despues):
                for k in a:
                    if k != "complete":
                        self.assertEqual(a[k], d[k], f"la columna {k} no se puede tocar")

    def test_es_idempotente(self):
        import tempfile

        from musepipe.stages.stage_h04_injection import finalize_stage_h04

        with tempfile.TemporaryDirectory() as tmp:
            root = self._run(tmp, threshold=3.5)
            finalize_stage_h04("r", project_root=root)
            segunda = finalize_stage_h04("r", project_root=root)
            self.assertEqual(segunda["rows_changed"], 0)

    def test_sin_la_columna_de_la_escala_para(self):
        import tempfile

        from musepipe.stages.stage_h04_injection import finalize_stage_h04

        with tempfile.TemporaryDirectory() as tmp:
            root = self._run(tmp, columna_std=False)
            with self.assertRaises(RuntimeError) as ctx:
                finalize_stage_h04("r", project_root=root)
            self.assertIn("recovered_snr_std", str(ctx.exception))


class TestSustratoPsfsub(unittest.TestCase):
    """De qué cubo sale `optimal_psfsub` en E4 y de cuál en C3.

    No es una puerta: es una declaración. El defecto que cubre es de los que no
    fallan —todo da rc=0 y los números salen— y sólo se ve leyendo dos módulos a
    la vez, que es exactamente lo que un QC debería ahorrarte.
    """

    def _paths(self, tmp, *, perobs, c3_source):
        import json as _json
        from pathlib import Path as _Path

        from musepipe.stages.stage_h04_injection import stage_h04_paths

        root = _Path(tmp)
        paths = stage_h04_paths("r", project_root=root)
        paths["paths"].ensure_base_dirs()
        stage_dir = paths["paths"].stage_dir
        if perobs:
            (stage_dir / "cube_psfsub_perobs.fits").write_bytes(b"")
        if c3_source is not None:
            (stage_dir / "spec_optimal_qc.json").write_text(
                _json.dumps({"psfsub_model": {"source": c3_source}})
            )
        return paths

    def test_declara_el_desajuste_cuando_c3_usa_el_cubo_de_c1b(self):
        import tempfile

        from musepipe.stages.stage_h04_injection import psfsub_substrate_check

        with tempfile.TemporaryDirectory() as tmp:
            paths = self._paths(tmp, perobs=True, c3_source="C1b_perobs_subtract")
            d = psfsub_substrate_check({}, paths)
            self.assertTrue(d["mismatch"])
            self.assertEqual(d["h04_substrate"], "in_memory_single_model_subtraction")
            self.assertEqual(d["c3_substrate"], "C1b_perobs_subtract")

    def test_sin_cubo_de_c1b_no_hay_desajuste(self):
        import tempfile

        from musepipe.stages.stage_h04_injection import psfsub_substrate_check

        with tempfile.TemporaryDirectory() as tmp:
            paths = self._paths(tmp, perobs=False, c3_source="combined_cube")
            d = psfsub_substrate_check({}, paths)
            self.assertFalse(d["mismatch"])
            self.assertFalse(d["perobs_cube_exists"])

    def test_el_cubo_puede_existir_sin_que_c3_lo_use(self):
        # `x02_psfsub_per_observation=false` deja el cubo en disco y a C3 sobre el
        # combinado: entonces los dos sustratos coinciden y no hay que avisar.
        import tempfile

        from musepipe.stages.stage_h04_injection import psfsub_substrate_check

        with tempfile.TemporaryDirectory() as tmp:
            paths = self._paths(tmp, perobs=True, c3_source="combined_cube")
            self.assertFalse(psfsub_substrate_check({}, paths)["mismatch"])

    def test_respeta_el_knob_que_declara_otro_cubo(self):
        import tempfile
        from pathlib import Path as _Path

        from musepipe.stages.stage_h04_injection import psfsub_substrate_check

        with tempfile.TemporaryDirectory() as tmp:
            paths = self._paths(tmp, perobs=False, c3_source="C1b_perobs_subtract")
            otro = _Path(tmp) / "otro.fits"
            otro.write_bytes(b"")
            d = psfsub_substrate_check({"x02_psfsub_cube_fits": str(otro)}, paths)
            self.assertTrue(d["perobs_cube_exists"])

    def test_sin_paths_declara_lo_que_sabe_en_vez_de_reventar(self):
        # `_qc_from_rows` se llama con paths=None en algunas pruebas de contrato:
        # la declaracion no puede ser la que rompa la etapa.
        from musepipe.stages.stage_h04_injection import psfsub_substrate_check

        d = psfsub_substrate_check({}, None)
        self.assertEqual(d["h04_substrate"], "in_memory_single_model_subtraction")
        self.assertIsNone(d["mismatch"])
        self.assertIsNone(d["perobs_cube_exists"])
