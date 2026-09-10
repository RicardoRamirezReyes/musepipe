import csv
import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from musepipe.stages.stage_h03_limits import (
    G_CGS,
    L_SUN_ERG_S,
    M_SUN_G,
    MAGNETOSPHERIC_FACTOR,
    PC_CM,
    R_SUN_CM,
    YR_S,
    compute_stage_h03_products,
    limit_conversion_chain,
    resolve_tail_estimator,
    stage_h03_paths,
    write_stage_h03_products,
    z_threshold_for,
)


def h03_physical_config(run_id, root):
    return {
        "run_id": run_id,
        "project_root": str(root),
        "h03_distance_pc": 10.0,
        "h03_distance_err_pc": 0.0,
        "h03_distance_source": "synthetic distance",
        "h03_av": 0.0,
        "h03_av_err": 0.0,
        "h03_av_source": "synthetic extinction",
        "h03_rv_sys_kms": 0.0,
        "h03_rv_source": "synthetic rv",
        "h03_a_halpha_over_av": 1.0,
        "h03_extinction_law_citation": "synthetic law",
        "h03_lacc_lha_relation": "identity",
        "h03_lacc_lha_citation": "synthetic relation",
        "h03_lacc_lha_a": 1.0,
        "h03_lacc_lha_b": 0.0,
        "h03_relation_scatter_dex": 0.3,
        "h03_companion_mass_msun": 1.0,
        "h03_companion_radius_rsun": 1.0,
        "h03_mass_source": "synthetic mass",
        "h03_radius_source": "synthetic radius",
        # Las sigmas sinteticas de abajo ya van en cgs, asi que la escala es 1.
        # Se declara explicitamente: H03 ya no asume un default (un run sin esta
        # clave ni BUNIT en el producto es un error, no un 1.0 silencioso).
        "h03_flux_unit_cgs": 1.0,
        "h03_canonical_method": "psffit",
        "h03_template_width_factors": [1.0, 2.0],
        "h03_matched_sigma_by_method": {"psffit": {"1": 2.0e-17, "2": 3.0e-17}},
        "h03_separation_arcsec": 1.7,
        "g3_lacc_relations": {
            "halpha_aoyama21": {"a": 0.95, "b": 1.61, "scatter_dex": 0.3,
                                "citation": "Aoyama et al. 2021", "validity_range": "planetary"}
        },
    }


class H03ChainTests(unittest.TestCase):
    def test_conversion_chain_matches_analytic_values(self):
        result = limit_conversion_chain(
            z_threshold=3.0,
            z_5sigma_extrap=5.0,
            matched_sigma=2.0e-17,
            throughput=0.5,
            throughput_err=0.0,
            distance_pc=10.0,
            distance_err_pc=0.0,
            av=0.0,
            av_err=0.0,
            a_halpha_over_av=1.0,
            lacc_lha_slope=1.0,
            lacc_lha_intercept=0.0,
            relation_scatter_dex=0.3,
            mass_msun=1.0,
            radius_rsun=1.0,
        )

        f_stat = 3.0 * 2.0e-17
        f_obs = f_stat / 0.5
        lha = 4.0 * math.pi * (10.0 * PC_CM) ** 2 * f_obs
        lacc_lsun = lha / L_SUN_ERG_S
        mdot = (MAGNETOSPHERIC_FACTOR * (lacc_lsun * L_SUN_ERG_S) * R_SUN_CM
                / (G_CGS * M_SUN_G) * YR_S / M_SUN_G)

        self.assertAlmostEqual(result["f_stat_99"], f_stat)
        self.assertAlmostEqual(result["f_lim_observed"], f_obs)
        self.assertAlmostEqual(result["f_lim_dereddened"], f_obs)
        self.assertAlmostEqual(result["l_halpha_erg_s"], lha)
        self.assertAlmostEqual(result["l_acc_lsun"], lacc_lsun)
        self.assertAlmostEqual(result["mdot_msun_yr"], mdot)
        self.assertAlmostEqual(result["mdot_err_dex"], 0.3)

    def test_limit_and_detection_use_the_same_magnetospheric_factor(self):
        """El limite de E3 y la deteccion de G3 tienen que ser la MISMA formula.

        No lo eran: G3 aplicaba `(1 - R/R_in)^-1 = 1.25` via
        `models.accretion.mdot_from_lacc` y E3 devolvia `L_acc R / (G M)` a secas,
        asi que el limite salia 1.25x mas apretado que la medida con la que el
        paper lo compara -- y las dos cifras viajaban juntas en la misma tabla.
        """
        from musepipe.models.accretion import mdot_from_lacc

        result = limit_conversion_chain(
            z_threshold=3.0,
            z_5sigma_extrap=5.0,
            matched_sigma=2.0e-17,
            throughput=0.5,
            throughput_err=0.0,
            distance_pc=10.0,
            distance_err_pc=0.0,
            av=0.0,
            av_err=0.0,
            a_halpha_over_av=1.0,
            lacc_lha_slope=1.0,
            lacc_lha_intercept=0.0,
            relation_scatter_dex=0.3,
            mass_msun=0.02,
            radius_rsun=0.13,
        )
        por_g3 = mdot_from_lacc(result["l_acc_lsun"], 0.02, 0.13)
        self.assertAlmostEqual(result["mdot_msun_yr"] / por_g3, 1.0, places=12)

    def test_aoyama21_alt_relation_absent_and_present(self):
        base = dict(
            z_threshold=3.0, z_5sigma_extrap=5.0, matched_sigma=2.0e-17,
            throughput=0.5, throughput_err=0.0, distance_pc=10.0, distance_err_pc=0.0,
            av=0.0, av_err=0.0, a_halpha_over_av=1.0, lacc_lha_slope=1.13,
            lacc_lha_intercept=1.74, relation_scatter_dex=0.3, mass_msun=1.0, radius_rsun=1.0,
        )
        # Absent -> aoyama keys present but NaN, headline untouched.
        without = limit_conversion_chain(**base)
        self.assertTrue(math.isnan(without["mdot_aoyama21_msun_yr"]))
        self.assertTrue(math.isnan(without["l_acc_aoyama21_lsun"]))

        # Present -> parallel keys; the ratio of Mdot limits follows the analytic
        # 10^((a1-a2) logL_Halpha + (b1-b2)) with (a1,b1)=(1.13,1.74) Alcala and
        # (a2,b2)=(0.95,1.61) Aoyama+21. Alcala must be MORE restrictive here.
        with_alt = limit_conversion_chain(
            **base, alt_lacc_slope=0.95, alt_lacc_intercept=1.61, alt_scatter_dex=0.3
        )
        self.assertEqual(with_alt["mdot_msun_yr"], without["mdot_msun_yr"])  # headline unchanged
        log_lha = math.log10(with_alt["l_halpha_lsun"])
        expected_ratio = 10.0 ** ((1.13 - 0.95) * log_lha + (1.74 - 1.61))
        actual_ratio = with_alt["mdot_msun_yr"] / with_alt["mdot_aoyama21_msun_yr"]
        self.assertAlmostEqual(actual_ratio, expected_ratio, places=6)
        # L_Halpha << 1e-6 Lsun here -> Alcala limit is the tighter (smaller) one.
        self.assertLess(with_alt["mdot_msun_yr"], with_alt["mdot_aoyama21_msun_yr"])
        self.assertAlmostEqual(
            with_alt["mdot_aoyama21_err_dex"],
            math.sqrt((0.95 * (0.0)) ** 2 + 0.3 ** 2), places=6,
        )

    def test_stage_h03_writes_upper_limit_table_qc_and_context_plot(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_id = "synthetic_h03"
            paths = stage_h03_paths(run_id, project_root=root)
            paths["paths"].ensure_base_dirs()
            paths["stage_h01_qc_json"].write_text(
                json.dumps(
                    {
                        "verdict": {"verdict": "non_detection"},
                        # E1 siempre declara con que cola decide, y H03 toma de
                        # ahi el umbral del 99 % si el run no lo declara aparte.
                        "criterion": {"fap_estimator": "empirical"},
                        "line": {"rest_A": 6562.8, "rv_sys_kms": 0.0},
                        "templates": {"lsf_fwhm_A": 2.5},
                    }
                ),
                encoding="utf-8",
            )
            paths["stage_h02_qc_json"].write_text(json.dumps({"overall": "survives"}), encoding="utf-8")
            paths["stage_h04_qc_json"].write_text(
                json.dumps({"regression_historic": {"verdict": "pass"}, "throughput": {"psf_perturbation_pct": 0.0}}),
                encoding="utf-8",
            )
            null_maxima = np.linspace(-1.0, 2.5, 120)
            np.savez(paths["h01_null_maxima_npz"], psffit_null_maxima=null_maxima)
            with paths["halpha_detection_csv"].open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=["method", "template_factor", "matched_sigma"])
                writer.writeheader()
                writer.writerow({"method": "psffit", "template_factor": 1.0, "matched_sigma": 2.0e-17})
            with paths["injection_throughput_csv"].open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=["method", "template_factor", "snr", "throughput", "throughput_err"],
                )
                writer.writeheader()
                for factor in (1.0, 2.0):
                    writer.writerow({"method": "psffit", "template_factor": factor, "snr": 1.0, "throughput": 0.4, "throughput_err": 0.04})
                    writer.writerow({"method": "psffit", "template_factor": factor, "snr": 5.0, "throughput": 0.8, "throughput_err": 0.08})

            cfg = h03_physical_config(run_id, root)
            product = compute_stage_h03_products(cfg, paths)
            written = write_stage_h03_products(product, cfg, paths)

            self.assertTrue(paths["halpha_upper_limits_csv"].exists())
            self.assertTrue(paths["stage_h03_qc_json"].exists())
            self.assertTrue(paths["context_plot"].exists())
            self.assertEqual(written["qc"]["prerequisites"]["e4_v1"], "pass")
            self.assertEqual(written["qc"]["canonical_method"], "psffit")
            self.assertTrue(any(row["row_kind"] == "combined_final" for row in product.rows))
            # R1: parallel Aoyama+21 accretion limit is computed and surfaced.
            combined = next(r for r in product.rows if r["row_kind"] == "combined_final")
            self.assertIsNotNone(combined["mdot_aoyama21_msun_yr"])
            self.assertNotEqual(combined["mdot_aoyama21_msun_yr"], combined["mdot_msun_yr"])
            self.assertIsNotNone(
                written["qc"]["physical_inputs"]["lacc_aoyama21_relation"]
            )
            self.assertTrue(all("mdot_aoyama21" in lim for lim in written["qc"]["limits"]))


class H03TailEstimatorTests(unittest.TestCase):
    """De que cola sale el umbral del 99 %, y que pasa si nadie lo declara.

    H03 tenia `z_99_empirical` incrustado. Cuando E1 paso a decidir con la cola
    parametrica, los limites siguieron saliendo del cuantil empirico: el paper
    citaba una deteccion y un limite calibrados con umbrales distintos, y el
    limite quedaba un 32 % mas apretado de lo que da el metodo declarado.
    """

    TAIL = {"z_99_empirical": 2.5066, "z_99_gumbel": 3.3091}

    def test_estimator_follows_e1_when_the_run_does_not_declare_one(self):
        for estimador in ("empirical", "parametric"):
            with self.subTest(estimador=estimador):
                valor, fuente = resolve_tail_estimator(
                    {}, {"criterion": {"fap_estimator": estimador}})
                self.assertEqual(valor, estimador)
                self.assertEqual(fuente, "h01_qc.criterion.fap_estimator")

    def test_declared_knob_wins_over_e1(self):
        valor, fuente = resolve_tail_estimator(
            {"h03_tail_estimator": "empirical"},
            {"criterion": {"fap_estimator": "parametric"}})
        self.assertEqual(valor, "empirical")
        self.assertEqual(fuente, "config.h03_tail_estimator")

    def test_no_silent_default(self):
        """Sin knob y sin QC de E1 es un error, no `empirical` por su cuenta."""
        for cfg, qc in (({}, {}), ({}, None), ({"h03_tail_estimator": "gumbel"}, {})):
            with self.subTest(cfg=cfg, qc=qc), self.assertRaises(RuntimeError):
                resolve_tail_estimator(cfg, qc)

    def test_parametric_threshold_is_the_gumbel_one(self):
        self.assertEqual(z_threshold_for(self.TAIL, "empirical"), 2.5066)
        self.assertEqual(z_threshold_for(self.TAIL, "parametric"), 3.3091)


if __name__ == "__main__":
    unittest.main()
