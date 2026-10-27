"""`scripts/recomputa_e1_e3.py`: E1 y E3 de un run recalculados SIN escribir en `runs/`.

Las clases rápidas no tocan `runs/`: el remapeo a una copia, las rutas temporales del E1, la guarda que
para cualquier salida que caiga en `runs/` (también a través de un enlace simbólico, como los runs de
prueba que enlazan los productos del run del paper) y, con las etapas sustituidas por dobles que
escriben donde les dicen, que `recalcula` no deja nada en el run y borra su carpeta temporal.

`DecisionCTests` y `DecisionDTests` (marcadas `slow`, ~2 y ~3.5 min: E1 hace 2000 bootstraps por
método) corren la herramienta sobre los runs reales, comprueban que ningún fichero del run cambia
(tamaño y mtime de todo el árbol) y fijan las cifras del paper (decisiones C y D del 2026-10-26),
bit a bit contra las de la sesión del 2026-10-25. Buscan `runs/` en este clon o en
`MUSE_PROJECT_ROOT`; si faltan los runs, se saltan.
"""
import importlib.util
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _carga(nombre):
    spec = importlib.util.spec_from_file_location(nombre, ROOT / "scripts" / f"{nombre}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[nombre] = mod
    spec.loader.exec_module(mod)
    return mod


def _raiz_con_runs(*runs):
    for raiz in (ROOT, os.environ.get("MUSE_PROJECT_ROOT")):
        if raiz and all((Path(raiz) / "runs" / r / "config" / "config.json").exists() for r in runs):
            return Path(raiz)
    return None


def foto(carpeta):
    """(tamaño, mtime_ns) de cada fichero y el listado de cada carpeta, con el run ya resuelto."""
    base = Path(carpeta).resolve()
    out = {}
    for dp, dn, fn in os.walk(base):
        out[dp + "/"] = (tuple(sorted(dn)), tuple(sorted(fn)))
        for f in fn:
            st = os.stat(os.path.join(dp, f))
            out[os.path.join(dp, f)] = (st.st_size, st.st_mtime_ns)
    return out


def _proyecto_sintetico(tmp, run="SINTETICO", enlazado=True):
    """Un proyecto con runs/<run> como enlace a otra carpeta (como /mnt/2TB) y productos dentro."""
    tmp = Path(tmp)
    real = tmp / "disco" / run
    for sub in ("config", "stages", "tables", "plots", "logs"):
        (real / sub).mkdir(parents=True)
    (real / "config" / "config.json").write_text(json.dumps({"config": {"run_id": run}}), encoding="utf-8")
    for nombre in ("stage_h01_qc.json", "stage_h01_null_maxima.npz", "spec_calibrated_psffit_object.fits"):
        (real / "stages" / nombre).write_text("producto", encoding="utf-8")
    (real / "tables" / "halpha_detection_by_method.csv").write_text("method\n", encoding="utf-8")
    proyecto = tmp / "proyecto"
    (proyecto / "runs").mkdir(parents=True)
    if enlazado:
        (proyecto / "runs" / run).symlink_to(real, target_is_directory=True)
    return proyecto, real


class RutasTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.m = _carga("recomputa_e1_e3")

    def test_el_objetivo_y_su_copia(self):
        raiz = Path("/proyecto")
        self.assertEqual(self.m.parsea_objetivo("ROXs12b_invvar", raiz), ("ROXs12b_invvar", None))
        run, copia = self.m.parsea_objetivo("R@runs/R/stages/_antes_x", raiz)
        self.assertEqual((run, copia), ("R", Path("/proyecto/runs/R/stages/_antes_x")))
        self.assertEqual(self.m.parsea_objetivo("R@/otra/copia", raiz)[1], Path("/otra/copia"))
        with self.assertRaises(ValueError):
            self.m.parsea_objetivo("@/x", raiz)

    def test_la_copia_recibe_stages_y_tables_y_nada_mas(self):
        from musepipe.paths import RunPaths
        from musepipe.stages.stage_h03_limits import stage_h03_paths

        rp = RunPaths.from_project_root("R", "/p")
        copia = Path("/p/runs/R/stages/_antes_c4_c0_x")
        self.assertEqual(self.m.remapea(rp.stage_dir / "a.json", rp, copia), copia / "stages" / "a.json")
        self.assertEqual(self.m.remapea(rp.table_dir / "t.csv", rp, copia), copia / "tables" / "t.csv")
        self.assertEqual(self.m.remapea(Path("/p/musepipe/x.csv"), rp, copia), Path("/p/musepipe/x.csv"))
        self.assertEqual(self.m.remapea(rp.stage_dir / "a.json", rp, None), rp.stage_dir / "a.json")
        p3 = self.m.remapea_rutas(stage_h03_paths("R", "/p"), copia)
        self.assertIs(p3["paths"].__class__, RunPaths)
        self.assertEqual(p3["stage_h04_qc_json"], copia / "stages" / "stage_h04_qc.json")
        self.assertEqual(p3["injection_throughput_csv"], copia / "tables" / "injection_throughput_by_method.csv")
        self.assertEqual(p3["literature_csv"], Path("/p/musepipe/qc/data/halpha_literature_limits.csv"))

    def test_los_knobs_de_entrada_de_e1_apuntan_a_la_copia(self):
        from musepipe.stages.stage_h01_detect import stage_h01_paths

        p1 = stage_h01_paths("R", "/p")
        copia = Path("/c")
        ov = self.m.overrides_de_entradas({}, p1, copia)
        self.assertEqual(len(ov), 1 + 2 * 6)
        self.assertEqual(ov["h01_spec_calibrated_psffit_object"], "/c/stages/spec_calibrated_psffit_object.fits")
        self.assertEqual(ov["h01_controls_lpm_npz"], "/c/stages/spec_calibrated_lpm_controls.npz")
        self.assertEqual(ov["h01_stage00q_qc_json"], "/c/stages/stage00q_qc.json")
        # un knob que ya apunta fuera del run se respeta
        ov2 = self.m.overrides_de_entradas({"h01_controls_lpm_npz": "/fuera/lpm.npz"}, p1, copia)
        self.assertNotIn("h01_controls_lpm_npz", ov2)
        self.assertEqual(self.m.overrides_de_entradas({}, p1, None), {})

    def test_las_salidas_de_e1_quedan_en_la_temporal(self):
        from musepipe.stages.stage_h01_detect import stage_h01_paths

        with tempfile.TemporaryDirectory() as tmp:
            proyecto, real = _proyecto_sintetico(tmp)
            p1 = stage_h01_paths("SINTETICO", proyecto)
            temporal = Path(tmp) / "temporal"
            t = self.m.rutas_e1_temporales(p1, "SINTETICO", temporal)
            for s in self.m.salidas_de(t):
                self.assertTrue(s == temporal or temporal in s.parents, s)
            self.m.comprueba_fuera(self.m.salidas_de(t), self.m.carpetas_prohibidas(proyecto, "SINTETICO"))
            # las entradas no se tocan
            self.assertEqual(t["spec_calibrated_psffit_object"], p1["spec_calibrated_psffit_object"])

    def test_la_guarda_para_lo_que_cae_en_el_run_aunque_sea_por_un_enlace(self):
        with tempfile.TemporaryDirectory() as tmp:
            proyecto, real = _proyecto_sintetico(tmp)
            prohibidas = self.m.carpetas_prohibidas(proyecto, "SINTETICO")
            for ruta in (proyecto / "runs" / "SINTETICO" / "stages" / "stage_h01_qc.json",  # por el enlace del run
                         real / "tables" / "x.csv",                                       # el run resuelto
                         proyecto / "runs" / "OTRO" / "stages" / "y.json"):                # cualquier run
                with self.subTest(ruta=ruta), self.assertRaises(RuntimeError):
                    self.m.comprueba_fuera([ruta], prohibidas)
            # un run de prueba cuyo stages/ enlaza el de otro run
            (proyecto / "runs" / "PRUEBA" / "stages").mkdir(parents=True)
            enlace = proyecto / "runs" / "PRUEBA" / "stages" / "stage_h01_qc.json"
            enlace.symlink_to(real / "stages" / "stage_h01_qc.json")
            with self.assertRaises(RuntimeError):
                self.m.comprueba_fuera([enlace], self.m.carpetas_prohibidas(proyecto, "PRUEBA"))
            self.m.comprueba_fuera([Path(tmp) / "temporal" / "a.json"], prohibidas)


class SinEscribirTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.m = _carga("recomputa_e1_e3")

    def test_sin_la_bandera_no_corre(self):
        with mock.patch("sys.stderr"), self.assertRaises(SystemExit) as cm:
            self.m.main(["--run-id", "X"])
        self.assertEqual(cm.exception.code, 2)

    def test_no_escribe_el_json_dentro_de_runs(self):
        with tempfile.TemporaryDirectory() as tmp:
            proyecto, _real = _proyecto_sintetico(tmp)
            with self.assertRaises(RuntimeError):
                self.m.main(["--sin-escribir", "--run-id", "SINTETICO", "--project-root", str(proyecto),
                             "--json", str(proyecto / "runs" / "SINTETICO" / "e1.json")])

    def test_recalcula_no_deja_nada_en_el_run_y_borra_su_temporal(self):
        llamadas = {}

        def e1_doble(cfg, paths):
            llamadas["e1_lee"] = Path(paths["spec_calibrated_psffit_object"])
            return SimpleNamespace(
                rows=[{"method": "psffit", "matched_z": 2.5, "template_factor": 1.0, "global_empirical_fap": 0.03,
                       "global_parametric_fap": 0.02, "parametric_fap_ks_p": 0.5,
                       "parametric_fap_extrapolation_sd": 0.0, "n_controls": 33}],
                qc={"verdict": {"verdict": "non_detection"},
                    "parametric_fap": {"by_method": {"psffit": {"ci95": [0.01, 0.05]}}}})

        def escritor_doble(producto, cfg, paths):
            # lo que hace el escritor de la etapa: crea las carpetas de su RunPaths y escribe sus salidas
            paths["paths"].ensure_base_dirs()
            llamadas["escribe"] = []
            for clave in ("halpha_detection_csv", "null_maxima_npz", "stage_h01_qc_json", "summary_plot"):
                Path(paths[clave]).parent.mkdir(parents=True, exist_ok=True)
                Path(paths[clave]).write_text("e1 temporal", encoding="utf-8")
                llamadas["escribe"].append(Path(paths[clave]))

        def e3_doble(cfg, paths):
            llamadas["e3_lee"] = {k: Path(paths[k]) for k in ("stage_h01_qc_json", "h01_null_maxima_npz",
                                                               "halpha_detection_csv", "stage_h04_qc_json")}
            assert all(llamadas["e3_lee"][k].read_text() == "e1 temporal"
                       for k in ("stage_h01_qc_json", "h01_null_maxima_npz", "halpha_detection_csv"))
            return SimpleNamespace(rows=[{"row_kind": "combined_final", "method": "combined",
                                          "limit_definition": "detection_threshold", "z_threshold": 3.6,
                                          "f_stat_99": 6e-18, "throughput": 1.0, "mdot_msun_yr": 3.6e-14}])

        with tempfile.TemporaryDirectory() as tmp:
            proyecto, real = _proyecto_sintetico(tmp)
            base_tmp = Path(tmp) / "temporales"
            base_tmp.mkdir()
            antes = foto(real)
            with mock.patch.object(self.m, "compute_stage_h01_products", e1_doble), \
                    mock.patch.object(self.m, "write_stage_h01_products", escritor_doble), \
                    mock.patch.object(self.m, "compute_stage_h03_products", e3_doble):
                res = self.m.recalcula("SINTETICO", proyecto, tmp_base=base_tmp)
            self.assertEqual(foto(real), antes)
            self.assertEqual(list(base_tmp.iterdir()), [])  # la temporal se borró
            for p in llamadas["escribe"]:
                self.assertIn(base_tmp, p.parents)
            self.assertEqual(llamadas["e1_lee"].resolve(), (real / "stages" / "spec_calibrated_psffit_object.fits"))
            self.assertEqual(llamadas["e3_lee"]["stage_h04_qc_json"].resolve(), real / "stages" / "stage_h04_qc.json")
            self.assertEqual(res["e3"]["mdot_msun_yr"], 3.6e-14)
            self.assertEqual(res["e1"]["filas"]["psffit"]["fap_bootstrap_95"], [0.01, 0.05])

    def test_mas_hondo_es_contra_el_primero(self):
        r = [{"run_id": "A", "copia": None, "e3": {"mdot_msun_yr": 3.0}},
             {"run_id": "B", "copia": None, "e3": {"mdot_msun_yr": 4.0}},
             {"run_id": "C", "copia": None}]
        h = self.m.mas_hondo(r)
        self.assertEqual(len(h), 1)
        self.assertAlmostEqual(h[0]["mas_hondo"], 0.25, places=12)


RAIZ_C = _raiz_con_runs("ROXs42Bb_realigned", "ROXs42Bb_r78_20260927")
RAIZ_D = _raiz_con_runs("ROXs12b_invvar", "ROXs12b_OB3445598")
COPIA_NOCHE = "runs/ROXs12b_OB3445598/stages/_antes_c4_c0_20261025T030808"


@pytest.mark.slow
@unittest.skipUnless(RAIZ_C is not None, "sin los runs de 42B b (ni aquí ni en MUSE_PROJECT_ROOT)")
class DecisionCTests(unittest.TestCase):
    """E1 de ROXs 42B b a 55 y 78 px con c0; el de 55 px con su E3, para probar que no escribe."""

    @classmethod
    def setUpClass(cls):
        m = _carga("recomputa_e1_e3")
        runs = ("ROXs42Bb_realigned", "ROXs42Bb_r78_20260927")
        antes = {r: foto(RAIZ_C / "runs" / r) for r in runs}
        cls.r55 = m.recalcula("ROXs42Bb_realigned", RAIZ_C)
        cls.r78 = m.recalcula("ROXs42Bb_r78_20260927", RAIZ_C, solo_e1=True)
        cls.cambios = {r: foto(RAIZ_C / "runs" / r) == antes[r] for r in runs}

    def test_no_escribe_en_los_runs(self):
        self.assertEqual(self.cambios, {"ROXs42Bb_realigned": True, "ROXs42Bb_r78_20260927": True})

    def test_las_cifras_de_la_decision_c(self):
        p55, p78 = self.r55["e1"]["filas"]["psffit"], self.r78["e1"]["filas"]["psffit"]
        # la sesión (marginal_42bb/e1.json), bit a bit
        self.assertEqual((p55["global_parametric_fap"], p55["matched_z"]), (0.026954691238403196, 2.861296656647361))
        self.assertEqual((p78["global_parametric_fap"], p78["matched_z"]), (0.028490561699764733, 2.812763365804926))
        # lo que imprime el paper: 2.7e-2 y 2.8e-2, el intervalo [6.0e-3, 5.2e-2] y los dos non_detection
        self.assertEqual((f"{p55['global_parametric_fap']:.1e}", f"{p78['global_parametric_fap']:.1e}"),
                         ("2.7e-02", "2.8e-02"))
        self.assertEqual([f"{v:.1e}" for v in p55["fap_bootstrap_95"]], ["6.0e-03", "5.2e-02"])
        for r in (self.r55, self.r78):
            self.assertEqual(r["e1"]["veredicto"]["verdict"], "non_detection")
        # el E1 recalculado del run del paper es el que tiene escrito
        self.assertTrue(all(c["igual"] for f in self.r55["compara_con_disco"]["e1"].values() for c in f.values()))
        self.assertIn("e3", self.r55)


@pytest.mark.slow
@unittest.skipUnless(RAIZ_D is not None, "sin los runs de 12 b (ni aquí ni en MUSE_PROJECT_ROOT)")
class DecisionDTests(unittest.TestCase):
    """«15 % deeper (3.6 against 4.3e-14)»: la noche buena con c0, su copia de antes y el cubo invvar."""

    @classmethod
    def setUpClass(cls):
        m = _carga("recomputa_e1_e3")
        runs = ("ROXs12b_invvar", "ROXs12b_OB3445598")
        antes = {r: foto(RAIZ_D / "runs" / r) for r in runs}
        cls.invvar = m.recalcula("ROXs12b_invvar", RAIZ_D)
        cls.noche = m.recalcula("ROXs12b_OB3445598", RAIZ_D)
        copia = (RAIZ_D / COPIA_NOCHE).resolve()
        cls.copia = m.recalcula("ROXs12b_OB3445598", RAIZ_D, copia=copia) if copia.exists() else None
        cls.hondo = m.mas_hondo([cls.invvar, cls.noche])
        cls.cambios = {r: foto(RAIZ_D / "runs" / r) == antes[r] for r in runs}

    def test_no_escribe_en_los_runs(self):
        self.assertEqual(self.cambios, {"ROXs12b_invvar": True, "ROXs12b_OB3445598": True})

    def test_las_cifras_de_la_decision_d(self):
        self.assertEqual(self.invvar["e3"]["mdot_msun_yr"], 3.643371144406478e-14)
        self.assertTrue(self.invvar["compara_con_disco"]["e3"]["igual"])
        self.assertEqual(self.noche["e3"]["mdot_msun_yr"], 4.2992023765955676e-14)
        self.assertEqual(self.noche["e3"]["z_threshold"], 3.6777696887797946)
        self.assertEqual(self.hondo[0]["mas_hondo"], 0.15254718776658893)
        self.assertEqual(round(100 * self.hondo[0]["mas_hondo"]), 15)

    def test_la_copia_de_antes_de_c0_da_su_qc(self):
        if self.copia is None:
            self.skipTest(f"sin la copia {COPIA_NOCHE}")
        self.assertEqual(self.copia["e3"]["mdot_msun_yr"], 4.16151500457472e-14)
        self.assertTrue(self.copia["compara_con_disco"]["e3"]["igual"])
        self.assertEqual(round(self.copia["e3"]["z_threshold"], 2), 3.41)


if __name__ == "__main__":
    unittest.main()
