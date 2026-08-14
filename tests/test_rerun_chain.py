"""El conductor de la cadena: `scripts/rerun_chain.py`.

Ninguna prueba corre una etapa de verdad — `subprocess.run` va pinchado. Lo que
se fija aqui es lo que hace peligroso a un conductor: que se desincronice del
registro de etapas, que escriba en un run legacy, o que se coma un fallo.
"""
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]


def _carga():
    spec = importlib.util.spec_from_file_location(
        "rerun_chain", ROOT / "scripts" / "rerun_chain.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["rerun_chain"] = mod
    spec.loader.exec_module(mod)
    return mod


RC = _carga()


class _Proc:
    """Lo mínimo que el conductor mira de un `CompletedProcess`."""

    def __init__(self, returncode=0, stderr=""):
        self.returncode = returncode
        self.stderr = stderr
        self.stdout = ""


def _run_falso(tmp, run_id="RUN_X", legacy=False, qcs=()):
    """Un run de mentira en disco, con su config y los QC que se le pidan."""
    run_dir = Path(tmp) / "runs" / run_id
    (run_dir / "config").mkdir(parents=True)
    (run_dir / "config" / "config.json").write_text(json.dumps(
        {"meta": {"legacy": True} if legacy else {}, "config": {}}), encoding="utf-8")
    for rel in qcs:
        p = run_dir / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("{}", encoding="utf-8")
    return run_dir


class RegistryAgreementTests(unittest.TestCase):
    """La cadena del conductor tiene que ser la del registro, no una copia suya.

    `musepipe/stage_registry.py` es la fuente de verdad: si alguien renombra una
    etapa o le cambia el QC, esto falla en CI en vez de a las 3 de la mañana
    dejando el run a medias.
    """

    def test_every_stage_exists_in_the_registry(self):
        from musepipe.stage_registry import stage_ids

        conocidas = set(stage_ids())
        for etapa, _cmd, _qc in RC.CADENA:
            with self.subTest(etapa=etapa):
                self.assertIn(etapa, conocidas)

    def test_the_qc_is_the_one_the_registry_declares(self):
        for etapa, _cmd, qc in RC.CADENA:
            with self.subTest(etapa=etapa):
                self.assertIn(qc, RC.qc_del_registro(etapa),
                              f"{etapa}: el QC del conductor no es el del registro")

    def test_the_chain_runs_C1_first_and_G5_last(self):
        """El orden importa: cada etapa consume el producto de la anterior."""
        self.assertEqual(RC.IDS[0], "C1")
        self.assertEqual(RC.IDS[-1], "G5")
        self.assertEqual(len(set(RC.IDS)), len(RC.IDS), "hay etapas repetidas")


class SelectionTests(unittest.TestCase):
    def test_desde_and_hasta_take_a_closed_range(self):
        sel = [e for e, _c, _q in RC.selecciona("D1", "E1", None)]
        self.assertEqual(sel, ["D1", "D2", "E1"])

    def test_solo_takes_exactly_what_it_says(self):
        sel = [e for e, _c, _q in RC.selecciona(None, None, "C1, G5")]
        self.assertEqual(sel, ["C1", "G5"])

    def test_an_unknown_stage_stops_instead_of_running_the_rest(self):
        with self.assertRaises(SystemExit):
            RC.selecciona(None, None, "C1,NOEXISTE")

    def test_a_backwards_range_stops(self):
        with self.assertRaises(SystemExit):
            RC.selecciona("G5", "C1", None)


class SafetyTests(unittest.TestCase):
    def test_a_legacy_run_is_refused_before_launching_anything(self):
        with tempfile.TemporaryDirectory() as tmp:
            _run_falso(tmp, legacy=True)
            with mock.patch.object(subprocess, "run") as corre:
                with self.assertRaises(SystemExit) as caja:
                    RC.main(["--run-id", "RUN_X", "--project-root", tmp, "--solo", "C1"])
            self.assertIn("legacy", str(caja.exception))
            corre.assert_not_called()

    def test_dry_run_launches_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            _run_falso(tmp)
            with mock.patch.object(RC.subprocess, "run") as corre:
                rc = RC.main(["--run-id", "RUN_X", "--project-root", tmp, "--dry-run"])
            self.assertEqual(rc, 0)
            corre.assert_not_called()

    def test_the_PARAR_file_stops_before_the_first_stage(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = _run_falso(tmp)
            (run_dir / "logs").mkdir()
            (run_dir / "logs" / "PARAR").write_text("", encoding="utf-8")
            with mock.patch.object(RC.subprocess, "run") as corre:
                rc = RC.main(["--run-id", "RUN_X", "--project-root", tmp, "--solo", "C1,D1"])
            self.assertEqual(rc, 0)
            corre.assert_not_called()


class LogAndExitCodeTests(unittest.TestCase):
    def test_a_failing_stage_gives_a_non_zero_exit_and_lands_in_the_log(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = _run_falso(tmp)
            with mock.patch.object(RC.subprocess, "run",
                                   return_value=_Proc(2, "boom: se cayo\n")):
                rc = RC.main(["--run-id", "RUN_X", "--project-root", tmp, "--solo", "C1"])
            self.assertEqual(rc, 1)
            log = json.loads((run_dir / "logs" / "rerun_log.json").read_text(encoding="utf-8"))
            self.assertEqual(log[0]["etapa"], "C1")
            self.assertEqual(log[0]["rc"], 2)
            self.assertIn("boom", log[0]["stderr"])

    def test_a_stage_that_writes_no_qc_is_not_reported_as_OK(self):
        """rc=0 no basta: si el QC no se reescribe, la etapa no hizo nada.

        Le paso a G3 en el re-run del 2026-08-12: `python -m` sobre un modulo
        sin `main()` termina con rc 0 y sin escribir una linea.
        """
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = _run_falso(tmp)
            with mock.patch.object(RC.subprocess, "run", return_value=_Proc(0)):
                rc = RC.main(["--run-id", "RUN_X", "--project-root", tmp, "--solo", "C1"])
            self.assertEqual(rc, 0)   # no es un fallo del proceso...
            log = json.loads((run_dir / "logs" / "rerun_log.json").read_text(encoding="utf-8"))
            self.assertFalse(log[0]["qc_reescrito"])   # ...pero queda registrado

    def test_the_qc_counts_as_rewritten_only_if_its_mtime_moves(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = _run_falso(tmp, qcs=("stages/stage_e01_qc.json",))
            qc = run_dir / "stages" / "stage_e01_qc.json"

            def escribe(*_a, **_k):
                import os
                import time as _t
                os.utime(qc, (_t.time() + 10, _t.time() + 10))
                return _Proc(0)

            with mock.patch.object(RC.subprocess, "run", side_effect=escribe):
                RC.main(["--run-id", "RUN_X", "--project-root", tmp, "--solo", "C1"])
            log = json.loads((run_dir / "logs" / "rerun_log.json").read_text(encoding="utf-8"))
            self.assertTrue(log[0]["qc_reescrito"])

    def test_the_run_id_reaches_the_command(self):
        vistos = []

        with tempfile.TemporaryDirectory() as tmp:
            _run_falso(tmp)

            def espia(cmd, **_k):
                vistos.append(list(cmd))
                return _Proc(0)

            with mock.patch.object(RC.subprocess, "run", side_effect=espia):
                RC.main(["--run-id", "RUN_X", "--project-root", tmp, "--solo", "C1"])
        self.assertTrue(vistos)
        self.assertIn("RUN_X", vistos[0])
        self.assertNotIn("{run}", " ".join(vistos[0]))


if __name__ == "__main__":
    unittest.main()
