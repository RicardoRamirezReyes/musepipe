"""Contrato del registro de etapas (`musepipe/stage_registry.py`).

El registro es la fuente única de la parte machine-readable de cada etapa. Estos
tests impiden que vuelva a divergir de sus dos consumidores: el generador de
notebooks y `notebooks/_nbcommon.py`.

Contexto: plan `docs/2026-07-24_plan_multiobjeto_notebooks.md` (WP F1/F3).
"""
from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from musepipe import stage_registry as reg  # noqa: E402


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class RegistryShapeTests(unittest.TestCase):
    def test_ids_and_slugs_unique(self):
        ids = [s.id for s in reg.STAGES]
        slugs = [s.slug for s in reg.STAGES]
        self.assertCountEqual(ids, set(ids), "ids de etapa duplicados")
        self.assertCountEqual(slugs, set(slugs), "slugs de etapa duplicados")

    def test_qc_paths_belong_to_one_stage(self):
        """Una ruta de QC no puede pertenecer a dos etapas: `stage_for_qc` sería ambiguo."""
        owner: dict[str, str] = {}
        for stage in reg.STAGES:
            for path in stage.qc_paths:
                self.assertNotIn(
                    path, owner,
                    f"{path} reclamada por {owner.get(path)} y {stage.id}",
                )
                owner[path] = stage.id

    def test_stage_for_qc_resolves_canonical_and_aliases(self):
        for stage in reg.STAGES:
            for path in stage.qc_paths:
                self.assertEqual(reg.stage_for_qc(path).id, stage.id)

    def test_launch_stages_declare_a_template_per_profile(self):
        """Las etapas `launch` deben traer plantilla para el perfil que se use."""
        for stage in reg.STAGES:
            if stage.exec_kind != "launch":
                continue
            self.assertTrue(stage.launch, f"{stage.id}: exec_kind=launch sin plantillas")
            if stage.qc_schema_variant:
                for profile in stage.qc_schema_variant:
                    self.assertTrue(
                        profile in stage.launch or "*" in stage.launch,
                        f"{stage.id}: perfil {profile!r} sin plantilla de lanzamiento",
                    )

    def test_launch_sequences_are_non_empty_tuples_of_strings(self):
        """`launch` es una SECUENCIA: una etapa puede necesitar varios comandos."""
        for stage in reg.STAGES:
            for profile, commands in stage.launch.items():
                self.assertIsInstance(commands, tuple, f"{stage.id}/{profile}")
                self.assertTrue(commands, f"{stage.id}/{profile}: secuencia vacía")
                for cmd in commands:
                    self.assertIsInstance(cmd, str, f"{stage.id}/{profile}")
                    self.assertTrue(cmd.strip(), f"{stage.id}/{profile}: comando vacío")

    #: Subcomandos de A4 que NO van en la secuencia publicada, y por qué. Vive
    #: aquí y no en el registro porque es una afirmación sobre lo que se audita,
    #: no sobre cómo se ejecuta la etapa.
    A4_FUERA_DE_SECUENCIA = {
        "check-cube": (
            "paso de arranque: reescribe el documento ENTERO y desde 2026-08-24 se "
            "niega a borrar métricas medidas sin --force, así que en la secuencia "
            "haría abortar toda re-medida en el primer paso"
        ),
    }

    def test_every_a4_subcommand_that_writes_the_qc_is_published(self):
        """El fallo que costó un mes de M4/M5 ausentes, convertido en test.

        A4 son cinco subcomandos y el registro publicaba uno (`m3-flux`), así que
        lanzar la etapa desde su notebook no medía M4 ni M5 — y C2, al leer
        `m5_stat.status = unavailable`, se quedaba con la STAT nativa y un
        `stat_factor_box3` de 1.0, con el error declarado a la mitad del real.

        Los subcomandos se sacan por introspección del `argparse` real, no de una
        lista a mano: añadir mañana un `m6-loquesea` y no publicarlo pone esto en
        rojo sin que nadie tenga que acordarse de actualizar el test.
        """
        from musepipe.qc import cube_qc

        parser = cube_qc.build_parser() if hasattr(cube_qc, "build_parser") else None
        self.assertIsNotNone(
            parser, "cube_qc no expone build_parser(): la introspección no es posible"
        )
        subcomandos = set()
        for action in parser._subparsers._group_actions:  # noqa: SLF001 - argparse no da API
            subcomandos.update(action.choices)
        self.assertTrue(subcomandos, "no se encontró ningún subcomando en cube_qc")

        publicados = " ".join(reg.by_id("A4").launch["*"])
        for nombre in sorted(subcomandos - set(self.A4_FUERA_DE_SECUENCIA)):
            self.assertIn(
                f"cube_qc {nombre} ", publicados + " ",
                f"A4 no publica el subcomando {nombre!r}: lanzarla desde el notebook "
                f"no lo ejecutaría. Si es deliberado, decláralo en "
                f"A4_FUERA_DE_SECUENCIA con su razón.",
            )

    def test_a1_declares_both_reduction_profiles(self):
        """A1 emite QC distinto según la vía de reducción (problema P2)."""
        a1 = reg.by_id("A1")
        self.assertIn("cube_telcorr_qc.json", a1.qc_aliases)
        self.assertEqual(set(a1.qc_schema_variant), {"monolithic", "cascade"})

    def test_every_stage_is_parameterized(self):
        """Ninguna etapa puede estar atada a un objeto (exigencia del Track E).

        Si esto falla, alguien añadió una etapa con rutas fijas: la cadena deja
        de poder ejecutarse para un objeto nuevo sin tocar código.
        """
        blocked = sorted(s.id for s in reg.STAGES if not s.parameterized)
        self.assertEqual(blocked, [], "hay etapas atadas a un objeto: el Track E las cerró todas")


class BuilderAgreementTests(unittest.TestCase):
    """El generador de notebooks no puede divergir del registro."""

    @classmethod
    def setUpClass(cls):
        cls.builder = _load(
            "build_review_notebooks", ROOT / "scripts" / "build_review_notebooks.py"
        )

    def test_builder_validation_passes(self):
        # Lanza SystemExit con el detalle si STAGES diverge del registro.
        self.builder.validate_against_registry()

    def test_same_stage_set(self):
        self.assertEqual(
            sorted(s["id"] for s in self.builder.STAGES), sorted(reg.stage_ids())
        )


class NbcommonResolutionTests(unittest.TestCase):
    """`_nbcommon` resuelve por cadena y detecta contaminación cross-object."""

    @classmethod
    def setUpClass(cls):
        cls.nb = _load("_nbcommon", ROOT / "notebooks" / "_nbcommon.py")

    def test_cross_object_detection(self):
        same_object = [
            ("ROXs12b_raw", "ROXs12b_realigned"),
            ("ROXs12b_B_adp", "ROXs12b_realigned"),  # target_name='ROX12b': no falso positivo
            ("ROXs42Bb_raw", "ROXs42Bb_realigned"),
        ]
        for candidate, active in same_object:
            self.assertFalse(
                self.nb._is_cross_object(candidate, active),
                f"falso positivo: {candidate} vs {active}",
            )
        for candidate, active in [("ROXs12b_raw", "ROXs42Bb_realigned"),
                                  ("ROXs42Bb_realigned", "ROXs12b_realigned")]:
            self.assertTrue(
                self.nb._is_cross_object(candidate, active),
                f"no detectó cross-object: {candidate} vs {active}",
            )

    def test_effective_run_does_not_mutate_active_run(self):
        """Un override explícito no puede convertirse en el run activo.

        Si lo hiciera, el aviso cross-object nunca dispararía (el candidato
        pasaría a ser el activo y siempre coincidirían).
        """
        self.nb.resolve_run_id("ROXs42Bb_realigned")
        self.nb._effective_run("ROXs12b_raw")
        self.assertEqual(self.nb._ACTIVE_RUN, "ROXs42Bb_realigned")

    def test_none_means_active_run_not_default(self):
        self.nb.resolve_run_id("ROXs42Bb_realigned")
        self.assertEqual(self.nb._effective_run(None), "ROXs42Bb_realigned")
        self.assertNotEqual(self.nb._effective_run(None), self.nb.DEFAULT_RUN_ID)

    def test_requested_path_wins_over_alias(self):
        """Pedir una ruta concreta no puede devolver otra variante de esquema."""
        candidates = self.nb._candidates(
            "cube_telcorr_qc.json", "ROXs42Bb_realigned"
        )
        first_paths = [p for _run, p, _why in candidates]
        self.assertEqual(
            first_paths[0], "cube_telcorr_qc.json",
            "la ruta pedida debe probarse antes que cualquier alias",
        )
        # todos los candidatos de la ruta pedida, antes del primer alias
        idx_alias = next(i for i, p in enumerate(first_paths) if p != "cube_telcorr_qc.json")
        self.assertTrue(all(p == "cube_telcorr_qc.json" for p in first_paths[:idx_alias]))


if __name__ == "__main__":
    unittest.main()
