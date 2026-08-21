import unittest

from musepipe.stages.stage_g3_assemble import G3_REAL_KEYS, g3_entry_point


def real_config(**extra):
    cfg = {key: "x" for key in G3_REAL_KEYS}
    cfg.update(extra)
    return cfg


class G3EntryPointTests(unittest.TestCase):
    """Cual de las dos G3 le toca a un run, y por que no puede decidirlo el conductor."""

    def test_a_run_without_the_real_g3_config_gets_the_accretion_slice(self):
        self.assertEqual(g3_entry_point({"h01_lsf_fwhm_A": 2.4}), "accretion_slice")

    def test_a_run_that_declares_the_libraries_gets_the_real_g3(self):
        self.assertEqual(g3_entry_point(real_config()), "all")

    def test_a_half_declared_config_is_an_error_not_a_downgrade(self):
        # Degradar en silencio es exactamente el fallo: `rc=0` y el run peor.
        for missing in G3_REAL_KEYS:
            cfg = {k: "x" for k in G3_REAL_KEYS if k != missing}
            with self.subTest(falta=missing):
                with self.assertRaisesRegex(RuntimeError, "a medias"):
                    g3_entry_point(cfg)

    def test_the_two_objects_of_the_repo_resolve_to_different_entry_points(self):
        # ROXs 12 b no tiene bibliotecas y ROXs 42B b si: si esto deja de ser
        # cierto, o cambio un config o el criterio dejo de discriminar.
        self.assertEqual(g3_entry_point({}), "accretion_slice")
        self.assertEqual(g3_entry_point(real_config()), "all")


if __name__ == "__main__":
    unittest.main()
