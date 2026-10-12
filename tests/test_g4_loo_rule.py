"""G4 §7/§4.4: if the leader changes under leave-one-out, class is forced ambiguous."""

import unittest

from musepipe.classify import HYPOTHESES, classify, leave_one_out_leader, tied_at_top

THRESH = {"secure_bg_prob_max": 0.01, "probable_bg_prob_max": 0.10, "secure_min_supports": 3}


def cell(v):
    return {"verdict": v}


def only(hyp, verdict="supports", rest="neutral"):
    return {h: cell(verdict if h == hyp else rest) for h in HYPOTHESES}


class G4LeaveOneOutTests(unittest.TestCase):
    def test_stable_leader(self):
        matrix = {"T1": only("substellar_companion"), "T2": only("substellar_companion"),
                  "T7": only("substellar_companion")}
        leader, loo = leave_one_out_leader(matrix)
        self.assertEqual(leader, "substellar_companion")
        self.assertTrue(all(v == "substellar_companion" for v in loo.values()))

    def test_leader_change_forces_ambiguous(self):
        # T1 favours A, T2 favours B; dropping either flips the leader
        matrix = {"T1": only("substellar_companion"), "T2": only("m_star_background")}
        out = classify(matrix, background_prob=0.001, thresholds=THRESH)
        self.assertFalse(out["leave_one_out_stable"])
        self.assertEqual(out["robustness"], "ambiguous")

    def test_secure_requires_three_independent_supports(self):
        matrix = {"T1": only("substellar_companion"), "T2": only("substellar_companion"),
                  "T7": only("substellar_companion")}
        out = classify(matrix, background_prob=0.001, thresholds=THRESH)
        self.assertEqual(out["label"], "substellar_companion")
        self.assertEqual(out["robustness"], "secure")

    def test_two_supports_is_probable_not_secure(self):
        matrix = {"T1": only("substellar_companion"), "T2": only("substellar_companion")}
        out = classify(matrix, background_prob=0.001, thresholds=THRESH)
        self.assertEqual(out["robustness"], "probable")

    def test_high_background_downgrades(self):
        matrix = {"T1": only("m_star_background"), "T2": only("m_star_background"),
                  "T7": only("m_star_background")}
        out = classify(matrix, background_prob=0.05, thresholds=THRESH)  # 5% -> not secure
        self.assertEqual(out["robustness"], "probable")


BOUND = ("planet_forming", "substellar_companion", "brown_dwarf", "m_star_associated")


def bound_row():
    return {h: cell("supports" if h in BOUND else "excludes") for h in HYPOTHESES}


class G4TieTests(unittest.TestCase):
    """2026-10-10: un empate en cabeza no tiene lider (spec G4 §4.5 y §8)."""

    def test_tie_of_four_bound_is_not_secure(self):
        # el caso de ROXs 12 b sin masa: las cuatro ligadas con 3 apoyos, fondo excluido, P(bg) 2e-4
        matrix = {"T1": bound_row(), "T2": bound_row(), "T3": bound_row()}
        out = classify(matrix, background_prob=2e-4, thresholds=THRESH)
        self.assertEqual(out["tied_at_top"], list(BOUND))
        self.assertIsNone(out["label"])
        self.assertTrue(out["leave_one_out_stable"])  # el empate no cambia al quitar una prueba...
        self.assertEqual(out["robustness"], "ambiguous")  # ...pero no hay lider que asegurar
        self.assertEqual(out["n_independent_supports"], 3)

    def test_unique_leader_keeps_label_and_tied_singleton(self):
        matrix = {"T1": only("substellar_companion"), "T2": only("substellar_companion"),
                  "T7": only("substellar_companion")}
        out = classify(matrix, background_prob=0.001, thresholds=THRESH)
        self.assertEqual(out["tied_at_top"], ["substellar_companion"])
        self.assertEqual(out["label"], "substellar_companion")

    def test_omission_that_opens_a_tie_is_unstable(self):
        # lider unico por una sola prueba: quitarla deja un empate, y el orden de la
        # tupla lo rompia a favor del mismo lider (planet_forming va primero)
        matrix = {"T1": bound_row(), "T8": only("planet_forming")}
        out = classify(matrix, background_prob=0.001, thresholds=THRESH)
        self.assertEqual(out["label"], "planet_forming")
        self.assertFalse(out["leave_one_out_stable"])
        self.assertEqual(out["robustness"], "ambiguous")

    def test_margin_widens_the_tie(self):
        matrix = {"T1": bound_row(), "T8": only("planet_forming")}
        self.assertEqual(tied_at_top(matrix), ("planet_forming",))
        self.assertEqual(tied_at_top(matrix, margin=1.5), BOUND)


if __name__ == "__main__":
    unittest.main()
