"""G4 §7: log-likelihood combination and correlated-test grouping."""

import unittest

from musepipe.classify import HYPOTHESES, combine_evidence


def cell(v):
    return {"verdict": v}


class G4CombinationTests(unittest.TestCase):
    def test_supports_ranks_above_disfavors(self):
        matrix = {
            "T1": {h: cell("supports" if h == "substellar_companion" else "disfavors") for h in HYPOTHESES},
        }
        _ll, rank = combine_evidence(matrix)
        self.assertEqual(rank[0]["hypothesis"], "substellar_companion")
        self.assertEqual(rank[0]["log_l_rel"], 0.0)

    def test_correlated_group_counts_once(self):
        # three correlated tests all 'supports' one hypothesis; grouped => weight 1, not 3
        matrix = {t: {h: cell("supports" if h == "brown_dwarf" else "neutral") for h in HYPOTHESES}
                  for t in ("T3", "T4", "T6")}
        ll_grouped, _ = combine_evidence(matrix, groups=[("T3", "T4", "T6")])
        ll_indep, _ = combine_evidence(matrix)
        self.assertEqual(ll_grouped["brown_dwarf"], 1.0)   # one joint contribution
        self.assertEqual(ll_indep["brown_dwarf"], 3.0)     # counted independently

    def test_excludes_crushes_hypothesis(self):
        matrix = {"T7": {h: cell("excludes" if h == "artifact" else "neutral") for h in HYPOTHESES}}
        ll, rank = combine_evidence(matrix)
        self.assertLess(ll["artifact"], -100.0)
        self.assertNotEqual(rank[0]["hypothesis"], "artifact")

    def test_not_available_does_not_score(self):
        matrix = {"T3": {h: cell("not_available") for h in HYPOTHESES}}
        ll, _ = combine_evidence(matrix)
        self.assertTrue(all(v == 0.0 for v in ll.values()))


if __name__ == "__main__":
    unittest.main()
