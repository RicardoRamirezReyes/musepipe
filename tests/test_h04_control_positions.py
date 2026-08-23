import math
import unittest

from musepipe.stages.stage_h04_injection import resolve_h04_positions


def cfg(n=None, **extra):
    out = {"h04_real_position_yx": [155.6, 76.0],
           "h04_control_positions_yx": [[76.0, 14.4], [14.4, 94.1], [94.0, 155.6]]}
    if n is not None:
        out["h04_n_control_positions"] = n
    out.update(extra)
    return out


class ControlPositionCountTests(unittest.TestCase):
    """Cuantas posiciones de control, y por que el numero manda sobre tres cosas.

    Es el `n` de `completeness_at_5sigma` —con 3 controles solo puede valer 0,
    0.25, 0.5, 0.75 o 1—, la poblacion nula de V2 y la potencia de su puerta.
    """

    def test_the_default_is_still_three(self):
        pos = resolve_h04_positions(cfg())

        self.assertEqual(len(pos), 4)
        self.assertEqual([p["label"] for p in pos][0], "real")

    def test_an_explicit_list_must_match_the_declared_count(self):
        # Declarar 11 y pasar 3 es un error de config, no algo que se recorte solo.
        with self.assertRaisesRegex(RuntimeError, "control positions"):
            resolve_h04_positions(cfg(n=11))

    def test_zero_controls_is_refused(self):
        with self.assertRaisesRegex(RuntimeError, "at least one control"):
            resolve_h04_positions(cfg(n=0))

    def test_more_controls_stay_on_the_same_radius_and_spread_evenly(self):
        star = (85.0, 85.0)
        companion = (155.6, 76.0)
        n = 11
        pos = resolve_h04_positions({"h04_n_control_positions": n},
                                    _paths(star, companion))

        self.assertEqual(len(pos), n + 1)
        radios = [math.hypot(p["y"] - star[0], p["x"] - star[1]) for p in pos]
        for r in radios:
            self.assertAlmostEqual(r, radios[0], places=6, msg="los controles cambian de radio")
        angulos = sorted(math.degrees(math.atan2(p["y"] - star[0], p["x"] - star[1])) % 360
                         for p in pos)
        pasos = [b - a for a, b in zip(angulos, angulos[1:])]
        for paso in pasos:
            self.assertAlmostEqual(paso, 360.0 / (n + 1), places=4)


def _paths(star, companion):
    import json
    import tempfile
    from pathlib import Path

    tmp = Path(tempfile.mkdtemp())
    qc = tmp / "stage01c_qc.json"
    qc.write_text(json.dumps({"primary": {"pos_yx": list(star)},
                              "companion": {"pos_yx": list(companion)}}))
    return {"stage01c_qc_json": qc}


if __name__ == "__main__":
    unittest.main()
