"""Regresiones de `scaled_psf_model` y del baseline de E4.

Las tres cosas que se prueban aqui estuvieron rotas a la vez y en silencio:
`scaled_psf_model` solo sabia escalar coeficientes Moffat, asi que con el
modelo `psfao` —el que C1 elige siempre que maoppy este disponible— devolvia el
documento intacto. Consecuencias medidas sobre `ROXs12b_realigned`:

* C3 publicaba `psf_sensitivity.fwhm_pm10pct_flux_bias_pct = 0.0`, que parecia
  robustez y era una perturbacion que no ocurrio.
* E4 sacaba las variantes `psf_minus10` y `psf_plus10` bit a bit identicas.
* y, como `_baseline_key` incluia `psf_fwhm_scale`, esas filas ni siquiera
  encontraban su baseline: `psf_perturbation_pct` acababa midiendo el pedestal
  sin restar (22.3%) y E3 lo metia como sistematico del limite de Mdot.
"""

import unittest

import numpy as np

from musepipe.psf import evaluate_psf_model, scaled_psf_model
from musepipe.stages.stage_h04_injection import _apply_baseline_subtraction


def _moffat_doc(fwhm=4.0):
    coef = {"coefficients": [float(fwhm)], "kind": "poly"}
    return {
        "form": "moffat",
        "norm_radius_px": 12.0,
        "coefficients": {
            "fwhm_maj": dict(coef),
            "fwhm_min": dict(coef),
            "theta_deg": {"coefficients": [0.0], "kind": "poly"},
            "beta": {"coefficients": [2.5], "kind": "poly"},
            "y0": {"coefficients": [0.0], "kind": "poly"},
            "x0": {"coefficients": [0.0], "kind": "poly"},
        },
    }


def _psfao_doc():
    """Documento psfao minimo, con la forma que escribe C1."""

    names = ["r0", "C", "A", "alpha", "ratio", "theta", "beta"]
    return {
        "form": "psfao",
        "system": "muse_nfm",
        "param_names": names,
        "norm_radius_px": 12.0,
        "smoothed_poly": {
            "r0": [0.12], "C": [1e-6], "A": [10.0], "alpha": [0.5],
            "ratio": [1.0], "theta": [0.0], "beta": [1.6],
        },
    }


def _half_light_radius(psf, dy, dx):
    r = np.hypot(dy, dx).ravel()
    order = np.argsort(r)
    cum = np.cumsum(psf.ravel()[order])
    return float(np.interp(0.5 * cum[-1], cum, r[order]))


class ScaledPsfModelTests(unittest.TestCase):
    def test_moffat_scaling_multiplies_the_fwhm_coefficients(self):
        doc = _moffat_doc(4.0)
        scaled = scaled_psf_model(doc, 1.25)
        for key in ("fwhm_maj", "fwhm_min"):
            self.assertAlmostEqual(scaled["coefficients"][key]["coefficients"][0], 5.0)
        # El original no se toca.
        self.assertAlmostEqual(doc["coefficients"]["fwhm_maj"]["coefficients"][0], 4.0)

    def test_scale_one_is_a_no_op(self):
        doc = _moffat_doc()
        self.assertIs(scaled_psf_model(doc, 1.0), doc)

    def test_unknown_form_raises_instead_of_returning_it_unchanged(self):
        with self.assertRaises(ValueError):
            scaled_psf_model({"form": "gaussian"}, 1.1)

    def test_moffat_without_fwhm_coefficients_raises(self):
        with self.assertRaises(ValueError):
            scaled_psf_model({"form": "moffat", "coefficients": {}}, 1.1)

    def test_non_positive_scale_raises(self):
        for bad in (0.0, -1.0, float("nan")):
            with self.assertRaises(ValueError):
                scaled_psf_model(_moffat_doc(), bad)

    def test_psfao_scaling_actually_widens_the_psf(self):
        """El fallo original: devolvia el doc intacto y la PSF no se movia."""

        try:
            import maoppy  # noqa: F401
        except Exception:  # pragma: no cover - entorno sin maoppy
            self.skipTest("maoppy no disponible")

        doc = _psfao_doc()
        n = 121
        yy, xx = np.indices((n, n), dtype=float)
        dy, dx = yy - n // 2, xx - n // 2
        base = evaluate_psf_model(doc, 6562.8, dy, dx)
        r50_base = _half_light_radius(base, dy, dx)

        for scale in (0.9, 1.1, 1.3):
            widened = evaluate_psf_model(scaled_psf_model(doc, scale), 6562.8, dy, dx)
            self.assertFalse(
                np.allclose(base, widened),
                f"scale={scale} dejo la PSF psfao sin tocar",
            )
            ratio = _half_light_radius(widened, dy, dx) / r50_base
            self.assertAlmostEqual(ratio, scale, delta=0.05)

    def test_psfao_scaling_is_a_dilation(self):
        """P_s(r) == P_1(r/s)/s**2, que es lo que hace falta para que el
        ancho cambie sin inventar flujo."""

        try:
            import maoppy  # noqa: F401
        except Exception:  # pragma: no cover
            self.skipTest("maoppy no disponible")

        doc = _psfao_doc()
        n = 81
        yy, xx = np.indices((n, n), dtype=float)
        dy, dx = yy - n // 2, xx - n // 2
        s = 1.1
        got = evaluate_psf_model(scaled_psf_model(doc, s), 6562.8, dy, dx)
        want = evaluate_psf_model(doc, 6562.8, dy / s, dx / s) / s ** 2
        np.testing.assert_allclose(got, want, rtol=1e-10, atol=0.0)


def _row(method="psffit", *, injected, recovered, scale=1.0, position="real",
         continuum="none", template=1.0):
    return {
        "method": method,
        "position_label": position,
        "continuum_mode": continuum,
        "template_factor": template,
        "psf_fwhm_scale": scale,
        "injected_flux": injected,
        "recovered_flux": recovered,
    }


class H04BaselineTests(unittest.TestCase):
    def test_psf_variants_share_the_nominal_baseline(self):
        """El pedestal se mide SIN fuente inyectada, asi que no depende del
        ancho de PSF de la fuente: las filas psf_+-10% tienen que restar el
        mismo baseline que la nominal, no un 0.0 por defecto."""

        rows = [
            _row(injected=0.0, recovered=2558.4),
            _row(injected=12777.4, recovered=11069.8),
            _row(injected=12777.4, recovered=11069.8, scale=0.9),
            _row(injected=12777.4, recovered=11069.8, scale=1.1),
        ]
        out = _apply_baseline_subtraction(rows)
        bases = [row["recovered_flux_baseline"] for row in out]
        self.assertEqual(bases, [2558.4] * 4)
        thr = [row["throughput"] for row in out[1:]]
        self.assertTrue(np.allclose(thr, thr[0]))

    def test_a_missing_baseline_raises_instead_of_defaulting_to_zero(self):
        rows = [_row(injected=12777.4, recovered=11069.8)]
        with self.assertRaises(RuntimeError) as ctx:
            _apply_baseline_subtraction(rows)
        self.assertIn("baseline", str(ctx.exception).lower())


if __name__ == "__main__":
    unittest.main()
