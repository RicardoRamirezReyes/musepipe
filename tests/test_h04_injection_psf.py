"""La PSF con la que E4 INYECTA, separable de la que extrae.

Por qué existe el knob, medido el 2026-08-17 en ROXs 12 b: al cambiar C1 del
ajuste analítico a la mezcla por observación, **los seis métodos perdieron ~6 %
de throughput a la vez** (−5.7 a −6.6 %), sin rastro de la sensibilidad a la
FORMA, que habría movido la apertura 3.5× más que psffit. La causa está en
`injection.normalized_spatial_psf`, que normaliza la fuente inyectada por la
suma sobre **todo el frame** mientras la extracción lee el modelo en convención
NORMRAD: un modelo con más alas (la mezcla pone +9.0 % más luz fuera de 25 px)
inyecta menos núcleo para el mismo `injected_flux` nominal.

O sea que con la misma PSF en los dos lados el throughput **no es comparable
entre dos modelos**. Lo que se fija aquí:

* que el default no cambie nada (inyectar con la que extrae);
* que el knob llegue de verdad a la inyección;
* y que fijándolo, dos modelos de extracción distintos reciban el MISMO cubo
  inyectado — que es lo único que hace comparables sus throughputs.
"""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from musepipe.stages.stage_h01_detect import HALPHA_REST_A, matched_filter_point
from musepipe.stages.stage_h04_injection import (
    compute_stage_h04_products,
    stage_h04_paths,
)
from tests.test_optimal_analytic import constant_model_doc
from tests.test_h04_process_pool import _synthetic_config, _synthetic_extractor


def _matched_sigma(wave, fwhm):
    err = np.full(wave.size, 0.2, dtype=np.float64)
    _flux, sigma, _z = matched_filter_point(
        wave, np.zeros(wave.size, dtype=np.float64), err, HALPHA_REST_A, fwhm,
        np.ones(wave.size, dtype=bool),
    )
    return sigma


def _grid(extraction_fwhm=4.0, injection_model=None, norm_convention=None):
    """Corre la rejilla sintética y devuelve (filas, QC)."""

    wave = np.arange(6525.0, 6601.0, 1.0, dtype=np.float64)
    fwhm = 2.5
    cube = np.zeros((wave.size, 64, 64), dtype=np.float64)
    yy, xx = np.mgrid[0:64, 0:64]
    cube += (0.001 * (yy + xx)).astype(np.float64)[None, :, :]
    overrides = {}
    if injection_model is not None:
        overrides["h04_injection_psf_model"] = injection_model
    if norm_convention is not None:
        overrides["h04_injection_norm_convention"] = norm_convention
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        paths = stage_h04_paths("synthetic_h04_psf", project_root=root)
        paths["paths"].ensure_base_dirs()
        cfg = _synthetic_config("synthetic_h04_psf", root, _matched_sigma(wave, fwhm), **overrides)
        product = compute_stage_h04_products(
            cfg, paths,
            extractors={m: _synthetic_extractor for m in cfg["h04_methods"]},
            base_cube=cube, wavelengths_A=wave,
            psf_model=constant_model_doc(fwhm=extraction_fwhm),
        )
    return product.rows, product.qc


def _recovered(rows):
    return [r.get("recovered_flux") for r in rows]


class InjectionPsfTests(unittest.TestCase):
    def test_the_default_injects_with_the_extraction_psf(self):
        _rows, qc = _grid()
        prov = qc["psf_provenance"]

        self.assertFalse(prov["injection_psf_differs"])
        self.assertEqual(prov["injection_psf_form"], prov["extraction_psf_form"])

    def test_the_knob_reaches_the_injection(self):
        """Si no llegase, cambiar la PSF de inyección no movería nada."""

        base, _ = _grid()
        wide, qc = _grid(injection_model=constant_model_doc(fwhm=9.0))

        self.assertTrue(qc["psf_provenance"]["injection_psf_differs"])
        self.assertNotEqual(_recovered(base), _recovered(wide),
                            "la PSF de inyección no está llegando a `inject`")

    def test_pinning_makes_two_extraction_models_comparable(self):
        """Lo que el knob existe para permitir.

        El extractor sintético ignora el modelo, así que con la inyección fijada
        dos modelos de extracción distintos tienen que recibir EXACTAMENTE el
        mismo cubo y devolver las mismas filas. Sin fijarla no lo hacen, y esa
        diferencia es justo el −6 % espurio que se midió en ROXs 12 b.
        """

        pinned = constant_model_doc(fwhm=5.0)
        a, _ = _grid(extraction_fwhm=4.0, injection_model=pinned)
        b, _ = _grid(extraction_fwhm=8.0, injection_model=pinned)
        self.assertEqual(_recovered(a), _recovered(b))

        suelto_a, _ = _grid(extraction_fwhm=4.0)
        suelto_b, _ = _grid(extraction_fwhm=8.0)
        self.assertNotEqual(_recovered(suelto_a), _recovered(suelto_b))


class NormConventionTests(unittest.TestCase):
    """`injected_flux` en la misma convencion que `recovered_flux`.

    Con `frame` (el camino historico) la inyeccion fijaba el flujo sobre todo el
    recorte mientras la extraccion entrega NORMRAD, asi que el puente entre las
    dos dependia del MODELO: en ROXs 12 b cambiar el modelo de PSF de C1 movio el
    throughput de los seis metodos -6 % a la vez. Con `norm_radius` los dos lados
    hablan de lo mismo.
    """

    def test_the_default_is_the_normrad_convention(self):
        _rows, qc = _grid()
        self.assertEqual(qc["psf_provenance"]["injection_norm_convention"], "norm_radius")

    def test_the_psf_perturbation_stops_being_blind(self):
        """+-10 % de FWHM tiene que MOVER algo; con `frame` no movia nada."""

        _rows, qc_norm = _grid()
        _rows_f, qc_frame = _grid(norm_convention="frame")

        # `frame` la deja en cero salvo redondeo (3.99e-13 en este fixture):
        # ensanchar la PSF no cambia el total sobre el recorte.
        self.assertAlmostEqual(qc_frame["throughput"]["psf_perturbation_pct"], 0.0, places=9)
        self.assertGreater(qc_norm["throughput"]["psf_perturbation_pct"], 1e-6)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
