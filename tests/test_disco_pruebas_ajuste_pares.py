"""Las pruebas del disco en los ajustes (`scripts/disco_pruebas_ajuste.py`, apéndice D.11; 2026-10-30).

Las clases sintéticas no leen disco: cómo se clasifica una cadena por su json (ajuste, objeto, configuración), qué
cadenas entran, cómo se empareja cada prueba con su `c0`, el Δln Z y su σ, los bordes y el resumen de las plantillas.
`DatosRealesTests` (marca `external_data`) lee las cadenas del paper y las de las pruebas (`MUSE_DISCO_CADENAS`,
carpetas separadas por `:`; por defecto las de los lotes del 2026-10-29 y 10-30) y comprueba invariantes, no cifras.
"""
import math
import os
import unittest
from pathlib import Path

import pytest

from tests._datos import cadenas_teff, carga_script, raiz_temporal, requiere

#: las carpetas de las cadenas de las pruebas del disco en la máquina del autor (lotes del 2026-10-29 y 10-30)
DISCO_POR_DEFECTO = ("~/muse_noche_20261029_disco/lote/res", "~/muse_noche_20261030/lote/res",
                     "~/muse_noche_20261030/lote_pl3000/res")


def _q(p50, p16=None, p84=None):
    return dict(p16=p50 - 10 if p16 is None else p16, p50=p50, p84=p50 + 10 if p84 is None else p84)


def _json(etiqueta="x", logz=100.0, logz_err=0.3, teff=2700.0, **kw):
    d = dict(etiqueta=etiqueta, slug="ROXs12b", modo="libre", av="primaria", av_fija=None, rejilla="drift", datos="todo",
             r_analisis=500, nlive=250, semilla=0, logz=logz, logz_err=logz_err, ejes=dict(teff=[2000.0, 100.0, 11]),
             posterior=dict(der_teff=_q(teff), av=_q(3.0, 2.9, 3.1), rk=_q(0.5, 0.4, 0.6)), _json=f"{etiqueta}.json")
    d.update(kw)
    return d


class _C:
    def __init__(self, *ds):
        self.por_etiqueta = {d["etiqueta"]: d for d in ds}


class ClasificaTests(unittest.TestCase):
    def setUp(self):
        self.m = carga_script("disco_pruebas_ajuste")

    def test_ajuste(self):
        a = self.m.ajuste
        self.assertEqual(a(_json()), "c0")
        self.assertEqual(a(_json(rk_fijo=0.0)), "rk0")
        self.assertEqual(a(_json(td_rango=[300.0, 1500.0])), "td300-1500")
        self.assertEqual(a(_json(td_rango=[300.0, 3000.0])), "td300-3000")

    def test_objeto(self):
        o = self.m.objeto
        self.assertEqual(o(_json()), ("datos",))
        self.assertEqual(o(_json(verdad=dict(plantilla="LM601", spt="M7.5", lib="manara", av=1.8))), ("plantilla", "LM601"))
        v = dict(teff=2550.0, logg=3.75, av=1.8, r_rj=2.2)
        self.assertEqual(o(_json(verdad=v)), ("inyeccion", 2550.0, 3.75, 1.8, 2.2, None))
        self.assertEqual(o(_json(verdad=dict(v, rk=0.2, td=900.0), inyecta_polvo=[0.2, 900.0])),
                         ("inyeccion", 2550.0, 3.75, 1.8, 2.2, (0.2, 900.0)))

    def test_entra(self):
        e = self.m.entra
        self.assertTrue(e(_json(), del_paper=True))
        self.assertFalse(e(_json(rk_fijo=0.0), del_paper=True))          # una prueba no es una cadena del paper
        self.assertTrue(e(_json(rk_fijo=0.0), del_paper=False))
        self.assertFalse(e(_json(td_rango=[300.0, 1500.0], quita_muse="azul"), del_paper=False))
        self.assertFalse(e(_json(rk_fijo=0.0, mascara_muse="sin_o2b"), del_paper=False))
        self.assertFalse(e(_json(rk_fijo=0.0, opcion_nueva=1), del_paper=False))
        self.assertFalse(e(_json(modo="evo"), del_paper=False))

    def test_indice_rechaza_repetidas(self):
        a = _json("a", rk_fijo=0.0)
        idx = self.m.indice(_C(_json("p")), _C(a))
        self.assertEqual(sorted(k[2] for k in idx), ["c0", "rk0"])
        with self.assertRaises(SystemExit):
            self.m.indice(_C(), _C(a, dict(a, etiqueta="b", _json="b.json")))


class FilasTests(unittest.TestCase):
    def setUp(self):
        self.m = carga_script("disco_pruebas_ajuste")

    def test_borde(self):
        b = self.m.borde
        self.assertEqual(b(dict(p16=1450, p50=1490, p84=1499), 300, 1500, 24), "alto")
        self.assertEqual(b(dict(p16=0.0, p50=0.01, p84=0.03), 0.0, 1.5, 0.03), "bajo")
        self.assertIsNone(b(dict(p16=1800, p50=1900, p84=2000), 300, 3000, 54))
        self.assertIsNone(b(None, 0, 1, 0.1))

    def test_dlnz_y_su_sigma(self):
        ref, d = _json("p", logz=100.0, logz_err=0.3), _json("t", logz=93.5, logz_err=0.4, rk_fijo=0.0)
        f = self.m.fila(d, ref)
        self.assertAlmostEqual(f["dlnz"], -6.5)
        self.assertAlmostEqual(f["dlnz_err"], 0.5)
        g = self.m.fila(ref, ref)
        self.assertIsNone(g["dlnz"])
        self.assertIsNone(self.m.fila(d, None)["dlnz"])

    def test_bordes_de_la_fila(self):
        d = _json(teff=2995.0, td_rango=[300.0, 1500.0],
                  posterior=dict(der_teff=_q(2995.0, 2980.0, 2999.0), av=_q(3.0), rk=_q(1.48, 1.45, 1.50),
                                 td=_q(1490.0, 1470.0, 1499.0)))
        self.assertEqual(self.m.fila(d, None)["borde"], dict(teff="alto", rk="alto", td="alto"))
        d = _json(td_rango=[300.0, 3000.0], posterior=dict(der_teff=_q(2500.0), av=_q(3.0), rk=_q(0.7, 0.6, 0.8),
                                                            td=_q(1900.0, 1800.0, 2000.0)))
        self.assertEqual(self.m.fila(d, None)["borde"], {})

    def test_grupos_en_orden_y_sin_referencia(self):
        p = _json("p")
        pruebas = [_json("t3", td_rango=[300.0, 3000.0], posterior=dict(p["posterior"], td=_q(1900.0))),
                   _json("t1", td_rango=[300.0, 1500.0], posterior=dict(p["posterior"], td=_q(1400.0))),
                   _json("r", rk_fijo=0.0)]
        huerfana = _json("h", rejilla="btsettl", rk_fijo=0.0)
        solo_c0 = _json("s", semilla=5)
        gs = self.m.grupos(self.m.indice(_C(p, solo_c0), _C(*pruebas, huerfana)))
        self.assertEqual(len(gs), 2)
        drift = next(g for g in gs if g["config"]["rejilla"] == "drift")
        self.assertEqual([f["ajuste"] for f in drift["filas"]], ["c0", "rk0", "td300-1500", "td300-3000"])
        self.assertFalse(drift["sin_referencia"])
        self.assertTrue(next(g for g in gs if g["config"]["rejilla"] == "btsettl")["sin_referencia"])


class ResumenTests(unittest.TestCase):
    def setUp(self):
        self.m = carga_script("disco_pruebas_ajuste")

    def test_corte_por_tipo(self):
        ds = []
        for i, (spt, dz, rk) in enumerate((("M7", 2.0, 0.02), ("M6.5", 3.0, 0.08), ("M9", -30.0, 1.0),
                                           ("M8.5", -10.0, 0.5), ("L1", -60.0, 0.9))):
            v = dict(plantilla=f"P{i}", spt=spt, lib="x", av=1.8)
            ds += [_json(f"c{i}", logz=100.0, verdad=v, posterior=dict(der_teff=_q(2500.0), av=_q(2.0), rk=_q(rk))),
                   _json(f"r{i}", logz=100.0 + dz, verdad=v, rk_fijo=0.0,
                         posterior=dict(der_teff=_q(2500.0), av=_q(2.0)))]
        r = self.m.resumen_plantillas(self.m.grupos(self.m.indice(_C(*ds[0::2]), _C(*ds[1::2]))), corte="M8.5")
        self.assertEqual(r["drift|antes|rk0"]["spt"], ["M6.5", "M7"])
        self.assertEqual(r["drift|antes|rk0"]["dlnz"], [2.0, 3.0])
        self.assertEqual(r["drift|desde|rk0"]["spt"], ["M8.5", "M9", "L1"])
        self.assertEqual(r["drift|desde|rk0"]["dlnz"], [-60.0, -10.0])
        self.assertEqual(r["drift|desde|c0"]["rk"], [0.5, 1.0])


def _dirs_disco():
    v = os.environ.get("MUSE_DISCO_CADENAS")
    partes = [p for p in v.split(os.pathsep) if p] if v else list(DISCO_POR_DEFECTO)
    return [Path(p).expanduser() for p in partes]


@pytest.mark.external_data
@requiere("runs", "cadenas")
@unittest.skipUnless(any(d.is_dir() for d in _dirs_disco()),
                     "sin las cadenas de las pruebas del disco (define MUSE_DISCO_CADENAS)")
class DatosRealesTests(unittest.TestCase):
    """Las cadenas del paper y las de las pruebas del disco: invariantes de la clasificación y de las filas."""

    @classmethod
    def setUpClass(cls):
        cls.m = carga_script("disco_pruebas_ajuste")
        tc = carga_script("teff_cifras_texto")
        with raiz_temporal() as raiz:
            cls.res = cls.m.calcula(tc.Cadenas(cadenas_teff()), tc.Cadenas([d for d in _dirs_disco() if d.is_dir()]), raiz)

    def test_cada_prueba_tiene_su_c0(self):
        self.assertTrue(self.res["grupos"])
        self.assertEqual([g["objeto"] for g in self.res["grupos"] if g["sin_referencia"]], [])

    def test_las_filas_respetan_su_ajuste(self):
        for g in self.res["grupos"]:
            for f in g["filas"]:
                with self.subTest(etiqueta=f["etiqueta"]):
                    if f["ajuste"] == "c0":
                        self.assertIsNone(f["dlnz"])
                        self.assertIsNotNone(f["rk"])
                        self.assertIsNone(f["td"])
                    elif f["ajuste"].startswith("rk"):
                        self.assertIsNone(f["rk"])
                        self.assertIsNone(f["td"])
                    else:
                        lo, hi = f["td_rango"]
                        self.assertTrue(lo <= f["td"]["p16"] <= f["td"]["p84"] <= hi)
                    if f["dlnz"] is not None:
                        self.assertTrue(math.isfinite(f["dlnz"]) and f["dlnz_err"] > 0)
                    self.assertEqual(f["delta"] is not None, g["tipo"] == "datos")


if __name__ == "__main__":
    unittest.main()
