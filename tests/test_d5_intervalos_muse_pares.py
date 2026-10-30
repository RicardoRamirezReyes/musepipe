"""El Δ del D.5 sin intervalos del espectro MUSE (`scripts/d5_intervalos_muse.py`, 2026-10-30).

Las clases sintéticas no leen disco: qué cadena es una variante de `--quita-muse` sola, cómo se empareja con la del
paper (por su json, no por la etiqueta) y la lectura del Δ combinado que imprime el D.5. `DatosRealesTests` (marca
`external_data`): las cadenas del paper, sin ninguna variante, dan el D.5 de `paper/main.tex`, y una cifra cambiada
en el `.tex` se detecta.
"""
import re
import unittest

import pytest

from tests._datos import cadenas_teff, carga_script, raiz_proyecto, raiz_temporal, requiere

#: el D.5 con la forma del paper (dos compañeros; el segundo, «The combined value is»)
TEX = r"""
\subsection{Δ between grids}
\label{app:checks:grids}
For ROXs~12~b ... The combined \(\Delta\) is
\(-0.08^{+0.06}_{-0.06}\), \(+0.16^{+0.12}_{-0.05}\) and \(+0.28^{+0.12}_{-0.07}\)~dex, and the
fraction of the combined samples above the noise level is \(0.06\), \(1.00\) and \(1.00\). For ROXs~42B~b ...
The combined value is \(+0.03^{+0.07}_{-0.12}\), \(+0.16^{+0.09}_{-0.56}\) and
\(+0.10^{+0.06}_{-0.19}\)~dex, and the limit lies above the noise level in \(0.60\), \(0.80\)
and \(0.79\) of the combined samples. The combined value is below the noise level at \(A_V = 1.8\)~mag.
\subsection{Next}
The combined \(\Delta\) is \(+9.99^{+0}_{-0}\).
"""


def _json(**kw):
    d = dict(etiqueta="x", slug="ROXs12b", modo="libre", av="primaria", av_fija=None, rejilla="drift", datos="todo",
             r_analisis=500, nlive=1000, semilla=1)
    d.update(kw)
    return d


class _V:
    def __init__(self, *ds):
        self.por_etiqueta = {f"e{i}": dict(d, _json=f"e{i}.json") for i, d in enumerate(ds)}


class ParesTests(unittest.TestCase):
    def setUp(self):
        self.m = carga_script("d5_intervalos_muse")

    def test_solo_quita_muse_es_variante(self):
        q = self.m.quitado
        self.assertEqual(q(_json(quita_muse="azul")), "azul")
        self.assertIsNone(q(_json()))
        self.assertIsNone(q(_json(quita_muse="azul", rk_fijo=0.0)))
        self.assertIsNone(q(_json(quita_muse="azul", mascara_muse="sin_o2b")))
        self.assertIsNone(q(_json(quita_muse="rojo", verdad=dict(plantilla="LM601"))))

    def test_la_clave_usa_los_valores_por_defecto_del_json(self):
        con = _json(rejilla="btsettl")
        sin = {k: v for k, v in con.items() if k not in ("rejilla", "r_analisis", "datos")}
        self.assertEqual(self.m.clave(con), self.m.clave(sin))
        self.assertNotEqual(self.m.clave(con), self.m.clave(_json(rejilla="btsettl", nlive=50)))

    def test_indice_empareja_por_json_y_rechaza_repetidas(self):
        a, r = _json(quita_muse="azul"), _json(quita_muse="rojo")
        idx = self.m.indice_variantes(_V(a, r, _json(), _json(quita_muse="azul", td_rango=[300, 3000])))
        self.assertEqual(sorted(q for _, q in idx), ["azul", "rojo"])
        self.assertEqual(idx[(self.m.clave(_json()), "azul")]["_json"], "e0.json")
        with self.assertRaises(SystemExit):
            self.m.indice_variantes(_V(a, dict(a, etiqueta="y")))


class TexTests(unittest.TestCase):
    def setUp(self):
        self.m = carga_script("d5_intervalos_muse")

    def test_lee_las_dos_frases_de_su_seccion(self):
        d = self.m.d5_del_paper(TEX, ("ROXs12b", "ROXs42Bb"))
        self.assertEqual(d["ROXs12b"], {"1.8": ("-0.08", "0.06"), "3": ("+0.16", "1.00"), "prior": ("+0.28", "1.00")})
        self.assertEqual(d["ROXs42Bb"], {"1.8": ("+0.03", "0.60"), "3": ("+0.16", "0.80"), "prior": ("+0.10", "0.79")})

    def test_falla_si_no_estan_las_frases(self):
        with self.assertRaises(ValueError):
            self.m.d5_del_paper(TEX, ("ROXs12b",))
        with self.assertRaises(ValueError):
            self.m.d5_del_paper(TEX.replace("above the noise level is", "above it is"), ("ROXs12b", "ROXs42Bb"))


@pytest.mark.external_data
@requiere("runs", "cadenas")
class DatosRealesTests(unittest.TestCase):
    """Las cadenas del paper contra el D.5 de `paper/main.tex` del clon con los runs."""

    @classmethod
    def setUpClass(cls):
        cls.tex = raiz_proyecto() / "paper" / "main.tex"
        if not cls.tex.exists():
            raise unittest.SkipTest(f"sin {cls.tex}")
        cls.m = carga_script("d5_intervalos_muse")
        cls.tc = carga_script("teff_cifras_texto")

    def _calcula(self, tex):
        with raiz_temporal() as raiz:
            return self.m.calcula(self.tc.Cadenas(cadenas_teff()), self.tc.Cadenas([]), raiz, tex=tex)

    def test_las_cadenas_del_paper_dan_el_d5(self):
        res = self._calcula(self.tex.read_text())
        self.assertTrue(res["tex"]["iguales"], res["tex"]["distintos"])
        self.assertEqual(len(res["casos"]), 6)
        self.assertTrue(all(not c["variantes"] and not c["incompletas"] for c in res["casos"]))

    def test_una_cifra_cambiada_se_detecta(self):
        tex = self.tex.read_text()
        i = tex.index(r"\label{app:checks:grids}")
        cambiado = tex[:i] + re.sub(r"is \\\(0\.06\\\)", r"is \\(0.07\\)", tex[i:], count=1)
        self.assertNotEqual(cambiado, tex)
        res = self._calcula(cambiado)
        self.assertFalse(res["tex"]["iguales"])
        self.assertEqual([(x["slug"], x["av"]) for x in res["tex"]["distintos"]], [("ROXs12b", "1.8")])


if __name__ == "__main__":
    unittest.main()
