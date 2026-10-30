"""Las opciones de `scripts/teff_bayes.py` para las pruebas del disco (apéndice D.11) y del D.5 (2026-10-29).

`--rk-fijo`, `--td-libre`, `--inyecta-polvo` y `--quita-muse`. Lo que se prueba son invariantes, no cifras del run:
sin las opciones el modelo es el de siempre bit a bit; r_K = 0 fija da la verosimilitud del modelo con r_K = 0;
T_d libre en `T_D` da el modelo de siempre; los parámetros nuevos solo aparecen si se piden; las combinaciones sin
sentido fallan antes de leer nada. Las clases sintéticas no leen disco (una rejilla lineal en log F, donde la
interpolación es exacta: el patrón de `tests/test_teff_bayes.py`, copiado aquí). Las de datos reales leen los runs
del paper y, las `slow`, las rejillas de /mnt/2TB.
"""
import argparse
import math
import unittest

import numpy as np
import pytest

from tests._datos import carga_script, modelo_teff, raiz_temporal, requiere, run_del_paper, runs_del_paper


def _sintetico(tb, rk_fijo=None, td_rango=None):
    """Un `Modelo` sin leer disco: MUSE (vis) y K, rejilla 3×2 con log F lineal en T_eff, log g y λ."""
    m = tb.Modelo.__new__(tb.Modelo)
    m.modo, m.rk_fijo, m.td_rango = "libre", rk_fijo, td_rango
    m.T0, m.dT, m.nT, m.G0, m.dG, m.nG = 2000.0, 100.0, 3, 3.0, 0.5, 2
    m.d_pc, m.d_err, m.av_fija, m.av_modo, m.av0, m.av_err = 140.0, 5.0, None, "primaria", 1.8, 0.3

    def logf(T, g, lam):
        return -2.0 + 1e-3 * T + 0.1 * g - 1e-5 * (lam - 20000.0)
    m.lref = np.array([[logf(m.T0 + a * m.dT, m.G0 + b * m.dG, tb.LAM_K) for b in range(m.nG)] for a in range(m.nT)])
    m.segs = []
    for nombre, banda, w, sigma in (("MUSE", "vis", np.linspace(6400.0, 9100.0, 30), 0.20),
                                    ("K", "K", np.linspace(20500.0, 23500.0, 40), 0.05)):
        lg = np.array([[logf(m.T0 + a * m.dT, m.G0 + b * m.dG, w) for b in range(m.nG)] for a in range(m.nT)])
        bb = tb._planck(w, tb.T_D) / tb._planck(tb.LAM_K, tb.T_D) if banda != "vis" else np.zeros(w.size)
        e = np.full(w.size, 1e-17)
        m.segs.append(dict(nombre=nombre, banda=banda, w=w, d=np.full(w.size, 1e-15), e2=e ** 2, lg=lg,
                           a=np.full(w.size, 0.12), bb=bb, sigma=sigma))
    m.bandas = ["K", "vis"]
    m._prepara_priores()
    return m, logf


def _theta(m, **valores):
    base = dict(teff=2130.0, logg=3.2, r_rj=2.0, d_pc=140.0, av=1.5, rk=0.3, td=1500.0, s_MUSE=1.05, s_K=1.1,
                lnf_K=math.log(0.05), lnf_vis=math.log(0.1))
    base.update(valores)
    return np.array([base[k] for k in m.nombres])


class ParametrosTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tb = carga_script("teff_bayes")

    def test_sin_opciones_los_parametros_de_siempre(self):
        m, _ = _sintetico(self.tb)
        self.assertEqual(m.nombres, ["teff", "logg", "r_rj", "d_pc", "av", "rk", "s_MUSE", "s_K", "lnf_K", "lnf_vis"])

    def test_rk_fijo_quita_rk(self):
        m, _ = _sintetico(self.tb, rk_fijo=0.0)
        self.assertNotIn("rk", m.nombres)
        self.assertNotIn("td", m.nombres)

    def test_td_libre_va_detras_de_rk_y_es_uniforme(self):
        m, _ = _sintetico(self.tb, td_rango=(300.0, 1500.0))
        i = m.nombres.index("rk")
        self.assertEqual(m.nombres[i + 1], "td")
        u = np.full(m.ndim, 0.5)
        for x, t in ((0.0, 300.0), (0.25, 600.0), (1.0, 1500.0)):
            u[i + 1] = x
            self.assertEqual(m.prior_transform(u)[i + 1], t)

    def test_un_modelo_de_new_sin_atributos_tiene_los_valores_de_clase(self):
        m = self.tb.Modelo.__new__(self.tb.Modelo)
        self.assertIsNone(m.rk_fijo)
        self.assertIsNone(m.td_rango)


class ModeloTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tb = carga_script("teff_bayes")

    def test_rk_fijo_cero_es_el_modelo_con_rk_cero_bit_a_bit(self):
        libre, _ = _sintetico(self.tb)
        fijo, _ = _sintetico(self.tb, rk_fijo=0.0)
        th_l, th_f = _theta(libre, rk=0.0), _theta(fijo)
        for a, b in zip(libre.modelos(th_l), fijo.modelos(th_f)):
            self.assertTrue(np.array_equal(a, b))
        self.assertEqual(libre.loglike(th_l), fijo.loglike(th_f))

    def test_rk_fijo_no_nulo_es_el_modelo_con_ese_rk(self):
        libre, _ = _sintetico(self.tb)
        fijo, _ = _sintetico(self.tb, rk_fijo=0.4)
        self.assertEqual(libre.loglike(_theta(libre, rk=0.4)), fijo.loglike(_theta(fijo)))

    def test_td_libre_en_t_d_es_el_modelo_de_siempre(self):
        base, _ = _sintetico(self.tb)
        td, _ = _sintetico(self.tb, td_rango=(300.0, 1500.0))
        for a, b in zip(base.modelos(_theta(base)), td.modelos(_theta(td, td=self.tb.T_D))):
            self.assertTrue(np.array_equal(a, b))

    def test_el_exceso_sigue_a_la_ec_b1(self):
        """F_K = s · Ω · [F(λ) + r_K · F(2.2 µm) · B_λ(T_d)/B_2.2(T_d)] · 10^(−0.4 A_λ A_V), a cualquier T_d; MUSE
        no lleva polvo."""
        from musepipe.constants import PC_CM, RJUP_CM
        m, logf = _sintetico(self.tb, td_rango=(300.0, 1500.0))
        om = (2.0 * RJUP_CM / (140.0 * PC_CM)) ** 2
        fref = 10 ** logf(2130.0, 3.2, self.tb.LAM_K)
        for td in (500.0, 900.0):
            vis, k = m.modelos(_theta(m, td=td))
            w = m.segs[1]["w"]
            bb = self.tb._planck(w, td) / self.tb._planck(self.tb.LAM_K, td)
            esperado = 1.1 * om * (10 ** logf(2130.0, 3.2, w) + 0.3 * fref * bb) * 10 ** (-0.4 * 0.12 * 1.5)
            np.testing.assert_allclose(k, esperado, rtol=1e-12)
            w = m.segs[0]["w"]
            np.testing.assert_allclose(vis, 1.05 * om * 10 ** logf(2130.0, 3.2, w) * 10 ** (-0.4 * 0.12 * 1.5),
                                       rtol=1e-12)

    def test_polvo_mas_frio_enrojece_mas_la_banda_k(self):
        m, _ = _sintetico(self.tb, td_rango=(300.0, 1500.0))
        exceso = {}
        for td in (600.0, 900.0, 1500.0):
            k = m.modelos(_theta(m, td=td))[1]
            k0 = m.modelos(_theta(m, td=td, rk=0.0))[1]
            exceso[td] = (k - k0)[-1] / (k - k0)[0]   # rojo / azul del exceso en K
        self.assertGreater(exceso[600.0], exceso[900.0])
        self.assertGreater(exceso[900.0], exceso[1500.0])


class ValidaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tb = carga_script("teff_bayes")

    def test_sin_opciones_no_falla(self):
        self.tb.valida_disco()
        self.tb.valida_disco("muse")
        self.tb.valida_disco("ir")

    def test_combinaciones_sin_sentido(self):
        malas = [dict(rk_fijo=0.0, td_rango=(300.0, 1500.0)), dict(datos="muse", rk_fijo=0.0),
                 dict(datos="muse", td_rango=(300.0, 1500.0)), dict(rk_fijo=-0.1), dict(rk_fijo=float("nan")),
                 dict(td_rango=(1500.0, 300.0)), dict(td_rango=(0.0, 1500.0)), dict(td_rango=(300.0,)),
                 dict(quita_muse="nada"), dict(datos="ir", quita_muse="extremos"),
                 dict(inyecta_polvo=(0.2, 900.0)), dict(inyecta_bt=(2600, 4, 1.8, 2), inyecta_polvo=(-0.2, 900.0)),
                 dict(inyecta_bt=(2600, 4, 1.8, 2), inyecta_polvo=(0.2,))]
        for kw in malas:
            with self.assertRaises(ValueError, msg=kw):
                self.tb.valida_disco(**kw)

    def test_combinaciones_validas(self):
        for kw in (dict(rk_fijo=0.0), dict(td_rango=(300.0, 1500.0)), dict(quita_muse="extremos"),
                   dict(datos="muse", quita_muse="extremos"),
                   dict(inyecta_bt=(2600, 4, 1.8, 2), inyecta_polvo=(0.2, 900.0), td_rango=(300.0, 1500.0))):
            self.tb.valida_disco(**kw)

    def test_modelo_falla_antes_de_leer_nada(self):
        # un slug que no existe: si validara después de leer, el error sería otro
        with self.assertRaises(ValueError):
            self.tb.Modelo("no_existe", rk_fijo=0.0, td_rango=(300.0, 1500.0))


class QuitaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tb = carga_script("teff_bayes")

    def test_quita_los_bins_por_su_centro(self):
        s = dict(w=np.array([6290.0, 6300.0, 6899.9, 6900.0, 8000.0, 9000.0, 9349.0, 9350.0]),
                 bueno=np.array([True] * 7 + [False]), f=np.arange(8.0))
        q = self.tb.quita_intervalos(s, self.tb.INTERVALOS_MUSE["extremos"])
        self.assertEqual(q["bueno"].tolist(), [True, False, False, True, True, False, False, False])
        self.assertIs(q["f"], s["f"])                       # solo cambia `bueno`
        self.assertEqual(s["bueno"].sum(), 7)               # sin tocar el original

    def test_extremos_son_las_bandas_del_d5(self):
        dr = carga_script("delta_entre_rejillas")
        b = dr.BANDAS_A
        self.assertEqual(self.tb.INTERVALOS_MUSE["extremos"], ((b[0], b[1]), (b[-2], b[-1])))

    def test_azul_y_rojo_son_cada_extremo(self):
        """`azul` y `rojo` (2026-10-30) son los dos intervalos de `extremos` por separado: juntos quitan lo mismo."""
        I = self.tb.INTERVALOS_MUSE
        self.assertEqual(I["azul"] + I["rojo"], I["extremos"])
        s = dict(w=np.arange(6200.0, 9400.0, 15.0), bueno=np.ones(214, bool))
        az, ro, ex = (self.tb.quita_intervalos(s, I[k])["bueno"] for k in ("azul", "rojo", "extremos"))
        np.testing.assert_array_equal(az & ro, ex)
        self.assertEqual(((~az) & (~ro)).sum(), 0)          # no se solapan
        self.assertEqual(((~az).sum(), (~ro).sum()), (40, 23))   # [6300, 6900) y [9000, 9350) cada 15 Å


class EtiquetaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tb = carga_script("teff_bayes")

    def _a(self, **kw):
        base = dict(etiqueta=None, slug="ROXs12b", rejilla="btsettl", modo="libre", tracks="ATMO2020", av="primaria",
                    plantilla=None, av_fija=None, inyecta_bt=None, datos="todo", r250=False, semilla=1)
        base.update(kw)
        return argparse.Namespace(**base)

    def test_sin_las_opciones_nuevas_la_de_siempre(self):
        self.assertEqual(self.tb.etiqueta_de(self._a()), "ROXs12b_libre_avprimaria_s1")
        self.assertEqual(self.tb.etiqueta_de(self._a(rk_fijo=None, td_libre=None, quita_muse=None,
                                                     inyecta_polvo=None)), "ROXs12b_libre_avprimaria_s1")

    def test_piezas_nuevas_antes_de_la_semilla(self):
        a = self._a(av_fija=1.8, rk_fijo=0.0, quita_muse="extremos")
        self.assertEqual(self.tb.etiqueta_de(a), "ROXs12b_libre_avprimaria_avfija1.8_rk0_quita-extremos_s1")
        a = self._a(inyecta_bt=[2600.0, 4.0, 1.8, 2.0], td_libre=[300.0, 1500.0], inyecta_polvo=[0.2, 900.0])
        self.assertEqual(self.tb.etiqueta_de(a),
                         "ROXs12b_libre_avprimaria_bt-2600-4-1.8-2_td300-1500_polvo-0.2-900_s1")
        a = self._a(mascara_muse="sin_o2b", ventana_lineas=50.0, quita_muse="extremos")
        self.assertEqual(self.tb.etiqueta_de(a),
                         "ROXs12b_libre_avprimaria_mascara-sin_o2b_ventana-50_quita-extremos_s1")


# ------------------------------------------------------------------ datos reales
@pytest.mark.external_data
@requiere("runs")
class QuitaDatosTests(unittest.TestCase):
    """Sin rejillas: los bins del espectro MUSE de los runs del paper."""

    def test_extremos_quita_25_y_6_de_los_84_bins(self):
        tb = carga_script("teff_bayes")
        from scripts import tipo_ir_plantillas as tp
        with raiz_temporal():
            for slug, run in runs_del_paper().items():
                s = tp.segmento_muse(run)
                q = tb.quita_intervalos(s, tb.INTERVALOS_MUSE["extremos"])
                w = s["w"][s["bueno"]]
                self.assertEqual(int(s["bueno"].sum()), 84, slug)
                self.assertEqual(int(q["bueno"].sum()), 84 - 25 - 6, slug)
                quitados = s["w"][s["bueno"] & ~q["bueno"]]
                # los centros del texto del D.5 (los de 12 b): 6324–6844 y 9128–9199 Å; en 42B b, ±0.5 Å
                self.assertEqual(int(np.sum(quitados < 7000.0)), 25, slug)
                self.assertEqual(int(np.sum(quitados > 9000.0)), 6, slug)
                bordes = (quitados.min(), quitados[quitados < 7000.0].max(), quitados[quitados > 9000.0].min(),
                          quitados.max())
                np.testing.assert_allclose(bordes, (6324.0, 6844.0, 9128.0, 9199.0), atol=1.0, err_msg=slug)
                self.assertTrue(np.all((w[~np.isin(w, quitados)] > 6900.0) & (w[~np.isin(w, quitados)] < 9000.0)))


@pytest.mark.external_data
@pytest.mark.slow
@requiere("runs", "ir", "bibliotecas")
class ModeloDatosTests(unittest.TestCase):
    """Con las rejillas de /mnt/2TB (BT-Settl, R = 500), sobre el run del paper de ROXs 12 b."""

    @classmethod
    def setUpClass(cls):
        cls.tb = carga_script("teff_bayes")
        cls.slug = next(s for s in runs_del_paper() if run_del_paper(s).startswith("ROXs12b"))
        try:
            cls.base = modelo_teff(cls.slug)
        except (OSError, SystemExit) as exc:   # sin las rejillas de /mnt/2TB
            raise unittest.SkipTest(f"no se puede construir el modelo: {exc}")

    def _th(self, m, **v):
        th = dict(zip(self.base.nombres, [0.5 * (a + b) for a, b in zip(
            self.base.prior_transform(np.full(self.base.ndim, 0.2)), self.base.prior_transform(np.full(self.base.ndim, 0.8)))]))
        th.update(td=self.tb.T_D, **v)
        return np.array([th[k] for k in m.nombres])

    def test_rk_fijo_cero_da_la_verosimilitud_con_rk_cero(self):
        fijo = modelo_teff(self.slug, rk_fijo=0.0)
        self.assertNotIn("rk", fijo.nombres)
        self.assertEqual(fijo.loglike(self._th(fijo)), self.base.loglike(self._th(self.base, rk=0.0)))

    def test_td_libre_en_t_d_da_la_verosimilitud_de_siempre(self):
        td = modelo_teff(self.slug, td_rango=(300.0, 1500.0))
        self.assertEqual(td.nombres[td.nombres.index("rk") + 1], "td")
        self.assertEqual(td.loglike(self._th(td)), self.base.loglike(self._th(self.base)))

    def test_quita_extremos_deja_53_bins_de_muse(self):
        q = modelo_teff(self.slug, quita_muse="extremos")
        n = {s["nombre"]: int(s["d"].size) for s in q.segs}
        n0 = {s["nombre"]: int(s["d"].size) for s in self.base.segs}
        self.assertEqual((n0["MUSE"], n["MUSE"]), (84, 53))
        self.assertEqual({k: v for k, v in n.items() if k != "MUSE"}, {k: v for k, v in n0.items() if k != "MUSE"})

    def test_el_polvo_inyectado_solo_cambia_el_ir(self):
        bt = (2600.0, 4.0, 1.8, 2.0)
        sin = modelo_teff(self.slug, inyecta_bt=bt)
        con = modelo_teff(self.slug, inyecta_bt=bt, inyecta_polvo=(0.5, 900.0))
        self.assertEqual(con.verdad, dict(teff=2600.0, logg=4.0, av=1.8, r_rj=2.0, rk=0.5, td=900.0))
        self.assertEqual(sin.verdad, dict(teff=2600.0, logg=4.0, av=1.8, r_rj=2.0))
        for a, b in zip(sin.segs, con.segs):
            if a["banda"] == "vis":   # el mismo ruido (misma semilla) y nada de polvo
                self.assertTrue(np.array_equal(a["d"], b["d"]))
            else:                     # el exceso suma flujo, y más en el rojo de K
                self.assertGreater(float(np.median(b["d"] - a["d"])), 0.0, a["nombre"])


if __name__ == "__main__":
    unittest.main()
