"""Cotas del disco circumplanetario (`scripts/disco_cotas.py`, Sect. 5.1 y métodos del paper).

Las primeras clases no leen nada de disco: cada fórmula contra un caso calculado a mano (sublimación, radio de
Hill, radio magnetosférico y corrotación, balance de energía, disco plano pasivo, escala de altura, disco grueso
de ALMA, columna del disco de canto, colores y extinción), la maquinaria de bandas con una fotosfera de cuerpo
negro, el Montecarlo de la marea sin dispersión, la tabla de literatura y la lectura de las cadenas.
`CifrasDelPaperTests` lee los runs del paper, el IR de /mnt/2TB y las plantillas (marcas `slow` y
`external_data`); si falta algo se salta, no falla.
"""
import importlib.util
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _carga(nombre):
    spec = importlib.util.spec_from_file_location(nombre, ROOT / "scripts" / f"{nombre}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[nombre] = mod
    spec.loader.exec_module(mod)
    return mod


dc = _carga("disco_cotas")


class SublimacionTests(unittest.TestCase):
    def test_caso_a_mano(self):
        # ½ √4 (3000/1500)² = 4
        self.assertAlmostEqual(dc.radio_sublimacion(3000.0, 1500.0, 4.0), 4.0, places=12)
        self.assertAlmostEqual(dc.radio_sublimacion(3000.0, 1500.0), 2.0, places=12)

    def test_segunda_forma_de_monnier(self):
        # Ec. 1 de Monnier & Millan-Gabet 2002: R_s = 1.1 √Q_R (L/1000 L☉)^½ (T_s/1500 K)^−2 au
        L = 1000.0
        R = 1.0e12                                       # cualquier R: T* sale de L
        teff = (L * dc.LSUN_ERG_S / (4 * np.pi * R ** 2 * dc.SIGMA_SB_CGS)) ** 0.25
        rs_au = dc.radio_sublimacion(teff, 1500.0) * R / dc.AU_CM
        # el 1.1 del artículo está redondeado a dos cifras: a mano, 0.5 √(L/4πσ)/T_s² = 1.089 au
        self.assertAlmostEqual(rs_au, 1.1, delta=0.05)
        self.assertAlmostEqual(rs_au, 1.089, delta=0.001)

    def test_teff_del_sol(self):
        self.assertAlmostEqual(dc.teff_de_lbol(0.0, dc.RSUN_CM), 5772.0, delta=1.0)


class HillTests(unittest.TestCase):
    def test_caso_a_mano(self):
        # (3e-3 / 3)^⅓ = 0.1: R_H = 10 au a 100 au; en el pericentro con e = 0.5, 5 au
        self.assertAlmostEqual(dc.radio_de_hill(100.0, 0.0, 3e-3, 1.0), 10.0, places=10)
        self.assertAlmostEqual(dc.radio_de_hill(100.0, 0.5, 3e-3, 1.0), 5.0, places=10)

    def test_marea_sin_dispersion_es_la_formula(self):
        e = dict(slug="x", m_msun=3e-3, m_err_msun=0.0, m_host_msun=(1.0, 0.0, 0.0), rho_arcsec=1.0, d_pc=100.0,
                 orbita=dict(a=(200.0, 0.0, 0.0), e=(0.5, 0.0, 0.0)), e_max95=0.8)
        r = dc.marea([e], (1 / 3, 0.4), n=1000)["x"]
        self.assertEqual(r["orbita"], "publicada")
        self.assertAlmostEqual(r["RH_rho_au"][1], 10.0, places=9)
        self.assertAlmostEqual(r["RH_peri_au"][1], 10.0, places=9)        # 200 (1 − 0.5) · 0.1
        self.assertAlmostEqual(r["rt_min_peri_au"][0], 10.0 / 3, places=9)
        self.assertAlmostEqual(r["rt_max_peri_au"][2], 4.0, places=9)
        self.assertAlmostEqual(r["rt_max_emax95_au"][1], 0.4 * 100.0 * 0.2 * 0.1, places=9)

    def test_marea_es_reproducible_y_supone_la_orbita_si_falta(self):
        e = dict(slug="x", m_msun=0.018, m_err_msun=0.005, m_host_msun=(0.65, 0.05, 0.09), rho_arcsec=1.8,
                 d_pc=138.6, orbita=None, e_max95=0.83)
        a = dc.marea([e], (1 / 3, 0.4), n=5000)["x"]
        b = dc.marea([e], (1 / 3, 0.4), n=5000)["x"]
        self.assertEqual(a, b)
        self.assertTrue(a["orbita"].startswith("supuesta"))
        rho = 1.8 * 138.6
        self.assertTrue(rho <= a["a_au"][0] and a["a_au"][2] <= 1.4 * rho)


class MagnetosferaTests(unittest.TestCase):
    def test_caso_construido(self):
        # 4GM = 1e26, Ṁ = 1e20 g/s, μ = BR³ = 1e34: μ⁴/(4GMṀ²) = 1e136/1e66 = 1e70 → r_M = ξ · 1e10 cm
        M = 1e26 / (4 * dc.G_CGS)
        r = dc.radio_magnetosferico(1e4, 1e10, M, 1e20, 0.7)
        self.assertAlmostEqual(r / 7e9, 1.0, places=12)
        self.assertAlmostEqual(dc.campo_para_radio(r, 1e10, M, 1e20, 0.7) / 1e4, 1.0, places=12)
        self.assertAlmostEqual(dc.tasa_para_radio(r, 1e4, 1e10, M, 0.7) / 1e20, 1.0, places=12)

    def test_escalas(self):
        M, R, md = 1e31, 1.5e10, 1e13
        r0 = dc.radio_magnetosferico(100.0, R, M, md, 0.7)
        self.assertAlmostEqual(dc.radio_magnetosferico(200.0, R, M, md, 0.7) / r0, 2 ** (4 / 7), places=12)
        self.assertAlmostEqual(dc.radio_magnetosferico(100.0, R, M, 2 * md, 0.7) / r0, 2 ** (-2 / 7), places=12)

    def test_valores_fiduciales_de_hartmann(self):
        # HHC16 Sect. 2.2: M = 0.5 M☉, R = 2 R☉, Ṁ = 1e-8 M☉/yr, B = 1 kG, ξ = 0.7 → «r_M ≈ 4–6 R*»; a mano, 4.56 R*
        R, M = 2 * dc.RSUN_CM, 0.5 * dc.M_SUN_G
        r = dc.radio_magnetosferico(1e3, R, M, 1e-8 * dc.M_SUN_G / dc.YR_S, 0.7) / R
        self.assertAlmostEqual(r, 4.56, delta=0.02)

    def test_corrotacion(self):
        # v = v_ruptura/8 → (r_co/R)³ = (v_ruptura/v)² = 64 → r_co = 4R
        R, M = 1.5e10, 3.6e31
        v = math.sqrt(dc.G_CGS * M / R) / 8
        rco, p = dc.radio_corrotacion(R, M, v)
        self.assertAlmostEqual(rco / R, 4.0, places=10)
        self.assertAlmostEqual(p, 2 * math.pi * R / v, places=6)

    def test_rin_del_pipeline(self):
        self.assertAlmostEqual(1 / (1 - 1 / dc.MAGNETOSPHERIC_FACTOR), 5.0, places=12)


class EnergiaTests(unittest.TestCase):
    def test_fotosfera_de_cuerpo_negro_al_mismo_T(self):
        # F_fot = πB(T): polvo a T_d = T con exceso r_K reemite exactamente r_K de L*
        nu = 1.4e14
        b = dc.planck_nu(nu, 2400.0)
        self.assertAlmostEqual(dc.fraccion_reemitida(0.3, np.pi * b, b, 2400.0, 2400.0), 0.3, places=12)
        # a la mitad de T_d con el mismo B_K, f cae como (T_d/T)⁴
        self.assertAlmostEqual(dc.fraccion_reemitida(0.3, np.pi * b, b, 1200.0, 2400.0), 0.3 / 16, places=12)

    def test_planck_rayleigh_jeans(self):
        nu, T = 1e9, 100.0
        self.assertAlmostEqual(dc.planck_nu(nu, T) / (2 * nu ** 2 * dc.K_B_CGS * T / dc.C_CGS ** 2), 1.0, delta=1e-3)


class DiscoPlanoTests(unittest.TestCase):
    def test_reprocesa_un_cuarto(self):
        r = np.geomspace(1.0, 1e5, 20000)
        self.assertAlmostEqual(dc.fraccion_disco_plano(r, 2400.0), 0.25, delta=1e-3)

    def test_asintota_de_chiang_goldreich(self):
        # CG97 Ec. 4: T ≈ (2/3π)^¼ (R/r)^¾ T*
        r = 1000.0
        t = dc.temperatura_disco_plano(r, 2400.0)
        self.assertAlmostEqual(t / ((2 / (3 * np.pi)) ** 0.25 * r ** -0.75 * 2400.0), 1.0, delta=1e-5)

    def test_escala_de_altura(self):
        # T con kT/(μ m_H) = 1e10 (c_s = 1 km/s) y M con GM/r = 1e14 (v_K = 100 km/s): H/r = 0.01
        T = 1e10 * dc.MU_GAS * dc.M_H_G / dc.K_B_CGS
        r = 1e10
        M = 1e14 * r / dc.G_CGS
        self.assertAlmostEqual(dc.escala_altura(M, r, T, 1.0), 0.01, places=12)


class AlmaTests(unittest.TestCase):
    def test_equilibrio_a_1_au(self):
        self.assertAlmostEqual(dc.temperatura_equilibrio(1.0, dc.AU_CM), 278.3, delta=0.1)

    def test_disco_isotermo(self):
        # sin irradiación (T* ≈ 0), T = T_fondo: flujo = cos i π (r² − r0²) B_ν/d² → r_max a mano
        nu, T, d, cosi = dc.C_CGS / 0.088, 15.0, 140 * dc.PC_CM, 0.5
        f = 0.2e-26
        r = dc.radio_max_grueso(f, d, nu, 1e10, 1e-3, T, 0.02, cosi)
        r_mano = math.sqrt(f * d ** 2 / (math.pi * cosi * dc.planck_nu(nu, T)) + (0.01 * dc.AU_CM) ** 2) / dc.AU_CM
        self.assertAlmostEqual(r["rmax_au"] / r_mano, 1.0, delta=3e-3)

    def test_sin_cruce_es_infinito(self):
        r = dc.radio_max_grueso(1.0, 140 * dc.PC_CM, dc.C_CGS / 0.088, 1e10, 1e-3, 15.0, 0.02, 1.0)
        self.assertEqual(r["rmax_au"], math.inf)


class CantoTests(unittest.TestCase):
    def test_columna_a_mano(self):
        # r_in = 1, r_d = 2, M = 4π → Σ_0 = 1 y N_0 = 2 (1 − ½) / (√(2π) h) = 1/(√(2π) h)
        h = 0.1
        p = dc.probabilidad_canto(4 * np.pi, 1.0, 2.0, h, 1e-9, 1e-10)
        self.assertAlmostEqual(p["col_plano_gcm2"], 1 / (np.sqrt(2 * np.pi) * h), places=12)

    def test_angulos_y_probabilidad(self):
        h = 0.1
        n0 = 1 / (np.sqrt(2 * np.pi) * h)
        self.assertAlmostEqual(dc.angulo_de_columna(n0, n0 * math.exp(-0.5), h), h, places=12)
        self.assertEqual(dc.angulo_de_columna(n0, 2 * n0, h), 0.0)
        p = dc.probabilidad_canto(4 * np.pi, 1.0, 2.0, h, n0 * math.exp(-0.5), n0 * math.exp(-2.0))
        self.assertAlmostEqual(p["P"], math.sin(2 * h) - math.sin(h), places=12)


class FotometriaTests(unittest.TestCase):
    def test_cambio_de_color(self):
        self.assertEqual(dc.cambio_de_color(0.2, 0.2), 0.0)
        self.assertAlmostEqual(dc.cambio_de_color(0.0, 10 ** 0.4 - 1), 1.0, places=12)

    def test_extincion_ir_es_la_de_musepipe(self):
        from musepipe.models.extinction import CCMExtinction
        ccm = CCMExtinction(3.1, citation="Cardelli, Clayton & Mathis 1989")
        for lam in (1.25, 2.2, 3.2):
            self.assertAlmostEqual(dc.a_lambda_sobre_av_ir(lam), ccm.a_lambda_over_av(lam * 1e4), places=12)

    def test_flujo_w4_a_mano(self):
        # 8.2839 Jy · 10^(−0.4 · 6.558) = 19.726 mJy
        _, mjy = dc.nu_l_nu_lsun(6.558, 8.2839, 22.0883, 138.6 * dc.PC_CM)
        self.assertAlmostEqual(mjy, 19.726, delta=0.002)


def _fotosfera_cuerpo_negro(teffs=(2000.0, 2400.0, 2800.0), logg=4.0):
    lam = np.sqrt(dc.bordes_logaritmicos()[1:] * dc.bordes_logaritmicos()[:-1])
    lam_cm = lam * 1e-4
    flam = [np.pi * 2 * dc.H_CGS * dc.C_CGS ** 2 / lam_cm ** 5
            / np.expm1(dc.H_CGS * dc.C_CGS / (lam_cm * dc.K_B_CGS * t)) * 1e-8 for t in teffs]
    return dc.Fotosfera(lam, list(teffs), [logg] * len(teffs), np.array(flam))


class FotosferaTests(unittest.TestCase):
    def test_cuerpo_negro_da_f_igual_a_rk(self):
        fot = _fotosfera_cuerpo_negro()
        f, r_lp = fot.por_rk(2400.0, 2400.0)
        self.assertAlmostEqual(f, 1.0, places=9)
        self.assertAlmostEqual(r_lp, 1.0, places=9)

    def test_color_de_vega_es_cero(self):
        fot = _fotosfera_cuerpo_negro(teffs=(dc.T_VEGA_K, dc.T_VEGA_K + 1))
        self.assertAlmostEqual(fot.color(fot.fnu(dc.T_VEGA_K), "K", "Lp"), 0.0, places=10)

    def test_completa_con_rayleigh_jeans(self):
        fot = _fotosfera_cuerpo_negro()
        flam = fot.flam.copy()
        corte = np.searchsorted(fot.lam, 23.4)
        flam[1, corte:] = 0.0
        f2 = dc.Fotosfera(fot.lam, fot.teff, fot.logg, flam)
        j = corte - 1
        self.assertTrue(np.allclose(f2.flam[1, corte:], flam[1, j] * (fot.lam[corte:] / fot.lam[j]) ** -4))
        self.assertTrue(np.array_equal(f2.flam[0], fot.flam[0]))

    def test_disco_plano_con_borde_lejos_no_da_exceso_en_k(self):
        fot = _fotosfera_cuerpo_negro()
        r = fot.disco_plano(2400.0, 1.5e10, 10.0, 1.0)
        self.assertLess(r["r_K"], 1e-6)
        # lejos del borde T⁴ ≈ (2/3π) r⁻³ T*⁴: las dos caras reemiten (2/3π)(1/r_in − 1/r_out) ≈ 0.0212 de L*
        self.assertAlmostEqual(r["f_L"], 2 / (3 * np.pi) / 10.0, delta=2e-4)


class NodosTests(unittest.TestCase):
    def test_nodos_de_la_fotosfera(self):
        cad = [dict(rejilla="BT-Settl", teff=2719.0), dict(rejilla="DRIFT-PHOENIX", teff=2841.0)]
        self.assertEqual(dc.nodos_fotosfera(2408.2, cad), [2400.0, 2700.0])
        self.assertEqual(dc.nodos_fotosfera(2209.5, [dict(rejilla="BT-Settl", teff=2335.0)]), [2200.0, 2300.0])
        self.assertEqual(dc.nodos_fotosfera(2408.2, None), [2400.0, 2400.0])


class LiteraturaTests(unittest.TestCase):
    def setUp(self):
        self.lit = dc.Literatura()

    def test_valores_y_fracciones(self):
        self.assertEqual(self.lit.valor("fraccion_hill_min"), 1 / 3)
        self.assertEqual(self.lit.valor("fraccion_hill_max"), 0.4)
        self.assertEqual(self.lit.valor("nh_por_av"), 2.21e21)

    def test_cada_fuente_esta_en_el_readme(self):
        readme = (dc.LITERATURA.parent / "README.md").read_text()
        for f in {r["fuente"] for r in self.lit.filas_}:
            self.assertIn(f"| {f} |", readme, f"la fuente {f} no tiene cita en el README")

    def test_cada_objeto_del_paper_tiene_sus_filas(self):
        for ficha in sorted((ROOT / "targets").glob("*.json")):
            if "paper" not in json.loads(ficha.read_text()):
                continue
            s = ficha.stem
            for clave in ("masa_primaria", "e_max95", "alma_flujo"):
                self.assertTrue(self.lit.tiene(clave, s), f"{s}: falta {clave}")
            self.assertTrue(self.lit.tiene("log_lbol_primaria", s) or self.lit.tiene("mbol_primaria", s))
            self.assertTrue(self.lit.filas("vsini", s))


class CadenasTests(unittest.TestCase):
    def test_lee_solo_las_de_la_tabla_y_en_su_orden(self):
        def corrida(rejilla, av_fija, teff, rk, **extra):
            d = dict(slug="obj", modo="libre", av="primaria", rejilla=rejilla, av_fija=av_fija,
                     posterior=dict(teff=dict(p16=0, p50=teff, p84=0), rk=dict(p16=0, p50=rk, p84=0)))
            d.update(extra)
            return d
        with tempfile.TemporaryDirectory() as tmp:
            for i, d in enumerate((corrida("drift", 1.8, 2702.0, 1.47), corrida("btsettl", None, 2719.0, 0.48),
                                   corrida("btsettl", 1.8, 2589.0, 1.41), corrida("btsettl", 3.0, 2671.0, 0.75),
                                   corrida("btsettl", 1.8, 9999.0, 9.9, datos="muse"),
                                   corrida("ames_cond", None, 9999.0, 9.9))):
                (Path(tmp) / f"c{i}.json").write_text(json.dumps(d))
            filas = dc.carga_cadenas([tmp])["obj"]
        self.assertEqual([(f["rejilla"], f["av"]) for f in filas],
                         [("BT-Settl", 1.8), ("BT-Settl", 3.0), ("BT-Settl", "prior"), ("DRIFT-PHOENIX", 1.8)])
        self.assertEqual(filas[2]["teff"], 2719.0)


def _hay_datos():
    return (dc.BTSETTL_RAW.is_dir() and (ROOT / "runs" / "ROXs12b_invvar" / "stages").is_dir()
            and (ROOT / "runs" / "ROXs42Bb_realigned" / "stages").is_dir())


@pytest.mark.slow
@pytest.mark.external_data
@unittest.skipUnless(_hay_datos(), "sin el BT-Settl de rango completo en /mnt/2TB o sin los runs del paper")
class CifrasDelPaperTests(unittest.TestCase):
    """Las cifras de la Sect. 5.1 y de métodos que no dependen de la Tabla B.6, a la precisión del paper."""

    @classmethod
    def setUpClass(cls):
        from scripts import paper_figures as pf
        pf.PROYECTO = ROOT
        lit = dc.Literatura()
        objetos = pf.objetos_del_paper()
        entradas = [dc.entradas_del_objeto(o, lit) for o in objetos]
        rk = {e["slug"]: dc.rk_del_co(e["slug"], ROOT) for e in entradas}
        fot = dc.Fotosfera(*dc.carga_btsettl())
        cls.r = dc.calcula(entradas, fot, lit, {}, rk)

    def test_energia_y_marea(self):
        a, b = self.r["ROXs12b"], self.r["ROXs42Bb"]
        # sin las cadenas de la Tabla B.6 la fotosfera es solo la del nodo de T_eff(L_bol, R): «about 0.8–0.9»
        for x in a["f_por_rK_1500"] + b["f_por_rK_1500"]:
            self.assertTrue(0.75 <= x < 0.95, x)
        self.assertLessEqual(a["disco_plano"]["Rin=1.5R,cosi=1"]["r_K"], 0.1)
        self.assertEqual([round(x) for x in a["marea"]["rt_peri_16_84_au"]], [12, 24])
        self.assertEqual([round(x) for x in b["marea"]["rt_peri_16_84_au"]], [4, 14])
        self.assertLess(max(a["alma"]["rmax_au"] + b["alma"]["rmax_au"]), 1.2)

    def test_color_y_22um(self):
        c = self.r["ROXs12b"]["color_KLp"]
        self.assertEqual([round(v, 2) for v in c["KLp_compañero"].values()], [0.94, 1.27])
        self.assertEqual((round(c["Ks_W1"], 2), round(c["Ks_W2"], 2)), (0.29, 0.39))
        self.assertEqual(round(self.r["ROXs12b"]["exceso_22um"]["W4"]["exceso_sobre_Lbol"], 1), 0.9)

    def test_magnetosfera_y_canto(self):
        m = self.r["ROXs12b"]["magnetosfera"]
        self.assertEqual(round(m["B_rin_G"]["estelar"] / 1e3, 2), 0.03)
        self.assertEqual(round(m["B_rin_G"]["planetaria"] / 1e3, 2), 0.13)
        self.assertEqual(round(m["rM_R"]["B=300,estelar"]), 18)
        rco = sorted(v["rco_max_R"] for v in m["corrotacion"].values())
        self.assertEqual((round(rco[0]), round(rco[-1])), (6, 15))
        p = self.r["ROXs12b"]["canto"]["P"]
        self.assertEqual((round(100 * p[0]), round(100 * p[1])), (1, 6))
