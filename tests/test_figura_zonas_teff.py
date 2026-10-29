"""Zonas de los espectros en los ajustes de T_eff (`scripts/figura_zonas_teff.py`).

`ClasificaMuseTests` y `ClasificaIrTests` no leen nada de disco: un espectro sintético con una banda telúrica, una
ventana de línea solapada con ella, una banda de H2O, un canal malo, el hueco del láser y el borde del rango, que
pasa por las MISMAS funciones de `musepipe` que el ajuste (`degrade_observed`, `fit_spectrum_from_inputs`). La
razón de cada canal tiene que ser la esperada (prioridad, margen de R = 500 con su origen, bins descartados) y los
bins que salen de los canales marcados `usado` tienen que ser los de `fit_spectrum_from_inputs`.

`DatosRealesTests` (marca `external_data`) reconstruye las zonas de los dos compañeros del paper desde los runs y
las compara con `tipo_ir_plantillas.preparar` (lo que comprueba `comprueba`), y fija lo que dice el apéndice B: 84
bins de 15 Å entre 6324 y 9199 Å (32 entre 6420 y 9165 Å a R = 250). Con las cadenas, además, el `teff_bayes.Modelo`
de la cadena BT-Settl con el prior: sus píxeles son los marcados y su `loglike` da el `logl_max` de la cadena. Lee
`MUSE_PROJECT_ROOT` (la raíz con `runs/`, por defecto el repo) y `MUSE_TEFF_CADENAS` (por defecto la carpeta de
cadenas del paper); si falta algo se salta, no falla.
"""
import json
import os
import sys
import unittest
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from musepipe.models import observed as obs  # noqa: E402
from scripts import figura_zonas_teff as fz  # noqa: E402
from scripts import teff_bayes as tb  # noqa: E402


#: FWHM del núcleo del sintético (Å): ±2 canales de margen, para que un canal malo suelto no descarte su bin de
#: 12 (con el de R = 500 real, 12–19 Å, sí lo descarta)
NUCLEO_A = 3.0


def _sintetico(n_canales=12, rango=(6300.0, 6900.0), nucleo_A=NUCLEO_A):
    """Espectro de 5700–7000 Å a 1.25 Å con flujo 1 + rampa, error 0.1 y NaN en el hueco del láser."""
    w = np.arange(5700.0, 7000.0, 1.25)
    f = 1.0 + 1e-4 * (w - 6000.0)
    e = np.full(w.size, 0.1)
    laser = (w >= fz.AO_LASER_WINDOW_A[0]) & (w <= fz.AO_LASER_WINDOW_A[1])
    f[laser] = np.nan
    e[laser] = np.nan
    comp = {
        "o2": (w >= 6500.0) & (w <= 6540.0),
        "linea": (w >= 6530.0) & (w <= 6560.0),          # solapa con O2: gana O2
        "h2o": (w >= 6700.0) & (w <= 6760.0),
        "canal_malo": np.isin(np.arange(w.size), [np.argmin(np.abs(w - 6620.0))]),
    }
    nucleo = np.full(w.size, nucleo_A)
    mascara = comp["o2"] | comp["linea"] | comp["h2o"] | comp["canal_malo"]
    cov = {"block_bounds": np.array([[0, w.size]]), "n_eff_over_n_by_block": np.array([0.7])}
    ratio = obs.neff_over_n_per_channel(w, cov, None)
    f_d, e_d, m_d, n_d = obs.degrade_observed(w, f, e, mascara, nucleo, ratio)
    c = fz.clasifica_muse(w, f, e, comp, rango, nucleo, f_d, e_d, n_d, n_canales=n_canales)
    inputs = {"wave": w, "flux": f, "err": e, "mask": mascara, "cov": cov, "bad_mask": None,
              "err_column": "flux_err_total", "mask_prov": {}}
    fs = obs.fit_spectrum_from_inputs(inputs, n_channels=n_canales, wave_range=list(rango),
                                      data_kernel_fwhm_A=nucleo_A)
    return w, comp, c, m_d, fs


class ClasificaMuseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.w, cls.comp, cls.c, cls.m_d, cls.fs = _sintetico()

    def test_la_mascara_crecida_es_la_de_degrade_observed(self):
        np.testing.assert_array_equal(self.c["crecida"], self.m_d)

    def test_los_bins_usados_son_los_de_fit_spectrum_from_inputs(self):
        np.testing.assert_allclose(self.c["w_bin"], self.fs.wave_bin, rtol=1e-13, atol=0)
        np.testing.assert_allclose(self.c["f_bin"], self.fs.flux_bin, rtol=1e-13, atol=0)
        np.testing.assert_allclose(self.c["e_bin"], self.fs.err_bin, rtol=1e-13, atol=0)
        self.assertEqual(int(self.c["bin_usado"].sum()), self.fs.n_bins)

    def test_cada_canal_usado_esta_en_un_bin_usado_y_al_reves(self):
        r, ids = self.c["razon"], self.c["bin_id"]
        usados = np.nonzero(self.c["bin_usado"])[0]
        self.assertTrue(np.all(np.isin(ids[r == "usado"], usados)))
        for j in usados:
            self.assertTrue(np.any(r[ids == j] == "usado"))

    def test_prioridad_de_las_razones(self):
        w, r = self.w, self.c["razon"]
        self.assertTrue(np.all(r[(w >= 5780.0) & (w <= 6050.0)] == "laser"))
        self.assertTrue(np.all(r[(w < 6300.0) & ~((w >= 5780.0) & (w <= 6050.0))] == "rango"))
        self.assertTrue(np.all(r[w > 6900.0] == "rango"))
        self.assertTrue(np.all(r[self.comp["o2"]] == "o2"))            # también lo que solapa con la línea
        self.assertTrue(np.all(r[self.comp["linea"] & ~self.comp["o2"]] == "linea"))
        self.assertTrue(np.all(r[self.comp["h2o"]] == "h2o"))
        self.assertEqual(list(r[self.comp["canal_malo"]]), ["canal_malo"])

    def test_el_margen_es_el_nucleo_alrededor_de_lo_excluido_con_su_origen(self):
        w, r, o = self.w, self.c["razon"], self.c["origen"]
        # justo fuera de la banda de H2O, a menos de un núcleo: margen de H2O
        for x in (6700.0 - 0.8 * NUCLEO_A, 6760.0 + 0.8 * NUCLEO_A):
            i = int(np.argmin(np.abs(w - x)))
            self.assertIn(r[i], ("margen", "bin"))
            self.assertEqual(o[i], "h2o")
        # entre O2 (hasta 6540) y la línea (hasta 6560): el margen a la derecha de la línea es de la línea
        i = int(np.argmin(np.abs(w - (6560.0 + 0.8 * NUCLEO_A))))
        self.assertEqual(o[i], "linea")
        # a más de un núcleo de todo, libre
        i = int(np.argmin(np.abs(w - 6450.0)))
        self.assertEqual(r[i], "usado")
        # ningún margen sin origen, y el margen nunca sale de ±1 núcleo de un canal excluido
        self.assertTrue(np.all(o[r == "margen"] != ""))
        malo = np.zeros(w.size, bool)
        for m in self.comp.values():
            malo |= m
        for i in np.nonzero(r == "margen")[0]:
            self.assertLessEqual(float(np.min(np.abs(w[malo] - w[i]))), NUCLEO_A)

    def test_bin_descartado_por_mas_de_la_mitad_excluida(self):
        r, ids = self.c["razon"], self.c["bin_id"]
        descartados = [j for j in range(self.c["bin_usado"].size) if not self.c["bin_usado"][j]]
        self.assertTrue(descartados)
        con_libres = 0
        for j in descartados:
            en = ids == j
            excluidos = np.isin(r[en], ("o2", "h2o", "linea", "canal_malo", "margen", "sin_dato"))
            self.assertGreater(excluidos.mean(), fz.MAX_FRAC_EXCLUIDA)
            libres = r[en][~excluidos]
            self.assertTrue(np.all(libres == "bin"))
            con_libres += libres.size
        self.assertGreater(con_libres, 0, "el sintético debe tener algún bin descartado con canales libres")
        self.assertTrue(np.all(self.c["origen"][r == "bin"] != ""))

    def test_un_canal_malo_suelto_no_descarta_su_bin(self):
        ids, r = self.c["bin_id"], self.c["razon"]
        j = int(ids[self.comp["canal_malo"]][0])
        self.assertTrue(self.c["bin_usado"][j])
        self.assertIn("usado", set(r[ids == j]))


class ClasificaIrTests(unittest.TestCase):
    def test_telurico_y_sin_dato(self):
        w = np.arange(11000.0, 14000.0, 25.0)
        f = np.ones(w.size)
        e = np.full(w.size, 0.05)
        f[3] = np.nan
        e[40] = 0.0
        r = fz.clasifica_ir(w, f, e, bandas=[(11500.0, 11600.0), (13000.0, 13100.0)])
        self.assertEqual(r[3], "sin_dato")
        self.assertEqual(r[40], "sin_dato")
        np.testing.assert_array_equal(r[(w >= 11500.0) & (w <= 11600.0)], "telurico_ir")
        self.assertEqual(int((r == "usado").sum()), w.size - 2 - 5 - 5)
        sin = fz.clasifica_ir(w, f, e, telurico=False)
        self.assertEqual(int((sin == "usado").sum()), w.size - 2)

    def test_tramos_con_bordes_de_pixel(self):
        w = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        sel = np.array([True, True, False, True, False])
        self.assertEqual(fz.tramos(w, sel), [(0.5, 2.5), (3.5, 4.5)])
        lo, hi = w - 0.4, w + 0.4
        self.assertEqual(fz.tramos(w, sel, lo, hi), [(0.6, 2.4), (3.6, 4.4)])
        self.assertEqual(fz.tramos(w, np.zeros(5, bool)), [])

    def test_cobertura_marca_sin_modelo_tambien_en_los_canales(self):
        _, _, c, _, _ = _sintetico()
        nb = int(c["bin_usado"].sum())
        seg = dict(nombre="MUSE", brazo="vis", w=c["w_bin"], razon=np.full(nb, "usado", dtype=object))
        canales = dict(razon=c["razon"].copy(), bin_id=c["bin_id"], bin_usado=c["bin_usado"])
        base = np.ones(nb)
        base[0] = np.nan
        z = fz.aplica_cobertura(dict(segs=[seg], canales=canales), {0: [base], 1: [np.ones(nb)]})
        self.assertEqual(z["segs"][0]["razon"][0], "sin_modelo")
        self.assertEqual(int((z["segs"][0]["razon"] == "usado").sum()), nb - 1)
        j0 = int(np.nonzero(c["bin_usado"])[0][0])
        afectados = canales["bin_id"] == j0
        self.assertTrue(np.all(canales["razon"][afectados & (c["razon"] == "usado")] == "sin_modelo"))


# ------------------------------------------------------------------ datos reales
PROYECTO = Path(os.environ.get("MUSE_PROJECT_ROOT", ROOT))
CADENAS = Path(os.environ.get("MUSE_TEFF_CADENAS", "~/muse_noche_20261024_c4/cadenas")).expanduser()


def _slugs():
    return [s for s, _ in fz.objetos(ROOT)]


@pytest.mark.external_data
class DatosRealesTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        runs = [PROYECTO / "runs" / json.loads(f.read_text())["paper"]["run"]
                for f in sorted((ROOT / "targets").glob("*.json")) if "paper" in json.loads(f.read_text())]
        if not runs or not all(p.exists() for p in runs):
            raise unittest.SkipTest("faltan los runs del paper (MUSE_PROJECT_ROOT)")
        from scripts import tipo_ir_indices as tir
        if not tir.DEFAULT_CDS.is_dir() or not tb.REJILLAS["btsettl"]["dir"].is_dir():
            raise unittest.SkipTest("faltan los espectros IR del CDS o la rejilla BT-Settl (/mnt/2TB)")
        cls.raiz_antes = tb.ROOT
        tb.fija_raiz(PROYECTO)
        cls.res = {}
        try:
            for slug in _slugs():
                z, inf, _, _ = fz.comprueba(slug)   # sale con AssertionError si no cuadra con preparar
                cls.res[slug] = (z, inf, fz.resumen(z))
        except (OSError, SystemExit) as exc:
            tb.fija_raiz(cls.raiz_antes)
            raise unittest.SkipTest(f"no se pueden preparar los datos: {exc}")

    @classmethod
    def tearDownClass(cls):
        tb.fija_raiz(cls.raiz_antes)

    def test_la_reconstruccion_es_la_de_preparar(self):
        for slug, (_, inf, _) in self.res.items():
            for nombre, d in inf["preparar"].items():
                self.assertLessEqual(d, 1e-12, (slug, nombre))

    def test_muse_84_bins_entre_6324_y_9199_y_32_a_r250(self):
        for slug, (_, _, r) in self.res.items():
            m = r["muse"]
            self.assertEqual(m["n_bins_usados"], 84, slug)
            self.assertEqual(round(m["primer_bin_A"]), 6324, slug)
            self.assertEqual(round(m["ultimo_bin_A"]), 9199, slug)
            self.assertEqual(m["r250"]["n_bins"], 32, slug)
            self.assertEqual(round(m["r250"]["primer_A"]), 6420, slug)
            self.assertEqual(round(m["r250"]["ultimo_A"]), 9165, slug)

    def test_las_razones_cubren_todo_y_el_laser_es_lo_unico_de_04b(self):
        for slug, (z, _, r) in self.res.items():
            m = r["muse"]
            self.assertEqual(sum(d["n_canales"] for d in m["razones"].values()), m["n_canales"], slug)
            # los canales malos de 04b caen todos en el hueco del láser, que tiene prioridad
            self.assertNotIn("canal_malo", m["razones"], slug)
            self.assertEqual(m["razones"]["laser"]["n_canales"], m["n_canales_malos_04b"], slug)
            for e in r["ir"]:
                self.assertEqual(sum(d["n"] for d in e["razones"].values()), e["n"], (slug, e["nombre"]))

    def test_el_modelo_de_la_cadena_ve_los_mismos_pixeles(self):
        if not CADENAS.is_dir():
            self.skipTest("faltan las cadenas (MUSE_TEFF_CADENAS)")
        from scripts import teff_cifras_texto as tc
        K = tc.Cadenas([CADENAS])
        slug = _slugs()[0]
        d = K.por_etiqueta.get(tc.etiqueta(slug))
        if d is None:
            self.skipTest(f"falta la cadena {tc.etiqueta(slug)}")
        _, inf, _, _ = fz.comprueba(slug, cadena=d)
        self.assertEqual(inf["modelo"]["desvio_relativo_max"], 0.0)
        self.assertEqual(inf["modelo"]["n_puntos"], {k: int(v) for k, v in d["n_puntos"].items()})


if __name__ == "__main__":
    unittest.main()
