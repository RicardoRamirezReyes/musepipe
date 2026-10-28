"""`scripts/centro_companero.py`: la posición del compañero (B3, traza cromática, flujo y E1) SIN escribir en `runs/`.

Las clases rápidas no tocan `runs/`: la geometría de los desplazamientos, la traza relativa a la banda de B3 y sus
tramos por bin del modelo de PSF, los centroides de B3 (`centroid_2d`, `chromatic_centroids`) sobre fuentes
sintéticas desplazadas, la fuente falsa (`inyecta`) con la luz donde se pide (Moffat y Psfao, y el evaluador de antes
del 2026-10-21, que la ponía −(k−1)/(2k) px fuera), la estadística de las inyecciones, el remapeo a una copia y la
guarda: sin `--sin-escribir` no corre, y una salida dentro de `runs/` —también a través del enlace con el que los runs
viven en otro disco— para antes de calcular nada y no crea el fichero.

`VerificaRunReal` (marcada `slow`, ~1 min) corre la verificación sobre el primer run del paper: con la posición de B3
y sin desplazamiento, aperture y psffit extraídos en la ventana de Hα reproducen bit a bit el producto de C, sus
controles, los controles calibrados de D2 y el z y las FAP de E1 del disco, y ningún fichero del run cambia. Busca
`runs/` en este clon o en `MUSE_PROJECT_ROOT`; si falta, se salta.
"""
import importlib.util
import json
import math
import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def _carga(nombre):
    spec = importlib.util.spec_from_file_location(nombre, ROOT / "scripts" / f"{nombre}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[nombre] = mod
    spec.loader.exec_module(mod)
    return mod


cc = _carga("centro_companero")


def _moffat_doc(fwhm=4.0):
    """Moffat constante en λ, como la de `tests/test_psf_normalization.py`."""
    coeffs = {}
    for key, value in {"y0": 0.0, "x0": 0.0, "fwhm_maj": fwhm, "fwhm_min": fwhm, "theta_deg": 0.0,
                       "beta": 2.5}.items():
        coeffs[key] = {"model": "polynomial", "degree": 0, "coefficients": [value], "wave_ref_A": 7000.0,
                       "wave_scale_A": 1000.0}
    return {"form": "moffat", "norm_radius_px": 12.0, "coefficients": coeffs, "hybrid": False}


def _psfao_doc():
    names = ["r0", "C", "A", "alpha", "ratio", "theta", "beta"]
    return {
        "form": "psfao",
        "system": "muse_nfm",
        "param_names": names,
        "norm_radius_px": 12.0,
        "psfao_wave_bin_A": 100.0,
        "smoothed_poly": {
            "r0": [0.12], "C": [1e-6], "A": [10.0], "alpha": [0.5],
            "ratio": [1.0], "theta": [0.0], "beta": [1.6],
        },
    }


def _gauss(shape, yx, sigma, amp=1.0):
    yy, xx = np.indices(shape, dtype=np.float64)
    return amp * np.exp(-0.5 * ((yy - yx[0]) ** 2 + (xx - yx[1]) ** 2) / sigma ** 2)


class GeometriaTests(unittest.TestCase):
    PRIM = (85.0, 85.0)
    COMP = (85.0, 131.7)  # radial = +x

    def test_desplazamientos_en_las_cuatro_direcciones(self):
        r = math.hypot(self.COMP[0] - self.PRIM[0], self.COMP[1] - self.PRIM[1])
        out = cc.desplazada(self.COMP, self.PRIM, "r_out", 0.5)
        self.assertAlmostEqual(out[0], 85.0)
        self.assertAlmostEqual(out[1], 132.2)
        self.assertAlmostEqual(cc.desplazada(self.COMP, self.PRIM, "r_in", 0.5)[1], 131.2)
        tp = cc.desplazada(self.COMP, self.PRIM, "t_plus", 0.5)
        tm = cc.desplazada(self.COMP, self.PRIM, "t_minus", 0.5)
        self.assertAlmostEqual(tp[0], 84.5)  # +90° desde +x es −y en [dy, dx] = (−dx, dy)
        self.assertAlmostEqual(tm[0], 85.5)
        for p in (tp, tm):
            self.assertAlmostEqual(math.hypot(p[0] - 85.0, p[1] - 85.0), math.hypot(r, 0.5))

    def test_vectores_unitarios_y_ortogonales(self):
        er, et = cc.vectores((10.0, 3.0), (2.0, -1.0))
        self.assertAlmostEqual(float(np.hypot(*er)), 1.0)
        self.assertAlmostEqual(float(np.dot(er, et)), 0.0)
        with self.assertRaises(ValueError):
            cc.vectores((1.0, 1.0), (1.0, 1.0))

    def test_casos_nombra_desplazamientos_y_trazas(self):
        lista = cc.casos(self.COMP, self.PRIM, {"traza": None, "traza_12": None})
        self.assertEqual(len(lista), 1 + len(cc.DESPLAZAMIENTOS_PX) * len(cc.DIRECCIONES) + 2)
        self.assertEqual(lista["cero"], self.COMP)
        self.assertEqual(lista["traza_12"], ("traza", "traza_12"))
        self.assertAlmostEqual(lista["r_out_1"][1], 132.7)


class TrazaTests(unittest.TestCase):
    @staticmethod
    def _filas(pend_y=1e-4, pend_x=-2e-4, lam0=9000.0):
        edges = np.linspace(4750.0, 9350.0, 7)
        filas = []
        for i in range(6):
            lam = 0.5 * (edges[i] + edges[i + 1])
            filas.append({"wave_min_A": edges[i], "wave_max_A": edges[i + 1],
                          "primary_y": 85.0 + pend_y * (lam - lam0), "primary_x": 85.0 + pend_x * (lam - lam0),
                          "companion_y": 120.0, "companion_x": 40.0})
        return filas

    def test_cero_en_la_banda_de_b3_e_interpolada_entre_bandas(self):
        filas = self._filas()
        tr = cc.traza_relativa(filas, 9075.0, "primary")
        # 9075 Å cae después del último centro (8966.7 Å): np.interp toma el extremo y ahí la traza vale 0
        self.assertEqual(cc.desplazamiento_en(9075.0, tr), (0.0, 0.0))
        lam_ult = 0.5 * (filas[-1]["wave_min_A"] + filas[-1]["wave_max_A"])
        dy, dx = cc.desplazamiento_en(6563.0, tr)
        self.assertAlmostEqual(dy, 1e-4 * (6563.0 - lam_ult), places=10)
        self.assertAlmostEqual(dx, -2e-4 * (6563.0 - lam_ult), places=10)
        _lam, dyc, dxc = cc.traza_relativa(filas, 9075.0, "companion")
        self.assertTrue(np.all(dyc == 0.0) and np.all(dxc == 0.0))

    def test_tramos_siguen_el_bin_del_modelo_de_psf(self):
        wave = np.arange(6470.0, 6660.0, 1.25)
        claves = cc.claves_bin(wave, 100.0)
        # el mismo redondeo que musepipe.psf (`round` de Python: mitad al par)
        self.assertEqual(list(claves), [round(w / 100.0) for w in wave])
        t = cc.tramos(wave, 100.0)
        self.assertEqual(t[0][0], 0)
        self.assertEqual(t[-1][1], wave.size)
        for (a, b), (c, d) in zip(t, t[1:]):
            self.assertEqual(b, c)
        for a, b in t:
            self.assertEqual(len(set(claves[a:b])), 1)
        self.assertEqual(len(t), len(set(claves)))

    def test_tramos_de_caso(self):
        wave = np.arange(6470.0, 6660.0, 1.25)
        filas = self._filas()
        trazas = {"traza": cc.traza_relativa(filas, 9075.0, "primary")}
        comp = (100.0, 50.0)
        acrom = cc.tramos_de_caso((100.3, 50.1), wave, 100.0, comp, trazas)
        self.assertEqual(acrom, [(0, wave.size, (100.3, 50.1))])
        crom = cc.tramos_de_caso(("traza", "traza"), wave, 100.0, comp, trazas)
        self.assertEqual(len(crom), len(cc.tramos(wave, 100.0)))
        for i0, i1, yx in crom:
            dy, dx = cc.desplazamiento_en(float(np.mean(wave[i0:i1])), trazas["traza"])
            self.assertAlmostEqual(yx[0], comp[0] + dy)
            self.assertAlmostEqual(yx[1], comp[1] + dx)


class CentroidesB3Tests(unittest.TestCase):
    """Las funciones de B3 que usa el script, sobre fuentes sintéticas desplazadas."""

    def test_centroid_2d_recupera_una_fuente_desplazada(self):
        from musepipe.stages.stage01c_localize import centroid_2d

        rng = np.random.default_rng(1)
        for verdad in ((40.3, 52.7), (40.0, 52.0), (39.55, 51.45)):
            img = _gauss((80, 100), verdad, 1.3, amp=50.0) + rng.normal(0.0, 0.05, (80, 100))
            (y, x), _err = centroid_2d(img, (round(verdad[0]), round(verdad[1])), stamp_half_size=4, fwhm_px=3.0)
            self.assertLess(abs(y - verdad[0]), 0.02)
            self.assertLess(abs(x - verdad[1]), 0.02)

    def test_chromatic_centroids_sigue_una_traza_lineal(self):
        from musepipe.stages.stage01c_localize import chromatic_centroids

        wave = np.linspace(4750.0, 9350.0, 230)
        pend = np.array([2e-4, -1.5e-4])  # px/Å: ~0.5 px entre Hα y la banda de B3
        prim0, comp0 = np.array([50.0, 50.0]), np.array([52.0, 95.3])
        cubo = np.zeros((wave.size, 100, 140))
        for z, lam in enumerate(wave):
            d = pend * (lam - 9075.0)
            cubo[z] = _gauss((100, 140), prim0 + d, 1.5, amp=1000.0) + _gauss((100, 140), comp0 + d, 1.5, amp=20.0)
        cfg = {"stage01c_chromatic_bins": 6, "stage01c_bad_windows_A": [],
               "stage01c_primary_stamp_half_size": 6, "stage01c_centroid_stamp_half_size": 4}
        filas, _drift = chromatic_centroids(cubo, wave, tuple(prim0), tuple(comp0), cfg, 3.5)
        self.assertEqual(len(filas), 6)
        lam_ref = 9075.0
        for clave in ("primary", "companion"):
            lam, dy, dx = cc.traza_relativa(filas, lam_ref, clave)
            ult = lam[-1]
            esperado_y = pend[0] * (lam - ult)
            esperado_x = pend[1] * (lam - ult)
            np.testing.assert_allclose(dy, esperado_y, atol=0.03)
            np.testing.assert_allclose(dx, esperado_x, atol=0.03)


class FuenteFalsaTests(unittest.TestCase):
    def test_inyecta_pone_la_luz_donde_se_pide_moffat(self):
        doc = _moffat_doc()
        wave = np.linspace(6000.0, 6500.0, 6)
        cubo = np.zeros((wave.size, 60, 60), dtype=np.float32)
        centro = (30.3, 27.6)
        out = cc.inyecta(cubo, wave, doc, np.full(wave.size, 2.0), centro, bin_A=0.0)
        self.assertEqual(out.dtype, np.float32)
        self.assertTrue(np.all(cubo == 0.0))  # no toca la entrada
        for z in range(wave.size):
            y, x = cc.centro_de_luz(out[z])
            self.assertLess(abs(y - centro[0]), 0.02)
            self.assertLess(abs(x - centro[1]), 0.02)
        # solo dentro del sello
        iy, ix = round(centro[0]), round(centro[1])
        fuera = np.ones((60, 60), bool)
        fuera[iy - cc.SEMI_SELLO_PX:iy + cc.SEMI_SELLO_PX + 1, ix - cc.SEMI_SELLO_PX:ix + cc.SEMI_SELLO_PX + 1] = False
        self.assertTrue(np.all(out[:, fuera] == 0.0))

    def test_evaluador_centrado_y_el_de_antes(self):
        pytest.importorskip("maoppy")
        import musepipe.psf as psf

        doc = _psfao_doc()
        yy, xx = np.mgrid[0:81, 0:81].astype(np.float64)
        dy, dx = yy - 40.0, xx - 40.0
        antes = psf.MAOPPY_CENTRADO_PX
        try:
            centrada = psf.evaluate_psf_model(doc, 7000.0, dy, dx)
            y, x = cc.centro_de_luz(centrada)
            self.assertLess(abs(y - 40.0), 0.01)
            self.assertLess(abs(x - 40.0), 0.01)
            # la imagen de la PSF centrada, movida s px hacia −y y −x, para comparar con la de antes
            movidas = {round(s, 3): psf.evaluate_psf_model(doc, 7000.0, dy + s, dx + s)
                       for s in np.arange(0.0, 0.601, 0.01)}
            cc.usa_evaluador_descentrado()
            vieja = psf.evaluate_psf_model(doc, 7000.0, dy, dx)
            s_mejor = min(movidas, key=lambda s: float(np.sum((movidas[s] - vieja) ** 2)))
            # k = 3 por encima de ~6400 Å: la de antes ponía la luz −(k−1)/(2k) = −1/3 px en y y en x
            self.assertLess(abs(s_mejor - 1.0 / 3.0), 0.02)
            self.assertLess(cc.centro_de_luz(vieja)[0], 40.0 - 0.2)
        finally:
            psf.MAOPPY_CENTRADO_PX = antes
            psf._psfao_image_cached.cache_clear()
            psf._mixture_image_cached.cache_clear()
        img = psf.evaluate_psf_model(doc, 7000.0, yy - 40.0, xx - 40.0)
        self.assertLess(abs(cc.centro_de_luz(img)[0] - 40.0), 0.01)


class EstadisticaTests(unittest.TestCase):
    def test_sesgo_sigma_y_rms(self):
        verdad = np.zeros((4, 2))
        medida = np.array([[0.1, 0.0], [-0.1, 0.0], [0.1, 0.2], [-0.1, 0.2]])
        est = cc.estadistica_posicion(verdad, medida)
        self.assertEqual(est["n"], 4)
        np.testing.assert_allclose(est["sesgo_yx"], [0.0, 0.1])
        np.testing.assert_allclose(est["sigma_yx"], [np.std([0.1, -0.1, 0.1, -0.1], ddof=1),
                                                     np.std([0, 0, 0.2, 0.2], ddof=1)])
        self.assertAlmostEqual(est["rms_px"], math.sqrt((0.01 * 4 + 0.04 * 2) / 4))

    def test_filas_no_finitas_fuera(self):
        est = cc.estadistica_posicion([[0, 0], [0, 0]], [[np.nan, np.nan], [0.3, 0.4]])
        self.assertEqual(est["n"], 1)
        self.assertAlmostEqual(est["rms_px"], 0.5)

    def test_clasifica_otra_fuente_y_fallos(self):
        filas = [{"k": 0, "estado": "ok", "verdad_yx": [10.0, 10.0], "medida_yx": [10.03, 9.98], "banda_A": [8800, 9350]},
                 {"k": 1, "estado": "ok", "verdad_yx": [10.0, 10.0], "medida_yx": [10.0, 10.04], "banda_A": [8800, 9350]},
                 {"k": 2, "estado": "ok", "verdad_yx": [10.0, 10.0], "medida_yx": [13.0, 14.0], "banda_A": [7500, 9000]},
                 {"k": 3, "estado": "AmbiguousDetectionError", "verdad_yx": [10.0, 10.0],
                  "medida_yx": [float("nan")] * 2}]
        c = cc.clasifica_inyecciones(filas, 5.0)
        self.assertEqual(c["identificadas"]["n"], 2)
        self.assertEqual(c["n_otra_fuente"], 1)
        self.assertEqual(c["otra_fuente"][0]["k"], 2)
        self.assertAlmostEqual(c["otra_fuente"][0]["distancia_px"], 5.0)
        self.assertEqual(c["fallos"], {"AmbiguousDetectionError": 1})
        self.assertEqual(c["umbral_px"], 2.5)
        self.assertEqual(c["bandas_usadas"], [(7500, 9000), (8800, 9350)])

    def test_anillo_de_inyeccion(self):
        rng = np.random.default_rng(0)
        prim, comp = (85.0, 85.0), (85.0, 131.7)
        evita = [(85.0 + 46.7, 85.0)]  # una fuente a 90°
        anillo = cc.anillo_inyeccion(prim, comp, n=40, excluye_deg=35.0, shape=(170, 170), margen_px=16,
                                     evita=evita, evita_px=10.0, rng=rng)
        self.assertGreater(len(anillo), 20)
        for y, x in anillo:
            r = math.hypot(y - 85.0, x - 85.0)
            self.assertLess(abs(r - 46.7), 0.75)
            ang = abs(math.degrees(math.atan2(y - 85.0, x - 85.0)))
            self.assertGreaterEqual(ang, 35.0 - 1.0)
            self.assertGreaterEqual(math.hypot(y - evita[0][0], x - evita[0][1]), 10.0)


class ProductosTests(unittest.TestCase):
    def test_concatena_tramos(self):
        from musepipe.extraction.product import SpectrumProduct

        def p(w0, n, h):
            w = w0 + np.arange(n, dtype=float)
            return SpectrumProduct(wave_A=w, flux=w * 2, flux_err=np.ones(n), flux_err_emp=np.ones(n) * 2,
                                   apcorr=np.ones(n), npix_eff=np.ones(n) * 9, flags=np.zeros(n, np.int32),
                                   header=dict(h), extra_columns={"bkg": w * 0 + 1})

        a = p(6500.0, 3, {"METHOD": "psffit", "SRCPOS_Y": 1.0})
        b = p(6503.0, 2, {"METHOD": "psffit", "SRCPOS_Y": 2.0})
        c = cc.concatena([a, b])
        np.testing.assert_array_equal(c.wave_A, np.arange(6500.0, 6505.0))
        np.testing.assert_array_equal(c.flux, 2 * np.arange(6500.0, 6505.0))
        self.assertEqual(c.extra_columns["bkg"].size, 5)
        self.assertEqual(c.header["SRCPOS_Y"], 1.0)
        self.assertIs(cc.concatena([a]), a)

    def test_remapea_solo_lo_que_esta_en_la_copia(self):
        from musepipe.paths import RunPaths

        with tempfile.TemporaryDirectory() as tmp:
            rp = RunPaths.from_project_root("SINTETICO", tmp)
            copia = Path(tmp) / "copia"
            (copia / "stages").mkdir(parents=True)
            (copia / "stages" / "psf_model.json").write_text("{}")
            paths = {"paths": rp, "psf_model_json": rp.stage_dir / "psf_model.json",
                     "stage02_cube_fits": rp.stage_dir / "stage02_xcorr_cube_stack.fits", "otra": "texto"}
            out = cc.remapea_si_existe(paths, copia)
            self.assertEqual(out["psf_model_json"], copia / "stages" / "psf_model.json")
            self.assertEqual(out["stage02_cube_fits"], paths["stage02_cube_fits"])
            self.assertEqual(out["otra"], "texto")
            self.assertEqual(cc.remapea_si_existe(paths, None), paths)


class GuardaTests(unittest.TestCase):
    """Sin `--sin-escribir` no corre; una salida dentro de `runs/` para ANTES de calcular nada."""

    def _proyecto(self, tmp):
        tmp = Path(tmp)
        real = tmp / "disco" / "SINTETICO"
        (real / "stages").mkdir(parents=True)
        (tmp / "proyecto" / "runs").mkdir(parents=True)
        os.symlink(real, tmp / "proyecto" / "runs" / "SINTETICO")
        return tmp / "proyecto", real

    def test_sin_sin_escribir_no_corre(self):
        with tempfile.TemporaryDirectory() as tmp:
            raiz, _real = self._proyecto(tmp)
            with self.assertRaises(SystemExit):
                cc.main(["--project-root", str(raiz), "--run-id", "SINTETICO",
                         "--json", str(Path(tmp) / "fuera.json")])
            self.assertFalse((Path(tmp) / "fuera.json").exists())

    def test_salida_dentro_de_runs_para(self):
        with tempfile.TemporaryDirectory() as tmp:
            raiz, real = self._proyecto(tmp)
            for salida in (raiz / "runs" / "SINTETICO" / "stages" / "x.json",  # por el enlace
                           real / "stages" / "x.json",  # el destino del enlace, ya resuelto
                           raiz / "runs" / "otro.json"):
                for opcion in ("--json", "--figura"):
                    with self.assertRaises(RuntimeError):
                        cc.main(["--project-root", str(raiz), "--sin-escribir", "--run-id", "SINTETICO",
                                 opcion, str(salida)])
                    self.assertFalse(Path(salida).exists())
            self.assertEqual(sorted(p.name for p in (real / "stages").iterdir()), [])

    def test_partes_y_metodos_desconocidos(self):
        with tempfile.TemporaryDirectory() as tmp:
            raiz, _real = self._proyecto(tmp)
            for extra in (["--partes", "e1,nada"], ["--metodos", "psffit,box9"]):
                with self.assertRaises(SystemExit):
                    cc.main(["--project-root", str(raiz), "--sin-escribir", "--run-id", "SINTETICO", *extra])


def _raiz_con_runs():
    for raiz in (ROOT, os.environ.get("MUSE_PROJECT_ROOT")):
        if not raiz:
            continue
        try:
            slugs = cc.objetos_del_paper(Path(raiz))
            run = cc.run_del_objetivo(slugs[0], Path(raiz))
        except (SystemExit, IndexError, OSError, KeyError):
            continue
        if (Path(raiz) / "runs" / run / "config" / "config.json").exists():
            return Path(raiz), run
    return None, None


def _foto(carpeta):
    base = Path(carpeta).resolve()
    out = {}
    for dp, dn, fn in os.walk(base):
        out[dp + "/"] = (tuple(sorted(dn)), tuple(sorted(fn)))
        for f in fn:
            st = os.stat(os.path.join(dp, f))
            out[os.path.join(dp, f)] = (st.st_size, st.st_mtime_ns)
    return out


@pytest.mark.slow
class VerificaRunReal(unittest.TestCase):
    """δ = 0 reproduce el disco bit a bit (aperture y psffit, ventana de Hα) y no escribe nada en el run."""

    def test_reproduce_c_d2_y_e1_sin_escribir(self):
        raiz, run = _raiz_con_runs()
        if raiz is None:
            self.skipTest("sin runs/ del paper (ni en este clon ni en MUSE_PROJECT_ROOT)")
        rd = raiz / "runs" / run
        antes = {sub: _foto(rd / sub) for sub in ("stages", "tables", "config")}
        out = cc.mide_extraccion(run, raiz, copia=None, metodos=("aperture", "psffit"), fondo_c4=None,
                                 trazas_b3={}, n_procesos=1, partes=("verifica",), controles_produccion=False)
        v = out["verificacion"]
        for m in ("aperture", "psffit"):
            d = v["productos"][m]
            for clave in ("flux", "flux_err", "flux_err_emp", "controles", "controles_calibrados_d2"):
                self.assertEqual(d[clave], 0.0, f"{m} {clave}")
            for clave, x in v["e1"][m].items():
                self.assertLess(abs(x["dif"]), 1e-12, f"{m} {clave}")
        self.assertEqual(v["veredicto_aqui"]["verdict"], v["veredicto_disco"]["verdict"])
        self.assertEqual(out["fondo_c4"], json.loads((rd / "config" / "config.json").read_text())["config"].get(
            "x03_background", "shared"))
        despues = {sub: _foto(rd / sub) for sub in ("stages", "tables", "config")}
        self.assertEqual(antes, despues)


if __name__ == "__main__":
    unittest.main()
