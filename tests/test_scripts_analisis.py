"""Los tres scripts de analisis que sostienen numeros citados fuera del repo.

`scripts/snr50_pares.py` da el par de SNR50 que esta en el paper; `scripts/
niveles_inyeccion.py`, el 2.9 % de sesgo entre niveles de inyeccion de
`docs/2026-09-18_nivel_de_inyeccion_e4.md`; `scripts/instantanea_run.py` es lo
que permite decir "no se movio nada" despues de re-correr una cadena.

Ninguna prueba toca `runs/`: lo que se fija aqui son las funciones puras y las
DOS regresiones que ya costaron trabajo -leer un espectro que es BINTABLE, y
guardar la lista de issues de F1 y no solo su recuento-.
"""
import importlib.util
import json
import sys
import tempfile
import unittest
import unittest.mock
from pathlib import Path

import numpy as np
from astropy.io import fits

ROOT = Path(__file__).resolve().parents[1]


def _carga(nombre):
    spec = importlib.util.spec_from_file_location(nombre, ROOT / "scripts" / f"{nombre}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[nombre] = mod
    spec.loader.exec_module(mod)
    return mod


class SNR50Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.m = _carga("snr50_pares")

    def test_la_logistica_recupera_sus_parametros(self):
        x = np.log10(np.repeat([0.2, 0.5, 1.0, 2.0, 5.0], 400))
        b0, b1 = -1.0, 4.0
        rng = np.random.default_rng(7)
        y = (rng.random(x.size) < 1 / (1 + np.exp(-(b0 + b1 * x)))).astype(float)
        est = self.m._logistica(x, y)
        self.assertIsNotNone(est)
        self.assertAlmostEqual(est[0], b0, delta=0.25)
        self.assertAlmostEqual(est[1], b1, delta=0.6)

    def test_snr50_es_donde_la_curva_cruza_la_mitad(self):
        # SNR50 = 10**(-b0/b1) = 10**(1/4) = 1.778
        rng = np.random.default_rng(11)
        datos = {}
        for pos in range(40):
            filas = []
            for s in (0.2, 0.5, 1.0, 2.0, 5.0):
                p = 1 / (1 + np.exp(-(-1.0 + 4.0 * np.log10(s))))
                filas.append((s, bool(rng.random() < p)))
            datos[f"control{pos}"] = filas
        self.assertAlmostEqual(self.m.snr50_logistico(datos), 10 ** 0.25, delta=0.25)

    def test_el_empirico_interpola_el_cruce(self):
        # 40 % a s=1 y 60 % a s=2 -> cruce en 1.5
        datos = {f"c{i}": [(1.0, i < 4), (2.0, i < 6)] for i in range(10)}
        self.assertAlmostEqual(self.m.snr50_empirico(datos), 1.5, places=6)

    def test_el_bootstrap_remuestrea_POSICIONES(self):
        # Una sola posicion repetida: cualquier remuestreo de posiciones da la
        # misma muestra, asi que la dispersion tiene que ser exactamente cero.
        # Si remuestreara filas, no lo seria.
        una = [(0.2, False), (0.5, False), (1.0, True), (2.0, True), (5.0, True)]
        datos = {f"c{i}": list(una) for i in range(6)}
        rng = np.random.default_rng(3)
        v = self.m.bootstrap(datos, rng, b=25)
        finitos = v[np.isfinite(v)]
        self.assertGreater(finitos.size, 0)
        # cero salvo el ruido del IRLS (1e-16), no la dispersion de una muestra distinta
        self.assertLess(float(np.nanstd(finitos)), 1e-12)

    def test_main_escribe_su_salida(self):
        # Regresion: el namespace de argparse se llamaba `a` y un `a, bb = ...`
        # dentro del bucle lo pisaba, asi que la tabla se imprimia entera y el
        # fichero NO se escribia (AttributeError en la ultima linea).
        una = [(0.2, False), (0.5, False), (1.0, True), (2.0, True), (5.0, True)]
        falso = {f"c{i}": list(una) for i in range(6)}
        with tempfile.TemporaryDirectory() as d:
            salida = Path(d) / "pares.json"
            with unittest.mock.patch.object(self.m, "carga",
                                            side_effect=lambda run: {"aperture": falso}):
                self.m.main(["--runs", "RUN_A", "RUN_B",
                             "--etiquetas", "A", "B", "--salida", str(salida)])
            self.assertTrue(salida.exists())
            escrito = json.loads(salida.read_text())
        self.assertEqual(escrito["runs"], {"A": "RUN_A", "B": "RUN_B"})
        self.assertIn("aperture", escrito["metodos"])

    def test_main_exige_los_runs(self):
        with self.assertRaises(SystemExit):
            self.m.main(["--salida", "x.json"])


class NivelesTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.m = _carga("niveles_inyeccion")

    def test_la_linea_base_se_resta_por_serie(self):
        filas = [
            {"k": 0.0, "posicion": "c1", "method": "aperture", "recovered_flux": 100.0, "recovered_sigma": 10.0},
            {"k": 1.0, "posicion": "c1", "method": "aperture", "recovered_flux": 160.0, "recovered_sigma": 10.0},
            {"k": 0.0, "posicion": "c2", "method": "aperture", "recovered_flux": -50.0, "recovered_sigma": 10.0},
            {"k": 1.0, "posicion": "c2", "method": "aperture", "recovered_flux": 10.0, "recovered_sigma": 10.0},
        ]
        self.m.resta_linea_base(filas, lambda f: (f["posicion"], f["method"]))
        self.assertEqual([f["delta_flux"] for f in filas], [0.0, 60.0, 0.0, 60.0])
        self.assertEqual(filas[1]["snr_delta"], 6.0)
        # el pedestal de c2 (-50) no contamina: las dos series dan el mismo delta
        self.assertEqual(filas[1]["delta_flux"], filas[3]["delta_flux"])

    def test_invvar_pesa_por_uno_sobre_sigma_al_cuadrado(self):
        grupo = [{"delta_flux": 10.0, "recovered_sigma": 1.0},
                 {"delta_flux": 20.0, "recovered_sigma": 2.0}]
        valor, sigma, n = self.m.combina_invvar(grupo)
        self.assertEqual(n, 2)
        self.assertAlmostEqual(valor, (10 * 1 + 20 * 0.25) / 1.25)
        self.assertAlmostEqual(sigma, (1 / 1.25) ** 0.5)

    def test_invvar_se_niega_con_una_sola_medida(self):
        valor, sigma, n = self.m.combina_invvar([{"delta_flux": 10.0, "recovered_sigma": 1.0}])
        self.assertTrue(np.isnan(valor) and np.isnan(sigma))
        self.assertEqual(n, 1)

    def test_la_banda_incluye_sus_bordes(self):
        wave = np.arange(6500.0, 6600.0, 1.0)
        cube = np.arange(wave.size * 4, dtype=float).reshape(wave.size, 2, 2)
        c, w, _ = self.m._banda(cube, wave, [6520.0, 6530.0])
        self.assertEqual(w[0], 6520.0)
        self.assertEqual(w[-1], 6530.0)
        self.assertEqual(c.shape[0], w.size)


class InstantaneaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.m = _carga("instantanea_run")

    def _escribe_bintable(self, path, flujo):
        wave = np.linspace(6400.0, 6700.0, flujo.size)
        cols = fits.ColDefs([
            fits.Column(name="wave_A", format="D", array=wave),
            fits.Column(name="flux", format="D", array=flujo),
        ])
        hdu = fits.BinTableHDU.from_columns(cols)
        hdu.header["BUNIT"] = "10**(-20)*erg/s/cm**2/Angstrom"
        fits.HDUList([fits.PrimaryHDU(), hdu]).writeto(path, overwrite=True)

    def test_lee_un_espectro_que_es_BINTABLE(self):
        # La regresion del 2026-09-17: abrirlo como imagen da TypeError y la
        # instantanea guarda {"error": ...} en vez del espectro.
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "spec_psffit_object.fits"
            flujo = np.linspace(1.0, 2.0, 400)
            self._escribe_bintable(p, flujo)
            got = self.m.espectro(p)
            self.assertNotIn("error", got)
            self.assertEqual(got["n"], 400)
            self.assertAlmostEqual(got["mediana"], float(np.median(flujo)))
            self.assertTrue(got["halpha_6555_6575"] > 0)

    def test_la_firma_cambia_si_cambia_el_flujo(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "a.fits"
            self._escribe_bintable(p, np.ones(50))
            uno = self.m.espectro(p)["sha256"]
            flujo = np.ones(50)
            flujo[7] += 1e-12
            self._escribe_bintable(p, flujo)
            self.assertNotEqual(uno, self.m.espectro(p)["sha256"])

    def test_lee_tambien_una_imagen_1D(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "viejo.fits"
            hdu = fits.PrimaryHDU(np.linspace(1.0, 2.0, 300))
            hdu.header["CRVAL1"] = 6400.0
            hdu.header["CDELT1"] = 1.0
            hdu.header["CRPIX1"] = 1.0
            hdu.writeto(p)
            got = self.m.espectro(p)
            self.assertNotIn("error", got)
            self.assertEqual(got["n"], 300)

    def test_compara_nombra_el_issue_que_desaparece(self):
        # La otra regresion: guardar solo el recuento dejo un `major` de 42B b
        # sin identificar (24 -> 23).
        antes = {"R": {"espectros": {}, "qc": {"F1": {"overall_status": "red",
                 "prioridades": {"major": 2},
                 "issues": ["major|figures|fig02 no disponible", "major|C1_psf|anillo"]}}}}
        despues = {"R": {"espectros": {}, "qc": {"F1": {"overall_status": "red",
                   "prioridades": {"major": 1},
                   "issues": ["major|C1_psf|anillo"]}}}}
        with tempfile.TemporaryDirectory() as d:
            pa, pd_ = Path(d) / "a.json", Path(d) / "b.json"
            pa.write_text(json.dumps(antes))
            pd_.write_text(json.dumps(despues))
            salida = "\n".join(self.m.compara(pa, pd_))
        self.assertIn("F1 issue que DESAPARECE", salida)
        self.assertIn("fig02 no disponible", salida)


if __name__ == "__main__":
    unittest.main()
