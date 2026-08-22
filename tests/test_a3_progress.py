import json
import tempfile
import time
import unittest
from pathlib import Path

from musepipe.reduction.progress import Cronometro, Ledger, bloque_timing, params_hash


class ParamsHashTests(unittest.TestCase):
    def test_the_order_of_the_keys_is_not_a_different_recipe(self):
        self.assertEqual(params_hash({"a": 1, "b": 2}), params_hash({"b": 2, "a": 1}))

    def test_a_changed_value_is_a_different_recipe(self):
        self.assertNotEqual(params_hash({"WLC_CONST": "0"}), params_hash({"WLC_CONST": "-0.05"}))

    def test_it_survives_a_path_or_a_numpy_scalar(self):
        # Sin `default=str` esto revienta, y el hash se calcula sobre payloads que
        # llevan rutas de verdad.
        params_hash({"cube": Path("/tmp/x.fits"), "radius": 8.0})


class LedgerTests(unittest.TestCase):
    """Las cuatro condiciones para saltarse una unidad, y qué pasa si falla una."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.producto = self.dir / "producto.fits"
        self.producto.write_text("x")

    def tearDown(self):
        self.tmp.cleanup()

    def _ledger(self):
        return Ledger(self.dir / "a3_progress.json")

    def test_a_recorded_unit_is_done(self):
        led = self._ledger()
        led.anota("fit:exp1", rc=0, segundos=42.0, producto=self.producto, params="abc")

        self.assertTrue(self._ledger().hecha("fit:exp1", params="abc"))

    def test_a_different_parameter_hash_is_not_done(self):
        led = self._ledger()
        led.anota("fit:exp1", rc=0, segundos=1.0, producto=self.producto, params="abc")

        # Cambiar un knob y reanudar NO puede devolver el resultado viejo.
        self.assertFalse(self._ledger().hecha("fit:exp1", params="otro"))

    def test_a_missing_product_is_not_done(self):
        led = self._ledger()
        led.anota("fit:exp1", rc=0, segundos=1.0, producto=self.producto, params="abc")
        self.producto.unlink()

        self.assertFalse(self._ledger().hecha("fit:exp1", params="abc"))

    def test_a_product_that_went_back_in_time_is_not_done(self):
        led = self._ledger()
        led.anota("fit:exp1", rc=0, segundos=1.0, producto=self.producto, params="abc")
        # Restaurar un snapshot devuelve el mtime hacia atras: la unidad vuelve a la cola.
        import os
        marca = self.producto.stat().st_mtime - 3600
        os.utime(self.producto, (marca, marca))

        self.assertFalse(self._ledger().hecha("fit:exp1", params="abc"))

    def test_a_failed_unit_is_not_done(self):
        led = self._ledger()
        led.anota("fit:exp1", rc=2, segundos=1.0, producto=self.producto, params="abc")

        self.assertFalse(self._ledger().hecha("fit:exp1", params="abc"))

    def test_the_last_row_wins(self):
        led = self._ledger()
        led.anota("fit:exp1", rc=2, segundos=1.0, params="abc")
        led.anota("fit:exp1", rc=0, segundos=2.0, producto=self.producto, params="abc")

        self.assertTrue(self._ledger().hecha("fit:exp1", params="abc"))
        self.assertEqual(len(self._ledger().rows), 2, "la bitacora se acumula, no se pisa")

    def test_it_flushes_on_every_unit(self):
        led = self._ledger()
        led.anota("uno", rc=0, segundos=1.0)

        # Una bitacora que solo se escribe al terminar no sirve para reanudar nada.
        cargado = json.loads((self.dir / "a3_progress.json").read_text())
        self.assertEqual(len(cargado), 1)

    def test_a_truncated_ledger_redoes_the_work_instead_of_reading_garbage(self):
        (self.dir / "a3_progress.json").write_text("[{'roto': ")

        self.assertEqual(self._ledger().rows, [])

    def test_the_stop_file_is_seen(self):
        led = self._ledger()
        self.assertFalse(led.parar())
        (self.dir / "PARAR").write_text("")
        self.assertTrue(led.parar())


class TimingTests(unittest.TestCase):
    def test_the_block_estimates_what_is_left(self):
        filas = [{"clave": "a", "segundos": 40.0}, {"clave": "b", "segundos": 60.0}]

        bloque = bloque_timing(filas, unidades_totales=5)

        self.assertEqual(bloque["segundos_totales"], 100.0)
        self.assertEqual(bloque["segundos_por_unidad"], 50.0)
        self.assertEqual(bloque["segundos_restantes_estimados"], 150.0)

    def test_it_counts_units_not_retries(self):
        # La bitacora se acumula entre relanzamientos: contar filas contaba
        # reintentos, y el bloque publicaba mas unidades hechas que totales.
        filas = [{"clave": "a", "segundos": 10.0}, {"clave": "a", "segundos": 30.0},
                 {"clave": "b", "segundos": 50.0}]

        bloque = bloque_timing(filas, unidades_totales=2)

        self.assertEqual(bloque["n_unidades_hechas"], 2)
        self.assertEqual(bloque["segundos_totales"], 80.0, "gana la ultima fila de cada clave")
        self.assertNotIn("segundos_restantes_estimados", bloque)

    def test_a_finished_phase_has_nothing_left(self):
        bloque = bloque_timing([{"clave": "a", "segundos": 10.0}], unidades_totales=1)

        self.assertNotIn("segundos_restantes_estimados", bloque)

    def test_the_stopwatch_measures_something(self):
        with Cronometro() as reloj:
            time.sleep(0.01)

        self.assertGreater(reloj.segundos, 0.005)


if __name__ == "__main__":
    unittest.main()
