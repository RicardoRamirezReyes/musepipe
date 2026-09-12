"""El trozo del combinado se dimensiona a la memoria, y lo deja escrito.

`DEFAULT_CHUNK_CHANNELS = 128` se eligió cuando esta máquina tenía 62 GB y
`muse_exp_combine` moría. Trocear de más no ahorra memoria -el coste fijo son
los cuatro acumuladores del cubo entero, que no se trocean- y multiplica las
vueltas: 3681 canales en trozos de 128 son 29 pasadas.
"""
import unittest

from musepipe.reduction import stream_combine as sc


class TrozoAutomaticoTests(unittest.TestCase):
    def test_mas_memoria_da_trozo_mayor(self):
        pequena = sc.auto_chunk_channels(
            n_exposures=30, crop_npix=200, n_channels=3681, method="sigclip",
            available_bytes=int(16e9))
        grande = sc.auto_chunk_channels(
            n_exposures=30, crop_npix=200, n_channels=3681, method="sigclip",
            available_bytes=int(80e9))
        self.assertLess(pequena["chunk_channels"], grande["chunk_channels"])

    def test_el_coste_fijo_se_descuenta_antes_del_trozo(self):
        """Los acumuladores no se trocean: si no caben, el trozo no lo arregla."""
        fijo = sc.accumulator_bytes(3681, 200)
        self.assertGreater(fijo, 3e9)
        apretado = sc.auto_chunk_channels(
            n_exposures=30, crop_npix=200, n_channels=3681, method="sigclip",
            available_bytes=fijo)           # justo lo que ocupa el coste fijo
        self.assertEqual(apretado["source"], "floor_no_budget")
        self.assertEqual(apretado["chunk_channels"], sc.MIN_AUTO_CHUNK_CHANNELS)

    def test_nunca_pasa_del_techo_ni_del_cubo(self):
        enorme = sc.auto_chunk_channels(
            n_exposures=2, crop_npix=50, n_channels=100, method="mean",
            available_bytes=int(500e9))
        self.assertLessEqual(enorme["chunk_channels"], 100)      # ni mas que el cubo
        self.assertLessEqual(enorme["chunk_channels"], sc.MAX_AUTO_CHUNK_CHANNELS)

    def test_sin_meminfo_cae_al_valor_historico(self):
        """Sin poder medir, se comporta como antes: reproducible, no optimista."""
        d = sc.auto_chunk_channels(
            n_exposures=30, crop_npix=200, n_channels=3681, method="sigclip",
            available_bytes=None) if sc.available_memory_bytes() is None else None
        if d is None:
            original = sc.available_memory_bytes
            sc.available_memory_bytes = lambda: None
            try:
                d = sc.auto_chunk_channels(
                    n_exposures=30, crop_npix=200, n_channels=3681, method="sigclip")
            finally:
                sc.available_memory_bytes = original
        self.assertEqual(d["source"], "fallback_no_meminfo")
        self.assertEqual(d["chunk_channels"], sc.DEFAULT_CHUNK_CHANNELS)

    def test_sigclip_pide_mas_por_canal_que_mean(self):
        """`mean` procesa una exposicion cada vez; `sigclip` las apila todas."""
        con_pila = sc.chunk_bytes_per_channel(30, 200, "sigclip")
        sin_pila = sc.chunk_bytes_per_channel(30, 200, "mean")
        self.assertGreater(con_pila, 10 * sin_pila)

    def test_el_plan_declara_de_donde_salio_el_trozo(self):
        """Un tamaño que cambia de maquina a maquina sin rastro seria
        irreproducible: el plan guarda el entero Y la cuenta."""
        d = sc.auto_chunk_channels(
            n_exposures=30, crop_npix=200, n_channels=3681, method="sigclip",
            available_bytes=int(80e9))
        for clave in ("available_bytes", "accumulator_bytes", "bytes_per_channel",
                      "memory_fraction", "source", "estimated_peak_bytes"):
            self.assertIn(clave, d)

    def test_el_pico_estimado_cabe_en_lo_que_se_vio(self):
        for disponible in (20e9, 40e9, 80e9, 200e9):
            d = sc.auto_chunk_channels(
                n_exposures=30, crop_npix=200, n_channels=3681, method="sigclip",
                available_bytes=int(disponible))
            if d["source"] != "auto":
                continue
            with self.subTest(disponible=disponible):
                self.assertLess(d["estimated_peak_bytes"], disponible)


if __name__ == "__main__":
    unittest.main()
