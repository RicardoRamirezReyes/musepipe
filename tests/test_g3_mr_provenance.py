"""De donde saca G3 la masa y el radio, y cuando se niega a usar los suyos.

La rodaja de acrecion PREFIERE la masa y el radio derivados por el G3 real a los
del config. Eso esta bien cuando el ajuste atmosferico paso sus condiciones de
parada, y es un desastre cuando no: en ROXs 12 b un ajuste con chi2_red = 11.52
(su gate era 3) y A_V pegado al borde de la rejilla dejo `g3_rows_derived.json`
en `stages/`, la rodaja lo consumio como bueno, y el Mdot de cabecera se movio un
31 % -de 1.25e-13 a 1.60e-13- con las tres etapas en rc=0 y sin un solo aviso.
G4 ademas pasaba de `planet_forming` a `brown_dwarf`.

El veredicto viaja en `g3_atmo_fit_qc.json`, que es un fichero aparte y que la
rodaja NO sobrescribe. Ver `docs/2026-09-03_g3_atmosferas_12b.md`.
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from musepipe.stages.stage_g3_accretion import _ajuste_atmosferico_utilizable
from musepipe.stages.stage_g3_assemble import _sella_ajuste_atmosferico


class SelloDelAjusteTests(unittest.TestCase):
    """`_sella_ajuste_atmosferico` deja el veredicto donde el consumidor lo ve."""

    def _qc(self, tmp, **extra):
        path = Path(tmp) / "g3_atmo_fit_qc.json"
        path.write_text(json.dumps({"stage": "g3_atmo_fit", **extra}))
        return path

    def test_sin_stops_queda_utilizable(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self._qc(tmp)
            self.assertTrue(_sella_ajuste_atmosferico(path, []))
            qc = json.loads(path.read_text())
            self.assertEqual(qc["stops"], [])
            self.assertTrue(qc["usable_for_downstream"])

    def test_con_stops_queda_no_utilizable_y_los_conserva(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self._qc(tmp)
            stops = ["chi2_red=11.52 > 3 (atmo fit)", "V2 undeclared 3-sigma edge on axes [\'av\']"]
            self.assertFalse(_sella_ajuste_atmosferico(path, stops))
            qc = json.loads(path.read_text())
            self.assertEqual(qc["stops"], stops)
            self.assertFalse(qc["usable_for_downstream"])

    def test_no_pisa_lo_que_ya_hay(self):
        """El sello anade dos campos; el resto del QC del ajuste sigue igual."""
        with tempfile.TemporaryDirectory() as tmp:
            path = self._qc(tmp, best={"chi2_red": 11.52}, axes={"av_n": 51})
            _sella_ajuste_atmosferico(path, ["chi2_red=11.52 > 3 (atmo fit)"])
            qc = json.loads(path.read_text())
            self.assertEqual(qc["best"], {"chi2_red": 11.52})
            self.assertEqual(qc["axes"], {"av_n": 51})


class GuardaDeLaRodajaTests(unittest.TestCase):
    """La rodaja solo acepta M,R propios de un ajuste que conste que paso."""

    def _stage_dir(self, tmp, qc):
        d = Path(tmp)
        if qc is not None:
            (d / "g3_atmo_fit_qc.json").write_text(json.dumps(qc))
        return d

    def test_sin_ajuste_en_el_run_no_estorba(self):
        """Un run que solo corre la rodaja no tiene ese QC: se sigue como siempre."""
        with tempfile.TemporaryDirectory() as tmp:
            ok, motivo = _ajuste_atmosferico_utilizable(self._stage_dir(tmp, None))
            self.assertTrue(ok)
            self.assertIn("no hay ajuste", motivo)

    def test_ajuste_que_paso_se_usa(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = self._stage_dir(tmp, {"stops": [], "usable_for_downstream": True})
            ok, motivo = _ajuste_atmosferico_utilizable(d)
            self.assertTrue(ok)
            self.assertIn("paso sus condiciones", motivo)

    def test_ajuste_que_fallo_se_rechaza_y_dice_por_que(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = self._stage_dir(tmp, {"stops": ["chi2_red=11.52 > 3 (atmo fit)"],
                                      "usable_for_downstream": False})
            ok, motivo = _ajuste_atmosferico_utilizable(d)
            self.assertFalse(ok)
            self.assertIn("chi2_red=11.52", motivo)

    def test_qc_viejo_sin_sello_se_rechaza(self):
        """Quien no dice que paso sus gates, no los paso que se sepa.

        Es el caso real: los productos de ROXs 12 b son anteriores al sello.
        """
        with tempfile.TemporaryDirectory() as tmp:
            d = self._stage_dir(tmp, {"stage": "g3_atmo_fit", "best": {"chi2_red": 11.52}})
            ok, motivo = _ajuste_atmosferico_utilizable(d)
            self.assertFalse(ok)
            self.assertIn("sin el sello", motivo)

    def test_qc_ilegible_se_rechaza(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            (d / "g3_atmo_fit_qc.json").write_text("{no es json")
            ok, motivo = _ajuste_atmosferico_utilizable(d)
            self.assertFalse(ok)
            self.assertIn("ilegible", motivo)


if __name__ == "__main__":
    unittest.main()
