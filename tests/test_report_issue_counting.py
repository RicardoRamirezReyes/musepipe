"""F1 no puede contar dos veces el mismo problema.

`_stage_status` construye el `summary` de una etapa uniendo el texto de sus
`open_issues`. Si ademas se anade ese resumen como un issue mas, cada problema de
una etapa roja aparece dos veces: una suelto y otra dentro del resumen. El
veredicto no cambia -basta un blocking para bloquear- pero el NUMERO es el que se
cita en los traspasos y en las revisiones.

Medido el 2026-08-29: los dos objetos declaraban 8 `blocking` de los que 2 eran
resumenes, o sea 6 problemas distintos.
"""

import unittest

from musepipe.report import aggregate_open_issues


class AggregateOpenIssuesTests(unittest.TestCase):
    def test_una_etapa_roja_con_issues_no_suma_su_resumen(self):
        filas = [{"stage": "C1_psf", "status": "red", "summary": "a; b"}]
        qc = {"C1_psf": {"qc": {"open_issues": ["a", "b"]}}}
        out = aggregate_open_issues(filas, qc)
        self.assertEqual([i["issue"] for i in out], ["a", "b"])

    def test_una_etapa_roja_SIN_issues_conserva_su_resumen(self):
        # Es el unico registro de por que esta roja: quitarlo la haria invisible.
        filas = [{"stage": "C2_aperture", "status": "red", "summary": "check v4 fail"}]
        qc = {"C2_aperture": {"qc": {"open_issues": []}}}
        out = aggregate_open_issues(filas, qc)
        self.assertEqual([(i["stage"], i["issue"], i["priority"]) for i in out],
                         [("C2_aperture", "check v4 fail", "blocking")])

    def test_el_resumen_de_una_etapa_no_tapa_el_de_otra(self):
        filas = [{"stage": "C1_psf", "status": "red", "summary": "a"},
                 {"stage": "C2_aperture", "status": "red", "summary": "solo resumen"}]
        qc = {"C1_psf": {"qc": {"open_issues": ["a"]}}, "C2_aperture": {"qc": {"open_issues": []}}}
        out = aggregate_open_issues(filas, qc)
        self.assertEqual(sorted(i["issue"] for i in out), ["a", "solo resumen"])

    def test_una_etapa_verde_con_notas_no_gana_un_blocking(self):
        filas = [{"stage": "D1_compare", "status": "yellow", "summary": "nota"}]
        qc = {"D1_compare": {"qc": {"open_issues": [{"issue": "nota", "priority": "major"}]}}}
        out = aggregate_open_issues(filas, qc)
        self.assertEqual([i["priority"] for i in out], ["major"])


if __name__ == "__main__":
    unittest.main()
