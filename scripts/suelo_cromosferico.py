#!/usr/bin/env python
"""El test cromosférico de Manara et al. (2013): ¿está la L_acc de la Hα por encima del ruido cromosférico?

Solo lectura y fuera de la cadena. Manara et al. (2013, A&A 551, A107) calibraron con plantillas Class III
(no acretoras) el nivel de emisión cromosférica que, leído como acreción, daría una L_acc espuria:
log(L_acc,noise/L★) = A · log T_eff + B (`MANARA13`). Alcalá et al. (2017), cuya relación L_acc–L_Hα usa la
cadena, aplica este test y marca como acretoras débiles o dudosas las fuentes que caen en ese nivel o por
debajo. Aquí se aplica a la L_acc de la Hα de cada compañero (la de G3 si hay detección; la de E3, cota, si no),
con la L_bol y las T_eff que declara `targets/<slug>.json` -> `fisica` (con su fuente), y con otras T_eff si se
pasan con `--teff`.

El nivel de ruido se dedujo con relaciones estelares, así que el test usa la L_acc de la relación estelar
(Alcalá): compara luminosidades de línea y no depende de la relación con la que se publique el Ṁ.

La relación se ajustó de K7 a M9.5 (4060–2330 K) y sus autores la dan por aplicable entre 2500 y 4000 K
(`MANARA13["teff_aplicable_k"]`). Por debajo de 2500 K la fila sale marcada `extrapolada`.

uso: python scripts/suelo_cromosferico.py SLUG [--teff K ...] [--lacc L_SOL] [--json SALIDA]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

#: Manara et al. 2013, ec. (2): log(L_acc,noise/L★) = A log T_eff + B, con sus errores y su intervalo
MANARA13 = dict(A=6.17, A_err=0.53, B=-24.54, B_err=1.88, teff_aplicable_k=(2500.0, 4000.0),
                cita="Manara et al. 2013 (A&A 551, A107), eq. 2")


def suelo(teff_k: float) -> float:
    """log(L_acc,noise/L★) a una T_eff."""
    return MANARA13["A"] * math.log10(teff_k) + MANARA13["B"]


def lacc_hal(run_id: str) -> tuple[float, str]:
    """L_acc de la Hα (L_sol) del run, con la relación estelar, y si es medida o cota.

    Detección (G3 sin etiqueta `upper_limit`): la L_acc combinada de G3 (la del Ṁ del paper). No detección: la
    cota de E3 del método canónico de D2, con la relación L_acc–L_Hα de la config (Alcalá 2017), la misma de la
    tabla de acreción."""
    from musepipe.config import load_run_config
    from musepipe.stages.stage_h03_limits import L_SUN_ERG_S, lacc_lsun_from_lha
    rd = ROOT / "runs" / run_id / "stages"
    g3 = json.loads((rd / "stage_g3_qc.json").read_text())
    if g3.get("mdot_label") != "upper_limit":
        return float(g3["combined_accretion"]["l_acc_lsun"]), "detection"
    cfg = load_run_config(run_id, project_root=ROOT).config
    canon = json.loads((rd / "stage_x11_qc.json").read_text())["canonical_method"]
    h3 = json.loads((rd / "stage_h03_qc.json").read_text())
    lim = next(x for x in h3["limits"] if x["method"] == canon)
    lha = float(lim["l_halpha"]) / L_SUN_ERG_S
    return float(lacc_lsun_from_lha(lha, float(cfg["h03_lacc_lha_a"]), float(cfg["h03_lacc_lha_b"]))), "upper_limit"


def teffs_de(fisica: dict, extra=()) -> dict[str, float]:
    """Las T_eff a evaluar: la principal de `fisica`, sus alternativas declaradas y las de la línea de órdenes."""
    teffs = {str(fisica["teff_fuente"]): float(fisica["teff_k"])}
    teffs.update({str(k): float(v) for k, v in (fisica.get("teff_alternativas_k") or {}).items()})
    teffs.update({f"--teff {t:.0f}": float(t) for t in extra})
    return teffs


def evalua(log_lacc_lbol: float, teffs: dict) -> list[dict]:
    """Margen sobre el suelo para cada T_eff (positivo = por encima del nivel cromosférico)."""
    lo, hi = MANARA13["teff_aplicable_k"]
    return [dict(etiqueta=k, teff_k=float(t), suelo=round(suelo(t), 3), margen_dex=round(log_lacc_lbol - suelo(t), 3),
                 extrapolada=not (lo <= float(t) <= hi))
            for k, t in teffs.items()]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("slug")
    ap.add_argument("--teff", type=float, nargs="*", default=[], help="T_eff adicionales (K)")
    ap.add_argument("--lacc", type=float, help="L_acc de la Hα (L_sol); por defecto, la del run del paper (`lacc_hal`)")
    ap.add_argument("--json", type=Path)
    a = ap.parse_args(argv)
    from musepipe.targets import load_target
    perfil = load_target(a.slug, project_root=ROOT)
    fis = (perfil or {}).get("fisica")
    if not fis:
        raise SystemExit(f"{a.slug}: targets/ sin 'fisica' (L_bol y T_eff con su fuente)")
    tipo = "dada"
    lacc = a.lacc
    if lacc is None:
        lacc, tipo = lacc_hal(perfil["paper"]["run"])
    log_ratio = math.log10(lacc) - float(fis["log_lbol"])
    filas = evalua(log_ratio, teffs_de(fis, a.teff))
    out = dict(slug=a.slug, lacc_lsun=lacc, lacc_tipo=tipo, log_lbol=fis["log_lbol"],
               log_lbol_fuente=fis["log_lbol_fuente"], log_lacc_lbol=round(log_ratio, 3), relacion=MANARA13, filas=filas)
    rel = "<" if tipo == "upper_limit" else "="
    print(f"{a.slug}: log(L_acc/L_bol) {rel} {log_ratio:.2f}  (L_acc {lacc:.2e} L_sol, {tipo}; log L_bol {fis['log_lbol']}, {fis['log_lbol_fuente']})")
    for f in filas:
        nota = "  [extrapolada: fuera de 2500-4000 K]" if f["extrapolada"] else ""
        print(f"   T_eff {f['teff_k']:.0f} K ({f['etiqueta']}): suelo {f['suelo']:.2f} → margen {f['margen_dex']:+.2f} dex{nota}")
    if a.json:
        a.json.write_text(json.dumps(out, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
