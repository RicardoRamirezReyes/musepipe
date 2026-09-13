"""Copia de trabajo de un run para re-correr una etapa sin tocar el original.

Usada el 2026-09-12 para verificar que re-correr D2 con `x11_bkg_selfsub_frac`
declarado no movia un solo espectro
(`docs/2026-09-12_selfsub_declarado_y_la_nota_de_f1.md` §3).

Copia FISICA de todo lo que pesa menos de 20 MB y NADA de symlinks: D2
sobrescribe siete FITS, seis npz, cuatro PNG y su QC, y un enlace escribiria en
el run bueno.
"""
import json, shutil, sys
from pathlib import Path

if len(sys.argv) != 3:
    raise SystemExit("uso: copia_run_para_probar.py <run_origen> <run_copia>")
SRC, DEST = sys.argv[1], sys.argv[2]
LIMITE = 20 * 1024 * 1024
src, dest = Path("runs") / SRC, Path("runs") / DEST
assert src.is_dir() and not dest.exists(), f"{dest} ya existe o falta el origen"
cfg = json.loads((src / "config" / "config.json").read_text(encoding="utf-8"))
assert not (cfg.get("meta") or {}).get("legacy"), "origen legacy: no se copia"

n = saltados = 0
for f in sorted(src.rglob("*")):
    if f.is_dir() or f.is_symlink():
        continue
    if f.stat().st_size >= LIMITE:
        saltados += 1
        continue
    d = dest / f.relative_to(src)
    d.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(f, d)
    n += 1

meta = dict(cfg.get("meta") or {})
meta["copied_from"] = SRC
meta["copied_utc"] = __import__("datetime").date.today().isoformat()
meta["note"] = (
    f"COPIA DE TRABAJO de {SRC}, montada por scripts/copia_run_para_probar.py para re-correr "
    "una etapa y comparar sus productos contra los del run original antes de re-correrlo de "
    "verdad. Sin los cubos: solo lo de menos de 20 MB. NO es un run cientifico."
)
cfg["meta"] = meta
cfg["config"]["run_id"] = DEST
(dest / "config" / "config.json").write_text(
    json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")
print(f"{DEST}: {n} ficheros, {saltados} saltados por tamaño, "
      f"selfsub={cfg['config'].get('x11_bkg_selfsub_frac')}")
