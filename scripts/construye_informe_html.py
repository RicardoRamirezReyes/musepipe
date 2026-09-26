"""Empotra los PNG de un informe en su HTML como data: URIs.

Los Artifacts bloquean imagenes externas por CSP, asi que una figura solo llega al
lector si viaja dentro del fichero. Este script sustituye cada marca
``{{FIG:<nombre>}}`` por la figura correspondiente de ``--fig-dir``, con su pie sacado
del ``manifest.json`` que escribieron los scripts de figuras, para que el texto del pie
no se pueda desincronizar de la figura que describe.

    python scripts/construye_informe_html.py --src <plantilla>.html \\
        --fig-dir reports/20260910/psf --out <salida>.html
"""
from __future__ import annotations

import argparse
import base64
import json
import re
from pathlib import Path

MARCA = re.compile(r"\{\{FIG:([A-Za-z0-9_]+)\}\}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--fig-dir", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    figdir = Path(args.fig_dir)
    manifest = json.loads((figdir / "manifest.json").read_text())
    pies = {f["name"]: f for f in manifest["figures"]}

    usadas, faltan = set(), []

    def _sub(m):
        name = m.group(1)
        info = pies.get(name)
        p = figdir / f"{name}.png"
        if info is None or not p.exists():
            faltan.append(name)
            return f"<!-- figura ausente: {name} -->"
        usadas.add(name)
        b64 = base64.b64encode(p.read_bytes()).decode("ascii")
        pie = info["caption"]
        return (f'<figure>\n'
                f'  <img src="data:image/png;base64,{b64}" alt="{pie[:120]}">\n'
                f'  <figcaption><span class="fignum">{name.split("_")[0]}</span>'
                f'{pie}</figcaption>\n'
                f'</figure>')

    html = MARCA.sub(_sub, Path(args.src).read_text())
    Path(args.out).write_text(html)

    sin_usar = sorted(set(pies) - usadas)
    kb = len(html.encode()) / 1024
    print(f"{len(usadas)} figuras empotradas, {kb:.0f} KB -> {args.out}")
    if faltan:
        print("  AUSENTES (marca sin figura):", ", ".join(faltan))
    if sin_usar:
        print("  generadas pero NO usadas:", ", ".join(sin_usar))
    return 1 if faltan else 0


if __name__ == "__main__":
    raise SystemExit(main())
