#!/usr/bin/env python
"""Las cifras de extincion del texto del paper, desde los productos.

Hermano de `paper_tables.py` y `paper_figures.py`, con la misma regla: lee lo
que las etapas escribieron y no inventa entradas. Imprime, por objeto y para el
par, las cifras que la prosa cita y que antes se copiaban a mano de un doc:

- la cota INFERIOR de A_V que da Hbeta no detectada frente a Halpha (Balmer);
- la cota SUPERIOR que dan las Paschen no detectadas frente a Halpha;
- los contrastes del par: en flujo de linea y en Mdot, con el MISMO estimador
  (el filtro adaptado) en los dos lados, y la extincion diferencial que anula
  el de Mdot; y el cociente de los Mdot de la Tabla 5, con su desglose.

Las dos cotas usan el MISMO flujo de Halpha, el de integracion directa de G2,
que es el estimador de los limites a 5 sigma de G2 contra los que se compara.
Las razones intrinsecas son las del caso B de Storey & Hummer (1995), en el
nodo T = 10^4 K, n_e = 10^2 cm^-3 (Halpha/Hbeta = 2.863), y se barre
T = 5000-20000 K, n_e = 10^2-10^6 cm^-3 para declarar cuanto dependen de eso.
Una Paschen cuya ventana de G2 se solapa con la de otra linea del catalogo
(el triplete de Ca II) no se usa: su limite no es solo de Paschen.

Hasta el 2026-10-12 la cota de Paschen del paper («15-21 mag») no salia de
ningun codigo (`docs/2026-10-12_e4_extractores_de_produccion.md`).

Desde G2 v4 (`docs/spec_G2_v4_codex_spectral_measurements.md`) el limite citable es el
del 99 % del filtro adaptado (`flux_upper_limit_99`, la regla de E3 v2), con la escala de
la nula que manda (controles, placebos o el presupuesto propagado). Si la tabla de G2 es
de antes de la v4 y no lo trae, se usa el de 5 sigma de integracion directa, como hasta
ahora, y se dice cual (`limite`). El flujo de Halpha sigue siendo el de integracion
directa de G2: es el mayor de los dos estimadores, asi que las dos cotas salen por el
lado menos restrictivo.

    python scripts/paper_cifras_extincion.py
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from musepipe.models.case_b import (  # noqa: E402
    av_lower_limit_undetected_bluer,
    av_upper_limit_undetected_redder,
    case_b_ratio,
)
from musepipe.models.extinction import CCMExtinction  # noqa: E402
from scripts.paper_figures import _cadena_mdot, _entrada_extincion, objetos_del_paper  # noqa: E402

NODO = (1.0e4, 1.0e2)
BARRIDO_T = (5000.0, 7500.0, 10000.0, 12500.0, 15000.0, 20000.0)
BARRIDO_NE = (1e2, 1e3, 1e4, 1e5, 1e6)


def _ventana(fila):
    lo, hi = (float(v) for v in str(fila["line_window_A"]).split(";"))
    return lo, hi


def _solapa(fila, otras):
    lo, hi = _ventana(fila)
    return [o["name"] for o in otras if not (_ventana(o)[1] < lo or _ventana(o)[0] > hi)]


def _limite(fila):
    """`(limite, cual)`: el del 99 % de G2 v4 si la tabla lo trae; si no, el de 5 sigma."""
    v = fila.get("flux_upper_limit_99")
    if v not in (None, "") and math.isfinite(float(v)):
        return float(v), "99% (G2 v4, filtro adaptado)"
    return float(fila["flux_upper_limit_5sigma"]), "5 sigma (G2 v3, integracion directa)"


def cotas_de_lineas(obj, ccm):
    g2 = {r["name"]: r for r in obj.filas("g2_line_measurements.csv")}
    escala = float(obj.config["h03_flux_unit_cgs"])
    ha = g2["Halpha"]
    if ha["status"] != "detected":
        return {"objeto": obj.nombre, "nota": f"Halpha {ha['status']}: sin cota de lineas"}
    f_ha = float(ha["flux_direct"])
    k_ha = ccm.a_lambda_over_av(float(ha["rest_A"]))
    out = {"objeto": obj.nombre, "f_ha_direct_cgs": f_ha * escala}

    hb = g2["Hbeta"]
    lim_hb, cual = _limite(hb)
    r_hb = case_b_ratio("Halpha", "Hbeta", temperature_K=NODO[0], density_cm3=NODO[1])
    # el tiempo para una deteccion a 5 sigma: con la sigma empirica del filtro adaptado
    # (G2 v4) o, en una tabla anterior, con el limite de 5 sigma, que ya es 5 sigma.
    s_mf = hb.get("flux_mf_sigma_emp")
    cinco = (5.0 * float(s_mf) if s_mf not in (None, "") and math.isfinite(float(s_mf))
             else float(hb["flux_upper_limit_5sigma"]))
    out["balmer"] = {
        "limite": cual,
        "limite_cgs": lim_hb * escala,
        "limite_sobre_ha": lim_hb / f_ha,
        "razon_caso_b": r_hb,
        "factor_bajo_el_limite": lim_hb / (f_ha / r_hb),
        "tiempo_para_detectar_5sigma": (cinco / (f_ha / r_hb)) ** 2,
        "av_min": av_lower_limit_undetected_bluer(
            f_ha, lim_hb, r_hb, k_ha, ccm.a_lambda_over_av(float(hb["rest_A"]))),
    }

    paschen = [r for n, r in g2.items() if n.startswith("Pa ")]
    otras = [r for n, r in g2.items() if not n.startswith(("Pa ", "Halpha", "Hbeta"))
             and r.get("line_window_A")]
    por_linea = {}
    for fila in sorted(paschen, key=lambda r: float(r["rest_A"])):
        nombre = fila["name"]
        if fila["status"] == "detected":
            por_linea[nombre] = {"excluida": "detectada"}
            continue
        mezcla = _solapa(fila, otras)
        if mezcla:
            por_linea[nombre] = {"excluida": "ventana solapada con " + ", ".join(mezcla)}
            continue
        k_pa = ccm.a_lambda_over_av(float(fila["rest_A"]))
        lim, cual = _limite(fila)
        barrido = [av_upper_limit_undetected_redder(
            f_ha, lim, case_b_ratio("Halpha", nombre, temperature_K=t, density_cm3=d), k_ha, k_pa)
            for t in BARRIDO_T for d in BARRIDO_NE]
        por_linea[nombre] = {
            "limite": cual,
            "limite_cgs": lim * escala,
            "razon_caso_b": case_b_ratio("Halpha", nombre, temperature_K=NODO[0], density_cm3=NODO[1]),
            "av_max": av_upper_limit_undetected_redder(
                f_ha, lim, case_b_ratio("Halpha", nombre, temperature_K=NODO[0], density_cm3=NODO[1]),
                k_ha, k_pa),
            "av_max_barrido": (min(barrido), max(barrido)),
        }
    out["paschen"] = por_linea
    return out


def contrastes(objetos):
    a, b = (_entrada_extincion(o) for o in objetos)
    if a["limite"] or not b["limite"]:
        raise SystemExit("el par se compara como deteccion (primero) contra limite (segundo)")
    pa, pb = a["phys"], b["phys"]
    pend = 0.4 * pa["lacc_lha_a"] * pa["a_halpha_over_av"]
    mdot_a_mf = _cadena_mdot(pa, a["f_mf"], pa["av"])[2]
    mdot_b = b["mdot_pub"]
    c_flujo = math.log10(a["f_mf"] / b["f_mf"])
    c_mdot_mf = math.log10(mdot_a_mf / mdot_b)
    c_tabla = math.log10(a["mdot_pub"] / mdot_b)
    # desglose del contraste de Mdot con el filtro adaptado: cada termino de la
    # cadena por separado (log Mdot = a log L_Ha + b + log R - log M + cte).
    desglose = {
        "flujo (filtro adaptado)": c_flujo,
        "pendiente a de la relacion": (pa["lacc_lha_a"] - 1.0) * c_flujo,
        "distancia": pa["lacc_lha_a"] * 2.0 * math.log10(pa["distance_pc"] / pb["distance_pc"]),
        "A_V adoptada": pa["lacc_lha_a"] * 0.4 * pa["a_halpha_over_av"] * (pa["av"] - pb["av"]),
        "radio": math.log10(pa["companion_radius_rsun"] / pb["companion_radius_rsun"]),
        "masa": -math.log10(pa["companion_mass_msun"] / pb["companion_mass_msun"]),
    }
    return {
        "pendiente_dex_por_mag": pend,
        "contraste_flujo_dex": c_flujo,
        "mdot_a_filtro_adaptado": mdot_a_mf,
        "mdot_b_limite": mdot_b,
        "contraste_mdot_filtro_adaptado_dex": c_mdot_mf,
        "factor_mdot_filtro_adaptado": 10 ** c_mdot_mf,
        "av_diferencial_que_lo_anula": c_mdot_mf / pend,
        "mdot_a_tabla5": a["mdot_pub"],
        "contraste_mdot_tabla5_dex": c_tabla,
        "factor_mdot_tabla5": 10 ** c_tabla,
        "desglose_filtro_adaptado": desglose,
        "suma_desglose": sum(desglose.values()),
    }


def _imprime(d, sangria=""):
    for k, v in d.items():
        if isinstance(v, dict):
            print(f"{sangria}{k}:")
            _imprime(v, sangria + "  ")
        elif isinstance(v, float):
            print(f"{sangria}{k}: {v:.6g}")
        elif isinstance(v, tuple):
            print(f"{sangria}{k}: " + " - ".join(f"{x:.4g}" for x in v))
        else:
            print(f"{sangria}{k}: {v}")


def main(argv=None) -> int:
    objetos = objetos_del_paper()
    ccm = CCMExtinction(3.1, citation="Cardelli, Clayton & Mathis 1989")
    print(f"% caso B: Storey & Hummer (1995), T = {NODO[0]:g} K, n_e = {NODO[1]:g} cm^-3")
    for obj in objetos:
        print(f"% ---- {obj.nombre} ({obj.run_id}) ----")
        _imprime(cotas_de_lineas(obj, ccm))
    print(f"% ---- par ({', '.join(o.run_id for o in objetos)}) ----")
    _imprime(contrastes(objetos))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
