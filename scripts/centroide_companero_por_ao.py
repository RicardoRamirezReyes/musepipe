#!/usr/bin/env python
"""El centroide del Hα del compañero, agrupando las exposiciones de la primera noche por la AO.

Solo lectura. Continúa docs/2026-09-26_desplazamiento_primaria_y_ao_por_exposicion.md: la primaria no
está desplazada en λ en el bloque 2 de la noche buena, y ese bloque es el único tramo de la noche en
que el Strehl que el DRS calcula de la telemetría del RTC se hunde. Aquí se pregunta si el centroide
del compañero sigue al bloque o al Strehl.

Entradas, las dos ya producidas por otros scripts:

  C7_NPZ    `scripts/cifras_por_noche_c7.py extraer RUN C7_NPZ` (compañero y 33 controles, box3)
  COND_JSON `scripts/desplazamiento_primaria_por_exposicion.py RUN PREFIJO` (PREFIJO.json lleva
            las condiciones de cabecera de cada exposición)

Combina por inversa de varianza como `cifras_por_noche_c7.py` (σ de los controles en la banda de
pesos, sin encoger). Centroide = media pesada por la emisión positiva en ±6 Å de 6562.8 Å con el
continuo (mediana en 10-40 Å) restado. Su error sale de los controles: la dispersión del centroide
al sumar a la señal cada control combinado, dividida por √2. La significación de un reparto se mide
por permutación de las exposiciones de la noche, con el mismo tamaño de grupo.

uso: python scripts/centroide_companero_por_ao.py C7_NPZ COND_JSON [--umbral-strehl 12]
                                                  [--permutaciones 5000] [--json SALIDA]
"""
import argparse
import json

import numpy as np

SEMILLA = 20260926
CENTRO_A = 6562.8
MEDIA_VENTANA_A = 6.0
CONTINUO_A = (10.0, 40.0)


class Grupos:
    def __init__(self, npz, cond):
        d = np.load(npz, allow_pickle=False)
        self.ids = [str(x) for x in d["exposure_id"]]
        wave = d["wave"]
        banda = (wave >= d["weight_band"][0]) & (wave <= d["weight_band"][1])
        # σ por exposición de los controles en la banda de pesos: no depende del reparto
        self.sigma = np.asarray([np.nanstd(np.nanmedian(c[:, banda], axis=1)) for c in d["controls"]])
        ventana = np.abs(wave - CENTRO_A) <= CONTINUO_A[1]
        self.wave = wave[ventana]
        dist = np.abs(self.wave - CENTRO_A)
        self.linea = dist <= MEDIA_VENTANA_A
        self.continuo = dist >= CONTINUO_A[0]
        self.flux = d["flux"][:, ventana]
        self.controles = d["controls"][:, :, ventana]
        noche = np.asarray([i.split("_")[0] for i in self.ids])
        orden = np.argsort(d["mjd"])
        self.N1 = [int(i) for i in orden if noche[i] == sorted(set(noche.tolist()))[0]]
        por_id = {f["id"]: f["condiciones"] for f in cond["por_exposicion"]}
        faltan = [i for i in self.ids if i not in por_id]
        if faltan:
            raise SystemExit(f"{len(faltan)} exposiciones sin condiciones en COND_JSON: {faltan[:3]}")
        self.strehl = np.asarray([_num(por_id[i].get("strehl_rtc")) for i in self.ids])
        self.tau0_s = np.asarray([_num(por_id[i].get("tau0_s")) for i in self.ids])

    def combinar(self, idx):
        w = 1.0 / self.sigma[idx] ** 2
        w /= w.sum()
        return (np.nansum(w[:, None] * self.flux[idx], axis=0),
                np.nansum(w[:, None, None] * self.controles[idx], axis=0))

    def centroide(self, espectro):
        y = np.clip(espectro[self.linea] - np.nanmedian(espectro[self.continuo]), 0, None)
        return float(np.sum(self.wave[self.linea] * y) / np.sum(y)) if np.sum(y) > 0 else float("nan")

    def medir(self, idx):
        flux, ctrl = self.combinar(idx)
        c0 = self.centroide(flux)
        ds = [self.centroide(flux + c - np.nanmedian(c[self.continuo])) - c0 for c in ctrl]
        return {"n": len(idx), "centroide_A": c0,
                "error_A": float(np.nanstd(ds, ddof=1) / np.sqrt(2.0)),
                "strehl_mediano": float(np.nanmedian(self.strehl[idx])),
                "tau0_mediano_ms": float(np.nanmedian(self.tau0_s[idx]) * 1e3)}

    def permutacion(self, grupo, n_perm, rng):
        """p de que |Δcentroide| entre `grupo` y el resto de N1 salga igual o mayor al azar."""
        resto = [i for i in self.N1 if i not in grupo]
        obs = self.centroide(self.combinar(grupo)[0]) - self.centroide(self.combinar(resto)[0])
        k = len(grupo)
        n = 0
        for _ in range(n_perm):
            p = rng.permutation(self.N1)
            d = self.centroide(self.combinar(list(p[:k]))[0]) - self.centroide(self.combinar(list(p[k:]))[0])
            n += abs(d) >= abs(obs)
        return float(obs), float((n + 1) / (n_perm + 1))


def _num(x):
    return float(x) if isinstance(x, (int, float)) and not isinstance(x, bool) else float("nan")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("c7_npz")
    ap.add_argument("cond_json")
    ap.add_argument("--umbral-strehl", type=float, default=12.0,
                    help="Strehl del RTC por debajo del cual la exposición es de AO baja (defecto 12)")
    ap.add_argument("--permutaciones", type=int, default=5000)
    ap.add_argument("--json", default=None)
    a = ap.parse_args(argv)
    g = Grupos(a.c7_npz, json.load(open(a.cond_json)))
    rng = np.random.default_rng(SEMILLA)
    N1 = g.N1
    bajo = [i for i in N1 if g.strehl[i] < a.umbral_strehl]
    alto = [i for i in N1 if g.strehl[i] >= a.umbral_strehl]
    bloques = [list(map(int, p)) for p in np.array_split(np.asarray(N1), 3)]
    out = {"umbral_strehl": a.umbral_strehl, "permutaciones": a.permutaciones, "semilla": SEMILLA,
           "strehl_orden_temporal": [float(g.strehl[i]) for i in N1],
           "ao_baja_posiciones": [N1.index(i) + 1 for i in bajo],
           "ao_alta": g.medir(alto), "ao_baja": g.medir(bajo),
           "bloques": [g.medir(b) for b in bloques]}
    out["ao_baja_menos_alta_A"], out["p_ao"] = g.permutacion(bajo, a.permutaciones, rng)
    resto = bloques[0] + bloques[2]
    out["bloque2_menos_resto_A"], out["p_bloque2"] = g.permutacion(bloques[1], a.permutaciones, rng)
    out["r_strehl_tiempo"] = float(np.corrcoef(g.strehl[N1], np.arange(len(N1)))[0, 1])
    for nombre in ("ao_alta", "ao_baja"):
        m = out[nombre]
        print(f"{nombre:8s} n={m['n']:2d}  centroide {m['centroide_A']:.2f} ± {m['error_A']:.2f} Å"
              f"  (Strehl {m['strehl_mediano']:.1f}, τ₀ {m['tau0_mediano_ms']:.1f} ms)")
    for k, m in enumerate(out["bloques"], 1):
        print(f"bloque {k}  n={m['n']:2d}  centroide {m['centroide_A']:.2f} ± {m['error_A']:.2f} Å"
              f"  (Strehl {m['strehl_mediano']:.1f}, τ₀ {m['tau0_mediano_ms']:.1f} ms)")
    print(f"AO baja − alta: {out['ao_baja_menos_alta_A']:+.2f} Å, p = {out['p_ao']:.4f}  "
          f"(posiciones de AO baja: {out['ao_baja_posiciones']})")
    print(f"bloque 2 − resto (n={len(resto)}): {out['bloque2_menos_resto_A']:+.2f} Å, p = {out['p_bloque2']:.4f}")
    print(f"r(Strehl, tiempo) en la noche: {out['r_strehl_tiempo']:+.2f}")
    if a.json:
        with open(a.json, "w") as fh:
            json.dump(out, fh, indent=1)


if __name__ == "__main__":
    main()
