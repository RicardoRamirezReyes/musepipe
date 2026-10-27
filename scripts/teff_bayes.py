#!/usr/bin/env python
"""T_eff de los compañeros por muestreo anidado (dynesty) con BT-Settl o DRIFT-PHOENIX sobre MUSE + J/H/K.

Solo lectura y fuera de la cadena (pedido del autor 2026-10-14: «adelante con la MCMC, instala dynesty», y la
segunda rejilla, «adelante con DRIFT-PHOENIX»). Da la T_eff, el tipo equivalente y el margen sobre el suelo
cromosférico del apéndice B del paper (`docs/2026-10-14_teff_muestreo_anidado.md`). Reutiliza los segmentos, las
escalas con prior y la preparación de modelos de `scripts/ajuste_conjunto.py` (R = 500, máscaras de telúrico y de
H2O), pero en vez de una rejilla con polinomios usa un modelo FÍSICO con flujo absoluto:

    F_k(λ) = s_k · (R/d)² · [F_mod(λ; T_eff, log g) + r_K · F_mod(2.2 µm) · B_λ(T_d)/B_2.2(T_d)] · 10^(−0.4 A_λ/A_V · A_V)

- F_mod es flujo de superficie (comprobado: BT-Settl ∫ 0.4–2.55 µm = 0.83 σT⁴ a 2500 K; DRIFT-PHOENIX sobre su
  rango completo, 0.97–1.00 σT⁴), interpolado bilineal en log F sobre la rejilla (`REJILLAS`). El nodo de
  DRIFT-PHOENIX que SVO no tiene (2500 K, log g 3.0) se rellena con la media en log F de sus vecinos en T_eff.
- s_k: escala por segmento con el prior de `ajuste_conjunto` (N(1, 0.05) en las bandas ancladas a la
  fotometría, N(1, 0.20) en MUSE, log-uniforme 0.2–5 en las no ancladas).
- Error: σ² = e² + (f_b · F)², con un f_b libre por banda (vis, J, H, K): el desajuste del modelo, que con
  S/N 50–170 en el IR domina sobre el ruido.

Modos de `ajusta`:
- `libre`: T_eff, log g y R libres (lo que dice el espectro solo). A_V con el prior de la primaria (`--av
  primaria`), libre en 0–8 (`--av libre`) o fija (`--av-fija`, el perfil en A_V).
- `evo`: (edad, L_bol) → tracks (BHAC15 o ATMO2020) → T_eff, R, log g, M. La edad con el prior de la
  literatura (`g3_age_myr`, log-normal); L_bol plana en −4.5…−1.5, de modo que la fija el flujo absoluto del
  espectro y no la L_bol publicada.
- `evo_solo` (sin espectro): la L_bol publicada (`targets` → `fisica`) y el prior de edad por los tracks:
  la T_eff «evolutiva», la referencia contra la que se mide lo que aporta el espectro.

Pruebas: `--inyecta-bt T LOGG AV R_RJ` sustituye los datos por un modelo de la rejilla con la S/N real;
`--plantilla NOMBRE` por una plantilla empírica de tipo conocido (normalizada con UN solo factor, así que
conserva sus colores) enrojecida con `--av-inyectada` (por defecto la A_V de la primaria): la calibración que
traduce la T_eff de cada rejilla a un tipo equivalente.

`--datos muse` o `--datos ir` ajusta solo el espectro de MUSE o solo los J/H/K (pedido del autor 2026-10-16, para
ver lo que aporta cada uno): los mismos píxeles, escalas y priores que el ajuste conjunto, sin los segmentos del
otro brazo; sin ningún segmento IR, r_K sale de los parámetros (no tendría datos). Esas corridas no entran en la
tabla del paper (`_de_la_tabla`).

`--rejilla` admite, además de las dos del paper, las del mismo pedido («probar con más modelos»): AMES-Dusty,
AMES-Cond y BT-Dusty (PHOENIX con otro trato del polvo), Sonora Diamondback por fsed (otro código, con nubes, hasta
2400 K; desde 2000 K o, `diamondback1600_<fsed>`, desde 1600 K) y SPHINX (otro código, sin nubes). SPHINX está
muestreado a λ/Δλ = 250, así que su ajuste va en el modo R = 250 (`--r250`, automático para ella): los píxeles de
los datos se agrupan de a dos y el modelo se usa a su resolución. `--r250` con una rejilla de resolución alta es
el control que separa el efecto de la resolución del de la rejilla. Estas corridas tampoco entran en la tabla de
las cadenas: van a la de rejillas (`rejillas`, Tabla B.7 del paper, pedida el 2026-10-18).

`resume` lee los json/npz de uno o varios directorios de `ajusta` y da, por corrida, la T_eff y el margen
coherente sobre el suelo cromosférico de Manara et al. (2013, Ec. 2): por muestra, log(L_acc/L_bol) − suelo(T_eff),
con la L_acc del paper reescalada a la A_V de la muestra y la L_bol de la muestra (4πR²σT⁴; en `evo_solo`, la A_V
adoptada y la L_bol publicada). El suelo se toma sin su incertidumbre: sus autores no publican la covarianza
de los coeficientes.

uso: python scripts/teff_bayes.py ajusta SLUG [--rejilla btsettl|drift|ames_dusty|...] [--modo libre|evo|evo_solo]
                                       [--datos todo|muse|ir] [--r250]
                                       [--tracks ATMO2020] [--av libre|primaria] [--av-fija AV] [--nlive 500]
                                       [--semilla 0] [--inyecta-bt ...] [--plantilla ...] --salida DIR
     python scripts/teff_bayes.py resume DIR [DIR ...] [--json SALIDA]
     python scripts/teff_bayes.py tabla DIR [DIR ...]      # cuerpo LaTeX de la tabla de las cadenas (ap. B)
     python scripts/teff_bayes.py rejillas DIR [DIR ...]   # cuerpo LaTeX de la tabla de rejillas (ap. B)

El corner plot de las mismas cadenas lo dibuja `scripts/paper_figures.py --figure teff_corner --teff-chains DIR`;
el de solo MUSE / solo IR, `--figure teff_corner_data`; los de todas las cadenas, `scripts/teff_suplemento.py`.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import sys
import time
import warnings
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from musepipe.constants import LSUN_ERG_S, PC_CM, RJUP_CM, RSUN_CM, SIGMA_SB_CGS, spt_code  # noqa: E402
from musepipe.models.extinction import CCMExtinction  # noqa: E402

#: rejillas de atmósferas: directorio de npz (formato de `musepipe.models.cache`) y ejes (inicio, paso, n).
#: DRIFT-PHOENIX es la segunda rejilla, para el sistemático de modelo; ATMO 2020 se descartó porque sus autores
#: la dan por válida solo bajo 2000 K (`/mnt/2TB/IR_Data/atmo2020/README.md`). BT-Settl llega a log g 5.0 desde
#: 2026-10-17 (decisión del autor): con 2.5–4.5, la posterior de 12 b tocaba el techo de 4.5.
REJILLAS = {"btsettl": dict(dir=Path("/mnt/2TB/IR_Data/btsettl_cifist/npz"), teff=(2000.0, 100.0, 13),
                            logg=(2.5, 0.5, 6), nombre="BT-Settl CIFIST2011"),
            "drift": dict(dir=Path("/mnt/2TB/IR_Data/drift_phoenix/npz"), teff=(2000.0, 100.0, 11),
                          logg=(3.0, 0.5, 7), nombre="DRIFT-PHOENIX")}
#: más rejillas (pedido del autor 2026-10-16, reunión con Claudio: «probar con más modelos»). Las tres de SVO son
#: PHOENIX con otro trato del polvo: AMES-Dusty (en equilibrio, sin sedimentar), AMES-Cond (sin opacidad de polvo)
#: y BT-Dusty, que a [M/H] = 0 solo tiene log g < 4.5 desde 2600 K (sirve para 12 b, no para 42B b). Sonora
#: Diamondback es otro código, con nubes (una rejilla por fsed; «nc» sin nubes), y llega solo a 2400 K (42B b).
REJILLAS.update({
    "ames_dusty": dict(dir=Path("/mnt/2TB/IR_Data/ames_dusty/npz"), teff=(2000.0, 100.0, 11),
                       logg=(3.5, 0.5, 6), nombre="AMES-Dusty"),
    "ames_cond": dict(dir=Path("/mnt/2TB/IR_Data/ames_cond/npz"), teff=(2000.0, 100.0, 13),
                      logg=(2.5, 0.5, 8), nombre="AMES-Cond"),
    "bt_dusty": dict(dir=Path("/mnt/2TB/IR_Data/bt_dusty/npz"), teff=(2600.0, 100.0, 7),
                     logg=(3.0, 0.5, 6), nombre="BT-Dusty"),
    **{f"diamondback_{f}": dict(dir=Path(f"/mnt/2TB/IR_Data/sonora_diamondback/npz/{f}"), teff=(2000.0, 100.0, 5),
                                logg=(3.5, 0.5, 5), nombre=f"Sonora Diamondback ({'sin nubes' if f == 'nc' else 'f_sed = ' + f[1:]})")
       for f in ("f1", "f2", "f3", "f4", "f8", "nc")},
    # la misma desde 1600 K: con 2000–2400 K, 42B b quedaba contra el borde de 2000 K (f1–f3, 2026-10-17). En
    # `npz_1600/` los nodos nublados de 1600–1800 K entran con ∫F/σT⁴ hasta 1.17 (el exceso crece hacia log g alto;
    # en log g 3.5–4.0, ≤ 1.10); el resto son enlaces a `npz/`, que guarda solo los de ±8 %
    **{f"diamondback1600_{f}": dict(dir=Path(f"/mnt/2TB/IR_Data/sonora_diamondback/npz_1600/{f}"), teff=(1600.0, 100.0, 9),
                                    logg=(3.5, 0.5, 5), nombre=f"Sonora Diamondback 1600–2400 K ({'sin nubes' if f == 'nc' else 'f_sed = ' + f[1:]})")
       for f in ("f1", "f2", "f3", "f4", "f8", "nc")},
    # otro código, sin nubes, log g ≥ 4.0, [M/H] = 0 y C/O = 0.5; muestreado a λ/Δλ = 250: va en el modo R = 250
    "sphinx": dict(dir=Path("/mnt/2TB/IR_Data/sphinx/npz"), teff=(2000.0, 100.0, 13), logg=(4.0, 0.25, 7),
                   nombre="SPHINX (sin nubes)", r_nativa=250.0),
})


def _r250(rejilla, forzado=False):
    """¿Va el ajuste en el modo R = 250? Si se pide, o si la rejilla no llega a R = 500."""
    return bool(forzado) or REJILLAS[rejilla].get("r_nativa", np.inf) < 500
TRACKS = {"BHAC15": ("tracks_bhac15/bhac15_tracks.npz", "Baraffe et al. 2015"),
          "ATMO2020": ("tracks_atmo2020/atmo2020_ceq_tracks.npz", "Phillips et al. 2020")}
T_D = 1500.0
LAM_K = 22000.0
MALO = -1e300
#: suelo cromosférico de Manara et al. 2013, Ec. 2: log(L_acc,noise/L_bol) = A log T_eff + B
SUELO = (6.17, -24.54)
#: qué espectros entran en `ajusta` (`--datos`): todos, solo MUSE o solo los J/H/K
DATOS = ("todo", "muse", "ir")


def _entra(seg, datos):
    """¿Entra el segmento (`brazo` 'vis' o 'nir') con `datos`?"""
    return datos == "todo" or (seg["brazo"] == "vis") == (datos == "muse")


def _planck(w, T):
    return 1.0 / (w ** 5 * np.expm1(1.4388e8 / (w * T)))


def _ppf_trunc_normal(u, mu, sig, lo=0.0, hi=np.inf):
    a, b = (lo - mu) / sig, (hi - mu) / sig
    return float(stats.truncnorm.ppf(u, a, b, loc=mu, scale=sig))


def _empareja(s):
    """El segmento con sus píxeles agrupados de a dos (modo R = 250): media del flujo, σ = ½ √(σ₁² + σ₂²), bueno
    si lo son los dos. Solo se juntan píxeles contiguos (el borde de uno es el del otro); un píxel suelto se pierde."""
    i = np.arange(0, s["w"].size - 1, 2)
    i = i[np.abs(s["hi"][i] - s["lo"][i + 1]) <= 0.05 * (s["hi"][i] - s["lo"][i])]
    j = i + 1
    return {**s, "w": 0.5 * (s["w"][i] + s["w"][j]), "lo": s["lo"][i], "hi": s["hi"][j],
            "f": 0.5 * (s["f"][i] + s["f"][j]), "e": 0.5 * np.hypot(s["e"][i], s["e"][j]),
            "bueno": s["bueno"][i] & s["bueno"][j]}


def _bases_nativas(pls, segs):
    """Modelos que ya están a la resolución del análisis (SPHINX, λ/Δλ = 250 por punto): sin degradar,
    interpolados en el centro de cada píxel; NaN fuera de su cobertura."""
    out = {}
    for i, t in enumerate(pls):
        fila = []
        for s in segs:
            w, f = t[s["brazo"]]
            m = np.interp(s["w"], w, f)
            m[(s["w"] < w[0]) | (s["w"] > w[-1])] = np.nan
            fila.append(m)
        out[i] = fila
    return out


def _prepara(slug, R, r250=False):
    """Como `ajuste_conjunto.preparar(..., btsettl=DIR)` pero con los rangos de la rejilla elegida (aquel fija
    2000–3200 K y log g 2.5–4.5): segmentos en cgs, modelos base por segmento y `Conjunto` (píxeles comunes).

    `r250`: los píxeles de los datos se agrupan de a dos (MUSE en bins de 30 Å, el IR a λ/Δλ ≈ 250). Lo exige una
    rejilla con `r_nativa` < 500 (SPHINX), cuyos modelos se usan tal cual (`_bases_nativas`); con una rejilla de
    resolución alta, los modelos se degradan a R = 500 y se promedian en el píxel agrupado, como los datos."""
    from scripts import ajuste_conjunto as aj
    from scripts import tipo_ir_plantillas as tp
    p = tp.preparar(slug, "combinado", av_fija=True, bibliotecas=("manara",), h2o_muse=True)
    segs = aj.segmentos_cgs(slug, p)
    if r250:
        segs = [_empareja(s) for s in segs]
    t0, dt, nt = R["teff"]
    g0, dg, ng = R["logg"]
    p["pls"] = aj.modelos_btsettl(R["dir"], teff=(t0, t0 + (nt - 1) * dt), logg=(g0, g0 + (ng - 1) * dg))
    p["bases"] = (_bases_nativas(p["pls"], segs) if R.get("r_nativa", np.inf) < 500
                  else tp.modelos_base(p["pls"], segs))
    fac = tp.factores_equilibrio(segs, p["bases"], [p["av_tabla"]])
    return p, aj.Conjunto(segs, p["bases"], 0, fac)


class Modelo:
    def __init__(self, slug, modo="libre", av="primaria", tracks="ATMO2020", inyecta_bt=None, plantilla=None,
                 av_inyectada=None, semilla=0, av_fija=None, rejilla="btsettl", dir_rejilla=None, datos="todo",
                 r250=False):
        from musepipe.config import load_run_config
        from musepipe.models.tracks import TrackGrid
        from scripts import tipo_ir_indices as tir
        warnings.simplefilter("ignore", RuntimeWarning)
        if datos not in DATOS:
            raise ValueError(f"datos={datos!r}: debe ser uno de {DATOS}")
        self.slug, self.modo, self.av_modo, self.av_fija, self.datos = slug, modo, av, av_fija, datos
        R = dict(REJILLAS[rejilla])
        if dir_rejilla is not None:
            R["dir"] = Path(dir_rejilla)
        self.rejilla, self.rejilla_nombre = rejilla, R["nombre"]
        self.T0, self.dT, self.nT = R["teff"]
        self.G0, self.dG, self.nG = R["logg"]
        self.T0, self.G0 = float(self.T0), float(self.G0)
        self.r_analisis = 250 if _r250(rejilla, r250) else 500
        p, conj = _prepara(slug, R, r250=self.r_analisis == 250)
        self.run = p["run"]
        cfg = load_run_config(self.run, project_root=ROOT).config
        self.cfg = cfg
        self.d_pc, self.d_err = float(cfg["h03_distance_pc"]), float(cfg["h03_distance_err_pc"])
        self.av0, self.av_err = float(cfg["h03_av"]), float(cfg["h03_av_err"])
        edad, (elo, ehi) = float(cfg["g3_age_myr"]), cfg["g3_age_err_myr"]
        self.edad0 = edad
        self.edad_sig_ln = 0.5 * (math.log((edad + ehi) / edad) + math.log(edad / (edad - elo)))
        perfil = tir.perfil(slug)
        self.fisica = perfil.get("fisica", {})
        ir = perfil["ir_spectra"]["spectra"]

        # rejilla [T, g] -> índice del modelo
        idx = np.full((self.nT, self.nG), -1)
        for i, t in enumerate(p["pls"]):
            idx[int(round((t["teff"] - self.T0) / self.dT)), int(round((t["logg"] - self.G0) / self.dG))] = i
        self.rellenados = []
        for a in range(self.nT):
            for b in range(self.nG):
                if idx[a, b] >= 0:
                    continue
                if not (0 < a < self.nT - 1 and idx[a - 1, b] >= 0 and idx[a + 1, b] >= 0):
                    raise SystemExit(f"rejilla {rejilla} incompleta en ({self.T0 + a * self.dT}, {self.G0 + b * self.dG})")
                m0, m1 = p["bases"][idx[a - 1, b]], p["bases"][idx[a + 1, b]]
                nuevo = len(p["pls"])
                p["pls"].append(dict(teff=self.T0 + a * self.dT, logg=self.G0 + b * self.dG, interpolado=True))
                p["bases"][nuevo] = [np.sqrt(x0 * x1) for x0, x1 in zip(m0, m1)]   # media en log F
                idx[a, b] = nuevo
                self.rellenados.append((self.T0 + a * self.dT, self.G0 + b * self.dG))

        datos_sint = None
        if plantilla or inyecta_bt:
            datos_sint = self._sinteticos(p, conj, plantilla, inyecta_bt, av_inyectada, idx, semilla)

        self.segs = []
        k_ref = next(k for k, s in enumerate(conj.segs) if s["brazo"] == "nir" and s["w"][0] <= LAM_K <= s["w"][-1])
        self.lref = np.empty((self.nT, self.nG))
        for a in range(self.nT):
            for b in range(self.nG):
                s, m = conj.segs[k_ref], p["bases"][idx[a, b]][k_ref]
                ok = np.isfinite(m)
                self.lref[a, b] = math.log10(float(np.interp(LAM_K, s["w"][ok], m[ok])))
        for k, s in enumerate(conj.segs):
            if not _entra(s, datos):
                continue
            c = conj.comun[k].copy()
            f, e = s["f"], s["e"]
            if datos_sint is not None:
                f, e = datos_sint[k]
                c &= np.isfinite(f) & np.isfinite(e) & (e > 0)
            w = s["w"][c]
            lg = np.empty((self.nT, self.nG, int(c.sum())))
            for a in range(self.nT):
                for b in range(self.nG):
                    lg[a, b] = np.log10(p["bases"][idx[a, b]][k][c])
            banda = "vis" if s["brazo"] == "vis" else ir[s["nombre"]]["band"]
            self.segs.append(dict(
                nombre=s["nombre"], banda=banda, w=w, d=f[c], e2=e[c] ** 2, lg=lg,
                a=np.asarray(CCMExtinction(3.1, citation="CCM89").a_lambda_over_av(w), float),
                bb=(_planck(w, T_D) / _planck(LAM_K, T_D)) if s["brazo"] == "nir" else np.zeros(w.size),
                sigma=s["sigma_escala"]))
        self.bandas = sorted({s["banda"] for s in self.segs})
        self.track = None
        if modo == "evo":
            rel, cita = TRACKS[tracks]
            root = (ROOT / cfg["g3_libraries_root"]).resolve()
            self.track = TrackGrid(root / rel, family=tracks, citation=cita, version=None)
            self.tracks_nombre = tracks
        self._prepara_priores()

    # -------------------------------------------------------------- datos de prueba
    def _sinteticos(self, p, conj, plantilla, inyecta_bt, av_inj, idx, semilla):
        """[(f, e)] por segmento: una plantilla o un modelo de la rejilla, con la S/N por píxel del dato real y ruido."""
        from scripts import ajuste_conjunto as aj
        from scripts import tipo_ir_plantillas as tp
        rng = np.random.default_rng(semilla)
        if plantilla:
            pls = tp.plantillas(self.cfg, ("manara", "lateM", "xshyne"))
            i = next(i for i, t in enumerate(pls) if t["objeto"] == plantilla)
            base = tp.modelos_base([pls[i]], conj.segs)[0]
            av = self.av0 if av_inj is None else float(av_inj)
            sin = [base[k] * 10 ** (-0.4 * np.asarray(CCMExtinction(3.1, citation="CCM89").a_lambda_over_av(s["w"]),
                                                       float) * av) for k, s in enumerate(conj.segs)]
            # UN solo factor (conserva los colores): el del segmento anclado a la fotometría, o MUSE
            k0 = next((k for k, s in enumerate(conj.segs) if s["sigma_escala"] == aj.SIGMA_IR),
                      next(k for k, s in enumerate(conj.segs) if s["brazo"] == "vis"))
            ok = conj.comun[k0] & np.isfinite(sin[k0])
            fac = float(np.median(conj.segs[k0]["f"][ok]) / np.median(sin[k0][ok]))
            sin = [x * fac for x in sin]
            self.verdad = dict(plantilla=plantilla, spt=pls[i]["spt"], lib=pls[i]["lib"], av=av)
        else:
            T, G, av, r_rj = inyecta_bt
            x, y = (T - self.T0) / self.dT, (G - self.G0) / self.dG
            i0, j0 = min(int(x), self.nT - 2), min(int(y), self.nG - 2)
            t, u = x - i0, y - j0
            om = (r_rj * RJUP_CM / (self.d_pc * PC_CM)) ** 2
            sin = []
            for k, s in enumerate(conj.segs):
                L = [np.log10(p["bases"][idx[a, b]][k]) for a, b in ((i0, j0), (i0 + 1, j0), (i0, j0 + 1), (i0 + 1, j0 + 1))]
                lf = (1 - t) * (1 - u) * L[0] + t * (1 - u) * L[1] + (1 - t) * u * L[2] + t * u * L[3]
                ext = 10 ** (-0.4 * np.asarray(CCMExtinction(3.1, citation="CCM89").a_lambda_over_av(s["w"]), float) * av)
                sin.append(om * 10 ** lf * ext)
            self.verdad = dict(teff=T, logg=G, av=av, r_rj=r_rj)
        out = []
        for k, s in enumerate(conj.segs):
            with np.errstate(invalid="ignore", divide="ignore"):
                snr = np.abs(s["f"]) / s["e"]
                e = np.abs(sin[k]) / np.where(snr > 0, snr, np.nan)
            out.append((sin[k] + rng.normal(0.0, np.nan_to_num(e)), e))
        return out

    # -------------------------------------------------------------- parámetros
    def _prepara_priores(self):
        n, t = [], []

        def add(nombre, f):
            n.append(nombre)
            t.append(f)
        if self.modo == "libre":
            add("teff", lambda u: self.T0 + u * (self.nT - 1) * self.dT)
            add("logg", lambda u: self.G0 + u * (self.nG - 1) * self.dG)
            add("r_rj", lambda u: 10 ** (math.log10(0.5) + u * (math.log10(6.0) - math.log10(0.5))))
        else:
            mu, sg = math.log10(self.edad0), self.edad_sig_ln / math.log(10)
            add("log_edad_myr", lambda u: float(stats.norm.ppf(u, mu, sg)))
            add("log_lbol", lambda u: -4.5 + 3.0 * u)
        add("d_pc", lambda u: float(stats.norm.ppf(u, self.d_pc, self.d_err)))
        if self.av_fija is not None:
            add("av", lambda u: self.av_fija + 0.02 * (u - 0.5))   # perfil: A_V fija (±0.01)
        elif self.av_modo == "libre":
            add("av", lambda u: 8.0 * u)
        else:
            add("av", lambda u: _ppf_trunc_normal(u, self.av0, self.av_err))
        if any(s["banda"] != "vis" for s in self.segs):   # sin IR, r_K no tiene datos
            add("rk", lambda u: 1.5 * u)
        for s in self.segs:
            if s["sigma"] is None:
                add(f"s_{s['nombre']}", lambda u: 10 ** (math.log10(0.2) + u * 2 * math.log10(5.0)))
            else:
                add(f"s_{s['nombre']}", (lambda sg: (lambda u: _ppf_trunc_normal(u, 1.0, sg, lo=0.05)))(s["sigma"]))
        for b in self.bandas:
            add(f"lnf_{b}", lambda u: math.log(1e-3) + u * (math.log(1.0) - math.log(1e-3)))
        self.nombres, self._t = n, t
        self.ndim = len(n)
        self.pos = {k: i for i, k in enumerate(n)}

    def prior_transform(self, u):
        return np.array([f(x) for f, x in zip(self._t, u)])

    # -------------------------------------------------------------- física
    def fisica_de(self, th):
        """(teff, logg, r_cm, masa_mjup, lbol_lsun) del vector de parámetros; None si cae fuera."""
        P = self.pos
        if self.modo == "libre":
            teff, logg, r_cm = th[P["teff"]], th[P["logg"]], th[P["r_rj"]] * RJUP_CM
            lbol = 4 * math.pi * r_cm ** 2 * SIGMA_SB_CGS * teff ** 4 / LSUN_ERG_S
            return teff, logg, r_cm, float("nan"), lbol
        tr = self.track.lookup(10 ** th[P["log_lbol"]], 10 ** th[P["log_edad_myr"]] / 1e3)
        if not tr["in_range"] or tr["clamped"]:
            return None
        teff, logg = tr["teff_k"], tr["logg"]
        if not (self.T0 <= teff <= self.T0 + (self.nT - 1) * self.dT and self.G0 <= logg <= self.G0 + (self.nG - 1) * self.dG):
            return None
        return teff, logg, tr["radius_rsun"] * RSUN_CM, tr["mass_msun"] * 1047.57, 10 ** th[P["log_lbol"]]

    def modelos(self, th):
        f = self.fisica_de(th)
        if f is None:
            return None
        teff, logg, r_cm, _, _ = f
        P = self.pos
        x, y = (teff - self.T0) / self.dT, (logg - self.G0) / self.dG
        i, j = min(int(x), self.nT - 2), min(int(y), self.nG - 2)
        t, u = x - i, y - j
        c00, c10, c01, c11 = (1 - t) * (1 - u), t * (1 - u), (1 - t) * u, t * u
        lr = self.lref
        fref = 10 ** (c00 * lr[i, j] + c10 * lr[i + 1, j] + c01 * lr[i, j + 1] + c11 * lr[i + 1, j + 1])
        om = (r_cm / (th[P["d_pc"]] * PC_CM)) ** 2
        av, rk = th[P["av"]], (th[P["rk"]] if "rk" in P else 0.0)
        out = []
        for s in self.segs:
            L = s["lg"]
            F = 10 ** (c00 * L[i, j] + c10 * L[i + 1, j] + c01 * L[i, j + 1] + c11 * L[i + 1, j + 1])
            out.append(th[P[f"s_{s['nombre']}"]] * om * (F + rk * fref * s["bb"]) * 10 ** (-0.4 * s["a"] * av))
        return out

    def loglike(self, th):
        M = self.modelos(th)
        if M is None:
            return MALO
        tot = 0.0
        for s, m in zip(self.segs, M):
            fb = math.exp(th[self.pos[f"lnf_{s['banda']}"]])
            var = s["e2"] + (fb * m) ** 2
            r = s["d"] - m
            tot += -0.5 * float(np.sum(r * r / var + np.log(var)))
        return tot

    def chi2(self, th, inflado=False):
        M = self.modelos(th)
        out = {}
        for s, m in zip(self.segs, M):
            var = s["e2"] + ((math.exp(th[self.pos[f"lnf_{s['banda']}"]]) * m) ** 2 if inflado else 0.0)
            out[s["nombre"]] = float(np.sum((s["d"] - m) ** 2 / var)) / s["d"].size
        return out


def evo_solo(mod_cfg, slug, tracks, n=20000, semilla=0, root=None):
    """T_eff, R, M y log g de la L_bol publicada y el prior de edad, sin espectro (la referencia). `root`: la raíz
    del proyecto contra la que se resuelve `g3_libraries_root` (por defecto este clon; desde un worktree, la del
    clon con los datos)."""
    from musepipe.models.tracks import TrackGrid
    from scripts import tipo_ir_indices as tir
    rng = np.random.default_rng(semilla)
    cfg = mod_cfg
    perfil = tir.perfil(slug)
    fis = perfil["fisica"]
    edad, (elo, ehi) = float(cfg["g3_age_myr"]), cfg["g3_age_err_myr"]
    sig = 0.5 * (math.log((edad + ehi) / edad) + math.log(edad / (edad - elo)))
    rel, cita = TRACKS[tracks]
    raiz = ROOT if root is None else Path(root)
    tg = TrackGrid((raiz / cfg["g3_libraries_root"]).resolve() / rel, family=tracks, citation=cita, version=None)
    lbol = 10 ** rng.normal(fis["log_lbol"], fis["log_lbol_err"], n)
    age = np.exp(rng.normal(math.log(edad), sig, n)) / 1e3
    s = tg.sample(lbol, age)
    ok = s["in_range"]
    return dict(teff=s["teff_k"][ok], logg=s["logg"][ok], r_rj=s["radius_rsun"][ok] * RSUN_CM / RJUP_CM,
                masa_mjup=s["mass_msun"][ok] * 1047.57, fraccion_en_rejilla=float(ok.mean()))


def percentiles(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if not x.size:
        return None
    q = np.percentile(x, [2.5, 16, 50, 84, 97.5])
    return dict(p2_5=float(q[0]), p16=float(q[1]), p50=float(q[2]), p84=float(q[3]), p97_5=float(q[4]))


def corre(mod, nlive, semilla, dlogz=0.5):
    import dynesty
    from dynesty.utils import resample_equal
    t0 = time.time()
    rng = np.random.default_rng(semilla)
    sampler = dynesty.NestedSampler(mod.loglike, mod.prior_transform, mod.ndim, nlive=nlive, sample="rslice",
                                    bound="multi", rstate=rng)
    sampler.run_nested(dlogz=dlogz, print_progress=False)
    res = sampler.results
    w = np.exp(res.logwt - res.logz[-1])
    eq = resample_equal(res.samples, w / w.sum(), rstate=rng)
    der = np.array([mod.fisica_de(th) or (np.nan,) * 5 for th in eq], float)
    imax = int(np.argmax(res.logl))
    return dict(segundos=round(time.time() - t0, 1), ncall=int(np.sum(res.ncall)), logz=float(res.logz[-1]),
                logz_err=float(res.logzerr[-1]), muestras=eq, derivadas=der, mejor=res.samples[imax],
                logl_max=float(res.logl[imax]))


def ajusta(a):
    from musepipe.config import load_run_config
    from scripts import tipo_ir_indices as tir
    a.salida.mkdir(parents=True, exist_ok=True)
    et = a.etiqueta or "_".join(x for x in (a.slug, a.rejilla if a.rejilla != "btsettl" else None, a.modo, a.tracks if a.modo != "libre" else None, f"av{a.av}",
                                             f"pl-{a.plantilla}" if a.plantilla else None,
                                             f"avfija{a.av_fija:g}" if a.av_fija is not None else None,
                                             "bt-" + "-".join(f"{v:g}" for v in a.inyecta_bt) if a.inyecta_bt else None,
                                             f"solo{a.datos}" if a.datos != "todo" else None,
                                             "r250" if _r250(a.rejilla, a.r250) else None,
                                             f"s{a.semilla}") if x)
    if a.modo == "evo_solo":
        cfg = load_run_config(tir.perfil(a.slug)["paper"]["run"], project_root=ROOT).config
        r = evo_solo(cfg, a.slug, a.tracks, semilla=a.semilla)
        out = dict(etiqueta=et, slug=a.slug, modo=a.modo, tracks=a.tracks,
                   fraccion_en_rejilla=r["fraccion_en_rejilla"],
                   **{k: percentiles(v) for k, v in r.items() if k != "fraccion_en_rejilla"})
        np.savez(a.salida / f"{et}.npz", **{k: v for k, v in r.items() if k != "fraccion_en_rejilla"})
    else:
        mod = Modelo(a.slug, a.modo, a.av, a.tracks, a.inyecta_bt, a.plantilla, a.av_inyectada, a.semilla, a.av_fija,
                     a.rejilla, a.dir_rejilla, a.datos, a.r250)
        r = corre(mod, a.nlive, a.semilla)
        eq, der = r["muestras"], r["derivadas"]
        post = {k: percentiles(eq[:, i]) for i, k in enumerate(mod.nombres)}
        for i, k in enumerate(("teff", "logg", "r_cm", "masa_mjup", "lbol_lsun")):
            post[f"der_{k}"] = percentiles(der[:, i] / (RJUP_CM if k == "r_cm" else 1.0))
        out = dict(etiqueta=et, slug=a.slug, run=mod.run, modo=a.modo, av=a.av, av_fija=a.av_fija,
                   rejilla=a.rejilla, datos=a.datos, r_analisis=mod.r_analisis, nodos_interpolados=mod.rellenados,
                   # los ejes con los que corrió (inicio, paso, n): una rejilla puede ampliarse después
                   ejes=dict(teff=[mod.T0, mod.dT, mod.nT], logg=[mod.G0, mod.dG, mod.nG]),
                   tracks=a.tracks if a.modo == "evo" else None, nlive=a.nlive, semilla=a.semilla,
                   ndim=mod.ndim, nombres=mod.nombres, segundos=r["segundos"], ncall=r["ncall"],
                   logz=r["logz"], logz_err=r["logz_err"], logl_max=r["logl_max"],
                   chi2nu_mejor=mod.chi2(r["mejor"]), chi2nu_mejor_inflado=mod.chi2(r["mejor"], inflado=True),
                   mejor=dict(zip(mod.nombres, map(float, r["mejor"]))),
                   verdad=getattr(mod, "verdad", None), posterior=post,
                   n_puntos={s["nombre"]: int(s["d"].size) for s in mod.segs})
        np.savez(a.salida / f"{et}.npz", muestras=eq, derivadas=der, nombres=np.array(mod.nombres))
    (a.salida / f"{et}.json").write_text(json.dumps(out, indent=1, ensure_ascii=False, default=float))
    t = out.get("posterior", {}).get("der_teff") or out.get("teff")
    print(f"{et}: T_eff p50 {t['p50']:.0f} [{t['p16']:.0f}, {t['p84']:.0f}]"
          + (f"  ({out['segundos']} s, {out['ncall']} llamadas, logZ {out['logz']:.1f})" if "logz" in out else ""))
    return 0


# ================================================================== resumen y margen cromosférico
def suelo(teff):
    """log(L_acc,noise/L_bol) de Manara et al. (2013, Ec. 2) a la T_eff dada."""
    return SUELO[0] * np.log10(teff) + SUELO[1]


def teff_umbral(log_lacc_lbol):
    """La T_eff a la que el suelo iguala un log(L_acc/L_bol) dado."""
    return 10 ** ((log_lacc_lbol - SUELO[1]) / SUELO[0])


def margen(teff, av, lbol_lsun, lacc0_lsun, av0, a, ahav):
    """log(L_acc/L_bol) − suelo(T_eff) por muestra. La L_acc publicada (con la A_V adoptada `av0`) se reescala a
    la A_V de la muestra: la L_Hα desenrojecida cambia 0.4·A_Hα/A_V dex por magnitud y la L_acc, `a` veces eso
    (la pendiente de la relación L_acc–L_Hα)."""
    lacc = lacc0_lsun * 10 ** (0.4 * a * ahav * (np.asarray(av, float) - av0))
    return np.log10(lacc / np.asarray(lbol_lsun, float)) - suelo(np.asarray(teff, float))


def umbral(slug, root=None):
    """(T*, log L_acc/L_bol, tipo): el umbral cromosférico con la L_acc del paper y la L_bol publicada. `root`: la
    raíz que contiene `runs/` (por defecto la del repo; desde un worktree, la del clon que los tiene)."""
    from scripts import suelo_cromosferico as sc
    from scripts import tipo_ir_indices as tir
    root = ROOT if root is None else Path(root)
    perfil = tir.perfil(slug, root=root)
    lacc, tipo = sc.lacc_hal(perfil["paper"]["run"], root=root)
    ratio = math.log10(lacc) - float(perfil["fisica"]["log_lbol"])
    return teff_umbral(ratio), ratio, tipo


def entrada_margen(slug, root=None):
    """L_acc publicada (con la A_V adoptada), A_V adoptada, pendiente de la relación y A_Hα/A_V del run del paper.
    `root`: la raíz que contiene `runs/` (por defecto la del repo; desde un worktree, la del clon que los tiene)."""
    from musepipe.config import load_run_config
    from musepipe.stages.stage_h03_limits import physical_inputs_from_config
    from scripts import suelo_cromosferico as sc
    from scripts import tipo_ir_indices as tir
    root = ROOT if root is None else Path(root)
    perfil = tir.perfil(slug, root=root)
    run = perfil["paper"]["run"]
    lacc, tipo = sc.lacc_hal(run, root=root)
    cfg = load_run_config(run, project_root=root).config
    phys = physical_inputs_from_config(cfg)
    return dict(lacc0=lacc, tipo=tipo, av0=float(cfg["h03_av"]), a=float(phys["lacc_lha_a"]),
                ahav=float(phys["a_halpha_over_av"]), lbol_lit=10 ** float(perfil["fisica"]["log_lbol"]))


def margen_de_corrida(d, e):
    """El margen por muestra de una corrida (`d`: su json con `_npz`; `e`: `entrada_margen`). En `evo_solo`, la
    A_V adoptada y la L_bol publicada."""
    z = np.load(d["_npz"])
    if d["modo"] == "evo_solo":
        teff = z["teff"]
        av = np.full(teff.size, e["av0"])
        lbol = np.full(teff.size, e["lbol_lit"])
    else:
        nombres = list(z["nombres"])
        teff, lbol = z["derivadas"][:, 0], z["derivadas"][:, 4]
        av = z["muestras"][:, nombres.index("av")]
    return margen(teff, av, lbol, e["lacc0"], e["av0"], e["a"], e["ahav"])


def _carga(dirs):
    out = []
    for dd in dirs:
        for f in sorted(glob.glob(str(Path(dd) / "*.json"))):
            d = json.loads(Path(f).read_text())
            z = Path(dd) / (Path(f).stem + ".npz")
            d["_npz"] = str(z) if z.exists() else None
            out.append(d)
    return out


def _teff_muestras(d):
    z = np.load(d["_npz"])
    return z["teff"] if d["modo"] == "evo_solo" else z["derivadas"][:, 0]


def _fmt(p, nd=0):
    if not p:
        return "—"
    f = f"{{:.{nd}f}}"
    return f"{f.format(p['p50'])} [{f.format(p['p16'])}, {f.format(p['p84'])}]"


def resume(dirs, salida=None):
    """Tablas en markdown (stdout) y, con `salida`, el json: corridas sobre los datos (T_eff, margen, log g, R,
    M, A_V, r_K, χ²_ν por banda, f), la calibración con plantillas, la sensibilidad del modo `evo` y las
    inyecciones."""
    todos = _carga(dirs)
    slugs = sorted({d["slug"] for d in todos})
    res = {"umbral": {}, "datos": [], "calibracion": [], "sensibilidad": [], "inyeccion": []}
    for slug in slugs:
        T, ratio, tipo = umbral(slug)
        res["umbral"][slug] = dict(teff_umbral=T, log_lacc_lbol=ratio, tipo=tipo)
    print("## Umbral cromosférico\n")
    for s, u in res["umbral"].items():
        print(f"- {s}: log L_acc/L_bol {'<' if u['tipo'] == 'upper_limit' else '='} {u['log_lacc_lbol']:.3f} "
              f"→ el suelo lo iguala a T* = {u['teff_umbral']:.0f} K")

    print("\n## Corridas sobre los datos\n")
    print("| objeto | modo | T_eff (K) | P(T<T*) | margen coherente (dex) | P(sobre el suelo) | log g | R (R_J) | M (M_J) | A_V | r_K | χ²_ν dato (vis/J/H/K) | f (vis/J/H/K) |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    entradas = {s: entrada_margen(s) for s in slugs}
    for d in todos:
        if d.get("plantilla") or "pl-" in d["etiqueta"] or "bt-" in d["etiqueta"]:
            continue
        t = _teff_muestras(d)
        T = res["umbral"][d["slug"]]["teff_umbral"]
        pT = float(np.mean(t < T))
        mg = margen_de_corrida(d, entradas[d["slug"]])
        mgp = np.percentile(mg[np.isfinite(mg)], [16, 50, 84])
        mtxt = f"{mgp[1]:+.2f} [{mgp[0]:+.2f}, {mgp[2]:+.2f}]"
        psobre = float(np.mean(mg > 0))
        modo = d["modo"] + (f" {d['tracks']}" if d.get("tracks") else "") + (
            (f", A_V fija {d['av_fija']:g}" if d.get("av_fija") is not None else f", A_V {d['av']}") if d["modo"] != "evo_solo" else "") + (
            f", solo {d['datos'].upper()}" if d.get("datos", "todo") != "todo" else "")
        if d["modo"] == "evo_solo":
            fila = dict(slug=d["slug"], modo=modo, teff=d["teff"], p_bajo_umbral=pT, logg=d["logg"], r=d["r_rj"],
                        m=d["masa_mjup"], margen=mgp.tolist(), p_sobre_suelo=psobre)
            print(f"| {d['slug']} | {modo} (sin espectro) | {_fmt(d['teff'])} | {pT:.2f} | {mtxt} | {psobre:.2f} | {_fmt(d['logg'], 2)} | "
                  f"{_fmt(d['r_rj'], 2)} | {_fmt(d['masa_mjup'], 1)} | — | — | — | — |")
        else:
            P = d["posterior"]
            bandas = ["vis", "J", "H", "K"]
            chi = d["chi2nu_mejor"]
            # χ²_ν por banda: media de los segmentos de esa banda
            por_banda = {}
            for nom, v in chi.items():
                b = "vis" if nom == "MUSE" else nom.split("_")[-1]
                por_banda.setdefault(b, []).append(v)
            chis = "/".join(f"{np.mean(por_banda[b]):.1f}" if b in por_banda else "—" for b in bandas)
            fs = "/".join(f"{math.exp(P[f'lnf_{b}']['p50']):.3f}" if f"lnf_{b}" in P else "—" for b in bandas)
            fila = dict(slug=d["slug"], modo=modo, teff=P["der_teff"], p_bajo_umbral=pT, logg=P["der_logg"],
                        r=P["der_r_cm"], m=P.get("der_masa_mjup"), av=P["av"], rk=P.get("rk"), chi2=chi,
                        logz=d["logz"], segundos=d["segundos"], margen=mgp.tolist(), p_sobre_suelo=psobre,
                        av_fija=d.get("av_fija"), logl_max=d["logl_max"])
            print(f"| {d['slug']} | {modo} | {_fmt(P['der_teff'])} | {pT:.2f} | {mtxt} | {psobre:.2f} | {_fmt(P['der_logg'], 2)} | "
                  f"{_fmt(P['der_r_cm'], 2)} | {_fmt(P.get('der_masa_mjup'), 1)} | {_fmt(P['av'], 2)} | {_fmt(P.get('rk'), 2)} | "
                  f"{chis} | {fs} |")
        res["datos"].append(fila)

    print("\n## Calibración con plantillas (modo libre, A_V de la primaria inyectada y como prior)\n")
    print("| objeto | plantilla | tipo | T_eff ajustada (K) | log g | A_V (inyectada) | χ²_ν dato (vis/J/H/K) |")
    print("|---|---|---|---|---|---|---|")
    for d in todos:
        if d["modo"] != "libre" or not d.get("verdad") or "plantilla" not in (d.get("verdad") or {}):
            continue
        v, P = d["verdad"], d["posterior"]
        fila = dict(slug=d["slug"], plantilla=v["plantilla"], spt=v["spt"], codigo=spt_code(v["spt"]),
                    teff=P["der_teff"], logg=P["der_logg"], av=P["av"], av_inyectada=v["av"])
        res["calibracion"].append(fila)
    for f in sorted(res["calibracion"], key=lambda r: (r["slug"], r["codigo"])):
        print(f"| {f['slug']} | {f['plantilla']} | {f['spt']} | {_fmt(f['teff'])} | {_fmt(f['logg'], 2)} | "
              f"{_fmt(f['av'], 2)} ({f['av_inyectada']}) | |")

    print("\n## Sensibilidad: modo evolutivo sobre plantillas con el flujo del compañero\n")
    print("| objeto | plantilla | tipo | T_eff evo (K) | T_eff libre de la misma plantilla (K) |")
    print("|---|---|---|---|---|")
    libre = {(f["slug"], f["plantilla"]): f["teff"] for f in res["calibracion"]}
    for d in todos:
        if d["modo"] != "evo" or not d.get("verdad"):
            continue
        v, P = d["verdad"], d["posterior"]
        res["sensibilidad"].append(dict(slug=d["slug"], plantilla=v["plantilla"], spt=v["spt"], teff_evo=P["der_teff"],
                                        teff_libre=libre.get((d["slug"], v["plantilla"]))))
        print(f"| {d['slug']} | {v['plantilla']} | {v['spt']} | {_fmt(P['der_teff'])} | "
              f"{_fmt(libre.get((d['slug'], v['plantilla'])))} |")

    print("\n## Inyección de un modelo de la rejilla (maquinaria)\n")
    for d in todos:
        v = d.get("verdad") or {}
        if "teff" in v:
            P = d["posterior"]
            res["inyeccion"].append(dict(slug=d["slug"], verdad=v, teff=P["der_teff"], logg=P["der_logg"],
                                         r=P["r_rj"], av=P["av"]))
            print(f"- {d['slug']}: verdad {v} → T_eff {_fmt(P['der_teff'])}, log g {_fmt(P['der_logg'], 2)}, "
                  f"R {_fmt(P['r_rj'], 2)}, A_V {_fmt(P['av'], 2)}")
    if salida is not None:
        Path(salida).write_text(json.dumps(res, indent=1, ensure_ascii=False, default=float))
    return res


#: nombre de cada rejilla en las tablas y figuras del paper
NOMBRE_REJILLA = {"btsettl": "BT-Settl", "drift": "DRIFT-PHOENIX"}


def _pm(p, nd=0, f=None, signo=False):
    """`$m^{+a}_{-b}$` (LaTeX): la mediana y el intervalo 16–84 % de un dict de `percentiles`; `f` transforma los
    percentiles antes (monótona creciente, así que conserva el orden); `signo` escribe el + de la mediana."""
    f = f or (lambda x: x)
    m, lo, hi = (f(p[k]) for k in ("p50", "p16", "p84"))
    t = f"{{:.{nd}f}}"
    tm = f"{{:{'+' if signo else ''}.{nd}f}}"
    return f"${tm.format(m)}^{{+{t.format(hi - m)}}}_{{-{t.format(m - lo)}}}$"


def _de_la_tabla(d):
    """Una corrida sobre los datos (no plantilla ni inyección), `libre`, con la A_V fija o con el prior, con
    todos los espectros (no `--datos muse|ir`), con una de las rejillas del paper (`NOMBRE_REJILLA`) y a R = 500."""
    return (d["modo"] == "libre" and d.get("av") == "primaria" and not d.get("verdad")
            and d.get("datos", "todo") == "todo" and d.get("rejilla", "btsettl") in NOMBRE_REJILLA
            and d.get("r_analisis", 500) == 500)


def filas_tabla(corridas, deltas, nombres):
    """El cuerpo LaTeX de la tabla de las cadenas del apéndice B: por compañero (en el orden de `nombres`,
    slug -> nombre en el texto), por rejilla y con A_V 1.8, 3.0 y con prior, la mediana y el 16–84 % de T_eff,
    log g, R, A_V, r_K, el error de modelo en MUSE y Δ, y ln Z relativo al mejor ajuste del compañero. `deltas`:
    etiqueta -> `percentiles` del margen. Solo las corridas `libre` con A_V fija o con prior; una A_V fija va
    sin intervalo; las plantillas e inyecciones (`verdad`) no entran."""
    filas = []
    for slug, nombre in nombres.items():
        suyas = [d for d in corridas if d["slug"] == slug and _de_la_tabla(d)]
        if not suyas:
            continue
        lnz_max = max(d["logz"] for d in suyas)
        filas.append(rf"\multicolumn{{9}}{{l}}{{{nombre}}} \\")
        orden = lambda d: (list(NOMBRE_REJILLA).index(d.get("rejilla", "btsettl")),
                           math.inf if d.get("av_fija") is None else d["av_fija"])
        for d in sorted(suyas, key=orden):
            P = d["posterior"]
            av = (f"${d['av_fija']:.1f}$" if d.get("av_fija") is not None else _pm(P["av"], 2))
            celdas = [NOMBRE_REJILLA[d.get("rejilla", "btsettl")], _pm(P["teff"]), _pm(P["logg"], 2),
                      _pm(P["r_rj"], 2), av, _pm(P["rk"], 2), _pm(P["lnf_vis"], 1, lambda x: 100 * math.exp(x)),
                      _pm(deltas[d["etiqueta"]], 2, signo=True), f"${d['logz'] - lnz_max:.1f}$"]
            filas.append("\\quad " + " & ".join(celdas) + r" \\")
    return filas


def tabla(dirs):
    """Imprime `filas_tabla` para las corridas de `dirs`, con el margen de cada muestra (como `resume`)."""
    from scripts import tipo_ir_indices as tir
    todos = [d for d in _carga(dirs) if _de_la_tabla(d)]
    slugs = sorted({d["slug"] for d in todos})
    nombres = {s: tir.perfil(s)["paper"]["display"].replace(" ", "~") for s in slugs}
    entradas = {s: entrada_margen(s) for s in slugs}
    deltas = {d["etiqueta"]: percentiles(margen_de_corrida(d, entradas[d["slug"]])) for d in todos}
    filas = filas_tabla(todos, deltas, nombres)
    print("\n".join(filas))
    return filas


#: las rejillas de la tabla de rejillas del apéndice B, en el orden de sus filas, con su nombre en LaTeX. De
#: Diamondback entra la ampliada desde 1600 K: la de 2000 K recortaba la cola de 42B b
#: (docs/2026-10-17_teff_mas_rejillas.md).
REJILLAS_TABLA = {"btsettl": "BT-Settl", "drift": "DRIFT-PHOENIX", "ames_dusty": "AMES-Dusty",
                  "ames_cond": "AMES-Cond", "bt_dusty": "BT-Dusty",
                  **{f"diamondback1600_{f}": ("Diamondback, no clouds" if f == "nc"
                                              else rf"Diamondback, \(f_{{\rm sed}} = {f[1:]}\)")
                     for f in ("f1", "f2", "f3", "f4", "f8", "nc")},
                  "sphinx": "SPHINX"}


def _de_rejillas(d):
    """Una corrida de la tabla de rejillas: como `_de_la_tabla`, pero con cualquier rejilla de `REJILLAS_TABLA` y
    a cualquier resolución."""
    return (d["modo"] == "libre" and d.get("av") == "primaria" and not d.get("verdad")
            and d.get("datos", "todo") == "todo" and d.get("rejilla", "btsettl") in REJILLAS_TABLA)


def en_el_borde(d):
    """Los parámetros ('teff', 'logg') cuyo 2.5 o 97.5 % cae a menos de medio paso del borde de su rejilla."""
    rej = REJILLAS[d.get("rejilla", "btsettl")]
    fuera = []
    for p in ("teff", "logg"):
        x0, paso, n = rej[p]
        q = d["posterior"][p]
        if q["p2_5"] - x0 < paso / 2 or x0 + paso * (n - 1) - q["p97_5"] < paso / 2:
            fuera.append(p)
    return fuera


def pares_que_encierran(plantillas, teff):
    """Los pares de plantillas consecutivas en tipo cuyas T_eff ajustadas encierran `teff` (la regla de
    `~/muse_noche_20261017/analiza.py`, `encierran`). `plantillas`: dicts con 'spt', 'codigo' (`spt_code`) y
    'teff', en el orden de desempate (el de sus etiquetas); se ordenan por 'codigo' de forma estable, así que dos
    plantillas del mismo tipo quedan en ese orden."""
    cc = sorted(plantillas, key=lambda c: c["codigo"])
    return [(a, b) for a, b in zip(cc, cc[1:]) if min(a["teff"], b["teff"]) <= teff <= max(a["teff"], b["teff"])]


def tipo_equivalente(plantillas, teff, rejilla=None):
    """El tipo espectral equivalente de un compañero: su T_eff (con el prior en A_V) entre las de plantillas de
    tipo conocido ajustadas con la MISMA rejilla y el ruido del compañero (Tabla B.5 y Sects. 4.1–4.2 del paper).

    - Un solo par de plantillas consecutivas la encierra (`pares_que_encierran`): el tipo es el par, `A–B`, o `A`
      si las dos son del mismo tipo. Si una plantilla del par está saturada contra un borde de la rejilla (a
      menos de medio paso, con `rejilla`), el par no ordena las T_eff: se lee la plantilla del par más cercana
      (DRIFT-PHOENIX con el ruido de ROXs 12 b, cuyas M6.5–M8 dan 2960–2990 K; `autor_9b.md` §2 del 2026-10-23).
    - Ningún par, y más fría que todas: `posterior` al tipo más tardío («≥L1»: tan fría como la plantilla más fría
      de ese tipo o más). Más caliente que todas: `anterior` al más temprano.
    - Más de un par: ambiguo (`ValueError`); el tipo no es una lectura única y hay que mirarlo.

    Devuelve dict(clase='par'|'una'|'posterior'|'anterior', tipos=(spt, ...), pares=[...], saturadas=[...])."""
    pares = pares_que_encierran(plantillas, teff)
    if len(pares) > 1:
        raise ValueError("tipo equivalente ambiguo: " + ", ".join(f"{a['spt']}–{b['spt']}" for a, b in pares))
    satura = []
    if rejilla is not None:
        x0, paso, n = REJILLAS[rejilla]["teff"]
        satura = [p for p in plantillas if p["teff"] - x0 < paso / 2 or x0 + paso * (n - 1) - p["teff"] < paso / 2]
    nombres = lambda ps: [p.get("plantilla", p["spt"]) for p in ps]  # noqa: E731
    out = dict(pares=[(a["spt"], b["spt"]) for a, b in pares], saturadas=nombres(satura))
    if pares:
        a, b = pares[0]
        if any(p is a or p is b for p in satura):
            cerca = min((a, b), key=lambda p: abs(p["teff"] - teff))
            return dict(out, clase="una", tipos=(cerca["spt"],))
        if a["spt"] == b["spt"]:
            return dict(out, clase="una", tipos=(a["spt"],))
        return dict(out, clase="par", tipos=(a["spt"], b["spt"]))
    if teff < min(p["teff"] for p in plantillas):
        ultimo = max(plantillas, key=lambda p: p["codigo"])["spt"]
        return dict(out, clase="posterior", tipos=(ultimo,))
    primero = min(plantillas, key=lambda p: p["codigo"])["spt"]
    return dict(out, clase="anterior", tipos=(primero,))


def filas_rejillas(corridas, deltas, nombres):
    """El cuerpo LaTeX de la tabla de rejillas del apéndice B: por compañero (orden de `nombres`) y por rejilla
    (orden de `REJILLAS_TABLA`, primero a R = 500 y después a R = 250), la mediana y el 16–84 % de T_eff, log g y
    A_V del ajuste con el prior en A_V, la mediana de Δ con A_V 1.8, 3.0 y con el prior, y el ln Z del ajuste con
    el prior menos el de BT-Settl a la misma resolución. Un valor con el 2.5 o el 97.5 % en el borde de su rejilla
    lleva `\\tablefootmark{a}`. `deltas`: etiqueta -> `percentiles` del margen. Lo que falta va como `\\ldots`."""
    filas = []
    for slug, nombre in nombres.items():
        suyas = {}
        for d in corridas:
            if d["slug"] != slug or not _de_rejillas(d):
                continue
            clave = (d.get("r_analisis", 500), d.get("rejilla", "btsettl"), d.get("av_fija"))
            if clave in suyas:
                raise SystemExit(f"dos corridas de {slug} con {clave}: {suyas[clave]['etiqueta']} y {d['etiqueta']}")
            suyas[clave] = d
        grupos = sorted({k[:2] for k in suyas}, key=lambda k: (-k[0], list(REJILLAS_TABLA).index(k[1])))
        if not grupos:
            continue
        filas.append(rf"\multicolumn{{8}}{{l}}{{{nombre}}} \\")
        for r, rejilla in grupos:
            prior = suyas.get((r, rejilla, None))
            ref = suyas.get((r, "btsettl", None))
            celdas = [REJILLAS_TABLA[rejilla] + (r" (\(R = 250\))" if r == 250 else "")]
            if prior is None:
                celdas += [r"\ldots"] * 3
            else:
                borde = en_el_borde(prior)
                marca = lambda p: r"\tablefootmark{a}" if p in borde else ""
                P = prior["posterior"]
                celdas += [_pm(P["teff"]) + marca("teff"), _pm(P["logg"], 2) + marca("logg"), _pm(P["av"], 2)]
            for av in (1.8, 3.0, None):
                d = suyas.get((r, rejilla, av))
                celdas.append(f"${deltas[d['etiqueta']]['p50']:+.2f}$" if d is not None else r"\ldots")
            celdas.append(f"${prior['logz'] - ref['logz']:.1f}$" if prior is not None and ref is not None
                          else r"\ldots")
            filas.append("\\quad " + " & ".join(celdas) + r" \\")
    return filas


def tabla_rejillas(dirs):
    """Imprime `filas_rejillas` para las corridas de `dirs`, con el margen de cada muestra (como `tabla`)."""
    from scripts import tipo_ir_indices as tir
    todos = [d for d in _carga(dirs) if _de_rejillas(d)]
    slugs = sorted({d["slug"] for d in todos})
    nombres = {s: tir.perfil(s)["paper"]["display"].replace(" ", "~") for s in slugs}
    entradas = {s: entrada_margen(s) for s in slugs}
    deltas = {d["etiqueta"]: percentiles(margen_de_corrida(d, entradas[d["slug"]])) for d in todos}
    filas = filas_rejillas(todos, deltas, nombres)
    print("\n".join(filas))
    return filas


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="orden", required=True)
    aa = sub.add_parser("ajusta", help="una corrida de muestreo anidado (o `evo_solo`)")
    aa.add_argument("slug")
    aa.add_argument("--modo", choices=("libre", "evo", "evo_solo"), default="libre")
    aa.add_argument("--datos", choices=DATOS, default="todo", help="todos los espectros, solo MUSE o solo J/H/K")
    aa.add_argument("--r250", action="store_true",
                    help="píxeles de a dos (λ/Δλ ≈ 250); automático con una rejilla de menos resolución (SPHINX)")
    aa.add_argument("--tracks", choices=tuple(TRACKS), default="ATMO2020")
    aa.add_argument("--av", choices=("libre", "primaria"), default="primaria")
    aa.add_argument("--nlive", type=int, default=500)
    aa.add_argument("--semilla", type=int, default=0)
    aa.add_argument("--inyecta-bt", type=float, nargs=4, metavar=("TEFF", "LOGG", "AV", "R_RJ"))
    aa.add_argument("--plantilla")
    aa.add_argument("--av-inyectada", type=float)
    aa.add_argument("--av-fija", type=float, help="perfil: A_V fija en este valor")
    aa.add_argument("--rejilla", choices=tuple(REJILLAS), default="btsettl")
    aa.add_argument("--dir-rejilla", type=Path, help="directorio de npz de la rejilla (por defecto el de REJILLAS)")
    aa.add_argument("--etiqueta", default=None)
    aa.add_argument("--salida", type=Path, required=True)
    ar = sub.add_parser("resume", help="tablas y margen cromosférico de uno o varios directorios de `ajusta`")
    ar.add_argument("dirs", type=Path, nargs="+")
    ar.add_argument("--json", type=Path)
    at = sub.add_parser("tabla", help="cuerpo LaTeX de la tabla de las cadenas del apéndice B del paper")
    at.add_argument("dirs", type=Path, nargs="+")
    ag = sub.add_parser("rejillas", help="cuerpo LaTeX de la tabla de rejillas del apéndice B del paper")
    ag.add_argument("dirs", type=Path, nargs="+")
    a = ap.parse_args(argv)
    if a.orden == "ajusta":
        return ajusta(a)
    if a.orden == "tabla":
        tabla(a.dirs)
        return 0
    if a.orden == "rejillas":
        tabla_rejillas(a.dirs)
        return 0
    resume(a.dirs, a.json)
    return 0


if __name__ == "__main__":
    sys.exit(main())
