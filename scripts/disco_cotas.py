#!/usr/bin/env python
"""Cotas del posible disco circumplanetario (CPD) de los dos compañeros: las cifras de la Sect. 5.1 del paper
(párrafo «These data do not detect a disk…» y la frase de L′) y las del factor magnetosférico de la sección de
métodos (tras «the prefactor is 1, and every Ṁ would be 0.10 dex lower»).

Hermano de `paper_cifras_extincion.py`, con la misma regla: lee lo que las etapas escribieron y no inventa
entradas. Porta al pipeline el cálculo del agente `disco` del 2026-10-26 (`~/muse_sesion_20261026/claudio/disco/`,
`disco.py` y `extrae_btsettl.py`), que vivía fuera del repo.

De dónde sale cada entrada:

- **Medidas de este trabajo**, de los productos: Ṁ estelar de G3 (`mdot_p50_msun_yr`, detección) o de E3 (el límite
  al 99 % del método canónico); Ṁ planetaria (Aoyama et al. 2021) con el factor en L_acc de la misma L_Hα (detección)
  o la fila `mdot_aoyama21` de E3 (límite); separación de B3 (`astrometry.sep_arcsec`); r_K por la profundidad del
  CO (`scripts/veiling.py`, `r_k_co`); T_eff y r_K de los ajustes de la Tabla B.6 (las cadenas de
  `scripts/teff_bayes.py`, `--teff-chains`).
- **Adoptadas en el config del run** (`h03_*`): masa, radio, distancia y A_V de la primaria. L_bol: `targets/<slug>.json`
  → `fisica.log_lbol`.
- **Literatura**: `scripts/data/disco_literatura/disco_literatura.csv` (cada fila con su cita en el README).
- **BT-Settl** (Allard et al. 2012): los VOTables de rango completo de SVO en `/mnt/2TB/IR_Data/btsettl_cifist/raw`
  (los npz de `teff_bayes.py` llegan solo a 2.55 µm y aquí hacen falta L′ y M′), promediados en intervalos
  logarítmicos de λ/Δλ = 200 entre 0.9 y 40 µm, como `extrae_btsettl.py` (`--r-btsettl` lo cambia).

Etiquetas, las del agente: **[M]** medida, **[D]** derivada con supuestos explícitos, **[E]** especulativa.

1. *Energía* [D]. Polvo de cuerpo negro a T_d que añade un exceso r_K en K reemite f = (A/πR²)(T_d/T_eff)⁴ de la
   L_bol, con A/πR² = r_K F_K,fot/(π B_K(T_d)) (emisor isótropo equivalente; la geometría lo cambia en un factor
   ~2). F_K,fot es la fotosfera BT-Settl (log g 4.0) en los nodos que encierran la T_eff de L_bol y R y la mayor de
   los ajustes BT-Settl de la Tabla B.6. Paper: «0.8–0.9 r_K at 1500 K».
2. *Disco plano pasivo* [D] (Chiang & Goldreich 1997): T⁴ = T*⁴ [arcsin x − x√(1−x²)]/π, x = R/r, que reprocesa 1/4
   de L* entre R e ∞ y tiende a la Ec. 4 de CG97. Borde interior en el radio de sublimación de Monnier & Millan-Gabet
   (2002, Ec. 1), R_s/R = ½√Q_R (T*/T_s)²; con R_in = 1.5 R (el valor del agente), r_K ≲ 0.1. Paper: «gives
   r_K ≲ 0.1». El script da también el disco con R_in = R_s(1500 K, Q_R = 1 y 4): con polvo gris (Q_R = 1) el borde
   queda en 1.29 R (12 b) y 1.08 R (42B b) y el disco da r_K = 0.16 y 0.21, así que el «≲ 0.1» vale para R_in ≳ 1.5 R.
3. *Pared gruesa* [D/literatura]: cobertura ~20 % (Dullemond & Monnier 2010, Sect. 3.5); el script da además la
   escala de altura hidrostática H/r = c_s/v_K en la pared (μ = 2.34).
4. *r_K del CO* [M] y su f [D]: paper «35–40%» (12 b, r_K = 0.44) y «about 17%» (42B b, r_K ≈ 0.2); el script da
   0.34–0.38 y 0.17–0.18.
5. *Tabla B.6* [M → D]: el f de cada ajuste en flujo absoluto, a T_d = 1500 K. Paper «from 40% to more than the whole
   luminosity»; el script da 0.37–1.13 (el mínimo es el ajuste BT-Settl con prior de 12 b, r_K = 0.48).
6. *ALMA* [M → D]: radio máximo de un disco ópticamente grueso bajo el 3σ de Wu et al. (2020), con T(r) = max(T del
   disco abocinado de CG97 Ec. 1 con ángulo rasante 0.02–0.05, equilibrio con la primaria) y cos i = 0.5–1. Paper
   «smaller than about 1 au».
7. *Marea* [D]: R_H = a(1−e)(M_p/3M*)^{1/3} y r_t = R_H/3–0.4 R_H (Ayliffe & Bate 2009; Martin & Lubow 2011), con un
   Montecarlo de 2×10⁵ muestras (semilla 20261026); con la órbita publicada si la hay (Inglis et al. 2026) y, si no,
   a = ρ·U(1, 1.4), e ~ U(0, 0.5). Se cita el pericentro: el 16 % de R_H/3 y el 84 % de 0.4 R_H. Paper «12–24 au» y
   «4–14 au».
8. *22 µm de WISE* [M → D]: νL_ν(W4) del sistema sin resolver frente a L_bol del compañero, quitando (o no) la
   fotosfera de la primaria con el Ks−W4 de una M0 joven (Pecaut & Mamajek 2013, Tabla 6). Paper «about 0.9 times».
9. *K−L′* [M → D]: (K−L′)_b = ΔK − ΔL (Kraus et al. 2014, misma noche) + (K−L′)_primaria, con la primaria sin exceso
   (Bowler et al. 2017) o con el exceso WISE interpolado de W1/W2 a 3.78 µm; y la T_d más fría que permite cada r_K
   (cambio de color frente al margen observado + 2σ). Paper «K−L′ = 0.94–1.27» y «cooler than about 900 K» (en
   rejilla de 10 K: 920–990 K con el exceso WISE de la primaria, 1100–1200 K sin él, para r_K = 0.3–0.44).
10. *Magnetosfera* [D/E]: r_M = ξ(μ⁴/4GMṀ²)^{1/7}, μ = BR³, ξ = 0.7 (Hartmann, Herczeg & Calvet 2016, Ec. 1; su
    prefactor numérico «18ξ» no se usa: con sus propios valores fiduciales la forma con μ da 13.0 R☉);
    corrotación r_co = (GM P²/4π²)^{1/3} con P ≤ 2πR/v sin i. El R_in del pipeline (5R) sale de
    `MAGNETOSPHERIC_FACTOR`. Métodos: «0.03 kG … 0.13 kG», «0.3 kG … about 18R», «at most 6–15R». Sect. 5.1, [E]:
    «above about 0.1–0.2 kG would truncate the disk … outside the corotation radius».
11. *Disco de canto* [E]: columna Σ ∝ 1/r entre 2R y la marea, H/r = 0.05–0.2, M_disco = 10⁻⁸–10⁻⁴ M☉; probabilidad
    a priori de que una orientación al azar dé ΔA_V de 1 a 3 mag (N_H = 2.21×10²¹ A_V, Güver & Özel 2009). Paper
    «only 1–6%».

    python scripts/disco_cotas.py --project-root ../MUSE-accretion-pipeline \\
        --teff-chains ~/muse_noche_20261024_c4/cadenas/A1/res --teff-chains ~/muse_noche_20261024_c4/cadenas/A2/res_drift \\
        --salida DIR

Sin `--teff-chains` no hay Tabla B.6: el punto 5 se salta y el nodo caliente de la fotosfera queda en la T_eff de
L_bol y R. Con `--salida` se escribe `disco_cotas.json` y se guarda ahí el BT-Settl promediado
(`btsettl_ir_R200.npz`), que se reutiliza si coincide (leer los 24 VOTables tarda ~2 min).
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np
from astropy import constants as _cte

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from musepipe.constants import LSUN_ERG_S, MSUN_OVER_MJUP, PC_CM, RSUN_CM, SIGMA_SB_CGS  # noqa: E402
from musepipe.stages.stage_h03_limits import G_CGS, M_SUN_G, MAGNETOSPHERIC_FACTOR, YR_S  # noqa: E402

# ---------------------------------------------------------------- constantes cgs
#: las que `musepipe` no tiene, de astropy (CODATA 2018 / IAU 2012)
H_CGS = _cte.h.cgs.value
K_B_CGS = _cte.k_B.cgs.value
C_CGS = _cte.c.cgs.value
AU_CM = _cte.au.cgs.value
#: masa del átomo de H (peso atómico estándar 1.00784 u)
M_H_G = 1.00784 * _cte.u.cgs.value

# ---------------------------------------------------------------- supuestos declarados (no son literatura)
LITERATURA = ROOT / "scripts" / "data" / "disco_literatura" / "disco_literatura.csv"
BTSETTL_RAW = Path("/mnt/2TB/IR_Data/btsettl_cifist/raw")
#: nodos de BT-Settl que se leen (los de extrae_btsettl.py) y el rango en µm del promedio
TEFFS_BT = (2100, 2200, 2300, 2400, 2500, 2600, 2700, 2800)
LOGGS_BT = (3.5, 4.0, 4.5)
LAM_BT_UM = (0.9, 40.0)
R_BT = 200.0
#: bandas de caja (µm): aproximación declarada
BANDAS_UM = dict(K=(2.00, 2.30), Lp=(3.42, 4.12), Ms=(4.55, 4.79), I80=(6.40, 9.30), W4=(20.0, 25.0))
#: λ de K para la extinción (µm)
LAMBDA_K_UM = 2.15
#: Vega como cuerpo negro (error ~2 % entre 2 y 8 µm, según el agente `disco`)
T_VEGA_K = 9550.0
LOGG_FOTOSFERA = 4.0
T_POLVO_K = (500.0, 700.0, 900.0, 1200.0, 1500.0, 1800.0)
T_POLVO_REF_K = 1500.0
RK_REJILLA = (0.03, 0.1, 0.2, 0.3)
#: borde interior del disco plano (en R) y cos i; el radio exterior
RIN_PLANO_R = (1.5, 3.0, 5.0, 10.0)
COSI = (1.0, 0.5)
ROUT_PLANO_AU = 20.0
#: radio de sublimación: T_s y Q_R (Monnier & Millan-Gabet 2002 usan 1500 K como referencia)
T_SUB_K = (1200.0, 1500.0, 2000.0)
Q_R = (1.0, 4.0)
#: escala de altura hidrostática en la pared: r (en R), T (K) y peso molecular medio
PARED_R = (2.0, 4.0)
PARED_T_K = (1200.0, 1500.0)
MU_GAS = 2.34
#: Montecarlo de la marea
SEMILLA = 20261026
N_MC = 200_000
M_P_MIN_MJUP = 3.0
M_HOST_MIN_MSUN = 0.2
A_MIN_AU = 60.0
E_MAX = 0.9
#: órbita sin ajuste publicado: a = ρ·U(1, 1.4) y e ~ U(0, 0.5)
A_SOBRE_RHO = (1.0, 1.4)
E_UNIFORME = (0.0, 0.5)
#: disco ópticamente grueso: ángulo rasante del disco abocinado y rejilla radial (au)
ALFA_RASANTE = (0.02, 0.05)
RADIOS_ALMA_AU = (0.01, 60.0, 3000)
#: magnetosfera: campos dipolares (G)
CAMPOS_G = (100.0, 300.0, 1000.0, 3000.0)
#: disco de canto: masas (M☉), H/r, borde interior (en R) y ventana de ΔA_V (mag)
M_DISCO_MSUN = (1e-8, 1e-6, 1e-4)
H_SOBRE_R = (0.05, 0.1, 0.2)
RIN_CANTO_R = 2.0
DAV_VENTANA = (1.0, 3.0)
#: gas por átomo de H (masa por H, MIE)
MASA_POR_H = 1.4


# ---------------------------------------------------------------- fórmulas (sin datos)
def planck_nu(nu, T):
    """B_ν(T) en erg s⁻¹ cm⁻² Hz⁻¹ sr⁻¹."""
    x = H_CGS * nu / (K_B_CGS * T)
    with np.errstate(over="ignore"):
        return 2 * H_CGS * nu ** 3 / C_CGS ** 2 / np.expm1(x)


def teff_de_lbol(log_lbol, radio_cm):
    """T_eff de L = 4πR²σT⁴."""
    return (10 ** log_lbol * LSUN_ERG_S / (4 * np.pi * radio_cm ** 2 * SIGMA_SB_CGS)) ** 0.25


def radio_sublimacion(t_estrella, t_sub, q_r=1.0):
    """[D] Monnier & Millan-Gabet (2002), Ec. 1: R_s/R* = ½ √Q_R (T*/T_s)²."""
    return 0.5 * np.sqrt(q_r) * (t_estrella / t_sub) ** 2


def radio_de_hill(a, e, m_p, m_estrella):
    """[D] R_H = a (1 − e) (M_p/3M*)^{1/3}, en las unidades de `a`; con e > 0, en el pericentro."""
    return a * (1 - e) * (m_p / (3 * m_estrella)) ** (1 / 3)


def fraccion_reemitida(r_k, f_fot_k, b_k_polvo, t_polvo, teff):
    """[D] f = L_polvo/L* de un cuerpo negro a T_polvo que añade r_K en K, como emisor isótropo:
    f = (A/πR²)(T_d/T_eff)⁴ con A/πR² = r_K F_K,fot / (π B_K(T_d)); F_K,fot es el flujo de superficie."""
    return r_k * f_fot_k / (np.pi * b_k_polvo) * (t_polvo / teff) ** 4


def temperatura_disco_plano(r_sobre_R, teff):
    """[D] Disco plano pasivo (Chiang & Goldreich 1997): T⁴ = T*⁴ [arcsin x − x√(1−x²)]/π, x = R/r."""
    x = 1 / r_sobre_R
    return teff * ((np.arcsin(x) - x * np.sqrt(1 - x ** 2)) / np.pi) ** 0.25


def fraccion_disco_plano(r_sobre_R, teff):
    """Fracción de L* = 4πR²σT*⁴ que reemiten las dos caras de un disco plano entre r[0] y r[-1] (en R)."""
    T = temperatura_disco_plano(r_sobre_R, teff)
    ld = 2 * np.trapz(2 * np.pi * r_sobre_R * SIGMA_SB_CGS * T ** 4, r_sobre_R)
    return float(ld / (4 * np.pi * SIGMA_SB_CGS * teff ** 4))


def escala_altura(masa_g, radio_cm, T, r_sobre_R, mu=MU_GAS):
    """[D] H/r = c_s/v_K con c_s² = kT/(μ m_H) y v_K² = GM/r."""
    cs = np.sqrt(K_B_CGS * T / (mu * M_H_G))
    vk = np.sqrt(G_CGS * masa_g / (r_sobre_R * radio_cm))
    return float(cs / vk)


def temperatura_equilibrio(l_lsun, r_cm):
    """Cuerpo negro en equilibrio a la distancia r de una estrella: T = [L/(16πσr²)]^{1/4} (≈ 278.3 K a 1 au de L☉)."""
    return (l_lsun * LSUN_ERG_S / (16 * np.pi * SIGMA_SB_CGS * r_cm ** 2)) ** 0.25


def radio_max_grueso(f_lim_cgs, d_cm, nu, radio_cm, teff, t_fondo, alfa, cosi, radios_au=RADIOS_ALMA_AU):
    """[D] Radio (au) a partir del cual un disco ópticamente grueso supera `f_lim_cgs` a la frecuencia ν: T(r) =
    max((α/2)^{1/4} (R/r)^{1/2} T*, T_fondo) (Chiang & Goldreich 1997, Ec. 1), flujo = cos i ∫ 2πr B_ν(T) dr / d².
    `inf` si ni el disco entero de la rejilla llega al límite."""
    rr = np.geomspace(radios_au[0], radios_au[1], radios_au[2]) * AU_CM
    t_irr = (alfa / 2) ** 0.25 * np.sqrt(radio_cm / rr) * teff
    T = np.maximum(t_irr, t_fondo)
    acum = np.concatenate([[0], np.cumsum(np.diff(rr) * 2 * np.pi * rr[1:] * planck_nu(nu, T[1:]))])
    flujo = cosi * acum / d_cm ** 2
    rmax = float(np.interp(f_lim_cgs, flujo, rr) / AU_CM) if flujo[-1] > f_lim_cgs else math.inf
    return dict(alfa=alfa, cosi=cosi, rmax_au=rmax, T_1au_K=float(np.interp(AU_CM, rr, T)))


def radio_magnetosferico(B, radio_cm, masa_g, mdot_g_s, xi):
    """[D] r_M = ξ (μ⁴/4GMṀ²)^{1/7}, μ = BR³ (Hartmann, Herczeg & Calvet 2016, Ec. 1), en cm."""
    mu = B * radio_cm ** 3
    return xi * (mu ** 4 / (4 * G_CGS * masa_g * mdot_g_s ** 2)) ** (1 / 7)


def campo_para_radio(r_m_cm, radio_cm, masa_g, mdot_g_s, xi):
    """El dipolo B (G) con el que r_M = r_m: la inversa de `radio_magnetosferico`."""
    mu = ((r_m_cm / xi) ** 7 * 4 * G_CGS * masa_g * mdot_g_s ** 2) ** 0.25
    return mu / radio_cm ** 3


def tasa_para_radio(r_m_cm, B, radio_cm, masa_g, xi):
    """La Ṁ (g/s) con la que r_M = r_m para un dipolo B."""
    mu = B * radio_cm ** 3
    return np.sqrt(mu ** 4 / (4 * G_CGS * masa_g) / (r_m_cm / xi) ** 7)


def radio_corrotacion(radio_cm, masa_g, vsini_cm_s):
    """[D] Cota superior de r_co (cm): P ≤ 2πR/(v sin i) (sin i ≤ 1) y r_co = (GMP²/4π²)^{1/3}."""
    p_max = 2 * np.pi * radio_cm / vsini_cm_s
    return (G_CGS * masa_g * p_max ** 2 / (4 * np.pi ** 2)) ** (1 / 3), p_max


def angulo_de_columna(col_plano, columna, h):
    """Elevación θ (rad) a la que la columna de un disco gaussiano en altura, N(θ) = N_0 exp(−θ²/2h²), vale `columna`."""
    return h * np.sqrt(2 * np.log(col_plano / columna)) if col_plano > columna else 0.0


def probabilidad_canto(m_disco_g, r_in_cm, r_d_cm, h, sigma_lo, sigma_hi):
    """[E] Σ(r) ∝ 1/r entre r_in y r_d con H/r = h: columna hacia el objeto en el plano y probabilidad a priori
    |sin θ_lo − sin θ_hi| de que una orientación al azar dé una columna entre `sigma_lo` y `sigma_hi` (g/cm²)."""
    s0 = m_disco_g / (2 * np.pi * r_d_cm * (r_d_cm - r_in_cm))
    col0 = s0 * r_d_cm * (1 / r_in_cm - 1 / r_d_cm) / (np.sqrt(2 * np.pi) * h)
    th_lo, th_hi = angulo_de_columna(col0, sigma_lo, h), angulo_de_columna(col0, sigma_hi, h)
    return dict(col_plano_gcm2=float(col0), theta_lo_deg=float(np.degrees(th_lo)),
                theta_hi_deg=float(np.degrees(th_hi)), P=float(abs(np.sin(th_lo) - np.sin(th_hi))))


def cambio_de_color(r_1, r_2):
    """Δ(1−2) = 2.5 log[(1 + r_2)/(1 + r_1)]: lo que enrojece el color 1−2 un exceso r_1 en la banda 1 y r_2 en la 2."""
    return 2.5 * np.log10((1 + r_2) / (1 + r_1))


def a_lambda_sobre_av_ir(lam_um, rv=3.1):
    """Cardelli, Clayton & Mathis (1989), rama IR (Ec. 2a–b): (0.574 − 0.527/R_V) x^1.61, x = 1/λ[µm]. Válida para
    0.9–3.3 µm (la misma que `CCMExtinction`); más allá es una extrapolación declarada."""
    return (0.574 - 0.527 / rv) * (1 / lam_um) ** 1.61


def nu_l_nu_lsun(mag, cero_jy, lam_um, d_cm):
    """νL_ν (L☉) de una magnitud Vega con su F_ν de magnitud cero."""
    fnu = cero_jy * 10 ** (-0.4 * mag) * 1e-23
    return 4 * np.pi * d_cm ** 2 * (C_CGS / (lam_um * 1e-4)) * fnu / LSUN_ERG_S, fnu / 1e-26


def _split_normal(rng, c, up, lo, n):
    z = rng.standard_normal(n)
    return np.where(z > 0, c + up * z, c + lo * z)


def _pct(x, q=(16, 50, 84)):
    return [float(v) for v in np.percentile(x, q)]


# ---------------------------------------------------------------- literatura
class Literatura:
    """Las filas de `disco_literatura.csv`; `objeto` = slug o `*`."""

    def __init__(self, ruta=LITERATURA):
        self.ruta = Path(ruta)
        with open(self.ruta, newline="") as fh:
            self.filas_ = [r for r in csv.DictReader(l for l in fh if not l.lstrip().startswith("#"))]
        for r in self.filas_:
            r["valor_num"] = float(Fraction(r["valor"]))
            for k in ("err_mas", "err_menos"):
                r[k + "_num"] = float(Fraction(r[k])) if r[k] else None

    def filas(self, clave, objeto="*"):
        propias = [r for r in self.filas_ if r["clave"] == clave and r["objeto"] == objeto]
        return propias or [r for r in self.filas_ if r["clave"] == clave and r["objeto"] == "*"]

    def tiene(self, clave, objeto="*"):
        return bool(self.filas(clave, objeto))

    def valor(self, clave, objeto="*"):
        f = self.filas(clave, objeto)
        if len(f) != 1:
            raise SystemExit(f"{self.ruta.name}: {len(f)} filas para {clave!r} ({objeto}); se esperaba una")
        return f[0]["valor_num"]

    def con_error(self, clave, objeto="*"):
        f = self.filas(clave, objeto)[0]
        return f["valor_num"], f["err_mas_num"], f["err_menos_num"]


# ---------------------------------------------------------------- fotosfera BT-Settl
def promedia_votable(ruta, bordes_um, w_min_um=0.85, w_max_um=41.0):
    """Media de F_λ (erg/s/cm²/Å, superficie) en cada intervalo de `bordes_um` (integración trapezoidal / ancho):
    lo que hace `extrae_btsettl.py` del agente `disco`, que recorta el VOTable a 0.85–41 µm."""
    from astropy.io.votable import parse_single_table

    tab = parse_single_table(str(ruta)).to_table()
    w = np.asarray(tab["WAVELENGTH"], float) / 1e4
    fl = np.asarray(tab["FLUX"], float)
    m = (w > w_min_um) & (w < w_max_um) & np.isfinite(fl)
    w, fl = w[m], fl[m]
    o = np.argsort(w)
    w, fl = w[o], fl[o]
    acum = np.concatenate([[0], np.cumsum(0.5 * (fl[1:] + fl[:-1]) * np.diff(w))])
    return np.diff(np.interp(bordes_um, w, acum)) / np.diff(bordes_um)


def bordes_logaritmicos(lo_um=LAM_BT_UM[0], hi_um=LAM_BT_UM[1], r=R_BT):
    return np.exp(np.arange(np.log(lo_um), np.log(hi_um), 1 / r))


def carga_btsettl(raw=BTSETTL_RAW, r=R_BT, cache=None, teffs=TEFFS_BT, loggs=LOGGS_BT):
    """`(lam_um, teff, logg, flam)` de los nodos de BT-Settl, promediados a λ/Δλ = r. Si `cache` existe y es de la
    misma rejilla se lee; si no, se calcula (~5 s por VOTable) y, con `cache`, se guarda."""
    bordes = bordes_logaritmicos(r=r)
    if cache is not None and Path(cache).exists():
        z = np.load(cache)
        if (float(z["r"]) == r and tuple(z["teffs"]) == tuple(teffs) and tuple(z["loggs"]) == tuple(loggs)
                and str(z["raw"]) == str(raw)):
            return z["lam_um"], z["teff"], z["logg"], z["flam"]
    filas, meta = [], []
    for t in teffs:
        for g in loggs:
            f = Path(raw) / f"teff{t}_logg{g:.1f}.xml"
            if not f.exists():
                raise SystemExit(f"falta {f}: sin la rejilla BT-Settl de rango completo no hay fotosfera en L′ ni M′")
            filas.append(promedia_votable(f, bordes))
            meta.append((t, g))
    lam = np.sqrt(bordes[1:] * bordes[:-1])
    teff = np.array([m[0] for m in meta])
    logg = np.array([m[1] for m in meta])
    flam = np.array(filas)
    if cache is not None:
        Path(cache).parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache, lam_um=lam, edges_um=bordes, teff=teff, logg=logg, flam=flam, r=r,
                 teffs=np.array(teffs), loggs=np.array(loggs), raw=str(raw))
    return lam, teff, logg, flam


class Fotosfera:
    """Fotosfera BT-Settl en ν, con medias de banda de caja y colores Vega (Vega = cuerpo negro de 9550 K)."""

    def __init__(self, lam_um, teff, logg, flam):
        self.lam = np.asarray(lam_um, float)
        self.nu = C_CGS / (self.lam * 1e-4)
        self.teff = np.asarray(teff, float)
        self.logg = np.asarray(logg, float)
        fl = np.array(flam, float)
        # un nodo de SVO (2700 K, log g 4.0) viene a cero por encima de 23.4 µm: se completa con Rayleigh-Jeans
        # (F_λ ∝ λ⁻⁴) desde el último intervalo bueno; solo afecta a la banda W4 de ese nodo.
        for i in range(fl.shape[0]):
            malo = ~(fl[i] > 0)
            if malo.any():
                j = np.where(~malo)[0].max()
                fl[i, malo] = fl[i, j] * (self.lam[malo] / self.lam[j]) ** -4
        self.flam = fl

    def fnu(self, teff, logg=LOGG_FOTOSFERA):
        """F_ν de superficie (erg/s/cm²/Hz): el nodo de log g más cercano, interpolado en log F entre los dos T_eff
        que encierran `teff` (fuera de la rejilla, el nodo del borde)."""
        gsel = self.logg == self.logg[np.argmin(np.abs(self.logg - logg))]
        ts = np.unique(self.teff[gsel])
        t0 = ts[ts <= teff].max() if np.any(ts <= teff) else ts.min()
        t1 = ts[ts >= teff].min() if np.any(ts >= teff) else ts.max()
        f0 = self.flam[gsel & (self.teff == t0)][0]
        f1 = self.flam[gsel & (self.teff == t1)][0]
        w = 0.0 if t1 == t0 else (teff - t0) / (t1 - t0)
        flam = np.exp((1 - w) * np.log(f0) + w * np.log(f1))
        return flam * (self.lam * 1e-4) ** 2 / C_CGS * 1e8

    def media(self, fnu, banda):
        lo, hi = BANDAS_UM[banda]
        m = (self.lam >= lo) & (self.lam <= hi)
        return float(np.trapz(fnu[m], self.nu[m]) / np.trapz(np.ones(m.sum()), self.nu[m]))

    def cuerpo_negro(self, T, banda):
        return self.media(planck_nu(self.nu, T), banda)

    def color(self, fnu, b1, b2):
        vega = self.cuerpo_negro(T_VEGA_K, b1) / self.cuerpo_negro(T_VEGA_K, b2)
        return -2.5 * np.log10((self.media(fnu, b1) / self.media(fnu, b2)) / vega)

    def por_rk(self, teff, t_polvo, banda="Lp"):
        """`(f/r_K, r_banda/r_K)`: la luminosidad reemitida por unidad de r_K y el exceso en `banda` por unidad de
        r_K de un cuerpo negro a `t_polvo` sobre la fotosfera de `teff`."""
        fph = self.fnu(teff)
        f = fraccion_reemitida(1.0, self.media(fph, "K"), self.cuerpo_negro(t_polvo, "K"), t_polvo, teff)
        r_b = ((self.cuerpo_negro(t_polvo, banda) / self.cuerpo_negro(t_polvo, "K"))
               / (self.media(fph, banda) / self.media(fph, "K")))
        return float(f), float(r_b)

    def disco_plano(self, teff, radio_cm, rin_R, cosi, rout_au=ROUT_PLANO_AU, bandas=("K", "Lp")):
        """[D] Exceso r_banda de un disco plano pasivo entre `rin_R` y `rout_au` visto con cos i, y la fracción de L*
        que reprocesa (dos caras)."""
        fph = self.fnu(teff)
        r = np.geomspace(rin_R, rout_au * AU_CM / radio_cm, 4000)
        T = temperatura_disco_plano(r, teff)
        out = {}
        for b in bandas:
            bb = np.array([self.cuerpo_negro(t, b) for t in T])
            out[f"r_{b}"] = float(cosi * np.trapz(2 * np.pi * r * bb, r) / self.media(fph, b))
        out["T_in_K"] = float(T[0])
        out["f_L"] = fraccion_disco_plano(r, teff)
        return out


# ---------------------------------------------------------------- entradas del run
def _mdot_del_objeto(obj, phys):
    """Ṁ estelar y planetaria de la Tabla 5, de los productos: la detección, de G3 (y la planetaria con el factor en
    L_acc de Aoyama et al. 2021 frente a la estelar a la misma L_Hα, como la nota b de la tabla); el límite, de E3."""
    from musepipe.stages.stage_h03_limits import lacc_lsun_from_lha
    from scripts import paper_figures as pf

    e = pf._entrada_extincion(obj)
    ao = phys.get("lacc_aoyama21")
    if not ao:
        raise SystemExit(f"{obj.nombre}: el config no declara g3_lacc_relations.halpha_aoyama21")
    if not e["limite"]:
        lha, lacc, _ = pf._cadena_mdot(phys, e["f_obs"], phys["av"])
        factor = lacc_lsun_from_lha(lha, ao["a"], ao["b"]) / lacc
        return dict(estelar=e["mdot_pub"], planetaria=factor * e["mdot_pub"], limite=False,
                    factor_aoyama=factor, procedencia="G3 mdot_p50 (detección); planetaria = factor × estelar")
    h03 = obj.qc("stage_h03_qc.json")
    fila = next(L for L in h03["limits"] if L["method"] == h03["canonical_method"])
    return dict(estelar=e["mdot_pub"], planetaria=float(fila["mdot_aoyama21"]), limite=True,
                factor_aoyama=float(fila["mdot_aoyama21"]) / e["mdot_pub"],
                procedencia=f"E3, límite del método canónico ({h03['canonical_method']})")


def entradas_del_objeto(obj, lit):
    """Las entradas de un objeto: medidas y adoptadas del run, L_bol de `targets/`, literatura del CSV."""
    from musepipe.stages.stage_h03_limits import physical_inputs_from_config
    from musepipe.targets import load_target

    cfg = obj.config
    phys = physical_inputs_from_config(dict(cfg))
    ficha = load_target(obj.slug, project_root=ROOT) or {}
    if "log_lbol" not in (ficha.get("fisica") or {}):
        raise SystemExit(f"targets/{obj.slug}.json no declara fisica.log_lbol")
    s = obj.slug
    e = dict(slug=s, nombre=obj.nombre, run=obj.run_id,
             d_pc=float(phys["distance_pc"]),
             rho_arcsec=float(obj.qc("stage01c_qc.json")["astrometry"]["sep_arcsec"]),
             m_msun=float(phys["companion_mass_msun"]), m_err_msun=float(cfg["h03_companion_mass_err_msun"]),
             r_rsun=float(phys["companion_radius_rsun"]), log_lbol=float(ficha["fisica"]["log_lbol"]),
             av=float(phys["av"]), mdot=_mdot_del_objeto(obj, phys))
    e["m_host_msun"] = lit.con_error("masa_primaria", s)
    if lit.tiene("log_lbol_primaria", s):
        e["l_host_lsun"] = 10 ** lit.valor("log_lbol_primaria", s) * (e["d_pc"] / lit.valor("d_lbol_primaria", s)) ** 2
    else:
        e["l_host_lsun"] = (10 ** (-0.4 * (lit.valor("mbol_primaria", s) - lit.valor("mbol_sol")))
                            * (e["d_pc"] / lit.valor("d_mbol_primaria", s)) ** 2)
    e["orbita"] = (dict(a=lit.con_error("a_orbita", s), e=lit.con_error("e_orbita", s))
                   if lit.tiene("a_orbita", s) else None)
    e["e_max95"] = lit.valor("e_max95", s)
    flujo, err, _ = lit.con_error("alma_flujo", s)
    e["alma_3sigma_mjy"] = 3 * err
    e["vsini_kms"] = {r["fuente"]: r["valor_num"] for r in lit.filas("vsini", s)}
    if lit.tiene("contraste_K", s):
        e["fotometria"] = {k: lit.valor(k, s) for k in (
            "contraste_K", "contraste_Lp", "ks_primaria", "w1_primaria", "w2_primaria", "w3_primaria",
            "w4_primaria", "kml_primaria_supuesta", "ks_w1_fotosfera", "ks_w2_fotosfera", "ks_w3_fotosfera",
            "ks_w4_fotosfera")}
        e["fotometria"]["kml_err"] = lit.con_error("kml_primaria_supuesta", s)[1]
    return e


def carga_cadenas(dirs):
    """{slug: [(rejilla, A_V, T_eff, r_K)]} de los ajustes de la Tabla B.6 (`teff_bayes._de_la_tabla`), en el orden
    de la tabla. Medianas de la posterior."""
    from scripts.teff_bayes import NOMBRE_REJILLA, _carga, _de_la_tabla

    out = {}
    for d in _carga(dirs):
        if not _de_la_tabla(d):
            continue
        av = "prior" if d.get("av_fija") is None else float(d["av_fija"])
        out.setdefault(d["slug"], []).append(dict(
            rejilla=NOMBRE_REJILLA[d.get("rejilla", "btsettl")], av=av,
            teff=float(d["posterior"]["teff"]["p50"]), rk=float(d["posterior"]["rk"]["p50"]),
            _orden=(list(NOMBRE_REJILLA).index(d.get("rejilla", "btsettl")), math.inf if av == "prior" else av)))
    for filas in out.values():
        filas.sort(key=lambda f: f["_orden"])
        for f in filas:
            f.pop("_orden")
    return out


def rk_del_co(slug, raiz):
    """{espectro: r_K} por la profundidad del CO (`scripts/veiling.py`, Apéndice B del paper), o None sin datos."""
    import warnings

    try:
        from scripts import veiling
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            r = veiling.r_k_co(slug, root=raiz)
    except (OSError, RuntimeError, SystemExit, FileNotFoundError) as exc:
        print(f"% aviso: sin r_K del CO para {slug} ({exc})")
        return None
    return {k: float(v["r_K"]) for k, v in r["espectros"].items()}


# ---------------------------------------------------------------- cálculo
def marea(entradas, fracciones, semilla=SEMILLA, n=N_MC):
    """[D] Radio de Hill y de truncamiento por marea, por objeto. UN generador para todos, en el orden de
    `entradas` (el de `paper.order`): así lo hacía el agente y así se reproducen sus percentiles."""
    f_min, f_max = fracciones
    rng = np.random.default_rng(semilla)
    out = {}
    for e in entradas:
        mp = np.clip(rng.normal(e["m_msun"], e["m_err_msun"], n), M_P_MIN_MJUP / MSUN_OVER_MJUP, None)
        ms = np.clip(_split_normal(rng, *e["m_host_msun"], n), M_HOST_MIN_MSUN, None)
        rho = e["rho_arcsec"] * e["d_pc"]
        if e["orbita"]:
            a = np.clip(_split_normal(rng, *e["orbita"]["a"], n), A_MIN_AU, None)
            ecc = np.clip(_split_normal(rng, *e["orbita"]["e"], n), 0, E_MAX)
            orbita = "publicada"
        else:
            a = rho * rng.uniform(*A_SOBRE_RHO, n)
            ecc = rng.uniform(*E_UNIFORME, n)
            orbita = f"supuesta: a = rho*U{A_SOBRE_RHO}, e ~ U{E_UNIFORME}"
        rh_rho = radio_de_hill(rho, 0.0, mp, ms)
        rh_peri = radio_de_hill(a, ecc, mp, ms)
        r = dict(orbita=orbita, rho_au=rho, q=_pct(mp / ms), RH_rho_au=_pct(rh_rho), a_au=_pct(a),
                 RH_peri_au=_pct(rh_peri))
        for f, etq in ((f_min, "min"), (f_max, "max")):
            r[f"rt_{etq}_rho_au"] = _pct(f * rh_rho)
            r[f"rt_{etq}_peri_au"] = _pct(f * rh_peri)
        r["rt_max_emax95_au"] = _pct(f_max * radio_de_hill(rho, e["e_max95"], mp, ms))
        r["rt_peri_16_84_au"] = [r["rt_min_peri_au"][0], r["rt_max_peri_au"][2]]
        out[e["slug"]] = r
    return out


def nodos_fotosfera(teff_lr, cadenas_obj, paso=100.0):
    """[lo, hi] de la fotosfera de referencia: la T_eff de L_bol y R y la mayor de los ajustes BT-Settl de la Tabla
    B.6, cada una al nodo de 100 K más cercano (la elección del agente: 2400–2700 K y 2200–2300 K)."""
    lo = paso * round(teff_lr / paso)
    bt = [c["teff"] for c in (cadenas_obj or []) if c["rejilla"] == "BT-Settl"]
    hi = paso * round(max(bt) / paso) if bt else lo
    return [lo, max(lo, hi)]


def calcula_objeto(e, fot, lit, cadenas_obj, rk_co, rt_ref_au):
    out = dict(entradas={k: v for k, v in e.items()})
    R = e["r_rsun"] * RSUN_CM
    M = e["m_msun"] * M_SUN_G
    d = e["d_pc"] * PC_CM
    teff = float(teff_de_lbol(e["log_lbol"], R))
    out["teff_lbol_R_K"] = teff
    nodos = nodos_fotosfera(teff, cadenas_obj)
    out["teff_fotosfera_K"] = nodos

    # 1. energía [D]
    out["sublimacion_R"] = {f"Ts={ts:g},QR={q:g}": float(radio_sublimacion(teff, ts, q)) for ts in T_SUB_K for q in Q_R}
    energia = {}
    for t in nodos:
        for td in T_POLVO_K:
            f, r_lp = fot.por_rk(t, td)
            energia[f"{t:g}/{td:g}"] = dict(teff=t, T_polvo=td, f_por_rK=f, rLp_por_rK=r_lp)
    out["energia"] = energia
    f_ref = [energia[f"{t:g}/{T_POLVO_REF_K:g}"]["f_por_rK"] for t in nodos]
    out["f_por_rK_1500"] = [min(f_ref), max(f_ref)]

    # 2. disco plano pasivo [D]
    rins = list(RIN_PLANO_R) + [radio_sublimacion(teff, T_POLVO_REF_K, q) for q in Q_R]
    out["disco_plano"] = {f"Rin={ri:.3g}R,cosi={ci:g}": dict(rin_R=float(ri), cosi=ci, **fot.disco_plano(teff, R, ri, ci))
                          for ri in rins for ci in COSI}

    # 3. pared [D/literatura]
    out["pared"] = dict(cobertura_literatura=lit.valor("cobertura_pared"),
                        H_r={f"r={rr:g}R,T={T:g}": escala_altura(M, R, T, rr) for rr in PARED_R for T in PARED_T_K})

    # 4. r_K del CO [M] y su f [D]
    if rk_co:
        out["rK_co"] = {k: dict(rK=v, f=[v * x for x in out["f_por_rK_1500"]]) for k, v in rk_co.items()}

    # 5. Tabla B.6 [M → D]
    if cadenas_obj:
        filas = []
        for c in cadenas_obj:
            f, _ = fot.por_rk(c["teff"], T_POLVO_REF_K)
            filas.append(dict(c, f=c["rk"] * f))
        out["tabla_B6"] = filas

    # 6. ALMA [D]
    nu = C_CGS / (lit.valor("alma_lambda") * 0.1)
    rho_cm = e["rho_arcsec"] * e["d_pc"] * AU_CM
    t_fondo = float(temperatura_equilibrio(e["l_host_lsun"], rho_cm))
    f_lim = e["alma_3sigma_mjy"] * 1e-26
    out["alma"] = dict(f3sigma_mJy=e["alma_3sigma_mjy"], T_equilibrio_K=t_fondo,
                       grueso=[radio_max_grueso(f_lim, d, nu, R, teff, t_fondo, a, c)
                               for a in ALFA_RASANTE for c in COSI])
    rm = [g["rmax_au"] for g in out["alma"]["grueso"]]
    out["alma"]["rmax_au"] = [min(rm), max(rm)]

    # 10. magnetosfera [D/E]
    xi = lit.valor("xi_magnetosfera")
    r_in = 1 / (1 - 1 / MAGNETOSPHERIC_FACTOR)
    mdots = {k: e["mdot"][k] * M_SUN_G / YR_S for k in ("estelar", "planetaria")}
    mag = dict(xi=xi, rin_R=r_in, v_ruptura_kms=float(np.sqrt(G_CGS * M / R) / 1e5),
               rM_R={f"B={B:g},{k}": float(radio_magnetosferico(B, R, M, md, xi) / R)
                     for B in CAMPOS_G for k, md in mdots.items()},
               B_rin_G={k: float(campo_para_radio(r_in * R, R, M, md, xi)) for k, md in mdots.items()},
               mdot_rin_msun_yr={f"B={B:g}": float(tasa_para_radio(r_in * R, B, R, M, xi) / M_SUN_G * YR_S)
                                 for B in CAMPOS_G})
    cor = {}
    for fuente, v in e["vsini_kms"].items():
        rco, pmax = radio_corrotacion(R, M, v * 1e5)
        cor[fuente] = dict(vsini_kms=v, P_max_d=float(pmax / 86400), rco_max_R=float(rco / R),
                           B_max_G={k: float(campo_para_radio(rco, R, M, md, xi)) for k, md in mdots.items()})
    mag["corrotacion"] = cor
    mag["v_caida_libre_rin_kms"] = float(np.sqrt(2 * G_CGS * M / R * (1 - 1 / r_in)) / 1e5)
    out["magnetosfera"] = mag

    # 11. canto [E]
    nh = lit.valor("nh_por_av")
    sig = [MASA_POR_H * M_H_G * nh * a for a in DAV_VENTANA]
    out["canto"] = dict(rt_au=rt_ref_au, sigma_ventana_gcm2=sig,
                        filas=[dict(m_disco_msun=md, h=h, **probabilidad_canto(md * M_SUN_G, RIN_CANTO_R * R,
                                                                               rt_ref_au * AU_CM, h, *sig))
                               for md in M_DISCO_MSUN for h in H_SOBRE_R])
    ps = [f["P"] for f in out["canto"]["filas"]]
    out["canto"]["P"] = [min(ps), max(ps)]

    # 8 y 9. fotometría de la primaria: K−L′ y 22 µm
    if "fotometria" in e:
        out.update(fotometria_primaria(e, fot, lit, nodos, rk_co, d))
    return out


def fotometria_primaria(e, fot, lit, nodos, rk_co, d_cm):
    ph = e["fotometria"]
    out = {}
    # 9. K−L′ [M → D]
    kw1, kw2 = ph["ks_primaria"] - ph["w1_primaria"], ph["ks_primaria"] - ph["w2_primaria"]
    lw1, lw2, llp = (lit.valor(k) for k in ("lambda_W1", "lambda_W2", "lambda_Lp"))
    kml_a = kw1 + (llp - lw1) / (lw2 - lw1) * (kw2 - kw1)
    base = ph["contraste_K"] - ph["contraste_Lp"]
    kml = {"primaria fotosferica": base + ph["kml_primaria_supuesta"], "primaria con exceso WISE": base + kml_a}
    E = a_lambda_sobre_av_ir(LAMBDA_K_UM) - a_lambda_sobre_av_ir(float(np.mean(BANDAS_UM["Lp"])))
    modelos = [fot.color(fot.fnu(t, g), "K", "Lp")
               for t in np.arange(nodos[0] - 100, nodos[1] + 101, 100) for g in LOGGS_BT]
    margen = {k: float(v - E * e["av"] - min(modelos) + 2 * ph["kml_err"]) for k, v in kml.items()}
    rks = list(RK_REJILLA) + sorted((rk_co or {}).values())
    tabla = []
    for td in T_POLVO_K:
        f, r_lp = fot.por_rk(nodos[0], td)
        for rk in rks:
            tabla.append(dict(T_polvo=td, rK=rk, dKLp=float(cambio_de_color(rk, rk * r_lp)), f=rk * f))
    # la T_d más fría que no enrojece K−L′ por encima del margen, en una rejilla de 10 K
    finas = np.arange(T_POLVO_K[0], T_POLVO_K[-1] + 1, 10.0)
    dkl = {rk: np.array([cambio_de_color(rk, rk * fot.por_rk(nodos[0], td)[1]) for td in finas]) for rk in rks}
    t_min = {f"rK={rk:.3g},{k}": (float(finas[np.argmax(dkl[rk] <= m)]) if np.any(dkl[rk] <= m) else None)
             for rk in rks for k, m in margen.items()}
    out["color_KLp"] = dict(Ks_W1=kw1, Ks_W2=kw2, Ks_W1_fot=ph["ks_w1_fotosfera"], Ks_W2_fot=ph["ks_w2_fotosfera"],
                            KLp_primaria_WISE=kml_a, KLp_compañero=kml, E_KLp_por_AV=E,
                            KLp_modelo=[min(modelos), max(modelos)], margen_2sigma=margen, tabla=tabla,
                            T_polvo_min_K=t_min)
    # 8. 22 µm [M → D]
    lbol = 10 ** e["log_lbol"]
    w = {}
    for b, col in (("W3", "ks_w3_fotosfera"), ("W4", "ks_w4_fotosfera")):
        lam, cero = lit.valor(f"lambda_{b}"), lit.valor(f"cero_{b}")
        nl, mjy = nu_l_nu_lsun(ph[f"{b.lower()}_primaria"], cero, lam, d_cm)
        fot_a = ph["ks_primaria"] - ph[col]
        _, mjy_fot = nu_l_nu_lsun(fot_a, cero, lam, d_cm)
        _, mjy_fot_roja = nu_l_nu_lsun(fot_a - a_lambda_sobre_av_ir(LAMBDA_K_UM) * e["av"], cero, lam, d_cm)
        w[b] = dict(F_mJy=float(mjy), nuLnu_lsun=float(nl), sobre_Lbol=float(nl / lbol),
                    fotosfera_A_mJy=float(mjy_fot), exceso_sobre_Lbol=float(nl / lbol * (1 - mjy_fot / mjy)),
                    fotosfera_A_enrojecida_mJy=float(mjy_fot_roja),
                    exceso_sobre_Lbol_enrojecida=float(nl / lbol * (1 - mjy_fot_roja / mjy)))
    out["exceso_22um"] = w
    return out


def calcula(objetos_entradas, fot, lit, cadenas, rk_co):
    fr = (lit.valor("fraccion_hill_min"), lit.valor("fraccion_hill_max"))
    hill = marea(objetos_entradas, fr)
    res = {}
    for e in objetos_entradas:
        s = e["slug"]
        rt_ref = hill[s]["rt_max_rho_au"][1]
        r = calcula_objeto(e, fot, lit, cadenas.get(s), rk_co.get(s), rt_ref)
        r["marea"] = hill[s]
        res[s] = r
    return res


# ---------------------------------------------------------------- salida
def _rango(v, nd=2):
    lo, hi = min(v), max(v)
    return f"{lo:.{nd}f}" if f"{lo:.{nd}f}" == f"{hi:.{nd}f}" else f"{lo:.{nd}f}–{hi:.{nd}f}"


def cifras_del_paper(res):
    """(etiqueta, frase del paper, lo que da el script) de cada cifra."""
    o = list(res.values())
    filas = []
    f15 = [x for r in o for x in r["f_por_rK_1500"]]
    filas.append(("[D]", "Sect. 5.1: «must re-emit about 0.8–0.9 r_K» (1500 K)",
                  "; ".join(f"{r['entradas']['nombre']} {_rango(r['f_por_rK_1500'], 3)} (T_eff {r['teff_fotosfera_K'][0]:g}–"
                            f"{r['teff_fotosfera_K'][1]:g} K)" for r in o) + f"; todo: {_rango(f15, 3)}"))
    filas.append(("[D]", "Sect. 5.1: «flat passive disk … sublimation radius gives r_K ≲ 0.1»",
                  "; ".join(f"{r['entradas']['nombre']} R_in = 1.5 R: r_K = {r['disco_plano']['Rin=1.5R,cosi=1']['r_K']:.3f} "
                            f"(cos i 0.5: {r['disco_plano']['Rin=1.5R,cosi=0.5']['r_K']:.3f}); R_s(1500 K, Q_R 1–4) = "
                            f"{r['sublimacion_R']['Ts=1500,QR=1']:.2f}–{r['sublimacion_R']['Ts=1500,QR=4']:.2f} R → r_K = "
                            + _rango([v["r_K"] for k, v in r["disco_plano"].items()
                                      if v["cosi"] == 1.0 and k not in {f"Rin={x:.3g}R,cosi=1" for x in RIN_PLANO_R}], 3)
                            for r in o)))
    hr = [x for r in o for x in r["pared"]["H_r"].values()]
    filas.append(("[D]", "Sect. 5.1: «a thick inner rim can re-emit about 20%» (Dullemond & Monnier 2010)",
                  f"literatura {o[0]['pared']['cobertura_literatura']:.2f}; H/r hidrostática en la pared {_rango(hr, 3)} "
                  f"→ 4–5 H/r = {4 * min(hr):.2f}–{5 * max(hr):.2f}"))
    for r in o:
        for esp, v in (r.get("rK_co") or {}).items():
            filas.append(("[M]/[D]", f"Sect. 5.1: r_K del CO de {r['entradas']['nombre']} ({esp}) y su f",
                          f"r_K = {v['rK']:.3f} → f = {_rango(v['f'], 3)}"))
    b6 = [f for r in o for f in r.get("tabla_B6", [])]
    if b6:
        filas.append(("[D]", "Sect. 5.1: Tabla B.6, r_K 0.5–1.5 «from 40% to more than the whole luminosity»",
                      f"r_K {_rango([f['rk'] for f in b6])} → f {_rango([f['f'] for f in b6], 3)}"))
    filas.append(("[D]", "Sect. 5.1: «optically thick disk below the ALMA limits … smaller than about 1 au»",
                  "; ".join(f"{r['entradas']['nombre']} {_rango(r['alma']['rmax_au'])} au" for r in o)))
    filas.append(("[D]", "Sect. 5.1: marea «12–24 au» y «4–14 au» (pericentro, R_H/3 al 16 % – 0.4 R_H al 84 %)",
                  "; ".join(f"{r['entradas']['nombre']} {r['marea']['rt_peri_16_84_au'][0]:.1f}–"
                            f"{r['marea']['rt_peri_16_84_au'][1]:.1f} au" for r in o)))
    for r in o:
        c = r.get("color_KLp")
        if not c:
            continue
        n = r["entradas"]["nombre"]
        filas.append(("[M]", f"Sect. 5.1: Ks−W1, Ks−W2 de la primaria de {n} frente a la fotosfera",
                      f"{c['Ks_W1']:.2f}, {c['Ks_W2']:.2f} frente a {c['Ks_W1_fot']:.2f}, {c['Ks_W2_fot']:.2f}"))
        filas.append(("[D]", f"Sect. 5.1: «K−L′ = 0.94–1.27 mag for the companion» ({n})",
                      _rango(list(c["KLp_compañero"].values()))))
        filas.append(("[D]", f"Sect. 5.1: «excludes dust cooler than about 900 K» ({n}; T_d mínima permitida)",
                      "; ".join(f"{k}: {v:g} K" if v else f"{k}: ninguna" for k, v in c["T_polvo_min_K"].items()
                                if not k.startswith("rK=0.03") and not k.startswith("rK=0.1,")
                                and not k.startswith("rK=0.2,"))))
        w4 = r["exceso_22um"]["W4"]
        filas.append(("[D]", f"Sect. 5.1: 22 µm «about 0.9 times the luminosity» ({n})",
                      f"νL_ν/L_bol = {w4['sobre_Lbol']:.3f} sin quitar nada; {w4['exceso_sobre_Lbol']:.3f} sin la "
                      f"fotosfera de A ({w4['fotosfera_A_mJy']:.2f} mJy); {w4['exceso_sobre_Lbol_enrojecida']:.3f} con "
                      f"ella enrojecida por A_Ks ({w4['fotosfera_A_enrojecida_mJy']:.2f} mJy)"))
    for r in o:
        m = r["magnetosfera"]
        n = r["entradas"]["nombre"]
        cota = " (Ṁ es límite)" if r["entradas"]["mdot"]["limite"] else ""
        filas.append(("[D]", f"Métodos: R_in = 5R ↔ «0.03 kG» estelar / «0.13 kG» planetaria ({n}{cota})",
                      f"{m['B_rin_G']['estelar'] / 1e3:.3f} / {m['B_rin_G']['planetaria'] / 1e3:.3f} kG"))
        filas.append(("[D]", f"Métodos: «0.3 kG … about 18R» ({n}{cota}, Ṁ estelar)",
                      f"{m['rM_R']['B=300,estelar']:.1f} R"))
        filas.append(("[D]", f"Métodos: corrotación «at most 6–15R» ({n})",
                      "; ".join(f"{k} ({v['vsini_kms']:g} km/s) {v['rco_max_R']:.1f} R" for k, v in m["corrotacion"].items())))
        filas.append(("[E]", f"Sect. 5.1: «dipole field above about 0.1–0.2 kG … outside the corotation radius» ({n}{cota})",
                      "; ".join(f"{k} {v['B_max_G']['estelar'] / 1e3:.3f} kG" for k, v in m["corrotacion"].items())))
        filas.append(("[E]", f"Sect. 5.1: canto «prior probability of only 1–6%» ({n})",
                      f"P = {_rango(r['canto']['P'], 3)}"))
    return filas


def imprime(res, cadenas_dirs):
    for s, r in res.items():
        e = r["entradas"]
        print(f"% ==== {e['nombre']} ({e['run']}) ====")
        print(f"%   entradas: d = {e['d_pc']:g} pc, ρ = {e['rho_arcsec']:.4f}\" (B3), M = {e['m_msun']:.6g} M☉, "
              f"R = {e['r_rsun']:.6g} R☉, log L_bol = {e['log_lbol']:g}, A_V = {e['av']:g}; Ṁ estelar = "
              f"{e['mdot']['estelar']:.4e}, planetaria = {e['mdot']['planetaria']:.4e} M☉/yr "
              f"(factor {e['mdot']['factor_aoyama']:.2f}; {e['mdot']['procedencia']})")
        print(f"%   T_eff(L_bol, R) = {r['teff_lbol_R_K']:.1f} K; fotosfera BT-Settl en {r['teff_fotosfera_K']} K")
    if not cadenas_dirs:
        print("% aviso: sin --teff-chains no hay Tabla B.6 (punto 5) y la fotosfera usa solo la T_eff de L_bol y R")
    print("% ==== cifras del paper ====")
    for etq, frase, valor in cifras_del_paper(res):
        print(f"{etq:<8} {frase}\n           → {valor}")


def _json_default(x):
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, float) and not math.isfinite(x):
        return str(x)
    raise TypeError(type(x))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--project-root", default=None, help="raíz con runs/ (desde un worktree, la del clon que los tiene)")
    ap.add_argument("--teff-chains", action="append", default=[], metavar="DIR",
                    help="directorio de `teff_bayes.py ajusta` (repetible): las cadenas de la Tabla B.6")
    ap.add_argument("--literatura", type=Path, default=LITERATURA)
    ap.add_argument("--btsettl-raw", type=Path, default=BTSETTL_RAW)
    ap.add_argument("--r-btsettl", type=float, default=R_BT, help="λ/Δλ del promedio de BT-Settl (200 = el del agente)")
    ap.add_argument("--salida", type=Path, default=None, help="escribe disco_cotas.json y la caché de BT-Settl")
    a = ap.parse_args(argv)

    from scripts import paper_figures as pf
    proyecto = Path(a.project_root).resolve() if a.project_root else ROOT
    pf.PROYECTO = proyecto
    objetos = pf.objetos_del_paper()
    lit = Literatura(a.literatura)
    entradas = [entradas_del_objeto(o, lit) for o in objetos]
    cadenas = carga_cadenas([Path(x).expanduser() for x in a.teff_chains]) if a.teff_chains else {}
    rk_co = {e["slug"]: rk_del_co(e["slug"], proyecto) for e in entradas}
    cache = a.salida / f"btsettl_ir_R{a.r_btsettl:g}.npz" if a.salida else None
    fot = Fotosfera(*carga_btsettl(a.btsettl_raw, a.r_btsettl, cache))
    res = calcula(entradas, fot, lit, cadenas, rk_co)
    imprime(res, a.teff_chains)
    if a.salida:
        a.salida.mkdir(parents=True, exist_ok=True)
        doc = dict(objetos=res, cifras_del_paper=cifras_del_paper(res), teff_chains=a.teff_chains,
                   literatura=str(a.literatura), btsettl_raw=str(a.btsettl_raw), r_btsettl=a.r_btsettl,
                   project_root=str(proyecto), semilla=SEMILLA, n_mc=N_MC)
        (a.salida / "disco_cotas.json").write_text(json.dumps(doc, indent=1, ensure_ascii=False, default=_json_default))
        print(f"% escrito {a.salida / 'disco_cotas.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
