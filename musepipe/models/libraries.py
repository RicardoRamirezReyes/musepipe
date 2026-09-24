"""Declaración de cada biblioteca espectral de G3: una prueba por biblioteca.

Plan ``docs/2026-09-23_plan_g3_bibliotecas_y_resolucion.md`` y decisión
``docs/2026-09-23_decision_g3_resolucion_y_bibliotecas.md`` (Q1–Q4 aprobadas
por el autor el 2026-09-23).

Cada biblioteca se ajusta como una **prueba independiente**, con su bloque en
el QC; ninguna cifra combina bibliotecas. Para eso cada una tiene que decir de
sí misma lo que antes se suponía:

* **marco** de longitud de onda (``air`` / ``vacuum``): MUSE está en aire;
* **resolución** (poder resolutivo R, FWHM(λ) = λ/R), o que es **nítida**
  (sintética muestreada muy por debajo de la LSF). Ninguna biblioteca se trata
  como infinitamente fina si no lo declara;
* **cita**, **instrumento**, **clase de gravedad** y **edad**.

Orden de resolución de cada campo, el mismo que usa el repo para la unidad de
flujo y el marco (no hay valor silencioso por defecto):

1. la entrada de la biblioteca en el config (``g3_template_libraries``);
2. el ``PROVENANCE.json`` de la propia biblioteca (lo que la biblioteca dice
   de sí misma; las etiquetas de config se quedan viejas: ``9dbb3d4``);
3. la tabla documentada :data:`DOCUMENTED_DEFAULTS` de este módulo, con su
   fuente bibliográfica — solo para las tres bibliotecas descargadas el
   2026-07-16/17, cuyos ``PROVENANCE.json`` no declaran R ni marco;
4. si falta aun así: ``RuntimeError``.

Una biblioteca nueva (p. ej. ``templates_young_lateM_xshooter``,
``templates_xshyne_L_xshooter``) se añade SOLO por config: una entrada más en
``g3_template_libraries`` con ``name``, ``subdir``, ``gravity_class`` y
``age``; el marco y la R los lee de su ``PROVENANCE.json``.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .airvac import AIR, normalize_frame
from .manifest import PROVENANCE_NAME

#: Marco del DATO. Los cubos del DRS de MUSE declaran ``CTYPE3 = 'AWAV'``
#: (longitud de onda en aire, FITS WCS Paper III: Greisen et al. 2006).
#: Knob: ``g3_data_wave_frame``.
DATA_FRAME_DEFAULT = AIR

#: Claves de ``PROVENANCE.json`` que se aceptan para cada campo (las
#: bibliotecas nuevas las escribe otro proceso; se leen todas las variantes
#: razonables y se registra cuál se usó).
_PROV_KEYS = {
    "wave_frame": ("wave_frame", "wavelength_frame", "wave_medium", "frame", "medium"),
    "resolution_R": ("resolution_R", "resolving_power", "R", "spectral_resolution_R",
                     "resolution"),
    "resolution_fwhm_A": ("resolution_fwhm_A", "fwhm_A"),
    "resolution_R_by_object": ("resolution_R_by_object", "R_by_object"),
    "resolution_sharp": ("resolution_sharp", "sharp"),
    "citation": ("citation", "citations"),
    "instrument": ("instrument", "arm", "spectrograph"),
    "gravity_class": ("gravity_class",),
    "age": ("age", "age_myr", "age_note"),
}

#: Resolución de las plantillas de Manara+2013 por objeto, de su Tabla 2 (rendija
#: VIS) y §2.2: R = 17400 / 8800 / 5400 para 0.4″ / 0.9″ / 1.5″.
_MANARA13_R_BY_OBJECT = {
    # 0.4″ → R = 17400
    "TWA9A": 17400.0, "TWA6": 17400.0, "TWA25": 17400.0, "TWA14": 17400.0,
    "TWA13B": 17400.0, "TWA13A": 17400.0, "TWA2A": 17400.0, "TWA9B": 17400.0,
    "TWA15B": 17400.0, "TWA7": 17400.0, "TWA15A": 17400.0,
    # 0.9″ → R = 8800
    "SO879": 8800.0, "Sz122": 8800.0, "Sz121": 8800.0, "Sz94": 8800.0,
    "SO797": 8800.0, "SO641": 8800.0, "SO925": 8800.0, "SO999": 8800.0,
    "Sz107": 8800.0, "TWA26": 8800.0,
    # 1.5″ → R = 5400
    "TWA29": 5400.0,
}

#: Declaraciones documentadas de las bibliotecas cuyos PROVENANCE.json no las
#: traen. Cada valor lleva su fuente en ``*_source``.
DOCUMENTED_DEFAULTS = {
    "templates_young": {
        "kind": "empirical",
        "gravity_class": "young",
        "age": "Class III PMS (σ Ori, Lupus, TWA, …; ~1–10 Myr)",
        "instrument": "VLT/X-shooter VIS",
        "wave_frame": "air",
        "wave_frame_source": ("pipeline de X-shooter (Modigliani et al. 2010): "
                              "calibración en longitudes de onda de aire; declarado "
                              "por el autor 2026-09-23"),
        "resolution_R": 8800.0,
        "resolution_R_by_object": dict(_MANARA13_R_BY_OBJECT),
        "resolution_source": (
            "Manara et al. 2013 (A&A 551, A107) §2.2 y Tabla 2: VIS R = 17400/8800/5400 "
            "para rendija 0.4″/0.9″/1.5″, asignada por objeto. Los objetos de Manara "
            "et al. 2017 (A&A 605, A86) llevan R = 8800 (0.9″) SIN verificar su rendija"),
    },
    "templates_field": {
        "kind": "empirical",
        "gravity_class": "field",
        "age": "field (Gyr)",
        "instrument": "SDSS/BOSS spectrographs (compuestos de Kesseli et al. 2017)",
        "wave_frame": "vacuum",
        "wave_frame_source": "convención SDSS: longitudes de onda en vacío",
        "resolution_R": 2000.0,
        "resolution_source": (
            "Kesseli et al. 2017 (ApJS 230, 16; arXiv:1702.06957), resumen: «a "
            "resolution of better than R ∼ 2000»; espectrógrafos BOSS R = 1560–2650 "
            "(Smee et al. 2013, AJ 146, 32)"),
    },
    "bt-settl-cifist": {
        "kind": "atmosphere",
        "gravity_class": None,
        "age": None,
        "instrument": "synthetic (PHOENIX BT-Settl CIFIST2011, SVO)",
        "wave_frame": "vacuum",
        "wave_frame_source": "metadatos de los nodos: 'vacuum (synthetic)'",
        "resolution_sharp": True,
        "resolution_source": ("sintética: paso 0.02 Å (R_muestreo ≈ 3.5e5 a 7000 Å), "
                              "≪ LSF de MUSE ≈ 2.3 Å; se declara nítida"),
    },
}

#: Las dos bibliotecas empíricas del D1/D2 congelado, en ausencia de
#: ``g3_template_libraries`` en el config.
DEFAULT_TEMPLATE_ENTRIES = (
    {"name": "templates_young", "subdir": "templates_young", "gravity_class": "young"},
    {"name": "templates_field", "subdir": "templates_field", "gravity_class": "field"},
)

#: Par de gravedad por defecto para el Δχ² que consume G4 (Q3).
DEFAULT_GRAVITY_PAIR = {"young": "templates_young", "field": "templates_field"}


@dataclass
class LibraryDeclaration:
    """Lo que una biblioteca declara de sí misma, con la fuente de cada campo."""

    name: str
    subdir: str
    kind: str = "empirical"
    gravity_class: str | None = None
    age: str | None = None
    citation: str | None = None
    instrument: str | None = None
    wave_frame: str | None = None
    resolution_R: float | None = None
    resolution_fwhm_A: float | None = None
    resolution_sharp: bool = False
    resolution_R_by_object: dict = field(default_factory=dict)
    #: ``False`` cuando la biblioteca NO está corregida de telúricas: sus
    #: bandas se enmascaran en el dato para ESTA prueba (``telluric_mask_bands``).
    telluric_corrected: bool | None = None
    telluric_mask_bands: list = field(default_factory=list)
    velocity_frame: str | None = None
    sources: dict = field(default_factory=dict)
    notes: dict = field(default_factory=dict)
    provenance: dict | None = None

    # -- resolución -------------------------------------------------------- #
    def resolution_for(self, spectrum_meta=None):
        """``(template_R, template_fwhm_A)`` para un espectro concreto.

        Prioridad: la R del propio espectro (``meta['resolution_R']``) → la de
        su objeto (``resolution_R_by_object``) → la de la biblioteca. Nítida →
        ``R = inf``.
        """
        meta = spectrum_meta or {}
        for key in ("resolution_R", "R"):
            if meta.get(key) not in (None, ""):
                return float(meta[key]), None
        obj = meta.get("object")
        if obj is not None and str(obj) in self.resolution_R_by_object:
            return float(self.resolution_R_by_object[str(obj)]), None
        if self.resolution_sharp:
            return math.inf, None
        if self.resolution_R is not None:
            return float(self.resolution_R), None
        if self.resolution_fwhm_A is not None:
            return None, float(self.resolution_fwhm_A)
        raise RuntimeError(f"{self.name}: resolución no declarada")

    def min_R(self):
        """La R más gruesa que declara (para decidir si es más gruesa que MUSE)."""
        vals = [float(v) for v in self.resolution_R_by_object.values()]
        if self.resolution_sharp:
            vals.append(math.inf)
        if self.resolution_R is not None:
            vals.append(float(self.resolution_R))
        return min(vals) if vals else None

    def fwhm_at(self, wave_A, spectrum_meta=None):
        """FWHM [Å] declarada a ``wave_A`` (0 si nítida)."""
        import numpy as np
        r, fw = self.resolution_for(spectrum_meta)
        w = np.asarray(wave_A, dtype=float)
        if r is not None:
            return np.zeros_like(w) if math.isinf(r) else w / r
        return np.full_like(w, fw)

    def to_qc(self, *, n_by_spt=None):
        out = {k: v for k, v in asdict(self).items() if k != "provenance"}
        if out.get("resolution_R_by_object"):
            out["resolution_R_by_object_n"] = len(out["resolution_R_by_object"])
        if n_by_spt is not None:
            out["n_spectra_by_spt"] = dict(n_by_spt)
            out["n_spectra"] = int(sum(n_by_spt.values()))
        if self.provenance is not None:
            out["provenance_json"] = {k: self.provenance.get(k) for k in sorted(self.provenance)
                                      if not isinstance(self.provenance.get(k), (list, dict))
                                      or k in ("catalogs", "R_range", "eso_programmes")}
        return out


def _first(d, keys):
    for k in keys:
        if isinstance(d, dict) and d.get(k) not in (None, ""):
            return k, d[k]
    return None, None


def _parse_R(value):
    """R declarada como número o texto ('R~5400', '5400', 'R = 8800')."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, dict):
        return None
    m = re.search(r"(\d[\d_,]*\.?\d*)", str(value))
    return float(m.group(1).replace(",", "").replace("_", "")) if m else None


def read_provenance(family_dir):
    path = Path(family_dir) / PROVENANCE_NAME
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_declaration(entry, family_dir):
    """Declaración de una biblioteca: config → PROVENANCE.json → defaults documentados.

    ``entry`` es un dict con al menos ``name`` y ``subdir``. Falta marco o
    resolución → ``RuntimeError`` (no se supone nada).
    """
    entry = dict(entry)
    name = str(entry.get("name") or entry.get("subdir"))
    subdir = str(entry.get("subdir") or name)
    prov = read_provenance(family_dir)
    default = DOCUMENTED_DEFAULTS.get(subdir, {})
    decl = LibraryDeclaration(name=name, subdir=subdir, provenance=prov)

    def pick(fld, *, parse=None):
        if entry.get(fld) not in (None, ""):
            decl.sources[fld] = "config"
            return parse(entry[fld]) if parse else entry[fld]
        key, val = _first(prov or {}, _PROV_KEYS.get(fld, (fld,)))
        if key is not None:
            parsed = parse(val) if parse else val
            if parsed not in (None, ""):
                decl.sources[fld] = f"{PROVENANCE_NAME}:{key}"
                return parsed
        if fld == "resolution_R_by_object" and isinstance((prov or {}).get("objects"), list):
            by = {str(o.get("object")): float(o["R"]) for o in prov["objects"]
                  if o.get("object") is not None and o.get("R") not in (None, "")}
            if by:
                decl.sources[fld] = f"{PROVENANCE_NAME}:objects[].R"
                return by
        if default.get(fld) not in (None, ""):
            decl.sources[fld] = "DOCUMENTED_DEFAULTS (musepipe.models.libraries)"
            return default[fld]
        return None

    decl.kind = str(entry.get("kind") or default.get("kind") or "empirical")
    decl.citation = pick("citation")
    if isinstance(decl.citation, list):
        decl.citation = "; ".join(str(c) for c in decl.citation)
    decl.instrument = pick("instrument")
    decl.gravity_class = pick("gravity_class")
    decl.age = pick("age")
    if decl.age is not None and not isinstance(decl.age, str):
        decl.age = json.dumps(decl.age)

    frame_raw = pick("wave_frame")
    decl.wave_frame = normalize_frame(frame_raw)
    if decl.wave_frame is None:
        raise RuntimeError(
            f"biblioteca {name}: marco de longitud de onda no declarado ({frame_raw!r}); "
            "declárese en su PROVENANCE.json o en la entrada del config (wave_frame)")
    decl.notes["wave_frame_declared"] = frame_raw
    if default.get("wave_frame_source") and decl.sources.get("wave_frame", "").startswith("DOC"):
        decl.notes["wave_frame_source"] = default["wave_frame_source"]

    sharp = pick("resolution_sharp")
    decl.resolution_sharp = bool(sharp) if sharp is not None else False
    decl.resolution_R = pick("resolution_R", parse=_parse_R)
    fw = pick("resolution_fwhm_A")
    decl.resolution_fwhm_A = float(fw) if fw not in (None, "") else None
    by_obj = pick("resolution_R_by_object")
    decl.resolution_R_by_object = ({str(k): float(v) for k, v in by_obj.items()}
                                   if isinstance(by_obj, dict) else {})
    if decl.resolution_R is None and isinstance((prov or {}).get("R_range"), list):
        # R por espectro + rango: la R de biblioteca es la más gruesa del rango
        decl.resolution_R = float(min(prov["R_range"]))
        decl.sources["resolution_R"] = f"{PROVENANCE_NAME}:R_range (min)"
    if not (decl.resolution_sharp or decl.resolution_R or decl.resolution_fwhm_A
            or decl.resolution_R_by_object):
        raise RuntimeError(
            f"biblioteca {name}: resolución no declarada (ni R, ni FWHM, ni 'sharp'); "
            "ninguna biblioteca se trata como infinitamente fina sin declararlo")
    if default.get("resolution_source") and any(
            decl.sources.get(k, "").startswith("DOC") for k in
            ("resolution_R", "resolution_sharp", "resolution_R_by_object")):
        decl.notes["resolution_source"] = default["resolution_source"]
    if decl.citation is None:
        raise RuntimeError(f"biblioteca {name}: sin cita (spec G3 §1.2)")

    # Telúricas: una biblioteca que declara ``telluric_corrected: false`` lleva
    # las bandas sin corregir; se enmascaran EN EL DATO para su prueba (las de
    # severidad en ``telluric_mask_severities``, por defecto moderada y fuerte).
    tc = entry.get("telluric_corrected", (prov or {}).get("telluric_corrected"))
    decl.telluric_corrected = None if tc is None else bool(tc)
    if "telluric_mask_bands_A" in entry:
        decl.telluric_mask_bands = [{"lo_A": float(lo), "hi_A": float(hi), "name": "config"}
                                    for lo, hi in entry["telluric_mask_bands_A"]]
        decl.sources["telluric_mask_bands"] = "config"
    elif decl.telluric_corrected is False:
        sev = set(entry.get("telluric_mask_severities", ("moderate", "strong")))
        bands = [b for b in (prov or {}).get("telluric_bands", [])
                 if b.get("severity", "strong") in sev]
        if not bands:
            raise RuntimeError(f"biblioteca {name}: telluric_corrected=false sin bandas "
                               "declaradas (telluric_bands) que enmascarar")
        decl.telluric_mask_bands = [{"lo_A": float(b["lo_A"]), "hi_A": float(b["hi_A"]),
                                     "name": b.get("name"), "severity": b.get("severity")}
                                    for b in bands]
        decl.sources["telluric_mask_bands"] = (f"{PROVENANCE_NAME}:telluric_bands "
                                               f"(severity in {sorted(sev)})")
    frame_text = str(frame_raw or "")
    decl.velocity_frame = ("topocentric" if "topocentric" in frame_text.lower()
                           else entry.get("velocity_frame"))
    return decl


def template_library_entries(cfg):
    """Lista de entradas de bibliotecas de plantillas declaradas en el config.

    ``g3_template_libraries``: lista de dicts ``{name, subdir, gravity_class,
    age, [wave_frame, resolution_R, citation, …]}``. Sin la clave, las dos
    del D1/D2 (Manara joven, Kesseli campo).
    """
    entries = cfg.get("g3_template_libraries")
    if not entries:
        return [dict(e) for e in DEFAULT_TEMPLATE_ENTRIES]
    out = []
    seen = set()
    for e in entries:
        e = dict(e)
        if "name" not in e and "subdir" not in e:
            raise RuntimeError(f"entrada de biblioteca sin name/subdir: {e}")
        e.setdefault("name", e.get("subdir"))
        e.setdefault("subdir", e["name"])
        if e["name"] in seen:
            raise RuntimeError(f"biblioteca duplicada en g3_template_libraries: {e['name']}")
        seen.add(e["name"])
        out.append(e)
    return out


def gravity_pair(cfg):
    """``{'young': name, 'field': name}`` del Δχ² de gravedad que lee G4 (Q3)."""
    pair = dict(DEFAULT_GRAVITY_PAIR)
    pair.update(dict(cfg.get("g3_gravity_pair") or {}))
    return pair


def data_frame(cfg):
    frame = normalize_frame(cfg.get("g3_data_wave_frame", DATA_FRAME_DEFAULT))
    if frame is None:
        raise RuntimeError(f"g3_data_wave_frame inválido: {cfg.get('g3_data_wave_frame')!r}")
    return frame


__all__ = [
    "DATA_FRAME_DEFAULT", "DEFAULT_GRAVITY_PAIR", "DEFAULT_TEMPLATE_ENTRIES",
    "DOCUMENTED_DEFAULTS", "LibraryDeclaration", "data_frame", "gravity_pair",
    "read_provenance", "resolve_declaration", "template_library_entries",
]
