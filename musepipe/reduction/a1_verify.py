"""Envoltorio de A1: ensambla `stage00r_qc.json` y corre la batería V1–V6.

El QC de A1 tiene dos piezas y hasta hoy solo se escribía una sola por vía. La
reducción en cascada deja en disco la evidencia de cada paso —la auditoría de
calibraciones por noche, los productos por exposición, el QC del combine por
vóxel— pero **nadie reunía eso en el documento envoltorio** (`00r_raw_reduction`)
que declara la reducción entera con sus fases y sus verificaciones.
`esorex_driver.stage00r_qc_skeleton` escribe la plantilla con todo a `false`/
`null`, y ahí se quedaba.

Consecuencia medida: `runs/ROXs42Bb_raw/stages/stage00r_qc.json` era la plantilla
vacía, y como A2 exige de ese fichero `gates_passed` y V1–V6, **A2 no podía
correr sobre ROXs 42B b** — y sin A2 no hay máscara de cielo, y sin máscara A4 no
puede medir M4/M5. Todo el bloque C de ese objeto arrastra `STAT` con factor 1.0
por esa cadena.

Este módulo cierra el hueco leyendo únicamente lo que la reducción ya dejó
escrito. Dos principios:

* **Las fases se conceden por evidencia, no por aserción.** Una `fase*` entra en
  `gates_passed` solo si su producto está en disco, y el QC publica *qué*
  fichero la sostiene. Escribir una fase porque «seguro que corrió» es
  exactamente lo que hace que un QC no valga como registro.
* **Lo que no se puede medir se declara `unavailable`, no `false`.** `false` es
  un veredicto y significa que la prueba falló; la plantilla del driver usaba
  `false` para «nadie la corrió», que es lo que hacía imposible distinguir un
  esbozo de un fallo (`a1_review.is_qc_skeleton` existe justo por eso).

La batería reusa `musepipe/reduction/verify.py` tal cual, que define las seis y
hasta ahora no tenía ningún llamador en el repo.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from astropy.io import fits

from . import verify
from .a1_review import _root, is_qc_skeleton, resolve_a1_run

STAGE_NAME = "00r_raw_reduction"
COMBINE_STAGE = "stream_combine"

#: Radio, en spaxels, de la región de fuente que V6 exige libre de cielo. La
#: spec A1 §V6 pide «no solapa con las posiciones de la primaria ni del
#: compañero» sin fijar radio; se declara aquí y viaja al QC. Medido en
#: ROXs 42B b: el veredicto es 30/30 limpias de 5 a 12 px, así que no es el
#: radio quien decide.
V6_SOURCE_RADIUS_PX = 8.0

#: Radio de apertura de V5, el mismo con el que se midió ROXs 12 b.
V5_APERTURE_RADIUS_PX = 80.0

#: Ventana de comparación de V4, la que declara el QC de ROXs 12 b
#: (`v4_method: 'PSF-matched comparison within +/-15 px'`).
V4_HALF_WINDOW_PX = 15


class A1VerifyError(RuntimeError):
    """La evidencia en disco no permite escribir el envoltorio."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _result(res: verify.VerificationResult, **extra) -> dict:
    """Verificación en la forma `{ok, value, message}` del esquema monolítico.

    Se escribe un **booleano explícito** en `ok`. La forma escalar
    (`v4_adp_whitelight_corr: 0.9541`) es la que hace que la puerta de A2, que
    evalúa `bool(valor)`, dé por buena cualquier verificación con un número
    distinto de cero — un residuo de 0.00122 es «verdadero» en Python.
    """

    return {"ok": bool(res.passed), "value": res.value, "message": res.message, **extra}


# --------------------------------------------------------------------------
# Localización de la evidencia
# --------------------------------------------------------------------------


def combine_qc(run_id: str, root: Path) -> tuple[dict, Path]:
    """QC del combine por vóxel (`cube_telcorr_qc.json`, `stream_combine_v1`)."""

    path = root / "runs" / run_id / "cube_telcorr_qc.json"
    payload = _load_json(path)
    if payload.get("stage") != COMBINE_STAGE:
        raise A1VerifyError(
            f"{path} no es el QC de un combine en cascada (stage={payload.get('stage')!r}). "
            "Este módulo solo sabe reconstruir el envoltorio de una reducción en cascada."
        )
    return payload, path


def contributing_nights(payload: dict) -> dict[str, int]:
    """Noches que aportan exposiciones, con su recuento."""

    out: dict[str, int] = {}
    for exposure in payload.get("exposures") or ():
        night = str(exposure.get("exposure_id", "")).split("_")[0]
        if night:
            out[night] = out.get(night, 0) + 1
    return dict(sorted(out.items()))


def adp_references(nights, root: Path, explicit=None) -> dict[str, Path]:
    """ADP de ESO por noche contribuyente.

    Un combine multinoche **no tiene un ADP único**: cada noche tiene el suyo y
    el cubo es la suma de las dos. V4 y V5 se miden contra todos y publican
    todos; el veredicto lo fija `_v4` (§V4).
    """

    if explicit:
        out = {}
        for item in explicit:
            path = Path(item).expanduser()
            if not path.exists():
                raise A1VerifyError(f"ADP declarado que no existe: {path}")
            out[path.parent.name] = path
        return out

    data_root = root.parent / "Data"
    out: dict[str, Path] = {}
    for night in nights:
        stamp = night.replace("-", "")
        hits = sorted(glob.glob(str(data_root / "*" / stamp / "ADP*.fits")))
        if hits:
            out[night] = Path(hits[0])
    if not out:
        raise A1VerifyError(
            "No se encontró ningún ADP de referencia para las noches "
            f"{list(nights)} bajo {data_root}. Pásalos con --adp."
        )
    return out


def _cube_to_mask_frame(cube_header, mask_header, yx):
    """Pasa una posición del cubo por exposición al frame de su `SKY_MASK`.

    No es el mismo sistema y **no coinciden ni en forma ni en origen**: el cubo
    es celeste (`CTYPE='RA---TAN'`, 429×433 aquí) y la máscara es un frame de
    *offsets en píxeles* respecto al centro del campo (`CTYPE='PIXEL'`,
    `CRVAL=(-159.07, -151.35)`, 304×315). Aplicar los centroides del cubo
    directamente sobre la máscara no da error —caen dentro del array— pero mide
    otra cosa: comprobado, invierte el veredicto de V6.
    """

    y, x = yx
    dy = y + 1 - float(cube_header["CRPIX2"])
    dx = x + 1 - float(cube_header["CRPIX1"])
    j = (dy - float(mask_header["CRVAL2"])) / float(mask_header["CD2_2"]) + float(mask_header["CRPIX2"]) - 1
    i = (dx - float(mask_header["CRVAL1"])) / float(mask_header["CD1_1"]) + float(mask_header["CRPIX1"]) - 1
    return j, i


def companion_offset_yx(run_id: str, root: Path) -> tuple[float, float]:
    """Offset compañero−primaria, en píxeles, del QC de B3.

    Se toma como diferencia de posiciones y no de `sep`/`PA` a propósito: el
    combine alinea por traslación pura, así que el mismo `(dy, dx)` vale en el
    frame de cada exposición, y así no hay que rehacer la trigonometría del
    ángulo norte (que difiere entre objetos: −53° y 180°).
    """

    payload = _load_json(root / "runs" / run_id / "stages" / "stage01c_qc.json")
    primary = (payload.get("primary") or {}).get("pos_yx")
    companion = (payload.get("companion") or {}).get("pos_yx")
    if not primary or not companion:
        raise A1VerifyError("stage01c_qc.json (B3) no declara las dos posiciones; V6 no puede situar el compañero.")
    return float(companion[0]) - float(primary[0]), float(companion[1]) - float(primary[1])


# --------------------------------------------------------------------------
# Las seis verificaciones
# --------------------------------------------------------------------------


def _v3(cube: Path, adps: dict[str, Path], combine: dict) -> dict:
    """V3 con la tolerancia que declara el propio combine, no una fija.

    El combine por vóxel apila **sin remuestrear en λ** y publica la dispersión
    resultante (`wavelength.crval3_spread_channels`). Exigir que `CRVAL3`
    coincida con el del ADP dentro de 1e-6 Å le pide al combinado una precisión
    que él mismo declara no tener: sus exposiciones difieren entre sí más que
    eso. La tolerancia es esa dispersión, en Å, y viaja al QC junto al delta
    medido. El paso espectral se sigue exigiendo idéntico.
    """

    wavelength = combine.get("wavelength") or {}
    spread = float(wavelength.get("crval3_spread_channels") or 0.0)
    step = float(wavelength.get("cd3_3") or 1.25)
    tolerance = max(spread * step, 1e-6)
    best = None
    deltas = {}
    with fits.open(cube, memmap=True) as hdul:
        crval3 = float(hdul["DATA"].header["CRVAL3"])
    for night, adp in adps.items():
        with fits.open(adp, memmap=True) as hdul:
            hdu = hdul["DATA"] if "DATA" in hdul else hdul[1]
            deltas[night] = abs(crval3 - float(hdu.header["CRVAL3"]))
        res = verify.verify_wcs_headers(cube, adp, crval3_tolerance=tolerance)
        if best is None or (res.passed and not best[1].passed):
            best = (night, res)
    night, res = best
    return _result(
        res,
        v3_crval3_tolerance_A=tolerance,
        v3_crval3_delta_A={k: round(v, 8) for k, v in deltas.items()},
        v3_reference_night=night,
        v3_note=("tolerancia = dispersión de CRVAL3 que declara el combine "
                 f"({spread} canales x {step} A); el combinado no remuestrea en lambda"),
    )


def _v4(cube: Path, adps: dict[str, Path]) -> dict:
    """V4 contra el ADP de **cada** noche contribuyente; pasa con la mejor.

    Un combine de dos noches no tiene un ADP único de referencia, así que se
    mide contra todos y se publican todos. Basta con que el cubo reproduzca la
    escena de una de las noches para que la comparación con el archivo sostenga
    la reducción; las demás quedan escritas, no escondidas.
    """

    by_night = {}
    best = None
    for night, adp in sorted(adps.items()):
        res = verify.verify_whitelight_vs_adp_psf_matched(cube, adp, half_window=V4_HALF_WINDOW_PX)
        # `passes_threshold`, NO `ok`: esto es una MEDIDA por noche, y el veredicto
        # es el de arriba ("pasa con la mejor"). F1 recorre los estados anidados y
        # trata cualquier `ok: false` como rojo bloqueante, asi que publicar aqui
        # un `ok` convertia una cifra informativa en una compuerta.
        by_night[night] = {"corr": res.value, "passes_threshold": bool(res.passed),
                           "message": res.message}
        if best is None or (res.value or 0) > (best[1].value or 0):
            best = (night, res)
    night, res = best
    return _result(
        res,
        v4_by_night=by_night,
        v4_reference_night=night,
        v4_method=f"PSF-matched comparison within +/-{V4_HALF_WINDOW_PX} px",
        v4_rule="pasa si supera el umbral contra al menos una noche contribuyente",
    )


def _v5(cube: Path, adps: dict[str, Path], primary_yx) -> dict:
    """V5 contra cada ADP, con la primaria localizada por pico en cada uno."""

    by_night = {}
    best = None
    for night, adp in sorted(adps.items()):
        adp_cube, _ = verify.read_cube_data(adp)
        whitelight = verify.whitelight_image(adp_cube)
        ay, ax = np.unravel_index(np.nanargmax(whitelight), whitelight.shape)
        del adp_cube, whitelight
        res = verify.verify_star_spectrum_vs_adp(
            cube, adp, star_yx=primary_yx, adp_star_yx=(float(ay), float(ax)),
            radius=V5_APERTURE_RADIUS_PX,
        )
        by_night[night] = {"ratio_rms": res.value, "passes_threshold": bool(res.passed),
                           "message": res.message, "adp_star_yx": [float(ay), float(ax)]}
        if best is None or (res.passed and not best[1].passed):
            best = (night, res)
    night, res = best
    return _result(
        res,
        v5_by_night=by_night,
        v5_reference_night=night,
        v5_aperture_radius_px=V5_APERTURE_RADIUS_PX,
    )


def _v6(combine: dict, offset_yx, radius_px: float) -> dict:
    """V6 exposición a exposición, con las posiciones llevadas al frame de la máscara."""

    dy, dx = offset_yx
    total = 0
    clean = 0
    dirty = []
    for exposure in combine.get("exposures") or ():
        cube_path = Path(exposure["file"])
        mask_path = cube_path.parent / "SKY_MASK_0001.fits"
        if not mask_path.exists():
            raise verify.VerificationError(f"Falta la máscara de cielo de {exposure['exposure_id']}: {mask_path}")
        total += 1
        with fits.open(cube_path, memmap=True) as hdul:
            cube_header = (hdul["DATA"] if "DATA" in hdul else hdul[1]).header
        with fits.open(mask_path, memmap=True) as hdul:
            mask_header = hdul[0].header
            mask = np.asarray(hdul[0].data, dtype=bool)
        y0, x0 = float(exposure["y_center"]), float(exposure["x_center"])
        positions = [
            _cube_to_mask_frame(cube_header, mask_header, (y0, x0)),
            _cube_to_mask_frame(cube_header, mask_header, (y0 + dy, x0 + dx)),
        ]
        res = verify.verify_sky_mask_clean(mask, positions, radius_px=radius_px, true_means_sky=True)
        if res.passed:
            clean += 1
        else:
            dirty.append({"exposure_id": exposure["exposure_id"], "message": res.message})
    passed = total > 0 and clean == total
    return {
        "ok": passed,
        "value": passed,
        "message": f"sky_mask_clean_exposures={clean}/{total}",
        "v6_exposures_clean": clean,
        "v6_exposures_total": total,
        "v6_source_radius_px": radius_px,
        "v6_dirty": dirty,
        "v6_note": ("la SKY_MASK vive en un frame de offsets en pixeles (CTYPE=PIXEL), "
                    "no en el del cubo: las posiciones se transforman antes de comparar"),
    }


def run_battery(run_id: str, root: Path, combine: dict, cube: Path, adps: dict[str, Path],
                *, v6_radius_px: float = V6_SOURCE_RADIUS_PX) -> dict:
    """V1–V6 sobre el cubo combinado y los productos por exposición."""

    verification: dict[str, object] = {}
    verification["v1_stat_present"] = _result(verify.verify_stat(cube))
    verification["v2_std_residual_rms"] = {
        "ok": None,
        "status": "unavailable",
        "value": None,
        "message": ("no hay curva de respuesta de referencia contra la que comparar las "
                    "STD_RESPONSE de cada noche; se declara sin medir, no como fallo"),
    }
    verification["v3_wcs_ok"] = _v3(cube, adps, combine)
    verification["v4_adp_whitelight_corr"] = _v4(cube, adps)
    center = (combine.get("reference") or {}).get("center_pixel_yx") or [100, 100]
    verification["v5_adp_star_spec_ratio_rms"] = _v5(cube, adps, (float(center[0]), float(center[1])))
    verification["v6_sky_mask_clean"] = _v6(combine, companion_offset_yx(run_id, root), v6_radius_px)
    return verification


# --------------------------------------------------------------------------
# Fases por evidencia
# --------------------------------------------------------------------------


def gates_from_evidence(a1_run: str, root: Path, combine: dict, cube: Path) -> tuple[list[str], dict]:
    """`gates_passed` con el fichero que sostiene cada fase.

    Sin evidencia la fase no se escribe. El QC publica el porqué de cada una
    para que «fase2» no sea una palabra sino un puntero.
    """

    gates: list[str] = []
    evidence: dict[str, str] = {}

    audit = root / "runs" / a1_run / "stages" / "stage00r_p0_calibration_audit.json"
    payload = _load_json(audit)
    if payload.get("groups"):
        gates.append("fase0")
        evidence["fase0"] = f"{audit} ({len(payload['groups'])} noches asociadas)"

    exposures = combine.get("exposures") or ()
    reduced = [e for e in exposures if Path(e["file"]).exists()]
    if reduced:
        gates.extend(["fase1", "fase2"])
        evidence["fase1"] = f"{len(reduced)}/{len(exposures)} DATACUBE_FINAL por exposición en disco"
        evidence["fase2"] = evidence["fase1"]

    if cube.exists() and combine.get("output"):
        gates.append("fase3")
        evidence["fase3"] = f"{cube} ({combine.get('n_exposures')} exposiciones combinadas)"

    return gates, evidence


# --------------------------------------------------------------------------
# Ensamblado y escritura
# --------------------------------------------------------------------------


def build_payload(run_id: str, *, project_root=None, adp=None,
                  v6_radius_px: float = V6_SOURCE_RADIUS_PX) -> tuple[dict, Path]:
    root = _root(project_root)
    a1_run = resolve_a1_run(run_id, project_root=root)
    combine, combine_path = combine_qc(run_id, root)

    cube = Path(str(combine.get("output") or ""))
    if not cube.exists():
        raise A1VerifyError(f"El QC del combine declara un cubo que no existe: {cube}")

    nights = contributing_nights(combine)
    adps = adp_references(nights, root, explicit=adp)
    verification = run_battery(run_id, root, combine, cube, adps, v6_radius_px=v6_radius_px)
    gates, evidence = gates_from_evidence(a1_run, root, combine, cube)
    gates += [tag for tag, key in (("V1", "v1_stat_present"), ("V3", "v3_wcs_ok"),
                                   ("V4", "v4_adp_whitelight_corr"),
                                   ("V5", "v5_adp_star_spec_ratio_rms"),
                                   ("V6", "v6_sky_mask_clean"))
              if verification[key].get("ok")]

    audit = _load_json(root / "runs" / a1_run / "stages" / "stage00r_p0_calibration_audit.json")
    recipes = [{"name": f"night_{night}_calibrations_and_scibasic",
                "status": str((group or {}).get("status") or ""),
                "manifest": str((group or {}).get("manifest") or "")}
               for night, group in sorted((audit.get("groups") or {}).items())]
    recipes.append({"name": COMBINE_STAGE, "status": "ok", "manifest": str(combine_path)})

    payload = {
        "stage": STAGE_NAME,
        "run_id": a1_run,
        "timestamp_utc": verify_now(),
        "status": "pass" if all(v.get("ok") for k, v in verification.items() if v.get("ok") is not None) else "review_needed",
        "assembled_by": "musepipe.reduction.a1_verify",
        "reduction_profile": "cascade",
        "inputs": {
            "adp_reference": {night: {"file": str(path), "sha256": _sha256(path)}
                              for night, path in sorted(adps.items())},
            "ins_mode": "NFM-AO-N",
            "observing_nights": list(nights),
            "exposures_by_night": nights,
            "n_science_exposures": int(combine.get("n_exposures") or 0),
        },
        "recipes": recipes,
        "scipost_params": {"external_combine": {
            "method": combine.get("method"), "sigclip_k": combine.get("sigclip_k"),
            "weight": combine.get("weight_mode"), "interp_kernel": combine.get("interp_kernel"),
        }},
        "warnings_by_recipe": {COMBINE_STAGE: list(combine.get("warnings") or ())},
        "products": {
            "datacube": str(cube),
            "datacube_sha256": str(combine.get("output_sha256") or ""),
            "cube_shape_zyx": combine.get("cube_shape"),
            "finite_fraction": combine.get("finite_fraction"),
            "median_exposure_contributions": combine.get("count_median"),
            "rejected_contribution_fraction": combine.get("rejected_fraction"),
        },
        "verification": verification,
        "gates_passed": gates,
        "gate_evidence": evidence,
        "open_issues": [],
    }
    return payload, root / "runs" / a1_run / "stages" / "stage00r_qc.json"


def verify_now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--project-root", default=None)
    parser.add_argument("--adp", action="append",
                        help="ADP de referencia (repetible). Por defecto se buscan por noche bajo Data/.")
    parser.add_argument("--v6-radius-px", type=float, default=V6_SOURCE_RADIUS_PX)
    parser.add_argument("--force", action="store_true",
                        help="sobrescribir un stage00r_qc.json que NO sea el esqueleto vacío")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    try:
        payload, out_path = build_payload(args.run_id, project_root=args.project_root,
                                          adp=args.adp, v6_radius_px=args.v6_radius_px)
    except (A1VerifyError, verify.VerificationError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    for tag, block in payload["verification"].items():
        state = "unavailable" if block.get("ok") is None else ("ok" if block["ok"] else "FALLA")
        print(f"  {tag:28s} {state:12s} {block.get('message','')}")
    print(f"  gates_passed: {payload['gates_passed']}")

    # El guardia es de ESCRITURA: un `--dry-run` sobre un run ya reconstruido
    # tiene que poder mirarse, que es justo cuando más falta hace.
    if args.dry_run:
        print(f"(dry-run) no se escribió {out_path}")
        return 0

    existing = _load_json(out_path)
    if existing and not is_qc_skeleton(existing) and not args.force:
        print(f"ERROR: {out_path} ya tiene una reducción escrita (no es el esqueleto). "
              "Usa --force si de verdad quieres pisarla.", file=sys.stderr)
        return 2

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
