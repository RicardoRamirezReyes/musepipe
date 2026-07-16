"""Fetch, convert and manifest the five external libraries of the G3 real plan
(docs/plan_g3_real_2026-07-16.md, WP-G3R-2). Pure mechanics — no science.

Each family is a subcommand. A subcommand either downloads a directly-available
public source and converts it to the frozen internal cache format
(:mod:`musepipe.models.cache`), or — when the source needs manual download /
registration — prints EXACT instructions and the expected input layout, then
exits non-zero (plan WP-G3R-2, task 2). Every family writes a ``PROVENANCE.json``
and a ``MANIFEST.sha256`` so :func:`verify_manifest` can gate the consumers.

Subdirectories under ``g3_libraries_root`` are frozen by decision D14.

Usage::

    python scripts/fetch_g3_libraries.py tracks-bhac15 [--run-id ROXs12b_B_adp]
    python scripts/fetch_g3_libraries.py templates-young --input-dir <dir>
    python scripts/fetch_g3_libraries.py verify            # verify all present
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from musepipe.config import load_run_config, project_root_path  # noqa: E402
from musepipe.models.cache import write_spectrum_npz, write_tracks_npz  # noqa: E402
from musepipe.models.manifest import (  # noqa: E402
    LIBRARY_SUBDIRS,
    MANIFEST_NAME,
    verify_manifest,
    write_manifest,
)

# Frozen family subdirectories (decision D14); shared with the input gate.
FAMILY_SUBDIR = LIBRARY_SUBDIRS

BHAC15_ISO_URL = (
    "https://perso.ens-lyon.fr/isabelle.baraffe/BHAC15dir/BHAC15_iso.2mass"
)
_HTTP_TIMEOUT_S = 120
_USER_AGENT = "musepipe-g3-fetch/1.0 (research; contact repo maintainer)"


# --------------------------------------------------------------------------- #
# Generic helpers
# --------------------------------------------------------------------------- #
def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _download(url: str, dest: Path) -> Path:
    """Stream ``url`` to ``dest``. RuntimeError on any transport failure."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=_HTTP_TIMEOUT_S) as resp:
            data = resp.read()
    except Exception as exc:  # noqa: BLE001 - report any transport error uniformly
        raise RuntimeError(f"download failed: {url}\n  {exc}") from exc
    dest.write_bytes(data)
    return dest


def _dir_size_bytes(path: Path) -> int:
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def _finalise(family_dir: Path, cache_rels: list[str], provenance: dict) -> dict:
    """Write PROVENANCE.json + MANIFEST.sha256 covering the cache products."""
    prov_path = family_dir / "PROVENANCE.json"
    prov_path.write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n",
                         encoding="utf-8")
    rels = sorted(set(cache_rels) | {"PROVENANCE.json"})
    write_manifest(family_dir, rels)
    info = verify_manifest(family_dir)
    size_mb = _dir_size_bytes(family_dir) / 1e6
    print(f"OK  {family_dir.name}: {len(cache_rels)} cache file(s), "
          f"{size_mb:.1f} MB, manifest {info['n_files']} entries verified.")
    return info


def _instruct(family_dir: Path, *, citation: str, source_hint: str,
              layout: str) -> "NoReturn":  # type: ignore[name-defined]
    """Print exact manual-acquisition instructions and stop (exit 2)."""
    print(
        f"\nMANUAL ACQUISITION REQUIRED for '{family_dir.name}'\n"
        f"  Frozen citation (D-decision): {citation}\n"
        f"  Suggested source: {source_hint}\n"
        f"  Place the downloaded source in an --input-dir, then re-run this\n"
        f"  subcommand with --input-dir <that dir>. Expected layout:\n"
        f"{layout}\n"
        f"  Nothing was written under {family_dir}. This is a plan STOP point:\n"
        f"  if the source is dead or licence-restricted, report to the human\n"
        f"  with concrete alternatives (do NOT substitute a different library).",
        file=sys.stderr,
    )
    raise SystemExit(2)


# --------------------------------------------------------------------------- #
# Pure parsers/converters (unit-tested with synthetic input, no network)
# --------------------------------------------------------------------------- #
def parse_bhac15_iso(text: str) -> dict:
    """Parse a BHAC15 isochrone file into the internal track arrays.

    Format (verified 2026-07-16 against BHAC15_iso.2mass): age blocks headed by
    ``!  t (Gyr) =   <age>`` and data rows ``M/Ms Teff log(L/Ls) log(g) R/Rs
    [Li/Li0 magnitudes...]``. Only the first five physical columns are used.
    """
    ages, masses, teffs, lbols, radii, loggs = [], [], [], [], [], []
    current_age = None
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("!"):
            if "t (Gyr)" in line:
                # e.g. "!  t (Gyr) =   0.0005"
                current_age = float(line.split("=", 1)[1].strip())
            continue
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            mass, teff, logl, logg, radius = (float(parts[i]) for i in range(5))
        except ValueError:
            continue
        if current_age is None:
            raise RuntimeError(
                "BHAC15 data row seen before any 't (Gyr)' age header")
        ages.append(current_age)
        masses.append(mass)
        teffs.append(teff)
        lbols.append(10.0 ** logl)
        loggs.append(logg)
        radii.append(radius)
    if not masses:
        raise RuntimeError("no BHAC15 data rows parsed (unexpected format)")
    return {
        "mass_msun": np.asarray(masses, float),
        "age_gyr": np.asarray(ages, float),
        "teff_k": np.asarray(teffs, float),
        "l_bol_lsun": np.asarray(lbols, float),
        "radius_rsun": np.asarray(radii, float),
        "logg": np.asarray(loggs, float),
    }


def _read_two_column_ascii(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.loadtxt(path, comments=("#", "!"), usecols=(0, 1))
    if data.ndim != 2 or data.shape[1] < 2:
        raise RuntimeError(f"expected 2-column wave/flux ASCII: {path}")
    return data[:, 0].astype(float), data[:, 1].astype(float)


def convert_spectra_from_intermediate(input_dir: Path, family_dir: Path, *,
                                      key_cols: tuple[str, ...],
                                      provenance_base: dict) -> list[str]:
    """Convert a prepared intermediate directory of ASCII spectra to cache npz.

    ``input_dir/index.csv`` must have columns ``key_cols + ('file',)`` and may
    carry ``wave_frame`` (``air``/``vacuum``). Each referenced file is a
    two-column ``wave_A flux`` ASCII. Returns the cache-relative paths written.
    """
    index_path = input_dir / "index.csv"
    if not index_path.exists():
        raise RuntimeError(f"missing intermediate index: {index_path}")
    with index_path.open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise RuntimeError(f"empty index: {index_path}")
    needed = set(key_cols) | {"file"}
    missing_cols = needed - set(rows[0])
    if missing_cols:
        raise RuntimeError(f"{index_path} missing columns: {sorted(missing_cols)}")

    rels: list[str] = []
    for row in rows:
        src = input_dir / row["file"]
        wave, flux = _read_two_column_ascii(src)
        identity = {k: row[k] for k in key_cols}
        meta = {**provenance_base, **identity,
                "wave_frame": row.get("wave_frame", "unspecified")}
        stem = "_".join(str(row[k]) for k in key_cols).replace(" ", "").replace("/", "-")
        rel = f"{stem}.npz"
        write_spectrum_npz(family_dir / rel, wave, flux, meta)
        rels.append(rel)
    return rels


# --------------------------------------------------------------------------- #
# Family subcommands
# --------------------------------------------------------------------------- #
def _resolve_family_dir(cfg: dict, subcommand: str, project_root: Path) -> Path:
    if "g3_libraries_root" not in cfg:
        raise RuntimeError("config missing 'g3_libraries_root' (D14).")
    root = (project_root / str(cfg["g3_libraries_root"])).resolve()
    family_dir = root / FAMILY_SUBDIR[subcommand]
    family_dir.mkdir(parents=True, exist_ok=True)
    return family_dir


def fetch_tracks_bhac15(args, cfg: dict, family_dir: Path) -> None:
    src_path = family_dir / "_source" / "BHAC15_iso.2mass"
    if args.input_dir:
        given = Path(args.input_dir) / "BHAC15_iso.2mass"
        if not given.exists():
            raise RuntimeError(f"--input-dir given but {given} not found")
        text = given.read_text(encoding="utf-8", errors="replace")
        src_url = f"local:{given}"
    else:
        _download(BHAC15_ISO_URL, src_path)
        text = src_path.read_text(encoding="utf-8", errors="replace")
        src_url = BHAC15_ISO_URL
    arrays = parse_bhac15_iso(text)
    citation = _tracks_citation(cfg, "BHAC15")
    meta = {"family": "BHAC15", "citation": citation, "url": src_url,
            "downloaded_utc": _utc_now(),
            "license": "public (ENS Lyon; cite Baraffe et al. 2015)",
            "n_points": int(arrays["mass_msun"].size),
            "note": "photometric columns beyond R/Rs dropped; L from 10**log(L/Ls)"}
    rel = "bhac15_tracks.npz"
    write_tracks_npz(family_dir / rel, arrays, meta)
    provenance = {"family": "tracks_bhac15", "citation": citation, "url": src_url,
                  "downloaded_utc": meta["downloaded_utc"], "n_points": meta["n_points"],
                  "teff_k_range": [float(arrays["teff_k"].min()), float(arrays["teff_k"].max())],
                  "mass_msun_range": [float(arrays["mass_msun"].min()), float(arrays["mass_msun"].max())],
                  "age_gyr_range": [float(arrays["age_gyr"].min()), float(arrays["age_gyr"].max())]}
    _finalise(family_dir, [rel], provenance)


def fetch_tracks_atmo2020(args, cfg: dict, family_dir: Path) -> None:
    citation = _tracks_citation(cfg, "ATMO2020")
    if not args.input_dir:
        _instruct(
            family_dir, citation=citation,
            source_hint=("ATMO2020 evolutionary tracks, Phillips et al. 2020 "
                         "(A&A 637, A38). Host http://opendata.erc-atmo.eu is a "
                         "single-page app with no directory listing — download "
                         "the 'ATMO_2020 evolutionary tracks' (CEQ chemistry) by "
                         "hand."),
            layout=(
                "    <input-dir>/index.csv  columns: mass_msun,file\n"
                "    <input-dir>/<file>     ASCII track: age_gyr teff_k "
                "log(L/Lsun) radius_rsun logg  (one mass per file)\n"
                "    (confirm the column order in the header before running; do "
                "NOT guess units — plan WP-G3R-5 STOP)"),
        )
    # Intermediate-directory ingestion: one ASCII file per mass, columns
    # age_gyr teff_k logL radius_rsun logg (documented in the index).
    input_dir = Path(args.input_dir)
    index_path = input_dir / "index.csv"
    if not index_path.exists():
        raise RuntimeError(f"missing intermediate index: {index_path}")
    with index_path.open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    ages, masses, teffs, lbols, radii, loggs = [], [], [], [], [], []
    for row in rows:
        mass = float(row["mass_msun"])
        data = np.loadtxt(input_dir / row["file"], comments=("#", "!"))
        data = np.atleast_2d(data)
        for age, teff, logl, radius, logg in data[:, :5]:
            ages.append(age); masses.append(mass); teffs.append(teff)
            lbols.append(10.0 ** logl); radii.append(radius); loggs.append(logg)
    arrays = {"mass_msun": np.asarray(masses, float), "age_gyr": np.asarray(ages, float),
              "teff_k": np.asarray(teffs, float), "l_bol_lsun": np.asarray(lbols, float),
              "radius_rsun": np.asarray(radii, float), "logg": np.asarray(loggs, float)}
    meta = {"family": "ATMO2020", "citation": citation, "url": f"local:{input_dir}",
            "downloaded_utc": _utc_now(),
            "license": "public (ERC ATMO; cite Phillips et al. 2020)",
            "n_points": int(arrays["mass_msun"].size)}
    rel = "atmo2020_tracks.npz"
    write_tracks_npz(family_dir / rel, arrays, meta)
    provenance = {"family": "tracks_atmo2020", "citation": citation,
                  "url": meta["url"], "downloaded_utc": meta["downloaded_utc"],
                  "n_points": meta["n_points"]}
    _finalise(family_dir, [rel], provenance)


def _fetch_spectra_family(args, cfg, family_dir, *, citation, key_cols,
                          source_hint, kind_label):
    if not args.input_dir:
        _instruct(
            family_dir, citation=citation, source_hint=source_hint,
            layout=(
                f"    <input-dir>/index.csv  columns: {','.join(key_cols)},file"
                f"[,wave_frame]\n"
                f"    <input-dir>/<file>     two-column ASCII: wave_A flux "
                f"(F_lambda); {kind_label}\n"
                f"    (native FITS/VOTable must be reduced to this intermediate "
                f"first; do NOT guess wavelength frame/units — plan STOP)"),
        )
    provenance_base = {"family": family_dir.name, "citation": citation,
                       "downloaded_utc": _utc_now(), "source": source_hint}
    rels = convert_spectra_from_intermediate(
        Path(args.input_dir), family_dir, key_cols=key_cols,
        provenance_base=provenance_base)
    provenance = {"family": family_dir.name, "citation": citation,
                  "downloaded_utc": provenance_base["downloaded_utc"],
                  "n_spectra": len(rels), "source": source_hint}
    _finalise(family_dir, rels, provenance)


def fetch_bt_settl(args, cfg, family_dir):
    _fetch_spectra_family(
        args, cfg, family_dir,
        citation=str(cfg.get("g3_atmosphere_citation", "Allard et al. 2012")),
        key_cols=("teff_k", "logg"),
        source_hint=("BT-Settl CIFIST grid (Allard et al. 2012) via the SVO "
                     "Theoretical Spectra server; nodes on the D4 axes "
                     "Teff 2000-4500 K/100, logg 3.5-5.5/0.5"),
        kind_label="one file per (Teff, logg) node")


def fetch_templates_young(args, cfg, family_dir):
    _fetch_spectra_family(
        args, cfg, family_dir,
        citation=str(cfg.get("g3_template_citation", "Manara et al. 2013/2017")),
        key_cols=("spt",),
        source_hint=("X-shooter Class III young M-L templates (Manara et al. "
                     "2013 A&A 551 A107; 2017 A&A 605 A86), VIS arm, via CDS/"
                     "VizieR or the authors' repository"),
        kind_label="one file per spectral type (VIS arm)")


def fetch_templates_field(args, cfg, family_dir):
    _fetch_spectra_family(
        args, cfg, family_dir,
        citation=str(cfg.get("g3_template_field_citation", "Kesseli et al. 2017")),
        key_cols=("spt",),
        source_hint=("SDSS empirical field templates O5-L3 (Kesseli et al. 2017 "
                     "ApJS 230 16) via CDS/VizieR J/ApJS/230/16"),
        kind_label="one file per spectral type")


def _tracks_citation(cfg: dict, family: str) -> str:
    families = list(cfg.get("g3_tracks_families", []))
    citations = list(cfg.get("g3_tracks_citations", []))
    if family in families and len(citations) == len(families):
        return str(citations[families.index(family)])
    return {"BHAC15": "Baraffe et al. 2015",
            "ATMO2020": "Phillips et al. 2020"}[family]


DISPATCH = {
    "bt-settl": fetch_bt_settl,
    "templates-young": fetch_templates_young,
    "templates-field": fetch_templates_field,
    "tracks-bhac15": fetch_tracks_bhac15,
    "tracks-atmo2020": fetch_tracks_atmo2020,
}


def cmd_verify(cfg: dict, project_root: Path) -> int:
    """Verify every family that is present; report the ones that are missing."""
    root = (project_root / str(cfg["g3_libraries_root"])).resolve()
    rc = 0
    for sub, subdir in FAMILY_SUBDIR.items():
        family_dir = root / subdir
        if not (family_dir / MANIFEST_NAME).exists():
            print(f"--  {subdir}: not present (run '{sub}')")
            continue
        try:
            info = verify_manifest(family_dir)
            print(f"OK  {subdir}: {info['n_files']} entries verified.")
        except RuntimeError as exc:
            print(f"FAIL {subdir}: {exc}")
            rc = 1
    return rc


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("subcommand",
                        choices=list(DISPATCH) + ["verify"])
    parser.add_argument("--run-id", default="ROXs12b_B_adp")
    parser.add_argument("--input-dir", default=None,
                        help="directory of prepared source files (manual sources)")
    args = parser.parse_args(argv)

    project_root = project_root_path(Path(__file__).resolve().parents[1])
    cfg = dict(load_run_config(args.run_id, project_root=project_root).config)

    if args.subcommand == "verify":
        return cmd_verify(cfg, project_root)

    family_dir = _resolve_family_dir(cfg, args.subcommand, project_root)
    DISPATCH[args.subcommand](args, cfg, family_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
