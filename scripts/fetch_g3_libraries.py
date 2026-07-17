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
import re
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from astropy.io import fits

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
ATMO2020_TAR_URL = (
    "https://perso.ens-lyon.fr/isabelle.baraffe/ATMO2020/ATMO_2020_models.tar.gz"
)
CDS_FTP = "https://cdsarc.cds.unistra.fr/ftp"
# Manara Class III young templates: 2013 catalogue + 2017 additions (D1).
MANARA_CATALOGS = ("J/A+A/551/A107", "J/A+A/605/A86")
# Kesseli SDSS field templates (D2).
KESSELI_CATALOG = "J/ApJS/230/16"
# BT-Settl CIFIST atmosphere grid via the SVO Theoretical Spectra server (D4).
SVO_BTSETTL_SSAP = "http://svo2.cab.inta-csic.es/theory/newov2/ssap.php?model=bt-settl-cifist"
_HTTP_TIMEOUT_S = 300
_USER_AGENT = "musepipe-g3-fetch/1.0 (research; contact repo maintainer)"


# --------------------------------------------------------------------------- #
# Generic helpers
# --------------------------------------------------------------------------- #
def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _download(url: str, dest: Path, *, tries: int = 1) -> Path:
    """Stream ``url`` to ``dest`` with ``tries`` attempts. RuntimeError if all
    attempts fail (message includes the last transport error)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    last = None
    for attempt in range(1, tries + 1):
        try:
            with urllib.request.urlopen(req, timeout=_HTTP_TIMEOUT_S) as resp:
                data = resp.read()
            dest.write_bytes(data)
            return dest
        except Exception as exc:  # noqa: BLE001 - report any transport error uniformly
            last = exc
    raise RuntimeError(f"download failed after {tries} tries: {url}\n  {last}")


def _download_text(url: str) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=_HTTP_TIMEOUT_S) as resp:
            return resp.read().decode("utf-8", errors="replace")
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"download failed: {url}\n  {exc}") from exc


def _hrefs(html: str, *, suffix: str) -> list[str]:
    """Return distinct hrefs ending in ``suffix`` from a directory-index page."""
    out = []
    for m in re.finditer(r'href="([^"]+)"', html):
        name = m.group(1)
        if name.endswith(suffix) and "/" not in name and name not in out:
            out.append(name)
    return out


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


def parse_atmo2020_ceq(text: str) -> dict:
    """Parse ATMO2020 CEQ evolutionary tracks into the internal track arrays.

    Format (verified 2026-07-16 against evolutionary_tracks/ATMO_CEQ/
    MKO_WISE_IRAC/*_ATMO_CEQ_vega.txt): columns Mass[Msun] Age[Gyr] Teff[K]
    Luminosity Radius[Rsun] log(g) + photometry. Despite the 'L/Lsun' header,
    the Luminosity column is log10(L/Lsun) (negative values), like BHAC15.
    Each row carries its own mass, so concatenated files parse in one pass.
    """
    masses, ages, teffs, lbols, radii, loggs = [], [], [], [], [], []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 6:
            continue
        try:
            mass, age, teff, logl, radius, logg = (float(parts[i]) for i in range(6))
        except ValueError:
            continue
        masses.append(mass); ages.append(age); teffs.append(teff)
        lbols.append(10.0 ** logl); radii.append(radius); loggs.append(logg)
    if not masses:
        raise RuntimeError("no ATMO2020 CEQ data rows parsed (unexpected format)")
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


def wave_from_linear_wcs(header) -> np.ndarray:
    """1-D linear-WCS wavelength axis from CRVAL1/CDELT1(or CD1_1)/CRPIX1."""
    n = int(header["NAXIS1"])
    crval = float(header["CRVAL1"])
    crpix = float(header.get("CRPIX1", 1.0))
    if "CDELT1" in header:
        cdelt = float(header["CDELT1"])
    elif "CD1_1" in header:
        cdelt = float(header["CD1_1"])
    else:
        raise RuntimeError("linear-WCS spectrum missing CDELT1/CD1_1")
    return crval + (np.arange(n) + 1.0 - crpix) * cdelt


def to_angstrom(wave, *, source_hint: str = "") -> tuple[np.ndarray, str]:
    """Resolve an optical wavelength axis to Angstrom by physical range.

    Optical spectra live in either nm (~300-1100) or Angstrom (~3000-11000);
    the two windows don't overlap, so the unit is decided by the numeric range,
    NOT by a header label (Manara's WAT1 mislabels nm as 'angstroms'). Anything
    outside both windows raises ``RuntimeError`` (a plan STOP, not a guess).
    """
    wave = np.asarray(wave, dtype=np.float64)
    wmin, wmax = float(np.nanmin(wave)), float(np.nanmax(wave))
    if 200.0 <= wmin and wmax <= 1200.0:
        return wave * 10.0, "nm"
    if 2000.0 <= wmin and wmax <= 12000.0:
        return wave, "angstrom"
    raise RuntimeError(
        f"cannot resolve wavelength unit for {source_hint}: range "
        f"{wmin:.3f}-{wmax:.3f} is neither optical nm nor Angstrom")


def read_kesseli_fits(path) -> tuple[np.ndarray, np.ndarray]:
    """Kesseli+2017 template: BinTable ext 1 with LogLam (log10 Å) and Flux."""
    with fits.open(path) as hdul:
        data = hdul[1].data
        names = [n.lower() for n in data.columns.names]
        if "loglam" not in names or "flux" not in names:
            raise RuntimeError(f"unexpected Kesseli columns: {data.columns.names}")
        loglam = np.asarray(data[data.columns.names[names.index("loglam")]], float)
        flux = np.asarray(data[data.columns.names[names.index("flux")]], float)
    return 10.0 ** loglam, flux


def read_manara_visual_fits(path) -> tuple[np.ndarray, np.ndarray, str]:
    """Manara Class III VIS spectrum: 1-D linear WCS (values in nm)."""
    with fits.open(path) as hdul:
        header = hdul[0].header
        flux = np.asarray(hdul[0].data, dtype=np.float64)
    wave = wave_from_linear_wcs(header)
    wave_A, unit = to_angstrom(wave, source_hint=str(path))
    return wave_A, flux, unit


def _kesseli_wanted(name: str) -> str | None:
    """Return the SpT for a Kesseli file we cache (solar-metallicity dwarfs and
    bare main-sequence composites), else None. Filename encodes the identity
    (CDS convention), e.g. 'M0_+0.0_Dwarf.fits' or 'A0.fits'; giants, subdwarfs
    and non-solar metallicities are excluded to keep the field library clean."""
    if not name.endswith(".fits"):
        return None
    stem = name[:-5]
    parts = stem.split("_")
    spt = parts[0]
    if not re.fullmatch(r"[OBAFGKMLT]\d(\.\d)?", spt):
        return None
    if len(parts) == 1:  # bare composite (early types, treated as dwarf)
        return spt
    if len(parts) == 3 and parts[1] == "+0.0" and parts[2] == "Dwarf":
        return spt
    return None


def read_svo_spectrum_votable(path) -> tuple[np.ndarray, np.ndarray]:
    """SVO BT-Settl spectrum VOTable: WAVELENGTH (Å) + FLUX (erg/cm2/s/Å)."""
    import warnings
    from astropy.io.votable import parse as _parse_votable
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tab = _parse_votable(path).get_first_table().to_table()
    names = {n.upper(): n for n in tab.colnames}
    if "WAVELENGTH" not in names or "FLUX" not in names:
        raise RuntimeError(f"unexpected SVO spectrum columns: {tab.colnames}")
    wave = np.asarray(tab[names["WAVELENGTH"]], dtype=np.float64)
    flux = np.asarray(tab[names["FLUX"]], dtype=np.float64)
    return wave, flux


def select_btsettl_nodes(rows, *, teff_range, logg_range) -> list:
    """Filter SSAP rows to the D4 box at solar metallicity (meta=alpha=0).

    Returns sorted ``[(teff, logg, access_url), ...]``. ``rows`` is any iterable
    of mappings with keys teff, logg, meta, alpha and 'Access.Reference'.
    """
    out = []
    for r in rows:
        if float(r["meta"]) != 0.0 or float(r["alpha"]) != 0.0:
            continue
        teff, logg = float(r["teff"]), float(r["logg"])
        if teff_range[0] <= teff <= teff_range[1] and logg_range[0] <= logg <= logg_range[1]:
            out.append((teff, logg, str(r["Access.Reference"])))
    return sorted(out)


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
    """ATMO2020 CEQ evolutionary tracks (D5) from ENS Lyon (44 MB tarball).

    The CEQ/MKO_WISE_IRAC/*_ATMO_CEQ_vega.txt files carry Teff/L/R/g vs age per
    mass; only the six physical columns are kept. --input-dir points at an
    already-extracted directory of those files (skips the download)."""
    import tarfile

    citation = _tracks_citation(cfg, "ATMO2020")
    src = family_dir / "_source"
    if args.input_dir:
        files = sorted(Path(args.input_dir).rglob("*_ATMO_CEQ_vega.txt"))
        if not files:
            raise RuntimeError(
                f"no *_ATMO_CEQ_vega.txt under --input-dir {args.input_dir}")
        text = "\n".join(f.read_text(encoding="utf-8", errors="replace") for f in files)
        n_files, src_url = len(files), f"local:{args.input_dir}"
    else:
        tar_path = _download(ATMO2020_TAR_URL, src / "ATMO_2020_models.tar.gz", tries=3)
        texts = []
        with tarfile.open(tar_path) as tf:  # read members in-place, no extraction
            for m in tf.getmembers():
                if (m.isfile() and "ATMO_CEQ/MKO_WISE_IRAC" in m.name
                        and m.name.endswith("_ATMO_CEQ_vega.txt")):
                    texts.append(tf.extractfile(m).read().decode("utf-8", "replace"))
        if not texts:
            raise RuntimeError("ATMO2020 tarball has no ATMO_CEQ/MKO_WISE_IRAC tracks")
        text, n_files, src_url = "\n".join(texts), len(texts), ATMO2020_TAR_URL
    arrays = parse_atmo2020_ceq(text)
    meta = {"family": "ATMO2020", "chemistry": "CEQ", "citation": citation,
            "url": src_url, "downloaded_utc": _utc_now(), "n_mass_files": n_files,
            "license": ("public (ENS Lyon / ERC ATMO); cite Phillips et al. 2020; "
                        "authors request contact before publication"),
            "n_points": int(arrays["mass_msun"].size),
            "note": "Luminosity column is log10(L/Lsun) despite header; stored linear"}
    rel = "atmo2020_ceq_tracks.npz"
    write_tracks_npz(family_dir / rel, arrays, meta)
    provenance = {"family": "tracks_atmo2020", "citation": citation, "url": src_url,
                  "downloaded_utc": meta["downloaded_utc"], "chemistry": "CEQ",
                  "n_points": meta["n_points"], "n_mass_files": n_files,
                  "mass_msun_range": [float(arrays["mass_msun"].min()),
                                      float(arrays["mass_msun"].max())],
                  "teff_k_range": [float(arrays["teff_k"].min()),
                                   float(arrays["teff_k"].max())]}
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
    """BT-Settl CIFIST grid (D4) via SVO. Each node is a ~60 MB full-range
    spectrum trimmed to the cache window; the full D4 box is ~146 nodes (~9 GB
    of transfer). Use --limit N to fetch a subset (validation)."""
    citation = str(cfg.get("g3_atmosphere_citation", "Allard et al. 2012"))
    if args.input_dir:
        _fetch_spectra_family(args, cfg, family_dir, citation=citation,
                              key_cols=("teff_k", "logg"),
                              source_hint="manual intermediate",
                              kind_label="one file per (Teff, logg) node")
        return
    import warnings
    from astropy.io.votable import parse as _parse_votable
    teff_axis = list(cfg.get("g3_atmo_teff_axis_k", [2000.0, 4500.0, 100.0]))
    logg_axis = list(cfg.get("g3_atmo_logg_axis", [3.5, 5.5, 0.5]))
    src = family_dir / "_source"
    ssap_path = _download(SVO_BTSETTL_SSAP, src / "ssap_index.xml")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rows = _parse_votable(ssap_path).get_first_table().to_table()
    nodes = select_btsettl_nodes(rows, teff_range=(teff_axis[0], teff_axis[1]),
                                 logg_range=(logg_axis[0], logg_axis[1]))
    if getattr(args, "limit", None):
        nodes = nodes[: int(args.limit)]
    if not nodes:
        raise RuntimeError("no BT-Settl nodes selected within the D4 box.")
    downloaded_utc = _utc_now()
    rels: list[str] = []
    failed: list = []
    print(f"    {len(nodes)} BT-Settl nodes to fetch (~60 MB raw each)...")
    for i, (teff, logg, url) in enumerate(nodes, 1):
        rel = f"teff{int(teff)}_logg{logg:.1f}.npz"
        target = family_dir / rel
        if target.exists() and target.stat().st_size > 0:
            rels.append(rel)  # resume: node already cached
            continue
        try:
            raw = _download(url, src / f"tmp_{int(teff)}_{logg}.xml", tries=3)
            wave, flux = read_svo_spectrum_votable(raw)
            raw.unlink()  # keep only the trimmed cache product
            meta = {"teff_k": teff, "logg": logg, "meta_fe_h": 0.0, "alpha": 0.0,
                    "citation": citation, "grid": "BT-Settl CIFIST2011",
                    "flux_unit": "erg/cm2/s/A", "wave_unit": "angstrom",
                    "wave_frame": "vacuum (synthetic)", "source_url": url,
                    "downloaded_utc": downloaded_utc}
            write_spectrum_npz(target, wave, flux, meta)
            rels.append(rel)
        except Exception as exc:  # noqa: BLE001 - one bad node must not abort 9 GB
            print(f"      WARN teff{int(teff)} logg{logg}: {exc}", file=sys.stderr)
            failed.append([teff, logg])
        if i % 10 == 0 or i == len(nodes):
            print(f"      {i}/{len(nodes)} processed "
                  f"({len(rels)} cached, {len(failed)} failed).")
    if not rels:
        raise RuntimeError("BT-Settl: no nodes cached (all downloads failed).")
    provenance = {"family": "bt-settl-cifist", "citation": citation,
                  "grid": "BT-Settl CIFIST2011 (SVO)", "downloaded_utc": downloaded_utc,
                  "n_nodes": len(rels), "n_failed": len(failed), "failed_nodes": failed,
                  "teff_axis_k": teff_axis, "logg_axis": logg_axis,
                  "metallicity": "solar (meta=0, alpha=0)", "ssap": SVO_BTSETTL_SSAP,
                  "partial": bool(getattr(args, "limit", None))}
    _finalise(family_dir, rels, provenance)


def fetch_templates_young(args, cfg, family_dir):
    """Manara Class III young templates (D1), VIS arm, from CDS (2013 + 2017)."""
    citation = str(cfg.get("g3_template_citation", "Manara et al. 2013/2017"))
    if args.input_dir:
        _fetch_spectra_family(args, cfg, family_dir, citation=citation,
                              key_cols=("spt",), source_hint="manual intermediate",
                              kind_label="one file per spectral type (VIS arm)")
        return
    downloaded_utc = _utc_now()
    src_dir = family_dir / "_source"
    rels: list[str] = []
    for cat in MANARA_CATALOGS:
        base = f"{CDS_FTP}/{cat}"
        try:
            index = _download_text(f"{base}/spectra.dat")
        except RuntimeError as exc:
            print(f"WARN  {cat}: index unavailable ({exc}); skipping.", file=sys.stderr)
            continue
        n_cat = 0
        for line in index.splitlines():
            fields = line.split()
            vfiles = [f for f in fields if f.endswith("_V.fit")]
            if len(fields) < 2 or not vfiles:
                continue
            # Column layout differs (2013 has no coords; 2017 has RA/Dec), but the
            # order [SpType, U, V, N] is invariant → SpType is 2 tokens before V.
            vidx = fields.index(vfiles[0])
            obj, vfile = fields[0], vfiles[0]
            spt = fields[vidx - 2] if vidx >= 2 else fields[1]
            local = _download(f"{base}/sp/{vfile}",
                              src_dir / cat.replace("/", "_") / vfile)
            wave_A, flux, unit = read_manara_visual_fits(local)
            meta = {"spt": spt, "object": obj, "gravity_class": "young",
                    "citation": citation, "catalog": cat, "source_file": vfile,
                    "wave_unit_source": unit,
                    "wave_frame": "X-shooter reduced VIS; verify air/vacuum in WP-6",
                    "note": "WAT1 mislabels nm as angstroms; unit set by physical range",
                    "downloaded_utc": downloaded_utc}
            rel = f"{obj}_{spt}.npz".replace("/", "-").replace(" ", "")
            write_spectrum_npz(family_dir / rel, wave_A, flux, meta)
            rels.append(rel)
            n_cat += 1
        print(f"    {cat}: {n_cat} VIS templates.")
    if not rels:
        raise RuntimeError("no Manara VIS templates fetched (all catalogs failed).")
    provenance = {"family": "templates_young", "citation": citation,
                  "catalogs": list(MANARA_CATALOGS), "arm": "X-shooter VIS",
                  "downloaded_utc": downloaded_utc, "n_spectra": len(rels)}
    _finalise(family_dir, rels, provenance)


def fetch_templates_field(args, cfg, family_dir):
    """Kesseli+2017 SDSS field templates (D2): solar-metallicity dwarfs from CDS."""
    citation = str(cfg.get("g3_template_field_citation", "Kesseli et al. 2017"))
    if args.input_dir:
        _fetch_spectra_family(args, cfg, family_dir, citation=citation,
                              key_cols=("spt",), source_hint="manual intermediate",
                              kind_label="one file per spectral type")
        return
    base = f"{CDS_FTP}/{KESSELI_CATALOG}/fits"
    names = _hrefs(_download_text(base + "/"), suffix=".fits")
    chosen: dict[str, str] = {}
    for name in names:
        spt = _kesseli_wanted(name)
        if spt is None:
            continue
        if spt not in chosen or "_+0.0_Dwarf" in name:  # prefer explicit solar dwarf
            chosen[spt] = name
    if not chosen:
        raise RuntimeError("no Kesseli field templates matched the selection.")
    downloaded_utc = _utc_now()
    src_dir = family_dir / "_source"
    rels: list[str] = []
    for spt, name in sorted(chosen.items()):
        local = _download(f"{base}/{name}", src_dir / name)
        wave_A, flux = read_kesseli_fits(local)
        meta = {"spt": spt, "gravity_class": "field", "lum_class": "dwarf",
                "citation": citation, "catalog": KESSELI_CATALOG, "source_file": name,
                "wave_frame": "vacuum (SDSS)", "flux_norm": "normalized",
                "downloaded_utc": downloaded_utc}
        write_spectrum_npz(family_dir / f"{spt}.npz", wave_A, flux, meta)
        rels.append(f"{spt}.npz")
    provenance = {"family": "templates_field", "citation": citation,
                  "catalog": KESSELI_CATALOG, "downloaded_utc": downloaded_utc,
                  "n_spectra": len(rels),
                  "selection": "solar-metallicity dwarfs + bare composites"}
    _finalise(family_dir, rels, provenance)


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
    parser.add_argument("--limit", type=int, default=None,
                        help="cap the number of grid nodes (bt-settl validation)")
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
