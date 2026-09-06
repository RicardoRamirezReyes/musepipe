"""External-library integrity manifests and cache location (WP-G3R-2).

Pure mechanics — no science. The external library FILES never enter the repo
(spec G3 §1.3); they live under ``g3_libraries_root`` (decision D14) with a
``MANIFEST.sha256`` per family. ``verify_manifest`` checks integrity;
``library_root`` resolves the configured root relative to the project.

Manifest format (one entry per line, blank/``#`` lines ignored)::

    <sha256 hex>  <path relative to the family directory>
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path

from ..config import project_root_path

MANIFEST_NAME = "MANIFEST.sha256"
_READ_CHUNK = 1 << 20  # 1 MiB

# Frozen family subdirectories under g3_libraries_root (decision D14). Keyed by
# the fetch-script subcommand; the value is the on-disk directory name.
LIBRARY_SUBDIRS = {
    "bt-settl": "bt-settl-cifist",
    "templates-young": "templates_young",
    "templates-field": "templates_field",
    "tracks-bhac15": "tracks_bhac15",
    "tracks-atmo2020": "tracks_atmo2020",
}


def _sha256_of_file(path: str | Path, *, chunk: int = _READ_CHUNK) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def _sha256_of_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def parse_manifest(text: str) -> list[tuple[str, str]]:
    """Parse manifest ``text`` into ``[(sha256_lower, relative_path), ...]``.

    Blank lines and ``#`` comments are ignored. A line that is not exactly a
    sha256 (64 hex chars) followed by a path raises ``RuntimeError``.
    """
    entries: list[tuple[str, str]] = []
    for lineno, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(None, 1)
        if len(parts) != 2:
            raise RuntimeError(
                f"malformed manifest line {lineno}: {raw!r} "
                "(expected '<sha256>  <relative_path>')")
        sha, rel = parts[0].lower(), parts[1].strip()
        if len(sha) != 64 or any(c not in "0123456789abcdef" for c in sha):
            raise RuntimeError(
                f"manifest line {lineno}: {parts[0]!r} is not a sha256 hex digest")
        entries.append((sha, rel))
    return entries


def write_manifest(family_dir: str | Path, relative_paths) -> Path:
    """Compute sha256 for each of ``relative_paths`` under ``family_dir`` and
    write ``MANIFEST.sha256``. Returns the manifest path. Paths are stored
    sorted for reproducibility. ``RuntimeError`` if a listed file is missing."""
    family_dir = Path(family_dir)
    rels = sorted(str(Path(p).as_posix()) for p in relative_paths)
    if not rels:
        raise RuntimeError(f"refusing to write an empty manifest in {family_dir}")
    lines = []
    for rel in rels:
        path = family_dir / rel
        if not path.exists():
            raise RuntimeError(f"cannot manifest missing file: {path}")
        lines.append(f"{_sha256_of_file(path)}  {rel}")
    manifest_path = family_dir / MANIFEST_NAME
    manifest_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return manifest_path


def verify_manifest(family_dir: str | Path) -> dict:
    """Verify every file listed in ``family_dir/MANIFEST.sha256``.

    Returns ``{n_files, sha256_of_manifest, verified_utc}`` on success. Raises
    ``RuntimeError`` (listing missing and corrupt entries) otherwise.
    """
    family_dir = Path(family_dir)
    manifest_path = family_dir / MANIFEST_NAME
    if not manifest_path.exists():
        raise RuntimeError(f"missing manifest: {manifest_path}")
    text = manifest_path.read_text(encoding="utf-8")
    entries = parse_manifest(text)
    if not entries:
        raise RuntimeError(f"empty manifest (no file entries): {manifest_path}")

    missing: list[str] = []
    corrupt: list[str] = []
    for sha, rel in entries:
        path = family_dir / rel
        if not path.exists():
            missing.append(rel)
            continue
        if _sha256_of_file(path) != sha:
            corrupt.append(rel)
    if missing or corrupt:
        raise RuntimeError(
            f"manifest verification failed for {family_dir}: "
            f"missing={missing} corrupt={corrupt}")

    return {
        "n_files": len(entries),
        "sha256_of_manifest": _sha256_of_text(text),
        "verified_utc": datetime.now(timezone.utc).isoformat(),
    }


def library_root(cfg, *, project_root: str | Path | None = None) -> Path:
    """Resolve ``g3_libraries_root`` (D14) to an absolute path.

    ``cfg`` is the run config dict. The value is relative to the project root
    (a sibling of the repo, e.g. ``../Data/external_libraries``). Raises
    ``RuntimeError`` if the key is absent or the resolved directory is missing.
    """
    if "g3_libraries_root" not in cfg:
        raise RuntimeError(
            "config is missing 'g3_libraries_root' (frozen decision D14).")
    root = project_root_path(project_root) / str(cfg["g3_libraries_root"])
    root = root.resolve()
    if not root.is_dir():
        raise RuntimeError(
            f"external libraries root does not exist: {root} "
            "(run scripts/fetch_g3_libraries.py to populate it).")
    return root


PROVENANCE_NAME = "PROVENANCE.json"


def library_provenance(cfg, family: str = "bt-settl", *,
                       project_root: str | Path | None = None) -> dict | None:
    """Read a family's ``PROVENANCE.json``, or ``None`` if it is not on disk.

    The fetch script writes it next to the manifest, so it is what the library
    says about itself: grid, axes, citation and whether the download was
    complete. A declared ``g3_atmosphere_family`` is a config string and can go
    stale -- both runs carried ``[deferred, pending data]`` for seven weeks
    after the library was fetched complete on 2026-07-17. Never returns a
    partial read: a malformed file raises rather than passing as absent.
    """
    import json

    try:
        root = library_root(cfg, project_root=project_root)
    except RuntimeError:
        return None
    path = root / LIBRARY_SUBDIRS.get(family, family) / PROVENANCE_NAME
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def library_is_complete(prov: dict | None) -> bool:
    """True when the provenance says the download finished with no gaps."""
    if not prov:
        return False
    return prov.get("partial") is False and not prov.get("n_failed")


__all__ = [
    "LIBRARY_SUBDIRS",
    "PROVENANCE_NAME",
    "MANIFEST_NAME",
    "library_is_complete",
    "library_provenance",
    "library_root",
    "parse_manifest",
    "verify_manifest",
    "write_manifest",
]
