#!/usr/bin/env python3
"""Build a manual MUSE OFFSET_LIST using only the brightest central source."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from astropy.io import fits
from astropy.table import Table

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from musepipe.reduction.esorex_driver import sha256_file  # noqa: E402


class PrimaryOffsetError(RuntimeError):
    """Raised when source lists cannot define a primary-only offset table."""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Create a primary-only manual OFFSET_LIST without executing EsoRex.")
    parser.add_argument("--alignment-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    alignment_dir = Path(args.alignment_dir)
    automatic = alignment_dir / "OFFSET_LIST.fits"
    source_lists = sorted(alignment_dir.glob("SOURCE_LIST_*.fits"))
    offsets = Table.read(automatic)
    if len(offsets) < 2 or len(source_lists) != len(offsets):
        raise PrimaryOffsetError("OFFSET_LIST and SOURCE_LIST count mismatch")
    primaries: list[tuple[float, float, float]] = []
    for source_list in source_lists:
        sources = Table.read(source_list)
        for column in ("Flux", "RA", "DEC"):
            if column not in sources.colnames:
                raise PrimaryOffsetError(f"{source_list}: missing {column}")
        if len(sources) == 0:
            raise PrimaryOffsetError(f"{source_list}: no detected sources")
        primary = sources[sources["Flux"].argmax()]
        primaries.append((float(primary["RA"]), float(primary["DEC"]), float(primary["Flux"])))
    reference_ra, reference_dec, _ = primaries[0]
    output = Path(args.output)
    if output.exists():
        raise PrimaryOffsetError(f"refusing to overwrite {output}")
    ra_offsets = [ra - reference_ra for ra, _, _ in primaries]
    dec_offsets = [dec - reference_dec for _, dec, _ in primaries]
    with fits.open(automatic) as hdul:
        table_hdu = next((hdu for hdu in hdul if isinstance(hdu, fits.BinTableHDU)), None)
        if table_hdu is None:
            raise PrimaryOffsetError(f"{automatic}: no binary table")
        table_hdu.data["RA_OFFSET"] = ra_offsets
        table_hdu.data["DEC_OFFSET"] = dec_offsets
        table_hdu.data["FLUX_SCALE"] = 0.0
        table_hdu.header["OFFMETHOD"] = "PRIMARY"
        table_hdu.header["REFRA"] = reference_ra
        table_hdu.header["REFDEC"] = reference_dec
        hdul.writeto(output)
    report = {
        "status": "awaiting_manual_offset_approval",
        "method": "brightest_source_per_SOURCE_LIST",
        "reference_ra_deg": reference_ra,
        "reference_dec_deg": reference_dec,
        "automatic_offset_list": str(automatic),
        "manual_offset_list": str(output),
        "manual_offset_list_sha256": sha256_file(output),
        "source_lists": [str(path) for path in source_lists],
        "primary_fluxes": [flux for _, _, flux in primaries],
    }
    report_path = output.with_suffix(".json")
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Primary-only OFFSET_LIST complete: {len(offsets)} exposures -> {output}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, PrimaryOffsetError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
