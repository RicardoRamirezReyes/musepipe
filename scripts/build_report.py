#!/usr/bin/env python3
"""Build the deterministic final report for a MUSE run."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


def _project_root_from_script():
    return Path(__file__).resolve().parents[1]


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="build_report.py",
        description="Build runs/<RUN_ID>/report from existing QC and products.",
    )
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--project-root", default=None)
    parser.add_argument("--allow-run-id-mismatch", action="store_true")
    args = parser.parse_args(argv)
    project_root = Path(args.project_root).resolve() if args.project_root else _project_root_from_script()
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from musepipe.report import build_report

    result = build_report(
        args.run_id,
        project_root=project_root,
        allow_run_id_mismatch=args.allow_run_id_mismatch,
    )
    print(result["paths"].report_md)


if __name__ == "__main__":
    main()
