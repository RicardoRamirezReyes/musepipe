#!/usr/bin/env python
"""Build the G5 characterization package (extends the F1 report)."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from musepipe.characterization import build_characterization


def main(argv=None):
    ap = argparse.ArgumentParser(prog="build_characterization.py")
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--project-root", default=None)
    args = ap.parse_args(argv)
    summary = build_characterization(args.run_id, project_root=args.project_root)
    out = Path(summary["run_id"])
    print(f"characterization: class={summary['final_class']} "
          f"traceable={summary['traceability']['complete']} "
          f"tables={len(summary['tables_generated'])} issues={len(summary['open_issues'])}")


if __name__ == "__main__":
    main()
