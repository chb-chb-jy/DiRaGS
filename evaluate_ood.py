"""Evaluate a source-selected A-E checkpoint on the frozen four-domain list."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

if __name__ == "__main__":
    from dirags.ood_cli import main

    raise SystemExit(main())
