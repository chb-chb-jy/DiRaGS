"""Train the complete DiRaGS model."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

if __name__ == "__main__":
    from dirags.cli import main

    raise SystemExit(main())
