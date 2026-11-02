"""Verify external definition loading without constructing a model or training."""
import argparse
from pathlib import Path
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backbone", choices=('main', 'backbone_1', 'backbone_2'), required=True)
    args = parser.parse_args()
    if args.backbone == 'main':
        from dirags.integrations.segmentation import upstream
        upstream(False)
    elif args.backbone == 'backbone_1':
        from dirags.integrations.backbone_1 import modules
        modules()
    else:
        from dirags.integrations.backbone_2 import module, DiceBCELoss
        module(); DiceBCELoss()
    print(f"PASS: {args.backbone} external source identity and required definitions")


if __name__ == "__main__":
    main()
