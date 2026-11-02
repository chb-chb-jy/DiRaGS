'Train the independent BackboneTwo baseline or PDC cross-backbone configuration.'
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
if __name__ == "__main__":
    from dirags.cross_backbone.backbone_2_cli import main
    raise SystemExit(main())
