'Train BackboneOne baseline or PDC using a frozen cross-backbone config.'
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
if __name__ == "__main__":
    from dirags.cross_backbone.backbone_1_cli import main
    raise SystemExit(main())
