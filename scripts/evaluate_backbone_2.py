'BackboneTwo OOD4 evaluation with canonical PIL GT and original BGR inputs.'
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
if __name__ == "__main__":
    from dirags.cross_backbone.backbone_2_ood import main
    raise SystemExit(main())
