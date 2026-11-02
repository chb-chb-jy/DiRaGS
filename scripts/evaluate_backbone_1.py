'Evaluate BackboneOne baseline/PDC with the common frozen OOD4 metrics.'
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
if __name__ == "__main__":
    from dirags.cross_backbone.backbone_1_ood import main
    raise SystemExit(main())
