"""Read external-source and historical identity mappings from provenance data."""
from functools import lru_cache
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


@lru_cache(maxsize=2)
def _read(name):
    return json.loads((ROOT / "provenance" / name).read_text(encoding="utf-8"))


def profile(name):
    return _read("upstream_sources.json")[name]


def legacy_text(key):
    return _read("compatibility.json")[key]


def resolve(obj, path):
    for part in path.split("."):
        obj = getattr(obj, part)
    return obj


def evidence_axes(model):
    """Return the two evidence branches without registering module aliases."""
    ports = getattr(model, "evidence_blocks", None)
    if ports is not None:
        return {"height": ports.height_block, "width": ports.width_block}
    return {axis:resolve(model,path) for axis,path in profile("main")["evidence_axes"].items()}


def evidence_stages(block):
    return {stage:resolve(block,path) for stage,path in profile("main")["evidence_stages"].items()}
