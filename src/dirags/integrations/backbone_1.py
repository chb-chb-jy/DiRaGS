"""External interface for cross-architecture experiment group one."""
from functools import lru_cache
import math
import os
from pathlib import Path
import torch
from torch import nn
import torch.nn.functional as F
from .source import definitions,source_path
from .registry import profile


@lru_cache(maxsize=4)
def _modules(root,loss_path):
    spec=profile("backbone_1")
    root=Path(root)
    namespace=dict(torch=torch,nn=nn,F=F,math=math)
    encoder=definitions(root/spec["encoder_file"],set(spec["encoder_definitions"]),namespace,expected=spec["encoder_sha256"])
    model_namespace=dict(namespace)
    model_namespace[spec["encoder_factory"]]=getattr(encoder,spec["encoder_factory"])
    network=definitions(root/spec["model_file"],set(spec["model_definitions"]),model_namespace,expected=spec["model_sha256"])
    loss=definitions(loss_path,set(spec["loss_definitions"]),namespace,expected=spec["loss_sha256"])
    return network,getattr(loss,spec["loss_factory"])


def modules():
    spec=profile("backbone_1")
    model_path=source_path("DIRAGS_BACKBONE_1_ROOT",spec["model_file"])
    root=model_path.parents[1]
    loss=source_path("DIRAGS_BACKBONE_1_TRAIN_SOURCE") if os.environ.get("DIRAGS_BACKBONE_1_TRAIN_SOURCE") else root/spec["loss_file"]
    return _modules(str(root),str(loss))


def structure_loss(logits,mask):
    if logits.shape!=mask.shape:
        raise ValueError("group-one output/mask shape mismatch")
    return modules()[1](logits,mask)
