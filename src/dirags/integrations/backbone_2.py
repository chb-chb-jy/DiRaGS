"""External interface for cross-architecture experiment group two."""
from functools import lru_cache,partial
import math
from pathlib import Path
import torch
from torch import nn
import torch.nn.functional as F
from .source import definitions,source_path
from .registry import profile


@lru_cache(maxsize=4)
def _module(root):
    from timm.models.layers import DropPath,to_2tuple,trunc_normal_
    from timm.models.vision_transformer import _cfg
    spec=profile("backbone_2")
    root=Path(root)
    namespace=dict(torch=torch,nn=nn,F=F,math=math,partial=partial,DropPath=DropPath,
        to_2tuple=to_2tuple,trunc_normal_=trunc_normal_,_cfg=_cfg,register_model=lambda cls: cls)
    encoder=definitions(root/spec["encoder_file"],set(spec["encoder_definitions"]),namespace,expected=spec["encoder_sha256"])
    utility=definitions(root/spec["utility_file"],set(spec["utility_definitions"]),dict(F=F),expected=spec["utility_sha256"])
    namespace=dict(torch=torch,nn=nn,F=F)
    namespace[spec["encoder_global"]]=encoder
    namespace[spec["utility_factory"]]=getattr(utility,spec["utility_factory"])
    result=definitions(root/spec["model_file"],set(spec["model_definitions"]),namespace,expected=spec["model_sha256"])
    result.reference_encoder_factory=getattr(encoder,spec["encoder_factory"])
    result.model_factory=getattr(result,spec["model_factory"])
    return result


def module():
    return _module(str(source_path("DIRAGS_BACKBONE_2_SOURCE_DIR",profile("backbone_2")["model_file"]).parent))


@lru_cache(maxsize=4)
def _loss(path):
    spec=profile("backbone_2")
    result=definitions(path,set(spec["loss_definitions"]),dict(torch=torch,nn=nn,F=F),expected=spec["loss_sha256"])
    return getattr(result,spec["loss_factory"])


def DiceBCELoss(*args,**kwargs):
    return _loss(str(source_path("DIRAGS_BACKBONE_2_SOURCE_DIR",profile("backbone_2")["loss_file"])))(*args,**kwargs)
