"""Generic segmentation interface with external definitions and unchanged states."""
from dataclasses import dataclass
import importlib
import math
from types import MethodType
from typing import Tuple, Optional
import torch
from torch import nn
import torch.nn.functional as F
import torch.utils.checkpoint
from einops import rearrange, repeat
from .source import definitions, source_path
from .registry import profile


@dataclass
class ModelConfig:
    pretrained_encoder: bool = True


@dataclass
class ModelOutput:
    logits: torch.Tensor
    deep_logits: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    pred_x: Optional[torch.Tensor] = None
    pred_y: Optional[torch.Tensor] = None
    alpha_h: Optional[torch.Tensor] = None
    alpha_w: Optional[torch.Tensor] = None
    occupancy_logits_h: Optional[torch.Tensor] = None
    occupancy_logits_w: Optional[torch.Tensor] = None
    support_h: Optional[torch.Tensor] = None
    support_w: Optional[torch.Tensor] = None


def upstream(pretrained=False):
    spec=profile("main")
    path=source_path("DIRAGS_MAIN_SOURCE")
    module_name, function_name=spec["encoder_factory_import"].split(":")
    factory=getattr(importlib.import_module(module_name),function_name)
    def encoder(**kwargs):
        kwargs["pretrained"]=pretrained
        return factory(**kwargs)
    namespace=dict(torch=torch,nn=nn,F=F,math=math,rearrange=rearrange,repeat=repeat)
    namespace[spec["encoder_factory_global"]]=encoder
    return definitions(path,set(spec["definitions"]),namespace,expected=spec["sha256"])


def build_segmentation(config=None,in_channels=3):
    config=config or ModelConfig()
    spec=profile("main")
    module=upstream(config.pretrained_encoder)
    model=getattr(module,spec["model_factory"])(getattr(module,spec["settings_factory"])(),in_channels=in_channels)
    original_forward=model.forward
    def forward(self,image,return_aux=False):
        result=original_forward(image,return_deep=return_aux)
        return ModelOutput(result[0],tuple(result[1:])) if return_aux else result
    model.forward=MethodType(forward,model)
    model.config=config
    model.occupancy_parameters=lambda: ()
    return model


def ComboLoss(ce_weight=.2,dice_weight=.3,focal_weight=.2,tversky_weight=.2,
              iou_weight=.3,boundary_weight=.1,focal_gamma=2.,tversky_alpha=.5,tversky_beta=.5):
    result=getattr(upstream(),profile("main")["loss_factory"])(bce_weight=ce_weight,
        dice_weight=dice_weight,focal_weight=focal_weight,tversky_weight=tversky_weight,
        iou_weight=iou_weight,boundary_weight=boundary_weight,focal_gamma=focal_gamma,
        tversky_alpha=tversky_alpha,tversky_beta=tversky_beta)
    result.ce_weight=ce_weight
    return result


class DeepSupervisionLoss(nn.Module):
    """Weight four outputs; component calculations remain external."""
    def __init__(self,weights=(.6,.2,.1,.1)):
        super().__init__()
        if len(weights)!=4 or not math.isclose(sum(weights),1.):
            raise ValueError("deep supervision requires four weights summing to one")
        self.weights=tuple(map(float,weights))
        self.criterion=ComboLoss()
    def forward(self,outputs,target):
        logits=(outputs.logits,*outputs.deep_logits) if hasattr(outputs,"logits") else tuple(outputs)
        if len(logits)!=4:
            raise ValueError("expected main, ds2, ds3, ds4 logits")
        return sum(weight*self.criterion(prediction,target) for weight,prediction in zip(self.weights,logits))
