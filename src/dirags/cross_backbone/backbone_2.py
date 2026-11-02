"BackboneTwo's original batch Dice+BCE loss and sequential PDC training step."
from pathlib import Path
from unittest import mock
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..methods.pdc_common import PseudoDomainConfig
from ..methods.pdc import build_paired_raw_replay
from ..training.dual_view import (_capture_torch_rng, _restore_torch_rng,
                                _capture_bn_buffers, _restore_bn_buffers)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load_checkpoint(path):
    try:
        return torch.load(Path(path), map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(Path(path), map_location="cpu")


def build_model(pretrained=None):
    from ..integrations.backbone_2 import module as external_module
    module = external_module()
    state = load_checkpoint(pretrained) if pretrained is not None else {}
    if pretrained is not None:
        # The formal qualifier constructs a reference EncoderTwo before constructing
        # BackboneTwo. This consumes RNG and therefore affects decoder initialization.
        # Retain that construction; restoring RNG here would change training.
        reference = module.reference_encoder_factory().state_dict()
        if not all(k in state and tuple(state[k].shape) == tuple(v.shape)
                   and bool(torch.isfinite(state[k]).all()) for k, v in reference.items()):
            raise ValueError('pretrained state does not cover the complete reference EncoderTwo')
        del reference
    # Constructor I/O adapter only: the original network loads a relative path.
    # Full selected states restore every model tensor strictly at inference.
    with mock.patch.object(module.torch, "load", return_value=state):
        model = module.model_factory().cpu()
    if pretrained is not None:
        reference = model.backbone.state_dict()
        if not all(k in state and tuple(state[k].shape) == tuple(v.shape)
                   and torch.equal(state[k].cpu(), v.cpu()) for k, v in reference.items()):
            raise ValueError('ImageNet state does not cover the full BackboneTwo backbone')
    return model, DiceBCELoss()


from ..integrations.backbone_2 import DiceBCELoss


@torch.no_grad()
def build_pseudo(images, masks, names, *, training_seed=42, epoch=0):
    if images.ndim != 4 or images.shape[1] != 3 or masks.shape != (len(images), 1, *images.shape[-2:]):
        raise ValueError('expected BGR [B,3,H,W] and masks [B,1,H,W]')
    if len(names) != len(images) or len(set(map(str, names))) != len(names):
        raise ValueError('one unique stable source filename per image is required')
    for value in (images, masks):
        if not bool(torch.isfinite(value).all()) or bool(((value < 0) | (value > 1)).any()):
            raise ValueError('BackboneTwo images and soft targets must be finite in [0,1]')
    rgb = images[:, [2, 1, 0]]
    # This hard mask is used ONLY for PDC region construction. The original
    # linearly resized soft target supervises both views without alteration.
    support = masks.detach().ge(.5)
    cfg = PseudoDomainConfig()
    result = torch.cat([build_paired_raw_replay(rgb[i:i+1], support[i, 0].cpu(),
        training_seed=training_seed, epoch=epoch, sample_identity=str(name), config=cfg).d1
        for i, name in enumerate(names)])
    return result[:, [2, 1, 0]].contiguous()

def train_pdc_minibatch(model, optimizer, loss_fn, images, masks, names, *, epoch):
    if not model.training:
        raise RuntimeError('paired training requires model.train()')
    if isinstance(epoch, bool) or not isinstance(epoch, int) or not 1 <= epoch <= 300:
        raise ValueError('epoch must be a one-based integer in 1..300')
    pseudo = build_pseudo(images, masks, names, epoch=epoch-1)
    optimizer.zero_grad(set_to_none=True)
    state0 = _capture_torch_rng()
    clean = model(images)
    if clean.shape != masks.shape or not bool(torch.isfinite(clean).all()):
        raise ValueError('invalid clean output')
    clean_loss = loss_fn(clean, masks)
    if not bool(torch.isfinite(clean_loss)):
        raise FloatingPointError('nonfinite clean loss')
    state1, clean_bn = _capture_torch_rng(), _capture_bn_buffers(model)
    try:
        # Two backwards, one optimizer step: same 0.5*(Lclean+Lpseudo)
        # objective without retaining two full EncoderTwo-B3 autograd graphs.
        (.5 * clean_loss).backward()
        clean_value = float(clean_loss.detach())
        del clean, clean_loss
        _restore_torch_rng(state0)
        predicted = model(pseudo)
        if predicted.shape != masks.shape or not bool(torch.isfinite(predicted).all()):
            raise ValueError('invalid pseudo output')
        pseudo_loss = loss_fn(predicted, masks)
        if not bool(torch.isfinite(pseudo_loss)):
            raise FloatingPointError('nonfinite pseudo loss')
        (.5 * pseudo_loss).backward()
        pseudo_value = float(pseudo_loss.detach())
    finally:
        _restore_torch_rng(state1)
        _restore_bn_buffers(model, clean_bn)
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    if not gradients or not all(bool(torch.isfinite(g).all()) for g in gradients):
        raise FloatingPointError('missing/nonfinite paired gradients')
    # Original BackboneTwo runner uses neither clipping nor AMP.
    optimizer.step()
    return {'loss': .5*(clean_value+pseudo_value), 'clean_loss': clean_value,
            'pseudo_loss': pseudo_value, 'optimizer_steps': 1}

def per_sample_iou_dice(logits, target):
    import torch
    require(logits.shape == target.shape, "logit/target shape mismatch")
    prediction = torch.sigmoid(logits) >= 0.5
    truth = target >= 0.5
    dims = tuple(range(1, prediction.ndim))
    intersection = (prediction & truth).sum(dim=dims).float()
    union = (prediction | truth).sum(dim=dims).float()
    area = prediction.sum(dim=dims).float() + truth.sum(dim=dims).float()
    iou = torch.where(union > 0, intersection / union, torch.ones_like(union))
    dice = torch.where(area > 0, 2 * intersection / area, torch.ones_like(area))
    return iou, dice

def evaluate(model, loader, loss_fn, device, max_batches=None):
    model.eval()
    loss_sum = iou_sum = dice_sum = 0.0
    samples = 0
    with torch.inference_mode():
        for batch, (images, masks) in enumerate(loader, 1):
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            logits = model(images)
            require(logits.shape == masks.shape and bool(torch.isfinite(logits).all()),
                    "invalid source evaluation output")
            loss = loss_fn(logits, masks)
            iou, dice = per_sample_iou_dice(logits, masks)
            require(bool(torch.isfinite(loss)) and bool(torch.isfinite(iou).all()) and
                    bool(torch.isfinite(dice).all()), "non-finite source evaluation")
            samples += len(images)
            loss_sum += float(loss) * len(images)
            iou_sum += float(iou.sum())
            dice_sum += float(dice.sum())
            if max_batches is not None and batch >= max_batches:
                break
    require(samples > 0, "empty source evaluation loader")
    return {
        "loss": loss_sum / samples, "iou": iou_sum / samples,
        "dice": dice_sum / samples, "samples": samples,
    }

def train_baseline_minibatch(model, optimizer, loss_fn, images, masks):
    optimizer.zero_grad(set_to_none=True)
    logits = model(images)
    require(logits.shape == masks.shape and bool(torch.isfinite(logits).all()), "invalid training output")
    loss = loss_fn(logits, masks)
    require(bool(torch.isfinite(loss)), "non-finite training loss")
    loss.backward()
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    require(bool(gradients) and all(bool(torch.isfinite(g).all()) for g in gradients), "missing or non-finite gradients")
    optimizer.step()
    return float(loss.detach())
