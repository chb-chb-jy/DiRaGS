"""BackboneOne losses and training steps from the verified seed42 formal implementation.

PDC reuses the public sampler. No DST/GCS runs in this experiment group.
The deprecated BCE reduce argument intentionally preserves actual formal loss.
"""
from __future__ import annotations
from contextlib import contextmanager
from pathlib import Path
from unittest import mock
import torch
import torch.nn.functional as F
from ..methods.pdc_common import PseudoDomainConfig
from ..methods.pdc_integration import build_pdc_pseudo_domain
from ..training.dual_view import (_capture_torch_rng, _restore_torch_rng,
                                _capture_bn_buffers, _restore_bn_buffers)

TRAIN_SIZES = (192, 256, 320)
IMAGE_SIZE = 256


def build_backbone_1(pretrained=None):
    """Construct without implicit download; optionally supply ImageNet state."""
    from ..integrations.backbone_1 import modules
    module, _ = modules()
    from ..integrations.registry import profile
    spec = profile("backbone_1")
    factory = getattr(module, spec["encoder_factory"])
    state = None
    if pretrained is not None:
        state = load_checkpoint(pretrained)
    def backbone(pretrained=True, **kwargs):
        result = factory(pretrained=False, **kwargs)
        if state is not None:
            result.load_state_dict(state, strict=True)
        return result
    with mock.patch.object(module, spec["encoder_factory"], backbone):
        return getattr(module, spec["model_factory"])(channel=32).cpu()


def load_checkpoint(path):
    try:
        return torch.load(Path(path), map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(Path(path), map_location="cpu")


@torch.no_grad()
def build_pseudo(images, masks, names, *, training_seed, epoch):
    if masks.shape != (images.shape[0], 1, *images.shape[-2:]):
        raise ValueError('BackboneOne masks must be [B,1,H,W]')
    if not torch.isfinite(images).all().item() or not torch.isfinite(masks).all().item():
        raise ValueError("nonfinite PDC input")
    if not ((masks == 0) | (masks == 1)).all().item():
        raise ValueError("PDC requires hard base-grid masks")
    return build_pdc_pseudo_domain(images, masks[:, 0], names, training_seed=training_seed,
                                  epoch=epoch, config=PseudoDomainConfig())


from ..integrations.backbone_1 import structure_loss


def supervised_loss(outputs, mask: torch.Tensor) -> torch.Tensor:
    if not isinstance(outputs, (tuple, list)) or len(outputs) != 4:
        raise ValueError('BackboneOne must return four supervised lateral maps')
    return sum(structure_loss(output, mask) for output in outputs)

def resize_training_batch(images, masks, size: int):
    if size not in TRAIN_SIZES:
        raise ValueError('unexpected BackboneOne multi-scale size')
    if size == IMAGE_SIZE:
        return images, masks
    # Preserve official multi-scale interpolation, including soft resized masks.
    return (
        F.interpolate(images, size=(size, size), mode="bilinear", align_corners=True),
        F.interpolate(masks, size=(size, size), mode="bilinear", align_corners=True),
    )

def clip_official_gradients(optimizer, margin: float = 0.5) -> None:
    # Official elementwise clamp, NOT a global norm cap.
    for group in optimizer.param_groups:
        for parameter in group["params"]:
            if parameter.grad is not None:
                parameter.grad.clamp_(-margin, margin)

def inference_logits(outputs) -> torch.Tensor:
    if not isinstance(outputs, (tuple, list)) or len(outputs) != 4:
        raise ValueError('BackboneOne must return four lateral maps')
    return outputs[-1]  # Official MyTest.py uses lateral_map_2.

@contextmanager
def paired_forward(model, clean, pseudo):
    """Replay clean RNG for pseudo; restore clean RNG/BN after paired backward."""
    if not model.training:
        raise RuntimeError('paired_forward requires training mode')
    state0 = _capture_torch_rng()
    clean_outputs = model(clean)
    state1 = _capture_torch_rng()
    clean_bn = _capture_bn_buffers(model)
    _restore_torch_rng(state0)
    try:
        pseudo_outputs = model(pseudo)
        _restore_torch_rng(state1)
        # Buffer restoration is deliberately after backward, not between two
        # live autograd graphs: copy_ would invalidate BN version counters.
        yield clean_outputs, pseudo_outputs
    finally:
        _restore_torch_rng(state1)
        _restore_bn_buffers(model, clean_bn)

def train_pdc_minibatch(model, optimizer, images, masks, names, config, *, epoch):
    if isinstance(epoch, bool) or not isinstance(epoch, int) or not 1 <= epoch <= config['epochs']:
        raise ValueError('epoch must be a one-based integer within the frozen budget')
    pseudo = build_pseudo(images, masks, names,
        training_seed=config['training_seed'], epoch=epoch-1)
    total, clean_total, pseudo_total = [], [], []
    for size in config['training_sizes']:
        optimizer.zero_grad(set_to_none=True)
        x, y = resize_training_batch(images, masks, size)
        xp, yp = resize_training_batch(pseudo, masks, size)
        if not torch.equal(y, yp):
            raise RuntimeError('paired supervision masks diverged')
        with paired_forward(model, x, xp) as (clean_out, pseudo_out):
            clean_loss = supervised_loss(clean_out, y)
            pseudo_loss = supervised_loss(pseudo_out, y)
            loss = .5 * (clean_loss + pseudo_loss)
            if not torch.isfinite(loss).item():
                raise FloatingPointError('nonfinite paired loss')
            loss.backward()
        gradients = [p.grad for p in model.parameters() if p.grad is not None]
        if not gradients or not all(torch.isfinite(g).all().item() for g in gradients):
            raise FloatingPointError('missing/nonfinite BackboneOne+PDC gradient')
        clip_official_gradients(optimizer, config['gradient_clip']['margin'])
        optimizer.step()
        total.append(loss.item())
        clean_total.append(clean_loss.item())
        pseudo_total.append(pseudo_loss.item())
    return {'loss':sum(total)/len(total), 'clean_loss':sum(clean_total)/len(total),
            'pseudo_loss':sum(pseudo_total)/len(total), 'optimizer_steps':len(total)}

def train_baseline_minibatch(model, optimizer, images, masks, config: dict) -> float:
    losses = []
    for size in config["training_sizes"]:
        optimizer.zero_grad(set_to_none=True)
        scaled_images, scaled_masks = resize_training_batch(images, masks, size)
        outputs = model(scaled_images)
        loss = supervised_loss(outputs, scaled_masks)
        if not torch.isfinite(loss).item():
            raise FloatingPointError('non-finite BackboneOne training loss')
        loss.backward()
        gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
        if not gradients or not torch.stack([torch.isfinite(gradient).all() for gradient in gradients]).all().item():
            raise FloatingPointError('missing or non-finite BackboneOne gradients')
        clip_official_gradients(optimizer, config["gradient_clip"]["margin"])
        optimizer.step()
        losses.append(loss.item())
        del scaled_images, scaled_masks, outputs, loss, gradients
    return sum(losses) / len(losses)

@torch.no_grad()
def evaluate(model, loader, device, max_batches: int | None = None) -> dict:
    model.eval()
    total_loss, total_iou, total_dice, samples = 0.0, 0.0, 0.0, 0
    for index, (images, masks) in enumerate(loader, start=1):
        images, masks = images.to(device), masks.to(device)
        outputs = model(images)
        loss = supervised_loss(outputs, masks)
        logits = inference_logits(outputs)
        if not torch.isfinite(loss).item() or not torch.isfinite(logits).all().item():
            raise FloatingPointError('non-finite BackboneOne validation output or loss')
        prediction, truth = torch.sigmoid(logits) >= 0.5, masks >= 0.5
        dims = tuple(range(1, prediction.ndim))
        intersection = (prediction & truth).sum(dim=dims).float()
        union = (prediction | truth).sum(dim=dims).float()
        area = prediction.sum(dim=dims).float() + truth.sum(dim=dims).float()
        iou = torch.where(union > 0, intersection / union, torch.ones_like(union))
        dice = torch.where(area > 0, 2 * intersection / area, torch.ones_like(area))
        samples += len(images)
        total_loss += loss.item() * len(images)
        total_iou += iou.sum().item()
        total_dice += dice.sum().item()
        if max_batches is not None and index >= max_batches:
            break
    if samples == 0:
        raise ValueError("empty source evaluation loader")
    return {"loss": total_loss / samples, "iou": total_iou / samples,
            "dice": total_dice / samples, "samples": samples}

def learning_rate_for_epoch(config: dict, epoch: int) -> float:
    if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 1:
        raise ValueError("epoch must be a positive one-based integer")
    scheduler = config["scheduler"]
    return config["optimizer"]["learning_rate"] * scheduler["decay_rate"] ** (
        epoch // scheduler["decay_epoch"]
    )

