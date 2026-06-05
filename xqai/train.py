"""Training loop skeleton for xqai (INTERFACES.md §5, §9).

Two modes:

- ``supervised``: train PVNet on a fixed dataset of ``(planes, pi, z)`` triples
  (e.g. from human game records or a teacher), using :func:`az_loss`.
- ``rl``: AlphaZero-style — sample minibatches from a :class:`ReplayBuffer`
  filled by self-play workers, optimize with :func:`az_loss`.

Common machinery: AdamW or SGD, BF16 autocast (configurable), cosine or step LR
schedule, periodic checkpointing to ``paths.checkpoints``. Everything is sized
from ``configs/default.yaml`` but can be overridden; the whole thing runs at a
tiny scale on CPU for smoke tests.
"""

from __future__ import annotations

import math
import os
import time
from typing import Iterable, Iterator

import numpy as np
import torch
import torch.nn as nn

from .config import load_config
from .network import PVNet, az_loss


# --------------------------------------------------------------------------- #
# Optimizer / schedule                                                        #
# --------------------------------------------------------------------------- #
def build_optimizer(model: nn.Module, *, kind: str, lr: float, weight_decay: float):
    if kind.lower() == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    if kind.lower() == "sgd":
        return torch.optim.SGD(
            model.parameters(), lr=lr, momentum=0.9, nesterov=True, weight_decay=weight_decay
        )
    raise ValueError(f"unknown optimizer kind {kind!r}")


def lr_at(step: int, total_steps: int, base_lr: float, *, schedule: str, min_lr: float) -> float:
    """Cosine or step LR. ``step`` is 0-based."""
    if schedule == "cosine":
        t = min(step / max(total_steps, 1), 1.0)
        return min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * t))
    if schedule == "step":
        # Decay by 10x at 1/2 and 3/4 of training.
        if step >= 0.75 * total_steps:
            return base_lr * 0.01
        if step >= 0.5 * total_steps:
            return base_lr * 0.1
        return base_lr
    if schedule == "const":
        return base_lr
    raise ValueError(f"unknown lr schedule {schedule!r}")


def _amp_dtype(precision: str):
    p = (precision or "fp32").lower()
    if p in ("bf16", "bfloat16"):
        return torch.bfloat16
    if p in ("fp16", "float16", "half"):
        return torch.float16
    return None  # fp32 / disabled


# --------------------------------------------------------------------------- #
# Checkpointing                                                               #
# --------------------------------------------------------------------------- #
def save_checkpoint(model: nn.Module, optimizer, step: int, ckpt_dir: str, tag: str = "") -> str:
    os.makedirs(ckpt_dir, exist_ok=True)
    name = f"ckpt_{tag + '_' if tag else ''}{step:08d}.pt"
    path = os.path.join(ckpt_dir, name)
    torch.save(
        {
            "step": step,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict() if optimizer is not None else None,
            "channels": getattr(model, "channels", None),
            "blocks": getattr(model, "blocks", None),
        },
        path,
    )
    return path


# --------------------------------------------------------------------------- #
# One optimization step                                                       #
# --------------------------------------------------------------------------- #
def _train_step(model, optimizer, batch, *, device, amp_dtype, weight_decay) -> float:
    planes, pi, z, mask = batch
    planes = planes.to(device, non_blocking=True)
    pi = pi.to(device, non_blocking=True)
    z = z.to(device, non_blocking=True)
    mask = mask.to(device, non_blocking=True)

    optimizer.zero_grad(set_to_none=True)
    use_amp = amp_dtype is not None
    ctx = (
        torch.autocast(device_type=device.type, dtype=amp_dtype)
        if use_amp
        else _nullcontext()
    )
    with ctx:
        policy_logits, value = model(planes)
        # L2 handled by the optimizer's weight_decay, so pass model=None / c=0
        # to az_loss to avoid double-counting.
        loss = az_loss(policy_logits, value, pi, z, mask, c=0.0, model=None)
    loss.backward()
    optimizer.step()
    return float(loss.detach().cpu())


class _nullcontext:
    def __enter__(self):
        return None

    def __exit__(self, *a):
        return False


# --------------------------------------------------------------------------- #
# Supervised mode                                                             #
# --------------------------------------------------------------------------- #
def train_supervised(
    dataset: Iterable,
    *,
    steps: int,
    cfg=None,
    model: nn.Module | None = None,
    device: str | torch.device | None = None,
) -> nn.Module:
    """Train on an iterable of ``(planes, pi, z, mask)`` torch-tensor batches.

    ``dataset`` may be a DataLoader-like iterable yielding ready batches, or any
    iterator; it is cycled if it is exhausted before ``steps``.
    """
    cfg = cfg or load_config()
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    if model is None:
        model = PVNet(cfg.network.channels, cfg.network.blocks)
    model = model.to(device)
    model.train()

    optimizer = build_optimizer(
        model, kind="adamw", lr=cfg.train.lr_supervised, weight_decay=cfg.train.weight_decay
    )
    amp_dtype = _amp_dtype(cfg.train.precision) if device.type == "cuda" else None
    ckpt_dir = cfg.paths.checkpoints
    export_every = int(cfg.train.export_every)

    it = _cycle(dataset)
    for step in range(steps):
        for g in optimizer.param_groups:
            g["lr"] = lr_at(step, steps, cfg.train.lr_supervised, schedule="cosine", min_lr=1e-4)
        batch = next(it)
        loss = _train_step(
            model, optimizer, batch, device=device, amp_dtype=amp_dtype,
            weight_decay=cfg.train.weight_decay,
        )
        if export_every and (step + 1) % export_every == 0:
            save_checkpoint(model, optimizer, step + 1, ckpt_dir, tag="sup")
    return model


# --------------------------------------------------------------------------- #
# RL mode                                                                     #
# --------------------------------------------------------------------------- #
def train_rl(
    replay,
    *,
    steps: int,
    cfg=None,
    model: nn.Module | None = None,
    device: str | torch.device | None = None,
    min_buffer: int = 1,
) -> nn.Module:
    """Sample from a :class:`ReplayBuffer` and optimize PVNet with az_loss."""
    cfg = cfg or load_config()
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    if model is None:
        model = PVNet(cfg.network.channels, cfg.network.blocks)
    model = model.to(device)
    model.train()

    optimizer = build_optimizer(
        model, kind="sgd", lr=cfg.train.lr_rl, weight_decay=cfg.train.weight_decay
    )
    amp_dtype = _amp_dtype(cfg.train.precision) if device.type == "cuda" else None
    ckpt_dir = cfg.paths.checkpoints
    export_every = int(cfg.train.export_every)
    batch_size = int(cfg.train.batch_size)

    for step in range(steps):
        # Wait for the buffer to fill enough (in a real run this loops/sleeps).
        while len(replay) < min_buffer:
            time.sleep(0.05)
        for g in optimizer.param_groups:
            g["lr"] = lr_at(step, steps, cfg.train.lr_rl, schedule="cosine", min_lr=1e-4)
        batch = replay.sample(batch_size)
        loss = _train_step(
            model, optimizer, batch, device=device, amp_dtype=amp_dtype,
            weight_decay=cfg.train.weight_decay,
        )
        if export_every and (step + 1) % export_every == 0:
            save_checkpoint(model, optimizer, step + 1, ckpt_dir, tag="rl")
    return model


def _cycle(iterable: Iterable) -> Iterator:
    while True:
        produced = False
        for item in iterable:
            produced = True
            yield item
        if not produced:
            raise RuntimeError("empty dataset passed to training loop")


__all__ = [
    "train_supervised",
    "train_rl",
    "build_optimizer",
    "lr_at",
    "save_checkpoint",
]
