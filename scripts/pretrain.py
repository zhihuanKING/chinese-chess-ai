#!/usr/bin/env python
"""Supervised pre-training for xqai PVNet (INTERFACES.md §5, task #6).

Reads ``data/processed/shard_*.npz`` samples ``(planes[15,10,9] fp16, pi_index
int32 | pi[8100] fp16, z int8)``, trains :class:`xqai.network.PVNet` with
:func:`xqai.network.az_loss` (policy cross-entropy + value MSE), AdamW, BF16
autocast and a cosine LR schedule. Single-GPU by default; multi-GPU via
``torchrun`` (DDP) when ``--gpus > 1``.

Each epoch a checkpoint is written to ``checkpoints/pretrained_*.pt`` and the
policy top-1 on a small set of diagnostic positions is logged.

This script only *reads* the existing ``xqai`` modules; it never modifies them.

Examples
--------
Single GPU::

    .venv/bin/python scripts/pretrain.py \
        --data data/processed --epochs 10 --batch 1024 \
        --channels 256 --blocks 15 --gpus 1 --out checkpoints

Multi GPU (DDP, one process per GPU)::

    .venv/bin/torchrun --standalone --nproc_per_node=4 \
        scripts/pretrain.py --data data/processed --gpus 4 ...
"""

from __future__ import annotations

import argparse
import glob
import math
import os
import sys
import time

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, DistributedSampler

# Make ``import xqai`` work no matter the cwd.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from xqai.encoding import ACTION_DIM, NUM_COLS, NUM_PLANES, NUM_ROWS, encode, legal_mask
from xqai.network import PVNet, az_loss


# --------------------------------------------------------------------------- #
# Dataset                                                                     #
# --------------------------------------------------------------------------- #
class ShardDataset(Dataset):
    """Concatenation of ``shard_*.npz`` files, memory-mapped and lazily read.

    Each shard holds ``planes [N,15,10,9] fp16`` and ``z [N] int8`` plus EITHER
    a dense ``pi [N,8100] fp16`` policy target OR a sparse ``pi_index [N] int32``
    (a single best move -> one-hot target). Both forms are supported; the
    sparse form is the one our data pipeline (``scripts/prepare_data.py`` /
    ``scripts/gen_pikafish_data.py``) emits.

    The mask passed to ``az_loss`` must mark legal moves. With sparse one-hot
    targets we don't have the full legal set stored, so the mask is derived from
    the policy itself (legal = positive prob). That is sufficient for the
    cross-entropy term (a single-move target only needs that move legal); the
    softmax denominator then spans the whole action space, which is the standard
    behaviour for a one-hot supervised target.
    """

    def __init__(self, data_dir: str):
        self.files = sorted(glob.glob(os.path.join(data_dir, "shard_*.npz")))
        if not self.files:
            raise FileNotFoundError(
                f"no shard_*.npz found under {data_dir!r}; run the data pipeline first"
            )
        # Build a flat index: per-shard length + cumulative offsets. We open
        # each shard lazily (mmap) and cache the handle to avoid re-parsing the
        # zip directory on every __getitem__.
        self._lengths: list[int] = []
        self._dense: list[bool] = []
        for f in self.files:
            with np.load(f, mmap_mode="r") as d:
                self._lengths.append(int(d["planes"].shape[0]))
                self._dense.append("pi" in d.files)
        self._cum = np.cumsum([0] + self._lengths)
        self._total = int(self._cum[-1])
        self._cache: dict[int, dict] = {}

    def __len__(self) -> int:
        return self._total

    def _shard(self, si: int) -> dict:
        d = self._cache.get(si)
        if d is None:
            npz = np.load(self.files[si], mmap_mode="r")
            d = {"planes": npz["planes"], "z": npz["z"]}
            if "pi" in npz.files:
                d["pi"] = npz["pi"]
            else:
                d["pi_index"] = npz["pi_index"]
            self._cache[si] = d
        return d

    def __getitem__(self, idx: int):
        si = int(np.searchsorted(self._cum, idx, side="right") - 1)
        off = idx - int(self._cum[si])
        d = self._shard(si)

        planes = np.asarray(d["planes"][off], dtype=np.float32)  # [15,10,9]
        z = np.float32(d["z"][off])

        pi = np.zeros(ACTION_DIM, dtype=np.float32)
        if "pi" in d:
            pi[:] = np.asarray(d["pi"][off], dtype=np.float32)
            s = pi.sum()
            if s > 0:
                pi /= s
            mask = (pi > 0.0).astype(np.float32)
        else:
            mv = int(d["pi_index"][off])
            pi[mv] = 1.0
            # Sparse target: only the target move is known legal. Mask it (plus
            # nothing else) so the CE term is well defined; softmax spans all.
            mask = np.zeros(ACTION_DIM, dtype=np.float32)
            mask[mv] = 1.0

        return (
            torch.from_numpy(planes),
            torch.from_numpy(pi),
            torch.tensor(z),
            torch.from_numpy(mask),
        )


# --------------------------------------------------------------------------- #
# Diagnostics                                                                  #
# --------------------------------------------------------------------------- #
_DIAG_FENS = [
    # Opening position.
    "rheakaehr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RHEAKAEHR w - - 0 1",
    # A common cannon opening for red.
    "rheakaehr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C2C4/9/RHEAKAEHR b - - 1 1",
]


def _diag_positions():
    """Build diagnostic ``(planes, mask)`` tensors; tolerate a missing core."""
    try:
        from xqai._xqcore import Position
    except Exception:
        return None
    planes, masks = [], []
    for fen in _DIAG_FENS:
        try:
            p = Position.from_fen(fen)
        except Exception:
            p = Position()
        planes.append(encode(p))
        masks.append(legal_mask(p))
    return (
        torch.from_numpy(np.stack(planes).astype(np.float32)),
        torch.from_numpy(np.stack(masks).astype(np.float32)),
    )


@torch.no_grad()
def diag_top1(model: nn.Module, diag, device) -> str:
    if diag is None:
        return "n/a (no _xqcore)"
    planes, masks = diag
    planes = planes.to(device)
    masks = masks.to(device)
    was_training = model.training
    model.eval()
    logits, value = model(planes)
    neg_inf = torch.finfo(logits.dtype).min
    masked = torch.where(masks > 0, logits, torch.full_like(logits, neg_inf))
    top1 = masked.argmax(dim=-1)
    if was_training:
        model.train()
    parts = []
    for i in range(top1.shape[0]):
        mv = int(top1[i])
        parts.append(f"pos{i}: move={mv}(from={mv // 90},to={mv % 90}) v={float(value[i]):+.3f}")
    return " | ".join(parts)


# --------------------------------------------------------------------------- #
# Training                                                                     #
# --------------------------------------------------------------------------- #
def _amp_dtype(precision: str):
    p = (precision or "fp32").lower()
    if p in ("bf16", "bfloat16"):
        return torch.bfloat16
    if p in ("fp16", "float16", "half"):
        return torch.float16
    return None


def _is_dist() -> bool:
    return dist.is_available() and dist.is_initialized()


def _rank() -> int:
    return dist.get_rank() if _is_dist() else 0


def _world_size() -> int:
    return dist.get_world_size() if _is_dist() else 1


def main() -> int:
    ap = argparse.ArgumentParser(description="Supervised pre-training for xqai PVNet")
    ap.add_argument("--data", default="data/processed", help="dir with shard_*.npz")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--channels", type=int, default=128)
    ap.add_argument("--blocks", type=int, default=10)
    ap.add_argument("--gpus", type=int, default=1, help="GPUs to use (DDP if >1, via torchrun)")
    ap.add_argument("--out", default="checkpoints", help="checkpoint output dir")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--precision", default="bf16", choices=["bf16", "fp16", "fp32"])
    ap.add_argument("--workers", type=int, default=4, help="DataLoader workers")
    ap.add_argument("--min-lr", type=float, default=1e-5)
    ap.add_argument("--log-every", type=int, default=50)
    args = ap.parse_args()

    # ---- DDP init (only when launched under torchrun) -------------------- #
    use_ddp = args.gpus > 1 and "RANK" in os.environ
    if use_ddp:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        if args.gpus > 1 and "RANK" not in os.environ:
            print(
                "[pretrain] --gpus>1 requested but not launched via torchrun; "
                "falling back to single-GPU. Use: torchrun --standalone "
                f"--nproc_per_node={args.gpus} scripts/pretrain.py ...",
                flush=True,
            )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    rank = _rank()
    world = _world_size()

    # ---- data ------------------------------------------------------------ #
    ds = ShardDataset(args.data)
    if use_ddp:
        sampler = DistributedSampler(ds, num_replicas=world, rank=rank, shuffle=True)
        loader = DataLoader(
            ds, batch_size=args.batch, sampler=sampler, num_workers=args.workers,
            pin_memory=(device.type == "cuda"), drop_last=True,
        )
    else:
        sampler = None
        loader = DataLoader(
            ds, batch_size=args.batch, shuffle=True, num_workers=args.workers,
            pin_memory=(device.type == "cuda"), drop_last=True,
        )

    if rank == 0:
        print(f"[pretrain] samples={len(ds)} shards={len(ds.files)} world={world} "
              f"net={args.channels}x{args.blocks} batch={args.batch} epochs={args.epochs}",
              flush=True)

    # ---- model / optim --------------------------------------------------- #
    model = PVNet(channels=args.channels, blocks=args.blocks).to(device)
    raw_model = model
    if use_ddp:
        model = nn.parallel.DistributedDataParallel(model, device_ids=[device.index])
        raw_model = model.module

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    amp_dtype = _amp_dtype(args.precision) if device.type == "cuda" else None
    scaler = torch.amp.GradScaler("cuda", enabled=(amp_dtype == torch.float16))

    steps_per_epoch = max(1, len(loader))
    total_steps = steps_per_epoch * args.epochs
    diag = _diag_positions() if rank == 0 else None
    if rank == 0:
        os.makedirs(args.out, exist_ok=True)

    global_step = 0
    for epoch in range(args.epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)
        model.train()
        ep_loss = 0.0
        ep_n = 0
        t0 = time.time()
        for planes, pi, z, mask in loader:
            lr = args.min_lr + 0.5 * (args.lr - args.min_lr) * (
                1.0 + math.cos(math.pi * min(global_step / max(total_steps, 1), 1.0))
            )
            for g in optimizer.param_groups:
                g["lr"] = lr

            planes = planes.to(device, non_blocking=True)
            pi = pi.to(device, non_blocking=True)
            z = z.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            ctx = (torch.autocast("cuda", dtype=amp_dtype)
                   if amp_dtype is not None else _Null())
            with ctx:
                logits, value = model(planes)
                # L2 via optimizer weight_decay -> c=0 here to avoid double count.
                loss = az_loss(logits, value, pi, z, mask, c=0.0, model=None)

            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            lv = float(loss.detach())
            ep_loss += lv
            ep_n += 1
            global_step += 1
            if rank == 0 and args.log_every and global_step % args.log_every == 0:
                print(f"[pretrain] e{epoch} step{global_step}/{total_steps} "
                      f"loss={lv:.4f} lr={lr:.2e}", flush=True)

        # ---- epoch end: checkpoint + diagnostics (rank 0 only) ----------- #
        if rank == 0:
            avg = ep_loss / max(ep_n, 1)
            top1 = diag_top1(raw_model, diag, device)
            dt = time.time() - t0
            print(f"[pretrain] epoch {epoch} done avg_loss={avg:.4f} "
                  f"({dt:.1f}s) diag[{top1}]", flush=True)
            ckpt = os.path.join(args.out, f"pretrained_e{epoch:03d}.pt")
            torch.save(
                {
                    "epoch": epoch,
                    "model": raw_model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "channels": args.channels,
                    "blocks": args.blocks,
                    "avg_loss": avg,
                },
                ckpt,
            )
            # Maintain a "best" symlink-like copy for the RL cold start.
            best = os.path.join(args.out, "pretrained_best.pt")
            tmp = best + ".tmp"
            torch.save(
                {"model": raw_model.state_dict(), "channels": args.channels,
                 "blocks": args.blocks, "epoch": epoch, "avg_loss": avg},
                tmp,
            )
            os.replace(tmp, best)
            print(f"[pretrain] wrote {ckpt} and {best}", flush=True)

    if use_ddp:
        dist.barrier()
        dist.destroy_process_group()
    return 0


class _Null:
    def __enter__(self):
        return None

    def __exit__(self, *a):
        return False


if __name__ == "__main__":
    raise SystemExit(main())
