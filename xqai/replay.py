"""ReplayBuffer for xqai (INTERFACES.md §7).

A fixed-capacity **ring buffer** backed by ``multiprocessing.shared_memory``
segments living under ``/dev/shm`` so that multiple self-play workers and the
learner process can read/write the same buffer without copying through pipes.

Each slot stores one training sample:

- ``planes`` : ``float16 [NUM_PLANES, 10, 9]``  (board encoding, §4)
- ``pi``     : ``float16 [ACTION_DIM]``         (MCTS improved policy)
- ``z``      : ``int8``                          (game outcome in {-1,0,+1}, from
  the side-to-move's perspective at that position)

A small ``meta`` segment holds the shared cursor / size / a monotonically
increasing global write counter (used for recency weighting).

Sampling
--------
- **Recency-weighted**: more recently written samples are sampled with higher
  probability (weight grows linearly with how recently the slot was written),
  toggled by ``recent_weight``.
- **Left/right mirror augmentation**: with prob 0.5 each sampled item is flipped
  horizontally (``col -> 8 - col``) on the board planes, and the policy target
  is permuted by the corresponding move flip (``from/to`` columns mirrored).
  Vertical/perspective handling is already baked into :func:`encode`, so only a
  horizontal flip is a valid symmetry here.

``sample(batch)`` returns torch tensors ``(planes, pi, z, mask)`` where ``mask``
is derived from ``pi`` (legal = any move with positive target prob) so the
training loss can mask illegal logits without needing the original Position.
"""

from __future__ import annotations

import threading
from multiprocessing import shared_memory

import numpy as np
import torch

from .encoding import ACTION_DIM, NUM_COLS, NUM_PLANES, NUM_ROWS, NUM_SQUARES

# Per-sample shapes / dtypes.
_PLANES_SHAPE = (NUM_PLANES, NUM_ROWS, NUM_COLS)
_PLANES_SIZE = NUM_PLANES * NUM_ROWS * NUM_COLS

# meta layout (int64): [0]=write_cursor (next slot), [1]=size (filled count),
# [2]=total_writes (monotonic).
_META_LEN = 3


# --------------------------------------------------------------------------- #
# Horizontal mirror precompute (col -> 8 - col)                               #
# --------------------------------------------------------------------------- #
def _build_hflip_move_perm() -> np.ndarray:
    """Permutation ``perm`` s.t. ``pi_flipped[perm[m]] = pi[m]``.

    A horizontal flip maps square ``sq=(row,col)`` -> ``(row, 8-col)`` and a
    move ``from*90+to`` to the move with both squares flipped.
    """
    perm = np.empty(ACTION_DIM, dtype=np.int64)
    for m in range(ACTION_DIM):
        f, t = divmod(m, NUM_SQUARES)
        fr, fc = divmod(f, NUM_COLS)
        tr, tc = divmod(t, NUM_COLS)
        f2 = fr * NUM_COLS + (NUM_COLS - 1 - fc)
        t2 = tr * NUM_COLS + (NUM_COLS - 1 - tc)
        perm[m] = f2 * NUM_SQUARES + t2
    return perm


_HFLIP_MOVE_PERM = _build_hflip_move_perm()


# --------------------------------------------------------------------------- #
# ReplayBuffer                                                                #
# --------------------------------------------------------------------------- #
class ReplayBuffer:
    """Shared-memory ring buffer of (planes, pi, z) samples.

    Parameters
    ----------
    capacity : int
        Max number of samples retained (oldest overwritten). Default 2e6.
    name : str
        Base name for the shared-memory segments. Workers attach with the same
        ``name`` and ``create=False``.
    create : bool
        If True, allocate fresh segments (the learner / first process). If
        False, attach to existing segments created by another process.
    recent_weight : bool
        Enable recency-weighted sampling.
    mirror_augment : bool
        Enable random horizontal mirror augmentation in :meth:`sample`.
    seed : int | None
        RNG seed for sampling/augmentation.
    """

    def __init__(
        self,
        capacity: int = 2_000_000,
        name: str = "xqai_replay",
        create: bool = True,
        recent_weight: bool = True,
        mirror_augment: bool = True,
        seed: int | None = None,
    ):
        self.capacity = int(capacity)
        self.name = name
        self.recent_weight = recent_weight
        self.mirror_augment = mirror_augment
        self._rng = np.random.default_rng(seed)
        self._lock = threading.Lock()

        planes_bytes = self.capacity * _PLANES_SIZE * np.dtype(np.float16).itemsize
        pi_bytes = self.capacity * ACTION_DIM * np.dtype(np.float16).itemsize
        z_bytes = self.capacity * np.dtype(np.int8).itemsize
        meta_bytes = _META_LEN * np.dtype(np.int64).itemsize

        self._shm = {}
        self._shm["planes"] = self._make_shm(f"{name}_planes", planes_bytes, create)
        self._shm["pi"] = self._make_shm(f"{name}_pi", pi_bytes, create)
        self._shm["z"] = self._make_shm(f"{name}_z", z_bytes, create)
        self._shm["meta"] = self._make_shm(f"{name}_meta", meta_bytes, create)

        self.planes = np.ndarray(
            (self.capacity, *_PLANES_SHAPE), dtype=np.float16, buffer=self._shm["planes"].buf
        )
        self.pi = np.ndarray(
            (self.capacity, ACTION_DIM), dtype=np.float16, buffer=self._shm["pi"].buf
        )
        self.z = np.ndarray((self.capacity,), dtype=np.int8, buffer=self._shm["z"].buf)
        # write_order[slot] = global write counter when that slot was last
        # written (for recency weighting). Kept in shared memory too.
        self._order_shm = self._make_shm(
            f"{name}_order", self.capacity * np.dtype(np.int64).itemsize, create
        )
        self.write_order = np.ndarray(
            (self.capacity,), dtype=np.int64, buffer=self._order_shm.buf
        )
        self.meta = np.ndarray((_META_LEN,), dtype=np.int64, buffer=self._shm["meta"].buf)
        if create:
            self.meta[:] = 0
            self.write_order[:] = -1

    # -- shared memory helpers --------------------------------------------- #
    @staticmethod
    def _make_shm(seg_name: str, size: int, create: bool) -> shared_memory.SharedMemory:
        if create:
            # Unlink any stale segment from a crashed run, then create fresh.
            try:
                stale = shared_memory.SharedMemory(name=seg_name)
                stale.close()
                stale.unlink()
            except FileNotFoundError:
                pass
            return shared_memory.SharedMemory(name=seg_name, create=True, size=max(size, 1))
        return shared_memory.SharedMemory(name=seg_name, create=False)

    # -- properties --------------------------------------------------------- #
    @property
    def size(self) -> int:
        return int(self.meta[1])

    @property
    def total_writes(self) -> int:
        return int(self.meta[2])

    def __len__(self) -> int:
        return self.size

    # -- writing ------------------------------------------------------------ #
    def add(self, planes: np.ndarray, pi: np.ndarray, z: int) -> None:
        """Add a single sample. ``planes`` [15,10,9], ``pi`` [8100], ``z`` int."""
        with self._lock:
            cursor = int(self.meta[0])
            self.planes[cursor] = np.asarray(planes, dtype=np.float16).reshape(_PLANES_SHAPE)
            self.pi[cursor] = np.asarray(pi, dtype=np.float16).reshape(ACTION_DIM)
            self.z[cursor] = np.int8(z)
            self.meta[2] += 1
            self.write_order[cursor] = self.meta[2]
            self.meta[0] = (cursor + 1) % self.capacity
            if self.meta[1] < self.capacity:
                self.meta[1] += 1

    def add_batch(self, planes: np.ndarray, pi: np.ndarray, z: np.ndarray) -> None:
        """Add many samples. ``planes`` [N,15,10,9], ``pi`` [N,8100], ``z`` [N]."""
        planes = np.asarray(planes, dtype=np.float16)
        pi = np.asarray(pi, dtype=np.float16)
        z = np.asarray(z, dtype=np.int8)
        for i in range(planes.shape[0]):
            self.add(planes[i], pi[i], int(z[i]))

    # -- sampling ----------------------------------------------------------- #
    def _sample_indices(self, batch: int) -> np.ndarray:
        size = self.size
        if size == 0:
            raise RuntimeError("ReplayBuffer is empty; cannot sample")
        if self.recent_weight:
            order = self.write_order[:size].astype(np.float64)
            # Map global write counters to [1, size] ranks; newer -> larger.
            # rank by recency: most recent total_writes = highest weight.
            ages = self.total_writes - order  # 0 = newest
            # Linear recency weight: newest gets weight ~size, oldest ~1.
            weights = np.maximum(size - ages, 1.0)
            weights /= weights.sum()
            return self._rng.choice(size, size=batch, p=weights)
        return self._rng.integers(0, size, size=batch)

    def sample(self, batch: int):
        """Sample a minibatch -> ``(planes, pi, z, mask)`` torch tensors.

        - ``planes`` : float32 ``[B, 15, 10, 9]``
        - ``pi``     : float32 ``[B, 8100]`` (rows sum to 1)
        - ``z``      : float32 ``[B]``
        - ``mask``   : float32 ``[B, 8100]`` legal mask derived from ``pi``
        """
        idx = self._sample_indices(batch)
        planes = self.planes[idx].astype(np.float32)        # [B,15,10,9]
        pi = self.pi[idx].astype(np.float32)                # [B,8100]
        z = self.z[idx].astype(np.float32)                  # [B]

        if self.mirror_augment:
            flip = self._rng.random(batch) < 0.5
            if flip.any():
                fi = np.nonzero(flip)[0]
                # Horizontal flip of board planes: reverse the column axis.
                planes[fi] = planes[fi][:, :, :, ::-1]
                # Permute policy by the corresponding move permutation.
                pi[fi] = pi[fi][:, _HFLIP_MOVE_PERM]

        mask = (pi > 0.0).astype(np.float32)
        return (
            torch.from_numpy(np.ascontiguousarray(planes)),
            torch.from_numpy(np.ascontiguousarray(pi)),
            torch.from_numpy(z),
            torch.from_numpy(mask),
        )

    # -- lifecycle ---------------------------------------------------------- #
    def close(self) -> None:
        """Detach from the shared-memory segments (does not unlink)."""
        for shm in list(self._shm.values()) + [self._order_shm]:
            try:
                shm.close()
            except Exception:
                pass

    def unlink(self) -> None:
        """Destroy the shared-memory segments (creator only)."""
        for shm in list(self._shm.values()) + [self._order_shm]:
            try:
                shm.close()
            except Exception:
                pass
            try:
                shm.unlink()
            except FileNotFoundError:
                pass


__all__ = ["ReplayBuffer"]
