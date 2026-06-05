"""DummyNet — a zero-cost stand-in for :class:`xqai.network.PVNet`.

This is the "tracer bullet" network: it has the **exact same forward
signature** as ``PVNet`` (input ``[B, NUM_PLANES, 10, 9]`` float32 ->
``(policy_logits [B, ACTION_DIM], value [B, 1])``) but does essentially no
computation. It lets the whole self-play / replay / training pipeline be wired
up and run end-to-end on a CPU before any real network is trained, and before
a GPU is even available.

Behaviour
---------
- ``policy_logits``: all zeros -> a **uniform** distribution after softmax
  (the MCTS code masks illegal moves itself, so uniform-over-legal falls out).
- ``value``: ~0 (a tiny learnable bias so ``.parameters()`` is non-empty and the
  module can be ``.to(device)`` / ``state_dict()``-ed like a real net).

It deliberately keeps a single trivial parameter so optimizers, AMP autocast and
checkpointing code paths exercise the same machinery as with ``PVNet``.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .encoding import ACTION_DIM, NUM_PLANES


class DummyNet(nn.Module):
    """Uniform-policy / zero-value network matching :class:`PVNet`'s interface.

    Parameters mirror ``PVNet`` so it is a drop-in replacement; all but
    ``action_dim`` / ``num_planes`` are accepted and ignored.
    """

    def __init__(
        self,
        channels: int = 128,
        blocks: int = 10,
        action_dim: int = ACTION_DIM,
        num_planes: int = NUM_PLANES,
    ):
        super().__init__()
        self.channels = channels
        self.blocks = blocks
        self.action_dim = action_dim
        self.num_planes = num_planes
        # A single trivial parameter: keeps ``.parameters()`` non-empty so the
        # module behaves like a real net for .to()/optimizer/state_dict code.
        self._value_bias = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(policy_logits [B, action_dim], value [B, 1])``.

        ``policy_logits`` are all zeros (uniform after softmax). ``value`` is the
        learnable bias broadcast over the batch (initialised to 0).
        """
        b = x.shape[0]
        policy_logits = torch.zeros(
            b, self.action_dim, dtype=x.dtype, device=x.device
        )
        value = self._value_bias.to(dtype=x.dtype, device=x.device).expand(b, 1)
        return policy_logits, value


__all__ = ["DummyNet"]
