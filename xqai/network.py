"""Policy-Value network for xqai (INTERFACES.md §5).

PVNet is an AlphaZero-style residual conv tower with two heads:

- policy head -> ``[B, ACTION_DIM]`` raw logits (move = from*90 + to).
- value head  -> ``[B, 1]`` scalar in ``[-1, 1]`` (tanh), from the side-to-move's
  perspective.

Input ``x`` is ``[B, NUM_PLANES, 10, 9]`` float32 (see :mod:`xqai.encoding`).

Loss (§5)::

    L = (z - v)^2 - sum(pi * log_softmax(masked_logits)) + c * ||w||^2,  c = 1e-4

Illegal moves are masked to ``-inf`` before the softmax via ``legal_mask``.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoding import ACTION_DIM, NUM_COLS, NUM_PLANES, NUM_ROWS


class _ResidualBlock(nn.Module):
    """Standard AlphaZero residual block: (conv-bn-relu) x2 + skip, then relu."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + x
        return F.relu(out)


class PVNet(nn.Module):
    """Policy-value residual network.

    Parameters
    ----------
    channels:
        Tower width. 128 for the small net, 256 for the medium net.
    blocks:
        Number of residual blocks. 10 (small) / 15 (medium), up to 20.
    action_dim:
        Policy output size (default ``ACTION_DIM = 8100``).
    num_planes:
        Input plane count (default ``NUM_PLANES = 15``).
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

        board_cells = NUM_ROWS * NUM_COLS  # 90

        # Stem: Conv3x3 + BN + ReLU.
        self.stem = nn.Sequential(
            nn.Conv2d(num_planes, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

        # Residual tower.
        self.tower = nn.Sequential(*[_ResidualBlock(channels) for _ in range(blocks)])

        # Policy head: Conv1x1 -> BN -> ReLU -> Flatten -> Linear -> logits.
        policy_channels = 32
        self.policy_conv = nn.Sequential(
            nn.Conv2d(channels, policy_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(policy_channels),
            nn.ReLU(inplace=True),
        )
        self.policy_fc = nn.Linear(policy_channels * board_cells, action_dim)

        # Value head: Conv1x1 -> BN -> ReLU -> Flatten -> Linear -> ReLU -> Linear(1) -> tanh.
        value_channels = 8
        self.value_conv = nn.Sequential(
            nn.Conv2d(channels, value_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(value_channels),
            nn.ReLU(inplace=True),
        )
        self.value_fc1 = nn.Linear(value_channels * board_cells, channels)
        self.value_fc2 = nn.Linear(channels, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(policy_logits [B, action_dim], value [B, 1] in [-1, 1])``."""
        x = self.stem(x)
        x = self.tower(x)

        p = self.policy_conv(x)
        p = p.flatten(start_dim=1)
        policy_logits = self.policy_fc(p)

        v = self.value_conv(x)
        v = v.flatten(start_dim=1)
        v = F.relu(self.value_fc1(v))
        value = torch.tanh(self.value_fc2(v))

        return policy_logits, value


def masked_log_softmax(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Log-softmax over legal moves only.

    Illegal entries (``mask == 0``) are set to ``-inf`` before the softmax so
    they get zero probability. ``mask`` is ``[B, action_dim]`` (or broadcastable)
    with 1 = legal. To avoid NaNs from rows that are entirely masked, such rows
    fall back to an unmasked log-softmax.

    Returns log-probabilities ``[B, action_dim]``; illegal entries are ``-inf``.

    .. warning::
        This is for **inference / move selection** (MCTS priors), where masking
        to legal moves is correct. Do **NOT** use it inside the training policy
        loss. A masked (subset) softmax leaves the logits of all out-of-mask
        moves completely unconstrained by the gradient; those logits then drift
        arbitrarily during training. If the mask is anything narrower than "all
        legal moves" (e.g. an MCTS-visit-derived mask, or a single-move one-hot)
        the unvisited/illegal logits can drift high and get picked at play time
        -> the model degrades. The training loss must instead be a cross-entropy
        over the **full** action space (see :func:`az_loss`), which actively
        pushes every non-target logit down. This was the root cause of two
        confirmed RL/pretrain degradation bugs.
    """
    mask = mask.to(dtype=logits.dtype)
    has_legal = mask.sum(dim=-1, keepdim=True) > 0
    neg_inf = torch.finfo(logits.dtype).min
    # Where there is at least one legal move, mask illegal entries; otherwise
    # leave logits untouched so the row stays finite.
    masked_logits = torch.where(
        (mask > 0) | (~has_legal),
        logits,
        torch.full_like(logits, neg_inf),
    )
    return F.log_softmax(masked_logits, dim=-1)


def az_loss(
    policy_logits: torch.Tensor,
    value: torch.Tensor,
    pi_target: torch.Tensor,
    z: torch.Tensor,
    mask: torch.Tensor | None = None,
    c: float = 1e-4,
    model: nn.Module | None = None,
) -> torch.Tensor:
    """AlphaZero loss (§5): value MSE + policy cross-entropy + L2.

    The policy term is a cross-entropy over the **full** action space
    (``-sum_a pi_a * log_softmax(logits)_a``). It is deliberately **not** a
    masked / subset softmax.

    .. warning::
        Do not reintroduce masking into the policy loss. Masking the softmax to
        a subset of moves (legal-only, MCTS-visited-only, or a single-move
        one-hot) leaves the out-of-mask logits unconstrained by the gradient;
        they drift during training and can be selected at play time, causing the
        model to degrade. The full-space cross-entropy used here actively pushes
        every non-target logit down. ``pi_target`` is already zero on
        illegal/unvisited moves, so the full-space CE never rewards them. The
        ``mask`` argument is accepted only for call-site backward compatibility
        and is **ignored** for the policy loss (masking belongs in MCTS / move
        selection via :func:`masked_log_softmax`, not in training).

    Parameters
    ----------
    policy_logits : ``[B, action_dim]`` raw logits from :meth:`PVNet.forward`.
    value         : ``[B, 1]`` predicted value in ``[-1, 1]`` (side-to-move view).
    pi_target     : ``[B, action_dim]`` MCTS target distribution (rows sum to 1).
    z             : ``[B]`` or ``[B, 1]`` game outcome in ``[-1, 1]``
                    (side-to-move view, matching :func:`xqai.encoding.encode`).
    mask          : ignored; kept for backward compatibility (see warning).
    c             : L2 weight-decay coefficient (default 1e-4).
    model         : optional module whose weights contribute the ``c*||w||^2``
                    term. If ``None`` the L2 term is omitted (e.g. when weight
                    decay is handled by the optimizer instead).

    Returns
    -------
    A scalar loss tensor (mean over the batch, plus the optional L2 term).
    """
    del mask  # intentionally unused: see warning above.
    value = value.reshape(-1)
    z = z.reshape(-1).to(dtype=value.dtype)
    value_loss = F.mse_loss(value, z)

    # Full action-space cross-entropy (NOT a masked/subset softmax). pi_target is
    # zero on illegal/unvisited moves, so this never rewards them while actively
    # suppressing every non-target logit.
    log_probs = F.log_softmax(policy_logits.float(), dim=-1)
    policy_loss = -(pi_target.float() * log_probs).sum(dim=-1).mean()

    loss = value_loss + policy_loss

    if model is not None and c > 0:
        l2 = torch.zeros((), device=loss.device, dtype=loss.dtype)
        for p in model.parameters():
            if p.requires_grad:
                l2 = l2 + p.pow(2).sum()
        loss = loss + c * l2

    return loss


__all__ = ["PVNet", "masked_log_softmax", "az_loss"]
