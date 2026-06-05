"""Smoke test for :mod:`xqai.network` — runs on CPU, no _xqcore dependency.

Builds a small PVNet(128, 10), runs a random forward pass, checks output
shapes and value range, then runs one az_loss backward pass with random
pi/z/mask. Run with::

    python -m xqai._smoke_network
"""

from __future__ import annotations

import torch

from xqai.encoding import ACTION_DIM, NUM_PLANES, NUM_ROWS, NUM_COLS
from xqai.network import PVNet, az_loss, masked_log_softmax


def main() -> None:
    torch.manual_seed(0)
    device = torch.device("cpu")

    batch = 8
    net = PVNet(channels=128, blocks=10).to(device)
    net.train()

    x = torch.randn(batch, NUM_PLANES, NUM_ROWS, NUM_COLS, device=device)
    policy_logits, value = net(x)

    print(f"input          : {tuple(x.shape)}")
    print(f"policy_logits  : {tuple(policy_logits.shape)} (expect ({batch}, {ACTION_DIM}))")
    print(f"value          : {tuple(value.shape)} (expect ({batch}, 1))")

    assert policy_logits.shape == (batch, ACTION_DIM), policy_logits.shape
    assert value.shape == (batch, 1), value.shape

    vmin = float(value.min().detach())
    vmax = float(value.max().detach())
    print(f"value range    : [{vmin:.4f}, {vmax:.4f}] (expect within [-1, 1])")
    assert vmin >= -1.0 - 1e-5 and vmax <= 1.0 + 1e-5, (vmin, vmax)

    # Random legal masks: each row gets a random subset of legal moves (>=1).
    mask = (torch.rand(batch, ACTION_DIM, device=device) < 0.05).float()
    # Guarantee at least one legal move per row.
    forced = torch.randint(0, ACTION_DIM, (batch,), device=device)
    mask[torch.arange(batch), forced] = 1.0

    # Random pi target supported on legal moves, normalized per row.
    pi = torch.rand(batch, ACTION_DIM, device=device) * mask
    pi = pi / pi.sum(dim=-1, keepdim=True)

    # Random outcomes in [-1, 1].
    z = (torch.rand(batch, device=device) * 2.0) - 1.0

    # masked_log_softmax sanity: illegal entries are -inf, legal probs sum to 1.
    logp = masked_log_softmax(policy_logits, mask)
    probs = logp.exp()
    row_sums = probs.sum(dim=-1)
    rs = row_sums.detach()
    print(f"masked prob sums: min={float(rs.min()):.4f} max={float(rs.max()):.4f} (expect ~1)")
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-4), row_sums

    loss = az_loss(policy_logits, value, pi, z, mask, c=1e-4, model=net)
    print(f"az_loss        : {float(loss.detach()):.4f}")
    assert torch.isfinite(loss), loss

    loss.backward()
    grad_norm = sum(
        float(p.grad.detach().pow(2).sum()) for p in net.parameters() if p.grad is not None
    ) ** 0.5
    print(f"grad L2 norm   : {grad_norm:.4f} (expect > 0, finite)")
    assert grad_norm > 0.0

    print("\nSMOKE TEST PASSED")


if __name__ == "__main__":
    main()
