"""End-to-end "tracer bullet" smoke test for the self-play / train pipeline.

If the C++ rules kernel ``xqai._xqcore`` is importable, this runs the whole
loop at a tiny scale on CPU:

    a few parallel Positions
      -> PUCTPlanner + DummyNet self-play for a few plies
      -> samples written to a (small) ReplayBuffer in /dev/shm
      -> sample one minibatch
      -> one az_loss forward/backward through a small PVNet(128, 10) on CPU.

If ``_xqcore`` is **not** built yet, it prints a friendly notice and exits
cleanly (return code 0) rather than crashing — the rest of the pipeline code is
written against the §3 Position contract and will work once the extension lands.

Run with::

    python -m xqai._smoke_pipeline
"""

from __future__ import annotations

import sys


def main() -> int:
    print("[smoke] xqai self-play/train pipeline tracer bullet")

    # --- dependency / extension probing ----------------------------------- #
    try:
        import numpy as np  # noqa: F401
        import torch
    except Exception as exc:  # pragma: no cover
        print(f"[smoke] SKIP: torch/numpy not available ({exc}).")
        return 0

    try:
        from xqai._xqcore import Position  # noqa: F401
    except Exception as exc:
        print(
            "[smoke] SKIP: xqai._xqcore not built yet "
            f"({type(exc).__name__}: {exc}).\n"
            "[smoke] Build the C++ extension (cpp/CMakeLists.txt -> _xqcore), "
            "then re-run. Pipeline code is written to the §3 Position contract "
            "and will run unchanged once it is available."
        )
        return 0

    # --- imports (safe now) ----------------------------------------------- #
    from xqai.dummynet import DummyNet
    from xqai.mcts import PUCTPlanner
    from xqai.network import PVNet, az_loss
    from xqai.replay import ReplayBuffer
    from xqai.selfplay import SelfPlayWorker

    torch.manual_seed(0)
    device = torch.device("cpu")

    # --- 1. self-play with DummyNet + PUCT -------------------------------- #
    print("[smoke] running self-play: 4 games, n_sim=8, a few plies ...")
    net = DummyNet().to(device).eval()
    planner = PUCTPlanner(c_puct=1.5, seed=0)

    buf = ReplayBuffer(
        capacity=4096, name="xqai_smoke", create=True,
        recent_weight=True, mirror_augment=True, seed=0,
    )
    try:
        worker = SelfPlayWorker(
            net, planner, replay=buf,
            parallel_games=4, n_sim=8, temp_moves=4,
            resign_threshold=None,  # disable to keep games going in smoke
            max_plies=12, seed=0,
        )
        stats = worker.run(num_games=4)
        print(f"[smoke] self-play stats: {stats}")
        print(f"[smoke] replay buffer size: {len(buf)}")

        if len(buf) == 0:
            print("[smoke] WARN: no samples produced (games may have ended "
                  "instantly); skipping train step.")
            return 0

        # --- 2. sample a minibatch ---------------------------------------- #
        batch_size = min(8, len(buf))
        planes, pi, z, mask = buf.sample(batch_size)
        print(f"[smoke] sampled minibatch: planes={tuple(planes.shape)} "
              f"pi={tuple(pi.shape)} z={tuple(z.shape)} mask={tuple(mask.shape)}")

        # --- 3. one az_loss forward/backward on small PVNet --------------- #
        print("[smoke] one az_loss train step on PVNet(128, 10) [CPU] ...")
        model = PVNet(channels=128, blocks=10).to(device).train()
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        opt.zero_grad(set_to_none=True)
        logits, value = model(planes.to(device))
        loss = az_loss(logits, value, pi.to(device), z.to(device), mask.to(device))
        loss_val = loss.detach().item()
        loss.backward()
        opt.step()
        print(f"[smoke] az_loss = {loss_val:.4f}  (backward + step OK)")

        print("[smoke] PASS: full pipeline ran end-to-end.")
        return 0
    finally:
        buf.unlink()


if __name__ == "__main__":
    sys.exit(main())
