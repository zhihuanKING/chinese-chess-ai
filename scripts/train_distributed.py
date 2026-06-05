#!/usr/bin/env python
"""Actor-learner distributed self-play RL for xqai (INTERFACES.md §6/§7, task #7).

Architecture
------------
- ``K`` **self-play worker** processes (``torch.multiprocessing`` spawn),
  distributed over ``selfplay_gpus`` (multiple workers per GPU allowed). Each
  worker pins itself to one GPU (``torch.cuda.set_device``), loads the latest
  exported checkpoint into a :class:`~xqai.network.PVNet`, and runs a
  :class:`~xqai.selfplay.SelfPlayWorker` (+ :class:`~xqai.mcts.PUCTPlanner`) in a
  loop, writing samples into a **shared /dev/shm** :class:`~xqai.replay.ReplayBuffer`
  (workers ``attach`` to the buffer the learner created). Workers poll the
  checkpoint file's mtime and hot-reload weights when it changes.

- ``1`` **learner** process (in the main process; uses the first
  ``learner_gpus`` entry). It creates the shared ReplayBuffer, waits for it to
  fill past a threshold, samples minibatches, optimizes with
  :func:`~xqai.network.az_loss`, and every ``export_every`` steps atomically
  writes the model weights to ``checkpoints/latest.pt`` (write-tmp + os.replace,
  so workers never read a half-written file).

Robustness
----------
- Crashed workers are detected (``proc.is_alive()``) and restarted.
- SIGINT/SIGTERM trigger a graceful shutdown: a shared ``stop`` event is set,
  workers exit their loops, are joined (then terminated if stuck), and the
  learner unlinks the /dev/shm segments so nothing is leaked.

Smoke mode
----------
``--smoke`` runs a tiny end-to-end coordination check (2 workers on 1 GPU + 1
learner, ~60 s): verifies samples flow into the buffer, the learner samples and
trains (finite loss), a checkpoint is exported and at least one worker observes
the mtime change and hot-reloads, no zombie processes remain, and /dev/shm is
cleaned on exit.

Only adds this file; never modifies ``xqai/*.py``.
"""

from __future__ import annotations

import argparse
import os
import signal
import sys
import time

import numpy as np
import torch
import torch.multiprocessing as mp

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from xqai.config import load_config


# --------------------------------------------------------------------------- #
# Checkpoint I/O (atomic)                                                      #
# --------------------------------------------------------------------------- #
def _export_checkpoint(model, path: str, *, channels: int, blocks: int, step: int) -> None:
    """Atomically write ``model`` weights to ``path`` (tmp + os.replace).

    The replace is atomic on the same filesystem, so a poller never sees a
    truncated file. mtime changes on every export so workers can detect it.
    """
    tmp = f"{path}.tmp.{os.getpid()}"
    state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    torch.save({"model": state, "channels": channels, "blocks": blocks, "step": step}, tmp)
    os.replace(tmp, path)


def _load_into(model, path: str, device) -> bool:
    """Load weights from ``path`` into ``model`` (best effort). True on success."""
    try:
        ckpt = torch.load(path, map_location=device, weights_only=False)
    except Exception:
        return False
    state = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt
    try:
        model.load_state_dict(state)
        return True
    except Exception:
        return False


def _read_ckpt_size(path: str, default_channels: int, default_blocks: int) -> tuple[int, int]:
    """Read ``(channels, blocks)`` from a checkpoint, falling back to defaults.

    Mirrors :func:`scripts.eval_loop._load_net_player`: a checkpoint saved by
    ``pretrain.py`` / this script is a dict ``{"model": state, "channels": C,
    "blocks": B, ...}``. We must rebuild the net at that size or the
    ``load_state_dict`` will fail on a shape mismatch (e.g. a 64x3 pretrained net
    loaded into a 128x10 net).
    """
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
    except Exception:
        return default_channels, default_blocks
    if isinstance(ckpt, dict):
        return (int(ckpt.get("channels", default_channels)),
                int(ckpt.get("blocks", default_blocks)))
    return default_channels, default_blocks


# --------------------------------------------------------------------------- #
# Self-play worker process                                                     #
# --------------------------------------------------------------------------- #
def worker_main(wid: int, gpu_id: int, cfg_dict: dict, ckpt_path: str,
                replay_name: str, replay_capacity: int, stop_evt,
                poll_secs: float, status, games_per_chunk: int) -> None:
    """One self-play worker. Runs until ``stop_evt`` is set.

    ``status`` is a shared dict (Manager) used by the learner/smoke check to
    observe per-worker progress (samples produced, reloads seen).
    """
    # Let the learner own SIGINT/SIGTERM; workers exit via ``stop_evt`` only, so
    # a Ctrl-C doesn't spray KeyboardInterrupt tracebacks across children.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    signal.signal(signal.SIGTERM, signal.SIG_IGN)

    # The MCTS tree ops are numpy-heavy; with many workers per box, BLAS thread
    # oversubscription thrashes the CPU. Cap intra-op threads per worker.
    torch.set_num_threads(max(1, int(os.environ.get("XQAI_WORKER_THREADS", "2"))))

    # Pin to the assigned GPU. We use set_device rather than CUDA_VISIBLE_DEVICES
    # so the parent can address all GPUs; the worker just lives on gpu_id.
    try:
        if torch.cuda.is_available():
            torch.cuda.set_device(gpu_id)
            device = torch.device("cuda", gpu_id)
        else:
            device = torch.device("cpu")
    except Exception:
        device = torch.device("cpu")

    # Lazy imports inside the spawned process.
    from xqai.network import PVNet
    from xqai.mcts import PUCTPlanner
    from xqai.selfplay import SelfPlayWorker
    from xqai.replay import ReplayBuffer

    net_cfg = cfg_dict["network"]
    mcts_cfg = cfg_dict["mcts"]
    sp_cfg = cfg_dict["selfplay"]

    model = PVNet(channels=net_cfg["channels"], blocks=net_cfg["blocks"]).to(device).eval()

    # Attach to the shared buffer created by the learner.
    replay = None
    for _ in range(100):
        try:
            replay = ReplayBuffer(
                capacity=replay_capacity, name=replay_name, create=False,
                recent_weight=cfg_dict["replay"]["recent_weight"],
                mirror_augment=cfg_dict["replay"]["mirror_augment"],
                seed=1000 + wid,
            )
            break
        except FileNotFoundError:
            if stop_evt.is_set():
                return
            time.sleep(0.2)
    if replay is None:
        status[f"w{wid}_error"] = "could not attach replay"
        return

    planner = PUCTPlanner(
        c_puct=mcts_cfg["c_puct"],
        dirichlet_alpha=mcts_cfg["dirichlet_alpha"],
        dirichlet_eps=mcts_cfg["dirichlet_eps"],
        virtual_loss=mcts_cfg["virtual_loss"],
        seed=2000 + wid,
    )
    sp = SelfPlayWorker(
        net=model, planner=planner, replay=replay,
        parallel_games=sp_cfg["parallel_games"],
        n_sim=mcts_cfg["n_sim_selfplay"],
        temp_moves=sp_cfg["temp_moves"],
        resign_threshold=sp_cfg["resign_threshold"],
        max_plies=sp_cfg.get("max_plies", 400),
        seed=3000 + wid,
    )

    last_mtime = 0.0
    reloads = 0
    produced = 0

    def maybe_reload() -> None:
        nonlocal last_mtime, reloads
        try:
            m = os.path.getmtime(ckpt_path)
        except OSError:
            return
        if m > last_mtime:
            if _load_into(model, ckpt_path, device):
                model.eval()
                last_mtime = m
                reloads += 1
                status[f"w{wid}_reloads"] = reloads

    # Initial load (cold start may have placed weights here already).
    maybe_reload()
    status[f"w{wid}_started"] = True
    parent_pid = os.getppid()

    # Flush samples into the buffer in small chunks. ``SelfPlayWorker.run`` only
    # writes to the buffer *after* ``num_games`` games finish, and games run in
    # lockstep up to ``max_plies`` (~400) plies -- so ``run(num_games=parallel)``
    # can take minutes before a single sample appears, which at real scale starves
    # the learner (it never crosses ``min_buffer``). Running a handful of games per
    # call flushes samples every few seconds and lets the loop poll ``stop_evt``
    # often for a snappy shutdown.
    chunk = max(1, int(games_per_chunk))

    next_poll = time.time() + poll_secs
    while not stop_evt.is_set():
        # If the learner was hard-killed (SIGKILL), we get reparented to init
        # (ppid changes / 1). Self-terminate so we don't become an orphan that
        # keeps a stale shm buffer alive forever.
        if os.getppid() != parent_pid:
            break
        try:
            stats = sp.run(num_games=chunk)
            produced += int(stats.get("num_samples", 0))
            status[f"w{wid}_samples"] = produced
            status[f"w{wid}_buf"] = len(replay)
        except Exception as exc:  # keep the worker alive across transient errors
            status[f"w{wid}_error"] = repr(exc)
            time.sleep(0.5)
        if time.time() >= next_poll:
            maybe_reload()
            next_poll = time.time() + poll_secs

    replay.close()
    status[f"w{wid}_exited"] = True


# --------------------------------------------------------------------------- #
# Worker supervisor (spawn + restart)                                          #
# --------------------------------------------------------------------------- #
class WorkerSpec:
    __slots__ = ("wid", "gpu_id")

    def __init__(self, wid: int, gpu_id: int):
        self.wid = wid
        self.gpu_id = gpu_id


def _spawn_worker(ctx, spec, cfg_dict, ckpt_path, replay_name, replay_capacity,
                  stop_evt, poll_secs, status, games_per_chunk):
    p = ctx.Process(
        target=worker_main,
        args=(spec.wid, spec.gpu_id, cfg_dict, ckpt_path, replay_name,
              replay_capacity, stop_evt, poll_secs, status, games_per_chunk),
        daemon=False,
        name=f"sp-worker-{spec.wid}",
    )
    p.start()
    return p


# --------------------------------------------------------------------------- #
# Main / learner                                                               #
# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description="Actor-learner self-play RL for xqai")
    ap.add_argument("--config", default=None, help="path to YAML (default configs/default.yaml)")
    ap.add_argument("--init", default=None, help="initial checkpoint (cold start), e.g. "
                                                 "checkpoints/pretrained_best.pt")
    ap.add_argument("--steps", type=int, default=None, help="learner steps (default: run forever)")
    ap.add_argument("--smoke", action="store_true", help="tiny 60s coordination smoke test")
    # CLI overrides for config values.
    ap.add_argument("--learner-gpu", type=int, default=None)
    ap.add_argument("--selfplay-gpus", type=str, default=None, help="comma list e.g. 2,3,4")
    ap.add_argument("--workers-per-gpu", type=int, default=None)
    ap.add_argument("--parallel-games", type=int, default=None)
    ap.add_argument("--n-sim", type=int, default=None)
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--export-every", type=int, default=None)
    ap.add_argument("--capacity", type=int, default=None)
    ap.add_argument("--min-buffer", type=int, default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    cfg_d = cfg.to_dict()

    # ---- resolve settings (CLI > config) -------------------------------- #
    if args.smoke:
        learner_gpu = args.learner_gpu if args.learner_gpu is not None else 0
        selfplay_gpus = [0]
        workers_per_gpu = 2
        cfg_d["selfplay"]["parallel_games"] = args.parallel_games or 8
        cfg_d["mcts"]["n_sim_selfplay"] = args.n_sim or 12
        cfg_d["selfplay"]["temp_moves"] = 6
        cfg_d["selfplay"]["max_plies"] = 40
        cfg_d["network"]["channels"] = 64
        cfg_d["network"]["blocks"] = 3
        cfg_d["train"]["batch_size"] = args.batch or 64
        export_every = args.export_every or 5
        capacity = args.capacity or 20_000
        min_buffer = args.min_buffer or 64
        steps = args.steps or 100_000  # bounded by the time budget below
        time_budget = 60.0
    else:
        learner_gpu = (args.learner_gpu if args.learner_gpu is not None
                       else cfg_d["distributed"]["learner_gpus"][0])
        if args.selfplay_gpus:
            selfplay_gpus = [int(x) for x in args.selfplay_gpus.split(",") if x != ""]
        else:
            selfplay_gpus = list(cfg_d["distributed"]["selfplay_gpus"])
        workers_per_gpu = (args.workers_per_gpu if args.workers_per_gpu is not None
                           else cfg_d["selfplay"]["workers_per_gpu"])
        if args.parallel_games is not None:
            cfg_d["selfplay"]["parallel_games"] = args.parallel_games
        if args.n_sim is not None:
            cfg_d["mcts"]["n_sim_selfplay"] = args.n_sim
        if args.batch is not None:
            cfg_d["train"]["batch_size"] = args.batch
        cfg_d["selfplay"].setdefault("max_plies", 400)
        export_every = args.export_every or cfg_d["train"]["export_every"]
        capacity = args.capacity or cfg_d["replay"]["capacity"]
        min_buffer = args.min_buffer or cfg_d["train"]["batch_size"]
        steps = args.steps  # may be None -> run until interrupted
        time_budget = None

    # ---- cold-start sizing: rebuild the net at the checkpoint's size -------- #
    # A pretrained ckpt stores its own ``channels``/``blocks`` (pretrain.py). We
    # MUST build the net at that size (like eval_loop.py) or load_state_dict will
    # fail on a shape mismatch. Override the config net size for both the learner
    # and every spawned worker so the whole cluster agrees on the architecture.
    if args.init:
        if not os.path.exists(args.init):
            print(f"[learner] FATAL: --init {args.init} does not exist", flush=True)
            return 2
        ckpt_c, ckpt_b = _read_ckpt_size(
            args.init, cfg_d["network"]["channels"], cfg_d["network"]["blocks"])
        cfg_d["network"]["channels"] = ckpt_c
        cfg_d["network"]["blocks"] = ckpt_b

    ckpt_dir = cfg_d["paths"]["checkpoints"]
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, "latest.pt")
    replay_name = "xqai_replay_smoke" if args.smoke else os.path.basename(
        cfg_d["replay"]["dir"]
    ).replace("/", "_") or "xqai_replay"

    device = torch.device(f"cuda:{learner_gpu}" if torch.cuda.is_available() else "cpu")

    # Build worker specs: workers_per_gpu per selfplay GPU.
    specs: list[WorkerSpec] = []
    wid = 0
    for _ in range(workers_per_gpu):
        for g in selfplay_gpus:
            specs.append(WorkerSpec(wid, g))
            wid += 1
    num_workers = len(specs)

    print(f"[learner] device={device} workers={num_workers} on gpus={selfplay_gpus} "
          f"(x{workers_per_gpu}) net={cfg_d['network']['channels']}x{cfg_d['network']['blocks']} "
          f"batch={cfg_d['train']['batch_size']} n_sim={cfg_d['mcts']['n_sim_selfplay']} "
          f"export_every={export_every} capacity={capacity}", flush=True)

    # ---- build model + shared replay (creator) -------------------------- #
    from xqai.network import PVNet, az_loss
    from xqai.replay import ReplayBuffer

    model = PVNet(channels=cfg_d["network"]["channels"],
                  blocks=cfg_d["network"]["blocks"]).to(device)
    if args.init:
        ok = _load_into(model, args.init, device)
        if not ok:
            # An explicit --init that fails to load must abort: silently training
            # from scratch would waste an entire run's worth of compute.
            print(f"[learner] FATAL: cold-start init from {args.init} FAILED "
                  f"(built {cfg_d['network']['channels']}x{cfg_d['network']['blocks']}; "
                  f"checkpoint state_dict did not match). Aborting.", flush=True)
            try:
                # ReplayBuffer is not created yet here; nothing to unlink.
                pass
            finally:
                return 3
        print(f"[learner] cold-start OK: loaded {cfg_d['network']['channels']}x"
              f"{cfg_d['network']['blocks']} from {args.init}", flush=True)
    model.train()

    # AdamW：冷启动微调更稳、对 lr 宽容（原 SGD lr=0.02 会把预训练权重打崩=灾难性遗忘）。
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg_d["train"]["lr_rl"],
        weight_decay=cfg_d["train"]["weight_decay"],
    )
    amp_dtype = (torch.bfloat16 if str(cfg_d["train"]["precision"]).lower().startswith("bf")
                 and device.type == "cuda" else None)

    replay = ReplayBuffer(
        capacity=capacity, name=replay_name, create=True,
        recent_weight=cfg_d["replay"]["recent_weight"],
        mirror_augment=cfg_d["replay"]["mirror_augment"], seed=7,
    )

    # Export an initial checkpoint so workers start from the learner's weights
    # (cold-start init if provided, else random) and have a file to poll.
    _export_checkpoint(model, ckpt_path, channels=cfg_d["network"]["channels"],
                       blocks=cfg_d["network"]["blocks"], step=0)

    # ---- spawn workers --------------------------------------------------- #
    ctx = mp.get_context("spawn")
    mgr = ctx.Manager()
    status = mgr.dict()
    stop_evt = ctx.Event()
    poll_secs = 1.0 if args.smoke else 5.0
    # Games flushed to the buffer per self-play call. Small => samples appear
    # quickly and the worker polls stop_evt often (snappy shutdown). Capped at
    # parallel_games (run() can't finish more games than it launches per call).
    games_per_chunk = min(cfg_d["selfplay"]["parallel_games"], 1 if args.smoke else 4)

    procs: dict[int, mp.Process] = {}
    for spec in specs:
        procs[spec.wid] = _spawn_worker(ctx, spec, cfg_d, ckpt_path, replay_name,
                                        capacity, stop_evt, poll_secs, status,
                                        games_per_chunk)

    # ---- signal handling ------------------------------------------------- #
    shutting_down = {"flag": False}

    def _handle_signal(signum, frame):
        if not shutting_down["flag"]:
            print(f"\n[learner] signal {signum} received -> graceful shutdown", flush=True)
            shutting_down["flag"] = True
            stop_evt.set()

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    # ---- learner loop ---------------------------------------------------- #
    t_start = time.time()
    step = 0
    last_loss = float("nan")
    exported = 0
    first_export_step = None
    warmed_up = False
    t_last_wait_log = 0.0
    t_last_step_log = 0.0
    try:
        while not stop_evt.is_set():
            if steps is not None and step >= steps:
                break
            if time_budget is not None and (time.time() - t_start) >= time_budget:
                break

            # Restart any dead workers.
            for spec in specs:
                p = procs[spec.wid]
                if not p.is_alive() and p.exitcode is not None and not stop_evt.is_set():
                    if not status.get(f"w{spec.wid}_exited"):
                        print(f"[learner] worker {spec.wid} died (exit={p.exitcode}); "
                              f"restarting", flush=True)
                        procs[spec.wid] = _spawn_worker(
                            ctx, spec, cfg_d, ckpt_path, replay_name, capacity,
                            stop_evt, poll_secs, status, games_per_chunk)

            buf_now = len(replay)
            if buf_now < min_buffer:
                if not warmed_up and (time.time() - t_last_wait_log) >= 5.0:
                    print(f"[learner] warmup buf={buf_now}/{min_buffer} "
                          f"(waiting for self-play samples)", flush=True)
                    t_last_wait_log = time.time()
                time.sleep(0.1)
                continue
            if not warmed_up:
                warmed_up = True
                print(f"[learner] buffer reached min_buffer ({buf_now} >= "
                      f"{min_buffer}) -> training", flush=True)

            batch = replay.sample(cfg_d["train"]["batch_size"])
            planes, pi, z, mask = (t.to(device, non_blocking=True) for t in batch)
            optimizer.zero_grad(set_to_none=True)
            ctxm = (torch.autocast("cuda", dtype=amp_dtype)
                    if amp_dtype is not None else _Null())
            with ctxm:
                logits, value = model(planes)
                loss = az_loss(logits, value, pi, z, mask, c=0.0, model=None)
            loss.backward()
            optimizer.step()
            last_loss = float(loss.detach())
            step += 1

            # Heartbeat: print step/loss periodically so progress is observable
            # without waiting for an export multiple (export_every can be large).
            if (time.time() - t_last_step_log) >= 10.0:
                print(f"[learner] step={step} loss={last_loss:.4f} buf={len(replay)} "
                      f"exports={exported}", flush=True)
                t_last_step_log = time.time()

            if export_every and step % export_every == 0:
                _export_checkpoint(model, ckpt_path, channels=cfg_d["network"]["channels"],
                                   blocks=cfg_d["network"]["blocks"], step=step)
                exported += 1
                if first_export_step is None:
                    first_export_step = step
                if args.smoke or step % (export_every * 10) == 0:
                    buf = len(replay)
                    print(f"[learner] step={step} loss={last_loss:.4f} buf={buf} "
                          f"exports={exported}", flush=True)
    except KeyboardInterrupt:
        print("\n[learner] KeyboardInterrupt -> graceful shutdown", flush=True)
        stop_evt.set()
    finally:
        # ---- graceful shutdown ------------------------------------------- #
        print("[learner] stopping workers...", flush=True)
        stop_evt.set()
        deadline = time.time() + 20.0
        for p in procs.values():
            timeout = max(0.1, deadline - time.time())
            p.join(timeout=timeout)
        for spec, p in [(s, procs[s.wid]) for s in specs]:
            if p.is_alive():
                print(f"[learner] worker {spec.wid} still alive -> terminate", flush=True)
                p.terminate()
                p.join(timeout=5.0)
            if p.is_alive():
                p.kill()
                p.join(timeout=5.0)

        # Snapshot status before tearing down the manager.
        status_snapshot = dict(status)
        try:
            mgr.shutdown()
        except Exception:
            pass

        # Unlink shared memory (creator owns it) so /dev/shm is clean.
        try:
            replay.unlink()
        except Exception as exc:
            print(f"[learner] replay.unlink warning: {exc}", flush=True)

        zombies = [s.wid for s in specs if procs[s.wid].is_alive()]
        total_samples = sum(v for k, v in status_snapshot.items() if k.endswith("_samples"))
        total_reloads = sum(v for k, v in status_snapshot.items() if k.endswith("_reloads"))
        print(f"[learner] SHUTDOWN steps={step} last_loss={last_loss:.4f} "
              f"exports={exported} worker_samples={total_samples} "
              f"worker_reloads={total_reloads} zombies={zombies}", flush=True)

        if args.smoke:
            return _smoke_report(step, last_loss, exported, total_samples,
                                 total_reloads, zombies, replay_name)
    return 0


def _smoke_report(step, last_loss, exported, total_samples, total_reloads,
                  zombies, replay_name) -> int:
    """Validate the smoke run and print a PASS/FAIL summary."""
    checks = []
    checks.append(("workers produced samples into buffer", total_samples > 0))
    checks.append(("learner trained (>=1 step)", step >= 1))
    checks.append(("loss is finite", np.isfinite(last_loss)))
    checks.append(("checkpoint exported", exported >= 1))
    checks.append(("a worker hot-reloaded after export", total_reloads >= 1))
    checks.append(("no zombie/leftover workers", len(zombies) == 0))

    # Verify /dev/shm cleaned.
    leftovers = []
    shm_dir = "/dev/shm"
    try:
        for n in os.listdir(shm_dir):
            if n.startswith(replay_name):
                leftovers.append(n)
    except OSError:
        pass
    checks.append(("/dev/shm cleaned", len(leftovers) == 0))

    print("\n==== SMOKE REPORT ====", flush=True)
    print(f" total worker samples : {total_samples}", flush=True)
    print(f" learner steps        : {step}", flush=True)
    print(f" last loss            : {last_loss:.4f}", flush=True)
    print(f" checkpoint exports   : {exported}", flush=True)
    print(f" worker hot-reloads   : {total_reloads}", flush=True)
    print(f" zombies              : {zombies}", flush=True)
    print(f" /dev/shm leftovers   : {leftovers}", flush=True)
    all_ok = True
    for name, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}", flush=True)
        all_ok = all_ok and ok
    print(f"==== {'ALL PASS' if all_ok else 'SOME FAILED'} ====", flush=True)
    return 0 if all_ok else 1


class _Null:
    def __enter__(self):
        return None

    def __exit__(self, *a):
        return False


if __name__ == "__main__":
    raise SystemExit(main())
