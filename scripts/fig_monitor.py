#!/usr/bin/env python
"""Report 7.7 training-monitor figures.

Task A: supervised training monitoring curves (one point per pstep checkpoint):
  - policy entropy (mean over a fixed eval subset, full 8100-way softmax)
  - top-1 move-match accuracy (argmax policy == supervised pi_index)
  - value-head sign accuracy (sign(value) == sign(z), z==0 excluded)

Task B:
  - B1: result distribution / draw rate from data labels (z)
  - B2: game-length distribution from a small self-play batch with the cold-start net

Outputs (PNG dpi=140 + CSV) -> logs/fig/. Figures are labelled in English.
All numbers are computed from real checkpoints / real labelled data; nothing is
fabricated. Anything not collected is reported explicitly as "not collected".
"""
from __future__ import annotations

import csv
import glob
import os
import re
import time

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from xqai.network import PVNet

ROOT = "/mnt/nvme3n1/gameTheory"
CKPT_DIR = os.path.join(ROOT, "checkpoints")
DATA_DIR = os.path.join(ROOT, "data/processed")
OUT_DIR = os.path.join(ROOT, "logs/fig")
DEVICE = torch.device("cuda:0")

os.makedirs(OUT_DIR, exist_ok=True)


def log(msg: str) -> None:
    print(f"[fig_monitor] {msg}", flush=True)


# --------------------------------------------------------------------------- #
# Eval subset
# --------------------------------------------------------------------------- #
def load_eval_subset(shard_ids=(1000, 1001, 1002, 1003)):
    planes_l, pi_l, z_l = [], [], []
    for sid in shard_ids:
        path = os.path.join(DATA_DIR, f"shard_{sid:05d}.npz")
        if not os.path.exists(path):
            log(f"WARN shard missing: {path}")
            continue
        d = np.load(path)
        planes_l.append(d["planes"].astype(np.float32))
        pi_l.append(d["pi_index"].astype(np.int64))
        z_l.append(d["z"].astype(np.int64))
    planes = np.concatenate(planes_l)
    pi = np.concatenate(pi_l)
    z = np.concatenate(z_l)
    log(f"eval subset: {planes.shape[0]} samples from shards {list(shard_ids)}")
    return planes, pi, z


def load_net(path):
    ck = torch.load(path, weights_only=False, map_location="cpu")
    net = PVNet(channels=ck["channels"], blocks=ck["blocks"])
    net.load_state_dict(ck["model"])
    net.eval().to(DEVICE)
    return net


def eval_checkpoint(net, planes, pi, z, mb=2048):
    """Return (mean_entropy, top1_acc, value_sign_acc) over the eval subset.

    - entropy: mean over samples of the full 8100-way softmax entropy (nats).
    - top1: fraction where argmax(policy_logits) == supervised pi_index.
    - value_sign: among samples with z != 0, fraction where sign(value)==sign(z).
    """
    n = planes.shape[0]
    pi_t = torch.from_numpy(pi).to(DEVICE)
    z_t = torch.from_numpy(z).to(DEVICE)

    ent_sum = 0.0
    top1_hits = 0
    vs_hits = 0
    vs_total = 0

    with torch.no_grad():
        for i in range(0, n, mb):
            x = torch.from_numpy(planes[i:i + mb]).to(DEVICE)
            logits, value = net(x)  # logits [B,8100], value [B,1] in [-1,1]
            logp = torch.log_softmax(logits.float(), dim=-1)
            p = logp.exp()
            ent = -(p * logp).sum(dim=-1)  # [B], full-space entropy (nats)
            ent_sum += float(ent.sum())

            pred = logits.argmax(dim=-1)
            top1_hits += int((pred == pi_t[i:i + mb]).sum())

            v = value.reshape(-1)
            zb = z_t[i:i + mb]
            nz = zb != 0
            if int(nz.sum()) > 0:
                vsign = torch.sign(v[nz])
                zsign = torch.sign(zb[nz].float())
                vs_hits += int((vsign == zsign).sum())
                vs_total += int(nz.sum())

    mean_ent = ent_sum / n
    top1 = top1_hits / n
    vsacc = vs_hits / vs_total if vs_total else float("nan")
    return mean_ent, top1, vsacc


def task_a(planes, pi, z):
    paths = sorted(glob.glob(os.path.join(CKPT_DIR, "pstep_*.pt")))
    rows = []
    for path in paths:
        m = re.search(r"pstep_(\d+)\.pt", os.path.basename(path))
        step = int(m.group(1))
        t0 = time.time()
        net = load_net(path)
        ent, top1, vsacc = eval_checkpoint(net, planes, pi, z)
        del net
        torch.cuda.empty_cache()
        rows.append((step, ent, top1, vsacc))
        log(f"step {step:6d}: entropy={ent:.4f} top1={top1:.4f} "
            f"val_sign={vsacc:.4f}  ({time.time()-t0:.1f}s)")
    rows.sort()
    steps = [r[0] for r in rows]
    ent = [r[1] for r in rows]
    top1 = [r[2] for r in rows]
    vs = [r[3] for r in rows]

    # CSV (combined)
    with open(os.path.join(OUT_DIR, "train_monitor.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step", "policy_entropy_nats", "top1_acc", "value_sign_acc"])
        for r in rows:
            w.writerow(r)

    # Per-metric CSVs
    def write_one(name, ys, col):
        with open(os.path.join(OUT_DIR, name), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["step", col])
            for s, y in zip(steps, ys):
                w.writerow([s, y])

    write_one("policy_entropy.csv", ent, "policy_entropy_nats")
    write_one("policy_top1acc.csv", top1, "top1_acc")
    write_one("value_sign_acc.csv", vs, "value_sign_acc")

    # Individual figures
    plt.figure(figsize=(7, 4.5))
    plt.plot(steps, ent, "o-", color="#1f77b4")
    plt.xlabel("Supervised training step")
    plt.ylabel("Mean policy entropy (nats, 8100-way softmax)")
    plt.title("Policy entropy decreases as supervised training sharpens")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "fig_policy_entropy.png"), dpi=140)
    plt.close()

    final_top1 = top1[-1]
    plt.figure(figsize=(7, 4.5))
    plt.plot(steps, [v * 100 for v in top1], "o-", color="#2ca02c")
    plt.xlabel("Supervised training step")
    plt.ylabel("Top-1 move-match accuracy (%)")
    plt.title(f"Top-1 move-match accuracy rises to ~{final_top1*100:.1f}% "
              f"over supervised training")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "fig_policy_top1acc.png"), dpi=140)
    plt.close()

    plt.figure(figsize=(7, 4.5))
    plt.plot(steps, [v * 100 for v in vs], "o-", color="#d62728")
    plt.axhline(50, ls="--", color="grey", alpha=0.6, label="chance (50%)")
    plt.xlabel("Supervised training step")
    plt.ylabel("Value-head sign accuracy (%, z!=0 only)")
    plt.title("Value-head sign accuracy vs game result")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "fig_value_sign_acc.png"), dpi=140)
    plt.close()

    # Combined multi-panel
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
    ax[0].plot(steps, ent, "o-", color="#1f77b4")
    ax[0].set_title("Policy entropy")
    ax[0].set_xlabel("step"); ax[0].set_ylabel("entropy (nats)")
    ax[0].grid(True, alpha=0.3)
    ax[1].plot(steps, [v * 100 for v in top1], "o-", color="#2ca02c")
    ax[1].set_title("Top-1 move-match accuracy")
    ax[1].set_xlabel("step"); ax[1].set_ylabel("accuracy (%)")
    ax[1].grid(True, alpha=0.3)
    ax[2].plot(steps, [v * 100 for v in vs], "o-", color="#d62728")
    ax[2].axhline(50, ls="--", color="grey", alpha=0.6)
    ax[2].set_title("Value-head sign accuracy")
    ax[2].set_xlabel("step"); ax[2].set_ylabel("accuracy (%)")
    ax[2].grid(True, alpha=0.3)
    fig.suptitle("Supervised training monitoring (fixed eval subset, "
                 f"{planes.shape[0]} samples)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "fig_train_monitor.png"), dpi=140)
    plt.close(fig)

    return {
        "final_entropy": ent[-1],
        "final_top1": top1[-1],
        "final_value_sign": vs[-1],
        "steps": steps,
    }


# --------------------------------------------------------------------------- #
# Task B1: result distribution / draw rate from labels
# --------------------------------------------------------------------------- #
def task_b1(n_shards=20):
    paths = sorted(glob.glob(os.path.join(DATA_DIR, "shard_*.npz")))[:n_shards]
    red = draw = black = 0
    for p in paths:
        z = np.load(p)["z"]
        red += int((z > 0).sum())
        draw += int((z == 0).sum())
        black += int((z < 0).sum())
    total = red + draw + black
    rate = {"red_win": red / total, "draw": draw / total, "black_win": black / total}
    log(f"label dist over {n_shards} shards ({total} samples): "
        f"red={rate['red_win']:.3f} draw={rate['draw']:.3f} black={rate['black_win']:.3f}")

    with open(os.path.join(OUT_DIR, "result_dist.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["outcome", "count", "fraction"])
        w.writerow(["red_win", red, rate["red_win"]])
        w.writerow(["draw", draw, rate["draw"]])
        w.writerow(["black_win", black, rate["black_win"]])
        w.writerow(["total", total, 1.0])

    plt.figure(figsize=(6.5, 4.5))
    labels = ["red-win", "draw", "black-win"]
    vals = [rate["red_win"], rate["draw"], rate["black_win"]]
    colors = ["#d62728", "#7f7f7f", "#1f1f1f"]
    bars = plt.bar(labels, [v * 100 for v in vals], color=colors)
    for b, v in zip(bars, vals):
        plt.text(b.get_x() + b.get_width() / 2, v * 100 + 0.5,
                 f"{v*100:.1f}%", ha="center", va="bottom")
    plt.ylabel("Fraction of labelled samples (%)")
    plt.title(f"Game-result label distribution "
              f"(draw rate {rate['draw']*100:.1f}%, N={total})")
    plt.ylim(0, max(vals) * 100 * 1.2)
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "fig_result_dist.png"), dpi=140)
    plt.close()
    return {"draw_rate_labels": rate["draw"], "total": total, **rate}


# --------------------------------------------------------------------------- #
# Task B2: game-length distribution from a small self-play batch
# --------------------------------------------------------------------------- #
def task_b2(num_games=40, n_sim=40, parallel=20, max_plies=300, time_budget_s=720):
    """Collect per-game half-move (ply) counts via cold-start self-play.

    Returns (lengths, info_str). On failure / timeout returns (None, reason).
    """
    try:
        from xqai._xqcore import Position  # noqa: F401
    except Exception as e:
        return None, f"_xqcore import failed: {e}"

    try:
        from xqai.mcts import PUCTPlanner
        from xqai.encoding import BLACK, flip_move
    except Exception as e:
        return None, f"import failed: {e}"

    path = os.path.join(CKPT_DIR, "pretrained_best.pt")
    net = load_net(path)
    planner = PUCTPlanner(add_noise=True, seed=0)
    rng = np.random.default_rng(0)

    from xqai._xqcore import Position
    _ONGOING, _RED_WIN, _BLACK_WIN, _DRAW = 0, 1, 2, 3

    def sample_move(pi, temperature):
        nz = np.nonzero(pi)[0]
        if nz.size == 0:
            return -1
        if temperature <= 1e-6:
            return int(nz[np.argmax(pi[nz])])
        p = pi[nz].astype(np.float64) ** (1.0 / temperature)
        s = p.sum()
        if s <= 0:
            return int(rng.choice(nz))
        p /= s
        return int(rng.choice(nz, p=p))

    lengths = []
    results = []
    t_start = time.time()
    temp_moves = 20

    finished = 0
    while finished < num_games and time.time() - t_start < time_budget_s:
        batch = min(parallel, num_games - finished)
        games = [{"pos": Position(), "ply": 0, "done": False, "res": _ONGOING}
                 for _ in range(batch)]
        ply = 0
        while ply < max_plies and time.time() - t_start < time_budget_s:
            live = [g for g in games if not g["done"]]
            if not live:
                break
            pis = planner.search([g["pos"] for g in live], net, n_sim)
            temperature = 1.0 if ply < temp_moves else 0.0
            for g, pi in zip(live, pis):
                pos = g["pos"]
                mv = sample_move(pi, temperature)
                if mv < 0:
                    g["res"] = _BLACK_WIN if pos.side_to_move() == 0 else _RED_WIN
                    g["done"] = True
                    continue
                real = flip_move(mv) if pos.side_to_move() == BLACK else mv
                pos.push(real)
                g["ply"] += 1
                res = pos.result()
                if res != _ONGOING:
                    g["res"] = res
                    g["done"] = True
            ply += 1
        for g in games:
            if not g["done"]:
                g["res"] = _DRAW
            lengths.append(g["ply"])
            results.append(g["res"])
            finished += 1
        log(f"self-play: {finished}/{num_games} games done "
            f"({time.time()-t_start:.1f}s)")

    del net
    torch.cuda.empty_cache()

    if not lengths:
        return None, "no games collected"

    lengths = np.array(lengths)
    results = np.array(results)
    draws = int((results == _DRAW).sum())
    info = {
        "num_games": len(lengths),
        "mean_plies": float(lengths.mean()),
        "median_plies": float(np.median(lengths)),
        "min_plies": int(lengths.min()),
        "max_plies": int(lengths.max()),
        "draw_rate_selfplay": draws / len(lengths),
        "n_sim": n_sim,
    }

    with open(os.path.join(OUT_DIR, "game_length.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["game_idx", "half_moves", "result_code"])
        for i, (L, r) in enumerate(zip(lengths, results)):
            w.writerow([i, int(L), int(r)])

    plt.figure(figsize=(7, 4.5))
    plt.hist(lengths, bins=20, color="#9467bd", edgecolor="white")
    plt.axvline(lengths.mean(), color="red", ls="--",
                label=f"mean={lengths.mean():.1f}")
    plt.axvline(np.median(lengths), color="black", ls=":",
                label=f"median={np.median(lengths):.0f}")
    plt.xlabel("Game length (half-moves / plies)")
    plt.ylabel("Number of games")
    plt.title(f"Self-play game-length distribution "
              f"(cold-start net, n_sim={n_sim}, {len(lengths)} games)")
    plt.legend()
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "fig_game_length.png"), dpi=140)
    plt.close()
    return info, "ok"


def main():
    log(f"torch {torch.__version__}, cuda={torch.cuda.is_available()}, "
        f"device={torch.cuda.get_device_name(0)}")
    planes, pi, z = load_eval_subset()

    a = task_a(planes, pi, z)
    b1 = task_b1()

    log("starting self-play (task B2)...")
    b2, b2_status = task_b2()

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"[A] supervised monitor, steps {a['steps'][0]}..{a['steps'][-1]} "
          f"({len(a['steps'])} ckpts)")
    print(f"    final policy entropy   : {a['final_entropy']:.4f} nats")
    print(f"    final top-1 move match : {a['final_top1']*100:.2f}%")
    print(f"    final value-sign acc   : {a['final_value_sign']*100:.2f}%")
    print(f"[B1] label result dist: red={b1['red_win']*100:.1f}% "
          f"draw={b1['draw']*100:.1f}% black={b1['black_win']*100:.1f}% "
          f"(N={b1['total']})")
    if b2 is not None:
        print(f"[B2] self-play game length: mean={b2['mean_plies']:.1f} "
              f"median={b2['median_plies']:.0f} plies, "
              f"range [{b2['min_plies']},{b2['max_plies']}], "
              f"draw_rate={b2['draw_rate_selfplay']*100:.1f}% "
              f"({b2['num_games']} games, n_sim={b2['n_sim']})")
    else:
        print(f"[B2] game length: NOT COLLECTED ({b2_status})")
    print("=" * 70)
    print(f"outputs in {OUT_DIR}")


if __name__ == "__main__":
    main()
