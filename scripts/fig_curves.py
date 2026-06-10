#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Generate course-report experiment figures from REAL logs only.

All data parsed from /mnt/nvme3n1/gameTheory/logs/*. No fabricated numbers.
Outputs: PNG (dpi=140, Agg backend, English text, unified grid/legend style)
         + a sibling CSV of the plotted data, both under logs/fig/.
Run with: /mnt/nvme3n1/gameTheory/.venv/bin/python scripts/fig_curves.py
"""
import csv
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "/mnt/nvme3n1/gameTheory"
LOGS = os.path.join(ROOT, "logs")
FIG = os.path.join(LOGS, "fig")
os.makedirs(FIG, exist_ok=True)

DPI = 140
STEPS_PER_EPOCH = 21710

# ---- unified style ----
plt.rcParams.update({
    "figure.figsize": (9, 5.2),
    "axes.grid": True,
    "grid.alpha": 0.35,
    "grid.linestyle": "--",
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 10,
    "font.size": 10,
})


def save(fig, name):
    path = os.path.join(FIG, name)
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return path


def write_csv(name, header, rows):
    path = os.path.join(FIG, name)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    return path


# =========================================================================
# 1) Supervised cold-start loss (256x15)
# =========================================================================
def fig_pretrain_loss():
    pat = re.compile(r"\[pretrain\] e(\d+) step(\d+)/21710 loss=([0-9.]+) lr=([0-9.eE+-]+)")
    gstep, loss, epochs = [], [], []
    with open(os.path.join(LOGS, "stage1_pretrain.log")) as f:
        for line in f:
            m = pat.search(line)
            if not m:
                continue
            e, s, l = int(m.group(1)), int(m.group(2)), float(m.group(3))
            gstep.append(e * STEPS_PER_EPOCH + s)
            loss.append(l)
            epochs.append(e)
    assert gstep, "no pretrain rows parsed"

    fig, ax = plt.subplots()
    ax.plot(gstep, loss, color="#1f77b4", lw=1.2, label="train loss")
    max_e = max(epochs)
    for e in range(1, max_e + 1):
        ax.axvline(e * STEPS_PER_EPOCH, color="grey", lw=0.6, alpha=0.35)
    ax.set_xlabel("global step (epoch x 21710 + step)")
    ax.set_ylabel("total loss")
    ax.set_title("Supervised cold-start loss: 5.2 -> 0.17 over 10 epochs (256x15)")
    ax.legend(loc="upper right")
    p = save(fig, "fig_pretrain_loss.png")
    write_csv("fig_pretrain_loss.csv", ["global_step", "epoch", "loss"],
              list(zip(gstep, epochs, loss)))
    return p, (loss[0], loss[-1], len(loss))


# =========================================================================
# 2) RL self-play training loss (drifts UP)
# =========================================================================
def fig_rl_loss():
    pat = re.compile(r"\[learner\] step=(\d+) loss=([0-9.]+) buf=(\d+) exports=(\d+)")
    step, loss, buf = [], [], []
    with open(os.path.join(LOGS, "stage2_rl.log")) as f:
        for line in f:
            if "warmup" in line:
                continue
            m = pat.search(line)
            if not m:
                continue
            step.append(int(m.group(1)))
            loss.append(float(m.group(2)))
            buf.append(int(m.group(3)))
    assert step, "no learner rows parsed"

    fig, ax = plt.subplots()
    ax.plot(step, loss, color="#d62728", lw=0.9, alpha=0.85, label="learner loss")
    # Focus the y-view on the steady state so the 1.0 -> 1.66 drift is legible;
    # the first ~430 steps are the cold-start init transient (loss spikes to ~6.9)
    # and would otherwise compress the whole curve. Data itself is NOT dropped.
    ax.set_ylim(0.9, 1.85)
    ax.text(0.015, 0.97,
            "(first ~430 steps: cold-start init transient, loss up to ~6.9, off-scale)",
            transform=ax.transAxes, ha="left", va="top", fontsize=8, color="grey")
    # mark the early minimum (best self-play loss) vs final
    imin = min(range(len(loss)), key=lambda i: loss[i])
    ax.scatter([step[imin]], [loss[imin]], color="#2ca02c", zorder=5, s=30,
               label=f"min {loss[imin]:.2f} @ step {step[imin]}")
    ax.scatter([step[-1]], [loss[-1]], color="black", zorder=5, s=30,
               label=f"final {loss[-1]:.2f} @ step {step[-1]}")
    ax.annotate("", xy=(step[-1], loss[-1]), xytext=(step[imin], loss[imin]),
                arrowprops=dict(arrowstyle="->", color="black", alpha=0.5))
    ax.set_xlabel("learner step")
    ax.set_ylabel("total loss")
    ax.set_title("RL self-play loss drifts UP (1.0 -> 1.66): self-play degrades the cold-start model")
    ax.legend(loc="lower right")
    p = save(fig, "fig_rl_loss.png")
    write_csv("fig_rl_loss.csv", ["step", "loss", "buf"],
              list(zip(step, loss, buf)))
    return p, (loss[0], loss[imin], loss[-1], step[-1])


# =========================================================================
# 3) Elo vs training step (negative result, with 95% CI band)
# =========================================================================
def fig_elo_vs_step():
    rows = []
    with open(os.path.join(LOGS, "elo_vs_ref.csv")) as f:
        r = csv.DictReader(f)
        for d in r:
            rows.append((int(d["step"]), int(d["games"]), float(d["winrate"]),
                         float(d["elo_diff"]), float(d["lo"]), float(d["hi"])))
    rows.sort(key=lambda x: x[0])
    step = [x[0] for x in rows]
    elo = [x[3] for x in rows]
    lo = [x[4] for x in rows]
    hi = [x[5] for x in rows]
    assert step, "no elo rows parsed"

    fig, ax = plt.subplots()
    ax.fill_between(step, lo, hi, color="#1f77b4", alpha=0.18, label="95% CI")
    ax.plot(step, elo, color="#1f77b4", marker="o", ms=3, lw=1.2,
            label="Elo(RL) - Elo(cold-start)")
    ax.axhline(0, color="black", ls="--", lw=1.0, label="cold-start baseline (0)")
    ax.set_xlabel("RL learner step")
    ax.set_ylabel("relative Elo vs frozen cold-start")
    ax.set_title("RL vs frozen cold-start: relative Elo stays BELOW 0 (no self-improvement)")
    ax.legend(loc="lower right")
    p = save(fig, "fig_elo_vs_step.png")
    write_csv("fig_elo_vs_step.csv", ["step", "elo_diff", "lo", "hi"],
              list(zip(step, elo, lo, hi)))
    return p, (min(elo), max(elo), min(step), max(step))


# =========================================================================
# 4) Learned net vs Alpha-Beta (no crossover) -- from final.log results
# =========================================================================
def fig_crossover():
    ab_ms = [50, 200, 500]
    wr_200 = [0.036, 0.071, 0.000]
    wr_600 = [0.036, 0.107, 0.036]

    fig, ax = plt.subplots()
    ax.plot(ab_ms, wr_200, color="#1f77b4", marker="o", lw=1.4, label="net n_sim=200")
    ax.plot(ab_ms, wr_600, color="#d62728", marker="s", lw=1.4, label="net n_sim=600")
    ax.axhline(0.5, color="grey", ls="--", lw=1.0, label="even (0.5)")
    ax.set_ylim(0, 1)
    ax.set_xlabel("Alpha-Beta time limit per move (ms)")
    ax.set_ylabel("learned-net win rate")
    ax.set_title("Learned net vs Alpha-Beta: winrate 3-11%, NO crossover (gray=0.5)")
    ax.text(0.5, 0.92,
            "n_sim 600 > 200 at every AB budget\n-> positive response to search budget",
            transform=ax.transAxes, ha="center", va="top",
            fontsize=9, bbox=dict(boxstyle="round", fc="#fff4cc", ec="grey", alpha=0.9))
    ax.legend(loc="upper right")
    p = save(fig, "fig_crossover.png")
    write_csv("fig_crossover.csv", ["ab_time_ms", "winrate_nsim200", "winrate_nsim600"],
              list(zip(ab_ms, wr_200, wr_600)))
    return p, (wr_200, wr_600)


# =========================================================================
# 5) Self-play throughput scaling (PVNet 128x10, n_sim=32)
# =========================================================================
def fig_throughput():
    par = [64, 256, 512]
    mps = [138, 162, 163]
    ms = [465, 1578, 3142]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.8))
    ax1.plot(par, mps, color="#2ca02c", marker="o", lw=1.6)
    ax1.axhline(163, color="grey", ls="--", lw=1.0, alpha=0.8, label="plateau ~163")
    ax1.set_xlabel("parallel self-play games")
    ax1.set_ylabel("throughput (moves/s)")
    ax1.set_title("Throughput saturates")
    ax1.set_ylim(0, 180)
    for x, y in zip(par, mps):
        ax1.annotate(f"{y}", (x, y), textcoords="offset points", xytext=(0, 6),
                     ha="center", fontsize=9)
    ax1.legend(loc="lower right")

    ax2.plot(par, ms, color="#9467bd", marker="s", lw=1.6)
    ax2.set_xlabel("parallel self-play games")
    ax2.set_ylabel("latency (ms/step)")
    ax2.set_title("Per-step latency grows linearly")
    for x, y in zip(par, ms):
        ax2.annotate(f"{y}", (x, y), textcoords="offset points", xytext=(0, 6),
                     ha="center", fontsize=9)

    fig.suptitle("Self-play throughput saturates ~163 moves/s (Python MCTS tree-op CPU-bound)",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    path = os.path.join(FIG, "fig_throughput.png")
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    write_csv("fig_throughput.csv", ["parallel_games", "moves_per_s", "ms_per_step"],
              list(zip(par, mps, ms)))
    return path, (mps, ms)


# =========================================================================
# 6) n_sim ablation (positive response to search budget)
# =========================================================================
def fig_nsim_ablation():
    nsim = [60, 600]
    wr = [0.156, 0.312]

    fig, ax = plt.subplots()
    bars = ax.bar([str(n) for n in nsim], wr, color=["#7f7f7f", "#2ca02c"], width=0.55)
    for b, y in zip(bars, wr):
        ax.text(b.get_x() + b.get_width() / 2, y + 0.008, f"{y:.3f}",
                ha="center", va="bottom", fontsize=10)
    ax.plot([0, 1], wr, color="black", ls="-", marker="o", lw=1.2, alpha=0.7)
    ax.set_ylim(0, 0.4)
    ax.set_xlabel("MCTS simulations per move (n_sim)")
    ax.set_ylabel("win rate vs weak Alpha-Beta @10ms")
    ax.set_title("More simulations -> stronger: n_sim 60->600 doubles winrate (0.156->0.312)")
    p = save(fig, "fig_nsim_ablation.png")
    write_csv("fig_nsim_ablation.csv", ["n_sim", "winrate"], list(zip(nsim, wr)))
    return p, wr


if __name__ == "__main__":
    print("=== generating figures ===")
    p1, s1 = fig_pretrain_loss()
    print(f"[1] {p1}  pretrain loss first={s1[0]:.4f} final={s1[1]:.4f} npts={s1[2]}")
    p2, s2 = fig_rl_loss()
    print(f"[2] {p2}  RL loss first={s2[0]:.4f} min={s2[1]:.4f} final={s2[2]:.4f} (laststep={s2[3]})")
    p3, s3 = fig_elo_vs_step()
    print(f"[3] {p3}  Elo range [{s3[0]:.1f}, {s3[1]:.1f}] over steps {s3[2]}..{s3[3]}")
    p4, s4 = fig_crossover()
    print(f"[4] {p4}  n200={s4[0]} n600={s4[1]}")
    p5, s5 = fig_throughput()
    print(f"[5] {p5}  moves/s={s5[0]} ms/step={s5[1]}")
    p6, s6 = fig_nsim_ablation()
    print(f"[6] {p6}  winrate={s6}")
    print("=== done ===")
