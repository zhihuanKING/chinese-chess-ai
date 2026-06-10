#!/usr/bin/env python
"""Generate course-report architecture / structure diagrams as PNGs.

Pure-matplotlib (patches + annotate, Agg backend). No runtime data needed;
these are static "schematic" diagrams of the project's real architecture, drawn
from xqai/network.py, xqai/selfplay.py, scripts/train_distributed.py and the
default config. All text is English (Chinese fonts garble easily).

Outputs (logs/fig/, dpi=150):
  fig_arch_network.png        Fig 4.1  policy-value dual-head network
  fig_arch_actor_learner.png  Fig 4.2  actor-learner self-play training loop
  fig_arch_system.png         Fig 5.1  layered system / cross-language modules
  fig_topology_8gpu.png       Fig 5.2  dedicated 8x A800 distributed topology
"""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs", "fig")

# ---- gentle palette -------------------------------------------------------- #
C_INPUT = "#cfe8f3"   # input / data
C_STEM = "#dde7c7"    # stem / conv
C_RES = "#f5e1a4"     # residual tower
C_POL = "#f6c6c6"     # policy head
C_VAL = "#c9d8f0"     # value head
C_ACTOR = "#d8e9d0"   # actors
C_LEARN = "#f3d6b3"   # learner
C_BUF = "#e7dbf0"     # buffer
C_EVAL = "#f0e0c0"    # eval / gating
C_CPP = "#cdb4db"     # C++ layer
C_PY = "#a2d2ff"      # python layer
C_GPU = "#bde0fe"     # gpu
C_MEM = "#ffd6a5"     # memory / nvme
C_EDGE = "#3a3a3a"
C_TEXT = "#1a1a1a"

plt.rcParams.update({
    "font.size": 11,
    "font.family": "DejaVu Sans",
    "axes.edgecolor": C_EDGE,
})


def _box(ax, x, y, w, h, text, fc, *, fontsize=11, ec=C_EDGE, lw=1.4,
         style="round,pad=0.02,rounding_size=0.06", weight="normal", text_color=C_TEXT):
    """Draw a rounded box centered at (x, y) with wrapped text."""
    box = FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle=style, linewidth=lw, edgecolor=ec, facecolor=fc, zorder=2,
    )
    ax.add_patch(box)
    ax.text(x, y, text, ha="center", va="center", fontsize=fontsize,
            color=text_color, zorder=3, weight=weight, wrap=True)
    return (x, y, w, h)


def _arrow(ax, p0, p1, *, label=None, color=C_EDGE, lw=1.8, rad=0.0,
           ls="-", label_off=(0, 0), fontsize=9, label_color=None):
    """Draw an annotated arrow from point p0 to p1."""
    arr = FancyArrowPatch(
        p0, p1, arrowstyle="-|>", mutation_scale=16, linewidth=lw,
        color=color, connectionstyle=f"arc3,rad={rad}", linestyle=ls, zorder=1,
    )
    ax.add_patch(arr)
    if label:
        mx = (p0[0] + p1[0]) / 2 + label_off[0]
        my = (p0[1] + p1[1]) / 2 + label_off[1]
        ax.text(mx, my, label, ha="center", va="center", fontsize=fontsize,
                color=label_color or color, zorder=4,
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85))


def _save(fig, name):
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


# =========================================================================== #
# Fig 4.1  Policy-Value dual-head network                                     #
# =========================================================================== #
def fig_arch_network():
    fig, ax = plt.subplots(figsize=(14, 9))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 9)
    ax.axis("off")
    ax.set_title("Fig 4.1  Policy-Value Dual-Head Residual Network (PVNet)",
                 fontsize=15, weight="bold", pad=14)

    # --- main vertical/diagonal trunk on the left/center ---
    bw, bh = 2.7, 0.78
    cx = 3.2
    _box(ax, cx, 8.0, bw, bh, "Input\n15 x 10 x 9 planes", C_INPUT, fontsize=10.5)
    _box(ax, cx, 6.85, bw, bh, "Stem: Conv 3x3 (C) + BN + ReLU", C_STEM, fontsize=10.5)
    res = _box(ax, cx, 5.45, bw, 1.1,
               "Residual Tower\nN x Residual Block", C_RES, fontsize=11, weight="bold")

    _arrow(ax, (cx, 8.0 - bh / 2), (cx, 6.85 + bh / 2))
    _arrow(ax, (cx, 6.85 - bh / 2), (cx, 5.45 + 0.55))

    # fork point
    fork_y = 5.45 - 0.55
    _arrow(ax, (cx, fork_y), (cx, 4.2), lw=1.4)
    ax.text(cx, 4.35, "shared features  [B, C, 10, 9]", ha="center", va="center",
            fontsize=9, color="#444")

    # --- policy head (left branch) ---
    px = 1.9
    py_top = 3.7
    _arrow(ax, (cx, 4.2), (px, py_top + 0.35), rad=0.15)
    pbw, pbh = 3.0, 0.62
    p1 = _box(ax, px, py_top, pbw, pbh, "Policy head:  Conv 1x1 + BN + ReLU", C_POL, fontsize=9.5)
    p2 = _box(ax, px, py_top - 0.95, pbw, pbh, "Flatten -> Linear", C_POL, fontsize=9.5)
    p3 = _box(ax, px, py_top - 1.9, pbw, pbh, "8100 logits", C_POL, fontsize=10, weight="bold")
    p4 = _box(ax, px, py_top - 2.95, pbw, 0.78,
              "+ legal mask\n-> softmax  -> pi", C_POL, fontsize=9.5)
    _arrow(ax, (px, py_top - pbh / 2), (px, py_top - 0.95 + pbh / 2), lw=1.4)
    _arrow(ax, (px, py_top - 0.95 - pbh / 2), (px, py_top - 1.9 + pbh / 2), lw=1.4)
    _arrow(ax, (px, py_top - 1.9 - pbh / 2), (px, py_top - 2.95 + 0.39), lw=1.4)

    # --- value head (right branch) ---
    vx = 5.5
    _arrow(ax, (cx, 4.2), (vx, py_top + 0.35), rad=-0.15)
    vbw, vbh = 3.0, 0.62
    _box(ax, vx, py_top, vbw, vbh, "Value head:  Conv 1x1 + BN + ReLU", C_VAL, fontsize=9.5)
    _box(ax, vx, py_top - 0.95, vbw, vbh, "Flatten -> Linear(C) -> ReLU", C_VAL, fontsize=9.5)
    _box(ax, vx, py_top - 1.9, vbw, vbh, "Linear(1)", C_VAL, fontsize=10, weight="bold")
    _box(ax, vx, py_top - 2.95, vbw, 0.78, "tanh\n-> v in [-1, 1]", C_VAL, fontsize=9.5)
    _arrow(ax, (vx, py_top - vbh / 2), (vx, py_top - 0.95 + vbh / 2), lw=1.4)
    _arrow(ax, (vx, py_top - 0.95 - vbh / 2), (vx, py_top - 1.9 + vbh / 2), lw=1.4)
    _arrow(ax, (vx, py_top - 1.9 - vbh / 2), (vx, py_top - 2.95 + 0.39), lw=1.4)

    # --- residual block inset (right side) ---
    ix = 10.6
    iy = 6.6
    iw = 4.6
    ih = 5.6
    inset = FancyBboxPatch((ix - iw / 2, iy - ih / 2), iw, ih,
                           boxstyle="round,pad=0.05,rounding_size=0.1",
                           linewidth=1.6, edgecolor="#888", facecolor="#fbfbf4",
                           linestyle="--", zorder=1)
    ax.add_patch(inset)
    ax.text(ix, iy + ih / 2 - 0.32, "Residual Block (internals)",
            ha="center", va="center", fontsize=11, weight="bold", color="#333")

    rbw, rbh = 2.6, 0.55
    ry = iy + ih / 2 - 1.0
    gap = 0.82
    steps = ["Conv 3x3", "BN", "ReLU", "Conv 3x3", "BN"]
    ys = []
    for i, s in enumerate(steps):
        yy = ry - i * gap
        ys.append(yy)
        _box(ax, ix, yy, rbw, rbh, s, C_RES, fontsize=9.5)
        if i > 0:
            _arrow(ax, (ix, ys[i - 1] - rbh / 2), (ix, yy + rbh / 2), lw=1.3)
    # add (+skip) box then ReLU
    add_y = ys[-1] - gap
    _box(ax, ix, add_y, rbw, rbh, "(+) skip connection", C_RES, fontsize=9.5, weight="bold")
    _arrow(ax, (ix, ys[-1] - rbh / 2), (ix, add_y + rbh / 2), lw=1.3)
    relu_y = add_y - gap
    _box(ax, ix, relu_y, rbw, rbh, "ReLU", C_RES, fontsize=9.5)
    _arrow(ax, (ix, add_y - rbh / 2), (ix, relu_y + rbh / 2), lw=1.3)
    # skip arrow curving from top input down to the add node
    skip = FancyArrowPatch((ix + rbw / 2 + 0.15, ys[0] + 0.15),
                           (ix + rbw / 2 + 0.15, add_y),
                           arrowstyle="-|>", mutation_scale=14, linewidth=1.6,
                           color="#b5651d", connectionstyle="arc3,rad=-0.55", zorder=1)
    ax.add_patch(skip)
    ax.text(ix + rbw / 2 + 1.05, (ys[0] + add_y) / 2, "skip", rotation=90,
            ha="center", va="center", fontsize=9, color="#b5651d")

    # link tower box to inset
    _arrow(ax, (res[0] + bw / 2, 5.45), (ix - iw / 2, 5.45), color="#888",
           ls="--", lw=1.4, label="expand", fontsize=9)

    # --- two model sizes annotation (placed clear of the head columns) ---
    note = ("Two model scales\n"
            "main:  128 ch x 10 blocks  (~26.4M params)\n"
            "final: 256 ch x 15 blocks  (~41.3M params)")
    ax.text(10.6, 1.55, note, ha="center", va="center", fontsize=11,
            bbox=dict(boxstyle="round,pad=0.5", fc="#fff7e0", ec="#caa84a", lw=1.4))

    return _save(fig, "fig_arch_network.png")


# =========================================================================== #
# Fig 4.2  Actor-learner self-play training loop                              #
# =========================================================================== #
def fig_arch_actor_learner():
    fig, ax = plt.subplots(figsize=(14, 8.5))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8.5)
    ax.axis("off")
    ax.set_title("Fig 4.2  Actor-Learner Self-Play Training Loop",
                 fontsize=15, weight="bold", pad=14)

    # Actors (left)
    ax_x = 2.4
    _box(ax, ax_x, 6.4, 3.6, 1.7,
         "Self-play Actors\nGPU 1-7  (5 workers / GPU)\nvectorized batched MCTS\n(PUCT / Gumbel, n_sim=32)",
         C_ACTOR, fontsize=10, weight="bold")
    # small stacked-card effect
    for dx, dy in [(0.16, -0.16), (0.32, -0.32)]:
        card = FancyBboxPatch((ax_x - 1.8 + dx, 6.4 - 0.85 + dy), 3.6, 1.7,
                              boxstyle="round,pad=0.02,rounding_size=0.06",
                              linewidth=1.2, edgecolor=C_EDGE, facecolor=C_ACTOR,
                              zorder=0, alpha=0.55)
        ax.add_patch(card)

    # Replay buffer (top center)
    buf_x, buf_y = 7.0, 6.4
    _box(ax, buf_x, buf_y, 3.8, 1.5,
         "Shared ReplayBuffer\n/dev/shm  (shared memory)\ncapacity 5,000,000 positions\nrecent-weighted + mirror aug",
         C_BUF, fontsize=10, weight="bold")

    # Learner (right)
    ln_x, ln_y = 11.4, 6.4
    _box(ax, ln_x, ln_y, 3.4, 1.7,
         "Learner  (GPU 0)\nAdamW, bf16 autocast\nAlphaZero loss\n(value MSE + policy CE)",
         C_LEARN, fontsize=10, weight="bold")

    # Checkpoint export (bottom center)
    ck_x, ck_y = 7.0, 3.4
    _box(ax, ck_x, ck_y, 3.4, 1.1,
         "checkpoints/latest.pt\n(atomic tmp + os.replace)", C_MEM, fontsize=10, weight="bold")

    # Eval / gating (bottom)
    ev_x, ev_y = 7.0, 1.3
    _box(ax, ev_x, ev_y, 4.6, 1.0,
         "Elo-gating evaluation -> promote strongest weights",
         C_EVAL, fontsize=10)

    # --- arrows forming the closed loop ---
    # actors -> buffer  (samples)
    _arrow(ax, (ax_x + 1.8, 6.8), (buf_x - 1.9, 6.8), label="samples\n(planes, pi, z)",
           color="#2e7d32", lw=2.2, label_off=(0, 0.55), fontsize=9)
    # buffer -> learner (sample minibatch)
    _arrow(ax, (buf_x + 1.9, 6.4), (ln_x - 1.7, 6.4), label="sample\nminibatch",
           color="#1565c0", lw=2.2, label_off=(0, 0.55), fontsize=9)
    # learner -> checkpoint (export weights)
    _arrow(ax, (ln_x, ln_y - 0.85), (ck_x + 1.7, ck_y), label="export weights\nevery N steps",
           color="#b5651d", lw=2.2, rad=-0.2, label_off=(1.4, 0.5), fontsize=9)
    # checkpoint -> actors (hot reload, poll mtime)  -> closes the loop
    _arrow(ax, (ck_x - 1.7, ck_y), (ax_x, 6.4 - 0.85),
           label="hot reload\n(poll mtime)", color="#6a1b9a", lw=2.2, rad=-0.2,
           label_off=(-1.5, 0.4), fontsize=9)

    # eval bypass: checkpoint <-> eval
    _arrow(ax, (ck_x, ck_y - 0.55), (ev_x, ev_y + 0.5), label="best.pt", color="#555",
           lw=1.6, ls="--", fontsize=9, label_off=(1.0, 0))

    ax.text(7.0, 0.35,
            "Decoupled actor-learner: actors generate data continuously; "
            "the single learner trains and periodically publishes new weights.",
            ha="center", va="center", fontsize=9.5, color="#555")

    return _save(fig, "fig_arch_actor_learner.png")


# =========================================================================== #
# Fig 5.1  Layered system / cross-language modules                            #
# =========================================================================== #
def fig_arch_system():
    fig, ax = plt.subplots(figsize=(13.5, 9))
    ax.set_xlim(0, 13.5)
    ax.set_ylim(0, 9)
    ax.axis("off")
    ax.set_title("Fig 5.1  Layered System Architecture (cross-language boundary)",
                 fontsize=15, weight="bold", pad=14)

    # ---- Python + PyTorch layer (top) ----
    py_band = FancyBboxPatch((0.5, 5.7), 12.5, 2.9,
                             boxstyle="round,pad=0.04,rounding_size=0.1",
                             linewidth=1.8, edgecolor="#3a6ea5", facecolor="#eaf3fc", zorder=0)
    ax.add_patch(py_band)
    ax.text(0.85, 8.35, "Python + PyTorch layer", ha="left", va="center",
            fontsize=12.5, weight="bold", color="#1f4e79")

    py_mods = ["encoding", "PVNet", "vectorized MCTS\n(PUCT + Gumbel)",
               "selfplay", "replay", "train", "arena", "eval"]
    n = len(py_mods)
    mw = 1.35
    x0 = 1.05
    span = 11.9
    step = (span - mw) / (n - 1)
    for i, m in enumerate(py_mods):
        _box(ax, x0 + mw / 2 + i * step, 6.7, mw, 1.15, m, C_PY, fontsize=9)

    # ---- pybind11 boundary ----
    bnd_y = 4.9
    band = FancyBboxPatch((0.5, bnd_y - 0.42), 12.5, 0.84,
                          boxstyle="round,pad=0.02,rounding_size=0.08",
                          linewidth=1.6, edgecolor="#7a5195", facecolor="#efe6f6", zorder=1)
    ax.add_patch(band)
    ax.text(6.75, bnd_y, "pybind11 boundary  ->  xqai._xqcore  (compiled extension)",
            ha="center", va="center", fontsize=11.5, weight="bold", color="#5a3d77")

    # ---- C++17 layer (bottom) ----
    cpp_band = FancyBboxPatch((0.5, 1.3), 12.5, 3.0,
                              boxstyle="round,pad=0.04,rounding_size=0.1",
                              linewidth=1.8, edgecolor="#6a4f9c", facecolor="#f1ebf8", zorder=0)
    ax.add_patch(cpp_band)
    ax.text(0.85, 4.05, "C++17 layer", ha="left", va="center",
            fontsize=12.5, weight="bold", color="#4a2f7c")

    _box(ax, 3.7, 2.6, 5.0, 1.7,
         "Rules kernel\nmoves / terminal / Zobrist / FEN\n(perft 1-5 verified)",
         C_CPP, fontsize=10, weight="bold")
    _box(ax, 9.7, 2.6, 5.4, 1.7,
         "Alpha-Beta engine\nnegamax + alpha-beta + TT\niterative deepening +\nquiescence + MVV-LVA",
         C_CPP, fontsize=10, weight="bold")

    # arrows across boundary (data both ways)
    _arrow(ax, (6.75, 6.1), (6.75, bnd_y + 0.45), color="#444", lw=2.0, rad=0.0)
    _arrow(ax, (6.75, bnd_y - 0.45), (6.75, 3.5), color="#444", lw=2.0, rad=0.0)
    ax.text(7.05, 5.55, "Position / legal moves / result", ha="left", va="center",
            fontsize=8.5, color="#444")

    # ---- only cross-language contract callout ----
    ax.text(6.75, 0.55,
            "Sole cross-language contract = checkpoint weight file "
            "(latest.pt / best.pt); no LibTorch C++ inference service.",
            ha="center", va="center", fontsize=10.5, weight="bold", color="#8a3b00",
            bbox=dict(boxstyle="round,pad=0.4", fc="#fff2e0", ec="#d2843b", lw=1.5))

    return _save(fig, "fig_arch_system.png")


# =========================================================================== #
# Fig 5.2  Dedicated 8x A800 distributed topology                             #
# =========================================================================== #
def fig_topology_8gpu():
    fig, ax = plt.subplots(figsize=(14, 9))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 9)
    ax.axis("off")
    ax.set_title("Fig 5.2  Dedicated 8x A800-40GB Distributed Topology",
                 fontsize=15, weight="bold", pad=14)

    # GPU row
    gpu_y = 6.7
    gw, gh = 1.45, 1.2
    xs = [1.1 + i * 1.62 for i in range(8)]
    centers = []
    for i, gx in enumerate(xs):
        gcx = gx + gw / 2
        centers.append(gcx)
        if i == 0:
            fc, role = C_LEARN, "Learner"
        else:
            fc, role = C_GPU, "Actor"
        _box(ax, gcx, gpu_y, gw, gh, f"GPU {i}\nA800-40GB\n{role}", fc,
             fontsize=9, weight="bold")
        if i >= 1:
            ax.text(gcx, gpu_y - gh / 2 - 0.28, "5 workers", ha="center", va="center",
                    fontsize=7.5, color="#555")

    # NVLink mesh line under the GPUs
    nv_y = gpu_y + gh / 2 + 0.45
    ax.plot([centers[0], centers[-1]], [nv_y, nv_y], color="#2e7d32", lw=3, zorder=1)
    for c in centers:
        ax.plot([c, c], [gpu_y + gh / 2, nv_y], color="#2e7d32", lw=2, zorder=1)
        ax.plot(c, nv_y, "o", color="#2e7d32", ms=5, zorder=2)
    ax.text(centers[-1] + 0.25, nv_y, "NVLink", ha="left", va="center",
            fontsize=10.5, weight="bold", color="#2e7d32")

    # Roles legend
    ax.text(centers[0], gpu_y + gh / 2 + 1.05, "Learner", ha="center", va="center",
            fontsize=10, weight="bold", color="#b5651d",
            bbox=dict(boxstyle="round,pad=0.2", fc="#fbe3c6", ec="#b5651d"))
    ax.text((centers[1] + centers[-1]) / 2, gpu_y + gh / 2 + 1.05,
            "Self-play actors (GPU 1-7)", ha="center", va="center",
            fontsize=10, weight="bold", color="#1565c0",
            bbox=dict(boxstyle="round,pad=0.2", fc="#d6e7fb", ec="#1565c0"))

    # ReplayBuffer (shared memory) center
    buf_x, buf_y = 7.0, 3.9
    _box(ax, buf_x, buf_y, 6.5, 1.2,
         "ReplayBuffer  ->  /dev/shm  (shared memory, 5,000,000 positions)",
         C_BUF, fontsize=10.5, weight="bold")

    # arrows: all actors write to buffer, learner reads from buffer
    for c in centers[1:]:
        _arrow(ax, (c, gpu_y - gh / 2), (buf_x - 2.0 + (c / centers[-1]) * 4.0, buf_y + 0.6),
               color="#2e7d32", lw=1.3, rad=0.05)
    # learner read/write
    _arrow(ax, (buf_x - 2.6, buf_y), (centers[0], gpu_y - gh / 2),
           color="#1565c0", lw=2.0, rad=-0.2, label="sample", fontsize=9,
           label_off=(-0.6, 0.6))
    ax.text(buf_x, buf_y + 0.95, "samples in  /  minibatches out",
            ha="center", va="center", fontsize=8.5, color="#555")

    # NVMe storage
    _box(ax, 3.0, 1.7, 4.6, 1.1,
         "NVMe  /mnt/nvme3n1  (1.7 TB)\ndata + checkpoints", C_MEM, fontsize=10, weight="bold")
    # CPU
    _box(ax, 10.5, 1.7, 5.0, 1.1,
         "CPU: 2x EPYC 7K62 / 192 threads\nPython MCTS tree ops", C_STEM, fontsize=10, weight="bold")

    # buffer <-> nvme (checkpoint persistence), buffer<->cpu (tree ops feed)
    _arrow(ax, (buf_x - 1.6, buf_y - 0.6), (3.0, 1.7 + 0.55), color="#555",
           lw=1.6, ls="--", rad=0.15, label="persist ckpt", fontsize=8.5, label_off=(-0.8, 0.3))
    _arrow(ax, (10.5, 1.7 + 0.55), (buf_x + 1.6, buf_y - 0.6), color="#555",
           lw=1.6, ls="--", rad=0.15, label="tree ops", fontsize=8.5, label_off=(0.9, 0.3))

    ax.text(7.0, 0.5,
            "Dedicated single node: NVLink-coupled GPUs (green) move weights/data; "
            "shared-memory buffer decouples actors from the learner.",
            ha="center", va="center", fontsize=9.5, color="#555")

    return _save(fig, "fig_topology_8gpu.png")


def main():
    paths = [
        fig_arch_network(),
        fig_arch_actor_learner(),
        fig_arch_system(),
        fig_topology_8gpu(),
    ]
    for p in paths:
        print(p)


if __name__ == "__main__":
    main()
