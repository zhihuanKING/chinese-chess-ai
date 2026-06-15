#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Final external-anchor figures for the strength-push sprint (REAL logs only).

Reads logs/final/r1final_*.csv (128 games/anchor, eval_anchor_vec, n_sim=200)
and emits two report-ready PNGs + a tidy summary CSV under logs/fig/:
  fig_anchor_final.png  - final Elo (and winrate) vs every external anchor
  fig_sprint_ladder.png - 3-stage Elo progression vs AB-200ms & Pikafish-d1

Stage progression (base/round-1) sourced from the day1 sprint judgment
(commit 35d05fc): "AB200 -314->-198, pf1 -411->-293"; final = 128-game终测.
No fabricated numbers. Run: .venv/bin/python scripts/fig_anchor_final.py
"""
import csv
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "/mnt/nvme3n1/gameTheory"
LOGS = os.path.join(ROOT, "logs")
FIG = os.path.join(LOGS, "fig")
os.makedirs(FIG, exist_ok=True)
DPI = 140

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

# nice display label + plotting order for each opponent tag
ORDER = [
    ("random", "Random"),
    ("xqab:200", "AlphaBeta 200ms"),
    ("xqab:500", "AlphaBeta 500ms"),
    ("pikafish:1", "Pikafish d1"),
    ("pikafish:2", "Pikafish d2"),
    ("pikafish:3", "Pikafish d3"),
    ("pikafish:4", "Pikafish d4"),
]

# v1->r1 ΔElo per engine anchor, from docx/棋力提升_最终结论.md (authoritative
# final conclusion, 2026-06-15). v1 baseline Elo = r1 Elo - delta.
V1_TO_R1_DELTA = {
    "xqab:200": 144, "xqab:500": 251, "pikafish:1": 173,
    "pikafish:2": 94, "pikafish:3": 204, "pikafish:4": 355,
}


def load_final():
    rows = {}
    for path in glob.glob(os.path.join(LOGS, "final", "r1final_*.csv")):
        with open(path) as f:
            for r in csv.DictReader(f):
                rows[r["opponent"]] = {
                    "winrate": float(r["winrate"]),
                    "elo": float(r["elo_diff"]),
                    "lo": float(r["lo"]),
                    "hi": float(r["hi"]),
                    "W": int(r["W"]), "D": int(r["D"]), "L": int(r["L"]),
                    "games": int(r["games"]),
                }
    return rows


def fig_final_bars(data):
    """Horizontal bars: final Elo vs each anchor (random capped for scale)."""
    labels, elos, wrs = [], [], []
    for tag, disp in ORDER:
        if tag not in data:
            continue
        labels.append(disp)
        # cap +1600 random sanity bar so the informative anchors stay readable
        elos.append(min(data[tag]["elo"], 250.0))
        wrs.append(data[tag]["winrate"])
    y = range(len(labels))
    fig, ax = plt.subplots()
    colors = ["#4c9f70" if e >= 0 else "#c0504d" for e in elos]
    bars = ax.barh(list(y), elos, color=colors, alpha=0.85)
    ax.axvline(0, color="grey", lw=1)
    ax.set_yticks(list(y))
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("Relative Elo vs anchor (final model, 128 games, n_sim=200)")
    ax.set_title("Final strength vs external anchors (higher = stronger)")
    for i, (tag, disp) in enumerate([p for p in ORDER if p[0] in data]):
        d = data[tag]
        note = f"{d['winrate']*100:.0f}%  ({d['W']}-{d['D']}-{d['L']})"
        if tag == "random":
            note = f"100% ({d['W']}-{d['D']}-{d['L']})  Elo +1600"
        xpos = elos[i] + (8 if elos[i] >= 0 else -8)
        ha = "left" if elos[i] >= 0 else "right"
        ax.text(xpos, i, note, va="center", ha=ha, fontsize=9)
    ax.margins(x=0.18)
    path = os.path.join(FIG, "fig_anchor_final.png")
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return path


def fig_sprint_ladder(data):
    """3-stage Elo progression vs AB-200ms and Pikafish-d1."""
    stages = ["v1 cold-start\n(base)", "Round-1\n(DAgger fix)", "Final\n(128 games)"]
    ab = [-314, -198, data["xqab:200"]["elo"]]      # final from log
    pf = [-411, -293, data["pikafish:1"]["elo"]]
    x = range(len(stages))
    fig, ax = plt.subplots()
    ax.plot(list(x), ab, "o-", color="#1f6fb4", lw=2.2, ms=8, label="vs AlphaBeta 200ms")
    ax.plot(list(x), pf, "s-", color="#d08522", lw=2.2, ms=8, label="vs Pikafish d1")
    for xi, v in zip(x, ab):
        ax.annotate(f"{v:+.0f}", (xi, v), textcoords="offset points",
                    xytext=(0, 10), ha="center", fontsize=9, color="#1f6fb4")
    for xi, v in zip(x, pf):
        ax.annotate(f"{v:+.0f}", (xi, v), textcoords="offset points",
                    xytext=(0, -16), ha="center", fontsize=9, color="#d08522")
    ax.set_xticks(list(x))
    ax.set_xticklabels(stages)
    ax.set_ylabel("Relative Elo vs anchor")
    ax.set_title("Strength-push progression: external-anchor Elo across 3 stages")
    ax.legend(loc="lower right")
    ax.text(0.02, 0.04,
            f"net gain  AB-200ms: {ab[-1]-ab[0]:+.0f} Elo   "
            f"Pikafish-d1: {pf[-1]-pf[0]:+.0f} Elo",
            transform=ax.transAxes, fontsize=9,
            bbox=dict(boxstyle="round", fc="#fff4cc", ec="grey", alpha=0.9))
    path = os.path.join(FIG, "fig_sprint_ladder.png")
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return path


def fig_v1_vs_r1(data):
    """Grouped bars: v1 cold-start baseline vs r1 (DAgger) per engine anchor.
    Headline: distillation + DAgger lifts strength by +203 Elo on average."""
    engines = [(t, d) for t, d in ORDER if t in V1_TO_R1_DELTA and t in data]
    labels = [d for _, d in engines]
    r1 = [data[t]["elo"] for t, _ in engines]
    v1 = [data[t]["elo"] - V1_TO_R1_DELTA[t] for t, _ in engines]
    deltas = [V1_TO_R1_DELTA[t] for t, _ in engines]
    avg = sum(deltas) / len(deltas)
    x = range(len(labels))
    w = 0.38
    fig, ax = plt.subplots()
    ax.bar([i - w / 2 for i in x], v1, w, label="v1 cold-start (base)",
           color="#b0b7c0")
    ax.bar([i + w / 2 for i in x], r1, w, label="r1 (DAgger distill)",
           color="#1f6fb4")
    ax.axhline(0, color="grey", lw=1)
    for i, dlt in enumerate(deltas):
        top = max(v1[i], r1[i])
        ax.annotate(f"+{dlt}", (i, top), textcoords="offset points",
                    xytext=(0, 6), ha="center", fontsize=9, color="#22863a",
                    fontweight="bold")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("Relative Elo vs anchor")
    ax.set_title("Strength gain from supervised distillation + DAgger "
                 f"(avg +{int(avg)} Elo over 6 engine anchors)")
    ax.legend(loc="lower left")
    path = os.path.join(FIG, "fig_v1_vs_r1.png")
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return path


def dump_csv(data):
    path = os.path.join(FIG, "fig_anchor_final.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["anchor", "games", "winrate", "elo", "lo", "hi", "W", "D", "L"])
        for tag, _ in ORDER:
            if tag not in data:
                continue
            d = data[tag]
            w.writerow([tag, d["games"], f"{d['winrate']:.4f}", f"{d['elo']:.1f}",
                        f"{d['lo']:.1f}", f"{d['hi']:.1f}", d["W"], d["D"], d["L"]])
    return path


def main():
    data = load_final()
    if not data:
        raise SystemExit("no logs/final/r1final_*.csv found")
    p1 = fig_final_bars(data)
    p2 = fig_sprint_ladder(data)
    p3 = fig_v1_vs_r1(data)
    p4 = dump_csv(data)
    for p in (p1, p2, p3, p4):
        print("wrote", p)


if __name__ == "__main__":
    main()
