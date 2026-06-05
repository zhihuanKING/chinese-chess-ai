#!/usr/bin/env python
"""Unattended Elo learning-curve evaluator for xqai (INTERFACES.md §9, task #8).

Watches ``checkpoints/latest.pt`` for new exports (mtime). On every new export it
loads the weights into a :class:`xqai.network.PVNet`, wraps it as a
:class:`xqai.arena.NetPlayer` (PVNet + :class:`xqai.mcts.PUCTPlanner`), and plays
``--games`` paired-opening games (red/black each half) against a **fixed
baseline** via :func:`xqai.arena.play_match`. The baseline is the C++ ``xqab``
Alpha-Beta engine by default (UCCI subprocess, fixed ``--baseline-ms`` movetime);
``--baseline pikafish`` switches to the Pikafish engine.

For every evaluation it appends one row to ``logs/elo_curve.csv`` with columns::

    wall_time, ckpt_mtime, train_step, games, winrate, elo, elo_lo, elo_hi

(``train_step`` is read from the checkpoint's ``step`` field if present, else -1;
``elo_lo/elo_hi`` are the logistic Elo of the Wilson CI bounds on the score rate.)

Exit conditions: the STOP file exists (default
``/mnt/nvme3n1/gameTheory/.pipeline_stop``) or ``--duration`` seconds elapse.
A single failed evaluation is logged and skipped; the loop never crashes.

At the end it optionally renders ``logs/elo_curve.png`` (train_step -> Elo) with
matplotlib if available.

This script only *reads* the existing ``xqai`` modules; it never modifies them.

Examples
--------
Smoke (one round, 6 games vs xqab, low n_sim, prints + writes a csv row)::

    .venv/bin/python scripts/eval_loop.py --smoke

Unattended (poll every 5 min, 40 games vs xqab @200ms, run 12h)::

    .venv/bin/python scripts/eval_loop.py --interval 300 --games 40 \
        --n-sim 200 --baseline xqab --baseline-ms 200 --duration 43200
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time

import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from xqai.arena import NetPlayer, SubprocessUCCIPlayer, elo_from_winrate, play_match
from xqai.config import load_config

# Default STOP-file path (kept in sync with run_pipeline.sh / configs convention).
DEFAULT_STOP_FILE = os.path.join(_ROOT, ".pipeline_stop")
DEFAULT_CKPT = os.path.join(_ROOT, "checkpoints", "latest.pt")
DEFAULT_CSV = os.path.join(_ROOT, "logs", "elo_curve.csv")

CSV_HEADER = ["wall_time", "ckpt_mtime", "train_step", "games",
              "winrate", "elo", "elo_lo", "elo_hi"]


# --------------------------------------------------------------------------- #
# Baseline engine construction                                                #
# --------------------------------------------------------------------------- #
def _baseline_player(kind: str, movetime_ms: int) -> SubprocessUCCIPlayer:
    """Build the fixed baseline UCCI subprocess player."""
    kind = kind.lower()
    if kind == "xqab":
        cmd = [os.path.join(_ROOT, "cpp", "build", "xqab")]
        name = f"xqab@{movetime_ms}ms"
    elif kind == "pikafish":
        cmd = [os.path.join(_ROOT, "third_party", "Pikafish", "src", "pikafish")]
        name = f"pikafish@{movetime_ms}ms"
    else:
        raise ValueError(f"unknown baseline {kind!r} (want 'xqab' or 'pikafish')")
    if not os.path.exists(cmd[0]):
        raise FileNotFoundError(f"baseline engine not found: {cmd[0]}")
    return SubprocessUCCIPlayer(cmd, movetime_ms=movetime_ms, name=name)


# --------------------------------------------------------------------------- #
# Net player from checkpoint                                                  #
# --------------------------------------------------------------------------- #
def _load_net_player(ckpt_path: str, cfg_d: dict, n_sim: int, device) -> tuple[NetPlayer, int]:
    """Load ``ckpt_path`` into a fresh PVNet+PUCT NetPlayer. Returns (player, step)."""
    from xqai.mcts import PUCTPlanner
    from xqai.network import PVNet

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and "model" in ckpt:
        state = ckpt["model"]
        channels = int(ckpt.get("channels", cfg_d["network"]["channels"]))
        blocks = int(ckpt.get("blocks", cfg_d["network"]["blocks"]))
        step = int(ckpt.get("step", ckpt.get("epoch", -1)))
    else:
        state = ckpt
        channels = cfg_d["network"]["channels"]
        blocks = cfg_d["network"]["blocks"]
        step = -1

    net = PVNet(channels=channels, blocks=blocks).to(device).eval()
    net.load_state_dict(state)

    mc = cfg_d["mcts"]
    # No Dirichlet root noise during evaluation: deterministic, strongest play.
    planner = PUCTPlanner(
        c_puct=mc["c_puct"],
        dirichlet_alpha=mc["dirichlet_alpha"],
        dirichlet_eps=mc["dirichlet_eps"],
        virtual_loss=mc["virtual_loss"],
        add_noise=False,
        seed=12345,
    )
    player = NetPlayer(net, planner, n_sim=n_sim, name=f"net@step{step}")
    return player, step


# --------------------------------------------------------------------------- #
# One evaluation                                                              #
# --------------------------------------------------------------------------- #
def _evaluate_once(ckpt_path: str, ckpt_mtime: float, cfg_d: dict, *, games: int,
                   n_sim: int, baseline: str, baseline_ms: int, device,
                   csv_path: str) -> dict | None:
    """Run one full match vs the fixed baseline and append a CSV row.

    Returns the result dict, or ``None`` on failure (already logged).
    """
    player = baseline_player = None
    try:
        player, step = _load_net_player(ckpt_path, cfg_d, n_sim, device)
        baseline_player = _baseline_player(baseline, baseline_ms)
        # A vs B from A's (the net's) perspective.
        res = play_match(player, baseline_player, games=games)

        winrate = res["a_score_rate"]
        lo, hi = res["ci95"]
        row = {
            "wall_time": f"{time.time():.0f}",
            "ckpt_mtime": f"{ckpt_mtime:.0f}",
            "train_step": step,
            "games": res["games"],
            "winrate": f"{winrate:.4f}",
            "elo": f"{elo_from_winrate(winrate):.1f}",
            "elo_lo": f"{elo_from_winrate(lo):.1f}",
            "elo_hi": f"{elo_from_winrate(hi):.1f}",
        }
        _append_csv(csv_path, row)
        print(f"[eval] step={step} games={res['games']} "
              f"W/D/L={res['a_wins']}/{res['a_draws']}/{res['a_losses']} "
              f"winrate={winrate:.3f} elo={row['elo']} "
              f"CI[{row['elo_lo']},{row['elo_hi']}] vs {baseline_player.name}",
              flush=True)
        return {**res, "train_step": step, "row": row}
    except Exception as exc:  # never let a single eval kill the loop
        print(f"[eval] evaluation FAILED (continuing): {exc!r}", flush=True)
        return None
    finally:
        # Tear down the engine subprocess; the net player needs no cleanup.
        if baseline_player is not None:
            try:
                baseline_player.close()
            except Exception:
                pass


def _append_csv(csv_path: str, row: dict) -> None:
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    new = not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_HEADER)
        if new:
            w.writeheader()
        w.writerow(row)


# --------------------------------------------------------------------------- #
# Plot                                                                        #
# --------------------------------------------------------------------------- #
def _plot_curve(csv_path: str, png_path: str) -> bool:
    """Render train_step -> Elo (with CI band) to ``png_path``. Best effort."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[eval] matplotlib unavailable, skip plot: {exc!r}", flush=True)
        return False
    if not os.path.exists(csv_path):
        return False
    steps, elos, los, his = [], [], [], []
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            try:
                steps.append(int(r["train_step"]))
                elos.append(float(r["elo"]))
                los.append(float(r["elo_lo"]))
                his.append(float(r["elo_hi"]))
            except (ValueError, KeyError):
                continue
    if not steps:
        return False
    # Sort by step for a clean monotone x-axis.
    order = sorted(range(len(steps)), key=lambda i: steps[i])
    steps = [steps[i] for i in order]
    elos = [elos[i] for i in order]
    los = [los[i] for i in order]
    his = [his[i] for i in order]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(steps, elos, "-o", color="C0", label="Elo vs baseline")
    ax.fill_between(steps, los, his, color="C0", alpha=0.2, label="95% CI")
    ax.axhline(0.0, color="grey", ls="--", lw=0.8)
    ax.set_xlabel("train step")
    ax.set_ylabel("Elo (logistic, vs fixed baseline)")
    ax.set_title("xqai RL learning curve")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    os.makedirs(os.path.dirname(png_path), exist_ok=True)
    fig.savefig(png_path, dpi=120)
    plt.close(fig)
    print(f"[eval] wrote plot {png_path}", flush=True)
    return True


# --------------------------------------------------------------------------- #
# Main                                                                        #
# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description="Unattended Elo learning-curve evaluator")
    ap.add_argument("--config", default=None, help="YAML (default configs/default.yaml)")
    ap.add_argument("--ckpt", default=DEFAULT_CKPT, help="checkpoint to watch")
    ap.add_argument("--interval", type=float, default=300.0,
                    help="seconds between mtime checks")
    ap.add_argument("--games", type=int, default=40, help="games per evaluation")
    ap.add_argument("--n-sim", type=int, default=200, help="MCTS sims per move for the net")
    ap.add_argument("--baseline", default="xqab", choices=["xqab", "pikafish"])
    ap.add_argument("--baseline-ms", type=int, default=200, help="baseline movetime ms")
    ap.add_argument("--duration", type=float, default=None,
                    help="max wall-clock seconds before exiting (default: until STOP)")
    ap.add_argument("--stop-file", default=DEFAULT_STOP_FILE)
    ap.add_argument("--out", default=DEFAULT_CSV, help="output CSV path")
    ap.add_argument("--plot", action="store_true", help="render PNG on exit")
    ap.add_argument("--smoke", action="store_true",
                    help="one round, 6 games vs xqab, low n_sim; verify a CSV row is produced")
    args = ap.parse_args()

    cfg_d = load_config(args.config).to_dict()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- smoke: a single evaluation, then exit ------------------------- #
    if args.smoke:
        games = 6
        n_sim = min(args.n_sim, 16)
        baseline_ms = min(args.baseline_ms, 100)
        print(f"[eval] SMOKE: ckpt={args.ckpt} games={games} n_sim={n_sim} "
              f"baseline=xqab@{baseline_ms}ms device={device}", flush=True)
        if not os.path.exists(args.ckpt):
            print(f"[eval] SMOKE FAIL: checkpoint not found: {args.ckpt}", flush=True)
            return 1
        mtime = os.path.getmtime(args.ckpt)
        res = _evaluate_once(args.ckpt, mtime, cfg_d, games=games, n_sim=n_sim,
                             baseline="xqab", baseline_ms=baseline_ms, device=device,
                             csv_path=args.out)
        ok = res is not None and os.path.exists(args.out) and os.path.getsize(args.out) > 0
        if args.plot:
            _plot_curve(args.out, os.path.splitext(args.out)[0] + ".png")
        print(f"[eval] SMOKE {'PASS' if ok else 'FAIL'} (csv={args.out})", flush=True)
        return 0 if ok else 1

    # ---- unattended watch loop ----------------------------------------- #
    print(f"[eval] watch ckpt={args.ckpt} interval={args.interval}s games={args.games} "
          f"n_sim={args.n_sim} baseline={args.baseline}@{args.baseline_ms}ms "
          f"duration={args.duration} stop={args.stop_file} device={device}", flush=True)

    t_start = time.time()
    last_mtime = 0.0
    n_evals = 0
    try:
        while True:
            if os.path.exists(args.stop_file):
                print(f"[eval] STOP file {args.stop_file} present -> exit", flush=True)
                break
            if args.duration is not None and (time.time() - t_start) >= args.duration:
                print("[eval] duration reached -> exit", flush=True)
                break

            try:
                mtime = os.path.getmtime(args.ckpt)
            except OSError:
                mtime = 0.0
            if mtime > last_mtime:
                last_mtime = mtime
                res = _evaluate_once(args.ckpt, mtime, cfg_d, games=args.games,
                                     n_sim=args.n_sim, baseline=args.baseline,
                                     baseline_ms=args.baseline_ms, device=device,
                                     csv_path=args.out)
                if res is not None:
                    n_evals += 1

            # Sleep the interval in short slices so STOP/duration is responsive.
            slept = 0.0
            while slept < args.interval:
                if os.path.exists(args.stop_file):
                    break
                if args.duration is not None and (time.time() - t_start) >= args.duration:
                    break
                step = min(2.0, args.interval - slept)
                time.sleep(step)
                slept += step
    except KeyboardInterrupt:
        print("[eval] KeyboardInterrupt -> exit", flush=True)

    if args.plot:
        _plot_curve(args.out, os.path.splitext(args.out)[0] + ".png")
    print(f"[eval] done: {n_evals} evaluations written to {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
