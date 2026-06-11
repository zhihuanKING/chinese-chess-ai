#!/usr/bin/env python
"""终点定论评测:一次性大样本对弈 ckpt vs ref(默认 100 局)。

与 eval_vs_ref.py 同口径(同开局池种子 12345、PUCT add_noise=False、配对换色),
但单次跑满 games 局后写一行 CSV 即退出 —— 用于各消融臂终点的统计定论,
弥补训练期间 12 局/轮曲线点的低统计功效。
"""
import argparse, csv, os, sys, time

import numpy as np
import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
from xqai.network import PVNet
from xqai.mcts import PUCTPlanner
from xqai import arena


def load(path, dev):
    ck = torch.load(path, map_location=dev, weights_only=False)
    net = PVNet(ck.get("channels", 128), ck.get("blocks", 10)).to(dev).eval()
    net.load_state_dict(ck["model"])
    return net, int(ck.get("step", ck.get("epoch", -1)))


def build_pool(n, seed=12345, plies=6):
    """与 eval_vs_ref 相同的开局生成:初始局面随机走 plies 步。"""
    from xqai._xqcore import Position
    rng = np.random.default_rng(seed)
    pool = []
    while len(pool) < n:
        p = Position(); ok = True
        for _ in range(plies):
            lm = list(p.legal_moves())
            if not lm or p.result() != 0:
                ok = False; break
            p.push(int(rng.choice(lm)))
        if ok and p.result() == 0:
            fen = p.fen()
            if fen not in pool:
                pool.append(fen)
    return pool


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--games", type=int, default=100)
    ap.add_argument("--n-sim", type=int, default=80)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--tag", default="final")
    ap.add_argument("--out", default="logs/eval_final.csv")
    ap.add_argument("--seed", type=int, default=777,
                    help="开局子集抽样种子;各消融臂共用同一值 -> 同一组开局,臂间可比")
    a = ap.parse_args()

    dev = a.device
    ref_net, _ = load(a.ref, dev)
    cur_net, step = load(a.ckpt, dev)
    planner = PUCTPlanner(add_noise=False)
    refP = arena.NetPlayer(ref_net, planner, n_sim=a.n_sim, name="ref")
    curP = arena.NetPlayer(cur_net, planner, n_sim=a.n_sim, name=a.tag)

    n_pairs = (a.games + 1) // 2
    pool = build_pool(max(n_pairs, 64))
    pick = np.random.default_rng(a.seed).choice(len(pool), size=n_pairs, replace=False)
    openings = [pool[i] for i in pick]

    t0 = time.time()
    r = arena.play_match(curP, refP, games=a.games, openings=openings)
    secs = int(time.time() - t0)
    wr = r["a_score_rate"]
    lo, hi = r.get("ci95", (None, None))
    elo = arena.elo_from_winrate(wr)

    new = not os.path.exists(a.out)
    with open(a.out, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["tag", "step", "games", "winrate", "elo_diff",
                        "lo", "hi", "W", "D", "L", "secs"])
        w.writerow([a.tag, step, r["games"], f"{wr:.4f}", f"{elo:.1f}",
                    f"{arena.elo_from_winrate(lo):.1f}" if lo is not None else "",
                    f"{arena.elo_from_winrate(hi):.1f}" if hi is not None else "",
                    r["a_wins"], r["a_draws"], r["a_losses"], secs])
    print(f"[final] tag={a.tag} step={step} games={r['games']} "
          f"winrate={wr:.4f} elo={elo:+.1f} "
          f"W/D/L={r['a_wins']}/{r['a_draws']}/{r['a_losses']} ({secs}s)",
          flush=True)


if __name__ == "__main__":
    main()
