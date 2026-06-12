#!/usr/bin/env python
"""向量化锚点评估:scripts/eval_anchor.py 的批量版(同 CLI 语义 + --parallel)。

与 eval_anchor.py 的差异:
- 用 xqai.arena_vec.play_match_vec 跑 N 局并行:net 侧每个 lockstep 轮把所有
  "轮到 net 行棋"的局面合成一个 planner.search 批(GPU 一批前向服务所有在飞对局);
  引擎对手每局一个子进程实例(上限 --max-engines),异步发 go、并发收 bestmove。
- 引擎走 ``position fen X moves ...`` 全历史协议(引擎能看见重复局面)。
- CSV 输出格式与 eval_anchor.py 逐字段一致(同一文件可混写)。

opponent 规格:random | net:<ckpt> | pikafish:<depth> | pikafishmt:<ms> | xqab:<ms>
"""
from __future__ import annotations

import argparse
import csv
import os
import time

import numpy as np
import torch

from xqai import arena_vec
from xqai.arena_vec import EngineVecPlayer, NetVecPlayer, SerialVecPlayer, _EngineClient
from xqai.mcts import PUCTPlanner
from xqai.network import PVNet


def load_net(path, dev):
    ck = torch.load(path, map_location=dev, weights_only=False)
    net = PVNet(ck.get("channels", 128), ck.get("blocks", 10)).to(dev).eval()
    net.load_state_dict(ck["model"])
    return net, int(ck.get("step", ck.get("epoch", -1)))


class RandomPlayer:
    name = "random"

    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)

    def reset(self):
        pass

    def select_move(self, pos):
        lm = list(pos.legal_moves())
        return int(self.rng.choice(lm)) if lm else -1


def make_openings(n, plies=6, seed=12345):
    """与 eval_anchor.make_openings 同算法/同种子 => 同开局池。"""
    from xqai._xqcore import Position

    rng = np.random.default_rng(seed)
    ops = []
    while len(ops) < n:
        p = Position()
        ok = True
        for _ in range(plies):
            lm = list(p.legal_moves())
            if not lm or p.result() != 0:
                ok = False
                break
            p.push(int(rng.choice(lm)))
        if ok and p.result() == 0:
            ops.append(p.fen())
    return ops


def _root() -> str:
    return os.environ.get(
        "XQAI_ROOT",
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    )


def build_opponent(spec, gpu, n_sim=200, max_engines=16):
    """Returns a vec player for ``spec`` (engines get one subprocess per game)."""
    root = _root()
    if spec == "random":
        return SerialVecPlayer(RandomPlayer(seed=7))
    if spec.startswith("net:"):
        p = spec.split(":", 1)[1]
        onet, _ = load_net(p, f"cuda:{gpu}")
        oplan = PUCTPlanner(add_noise=False)
        return NetVecPlayer(
            onet, oplan, n_sim=n_sim, name=f"net@{os.path.basename(p)}"
        )
    if spec.startswith("pikafish:") or spec.startswith("pikafishmt:"):
        b = os.path.join(root, "third_party/Pikafish/src/pikafish")
        nn = os.path.join(root, "third_party/Pikafish/src/pikafish.nnue")
        opts = {"EvalFile": nn, "Threads": 1, "Hash": 64}
        if spec.startswith("pikafish:"):
            d = int(spec.split(":")[1])
            name = f"pikafish_d{d}"
            factory = lambda: _EngineClient(
                [b], proto="uci", depth=d, movetime_ms=None,
                translate_fen=True, options=opts, name=name)
        else:
            ms = int(spec.split(":")[1])
            name = f"pikafish_{ms}ms"
            factory = lambda: _EngineClient(
                [b], proto="uci", depth=None, movetime_ms=ms,
                translate_fen=True, options=opts, name=name)
        return EngineVecPlayer(factory, max_engines=max_engines, name=name)
    if spec.startswith("xqab:"):
        ms = int(spec.split(":")[1])
        b = os.path.join(root, "cpp/build/xqab")
        name = f"xqab_{ms}ms"
        return EngineVecPlayer(
            lambda: _EngineClient([b], proto="ucci", movetime_ms=ms, name=name),
            max_engines=max_engines,
            name=name,
        )
    raise ValueError(f"unknown opponent {spec}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/final_best.pt")
    ap.add_argument("--opponent", required=True,
                    help="random | net:<ckpt> | pikafish:<depth> | pikafishmt:<ms> | xqab:<ms>")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--n-sim", type=int, default=200)
    ap.add_argument("--games", type=int, default=30)
    ap.add_argument("--max-plies", type=int, default=300)
    ap.add_argument("--out", default="logs/anchor_ladder.csv")
    ap.add_argument("--tag", default="")
    ap.add_argument("--opening-seed", type=int, default=12345,
                    help="开局池种子;分片并行评测时各分片必须给不同值,否则对弈重复、样本造假")
    ap.add_argument("--parallel", type=int, default=16, help="同时在飞的对局数")
    ap.add_argument("--max-engines", type=int, default=None,
                    help="引擎子进程上限(默认 = --parallel)")
    a = ap.parse_args()

    dev = f"cuda:{a.gpu}"
    net, _ = load_net(a.ckpt, dev)
    planner = PUCTPlanner(add_noise=False)
    netP = NetVecPlayer(net, planner, n_sim=a.n_sim, name=f"net_n{a.n_sim}")
    max_engines = a.max_engines if a.max_engines is not None else a.parallel
    opp = build_opponent(a.opponent, a.gpu, n_sim=a.n_sim, max_engines=max_engines)
    ops = make_openings(max(a.games // 2 + 2, 12), seed=a.opening_seed)

    t0 = time.time()
    r = arena_vec.play_match_vec(
        netP, opp, games=a.games, openings=ops, max_plies=a.max_plies,
        parallel=a.parallel, max_engines=max_engines)
    dt = time.time() - t0
    opp.close()

    wr = r["a_score_rate"]
    lo, hi = r["ci95"]
    elo = arena_vec.elo_from_winrate(wr)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    new = not os.path.exists(a.out)
    with open(a.out, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["tag", "opponent", "n_sim", "games", "winrate", "lo", "hi",
                        "elo_diff", "W", "D", "L", "secs"])
        w.writerow([a.tag, a.opponent, a.n_sim, r["games"], f"{wr:.4f}",
                    f"{lo:.4f}", f"{hi:.4f}", f"{elo:.1f}",
                    r["a_wins"], r["a_draws"], r["a_losses"], f"{dt:.0f}"])
    print(f"[anchor-vec] {a.opponent:16s} n_sim={a.n_sim} games={r['games']} "
          f"par={a.parallel} winrate={wr:.3f} [{lo:.3f},{hi:.3f}] elo={elo:+.0f} "
          f"W{r['a_wins']}/D{r['a_draws']}/L{r['a_losses']} ({dt:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
