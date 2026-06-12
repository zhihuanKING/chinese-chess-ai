#!/usr/bin/env python
"""arena_vec 验证:与串行 arena 的一致性 + 自对打≈50% + 加速比。

子命令(--mode):
  serial-random : 串行 arena.play_match,net vs random,--games 局(基线)
  vec-random    : play_match_vec,net vs random,同开局集(与上对比 CI 重叠)
  vec-selfplay  : play_match_vec,net vs net(同权重),应≈50%
  speed-serial  : 串行 net vs pikafish:<depth>,计时
  speed-vec     : 向量化 net vs pikafish:<depth>,--parallel,计时

开局集由 --opening-seed 决定(两版用同一种子 => 同开局)。
"""
from __future__ import annotations

import argparse
import os
import subprocess
import threading
import time

import numpy as np
import torch

from xqai import arena, arena_vec
from xqai.arena import _ucci_to_move
from xqai.arena_vec import EngineVecPlayer, NetVecPlayer, SerialVecPlayer, _EngineClient
from xqai.mcts import PUCTPlanner
from xqai.network import PVNet

ROOT = os.environ.get("XQAI_ROOT", "/mnt/nvme3n1/gameTheory")
PIKA = os.path.join(ROOT, "third_party/Pikafish/src/pikafish")
PIKA_NNUE = os.path.join(ROOT, "third_party/Pikafish/src/pikafish.nnue")
_TRANS = str.maketrans("heHE", "nbNB")


def load_net(path, dev):
    ck = torch.load(path, map_location=dev, weights_only=False)
    net = PVNet(ck.get("channels", 128), ck.get("blocks", 10)).to(dev).eval()
    net.load_state_dict(ck["model"])
    return net


class RandomPlayer:
    name = "random"

    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)

    def reset(self):
        pass

    def select_move(self, pos):
        lm = list(pos.legal_moves())
        return int(self.rng.choice(lm)) if lm else -1


class PikafishUCIPlayer:
    """串行版 pikafish(从 eval_anchor.py 复刻,做 speed-serial 基线)。"""

    def __init__(self, depth, name="pikafish"):
        self.depth = depth
        self.name = name
        self.proc = subprocess.Popen(
            [PIKA], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, text=True, bufsize=1)
        self._send("uci"); self._wait_for("uciok")
        self._send(f"setoption name EvalFile value {PIKA_NNUE}")
        self._send("setoption name Threads value 1")
        self._send("setoption name Hash value 64")
        self._send("isready"); self._wait_for("readyok")

    def _send(self, line):
        self.proc.stdin.write(line + "\n"); self.proc.stdin.flush()

    def _wait_for(self, token, max_lines=2000):
        for _ in range(max_lines):
            line = self.proc.stdout.readline()
            if not line or token in line:
                return line
        return ""

    def reset(self):
        self._send("ucinewgame")

    def select_move(self, pos):
        self._send(f"position fen {pos.fen().translate(_TRANS)}")
        self._send(f"go depth {self.depth}")
        deadline = time.monotonic() + 35.0
        wd = threading.Timer(35.0, self._kill)
        wd.daemon = True; wd.start()
        try:
            while time.monotonic() < deadline:
                line = self.proc.stdout.readline()
                if not line:
                    return -1
                if line.startswith("bestmove"):
                    parts = line.split()
                    return _ucci_to_move(parts[1]) if len(parts) >= 2 else -1
            return -1
        finally:
            wd.cancel()

    def _kill(self):
        if self.proc.poll() is None:
            self.proc.kill()

    def close(self):
        try:
            self._send("quit"); self.proc.wait(timeout=2)
        except Exception:
            self._kill()


def make_openings(n, plies=6, seed=999):
    from xqai._xqcore import Position
    rng = np.random.default_rng(seed)
    ops = []
    while len(ops) < n:
        p = Position(); ok = True
        for _ in range(plies):
            lm = list(p.legal_moves())
            if not lm or p.result() != 0:
                ok = False; break
            p.push(int(rng.choice(lm)))
        if ok and p.result() == 0:
            ops.append(p.fen())
    return ops


def report(tag, r, dt):
    lo, hi = r["ci95"]
    print(f"[{tag}] games={r['games']} W{r['a_wins']}/D{r['a_draws']}/L{r['a_losses']} "
          f"rate={r['a_score_rate']:.4f} CI95=[{lo:.4f},{hi:.4f}] "
          f"elo={r['elo_diff']:+.1f} secs={dt:.1f}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True,
                    choices=["serial-random", "vec-random", "vec-selfplay",
                             "speed-serial", "speed-vec"])
    ap.add_argument("--ckpt", default=os.path.join(ROOT, "checkpoints/final_best.pt"))
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--games", type=int, default=64)
    ap.add_argument("--n-sim", type=int, default=64)
    ap.add_argument("--max-plies", type=int, default=300)
    ap.add_argument("--parallel", type=int, default=16)
    ap.add_argument("--depth", type=int, default=1, help="pikafish depth for speed modes")
    ap.add_argument("--opening-seed", type=int, default=999)
    a = ap.parse_args()

    dev = f"cuda:{a.gpu}"
    net = load_net(a.ckpt, dev)
    ops = make_openings(max(a.games // 2 + 2, 12), seed=a.opening_seed)
    t0 = time.time()

    if a.mode == "serial-random":
        netP = arena.NetPlayer(net, PUCTPlanner(add_noise=False), n_sim=a.n_sim, name="net")
        r = arena.play_match(netP, RandomPlayer(seed=7), games=a.games,
                             openings=ops, max_plies=a.max_plies)
    elif a.mode == "vec-random":
        netP = NetVecPlayer(net, PUCTPlanner(add_noise=False), n_sim=a.n_sim, name="net")
        r = arena_vec.play_match_vec(netP, SerialVecPlayer(RandomPlayer(seed=7)),
                                     games=a.games, openings=ops,
                                     max_plies=a.max_plies, parallel=a.parallel)
    elif a.mode == "vec-selfplay":
        pa = NetVecPlayer(net, PUCTPlanner(add_noise=False), n_sim=a.n_sim, name="netA")
        pb = NetVecPlayer(net, PUCTPlanner(add_noise=False), n_sim=a.n_sim, name="netB")
        r = arena_vec.play_match_vec(pa, pb, games=a.games, openings=ops,
                                     max_plies=a.max_plies, parallel=a.parallel)
    elif a.mode == "speed-serial":
        netP = arena.NetPlayer(net, PUCTPlanner(add_noise=False), n_sim=a.n_sim, name="net")
        opp = PikafishUCIPlayer(a.depth, name=f"pikafish_d{a.depth}")
        r = arena.play_match(netP, opp, games=a.games, openings=ops,
                             max_plies=a.max_plies)
        opp.close()
    else:  # speed-vec
        netP = NetVecPlayer(net, PUCTPlanner(add_noise=False), n_sim=a.n_sim, name="net")
        name = f"pikafish_d{a.depth}"
        opp = EngineVecPlayer(
            lambda: _EngineClient(
                [PIKA], proto="uci", depth=a.depth, movetime_ms=None,
                translate_fen=True,
                options={"EvalFile": PIKA_NNUE, "Threads": 1, "Hash": 64},
                name=name),
            max_engines=a.parallel, name=name)
        r = arena_vec.play_match_vec(netP, opp, games=a.games, openings=ops,
                                     max_plies=a.max_plies, parallel=a.parallel)
        opp.close()

    report(a.mode, r, time.time() - t0)


if __name__ == "__main__":
    main()
