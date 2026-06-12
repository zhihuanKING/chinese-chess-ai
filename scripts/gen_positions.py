#!/usr/bin/env python
"""DAgger 局面采集器（GPU 侧，批量自对弈只记 FEN，不写 replay）。

复用 xqai.selfplay 的 lockstep 模式：B 局对局齐步推进，PUCT/Gumbel planner 的
批量 ``search(positions, net, n_sim)`` 把所有叶评估并成单次 net forward。
每局完局后输出该局走过的所有局面到 jsonl shard（先写 .tmp 再原子 rename，
供 scripts/label_service.py 安全认领）：

    {"fen": ..., "ply": ..., "gid": ..., "outcome_red": +1/-1/0, "move": "h2e2",
     "explore": 0/1}

- ``fen``      : xqai._xqcore.Position.fen() 原格式（h=马 e=象；标注侧自行转换）。
- ``ply``      : 该局面在本局中的步号（0 起，连续）。
- ``outcome_red``: 真实终局（红视角 +1/-1/0），供 value 标定交叉验证。
- ``move``     : 该局面实际走出的着法（ICCS 坐标），供复盘校验。
- ``explore``  : 全程 τ=1 的探索局标记（占 --explore-frac）。

用法（正式采集示例）::

    .venv/bin/python scripts/gen_positions.py \
        --ckpt checkpoints_v2_rl/ref_coldstart_frozen.pt --gpu 0 \
        --n-sim 128 --parallel-games 256 --games 4096 \
        --out-dir data/dagger/queue_p1
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)
_ROOT = os.path.dirname(_THIS_DIR)

from prepare_data import move_to_iccs  # noqa: E402

from xqai.encoding import BLACK  # noqa: E402
from xqai.encoding import flip_move  # noqa: E402
from xqai.mcts import GumbelPlanner, PUCTPlanner  # noqa: E402
from xqai.selfplay import _sample_move  # noqa: E402  (复用采样逻辑)

# Result codes (xqai._xqcore §3)
_ONGOING, _RED_WIN, _BLACK_WIN, _DRAW = 0, 1, 2, 3
_OUTCOME_RED = {_RED_WIN: 1, _BLACK_WIN: -1, _DRAW: 0}


def _new_position():
    from xqai._xqcore import Position
    return Position()


def load_net(path: str, device: torch.device):
    """加载 PVNet checkpoint（与 scripts/eval_anchor.load_net 同约定）。"""
    from xqai.network import PVNet
    ck = torch.load(path, map_location=device, weights_only=False)
    net = PVNet(ck.get("channels", 128), ck.get("blocks", 10)).to(device).eval()
    net.load_state_dict(ck["model"])
    return net


class _Game:
    """一局进行中的对局：局面 + 已走过的 (fen, ply, move_iccs) 记录。"""

    __slots__ = ("pos", "records", "ply", "explore", "gid")

    def __init__(self, gid: str, explore: bool):
        self.pos = _new_position()
        self.records: list[tuple[str, int, str]] = []
        self.ply = 0
        self.explore = explore
        self.gid = gid


class _ShardWriter:
    """按完局数攒 jsonl shard：写 .tmp，满 --shard-games 局原子 rename 落盘。"""

    def __init__(self, out_dir: str, tag: str, shard_games: int):
        self.out_dir = out_dir
        self.tag = tag
        self.shard_games = int(shard_games)
        self.seq = 0
        self._lines: list[str] = []
        self._games = 0
        self.total_positions = 0
        os.makedirs(out_dir, exist_ok=True)

    def add_game(self, lines: list[str]) -> None:
        self._lines.extend(lines)
        self._games += 1
        self.total_positions += len(lines)
        if self._games >= self.shard_games:
            self.flush()

    def flush(self) -> None:
        if not self._lines:
            return
        name = f"positions_{self.tag}_{self.seq:04d}.jsonl"
        tmp = os.path.join(self.out_dir, name + ".tmp")
        with open(tmp, "w", encoding="utf-8") as fh:
            fh.write("\n".join(self._lines) + "\n")
        os.replace(tmp, os.path.join(self.out_dir, name))  # 原子可见
        print(f"[shard] {name}: {self._games} games, {len(self._lines)} positions",
              flush=True)
        self.seq += 1
        self._lines = []
        self._games = 0


def collect(args) -> None:
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    if args.dummy:
        from xqai.dummynet import DummyNet
        net = DummyNet().to(device).eval()
    else:
        net = load_net(args.ckpt, device)

    if args.planner == "gumbel":
        planner = GumbelPlanner(seed=args.seed)
    else:
        planner = PUCTPlanner(seed=args.seed)

    rng = np.random.default_rng(args.seed)
    tag = f"{int(time.time())}_{os.getpid()}"
    writer = _ShardWriter(args.out_dir, tag, args.shard_games)

    n_started = 0

    def new_game() -> _Game:
        nonlocal n_started
        g = _Game(gid=f"{tag}-{n_started}",
                  explore=bool(rng.random() < args.explore_frac))
        n_started += 1
        return g

    games = [new_game() for _ in range(args.parallel_games)]
    finished = 0
    t0 = time.monotonic()
    last_log = t0

    def finish_game(g: _Game, result: int) -> None:
        nonlocal finished
        outcome_red = _OUTCOME_RED[result]
        lines = [
            json.dumps({"fen": fen, "ply": ply, "gid": g.gid,
                        "outcome_red": outcome_red, "move": mv,
                        "explore": int(g.explore)},
                       ensure_ascii=False, separators=(",", ":"))
            for fen, ply, mv in g.records
        ]
        writer.add_game(lines)
        finished += 1

    with torch.no_grad():
        while finished < args.games:
            positions = [g.pos for g in games]
            # === 批量 MCTS：B 局并一次（叶评估在 planner 内并 batch）===
            pis = planner.search(positions, net, args.n_sim)

            for gi, (g, pi) in enumerate(zip(games, pis)):
                pos = g.pos
                temperature = 1.0 if (g.explore or g.ply < args.temp_moves) else 0.0
                move_norm = _sample_move(pi, temperature, rng)
                if move_norm < 0:
                    # 无合法着法：行棋方负（该终局局面不记录，标注不了）。
                    res = _BLACK_WIN if pos.side_to_move() == 0 else _RED_WIN
                    finish_game(g, res)
                    games[gi] = new_game()
                    continue

                real_move = (flip_move(move_norm)
                             if pos.side_to_move() == BLACK else move_norm)
                # 记录走过的局面（push 之前的 FEN + 实际着法）。
                g.records.append((pos.fen(), g.ply, move_to_iccs(real_move)))
                pos.push(real_move)
                g.ply += 1

                res = pos.result()
                if res != _ONGOING:
                    finish_game(g, res)
                    games[gi] = new_game()
                elif g.ply >= args.max_plies:
                    finish_game(g, _DRAW)
                    games[gi] = new_game()

            now = time.monotonic()
            if now - last_log >= 30.0:
                rate = finished / max(now - t0, 1e-9) * 3600.0
                prate = writer.total_positions / max(now - t0, 1e-9) * 3600.0
                print(f"[gen] finished={finished}/{args.games} "
                      f"positions={writer.total_positions} "
                      f"({rate:.0f} games/h, {prate:.0f} pos/h)", flush=True)
                last_log = now

    writer.flush()  # 余量（不足 shard-games 的尾巴）也落盘
    dt = time.monotonic() - t0
    print(f"[gen] DONE: {finished} games, {writer.total_positions} positions "
          f"in {dt:.1f}s -> {args.out_dir}", flush=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="DAgger FEN collector (GPU lockstep self-play)")
    ap.add_argument("--ckpt", default=os.path.join(_ROOT, "checkpoints_v2_rl/ref_coldstart_frozen.pt"))
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--n-sim", type=int, default=128)
    ap.add_argument("--parallel-games", type=int, default=256)
    ap.add_argument("--games", type=int, required=True, help="目标完局数")
    ap.add_argument("--temp-moves", type=int, default=30,
                    help="非探索局的开局 τ=1 步数")
    ap.add_argument("--explore-frac", type=float, default=0.2,
                    help="全程 τ=1 探索局占比（增加覆盖）")
    ap.add_argument("--max-plies", type=int, default=400)
    ap.add_argument("--planner", choices=["puct", "gumbel"], default="puct")
    ap.add_argument("--out-dir", default=os.path.join(_ROOT, "data/dagger/queue_p1"))
    ap.add_argument("--shard-games", type=int, default=256,
                    help="每 N 完局写一个 jsonl shard")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dummy", action="store_true",
                    help="用 DummyNet 代替 ckpt（无 GPU 自测）")
    args = ap.parse_args(argv)
    collect(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
