#!/usr/bin/env python
"""绝对棋力锚点评估:net(冷启动主力) vs {random, Pikafish 限档, xqab}。

给报告 §7"效果"补一条 Elo 天梯:random(地板) → net → xqab/Pikafish(天花板),
为学习型网络定一个绝对棋力坐标(而非仅"负于 AB"的相对结论)。

Pikafish 是 **UCI** 引擎(非 UCCI),且 FEN 棋子字母用 n(马)/b(象),
与本项目 h/e 不同 —— 本脚本内置 UCI 握手 + FEN 翻译(h<->n, e<->b)。

单次只跑一对(--opponent),多对用 shell 并行铺到多卡。
"""
from __future__ import annotations
import argparse, csv, json, os, subprocess, threading, time
import numpy as np, torch

from xqai.network import PVNet
from xqai.mcts import PUCTPlanner
from xqai import arena
from xqai.arena import _ucci_to_move


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


_TRANS = str.maketrans("heHE", "nbNB")


class PikafishUCIPlayer:
    """Pikafish over UCI. Translates our FEN piece letters h/e -> n/b."""

    def __init__(self, binary, nnue, *, depth=None, movetime_ms=None, threads=1,
                 hash_mb=64, name="pikafish"):
        self.cmd = [binary]
        self.nnue = nnue
        self.depth = depth
        self.movetime_ms = movetime_ms
        self.threads = threads
        self.hash_mb = hash_mb
        self.name = name
        self.proc = None
        self._start()

    def _start(self):
        self.proc = subprocess.Popen(
            self.cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, text=True, bufsize=1)
        self._send("uci"); self._wait_for("uciok")
        if self.nnue:
            self._send(f"setoption name EvalFile value {self.nnue}")
        self._send(f"setoption name Threads value {self.threads}")
        self._send(f"setoption name Hash value {self.hash_mb}")
        self._send("isready"); self._wait_for("readyok")

    def _send(self, line):
        self.proc.stdin.write(line + "\n"); self.proc.stdin.flush()

    def _wait_for(self, token, max_lines=2000):
        for _ in range(max_lines):
            line = self.proc.stdout.readline()
            if not line:
                break
            if token in line:
                return line
        return ""

    def reset(self):
        self._send("ucinewgame")

    # arena._play_game 看到此标志后会传 (start_fen, moves) —— 引擎拿到完整着法
    # 历史,重复检测/避重才生效。
    wants_moves = True

    def select_move(self, pos, start_fen=None, moves=None):
        # 只翻译 FEN 里的棋子字母 h/e -> n/b(Pikafish 记谱);着法串是坐标格式
        # (列 a..i + 行 0..9),其中 'h'/'e' 是列字母,绝不能翻译。
        fen = (start_fen if start_fen else pos.fen()).translate(_TRANS)
        cmd = f"position fen {fen}"
        if start_fen and moves:
            cmd += " moves " + " ".join(moves)
        self._send(cmd)
        if self.depth is not None:
            self._send(f"go depth {self.depth}")
        else:
            self._send(f"go movetime {self.movetime_ms}")
        budget_ms = self.movetime_ms if self.movetime_ms else 30_000
        deadline = time.monotonic() + budget_ms / 1000.0 + 5.0
        wd = threading.Timer(max(deadline - time.monotonic(), 0.1), self._kill)
        wd.daemon = True; wd.start()
        try:
            for _ in range(1_000_000):
                if time.monotonic() > deadline:
                    return -1
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
        if self.proc and self.proc.poll() is None:
            try: self.proc.kill()
            except Exception: pass

    def close(self):
        if self.proc:
            try:
                self._send("quit"); self.proc.wait(timeout=2)
            except Exception:
                self.proc.kill()


def make_openings(n, plies=6, seed=12345):
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


def build_opponent(spec, gpu, n_sim=200):
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if spec == "random":
        return RandomPlayer(seed=7), None
    if spec.startswith("net:"):
        p = spec.split(":", 1)[1]
        onet, ostep = load_net(p, f"cuda:{gpu}")
        oplan = PUCTPlanner(add_noise=False)
        return arena.NetPlayer(onet, oplan, n_sim=n_sim, name=f"net@{os.path.basename(p)}"), None
    if spec.startswith("pikafish:"):
        d = int(spec.split(":")[1])
        b = os.path.join(root, "third_party/Pikafish/src/pikafish")
        n = os.path.join(root, "third_party/Pikafish/src/pikafish.nnue")
        p = PikafishUCIPlayer(b, n, depth=d, name=f"pikafish_d{d}")
        return p, p
    if spec.startswith("pikafishmt:"):
        ms = int(spec.split(":")[1])
        b = os.path.join(root, "third_party/Pikafish/src/pikafish")
        n = os.path.join(root, "third_party/Pikafish/src/pikafish.nnue")
        p = PikafishUCIPlayer(b, n, movetime_ms=ms, name=f"pikafish_{ms}ms")
        return p, p
    if spec.startswith("xqab:"):
        ms = int(spec.split(":")[1])
        b = os.path.join(root, "cpp/build/xqab")
        p = arena.SubprocessUCCIPlayer([b], movetime_ms=ms, name=f"xqab_{ms}ms")
        return p, p
    raise ValueError(f"unknown opponent {spec}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/final_best.pt")
    ap.add_argument("--opponent", required=True,
                    help="random | pikafish:<depth> | pikafishmt:<ms> | xqab:<ms>")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--n-sim", type=int, default=200)
    ap.add_argument("--games", type=int, default=30)
    ap.add_argument("--max-plies", type=int, default=300)
    ap.add_argument("--out", default="logs/anchor_ladder.csv")
    ap.add_argument("--tag", default="")
    ap.add_argument("--dump-games", default="",
                    help="每局追加一行 JSON(开局FEN/着法/红方视角结果/net执色),DAgger 标注原料")
    ap.add_argument("--opening-seed", type=int, default=12345,
                    help="开局池种子;分片并行评测时各分片必须给不同值,否则对弈重复、样本造假")
    a = ap.parse_args()

    dev = f"cuda:{a.gpu}"
    net, step = load_net(a.ckpt, dev)
    planner = PUCTPlanner(add_noise=False)
    netP = arena.NetPlayer(net, planner, n_sim=a.n_sim, name=f"net_n{a.n_sim}")
    opp, closable = build_opponent(a.opponent, a.gpu, n_sim=a.n_sim)
    ops = make_openings(max(a.games // 2 + 2, 12), seed=a.opening_seed)

    t0 = time.time()
    r = arena.play_match(netP, opp, games=a.games, openings=ops, max_plies=a.max_plies)
    dt = time.time() - t0
    if closable is not None:
        try: closable.close()
        except Exception: pass

    if a.dump_games:
        d = os.path.dirname(a.dump_games)
        if d:
            os.makedirs(d, exist_ok=True)
        with open(a.dump_games, "a") as f:
            for g in r["game_records"]:
                # play_match(netP, opp): player_a 恒为 net, a_is_red 即 net 执红。
                f.write(json.dumps({
                    "opening_fen": g["opening_fen"],
                    "moves": g["moves"],
                    "result": g["result"],  # 红方视角 ±1/0
                    "net_side": "red" if g["a_is_red"] else "black",
                    "opponent": a.opponent,
                    "tag": a.tag,
                }) + "\n")

    wr = r["a_score_rate"]; lo, hi = r["ci95"]
    elo = arena.elo_from_winrate(wr)
    new = not os.path.exists(a.out)
    with open(a.out, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["tag", "opponent", "n_sim", "games", "winrate", "lo", "hi",
                        "elo_diff", "W", "D", "L", "secs"])
        w.writerow([a.tag, a.opponent, a.n_sim, r["games"], f"{wr:.4f}",
                    f"{lo:.4f}", f"{hi:.4f}", f"{elo:.1f}",
                    r["a_wins"], r["a_draws"], r["a_losses"], f"{dt:.0f}"])
    print(f"[anchor] {a.opponent:16s} n_sim={a.n_sim} games={r['games']} "
          f"winrate={wr:.3f} [{lo:.3f},{hi:.3f}] elo={elo:+.0f} "
          f"W{r['a_wins']}/D{r['a_draws']}/L{r['a_losses']} ({dt:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
