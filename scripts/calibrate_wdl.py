"""WDL 校准：拟合 Pikafish cp -> 期望分 E 的 logistic 映射（任务 A0.1）。

从 data/raw/pikafish_v2/*.txt（每行一盘 UCI 着法序列 + 结果记号）随机抽 ~20000
个 (对局, ply) 前缀，起 N 个 pikafish 进程（UCI_ShowWDL true, Threads 1），
`go depth D` 后解析最深 multipv-1 的 ``score cp X ... wdl W D L``，对
期望分 E=(w+0.5d)/1000 关于 cp 拟合::

    E(cp) = 1 / (1 + exp(-(cp - b) / s))

输出拟合参数 + 分箱残差数据到 logs/wdl_calib.json。

注意视角：UCI 的 cp 与 wdl 都是 **行棋方** 视角，与 processed_v2 的
z=tanh(root_cp/500)（同为行棋方视角）一致，因此下游 z_wdl 变换无需翻转符号。

用法::

    .venv/bin/python scripts/calibrate_wdl.py \
        --raw data/raw/pikafish_v2 --n 20000 --workers 120 --depth 14 \
        --out logs/wdl_calib.json
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import multiprocessing as mp
import os
import random
import re
import sys
import time

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_THIS_DIR)
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from gen_pikafish_data import DEFAULT_ENGINE, EngineError, PikafishEngine  # noqa: E402

_MOVE_RE = re.compile(r"^[a-i][0-9][a-i][0-9]$")
# wdl 紧跟在 score 之后（Stockfish/Pikafish 的 info 行序），mate 行没有 cp 含义。
_INFO_WDL_RE = re.compile(
    r"depth (\d+) .*?multipv (\d+) score (cp|mate) (-?\d+)"
    r"(?: (?:upper|lower)bound)? wdl (\d+) (\d+) (\d+)")


# --------------------------------------------------------------------------
# 抽样：从原始对局 txt 抽 (move_prefix) 任务
# --------------------------------------------------------------------------
def sample_prefixes(raw_dir: str, n: int, seed: int = 0,
                    min_ply: int = 1) -> list:
    """全量读对局，均匀抽 n 个 (对局, ply) 前缀（ply>=min_ply，避开起始局面扎堆）。"""
    games = []
    for f in sorted(glob.glob(os.path.join(raw_dir, "*.txt"))):
        with open(f) as fh:
            for line in fh:
                toks = [t for t in line.split() if _MOVE_RE.match(t)]
                if len(toks) >= min_ply + 4:
                    games.append(toks)
    if not games:
        raise SystemExit(f"no games found under {raw_dir}")
    rng = random.Random(seed)
    tasks = []
    for _ in range(n):
        g = games[rng.randrange(len(games))]
        ply = rng.randrange(min_ply, len(g))   # 局面 = 前 ply 步之后
        tasks.append(" ".join(g[:ply]))
    return tasks


# --------------------------------------------------------------------------
# worker：常驻引擎，崩溃自动重启
# --------------------------------------------------------------------------
_G = {}


def _start_engine(path: str, depth: int):
    eng = PikafishEngine(path, threads=1, hash_mb=16, multipv=1)
    eng.start()
    eng._send("setoption name UCI_ShowWDL value true")
    eng._send("isready")
    eng._wait_for("readyok", 10.0)
    _G["eng"] = eng
    _G["depth"] = depth


def _init_worker(path: str, depth: int):
    _G["path"] = path
    try:
        _start_engine(path, depth)
    except EngineError:
        _G["eng"] = None  # 首个任务里再重试


def _analyse_wdl(eng: PikafishEngine, prefix: str, depth: int, timeout: float = 60.0):
    """返回 (cp, w, d, l) 或 None（mate / 解析失败）。"""
    eng._send("position startpos moves " + prefix if prefix
              else "position startpos")
    eng._send(f"go depth {depth}")
    deadline = time.monotonic() + timeout
    best = None  # (depth, cp, w, d, l)
    while True:
        line = eng._readline(deadline)
        if line.startswith("info") and " wdl " in line:
            m = _INFO_WDL_RE.search(line)
            if m and int(m.group(2)) == 1 and m.group(3) == "cp":
                d = int(m.group(1))
                if best is None or d >= best[0]:
                    best = (d, int(m.group(4)), int(m.group(5)),
                            int(m.group(6)), int(m.group(7)))
        elif line.startswith("bestmove"):
            break
    if best is None:
        return None
    return best[1:]


def _work(prefix: str):
    for attempt in range(2):
        eng = _G.get("eng")
        if eng is None:
            try:
                _start_engine(_G["path"], _G["depth"])
                eng = _G["eng"]
            except EngineError:
                continue
        try:
            return _analyse_wdl(eng, prefix, _G["depth"])
        except EngineError:
            try:
                eng.close()
            except Exception:
                pass
            _G["eng"] = None  # 下一轮重启
    return None


# --------------------------------------------------------------------------
# 拟合（纯 numpy：logit 线性初值 + Gauss-Newton 精修）
# --------------------------------------------------------------------------
def fit_logistic(cp, e):
    import numpy as np
    cp = np.asarray(cp, dtype=np.float64)
    e = np.asarray(e, dtype=np.float64)
    # 初值：logit 回归（裁剪避免 ±inf）
    ec = np.clip(e, 1e-3, 1 - 1e-3)
    y = np.log(ec / (1 - ec))            # y = (cp - b)/s
    A = np.stack([cp, np.ones_like(cp)], axis=1)
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    k, c = coef                           # y = k*cp + c => s=1/k, b=-c/k
    s = 1.0 / k if k > 1e-9 else 300.0
    b = -c * s
    # Gauss-Newton on (b, s) for squared error of E
    for _ in range(50):
        t = (cp - b) / s
        f = 1.0 / (1.0 + np.exp(-t))
        r = e - f
        g = f * (1.0 - f)                 # dE/dt
        J_b = -g / s                      # dE/db
        J_s = -g * t / s                  # dE/ds
        J = np.stack([J_b, J_s], axis=1)
        JTJ = J.T @ J
        JTr = J.T @ r
        try:
            delta = np.linalg.solve(JTJ + 1e-9 * np.eye(2), JTr)
        except np.linalg.LinAlgError:
            break
        b += delta[0]
        s += delta[1]
        if abs(delta[0]) < 1e-6 and abs(delta[1]) < 1e-6:
            break
    rmse = float(np.sqrt(np.mean((e - 1.0 / (1.0 + np.exp(-(cp - b) / s))) ** 2)))
    return float(b), float(s), rmse


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw", default="data/raw/pikafish_v2")
    ap.add_argument("--engine", default=DEFAULT_ENGINE)
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--workers", type=int, default=120)
    ap.add_argument("--depth", type=int, default=14)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="logs/wdl_calib.json")
    args = ap.parse_args()

    import numpy as np

    tasks = sample_prefixes(args.raw, args.n, seed=args.seed)
    print(f"[calib] {len(tasks)} positions, {args.workers} engines, "
          f"depth {args.depth}", flush=True)

    t0 = time.time()
    rows = []
    fails = 0
    ctx = mp.get_context("spawn")  # 干净进程，不带父进程状态
    with ctx.Pool(args.workers, initializer=_init_worker,
                  initargs=(args.engine, args.depth)) as pool:
        for i, r in enumerate(pool.imap_unordered(_work, tasks, chunksize=8)):
            if r is None:
                fails += 1
            else:
                rows.append(r)
            if (i + 1) % 2000 == 0:
                dt = time.time() - t0
                print(f"[calib] {i+1}/{len(tasks)} ok={len(rows)} fail={fails} "
                      f"({dt:.0f}s, {(i+1)/dt:.1f} pos/s)", flush=True)

    print(f"[calib] done: ok={len(rows)} fail={fails} in {time.time()-t0:.0f}s",
          flush=True)
    if len(rows) < 1000:
        raise SystemExit("too few successful analyses; aborting fit")

    arr = np.asarray(rows, dtype=np.float64)      # [M,4] cp,w,d,l
    cp, w, d, l = arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3]
    e = (w + 0.5 * d) / 1000.0
    b, s, rmse = fit_logistic(cp, e)

    def E(x):
        return 1.0 / (1.0 + math.exp(-(x - b) / s))

    z_cap = 2.0 * E(2000.0) - 1.0
    print(f"[calib] fit: b={b:.2f} s={s:.2f} rmse={rmse:.4f} "
          f"E(0)={E(0):.4f} E(100)={E(100):.4f} E(500)={E(500):.4f} "
          f"E(2000)={E(2000):.6f} z_cap={z_cap:.6f}", flush=True)

    # 分箱残差（报告画图用）
    edges = np.arange(-1525.0, 1526.0, 50.0)
    centers, emp, fitted, counts = [], [], [], []
    for j in range(len(edges) - 1):
        m = (cp >= edges[j]) & (cp < edges[j + 1])
        if m.sum() < 5:
            continue
        c0 = 0.5 * (edges[j] + edges[j + 1])
        centers.append(c0)
        emp.append(float(e[m].mean()))
        fitted.append(E(c0))
        counts.append(int(m.sum()))

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    out = {
        "b": b, "s": s, "rmse": rmse,
        "z_cap": z_cap, "z_cap_cp": 2000.0,
        "n_ok": len(rows), "n_fail": fails,
        "depth": args.depth, "seed": args.seed,
        "engine": args.engine,
        "model": "E(cp)=1/(1+exp(-(cp-b)/s)), cp/E both side-to-move view",
        "cp_stats": {"mean": float(cp.mean()), "std": float(cp.std()),
                     "q": {str(q): float(np.percentile(cp, q))
                           for q in (1, 5, 25, 50, 75, 95, 99)}},
        "bins": {"center_cp": centers, "empirical_E": emp,
                 "fitted_E": fitted, "count": counts},
    }
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"[calib] wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
