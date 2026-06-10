"""出报告杀手锏图:
 mode=learning  : 一组 checkpoint(按 step 排序) vs 固定参考/random -> 监督学习曲线 csv+png
 mode=crossover : 一个 ckpt 在多个 n_sim vs xqab@多个 ms -> 学习型 vs Alpha-Beta crossover csv+png
复用 arena(NetPlayer+PUCTPlanner(add_noise=False)+多样化随机开局)。单进程;并行由调用方用 CUDA_VISIBLE_DEVICES 分卡。"""
import argparse, glob, csv, re
import numpy as np, torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from xqai._xqcore import Position
from xqai.network import PVNet
from xqai.mcts import PUCTPlanner
from xqai import arena

DEV = "cuda:0"

def load_net(path):
    ck = torch.load(path, map_location=DEV, weights_only=False)
    net = PVNet(ck.get("channels", 128), ck.get("blocks", 10)).to(DEV).eval()
    net.load_state_dict(ck["model"])
    step = ck.get("step", ck.get("epoch", 0))
    return net, int(step if step is not None else 0)

class RandomPlayer:
    name = "random"
    def __init__(s): s.rng = np.random.default_rng(0)
    def reset(s): pass
    def close(s): pass
    def select_move(s, pos):
        lm = list(pos.legal_moves()); return int(s.rng.choice(lm)) if lm else -1

def gen_openings(n, plies=6, seed=7):
    rng = np.random.default_rng(seed); ops = []
    while len(ops) < n:
        p = Position(); ok = True
        for _ in range(plies):
            lm = list(p.legal_moves())
            if not lm or p.result() != 0: ok = False; break
            p.push(int(rng.choice(lm)))
        if ok and p.result() == 0: ops.append(p.fen())
    return ops

def step_of(path):
    m = re.search(r"pstep_(\d+)", path) or re.search(r"_e(\d+)", path)
    return int(m.group(1)) if m else 0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["learning", "crossover"])
    ap.add_argument("--ckpts", default="checkpoints/pstep_*.pt")
    ap.add_argument("--ckpt", default="checkpoints/pretrained_best.pt")
    ap.add_argument("--ref", default="random", help="learning 基线: net 路径 或 'random'")
    ap.add_argument("--nsims", default="200,800")
    ap.add_argument("--ab-ms", default="50,200,500,1000")
    ap.add_argument("--n-sim", type=int, default=800)
    ap.add_argument("--games", type=int, default=24)
    ap.add_argument("--out-prefix", default="logs/curve")
    a = ap.parse_args()
    planner = PUCTPlanner(add_noise=False)
    ops = gen_openings(max(a.games // 2 + 2, 12))

    if a.mode == "learning":
        files = sorted(glob.glob(a.ckpts), key=step_of)
        if not files:
            raise SystemExit("no ckpts match " + a.ckpts)
        refnet = None
        if a.ref and a.ref != "random":
            refnet, _ = load_net(a.ref)
        rows = []
        for f in files:
            net, step = load_net(f)
            A = arena.NetPlayer(net, planner, n_sim=a.n_sim, name="cand")
            B = arena.NetPlayer(refnet, planner, n_sim=a.n_sim, name="ref") if refnet is not None else RandomPlayer()
            r = arena.play_match(A, B, games=a.games, openings=ops)
            wr = r["a_score_rate"]; rows.append((step, wr, arena.elo_from_winrate(wr)))
            print(f"[learning] step={step} winrate={wr:.3f} elo={arena.elo_from_winrate(wr):.0f}", flush=True)
            del net; torch.cuda.empty_cache()
        with open(a.out_prefix + "_learning.csv", "w", newline="") as fh:
            w = csv.writer(fh); w.writerow(["step", "winrate", "elo"]); w.writerows(rows)
        xs = [r[0] for r in rows]; ys = [r[2] for r in rows]
        plt.figure(); plt.plot(xs, ys, "-o"); plt.xlabel("training step"); plt.ylabel("relative Elo")
        plt.title(f"Supervised learning curve (vs {a.ref})"); plt.grid(True)
        plt.savefig(a.out_prefix + "_learning.png", dpi=120)
        print("wrote " + a.out_prefix + "_learning.png", flush=True)
    else:
        net, _ = load_net(a.ckpt)
        nsims = [int(x) for x in a.nsims.split(",")]
        abms = [int(x) for x in a.ab_ms.split(",")]
        rows = []; plt.figure()
        for ns in nsims:
            ys = []; A = arena.NetPlayer(net, planner, n_sim=ns, name="cand")
            for ms in abms:
                B = arena.SubprocessUCCIPlayer(["cpp/build/xqab"], movetime_ms=ms)
                r = arena.play_match(A, B, games=a.games, openings=ops)
                wr = r["a_score_rate"]; rows.append((ns, ms, wr)); ys.append(wr)
                print(f"[crossover] nsim={ns} ab={ms}ms winrate={wr:.3f}", flush=True)
                try: B.close()
                except Exception: pass
            plt.plot(abms, ys, "-o", label=f"net n_sim={ns}")
        with open(a.out_prefix + "_crossover.csv", "w", newline="") as fh:
            w = csv.writer(fh); w.writerow(["nsim", "ab_ms", "winrate"]); w.writerows(rows)
        plt.axhline(0.5, ls="--", c="gray"); plt.xlabel("Alpha-Beta time (ms)"); plt.ylabel("net winrate")
        plt.title("Learned net vs Alpha-Beta (crossover)"); plt.legend(); plt.grid(True)
        plt.savefig(a.out_prefix + "_crossover.png", dpi=120)
        print("wrote " + a.out_prefix + "_crossover.png", flush=True)

if __name__ == "__main__":
    main()
