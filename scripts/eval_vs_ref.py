"""学习曲线评估:latest.pt(RL中) vs 冻结的参考模型(冷启动)。
从 50% 起步,RL 有效则胜率/相对Elo上升 —— 直接反映 RL 自我提升。net-vs-net,可与 RL 并行。"""
import argparse, csv, os, time
import numpy as np, torch
from xqai.network import PVNet
from xqai.mcts import PUCTPlanner
from xqai import arena

def load(path, dev):
    ck = torch.load(path, map_location=dev, weights_only=False)
    net = PVNet(ck.get("channels", 128), ck.get("blocks", 10)).to(dev).eval()
    net.load_state_dict(ck["model"])
    step = int(ck.get("step", ck.get("epoch", -1)))
    return net, step

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default="checkpoints/ref_coldstart.pt")
    ap.add_argument("--ckpt", default="checkpoints/latest.pt")
    ap.add_argument("--interval", type=float, default=240)
    ap.add_argument("--games", type=int, default=12)
    ap.add_argument("--n-sim", type=int, default=40)
    ap.add_argument("--duration", type=float, default=36000)
    ap.add_argument("--out", default="logs/elo_vs_ref.csv")
    ap.add_argument("--stop-file", default=".pipeline_stop")
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    dev = a.device
    ref_net, _ = load(a.ref, dev)
    # Evaluation must be deterministic-strength: disable root Dirichlet noise
    # (only useful for self-play exploration), else both players are weakened
    # and the measured Elo is noisier / biased.
    planner = PUCTPlanner(add_noise=False)
    refP = arena.NetPlayer(ref_net, planner, n_sim=a.n_sim, name="coldstart")

    # 多样化开局:确定性对弈若都从初始局面开始,每局相同 -> 无统计力。
    # 从初始局面随机走 6 步生成不同起点,每个起点做红黑各一局(配对)。
    # 池子要远大于单轮用量(64 >> games/2),且**每轮评测重新抽样**:
    # 否则所有 step 共用同一小撮固定开局,曲线各点高度相关、CI 虚假。
    from xqai._xqcore import Position
    _orng = np.random.default_rng(12345)
    POOL = []
    while len(POOL) < max(a.games // 2 + 2, 64):
        _p = Position(); _ok = True
        for _ in range(6):
            _lm = list(_p.legal_moves())
            if not _lm or _p.result() != 0:
                _ok = False; break
            _p.push(int(_orng.choice(_lm)))
        if _ok and _p.result() == 0:
            POOL.append(_p.fen())
    _pick_rng = np.random.default_rng()  # 非固定种子:每轮抽不同开局子集

    new = not os.path.exists(a.out)
    f = open(a.out, "a", newline="")
    w = csv.writer(f)
    if new:
        w.writerow(["wall", "step", "games", "winrate", "elo_diff", "lo", "hi"]); f.flush()

    t0 = time.time(); last_mtime = None
    while time.time() - t0 < a.duration:
        if os.path.exists(a.stop_file): break
        try:
            mt = os.path.getmtime(a.ckpt) if os.path.exists(a.ckpt) else None
            if mt is not None and mt != last_mtime:
                last_mtime = mt
                curP_net, step = load(a.ckpt, dev)
                curP = arena.NetPlayer(curP_net, planner, n_sim=a.n_sim, name="rl")
                n_ops = max(1, (a.games + 1) // 2)
                ops = [POOL[i] for i in _pick_rng.choice(len(POOL), size=n_ops,
                                                         replace=False)]
                r = arena.play_match(curP, refP, games=a.games, openings=ops)
                wr = r["a_score_rate"]; elo = arena.elo_from_winrate(wr)
                lo, hi = r.get("ci95", (None, None))
                w.writerow([int(time.time()), step, a.games, f"{wr:.3f}", f"{elo:.1f}",
                            f"{arena.elo_from_winrate(lo):.1f}" if lo is not None else "",
                            f"{arena.elo_from_winrate(hi):.1f}" if hi is not None else ""])
                f.flush()
                print(f"[vs_ref] step={step} games={a.games} winrate={wr:.3f} elo_diff={elo:+.1f}", flush=True)
                del curP_net; torch.cuda.empty_cache()
        except Exception as e:
            print(f"[vs_ref] eval err: {e}", flush=True)
        time.sleep(a.interval)
    f.close()

if __name__ == "__main__":
    main()
