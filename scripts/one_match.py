"""单场评估: ckpt vs baseline(random|net:<path>|xqab:<ms>), 多样开局。用于并行拼曲线/对照。"""
import argparse, numpy as np, torch
from xqai._xqcore import Position
from xqai.network import PVNet
from xqai.mcts import PUCTPlanner
from xqai import arena

def load_net(path, dev):
    ck = torch.load(path, map_location=dev, weights_only=False)
    net = PVNet(ck.get("channels", 128), ck.get("blocks", 10)).to(dev).eval()
    net.load_state_dict(ck["model"]); return net

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

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
ap.add_argument("--baseline", required=True)
ap.add_argument("--n-sim", type=int, default=200)
ap.add_argument("--games", type=int, default=20)
ap.add_argument("--tag", default="")
a = ap.parse_args()
dev = "cuda:0"
net = load_net(a.ckpt, dev); planner = PUCTPlanner(add_noise=False)
A = arena.NetPlayer(net, planner, n_sim=a.n_sim, name="cand")
if a.baseline == "random":
    B = RandomPlayer()
elif a.baseline.startswith("net:"):
    B = arena.NetPlayer(load_net(a.baseline[4:], dev), planner, n_sim=a.n_sim, name="ref")
elif a.baseline.startswith("xqab:"):
    B = arena.SubprocessUCCIPlayer(["cpp/build/xqab"], movetime_ms=int(a.baseline[5:]))
else:
    raise SystemExit("bad baseline")
ops = gen_openings(max(a.games // 2 + 2, 12))
r = arena.play_match(A, B, games=a.games, openings=ops)
wr = r["a_score_rate"]
print(f"RESULT tag={a.tag} winrate={wr:.3f} elo={arena.elo_from_winrate(wr):.0f} "
      f"W{r['a_wins']}/D{r['a_draws']}/L{r['a_losses']}", flush=True)
try:
    if hasattr(B, "close"): B.close()
except Exception: pass
