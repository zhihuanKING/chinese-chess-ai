"""决定性诊断:冷启动模型 vs 随机走子。
赢不了随机 → 下棋路径(NetPlayer/视角)有 bug;碾压随机 → 模型OK,是AB标定问题。"""
import numpy as np, torch
from xqai._xqcore import Position
from xqai.network import PVNet
from xqai.mcts import PUCTPlanner
from xqai import arena

class RandomPlayer:
    name = "random"
    def __init__(self): self.rng = np.random.default_rng(0)
    def reset(self): pass
    def close(self): pass
    def select_move(self, pos):
        lm = list(pos.legal_moves())
        return int(self.rng.choice(lm)) if lm else -1

def load(path):
    ck = torch.load(path, map_location="cuda:0", weights_only=False)
    net = PVNet(ck.get("channels", 128), ck.get("blocks", 10)).cuda().eval()
    net.load_state_dict(ck["model"]); return net

net = load("checkpoints/pretrained_best.pt")
# 也单独看：纯策略(n_sim 极小)选的首着是否合理
planner = PUCTPlanner()
p0 = Position()
pi = planner.search([p0], net, 16)[0]
top = int(np.argmax(pi))
print(f"初始局面 模型首选(规范化index)= {top} -> from={top//90} to={top%90}")

P = arena.NetPlayer(net, planner, n_sim=40, name="pretrained")
R = RandomPlayer()
print("对随机走子 8 局:", arena.play_match(P, R, games=8))
