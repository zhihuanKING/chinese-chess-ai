"""诊断:RL最新 vs 预训练，预训练/RL vs 不同强度 Alpha-Beta。判断 RL 是否退化模型。"""
import torch
from xqai._xqcore import Position
from xqai.network import PVNet
from xqai.mcts import PUCTPlanner
from xqai import arena

DEV = "cuda:0"

def load_net(path):
    ck = torch.load(path, map_location=DEV, weights_only=False)
    if isinstance(ck, dict) and "model" in ck:
        C = ck.get("channels", 128); B = ck.get("blocks", 10); sd = ck["model"]
    else:
        C, B, sd = 128, 10, ck
    net = PVNet(C, B).to(DEV).eval()
    net.load_state_dict(sd)
    return net, f"{C}x{B}"

pre, ds_pre = load_net("checkpoints/pretrained_best.pt")
lat, ds_lat = load_net("checkpoints/latest.pt")
print(f"pretrained={ds_pre}  latest={ds_lat}")
planner = PUCTPlanner()
P_pre = arena.NetPlayer(pre, planner, n_sim=40, name="pretrained")
P_lat = arena.NetPlayer(lat, planner, n_sim=40, name="rl_latest")
ab50 = arena.SubprocessUCCIPlayer(["cpp/build/xqab"], movetime_ms=50)
ab200 = arena.SubprocessUCCIPlayer(["cpp/build/xqab"], movetime_ms=200)

def run(a, b, g, tag):
    r = arena.play_match(a, b, games=g)
    print(f"[{tag}] ->", r)

try:
    run(P_pre, ab50, 8, "预训练 vs xqab@50ms(弱AB,看冷启动有无棋力)")
    run(P_pre, ab200, 8, "预训练 vs xqab@200ms(强AB)")
    run(P_lat, P_pre, 8, "RL最新 vs 预训练(确认RL是否退化)")
finally:
    for e in (ab50, ab200):
        try: e.close()
        except Exception: pass
