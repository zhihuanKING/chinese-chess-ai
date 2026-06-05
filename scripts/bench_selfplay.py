"""单卡自对弈吞吐基准：不同并行对局数下 PUCT(n_sim) 的 moves/s 与估算局/小时。"""
import time, torch
from xqai._xqcore import Position
from xqai.network import PVNet
from xqai.mcts import PUCTPlanner

net = PVNet(128, 10).cuda().eval()
planner = PUCTPlanner()
NSIM = 32
print(f"PVNet(128,10) on {torch.cuda.get_device_name(0)}, n_sim={NSIM}, 假设~100步/局\n")
for B in [64, 128, 256, 512]:
    pos = [Position() for _ in range(B)]
    planner.search(pos, net, NSIM)            # warmup
    torch.cuda.synchronize()
    t0 = time.time(); iters = 5
    for _ in range(iters):
        planner.search(pos, net, NSIM)
    torch.cuda.synchronize()
    dt = time.time() - t0
    mps = B * iters / dt
    print(f"  并行={B:4d}  {dt/iters*1000:6.0f} ms/ply  {mps:7.0f} moves/s  单卡≈{mps/100*3600:7.0f} 局/小时")
print("\n注：6 张 actor 卡并行 → 总吞吐 ≈ 单卡 ×6")
