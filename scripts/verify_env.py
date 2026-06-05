"""全面环境复检：8 卡 GPU + _xqcore + PVNet GPU(BF16) 前向 + MCTS 真网络自对弈。"""
import time, numpy as np, torch

def section(t): print(f"\n=== {t} ===")

section("torch / CUDA")
print("torch", torch.__version__, "| cuda", torch.version.cuda,
      "| avail", torch.cuda.is_available(), "| ndev", torch.cuda.device_count(),
      "| bf16", torch.cuda.is_bf16_supported())
assert torch.cuda.is_available(), "CUDA 不可用!"
assert torch.cuda.device_count() == 8, "不是 8 卡!"

section("逐卡 GPU 运算(matmul + BF16)")
for i in range(torch.cuda.device_count()):
    d = f"cuda:{i}"
    x = torch.randn(4096, 4096, device=d, dtype=torch.bfloat16)
    t0 = time.time(); y = x @ x; torch.cuda.synchronize(i); dt = time.time() - t0
    assert torch.isfinite(y.float()).all()
    print(f"  GPU{i} {torch.cuda.get_device_name(i)}  bf16 4k matmul {dt*1000:.1f}ms  ok")

section("_xqcore 规则内核")
from xqai._xqcore import Position
p = Position()
print("  初始合法着法", len(p.legal_moves()), "| perft3", Position.perft(p.fen(), 3),
      "| perft4", Position.perft(p.fen(), 4))
assert Position.perft(p.fen(), 3) == 79666 and Position.perft(p.fen(), 4) == 3290240

section("PVNet GPU 前向(BF16),两种规模")
from xqai.network import PVNet
from xqai import encoding
for ch, bl in [(128, 10), (256, 15)]:
    net = PVNet(ch, bl).cuda().eval()
    nparam = sum(p.numel() for p in net.parameters())
    x = torch.randn(256, encoding.NUM_PLANES, 10, 9, device="cuda")
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        pol, val = net(x)
    assert tuple(pol.shape) == (256, encoding.ACTION_DIM) and tuple(val.shape) == (256, 1)
    assert torch.isfinite(pol.float()).all() and val.min() >= -1.01 and val.max() <= 1.01
    print(f"  PVNet({ch},{bl}) {nparam/1e6:.1f}M  GPU前向 policy{tuple(pol.shape)} value{tuple(val.shape)} ok")
    del net; torch.cuda.empty_cache()

section("真网络 + PUCT-MCTS GPU 自对弈(向量化批量)")
from xqai.mcts import PUCTPlanner
net = PVNet(128, 10).cuda().eval()
planner = PUCTPlanner()
positions = [Position() for _ in range(8)]
t0 = time.time()
pis = planner.search(positions, net, n_sim=32)
dt = time.time() - t0
assert len(pis) == 8 and all(abs(float(np.sum(pi)) - 1.0) < 1e-3 for pi in pis)
print(f"  8 局并行 PUCT(n_sim=32) GPU 推理 {dt*1000:.0f}ms, π 和=1 ✔")

print("\n✅✅ 全面复检通过：8 卡 GPU + 规则内核 + 网络 GPU 前向 + MCTS 真网络自对弈 全部正常。")
