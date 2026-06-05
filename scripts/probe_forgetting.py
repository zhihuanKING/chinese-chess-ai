"""快速量化"灾难性遗忘":在留出的冷启动数据上比较各 checkpoint 的 policy top-1 / CE / value MSE。
不打对弈,几秒出结果——证据而非假设。"""
import sys, glob, numpy as np, torch
import torch.nn.functional as F
from xqai.network import PVNet

DEV = "cuda:0"

def load(path):
    ck = torch.load(path, map_location=DEV, weights_only=False)
    C = ck.get("channels", 128); B = ck.get("blocks", 10)
    net = PVNet(C, B).to(DEV).eval(); net.load_state_dict(ck["model"])
    return net, f"{C}x{B}"

# 留出集:取最后 3 个分片(预训练用的是全部，这里只为相对比较，足够)
fs = sorted(glob.glob("data/processed/shard_*.npz"))[-3:]
P, PI, Z = [], [], []
for f in fs:
    d = np.load(f)
    P.append(d["planes"]); Z.append(d["z"])
    PI.append(d["pi_index"] if "pi_index" in d.files else d["pi"])
P = np.concatenate(P); PI = np.concatenate(PI); Z = np.concatenate(Z)
n = min(4096, len(P))
idx = np.random.default_rng(0).choice(len(P), size=n, replace=False)
x = torch.tensor(P[idx], dtype=torch.float32, device=DEV)
tgt = torch.tensor(PI[idx].astype(np.int64), device=DEV)
zt = torch.tensor(Z[idx].astype(np.float32), device=DEV)
print(f"留出样本 n={n}  (分片 {len(fs)} 个)")

def evalnet(path, tag):
    try:
        net, ds = load(path)
    except Exception as e:
        print(f"{tag}: 加载失败 {e}"); return
    with torch.no_grad():
        pol, val = net(x)
        top1 = (pol.argmax(1) == tgt).float().mean().item()
        ce = F.cross_entropy(pol, tgt).item()
        vmse = ((val.squeeze(1) - zt) ** 2).mean().item()
    print(f"{tag:28s}[{ds}] policy_top1={top1:.3f}  CE={ce:.3f}  value_MSE={vmse:.3f}")

for path, tag in [
    ("checkpoints/pretrained_best.pt", "pretrained(冷启动)"),
    ("checkpoints/latest.pt", "latest(SGD0.02 RL 后)"),
]:
    evalnet(path, tag)
print("\n判读: 若 latest 的 top1 远低于 pretrained / CE 暴涨 → SGD0.02 确实灾难性遗忘。")
