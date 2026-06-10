#!/usr/bin/env python
"""监督续训对照(无自对弈):从冷启动继续在 Pikafish 软标签上训练。

用途:为 value-safe RL 的 `safe` arm(冻 value + 40% 监督 rehearsal,涨到 ~0.85 vs 冷启动)
提供归因对照 —— 若纯监督续训也能涨到同等水平,则增益主要来自"更多教师蒸馏"而非自对弈;
若纯监督续训仅持平/小涨而 safe 明显更高,则被锚定的自对弈确有正贡献。

与 safe arm 同 init / 同 lr / 同冻 value / 同导出节奏,唯一差别:100% 监督、0 自对弈。
"""
from __future__ import annotations
import argparse, glob, os, sys, time
import numpy as np, torch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
from xqai.network import PVNet


def load_pool(data_dir, n_shards, seed=0):
    files = sorted(glob.glob(os.path.join(data_dir, "shard_*.npz")))
    rng = np.random.default_rng(seed); rng.shuffle(files)
    files = files[:n_shards]
    pl, pis, zs = [], [], []
    for f in files:
        d = np.load(f)
        if "pi" not in d.files:
            continue
        pl.append(np.asarray(d["planes"], dtype=np.float16))
        pi = np.asarray(d["pi"], dtype=np.float32)
        s = pi.sum(1, keepdims=True); s[s == 0] = 1.0; pi /= s
        pis.append(pi.astype(np.float16))
        zs.append(np.asarray(d["z"], dtype=np.float16))
    P = torch.from_numpy(np.concatenate(pl))
    PI = torch.from_numpy(np.concatenate(pis))
    Z = torch.from_numpy(np.concatenate(zs))
    print(f"[sup] pool {P.shape[0]} samples / {len(files)} shards", flush=True)
    return P, PI, Z


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", default="checkpoints_v2_rl/best.pt")
    ap.add_argument("--data", default="data/processed_v2")
    ap.add_argument("--shards", type=int, default=60)
    ap.add_argument("--steps", type=int, default=10000)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=1.0e-3)
    ap.add_argument("--weight-decay", type=float, default=1.0e-4)
    ap.add_argument("--freeze-value", action="store_true")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--export-every", type=int, default=800)
    ap.add_argument("--ckpt-dir", default="checkpoints_vsafe_suponly")
    args = ap.parse_args()

    dev = torch.device(f"cuda:{args.gpu}")
    ck = torch.load(args.init, map_location=dev, weights_only=False)
    C, B = int(ck.get("channels", 256)), int(ck.get("blocks", 15))
    model = PVNet(C, B).to(dev)
    model.load_state_dict(ck["model"]); model.train()
    if args.freeze_value:
        for mn in ("value_conv", "value_fc1", "value_fc2"):
            mod = getattr(model, mn, None)
            if mod is not None:
                for p in mod.parameters():
                    p.requires_grad = False
                mod.eval()  # BN running stats are buffers; freeze them too
        print("[sup] value head FROZEN (params + BN stats; vloss excluded "
              "from total loss)", flush=True)

    P, PI, Z = load_pool(args.data, args.shards, seed=11)
    n = P.shape[0]
    rng = np.random.default_rng(123)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=args.lr, weight_decay=args.weight_decay)

    os.makedirs(args.ckpt_dir, exist_ok=True)
    path = os.path.join(args.ckpt_dir, "latest.pt")

    def export(step):
        state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        tmp = f"{path}.tmp.{os.getpid()}"
        torch.save({"model": state, "channels": C, "blocks": B, "step": step}, tmp)
        os.replace(tmp, path)

    export(0)
    t_log = 0.0
    for step in range(1, args.steps + 1):
        idx = torch.from_numpy(rng.integers(0, n, size=args.batch))
        planes = P[idx].float().to(dev, non_blocking=True)
        pi = PI[idx].float().to(dev, non_blocking=True)
        z = Z[idx].float().to(dev, non_blocking=True)
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits, value = model(planes)
            logp = torch.log_softmax(logits.float(), dim=1)
            ploss = -(pi.float() * logp).sum(1).mean()
            vloss = torch.nn.functional.mse_loss(value.float().squeeze(1), z.float())
            # 冻结时 vloss 不计入总损失:其梯度会穿过冻结头进 trunk,破坏隔离。
            loss = ploss if args.freeze_value else vloss + ploss
        loss.backward(); opt.step()
        if step % args.export_every == 0:
            export(step)
        if time.time() - t_log >= 10:
            print(f"[sup] step={step} loss={float(loss):.4f}", flush=True)
            t_log = time.time()
    export(args.steps)
    print(f"[sup] done {args.steps} steps -> {path}", flush=True)


if __name__ == "__main__":
    main()
