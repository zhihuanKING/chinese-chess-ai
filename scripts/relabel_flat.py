"""v2 数据标签离线修复（任务 A0.2）：基于 data/processed_v2_flat 生成新标签。

不动 planes/pi_idx，只生成三个新文件（与原行序一一对应）：

- ``z_wdl.npy``  [N]f16：value 标签 WDL 重校准。
  旧 z=tanh(root_cp/500)（行棋方视角，prepare_data.encode_soft_sample）可逆：
  cp = 500*atanh(clip(z, ±0.9995))；再经 logs/wdl_calib.json 的 logistic
  E(cp)=1/(1+exp(-(cp-b)/s)) 映射 z_new = 2E(cp)-1（仍为行棋方视角，不翻符号，
  因为校准用的 cp/wdl 也是行棋方视角的 UCI 输出）。
  饱和样本（|z| >= tanh(2000/500)，含 mate 标注的 ±1）直接 sign(z)*z_cap，
  z_cap = 2E(2000)-1（取自校准 json）。

- ``pi_val_T50.npy`` / ``pi_val_T30.npy`` [N,8]f16：policy 温度变体。
  原 pi=softmax(cp/100)（top-K 稀疏，K=8，pi_idx 定宽行、空位 val=0），
  温度 T 重调即 pi^(100/T) 后按行（仅非零项）重归一化——对行内缩放不变，
  所以不依赖原行和是否精确为 1。padding 的 0 保持 0。

用法::

    .venv/bin/python scripts/relabel_flat.py \
        --flat data/processed_v2_flat --calib logs/wdl_calib.json
"""
from __future__ import annotations

import argparse
import json
import math
import os

import numpy as np

Z_CLIP = 0.9995          # atanh 反演裁剪
SAT_CP = 2000.0          # 饱和判定阈值（cp）
TEMPS = (50.0, 30.0)     # 温度变体（原 T=100）


def load_calib(path: str):
    with open(path) as fh:
        c = json.load(fh)
    return float(c["b"]), float(c["s"]), float(c["z_cap"])


def relabel_z(z_old: np.ndarray, b: float, s: float, z_cap: float) -> np.ndarray:
    """fp16[N] -> fp16[N]，全程 float64 计算后落 fp16。"""
    z = z_old.astype(np.float64)
    sat_thr = math.tanh(SAT_CP / 500.0)               # ≈0.99933
    sat = np.abs(z) >= sat_thr
    cp = 500.0 * np.arctanh(np.clip(z, -Z_CLIP, Z_CLIP))
    e = 1.0 / (1.0 + np.exp(-(cp - b) / s))
    z_new = 2.0 * e - 1.0
    z_new[sat] = np.sign(z[sat]) * z_cap
    return z_new.astype(np.float16)


def relabel_pi(pi_val: np.ndarray, temp: float) -> np.ndarray:
    """fp16[M,8] -> fp16[M,8]：非零项 ^(100/T) 后按行重归一化，零保持零。"""
    p = pi_val.astype(np.float64)
    k = 100.0 / temp
    pw = np.zeros_like(p)
    nz = p > 0
    pw[nz] = np.exp(k * np.log(p[nz]))                # p^k，f64 不会下溢出 0
    rs = pw.sum(axis=1, keepdims=True)
    np.divide(pw, rs, out=pw, where=rs > 0)
    return pw.astype(np.float16)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--flat", default="data/processed_v2_flat")
    ap.add_argument("--calib", default="logs/wdl_calib.json")
    ap.add_argument("--chunk", type=int, default=1 << 20)
    args = ap.parse_args()

    b, s, z_cap = load_calib(args.calib)
    print(f"[relabel] calib b={b:.2f} s={s:.2f} z_cap={z_cap:.6f}", flush=True)

    j = lambda n: os.path.join(args.flat, n)
    z_old = np.load(j("z.npy"), mmap_mode="r")
    pi_val = np.load(j("pi_val.npy"), mmap_mode="r")
    N = z_old.shape[0]
    assert pi_val.shape == (N, 8), pi_val.shape

    # ---- z_wdl --------------------------------------------------------- #
    from numpy.lib.format import open_memmap
    z_out = open_memmap(j("z_wdl.npy"), mode="w+", dtype=np.float16, shape=(N,))
    n_sat = 0
    sat_thr = math.tanh(SAT_CP / 500.0)
    for off in range(0, N, args.chunk):
        blk = np.asarray(z_old[off:off + args.chunk])
        z_out[off:off + len(blk)] = relabel_z(blk, b, s, z_cap)
        n_sat += int((np.abs(blk.astype(np.float64)) >= sat_thr).sum())
    z_out.flush()
    print(f"[relabel] z_wdl.npy done N={N} saturated={n_sat} "
          f"({n_sat/N:.3%})", flush=True)

    # ---- 分布前后对比 --------------------------------------------------- #
    zo = np.asarray(z_old).astype(np.float64)
    zn = np.asarray(np.load(j("z_wdl.npy"), mmap_mode="r")).astype(np.float64)
    qs = (1, 5, 25, 50, 75, 95, 99)
    for name, a in (("old(tanh cp/500)", zo), ("new(2E-1)", zn)):
        print(f"[relabel] z {name}: mean={a.mean():+.4f} std={a.std():.4f} "
              f"|z|<0.05: {np.mean(np.abs(a) < 0.05):.3%} "
              f"|z|>0.9: {np.mean(np.abs(a) > 0.9):.3%}", flush=True)
        print(f"           q{qs} = "
              f"{np.round(np.percentile(a, qs), 4).tolist()}", flush=True)

    # ---- 温度变体 -------------------------------------------------------- #
    underflow = {t: 0 for t in TEMPS}
    outs = {t: open_memmap(j(f"pi_val_T{int(t)}.npy"), mode="w+",
                           dtype=np.float16, shape=(N, 8)) for t in TEMPS}
    for off in range(0, N, args.chunk):
        blk = np.asarray(pi_val[off:off + args.chunk])
        nz_old = blk > 0
        for t in TEMPS:
            out = relabel_pi(blk, t)
            underflow[t] += int((nz_old & (out == 0)).sum())
            outs[t][off:off + len(blk)] = out
    for t in TEMPS:
        outs[t].flush()
        print(f"[relabel] pi_val_T{int(t)}.npy done "
              f"nonzero->0 underflow(f16)={underflow[t]}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
