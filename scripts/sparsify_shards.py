"""把 v2 稠密 pi[8100] shard 转成 稀疏(pi_idx/pi_val,K=8)+ 不压缩(可 mmap)。
逐片处理、不压缩存储,供 ShardDataset mmap 共享页缓存,根治 OOM。"""
import sys, os, glob, numpy as np
import multiprocessing as mp
SRC=sys.argv[1] if len(sys.argv)>1 else "data/processed_v2"
DST=sys.argv[2] if len(sys.argv)>2 else "data/processed_v2_mmap"
K=8
def convert(f):
    d=np.load(f); planes=d["planes"]; z=d["z"]
    if "pi" not in d.files:  # 已是稀疏/旧格式,跳过
        return f"skip {os.path.basename(f)}"
    pi=d["pi"].astype(np.float32)                      # [N,8100]
    top=np.argpartition(pi,-K,axis=1)[:,-K:]           # 每行 top-K 索引
    val=np.take_along_axis(pi,top,axis=1)              # 对应概率
    out=os.path.join(DST,os.path.basename(f))
    np.savez(out,                                       # 不压缩→可 mmap
             planes=planes.astype(np.float16),
             pi_idx=top.astype(np.int16),
             pi_val=val.astype(np.float16),
             z=z.astype(np.float16))
    return f"ok {os.path.basename(f)} sum={val.sum(1).mean():.3f}"
if __name__=="__main__":
    os.makedirs(DST,exist_ok=True)
    files=sorted(glob.glob(os.path.join(SRC,"shard_*.npz")))
    print(f"convert {len(files)} shards {SRC} -> {DST} (K={K}, uncompressed)")
    with mp.Pool(8) as p:                              # 限 8 并行,避免再 OOM
        for i,r in enumerate(p.imap_unordered(convert,files,chunksize=4)):
            if i%200==0: print(f"  {i}/{len(files)} {r}",flush=True)
    print("done")
