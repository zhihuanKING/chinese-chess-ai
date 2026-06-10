"""把稀疏 shard 拼成扁平 .npy memmap 存储(行级 mmap,根治随机访问 OOM/慢)。
输出 4 个文件到 OUT/: planes.npy[N,15,10,9]f16, pi_idx.npy[N,8]i16,
pi_val.npy[N,8]f16, z.npy[N]f16。逐片写、低内存。"""
import sys, os, glob, numpy as np
from numpy.lib.format import open_memmap
SRC=sys.argv[1] if len(sys.argv)>1 else "data/processed_v2_mmap"
OUT=sys.argv[2] if len(sys.argv)>2 else "data/processed_v2_flat"
os.makedirs(OUT,exist_ok=True)
files=sorted(glob.glob(os.path.join(SRC,"shard_*.npz")))
lens=[]
for f in files:
    with np.load(f,mmap_mode="r") as d: lens.append(int(d["planes"].shape[0]))
N=sum(lens); print(f"{len(files)} shards, N={N} samples -> {OUT}")
mp=open_memmap(os.path.join(OUT,"planes.npy"),mode="w+",dtype=np.float16,shape=(N,15,10,9))
mi=open_memmap(os.path.join(OUT,"pi_idx.npy"),mode="w+",dtype=np.int16,shape=(N,8))
mv=open_memmap(os.path.join(OUT,"pi_val.npy"),mode="w+",dtype=np.float16,shape=(N,8))
mz=open_memmap(os.path.join(OUT,"z.npy"),mode="w+",dtype=np.float16,shape=(N,))
off=0
for i,f in enumerate(files):
    d=np.load(f); n=d["planes"].shape[0]
    mp[off:off+n]=d["planes"]; mi[off:off+n]=d["pi_idx"]
    mv[off:off+n]=d["pi_val"]; mz[off:off+n]=d["z"]
    off+=n
    if i%200==0: print(f"  {i}/{len(files)} off={off}",flush=True)
mp.flush();mi.flush();mv.flush();mz.flush()
print(f"done N={off}")
