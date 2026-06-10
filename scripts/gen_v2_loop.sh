#!/usr/bin/env bash
# v2 高质量数据生成循环:分块、断点续、CPU 满载(180 worker)、带质量门。
# 用法: VALID_PID=<前一个gen的PID> TARGET_SHARDS=1221 setsid bash scripts/gen_v2_loop.sh &
set -u
ROOT=/mnt/nvme3n1/gameTheory
PY="$ROOT/.venv/bin/python"
DATA="$ROOT/data/processed_v2"
RAW="$ROOT/data/raw/pikafish_v2"
LOG="$ROOT/logs/gen_v2.log"
TARGET="${TARGET_SHARDS:-1221}"
CHUNK="${CHUNK_GAMES:-2000}"
WORKERS="${GEN_WORKERS:-180}"
VALID_PID="${VALID_PID:-0}"

shards(){ ls "$DATA"/*.npz 2>/dev/null | wc -l; }
log(){ echo "[gen_v2 $(date '+%F %T')] $*" >> "$LOG"; }

# 1) 等在跑的验证块结束,避免两个 gen 同时按 len(existing) 算分片号而撞号
if [ "$VALID_PID" -gt 0 ]; then
  log "waiting for in-flight chunk PID $VALID_PID ..."
  while kill -0 "$VALID_PID" 2>/dev/null; do sleep 10; done
  log "in-flight chunk finished; shards now=$(shards)"
fi

# 2) 质量门:首个 v2 shard 必须 pi 行和≈1、z 连续,否则中止(别白跑 11h)
"$PY" - <<'PYEOF' >> "$LOG" 2>&1
import glob, numpy as np, sys
fs=sorted(glob.glob("/mnt/nvme3n1/gameTheory/data/processed_v2/shard_*.npz"))
if not fs: print("[gen_v2] GATE FAIL: no shards"); sys.exit(1)
d=np.load(fs[0]); pi=d["pi"].astype(np.float32); z=d["z"].astype(np.float32)
s=pi.sum(1)
ok = (s.min()>=0.99 and s.max()<=1.01 and len(np.unique(z))>10)
print(f"[gen_v2] GATE {'PASS' if ok else 'FAIL'}: shards={len(fs)} "
      f"pi_sum[{s.min():.3f},{s.max():.3f}] z_uniq={len(np.unique(z))} "
      f"nzero/row[{int((pi>0).sum(1).min())},{int((pi>0).sum(1).max())}]")
sys.exit(0 if ok else 1)
PYEOF
if [ $? -ne 0 ]; then log "QUALITY GATE FAILED -> abort"; exit 1; fi

# 3) 分块循环生成到目标分片数
log "loop start target=$TARGET chunk=$CHUNK workers=$WORKERS shards=$(shards)"
while [ "$(shards)" -lt "$TARGET" ]; do
  [ -f "$ROOT/.gen_v2_stop" ] && { log "stop file -> exit"; break; }
  "$PY" "$ROOT/scripts/gen_pikafish_data.py" --games "$CHUNK" --workers "$WORKERS" \
     --depth 14 --multipv 8 --policy-temp 100 --value-scale 500 \
     --out "$DATA" --raw-out "$RAW" >> "$LOG" 2>&1
  log "progress shards=$(shards)/$TARGET"
done
log "DONE shards=$(shards)"
