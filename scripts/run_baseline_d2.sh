#!/usr/bin/env bash
# D2 基线评测:3 底座 ckpt × 7 锚点 × 4 分片 × 16 局 = 84 分片 / 1344 局。
#
# 排程:84 个分片一次性全部铺开,GPU 轮转分配(每卡 10~11 个分片,每分片显存
# <1GB,A800-40G 远未打满);其中带引擎的分片 = 3 模型 × 6 引擎锚点 × 4 分片
# = 72 个引擎子进程(pikafish/xqab 均单线程),< 90 的并发上限,无需分批。
# OMP_NUM_THREADS=2 防 84 进程 × torch 默认全核 intra-op 把 192 线程打爆。
# 预计:瓶颈是 net n_sim=200 的 PUCT(每步 ~0.5s) + xqab:500(每步 0.5s),
# 单分片 16 局 ≈ 30–60 min,全并行 → 总墙钟 ≈ 1–1.5h(< 2h 目标)。
#
# 用法: bash scripts/run_baseline_d2.sh
# 只启动 + 打印 PID 清单,不阻塞等待。进度看 logs/d2/run/*.log,
# 结果 CSV 在 logs/d2/(每分片独立文件,避免并发追加同一 CSV 撕裂),
# 对局收割 jsonl 在 data/dagger/games_raw/(对局记录格式;由
# scripts/games_to_positions.py 复盘转换成局面记录后才进 queue_p0)。
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="$ROOT/.venv/bin/python"
OUTDIR="$ROOT/logs/d2"
RUNLOG="$ROOT/logs/d2/run"
DUMPDIR="$ROOT/data/dagger/games_raw"
mkdir -p "$OUTDIR" "$RUNLOG" "$DUMPDIR"
cd "$ROOT"

# 三个底座(名字 -> ckpt)
MODELS_KEYS=(v1 v2cold safe)
declare -A MODELS=(
  [v1]="$ROOT/checkpoints/final_best.pt"
  [v2cold]="$ROOT/checkpoints_v2_rl/ref_coldstart_frozen.pt"
  [safe]="$ROOT/checkpoints_vsafe2_safe/latest.pt"
)
ANCHORS=(random xqab:200 xqab:500 pikafish:1 pikafish:2 pikafish:3 pikafish:4)
NSHARDS=4
GAMES=16
NGPU=8

for m in "${MODELS_KEYS[@]}"; do
  if [[ ! -f "${MODELS[$m]}" ]]; then
    echo "[FATAL] missing ckpt: ${MODELS[$m]}" >&2; exit 1
  fi
done

gpu=0
total=0
PIDFILE="$RUNLOG/pids.txt"
: > "$PIDFILE"
echo "== D2 baseline launch $(date '+%F %T') =="
for m in "${MODELS_KEYS[@]}"; do
  for anc in "${ANCHORS[@]}"; do
    aname="${anc//:/-}"
    for s in $(seq 0 $((NSHARDS - 1))); do
      seed=$((12345 + 7777 * s))                       # 分片独立开局池,防对弈重复
      name="d2_${m}_${aname}_s${s}"
      OMP_NUM_THREADS=2 nohup "$PY" "$ROOT/scripts/eval_anchor.py" \
        --ckpt "${MODELS[$m]}" \
        --opponent "$anc" \
        --gpu "$gpu" \
        --games "$GAMES" \
        --opening-seed "$seed" \
        --tag "d2_${m}_s${s}" \
        --out "$OUTDIR/${name}.csv" \
        --dump-games "$DUMPDIR/${name}.jsonl" \
        > "$RUNLOG/${name}.log" 2>&1 &
      pid=$!
      echo "$pid gpu$gpu $name" | tee -a "$PIDFILE"
      total=$((total + 1))
      gpu=$(((gpu + 1) % NGPU))
    done
  done
done
echo "== launched $total shards (PID list: $PIDFILE) =="
echo "watch:  ls $OUTDIR/*.csv | wc -l   # 84 = all done"
echo "        tail -n1 $RUNLOG/*.log"
