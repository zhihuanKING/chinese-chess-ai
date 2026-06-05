#!/usr/bin/env bash
# 流水线健康巡检：打印 HEALTHY/WARN/CRITICAL + 关键指标 + 原因。
# 给人看，也给定时监控循环用。检测：进程存活/学习步停滞/loss异常/buffer增长/GPU占用/Elo更新。
cd /mnt/nvme3n1/gameTheory
ST=logs/.health_prev          # 上次巡检状态(step,time)
verdict=HEALTHY; reasons=()

log(){ echo -e "$1"; }
warn(){ verdict=WARN; reasons+=("WARN: $1"); }
crit(){ verdict=CRITICAL; reasons+=("CRIT: $1"); }

now=$(date +%s)

# --- 1) 进程存活 ---
pipe_alive=0; rl_alive=0; eval_alive=0
[ -f logs/pipeline.pid ] && kill -0 "$(cat logs/pipeline.pid)" 2>/dev/null && pipe_alive=1
[ -f logs/rl.pid ] && kill -0 "$(cat logs/rl.pid)" 2>/dev/null && rl_alive=1
[ -f logs/eval.pid ] && kill -0 "$(cat logs/eval.pid)" 2>/dev/null && eval_alive=1
log "进程: pipeline=$pipe_alive rl=$rl_alive eval=$eval_alive"

# --- 2) 当前阶段 ---
[ -f logs/PIPELINE_STATUS.md ] && { log "--- PIPELINE_STATUS 末尾 ---"; tail -4 logs/PIPELINE_STATUS.md; }
grep -q "FAIL" logs/PIPELINE_STATUS.md 2>/dev/null && crit "PIPELINE_STATUS 出现 FAIL"

# --- 3) RL 学习进度(从最新 [learner] step= 行) ---
rlline=$(grep -h "\[learner\] step=" logs/stage2_rl.log 2>/dev/null | tail -1)
if [ -n "$rlline" ]; then
  log "RL: $rlline"
  step=$(echo "$rlline" | grep -oP 'step=\K[0-9]+'); loss=$(echo "$rlline" | grep -oP 'loss=\K[0-9.naN-]+'); buf=$(echo "$rlline" | grep -oP 'buf=\K[0-9]+')
  # loss 异常
  echo "$loss" | grep -qiE "nan|inf" && crit "loss=$loss 非有限"
  # 停滞检测(与上次比)
  if [ -f "$ST" ]; then
    pstep=$(cut -d' ' -f1 "$ST"); ptime=$(cut -d' ' -f2 "$ST")
    if [ -n "$pstep" ] && [ "$step" = "$pstep" ] && [ $((now-ptime)) -gt 300 ] && [ "$rl_alive" = 1 ]; then
      crit "学习步停滞: step 在 $((now-ptime))s 内未变(=$step)"
    fi
  fi
  echo "$step $now" > "$ST"
elif [ "$rl_alive" = 1 ]; then
  warn "RL 进程在但日志无 [learner] step= 行"
fi

# --- 4) 数据分片 ---
shards=$(ls data/processed/*.npz 2>/dev/null | wc -l); log "数据分片: $shards"

# --- 5) GPU 占用(训练时应有卡 util>0 / mem>1GB) ---
log "--- GPU ---"; nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader 2>/dev/null
busy=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | awk '$1>10' | wc -l)
[ "$rl_alive" = 1 ] && [ "$busy" -eq 0 ] && warn "RL 在跑但所有 GPU util≈0(可能 CPU 瓶颈或卡死)"

# --- 6) Elo 曲线更新 ---
if [ -f logs/elo_curve.csv ]; then
  rows=$(($(wc -l < logs/elo_curve.csv)-1)); last=$(tail -1 logs/elo_curve.csv)
  log "Elo 曲线: ${rows}点, 最新: $last"
fi

# --- 7) /dev/shm 残留(异常退出迹象) ---
shm=$(ls /dev/shm 2>/dev/null | grep -i xqai | wc -l); [ "$shm" -gt 2 ] && warn "/dev/shm xqai 段=$shm(可能泄漏)"

echo "================="
echo "巡检结论: $verdict"
for r in "${reasons[@]}"; do echo "  - $r"; done
[ "$verdict" = HEALTHY ] && exit 0 || { [ "$verdict" = WARN ] && exit 1 || exit 2; }
