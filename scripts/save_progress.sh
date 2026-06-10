#!/usr/bin/env bash
# 自动进度保存:由 Stop hook 每次回答结束时调用。仅在状态变化时追加快照(有进度才存)。
cd /mnt/nvme3n1/gameTheory 2>/dev/null || exit 0
LOG="docx/进度自动保存.md"
MARK="logs/.progress_last"
mkdir -p logs docx 2>/dev/null
[ -f "$LOG" ] || echo "# 进度自动保存日志(Stop hook 自动追加,有变化才记)" > "$LOG"

alive=no; kill -0 "$(cat logs/pipeline.pid 2>/dev/null)" 2>/dev/null && alive=yes
stage=$(grep -E "^\| (STAGE|DONE)" logs/PIPELINE_STATUS.md 2>/dev/null | tail -1 | tr -s ' ' | cut -c1-60)
rlstep=$(grep "\[learner\] step=" logs/stage2_rl.log 2>/dev/null | tail -1 | grep -oE "step=[0-9]+ loss=[0-9.]+")
elo=$(tail -1 logs/elo_vs_ref.csv 2>/dev/null | cut -c1-60)
shards=$(ls data/processed/*.npz 2>/dev/null | wc -l)
sig="${alive}|${stage}|${rlstep}|${elo}|${shards}"
[ -f "$MARK" ] && [ "$(cat "$MARK" 2>/dev/null)" = "$sig" ] && exit 0   # 无变化,不存
echo "$sig" > "$MARK"
echo "- $(date '+%F %T') | pipeline=$alive | ${stage} | RL:${rlstep:-none} | shards:${shards} | elo_vs_ref:${elo}" >> "$LOG"
exit 0
