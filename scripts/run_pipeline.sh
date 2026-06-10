#!/usr/bin/env bash
# =============================================================================
# run_pipeline.sh — 方案B(炸开硬件)端到端编排,setsid 脱离 SSH 自动跑数小时。
#   STAGE0 数据生成 (Pikafish depth12 自对弈, --workers 32, 分块补足)
#   STAGE1 监督预训练 (256×15, --save-every 存细粒度学习曲线 checkpoint)
#   STAGE2 RL 自对弈   (Gumbel, learner GPU0 + selfplay GPU1-6, setsid 后台)
#   STAGE3 评估        (独占 GPU7: eval_vs_ref n_sim800 学习曲线)
#   心跳 $HOURS 后优雅停 + finalize(make_curves 出 crossover/learning 图) + DONE
# -----------------------------------------------------------------------------
# 启动(完全脱离终端,断 SSH 不死):
#   cd /mnt/nvme3n1/gameTheory
#   setsid nohup bash scripts/run_pipeline.sh >logs/pipeline.log 2>&1 &
# 监控: tail -f logs/pipeline.log ; cat logs/PIPELINE_STATUS.md ; tail logs/elo_vs_ref.csv
# 停止: touch /mnt/nvme3n1/gameTheory/.pipeline_stop   (或 kill $(cat logs/pipeline.pid))
# =============================================================================
set -u

: "${ROOT:=/mnt/nvme3n1/gameTheory}"
: "${PY:=${ROOT}/.venv/bin/python}"
: "${UV_CACHE_DIR:=${ROOT}/.uvcache}"

# STAGE0 数据(v2 软标签:更强教师 + MultiPV 软策略 + 引擎价值;切随机开局泄漏)
: "${TARGET_SHARDS:=1800}"
: "${GEN_GAMES_PER_CHUNK:=5000}"
: "${GEN_WORKERS:=180}"            # 榨满 192 线程(实测 64/128/190 均稳,留余核给主进程/OS)
: "${GEN_DEPTH:=14}"               # 时间不限:depth14 抬高 teacher 天花板
: "${GEN_MAX_CHUNKS:=20}"
: "${GEN_MULTIPV:=8}"             # >1 启用软标签(softmax(cp/temp)策略 + tanh(cp/scale)价值)
: "${GEN_POLICY_TEMP:=100}"       # 软策略温度(cp)
: "${GEN_VALUE_SCALE:=500}"       # 价值 squash 尺度(cp)
# ⚠ 软标签 shard(pi 稠密)与旧 one-hot shard(pi_index)schema 不同。正式跑 v2
# 请把 DATA 指向干净的新目录(如 data/processed_v2),勿与旧 1088 片混放。

# STAGE1 预训练 256×15
: "${PRETRAIN_EPOCHS:=10}"
: "${PRETRAIN_BATCH:=2048}"
: "${PRETRAIN_CHANNELS:=256}"
: "${PRETRAIN_BLOCKS:=15}"
: "${PRETRAIN_SAVE_EVERY:=1000}"
: "${PRETRAIN_GPUS:=8}"            # 榨满 8 卡:走 torchrun DDP(见 STAGE1)

# STAGE2 RL(planner 由 config 决定=gumbel)
: "${RL_INIT:=${ROOT}/checkpoints/pretrained_best.pt}"
: "${RL_LEARNER_GPU:=0}"
: "${RL_SELFPLAY_GPUS:=1,2,3,4,5,6}"
: "${RL_WORKERS_PER_GPU:=5}"
: "${RL_PARALLEL_GAMES:=256}"
: "${RL_N_SIM:=32}"
: "${RL_BATCH:=4096}"
: "${RL_EXPORT_EVERY:=800}"

# STAGE3 评估(独占 GPU7)
: "${EVAL_GPU:=7}"
: "${EVAL_INTERVAL:=180}"
: "${EVAL_GAMES:=20}"
: "${EVAL_N_SIM:=100}"     # 训练中评估:适中n_sim,频繁出点(高n_sim一轮要几十分钟,跟不上)
: "${FINAL_N_SIM:=400}"    # finalize 出图用更高搜索质量

# 门控(冠军晋升,斩断退化反馈):latest.pt(候选) vs best.pt(冠军),胜率>=阈值才晋升
: "${GATE_GPU:=7}"         # 与 eval 复用 GPU7(评估与门控都是轻量 net-vs-net,错开跑)
: "${GATE_GAMES:=100}"     # 默认取 config train.gating_games
: "${GATE_WINRATE:=0.55}"  # 默认取 config train.gating_winrate
: "${GATE_N_SIM:=100}"
: "${GATE_INTERVAL:=30}"

# finalize crossover
: "${XOVER_NSIMS:=200,800}"
: "${XOVER_AB_MS:=50,200,500,1000}"

: "${HOURS:=12}"
: "${HEARTBEAT_SECS:=180}"
: "${STOP_FILE:=${ROOT}/.pipeline_stop}"

LOGS="${ROOT}/logs"; CKPT="${ROOT}/checkpoints"; DATA="${DATA:-${ROOT}/data/processed}"
STATUS="${LOGS}/PIPELINE_STATUS.md"
GEN="${ROOT}/scripts/gen_pikafish_data.py"
PRETRAIN="${ROOT}/scripts/pretrain.py"
TRAIN="${ROOT}/scripts/train_distributed.py"
EVAL="${ROOT}/scripts/eval_vs_ref.py"
GATE="${ROOT}/scripts/gate.py"
CURVES="${ROOT}/scripts/make_curves.py"

export UV_CACHE_DIR PYTHONUNBUFFERED=1
cd "${ROOT}" || { echo "cannot cd ${ROOT}"; exit 1; }
mkdir -p "${LOGS}" "${CKPT}" "${DATA}" "${ROOT}/data/raw/pikafish_selfplay"
echo $$ > "${LOGS}/pipeline.pid"
now() { date '+%Y-%m-%d %H:%M:%S'; }
shards() { ls "${DATA}"/shard_*.npz 2>/dev/null | wc -l; }

st_init() {
  { echo "# xqai PIPELINE STATUS (方案B max-scale)"; echo
    echo "- orchestrator PID: $$"; echo "- started: $(now)"; echo "- HOURS: ${HOURS}"
    echo "- net: ${PRETRAIN_CHANNELS}x${PRETRAIN_BLOCKS}  planner: gumbel(config)  eval_n_sim: ${EVAL_N_SIM}"
    echo "- STOP: touch ${STOP_FILE}"; echo
    echo "| stage | time | key | result |"; echo "|---|---|---|---|"; } > "${STATUS}"
}
st() { echo "| $1 | $(now) | $2 | $3 |" >> "${STATUS}"; }

finalize() {
  echo "[pipeline] finalize $(now)"
  [ -f "${LOGS}/rl.pid" ]   && kill "$(cat ${LOGS}/rl.pid)"   2>/dev/null
  [ -f "${LOGS}/gate.pid" ] && kill "$(cat ${LOGS}/gate.pid)" 2>/dev/null
  [ -f "${LOGS}/eval.pid" ] && kill "$(cat ${LOGS}/eval.pid)" 2>/dev/null
  sleep 8
  bash "${ROOT}/scripts/safe_kill.sh" train_distributed.py gate.py eval_vs_ref.py 2>/dev/null
  for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | sort -u); do kill -9 "$p" 2>/dev/null; done
  rm -f /dev/shm/*xqai* 2>/dev/null
  # 最终主力 = 门控冠军 best.pt(单调晋升);无则退回 latest.pt 再退回 pretrained。
  if [ -f "${CKPT}/best.pt" ]; then cp "${CKPT}/best.pt" "${CKPT}/final_best.pt"
  elif [ -f "${CKPT}/latest.pt" ]; then cp "${CKPT}/latest.pt" "${CKPT}/final_best.pt"; fi
  BEST="${CKPT}/final_best.pt"; [ -f "${BEST}" ] || BEST="${CKPT}/pretrained_best.pt"
  CUDA_VISIBLE_DEVICES=${EVAL_GPU} ${PY} ${CURVES} --mode crossover --ckpt "${BEST}" \
     --nsims "${XOVER_NSIMS}" --ab-ms "${XOVER_AB_MS}" --games "${EVAL_GAMES}" \
     --out-prefix "${LOGS}/curve" >> "${LOGS}/curves.log" 2>&1 || echo "[pipeline] crossover 出图失败"
  CUDA_VISIBLE_DEVICES=${EVAL_GPU} ${PY} ${CURVES} --mode learning --ckpts "${CKPT}/pstep_*.pt" \
     --ref random --n-sim ${FINAL_N_SIM} --games "${EVAL_GAMES}" \
     --out-prefix "${LOGS}/curve" >> "${LOGS}/curves.log" 2>&1 || echo "[pipeline] learning 出图失败"
  st "DONE" "final_best.pt + 曲线图" "PASS"
  echo "[pipeline] DONE $(now)"
}
trap finalize EXIT

st_init

# ---- STAGE0 数据 ---------------------------------------------------------- #
echo "[pipeline] STAGE0 数据 target=${TARGET_SHARDS} depth=${GEN_DEPTH} (现有 $(shards))"
rounds=0
while [ "$(shards)" -lt "${TARGET_SHARDS}" ] && [ "${rounds}" -lt "${GEN_MAX_CHUNKS}" ]; do
  [ -f "${STOP_FILE}" ] && break
  echo "[pipeline] STAGE0 chunk $((rounds+1)): ${GEN_GAMES_PER_CHUNK} 局 depth ${GEN_DEPTH}"
  ${PY} ${GEN} --games "${GEN_GAMES_PER_CHUNK}" --workers "${GEN_WORKERS}" --depth "${GEN_DEPTH}" \
     --multipv "${GEN_MULTIPV}" --policy-temp "${GEN_POLICY_TEMP}" --value-scale "${GEN_VALUE_SCALE}" \
     --out "${DATA}" --raw-out "${ROOT}/data/raw/pikafish_selfplay" >> "${LOGS}/stage0_gen.log" 2>&1
  rounds=$((rounds+1))
done
if [ "$(shards)" -lt 1 ]; then st "STAGE0" "0 分片" "FAIL"; echo "[pipeline] STAGE0 FAIL"; exit 1; fi
st "STAGE0" "$(shards) 分片" "PASS"

# ---- STAGE1 预训练 256×15 ------------------------------------------------- #
echo "[pipeline] STAGE1 预训练 ${PRETRAIN_CHANNELS}x${PRETRAIN_BLOCKS} epochs=${PRETRAIN_EPOCHS} save-every=${PRETRAIN_SAVE_EVERY} gpus=${PRETRAIN_GPUS}"
# 榨满多卡:>1 走 torchrun DDP(每卡一进程),否则单进程单卡。
if [ "${PRETRAIN_GPUS}" -gt 1 ]; then
  CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((PRETRAIN_GPUS-1))) \
  "${ROOT}/.venv/bin/torchrun" --standalone --nproc_per_node="${PRETRAIN_GPUS}" \
     ${PRETRAIN} --data "${DATA}" --epochs "${PRETRAIN_EPOCHS}" --batch "${PRETRAIN_BATCH}" \
     --channels "${PRETRAIN_CHANNELS}" --blocks "${PRETRAIN_BLOCKS}" --gpus "${PRETRAIN_GPUS}" \
     --save-every "${PRETRAIN_SAVE_EVERY}" --out "${CKPT}" >> "${LOGS}/stage1_pretrain.log" 2>&1
else
  ${PY} ${PRETRAIN} --data "${DATA}" --epochs "${PRETRAIN_EPOCHS}" --batch "${PRETRAIN_BATCH}" \
     --channels "${PRETRAIN_CHANNELS}" --blocks "${PRETRAIN_BLOCKS}" --gpus "${PRETRAIN_GPUS}" \
     --save-every "${PRETRAIN_SAVE_EVERY}" --out "${CKPT}" >> "${LOGS}/stage1_pretrain.log" 2>&1
fi
if [ ! -f "${CKPT}/pretrained_best.pt" ]; then st "STAGE1" "无 pretrained_best" "FAIL"; echo "[pipeline] STAGE1 FAIL"; exit 1; fi
cp "${CKPT}/pretrained_best.pt" "${CKPT}/ref_coldstart.pt"
st "STAGE1" "pretrained_best.pt(${PRETRAIN_CHANNELS}x${PRETRAIN_BLOCKS})" "PASS"

# ---- STAGE2 RL(Gumbel) ---------------------------------------------------- #
echo "[pipeline] STAGE2 RL gumbel learner=${RL_LEARNER_GPU} selfplay=${RL_SELFPLAY_GPUS} workers/gpu=${RL_WORKERS_PER_GPU}"
# 全新一轮:清掉旧候选/冠军,用冷启动权重 seed 冠军 best.pt(self-play worker 读它,
# 一开始就有冠军可读)。learner 启动也会兜底 seed,但这里显式做更直观。
rm -f /dev/shm/*xqai* "${CKPT}/latest.pt" "${CKPT}/best.pt" 2>/dev/null
cp "${RL_INIT}" "${CKPT}/best.pt"
setsid ${PY} ${TRAIN} --init "${RL_INIT}" --learner-gpu "${RL_LEARNER_GPU}" \
   --selfplay-gpus "${RL_SELFPLAY_GPUS}" --workers-per-gpu "${RL_WORKERS_PER_GPU}" \
   --parallel-games "${RL_PARALLEL_GAMES}" --n-sim "${RL_N_SIM}" --batch "${RL_BATCH}" \
   --export-every "${RL_EXPORT_EVERY}" >> "${LOGS}/stage2_rl.log" 2>&1 &
echo $! > "${LOGS}/rl.pid"
sleep 8
st "STAGE2" "RL PID $(cat ${LOGS}/rl.pid)" "PASS"

# ---- STAGE2b 门控(latest.pt 候选 -> best.pt 冠军) ------------------------- #
echo "[pipeline] STAGE2b gate on GPU${GATE_GPU} games=${GATE_GAMES} winrate>=${GATE_WINRATE} n_sim=${GATE_N_SIM}"
rm -f "${LOGS}/elo_ladder.csv" 2>/dev/null
GATE_DUR=$(awk "BEGIN{print ${HOURS}*3600}")
CUDA_VISIBLE_DEVICES=${GATE_GPU} setsid ${PY} ${GATE} \
   --candidate "${CKPT}/latest.pt" --champion "${CKPT}/best.pt" \
   --games "${GATE_GAMES}" --winrate "${GATE_WINRATE}" --n-sim "${GATE_N_SIM}" \
   --interval "${GATE_INTERVAL}" --duration "${GATE_DUR}" --stop-file "${STOP_FILE}" \
   --ladder "${LOGS}/elo_ladder.csv" >> "${LOGS}/stage2b_gate.log" 2>&1 &
echo $! > "${LOGS}/gate.pid"
st "STAGE2b" "gate PID $(cat ${LOGS}/gate.pid)" "PASS"

# ---- STAGE3 评估(独占 GPU7) ---------------------------------------------- #
# 学习曲线追踪 best.pt(冠军):门控保证冠军单调不退,曲线反映真实进展(原 latest.pt 含退化候选会抖)。
echo "[pipeline] STAGE3 eval_vs_ref(best.pt) on GPU${EVAL_GPU} n_sim=${EVAL_N_SIM}"
rm -f "${LOGS}/elo_vs_ref.csv" 2>/dev/null
EVAL_DUR=$(awk "BEGIN{print ${HOURS}*3600}")
CUDA_VISIBLE_DEVICES=${EVAL_GPU} setsid ${PY} ${EVAL} --interval "${EVAL_INTERVAL}" \
   --games "${EVAL_GAMES}" --n-sim "${EVAL_N_SIM}" --ref "${CKPT}/ref_coldstart.pt" \
   --ckpt "${CKPT}/best.pt" --duration "${EVAL_DUR}" >> "${LOGS}/stage3_eval.log" 2>&1 &
echo $! > "${LOGS}/eval.pid"
st "STAGE3" "eval PID $(cat ${LOGS}/eval.pid)" "PASS"

# ---- 心跳 ----------------------------------------------------------------- #
END=$(awk "BEGIN{print int($(date +%s) + ${HOURS}*3600)}")
while :; do
  sleep "${HEARTBEAT_SECS}"
  [ -f "${STOP_FILE}" ] && { echo "[pipeline] STOP 文件,收尾"; break; }
  kill -0 "$(cat ${LOGS}/rl.pid 2>/dev/null)" 2>/dev/null || { echo "[pipeline] RL 退出,收尾"; break; }
  rl=$(grep "\[learner\] step=" "${LOGS}/stage2_rl.log" 2>/dev/null | tail -1)
  elo=$(tail -1 "${LOGS}/elo_vs_ref.csv" 2>/dev/null)
  gate=$(tail -1 "${LOGS}/elo_ladder.csv" 2>/dev/null)
  echo "[hb $(now)] ${rl} | elo_vs_ref: ${elo} | ladder: ${gate}" >> "${STATUS}"
  [ "$(date +%s)" -ge "${END}" ] && { echo "[pipeline] 到 HOURS,收尾"; break; }
done
# finalize 由 trap EXIT 执行
