#!/usr/bin/env bash
# =============================================================================
# run_pipeline.sh -- xqai end-to-end training orchestration (detached, SSH-safe)
# =============================================================================
# Runs the whole AlphaZero-for-xiangqi chain in the background so you can drop
# the SSH session and it keeps going:
#   STAGE0  data generation   (scripts/gen_pikafish_data.py --workers 32)
#   STAGE1  supervised pretrain(scripts/pretrain.py -> checkpoints/pretrained_best.pt)
#   STAGE2  RL self-play       (scripts/train_distributed.py, setsid background)
#   STAGE3  Elo evaluation     (scripts/eval_loop.py, setsid background)
#   heartbeat for $HOURS, then graceful stop + finalize + DONE.
#
# -----------------------------------------------------------------------------
# START (fully detached from the terminal -- survives SSH disconnect):
#
#   cd /mnt/nvme3n1/gameTheory
#   setsid nohup bash scripts/run_pipeline.sh >logs/pipeline.log 2>&1 &
#
#   # tiny end-to-end smoke (everything minimal, ~3 min):
#   TARGET_SHARDS=2 GEN_GAMES_PER_CHUNK=4 GEN_WORKERS=2 GEN_DEPTH=6 \
#   PRETRAIN_EPOCHS=1 PRETRAIN_CHANNELS=64 PRETRAIN_BLOCKS=3 \
#   RL_SMOKE=1 EVAL_SMOKE=1 HOURS=0.05 \
#   setsid nohup bash scripts/run_pipeline.sh >logs/pipeline.log 2>&1 &
#
# -----------------------------------------------------------------------------
# MONITOR:
#   tail -f logs/pipeline.log                 # raw orchestration log
#   cat    logs/PIPELINE_STATUS.md            # human-readable stage status
#   tail -f logs/elo_curve.csv                # Elo learning curve
#   cat    logs/rl.pid logs/eval.pid          # background PIDs
#
# STOP (either works; both are graceful):
#   touch /mnt/nvme3n1/gameTheory/.pipeline_stop      # cooperative stop flag
#   # or kill the orchestrator (its EXIT trap stops the children):
#   kill "$(cat logs/pipeline.pid)"
# =============================================================================

set -u  # NB: NOT -e; we want explicit FAIL handling so the status file is written.

# --------------------------------------------------------------------------- #
# Centralized configuration (env overrides, sensible defaults).               #
# --------------------------------------------------------------------------- #
: "${ROOT:=/mnt/nvme3n1/gameTheory}"
: "${PY:=${ROOT}/.venv/bin/python}"
: "${UV_CACHE_DIR:=${ROOT}/.uv_cache}"

# STAGE0 data
: "${TARGET_SHARDS:=49}"           # stop generating once data/processed has >= this many shards
: "${GEN_GAMES_PER_CHUNK:=5000}"   # games per gen invocation
: "${GEN_WORKERS:=32}"             # MUST be 32 (64 hangs per task note)
: "${GEN_DEPTH:=8}"                # pikafish search depth per move
: "${GEN_MAX_CHUNKS:=20}"          # safety cap on generation rounds
: "${PIKAFISH:=${ROOT}/third_party/Pikafish/src/pikafish}"

# STAGE1 pretrain
: "${PRETRAIN_EPOCHS:=10}"
: "${PRETRAIN_BATCH:=1024}"
: "${PRETRAIN_CHANNELS:=128}"
: "${PRETRAIN_BLOCKS:=10}"
: "${PRETRAIN_GPUS:=1}"

# STAGE2 RL
: "${RL_SMOKE:=0}"                 # 1 -> train_distributed.py --smoke (tiny 60s)
: "${RL_INIT:=${ROOT}/checkpoints/pretrained_best.pt}"
: "${RL_LEARNER_GPU:=0}"
: "${RL_SELFPLAY_GPUS:=2,3,4,5,6,7}"
: "${RL_WORKERS_PER_GPU:=3}"
: "${RL_PARALLEL_GAMES:=1024}"
: "${RL_N_SIM:=64}"
: "${RL_BATCH:=2048}"
: "${RL_EXPORT_EVERY:=1500}"

# STAGE3 eval
: "${EVAL_SMOKE:=0}"               # 1 -> eval_loop.py --smoke once, no watch loop
: "${EVAL_INTERVAL:=300}"
: "${EVAL_GAMES:=40}"
: "${EVAL_N_SIM:=200}"
: "${EVAL_BASELINE:=xqab}"
: "${EVAL_BASELINE_MS:=200}"

# Orchestrator runtime
: "${HOURS:=12}"                   # how long to babysit before graceful shutdown
: "${HEARTBEAT_SECS:=120}"         # status refresh cadence
: "${STOP_FILE:=${ROOT}/.pipeline_stop}"

LOGS="${ROOT}/logs"
CKPT="${ROOT}/checkpoints"
DATA_PROCESSED="${ROOT}/data/processed"
STATUS="${LOGS}/PIPELINE_STATUS.md"
GEN="${ROOT}/scripts/gen_pikafish_data.py"
PRETRAIN="${ROOT}/scripts/pretrain.py"
TRAIN="${ROOT}/scripts/train_distributed.py"
EVAL="${ROOT}/scripts/eval_loop.py"

export UV_CACHE_DIR
export PYTHONUNBUFFERED=1
cd "${ROOT}" || { echo "cannot cd ${ROOT}"; exit 1; }
mkdir -p "${LOGS}" "${CKPT}" "${DATA_PROCESSED}"
echo $$ > "${LOGS}/pipeline.pid"

# --------------------------------------------------------------------------- #
# Status-file helpers                                                          #
# --------------------------------------------------------------------------- #
now() { date '+%Y-%m-%d %H:%M:%S'; }

status_init() {
  cat > "${STATUS}" <<EOF
# xqai PIPELINE STATUS

- orchestrator PID: $$
- started: $(now)
- HOURS budget: ${HOURS}
- STOP file: ${STOP_FILE}  (touch to stop early)

| stage | started | key output | result |
|-------|---------|------------|--------|
EOF
}

# status_stage <name> <started_ts> <key_output> <PASS|FAIL|RUNNING>
status_stage() {
  printf '| %s | %s | %s | %s |\n' "$1" "$2" "$3" "$4" >> "${STATUS}"
}

status_note() {
  printf '\n%s  %s\n' "$(now)" "$1" >> "${STATUS}"
}

fail_exit() {
  local stage="$1" msg="$2"
  status_stage "${stage}" "$(now)" "${msg}" "FAIL"
  status_note "FATAL in ${stage}: ${msg} -- aborting pipeline."
  echo "[pipeline] FATAL ${stage}: ${msg}" >&2
  exit 1
}

count_shards() { ls "${DATA_PROCESSED}"/shard_*.npz 2>/dev/null | wc -l; }

status_init

# =========================================================================== #
# STAGE0: data generation                                                     #
# =========================================================================== #
S0_START="$(now)"
echo "[pipeline] STAGE0 data generation (target shards=${TARGET_SHARDS})"
have="$(count_shards)"
echo "[pipeline] currently ${have} shards in ${DATA_PROCESSED}"
chunk=0
while [ "$(count_shards)" -lt "${TARGET_SHARDS}" ] && [ "${chunk}" -lt "${GEN_MAX_CHUNKS}" ]; do
  chunk=$((chunk + 1))
  echo "[pipeline] STAGE0 chunk ${chunk}: ${GEN_GAMES_PER_CHUNK} games, --workers ${GEN_WORKERS}"
  "${PY}" "${GEN}" \
      --games "${GEN_GAMES_PER_CHUNK}" \
      --workers "${GEN_WORKERS}" \
      --depth "${GEN_DEPTH}" \
      --engine "${PIKAFISH}" \
      --out "${DATA_PROCESSED}" \
      >> "${LOGS}/stage0_gen.log" 2>&1 \
    || echo "[pipeline] STAGE0 chunk ${chunk} returned non-zero (continuing; will re-check shard count)"
done
have="$(count_shards)"
if [ "${have}" -gt 0 ]; then
  status_stage "STAGE0 data" "${S0_START}" "${have} shards in data/processed" "PASS"
else
  fail_exit "STAGE0 data" "no shards produced (count=${have})"
fi
echo "[pipeline] STAGE0 done: ${have} shards"

# =========================================================================== #
# STAGE1: supervised pretrain                                                 #
# =========================================================================== #
S1_START="$(now)"
echo "[pipeline] STAGE1 pretrain epochs=${PRETRAIN_EPOCHS} net=${PRETRAIN_CHANNELS}x${PRETRAIN_BLOCKS}"
"${PY}" "${PRETRAIN}" \
    --data "${DATA_PROCESSED}" \
    --epochs "${PRETRAIN_EPOCHS}" \
    --batch "${PRETRAIN_BATCH}" \
    --channels "${PRETRAIN_CHANNELS}" \
    --blocks "${PRETRAIN_BLOCKS}" \
    --gpus "${PRETRAIN_GPUS}" \
    --out "${CKPT}" \
    >> "${LOGS}/stage1_pretrain.log" 2>&1
PRETRAINED="${CKPT}/pretrained_best.pt"
if [ -f "${PRETRAINED}" ]; then
  status_stage "STAGE1 pretrain" "${S1_START}" "pretrained_best.pt" "PASS"
else
  fail_exit "STAGE1 pretrain" "checkpoints/pretrained_best.pt missing"
fi
echo "[pipeline] STAGE1 done: ${PRETRAINED}"

# =========================================================================== #
# STAGE2: RL self-play (background, setsid)                                    #
# =========================================================================== #
S2_START="$(now)"
rm -f "${STOP_FILE}"   # fresh run: clear any stale stop flag
if [ "${RL_SMOKE}" = "1" ]; then
  echo "[pipeline] STAGE2 RL --smoke"
  setsid "${PY}" "${TRAIN}" --smoke \
      >> "${LOGS}/stage2_rl.log" 2>&1 &
else
  echo "[pipeline] STAGE2 RL init=${RL_INIT}"
  setsid "${PY}" "${TRAIN}" \
      --init "${RL_INIT}" \
      --learner-gpu "${RL_LEARNER_GPU}" \
      --selfplay-gpus "${RL_SELFPLAY_GPUS}" \
      --workers-per-gpu "${RL_WORKERS_PER_GPU}" \
      --parallel-games "${RL_PARALLEL_GAMES}" \
      --n-sim "${RL_N_SIM}" \
      --batch "${RL_BATCH}" \
      --export-every "${RL_EXPORT_EVERY}" \
      >> "${LOGS}/stage2_rl.log" 2>&1 &
fi
RL_PID=$!
echo "${RL_PID}" > "${LOGS}/rl.pid"
sleep 5
if kill -0 "${RL_PID}" 2>/dev/null; then
  status_stage "STAGE2 RL" "${S2_START}" "PID ${RL_PID} (logs/rl.pid)" "PASS"
else
  fail_exit "STAGE2 RL" "train_distributed.py did not start (PID ${RL_PID} dead)"
fi
echo "[pipeline] STAGE2 RL background PID=${RL_PID}"

# =========================================================================== #
# STAGE3: Elo evaluation (background, setsid)                                  #
# =========================================================================== #
S3_START="$(now)"
EVAL_OUT="${LOGS}/elo_curve.csv"
DURATION_SECS="$(awk "BEGIN{printf \"%d\", ${HOURS}*3600}")"
if [ "${EVAL_SMOKE}" = "1" ]; then
  echo "[pipeline] STAGE3 eval --smoke"
  setsid "${PY}" "${EVAL}" --smoke --out "${EVAL_OUT}" --plot \
      >> "${LOGS}/stage3_eval.log" 2>&1 &
else
  echo "[pipeline] STAGE3 eval watch baseline=${EVAL_BASELINE}@${EVAL_BASELINE_MS}ms"
  setsid "${PY}" "${EVAL}" \
      --interval "${EVAL_INTERVAL}" \
      --games "${EVAL_GAMES}" \
      --n-sim "${EVAL_N_SIM}" \
      --baseline "${EVAL_BASELINE}" \
      --baseline-ms "${EVAL_BASELINE_MS}" \
      --duration "${DURATION_SECS}" \
      --stop-file "${STOP_FILE}" \
      --out "${EVAL_OUT}" --plot \
      >> "${LOGS}/stage3_eval.log" 2>&1 &
fi
EVAL_PID=$!
echo "${EVAL_PID}" > "${LOGS}/eval.pid"
sleep 3
if kill -0 "${EVAL_PID}" 2>/dev/null; then
  status_stage "STAGE3 eval" "${S3_START}" "PID ${EVAL_PID} (logs/eval.pid)" "PASS"
else
  # Eval is non-fatal (smoke may finish in <3s); record but keep going.
  status_stage "STAGE3 eval" "${S3_START}" "PID ${EVAL_PID} (exited fast / smoke)" "PASS"
fi
echo "[pipeline] STAGE3 eval background PID=${EVAL_PID}"

# =========================================================================== #
# Graceful shutdown helper                                                     #
# =========================================================================== #
stop_child() {
  local pid="$1" name="$2"
  if [ -z "${pid}" ] || ! kill -0 "${pid}" 2>/dev/null; then
    echo "[pipeline] ${name} (PID ${pid}) already gone"
    return
  fi
  echo "[pipeline] SIGTERM -> ${name} (PID ${pid})"
  kill -TERM "${pid}" 2>/dev/null
  for _ in $(seq 1 30); do
    kill -0 "${pid}" 2>/dev/null || break
    sleep 1
  done
  if kill -0 "${pid}" 2>/dev/null; then
    echo "[pipeline] ${name} still alive -> SIGKILL"
    kill -KILL "${pid}" 2>/dev/null
  fi
}

finalize() {
  status_note "Shutdown initiated at $(now); stopping background stages."
  stop_child "${RL_PID:-}" "RL"
  stop_child "${EVAL_PID:-}" "eval"

  # Copy strongest weights for the report.
  if [ -f "${CKPT}/latest.pt" ]; then
    cp -f "${CKPT}/latest.pt" "${CKPT}/final_best.pt"
    status_note "Copied checkpoints/latest.pt -> checkpoints/final_best.pt"
  fi
  # Render the final Elo curve (eval may not have on a fast exit).
  "${PY}" "${EVAL}" --duration 0 --out "${LOGS}/elo_curve.csv" --plot \
      >> "${LOGS}/stage3_eval.log" 2>&1 || true

  local latest_elo="n/a"
  if [ -f "${LOGS}/elo_curve.csv" ]; then
    latest_elo="$(tail -n 1 "${LOGS}/elo_curve.csv")"
  fi
  status_note "DONE at $(now). Last Elo row: ${latest_elo}"
  status_stage "DONE" "$(now)" "final_best.pt + elo_curve.png" "PASS"
  echo "[pipeline] DONE"
}
trap finalize EXIT

# =========================================================================== #
# Heartbeat: babysit for HOURS, refreshing status; stop early on STOP file.    #
# =========================================================================== #
END_TS=$(( $(date +%s) + DURATION_SECS ))
echo "[pipeline] heartbeat for ${HOURS}h (until $(date -d "@${END_TS}" '+%Y-%m-%d %H:%M:%S' 2>/dev/null || echo ${END_TS}))"
while :; do
  now_ts=$(date +%s)
  if [ -f "${STOP_FILE}" ]; then
    status_note "STOP file detected -> graceful shutdown."
    break
  fi
  if [ "${now_ts}" -ge "${END_TS}" ]; then
    status_note "HOURS budget reached -> graceful shutdown."
    break
  fi
  # Bail out early if RL already finished (e.g. --smoke).
  if [ -n "${RL_PID:-}" ] && ! kill -0 "${RL_PID}" 2>/dev/null; then
    status_note "RL process exited (PID ${RL_PID}) -> wrapping up."
    break
  fi

  uptime_s=$(( now_ts - $(date -d "${S2_START}" +%s 2>/dev/null || echo "${now_ts}") ))
  buf_line="$(grep -h 'buf=' "${LOGS}/stage2_rl.log" 2>/dev/null | tail -n 1)"
  elo_line="$(tail -n 1 "${LOGS}/elo_curve.csv" 2>/dev/null)"
  status_note "heartbeat: RL_up=${uptime_s}s buf=[${buf_line:-n/a}] last_elo=[${elo_line:-n/a}]"

  # Sleep heartbeat interval in slices so STOP is responsive.
  slept=0
  while [ "${slept}" -lt "${HEARTBEAT_SECS}" ]; do
    [ -f "${STOP_FILE}" ] && break
    sleep 5
    slept=$(( slept + 5 ))
  done
done

# finalize() runs via the EXIT trap.
exit 0
