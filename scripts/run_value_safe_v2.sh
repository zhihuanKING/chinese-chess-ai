#!/usr/bin/env bash
# 修复版 value-safe 消融重跑(commit a6cf94f 之后的代码:S1截断假和棋/S2S3冻结
# 失效/replay跨进程锁 均已修复)。
#
# 设计:与旧版同源对照(同 init、同冻结冠军自对弈、同 Gumbel n_sim=32、同 10000 步),
# 2x2 消融矩阵 {冻value, 不冻} x {rehearsal, 无} 的三个臂 + 纯监督对照:
#   frzV     : 冻 value 头
#   safe     : 冻 value 头 + 40% Pikafish rehearsal(冻结下实为 policy 蒸馏锚)
#   rehNoFrz : 40% rehearsal、不冻(value 同时吃自对弈真 z + Pikafish 软 z)
#   suponly  : 纯监督续训 + 冻 value(蒸馏对照,无自对弈)
#
# 资源对齐(复审结论):RL 臂若并行共卡会"隐性少吃数据",故分两批、每个 RL 臂
# 都用完全相同的模板(1 learner 卡 + 3 selfplay 卡 x 4 worker);冻结冠军下数据
# 分布平稳,分批不损可比性。
#   批1: frzV(L=0,SP=1,2,3) + safe(L=7,SP=4,5,6)
#   批2: rehNoFrz(L=0,SP=1,2,3) + suponly(GPU7 单卡)
# 末尾每臂 100 局终点定论(eval_final.py)。
set -u
cd /mnt/nvme3n1/gameTheory
source .venv/bin/activate

CFG=configs/v2_rl.yaml
# 冷启动主力 = 所有臂的 init 与评测参考。best.pt 是"活"文件(任何不带
# --ckpt-dir 的杂进程都可能原子覆盖它),先做只读快照,init/ref 全用快照:
# 否则批2的 init 或末尾 eval_final 重新读盘时参考可能已被换掉。
cp -f checkpoints_v2_rl/best.pt checkpoints_v2_rl/ref_coldstart_frozen.pt
INIT=checkpoints_v2_rl/ref_coldstart_frozen.pt
REF=$INIT
STEPS=10000
MINBUF=100000                    # 复审建议:避免 buffer 刚够 1 个 batch 就起训
EVAL_GAMES=12                    # 12 局/轮 -> 曲线点密度 ~ 每个 export 一点

launch_rl_arm() { # name learner_gpu selfplay_gpus extra_flags...
  local name="$1" lgpu="$2" sgpus="$3"; shift 3
  local cdir="checkpoints_vsafe2_${name}"
  rm -rf "$cdir"; mkdir -p "$cdir"
  rm -f "logs/vsafe2_${name}_vs_cold.csv"   # eval append;旧行会串线
  echo "[launch] arm=$name learner=cuda:$lgpu selfplay=$sgpus flags: $*"
  nohup python scripts/train_value_safe.py --config "$CFG" --init "$INIT" \
      --steps "$STEPS" --learner-gpu "$lgpu" --selfplay-gpus "$sgpus" \
      --workers-per-gpu 4 --min-buffer "$MINBUF" --ckpt-dir "$cdir" \
      --replay-suffix "v2${name}" "$@" \
      > "logs/vsafe2_${name}.log" 2>&1 &
  echo $! > "logs/vsafe2_${name}.pid"
  echo "  train pid=$(cat logs/vsafe2_${name}.pid)"
  sleep 8  # learner 先建 buffer + 导出初始 latest.pt
  nohup python scripts/eval_vs_ref.py --ref "$REF" --ckpt "$cdir/latest.pt" \
      --device "cuda:$lgpu" --interval 60 --games "$EVAL_GAMES" --n-sim 80 \
      --duration 86400 --out "logs/vsafe2_${name}_vs_cold.csv" \
      > "logs/vsafe2_${name}_eval.log" 2>&1 &
  echo $! > "logs/vsafe2_${name}_eval.pid"
  echo "  eval  pid=$(cat logs/vsafe2_${name}_eval.pid)"
}

wait_pid() { while kill -0 "$1" 2>/dev/null; do sleep 30; done; }

check_arm() { # name: 训练退出后必须有 latest.pt,否则是秒崩(--init缺失/参数错),
              # 带着空目录继续跑只会产出无意义 eval —— 立刻终止整个流水线。
  local cdir="checkpoints_vsafe2_$1"
  if [ ! -f "$cdir/latest.pt" ]; then
    echo "[FATAL] arm $1: training exited without $cdir/latest.pt; see logs/vsafe2_$1.log" >&2
    exit 1
  fi
}

# 等 eval 把"最终 export(step==STEPS)"那一点写进曲线 CSV 再杀 eval。
# 固定 sleep 不行:一轮 12 局 n_sim=80 实测可达 20-40min,训练刚结束时 eval
# 多半正卡在旧 ckpt 的一轮中,30min 后强杀必丢终点 —— eval_vs_ref 里
# "FINAL 不再被永久跳过"的修复会在脚本层被抵消。上限 2h 兜底。
wait_final_point() { # csv
  local csv="$1" deadline=$(( $(date +%s) + 7200 ))
  until [ -f "$csv" ] && awk -F, -v s="$STEPS" 'NR>1 && $2==s {f=1} END{exit !f}' "$csv"; do
    if [ "$(date +%s)" -ge "$deadline" ]; then
      echo "  [warn] $csv: final point (step=$STEPS) still missing after 2h; giving up" >&2
      return 1
    fi
    sleep 60
  done
  echo "  final curve point landed in $csv"
}

echo "=== batch 1: frzV + safe ($(date)) ==="
launch_rl_arm frzV 0 1,2,3 --freeze-value
launch_rl_arm safe 7 4,5,6 --freeze-value \
    --rehearse-dir data/processed_v2 --rehearse-frac 0.4 --rehearse-shards 60
wait_pid "$(cat logs/vsafe2_frzV.pid)"
wait_pid "$(cat logs/vsafe2_safe.pid)"
check_arm frzV; check_arm safe
echo "=== batch 1 training done ($(date)); waiting for final eval points ==="
wait_final_point logs/vsafe2_frzV_vs_cold.csv
wait_final_point logs/vsafe2_safe_vs_cold.csv
kill "$(cat logs/vsafe2_frzV_eval.pid)" "$(cat logs/vsafe2_safe_eval.pid)" 2>/dev/null

echo "=== batch 2: rehNoFrz + suponly ($(date)) ==="
launch_rl_arm rehNoFrz 0 1,2,3 \
    --rehearse-dir data/processed_v2 --rehearse-frac 0.4 --rehearse-shards 60
# suponly: 纯监督单卡(GPU7),eval 同卡
rm -rf checkpoints_vsafe2_suponly; mkdir -p checkpoints_vsafe2_suponly
rm -f logs/vsafe2_suponly_vs_cold.csv
nohup python scripts/train_sup_continue.py --init "$INIT" --data data/processed_v2 \
    --shards 60 --steps "$STEPS" --freeze-value --gpu 7 \
    --ckpt-dir checkpoints_vsafe2_suponly > logs/vsafe2_suponly.log 2>&1 &
echo $! > logs/vsafe2_suponly.pid
sleep 8
nohup python scripts/eval_vs_ref.py --ref "$REF" \
    --ckpt checkpoints_vsafe2_suponly/latest.pt \
    --device cuda:7 --interval 60 --games "$EVAL_GAMES" --n-sim 80 \
    --duration 86400 --out logs/vsafe2_suponly_vs_cold.csv \
    > logs/vsafe2_suponly_eval.log 2>&1 &
echo $! > logs/vsafe2_suponly_eval.pid
wait_pid "$(cat logs/vsafe2_rehNoFrz.pid)"
wait_pid "$(cat logs/vsafe2_suponly.pid)"
check_arm rehNoFrz; check_arm suponly
echo "=== batch 2 training done ($(date)); waiting for final eval points ==="
wait_final_point logs/vsafe2_rehNoFrz_vs_cold.csv
wait_final_point logs/vsafe2_suponly_vs_cold.csv
kill "$(cat logs/vsafe2_rehNoFrz_eval.pid)" "$(cat logs/vsafe2_suponly_eval.pid)" 2>/dev/null

echo "=== final 100-game verdict evals ($(date)) ==="
rm -f logs/vsafe2_final_*.csv   # eval_final 是 append;重跑防新旧行混杂
for pair in frzV:0 safe:1 rehNoFrz:2 suponly:3; do
  name="${pair%%:*}"; g="${pair##*:}"
  nohup python scripts/eval_final.py --ref "$REF" \
      --ckpt "checkpoints_vsafe2_${name}/latest.pt" --games 100 --n-sim 80 \
      --device "cuda:$g" --tag "$name" --out "logs/vsafe2_final_${name}.csv" \
      > "logs/vsafe2_final_${name}.log" 2>&1 &
done
wait
echo "=== ALL DONE ($(date)) ==="
echo "curves: logs/vsafe2_*_vs_cold.csv ; verdicts: logs/vsafe2_final_*.csv"
