#!/usr/bin/env bash
# value-safe RL 消融:检验"冻结 value 头 / 监督锚定"能否消除 warm-start 自对弈退化。
# 与 v2 同源对照(同冷启动 init、同冻结冠军自对弈、同 Gumbel n_sim=32),唯一变量=learner 更新规则。
#   Arm frzV : 冻 value 头(隔离 value 漂移机理)
#   Arm safe : 冻 value 头 + 掺 40% 监督(Pikafish 软标签 = value+policy 锚)
# 基线 = 已有 v2(退化:门控全 REJECT、vs 冷启动 winrate 0.27-0.5)。
set -u
cd /mnt/nvme3n1/gameTheory
source .venv/bin/activate

CFG=configs/v2_rl.yaml
INIT=checkpoints_v2_rl/best.pt   # v2 冷启动 step0(=监督主力),与评估参考同源
REF=checkpoints_v2_rl/best.pt
STEPS=10000

launch_arm() { # name  learner_gpu  selfplay_gpus  extra_flags...
  local name="$1" lgpu="$2" sgpus="$3"; shift 3
  local cdir="checkpoints_vsafe_${name}"
  rm -rf "$cdir"; mkdir -p "$cdir"
  rm -f "logs/vsafe_${name}_vs_cold.csv"   # eval_vs_ref appends; stale rows would splice runs
  echo "[launch] arm=$name learner=cuda:$lgpu selfplay=$sgpus ckpt=$cdir flags: $*"
  nohup python scripts/train_value_safe.py --config "$CFG" --init "$INIT" \
      --steps "$STEPS" --learner-gpu "$lgpu" --selfplay-gpus "$sgpus" \
      --workers-per-gpu 4 --ckpt-dir "$cdir" --replay-suffix "$name" "$@" \
      > "logs/vsafe_${name}.log" 2>&1 &
  echo "  train pid=$!"
  sleep 8  # 让 learner 先建 buffer + 导出初始 latest.pt
  nohup python scripts/eval_vs_ref.py --ref "$REF" --ckpt "$cdir/latest.pt" \
      --device "cuda:$lgpu" --interval 90 --games 20 --n-sim 80 --duration 36000 \
      --out "logs/vsafe_${name}_vs_cold.csv" \
      > "logs/vsafe_${name}_eval.log" 2>&1 &
  echo "  eval  pid=$!"
}

# learner 放空卡(0/7);self-play 与锚点共卡(40GB 富余,锚点仅占 ~850MB)。
launch_arm frzV 0 1,2,3 --freeze-value
launch_arm safe 7 4,5,6 --freeze-value --rehearse-dir data/processed_v2 --rehearse-frac 0.4 --rehearse-shards 60

echo "[launch] both arms up. tail logs/vsafe_*.log ; curves in logs/vsafe_*_vs_cold.csv"
