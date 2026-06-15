#!/usr/bin/env bash
# B 循环的 GPU7 边车:周期锚点 ladder + DAgger 飞轮采集,直至 .pipeline_stop。
# ladder: 每 2h 对当前冠军跑 AB200+pf1 各 64 局(向量化),追加 logs/ladder/b_ladder.csv
# 飞轮:  每轮重启 gen_positions 读最新冠军,持续给标注服务供料(每轮 2000 局)
set -u
cd /mnt/nvme3n1/gameTheory
source .venv/bin/activate
# 限 torch intra-op 线程,防止评测/采集抢满 192 核饿死自对弈 worker
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
CHAMP=checkpoints_b/best.pt
LCSV=logs/ladder/b_ladder.csv
mkdir -p logs/ladder
[ -f "$LCSV" ] || echo "wall,step,anchor,games,winrate,elo" > "$LCSV"

ladder_once() {
  local step; step=$(.venv/bin/python -c "import torch;print(torch.load('$CHAMP',map_location='cpu',weights_only=False).get('step','?'))" 2>/dev/null || echo "?")
  for opp in xqab:200 pikafish:1; do
    local tag="${opp/:/}"
    .venv/bin/python scripts/eval_anchor_vec.py --ckpt "$CHAMP" --opponent "$opp" \
        --gpu 7 --n-sim 200 --games 64 --parallel 12 --opening-seed 12345 \
        --tag "lad_${tag}" --out "/tmp/b_ladder_${tag}.csv" > /dev/null 2>&1
    local row; row=$(tail -n 1 "/tmp/b_ladder_${tag}.csv")
    local wr elo
    wr=$(echo "$row" | cut -d, -f5); elo=$(echo "$row" | cut -d, -f8)
    echo "$(date +%s),$step,$opp,64,$wr,$elo" >> "$LCSV"
    rm -f "/tmp/b_ladder_${tag}.csv"
  done
  echo "[ladder] step=$step done $(date)"
}

last_ladder=0
while [ ! -f .pipeline_stop ]; do
  # 飞轮:一轮 2000 局采集(读最新冠军;completion 或 stop 后重启拿新权重)
  .venv/bin/python scripts/gen_positions.py --ckpt "$CHAMP" --gpu 7 \
      --n-sim 64 --parallel-games 128 --games 2000 --explore-frac 0.4 \
      --temp-moves 60 --out-dir data/dagger/queue_p1 >> logs/b_flywheel.log 2>&1 &
  FW=$!
  # 等飞轮轮次结束或到达 ladder 周期
  while kill -0 "$FW" 2>/dev/null; do
    sleep 60
    if [ -f .pipeline_stop ]; then kill "$FW" 2>/dev/null; break; fi
    if [ $(( $(date +%s) - last_ladder )) -ge 7200 ]; then
      kill "$FW" 2>/dev/null; wait "$FW" 2>/dev/null
      ladder_once
      last_ladder=$(date +%s)
    fi
  done
done
echo "[sidecar] stopped $(date)"
