#!/usr/bin/env bash
# v5：干净安装（无自杀式 pkill）。正常网络 + 同盘缓存(hardlink快) + cu124 torch
set -uo pipefail
cd /mnt/nvme3n1/gameTheory
export UV_CACHE_DIR=/mnt/nvme3n1/.uvcache
PY=.venv/bin/python

echo "[v5] 重建干净 venv..."
rm -rf .venv
uv venv --python 3.11 .venv 2>&1 | tail -1

echo "[v5] 安装 torch==2.6.0(cu124) + 依赖..."
uv pip install --python "$PY" "torch==2.6.0" numpy pyyaml tqdm tensorboard matplotlib pandas cmake ninja pybind11 scikit-build-core 2>&1 | tail -6

echo "[v5] torch 验证:"
$PY -c "import torch;print('torch',torch.__version__,'cuda',torch.version.cuda,'avail',torch.cuda.is_available(),'ndev',torch.cuda.device_count())"

echo "[v5] 编译 _xqcore..."
rm -rf cpp/build
uv pip install --python "$PY" --no-build-isolation -e . 2>&1 | tail -8

echo "[v5] 集成验证:"
$PY -c "from xqai._xqcore import Position; p=Position(); print('perft3',Position.perft(p.fen(),3),'perft4',Position.perft(p.fen(),4))"
echo "[v5] 端到端冒烟:"
$PY -m xqai._smoke_pipeline 2>&1 | tail -20
echo V5_DONE
