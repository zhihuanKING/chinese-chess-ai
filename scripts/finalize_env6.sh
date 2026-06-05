#!/usr/bin/env bash
# v6：清华国内镜像装 torch==2.6.0(cu124) —— 直连域内、快。同盘缓存。
set -uo pipefail
cd /mnt/nvme3n1/gameTheory
export UV_CACHE_DIR=/mnt/nvme3n1/.uvcache
IDX=https://pypi.tuna.tsinghua.edu.cn/simple
PY=.venv/bin/python

echo "[v6] 重建干净 venv..."
rm -rf .venv
uv venv --python 3.11 .venv 2>&1 | tail -1

echo "[v6] 清华源安装 torch==2.6.0 + 依赖..."
uv pip install --python "$PY" --index-url "$IDX" \
  "torch==2.6.0" numpy pyyaml tqdm tensorboard matplotlib pandas cmake ninja pybind11 scikit-build-core 2>&1 | tail -8

echo "[v6] torch 验证:"
$PY -c "import torch;print('torch',torch.__version__,'cuda',torch.version.cuda,'avail',torch.cuda.is_available(),'ndev',torch.cuda.device_count())"

echo "[v6] 编译 _xqcore..."
rm -rf cpp/build
uv pip install --python "$PY" --no-build-isolation -e . 2>&1 | tail -8

echo "[v6] 集成验证:"
$PY -c "from xqai._xqcore import Position; p=Position(); print('perft3',Position.perft(p.fen(),3),'perft4',Position.perft(p.fen(),4))"
echo "[v6] 端到端冒烟:"
$PY -m xqai._smoke_pipeline 2>&1 | tail -20
echo V6_DONE
