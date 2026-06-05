#!/usr/bin/env bash
# 等基础安装完成 → 换 cu124 torch → 编译 _xqcore → 真·端到端冒烟
set -uo pipefail
cd /mnt/nvme3n1/gameTheory
PY=.venv/bin/python

echo "[finalize] 等待基础安装完成 (ALL_INSTALLS_DONE)..."
for i in $(seq 1 600); do
  grep -q ALL_INSTALLS_DONE logs/setup.log 2>/dev/null && { echo "[finalize] 基础安装完成"; break; }
  sleep 5
done

echo "[finalize] 验证构建工具..."
$PY -c "import cmake,ninja,pybind11;print('build tools', pybind11.__version__)" || { echo "构建工具缺失，退出"; exit 1; }

echo "[finalize] 把 torch 换成 cu124（匹配驱动 550 / CUDA 12.4）..."
uv pip install --python "$PY" --index-url https://download.pytorch.org/whl/cu124 torch

echo "[finalize] 编译 _xqcore 扩展（scikit-build-core + pybind11）..."
uv pip install --python "$PY" --no-build-isolation -e . 2>&1 | tail -30

echo "[finalize] ===== 验证 ====="
$PY -c "import torch;print('torch',torch.__version__,'cuda_avail',torch.cuda.is_available(),'ndev',torch.cuda.device_count())"
$PY -c "from xqai._xqcore import Position; p=Position(); print('legal_moves', len(p.legal_moves())); print('perft(3)=', Position.perft(p.fen(),3), '(应=79666)')"
echo "[finalize] 跑端到端曳光弹冒烟..."
$PY -m xqai._smoke_pipeline 2>&1 | tail -20
echo "FINALIZE_DONE"
