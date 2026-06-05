#!/usr/bin/env bash
# 修复版：强制 cu124 torch + 重建 _xqcore(已修 PIC) + 真·端到端冒烟
set -uo pipefail
cd /mnt/nvme3n1/gameTheory
PY=.venv/bin/python
export UV_LINK_MODE=copy

echo "[v2] 卸载 cu130 torch..."
uv pip uninstall --python "$PY" torch 2>&1 | tail -2 || true

echo "[v2] 从 cu124 源安装 torch==2.6.0（匹配驱动550/CUDA12.4），依赖回退 pypi..."
uv pip install --python "$PY" \
  --index-url https://download.pytorch.org/whl/cu124 \
  --extra-index-url https://pypi.org/simple \
  "torch==2.6.0" 2>&1 | tail -8

echo "[v2] torch 验证..."
$PY -c "import torch;print('torch',torch.__version__,'cuda_avail',torch.cuda.is_available(),'ndev',torch.cuda.device_count(),'dev0',torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NA')"

echo "[v2] 重建 _xqcore（PIC 已修）..."
rm -rf cpp/build
uv pip install --python "$PY" --no-build-isolation -e . 2>&1 | tail -15

echo "[v2] ===== 集成验证 ====="
$PY -c "from xqai._xqcore import Position; p=Position(); print('legal_moves',len(p.legal_moves())); print('perft(3)=',Position.perft(p.fen(),3),'(应=79666)'); print('perft(4)=',Position.perft(p.fen(),4),'(应=3290240)')"
echo "[v2] 端到端曳光弹冒烟（真 _xqcore + PUCT 自对弈 + 训练一步）..."
$PY -m xqai._smoke_pipeline 2>&1 | tail -25
echo "V2_DONE"
