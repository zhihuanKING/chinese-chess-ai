#!/usr/bin/env bash
# v3：从 pypi 直装 torch==2.6.0（默认 wheel=cu124，匹配驱动550），重建 _xqcore，端到端冒烟
set -uo pipefail
cd /mnt/nvme3n1/gameTheory
PY=.venv/bin/python
export UV_LINK_MODE=copy

echo "[v3] 安装 torch==2.6.0（pypi 默认 = cu124）..."
uv pip install --python "$PY" "torch==2.6.0" 2>&1 | tail -8

echo "[v3] torch 验证..."
$PY -c "import torch;print('torch',torch.__version__,'cuda_avail',torch.cuda.is_available(),'ndev',torch.cuda.device_count(),'name0',torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NA')"

echo "[v3] 重建 _xqcore（PIC 已修）..."
rm -rf cpp/build
uv pip install --python "$PY" --no-build-isolation -e . 2>&1 | tail -15

echo "[v3] ===== 集成验证 ====="
$PY -c "from xqai._xqcore import Position; p=Position(); print('legal',len(p.legal_moves())); print('perft3=',Position.perft(p.fen(),3),'(应=79666)'); print('perft4=',Position.perft(p.fen(),4),'(应=3290240)')"
echo "[v3] 端到端曳光弹冒烟..."
$PY -m xqai._smoke_pipeline 2>&1 | tail -25
echo "V3_DONE"
