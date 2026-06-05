#!/usr/bin/env bash
# 在【你本机】（全网可达 github）运行：bash scripts/download_data.sh
# 我的沙箱 shell 下不动 github 主站，所以这步交给你跑。
set -e
ROOT=/mnt/nvme3n1/gameTheory
RAW=$ROOT/data/raw
mkdir -p "$RAW" "$ROOT/third_party"
cd "$ROOT/third_party"

echo "==> 1) Pikafish（强对手 / 锚定 / 可选数据标注）"
git clone --depth 1 https://github.com/official-pikafish/Pikafish.git || echo "Pikafish 已存在/跳过"
# NNUE 权重见 release：https://github.com/official-pikafish/Networks
echo "   提示：到 Pikafish/src 执行 make -j 编译；NNUE .nnue 权重从 Networks release 下载放入 src"

echo "==> 2) 参考实现：编码/棋谱/管线（ChineseChess-AlphaZero）"
git clone --depth 1 https://github.com/NeymarL/ChineseChess-AlphaZero.git || echo "已存在/跳过"
# 该仓库含 2086 动作映射与部分棋谱，可参考

echo "==> 3) 参考实现：MCTS/Gumbel 正确性对拍（LightZero / OpenSpiel）"
git clone --depth 1 https://github.com/opendilab/LightZero.git || echo "已存在/跳过"

echo "==> 4) 人类棋谱（按需补充你的来源，放到 $RAW）"
echo "   例如东萍/PGN-CN 棋谱集；放好后运行 python scripts/prepare_data.py"

echo "全部完成。third_party/ 下为参考代码，data/raw/ 下放原始棋谱。"
