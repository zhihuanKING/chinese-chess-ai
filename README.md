# 中国象棋博弈 AI（Xiangqi AI）

> 课程设计 · 计算博弈棋牌类 AI。主线：**让深度神经网络通过自对弈从零学会中国象棋**——监督冷启动 + **Gumbel-AlphaZero** 低模拟自对弈在一周内破局，并用实验证明胜过传统 Alpha-Beta 引擎。全程 AI 互搏，不涉及人机对弈。

## 项目简介

本项目实现一套完整的中国象棋 AI 训练与评测流水线：

- **C++17 规则内核**：高性能 movegen / make-unmake / zobrist，`perft` 1–5 命中标准值。
- **Alpha-Beta 基线引擎**：Negamax + α-β 剪枝 + 置换表 + 迭代加深 + 静态搜索，作为对照组。
- **AlphaZero 范式学习引擎**：双头价值-策略网络（PVNet）+ 向量化批量 MCTS（PUCT 先行、**Gumbel-AlphaZero** 后挂同接口）+ 自对弈 + ReplayBuffer + 分布式训练。
- **监督冷启动**：用 Pikafish 标注 / 人类棋谱做预训练，加速破局。
- **Arena 评测**：引擎对弈 → PGN/结果 → BayesElo/Ordo 反推 Elo。

核心实验结论用三张图呈现：Elo-vs-训练步（叠加冷启动）、学习型 vs Alpha-Beta 同预算 crossover、Gumbel `n_sim` 消融。

## 目录结构

```
cpp/                 C++17 规则内核 + Alpha-Beta + pybind 绑定 + perft/doctest 单测
  include/xq/        position.h / search.h / types.h / zobrist.h
  src/               position.cpp / search.cpp / eval.cpp / *_main.cpp
  bindings/          module.cpp  -> 编译出扩展模块 xqai._xqcore
  tests/             test_perft.cpp / test_rules.cpp
xqai/                Python 学习引擎
  encoding.py        棋盘/着法张量编码、合法掩码
  network.py         PVNet 双头网络
  mcts.py            PUCTPlanner + GumbelPlanner（统一 search 接口）
  selfplay.py        向量化批量自对弈
  replay.py          ReplayBuffer（/dev/shm 环形缓冲）
  train.py / arena.py / config.py
scripts/             数据准备 / 监督预训练 / 分布式自对弈 / 评测 / 画图
configs/default.yaml 全局超参（网络/MCTS/优化/数据/分布式）
docx/                实验方案、报告框架、上手引导
INTERFACES.md        跨语言接口契约（C++/Python 各模块唯一约定，先读这个）
```

> `data/`、`checkpoints/`、`logs/`、`.venv/`、`third_party/` 等大文件/产物目录已被 `.gitignore` 排除，不入库。模型权重如需分享请走 GitHub Releases 或网盘。

## 环境与安装

硬件实测环境：8× A800-40GB（NVLink）。驱动 550 → CUDA 12.4，**PyTorch 必须用 cu124 wheel**。

```bash
# 1. 用 uv 创建虚拟环境（Python >= 3.10）
uv venv .venv && source .venv/bin/activate

# 2. 安装 PyTorch（cu124，清华镜像，torch==2.6.0 即 cu124）
uv pip install torch==2.6.0 --index-url https://pypi.tuna.tsinghua.edu.cn/simple

# 3. 可编辑安装本项目（自动用 scikit-build-core + CMake/Ninja 编译 C++ 扩展 xqai._xqcore）
uv pip install -e . --index-url https://pypi.tuna.tsinghua.edu.cn/simple
```

依赖：numpy、torch(cu124)、pyyaml、tqdm、tensorboard、matplotlib、pandas；构建依赖 g++11 + CMake/Ninja + pybind11 + scikit-build-core。

## 快速开始

```bash
# C++ 规则内核 perft 校验
./cpp/build/xq_perft           # 或运行 perft_main 检验 1–5 层叶子数

# Python 端冒烟测试
python -m xqai._smoke_encoding
python -m xqai._smoke_network
python -m xqai._smoke_pipeline

# 监督预训练（冷启动）
python scripts/pretrain.py --config configs/default.yaml

# 分布式自对弈训练
python scripts/train_distributed.py --config configs/default.yaml

# Arena 对弈评测（学习引擎 vs Alpha-Beta）
python scripts/eval_loop.py
```

## 关键约定（节选自 `INTERFACES.md`）

- **坐标**：9 列×10 行=90 格，`row=sq//9`、`col=sq%9`；红方在底部、红先行。
- **着法编码**：`move = from_sq*90 + to_sq`，`ACTION_DIM = 8100`（合法掩码置 -inf）。
- **网络**：PVNet 输入 `[B, 15, 10, 9]`，输出 `(policy_logits[B,8100], value[B,1]∈[-1,1])`。
- **规则简化版**（报告需声明）：第 3 次重复判和 + 120 回合自然限着判和。

完整契约见 [`INTERFACES.md`](INTERFACES.md)，实验方案与报告框架见 `docx/`。

## 技术要点

- 算法：AlphaZero 范式，核心 **Gumbel-AlphaZero(ICLR'22)**；PUCT 先跑通学习曲线，Gumbel 同接口后挂。融合 KataGo playout-cap、ReZero reanalyze。
- 学习循环用 **Python 向量化批量 MCTS**（参考 TurboZero/MCTX），C++ 只做规则内核与 Alpha-Beta 基线。
- 网络双轨：128×10 快迭代 / 256×15 最终主力（规模作消融）。

## 许可

课程设计用途。第三方参考实现（Pikafish 等）各自遵循其原始许可，未包含在本仓库中。
