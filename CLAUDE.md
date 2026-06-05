# CLAUDE.md — 项目约定与工作风格（每次会话自动加载，无需重提）

## 项目
中国象棋博弈 AI 课程设计（计算博弈棋牌类 AI，中文报告）。主线：**让深度神经网络通过自对弈从零学会中国象棋（全程 AI 互搏，不涉及人机），用监督冷启动 + Gumbel-AlphaZero 低模拟自对弈在一周内破局，并实验证明胜过传统 Alpha-Beta 引擎**。详见 `docx/最终实验方案_实验内容_报告框架_v2.md`、跨语言契约 `INTERFACES.md`。

## 工作风格（用户偏好，务必遵守）
- **中文交流。**
- **不省资源，用满硬件，速度和效率优先**（不要为省算力而保守）。
- **并行推进**：多个独立任务用多 Agent 并行写代码。
- **写完代码立即自动 code review + 自动修复**（对抗性审查 → Edit 修 → 跑验证），不必等用户再要求。
- **失败即时清理**：安装/进程失败后立刻清残留——失败的 uv 缓存（`rm -rf /mnt/nvme1n1/.cache/uv/*`，曾堆到 27G）、孤儿进程（按 PID 或 `pkill -x <名>`，**禁用 `pkill -f <本命令含的字串>` 以免自杀**）、陈旧 pid/日志、`/dev/shm/*xqai*` 泄漏段。不让失败产物白占算力/磁盘。
- **所有文件放在工作空间** `/mnt/nvme3n1/gameTheory`（含 docx/、代码、数据、记忆）。
- 需要用户在其终端跑命令时，提示用 `! 命令`（如登录、本机全网下载）。
- 每次回答，先说rking

## 硬件 / 环境（已实测）
- **8× A800-SXM4-40GB（NVLink），独享整机**；192 线程（2×EPYC 7K62）；1TB 内存；`/mnt/nvme3n1` 1.7TB（数据/权重放这）。
- ⚠️ 根分区 `/` 紧张（避免往这写大文件）；ReplayBuffer 放 `/dev/shm`。
- ⚠️ **驱动 550 = CUDA 12.4 → torch 必须用 cu124**（`torch==2.6.0` 的 pypi 默认 wheel 即 cu124；默认最新 torch 是 cu130，会导致 GPU 不可用，切勿用）。
- ⚠️ **uv 缓存软链到 `/mnt/nvme1n1`（跨文件系统）**，安装时设 `UV_LINK_MODE=copy` 抑制告警。
- 资源分配：actor 卡数 >> learner（如 2 learner / 6 self-play）；每卡塞 2–4 worker；消融当独立任务并行跑。

## 工具链
- **Python：uv 管理**（venv 在 `.venv`，`uv pip install`）。
- **C++17**：g++ 11.4 + CMake/Ninja（pip wheel 装）+ pybind11 + scikit-build-core；C++ 单测用 doctest。`uv pip install -e .` 自动编译扩展 `xqai._xqcore`。
- 关键库：PyTorch（cu124）、numpy、tensorboard、matplotlib、pyyaml。

## 网络（重要）
- 我的沙箱 shell 默认下不动 github 主站/huggingface。
- **用户在本机执行 `clashon` 后，代理 `127.0.0.1:32917` 可访问 github**（`git clone`、raw、codeload 均可）；**huggingface 仍不通**。
- ⚠️ **pip 安装必须走清华镜像** `--index-url https://pypi.tuna.tsinghua.edu.cn/simple`（实测 **24MB/s**）；**直连 pypi 被限速 ~8KB/s、clash 代理 ~136KB/s**（torch ~4.5GB 否则一晚装不完）。`torch==2.6.0` 清华源即 cu124。uv 设 `UV_CACHE_DIR=/mnt/nvme3n1/.uvcache`（与 venv 同盘 hardlink 快）。
- 大数据/参考仓库：clash 开启后我可自行 `git clone`（带 dangerouslyDisableSandbox）。

## 关键技术决策
- 算法：AlphaZero 范式，核心 **Gumbel-AlphaZero(ICLR22)**；**PUCT 先行跑通学习曲线，Gumbel 同接口后挂**；融合 KataGo playout-cap、ReZero reanalyze；引用 Search-contempt(2025)。
- 学习循环用 **Python 向量化批量 MCTS（参考 TurboZero/MCTX），不上 C++ LibTorch 推理服务**（一周可落地关键）。C++ 只做规则内核 + Alpha-Beta 基线。
- 网络双轨：128×10 快迭代 + 256×15 最终主力，规模当消融。
- 动作编码：先 from*90+to=8100 + 合法掩码（零映射 bug），2086 紧凑编码作消融。
- 冷启动数据用 Pikafish 标注 / 人类棋谱（github 可下）。
- 三张杀手锏图：Elo-vs-训练步（叠加冷启动）/ 学习型 vs Alpha-Beta 同预算 crossover / Gumbel n_sim 消融。

## 代码现状（持续更新）
- `cpp/`：规则内核（perft 1–5 命中标准值）+ Alpha-Beta + pybind，已审查修复（FEN 往返 bug 已修）。
- `xqai/`：encoding/network/mcts(PUCT+Gumbel)/selfplay/replay(/dev/shm)/train/arena/dummynet，已审查修复（Gumbel 价值符号 bug 已修）。
- 待办：torch cu124 装好 → 编译 _xqcore → GPU 端到端冒烟 → 多卡自对弈 + 监督预训练。
