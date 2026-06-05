# 跨语言接口契约（所有并行开发必须遵守，先读这个）

> 这是 C++ / Python 各模块之间的**唯一契约**。任何模块只依赖本文定义的接口，不依赖别人内部实现。改契约必须先改本文件并知会。

## 1. 棋盘与坐标约定
- 棋盘 9 列 × 10 行 = 90 格。`sq ∈ [0,90)`，`row = sq // 9`（0..9），`col = sq % 9`（0..8）。
- **红方在底部（row 9 一侧），黑方在顶部（row 0 一侧）**。红先行。
- 棋子在 `board[]` 中用 `int8`：`0`=空；**正数=红、负数=黑**；绝对值 `1..7` 依次 = `将/帅 K, 士/仕 A, 象/相 E, 马 H, 车 R, 炮 C, 兵/卒 P`。

## 2. 着法整数编码（全局统一）
- **`move = from_sq * 90 + to_sq`**，范围 `[0, 8100)`。象棋无升变，编码无歧义。
- Python 侧：`mv_from(m)=m//90`，`mv_to(m)=m%90`。
- **动作空间维度 `ACTION_DIM = 8100`（起步，零映射 bug）**。2086 紧凑编码作为后续优化/消融，由 `xqai/encoding.py` 内可切换，不改本契约的 move 整数定义。

## 3. C++ pybind 模块：`xqai._xqcore`
编译目标：`cpp/CMakeLists.txt` 产出扩展模块 `_xqcore`（装入 `xqai` 包）。导出一个 `Position` 类：

```python
from xqai._xqcore import Position, RED, BLACK, ONGOING, RED_WIN, BLACK_WIN, DRAW

p = Position()                 # 初始局面
p = Position.from_fen(fen)     # FEN 构造
p.fen() -> str                 # 导出 FEN（往返必须无损）
p.side_to_move() -> int        # RED=0 / BLACK=1
p.legal_moves() -> list[int]   # 当前方所有合法着法（已过滤送将/自杀着）
p.is_check() -> bool           # 当前方是否被将军
p.push(move:int) -> None       # 走子（就地修改，维护历史栈）
p.pop() -> None                # 撤销上一步
p.result() -> int              # ONGOING/RED_WIN/BLACK_WIN/DRAW（含将死/困毙/重复判和/自然限着）
p.zobrist() -> int             # 当前局面 64 位 zobrist（make/unmake 后必须可复现）
p.board() -> bytes             # 长度 90 的 int8 视图（见 §1 编码），row-major
p.repetition_count() -> int    # 当前局面历史重复次数
p.ply() -> int                 # 已走半回合数
Position.perft(fen:str, depth:int) -> int   # 静态方法，叶子计数（测试用，C++实现）
```
- 规则简化版（报告需声明）：**第 3 次重复判和 + 120 回合自然限着判和**；长将/长捉暂按重复处理，完整规则列展望。
- 性能：`legal_moves/push/pop` 是热路径，C++ 内部用 mailbox(int8[90]) + 增量 zobrist。

## 4. Python 侧张量约定（`xqai/encoding.py`）
```python
ACTION_DIM = 8100
NUM_PLANES = 15                      # 起步，可加（消融）
def encode(pos) -> np.ndarray        # 返回 float32 [NUM_PLANES, 10, 9]
def legal_mask(pos) -> np.ndarray    # 返回 bool/float32 [ACTION_DIM]，合法=1
def policy_target(visit_counts: dict[int,int]) -> np.ndarray  # [ACTION_DIM]，归一化
```
输入平面（**始终以"当前行动方"视角规范化**：黑走时把棋盘翻转成红视角再编码）：
- 平面 0..6：当前方 K,A,E,H,R,C,P 的 one-hot；平面 7..13：对方 7 类；平面 14：重复计数(归一化)。

## 5. 神经网络约定（`xqai/network.py`）
```python
class PVNet(nn.Module):
    # 输入 x: float32 [B, NUM_PLANES, 10, 9]
    # 返回 (policy_logits [B, ACTION_DIM], value [B, 1] in [-1,1] tanh)
    def __init__(self, channels=128, blocks=10, action_dim=ACTION_DIM, num_planes=NUM_PLANES): ...
```
- 非法着法在 MCTS/损失里用 `legal_mask` 置 -inf 后 softmax。
- 损失：`L = (z - v)^2 - sum(pi * log_softmax(masked_logits)) + c*||w||^2`，c=1e-4。

## 6. MCTS / 自对弈约定（`xqai/mcts.py`, `xqai/selfplay.py`）
- `Planner` 统一接口：`def search(positions: list[Position], net, n_sim:int) -> list[np.ndarray]`，返回每局的改进策略 π（[ACTION_DIM]）。
- 两种实现：`PUCTPlanner`（先行）、`GumbelPlanner`（后挂，同接口）。
- **向量化批量**：一次处理 B 局并行对局，把各局待评估叶子拼成一个大 batch 调 `net` 前向（参考 TurboZero/MCTX）。
- 自对弈样本三元组 `(planes:[C,10,9] float16, pi:[ACTION_DIM] 稀疏, z:int8)`，写入 ReplayBuffer。

## 7. ReplayBuffer（`xqai/replay.py`）
- 环形缓冲，容量可配（默认 2e6，最大 5e6），**存于 `/dev/shm`**（共享内存，多进程读写）。
- 近期加权采样 + 左右镜像增广。

## 8. Alpha-Beta 基线（`cpp/`，独立可执行 + 可选 pybind）
- 复用 §3 的规则内核（同一份 movegen/rules）。
- Negamax + α-β 剪枝 + 置换表(zobrist) + 迭代加深 + 静态搜索(quiescence) + 走子排序。
- 分阶段手写评估：子力 + 位置表 + 机动性 + 将安全。
- 接口：`int search(Position&, int time_ms)` 返回 best move；命令行支持 UCCI 子集，便于 Arena 对弈。

## 9. 配置 / 评估
- 全局配置 `configs/default.yaml`（网络/MCTS/优化/数据/分布式 超参）。
- Arena：引擎对弈→PGN/结果，BayesElo/Ordo 反推 Elo（配对开局、红黑各半）。

## 10. 目录
```
cpp/{include/xq,src,bindings,tests}  C++ 规则内核 + AB + pybind + perft/单测
xqai/{encoding,network,mcts,selfplay,replay,train,arena,config}.py
scripts/   下数据/跑实验/画图
configs/   yaml
data/ checkpoints/ logs/   (gitignore)
```
