#!/usr/bin/env python
"""Pikafish 常驻标注服务（CPU 侧，消费 DAgger FEN 队列）。

轮询 data/dagger/{queue_p0 > queue_p1 > queue_p2}（严格优先级：p0=对引擎败局
> p1=新鲜自对弈 > p2=回填），认领 ``*.jsonl`` shard（原子 rename 成 ``.working``
防多实例争抢），对每个 FEN 用 Pikafish ``depth14 MultiPV4``（``position fen``
直发，FEN 棋子字母 h/e -> n/b 翻译）标注，产出训练可用 shard 到
data/dagger/labeled/：

    shard_<tag>_<seq>.npz（np.savez 不压缩，可 mmap，ShardDataset 直接可读）
      planes      [N,15,10,9] f16   (xqai.encoding.encode，归一化行棋方视角)
      pi_idx      [N,K]       i16   稀疏软策略 索引（归一化视角，pad=idx0/val0）
      pi_val      [N,K]       f16   稀疏软策略 概率（softmax(cp/policy_temp)）
      z           [N]         f16   tanh(root_cp/value_scale)（兼容旧管线）
      cp          [N]         f32   root_cp 原始值（npz 内即 cp.npy；供 A0 的
                                     WDL 校准重映射成 z）
      outcome_red [N]         i8    真实终局（红视角；交叉验证用）
      stm         [N]         i8    行棋方 0=红 1=黑（z/cp 为行棋方视角，
                                     outcome_red 转 mover 视角需要它）
      ply         [N]         i16

编码复用 scripts/prepare_data.encode_soft_sample（含黑方 flip_move、cp 为行棋
方视角——与 Pikafish UCI 约定一致），稀疏化仅取其非零项（MultiPV<=K）。

去重：跨 shard 维护 data/dagger/seen_fens.txt（追加式，key=FEN 前两段
"局面 行棋方"，append 时 flock 防多实例交错）。

鲁棒性：
- 每个 worker 一个引擎进程；引擎崩溃自动重启该 worker 的引擎并重试一次。
- 终局/无着法 FEN（bestmove (none)）直接跳过，不重启。
- SIGTERM/SIGINT 优雅退出：把进行中的 ``.working`` 改回排队名（已落盘 chunk
  的 FEN 已进 seen 集合，重排队后自动跳过，不重复标注）。
- 启动时回收陈旧 ``.working``（mtime 超过 --stale-working-min 视为残留）。
- 每 60s 打一行吞吐统计（局面/h、各队列积压 shard 数）。

用法（正式服务示例）::

    .venv/bin/python scripts/label_service.py --workers 150

烟测::

    .venv/bin/python scripts/label_service.py --workers 8 --max-shards 1
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import signal
import sys
import time
from typing import Optional

import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)
_ROOT = os.path.dirname(_THIS_DIR)

import gen_pikafish_data as gp  # noqa: E402  (PikafishEngine + 二进制路径)
import prepare_data as pdata  # noqa: E402  (encode_soft_sample)

# Pikafish FEN 用 n(马)/b(象)，我们用 h/e；坐标着法两边一致（见 eval_anchor）。
_TRANS = str.maketrans("heHE", "nbNB")

QUEUE_NAMES = ["queue_p0", "queue_p1", "queue_p2"]  # 严格优先级

_STOP = False


def _on_signal(signum, frame):  # noqa: ARG001
    global _STOP
    _STOP = True
    print(f"[label] got signal {signum}, finishing up ...", flush=True)


# ==========================================================================
# worker 进程侧：常驻引擎 + 自动重启
# ==========================================================================
_W: dict = {}


def _w_init(engine_path: str, depth: int, multipv: int, hash_mb: int,
            timeout: float) -> None:
    # 主进程统一处理 Ctrl+C / TERM；worker 忽略，由 pool.terminate 收尾。
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    _W.update(engine_path=engine_path, depth=depth, multipv=multipv,
              hash_mb=hash_mb, timeout=timeout, eng=None)
    try:
        _w_start()
    except gp.EngineError as exc:  # 启动失败不让 Pool 初始化炸掉；任务内重试
        print(f"[worker {os.getpid()}] engine start failed: {exc}", flush=True)


def _w_start() -> None:
    eng = gp.PikafishEngine(_W["engine_path"], threads=1,
                            hash_mb=_W["hash_mb"], multipv=_W["multipv"])
    eng.start()
    _W["eng"] = eng


def _w_label(task):
    """task = (uid, fen_ours) -> (uid, ordered|None, root_cp|None)。

    引擎崩溃（EOF/超时/写失败）则重启引擎重试一次；终局局面直接放弃。
    """
    uid, fen = task
    fen_uci = fen.translate(_TRANS)
    for _attempt in range(2):
        try:
            if _W.get("eng") is None:
                _w_start()
            ordered, root_cp = _W["eng"].analyse(
                depth=_W["depth"], timeout=_W["timeout"], fen=fen_uci)
            return uid, ordered, root_cp
        except gp.EngineError as exc:
            msg = str(exc)
            if "no legal bestmove" in msg or "no multipv info" in msg:
                # 终局/无解局面：引擎还活着，跳过该样本即可。
                return uid, None, None
            # 崩溃：清理 + 重启后重试。
            eng = _W.get("eng")
            _W["eng"] = None
            if eng is not None:
                try:
                    eng.close()
                except Exception:
                    pass
        except Exception as exc:  # 防御：任何异常都不能杀死 worker
            print(f"[worker {os.getpid()}] unexpected: {exc!r}", flush=True)
            return uid, None, None
    return uid, None, None


# ==========================================================================
# 主进程侧：队列认领 / 去重 / 编码落盘
# ==========================================================================
def fen_key(fen: str) -> str:
    """去重 key：FEN 前两段（局面 + 行棋方）。"""
    parts = fen.split()
    return " ".join(parts[:2])


def load_seen(path: str) -> set:
    seen = set()
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    seen.add(line)
    return seen


def append_seen(path: str, keys: list) -> None:
    if not keys:
        return
    with open(path, "a", encoding="utf-8") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        fh.write("\n".join(keys) + "\n")
        fh.flush()
        fcntl.flock(fh, fcntl.LOCK_UN)


def recover_stale_working(qdirs: list, stale_min: float) -> None:
    """把超过 stale_min 分钟没动静的 .working 改回排队名（残留回收）。"""
    now = time.time()
    for qd in qdirs:
        try:
            names = os.listdir(qd)
        except FileNotFoundError:
            continue
        for n in names:
            if not n.endswith(".jsonl.working"):
                continue
            p = os.path.join(qd, n)
            try:
                if now - os.path.getmtime(p) > stale_min * 60.0:
                    os.rename(p, p[:-len(".working")])
                    print(f"[label] recovered stale {p}", flush=True)
            except OSError:
                pass


def claim_shard(qdirs: list) -> Optional[tuple]:
    """按优先级认领一个 shard：rename 成 .working。返回 (working_path, orig_path)。"""
    for qd in qdirs:
        try:
            names = sorted(n for n in os.listdir(qd) if n.endswith(".jsonl"))
        except FileNotFoundError:
            continue
        for n in names:
            src = os.path.join(qd, n)
            dst = src + ".working"
            try:
                os.rename(src, dst)  # 原子：多实例只有一个成功
                return dst, src
            except OSError:
                continue  # 被别的实例抢走/已消失
    return None


class LabeledWriter:
    """攒样本写 labeled npz shard；flush 时把该 chunk 的 fen key 追加进 seen 文件。"""

    def __init__(self, out_dir: str, tag: str, k: int, shard_size: int,
                 seen_path: str, seen: set, policy_temp: float,
                 value_scale: float):
        self.out_dir = out_dir
        self.tag = tag
        self.k = int(k)
        self.shard_size = int(shard_size)
        self.seen_path = seen_path
        self.seen = seen
        self.policy_temp = policy_temp
        self.value_scale = value_scale
        self.seq = 0
        self.total_written = 0
        self._reset()
        os.makedirs(out_dir, exist_ok=True)

    def _reset(self) -> None:
        self._planes: list = []
        self._idx: list = []
        self._val: list = []
        self._z: list = []
        self._cp: list = []
        self._outcome: list = []
        self._stm: list = []
        self._ply: list = []
        self._keys: list = []

    def add(self, fen: str, ordered, root_cp: float, outcome_red: int,
            ply: int, key: str) -> bool:
        """编码一个标注样本进缓冲。返回 False 表示编码失败（跳过）。"""
        side_is_black = fen.split()[1] == "b"
        try:
            planes, pi, z = pdata.encode_soft_sample(
                fen, side_is_black, ordered, root_cp,
                policy_temp=self.policy_temp, value_scale=self.value_scale)
        except Exception as exc:
            print(f"[label] encode failed ({exc}) fen={fen!r}", flush=True)
            return False
        # 稀疏化：MultiPV<=K 个非零项 -> pi_idx/pi_val（pad idx=0/val=0；
        # move 0 即 from0->to0 不可能合法，scatter 后被 pi>0 掩码剔除）。
        pi32 = pi.astype(np.float32)
        nz = np.nonzero(pi32)[0]
        if nz.size == 0:
            return False
        if nz.size > self.k:  # multipv > k 时只留 top-k
            nz = nz[np.argsort(pi32[nz])[::-1][:self.k]]
        idx = np.zeros(self.k, dtype=np.int16)
        val = np.zeros(self.k, dtype=np.float16)
        idx[:nz.size] = nz.astype(np.int16)
        val[:nz.size] = pi32[nz].astype(np.float16)

        self._planes.append(planes)
        self._idx.append(idx)
        self._val.append(val)
        self._z.append(z)
        self._cp.append(np.float32(root_cp))
        self._outcome.append(np.int8(outcome_red))
        self._stm.append(np.int8(1 if side_is_black else 0))
        self._ply.append(np.int16(ply))
        self._keys.append(key)
        self.seen.add(key)  # 进程内立即去重；文件在 flush 时追加
        if len(self._planes) >= self.shard_size:
            self.flush()
        return True

    def flush(self) -> None:
        if not self._planes:
            return
        name = f"shard_{self.tag}_{self.seq:05d}.npz"
        tmp = os.path.join(self.out_dir, "." + name + ".tmp")
        np.savez(  # 不压缩 -> ShardDataset 可 mmap
            tmp,
            planes=np.stack(self._planes).astype(np.float16),
            pi_idx=np.stack(self._idx),
            pi_val=np.stack(self._val),
            z=np.asarray(self._z, dtype=np.float16),
            cp=np.asarray(self._cp, dtype=np.float32),
            outcome_red=np.asarray(self._outcome, dtype=np.int8),
            stm=np.asarray(self._stm, dtype=np.int8),
            ply=np.asarray(self._ply, dtype=np.int16),
        )
        # np.savez 会自动补 .npz 后缀到 tmp 路径
        os.replace(tmp + ".npz", os.path.join(self.out_dir, name))
        append_seen(self.seen_path, self._keys)
        self.total_written += len(self._planes)
        print(f"[label] wrote {name}: {len(self._planes)} samples", flush=True)
        self.seq += 1
        self._reset()


class Stats:
    def __init__(self, qdirs: list):
        self.qdirs = qdirs
        self.t0 = time.monotonic()
        self.last = self.t0
        self.labeled = 0
        self.failed = 0
        self.skipped_dup = 0

    def maybe_print(self, force: bool = False) -> None:
        now = time.monotonic()
        if not force and now - self.last < 60.0:
            return
        self.last = now
        rate = self.labeled / max(now - self.t0, 1e-9) * 3600.0
        backlog = []
        for qd in self.qdirs:
            try:
                n = sum(1 for x in os.listdir(qd) if x.endswith(".jsonl"))
            except FileNotFoundError:
                n = 0
            backlog.append(str(n))
        print(f"[stats] labeled={self.labeled} ({rate:.0f} pos/h) "
              f"failed={self.failed} dup_skipped={self.skipped_dup} "
              f"backlog p0/p1/p2={'/'.join(backlog)} shards", flush=True)


def process_shard(pool, working: str, orig: str, writer: LabeledWriter,
                  seen: set, stats: Stats, chunksize: int) -> bool:
    """标注一个已认领 shard。返回 True=完成（删源文件），False=中断（已回排队）。"""
    items: list = []  # (fen, outcome_red, ply, key)
    local = set()
    with open(working, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                fen = rec["fen"]
            except (ValueError, KeyError):
                continue
            key = fen_key(fen)
            if key in seen or key in local:
                stats.skipped_dup += 1
                continue
            local.add(key)
            items.append((fen, int(rec.get("outcome_red", 0)),
                          int(rec.get("ply", -1)), key))

    print(f"[label] processing {os.path.basename(orig)}: "
          f"{len(items)} unique positions", flush=True)
    if not items:
        os.remove(working)
        return True

    tasks = [(i, it[0]) for i, it in enumerate(items)]
    interrupted = False
    try:
        for uid, ordered, root_cp in pool.imap_unordered(
                _w_label, tasks, chunksize=chunksize):
            fen, outcome_red, ply, key = items[uid]
            if ordered is None:
                stats.failed += 1
            elif writer.add(fen, ordered, root_cp, outcome_red, ply, key):
                stats.labeled += 1
            else:
                stats.failed += 1
            stats.maybe_print()
            if _STOP:
                interrupted = True
                break
    except Exception as exc:
        print(f"[label] pool error: {exc!r}; requeueing shard", flush=True)
        interrupted = True

    writer.flush()  # 每 shard 落一次盘（已写 chunk 的 key 进了 seen，可安全重排队）
    if interrupted:
        try:
            os.rename(working, orig)  # 回排队：剩余未标 FEN 下次继续
            print(f"[label] requeued {os.path.basename(orig)}", flush=True)
        except OSError:
            pass
        return False
    os.remove(working)  # 消费完毕
    return True


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Pikafish 常驻标注服务（DAgger 队列消费）")
    ap.add_argument("--base-dir", default=os.path.join(_ROOT, "data/dagger"))
    ap.add_argument("--engine", default=gp.DEFAULT_ENGINE)
    ap.add_argument("--workers", type=int, default=150)
    ap.add_argument("--depth", type=int, default=14)
    ap.add_argument("--multipv", type=int, default=4)
    ap.add_argument("--hash", type=int, default=16, dest="hash_mb",
                    help="每引擎 Hash MB（150 进程 x (NNUE~45MB+Hash) ~10GB，1TB 内存无压力）")
    ap.add_argument("--move-timeout", type=float, default=120.0,
                    help="单 FEN 分析墙钟上限（秒）")
    ap.add_argument("--shard-size", type=int, default=8192,
                    help="labeled npz 每片样本数")
    ap.add_argument("--topk", type=int, default=0,
                    help="稀疏策略 K（0=取 multipv）")
    ap.add_argument("--policy-temp", type=float, default=100.0)
    ap.add_argument("--value-scale", type=float, default=500.0)
    ap.add_argument("--poll-interval", type=float, default=10.0)
    ap.add_argument("--stale-working-min", type=float, default=120.0,
                    help="启动时回收超过 N 分钟的 .working 残留")
    ap.add_argument("--max-shards", type=int, default=0,
                    help=">0 时处理 N 个 shard 后退出（烟测用）")
    ap.add_argument("--chunksize", type=int, default=2)
    args = ap.parse_args(argv)

    qdirs = [os.path.join(args.base_dir, q) for q in QUEUE_NAMES]
    out_dir = os.path.join(args.base_dir, "labeled")
    seen_path = os.path.join(args.base_dir, "seen_fens.txt")
    k = args.topk if args.topk > 0 else args.multipv

    signal.signal(signal.SIGTERM, _on_signal)
    signal.signal(signal.SIGINT, _on_signal)

    recover_stale_working(qdirs, args.stale_working_min)
    seen = load_seen(seen_path)
    print(f"[label] engine={args.engine} workers={args.workers} "
          f"depth={args.depth} multipv={args.multipv} | seen={len(seen)} fens",
          flush=True)

    tag = f"{int(time.time())}_{os.getpid()}"
    writer = LabeledWriter(out_dir, tag, k, args.shard_size, seen_path, seen,
                           args.policy_temp, args.value_scale)
    stats = Stats(qdirs)

    import multiprocessing as mp
    ctx = mp.get_context("spawn")  # 干净的子进程（与项目其余 mp 用法一致）
    pool = ctx.Pool(processes=args.workers, initializer=_w_init,
                    initargs=(args.engine, args.depth, args.multipv,
                              args.hash_mb, args.move_timeout))
    done_shards = 0
    try:
        while not _STOP:
            claim = claim_shard(qdirs)
            if claim is None:
                stats.maybe_print()
                # 空闲轮询（小步 sleep 以便信号及时响应）
                t_end = time.monotonic() + args.poll_interval
                while time.monotonic() < t_end and not _STOP:
                    time.sleep(0.2)
                continue
            working, orig = claim
            completed = process_shard(pool, working, orig, writer, seen,
                                      stats, args.chunksize)
            if completed:
                done_shards += 1
                if args.max_shards and done_shards >= args.max_shards:
                    print(f"[label] max-shards={args.max_shards} reached", flush=True)
                    break
    finally:
        writer.flush()
        stats.maybe_print(force=True)
        pool.terminate()  # worker 死后引擎读到 stdin EOF 自行退出
        pool.join()
        print(f"[label] exit: total_written={writer.total_written} "
              f"shards_done={done_shards}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
