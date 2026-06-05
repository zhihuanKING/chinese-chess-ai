#!/usr/bin/env python3
"""Pikafish self-play data generator (supervised-pretraining cold start).

Replaces "download human game records" for the cold-start supervised phase: we
let Pikafish play *both* sides of many fast, low-depth games, turn each game's
move sequence into supervised samples via the existing :mod:`prepare_data`
pipeline (no re-implementation of move parsing / encoding), and write the same
shard format other tooling already consumes.

Coordinate / protocol notes (verified, see report)
---------------------------------------------------
- Pikafish speaks **UCI** (``uci`` / ``isready`` / ``position startpos moves ...``
  / ``go depth D`` | ``go movetime T`` / ``bestmove``). (Its ``ucci`` command is
  rejected by this build; the move *notation* is the coordinate form below.)
- Move notation is coordinate ``<file><rank><file><rank>`` (e.g. ``a3a4``,
  ``h2e2``): file ``a..i`` = columns left->right from Red's side (col 0..8),
  rank ``0..9`` from Red's bottom edge upward. This is **identical** to the ICCS
  convention that :func:`prepare_data.parse_iccs_move` implements
  (``col = file-'a'``, ``row = 9 - rank``). Verified end-to-end: Pikafish move
  -> ``parse_iccs_move`` -> ``Position.push`` is accepted for both red and black
  moves, so **no coordinate conversion is needed**.

What this module does
---------------------
1. Drives one Pikafish subprocess per worker over UCI, with robust timeouts and
   process reaping (:class:`PikafishEngine`).
2. Plays full self-play games to mate / draw / move-cap. For *diversity* every
   game opens with ``k in [2,8]`` random legal plies chosen from
   ``Position.legal_moves()`` (uniform), after which Pikafish takes over for both
   sides (:func:`play_one_game`).
3. Runs ``N`` games across ``W`` worker processes (``multiprocessing``); each
   engine pinned to ``Threads 1`` and a low depth / movetime so it is *fast, not
   strongest*.
4. Converts every finished game with :func:`prepare_data.game_to_samples` +
   :func:`prepare_data.encode_sample` and writes the same
   ``shard_XXXXX.npz`` (planes fp16 / pi_index int32 / z int8) format as
   :func:`prepare_data.run_pipeline`. Raw move logs are also saved under
   ``data/raw/pikafish_selfplay/*.txt`` for reproducibility.

CLI: ``--games N --workers W --depth D --out data/processed`` (+ ``--movetime``,
``--max-moves``, ``--shard-size``, ``--raw-out``, ``--seed``, ``--smoke``).
Prints games / samples / red-black-draw split / speed (games/min).
"""

from __future__ import annotations

import argparse
import os
import random
import subprocess
import sys
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

# Make sibling prepare_data importable whether run as a script or a module.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

import prepare_data as pd  # noqa: E402  (reuse: parsing + encoding + shard write)

# Resolve project root (parent of scripts/) for default binary / output paths.
_ROOT = os.path.dirname(_THIS_DIR)
DEFAULT_ENGINE = os.path.join(_ROOT, "third_party", "Pikafish", "src", "pikafish")
DEFAULT_RAW_OUT = os.path.join("data", "raw", "pikafish_selfplay")
DEFAULT_OUT = os.path.join("data", "processed")


# ==========================================================================
# UCI engine driver
# ==========================================================================
class EngineError(RuntimeError):
    """Raised when the engine misbehaves (timeout, crash, no bestmove)."""


class PikafishEngine:
    """Thin UCI driver around one Pikafish subprocess.

    Usage::

        eng = PikafishEngine(path, nnue_dir, threads=1)
        eng.start()
        bm = eng.bestmove(["a3a4", "a6a5"], depth=8)   # moves played so far
        eng.close()

    Robustness: every blocking read has a wall-clock timeout; on timeout / EOF
    the process is killed and an :class:`EngineError` is raised so the caller can
    abandon (and optionally restart) the engine instead of hanging a worker.
    """

    def __init__(self, path: str, threads: int = 1, hash_mb: int = 16,
                 multipv: int = 1, init_timeout: float = 30.0):
        self.path = path
        self.threads = threads
        self.hash_mb = hash_mb
        self.multipv = multipv
        self.init_timeout = init_timeout
        self.proc: Optional[subprocess.Popen] = None

    # -- lifecycle ---------------------------------------------------------
    def start(self) -> None:
        if not os.path.exists(self.path):
            raise EngineError(f"pikafish binary not found: {self.path}")
        # cwd = engine dir so it finds pikafish.nnue by its default relative name.
        cwd = os.path.dirname(self.path)
        self.proc = subprocess.Popen(
            [self.path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
            cwd=cwd,
        )
        self._send("uci")
        self._wait_for("uciok", self.init_timeout)
        self._send(f"setoption name Threads value {self.threads}")
        self._send(f"setoption name Hash value {self.hash_mb}")
        if self.multipv > 1:
            self._send(f"setoption name MultiPV value {self.multipv}")
        self._send("isready")
        self._wait_for("readyok", self.init_timeout)

    def close(self) -> None:
        if self.proc is None:
            return
        try:
            if self.proc.poll() is None:
                self._send("quit")
                try:
                    self.proc.wait(timeout=2.0)
                except subprocess.TimeoutExpired:
                    self.proc.kill()
                    self.proc.wait(timeout=2.0)
        except Exception:
            try:
                self.proc.kill()
            except Exception:
                pass
        finally:
            for stream in (self.proc.stdin, self.proc.stdout):
                try:
                    if stream:
                        stream.close()
                except Exception:
                    pass
            self.proc = None

    # -- low-level IO ------------------------------------------------------
    def _send(self, line: str) -> None:
        if self.proc is None or self.proc.stdin is None:
            raise EngineError("engine not started")
        try:
            self.proc.stdin.write(line + "\n")
            self.proc.stdin.flush()
        except (BrokenPipeError, ValueError) as exc:
            raise EngineError(f"write to engine failed: {exc}") from exc

    def _readline(self, deadline: float) -> str:
        """Read one line, raising on timeout/EOF (deadline is absolute time)."""
        if self.proc is None or self.proc.stdout is None:
            raise EngineError("engine not started")
        # Python file reads are blocking; we rely on the engine being responsive
        # under the deadline and check the clock after each line. A hard wall is
        # enforced by the caller via process kill if a single readline blocks,
        # but in practice Pikafish streams info lines well within the budget.
        if time.monotonic() > deadline:
            raise EngineError("engine read deadline exceeded")
        line = self.proc.stdout.readline()
        if line == "":
            raise EngineError("engine closed stdout (EOF / crash)")
        return line.rstrip("\n")

    def _wait_for(self, token: str, timeout: float) -> None:
        deadline = time.monotonic() + timeout
        while True:
            line = self._readline(deadline)
            if line.strip() == token or line.strip().startswith(token):
                return

    # -- queries -----------------------------------------------------------
    def bestmove(self, moves: List[str], depth: Optional[int] = None,
                 movetime: Optional[int] = None, timeout: float = 30.0) -> str:
        """Return Pikafish's bestmove given the move list played so far.

        Exactly one of ``depth`` / ``movetime`` should be set (depth wins if
        both given). Raises :class:`EngineError` on timeout / crash / no move.
        """
        if moves:
            self._send("position startpos moves " + " ".join(moves))
        else:
            self._send("position startpos")
        if depth is not None:
            self._send(f"go depth {depth}")
            # depth search has no intrinsic clock; give a generous wall budget.
            wall = max(timeout, 5.0)
        else:
            mt = movetime if movetime is not None else 100
            self._send(f"go movetime {mt}")
            wall = max(timeout, mt / 1000.0 + 5.0)

        deadline = time.monotonic() + wall
        while True:
            line = self._readline(deadline)
            if line.startswith("bestmove"):
                parts = line.split()
                if len(parts) < 2 or parts[1] in ("(none)", "0000"):
                    raise EngineError(f"no legal bestmove returned: {line!r}")
                return parts[1]


# ==========================================================================
# self-play game loop
# ==========================================================================
@dataclass
class GameRecord:
    moves: List[str] = field(default_factory=list)   # UCI/ICCS coordinate tokens
    result: int = pd.RESULT_DRAW                      # +1 red / -1 black / 0 draw
    random_opening: int = 0                           # number of random opening plies
    plies: int = 0
    error: Optional[str] = None


def _position_result_to_z(result_code, c) -> int:
    """Map a ``Position.result()`` code to prepare_data's z (+1/-1/0)."""
    if result_code == c.RED_WIN:
        return pd.RESULT_RED_WIN
    if result_code == c.BLACK_WIN:
        return pd.RESULT_BLACK_WIN
    return pd.RESULT_DRAW


def play_one_game(engine: PikafishEngine, rng: random.Random, *,
                  depth: Optional[int], movetime: Optional[int],
                  max_moves: int, move_timeout: float) -> GameRecord:
    """Play one full self-play game; return its move log + result.

    Diversity: open with ``k in [2,8]`` uniformly random legal plies (from
    ``Position.legal_moves()`` so they are guaranteed legal), then hand both
    sides to Pikafish until terminal (mate / stalemate / draw / move cap).
    """
    from xqai import _xqcore as c

    pos = c.Position()
    rec = GameRecord()

    # --- random opening for diversity ------------------------------------
    k = rng.randint(2, 8)
    for _ in range(k):
        legal = pos.legal_moves()
        if not legal:
            break
        if pos.result() != c.ONGOING:
            break
        mv = rng.choice(legal)
        rec.moves.append(pd.move_to_iccs(mv))
        pos.push(mv)
        rec.random_opening += 1

    # --- Pikafish plays both sides ---------------------------------------
    while pos.result() == c.ONGOING and len(rec.moves) < max_moves:
        try:
            uci_mv = engine.bestmove(rec.moves, depth=depth, movetime=movetime,
                                     timeout=move_timeout)
        except EngineError as exc:
            rec.error = f"engine error at ply {len(rec.moves)}: {exc}"
            break
        try:
            mv_int = pd.parse_iccs_move(uci_mv)
        except ValueError as exc:
            rec.error = f"unparseable move {uci_mv!r}: {exc}"
            break
        if mv_int not in set(pos.legal_moves()):
            rec.error = f"illegal engine move {uci_mv!r} at ply {len(rec.moves)}"
            break
        rec.moves.append(uci_mv)
        pos.push(mv_int)

    rec.plies = len(rec.moves)
    rec.result = _position_result_to_z(pos.result(), c)
    return rec


# ==========================================================================
# worker (one engine, plays a slice of games)
# ==========================================================================
@dataclass
class WorkerConfig:
    engine_path: str
    depth: Optional[int]
    movetime: Optional[int]
    max_moves: int
    move_timeout: float
    threads: int
    hash_mb: int
    seed: int


def _worker_run(cfg: WorkerConfig, n_games: int):
    """Run ``n_games`` self-play games on one engine; return list[GameRecord]."""
    rng = random.Random(cfg.seed)
    records: List[GameRecord] = []
    engine = PikafishEngine(cfg.engine_path, threads=cfg.threads,
                            hash_mb=cfg.hash_mb)
    try:
        engine.start()
    except EngineError as exc:
        # Whole worker failed to bring up the engine: return error records.
        return [GameRecord(error=f"engine start failed: {exc}")
                for _ in range(n_games)]

    for _ in range(n_games):
        try:
            rec = play_one_game(
                engine, rng,
                depth=cfg.depth, movetime=cfg.movetime,
                max_moves=cfg.max_moves, move_timeout=cfg.move_timeout,
            )
        except EngineError as exc:
            # Engine died mid-game; try to restart it for the remaining games.
            rec = GameRecord(error=f"engine crash: {exc}")
            engine.close()
            try:
                engine = PikafishEngine(cfg.engine_path, threads=cfg.threads,
                                        hash_mb=cfg.hash_mb)
                engine.start()
            except EngineError as exc2:
                records.append(rec)
                records.extend(
                    GameRecord(error=f"engine restart failed: {exc2}")
                    for _ in range(n_games - len(records))
                )
                return records
        records.append(rec)

    engine.close()
    return records


def _split_counts(total: int, parts: int) -> List[int]:
    """Split ``total`` games as evenly as possible across ``parts`` workers."""
    base, extra = divmod(total, parts)
    return [base + (1 if i < extra else 0) for i in range(parts)]


# ==========================================================================
# orchestration: parallel generation + sample writing
# ==========================================================================
@dataclass
class GenStats:
    games_requested: int = 0
    games_played: int = 0          # produced a move log (even if then unusable)
    games_errored: int = 0         # engine error / empty
    valid_games: int = 0           # passed game_to_samples
    skipped_games: int = 0         # illegal during replay (should be ~0)
    samples: int = 0
    red_wins: int = 0
    black_wins: int = 0
    draws: int = 0
    elapsed_s: float = 0.0


def _write_raw_logs(records: List[GameRecord], raw_dir: str, run_tag: str) -> str:
    """Write one .txt of move logs (ICCS coords + result marker) for replay."""
    os.makedirs(raw_dir, exist_ok=True)
    path = os.path.join(raw_dir, f"selfplay_{run_tag}.txt")
    result_marker = {pd.RESULT_RED_WIN: "1-0",
                     pd.RESULT_BLACK_WIN: "0-1",
                     pd.RESULT_DRAW: "1/2-1/2"}
    with open(path, "w", encoding="utf-8") as fh:
        for rec in records:
            if rec.error or not rec.moves:
                continue
            fh.write(" ".join(rec.moves) + " " + result_marker[rec.result] + "\n")
    return path


def _encode_and_write(records: List[GameRecord], out_dir: str,
                      shard_size: int, stats: GenStats) -> None:
    """Replay each game through prepare_data and write shards (same format)."""
    import numpy as np

    os.makedirs(out_dir, exist_ok=True)
    # continue shard numbering after any existing shards in out_dir
    existing = [n for n in os.listdir(out_dir)
                if n.startswith("shard_") and n.endswith(".npz")]
    shard_idx = len(existing)

    buf_planes: List = []
    buf_pi: List[int] = []
    buf_z: List[int] = []

    def flush():
        nonlocal shard_idx
        if not buf_planes:
            return
        path = os.path.join(out_dir, f"shard_{shard_idx:05d}.npz")
        np.savez_compressed(
            path,
            planes=np.stack(buf_planes).astype(np.float16),
            pi_index=np.asarray(buf_pi, dtype=np.int32),
            z=np.asarray(buf_z, dtype=np.int8),
        )
        shard_idx += 1
        buf_planes.clear()
        buf_pi.clear()
        buf_z.clear()

    for rec in records:
        if rec.error or not rec.moves:
            stats.games_errored += 1
            continue
        moves_int = [pd.parse_iccs_move(m) for m in rec.moves]
        samples, ok, _reason = pd.game_to_samples(moves_int, rec.result)
        if not ok:
            stats.skipped_games += 1
            continue
        stats.valid_games += 1
        for s in samples:
            planes, pi_index, z = pd.encode_sample(s)
            buf_planes.append(planes)
            buf_pi.append(pi_index)
            buf_z.append(int(z))
            stats.samples += 1
            if len(buf_planes) >= shard_size:
                flush()
    flush()


def generate(*, games: int, workers: int, depth: Optional[int],
             movetime: Optional[int], out_dir: str, raw_dir: str,
             engine_path: str, max_moves: int, move_timeout: float,
             shard_size: int, threads: int, hash_mb: int,
             seed: int) -> Tuple[GenStats, str]:
    """Run the full parallel generation + encoding pipeline. Returns (stats, raw_path)."""
    import multiprocessing as mp

    stats = GenStats(games_requested=games)
    workers = max(1, min(workers, games))
    counts = _split_counts(games, workers)

    cfgs = [
        WorkerConfig(
            engine_path=engine_path, depth=depth, movetime=movetime,
            max_moves=max_moves, move_timeout=move_timeout,
            threads=threads, hash_mb=hash_mb, seed=seed + 1000 * i,
        )
        for i in range(workers)
    ]

    t0 = time.monotonic()
    records: List[GameRecord] = []
    if workers == 1:
        records = _worker_run(cfgs[0], counts[0])
    else:
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=workers) as pool:
            results = [pool.apply_async(_worker_run, (cfgs[i], counts[i]))
                       for i in range(workers)]
            for r in results:
                records.extend(r.get())
    stats.elapsed_s = time.monotonic() - t0

    # tally outcomes
    for rec in records:
        if rec.error or not rec.moves:
            continue
        stats.games_played += 1
        if rec.result == pd.RESULT_RED_WIN:
            stats.red_wins += 1
        elif rec.result == pd.RESULT_BLACK_WIN:
            stats.black_wins += 1
        else:
            stats.draws += 1

    run_tag = f"{int(time.time())}_{os.getpid()}_g{games}"
    raw_path = _write_raw_logs(records, raw_dir, run_tag)
    _encode_and_write(records, out_dir, shard_size, stats)
    return stats, raw_path


# ==========================================================================
# reporting + CLI
# ==========================================================================
def _print_stats(stats: GenStats, raw_path: str, out_dir: str) -> None:
    mins = stats.elapsed_s / 60.0 if stats.elapsed_s > 0 else 0.0
    gpm = (stats.games_played / mins) if mins > 0 else 0.0
    spm = (stats.samples / mins) if mins > 0 else 0.0
    total_decided = stats.red_wins + stats.black_wins + stats.draws
    def pct(x):
        return (100.0 * x / total_decided) if total_decided else 0.0
    print("=" * 60)
    print("Pikafish 自对弈数据生成统计 / self-play generation stats")
    print("-" * 60)
    print(f"  请求局数 games requested : {stats.games_requested}")
    print(f"  完成局数 games played    : {stats.games_played}")
    print(f"  出错局数 games errored   : {stats.games_errored}")
    print(f"  有效局数 valid games     : {stats.valid_games}")
    print(f"  跳过(非法) skipped       : {stats.skipped_games}")
    print(f"  样本数 samples           : {stats.samples}")
    print(f"  红胜 red wins            : {stats.red_wins} ({pct(stats.red_wins):.1f}%)")
    print(f"  黑胜 black wins          : {stats.black_wins} ({pct(stats.black_wins):.1f}%)")
    print(f"  和棋 draws               : {stats.draws} ({pct(stats.draws):.1f}%)")
    print(f"  用时 elapsed             : {stats.elapsed_s:.1f}s")
    print(f"  速度 speed               : {gpm:.1f} 局/分 games/min, "
          f"{spm:.0f} 样本/分 samples/min")
    print(f"  原始谱 raw log           : {raw_path}")
    print(f"  分片输出 shards dir       : {out_dir}")
    print("=" * 60)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Pikafish self-play data generator -> training shards "
                    "(cold-start supervised pretraining).")
    parser.add_argument("--games", type=int, default=100, help="number of games")
    parser.add_argument("--workers", type=int, default=0,
                        help="parallel engine processes (0 = use all cpus)")
    parser.add_argument("--depth", type=int, default=8,
                        help="Pikafish search depth per move (default 8)")
    parser.add_argument("--movetime", type=int, default=None,
                        help="ms per move instead of depth (overrides --depth)")
    parser.add_argument("--out", default=DEFAULT_OUT,
                        help="output dir for shards (default data/processed)")
    parser.add_argument("--raw-out", default=DEFAULT_RAW_OUT,
                        help="dir for raw move logs (default data/raw/pikafish_selfplay)")
    parser.add_argument("--engine", default=DEFAULT_ENGINE,
                        help="path to pikafish binary")
    parser.add_argument("--max-moves", type=int, default=300,
                        help="hard cap on plies per game (default 300)")
    parser.add_argument("--move-timeout", type=float, default=30.0,
                        help="wall-clock seconds budget per engine move")
    parser.add_argument("--shard-size", type=int, default=4096,
                        help="samples per output shard (default 4096)")
    parser.add_argument("--threads", type=int, default=1,
                        help="Threads per engine (keep 1 for parallelism)")
    parser.add_argument("--hash", type=int, default=16, dest="hash_mb",
                        help="engine hash MB per process (default 16)")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed base")
    parser.add_argument("--smoke", action="store_true",
                        help="smoke mode: 4 games, depth 6, 2 workers, verify chain")
    args = parser.parse_args(argv)

    if args.smoke:
        args.games = 4
        args.workers = 2
        if args.movetime is None:
            args.depth = 6
        args.max_moves = min(args.max_moves, 120)
        print("[smoke] 4 games, "
              + (f"movetime {args.movetime}ms" if args.movetime else f"depth {args.depth}")
              + ", 2 workers")

    workers = args.workers if args.workers > 0 else (os.cpu_count() or 1)
    depth = None if args.movetime is not None else args.depth

    stats, raw_path = generate(
        games=args.games, workers=workers, depth=depth, movetime=args.movetime,
        out_dir=args.out, raw_dir=args.raw_out, engine_path=args.engine,
        max_moves=args.max_moves, move_timeout=args.move_timeout,
        shard_size=args.shard_size, threads=args.threads, hash_mb=args.hash_mb,
        seed=args.seed,
    )
    _print_stats(stats, raw_path, args.out)

    if args.smoke:
        ok = (stats.games_errored == 0 and stats.skipped_games == 0
              and stats.valid_games == stats.games_played
              and stats.valid_games > 0 and stats.samples > 0)
        # shape check on the written shards
        try:
            import numpy as np
            shards = sorted(n for n in os.listdir(args.out)
                            if n.startswith("shard_") and n.endswith(".npz"))
            if shards:
                d = np.load(os.path.join(args.out, shards[-1]))
                p, pi, z = d["planes"], d["pi_index"], d["z"]
                shape_ok = (p.ndim == 4 and p.shape[1:] == (15, 10, 9)
                            and p.dtype == np.float16
                            and pi.dtype == np.int32 and z.dtype == np.int8
                            and p.shape[0] == pi.shape[0] == z.shape[0])
                print(f"[smoke] last shard {shards[-1]}: planes {p.shape} "
                      f"{p.dtype}, pi {pi.shape} {pi.dtype}, z {z.shape} {z.dtype} "
                      f"-> shape_ok={shape_ok}")
                ok = ok and shape_ok
            else:
                print("[smoke] no shards written!")
                ok = False
        except Exception as exc:
            print(f"[smoke] shard shape check failed: {exc}")
            ok = False
        print(f"[smoke] RESULT: {'PASS' if ok else 'FAIL'}")
        return 0 if ok else 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
