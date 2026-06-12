"""Vectorized arena evaluation for xqai (batch counterpart of :mod:`xqai.arena`).

:func:`xqai.arena.play_match` plays games one by one; every NetPlayer move is a
``planner.search([pos], ...)`` with batch size 1, so the GPU idles. This module
keeps **N games in flight** and, per lockstep round, batches all positions where
the same (net) player is to move into ONE ``planner.search`` call -- the same
trick :mod:`xqai.selfplay` uses, applied to evaluation matches.

Pairing / scoring semantics are *identical* to :func:`xqai.arena.play_match`:

- ``ceil(games/2)`` openings (cycled from ``openings``; None => startpos), each
  played twice with colours swapped, truncated to ``games`` (an odd ``games``
  means the last opening is only played with A as red).
- per game: result checked at the top of each of ``max_plies`` iterations; an
  illegal move forfeits for the mover; ``max_plies`` moves played => DRAW
  (even if the very last move mated -- this mirrors the serial loop exactly).
- the returned dict has the same keys (A's perspective, Wilson CI, Elo).

Engine opponents get **one engine subprocess per in-flight game** (capped by
``max_engines``); each move is sent with the full move history::

    position fen <start_fen> moves <m1> <m2> ...

so engines see repetitions (the serial arena's per-position ``position fen``
left them blind to them). Engine replies are awaited concurrently; replies
cannot cross games because every game is pinned to its own subprocess pipe.

Evaluation must be deterministic: net players pick argmax of the improved
policy and the planner should be constructed with ``add_noise=False`` (a
loud warning is emitted otherwise).
"""

from __future__ import annotations

import subprocess
import threading
import time
import warnings
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from . import arena
from .arena import (  # re-exported for convenience / shared semantics
    _ucci_to_move,
    elo_from_winrate,
    wilson_interval,
)
from .encoding import BLACK, flip_move

_ONGOING = 0
_RED_WIN = 1
_BLACK_WIN = 2
_DRAW = 3

# Pikafish (UCI) spells knight/elephant n/b where this project uses h/e.
_PIKA_TRANS = str.maketrans("heHE", "nbNB")


def _position_from_fen(fen: str | None):
    from xqai._xqcore import Position

    return Position.from_fen(fen) if fen else Position()


def move_to_ucci(m: int) -> str:
    """Inverse of :func:`xqai.arena._ucci_to_move` (``from*90+to`` -> ``h2e2``)."""
    f, t = m // 90, m % 90
    return (
        chr(ord("a") + f % 9)
        + str(9 - f // 9)
        + chr(ord("a") + t % 9)
        + str(9 - t // 9)
    )


# --------------------------------------------------------------------------- #
# Per-game state                                                              #
# --------------------------------------------------------------------------- #
class _VecGame:
    __slots__ = (
        "pos",
        "start_fen",
        "a_is_red",
        "moves_ucci",
        "ply",
        "result",
        "engines",
    )

    def __init__(self, opening_fen: str | None, a_is_red: bool):
        self.pos = _position_from_fen(opening_fen)
        # Canonical start FEN from the engine itself (round-trips since the
        # FEN bug fix) -- engines get "position fen <this> moves ...".
        self.start_fen = self.pos.fen()
        self.a_is_red = a_is_red
        self.moves_ucci: list[str] = []  # real-frame history for engines
        self.ply = 0
        self.result = _ONGOING
        self.engines: dict[int, "_EngineClient"] = {}  # id(vec_player) -> client


# --------------------------------------------------------------------------- #
# Vec-player adapters                                                         #
# --------------------------------------------------------------------------- #
class _VecPlayerBase:
    """Adapter interface: pick real-frame moves for a *batch* of games."""

    name = "vec"
    max_parallel: float = float("inf")
    # Round phase: 0 = cheap/concurrent movers (engines, random) go first so
    # the games they advance can join the same round's (GPU) net batch.
    phase: int = 1

    def begin_game(self, game: _VecGame) -> None:  # per-game setup (reset)
        pass

    def end_game(self, game: _VecGame) -> None:  # per-game teardown
        pass

    def batch_select(self, games: list[_VecGame]) -> list[int]:
        raise NotImplementedError

    def close(self) -> None:
        pass


class NetVecPlayer(_VecPlayerBase):
    """Batched twin of :class:`xqai.arena.NetPlayer` (argmax of improved pi)."""

    def __init__(self, net, planner, n_sim: int = 200, name: str = "net"):
        self.net = net
        self.planner = planner
        self.n_sim = int(n_sim)
        self.name = name
        if getattr(planner, "add_noise", False):
            warnings.warn(
                f"NetVecPlayer({name}): planner has add_noise=True -- evaluation "
                "should be deterministic (add_noise=False).",
                stacklevel=2,
            )

    @classmethod
    def from_net_player(cls, p: "arena.NetPlayer") -> "NetVecPlayer":
        return cls(p.net, p.planner, n_sim=p.n_sim, name=p.name)

    def batch_select(self, games: list[_VecGame]) -> list[int]:
        positions = [g.pos for g in games]
        # === ONE batched MCTS (single net forward per sim) for all games. ===
        pis = self.planner.search(positions, self.net, self.n_sim)
        moves: list[int] = []
        for pos, pi in zip(positions, pis):
            nz = np.nonzero(pi)[0]
            if nz.size == 0:  # same fallback as arena.NetPlayer.select_move
                legal = pos.legal_moves()
                moves.append(legal[0] if legal else -1)
                continue
            move_norm = int(nz[np.argmax(pi[nz])])
            moves.append(
                flip_move(move_norm) if pos.side_to_move() == BLACK else move_norm
            )
        return moves


class SerialVecPlayer(_VecPlayerBase):
    """Wraps any serial ``Player`` (e.g. a random mover) -- looping is cheap."""

    phase = 0

    def __init__(self, player):
        self.player = player
        self.name = getattr(player, "name", "player")

    def begin_game(self, game: _VecGame) -> None:
        self.player.reset()

    def batch_select(self, games: list[_VecGame]) -> list[int]:
        return [self.player.select_move(g.pos) for g in games]


class _EngineClient:
    """One engine subprocess speaking UCI or a UCCI subset.

    Unlike :class:`xqai.arena.SubprocessUCCIPlayer` it always sends the full
    move history (``position fen X moves ...``) so the engine sees repetitions,
    and it splits ``send_go`` / ``read_bestmove`` so many engines think
    concurrently.
    """

    def __init__(
        self,
        cmd: list[str],
        *,
        proto: str = "ucci",  # "ucci" | "uci"
        depth: int | None = None,
        movetime_ms: int | None = 1000,
        translate_fen: bool = False,  # h/e -> n/b (pikafish)
        options: dict[str, object] | None = None,
        name: str = "engine",
    ):
        self.cmd = cmd
        self.proto = proto
        self.depth = depth
        self.movetime_ms = movetime_ms
        self.translate_fen = translate_fen
        self.options = options or {}
        self.name = name
        self.proc: subprocess.Popen | None = None
        self._start()

    def _start(self) -> None:
        self.proc = subprocess.Popen(
            self.cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
        )
        if self.proto == "uci":
            self._send("uci")
            self._wait_for("uciok")
        else:
            self._send("ucci")
            self._wait_for("ucciok")
        for k, v in self.options.items():
            self._send(f"setoption name {k} value {v}")
        self._send("isready")
        self._wait_for("readyok")

    def _send(self, line: str) -> None:
        assert self.proc and self.proc.stdin
        self.proc.stdin.write(line + "\n")
        self.proc.stdin.flush()

    def _wait_for(self, token: str, max_lines: int = 5000) -> str:
        assert self.proc and self.proc.stdout
        for _ in range(max_lines):
            line = self.proc.stdout.readline()
            if not line:
                break
            if token in line:
                return line
        return ""

    def alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def new_game(self) -> None:
        self._send("ucinewgame")

    def send_go(self, start_fen: str, moves_ucci: list[str]) -> None:
        """Write ``position ... moves ...`` + ``go``; returns immediately."""
        fen = start_fen.translate(_PIKA_TRANS) if self.translate_fen else start_fen
        cmd = f"position fen {fen}"
        if moves_ucci:
            cmd += " moves " + " ".join(moves_ucci)
        try:
            self._send(cmd)
            if self.depth is not None:
                self._send(f"go depth {self.depth}")
            else:
                self._send(f"go movetime {self.movetime_ms or 1000}")
        except (BrokenPipeError, OSError):
            pass  # dead engine -> read_bestmove returns -1 -> forfeit

    def read_bestmove(self) -> int:
        """Block until ``bestmove``; -1 on timeout / death (mover forfeits).

        Any -1 path **kills the subprocess**: a timed-out engine may still
        print its ``bestmove`` later, and if this (pooled) client were reused
        for another game that stale reply would be read as the new game's move.
        Dead clients are replaced on reuse (see ``EngineVecPlayer.begin_game``).
        """
        if not self.alive():
            return -1
        assert self.proc and self.proc.stdout
        budget_ms = self.movetime_ms if self.movetime_ms else 30_000
        deadline = time.monotonic() + budget_ms / 1000.0 + 5.0
        watchdog = threading.Timer(max(deadline - time.monotonic(), 0.1), self._kill)
        watchdog.daemon = True
        watchdog.start()
        move = -1
        try:
            for _ in range(1_000_000):
                if time.monotonic() > deadline:
                    break
                line = self.proc.stdout.readline()
                if not line:
                    break
                if line.startswith("bestmove"):
                    parts = line.split()
                    if len(parts) >= 2:
                        move = _ucci_to_move(parts[1])
                    break
            return move
        finally:
            watchdog.cancel()
            if move < 0:
                self._kill()  # never reuse a pipe that may hold a stale reply

    def _kill(self) -> None:
        if self.proc and self.proc.poll() is None:
            try:
                self.proc.kill()
            except Exception:
                pass

    def close(self) -> None:
        if self.proc:
            try:
                self._send("quit")
                self.proc.wait(timeout=2)
            except Exception:
                self._kill()


class EngineVecPlayer(_VecPlayerBase):
    """One engine subprocess per in-flight game, replies awaited concurrently.

    ``factory()`` must return a fresh :class:`_EngineClient`. Instances are
    pooled and reused across games (``ucinewgame`` between games); at most
    ``max_engines`` subprocesses ever exist, and :func:`play_match_vec` clamps
    its in-flight game count to that, so ``begin_game`` always finds capacity.
    """

    phase = 0

    def __init__(self, factory, *, max_engines: int = 16, name: str = "engine"):
        self.factory = factory
        self.max_engines = int(max_engines)
        self.max_parallel = int(max_engines)
        self.name = name
        self._free: list[_EngineClient] = []
        self._all: list[_EngineClient] = []
        self._pool = ThreadPoolExecutor(max_workers=self.max_engines)

    def begin_game(self, game: _VecGame) -> None:
        if id(self) in game.engines:
            # Same EngineVecPlayer on both sides of the match: reuse the one
            # client for both colours (don't spawn + leak a second).
            return
        if self._free:
            client = self._free.pop()
            if not client.alive():  # crashed earlier -> replace
                self._all.remove(client)
                client = self._spawn()
        else:
            if len(self._all) >= self.max_engines:
                raise RuntimeError(
                    f"EngineVecPlayer({self.name}): out of engine capacity "
                    f"({self.max_engines}); play_match_vec should clamp parallel."
                )
            client = self._spawn()
        client.new_game()
        game.engines[id(self)] = client

    def _spawn(self) -> _EngineClient:
        client = self.factory()
        self._all.append(client)
        return client

    def end_game(self, game: _VecGame) -> None:
        client = game.engines.pop(id(self), None)
        if client is not None:
            self._free.append(client)

    def fire(self, games: list[_VecGame]) -> None:
        """Write "position + go" to every game's engine (non-blocking, tiny)."""
        for g in games:
            g.engines[id(self)].send_go(g.start_fen, g.moves_ucci)

    def collect(self, games: list[_VecGame]) -> list[int]:
        """Await every bestmove concurrently. Each game owns its own
        subprocess pipe, so replies can never be attributed to the wrong game;
        order is preserved by ThreadPoolExecutor.map."""
        clients = [g.engines[id(self)] for g in games]
        return list(self._pool.map(lambda c: c.read_bestmove(), clients))

    def batch_select(self, games: list[_VecGame]) -> list[int]:
        self.fire(games)
        return self.collect(games)

    def close(self) -> None:
        for c in self._all:
            c.close()
        self._all.clear()
        self._free.clear()
        self._pool.shutdown(wait=False)


def _as_vec_player(player, max_engines: int) -> _VecPlayerBase:
    """Adapt an arena-style player / vec player into a :class:`_VecPlayerBase`."""
    if isinstance(player, _VecPlayerBase):
        return player
    if isinstance(player, arena.NetPlayer):
        return NetVecPlayer.from_net_player(player)
    if isinstance(player, arena.SubprocessUCCIPlayer):
        cmd, depth, mt = player.cmd, player.depth, player.movetime_ms
        name = player.name
        return EngineVecPlayer(
            lambda: _EngineClient(
                list(cmd), proto="ucci", depth=depth, movetime_ms=mt, name=name
            ),
            max_engines=max_engines,
            name=name,
        )
    if hasattr(player, "select_move"):
        return SerialVecPlayer(player)
    raise TypeError(f"cannot adapt {player!r} into a vectorized player")


# --------------------------------------------------------------------------- #
# Vectorized match runner                                                     #
# --------------------------------------------------------------------------- #
def play_match_vec(
    player_a,
    player_b,
    *,
    games: int = 100,
    openings: list[str] | None = None,
    max_plies: int = 400,
    parallel: int = 16,
    max_engines: int = 16,
) -> dict:
    """Vectorized :func:`xqai.arena.play_match` (same pairing, scoring, dict).

    ``player_a`` / ``player_b`` may be vec players (:class:`NetVecPlayer`,
    :class:`EngineVecPlayer`, :class:`SerialVecPlayer`) or serial arena players
    (auto-adapted). Up to ``parallel`` games are in flight; finished games are
    immediately replaced from the schedule (a persistent pool -- no lockstep
    barrier on the slowest game).
    """
    vp_a = _as_vec_player(player_a, max_engines)
    vp_b = _as_vec_player(player_b, max_engines)
    if games <= 0:
        return {
            "player_a": getattr(vp_a, "name", "A"),
            "player_b": getattr(vp_b, "name", "B"),
            "games": 0,
            "a_wins": 0,
            "a_draws": 0,
            "a_losses": 0,
            "a_score_rate": 0.0,
            "ci95": (0.0, 0.0),
            "elo_diff": elo_from_winrate(0.0),
        }

    # --- schedule: identical to play_match's pairing ----------------------- #
    n_pairs = (games + 1) // 2
    if openings:
        op_list = [openings[i % len(openings)] for i in range(n_pairs)]
    else:
        op_list = [None] * n_pairs
    schedule: list[tuple[str | None, bool]] = []  # (opening_fen, a_is_red)
    for op in op_list:
        schedule.append((op, True))  # A = red
        schedule.append((op, False))  # A = black (colour-swapped pair)
    schedule = schedule[:games]
    next_game = 0

    width = min(int(parallel), vp_a.max_parallel, vp_b.max_parallel)
    width = max(1, min(int(width), len(schedule)))

    def _start_game() -> _VecGame:
        nonlocal next_game
        op, a_is_red = schedule[next_game]
        next_game += 1
        g = _VecGame(op, a_is_red)
        vp_a.begin_game(g)
        vp_b.begin_game(g)
        return g

    a_wins = a_draws = a_losses = 0
    played = 0
    live: list[_VecGame] = [_start_game() for _ in range(width)]

    def _score(g: _VecGame, res: int) -> None:
        nonlocal a_wins, a_draws, a_losses, played
        played += 1
        if res == _DRAW:
            a_draws += 1
        elif (res == _RED_WIN) == g.a_is_red:
            a_wins += 1
        else:
            a_losses += 1
        vp_a.end_game(g)
        vp_b.end_game(g)

    while live:
        # ---- 1) settle finished games, refill from the schedule ------------ #
        # Order mirrors the serial loop: the ply cap is the loop bound (checked
        # first -- max_plies moves played is a DRAW even if the last move
        # mated), then pos.result(), then a move is requested.
        still: list[_VecGame] = []
        queue: list[_VecGame] = list(live)
        while queue:  # refills are re-checked too (a terminal opening FEN
            g = queue.pop()  # must be settled, not asked for a move)
            if g.result != _ONGOING:  # forfeit decided while applying moves
                res = g.result
            elif g.ply >= max_plies:
                res = _DRAW
            else:
                res = g.pos.result()
            if res != _ONGOING:
                _score(g, res)
                if next_game < len(schedule):
                    queue.append(_start_game())
            else:
                still.append(g)
        live = still
        if not live:
            break

        # ---- 2+3) phased move selection ------------------------------------ #
        # Cheap / concurrent players (engines, random) move FIRST; the games
        # they just advanced are then *merged into the same round's net batch*
        # (eligibility re-checked exactly like the serial pre-move checks).
        # Every mixed net-vs-engine game advances 2 plies per round with ONE
        # net search, so rounds ~= max_game_plies/2 instead of max_game_plies,
        # and the net batch is ~all live games instead of half. (An earlier
        # fire-search-collect overlap was measured strictly worse: it kept the
        # 1-ply-per-round cadence, doubling both searches and engine waits.)
        def _eligible(g: _VecGame) -> bool:
            # Mirrors the serial loop's pre-move checks (order included).
            return (
                g.result == _ONGOING
                and g.ply < max_plies
                and g.pos.result() == _ONGOING
            )

        def _apply(batch: list[_VecGame], moves: list[int]) -> None:
            for g, move in zip(batch, moves):
                legal = g.pos.legal_moves()
                if move not in legal:  # illegal / timeout => mover forfeits
                    g.result = (
                        _BLACK_WIN if g.pos.side_to_move() == 0 else _RED_WIN
                    )
                    continue
                g.pos.push(move)
                g.moves_ucci.append(move_to_ucci(move))
                g.ply += 1

        order: list[_VecPlayerBase] = [vp_a] if vp_a is vp_b else (
            [vp_a, vp_b] if vp_a.phase <= vp_b.phase else [vp_b, vp_a]
        )
        for vp in order:
            batch = []
            for g in live:
                if not _eligible(g):
                    continue
                red_to_move = g.pos.side_to_move() == 0
                if (vp_a if (red_to_move == g.a_is_red) else vp_b) is vp:
                    batch.append(g)
            if batch:
                _apply(batch, vp.batch_select(batch))

    score = a_wins + 0.5 * a_draws
    rate = score / played if played else 0.0
    lo, hi = wilson_interval(score, played)
    return {
        "player_a": getattr(vp_a, "name", "A"),
        "player_b": getattr(vp_b, "name", "B"),
        "games": played,
        "a_wins": a_wins,
        "a_draws": a_draws,
        "a_losses": a_losses,
        "a_score_rate": rate,
        "ci95": (lo, hi),
        "elo_diff": elo_from_winrate(rate),
    }


__all__ = [
    "play_match_vec",
    "NetVecPlayer",
    "EngineVecPlayer",
    "SerialVecPlayer",
    "move_to_ucci",
    "wilson_interval",
    "elo_from_winrate",
]
