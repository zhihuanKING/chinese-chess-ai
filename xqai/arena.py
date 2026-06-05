"""Arena evaluation for xqai (INTERFACES.md §9).

Plays ``N`` games between two *players* and reports win/draw/loss, win rate and
a binomial confidence interval. A **player** is anything that, given a
:class:`xqai._xqcore.Position`, returns a legal move integer (in the **real
board frame**). Three player kinds are provided:

- :class:`NetPlayer`     : a PVNet/DummyNet + a Planner (MCTS).
- :class:`SubprocessUCCIPlayer` : an external engine speaking a UCCI subset
  (also covers the C++ ``xqab`` baseline if it exposes UCCI).

Games are played in **paired openings, colours swapped** (each opening is
played twice, players alternating red/black) to remove first-move bias, exactly
as INTERFACES.md §9 asks.
"""

from __future__ import annotations

import math
import subprocess
import threading
import time
from typing import Protocol

import numpy as np

from .encoding import BLACK, flip_move

_ONGOING = 0
_RED_WIN = 1
_BLACK_WIN = 2
_DRAW = 3


def _new_position():
    from xqai._xqcore import Position
    return Position()


def _position_from_fen(fen: str):
    from xqai._xqcore import Position
    return Position.from_fen(fen)


# --------------------------------------------------------------------------- #
# Player protocol + implementations                                           #
# --------------------------------------------------------------------------- #
class Player(Protocol):
    name: str

    def select_move(self, pos) -> int:
        """Return a legal move integer (real board frame) for ``pos``."""
        ...

    def reset(self) -> None:
        """Reset any per-game internal state (optional)."""
        ...


class NetPlayer:
    """A network + Planner player. Picks the most-visited root move."""

    def __init__(self, net, planner, n_sim: int = 200, name: str = "net"):
        self.net = net
        self.planner = planner
        self.n_sim = int(n_sim)
        self.name = name

    def reset(self) -> None:
        pass

    def select_move(self, pos) -> int:
        # Planner returns improved policy in the normalized frame for [pos].
        pi = self.planner.search([pos], self.net, self.n_sim)[0]
        nz = np.nonzero(pi)[0]
        if nz.size == 0:
            # Fall back to any legal move.
            legal = pos.legal_moves()
            return legal[0] if legal else -1
        move_norm = int(nz[np.argmax(pi[nz])])
        # Convert normalized-frame move to real board frame.
        return flip_move(move_norm) if pos.side_to_move() == BLACK else move_norm


class SubprocessUCCIPlayer:
    """Drives an external engine over a UCCI subset via stdin/stdout.

    Protocol used (UCCI subset): on each move we send::

        position fen <FEN>
        go depth <d>          (or 'go movetime <ms>')

    and read lines until ``bestmove <ucci_move>``. UCCI coordinate moves like
    ``h2e2`` are converted to the project's ``from*90+to`` integer encoding.
    """

    def __init__(self, cmd: list[str], *, depth: int | None = None, movetime_ms: int | None = 1000,
                 name: str = "ucci"):
        self.cmd = cmd
        self.depth = depth
        self.movetime_ms = movetime_ms
        self.name = name
        self.proc: subprocess.Popen | None = None
        self._start()

    def _start(self) -> None:
        self.proc = subprocess.Popen(
            self.cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, text=True, bufsize=1,
        )
        self._send("ucci")
        self._wait_for("ucciok")
        self._send("isready")
        self._wait_for("readyok")

    def reset(self) -> None:
        self._send("ucinewgame")

    def _send(self, line: str) -> None:
        assert self.proc and self.proc.stdin
        self.proc.stdin.write(line + "\n")
        self.proc.stdin.flush()

    def _wait_for(self, token: str, max_lines: int = 1000) -> str:
        assert self.proc and self.proc.stdout
        for _ in range(max_lines):
            line = self.proc.stdout.readline()
            if not line:
                break
            if token in line:
                return line
        return ""

    def select_move(self, pos) -> int:
        self._send(f"position fen {pos.fen()}")
        if self.depth is not None:
            self._send(f"go depth {self.depth}")
        else:
            self._send(f"go movetime {self.movetime_ms}")
        assert self.proc and self.proc.stdout
        # Wall-clock deadline so a hung / silent engine can never block the whole
        # arena forever. Give the engine its think time plus a generous margin;
        # if it blows the deadline we kill it (counts as a forfeit upstream).
        budget_ms = self.movetime_ms if self.movetime_ms else 30_000
        deadline = time.monotonic() + (budget_ms / 1000.0) + 5.0
        watchdog = threading.Timer(max(deadline - time.monotonic(), 0.1), self._kill)
        watchdog.daemon = True
        watchdog.start()
        try:
            for _ in range(100000):
                if time.monotonic() > deadline:
                    return -1
                line = self.proc.stdout.readline()
                if not line:
                    return -1
                if line.startswith("bestmove"):
                    parts = line.split()
                    if len(parts) >= 2:
                        return _ucci_to_move(parts[1])
                    return -1
            return -1
        finally:
            watchdog.cancel()

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
                self.proc.kill()


def _ucci_to_move(ucci: str) -> int:
    """Convert a UCCI coordinate move (e.g. ``h2e2``) to ``from*90+to``.

    UCCI files a..i = columns 0..8 (left->right from red's view); ranks 0..9
    bottom->top. Our board is row 0 = top (black side), row = sq//9. UCCI rank r
    (0=bottom) maps to our row ``9 - r``.
    """
    if len(ucci) < 4:
        return -1
    fc = ord(ucci[0]) - ord("a")
    fr = int(ucci[1])
    tc = ord(ucci[2]) - ord("a")
    tr = int(ucci[3])
    f = (9 - fr) * 9 + fc
    t = (9 - tr) * 9 + tc
    return f * 90 + t


# --------------------------------------------------------------------------- #
# Match runner                                                                #
# --------------------------------------------------------------------------- #
def _play_game(red_player: "Player", black_player: "Player", *, opening_fen: str | None,
               max_plies: int = 400) -> int:
    """Play one game; return result code (RED_WIN / BLACK_WIN / DRAW)."""
    pos = _position_from_fen(opening_fen) if opening_fen else _new_position()
    red_player.reset()
    black_player.reset()
    for _ in range(max_plies):
        res = pos.result()
        if res != _ONGOING:
            return res
        player = red_player if pos.side_to_move() == 0 else black_player
        move = player.select_move(pos)
        legal = pos.legal_moves()
        if move not in legal:
            # Illegal move => the mover forfeits.
            return _BLACK_WIN if pos.side_to_move() == 0 else _RED_WIN
        pos.push(move)
    return _DRAW


def wilson_interval(wins: float, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score binomial CI for a win *rate* (draws counted as 0.5)."""
    if n == 0:
        return (0.0, 0.0)
    p = wins / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - margin), min(1.0, centre + margin))


def elo_from_winrate(winrate: float) -> float:
    """Reverse-engineer an Elo difference from a score rate (logistic model)."""
    winrate = min(max(winrate, 1e-4), 1 - 1e-4)
    return -400.0 * math.log10(1.0 / winrate - 1.0)


def play_match(player_a: "Player", player_b: "Player", *, games: int = 100,
               openings: list[str] | None = None, max_plies: int = 400) -> dict:
    """Play ``games`` games, A vs B, paired openings with colours swapped.

    Returns a stats dict from **A's** perspective: wins/draws/losses, score
    rate (win=1, draw=0.5), Wilson CI, and a logistic Elo estimate.
    """
    # Build the opening list: each opening used for a colour-swapped pair, so we
    # need ceil(games/2) openings. None => standard start position.
    n_pairs = (games + 1) // 2
    if openings:
        op_list = [openings[i % len(openings)] for i in range(n_pairs)]
    else:
        op_list = [None] * n_pairs

    a_wins = a_draws = a_losses = 0
    played = 0
    for op in op_list:
        for swap in (False, True):  # A=red then A=black (paired)
            if played >= games:
                break
            if not swap:
                res = _play_game(player_a, player_b, opening_fen=op, max_plies=max_plies)
                a_is_red = True
            else:
                res = _play_game(player_b, player_a, opening_fen=op, max_plies=max_plies)
                a_is_red = False
            played += 1
            if res == _DRAW:
                a_draws += 1
            elif (res == _RED_WIN) == a_is_red:
                a_wins += 1
            else:
                a_losses += 1

    score = a_wins + 0.5 * a_draws
    rate = score / played if played else 0.0
    lo, hi = wilson_interval(score, played)
    return {
        "player_a": getattr(player_a, "name", "A"),
        "player_b": getattr(player_b, "name", "B"),
        "games": played,
        "a_wins": a_wins,
        "a_draws": a_draws,
        "a_losses": a_losses,
        "a_score_rate": rate,
        "ci95": (lo, hi),
        "elo_diff": elo_from_winrate(rate),
    }


__all__ = [
    "NetPlayer",
    "SubprocessUCCIPlayer",
    "play_match",
    "wilson_interval",
    "elo_from_winrate",
]
