"""Vectorized self-play worker for xqai (INTERFACES.md §6).

Maintains ``B`` parallel :class:`xqai._xqcore.Position` games and drives them all
forward in lockstep using a :class:`~xqai.mcts.Planner` (PUCT or Gumbel). The
planner's ``search`` already batches the ``B`` games' leaf evaluations into a
single ``net`` forward (the GPU-saturating step), so each *move ply* costs one
batched MCTS over all live games.

Per ply:

1. Run ``planner.search(live_positions, net, n_sim)`` -> improved policy ``pi``
   per live game (normalized frame).
2. Sample a move from ``pi`` (temperature 1 for the first ``temp_moves`` plies,
   then argmax / temperature ~0).
3. Record the training sample ``(planes, pi, side_to_move)`` for that ply.
4. Apply the move; check for resignation (value-based) and terminal results.
5. When a game ends, back-fill the outcome ``z`` (from each ply's mover view)
   into its recorded samples and write them to the :class:`ReplayBuffer`.

Everything is kept in the **normalized side-to-move frame** so it lines up with
:func:`encode` / :func:`legal_mask` / the network policy. The sampled move is
converted to the real board frame via :func:`flip_move` when black is to move.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .encoding import BLACK, encode, flip_move

# Result codes mirror xqai._xqcore (§3); defined locally to avoid importing the
# (possibly unbuilt) C++ extension at module load.
_ONGOING = 0
_RED_WIN = 1
_BLACK_WIN = 2
_DRAW = 3


def _new_position():
    """Create a fresh initial Position (imported lazily so import never fails)."""
    from xqai._xqcore import Position  # may raise ImportError if not built
    return Position()


class _GameRecord:
    """Accumulates per-ply samples for one game until it terminates."""

    __slots__ = ("pos", "planes", "pis", "movers", "done", "result")

    def __init__(self, pos):
        self.pos = pos
        self.planes: list[np.ndarray] = []
        self.pis: list[np.ndarray] = []
        self.movers: list[int] = []  # side to move at each recorded ply
        self.done = False
        self.result = _ONGOING


def _outcome_z(result: int, mover: int) -> int:
    """Game outcome from ``mover``'s perspective: +1 win / -1 loss / 0 draw."""
    if result == _DRAW:
        return 0
    if result == _ONGOING:
        return 0
    winner_is_red = result == _RED_WIN
    mover_is_red = mover == 0
    return 1 if winner_is_red == mover_is_red else -1


def _sample_move(pi: np.ndarray, temperature: float, rng: np.random.Generator) -> int:
    """Sample a move index from improved policy ``pi`` at the given temperature.

    ``temperature == 0`` -> argmax (deterministic). Otherwise sample from
    ``pi ** (1/T)`` renormalized. Returns a move index in the normalized frame.
    """
    nz = np.nonzero(pi)[0]
    if nz.size == 0:
        return -1
    if temperature <= 1e-6:
        return int(nz[np.argmax(pi[nz])])
    p = pi[nz].astype(np.float64) ** (1.0 / temperature)
    s = p.sum()
    if s <= 0:
        return int(rng.choice(nz))
    p /= s
    return int(rng.choice(nz, p=p))


class SelfPlayWorker:
    """Runs ``parallel_games`` self-play games in lockstep and fills a buffer.

    Parameters
    ----------
    net : nn.Module
        ``PVNet`` or ``DummyNet`` (same forward signature).
    planner : Planner
        ``PUCTPlanner`` or ``GumbelPlanner``.
    replay : ReplayBuffer | None
        Destination buffer. If None, :meth:`run` returns samples instead.
    parallel_games : int
        Number of games advanced together (batch width for the planner).
    n_sim : int
        Simulations per move.
    temp_moves : int
        Number of opening plies (per game) played at temperature 1.
    resign_threshold : float
        If a game's root value drops below this (from the mover's view) the
        mover resigns. ``None`` disables resignation.
    max_plies : int
        Hard cap on plies per game (safety; rules also enforce a draw cap).
    """

    def __init__(
        self,
        net,
        planner,
        replay=None,
        parallel_games: int = 8,
        n_sim: int = 64,
        temp_moves: int = 30,
        resign_threshold: float | None = -0.95,
        max_plies: int = 400,
        seed: int | None = None,
    ):
        self.net = net
        self.planner = planner
        self.replay = replay
        self.parallel_games = int(parallel_games)
        self.n_sim = int(n_sim)
        self.temp_moves = int(temp_moves)
        self.resign_threshold = resign_threshold
        self.max_plies = int(max_plies)
        self._rng = np.random.default_rng(seed)

    def run(self, num_games: int | None = None) -> dict[str, Any]:
        """Play games until ``num_games`` finish (default: one full batch).

        Returns a stats dict; samples are written to ``self.replay`` if set,
        otherwise collected and returned under ``"samples"``.
        """
        target = self.parallel_games if num_games is None else int(num_games)

        games = [_GameRecord(_new_position()) for _ in range(self.parallel_games)]
        finished = 0
        ply = 0
        collected: list[tuple[np.ndarray, np.ndarray, int]] = []
        total_samples = 0

        while finished < target and ply < self.max_plies:
            live = [g for g in games if not g.done]
            if not live:
                break

            live_positions = [g.pos for g in live]
            # === BATCHED MCTS over all live games (single net forward inside). ===
            pis = self.planner.search(live_positions, self.net, self.n_sim)

            temperature = 1.0 if ply < self.temp_moves else 0.0
            for g, pi in zip(live, pis):
                pos = g.pos
                # Record the training sample for this ply (normalized frame).
                g.planes.append(encode(pos).astype(np.float16))
                g.pis.append(pi.astype(np.float16))
                g.movers.append(pos.side_to_move())

                move_norm = _sample_move(pi, temperature, self._rng)
                if move_norm < 0:
                    # No legal move available -> mover loses (mate/stalemate).
                    g.result = _BLACK_WIN if pos.side_to_move() == 0 else _RED_WIN
                    self._finish_game(g, collected)
                    finished += 1
                    continue

                # Optional resignation based on the root value estimate.
                if self.resign_threshold is not None:
                    val = self._root_value(pi, pos)
                    if val is not None and val < self.resign_threshold:
                        # Mover resigns -> opponent wins.
                        g.result = _BLACK_WIN if pos.side_to_move() == 0 else _RED_WIN
                        self._finish_game(g, collected)
                        finished += 1
                        continue

                real_move = (
                    flip_move(move_norm) if pos.side_to_move() == BLACK else move_norm
                )
                pos.push(real_move)

                res = pos.result()
                if res != _ONGOING:
                    g.result = res
                    self._finish_game(g, collected)
                    finished += 1

            ply += 1

        # Any games still ongoing at max_plies are scored as draws.
        for g in games:
            if not g.done and g.planes:
                g.result = _DRAW
                self._finish_game(g, collected)

        # Flush collected samples to the replay buffer if present.
        if self.replay is not None and collected:
            planes = np.stack([c[0] for c in collected])
            pis = np.stack([c[1] for c in collected])
            zs = np.array([c[2] for c in collected], dtype=np.int8)
            self.replay.add_batch(planes, pis, zs)
            total_samples = len(collected)
            collected = []

        stats = {
            "games_finished": finished,
            "plies": ply,
            "samples": total_samples if self.replay is not None else collected,
            "num_samples": (total_samples if self.replay is not None else len(collected)),
        }
        return stats

    # -- internals ---------------------------------------------------------- #
    def _root_value(self, pi: np.ndarray, pos) -> float | None:
        """Cheap value proxy for resignation.

        The planner doesn't expose the root value through ``search``; we run a
        single batched net forward on this position to read its value head. This
        is one extra forward per resignation check (kept simple).
        """
        try:
            import torch
            from .mcts import _net_device

            device = _net_device(self.net)
            x = torch.from_numpy(encode(pos)[None]).to(device, torch.float32)
            was_training = self.net.training
            self.net.eval()
            with torch.no_grad():
                _, value = self.net(x)
            if was_training:
                self.net.train()
            return float(value.reshape(-1)[0].cpu())
        except Exception:
            return None

    def _finish_game(self, g: _GameRecord, collected: list) -> None:
        """Back-fill outcome z into the game's recorded samples."""
        g.done = True
        for planes, pi, mover in zip(g.planes, g.pis, g.movers):
            z = _outcome_z(g.result, mover)
            collected.append((planes, pi, z))


__all__ = ["SelfPlayWorker"]
