"""Smoke tests for xqai.arena_vec (run as a script; needs xqai._xqcore built).

Covers: move<->UCCI roundtrip, pairing/scoring counts, odd ``games``,
``games=0``, max_plies draw, opening FEN injection, and a tiny vec match
(DummyNet vs random) finishing with consistent W/D/L accounting.
"""

from __future__ import annotations

import random

import numpy as np

from xqai.arena import _ucci_to_move
from xqai.arena_vec import (
    NetVecPlayer,
    SerialVecPlayer,
    move_to_ucci,
    play_match_vec,
)
from xqai.dummynet import DummyNet
from xqai.mcts import PUCTPlanner


class _RandomPlayer:
    name = "random"

    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)

    def reset(self):
        pass

    def select_move(self, pos):
        lm = list(pos.legal_moves())
        return int(self.rng.choice(lm)) if lm else -1


def main() -> None:
    # 1) move <-> UCCI roundtrip over random square pairs.
    random.seed(0)
    for _ in range(5000):
        m = random.randrange(90) * 90 + random.randrange(90)
        assert _ucci_to_move(move_to_ucci(m)) == m, (m, move_to_ucci(m))
    print("[1] move<->ucci roundtrip OK (5000 random)")

    from xqai._xqcore import Position

    start_fen = Position().fen()
    print("[2] start fen:", repr(start_fen))

    net = DummyNet()
    planner = PUCTPlanner(add_noise=False)
    a = NetVecPlayer(net, planner, n_sim=8, name="dummy")
    b = SerialVecPlayer(_RandomPlayer(seed=1))

    # 3) counts add up; persistent pool refill works (parallel < games).
    r = play_match_vec(a, b, games=8, openings=None, max_plies=60, parallel=4)
    assert r["games"] == 8, r
    assert r["a_wins"] + r["a_draws"] + r["a_losses"] == 8, r
    print("[3] vec dummy-vs-random:", r)

    # 4) odd games => last opening only played with A as red.
    r1 = play_match_vec(a, b, games=3, openings=None, max_plies=40, parallel=8)
    assert r1["games"] == 3, r1
    print("[4] odd games OK")

    # 5) games=0 edge.
    assert play_match_vec(a, b, games=0)["games"] == 0
    print("[5] games=0 OK")

    # 6) max_plies cap => all draws when nobody can win (max_plies tiny).
    r2 = play_match_vec(a, b, games=4, openings=None, max_plies=4, parallel=4)
    assert r2["games"] == 4 and r2["a_draws"] == 4, r2
    print("[6] max_plies draw cap OK")

    # 7) opening FEN injection: a mate-in-0-ish opening (red hugely ahead)
    #    must influence results; use a FEN reached after a few plies and just
    #    check it is accepted and games complete.
    p = Position()
    for mv in list(p.legal_moves())[:1]:
        p.push(mv)
    op = p.fen()
    r3 = play_match_vec(a, b, games=2, openings=[op], max_plies=30, parallel=2)
    assert r3["games"] == 2, r3
    print("[7] opening FEN injection OK")

    print("smoke arena_vec ALL PASS")


if __name__ == "__main__":
    main()
