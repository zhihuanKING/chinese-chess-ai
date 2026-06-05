"""Smoke test for :mod:`xqai.encoding` perspective normalization (§1, §2, §4).

No ``_xqcore`` dependency: uses a minimal ``MockPosition`` implementing the
subset of the §3 ``Position`` interface that :mod:`xqai.encoding` consumes
(``board()`` / ``legal_moves()`` / ``side_to_move()`` / ``repetition_count()``).

The point of this test is the *black-to-move perspective consistency*: when
black is to move, :func:`encode` flips the board top/bottom and swaps colors so
the black pieces land in the "current side" planes 0..6 at the bottom, and
:func:`legal_mask` must put its set bits in the **same** (flipped) frame. We
assert both views agree square-for-square.

Run with::

    python -m xqai._smoke_encoding
"""

from __future__ import annotations

import numpy as np

from xqai.encoding import (
    BLACK,
    NUM_COLS,
    NUM_SQUARES,
    RED,
    encode,
    flip_move,
    flip_square,
    legal_mask,
    make_move,
    mv_from,
    mv_to,
    policy_target,
)


class MockPosition:
    """Minimal stand-in for the C++ ``Position`` (§3) for encoding tests.

    ``board`` is a length-90 row-major int8 sequence (§1 convention: positive =
    red, negative = black). ``moves`` are packed move integers in the *raw*
    board frame (§2). ``stm`` is RED/BLACK. ``rep`` is the repetition count.
    """

    def __init__(self, board, moves, stm, rep=0):
        self._board = np.asarray(board, dtype=np.int8)
        assert self._board.size == NUM_SQUARES
        self._moves = list(moves)
        self._stm = int(stm)
        self._rep = int(rep)

    def board(self) -> bytes:
        return self._board.tobytes()

    def legal_moves(self):
        return list(self._moves)

    def side_to_move(self) -> int:
        return self._stm

    def repetition_count(self) -> int:
        return self._rep


def _sq(row: int, col: int) -> int:
    return row * NUM_COLS + col


def test_flip_self_inverse() -> None:
    for sq in range(NUM_SQUARES):
        assert flip_square(flip_square(sq)) == sq, sq
    # A few moves end to end.
    for m in (0, 1, 90, 8099, make_move(_sq(0, 0), _sq(9, 8)), make_move(_sq(2, 4), _sq(7, 1))):
        assert flip_move(flip_move(m)) == m, m
    print("flip_square / flip_move are self-inverse over all squares: OK")


def test_red_identity() -> None:
    """Red to move: encode uses the board as-is, mask uses raw moves."""
    board = np.zeros(NUM_SQUARES, dtype=np.int8)
    # Red chariot (R=5) at bottom-left corner row 9, col 0.
    rk = _sq(9, 0)
    board[rk] = 5
    # Black chariot at top row 0, col 0.
    bk = _sq(0, 0)
    board[bk] = -5
    move = make_move(rk, _sq(8, 0))  # red chariot forward one row
    pos = MockPosition(board, [move], RED)

    planes = encode(pos)
    # Red chariot is current side -> plane 4 (R = index 5-1).
    assert planes[4, 9, 0] == 1.0
    # Black chariot is opponent -> plane 7+4 = 11.
    assert planes[11, 0, 0] == 1.0

    mask = legal_mask(pos)
    assert mask.shape == (8100,)
    assert mask.dtype == np.float32
    assert mask.sum() == 1.0
    assert mask[move] == 1.0  # red: no flip
    print("red-to-move identity (no flip) consistency: OK")


def test_black_perspective_consistency() -> None:
    """Black to move: encode flips+swaps, legal_mask must flip too.

    Concrete position (raw board frame):
      - black chariot (-5) at row 0, col 0 (top-left).
      - black cannon  (-6) at row 2, col 4.
      - red king (1) at row 9, col 4 (bottom).
    Black is to move. Two legal moves for black pieces (raw frame).
    """
    board = np.zeros(NUM_SQUARES, dtype=np.int8)
    b_rook = _sq(0, 0)
    b_cannon = _sq(2, 4)
    r_king = _sq(9, 4)
    board[b_rook] = -5
    board[b_cannon] = -6
    board[r_king] = 1

    raw_moves = [
        make_move(b_rook, _sq(0, 3)),    # black chariot slides right
        make_move(b_cannon, _sq(5, 4)),  # black cannon advances
    ]
    pos = MockPosition(board, raw_moves, BLACK)

    planes = encode(pos)

    # --- 1. Black pieces must be the CURRENT side (planes 0..6) at the BOTTOM.
    # Black chariot raw (0,0) -> flipped row 9 -> plane 4 (R) at (9,0).
    assert planes[4, 9, 0] == 1.0, "black chariot should be current-side R at bottom"
    # Black cannon raw (2,4) -> flipped row 7 -> plane 5 (C) at (7,4).
    assert planes[5, 7, 4] == 1.0, "black cannon should be current-side C, row flipped"
    # Red king raw (9,4) -> opponent, flipped row 0 -> plane 7 (K) at (0,4).
    assert planes[7, 0, 4] == 1.0, "red king should be opponent K, flipped to top"
    # Sanity: current-side planes only carry the (flipped) black pieces.
    assert planes[0:7].sum() == 2.0
    assert planes[7:14].sum() == 1.0

    # --- 2. legal_mask must place set bits in the SAME flipped frame.
    mask = legal_mask(pos)
    assert mask.shape == (8100,)
    assert mask.dtype == np.float32
    assert mask.sum() == len(raw_moves)

    set_idx = np.flatnonzero(mask)
    for idx in set_idx:
        f, t = mv_from(int(idx)), mv_to(int(idx))
        # Decoded from-square must coincide with a current-side (black) piece
        # in the encoded planes -- i.e. one of planes 0..6 is set there.
        fr, fc = divmod(f, NUM_COLS)
        assert planes[0:7, fr, fc].sum() == 1.0, (
            f"mask from-square {f} (row {fr},col {fc}) is not on a current-side "
            "piece in the encoded planes -- perspective mismatch!"
        )

    # --- 3. Each set mask index equals flip_move(raw_move): same frame, exactly.
    expected = {flip_move(m) for m in raw_moves}
    assert set(int(i) for i in set_idx) == expected, (set(set_idx), expected)

    # --- 4. flip_move back recovers the raw board move (round trip).
    for idx in set_idx:
        raw = flip_move(int(idx))
        assert raw in raw_moves, raw

    print("black-to-move encode/legal_mask perspective consistency: OK")


def test_policy_target() -> None:
    # Empty visit counts -> all-zero, no NaN.
    t = policy_target({})
    assert t.shape == (8100,)
    assert t.dtype == np.float32
    assert np.all(t == 0.0)
    assert not np.isnan(t).any()
    # Normalized.
    t = policy_target({10: 3, 20: 1})
    assert abs(t.sum() - 1.0) < 1e-6
    assert abs(t[10] - 0.75) < 1e-6 and abs(t[20] - 0.25) < 1e-6
    # Non-positive counts ignored.
    t = policy_target({5: 0, 6: -4})
    assert np.all(t == 0.0)
    print("policy_target normalization / empty-safe: OK")


def main() -> None:
    test_flip_self_inverse()
    test_red_identity()
    test_black_perspective_consistency()
    test_policy_target()
    print("\nENCODING SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
