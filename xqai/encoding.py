"""Board / move tensor encoding for xqai (INTERFACES.md §1, §2, §4).

Coordinate & piece conventions (§1)
-----------------------------------
- 9 columns x 10 rows = 90 squares. ``sq in [0, 90)``,
  ``row = sq // 9`` (0..9), ``col = sq % 9`` (0..8).
- Red is at the bottom (row 9 side), Black at the top (row 0 side). Red moves
  first. In ``board()``: 0 = empty, positive = red, negative = black; absolute
  value 1..7 = K, A, E, H, R, C, P (将/帅, 士/仕, 象/相, 马, 车, 炮, 兵/卒).

Move encoding (§2)
------------------
- ``move = from_sq * 90 + to_sq``, range ``[0, 8100)``. ``ACTION_DIM = 8100``.
- ``mv_from(m) = m // 90``, ``mv_to(m) = m % 90``.

Perspective / normalization convention (§4)
-------------------------------------------
Everything is encoded **from the perspective of the side to move**, so the
network always "sees" itself at the bottom playing upward:

- When RED is to move, the raw board is used as-is.
- When BLACK is to move, the board is flipped top/bottom (``row -> 9 - row``,
  i.e. ``sq -> (9 - sq//9) * 9 + sq % 9``) **and** colors are swapped, so the
  black pieces land at the bottom and are written into the "current side"
  planes 0..6, while the (now-top) red pieces go into planes 7..13.

Because the planes are in the normalized (current-side) frame, any policy /
mask defined over moves must be in the **same** frame. The raw legal moves from
``pos.legal_moves()`` are in the *original* board frame; for a black-to-move
position they must be passed through :func:`flip_move` to live in the
normalized frame. :func:`legal_mask` does this automatically. To turn a network
move back into a real board move, apply :func:`flip_move` again when black is to
move (``flip_move`` is its own inverse).

Plane layout (15 planes, §4)
----------------------------
- planes 0..6  : current side's K, A, E, H, R, C, P one-hot.
- planes 7..13 : opponent's K, A, E, H, R, C, P one-hot.
- plane 14     : repetition count, normalized as ``repetition_count() / 3``
  (the 3rd repetition is a draw, §3), broadcast over the whole board.
"""

from __future__ import annotations

import numpy as np

# --- contract constants ---------------------------------------------------
ACTION_DIM = 8100
NUM_PLANES = 15

NUM_ROWS = 10
NUM_COLS = 9
NUM_SQUARES = NUM_ROWS * NUM_COLS  # 90

# Side codes mirror xqai._xqcore (RED=0, BLACK=1, §3). Defined locally so this
# module never needs to import the (possibly unbuilt) C++ extension.
RED = 0
BLACK = 1

# Max repetition before a forced draw (§3: 3rd repetition = draw).
_REPETITION_NORM = 3.0


# --- tiny move helpers (§2) ------------------------------------------------
def mv_from(m: int) -> int:
    """Source square of a packed move integer."""
    return m // NUM_SQUARES


def mv_to(m: int) -> int:
    """Destination square of a packed move integer."""
    return m % NUM_SQUARES


def make_move(from_sq: int, to_sq: int) -> int:
    """Pack ``(from_sq, to_sq)`` into a move integer (§2)."""
    return from_sq * NUM_SQUARES + to_sq


def sq_row(sq: int) -> int:
    """Row (0..9) of a square."""
    return sq // NUM_COLS


def sq_col(sq: int) -> int:
    """Column (0..8) of a square."""
    return sq % NUM_COLS


def flip_square(sq: int) -> int:
    """Vertical flip of a square: ``row -> 9 - row`` (col unchanged).

    This is the square transform used when normalizing a black-to-move
    position to the current-side (red-at-bottom) frame. It is its own inverse.
    """
    row, col = divmod(sq, NUM_COLS)
    return (NUM_ROWS - 1 - row) * NUM_COLS + col


def flip_move(m: int) -> int:
    """Vertically flip both squares of a move (§4 perspective helper).

    Maps a move between the original board frame and the normalized
    (current-side) frame. Self-inverse: ``flip_move(flip_move(m)) == m``.
    """
    return make_move(flip_square(mv_from(m)), flip_square(mv_to(m)))


# --- board encoding (§4) ---------------------------------------------------
def _board_to_array(pos) -> np.ndarray:
    """Return ``pos.board()`` as an int8 ``[10, 9]`` array (row-major)."""
    raw = pos.board()  # bytes / buffer of length 90, int8, row-major
    arr = np.frombuffer(bytes(raw), dtype=np.int8)
    if arr.size != NUM_SQUARES:
        raise ValueError(f"board() must have {NUM_SQUARES} entries, got {arr.size}")
    return arr.reshape(NUM_ROWS, NUM_COLS)


def encode(pos) -> np.ndarray:
    """Encode a position into the network input tensor ``[15, 10, 9]`` float32.

    Always normalized to the side-to-move's perspective (see module docstring).
    Planes 0..6 = current side's K,A,E,H,R,C,P; 7..13 = opponent's; 14 =
    ``repetition_count() / 3`` broadcast over the board.
    """
    board = _board_to_array(pos)  # [10, 9] int8, raw frame
    side = pos.side_to_move()

    if side == BLACK:
        # Flip top/bottom so black ends up at the bottom, and swap colors so
        # the black pieces become "positive" (current side).
        board = -board[::-1, :]

    planes = np.zeros((NUM_PLANES, NUM_ROWS, NUM_COLS), dtype=np.float32)

    # Current side: positive values 1..7 -> planes 0..6.
    # Opponent: negative values -1..-7 -> planes 7..13.
    for piece in range(1, 8):
        planes[piece - 1] = (board == piece)
        planes[7 + piece - 1] = (board == -piece)

    rep = float(pos.repetition_count()) / _REPETITION_NORM
    planes[14].fill(rep)

    return planes


def legal_mask(pos) -> np.ndarray:
    """Legal-move mask ``[8100]`` float32 (1 = legal) in the normalized frame.

    ``pos.legal_moves()`` returns moves in the *original* board frame. When
    black is to move the position is encoded from a flipped perspective, so
    each legal move is passed through :func:`flip_move` to match the encoded
    planes / network policy. The resulting mask is therefore always aligned
    with :func:`encode`'s output.
    """
    mask = np.zeros(ACTION_DIM, dtype=np.float32)
    flip = pos.side_to_move() == BLACK
    for m in pos.legal_moves():
        idx = flip_move(m) if flip else m
        mask[idx] = 1.0
    return mask


def policy_target(visit_counts: "dict[int, int]") -> np.ndarray:
    """Build a normalized policy target ``[8100]`` from MCTS visit counts.

    ``visit_counts`` maps move integer -> visit count. The result sums to 1 over
    visited moves (all-zero input yields an all-zero vector). Move integers are
    assumed to already be in the frame the caller wants the target in (the
    self-play pipeline keeps everything in the normalized frame).
    """
    target = np.zeros(ACTION_DIM, dtype=np.float32)
    total = 0
    for move, count in visit_counts.items():
        if count <= 0:
            continue
        target[move] += count
        total += count
    if total > 0:
        target /= float(total)
    return target


__all__ = [
    "ACTION_DIM",
    "NUM_PLANES",
    "NUM_ROWS",
    "NUM_COLS",
    "NUM_SQUARES",
    "RED",
    "BLACK",
    "mv_from",
    "mv_to",
    "make_move",
    "sq_row",
    "sq_col",
    "flip_square",
    "flip_move",
    "encode",
    "legal_mask",
    "policy_target",
]
