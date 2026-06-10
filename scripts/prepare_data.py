#!/usr/bin/env python3
"""Supervised-pretraining data pipeline for the xqai Chinese-Chess project.

This module turns human game records (ICCS coordinate notation or WXF/Chinese
notation) into supervised training samples, following the cross-language
contract in ``INTERFACES.md`` (§1 coords/pieces, §2 move=from*90+to, §3
``Position`` API, §4 tensor encoding).

It deliberately depends on **only the standard library** for the parsing /
round-trip layer (so it runs under ``/usr/bin/python3`` with no third-party
packages). The tensor-encoding / sample-writing layer additionally needs
``xqai._xqcore`` (the C++ rules kernel, §3), ``xqai.encoding`` (§4) and
``numpy``; all of those are imported lazily inside ``try/except`` so the parser
and its tests work even when the extension is not built.

=====================================================================
Coordinate conventions (must match INTERFACES.md §1)
=====================================================================
- 9 columns x 10 rows = 90 squares. ``sq in [0, 90)``,
  ``row = sq // 9`` (0..9), ``col = sq % 9`` (0..8).
- **Red is at the bottom (row 9 side), Black at the top (row 0 side).**
  Red moves first.
- ``move = from_sq * 90 + to_sq`` (§2), with ``mv_from = move // 90`` and
  ``mv_to = move % 90``.

ICCS coordinate notation
------------------------
ICCS writes a move as ``<file><rank><file><rank>`` (e.g. ``h2e2``):
- *file*  is a letter ``a..i`` numbering columns **from Red's left to right**
  (a=0 .. i=8). Our ``col`` uses the same left-to-right orientation, so
  ``col = ord(file) - ord('a')`` directly (a..i -> 0..8).
- *rank*  is a digit ``0..9`` numbering rows **from the Red bottom edge upward**
  (rank 0 = Red's back rank, rank 9 = Black's back rank). Our ``row`` is the
  opposite (row 0 = top / Black, row 9 = bottom / Red), so we must invert:
  ``row = 9 - rank``.
  Therefore ``sq = (9 - rank) * 9 + (file - 'a')``.

  Sanity check: ICCS ``e0`` is the Red king's start (centre file, Red back
  rank) -> file e=4, rank 0 -> ``row = 9, col = 4`` -> ``sq = 9*9+4 = 85``,
  which is indeed the centre of the bottom (Red) edge. Good.

WXF / Chinese notation
----------------------
Four characters ``<piece><file><action><target>`` describing one move:
- Red pieces use Chinese piece names (帅/仕/相/马/车/炮/兵 and the traditional
  将/士/象/馬/車/砲/卒 variants) and Chinese *file* numerals (一..九), counted
  **from Red's right to left** (so Red file 一 is Red's rightmost column).
- Black pieces conventionally also use Chinese names but Arabic *file* digits
  (1..9), counted **from Black's right to left**. (Many corpora use Chinese
  numerals for both sides; we accept Chinese OR Arabic digits for either side
  and decide the side by piece-name table + an optional explicit prefix.)
- action 进 (forward, toward the enemy), 退 (backward), 平 (sideways).
- ``前/后`` (and 中, 二/三...) disambiguate when one side has several identical
  pieces on the same file (e.g. ``前炮进二``).

This parser aims to be *as correct as practical*; ambiguous / rare structures
are marked ``TODO`` in the relevant function docstrings.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

# --------------------------------------------------------------------------
# Contract constants (kept local; do NOT import xqai.encoding at module load
# time so the parser works without numpy / the C++ extension). The §4 module
# re-exports the same values.
# --------------------------------------------------------------------------
NUM_ROWS = 10
NUM_COLS = 9
NUM_SQUARES = NUM_ROWS * NUM_COLS  # 90
ACTION_DIM = 8100

RED = 0
BLACK = 1

# Piece codes (abs value), §1: 1..7 = K,A,E,H,R,C,P.
K, A, E, H, R, C, P = 1, 2, 3, 4, 5, 6, 7


# --------------------------------------------------------------------------
# tiny move helpers (§2) -- duplicated from xqai.encoding so this file has no
# import-time dependency on it.
# --------------------------------------------------------------------------
def make_move(from_sq: int, to_sq: int) -> int:
    """Pack ``(from_sq, to_sq)`` into a move integer (§2)."""
    return from_sq * NUM_SQUARES + to_sq


def mv_from(m: int) -> int:
    return m // NUM_SQUARES


def mv_to(m: int) -> int:
    return m % NUM_SQUARES


def rc_to_sq(row: int, col: int) -> int:
    """``(row, col) -> sq`` with ``sq = row*9 + col`` (§1)."""
    return row * NUM_COLS + col


def sq_to_rc(sq: int) -> Tuple[int, int]:
    """``sq -> (row, col)`` (§1)."""
    return divmod(sq, NUM_COLS)


# ==========================================================================
# 1a. ICCS coordinate notation
# ==========================================================================
_ICCS_TOKEN_RE = re.compile(r"^[a-iA-I][0-9][a-iA-I][0-9]$")


def iccs_coord_to_sq(file_ch: str, rank_ch: str) -> int:
    """Convert one ICCS ``<file><rank>`` pair to a square ``sq`` (§1).

    ``col = file - 'a'`` (a..i = 0..8, Red's left to right);
    ``row = 9 - rank`` (rank 0 = Red bottom edge, our row 9).
    """
    col = ord(file_ch.lower()) - ord("a")
    if not 0 <= col < NUM_COLS:
        raise ValueError(f"ICCS file out of range: {file_ch!r}")
    rank = int(rank_ch)
    row = (NUM_ROWS - 1) - rank
    return rc_to_sq(row, col)


def sq_to_iccs(sq: int) -> str:
    """Inverse of :func:`iccs_coord_to_sq` for a single square -> ``<file><rank>``."""
    row, col = sq_to_rc(sq)
    file_ch = chr(ord("a") + col)
    rank = (NUM_ROWS - 1) - row
    return f"{file_ch}{rank}"


def parse_iccs_move(token: str) -> int:
    """Parse a single ICCS move token (e.g. ``h2e2``) into a move integer (§2)."""
    token = token.strip()
    if not _ICCS_TOKEN_RE.match(token):
        raise ValueError(f"not a valid ICCS move token: {token!r}")
    from_sq = iccs_coord_to_sq(token[0], token[1])
    to_sq = iccs_coord_to_sq(token[2], token[3])
    return make_move(from_sq, to_sq)


def move_to_iccs(move: int) -> str:
    """Inverse of :func:`parse_iccs_move` (move integer -> ``<from><to>`` token)."""
    return sq_to_iccs(mv_from(move)) + sq_to_iccs(mv_to(move))


def is_iccs_move(token: str) -> bool:
    return bool(_ICCS_TOKEN_RE.match(token.strip()))


# ==========================================================================
# 1b. WXF / Chinese notation
# ==========================================================================
# Piece-name tables. Both traditional and simplified / red & black variants
# map to the same abs piece code (§1). We also track which names are
# *canonically* red vs black to help auto-detect the side when no explicit
# prefix is given, but the parser does not strictly require it.
_PIECE_NAMES: Dict[str, int] = {
    # King / 将帅
    "帅": K, "將": K, "将": K,
    # Advisor / 士仕
    "仕": A, "士": A,
    # Elephant / 相象
    "相": E, "象": E,
    # Horse / 马馬
    "马": H, "馬": H,
    # Chariot / 车車
    "车": R, "車": R,
    # Cannon / 炮砲
    "炮": C, "砲": C, "炰": C,
    # Pawn / 兵卒
    "兵": P, "卒": P,
}

# Names that are conventionally the RED side's spelling (used only as a hint).
_RED_PIECE_NAMES = {"帅", "仕", "相", "马", "车", "炮", "兵"}
_BLACK_PIECE_NAMES = {"將", "将", "士", "象", "馬", "車", "砲", "卒"}

# Chinese numerals 一..九 -> 1..9 (used for files, and for distances).
_CN_DIGITS: Dict[str, int] = {
    "一": 1, "二": 2, "三": 3, "四": 4, "五": 5,
    "六": 6, "七": 7, "八": 8, "九": 9,
}
_ARABIC_DIGITS = {str(d): d for d in range(1, 10)}

_ACTION_FWD = "进進"   # forward
_ACTION_BACK = "退"    # backward
_ACTION_FLAT = "平"    # sideways

_PREFIX_FRONT = "前"
_PREFIX_REAR = "后後"
_PREFIX_MID = "中"

# Pieces that move along a "slanted" / fixed-shape path so that a 进/退 move
# also changes file -> the third char (target) is a destination *file*, not a
# distance. (Horse, Advisor, Elephant.)
_DIAGONAL_PIECES = {H, A, E}


def _digit_value(ch: str) -> Optional[int]:
    if ch in _CN_DIGITS:
        return _CN_DIGITS[ch]
    if ch in _ARABIC_DIGITS:
        return _ARABIC_DIGITS[ch]
    return None


def _is_chinese_digit(ch: str) -> bool:
    return ch in _CN_DIGITS


def _looks_like_wxf(token: str) -> bool:
    """Heuristic: does this token look like a Chinese/WXF move?"""
    token = token.strip()
    if len(token) != 4:
        # Some movers like 前/后 prefix still keep 4 chars; >4 unusual.
        return False
    # Must contain at least one known piece name OR a front/rear prefix, and a
    # known action char.
    has_action = any(c in (_ACTION_FWD + _ACTION_BACK + _ACTION_FLAT) for c in token)
    has_piece = any(c in _PIECE_NAMES for c in token)
    has_prefix = any(c in (_PREFIX_FRONT + _PREFIX_REAR + _PREFIX_MID) for c in token)
    return has_action and (has_piece or has_prefix)


@dataclass
class _BoardState:
    """A minimal mailbox board used only to *resolve* WXF moves into squares.

    ``cells[sq]`` is the signed piece code (§1: positive=red, negative=black).
    This is **not** a rules engine -- it just tracks where the pieces are so a
    file-relative WXF move can be turned into ``(from_sq, to_sq)``. Legality is
    validated later by the real ``Position`` (§3).
    """

    cells: List[int] = field(default_factory=lambda: [0] * NUM_SQUARES)

    @classmethod
    def initial(cls) -> "_BoardState":
        b = cls()
        # Black (top, rows 0..) negative; Red (bottom, rows 9..) positive.
        back_black = [-R, -H, -E, -A, -K, -A, -E, -H, -R]
        for col, pc in enumerate(back_black):
            b.cells[rc_to_sq(0, col)] = pc
        b.cells[rc_to_sq(2, 1)] = -C
        b.cells[rc_to_sq(2, 7)] = -C
        for col in (0, 2, 4, 6, 8):
            b.cells[rc_to_sq(3, col)] = -P
        back_red = [R, H, E, A, K, A, E, H, R]
        for col, pc in enumerate(back_red):
            b.cells[rc_to_sq(9, col)] = pc
        b.cells[rc_to_sq(7, 1)] = C
        b.cells[rc_to_sq(7, 7)] = C
        for col in (0, 2, 4, 6, 8):
            b.cells[rc_to_sq(6, col)] = P
        return b

    def apply(self, from_sq: int, to_sq: int) -> None:
        self.cells[to_sq] = self.cells[from_sq]
        self.cells[from_sq] = 0


class WXFParseError(ValueError):
    """Raised when a Chinese/WXF token cannot be resolved on the given board."""


def _wxf_file_to_col(file_value: int, side: int) -> int:
    """Map a WXF *file number* (1..9) to a board ``col`` (0..8).

    Files are counted from each side's **own right** toward its left:
    - Red counts from Red's right. Red sits at the bottom looking up; Red's
      right is our col 8, so Red file ``f`` -> ``col = 9 - f``.
    - Black counts from Black's right. Black sits at the top looking down;
      Black's right is our col 0, so Black file ``f`` -> ``col = f - 1``.
    """
    if side == RED:
        return NUM_COLS - file_value
    return file_value - 1


def _forward_sign(side: int) -> int:
    """Row delta for one step *forward* (toward the enemy) for ``side``.

    Red is at the bottom (large rows) moving up -> rows decrease -> -1.
    Black is at the top (small rows) moving down -> rows increase -> +1.
    """
    return -1 if side == RED else +1


def parse_wxf_move(token: str, board: "_BoardState", side: int) -> int:
    """Resolve one Chinese/WXF move ``token`` to a move integer (§2).

    ``board`` is the current :class:`_BoardState` (mutated by the caller after
    a successful parse); ``side`` is RED/BLACK whose turn it is. Returns the
    packed move integer. Raises :class:`WXFParseError` on anything it cannot
    resolve on the given board.

    Supported: 进/退/平, Chinese-or-Arabic file numerals, and the 前/后/中
    disambiguation prefix for stacked identical pieces on one file.

    TODO (边界 / not yet handled):
      * Three-or-more identical pieces on one file addressed as 前/中/后 where
        more than 3 stack (rare; only 前/中/后 + the "前/后 of a tandem" common
        cases are covered).
      * The "兵/卒 + 前/后" pawn ordering when several pawns share a file and
        are addressed by ordinal numerals (一/二/三...) rather than 前/后 — the
        common 前/后 form is handled; numeric ordinals over pawns are TODO.
      * Some engines write the 进/退 *distance* for a slanted horse move as the
        destination file using the *opponent* numeral system — we assume the
        moving side's numeral system throughout.
    """
    token = token.strip()
    if len(token) != 4:
        raise WXFParseError(f"WXF token must be 4 chars: {token!r}")

    c0, c1, c2, c3 = token[0], token[1], token[2], token[3]

    fsign = _forward_sign(side)
    sign = 1 if side == RED else -1  # board cell sign for this side

    # ---- locate the source piece ----------------------------------------
    prefix = None
    if c0 in _PREFIX_FRONT:
        prefix = "front"
        piece = _PIECE_NAMES.get(c1)
        if piece is None:
            raise WXFParseError(f"unknown piece after prefix: {token!r}")
        # file numeral is absent in the 前X... form; column comes from the
        # stacked pair's column.
        file_value = None
    elif c0 in _PREFIX_REAR:
        prefix = "rear"
        piece = _PIECE_NAMES.get(c1)
        if piece is None:
            raise WXFParseError(f"unknown piece after prefix: {token!r}")
        file_value = None
    elif c0 in _PREFIX_MID:
        prefix = "mid"
        piece = _PIECE_NAMES.get(c1)
        if piece is None:
            raise WXFParseError(f"unknown piece after prefix: {token!r}")
        file_value = None
    else:
        piece = _PIECE_NAMES.get(c0)
        if piece is None:
            raise WXFParseError(f"unknown piece name: {token!r}")
        file_value = _digit_value(c1)
        if file_value is None:
            raise WXFParseError(f"bad source file numeral: {token!r}")

    signed_piece = sign * piece

    if prefix is None:
        col = _wxf_file_to_col(file_value, side)
        rows_with_piece = [
            r for r in range(NUM_ROWS) if board.cells[rc_to_sq(r, col)] == signed_piece
        ]
        if not rows_with_piece:
            raise WXFParseError(f"no {token[0]} on resolved file (col={col}): {token!r}")
        if len(rows_with_piece) > 1:
            # Ambiguous without a prefix. Conventionally this should not happen
            # for legal records, but be defensive: pick the one closest to the
            # mover's back rank is undefined, so reject.
            raise WXFParseError(
                f"ambiguous file for {token!r}: pieces on rows {rows_with_piece}"
            )
        from_row = rows_with_piece[0]
    else:
        # Prefix form: find the file (column) that has >=2 of this piece, then
        # choose front/mid/rear along the mover's forward direction.
        cols_with_stack = {}
        for cc in range(NUM_COLS):
            rows = [
                r for r in range(NUM_ROWS)
                if board.cells[rc_to_sq(r, cc)] == signed_piece
            ]
            if len(rows) >= 2:
                cols_with_stack[cc] = rows
        if not cols_with_stack:
            raise WXFParseError(f"no stacked {token[1]} for prefix move: {token!r}")
        if len(cols_with_stack) > 1:
            raise WXFParseError(
                f"multiple files stack {token[1]}; ambiguous prefix: {token!r}"
            )
        col, rows = next(iter(cols_with_stack.items()))
        # "front" = closer to the enemy = further forward along fsign.
        # forward means decreasing row for red, increasing for black, so the
        # most-forward piece is min row for red / max row for black. Sorting by
        # ``-r*fsign`` ascending puts the most-forward piece first for both.
        rows_sorted_forward = sorted(rows, key=lambda r: -r * fsign)
        # rows_sorted_forward[0] is the most-forward piece.
        if prefix == "front":
            from_row = rows_sorted_forward[0]
        elif prefix == "rear":
            from_row = rows_sorted_forward[-1]
        else:  # mid
            if len(rows_sorted_forward) < 3:
                raise WXFParseError(f"中 needs 3 stacked pieces: {token!r}")
            from_row = rows_sorted_forward[len(rows_sorted_forward) // 2]

    from_sq = rc_to_sq(from_row, col)

    # ---- action + target -------------------------------------------------
    action = c2
    target_value = _digit_value(c3)
    if action in _ACTION_FLAT:
        # sideways: target is a destination file, same row.
        if target_value is None:
            raise WXFParseError(f"bad target file for 平: {token!r}")
        to_col = _wxf_file_to_col(target_value, side)
        to_row = from_row
        to_sq = rc_to_sq(to_row, to_col)
    elif action in _ACTION_FWD or action in _ACTION_BACK:
        forward = action in _ACTION_FWD
        step_dir = fsign if forward else -fsign
        if piece in _DIAGONAL_PIECES:
            # Slanted movers (H/A/E): target is the destination *file*.
            if target_value is None:
                raise WXFParseError(f"bad target file for slanted move: {token!r}")
            to_col = _wxf_file_to_col(target_value, side)
            dcol = abs(to_col - col)
            to_row = _slanted_to_row(piece, from_row, step_dir, dcol, token)
            to_sq = rc_to_sq(to_row, to_col)
        else:
            # Straight movers (K/R/C/P): target is a distance (number of rows).
            if target_value is None:
                raise WXFParseError(f"bad distance for 进/退: {token!r}")
            to_row = from_row + step_dir * target_value
            to_col = col
            to_sq = rc_to_sq(to_row, to_col)
    else:
        raise WXFParseError(f"unknown action char {action!r} in {token!r}")

    # Validate against the board geometry. ``to_col`` already comes from a
    # 1..9 file numeral (so it is in [0,8]); the only way a move can leave the
    # board is a row over/underflow, which we check explicitly via ``to_row``
    # rather than relying on the packed ``to_sq`` (a negative row times 9 plus a
    # column happens to stay negative here, but checking the row is unambiguous).
    if not (0 <= to_row < NUM_ROWS) or not (0 <= to_col < NUM_COLS):
        raise WXFParseError(f"resolved move off-board: {token!r}")
    if not (0 <= to_sq < NUM_SQUARES):
        raise WXFParseError(f"resolved move off-board: {token!r}")

    return make_move(from_sq, to_sq)


def _slanted_to_row(piece: int, from_row: int, step_dir: int, dcol: int, token: str) -> int:
    """Destination row for a slanted (H/A/E) forward/backward move.

    - Horse (马): an L-shape, total row delta is 2 if it moves 1 file, or 1 if
      it moves 2 files. So ``|drow| = 3 - dcol`` (dcol in {1,2}).
    - Advisor (士/仕): always moves exactly 1 row and 1 file -> ``|drow| = 1``.
    - Elephant (相/象): always moves exactly 2 rows and 2 files -> ``|drow| = 2``.
    """
    if piece == H:
        if dcol == 1:
            drow = 2
        elif dcol == 2:
            drow = 1
        else:
            raise WXFParseError(f"illegal horse file delta {dcol}: {token!r}")
    elif piece == A:
        drow = 1
    elif piece == E:
        drow = 2
    else:  # pragma: no cover - guarded by caller
        raise WXFParseError(f"not a slanted piece: {token!r}")
    return from_row + step_dir * drow


# ==========================================================================
# 1c. Notation auto-detection + full-game parsing
# ==========================================================================
def detect_notation(token: str) -> str:
    """Return ``"iccs"``, ``"wxf"`` or ``"unknown"`` for a single move token."""
    token = token.strip()
    if not token:
        return "unknown"
    if is_iccs_move(token):
        return "iccs"
    if _looks_like_wxf(token):
        return "wxf"
    return "unknown"


# Tokens that are clearly not moves (result markers, move numbers, comments).
_RESULT_TOKENS = {
    "1-0", "0-1", "1/2-1/2", "*",
    "红胜", "黑胜", "和棋", "和局", "胜", "负", "平局",
}
_MOVE_NUMBER_RE = re.compile(r"^\d+[.．、]?$")


def tokenize_movetext(text: str) -> List[str]:
    """Split a line/block of move text into candidate move tokens.

    Strips move numbers (``1.``), result markers, brace/paren comments and
    punctuation. Keeps both ICCS tokens and Chinese 4-char groups.
    """
    # remove {...} and (...) comments
    text = re.sub(r"\{[^}]*\}", " ", text)
    text = re.sub(r"\([^)]*\)", " ", text)
    # normalize separators
    text = text.replace("\t", " ")
    raw = re.split(r"[\s,;]+", text)
    tokens: List[str] = []
    for t in raw:
        t = t.strip()
        if not t:
            continue
        if t in _RESULT_TOKENS:
            continue
        if _MOVE_NUMBER_RE.match(t):
            continue
        # a leading move number glued to an ICCS token e.g. "1.h2e2"
        m = re.match(r"^\d+[.．、](.+)$", t)
        if m:
            t = m.group(1)
        # Chinese movetext often has no spaces; a 4-char Chinese group is one
        # move. ICCS is always 4 ASCII chars. If the token is a run of CJK,
        # chop it into 4-char moves.
        if _is_cjk_run(t):
            for i in range(0, len(t) - len(t) % 4, 4):
                tokens.append(t[i:i + 4])
            # a trailing remainder shorter than 4 is dropped (likely noise)
        else:
            tokens.append(t)
    return tokens


def _is_cjk_run(s: str) -> bool:
    """A run of glued WXF moves: contains real CJK and every char is a legal
    WXF char (CJK ideograph, Chinese numeral, or Arabic digit for black files).
    Used to chop space-less Chinese movetext into 4-char moves.
    """
    if len(s) < 4:
        return False
    has_cjk = any(_is_cjk(ch) for ch in s)
    all_wxf = all(_is_cjk(ch) or ch in _ARABIC_DIGITS for ch in s)
    return has_cjk and all_wxf


def _is_cjk(ch: str) -> bool:
    o = ord(ch)
    return 0x3400 <= o <= 0x9FFF or ch in _CN_DIGITS


@dataclass
class ParsedGame:
    """A parsed game: a list of move integers (§2) plus metadata."""

    moves: List[int] = field(default_factory=list)
    notation: str = "unknown"
    result: Optional[int] = None  # RED_WIN=+1, BLACK_WIN=-1, DRAW=0 (z convention)
    raw_tokens: List[str] = field(default_factory=list)
    error: Optional[str] = None


# z / result encoding for samples (from RED's absolute perspective):
RESULT_RED_WIN = 1
RESULT_BLACK_WIN = -1
RESULT_DRAW = 0

_RESULT_MAP = {
    "1-0": RESULT_RED_WIN, "红胜": RESULT_RED_WIN,
    "0-1": RESULT_BLACK_WIN, "黑胜": RESULT_BLACK_WIN,
    "1/2-1/2": RESULT_DRAW, "和棋": RESULT_DRAW, "和局": RESULT_DRAW,
    "平局": RESULT_DRAW, "*": None,
}


def parse_game_movetext(text: str, result: Optional[int] = None) -> ParsedGame:
    """Parse a whole game's movetext into move integers (§2).

    Auto-detects per-token whether it is ICCS or Chinese/WXF. Chinese moves are
    resolved against an internally-maintained :class:`_BoardState`, alternating
    sides starting from RED (§1: red moves first). ICCS moves are absolute and
    are also replayed on the board state so a later Chinese move stays correct
    even in mixed records.
    """
    # extract trailing result marker if present and not given
    if result is None:
        for tok, val in _RESULT_MAP.items():
            if tok in text:
                result = val
                break

    tokens = tokenize_movetext(text)
    game = ParsedGame(notation="unknown", result=result, raw_tokens=list(tokens))
    board = _BoardState.initial()
    side = RED

    notations_seen = set()
    for tok in tokens:
        kind = detect_notation(tok)
        try:
            if kind == "iccs":
                move = parse_iccs_move(tok)
            elif kind == "wxf":
                move = parse_wxf_move(tok, board, side)
            else:
                # not a move token; skip silently
                continue
        except (ValueError, WXFParseError) as exc:
            game.error = f"parse error on {tok!r}: {exc}"
            return game
        notations_seen.add(kind)
        game.moves.append(move)
        board.apply(mv_from(move), mv_to(move))
        side = BLACK if side == RED else RED

    if notations_seen == {"iccs"}:
        game.notation = "iccs"
    elif notations_seen == {"wxf"}:
        game.notation = "wxf"
    elif notations_seen:
        game.notation = "mixed"
    return game


# ==========================================================================
# 2/3. Sample generation + tensor encoding (needs §3 Position + numpy)
# ==========================================================================
@dataclass
class Sample:
    """One supervised sample (pre-tensor)."""

    fen: str
    move_int: int          # board-frame move (§2)
    side_to_move: int      # RED / BLACK
    result: int            # +1 red win / -1 black win / 0 draw (absolute)


def _try_import_xqcore():
    """Import the (possibly unbuilt) C++ rules kernel. Returns module or None."""
    try:
        from xqai import _xqcore  # type: ignore
        return _xqcore
    except Exception:
        return None


def _try_import_encoding():
    try:
        from xqai import encoding  # type: ignore
        return encoding
    except Exception:
        return None


def game_to_samples(moves: Iterable[int], result: int) -> Tuple[List[Sample], bool, Optional[str]]:
    """Replay ``moves`` from the initial position, yielding one sample per ply.

    Uses ``xqai._xqcore.Position`` (§3): pushes each move in turn and records
    ``(fen, move_int, side_to_move, result)`` *before* the push. Validates each
    move against ``pos.legal_moves()``.

    Returns ``(samples, ok, reason)``. If any move is illegal (or the engine is
    unavailable) ``ok`` is False and the partial samples are discarded by the
    caller (data cleaning: illegal -> skip whole game and count it).
    """
    xqcore = _try_import_xqcore()
    if xqcore is None:
        return [], False, "xqai._xqcore unavailable (extension not built)"

    pos = xqcore.Position()
    samples: List[Sample] = []
    for i, move in enumerate(moves):
        legal = set(pos.legal_moves())
        if move not in legal:
            return [], False, f"illegal move at ply {i}: {move} ({move_to_iccs(move)})"
        samples.append(
            Sample(
                fen=pos.fen(),
                move_int=move,
                side_to_move=pos.side_to_move(),
                result=result,
            )
        )
        pos.push(move)
    return samples, True, None


def encode_sample(sample: Sample):
    """Turn a :class:`Sample` into ``(planes[15,10,9] fp16, pi_index int, z int8)``.

    Needs ``numpy``, ``xqai.encoding`` (§4) and ``xqai._xqcore`` (§3). The pi
    index is the move in the **normalized (current-side) frame** (§4) so it is
    consistent with ``encode``/``legal_mask``. ``z`` is the game result from the
    perspective of the side to move (+1 win / -1 loss / 0 draw).
    """
    import numpy as np  # local import: optional dependency

    xqcore = _try_import_xqcore()
    encoding = _try_import_encoding()
    if xqcore is None or encoding is None:
        raise RuntimeError("xqai._xqcore / xqai.encoding unavailable")

    pos = xqcore.Position.from_fen(sample.fen)
    planes = encoding.encode(pos).astype(np.float16)

    # move into the normalized frame for the policy index (§4: flip when black)
    if sample.side_to_move == BLACK:
        pi_index = encoding.flip_move(sample.move_int)
    else:
        pi_index = sample.move_int

    # z from the side-to-move's perspective
    if sample.result == RESULT_DRAW:
        z = 0
    elif sample.side_to_move == RED:
        z = sample.result            # +1 if red won
    else:
        z = -sample.result           # black to move: black win (-1) -> +1
    return planes, int(pi_index), np.int8(z)


# Policy action space (move = from*90 + to), kept in sync with xqai.encoding.
POLICY_SIZE = 8100


def encode_soft_sample(fen: str, side_is_black: bool,
                       move_cp_list, root_cp: float, *,
                       policy_temp: float = 100.0, value_scale: float = 500.0):
    """High-quality label encoder for engine-annotated positions (§ data v2).

    Unlike :func:`encode_sample` (one-hot move + final game result), this turns
    a Pikafish **MultiPV** analysis into a *soft* policy target and an engine-eval
    value target:

    - ``move_cp_list``: ``[(move_uci, cp), ...]`` for the top-k moves, cp scores
      from the **side-to-move's** perspective (UCI convention). The policy is
      ``softmax(cp / policy_temp)`` over these moves, written into a dense
      ``[8100]`` vector (zero elsewhere) in the normalized (current-side) frame
      so it is consistent with ``encode`` / ``flip_move`` (§4).
    - ``root_cp``: the position's engine score (best line), squashed to
      ``v = tanh(root_cp / value_scale) in (-1, 1)`` — a dense value signal that
      avoids the draw-dominated, sparse final-result label.

    Returns ``(planes f16[15,10,9], pi f16[8100], z f16 scalar)``. The shard
    ``"pi"`` key triggers :class:`pretrain.ShardDataset`'s soft-label path;
    ``"z"`` is read as float, so a continuous value just works.
    """
    import numpy as np

    xqcore = _try_import_xqcore()
    encoding = _try_import_encoding()
    if xqcore is None or encoding is None:
        raise RuntimeError("xqai._xqcore / xqai.encoding unavailable")

    pos = xqcore.Position.from_fen(fen)
    planes = encoding.encode(pos).astype(np.float16)

    pi = np.zeros(POLICY_SIZE, dtype=np.float32)
    if move_cp_list:
        cps = np.array([cp for _, cp in move_cp_list], dtype=np.float64)
        w = np.exp((cps - cps.max()) / float(policy_temp))
        w /= w.sum()
        for (mv_uci, _), prob in zip(move_cp_list, w):
            mv_int = parse_iccs_move(mv_uci)
            idx = encoding.flip_move(mv_int) if side_is_black else mv_int
            pi[int(idx)] += float(prob)
    z = float(np.tanh(float(root_cp) / float(value_scale)))
    return planes, pi.astype(np.float16), np.float16(z)


# ==========================================================================
# 3. round-trip verification
# ==========================================================================
def verify_move_roundtrip(move: int) -> bool:
    """Check ``mv_from*90 + mv_to == move`` (§2 packing is lossless)."""
    return make_move(mv_from(move), mv_to(move)) == move


def verify_iccs_roundtrip(token: str) -> bool:
    """ICCS round trip: ``token -> move -> token`` is identity (case-folded)."""
    move = parse_iccs_move(token)
    return move_to_iccs(move) == token.lower()


# ==========================================================================
# 4. file ingestion + CLI
# ==========================================================================
def iter_games_from_text(text: str) -> Iterable[str]:
    """Yield per-game movetext blocks from a file's contents.

    A ``.pgn``-style file separates games by blank lines / ``[Event ...]`` tag
    blocks. A plain ``.txt`` file is treated as one game per non-empty line, but
    if a result marker is found we also split on it. This is intentionally
    lenient.
    """
    # PGN: split on tag-section starts. We keep movetext after the tags.
    if "[" in text and "]" in text and re.search(r"^\s*\[", text, re.M):
        blocks = re.split(r"(?=^\s*\[Event )", text, flags=re.M)
        for blk in blocks:
            if not blk.strip():
                continue
            # strip tag lines
            movetext = re.sub(r"^\s*\[[^\]]*\]\s*$", "", blk, flags=re.M)
            if movetext.strip():
                yield movetext
        return
    # plain text: one game per non-empty line
    for line in text.splitlines():
        if line.strip():
            yield line


def iter_record_files(input_dir: str) -> Iterable[str]:
    for root, _dirs, files in os.walk(input_dir):
        for name in sorted(files):
            if name.lower().endswith((".pgn", ".txt")):
                yield os.path.join(root, name)


@dataclass
class PipelineStats:
    files: int = 0
    total_games: int = 0
    valid_games: int = 0
    skipped_games: int = 0
    samples: int = 0
    skip_reasons: Dict[str, int] = field(default_factory=dict)

    def add_skip(self, reason: str) -> None:
        self.skipped_games += 1
        # collapse to a coarse key
        key = reason.split(":")[0]
        self.skip_reasons[key] = self.skip_reasons.get(key, 0) + 1


def run_pipeline(input_dir: str, output_dir: str, limit: Optional[int] = None,
                 shard_size: int = 4096) -> PipelineStats:
    """End-to-end: parse + clean + (optionally) encode + write shards.

    Tensor encoding / shard writing happens only when ``numpy`` and the C++
    extension (§3) are importable; otherwise we still parse, clean and report
    statistics (so the pipeline is useful even before the kernel is built).
    """
    stats = PipelineStats()
    os.makedirs(output_dir, exist_ok=True)

    try:
        import numpy as np  # noqa: F401
        have_numpy = True
    except Exception:
        have_numpy = False
    have_engine = _try_import_xqcore() is not None and _try_import_encoding() is not None
    can_encode = have_numpy and have_engine

    shard_planes: List = []
    shard_pi: List[int] = []
    shard_z: List[int] = []
    shard_idx = 0

    def flush_shard():
        nonlocal shard_idx, shard_planes, shard_pi, shard_z
        if not shard_planes:
            return
        import numpy as np
        path = os.path.join(output_dir, f"shard_{shard_idx:05d}.npz")
        np.savez_compressed(
            path,
            planes=np.stack(shard_planes).astype(np.float16),
            pi_index=np.asarray(shard_pi, dtype=np.int32),
            z=np.asarray(shard_z, dtype=np.int8),
        )
        shard_idx += 1
        shard_planes, shard_pi, shard_z = [], [], []

    for path in iter_record_files(input_dir):
        stats.files += 1
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as fh:
                text = fh.read()
        except OSError as exc:
            stats.add_skip(f"io error: {exc}")
            continue

        for movetext in iter_games_from_text(text):
            if limit is not None and stats.total_games >= limit:
                break
            stats.total_games += 1
            game = parse_game_movetext(movetext)
            if game.error or not game.moves:
                stats.add_skip(game.error or "empty game")
                continue
            result = game.result if game.result is not None else RESULT_DRAW

            if not can_encode:
                # parse-only mode: count the game as valid (parsing succeeded)
                stats.valid_games += 1
                stats.samples += len(game.moves)
                continue

            samples, ok, reason = game_to_samples(game.moves, result)
            if not ok:
                stats.add_skip(reason or "illegal game")
                continue
            stats.valid_games += 1
            for s in samples:
                planes, pi_index, z = encode_sample(s)
                shard_planes.append(planes)
                shard_pi.append(pi_index)
                shard_z.append(int(z))
                stats.samples += 1
                if len(shard_planes) >= shard_size:
                    flush_shard()
        if limit is not None and stats.total_games >= limit:
            break

    if can_encode:
        flush_shard()

    return stats


def _print_stats(stats: PipelineStats, encoded: bool) -> None:
    print("=" * 56)
    print("数据管线统计 / pipeline statistics")
    print("-" * 56)
    print(f"  文件 files            : {stats.files}")
    print(f"  总局 total games      : {stats.total_games}")
    print(f"  有效局 valid games    : {stats.valid_games}")
    print(f"  跳过局 skipped games  : {stats.skipped_games}")
    print(f"  样本数 samples        : {stats.samples}")
    if not encoded:
        print("  (parse-only mode: numpy / xqai._xqcore unavailable, no shards written)")
    if stats.skip_reasons:
        print("  跳过原因 skip reasons:")
        for reason, cnt in sorted(stats.skip_reasons.items(), key=lambda kv: -kv[1]):
            print(f"     {cnt:6d}  {reason}")
    print("=" * 56)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="xqai supervised-pretraining data pipeline "
                    "(parse Xiangqi records -> move ints -> training shards).",
    )
    parser.add_argument("--input", default="data/raw",
                        help="directory of .pgn/.txt game records (default: data/raw)")
    parser.add_argument("--output", default="data/processed",
                        help="output directory for shards (default: data/processed)")
    parser.add_argument("--limit", type=int, default=None,
                        help="process at most N games (for quick runs)")
    parser.add_argument("--shard-size", type=int, default=4096,
                        help="samples per output shard (default: 4096)")
    args = parser.parse_args(argv)

    try:
        import numpy  # noqa: F401
        have_numpy = True
    except Exception:
        have_numpy = False
    encoded = have_numpy and (_try_import_xqcore() is not None) \
        and (_try_import_encoding() is not None)

    stats = run_pipeline(args.input, args.output, limit=args.limit,
                         shard_size=args.shard_size)
    _print_stats(stats, encoded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
