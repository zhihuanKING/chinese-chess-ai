#!/usr/bin/env python3
"""Pure-parsing / round-trip tests for ``scripts/prepare_data.py``.

Runs under the **system** interpreter with only the standard library:

    /usr/bin/python3 scripts/test_prepare_data.py

The numpy / ``xqai._xqcore`` (§3, §4) dependent paths are guarded with
``try/except`` and reported as SKIPPED when those are unavailable -- they are
*not* required for this test to pass. (Do not touch ``.venv``.)
"""

from __future__ import annotations

import os
import sys
import traceback

# make "import prepare_data" work no matter the cwd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import prepare_data as pd  # noqa: E402


# --------------------------------------------------------------------------
# tiny test harness (no third-party deps)
# --------------------------------------------------------------------------
_PASS = 0
_FAIL = 0
_SKIP = 0


def check(cond, msg):
    global _PASS, _FAIL
    if cond:
        _PASS += 1
    else:
        _FAIL += 1
        print(f"  FAIL: {msg}")


def eq(got, want, msg):
    check(got == want, f"{msg}: got {got!r}, want {want!r}")


def section(name):
    print(f"\n[{name}]")


# --------------------------------------------------------------------------
# §1 coordinate helpers
# --------------------------------------------------------------------------
def test_coords():
    section("coordinate helpers (§1)")
    # sq packing
    eq(pd.rc_to_sq(9, 4), 85, "red king start sq")
    eq(pd.sq_to_rc(85), (9, 4), "sq -> (row,col)")
    eq(pd.make_move(0, 90 - 1), 0 * 90 + 89, "make_move packing")
    # move helpers (§2)
    m = pd.make_move(85, 76)
    eq(pd.mv_from(m), 85, "mv_from")
    eq(pd.mv_to(m), 76, "mv_to")


# --------------------------------------------------------------------------
# §1a ICCS parsing + round trip
# --------------------------------------------------------------------------
def test_iccs():
    section("ICCS coordinate notation (§1a)")
    # e0 = red king start: file e=4, rank 0 -> row 9, col 4 -> sq 85
    eq(pd.iccs_coord_to_sq("e", "0"), 85, "ICCS e0 -> red king sq")
    # a9 = black's left-rook corner (col 0, rank 9 -> row 0) -> sq 0
    eq(pd.iccs_coord_to_sq("a", "9"), 0, "ICCS a9 -> sq 0")
    # i0 = red right corner: col 8, rank 0 -> row 9 -> sq 9*9+8 = 89
    eq(pd.iccs_coord_to_sq("i", "0"), 89, "ICCS i0 -> sq 89")

    # classic cannon move h2e2 (right cannon to center file, both on rank 2)
    mv = pd.parse_iccs_move("h2e2")
    fr, to = pd.mv_from(mv), pd.mv_to(mv)
    # h2: col 7, rank 2 -> row 7 -> sq 70 ; e2: col 4, rank 2 -> sq 67
    eq(fr, 7 * 9 + 7, "h2 from-sq")
    eq(to, 7 * 9 + 4, "e2 to-sq")

    # detection
    eq(pd.detect_notation("h2e2"), "iccs", "detect iccs")
    eq(pd.detect_notation("H2E2"), "iccs", "detect iccs upper")


def test_iccs_roundtrip():
    section("ICCS round-trip: coord -> move -> coord")
    for tok in ["a0a9", "e0e1", "h2e2", "i9a0", "b0c2", "e3e4"]:
        check(pd.verify_iccs_roundtrip(tok), f"iccs roundtrip {tok}")
        mv = pd.parse_iccs_move(tok)
        check(pd.verify_move_roundtrip(mv), f"move pack roundtrip {tok}")
        # full bidirectional: coords -> move -> coords identity
        eq(pd.move_to_iccs(mv), tok.lower(), f"move_to_iccs {tok}")
    # every square round-trips through sq<->iccs
    allok = all(
        pd.iccs_coord_to_sq(pd.sq_to_iccs(s)[0], pd.sq_to_iccs(s)[1]) == s
        for s in range(90)
    )
    check(allok, "all 90 squares iccs roundtrip")


# --------------------------------------------------------------------------
# §1b WXF / Chinese parsing
# --------------------------------------------------------------------------
def test_wxf_basic():
    section("WXF / Chinese notation (§1b)")
    board = pd._BoardState.initial()

    # Red: 炮二平五  (right cannon -> center). Red file 二=2 -> col 7 (cannon
    # at row 7,col 7). 平五 -> col 4, same row.
    mv = pd.parse_wxf_move("炮二平五", board, pd.RED)
    eq(pd.mv_from(mv), pd.rc_to_sq(7, 7), "炮二平五 from")
    eq(pd.mv_to(mv), pd.rc_to_sq(7, 4), "炮二平五 to")
    board.apply(pd.mv_from(mv), pd.mv_to(mv))

    # Black: 马8进7  (Arabic file). Black file 8 -> col 7 (horse at row0,col7).
    # 进 (forward for black = +1 row), target file 7 -> col 6, dcol=1 -> drow=2.
    mv2 = pd.parse_wxf_move("马8进7", board, pd.BLACK)
    eq(pd.mv_from(mv2), pd.rc_to_sq(0, 7), "马8进7 from")
    eq(pd.mv_to(mv2), pd.rc_to_sq(2, 6), "马8进7 to")
    board.apply(pd.mv_from(mv2), pd.mv_to(mv2))

    # Red horse develop: 马二进三 -> file 二=col7 horse(row9,col7); 进 forward
    # (-1 row for red) target file 三=col6, dcol=1 -> drow=2 -> row7.
    board2 = pd._BoardState.initial()
    mv3 = pd.parse_wxf_move("马二进三", board2, pd.RED)
    eq(pd.mv_from(mv3), pd.rc_to_sq(9, 7), "马二进三 from")
    eq(pd.mv_to(mv3), pd.rc_to_sq(7, 6), "马二进三 to")


def test_wxf_chariot_forward_back():
    section("WXF straight movers (车/兵): distance semantics")
    board = pd._BoardState.initial()
    # Red 车一进一: rook file 一=1 -> col 8 (rook at row9,col8). 进一 -> 1 row
    # forward -> row 8.
    mv = pd.parse_wxf_move("车一进一", board, pd.RED)
    eq(pd.mv_from(mv), pd.rc_to_sq(9, 8), "车一进一 from")
    eq(pd.mv_to(mv), pd.rc_to_sq(8, 8), "车一进一 to (1 forward)")

    # Red 兵三进一: pawn file 三=col6 (red pawn at row6,col6). 进一 -> row5.
    mv2 = pd.parse_wxf_move("兵三进一", board, pd.RED)
    eq(pd.mv_from(mv2), pd.rc_to_sq(6, 6), "兵三进一 from")
    eq(pd.mv_to(mv2), pd.rc_to_sq(5, 6), "兵三进一 to")

    # Red 车一平二: sideways file 一(col8) -> file 二(col7), same row.
    mv3 = pd.parse_wxf_move("车一平二", board, pd.RED)
    eq(pd.mv_to(mv3), pd.rc_to_sq(9, 7), "车一平二 to (sideways)")


def test_wxf_front_rear():
    section("WXF stacked-piece prefix 前/后")
    # build a position with two red cannons on the same file (col 4).
    b = pd._BoardState()
    b.cells[pd.rc_to_sq(7, 4)] = pd.C   # rear (closer to red base / higher row)
    b.cells[pd.rc_to_sq(5, 4)] = pd.C   # front (closer to enemy / lower row)
    # 前炮进一 -> the front (row5) cannon moves 1 forward -> row4.
    mv = pd.parse_wxf_move("前炮进一", b, pd.RED)
    eq(pd.mv_from(mv), pd.rc_to_sq(5, 4), "前炮 picks front cannon (row5)")
    eq(pd.mv_to(mv), pd.rc_to_sq(4, 4), "前炮进一 to row4")
    # 后炮进一 -> the rear (row7) cannon.
    mv2 = pd.parse_wxf_move("后炮进一", b, pd.RED)
    eq(pd.mv_from(mv2), pd.rc_to_sq(7, 4), "后炮 picks rear cannon (row7)")


def test_wxf_red_black_symmetry():
    """Red/Black direction symmetry (§1b): the same move type for each side must
    map to vertically-mirrored squares.

    The board is mirror-symmetric about the horizontal centre line, so a Red
    piece on ``row r`` corresponds to a Black piece on ``row 9 - r`` in the
    *same* column. Files are counted from each side's own right, so a Red move
    written ``<f_r>`` and the mirrored Black move written ``<f_b>`` land on the
    same column. We therefore assert, for several move types:

      * source columns match (col_red == col_black),
      * source rows mirror (row_red + row_black == 9),
      * destination columns match and destination rows mirror.

    This catches sign errors in 进/退 direction, file-numeral handedness
    (red col=9-f vs black col=f-1) and the slanted-row sign.
    """
    section("WXF red/black direction symmetry (§1b)")

    def rc(sq):
        return pd.sq_to_rc(sq)

    def assert_mirror(name, red_tok, black_tok, red_board=None, black_board=None):
        rb = red_board if red_board is not None else pd._BoardState.initial()
        bb = black_board if black_board is not None else pd._BoardState.initial()
        mr = pd.parse_wxf_move(red_tok, rb, pd.RED)
        mb = pd.parse_wxf_move(black_tok, bb, pd.BLACK)
        (frr, frc) = rc(pd.mv_from(mr))
        (brr, brc) = rc(pd.mv_from(mb))
        (trr, trc) = rc(pd.mv_to(mr))
        (tbr, tbc) = rc(pd.mv_to(mb))
        eq(frc, brc, f"{name}: source col matches")
        eq(frr + brr, 9, f"{name}: source row mirrors (sum==9)")
        eq(trc, tbc, f"{name}: dest col matches")
        eq(trr + tbr, 9, f"{name}: dest row mirrors (sum==9)")
        # sanity: both resolved squares are on-board
        check(0 <= pd.mv_to(mr) < 90 and 0 <= pd.mv_to(mb) < 90,
              f"{name}: both dests on board")

    # Cannon sideways (平): red 炮二平五 vs black 炮8平5 (file2/8 -> col7, file5 -> col4)
    assert_mirror("炮平 (cannon flat)", "炮二平五", "炮8平5")

    # Horse forward (进, slanted): red 马二进三 vs black 马8进7
    assert_mirror("马进 (horse fwd)", "马二进三", "马8进7")

    # Chariot forward (进, straight/distance): red 车一进一 vs black 车9进1
    assert_mirror("车进 (chariot fwd)", "车一进一", "车9进1")

    # Pawn forward (进, straight): red 兵三进一 vs black 卒7进1
    assert_mirror("兵进 (pawn fwd)", "兵三进一", "卒7进1")

    # Advisor forward (进, slanted +/-1 row): use a custom mirrored board so the
    # advisor sits at a non-initial square for both sides.
    rb = pd._BoardState()
    rb.cells[pd.rc_to_sq(8, 4)] = pd.A          # red advisor at (8,4)
    bb = pd._BoardState()
    bb.cells[pd.rc_to_sq(1, 4)] = -pd.A         # mirror: black advisor at (1,4)
    # red 仕五进四 (file五->col4 -> file四->col5); black 士5进6 (file5->col4 -> file6->col5)
    assert_mirror("仕进 (advisor fwd)", "仕五进四", "士5进6",
                  red_board=rb, black_board=bb)

    # Elephant forward (进, slanted 2 rows): mirrored custom boards.
    rb2 = pd._BoardState()
    rb2.cells[pd.rc_to_sq(9, 2)] = pd.E         # red elephant at (9,2)
    bb2 = pd._BoardState()
    bb2.cells[pd.rc_to_sq(0, 2)] = -pd.E        # mirror: black elephant at (0,2)
    # red 相七进五 (file七->col2 -> file五->col4); black 象3进5 (file3->col2 -> file5->col4)
    assert_mirror("相进 (elephant fwd)", "相七进五", "象3进5",
                  red_board=rb2, black_board=bb2)

    # Retreat (退) symmetry, straight cannon: mirrored custom boards.
    rb3 = pd._BoardState()
    rb3.cells[pd.rc_to_sq(6, 4)] = pd.C         # red cannon at (6,4)
    bb3 = pd._BoardState()
    bb3.cells[pd.rc_to_sq(3, 4)] = -pd.C        # mirror: black cannon at (3,4)
    assert_mirror("炮退 (cannon back)", "炮五退一", "炮5退1",
                  red_board=rb3, black_board=bb3)

    # Front/rear prefix (前/后) symmetry: stacked cannons, mirrored.
    rb4 = pd._BoardState()
    rb4.cells[pd.rc_to_sq(5, 4)] = pd.C         # red front (min row)
    rb4.cells[pd.rc_to_sq(7, 4)] = pd.C         # red rear
    bb4 = pd._BoardState()
    bb4.cells[pd.rc_to_sq(4, 4)] = -pd.C        # black front (max row)
    bb4.cells[pd.rc_to_sq(2, 4)] = -pd.C        # black rear
    assert_mirror("前炮 (front prefix)", "前炮进一", "前炮进一",
                  red_board=rb4, black_board=bb4)
    # rebuild boards (assert_mirror only reads, but be explicit) for 后
    assert_mirror("后炮 (rear prefix)", "后炮进一", "后炮进一",
                  red_board=rb4, black_board=bb4)


def test_wxf_detect():
    section("WXF auto-detection")
    eq(pd.detect_notation("炮二平五"), "wxf", "detect 炮二平五")
    eq(pd.detect_notation("马8进7"), "wxf", "detect 马8进7")
    eq(pd.detect_notation("前炮进二"), "wxf", "detect 前炮进二")
    eq(pd.detect_notation("xyzz"), "unknown", "detect garbage")


# --------------------------------------------------------------------------
# 1c. full-game parsing (synthetic records)
# --------------------------------------------------------------------------
# A short ICCS game (a real opening sequence; all coordinate moves).
ICCS_GAME = "1. h2e2 h7e7 2. h0g2 h9g7 3. b0c2 b9c7 1/2-1/2"

# The same kind of opening in Chinese notation (independent synthetic game).
WXF_GAME = "1. 炮二平五 马8进7 2. 马二进三 车9进1 3. 车一平二 车9平4 *"

# A plain-text one-liner (no move numbers, ICCS).
ICCS_PLAIN = "h2e2 h7e7 b0c2"


def test_full_games():
    section("full-game parsing (synthetic)")
    g = pd.parse_game_movetext(ICCS_GAME)
    eq(g.error, None, "iccs game no error")
    eq(g.notation, "iccs", "iccs game notation")
    eq(len(g.moves), 6, "iccs game move count")
    eq(g.result, pd.RESULT_DRAW, "iccs game result draw")
    for mv in g.moves:
        check(pd.verify_move_roundtrip(mv), "iccs game move roundtrip")

    g2 = pd.parse_game_movetext(WXF_GAME)
    eq(g2.error, None, f"wxf game no error (got {g2.error})")
    eq(g2.notation, "wxf", "wxf game notation")
    eq(len(g2.moves), 6, "wxf game move count")
    for mv in g2.moves:
        check(pd.verify_move_roundtrip(mv), "wxf game move roundtrip")

    g3 = pd.parse_game_movetext(ICCS_PLAIN)
    eq(len(g3.moves), 3, "plain iccs move count")


def test_tokenizer():
    section("movetext tokenizer")
    toks = pd.tokenize_movetext("1. h2e2 h7e7 2. b0c2 1/2-1/2")
    eq(toks, ["h2e2", "h7e7", "b0c2"], "iccs tokens, numbers/result stripped")
    # glued CJK run splits into 4-char moves
    toks2 = pd.tokenize_movetext("炮二平五马8进7")
    check("炮二平五" in toks2, "cjk run split keeps 炮二平五")


# --------------------------------------------------------------------------
# 2/3. optional encoding path (skipped without numpy + _xqcore)
# --------------------------------------------------------------------------
def test_encoding_optional():
    section("sample encoding (optional: numpy + xqai._xqcore §3/§4)")
    global _SKIP
    try:
        import numpy  # noqa: F401
    except Exception:
        _SKIP += 1
        print("  SKIP: numpy unavailable")
        return
    if pd._try_import_xqcore() is None or pd._try_import_encoding() is None:
        _SKIP += 1
        print("  SKIP: xqai._xqcore / xqai.encoding unavailable (extension not built)")
        return
    # if we get here, exercise the real pipeline on the ICCS game
    g = pd.parse_game_movetext(ICCS_GAME)
    samples, ok, reason = pd.game_to_samples(g.moves, pd.RESULT_DRAW)
    check(ok, f"game_to_samples ok (reason={reason})")
    if ok and samples:
        planes, pi, z = pd.encode_sample(samples[0])
        eq(tuple(planes.shape), (15, 10, 9), "planes shape [15,10,9]")
        check(0 <= pi < pd.ACTION_DIM, "pi index in range")


def test_illegal_cleaning_optional():
    section("illegal-move cleaning (optional engine)")
    global _SKIP
    if pd._try_import_xqcore() is None:
        _SKIP += 1
        print("  SKIP: engine unavailable, cleaning path needs §3 Position")
        return
    # a blatantly illegal first move (rook jumping across the board)
    bad = [pd.make_move(0, 89)]
    samples, ok, reason = pd.game_to_samples(bad, pd.RESULT_DRAW)
    check(not ok, "illegal game rejected")
    eq(samples, [], "illegal game yields no samples")


def main():
    tests = [
        test_coords,
        test_iccs,
        test_iccs_roundtrip,
        test_wxf_basic,
        test_wxf_chariot_forward_back,
        test_wxf_front_rear,
        test_wxf_red_black_symmetry,
        test_wxf_detect,
        test_full_games,
        test_tokenizer,
        test_encoding_optional,
        test_illegal_cleaning_optional,
    ]
    for t in tests:
        try:
            t()
        except Exception:
            global _FAIL
            _FAIL += 1
            print(f"  FAIL (exception in {t.__name__}):")
            traceback.print_exc()

    print("\n" + "=" * 56)
    print(f"PASS={_PASS}  FAIL={_FAIL}  SKIP={_SKIP}")
    print("=" * 56)
    return 0 if _FAIL == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
