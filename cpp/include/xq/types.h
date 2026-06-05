// Core constants and small helpers for the Xiangqi (Chinese Chess) kernel.
// All conventions follow INTERFACES.md §1 / §2.
#ifndef XQ_TYPES_H
#define XQ_TYPES_H

#include <cstdint>

namespace xq {

// ---- Board geometry (INTERFACES.md §1) -------------------------------------
// 9 columns x 10 rows = 90 squares. sq in [0,90).
//   row = sq / 9  (0..9),  col = sq % 9  (0..8)
// RED is at the bottom (row 9 side), BLACK at the top (row 0 side). RED moves first.
constexpr int kCols = 9;
constexpr int kRows = 10;
constexpr int kSquares = 90;

inline constexpr int row_of(int sq) { return sq / kCols; }
inline constexpr int col_of(int sq) { return sq % kCols; }
inline constexpr int make_sq(int row, int col) { return row * kCols + col; }
inline constexpr bool on_board(int row, int col) {
    return row >= 0 && row < kRows && col >= 0 && col < kCols;
}

// ---- Move encoding (INTERFACES.md §2) --------------------------------------
// move = from_sq * 90 + to_sq, range [0, 8100).
using Move = int;
constexpr Move kNoMove = -1;
constexpr int kActionDim = 8100;

inline constexpr Move make_move(int from, int to) { return from * kSquares + to; }
inline constexpr int move_from(Move m) { return m / kSquares; }
inline constexpr int move_to(Move m) { return m % kSquares; }

// ---- Piece encoding (INTERFACES.md §1) -------------------------------------
// board[] is int8: 0 = empty; positive = RED, negative = BLACK.
// |value| in 1..7: 1=K(将/帅) 2=A(士/仕) 3=E(象/相) 4=H(马) 5=R(车) 6=C(炮) 7=P(兵/卒)
using Piece = int8_t;
enum PieceType {
    PT_NONE = 0,
    PT_KING = 1,    // 将/帅
    PT_ADVISOR = 2, // 士/仕
    PT_ELEPHANT = 3,// 象/相
    PT_HORSE = 4,   // 马
    PT_ROOK = 5,    // 车
    PT_CANNON = 6,  // 炮
    PT_PAWN = 7,    // 兵/卒
};

inline constexpr int piece_type(Piece p) { return p < 0 ? -p : p; }
inline constexpr bool is_red(Piece p) { return p > 0; }
inline constexpr bool is_black(Piece p) { return p < 0; }

// ---- Side to move ----------------------------------------------------------
enum Side { RED = 0, BLACK = 1 };
inline constexpr Side opp(Side s) { return s == RED ? BLACK : RED; }
// sign multiplier for a side's own pieces in board[]: RED=+1, BLACK=-1.
inline constexpr int side_sign(Side s) { return s == RED ? 1 : -1; }

// ---- Game result (INTERFACES.md §3) ----------------------------------------
enum Result { ONGOING = 0, RED_WIN = 1, BLACK_WIN = 2, DRAW = 3 };

}  // namespace xq

#endif  // XQ_TYPES_H
