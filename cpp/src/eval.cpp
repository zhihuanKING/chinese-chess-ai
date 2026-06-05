#include "xq/search.h"
#include "xq/types.h"

namespace xq {

// ---- Material values (centipawns) ------------------------------------------
// King given a large value so material counting never under-weighs it (king
// is never actually captured in legal play; checkmate is scored separately).
static const int kMaterial[8] = {
    0,      // none
    10000,  // K 将/帅
    200,    // A 士/仕
    200,    // E 象/相
    450,    // H 马
    1000,   // R 车
    500,    // C 炮
    100,    // P 兵/卒
};

// ---- Piece-square tables ---------------------------------------------------
// Tables are written from RED's perspective, indexed [row][col] with row 0 at
// the top (black side). For BLACK we mirror vertically (row -> 9-row).
// Values are small positional nudges in centipawns.

// Pawn: reward advancing across the river and central files.
static const int PST_PAWN[10][9] = {
    { 0,  0,  0,  0,  0,  0,  0,  0,  0},
    { 0,  0,  0,  0,  0,  0,  0,  0,  0},
    { 0,  0,  0,  0,  0,  0,  0,  0,  0},
    {-2,  0, -2,  0,  6,  0, -2,  0, -2},  // start row (black pawns sit here)
    { 2,  0,  8,  0,  8,  0,  8,  0,  2},
    {10, 18, 22, 35, 40, 35, 22, 18, 10},  // just crossed river
    {20, 27, 30, 40, 42, 40, 30, 27, 20},
    {20, 30, 45, 55, 55, 55, 45, 30, 20},
    {20, 30, 50, 65, 70, 65, 50, 30, 20},
    { 0,  3,  6,  9, 12,  9,  6,  3,  0},
};

// Horse: prize central, active outposts; penalize edges.
static const int PST_HORSE[10][9] = {
    { 0, -3,  2,  2,  0,  2,  2, -3,  0},
    {-3,  2,  4,  5,  0,  5,  4,  2, -3},
    { 2,  4,  6,  8,  8,  8,  6,  4,  2},
    { 2,  5,  8, 10, 12, 10,  8,  5,  2},
    { 2,  6,  9, 12, 14, 12,  9,  6,  2},
    { 4,  8, 11, 14, 14, 14, 11,  8,  4},
    { 4,  8, 12, 14, 14, 14, 12,  8,  4},
    { 2,  6,  9, 12, 12, 12,  9,  6,  2},
    {-3,  2,  4,  5,  4,  5,  4,  2, -3},
    { 0, -3,  2,  2,  0,  2,  2, -3,  0},
};

// Rook: open files / advanced ranks slightly preferred.
static const int PST_ROOK[10][9] = {
    {-2,  4,  4,  6,  8,  6,  4,  4, -2},
    { 4,  8,  8, 10, 12, 10,  8,  8,  4},
    { 2,  6,  6,  8, 10,  8,  6,  6,  2},
    { 2,  6,  6,  8, 10,  8,  6,  6,  2},
    { 2,  6,  6,  8, 10,  8,  6,  6,  2},
    { 4,  8,  8, 10, 12, 10,  8,  8,  4},
    { 6, 10, 10, 12, 14, 12, 10, 10,  6},
    { 6, 10, 10, 12, 14, 12, 10, 10,  6},
    { 8, 10, 12, 14, 14, 14, 12, 10,  8},
    {-2,  8,  8, 12, 12, 12,  8,  8, -2},
};

// Cannon: central files & ranks behind screens.
static const int PST_CANNON[10][9] = {
    { 2,  2,  0, -2, -4, -2,  0,  2,  2},
    { 0,  0,  0,  2,  4,  2,  0,  0,  0},
    { 0,  2,  4,  6,  8,  6,  4,  2,  0},
    { 0,  2,  6, 10, 12, 10,  6,  2,  0},
    { 2,  4,  8, 10, 12, 10,  8,  4,  2},
    { 2,  4,  8, 10, 12, 10,  8,  4,  2},
    { 0,  2,  6,  8, 10,  8,  6,  2,  0},
    { 0,  2,  4,  6,  8,  6,  4,  2,  0},
    { 0,  0,  2,  4,  4,  4,  2,  0,  0},
    { 2,  2,  2,  4,  6,  4,  2,  2,  2},
};

static const int* pst_for(int type) {
    switch (type) {
        case PT_PAWN:   return &PST_PAWN[0][0];
        case PT_HORSE:  return &PST_HORSE[0][0];
        case PT_ROOK:   return &PST_ROOK[0][0];
        case PT_CANNON: return &PST_CANNON[0][0];
        default:        return nullptr;  // K/A/E: positional value ~ 0
    }
}

int evaluate(const Position& p) {
    const int8_t* b = p.board_data();
    int score = 0;  // from RED's perspective first, flip at the end.

    for (int sq = 0; sq < kSquares; ++sq) {
        Piece pc = b[sq];
        if (pc == 0) continue;
        int t = piece_type(pc);
        int val = kMaterial[t];

        const int* pst = pst_for(t);
        int pos = 0;
        if (pst) {
            int r = row_of(sq), c = col_of(sq);
            if (is_red(pc)) {
                // RED is at the bottom; tables are red-perspective with row 0
                // at top, so red pieces are read mirrored (9-r) to make the
                // "advanced" rows score high for red as it pushes upward.
                pos = pst[(kRows - 1 - r) * kCols + c];
            } else {
                pos = pst[r * kCols + c];
            }
        }

        int contrib = val + pos;
        score += is_red(pc) ? contrib : -contrib;
    }

    // ---- Simple king safety: penalize an exposed king (missing guards) ----
    int red_guards = 0, black_guards = 0;
    for (int sq = 0; sq < kSquares; ++sq) {
        int t = piece_type(b[sq]);
        if (t == PT_ADVISOR || t == PT_ELEPHANT) {
            if (is_red(b[sq])) red_guards++; else black_guards++;
        }
    }
    score += (red_guards - black_guards) * 5;

    // Return from the side-to-move's perspective (negamax convention).
    return p.side_to_move() == RED ? score : -score;
}

}  // namespace xq
