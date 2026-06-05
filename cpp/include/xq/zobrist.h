// Zobrist hashing tables for incremental position hashing (INTERFACES.md §3).
// 64-bit keys, reproducible across make/unmake. Deterministic seed so that
// hashes are stable across runs (important for transposition tables and tests).
#ifndef XQ_ZOBRIST_H
#define XQ_ZOBRIST_H

#include <cstdint>

#include "xq/types.h"

namespace xq {

// Index helper: pieces are mapped to [0,14):
//   0..6  -> RED   K,A,E,H,R,C,P  (type 1..7)
//   7..13 -> BLACK K,A,E,H,R,C,P
inline int zob_piece_index(Piece p) {
    int t = piece_type(p);  // 1..7
    return is_red(p) ? (t - 1) : (7 + t - 1);
}

struct Zobrist {
    uint64_t pieces[14][kSquares];
    uint64_t side;  // XORed when side to move is BLACK

    Zobrist() { init(); }

    // splitmix64 deterministic PRNG.
    static uint64_t mix(uint64_t& x) {
        x += 0x9e3779b97f4a7c15ULL;
        uint64_t z = x;
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
        return z ^ (z >> 31);
    }

    void init() {
        uint64_t s = 0xC0FFEE123456789ULL;
        for (int i = 0; i < 14; ++i)
            for (int q = 0; q < kSquares; ++q) pieces[i][q] = mix(s);
        side = mix(s);
    }
};

// Single shared, lazily-initialized table.
inline const Zobrist& zobrist_table() {
    static const Zobrist z;
    return z;
}

}  // namespace xq

#endif  // XQ_ZOBRIST_H
