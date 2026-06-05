// Position: mailbox board + full Xiangqi rules, move generation, make/unmake,
// incremental zobrist, FEN round-trip, repetition tracking and result().
// Implements the contract in INTERFACES.md §3.
#ifndef XQ_POSITION_H
#define XQ_POSITION_H

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include "xq/types.h"

namespace xq {

class Position {
public:
    // Initial Xiangqi starting position (RED to move).
    Position();

    // Construct from FEN. Throws std::invalid_argument on malformed input.
    static Position from_fen(const std::string& fen);

    // ---- Contract API (INTERFACES.md §3) ----
    std::string fen() const;                 // lossless round-trip
    Side side_to_move() const { return stm_; }
    std::vector<Move> legal_moves() const;   // filtered: no leaving/moving into check
    bool is_check() const;                   // is the side to move in check?
    void push(Move m);                       // make move, maintain history
    void pop();                              // undo last move
    Result result() const;                   // ONGOING/RED_WIN/BLACK_WIN/DRAW
    uint64_t zobrist() const { return hash_; }
    const int8_t* board_data() const { return board_; }  // length 90, row-major
    int repetition_count() const;            // repetitions of current position
    int ply() const { return ply_; }

    // Static leaf counter for testing (INTERFACES.md §3).
    static uint64_t perft(const std::string& fen, int depth);

    // ---- Extra helpers used by the engine / tests ----
    // Pseudo-legal generation (does not filter self-check). Public for testing.
    void gen_pseudo(std::vector<Move>& out) const;
    // Is `s`'s king attacked given the current board?
    bool king_attacked(Side s) const;
    // Capture-only moves (for quiescence); pseudo-legal.
    void gen_captures(std::vector<Move>& out) const;
    Piece piece_at(int sq) const { return board_[sq]; }
    int king_square(Side s) const;

    // Make/unmake working directly on Move (used by perft & search).
    // do_move returns true if the move is legal (own king not left in check),
    // leaving the move applied; if illegal it is fully undone and returns false.
    bool do_move(Move m);
    void undo_move();

private:
    int8_t board_[kSquares];
    Side stm_;
    int ply_;            // half-moves played from the constructed root
    int rule60_;         // half-moves since last capture (for 120-move natural draw)
    uint64_t hash_;

    struct Undo {
        Move move;
        Piece captured;
        int rule60;
        uint64_t hash;
    };
    std::vector<Undo> history_;
    // Count of each zobrist key seen along the played path (for repetition).
    std::unordered_map<uint64_t, int> seen_;

    void set_start();
    void recompute_hash();
    void clear();

    // attack predicates from a given square by a given side's piece occupying `from`.
    bool attacks_square(int from, int target) const;  // can piece at `from` reach target?
};

}  // namespace xq

#endif  // XQ_POSITION_H
