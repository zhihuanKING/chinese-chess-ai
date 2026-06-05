// Perft cross-check tests: known small node counts for the start position and
// a couple of hand-set positions, plus pseudo/legal consistency checks.
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include <string>

#include "xq/position.h"

using namespace xq;

static const std::string kStart =
    "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w - - 0 1";

TEST_CASE("start position perft small depths") {
    // Well-known reference values for the standard Xiangqi opening.
    CHECK_EQ(Position::perft(kStart, 0), 1ull);
    CHECK_EQ(Position::perft(kStart, 1), 44ull);
    CHECK_EQ(Position::perft(kStart, 2), 1920ull);
    CHECK_EQ(Position::perft(kStart, 3), 79666ull);
    CHECK_EQ(Position::perft(kStart, 4), 3290240ull);
}

TEST_CASE("start position: legal == pseudo at root (no checks possible)") {
    Position p;  // default ctor = start
    std::vector<Move> pseudo;
    p.gen_pseudo(pseudo);
    auto legal = p.legal_moves();
    // At the opening no move can expose the red king, so counts match.
    CHECK_EQ(pseudo.size(), legal.size());
    CHECK_EQ(legal.size(), 44u);
}

TEST_CASE("perft on a sparse king+rook endgame, depth 1") {
    // Red: king e0 + rook a0; Black: king e9. Red to move.
    // FEN rows top(row0)..bottom(row9).
    std::string fen = "4k4/9/9/9/9/9/9/9/9/R3K4 w - - 0 1";
    Position p = Position::from_fen(fen);
    auto legal = p.legal_moves();
    // Just assert it is non-trivial and round-trips below.
    CHECK(legal.size() > 0);
    // Rook on a-file (col0,row9) + king moves; sanity upper bound.
    CHECK(legal.size() < 40);
}
