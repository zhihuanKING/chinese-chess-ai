// Command-line perft tool.
//   perft <depth> [fen]
// Defaults to the standard Xiangqi starting position if no FEN is given.
// Also accepts a "divide" mode: perft divide <depth> [fen] prints per-move counts.
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "xq/position.h"

using namespace xq;

static const char* kStartFen =
    "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w - - 0 1";

namespace {
uint64_t perft_rec(Position& p, int depth) {
    if (depth == 0) return 1;
    std::vector<Move> pseudo;
    p.gen_pseudo(pseudo);
    uint64_t n = 0;
    for (Move m : pseudo)
        if (p.do_move(m)) {
            n += perft_rec(p, depth - 1);
            p.undo_move();
        }
    return n;
}
}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::printf("usage: perft <depth> [fen]\n");
        std::printf("       perft divide <depth> [fen]\n");
        return 1;
    }

    int argi = 1;
    bool divide = false;
    if (std::string(argv[1]) == "divide") {
        divide = true;
        argi = 2;
        if (argc < 3) {
            std::printf("usage: perft divide <depth> [fen]\n");
            return 1;
        }
    }

    int depth = std::atoi(argv[argi++]);
    std::string fen = (argi < argc) ? argv[argi] : kStartFen;

    Position p = Position::from_fen(fen);

    if (divide) {
        std::vector<Move> pseudo;
        p.gen_pseudo(pseudo);
        uint64_t total = 0;
        for (Move m : pseudo) {
            if (p.do_move(m)) {
                uint64_t n = perft_rec(p, depth - 1);
                p.undo_move();
                std::printf("%d -> %d : %llu\n", move_from(m), move_to(m),
                            static_cast<unsigned long long>(n));
                total += n;
            }
        }
        std::printf("total: %llu\n", static_cast<unsigned long long>(total));
    } else {
        uint64_t n = perft_rec(p, depth);
        std::printf("perft(%d) = %llu\n", depth,
                    static_cast<unsigned long long>(n));
    }
    return 0;
}
