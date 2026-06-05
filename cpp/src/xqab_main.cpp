// xqab: Alpha-Beta engine with a small UCCI-subset command-line interface.
//
// Interactive UCCI subset (read from stdin, one command per line):
//   ucci                       -> "id name xqab" ... "ucciok"
//   isready                    -> "readyok"
//   position startpos [moves m1 m2 ...]
//   position fen <FEN> [moves m1 m2 ...]   (FEN may be multiple tokens)
//   go time <ms>  | go depth <d> | go movetime <ms>
//                              -> "bestmove <iccs>"
//   quit
//
// Moves are accepted/printed in ICCS-like coordinates (e.g. "h2e2"): file letter
// a..i (col 0..8) + rank digit 0..9 where rank 0 is the bottom (red) edge.
//
// Non-interactive one-shot helper:
//   xqab --bestmove <ms> "<FEN>"   -> prints best move integer and ICCS string.
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "xq/position.h"
#include "xq/search.h"

using namespace xq;

static const char* kStartFen =
    "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w - - 0 1";

// ---- coordinate <-> ICCS conversion --------------------------------------
// Internal sq: row 0 at top (black), row 9 at bottom (red), col 0..8.
// ICCS: file 'a'..'i' = col; rank '0'..'9' counts from the bottom, i.e.
//   iccs_rank = 9 - row.
static std::string sq_to_iccs(int sq) {
    int r = row_of(sq), c = col_of(sq);
    std::string s;
    s += static_cast<char>('a' + c);
    s += static_cast<char>('0' + (9 - r));
    return s;
}

static int iccs_to_sq(const std::string& tok, size_t off) {
    int c = tok[off] - 'a';
    int rank = tok[off + 1] - '0';
    int r = 9 - rank;
    return make_sq(r, c);
}

static std::string move_to_iccs(Move m) {
    return sq_to_iccs(move_from(m)) + sq_to_iccs(move_to(m));
}

static Move iccs_to_move(const std::string& tok) {
    if (tok.size() < 4) return kNoMove;
    int from = iccs_to_sq(tok, 0);
    int to = iccs_to_sq(tok, 2);
    return make_move(from, to);
}

static void apply_moves(Position& p, std::istringstream& iss) {
    std::string tok;
    while (iss >> tok) {
        Move m = iccs_to_move(tok);
        if (m != kNoMove) p.push(m);
    }
}

int main(int argc, char** argv) {
    // ---- one-shot mode ----
    if (argc >= 3 && std::string(argv[1]) == "--bestmove") {
        int ms = std::atoi(argv[2]);
        std::string fen = (argc >= 4) ? argv[3] : kStartFen;
        Position p = Position::from_fen(fen);
        Move m = search(p, ms);
        if (m == kNoMove) {
            std::printf("bestmove (none)\n");
        } else {
            std::printf("move=%d iccs=%s\n", m, move_to_iccs(m).c_str());
        }
        return 0;
    }

    // ---- UCCI-subset interactive loop ----
    Position pos = Position::from_fen(kStartFen);
    std::string line;
    while (std::getline(std::cin, line)) {
        std::istringstream iss(line);
        std::string cmd;
        iss >> cmd;
        if (cmd.empty()) continue;

        if (cmd == "ucci") {
            std::cout << "id name xqab\n";
            std::cout << "id author xqai\n";
            std::cout << "ucciok\n";
            std::cout.flush();
        } else if (cmd == "isready") {
            std::cout << "readyok\n";
            std::cout.flush();
        } else if (cmd == "position") {
            std::string kind;
            iss >> kind;
            if (kind == "startpos") {
                pos = Position::from_fen(kStartFen);
                std::string maybe;
                if (iss >> maybe && maybe == "moves") apply_moves(pos, iss);
            } else if (kind == "fen") {
                // FEN consumes the next tokens until "moves" or end of line.
                std::vector<std::string> toks;
                std::string t;
                bool have_moves = false;
                while (iss >> t) {
                    if (t == "moves") {
                        have_moves = true;
                        break;
                    }
                    toks.push_back(t);
                }
                std::ostringstream fen;
                for (size_t i = 0; i < toks.size(); ++i) {
                    if (i) fen << ' ';
                    fen << toks[i];
                }
                pos = Position::from_fen(fen.str());
                if (have_moves) apply_moves(pos, iss);
            }
        } else if (cmd == "go") {
            std::string what;
            int ms = 1000;
            int depth = -1;
            while (iss >> what) {
                if (what == "time" || what == "movetime") {
                    iss >> ms;
                } else if (what == "depth") {
                    iss >> depth;
                }
            }
            Move m;
            if (depth > 0)
                m = search_depth(pos, depth);
            else
                m = search(pos, ms);
            if (m == kNoMove)
                std::cout << "bestmove (none)\n";
            else
                std::cout << "bestmove " << move_to_iccs(m) << "\n";
            std::cout.flush();
        } else if (cmd == "quit") {
            break;
        }
        // unknown commands are silently ignored (UCCI tolerant behaviour).
    }
    return 0;
}
