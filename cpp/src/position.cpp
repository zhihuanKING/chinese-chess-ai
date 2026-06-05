#include "xq/position.h"

#include <algorithm>
#include <cstring>
#include <sstream>
#include <stdexcept>

#include "xq/zobrist.h"

namespace xq {

// ===========================================================================
// Construction / board setup
// ===========================================================================

Position::Position() { set_start(); }

void Position::clear() {
    std::memset(board_, 0, sizeof(board_));
    stm_ = RED;
    ply_ = 0;
    rule60_ = 0;
    hash_ = 0;
    history_.clear();
    seen_.clear();
}

void Position::set_start() {
    clear();
    // row 0 = black back rank (top), row 9 = red back rank (bottom).
    // Order on back rank: R H E A K A E H R (cols 0..8).
    static const int8_t back[9] = {PT_ROOK,    PT_HORSE,   PT_ELEPHANT,
                                   PT_ADVISOR, PT_KING,    PT_ADVISOR,
                                   PT_ELEPHANT, PT_HORSE,  PT_ROOK};
    for (int c = 0; c < kCols; ++c) {
        board_[make_sq(0, c)] = -back[c];  // black
        board_[make_sq(9, c)] = back[c];   // red
    }
    // cannons: row 2 (black) and row 7 (red), cols 1 and 7.
    board_[make_sq(2, 1)] = -PT_CANNON;
    board_[make_sq(2, 7)] = -PT_CANNON;
    board_[make_sq(7, 1)] = PT_CANNON;
    board_[make_sq(7, 7)] = PT_CANNON;
    // pawns: row 3 (black) and row 6 (red), cols 0,2,4,6,8.
    for (int c = 0; c < kCols; c += 2) {
        board_[make_sq(3, c)] = -PT_PAWN;
        board_[make_sq(6, c)] = PT_PAWN;
    }
    recompute_hash();
    seen_[hash_] = 1;
}

void Position::recompute_hash() {
    const Zobrist& z = zobrist_table();
    uint64_t h = 0;
    for (int q = 0; q < kSquares; ++q) {
        Piece p = board_[q];
        if (p != 0) h ^= z.pieces[zob_piece_index(p)][q];
    }
    if (stm_ == BLACK) h ^= z.side;
    hash_ = h;
}

int Position::king_square(Side s) const {
    Piece target = static_cast<Piece>(side_sign(s) * PT_KING);
    for (int q = 0; q < kSquares; ++q)
        if (board_[q] == target) return q;
    return -1;
}

// ===========================================================================
// Attack detection
// ===========================================================================

// Can the piece sitting at `from` attack (move to / capture on) `target`?
// Geometry-only: caller ensures `from` is occupied. Blocking rules respected.
bool Position::attacks_square(int from, int target) const {
    Piece p = board_[from];
    if (p == 0) return false;
    int t = piece_type(p);
    Side s = is_red(p) ? RED : BLACK;
    int fr = row_of(from), fc = col_of(from);
    int tr = row_of(target), tc = col_of(target);
    int dr = tr - fr, dc = tc - fc;

    switch (t) {
        case PT_KING: {
            // One orthogonal step inside palace. (Flying-general handled
            // separately as a direct king-vs-king column attack below.)
            if ((std::abs(dr) + std::abs(dc)) != 1) return false;
            // Palace bounds for target.
            int pc = tc;
            bool col_ok = (pc >= 3 && pc <= 5);
            bool row_ok = (s == RED) ? (tr >= 7 && tr <= 9) : (tr >= 0 && tr <= 2);
            return col_ok && row_ok;
        }
        case PT_ADVISOR: {
            if (std::abs(dr) != 1 || std::abs(dc) != 1) return false;
            bool col_ok = (tc >= 3 && tc <= 5);
            bool row_ok = (s == RED) ? (tr >= 7 && tr <= 9) : (tr >= 0 && tr <= 2);
            return col_ok && row_ok;
        }
        case PT_ELEPHANT: {
            if (std::abs(dr) != 2 || std::abs(dc) != 2) return false;
            // cannot cross the river.
            bool side_ok = (s == RED) ? (tr >= 5) : (tr <= 4);
            if (!side_ok) return false;
            // elephant eye must be empty.
            int eye = make_sq(fr + dr / 2, fc + dc / 2);
            return board_[eye] == 0;
        }
        case PT_HORSE: {
            // L-shape with leg-block (蹩马腿).
            if (!((std::abs(dr) == 2 && std::abs(dc) == 1) ||
                  (std::abs(dr) == 1 && std::abs(dc) == 2)))
                return false;
            int legr = fr, legc = fc;
            if (std::abs(dr) == 2)
                legr = fr + dr / 2;  // vertical leg
            else
                legc = fc + dc / 2;  // horizontal leg
            return board_[make_sq(legr, legc)] == 0;
        }
        case PT_ROOK: {
            if (dr != 0 && dc != 0) return false;
            int stepr = (dr == 0) ? 0 : (dr > 0 ? 1 : -1);
            int stepc = (dc == 0) ? 0 : (dc > 0 ? 1 : -1);
            int r = fr + stepr, c = fc + stepc;
            while (r != tr || c != tc) {
                if (board_[make_sq(r, c)] != 0) return false;  // blocked
                r += stepr;
                c += stepc;
            }
            return true;
        }
        case PT_CANNON: {
            if (dr != 0 && dc != 0) return false;
            int stepr = (dr == 0) ? 0 : (dr > 0 ? 1 : -1);
            int stepc = (dc == 0) ? 0 : (dc > 0 ? 1 : -1);
            int r = fr + stepr, c = fc + stepc;
            int screens = 0;
            while (r != tr || c != tc) {
                if (board_[make_sq(r, c)] != 0) screens++;
                r += stepr;
                c += stepc;
            }
            // Cannon attacks a target only over exactly one screen (capture).
            // (A move to an empty square is not an "attack".)
            return screens == 1;
        }
        case PT_PAWN: {
            // forward by one; after crossing river also sideways by one.
            int forward = (s == RED) ? -1 : 1;  // red moves up (decreasing row)
            if (dr == forward && dc == 0) return true;
            bool crossed = (s == RED) ? (fr <= 4) : (fr >= 5);
            if (crossed && dr == 0 && std::abs(dc) == 1) return true;
            return false;
        }
    }
    return false;
}

// Is side `s`'s king attacked by any enemy piece (incl. flying-general)?
bool Position::king_attacked(Side s) const {
    int ksq = king_square(s);
    if (ksq < 0) return true;  // king captured == "attacked"

    // Flying-general (飞将): if the two kings share a file with no piece in
    // between, the side to move is considered in check (the move that creates
    // this face-off is illegal).
    int eksq = king_square(opp(s));
    if (eksq >= 0 && col_of(ksq) == col_of(eksq)) {
        int c = col_of(ksq);
        int r0 = std::min(row_of(ksq), row_of(eksq));
        int r1 = std::max(row_of(ksq), row_of(eksq));
        bool blocked = false;
        for (int r = r0 + 1; r < r1; ++r)
            if (board_[make_sq(r, c)] != 0) {
                blocked = true;
                break;
            }
        if (!blocked) return true;
    }

    Side e = opp(s);
    int esign = side_sign(e);
    for (int q = 0; q < kSquares; ++q) {
        Piece p = board_[q];
        if (p == 0) continue;
        if ((p > 0 ? 1 : -1) != esign) continue;  // only enemy pieces
        if (attacks_square(q, ksq)) return true;
    }
    return false;
}

bool Position::is_check() const { return king_attacked(stm_); }

// ===========================================================================
// Pseudo-legal move generation
// ===========================================================================

namespace {
// Push a candidate move if the destination is empty or holds an enemy piece.
inline void try_add(const int8_t* b, int mysign, int from, int to,
                    std::vector<Move>& out) {
    int8_t d = b[to];
    if (d == 0 || ((d > 0 ? 1 : -1) != mysign)) out.push_back(make_move(from, to));
}
}  // namespace

void Position::gen_pseudo(std::vector<Move>& out) const {
    int mysign = side_sign(stm_);
    Side s = stm_;
    for (int from = 0; from < kSquares; ++from) {
        Piece p = board_[from];
        if (p == 0 || (p > 0 ? 1 : -1) != mysign) continue;
        int t = piece_type(p);
        int fr = row_of(from), fc = col_of(from);

        switch (t) {
            case PT_KING: {
                static const int dxy[4][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
                for (auto& d : dxy) {
                    int nr = fr + d[0], nc = fc + d[1];
                    if (nc < 3 || nc > 5) continue;
                    bool row_ok = (s == RED) ? (nr >= 7 && nr <= 9)
                                             : (nr >= 0 && nr <= 2);
                    if (!row_ok || !on_board(nr, nc)) continue;
                    try_add(board_, mysign, from, make_sq(nr, nc), out);
                }
                break;
            }
            case PT_ADVISOR: {
                static const int dxy[4][2] = {{1, 1}, {1, -1}, {-1, 1}, {-1, -1}};
                for (auto& d : dxy) {
                    int nr = fr + d[0], nc = fc + d[1];
                    if (nc < 3 || nc > 5) continue;
                    bool row_ok = (s == RED) ? (nr >= 7 && nr <= 9)
                                             : (nr >= 0 && nr <= 2);
                    if (!row_ok) continue;
                    try_add(board_, mysign, from, make_sq(nr, nc), out);
                }
                break;
            }
            case PT_ELEPHANT: {
                static const int dxy[4][2] = {{2, 2}, {2, -2}, {-2, 2}, {-2, -2}};
                for (auto& d : dxy) {
                    int nr = fr + d[0], nc = fc + d[1];
                    if (!on_board(nr, nc)) continue;
                    bool side_ok = (s == RED) ? (nr >= 5) : (nr <= 4);
                    if (!side_ok) continue;
                    if (board_[make_sq(fr + d[0] / 2, fc + d[1] / 2)] != 0) continue;
                    try_add(board_, mysign, from, make_sq(nr, nc), out);
                }
                break;
            }
            case PT_HORSE: {
                static const int dxy[8][2] = {{2, 1},  {2, -1}, {-2, 1}, {-2, -1},
                                              {1, 2},  {1, -2}, {-1, 2}, {-1, -2}};
                for (auto& d : dxy) {
                    int nr = fr + d[0], nc = fc + d[1];
                    if (!on_board(nr, nc)) continue;
                    int legr = fr, legc = fc;
                    if (std::abs(d[0]) == 2)
                        legr = fr + d[0] / 2;
                    else
                        legc = fc + d[1] / 2;
                    if (board_[make_sq(legr, legc)] != 0) continue;  // 蹩马腿
                    try_add(board_, mysign, from, make_sq(nr, nc), out);
                }
                break;
            }
            case PT_ROOK: {
                static const int dxy[4][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
                for (auto& d : dxy) {
                    int nr = fr + d[0], nc = fc + d[1];
                    while (on_board(nr, nc)) {
                        int to = make_sq(nr, nc);
                        if (board_[to] == 0) {
                            out.push_back(make_move(from, to));
                        } else {
                            if ((board_[to] > 0 ? 1 : -1) != mysign)
                                out.push_back(make_move(from, to));
                            break;
                        }
                        nr += d[0];
                        nc += d[1];
                    }
                }
                break;
            }
            case PT_CANNON: {
                static const int dxy[4][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
                for (auto& d : dxy) {
                    int nr = fr + d[0], nc = fc + d[1];
                    // non-capturing slides over empties.
                    while (on_board(nr, nc) && board_[make_sq(nr, nc)] == 0) {
                        out.push_back(make_move(from, make_sq(nr, nc)));
                        nr += d[0];
                        nc += d[1];
                    }
                    // first non-empty = screen; look for capture beyond it.
                    if (on_board(nr, nc)) {
                        nr += d[0];
                        nc += d[1];
                        while (on_board(nr, nc)) {
                            int to = make_sq(nr, nc);
                            if (board_[to] != 0) {
                                if ((board_[to] > 0 ? 1 : -1) != mysign)
                                    out.push_back(make_move(from, to));
                                break;
                            }
                            nr += d[0];
                            nc += d[1];
                        }
                    }
                }
                break;
            }
            case PT_PAWN: {
                int forward = (s == RED) ? -1 : 1;
                int nr = fr + forward, nc = fc;
                if (on_board(nr, nc)) try_add(board_, mysign, from, make_sq(nr, nc), out);
                bool crossed = (s == RED) ? (fr <= 4) : (fr >= 5);
                if (crossed) {
                    for (int dc : {-1, 1}) {
                        int sc = fc + dc;
                        if (sc >= 0 && sc < kCols)
                            try_add(board_, mysign, from, make_sq(fr, sc), out);
                    }
                }
                break;
            }
        }
    }
}

void Position::gen_captures(std::vector<Move>& out) const {
    std::vector<Move> all;
    gen_pseudo(all);
    for (Move m : all)
        if (board_[move_to(m)] != 0) out.push_back(m);
}

// ===========================================================================
// make / unmake
// ===========================================================================

bool Position::do_move(Move m) {
    const Zobrist& z = zobrist_table();
    int from = move_from(m), to = move_to(m);
    Piece moved = board_[from];
    Piece captured = board_[to];

    Undo u{m, captured, rule60_, hash_};

    // update hash: remove moved from `from`, remove captured at `to`, add moved at `to`.
    hash_ ^= z.pieces[zob_piece_index(moved)][from];
    if (captured != 0) hash_ ^= z.pieces[zob_piece_index(captured)][to];
    hash_ ^= z.pieces[zob_piece_index(moved)][to];

    board_[to] = moved;
    board_[from] = 0;

    // flip side
    hash_ ^= z.side;
    Side mover = stm_;
    stm_ = opp(stm_);

    // Legality: mover's own king must not be in check, and kings must not face.
    bool illegal = king_attacked(mover);

    if (illegal) {
        // undo immediately
        board_[from] = moved;
        board_[to] = captured;
        stm_ = mover;
        hash_ = u.hash;
        return false;
    }

    rule60_ = (captured != 0) ? 0 : rule60_ + 1;
    history_.push_back(u);
    ++ply_;
    return true;
}

void Position::undo_move() {
    Undo u = history_.back();
    history_.pop_back();
    int from = move_from(u.move), to = move_to(u.move);
    Piece moved = board_[to];
    board_[from] = moved;
    board_[to] = u.captured;
    stm_ = opp(stm_);
    rule60_ = u.rule60;
    hash_ = u.hash;
    --ply_;
}

void Position::push(Move m) {
    // push assumes a legal move (contract); but we still apply via do_move so
    // the king-facing / self-check invariant holds. If a caller pushes an
    // illegal move we still apply it raw to keep history consistent.
    if (!do_move(m)) {
        // Apply raw (illegal) — keep behaviour predictable for callers that
        // bypass legal_moves(); maintains history so pop() works.
        const Zobrist& z = zobrist_table();
        int from = move_from(m), to = move_to(m);
        Piece moved = board_[from];
        Piece captured = board_[to];
        Undo u{m, captured, rule60_, hash_};
        hash_ ^= z.pieces[zob_piece_index(moved)][from];
        if (captured != 0) hash_ ^= z.pieces[zob_piece_index(captured)][to];
        hash_ ^= z.pieces[zob_piece_index(moved)][to];
        board_[to] = moved;
        board_[from] = 0;
        hash_ ^= z.side;
        stm_ = opp(stm_);
        rule60_ = (captured != 0) ? 0 : rule60_ + 1;
        history_.push_back(u);
        ++ply_;
    }
    seen_[hash_]++;
}

void Position::pop() {
    // decrement repetition count for the position we are leaving.
    auto it = seen_.find(hash_);
    if (it != seen_.end()) {
        if (--(it->second) <= 0) seen_.erase(it);
    }
    undo_move();
}

// ===========================================================================
// legal moves & result
// ===========================================================================

std::vector<Move> Position::legal_moves() const {
    std::vector<Move> pseudo;
    gen_pseudo(pseudo);
    std::vector<Move> legal;
    legal.reserve(pseudo.size());
    Position& self = const_cast<Position&>(*this);
    for (Move m : pseudo) {
        if (self.do_move(m)) {
            self.undo_move();
            legal.push_back(m);
        }
    }
    return legal;
}

int Position::repetition_count() const {
    auto it = seen_.find(hash_);
    return it == seen_.end() ? 0 : it->second;
}

Result Position::result() const {
    // Threefold-equivalent repetition: contract uses "3rd repetition => draw".
    if (repetition_count() >= 3) return DRAW;
    // 120-move natural draw: 120 full moves = 240 half-moves without capture.
    if (rule60_ >= 240) return DRAW;

    if (legal_moves().empty()) {
        // No legal moves: checkmate (将死) or stalemate (困毙). In Xiangqi
        // both lose for the side to move.
        return (stm_ == RED) ? BLACK_WIN : RED_WIN;
    }
    return ONGOING;
}

// ===========================================================================
// FEN parse / export (lossless round-trip)
// ===========================================================================
// FEN piece letters: K A B/E N/H R C P uppercase=red, lowercase=black.
// We use the common Xiangqi mapping with E (elephant) and H (horse), but also
// accept B (bishop=elephant) and N (knight=horse) on input for compatibility.

namespace {
char piece_to_fen(Piece p) {
    static const char red[8] = {'.', 'K', 'A', 'E', 'H', 'R', 'C', 'P'};
    int t = piece_type(p);
    char c = red[t];
    return is_red(p) ? c : static_cast<char>(c - 'A' + 'a');
}

Piece fen_to_piece(char c) {
    bool red = (c >= 'A' && c <= 'Z');
    char u = red ? c : static_cast<char>(c - 'a' + 'A');
    int t = 0;
    switch (u) {
        case 'K': t = PT_KING; break;
        case 'A': t = PT_ADVISOR; break;
        case 'E': case 'B': t = PT_ELEPHANT; break;
        case 'H': case 'N': t = PT_HORSE; break;
        case 'R': t = PT_ROOK; break;
        case 'C': t = PT_CANNON; break;
        case 'P': t = PT_PAWN; break;
        default: return 0;
    }
    return static_cast<Piece>(red ? t : -t);
}
}  // namespace

Position Position::from_fen(const std::string& fen) {
    Position p;
    p.clear();
    std::istringstream iss(fen);
    std::string boardpart, side;
    if (!(iss >> boardpart)) throw std::invalid_argument("empty FEN");
    iss >> side;  // optional fields follow

    int row = 0, col = 0;
    for (char c : boardpart) {
        if (c == '/') {
            if (col != kCols)
                throw std::invalid_argument("FEN row width mismatch");
            ++row;
            col = 0;
            continue;
        }
        if (c >= '1' && c <= '9') {
            col += c - '0';
        } else {
            if (row >= kRows || col >= kCols)
                throw std::invalid_argument("FEN out of range");
            Piece pc = fen_to_piece(c);
            if (pc == 0) throw std::invalid_argument(std::string("bad FEN char: ") + c);
            p.board_[make_sq(row, col)] = pc;
            ++col;
        }
    }
    if (row != kRows - 1)
        throw std::invalid_argument("FEN must have 10 rows");

    p.stm_ = RED;
    if (side == "b" || side == "B") p.stm_ = BLACK;
    // optional halfmove clock (field index 5 in standard, but layouts vary);
    // attempt to read a trailing integer as the rule60 counter if present.
    // Standard Xiangqi FEN: <board> <side> <- - <halfmove> <fullmove>
    std::string f3, f4, f5, f6;
    iss >> f3 >> f4 >> f5 >> f6;
    if (!f5.empty()) {
        try {
            p.rule60_ = std::stoi(f5);
        } catch (...) {
            p.rule60_ = 0;
        }
    }
    // Fullmove number (field 6). Reconstruct ply_ so that fen() round-trips it:
    // fullmove N with side s => ply = (N-1)*2 + (s==BLACK ? 1 : 0), and
    // fen() exports ply_/2 + 1 == N for both sides.
    if (!f6.empty()) {
        try {
            int fullmove = std::stoi(f6);
            if (fullmove < 1) fullmove = 1;
            p.ply_ = (fullmove - 1) * 2 + (p.stm_ == BLACK ? 1 : 0);
        } catch (...) {
            p.ply_ = 0;
        }
    }
    p.recompute_hash();
    p.seen_[p.hash_] = 1;
    return p;
}

std::string Position::fen() const {
    std::ostringstream os;
    for (int r = 0; r < kRows; ++r) {
        int empty = 0;
        for (int c = 0; c < kCols; ++c) {
            Piece p = board_[make_sq(r, c)];
            if (p == 0) {
                ++empty;
            } else {
                if (empty) {
                    os << empty;
                    empty = 0;
                }
                os << piece_to_fen(p);
            }
        }
        if (empty) os << empty;
        if (r != kRows - 1) os << '/';
    }
    os << ' ' << (stm_ == RED ? 'w' : 'b');
    os << " - - " << rule60_ << ' ' << (ply_ / 2 + 1);
    return os.str();
}

// ===========================================================================
// perft
// ===========================================================================

namespace {
uint64_t perft_rec(Position& p, int depth) {
    if (depth == 0) return 1;
    std::vector<Move> pseudo;
    p.gen_pseudo(pseudo);
    uint64_t nodes = 0;
    for (Move m : pseudo) {
        if (p.do_move(m)) {
            nodes += perft_rec(p, depth - 1);
            p.undo_move();
        }
    }
    return nodes;
}
}  // namespace

uint64_t Position::perft(const std::string& fen, int depth) {
    Position p = from_fen(fen);
    return perft_rec(p, depth);
}

}  // namespace xq
