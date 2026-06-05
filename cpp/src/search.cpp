#include "xq/search.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <vector>

#include "xq/types.h"

namespace xq {

namespace {

constexpr int kInf = 1000000;
constexpr int kMateScore = 900000;  // mate score base (depth-adjusted)

using Clock = std::chrono::steady_clock;

// ---- Transposition table ----
enum Bound : uint8_t { BOUND_EXACT, BOUND_LOWER, BOUND_UPPER };

struct TTEntry {
    uint64_t key = 0;
    int32_t value = 0;
    int16_t depth = -1;
    uint8_t bound = BOUND_EXACT;
    Move best = kNoMove;
};

struct SearchState {
    std::vector<TTEntry> tt;
    uint64_t tt_mask = 0;
    Clock::time_point deadline;
    bool stop = false;
    uint64_t nodes = 0;

    explicit SearchState(size_t mb = 64) {
        size_t entries = (mb * 1024 * 1024) / sizeof(TTEntry);
        size_t pow2 = 1;
        while (pow2 * 2 <= entries) pow2 *= 2;
        if (pow2 < 1024) pow2 = 1024;
        tt.assign(pow2, TTEntry{});
        tt_mask = pow2 - 1;
    }

    TTEntry* probe(uint64_t key) {
        TTEntry& e = tt[key & tt_mask];
        return e.key == key ? &e : nullptr;
    }
    void store(uint64_t key, int value, int depth, Bound b, Move best) {
        TTEntry& e = tt[key & tt_mask];
        if (e.key != key || depth >= e.depth) {
            e.key = key;
            e.value = value;
            e.depth = static_cast<int16_t>(depth);
            e.bound = b;
            e.best = best;
        }
    }
};

// Material order for MVV-LVA (index by piece type 1..7).
const int kPieceOrder[8] = {0, 7, 1, 1, 4, 6, 5, 2};

inline int mvv_lva(const Position& p, Move m) {
    int victim = piece_type(p.piece_at(move_to(m)));
    int attacker = piece_type(p.piece_at(move_from(m)));
    // higher victim, lower attacker => higher score.
    return kPieceOrder[victim] * 16 - kPieceOrder[attacker];
}

void order_moves(const Position& p, std::vector<Move>& moves, Move tt_move) {
    std::sort(moves.begin(), moves.end(), [&](Move a, Move b) {
        if (a == tt_move) return true;
        if (b == tt_move) return false;
        bool ca = p.piece_at(move_to(a)) != 0;
        bool cb = p.piece_at(move_to(b)) != 0;
        if (ca != cb) return ca;  // captures first
        if (ca && cb) return mvv_lva(p, a) > mvv_lva(p, b);
        return false;
    });
}

bool time_up(SearchState& ss) {
    if (ss.stop) return true;
    // check the clock every so often to limit syscall overhead.
    if ((ss.nodes & 2047) == 0 && Clock::now() >= ss.deadline) ss.stop = true;
    return ss.stop;
}

int quiescence(Position& p, int alpha, int beta, SearchState& ss) {
    ss.nodes++;
    int stand = evaluate(p);
    if (stand >= beta) return beta;
    if (stand > alpha) alpha = stand;

    std::vector<Move> caps;
    p.gen_captures(caps);
    order_moves(p, caps, kNoMove);

    for (Move m : caps) {
        if (!p.do_move(m)) continue;  // skips self-check / king-facing
        int score = -quiescence(p, -beta, -alpha, ss);
        p.undo_move();
        if (ss.stop) return alpha;
        if (score >= beta) return beta;
        if (score > alpha) alpha = score;
    }
    return alpha;
}

int negamax(Position& p, int depth, int alpha, int beta, int ply,
            SearchState& ss) {
    ss.nodes++;
    if (time_up(ss)) return 0;

    // Draw by repetition / natural limit short-circuit.
    if (ply > 0) {
        if (p.repetition_count() >= 3) return 0;
    }

    int alpha_orig = alpha;
    uint64_t key = p.zobrist();
    Move tt_move = kNoMove;
    if (TTEntry* e = ss.probe(key)) {
        tt_move = e->best;
        if (e->depth >= depth) {
            if (e->bound == BOUND_EXACT) return e->value;
            if (e->bound == BOUND_LOWER && e->value > alpha) alpha = e->value;
            else if (e->bound == BOUND_UPPER && e->value < beta) beta = e->value;
            if (alpha >= beta) return e->value;
        }
    }

    if (depth <= 0) return quiescence(p, alpha, beta, ss);

    std::vector<Move> pseudo;
    p.gen_pseudo(pseudo);
    order_moves(p, pseudo, tt_move);

    int best = -kInf;
    Move best_move = kNoMove;
    int legal = 0;

    for (Move m : pseudo) {
        if (!p.do_move(m)) continue;
        ++legal;
        int score = -negamax(p, depth - 1, -beta, -alpha, ply + 1, ss);
        p.undo_move();
        if (ss.stop) return best > -kInf ? best : 0;

        if (score > best) {
            best = score;
            best_move = m;
        }
        if (score > alpha) alpha = score;
        if (alpha >= beta) break;  // cutoff
    }

    if (legal == 0) {
        // No legal moves: checkmate or stalemate. Both lose in Xiangqi.
        // Mate score is depth-adjusted so shorter mates are preferred.
        return -kMateScore + ply;
    }

    Bound b = BOUND_EXACT;
    if (best <= alpha_orig) b = BOUND_UPPER;
    else if (best >= beta) b = BOUND_LOWER;
    ss.store(key, best, depth, b, best_move);
    return best;
}

}  // namespace

Move search_depth(Position& p, int depth) {
    SearchState ss(64);
    ss.deadline = Clock::now() + std::chrono::hours(24);

    std::vector<Move> pseudo;
    p.gen_pseudo(pseudo);
    order_moves(p, pseudo, kNoMove);

    Move best = kNoMove;
    int alpha = -kInf, beta = kInf;
    int best_score = -kInf;
    for (Move m : pseudo) {
        if (!p.do_move(m)) continue;
        int score = -negamax(p, depth - 1, -beta, -alpha, 1, ss);
        p.undo_move();
        if (score > best_score) {
            best_score = score;
            best = m;
        }
        if (score > alpha) alpha = score;
    }
    return best;
}

Move search(Position& p, int time_ms) {
    SearchState ss(64);
    ss.deadline = Clock::now() + std::chrono::milliseconds(time_ms);

    Move best = kNoMove;
    // Iterative deepening.
    for (int depth = 1; depth <= 64; ++depth) {
        int alpha = -kInf, beta = kInf;
        Move iter_best = kNoMove;
        int best_score = -kInf;

        std::vector<Move> pseudo;
        p.gen_pseudo(pseudo);
        order_moves(p, pseudo, best);  // try previous best first

        bool completed = true;
        for (Move m : pseudo) {
            if (!p.do_move(m)) continue;
            int score = -negamax(p, depth - 1, -beta, -alpha, 1, ss);
            p.undo_move();
            if (ss.stop) {
                completed = false;
                break;
            }
            if (score > best_score) {
                best_score = score;
                iter_best = m;
            }
            if (score > alpha) alpha = score;
        }

        if (iter_best != kNoMove && (completed || best == kNoMove))
            best = iter_best;

        if (!completed) break;
        if (Clock::now() >= ss.deadline) break;
        // Found a forced mate — no need to search deeper.
        if (best_score >= kMateScore - 100) break;
    }
    return best;
}

}  // namespace xq
