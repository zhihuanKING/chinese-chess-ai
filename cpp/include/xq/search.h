// Alpha-Beta engine (INTERFACES.md §8): negamax + alpha-beta + transposition
// table + iterative deepening + quiescence + MVV-LVA move ordering, plus a
// staged hand-written evaluation (material + piece-square + king safety).
#ifndef XQ_SEARCH_H
#define XQ_SEARCH_H

#include "xq/position.h"

namespace xq {

// Static evaluation in centipawns, from the side-to-move's perspective.
int evaluate(const Position& p);

// Iterative-deepening search; spends up to time_ms milliseconds and returns the
// best move integer (move = from*90+to). Returns kNoMove if no legal move.
Move search(Position& p, int time_ms);

// Lower-level entry used by tests: fixed-depth search returning best move.
Move search_depth(Position& p, int depth);

}  // namespace xq

#endif  // XQ_SEARCH_H
