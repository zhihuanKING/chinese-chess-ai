"""MCTS planners for xqai (INTERFACES.md §6).

Unified ``Planner`` interface::

    def search(positions: list[Position], net, n_sim: int) -> list[np.ndarray]

Given a list of ``B`` :class:`xqai._xqcore.Position` objects, runs ``n_sim``
simulations per game and returns, for each game, the improved policy
``pi`` (a ``float32`` ``[ACTION_DIM]`` numpy vector, in the **normalized
side-to-move frame** — same frame as :func:`xqai.encoding.encode` /
:func:`xqai.encoding.legal_mask`).

Two implementations, both **vectorized across the B games**:

- :class:`PUCTPlanner`  — classic AlphaZero PUCT.
- :class:`GumbelPlanner` — Gumbel-AlphaZero (Danihelka et al., ICLR 2022).

The single most important performance property (INTERFACES.md §6, "向量化批量"):
**all games advance in lockstep, and at every simulation step the leaf
positions of all B games are encoded and concatenated into ONE big batch that
is fed to ``net`` in a single forward call.** Search the file for the marker

    # === BATCHED NN EVAL ===

to find exactly where this happens (PUCT: see :meth:`PUCTPlanner._evaluate`,
Gumbel: :meth:`GumbelPlanner._eval_batch`). This is what keeps the GPU busy;
nothing here ever calls ``net`` on a single position.

Move frame
----------
Internally each tree node stores moves in the **normalized frame** (so they
line up with the network's policy output). When a move is applied to the real
``Position`` (which lives in the original board frame) it is converted back via
:func:`xqai.encoding.flip_move` iff the side to move at that node is BLACK.
"""

from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np
import torch

from .encoding import (
    ACTION_DIM,
    BLACK,
    encode,
    flip_move,
    legal_mask,
    policy_target,
)


# --------------------------------------------------------------------------- #
# Shared helpers                                                              #
# --------------------------------------------------------------------------- #
def _net_device(net) -> torch.device:
    """Best-effort device of a module (falls back to CPU)."""
    try:
        return next(net.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _to_real_move(move_norm: int, side_to_move: int) -> int:
    """Convert a normalized-frame move to a real board-frame move."""
    return flip_move(move_norm) if side_to_move == BLACK else move_norm


def _legal_moves_norm(pos) -> list[int]:
    """Legal moves of ``pos`` expressed in the normalized frame."""
    flip = pos.side_to_move() == BLACK
    return [flip_move(m) if flip else m for m in pos.legal_moves()]


def _batched_eval(net, leaf_positions: Sequence[Any]):
    """# === BATCHED NN EVAL ===

    Encode every leaf position, stack into a single ``[B, NUM_PLANES, 10, 9]``
    tensor and run **one** ``net`` forward. Returns ``(policy_probs, values,
    masks)`` as numpy arrays:

    - ``policy_probs`` : ``[B, ACTION_DIM]`` float32, softmax over **legal**
      moves only (illegal entries are exactly 0).
    - ``values``       : ``[B]`` float32 in ``[-1, 1]``.
    - ``masks``        : ``[B, ACTION_DIM]`` float32 legal masks.

    This is the one and only place either planner touches ``net``; all B games'
    leaves go through here together so the GPU sees a fat batch.
    """
    if len(leaf_positions) == 0:
        return (
            np.zeros((0, ACTION_DIM), np.float32),
            np.zeros((0,), np.float32),
            np.zeros((0, ACTION_DIM), np.float32),
        )

    device = _net_device(net)
    planes = np.stack([encode(p) for p in leaf_positions], axis=0)  # [B,15,10,9]
    masks = np.stack([legal_mask(p) for p in leaf_positions], axis=0)  # [B,8100]

    x = torch.from_numpy(planes).to(device=device, dtype=torch.float32)
    mask_t = torch.from_numpy(masks).to(device=device)

    was_training = net.training
    net.eval()
    with torch.no_grad():
        logits, value = net(x)  # ONE forward for all B leaves
        # Masked softmax over legal moves (illegal -> 0 probability).
        neg_inf = torch.finfo(logits.dtype).min
        masked = torch.where(mask_t > 0, logits, torch.full_like(logits, neg_inf))
        # Rows with no legal move (terminal leaves) -> all -inf; guard to 0.
        has_legal = (mask_t > 0).any(dim=-1, keepdim=True)
        masked = torch.where(has_legal, masked, torch.zeros_like(masked))
        probs = torch.softmax(masked, dim=-1)
        probs = torch.where(
            has_legal.expand_as(probs), probs, torch.zeros_like(probs)
        )
    if was_training:
        net.train()

    return (
        probs.detach().float().cpu().numpy(),
        value.detach().float().reshape(-1).cpu().numpy(),
        masks,
    )


# --------------------------------------------------------------------------- #
# PUCT planner                                                                #
# --------------------------------------------------------------------------- #
class _PUCTNode:
    """A single MCTS node. Children are kept in parallel arrays for speed.

    All moves are in the **normalized frame**.
    """

    __slots__ = (
        "to_play",
        "is_expanded",
        "is_terminal",
        "terminal_value",
        "moves",
        "P",
        "N",
        "W",
        "vloss",
        "children",
    )

    def __init__(self, to_play: int):
        self.to_play = to_play            # side to move at this node (RED/BLACK)
        self.is_expanded = False
        self.is_terminal = False
        self.terminal_value = 0.0         # value from this node's mover view
        self.moves: list[int] = []        # normalized-frame legal moves
        self.P: np.ndarray | None = None  # prior per child
        self.N: np.ndarray | None = None  # visit count per child
        self.W: np.ndarray | None = None  # total value per child (mover view)
        self.vloss: np.ndarray | None = None  # pending virtual losses per child
        self.children: list["_PUCTNode | None"] = []


class PUCTPlanner:
    """Vectorized batched PUCT MCTS (AlphaZero-style, INTERFACES.md §6).

    All B games run in lockstep. Each simulation: every game walks its tree
    from the root to a leaf (PUCT selection with virtual loss); the B leaf
    positions are then **batched into a single ``net`` forward**
    (see :meth:`_evaluate` / the ``# === BATCHED NN EVAL ===`` marker); each
    leaf is expanded with the returned priors and its value is backed up.

    PUCT score for a child::

        U(a) = Q(a) + c_puct * P(a) * sqrt(sum_b N(b)) / (1 + N(a))

    with ``Q(a) = W(a)/N(a)`` from the current mover's perspective (so values are
    negated on each ply as we walk down/up the tree).
    """

    def __init__(
        self,
        c_puct: float = 1.5,
        dirichlet_alpha: float = 0.2,
        dirichlet_eps: float = 0.25,
        virtual_loss: float = 3.0,
        add_noise: bool = True,
        seed: int | None = None,
    ):
        self.c_puct = float(c_puct)
        self.dirichlet_alpha = float(dirichlet_alpha)
        self.dirichlet_eps = float(dirichlet_eps)
        self.virtual_loss = float(virtual_loss)
        self.add_noise = add_noise
        self._rng = np.random.default_rng(seed)

    # -- public API --------------------------------------------------------- #
    def search(self, positions, net, n_sim: int) -> list[np.ndarray]:
        b = len(positions)
        if b == 0:
            return []

        roots = [_PUCTNode(p.side_to_move()) for p in positions]

        # Expand all roots in one batch, then add Dirichlet noise to root priors.
        self._evaluate(positions, roots, net)
        if self.add_noise:
            for root in roots:
                self._add_dirichlet(root)

        for _ in range(int(n_sim)):
            # --- selection phase: one leaf per game, in lockstep ----------- #
            leaf_nodes: list[_PUCTNode] = []
            leaf_positions: list[Any] = []
            leaf_paths: list[list[tuple[_PUCTNode, int]]] = []
            game_idx: list[int] = []

            for gi in range(b):
                pos = positions[gi]
                node = roots[gi]
                path: list[tuple[_PUCTNode, int]] = []
                applied = 0
                # Descend until we hit an unexpanded or terminal node.
                while node.is_expanded and not node.is_terminal:
                    child_i = self._select_child(node)
                    move_norm = node.moves[child_i]
                    # Apply virtual loss so other games / later sims diverge.
                    node.vloss[child_i] += self.virtual_loss
                    path.append((node, child_i))
                    real_move = _to_real_move(move_norm, node.to_play)
                    pos.push(real_move)
                    applied += 1
                    child = node.children[child_i]
                    if child is None:
                        child = _PUCTNode(pos.side_to_move())
                        node.children[child_i] = child
                    node = child

                if node.is_terminal:
                    # Already-known terminal leaf: back up immediately.
                    self._backup(path, node.terminal_value)
                    for _ in range(applied):
                        pos.pop()
                else:
                    leaf_nodes.append(node)
                    leaf_positions.append(pos)
                    leaf_paths.append(path)
                    game_idx.append(gi)

            # --- evaluation phase: ONE batched forward for all leaves ------ #
            if leaf_positions:
                values = self._evaluate(leaf_positions, leaf_nodes, net)
                # --- backup + undo the pushes we made during selection ----- #
                for k in range(len(leaf_positions)):
                    self._backup(leaf_paths[k], values[k])
                    pos = leaf_positions[k]
                    for _ in range(len(leaf_paths[k])):
                        pos.pop()

        # --- build improved policies from root visit counts ---------------- #
        out: list[np.ndarray] = []
        for root in roots:
            visit_counts = {
                root.moves[i]: int(root.N[i]) for i in range(len(root.moves))
            }
            out.append(policy_target(visit_counts))
        return out

    # -- internals ---------------------------------------------------------- #
    def _select_child(self, node: _PUCTNode) -> int:
        n = node.N + node.vloss
        sqrt_total = math.sqrt(float(n.sum()))
        # Q from the current mover's perspective. Virtual loss is modelled as a
        # losing playout (subtract vloss from W) to discourage collisions.
        q = np.where(n > 0, (node.W - node.vloss) / np.maximum(n, 1.0), 0.0)
        u = q + self.c_puct * node.P * sqrt_total / (1.0 + n)
        return int(np.argmax(u))

    def _evaluate(self, leaf_positions, leaf_nodes, net) -> np.ndarray:
        """Expand a batch of leaves with one net forward; return leaf values.

        Values are returned from each **leaf mover's** perspective.
        """
        # === BATCHED NN EVAL ===  (all B leaves -> single net() call)
        probs, values, masks = _batched_eval(net, leaf_positions)

        out_values = np.empty(len(leaf_positions), dtype=np.float32)
        for k, (pos, node) in enumerate(zip(leaf_positions, leaf_nodes)):
            res = pos.result()
            if res != _ONGOING:
                node.is_terminal = True
                node.is_expanded = True
                node.terminal_value = _terminal_value(res, node.to_play)
                out_values[k] = node.terminal_value
                continue

            moves = _legal_moves_norm(pos)
            if not moves:
                # No legal moves but engine says ongoing -> treat as loss.
                node.is_terminal = True
                node.is_expanded = True
                node.terminal_value = -1.0
                out_values[k] = -1.0
                continue

            priors = probs[k][moves]
            s = priors.sum()
            if s > 0:
                priors = priors / s
            else:
                priors = np.full(len(moves), 1.0 / len(moves), dtype=np.float32)

            node.moves = moves
            node.P = priors.astype(np.float32)
            node.N = np.zeros(len(moves), dtype=np.float32)
            node.W = np.zeros(len(moves), dtype=np.float32)
            node.vloss = np.zeros(len(moves), dtype=np.float32)
            node.children = [None] * len(moves)
            node.is_expanded = True
            out_values[k] = float(values[k])
        return out_values

    def _add_dirichlet(self, root: _PUCTNode) -> None:
        if root.is_terminal or root.P is None or len(root.P) == 0:
            return
        noise = self._rng.dirichlet([self.dirichlet_alpha] * len(root.P))
        root.P = (
            (1.0 - self.dirichlet_eps) * root.P + self.dirichlet_eps * noise
        ).astype(np.float32)

    def _backup(self, path, leaf_value: float) -> None:
        """Propagate ``leaf_value`` up the path, flipping sign each ply and
        removing the virtual loss added during selection."""
        # ``leaf_value`` is from the leaf mover's perspective. The last node on
        # the path moved to reach the leaf, so its mover is the opponent of the
        # leaf mover -> negate when applying to the parent edge.
        v = leaf_value
        for node, child_i in reversed(path):
            v = -v  # flip to this node's mover perspective
            node.N[child_i] += 1.0
            node.W[child_i] += v
            node.vloss[child_i] -= self.virtual_loss


# --------------------------------------------------------------------------- #
# Gumbel planner                                                              #
# --------------------------------------------------------------------------- #
class GumbelPlanner:
    """Gumbel-AlphaZero planner (Danihelka et al., ICLR 2022).

    Reference: "Policy improvement by planning with Gumbel", I. Danihelka,
    A. Guez, J. Schrittwieser, D. Silver, ICLR 2022; and the reference
    implementation in DeepMind ``mctx`` / OpenSpiel.

    Simplified-but-correct scheme (root-only Gumbel + Sequential Halving):

    1. At the root, sample Gumbel noise ``g(a) ~ Gumbel(0,1)`` per legal move and
       pick the top ``m`` actions by ``g(a) + logits(a)`` as the candidate set.
    2. Distribute the ``n_sim`` simulations across ``ceil(log2(m))`` Sequential
       Halving phases; in each phase every surviving candidate is simulated an
       equal number of times, then the worse half (by the Gumbel selection
       score below) is dropped.
    3. The acted-upon root child is chosen by maximizing
       ``g(a) + logits(a) + sigma(q_completed(a))`` over the final survivor.
    4. The **improved policy** returned as ``pi`` is the *completed*-value
       softmax  ``softmax(logits + sigma(q_completed))``  over all legal moves
       (the Gumbel paper's policy-improvement target), where ``q_completed``
       uses the network value for unvisited actions.

    Each candidate's child subtree below the root is searched with the same
    batched PUCT machinery, so the per-simulation NN evaluations of all B games
    are still concatenated into a single forward (the
    ``# === BATCHED NN EVAL ===`` marker in :meth:`_eval_batch`).
    """

    def __init__(
        self,
        c_puct: float = 1.5,
        gumbel_m: int = 16,
        c_visit: float = 50.0,
        c_scale: float = 1.0,
        virtual_loss: float = 3.0,
        seed: int | None = None,
    ):
        self.c_puct = float(c_puct)
        self.gumbel_m = int(gumbel_m)
        self.c_visit = float(c_visit)
        self.c_scale = float(c_scale)
        self.virtual_loss = float(virtual_loss)
        self._rng = np.random.default_rng(seed)

    # -- public API --------------------------------------------------------- #
    def search(self, positions, net, n_sim: int) -> list[np.ndarray]:
        b = len(positions)
        if b == 0:
            return []
        n_sim = int(n_sim)

        roots = [_PUCTNode(p.side_to_move()) for p in positions]
        # Root expansion: one batched forward for all B roots.
        root_logits = self._eval_batch(positions, roots, net, want_logits=True)

        # Per-game Gumbel state.
        states: list[_GumbelState] = []
        for gi in range(b):
            root = roots[gi]
            if root.is_terminal or not root.moves:
                states.append(_GumbelState.empty())
                continue
            logits = root_logits[gi][np.array(root.moves)]
            g = self._rng.gumbel(size=len(root.moves)).astype(np.float32)
            m = min(self.gumbel_m, len(root.moves))
            # Top-m candidates by g + logits.
            cand = np.argsort(-(g + logits))[:m]
            states.append(
                _GumbelState(logits=logits.astype(np.float32), g=g, cand=cand)
            )

        # Sequential Halving schedule over the candidate set.
        # Each game may have a different m; we run a shared phase loop and let
        # each game manage its own survivor list / per-candidate sim budget.
        for gi in range(b):
            st = states[gi]
            if st.is_empty:
                continue
            self._sequential_halving(positions[gi], roots[gi], st, net, n_sim)

        # Build improved policies (completed-Q softmax over all legal moves).
        out: list[np.ndarray] = []
        for gi in range(b):
            root = roots[gi]
            st = states[gi]
            if root.is_terminal or not root.moves or st.is_empty:
                out.append(policy_target({}))
                continue
            pi = self._improved_policy(root, st)
            out.append(pi)
        return out

    # -- Sequential Halving ------------------------------------------------- #
    def _sequential_halving(self, pos, root, st, net, n_sim: int) -> None:
        survivors = list(st.cand)
        m = len(survivors)
        num_phases = max(1, int(math.ceil(math.log2(m)))) if m > 1 else 1
        used = 0
        for phase in range(num_phases):
            if len(survivors) <= 1:
                break
            remaining = n_sim - used
            phases_left = num_phases - phase
            # Equal sims per survivor this phase.
            per = max(1, remaining // (phases_left * len(survivors)))
            for _ in range(per):
                # Simulate each surviving candidate once. The leaf evaluations
                # across candidates are batched into a single net forward.
                self._simulate_candidates(pos, root, survivors, net)
                used += len(survivors)
                if used >= n_sim:
                    break
            # Drop the worse half by Gumbel selection score.
            scores = self._gumbel_scores(root, st, survivors)
            keep = max(1, len(survivors) // 2)
            order = np.argsort(-scores)[:keep]
            survivors = [survivors[i] for i in order]
            if used >= n_sim:
                break
        st.survivors = survivors

    def _simulate_candidates(self, pos, root, survivors, net) -> None:
        """One simulation through each surviving root child, then batched eval.

        For each candidate we descend (forcing the first edge to that candidate,
        PUCT below) to a leaf, recording the path. Terminal leaves are backed up
        immediately. The remaining (non-terminal) leaves are then expanded with
        ONE batched ``net`` forward in :meth:`_eval_collected`. All candidates
        alias the same ``pos``, so we always descend-then-pop, storing only the
        path; the leaf state is re-derived from the path when needed.
        """
        leaf_nodes: list[_PUCTNode] = []
        leaf_paths: list[list[tuple[_PUCTNode, int]]] = []

        for ci in survivors:
            node = root
            path: list[tuple[_PUCTNode, int]] = []
            child_i = ci  # force the first edge to this candidate
            while True:
                node.vloss[child_i] += self.virtual_loss
                path.append((node, child_i))
                pos.push(_to_real_move(node.moves[child_i], node.to_play))
                child = node.children[child_i]
                if child is None:
                    child = _PUCTNode(pos.side_to_move())
                    node.children[child_i] = child
                node = child
                if not node.is_expanded or node.is_terminal:
                    break
                child_i = self._select_child(node)

            # Undo all pushes (we keep only the path; leaf re-derived later).
            for _ in range(len(path)):
                pos.pop()

            if node.is_terminal:
                self._backup(path, node.terminal_value)
            else:
                leaf_nodes.append(node)
                leaf_paths.append(path)

        if leaf_nodes:
            self._eval_collected(pos, leaf_nodes, leaf_paths, net)

    def _eval_collected(self, pos, leaf_nodes, leaf_paths, net) -> None:
        """Expand a batch of leaves (all aliasing ``pos``) with ONE net forward.

        We re-walk each path to reach the leaf state, encode it, then pop back —
        collecting all encodings, then run a single batched forward.
        """
        planes_list = []
        mask_list = []
        for path in leaf_paths:
            for nd, ci in path:
                pos.push(_to_real_move(nd.moves[ci], nd.to_play))
            planes_list.append(encode(pos))
            mask_list.append(legal_mask(pos))
            for _ in range(len(path)):
                pos.pop()

        device = _net_device(net)
        x = torch.from_numpy(np.stack(planes_list)).to(device, torch.float32)
        mask_t = torch.from_numpy(np.stack(mask_list)).to(device)
        was_training = net.training
        net.eval()
        with torch.no_grad():
            # === BATCHED NN EVAL ===  (all collected leaves -> single forward)
            logits, value = net(x)
            neg_inf = torch.finfo(logits.dtype).min
            masked = torch.where(mask_t > 0, logits, torch.full_like(logits, neg_inf))
            has_legal = (mask_t > 0).any(-1, keepdim=True)
            masked = torch.where(has_legal, masked, torch.zeros_like(masked))
            probs = torch.softmax(masked, -1)
            probs = torch.where(has_legal.expand_as(probs), probs, torch.zeros_like(probs))
        if was_training:
            net.train()
        probs_np = probs.detach().float().cpu().numpy()
        values_np = value.detach().float().reshape(-1).cpu().numpy()

        for k, (node, path) in enumerate(zip(leaf_nodes, leaf_paths)):
            # Re-push to reach the leaf state for expansion.
            for nd, ci in path:
                pos.push(_to_real_move(nd.moves[ci], nd.to_play))
            res = pos.result()
            if res != _ONGOING:
                node.is_terminal = True
                node.is_expanded = True
                node.terminal_value = _terminal_value(res, node.to_play)
                v = node.terminal_value
            else:
                moves = _legal_moves_norm(pos)
                if not moves:
                    node.is_terminal = True
                    node.is_expanded = True
                    node.terminal_value = -1.0
                    v = -1.0
                else:
                    pr = probs_np[k][moves]
                    s = pr.sum()
                    pr = pr / s if s > 0 else np.full(len(moves), 1.0 / len(moves), np.float32)
                    node.moves = moves
                    node.P = pr.astype(np.float32)
                    node.N = np.zeros(len(moves), np.float32)
                    node.W = np.zeros(len(moves), np.float32)
                    node.vloss = np.zeros(len(moves), np.float32)
                    node.children = [None] * len(moves)
                    node.is_expanded = True
                    v = float(values_np[k])
            for _ in range(len(path)):
                pos.pop()
            self._backup(path, v)

    # -- scoring / policy --------------------------------------------------- #
    def _root_q(self, root: _PUCTNode) -> np.ndarray:
        """Completed Q at the root for every legal move (root mover's view).

        Visited children use ``W/N``; unvisited use the value estimate, which we
        approximate by 0 (a 0-centered prior) — kept simple per the docstring.

        ``W`` is accumulated by :meth:`_backup`, which flips the sign on the
        first (leaf->edge) step, so ``W/N`` is **already** expressed from this
        node's (the root's) mover perspective — same convention as
        :meth:`_select_child`'s ``q``. No extra negation here (negating again was
        a bug that inverted the completed-Q sign and made Gumbel prefer losing
        moves).
        """
        n = root.N
        return np.where(n > 0, root.W / np.maximum(n, 1.0), 0.0)

    def _sigma(self, q: np.ndarray, root: _PUCTNode) -> np.ndarray:
        max_n = float(root.N.max()) if root.N.size else 0.0
        return (self.c_visit + max_n) * self.c_scale * q

    def _gumbel_scores(self, root, st, survivors) -> np.ndarray:
        q = self._root_q(root)
        sig = self._sigma(q, root)
        idx = np.array(survivors)
        return st.g[idx] + st.logits[idx] + sig[idx]

    def _improved_policy(self, root, st) -> np.ndarray:
        """Completed-value softmax over all legal root moves -> pi[ACTION_DIM]."""
        q = self._root_q(root)
        sig = self._sigma(q, root)
        scaled = st.logits + sig
        scaled = scaled - scaled.max()
        w = np.exp(scaled)
        w /= w.sum()
        pi = np.zeros(ACTION_DIM, dtype=np.float32)
        for i, mv in enumerate(root.moves):
            pi[mv] = w[i]
        return pi

    # -- batched root/leaf eval (shared) ------------------------------------ #
    def _eval_batch(self, positions, nodes, net, want_logits: bool = False):
        """# === BATCHED NN EVAL ===  expand a batch of roots in one forward."""
        device = _net_device(net)
        planes = np.stack([encode(p) for p in positions])
        masks = np.stack([legal_mask(p) for p in positions])
        x = torch.from_numpy(planes).to(device, torch.float32)
        mask_t = torch.from_numpy(masks).to(device)
        was_training = net.training
        net.eval()
        with torch.no_grad():
            logits, value = net(x)  # ONE forward for all B roots
            neg_inf = torch.finfo(logits.dtype).min
            masked = torch.where(mask_t > 0, logits, torch.full_like(logits, neg_inf))
            has_legal = (mask_t > 0).any(-1, keepdim=True)
            masked = torch.where(has_legal, masked, torch.zeros_like(masked))
            probs = torch.softmax(masked, -1)
            probs = torch.where(has_legal.expand_as(probs), probs, torch.zeros_like(probs))
        if was_training:
            net.train()
        probs_np = probs.detach().float().cpu().numpy()
        values_np = value.detach().float().reshape(-1).cpu().numpy()
        logits_np = logits.detach().float().cpu().numpy()

        for k, (pos, node) in enumerate(zip(positions, nodes)):
            res = pos.result()
            if res != _ONGOING:
                node.is_terminal = True
                node.is_expanded = True
                node.terminal_value = _terminal_value(res, node.to_play)
                continue
            moves = _legal_moves_norm(pos)
            if not moves:
                node.is_terminal = True
                node.is_expanded = True
                node.terminal_value = -1.0
                continue
            pr = probs_np[k][moves]
            s = pr.sum()
            pr = pr / s if s > 0 else np.full(len(moves), 1.0 / len(moves), np.float32)
            node.moves = moves
            node.P = pr.astype(np.float32)
            node.N = np.zeros(len(moves), np.float32)
            node.W = np.zeros(len(moves), np.float32)
            node.vloss = np.zeros(len(moves), np.float32)
            node.children = [None] * len(moves)
            node.is_expanded = True
        return logits_np if want_logits else (probs_np, values_np)

    # PUCT helpers reused for in-subtree descent.
    def _select_child(self, node: _PUCTNode) -> int:
        n = node.N + node.vloss
        sqrt_total = math.sqrt(float(n.sum()))
        q = np.where(n > 0, (node.W - node.vloss) / np.maximum(n, 1.0), 0.0)
        u = q + self.c_puct * node.P * sqrt_total / (1.0 + n)
        return int(np.argmax(u))

    def _backup(self, path, leaf_value: float) -> None:
        v = leaf_value
        for node, child_i in reversed(path):
            v = -v
            node.N[child_i] += 1.0
            node.W[child_i] += v
            node.vloss[child_i] -= self.virtual_loss


class _GumbelState:
    __slots__ = ("logits", "g", "cand", "survivors", "is_empty")

    def __init__(self, logits=None, g=None, cand=None):
        self.logits = logits
        self.g = g
        self.cand = cand
        self.survivors = list(cand) if cand is not None else []
        self.is_empty = logits is None

    @classmethod
    def empty(cls) -> "_GumbelState":
        s = cls()
        s.is_empty = True
        return s


# --------------------------------------------------------------------------- #
# Terminal-result handling                                                    #
# --------------------------------------------------------------------------- #
# Result codes mirror xqai._xqcore (§3). Defined locally so this module never
# needs to import the (possibly unbuilt) C++ extension.
_ONGOING = 0
_RED_WIN = 1
_BLACK_WIN = 2
_DRAW = 3


def _terminal_value(result: int, to_play: int) -> float:
    """Game-result value from ``to_play``'s perspective (RED=0, BLACK=1)."""
    if result == _DRAW or result == _ONGOING:
        return 0.0
    winner_is_red = result == _RED_WIN
    mover_is_red = to_play == 0
    return 1.0 if winner_is_red == mover_is_red else -1.0


__all__ = ["PUCTPlanner", "GumbelPlanner"]
