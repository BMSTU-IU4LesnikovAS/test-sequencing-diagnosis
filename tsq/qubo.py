"""Quadratic (QUBO) surrogate of the static-order test-sequencing cost and
two physics-inspired solvers for it: a continuous Hopfield-Tank network and
ballistic simulated bifurcation (bSB).

Encoding: x[j, k] = 1 if test j is at priority position k (n x n permutation
matrix, n^2 binary variables).

Surrogate (pairwise / linear-ordering approximation). In a static-order policy
test j is applied for fault i only if, when j is reached, the faults still
confused with i are split by j. Keeping only pairwise interactions, the
probability that j is applied is approximated by
        a_j  ~  1 - sum_{j' before j} r[j, j'],
where r[j, j'] = 1 - P(j still splits the ambiguity set left by j' alone).
The surrogate cost is therefore
        E(order) = sum_j c_j - sum_{j' before j} c_j r[j, j'],
a linear ordering problem, which is quadratic in x through
before(j', j) = sum_{k' < k} x[j', k'] x[j, k].
One-hot row/column penalties make every minimum a permutation.
"""
from __future__ import annotations

import time

import numpy as np
from scipy.optimize import linear_sum_assignment

from .problems import Problem
from .solvers import Ctx, order_cost


def reduction_matrix(pr: Problem) -> np.ndarray:
    """r[j, jp] = 1 - P(test j still splits the faults left together by test jp)."""
    D, p = pr.D, pr.p
    n = pr.n
    r = np.zeros((n, n))
    for jp in range(n):
        for j in range(n):
            if j == jp:
                r[j, jp] = 1.0
                continue
            applied = 0.0
            for g in (D[:, jp], ~D[:, jp]):
                col = D[g, j]
                if col.any() and (~col).any():
                    applied += p[g].sum()
            r[j, jp] = 1.0 - applied
    return r


def surrogate_cost(order: list[int], c: np.ndarray, r: np.ndarray) -> float:
    pos = np.empty(len(order), int)
    pos[order] = np.arange(len(order))
    before = pos[None, :] < pos[:, None]       # before[j, jp]: jp is before j
    return float(c.sum() - (c[:, None] * r * before).sum())


def build_qubo(pr: Problem, penalty: float | None = None):
    """Return (Q, const) with energy x^T Q x + const over x = vec(x[j, k]) (row-major),
    Q upper-triangular-free symmetric form; objective scaled to max |coef| = 1."""
    n = pr.n
    c = np.asarray(pr.c, float)
    r = reduction_matrix(pr)
    W = c[:, None] * r                          # W[j, jp] gained when jp before j
    np.fill_diagonal(W, 0.0)
    scale = W.max() if W.max() > 1e-9 else 1.0   # guard: degenerate surrogate (all r = 0)
    W = W / scale
    N = n * n
    Q = np.zeros((N, N))
    idx = lambda j, k: j * n + k
    # objective: - sum_{j != jp} W[j, jp] sum_{kp < k} x[jp, kp] x[j, k]
    for j in range(n):
        for jp in range(n):
            if j == jp or W[j, jp] == 0:
                continue
            for k in range(n):
                for kp in range(k):
                    a, b = idx(jp, kp), idx(j, k)
                    Q[a, b] -= W[j, jp] / 2
                    Q[b, a] -= W[j, jp] / 2
    if penalty is None:
        penalty = 2.0 * W.sum(axis=1).max() + 1.0
    # one-hot penalties: A * sum_rows (sum x - 1)^2 + A * sum_cols (sum x - 1)^2
    # (sum x - 1)^2 = sum_a x_a + 2 sum_{a<b} x_a x_b - 2 sum x_a + 1  (x binary)
    for j in range(n):
        for k in range(n):
            a = idx(j, k)
            Q[a, a] += penalty * (-1.0) * 2          # row and column linear parts
            for kk in range(n):
                if kk != k:
                    Q[a, idx(j, kk)] += penalty
            for jj in range(n):
                if jj != j:
                    Q[a, idx(jj, k)] += penalty
    const = 2.0 * n * penalty
    return Q, const, {"scale": scale, "penalty": penalty, "r": r}


def decode(x: np.ndarray, n: int) -> tuple[list[int], bool]:
    """Turn a (possibly fractional or infeasible) n*n vector into a priority order.
    Returns the order (via an assignment repair) and whether x was already a
    valid permutation matrix."""
    X = x.reshape(n, n)
    B = (X > 0.5).astype(int)
    feasible = bool((B.sum(0) == 1).all() and (B.sum(1) == 1).all())
    rows, cols = linear_sum_assignment(-X)
    order = [0] * n
    for j, k in zip(rows, cols):
        order[k] = j
    return order, feasible


# ---------------------------------------------------------------- Hopfield
def hopfield(pr: Problem, seed: int, restarts: int = 10, steps: int = 3000,
             dt: float = 0.01, u0: float = 0.05, tau: float = 1.0,
             penalty_mult: float = 1.0) -> dict:
    """Continuous Hopfield-Tank network minimising V^T Q V. Several random
    restarts run in parallel; the best decoded order (true cost) is returned."""
    t0 = time.perf_counter()
    Q, const, info = build_qubo(pr)
    if penalty_mult != 1.0:
        Q, const, info = build_qubo(pr, penalty=info["penalty"] * penalty_mult)
    n, N = pr.n, pr.n * pr.n
    rng = np.random.default_rng(seed)
    U = rng.uniform(-0.01, 0.01, size=(restarts, N)) + u0 * np.arctanh(2 / n - 1)
    G = Q + Q.T
    for _ in range(steps):
        V = 0.5 * (1 + np.tanh(U / u0))
        U += dt * (-U / tau - V @ G)
    V = 0.5 * (1 + np.tanh(U / u0))
    ctx = Ctx(pr)
    best, best_order, feas = np.inf, None, 0
    for v in V:
        order, ok = decode(v, n)
        feas += ok
        cost = order_cost(ctx, order)
        if cost < best:
            best, best_order = cost, order
    return {"cost": best, "order": best_order, "feasible_frac": feas / restarts,
            "time": time.perf_counter() - t0}


# ---------------------------------------------------------------- simulated bifurcation
def qubo_to_ising(Q: np.ndarray):
    """x = (s + 1) / 2.  x^T Q x = s^T (Q/4) s + s^T (Q 1 / 2) + const."""
    Qs = (Q + Q.T) / 2
    J = -Qs / 2                      # E_ising = -1/2 s^T J s - h^T s
    np.fill_diagonal(J, 0.0)
    h = -(Qs.sum(axis=1)) / 2
    return J, h


def simulated_bifurcation(pr: Problem, seed: int, agents: int = 10, steps: int = 2000,
                          dt: float = 0.5, penalty_mult: float = 1.0) -> dict:
    """Ballistic simulated bifurcation (Goto et al., 2021) on the Ising form of
    the QUBO; several agents in parallel, best decoded order returned."""
    t0 = time.perf_counter()
    Q, const, info = build_qubo(pr)
    if penalty_mult != 1.0:
        Q, const, info = build_qubo(pr, penalty=info["penalty"] * penalty_mult)
    J, h = qubo_to_ising(Q)
    N = J.shape[0]
    rng = np.random.default_rng(seed)
    a0 = 1.0
    rms = np.sqrt((J ** 2).sum() / (N * (N - 1)))
    c0 = 0.5 / (rms * np.sqrt(N))
    x = rng.uniform(-0.1, 0.1, size=(agents, N))
    y = rng.uniform(-0.1, 0.1, size=(agents, N))
    for t in range(steps):
        a = a0 * t / steps
        y += dt * (-(a0 - a) * x + c0 * (x @ J + h))
        x += dt * a0 * y
        wall = np.abs(x) > 1
        x[wall] = np.sign(x[wall])
        y[wall] = 0.0
    s = np.sign(x)
    s[s == 0] = 1
    xs = (s + 1) / 2
    ctx = Ctx(pr)
    n = pr.n
    best, best_order, feas = np.inf, None, 0
    for v in xs:
        order, ok = decode(v, n)
        feas += ok
        cost = order_cost(ctx, order)
        if cost < best:
            best, best_order = cost, order
    return {"cost": best, "order": best_order, "feasible_frac": feas / agents,
            "time": time.perf_counter() - t0}


# ---------------------------------------------------------------- surrogate reference
def sa_surrogate(pr: Problem, seed: int, iters: int = 20000) -> dict:
    """Annealing directly on the surrogate (permutation space, no penalties):
    shows how good the surrogate optimum itself is, independent of the
    Hopfield/bSB solvers."""
    t0 = time.perf_counter()
    rng = np.random.default_rng(seed)
    c = np.asarray(pr.c, float)
    r = reduction_matrix(pr)
    n = pr.n
    order = rng.permutation(n).tolist()
    cur = surrogate_cost(order, c, r)
    best, best_order = cur, list(order)
    T0 = 0.05 * abs(cur) + 1e-9
    for k in range(iters):
        T = T0 * (1e-3) ** (k / iters)
        a, b = rng.integers(0, n, size=2)
        if a == b:
            continue
        cand = list(order)
        x = cand.pop(a)
        cand.insert(b, x)
        v = surrogate_cost(cand, c, r)
        if v <= cur or rng.random() < np.exp(-(v - cur) / T):
            order, cur = cand, v
            if cur < best:
                best, best_order = cur, list(order)
    ctx = Ctx(pr)
    return {"cost": order_cost(ctx, best_order), "surrogate": best, "order": best_order,
            "time": time.perf_counter() - t0}
