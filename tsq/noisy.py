"""Test sequencing with imperfect (unreliable) tests.

Model (Raghavan, Shakeri, Pattipati 1999 type): if fault i is present, test j
reports "fail" with probability 1 - miss_j when D[i, j] = 1 (detection) and
with probability fa_j when D[i, j] = 0 (false alarm); outcomes are
conditionally independent given the fault. Each test is applied at most once
on a path. At any node the strategy may stop and declare the fault with the
largest posterior probability. Objective: expected test cost plus
L * P(wrong fault declared).

Every function works with the unnormalised weight vector w_i = p_i * P(outcomes
so far | fault i); W = sum(w) is the probability of reaching the node, so all
costs below are already weighted by the probability of the node.
"""
from __future__ import annotations

import math
import time

import numpy as np

from .problems import Problem


class NoisyModel:
    def __init__(self, pr: Problem, miss: float, fa: float, L: float):
        self.pr = pr
        self.m, self.n = pr.m, pr.n
        self.p = np.asarray(pr.p, float)
        self.c = np.asarray(pr.c, float)
        D = np.asarray(pr.D, bool)
        # P(fail | fault i) for every test j
        self.pf = np.where(D, 1.0 - miss, fa)
        self.L = float(L)
        self.miss, self.fa = miss, fa

    def stop_cost(self, w: np.ndarray) -> float:
        return self.L * float(w.sum() - w.max())


class Budget(Exception):
    pass


def exact_noisy(M: NoisyModel, time_limit: float = 600.0, max_states: int = 5_000_000) -> dict:
    """Optimal strategy by DP over information states (set of applied tests and
    their outcomes), with branch-and-bound pruning against the stop option."""
    memo: dict[tuple[int, int], float] = {}
    t0 = time.perf_counter()
    n = M.n
    pf = M.pf
    c = M.c

    def V(A: int, F: int, w: np.ndarray) -> float:
        key = A | (F << 32)
        v = memo.get(key)
        if v is not None:
            return v
        if len(memo) > max_states or time.perf_counter() - t0 > time_limit:
            raise Budget()
        W = float(w.sum())
        best = M.stop_cost(w)
        if best > 0.0:
            for j in range(n):
                bit = 1 << j
                if A & bit:
                    continue
                v = c[j] * W
                if v >= best:
                    continue
                wf = w * pf[:, j]
                wp = w - wf
                v += V(A | bit, F | bit, wf)
                if v >= best:
                    continue
                v += V(A | bit, F, wp)
                if v < best:
                    best = v
        memo[key] = best
        return best

    try:
        val = V(0, 0, M.p.copy())
        return {"cost": val, "optimal": True, "states": len(memo), "time": time.perf_counter() - t0}
    except Budget:
        return {"cost": None, "optimal": False, "states": len(memo), "time": time.perf_counter() - t0}


# ------------------------------------------------------------------ policies
def _entropy_split(w: np.ndarray, pfj: np.ndarray) -> float:
    """Mutual information between the fault and the outcome of one test."""
    W = w.sum()
    q = w / W
    a = float((q * pfj).sum())                    # P(fail)

    def H(x):
        x = x[x > 0]
        return float(-(x * np.log2(x)).sum())
    if a <= 0 or a >= 1:
        return 0.0
    post_f = q * pfj / a
    post_p = q * (1 - pfj) / (1 - a)
    return H(q) - a * H(post_f) - (1 - a) * H(post_p)


def greedy_choice(M: NoisyModel, A: int, w: np.ndarray):
    """Information gain per unit cost; stop when the expected misdiagnosis loss
    is below the cost of the cheapest remaining test, or no test is informative."""
    W = float(w.sum())
    cand = [j for j in range(M.n) if not A & (1 << j)]
    if not cand:
        return None
    if M.stop_cost(w) <= min(M.c[j] for j in cand) * W:
        return None
    best, arg = 0.0, None
    for j in cand:
        g = _entropy_split(w, M.pf[:, j]) / M.c[j]
        if g > best + 1e-15:
            best, arg = g, j
    return arg


def policy_value(M: NoisyModel, choose, memo=None) -> float:
    """Exact expected cost of a stationary policy choose(A, F, w) -> test or None."""
    memo = {} if memo is None else memo

    def V(A, F, w):
        key = A | (F << 32)
        if key in memo:
            return memo[key]
        j = choose(A, F, w)
        if j is None:
            v = M.stop_cost(w)
        else:
            bit = 1 << j
            wf = w * M.pf[:, j]
            v = M.c[j] * float(w.sum()) + V(A | bit, F | bit, wf) + V(A | bit, F, w - wf)
        memo[key] = v
        return v
    return V(0, 0, M.p.copy())


def greedy_noisy(M: NoisyModel) -> dict:
    t0 = time.perf_counter()
    v = policy_value(M, lambda A, F, w: greedy_choice(M, A, w))
    return {"cost": v, "time": time.perf_counter() - t0}


def rollout_chooser(M: NoisyModel):
    """One-step lookahead over {stop} and every unused test, each completed by
    the greedy base policy."""
    base_memo: dict = {}

    def base_value(A, F, w):
        key = A | (F << 32)
        if key in base_memo:
            return base_memo[key]
        j = greedy_choice(M, A, w)
        if j is None:
            v = M.stop_cost(w)
        else:
            bit = 1 << j
            wf = w * M.pf[:, j]
            v = M.c[j] * float(w.sum()) + base_value(A | bit, F | bit, wf) + base_value(A | bit, F, w - wf)
        base_memo[key] = v
        return v

    def choose(A, F, w):
        best, arg = M.stop_cost(w), None
        W = float(w.sum())
        for j in range(M.n):
            bit = 1 << j
            if A & bit:
                continue
            wf = w * M.pf[:, j]
            v = M.c[j] * W + base_value(A | bit, F | bit, wf) + base_value(A | bit, F, w - wf)
            if v < best - 1e-15:
                best, arg = v, j
        return arg
    return choose


def rollout_noisy(M: NoisyModel) -> dict:
    t0 = time.perf_counter()
    v = policy_value(M, rollout_chooser(M))
    return {"cost": v, "time": time.perf_counter() - t0}


def perfect_tree_noisy(M: NoisyModel, tree: dict) -> dict:
    """Apply the optimal tree designed for perfect tests (tree: ambiguity-set
    bitmask -> test) to noisy outcomes; at a leaf the remaining fault is declared."""
    t0 = time.perf_counter()
    D = np.asarray(M.pr.D, bool)
    fail = [sum(1 << i for i in range(M.m) if D[i, j]) for j in range(M.n)]

    def V(S, w):
        if S & (S - 1) == 0:
            i = S.bit_length() - 1
            return M.L * float(w.sum() - w[i])
        j = tree[S]
        wf = w * M.pf[:, j]
        return (M.c[j] * float(w.sum()) + V(S & fail[j], wf) + V(S & ~fail[j], w - wf))
    v = V((1 << M.m) - 1, M.p.copy())
    return {"cost": v, "time": time.perf_counter() - t0}


def perfect_optimal_tree(pr: Problem) -> dict:
    """Optimal tree for perfect tests by plain memoised recursion (small m)."""
    D = np.asarray(pr.D, bool)
    m, n = pr.m, pr.n
    p = np.asarray(pr.p, float)
    c = np.asarray(pr.c, float)
    fail = [sum(1 << i for i in range(m) if D[i, j]) for j in range(n)]
    memo, arg = {}, {}

    def P(S):
        return float(sum(p[i] for i in range(m) if S >> i & 1))

    def J(S):
        if S & (S - 1) == 0:
            return 0.0
        if S in memo:
            return memo[S]
        best, a = math.inf, None
        PS = P(S)
        for j in range(n):
            A, B = S & fail[j], S & ~fail[j]
            if A and B:
                v = c[j] * PS + J(A) + J(B)
                if v < best:
                    best, a = v, j
        memo[S], arg[S] = best, a
        return best
    J((1 << m) - 1)
    return arg


def misdiagnosis_prob(M: NoisyModel, choose=None, tree=None) -> tuple[float, float]:
    """(expected test cost, P(misdiagnosis)) of a policy, by running it with L = 0
    bookkeeping separately."""
    L = M.L
    M.L = 1.0
    if tree is not None:
        err_plus = perfect_tree_noisy(M, tree)["cost"]
        M.L = 0.0
        tc = perfect_tree_noisy(M, tree)["cost"]
    else:
        # the policy itself must be evaluated with the true L (its decisions depend on L)
        M.L = L
        memo_choice = {}

        def frozen(A, F, w):
            k = A | (F << 32)
            if k not in memo_choice:
                memo_choice[k] = choose(A, F, w)
            return memo_choice[k]
        policy_value(M, frozen)
        M.L = 1.0
        err_plus = policy_value(M, frozen)
        M.L = 0.0
        tc = policy_value(M, frozen)
    M.L = L
    return tc, err_plus - tc


def model_from_pf(p: np.ndarray, c: np.ndarray, pf: np.ndarray, L: float, D=None) -> NoisyModel:
    """Noisy model with an arbitrary P(fail | fault, test) matrix (e.g. estimated
    by Monte Carlo from component tolerances)."""
    from .problems import Problem
    D = (pf >= 0.5) if D is None else D
    pr = Problem("custom", np.asarray(p, float), np.asarray(c, float), np.asarray(D, bool), {})
    M = NoisyModel(pr, 0.0, 0.0, L)
    M.pf = np.asarray(pf, float)
    return M


def dictionary_policy(M: NoisyModel) -> dict:
    """Fault-dictionary baseline: apply every test, then declare the fault with
    the largest posterior. Exact expected cost by enumerating outcome patterns."""
    n = M.n
    W = M.p.copy()
    tc = float(M.c.sum())
    err = 0.0
    # enumerate outcome vectors iteratively to keep memory small
    def rec(j, w):
        nonlocal err
        if j == n:
            err += float(w.sum() - w.max())
            return
        wf = w * M.pf[:, j]
        rec(j + 1, wf)
        rec(j + 1, w - wf)
    rec(0, W)
    return {"cost": tc + M.L * err, "test_cost": tc, "p_err": err}
