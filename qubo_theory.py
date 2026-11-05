"""How well can ANY QUBO represent the static-order test-sequencing cost?

For n tests, every QUBO over the permutation encoding x[j,k] restricted to
permutations is a polynomial of degree <= 2 in x. The space of such functions
on S_n is a strict subspace of all functions on S_n when n >= 4. We project the
true cost J(pi) (over all n! orders) onto that subspace by least squares and
report:
  - share of the cost variance outside the quadratic subspace (higher-order
    interactions that no QUBO can represent);
  - true cost of the minimiser of the best-fitting quadratic (best case for any
    data-driven QUBO, e.g. AutoQubo), relative to the best static order and to
    the optimal adaptive tree.
Control: an asymmetric TSP path cost, which is exactly quadratic, must give a
zero residual.
"""
import itertools
import json

import numpy as np

from tsq.circuits import parse_verilog
from tsq.problems import CircuitPool, synthetic
from tsq.solvers import Ctx, exact, order_cost


def design(n):
    perms = list(itertools.permutations(range(n)))
    N = n * n
    pairs = [(a, b) for a in range(N) for b in range(a, N)]
    X = np.zeros((len(perms), len(pairs) + 1))
    X[:, -1] = 1.0
    for r, perm in enumerate(perms):
        x = np.zeros(N)
        for k, j in enumerate(perm):
            x[j * n + k] = 1.0
        X[r, :-1] = [x[a] * x[b] for a, b in pairs]
    return perms, X


def analyse(y, X):
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    fit = X @ coef
    var = ((y - y.mean()) ** 2).sum()
    resid = ((y - fit) ** 2).sum()
    return 1.0 - resid / var if var > 0 else 1.0, fit


def main():
    out = []
    for n in (6, 7):
        perms, X = design(n)
        rank = np.linalg.matrix_rank(X)
        insts = []
        for s in range(15):
            insts.append(("synthetic", synthetic(10, n, seed=800000 + 100 * n + s, cost_mode="random")))
        pools = {c: CircuitPool(parse_verilog(f"data/iscas85/{c}.v"), 256, seed=0)
                 for c in ("c432", "c499", "c880", "c1355", "c1908")}
        for c, pool in pools.items():
            for s in range(3):
                pr = pool.instance(10, n, seed=800000 + 100 * n + s, cost_mode="random")
                if pr.n == n:
                    insts.append(("iscas", pr))
        for kind, pr in insts:
            ctx = Ctx(pr)
            y = np.array([order_cost(ctx, list(p)) for p in perms])
            r2, fit = analyse(y, X)
            best_static = y.min()
            arg_fit = int(np.argmin(fit))
            opt = exact(pr)["cost"]
            # control: random asymmetric TSP path cost (exactly quadratic)
            rng = np.random.default_rng(1)
            Dm = rng.uniform(1, 10, size=(n, n))
            yt = np.array([sum(Dm[p[k], p[k + 1]] for k in range(n - 1)) for p in perms])
            r2_ctrl, _ = analyse(yt, X)
            out.append({"kind": kind, "n": n, "m": pr.m, "rank": int(rank), "nperm": len(perms),
                        "r2": r2, "nonquad_share": 1 - r2, "r2_control": r2_ctrl,
                        "fit_min_gap_static": 100 * (y[arg_fit] / best_static - 1),
                        "fit_min_gap_opt": 100 * (y[arg_fit] / opt - 1),
                        "best_static_gap_opt": 100 * (best_static / opt - 1),
                        "fit_min_is_best_static": bool(y[arg_fit] <= best_static + 1e-12)})
            print(kind, n, round(r2, 3), round(out[-1]["fit_min_gap_opt"], 1), flush=True)
    json.dump(out, open("results/qubo_theory.json", "w"))


if __name__ == "__main__":
    main()
