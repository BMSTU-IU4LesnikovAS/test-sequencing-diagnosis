"""(a) Trend of the non-quadratic share with n (n = 6, 7, 8).
(b) Data-driven QUBO: fit a quadratic on K random orders, take its minimiser,
    compare its true cost with the optimum and with the greedy rule.
"""
import itertools, json, sys
import numpy as np
from tsq.circuits import parse_verilog
from tsq.problems import CircuitPool, synthetic
from tsq.solvers import Ctx, exact, greedy, order_cost
from qubo_theory import design


def instances(n, k_syn, k_isc, base):
    out = [("synthetic", synthetic(10, n, seed=base + s, cost_mode="random")) for s in range(k_syn)]
    pools = [CircuitPool(parse_verilog(f"data/iscas85/{c}.v"), 256, seed=0)
             for c in ("c432", "c499", "c880", "c1355", "c1908")]
    s = 0
    while sum(1 for k, _ in out if k == "iscas") < k_isc and s < 50:
        pr = pools[s % 5].instance(10, n, seed=base + s, cost_mode="random")
        if pr.n == n:
            out.append(("iscas", pr))
        s += 1
    return out


mode = sys.argv[1]
res = []
if mode == "trend":
    n = 8
    perms, X = design(n)
    for kind, pr in instances(n, 4, 4, 810000):
        ctx = Ctx(pr)
        y = np.array([order_cost(ctx, list(p)) for p in perms])
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        fit = X @ coef
        r2 = 1 - ((y - fit) ** 2).sum() / ((y - y.mean()) ** 2).sum()
        res.append({"kind": kind, "n": n, "r2": float(r2)})
        print(kind, n, round(r2, 4), flush=True)
    json.dump(res, open("results/qubo_theory_n8.json", "w"))
else:
    n = 7
    perms, X = design(n)
    rng = np.random.default_rng(0)
    for kind, pr in instances(n, 10, 10, 820000):
        ctx = Ctx(pr)
        y = np.array([order_cost(ctx, list(p)) for p in perms])
        opt = exact(pr)["cost"]
        g = greedy(pr, "split")["cost"]
        row = {"kind": kind, "greedy_gap": 100 * (g / opt - 1), "best_static_gap": 100 * (y.min() / opt - 1)}
        for K in (50, 100, 200, 500, 1000, 2000):
            gq, gr = [], []
            for rep in range(5):
                idx = rng.choice(len(perms), size=K, replace=False)
                coef, *_ = np.linalg.lstsq(X[idx], y[idx], rcond=None)
                fit = X @ coef
                gq.append(100 * (y[int(np.argmin(fit))] / opt - 1))
                # the best order among the same K evaluated samples (plain random search)
                gr.append(100 * (y[idx].min() / opt - 1))
            row[f"K{K}"] = float(np.mean(gq))
            row[f"rand{K}"] = float(np.mean(gr))
            row[f"K{K}_reps_qubo_worse"] = int(sum(a > b + 1e-9 for a, b in zip(gq, gr)))
        res.append(row)
        print(kind, {k: round(v, 1) for k, v in row.items() if isinstance(v, float)}, flush=True)
    json.dump(res, open("results/qubo_datadriven.json", "w"))
