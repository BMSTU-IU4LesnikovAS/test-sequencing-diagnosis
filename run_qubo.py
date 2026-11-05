"""Hopfield / QUBO analysis.

decomposition: for each instance, the loss of the QUBO route is split into
  (1) static-order restriction  : best static order vs. optimal adaptive tree
  (2) surrogate error           : surrogate optimum vs. best static order
  (3) solver error              : Hopfield / bSB vs. surrogate optimum
penalty: sensitivity of Hopfield and bSB to the one-hot penalty weight.

usage: python3 run_qubo.py decomposition | penalty
"""
import json
import os
import sys
from multiprocessing import Pool

import numpy as np
from scipy.stats import spearmanr

from tsq.problems import synthetic
from tsq.qubo import (hopfield, reduction_matrix, sa_surrogate, simulated_bifurcation,
                      surrogate_cost)
from tsq.solvers import Ctx, exact, greedy, order_cost, sa_order

N_FOR_M = {10: 15, 20: 25, 30: 30}
SEEDS = 3


def decomposition(args):
    m, skew, s = args
    pr = synthetic(m, N_FOR_M[m], seed=500000 + 1000 * m + s, skew=skew, cost_mode="random")
    ex = exact(pr)["cost"]
    best_static = min(sa_order(pr, seed=k, iters=20000)["cost"] for k in range(SEEDS))
    surr = [sa_surrogate(pr, seed=k, iters=20000) for k in range(SEEDS)]
    surr_best = min(surr, key=lambda r: r["surrogate"])
    hop = [hopfield(pr, seed=k) for k in range(SEEDS)]
    sb = [simulated_bifurcation(pr, seed=k) for k in range(SEEDS)]
    ctx = Ctx(pr)
    r = reduction_matrix(pr)
    rng = np.random.default_rng(s)
    A, B = [], []
    for _ in range(300):
        o = rng.permutation(pr.n).tolist()
        A.append(surrogate_cost(o, pr.c, r))
        B.append(order_cost(ctx, o))
    return {"key": repr(args), "m": m, "n": pr.n, "skew": skew, "exact": ex,
            "greedy_split": greedy(pr, "split")["cost"], "best_static": best_static,
            "surrogate_opt_true_cost": surr_best["cost"],
            "hopfield": [h["cost"] for h in hop], "hopfield_feas": [h["feasible_frac"] for h in hop],
            "hopfield_time": [h["time"] for h in hop],
            "sb": [b["cost"] for b in sb], "sb_feas": [b["feasible_frac"] for b in sb],
            "sb_time": [b["time"] for b in sb],
            "spearman_surrogate_true": float(spearmanr(A, B).statistic)}


def penalty(args):
    mult, s = args
    pr = synthetic(20, 25, seed=600000 + s, skew="moderate", cost_mode="random")
    ex = exact(pr)["cost"]
    h = hopfield(pr, seed=0, penalty_mult=mult)
    b = simulated_bifurcation(pr, seed=0, penalty_mult=mult)
    return {"key": repr(args), "mult": mult, "exact": ex, "hopfield": h["cost"],
            "hopfield_feas": h["feasible_frac"], "sb": b["cost"], "sb_feas": b["feasible_frac"]}


def main():
    mode = sys.argv[1]
    path = f"results/qubo_{mode}.jsonl"
    if mode == "decomposition":
        jobs = [(m, sk, s) for m in (10, 20, 30) for sk in ("moderate", "skewed") for s in range(10)]
        fn = decomposition
    else:
        jobs = [(mult, s) for mult in (0.25, 0.5, 1.0, 2.0, 4.0) for s in range(10)]
        fn = penalty
    done = set()
    if os.path.exists(path):
        done = {json.loads(l)["key"] for l in open(path)}
    todo = [j for j in jobs if repr(j) not in done]
    print(f"{len(todo)} to run", flush=True)
    with Pool(2) as pool, open(path, "a") as fh:
        for res in pool.imap_unordered(fn, todo):
            fh.write(json.dumps(res) + "\n")
            fh.flush()


if __name__ == "__main__":
    main()
