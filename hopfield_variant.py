"""Robustness check: Hopfield-Tank network in the textbook form (no
self-connections, linear QUBO terms as input biases) versus the form used in
the main experiments (full Q including its diagonal as quadratic terms)."""
import json, sys
from multiprocessing import Pool
import numpy as np
from tsq.problems import synthetic
from tsq.qubo import build_qubo, decode
from tsq.solvers import Ctx, exact, order_cost

N_FOR_M = {10: 15, 20: 25, 30: 30}


def hop_textbook(pr, seed, restarts=10, steps=3000, dt=0.01, u0=0.05, tau=1.0):
    Q, const, info = build_qubo(pr)
    b = np.diag(Q).copy()
    Qo = Q - np.diag(b)
    G = Qo + Qo.T
    n, N = pr.n, pr.n * pr.n
    rng = np.random.default_rng(seed)
    U = rng.uniform(-0.01, 0.01, size=(restarts, N)) + u0 * np.arctanh(2 / n - 1)
    for _ in range(steps):
        V = 0.5 * (1 + np.tanh(U / u0))
        U += dt * (-U / tau - V @ G - b)
    V = 0.5 * (1 + np.tanh(U / u0))
    ctx = Ctx(pr)
    best, feas = np.inf, 0
    for v in V:
        order, ok = decode(v, n)
        feas += ok
        best = min(best, order_cost(ctx, order))
    return best, feas / restarts


def job(args):
    m, skew, s = args
    pr = synthetic(m, N_FOR_M[m], seed=500000 + 1000 * m + s, skew=skew, cost_mode="random")
    ex = exact(pr)["cost"]
    r = [hop_textbook(pr, k) for k in range(3)]
    return {"key": repr(args), "m": m, "exact": ex, "hop_tb": [x[0] for x in r], "hop_tb_feas": [x[1] for x in r]}


if __name__ == "__main__":
    jobs = [(m, sk, s) for m in (10, 20, 30) for sk in ("moderate", "skewed") for s in range(10)]
    with Pool(1) as p, open("results/qubo_hopfield_textbook.jsonl", "w") as fh:
        for r in p.imap_unordered(job, jobs):
            fh.write(json.dumps(r) + "\n"); fh.flush()
