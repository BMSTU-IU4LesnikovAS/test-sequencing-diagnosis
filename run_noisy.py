"""Imperfect tests: optimal strategy vs heuristics vs the perfect-test optimal
tree applied to noisy outcomes. usage: python3 run_noisy.py <workers>"""
import json, os, sys
from multiprocessing import Pool
import numpy as np
from tsq.circuits import parse_verilog
from tsq.problems import CircuitPool, synthetic
from tsq.noisy import (NoisyModel, exact_noisy, greedy_choice, rollout_chooser, policy_value,
                       perfect_optimal_tree, perfect_tree_noisy, misdiagnosis_prob)

PATH = "results/noisy.jsonl"
M_, N_ = 12, 12
QS = (0.01, 0.02, 0.05, 0.10)
LMULT = 20.0
_POOLS = {}


def make(kind, s):
    if kind == "syn":
        return synthetic(M_, N_, seed=950000 + s, skew="moderate", cost_mode="random")
    c = ("c432", "c499", "c880", "c1355", "c1908")[s % 5]
    if c not in _POOLS:
        _POOLS[c] = CircuitPool(parse_verilog(f"data/iscas85/{c}.v"), 256, seed=0)
    return _POOLS[c].instance(M_, N_, seed=960000 + s, skew="moderate", cost_mode="random")


def job(args):
    kind, s, q, lm = args
    pr = make(kind, s)
    M = NoisyModel(pr, q, q, lm * float(np.mean(pr.c)))
    ex = exact_noisy(M)
    tree = perfect_optimal_tree(pr)
    out = {"key": repr(args), "set": kind, "s": s, "q": q, "Lmult": lm, "m": pr.m, "n": pr.n,
           "density": float(pr.D.mean()), "exact": ex}
    pt = perfect_tree_noisy(M, tree)
    tc, er = misdiagnosis_prob(M, tree=tree)
    out["perfect_tree"] = {"cost": pt["cost"], "test_cost": tc, "p_err": er}
    import gc; gc.collect()
    for name, ch in (("greedy", lambda A, F, w: greedy_choice(M, A, w)), ("rollout", rollout_chooser(M))):
        tc, er = misdiagnosis_prob(M, choose=ch)
        out[name] = {"cost": tc + M.L * er, "test_cost": tc, "p_err": er}
        gc.collect()
    # optimal policy's error probability: re-run DP storing decisions
    out["exact_detail"] = optimal_detail(M)
    return out


def optimal_detail(M):
    """Decisions of the optimal policy (no pruning) -> its test cost and error probability."""
    from tsq.noisy import policy_value
    memo, dec = {}, {}
    n = M.n

    def V(A, F, w):
        k = A | (F << 32)
        if k in memo:
            return memo[k]
        W = float(w.sum())
        best, arg = M.stop_cost(w), None
        for j in range(n):
            bit = 1 << j
            if A & bit:
                continue
            wf = w * M.pf[:, j]
            v = M.c[j] * W + V(A | bit, F | bit, wf) + V(A | bit, F, w - wf)
            if v < best - 1e-12:
                best, arg = v, j
        memo[k], dec[k] = best, arg
        return best
    V(0, 0, M.p.copy())
    tc, er = misdiagnosis_prob(M, choose=lambda A, F, w: dec[A | (F << 32)])
    return {"test_cost": tc, "p_err": er}


if __name__ == "__main__":
    jobs = [(k, s, q, LMULT) for k in ("syn", "isc") for s in range(20) for q in QS]
    jobs += [(k, s, 0.05, lm) for k in ("syn", "isc") for s in range(10) for lm in (5.0, 100.0)]
    done = {json.loads(l)["key"] for l in open(PATH)} if os.path.exists(PATH) else set()
    todo = [j for j in jobs if repr(j) not in done]
    print(len(todo), "to run", flush=True)
    with Pool(int(sys.argv[1]), maxtasksperchild=4) as pool, open(PATH, "a") as fh:
        for r in pool.imap_unordered(job, todo):
            fh.write(json.dumps(r) + "\n"); fh.flush()
