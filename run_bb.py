"""Branch-and-bound variants on the same instances as the main runs.
usage: python3 run_bb.py iscas | synthetic"""
import json, os, sys
from multiprocessing import Pool
from tsq.circuits import parse_verilog
from tsq.problems import CircuitPool, synthetic
from tsq.solvers import exact_bb

LIMIT = float(os.environ.get("BB_LIMIT", "60"))
_POOLS = {}


def solve(pr, key):
    out = {"key": key, "instance": pr.name, "m": pr.m, "n": pr.n, "density": float(pr.D.mean())}
    for b in ("none", "huffman", "both"):
        out[b] = exact_bb(pr, bound=b, time_limit=LIMIT)
    return out


def isc(args):
    c, m, sk, s = args
    if c not in _POOLS:
        _POOLS[c] = CircuitPool(parse_verilog(f"data/iscas85/{c}.v"), 256, seed=0)
    pr = _POOLS[c].instance(m, m + 5, seed=1000 * m + s, skew=sk, cost_mode="random")
    o = solve(pr, repr(args)); o["circuit"] = c; o["skew"] = sk
    return o


N_FOR_M = {60: 40, 80: 50}


def syn(args):
    m, sk, co, s = args
    pr = synthetic(m, N_FOR_M[m], seed=1000 * m + s, skew=sk, cost_mode=co)
    o = solve(pr, repr(args)); o["skew"] = sk; o["cost_mode"] = co
    return o


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "iscas":
        jobs = [(c, m, sk, s) for c in ("c432", "c499", "c880", "c1355", "c1908") for m in (20, 30, 40)
                for sk in ("moderate", "skewed") for s in range(10 if m < 40 else 5)]
        fn, path = isc, "results/bb_iscas.jsonl"
    else:
        jobs = [(m, sk, co, s) for m in (60, 80) for sk in ("uniform", "moderate", "skewed")
                for co in ("unit", "random") for s in range(5)]
        fn, path = syn, "results/bb_synthetic.jsonl"
    done = {json.loads(l)["key"] for l in open(path)} if os.path.exists(path) else set()
    todo = [j for j in jobs if repr(j) not in done]
    print(len(todo), "to run", flush=True)
    with Pool(2) as pool, open(path, "a") as fh:
        for r in pool.imap_unordered(fn, todo):
            fh.write(json.dumps(r) + "\n"); fh.flush()
