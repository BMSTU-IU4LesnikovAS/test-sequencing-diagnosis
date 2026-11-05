"""AO* (Pattipati & Alexandridis 1990, Huffman heuristic) versus branch and
bound (scheme B + Huffman) on the same instances and under the same limits."""
import json, os, sys
from multiprocessing import Pool
import run_bb
from tsq.solvers import ao_star, exact_bb

LIMIT = 60.0
PATH = "results/ao_compare.jsonl"


def solve(pr, key, extra):
    out = {"key": key, "instance": pr.name, "m": pr.m, "n": pr.n, "density": float(pr.D.mean()), **extra}
    out["ao"] = ao_star(pr, time_limit=LIMIT)
    out["bb"] = exact_bb(pr, bound="huffman", time_limit=LIMIT)
    return out


def isc(args):
    c, m, sk, s = args
    if c not in run_bb._POOLS:
        run_bb._POOLS[c] = run_bb.CircuitPool(run_bb.parse_verilog(f"data/iscas85/{c}.v"), 256, seed=0)
    pr = run_bb._POOLS[c].instance(m, m + 5, seed=1000 * m + s, skew=sk, cost_mode="random")
    return solve(pr, repr(("isc",) + args), {"set": "ISCAS"})


def syn(args):
    m, sk, co, s = args
    pr = run_bb.synthetic(m, run_bb.N_FOR_M[m], seed=1000 * m + s, skew=sk, cost_mode=co)
    return solve(pr, repr(("syn",) + args), {"set": "synthetic"})


def job(a):
    return isc(a[1:]) if a[0] == "isc" else syn(a[1:])


if __name__ == "__main__":
    jobs = [("isc", c, m, sk, s) for c in ("c432", "c499", "c880", "c1355", "c1908") for m in (30, 40)
            for sk in ("moderate", "skewed") for s in range(10 if m < 40 else 5)]
    jobs += [("syn", m, sk, co, s) for m in (60, 80) for sk in ("uniform", "moderate", "skewed")
             for co in ("unit", "random") for s in range(5)]
    done = {json.loads(l)["key"] for l in open(PATH)} if os.path.exists(PATH) else set()
    todo = [j for j in jobs if repr(j) not in done]
    print(len(todo), "to run", flush=True)
    with Pool(2, maxtasksperchild=4) as pool, open(PATH, "a") as fh:
        for r in pool.imap_unordered(job, todo):
            fh.write(json.dumps(r) + "\n"); fh.flush()
