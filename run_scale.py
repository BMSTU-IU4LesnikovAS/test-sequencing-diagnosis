"""Scaling runs: branch and bound (Huffman bound) on larger instances, with
greedy and rollout gaps where the optimum is found.
usage: python3 run_scale.py <workers>"""
import json, os, sys
from multiprocessing import Pool
from tsq.circuits import parse_verilog
from tsq.problems import CircuitPool, synthetic
from tsq.solvers import exact_bb, greedy, rollout

LIMIT = float(os.environ.get("BB_LIMIT", "300"))
PATH = "results/scale.jsonl"
_POOLS = {}


def solve(pr, key, extra):
    bb = exact_bb(pr, bound="huffman", time_limit=LIMIT)
    out = {"key": key, "instance": pr.name, "m": pr.m, "n": pr.n,
           "density": float(pr.D.mean()), "bb": bb, **extra}
    for name, f in (("greedy_ig", lambda: greedy(pr, "ig")),
                    ("greedy_split", lambda: greedy(pr, "split")),
                    ("rollout", lambda: rollout(pr))):
        r = f()
        out[name] = {"cost": r["cost"], "time": r["time"]}
    return out


def job(args):
    if args[0] == "isc":
        _, c, m, s = args
        if c not in _POOLS:
            _POOLS[c] = CircuitPool(parse_verilog(f"data/iscas85/{c}.v"), 256, seed=0)
        pr = _POOLS[c].instance(m, m + 5, seed=900000 + 1000 * m + s, skew="moderate",
                                cost_mode="random")
        return solve(pr, repr(args), {"set": "ISCAS", "circuit": c})
    _, m, n, s = args
    pr = synthetic(m, n, seed=900000 + 1000 * m + s, skew="moderate", cost_mode="random")
    return solve(pr, repr(args), {"set": "synthetic"})


if __name__ == "__main__":
    jobs = [("isc", c, m, s) for m in (50, 60) for c in ("c432", "c499", "c880", "c1355", "c1908")
            for s in range(3)]
    jobs += [("syn", m, n, s) for m, n in ((120, 70), (150, 80)) for s in range(8)]
    done = {json.loads(l)["key"] for l in open(PATH)} if os.path.exists(PATH) else set()
    todo = [j for j in jobs if repr(j) not in done]
    print(len(todo), "to run", flush=True)
    with Pool(int(sys.argv[1])) as pool, open(PATH, "a") as fh:
        for r in pool.imap_unordered(job, todo):
            fh.write(json.dumps(r) + "\n")
            fh.flush()
