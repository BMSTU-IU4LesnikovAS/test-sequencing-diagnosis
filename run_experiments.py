"""Main experiments. Results are appended to JSON-lines files so the run can be
resumed: instances already present in the output file are skipped.

usage:
  python3 run_experiments.py synthetic <m> [<m> ...]
  python3 run_experiments.py iscas <circuit> [<circuit> ...]
"""
import json
import os
import sys
from multiprocessing import Pool

from tsq.circuits import parse_verilog
from tsq.problems import CircuitPool, synthetic
from tsq.solvers import exact, greedy, rollout, sa_order

N_FOR_M = {10: 15, 20: 25, 30: 30, 40: 35, 60: 40, 80: 50, 100: 60}
SKEWS = ["uniform", "moderate", "skewed"]
COSTS = ["unit", "random"]
INSTANCES = 10
SA_SEEDS = 10
SA_ITERS = 5000
EXACT_TIME_LIMIT = float(os.environ.get("EXACT_TIME_LIMIT", "600"))


def solve_all(pr):
    out = {"instance": pr.name, **pr.meta}
    out.pop("faults", None)
    out.pop("vectors", None)
    e = exact(pr, bound="huffman", time_limit=EXACT_TIME_LIMIT)
    # the unbounded baseline is only run up to m = 60 (it mostly times out beyond)
    e0 = (exact(pr, bound="none", time_limit=float(os.environ.get("NOBOUND_LIMIT", "60"))) if pr.m <= 60
          else {"cost": None, "optimal": False, "states": None, "time": None, "skipped": True})
    out["exact"] = e
    out["exact_nobound"] = e0
    out["greedy_ig"] = greedy(pr, "ig")
    out["greedy_split"] = greedy(pr, "split")
    out["rollout"] = rollout(pr)
    sa = [sa_order(pr, seed=s, iters=SA_ITERS) for s in range(SA_SEEDS)]
    out["sa_order"] = {"costs": [r["cost"] for r in sa], "times": [r["time"] for r in sa]}
    for k in ("greedy_ig", "greedy_split", "rollout"):
        out[k] = {"cost": out[k]["cost"], "time": out[k]["time"]}
    return out


def _syn(args):
    m, skew, cost, s = args
    out = solve_all(synthetic(m, N_FOR_M[m], seed=1000 * m + s, skew=skew, cost_mode=cost))
    out["key"] = repr(args)
    return out


_POOLS = {}


def _isc(args):
    circ, m, skew, s = args
    if circ not in _POOLS:
        _POOLS[circ] = CircuitPool(parse_verilog(f"data/iscas85/{circ}.v"), 256, seed=0)
    pr = _POOLS[circ].instance(m, m + 5, seed=1000 * m + s, skew=skew, cost_mode="random")
    out = solve_all(pr)
    out["key"] = repr(args)
    return out


def done_set(path):
    if not os.path.exists(path):
        return set()
    with open(path) as fh:
        return {json.loads(line)["key"] for line in fh}


def main():
    mode, keys = sys.argv[1], sys.argv[2:]
    os.makedirs("results", exist_ok=True)
    if mode == "synthetic":
        path = "results/synthetic.jsonl"
        jobs = [(int(m), sk, co, s) for m in keys for sk in SKEWS for co in COSTS
                for s in range(INSTANCES)]
        names = [f"syn_m{m}_n{N_FOR_M[m]}_{sk}_{co}_s{1000 * m + s}" for m, sk, co, s in jobs]
        fn = _syn
    else:
        path = "results/iscas.jsonl"
        # real detection matrices are sparse and much harder for the exact
        # solver, so ISCAS uses m = 20, 30, 40 (5 instances per cell at m = 40)
        jobs = [(c, m, sk, s) for c in keys for m in (20, 30, 40) for sk in ("moderate", "skewed")
                for s in range(INSTANCES if m < 40 else 5)]
        names = [f"{c}_m{m}_n{m + 5}_{sk}_random_s{1000 * m + s}" for c, m, sk, s in jobs]
        fn = _isc
    done = done_set(path)
    todo = [j for j in jobs if repr(j) not in done]
    print(f"{len(todo)} instances to run, {len(jobs) - len(todo)} already done", flush=True)
    with Pool(2) as pool, open(path, "a") as fh:
        for res in pool.imap_unordered(fn, todo):
            fh.write(json.dumps(res) + "\n")
            fh.flush()
            print(res["instance"], "exact %.2fs" % res["exact"]["time"], flush=True)


if __name__ == "__main__":
    main()
