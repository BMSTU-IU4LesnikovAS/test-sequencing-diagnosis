"""Pilot run: heuristics vs. exact optimum on small synthetic and ISCAS'85 instances."""
import csv, sys, time
import numpy as np
from tsq.circuits import parse_verilog
from tsq.problems import synthetic, CircuitPool
from tsq.solvers import exact, greedy, rollout, sa_order

rows = []
def run(pr):
    ex = exact(pr)
    res = {"exact": ex["cost"]}
    times = {"exact": ex["time"]}
    for name, f in [("greedy_ig", lambda: greedy(pr, "ig")),
                    ("greedy_split", lambda: greedy(pr, "split")),
                    ("rollout", lambda: rollout(pr)),
                    ("sa_order", lambda: sa_order(pr, seed=0, iters=5000))]:
        r = f(); res[name] = r["cost"]; times[name] = r["time"]
    for k in res:
        rows.append({"instance": pr.name, "kind": pr.meta["kind"], "m": pr.m, "n": pr.n,
                     "method": k, "cost": res[k], "gap_pct": 100 * (res[k] / ex["cost"] - 1),
                     "time_s": times[k]})

for s in range(20):
    run(synthetic(20, 25, seed=s))
pools = {c: CircuitPool(parse_verilog(f"data/iscas85/{c}.v"), 256, seed=0)
         for c in ["c432", "c499", "c880", "c1355", "c1908"]}
for c, pool in pools.items():
    for s in range(4):
        run(pool.instance(20, 25, seed=s))

with open("results/pilot.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)

import pandas as pd
df = pd.DataFrame(rows)
print(df.groupby(["kind", "method"]).agg(gap_mean=("gap_pct", "mean"), gap_std=("gap_pct", "std"),
      gap_max=("gap_pct", "max"), time_ms=("time_s", lambda x: 1000 * x.mean())).round(2))
print(df[df.kind == "iscas85"].groupby("instance").n.first())
