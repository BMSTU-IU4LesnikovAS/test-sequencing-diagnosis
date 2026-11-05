"""Tightness of the admissible lower bounds: LB(S) / J*(S) over all subsets S of
small instances (dense synthetic vs. sparse ISCAS'85-derived)."""
import functools, json, sys
import numpy as np
import heapq
from tsq.circuits import parse_verilog
from tsq.problems import synthetic, CircuitPool


def bounds(pr):
    D, p, c = pr.D, pr.p, pr.c
    m, n = D.shape
    F = [sum(1 << i for i in range(m) if D[i, j]) for j in range(n)]

    @functools.lru_cache(None)
    def J(S):
        if S & (S - 1) == 0:
            return 0.0
        v = [c[j] * sum(p[i] for i in range(m) if S >> i & 1) + J(S & F[j]) + J(S & ~F[j])
             for j in range(n) if (S & F[j]) and (S & ~F[j])]
        return min(v) if v else 0.0

    rows = []
    for S in range(1, 1 << m):
        idx = [i for i in range(m) if S >> i & 1]
        split = [j for j in range(n) if (S & F[j]) and (S & ~F[j])]
        if len(idx) < 3 or not split:
            continue
        js = J(S)
        h = [p[i] for i in idx]; heapq.heapify(h); tot = 0.0
        while len(h) > 1:
            a = heapq.heappop(h); b = heapq.heappop(h); tot += a + b; heapq.heappush(h, a + b)
        huff = min(c[split]) * tot
        sub = D[np.ix_(idx, split)]; a = sub.sum(0); b = len(idx) - a
        r = np.where(sub, (c[split] / b)[None, :], (c[split] / a)[None, :]).min(1)
        sig = [tuple(D[i]) for i in idx]
        U = np.array([len(idx) - sig.count(x) for x in sig])
        cov = float((p[idx] * U * r).sum())
        rows.append((len(idx), huff / js, cov / js, max(huff, cov) / js))
    return rows


out = []
pools = {c: CircuitPool(parse_verilog(f"data/iscas85/{c}.v"), 256, seed=0)
         for c in ["c432", "c499", "c880", "c1355", "c1908"]}
for s in range(10):
    for kind, pr in [("synthetic", synthetic(12, 10, seed=700000 + s, cost_mode="random")),
                     *[(f"iscas_{c}", pool.instance(12, 12, seed=700000 + s, cost_mode="random"))
                       for c, pool in pools.items() if s < 4]]:
        dens = float(pr.D.mean())
        for size, h, cv, mx in bounds(pr):
            out.append({"kind": kind.split("_")[0], "inst": f"{kind}_{s}", "density": dens,
                        "size": size, "huffman": h, "cover": cv, "both": mx})
json.dump(out, open("results/tightness.json", "w"))
import pandas as pd
df = pd.DataFrame(out)
print(df.groupby("kind")[["density", "huffman", "cover", "both"]].mean().round(3))
print(df.groupby("kind").apply(lambda d: (d.cover > d.huffman).mean()).round(3).rename("cover_better_share"))
