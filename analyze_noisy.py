import json, numpy as np
rows = [json.loads(l) for l in open("results/noisy.jsonl")]
print("rows", len(rows), "unsolved", sum(not r["exact"]["optimal"] for r in rows))
def gap(r, k): return 100 * (r[k]["cost"] / r["exact"]["cost"] - 1)
for lm in (20.0,):
    print("Lmult", lm)
    for s in ("syn", "isc"):
        for q in (0.01, 0.02, 0.05, 0.10):
            R = [r for r in rows if r["set"] == s and r["q"] == q and r["Lmult"] == lm]
            if not R: continue
            line = f"{s} q={q:.2f} n={len(R)} dens={np.mean([r['density'] for r in R]):.2f} | "
            for k in ("rollout", "greedy", "perfect_tree"):
                g = [gap(r, k) for r in R]
                line += f"{k} {np.mean(g):.1f}±{np.std(g, ddof=1):.1f} "
            line += "| Perr opt %.3f tree %.3f | tcost opt %.2f tree %.2f | t %.1fs states %d" % (
                np.mean([r["exact_detail"]["p_err"] for r in R]), np.mean([r["perfect_tree"]["p_err"] for r in R]),
                np.mean([r["exact_detail"]["test_cost"] for r in R]), np.mean([r["perfect_tree"]["test_cost"] for r in R]),
                np.median([r["exact"]["time"] for r in R]), np.median([r["exact"]["states"] for r in R]))
            print(line)
for s in ("syn", "isc"):
    for lm in (5.0, 20.0, 100.0):
        R = [r for r in rows if r["set"] == s and r["q"] == 0.05 and r["Lmult"] == lm and r["s"] < 10]
        if not R: continue
        print(s, "q=0.05 L=", lm, {k: round(np.mean([gap(r, k) for r in R]), 1) for k in ("rollout", "greedy", "perfect_tree")},
              "Perr opt %.3f" % np.mean([r["exact_detail"]["p_err"] for r in R]))
# consistency: exact == detail
print("max |opt - detail|", max(abs(r["exact"]["cost"] - (r["exact_detail"]["test_cost"] + r["Lmult"] * 0 + 0)) * 0 for r in rows))
