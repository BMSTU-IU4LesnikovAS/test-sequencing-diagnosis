import json, numpy as np
R = [json.loads(l) for l in open("results/ao_compare.jsonl")]
print(len(R), "rows; dup", len(R) - len({r["key"] for r in R}))
for s, m in (("ISCAS", 30), ("ISCAS", 40), ("synthetic", 60), ("synthetic", 80)):
    X = [r for r in R if r["set"] == s and r["m"] == m]
    a = [r["ao"] for r in X]; b = [r["bb"] for r in X]
    both = [r for r in X if r["ao"]["optimal"] and r["bb"]["optimal"]]
    agree = all(abs(r["ao"]["cost"] - r["bb"]["cost"]) < 1e-9 for r in both)
    tm = lambda L: np.median([x["time"] if x["optimal"] else np.inf for x in L])
    tr = np.median([r["ao"]["time"] / r["bb"]["time"] for r in both])
    nr = np.median([r["ao"]["states"] / max(1, r["bb"]["states"]) for r in both])
    print(f"{s} m={m} n_inst={len(X)} solved AO* {100*np.mean([x['optimal'] for x in a]):.0f}% BB {100*np.mean([x['optimal'] for x in b]):.0f}% "
          f"| t med AO* {tm(a):.2f} BB {tm(b):.2f} | time ratio med {tr:.2f} | nodes/states med {nr:.2f} | agree {agree} on {len(both)}")
