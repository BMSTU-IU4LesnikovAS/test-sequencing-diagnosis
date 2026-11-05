"""Recompute every number quoted in the paper text from the raw result files."""
import json
import numpy as np


def jl(p):
    return [json.loads(l) for l in open(p)]


syn = jl("results/synthetic.jsonl")
isc = [r for r in jl("results/iscas.jsonl") if r["m"] < 40 or eval(r["key"])[3] < 5]
print("instances", len(syn), len(isc))

# --- 5.1 scheme A: Huffman vs none on instances solved by both
for m in (40, 60):
    rs, rt = [], []
    for r in syn:
        if r["m"] == m and r["exact"]["optimal"] and r["exact_nobound"].get("optimal"):
            rs.append(r["exact_nobound"]["states"] / r["exact"]["states"])
            rt.append(r["exact_nobound"]["time"] / r["exact"]["time"])
    print(f"schemeA syn m={m}: both solved {len(rs)}, states ratio med {np.median(rs):.1f}, time ratio med {np.median(rt):.2f}")
for m in (20, 30, 40):
    rt = [r["exact_nobound"]["time"] / r["exact"]["time"] for r in isc
          if r["m"] == m and r["exact"]["optimal"] and r["exact_nobound"].get("optimal")]
    rs = [r["exact_nobound"]["states"] / r["exact"]["states"] for r in isc
          if r["m"] == m and r["exact"]["optimal"] and r["exact_nobound"].get("optimal")]
    print(f"schemeA iscas m={m}: states ratio med {np.median(rs):.2f}, time ratio med {np.median(rt):.2f}")
# max solve time m=60 with bound, time limits actually used
print("syn m=60 max time", max(r["exact"]["time"] for r in syn if r["m"] == 60))
for m in (80, 100):
    uns = [r["exact"]["time"] for r in syn if r["m"] == m and not r["exact"]["optimal"]]
    print(f"syn m={m} unsolved stop times", [round(t) for t in uns])
    st = [r["exact"]["states"] for r in syn if r["m"] == m and not r["exact"]["optimal"]]
    print("   states at stop", st)
uns = [(r["instance"], round(r["exact"]["time"]), r["exact"]["states"]) for r in isc if not r["exact"]["optimal"]]
print("iscas unsolved (scheme A)", uns)
dens = {}
for r in isc:
    dens.setdefault(r["circuit"] if "circuit" in r else r["instance"].split("_")[0], [])
print()

# --- empirical growth exponents (scheme A, median states vs m)
def fit_exp(rows, key, ms):
    xs, ys = [], []
    for m in ms:
        v = [r[key]["states"] for r in rows if r["m"] == m and r[key].get("optimal")]
        if v:
            xs.append(m)
            ys.append(np.log2(np.median(v)))
    b = np.polyfit(xs, ys, 1)
    return b[0], xs
print("states exponent syn huffman", fit_exp(syn, "exact", (10, 20, 30, 40, 60, 80, 100)))
print("states exponent syn none   ", fit_exp(syn, "exact_nobound", (10, 20, 30, 40, 60)))
print("states exponent iscas huff ", fit_exp(isc, "exact", (20, 30, 40)))
print("states exponent iscas none ", fit_exp(isc, "exact_nobound", (20, 30, 40)))


def time_exp(rows, key, ms):
    xs, ys = [], []
    for m in ms:
        v = [r[key]["time"] for r in rows if r["m"] == m]
        xs.append(np.log(m))
        ys.append(np.log(np.median(v)))
    return np.polyfit(xs, ys, 1)[0]
for k in ("greedy_ig", "greedy_split", "rollout"):
    print(k, "time exponent syn", round(time_exp(syn, k, (10, 20, 30, 40, 60, 80, 100)), 2),
          "iscas", round(time_exp(isc, k, (20, 30, 40)), 2))

# --- scheme B comparison
for path, label in (("results/bb_iscas.jsonl", "ISCAS"), ("results/bb_synthetic.jsonl", "syn")):
    rows = jl(path)
    for m in sorted({r["m"] for r in rows}):
        R = [r for r in rows if r["m"] == m]
        both = [r for r in R if r["none"]["optimal"] and r["huffman"]["optimal"]]
        sr = [r["none"]["states"] / r["huffman"]["states"] for r in both]
        hb = [r for r in R if r["huffman"]["optimal"] and r["both"]["optimal"]]
        cr = [1 - r["both"]["states"] / r["huffman"]["states"] for r in hb]
        crmax = max(r["huffman"]["states"] / r["both"]["states"] for r in hb) if hb else None
        tr = [r["both"]["time"] / r["huffman"]["time"] for r in hb]
        # cost agreement
        agree = all(abs(r["huffman"]["cost"] - r["both"]["cost"]) < 1e-9 for r in hb)
        print(f"BB {label} m={m}: n={len(R)} none/huff states med {np.median(sr) if sr else float('nan'):.2f} "
              f"(on {len(both)}), cover state reduction med {100*np.median(cr):.1f}% max x{crmax:.2f}, "
              f"time ratio both/huff med {np.median(tr):.1f}, costs agree {agree}")
# BB vs scheme A optimum agreement
idx = {r["instance"]: r for r in syn + isc}
bad = 0
cnt = 0
for path in ("results/bb_iscas.jsonl", "results/bb_synthetic.jsonl"):
    for r in jl(path):
        a = idx.get(r["instance"])
        if a and a["exact"]["optimal"] and r["huffman"]["optimal"]:
            cnt += 1
            bad += abs(a["exact"]["cost"] - r["huffman"]["cost"]) > 1e-9
print("scheme A vs B optimum compared", cnt, "mismatches", bad)
print()

# --- tightness
t = json.load(open("results/tightness.json"))
for kind in ("synthetic", "iscas"):
    T = [x for x in t if x["kind"] == kind]
    print(kind, "subsets", len(T), "mean LB/J huff %.2f cover %.2f both %.2f, cover better on %.0f%%" % (
        np.mean([x["huffman"] for x in T]), np.mean([x["cover"] for x in T]), np.mean([x["both"] for x in T]),
        100 * np.mean([x["cover"] > x["huffman"] + 1e-12 for x in T])))
print()

# --- scaling
sc = jl("results/scale.jsonl")
for (s, m) in (("ISCAS", 50), ("ISCAS", 60), ("synthetic", 120), ("synthetic", 150)):
    R = [r for r in sc if r["set"] == s and r["m"] == m]
    sol = [r for r in R if r["bb"]["optimal"]]
    g = lambda k: np.mean([100 * (r[k]["cost"] / r["bb"]["cost"] - 1) for r in sol])
    print(f"scale {s} m={m} n={R[0]['n']}: {len(sol)}/{len(R)} solved, t med {np.median([r['bb']['time'] for r in sol]):.1f} "
          f"max {max(r['bb']['time'] for r in sol):.1f}; gaps IG {g('greedy_ig'):.2f} bal {g('greedy_split'):.2f} "
          f"roll {g('rollout'):.2f}; rollout t med {np.median([r['rollout']['time'] for r in R]):.3f}; "
          f"unsolved density {[round(r['density'],3) for r in R if not r['bb']['optimal']]} "
          f"{[r.get('circuit') for r in R if not r['bb']['optimal']]}; mean density {np.mean([r['density'] for r in R]):.2f}")
print()

# --- QUBO theory
th = json.load(open("results/qubo_theory.json"))
for n in (6, 7):
    for kind in ("synthetic", "iscas"):
        T = [x for x in th if x["n"] == n and x["kind"] == kind]
        nq = [100 * x["nonquad_share"] for x in T]
        print(f"theory n={n} {kind}: k={len(T)} nonquad mean {np.mean(nq):.1f} ({min(nq):.1f}-{max(nq):.1f}) "
              f"fitmin gap opt {np.mean([x['fit_min_gap_opt'] for x in T]):.2f} best static {np.mean([x['best_static_gap_opt'] for x in T]):.2f} "
              f"ctrl R2 min {min(x['r2_control'] for x in T):.6f} rank {T[0]['rank']}")
t8 = json.load(open("results/qubo_theory_n8.json"))
for kind in ("synthetic", "iscas"):
    nq = [100 * (1 - x["r2"]) for x in t8 if x["kind"] == kind]
    print(f"theory n=8 {kind}: k={len(nq)} nonquad mean {np.mean(nq):.1f} ({min(nq):.1f}-{max(nq):.1f})")
dd = json.load(open("results/qubo_datadriven.json"))
print("datadriven instances", len(dd), "greedy split gap mean %.2f" % np.mean([x["greedy_gap"] for x in dd]))
for K in (50, 100, 200, 500, 1000, 2000):
    q = [x[f"K{K}"] for x in dd]
    rr = [x[f"rand{K}"] for x in dd]
    print(f"K={K}: qubo {np.mean(q):.1f} rand {np.mean(rr):.1f} worse frac {100*np.mean([a > b + 1e-9 for a, b in zip(q, rr)]):.0f}%")
