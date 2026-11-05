"""Aggregate results into tables (CSV + Markdown) and figures (PNG)."""
import itertools
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

OUT = "results/tables"
FIG = "results/figures"
os.makedirs(OUT, exist_ok=True)
os.makedirs(FIG, exist_ok=True)
HEUR = ["greedy_ig", "greedy_split", "rollout", "sa_order"]
LABEL = {"greedy_ig": "Greedy (IG/cost)", "greedy_split": "Greedy (balance/cost)",
         "rollout": "Rollout", "sa_order": "SA (static order)"}


def load(path):
    return [json.loads(l) for l in open(path)]


def tidy(rows):
    recs = []
    for d in rows:
        base = {k: d.get(k) for k in ("instance", "m", "n", "skew", "cost_mode", "circuit")}
        ex = d["exact"]
        base.update(opt=ex["optimal"], opt_cost=ex["cost"], t_exact=ex["time"],
                    states=ex["states"],
                    opt_nb=(None if d["exact_nobound"].get("skipped") else d["exact_nobound"]["optimal"]),
                    t_nb=d["exact_nobound"].get("time"))
        for h in ("greedy_ig", "greedy_split", "rollout"):
            base[f"c_{h}"] = d[h]["cost"]
            base[f"t_{h}"] = d[h]["time"]
        base["c_sa_order"] = float(np.mean(d["sa_order"]["costs"]))
        base["c_sa_best"] = float(np.min(d["sa_order"]["costs"]))
        base["t_sa_order"] = float(np.mean(d["sa_order"]["times"]))
        recs.append(base)
    df = pd.DataFrame(recs)
    for h in HEUR + ["sa_best"]:
        df[f"g_{h}"] = np.where(df.opt, 100 * (df[f"c_{h}"] / df.opt_cost - 1), np.nan)
    return df


def ms(x):
    x = x.dropna()
    return f"{x.mean():.2f} ± {x.std(ddof=1):.2f}" if len(x) > 1 else (f"{x.mean():.2f}" if len(x) else "—")


def holm(pvals):
    order = np.argsort(pvals)
    adj = np.empty(len(pvals))
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(pvals) - rank) * pvals[i])
        adj[i] = min(1.0, running)
    return adj


def exact_table(df, by):
    g = df.groupby(by)
    t = pd.DataFrame({
        "instances": g.size(),
        "solved_%": g.opt.mean() * 100,
        "time_median_s": g.t_exact.median(),
        "time_max_s": g.t_exact.max(),
        "states_median": g.states.median(),
        "solved_within_nobound_limit_%": None,
        "nobound_solved_%": g.opt_nb.apply(lambda s: 100 * s.astype(float).mean() if s.notna().all() else np.nan),
        "nobound_time_median_s": g.t_nb.median(),
    })
    t["solved_within_nobound_limit_%"] = g.apply(
        lambda d: 100 * ((d.opt) & (d.t_exact <= LIMIT_NB)).mean() if d.opt_nb.notna().all() else np.nan)
    return t.round(2)


def gap_table(df, by):
    rows = []
    for key, sub in df.groupby(by):
        r = dict(zip(by if isinstance(by, list) else [by], key if isinstance(key, tuple) else [key]))
        r["n_opt"] = int(sub.opt.sum())
        for h in HEUR:
            r[LABEL[h] + " gap %"] = ms(sub[f"g_{h}"])
            r[LABEL[h] + " ms"] = f"{1000 * sub[f't_{h}'].mean():.1f}"
        rows.append(r)
    return pd.DataFrame(rows)


def wilcoxon_table(df):
    sub = df[df.opt]
    pairs = list(itertools.combinations(HEUR, 2))
    p, med = [], []
    for a, b in pairs:
        d = sub[f"g_{a}"] - sub[f"g_{b}"]
        p.append(wilcoxon(sub[f"g_{a}"], sub[f"g_{b}"], zero_method="zsplit").pvalue)
        med.append(d.median())
    return pd.DataFrame({"A": [LABEL[a] for a, _ in pairs], "B": [LABEL[b] for _, b in pairs],
                         "median gap A-B (pp)": np.round(med, 3), "p": p, "p Holm": holm(np.array(p))})


def save(t, name):
    t.to_csv(f"{OUT}/{name}.csv", index=isinstance(t.index, pd.MultiIndex) or t.index.name is not None)
    with open(f"{OUT}/{name}.md", "w") as fh:
        fh.write(t.to_markdown(index=isinstance(t.index, pd.MultiIndex) or t.index.name is not None))


LIMIT_NB = 60.0
syn = tidy(load("results/synthetic.jsonl"))
isc = tidy(load("results/iscas.jsonl"))
isc = isc[isc.m.isin([20, 30, 40])]
# keep the planned grid only (5 instances per cell at m = 40)
isc["s"] = isc.instance.str.extract(r"_s(\d+)$").astype(int) % 1000
isc = isc[(isc.m < 40) | (isc.s < 5)]

save(exact_table(syn, "m"), "T1_exact_synthetic")
LIMIT_NB = 30.0
save(exact_table(isc, ["circuit", "m"]), "T2_exact_iscas")
save(exact_table(isc, "m"), "T2b_exact_iscas_by_m")
save(gap_table(syn, ["m"]), "T3_gaps_synthetic_by_m")
save(gap_table(syn, ["skew", "cost_mode"]), "T4_gaps_synthetic_by_prior_cost")
save(gap_table(isc, ["m"]), "T5_gaps_iscas_by_m")
save(gap_table(isc, ["circuit"]), "T5b_gaps_iscas_by_circuit")
save(wilcoxon_table(syn), "T6_wilcoxon_synthetic")
save(wilcoxon_table(isc), "T6b_wilcoxon_iscas")

# ---- QUBO decomposition
q = pd.DataFrame(load("results/qubo_decomposition.jsonl"))
q["static_gap"] = 100 * (q.best_static / q.exact - 1)
q["surrogate_gap"] = 100 * (q.surrogate_opt_true_cost / q.exact - 1)
q["hop_gap"] = q.apply(lambda r: 100 * (np.mean(r.hopfield) / r.exact - 1), axis=1)
q["sb_gap"] = q.apply(lambda r: 100 * (np.mean(r.sb) / r.exact - 1), axis=1)
q["greedy_gap"] = 100 * (q.greedy_split / q.exact - 1)
q["hop_feas"] = q.hopfield_feas.apply(np.mean) * 100
q["sb_feas"] = q.sb_feas.apply(np.mean) * 100
q["hop_t"] = q.hopfield_time.apply(np.mean)
q["sb_t"] = q.sb_time.apply(np.mean)
rows = []
for m, sub in q.groupby("m"):
    rows.append({"m": m, "n": int(sub.n.iloc[0]), "instances": len(sub),
                 "Greedy (balance) gap %": ms(sub.greedy_gap),
                 "Best static order gap %": ms(sub.static_gap),
                 "Surrogate optimum gap %": ms(sub.surrogate_gap),
                 "Hopfield gap %": ms(sub.hop_gap), "Hopfield feasible %": f"{sub.hop_feas.mean():.0f}",
                 "Hopfield s": f"{sub.hop_t.mean():.2f}",
                 "bSB gap %": ms(sub.sb_gap), "bSB feasible %": f"{sub.sb_feas.mean():.0f}",
                 "bSB s": f"{sub.sb_t.mean():.2f}",
                 "Spearman(surrogate, true)": ms(sub.spearman_surrogate_true)})
save(pd.DataFrame(rows), "T7_qubo_decomposition")

pen = pd.DataFrame(load("results/qubo_penalty.jsonl"))
pen["hop_gap"] = 100 * (pen.hopfield / pen.exact - 1)
pen["sb_gap"] = 100 * (pen.sb / pen.exact - 1)
pt = pen.groupby("mult").agg(hop_gap=("hop_gap", "mean"), hop_feas=("hopfield_feas", "mean"),
                             sb_gap=("sb_gap", "mean"), sb_feas=("sb_feas", "mean"))
pt[["hop_feas", "sb_feas"]] *= 100
save(pt.round(1), "T8_penalty_sensitivity")

# ---------------------------------------------------------------- figures
GREY, BLUE = "#8a8a8a", "#1f5fbf"

# F1: exact solver time vs m (median, IQR), with and without the bound
fig, ax = plt.subplots(figsize=(6.4, 4))
for data, lab, col, mk in [(syn, "synthetic", BLUE, "o"), (isc, "ISCAS'85", "#c0392b", "s")]:
    g = data.groupby("m").t_exact
    ax.errorbar(g.median().index, g.median(), yerr=[g.median() - g.quantile(.25), g.quantile(.75) - g.median()],
                marker=mk, color=col, capsize=3, label=f"exact + Huffman bound ({lab})")
    s = data[data.m <= 60].groupby("m").t_nb
    ax.plot(s.median().index, s.median(), marker=mk, color=col, ls="--", alpha=.6,
            label=f"exact, no bound ({lab})")
ax.set_yscale("log")
ax.set_xlabel("number of candidate faults m")
ax.set_ylabel("solve time, s (median, IQR)")
ax.set_title("Exact solver run time")
ax.grid(alpha=.3, which="both")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(f"{FIG}/F1_exact_time.png", dpi=300)

# F2: heuristic gaps vs m (synthetic)
fig, ax = plt.subplots(figsize=(6.4, 4))
cols = {"greedy_ig": GREY, "greedy_split": "#555555", "rollout": BLUE, "sa_order": "#c0392b"}
for h in HEUR:
    g = syn[syn.opt].groupby("m")[f"g_{h}"]
    med, q1, q3 = g.median(), g.quantile(.25), g.quantile(.75)
    ax.errorbar(med.index, med, yerr=[med - q1, q3 - med], marker="o", capsize=3, color=cols[h], label=LABEL[h])
ax.set_xlabel("number of candidate faults m")
ax.set_ylabel("gap to optimum, % (median, IQR)")
ax.set_title("Heuristic gap to the exact optimum (synthetic)")
ax.grid(alpha=.3)
ax.legend(fontsize=8, ncol=2, loc="upper left")
ax.set_ylim(bottom=0)
fig.tight_layout()
fig.savefig(f"{FIG}/F2_heuristic_gaps.png", dpi=300)

# F3: QUBO loss decomposition
fig, ax = plt.subplots(figsize=(6.4, 4))
labels = ["Greedy\n(balance)", "Best static\norder", "Surrogate\noptimum", "Hopfield", "bSB"]
keys = ["greedy_gap", "static_gap", "surrogate_gap", "hop_gap", "sb_gap"]
width = 0.25
for i, (m, sub) in enumerate(q.groupby("m")):
    vals = [sub[k].mean() for k in keys]
    errs = [sub[k].std() for k in keys]
    ax.bar(np.arange(5) + (i - 1) * width, vals, width, yerr=errs, capsize=2,
           color=BLUE, alpha=0.35 + 0.3 * i, label=f"m = {m}")
ax.set_xticks(np.arange(5))
ax.set_xticklabels(labels, fontsize=8)
ax.set_ylabel("gap to optimum, %")
ax.set_title("Where the QUBO route loses quality")
ax.grid(alpha=.3, axis="y")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(f"{FIG}/F3_qubo_decomposition.png", dpi=300)

# F4: penalty sensitivity
fig, ax1 = plt.subplots(figsize=(6.4, 4))
ax1.plot(pt.index, pt.hop_gap, "o-", color=BLUE, label="Hopfield gap")
ax1.plot(pt.index, pt.sb_gap, "s-", color="#c0392b", label="bSB gap")
ax1.set_xscale("log", base=2)
ax1.set_xlabel("penalty multiplier")
ax1.set_ylabel("gap to optimum, %")
ax2 = ax1.twinx()
ax2.plot(pt.index, pt.hop_feas, "o--", color=BLUE, alpha=.5, label="Hopfield feasible")
ax2.plot(pt.index, pt.sb_feas, "s--", color="#c0392b", alpha=.5, label="bSB feasible")
ax2.set_ylabel("feasible outputs, %")
h1, l1 = ax1.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax1.legend(h1 + h2, l1 + l2, fontsize=8, loc="upper center")
ax1.set_title("Sensitivity to the one-hot penalty (m = 20)")
ax1.grid(alpha=.3)
fig.tight_layout()
fig.savefig(f"{FIG}/F4_penalty.png", dpi=300)

print("synthetic instances:", len(syn), " ISCAS instances:", len(isc), " QUBO:", len(q), " penalty:", len(pen))
