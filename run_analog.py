"""Analog case study (KHN filter): build P(fail|fault) by Monte Carlo, merge
ambiguity groups, solve with perfect-test and imperfect-test models."""
import json
import numpy as np
from tsq.analog import build_case, faults, PASSIVES, OPAMPS, OUTPUTS, FREQS
from tsq.problems import Problem
from tsq.solvers import exact, exact_bb, greedy, rollout
from tsq.noisy import (model_from_pf, exact_noisy, greedy_choice, rollout_chooser, policy_value,
                       perfect_optimal_tree, perfect_tree_noisy, misdiagnosis_prob, dictionary_policy)

case = build_case()
pf = case["pf"]
fl = faults()
# illustrative relative failure rates (per component) and failure-mode shares
rate = {"R": 1.0, "C": 2.0}
share = {"open": 0.4, "short": 0.2, 1.5: 0.2, 0.5: 0.2}
w = []
for name, el, mod in fl:
    if el in OPAMPS:
        w.append(5.0)
    else:
        w.append(rate[PASSIVES[el][0]] * share[mod])
w = np.array(w); w /= w.sum()
cost = np.array([1.0 if o == "lp" else 3.0 for o in OUTPUTS for _ in FREQS])

D = pf >= 0.5
# ambiguity groups: identical perfect-test signatures
keys = {}
for i in range(len(fl)):
    keys.setdefault(tuple(D[i]), []).append(i)
groups = list(keys.values())
gp = np.array([w[g].sum() for g in groups])
gpf = np.array([(w[g][:, None] * pf[g]).sum(0) / w[g].sum() for g in groups])
gD = np.array([D[g[0]] for g in groups])
names = [" / ".join(case["faults"][i] for i in g) for g in groups]
# a group whose signature is all-pass is undetectable by these tests
print("faults", len(fl), "groups", len(groups), "undetectable groups", sum(1 for g in gD if not g.any()))
for nm, d in zip(names, gD):
    if len(nm) > 0 and (sum(1 for x in names if x == nm) or True):
        pass
pr = Problem("khn", gp, cost, gD, {})
res = {"faults": case["faults"], "tests": case["tests"], "groups": names, "group_prior": gp.tolist(),
       "pf": pf.tolist(), "cost": cost.tolist(), "nominal_db": case["nominal_db"].tolist()}
# soft entries: how far from deterministic is the matrix
soft = ((pf > 0.02) & (pf < 0.98)).mean()
res["soft_share"] = float(soft)
print("share of P(fail) entries strictly between 0.02 and 0.98:", round(soft, 3))
# ---- perfect-test model
e = exact(pr)
res["perfect"] = {"exact": e["cost"], "greedy_ig": greedy(pr, "ig")["cost"],
                  "greedy_split": greedy(pr, "split")["cost"], "rollout": rollout(pr)["cost"],
                  "dictionary": float(cost.sum()), "time": e["time"]}
print("perfect", res["perfect"])
# ---- imperfect tests with Monte Carlo probabilities
res["noisy"] = []
for lm in (5.0, 20.0, 100.0):
    M = model_from_pf(gp, cost, gpf, lm * cost.mean(), D=gD)
    ex = exact_noisy(M)
    row = {"Lmult": lm, "L": M.L, "exact": ex}
    tree = perfect_optimal_tree(pr)
    tc, er = misdiagnosis_prob(M, tree=tree)
    row["perfect_tree"] = {"cost": tc + M.L * er, "test_cost": tc, "p_err": er}
    for nm, ch in (("greedy", lambda A, F, ww: greedy_choice(M, A, ww)), ("rollout", rollout_chooser(M))):
        tc, er = misdiagnosis_prob(M, choose=ch)
        row[nm] = {"cost": tc + M.L * er, "test_cost": tc, "p_err": er}
    row["dictionary"] = dictionary_policy(M)
    # optimal policy decomposition
    from run_noisy import optimal_detail
    row["exact_detail"] = optimal_detail(M)
    res["noisy"].append(row)
    print(lm, {k: (round(v["cost"], 3) if isinstance(v, dict) and v.get("cost") is not None else v)
               for k, v in row.items() if k not in ("exact_detail",)}, row["exact_detail"])
json.dump(res, open("results/analog.json", "w"), default=float)
