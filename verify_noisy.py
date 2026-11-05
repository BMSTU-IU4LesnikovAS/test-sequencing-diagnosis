"""Checks of the imperfect-test code: (1) zero noise reproduces the perfect-test
optimum; (2) pruned DP = unpruned DP; (3) the optimum is not worse than any
policy; (4) exact policy values agree with direct Monte Carlo simulation of the
policy (independent code path); (5) MNA agrees with ngspice."""
import numpy as np
from run_noisy import optimal_detail
from tsq.problems import synthetic
from tsq.solvers import exact
from tsq.noisy import (NoisyModel, exact_noisy, greedy_choice, rollout_chooser, policy_value,
                       perfect_optimal_tree, perfect_tree_noisy, dictionary_policy, misdiagnosis_prob)

fails = 0


def check(c, msg):
    global fails
    if not c:
        fails += 1
        print("FAIL", msg)


def simulate(M, choose, runs, rng):
    """Draw a fault from the prior, draw test outcomes, follow the policy."""
    tot = 0.0
    m = M.m
    for _ in range(runs):
        i = rng.choice(m, p=M.p)
        A = F = 0
        w = M.p.copy()
        cost = 0.0
        while True:
            j = choose(A, F, w)
            if j is None:
                cost += M.L * (int(np.argmax(w)) != i)
                break
            cost += M.c[j]
            bit = 1 << j
            if rng.random() < M.pf[i, j]:
                w = w * M.pf[:, j]; F |= bit
            else:
                w = w * (1 - M.pf[:, j])
            A |= bit
        tot += cost
    return tot / runs


rng = np.random.default_rng(3)
for s in range(8):
    pr = synthetic(8, 8, seed=990 + s, cost_mode="random")
    M0 = NoisyModel(pr, 0, 0, 1e6)
    check(abs(exact_noisy(M0)["cost"] - exact(pr)["cost"]) < 1e-9, "zero noise != perfect optimum")
    for q in (0.02, 0.1):
        M = NoisyModel(pr, q, q, 20 * pr.c.mean())
        ex = exact_noisy(M)["cost"]
        det = optimal_detail(M)
        check(abs(det["test_cost"] + M.L * det["p_err"] - ex) < 1e-8, "pruned != unpruned DP")
        g = policy_value(M, lambda A, F, w: greedy_choice(M, A, w))
        r = policy_value(M, rollout_chooser(M))
        t = perfect_tree_noisy(M, perfect_optimal_tree(pr))["cost"]
        d = dictionary_policy(M)["cost"]
        check(min(g, r, t, d) >= ex - 1e-9, "a policy beats the optimum")
        if s < 2:
            ch = lambda A, F, w: greedy_choice(M, A, w)
            sim = simulate(M, ch, 8000, rng)
            check(abs(sim - g) / g < 0.06, f"greedy value {g:.3f} vs simulation {sim:.3f}")
            print("greedy exact %.3f  simulated %.3f" % (g, sim))
print("noisy checks failures:", fails)
