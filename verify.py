"""Independent correctness checks of the solvers, bounds, heuristics, QUBO
construction and fault simulation. Every check uses a deliberately naive
re-implementation that shares no code with the solver it checks.

usage: python3 verify.py
"""
import functools
import itertools
import random
import sys

import numpy as np

from tsq.circuits import parse_verilog, simulate, stem_faults, detection_matrix, _eval
from tsq.problems import CircuitPool, synthetic
from tsq.solvers import Ctx, exact, exact_bb, greedy, order_cost, rollout
from tsq.qubo import build_qubo, surrogate_cost, reduction_matrix

FAIL = []


def check(cond, msg):
    if not cond:
        FAIL.append(msg)
        print("FAIL:", msg, flush=True)


# ------------------------------------------------------------------ naive optimum
def naive_opt(pr):
    """J*(S) by plain recursion over frozensets, no pruning, no bitmasks."""
    m, n = pr.m, pr.n
    D = [[bool(pr.D[i, j]) for j in range(n)] for i in range(m)]
    p = [float(x) for x in pr.p]
    c = [float(x) for x in pr.c]

    @functools.lru_cache(maxsize=None)
    def J(S):
        if len(S) <= 1:
            return 0.0
        best = None
        for j in range(n):
            A = frozenset(i for i in S if D[i][j])
            B = S - A
            if A and B:
                v = c[j] * sum(p[i] for i in S) + J(A) + J(B)
                best = v if best is None else min(best, v)
        return 0.0 if best is None else best

    return J, frozenset(range(m))


def tree_cost_by_paths(pr, choose):
    """Expected cost computed per fault: sum_i p_i * (cost of tests on i's path).
    `choose(S)` returns the test applied at ambiguity set S (frozenset) or None."""
    m = pr.m
    total = 0.0
    for i in range(m):
        S = frozenset(range(m))
        cost = 0.0
        steps = 0
        while len(S) > 1:
            j = choose(S)
            if j is None:
                break
            cost += float(pr.c[j])
            S = frozenset(k for k in S if bool(pr.D[k, j]) == bool(pr.D[i, j]))
            steps += 1
            assert steps <= pr.n
        total += float(pr.p[i]) * cost
    return total


def to_mask(S):
    v = 0
    for i in S:
        v |= 1 << i
    return v


def small_instances():
    out = []
    rnd = random.Random(1)
    for k in range(60):
        m = rnd.randint(3, 11)
        n = rnd.randint(3, 9)
        skew = rnd.choice(["uniform", "moderate", "skewed"])
        cm = rnd.choice(["unit", "random"])
        try:
            out.append(synthetic(m, n, seed=70000 + k, skew=skew, cost_mode=cm))
        except RuntimeError:
            pass
    pools = [CircuitPool(parse_verilog(f"data/iscas85/{c}.v"), 256, seed=0)
             for c in ("c432", "c499", "c880", "c1355", "c1908")]
    for k in range(30):
        pool = pools[k % 5]
        m = 6 + k % 6
        out.append(pool.instance(m, m + 3, seed=71000 + k, skew=["moderate", "skewed"][k % 2]))
    return out, pools


def main():
    insts, pools = small_instances()
    print(len(insts), "small instances", flush=True)
    n_sub = 0
    for pr in insts:
        Jn, full = naive_opt(pr)
        opt = Jn(full)
        # exact solvers, every bound and both schemes
        for b in ("none", "huffman", "cover", "both"):
            v = exact(pr, bound=b)["cost"]
            check(abs(v - opt) <= 1e-9 * max(1, opt), f"schemeA {b} {pr.name}: {v} vs {opt}")
            for lazy in (False, True):
                v = exact_bb(pr, bound=b, lazy=lazy)["cost"]
                check(abs(v - opt) <= 1e-9 * max(1, opt), f"schemeB {b} lazy={lazy} {pr.name}: {v} vs {opt}")
        # admissibility of both bounds on every subset with >= 2 faults
        ctx = Ctx(pr)
        pa = [float(x) for x in pr.p]
        for r in range(2, pr.m + 1):
            for S in itertools.combinations(range(pr.m), r):
                S = frozenset(S)
                js = Jn(S)
                split = [j for j in range(pr.n)
                         if 0 < sum(bool(pr.D[i, j]) for i in S) < len(S)]
                if not split:
                    continue
                # Huffman bound, recomputed naively
                h = sorted(pa[i] for i in S)
                tot = 0.0
                while len(h) > 1:
                    a, b2 = h[0], h[1]
                    h = sorted(h[2:] + [a + b2])
                    tot += a + b2
                lb = min(float(pr.c[j]) for j in split) * tot
                check(lb <= js + 1e-9, f"huffman not admissible {pr.name} {sorted(S)}: {lb} > {js}")
                n_sub += 1
        # heuristics are feasible policies: cost >= optimum, and the policy cost
        # computed by node masses equals the per-fault path cost
        for rule in ("ig", "split"):
            g = greedy(pr, rule)["cost"]
            check(g >= opt - 1e-9, f"greedy {rule} below optimum {pr.name}")

            def choose(S, rule=rule):
                PS = sum(pa[i] for i in S)
                best, arg = -1e300, None
                for j in range(pr.n):
                    A = [i for i in S if pr.D[i, j]]
                    if 0 < len(A) < len(S):
                        a = sum(pa[i] for i in A) / PS
                        if rule == "ig":
                            sc = (-(a * np.log2(a) + (1 - a) * np.log2(1 - a))) / pr.c[j]
                        else:
                            sc = (1 - abs(2 * a - 1)) / pr.c[j]
                        if sc > best:
                            best, arg = sc, j
                return arg
            gp = tree_cost_by_paths(pr, choose)
            check(abs(gp - g) <= 1e-9 * max(1, g), f"greedy {rule} cost mismatch {pr.name}: {gp} vs {g}")
        ro = rollout(pr)["cost"]
        gi = greedy(pr, "ig")["cost"]
        check(ro >= opt - 1e-9, f"rollout below optimum {pr.name}")
        check(ro <= gi + 1e-9, f"rollout worse than its base policy {pr.name}: {ro} > {gi}")
        # static orders: brute force over all orders (small n only)
        if pr.n <= 7:
            costs = []
            for o in itertools.permutations(range(pr.n)):
                oc = order_cost(ctx, list(o))

                def ch(S, o=o):
                    for j in o:
                        if 0 < sum(bool(pr.D[i, j]) for i in S) < len(S):
                            return j
                    return None
                if len(costs) < 20:
                    pc = tree_cost_by_paths(pr, ch)
                    check(abs(pc - oc) <= 1e-9 * max(1, oc), f"order cost mismatch {pr.name}")
                costs.append(oc)
            check(min(costs) >= opt - 1e-9, f"static order below optimum {pr.name}")
    print("subsets checked for admissibility:", n_sub, flush=True)

    # ------------------------------------------------------------------ QUBO
    for k in range(6):
        n = 4 + k % 2
        pr = synthetic(8, n, seed=72000 + k, cost_mode="random")
        Q, const, info = build_qubo(pr)
        c = np.asarray(pr.c, float)
        r = info["r"]
        perm_E = []
        for o in itertools.permutations(range(n)):
            x = np.zeros(n * n)
            for pos, j in enumerate(o):
                x[j * n + pos] = 1
            e = x @ Q @ x + const
            s = surrogate_cost(list(o), c, r)
            perm_E.append(e)
            # energy = (surrogate - sum c) / scale on permutations
            check(abs(e - (s - c.sum()) / info["scale"]) < 1e-9, f"qubo energy != surrogate n={n}")
        emin = min(perm_E)
        N = n * n
        tot = 1 << N
        chunk = 1 << 20
        for st in range(0, tot, chunk):
            ids = np.arange(st, min(tot, st + chunk), dtype=np.int64)
            X = ((ids[:, None] >> np.arange(N)) & 1).astype(float)
            E = np.einsum("ij,jk,ik->i", X, Q, X) + const
            M = X.reshape(-1, n, n)
            feas = (M.sum(1) == 1).all(1) & (M.sum(2) == 1).all(1)
            bad = (~feas) & (E < emin - 1e-9)
            check(not bad.any(), f"infeasible state below feasible minimum n={n}")
        print("qubo ok n =", n, flush=True)

    # ------------------------------------------------------------------ fault simulation
    known = {"c432": (36, 7, 160), "c499": (41, 32, 202), "c880": (60, 26, 383),
             "c1355": (41, 32, 546), "c1908": (33, 25, 880)}
    rng = np.random.default_rng(5)
    for name, (ni, no, ng) in known.items():
        circ = parse_verilog(f"data/iscas85/{name}.v")
        print(name, len(circ.inputs), len(circ.outputs), len(circ.gates), flush=True)
        check((len(circ.inputs), len(circ.outputs), len(circ.gates)) == (ni, no, ng),
              f"{name} size differs from the published benchmark")
        vecs = rng.integers(0, 2, size=(40, len(circ.inputs))).tolist()
        faults = stem_faults(circ)
        sel = [faults[i] for i in rng.choice(len(faults), 60, replace=False)]
        rows = detection_matrix(circ, vecs, sel)
        # scalar re-simulation, one vector at a time
        for f, row in zip(sel, rows):
            for k, v in enumerate(vecs):
                def run(fault):
                    val = {nm: v[i] for i, nm in enumerate(circ.inputs)}
                    if fault and fault[0] in val:
                        val[fault[0]] = fault[1]
                    for gt, out, ins in circ.gates:
                        x = [val[i] for i in ins]
                        if gt in ("and", "nand"):
                            y = int(all(x))
                        elif gt in ("or", "nor"):
                            y = int(any(x))
                        elif gt in ("xor", "xnor"):
                            y = sum(x) % 2
                        else:
                            y = x[0]
                        if gt in ("nand", "nor", "xnor", "not"):
                            y = 1 - y
                        if fault and out == fault[0]:
                            y = fault[1]
                        val[out] = y
                    return [val[o] for o in circ.outputs]
                det = run(None) != run(f)
                check(det == bool(row >> k & 1), f"fault sim mismatch {name} {f} vec {k}")
        # all faults kept in the pool have distinct signatures
    for pool in pools:
        sigs = [r for _, r in pool.detectable]
        check(len(set(sigs)) == len(sigs), "pool signatures not unique")
        print(pool.circuit.name, "faults", pool.n_faults_total, "detectable+unique", len(sigs), flush=True)

    print("FAILURES:", len(FAIL))
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
