"""Test-sequencing problem instances.

A problem has m candidate faults (the system is known to be faulty, single
fault assumption), prior probabilities p, n binary pass/fail tests with costs c,
and a detection matrix D (D[i, j] = 1 if test j fails when fault i is present).
Tests are perfect (no false alarms or missed detections).

Internally each test is stored as a bitmask over faults: bit i of fail[j] is 1
when fault i makes test j fail.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np

from .circuits import Circuit, detection_matrix, stem_faults


@dataclass
class Problem:
    name: str
    p: np.ndarray          # (m,) priors, sum to 1
    c: np.ndarray          # (n,) test costs
    D: np.ndarray          # (m, n) bool detection matrix
    meta: dict

    @property
    def m(self) -> int:
        return len(self.p)

    @property
    def n(self) -> int:
        return len(self.c)

    @property
    def fail(self) -> list[int]:
        masks = []
        for j in range(self.n):
            word = 0
            for i in np.nonzero(self.D[:, j])[0]:
                word |= 1 << int(i)
            masks.append(word)
        return masks

    def to_json(self) -> dict:
        return {"name": self.name, "p": self.p.tolist(), "c": self.c.tolist(),
                "D": self.D.astype(int).tolist(), "meta": self.meta}

    @staticmethod
    def from_json(d: dict) -> "Problem":
        return Problem(d["name"], np.array(d["p"]), np.array(d["c"], dtype=float),
                       np.array(d["D"], dtype=bool), d["meta"])


def _priors(rng, m, skew):
    alpha = {"uniform": 50.0, "moderate": 1.0, "skewed": 0.3}[skew]
    p = rng.dirichlet(np.full(m, alpha))
    return p / p.sum()


def _costs(rng, n, cost_mode):
    if cost_mode == "unit":
        return np.ones(n)
    return rng.integers(1, 11, size=n).astype(float)


def synthetic(m: int, n: int, seed: int, skew: str = "moderate",
              cost_mode: str = "random", density=(0.2, 0.6)) -> Problem:
    """Random detection matrix with all fault signatures distinct."""
    rng = np.random.default_rng(seed)
    q = rng.uniform(*density, size=n)
    for _ in range(1000):
        D = rng.random((m, n)) < q
        if len({tuple(r) for r in D}) == m:
            break
    else:
        raise RuntimeError("could not draw a fully diagnosable matrix")
    return Problem(f"syn_m{m}_n{n}_{skew}_{cost_mode}_s{seed}", _priors(rng, m, skew),
                   _costs(rng, n, cost_mode), D,
                   {"kind": "synthetic", "m": m, "n": n, "skew": skew,
                    "cost_mode": cost_mode, "seed": seed})


class CircuitPool:
    """Random test-vector pool and fault signatures for one circuit."""

    def __init__(self, circuit: Circuit, n_vectors: int = 256, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.circuit = circuit
        self.vectors = rng.integers(0, 2, size=(n_vectors, len(circuit.inputs))).tolist()
        self.faults = stem_faults(circuit)
        rows = detection_matrix(circuit, self.vectors, self.faults)
        # keep detected faults with a unique signature over the whole pool
        seen, keep = set(), []
        for f, r in zip(self.faults, rows):
            if r and r not in seen:
                seen.add(r)
                keep.append((f, r))
        self.detectable = keep
        self.n_vectors = n_vectors
        self.n_faults_total = len(self.faults)

    def instance(self, m: int, n: int, seed: int, skew: str = "moderate",
                 cost_mode: str = "random") -> Problem:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(self.detectable), size=m, replace=False)
        chosen = [self.detectable[i] for i in idx]
        sig = [r for _, r in chosen]
        V = self.n_vectors
        # pick tests until all fault pairs are distinguished, then pad randomly
        tests: list[int] = []
        order = rng.permutation(V).tolist()
        cols = {t: [(r >> t) & 1 for r in sig] for t in order}
        useful = [t for t in order if 0 < sum(cols[t]) < m]

        def classes(ts):
            keys = {}
            for i in range(m):
                keys.setdefault(tuple(cols[t][i] for t in ts), []).append(i)
            return [v for v in keys.values() if len(v) > 1]

        # greedy: add the vector that separates the most still-confused pairs
        amb = classes(tests)
        while amb:
            best_t, best_gain = None, 0
            for t in useful:
                if t in tests:
                    continue
                gain = 0
                for grp in amb:
                    k = sum(cols[t][i] for i in grp)
                    gain += k * (len(grp) - k)
                if gain > best_gain:
                    best_t, best_gain = t, gain
            if best_t is None:
                break
            tests.append(best_t)
            amb = classes(tests)
        for t in useful:
            if len(tests) >= n:
                break
            if t not in tests:
                tests.append(t)
        rng.shuffle(tests)
        D = np.array([[(r >> t) & 1 for t in tests] for r in sig], dtype=bool)
        # priors: proportional to the fan-in of the gate that drives the faulty net
        weight = np.array([self.circuit.fanin_count.get(f[0], 1) for f, _ in chosen], float)
        weight *= rng.lognormal(0.0, 0.5, size=m)
        if skew == "skewed":
            weight = weight ** 3
        p = weight / weight.sum()
        return Problem(f"{self.circuit.name}_m{m}_n{len(tests)}_{skew}_{cost_mode}_s{seed}", p,
                       _costs(rng, len(tests), cost_mode), D,
                       {"kind": "iscas85", "circuit": self.circuit.name, "m": m,
                        "n": len(tests), "skew": skew, "cost_mode": cost_mode,
                        "seed": seed, "faults": [list(f) for f, _ in chosen],
                        "vectors": tests})


def save(problems: list[Problem], path: str) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump([pr.to_json() for pr in problems], fh)


def load(path: str) -> list[Problem]:
    with open(path, encoding="utf-8") as fh:
        return [Problem.from_json(d) for d in json.load(fh)]
