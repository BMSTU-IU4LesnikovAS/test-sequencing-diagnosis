"""Analog case study: KHN (Kerwin-Huelsman-Newcomb) state-variable filter.

Linear AC analysis by modified nodal analysis (MNA) in NumPy, validated
against ngspice. Faults: open / short / +50 % / -50 % of every passive
component and gain degradation of every op-amp. Tests: gain magnitude (dB) at
the LP, BP, HP outputs at four frequencies. Component tolerances are handled by
Monte Carlo: a test "fails" when the measured gain leaves the band that holds
99.8 % of fault-free circuits; P(fail | fault) is estimated from Monte Carlo.
"""
from __future__ import annotations

import os
import subprocess
import tempfile

import numpy as np

# ---------------------------------------------------------------- circuit
# passive elements: name -> (type, node+, node-, nominal value)
PASSIVES = {
    "R1": ("R", "in", "n1m", 10e3),
    "R2": ("R", "lp", "n1m", 10e3),
    "R3": ("R", "hp", "n1m", 10e3),
    "R4": ("R", "bp", "n1p", 50e3),
    "R5": ("R", "n1p", "0", 10e3),
    "R6": ("R", "hp", "n2m", 10e3),
    "C1": ("C", "n2m", "bp", 1.6e-9),
    "R7": ("R", "bp", "n3m", 10e3),
    "C2": ("C", "n3m", "lp", 1.6e-9),
}
# op-amps: name -> (non-inverting, inverting, output)
OPAMPS = {"A1": ("n1p", "n1m", "hp"), "A2": ("0", "n2m", "bp"), "A3": ("0", "n3m", "lp")}
A0, GBW = 2e5, 1e6               # single-pole op-amp model
TOL = {"R": 0.01, "C": 0.05}      # component tolerances (3 sigma), op-amp GBW +-20 %
NODES = ["in", "n1m", "n1p", "hp", "n2m", "bp", "n3m", "lp"]
OUTPUTS = ["lp", "bp", "hp"]
FREQS = [1e3, 1e4, 2e4, 1e5]
OPEN_R, SHORT_R = 1e9, 1e-3       # an open or shorted element


def default_params():
    p = {k: v[3] for k, v in PASSIVES.items()}
    for a in OPAMPS:
        p[a + ".A0"] = A0
        p[a + ".GBW"] = GBW
    return p


def faults():
    """(name, modifier) pairs; the modifier maps a parameter dict to a faulty one."""
    out = []
    for k, (t, *_r) in PASSIVES.items():
        out.append((f"{k} open", k, "open"))
        out.append((f"{k} short", k, "short"))
        out.append((f"{k} +50%", k, 1.5))
        out.append((f"{k} -50%", k, 0.5))
    for a in OPAMPS:
        out.append((f"{a} degraded", a, "degraded"))
    return out


def apply_fault(params: dict, kinds: dict, fault) -> tuple[dict, dict]:
    """Returns (params, element kind overrides): an open capacitor becomes a
    1 GOhm resistor, a shorted one 1 mOhm; a degraded op-amp has A0 = 100."""
    _name, el, mod = fault
    params, kinds = dict(params), dict(kinds)
    if el in OPAMPS:
        params[el + ".A0"] = 100.0
        params[el + ".GBW"] = params[el + ".GBW"] * 100.0 / A0
        return params, kinds
    if mod == "open":
        kinds[el] = "R"
        params[el] = OPEN_R
    elif mod == "short":
        kinds[el] = "R"
        params[el] = SHORT_R
    else:
        params[el] = params[el] * mod
    return params, kinds


# ---------------------------------------------------------------- MNA
def mna_gain_db(params: dict, kinds: dict | None = None, freqs=FREQS) -> np.ndarray:
    """|V(out)/V(in)| in dB for every output and frequency: array (len(OUTPUTS), len(freqs))."""
    kinds = kinds or {}
    idx = {n: i for i, n in enumerate(NODES)}
    N = len(NODES)
    nv = 1 + len(OPAMPS)                     # input source + one VCVS per op-amp
    res = np.empty((len(OUTPUTS), len(freqs)))
    for fi, f in enumerate(freqs):
        s = 2j * np.pi * f
        Y = np.zeros((N + nv, N + nv), complex)
        b = np.zeros(N + nv, complex)

        def stamp(a, bnode, y):
            ia, ib = idx.get(a), idx.get(bnode)
            if ia is not None:
                Y[ia, ia] += y
            if ib is not None:
                Y[ib, ib] += y
            if ia is not None and ib is not None:
                Y[ia, ib] -= y
                Y[ib, ia] -= y
        for k, (t, a, bnode, _v) in PASSIVES.items():
            t = kinds.get(k, t)
            val = params[k]
            stamp(a, bnode, 1.0 / val if t == "R" else s * val)
        # input voltage source: V(in) = 1
        r = N
        Y[r, idx["in"]] = 1.0
        Y[idx["in"], r] = 1.0
        b[r] = 1.0
        # op-amps as VCVS: V(out) = A(s) (V+ - V-), A(s) = A0 / (1 + s A0 / (2 pi GBW))
        for q, (a, (pn, mn, on)) in enumerate(OPAMPS.items()):
            r = N + 1 + q
            A = params[a + ".A0"] / (1 + s * params[a + ".A0"] / (2 * np.pi * params[a + ".GBW"]))
            Y[idx[on], r] += 1.0                 # source current into the output node
            Y[r, idx[on]] += 1.0
            if pn in idx:
                Y[r, idx[pn]] -= A
            if mn in idx:
                Y[r, idx[mn]] += A
        x = np.linalg.solve(Y, b)
        for oi, o in enumerate(OUTPUTS):
            res[oi, fi] = 20 * np.log10(abs(x[idx[o]]))
    return res


# ---------------------------------------------------------------- ngspice
def ngspice_gain_db(params: dict, kinds: dict | None = None, freqs=FREQS) -> np.ndarray:
    """Same quantity computed by ngspice (op-amp: VCVS + RC pole + buffer)."""
    kinds = kinds or {}
    lines = ["* KHN state-variable filter", "Vin in 0 AC 1"]
    for k, (t, a, bnode, _v) in PASSIVES.items():
        t = kinds.get(k, t)
        lines.append(f"{t}{k} {a} {bnode} {params[k]:.10g}")
    for a, (pn, mn, on) in OPAMPS.items():
        A, G = params[a + ".A0"], params[a + ".GBW"]
        fp = G / A
        Rp, Cp = 1e3, 1.0 / (2 * np.pi * 1e3 * fp)
        lines += [f"E{a}g {a}x 0 {pn} {mn} {A:.10g}", f"R{a}p {a}x {a}y {Rp:.10g}",
                  f"C{a}p {a}y 0 {Cp:.10g}", f"E{a}o {on} 0 {a}y 0 1"]
    out = np.empty((len(OUTPUTS), len(freqs)))
    with tempfile.TemporaryDirectory() as d:
        for fi, f in enumerate(freqs):
            data = os.path.join(d, f"o{fi}.txt")
            net = lines + [".control", f"ac lin 1 {f:.10g} {f:.10g}",
                           f"wrdata {data} vdb(lp) vdb(bp) vdb(hp)", "quit", ".endc", ".end"]
            cir = os.path.join(d, f"c{fi}.cir")
            open(cir, "w").write("\n".join(net) + "\n")
            subprocess.run(["ngspice", "-b", cir], capture_output=True, check=True)
            vals = open(data).read().split()
            nums = [float(v) for v in vals]
            # wrdata writes "freq value" pairs for each vector
            out[:, fi] = [nums[1], nums[3], nums[5]]
    return out


# ---------------------------------------------------------------- Monte Carlo
def sample_params(rng, base: dict) -> dict:
    p = dict(base)
    for k, (t, *_r) in PASSIVES.items():
        p[k] = base[k] * (1 + rng.normal(0, TOL[t] / 3))
    for a in OPAMPS:
        p[a + ".GBW"] = base[a + ".GBW"] * (1 + rng.normal(0, 0.2 / 3))
    return p


def build_case(n_mc: int = 300, n_nom: int = 2000, seed: int = 0):
    """Returns dict with test names, fault names, P(fail | fault) matrix and the
    tolerance bands."""
    rng = np.random.default_rng(seed)
    base = default_params()
    nom = np.array([mna_gain_db(sample_params(rng, base)).ravel() for _ in range(n_nom)])
    lo = np.quantile(nom, 0.001, axis=0)
    hi = np.quantile(nom, 0.999, axis=0)
    fl = faults()
    pf = np.empty((len(fl), lo.size))
    for i, f in enumerate(fl):
        G = []
        for _ in range(n_mc):
            p, k = apply_fault(sample_params(rng, base), {}, f)
            G.append(mna_gain_db(p, k).ravel())
        G = np.array(G)
        pf[i] = ((G < lo) | (G > hi)).mean(axis=0)
    tests = [f"{o.upper()} @ {f/1e3:g} kHz" for o in OUTPUTS for f in FREQS]
    return {"tests": tests, "faults": [f[0] for f in fl], "pf": pf, "lo": lo, "hi": hi,
            "nominal_db": mna_gain_db(base).ravel()}
