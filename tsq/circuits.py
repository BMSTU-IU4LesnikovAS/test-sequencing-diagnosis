"""Gate-level netlist parsing and bit-parallel stuck-at fault simulation.

Reads structural Verilog netlists of the ISCAS'85 benchmarks (one primitive
gate per statement) and simulates many input vectors at once by packing them
into the bits of Python integers.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

GATE_TYPES = {"and", "nand", "or", "nor", "xor", "xnor", "not", "buf"}


@dataclass
class Circuit:
    name: str
    inputs: list[str]
    outputs: list[str]
    gates: list[tuple[str, str, list[str]]]  # (type, output net, input nets), topologically ordered
    fanin_count: dict[str, int] = field(default_factory=dict)

    @property
    def nets(self) -> list[str]:
        return self.inputs + [g[1] for g in self.gates]


def parse_verilog(path: str) -> Circuit:
    text = open(path, encoding="utf-8").read()
    text = re.sub(r"//[^\n]*", "", text)
    name = re.search(r"module\s+(\w+)", text).group(1)
    inputs, outputs, raw_gates = [], [], []
    for stmt in text.split(";"):
        stmt = " ".join(stmt.split())
        if not stmt:
            continue
        kw = stmt.split(" ", 1)[0]
        if kw == "input":
            inputs += [s.strip() for s in stmt[len("input"):].split(",") if s.strip()]
        elif kw == "output":
            outputs += [s.strip() for s in stmt[len("output"):].split(",") if s.strip()]
        elif kw in GATE_TYPES:
            ports = re.search(r"\((.*)\)", stmt).group(1)
            nets = [s.strip() for s in ports.split(",")]
            raw_gates.append((kw, nets[0], nets[1:]))
    # topological order
    driver = {g[1]: g for g in raw_gates}
    ordered, done = [], set(inputs)
    pending = list(raw_gates)
    while pending:
        progress = False
        rest = []
        for g in pending:
            if all(i in done for i in g[2]):
                ordered.append(g)
                done.add(g[1])
                progress = True
            else:
                rest.append(g)
        if not progress:
            raise ValueError(f"combinational loop or undriven net in {path}")
        pending = rest
    fanin = {g[1]: len(g[2]) for g in ordered}
    assert set(driver) == {g[1] for g in ordered}
    return Circuit(name, inputs, outputs, ordered, fanin)


def _eval(gtype: str, vals: list[int], mask: int) -> int:
    if gtype in ("and", "nand"):
        r = mask
        for v in vals:
            r &= v
    elif gtype in ("or", "nor"):
        r = 0
        for v in vals:
            r |= v
    elif gtype in ("xor", "xnor"):
        r = 0
        for v in vals:
            r ^= v
    elif gtype in ("not", "buf"):
        r = vals[0]
    else:
        raise ValueError(gtype)
    if gtype in ("nand", "nor", "xnor", "not"):
        r = ~r & mask
    return r


def simulate(c: Circuit, pi_values: dict[str, int], nvec: int,
             fault: tuple[str, int] | None = None) -> dict[str, int]:
    """Simulate nvec vectors in parallel. fault = (net, stuck_value) on a stem."""
    mask = (1 << nvec) - 1
    val = dict(pi_values)
    if fault is not None and fault[0] in val:
        val[fault[0]] = mask if fault[1] else 0
    for gtype, out, ins in c.gates:
        v = _eval(gtype, [val[i] for i in ins], mask)
        if fault is not None and out == fault[0]:
            v = mask if fault[1] else 0
        val[out] = v
    return {o: val[o] for o in c.outputs}


def stem_faults(c: Circuit) -> list[tuple[str, int]]:
    """Single stuck-at-0/1 faults on every net (stem faults only)."""
    return [(n, s) for n in c.nets for s in (0, 1)]


def detection_matrix(c: Circuit, vectors: list[list[int]],
                     faults: list[tuple[str, int]]) -> list[int]:
    """For every fault return an nvec-bit integer: bit k = 1 if vector k detects it
    (any primary output differs from the fault-free response)."""
    nvec = len(vectors)
    pi = {}
    for idx, name in enumerate(c.inputs):
        word = 0
        for k, vec in enumerate(vectors):
            if vec[idx]:
                word |= 1 << k
        pi[name] = word
    good = simulate(c, pi, nvec)
    rows = []
    for f in faults:
        bad = simulate(c, pi, nvec, f)
        diff = 0
        for o in c.outputs:
            diff |= good[o] ^ bad[o]
        rows.append(diff)
    return rows
