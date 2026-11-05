# Diagnostic test sequencing: experiments

Code and results for comparing an exact solver, classical heuristics and
QUBO-based (Hopfield, simulated bifurcation) solvers on the optimal diagnostic
test sequencing problem (single fault, perfect binary tests, fault priors,
test costs). Everything runs on a CPU.

## Contents

- `tsq/circuits.py` — ISCAS'85 structural Verilog parser, bit-parallel
  stuck-at fault simulation (stem faults).
- `tsq/problems.py` — synthetic instances and instances from fault
  simulation of ISCAS'85 circuits.
- `tsq/solvers.py` — exact AND/OR dynamic programming with an admissible
  Huffman lower bound (and without it), greedy (information gain per cost,
  balanced split per cost), rollout, simulated annealing over static orders.
- `tsq/qubo.py` — pairwise (linear-ordering) QUBO surrogate of the
  static-order cost, continuous Hopfield-Tank network, ballistic simulated
  bifurcation (bSB), annealing on the surrogate.
- `run_experiments.py` — main runs (resumable; results in `results/*.jsonl`).
- `run_qubo.py` — QUBO loss decomposition and penalty sensitivity.
- `analyze.py` — tables (`results/tables`, CSV + Markdown) and figures
  (`results/figures`).
- `pilot.py` — first pilot run.
- `data/iscas85/` — circuits and their source (`SOURCE.txt`).

## Reproduce

```
pip install numpy scipy pandas matplotlib tabulate
export PYTHONPATH=.
python3 run_experiments.py synthetic 10 20 30 40 60 80 100   # several hours on 2 cores
EXACT_TIME_LIMIT=240 NOBOUND_LIMIT=30 python3 run_experiments.py iscas c432 c499 c880 c1355 c1908
python3 run_qubo.py decomposition
python3 run_qubo.py penalty
python3 analyze.py
```

Settings used for the reported results: synthetic runs with exact time limit
600 s (m <= 80) and 240 s (part of m = 100), unbounded baseline only for
m <= 60 with a 60 s limit; ISCAS runs with 240 s / 30 s. All instances are
generated from fixed seeds encoded in the `key` field of every result line.

Software: Python 3, NumPy 2.5.3, SciPy 1.18.1, pandas 3.0.5, Matplotlib 3.11.2.
Hardware: 2 CPU cores, 7 GB RAM (cloud container).

## Additions (November 2026)
- `tsq/noisy.py`, `run_noisy.py`, `analyze_noisy.py` — imperfect tests: exact DP over information states, greedy, rollout, perfect-test tree, fault dictionary (`results/noisy.jsonl`).
- `tsq/analog.py`, `run_analog.py` — KHN state-variable filter case study: MNA validated against ngspice 42, Monte Carlo tolerances (`results/analog.json`).
- `tsq/solvers.py: ao_star`, `run_ao.py`, `analyze_ao.py` — classical AO* (Pattipati & Alexandridis 1990) versus branch and bound (`results/ao_compare.jsonl`).
- `verify.py`, `verify_noisy.py`, `check_numbers.py` — independent correctness checks and recomputation of every number in the paper.
