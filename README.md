# Hybrid Loading Optimizer

Educational prototype of a **hybrid quantum-classical system** for a simplified transport loading problem.

The project demonstrates how a logistics task can be converted into a small binary optimization problem and solved with a simplified QAOA-like workflow on a local Qiskit Aer simulator.

## What's new in this version

- Language selection: **English / Russian**.
- Input mode selection:
  - use `data/sample_input.json`;
  - enter cargo weights and truck capacities manually.
- Brute-force validation for small examples.

## Problem

Cargo items must be assigned to exactly **2 trucks**.

Each cargo item is encoded by one bit:

- bit `0` means the cargo item is assigned to truck 0;
- bit `1` means the cargo item is assigned to truck 1.

The objective function includes:

1. a large penalty for exceeding truck capacity;
2. a smaller penalty for uneven loading of trucks.

The result is validated with brute-force search over all possible assignments.

## Project structure

```text
hybrid-loading-optimizer/
├── main.py
├── README.md
├── requirements.txt
├── data/
│   └── sample_input.json
└── outputs/
    └── example_result.txt
```

## Requirements

- Python 3.10 or newer
- pip
- Virtual environment is recommended

## Installation

Clone or unpack the project folder, then open a terminal inside the project directory.

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it.

Windows:

```bash
.venv\Scripts\activate
```

macOS/Linux:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run: interactive mode

```bash
python main.py
```

The program will ask:

```text
Choose language / Выберите язык:
1 — Русский
2 — English

Choose input mode:
1 — Use data/sample_input.json
2 — Enter data manually
```

## Run: non-interactive examples

English interface, JSON file input:

```bash
python main.py --lang en --input-mode file
```

Russian interface, JSON file input:

```bash
python main.py --lang ru --input-mode file
```

Manual input mode:

```bash
python main.py --lang ru --input-mode manual
```

Example manual values:

```text
Cargo weights / Веса грузов: 4 5 6
Truck capacities / Грузоподъемность машин: 9 10
```

## Input data format

The default input file is located at:

```text
data/sample_input.json
```

Example:

```json
{
  "cargo_weights": [4, 5, 6],
  "truck_capacities": [9, 10],
  "shots": 1024,
  "max_iterations": 40,
  "initial_params": [0.5, 0.5],
  "overload_penalty_weight": 100.0,
  "balance_penalty_weight": 1.0,
  "seed_simulator": 42,
  "seed_transpiler": 42
}
```

## Expected output

The exact measurement counts may differ because the algorithm is probabilistic, but the program should print:

- input cargo weights and truck capacities;
- optimized QAOA parameters;
- final measurement counts;
- best QAOA solution;
- brute-force validation result.

Example:

```text
=== Input data ===
Cargo weights: [4, 5, 6]
Truck capacities: [9, 10]

=== Best QAOA solution ===
Bitstring: 001
Assignments:
cargo_0 -> truck_0
cargo_1 -> truck_0
cargo_2 -> truck_1
Truck loads: [9, 6]
Solution cost: 3.0

=== Brute-force validation ===
Best brute-force bitstring: 001
Brute-force cost: 3.0
Brute-force loads: [9, 6]

Validation result: PASS — QAOA found an optimal-cost solution.
```

If the selected bitstring differs but the cost is the same as brute-force, the result is also valid because the mini-task can have several equivalent optimal solutions.

## Technical notes

This is an educational prototype. The QAOA cost layer is simplified and does not encode a full industrial QUBO model. The purpose is to demonstrate the hybrid workflow:

```text
input data -> cost function -> QAOA circuit -> classical optimizer -> measured bitstrings -> validation
```

## Limitations

- Supports exactly 2 trucks.
- Uses one qubit per cargo item.
- Manual mode is limited to small examples, recommended: 2–8 cargo items.
- Industrial-scale loading problems require decomposition, stronger classical baselines, more advanced QUBO construction, and integration with WMS/ERP/TMS systems.

## Suggested next steps

- Add a classical greedy baseline.
- Add support for cargo volumes and incompatibility constraints.
- Move the optimization code into separate modules under `src/`.
- Add unit tests with `pytest`.
- Add a Streamlit interface for visualization.
