"""Hybrid Loading Optimizer.

Educational prototype of a hybrid quantum-classical system for a small
transport loading problem. The example assigns 3 cargo items to 2 trucks and
uses a simplified QAOA-like circuit on a local Qiskit Aer simulator.

The prototype also validates the result with a brute-force search over all
possible assignments.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from scipy.optimize import minimize

try:
    from qiskit import QuantumCircuit, transpile
    from qiskit_aer import AerSimulator
except ImportError as exc:  # pragma: no cover - user-facing dependency error
    raise SystemExit(
        "Qiskit dependencies are not installed. Run: pip install -r requirements.txt"
    ) from exc


DEFAULT_CONFIG_PATH = Path("data/sample_input.json")


@dataclass(frozen=True)
class LoadingConfig:
    """Configuration for the loading optimization prototype.

    Args:
        cargo_weights: Weight of each cargo item.
        truck_capacities: Maximum capacity of each truck.
        shots: Number of measurement shots for the simulator.
        max_iterations: Maximum number of COBYLA iterations.
        initial_params: Initial QAOA parameters [gamma, beta].
        overload_penalty_weight: Penalty multiplier for capacity violations.
        balance_penalty_weight: Penalty multiplier for uneven loading.
        seed_simulator: Random seed for the simulator.
        seed_transpiler: Random seed for transpilation.

    Example:
        config = LoadingConfig(
            cargo_weights=np.array([4, 5, 6]),
            truck_capacities=np.array([9, 10]),
            shots=1024,
            max_iterations=40,
            initial_params=np.array([0.5, 0.5]),
            overload_penalty_weight=100.0,
            balance_penalty_weight=1.0,
            seed_simulator=42,
            seed_transpiler=42,
        )
    """

    cargo_weights: np.ndarray
    truck_capacities: np.ndarray
    shots: int
    max_iterations: int
    initial_params: np.ndarray
    overload_penalty_weight: float
    balance_penalty_weight: float
    seed_simulator: int
    seed_transpiler: int

    @property
    def n_cargo(self) -> int:
        """Return the number of cargo items.

        Returns:
            Number of cargo items represented by qubits.
        """
        return len(self.cargo_weights)

    @property
    def n_trucks(self) -> int:
        """Return the number of available trucks.

        Returns:
            Number of trucks in the simplified prototype.
        """
        return len(self.truck_capacities)


def load_config(config_path: Path = DEFAULT_CONFIG_PATH) -> LoadingConfig:
    """Load experiment configuration from a JSON file.

    Args:
        config_path: Path to the JSON configuration file.

    Returns:
        Parsed loading optimization configuration.

    Example:
        config = load_config(Path("data/sample_input.json"))
    """
    with config_path.open("r", encoding="utf-8") as file:
        raw_config = json.load(file)

    return LoadingConfig(
        cargo_weights=np.array(raw_config["cargo_weights"], dtype=float),
        truck_capacities=np.array(raw_config["truck_capacities"], dtype=float),
        shots=int(raw_config.get("shots", 1024)),
        max_iterations=int(raw_config.get("max_iterations", 40)),
        initial_params=np.array(raw_config.get("initial_params", [0.5, 0.5]), dtype=float),
        overload_penalty_weight=float(raw_config.get("overload_penalty_weight", 100.0)),
        balance_penalty_weight=float(raw_config.get("balance_penalty_weight", 1.0)),
        seed_simulator=int(raw_config.get("seed_simulator", 42)),
        seed_transpiler=int(raw_config.get("seed_transpiler", 42)),
    )


def validate_config(config: LoadingConfig) -> None:
    """Validate basic assumptions of the prototype.

    The current educational encoding uses one qubit per cargo item and one bit
    to select a truck. Therefore it supports exactly two trucks.

    Args:
        config: Loading optimization configuration.

    Raises:
        ValueError: If the configuration is not compatible with the prototype.

    Example:
        validate_config(config)
    """
    if config.n_trucks != 2:
        raise ValueError("This prototype supports exactly 2 trucks.")
    if config.n_cargo == 0:
        raise ValueError("At least one cargo item is required.")
    if np.any(config.cargo_weights <= 0):
        raise ValueError("Cargo weights must be positive.")
    if np.any(config.truck_capacities <= 0):
        raise ValueError("Truck capacities must be positive.")
    if config.initial_params.shape != (2,):
        raise ValueError("initial_params must contain exactly [gamma, beta].")


def calculate_truck_loads(bitstring: str, cargo_weights: np.ndarray) -> np.ndarray:
    """Calculate truck loads for a binary assignment string.

    In this simplified encoding, each bit corresponds to one cargo item:
    ``0`` assigns the item to truck 0, while ``1`` assigns it to truck 1.

    Args:
        bitstring: Binary assignment string, e.g. ``"001"``.
        cargo_weights: Weight of each cargo item.

    Returns:
        NumPy array with two values: loads of truck 0 and truck 1.

    Example:
        calculate_truck_loads("001", np.array([4, 5, 6]))
        # returns array([9., 6.])
    """
    loads = np.zeros(2, dtype=float)

    for cargo_index, bit in enumerate(bitstring):
        truck_index = int(bit)
        loads[truck_index] += cargo_weights[cargo_index]

    return loads


def calculate_overload_penalty(
    loads: np.ndarray,
    capacities: np.ndarray,
    penalty_weight: float,
) -> float:
    """Calculate penalty for exceeding truck capacities.

    Args:
        loads: Current loads of trucks.
        capacities: Maximum capacities of trucks.
        penalty_weight: Multiplier for overload penalty.

    Returns:
        Quadratic overload penalty.

    Example:
        calculate_overload_penalty(np.array([11, 4]), np.array([9, 10]), 100.0)
        # returns 400.0
    """
    penalty = 0.0
    for load, capacity in zip(loads, capacities):
        overload = max(0.0, load - capacity)
        penalty += penalty_weight * overload**2
    return float(penalty)


def calculate_balance_penalty(loads: np.ndarray, penalty_weight: float) -> float:
    """Calculate penalty for uneven load distribution.

    Args:
        loads: Current loads of trucks.
        penalty_weight: Multiplier for load-balance penalty.

    Returns:
        Absolute load difference multiplied by the penalty weight.

    Example:
        calculate_balance_penalty(np.array([9, 6]), 1.0)
        # returns 3.0
    """
    return float(penalty_weight * abs(loads[0] - loads[1]))


def calculate_solution_cost(bitstring: str, config: LoadingConfig) -> float:
    """Calculate total cost of a cargo assignment.

    The cost contains two components:
    1. A large quadratic penalty for capacity violations.
    2. A smaller penalty for uneven load distribution.

    Args:
        bitstring: Binary assignment string.
        config: Loading optimization configuration.

    Returns:
        Total solution cost.

    Example:
        config = load_config()
        calculate_solution_cost("001", config)
    """
    loads = calculate_truck_loads(bitstring, config.cargo_weights)
    overload_penalty = calculate_overload_penalty(
        loads,
        config.truck_capacities,
        config.overload_penalty_weight,
    )
    balance_penalty = calculate_balance_penalty(loads, config.balance_penalty_weight)
    return overload_penalty + balance_penalty


def decode_solution(bitstring: str) -> Dict[str, str]:
    """Convert a binary assignment string into a human-readable plan.

    Args:
        bitstring: Binary assignment string.

    Returns:
        Dictionary mapping cargo labels to truck labels.

    Example:
        decode_solution("001")
        # returns {'cargo_0': 'truck_0', 'cargo_1': 'truck_0', 'cargo_2': 'truck_1'}
    """
    return {
        f"cargo_{cargo_index}": f"truck_{int(bit)}"
        for cargo_index, bit in enumerate(bitstring)
    }


def create_qaoa_circuit(gamma: float, beta: float, config: LoadingConfig) -> QuantumCircuit:
    """Create a simplified p=1 QAOA-like circuit.

    The circuit is intentionally compact and educational. It does not encode the
    full QUBO exactly; instead, it demonstrates the hybrid QAOA workflow for a
    representative small loading problem.

    Args:
        gamma: Cost-layer parameter.
        beta: Mixer-layer parameter.
        config: Loading optimization configuration.

    Returns:
        Quantum circuit with measurements.

    Example:
        config = load_config()
        circuit = create_qaoa_circuit(0.5, 0.5, config)
    """
    circuit = QuantumCircuit(config.n_cargo, config.n_cargo)

    # Create a uniform superposition over all possible assignments.
    circuit.h(range(config.n_cargo))

    # Simplified cost layer. The RZ rotations bias states according to cargo
    # weights, while ZZ interactions introduce dependencies between decisions.
    for cargo_index, weight in enumerate(config.cargo_weights):
        circuit.rz(2 * gamma * weight, cargo_index)

    for cargo_index in range(config.n_cargo - 1):
        circuit.cx(cargo_index, cargo_index + 1)
        circuit.rz(2 * gamma, cargo_index + 1)
        circuit.cx(cargo_index, cargo_index + 1)

    # Mixer layer explores neighboring binary assignments.
    for qubit_index in range(config.n_cargo):
        circuit.rx(2 * beta, qubit_index)

    circuit.measure(range(config.n_cargo), range(config.n_cargo))
    return circuit


def run_circuit(
    gamma: float,
    beta: float,
    config: LoadingConfig,
    simulator: AerSimulator,
) -> Dict[str, int]:
    """Run the QAOA circuit on a local Aer simulator.

    Args:
        gamma: Cost-layer parameter.
        beta: Mixer-layer parameter.
        config: Loading optimization configuration.
        simulator: Qiskit Aer simulator instance.

    Returns:
        Measurement counts as a dictionary mapping bitstrings to shot counts.

    Example:
        simulator = AerSimulator(seed_simulator=42)
        counts = run_circuit(0.5, 0.5, config, simulator)
    """
    circuit = create_qaoa_circuit(gamma, beta, config)
    compiled = transpile(
        circuit,
        simulator,
        seed_transpiler=config.seed_transpiler,
    )
    result = simulator.run(compiled, shots=config.shots).result()
    return result.get_counts()


def expected_cost_from_counts(counts: Dict[str, int], config: LoadingConfig) -> float:
    """Estimate expected assignment cost from measurement counts.

    Qiskit returns classical bitstrings in little-endian order for this circuit,
    so each bitstring is reversed before interpreting it as a cargo assignment.

    Args:
        counts: Measurement counts from the simulator.
        config: Loading optimization configuration.

    Returns:
        Expected cost averaged over measured bitstrings.

    Example:
        expected_cost_from_counts({'100': 512, '001': 512}, config)
    """
    total_shots = sum(counts.values())
    average_cost = 0.0

    for raw_bitstring, count in counts.items():
        corrected_bitstring = raw_bitstring[::-1]
        average_cost += calculate_solution_cost(corrected_bitstring, config) * count

    return average_cost / total_shots


def optimize_qaoa(config: LoadingConfig) -> Tuple[np.ndarray, float, List[float]]:
    """Optimize QAOA parameters with COBYLA.

    Args:
        config: Loading optimization configuration.

    Returns:
        Tuple containing optimal parameters, final expected cost, and energy
        history recorded at each optimizer call.

    Example:
        params, cost, history = optimize_qaoa(config)
    """
    simulator = AerSimulator(seed_simulator=config.seed_simulator)
    history: List[float] = []

    def objective(params: np.ndarray) -> float:
        gamma, beta = params
        counts = run_circuit(gamma, beta, config, simulator)
        avg_cost = expected_cost_from_counts(counts, config)
        history.append(avg_cost)
        return avg_cost

    result = minimize(
        objective,
        config.initial_params,
        method="COBYLA",
        options={"maxiter": config.max_iterations},
    )

    return result.x, float(result.fun), history


def select_best_measured_solution(counts: Dict[str, int], config: LoadingConfig) -> str:
    """Select the best valid-looking solution from measured bitstrings.

    The selection is based on the classical cost function, not only on the most
    frequent bitstring. This makes the prototype more robust because QAOA output
    is probabilistic and can contain several high-probability candidates.

    Args:
        counts: Measurement counts from the final QAOA run.
        config: Loading optimization configuration.

    Returns:
        Best corrected bitstring according to the cost function.

    Example:
        best = select_best_measured_solution({'100': 300, '001': 724}, config)
    """
    return min(counts.keys(), key=lambda bit: calculate_solution_cost(bit[::-1], config))[::-1]


def brute_force_search(config: LoadingConfig) -> Tuple[str, float, np.ndarray]:
    """Find the exact optimum by checking all assignments.

    This validation is feasible only for small educational examples because the
    number of assignments grows exponentially with the number of cargo items.

    Args:
        config: Loading optimization configuration.

    Returns:
        Tuple with best bitstring, best cost, and truck loads.

    Example:
        bitstring, cost, loads = brute_force_search(config)
    """
    candidates = []

    for bits in product("01", repeat=config.n_cargo):
        bitstring = "".join(bits)
        cost = calculate_solution_cost(bitstring, config)
        loads = calculate_truck_loads(bitstring, config.cargo_weights)
        candidates.append((bitstring, cost, loads))

    return min(candidates, key=lambda item: item[1])


def format_assignment(assignment: Dict[str, str]) -> str:
    """Format cargo-to-truck assignment for console output.

    Args:
        assignment: Mapping from cargo labels to truck labels.

    Returns:
        Multi-line string with one assignment per line.

    Example:
        print(format_assignment({'cargo_0': 'truck_0'}))
    """
    return "\n".join(f"{cargo} -> {truck}" for cargo, truck in assignment.items())


def main() -> None:
    """Run the full educational optimization workflow.

    The workflow includes configuration loading, QAOA optimization, final circuit
    execution, best-solution decoding, and brute-force validation.
    """
    config = load_config()
    validate_config(config)

    optimal_params, optimal_expected_cost, history = optimize_qaoa(config)
    simulator = AerSimulator(seed_simulator=config.seed_simulator)
    final_counts = run_circuit(optimal_params[0], optimal_params[1], config, simulator)

    best_bitstring = select_best_measured_solution(final_counts, config)
    best_assignment = decode_solution(best_bitstring)
    best_loads = calculate_truck_loads(best_bitstring, config.cargo_weights)
    best_cost = calculate_solution_cost(best_bitstring, config)

    brute_bitstring, brute_cost, brute_loads = brute_force_search(config)

    print("=== Input data ===")
    print("Cargo weights:", config.cargo_weights.astype(int).tolist())
    print("Truck capacities:", config.truck_capacities.astype(int).tolist())

    print("\n=== QAOA parameters ===")
    print("gamma_opt:", round(float(optimal_params[0]), 6))
    print("beta_opt:", round(float(optimal_params[1]), 6))
    print("optimal_expected_cost:", round(optimal_expected_cost, 6))

    print("\n=== Final measurement counts ===")
    print(final_counts)

    print("\n=== Best QAOA solution ===")
    print("Bitstring:", best_bitstring)
    print("Assignments:")
    print(format_assignment(best_assignment))
    print("Truck loads:", best_loads.astype(int).tolist())
    print("Solution cost:", round(best_cost, 6))

    print("\n=== Brute-force validation ===")
    print("Best brute-force bitstring:", brute_bitstring)
    print("Brute-force cost:", round(brute_cost, 6))
    print("Brute-force loads:", brute_loads.astype(int).tolist())

    print("\n=== Optimization history ===")
    print([round(value, 6) for value in history])

    if abs(best_cost - brute_cost) < 1e-9:
        print("\nValidation result: PASS — QAOA found an optimal-cost solution.")
    else:
        print("\nValidation result: CHECK — QAOA found a non-optimal candidate.")


if __name__ == "__main__":
    main()
