"""Hybrid Loading Optimizer.

Educational prototype of a hybrid quantum-classical system for a small
transport loading problem. The program assigns cargo items to 2 trucks and uses
a simplified QAOA-like circuit on a local Qiskit Aer simulator.

New in this version:
- interactive language selection: English / Russian;
- input mode selection: JSON file or manual console input;
- validation with brute-force search for small examples.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Dict, List, Literal, Tuple

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
DEFAULT_SHOTS = 1024
DEFAULT_MAX_ITERATIONS = 40
DEFAULT_INITIAL_PARAMS = np.array([0.5, 0.5], dtype=float)
DEFAULT_OVERLOAD_PENALTY_WEIGHT = 100.0
DEFAULT_BALANCE_PENALTY_WEIGHT = 1.0
DEFAULT_SEED_SIMULATOR = 42
DEFAULT_SEED_TRANSPILER = 42
MAX_MANUAL_CARGO_ITEMS = 8

Language = Literal["en", "ru"]
InputMode = Literal["file", "manual"]


TRANSLATIONS: Dict[str, Dict[Language, str]] = {
    "choose_language": {
        "en": "Choose language / Выберите язык:\n1 — Русский\n2 — English\n> ",
        "ru": "Выберите язык / Choose language:\n1 — Русский\n2 — English\n> ",
    },
    "choose_input_mode": {
        "en": "Choose input mode:\n1 — Use data/sample_input.json\n2 — Enter data manually\n> ",
        "ru": "Выберите режим ввода данных:\n1 — Использовать data/sample_input.json\n2 — Ввести данные вручную\n> ",
    },
    "enter_cargo_weights": {
        "en": "Enter cargo weights separated by spaces (example: 4 5 6): ",
        "ru": "Введите веса грузов через пробел (пример: 4 5 6): ",
    },
    "enter_truck_capacities": {
        "en": "Enter exactly 2 truck capacities separated by spaces (example: 9 10): ",
        "ru": "Введите грузоподъемность ровно 2 машин через пробел (пример: 9 10): ",
    },
    "invalid_number_list": {
        "en": "Invalid input. Please enter positive numbers separated by spaces.",
        "ru": "Некорректный ввод. Введите положительные числа через пробел.",
    },
    "invalid_truck_count": {
        "en": "This prototype supports exactly 2 trucks.",
        "ru": "Этот прототип поддерживает ровно 2 машины.",
    },
    "too_many_cargo": {
        "en": f"Too many cargo items for an educational demo. Use no more than {MAX_MANUAL_CARGO_ITEMS}.",
        "ru": f"Слишком много грузов для учебной демонстрации. Используйте не больше {MAX_MANUAL_CARGO_ITEMS}.",
    },
    "input_data": {"en": "=== Input data ===", "ru": "=== Входные данные ==="},
    "cargo_weights": {"en": "Cargo weights:", "ru": "Веса грузов:"},
    "truck_capacities": {"en": "Truck capacities:", "ru": "Грузоподъемность машин:"},
    "qaoa_params": {"en": "=== QAOA parameters ===", "ru": "=== Параметры QAOA ==="},
    "final_counts": {
        "en": "=== Final measurement counts ===",
        "ru": "=== Финальное распределение измерений ===",
    },
    "best_solution": {"en": "=== Best QAOA solution ===", "ru": "=== Лучшее решение QAOA ==="},
    "bitstring": {"en": "Bitstring:", "ru": "Битовая строка:"},
    "assignments": {"en": "Assignments:", "ru": "Назначения:"},
    "truck_loads": {"en": "Truck loads:", "ru": "Загрузка машин:"},
    "solution_cost": {"en": "Solution cost:", "ru": "Стоимость решения:"},
    "brute_validation": {
        "en": "=== Brute-force validation ===",
        "ru": "=== Проверка полным перебором ===",
    },
    "brute_bitstring": {
        "en": "Best brute-force bitstring:",
        "ru": "Лучшая битовая строка brute force:",
    },
    "brute_cost": {"en": "Brute-force cost:", "ru": "Стоимость brute force:"},
    "brute_loads": {"en": "Brute-force loads:", "ru": "Загрузка brute force:"},
    "history": {"en": "=== Optimization history ===", "ru": "=== История оптимизации ==="},
    "validation_pass": {
        "en": "Validation result: PASS — QAOA found an optimal-cost solution.",
        "ru": "Результат проверки: PASS — QAOA нашел решение с оптимальной стоимостью.",
    },
    "validation_check": {
        "en": "Validation result: CHECK — QAOA found a non-optimal candidate.",
        "ru": "Результат проверки: CHECK — QAOA нашел неоптимальный кандидат.",
    },
}


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
        """Return the number of cargo items."""
        return len(self.cargo_weights)

    @property
    def n_trucks(self) -> int:
        """Return the number of available trucks."""
        return len(self.truck_capacities)


def tr(key: str, language: Language) -> str:
    """Return a translated interface message.

    Args:
        key: Translation dictionary key.
        language: Interface language code.

    Returns:
        Localized message.
    """
    return TRANSLATIONS[key][language]


def parse_number_list(raw_value: str) -> np.ndarray:
    """Parse a whitespace-separated list of positive numbers.

    Args:
        raw_value: User input such as ``"4 5 6"``.

    Returns:
        NumPy array of positive floats.

    Raises:
        ValueError: If the input contains non-positive or invalid values.
    """
    values = np.array([float(item) for item in raw_value.split()], dtype=float)
    if values.size == 0 or np.any(values <= 0):
        raise ValueError("All values must be positive.")
    return values


def ask_language() -> Language:
    """Ask the user to choose the interface language."""
    while True:
        choice = input(tr("choose_language", "ru")).strip()
        if choice == "1":
            return "ru"
        if choice == "2":
            return "en"
        print("Введите 1 или 2 / Enter 1 or 2.")


def ask_input_mode(language: Language) -> InputMode:
    """Ask the user to choose between JSON input and manual input."""
    while True:
        choice = input(tr("choose_input_mode", language)).strip()
        if choice == "1":
            return "file"
        if choice == "2":
            return "manual"
        print("Введите 1 или 2." if language == "ru" else "Enter 1 or 2.")


def create_config_from_values(cargo_weights: np.ndarray, truck_capacities: np.ndarray) -> LoadingConfig:
    """Create a config object using default algorithm parameters.

    Args:
        cargo_weights: User-defined cargo weights.
        truck_capacities: User-defined capacities for exactly two trucks.

    Returns:
        LoadingConfig with default QAOA and validation settings.
    """
    return LoadingConfig(
        cargo_weights=cargo_weights,
        truck_capacities=truck_capacities,
        shots=DEFAULT_SHOTS,
        max_iterations=DEFAULT_MAX_ITERATIONS,
        initial_params=DEFAULT_INITIAL_PARAMS.copy(),
        overload_penalty_weight=DEFAULT_OVERLOAD_PENALTY_WEIGHT,
        balance_penalty_weight=DEFAULT_BALANCE_PENALTY_WEIGHT,
        seed_simulator=DEFAULT_SEED_SIMULATOR,
        seed_transpiler=DEFAULT_SEED_TRANSPILER,
    )


def load_config(config_path: Path = DEFAULT_CONFIG_PATH) -> LoadingConfig:
    """Load experiment configuration from a JSON file.

    Args:
        config_path: Path to the JSON configuration file.

    Returns:
        Parsed loading optimization configuration.
    """
    with config_path.open("r", encoding="utf-8") as file:
        raw_config = json.load(file)

    return LoadingConfig(
        cargo_weights=np.array(raw_config["cargo_weights"], dtype=float),
        truck_capacities=np.array(raw_config["truck_capacities"], dtype=float),
        shots=int(raw_config.get("shots", DEFAULT_SHOTS)),
        max_iterations=int(raw_config.get("max_iterations", DEFAULT_MAX_ITERATIONS)),
        initial_params=np.array(raw_config.get("initial_params", [0.5, 0.5]), dtype=float),
        overload_penalty_weight=float(
            raw_config.get("overload_penalty_weight", DEFAULT_OVERLOAD_PENALTY_WEIGHT)
        ),
        balance_penalty_weight=float(
            raw_config.get("balance_penalty_weight", DEFAULT_BALANCE_PENALTY_WEIGHT)
        ),
        seed_simulator=int(raw_config.get("seed_simulator", DEFAULT_SEED_SIMULATOR)),
        seed_transpiler=int(raw_config.get("seed_transpiler", DEFAULT_SEED_TRANSPILER)),
    )


def load_manual_config(language: Language) -> LoadingConfig:
    """Load problem data from console input.

    Args:
        language: Interface language used for prompts and validation messages.

    Returns:
        LoadingConfig based on user-entered weights and truck capacities.
    """
    while True:
        try:
            cargo_weights = parse_number_list(input(tr("enter_cargo_weights", language)))
            if cargo_weights.size > MAX_MANUAL_CARGO_ITEMS:
                print(tr("too_many_cargo", language))
                continue
            truck_capacities = parse_number_list(input(tr("enter_truck_capacities", language)))
            if truck_capacities.size != 2:
                print(tr("invalid_truck_count", language))
                continue
            config = create_config_from_values(cargo_weights, truck_capacities)
            validate_config(config)
            return config
        except ValueError:
            print(tr("invalid_number_list", language))


def validate_config(config: LoadingConfig) -> None:
    """Validate basic assumptions of the prototype.

    The current educational encoding uses one qubit per cargo item and one bit
    to select a truck. Therefore it supports exactly two trucks.

    Args:
        config: Loading optimization configuration.

    Raises:
        ValueError: If the configuration is not compatible with the prototype.
    """
    if config.n_trucks != 2:
        raise ValueError("This prototype supports exactly 2 trucks.")
    if config.n_cargo == 0:
        raise ValueError("At least one cargo item is required.")
    if config.n_cargo > MAX_MANUAL_CARGO_ITEMS:
        raise ValueError(f"Use no more than {MAX_MANUAL_CARGO_ITEMS} cargo items.")
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
    """Calculate penalty for exceeding truck capacities."""
    penalty = 0.0
    for load, capacity in zip(loads, capacities):
        overload = max(0.0, load - capacity)
        penalty += penalty_weight * overload**2
    return float(penalty)


def calculate_balance_penalty(loads: np.ndarray, penalty_weight: float) -> float:
    """Calculate penalty for uneven load distribution."""
    return float(penalty_weight * abs(loads[0] - loads[1]))


def calculate_solution_cost(bitstring: str, config: LoadingConfig) -> float:
    """Calculate total cost of a cargo assignment."""
    loads = calculate_truck_loads(bitstring, config.cargo_weights)
    overload_penalty = calculate_overload_penalty(
        loads,
        config.truck_capacities,
        config.overload_penalty_weight,
    )
    balance_penalty = calculate_balance_penalty(loads, config.balance_penalty_weight)
    return overload_penalty + balance_penalty


def decode_solution(bitstring: str) -> Dict[str, str]:
    """Convert a binary assignment string into a human-readable plan."""
    return {
        f"cargo_{cargo_index}": f"truck_{int(bit)}"
        for cargo_index, bit in enumerate(bitstring)
    }


def create_qaoa_circuit(gamma: float, beta: float, config: LoadingConfig) -> QuantumCircuit:
    """Create a simplified p=1 QAOA-like circuit.

    The circuit is intentionally compact and educational. It does not encode the
    full QUBO exactly; instead, it demonstrates the hybrid QAOA workflow for a
    representative small loading problem.
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
    """Run the QAOA circuit on a local Aer simulator."""
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
    """
    total_shots = sum(counts.values())
    average_cost = 0.0

    for raw_bitstring, count in counts.items():
        corrected_bitstring = raw_bitstring[::-1]
        average_cost += calculate_solution_cost(corrected_bitstring, config) * count

    return average_cost / total_shots


def optimize_qaoa(config: LoadingConfig) -> Tuple[np.ndarray, float, List[float]]:
    """Optimize QAOA parameters with COBYLA."""
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
    """Select the lowest-cost solution from measured bitstrings."""
    return min(counts.keys(), key=lambda bit: calculate_solution_cost(bit[::-1], config))[::-1]


def brute_force_search(config: LoadingConfig) -> Tuple[str, float, np.ndarray]:
    """Find the exact optimum by checking all assignments."""
    candidates = []

    for bits in product("01", repeat=config.n_cargo):
        bitstring = "".join(bits)
        cost = calculate_solution_cost(bitstring, config)
        loads = calculate_truck_loads(bitstring, config.cargo_weights)
        candidates.append((bitstring, cost, loads))

    return min(candidates, key=lambda item: item[1])


def format_assignment(assignment: Dict[str, str]) -> str:
    """Format cargo-to-truck assignment for console output."""
    return "\n".join(f"{cargo} -> {truck}" for cargo, truck in assignment.items())


def format_number_list(values: np.ndarray) -> List[float | int]:
    """Format numeric arrays without .0 where possible."""
    result: List[float | int] = []
    for value in values.tolist():
        result.append(int(value) if float(value).is_integer() else float(value))
    return result


def print_results(
    config: LoadingConfig,
    language: Language,
    optimal_params: np.ndarray,
    optimal_expected_cost: float,
    final_counts: Dict[str, int],
    history: List[float],
) -> None:
    """Print localized optimization results to the console."""
    best_bitstring = select_best_measured_solution(final_counts, config)
    best_assignment = decode_solution(best_bitstring)
    best_loads = calculate_truck_loads(best_bitstring, config.cargo_weights)
    best_cost = calculate_solution_cost(best_bitstring, config)

    brute_bitstring, brute_cost, brute_loads = brute_force_search(config)

    print(tr("input_data", language))
    print(tr("cargo_weights", language), format_number_list(config.cargo_weights))
    print(tr("truck_capacities", language), format_number_list(config.truck_capacities))

    print(f"\n{tr('qaoa_params', language)}")
    print("gamma_opt:", round(float(optimal_params[0]), 6))
    print("beta_opt:", round(float(optimal_params[1]), 6))
    print("optimal_expected_cost:", round(optimal_expected_cost, 6))

    print(f"\n{tr('final_counts', language)}")
    print(final_counts)

    print(f"\n{tr('best_solution', language)}")
    print(tr("bitstring", language), best_bitstring)
    print(tr("assignments", language))
    print(format_assignment(best_assignment))
    print(tr("truck_loads", language), format_number_list(best_loads))
    print(tr("solution_cost", language), round(best_cost, 6))

    print(f"\n{tr('brute_validation', language)}")
    print(tr("brute_bitstring", language), brute_bitstring)
    print(tr("brute_cost", language), round(brute_cost, 6))
    print(tr("brute_loads", language), format_number_list(brute_loads))

    print(f"\n{tr('history', language)}")
    print([round(value, 6) for value in history])

    if abs(best_cost - brute_cost) < 1e-9:
        print(f"\n{tr('validation_pass', language)}")
    else:
        print(f"\n{tr('validation_check', language)}")


def parse_args() -> argparse.Namespace:
    """Parse optional command-line arguments.

    Running without arguments keeps the program interactive. Arguments are useful
    for demonstrations, README examples, and repeatable testing.
    """
    parser = argparse.ArgumentParser(description="Hybrid Loading Optimizer")
    parser.add_argument("--lang", choices=["en", "ru"], help="Interface language")
    parser.add_argument(
        "--input-mode",
        choices=["file", "manual"],
        help="Load data from JSON file or enter it manually",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to JSON config file when --input-mode=file",
    )
    return parser.parse_args()


def main() -> None:
    """Run the full educational optimization workflow."""
    args = parse_args()
    language: Language = args.lang or ask_language()
    input_mode: InputMode = args.input_mode or ask_input_mode(language)

    if input_mode == "file":
        config = load_config(args.config)
        validate_config(config)
    else:
        config = load_manual_config(language)

    optimal_params, optimal_expected_cost, history = optimize_qaoa(config)
    simulator = AerSimulator(seed_simulator=config.seed_simulator)
    final_counts = run_circuit(optimal_params[0], optimal_params[1], config, simulator)

    print_results(
        config=config,
        language=language,
        optimal_params=optimal_params,
        optimal_expected_cost=optimal_expected_cost,
        final_counts=final_counts,
        history=history,
    )


if __name__ == "__main__":
    main()
