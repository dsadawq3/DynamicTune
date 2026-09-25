"""Parameterized structured probes with deterministic train/validation/holdout splits."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import hashlib
import numpy as np

from .types import Probe, ProbeSplit


@dataclass(frozen=True)
class ProbeGeneratorConfig:
    state_dim: int = 8
    per_family: int = 12
    symbolic_steps: int = 8
    dependency_nodes: int = 12
    control_nodes: int = 14
    reaction_atoms: int = 9
    context_length: int = 256
    seed: int = 17
    validation_fraction: float = 0.20
    holdout_fraction: float = 0.20


class ProbeGenerator:
    families = ("symbolic", "dependency_chain", "control_data_flow", "reaction_graph", "long_context", "counterfactual", "adversarial_stability")

    def __init__(self, config: ProbeGeneratorConfig | None = None) -> None:
        self.config = config or ProbeGeneratorConfig()
        if self.config.state_dim < 2 or self.config.per_family < 3:
            raise ValueError("state_dim >= 2 and per_family >= 3 are required")

    def generate(self) -> list[Probe]:
        rng = np.random.default_rng(self.config.seed)
        result: list[Probe] = []
        for family in self.families:
            for index in range(self.config.per_family):
                initial = rng.normal(0.0, 1.0, self.config.state_dim)
                payload = self._payload(family, index, rng)
                pair_id = None
                perturbation = None
                if family in {"counterfactual", "adversarial_stability"}:
                    pair_id = f"{'cf' if family == 'counterfactual' else 'adv'}-{index:04d}"
                    perturbation = rng.normal(0.0, 0.10 if family == "counterfactual" else 0.02, self.config.state_dim)
                    base_payload = {**payload, "base_case": True, "paired_evaluation": True}
                    alt_payload = {**payload, "base_case": False, "paired_evaluation": True, "counterfactual_delta_norm": float(np.linalg.norm(perturbation))}
                    result.append(Probe(f"{family}-{index:04d}-base", family, base_payload, initial, pair_id, None))
                    result.append(Probe(f"{family}-{index:04d}-perturbed", family, alt_payload, initial + perturbation, pair_id, perturbation))
                    continue
                result.append(Probe(f"{family}-{index:04d}", family, payload, initial, pair_id, perturbation))
        return result

    def split(self, probes: Sequence[Probe] | None = None) -> ProbeSplit:
        values = list(probes if probes is not None else self.generate())
        grouped: dict[str, list[Probe]] = {family: [] for family in self.families}
        for probe in values:
            grouped.setdefault(probe.family, []).append(probe)
        rng = np.random.default_rng(self.config.seed + 1009)
        train, validation, holdout = [], [], []
        for family, members in grouped.items():
            grouped_units: dict[str, list[Probe]] = {}
            for probe in members:
                grouped_units.setdefault(probe.pair_id or probe.probe_id, []).append(probe)
            unit_ids = list(grouped_units)
            order = rng.permutation(len(unit_ids))
            n_holdout = max(1, int(round(len(unit_ids) * self.config.holdout_fraction)))
            n_validation = max(1, int(round(len(unit_ids) * self.config.validation_fraction)))
            for rank, pos in enumerate(order):
                bucket = "holdout" if rank < n_holdout else "validation" if rank < n_holdout + n_validation else "train"
                for probe in grouped_units[unit_ids[int(pos)]]:
                    updated = Probe(probe.probe_id, probe.family, probe.payload, probe.initial_state, probe.pair_id, probe.perturbation, bucket)
                    {"train": train, "validation": validation, "holdout": holdout}[bucket].append(updated)
        return ProbeSplit(tuple(train), tuple(validation), tuple(holdout))

    def _payload(self, family: str, index: int, rng: np.random.Generator) -> dict[str, Any]:
        if family == "symbolic":
            operations = [str(x) for x in rng.choice(["add", "mul", "mod", "xor", "compare", "shift"], self.config.symbolic_steps)]
            return {"expression_dag": [{"node": i, "op": op, "parents": [max(0, i - 1), max(0, i - 2)]} for i, op in enumerate(operations)], "answer_depth": self.config.symbolic_steps, "dependency_width": 2}
        if family == "dependency_chain":
            edges = [[i, i + 1] for i in range(self.config.dependency_nodes - 1)]
            edges += [[i, i + 3] for i in range(self.config.dependency_nodes - 3) if i % 2 == 0]
            return {"nodes": self.config.dependency_nodes, "edges": edges, "topological_order": list(range(self.config.dependency_nodes))}
        if family == "control_data_flow":
            edges = [[i, i + 1] for i in range(self.config.control_nodes - 1)]
            edges += [[i, min(self.config.control_nodes - 1, i + 3)] for i in range(0, self.config.control_nodes - 3, 3)]
            return {"basic_blocks": self.config.control_nodes, "control_edges": edges, "data_edges": [[0, i] for i in range(1, self.config.control_nodes, 2)], "branch_points": list(range(2, self.config.control_nodes, 4))}
        if family == "reaction_graph":
            bonds = [[i, i + 1, int(rng.integers(1, 4))] for i in range(self.config.reaction_atoms - 1)]
            return {"reactants": [{"element": str(rng.choice(["C", "N", "O", "S"])), "charge": int(rng.integers(-1, 2))} for _ in range(self.config.reaction_atoms)], "bonds": bonds, "reaction_center": [index % self.config.reaction_atoms, (index + 2) % self.config.reaction_atoms]}
        if family == "long_context":
            anchors = sorted(set(int(x) for x in rng.integers(0, self.config.context_length, size=7)))
            return {"context_length": self.config.context_length, "anchor_positions": anchors, "distractor_count": self.config.context_length - len(anchors), "query_position": self.config.context_length - 1}
        if family == "counterfactual":
            return {"base_graph": [[0, 1], [1, 2], [2, 3]], "intervention": {"node": index % 4, "value": int(index % 3)}, "paired_evaluation": True}
        return {"perturbation_scale": 0.02, "directions": int(self.config.state_dim), "stability_horizon": 6 + index % 5, "near_boundary": bool(index % 2)}


def stable_probe_vector(probe: Probe, state_dim: int) -> np.ndarray:
    """A deterministic non-semantic embedding for synthetic systems only."""

    if state_dim < 1:
        raise ValueError("state_dim must be positive")
    digest = hashlib.blake2b(probe.probe_id.encode("utf-8"), digest_size=32).digest()
    seed = int.from_bytes(digest[:8], "little", signed=False)
    rng = np.random.default_rng(seed)
    vector = rng.normal(size=state_dim)
    return vector / max(np.linalg.norm(vector), 1e-12)


class PythonProbeGenerator:
    """Deterministic structured Python code probe generator with pair-safe train/val/holdout splits."""

    families = (
        "python_functions",
        "python_control_flow",
        "python_data_structures",
        "python_classes",
        "python_algorithms",
        "python_counterfactual",
        "python_adversarial",
    )

    def __init__(self, config: ProbeGeneratorConfig | None = None) -> None:
        self.config = config or ProbeGeneratorConfig()
        if self.config.state_dim < 2 or self.config.per_family < 3:
            raise ValueError("state_dim >= 2 and per_family >= 3 are required")

    def generate(self) -> list[Probe]:
        rng = np.random.default_rng(self.config.seed)
        result: list[Probe] = []
        for family in self.families:
            for index in range(self.config.per_family):
                initial = rng.normal(0.0, 1.0, self.config.state_dim)
                pair_id = None
                perturbation = None
                if family in {"python_counterfactual", "python_adversarial"}:
                    pair_id = f"{'pycf' if family == 'python_counterfactual' else 'pyadv'}-{index:04d}"
                    perturbation = rng.normal(0.0, 0.10 if family == "python_counterfactual" else 0.02, self.config.state_dim)
                    base_payload, alt_payload = self._paired_payloads(family, index, rng)
                    base_payload["paired_evaluation"] = True
                    base_payload["base_case"] = True
                    alt_payload["paired_evaluation"] = True
                    alt_payload["base_case"] = False
                    alt_payload["counterfactual_delta_norm"] = float(np.linalg.norm(perturbation))
                    result.append(Probe(f"{family}-{index:04d}-base", family, base_payload, initial, pair_id, None))
                    result.append(Probe(f"{family}-{index:04d}-perturbed", family, alt_payload, initial + perturbation, pair_id, perturbation))
                    continue
                payload = self._payload(family, index, rng)
                result.append(Probe(f"{family}-{index:04d}", family, payload, initial, pair_id, perturbation))
        return result

    def split(self, probes: Sequence[Probe] | None = None) -> ProbeSplit:
        values = list(probes if probes is not None else self.generate())
        grouped: dict[str, list[Probe]] = {family: [] for family in self.families}
        for probe in values:
            grouped.setdefault(probe.family, []).append(probe)
        rng = np.random.default_rng(self.config.seed + 2027)
        train, validation, holdout = [], [], []
        for family, members in grouped.items():
            grouped_units: dict[str, list[Probe]] = {}
            for probe in members:
                grouped_units.setdefault(probe.pair_id or probe.probe_id, []).append(probe)
            unit_ids = list(grouped_units)
            order = rng.permutation(len(unit_ids))
            n_holdout = max(1, int(round(len(unit_ids) * self.config.holdout_fraction)))
            n_validation = max(1, int(round(len(unit_ids) * self.config.validation_fraction)))
            for rank, pos in enumerate(order):
                bucket = "holdout" if rank < n_holdout else "validation" if rank < n_holdout + n_validation else "train"
                for probe in grouped_units[unit_ids[int(pos)]]:
                    updated = Probe(probe.probe_id, probe.family, probe.payload, probe.initial_state, probe.pair_id, probe.perturbation, bucket)
                    {"train": train, "validation": validation, "holdout": holdout}[bucket].append(updated)
        return ProbeSplit(tuple(train), tuple(validation), tuple(holdout))

    def _payload(self, family: str, index: int, rng: np.random.Generator) -> dict[str, Any]:
        if family == "python_functions":
            ops = ["a + b", "a * b", "max(a, b)", "a - b", "pow(a, b, 1000)", "a // max(1, b)"]
            op = ops[index % len(ops)]
            factor = int(rng.integers(2, 10))
            code = (
                f"def compute_step_{index}(a: int, b: int = {factor}) -> int:\n"
                f"    \"\"\"Execute step {index} mathematical operation.\"\"\"\n"
                f"    result = {op}\n"
                f"    return result + {factor}\n"
            )
            return {
                "code": code,
                "domain": "python",
                "ast_kind": "FunctionDef",
                "function_name": f"compute_step_{index}",
                "operation": op,
                "arg_count": 2,
                "has_docstring": True,
            }
        if family == "python_control_flow":
            threshold = int(rng.integers(5, 50))
            step = int(rng.integers(1, 4))
            code = (
                f"def scan_window_{index}(items: list[int], limit: int = {threshold}) -> list[int]:\n"
                f"    collected = []\n"
                f"    for i in range(0, len(items), {step}):\n"
                f"        val = items[i]\n"
                f"        if val > limit:\n"
                f"            collected.append(val)\n"
                f"        elif val < 0:\n"
                f"            break\n"
                f"    return collected\n"
            )
            return {
                "code": code,
                "domain": "python",
                "ast_kind": "ControlFlowLoop",
                "threshold": threshold,
                "step": step,
                "has_conditional": True,
            }
        if family == "python_data_structures":
            key_name = f"key_{index % 5}"
            code = (
                f"def build_index_{index}(records: list[dict]) -> dict[str, int]:\n"
                f"    mapping = {{}}\n"
                f"    for r in records:\n"
                f"        val = r.get('{key_name}', {index})\n"
                f"        mapping[val] = mapping.get(val, 0) + 1\n"
                f"    return {{k: v for k, v in mapping.items() if v > 1}}\n"
            )
            return {
                "code": code,
                "domain": "python",
                "ast_kind": "DictComprehension",
                "lookup_key": key_name,
                "data_structures": ["dict", "list"],
            }
        if family == "python_classes":
            capacity = int(rng.integers(16, 128))
            code = (
                f"class StateBuffer_{index}:\n"
                f"    \"\"\"Buffer state tracker {index}.\"\"\"\n"
                f"    def __init__(self, capacity: int = {capacity}):\n"
                f"        self.capacity = capacity\n"
                f"        self.items = []\n"
                f"    def add(self, item: int) -> bool:\n"
                f"        if len(self.items) < self.capacity:\n"
                f"            self.items.append(item)\n"
                f"            return True\n"
                f"        return False\n"
            )
            return {
                "code": code,
                "domain": "python",
                "ast_kind": "ClassDef",
                "class_name": f"StateBuffer_{index}",
                "capacity": capacity,
                "methods": ["__init__", "add"],
            }
        if family == "python_algorithms":
            algo_type = index % 4
            if algo_type == 0:
                code = (
                    f"def binary_search_{index}(sorted_list: list[int], target: int) -> int:\n"
                    f"    lo, hi = 0, len(sorted_list) - 1\n"
                    f"    while lo <= hi:\n"
                    f"        mid = (lo + hi) // 2\n"
                    f"        if sorted_list[mid] == target:\n"
                    f"            return mid\n"
                    f"        elif sorted_list[mid] < target:\n"
                    f"            lo = mid + 1\n"
                    f"        else:\n"
                    f"            hi = mid - 1\n"
                    f"    return -1\n"
                )
                algo_name = "binary_search"
            elif algo_type == 1:
                code = (
                    f"def greatest_common_divisor_{index}(a: int, b: int) -> int:\n"
                    f"    while b != 0:\n"
                    f"        a, b = b, a % b\n"
                    f"    return abs(a)\n"
                )
                algo_name = "euclidean_gcd"
            elif algo_type == 2:
                code = (
                    f"def partition_array_{index}(arr: list[int], low: int, high: int) -> int:\n"
                    f"    pivot = arr[high]\n"
                    f"    i = low - 1\n"
                    f"    for j in range(low, high):\n"
                    f"        if arr[j] <= pivot:\n"
                    f"            i += 1\n"
                    f"            arr[i], arr[j] = arr[j], arr[i]\n"
                    f"    arr[i + 1], arr[high] = arr[high], arr[i + 1]\n"
                    f"    return i + 1\n"
                )
                algo_name = "lomuto_partition"
            else:
                code = (
                    f"def is_prime_number_{index}(n: int) -> bool:\n"
                    f"    if n < 2:\n"
                    f"        return False\n"
                    f"    for d in range(2, int(n ** 0.5) + 1):\n"
                    f"        if n % d == 0:\n"
                    f"            return False\n"
                    f"    return True\n"
                )
                algo_name = "prime_trial_division"
            return {
                "code": code,
                "domain": "python",
                "ast_kind": "Algorithm",
                "algorithm_name": algo_name,
                "complexity": "O(log N)" if algo_type in {0, 1} else "O(N)",
            }
        return {"perturbation_scale": 0.02, "directions": int(self.config.state_dim), "stability_horizon": 6 + index % 5}

    def _paired_payloads(self, family: str, index: int, rng: np.random.Generator) -> tuple[dict[str, Any], dict[str, Any]]:
        if family == "python_counterfactual":
            bound = int(rng.integers(10, 100))
            base_code = (
                f"def check_bound_{index}(val: int, bound: int = {bound}) -> bool:\n"
                f"    \"\"\"Return whether value strictly exceeds bound.\"\"\"\n"
                f"    return val > bound\n"
            )
            alt_code = (
                f"def check_bound_{index}(val: int, bound: int = {bound}) -> bool:\n"
                f"    \"\"\"Return whether value is strictly below bound.\"\"\"\n"
                f"    return val < bound\n"
            )
            base_payload = {
                "code": base_code,
                "domain": "python",
                "ast_kind": "CounterfactualPredicate",
                "operator": ">",
                "bound": bound,
            }
            alt_payload = {
                "code": alt_code,
                "domain": "python",
                "ast_kind": "CounterfactualPredicate",
                "operator": "<",
                "bound": bound,
            }
            return base_payload, alt_payload
        depth = int(rng.integers(2, 6))
        base_code = (
            f"def make_evaluators_{index}(count: int = {depth}):\n"
            f"    \"\"\"Generate early-bound closure evaluators.\"\"\"\n"
            f"    return [lambda x, idx=i: x * idx + {index} for i in range(count)]\n"
        )
        alt_code = (
            f"def make_evaluators_{index}(count: int = {depth}):\n"
            f"    \"\"\"Generate late-bound closure evaluators.\"\"\"\n"
            f"    return [lambda x: x * i + {index} for i in range(count)]\n"
        )
        base_payload = {
            "code": base_code,
            "domain": "python",
            "ast_kind": "ClosureBinding",
            "scoping": "early_bound_default",
            "depth": depth,
        }
        alt_payload = {
            "code": alt_code,
            "domain": "python",
            "ast_kind": "ClosureBinding",
            "scoping": "late_bound_free_var",
            "depth": depth,
        }
        return base_payload, alt_payload
