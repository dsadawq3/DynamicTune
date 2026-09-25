"""Small deterministic dynamical systems used for local scientific verification."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Mapping

import numpy as np

from .types import Probe, finite_array


@dataclass
class LinearResidualSystem:
    matrices: np.ndarray
    biases: np.ndarray
    dt: float = 0.15
    model_name: str = "linear-residual"

    def __post_init__(self) -> None:
        self.matrices = finite_array(self.matrices, ndim=3, name="system matrices")
        self.biases = finite_array(self.biases, ndim=2, name="system biases")
        if self.matrices.shape[0] != self.biases.shape[0] or self.matrices.shape[1] != self.matrices.shape[2] or self.biases.shape[1] != self.matrices.shape[1]:
            raise ValueError("system dimensions are inconsistent")
        if self.dt <= 0:
            raise ValueError("system dt must be positive")
        self.layer_count = int(self.matrices.shape[0])
        self.state_dim = int(self.matrices.shape[1])
        self.has_residual_state = True

    def initial_state(self, probe: Probe) -> np.ndarray:
        if probe.initial_state.size != self.state_dim:
            raise ValueError("probe initial state has the wrong dimension")
        return probe.initial_state.copy()

    def step(self, state: np.ndarray, layer_index: int, probe: Probe) -> np.ndarray:
        x = finite_array(state, ndim=1, name="state")
        probe_drive = 0.01 * np.sin(float(np.dot(x, probe.initial_state)))
        return x + self.dt * (x @ self.matrices[layer_index] + self.biases[layer_index] + probe_drive)

    def residual_state(self, state: np.ndarray, layer_index: int, probe: Probe) -> np.ndarray:
        return finite_array(state, ndim=1, name="residual")

    def weights(self) -> Mapping[str, np.ndarray]:
        return {f"layers.{i}.weight": np.eye(self.state_dim) + self.dt * self.matrices[i] for i in range(self.layer_count)} | {f"layers.{i}.bias": self.biases[i].copy() for i in range(self.layer_count)}

    def set_weights(self, values: Mapping[str, np.ndarray]) -> None:
        for i in range(self.layer_count):
            self.matrices[i] = (np.asarray(values[f"layers.{i}.weight"]) - np.eye(self.state_dim)) / self.dt
            self.biases[i] = np.asarray(values[f"layers.{i}.bias"]).copy()


@dataclass
class NonlinearEmergentSystem(LinearResidualSystem):
    nonlinearity: float = 0.08
    model_name: str = "nonlinear-emergent"

    def step(self, state: np.ndarray, layer_index: int, probe: Probe) -> np.ndarray:
        x = finite_array(state, ndim=1, name="state")
        drive = 0.01 * np.sin(float(np.dot(x, probe.initial_state)))
        field = x @ self.matrices[layer_index] + self.biases[layer_index] + drive
        return x + self.dt * (field + self.nonlinearity * np.tanh(field))


@dataclass
class QuadraticResidualSystem(LinearResidualSystem):
    """Polynomial residual system with an explicit second-order vector field."""

    quadratic_terms: np.ndarray = None  # type: ignore[assignment]
    model_name: str = "quadratic-residual"

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.quadratic_terms is None:
            raise ValueError("quadratic_terms are required")
        self.quadratic_terms = finite_array(self.quadratic_terms, ndim=4, name="quadratic terms")
        if self.quadratic_terms.shape != (self.layer_count, self.state_dim, self.state_dim, self.state_dim):
            raise ValueError("quadratic_terms must have shape [layers, output_dim, input_dim, input_dim]")

    def step(self, state: np.ndarray, layer_index: int, probe: Probe) -> np.ndarray:
        x = finite_array(state, ndim=1, name="state")
        field = x @ self.matrices[layer_index] + self.biases[layer_index]
        field = field + np.einsum("jik,i,k->j", self.quadratic_terms[layer_index], x, x)
        return x + self.dt * field


@dataclass
class BifurcationSystem(LinearResidualSystem):
    branch_gain: float = 0.40
    model_name: str = "bifurcation-system"

    def step(self, state: np.ndarray, layer_index: int, probe: Probe) -> np.ndarray:
        x = finite_array(state, ndim=1, name="state")
        field = x @ self.matrices[layer_index] + self.biases[layer_index]
        branch = np.zeros_like(x)
        branch[0] = self.branch_gain * (float(probe.initial_state[0]) * x[0] - x[0] ** 3)
        return x + self.dt * (field + branch)


def random_stable_system(*, layers: int = 6, state_dim: int = 4, seed: int = 7, dt: float = 0.10, nonlinear: bool = False) -> LinearResidualSystem:
    rng = np.random.default_rng(seed)
    matrices = []
    for _ in range(layers):
        raw = rng.normal(size=(state_dim, state_dim))
        raw = raw / max(np.linalg.svd(raw, compute_uv=False)[0], 1e-12)
        matrices.append(0.35 * raw - 0.12 * np.eye(state_dim))
    biases = rng.normal(0, 0.03, size=(layers, state_dim))
    cls = NonlinearEmergentSystem if nonlinear else LinearResidualSystem
    return cls(np.asarray(matrices), biases, dt=dt)


class FakeLayer:
    def __init__(self, weight: np.ndarray, bias: np.ndarray) -> None:
        self.weight = np.asarray(weight).copy()
        self.bias = np.asarray(bias).copy()

    def transition(self, state: np.ndarray, probe: Probe) -> np.ndarray:
        return finite_array(state, ndim=1, name="fake state") @ self.weight + self.bias


class FakeHFLikeModel:
    """Framework-free object with familiar config/layers/get_weights hooks."""

    def __init__(self, *, layers: int = 4, state_dim: int = 4, seed: int = 11) -> None:
        rng = np.random.default_rng(seed)
        self.state_dim = state_dim
        self.config = SimpleNamespace(num_hidden_layers=layers, hidden_size=state_dim)
        self.layers = [FakeLayer(np.eye(state_dim) + 0.02 * rng.normal(size=(state_dim, state_dim)), 0.01 * rng.normal(size=state_dim)) for _ in range(layers)]

    def initial_state(self, probe: Probe) -> np.ndarray:
        return probe.initial_state.copy()

    def get_weights(self) -> Mapping[str, np.ndarray]:
        result = {}
        for i, layer in enumerate(self.layers):
            result[f"layers.{i}.weight"] = layer.weight.copy()
            result[f"layers.{i}.bias"] = layer.bias.copy()
        return result

    def set_weights(self, values: Mapping[str, np.ndarray]) -> None:
        for i, layer in enumerate(self.layers):
            layer.weight = np.asarray(values[f"layers.{i}.weight"]).copy()
            layer.bias = np.asarray(values[f"layers.{i}.bias"]).copy()
