"""Rank-aware local jet transport in the student chart.

The implementation estimates a local 1-jet and a directional 2-jet from a
bundle of paired trajectories.  The second-order object is explicitly a
directional Hessian sketch; it is never serialized or described as a full
Hessian tensor.  Teacher samples are first mapped into the student chart, then
projected through the measured teacher and student tangent projectors.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from .depth import MonotoneCorrespondence
from .types import AlignmentResult, TrajectoryTrace, finite_array, stable_l2


def _robust_weights(residual: np.ndarray, *, loss: str, tuning: float) -> np.ndarray:
    values = np.maximum(np.asarray(residual, dtype=np.float64), 0.0)
    scale = max(float(np.median(values)), 1e-8)
    u = values / (tuning * scale)
    if loss == "huber":
        return np.minimum(1.0, 1.0 / np.maximum(u, 1.0))
    if loss == "tukey":
        result = np.zeros_like(u)
        inside = u < 1.0
        result[inside] = (1.0 - u[inside] ** 2) ** 2
        return result
    raise ValueError("robust loss must be 'huber' or 'tukey'")


def _robust_regression(
    design: np.ndarray,
    values: np.ndarray,
    weights: np.ndarray,
    *,
    ridge: float,
    loss: str,
    iterations: int,
    tuning: float,
) -> tuple[np.ndarray, float, float]:
    x = finite_array(design, ndim=2, name="jet design")
    y = finite_array(values, ndim=2, name="jet values")
    base_weights = finite_array(weights, ndim=1, name="jet weights")
    if x.shape[0] != y.shape[0] or len(base_weights) != len(x) or ridge < 0 or not np.isfinite(ridge):
        raise ValueError("jet regression arrays or ridge are invalid")
    if not 1 <= iterations <= 20 or tuning <= 0 or not np.isfinite(tuning):
        raise ValueError("jet robust regression parameters are invalid")
    active = np.maximum(base_weights, 0.0)
    if not np.any(active > 0):
        raise ValueError("jet regression has no positive-weight samples")
    coefficient = np.zeros((x.shape[1], y.shape[1]), dtype=np.float64)
    for _ in range(iterations):
        scale = np.maximum(1.0, np.median(np.abs(x), axis=0))
        normalized = x / scale
        sqrt_weights = np.sqrt(np.maximum(active, 0.0))
        design_weighted = normalized * sqrt_weights[:, None]
        value_weighted = y * sqrt_weights[:, None]
        if design_weighted.shape[1] > design_weighted.shape[0] and ridge > 0.0:
            # Exact dual ridge solve.  It avoids a (d+1)^2 Gram matrix when
            # a local jet has many coordinates but only a small probe bundle.
            dual_gram = design_weighted @ design_weighted.T + ridge * np.eye(design_weighted.shape[0])
            dual_gram = (dual_gram + dual_gram.T) / 2.0
            try:
                dual_solution = np.linalg.solve(dual_gram, value_weighted)
            except np.linalg.LinAlgError:
                dual_solution = np.linalg.lstsq(dual_gram, value_weighted, rcond=1e-10)[0]
            coefficient_normalized = design_weighted.T @ dual_solution
        else:
            gram = design_weighted.T @ design_weighted + ridge * np.eye(x.shape[1])
            gram = (gram + gram.T) / 2.0
            try:
                coefficient_normalized = np.linalg.solve(gram, design_weighted.T @ value_weighted)
            except np.linalg.LinAlgError:
                coefficient_normalized = np.linalg.lstsq(design_weighted, value_weighted, rcond=1e-10)[0]
        coefficient = coefficient_normalized / scale[:, None]
        prediction = x @ coefficient
        residual = np.asarray(stable_l2(prediction - y, axis=1, name="jet regression residual"), dtype=np.float64)
        active = np.maximum(base_weights, 0.0) * _robust_weights(residual, loss=loss, tuning=tuning)
    if not np.all(np.isfinite(coefficient)):
        raise FloatingPointError("local jet regression produced non-finite coefficients")
    residual = np.asarray(stable_l2(x @ coefficient - y, axis=1, name="jet final residual"), dtype=np.float64)
    weighted_residual = float(np.sum(active * residual) / max(np.sum(active), 1e-12))
    uncertainty = float(np.median(np.abs(residual - np.median(residual))) / max(np.median(residual), 1e-8))
    return coefficient, weighted_residual, min(1.0, max(0.0, uncertainty))


def _basis(samples: np.ndarray, rank: int, tolerance: float = 1e-8) -> np.ndarray:
    values = finite_array(samples, ndim=2, name="jet tangent samples")
    centered = values - values.mean(axis=0, keepdims=True)
    if not np.any(np.abs(centered) > 0):
        return np.zeros((values.shape[1], 0), dtype=np.float64)
    _, singular, right_transpose = np.linalg.svd(centered, full_matrices=False)
    if singular.size == 0 or singular[0] <= 0:
        return np.zeros((values.shape[1], 0), dtype=np.float64)
    keep = singular > tolerance * max(1.0, singular[0])
    return right_transpose[keep][: min(rank, int(np.count_nonzero(keep)))].T


def _pair_key(trace: TrajectoryTrace) -> str:
    explicit = trace.metadata.get("probe_pair_id")
    if explicit is not None:
        return str(explicit)
    probe_id = str(trace.probe_id)
    for suffix in ("-base", "-perturbed"):
        if probe_id.endswith(suffix):
            return probe_id[: -len(suffix)]
    return probe_id


def _paired_variation_basis(
    traces: Sequence[TrajectoryTrace],
    node: int,
    rank: int,
    transform: Any | None = None,
) -> tuple[np.ndarray, int]:
    groups: dict[str, list[TrajectoryTrace]] = {}
    for trace in traces:
        groups.setdefault(_pair_key(trace), []).append(trace)
    differences = []
    bundle_count = 0
    for members in groups.values():
        if len(members) < 2:
            continue
        bundle_count += 1
        center = members[0].hidden_states[node]
        if transform is not None:
            center = transform(center)
        for member in members[1:]:
            value = member.hidden_states[node]
            if transform is not None:
                value = transform(value)
            differences.append(value - center)
    if not differences:
        return np.zeros((traces[0].state_dim, 0)), bundle_count
    return _basis(np.asarray(differences), rank), bundle_count


def _paired_variation_basis_at_nodes(
    traces: Sequence[TrajectoryTrace],
    nodes: Sequence[int],
    rank: int,
    transform: Any | None = None,
    transform_accepts_depth: bool = False,
) -> tuple[np.ndarray, int]:
    """Recover a perturbation basis using one depth-mapped node per trace.

    Teacher and student charts may expose different layer counts.  A shared
    integer index is therefore not a valid correspondence.  This helper keeps
    the perturbation pairing while allowing the caller to choose the node
    nearest the continuous teacher-depth image of each student node.
    """

    if len(traces) != len(nodes) or not traces:
        raise ValueError("paired variation traces and mapped nodes are inconsistent")
    groups: dict[str, list[tuple[TrajectoryTrace, int]]] = {}
    for trace, node in zip(traces, nodes):
        if int(node) < 0 or int(node) >= trace.layer_count:
            raise ValueError("mapped perturbation node is outside the trace")
        groups.setdefault(_pair_key(trace), []).append((trace, int(node)))
    differences = []
    bundle_count = 0
    for members in groups.values():
        if len(members) < 2:
            continue
        bundle_count += 1
        center_trace, center_node = members[0]
        center = center_trace.hidden_states[center_node]
        if transform is not None:
            center = transform(center, depth=center_trace.depth_coordinates[center_node]) if transform_accepts_depth else transform(center)
        for member, node in members[1:]:
            value = member.hidden_states[node]
            if transform is not None:
                value = transform(value, depth=member.depth_coordinates[node]) if transform_accepts_depth else transform(value)
            differences.append(value - center)
    if not differences:
        return np.zeros((traces[0].state_dim, 0)), bundle_count
    return _basis(np.asarray(differences), rank), bundle_count


def _mapped_depth(trace: TrajectoryTrace, student: TrajectoryTrace, correspondence: MonotoneCorrespondence | None) -> np.ndarray:
    if correspondence is None:
        normalized = (trace.depth_coordinates - trace.depth_coordinates[0]) / max(float(trace.depth_coordinates[-1] - trace.depth_coordinates[0]), 1e-12)
        return student.depth_coordinates[0] + normalized * float(student.depth_coordinates[-1] - student.depth_coordinates[0])
    pairs = sorted(correspondence.pairs, key=lambda pair: pair[1])
    teacher_index = np.asarray([pair[1] for pair in pairs], dtype=int)
    student_index = np.asarray([pair[0] for pair in pairs], dtype=int)
    return np.interp(trace.depth_coordinates, trace.depth_coordinates[teacher_index], student.depth_coordinates[student_index])


def _quadratic_features(delta: np.ndarray) -> np.ndarray:
    return np.einsum("ni,nj->nij", delta, delta).reshape(len(delta), -1)


def _fit_local_jet(states: np.ndarray, velocities: np.ndarray, weights: np.ndarray, *, rank: int, quadratic: bool, ridge: float, robust_loss: str, robust_iterations: int) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray, float, float, np.ndarray]:
    x = finite_array(states, ndim=2, name="local jet states")
    v = finite_array(velocities, ndim=2, name="local jet velocities")
    if x.shape != v.shape or len(weights) != len(x):
        raise ValueError("local jet state and velocity arrays are inconsistent")
    positive = np.asarray(weights) > 1e-9
    if np.count_nonzero(positive) < 2:
        raise ValueError("local jet needs at least two positive-weight samples")
    x = x[positive]
    v = v[positive]
    w = np.asarray(weights, dtype=np.float64)[positive]
    center = np.average(x, axis=0, weights=w)
    delta = x - center
    basis = _basis(delta, rank)
    linear_design = np.column_stack([delta, np.ones(len(delta))])
    linear_coef, residual, uncertainty = _robust_regression(linear_design, v, w, ridge=ridge, loss=robust_loss, iterations=robust_iterations, tuning=1.345)
    jacobian = linear_coef[:-1]
    hessian = np.zeros((rank, x.shape[1]), dtype=np.float64)
    # A non-quadratic local jet must remain O(d^2) at most.  In particular,
    # do not allocate a dense d^3 placeholder for the disabled quadratic
    # channel: GPT-2 sequence charts make that tensor infeasible even when
    # only the linear 1-jet is requested.
    quadratic_tensor: np.ndarray | None = None
    if quadratic:
        design = np.column_stack([delta, np.ones(len(delta)), _quadratic_features(delta)])
        coefficient, q_residual, q_uncertainty = _robust_regression(design, v, w, ridge=max(ridge, 1e-6), loss=robust_loss, iterations=robust_iterations, tuning=1.345)
        flat = coefficient[x.shape[1] + 1 :]
        quadratic_tensor = flat.T.reshape(x.shape[1], x.shape[1], x.shape[1])
        uncertainty = min(1.0, 0.5 * uncertainty + 0.5 * q_uncertainty)
        residual = max(residual, q_residual)
    if quadratic and basis.shape[1] and quadratic_tensor is not None:
        directions = np.zeros((rank, x.shape[1]), dtype=np.float64)
        directions[: basis.shape[1]] = basis.T
        symmetric = 0.5 * (quadratic_tensor + quadratic_tensor.transpose(0, 2, 1))
        hessian = 2.0 * np.einsum("oij,ri,rj->ro", symmetric, directions, directions)
    return center, jacobian, quadratic_tensor, hessian, residual, uncertainty, basis


@dataclass(frozen=True)
class LocalJetTransport:
    """Student-chart local jet transport with explicit capacity evidence."""

    coordinates: np.ndarray
    student_jacobians: np.ndarray
    teacher_jacobians: np.ndarray
    transported_jacobians: np.ndarray
    student_hessian_sketches: np.ndarray
    teacher_hessian_sketches: np.ndarray
    transported_hessian_sketches: np.ndarray
    student_projectors: np.ndarray
    teacher_projectors: np.ndarray
    principal_angles: np.ndarray
    residual_budget: np.ndarray
    uncertainty: np.ndarray
    confidence: np.ndarray
    confidence_low: np.ndarray
    confidence_high: np.ndarray
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        coordinates = finite_array(self.coordinates, ndim=1, name="jet coordinates")
        matrices = [finite_array(value, ndim=3, name="jet matrix") for value in (self.student_jacobians, self.teacher_jacobians, self.transported_jacobians, self.student_projectors, self.teacher_projectors)]
        hessians = [finite_array(value, ndim=3, name="directional Hessian sketch") for value in (self.student_hessian_sketches, self.teacher_hessian_sketches, self.transported_hessian_sketches)]
        angles = finite_array(self.principal_angles, ndim=2, name="jet principal angles")
        vectors = [finite_array(value, ndim=1, name="jet node diagnostic") for value in (self.residual_budget, self.uncertainty, self.confidence, self.confidence_low, self.confidence_high)]
        n = len(coordinates)
        if n == 0 or np.any(np.diff(coordinates) <= 0):
            raise ValueError("jet coordinates must be non-empty and increasing")
        if any(value.shape != (n, matrices[0].shape[1], matrices[0].shape[2]) for value in matrices):
            raise ValueError("jet matrix arrays have inconsistent shapes")
        if any(value.shape[0] != n or value.shape[2] != matrices[0].shape[1] for value in hessians) or angles.shape[0] != n or any(len(value) != n for value in vectors):
            raise ValueError("jet diagnostic arrays have inconsistent shapes")
        if any(np.any(value < 0.0) or np.any(value > 1.0) for value in (vectors[0], vectors[1], vectors[2], vectors[3], vectors[4])):
            raise ValueError("jet budgets, uncertainty, and confidence must be in [0, 1]")
        object.__setattr__(self, "coordinates", coordinates)
        for name, value in zip(("student_jacobians", "teacher_jacobians", "transported_jacobians", "student_projectors", "teacher_projectors"), matrices):
            object.__setattr__(self, name, value)
        for name, value in zip(("student_hessian_sketches", "teacher_hessian_sketches", "transported_hessian_sketches"), hessians):
            object.__setattr__(self, name, value)
        object.__setattr__(self, "principal_angles", angles)
        for name, value in zip(("residual_budget", "uncertainty", "confidence", "confidence_low", "confidence_high"), vectors):
            object.__setattr__(self, name, value)

    def _interpolate(self, values: np.ndarray, depth: float) -> np.ndarray:
        return np.column_stack([np.interp(float(depth), self.coordinates, values[:, index] if values.ndim == 2 else values[:, index, 0]) for index in range(values.shape[1])]) if values.ndim == 2 else np.asarray([
            np.column_stack([np.interp(float(depth), self.coordinates, values[:, output, input]) for input in range(values.shape[2])])
            for output in range(values.shape[1])
        ])

    def matrix_at(self, depth: float) -> np.ndarray:
        if depth <= self.coordinates[0]:
            return self.transported_jacobians[0].copy()
        if depth >= self.coordinates[-1]:
            return self.transported_jacobians[-1].copy()
        index = int(np.searchsorted(self.coordinates, depth) - 1)
        t = (float(depth) - self.coordinates[index]) / (self.coordinates[index + 1] - self.coordinates[index])
        return (1.0 - t) * self.transported_jacobians[index] + t * self.transported_jacobians[index + 1]

    def projector_at(self, depth: float) -> np.ndarray:
        if depth <= self.coordinates[0]:
            return self.student_projectors[0].copy()
        if depth >= self.coordinates[-1]:
            return self.student_projectors[-1].copy()
        index = int(np.searchsorted(self.coordinates, depth) - 1)
        t = (float(depth) - self.coordinates[index]) / (self.coordinates[index + 1] - self.coordinates[index])
        return (1.0 - t) * self.student_projectors[index] + t * self.student_projectors[index + 1]

    def confidence_at(self, depth: float) -> float:
        return float(np.interp(float(depth), self.coordinates, self.confidence, left=self.confidence[0], right=self.confidence[-1]))

    def project_velocity(self, velocity: np.ndarray, depth: float) -> np.ndarray:
        value = finite_array(velocity, ndim=1, name="jet velocity")
        projector = self.projector_at(depth)
        if value.size != projector.shape[0]:
            raise ValueError("jet velocity has the wrong student dimension")
        return finite_array(value @ projector.T, ndim=1, name="projected jet velocity")


def fit_rank_aware_local_jet_transport(
    student_traces: Sequence[TrajectoryTrace],
    teacher_traces: Sequence[TrajectoryTrace],
    alignment: AlignmentResult,
    correspondences: Sequence[MonotoneCorrespondence] | None = None,
    *,
    rank: int | None = None,
    bandwidth: float | None = None,
    ridge: float = 1e-4,
    quadratic: bool = True,
    robust_loss: str = "huber",
    robust_iterations: int = 4,
) -> LocalJetTransport:
    """Fit rank-aware local 1-jets and directional 2-jets from paired traces."""

    if len(student_traces) != len(teacher_traces) or not student_traces:
        raise ValueError("paired teacher/student traces are required for local jet transport")
    if alignment.source_dim != teacher_traces[0].state_dim or alignment.target_dim != student_traces[0].state_dim:
        raise ValueError("teacher-to-student alignment dimensions do not match local jet transport")
    if any(a.probe_id != b.probe_id for a, b in zip(student_traces, teacher_traces)):
        raise ValueError("local jet traces must retain paired probe IDs")
    student_dim = student_traces[0].state_dim
    jet_rank = min(int(rank or student_dim), student_dim)
    if jet_rank < 1 or ridge < 0 or not np.isfinite(ridge):
        raise ValueError("local jet rank and ridge are invalid")
    if correspondences is not None and len(correspondences) != len(student_traces):
        raise ValueError("one depth correspondence is required per paired trace")
    coordinates = student_traces[0].depth_coordinates[:-1].copy()
    if any(trace.depth_coordinates[:-1].shape != coordinates.shape or not np.allclose(trace.depth_coordinates[:-1], coordinates) for trace in student_traces):
        raise ValueError("local jet transport requires a common student transition grid")
    default_band = float(bandwidth or max(0.10, 0.75 * np.median(np.diff(student_traces[0].depth_coordinates))))
    if default_band <= 0 or not np.isfinite(default_band):
        raise ValueError("local jet bandwidth must be positive and finite")

    student_jacobians = []
    teacher_jacobians = []
    transported_jacobians = []
    student_hessians = []
    teacher_hessians = []
    transported_hessians = []
    student_projectors = []
    teacher_projectors = []
    angle_rows = []
    residual_budgets = []
    uncertainties = []
    confidences = []
    teacher_variation_nodes: list[list[int]] = []
    paired_bundle_count = 0
    student_states = []
    student_velocities = []
    student_depths = []
    teacher_states = []
    teacher_velocities = []
    teacher_depths = []
    for index, (student, teacher) in enumerate(zip(student_traces, teacher_traces)):
        for transition in student.transitions:
            student_states.append(transition.source_state)
            student_velocities.append(transition.vector_field)
            student_depths.append(transition.source_depth)
        mapped_depth = _mapped_depth(teacher, student, None if correspondences is None else correspondences[index])
        for transition_index, transition in enumerate(teacher.transitions):
            ds = float(mapped_depth[transition_index + 1] - mapped_depth[transition_index])
            if ds <= 1e-10:
                continue
            teacher_states.append(alignment.apply(transition.source_state, depth=transition.source_depth))
            teacher_velocities.append(alignment.linear_apply(transition.target_state - transition.source_state, depth=transition.source_depth) / ds)
            teacher_depths.append(mapped_depth[transition_index])
    student_states_array = np.asarray(student_states)
    student_velocities_array = np.asarray(student_velocities)
    student_depths_array = np.asarray(student_depths)
    teacher_states_array = np.asarray(teacher_states)
    teacher_velocities_array = np.asarray(teacher_velocities)
    teacher_depths_array = np.asarray(teacher_depths)
    for node, coordinate in enumerate(coordinates):
        student_weights = np.exp(-0.5 * ((student_depths_array - coordinate) / default_band) ** 2)
        teacher_weights = np.exp(-0.5 * ((teacher_depths_array - coordinate) / default_band) ** 2)
        _, student_jacobian, student_q, student_hessian, student_residual, student_uncertainty, student_basis = _fit_local_jet(student_states_array, student_velocities_array, student_weights, rank=jet_rank, quadratic=quadratic, ridge=ridge, robust_loss=robust_loss, robust_iterations=robust_iterations)
        _, teacher_jacobian, teacher_q, teacher_hessian, teacher_residual, teacher_uncertainty, teacher_basis = _fit_local_jet(teacher_states_array, teacher_velocities_array, teacher_weights, rank=jet_rank, quadratic=quadratic, ridge=ridge, robust_loss=robust_loss, robust_iterations=robust_iterations)
        paired_student_basis, student_pairs = _paired_variation_basis(student_traces, node, jet_rank)
        mapped_teacher_nodes = []
        for student_trace, teacher_trace, correspondence in zip(
            student_traces,
            teacher_traces,
            () if correspondences is None else correspondences,
        ):
            mapped_depth = _mapped_depth(teacher_trace, student_trace, correspondence)
            mapped_teacher_nodes.append(int(np.argmin(np.abs(mapped_depth - float(coordinate)))))
        if correspondences is None:
            mapped_teacher_nodes = [
                int(np.argmin(np.abs(_mapped_depth(teacher_trace, student_trace, None) - float(coordinate))))
                for student_trace, teacher_trace in zip(student_traces, teacher_traces)
            ]
        paired_teacher_basis, teacher_pairs = _paired_variation_basis_at_nodes(
            teacher_traces,
            mapped_teacher_nodes,
            jet_rank,
            alignment.linear_apply,
            True,
        )
        teacher_variation_nodes.append(mapped_teacher_nodes)
        paired_bundle_count = max(paired_bundle_count, student_pairs, teacher_pairs)
        if paired_student_basis.shape[1]:
            student_basis = paired_student_basis
        if paired_teacher_basis.shape[1]:
            teacher_basis = paired_teacher_basis
        student_projector = student_basis @ student_basis.T if student_basis.shape[1] else np.zeros((student_dim, student_dim))
        teacher_projector = teacher_basis @ teacher_basis.T if teacher_basis.shape[1] else np.zeros((student_dim, student_dim))
        angle_row = np.full(jet_rank, np.pi / 2.0)
        if student_basis.shape[1] and teacher_basis.shape[1]:
            _, singular, _ = np.linalg.svd(teacher_basis.T @ student_basis, full_matrices=False)
            angle_row[: len(singular)] = np.arccos(np.clip(singular, -1.0, 1.0))
        teacher_tangent_jacobian = teacher_projector @ teacher_jacobian @ teacher_projector
        transported_jacobian = student_projector @ teacher_tangent_jacobian @ student_projector
        if quadratic:
            if teacher_q is None or student_q is None:
                raise RuntimeError("quadratic local jet requested without a quadratic tensor")
            teacher_tangent_q = np.einsum("ai,oij,jb->oab", teacher_projector, teacher_q, teacher_projector)
            student_directions = np.zeros((jet_rank, student_dim))
            if student_basis.shape[1]:
                student_directions[: student_basis.shape[1]] = student_basis.T
            transported_hessian = 2.0 * np.einsum("oij,ri,rj->ro", teacher_tangent_q, student_directions, student_directions)
            transported_hessian = transported_hessian @ student_projector.T
        else:
            # Preserve the directional-Hessian shape contract while making it
            # explicit that no second-order object was fitted or transported.
            transported_hessian = np.zeros_like(teacher_hessian)
        jacobian_residual = float(stable_l2(teacher_jacobian - transported_jacobian, name="jet Jacobian capacity residual") / max(1.0, float(stable_l2(teacher_jacobian, name="teacher local Jacobian"))))
        hessian_residual = float(stable_l2(teacher_hessian - transported_hessian, name="jet Hessian capacity residual") / max(1.0, float(stable_l2(teacher_hessian, name="teacher directional Hessian"))))
        overlap_score = float(np.mean(np.cos(angle_row)))
        residual_budget = float(np.clip(1.0 - 0.5 * (jacobian_residual + hessian_residual), 0.0, 1.0))
        uncertainty = float(np.clip(0.35 * student_uncertainty + 0.35 * teacher_uncertainty + 0.30 / np.sqrt(max(len(student_states_array), 1)), 0.0, 1.0))
        confidence = float(np.clip(overlap_score * residual_budget * (1.0 - uncertainty), 0.0, 1.0))
        student_jacobians.append(student_jacobian)
        teacher_jacobians.append(teacher_jacobian)
        transported_jacobians.append(transported_jacobian)
        student_hessians.append(student_hessian)
        teacher_hessians.append(teacher_hessian)
        transported_hessians.append(transported_hessian)
        student_projectors.append(student_projector)
        teacher_projectors.append(teacher_projector)
        angle_rows.append(angle_row)
        residual_budgets.append(residual_budget)
        uncertainties.append(uncertainty)
        confidences.append(confidence)
    uncertainty_array = np.asarray(uncertainties)
    confidence_array = np.asarray(confidences)
    return LocalJetTransport(
        coordinates,
        np.asarray(student_jacobians),
        np.asarray(teacher_jacobians),
        np.asarray(transported_jacobians),
        np.asarray(student_hessians),
        np.asarray(teacher_hessians),
        np.asarray(transported_hessians),
        np.asarray(student_projectors),
        np.asarray(teacher_projectors),
        np.asarray(angle_rows),
        np.asarray(residual_budgets),
        uncertainty_array,
        confidence_array,
        np.clip(confidence_array - 1.96 * uncertainty_array, 0.0, 1.0),
        np.clip(confidence_array + 1.96 * uncertainty_array, 0.0, 1.0),
        {
            "rank": jet_rank,
            "robust_loss": robust_loss,
            "robust_iterations": robust_iterations,
            "paired_perturbation_bundles": paired_bundle_count,
            "hessian_kind": "directional_sketch",
            "teacher_to_student": True,
            "teacher_variation_node_indices": teacher_variation_nodes,
            "capacity_budget_definition": "1 - normalized projected Jacobian/Hessian residual",
        },
    )
