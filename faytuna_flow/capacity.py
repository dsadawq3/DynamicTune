"""Student bottleneck diagnostics for paired teacher/student flow targets."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from .types import AlignmentResult, CapacityDiagnostics, finite_array, stable_l2


def _rank(values: np.ndarray, tolerance: float = 1e-8) -> int:
    centered = values - values.mean(axis=0, keepdims=True)
    scale = max(1.0, float(np.max(np.abs(centered))))
    singular = np.linalg.svd(centered / scale, compute_uv=False)
    if singular.size == 0 or singular[0] <= 0:
        return 0
    return int(np.sum(singular > tolerance * max(1.0, singular[0])))


def _basis(values: np.ndarray, tolerance: float = 1e-8) -> np.ndarray:
    centered = values - values.mean(axis=0, keepdims=True)
    if centered.size == 0:
        return np.zeros((values.shape[1], 0))
    scale = max(1.0, float(np.max(np.abs(centered))))
    _, singular, right_transpose = np.linalg.svd(centered / scale, full_matrices=False)
    if singular.size == 0:
        return np.zeros((values.shape[1], 0))
    keep = singular > tolerance * max(1.0, singular[0])
    return right_transpose[keep].T


def _coverage(values: np.ndarray, basis: np.ndarray) -> float:
    centered = values - values.mean(axis=0, keepdims=True)
    energy = float(stable_l2(centered, name="capacity centered states"))
    if energy <= 1e-9:
        return 1.0
    if basis.shape[1] == 0:
        return 0.0
    projected = centered @ basis @ basis.T
    ratio = float(stable_l2(projected, name="capacity projected states") / energy)
    return float(np.clip(ratio * ratio, 0.0, 1.0))


def diagnose_capacity(student_states: np.ndarray, teacher_states_original: np.ndarray, teacher_states_in_student: np.ndarray, teacher_velocities_in_student: np.ndarray, alignment: AlignmentResult, *, bottleneck_threshold: float = 0.85) -> CapacityDiagnostics:
    """Measure representational headroom before accepting flow corrections.

    The original teacher rank is retained even when the alignment compresses
    it into student coordinates. This prevents a low-dimensional student map
    from hiding that the target was intrinsically richer.
    """

    student = finite_array(student_states, ndim=2, name="capacity student states")
    teacher_original = finite_array(teacher_states_original, ndim=2, name="capacity teacher states")
    teacher_student = finite_array(teacher_states_in_student, ndim=2, name="capacity transported teacher states")
    teacher_velocity = finite_array(teacher_velocities_in_student, ndim=2, name="capacity teacher velocities")
    if student.shape[1] != teacher_student.shape[1] or teacher_velocity.shape[1] != student.shape[1]:
        raise ValueError("capacity arrays are not in the student chart")
    student_rank = _rank(student)
    teacher_rank = _rank(teacher_original)
    tangent = _basis(student)
    velocity_basis = _basis(teacher_velocity)
    state_rank_ratio = min(student_rank, teacher_rank) / max(teacher_rank, 1)
    state_coverage = min(state_rank_ratio, _coverage(teacher_student, tangent))
    velocity_coverage = _coverage(teacher_velocity, tangent)
    centered_teacher = teacher_student - teacher_student.mean(axis=0, keepdims=True)
    if tangent.shape[1] == 0:
        residual_transport = 1.0 if float(stable_l2(centered_teacher, name="capacity teacher residual")) > 1e-12 else 0.0
    else:
        residual_transport = float(stable_l2(centered_teacher - centered_teacher @ tangent @ tangent.T, name="capacity teacher residual") / max(float(stable_l2(centered_teacher, name="capacity teacher state")), 1e-12))
    if tangent.shape[1] == 0:
        irreducible = 1.0 if float(stable_l2(teacher_velocity, name="capacity velocity")) > 1e-12 else 0.0
    else:
        residual_velocity = teacher_velocity - teacher_velocity @ tangent @ tangent.T
        irreducible = float(stable_l2(residual_velocity, name="capacity residual velocity") / max(float(stable_l2(teacher_velocity, name="capacity teacher velocity")), 1e-12))
    condition = min(float(alignment.condition_number), 1e12) if np.isfinite(alignment.condition_number) else 1e12
    bottleneck = bool(teacher_rank > max(student_rank, 1) or state_coverage < bottleneck_threshold or velocity_coverage < bottleneck_threshold or irreducible > 1.0 - bottleneck_threshold)
    penalty = float(np.clip(min(state_coverage, velocity_coverage) * np.exp(-irreducible) * (1.0 / (1.0 + np.log10(max(condition, 1.0)))), 0.0, 1.0))
    notes = []
    if teacher_rank > student_rank:
        notes.append("teacher state rank exceeds student state rank; target flow is capacity-compressed")
    if velocity_coverage < bottleneck_threshold:
        notes.append("teacher velocity has components outside the measured student tangent subspace")
    if condition >= 1e12:
        notes.append("alignment condition was non-finite and capped for diagnostics")
    return CapacityDiagnostics(student_rank, teacher_rank, int(tangent.shape[1]), int(velocity_basis.shape[1]), state_coverage, velocity_coverage, residual_transport, irreducible, condition, bottleneck, penalty, tuple(notes))


def project_correction_to_student_tangent(correction: np.ndarray, bias: np.ndarray, student_states: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Keep a correction's input/output action on the measured student manifold."""

    delta = finite_array(correction, ndim=2, name="flow correction")
    offset = finite_array(bias, ndim=1, name="flow correction bias")
    states = finite_array(student_states, ndim=2, name="student manifold states")
    basis = _basis(states)
    if basis.shape[1] == 0:
        return np.zeros_like(delta), np.zeros_like(offset)
    projector = basis @ basis.T
    return projector @ delta @ projector, offset @ projector
