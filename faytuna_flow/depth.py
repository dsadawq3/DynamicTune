"""Continuous depth reconstruction, monotone correspondence, gaps, and bifurcations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .types import BifurcationPoint, GapInterval, TrajectoryTrace, finite_array, stable_l2


@dataclass(frozen=True)
class DepthPath:
    coordinates: np.ndarray
    arc_length: float
    uncertainty: np.ndarray

    def __post_init__(self) -> None:
        coordinates = finite_array(self.coordinates, ndim=1, name="depth path coordinates")
        uncertainty = finite_array(self.uncertainty, ndim=1, name="depth path uncertainty")
        if len(coordinates) < 2 or len(coordinates) != len(uncertainty) or np.any(np.diff(coordinates) <= 0) or self.arc_length <= 0 or not np.isfinite(self.arc_length):
            raise ValueError("depth path arrays are inconsistent")
        object.__setattr__(self, "coordinates", coordinates)
        object.__setattr__(self, "uncertainty", np.maximum(uncertainty, 0.0))


@dataclass(frozen=True)
class MonotoneCorrespondence:
    pairs: tuple[tuple[int, int], ...]
    gaps: tuple[GapInterval, ...]
    cost: float
    confidence: float
    metadata: dict[str, object] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        pairs = tuple((int(a), int(b)) for a, b in self.pairs)
        if len(pairs) < 2 or any(a0 >= a1 or b0 >= b1 for (a0, b0), (a1, b1) in zip(pairs, pairs[1:])):
            raise ValueError("monotone correspondence pairs must be strictly ordered")
        if not np.isfinite(self.cost) or self.cost < 0 or not np.isfinite(self.confidence) or not 0.0 <= self.confidence <= 1.0:
            raise ValueError("monotone correspondence cost/confidence is invalid")
        object.__setattr__(self, "pairs", pairs)
        object.__setattr__(self, "metadata", dict(self.metadata or {}))


def reconstruct_depth_path(states: np.ndarray, *, curvature_weight: float = 0.15, uncertainty_floor: float = 1e-3) -> DepthPath:
    values = finite_array(states, ndim=2, name="path states")
    if len(values) < 2:
        raise ValueError("at least two states are required")
    if curvature_weight < 0 or not np.isfinite(curvature_weight) or uncertainty_floor < 0 or not np.isfinite(uncertainty_floor):
        raise ValueError("depth curvature and uncertainty parameters must be finite and non-negative")
    scale = max(1.0, float(np.max(np.abs(values))))
    with np.errstate(over="raise", invalid="raise"):
        deltas = (values[1:] / scale - values[:-1] / scale) * scale
    lengths = np.asarray(stable_l2(deltas, axis=1, name="depth path deltas"), dtype=np.float64)
    turns = np.zeros(len(deltas))
    for i in range(1, len(deltas)):
        a, b = deltas[i - 1], deltas[i]
        norm_a = max(float(stable_l2(a, name="depth turn vector")), 1e-12)
        norm_b = max(float(stable_l2(b, name="depth turn vector")), 1e-12)
        turns[i] = np.arccos(np.clip(float(np.dot(a / norm_a, b / norm_b)), -1.0, 1.0))
    effective = np.maximum(lengths * (1.0 + curvature_weight * turns / np.pi), 1e-8)
    coordinates = np.concatenate([[0.0], np.cumsum(effective)])
    total = coordinates[-1]
    if not np.isfinite(total) or total <= 0:
        raise FloatingPointError("depth path arc length is non-finite")
    coordinates /= total
    uncertainty = uncertainty_floor + curvature_weight * np.concatenate([[0.0], turns])
    return DepthPath(coordinates, float(total), uncertainty)


def interpolate_path(states: np.ndarray, coordinates: np.ndarray, query: float) -> np.ndarray:
    values = finite_array(states, ndim=2, name="interpolation states")
    coords = finite_array(coordinates, ndim=1, name="interpolation coordinates")
    if len(coords) != len(values) or len(coords) < 2 or np.any(np.diff(coords) <= 0):
        raise ValueError("states and coordinates must be aligned and strictly increasing")
    q = float(query)
    if not np.isfinite(q):
        raise ValueError("query depth must be finite")
    if q <= coords[0]:
        return values[0].copy()
    if q >= coords[-1]:
        return values[-1].copy()
    i = int(np.searchsorted(coords, q) - 1)
    t = (q - coords[i]) / (coords[i + 1] - coords[i])
    return (1.0 - t) * values[i] + t * values[i + 1]


def _pair_cost(student: np.ndarray, teacher: np.ndarray, i: int, j: int, scale: float) -> float:
    # State mismatch is compared with trajectory extent as well as one-step
    # displacement. Otherwise modest teacher/student drift is incorrectly
    # interpreted as a layer deletion and the DP consumes every node as a gap.
    return float(min(float(stable_l2(student[i] - teacher[j], name="depth pair residual")) / max(scale, 1e-12), 25.0))


def _normalize_depth_chart(values: np.ndarray) -> tuple[np.ndarray, float]:
    """Remove per-chart offset/scale before comparing path shape.

    A teacher-to-student map fitted on a small probe set can be poorly
    determined outside the initial-state subspace.  Using its raw magnitudes
    as the depth cost then makes every intermediate node look like a gap.  A
    depth correspondence needs a shape descriptor as well as the supplied
    cross-model chart map, so its state cost is normalized within each path.
    The original states are still used for flow transport after correspondence
    is selected; this normalization does not claim chart equivalence.
    """

    centered = values - values[0]
    extent = np.asarray(stable_l2(centered, axis=1, name="depth chart extent"), dtype=np.float64)
    steps = np.asarray(stable_l2(np.diff(values, axis=0), axis=1, name="depth chart steps"), dtype=np.float64)
    nonzero_extent = extent[1:][extent[1:] > 1e-12]
    nonzero_steps = steps[steps > 1e-12]
    scale = max(
        float(np.median(nonzero_extent)) if nonzero_extent.size else 0.0,
        2.0 * float(np.median(nonzero_steps)) if nonzero_steps.size else 0.0,
        1e-8,
    )
    return centered / scale, scale


def _path_velocity(states: np.ndarray, coordinates: np.ndarray) -> np.ndarray:
    differences = states[1:] - states[:-1]
    return differences / np.diff(coordinates)[:, None]


def _path_curvature(velocities: np.ndarray) -> np.ndarray:
    result = np.zeros(len(velocities), dtype=np.float64)
    for index in range(1, len(velocities)):
        previous = float(stable_l2(velocities[index - 1], name="depth velocity"))
        current = float(stable_l2(velocities[index], name="depth velocity"))
        result[index] = float(stable_l2(velocities[index] - velocities[index - 1], name="depth curvature increment") / max(1.0, previous, current))
    return result


def monotone_correspondence(
    student: np.ndarray,
    teacher: np.ndarray,
    *,
    student_coordinates: np.ndarray | None = None,
    teacher_coordinates: np.ndarray | None = None,
    gap_penalty: float = 0.75,
    slope_penalty: float = 0.10,
    depth_penalty: float = 0.20,
    transition_weight: float = 0.35,
    curvature_weight: float = 0.20,
    uncertainty_weight: float = 0.10,
    student_vector_fields: np.ndarray | None = None,
    teacher_vector_fields: np.ndarray | None = None,
    student_curvature: np.ndarray | None = None,
    teacher_curvature: np.ndarray | None = None,
    student_uncertainty: np.ndarray | None = None,
    teacher_uncertainty: np.ndarray | None = None,
) -> MonotoneCorrespondence:
    """Dynamic-programming alignment that retains unmatched observations as gaps."""

    x = finite_array(student, ndim=2, name="student path")
    y = finite_array(teacher, ndim=2, name="teacher path")
    if len(x) < 2 or len(y) < 2:
        raise ValueError("paths need at least two observations")
    n, m = len(x), len(y)
    sx = np.linspace(0.0, 1.0, len(x)) if student_coordinates is None else finite_array(student_coordinates, ndim=1, name="student depth coordinates")
    ty = np.linspace(0.0, 1.0, len(y)) if teacher_coordinates is None else finite_array(teacher_coordinates, ndim=1, name="teacher depth coordinates")
    if len(sx) != len(x) or len(ty) != len(y) or np.any(np.diff(sx) <= 0) or np.any(np.diff(ty) <= 0):
        raise ValueError("depth coordinate arrays must match paths and be strictly increasing")
    if min(gap_penalty, slope_penalty, depth_penalty, transition_weight, curvature_weight, uncertainty_weight) < 0 or gap_penalty <= 0 or not np.isfinite(gap_penalty + slope_penalty + depth_penalty + transition_weight + curvature_weight + uncertainty_weight):
        raise ValueError("gap, slope, and depth penalties must be finite and non-negative, with positive gap penalty")
    normalized_x, student_state_scale = _normalize_depth_chart(x)
    normalized_y, teacher_state_scale = _normalize_depth_chart(y)
    x_extent = float(np.median(np.linalg.norm(normalized_x, axis=1)))
    y_extent = float(np.median(np.linalg.norm(normalized_y, axis=1)))
    x_step = float(np.median(np.linalg.norm(np.diff(normalized_x, axis=0), axis=1)))
    y_step = float(np.median(np.linalg.norm(np.diff(normalized_y, axis=0), axis=1)))
    scale = max(x_extent, y_extent, 2.0 * x_step, 2.0 * y_step, 1e-8)
    student_velocity = _path_velocity(normalized_x, sx)
    teacher_velocity = _path_velocity(normalized_y, ty)
    if student_vector_fields is not None:
        student_velocity = finite_array(student_vector_fields, ndim=2, name="student path vector fields")
    if teacher_vector_fields is not None:
        teacher_velocity = finite_array(teacher_vector_fields, ndim=2, name="teacher path vector fields")
    if len(student_velocity) != n - 1 or len(teacher_velocity) != m - 1 or student_velocity.shape[1] != teacher_velocity.shape[1]:
        raise ValueError("path vector fields must have one same-chart row per transition")
    velocity_scale = max(1e-8, float(np.median(np.abs(np.vstack([student_velocity, teacher_velocity])))))
    student_curve = _path_curvature(student_velocity) if student_curvature is None else finite_array(student_curvature, ndim=1, name="student path curvature")
    teacher_curve = _path_curvature(teacher_velocity) if teacher_curvature is None else finite_array(teacher_curvature, ndim=1, name="teacher path curvature")
    if len(student_curve) != n - 1 or len(teacher_curve) != m - 1:
        raise ValueError("path curvature arrays must have one row per transition")
    student_node_uncertainty = np.zeros(n, dtype=np.float64) if student_uncertainty is None else np.maximum(finite_array(student_uncertainty, ndim=1, name="student path uncertainty"), 0.0)
    teacher_node_uncertainty = np.zeros(m, dtype=np.float64) if teacher_uncertainty is None else np.maximum(finite_array(teacher_uncertainty, ndim=1, name="teacher path uncertainty"), 0.0)
    if len(student_node_uncertainty) != n or len(teacher_node_uncertainty) != m:
        raise ValueError("path uncertainty arrays must have one row per node")
    velocity_scale = max(velocity_scale, 1e-8)
    pair_cost_matrix = np.minimum(np.linalg.norm(normalized_x[:, None, :] - normalized_y[None, :, :], axis=-1) / max(scale, 1e-12), 25.0)
    student_speeds = np.linalg.norm(student_velocity, axis=1)
    teacher_speeds = np.linalg.norm(teacher_velocity, axis=1)
    dp = np.full((n, m), np.inf)
    parent = np.full((n, m, 2), -1, dtype=np.int64)
    dp[0, 0] = pair_cost_matrix[0, 0]
    for i in range(n):
        for j in range(m):
            if not np.isfinite(dp[i, j]):
                continue
            if i + 1 < n and j + 1 < m:
                slope = abs(float(sx[i + 1] - sx[0]) / max(float(sx[-1] - sx[0]), 1e-12) - float(ty[j + 1] - ty[0]) / max(float(ty[-1] - ty[0]), 1e-12))
                student_position = float((sx[i + 1] - sx[0]) / max(sx[-1] - sx[0], 1e-12))
                teacher_position = float((ty[j + 1] - ty[0]) / max(ty[-1] - ty[0], 1e-12))
                student_speed = float(student_speeds[i])
                teacher_speed = float(teacher_speeds[j])
                if student_speed <= 1e-12 or teacher_speed <= 1e-12:
                    direction_cost = 1.0 if abs(student_speed - teacher_speed) > 1e-12 else 0.0
                else:
                    cosine = float(np.dot(student_velocity[i], teacher_velocity[j]) / (student_speed * teacher_speed))
                    direction_cost = 0.5 * (1.0 - float(np.clip(cosine, -1.0, 1.0)))
                speed_cost = abs(float(np.log(max(student_speed, 1e-12) / max(teacher_speed, 1e-12))))
                velocity_cost = min(0.5 * direction_cost + 0.5 * min(speed_cost, 25.0), 25.0)
                curvature_cost = min(abs(float(student_curve[i]) - float(teacher_curve[j])), 25.0)
                uncertainty_cost = min(float(student_node_uncertainty[i + 1] + teacher_node_uncertainty[j + 1]), 25.0)
                candidate = dp[i, j] + pair_cost_matrix[i + 1, j + 1] + slope_penalty * slope + depth_penalty * abs(student_position - teacher_position) + transition_weight * min(velocity_cost, 25.0) + curvature_weight * curvature_cost + uncertainty_weight * uncertainty_cost
                if candidate < dp[i + 1, j + 1]:
                    dp[i + 1, j + 1] = candidate
                    parent[i + 1, j + 1] = (i, j)
            if i + 1 < n:
                candidate = dp[i, j] + gap_penalty
                if candidate < dp[i + 1, j]:
                    dp[i + 1, j] = candidate
                    parent[i + 1, j] = (i, j)
            if j + 1 < m:
                candidate = dp[i, j] + gap_penalty
                if candidate < dp[i, j + 1]:
                    dp[i, j + 1] = candidate
                    parent[i, j + 1] = (i, j)
    i, j = n - 1, m - 1
    if not np.isfinite(dp[i, j]):
        raise RuntimeError("monotone correspondence has no finite path")
    moves: list[tuple[str, int, int]] = []
    while not (i == 0 and j == 0):
        pi, pj = parent[i, j]
        if pi < 0:
            raise RuntimeError("invalid dynamic-programming backtrace")
        if i == pi + 1 and j == pj + 1:
            moves.append(("pair", i, j))
        elif i == pi + 1:
            moves.append(("student_gap", i, j))
        else:
            moves.append(("teacher_gap", i, j))
        i, j = int(pi), int(pj)
    moves.reverse()
    pairs = [(0, 0)]
    gaps: list[GapInterval] = []
    current_i, current_j = 0, 0
    for kind, next_i, next_j in moves:
        if kind == "pair":
            pairs.append((next_i, next_j))
        elif kind == "student_gap":
            gaps.append(GapInterval(next_i, next_j, float(sx[next_i]), float(ty[next_j]), gap_penalty, "student observation has no monotone teacher match"))
        else:
            gaps.append(GapInterval(next_i, next_j, float(sx[next_i]), float(ty[next_j]), gap_penalty, "teacher observation has no monotone student match"))
        current_i, current_j = next_i, next_j
    pairs = sorted(set(pairs))
    # A continuous map needs observed endpoints even when the dynamic program
    # chose an interior gap-heavy path.  Anchor the final observed endpoint
    # and expose the mismatch as an explicit gap penalty; no new layer is
    # created and confidence remains low when this anchor is costly.
    endpoint_anchor_added = False
    if len(pairs) < 2:
        pairs.append((n - 1, m - 1))
        gaps.append(GapInterval(pairs[0][0], m - 1, float(sx[pairs[0][0]]), float(ty[m - 1]), gap_penalty, "endpoint anchor retained after gap-heavy path"))
        endpoint_anchor_added = True
    elif pairs[-1][0] < n - 1 and pairs[-1][1] < m - 1:
        gaps.append(GapInterval(pairs[-1][0], m - 1, float(sx[pairs[-1][0]]), float(ty[m - 1]), gap_penalty, "endpoint anchor retained after gap-heavy path"))
        pairs.append((n - 1, m - 1))
        endpoint_anchor_added = True
    normalized_cost = float(dp[-1, -1] / max(1, n + m))
    confidence = float(np.exp(np.clip(-normalized_cost, -700.0, 0.0)) * (len(pairs) / max(n, m)))
    feature_metadata = {
        "objective": "state+depth+slope+transition_velocity+curvature+uncertainty+gaps",
        "transition_weight": transition_weight,
        "curvature_weight": curvature_weight,
        "uncertainty_weight": uncertainty_weight,
        "gap_penalty": gap_penalty,
        "gap_count": len(gaps),
        "matched_fraction": len(pairs) / max(n, m),
        "student_coverage": len(pairs) / n,
        "teacher_coverage": len(pairs) / m,
        "endpoint_anchor_added": endpoint_anchor_added,
        "state_normalization": "per_path_centered_robust_scale",
        "student_state_scale": student_state_scale,
        "teacher_state_scale": teacher_state_scale,
        "velocity_cost": "direction_cosine_plus_log_speed",
    }
    return MonotoneCorrespondence(tuple(pairs), tuple(gaps), normalized_cost, confidence, feature_metadata)


def detect_bifurcations(traces: Sequence[TrajectoryTrace], *, micro_threshold: float = 2.5, meso_threshold: float = 2.5, macro_threshold: float = 2.5) -> tuple[BifurcationPoint, ...]:
    """Detect scale-separated instability candidates; these are hypotheses, not proofs."""

    if not traces:
        return ()
    min_nodes = min(t.layer_count for t in traces)
    points: list[BifurcationPoint] = []
    for node in range(1, min_nodes - 1):
        curvatures = np.asarray([t.transitions[node - 1].curvature for t in traces])
        spectral_jumps = np.asarray([
            np.linalg.norm(t.transitions[node].singular_values - t.transitions[node - 1].singular_values) / max(np.linalg.norm(t.transitions[node - 1].singular_values), 1e-12)
            if t.transitions[node].singular_values is not None and t.transitions[node - 1].singular_values is not None else 0.0
            for t in traces
        ])
        states = np.asarray([t.hidden_states[node] for t in traces])
        if len(states) > 1:
            centered = states - states.mean(axis=0)
            spread = float(np.sqrt(np.mean(np.sum(centered * centered, axis=1))))
            before = np.asarray([t.hidden_states[node - 1] for t in traces])
            before_spread = float(np.sqrt(np.mean(np.sum((before - before.mean(axis=0)) ** 2, axis=1))))
            separation = spread / max(before_spread, 1e-8)
        else:
            separation = 1.0
        micro = _robust_z(curvatures)
        meso = _robust_z(spectral_jumps)
        macro = max(0.0, separation - 1.0)
        candidates = [("micro", micro, micro_threshold), ("meso", meso, meso_threshold), ("macro", macro, macro_threshold)]
        level, score, threshold = max(candidates, key=lambda item: item[1])
        if score >= threshold:
            confidence = float(1.0 - np.exp(-(score - threshold + 0.1)))
            points.append(BifurcationPoint(float(np.mean([t.depth_coordinates[node] for t in traces])), level, float(score), confidence, {"micro_curvature_z": micro, "meso_spectral_z": meso, "macro_separation": macro}))
    return tuple(points)


def _robust_z(values: np.ndarray) -> float:
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    return max(0.0, (float(np.max(values)) - median) / max(1.4826 * mad, 1e-6))
