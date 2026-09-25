"""Causal validation that separates geometric fit from functional effect."""

from __future__ import annotations

from dataclasses import replace as dataclass_replace
from typing import Any, Callable, Sequence

import numpy as np

from .types import AlignmentResult, FlowOperator, Probe, TrajectoryTrace, TransitionObservation, ValidationReport, finite_array, stable_l2


def intervene_trace(trace: TrajectoryTrace, *, node: int, delta: np.ndarray | None = None, replace: np.ndarray | None = None) -> TrajectoryTrace:
    if node <= 0 or node >= trace.layer_count - 1:
        raise ValueError("intervention node must be an internal trajectory node")
    states = trace.hidden_states.copy()
    if replace is not None and delta is not None:
        raise ValueError("choose replace or delta")
    if replace is not None:
        candidate = finite_array(replace, ndim=1, name="replacement state")
        if candidate.size != trace.state_dim:
            raise ValueError("replacement state has the wrong dimension")
        states[node] = candidate
    else:
        candidate_delta = np.zeros(trace.state_dim) if delta is None else finite_array(delta, ndim=1, name="intervention delta")
        if candidate_delta.size != trace.state_dim:
            raise ValueError("intervention delta has the wrong dimension")
        states[node] += candidate_delta
    transitions = []
    for i, old in enumerate(trace.transitions):
        source, target = states[i], states[i + 1]
        d = target - source
        transitions.append(dataclass_replace(old, source_state=source, target_state=target, delta=d, vector_field=d / (old.target_depth - old.source_depth), jacobian=None, hessian_sketch=None, singular_values=None, uncertainty={**old.uncertainty, "intervention": 1.0}))
    return dataclass_replace(trace, hidden_states=states, transitions=tuple(transitions), metadata={**trace.metadata, "intervened_node": node})


def rollout_intervention(connector: Any, probe: Probe, baseline: TrajectoryTrace, *, node: int, delta: np.ndarray | None = None, replace: np.ndarray | None = None) -> TrajectoryTrace:
    """Inject a state and rerun every downstream observed transition.

    This is the causal connector path. ``intervene_trace`` is intentionally a
    cheaper record-only operation for diagnostics; this function lets the
    model's subsequent dynamics respond to the intervention.
    """

    if node <= 0 or node >= len(connector.layer_ids):
        raise ValueError("intervention node must be an internal connector node")
    if replace is not None and delta is not None:
        raise ValueError("choose replace or delta")
    states = [finite_array(connector.initial_state(probe), ndim=1, name="initial state")]
    for layer_index in range(node):
        states.append(finite_array(connector.transition(states[-1], layer_index, probe), ndim=1, name="pre-intervention state"))
    if replace is not None:
        injected = finite_array(replace, ndim=1, name="replacement state")
    else:
        increment = np.zeros(baseline.state_dim) if delta is None else finite_array(delta, ndim=1, name="intervention delta")
        injected = states[node] + increment
    if injected.size != baseline.state_dim:
        raise ValueError("intervention state has the wrong dimension")
    states[node] = injected
    for layer_index in range(node, len(connector.layer_ids)):
        states.append(finite_array(connector.transition(states[-1], layer_index, probe), ndim=1, name="post-intervention state"))
    if len(states) != baseline.layer_count:
        raise ValueError("connector layer count and baseline trace disagree")
    transitions = []
    for i, old in enumerate(baseline.transitions):
        source, target = states[i], states[i + 1]
        d = target - source
        transitions.append(dataclass_replace(old, source_state=source, target_state=target, delta=d, vector_field=d / (old.target_depth - old.source_depth), jacobian=None, hessian_sketch=None, singular_values=None, uncertainty={**old.uncertainty, "intervention": 1.0}))
    return dataclass_replace(baseline, hidden_states=np.asarray(states), residual_states=None, transitions=tuple(transitions), metadata={**baseline.metadata, "causal_intervention_node": node})


def intervene_flow(trace: TrajectoryTrace, flow: FlowOperator, *, transition: int, gain: float = 1.0) -> TrajectoryTrace:
    """Replace one observed velocity with a fitted field velocity.

    It is a controlled flow-space intervention; it does not pretend to rerun
    model weights. Use ``rollout_intervention`` when model re-execution exists.
    """

    if transition < 0 or transition >= len(trace.transitions):
        raise ValueError("flow transition is outside the trace")
    if not np.isfinite(gain):
        raise ValueError("flow intervention gain must be finite")
    transitions = list(trace.transitions)
    old = transitions[transition]
    fitted = flow.predict(old.source_state, old.source_depth)
    velocity = old.vector_field + gain * (fitted - old.vector_field)
    target = old.source_state + velocity * (old.target_depth - old.source_depth)
    transitions[transition] = dataclass_replace(old, target_state=target, delta=target - old.source_state, vector_field=velocity, jacobian=None, hessian_sketch=None, singular_values=None, uncertainty={**old.uncertainty, "intervention": 1.0})
    states = trace.hidden_states.copy()
    states[transition + 1] = target
    return dataclass_replace(trace, hidden_states=states, transitions=tuple(transitions), metadata={**trace.metadata, "flow_intervention_transition": transition})


def _mapped_states(teacher: TrajectoryTrace, student: TrajectoryTrace, alignment: AlignmentResult | None) -> np.ndarray:
    values = teacher.hidden_states if alignment is None else alignment.apply(teacher.hidden_states, depth=teacher.depth_coordinates)
    if values.shape[1] != student.state_dim:
        raise ValueError("teacher and student charts have different dimensions; supply teacher_to_student alignment")
    return np.column_stack([np.interp(student.depth_coordinates, teacher.depth_coordinates, values[:, column]) for column in range(values.shape[1])])


def _trajectory_error(a: Sequence[TrajectoryTrace], b: Sequence[TrajectoryTrace], alignment: AlignmentResult | None = None) -> float:
    if len(a) != len(b) or not a:
        raise ValueError("validation trace sets must be paired and non-empty")
    return float(np.mean([float(stable_l2(x.hidden_states - _mapped_states(y, x, alignment), name="trajectory validation residual")) / max(1.0, float(stable_l2(x.hidden_states, name="trajectory validation state"))) for x, y in zip(a, b)]))


def validate_causal_transfer(teacher: Sequence[TrajectoryTrace], baseline_student: Sequence[TrajectoryTrace], intervened_student: Sequence[TrajectoryTrace], *, alignment_error: float, teacher_to_student: AlignmentResult | None = None, stability_fn: Callable[[Sequence[TrajectoryTrace]], float] | None = None, collapse_threshold: float = 0.05, require_holdout: bool = True) -> ValidationReport:
    if not np.isfinite(alignment_error) or alignment_error < 0:
        raise ValueError("alignment_error must be finite and non-negative")
    if collapse_threshold < 0 or not np.isfinite(collapse_threshold):
        raise ValueError("collapse_threshold must be finite and non-negative")
    if len(teacher) != len(baseline_student) or len(teacher) != len(intervened_student) or not teacher:
        raise ValueError("teacher, baseline, and intervened holdout sets must be equally sized and non-empty")
    if any(t.probe_id != b.probe_id or b.probe_id != i.probe_id for t, b, i in zip(teacher, baseline_student, intervened_student)):
        raise ValueError("causal validation traces must retain probe pairing")
    baseline_error = _trajectory_error(baseline_student, teacher, teacher_to_student)
    intervened_error = _trajectory_error(intervened_student, teacher, teacher_to_student)
    effect = baseline_error - intervened_error
    stability = float(stability_fn(intervened_student) if stability_fn is not None else 1.0)
    if not np.isfinite(stability):
        raise FloatingPointError("causal stability score is non-finite")
    variances = [float(stable_l2(trace.hidden_states - trace.hidden_states.mean(axis=0, keepdims=True), name="intervened trajectory spread") / max(np.sqrt(trace.layer_count), 1.0)) for trace in intervened_student]
    baseline_variances = [float(stable_l2(trace.hidden_states - trace.hidden_states.mean(axis=0, keepdims=True), name="baseline trajectory spread") / max(np.sqrt(trace.layer_count), 1.0)) for trace in baseline_student]
    collapse = float(np.mean(variances) / max(np.mean(baseline_variances), 1e-12))
    passed = bool(intervened_error < baseline_error and effect > 0 and stability > 0.5 and collapse >= collapse_threshold)
    baseline_output_error = float(np.mean([float(stable_l2(student.hidden_states[-1] - _mapped_states(target, student, teacher_to_student)[-1], name="baseline output residual")) for student, target in zip(baseline_student, teacher)]))
    intervened_output_error = float(np.mean([float(stable_l2(student.hidden_states[-1] - _mapped_states(target, student, teacher_to_student)[-1], name="intervened output residual")) for student, target in zip(intervened_student, teacher)]))
    labels = [trace.metadata.get("probe_split") for trace in tuple(teacher) + tuple(baseline_student) + tuple(intervened_student)]
    holdout_ok = bool(labels and all(label == "holdout" for label in labels))
    notes = []
    if alignment_error < 0.05 and effect <= 0:
        notes.append("geometric match did not produce a positive causal effect")
    if collapse < collapse_threshold:
        notes.append("intervened trajectories show variance collapse")
    if require_holdout and not holdout_ok:
        notes.append("causal success was not marked on an explicitly labelled holdout split")
    passed = bool(passed and (holdout_ok or not require_holdout))
    real_connector_rerun = any("causal_intervention_node" in trace.metadata for trace in intervened_student)
    return ValidationReport(float(alignment_error), baseline_error, intervened_error, effect, stability, collapse, len(intervened_student), passed, {"effect_to_baseline": effect / max(baseline_error, 1e-12), "stability": stability, "baseline_output_error": baseline_output_error, "intervened_output_error": intervened_output_error, "output_effect": baseline_output_error - intervened_output_error, "holdout_separation": float(holdout_ok), "real_connector_rerun": real_connector_rerun, "semantic_claim": "not_established"}, tuple(notes))
