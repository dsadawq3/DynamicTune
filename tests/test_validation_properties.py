import numpy as np

from faytuna_flow.connectors import SyntheticConnector
from faytuna_flow.flow import fit_flow_operator
from faytuna_flow.observation import ObservationProtocol
from faytuna_flow.probes import ProbeGenerator, ProbeGeneratorConfig
from faytuna_flow.synthetic import random_stable_system
from faytuna_flow.validation import intervene_flow, intervene_trace, rollout_intervention
from faytuna_flow.pipeline import collect_synthetic_pair
from faytuna_flow.geometry import fit_alignment
from faytuna_flow.validation import validate_causal_transfer


def test_causal_rollout_reruns_downstream_dynamics_and_flow_intervention_is_finite():
    system = random_stable_system(layers=5, state_dim=3, seed=51, nonlinear=True)
    connector = SyntheticConnector(system, model_id="causal-synthetic")
    probe = ProbeGenerator(ProbeGeneratorConfig(state_dim=3, per_family=3, seed=52)).generate()[0]
    trace = ObservationProtocol().collect(connector, [probe])[0]
    rolled = rollout_intervention(connector, probe, trace, node=2, delta=np.array([0.3, 0.0, -0.1]))
    record_only = intervene_trace(trace, node=2, delta=np.array([0.3, 0.0, -0.1]))
    assert np.all(np.isfinite(rolled.hidden_states))
    assert np.linalg.norm(rolled.hidden_states[-1] - trace.hidden_states[-1]) > 0.0
    assert np.allclose(record_only.hidden_states[2], rolled.hidden_states[2])
    flow = fit_flow_operator([trace])
    changed = intervene_flow(trace, flow, transition=1, gain=0.5)
    assert np.all(np.isfinite(changed.hidden_states))
    assert changed.metadata["flow_intervention_transition"] == 1


def test_randomized_finite_and_monotone_invariants_without_external_property_framework():
    # Dependency-free property test: many seeds cover scales and depths that a
    # single hand-picked example would miss.
    for seed in range(20):
        layers = 2 + seed % 5
        dim = 2 + seed % 4
        system = random_stable_system(layers=layers, state_dim=dim, seed=100 + seed, nonlinear=bool(seed % 2))
        connector = SyntheticConnector(system, model_id=f"random-{seed}")
        probes = ProbeGenerator(ProbeGeneratorConfig(state_dim=dim, per_family=3, seed=200 + seed)).generate()[:4]
        traces = ObservationProtocol(hessian_rank=min(dim, 2)).collect(connector, probes, seed=300 + seed)
        for trace in traces:
            assert np.all(np.isfinite(trace.hidden_states))
            assert np.all(np.diff(trace.depth_coordinates) > 0)
            assert np.all(np.isfinite(np.concatenate([transition.vector_field for transition in trace.transitions])))


def test_causal_validation_maps_teacher_chart_and_requires_explicit_holdout():
    teacher_bundle = collect_synthetic_pair(teacher_dim=5, student_dim=3, teacher_layers=5, student_layers=4, per_family=3, seed=530, split="holdout")
    # Use a separate baseline/intervention chart with the same probes so the
    # validation path exercises teacher→student dimensional transport.
    baseline_bundle = collect_synthetic_pair(teacher_dim=5, student_dim=3, teacher_layers=5, student_layers=4, per_family=3, seed=530, split="holdout")
    teacher_initial = np.asarray([trace.hidden_states[0] for trace in teacher_bundle.teacher])
    student_initial = np.asarray([trace.hidden_states[0] for trace in baseline_bundle.student])
    alignment = fit_alignment(teacher_initial, student_initial, kind="low_rank", rank=3, source_role="teacher", target_role="student")
    intervened = [rollout_intervention(type("C", (), {"layer_ids": trace.layer_ids[1:], "initial_state": lambda self, probe, t=trace: t.hidden_states[0], "transition": lambda self, state, layer, probe, t=trace: t.hidden_states[layer + 1]})(), probe, trace, node=1, delta=np.zeros(3)) for trace, probe in zip(baseline_bundle.student, [type("P", (), {"initial_state": trace.hidden_states[0]})() for trace in baseline_bundle.student])]
    report = validate_causal_transfer(teacher_bundle.teacher, baseline_bundle.student, intervened, alignment_error=alignment.paired_error, teacher_to_student=alignment)
    assert report.metrics["holdout_separation"] == 1.0
    assert np.isfinite(report.baseline_functional_error)
