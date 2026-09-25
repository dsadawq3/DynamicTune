from pathlib import Path

import numpy as np

from faytuna_flow.artifacts import load_alignment, load_flow, load_traces, save_alignment, save_flow, save_traces
from faytuna_flow.connectors import SyntheticConnector
from faytuna_flow.flow import fit_flow_operator
from faytuna_flow.geometry import fit_alignment
from faytuna_flow.observation import ObservationProtocol
from faytuna_flow.probes import ProbeGenerator, ProbeGeneratorConfig
from faytuna_flow.synthetic import random_stable_system
from faytuna_flow.types import Probe


def test_trace_alignment_and_flow_artifact_round_trips(tmp_path: Path):
    connector = SyntheticConnector(random_stable_system(layers=3, state_dim=3, seed=71), model_id="artifact-model")
    probes = ProbeGenerator(ProbeGeneratorConfig(state_dim=3, per_family=3, seed=72)).generate()[:5]
    probes[0] = Probe(probes[0].probe_id, probes[0].family, {"token_ids": [5, 8, 13], "position_ids": [0, 1, 2], "token_position_map": {"query": 2}}, probes[0].initial_state)
    traces = ObservationProtocol().collect(connector, probes)
    trace_path = save_traces(traces, tmp_path / "traces.npz")
    restored = load_traces(trace_path)
    probe_path = save_traces([traces[0]], tmp_path / "probe.npz", compressed=False, durable=True)
    probe_restored = load_traces(probe_path)
    assert len(restored) == len(traces)
    assert np.allclose(restored[0].hidden_states, traces[0].hidden_states)
    assert np.allclose(restored[0].transitions[0].vector_field, traces[0].transitions[0].vector_field)
    assert restored[0].token_ids.tolist() == [5, 8, 13]
    assert restored[0].position_ids.tolist() == [0, 1, 2]
    assert restored[0].token_position_map["query"] == 2
    assert restored[0].uncertainty is not None
    assert np.array_equal(probe_restored[0].hidden_states, traces[0].hidden_states)
    assert not list(tmp_path.glob(".*.tmp")) and not list(tmp_path.glob(".*.npz"))
    assert "path_signature" in restored[0].metadata
    assert "differential_signature" in restored[0].metadata
    alignment = fit_alignment(np.asarray([t.hidden_states[0] for t in traces]), np.asarray([t.hidden_states[0] for t in traces]), kind="affine")
    alignment_path = save_alignment(alignment, tmp_path / "alignment.json")
    loaded_alignment = load_alignment(alignment_path)
    assert np.allclose(loaded_alignment.matrix, alignment.matrix)
    flow = fit_flow_operator(traces)
    flow_path = save_flow(flow, tmp_path / "flow.npz")
    loaded_flow = load_flow(flow_path)
    assert np.allclose(loaded_flow.matrices, flow.matrices)
