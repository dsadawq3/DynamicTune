import numpy as np
import json
from pathlib import Path
from types import SimpleNamespace

from faytuna_flow.connectors import HFLikeConnector, SyntheticConnector
from faytuna_flow.connectors import CapabilityError, TorchConnector, TorchHooks
from faytuna_flow.depth import detect_bifurcations, interpolate_path, monotone_correspondence, reconstruct_depth_path
from faytuna_flow.observation import ObservationProtocol, finite_jacobian, hessian_directional_sketch
from faytuna_flow.journal import ExperimentJournal, journal_traces
from faytuna_flow.gpt2 import GPT2Connector, GPT2ProbePolicy
from faytuna_flow.synthetic import BifurcationSystem, FakeHFLikeModel, LinearResidualSystem, random_stable_system
from faytuna_flow.types import Probe, ProbeSplit, TrajectoryTrace, TransitionObservation
from faytuna_flow.validation import rollout_intervention, validate_causal_transfer
from scripts.run_real_gpt2 import ProgressMonitor, collect_variant


def test_finite_differential_sketches_match_known_polynomial():
    function = lambda x: np.array([x[0] ** 2 + 3.0 * x[1], x[0] * x[1]])
    x = np.array([1.4, -0.7])
    jacobian = finite_jacobian(function, x, step=1e-5)
    assert np.allclose(jacobian, [[2.8, 3.0], [-0.7, 1.4]], atol=1e-5)
    sketch = hessian_directional_sketch(function, x, np.eye(2), step=1e-3)
    assert np.allclose(sketch[0], [2.0, 0.0], atol=1e-3)
    assert np.allclose(sketch[1], [0.0, 0.0], atol=1e-3)


def test_differential_sketches_remain_stable_at_non_unit_state_scale():
    function = lambda x: np.array([x[0] ** 2 / 1e3 + 0.5 * x[1], x[0] * x[1] / 1e3])
    x = np.array([1200.0, -800.0])
    small = finite_jacobian(function, x, step=1e-5)
    larger = finite_jacobian(function, x, step=5e-4)
    assert np.all(np.isfinite(small)) and np.all(np.isfinite(larger))
    assert np.allclose(small, larger, atol=1e-4)
    sketch = hessian_directional_sketch(function, x, np.eye(2), step=1e-3)
    assert np.all(np.isfinite(sketch))
    assert np.allclose(sketch[0], [2e-3, 0.0], atol=1e-5)


def test_observation_protocol_collects_finite_adaptive_trajectories():
    system = random_stable_system(layers=5, state_dim=3, seed=6, nonlinear=True)
    connector = SyntheticConnector(system, model_id="synthetic-test")
    probes = [Probe(f"p{i}", "algorithmic", {"chain": list(range(5))}, np.array([0.2 + i * 0.1, -0.1, 0.3])) for i in range(5)]
    traces = ObservationProtocol(hessian_rank=2).collect(connector, probes, seed=9)
    assert len(traces) == 5
    for trace in traces:
        assert trace.residual_states is not None
        assert np.all(np.diff(trace.depth_coordinates) > 0)
        assert np.all(np.isfinite(trace.hidden_states))
        assert all(t.jacobian.shape == (3, 3) for t in trace.transitions)
        assert all(t.hessian_sketch.shape == (2, 3) for t in trace.transitions)


def test_observation_seed_reproduces_differential_sketches():
    connector = SyntheticConnector(random_stable_system(layers=4, state_dim=3, seed=15, nonlinear=True))
    probe = Probe("deterministic", "adversarial_stability", {}, np.array([0.2, -0.4, 0.7]))
    first = ObservationProtocol(hessian_rank=3).collect(connector, [probe], seed=991)[0]
    second = ObservationProtocol(hessian_rank=3).collect(connector, [probe], seed=991)[0]
    assert np.array_equal(first.hidden_states, second.hidden_states)
    assert all(np.array_equal(a.hessian_sketch, b.hessian_sketch) for a, b in zip(first.transitions, second.transitions))


def test_large_sequence_state_auto_uses_bounded_directional_differential_observation(tmp_path: Path):
    class CountingLargeSystem:
        layer_count = 1
        state_dim = 16 * 1600

        def __init__(self):
            self.calls = 0

        def initial_state(self, probe):
            return np.zeros(self.state_dim, dtype=np.float64)

        def step(self, state, layer_index, probe):
            self.calls += 1
            return state + 0.01 * np.tanh(state) + 0.001

    system = CountingLargeSystem()
    connector = SyntheticConnector(system, model_id="gpt2-xl-like")
    probe = Probe("large", "long_context", {"sequence_length": 16}, np.zeros(system.state_dim))
    trace = ObservationProtocol(jacobian_rank=3, hessian_rank=2).collect(connector, [probe], seed=73)[0]
    observation = trace.metadata["differential_observation"]
    transition = trace.transitions[0]
    assert observation["backend"] == "directional_randomized_sketch"
    assert observation["jacobian_kind"] == "directional_sketch"
    assert observation["jacobian_rank"] == observation["jacobian_directions"] == 3
    assert observation["hessian_kind"] == "directional_sketch"
    assert observation["hessian_rank"] == 2
    assert observation["auto_downgraded"] is True
    assert transition.jacobian.shape == (system.state_dim, 3)
    assert transition.hessian_sketch.shape == (2, system.state_dim)
    # One trajectory call plus 2 directions per finite-difference step for
    # each Jacobian/Hessian rank and two step sizes: 1 + 4*3 + 4*2.
    assert system.calls == observation["estimated_forward_calls"] == 21
    assert observation["estimated_forward_calls"] < 2 * system.state_dim
    assert observation["estimated_peak_memory_bytes"] < 4_000_000
    assert observation["skipped_reason"]
    with __import__("pytest").raises(ValueError, match="coordinate finite Jacobian prohibited"):
        finite_jacobian(lambda value: value, np.zeros(513))

    journal_path = tmp_path / "large.jsonl"
    with ExperimentJournal(journal_path, run_id="large", seed=73) as journal:
        journal_traces(journal, [trace], model_role="teacher")
    events = [__import__("json").loads(line) for line in journal_path.read_text(encoding="utf-8").splitlines()]
    depth_event = next(event for event in events if event["event_type"] == "depth")
    assert depth_event["observation_backend"] == "directional_randomized_sketch"
    assert depth_event["jacobian_rank"] == 3
    assert depth_event["estimated_forward_calls"] == 21
    assert depth_event["estimated_peak_memory_bytes"] < 4_000_000


def test_gpt2_connector_does_not_label_post_layer_hidden_as_residual():
    torch = __import__("pytest").importorskip("torch")

    class ZeroEmbedding(torch.nn.Module):
        def forward(self, ids):
            return torch.zeros((*ids.shape, 768), dtype=torch.float32, device=ids.device)

    class Block(torch.nn.Module):
        def forward(self, hidden_states=None, **kwargs):
            return hidden_states + 0.001

    class TinyGPT2(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(()))
            self.transformer = torch.nn.Module()
            self.transformer.wte = ZeroEmbedding()
            self.transformer.wpe = ZeroEmbedding()
            self.transformer.h = torch.nn.ModuleList([Block() for _ in range(12)])
            self.config = SimpleNamespace(model_type="gpt2", n_layer=12, n_embd=768, n_head=12, n_positions=1024, vocab_size=50257)

    model = TinyGPT2()
    policy = GPT2ProbePolicy(lambda probe: {"input_ids": [5, 7], "position_ids": [0, 1], "attention_mask": [1, 1]}, sequence_length=2)
    connector = GPT2Connector(model, policy, variant="gpt2-small")
    assert connector.capabilities.residual_states is False
    assert connector.capabilities.automatic["residual_states"] is False
    assert "not labeled residual" in connector.capabilities.reasons["residual_states"]
    probe = Probe("gpt2", "long_context", {}, np.zeros(1536))
    trace = ObservationProtocol(jacobian_rank=2, hessian_rank=1).collect(connector, [probe], seed=4)[0]
    assert trace.residual_states is None
    assert trace.metadata["differential_observation"]["jacobian_kind"] == "directional_sketch"


def test_gpt2_connector_batches_directional_stencils_without_changing_call_contract():
    torch = __import__("pytest").importorskip("torch")

    class CountingBlock(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.calls = 0
            self.batch_sizes = []

        def forward(self, hidden_states=None, **kwargs):
            self.calls += 1
            self.batch_sizes.append(int(hidden_states.shape[0]))
            return hidden_states + 0.001 * torch.tanh(hidden_states)

    class TinyGPT2(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(()))
            self.transformer = torch.nn.Module()
            self.transformer.wte = torch.nn.Embedding(32, 768)
            self.transformer.wpe = torch.nn.Embedding(8, 768)
            self.transformer.h = torch.nn.ModuleList([CountingBlock() for _ in range(12)])
            self.config = SimpleNamespace(model_type="gpt2", n_layer=12, n_embd=768, n_head=12, n_positions=1024, vocab_size=50257)

    model = TinyGPT2()
    policy_calls = {"count": 0}

    def encode(probe):
        policy_calls["count"] += 1
        return {"input_ids": [5, 7], "position_ids": [0, 1], "attention_mask": [1, 1]}

    policy = GPT2ProbePolicy(encode, sequence_length=2)
    connector = GPT2Connector(model, policy, variant="gpt2-small")
    probe = Probe("gpt2-batched", "long_context", {}, np.zeros(1536))
    progress = []
    trace = ObservationProtocol(jacobian_rank=4, hessian_rank=3, jacobian_mode="directional").collect(connector, [probe], seed=7, progress_callback=progress.append)[0]
    # One nominal, one cached two-step Jacobian batch, and one cached
    # two-step Hessian batch per visible layer.  The result remains the same
    # directional finite-difference contract while avoiding 4+3 calls per
    # stencil.
    assert [block.calls for block in model.transformer.h] == [3] * 12
    # The nominal center is reused by the Hessian callback.  Rank-4 Jacobian
    # evaluation therefore carries 16 points and rank-3 Hessian evaluation
    # carries only its 12 off-center points; no finite-difference sample was
    # dropped.
    assert all(block.batch_sizes == [1, 16, 12] for block in model.transformer.h)
    assert trace.metadata["differential_observation"]["estimated_forward_calls"] == 36
    assert trace.metadata["differential_observation"]["estimated_stencil_samples"] == 12 * (1 + 4 * 4 + 4 * 3)
    assert trace.metadata["differential_observation"]["callback_nominal_reuse"] is True
    dispatch = [event for event in progress if event["event"] == "finite_difference_dispatch"]
    assert len(dispatch) == 12
    assert all(event["estimated_forward_calls"] == 3 and event["estimated_stencil_samples"] == 29 for event in dispatch)
    assert trace.metadata["differential_observation"]["jacobian_kind"] == "directional_sketch"
    assert trace.metadata["differential_observation"]["hessian_kind"] == "directional_sketch"
    observation = trace.metadata["differential_observation"]
    assert observation["callback_batched"] is True
    assert observation["callback_two_step_cache"] is True
    assert observation["estimated_callback_batch_memory_bytes"] == 2 * 16 * 1536 * 4
    assert observation["estimated_peak_memory_bytes"] >= observation["estimated_callback_batch_memory_bytes"]
    # One policy materialization serves the initial state and all layers; the
    # only second call is the explicit token-position metadata callback.
    assert policy_calls["count"] == 2


def test_hf_like_connector_reports_only_exposed_capabilities():
    model = FakeHFLikeModel(layers=3, state_dim=3, seed=12)
    connector = HFLikeConnector(model, model_id="fake-hf")
    probes = [Probe("hf-probe", "algorithmic", {"chain": [0, 1, 2]}, np.array([0.2, -0.3, 0.4]))]
    trace = ObservationProtocol(hessian_rank=2).collect(connector, probes)[0]
    assert connector.capabilities.hidden_states
    assert connector.capabilities.jacobian_sketch
    assert not connector.capabilities.attention_geometry
    assert not connector.capabilities.residual_states
    assert trace.residual_states is None
    assert connector.weights()


def test_torch_connector_is_batch_sequence_aware_and_never_passes_probe_to_module():
    torch = __import__("pytest").importorskip("torch")

    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = torch.nn.ModuleList([torch.nn.Linear(3, 3), torch.nn.Linear(3, 3)])

        def forward(self, hidden):
            for layer in self.layers:
                hidden = layer(hidden)
            return hidden

    model = Tiny()
    shape = (2, 4, 3)

    def encoder(probe, device):
        base = torch.as_tensor(probe.initial_state, dtype=torch.float32, device=device).reshape(1, 1, 3)
        return base.expand(*shape).contiguous()

    def decoder(state, probe):
        return torch.as_tensor(state, dtype=torch.float32).reshape(1, 1, 3).expand(*shape).contiguous()

    def observer(hidden):
        return hidden.mean(dim=(0, 1))

    hooks = TorchHooks(encoder, decoder, observer, token_position_observer=lambda probe: {"token_ids": [11, 12, 13], "position_ids": [0, 1, 2], "token_position_map": {"query": 2}})
    connector = TorchConnector(model, hooks, model_id="tiny-torch")
    probe = Probe("torch", "long_context", {"token_ids": [11, 12, 13]}, np.array([0.2, -0.4, 0.6]))
    trace = ObservationProtocol(hessian_rank=2).collect(connector, [probe])[0]
    assert trace.feature_space == "torch_hidden_chart"
    assert trace.token_ids.tolist() == [11, 12, 13]
    assert trace.position_ids.tolist() == [0, 1, 2]
    assert trace.token_position_map["query"] == 2
    assert trace.hidden_states.shape == (3, 3)
    assert connector.capabilities.automatic["batch_sequence_forward"] is True
    assert connector.capabilities.token_position_correspondence is True
    assert all(np.all(np.isfinite(item.hidden_states)) for item in [trace])
    assert isinstance(HFLikeConnector(model, hooks=hooks), TorchConnector)
    try:
        HFLikeConnector(model)
    except CapabilityError as error:
        assert "explicit state_encoder/state_decoder/state_observer" in str(error)
    else:
        raise AssertionError("torch module without hooks must fail clearly")


def test_causal_validation_uses_real_torch_connector_rerun():
    torch = __import__("pytest").importorskip("torch")

    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = torch.nn.ModuleList([
                torch.nn.Linear(2, 2, bias=False),
                torch.nn.Linear(2, 2, bias=False),
                torch.nn.Linear(2, 2, bias=False),
            ])
            for layer in self.layers:
                torch.nn.init.eye_(layer.weight)

    model = Tiny()

    def encoder(probe, device):
        return torch.as_tensor(probe.initial_state, dtype=torch.float32, device=device).reshape(1, 1, 2)

    def decoder(state, probe):
        return torch.as_tensor(state, dtype=torch.float32).reshape(1, 1, 2)

    connector = TorchConnector(model, TorchHooks(encoder, decoder, lambda hidden: hidden[0, 0]))
    probe = Probe("holdout-causal", "perturbation", {}, np.asarray([0.4, -0.2]), split="holdout")
    baseline = ObservationProtocol().collect(connector, [probe])[0]
    rerun = rollout_intervention(connector, probe, baseline, node=1, delta=np.asarray([0.25, 0.0]))
    assert not np.allclose(rerun.hidden_states[-1], baseline.hidden_states[-1])
    report = validate_causal_transfer((rerun,), (baseline,), (rerun,), alignment_error=0.0, require_holdout=True)
    assert report.passed
    assert report.metrics["real_connector_rerun"] is True


def test_optional_normalization_and_attention_geometry_are_carried():
    class GeometricSystem(LinearResidualSystem):
        def normalization_geometry(self, state, layer_index, probe):
            return {"rms": float(np.sqrt(np.mean(state ** 2))), "layer": layer_index}

        def attention_geometry(self, state, layer_index, probe):
            return {"entropy": float(np.log1p(np.linalg.norm(state))), "layer": layer_index}

    system = GeometricSystem(np.stack([np.eye(2) * -0.1] * 2), np.zeros((2, 2)), dt=0.1)
    trace = ObservationProtocol().collect(SyntheticConnector(system), [Probe("geo", "structured", {}, np.array([1.0, 0.5]))])[0]
    assert trace.transitions[0].normalization_geometry["layer"] == 0
    assert trace.transitions[0].attention_geometry["layer"] == 0


def test_depth_path_interpolation_and_monotone_gap_are_explicit():
    states = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 1.0], [3.0, 1.0]])
    path = reconstruct_depth_path(states)
    assert np.all(np.diff(path.coordinates) > 0)
    assert np.allclose(interpolate_path(states, path.coordinates, 0.5), [1.5, 0.5], atol=0.3)
    correspondence = monotone_correspondence(states, states[[0, 1, 3]])
    assert correspondence.gaps
    assert all(a[0] < b[0] and a[1] < b[1] for a, b in zip(correspondence.pairs, correspondence.pairs[1:]))
    assert all(np.isfinite(g.uncertainty) for g in correspondence.gaps)


def _trace_with_curvatures(curvatures, model_id):
    states = np.arange(6 * 2, dtype=float).reshape(6, 2) / 10.0
    coordinates = np.linspace(0.0, 1.0, 6)
    transitions = []
    for i, curvature in enumerate(curvatures):
        delta = states[i + 1] - states[i]
        transitions.append(TransitionObservation(i, i + 1, coordinates[i], coordinates[i + 1], states[i], states[i + 1], delta, delta / (coordinates[i + 1] - coordinates[i]), curvature=curvature))
    return TrajectoryTrace(model_id, model_id, tuple(range(-1, 5)), coordinates, states, None, tuple(transitions))


def test_bifurcation_detector_reports_a_micro_candidate_without_claiming_proof():
    traces = [_trace_with_curvatures([0.0, 0.0, 0.0, 0.0, 0.0], f"t{i}") for i in range(5)]
    traces[2] = _trace_with_curvatures([0.0, 0.0, 10.0, 0.0, 0.0], "outlier")
    points = detect_bifurcations(traces)
    assert points
    assert any(point.level == "micro" for point in points)
    assert all(0.0 <= point.confidence <= 1.0 for point in points)


def test_probe_stable_direction_seed_makes_resume_equivalent():
    system = random_stable_system(layers=3, state_dim=4, seed=44, nonlinear=True)
    connector = SyntheticConnector(system, model_id="resume-equivalence")
    probes = [Probe(f"resume-{index}", "algorithmic", {"step": index}, np.full(4, 0.1 * index)) for index in range(3)]
    batch = ObservationProtocol(jacobian_rank=2, hessian_rank=2).collect(connector, probes, seed=117)
    singles = [ObservationProtocol(jacobian_rank=2, hessian_rank=2).collect(connector, [probe], seed=117)[0] for probe in probes]
    for complete, resumed in zip(batch, singles):
        assert np.array_equal(complete.hidden_states, resumed.hidden_states)
        assert complete.metadata["probe_seed"] == resumed.metadata["probe_seed"]
        for first, second in zip(complete.transitions, resumed.transitions):
            assert np.array_equal(first.jacobian, second.jacobian)
            assert np.array_equal(first.hessian_sketch, second.hessian_sketch)


def test_progress_monitor_is_strict_jsonl_and_human_readable(tmp_path: Path):
    monitor = ProgressMonitor(tmp_path, run_id="smoke-progress")
    monitor.emit("observation", "finite_difference_dispatch", variant="gpt2-small", split="train", probe_index=0, probe_total=2, layer_index=1, layer_total=3, progress_units=0.67, progress_total=2, estimated_forward_calls=3, finite=True, completed_artifact=tmp_path / "probe_00000.npz")
    monitor.emit("observation", "probe_complete", variant="gpt2-small", split="train", probe_index=0, probe_total=2, layer_index=2, layer_total=3, progress_units=1.0, progress_total=2, finite=True)
    monitor.close()
    records = [json.loads(line) for line in (tmp_path / "progress.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(records) == 2
    assert all(record["schema_version"] == "faytuna-progress-v1" for record in records)
    assert all("NaN" not in json.dumps(record) and "Infinity" not in json.dumps(record) for record in records)
    assert set(records[0]["memory"]) == {"rss_bytes", "private_bytes", "commit_bytes"}
    assert records[0]["estimated_forward_calls"] == 3
    assert "finite_difference_dispatch" in (tmp_path / "progress.log").read_text(encoding="utf-8")


def test_real_runner_resume_skips_validated_probe_artifacts(tmp_path: Path, monkeypatch):
    torch = __import__("pytest").importorskip("torch")
    import scripts.run_real_gpt2 as real_runner

    class Block(torch.nn.Module):
        def __init__(self, counter):
            super().__init__()
            self.counter = counter

        def forward(self, hidden_states=None, **kwargs):
            self.counter["calls"] += 1
            return hidden_states + 0.001 * torch.tanh(hidden_states)

    def factory():
        torch.manual_seed(991)
        counter = {"calls": 0}
        model = torch.nn.Module()
        model.anchor = torch.nn.Parameter(torch.zeros(()))
        model.transformer = torch.nn.Module()
        model.transformer.wte = torch.nn.Embedding(32, 768)
        model.transformer.wpe = torch.nn.Embedding(8, 768)
        model.transformer.h = torch.nn.ModuleList([Block(counter) for _ in range(12)])
        model.config = SimpleNamespace(model_type="gpt2", n_layer=12, n_embd=768, n_head=12, n_positions=1024, vocab_size=50257)
        return model, counter

    counters = []

    def fake_load(path):
        model, counter = factory()
        counters.append(counter)
        return object(), model

    monkeypatch.setattr(real_runner, "_load_model", fake_load)
    policy = GPT2ProbePolicy(lambda probe: {"input_ids": [5, 7], "position_ids": [0, 1], "attention_mask": [1, 1]}, sequence_length=2)
    split = ProbeSplit(
        (Probe("resume-train", "algorithmic", {}, np.zeros(1536)),),
        (Probe("resume-validation", "algorithmic", {}, np.zeros(1536)),),
        (Probe("resume-holdout", "algorithmic", {}, np.zeros(1536)),),
    )
    output = tmp_path / "real-runner"
    first_monitor = ProgressMonitor(output, run_id="first")
    collect_variant("gpt2-small", tmp_path / "checkpoint", policy, split, output, (0, 1), 1, 1, monitor=first_monitor, resume=False)
    first_monitor.close()
    first_calls = counters[-1]["calls"]
    second_monitor = ProgressMonitor(output, run_id="second")
    collect_variant("gpt2-small", tmp_path / "checkpoint", policy, split, output, (0, 1), 1, 1, monitor=second_monitor, resume=True)
    second_monitor.close()
    assert first_calls == 18
    assert counters[-1]["calls"] == 0
    assert len(list((output / "gpt2-small" / "probes").rglob("*.npz"))) == 3
    records = [json.loads(line) for line in (output / "progress.jsonl").read_text(encoding="utf-8").splitlines()]
    assert any(record["event"] == "probe_resumed" for record in records)
    assert any(record["event"] == "split_resumed" for record in records)
    dispatch = next(record for record in records if record["event"] == "finite_difference_dispatch")
    assert dispatch["work_unit_name"] == "stencil_samples"
    assert dispatch["work_units_delta"] == 9
    assert dispatch["throughput_work_units_per_second"] is not None
    assert all("NaN" not in json.dumps(record) and "Infinity" not in json.dumps(record) for record in records)

    # A changed payload with the same probe id invalidates only that split;
    # the other complete split artifacts remain reusable.
    changed_split = ProbeSplit(
        (Probe("resume-train", "algorithmic", {"changed": True}, np.zeros(1536)),),
        split.validation,
        split.holdout,
    )
    third_monitor = ProgressMonitor(output, run_id="third")
    collect_variant("gpt2-small", tmp_path / "checkpoint", policy, changed_split, output, (0, 1), 1, 1, monitor=third_monitor, resume=True)
    third_monitor.close()
    assert counters[-1]["calls"] == 6
    third_records = [json.loads(line) for line in (output / "progress.jsonl").read_text(encoding="utf-8").splitlines()]
    assert any(record["event"] == "split_resumed" and record.get("split") == "validation" for record in third_records)


def test_proportional_student_layers_maps_depth_evenly():
    from scripts.run_real_gpt2 import _proportional_student_layers

    student_layers = _proportional_student_layers((0, 11, 23, 35, 47), teacher_total=48, student_total=12)
    assert student_layers == (0, 3, 5, 8, 11)
    for i in range(len(student_layers) - 1):
        assert student_layers[i + 1] - student_layers[i] <= 3

    assert _proportional_student_layers(None) is None
    custom = _proportional_student_layers((0, 24, 47), teacher_total=48, student_total=12)
    assert custom == (0, 6, 11)
