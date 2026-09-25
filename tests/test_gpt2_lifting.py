from __future__ import annotations

import json
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from faytuna_flow.gpt2 import (
    GPT2ActivationCaptureResult,
    GPT2TensorLiftMapping,
    build_gpt2_teacher_flow_activation_targets,
    build_gpt2_surgery_plan,
    capture_gpt2_conv1d_activations,
    export_gpt2_checkpoint_pair,
    fit_gpt2_conv1d_activation_least_squares,
    gpt2_depth_gain_weight,
)
from faytuna_flow.solver import ConstrainedCorrection
from faytuna_flow.types import FlowOperator, TrajectoryTrace, TransitionObservation


def _correction(dim: int, value: float = 0.02, *, accepted: bool = True) -> ConstrainedCorrection:
    matrices = np.eye(dim, dtype=np.float64)[None, :, :] * value
    return ConstrainedCorrection(
        matrices=matrices,
        biases=np.zeros((1, dim), dtype=np.float64),
        accepted=np.array([accepted]),
        reasons={} if accepted else {0: "confidence below threshold"},
        confidence=np.array([0.9]),
        diagnostics={},
    )


def _weights(hidden: int) -> dict[str, np.ndarray]:
    return {
        "transformer.h.0.attn.c_attn.weight": np.ones((hidden, 3 * hidden), dtype=np.float32),
        "transformer.h.0.attn.c_proj.weight": np.ones((hidden, hidden), dtype=np.float32),
        "transformer.h.0.mlp.c_fc.weight": np.ones((hidden, 4 * hidden), dtype=np.float32),
        "transformer.h.0.mlp.c_proj.weight": np.ones((4 * hidden, hidden), dtype=np.float32),
        "transformer.h.0.ln_1.weight": np.ones((hidden,), dtype=np.float32),
        "same_shape_but_not_gpt2.weight": np.ones((hidden, hidden), dtype=np.float32),
    }


def test_gpt2_lifting_is_shape_aware_and_uses_explicit_conv1d_orientation():
    hidden = 3
    projection = np.zeros((2 * hidden, 2), dtype=np.float64)
    projection[:hidden, :] = np.eye(hidden, 2)
    projection[hidden:, :] = np.eye(hidden, 2)
    weights = _weights(hidden)
    mapping = [
        GPT2TensorLiftMapping(0, "transformer.h.0.attn.c_attn.weight", side="input", block_index=0),
        GPT2TensorLiftMapping(0, "transformer.h.0.attn.c_proj.weight", side="output", block_index=0),
        GPT2TensorLiftMapping(0, "transformer.h.0.mlp.c_fc.weight", side="input", block_index=0),
        GPT2TensorLiftMapping(0, "transformer.h.0.mlp.c_proj.weight", side="output", block_index=0),
    ]
    plan = build_gpt2_surgery_plan(
        weights,
        _correction(2),
        chart_projection=projection,
        sequence_length=2,
        hidden_size=hidden,
        mapping=mapping,
        max_cross_token_ratio=2.0,
        mode="apply",
    )
    assert set(plan.applied_tensors) == {item.tensor_name for item in mapping}
    assert plan.metadata["dense_state_matrix_allocated"] is False
    assert plan.metadata["dense_state_matrix_forbidden"] is True
    assert "transformer.h.0.ln_1.weight" in plan.skipped_tensors
    assert "same_shape_but_not_gpt2.weight" in plan.skipped_tensors
    # The token-local contraction is diag([[.02, 0, 0], ...]) on hidden space.
    expected_hidden = np.diag([0.02, 0.02, 0.0])
    assert np.allclose(
        plan.updates["transformer.h.0.attn.c_proj.weight"],
        weights["transformer.h.0.attn.c_proj.weight"] + weights["transformer.h.0.attn.c_proj.weight"] @ expected_hidden,
    )
    assert np.allclose(
        plan.updates["transformer.h.0.mlp.c_proj.weight"],
        weights["transformer.h.0.mlp.c_proj.weight"] + weights["transformer.h.0.mlp.c_proj.weight"] @ expected_hidden,
    )
    blocks = projection.reshape(2, hidden, 2)
    compact = _correction(2).matrices[0]
    explicit_cross = sum(
        np.linalg.norm(blocks[left] @ compact @ blocks[right].T) ** 2
        for left in range(2)
        for right in range(2)
        if left != right
    ) ** 0.5
    assert np.isclose(
        plan.metadata["lift_reports"]["0"]["cross_token_norm"],
        explicit_cross,
        rtol=1e-10,
        atol=1e-12,
    )


def test_gpt2_activation_least_squares_lift_solves_row_vector_conv1d_effect():
    rng = np.random.default_rng(41)
    activations = rng.normal(size=(64, 5))
    planted = rng.normal(size=(5, 3))
    delta, report = fit_gpt2_conv1d_activation_least_squares(activations, activations @ planted, ridge=0.0)
    assert np.allclose(delta, planted, atol=1e-9)
    assert report["backend"] == "activation_to_weight_least_squares"
    assert report["input_rank"] == 5
    assert report["hooks_required"]


def test_gpt2_forward_hooks_capture_batch_sequence_rows_and_exact_lift():
    torch = pytest.importorskip("torch")

    class KwConv1D(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.eye(3))

        def forward(self, hidden=None):
            return hidden @ self.weight

    class HookModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.transformer = torch.nn.Module()
            self.transformer.h = torch.nn.ModuleList()
            block = torch.nn.Module()
            block.attn = torch.nn.Module()
            block.attn.c_proj = KwConv1D()
            block.mlp = torch.nn.Module()
            block.mlp.c_proj = KwConv1D()
            self.transformer.h.append(block)

        def forward(self, hidden=None):
            if hidden is None:
                raise ValueError("hidden is required")
            return self.transformer.h[0].mlp.c_proj(hidden=self.transformer.h[0].attn.c_proj(hidden=hidden))

    model = HookModel().eval()
    rng = np.random.default_rng(123)
    hidden = torch.as_tensor(rng.normal(size=(2, 4, 3)), dtype=torch.float32)
    module_name = "transformer.h.0.attn.c_proj"
    capture = capture_gpt2_conv1d_activations(model, [{"hidden": hidden}], [module_name])
    assert isinstance(capture, GPT2ActivationCaptureResult)
    assert capture.to_dict()["backend"] == "torch_forward_hooks"
    assert capture.forward_calls == 1
    assert capture.inputs[module_name].shape == (8, 3)
    assert capture.outputs[module_name].shape == (8, 3)
    assert np.all(np.isfinite(capture.inputs[module_name]))

    planted = np.asarray([[0.01, -0.02, 0.03], [0.02, 0.01, -0.01], [-0.03, 0.02, 0.01]], dtype=np.float64)
    tensor_name = module_name + ".weight"
    plan = build_gpt2_surgery_plan(
        model.state_dict(),
        _correction(1),
        chart_projection=None,
        sequence_length=4,
        hidden_size=3,
        mapping=[GPT2TensorLiftMapping(0, tensor_name, side="input", block_index=0)],
        mode="apply",
        activation_inputs={tensor_name: capture.inputs[module_name]},
        activation_target_deltas={tensor_name: capture.inputs[module_name] @ planted},
        activation_ridge=0.0,
    )
    assert plan.metadata["lift_method"] == "exact_activation_ls"
    assert plan.metadata["exact_activation_ls_tensors"] == (tensor_name,)
    assert plan.metadata["lift_reports"][f"activation_ls:{tensor_name}"]["backend"] == "activation_to_weight_least_squares"
    assert np.allclose(
        plan.updates[tensor_name],
        model.state_dict()[tensor_name].detach().numpy() + planted.astype(np.float32),
        atol=1e-6,
    )


def test_teacher_flow_target_builder_is_explicit_and_train_scoped():
    torch = pytest.importorskip("torch")

    class KwConv1D(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.eye(3))

        def forward(self, hidden=None):
            return hidden @ self.weight

    class HookModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.transformer = torch.nn.Module()
            self.transformer.h = torch.nn.ModuleList()
            block = torch.nn.Module()
            block.attn = torch.nn.Module()
            block.attn.c_proj = KwConv1D()
            block.mlp = torch.nn.Module()
            block.mlp.c_proj = KwConv1D()
            self.transformer.h.append(block)

        def forward(self, input_ids, position_ids=None, attention_mask=None):
            ids = input_ids.to(dtype=torch.float32)
            positions = torch.arange(ids.shape[1], device=ids.device, dtype=torch.float32).expand_as(ids)
            hidden = torch.stack((ids, positions, ids.square()), dim=-1)
            return self.transformer.h[0].mlp.c_proj(hidden=self.transformer.h[0].attn.c_proj(hidden=hidden))

    def state_for(ids):
        return np.concatenate([np.asarray([value, index, value * value], dtype=np.float64) for index, value in enumerate(ids)])

    traces = []
    for probe_id, ids in (("train-a", [1, 2, 3]), ("train-b", [4, 5, 6])):
        source = state_for(ids)
        target = source * 1.01
        transition = TransitionObservation(0, 1, 0.0, 1.0, source, target, target - source, target - source)
        traces.append(TrajectoryTrace("gpt2-small", probe_id, (-1, 0), np.asarray([0.0, 1.0]), np.vstack([source, target]), None, (transition,), {"probe_split": "train"}, token_ids=np.asarray(ids), position_ids=np.arange(3)))
    student = FlowOperator(np.asarray([0.0, 1.0]), np.zeros((2, 9, 9)), np.zeros((2, 9)), np.ones(2), np.ones(2), np.zeros(2))
    teacher_matrices = np.stack([np.eye(9) * 0.1, np.eye(9) * 0.1])
    teacher = FlowOperator(np.asarray([0.0, 1.0]), teacher_matrices, np.zeros((2, 9)), np.ones(2), np.ones(2), np.ones(2))
    tensor_name = "transformer.h.0.mlp.c_proj.weight"
    skipped_name = "transformer.h.0.attn.c_proj.weight"
    mapping = [
        GPT2TensorLiftMapping(0, skipped_name, side="input", block_index=0),
        GPT2TensorLiftMapping(0, tensor_name, side="output", block_index=0),
    ]
    inputs, targets, metadata = build_gpt2_teacher_flow_activation_targets(
        HookModel().eval(), traces, student, teacher, mapping, sequence_length=3, hidden_size=3, target_effect_ratio=1.0
    )
    assert metadata["target_fit_split"] == "train"
    assert metadata["target_kind"] == "depth_step_teacher_minus_student_flow_in_student_chart"
    assert metadata["target_is_module_attribution"] is False
    assert metadata["site_policy"] == "mlp_residual_only"
    assert metadata["direct_to_block_output"] is True
    assert skipped_name in metadata["skipped_site_mappings"]
    assert inputs[tensor_name].shape == targets[tensor_name].shape == (6, 3)
    assert np.allclose(targets[tensor_name], inputs[tensor_name] * 0.1, atol=1e-8)
    assert metadata["finite"] is True
    with pytest.raises(ValueError, match="train-only"):
        build_gpt2_teacher_flow_activation_targets(
            HookModel().eval(), [replace(traces[0], metadata={"probe_split": "validation"})],
            student, teacher, mapping, sequence_length=3, hidden_size=3
        )


def test_mlp_residual_site_recovers_planted_block_effect_without_serial_double_counting():
    """The direct GPT-2 site contract is true for the final MLP residual only."""

    torch = pytest.importorskip("torch")

    class ResidualBlock(torch.nn.Module):
        def __init__(self, width: int = 3):
            super().__init__()
            self.attn = torch.nn.Linear(width, width, bias=False)
            self.mlp = torch.nn.Linear(width, width, bias=False)

        def forward(self, x):
            after_attn = x + self.attn(x)
            return after_attn + self.mlp(after_attn), after_attn

    torch.manual_seed(5)
    block = ResidualBlock().eval()
    x = torch.randn(23, 3)
    with torch.no_grad():
        baseline, mlp_input = block(x)
    planted = torch.tensor([[0.02, -0.01, 0.03], [-0.01, 0.02, 0.01], [0.01, 0.00, -0.02]])
    with torch.no_grad():
        block.mlp.weight.add_(planted.T)  # torch Linear uses y=x @ weight.T
        updated, _ = block(x)
        block.mlp.weight.sub_(planted.T)
    target_block_delta = (updated - baseline).numpy()
    fitted, report = fit_gpt2_conv1d_activation_least_squares(mlp_input.numpy(), target_block_delta, ridge=0.0)
    assert np.allclose(fitted, planted.numpy(), atol=2e-6)
    assert report["fit_relative_residual"] < 3e-6
    # Assigning this same block target to the prior attention path as well is
    # a different intervention; it is not a valid decomposition of one delta.
    with torch.no_grad():
        block.attn.weight.add_(planted.T)
        block.mlp.weight.add_(planted.T)
        doubled, _ = block(x)
        block.attn.weight.sub_(planted.T)
        block.mlp.weight.sub_(planted.T)
    assert np.linalg.norm((doubled - baseline).numpy() - target_block_delta) > 1e-3


def test_teacher_flow_site_contract_cannot_fall_back_to_serial_chart_lifts():
    """Missing exact maps remain skipped; they never silently use a chart lift."""

    rng = np.random.default_rng(71)
    hidden = 3
    weights = _weights(hidden)
    mlp_name = "transformer.h.0.mlp.c_proj.weight"
    mapping = [
        GPT2TensorLiftMapping(0, "transformer.h.0.attn.c_attn.weight", side="input", block_index=0),
        GPT2TensorLiftMapping(0, "transformer.h.0.attn.c_proj.weight", side="input", block_index=0),
        GPT2TensorLiftMapping(0, "transformer.h.0.mlp.c_fc.weight", side="input", block_index=0),
        GPT2TensorLiftMapping(0, mlp_name, side="output", block_index=0),
    ]
    inputs = rng.normal(size=(29, 4 * hidden))
    planted = rng.normal(scale=0.01, size=(4 * hidden, hidden))
    metadata = {
        "site_policy": "mlp_residual_only",
        "skipped_site_mappings": {
            item.tensor_name: "not a direct residual site"
            for item in mapping[:-1]
        },
    }
    plan = build_gpt2_surgery_plan(
        weights,
        _correction(1),
        chart_projection=None,
        sequence_length=1,
        hidden_size=hidden,
        mapping=mapping,
        mode="apply",
        activation_inputs={mlp_name: inputs},
        activation_target_deltas={mlp_name: inputs @ planted},
        activation_ridge=0.0,
        activation_target_metadata=metadata,
    )
    assert plan.applied_tensors == (mlp_name,)
    assert plan.metadata["heuristic_chart_tensors"] == ()
    for item in mapping[:-1]:
        assert plan.skipped_tensors[item.tensor_name] == "not a direct residual site"


def test_gpt2_lifting_rejects_omitted_cross_token_coupling_and_apply_fails():
    hidden = 3
    projection = np.zeros((2 * hidden, 2), dtype=np.float64)
    projection[:hidden, :] = np.eye(hidden, 2)
    projection[hidden:, :] = np.eye(hidden, 2)
    weights = {"transformer.h.0.attn.c_proj.weight": np.eye(hidden, dtype=np.float32)}
    mapping = [GPT2TensorLiftMapping(0, "transformer.h.0.attn.c_proj.weight", side="output")]
    diagnostic = build_gpt2_surgery_plan(
        weights,
        _correction(2),
        chart_projection=projection,
        sequence_length=2,
        hidden_size=hidden,
        mapping=mapping,
        max_cross_token_ratio=0.1,
        mode="diagnostic",
    )
    assert not diagnostic.applied_tensors
    assert "cross-token correction ratio" in diagnostic.skipped_tensors[mapping[0].tensor_name]
    with pytest.raises(Exception, match="no applicable tensor updates"):
        build_gpt2_surgery_plan(
            weights,
            _correction(2),
            chart_projection=projection,
            sequence_length=2,
            hidden_size=hidden,
            mapping=mapping,
            max_cross_token_ratio=0.1,
            mode="apply",
        )


def test_gpt2_experimental_mode_lifts_raw_rejected_node_and_preserves_reason():
    hidden = 3
    projection = np.zeros((2 * hidden, 2), dtype=np.float64)
    projection[:hidden, :] = np.eye(hidden, 2)
    mapping = [GPT2TensorLiftMapping(0, "transformer.h.0.attn.c_proj.weight", side="output", block_index=0)]
    plan = build_gpt2_surgery_plan(
        {"transformer.h.0.attn.c_proj.weight": np.eye(hidden, dtype=np.float32)},
        _correction(2, accepted=False),
        chart_projection=projection,
        sequence_length=2,
        hidden_size=hidden,
        mapping=mapping,
        mode="experimental",
    )
    assert plan.metadata["trust_status"] == "experimental_untrusted"
    assert plan.metadata["acceptance_gate_bypassed"] is True
    assert plan.metadata["raw_unaccepted_transitions_used"] == [0]
    assert plan.metadata["rollback_reasons"]["0"] == "confidence below threshold"
    assert plan.metadata["correction_checksum"]
    assert plan.applied_tensors == (mapping[0].tensor_name,)
    assert plan.metadata["changed_tensor_details"][0]["tensor_name"] == mapping[0].tensor_name
    assert np.all(np.isfinite(plan.updates[mapping[0].tensor_name]))


def test_gpt2_lifting_never_allocates_dense_state_quadratic_for_large_sequence_shape():
    state_dim = 16 * 1600
    projection = np.zeros((state_dim, 1), dtype=np.float32)
    projection[0, 0] = 1.0
    weights = {"transformer.h.0.attn.c_proj.weight": np.eye(1600, dtype=np.float32)}
    plan = build_gpt2_surgery_plan(
        weights,
        _correction(1),
        chart_projection=projection,
        sequence_length=16,
        hidden_size=1600,
        mapping=[GPT2TensorLiftMapping(0, "transformer.h.0.attn.c_proj.weight", side="output")],
        mode="apply",
        max_cross_token_ratio=0.0,
    )
    assert plan.metadata["dense_state_matrix_allocated"] is False
    assert plan.metadata["dense_state_matrix_bytes_estimate"] == state_dim * state_dim * 8
    assert plan.updates["transformer.h.0.attn.c_proj.weight"].shape == (1600, 1600)


class _TinyHFModel:
    def __init__(self, hidden: int = 768):
        import torch

        self.weight = torch.nn.Parameter(torch.eye(hidden))
        self.config = SimpleNamespace(
            model_type="gpt2",
            n_layer=12,
            n_embd=hidden,
            n_head=12,
            n_ctx=1024,
            vocab_size=50257,
            to_dict=lambda: {
                "model_type": "gpt2",
                "n_layer": 12,
                "n_embd": hidden,
                "n_head": 12,
                "n_ctx": 1024,
                "vocab_size": 50257,
            },
        )

    def state_dict(self):
        return {"transformer.h.0.attn.c_proj.weight": self.weight.detach().clone()}

    def load_state_dict(self, values, strict=True):
        import torch

        with torch.no_grad():
            self.weight.copy_(values["transformer.h.0.attn.c_proj.weight"])
        return SimpleNamespace(missing_keys=[], unexpected_keys=[])

    def save_pretrained(self, destination, safe_serialization=True):
        import torch

        destination.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), destination / "pytorch_model.bin")
        (destination / "config.json").write_text(json.dumps(self.config.to_dict()), encoding="utf-8")


def test_gpt2_checkpoint_pair_exports_clean_and_candidate_and_restores_model(tmp_path):
    model = _TinyHFModel()
    projection = np.eye(768, dtype=np.float64)
    plan = build_gpt2_surgery_plan(
        {"transformer.h.0.attn.c_proj.weight": np.eye(768, dtype=np.float32)},
        _correction(768),
        chart_projection=projection,
        sequence_length=1,
        hidden_size=768,
        mapping=[GPT2TensorLiftMapping(0, "transformer.h.0.attn.c_proj.weight", side="output")],
        mode="apply",
    )
    output = export_gpt2_checkpoint_pair(model, tmp_path / "pair", variant="gpt2-small", experimental_plan=plan)
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["baseline"]["status"] == "clean_baseline"
    assert manifest["candidate"]["status"] == "candidate_exported"
    assert manifest["candidate"]["changed"] is True
    assert (output / "baseline" / "config.json").exists()
    assert (output / "candidate" / "pytorch_model.bin").exists()
    assert np.allclose(model.weight.detach().numpy(), np.eye(768))


def test_gpt2_experimental_export_marks_candidate_untrusted_and_lists_changes(tmp_path):
    model = _TinyHFModel(hidden=768)
    projection = np.eye(768, 2, dtype=np.float64)
    mapping = [GPT2TensorLiftMapping(0, "transformer.h.0.attn.c_proj.weight", side="output", block_index=0)]
    plan = build_gpt2_surgery_plan(
        model.state_dict(),
        _correction(2, accepted=False),
        chart_projection=projection,
        sequence_length=1,
        hidden_size=768,
        mapping=mapping,
        mode="experimental",
    )
    output = export_gpt2_checkpoint_pair(
        model,
        tmp_path / "experimental-pair",
        variant="gpt2-small",
        experimental_plan=plan,
        metadata={"experiment": "raw-rejected-node"},
    )
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["baseline"]["status"] == "clean_baseline"
    assert manifest["candidate"]["status"] == "experimental_untrusted"
    assert manifest["candidate"]["changed"] is True
    assert manifest["experimental_plan"]["trust_status"] == "experimental_untrusted"
    assert manifest["experimental_plan"]["acceptance_gate_bypassed"] is True
    assert manifest["experimental_plan"]["rollback_reasons"]["0"] == "confidence below threshold"
    assert manifest["experimental_plan"]["applied_tensors"] == [mapping[0].tensor_name]
    assert manifest["experimental_plan"]["lift_method"] == "first_order_token_local_chart_contraction"
    assert manifest["experimental_plan"]["exact_activation_ls_tensors"] == []
    assert "lift_reports" in manifest["experimental_plan"]
    assert np.allclose(model.weight.detach().numpy(), np.eye(768))


def test_gpt2_surgery_plan_rejects_attn_c_proj_input_side():
    """attn.c_proj input is concatenated attention heads, not residual stream."""
    weights = {"transformer.h.0.attn.c_proj.weight": np.eye(768, dtype=np.float32)}
    projection = np.eye(768, 2, dtype=np.float64)
    mapping = [GPT2TensorLiftMapping(0, "transformer.h.0.attn.c_proj.weight", side="input", block_index=0)]
    plan = build_gpt2_surgery_plan(
        weights,
        _correction(2),
        chart_projection=projection,
        sequence_length=1,
        hidden_size=768,
        mapping=mapping,
        mode="diagnostic",
    )
    assert not plan.applied_tensors
    assert "supports only side='output'" in plan.skipped_tensors["transformer.h.0.attn.c_proj.weight"]


def test_gpt2_depth_gain_weight_and_dual_residual_policy():
    torch = pytest.importorskip("torch")

    # Verify depth schedule weights
    assert gpt2_depth_gain_weight(0, 12, "boost_deep") < 0.40  # micro-gain on layer 0
    assert gpt2_depth_gain_weight(1, 12, "boost_deep") < 0.45  # micro-gain on layer 1
    assert gpt2_depth_gain_weight(2, 12, "boost_deep") < 0.52  # micro-gain on layer 2
    assert gpt2_depth_gain_weight(7, 12, "boost_deep") > 1.25  # powerful boost on layer 7
    assert gpt2_depth_gain_weight(8, 12, "boost_deep") > 1.25  # powerful boost on layer 8
    assert gpt2_depth_gain_weight(0, 12, "flat") == 1.0

    class DualResidualModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.transformer = torch.nn.Module()
            self.transformer.h = torch.nn.ModuleList()
            block = torch.nn.Module()
            block.attn = torch.nn.Module()
            block.attn.c_proj = torch.nn.Linear(3, 3, bias=False)
            block.mlp = torch.nn.Module()
            block.mlp.c_proj = torch.nn.Linear(3, 3, bias=False)
            self.transformer.h.append(block)

        def forward(self, input_ids, position_ids=None, attention_mask=None):
            ids = input_ids.to(dtype=torch.float32)
            hidden = torch.stack((ids, ids * 0.5, ids.square()), dim=-1)
            after_attn = hidden + self.transformer.h[0].attn.c_proj(hidden)
            return after_attn + self.transformer.h[0].mlp.c_proj(after_attn)

    def state_for(ids):
        return np.concatenate([np.asarray([value, value * 0.5, value * value], dtype=np.float64) for value in ids])

    traces = []
    for probe_id, ids in (("train-1", [1, 2, 3]), ("train-2", [4, 5, 6])):
        source = state_for(ids)
        target = source * 1.02
        transition = TransitionObservation(0, 1, 0.0, 1.0, source, target, target - source, target - source)
        traces.append(TrajectoryTrace("gpt2-small", probe_id, (-1, 0), np.asarray([0.0, 1.0]), np.vstack([source, target]), None, (transition,), {"probe_split": "train"}, token_ids=np.asarray(ids), position_ids=np.arange(3)))
    student = FlowOperator(np.asarray([0.0, 1.0]), np.zeros((2, 9, 9)), np.zeros((2, 9)), np.ones(2), np.ones(2), np.zeros(2))
    teacher = FlowOperator(np.asarray([0.0, 1.0]), np.stack([np.eye(9) * 0.1, np.eye(9) * 0.1]), np.zeros((2, 9)), np.ones(2), np.ones(2), np.zeros(2))
    attn_name = "transformer.h.0.attn.c_proj.weight"
    mlp_name = "transformer.h.0.mlp.c_proj.weight"
    mapping = [
        GPT2TensorLiftMapping(0, attn_name, side="output", block_index=0),
        GPT2TensorLiftMapping(0, mlp_name, side="output", block_index=0),
    ]
    inputs, targets, metadata = build_gpt2_teacher_flow_activation_targets(
        DualResidualModel().eval(), traces, student, teacher, mapping,
        sequence_length=3, hidden_size=3, target_effect_ratio=1.0,
        site_policy="dual_residual", dual_residual_ratio=0.35, depth_schedule=None, use_2jet=True,
    )
    assert metadata["site_policy"] == "dual_residual"
    assert metadata["site_contract"] == "gpt2_dual_residual_additive"
    assert metadata["dual_residual_ratio"] == 0.35
    assert attn_name in targets and mlp_name in targets
    # Verification that ratio splits targets proportionally
    attn_norm = np.linalg.norm(targets[attn_name])
    mlp_norm = np.linalg.norm(targets[mlp_name])
    assert np.isclose(attn_norm / mlp_norm, 0.35 / 0.65, rtol=1e-4)


def test_gpt2_least_squares_cache_and_effective_transition_mapping():
    rng = np.random.default_rng(99)
    hidden = 4
    weights = {
        "transformer.h.0.attn.c_proj.weight": np.eye(hidden, dtype=np.float32),
        "transformer.h.11.attn.c_proj.weight": np.eye(hidden, dtype=np.float32),
    }
    # Mapping references transition 11, but correction only has 2 transitions (0 and 1)
    mapping = [
        GPT2TensorLiftMapping(0, "transformer.h.0.attn.c_proj.weight", side="output", block_index=0),
        GPT2TensorLiftMapping(11, "transformer.h.11.attn.c_proj.weight", side="output", block_index=11),
    ]
    correction = _correction(hidden)
    cache = {}
    inputs = {
        "transformer.h.0.attn.c_proj.weight": rng.normal(size=(16, hidden)),
        "transformer.h.11.attn.c_proj.weight": rng.normal(size=(16, hidden)),
    }
    targets = {
        "transformer.h.0.attn.c_proj.weight": rng.normal(scale=0.01, size=(16, hidden)),
        "transformer.h.11.attn.c_proj.weight": rng.normal(scale=0.01, size=(16, hidden)),
    }
    plan1 = build_gpt2_surgery_plan(
        weights, correction, chart_projection=None, sequence_length=4, hidden_size=hidden,
        mapping=mapping, mode="apply", activation_inputs=inputs, activation_target_deltas=targets,
        activation_ridge=1e-5, activation_least_squares_cache=cache,
    )
    assert set(plan1.applied_tensors) == {m.tensor_name for m in mapping}
    # Cache should now be populated with both tensors
    assert len(cache) == 2
    # Calling again with cache reuses computed deltas without re-solving
    plan2 = build_gpt2_surgery_plan(
        weights, correction, chart_projection=None, sequence_length=4, hidden_size=hidden,
        mapping=mapping, mode="apply", activation_inputs=inputs, activation_target_deltas=targets,
        activation_ridge=1e-5, activation_least_squares_cache=cache,
    )
    assert np.allclose(plan1.updates["transformer.h.0.attn.c_proj.weight"], plan2.updates["transformer.h.0.attn.c_proj.weight"])


def test_gpt2_heun_2jet_predictor_corrector():
    torch = pytest.importorskip("torch")

    class KwConv1D(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.eye(3))

        def forward(self, hidden=None):
            return hidden @ self.weight

    class HookModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.transformer = torch.nn.Module()
            self.transformer.h = torch.nn.ModuleList()
            block = torch.nn.Module()
            block.attn = torch.nn.Module()
            block.attn.c_proj = KwConv1D()
            block.mlp = torch.nn.Module()
            block.mlp.c_proj = KwConv1D()
            self.transformer.h.append(block)

        def forward(self, input_ids, position_ids=None, attention_mask=None):
            ids = input_ids.to(dtype=torch.float32)
            positions = torch.arange(ids.shape[1], device=ids.device, dtype=torch.float32).expand_as(ids)
            hidden = torch.stack((ids, positions, ids.square()), dim=-1)
            return self.transformer.h[0].mlp.c_proj(hidden=self.transformer.h[0].attn.c_proj(hidden=hidden))

    def state_for(ids):
        return np.concatenate([np.asarray([value, index, value * value], dtype=np.float64) for index, value in enumerate(ids)])

    traces = []
    for probe_id, ids in (("train-a", [1, 2, 3]), ("train-b", [4, 5, 6])):
        source = state_for(ids)
        target = source * 1.01
        transition = TransitionObservation(0, 1, 0.0, 1.0, source, target, target - source, target - source)
        traces.append(TrajectoryTrace("gpt2-small", probe_id, (-1, 0), np.asarray([0.0, 1.0]), np.vstack([source, target]), None, (transition,), {"probe_split": "train"}, token_ids=np.asarray(ids), position_ids=np.arange(3)))
    student = FlowOperator(np.asarray([0.0, 1.0]), np.zeros((2, 9, 9)), np.zeros((2, 9)), np.ones(2), np.ones(2), np.zeros(2))
    teacher_matrices = np.stack([np.eye(9) * 0.1, np.eye(9) * 0.1])
    teacher = FlowOperator(np.asarray([0.0, 1.0]), teacher_matrices, np.zeros((2, 9)), np.ones(2), np.ones(2), np.ones(2))
    tensor_name = "transformer.h.0.mlp.c_proj.weight"
    mapping = [
        GPT2TensorLiftMapping(0, tensor_name, side="output", block_index=0),
    ]
    # In Heun 2-jet:
    # v1 = 0.1 * x, x_pred = x + 1.0 * v1 = 1.1 * x
    # v2 = 0.1 * x_pred = 0.11 * x
    # v_avg = 0.5 * (0.1 + 0.11) * x = 0.105 * x
    # delta_h = v_avg * 1.0 = 0.105 * x
    inputs, targets, metadata = build_gpt2_teacher_flow_activation_targets(
        HookModel().eval(), traces, student, teacher, mapping,
        sequence_length=3, hidden_size=3, target_effect_ratio=1.0, use_2jet=True,
    )
    assert metadata["use_2jet"] is True
    assert np.allclose(targets[tensor_name], inputs[tensor_name] * 0.105, atol=1e-8)


