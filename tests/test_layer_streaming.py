"""Tests for PipelinedLayerStreamer: Layer-by-layer VRAM streaming and Gated DeltaNet patching."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence
import numpy as np
import pytest
import torch
import torch.nn as nn

from faytuna_flow.layer_streaming import (
    PipelinedLayerStreamer,
    StreamOutput,
    apply_gated_deltanet_patch,
    patched_chunk_gated_delta_rule,
    resolve_execution_device,
    stream_hidden_states,
)


class MockTransformerBlock(nn.Module):
    """Simple non-trivial transformer block for streaming verification."""

    def __init__(self, hidden_size: int = 32) -> None:
        super().__init__()
        self.linear1 = nn.Linear(hidden_size, hidden_size * 2, bias=False)
        self.linear2 = nn.Linear(hidden_size * 2, hidden_size, bias=False)
        self.norm = nn.LayerNorm(hidden_size)

        # Initialize with deterministic weights
        nn.init.orthogonal_(self.linear1.weight)
        nn.init.orthogonal_(self.linear2.weight)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        residual = hidden_states
        h = self.norm(hidden_states)
        h = torch.relu(self.linear1(h))
        h = self.linear2(h)
        return residual + h


class MockSyntheticTransformer(nn.Module):
    """Multi-layer synthetic transformer backbone compatible with PipelinedLayerStreamer."""

    def __init__(self, vocab_size: int = 200, hidden_size: int = 32, num_layers: int = 4) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.layers = nn.ModuleList([MockTransformerBlock(hidden_size) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(hidden_size)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        output_hidden_states: bool = True,
        **kwargs: Any,
    ) -> Any:
        h = self.embed_tokens(input_ids)
        all_hidden: list[torch.Tensor] = [h]
        for layer in self.layers:
            h = layer(h, attention_mask=attention_mask, **kwargs)
            all_hidden.append(h)
        h_final = self.norm(h)
        all_hidden[-1] = h_final  # Mirror HuggingFace convention where last hidden state is post-norm

        class Output:
            pass

        out = Output()
        out.hidden_states = tuple(all_hidden)
        out.last_hidden_state = h_final
        return out


def test_resolve_execution_device() -> None:
    """Verify device resolution behaves predictably across device strings and fallbacks."""
    dev_cpu, dev_type = resolve_execution_device("cpu")
    assert dev_type == "cpu"
    assert dev_cpu == torch.device("cpu")

    dev_auto, dev_auto_type = resolve_execution_device("auto")
    assert dev_auto_type in {"cuda", "dml", "cpu"}

    dev_fallback, dev_fb_type = resolve_execution_device("non_existent_accelerator_999")
    assert dev_fb_type == "cpu"
    assert dev_fallback == torch.device("cpu")


def test_synthetic_transformer_layer_streaming_equivalence() -> None:
    """Verify ||H_stream - H_monolithic||_inf < 1e-5 on a multi-layer synthetic transformer."""
    torch.manual_seed(42)
    model = MockSyntheticTransformer(vocab_size=200, hidden_size=32, num_layers=4)
    model.eval()

    streamer = PipelinedLayerStreamer(model, device="cpu")
    input_ids = torch.randint(0, 199, (3, 16), dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)

    # 1. Standard monolithic forward pass
    with torch.no_grad():
        ref_out = model(input_ids, attention_mask=attention_mask, output_hidden_states=True)

    # 2. Pipelined layer-by-layer forward pass
    stream_out = streamer.stream_forward(input_ids, attention_mask=attention_mask, device="cpu")

    assert len(stream_out) == 3  # 3 prompts in batch

    # Check numerical equivalence across all 5 hidden states (1 embed + 4 layers)
    max_diff = 0.0
    for b_idx in range(3):
        prompt_states = stream_out[b_idx]
        assert len(prompt_states) == 5
        for l_idx in range(5):
            ref_h = ref_out.hidden_states[l_idx][b_idx].detach().cpu().numpy()
            stream_h = prompt_states[l_idx]
            assert ref_h.shape == stream_h.shape
            diff = float(np.max(np.abs(stream_h - ref_h)))
            if diff > max_diff:
                max_diff = diff
            np.testing.assert_allclose(stream_h, ref_h, atol=1e-5, rtol=1e-5)

    assert max_diff < 1e-5, f"Synthetic streamer max diff {max_diff} exceeded 1e-5 tolerance"


def test_dynamic_batching_and_unpadding() -> None:
    """Verify batched variable-length sequences are unpadded properly to match single runs."""
    torch.manual_seed(101)
    model = MockSyntheticTransformer(vocab_size=200, hidden_size=32, num_layers=3)
    streamer = PipelinedLayerStreamer(model, device="cpu")

    # Three sequences of differing lengths: 5, 8, 12 tokens
    seqs = [
        torch.randint(1, 199, (1, 5), dtype=torch.long),
        torch.randint(1, 199, (1, 8), dtype=torch.long),
        torch.randint(1, 199, (1, 12), dtype=torch.long),
    ]

    # Collect single-sample hidden states
    single_states = []
    for s in seqs:
        out = streamer.stream_forward(s, device="cpu")
        single_states.append(out[0])

    # Pad sequences to batch of shape (3, 12)
    padded = torch.zeros((3, 12), dtype=torch.long)
    mask = torch.zeros((3, 12), dtype=torch.long)
    for idx, s in enumerate(seqs):
        length = s.shape[1]
        padded[idx, :length] = s[0]
        mask[idx, :length] = 1

    # Collect batched hidden states
    batched_states = streamer.stream_forward(padded, attention_mask=mask, device="cpu")

    assert len(single_states) == len(batched_states) == 3

    for p_idx in range(3):
        expected_len = seqs[p_idx].shape[1]
        s_dict = single_states[p_idx]
        b_dict = batched_states[p_idx]
        for l_idx in s_dict:
            s_arr = s_dict[l_idx]
            b_arr = b_dict[l_idx]
            assert s_arr.shape[0] == expected_len
            assert b_arr.shape[0] == expected_len
            np.testing.assert_allclose(s_arr, b_arr, atol=1e-5, rtol=1e-5)


def test_target_layers_filtering() -> None:
    """Verify that target_layers parameter correctly restricts collected hidden states."""
    torch.manual_seed(77)
    model = MockSyntheticTransformer(vocab_size=100, hidden_size=16, num_layers=4)
    streamer = PipelinedLayerStreamer(model, device="cpu")

    input_ids = torch.randint(0, 99, (2, 8), dtype=torch.long)
    target_layers = [0, 2, 4]

    out = streamer.stream_forward(input_ids, target_layers=target_layers, device="cpu")
    assert len(out) == 2
    for p_states in out:
        assert set(p_states.keys()) == {0, 2, 4}


def test_stream_output_indexing_semantics() -> None:
    """Verify StreamOutput behaves cleanly both as list of dicts and dict for single prompts."""
    dummy = [{0: np.array([1.0]), 1: np.array([2.0])}]
    out = StreamOutput(dummy)

    # 1. List behavior
    assert len(out) == 1
    assert out[0][1].item() == 2.0

    # 2. Dict-like behavior for single prompt
    assert out.get(0).item() == 1.0
    assert out.get(1).item() == 2.0
    assert out.get(99) is None


def test_gated_deltanet_patch_mathematical_equivalence() -> None:
    """Verify patched_chunk_gated_delta_rule matches original rule with machine epsilon accuracy."""
    try:
        from transformers.models.qwen3_5.modeling_qwen3_5 import torch_chunk_gated_delta_rule as orig_rule
    except ImportError:
        pytest.skip("transformers Qwen 3.5 module not installed in current environment")

    B, S, H, Dk, Dv = 2, 32, 16, 128, 128
    torch.manual_seed(42)

    # Generate realistic DeltaNet inputs: negative continuous decay g, sigmoid beta, scaled q/k/v
    q = torch.randn(B, S, H, Dk) * 0.1
    k = torch.randn(B, S, H, Dk) * 0.1
    v = torch.randn(B, S, H, Dv) * 0.1
    g = -torch.nn.functional.softplus(torch.randn(B, S, H))
    beta = torch.sigmoid(torch.randn(B, S, H))

    out_orig, _ = orig_rule(q, k, v, g, beta, use_qk_l2norm_in_kernel=True)
    out_patched, _ = patched_chunk_gated_delta_rule(q, k, v, g, beta, use_qk_l2norm_in_kernel=True)

    max_diff = float(torch.max(torch.abs(out_orig - out_patched)).item())
    assert max_diff < 1e-6, f"Patched rule diff {max_diff} exceeds 1e-6 tolerance on CPU"


def test_qwen35_08b_full_layer_streaming_accuracy() -> None:
    """Full empirical verification on Qwen3.5-0.8B-Base: ||H_stream - H_monolithic||_inf < 1e-4."""
    model_path = Path(r"C:\models\Qwen3.5-0.8B-Base")
    if not model_path.exists():
        pytest.skip(f"Qwen3.5-0.8B-Base not found at {model_path}")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    test_prompts = [
        "The fundamental theorem of algebra states that every polynomial",
        "In deep learning, residual connections allow gradients to propagate",
    ]

    # Initialize streamer with auto device (uses DirectML if available, else CPU)
    streamer = PipelinedLayerStreamer(str(model_path), device="auto")
    stream_states = streamer.collect_hidden_states(test_prompts, batch_size=2)

    # Reference monolithic extraction using existing model instance in FP32
    inputs = tokenizer(test_prompts, return_tensors="pt", padding=True, truncation=True, max_length=64)
    with torch.no_grad():
        ref_out = streamer.model(**inputs, output_hidden_states=True)

    unpadded_lens = inputs["attention_mask"].sum(dim=1).tolist()
    assert len(stream_states) == len(test_prompts) == 2
    num_states = len(ref_out.hidden_states)

    global_max_diff = 0.0
    for p_idx in range(len(test_prompts)):
        p_len = int(unpadded_lens[p_idx])
        for l_idx in range(num_states):
            ref_h = ref_out.hidden_states[l_idx][p_idx, :p_len].detach().to(torch.float32).cpu().numpy()
            str_h = stream_states[p_idx][l_idx]
            assert ref_h.shape == str_h.shape, f"Shape mismatch at prompt {p_idx}, layer {l_idx}"
            diff = float(np.max(np.abs(str_h - ref_h)))
            if diff > global_max_diff:
                global_max_diff = diff

    print(f"\n[Test Result] Qwen3.5-0.8B Max Streaming Difference across all {num_states} states: {global_max_diff:.8e}")
    assert global_max_diff < 1e-4, f"Maximum difference {global_max_diff} exceeds required 1e-4 bound"


def test_convenience_stream_hidden_states_helper() -> None:
    """Verify stream_hidden_states helper runs and outputs valid formatted hidden states."""
    torch.manual_seed(99)
    model = MockSyntheticTransformer(vocab_size=100, hidden_size=16, num_layers=2)

    class DummyTokenizer:
        padding_side = "right"

        def __call__(self, text: list[str], **kwargs: Any) -> dict[str, torch.Tensor]:
            ids = torch.randint(1, 99, (len(text), 6), dtype=torch.long)
            mask = torch.ones_like(ids)
            return {"input_ids": ids, "attention_mask": mask}

    prompts = ["Sample prompt alpha", "Sample prompt beta"]
    tok = DummyTokenizer()

    states = stream_hidden_states(model, tok, prompts, device="cpu", batch_size=2)
    assert len(states) == 2
    assert len(states[0]) == 3  # 1 embed + 2 layers


def test_layer_outer_multi_batch_equivalence() -> None:
    """Verify that multi-batch Layer-Outer streaming matches single-batch streaming bit-for-bit."""
    torch.manual_seed(1234)
    model = MockSyntheticTransformer(vocab_size=100, hidden_size=16, num_layers=3)
    streamer = PipelinedLayerStreamer(model, device="cpu")

    class DummyTokenizer:
        padding_side = "right"

        def __call__(self, text: list[str], **kwargs: Any) -> dict[str, torch.Tensor]:
            # Deterministic token IDs based on string length
            ids = torch.stack([torch.tensor([(len(t) * 7 + j) % 95 + 1 for j in range(8)], dtype=torch.long) for t in text])
            mask = torch.ones_like(ids)
            return {"input_ids": ids, "attention_mask": mask}

    prompts = ["prompt_alpha", "prompt_beta", "prompt_gamma", "prompt_delta", "prompt_epsilon"]
    streamer.tokenizer = DummyTokenizer()

    # Reference: single batch run (all 5 prompts at once, batch_size=5)
    ref_states = streamer.collect_hidden_states(prompts, batch_size=5, device="cpu")

    # Layer-Outer run with batch_size=2 (triggers 3 batches: 2, 2, 1)
    chunked_states = streamer.collect_hidden_states(prompts, batch_size=2, device="cpu")

    assert len(ref_states) == len(chunked_states) == 5
    for p_idx in range(5):
        assert set(ref_states[p_idx].keys()) == set(chunked_states[p_idx].keys())
        for l_idx in ref_states[p_idx]:
            np.testing.assert_allclose(
                chunked_states[p_idx][l_idx],
                ref_states[p_idx][l_idx],
                atol=1e-6,
                rtol=1e-6,
            )

