"""Tests for Qwen 3.5 speed optimizations: zero-logit extraction, batching, and DirectML resolution."""

from __future__ import annotations

from typing import Any, Sequence
import numpy as np
import pytest
import torch
import torch.nn as nn

from faytuna_flow.connectors import TorchConnector, TorchHooks
from faytuna_flow.transformer_core import run_flow_guided_distillation
from scripts.run_qwen35_transfer import (
    collect_hidden_states,
    evaluate_nll_and_ppl,
    resolve_device,
)


class MockBackboneOutput:
    def __init__(self, hidden_states: Sequence[torch.Tensor], last_hidden_state: torch.Tensor | None = None) -> None:
        self.hidden_states = tuple(hidden_states)
        self.last_hidden_state = last_hidden_state if last_hidden_state is not None else hidden_states[-1]


class MockCausalLMOutput:
    def __init__(
        self,
        logits: torch.Tensor,
        hidden_states: Sequence[torch.Tensor],
        loss: torch.Tensor | None = None,
    ) -> None:
        self.logits = logits
        self.hidden_states = tuple(hidden_states)
        self.loss = loss


class MockBackbone(nn.Module):
    def __init__(self, vocab_size: int = 100, hidden_size: int = 32, num_layers: int = 3) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.layers = nn.ModuleList([
            nn.Linear(hidden_size, hidden_size, bias=False) for _ in range(num_layers)
        ])
        for layer in self.layers:
            nn.init.eye_(layer.weight)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        output_hidden_states: bool = True,
        **kwargs: Any,
    ) -> MockBackboneOutput:
        h = self.embed_tokens(input_ids)
        all_hidden: list[torch.Tensor] = [h]
        for layer in self.layers:
            h = layer(h)
            all_hidden.append(h)
        return MockBackboneOutput(all_hidden)


class MockCausalLM(nn.Module):
    def __init__(self, vocab_size: int = 100, hidden_size: int = 32, num_layers: int = 3) -> None:
        super().__init__()
        self.model = MockBackbone(vocab_size, hidden_size, num_layers)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        self.lm_head_called_count = 0

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        output_hidden_states: bool = True,
        **kwargs: Any,
    ) -> MockCausalLMOutput:
        backbone_out = self.model(
            input_ids,
            attention_mask=attention_mask,
            output_hidden_states=output_hidden_states,
            **kwargs,
        )
        self.lm_head_called_count += 1
        logits = self.lm_head(backbone_out.last_hidden_state)
        loss = None
        labels = kwargs.get("labels", None)
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = torch.nn.functional.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )
        return MockCausalLMOutput(logits, backbone_out.hidden_states, loss=loss)


class MockTokenizer:
    def __init__(self, vocab_size: int = 100) -> None:
        self.vocab_size = vocab_size
        self.pad_token = "<pad>"
        self.eos_token = "<eos>"
        self.pad_token_id = 0
        self.eos_token_id = 1
        self.padding_side = "right"

    def __call__(
        self,
        text: str | Sequence[str],
        return_tensors: str = "pt",
        padding: bool = False,
        truncation: bool = False,
        max_length: int = 64,
    ) -> dict[str, torch.Tensor]:
        if isinstance(text, str):
            words = text.strip().split()
            tokens = [(abs(hash(w)) % (self.vocab_size - 2)) + 2 for w in words]
            if truncation and len(tokens) > max_length:
                tokens = tokens[:max_length]
            return {
                "input_ids": torch.tensor([tokens], dtype=torch.long),
                "attention_mask": torch.ones((1, len(tokens)), dtype=torch.long),
            }

        # List of strings
        seqs: list[list[int]] = []
        for s in text:
            words = s.strip().split()
            toks = [(abs(hash(w)) % (self.vocab_size - 2)) + 2 for w in words]
            if truncation and len(toks) > max_length:
                toks = toks[:max_length]
            seqs.append(toks)

        max_len = max(len(s) for s in seqs) if padding else max(len(s) for s in seqs)
        padded_ids: list[list[int]] = []
        attn_masks: list[list[int]] = []
        for s in seqs:
            pad_len = max_len - len(s)
            padded_ids.append(s + [self.pad_token_id] * pad_len)
            attn_masks.append([1] * len(s) + [0] * pad_len)

        return {
            "input_ids": torch.tensor(padded_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attn_masks, dtype=torch.long),
        }


def test_zero_logit_extraction_bypasses_lm_head() -> None:
    """Verify collect_hidden_states targets backbone directly and never calls lm_head."""
    torch.manual_seed(42)
    model = MockCausalLM(vocab_size=500, hidden_size=64, num_layers=4)
    tokenizer = MockTokenizer(vocab_size=500)
    prompts = [
        "First prompt for testing zero logit speed",
        "Second prompt with different sequence length and words",
    ]

    states = collect_hidden_states(model, tokenizer, prompts, device="cpu", batch_size=2)

    # Crucial assertion: lm_head was NEVER invoked during zero-logit extraction
    assert model.lm_head_called_count == 0, (
        f"Expected lm_head_called_count to be 0, but was called {model.lm_head_called_count} times"
    )
    assert len(states) == 2
    # 1 embedding + 4 layers = 5 hidden states
    assert len(states[0]) == 5
    assert len(states[1]) == 5


def test_numerical_equivalence_zero_logit_vs_full_model() -> None:
    """Verify ||h_opt - h_orig||_inf < 1e-5 between zero-logit backbone and full model."""
    torch.manual_seed(123)
    model = MockCausalLM(vocab_size=200, hidden_size=32, num_layers=3)
    tokenizer = MockTokenizer(vocab_size=200)
    prompt = "Evaluating bit exact equivalence across all layers"

    # 1. Zero-logit extraction via collect_hidden_states
    opt_states = collect_hidden_states(model, tokenizer, [prompt], device="cpu", batch_size=1)[0]

    # 2. Reference extraction via full model forward with lm_head
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        ref_out = model(**inputs, output_hidden_states=True)

    max_inf_norm = 0.0
    for l_idx, opt_h in opt_states.items():
        ref_h = ref_out.hidden_states[l_idx][0].detach().cpu().numpy()
        inf_norm = float(np.max(np.abs(opt_h - ref_h)))
        if inf_norm > max_inf_norm:
            max_inf_norm = inf_norm
        np.testing.assert_allclose(opt_h, ref_h, atol=1e-5, rtol=1e-5)

    assert max_inf_norm < 1e-5, f"Maximum difference {max_inf_norm} exceeds 1e-5 bound"


def test_dynamic_batching_and_unpadding_equivalence() -> None:
    """Verify that batched extraction with attention unpadding matches single-sample extraction."""
    torch.manual_seed(999)
    model = MockCausalLM(vocab_size=200, hidden_size=32, num_layers=3)
    tokenizer = MockTokenizer(vocab_size=200)
    prompts = [
        "Short prompt",
        "This is a medium length test sentence for unpadding",
        "A significantly longer prompt designed to induce substantial dynamic padding zeroes in the batch",
        "Word",
    ]

    # Sequential single-item extraction (batch_size=1)
    single_states = collect_hidden_states(model, tokenizer, prompts, device="cpu", batch_size=1)

    # Batched extraction (batch_size=4)
    batched_states = collect_hidden_states(model, tokenizer, prompts, device="cpu", batch_size=4)

    assert len(single_states) == len(batched_states) == len(prompts)

    max_diff = 0.0
    for p_idx in range(len(prompts)):
        s_dict = single_states[p_idx]
        b_dict = batched_states[p_idx]
        assert set(s_dict.keys()) == set(b_dict.keys())
        for l_idx in s_dict:
            s_arr = s_dict[l_idx]
            b_arr = b_dict[l_idx]
            assert s_arr.shape == b_arr.shape, (
                f"Prompt {p_idx}, Layer {l_idx}: shape mismatch single {s_arr.shape} vs batched {b_arr.shape}"
            )
            diff = float(np.max(np.abs(s_arr - b_arr)))
            if diff > max_diff:
                max_diff = diff
            np.testing.assert_allclose(s_arr, b_arr, atol=1e-5, rtol=1e-5)

    assert max_diff < 1e-5, f"Batched vs single unpadding diff {max_diff} exceeds 1e-5"


def test_resolve_device_directml_and_cpu_fallback() -> None:
    """Verify resolve_device correctly resolves devices and falls back on error."""
    # 1. CPU resolution
    dev, dev_type = resolve_device("cpu")
    assert dev_type == "cpu"
    assert dev == torch.device("cpu")

    # 2. DML resolution (either torch_directml device or graceful cpu fallback)
    dev_dml, dev_dml_type = resolve_device("dml")
    assert dev_dml_type in {"dml", "cpu"}

    # 3. Invalid/unknown device gracefully falls back to CPU
    dev_fallback, dev_fb_type = resolve_device("invalid_device_nonexistent_99")
    assert dev_fb_type == "cpu"
    assert dev_fallback == torch.device("cpu")


def test_torch_connector_dml_device_handling() -> None:
    """Verify TorchConnector supports dml device string with graceful fallback."""
    backbone = MockBackbone(vocab_size=50, hidden_size=8, num_layers=2)
    hooks = TorchHooks(
        state_encoder=lambda probe, dev: torch.zeros((1, 1, 8), device=dev),
        state_decoder=lambda state, probe: torch.zeros((1, 1, 8)),
        state_observer=lambda tensor: tensor.detach().cpu().numpy().reshape(-1),
    )
    # Passing device="dml" must not crash even if DML fails or falls back
    connector = TorchConnector(backbone, hooks, device="dml")
    assert connector.device is not None
    assert connector.capabilities.hidden_states is True


def test_distillation_teacher_zero_logit_extraction() -> None:
    """Verify run_flow_guided_distillation uses teacher backbone directly and never calls teacher lm_head."""
    torch.manual_seed(42)
    student = MockCausalLM(vocab_size=100, hidden_size=16, num_layers=2)
    teacher = MockCausalLM(vocab_size=100, hidden_size=16, num_layers=2)

    token_batches = [
        torch.randint(2, 90, (2, 8), dtype=torch.long),
        torch.randint(2, 90, (2, 8), dtype=torch.long),
    ]

    report = run_flow_guided_distillation(
        student_model=student,
        teacher_model_or_traces=teacher,
        token_batches=token_batches,
        steps=3,
        lr=1e-3,
        lambda_flow=0.5,
        lambda_ce=0.5,
        device="cpu",
    )

    assert report["status"] == "completed"
    # Teacher lm_head was NEVER called during distillation because teacher_backbone was used
    assert teacher.lm_head_called_count == 0, (
        f"Expected teacher lm_head_called_count=0, but got {teacher.lm_head_called_count}"
    )


def test_evaluate_nll_and_ppl_batched() -> None:
    """Verify evaluate_nll_and_ppl works with batched inputs and produces finite perplexity."""
    torch.manual_seed(42)
    model = MockCausalLM(vocab_size=100, hidden_size=16, num_layers=2)
    tokenizer = MockTokenizer(vocab_size=100)
    prompts = [
        "Prompt one for perplexity testing",
        "Prompt two is slightly longer than prompt one",
        "Prompt three is here",
    ]

    res = evaluate_nll_and_ppl(model, tokenizer, prompts, device="cpu", batch_size=2)
    assert "mean_nll" in res
    assert "ppl" in res
    assert np.isfinite(res["mean_nll"])
    assert np.isfinite(res["ppl"])
    assert res["ppl"] > 0.0
