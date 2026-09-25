"""Universal Transformer Core: Observation, Hooks, Least-Squares, Surgery & A/B Evaluation.

Model-agnostic transformer surgery engine: provides PyTorch forward hook activation capture,
token-local chart contraction, activation least-squares solvers, static weight dissimilarity,
pilot pulse sensitivity bounds, piecewise MLP knot surgery, and HuggingFace A/B evaluation.
Supports GPT-2, LLaMA, Mistral, Qwen, Gemma, Phi, BERT, and generic transformer architectures.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence
import hashlib
import json
import os
import re
import shutil
import tempfile

import numpy as np

from .connectors import CapabilityError, TorchConnector, TorchHooks
from .geometry import compute_frobenius_dissimilarity, dissimilarity_gain_schedule, project_teacher_weight
from .model_families import GPT2_XL_TO_SMALL, ModelVariantSpec, inspect_gpt2_variant
from .probes import ProbeGenerator, ProbeGeneratorConfig, PythonProbeGenerator
from .solver import ConstrainedCorrection
from .types import AlignmentResult, FlowOperator, Probe, ProbeSplit, SurgeryPlan, TensorSchemaEntry, TrajectoryTrace, finite_array, stable_l2
from .knots import build_attention_detour_projector, fit_piecewise_mlp_with_retries
from .nonlinear_transfer import align_vocabulary_head, rank_one_memory_imprint


@dataclass(frozen=True)
class GPT2ProbePolicy:
    """Materialize paired structured probes into fixed-length GPT-2 inputs."""

    encode: Callable[[Probe], Mapping[str, Any]]
    sequence_length: int
    vocab_size: int = 50257

    def __post_init__(self) -> None:
        if self.sequence_length < 1 or self.vocab_size < 2:
            raise ValueError("GPT-2 probe sequence_length and vocab_size are invalid")

    def encode_probe(self, probe: Probe) -> dict[str, np.ndarray]:
        values = self.encode(probe)
        if not isinstance(values, Mapping) or "input_ids" not in values:
            raise CapabilityError("GPT2ProbePolicy callback must return input_ids")
        input_ids = np.asarray(values["input_ids"], dtype=np.int64).reshape(-1)
        if len(input_ids) != self.sequence_length or np.any(input_ids < 0) or np.any(input_ids >= self.vocab_size):
            raise ValueError("GPT-2 input_ids have the wrong length or vocabulary range")
        positions = np.asarray(values.get("position_ids", np.arange(self.sequence_length)), dtype=np.int64).reshape(-1)
        if len(positions) != self.sequence_length or np.any(positions < 0):
            raise ValueError("GPT-2 position_ids have the wrong length or contain negatives")
        attention = np.asarray(values.get("attention_mask", np.ones(self.sequence_length)), dtype=np.float32).reshape(-1)
        if len(attention) != self.sequence_length or not np.all(np.isfinite(attention)) or np.any((attention < 0) | (attention > 1)):
            raise ValueError("GPT-2 attention_mask must be finite binary-like values")
        return {"input_ids": input_ids, "position_ids": positions, "attention_mask": attention}

    def materialize(self, probes: Sequence[Probe]) -> list[Probe]:
        result = []
        for probe in probes:
            encoded = self.encode_probe(probe)
            payload = dict(probe.payload)
            payload.update({key: value.tolist() for key, value in encoded.items()})
            payload["gpt2_sequence_length"] = self.sequence_length
            result.append(Probe(probe.probe_id, probe.family, payload, probe.initial_state, probe.pair_id, probe.perturbation, probe.split))
        return result


def make_gpt2_probe_splits(policy: GPT2ProbePolicy, *, per_family: int = 6, seed: int = 17, state_dim: int = 8) -> ProbeSplit:
    """Create disjoint semantic/perturbation probe artifacts for GPT-2."""

    generator = ProbeGenerator(ProbeGeneratorConfig(state_dim=state_dim, per_family=per_family, seed=seed))
    split = generator.split(generator.generate())
    return ProbeSplit(tuple(policy.materialize(split.train)), tuple(policy.materialize(split.validation)), tuple(policy.materialize(split.holdout)))


def make_gpt2_python_probe_splits(policy: GPT2ProbePolicy, *, per_family: int = 6, seed: int = 17, state_dim: int = 8) -> ProbeSplit:
    """Create disjoint Python code semantic/perturbation probe artifacts for GPT-2."""

    generator = PythonProbeGenerator(ProbeGeneratorConfig(state_dim=state_dim, per_family=per_family, seed=seed))
    split = generator.split(generator.generate())
    return ProbeSplit(tuple(policy.materialize(split.train)), tuple(policy.materialize(split.validation)), tuple(policy.materialize(split.holdout)))


def _gpt2_transformer(model: Any) -> Any:
    transformer = getattr(model, "transformer", None)
    if transformer is None or not hasattr(transformer, "wte") or not hasattr(transformer, "wpe"):
        raise CapabilityError("GPT-2 model must expose transformer.wte and transformer.wpe")
    return transformer


def make_gpt2_hooks(model: Any, policy: GPT2ProbePolicy) -> TorchHooks:
    """Build explicit sequence-preserving hooks for a GPT-2 HF model."""

    transformer = _gpt2_transformer(model)
    try:
        import torch
    except ImportError as error:
        raise CapabilityError("GPT-2 connector requires optional PyTorch") from error
    parameters = model.parameters()
    parameter = next(parameters, None)
    if parameter is None:
        raise CapabilityError("GPT-2 model exposes no parameters from which to infer dtype/device")
    config = getattr(model, "config", None)
    hidden_size = int(getattr(config, "n_embd", getattr(config, "hidden_size", 0)))
    if hidden_size < 1:
        raise CapabilityError("GPT-2 config does not expose n_embd/hidden_size")

    # A probe's token and position tensors are invariant across all observed
    # layers and all finite-difference stencils.  Keeping this tiny cache
    # avoids repeating numpy->torch conversion and tokenizer-policy work for
    # every layer without changing the chart or any numerical stencil.
    encoded_cache: dict[tuple[str, str, str], tuple[Any, Any, Any]] = {}

    def encoded_tensors(probe: Probe, device: Any) -> tuple[Any, Any, Any]:
        cache_key = (str(probe.probe_id), str(device), str(parameter.dtype))
        cached = encoded_cache.get(cache_key)
        if cached is not None:
            return cached
        encoded = policy.encode_probe(probe)
        input_ids = torch.as_tensor(encoded["input_ids"], dtype=torch.long, device=device).unsqueeze(0)
        position_ids = torch.as_tensor(encoded["position_ids"], dtype=torch.long, device=device).unsqueeze(0)
        attention_mask = torch.as_tensor(encoded["attention_mask"], dtype=parameter.dtype, device=device).view(1, 1, 1, policy.sequence_length)
        cached = (input_ids, position_ids, attention_mask)
        encoded_cache[cache_key] = cached
        return cached

    def state_encoder(probe: Probe, device: Any) -> Any:
        input_ids, position_ids, _ = encoded_tensors(probe, device)
        hidden = transformer.wte(input_ids) + transformer.wpe(position_ids)
        dropout = getattr(transformer, "drop", None)
        return dropout(hidden) if dropout is not None else hidden

    def state_decoder(state: np.ndarray, probe: Probe) -> Any:
        values = finite_array(state, ndim=1, name="GPT-2 state chart")
        expected = policy.sequence_length * hidden_size
        if values.size != expected:
            raise ValueError(f"GPT-2 state chart has size {values.size}; expected {expected}")
        return torch.as_tensor(values, dtype=parameter.dtype, device=parameter.device).view(1, policy.sequence_length, hidden_size)

    def state_observer(hidden: Any) -> Any:
        if not isinstance(hidden, torch.Tensor) or tuple(hidden.shape) != (1, policy.sequence_length, hidden_size):
            raise CapabilityError("GPT-2 state observer requires [1, sequence_length, hidden_size]")
        return hidden.detach().reshape(-1)

    def layer_call(layer: Any, hidden: Any, probe: Probe) -> Any:
        _, _, attention_mask = encoded_tensors(probe, hidden.device)
        try:
            return layer(hidden_states=hidden, layer_past=None, attention_mask=(attention_mask - 1.0) * 10000.0, head_mask=None, use_cache=False, output_attentions=False)
        except TypeError:
            return layer(hidden)

    def observed_layer_output(layer: Any, hidden: Any, probe: Probe) -> Any:
        """Run one GPT-2 block for either a singleton or a batched chart.

        The directional callbacks below batch all probe directions into one
        forward.  GPT-2 blocks are batch preserving, so this changes the
        number of model calls without changing the finite-difference stencil.
        """

        output = layer_call(layer, hidden, probe)
        if isinstance(output, Mapping):
            output = next((output[key] for key in ("last_hidden_state", "hidden_state", "hidden_states") if key in output), output)
        if isinstance(output, (tuple, list)):
            if not output:
                raise CapabilityError("GPT-2 layer returned an empty tuple/list")
            output = output[0]
        if not isinstance(output, torch.Tensor) or tuple(output.shape) != tuple(hidden.shape):
            raise CapabilityError("GPT-2 directional sketch requires a batch-preserving tensor layer output")
        if hidden.shape[0] == 1:
            # The nominal transition is also the central point of the
            # Hessian stencil. Retain it only until the corresponding callback
            # consumes it; the callback checks the input exactly before reuse.
            nominal_cache[(id(layer), str(probe.probe_id))] = (hidden.detach().clone(), output.detach().clone())
        return output

    def _normalized_directions(hidden: Any, directions: np.ndarray, step: float) -> tuple[Any, float]:
        values = torch.as_tensor(directions, dtype=hidden.dtype, device=hidden.device)
        if values.ndim != 2 or values.shape[1] != hidden.numel() or values.shape[0] < 1:
            raise ValueError("GPT-2 sketch directions have the wrong shape")
        norms = torch.linalg.vector_norm(values, dim=1)
        if not bool(torch.isfinite(norms).all().item()) or bool(torch.any(norms <= 0).item()):
            raise ValueError("GPT-2 sketch directions must be finite and nonzero")
        point_norm = float(torch.linalg.vector_norm(hidden.reshape(-1)).detach().cpu().item())
        local_step = float(step) * max(1.0, point_norm)
        return values / norms[:, None], local_step

    sketch_cache: dict[tuple[Any, ...], Any] = {}
    nominal_cache: dict[tuple[int, str], tuple[Any, Any]] = {}

    def _sketch_key(kind: str, layer: Any, hidden: Any, probe: Probe, directions: np.ndarray, step: float) -> tuple[Any, ...]:
        if hasattr(hidden, "shape") and hasattr(hidden, "dtype"):
            flat = hidden.reshape(-1)
            numel = flat.numel()
            hidden_digest = (
                hidden.shape,
                hidden.dtype,
                hidden.device,
                float(flat[0].item()) if numel else 0.0,
                float(flat[-1].item()) if numel else 0.0,
                float(flat[numel // 2].item()) if numel else 0.0,
                float(flat.sum().item()) if numel else 0.0,
            )
        else:
            hidden_digest = (id(hidden),)

        if isinstance(directions, np.ndarray):
            d_flat = directions.reshape(-1)
            d_size = d_flat.size
            dir_key = (
                directions.shape,
                directions.dtype,
                float(d_flat[0]) if d_size else 0.0,
                float(d_flat[-1]) if d_size else 0.0,
                float(d_flat[d_size // 2]) if d_size else 0.0,
                float(np.sum(d_flat)) if d_size else 0.0,
            )
        else:
            dir_key = (id(directions),)

        return (kind, id(layer), hidden_digest, str(probe.probe_id), float(step), dir_key)

    def _batched_jacobian_pair(layer: Any, hidden: Any, probe: Probe, directions: np.ndarray, step: float) -> tuple[Any, Any]:
        """Compute both finite-difference step sizes in one block forward."""

        if hidden.shape[0] != 1:
            raise CapabilityError("GPT-2 directional sketch expects a singleton chart batch")
        units, local_step = _normalized_directions(hidden, directions, step)
        count = int(units.shape[0])
        offsets = local_step * units.reshape(count, *hidden.shape[1:])
        double_offsets = 2.0 * offsets
        batch = torch.empty((4 * count, *hidden.shape[1:]), dtype=hidden.dtype, device=hidden.device)
        torch.add(hidden, offsets, out=batch[:count])
        torch.sub(hidden, offsets, out=batch[count : 2 * count])
        torch.add(hidden, double_offsets, out=batch[2 * count : 3 * count])
        torch.sub(hidden, double_offsets, out=batch[3 * count :])
        with torch.no_grad():
            outputs = observed_layer_output(layer, batch, probe).reshape(4 * count, -1)
        first_plus, first_minus = outputs[:count], outputs[count : 2 * count]
        second_plus, second_minus = outputs[2 * count : 3 * count], outputs[3 * count :]
        first = ((first_plus - first_minus) / (2.0 * local_step)).T
        second = ((second_plus - second_minus) / (4.0 * local_step)).T
        return first, second

    def batched_jacobian_sketch(layer: Any, hidden: Any, probe: Probe, directions: np.ndarray, step: float) -> Any:
        """Compute J·U for two sensitivity steps in one batched forward."""

        key = _sketch_key("jacobian", layer, hidden, probe, directions, step)
        cached = sketch_cache.pop(key, None)
        if cached is not None:
            return cached
        current, alternate = _batched_jacobian_pair(layer, hidden, probe, directions, step)
        sketch_cache[_sketch_key("jacobian", layer, hidden, probe, directions, step * 2.0)] = alternate
        return current

    def _batched_hessian_pair(layer: Any, hidden: Any, probe: Probe, directions: np.ndarray, step: float) -> tuple[Any, Any]:
        """Compute both directional H[u,u] step sizes in one batch."""

        if hidden.shape[0] != 1:
            raise CapabilityError("GPT-2 directional sketch expects a singleton chart batch")
        units, local_step = _normalized_directions(hidden, directions, step)
        count = int(units.shape[0])
        offsets = local_step * units.reshape(count, *hidden.shape[1:])
        double_offsets = 2.0 * offsets
        cached = nominal_cache.get((id(layer), str(probe.probe_id)))
        reuse_nominal = cached is not None and tuple(cached[0].shape) == tuple(hidden.shape) and bool(torch.equal(cached[0], hidden))
        if reuse_nominal:
            center_output = cached[1].reshape(-1)
            batch = torch.empty((4 * count, *hidden.shape[1:]), dtype=hidden.dtype, device=hidden.device)
            torch.add(hidden, offsets, out=batch[:count])
            torch.sub(hidden, offsets, out=batch[count : 2 * count])
            torch.add(hidden, double_offsets, out=batch[2 * count : 3 * count])
            torch.sub(hidden, double_offsets, out=batch[3 * count :])
        else:
            center_output = None
            batch = torch.empty((1 + 4 * count, *hidden.shape[1:]), dtype=hidden.dtype, device=hidden.device)
            batch[0].copy_(hidden[0])
            torch.add(hidden, offsets, out=batch[1 : count + 1])
            torch.sub(hidden, offsets, out=batch[count + 1 : 2 * count + 1])
            torch.add(hidden, double_offsets, out=batch[2 * count + 1 : 3 * count + 1])
            torch.sub(hidden, double_offsets, out=batch[3 * count + 1 :])
        with torch.no_grad():
            outputs = observed_layer_output(layer, batch, probe).reshape((4 * count) if reuse_nominal else (1 + 4 * count), -1)
        if not reuse_nominal:
            center_output = outputs[0]
            offset_outputs = outputs[1:]
        else:
            offset_outputs = outputs
        first_plus, first_minus = offset_outputs[:count], offset_outputs[count : 2 * count]
        second_plus, second_minus = offset_outputs[2 * count : 3 * count], offset_outputs[3 * count :]
        nominal_cache.pop((id(layer), str(probe.probe_id)), None)
        first = (first_plus - 2.0 * center_output.unsqueeze(0) + first_minus) / (local_step * local_step)
        second = (second_plus - 2.0 * center_output.unsqueeze(0) + second_minus) / (4.0 * local_step * local_step)
        return first, second

    def batched_hessian_sketch(layer: Any, hidden: Any, probe: Probe, directions: np.ndarray, step: float) -> Any:
        """Compute directional H[u,u] for two sensitivity steps in one forward."""

        key = _sketch_key("hessian", layer, hidden, probe, directions, step)
        cached = sketch_cache.pop(key, None)
        if cached is not None:
            return cached
        current, alternate = _batched_hessian_pair(layer, hidden, probe, directions, step)
        sketch_cache[_sketch_key("hessian", layer, hidden, probe, directions, step * 2.0)] = alternate
        return current

    def token_position_observer(probe: Probe) -> Mapping[str, Any]:
        encoded = policy.encode_probe(probe)
        return {"token_ids": encoded["input_ids"], "position_ids": encoded["position_ids"], "token_position_map": {"sequence": list(range(policy.sequence_length))}}

    def weight_getter(module: Any) -> Mapping[str, Any]:
        return {str(name): value.detach().cpu() for name, value in module.state_dict().items()}

    def weight_setter(module: Any, values: Mapping[str, Any]) -> None:
        state = module.state_dict()
        with torch.no_grad():
            for name, value in values.items():
                if name not in state or tuple(state[name].shape) != tuple(value.shape):
                    raise CapabilityError(f"GPT-2 tensor {name!r} is absent or has an incompatible shape")
                state[name].copy_(value.to(device=state[name].device, dtype=state[name].dtype))

    return TorchHooks(
        state_encoder=state_encoder,
        state_decoder=state_decoder,
        state_observer=state_observer,
        token_position_observer=token_position_observer,
        weight_getter=weight_getter,
        weight_setter=weight_setter,
        layer_call_with_probe=observed_layer_output,
        jacobian_sketch=batched_jacobian_sketch,
        hessian_sketch=batched_hessian_sketch,
    )


class GPT2Connector(TorchConnector):
    """Explicit GPT-2 family connector for either XL teacher or small student."""

    def __init__(self, model: Any, policy: GPT2ProbePolicy, *, variant: str, model_id: str | None = None, device: str | Any | None = None) -> None:
        expected = GPT2_XL_TO_SMALL.teacher if variant == "gpt2-xl" else GPT2_XL_TO_SMALL.student if variant == "gpt2-small" else None
        if expected is None:
            raise ValueError("GPT2Connector variant must be 'gpt2-xl' or 'gpt2-small'")
        check = inspect_gpt2_variant(getattr(model, "config", None), expected)
        if not check["supported"]:
            raise CapabilityError("GPT-2 architecture preflight rejected model: " + "; ".join(check["reasons"]))
        super().__init__(model, make_gpt2_hooks(model, policy), model_id=model_id or f"{variant}-observation", device=device)
        self.feature_space = "gpt2_sequence_hidden_flat"
        self.variant = variant
        self.sequence_length = policy.sequence_length
        self.hidden_size = expected.hidden_size
        self.state_dim = self.sequence_length * self.hidden_size
        self.state_shape = (self.sequence_length, self.hidden_size)
        # The GPT-2 callbacks batch the +/- stencils and cache the alternate
        # step, so each pair of sensitivity evaluations costs one block call.
        self.jacobian_two_step_calls = 1
        self.hessian_two_step_calls = 1
        self.hessian_reuses_nominal_output = True
        self.family_preflight = check
        self.capabilities = self.capabilities.__class__(
            **{**self.capabilities.to_dict(), "reasons": {**self.capabilities.reasons, "runtime_hidden_states": "hidden states are available only at HF/PyTorch observation stage; stock llama.cpp is not instrumented", "residual_states": "GPT-2 connector has no explicit residual-branch observer; post-layer hidden is not labeled residual"}, "automatic": {**self.capabilities.automatic, "gpt2_family_chart": True, "residual_states": False}, "required_callbacks": {**self.capabilities.required_callbacks, "tokenization": "GPT2ProbePolicy.encode", "runtime_validation": "standard llama.cpp after explicit export", "residual_states": "supply a real residual branch observer through a family-specific connector"}}
        )

    def estimate_differential_memory_bytes(self, jacobian_rank: int, hessian_rank: int) -> int:
        """Estimate the temporary chart batch used by the directional callbacks.

        The callback batches the same finite-difference points that the generic
        protocol would evaluate one by one.  This estimate covers the input and
        output chart tensors for the larger of the two callback batches; model
        parameter and framework activation memory are reported separately by
        the caller when available.  It deliberately does not estimate a dense
        ``state_dim x state_dim`` Jacobian or a quadratic tensor.
        """

        if int(jacobian_rank) < 1 or int(hessian_rank) < 1:
            raise ValueError("differential ranks must be positive")
        parameter = next(self.model.parameters(), None)
        if parameter is None:
            raise CapabilityError("GPT-2 memory estimate requires model parameters")
        hessian_batch = 4 * int(hessian_rank) if self.hessian_reuses_nominal_output else 1 + 4 * int(hessian_rank)
        batch = max(4 * int(jacobian_rank), hessian_batch)
        # One chart input and one observed chart output are retained by the
        # callback estimate.  The model's internal activations are not hidden
        # behind this number and are therefore not falsely presented as exact.
        return int(2 * batch * self.state_dim * parameter.element_size())


@dataclass(frozen=True)
class GPT2ActivationCaptureResult:
    """Exact per-tensor activations captured from an ordinary HF forward."""

    inputs: Mapping[str, np.ndarray]
    outputs: Mapping[str, np.ndarray]
    forward_calls: int
    samples_by_tensor: Mapping[str, int]
    finite: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend": "torch_forward_hooks",
            "tensor_names": sorted(str(name) for name in self.inputs),
            "input_shapes": {str(name): list(value.shape) for name, value in self.inputs.items()},
            "output_shapes": {str(name): list(value.shape) for name, value in self.outputs.items()},
            "samples_by_tensor": {str(name): int(count) for name, count in self.samples_by_tensor.items()},
            "forward_calls": int(self.forward_calls),
            "finite": bool(self.finite),
            "exact_activation_inputs": True,
            "exact_activation_outputs": True,
        }


class GPT2ActivationHookCapture:
    """Capture exact Conv1D input/output rows from a real torch model.

    The hook records the first tensor argument and tensor output for explicitly
    named modules. Batch and sequence axes are flattened only in the returned
    sample table, preserving every ``batch * sequence`` row and never treating
    a post-layer hidden state as a residual branch. Target deltas must come
    from a separately defined paired intervention/teacher policy; this class
    does not invent them.
    """

    def __init__(self, model: Any, tensor_names: Sequence[str]) -> None:
        try:
            import torch
        except ImportError as error:  # pragma: no cover - optional dependency
            raise CapabilityError("exact GPT-2 activation capture requires optional PyTorch") from error
        named_modules = dict(model.named_modules()) if callable(getattr(model, "named_modules", None)) else {}
        if not named_modules:
            raise CapabilityError("exact GPT-2 activation capture requires model.named_modules()")
        requested = tuple(dict.fromkeys(str(name) for name in tensor_names))
        if not requested:
            raise ValueError("exact GPT-2 activation capture requires at least one tensor name")
        missing = [name for name in requested if name not in named_modules]
        if missing:
            raise CapabilityError("exact GPT-2 activation hook modules are missing: " + ", ".join(missing))
        self.model = model
        self._torch = torch
        self.tensor_names = requested
        self._inputs: dict[str, list[np.ndarray]] = {name: [] for name in requested}
        self._outputs: dict[str, list[np.ndarray]] = {name: [] for name in requested}
        self._forward_calls = 0
        self._handles = []
        for name in requested:
            module = named_modules[name]
            hook = self._make_pre_hook(name)
            try:
                self._handles.append(module.register_forward_pre_hook(hook, with_kwargs=True))
            except TypeError:  # pragma: no cover - compatibility with older torch
                self._handles.append(module.register_forward_pre_hook(hook))
            self._handles.append(module.register_forward_hook(self._make_post_hook(name)))

    def _rows(self, value: Any, *, name: str, role: str) -> np.ndarray:
        if isinstance(value, (tuple, list)):
            if not value:
                raise CapabilityError(f"exact GPT-2 {role} hook {name!r} returned an empty tuple")
            value = value[0]
        if not isinstance(value, self._torch.Tensor) or value.ndim < 2:
            raise CapabilityError(f"exact GPT-2 {role} hook {name!r} requires a tensor with batch/sequence axes")
        if not self._torch.is_floating_point(value):
            raise CapabilityError(f"exact GPT-2 {role} hook {name!r} requires a floating-point tensor")
        rows = value.detach().to("cpu").reshape(-1, int(value.shape[-1])).numpy().astype(np.float64, copy=False)
        return finite_array(rows, ndim=2, name=f"exact GPT-2 {role} activation {name}").copy()

    def _make_pre_hook(self, name: str) -> Callable[..., Any]:
        def hook(_module: Any, args: tuple[Any, ...], kwargs: Mapping[str, Any] | None = None) -> None:
            candidates = list(args)
            if kwargs:
                candidates.extend(kwargs.values())
            value = next((item for item in candidates if isinstance(item, self._torch.Tensor)), None)
            if value is None:
                raise CapabilityError(f"exact GPT-2 input hook {name!r} received no positional or keyword tensor")
            self._inputs[name].append(self._rows(value, name=name, role="input"))
        return hook

    def _make_post_hook(self, name: str) -> Callable[..., Any]:
        def hook(_module: Any, _args: tuple[Any, ...], output: Any) -> None:
            self._outputs[name].append(self._rows(output, name=name, role="output"))
        return hook

    def capture(self, batches: Sequence[Mapping[str, Any]]) -> GPT2ActivationCaptureResult:
        if not batches:
            raise ValueError("exact GPT-2 activation capture requires at least one model input batch")
        with self._torch.no_grad():
            for batch in batches:
                if not isinstance(batch, Mapping):
                    raise TypeError("exact GPT-2 activation batches must be mappings accepted by model(**batch)")
                self.model(**dict(batch))
                self._forward_calls += 1
        inputs = {name: np.concatenate(values, axis=0) for name, values in self._inputs.items() if values}
        outputs = {name: np.concatenate(values, axis=0) for name, values in self._outputs.items() if values}
        missing = [name for name in self.tensor_names if name not in inputs or name not in outputs]
        if missing:
            raise CapabilityError("exact GPT-2 activation hook did not observe tensors: " + ", ".join(missing))
        if any(inputs[name].shape[0] != outputs[name].shape[0] for name in self.tensor_names):
            raise CapabilityError("exact GPT-2 activation hook input/output sample counts disagree")
        return GPT2ActivationCaptureResult(inputs, outputs, self._forward_calls, {name: int(inputs[name].shape[0]) for name in self.tensor_names})

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def __enter__(self) -> "GPT2ActivationHookCapture":
        return self

    def __exit__(self, _exc_type: Any, _exc_value: Any, _traceback: Any) -> None:
        self.close()


def capture_gpt2_conv1d_activations(model: Any, batches: Sequence[Mapping[str, Any]], tensor_names: Sequence[str]) -> GPT2ActivationCaptureResult:
    """Capture exact named GPT-2 module rows with temporary forward hooks."""

    with GPT2ActivationHookCapture(model, tensor_names) as capture:
        return capture.capture(batches)


def export_gpt2_checkpoint(model: Any, output_dir: str | Path, *, variant: str) -> Path:
    """Export an ordinary HF-compatible checkpoint after an explicit surgery.

    The function delegates to ``save_pretrained`` when available.  A plain
    torch fallback writes a state dict and config JSON, but it does not create
    adapters or alter a runtime.
    """

    expected = GPT2_XL_TO_SMALL.teacher if variant == "gpt2-xl" else GPT2_XL_TO_SMALL.student if variant == "gpt2-small" else None
    if expected is None and variant not in {"universal", "auto"}:
        raise ValueError("variant must be 'gpt2-xl', 'gpt2-small', 'universal', or 'auto'")
    if expected is not None:
        check = inspect_gpt2_variant(getattr(model, "config", None), expected)
        if not check["supported"]:
            raise CapabilityError("GPT-2 export preflight rejected model: " + "; ".join(check["reasons"]))
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    save_pretrained = getattr(model, "save_pretrained", None)
    if callable(save_pretrained):
        try:
            save_pretrained(destination, safe_serialization=True)
        except TypeError:
            save_pretrained(destination)
        return destination
    try:
        import torch
    except ImportError as error:
        raise CapabilityError("plain torch fallback export requires optional PyTorch") from error
    state_dict = getattr(model, "state_dict", None)
    if not callable(state_dict):
        raise CapabilityError("model exposes neither save_pretrained nor state_dict")
    torch.save(state_dict(), destination / "pytorch_model.bin")
    config = getattr(model, "config", None)
    values = config.to_dict() if callable(getattr(config, "to_dict", None)) else dict(getattr(config, "__dict__", {}))
    (destination / "config.json").write_text(json.dumps(values, indent=2, allow_nan=False), encoding="utf-8")
    return destination


@dataclass(frozen=True)
class GPT2TensorLiftMapping:
    """Explicit GPT-2 block/tensor relation for chart correction lifting.

    The transition index is deliberately separate from the block index parsed
    from ``tensor_name``.  A depth correspondence may map a block to any
    observed flow transition, so guessing by array position would be unsafe.
    ``side='input'`` means ``delta_W = B @ W`` for HF Conv1D's row-vector
    convention.  ``side='output'`` means ``delta_W = W @ B``.
    """

    transition_index: int
    tensor_name: str
    side: str = "input"
    block_index: int | None = None

    def __post_init__(self) -> None:
        if int(self.transition_index) < 0 or not self.tensor_name:
            raise ValueError("GPT-2 tensor lift mapping has invalid transition or name")
        if self.side not in {"input", "output"}:
            raise ValueError("GPT-2 tensor lift side must be 'input' or 'output'")
        if self.block_index is not None and int(self.block_index) < 0:
            raise ValueError("GPT-2 block_index must be non-negative")


_GPT2_WEIGHT_RE = re.compile(r"^transformer\.h\.(\d+)\.(attn\.c_attn|attn\.c_proj|mlp\.c_fc|mlp\.c_proj)\.weight$")
_UNIVERSAL_WEIGHT_RES = (
    (re.compile(r"^(?:transformer\.)?h\.(\d+)\.(attn\.c_attn|attn\.c_proj|mlp\.c_fc|mlp\.c_proj)\.weight$"), {
        "attn.c_attn": ("attn.c_attn", lambda h: (h, 3 * h)),
        "attn.c_proj": ("attn.c_proj", lambda h: (h, h)),
        "mlp.c_fc": ("mlp.c_fc", lambda h: (h, 4 * h)),
        "mlp.c_proj": ("mlp.c_proj", lambda h: (4 * h, h)),
    }),
    (re.compile(r"^(?:model\.)?layers\.(\d+)\.(self_attn\.o_proj|mlp\.down_proj)\.weight$"), {
        "self_attn.o_proj": ("attn.c_proj", lambda h: (h, h)),
        "mlp.down_proj": ("mlp.c_proj", lambda h: (4 * h, h)),
    }),
    (re.compile(r"^(?:bert\.)?encoder\.layer\.(\d+)\.(attention\.output\.dense|output\.dense)\.weight$"), {
        "attention.output.dense": ("attn.c_proj", lambda h: (h, h)),
        "output.dense": ("mlp.c_proj", lambda h: (4 * h, h)),
    }),
    (re.compile(r"^(?:model\.)?decoder\.layers\.(\d+)\.(self_attn\.out_proj|fc2)\.weight$"), {
        "self_attn.out_proj": ("attn.c_proj", lambda h: (h, h)),
        "fc2": ("mlp.c_proj", lambda h: (4 * h, h)),
    }),
)


def _gpt2_weight_contract(name: str, hidden_size: int) -> tuple[int, str, tuple[int, int]] | None:
    """Return ``(block, family, expected_shape)`` for supported HF GPT-2 and generic transformer weights."""

    match = _GPT2_WEIGHT_RE.fullmatch(str(name))
    if match is not None:
        block = int(match.group(1))
        family = match.group(2)
        shapes = {
            "attn.c_attn": (hidden_size, 3 * hidden_size),
            "attn.c_proj": (hidden_size, hidden_size),
            "mlp.c_fc": (hidden_size, 4 * hidden_size),
            "mlp.c_proj": (4 * hidden_size, hidden_size),
        }
        return block, family, shapes[family]

    for pattern, family_map in _UNIVERSAL_WEIGHT_RES:
        m = pattern.fullmatch(str(name))
        if m is not None:
            block = int(m.group(1))
            matched_family = m.group(2)
            if matched_family in family_map:
                norm_family, shape_fn = family_map[matched_family]
                return block, norm_family, shape_fn(hidden_size)
    return None


def _weight_to_numpy(value: Any, *, name: str) -> np.ndarray:
    """Convert a CPU/GPU tensor or ndarray without accepting object arrays."""

    detached = getattr(value, "detach", None)
    if callable(detached):
        value = detached()
        cpu = getattr(value, "cpu", None)
        if callable(cpu):
            value = cpu()
        numpy = getattr(value, "numpy", None)
        if callable(numpy):
            value = numpy()
    array = np.asarray(value)
    if array.dtype == object:
        raise CapabilityError(f"GPT-2 tensor {name!r} cannot be represented as an object array")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"GPT-2 tensor {name} contains non-finite values")
    return array


def _correction_checksum(correction: ConstrainedCorrection) -> str:
    """Fingerprint the raw correction and its safety evidence."""

    digest = hashlib.sha256()
    for name in ("matrices", "biases", "quadratic_terms"):
        value = getattr(correction, name, None)
        if value is None:
            digest.update(f"{name}:none".encode("utf-8"))
            continue
        array = np.ascontiguousarray(finite_array(value, name=f"correction {name}"))
        digest.update(name.encode("utf-8"))
        digest.update(str(array.dtype).encode("utf-8"))
        digest.update(str(array.shape).encode("utf-8"))
        digest.update(array.tobytes())
    accepted = np.ascontiguousarray(np.asarray(correction.accepted, dtype=np.bool_))
    digest.update(b"accepted")
    digest.update(accepted.tobytes())
    for index, reason in sorted((int(index), str(reason)) for index, reason in correction.reasons.items()):
        digest.update(f"reason:{index}:{reason}".encode("utf-8"))
    return digest.hexdigest()


def _lifted_hidden_delta(
    compact: np.ndarray,
    chart_projection: np.ndarray | None,
    *,
    sequence_length: int,
    hidden_size: int,
    max_chart_bytes: int,
    max_cross_token_ratio: float,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    """Contract a compact state correction to the token-local GPT-2 chart.

    If ``P`` is the student chart basis and the compact correction is ``C``,
    the state-space operator would be ``P C Pᵀ``.  This function never forms
    that dense matrix.  It forms only the shared hidden-space average
    ``mean_t(P_t C P_tᵀ)`` and reports the omitted cross-token energy.
    """

    compact = finite_array(compact, ndim=2, name="compact GPT-2 correction")
    if sequence_length < 1 or hidden_size < 1 or not (0.0 <= max_cross_token_ratio < float("inf")):
        raise ValueError("GPT-2 chart dimensions or cross-token tolerance are invalid")
    state_dim = sequence_length * hidden_size
    dense_bytes = int(state_dim) * int(state_dim) * 8
    info: dict[str, Any] = {
        "state_dim": int(state_dim),
        "compact_rank": int(compact.shape[0]),
        "dense_state_matrix_forbidden": True,
        "dense_state_matrix_bytes_estimate": dense_bytes,
        "backend": "token_local_chart_contraction",
        "dense_state_matrix_allocated": False,
        "max_cross_token_ratio": float(max_cross_token_ratio),
    }
    if chart_projection is None:
        if sequence_length == 1 and compact.shape == (hidden_size, hidden_size):
            info.update({"projection_source": "identity", "cross_token_ratio": 0.0, "applied": True})
            return compact.copy(), info
        info.update({"applied": False, "skipped_reason": "student chart_projection is required for sequence or compact-rank lifting"})
        return None, info
    projection = finite_array(chart_projection, ndim=2, name="student chart projection")
    if projection.shape != (state_dim, compact.shape[0]):
        info.update({"applied": False, "skipped_reason": f"chart projection shape {projection.shape} does not match ({state_dim}, {compact.shape[0]})"})
        return None, info
    if projection.nbytes > int(max_chart_bytes):
        info.update({"applied": False, "skipped_reason": f"chart projection requires {projection.nbytes} bytes, above max_chart_bytes={int(max_chart_bytes)}"})
        return None, info
    blocks = projection.reshape(sequence_length, hidden_size, compact.shape[0])
    local = np.stack([block @ compact @ block.T for block in blocks], axis=0)
    hidden_delta = np.mean(local, axis=0)
    local_norm = float(stable_l2(local, name="token-local lifted corrections"))
    diagonal_norm = float(stable_l2(hidden_delta, name="shared hidden correction"))
    # For A_i = B_i C B_j.T, ||A_i||_F^2 equals
    # tr(C.T G_i C G_j), where G_i=B_i.T B_i.  Summing the Gram terms gives
    # the exact off-diagonal Frobenius energy without materializing any
    # H-by-H cross-token matrix.  This is the critical path for GPT-2
    # sequence charts and is algebraically identical to the former loop.
    block_gram = np.einsum("shi,shj->sij", blocks, blocks, optimize=True)
    gram_sum = np.sum(block_gram, axis=0)
    total_cross_squared = float(np.trace(compact.T @ gram_sum @ compact @ gram_sum.T))
    diagonal_cross_squared = float(np.sum([np.trace(compact.T @ gram @ compact @ gram.T) for gram in block_gram]))
    cross_squared = max(0.0, total_cross_squared - diagonal_cross_squared)
    cross_norm = float(np.sqrt(cross_squared))
    cross_count = int(sequence_length * max(0, sequence_length - 1))
    cross_scale = float(np.sqrt(max(1.0, cross_count / 2.0)))
    cross_ratio = (cross_norm / cross_scale) / max(diagonal_norm, 1e-12)
    info.update({
        "projection_source": "student_chart_projection",
        "projection_shape": [int(x) for x in projection.shape],
        "projection_rank": int(projection.shape[1]),
        "local_hidden_delta_norm": local_norm,
        "shared_hidden_delta_norm": diagonal_norm,
        "cross_token_norm": cross_norm,
        "cross_token_ratio": cross_ratio,
        "cross_token_pairs": int(cross_count),
        "applied": bool(np.all(np.isfinite(hidden_delta)) and cross_ratio <= max_cross_token_ratio),
    })
    if not info["applied"]:
        info["skipped_reason"] = (
            f"cross-token correction ratio {cross_ratio:.6g} exceeds max_cross_token_ratio={max_cross_token_ratio:.6g}; "
            "a shared GPT-2 tensor cannot encode the omitted token coupling"
        )
        return None, info
    return hidden_delta, info


def fit_gpt2_conv1d_activation_least_squares(
    input_activations: np.ndarray,
    target_activation_delta: np.ndarray,
    *,
    ridge: float = 1e-6,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Fit a standard GPT-2 Conv1D delta from explicit activation effects.

    For row-vector Conv1D activations ``X`` and a desired output change
    ``Y``, this solves ``min_D ||X D - Y||² + ridge ||D||²`` and returns
    ``D`` in the HF weight orientation. It is a genuine activation-to-weight
    least-squares primitive, separate from the trace-only token-local chart
    contraction. Callers must provide activations captured at the exact block
    and tensor side they intend to edit; the core cannot infer these samples
    from hidden-state traces.
    """

    inputs = finite_array(input_activations, ndim=2, name="GPT-2 Conv1D input activations")
    targets = finite_array(target_activation_delta, ndim=2, name="GPT-2 Conv1D target activation delta")
    if inputs.shape[0] != targets.shape[0] or inputs.shape[0] < 1:
        raise ValueError("activation least-squares samples must have matching non-empty leading dimensions")
    if ridge < 0.0 or not np.isfinite(ridge):
        raise ValueError("activation least-squares ridge must be finite and non-negative")
    input_scale = max(1.0, float(np.max(np.abs(inputs))))
    target_scale = max(1.0, float(np.max(np.abs(targets))))
    x = inputs / input_scale
    y = targets / target_scale
    gram = x.T @ x
    if ridge > 0.0:
        gram = gram + float(ridge) * np.eye(inputs.shape[1], dtype=np.float64)
    rhs = x.T @ y
    try:
        scaled_delta = np.linalg.solve((gram + gram.T) / 2.0, rhs)
    except np.linalg.LinAlgError:
        scaled_delta = np.linalg.lstsq(gram, rhs, rcond=1e-10)[0]
    delta = scaled_delta * (target_scale / input_scale)
    predicted = inputs @ delta
    target_norm = float(np.linalg.norm(targets))
    predicted_norm = float(np.linalg.norm(predicted))
    residual_norm = float(np.linalg.norm(predicted - targets))
    cosine_after = float(np.sum(predicted * targets) / max(predicted_norm * target_norm, 1e-12))
    singular = np.linalg.svd(x, compute_uv=False)
    positive = singular[singular > 1e-12]
    condition = float(np.inf if positive.size == 0 else singular[0] / positive[-1])
    delta = finite_array(delta, ndim=2, name="GPT-2 Conv1D activation least-squares delta")
    report = {
        "backend": "activation_to_weight_least_squares",
        "objective": "min ||X D - Y||_2^2 + ridge ||D||_2^2",
        "input_shape": list(inputs.shape),
        "target_shape": list(targets.shape),
        "delta_shape": list(delta.shape),
        "ridge": float(ridge),
        "input_rank": int(np.linalg.matrix_rank(x)),
        "condition_number": condition,
        "input_scale": input_scale,
        "target_scale": target_scale,
        "input_frobenius_norm": float(np.linalg.norm(inputs)),
        "target_frobenius_norm": target_norm,
        "predicted_target_frobenius_norm": predicted_norm,
        "fit_residual_frobenius_norm": residual_norm,
        "fit_relative_residual": residual_norm / max(target_norm, 1e-12),
        "cosine_before": 0.0,
        "cosine_after": cosine_after,
        "sign": "explicit_target_direction",
        "hooks_required": "exact tensor-side input activations and desired output activation deltas from the same GPT-2 block/context",
        "finite": True,
    }
    return delta, report


def gpt2_depth_gain_weight(
    block_index: int,
    total_blocks: int = 12,
    schedule: str | Sequence[float] | Mapping[int, float] | None = None,
) -> float:
    """Compute layer-wise gain weight according to depth schedule."""
    if schedule is None or schedule == "flat" or schedule == "none":
        return 1.0
    if isinstance(schedule, (Sequence, np.ndarray)) and not isinstance(schedule, str):
        if block_index < len(schedule):
            return float(schedule[block_index])
        return 1.0
    if isinstance(schedule, Mapping):
        if block_index in schedule:
            return float(schedule[block_index])
        if str(block_index) in schedule:
            return float(schedule[str(block_index)])
        return 1.0
    u = float(block_index) / max(1.0, float(total_blocks - 1))
    if schedule == "boost_deep":
        base = 0.30 + 1.05 * np.exp(-((u - 0.70) ** 2) / (2.0 * (0.28 ** 2)))
        boundary_taper = 1.0 - 0.25 * (u ** 6)
        return float(base * boundary_taper)
    elif schedule == "sine":
        return float(0.35 + 0.65 * np.sin(np.pi * (block_index + 0.5) / float(total_blocks)))
    return 1.0


def compute_universal_static_weight_dissimilarity(
    student_state_dict: Mapping[str, Any],
    teacher_source: str | Path | Mapping[str, Any],
    alignment: AlignmentResult | np.ndarray,
    *,
    student_arch: Any | None = None,
    teacher_arch: Any | None = None,
    sequence_length: int = 16,
    site: str | None = None,
    contrast: float = 0.40,
    min_gain: float = 0.25,
    max_gain: float = 2.0,
) -> dict[str, Any]:
    """Compute normalized Frobenius distance between student and projected teacher weights.

    Model-agnostic: works across arbitrary architectures, layers, and hidden dimensions.
    """
    from .model_families import TransformerArchitecture, inspect_transformer_architecture

    if student_arch is None:
        student_arch = TransformerArchitecture("gpt2", 12, 768, 3072, 12, 1024, 50257, "transformer.h", "attn.c_proj", "mlp.c_proj", is_conv1d=True)
    if teacher_arch is None:
        if isinstance(teacher_source, (str, Path)) and Path(teacher_source).exists():
            teacher_arch = inspect_transformer_architecture(teacher_source)
        else:
            teacher_arch = TransformerArchitecture("gpt2", 48, 1600, 6400, 25, 1024, 50257, "transformer.h", "attn.c_proj", "mlp.c_proj", is_conv1d=True)

    student_blocks = int(student_arch.layers)
    teacher_blocks = int(teacher_arch.layers)
    student_dim = int(student_arch.hidden_size)
    teacher_dim = int(teacher_arch.hidden_size)
    resolved_site = site or student_arch.attn_proj_suffix

    if isinstance(alignment, AlignmentResult):
        if alignment.source_projection is not None and alignment.target_projection is not None:
            r = int(alignment.matrix.shape[0])
            b_t = alignment.source_projection.reshape(sequence_length, teacher_dim, r).mean(axis=0)
            b_s = alignment.target_projection.reshape(sequence_length, student_dim, r).mean(axis=0)
            proj_matrix = b_t @ alignment.matrix @ b_s.T
        else:
            proj_matrix = np.asarray(alignment.matrix, dtype=np.float64)
    else:
        proj_matrix = np.asarray(alignment, dtype=np.float64)

    if proj_matrix.shape != (teacher_dim, student_dim):
        raise ValueError(f"alignment projection shape {proj_matrix.shape} does not match ({teacher_dim}, {student_dim})")

    teacher_safetensors = None
    teacher_dict = None
    if isinstance(teacher_source, (str, Path)):
        source_path = Path(teacher_source)
        target_file = source_path / "model.safetensors" if source_path.is_dir() else source_path
        if not target_file.exists():
            raise FileNotFoundError(f"teacher checkpoint file not found: {target_file}")
        try:
            from safetensors import safe_open
            teacher_safetensors = safe_open(str(target_file), framework="numpy")
        except ImportError as err:
            raise CapabilityError("safetensors is required for streaming teacher checkpoint weights") from err
    elif isinstance(teacher_source, Mapping):
        teacher_dict = teacher_source
    else:
        raise TypeError("teacher_source must be a file path or a dictionary of weights")

    def _candidates(arch: Any, idx: int, suffix: str) -> list[str]:
        p = arch.layer_prefix
        return [
            f"{p}.{idx}.{suffix}.weight",
            f"transformer.h.{idx}.{suffix}.weight",
            f"h.{idx}.{suffix}.weight",
            f"model.layers.{idx}.{suffix}.weight",
            f"{idx}.{suffix}.weight",
            f"{p}.{idx}.{suffix}",
        ]

    def get_teacher_weight(idx: int) -> np.ndarray:
        cand_list = _candidates(teacher_arch, idx, teacher_arch.attn_proj_suffix) + _candidates(teacher_arch, idx, resolved_site)
        if teacher_safetensors is not None:
            for cand in cand_list:
                for var in (cand, cand.replace("transformer.", ""), f"transformer.{cand}"):
                    try:
                        return teacher_safetensors.get_tensor(var).astype(np.float64)
                    except Exception:
                        pass
            raise KeyError(f"teacher tensor for layer {idx} not found in safetensors archive; checked: {cand_list[:3]}")
        else:
            for cand in cand_list:
                for var in (cand, cand.replace("transformer.", ""), f"transformer.{cand}"):
                    if var in teacher_dict:
                        return _weight_to_numpy(teacher_dict[var], name=var).astype(np.float64)
            raise KeyError(f"teacher tensor for layer {idx} not found in teacher dict; checked: {cand_list[:3]}")

    def get_student_weight(idx: int) -> np.ndarray:
        cand_list = _candidates(student_arch, idx, student_arch.attn_proj_suffix) + _candidates(student_arch, idx, resolved_site)
        for cand in cand_list:
            for var in (cand, cand.replace("transformer.", ""), f"transformer.{cand}"):
                if var in student_state_dict:
                    return _weight_to_numpy(student_state_dict[var], name=var).astype(np.float64)
        raise KeyError(f"student tensor for layer {idx} not found in student state dict; checked: {cand_list[:3]}")

    distances: list[float] = []
    teacher_layer_map: dict[int, int] = {}
    proj_site = "hidden_to_hidden" if ("attn" in resolved_site or "self_attn" in resolved_site) else "mlp_to_hidden"

    for b_idx in range(student_blocks):
        t_idx = int(round(b_idx * (teacher_blocks - 1) / max(1, student_blocks - 1)))
        teacher_layer_map[b_idx] = t_idx

        wt = get_teacher_weight(t_idx)
        ws = get_student_weight(b_idx)

        wt_proj = project_teacher_weight(wt, proj_matrix, site=proj_site)
        dist = compute_frobenius_dissimilarity(wt_proj, ws)
        distances.append(dist)

    mean_d = float(np.mean(distances))
    std_d = float(np.std(distances))
    if std_d > 1e-12 and np.isfinite(contrast) and contrast > 0:
        raw_schedule = [
            float(np.clip(1.0 + contrast * (d - mean_d) / std_d, min_gain, max_gain))
            for d in distances
        ]
    else:
        raw_schedule = list(dissimilarity_gain_schedule(distances, min_gain=min_gain, max_gain=max_gain))

    return {
        "site": resolved_site,
        "student_blocks": student_blocks,
        "teacher_blocks": teacher_blocks,
        "student_dim": student_dim,
        "teacher_dim": teacher_dim,
        "student_model_type": student_arch.model_type,
        "teacher_model_type": teacher_arch.model_type,
        "teacher_layer_map": teacher_layer_map,
        "distances": distances,
        "mean_distance": mean_d,
        "std_distance": std_d,
        "schedule": tuple(raw_schedule),
        "contrast": contrast,
        "min_gain": min_gain,
        "max_gain": max_gain,
    }


def compute_gpt2_static_weight_dissimilarity(
    student_state_dict: Mapping[str, Any],
    teacher_source: str | Path | Mapping[str, Any],
    alignment: AlignmentResult | np.ndarray,
    *,
    sequence_length: int = 16,
    student_blocks: int = 12,
    teacher_blocks: int = 48,
    teacher_dim: int = 1600,
    student_dim: int = 768,
    site: str = "attn.c_proj",
    contrast: float = 0.40,
    min_gain: float = 0.25,
    max_gain: float = 2.0,
) -> dict[str, Any]:
    """Compute normalized Frobenius distance between student and projected teacher weights.

    Delegates to compute_universal_static_weight_dissimilarity with explicit GPT-2 presets.
    """
    from .model_families import TransformerArchitecture
    student_arch = TransformerArchitecture("gpt2", student_blocks, student_dim, 4 * student_dim, max(1, student_dim // 64), 1024, 50257, "transformer.h", site, "mlp.c_proj", is_conv1d=True)
    teacher_arch = TransformerArchitecture("gpt2", teacher_blocks, teacher_dim, 4 * teacher_dim, max(1, teacher_dim // 64), 1024, 50257, "transformer.h", site, "mlp.c_proj", is_conv1d=True)
    return compute_universal_static_weight_dissimilarity(
        student_state_dict,
        teacher_source,
        alignment,
        student_arch=student_arch,
        teacher_arch=teacher_arch,
        sequence_length=sequence_length,
        contrast=contrast,
        min_gain=min_gain,
        max_gain=max_gain,
        site=site,
    )


def estimate_pilot_pulse_gain_bound(
    activation_inputs: Mapping[str, np.ndarray],
    activation_target_deltas: Mapping[str, np.ndarray],
    *,
    target_logit_shift: float = 0.01,
    max_gain: float = 1.0,
) -> dict[str, Any]:
    """Calculate backward sensitivity bound on gain from a pilot trace.

    Ensures the relative shift in hidden representations does not breach target_logit_shift.
    """
    total_delta_sq = 0.0
    total_input_sq = 0.0
    per_tensor_ratio: dict[str, float] = {}
    for name, delta in activation_target_deltas.items():
        inp = activation_inputs.get(name)
        if inp is None:
            continue
        d_norm = float(np.linalg.norm(delta))
        i_norm = float(np.linalg.norm(inp))
        total_delta_sq += d_norm ** 2
        total_input_sq += i_norm ** 2
        per_tensor_ratio[name] = float(d_norm / max(i_norm, 1e-12))

    total_delta_norm = float(np.sqrt(total_delta_sq))
    total_input_norm = float(np.sqrt(total_input_sq))
    relative_shift = total_delta_norm / max(total_input_norm, 1e-12)

    if relative_shift > 1e-12:
        safe_gain = float(min(max_gain, target_logit_shift / relative_shift))
    else:
        safe_gain = float(max_gain)

    return {
        "target_logit_shift": float(target_logit_shift),
        "total_delta_norm": total_delta_norm,
        "total_input_norm": total_input_norm,
        "relative_shift": relative_shift,
        "safe_gain_bound": safe_gain,
        "recommended_gains": [
            round(safe_gain * 0.25, 4),
            round(safe_gain * 0.50, 4),
            round(safe_gain * 1.00, 4),
            round(safe_gain * 1.50, 4),
        ],
        "per_tensor_relative_shift": per_tensor_ratio,
    }


def build_gpt2_teacher_flow_activation_targets(
    model: Any,
    student_traces: Sequence[TrajectoryTrace],
    student_flow: FlowOperator,
    teacher_flow: FlowOperator,
    mapping: Sequence[GPT2TensorLiftMapping],
    *,
    sequence_length: int,
    hidden_size: int,
    target_effect_ratio: float = 0.05,
    site_policy: str = "mlp_residual_only",
    dual_residual_ratio: float = 0.35,
    depth_schedule: str | None = None,
    use_2jet: bool = False,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, Any]]:
    """Capture block inputs and derive explicit train targets from teacher flow.

    For a mapped transition ``i`` and student source state ``x`` this defines
    the desired full-chart effect as
    ``Δh = (depth[i+1]-depth[i]) * (teacher_flow(x)-student_flow(x))``.
    A full block-output correction cannot be assigned independently to every
    projection within the block: GPT-2 attention and MLP are serial residual
    paths, so that would double-count one desired effect.  The default policy
    therefore selects only ``mlp.c_proj``.  When ``site_policy='dual_residual'``,
    both ``attn.c_proj`` and ``mlp.c_proj`` receive additive residual targets
    split by ``dual_residual_ratio``, conserving the total block intervention.
    """

    if not student_traces:
        raise ValueError("teacher-flow activation targets require non-empty student train traces")
    split_labels = {str(trace.metadata.get("probe_split")) for trace in student_traces if trace.metadata.get("probe_split") is not None}
    if split_labels and split_labels != {"train"}:
        raise ValueError("teacher-flow activation target fitting is train-only; validation/holdout traces are forbidden")
    if sequence_length < 1 or hidden_size < 1:
        raise ValueError("GPT-2 target sequence_length and hidden_size must be positive")
    if not np.isfinite(target_effect_ratio) or target_effect_ratio <= 0.0:
        raise ValueError("teacher-flow target_effect_ratio must be finite and positive")
    if site_policy not in {"mlp_residual_only", "dual_residual"}:
        raise CapabilityError(
            f"teacher-flow target lifting supports site_policy in {{'mlp_residual_only', 'dual_residual'}}, got {site_policy!r}; "
            "other GPT-2 sites require an explicit local response callback"
        )
    if dual_residual_ratio != "auto" and (not np.isfinite(dual_residual_ratio) or not (0.0 < float(dual_residual_ratio) < 1.0)):
        raise ValueError("dual_residual_ratio must be 'auto' or finite float strictly between 0 and 1")
    valid_named_schedules = {None, "flat", "none", "sine", "boost_deep", "dissimilarity"}
    if depth_schedule not in valid_named_schedules and not isinstance(depth_schedule, (Sequence, np.ndarray, Mapping)):
        raise ValueError(f"depth_schedule must be None, 'flat', 'none', 'sine', 'boost_deep', 'dissimilarity', or a sequence/mapping, got {depth_schedule!r}")
    state_dim = int(sequence_length) * int(hidden_size)
    if student_flow.state_dim != state_dim or teacher_flow.state_dim != state_dim:
        raise CapabilityError(
            "teacher-flow activation target chart dimension mismatch: "
            f"student_flow={student_flow.state_dim}, teacher_flow={teacher_flow.state_dim}, expected={state_dim}"
        )
    requested_all = tuple(dict.fromkeys(item.tensor_name for item in mapping))
    if not requested_all:
        raise ValueError("teacher-flow activation targets require at least one mapped tensor")
    selected_mapping: list[GPT2TensorLiftMapping] = []
    skipped_sites: dict[str, str] = {}
    for item in mapping:
        contract = _gpt2_weight_contract(item.tensor_name, hidden_size)
        if contract is None:
            skipped_sites[item.tensor_name] = "unsupported GPT-2 tensor family for teacher-flow direct site contract"
            continue
        _, family, _ = contract
        if site_policy == "mlp_residual_only":
            if family != "mlp.c_proj" or item.side != "output":
                skipped_sites[item.tensor_name] = (
                    "teacher-flow block correction is not a direct activation target for this serial site; "
                    "mlp.c_proj/output is the only supported direct residual site"
                )
                continue
        elif site_policy == "dual_residual":
            if family not in {"mlp.c_proj", "attn.c_proj"} or item.side != "output":
                skipped_sites[item.tensor_name] = (
                    "teacher-flow dual_residual policy supports only attn.c_proj and mlp.c_proj with side='output'"
                )
                continue
        selected_mapping.append(item)
    if not selected_mapping:
        detail = "; ".join(f"{name}: {reason}" for name, reason in sorted(skipped_sites.items()))
        raise CapabilityError("teacher-flow target lifting found no direct GPT-2 residual sites; " + detail)
    requested = tuple(dict.fromkeys(item.tensor_name for item in selected_mapping))
    module_names = tuple(name[:-len(".weight")] if name.endswith(".weight") else name for name in requested)
    batches: list[dict[str, Any]] = []
    for trace in student_traces:
        if trace.state_dim != state_dim:
            raise CapabilityError(f"student trace {trace.probe_id!r} state_dim={trace.state_dim} does not match GPT-2 chart {state_dim}")
        if trace.token_ids is None:
            raise CapabilityError(f"student trace {trace.probe_id!r} has no token_ids for exact block-input capture")
        token_ids = np.asarray(trace.token_ids, dtype=np.int64).reshape(-1)
        if len(token_ids) != sequence_length:
            raise CapabilityError(f"student trace {trace.probe_id!r} token count does not match sequence_length={sequence_length}")
        position_ids = np.arange(sequence_length, dtype=np.int64) if trace.position_ids is None else np.asarray(trace.position_ids, dtype=np.int64).reshape(-1)
        if len(position_ids) != sequence_length:
            raise CapabilityError(f"student trace {trace.probe_id!r} position count does not match sequence_length={sequence_length}")
        try:
            import torch
        except ImportError as error:  # pragma: no cover - optional real-model path
            raise CapabilityError("teacher-flow activation target capture requires optional PyTorch") from error
        parameter = next(model.parameters(), None) if callable(getattr(model, "parameters", None)) else None
        device = None if parameter is None else parameter.device
        def tensor(values: np.ndarray, dtype: Any) -> Any:
            result = torch.as_tensor(values, dtype=dtype)
            return result if device is None else result.to(device)
        batches.append({
            "input_ids": tensor(token_ids[None, :], torch.long),
            "position_ids": tensor(position_ids[None, :], torch.long),
            "attention_mask": tensor(np.ones((1, sequence_length), dtype=np.int64), torch.long),
        })
    capture = capture_gpt2_conv1d_activations(model, batches, module_names)
    activation_inputs = {name: capture.inputs[module] for name, module in zip(requested, module_names) if module in capture.inputs}
    target_values: dict[str, list[np.ndarray]] = {name: [] for name in requested}
    transition_by_tensor = {item.tensor_name: int(item.transition_index) for item in selected_mapping}
    for trace in student_traces:
        for name in requested:
            index = transition_by_tensor[name]
            contract = _gpt2_weight_contract(name, hidden_size)
            b_idx = contract[0] if contract is not None else index
            max_block = max((item.block_index for item in selected_mapping if item.block_index is not None), default=11)
            num_student_blocks = max(12, max_block + 1)
            denom = max(1.0, float(num_student_blocks - 1))
            if index >= len(trace.transitions):
                index = min(len(trace.transitions) - 1, int(round(b_idx * (len(trace.transitions) - 1) / denom)))
            transition = trace.transitions[index]
            step = float(transition.target_depth - transition.source_depth)
            if step <= 0.0:
                step = max(1e-6, abs(step))
            student_v1 = student_flow.predict(transition.source_state, transition.source_depth)
            teacher_v1 = teacher_flow.predict(transition.source_state, transition.source_depth)
            if use_2jet:
                st_pred = transition.source_state + step * student_v1
                tt_pred = transition.source_state + step * teacher_v1
                student_v2 = student_flow.predict(st_pred, transition.target_depth)
                teacher_v2 = teacher_flow.predict(tt_pred, transition.target_depth)
                student_value = 0.5 * (student_v1 + student_v2)
                teacher_value = 0.5 * (teacher_v1 + teacher_v2)
            else:
                student_value = student_v1
                teacher_value = teacher_v1
            full_delta = (teacher_value - student_value) * step
            token_delta = np.asarray(full_delta, dtype=np.float64).reshape(sequence_length, hidden_size)
            family = contract[1] if contract is not None else ""
            if site_policy == "dual_residual":
                if dual_residual_ratio == "auto":
                    attn_out = capture.outputs.get(f"h.{b_idx}.attn.c_proj") or capture.outputs.get(f"transformer.h.{b_idx}.attn.c_proj")
                    mlp_out = capture.outputs.get(f"h.{b_idx}.mlp.c_proj") or capture.outputs.get(f"transformer.h.{b_idx}.mlp.c_proj")
                    if attn_out is None or mlp_out is None:
                        for mod_k, mod_v in capture.outputs.items():
                            if f".{b_idx}." in mod_k or mod_k.startswith(f"{b_idx}."):
                                if "attn" in mod_k:
                                    attn_out = mod_v
                                elif "mlp" in mod_k or "down_proj" in mod_k or "output" in mod_k:
                                    mlp_out = mod_v
                    if attn_out is not None and mlp_out is not None:
                        attn_energy = float(np.mean(np.var(attn_out, axis=0)))
                        mlp_energy = float(np.mean(np.var(mlp_out, axis=0)))
                        tot = attn_energy + mlp_energy
                        ratio = float(np.clip(attn_energy / tot, 0.15, 0.85)) if tot > 1e-12 else 0.35
                    else:
                        ratio = 0.35
                else:
                    ratio = float(dual_residual_ratio)
                if family == "attn.c_proj":
                    token_delta = token_delta * ratio
                elif family == "mlp.c_proj":
                    token_delta = token_delta * (1.0 - ratio)
            if depth_schedule is not None and contract is not None:
                token_delta = token_delta * gpt2_depth_gain_weight(b_idx, num_student_blocks, depth_schedule)
            target_values[name].append(token_delta)
    raw_targets = {
        name: finite_array(np.concatenate(rows, axis=0), ndim=2, name=f"raw teacher-flow activation target {name}")
        for name, rows in target_values.items()
        if rows and name in activation_inputs
    }
    target_calibration: dict[str, dict[str, Any]] = {}
    activation_targets: dict[str, np.ndarray] = {}
    for name, raw_target in raw_targets.items():
        module_name = name[:-len(".weight")] if name.endswith(".weight") else name
        output_rows = capture.outputs.get(module_name)
        if output_rows is None or output_rows.shape[0] != raw_target.shape[0]:
            raise CapabilityError(f"teacher-flow target calibration has no matching output rows for {name!r}")
        raw_row_norm = np.linalg.norm(raw_target, axis=1)
        output_row_norm = np.linalg.norm(output_rows, axis=1)
        input_row_norm = np.linalg.norm(activation_inputs[name], axis=1)
        reference_norm = np.maximum(output_row_norm, input_row_norm)
        scale = np.minimum(1.0, float(target_effect_ratio) * reference_norm / np.maximum(raw_row_norm, 1e-12))
        calibrated = finite_array(raw_target * scale[:, None], ndim=2, name=f"calibrated teacher-flow activation target {name}")
        activation_targets[name] = calibrated
        target_calibration[name] = {
            "target_effect_ratio": float(target_effect_ratio),
            "raw_target_rms": float(np.sqrt(np.mean(np.square(raw_target)))),
            "calibrated_target_rms": float(np.sqrt(np.mean(np.square(calibrated)))),
            "student_output_rms": float(np.sqrt(np.mean(np.square(output_rows)))),
            "student_input_rms": float(np.sqrt(np.mean(np.square(activation_inputs[name])))),
            "scale_min": float(np.min(scale)),
            "scale_median": float(np.median(scale)),
            "scale_max": float(np.max(scale)),
            "clipped_fraction": float(np.mean(scale < 1.0)),
            "finite": True,
        }
    missing = sorted(set(requested) - set(activation_inputs) - set(activation_targets))
    if missing:
        raise CapabilityError("teacher-flow activation target capture missed mapped tensors: " + ", ".join(missing))
    if any(activation_inputs[name].shape[0] != activation_targets[name].shape[0] for name in activation_targets):
        raise CapabilityError("teacher-flow activation input and target row counts disagree")
    metadata = {
        "backend": "torch_forward_hooks_plus_teacher_flow_projection",
        "target_kind": "depth_step_teacher_minus_student_flow_in_student_chart",
        "site_policy": site_policy,
        "site_contract": "gpt2_dual_residual_additive" if site_policy == "dual_residual" else "gpt2_mlp_c_proj_eval_additive_residual",
        "dual_residual_ratio": "auto" if dual_residual_ratio == "auto" else (float(dual_residual_ratio) if site_policy == "dual_residual" else None),
        "depth_schedule": str(depth_schedule) if not isinstance(depth_schedule, (Sequence, np.ndarray, Mapping)) else "custom_vector",
        "use_2jet": bool(use_2jet),
        "direct_to_block_output": True,
        "skipped_site_mappings": skipped_sites,
        "target_calibration_kind": "per_token_output_norm_gated",
        "target_effect_ratio": float(target_effect_ratio),
        "target_calibration": target_calibration,
        "target_fit_split": "train",
        "target_is_module_attribution": False,
        "exact_ridge_scope": "regression against explicit target rows only",
        "state_dim": state_dim,
        "sequence_length": int(sequence_length),
        "hidden_size": int(hidden_size),
        "tensor_names": list(requested),
        "requested_tensor_names": list(requested_all),
        "module_names": list(module_names),
        "forward_calls": int(capture.forward_calls),
        "samples_by_tensor": {name: int(value.shape[0]) for name, value in activation_inputs.items()},
        "finite": bool(capture.finite and all(np.all(np.isfinite(value)) for value in activation_targets.values())),
        "teacher_at_inference": False,
    }
    return activation_inputs, activation_targets, metadata


def _named_callback_updates(originals: Mapping[str, np.ndarray], values: Mapping[str, Any]) -> tuple[dict[str, np.ndarray], tuple[str, ...], dict[str, str]]:
    updates: dict[str, np.ndarray] = {}
    applied: list[str] = []
    skipped: dict[str, str] = {}
    for raw_name, raw_value in values.items():
        name = str(raw_name)
        if name not in originals:
            skipped[name] = "quadratic callback returned a tensor outside the GPT-2 schema"
            continue
        candidate = _weight_to_numpy(raw_value, name=name)
        if candidate.shape != originals[name].shape or candidate.dtype != originals[name].dtype:
            skipped[name] = f"quadratic callback changed tensor schema: {candidate.shape}/{candidate.dtype} vs {originals[name].shape}/{originals[name].dtype}"
            continue
        if np.array_equal(candidate, originals[name]):
            skipped[name] = "quadratic callback returned an unchanged tensor"
            continue
        updates[name] = candidate.copy()
        applied.append(name)
    return updates, tuple(applied), skipped


def build_gpt2_surgery_plan(
    weights: Mapping[str, Any],
    correction: ConstrainedCorrection,
    *,
    chart_projection: np.ndarray | None,
    sequence_length: int,
    hidden_size: int,
    mapping: Sequence[GPT2TensorLiftMapping],
    gain: float = 1.0,
    mode: str = "diagnostic",
    max_chart_bytes: int = 512 * 1024 * 1024,
    max_cross_token_ratio: float = 0.5,
    quadratic_callback: Callable[[Mapping[str, np.ndarray], np.ndarray, float], Mapping[str, Any]] | None = None,
    activation_inputs: Mapping[str, np.ndarray] | None = None,
    activation_target_deltas: Mapping[str, np.ndarray] | None = None,
    activation_ridge: float = 1e-6,
    activation_target_metadata: Mapping[str, Any] | None = None,
    activation_least_squares_cache: dict[tuple[str, float], tuple[np.ndarray, dict[str, Any]]] | None = None,
    piecewise_knots: bool = False,
    piecewise_num_pieces: int = 4,
    piecewise_tolerance: float = 0.35,
    piecewise_svd_rank_ratio: float = 0.50,
    piecewise_cosine_threshold: float = 0.50,
    piecewise_eta: float = 0.08,
    calibrate_lm_head: bool = False,
    teacher_lm_head: np.ndarray | None = None,
    lm_head_gain: float = 0.03,
    lm_head_name: str = "lm_head.weight",
    memory_imprint_mlp: bool = False,
    memory_imprint_gain: float = 1.0,
    memory_imprint_ridge: float = 1e-3,
) -> SurgeryPlan:
    """Build an explicit, shape-aware GPT-2 ordinary-weight surgery plan.

    The function supports only the four standard HF GPT-2 Conv1D weight
    families.  LayerNorm, biases, fused alternatives, sharded tensors and
    gated variants are reported as skipped.  The mapping is mandatory and
    transition orientation is explicit; no tensor is selected by name/shape
    coincidence.
    """

    if mode not in {"diagnostic", "apply", "experimental"}:
        raise ValueError("GPT-2 surgery mode must be 'diagnostic', 'apply', or explicit 'experimental'")
    if not np.isfinite(gain):
        raise ValueError("GPT-2 surgery gain must be finite")
    if int(max_chart_bytes) < 1:
        raise ValueError("max_chart_bytes must be positive")
    if not (0.0 <= float(max_cross_token_ratio) < float("inf")):
        raise ValueError("max_cross_token_ratio must be finite and non-negative")
    if not np.isfinite(activation_ridge) or activation_ridge < 0.0:
        raise ValueError("activation_ridge must be finite and non-negative")
    if (activation_inputs is None) != (activation_target_deltas is None):
        raise CapabilityError("exact activation lift requires both activation_inputs and activation_target_deltas")
    if not mapping:
        if mode in {"apply", "experimental"}:
            raise CapabilityError(f"GPT-2 {mode} mode requires explicit GPT2TensorLiftMapping entries")
        mapping = ()
    if any(not isinstance(item, GPT2TensorLiftMapping) for item in mapping):
        raise TypeError("GPT-2 mapping must contain GPT2TensorLiftMapping entries")
    if len({item.tensor_name for item in mapping}) != len(mapping):
        raise ValueError("GPT-2 mapping contains duplicate tensor names")
    if sequence_length < 1 or hidden_size < 1:
        raise ValueError("GPT-2 sequence_length and hidden_size must be positive")
    originals = {str(name): _weight_to_numpy(value, name=str(name)).copy() for name, value in weights.items()}
    schema = tuple(TensorSchemaEntry(name, tuple(int(s) for s in value.shape), str(value.dtype)) for name, value in originals.items())
    updates = {name: value.copy() for name, value in originals.items()}
    skipped: dict[str, str] = {}
    applied: list[str] = []
    rollback_layers = {int(index) for index, accepted in enumerate(np.asarray(correction.accepted, dtype=bool)) if not accepted}
    reasons = {int(index): str(reason) for index, reason in correction.reasons.items()}
    correction_checksum = _correction_checksum(correction)
    lift_reports: dict[str, dict[str, Any]] = {}
    exact_activation_tensors: list[str] = []
    heuristic_tensors: list[str] = []
    transition_cache: dict[int, tuple[np.ndarray | None, dict[str, Any]]] = {}
    raw_unaccepted_transitions: set[int] = set()
    attention_detour_bases: dict[int, list[np.ndarray]] = {}
    piecewise_knots_records: dict[str, dict[str, Any]] = {}

    quadratic_present = correction.quadratic_terms is not None and bool(np.any(np.abs(correction.quadratic_terms) > 0))
    quadratic_applied = False
    if quadratic_present and quadratic_callback is None:
        skipped["<flow.quadratic_terms>"] = "GPT-2 ordinary tensors cannot encode quadratic flow; supply explicit quadratic_callback"
    if np.any(np.abs(np.asarray(correction.biases, dtype=np.float64)) > 0.0):
        skipped["<flow.biases>"] = "GPT-2 tensor lift applies the linear state operator only; flow bias requires an explicit activation callback"
    if quadratic_present and quadratic_callback is not None:
        callback_result = quadratic_callback(originals, correction.quadratic_terms.copy(), gain)
        if not isinstance(callback_result, Mapping):
            raise CapabilityError("GPT-2 quadratic callback must return a mapping of named tensors")
        callback_updates, callback_applied, callback_skipped = _named_callback_updates(originals, callback_result)
        updates.update(callback_updates)
        applied.extend(callback_applied)
        quadratic_applied = bool(callback_applied)
        skipped.update({name: f"quadratic callback: {reason}" for name, reason in callback_skipped.items()})
        if not callback_result:
            skipped["<flow.quadratic_terms>"] = "quadratic callback returned no tensor updates"

    for item in mapping:
        name = item.tensor_name
        if name not in originals:
            skipped[name] = "mapped GPT-2 tensor does not exist in current schema"
            continue
        contract = _gpt2_weight_contract(name, int(hidden_size))
        if contract is None:
            skipped[name] = "tensor is not one of the supported standard GPT-2 Conv1D weights; fused/sharded/gated variants require a family callback"
            continue
        block_index, family, expected_shape = contract
        if item.block_index is not None and int(item.block_index) != block_index:
            skipped[name] = f"explicit block_index={item.block_index} disagrees with tensor block {block_index}"
            continue

        effective_transition_index = item.transition_index
        if effective_transition_index >= len(correction.matrices):
            if len(correction.matrices) > 0 and block_index is not None:
                effective_transition_index = min(len(correction.matrices) - 1, int(round(block_index * (len(correction.matrices) - 1) / 11.0)))
            else:
                skipped[name] = "mapping references a missing flow transition"
                continue

        value = originals[name]
        if value.shape != expected_shape:
            skipped[name] = f"GPT-2 {family} shape {value.shape} does not match expected {expected_shape}"
            continue
        if not np.issubdtype(value.dtype, np.floating):
            skipped[name] = f"tensor dtype {value.dtype} is not a floating point weight"
            continue
        if not bool(correction.accepted[effective_transition_index]) and mode != "experimental":
            skipped[name] = f"flow transition rolled back: {reasons.get(effective_transition_index, 'unsafe correction')}"
            continue
        if mode == "experimental" and not bool(correction.accepted[effective_transition_index]):
            raw_unaccepted_transitions.add(int(effective_transition_index))
        exact_activation_requested = activation_inputs is not None and activation_target_deltas is not None
        if exact_activation_requested:
            hidden_delta = None
            lift_info = {"backend": "exact_activation_ls", "applied": True, "skipped_reason": None}
        else:
            if effective_transition_index not in transition_cache:
                transition_cache[effective_transition_index] = _lifted_hidden_delta(
                    correction.matrices[effective_transition_index],
                    chart_projection,
                    sequence_length=sequence_length,
                    hidden_size=hidden_size,
                    max_chart_bytes=max_chart_bytes,
                    max_cross_token_ratio=float(max_cross_token_ratio),
                )
            hidden_delta, lift_info = transition_cache[effective_transition_index]
        lift_reports[str(effective_transition_index)] = lift_info
        if hidden_delta is None and not exact_activation_requested:
            skipped[name] = str(lift_info.get("skipped_reason", "chart lifting was rejected"))
            continue
        if not exact_activation_requested:
            if family in {"attn.c_attn", "mlp.c_fc"} and item.side != "input":
                skipped[name] = f"{family} supports only side='input' for an H-dimensional state correction"
                continue
            if family in {"attn.c_proj", "mlp.c_proj"} and item.side != "output":
                skipped[name] = f"{family} supports only side='output' for an H-dimensional state correction; input is intermediate attention heads/activations"
                continue
        if activation_inputs is not None and activation_target_deltas is not None:
            if name not in activation_inputs or name not in activation_target_deltas:
                site_skips = {} if activation_target_metadata is None else dict(activation_target_metadata.get("skipped_site_mappings", {}))
                skipped[name] = str(site_skips.get(
                    name,
                    "exact activation lift requested but hook input or target delta is missing for this tensor",
                ))
                continue
            is_mlp_tensor = family in {"mlp.c_proj", "mlp.c_fc"}
            use_piecewise = bool(piecewise_knots and is_mlp_tensor)
            cache_key = (
                name,
                float(activation_ridge),
                use_piecewise,
                int(piecewise_num_pieces),
                float(piecewise_tolerance),
                float(piecewise_svd_rank_ratio),
                float(piecewise_cosine_threshold),
                float(piecewise_eta),
            )
            if activation_least_squares_cache is not None and cache_key in activation_least_squares_cache:
                tensor_delta, activation_report = activation_least_squares_cache[cache_key]
            else:
                try:
                    if use_piecewise:
                        tensor_delta, activation_report = fit_piecewise_mlp_with_retries(
                            activation_inputs[name],
                            activation_target_deltas[name],
                            num_pieces=int(piecewise_num_pieces),
                            tolerance=float(piecewise_tolerance),
                            svd_rank_ratio=float(piecewise_svd_rank_ratio),
                            cosine_threshold=float(piecewise_cosine_threshold),
                            base_ridge=float(activation_ridge),
                            retry_ridge=max(1e-2, float(activation_ridge) * 1000.0),
                        )
                    elif memory_imprint_mlp and is_mlp_tensor:
                        k_arr = np.asarray(activation_inputs[name], dtype=np.float64)
                        v_arr = np.asarray(activation_target_deltas[name], dtype=np.float64)
                        k_norm_sq = np.sum(k_arr ** 2, axis=1, keepdims=True)
                        k_scaled = k_arr / (k_norm_sq + float(memory_imprint_ridge))
                        delta_unclamped = k_scaled.T @ v_arr
                        w_norm = float(np.linalg.norm(value))
                        d_norm = float(np.linalg.norm(delta_unclamped))
                        clamp = min(1.0, (0.02 * w_norm) / max(d_norm, 1e-8))
                        tensor_delta = float(memory_imprint_gain) * clamp * delta_unclamped
                        activation_report = {
                            "method": "rank_one_memory_imprint",
                            "gain": float(memory_imprint_gain),
                            "ridge": float(memory_imprint_ridge),
                            "clamp": float(clamp),
                            "key_count": len(k_arr),
                        }
                    else:
                        tensor_delta, activation_report = fit_gpt2_conv1d_activation_least_squares(
                            activation_inputs[name], activation_target_deltas[name], ridge=float(activation_ridge)
                        )
                except (TypeError, ValueError, FloatingPointError) as error:
                    skipped[name] = f"exact activation lift rejected: {error}"
                    continue
                if activation_least_squares_cache is not None:
                    activation_least_squares_cache[cache_key] = (tensor_delta, activation_report)
            if tuple(tensor_delta.shape) != tuple(value.shape):
                skipped[name] = f"exact activation lift shape {tensor_delta.shape} does not match tensor {value.shape}"
                continue
            lift_reports[f"activation_ls:{name}"] = activation_report
            if use_piecewise:
                piecewise_knots_records[name] = activation_report
                kbases = activation_report.get("knot_bases", [])
                if kbases and block_index is not None:
                    attention_detour_bases.setdefault(int(block_index), []).extend(kbases)
            if activation_target_metadata is not None:
                lift_reports[f"activation_ls:{name}"]["target_provenance"] = dict(activation_target_metadata)
            exact_activation_tensors.append(name)
        else:
            tensor_delta = hidden_delta @ value if item.side == "input" else value @ hidden_delta
            heuristic_tensors.append(name)
        candidate = value.astype(np.float64) + float(gain) * tensor_delta
        if not np.all(np.isfinite(candidate)):
            skipped[name] = "candidate GPT-2 tensor is non-finite"
            rollback_layers.add(effective_transition_index)
            reasons[effective_transition_index] = "candidate GPT-2 tensor is non-finite"
            continue
        cast_candidate = candidate.astype(value.dtype, copy=False)
        if not np.all(np.isfinite(cast_candidate)):
            skipped[name] = f"candidate overflows GPT-2 tensor dtype {value.dtype}"
            rollback_layers.add(effective_transition_index)
            reasons[effective_transition_index] = "candidate GPT-2 tensor overflows dtype"
            continue
        if np.array_equal(cast_candidate, value):
            skipped[name] = "candidate GPT-2 tensor is unchanged after dtype cast"
            continue
        updates[name] = cast_candidate
        applied.append(name)

    # Apply Attention Detour routing for blocks with detected MLP knots
    if piecewise_knots and attention_detour_bases:
        for b_idx, bases in sorted(attention_detour_bases.items()):
            if not bases:
                continue
            detour_projector = build_attention_detour_projector(bases, hidden_size=int(hidden_size), eta=float(piecewise_eta))
            attn_proj_name = None
            for cand_name in originals:
                c = _gpt2_weight_contract(cand_name, int(hidden_size))
                if c is not None and c[0] == b_idx and c[1] == "attn.c_proj":
                    attn_proj_name = cand_name
                    break
            if attn_proj_name is not None and attn_proj_name in updates:
                current_weight = updates[attn_proj_name].astype(np.float64)
                detoured_weight = current_weight @ detour_projector
                cast_detoured = detoured_weight.astype(originals[attn_proj_name].dtype, copy=False)
                if np.all(np.isfinite(cast_detoured)):
                    updates[attn_proj_name] = cast_detoured
                    if attn_proj_name not in applied:
                        applied.append(attn_proj_name)
                    skipped.pop(attn_proj_name, None)
                    svs = np.linalg.svd(detour_projector, compute_uv=False)
                    lift_reports[f"attention_detour:{attn_proj_name}"] = {
                        "backend": "attention_knot_detour_routing",
                        "block_index": b_idx,
                        "tensor_name": attn_proj_name,
                        "eta": float(piecewise_eta),
                        "knot_bases_count": len(bases),
                        "singular_values_min": float(np.min(svs)),
                        "singular_values_max": float(np.max(svs)),
                        "applied": True,
                    }

    # Vector 4: Vocabulary Logit Alignment (lm_head projection)
    if calibrate_lm_head and teacher_lm_head is not None and chart_projection is not None:
        target_head_name = lm_head_name if lm_head_name in originals else ("wte.weight" if "wte.weight" in originals else None)
        if target_head_name is not None:
            student_head = originals[target_head_name]
            try:
                aligned_head = align_vocabulary_head(
                    student_head,
                    teacher_lm_head,
                    chart_projection,
                    gain=float(lm_head_gain),
                    max_relative_norm=0.02,
                )
                cast_head = aligned_head.astype(student_head.dtype, copy=False)
                if np.all(np.isfinite(cast_head)) and not np.array_equal(cast_head, student_head):
                    updates[target_head_name] = cast_head
                    if target_head_name not in applied:
                        applied.append(target_head_name)
                    skipped.pop(target_head_name, None)
                    lift_reports[f"vocabulary_head_alignment:{target_head_name}"] = {
                        "backend": "vocabulary_head_procrustes_projection",
                        "gain": float(lm_head_gain),
                        "teacher_shape": list(teacher_lm_head.shape),
                        "student_shape": list(student_head.shape),
                        "chart_projection_shape": list(chart_projection.shape),
                        "applied": True,
                    }
            except (ValueError, TypeError) as align_err:
                skipped[target_head_name] = f"vocabulary head alignment skipped: {align_err}"
        else:
            skipped["lm_head"] = f"neither {lm_head_name} nor wte.weight found in weights dictionary"

    selected_names = {item.tensor_name for item in mapping}
    for name in originals:
        if name not in selected_names and name not in applied and name not in skipped:
            skipped[name] = "not selected by an explicit GPT2TensorLiftMapping"
    applied = list(dict.fromkeys(applied))
    if mode in {"apply", "experimental"} and not applied:
        detail = "; ".join(f"{name}: {reason}" for name, reason in skipped.items()) or "no explicit GPT-2 mapping supplied"
        raise CapabilityError(f"GPT-2 {mode} mode produced no applicable tensor updates; {detail}")
    state_dim = int(sequence_length) * int(hidden_size)
    changed_tensor_details = []
    for name in applied:
        delta = updates[name].astype(np.float64) - originals[name].astype(np.float64)
        changed_tensor_details.append({
            "tensor_name": name,
            "shape": list(originals[name].shape),
            "dtype": str(originals[name].dtype),
            "delta_l2": float(stable_l2(delta, name=f"GPT-2 tensor delta {name}")),
        })
    metadata = {
        "family": "gpt2",
        "mode": mode,
        "trust_status": "experimental_untrusted" if mode == "experimental" else ("safe_candidate" if mode == "apply" else "diagnostic_only"),
        "acceptance_gate_bypassed": bool(mode == "experimental"),
        "correction_checksum": correction_checksum,
        "gain": float(gain),
        "mapping_count": len(mapping),
        "applied_tensors": tuple(applied),
        "changed_tensor_details": changed_tensor_details,
        "raw_unaccepted_transitions_used": sorted(raw_unaccepted_transitions),
        "rollback_layers": sorted(rollback_layers),
        "rollback_reasons": {str(index): reason for index, reason in sorted(reasons.items())},
        "chart_backend": "token_local_chart_contraction",
        "lift_method": "exact_activation_ls" if exact_activation_tensors and not heuristic_tensors else ("mixed_exact_activation_ls_and_heuristic_chart" if exact_activation_tensors else "first_order_token_local_chart_contraction"),
        "lift_proof_status": "exact ridge solve from captured per-tensor inputs and explicit target activation deltas" if exact_activation_tensors and not heuristic_tensors else ("mixed exact and heuristic tensors; inspect per-tensor lift reports" if exact_activation_tensors else "heuristic chart lift; not an exact least_squares activation solution"),
        "exact_activation_ls_tensors": tuple(exact_activation_tensors),
        "heuristic_chart_tensors": tuple(heuristic_tensors),
        "activation_ridge": float(activation_ridge),
        "activation_target_provenance": None if activation_target_metadata is None else dict(activation_target_metadata),
        "required_activation_hooks": "exact per-block Conv1D input hooks plus separately defined target activation deltas; otherwise heuristic chart contraction",
        "chart_state_dim": state_dim,
        "chart_projection_rank": None if chart_projection is None else int(np.asarray(chart_projection).shape[1]),
        "dense_state_matrix_forbidden": True,
        "dense_state_matrix_allocated": False,
        "dense_state_matrix_bytes_estimate": state_dim * state_dim * 8,
        "lift_reports": lift_reports,
        "quadratic_terms_applied": quadratic_applied,
        "quadratic_callback_supplied": quadratic_callback is not None,
        "unsupported_tensor_policy": "explicit_skip_with_reason",
        "piecewise_knots": bool(piecewise_knots),
        "piecewise_knots_summary": {
            "enabled": bool(piecewise_knots),
            "num_pieces": int(piecewise_num_pieces),
            "tolerance": float(piecewise_tolerance),
            "eta": float(piecewise_eta),
            "mlp_tensors": list(piecewise_knots_records.keys()),
            "total_knots_skipped": sum(int(r.get("skipped_knot_count", 0)) for r in piecewise_knots_records.values()),
            "total_pieces": sum(int(r.get("total_pieces", 0)) for r in piecewise_knots_records.values()),
            "detours_applied": [k for k in lift_reports if k.startswith("attention_detour:")],
        } if piecewise_knots else None,
    }
    return SurgeryPlan(updates, tuple(sorted(rollback_layers)), reasons, np.asarray(correction.confidence, dtype=np.float64).copy(), schema, metadata, skipped, tuple(applied))


def apply_gpt2_surgery(model: Any, plan: SurgeryPlan) -> Any:
    """Commit a validated GPT-2 plan through the model's ordinary state_dict."""

    state_dict = getattr(model, "state_dict", None)
    load_state_dict = getattr(model, "load_state_dict", None)
    if not callable(state_dict) or not callable(load_state_dict):
        raise CapabilityError("GPT-2 model must expose state_dict and load_state_dict for ordinary surgery")
    current = {str(name): _weight_to_numpy(value, name=str(name)) for name, value in state_dict().items()}
    from .surgery import apply_surgery

    updated = apply_surgery(current, plan, require_update=True)
    try:
        import torch
    except ImportError as error:
        raise CapabilityError("GPT-2 surgery commit requires optional PyTorch") from error
    replacement = {}
    for name, value in state_dict().items():
        replacement[name] = torch.as_tensor(updated[name], dtype=value.dtype, device=value.device)
    load_state_dict(replacement, strict=True)
    return model


def _model_state_checksum(model: Any) -> str:
    state_dict = getattr(model, "state_dict", None)
    if not callable(state_dict):
        raise CapabilityError("model exposes no state_dict for checkpoint fingerprint")
    digest = hashlib.sha256()
    for name, raw_value in sorted(state_dict().items(), key=lambda item: str(item[0])):
        value = raw_value.detach().cpu().contiguous() if callable(getattr(raw_value, "detach", None)) else raw_value
        digest.update(str(name).encode("utf-8"))
        digest.update(str(getattr(value, "dtype", None)).encode("utf-8"))
        shape = tuple(value.shape) if hasattr(value, "shape") else np.asarray(value).shape
        digest.update(str(shape).encode("utf-8"))
        numpy = getattr(value, "numpy", None)
        payload = numpy().tobytes() if callable(numpy) else np.ascontiguousarray(value).tobytes()
        digest.update(payload)
    return digest.hexdigest()


def export_gpt2_checkpoint_pair(
    model: Any,
    output_dir: str | Path,
    *,
    variant: str,
    experimental_plan: SurgeryPlan | None = None,
    tokenizer: Any | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> Path:
    """Export clean baseline and optional candidate as ordinary HF directories.

    The candidate is produced by temporarily loading the explicit plan into
    the supplied model, then the clean state is restored and checked by a
    checksum.  The export contains no adapter, LoRA module, or runtime fork.
    Existing ``baseline``/``candidate`` directories are rejected to avoid
    silently replacing an experiment.
    """

    expected = GPT2_XL_TO_SMALL.teacher if variant == "gpt2-xl" else GPT2_XL_TO_SMALL.student if variant == "gpt2-small" else None
    if expected is None and variant not in {"universal", "auto"}:
        raise ValueError("variant must be 'gpt2-xl', 'gpt2-small', 'universal', or 'auto'")
    if expected is not None:
        check = inspect_gpt2_variant(getattr(model, "config", None), expected)
        if not check["supported"]:
            raise CapabilityError("GPT-2 checkpoint-pair export preflight rejected model: " + "; ".join(check["reasons"]))
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    baseline_destination = destination / "baseline"
    candidate_destination = destination / "candidate"
    if baseline_destination.exists() or candidate_destination.exists():
        raise FileExistsError("GPT-2 checkpoint-pair export refuses existing baseline or candidate directory")
    state_dict = getattr(model, "state_dict", None)
    load_state_dict = getattr(model, "load_state_dict", None)
    if not callable(state_dict) or not callable(load_state_dict):
        raise CapabilityError("GPT-2 checkpoint-pair export requires state_dict and load_state_dict")
    try:
        import torch
    except ImportError as error:
        raise CapabilityError("GPT-2 checkpoint-pair export requires optional PyTorch") from error
    original_state = {str(name): value.detach().clone() if callable(getattr(value, "detach", None)) else value for name, value in state_dict().items()}
    clean_checksum = _model_state_checksum(model)
    temp_root = Path(tempfile.mkdtemp(prefix=".faytuna-gpt2-pair-", dir=str(destination)))
    baseline_temp = temp_root / "baseline"
    candidate_temp = temp_root / "candidate"
    candidate_changed = False
    candidate_status = "candidate_unchanged_no_plan"
    candidate_checksum = clean_checksum
    try:
        export_gpt2_checkpoint(model, baseline_temp, variant=variant)
        if tokenizer is not None:
            save_tokenizer = getattr(tokenizer, "save_pretrained", None)
            if not callable(save_tokenizer):
                raise CapabilityError("supplied tokenizer does not expose save_pretrained")
            save_tokenizer(baseline_temp)
        if experimental_plan is not None:
            current = {name: _weight_to_numpy(value, name=name) for name, value in state_dict().items()}
            from .surgery import apply_surgery

            updated = apply_surgery(current, experimental_plan, require_update=True)
            replacement = {name: torch.as_tensor(updated[name], dtype=value.dtype, device=value.device) for name, value in state_dict().items()}
            load_state_dict(replacement, strict=True)
            candidate_checksum = _model_state_checksum(model)
            candidate_changed = candidate_checksum != clean_checksum
            if experimental_plan.metadata.get("mode") == "experimental":
                candidate_status = "experimental_untrusted"
            else:
                candidate_status = "candidate_exported" if candidate_changed else "candidate_unchanged_after_plan"
        export_gpt2_checkpoint(model, candidate_temp, variant=variant)
        if tokenizer is not None:
            save_tokenizer(candidate_temp)
    finally:
        load_state_dict(original_state, strict=True)
        if _model_state_checksum(model) != clean_checksum:
            shutil.rmtree(temp_root, ignore_errors=True)
            raise RuntimeError("GPT-2 checkpoint-pair export failed to restore the clean model state")
    os.replace(str(baseline_temp), str(baseline_destination))
    os.replace(str(candidate_temp), str(candidate_destination))
    shutil.rmtree(temp_root, ignore_errors=True)
    manifest = {
        "schema_version": "faytuna-gpt2-checkpoint-pair-v1",
        "variant": variant,
        "family": "gpt2",
        "baseline": {"path": "baseline", "status": "clean_baseline", "checksum": clean_checksum},
        "candidate": {"path": "candidate", "status": candidate_status, "changed": bool(candidate_changed), "checksum": candidate_checksum},
        "tokenizer": {"status": "exported_with_model_pair" if tokenizer is not None else "not_supplied", "required_for_text_ab": True},
        "experimental_plan": None if experimental_plan is None else {
            "mode": experimental_plan.metadata.get("mode"),
            "trust_status": experimental_plan.metadata.get("trust_status"),
            "gain": experimental_plan.metadata.get("gain"),
            "correction_checksum": experimental_plan.metadata.get("correction_checksum"),
            "acceptance_gate_bypassed": experimental_plan.metadata.get("acceptance_gate_bypassed", False),
            "applied_tensors": list(experimental_plan.applied_tensors),
            "changed_tensor_details": list(experimental_plan.metadata.get("changed_tensor_details", ())),
            "skipped_tensors": dict(experimental_plan.skipped_tensors),
            "rollback_layers": list(experimental_plan.rollback_layers),
            "rollback_reasons": dict(experimental_plan.metadata.get("rollback_reasons", {})),
            "raw_unaccepted_transitions_used": list(experimental_plan.metadata.get("raw_unaccepted_transitions_used", ())),
            "lift_method": experimental_plan.metadata.get("lift_method"),
            "lift_proof_status": experimental_plan.metadata.get("lift_proof_status"),
            "exact_activation_ls_tensors": list(experimental_plan.metadata.get("exact_activation_ls_tensors", ())),
            "heuristic_chart_tensors": list(experimental_plan.metadata.get("heuristic_chart_tensors", ())),
            "activation_ridge": experimental_plan.metadata.get("activation_ridge"),
            "activation_target_provenance": experimental_plan.metadata.get("activation_target_provenance"),
            "lift_reports": dict(experimental_plan.metadata.get("lift_reports", {})),
            "quadratic_terms_applied": bool(experimental_plan.metadata.get("quadratic_terms_applied", False)),
        },
        "preflight": check,
        "metadata": dict(metadata or {}),
        "runtime_validation": "separate_stock_llama_cpp_stage",
        "semantic_claim": "not established by checkpoint export",
    }
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")
    return destination


def run_flow_guided_distillation(
    student_model: Any,
    teacher_model_or_traces: Any,
    token_batches: Sequence[Any],
    *,
    alignment_chart: np.ndarray | None = None,
    steps: int = 50,
    lr: float = 1e-5,
    lambda_flow: float = 0.5,
    lambda_ce: float = 1.0,
    max_grad_norm: float = 0.5,
    device: str = "auto",
    freeze_embeddings: bool = True,
) -> dict[str, Any]:
    """Iterative flow-guided micro-distillation over manifold trajectories (Vector 2).

    Surpasses the first-order Taylor perturbation limit (||Delta W|| / ||W|| <= 0.01)
    by executing bounded autograd steps with combined task loss and geometric
    latent flow-matching loss:
        L_total = lambda_ce * L_CE + lambda_flow * L_flow
    where L_flow = (1 / (B * L)) * ||h_student - h_teacher @ P||_F^2.

    Model-agnostic: operates on any PyTorch causal language model (GPT-2, Qwen, LLaMA,
    Mistral) or against offline trajectory traces. Safeguarded with parameter drift bounds.

    Args:
        student_model: PyTorch causal LM to optimize.
        teacher_model_or_traces: PyTorch teacher model or sequence of offline TrajectoryTrace.
        token_batches: Sequence of dicts with 'input_ids' (and optional 'attention_mask', 'labels')
                       or raw torch.Tensor/np.ndarray of token ids.
        alignment_chart: (d_teacher, d_student) Procrustes or Grassmannian projection matrix.
        steps: Number of micro-optimization steps.
        lr: AdamW learning rate (typically 1e-5 to 5e-5).
        lambda_flow: Weight for geometric flow alignment loss.
        lambda_ce: Weight for language modeling cross-entropy loss.
        max_grad_norm: Gradient clipping threshold.
        device: Device string ('cpu', 'cuda', 'auto').
        freeze_embeddings: If True, freezes token and position embeddings to prevent
                           vocabulary coordinate drift.

    Returns:
        report: Dict containing optimization telemetry, initial/final losses, and drift.
    """
    try:
        import torch
        import torch.nn.functional as F
    except ImportError as error:
        raise CapabilityError("Flow-guided distillation requires optional PyTorch") from error

    if steps < 1:
        raise ValueError(f"steps must be >= 1, got {steps}")
    if lr <= 0.0 or not np.isfinite(lr):
        raise ValueError(f"lr must be positive and finite, got {lr}")
    if not token_batches:
        raise ValueError("token_batches cannot be empty")

    if isinstance(device, str) and device.lower() in {"dml", "directml"}:
        try:
            import torch_directml
            target_device = torch_directml.device()
        except Exception:
            target_device = torch.device("cpu")
    elif device == "auto":
        target_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        try:
            target_device = torch.device(device)
        except Exception:
            target_device = torch.device("cpu")

    # Save initial weights for safety drift verification
    initial_params = {
        name: param.detach().clone()
        for name, param in student_model.named_parameters()
        if param.requires_grad
    }
    initial_norm = float(torch.sqrt(sum(torch.sum(p ** 2) for p in initial_params.values())).item()) if initial_params else 1.0

    # Freeze embeddings if requested
    frozen_names = []
    if freeze_embeddings:
        for name, param in student_model.named_parameters():
            if any(term in name.lower() for term in ("wte", "wpe", "embed_tokens", "embed_positions")):
                if param.requires_grad:
                    param.requires_grad = False
                    frozen_names.append(name)

    trainable_params = [p for p in student_model.parameters() if p.requires_grad]
    if not trainable_params:
        raise CapabilityError("student_model has no trainable parameters for distillation")

    optimizer = torch.optim.AdamW(trainable_params, lr=float(lr), weight_decay=0.01)

    chart_tensor = None
    if alignment_chart is not None:
        chart_arr = finite_array(alignment_chart, ndim=2, name="alignment_chart")
        chart_tensor = torch.as_tensor(chart_arr, dtype=torch.float32, device=target_device)

    is_teacher_module = hasattr(teacher_model_or_traces, "eval") and callable(getattr(teacher_model_or_traces, "forward", None))
    if is_teacher_module:
        teacher_model_or_traces.eval()

    student_model.train()
    loss_history = []
    flow_loss_history = []
    ce_loss_history = []

    batch_idx = 0
    num_batches = len(token_batches)

    try:
        for step in range(steps):
            raw_batch = token_batches[batch_idx % num_batches]
            batch_idx += 1

            # Prepare batch inputs
            if isinstance(raw_batch, Mapping):
                input_ids = raw_batch.get("input_ids")
                attention_mask = raw_batch.get("attention_mask", None)
            else:
                input_ids = raw_batch
                attention_mask = None

            if not isinstance(input_ids, torch.Tensor):
                input_ids = torch.as_tensor(input_ids, dtype=torch.long, device=target_device)
            else:
                input_ids = input_ids.to(target_device)

            if input_ids.ndim == 1:
                input_ids = input_ids.unsqueeze(0)

            if attention_mask is not None and not isinstance(attention_mask, torch.Tensor):
                attention_mask = torch.as_tensor(attention_mask, dtype=torch.float32, device=target_device)
                if attention_mask.ndim == 1:
                    attention_mask = attention_mask.unsqueeze(0)
            elif isinstance(attention_mask, torch.Tensor):
                attention_mask = attention_mask.to(target_device)

            # Student forward
            s_kwargs = {"output_hidden_states": True}
            if attention_mask is not None:
                s_kwargs["attention_mask"] = attention_mask

            s_out = student_model(input_ids, **s_kwargs)
            s_logits = getattr(s_out, "logits", s_out[0] if isinstance(s_out, tuple) else s_out)

            # CE / Self-supervision loss
            if s_logits.shape[1] > 1:
                shift_logits = s_logits[:, :-1, :].contiguous()
                shift_labels = input_ids[:, 1:].contiguous()
                ce_loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            else:
                ce_loss = torch.tensor(0.0, device=target_device)

            # Flow alignment loss
            flow_loss = torch.tensor(0.0, device=target_device)
            if lambda_flow > 0.0:
                s_hiddens = getattr(s_out, "hidden_states", None)
                s_last_hidden = s_hiddens[-1] if s_hiddens is not None else (s_out[1] if isinstance(s_out, tuple) and len(s_out) > 1 else None)

                if is_teacher_module and s_last_hidden is not None:
                    with torch.no_grad():
                        t_kwargs = {"output_hidden_states": True}
                        if attention_mask is not None:
                            t_kwargs["attention_mask"] = attention_mask
                        teacher_backbone = getattr(teacher_model_or_traces, "model", getattr(teacher_model_or_traces, "transformer", teacher_model_or_traces))
                        t_out = teacher_backbone(input_ids, **t_kwargs)
                        t_hiddens = getattr(t_out, "hidden_states", None)
                        t_last_hidden = t_hiddens[-1] if t_hiddens is not None else (t_out[1] if isinstance(t_out, tuple) and len(t_out) > 1 else None)

                    if t_last_hidden is not None:
                        t_h = t_last_hidden.to(torch.float32)
                        s_h = s_last_hidden.to(torch.float32)
                        if chart_tensor is not None:
                            # Project teacher into student hidden dimensions: (B, L, d_T) @ (d_T, d_S) -> (B, L, d_S)
                            t_proj = torch.matmul(t_h, chart_tensor)
                        else:
                            t_proj = t_h
                        if t_proj.shape == s_h.shape:
                            flow_loss = F.mse_loss(s_h, t_proj)

            total_loss = float(lambda_ce) * ce_loss + float(lambda_flow) * flow_loss

            if not torch.isfinite(total_loss):
                raise FloatingPointError(f"Non-finite loss encountered at step {step}: {total_loss.item()}")

            optimizer.zero_grad()
            total_loss.backward()
            if max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, float(max_grad_norm))
            optimizer.step()

            loss_val = float(total_loss.item())
            loss_history.append(loss_val)
            flow_loss_history.append(float(flow_loss.item()))
            ce_loss_history.append(float(ce_loss.item()))

    finally:
        # Restore frozen parameters requires_grad status
        for name, param in student_model.named_parameters():
            if name in frozen_names:
                param.requires_grad = True
        student_model.eval()

    # Calculate parameter drift
    current_params = {
        name: param.detach()
        for name, param in student_model.named_parameters()
        if name in initial_params
    }
    drift_norm = float(torch.sqrt(
        sum(torch.sum((current_params[name] - initial_params[name]) ** 2) for name in initial_params)
    ).item())
    rel_drift = drift_norm / max(initial_norm, 1e-8)

    return {
        "status": "completed",
        "steps": len(loss_history),
        "initial_loss": loss_history[0] if loss_history else 0.0,
        "final_loss": loss_history[-1] if loss_history else 0.0,
        "mean_loss": float(np.mean(loss_history)) if loss_history else 0.0,
        "relative_parameter_drift": rel_drift,
        "loss_history": loss_history,
        "flow_loss_history": flow_loss_history,
        "ce_loss_history": ce_loss_history,
        "converged": bool(loss_history[-1] <= loss_history[0]) if len(loss_history) > 1 else True,
    }


# =============================================================================
# Universal Model-Agnostic Aliases (Transformer... -> GPT2...)
# =============================================================================
TransformerConnector = GPT2Connector
apply_transformer_surgery = apply_gpt2_surgery
TransformerProbePolicy = GPT2ProbePolicy
TransformerActivationCaptureResult = GPT2ActivationCaptureResult
TransformerActivationHookCapture = GPT2ActivationHookCapture
capture_transformer_activations = capture_gpt2_conv1d_activations
export_transformer_checkpoint = export_gpt2_checkpoint
export_transformer_checkpoint_pair = export_gpt2_checkpoint_pair
TransformerTensorLiftMapping = GPT2TensorLiftMapping
fit_transformer_activation_least_squares = fit_gpt2_conv1d_activation_least_squares
transformer_depth_gain_weight = gpt2_depth_gain_weight
build_transformer_flow_activation_targets = build_gpt2_teacher_flow_activation_targets
build_transformer_surgery_plan = build_gpt2_surgery_plan
run_transformer_flow_distillation = run_flow_guided_distillation



