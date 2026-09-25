"""Capability-aware model connectors.

The numpy connector is used for deterministic research systems. The optional
PyTorch connector is intentionally explicit: a model family must provide the
probe encoder and state decoder that define the observed chart. A generic
``torch.nn.Module`` is never guessed to be sequence-safe merely because it has
an attribute called ``layers``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Protocol, Sequence

import numpy as np

from .types import CapabilityMatrix, Probe, finite_array


class CapabilityError(RuntimeError):
    """Raised when an operation needs a family callback that was not supplied."""


class DynamicalModel(Protocol):
    layer_count: int
    state_dim: int

    def initial_state(self, probe: Probe) -> np.ndarray: ...

    def step(self, state: np.ndarray, layer_index: int, probe: Probe) -> np.ndarray: ...


class Connector(Protocol):
    model_id: str
    capabilities: CapabilityMatrix
    layer_ids: tuple[int, ...]

    def initial_state(self, probe: Probe) -> np.ndarray: ...

    def transition(self, state: np.ndarray, layer_index: int, probe: Probe) -> np.ndarray: ...

    def normalization_geometry(self, state: np.ndarray, layer_index: int, probe: Probe) -> Mapping[str, Any] | None: ...

    def attention_geometry(self, state: np.ndarray, layer_index: int, probe: Probe) -> Mapping[str, Any] | None: ...

    def weights(self) -> Mapping[str, np.ndarray]: ...

    def set_weights(self, values: Mapping[str, np.ndarray]) -> None: ...


@dataclass
class SyntheticConnector:
    """Adapter for a pure numpy dynamical system used in research tests."""

    model: DynamicalModel
    model_id: str = "synthetic"
    layer_ids: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        self.feature_space = "synthetic_state"
        if self.layer_ids is None:
            self.layer_ids = tuple(range(int(self.model.layer_count)))
        if len(self.layer_ids) < 1:
            raise ValueError("a connector needs at least one visible layer")
        self.capabilities = CapabilityMatrix(
            hidden_states=True,
            residual_states=bool(getattr(self.model, "has_residual_state", False)),
            layer_transitions=True,
            vector_fields=True,
            jacobian_sketch=True,
            hessian_sketch=True,
            attention_geometry=hasattr(self.model, "attention_geometry"),
            normalization_geometry=hasattr(self.model, "normalization_geometry"),
            weight_surgery=hasattr(self.model, "weights") and hasattr(self.model, "set_weights"),
            reasons={"attention_geometry": "available only when the synthetic model exposes it"},
            automatic={"hidden_states": True, "jacobian_sketch": True, "hessian_sketch": True},
            required_callbacks={"attention_geometry": "model.attention_geometry when attention evidence is required"},
        )

    @property
    def state_dim(self) -> int:
        return int(self.model.state_dim)

    def initial_state(self, probe: Probe) -> np.ndarray:
        return finite_array(self.model.initial_state(probe), ndim=1, name="initial state")

    def transition(self, state: np.ndarray, layer_index: int, probe: Probe) -> np.ndarray:
        if layer_index < 0 or layer_index >= len(self.layer_ids):
            raise IndexError("layer index outside visible connector layers")
        return finite_array(self.model.step(finite_array(state, ndim=1, name="state"), int(self.layer_ids[layer_index]), probe), ndim=1, name="next state")

    def residual_state(self, state: np.ndarray, layer_index: int, probe: Probe) -> np.ndarray | None:
        fn = getattr(self.model, "residual_state", None)
        return None if fn is None else finite_array(fn(state, layer_index, probe), ndim=1, name="residual state")

    def normalization_geometry(self, state: np.ndarray, layer_index: int, probe: Probe) -> Mapping[str, Any] | None:
        fn = getattr(self.model, "normalization_geometry", None)
        return None if fn is None else dict(fn(state, layer_index, probe))

    def attention_geometry(self, state: np.ndarray, layer_index: int, probe: Probe) -> Mapping[str, Any] | None:
        fn = getattr(self.model, "attention_geometry", None)
        return None if fn is None else dict(fn(state, layer_index, probe))

    def weights(self) -> Mapping[str, np.ndarray]:
        fn = getattr(self.model, "weights", None)
        if fn is None:
            return {}
        return {str(k): finite_array(v, name=f"weight {k}").copy() for k, v in fn().items()}

    def set_weights(self, values: Mapping[str, np.ndarray]) -> None:
        fn = getattr(self.model, "set_weights", None)
        if fn is None:
            raise CapabilityError("synthetic model does not expose mutable weights")
        fn({str(k): finite_array(v, name=f"weight {k}") for k, v in values.items()})


@dataclass(frozen=True)
class TorchHooks:
    """Explicit chart and family hooks for a torch module.

    ``state_encoder`` returns a tensor shaped ``[batch, sequence, hidden]``
    (or another explicitly supported batch/sequence shape). ``state_observer``
    maps that tensor to a one-dimensional numpy chart vector. ``state_decoder``
    reverses that chart map and must reconstruct a tensor accepted by every
    layer. The default layer call receives only the tensor, matching ordinary
    ``torch.nn.Module`` semantics; pass ``layer_call`` for family-specific
    keyword arguments or cache objects.
    """

    state_encoder: Callable[[Probe, Any], Any]
    state_decoder: Callable[[np.ndarray, Probe], Any]
    state_observer: Callable[[Any], Any]
    layer_call: Callable[[Any, Any], Any] | None = None
    residual_observer: Callable[[Any, int, Probe], Any] | None = None
    normalization_observer: Callable[[Any, int, Probe], Mapping[str, Any]] | None = None
    attention_observer: Callable[[Any, int, Probe], Mapping[str, Any]] | None = None
    token_position_observer: Callable[[Probe], Mapping[str, Any]] | None = None
    weight_getter: Callable[[Any], Mapping[str, Any]] | None = None
    weight_setter: Callable[[Any, Mapping[str, Any]], None] | None = None
    layer_call_with_probe: Callable[[Any, Any, Probe], Any] | None = None
    jacobian_sketch: Callable[[Any, Any, Probe, np.ndarray, float], Any] | None = None
    hessian_sketch: Callable[[Any, Any, Probe, np.ndarray, float], Any] | None = None


def _torch_module_type() -> Any | None:
    try:
        import torch
        return torch.nn.Module
    except ImportError:
        return None


def _is_torch_module(value: Any) -> bool:
    module_type = _torch_module_type()
    return module_type is not None and isinstance(value, module_type)


def _find_layers(model: Any) -> Sequence[Any]:
    for path in ("layers", "transformer.h", "encoder.layer", "model.layers"):
        current = model
        try:
            for piece in path.split("."):
                current = getattr(current, piece)
            if len(current) > 0:
                return current
        except (AttributeError, TypeError):
            continue
    raise CapabilityError("could not locate visible layers; supply a family connector with explicit layer_ids")


def _unwrap_layer_output(output: Any) -> Any:
    if isinstance(output, Mapping):
        for key in ("last_hidden_state", "hidden_state", "hidden_states"):
            if key in output:
                output = output[key]
                break
    if isinstance(output, (tuple, list)):
        if not output:
            raise CapabilityError("layer returned an empty tuple/list")
        output = output[0]
    return output


class TorchConnector:
    """Optional real torch connector with explicit batch/sequence chart hooks."""

    def __init__(self, model: Any, hooks: TorchHooks, *, model_id: str | None = None, device: str | Any | None = None, layer_ids: Sequence[int] | None = None) -> None:
        if not _is_torch_module(model):
            raise TypeError("TorchConnector requires a torch.nn.Module; no checkpoint loading is performed")
        if not isinstance(hooks, TorchHooks):
            raise TypeError("TorchConnector requires TorchHooks with explicit state_encoder/state_decoder/state_observer")
        self.model = model
        self.hooks = hooks
        model.eval()
        self.feature_space = "torch_hidden_chart"
        self.model_id = model_id or model.__class__.__name__
        self.layers = _find_layers(model)
        self.layer_ids = tuple(int(x) for x in (layer_ids if layer_ids is not None else range(len(self.layers))))
        if not self.layer_ids or any(x < 0 or x >= len(self.layers) for x in self.layer_ids):
            raise ValueError("layer_ids must point to existing visible layers")
        try:
            import torch
            self._torch = torch
            if isinstance(device, str) and device.lower() in {"dml", "directml"}:
                try:
                    import torch_directml
                    self.device = torch_directml.device()
                except Exception:
                    self.device = torch.device("cpu")
            elif device is not None:
                try:
                    self.device = torch.device(device)
                except Exception:
                    self.device = torch.device("cpu")
            else:
                self.device = next(model.parameters(), torch.empty(0)).device

            if device is not None:
                try:
                    model.to(self.device)
                except BaseException:
                    self.device = torch.device("cpu")
                    model.to(self.device)
        except ImportError as error:
            raise CapabilityError("PyTorch is not installed; install the optional torch extra or use SyntheticConnector") from error
        self.capabilities = CapabilityMatrix(
            hidden_states=True,
            residual_states=hooks.residual_observer is not None,
            layer_transitions=True,
            vector_fields=True,
            jacobian_sketch=True,
            hessian_sketch=True,
            attention_geometry=hooks.attention_observer is not None,
            normalization_geometry=hooks.normalization_observer is not None,
            weight_surgery=hooks.weight_getter is not None and hooks.weight_setter is not None,
            token_position_correspondence=hooks.token_position_observer is not None,
            reasons={
                "state_chart": "explicit TorchHooks.state_encoder/state_decoder/state_observer",
                "residual_states": "requires TorchHooks.residual_observer",
                "attention_geometry": "requires TorchHooks.attention_observer",
                "normalization_geometry": "requires TorchHooks.normalization_observer",
                "weight_surgery": "requires TorchHooks.weight_getter and weight_setter",
                "token_position_correspondence": "requires TorchHooks.token_position_observer or probe payload token metadata",
            },
            automatic={"layer_discovery": True, "batch_sequence_forward": hooks.layer_call is None, "hidden_states": True, "jacobian_sketch": True, "hessian_sketch": True},
            required_callbacks={"state_chart": "state_encoder, state_decoder, and state_observer", "family_layer_call": "TorchHooks.layer_call for non-standard forward signatures", "jacobian_sketch": "optional TorchHooks.jacobian_sketch; otherwise directional finite differences", "hessian_sketch": "optional TorchHooks.hessian_sketch; otherwise directional finite differences"},
        )

    def _as_tensor(self, value: Any, *, name: str) -> Any:
        if not isinstance(value, self._torch.Tensor):
            raise CapabilityError(f"{name} callback must return a torch.Tensor")
        if value.ndim < 2:
            raise CapabilityError(f"{name} tensor must retain batch/sequence axes; got shape {tuple(value.shape)}")
        if not self._torch.is_floating_point(value):
            raise CapabilityError(f"{name} tensor must be floating point hidden state")
        if not self._torch.isfinite(value).all().item():
            raise FloatingPointError(f"{name} contains non-finite values")
        return value.to(self.device)

    def _observe(self, tensor: Any, *, name: str) -> np.ndarray:
        observed = self.hooks.state_observer(tensor)
        if isinstance(observed, self._torch.Tensor):
            observed = observed.detach().to("cpu").numpy()
        return finite_array(observed, ndim=1, name=name)

    def initial_state(self, probe: Probe) -> np.ndarray:
        with self._torch.no_grad():
            encoded = self._as_tensor(self.hooks.state_encoder(probe, self.device), name="state_encoder output")
            return self._observe(encoded, name="encoded observation")

    def _decode(self, state: np.ndarray, probe: Probe) -> Any:
        decoded = self.hooks.state_decoder(finite_array(state, ndim=1, name="state chart"), probe)
        return self._as_tensor(decoded, name="state_decoder output")

    def transition(self, state: np.ndarray, layer_index: int, probe: Probe) -> np.ndarray:
        if layer_index < 0 or layer_index >= len(self.layer_ids):
            raise IndexError("layer index outside visible torch layers")
        hidden = self._decode(state, probe)
        layer = self.layers[self.layer_ids[layer_index]]
        with self._torch.no_grad():
            if self.hooks.layer_call_with_probe is not None:
                output = self.hooks.layer_call_with_probe(layer, hidden, probe)
            else:
                output = self.hooks.layer_call(layer, hidden) if self.hooks.layer_call is not None else layer(hidden)
            output = self._as_tensor(_unwrap_layer_output(output), name=f"layer {layer_index} output")
            if output.shape != hidden.shape:
                raise CapabilityError(f"layer {layer_index} changed hidden shape from {tuple(hidden.shape)} to {tuple(output.shape)}; supply a state chart callback for this family")
            return self._observe(output, name="hidden observation")

    def residual_state(self, state: np.ndarray, layer_index: int, probe: Probe) -> np.ndarray | None:
        if self.hooks.residual_observer is None:
            return None
        value = self.hooks.residual_observer(self._decode(state, probe), layer_index, probe)
        return self._observe(self._as_tensor(value, name="residual observer output"), name="residual observation")

    def normalization_geometry(self, state: np.ndarray, layer_index: int, probe: Probe) -> Mapping[str, Any] | None:
        return None if self.hooks.normalization_observer is None else dict(self.hooks.normalization_observer(self._decode(state, probe), layer_index, probe))

    def attention_geometry(self, state: np.ndarray, layer_index: int, probe: Probe) -> Mapping[str, Any] | None:
        return None if self.hooks.attention_observer is None else dict(self.hooks.attention_observer(self._decode(state, probe), layer_index, probe))

    def jacobian_sketch(self, state: np.ndarray, layer_index: int, probe: Probe, directions: np.ndarray, step: float) -> np.ndarray | None:
        if self.hooks.jacobian_sketch is None:
            return None
        hidden = self._decode(state, probe)
        value = self.hooks.jacobian_sketch(self.layers[self.layer_ids[layer_index]], hidden, probe, directions, float(step))
        if isinstance(value, self._torch.Tensor):
            value = value.detach().to("cpu").numpy()
        return finite_array(value, ndim=2, name="TorchHooks Jacobian sketch")

    def hessian_sketch(self, state: np.ndarray, layer_index: int, probe: Probe, directions: np.ndarray, step: float) -> np.ndarray | None:
        if self.hooks.hessian_sketch is None:
            return None
        hidden = self._decode(state, probe)
        value = self.hooks.hessian_sketch(self.layers[self.layer_ids[layer_index]], hidden, probe, directions, float(step))
        if isinstance(value, self._torch.Tensor):
            value = value.detach().to("cpu").numpy()
        return finite_array(value, ndim=2, name="TorchHooks Hessian sketch")

    def token_position_metadata(self, probe: Probe) -> Mapping[str, Any]:
        return {} if self.hooks.token_position_observer is None else dict(self.hooks.token_position_observer(probe))

    def weights(self) -> Mapping[str, np.ndarray]:
        if self.hooks.weight_getter is None:
            raise CapabilityError("weight surgery is unavailable: supply TorchHooks.weight_getter")
        values = self.hooks.weight_getter(self.model)
        result = {}
        for name, value in values.items():
            if not isinstance(value, self._torch.Tensor):
                raise CapabilityError(f"weight_getter returned non-tensor {name}")
            result[str(name)] = finite_array(value.detach().to("cpu").numpy(), name=f"weight {name}").copy()
        return result

    def set_weights(self, values: Mapping[str, np.ndarray]) -> None:
        if self.hooks.weight_setter is None:
            raise CapabilityError("weight surgery is unavailable: supply TorchHooks.weight_setter")
        tensors = {str(name): self._torch.as_tensor(finite_array(value, name=f"weight {name}"), device=self.device) for name, value in values.items()}
        self.hooks.weight_setter(self.model, tensors)


class HFLikeConnector:
    """Framework-neutral connector for non-torch fake/HF-like layers.

    Passing a torch module without explicit hooks fails with a directed error,
    preventing the old numpy-plus-probe call path. Use ``TorchConnector`` with
    ``TorchHooks`` (or pass ``hooks=...`` to this constructor).
    """

    def __new__(cls, model: Any, model_id: str | None = None, *, hooks: TorchHooks | None = None, device: str | Any | None = None):
        if _is_torch_module(model):
            if hooks is None:
                raise CapabilityError("torch.nn.Module detected: use TorchConnector(model, TorchHooks(...)) with explicit state_encoder/state_decoder/state_observer")
            return TorchConnector(model, hooks, model_id=model_id, device=device)
        return super().__new__(cls)

    def __init__(self, model: Any, model_id: str | None = None, *, hooks: TorchHooks | None = None, device: str | Any | None = None) -> None:
        self.model = model
        self.model_id = model_id or model.__class__.__name__
        self.feature_space = "hf_like_hidden"
        self.layers = _find_layers(model)
        self.layer_ids = tuple(range(len(self.layers)))
        transition_automatic = all(hasattr(layer, "transition") or hasattr(layer, "forward") or callable(layer) for layer in self.layers)
        state_automatic = hasattr(model, "initial_state") or hasattr(model, "embed_probe")
        self.capabilities = CapabilityMatrix(
            hidden_states=True,
            residual_states=hasattr(model, "residual_states") or hasattr(model, "capture_residuals"),
            layer_transitions=transition_automatic,
            vector_fields=transition_automatic,
            jacobian_sketch=transition_automatic,
            hessian_sketch=transition_automatic,
            attention_geometry=hasattr(model, "attention_geometry"),
            normalization_geometry=hasattr(model, "normalization_geometry"),
            weight_surgery=hasattr(model, "get_weights") and hasattr(model, "set_weights"),
            reasons={"residual_states": "requires model.residual_states or capture_residuals hook", "attention_geometry": "requires an explicit attention_geometry hook"},
            automatic={"layer_discovery": True, "state_encoder": state_automatic, "layer_transitions": transition_automatic, "jacobian_sketch": transition_automatic, "hessian_sketch": transition_automatic},
            required_callbacks={"state_encoder": "model.initial_state or model.embed_probe; otherwise the probe chart is used as a declared fallback", "family_layer_call": "each visible layer must expose transition/forward/callable behavior; torch modules require TorchConnector/TorchHooks"},
        )

    def initial_state(self, probe: Probe) -> np.ndarray:
        if hasattr(self.model, "initial_state"):
            return finite_array(self.model.initial_state(probe), ndim=1, name="initial state")
        if hasattr(self.model, "embed_probe"):
            return finite_array(self.model.embed_probe(probe), ndim=1, name="embedded probe")
        return finite_array(probe.initial_state, ndim=1, name="probe initial state")

    def transition(self, state: np.ndarray, layer_index: int, probe: Probe) -> np.ndarray:
        layer = self.layers[layer_index]
        x = finite_array(state, ndim=1, name="state")
        if hasattr(layer, "transition"):
            out = layer.transition(x, probe)
        elif callable(layer):
            out = layer(x, probe)
        elif hasattr(layer, "forward"):
            out = layer.forward(x, probe)
        else:
            raise CapabilityError(f"layer {layer_index} exposes no explicit transition/forward callable")
        return finite_array(_unwrap_layer_output(out), ndim=1, name="next state")

    def residual_state(self, state: np.ndarray, layer_index: int, probe: Probe) -> np.ndarray | None:
        if hasattr(self.model, "capture_residuals"):
            return finite_array(self.model.capture_residuals(state, layer_index, probe), ndim=1, name="residual state")
        if hasattr(self.model, "residual_states"):
            return finite_array(self.model.residual_states[layer_index], ndim=1, name="residual state")
        return None

    def normalization_geometry(self, state: np.ndarray, layer_index: int, probe: Probe) -> Mapping[str, Any] | None:
        fn = getattr(self.model, "normalization_geometry", None)
        return None if fn is None else dict(fn(state, layer_index, probe))

    def attention_geometry(self, state: np.ndarray, layer_index: int, probe: Probe) -> Mapping[str, Any] | None:
        fn = getattr(self.model, "attention_geometry", None)
        return None if fn is None else dict(fn(state, layer_index, probe))

    def weights(self) -> Mapping[str, np.ndarray]:
        if not hasattr(self.model, "get_weights"):
            return {}
        return {str(k): finite_array(v, name=f"weight {k}").copy() for k, v in self.model.get_weights().items()}

    def set_weights(self, values: Mapping[str, np.ndarray]) -> None:
        if not hasattr(self.model, "set_weights"):
            raise CapabilityError("HF-like object does not expose get_weights/set_weights")
        self.model.set_weights({str(k): finite_array(v, name=f"weight {k}") for k, v in values.items()})
