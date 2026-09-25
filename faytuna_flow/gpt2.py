"""GPT-2 compatibility facade delegating to transformer_core.

Preserves 100% backward compatibility for all existing tests and scripts.
The universal implementation resides in `faytuna_flow.transformer_core`.
"""

from __future__ import annotations

from .transformer_core import *
from .transformer_core import (
    GPT2Connector,
    GPT2ActivationCaptureResult,
    GPT2ActivationHookCapture,
    GPT2ProbePolicy,
    GPT2TensorLiftMapping,
    apply_gpt2_surgery,
    build_gpt2_surgery_plan,
    build_gpt2_teacher_flow_activation_targets,
    capture_gpt2_conv1d_activations,
    compute_gpt2_static_weight_dissimilarity,
    compute_universal_static_weight_dissimilarity,
    estimate_pilot_pulse_gain_bound,
    export_gpt2_checkpoint,
    export_gpt2_checkpoint_pair,
    fit_gpt2_conv1d_activation_least_squares,
    gpt2_depth_gain_weight,
    make_gpt2_hooks,
    make_gpt2_probe_splits,
    make_gpt2_python_probe_splits,
    _gpt2_weight_contract,
    _lifted_hidden_delta,
    _weight_to_numpy,
    _correction_checksum,
)

__all__ = [
    "GPT2Connector",
    "GPT2ActivationCaptureResult",
    "GPT2ActivationHookCapture",
    "GPT2ProbePolicy",
    "GPT2TensorLiftMapping",
    "apply_gpt2_surgery",
    "build_gpt2_surgery_plan",
    "build_gpt2_teacher_flow_activation_targets",
    "capture_gpt2_conv1d_activations",
    "compute_gpt2_static_weight_dissimilarity",
    "compute_universal_static_weight_dissimilarity",
    "estimate_pilot_pulse_gain_bound",
    "export_gpt2_checkpoint",
    "export_gpt2_checkpoint_pair",
    "fit_gpt2_conv1d_activation_least_squares",
    "gpt2_depth_gain_weight",
    "make_gpt2_hooks",
    "make_gpt2_probe_splits",
    "make_gpt2_python_probe_splits",
]
