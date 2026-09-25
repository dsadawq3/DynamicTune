"""Explicit model-family presets and architecture preflight contracts.

The first practical target is the GPT-2 XL to GPT-2 small pair.  This module
only inspects a supplied config or an already constructed model; it never
downloads weights or constructs a checkpoint.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np


@dataclass(frozen=True)
class ModelVariantSpec:
    name: str
    model_type: str
    layers: int
    hidden_size: int
    attention_heads: int
    context_length: int
    vocab_size: int
    repository_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "model_type": self.model_type,
            "layers": self.layers,
            "hidden_size": self.hidden_size,
            "attention_heads": self.attention_heads,
            "context_length": self.context_length,
            "vocab_size": self.vocab_size,
            "repository_id": self.repository_id,
        }


@dataclass(frozen=True)
class ModelFamilyPreset:
    name: str
    description: str
    teacher: ModelVariantSpec
    student: ModelVariantSpec
    observation_requirements: tuple[str, ...]
    runtime_requirements: tuple[str, ...]
    unsupported_claims: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "teacher": self.teacher.to_dict(),
            "student": self.student.to_dict(),
            "observation_requirements": list(self.observation_requirements),
            "runtime_requirements": list(self.runtime_requirements),
            "unsupported_claims": list(self.unsupported_claims),
        }


GPT2_XL_TO_SMALL = ModelFamilyPreset(
    name="gpt2-xl-to-small",
    description="Observation-first GPT-2 XL teacher to GPT-2 small student transfer.",
    teacher=ModelVariantSpec("gpt2-xl", "gpt2", 48, 1600, 25, 1024, 50257, "openai-community/gpt2-xl"),
    student=ModelVariantSpec("gpt2-small", "gpt2", 12, 768, 12, 1024, 50257, "openai-community/gpt2"),
    observation_requirements=(
        "explicit tokenization/probe policy",
        "sequence-preserving GPT2Connector state chart",
        "teacher and student traces collected on identical paired probe splits",
    ),
    runtime_requirements=(
        "export an ordinary GPT-2 checkpoint after any approved surgery",
        "convert with the unmodified llama.cpp GPT-2 converter",
        "validate with the unmodified llama-cli and llama-perplexity binaries",
    ),
    unsupported_claims=(
        "ordinary llama-cli does not expose hidden-state traces",
        "alignment is not a semantic proof",
        "a compatible GGUF conversion is not evidence of causal transfer",
    ),
)

QWEN35_4B_TO_08B = ModelFamilyPreset(
    name="qwen35-4b-to-08b",
    description="Observation-first Qwen 3.5 4B teacher to Qwen 3.5 0.8B student transfer.",
    teacher=ModelVariantSpec("qwen3.5-4b-base", "qwen3_5", 32, 2560, 16, 262144, 248320, "Qwen/Qwen3.5-4B-Base"),
    student=ModelVariantSpec("qwen3.5-0.8b-base", "qwen3_5", 24, 1024, 8, 262144, 248320, "Qwen/Qwen3.5-0.8B-Base"),
    observation_requirements=(
        "explicit tokenization/probe policy",
        "Qwen 3.5 state chart over language backbone",
        "teacher and student traces collected on identical paired probe splits",
    ),
    runtime_requirements=(
        "export an ordinary Qwen 3.5 safetensors checkpoint after surgery",
        "verify with standard causal LM / conditional generation",
    ),
    unsupported_claims=(
        "multimodal early fusion requires separate vision token evaluation",
        "alignment is not an empirical semantic proof",
    ),
)

MODEL_FAMILY_PRESETS: Mapping[str, ModelFamilyPreset] = {
    GPT2_XL_TO_SMALL.name: GPT2_XL_TO_SMALL,
    "gpt2": GPT2_XL_TO_SMALL,
    QWEN35_4B_TO_08B.name: QWEN35_4B_TO_08B,
    "qwen35": QWEN35_4B_TO_08B,
    "qwen3.5": QWEN35_4B_TO_08B,
    "qwen": QWEN35_4B_TO_08B,
}


def _as_mapping(config: Any) -> Mapping[str, Any]:
    if isinstance(config, Mapping):
        return config
    if isinstance(config, (str, Path)):
        path = Path(config)
        if path.is_dir():
            config_file = path / "config.json"
            if config_file.exists():
                return json.loads(config_file.read_text(encoding="utf-8"))
        elif path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    to_dict = getattr(config, "to_dict", None)
    if callable(to_dict):
        value = to_dict()
        if isinstance(value, Mapping):
            return value
    values = getattr(config, "__dict__", None)
    if isinstance(values, Mapping):
        return values
    raise TypeError("Transformer config must be a mapping, JSON path, or config object")


def _first(config: Mapping[str, Any], *names: str) -> Any:
    for name in names:
        if name in config:
            return config[name]
    return None


def inspect_gpt2_variant(config: Any, expected: ModelVariantSpec) -> dict[str, Any]:
    values = _as_mapping(config)
    checks: dict[str, bool] = {}
    observed: dict[str, Any] = {}
    model_type = _first(values, "model_type")
    observed["model_type"] = model_type
    checks["model_type"] = model_type == expected.model_type
    aliases = {
        "layers": ("n_layer", "num_hidden_layers"),
        "hidden_size": ("n_embd", "hidden_size"),
        "attention_heads": ("n_head", "num_attention_heads"),
        "context_length": ("n_ctx", "n_positions", "max_position_embeddings"),
        "vocab_size": ("vocab_size",),
    }
    for field_name, names in aliases.items():
        value = _first(values, *names)
        observed[field_name] = value
        checks[field_name] = value is not None and int(value) == int(getattr(expected, field_name))
    missing_or_wrong = [name for name, passed in checks.items() if not passed]
    return {
        "variant": expected.name,
        "expected": expected.to_dict(),
        "observed": observed,
        "checks": checks,
        "supported": not missing_or_wrong,
        "reasons": [] if not missing_or_wrong else [f"GPT-2 {expected.name} check failed: {', '.join(missing_or_wrong)}"],
    }


def preflight_gpt2_pair(teacher_config: Any, student_config: Any, preset: ModelFamilyPreset = GPT2_XL_TO_SMALL) -> dict[str, Any]:
    teacher = inspect_gpt2_variant(teacher_config, preset.teacher)
    student = inspect_gpt2_variant(student_config, preset.student)
    reasons = list(teacher["reasons"]) + list(student["reasons"])
    return {
        "schema_version": "faytuna-gpt2-preflight-v1",
        "preset": preset.name,
        "status": "pass" if not reasons else "rejected",
        "observation_ready": not reasons,
        "runtime_validation": "separate_llama_cpp_stage",
        "teacher": teacher,
        "student": student,
        "reasons": reasons,
        "semantic_claim": "not established by architecture preflight",
    }


def get_model_family_preset(name: str) -> ModelFamilyPreset:
    try:
        return MODEL_FAMILY_PRESETS[str(name).lower()]
    except KeyError as error:
        raise ValueError(f"unknown model-family preset {name!r}; available: {sorted(set(MODEL_FAMILY_PRESETS))}") from error


def strict_json_payload(value: Any) -> Any:
    """Normalize optional preflight values without allowing non-standard JSON."""

    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): strict_json_payload(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [strict_json_payload(item) for item in value]
    if isinstance(value, np.ndarray):
        return strict_json_payload(value.tolist())
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    return value


@dataclass(frozen=True)
class TransformerArchitecture:
    model_type: str
    layers: int
    hidden_size: int
    intermediate_size: int
    attention_heads: int
    context_length: int
    vocab_size: int
    layer_prefix: str
    attn_proj_suffix: str
    mlp_proj_suffix: str
    is_conv1d: bool = False
    raw_config: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_type": self.model_type,
            "layers": self.layers,
            "hidden_size": self.hidden_size,
            "intermediate_size": self.intermediate_size,
            "attention_heads": self.attention_heads,
            "context_length": self.context_length,
            "vocab_size": self.vocab_size,
            "layer_prefix": self.layer_prefix,
            "attn_proj_suffix": self.attn_proj_suffix,
            "mlp_proj_suffix": self.mlp_proj_suffix,
            "is_conv1d": self.is_conv1d,
        }


def inspect_transformer_architecture(config_or_path_or_model: Any) -> TransformerArchitecture:
    """Inspect and detect transformer architecture automatically without hardcoding.

    Detects layers, hidden dimensions, heads, and tensor naming conventions for any
    HuggingFace or PyTorch model (GPT-2, LLaMA, Mistral, Qwen, Gemma, Phi, BERT, Falcon, etc.).
    """
    if hasattr(config_or_path_or_model, "config"):
        raw = getattr(config_or_path_or_model, "config")
    else:
        raw = config_or_path_or_model
    values = dict(_as_mapping(raw))

    # For multimodal and hybrid architectures (e.g. Qwen 3.5), extract language backbone parameters
    text_cfg = values.get("text_config") or values.get("llm_config")
    if isinstance(text_cfg, Mapping):
        for k in (
            "num_hidden_layers",
            "hidden_size",
            "intermediate_size",
            "num_attention_heads",
            "max_position_embeddings",
            "vocab_size",
            "layer_types",
            "full_attention_interval",
        ):
            if k in text_cfg and text_cfg[k] is not None:
                values[k] = text_cfg[k]

    layers = _first(values, "num_hidden_layers", "n_layer", "num_layers", "n_layers", "layers")
    if layers is None:
        layers = 12
    layers = int(layers)

    hidden_size = _first(values, "hidden_size", "n_embd", "d_model")
    if hidden_size is None:
        hidden_size = 768
    hidden_size = int(hidden_size)

    intermediate_size = _first(values, "intermediate_size", "n_inner", "mlp_dim", "feedforward_dim")
    if intermediate_size is None:
        intermediate_size = 4 * hidden_size
    intermediate_size = int(intermediate_size)

    heads = _first(values, "num_attention_heads", "n_head", "num_heads", "n_heads")
    if heads is None:
        heads = max(1, hidden_size // 64)
    heads = int(heads)

    ctx = _first(values, "max_position_embeddings", "n_positions", "n_ctx", "seq_length", "max_sequence_length")
    if ctx is None:
        ctx = 1024
    ctx = int(ctx)

    vocab = _first(values, "vocab_size")
    if vocab is None:
        vocab = 50257
    vocab = int(vocab)

    mtype = str(_first(values, "model_type") or "gpt2").lower()

    if "qwen3_5" in mtype or "qwen3.5" in mtype or "qwen35" in mtype or (isinstance(text_cfg, Mapping) and "linear_attention" in str(text_cfg.get("layer_types", []))):
        layer_prefix = "model.language_model.layers"
        attn_proj_suffix = "self_attn.o_proj"
        mlp_proj_suffix = "mlp.down_proj"
    elif any(k in mtype for k in ("llama", "mistral", "qwen", "gemma", "phi", "yi", "deepseek")):
        layer_prefix = "model.layers"
        attn_proj_suffix = "self_attn.o_proj"
        mlp_proj_suffix = "mlp.down_proj"
    elif any(k in mtype for k in ("bert", "roberta")):
        layer_prefix = "bert.encoder.layer"
        attn_proj_suffix = "attention.output.dense"
        mlp_proj_suffix = "output.dense"
    elif "opt" in mtype:
        layer_prefix = "model.decoder.layers"
        attn_proj_suffix = "self_attn.out_proj"
        mlp_proj_suffix = "fc2"
    else:
        layer_prefix = "transformer.h"
        attn_proj_suffix = "attn.c_proj"
        mlp_proj_suffix = "mlp.c_proj"

    clean_config = {k: strict_json_payload(v) for k, v in values.items() if not k.startswith("_")}

    return TransformerArchitecture(
        model_type=mtype,
        layers=layers,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        attention_heads=heads,
        context_length=ctx,
        vocab_size=vocab,
        layer_prefix=layer_prefix,
        attn_proj_suffix=attn_proj_suffix,
        mlp_proj_suffix=mlp_proj_suffix,
        is_conv1d=(mtype == "gpt2"),
        raw_config=clean_config,
    )


def generate_universal_mapping(
    student_arch: TransformerArchitecture,
    teacher_arch: TransformerArchitecture,
    *,
    total_transitions: int | None = None,
) -> tuple[list[dict[str, Any]], dict[int, int]]:
    """Generate universal block-to-block layer correspondence and tensor lift mapping."""
    ls = student_arch.layers
    lt = teacher_arch.layers
    if total_transitions is None:
        total_transitions = ls

    teacher_layer_map: dict[int, int] = {}
    mapping_entries: list[dict[str, Any]] = []

    for b_idx in range(ls):
        t_idx = int(round(b_idx * (lt - 1) / max(1, ls - 1)))
        teacher_layer_map[b_idx] = t_idx
        trans_idx = int(round(b_idx * (total_transitions - 1) / max(1, ls - 1)))

        student_layer_types = student_arch.raw_config.get("layer_types") or []
        if b_idx < len(student_layer_types) and student_layer_types[b_idx] == "linear_attention":
            attn_suffix = "linear_attn.out_proj"
        else:
            attn_suffix = student_arch.attn_proj_suffix

        attn_name = f"{student_arch.layer_prefix}.{b_idx}.{attn_suffix}.weight"
        mapping_entries.append({
            "transition_index": trans_idx,
            "tensor_name": attn_name,
            "side": "output",
            "block_index": b_idx,
        })

        mlp_name = f"{student_arch.layer_prefix}.{b_idx}.{student_arch.mlp_proj_suffix}.weight"
        mapping_entries.append({
            "transition_index": trans_idx,
            "tensor_name": mlp_name,
            "side": "output",
            "block_index": b_idx,
        })

    return mapping_entries, teacher_layer_map

