"""Unit tests for universal model-agnostic architecture inspection, mapping, and dissimilarity."""

from __future__ import annotations

import numpy as np
import pytest

from faytuna_flow.model_families import (
    TransformerArchitecture,
    generate_universal_mapping,
    inspect_transformer_architecture,
)
from faytuna_flow.gpt2 import (
    compute_universal_static_weight_dissimilarity,
    compute_gpt2_static_weight_dissimilarity,
    GPT2TensorLiftMapping,
)


def test_inspect_transformer_architecture_gpt2():
    config = {
        "model_type": "gpt2",
        "n_layer": 12,
        "n_embd": 768,
        "n_inner": 3072,
        "n_head": 12,
        "n_ctx": 1024,
        "vocab_size": 50257,
    }
    arch = inspect_transformer_architecture(config)
    assert arch.model_type == "gpt2"
    assert arch.layers == 12
    assert arch.hidden_size == 768
    assert arch.intermediate_size == 3072
    assert arch.is_conv1d is True
    assert arch.layer_prefix == "transformer.h"
    assert arch.attn_proj_suffix == "attn.c_proj"
    assert arch.mlp_proj_suffix == "mlp.c_proj"


def test_inspect_transformer_architecture_llama():
    config = {
        "model_type": "llama",
        "num_hidden_layers": 32,
        "hidden_size": 4096,
        "intermediate_size": 11008,
        "num_attention_heads": 32,
        "max_position_embeddings": 2048,
        "vocab_size": 32000,
    }
    arch = inspect_transformer_architecture(config)
    assert arch.model_type == "llama"
    assert arch.layers == 32
    assert arch.hidden_size == 4096
    assert arch.intermediate_size == 11008
    assert arch.is_conv1d is False
    assert arch.layer_prefix == "model.layers"
    assert arch.attn_proj_suffix == "self_attn.o_proj"
    assert arch.mlp_proj_suffix == "mlp.down_proj"


def test_inspect_transformer_architecture_qwen():
    config = {
        "model_type": "qwen2",
        "num_hidden_layers": 28,
        "hidden_size": 3584,
        "intermediate_size": 18944,
        "num_attention_heads": 28,
        "max_position_embeddings": 32768,
        "vocab_size": 152064,
    }
    arch = inspect_transformer_architecture(config)
    assert "qwen" in arch.model_type
    assert arch.layers == 28
    assert arch.hidden_size == 3584
    assert arch.is_conv1d is False
    assert arch.layer_prefix == "model.layers"


def test_inspect_transformer_architecture_qwen35():
    config = {
        "model_type": "qwen3_5",
        "text_config": {
            "hidden_size": 1024,
            "num_hidden_layers": 24,
            "intermediate_size": 3584,
            "num_attention_heads": 8,
            "max_position_embeddings": 262144,
            "vocab_size": 248320,
            "layer_types": ["linear_attention", "linear_attention", "linear_attention", "full_attention"],
        },
    }
    arch = inspect_transformer_architecture(config)
    assert arch.model_type == "qwen3_5"
    assert arch.layers == 24
    assert arch.hidden_size == 1024
    assert arch.intermediate_size == 3584
    assert arch.layer_prefix == "model.language_model.layers"
    assert arch.is_conv1d is False

    teacher_config = {
        "model_type": "qwen3_5",
        "text_config": {
            "hidden_size": 2560,
            "num_hidden_layers": 32,
            "intermediate_size": 9216,
            "layer_types": ["linear_attention", "linear_attention", "linear_attention", "full_attention"],
        },
    }
    t_arch = inspect_transformer_architecture(teacher_config)
    mapping, _ = generate_universal_mapping(arch, t_arch)
    assert mapping[0]["tensor_name"] == "model.language_model.layers.0.linear_attn.out_proj.weight"
    assert mapping[1]["tensor_name"] == "model.language_model.layers.0.mlp.down_proj.weight"
    assert mapping[6]["tensor_name"] == "model.language_model.layers.3.self_attn.o_proj.weight"
    assert mapping[7]["tensor_name"] == "model.language_model.layers.3.mlp.down_proj.weight"


def test_inspect_transformer_architecture_custom():
    config = {
        "model_type": "custom_bert",
        "layers": 6,
        "d_model": 512,
        "feedforward_dim": 2048,
        "num_heads": 8,
    }
    arch = inspect_transformer_architecture(config)
    assert arch.layers == 6
    assert arch.hidden_size == 512
    assert arch.intermediate_size == 2048
    assert arch.attention_heads == 8


def test_generate_universal_mapping_arbitrary_layers():
    student = TransformerArchitecture(
        model_type="gpt2",
        layers=6,
        hidden_size=512,
        intermediate_size=2048,
        attention_heads=8,
        context_length=1024,
        vocab_size=50257,
        layer_prefix="transformer.h",
        attn_proj_suffix="attn.c_proj",
        mlp_proj_suffix="mlp.c_proj",
        is_conv1d=True,
    )
    teacher = TransformerArchitecture(
        model_type="gpt2",
        layers=24,
        hidden_size=1024,
        intermediate_size=4096,
        attention_heads=16,
        context_length=1024,
        vocab_size=50257,
        layer_prefix="transformer.h",
        attn_proj_suffix="attn.c_proj",
        mlp_proj_suffix="mlp.c_proj",
        is_conv1d=True,
    )

    mapping_entries, teacher_map = generate_universal_mapping(student, teacher)
    # 6 student blocks * 2 sites (attn + mlp) = 12 entries
    assert len(mapping_entries) == 12
    assert len(teacher_map) == 6
    # Layer 0 maps to 0, Layer 5 maps to 23
    assert teacher_map[0] == 0
    assert teacher_map[5] == 23

    # Check mapping contents
    assert mapping_entries[0]["block_index"] == 0
    assert mapping_entries[0]["tensor_name"] == "transformer.h.0.attn.c_proj.weight"
    assert mapping_entries[1]["tensor_name"] == "transformer.h.0.mlp.c_proj.weight"
    assert mapping_entries[-1]["block_index"] == 5
    assert mapping_entries[-1]["tensor_name"] == "transformer.h.5.mlp.c_proj.weight"


def test_generate_universal_mapping_llama_style():
    student = TransformerArchitecture(
        model_type="llama",
        layers=8,
        hidden_size=1024,
        intermediate_size=4096,
        attention_heads=8,
        context_length=2048,
        vocab_size=32000,
        layer_prefix="model.layers",
        attn_proj_suffix="self_attn.o_proj",
        mlp_proj_suffix="mlp.down_proj",
        is_conv1d=False,
    )
    teacher = TransformerArchitecture(
        model_type="llama",
        layers=32,
        hidden_size=4096,
        intermediate_size=11008,
        attention_heads=32,
        context_length=2048,
        vocab_size=32000,
        layer_prefix="model.layers",
        attn_proj_suffix="self_attn.o_proj",
        mlp_proj_suffix="mlp.down_proj",
        is_conv1d=False,
    )

    mapping_entries, teacher_map = generate_universal_mapping(student, teacher)
    assert len(mapping_entries) == 16
    assert teacher_map[0] == 0
    assert teacher_map[7] == 31
    assert mapping_entries[0]["tensor_name"] == "model.layers.0.self_attn.o_proj.weight"
    assert mapping_entries[1]["tensor_name"] == "model.layers.0.mlp.down_proj.weight"


def test_compute_universal_static_weight_dissimilarity_mock():
    ds, dt = 8, 16
    ls, lt = 4, 12

    student_arch = TransformerArchitecture(
        model_type="custom",
        layers=ls,
        hidden_size=ds,
        intermediate_size=4 * ds,
        attention_heads=2,
        context_length=64,
        vocab_size=1000,
        layer_prefix="h",
        attn_proj_suffix="attn.c_proj",
        mlp_proj_suffix="mlp.c_proj",
    )
    teacher_arch = TransformerArchitecture(
        model_type="custom",
        layers=lt,
        hidden_size=dt,
        intermediate_size=4 * dt,
        attention_heads=4,
        context_length=64,
        vocab_size=1000,
        layer_prefix="h",
        attn_proj_suffix="attn.c_proj",
        mlp_proj_suffix="mlp.c_proj",
    )

    np.random.seed(42)
    student_dict = {
        f"h.{i}.attn.c_proj.weight": np.random.randn(ds, ds)
        for i in range(ls)
    }
    teacher_dict = {
        f"h.{i}.attn.c_proj.weight": np.random.randn(dt, dt)
        for i in range(lt)
    }

    proj = np.random.randn(dt, ds)

    res = compute_universal_static_weight_dissimilarity(
        student_dict,
        teacher_dict,
        proj,
        student_arch=student_arch,
        teacher_arch=teacher_arch,
        contrast=0.40,
        min_gain=0.25,
        max_gain=2.0,
    )

    assert res["student_blocks"] == ls
    assert res["teacher_blocks"] == lt
    assert res["student_dim"] == ds
    assert res["teacher_dim"] == dt
    assert len(res["distances"]) == ls
    assert len(res["schedule"]) == ls
    assert all(0.25 <= g <= 2.0 for g in res["schedule"])
    assert res["teacher_layer_map"][0] == 0
    assert res["teacher_layer_map"][ls - 1] == lt - 1
