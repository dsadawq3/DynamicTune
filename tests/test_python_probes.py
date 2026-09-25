import ast
import numpy as np
import pytest

from faytuna_flow.probes import ProbeGeneratorConfig, PythonProbeGenerator
from faytuna_flow.gpt2 import GPT2ProbePolicy, make_gpt2_python_probe_splits


def test_python_probe_generator_families_and_ast_syntax():
    generator = PythonProbeGenerator(ProbeGeneratorConfig(state_dim=6, per_family=4, seed=42))
    probes = generator.generate()
    families = {p.family for p in probes}
    assert families == set(PythonProbeGenerator.families)

    for probe in probes:
        assert "code" in probe.payload
        code_str = probe.payload["code"]
        assert isinstance(code_str, str)
        assert len(code_str.strip()) > 0
        parsed = ast.parse(code_str)
        assert parsed is not None


def test_python_probe_paired_counterfactuals_and_splits():
    generator = PythonProbeGenerator(ProbeGeneratorConfig(state_dim=6, per_family=5, seed=42))
    probes = generator.generate()
    split = generator.split(probes)

    all_probes = split.all()
    assert len(all_probes) == len(probes)
    assert len(split.train) > 0
    assert len(split.validation) > 0
    assert len(split.holdout) > 0

    paired_buckets = {}
    for probe in all_probes:
        if probe.pair_id is not None:
            paired_buckets.setdefault(probe.pair_id, set()).add(probe.split)
    assert len(paired_buckets) > 0
    assert all(len(buckets) == 1 for buckets in paired_buckets.values())


def test_make_gpt2_python_probe_splits_materialization():
    seq_len = 16
    policy = GPT2ProbePolicy(
        encode=lambda probe: {
            "input_ids": [100 + i for i in range(seq_len)],
            "position_ids": list(range(seq_len)),
            "attention_mask": [1.0] * seq_len,
        },
        sequence_length=seq_len,
    )
    split = make_gpt2_python_probe_splits(policy, per_family=3, seed=123, state_dim=4)
    for probe in split.train:
        assert "input_ids" in probe.payload
        assert len(probe.payload["input_ids"]) == seq_len
        assert probe.payload["gpt2_sequence_length"] == seq_len
