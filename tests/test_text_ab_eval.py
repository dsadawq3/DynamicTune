import json
from types import SimpleNamespace

import pytest

from scripts.text_ab_eval import DEFAULT_TUNE_CASES, _target_token_positions, evaluate_loaded_model, fingerprint_tokenizer, run_ab


torch = pytest.importorskip("torch")


class _TinyTokenizer:
    pad_token_id = 0
    eos_token_id = 0
    init_kwargs = {"name": "tiny"}
    _vocab = {"a": 1, "b": 2, "c": 3, "d": 4}

    def get_vocab(self):
        return dict(self._vocab)

    def __call__(self, text, return_tensors="pt", add_special_tokens=False):
        tokens = [part for part in text.split(" ") if part]
        ids = [self._vocab[token] for token in tokens]
        return {"input_ids": torch.tensor([ids], dtype=torch.long), "attention_mask": torch.ones((1, len(ids)), dtype=torch.long)}

    def decode(self, ids, skip_special_tokens=False):
        inverse = {value: key for key, value in self._vocab.items()}
        return " ".join(inverse[int(value)] for value in ids if int(value) != self.pad_token_id)


class _PathIndependentAddedToken:
    def __init__(self, content):
        self.content = content
        self.lstrip = False
        self.rstrip = False
        self.single_word = False
        self.normalized = True
        self.special = False


def test_tokenizer_fingerprint_ignores_copy_location_and_normalizes_added_tokens():
    baseline = _TinyTokenizer()
    candidate = _TinyTokenizer()
    baseline.init_kwargs = {
        "name_or_path": r"C:\\cache\\baseline",
        "cache_dir": r"C:\\cache\\one",
        "added_tokens_decoder": {99: _PathIndependentAddedToken("<extra>")},
        "model_max_length": 128,
    }
    candidate.init_kwargs = {
        "name_or_path": r"D:\\different\\candidate",
        "cache_dir": r"D:\\different\\cache",
        "added_tokens_decoder": {99: _PathIndependentAddedToken("<extra>")},
        "model_max_length": 128,
    }

    first = fingerprint_tokenizer(baseline)
    second = fingerprint_tokenizer(candidate)
    assert first == second
    assert first["canonicalization"] == "content+backend+config_without_paths_or_cache_fields"


class _TinyCausalModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1))
        self.config = SimpleNamespace(is_encoder_decoder=False)

    def forward(self, input_ids, attention_mask=None):
        logits = torch.full((input_ids.shape[0], input_ids.shape[1], 5), -5.0, dtype=torch.float32, device=input_ids.device)
        next_ids = (input_ids + 1).clamp(max=4)
        logits.scatter_(2, next_ids.unsqueeze(-1), 5.0)
        return SimpleNamespace(logits=logits)

    def generate(self, input_ids, attention_mask=None, max_new_tokens=1, **kwargs):
        result = input_ids.clone()
        for _ in range(int(max_new_tokens)):
            result = torch.cat([result, (result[:, -1:] + 1).clamp(max=4)], dim=1)
        return result


def test_tiny_hf_like_ab_metrics_are_finite_and_greedy_is_repeatable():
    cases = ({"id": "tiny", "prompt": "a ", "continuation": "b"},)
    tokenizer = _TinyTokenizer()
    model = _TinyCausalModel()
    first = evaluate_loaded_model(model, tokenizer, cases=cases, device=torch.device("cpu"), model_label="tiny")
    second = evaluate_loaded_model(model, tokenizer, cases=cases, device=torch.device("cpu"), model_label="tiny")
    assert first["aggregate"]["mean_target_nll"] < 1.0
    assert first["cases"][0]["greedy_exact_target"] is True
    assert first["cases"][0]["target_nll"] == second["cases"][0]["target_nll"]
    cached = evaluate_loaded_model(model, tokenizer, cases=cases, device=torch.device("cpu"), model_label="tiny", capture_logit_cache=True)
    assert tuple(cached["_logit_cache"]["tiny"].shape) == (5,)
    assert bool(torch.isfinite(cached["_logit_cache"]["tiny"]).all())
    json.dumps(first, allow_nan=False)


def test_gpt2_bpe_boundary_token_is_counted_by_offsets():
    class BoundaryTokenizer:
        def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
            if return_offsets_mapping:
                return {"input_ids": [[10, 11, 12]], "offset_mapping": [[(0, 5), (5, 8), (8, 12)]]}
            return {"input_ids": [[10, 11]] if text == "hello" else [[10, 11, 12]]}

    positions, method = _target_token_positions(BoundaryTokenizer(), "hello", " xyz", torch.tensor([[10, 11, 12]]))
    assert positions == [1, 2]
    assert method == "offset_mapping"


def test_greedy_generation_handles_bpe_boundary_merging():
    from scripts.text_ab_eval import _greedy_generation

    class BoundaryTokenizer:
        pad_token_id = 0
        eos_token_id = 0

        def __call__(self, text, return_tensors="pt", add_special_tokens=False, return_offsets_mapping=False):
            if return_offsets_mapping:
                return {"input_ids": torch.tensor([[10, 11, 12]]), "offset_mapping": [[(0, 5), (5, 8), (8, 12)]]}
            if text == "hello":
                return {"input_ids": torch.tensor([[10, 11]]), "attention_mask": torch.ones((1, 2), dtype=torch.long)}
            return {"input_ids": torch.tensor([[10, 11, 12]]), "attention_mask": torch.ones((1, 3), dtype=torch.long)}

    class EchoModel(torch.nn.Module):
        def generate(self, input_ids, attention_mask=None, max_new_tokens=1, **kwargs):
            return torch.cat([input_ids, torch.tensor([[11, 12]])], dim=1)

    result = _greedy_generation(EchoModel(), BoundaryTokenizer(), "hello", " xyz", torch.device("cpu"))
    assert result["target_generation_token_count"] == 2
    assert result["greedy_token_matches"] == 2
    assert result["greedy_exact_target"] is True


def test_missing_candidate_is_structured_and_does_not_try_to_load_a_model(tmp_path):
    baseline = tmp_path / "baseline"
    baseline.mkdir()
    payload = run_ab(baseline, tmp_path / "candidate-does-not-exist")
    assert payload["status"] == "candidate_missing"
    assert payload["candidate_is_real_checkpoint"] is False
    json.dumps(payload, allow_nan=False)


def test_adaptive_tune_cases_are_repeatable_diverse_and_evenly_partitionable():
    cases = tuple(DEFAULT_TUNE_CASES)
    assert len(cases) == 18
    assert len({case["id"] for case in cases}) == len(cases)
    assert {case["id"].split("_", 1)[0] for case in cases} >= {"science", "math", "code", "reasoning", "long", "perturbation"}
    split = len(cases) // 3
    assert all(set(case["id"] for case in cases[start : start + split]).isdisjoint(
        case["id"] for case in cases[start + split : start + 2 * split]
    ) for start in (0,))
    assert json.dumps(cases, allow_nan=False)
