"""Deterministic A/B evaluation for two ordinary local HF causal-LM checkpoints.

This is a text-level baseline evaluator. It deliberately does not collect
hidden states, fit flow transport, alter weights, or download model files.
Both model paths are loaded with ``local_files_only=True``. The metrics are
descriptive evidence for a real baseline/candidate pair; they do not establish
semantic transfer or causal improvement.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


SCHEMA_VERSION = "faytuna-text-ab-v1"

# These pairs are fixed in code so a baseline and candidate cannot silently
# receive different text. The trailing spaces are intentional: they keep the
# prompt/continuation boundary stable for GPT-2 BPE tokenization.
FIXED_EVAL_CASES: tuple[dict[str, str], ...] = (
    {"id": "knowledge_01", "prompt": "The capital of France is ", "continuation": "Paris."},
    {"id": "reasoning_01", "prompt": "If all squares are rectangles, and this figure is a square, then this figure is a ", "continuation": "rectangle."},
    {"id": "code_01", "prompt": "def add(a, b):\n    return ", "continuation": "a + b"},
    {"id": "sequence_01", "prompt": "January, February, March, ", "continuation": "April."},
    {"id": "science_01", "prompt": "At standard pressure, water freezes at ", "continuation": "0 degrees Celsius."},
    {"id": "narrative_01", "prompt": "Once upon a time, the small fox ", "continuation": "crossed the quiet field."},
)

# The six-case set above is retained for reproducibility of earlier reports.
# This larger deterministic set is used by the adaptive GPT-2 policy.  The
# cases are deliberately partitionable into three disjoint groups of six:
# train, validation, and holdout.  They are behavioral probes only; matching
# them cannot establish semantic transfer.
EXTENDED_TUNE_CASES: tuple[dict[str, str], ...] = (
    {"id": "science_02", "prompt": "The chemical symbol for sodium is ", "continuation": "Na."},
    {"id": "science_03", "prompt": "In a vacuum, light travels at approximately ", "continuation": "300,000 kilometers per second."},
    {"id": "science_04", "prompt": "Photosynthesis converts light energy into chemical energy using ", "continuation": "carbon dioxide and water."},
    {"id": "math_01", "prompt": "The derivative of x squared with respect to x is ", "continuation": "2x."},
    {"id": "math_02", "prompt": "A triangle with base 6 and height 4 has area ", "continuation": "12."},
    {"id": "math_03", "prompt": "If a sequence starts 2, 4, 8, 16, the next term is ", "continuation": "32."},
    {"id": "code_02", "prompt": "for value in values:\n    total += ", "continuation": "value"},
    {"id": "code_03", "prompt": "def is_even(n):\n    return n % 2 == ", "continuation": "0"},
    {"id": "code_04", "prompt": "items = [x * 2 for x in numbers if x > ", "continuation": "0]"},
    {"id": "reasoning_02", "prompt": "A is taller than B. B is taller than C. Therefore A is ", "continuation": "taller than C."},
    {"id": "reasoning_03", "prompt": "Every registered vehicle has a plate. This car is registered. It has a ", "continuation": "plate."},
    {"id": "reasoning_04", "prompt": "If the alarm is armed, opening the door triggers it. The alarm is armed and the door opens, so ", "continuation": "the alarm triggers."},
    {"id": "long_context_01", "prompt": "Context: red means stop; blue means wait; green means go. The requested signal is green. After checking the context carefully, the action is ", "continuation": "go."},
    {"id": "long_context_02", "prompt": "Notes: Mercury is first, Venus second, Earth third, Mars fourth. The planet immediately after Earth in these notes is ", "continuation": "Mars."},
    {"id": "long_context_03", "prompt": "Record: key A maps to 17, key B maps to 29, and key C maps to 41. Looking up key B gives ", "continuation": "29."},
    {"id": "perturbation_01", "prompt": "Counterfactual: if ice is heated above its melting point, it becomes ", "continuation": "liquid water."},
    {"id": "perturbation_02", "prompt": "Counterfactual: if the usual order is reversed, the first item after March is ", "continuation": "February."},
    {"id": "perturbation_03", "prompt": "Adversarial wording: a dependency chain says module C consumes B, and B consumes A. The earliest required module is ", "continuation": "A."},
)

# Public name for callers that want the policy's intended evaluation set.
DEFAULT_TUNE_CASES = EXTENDED_TUNE_CASES

PYTHON_CODE_EVAL_CASES: tuple[dict[str, str], ...] = (
    {"id": "py_func_add", "prompt": "def add(a, b):\n    return ", "continuation": "a + b"},
    {"id": "py_func_even", "prompt": "def is_even(n):\n    return n % 2 == ", "continuation": "0"},
    {"id": "py_loop_accum", "prompt": "for value in values:\n    total += ", "continuation": "value"},
    {"id": "py_comp_filter", "prompt": "items = [x * 2 for x in numbers if x > ", "continuation": "0]"},
    {"id": "py_str_split", "prompt": "words = text.strip().split(", "continuation": "\" \")"},
    {"id": "py_dict_assign", "prompt": "counts = {}\ncounts[\"total\"] = ", "continuation": "0"},
    {"id": "py_math_square", "prompt": "def square(x):\n    return x * ", "continuation": "x"},
    {"id": "py_list_pop", "prompt": "stack = []\nstack.append(42)\nval = stack.pop(", "continuation": ")"},
    {"id": "py_oop_init", "prompt": "class Point:\n    def __init__(self, x, y):\n        self.x = x\n        self.y = ", "continuation": "y"},
    {"id": "py_cond_guard", "prompt": "if count < 0:\n    return ", "continuation": "None"},
    {"id": "py_algo_factorial", "prompt": "def factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(", "continuation": "n - 1)"},
    {"id": "py_except_catch", "prompt": "try:\n    value = int(user_input)\nexcept ", "continuation": "ValueError:"},
)


def _torch() -> Any:
    try:
        import torch
    except ImportError as error:  # pragma: no cover - depends on optional env
        raise RuntimeError("PyTorch is required for text_ab_eval.py") from error
    return torch


def _strict(value: Any) -> Any:
    """Convert a result to JSON values without allowing NaN or Infinity."""

    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _strict(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_strict(item) for item in value]
    try:
        import numpy as np

        if isinstance(value, np.ndarray):
            return _strict(value.tolist())
        if isinstance(value, np.generic):
            return _strict(value.item())
    except ImportError:  # pragma: no cover - numpy is a project dependency
        pass
    if isinstance(value, bool) or value is None or isinstance(value, (str, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if hasattr(value, "item"):
        try:
            return _strict(value.item())
        except (TypeError, ValueError):
            pass
    return str(value)


def _write_strict(path: str | Path, payload: Mapping[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    safe = _strict(payload)
    destination.write_text(json.dumps(safe, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _write_jsonl(path: str | Path, records: Sequence[Mapping[str, Any]]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(_strict(record), ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")


def _sha256_bytes(chunks: Sequence[bytes]) -> str:
    digest = hashlib.sha256()
    for chunk in chunks:
        digest.update(chunk)
    return digest.hexdigest()


def _json_bytes(value: Any) -> bytes:
    return json.dumps(_strict(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _tensor_bytes(value: Any) -> bytes:
    torch = _torch()
    if hasattr(value, "detach"):
        tensor = value.detach().cpu().contiguous()
        # A byte view works for float16/bfloat16 and avoids an implicit lossy
        # conversion while hashing the already loaded ordinary checkpoint.
        return tensor.view(torch.uint8).numpy().tobytes()
    try:
        import numpy as np

        return np.asarray(value).tobytes()
    except (TypeError, ValueError):
            return repr(value).encode("utf-8")


_TOKENIZER_PATH_KEYS = frozenset(
    {
        "name_or_path",
        "_name_or_path",
        "path",
        "cache_dir",
        "cache_file",
        "tokenizer_file",
        "vocab_file",
        "merges_file",
        "special_tokens_map_file",
        "added_tokens_file",
    }
)
_DROP_TOKENIZER_FIELD = object()


def _is_path_dependent_tokenizer_key(key: str) -> bool:
    lowered = key.lower()
    return (
        lowered in _TOKENIZER_PATH_KEYS
        or lowered.endswith("_path")
        or lowered.endswith("_file")
        or "cache" in lowered
    )


def _canonical_tokenizer_value(value: Any, *, key: str | None = None) -> Any:
    """Make tokenizer metadata stable across local copies and cache layouts.

    HF tokenizers often place ``name_or_path`` and local file locations in
    ``init_kwargs``.  Those fields describe where an equivalent tokenizer was
    loaded, rather than tokenizer content.  AddedToken instances also have a
    process-dependent repr, so their public fields are serialized explicitly.
    """

    if key is not None and _is_path_dependent_tokenizer_key(key):
        return _DROP_TOKENIZER_FIELD

    # transformers.AddedToken and compatible objects expose these attributes;
    # do this before the generic object/string fallbacks because repr(obj) is
    # not a content fingerprint.
    if hasattr(value, "content") and isinstance(getattr(value, "content", None), str):
        fields = ("content", "lstrip", "rstrip", "single_word", "normalized", "special")
        return {
            field: _canonical_tokenizer_value(getattr(value, field, None), key=field)
            for field in fields
        }

    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for raw_key, raw_value in value.items():
            normalized_key = str(raw_key)
            normalized_value = _canonical_tokenizer_value(raw_value, key=normalized_key)
            if normalized_value is not _DROP_TOKENIZER_FIELD:
                result[normalized_key] = normalized_value
        return result

    if isinstance(value, (tuple, list)):
        return [_canonical_tokenizer_value(item) for item in value]

    if isinstance(value, (set, frozenset)):
        items = [_canonical_tokenizer_value(item) for item in value]
        return sorted(items, key=lambda item: _json_bytes(item))

    if isinstance(value, Path):
        # A bare Path should only occur in a path-bearing field, which is
        # removed by the parent mapping. Keep a deterministic representation
        # for unusual top-level callback/config values.
        return value.name

    if isinstance(value, (bool, type(None), int, float, str)):
        return value

    if hasattr(value, "item"):
        try:
            return _canonical_tokenizer_value(value.item())
        except (TypeError, ValueError):
            pass

    return str(value)


def _tokenizer_content_payload(tokenizer: Any) -> dict[str, Any]:
    """Collect content-bearing tokenizer data without load-location fields."""

    vocab_getter = getattr(tokenizer, "get_vocab", None)
    vocab = vocab_getter() if callable(vocab_getter) else None
    payload: dict[str, Any] = {"class": type(tokenizer).__name__}
    if vocab is not None:
        payload["vocab"] = sorted(
            (str(key), int(value)) for key, value in vocab.items()
        )

    init_kwargs = getattr(tokenizer, "init_kwargs", {})
    normalized_init_kwargs = _canonical_tokenizer_value(init_kwargs)
    payload["init_kwargs"] = normalized_init_kwargs

    # Fast tokenizers contain merges, normalizer, pre-tokenizer and added-token
    # definitions in the backend JSON. Include it when available so two
    # tokenizers with the same vocabulary but different merge rules do not
    # collide. The JSON is canonicalized recursively and path-like keys are
    # removed in the same way as init_kwargs.
    backend = getattr(tokenizer, "backend_tokenizer", None)
    to_str = getattr(backend, "to_str", None)
    if callable(to_str):
        try:
            backend_payload = json.loads(to_str())
        except (TypeError, ValueError):
            backend_payload = None
        if backend_payload is not None:
            payload["backend"] = _canonical_tokenizer_value(backend_payload)

    return payload


def fingerprint_tokenizer(tokenizer: Any) -> dict[str, Any]:
    """Fingerprint tokenizer content independently of local load paths."""

    payload = _tokenizer_content_payload(tokenizer)
    vocab = payload.get("vocab")
    vocab_hash = _sha256_bytes([_json_bytes(payload)])
    vocab_size = len(vocab) if isinstance(vocab, list) else None
    return {
        "class": type(tokenizer).__name__,
        "vocab_size": vocab_size,
        "sha256": vocab_hash,
        "canonicalization": "content+backend+config_without_paths_or_cache_fields",
    }


def fingerprint_model(model: Any, *, checkpoint_ref: str, include_weights: bool = True) -> dict[str, Any]:
    """Fingerprint config and tensor schema, optionally including all weights."""

    config = getattr(model, "config", {})
    if callable(getattr(config, "to_dict", None)):
        config = config.to_dict()
    state_getter = getattr(model, "state_dict", None)
    state = state_getter() if callable(state_getter) else {}
    schema_parts: list[bytes] = []
    weight_parts: list[bytes] = []
    parameter_count = 0
    for name in sorted(state):
        value = state[name]
        shape = list(getattr(value, "shape", ()))
        dtype = str(getattr(value, "dtype", type(value).__name__))
        schema_parts.append(_json_bytes({"name": str(name), "shape": shape, "dtype": dtype}))
        try:
            parameter_count += int(value.numel())
        except (AttributeError, TypeError, ValueError):
            try:
                parameter_count += int(value.size)
            except (AttributeError, TypeError, ValueError):
                pass
        if include_weights:
            weight_parts.extend([str(name).encode("utf-8"), _tensor_bytes(value)])
    config_hash = _sha256_bytes([_json_bytes(config)])
    schema_hash = _sha256_bytes(schema_parts)
    result: dict[str, Any] = {
        "checkpoint_ref": str(checkpoint_ref),
        "config_sha256": config_hash,
        "tensor_schema_sha256": schema_hash,
        "tensor_count": len(state),
        "parameter_count": parameter_count,
        "weights_hashed": bool(include_weights),
    }
    if include_weights:
        result["weights_sha256"] = _sha256_bytes(weight_parts)
    result["model_sha256"] = _sha256_bytes([_json_bytes(result)])
    return result


def resolve_device(requested: str) -> Any:
    torch = _torch()
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def seed_everything(seed: int) -> None:
    torch = _torch()
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _encoded(tokenizer: Any, text: str, device: Any) -> tuple[Any, Any]:
    torch = _torch()
    encoded = tokenizer(text, return_tensors="pt", add_special_tokens=False)
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded.get("attention_mask")
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)
    return input_ids, attention_mask.to(device)


def _logits(output: Any) -> Any:
    value = getattr(output, "logits", None)
    if value is not None:
        return value
    if isinstance(output, (tuple, list)) and output:
        return output[0]
    raise RuntimeError("causal model output does not expose logits")


def _next_token_logits(model: Any, tokenizer: Any, prompt: str, device: Any) -> Any:
    """Return the final-context logits used for a deterministic drift check."""

    torch = _torch()
    input_ids, attention_mask = _encoded(tokenizer, prompt, device)
    with torch.no_grad():
        logits = _logits(model(input_ids=input_ids, attention_mask=attention_mask))
    values = logits[:, -1, :].detach().cpu().float().contiguous()
    if not bool(torch.isfinite(values).all()):
        raise FloatingPointError("next-token logits are non-finite")
    return values[0]


def _target_token_positions(tokenizer: Any, prompt: str, continuation: str, full_ids: Any) -> tuple[list[int], str]:
    """Find target token positions without assuming a BPE prefix boundary.

    GPT-2 tokenizers may merge the prompt's trailing whitespace with the first
    continuation token.  In that case ``len(tokenize(prompt))`` is not the
    first target index in ``tokenize(prompt + continuation)``.  Fast HF
    tokenizers expose character offsets, which identify boundary-crossing
    tokens directly.  The prefix-length path remains available for tiny or
    slow tokenizer doubles that do not implement offsets.
    """

    full_length = int(full_ids.shape[1])
    try:
        encoded = tokenizer(prompt + continuation, add_special_tokens=False, return_offsets_mapping=True)
        offsets = encoded.get("offset_mapping") if isinstance(encoded, Mapping) else None
        if offsets is not None:
            if hasattr(offsets, "tolist"):
                offsets = offsets.tolist()
            if offsets and isinstance(offsets[0], (list, tuple)) and offsets and isinstance(offsets[0][0], (list, tuple)):
                offsets = offsets[0]
            positions = [index for index, pair in enumerate(offsets or ()) if len(pair) == 2 and int(pair[1]) > len(prompt) and int(pair[1]) > int(pair[0])]
            positions = [index for index in positions if 0 <= index < full_length]
            if positions:
                return positions, "offset_mapping"
    except (TypeError, ValueError, KeyError, NotImplementedError):
        pass
    prompt_encoded = tokenizer(prompt, add_special_tokens=False)
    prompt_ids = prompt_encoded["input_ids"] if isinstance(prompt_encoded, Mapping) else prompt_encoded
    if hasattr(prompt_ids, "shape"):
        prompt_length = int(prompt_ids.shape[-1])
    else:
        prompt_length = len(prompt_ids[0]) if prompt_ids and isinstance(prompt_ids[0], (list, tuple)) else len(prompt_ids)
    positions = list(range(min(prompt_length, full_length), full_length))
    if positions:
        return positions, "prompt_prefix_length"
    continuation_encoded = tokenizer(continuation, add_special_tokens=False)
    continuation_ids = continuation_encoded["input_ids"] if isinstance(continuation_encoded, Mapping) else continuation_encoded
    if hasattr(continuation_ids, "shape"):
        continuation_length = int(continuation_ids.shape[-1])
    else:
        continuation_length = len(continuation_ids[0]) if continuation_ids and isinstance(continuation_ids[0], (list, tuple)) else len(continuation_ids)
    fallback_start = max(0, full_length - continuation_length)
    return list(range(fallback_start, full_length)), "continuation_suffix_fallback"


def _target_loss(model: Any, tokenizer: Any, prompt: str, continuation: str, device: Any) -> dict[str, Any]:
    torch = _torch()
    full_ids, full_mask = _encoded(tokenizer, prompt + continuation, device)
    target_positions, boundary_method = _target_token_positions(tokenizer, prompt, continuation, full_ids)
    target_positions = [position for position in target_positions if position > 0]
    target_length = len(target_positions)
    if target_length < 1:
        raise ValueError("fixed continuation produced no target tokens")
    with torch.no_grad():
        output = model(input_ids=full_ids, attention_mask=full_mask)
        logits = _logits(output)
    labels = torch.full_like(full_ids, -100)
    labels[:, target_positions] = full_ids[:, target_positions]
    shifted_logits = logits[:, :-1, :].contiguous()
    shifted_labels = labels[:, 1:].contiguous()
    valid = shifted_labels != -100
    if int(valid.sum()) != target_length:
        raise RuntimeError("target-token mask is inconsistent with the prompt boundary")
    loss_sum = torch.nn.functional.cross_entropy(
        shifted_logits.reshape(-1, shifted_logits.shape[-1]),
        shifted_labels.reshape(-1),
        ignore_index=-100,
        reduction="sum",
    )
    mean_nll = float((loss_sum / valid.sum()).detach().cpu().item())
    perplexity = math.exp(min(mean_nll, 700.0))
    return {
        "target_token_count": target_length,
        "target_boundary_method": boundary_method,
        "target_nll": mean_nll,
        "target_perplexity": perplexity if math.isfinite(perplexity) else None,
    }


def _greedy_generation(model: Any, tokenizer: Any, prompt: str, continuation: str, device: Any) -> dict[str, Any]:
    torch = _torch()
    input_ids, attention_mask = _encoded(tokenizer, prompt, device)
    full_ids, _ = _encoded(tokenizer, prompt + continuation, device)
    target_positions, _ = _target_token_positions(tokenizer, prompt, continuation, full_ids)
    if target_positions:
        continuation_ids = full_ids[:, target_positions]
    else:
        continuation_ids = full_ids[:, input_ids.shape[1] :]
    target_count = max(1, int(continuation_ids.shape[1]))
    pad_id = getattr(tokenizer, "pad_token_id", None)
    if pad_id is None:
        pad_id = getattr(tokenizer, "eos_token_id", None)
    started = time.perf_counter()
    with torch.no_grad():
        generated = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=target_count,
            do_sample=False,
            num_beams=1,
            use_cache=True,
            pad_token_id=pad_id,
        )
    elapsed = time.perf_counter() - started
    generated_continuation = generated[:, input_ids.shape[1] :]
    compared = min(int(generated_continuation.shape[1]), int(continuation_ids.shape[1]))
    token_matches = int((generated_continuation[:, :compared] == continuation_ids[:, :compared]).sum().item()) if compared else 0
    exact = bool(generated_continuation.shape[1] == continuation_ids.shape[1] and torch.equal(generated_continuation, continuation_ids))
    decode = getattr(tokenizer, "decode", None)
    generated_text = decode(generated_continuation[0].tolist(), skip_special_tokens=False) if callable(decode) else None
    return {
        "generation_seconds": float(elapsed),
        "generated_token_count": int(generated_continuation.shape[1]),
        "target_generation_token_count": int(continuation_ids.shape[1]),
        "greedy_token_matches": token_matches,
        "greedy_token_accuracy": float(token_matches / max(1, int(continuation_ids.shape[1]))),
        "greedy_exact_target": exact,
        "generated_continuation": generated_text,
    }


def evaluate_loaded_model(
    model: Any,
    tokenizer: Any,
    *,
    cases: Sequence[Mapping[str, str]] = FIXED_EVAL_CASES,
    device: Any | None = None,
    model_label: str = "model",
    capture_logit_cache: bool = False,
) -> dict[str, Any]:
    """Evaluate an already loaded causal model; useful for tiny HF-like tests."""

    if device is None:
        device = resolve_device("cpu")
    model.eval()
    results = []
    logit_cache: dict[str, Any] = {}
    for case in cases:
        case_id = str(case["id"])
        prompt = str(case["prompt"])
        continuation = str(case["continuation"])
        loss_started = time.perf_counter()
        loss = _target_loss(model, tokenizer, prompt, continuation, device)
        loss["loss_seconds"] = float(time.perf_counter() - loss_started)
        generation = _greedy_generation(model, tokenizer, prompt, continuation, device)
        results.append({"case_id": case_id, "prompt": prompt, "continuation": continuation, **loss, **generation})
        if capture_logit_cache:
            logit_cache[case_id] = _next_token_logits(model, tokenizer, prompt, device)
    mean_nll = sum(float(item["target_nll"]) for item in results) / max(1, len(results))
    mean_ppl = sum(float(item["target_perplexity"]) for item in results) / max(1, len(results))
    report = {
        "label": model_label,
        "case_count": len(results),
        "cases": results,
        "aggregate": {
            "mean_target_nll": mean_nll,
            "mean_target_perplexity": mean_ppl,
            "mean_greedy_token_accuracy": sum(float(item["greedy_token_accuracy"]) for item in results) / max(1, len(results)),
            "greedy_exact_rate": sum(bool(item["greedy_exact_target"]) for item in results) / max(1, len(results)),
            "mean_loss_seconds": sum(float(item["loss_seconds"]) for item in results) / max(1, len(results)),
            "mean_generation_seconds": sum(float(item["generation_seconds"]) for item in results) / max(1, len(results)),
        },
    }
    if capture_logit_cache:
        report["_logit_cache"] = logit_cache
    return report


def load_local_model(checkpoint: str | Path, device: Any) -> tuple[Any, Any]:
    """Load an ordinary local HF checkpoint without a download fallback."""

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as error:  # pragma: no cover - optional dependency
        raise RuntimeError("transformers is required to load HF checkpoints") from error
    reference = str(checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(reference, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(reference, local_files_only=True)
    if bool(getattr(getattr(model, "config", None), "is_encoder_decoder", False)):
        raise ValueError("the evaluator requires a decoder-only causal LM")
    model.to(device)
    model.eval()
    if getattr(tokenizer, "pad_token_id", None) is None and getattr(tokenizer, "eos_token_id", None) is not None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def _missing_payload(baseline: str, candidate: str | None, status: str, reason: str) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "observation_stage": "not_run",
        "candidate_is_real_checkpoint": False if status == "candidate_missing" else None,
        "models": {"baseline": {"checkpoint_ref": baseline}, "candidate": None if candidate is None else {"checkpoint_ref": candidate}},
        "cases": [],
        "comparison": None,
        "reason": reason,
        "interpretation": "descriptive A/B baseline only; no semantic or causal claim",
    }


def run_ab(
    baseline: str | Path,
    candidate: str | Path | None,
    *,
    output: str | Path | None = None,
    jsonl: str | Path | None = None,
    device: str = "auto",
    seed: int = 17,
    hash_weights: bool = True,
    cases: Sequence[Mapping[str, str]] = FIXED_EVAL_CASES,
) -> dict[str, Any]:
    """Run A/B evaluation or return an honest structured missing-checkpoint status."""

    baseline_ref = str(baseline)
    candidate_ref = None if candidate is None else str(candidate)
    if not Path(baseline_ref).exists():
        payload = _missing_payload(baseline_ref, candidate_ref, "baseline_missing", "baseline checkpoint path does not exist locally")
        if output:
            _write_strict(output, payload)
        if jsonl:
            _write_jsonl(jsonl, [{"event": "evaluation_end", **payload}])
        return payload
    if candidate_ref is None or not Path(candidate_ref).exists():
        payload = _missing_payload(baseline_ref, candidate_ref, "candidate_missing", "candidate checkpoint path does not exist locally; no candidate was evaluated")
        if output:
            _write_strict(output, payload)
        if jsonl:
            _write_jsonl(jsonl, [{"event": "evaluation_end", **payload}])
        return payload
    cases = tuple(dict(case) for case in cases)
    if not cases or any("id" not in case or "prompt" not in case or "continuation" not in case for case in cases):
        raise ValueError("text A/B cases must contain non-empty id, prompt, and continuation fields")

    seed_everything(seed)
    selected_device = resolve_device(device)
    events: list[dict[str, Any]] = [{"event": "evaluation_start", "schema_version": SCHEMA_VERSION, "seed": int(seed), "device": str(selected_device), "observation_stage": "not_run"}]
    model_reports: dict[str, Any] = {}
    model_cases: dict[str, Any] = {}
    model_logits: dict[str, dict[str, Any]] = {}
    for label, reference in (("baseline", baseline_ref), ("candidate", candidate_ref)):
        load_started = time.perf_counter()
        model, tokenizer = load_local_model(reference, selected_device)
        load_seconds = time.perf_counter() - load_started
        report = evaluate_loaded_model(model, tokenizer, cases=cases, model_label=label, device=selected_device, capture_logit_cache=True)
        model_logits[label] = dict(report.pop("_logit_cache", {}))
        report["load_seconds"] = float(load_seconds)
        report["model_fingerprint"] = fingerprint_model(model, checkpoint_ref=reference, include_weights=hash_weights)
        report["tokenizer_fingerprint"] = fingerprint_tokenizer(tokenizer)
        report["dtype"] = str(next(model.parameters()).dtype) if callable(getattr(model, "parameters", None)) else None
        report["device"] = str(selected_device)
        model_reports[label] = report
        model_cases[label] = report["cases"]
        events.append({"event": "model_result", "model": label, "load_seconds": report["load_seconds"], "aggregate": report["aggregate"], "model_fingerprint": report["model_fingerprint"], "tokenizer_fingerprint": report["tokenizer_fingerprint"]})
        del model
        del tokenizer
        if selected_device.type == "cuda":
            _torch().cuda.empty_cache()

    baseline_aggregate = model_reports["baseline"]["aggregate"]
    candidate_aggregate = model_reports["candidate"]["aggregate"]
    logit_drift_cases = []
    for case in cases:
        case_id = str(case["id"])
        baseline_logits = model_logits["baseline"].get(case_id)
        candidate_logits = model_logits["candidate"].get(case_id)
        if baseline_logits is None or candidate_logits is None or tuple(baseline_logits.shape) != tuple(candidate_logits.shape):
            raise RuntimeError(f"logit cache is missing or has incompatible shape for case {case_id}")
        torch = _torch()
        denominator = max(1.0, float(torch.linalg.vector_norm(baseline_logits).item()))
        drift = float(torch.linalg.vector_norm(candidate_logits - baseline_logits).item() / denominator)
        if not math.isfinite(drift):
            raise FloatingPointError(f"logit drift is non-finite for case {case_id}")
        logit_drift_cases.append({"case_id": case_id, "next_token_logit_relative_l2": drift})
    mean_logit_drift = sum(float(item["next_token_logit_relative_l2"]) for item in logit_drift_cases) / max(1, len(logit_drift_cases))
    comparison = {
        "target_metric": "mean_target_nll",
        "lower_is_better": True,
        "baseline_mean_target_nll": baseline_aggregate["mean_target_nll"],
        "candidate_mean_target_nll": candidate_aggregate["mean_target_nll"],
        "candidate_minus_baseline_mean_target_nll": candidate_aggregate["mean_target_nll"] - baseline_aggregate["mean_target_nll"],
        "baseline_mean_target_perplexity": baseline_aggregate["mean_target_perplexity"],
        "candidate_mean_target_perplexity": candidate_aggregate["mean_target_perplexity"],
        "candidate_minus_baseline_mean_target_perplexity": candidate_aggregate["mean_target_perplexity"] - baseline_aggregate["mean_target_perplexity"],
        "candidate_minus_baseline_greedy_token_accuracy": candidate_aggregate["mean_greedy_token_accuracy"] - baseline_aggregate["mean_greedy_token_accuracy"],
        "next_token_logit_relative_l2_by_case": logit_drift_cases,
        "mean_next_token_logit_relative_l2": mean_logit_drift,
        "decision": "descriptive_only_no_claim",
    }
    events.append({"event": "comparison", **comparison})
    payload = {
        "schema_version": SCHEMA_VERSION,
        "status": "ok",
        "observation_stage": "not_run",
        "seed": int(seed),
        "device": str(selected_device),
        "fixed_case_ids": [str(case["id"]) for case in cases],
        "models": model_reports,
        "cases": model_cases,
        "comparison": comparison,
        "interpretation": "descriptive A/B baseline only; no semantic or causal claim",
    }
    events.append({"event": "evaluation_end", "status": "ok", "comparison": comparison})
    payload = _strict(payload)
    if output:
        _write_strict(output, payload)
    if jsonl:
        _write_jsonl(jsonl, events)
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Deterministic local-only A/B text evaluator for ordinary HF causal-LM checkpoints")
    parser.add_argument("--baseline", required=True, help="local baseline checkpoint directory")
    parser.add_argument("--candidate", required=False, help="local candidate checkpoint directory; missing means candidate_missing")
    parser.add_argument("--output", required=False, help="strict JSON report path")
    parser.add_argument("--jsonl", required=False, help="strict JSONL event report path")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--skip-weight-hash", action="store_true", help="hash config/schema only; default hashes loaded tensor bytes")
    parser.add_argument("--cases-suite", choices=["fixed", "extended", "python_code"], default="fixed", help="evaluation prompt suite")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    cases_map = {
        "fixed": FIXED_EVAL_CASES,
        "extended": EXTENDED_TUNE_CASES,
        "python_code": PYTHON_CODE_EVAL_CASES,
    }
    cases = cases_map.get(args.cases_suite, FIXED_EVAL_CASES)
    try:
        payload = run_ab(
            args.baseline,
            args.candidate,
            output=args.output,
            jsonl=args.jsonl,
            device=args.device,
            seed=args.seed,
            hash_weights=not args.skip_weight_hash,
            cases=cases,
        )
    except (RuntimeError, ValueError, OSError, KeyError) as error:
        payload = {
            "schema_version": SCHEMA_VERSION,
            "status": "error",
            "observation_stage": "not_run",
            "reason": f"{type(error).__name__}: {error}",
            "interpretation": "descriptive A/B baseline only; no semantic or causal claim",
        }
        if args.output:
            _write_strict(args.output, payload)
        if args.jsonl:
            _write_jsonl(args.jsonl, [{"event": "evaluation_end", **payload}])
    print(json.dumps(_strict(payload), ensure_ascii=False, indent=2, allow_nan=False))
    return 0 if payload.get("status") == "ok" else 2


if __name__ == "__main__":
    raise SystemExit(main())
