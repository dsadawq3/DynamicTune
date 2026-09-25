"""Contracts for ordinary checkpoint export and stock llama.cpp validation.

This module builds and checks commands for an unmodified llama.cpp checkout.
It does not fork, patch, instrument, or execute llama.cpp during the local test
suite.  Hidden-state observation belongs to the HF/PyTorch stage.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping

from .model_families import GPT2_XL_TO_SMALL, ModelFamilyPreset, preflight_gpt2_pair, strict_json_payload


@dataclass(frozen=True)
class LlamaCppValidationPlan:
    gguf_path: str
    llama_cli: str
    llama_perplexity: str | None
    prompt_file: str
    generation_command: tuple[str, ...]
    perplexity_command: tuple[str, ...] | None
    instrumentation_available: bool = False

    def to_dict(self) -> dict[str, Any]:
        return strict_json_payload({
            "schema_version": "faytuna-stock-llama-cpp-plan-v1",
            "gguf_path": self.gguf_path,
            "llama_cli": self.llama_cli,
            "llama_perplexity": self.llama_perplexity,
            "prompt_file": self.prompt_file,
            "generation_command": list(self.generation_command),
            "perplexity_command": None if self.perplexity_command is None else list(self.perplexity_command),
            "instrumentation_available": self.instrumentation_available,
            "hidden_state_note": "stock llama.cpp command line does not expose the HF observation trace contract",
        })


def _existing_file(path: str | Path | None, label: str) -> tuple[bool, str | None]:
    if path is None:
        return False, None
    value = Path(path)
    if not value.is_file():
        return False, f"{label} does not exist: {value}"
    return True, None


def _is_gguf(path: str | Path) -> tuple[bool, str | None]:
    value = Path(path)
    if value.suffix.lower() != ".gguf":
        return False, f"runtime model must have .gguf suffix: {value}"
    if not value.is_file():
        return False, f"GGUF file does not exist: {value}"
    try:
        with value.open("rb") as stream:
            magic = stream.read(4)
    except OSError as error:
        return False, f"cannot read GGUF header: {error}"
    if magic != b"GGUF":
        return False, f"file does not have GGUF magic: {value}"
    return True, None


def build_gguf_conversion_command(converter_script: str | Path, checkpoint_dir: str | Path, output_gguf: str | Path, *, python_executable: str | None = None) -> tuple[str, ...]:
    converter = Path(converter_script)
    checkpoint = Path(checkpoint_dir)
    output = Path(output_gguf)
    if not converter.is_file():
        raise FileNotFoundError(f"llama.cpp HF-to-GGUF converter does not exist: {converter}")
    if not checkpoint.is_dir():
        raise FileNotFoundError(f"exported GPT-2 checkpoint directory does not exist: {checkpoint}")
    if output.suffix.lower() != ".gguf":
        raise ValueError("GGUF output path must end with .gguf")
    return (python_executable or sys.executable, str(converter), str(checkpoint), "--outfile", str(output), "--outtype", "f16")


def build_stock_llama_cpp_plan(gguf_path: str | Path, llama_cli: str | Path, prompt_file: str | Path, *, llama_perplexity: str | Path | None = None, n_predict: int = 128) -> LlamaCppValidationPlan:
    good_model, model_reason = _is_gguf(gguf_path)
    good_cli, cli_reason = _existing_file(llama_cli, "llama-cli")
    good_prompt, prompt_reason = _existing_file(prompt_file, "prompt file")
    if n_predict < 1:
        raise ValueError("n_predict must be positive")
    reasons = [reason for reason in (model_reason, cli_reason, prompt_reason) if reason]
    if reasons or not good_model or not good_cli or not good_prompt:
        raise ValueError("stock llama.cpp validation preflight failed: " + "; ".join(reasons))
    perplexity_command = None
    if llama_perplexity is not None:
        good_perplexity, perplexity_reason = _existing_file(llama_perplexity, "llama-perplexity")
        if not good_perplexity:
            raise ValueError("stock llama.cpp validation preflight failed: " + str(perplexity_reason))
        perplexity_command = (str(llama_perplexity), "-m", str(gguf_path), "-f", str(prompt_file))
    generation_command = (str(llama_cli), "-m", str(gguf_path), "-f", str(prompt_file), "-n", str(int(n_predict)), "--temp", "0")
    return LlamaCppValidationPlan(str(gguf_path), str(llama_cli), None if llama_perplexity is None else str(llama_perplexity), str(prompt_file), generation_command, perplexity_command)


def preflight_stock_runtime(
    teacher_config: Any,
    student_config: Any,
    *,
    gguf_path: str | Path | None = None,
    llama_cli: str | Path | None = None,
    llama_perplexity: str | Path | None = None,
    prompt_file: str | Path | None = None,
    converter_script: str | Path | None = None,
    converter_checkpoint_dir: str | Path | None = None,
    preset: ModelFamilyPreset = GPT2_XL_TO_SMALL,
) -> dict[str, Any]:
    """Return a strict preflight report for observation and stock runtime stages."""

    architecture = preflight_gpt2_pair(teacher_config, student_config, preset)
    reasons = list(architecture["reasons"])
    conversion: dict[str, Any] = {"checked": False, "supported": None, "command": None, "reasons": []}
    if converter_script is not None:
        conversion["checked"] = True
        converter = Path(converter_script)
        if not converter.is_file():
            conversion["reasons"].append(f"converter script does not exist: {converter}")
        elif converter_checkpoint_dir is None:
            conversion["reasons"].append("converter_checkpoint_dir is required when converter_script is supplied")
        elif not Path(converter_checkpoint_dir).is_dir():
            conversion["reasons"].append(f"checkpoint directory does not exist: {converter_checkpoint_dir}")
        else:
            source = converter.read_text(encoding="utf-8", errors="ignore").lower()
            advertised = "gpt2" in source or "gpt-2" in source
            conversion["supported"] = bool(advertised)
            if not advertised:
                conversion["reasons"].append("converter script does not advertise GPT-2 support; no conversion pass is claimed")
            else:
                output = Path(gguf_path) if gguf_path is not None else Path(converter_checkpoint_dir) / "model-f16.gguf"
                try:
                    conversion["command"] = list(build_gguf_conversion_command(converter, converter_checkpoint_dir, output))
                except (ValueError, FileNotFoundError) as error:
                    conversion["reasons"].append(str(error))
        reasons.extend(conversion["reasons"])
    runtime: dict[str, Any] = {"checked": any(value is not None for value in (gguf_path, llama_cli, llama_perplexity)), "ready": None, "reasons": [], "plan": None}
    if runtime["checked"]:
        if gguf_path is None or llama_cli is None:
            runtime["reasons"].append("gguf_path and llama_cli are required together for stock runtime validation")
        else:
            try:
                prompt = prompt_file or Path(gguf_path).with_suffix(".prompt.txt")
                plan = build_stock_llama_cpp_plan(gguf_path, llama_cli, prompt, llama_perplexity=llama_perplexity)
                runtime["plan"] = plan.to_dict()
                runtime["ready"] = True
            except (ValueError, FileNotFoundError) as error:
                runtime["reasons"].append(str(error))
                runtime["ready"] = False
        reasons.extend(runtime["reasons"])
    status = "rejected" if not architecture["observation_ready"] or reasons else "observation_ready_runtime_not_checked" if not runtime["checked"] else "ready_for_stock_runtime"
    return strict_json_payload({
        "schema_version": "faytuna-gpt2-stock-runtime-preflight-v1",
        "preset": preset.name,
        "status": status,
        "observation": architecture,
        "conversion": conversion,
        "runtime": runtime,
        "reasons": reasons,
        "hidden_state_instrumentation": "HF/PyTorch observation only; unavailable through ordinary llama-cli",
        "semantic_claim": "not established by preflight or command construction",
    })


def run_stock_llama_cpp_validation(plan: LlamaCppValidationPlan, *, timeout_seconds: float = 300.0) -> dict[str, Any]:
    """Run only the commands represented by a validated stock plan.

    This helper is intentionally never called by local tests.  It invokes the
    user-supplied unmodified binaries and reports process/output evidence; it
    does not inspect hidden states or claim semantic transfer.
    """

    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")
    results = []
    for command in (plan.generation_command, plan.perplexity_command):
        if command is None:
            continue
        completed = subprocess.run(command, capture_output=True, text=True, timeout=timeout_seconds, check=False)
        results.append({"command": list(command), "returncode": int(completed.returncode), "stdout": completed.stdout, "stderr": completed.stderr, "finite_text": all(token not in (completed.stdout + completed.stderr).lower() for token in ("nan", "inf"))})
    return strict_json_payload({"schema_version": "faytuna-stock-llama-cpp-result-v1", "status": "pass" if results and all(item["returncode"] == 0 and item["finite_text"] for item in results) else "failed", "instrumentation_available": False, "results": results, "semantic_claim": "not established"})
