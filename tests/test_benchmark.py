import json
from pathlib import Path

from faytuna_flow.benchmark import BenchmarkConfig, memory_snapshot, run_benchmark
from faytuna_flow.cli import main


def test_actual_dimension_batch_benchmark_preserves_full_differential_stencil(tmp_path: Path):
    memory = memory_snapshot()
    assert set(memory) == {"rss_bytes", "private_bytes", "commit_bytes"}
    assert all(value is not None and value >= 0 for value in memory.values())
    progress_jsonl = tmp_path / "benchmark.jsonl"
    progress_log = tmp_path / "benchmark.log"
    payload = run_benchmark(
        BenchmarkConfig(state_dim=16 * 1600, layers=2, probes=1, jacobian_rank=8, hessian_rank=4, seed=23, memory_interval_seconds=0.005),
        progress_jsonl=progress_jsonl,
        progress_log=progress_log,
    )
    assert payload["status"] == "pass"
    assert payload["expected_stencil_samples"] == 2 * (1 + 4 * 8 + 4 * 4)
    assert payload["single_dispatch"]["dispatches"] == 2 * (1 + 4 * 8 + 4 * 4)
    assert payload["batch_dispatch"]["dispatches"] == 2 * 3
    assert payload["single_dispatch"]["stencil_samples"] == payload["batch_dispatch"]["stencil_samples"] == payload["expected_stencil_samples"]
    assert payload["single_dispatch"]["path_signature_area_backend"] == "randomized_frobenius_sketch"
    assert payload["batch_dispatch"]["path_signature_area_backend"] == "randomized_frobenius_sketch"
    assert payload["trace_comparison"]["max_abs_error"] <= 1e-12
    assert payload["trace_comparison"]["mean_abs_error"] <= 1e-13
    records = [json.loads(line) for line in progress_jsonl.read_text(encoding="utf-8").splitlines()]
    assert records
    assert all("NaN" not in json.dumps(record) and "Infinity" not in json.dumps(record) for record in records)
    assert all("throughput_stencil_samples_per_second" in record for record in records)
    assert "samples/s=" in progress_log.read_text(encoding="utf-8")


def test_benchmark_cli_writes_machine_readable_result(tmp_path: Path, capsys):
    output = tmp_path / "result.json"
    progress = tmp_path / "progress.jsonl"
    assert main(["benchmark-observation", "--state-dim", "25600", "--layers", "1", "--probes", "1", "--jacobian-rank", "2", "--hessian-rank", "1", "--output", str(output), "--progress-jsonl", str(progress)]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["schema_version"] == "faytuna-benchmark-v1"
    assert json.loads(output.read_text(encoding="utf-8"))["status"] == "pass"
