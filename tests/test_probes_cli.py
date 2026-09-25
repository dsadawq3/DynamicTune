import json
from pathlib import Path

from faytuna_flow.cli import main
from faytuna_flow.probes import ProbeGenerator, ProbeGeneratorConfig


def test_probe_generator_has_structured_families_and_pair_safe_splits():
    probes = ProbeGenerator(ProbeGeneratorConfig(state_dim=5, per_family=6, seed=31)).generate()
    families = {probe.family for probe in probes}
    assert families == set(ProbeGenerator.families)
    assert all(len(probe.payload) >= 3 for probe in probes)
    split = ProbeGenerator(ProbeGeneratorConfig(state_dim=5, per_family=6, seed=31)).split(probes)
    ids = [probe.probe_id for probe in split.all()]
    assert len(ids) == len(set(ids))
    assert set(probe.split for probe in split.all()) == {"train", "validation", "holdout"}
    buckets = {}
    for probe in split.all():
        if probe.pair_id:
            buckets.setdefault(probe.pair_id, set()).add(probe.split)
    assert buckets and all(len(values) == 1 for values in buckets.values())


def test_cli_inspect_collect_report(tmp_path: Path, capsys):
    assert main(["inspect", "--model", "synthetic"]) == 0
    inspect_output = json.loads(capsys.readouterr().out)
    assert inspect_output["capabilities"]["jacobian_sketch"] is True
    trace_path = tmp_path / "traces.npz"
    assert main(["collect-trace", "--output", str(trace_path), "--layers", "3", "--dimension", "3", "--probes", "7"]) == 0
    assert trace_path.exists()
    capsys.readouterr()
    assert main(["report", "--trace", str(trace_path)]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["trace_count"] == 7


def test_cli_paired_align_and_holdout_scorecard_are_connected(tmp_path: Path, capsys):
    paths = {}
    for split in ("train", "validation", "holdout"):
        output = tmp_path / split
        assert main(["collect-paired", "--output", str(output), "--teacher-dimension", "4", "--student-dimension", "2", "--teacher-layers", "5", "--student-layers", "3", "--per-family", "3", "--seed", "811", "--split", split]) == 0
        capsys.readouterr()
        paths[split] = output
    alignment = tmp_path / "alignment.json"
    assert main(["align", "--source", str(paths["train"] / "teacher.npz"), "--target", str(paths["train"] / "student.npz"), "--kind", "low_rank", "--source-role", "teacher", "--target-role", "student", "--output", str(alignment)]) == 0
    capsys.readouterr()
    output = tmp_path / "scorecard.json"
    assert main(["ablate", "--student", str(paths["train"] / "student.npz"), "--teacher", str(paths["train"] / "teacher.npz"), "--alignment", str(alignment), "--validation-student", str(paths["validation"] / "student.npz"), "--validation-teacher", str(paths["validation"] / "teacher.npz"), "--holdout-student", str(paths["holdout"] / "student.npz"), "--holdout-teacher", str(paths["holdout"] / "teacher.npz"), "--output", str(output)]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert output.exists()
    assert payload["schema_version"] == "faytuna-scorecard-v1"
    assert payload["holdout_used"] is True
    assert all(case["status"] in {"reference", "improved", "unchanged", "degraded", "rejected", "insufficient_data"} for case in payload["entries"])
    assert all("NaN" not in json.dumps(case) and "Infinity" not in json.dumps(case) for case in payload["entries"])
