"""Artifact-first paired teacher/student workflows for local experiments."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Sequence

import json
import numpy as np

from .artifacts import load_traces, save_traces
from .connectors import SyntheticConnector
from .flow import calibrate_flow_confidence, fit_flow_transfer
from .geometry import fit_alignment
from .observation import ObservationProtocol
from .probes import ProbeGenerator, ProbeGeneratorConfig
from .synthetic import random_stable_system
from .types import AlignmentResult, FlowFitResult, Probe, ProbeSplit, TrajectoryTrace


@dataclass(frozen=True)
class TraceBundle:
    """The paired artifact boundary consumed by every transfer stage."""

    teacher: tuple[TrajectoryTrace, ...]
    student: tuple[TrajectoryTrace, ...]
    stage: str
    split: str = "train"
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.teacher or len(self.teacher) != len(self.student):
            raise ValueError("TraceBundle requires equally sized non-empty paired traces")
        if any(a.probe_id != b.probe_id for a, b in zip(self.teacher, self.student)):
            raise ValueError("teacher and student trace artifacts must have matching probe IDs")
        if self.split not in {"train", "validation", "holdout"}:
            raise ValueError("TraceBundle split must be train, validation, or holdout")


def save_bundle(bundle: TraceBundle, directory: str | Path) -> Path:
    destination = Path(directory)
    destination.mkdir(parents=True, exist_ok=True)
    save_traces(bundle.teacher, destination / "teacher.npz")
    save_traces(bundle.student, destination / "student.npz")
    manifest = {"format": "faytuna-paired-traces-v1", "stage": bundle.stage, "split": bundle.split, "metadata": dict(bundle.metadata or {}), "teacher": "teacher.npz", "student": "student.npz"}
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    return destination


def load_bundle(directory: str | Path) -> TraceBundle:
    source = Path(directory)
    manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("format") != "faytuna-paired-traces-v1":
        raise ValueError("unsupported paired trace bundle")
    teacher = tuple(load_traces(source / manifest["teacher"]))
    student = tuple(load_traces(source / manifest["student"]))
    return TraceBundle(teacher, student, str(manifest["stage"]), str(manifest.get("split", "train")), manifest.get("metadata", {}))


def _paired_probes(config: ProbeGeneratorConfig) -> tuple[ProbeSplit, ProbeSplit]:
    generated = ProbeGenerator(config).generate()
    split = ProbeGenerator(config).split(generated)
    # A teacher and a student receive separate charts but share IDs, payloads,
    # perturbation pairing, and split assignment. The connector boundary is
    # therefore the only place that changes when a real model is introduced.
    return split, split


def collect_synthetic_pair(*, teacher_dim: int, student_dim: int, teacher_layers: int, student_layers: int, per_family: int = 5, seed: int = 0, split: str = "train") -> TraceBundle:
    if teacher_dim < 1 or student_dim < 1 or teacher_layers < 1 or student_layers < 1:
        raise ValueError("synthetic stage dimensions and layer counts must be positive")
    rng = np.random.default_rng(seed)
    teacher_projection = rng.normal(size=(teacher_dim, student_dim))
    teacher_split, _ = _paired_probes(ProbeGeneratorConfig(state_dim=teacher_dim, per_family=per_family, seed=seed))
    selected = tuple(getattr(teacher_split, split))
    student_probes = tuple(Probe(probe.probe_id, probe.family, probe.payload, probe.initial_state @ teacher_projection, pair_id=probe.pair_id, perturbation=None if probe.perturbation is None else probe.perturbation @ teacher_projection, split=probe.split) for probe in selected)
    teacher_connector = SyntheticConnector(random_stable_system(layers=teacher_layers, state_dim=teacher_dim, seed=seed + 11, nonlinear=True), model_id=f"synthetic-teacher-{teacher_dim}d-{teacher_layers}l")
    student_connector = SyntheticConnector(random_stable_system(layers=student_layers, state_dim=student_dim, seed=seed + 17, nonlinear=True), model_id=f"synthetic-student-{student_dim}d-{student_layers}l")
    protocol = ObservationProtocol()
    teacher_traces = tuple(protocol.collect(teacher_connector, selected, seed=seed + 23))
    student_traces = tuple(protocol.collect(student_connector, student_probes, seed=seed + 29))
    return TraceBundle(teacher_traces, student_traces, "synthetic_ground_truth", split, {"teacher_dim": teacher_dim, "student_dim": student_dim, "teacher_layers": teacher_layers, "student_layers": student_layers, "seed": seed, "both_models_observed": True})


def collect_staged_synthetic(*, per_family: int = 4, seed: int = 0) -> dict[str, TraceBundle]:
    """Create the three documented local stages without model checkpoints."""

    return {
        "synthetic_ground_truth": collect_synthetic_pair(teacher_dim=4, student_dim=4, teacher_layers=5, student_layers=5, per_family=per_family, seed=seed, split="train"),
        "tiny_small": collect_synthetic_pair(teacher_dim=5, student_dim=3, teacher_layers=6, student_layers=4, per_family=per_family, seed=seed + 1, split="train"),
        "larger_teacher_smaller_student": collect_synthetic_pair(teacher_dim=8, student_dim=4, teacher_layers=8, student_layers=5, per_family=per_family, seed=seed + 2, split="train"),
    }


def collect_synthetic_pair_splits(*, teacher_dim: int, student_dim: int, teacher_layers: int, student_layers: int, per_family: int = 5, seed: int = 0) -> dict[str, TraceBundle]:
    """Collect train, validation, and holdout artifacts with identical policies."""

    return {split: collect_synthetic_pair(teacher_dim=teacher_dim, student_dim=student_dim, teacher_layers=teacher_layers, student_layers=student_layers, per_family=per_family, seed=seed, split=split) for split in ("train", "validation", "holdout")}


def collect_staged_synthetic_splits(*, per_family: int = 4, seed: int = 0) -> dict[str, dict[str, TraceBundle]]:
    """Run all three local stages with disjoint labelled probe artifacts."""

    configurations = {
        "synthetic_ground_truth": (4, 4, 5, 5),
        "tiny_small": (5, 3, 6, 4),
        "larger_teacher_smaller_student": (8, 4, 8, 5),
    }
    return {name: collect_synthetic_pair_splits(teacher_dim=teacher_dim, student_dim=student_dim, teacher_layers=teacher_layers, student_layers=student_layers, per_family=per_family, seed=seed + index) for index, (name, (teacher_dim, student_dim, teacher_layers, student_layers)) in enumerate(configurations.items())}


def fit_bundle(bundle: TraceBundle, *, validation: TraceBundle | None = None, kind: str = "low_rank", signature_mode: str = "full") -> tuple[AlignmentResult, FlowFitResult]:
    teacher_initial = np.asarray([trace.hidden_states[0] for trace in bundle.teacher])
    student_initial = np.asarray([trace.hidden_states[0] for trace in bundle.student])
    rank = min(teacher_initial.shape[1], student_initial.shape[1])
    alignment = fit_alignment(teacher_initial, student_initial, kind=kind, rank=rank if kind in {"low_rank", "whitened_orthogonal", "riemannian"} else None, source_role="teacher", target_role="student")
    fit = fit_flow_transfer(bundle.student, bundle.teacher, alignment, signature_mode=signature_mode)
    if validation is not None:
        if any(a.probe_id != b.probe_id for a, b in zip(validation.teacher, validation.student)):
            raise ValueError("validation teacher/student probes are not paired")
        fit = calibrate_flow_confidence(fit, validation.student, validation.teacher, alignment, signature_mode=signature_mode)
    return alignment, fit
