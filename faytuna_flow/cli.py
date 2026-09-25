"""Command line interface for a complete small-scale flow-transfer workflow."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from .artifacts import load_alignment, load_flow, load_traces, save_alignment, save_flow, save_traces
from .connectors import SyntheticConnector
from .flow import calibrate_flow_confidence, fit_flow_transfer
from .geometry import fit_alignment, fit_depth_conditioned_alignment, select_dynamic_depth_conditioned_alignment, trajectory_alignment_samples
from .observation import ObservationProtocol
from .probes import ProbeGenerator, ProbeGeneratorConfig
from .report import make_report, report_json
from .solver import TrustRegionConfig, solve_flow_correction
from .surgery import apply_surgery, build_surgery_plan, export_weights, import_weights
from .synthetic import random_stable_system
from .validation import validate_causal_transfer
from .types import TensorTransitionMapping
from .pipeline import collect_synthetic_pair, load_bundle, save_bundle
from .scorecard import run_transfer_scorecard
from .model_families import GPT2_XL_TO_SMALL, get_model_family_preset, inspect_gpt2_variant, preflight_gpt2_pair, strict_json_payload
from .runtime import preflight_stock_runtime
from .adaptive_transfer import AdaptiveTransferPolicy, run_gpt2_weight_tune
from .journal import ExperimentJournal, journal_adaptive, journal_flow_fit, journal_preflight, journal_scorecard, journal_surgery, journal_traces
from .benchmark import BenchmarkConfig, run_benchmark
from .forensics import audit_flow_fit
from .gpt2 import GPT2TensorLiftMapping, build_gpt2_surgery_plan, export_gpt2_checkpoint_pair


def _add_journal_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--journal", required=False, help="write deterministic structured JSONL experiment events")
    parser.add_argument("--journal-summary", required=False, help="human-readable journal summary path")


def _journal(args: argparse.Namespace, *, run_id: str, seed: int = 0) -> ExperimentJournal | None:
    if not getattr(args, "journal", None):
        return None
    return ExperimentJournal(args.journal, args.journal_summary, run_id=run_id, seed=seed)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="faytuna-flow", description="Emergent-flow observation, transport, constrained surgery, and causal diagnostics")
    sub = parser.add_subparsers(dest="command", required=True)
    inspect = sub.add_parser("inspect", help="show truthful connector capabilities")
    inspect.add_argument("--model", choices=["synthetic", "gpt2-xl-to-small"], default="synthetic")
    collect = sub.add_parser("collect-trace", help="collect synthetic traces without checkpoints")
    collect.add_argument("--output", required=True)
    collect.add_argument("--layers", type=int, default=6)
    collect.add_argument("--dimension", type=int, default=4)
    collect.add_argument("--probes", type=int, default=12)
    _add_journal_args(collect)
    paired = sub.add_parser("collect-paired", help="collect paired teacher/student trace artifacts")
    paired.add_argument("--output", required=True)
    paired.add_argument("--teacher-dimension", type=int, default=6)
    paired.add_argument("--student-dimension", type=int, default=3)
    paired.add_argument("--teacher-layers", type=int, default=7)
    paired.add_argument("--student-layers", type=int, default=4)
    paired.add_argument("--per-family", type=int, default=4)
    paired.add_argument("--seed", type=int, default=17)
    paired.add_argument("--split", choices=["train", "validation", "holdout"], default="train")
    _add_journal_args(paired)
    align = sub.add_parser("align", help="fit a cross-model latent geometry map")
    align.add_argument("--source", required=True)
    align.add_argument("--target", required=True)
    align.add_argument("--output", required=True)
    align.add_argument("--kind", choices=["whitened_orthogonal", "riemannian", "affine", "low_rank", "ot_barycentric"], default="whitened_orthogonal")
    align.add_argument("--source-role", choices=["teacher", "student", "source"], default="teacher")
    align.add_argument("--target-role", choices=["teacher", "student", "target"], default="student")
    align.add_argument("--max-dense-features", type=int, default=5_000_000)
    align.add_argument("--scalable-rank", type=int, default=64)
    align.add_argument("--scalable-seed", type=int, default=0)
    align.add_argument("--scope", choices=["initial", "trajectory"], default="initial", help="fit from initial states or all observed depths with explicit interpolation")
    align.add_argument("--depth-conditioned", action="store_true", help="fit train-only local affine/low-rank maps at existing source depths; requires --scope trajectory")
    align.add_argument("--dynamic-aware", action="store_true", help="explicitly fit transition/Jacobian-aware depth maps and select weights on validation")
    align.add_argument("--validation-source", required=False, help="validation source trace artifact for --dynamic-aware selection")
    align.add_argument("--validation-target", required=False, help="validation target trace artifact for --dynamic-aware selection")
    _add_journal_args(align)
    fit = sub.add_parser("fit-flow", help="fit and transport the vector field")
    fit.add_argument("--student", required=True)
    fit.add_argument("--teacher", required=True)
    fit.add_argument("--alignment", required=True)
    fit.add_argument("--output", required=True)
    fit.add_argument("--validation-student", required=False)
    fit.add_argument("--validation-teacher", required=False)
    fit.add_argument("--signature-mode", choices=["none", "differential", "full"], default="full")
    fit.add_argument("--max-dense-features", type=int, default=5_000_000)
    fit.add_argument("--scalable-rank", type=int, default=64)
    fit.add_argument("--scalable-seed", type=int, default=0)
    _add_journal_args(fit)
    audit = sub.add_parser("audit-fit", help="audit an existing flow fit without model observation or checkpoint loading")
    audit.add_argument("--student", required=True)
    audit.add_argument("--teacher", required=True)
    audit.add_argument("--alignment", required=True)
    audit.add_argument("--scorecard", required=False, help="optional existing strict scorecard JSON for target identity checks")
    audit.add_argument("--signature-mode", choices=["none", "differential", "full"], default="full")
    audit.add_argument("--max-dense-features", type=int, default=5_000_000)
    audit.add_argument("--scalable-rank", type=int, default=64)
    audit.add_argument("--scalable-seed", type=int, default=0)
    audit.add_argument("--output", required=False)
    _add_journal_args(audit)
    scorecard = sub.add_parser("scorecard", aliases=["ablate"], help="scorecard alias: compare static, delta-flow, differential, full-flow, and controls")
    scorecard.add_argument("--student", required=True)
    scorecard.add_argument("--teacher", required=True)
    scorecard.add_argument("--alignment", required=True)
    scorecard.add_argument("--validation-student", required=False)
    scorecard.add_argument("--validation-teacher", required=False)
    scorecard.add_argument("--holdout-student", required=False)
    scorecard.add_argument("--holdout-teacher", required=False)
    scorecard.add_argument("--seed", type=int, default=0)
    scorecard.add_argument("--output", required=False)
    _add_journal_args(scorecard)
    preflight = sub.add_parser("gpt2-preflight", help="check GPT-2 XL/small architecture and optional stock llama.cpp conversion/runtime paths")
    preflight.add_argument("--teacher-config", required=True)
    preflight.add_argument("--student-config", required=True)
    preflight.add_argument("--gguf", required=False)
    preflight.add_argument("--llama-cli", required=False)
    preflight.add_argument("--llama-perplexity", required=False)
    preflight.add_argument("--prompt-file", required=False)
    preflight.add_argument("--converter-script", required=False)
    preflight.add_argument("--checkpoint-dir", required=False)
    _add_journal_args(preflight)
    gpt2_surgery = sub.add_parser("gpt2-surgery", help="explicit GPT-2 Conv1D chart lifting and ordinary HF baseline/candidate export; local checkpoint only")
    gpt2_surgery.add_argument("--checkpoint-dir", required=True, help="local student/teacher HF directory; local_files_only is enforced")
    gpt2_surgery.add_argument("--variant", choices=["gpt2-small", "gpt2-xl"], default="gpt2-small")
    gpt2_surgery.add_argument("--student", required=True, help="train student trace artifact")
    gpt2_surgery.add_argument("--teacher", required=True, help="train teacher trace artifact")
    gpt2_surgery.add_argument("--alignment", required=True)
    gpt2_surgery.add_argument("--mapping", required=True, help="JSON list of explicit {transition_index,tensor_name,side,block_index} entries")
    gpt2_surgery.add_argument("--sequence-length", type=int, required=True)
    gpt2_surgery.add_argument("--output", required=True, help="diagnostic plan JSON, or export directory in apply mode")
    gpt2_surgery.add_argument("--mode", choices=["diagnostic", "apply", "experimental"], default="diagnostic", help="diagnostic only, guarded apply, or explicit untrusted raw-correction export")
    gpt2_surgery.add_argument("--gain", type=float, default=1.0)
    gpt2_surgery.add_argument("--max-chart-bytes", type=int, default=512 * 1024 * 1024)
    gpt2_surgery.add_argument("--max-cross-token-ratio", type=float, default=0.5)
    _add_journal_args(gpt2_surgery)
    tune = sub.add_parser("tune", help="central adaptive transfer policy with validation and holdout guards")
    tune.add_argument("--student", required=True)
    tune.add_argument("--teacher", required=True)
    tune.add_argument("--alignment", required=True)
    tune.add_argument("--validation-student", required=False)
    tune.add_argument("--validation-teacher", required=False)
    tune.add_argument("--holdout-student", required=False)
    tune.add_argument("--holdout-teacher", required=False)
    tune.add_argument("--mode", choices=["safe", "experimental", "experimental_untrusted"], default="safe")
    tune.add_argument("--gains", default=None, help="optional comma-separated gain grid; policy bounds it by mode")
    tune.add_argument("--block-size", type=int, default=2)
    tune.add_argument("--min-effect", type=float, default=1e-3)
    tune.add_argument("--backend", choices=["auto", "dense", "scalable"], default="auto")
    tune.add_argument("--depth-strategy", choices=["auto", "existing_monotone", "gap_aware_monotone"], default="auto")
    tune.add_argument("--block-order", choices=["auto", "depth", "confidence"], default="auto")
    tune.add_argument("--line-search", choices=["auto", "backtracking", "grid"], default="auto")
    tune.add_argument("--checkpoint-dir", required=False, help="optional local GPT-2 small checkpoint for the same policy's weight/A-B orchestration")
    tune.add_argument("--mapping", required=False, help="explicit GPT-2 tensor mapping JSON; required with --checkpoint-dir")
    tune.add_argument("--output-root", required=False, help="checkpoint/A-B output directory; required with --checkpoint-dir")
    tune.add_argument("--sequence-length", type=int, required=False, help="sequence length used by the GPT-2 chart lift")
    tune.add_argument("--max-dense-features", type=int, default=5_000_000)
    tune.add_argument("--scalable-rank", type=int, default=64)
    tune.add_argument("--scalable-seed", type=int, default=17)
    tune.add_argument("--max-chart-bytes", type=int, default=512 * 1024 * 1024)
    tune.add_argument("--max-cross-token-ratio", type=float, default=0.5)
    tune.add_argument("--activation-lift", choices=["heuristic", "teacher_flow"], default="heuristic", help="heuristic chart contraction, or train-trace teacher-flow targets plus exact activation ridge LS")
    tune.add_argument("--teacher-flow-target-ratio", type=float, default=0.05, help="per-token teacher-flow target cap relative to the observed student block output/input norm")
    tune.add_argument("--site-policy", choices=["mlp_residual_only", "dual_residual"], default="dual_residual", help="dual_residual or mlp_residual_only")
    tune.add_argument("--dual-residual-ratio", type=float, default=0.35, help="attention fraction for dual residual steering")
    tune.add_argument("--depth-schedule", choices=["none", "flat", "sine", "boost_deep"], default="boost_deep", help="depth gain curve")
    tune.add_argument("--use-2jet", action="store_true", default=True, help="use 2-jet Heun predictor-corrector along flow")
    tune.add_argument("--output", required=False)
    _add_journal_args(tune)
    surgery = sub.add_parser("apply-surgery", help="apply safe flow corrections to ordinary tensors")
    surgery.add_argument("--weights", required=True)
    surgery.add_argument("--student", required=True)
    surgery.add_argument("--teacher", required=True)
    surgery.add_argument("--alignment", required=True)
    surgery.add_argument("--mapping", required=True, help="JSON list of explicit {transition_index,tensor_name,orientation} entries")
    surgery.add_argument("--output", required=True)
    surgery.add_argument("--gain", type=float, default=1.0)
    _add_journal_args(surgery)
    validate = sub.add_parser("validate", help="report geometric and trace-level validation diagnostics")
    validate.add_argument("--trace", required=False, help="single trace artifact for diagnostic validation")
    validate.add_argument("--teacher", required=False, help="teacher trace artifact for causal validation")
    validate.add_argument("--baseline", required=False, help="baseline student trace artifact")
    validate.add_argument("--intervened", required=False, help="intervened student trace artifact")
    validate.add_argument("--alignment", required=False)
    _add_journal_args(validate)
    report = sub.add_parser("report", help="write a JSON report from a trace artifact")
    report.add_argument("--trace", required=True)
    report.add_argument("--alignment", required=False)
    report.add_argument("--output", required=False)
    _add_journal_args(report)
    export = sub.add_parser("export", help="round-trip ordinary weight tensors")
    export.add_argument("--weights", required=True)
    export.add_argument("--output", required=True)
    _add_journal_args(export)
    benchmark = sub.add_parser("benchmark-observation", aliases=["benchmark"], help="benchmark full single vs batched differential observation on an actual-size synthetic chart")
    benchmark.add_argument("--state-dim", type=int, default=16 * 1600)
    benchmark.add_argument("--layers", type=int, default=48)
    benchmark.add_argument("--probes", type=int, default=1)
    benchmark.add_argument("--jacobian-rank", type=int, default=8)
    benchmark.add_argument("--hessian-rank", type=int, default=4)
    benchmark.add_argument("--seed", type=int, default=17)
    benchmark.add_argument("--memory-interval", type=float, default=0.02)
    benchmark.add_argument("--output", required=False, help="write benchmark-v1 JSON")
    benchmark.add_argument("--progress-jsonl", required=False, help="write strict benchmark progress JSONL")
    benchmark.add_argument("--progress-log", required=False, help="write human-readable benchmark progress")
    return parser


def _probe_batch(count: int, dimension: int, seed: int = 5):
    config = ProbeGeneratorConfig(state_dim=dimension, per_family=max(3, count // 7), seed=seed)
    probes = ProbeGenerator(config).generate()
    return probes[:count]


def _load_tensor_mapping(path: str) -> list[TensorTransitionMapping]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise SystemExit("tensor mapping must be a JSON list")
    return [TensorTransitionMapping(int(item["transition_index"]), str(item["tensor_name"]), str(item.get("orientation", "row_right")), str(item.get("role", "dense_square"))) for item in payload]


def _load_gpt2_mapping(path: str) -> list[GPT2TensorLiftMapping]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise SystemExit("GPT-2 mapping must be a JSON list")
    return [
        GPT2TensorLiftMapping(
            int(item["transition_index"]),
            str(item["tensor_name"]),
            str(item.get("side", "input")),
            None if item.get("block_index") is None else int(item["block_index"]),
        )
        for item in payload
    ]


def _gpt2_plan_payload(plan) -> dict:
    return strict_json_payload({
        "schema_version": "faytuna-gpt2-surgery-plan-v1",
        "mode": plan.metadata.get("mode"),
        "metadata": dict(plan.metadata),
        "rollback_layers": list(plan.rollback_layers),
        "applied_tensors": list(plan.applied_tensors),
        "skipped_tensors": dict(plan.skipped_tensors),
        "updates": {name: {"shape": list(np.asarray(value).shape), "dtype": str(np.asarray(value).dtype)} for name, value in plan.updates.items() if name in plan.applied_tensors},
    })


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "inspect":
        if args.model == "gpt2-xl-to-small":
            print(json.dumps(strict_json_payload({"preset": get_model_family_preset(args.model).to_dict(), "observation_stage": "HF/PyTorch explicit GPT2Connector", "runtime_stage": "stock llama.cpp after checkpoint/GGUF export", "hidden_state_instrumentation": False}), indent=2, allow_nan=False))
            return 0
        connector = SyntheticConnector(random_stable_system())
        print(json.dumps({"model_id": connector.model_id, "capabilities": connector.capabilities.to_dict(), "layer_ids": connector.layer_ids}, indent=2))
        return 0
    if args.command == "collect-trace":
        probes = _probe_batch(args.probes, args.dimension)
        connector = SyntheticConnector(random_stable_system(layers=args.layers, state_dim=args.dimension, seed=7), model_id="synthetic-demo")
        traces = ObservationProtocol().collect(connector, probes)
        save_traces(traces, args.output)
        journal = _journal(args, run_id="collect-trace", seed=5)
        if journal is not None:
            journal_traces(journal, traces, model_role="synthetic")
            journal.close()
        print(json.dumps({"output": str(args.output), "traces": len(traces), "state_dim": args.dimension}, indent=2))
        return 0
    if args.command == "collect-paired":
        bundle = collect_synthetic_pair(teacher_dim=args.teacher_dimension, student_dim=args.student_dimension, teacher_layers=args.teacher_layers, student_layers=args.student_layers, per_family=args.per_family, seed=args.seed, split=args.split)
        output = save_bundle(bundle, args.output)
        journal = _journal(args, run_id=f"collect-paired-{bundle.split}", seed=args.seed)
        if journal is not None:
            journal_traces(journal, bundle.teacher, model_role="teacher", model_metadata={"backend": "synthetic"})
            journal_traces(journal, bundle.student, model_role="student", model_metadata={"backend": "synthetic"})
            journal.close()
        print(json.dumps({"output": str(output), "teacher_traces": len(bundle.teacher), "student_traces": len(bundle.student), "split": bundle.split, "teacher_dimension": bundle.teacher[0].state_dim, "student_dimension": bundle.student[0].state_dim}, indent=2))
        return 0
    if args.command == "align":
        source = load_traces(args.source)
        target = load_traces(args.target)
        selection_report = None
        if args.dynamic_aware and not args.depth_conditioned:
            raise SystemExit("--dynamic-aware requires --depth-conditioned")
        if args.depth_conditioned and args.scope != "trajectory":
            raise SystemExit("--depth-conditioned requires --scope trajectory")
        if args.depth_conditioned and args.kind not in {"low_rank", "affine"}:
            raise SystemExit("--depth-conditioned supports only --kind low_rank or affine")
        if args.dynamic_aware and bool(args.validation_source) != bool(args.validation_target):
            raise SystemExit("--validation-source and --validation-target must be supplied together")
        if args.depth_conditioned:
            if args.dynamic_aware:
                if not args.validation_source:
                    raise SystemExit("--dynamic-aware requires disjoint --validation-source and --validation-target artifacts")
                result, selection_report = select_dynamic_depth_conditioned_alignment(
                    source,
                    target,
                    load_traces(args.validation_source),
                    load_traces(args.validation_target),
                    kind=args.kind,
                    source_role=args.source_role,
                    target_role=args.target_role,
                    max_dense_features=args.max_dense_features,
                    scalable_rank=args.scalable_rank,
                    scalable_seed=args.scalable_seed,
                )
            else:
                result = fit_depth_conditioned_alignment(
                    source,
                    target,
                    kind=args.kind,
                    source_role=args.source_role,
                    target_role=args.target_role,
                    max_dense_features=args.max_dense_features,
                    scalable_rank=args.scalable_rank,
                    scalable_seed=args.scalable_seed,
                )
            x, y, scope_metadata = trajectory_alignment_samples(source, target)
            scope_metadata = {
                **scope_metadata,
                "alignment_strategy": "depth_conditioned_dynamic_aware" if args.dynamic_aware else "depth_conditioned_local_maps",
                "depth_conditioned": True,
                "dynamic_aware": bool(args.dynamic_aware),
            }
        elif args.scope == "trajectory":
            x, y, scope_metadata = trajectory_alignment_samples(source, target)
            result = fit_alignment(x, y, kind=args.kind, source_role=args.source_role, target_role=args.target_role, max_dense_features=args.max_dense_features, scalable_rank=args.scalable_rank, scalable_seed=args.scalable_seed)
        else:
            x = np.asarray([t.hidden_states[0] for t in source])
            y = np.asarray([t.hidden_states[0] for t in target])
            scope_metadata = {"alignment_fit_scope": "initial_state_only", "sample_count": int(len(source)), "interpolation_creates_no_layer": True}
            result = fit_alignment(x, y, kind=args.kind, source_role=args.source_role, target_role=args.target_role, max_dense_features=args.max_dense_features, scalable_rank=args.scalable_rank, scalable_seed=args.scalable_seed)
        result = replace(result, metadata={**dict(result.metadata), **scope_metadata})
        save_alignment(result, args.output)
        journal = _journal(args, run_id="align")
        if journal is not None:
            journal.record("align", "alignment", scope="run", payload={"kind": result.kind, "source_role": args.source_role, "target_role": args.target_role, "source_shape": list(x.shape), "target_shape": list(y.shape), "paired_error": result.paired_error, "relational_error": result.relational_error, "cycle_error": result.cycle_error, "condition_number": result.condition_number, "ot_mass_error": result.ot_mass_error, "backend": result.metadata.get("representation"), "alignment_strategy": result.metadata.get("alignment_strategy", "global"), "depth_conditioned": bool(result.metadata.get("depth_conditioned", False)), "dynamic_aware": bool(result.metadata.get("dynamic_aware", False)), "depth_local_paired_error": result.metadata.get("depth_local_paired_error"), "global_shared_map_paired_error": result.metadata.get("global_shared_map_paired_error"), "depth_metrics": result.metadata.get("depth_metrics"), "dynamic_weights": result.metadata.get("dynamic_weights"), "validation_selection": selection_report, "feature_count": result.metadata.get("dense_map_feature_count"), "estimated_dense_bytes": result.metadata.get("dense_map_estimated_bytes"), "skipped_reason": result.metadata.get("dense_map_skipped_reason"), "source_projection_shape": result.metadata.get("source_projection_shape"), "target_projection_shape": result.metadata.get("target_projection_shape")})
            journal.close()
        print(json.dumps({"output": str(args.output), "paired_error": result.paired_error, "relational_error": result.relational_error, "backend": result.metadata.get("representation", "dense"), "alignment_strategy": result.metadata.get("alignment_strategy", "global"), "depth_conditioned": bool(result.metadata.get("depth_conditioned", False)), "dynamic_aware": bool(result.metadata.get("dynamic_aware", False)), "depth_local_paired_error": result.metadata.get("depth_local_paired_error"), "global_shared_map_paired_error": result.metadata.get("global_shared_map_paired_error"), "selected_candidate": result.metadata.get("selected_candidate"), "feature_count": result.metadata.get("dense_map_feature_count", int(result.source_dim * result.target_dim))}, indent=2, allow_nan=False))
        return 0
    if args.command == "fit-flow":
        student = load_traces(args.student)
        teacher = load_traces(args.teacher)
        alignment = load_alignment(args.alignment)
        result = fit_flow_transfer(student, teacher, alignment, signature_mode=args.signature_mode, max_dense_features=args.max_dense_features, scalable_rank=args.scalable_rank, scalable_seed=args.scalable_seed)
        if bool(args.validation_student) != bool(args.validation_teacher):
            raise SystemExit("--validation-student and --validation-teacher must be supplied together")
        if args.validation_student:
            result = calibrate_flow_confidence(result, load_traces(args.validation_student), load_traces(args.validation_teacher), alignment, signature_mode=args.signature_mode)
        save_flow(result.transported_teacher, args.output)
        correction_path = Path(args.output).with_suffix(Path(args.output).suffix + ".correction.npz")
        np.savez_compressed(correction_path, matrices=result.correction_matrices, biases=result.correction_biases, confidence=result.confidence, quadratic=np.asarray(result.correction_quadratic) if result.correction_quadratic is not None else np.zeros((0,)))
        journal = _journal(args, run_id="fit-flow")
        if journal is not None:
            journal_traces(journal, student, model_role="student")
            journal_traces(journal, teacher, model_role="teacher")
            journal_flow_fit(journal, result)
            journal.close()
        print(json.dumps({"output": str(args.output), "training_student_flow_error": result.validation_error, "mean_confidence": float(np.mean(result.confidence)), "requested_signature_mode": args.signature_mode, "effective_signature_mode": result.metadata.get("effective_signature_mode", args.signature_mode), "backend": result.metadata.get("scalable_backend", "dense_local_field"), "dense_memory_guard": result.metadata.get("dense_memory_guard"), "confidence_calibrated": bool(result.metadata.get("confidence_calibrated", False)), "scorecard_required_for_transfer_decision": True, "depth_report": None if result.depth_report is None else {"gap_count": result.depth_report.gap_count, "matched_fraction": result.depth_report.matched_fraction}, "capacity_bottleneck": None if result.capacity_diagnostics is None else result.capacity_diagnostics.bottleneck}, indent=2, allow_nan=False))
        return 0
    if args.command == "audit-fit":
        student = load_traces(args.student)
        teacher = load_traces(args.teacher)
        alignment = load_alignment(args.alignment)
        fit = fit_flow_transfer(
            student,
            teacher,
            alignment,
            signature_mode=args.signature_mode,
            max_dense_features=args.max_dense_features,
            scalable_rank=args.scalable_rank,
            scalable_seed=args.scalable_seed,
        )
        scorecard = None if not args.scorecard else json.loads(Path(args.scorecard).read_text(encoding="utf-8"))
        payload = audit_flow_fit(fit, alignment, train_student=student, train_teacher=teacher, scorecard=scorecard)
        rendered = json.dumps(payload, indent=2, allow_nan=False)
        if args.output:
            Path(args.output).write_text(rendered, encoding="utf-8")
        journal = _journal(args, run_id="audit-fit")
        if journal is not None:
            journal.record("audit-fit", "forensic-audit", scope="run", payload=payload)
            journal.close()
        print(rendered)
        return 0 if payload["status"] in {"consistent", "consistent_with_bottleneck"} else 2
    if args.command in {"ablate", "scorecard"}:
        student = load_traces(args.student)
        teacher = load_traces(args.teacher)
        alignment = load_alignment(args.alignment)
        if bool(args.validation_student) != bool(args.validation_teacher):
            raise SystemExit("--validation-student and --validation-teacher must be supplied together")
        if bool(args.holdout_student) != bool(args.holdout_teacher):
            raise SystemExit("--holdout-student and --holdout-teacher must be supplied together")
        validation_student = None if not args.validation_student else load_traces(args.validation_student)
        validation_teacher = None if not args.validation_teacher else load_traces(args.validation_teacher)
        holdout_student = None if not args.holdout_student else load_traces(args.holdout_student)
        holdout_teacher = None if not args.holdout_teacher else load_traces(args.holdout_teacher)
        try:
            result = run_transfer_scorecard(
                student,
                teacher,
                alignment,
                validation_student=validation_student,
                validation_teacher=validation_teacher,
                holdout_student=holdout_student,
                holdout_teacher=holdout_teacher,
                seed=args.seed,
            )
        except ValueError as error:
            payload = {
                "schema_version": "faytuna-scorecard-v1",
                "status": "rejected",
                "decision_status": "invalid_split_or_input",
                "validation_used": bool(validation_student is not None and validation_teacher is not None),
                "holdout_used": bool(holdout_student is not None and holdout_teacher is not None),
                "reason": f"{type(error).__name__}: {error}",
                "target_fit_split": "train",
                "metric_scope": "transported_teacher_operator_approximation",
            }
            rendered = json.dumps(payload, indent=2, allow_nan=False)
            if args.output:
                Path(args.output).write_text(rendered, encoding="utf-8")
            print(rendered)
            return 2
        journal = _journal(args, run_id="scorecard", seed=args.seed)
        if journal is not None:
            journal_traces(journal, student, model_role="student")
            journal_traces(journal, teacher, model_role="teacher")
            if validation_student is not None:
                journal_traces(journal, validation_student, stage="validation", model_role="student")
                journal_traces(journal, validation_teacher, stage="validation", model_role="teacher")
            if holdout_student is not None:
                journal_traces(journal, holdout_student, stage="holdout", model_role="student")
                journal_traces(journal, holdout_teacher, stage="holdout", model_role="teacher")
            journal_scorecard(journal, result)
            journal.close()
        payload = result.to_dict()
        rendered = json.dumps(payload, indent=2, allow_nan=False)
        if args.output:
            Path(args.output).write_text(rendered, encoding="utf-8")
        print(rendered)
        return 0 if result.holdout_used else 2
    if args.command == "gpt2-preflight":
        payload = preflight_stock_runtime(
            args.teacher_config,
            args.student_config,
            gguf_path=args.gguf,
            llama_cli=args.llama_cli,
            llama_perplexity=args.llama_perplexity,
            prompt_file=args.prompt_file,
            converter_script=args.converter_script,
            converter_checkpoint_dir=args.checkpoint_dir,
            preset=GPT2_XL_TO_SMALL,
        )
        journal = _journal(args, run_id="gpt2-preflight")
        if journal is not None:
            journal_preflight(journal, payload)
            journal.close()
        print(json.dumps(payload, indent=2, allow_nan=False))
        return 0 if payload["status"] in {"observation_ready_runtime_not_checked", "ready_for_stock_runtime"} else 2
    if args.command == "gpt2-surgery":
        checkpoint = Path(args.checkpoint_dir)
        if not checkpoint.is_dir():
            raise SystemExit(f"GPT-2 checkpoint directory does not exist locally: {checkpoint}")
        try:
            import torch  # noqa: F401
            from transformers import AutoModelForCausalLM
        except ImportError as error:
            raise SystemExit(f"gpt2-surgery requires optional torch and transformers; no checkpoint was loaded: {error}")
        try:
            model = AutoModelForCausalLM.from_pretrained(str(checkpoint), local_files_only=True)
        except Exception as error:
            raise SystemExit(f"local GPT-2 checkpoint load failed with local_files_only=True: {error}")
        model.eval()
        expected_variant = GPT2_XL_TO_SMALL.student if args.variant == "gpt2-small" else GPT2_XL_TO_SMALL.teacher
        architecture_check = inspect_gpt2_variant(getattr(model, "config", None), expected_variant)
        if not architecture_check["supported"]:
            raise SystemExit("GPT-2 surgery preflight rejected checkpoint: " + "; ".join(architecture_check["reasons"]))
        student = load_traces(args.student)
        teacher = load_traces(args.teacher)
        alignment = load_alignment(args.alignment)
        fit = fit_flow_transfer(student, teacher, alignment, signature_mode="full")
        guarded = solve_flow_correction(fit, config=TrustRegionConfig())
        # Safe apply uses the solver's accepted/zeroed correction exactly as
        # before. Experimental mode is explicit and carries the raw finite
        # correction tensors for forensic A/B evaluation while preserving the
        # solver's acceptance and rollback evidence in the plan.
        constrained = guarded if args.mode != "experimental" else replace(
            guarded,
            matrices=fit.correction_matrices.copy(),
            biases=fit.correction_biases.copy(),
            quadratic_terms=None if fit.correction_quadratic is None else fit.correction_quadratic.copy(),
        )
        config = getattr(model, "config", None)
        hidden_size = int(getattr(config, "n_embd", getattr(config, "hidden_size", 0)))
        if hidden_size < 1:
            raise SystemExit("GPT-2 checkpoint config does not expose n_embd/hidden_size")
        mapping = _load_gpt2_mapping(args.mapping)
        plan = build_gpt2_surgery_plan(
            model.state_dict(),
            constrained,
            chart_projection=fit.student.chart_projection,
            sequence_length=args.sequence_length,
            hidden_size=hidden_size,
            mapping=mapping,
            gain=args.gain,
            mode=args.mode,
            max_chart_bytes=args.max_chart_bytes,
            max_cross_token_ratio=args.max_cross_token_ratio,
        )
        payload = _gpt2_plan_payload(plan)
        payload["preflight"] = architecture_check
        if args.mode == "apply":
            output = export_gpt2_checkpoint_pair(model, args.output, variant=args.variant, experimental_plan=plan, metadata={"student_trace": str(args.student), "teacher_trace": str(args.teacher), "alignment": str(args.alignment), "clean_baseline_unchanged": True})
            payload["export"] = {"output": str(output), "status": "ordinary_hf_baseline_and_candidate_exported"}
            Path(args.output, "surgery-plan.json").write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
        else:
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            Path(args.output).write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
        journal = _journal(args, run_id="gpt2-surgery")
        if journal is not None:
            journal_surgery(journal, plan)
            journal.close()
        print(json.dumps(payload, indent=2, allow_nan=False))
        return 0 if args.mode in {"apply", "experimental"} and plan.applied_tensors else (0 if args.mode == "diagnostic" else 2)
    if args.command == "tune":
        if args.checkpoint_dir:
            if not args.mapping or not args.output_root or args.sequence_length is None:
                raise SystemExit("GPT-2 tune requires --mapping, --output-root, and --sequence-length with --checkpoint-dir")
            try:
                gains = None if args.gains is None else tuple(float(value.strip()) for value in args.gains.split(",") if value.strip())
            except ValueError as error:
                raise SystemExit(f"--gains must be a comma-separated finite numeric list: {error}")
            payload = run_gpt2_weight_tune(
                args.checkpoint_dir,
                args.student,
                args.teacher,
                args.alignment,
                args.mapping,
                args.output_root,
                sequence_length=args.sequence_length,
                gains=(0.01, 0.05, 0.1) if gains is None else gains,
                mode=args.mode,
                seed=17,
                max_dense_features=args.max_dense_features,
                scalable_rank=args.scalable_rank,
                scalable_seed=args.scalable_seed,
                max_chart_bytes=args.max_chart_bytes,
                max_cross_token_ratio=args.max_cross_token_ratio,
                activation_lift=args.activation_lift,
                teacher_flow_target_ratio=args.teacher_flow_target_ratio,
                site_policy=args.site_policy,
                dual_residual_ratio=args.dual_residual_ratio,
                depth_schedule=None if args.depth_schedule in {None, "none", "flat"} else args.depth_schedule,
                use_2jet=args.use_2jet,
            )
            journal = _journal(args, run_id="adaptive-gpt2-tune")
            if journal is not None:
                journal.record("adaptive-gpt2-tune", "gpt2-tune", scope="run", payload=payload)
                journal.close()
            rendered = json.dumps(payload, indent=2, allow_nan=False)
            if args.output:
                Path(args.output).write_text(rendered, encoding="utf-8")
            print(rendered)
            return 0 if payload.get("status") in {"ok", "diagnostic"} else 2
        student = load_traces(args.student)
        teacher = load_traces(args.teacher)
        alignment = load_alignment(args.alignment)
        if bool(args.validation_student) != bool(args.validation_teacher):
            raise SystemExit("--validation-student and --validation-teacher must be supplied together")
        if bool(args.holdout_student) != bool(args.holdout_teacher):
            raise SystemExit("--holdout-student and --holdout-teacher must be supplied together")
        try:
            gains = None if args.gains is None else tuple(float(value.strip()) for value in args.gains.split(",") if value.strip())
        except ValueError as error:
            raise SystemExit(f"--gains must be a comma-separated finite numeric list: {error}")
        fit = fit_flow_transfer(student, teacher, alignment, signature_mode="full")
        target = fit.transported_teacher
        target_metadata = dict(target.metadata)
        target_metadata.update({"target_fit_split": "train", "metric_scope": "transported_teacher_operator_approximation"})
        policy = AdaptiveTransferPolicy(
            mode=args.mode,
            gain_grid=gains,
            block_size=args.block_size,
            min_observable_effect=args.min_effect,
            backend=args.backend,
            depth_strategy=args.depth_strategy,
            block_order=args.block_order,
            line_search=args.line_search,
        )
        result = policy.tune(
            fit,
            replace(target, metadata=target_metadata),
            alignment=alignment,
            validation_student=None if not args.validation_student else load_traces(args.validation_student),
            validation_teacher=None if not args.validation_teacher else load_traces(args.validation_teacher),
            holdout_student=None if not args.holdout_student else load_traces(args.holdout_student),
            holdout_teacher=None if not args.holdout_teacher else load_traces(args.holdout_teacher),
        )
        journal = _journal(args, run_id="adaptive-transfer")
        if journal is not None:
            journal_traces(journal, student, model_role="student")
            journal_traces(journal, teacher, model_role="teacher")
            journal_flow_fit(journal, fit)
            journal_adaptive(journal, result)
            journal.close()
        rendered = json.dumps(result.to_dict(), indent=2, allow_nan=False)
        if args.output:
            Path(args.output).write_text(rendered, encoding="utf-8")
            selection = result.metadata.get("policy_selection", {})
            summary = [
                "Faytuna adaptive transfer policy",
                f"status: {result.status}",
                f"mode: {result.mode}",
                f"validation_used: {result.validation_used}",
                f"holdout_used: {result.holdout_used}",
                f"final_alpha: {result.metadata.get('final_alpha')}",
                f"backend: {selection.get('backend')}",
                f"depth_strategy: {selection.get('depth_strategy')}",
                f"block_order: {selection.get('block_order')}",
                f"line_search: {selection.get('line_search')}",
                f"tensor_families: {', '.join(selection.get('tensor_families', ())) if selection.get('tensor_families') else None}",
                "holdout_is_final_verdict: true",
            ]
            Path(args.output).with_suffix(".txt").write_text("\n".join(summary) + "\n", encoding="utf-8")
        print(rendered)
        return 0 if result.holdout_used else 2
    if args.command == "apply-surgery":
        student = load_traces(args.student)
        teacher = load_traces(args.teacher)
        alignment = load_alignment(args.alignment)
        fit = fit_flow_transfer(student, teacher, alignment)
        constrained = solve_flow_correction(fit, config=TrustRegionConfig())
        weights = import_weights(args.weights)
        plan = build_surgery_plan(weights, constrained, mapping=_load_tensor_mapping(args.mapping), gain=args.gain, mode="apply")
        updated = apply_surgery(weights, plan)
        export_weights(updated, args.output)
        journal = _journal(args, run_id="apply-surgery")
        if journal is not None:
            journal_surgery(journal, plan)
            journal.close()
        print(json.dumps({"output": str(args.output), "rollback_layers": list(plan.rollback_layers), "applied_tensors": list(plan.metadata["applied_tensors"]), "skipped_tensors": dict(plan.skipped_tensors), "quadratic_terms_applied": bool(plan.metadata.get("quadratic_terms_applied", False)), "status": "applied" if plan.applied_tensors else "rejected"}, indent=2, allow_nan=False))
        return 0
    if args.command == "validate":
        if args.teacher and args.baseline and args.intervened:
            alignment_error = 0.0 if not args.alignment else load_alignment(args.alignment).paired_error
            result = validate_causal_transfer(load_traces(args.teacher), load_traces(args.baseline), load_traces(args.intervened), alignment_error=alignment_error, teacher_to_student=None if not args.alignment else load_alignment(args.alignment))
            print(json.dumps({"geometric_alignment": result.geometric_alignment, "baseline_functional_error": result.baseline_functional_error, "intervened_functional_error": result.intervened_functional_error, "causal_effect": result.causal_effect, "stability_score": result.stability_score, "collapse_score": result.collapse_score, "passed": result.passed, "metrics": dict(result.metrics), "notes": list(result.notes)}, indent=2))
        elif args.trace:
            traces = load_traces(args.trace)
            result = make_report(traces, alignment=None if not args.alignment else load_alignment(args.alignment))
            print(report_json(result))
        else:
            raise SystemExit("validate requires --trace or --teacher --baseline --intervened")
        return 0
    if args.command == "report":
        traces = load_traces(args.trace)
        result = make_report(traces, alignment=None if not args.alignment else load_alignment(args.alignment))
        payload = report_json(result)
        if args.output:
            Path(args.output).write_text(payload, encoding="utf-8")
        print(payload)
        return 0
    if args.command == "export":
        output = export_weights(import_weights(args.weights), args.output)
        print(json.dumps({"output": str(output), "format": "ordinary-named-tensors"}, indent=2))
        return 0
    if args.command in {"benchmark-observation", "benchmark"}:
        payload = run_benchmark(BenchmarkConfig(state_dim=args.state_dim, layers=args.layers, probes=args.probes, jacobian_rank=args.jacobian_rank, hessian_rank=args.hessian_rank, seed=args.seed, memory_interval_seconds=args.memory_interval), output=args.output, progress_jsonl=args.progress_jsonl, progress_log=args.progress_log)
        print(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False))
        return 0 if payload.get("status") == "pass" else 2
    raise RuntimeError("unreachable command")


if __name__ == "__main__":
    raise SystemExit(main())
