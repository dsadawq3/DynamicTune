"""Activation audit for the numerical components of a flow experiment.

The profile answers whether a requested component was actually executed.  A
component can be mathematically enabled in a mode yet unavailable for a
connector or data shape; that state is explicit and carries a reason.  The
profile is evidence about execution, not evidence of semantic understanding.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np


MATH_COMPONENTS: tuple[str, ...] = (
    "alignment",
    "ot",
    "continuous_depth",
    "path_signature",
    "local_1jet",
    "hessian_2jet",
    "curvature",
    "robust_loss",
    "tangent_projector_transport",
    "capacity_projection",
    "quadratic_flow",
    "stability_barriers",
    "gain_schedule",
)


def _finite(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float(default)
    return number if np.isfinite(number) else float(default)


def _entry(enabled: bool, applied: bool, contribution: float | None, metric_delta: float | None, skipped_reason: str | None, *, metric_delta_scope: str = "unavailable_without_leave_one_component_out") -> dict[str, Any]:
    if not enabled:
        applied = False
        contribution = None
        metric_delta = None
        skipped_reason = skipped_reason or "explicitly disabled"
    elif not applied and not skipped_reason:
        skipped_reason = "component did not execute"
    if enabled and not applied and not skipped_reason:
        raise ValueError("an enabled but unapplied math component needs skipped_reason")
    return {
        "enabled": bool(enabled),
        "applied": bool(applied),
        "contribution": None if contribution is None else _finite(contribution),
        "metric_delta": None if metric_delta is None else _finite(metric_delta),
        "metric_delta_scope": metric_delta_scope,
        "skipped_reason": skipped_reason,
    }


def resolve_math_components(requested: Mapping[str, bool] | None = None) -> dict[str, bool]:
    """Validate explicit component switches and fill deterministic defaults."""

    values = {component: True for component in MATH_COMPONENTS}
    if requested is None:
        return values
    unknown = sorted(set(requested) - set(MATH_COMPONENTS))
    if unknown:
        raise ValueError(f"unknown math component flag(s): {unknown}; available: {list(MATH_COMPONENTS)}")
    for component, enabled in requested.items():
        if not isinstance(enabled, (bool, np.bool_)):
            raise TypeError(f"math component flag {component!r} must be boolean")
        values[component] = bool(enabled)
    # These are protocol requirements, so silently disabling them would be a
    # no-op flag.  The caller must choose a different observation contract.
    for component in ("alignment", "continuous_depth"):
        if not values[component]:
            raise ValueError(f"math component {component!r} cannot be disabled in flow transfer; refusing a no-op flag")
    return values


def build_math_profile(
    metadata: Mapping[str, Any],
    *,
    alignment_kind: str,
    signature_mode: str,
    correction_matrices: np.ndarray | None = None,
    correction_quadratic: np.ndarray | None = None,
    requested: Mapping[str, bool] | None = None,
    policy: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a complete per-component execution profile.

    ``metric_delta`` remains unavailable until a same-split
    leave-one-component-out refit supplies it. Local magnitudes belong in
    ``contribution`` and are not causal attributions.
    """

    switches = resolve_math_components(requested)
    mode = str(signature_mode)
    jet_applied = bool(metadata.get("rank_aware_local_jet_transport", False)) and mode in {"differential", "full"}
    path_applied = bool(metadata.get("multi_scale_path_signatures", False)) and mode in {"differential", "full"}
    tangent_applied = bool(metadata.get("tangent_transport", False)) and mode == "full"
    q_applied = correction_quadratic is not None and bool(metadata.get("state_feature_degree", 1) == 2)
    q_skipped_reason = str(metadata.get("quadratic_skipped_reason", "ordinary affine flow channel only"))
    capacity_applied = bool(metadata.get("student_manifold_projection", False))
    stability_applied = bool(metadata.get("deterministic_stability_seeds"))
    profiles: dict[str, dict[str, Any]] = {}
    profiles["alignment"] = _entry(switches["alignment"], True, 1.0 / (1.0 + max(0.0, _finite(metadata.get("alignment_relational_error")))), None, None)
    profiles["ot"] = _entry(
        switches["ot"],
        alignment_kind == "ot_barycentric",
        _finite(metadata.get("ot_mass_error"), 0.0),
        None,
        None if alignment_kind == "ot_barycentric" else "alignment kind is not ot_barycentric",
    )
    profiles["continuous_depth"] = _entry(switches["continuous_depth"], True, 1.0, None, None)
    profiles["path_signature"] = _entry(
        switches["path_signature"],
        path_applied,
        _finite(np.mean([item.get("path_score", 0.0) for item in metadata.get("signature_scores", ())])) if metadata.get("signature_scores") else None,
        None,
        None if path_applied else f"signature_mode={mode} does not execute path signature weighting",
    )
    profiles["local_1jet"] = _entry(
        switches["local_1jet"],
        jet_applied,
        _finite(np.mean(metadata.get("jet_confidence", [0.0]))) if jet_applied else None,
        None,
        None if jet_applied else "rank-aware local jet unavailable or mode excludes differential transport",
    )
    profiles["hessian_2jet"] = _entry(
        switches["hessian_2jet"],
        q_applied,
        _finite(np.linalg.norm(correction_quadratic)) if q_applied else None,
        None,
        None if q_applied else ("signature_mode does not request quadratic channel" if mode != "full" else q_skipped_reason),
    )
    profiles["curvature"] = _entry(switches["curvature"], True, _finite(np.mean([item.get("correspondence_metadata", {}).get("curvature_weight", 0.0) for item in metadata.get("signature_scores", ())])) if metadata.get("signature_scores") else 0.0, None, None)
    robust = str(metadata.get("robust_loss", "none"))
    profiles["robust_loss"] = _entry(switches["robust_loss"], robust in {"huber", "tukey"}, 1.0 if robust in {"huber", "tukey"} else 0.0, None, None if robust in {"huber", "tukey"} else "explicit robust loss disabled")
    profiles["tangent_projector_transport"] = _entry(
        switches["tangent_projector_transport"],
        tangent_applied,
        _finite(np.mean(metadata.get("local_tangent_gate", [0.0]))) if tangent_applied else None,
        None,
        None if tangent_applied else "tangent transport unavailable or signature_mode is not full",
    )
    profiles["capacity_projection"] = _entry(switches["capacity_projection"], capacity_applied, _finite(metadata.get("capacity_gate", 0.0)), None, None if capacity_applied else "student manifold projection not applied")
    profiles["quadratic_flow"] = _entry(switches["quadratic_flow"], q_applied, _finite(np.linalg.norm(correction_quadratic)) if q_applied else None, None, None if q_applied else q_skipped_reason)
    profiles["stability_barriers"] = _entry(switches["stability_barriers"], stability_applied, _finite(np.exp(-np.mean(metadata.get("ensemble_spread", [0.0])))) if stability_applied else None, None, None if stability_applied else "stability ensemble was not run")
    policy_applied = policy is not None and str(policy.get("mode", "")) == "adaptive_transfer"
    accepted = _finite(policy.get("final_alpha", 0.0)) if policy is not None else 0.0
    profiles["gain_schedule"] = _entry(switches["gain_schedule"], policy_applied, accepted, None, None if policy_applied else "only applied by the adaptive transfer policy")
    return {
        "schema_version": "faytuna-math-profile-v1",
        "signature_mode": mode,
        "alignment_kind": alignment_kind,
        "components": profiles,
        "no_op_flags_rejected": True,
    }


def compare_math_profiles(reference: Mapping[str, Any], candidate: Mapping[str, Any], *, reference_metric: float | None = None, candidate_metric: float | None = None) -> dict[str, Any]:
    """Compare profiles without attributing an aggregate delta to components.

    A component-level causal contribution requires a leave-one-component-out
    refit. This helper therefore leaves every component ``metric_delta``
    unavailable and labels the optional top-level delta as aggregate only.
    """

    result = {"schema_version": "faytuna-math-contribution-v1", "reference_metric": reference_metric, "candidate_metric": candidate_metric, "metric_delta": None if reference_metric is None or candidate_metric is None else float(candidate_metric - reference_metric), "metric_delta_scope": "aggregate_candidate_minus_reference_not_component_causal", "component_deltas_available": False, "components": {}}
    ref_components = dict(reference.get("components", {}))
    cand_components = dict(candidate.get("components", {}))
    for component in MATH_COMPONENTS:
        ref = dict(ref_components.get(component, {}))
        cand = dict(cand_components.get(component, {}))
        result["components"][component] = {
            "enabled": bool(cand.get("enabled", False)),
            "applied": bool(cand.get("applied", False)),
            "contribution": cand.get("contribution"),
            "contribution_delta": None,
            "metric_delta": None,
            "metric_delta_scope": "unavailable_without_leave_one_component_out",
            "causal_contribution": None,
            "skipped_reason": cand.get("skipped_reason"),
        }
    return result
