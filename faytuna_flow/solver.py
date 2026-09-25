"""Constrained student-side flow correction solver."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from .types import FlowFitResult, finite_array, stable_l2


@dataclass(frozen=True)
class TrustRegionConfig:
    max_step_norm: float = 0.25
    max_relative_step: float = 0.20
    max_spectral_norm: float = 2.0
    max_lipschitz: float = 2.5
    min_confidence: float = 0.20
    min_singular_value: float = 1e-5
    collapse_variance_ratio: float = 0.05
    finite_margin: float = 1e6
    max_quadratic_norm: float = 1.0


@dataclass(frozen=True)
class ConstrainedCorrection:
    matrices: np.ndarray
    biases: np.ndarray
    accepted: np.ndarray
    reasons: Mapping[int, str]
    confidence: np.ndarray
    diagnostics: Mapping[str, np.ndarray]
    quadratic_terms: np.ndarray | None = None
    step_sizes: np.ndarray | None = None


def spectral_norm(matrix: np.ndarray) -> float:
    values = finite_array(matrix, ndim=2, name="operator")
    return float(np.linalg.svd(values, compute_uv=False)[0])


def project_spectral(matrix: np.ndarray, bound: float) -> np.ndarray:
    values = finite_array(matrix, ndim=2, name="operator")
    if bound <= 0 or not np.isfinite(bound):
        raise ValueError("spectral bound must be positive and finite")
    norm = spectral_norm(values)
    return values.copy() if norm <= bound else values * (bound / norm)


def _safe_scale(delta: np.ndarray, max_step_norm: float, max_relative_step: float, reference: np.ndarray) -> np.ndarray:
    delta = finite_array(delta, ndim=2, name="correction")
    norm = float(stable_l2(delta, name="correction Frobenius norm"))
    ref_norm = float(stable_l2(reference, name="reference Frobenius norm"))
    relative = norm / max(ref_norm, 1e-12)
    step_budget = max(max_step_norm, max_relative_step * ref_norm) if ref_norm > 1.0 else max_step_norm
    scale = min(1.0, step_budget / max(norm, 1e-12), max_relative_step / max(relative, 1e-12))
    return delta * max(0.0, scale)



def _collapse_ratio(operator: np.ndarray, baseline: np.ndarray) -> float:
    baseline_sv = np.linalg.svd(baseline, compute_uv=False)
    candidate_sv = np.linalg.svd(operator, compute_uv=False)
    base_scale = max(float(np.mean(baseline_sv)), 1e-12)
    return float(np.mean(candidate_sv) / base_scale)


def _safe_scale_tensor(delta: np.ndarray, bound: float) -> np.ndarray:
    norm = float(stable_l2(delta, name="quadratic correction norm"))
    if norm <= bound:
        return delta.copy()
    return delta * (bound / max(norm, 1e-12))


def solve_flow_correction(flow_fit: FlowFitResult, *, config: TrustRegionConfig | None = None, baseline_operators: np.ndarray | None = None) -> ConstrainedCorrection:
    """Project per-node deltas and rollback unsafe nodes independently."""

    cfg = config or TrustRegionConfig()
    deltas = finite_array(flow_fit.correction_matrices, ndim=3, name="flow correction matrices")
    biases = finite_array(flow_fit.correction_biases, ndim=2, name="flow correction biases")
    quadratic = None if flow_fit.correction_quadratic is None else finite_array(flow_fit.correction_quadratic, ndim=4, name="quadratic flow corrections")
    if baseline_operators is None:
        baseline = flow_fit.student.matrices
    else:
        baseline = finite_array(baseline_operators, ndim=3, name="baseline operators")
    if baseline.shape != deltas.shape:
        raise ValueError("baseline operators and flow corrections have different shapes")
    accepted = np.zeros(len(deltas), dtype=bool)
    corrected = np.zeros_like(deltas)
    corrected_bias = np.zeros_like(biases)
    corrected_quadratic = None if quadratic is None else np.zeros_like(quadratic)
    confidence = np.clip(flow_fit.confidence, 0.0, 1.0).copy()
    reasons: dict[int, str] = {}
    observed_norm = np.zeros(len(deltas))
    observed_lipschitz = np.zeros(len(deltas))
    collapse = np.zeros(len(deltas))
    quadratic_norm = np.zeros(len(deltas))
    quadratic_jacobian_bound = np.zeros(len(deltas))
    input_radius = flow_fit.student.metadata.get("input_radius", 1.0)
    if not np.isfinite(float(input_radius)) or float(input_radius) <= 0:
        raise ValueError("student flow input_radius must be positive and finite")
    input_radius = float(input_radius)
    baseline_spectra = [np.linalg.svd(base, compute_uv=False) for base in baseline]
    coords = flow_fit.student.coordinates
    step_sizes_arr = np.ones(len(deltas), dtype=np.float64)
    if len(coords) > 1:
        for idx in range(len(deltas)):
            step_sizes_arr[idx] = float(coords[idx + 1] - coords[idx]) if idx < len(coords) - 1 else float(coords[-1] - coords[-2])
    step_sizes_arr = np.maximum(step_sizes_arr, 1e-6)
    for i, (delta, bias, base, conf) in enumerate(zip(deltas, biases, baseline, confidence)):
        if conf < cfg.min_confidence:
            reasons[i] = "confidence below threshold"
            continue
        has_quadratic = (quadratic is not None) or (flow_fit.student.quadratic_terms is not None)
        if has_quadratic:
            q_delta = np.zeros((deltas.shape[1], deltas.shape[1], deltas.shape[1])) if quadratic is None else quadratic[i]
            if not np.all(np.isfinite(q_delta)):
                reasons[i] = "non-finite proposed correction"
                continue
            if cfg.max_quadratic_norm <= 0 or not np.isfinite(cfg.max_quadratic_norm):
                raise ValueError("max_quadratic_norm must be positive and finite")
            q_delta = _safe_scale_tensor(q_delta, cfg.max_quadratic_norm)
        else:
            q_delta = None
        if not (np.all(np.isfinite(delta)) and np.all(np.isfinite(bias))):
            reasons[i] = "non-finite proposed correction"
            continue
        candidate_delta = _safe_scale(delta, cfg.max_step_norm, cfg.max_relative_step, base)
        delta_spectrum = np.linalg.svd(candidate_delta, compute_uv=False)
        delta_norm = float(delta_spectrum[0]) if delta_spectrum.size else 0.0
        if cfg.max_spectral_norm <= 0 or not np.isfinite(cfg.max_spectral_norm):
            raise ValueError("spectral bound must be positive and finite")
        if delta_norm > cfg.max_spectral_norm:
            candidate_delta = candidate_delta * (cfg.max_spectral_norm / delta_norm)
        candidate = base + candidate_delta
        candidate_spectrum = np.linalg.svd(candidate, compute_uv=False)
        candidate_norm = float(candidate_spectrum[0]) if candidate_spectrum.size else 0.0
        if cfg.max_lipschitz <= 0 or not np.isfinite(cfg.max_lipschitz):
            raise ValueError("spectral bound must be positive and finite")
        base_norm = float(baseline_spectra[i][0]) if baseline_spectra[i].size else 0.0
        if base_norm <= cfg.max_lipschitz:
            target_lipschitz = cfg.max_lipschitz
        else:
            target_lipschitz = base_norm + min(cfg.max_spectral_norm, max(1.0, base_norm) * cfg.max_relative_step)
        if candidate_norm > target_lipschitz:
            excess = candidate_norm - base_norm
            if excess > 1e-12 and target_lipschitz > base_norm:
                alpha = max(0.0, min(1.0, (target_lipschitz - base_norm) / excess))
                candidate_delta = candidate_delta * alpha
            else:
                candidate_delta = np.zeros_like(candidate_delta)
            candidate = base + candidate_delta
            candidate_spectrum = np.linalg.svd(candidate, compute_uv=False)
        base_frob = float(stable_l2(base, name="base Frobenius norm"))
        max_step_bound = max(cfg.max_step_norm, cfg.max_relative_step * base_frob) if base_frob > 1.0 else cfg.max_step_norm
        candidate_step_norm = float(stable_l2(candidate_delta, name="candidate Frobenius norm"))
        candidate_rel_step = candidate_step_norm / max(base_frob, 1e-12)
        observed_norm[i] = float(candidate_spectrum[0]) if candidate_spectrum.size else 0.0
        if has_quadratic:
            current_quadratic = np.zeros_like(q_delta) if flow_fit.student.quadratic_terms is None else flow_fit.student.quadratic_terms[i]
            candidate_quadratic = current_quadratic + q_delta
            available_quadratic_lipschitz = max(0.0, target_lipschitz - observed_norm[i])
            current_quadratic_bound = 2.0 * input_radius * float(stable_l2(current_quadratic, name="baseline quadratic norm"))
            proposed_quadratic_bound = 2.0 * input_radius * float(stable_l2(candidate_quadratic, name="candidate quadratic norm"))
            if current_quadratic_bound > available_quadratic_lipschitz + 1e-12:
                q_delta = np.zeros_like(q_delta)
                candidate_quadratic = current_quadratic.copy()
                quadratic_jacobian_bound[i] = current_quadratic_bound
                reasons[i] = "baseline quadratic Jacobian exceeds Lipschitz budget"
                continue
            if proposed_quadratic_bound > available_quadratic_lipschitz:
                remaining = max(0.0, available_quadratic_lipschitz - current_quadratic_bound)
                q_delta = q_delta * (remaining / max(proposed_quadratic_bound - current_quadratic_bound, 1e-12))
                candidate_quadratic = current_quadratic + q_delta
            quadratic_norm[i] = float(stable_l2(q_delta, name="accepted quadratic correction"))
            quadratic_jacobian_bound[i] = 2.0 * input_radius * float(stable_l2(candidate_quadratic, name="bounded quadratic norm"))
            observed_lipschitz[i] = observed_norm[i] + quadratic_jacobian_bound[i]
        else:
            quadratic_norm[i] = 0.0
            quadratic_jacobian_bound[i] = 0.0
            observed_lipschitz[i] = observed_norm[i]
        base_spectrum = baseline_spectra[i]
        collapse[i] = float(np.mean(candidate_spectrum) / max(float(np.mean(base_spectrum)), 1e-12))
        ds = float(step_sizes_arr[i])
        layer_operator = np.eye(candidate.shape[0]) + ds * candidate
        layer_sv = np.linalg.svd(layer_operator, compute_uv=False)
        if not np.all(np.isfinite(candidate)) or np.max(np.abs(candidate)) > cfg.finite_margin:
            reasons[i] = "finite safety check failed"
        elif candidate_step_norm > max_step_bound + 1e-12 or candidate_rel_step > cfg.max_relative_step + 1e-12:
            reasons[i] = "step norm bound failed"
        elif observed_lipschitz[i] > target_lipschitz + 1e-10:
            reasons[i] = "Lipschitz bound failed"
        elif layer_sv[-1] < cfg.min_singular_value:
            reasons[i] = "minimum singular value bound failed"
        elif collapse[i] < cfg.collapse_variance_ratio:
            reasons[i] = "collapse protection triggered"
        else:
            accepted[i] = True
            corrected[i] = candidate_delta
            corrected_bias[i] = np.clip(bias, -cfg.max_step_norm, cfg.max_step_norm)
            if corrected_quadratic is not None and q_delta is not None:
                corrected_quadratic[i] = q_delta
    diagnostics = {"spectral_norm": observed_norm, "lipschitz": observed_lipschitz, "quadratic_jacobian_bound": quadratic_jacobian_bound, "collapse_ratio": collapse, "quadratic_norm": quadratic_norm}
    return ConstrainedCorrection(corrected, corrected_bias, accepted, reasons, confidence, diagnostics, corrected_quadratic, step_sizes=step_sizes_arr)
