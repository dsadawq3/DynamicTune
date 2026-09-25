"""Multi-chart manifold tangent bundle projection and Grassmannian geodesic transport.

Partitions the activation space into local geometric charts with smooth
partition-of-unity blending, overcoming the rigid single-subspace bottleneck.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence
import numpy as np

from .types import finite_array


def spherical_kmeans(
    data: np.ndarray,
    n_clusters: int,
    *,
    max_iter: int = 50,
    tol: float = 1e-4,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Spherical k-means clustering using cosine similarity on the unit sphere.

    Args:
        data: (N, D) input representations.
        n_clusters: Number of geometric charts M.
        max_iter: Maximum iterations.
        tol: Convergence tolerance for centroid movement.
        random_state: Seed for reproducibility.

    Returns:
        centroids: (M, D) normalized cluster centroids on S^{D-1}.
        labels: (N,) cluster assignment indices.
    """
    x = finite_array(data, ndim=2, name="spherical_kmeans_data")
    n, d = x.shape
    if n == 0:
        raise ValueError("cannot cluster empty data")
    k = min(n_clusters, n)
    if k <= 1:
        c = np.mean(x, axis=0, keepdims=True)
        c_norm = np.linalg.norm(c, axis=1, keepdims=True)
        c = c / np.maximum(c_norm, 1e-12)
        return c, np.zeros(n, dtype=np.int64)

    # Normalize data to unit sphere
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    x_norm = x / np.maximum(norms, 1e-12)

    rng = np.random.RandomState(random_state)
    # k-means++ initialization on sphere
    centroids = np.zeros((k, d), dtype=np.float64)
    first_idx = rng.randint(0, n)
    centroids[0] = x_norm[first_idx]

    dists = 1.0 - np.clip(x_norm @ centroids[0], -1.0, 1.0)
    for c_idx in range(1, k):
        probs = np.maximum(dists, 0.0)
        total = np.sum(probs)
        if total > 0:
            probs = probs / total
            chosen = rng.choice(n, p=probs)
        else:
            chosen = rng.randint(0, n)
        centroids[c_idx] = x_norm[chosen]
        new_dists = 1.0 - np.clip(x_norm @ centroids[c_idx], -1.0, 1.0)
        dists = np.minimum(dists, new_dists)

    labels = np.zeros(n, dtype=np.int64)
    for _ in range(max_iter):
        # Assign to nearest centroid by cosine similarity (maximum dot product)
        similarities = x_norm @ centroids.T
        new_labels = np.argmax(similarities, axis=1)

        # Update centroids
        shift = 0.0
        new_centroids = np.zeros_like(centroids)
        for c_idx in range(k):
            members = x_norm[new_labels == c_idx]
            if len(members) > 0:
                mean_vec = np.sum(members, axis=0)
                norm = np.linalg.norm(mean_vec)
                new_centroids[c_idx] = mean_vec / max(norm, 1e-12)
            else:
                # Re-seed empty cluster from random point
                new_centroids[c_idx] = x_norm[rng.randint(0, n)]
            shift += float(np.linalg.norm(new_centroids[c_idx] - centroids[c_idx]))

        centroids = new_centroids
        labels = new_labels
        if shift < tol:
            break

    return centroids, labels


def _safe_svd(matrix: np.ndarray, full_matrices: bool = False):
    """Compute SVD with robust fallback to scipy gesvd and regularized matrix if LAPACK dgesdd fails."""
    try:
        return np.linalg.svd(matrix, full_matrices=full_matrices)
    except (np.linalg.LinAlgError, ValueError):
        import scipy.linalg
        try:
            return scipy.linalg.svd(matrix, full_matrices=full_matrices, lapack_driver="gesvd")
        except Exception:
            m_shape = matrix.shape
            reg = 1e-6 * (float(np.linalg.norm(matrix)) + 1e-8)
            eye = np.eye(m_shape[0], m_shape[1])
            m_reg = matrix + reg * eye
            try:
                return scipy.linalg.svd(m_reg, full_matrices=full_matrices, lapack_driver="gesvd")
            except Exception:
                return np.linalg.svd(m_reg, full_matrices=full_matrices)


def solve_orthogonal_procrustes(
    student_block: np.ndarray,
    teacher_block: np.ndarray,
) -> np.ndarray:
    """Solve orthogonal Procrustes map P in R^{d_T x d_S}.

    Minimizes || Y - X P^T ||_F subject to P^T P = I (or P P^T = I if d_T < d_S).
    Let M = Y^T X in R^{d_T x d_S}.
    SVD: M = U Sigma V^T => P = U V^T (padded/sliced if dimensions differ).
    """
    x = finite_array(student_block, ndim=2, name="student_block")
    y = finite_array(teacher_block, ndim=2, name="teacher_block")
    if len(x) != len(y):
        raise ValueError(f"length mismatch: student {len(x)} vs teacher {len(y)}")

    d_s = x.shape[1]
    d_t = y.shape[1]

    # Cross-covariance matrix M = Y^T @ X (d_T, d_S)
    m = y.T @ x
    u, _, vt = _safe_svd(m, full_matrices=False)

    # P = U @ V^T has shape (d_T, d_S)
    p = u @ vt
    return np.asarray(p, dtype=np.float64)


@dataclass(frozen=True)
class MultiChartAtlas:
    """Atlas of local tangent bundle charts with smooth partition-of-unity blending."""

    n_charts: int
    student_dim: int
    teacher_dim: int
    centroids: np.ndarray  # (M, d_S)
    projectors: tuple[np.ndarray, ...]  # M matrices of shape (d_T, d_S)
    temperature: float = 4.0

    def compute_weights(self, student_x: np.ndarray) -> np.ndarray:
        """Compute smooth partition-of-unity weights w_m(x) in [0, 1].

        Uses softmax over cosine distances:
            w_m(x) = exp(beta * cos(x, c_m)) / sum_j exp(beta * cos(x, c_j))
        """
        x = finite_array(student_x, ndim=2, name="student_x")
        norms = np.linalg.norm(x, axis=1, keepdims=True)
        x_norm = x / np.maximum(norms, 1e-12)

        # Dot product with normalized centroids: (N, M)
        sims = x_norm @ self.centroids.T
        scaled = sims * self.temperature

        # Stable softmax
        max_s = np.max(scaled, axis=1, keepdims=True)
        exp_s = np.exp(scaled - max_s)
        weights = exp_s / np.maximum(np.sum(exp_s, axis=1, keepdims=True), 1e-12)
        return weights

    def lift_student_to_teacher(self, student_x: np.ndarray) -> np.ndarray:
        """Smoothly lift student tokens into teacher space using blended charts.

        For each token i:
            y_i = sum_{m=1}^M w_m(x_i) * (x_i @ (P^{(m)})^T)
        """
        x = finite_array(student_x, ndim=2, name="student_x")
        weights = self.compute_weights(x)  # (N, M)

        n = len(x)
        lifted = np.zeros((n, self.teacher_dim), dtype=np.float64)

        for m_idx, proj in enumerate(self.projectors):
            # x @ proj.T has shape (N, d_T)
            proj_m = x @ proj.T
            w_m = weights[:, m_idx : m_idx + 1]
            lifted += w_m * proj_m

        return lifted

    def project_teacher_to_student(self, teacher_y: np.ndarray, reference_student_x: np.ndarray) -> np.ndarray:
        """Project teacher tokens back into student space using reference coordinates."""
        y = finite_array(teacher_y, ndim=2, name="teacher_y")
        weights = self.compute_weights(reference_student_x)  # (N, M)

        n = len(y)
        projected = np.zeros((n, self.student_dim), dtype=np.float64)

        for m_idx, proj in enumerate(self.projectors):
            # y @ proj has shape (N, d_S)
            proj_m = y @ proj
            w_m = weights[:, m_idx : m_idx + 1]
            projected += w_m * proj_m

        return projected


def build_multi_chart_atlas(
    student_representations: np.ndarray,
    teacher_representations: np.ndarray,
    *,
    n_charts: int = 4,
    temperature: float = 4.0,
    random_state: int = 42,
) -> MultiChartAtlas:
    """Construct a multi-chart atlas from paired empirical representations.

    Args:
        student_representations: (N, d_S) activations from student model.
        teacher_representations: (N, d_T) activations from teacher model.
        n_charts: Number of local tangent space charts.
        temperature: Softmax sharpness for partition-of-unity blending.
        random_state: Random seed for clustering.
    """
    x = finite_array(student_representations, ndim=2, name="student_representations")
    y = finite_array(teacher_representations, ndim=2, name="teacher_representations")

    if len(x) != len(y):
        raise ValueError(f"sample mismatch: student {len(x)} vs teacher {len(y)}")

    n, d_s = x.shape
    _, d_t = y.shape

    centroids, labels = spherical_kmeans(x, n_charts, random_state=random_state)
    actual_charts = len(centroids)

    projectors: list[np.ndarray] = []
    # Global fallback projector in case a cluster has very few points
    global_p = solve_orthogonal_procrustes(x, y)

    for c_idx in range(actual_charts):
        mask = (labels == c_idx)
        if np.sum(mask) >= max(d_s // 4, 10):
            try:
                p_m = solve_orthogonal_procrustes(x[mask], y[mask])
            except Exception:
                p_m = global_p.copy()
        else:
            p_m = global_p.copy()
        projectors.append(p_m)

    return MultiChartAtlas(
        n_charts=actual_charts,
        student_dim=d_s,
        teacher_dim=d_t,
        centroids=centroids,
        projectors=tuple(projectors),
        temperature=temperature,
    )


@dataclass(frozen=True)
class GrassmannianGeodesic:
    """Geodesic curve in the Grassmannian manifold Gr(d, D) between two subspaces."""

    principal_angles: np.ndarray  # Theta in [0, pi/2]
    u1: np.ndarray  # Orthonormal basis 1
    u2: np.ndarray  # Orthonormal basis 2
    v1: np.ndarray  # Left singular vectors
    v2: np.ndarray  # Right singular vectors
    subspace_dim: int
    ambient_dim: int

    def interpolate(self, t: float) -> np.ndarray:
        """Interpolate basis at geodesic parameter t in [0, 1].

        gamma(t) = U_1 V_1 cos(t Theta) + U_perp sin(t Theta)
        """
        t_clamped = float(np.clip(t, 0.0, 1.0))
        cos_t = np.cos(t_clamped * self.principal_angles)
        sin_t = np.sin(t_clamped * self.principal_angles)

        # Base term
        base = (self.u1 @ self.v1) * cos_t[None, :]

        # Orthogonal tangent direction
        # Projection of U_2 onto orthogonal complement of U_1:
        # U_2 V_2 - U_1 (U_1^T U_2 V_2) = U_2 V_2 - U_1 V_1 cos(Theta)
        u2_v2 = self.u2 @ self.v2
        u1_v1_cos = (self.u1 @ self.v1) * np.cos(self.principal_angles)[None, :]
        perp = u2_v2 - u1_v1_cos

        # Normalize orthogonal directions
        perp_norms = np.linalg.norm(perp, axis=0, keepdims=True)
        perp_dir = perp / np.maximum(perp_norms, 1e-12)

        interpolated = base + perp_dir * sin_t[None, :]
        # Re-orthonormalize via QR
        q, _ = np.linalg.qr(interpolated)
        return q[:, : self.subspace_dim]


def compute_grassmannian_geodesic(
    subspace_1: np.ndarray,
    subspace_2: np.ndarray,
) -> GrassmannianGeodesic:
    """Compute principal angles and geodesic curve between two subspaces of dimension d in R^D.

    Args:
        subspace_1: (D, d) orthonormal frame.
        subspace_2: (D, d) orthonormal frame.
    """
    q1 = finite_array(subspace_1, ndim=2, name="subspace_1")
    q2 = finite_array(subspace_2, ndim=2, name="subspace_2")

    if q1.shape != q2.shape:
        raise ValueError(f"subspace shapes must match: {q1.shape} vs {q2.shape}")

    d_ambient, d_sub = q1.shape
    # Ensure orthonormal frames
    u1, _ = np.linalg.qr(q1)
    u1 = u1[:, :d_sub]
    u2, _ = np.linalg.qr(q2)
    u2 = u2[:, :d_sub]

    # SVD of cross-Gram matrix
    cross = u1.T @ u2
    v1, singular, v2_t = _safe_svd(cross)
    v2 = v2_t.T

    # Principal angles theta_i = arccos(sigma_i)
    clipped_s = np.clip(singular, -1.0, 1.0)
    theta = np.arccos(clipped_s)

    return GrassmannianGeodesic(
        principal_angles=theta,
        u1=u1,
        u2=u2,
        v1=v1,
        v2=v2,
        subspace_dim=d_sub,
        ambient_dim=d_ambient,
    )
