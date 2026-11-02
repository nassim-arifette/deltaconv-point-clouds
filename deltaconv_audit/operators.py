"""Small NumPy audit of the quadratic MLS gradient in official DeltaConv.

Reference: rubenwiersma/deltaconv, commit 186fec369fa2ceb4559830bc421282dddb2300a2,
deltaconv/geometry/grad_div_mls.py. The baseline follows its polynomial, Gaussian
weights, ridge and gradient normalization. This is not the full DeltaConv model:
there is no divergence, vector stream, neural layer, training or classification.

Two explicit variations are tested: a rotation-invariant quadratic coefficient
penalty and a rotation-invariant bound for normalizing tangent vectors.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _points(points):
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError("points must be a finite (N, 3) array")
    return points


def neighbors(points, k=20):
    """Deterministic k-nearest neighbors including the center, as in upstream.

    Dense O(N²) search keeps the audit independent of compiled graph libraries;
    use this only for small numerical experiments, not large LiDAR scenes.
    """
    points = _points(points)
    if not 6 <= k <= len(points):
        raise ValueError("k must be between 6 and the number of points")
    delta = points[:, None] - points[None, :]
    d2 = np.einsum("ijk,ijk->ij", delta, delta)
    return np.argsort(d2, axis=1, kind="stable")[:, :k]


def tangent_basis(normals, angles=None):
    normal = _points(normals)
    lengths = np.linalg.norm(normal, axis=1, keepdims=True)
    if (lengths < 1e-12).any():
        raise ValueError("normal vectors must be nonzero")
    normal = normal / lengths
    axis = np.zeros_like(normal)
    axis[:, 0] = 1
    axis[np.abs(normal[:, 0]) > .9] = [0, 1, 0]
    x = np.cross(axis, normal)
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    y = np.cross(normal, x)
    if angles is not None:
        angles = np.asarray(angles, dtype=np.float64)
        if angles.shape != (len(normal),) or not np.isfinite(angles).all():
            raise ValueError("one finite tangent rotation is required per point")
        c, s = np.cos(angles)[:, None], np.sin(angles)[:, None]
        x, y = c * x + s * y, -s * x + c * y
    return np.stack([x, y], axis=2)


def estimate_normals(points, index):
    """Local PCA, with arbitrary signs; gradient reconstruction does not need oriented normals."""
    points = _points(points)
    local = points[index] - points[:, None]
    covariance = np.einsum("nki,nkj->nij", local, local)
    _, vectors = np.linalg.eigh(covariance)
    return vectors[:, :, 0]


@dataclass
class Gradient:
    index: np.ndarray
    coefficients: np.ndarray  # (N, k, 2), BEFORE network normalization
    basis: np.ndarray  # (N, 3, 2)
    scale: float
    condition: np.ndarray

    def apply(self, signal, *, normalized=False):
        signal = np.asarray(signal, dtype=np.float64)
        if signal.shape != (len(self.index),) or not np.isfinite(signal).all():
            raise ValueError("signal must be a finite scalar per point")
        local = np.einsum("nkj,nk->nj", self.coefficients, signal[self.index])
        ambient = np.einsum("nij,nj->ni", self.basis, local)
        return ambient / self.scale if normalized else ambient


def build_gradient(points, normals, index=None, *, k=20, regularizer=1e-3,
                   penalty="upstream", normalization="upstream", bandwidth="global", angles=None):
    """Build the baseline or an explicit ablation on exactly the same neighbors.

    In polynomial coordinates [1,u,v,u²,uv,v²], diag(1,1,1,1,.5,1)
    penalizes the Frobenius norm of the symmetric quadratic tensor. That penalty
    is invariant under an orthonormal change of tangent basis; the upstream I is
    not. The invariant normalizer is max_i sum_j ||g_ij||₂.
    """
    points = _points(points)
    if not np.isfinite(regularizer) or regularizer < 0:
        raise ValueError("regularizer must be finite and nonnegative")
    if penalty not in {"upstream", "tensor"} or normalization not in {"upstream", "invariant", "none"}:
        raise ValueError("unknown penalty or normalization")
    if bandwidth not in {"global", "local"}:
        raise ValueError("bandwidth must be global or local")
    index = neighbors(points, k) if index is None else np.asarray(index)
    if index.shape != (len(points), k) or not np.issubdtype(index.dtype, np.integer):
        raise ValueError("neighbor indices must have shape (N, k)")
    if (index < 0).any() or (index >= len(points)).any():
        raise ValueError("neighbor index out of range")
    basis = tangent_basis(normals, angles)
    if len(basis) != len(points):
        raise ValueError("one normal per point is required")
    delta = points[index] - points[:, None]
    distance = np.linalg.norm(delta, axis=2)
    width = distance.mean() if bandwidth == "global" else distance.mean(axis=1, keepdims=True)
    if np.any(np.asarray(width) < 1e-12):
        raise ValueError("collapsed point neighborhood")
    weights = np.exp(-distance ** 2 / width ** 2)
    weights /= np.maximum(weights.sum(axis=1, keepdims=True), 1e-5)
    uv = np.einsum("nkj,nji->nki", delta, basis)
    u, v = uv[:, :, 0], uv[:, :, 1]
    B = np.stack([np.ones_like(u), u, v, u*u, u*v, v*v], axis=2)
    BT = (weights[:, :, None] * B).transpose(0, 2, 1)
    diagonal = [1, 1, 1, 1, 1, 1] if penalty == "upstream" else [1, 1, 1, 1, .5, 1]
    gram = BT @ B + regularizer * np.diag(diagonal)[None]
    coefficients = np.linalg.solve(gram, BT)[:, 1:3, :].transpose(0, 2, 1)
    if normalization == "upstream":
        scale = np.linalg.norm(np.abs(coefficients).sum(axis=1), axis=1).max()
    elif normalization == "invariant":
        scale = np.linalg.norm(coefficients, axis=2).sum(axis=1).max()
    else:
        scale = 1.
    scale = float(scale) if scale > 1e-5 else 1.
    return Gradient(index, coefficients, basis, scale, np.linalg.cond(gram))

