import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from deltaconv_audit.operators import build_gradient, neighbors, tangent_basis
from deltaconv_audit.surfaces import sample_surface, scalar_field
from deltaconv_audit.experiment import run_audit


def test_exact_quadratic_derivative_on_plane_without_regularization():
    rng = np.random.default_rng(8)
    xy = rng.uniform(-1, 1, (100, 2))
    points = np.column_stack([xy, np.zeros(100)])
    normals = np.tile([0., 0., 1.], (100, 1))
    x, y = xy.T
    signal = 2+3*x-2*y+.3*x*x+.8*x*y-.2*y*y
    expected = np.column_stack([3+.6*x+.8*y, -2+.8*x-.4*y, np.zeros(100)])
    actual = build_gradient(points, normals, regularizer=0).apply(signal)
    np.testing.assert_allclose(actual, expected, atol=1e-9)


def test_invariant_penalty_and_bound_preserve_independent_tangent_rotations():
    points, normals, _ = sample_surface("sphere", 120, 1, "biased")
    signal, _ = scalar_field(points, normals)
    angles = np.random.default_rng(9).uniform(0, 2*np.pi, len(points))
    kwargs = dict(penalty="tensor", normalization="invariant", regularizer=1e-3)
    a = build_gradient(points, normals, **kwargs)
    b = build_gradient(points, normals, angles=angles, **kwargs)
    np.testing.assert_allclose(a.apply(signal), b.apply(signal), atol=1e-11)
    np.testing.assert_allclose(a.apply(signal, normalized=True), b.apply(signal, normalized=True), atol=1e-11)


def test_baseline_ridge_exposes_measurable_basis_dependence():
    points, normals, _ = sample_surface("saddle", 120, 1, "biased")
    signal, _ = scalar_field(points, normals)
    angles = np.random.default_rng(9).uniform(0, 2*np.pi, len(points))
    a = build_gradient(points, normals)
    b = build_gradient(points, normals, angles=angles)
    assert np.linalg.norm(a.apply(signal)-b.apply(signal)) > 1e-5


def test_rigid_rotation_of_geometry_preserves_ambient_gradient_equivariance():
    points, normals, _ = sample_surface("sphere", 120, 1)
    signal, _ = scalar_field(points, normals)
    q, _ = np.linalg.qr(np.random.default_rng(6).normal(size=(3, 3)))
    kwargs = dict(penalty="tensor", normalization="invariant")
    a = build_gradient(points, normals, **kwargs).apply(signal)
    b = build_gradient(points@q, normals@q, **kwargs).apply(signal)
    np.testing.assert_allclose(a@q, b, atol=1e-10)


def test_permutation_and_constant_field():
    points, normals, _ = sample_surface("sphere", 80, 1)
    p = np.random.default_rng(12).permutation(len(points))
    signal, _ = scalar_field(points, normals)
    kwargs = dict(regularizer=0, penalty="tensor", normalization="invariant")
    a = build_gradient(points, normals, **kwargs)
    b = build_gradient(points[p], normals[p], **kwargs)
    np.testing.assert_allclose(a.apply(signal)[p], b.apply(signal[p]), atol=1e-9)
    np.testing.assert_allclose(a.apply(np.ones(len(points))), 0, atol=1e-9)


def test_invalid_geometry_does_not_create_a_successful_gradient():
    with pytest.raises(ValueError, match="finite"):
        neighbors([[float("nan"), 0, 0]], 6)
    with pytest.raises(ValueError, match="nonzero"):
        tangent_basis(np.zeros((10, 3)))
    with pytest.raises(ValueError, match="collapsed"):
        build_gradient(np.zeros((20, 3)), np.tile([0, 0, 1.], (20, 1)))


def test_experiment_outputs_measured_evidence_and_refuses_overwrite(tmp_path):
    report = run_audit(tmp_path, n=40, seeds=(0,), noise=(0.,))
    assert report["status"] == "ok"
    assert report["evidence_kind"] == "measured_cpu_numpy_operator_audit"
    assert len(report["rows"]) == 2*2*2*4
    assert len(report["source_sha256"]) >= 3
    assert (tmp_path / "viewer_data.json").is_file()
    with pytest.raises(FileExistsError):
        run_audit(tmp_path, n=40, seeds=(0,), noise=(0.,))
