"""Analytic surfaces and exact tangent gradients, with declared sampling distributions."""
import numpy as np


def sample_surface(shape, n, seed, density="uniform"):
    if n < 20 or density not in {"uniform", "biased"}:
        raise ValueError("need n >= 20 and density uniform/biased")
    rng = np.random.default_rng(seed)
    if shape == "sphere":
        # Uniform z and azimuth give uniform area on the unit sphere.
        z = rng.uniform(-1, 1, n)
        if density == "biased":
            # 90% north vs 10% south, same geometric support; fixed total N.
            z = np.abs(z) * np.where(rng.random(n) < .9, 1, -1)
        phi = rng.uniform(0, 2*np.pi, n)
        radius = np.sqrt(1-z*z)
        points = np.column_stack([radius*np.cos(phi), radius*np.sin(phi), z])
        normals = points.copy()
        sparse_region = z < 0
    elif shape == "saddle":
        xy = rng.uniform(-1, 1, (n, 2))
        if density == "biased":
            xy[:, 0] = np.abs(xy[:, 0]) * np.where(rng.random(n) < .9, 1, -1)
        x, y = xy.T
        points = np.column_stack([x, y, .3*(x*x-y*y)])
        normals = np.column_stack([-.6*x, .6*y, np.ones(n)])
        normals /= np.linalg.norm(normals, axis=1, keepdims=True)
        sparse_region = x < 0
    else:
        raise ValueError("shape must be sphere or saddle")
    return points, normals, sparse_region


def scalar_field(points, normals):
    """f=sin(2x)+yz/4; project its ambient derivative onto the exact tangent plane."""
    x, y, z = points.T
    signal = np.sin(2*x) + y*z/4
    derivative = np.column_stack([2*np.cos(2*x), z/4, y/4])
    tangent = derivative - np.einsum("ni,ni->n", derivative, normals)[:, None] * normals
    return signal, tangent

