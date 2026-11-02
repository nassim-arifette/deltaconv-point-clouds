"""Run a fixed CPU audit and save measured values, provenance and visualization data."""
from __future__ import annotations

import argparse
import csv
import datetime
import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np

from . import UPSTREAM_COMMIT, UPSTREAM_URL
from .operators import build_gradient, estimate_normals, neighbors
from .surfaces import sample_surface, scalar_field

METHODS = {
    "A_upstream": dict(penalty="upstream", normalization="upstream", bandwidth="global"),
    "B_tensor_penalty": dict(penalty="tensor", normalization="upstream", bandwidth="global"),
    "C_invariant": dict(penalty="tensor", normalization="invariant", bandwidth="global"),
    "D_local_bandwidth": dict(penalty="tensor", normalization="invariant", bandwidth="local"),
}


def relative_l2(actual, expected):
    return float(np.linalg.norm(actual-expected) / max(np.linalg.norm(expected), 1e-12))


def run_audit(output, *, n=512, seeds=(0, 1, 2), noise=(0., .005, .02), k=20):
    output = Path(output)
    if (output / "results.json").exists():
        raise FileExistsError("results.json already exists; choose another output directory")
    output.mkdir(parents=True, exist_ok=True)
    if not seeds or not noise or len(set(seeds)) != len(seeds) or any(s < 0 for s in noise):
        raise ValueError("provide distinct seeds and nonnegative noise levels")
    started = time.perf_counter()
    rows, examples = [], []
    config = {
        "n_points": n, "k": k, "regularizer": .001, "seeds": list(seeds),
        "noise_sigma": list(noise), "noise_units": "coordinate units of the analytic surface",
        "shapes": ["sphere", "saddle"], "densities": ["uniform", "biased"],
        "normal_sources": ["exact", "pca"], "methods": METHODS,
        "signal": "sin(2*x) + y*z/4 evaluated on clean points",
        "gauge_test": "independent random tangent rotations, fixed geometry, signal and neighbors",
    }
    for shape in config["shapes"]:
        for density in config["densities"]:
            for seed in seeds:
                clean, exact_normals, sparse = sample_surface(shape, n, seed, density)
                signal, truth = scalar_field(clean, exact_normals)
                # Coupled corruption: same standard noise, scaled at every severity.
                rng = np.random.default_rng(seed + 7919)
                jitter = rng.normal(size=clean.shape)
                angles = rng.uniform(0, 2*np.pi, n)
                for sigma in noise:
                    points = clean + sigma*jitter
                    index = neighbors(points, k)
                    pca = estimate_normals(points, index)
                    for source, normals in [("exact", exact_normals), ("pca", pca)]:
                        case_id = f"{shape}_{density}_s{seed}_noise{sigma:g}_{source}"
                        keep = (seed == seeds[0] and source == "pca"
                                and ((density == "uniform" and sigma == noise[0])
                                     or (density == "biased" and sigma == noise[-1])))
                        example = {"case_id": case_id, "kind": "analytic_synthetic_surface",
                                   "points": points.tolist(), "clean_points": clean.tolist(),
                                   "truth_gradient": truth.tolist(), "methods": {}} if keep else None
                        for name, settings in METHODS.items():
                            row = {"case_id": case_id, "shape": shape, "density": density, "seed": seed,
                                   "sigma": sigma, "normal_source": source, "method": name,
                                   "sparse_points": int(sparse.sum()), "status": "ok"}
                            try:
                                g = build_gradient(points, normals, index, k=k, **settings)
                                rotated = build_gradient(points, normals, index, k=k, angles=angles, **settings)
                                predicted = g.apply(signal)
                                normalized = g.apply(signal, normalized=True)
                                row.update(
                                    gradient_relative_l2=relative_l2(predicted, truth),
                                    sparse_gradient_relative_l2=relative_l2(predicted[sparse], truth[sparse]),
                                    dense_gradient_relative_l2=relative_l2(predicted[~sparse], truth[~sparse]),
                                    gauge_raw_relative_l2=relative_l2(rotated.apply(signal), predicted),
                                    gauge_normalized_relative_l2=relative_l2(rotated.apply(signal, normalized=True), normalized),
                                    condition_p95=float(np.percentile(g.condition, 95)),
                                    normalization_scale=g.scale,
                                )
                                if example is not None:
                                    example["methods"][name] = {"gradient": predicted.tolist(),
                                        "point_error": np.linalg.norm(predicted-truth, axis=1).tolist()}
                            except (ValueError, np.linalg.LinAlgError, FloatingPointError) as exc:
                                row.update(status="failed", error=f"{type(exc).__name__}: {exc}")
                            rows.append(row)
                        if example is not None:
                            examples.append(example)
    summary = []
    for name in METHODS:
        subset = [r for r in rows if r["method"] == name and r["status"] == "ok"]
        summary.append({"method": name, "cases": len(subset),
                        "failed": sum(r["method"] == name and r["status"] != "ok" for r in rows),
                        **{metric+"_mean": float(np.mean([r[metric] for r in subset])) if subset else None
                           for metric in ["gradient_relative_l2", "sparse_gradient_relative_l2",
                                          "gauge_raw_relative_l2", "gauge_normalized_relative_l2"]},
                        "gauge_normalized_relative_l2_max": max((r["gauge_normalized_relative_l2"] for r in subset), default=None)})
    package = Path(__file__).parent
    report = {
        "schema_version": 1, "status": "ok" if all(r["status"] == "ok" for r in rows) else "partial_failure",
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "experiment": "MLS gradient coordinate-choice and sampling audit",
        "evidence_kind": "measured_cpu_numpy_operator_audit",
        "not_measured": ["DeltaConv classification accuracy", "learned network robustness", "GPU speed", "real scan performance"],
        "upstream": {"repo": UPSTREAM_URL, "commit": UPSTREAM_COMMIT, "execution": "formula-level NumPy audit, not upstream neural inference"},
        "environment": {"python": platform.python_version(), "numpy": np.__version__, "platform": platform.platform()},
        "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(package.glob("*.py"))},
        "config": config, "wall_seconds": time.perf_counter()-started, "summary": summary, "rows": rows,
    }
    (output / "results.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    (output / "viewer_data.json").write_text(json.dumps({"schema_version": 1, "examples": examples}, allow_nan=False)+"\n", encoding="utf-8")
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with (output / "metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results/runs/audit"))
    parser.add_argument("--points", type=int, default=512)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    args = parser.parse_args(argv)
    report = run_audit(args.output, n=args.points, seeds=tuple(args.seeds))
    print(json.dumps({"status": report["status"], "wall_seconds": report["wall_seconds"], "summary": report["summary"]}, indent=2))
    return 0 if report["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())

