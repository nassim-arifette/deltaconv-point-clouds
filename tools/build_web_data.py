"""Export the exact inputs of the four viewer examples for the in-browser MLS port.

The website recomputes the gradient live in TypeScript. To do that faithfully it
needs the very same points, PCA normals, neighbour indices, signal and random
tangent angles as the NumPy audit. They are regenerated here with the audit's
own functions, checked against results/cpu_reference/viewer_data.json, then
written to web/public/data/. Nothing is invented: every number shown by the site
is either recomputed from these inputs or copied from results.json.
"""
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from deltaconv_audit.operators import build_gradient, estimate_normals, neighbors  # noqa: E402
from deltaconv_audit.surfaces import sample_surface, scalar_field  # noqa: E402
from deltaconv_audit.experiment import METHODS  # noqa: E402
from tools.build_site_data import validate_paired_results  # noqa: E402

SOURCE = ROOT / "results/cpu_reference"
OUT = ROOT / "web/public/data"
FIXTURE = ROOT / "web/test/fixtures"


def regenerate(row, config):
    shape, density, seed, sigma = row["shape"], row["density"], row["seed"], row["sigma"]
    n, k = config["n_points"], config["k"]
    clean, exact_normals, sparse = sample_surface(shape, n, seed, density)
    signal, truth = scalar_field(clean, exact_normals)
    rng = np.random.default_rng(seed + 7919)
    jitter = rng.normal(size=clean.shape)
    angles = rng.uniform(0, 2*np.pi, n)
    points = clean + sigma*jitter
    index = neighbors(points, k)
    normals = estimate_normals(points, index) if row["normal_source"] == "pca" else exact_normals
    return dict(points=points, clean=clean, normals=normals, signal=signal, truth=truth,
                angles=angles, index=index, sparse=sparse)


def main():
    report = json.loads((SOURCE / "results.json").read_text(encoding="utf-8"))
    validate_paired_results(report)
    viewer = json.loads((SOURCE / "viewer_data.json").read_text(encoding="utf-8"))
    rows = {(r["case_id"], r["method"]): r for r in report["rows"]}
    config = report["config"]
    examples, fixtures = [], []
    for example in viewer["examples"]:
        case = example["case_id"]
        meta = rows[(case, "A_upstream")]
        data = regenerate(meta, config)
        if not np.array_equal(data["points"], np.asarray(example["points"])):
            raise ValueError(f"regenerated points differ from the audit for {case}")
        if not np.array_equal(data["truth"], np.asarray(example["truth_gradient"])):
            raise ValueError(f"regenerated truth differs from the audit for {case}")
        measured = {}
        for name, settings in METHODS.items():
            g = build_gradient(data["points"], data["normals"], data["index"], k=config["k"], **settings)
            if not np.allclose(g.apply(data["signal"]), example["methods"][name]["gradient"], rtol=0, atol=1e-12):
                raise ValueError(f"regenerated gradient differs from the audit for {case}/{name}")
            r = rows[(case, name)]
            measured[name] = {key: r[key] for key in (
                "gradient_relative_l2", "sparse_gradient_relative_l2", "dense_gradient_relative_l2",
                "gauge_raw_relative_l2", "gauge_normalized_relative_l2", "normalization_scale")}
        examples.append({
            "case_id": case, "shape": meta["shape"], "density": meta["density"], "seed": meta["seed"],
            "sigma": meta["sigma"], "normal_source": meta["normal_source"], "k": config["k"],
            "regularizer": config["regularizer"],
            "points": data["points"].tolist(), "normals": data["normals"].tolist(),
            "signal": data["signal"].tolist(), "truth": data["truth"].tolist(),
            "angles": data["angles"].tolist(), "index": data["index"].tolist(),
            "sparse": data["sparse"].astype(int).tolist(), "measured": measured,
        })
        fixtures.append({"case_id": case, "gradients": {m: example["methods"][m]["gradient"] for m in METHODS}})
    OUT.mkdir(parents=True, exist_ok=True)
    FIXTURE.mkdir(parents=True, exist_ok=True)
    (OUT / "clouds.json").write_text(json.dumps({"schema_version": 1, "examples": examples}), encoding="utf-8")
    keep = ("case_id", "shape", "density", "seed", "sigma", "normal_source", "method",
            "gradient_relative_l2", "sparse_gradient_relative_l2", "dense_gradient_relative_l2",
            "gauge_raw_relative_l2", "gauge_normalized_relative_l2", "condition_p95", "normalization_scale")
    audit = {"schema_version": 1, "created_utc": report["created_utc"], "wall_seconds": report["wall_seconds"],
             "environment": report["environment"], "upstream": report["upstream"], "config": config,
             "not_measured": report["not_measured"], "summary": report["summary"],
             "rows": [{k: r[k] for k in keep} for r in report["rows"]]}
    (OUT / "audit.json").write_text(json.dumps(audit), encoding="utf-8")
    (FIXTURE / "reference_gradients.json").write_text(json.dumps({"examples": fixtures}), encoding="utf-8")
    print(f"Exported {len(examples)} examples and {len(audit['rows'])} rows to {OUT}")


if __name__ == "__main__":
    main()
