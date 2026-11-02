"""Compare the NumPy baseline with the actual pinned upstream gradient.

Run inside the official DeltaConv environment. This check has not been executed
by the CPU-only audit environment; its output is separate evidence.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from deltaconv_audit import UPSTREAM_COMMIT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "results/runs/upstream_parity.json")
    args = parser.parse_args()
    repo = args.upstream.resolve()
    commit = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    if commit != UPSTREAM_COMMIT:
        raise RuntimeError(f"expected upstream {UPSTREAM_COMMIT}, got {commit}")
    for extra in ([], ["--cached"]):
        subprocess.run(["git", "-C", str(repo), "diff", *extra, "--exit-code", "--quiet"], check=True)
    if args.output.exists():
        raise FileExistsError(args.output)
    sys.path.insert(0, str(repo))
    import numpy as np
    import torch
    import deltaconv
    from deltaconv.geometry.grad_div_mls import build_grad_div
    from deltaconv_audit.operators import build_gradient, neighbors
    from deltaconv_audit.surfaces import sample_surface, scalar_field

    # Upstream creates its ridge identity with torch.eye without an explicit dtype.
    # Keep that identity in float64 too, so this is a double-precision parity check.
    torch.set_default_dtype(torch.float64)
    if not Path(deltaconv.__file__).resolve().is_relative_to(repo):
        raise RuntimeError("DeltaConv was imported from a different installation")
    points, normals, _ = sample_surface("sphere", 120, 42, "biased")
    signal, _ = scalar_field(points, normals)
    idx = neighbors(points, 20)
    reference = build_gradient(points, normals, idx)
    tensor = lambda x: torch.as_tensor(x, dtype=torch.float64)
    edges = torch.tensor(np.stack([np.repeat(np.arange(len(points)), 20), idx.ravel()]), dtype=torch.long)
    errors = {}
    for normalized in (False, True):
        grad, _ = build_grad_div(tensor(points), tensor(normals), tensor(reference.basis[:, :, 0]),
                                tensor(reference.basis[:, :, 1]), edges,
                                regularizer=.001, normalized=normalized)
        local = (grad @ tensor(signal[:, None])).reshape(-1, 2).numpy()
        actual = np.einsum("nij,nj->ni", reference.basis, local)
        expected = reference.apply(signal, normalized=normalized)
        errors[str(normalized)] = float(np.max(np.abs(actual-expected)))
        np.testing.assert_allclose(actual, expected, atol=1e-8, rtol=1e-8)
    report = {"status": "ok", "upstream_commit": commit, "torch": torch.__version__,
              "numpy": np.__version__, "max_absolute_errors": errors,
              "upstream_gradient_sha256": hashlib.sha256((repo / "deltaconv/geometry/grad_div_mls.py").read_bytes()).hexdigest()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

